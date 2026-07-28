"""Resize: every output position read from the source coordinate its scale maps it to.

One resize is a chain of one-dimensional ones — the interpolation is separable, so an N-d
result is the operand resized along one axis, then along the next — and that is how this is
emitted: one pass per named axis, through a pair of `double` working buffers, exactly as
ONNX's reference evaluator computes it. Within a pass every output position maps to a source
coordinate (that is what `coordinate_transformation_mode` names), takes a fixed number of
neighbouring elements around it, and weights them by the coefficients `mode` names —
`antialias` widening that footprint when the axis is being shrunk, `exclude_outside` dropping
the neighbours that fall off the end and renormalizing what is left.

The geometry is read at run time, not baked in. `scales`, `sizes` and `roi` are operands, and
a model that computes one — the shape the corpus's own Resize tests are written in — makes
the result's extent a function of input data. What the artifact is compiled for is the result
shape ONNX inferred; the kernel derives the extents the operands ask for and refuses, through
the status enum, to write a result of any other shape. So the buffers stay static and a value
the artifact was not compiled for is reported rather than silently computed.

`Upsample`, which ONNX deprecated in favour of `Resize`, is the same walk at the settings its
successor spells out: nearest neighbours taken at the floor of an asymmetrically mapped
coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from string import Template

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import checked_call, normalize_axis
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents

# Only Resize-19 and Upsample-10 are claimed. They are the revisions the reference evaluator
# is version-faithful for, and 19 is the one every Resize test in the backend corpus imports;
# Upsample-9, which the corpus's own test imports, is claimed alongside 10 because ONNX
# revised nothing between them that this kernel reads. A model importing an older revision
# gets the unsupported-version error.
_RESIZE_VERSIONS = (19,)
_UPSAMPLE_VERSIONS = (9, 10)

# The C values the kernel switches on, in the order its `switch` cases are written.
_MODES = {"nearest": 0, "linear": 1, "cubic": 2}
_NEAREST_MODES = {
    "round_prefer_floor": 0,
    "round_prefer_ceil": 1,
    "floor": 2,
    "ceil": 3,
}
_TRANSFORMS = {
    "half_pixel": 0,
    "half_pixel_symmetric": 1,
    "pytorch_half_pixel": 2,
    "align_corners": 3,
    "asymmetric": 4,
    "tf_crop_and_resize": 5,
}
_POLICIES = {"stretch": 0, "not_larger": 1, "not_smaller": 2}

# The two modes Upsample carries over into its successor; the rest arrived after it.
_UPSAMPLE_MODES = ("nearest", "linear")

# How far the largest extent the geometry may derive is allowed to reach before the kernel
# refuses it: comfortably past any buffer an artifact can hold, and well inside the range a
# `double` converts to an integer type exactly.
_EXTENT_LIMIT = 9.0e15

_TAPS_TEMPLATE = Template("""\
static int $name(int mode, int antialias, double scale)
{
    double reach;
    if (mode == $nearest) {
        return 2;
    }
    if (!antialias) {
        return (mode == $linear) ? 2 : 4;
    }
    /* Antialiasing widens the filter over the elements a shrinking axis merges together:
       its footprint reaches `reach / scale` to each side of the coordinate instead of
       `reach`. Growing an axis leaves it at its own width. */
    reach = (mode == $linear) ? 1.0 : 2.0;
    return 2 - 2 * ((int)floor(-reach / ((scale < 1.0) ? scale : 1.0)) + 1);
}""")

_COEFFICIENT_TEMPLATE = Template("""\
static double $name(
    int mode,
    int nearest_mode,
    int antialias,
    double ratio,
    double scale,
    double cubic_a,
    int tap,
    int taps)
{
    double x;
    double squared;
    if (mode == $nearest) {
        /* A coordinate that lands on an element takes that element, whichever way the
           rounding rule would break a tie: the ratio is 1 there, and no rule applies. */
        if (ratio == 1.0) {
            return (tap == 0) ? 0.0 : 1.0;
        }
        switch (nearest_mode) {
        case $round_prefer_ceil:
            return (tap == 0) ? (double)(ratio < 0.5) : (double)(ratio >= 0.5);
        case $floor:
            return (tap == 0) ? 1.0 : 0.0;
        case $ceil:
            return (tap == 0) ? 0.0 : 1.0;
        default:
            return (tap == 0) ? (double)(ratio <= 0.5) : (double)(ratio > 0.5);
        }
    }
    if (!antialias) {
        if (mode == $linear) {
            return (tap == 0) ? 1.0 - ratio : ratio;
        }
        switch (tap) {
        case 0:
            x = ratio + 1.0;
            return ((cubic_a * x - 5.0 * cubic_a) * x + 8.0 * cubic_a) * x
                - 4.0 * cubic_a;
        case 1:
            return ((cubic_a + 2.0) * ratio - (cubic_a + 3.0)) * ratio * ratio + 1.0;
        case 2:
            x = 1.0 - ratio;
            return ((cubic_a + 2.0) * x - (cubic_a + 3.0)) * x * x + 1.0;
        default:
            x = 2.0 - ratio;
            return ((cubic_a * x - 5.0 * cubic_a) * x + 8.0 * cubic_a) * x
                - 4.0 * cubic_a;
        }
    }
    /* The antialiased filter is sampled at the operand's own spacing scaled down to the
       result's, which is what spreads it over the elements being merged. */
    x = ((scale < 1.0) ? scale : 1.0) * ((double)(1 - taps / 2 + tap) - ratio);
    if (mode == $linear) {
        x = 1.0 - fabs(x);
        return (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
    }
    x = fabs(x);
    squared = x * x;
    if (x <= 1.0) {
        return (cubic_a + 2.0) * (x * squared) - (cubic_a + 3.0) * squared + 1.0;
    }
    if (x < 2.0) {
        return cubic_a * (x * squared) - 5.0 * cubic_a * squared + 8.0 * cubic_a * x
            - 4.0 * cubic_a;
    }
    return 0.0;
}""")

_COORDINATE_TEMPLATE = Template("""\
static double $name(
    int transform,
    double position,
    double scale,
    double width,
    double positions,
    double extent,
    double span,
    double shift)
{
    switch (transform) {
    case $half_pixel_symmetric:
        return (width / 2.0) * (1.0 - extent / positions)
            + (position + 0.5) / scale - 0.5;
    case $pytorch_half_pixel:
        return (positions == 1.0) ? -0.5 : (position + 0.5) / scale - 0.5;
    case $align_corners:
        return (positions == 1.0)
            ? 0.0
            : position * (width - 1.0) / (positions - 1.0);
    case $asymmetric:
        return position / scale;
    case $tf_crop_and_resize:
        return ((positions == 1.0)
            ? span * (width - 1.0) / 2.0
            : position * span * (width - 1.0) / (positions - 1.0)) + shift;
    default:
        return (position + 0.5) / scale - 0.5;
    }
}""")

# The extent one axis is asked for, and the scale that maps its coordinates. `scales` states
# the scale and the extent follows from it; `sizes` states the extent and the scale follows —
# unless a `keep_aspect_ratio_policy` overrides both with one scale shared across the axes.
_GEOMETRY_TEMPLATE = Template("""\
static int $name(
    const float* scales,
    const int64_t* sizes,
    size_t width,
    int index,
    int policy,
    double policy_scale,
    double* scale,
    size_t* extent)
{
    double positions;
    if (scales != NULL) {
        *scale = (double)scales[index];
        positions = trunc(*scale * (double)width);
    } else if (policy == $stretch) {
        *scale = (double)sizes[index] / (double)width;
        positions = (double)sizes[index];
    } else {
        *scale = policy_scale;
        positions = trunc(policy_scale * (double)width + 0.5);
    }
    /* Rejects a negative, infinite or absent-minded extent before the conversion below,
       which C leaves undefined for anything outside `size_t`. */
    if (!(positions >= 0.0 && positions <= $limit)) {
        return 1;
    }
    *extent = (size_t)positions;
    return 0;
}""")

_CLOSE_TEMPLATE = Template("""\
static int $name(double left, double right)
{
    /* Python's `math.isclose` at its default tolerances, which is what the identity test
       below is written against. */
    const double difference = fabs(left - right);
    const double largest = (fabs(left) > fabs(right)) ? fabs(left) : fabs(right);
    return difference <= 1e-9 * largest;
}""")

_INTEGER_STORE_TEMPLATE = Template("""\
static $element $name(double value)
{
    /* An interpolated value is rounded to the nearest even and then held inside the
       element type's range. A NaN, which no comparison below admits, lands on the minimum
       rather than in the undefined behaviour converting it would be. */
    const double rounded = rint(value);
    if (!(rounded > $lower)) {
        return $minimum;
    }
    if (rounded >= $upper) {
        return $maximum;
    }
    return ($element)rounded;
}""")

_FLOAT_STORE_TEMPLATE = Template("""\
static float $name(double value)
{
    /* Saturating rather than overflowing: converting a value outside `float`'s range is
       undefined in C, and the reference clips such a value instead of returning an
       infinity. */
    if (value < $lower) {
        return $minimum;
    }
    if (value > $upper) {
        return $maximum;
    }
    return (float)value;
}""")

_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $roi_element* roi,
    const float* scales,
    const int64_t* sizes,
    double* work,
    double* spare,
    size_t input_count,
    size_t output_count,
    int rank,
    const size_t* input_shape,
    const size_t* output_shape,
    int axis_count,
    const size_t* axes,
    int mode,
    int nearest_mode,
    int transform,
    int antialias,
    int exclude_outside,
    int policy,
    double cubic_a,
    double extrapolation_value)
{
    double* source = work;
    double* target = spare;
    double policy_scale = 0.0;
    size_t index;
    int step;

    /* One scale for every named axis, taken from the one that shrinks or grows the
       operand most, is what a policy other than `stretch` asks for. */
    if (sizes != NULL && policy != $stretch) {
        for (step = 0; step < axis_count; ++step) {
            const double ratio =
                (double)sizes[step] / (double)input_shape[axes[step]];
            const int closer = (policy == $not_larger)
                ? (ratio < policy_scale)
                : (ratio > policy_scale);
            if (step == 0 || closer) {
                policy_scale = ratio;
            }
        }
    }
    for (index = 0; index < input_count; ++index) {
        source[index] = (double)in[index];
    }

    for (step = 0; step < axis_count; ++step) {
        const size_t axis = axes[step];
        const size_t width = input_shape[axis];
        const size_t extent = output_shape[axis];
        const double edge = (double)width - 1.0;
        double scale = 1.0;
        double span = 1.0;
        double shift = 0.0;
        double positions;
        size_t derived = 0;
        size_t outer = 1;
        size_t inner = 1;
        size_t position;
        int other;
        int taps;

        if ($geometry(
                scales, sizes, width, step, policy, policy_scale, &scale, &derived) != 0
            || derived != extent) {
            return 1;
        }
        if (roi != NULL) {
            /* In the region's own element type, as the reference reads it: a nearest
               neighbour lands on the other side of a tie otherwise. */
            span = (double)(roi[axis_count + step] - roi[step]);
            shift = (double)(roi[step] * ($roi_element)edge);
        }
        /* An axis at its own scale over its own region is the identity, and the reference
           passes it through rather than mapping coordinates that would agree with it
           everywhere but at the ends. */
        if ($close(scale, 1.0) && extent == width
            && (roi == NULL
                || (roi[step] == 0.0
                    && $close((double)roi[axis_count + step], 1.0)))) {
            continue;
        }

        /* What lies around this axis at this point of the walk: the axes already resized
           carry their result extent, the rest still carry the operand's. */
        for (other = 0; other < rank; ++other) {
            size_t current = input_shape[other];
            int earlier;
            for (earlier = 0; earlier < step; ++earlier) {
                if (axes[earlier] == (size_t)other) {
                    current = output_shape[other];
                }
            }
            if ((size_t)other < axis) {
                outer *= current;
            } else if ((size_t)other > axis) {
                inner *= current;
            }
        }

        taps = $taps(mode, antialias, scale);
        positions = scale * (double)width;
        for (position = 0; position < extent; ++position) {
            const double placed = $coordinate(
                transform, (double)position, scale, (double)width, positions,
                (double)extent, span, shift);
            size_t plane;
            size_t offset;
            int tap;

            /* A region mapping outside the operand carries the extrapolation value; the
               test admits no coordinate a source position could not be taken from. */
            if (transform == $tf_crop_and_resize && !(placed >= 0.0 && placed <= edge)) {
                for (plane = 0; plane < outer; ++plane) {
                    double* row = target + (plane * extent + position) * inner;
                    for (offset = 0; offset < inner; ++offset) {
                        row[offset] = extrapolation_value;
                    }
                }
                continue;
            }
            {
                const double floored = floor(placed);
                const double fraction = placed - floored;
                /* The element the coordinate lands on is the one to its left, so a
                   coordinate that lands exactly on one carries a whole ratio, not none. */
                const double ratio = (fraction == 0.0) ? 1.0 : fraction;
                const ptrdiff_t start =
                    (ptrdiff_t)floored - taps / 2 + ((fraction == 0.0) ? 0 : 1);
                double denominator = 1.0;

                if (exclude_outside || antialias) {
                    double total = 0.0;
                    for (tap = 0; tap < taps; ++tap) {
                        const ptrdiff_t sampled = start + tap;
                        if (exclude_outside
                                && (sampled < 0 || sampled >= (ptrdiff_t)width)) {
                            continue;
                        }
                        total += $coefficient(
                            mode, nearest_mode, antialias, ratio, scale, cubic_a,
                            tap, taps);
                    }
                    denominator = (total == 0.0) ? 1.0 : total;
                }
                for (plane = 0; plane < outer; ++plane) {
                    double* row = target + (plane * extent + position) * inner;
                    for (offset = 0; offset < inner; ++offset) {
                        row[offset] = 0.0;
                    }
                }
                for (tap = 0; tap < taps; ++tap) {
                    const ptrdiff_t sampled = start + tap;
                    const int inside = (sampled >= 0) && (sampled < (ptrdiff_t)width);
                    /* Off the end, the operand's own edge element is read, which is what
                       padding it by repetition amounts to. */
                    const size_t clamped =
                        inside ? (size_t)sampled : ((sampled < 0) ? 0 : width - 1);
                    const double raw = (exclude_outside && !inside)
                        ? 0.0
                        : $coefficient(
                            mode, nearest_mode, antialias, ratio, scale, cubic_a,
                            tap, taps);
                    const double weight = raw / denominator;
                    for (plane = 0; plane < outer; ++plane) {
                        double* row = target + (plane * extent + position) * inner;
                        const double* taken =
                            source + (plane * width + clamped) * inner;
                        for (offset = 0; offset < inner; ++offset) {
                            row[offset] += weight * taken[offset];
                        }
                    }
                }
            }
        }
        {
            double* swapped = source;
            source = target;
            target = swapped;
        }
    }
    for (index = 0; index < output_count; ++index) {
        out[index] = $store;
    }
    return 0;
}""")


@dataclass(frozen=True)
class _Helpers:
    """The shared functions the kernel calls, named so its call sites can reach them."""

    taps: CFunction
    coefficient: CFunction
    coordinate: CFunction
    geometry: CFunction
    close: CFunction
    store: CFunction | None

    @property
    def functions(self) -> tuple[CFunction, ...]:
        listed = (
            self.taps,
            self.coefficient,
            self.coordinate,
            self.geometry,
            self.close,
        )
        return listed if self.store is None else (*listed, self.store)

    @property
    def store_expression(self) -> str:
        """How one interpolated value reaches the result buffer's element type."""
        return (
            "source[index]"
            if self.store is None
            else f"{self.store.name}(source[index])"
        )


@dataclass(frozen=True)
class _Options:
    """What the node asks for, as the values the kernel switches on."""

    mode: int
    nearest_mode: int
    transform: int
    antialias: int
    exclude_outside: int
    policy: int
    cubic_a: float
    extrapolation_value: float

    @property
    def arguments(self) -> list[str]:
        return [
            str(self.mode),
            str(self.nearest_mode),
            str(self.transform),
            str(self.antialias),
            str(self.exclude_outside),
            str(self.policy),
            scalar_literal(self.cubic_a, TensorProto.DOUBLE),
            scalar_literal(self.extrapolation_value, TensorProto.DOUBLE),
        ]


def _resize(context: NodeContext) -> NodeEmission:
    options = _Options(
        mode=_choice(context, "mode", "nearest", _MODES),
        nearest_mode=_choice(
            context, "nearest_mode", "round_prefer_floor", _NEAREST_MODES
        ),
        transform=_choice(
            context, "coordinate_transformation_mode", "half_pixel", _TRANSFORMS
        ),
        antialias=int(context.int_attribute("antialias") != 0),
        exclude_outside=int(context.int_attribute("exclude_outside") != 0),
        policy=_choice(context, "keep_aspect_ratio_policy", "stretch", _POLICIES),
        cubic_a=context.float_attribute("cubic_coeff_a"),
        extrapolation_value=context.float_attribute("extrapolation_value"),
    )
    return _emit(
        context,
        options,
        roi=_operand(context, 1),
        scales=_operand(context, 2),
        sizes=_operand(context, 3),
    )


def _upsample(context: NodeContext) -> NodeEmission:
    """Upsample, which ONNX deprecated in favour of the Resize settings it spells out.

    Its successor's own specification records the equivalence: `asymmetric` is described
    there as the coordinate mapping Resize-10 — the revision Upsample became — applies, and
    `floor` as the neighbour it takes.
    """
    options = _Options(
        mode=_choice(
            context, "mode", "nearest", {name: _MODES[name] for name in _UPSAMPLE_MODES}
        ),
        nearest_mode=_NEAREST_MODES["floor"],
        transform=_TRANSFORMS["asymmetric"],
        antialias=0,
        exclude_outside=0,
        policy=_POLICIES["stretch"],
        cubic_a=0.0,
        extrapolation_value=0.0,
    )
    return _emit(context, options, roi=None, scales=_operand(context, 1), sizes=None)


def _emit(
    context: NodeContext,
    options: _Options,
    *,
    roi: TensorRef | None,
    scales: TensorRef | None,
    sizes: TensorRef | None,
) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    if len(result.shape) != len(source.shape):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` resizes `{source.name}` of "
            f"shape {list(source.shape)} into `{result.name}` of shape "
            f"{list(result.shape)}; a resize leaves the rank alone."
        )
    if result.elem_type == TensorProto.BOOL:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` of a "
            f"`{element_type_name(result.elem_type)}` tensor is not supported by the C "
            "compiler; interpolating between two truth values has no defined result."
        )
    if options.antialias and options.mode == _MODES["nearest"]:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` asks for `antialias` in "
            "`nearest` mode, which ONNX defines only for the interpolating modes."
        )

    axes = _axes(context, len(source.shape))
    _verify_operands(context, source, result, axes, roi, scales, sizes)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    region = c_type(roi.elem_type if roi is not None else TensorProto.FLOAT)
    name = f"{context.prefix}_resize_{element}_{region}"
    helpers = _helpers(context, result.elem_type)
    definition = _TEMPLATE.substitute(
        name=name,
        element=element,
        roi_element=region,
        store=helpers.store_expression,
        geometry=helpers.geometry.name,
        close=helpers.close.name,
        taps=helpers.taps.name,
        coefficient=helpers.coefficient.name,
        coordinate=helpers.coordinate.name,
        stretch=_POLICIES["stretch"],
        not_larger=_POLICIES["not_larger"],
        tf_crop_and_resize=_TRANSFORMS["tf_crop_and_resize"],
    )
    work, spare = _scratch(context, source, result)
    call = checked_call(
        context,
        name,
        [
            result.expr,
            source.expr,
            "NULL" if roi is None else roi.expr,
            "NULL" if scales is None else scales.expr,
            "NULL" if sizes is None else sizes.expr,
            work.symbol,
            spare.symbol,
            f"{source.elem_count}u",
            f"{result.elem_count}u",
            str(len(source.shape)),
            extents(source.shape),
            extents(result.shape),
            str(len(axes)),
            extents(axes),
            *options.arguments,
        ],
    )
    return NodeEmission(
        functions=(*helpers.functions, CFunction(name, definition)),
        statements=(call,),
        scratch=(work, spare),
    )


def _helpers(context: NodeContext, elem_type: int) -> _Helpers:
    prefix = f"{context.prefix}_resize"
    return _Helpers(
        taps=CFunction(
            f"{prefix}_taps",
            _TAPS_TEMPLATE.substitute(
                name=f"{prefix}_taps",
                nearest=_MODES["nearest"],
                linear=_MODES["linear"],
            ),
        ),
        coefficient=CFunction(
            f"{prefix}_coefficient",
            _COEFFICIENT_TEMPLATE.substitute(
                name=f"{prefix}_coefficient",
                nearest=_MODES["nearest"],
                linear=_MODES["linear"],
                round_prefer_ceil=_NEAREST_MODES["round_prefer_ceil"],
                floor=_NEAREST_MODES["floor"],
                ceil=_NEAREST_MODES["ceil"],
            ),
        ),
        coordinate=CFunction(
            f"{prefix}_coordinate",
            _COORDINATE_TEMPLATE.substitute(name=f"{prefix}_coordinate", **_TRANSFORMS),
        ),
        geometry=CFunction(
            f"{prefix}_geometry",
            _GEOMETRY_TEMPLATE.substitute(
                name=f"{prefix}_geometry",
                stretch=_POLICIES["stretch"],
                limit=scalar_literal(_EXTENT_LIMIT, TensorProto.DOUBLE),
            ),
        ),
        close=CFunction(
            f"{prefix}_close", _CLOSE_TEMPLATE.substitute(name=f"{prefix}_close")
        ),
        store=_store_function(prefix, elem_type),
    )


def _store_function(prefix: str, elem_type: int) -> CFunction | None:
    """How an interpolated value reaches the result's element type, where it has to narrow."""
    if elem_type == TensorProto.DOUBLE:
        return None
    name = f"{prefix}_store_{c_type(elem_type)}"
    if elem_type == TensorProto.FLOAT:
        limit = float(np.finfo("float32").max)
        return CFunction(
            name,
            _FLOAT_STORE_TEMPLATE.substitute(
                name=name,
                lower=scalar_literal(-limit, TensorProto.DOUBLE),
                upper=scalar_literal(limit, TensorProto.DOUBLE),
                minimum=scalar_literal(-limit, TensorProto.FLOAT),
                maximum=scalar_literal(limit, TensorProto.FLOAT),
            ),
        )
    info = np.iinfo(numpy_dtype_name(elem_type))
    return CFunction(
        name,
        _INTEGER_STORE_TEMPLATE.substitute(
            name=name,
            element=c_type(elem_type),
            lower=scalar_literal(float(info.min), TensorProto.DOUBLE),
            upper=scalar_literal(float(info.max), TensorProto.DOUBLE),
            minimum=scalar_literal(info.min, elem_type),
            maximum=scalar_literal(info.max, elem_type),
        ),
    )


def _scratch(
    context: NodeContext, source: TensorRef, result: TensorRef
) -> tuple[ScratchBuffer, ...]:
    """The two working buffers one pass reads and the next writes.

    A pass over one axis reads the whole result of the pass before it, so neither can be
    computed in place; the artifact allocates nothing, so the space is reserved at compile
    time and counted in the reported footprint like every other buffer. Every intermediate
    fits an operand whose every axis carries the larger of its two extents.
    """
    count = math.prod(
        max(before, after) for before, after in zip(source.shape, result.shape)
    )
    return tuple(
        ScratchBuffer(f"{context.prefix}_resize_{role}", TensorProto.DOUBLE, count)
        for role in ("work", "spare")
    )


def _operand(context: NodeContext, index: int) -> TensorRef | None:
    """Operand `index`, where one holding nothing at all reads as one left out.

    ONNX's own evaluator reads an empty `roi` that way, and exporters routinely pass an
    empty `scales` alongside a `sizes` rather than leaving the position blank.
    """
    operand = context.optional_input(index)
    return operand if operand is not None and operand.elem_count > 0 else None


def _axes(context: NodeContext, rank: int) -> tuple[int, ...]:
    """The axes the node resizes, in the order its operands describe them."""
    declared = context.attribute("axes", None)
    if declared is None:
        return tuple(range(rank))
    axes = tuple(normalize_axis(context, int(axis), rank) for axis in declared)
    if len(set(axes)) != len(axes):
        raise CompileError(
            f"Node `{context.label}`: `axes` {[int(axis) for axis in declared]} names the "
            "same dimension more than once."
        )
    return axes


def _verify_operands(
    context: NodeContext,
    source: TensorRef,
    result: TensorRef,
    axes: tuple[int, ...],
    roi: TensorRef | None,
    scales: TensorRef | None,
    sizes: TensorRef | None,
) -> None:
    """Everything about this node the operands' shapes settle before it runs."""
    label = f"Node `{context.label}`: `{context.node.op_type}`"
    if (scales is None) == (sizes is None):
        raise CompileError(
            f"{label} needs exactly one of `scales` and `sizes` to say what to resize to; "
            f"this node passes {'both' if scales is not None else 'neither'}."
        )
    for operand, role in ((scales, "scales"), (sizes, "sizes"), (roi, "roi")):
        if operand is None:
            continue
        expected = 2 * len(axes) if role == "roi" else len(axes)
        if operand.shape != (expected,):
            raise CompileError(
                f"{label} takes `{role}` as {expected} value(s) for the {len(axes)} "
                f"axis/axes it resizes, but `{operand.name}` has shape "
                f"{list(operand.shape)}."
            )
    named = set(axes)
    for axis, (before, after) in enumerate(zip(source.shape, result.shape)):
        if axis not in named and before != after:
            raise CompileError(
                f"{label} does not resize axis {axis}, but `{source.name}` has "
                f"{before} element(s) there against `{result.name}`'s {after}."
            )
        if axis in named and before == 0 and after != 0:
            raise CompileError(
                f"{label} resizes axis {axis} of `{source.name}`, which holds no "
                f"elements, into {after} element(s); there is nothing to interpolate."
            )


def _choice(
    context: NodeContext, attribute: str, default: str, choices: dict[str, int]
) -> int:
    value = context.attribute(attribute, default)
    name = value.decode() if isinstance(value, bytes) else str(value)
    if name not in choices:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` asks for `{attribute}` "
            f"`{name}`, which is not one of the values ONNX defines for it "
            f"({', '.join(f'`{choice}`' for choice in choices)})."
        )
    return choices[name]


register_kernel("", "Resize", _RESIZE_VERSIONS, _resize)
register_kernel("", "Upsample", _UPSAMPLE_VERSIONS, _upsample)
