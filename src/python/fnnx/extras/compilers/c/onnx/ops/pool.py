"""The poolings: the same sliding window as a convolution, folding rather than weighting.

Every pooling walks the geometry `window.py` resolves — for each output position, each tap
reads the operand at `position * stride + tap * dilation - pad` along each spatial axis — and
folds what it finds into one value: the largest, the mean, or the Lp norm. So there is one
template, one kernel per fold and element type, and the geometry reaches it as call-site
literals. The `Global*` family is that same walk at a window the size of the operand's spatial
extent, which is why it shares those kernels outright.

Two things only a pooling has. `ceil_mode` rounds the number of window positions up instead of
down, which lets the last window hang off the end of the operand. And a tap is *counted*
separately from being *read*: it counts when it lands inside the operand widened by the node's
own pads — which is what `count_include_pad` averages over — while only a tap inside the
operand itself is read. The padding `ceil_mode` implies is neither read nor counted.

`MaxUnpool` runs a max pooling backwards: it scatters each element to the position that
pooling's `Indices` output recorded, leaving the rest at zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import combiner, expand, extents
from fnnx.extras.compilers.c.onnx.ops.reduce import extremum_identity, extremum_test
from fnnx.extras.compilers.c.onnx.ops.window import (
    auto_pad_mode,
    declared_pads,
    offsets,
    resolve_pads,
    spatial_attribute,
    spatial_extents,
)

# The parameters every pooling kernel takes after its buffers. `counted_shape` is the operand
# widened by the node's own pads: a tap at or past it belongs to the padding `ceil_mode`
# added, which no pooling reads or counts.
_GEOMETRY_PARAMETERS = """\
    size_t plane_count,
    size_t input_size,
    size_t output_size,
    size_t window_size,
    int spatial_rank,
    const size_t* input_shape,
    const size_t* counted_shape,
    const size_t* output_shape,
    const size_t* window_shape,
    const size_t* strides,
    const size_t* dilations,
    const ptrdiff_t* pads"""

# Where one tap of one window lands, as the offset into the operand plus the two flags a fold
# reads it through.
_WALK = Template("""\
                size_t remaining_position = position;
                size_t remaining_tap = tap;
                size_t offset = 0;
                size_t stride = 1;
                int inside = 1;
                int counted = 1;
                int axis;
                for (axis = spatial_rank - 1; axis >= 0; --axis) {
                    const ptrdiff_t coordinate =
                        (ptrdiff_t)(remaining_position % output_shape[axis])
                            * (ptrdiff_t)strides[axis]
                        + (ptrdiff_t)(remaining_tap % window_shape[axis])
                            * (ptrdiff_t)dilations[axis]
                        - pads[axis];
                    remaining_position /= output_shape[axis];
                    remaining_tap /= window_shape[axis];
                    if (coordinate >= (ptrdiff_t)counted_shape[axis]) {
                        counted = 0;
                    }
                    if (coordinate < 0 || coordinate >= (ptrdiff_t)input_shape[axis]) {
                        inside = 0;
                    } else {
                        offset += (size_t)coordinate * stride;
                    }
                    stride *= input_shape[axis];
                }
$combine""")

_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters$extra)
{
    size_t plane, position, tap;
    for (plane = 0; plane < plane_count; ++plane) {
        const $element* values = in + plane * input_size;
        $element* result = out + plane * output_size;
        for (position = 0; position < output_size; ++position) {
            $state
            for (tap = 0; tap < window_size; ++tap) {
$walk
            }
            result[position] = $finish;
        }
    }
}""")

# The indexed max pooling reports where each maximum was read, as a flat index into the whole
# operand. `index_strides` is what `storage_order` chooses: the operand's own row-major
# strides, or the column-major ones ONNX defines the other order as.
_INDEXED_TEMPLATE = Template("""\
static void $name(
    $element* out,
    int64_t* indices,
    const $element* in,
$parameters,
    const size_t* index_strides)
{
    size_t plane, position, tap;
    (void)counted_shape;
    for (plane = 0; plane < plane_count; ++plane) {
        const $element* values = in + plane * input_size;
        $element* result = out + plane * output_size;
        int64_t* chosen = indices + plane * output_size;
        for (position = 0; position < output_size; ++position) {
            $element best = $identity;
            size_t found = 0;
            int seen = 0;
            for (tap = 0; tap < window_size; ++tap) {
                size_t remaining_position = position;
                size_t remaining_tap = tap;
                size_t offset = 0;
                size_t reported = 0;
                size_t stride = 1;
                int inside = 1;
                int axis;
                for (axis = spatial_rank - 1; axis >= 0; --axis) {
                    const ptrdiff_t coordinate =
                        (ptrdiff_t)(remaining_position % output_shape[axis])
                            * (ptrdiff_t)strides[axis]
                        + (ptrdiff_t)(remaining_tap % window_shape[axis])
                            * (ptrdiff_t)dilations[axis]
                        - pads[axis];
                    remaining_position /= output_shape[axis];
                    remaining_tap /= window_shape[axis];
                    if (coordinate < 0 || coordinate >= (ptrdiff_t)input_shape[axis]) {
                        inside = 0;
                    } else {
                        offset += (size_t)coordinate * stride;
                        reported += (size_t)coordinate * index_strides[axis];
                    }
                    stride *= input_shape[axis];
                }
                if (inside) {
                    const $element x = values[offset];
                    if (!seen || ($better)) {
                        best = x;
                        found = reported;
                        seen = 1;
                    }
                }
            }
            result[position] = best;
            chosen[position] = (int64_t)(plane * input_size + found);
        }
    }
}""")

_UNPOOL_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const int64_t* indices,
    size_t count,
    size_t limit)
{
    size_t index;
    for (index = 0; index < limit; ++index) {
        out[index] = $zero;
    }
    for (index = 0; index < count; ++index) {
        const int64_t chosen = indices[index];
        /* ONNX leaves an index outside the result undefined; the artifact reports it
           rather than writing past the buffer. */
        if (chosen < 0 || (size_t)chosen >= limit) {
            return 1;
        }
        out[(size_t)chosen] = in[index];
    }
    return 0;
}""")

# Every pooling arrived at opset 1 and was revised repeatedly since — AveragePool gained
# `ceil_mode` at 10 and `dilations` at 19, MaxPool `Indices` at 8 and `dilations` at 10 — and
# 22 widened them all to bfloat16. Only 22 is claimed: it is the revision the reference
# evaluator is version-faithful for and the one every pooling test in the backend corpus
# imports, so it is the only one anything can vouch for. A model importing an older one gets
# the unsupported-version error.
_VERSIONS = (22,)


@dataclass(frozen=True)
class _Fold:
    """How one pooling folds a window, as C over a tap's `inside` and `counted` flags.

    `state` opens the fold, `combine` runs per tap and `finish` is what the position is
    written from; `parameters` are the kernel parameters only this fold takes and
    `arguments` the literals its call sites pass for them. All three expressions are
    `$`-templated over the element type.
    """

    name: str
    state: str
    combine: str
    finish: str
    parameters: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    helpers: tuple[CFunction, ...] = ()


# What a pooling's fold is built from: the node, and the element type it folds at.
_Recipe = Callable[[NodeContext, int], _Fold]


@dataclass(frozen=True)
class _Geometry:
    """A pooling's shape, resolved to the literals the kernel walks it with."""

    batch_count: int
    channels: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    window_shape: tuple[int, ...]
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    pads: tuple[int, ...]
    counted_shape: tuple[int, ...]

    @property
    def result_shape(self) -> tuple[int, ...]:
        return (self.batch_count, self.channels, *self.output_shape)

    @property
    def arguments(self) -> list[str]:
        """Call-site literals for the geometry parameters a pooling kernel takes."""
        return [
            f"{self.batch_count * self.channels}u",
            f"{math.prod(self.input_shape)}u",
            f"{math.prod(self.output_shape)}u",
            f"{math.prod(self.window_shape)}u",
            str(len(self.output_shape)),
            extents(self.input_shape),
            extents(self.counted_shape),
            extents(self.output_shape),
            extents(self.window_shape),
            extents(self.strides),
            extents(self.dilations),
            offsets(self.pads),
        ]

    def index_strides(self, storage_order: int) -> tuple[int, ...]:
        """Strides turning a spatial coordinate into the index `storage_order` asks for."""
        if storage_order == 0:
            return row_major_strides(self.input_shape)
        return row_major_strides(self.input_shape[::-1])[::-1]


# --------------------------------------------------------------------------------------
# The folds
# --------------------------------------------------------------------------------------


def _average_fold(context: NodeContext, elem_type: int) -> _Fold:
    """The mean over the taps a window covers, over as many of them as it counts.

    `count_include_pad` decides whether the padded positions are part of that count; the
    value they contribute is zero either way, so only the divisor changes. A window counting
    nothing at all divides by zero, which IEEE defines for the float families this op is
    defined over.

    The default is stated here rather than read off the schema because `GlobalAveragePool`
    folds through this too and has no such attribute — it pads nothing, so every tap it
    covers is counted whichever way the flag would go.
    """
    return _Fold(
        name="average",
        state="$element total = $zero; size_t inside_count = 0, counted_count = 0;",
        combine="""\
                if (counted) {
                    ++counted_count;
                    if (inside) {
                        total += values[offset];
                        ++inside_count;
                    }
                }""",
        finish="total / ($element)(include_pad ? counted_count : inside_count)",
        parameters=("size_t include_pad",),
        arguments=(f"{int(context.attribute('count_include_pad', 0) != 0)}u",),
    )


def _max_fold(context: NodeContext, elem_type: int) -> _Fold:
    """The largest tap inside the operand; a padded position is not a candidate for it.

    ONNX says nothing about a NaN in the window, and the reference evaluator's two pooling
    paths do not agree on one, so the fold takes numpy's `maximum`, as ReduceMax does.
    """
    largest = combiner(context, elem_type, largest=True)
    return _Fold(
        name="max",
        state=f"$element best = {extremum_identity(elem_type, largest=True)};",
        combine=f"""\
                (void)counted;
                if (inside) {{
                    best = {largest.name}(best, values[offset]);
                }}""",
        finish="best",
        helpers=(largest,),
    )


def _lp_fold(context: NodeContext, elem_type: int) -> _Fold:
    """The Lp norm of the taps a window covers: a padded position contributes `|0|^p`.

    The norm itself, over however many taps the window turns out to hold. ONNX's reference
    evaluator computes a window that `ceil_mode` clipped differently — it averages and scales
    the result back up by the whole kernel's tap count, which its own source records as a
    borrowed computation that differs from the spec's — where the backend corpus's stored
    outputs and onnxruntime both take the plain norm, as this does.
    """
    return _Fold(
        name="lp",
        state="$element total = $zero;",
        combine="""\
                (void)counted;
                if (inside) {
                    total += pow$f(fabs$f(values[offset]), ($element)order);
                }""",
        finish="pow$f(total, $one / ($element)order)",
        parameters=("int order",),
        arguments=(str(_lp_order(context)),),
    )


def _lp_order(context: NodeContext) -> int:
    order = context.int_attribute("p")
    if order < 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes the norm of order "
            f"{order}; ONNX defines `p` as the order of an Lp norm, which is positive."
        )
    return order


# --------------------------------------------------------------------------------------
# Emitting a pooling
# --------------------------------------------------------------------------------------


def _pool(context: NodeContext, recipe: _Recipe, geometry: _Geometry) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    fold = recipe(context, result.elem_type)
    verify_shape(context, result, geometry.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = f"{context.prefix}_pool_{fold.name}_{element}"
    definition = _TEMPLATE.substitute(
        name=name,
        element=element,
        parameters=_GEOMETRY_PARAMETERS,
        extra="".join(f",\n    {parameter}" for parameter in fold.parameters),
        state=expand(fold.state, result.elem_type),
        walk=_WALK.substitute(combine=expand(fold.combine, result.elem_type)),
        finish=expand(fold.finish, result.elem_type),
    )
    call = call_kernel(
        name, [result.expr, source.expr, *geometry.arguments, *fold.arguments]
    )
    return NodeEmission(
        functions=(*fold.helpers, CFunction(name, definition)), statements=(call,)
    )


def _pooling(recipe: _Recipe) -> Callable[[NodeContext], NodeEmission]:
    """A pooling over the window the node's own attributes describe."""
    return lambda context: _pool(context, recipe, _geometry(context))


def _global_pooling(recipe: _Recipe) -> Callable[[NodeContext], NodeEmission]:
    """A pooling over one window covering every spatial position of the operand."""
    return lambda context: _pool(context, recipe, _global_geometry(context))


def _max_pool(context: NodeContext) -> NodeEmission:
    """MaxPool, which also reports where each maximum came from when asked to."""
    geometry = _geometry(context)
    indices = context.outputs[1] if len(context.outputs) > 1 else None
    if indices is None:
        return _pool(context, _max_fold, geometry)

    source = context.require_input(0)
    result = context.require_output(0)
    verify_shape(context, result, geometry.result_shape)
    verify_shape(context, indices, geometry.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = f"{context.prefix}_pool_max_indexed_{element}"
    definition = _INDEXED_TEMPLATE.substitute(
        name=name,
        element=element,
        parameters=_GEOMETRY_PARAMETERS,
        identity=extremum_identity(result.elem_type, largest=True),
        better=extremum_test(result.elem_type, largest=True, last=False),
    )
    call = call_kernel(
        name,
        [
            result.expr,
            indices.expr,
            source.expr,
            *geometry.arguments,
            extents(geometry.index_strides(context.int_attribute("storage_order"))),
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _max_unpool(context: NodeContext) -> NodeEmission:
    """MaxUnpool: every element written back where the pooling that chose it read it."""
    source = context.require_input(0)
    positions = context.require_input(1)
    result = context.require_output(0)
    if positions.shape != source.shape:
        raise CompileError(
            f"Node `{context.label}`: `MaxUnpool` takes one index per value, but "
            f"`{positions.name}` has shape {list(positions.shape)} against "
            f"`{source.name}`'s {list(source.shape)}."
        )
    verify_shape(context, result, _unpooled_shape(context, source))
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = f"{context.prefix}_maxunpool_{element}"
    definition = _UNPOOL_TEMPLATE.substitute(
        name=name, element=element, zero=scalar_literal(0, result.elem_type)
    )
    call = checked_call(
        context,
        name,
        [
            result.expr,
            source.expr,
            positions.expr,
            f"{source.elem_count}u",
            f"{result.elem_count}u",
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


# --------------------------------------------------------------------------------------
# Reading the geometry off the node
# --------------------------------------------------------------------------------------


def _geometry(context: NodeContext) -> _Geometry:
    """The window a pooling node states, resolved to concrete extents and pads."""
    source = context.require_input(0)
    rank = _spatial_rank(context, source)
    window_shape = _kernel_shape(context, rank)
    strides = spatial_attribute(context, "strides", rank, 1)
    dilations = spatial_attribute(context, "dilations", rank, 1)
    begins, ends = resolve_pads(
        context, source.shape[2:], window_shape, dilations, strides
    )
    return _Geometry(
        batch_count=source.shape[0],
        channels=source.shape[1],
        input_shape=source.shape[2:],
        output_shape=_window_positions(
            context, source.shape[2:], window_shape, dilations, strides, begins, ends
        ),
        window_shape=window_shape,
        strides=strides,
        dilations=dilations,
        pads=begins,
        counted_shape=tuple(
            extent + end for extent, end in zip(source.shape[2:], ends)
        ),
    )


def _global_geometry(context: NodeContext) -> _Geometry:
    """A `Global*` pooling: one window covering every spatial position of the operand."""
    source = context.require_input(0)
    rank = _spatial_rank(context, source)
    return _Geometry(
        batch_count=source.shape[0],
        channels=source.shape[1],
        input_shape=source.shape[2:],
        output_shape=(1,) * rank,
        window_shape=source.shape[2:],
        strides=(1,) * rank,
        dilations=(1,) * rank,
        pads=(0,) * rank,
        counted_shape=source.shape[2:],
    )


def _spatial_rank(context: NodeContext, source: TensorRef) -> int:
    if len(source.shape) < 3:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` pools a batch of "
            f"multi-channel signals — a tensor of rank 3 or more — but `{source.name}` "
            f"has shape {list(source.shape)}."
        )
    return len(source.shape) - 2


def _kernel_shape(context: NodeContext, rank: int) -> tuple[int, ...]:
    window_shape = spatial_extents(context, "kernel_shape", rank)
    if window_shape is None:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` states no `kernel_shape`, "
            "which ONNX defines as a required attribute."
        )
    return window_shape


def _window_positions(
    context: NodeContext,
    input_shape: Sequence[int],
    window_shape: Sequence[int],
    dilations: Sequence[int],
    strides: Sequence[int],
    begins: Sequence[int],
    ends: Sequence[int],
) -> tuple[int, ...]:
    """How many windows fit each axis, under `ceil_mode`'s rounding.

    Rounding up lets the last window reach past the operand's padded end, where the taps
    beyond the pads are simply not read — unless it would *start* past the padding, in which
    case the window would be nothing but padding and ONNX drops the position instead.
    """
    ceil_mode = context.int_attribute("ceil_mode") != 0
    positions = []
    for extent, window, dilation, stride, begin, end in zip(
        input_shape, window_shape, dilations, strides, begins, ends
    ):
        reach = extent + begin + end - (window - 1) * dilation - 1
        if not ceil_mode:
            positions.append(reach // stride + 1)
            continue
        count = -(-reach // stride) + 1
        positions.append(count - 1 if (count - 1) * stride >= extent + begin else count)
    return tuple(positions)


def _unpooled_shape(context: NodeContext, source: TensorRef) -> tuple[int, ...]:
    """The extent the max pooling that produced this operand read, which is what is filled.

    ONNX also takes the result's shape from an optional `output_shape` operand. One the graph
    does not fix makes that shape depend on input data, which the frontend rejects before any
    kernel is reached; for one it does fix, ONNX's own shape inference derives nothing at all,
    so there is no shape to compile against either way.
    """
    rank = _spatial_rank(context, source)
    window_shape = _kernel_shape(context, rank)
    strides = spatial_attribute(context, "strides", rank, 1)
    begins, ends = declared_pads(context, rank, auto_pad_mode(context)) or (
        (0,) * rank,
        (0,) * rank,
    )
    return (
        *source.shape[:2],
        *(
            (extent - 1) * stride - begin - end + window
            for extent, stride, begin, end, window in zip(
                source.shape[2:], strides, begins, ends, window_shape
            )
        ),
    )


register_kernel("", "AveragePool", _VERSIONS, _pooling(_average_fold))
register_kernel("", "LpPool", _VERSIONS, _pooling(_lp_fold))
register_kernel("", "MaxPool", _VERSIONS, _max_pool)
register_kernel("", "GlobalAveragePool", _VERSIONS, _global_pooling(_average_fold))
register_kernel("", "GlobalLpPool", _VERSIONS, _global_pooling(_lp_fold))
register_kernel("", "GlobalMaxPool", _VERSIONS, _global_pooling(_max_fold))
register_kernel("", "MaxUnpool", _VERSIONS, _max_unpool)
