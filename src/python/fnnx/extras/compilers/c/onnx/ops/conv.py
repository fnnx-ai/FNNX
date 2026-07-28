"""The convolutions: the sliding window, and where each of its taps lands in the operand.

Every convolution is the same walk whatever its rank: for each output position, each tap of
the filter reads the operand at `position * stride + tap * dilation - pad` along each spatial
axis, skipping the taps that fall outside it. So there is one kernel per element type, taking
the geometry — the extents, strides, dilations and pads of each axis — as compile-time
literals, and `group` splits the channels into independent stacks addressed by the same walk.

`ConvTranspose` walks the same geometry backwards. It is the gradient of a convolution, so
each of *its* output positions reads the operand at `(position + pad - tap * dilation) /
stride` — the same relation solved the other way, which drops the taps the stride does not
divide. `DeformConv` keeps the forward walk but shifts each tap by an offset it reads at run
time, so the position it samples falls between elements and is interpolated. `Col2Im` runs the
backward walk on its own, with no filter to weight by: it is the same geometry folding a stack
of blocks back into the image they were cut from.

The geometry itself is resolved at compile time rather than in the kernel: `auto_pad` and
`output_shape` turn into concrete pads — through the shared reading of the window attributes
in `window.py`, plus the backward walk's own arithmetic here — and the result shape those pads
imply is checked against the one ONNX inferred, so a disagreement between the two stops the
compile instead of writing past a buffer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_type_name
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
    kernel_name,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents, math_suffix
from fnnx.extras.compilers.c.onnx.ops.window import (
    auto_pad_mode,
    declared_pads,
    offsets,
    output_extents,
    resolve_pads,
    spatial_attribute,
    spatial_extents,
)

# The geometry every walk of a sliding window takes, in the order `Geometry.arguments` fills
# it; the quantized convolutions take the same block after their own operands.
WINDOW_PARAMETERS = """\
    size_t batch_count,
    size_t groups,
    size_t group_channels,
    size_t group_filters,
    size_t input_size,
    size_t output_size,
    size_t window_size,
    int spatial_rank,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* window_shape,
    const size_t* strides,
    const size_t* dilations,
    const ptrdiff_t* pads"""

_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* weights,
    const $element* bias,
$parameters)
{
    size_t batch, group, filter, position, tap, channel;
    for (batch = 0; batch < batch_count; ++batch) {
        for (group = 0; group < groups; ++group) {
            const $element* plane =
                in + (batch * groups + group) * group_channels * input_size;
            for (filter = 0; filter < group_filters; ++filter) {
                const size_t channel_index = group * group_filters + filter;
                const $element* window =
                    weights + channel_index * group_channels * window_size;
                $element* result =
                    out + (batch * groups * group_filters + channel_index) * output_size;
                for (position = 0; position < output_size; ++position) {
                    $element sum = (bias != NULL) ? bias[channel_index] : $zero;
                    for (tap = 0; tap < window_size; ++tap) {
                        size_t remaining_position = position;
                        size_t remaining_tap = tap;
                        size_t offset = 0;
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
                            if (coordinate < 0
                                    || coordinate >= (ptrdiff_t)input_shape[axis]) {
                                inside = 0;
                            } else {
                                offset += (size_t)coordinate * stride;
                            }
                            stride *= input_shape[axis];
                        }
                        if (inside) {
                            for (channel = 0; channel < group_channels; ++channel) {
                                sum += plane[channel * input_size + offset]
                                     * window[channel * window_size + tap];
                            }
                        }
                    }
                    result[position] = sum;
                }
            }
        }
    }
}""")

_TRANSPOSE_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* weights,
    const $element* bias,
$parameters)
{
    size_t batch, group, filter, position, tap, channel;
    for (batch = 0; batch < batch_count; ++batch) {
        for (group = 0; group < groups; ++group) {
            const $element* plane =
                in + (batch * groups + group) * group_channels * input_size;
            for (filter = 0; filter < group_filters; ++filter) {
                const size_t channel_index = group * group_filters + filter;
                /* The filter's taps for every channel of this group, which `W` holds one
                   input channel at a time: (C, M / group, ...). */
                const $element* stack =
                    weights
                        + (group * group_channels * group_filters + filter) * window_size;
                $element* result =
                    out + (batch * groups * group_filters + channel_index) * output_size;
                for (position = 0; position < output_size; ++position) {
                    $element sum = (bias != NULL) ? bias[channel_index] : $zero;
                    for (tap = 0; tap < window_size; ++tap) {
                        size_t remaining_position = position;
                        size_t remaining_tap = tap;
                        size_t offset = 0;
                        size_t stride = 1;
                        int inside = 1;
                        int axis;
                        for (axis = spatial_rank - 1; axis >= 0; --axis) {
                            const ptrdiff_t step = (ptrdiff_t)strides[axis];
                            const ptrdiff_t reach =
                                (ptrdiff_t)(remaining_position % output_shape[axis])
                                + pads[axis]
                                - (ptrdiff_t)(remaining_tap % window_shape[axis])
                                    * (ptrdiff_t)dilations[axis];
                            const ptrdiff_t coordinate = reach / step;
                            remaining_position /= output_shape[axis];
                            remaining_tap /= window_shape[axis];
                            /* A reach the stride does not divide lands between two of the
                               operand's elements, where this tap contributes nothing. */
                            if (reach % step != 0 || coordinate < 0
                                    || coordinate >= (ptrdiff_t)input_shape[axis]) {
                                inside = 0;
                            } else {
                                offset += (size_t)coordinate * stride;
                            }
                            stride *= input_shape[axis];
                        }
                        if (inside) {
                            for (channel = 0; channel < group_channels; ++channel) {
                                sum += plane[channel * input_size + offset]
                                     * stack[channel * group_filters * window_size + tap];
                            }
                        }
                    }
                    result[position] = sum;
                }
            }
        }
    }
}""")

_SAMPLE_TEMPLATE = Template("""\
static $element $name(
    const $element* plane,
    size_t rows,
    size_t columns,
    $element row,
    $element column)
{
    $element row_floor, column_floor, row_fraction, column_fraction, total;
    ptrdiff_t top, left;
    int down, right;
    /* At or past either border every corner falls outside the plane, and a coordinate that
       is not a number cannot be floored into an index at all: both sample nothing. */
    if (!(row > -$one) || !(row < ($element)rows)
            || !(column > -$one) || !(column < ($element)columns)) {
        return $zero;
    }
    row_floor = floor$f(row);
    column_floor = floor$f(column);
    top = (ptrdiff_t)row_floor;
    left = (ptrdiff_t)column_floor;
    row_fraction = row - row_floor;
    column_fraction = column - column_floor;
    total = $zero;
    for (down = 0; down < 2; ++down) {
        const ptrdiff_t sampled_row = top + down;
        if (sampled_row < 0 || sampled_row >= (ptrdiff_t)rows) {
            continue;
        }
        for (right = 0; right < 2; ++right) {
            const ptrdiff_t sampled_column = left + right;
            if (sampled_column < 0 || sampled_column >= (ptrdiff_t)columns) {
                continue;
            }
            total += (down ? row_fraction : $one - row_fraction)
                   * (right ? column_fraction : $one - column_fraction)
                   * plane[(size_t)sampled_row * columns + (size_t)sampled_column];
        }
    }
    return total;
}""")

_DEFORM_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* weights,
    const $element* offsets,
    const $element* bias,
    const $element* mask,
    size_t batch_count,
    size_t groups,
    size_t group_channels,
    size_t group_filters,
    size_t offset_channels,
    size_t offset_groups,
    size_t input_rows,
    size_t input_columns,
    size_t output_rows,
    size_t output_columns,
    size_t window_rows,
    size_t window_columns,
    size_t row_stride,
    size_t column_stride,
    size_t row_dilation,
    size_t column_dilation,
    ptrdiff_t row_pad,
    ptrdiff_t column_pad)
{
    const size_t input_size = input_rows * input_columns;
    const size_t output_size = output_rows * output_columns;
    const size_t window_size = window_rows * window_columns;
    size_t batch, group, filter, row, column, channel, tap_row, tap_column;
    for (batch = 0; batch < batch_count; ++batch) {
        for (group = 0; group < groups; ++group) {
            for (filter = 0; filter < group_filters; ++filter) {
                const size_t channel_index = group * group_filters + filter;
                for (row = 0; row < output_rows; ++row) {
                    for (column = 0; column < output_columns; ++column) {
                        const size_t position = row * output_columns + column;
                        $element sum = (bias != NULL) ? bias[channel_index] : $zero;
                        for (channel = 0; channel < group_channels; ++channel) {
                            const size_t source = group * group_channels + channel;
                            const $element* plane =
                                in
                                    + (batch * groups * group_channels + source)
                                        * input_size;
                            /* One deformation per (offset group, tap): a plane of `mask`
                               weights, and two planes of `offsets` -- a coordinate per
                               spatial axis. */
                            const size_t deformation =
                                (batch * offset_groups + source / offset_channels)
                                    * window_size;
                            for (tap_row = 0; tap_row < window_rows; ++tap_row) {
                                for (tap_column = 0;
                                        tap_column < window_columns;
                                        ++tap_column) {
                                    const size_t tap =
                                        tap_row * window_columns + tap_column;
                                    const size_t shift =
                                        2 * (deformation + tap) * output_size + position;
                                    const $element sampled_row =
                                        ($element)((ptrdiff_t)(row * row_stride
                                            + tap_row * row_dilation) - row_pad)
                                        + offsets[shift];
                                    const $element sampled_column =
                                        ($element)((ptrdiff_t)(column * column_stride
                                            + tap_column * column_dilation) - column_pad)
                                        + offsets[shift + output_size];
                                    $element weight =
                                        weights[(channel_index * group_channels + channel)
                                            * window_size + tap];
                                    if (mask != NULL) {
                                        weight *=
                                            mask[(deformation + tap) * output_size
                                                + position];
                                    }
                                    sum += $sample(
                                        plane,
                                        input_rows,
                                        input_columns,
                                        sampled_row,
                                        sampled_column) * weight;
                                }
                            }
                        }
                        out[(batch * groups * group_filters + channel_index) * output_size
                            + position] = sum;
                    }
                }
            }
        }
    }
}""")

_COL2IM_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t plane_count,
    size_t column_count,
    size_t image_size,
    size_t window_size,
    int spatial_rank,
    const size_t* image_shape,
    const size_t* block_shape,
    const size_t* column_shape,
    const size_t* strides,
    const size_t* dilations,
    const ptrdiff_t* pads)
{
    size_t plane, position, tap;
    for (plane = 0; plane < plane_count; ++plane) {
        const $element* columns = in + plane * window_size * column_count;
        $element* result = out + plane * image_size;
        for (position = 0; position < image_size; ++position) {
            $element sum = $zero;
            for (tap = 0; tap < window_size; ++tap) {
                size_t remaining_position = position;
                size_t remaining_tap = tap;
                size_t offset = 0;
                size_t stride = 1;
                int inside = 1;
                int axis;
                for (axis = spatial_rank - 1; axis >= 0; --axis) {
                    const ptrdiff_t step = (ptrdiff_t)strides[axis];
                    const ptrdiff_t reach =
                        (ptrdiff_t)(remaining_position % image_shape[axis])
                        + pads[axis]
                        - (ptrdiff_t)(remaining_tap % block_shape[axis])
                            * (ptrdiff_t)dilations[axis];
                    const ptrdiff_t coordinate = reach / step;
                    remaining_position /= image_shape[axis];
                    remaining_tap /= block_shape[axis];
                    /* A reach the stride does not divide falls between two block positions,
                       where no block placed this tap here. */
                    if (reach % step != 0 || coordinate < 0
                            || coordinate >= (ptrdiff_t)column_shape[axis]) {
                        inside = 0;
                    } else {
                        offset += (size_t)coordinate * stride;
                    }
                    stride *= column_shape[axis];
                }
                if (inside) {
                    sum += columns[tap * column_count + offset];
                }
            }
            result[position] = sum;
        }
    }
}""")

# Conv arrived at opset 1, 11 revised `auto_pad` and 22 widened the element types. Only 22 is
# claimed: it is the revision the reference evaluator is version-faithful for and the one
# every Conv test in the backend corpus imports, so it is the only one anything can vouch
# for. A model importing an older one gets the unsupported-version error. ConvTranspose (1,
# 11, 22) and DeformConv (19, 22) are claimed at their newest revision for the same reason.
_VERSIONS = (22,)

# Col2Im has had one revision only, the one it arrived at.
_COL2IM_VERSIONS = (18,)

# DeformConv samples between elements, which the compiler only emits for two spatial axes:
# ONNX's reference evaluator implements no other rank, and the backend corpus tests none, so
# nothing could vouch for the code an N-d sampler would emit.
_DEFORM_SPATIAL_RANK = 2


@dataclass(frozen=True)
class Geometry:
    """The convolution's shape, resolved to the literals the kernel walks it with."""

    batch_count: int
    groups: int
    group_channels: int
    group_filters: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    window_shape: tuple[int, ...]
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    pads: tuple[int, ...]

    @property
    def result_shape(self) -> tuple[int, ...]:
        return (
            self.batch_count,
            self.groups * self.group_filters,
            *self.output_shape,
        )

    @property
    def arguments(self) -> list[str]:
        """Call-site literals for the geometry parameters a window kernel takes."""
        return [
            f"{self.batch_count}u",
            f"{self.groups}u",
            f"{self.group_channels}u",
            f"{self.group_filters}u",
            f"{math.prod(self.input_shape)}u",
            f"{math.prod(self.output_shape)}u",
            f"{math.prod(self.window_shape)}u",
            str(len(self.output_shape)),
            extents(self.input_shape),
            extents(self.output_shape),
            extents(self.window_shape),
            extents(self.strides),
            extents(self.dilations),
            offsets(self.pads),
        ]


def _conv(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    weights = context.require_input(1)
    bias = context.optional_input(2)
    result = context.require_output(0)
    geometry = convolution_geometry(context, source, weights)
    verify_bias(context, bias, geometry.groups * geometry.group_filters)
    verify_shape(context, result, geometry.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    definition = _TEMPLATE.substitute(
        name=name,
        element=element,
        zero=scalar_literal(0, result.elem_type),
        parameters=WINDOW_PARAMETERS,
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            weights.expr,
            "NULL" if bias is None else bias.expr,
            *geometry.arguments,
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def convolution_geometry(
    context: NodeContext, source: TensorRef, weights: TensorRef
) -> Geometry:
    """The walk a forward convolution takes, shared with the quantized convolutions."""
    rank = _spatial_rank(context, source, weights)
    groups = _groups(context)
    filters, group_channels = weights.shape[0], weights.shape[1]
    if source.shape[1] != group_channels * groups or filters % groups:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` in {groups} group(s) takes "
            f"{group_channels * groups} input and a multiple of {groups} output "
            f"channel(s), but `{source.name}` carries {source.shape[1]} and "
            f"`{weights.name}` produces {filters}."
        )

    window_shape = weights.shape[2:]
    _verify_kernel_shape(context, window_shape)
    strides = spatial_attribute(context, "strides", rank, 1)
    dilations = spatial_attribute(context, "dilations", rank, 1)
    begins, ends = resolve_pads(
        context, source.shape[2:], window_shape, dilations, strides
    )
    return Geometry(
        batch_count=source.shape[0],
        groups=groups,
        group_channels=group_channels,
        group_filters=filters // groups,
        input_shape=source.shape[2:],
        output_shape=output_extents(
            source.shape[2:], window_shape, dilations, strides, begins, ends
        ),
        window_shape=window_shape,
        strides=strides,
        dilations=dilations,
        pads=begins,
    )


# --------------------------------------------------------------------------------------
# The transposed convolution
# --------------------------------------------------------------------------------------


def _conv_transpose(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    weights = context.require_input(1)
    bias = context.optional_input(2)
    result = context.require_output(0)
    geometry = _transpose_geometry(context, source, weights)
    verify_bias(context, bias, geometry.groups * geometry.group_filters)
    verify_shape(context, result, geometry.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    definition = _TRANSPOSE_TEMPLATE.substitute(
        name=name,
        element=element,
        zero=scalar_literal(0, result.elem_type),
        parameters=WINDOW_PARAMETERS,
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            weights.expr,
            "NULL" if bias is None else bias.expr,
            *geometry.arguments,
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _transpose_geometry(
    context: NodeContext, source: TensorRef, weights: TensorRef
) -> Geometry:
    rank = _spatial_rank(context, source, weights)
    groups = _groups(context)
    channels, group_filters = weights.shape[0], weights.shape[1]
    # `W` is (C, M / group, ...) here rather than Conv's (M, C / group, ...): a transposed
    # convolution scatters each input channel over the filters instead of gathering.
    if source.shape[1] != channels or channels % groups:
        raise CompileError(
            f"Node `{context.label}`: `ConvTranspose` in {groups} group(s) takes a filter "
            f"holding one stack per input channel, in a multiple of {groups}, but "
            f"`{source.name}` carries {source.shape[1]} channel(s) against "
            f"`{weights.name}`'s {channels}."
        )

    window_shape = weights.shape[2:]
    _verify_kernel_shape(context, window_shape)
    strides = spatial_attribute(context, "strides", rank, 1)
    dilations = spatial_attribute(context, "dilations", rank, 1)
    begins, output_shape = _transpose_pads(
        context, source.shape[2:], window_shape, dilations, strides
    )
    return Geometry(
        batch_count=source.shape[0],
        groups=groups,
        group_channels=channels // groups,
        group_filters=group_filters,
        input_shape=source.shape[2:],
        output_shape=output_shape,
        window_shape=window_shape,
        strides=strides,
        dilations=dilations,
        pads=begins,
    )


def _transpose_pads(
    context: NodeContext,
    input_shape: Sequence[int],
    window_shape: Sequence[int],
    dilations: Sequence[int],
    strides: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The pad before each spatial axis and the result's extents, as ONNX derives them.

    A transposed convolution reaches `stride * (extent - 1) + output_padding + the window's
    dilated span` elements, and the pads crop that reach down to the result. Stating an
    `output_shape` or an `auto_pad` mode picks the result first and derives the pads from it
    instead, which is what makes the stride's ambiguity — a stride of 2 maps two operand
    extents onto one result extent — expressible.
    """
    rank = len(input_shape)
    mode = auto_pad_mode(context)
    declared = declared_pads(context, rank, mode)
    output_padding = spatial_attribute(context, "output_padding", rank, 0, minimum=0)
    reach = tuple(
        stride * (extent - 1) + padding + (window - 1) * dilation + 1
        for extent, window, dilation, stride, padding in zip(
            input_shape, window_shape, dilations, strides, output_padding
        )
    )
    requested = spatial_extents(context, "output_shape", rank, minimum=0)
    if declared is not None:
        begins, ends = declared
        return begins, requested or tuple(
            span - begin - end for span, begin, end in zip(reach, begins, ends)
        )
    if requested is None and mode in ("NOTSET", "VALID"):
        return (0,) * rank, reach

    # ONNX's SAME modes pad so that the result measures `extent * stride`; an explicit
    # `output_shape` names it outright. Either way the pads are what is left over, split
    # between the two ends — with the odd one going to the end SAME_UPPER names, as the
    # spec's own equations put it. A requested result the reach falls short of needs no pad
    # at all: the positions past the reach are ones no tap contributes to. A SAME mode is
    # left unclamped instead, so that a window whose span is narrower than its stride — where
    # ONNX's own shape inference stops agreeing with its equations — is caught by the result
    # shape rather than resolved here.
    output_shape = requested or tuple(
        extent * stride for extent, stride in zip(input_shape, strides)
    )
    totals = [span - extent for span, extent in zip(reach, output_shape)]
    if requested is not None:
        totals = [max(total, 0) for total in totals]
    begins = tuple(
        total // 2 if mode == "SAME_UPPER" else total - total // 2 for total in totals
    )
    return begins, output_shape


# --------------------------------------------------------------------------------------
# The deformable convolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Deformation:
    """What a deformable convolution reads beyond a plain one: offsets, and their groups."""

    geometry: Geometry
    offset_groups: int

    @property
    def offset_shape(self) -> tuple[int, ...]:
        """`offset`: two coordinates per tap per offset group, at every result position."""
        return (
            self.geometry.batch_count,
            self.offset_groups * math.prod(self.geometry.window_shape) * 2,
            *self.geometry.output_shape,
        )

    @property
    def mask_shape(self) -> tuple[int, ...]:
        return (self.offset_shape[0], self.offset_shape[1] // 2, *self.offset_shape[2:])


def _deform_conv(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    weights = context.require_input(1)
    offsets = context.require_input(2)
    bias = context.optional_input(3)
    mask = context.optional_input(4)
    result = context.require_output(0)
    deformation = _deform_geometry(context, source, weights)
    geometry = deformation.geometry
    _verify_operand(context, offsets, deformation.offset_shape)
    _verify_operand(context, mask, deformation.mask_shape)
    verify_bias(context, bias, geometry.groups * geometry.group_filters)
    verify_shape(context, result, geometry.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    sampler = CFunction(
        f"{context.prefix}_bilinear_{element}",
        _SAMPLE_TEMPLATE.substitute(
            name=f"{context.prefix}_bilinear_{element}",
            element=element,
            f=math_suffix(result.elem_type),
            one=scalar_literal(1, result.elem_type),
            zero=scalar_literal(0, result.elem_type),
        ),
    )
    name = kernel_name(context, element)
    definition = _DEFORM_TEMPLATE.substitute(
        name=name,
        element=element,
        zero=scalar_literal(0, result.elem_type),
        sample=sampler.name,
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            weights.expr,
            offsets.expr,
            "NULL" if bias is None else bias.expr,
            "NULL" if mask is None else mask.expr,
            f"{geometry.batch_count}u",
            f"{geometry.groups}u",
            f"{geometry.group_channels}u",
            f"{geometry.group_filters}u",
            f"{geometry.groups * geometry.group_channels // deformation.offset_groups}u",
            f"{deformation.offset_groups}u",
            *(f"{extent}u" for extent in geometry.input_shape),
            *(f"{extent}u" for extent in geometry.output_shape),
            *(f"{extent}u" for extent in geometry.window_shape),
            *(f"{stride}u" for stride in geometry.strides),
            *(f"{dilation}u" for dilation in geometry.dilations),
            *(str(pad) for pad in geometry.pads),
        ],
    )
    return NodeEmission(
        functions=(sampler, CFunction(name, definition)), statements=(call,)
    )


def _deform_geometry(
    context: NodeContext, source: TensorRef, weights: TensorRef
) -> _Deformation:
    rank = _spatial_rank(context, source, weights)
    if rank != _DEFORM_SPATIAL_RANK:
        raise CompileError(
            f"Node `{context.label}`: `DeformConv` is compiled for {_DEFORM_SPATIAL_RANK} "
            f"spatial axes — the rank ONNX's reference evaluator and backend tests cover — "
            f"but `{source.name}` of shape {list(source.shape)} has {rank}."
        )
    groups = _groups(context)
    offset_groups = context.int_attribute("offset_group")
    filters, group_channels = weights.shape[0], weights.shape[1]
    if source.shape[1] != group_channels * groups or filters % groups:
        raise CompileError(
            f"Node `{context.label}`: `DeformConv` in {groups} group(s) takes "
            f"{group_channels * groups} input and a multiple of {groups} output "
            f"channel(s), but `{source.name}` carries {source.shape[1]} and "
            f"`{weights.name}` produces {filters}."
        )
    if offset_groups < 1 or source.shape[1] % offset_groups:
        raise CompileError(
            f"Node `{context.label}`: `DeformConv` splits its {source.shape[1]} input "
            f"channel(s) into {offset_groups} offset group(s); ONNX defines "
            "`offset_group` as a positive count that divides them."
        )

    window_shape = weights.shape[2:]
    _verify_kernel_shape(context, window_shape)
    strides = spatial_attribute(context, "strides", rank, 1)
    dilations = spatial_attribute(context, "dilations", rank, 1)
    begins, ends = declared_pads(context, rank, "NOTSET") or ((0,) * rank, (0,) * rank)
    return _Deformation(
        geometry=Geometry(
            batch_count=source.shape[0],
            groups=groups,
            group_channels=group_channels,
            group_filters=filters // groups,
            input_shape=source.shape[2:],
            output_shape=output_extents(
                source.shape[2:], window_shape, dilations, strides, begins, ends
            ),
            window_shape=window_shape,
            strides=strides,
            dilations=dilations,
            pads=begins,
        ),
        offset_groups=offset_groups,
    )


# --------------------------------------------------------------------------------------
# Reading the geometry off the node
# --------------------------------------------------------------------------------------


def _spatial_rank(context: NodeContext, source: TensorRef, weights: TensorRef) -> int:
    """The number of axes the window slides along, once both operands agree on it."""
    op_type = context.node.op_type
    if len(source.shape) < 3:
        raise CompileError(
            f"Node `{context.label}`: `{op_type}` takes a batch of multi-channel signals "
            f"— a tensor of rank 3 or more — but `{source.name}` has shape "
            f"{list(source.shape)}."
        )
    if len(weights.shape) != len(source.shape):
        raise CompileError(
            f"Node `{context.label}`: `{op_type}` convolves `{source.name}` of shape "
            f"{list(source.shape)} with `{weights.name}` of shape "
            f"{list(weights.shape)}; ONNX defines both as rank "
            f"{len(source.shape)} — two leading axes and one per spatial axis."
        )
    return len(source.shape) - 2


def _groups(context: NodeContext) -> int:
    groups = context.int_attribute("group")
    if groups < 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` splits its channels into "
            f"{groups} group(s); ONNX defines `group` as a positive count."
        )
    return groups


def verify_bias(context: NodeContext, bias: TensorRef | None, channels: int) -> None:
    if bias is not None and bias.shape != (channels,):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes one bias per output "
            f"channel, but `{bias.name}` has shape {list(bias.shape)} against {channels} "
            "channel(s)."
        )


def _verify_operand(
    context: NodeContext, operand: TensorRef | None, expected: Sequence[int]
) -> None:
    """Refuse an operand the emitted addressing would read outside of.

    `offset` and `mask` are indexed by the geometry the other operands and the attributes
    fix, so one shaped for a different geometry is a compile error rather than a read past
    the end of a buffer.
    """
    if operand is not None and operand.shape != tuple(expected):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` addresses `{operand.name}` "
            f"as {list(expected)}, but it has shape {list(operand.shape)}."
        )


def _verify_kernel_shape(context: NodeContext, window_shape: tuple[int, ...]) -> None:
    """Refuse a `kernel_shape` that disagrees with the filter the node is actually handed.

    ONNX defines the attribute as inferred from `W` when absent, and says nothing about what
    a node stating a different one means — the reference pads for one shape and convolves
    with the other — so it is a compile error rather than a guess.
    """
    declared = context.attribute("kernel_shape", None)
    if declared is not None and tuple(int(value) for value in declared) != window_shape:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` declares `kernel_shape` "
            f"{[int(value) for value in declared]}, but the filter it is handed measures "
            f"{list(window_shape)}."
        )


# --------------------------------------------------------------------------------------
# Folding the columns back into an image
# --------------------------------------------------------------------------------------


def _col2im(context: NodeContext) -> NodeEmission:
    """Col2Im: every block written back over the image positions it was read from.

    The columns hold one value per (tap, block position) pair, and a block position is where
    a window of `block_shape` sat: the same geometry a convolution slides. So each image
    position sums the taps that reached it, found by the transposed convolution's own walk —
    `(position + pad - tap * dilation) / stride` — which is the relation solved for the block
    the tap came from, and which drops the taps the stride does not divide.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    image_shape = _shape_operand(context, 1, "image_shape")
    block_shape = _shape_operand(context, 2, "block_shape")
    rank = len(image_shape)
    if len(block_shape) != rank:
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` was given {len(block_shape)} block extent(s) "
            f"for a {rank}-dimensional image; ONNX defines one per spatial axis."
        )
    if len(source.shape) != 3:
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` takes a batch of column stacks — a tensor of "
            f"rank 3 — but `{source.name}` has shape {list(source.shape)}."
        )

    if result.elem_type == TensorProto.BOOL:
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` of a "
            f"`{element_type_name(result.elem_type)}` tensor is not supported by the C "
            "compiler; summing truth values has no defined result."
        )

    window_size = math.prod(block_shape)
    if window_size < 1 or source.shape[1] % window_size != 0:
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` folds blocks of {window_size} value(s), "
            f"which does not divide the {source.shape[1]} row(s) of `{source.name}` into "
            "whole channels."
        )
    channels = source.shape[1] // window_size
    verify_shape(context, result, (source.shape[0], channels, *image_shape))

    dilations = spatial_attribute(context, "dilations", rank, 1)
    strides = spatial_attribute(context, "strides", rank, 1)
    begins, ends = declared_pads(context, rank, "NOTSET") or ((0,) * rank, (0,) * rank)
    column_shape = output_extents(
        image_shape, block_shape, dilations, strides, begins, ends
    )
    if (
        any(extent < 1 for extent in column_shape)
        or math.prod(column_shape) != source.shape[2]
    ):
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` places {list(column_shape)} block(s) over an "
            f"image of {list(image_shape)}, but `{source.name}` holds {source.shape[2]} "
            "column(s)."
        )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = f"{context.prefix}_col2im_{element}"
    definition = _COL2IM_TEMPLATE.substitute(
        name=name, element=element, zero=scalar_literal(0, result.elem_type)
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            f"{source.shape[0] * channels}u",
            f"{source.shape[2]}u",
            f"{math.prod(image_shape)}u",
            f"{window_size}u",
            str(rank),
            extents(image_shape),
            extents(block_shape),
            extents(column_shape),
            extents(strides),
            extents(dilations),
            offsets(begins),
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _shape_operand(context: NodeContext, index: int, role: str) -> tuple[int, ...]:
    values = context.constant_input(index)
    if values is None:
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` takes its `{role}` from "
            f"`{context.require_input(index).name}`, which is not known at compile time; "
            "the shape of the result then depends on input data, which the C compiler "
            "cannot compile."
        )
    extents_ = tuple(int(value) for value in values.reshape(-1))
    if any(extent < 1 for extent in extents_):
        raise CompileError(
            f"Node `{context.label}`: `Col2Im` was given `{role}` {list(extents_)}; ONNX "
            "defines them as extents, which are positive."
        )
    return extents_


register_kernel("", "Conv", _VERSIONS, _conv)
register_kernel("", "ConvTranspose", _VERSIONS, _conv_transpose)
register_kernel("", "DeformConv", _VERSIONS, _deform_conv)
register_kernel("", "Col2Im", _COL2IM_VERSIONS, _col2im)
