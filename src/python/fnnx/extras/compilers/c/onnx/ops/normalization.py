"""The normalizations: standardizing a tensor by statistics taken from its own elements.

LayerNormalization, InstanceNormalization, GroupNormalization, MeanVarianceNormalization and
BatchNormalization in training mode all do the same thing to different groups of elements:
take the group's mean and variance, subtract, divide, and — for all but the fourth — apply a
scale and a bias that vary along an axis. They share one loop nest, the group and element
loops `axes.py` builds, walked three times: once for the mean, once for the squared
deviations, once to write the result. Which axes form a group, where the epsilon guarding the
division goes, and how the affine operands are addressed is what each op supplies.
BatchNormalization at inference, RMSNormalization, LpNormalization and LRN take no group
statistics at all and carry loops of their own.

The variance comes from the squared deviations rather than from `E[X^2] - E[X]^2`. The two
are the same quantity, and the deviation form is the one the reference evaluator and the
corpus's own expectations compute for every op here except GroupNormalization and
MeanVarianceNormalization, whose official function bodies subtract the squares — a
difference of one rounding error, on the side that cancellation cannot hurt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    broadcast_strides,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    GROUP_PARAMETERS,
    Grouping,
    call_kernel,
    group_axes,
    kernel_name,
    normalize_axes,
    normalize_axis,
    offset_helper,
    verify_same_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents, math_suffix

# BatchNormalization-15 only reworded 14's documentation and spelled out that the statistics
# of a float16 tensor accumulate in float. The revisions below 14 are a different op — they
# carry the `spatial` and `is_test` attributes and up to five outputs — and none is claimed.
_BATCH_VERSIONS = (14, 15)
_LAYER_VERSIONS = (17,)
_RMS_VERSIONS = (23,)
# GroupNormalization-18 took `scale` and `bias` per group rather than per channel and had no
# `stash_type`; it is a different op, so only the current revision is served.
_GROUP_VERSIONS = (21,)
# The remaining families changed only their type constraints across the revisions listed.
_INSTANCE_VERSIONS = (6, 22)
_LP_VERSIONS = (1, 22)
_MVN_VERSIONS = (9, 13)
_LRN_VERSIONS = (1, 13)

# MeanVarianceNormalization's ONNX function body adds its epsilon to the standard deviation
# rather than to the variance, and standardizes these axes when the node names none.
_MVN_EPSILON = 1e-9
_MVN_DEFAULT_AXES = (0, 2, 3)

# ONNX defines LpNormalization for these orders only.
_SUPPORTED_ORDERS = (1, 2)

# The three passes over a group: the mean, the squared deviations around it, and the write.
# A group with no elements at all divides zero by zero, which is the NaN numpy fills such a
# statistic with — the reference's own answer rather than a guess at what it should be.
_STANDARDIZE_TEMPLATE = Template("""\
static void $name(
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
$bases\
        $stash total = $stash_zero;
        $stash spread = $stash_zero;
        for (index = 0; index < group_size; ++index) {
            total += ($stash)in[base
                + $element_offset];
        }
        const $stash mean = total / ($stash)group_size;
        for (index = 0; index < group_size; ++index) {
            const $stash deviation = ($stash)in[base
                + $element_offset] - mean;
            spread += deviation * deviation;
        }
        const $stash variance = spread / ($stash)group_size;
        const $stash factor = $factor;
        for (index = 0; index < group_size; ++index) {
            const size_t position = base
                + $element_offset;
$positions\
            const $stash centred = ($stash)in[position] - mean;
            out[position] = ($element)(
                $formula);
        }
$statistics\
    }
}""")

# Inference-mode BatchNormalization: the statistics arrive as operands, so there is nothing
# to reduce and an element's channel follows from where it sits in the buffer.
_BATCH_TEST_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $affine* scale,
    const $affine* bias,
    const $statistic* mean,
    const $statistic* variance,
    size_t count,
    size_t inner,
    size_t channels,
    $stash epsilon)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        const size_t channel = (index / inner) % channels;
        out[index] = ($element)(($stash)scale[channel]
            * (($stash)in[index] - ($stash)mean[channel])
            / sqrt$f(($stash)variance[channel] + epsilon)
            + ($stash)bias[channel]);
    }
}""")

# RMSNormalization has no mean to centre on: each element is scaled by the reciprocal root of
# its group's mean square. The reference evaluator multiplies by that reciprocal rather than
# dividing by the root, and applies the linear coefficient afterwards, which is the rounding
# the corpus's own expectations were generated with.
_RMS_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* scale,
$parameters,
    const size_t* scale_kept_strides,
    const size_t* scale_reduced_strides,
    $element epsilon)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        const size_t scale_base =
            $offset(group, kept_rank, kept_shape, scale_kept_strides);
        $element total = $zero;
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base + $element_offset];
            total += x * x;
        }
        const $element factor =
            $one / sqrt$f(total / ($element)group_size + epsilon);
        for (index = 0; index < group_size; ++index) {
            const size_t position = base + $element_offset;
            const size_t scale_position = scale_base
                + $offset(index, reduced_rank, reduced_shape, scale_reduced_strides);
            out[position] = in[position] * factor * scale[scale_position];
        }
    }
}""")

# LpNormalization divides a group by its own norm, and answers a norm of zero with zero
# rather than with the NaN the division would give — as the reference evaluator does.
_LP_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element total = $zero;
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base
                + $element_offset];
            total += $term;
        }
        const $element norm = $norm;
        for (index = 0; index < group_size; ++index) {
            const size_t position = base
                + $element_offset;
            out[position] = (norm == $zero) ? $zero : in[position] / norm;
        }
    }
}""")

# LRN's group is a window of channels around each element's own, so the channel axis is
# walked explicitly rather than through the grouping helpers.
_LRN_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t batches,
    size_t channels,
    size_t inner,
    size_t before,
    size_t after,
    $element bias,
    $element scaled_alpha,
    $element beta)
{
    size_t batch, channel, position, neighbour;
    for (batch = 0; batch < batches; ++batch) {
        for (channel = 0; channel < channels; ++channel) {
            const size_t begin = (channel >= before) ? channel - before : 0;
            const size_t last = channel + after + 1;
            const size_t end = (last < channels) ? last : channels;
            for (position = 0; position < inner; ++position) {
                const size_t target = (batch * channels + channel) * inner + position;
                $element square_sum = $zero;
                for (neighbour = begin; neighbour < end; ++neighbour) {
                    const $element value =
                        in[(batch * channels + neighbour) * inner + position];
                    square_sum += value * value;
                }
                out[target] =
                    in[target] / pow$f(bias + scaled_alpha * square_sum, beta);
            }
        }
    }
}""")


@dataclass(frozen=True)
class _Operand:
    """An operand read alongside the data: a scale, a bias, a running statistic.

    `strides` addresses it from the data's own coordinates, so a per-channel vector and a
    tensor of the normalized shape are the same thing to the kernel — a stride per axis, zero
    on every axis the operand does not vary along. `per_group` marks one that varies no
    faster than the groups do and is read as `<name>_base`, which is all a statistic blended
    once per group needs; anything read per element carries `<name>_position` as well.
    """

    name: str
    ref: TensorRef
    strides: tuple[int, ...]
    per_group: bool = False


@dataclass(frozen=True)
class _Statistic:
    """A per-group value the node takes as an output of its own.

    `value` is C over the locals the group loop has computed — `mean`, `variance`, `factor` —
    and over any operand's `<name>_base`, which for a per-group operand is its own index.
    """

    index: int
    name: str
    ref: TensorRef
    value: str


@dataclass(frozen=True)
class _Scalar:
    """An attribute the kernel reads as an argument rather than as an inlined literal."""

    name: str
    elem_type: int
    value: float


def _standardize(
    context: NodeContext,
    *,
    data: TensorRef,
    result: TensorRef,
    grouping: Grouping,
    stash: int,
    factor: str,
    formula: str,
    operands: Sequence[_Operand] = (),
    statistics: Sequence[_Statistic] = (),
    scalars: Sequence[_Scalar] = (),
) -> NodeEmission:
    """Emit the group-statistics kernel and the call site addressing this node's buffers.

    `factor` is C over the group's `mean` and `variance`, and `formula` is C over `factor`,
    the element's own `centred` deviation from the mean, and each operand's
    `<name>[<name>_position]`; both are `$`-templated over the element and stash types.
    """
    offset = offset_helper(context.prefix)
    element_offset = (
        f"{offset.name}(index, reduced_rank, reduced_shape, reduced_strides)"
    )
    element = c_type(result.elem_type)

    parameters = [f"    {element}* out"]
    parameters += [
        f"    {c_type(entry.ref.elem_type)}* {entry.name}" for entry in statistics
    ]
    parameters.append(f"    const {c_type(data.elem_type)}* in")
    parameters += [
        f"    const {c_type(operand.ref.elem_type)}* {operand.name}"
        for operand in operands
    ]
    parameters.append(GROUP_PARAMETERS)
    for operand in operands:
        parameters.append(f"    const size_t* {operand.name}_kept_strides")
        if not operand.per_group:
            parameters.append(f"    const size_t* {operand.name}_reduced_strides")
    parameters += [
        f"    {c_type(scalar.elem_type)} {scalar.name}" for scalar in scalars
    ]

    def expand(text: str) -> str:
        return Template(text).substitute(
            element=element,
            stash=c_type(stash),
            f=math_suffix(stash),
            one=scalar_literal(1, stash),
            zero=scalar_literal(0, stash),
        )

    name = _kernel_name(
        context,
        "",
        [
            result.elem_type,
            stash,
            *(entry.ref.elem_type for entry in statistics),
            *(operand.ref.elem_type for operand in operands),
        ],
        f"p{len(operands)}" + "".join(f"s{entry.index}" for entry in statistics),
    )
    definition = _STANDARDIZE_TEMPLATE.substitute(
        name=name,
        parameters=",\n".join(parameters),
        offset=offset.name,
        element_offset=element_offset,
        element=element,
        stash=c_type(stash),
        stash_zero=scalar_literal(0, stash),
        factor=expand(factor),
        formula=expand(formula),
        bases="".join(
            f"        const size_t {operand.name}_base = {offset.name}(\n"
            f"            group, kept_rank, kept_shape, {operand.name}_kept_strides);\n"
            for operand in operands
        ),
        positions="".join(
            f"            const size_t {operand.name}_position = {operand.name}_base\n"
            f"                + {offset.name}(index, reduced_rank, reduced_shape,\n"
            f"                    {operand.name}_reduced_strides);\n"
            for operand in operands
            if not operand.per_group
        ),
        statistics="".join(
            f"        {entry.name}[group] = "
            f"({c_type(entry.ref.elem_type)})({expand(entry.value)});\n"
            for entry in statistics
        ),
    )

    arguments = [result.expr]
    arguments += [entry.ref.expr for entry in statistics]
    arguments.append(data.expr)
    arguments += [operand.ref.expr for operand in operands]
    arguments += grouping.arguments
    for operand in operands:
        kept, reduced = _split_strides(grouping, operand.strides)
        arguments += [kept] if operand.per_group else [kept, reduced]
    arguments += [scalar_literal(scalar.value, scalar.elem_type) for scalar in scalars]
    return NodeEmission(
        functions=(offset, CFunction(name, definition)),
        statements=(call_kernel(name, arguments),),
    )


def _split_strides(grouping: Grouping, strides: Sequence[int]) -> list[str]:
    """An operand's per-axis strides, split the way the grouping splits the data's axes."""
    return [
        extents([strides[axis] for axis in grouping.kept_axes]),
        extents([strides[axis] for axis in grouping.reduced_axes]),
    ]


def _kernel_name(
    context: NodeContext, variant: str, elem_types: Sequence[int], form: str
) -> str:
    """A name encoding everything the emitted code depends on beyond the call-site literals.

    Two nodes running the same op share a kernel when their element types, the operands and
    statistics they name, and the formula their attributes select all agree; anything else
    would be two kernels colliding on one name.
    """
    names = [numpy_dtype_name(elem_type) for elem_type in elem_types]
    types = names[0] if len(set(names)) == 1 else "_".join(names)
    return kernel_name(context, *(part for part in (variant, form, types) if part))


# --------------------------------------------------------------------------------------
# The ops
# --------------------------------------------------------------------------------------


def _layer_normalization(context: NodeContext) -> NodeEmission:
    """Standardize each row from `axis` on, then scale and shift it.

    `Scale` and `B` carry the normalized shape, but ONNX applies them by broadcasting, so
    they are addressed through strides over the data's own axes rather than by position.
    Stage one runs in the element type `stash_type` names, which is also the type ONNX gives
    the mean and inverse deviation this op can report.
    """
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    rank = len(data.shape)
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    grouping = group_axes(data.shape, tuple(range(axis, rank)))
    stash = _stash_element(context)

    operands = [_broadcast_operand(context, "scale", context.require_input(1), data)]
    formula = "($element)(centred * factor) * scale[scale_position]"
    bias = context.optional_input(2)
    if bias is not None:
        operands.append(_broadcast_operand(context, "bias", bias, data))
        formula += " + bias[bias_position]"

    statistics = _statistics(
        context, grouping, ((1, "mean_out", "mean"), (2, "inv_std_out", "factor"))
    )
    for statistic in statistics:
        if statistic.ref.elem_type != stash:
            raise CompileError(
                f"Node `{context.label}`: ONNX types `{statistic.ref.name}` by the "
                f"`stash_type` attribute, which names `{element_type_name(stash)}`, but "
                f"the graph gives it `{element_type_name(statistic.ref.elem_type)}`."
            )
    return _standardize(
        context,
        data=data,
        result=result,
        grouping=grouping,
        stash=stash,
        factor="$one / sqrt$f(variance + epsilon)",
        formula=formula,
        operands=operands,
        statistics=statistics,
        scalars=(_Scalar("epsilon", stash, context.float_attribute("epsilon")),),
    )


def _rms_normalization(context: NodeContext) -> NodeEmission:
    """Scale each row from `axis` on by the reciprocal root of its own mean square.

    ONNX's function body computes stage one in the element type `stash_type` names, casting
    the data to it and back; the reference evaluator ignores the attribute, computes in the
    data's own type, and refuses outright any value but the default. The kernel follows the
    reference — it is what both suites compare against — and refuses the other values for the
    same reason the reference's refusal gives them: nothing vouches for what they compute.
    """
    data = context.require_input(0)
    scale = context.require_input(1)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    stash = context.int_attribute("stash_type")
    if stash != TensorProto.FLOAT:
        raise CompileError(
            f"Node `{context.label}`: `stash_type` names element type "
            f"`{element_type_name(stash)}`. The ONNX reference evaluator refuses every "
            "value but the default here and takes RMSNormalization's statistics in the "
            "data's own type, so nothing vouches for what another one computes."
        )
    for operand in (scale, result):
        # ONNX's own inference refuses a model whose scale and data disagree; a graph that
        # reached here with one would have the kernel read a buffer at the wrong width, and
        # this is where that stops rather than where it corrupts memory.
        if operand.elem_type != data.elem_type:
            raise CompileError(
                f"Node `{context.label}`: RMSNormalization gives `{operand.name}` element "
                f"type `{element_type_name(operand.elem_type)}` while `{data.name}` has "
                f"`{element_type_name(data.elem_type)}`; the C compiler serves this op at "
                "one element type only."
            )

    rank = len(data.shape)
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    grouping = group_axes(data.shape, tuple(range(axis, rank)))
    offset = offset_helper(context.prefix)
    elem_type = data.elem_type
    name = _kernel_name(context, "", (elem_type,), "")
    definition = _RMS_TEMPLATE.substitute(
        name=name,
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        element_offset=(
            f"{offset.name}(index, reduced_rank, reduced_shape, reduced_strides)"
        ),
        element=c_type(elem_type),
        f=math_suffix(elem_type),
        one=scalar_literal(1, elem_type),
        zero=scalar_literal(0, elem_type),
    )
    strides = broadcast_strides(scale, data.shape, node_label=context.label)
    arguments = [
        result.expr,
        data.expr,
        scale.expr,
        *grouping.arguments,
        *_split_strides(grouping, strides),
        scalar_literal(context.float_attribute("epsilon"), elem_type),
    ]
    return NodeEmission(
        functions=(offset, CFunction(name, definition)),
        statements=(call_kernel(name, arguments),),
    )


def _instance_normalization(context: NodeContext) -> NodeEmission:
    """Standardize each channel of each instance over its spatial axes."""
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    rank = _require_channel_axis(context, data)
    grouping = group_axes(data.shape, tuple(range(2, rank)))
    elem_type = result.elem_type
    return _standardize(
        context,
        data=data,
        result=result,
        grouping=grouping,
        stash=elem_type,
        factor="sqrt$f(variance + epsilon)",
        formula="scale[scale_position] * centred / factor + bias[bias_position]",
        operands=_channel_operands(context, data, (("scale", 1), ("bias", 2))),
        scalars=(_Scalar("epsilon", elem_type, context.float_attribute("epsilon")),),
    )


def _group_normalization(context: NodeContext) -> NodeEmission:
    """Standardize each group of channels of each instance, then scale per channel.

    The data is read as though reshaped to `[N, num_groups, group_size, ...]`, which for a
    contiguous row-major buffer is a change of coordinates and nothing more.
    """
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    rank = _require_channel_axis(context, data)
    channels = _channels(data)
    groups = context.int_attribute("num_groups")
    if groups <= 0 or channels % groups:
        raise CompileError(
            f"Node `{context.label}`: GroupNormalization splits {channels} channel(s) into "
            f"`num_groups` = {groups} group(s), which does not divide them evenly."
        )
    size = channels // groups
    reshaped = (data.shape[0], groups, size, *data.shape[2:])
    grouping = group_axes(reshaped, tuple(range(2, len(reshaped))))
    # An element's channel is its group times the group's size plus its position within the
    # group, which is what these strides — over the reshaped axes — add up to.
    strides = (0, size, 1) + (0,) * (rank - 2)
    stash = _stash_element(context)
    return _standardize(
        context,
        data=data,
        result=result,
        grouping=grouping,
        stash=stash,
        factor="sqrt$f(variance + epsilon)",
        formula=(
            "($element)(centred / factor) * scale[scale_position] + bias[bias_position]"
        ),
        operands=[
            _Operand(name, _per_channel(context, index, channels), strides)
            for name, index in (("scale", 1), ("bias", 2))
        ],
        scalars=(_Scalar("epsilon", stash, context.float_attribute("epsilon")),),
    )


def _mean_variance_normalization(context: NodeContext) -> NodeEmission:
    """Standardize over the named axes, guarding the division at the standard deviation.

    ONNX defines this op as a function whose epsilon is added to the deviation rather than
    to the variance, which is what the reference evaluator and the corpus both compute.
    """
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    axes = context.attribute("axes", list(_MVN_DEFAULT_AXES))
    grouping = group_axes(data.shape, normalize_axes(context, axes, len(data.shape)))
    elem_type = result.elem_type
    return _standardize(
        context,
        data=data,
        result=result,
        grouping=grouping,
        stash=elem_type,
        factor="sqrt$f(variance) + epsilon",
        formula="centred / factor",
        scalars=(_Scalar("epsilon", elem_type, _MVN_EPSILON),),
    )


def _batch_normalization(context: NodeContext) -> NodeEmission:
    """Normalize per channel: by the statistics handed to it, or by the batch's own."""
    if context.int_attribute("training_mode"):
        return _batch_training(context)
    for index in (1, 2):
        extra = context.outputs[index] if index < len(context.outputs) else None
        if extra is not None:
            raise CompileError(
                f"Node `{context.label}`: BatchNormalization computes `{extra.name}` in "
                "training mode only, and this node runs at inference, where ONNX leaves "
                "the extra outputs undefined."
            )
    return _batch_test(context)


def _batch_test(context: NodeContext) -> NodeEmission:
    """Inference: the mean and variance are operands, so nothing is reduced."""
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    channels = _channels(data)
    scale, bias, mean, variance = (
        _per_channel(context, index, channels) for index in (1, 2, 3, 4)
    )

    stash = _widest(data.elem_type, scale.elem_type, mean.elem_type)
    name = _kernel_name(
        context, "test", (result.elem_type, stash, scale.elem_type, mean.elem_type), ""
    )
    definition = _BATCH_TEST_TEMPLATE.substitute(
        name=name,
        element=c_type(result.elem_type),
        affine=c_type(scale.elem_type),
        statistic=c_type(mean.elem_type),
        stash=c_type(stash),
        f=math_suffix(stash),
    )
    arguments = [
        result.expr,
        data.expr,
        scale.expr,
        bias.expr,
        mean.expr,
        variance.expr,
        f"{result.elem_count}u",
        f"{math.prod(data.shape[2:])}u",
        f"{channels}u",
        scalar_literal(context.float_attribute("epsilon"), stash),
    ]
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def _batch_training(context: NodeContext) -> NodeEmission:
    """Training: the batch's own statistics normalize it, and carry the running ones on.

    The running statistics are read and written per channel, which is exactly one group
    here, so the operands they blend are only passed when the node asks for them.
    """
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    rank = len(data.shape)
    channels = _channels(data)
    grouping = group_axes(data.shape, tuple(axis for axis in range(rank) if axis != 1))
    strides = _channel_strides(rank)

    operands = _channel_operands(context, data, (("scale", 1), ("bias", 2)))
    statistics = _statistics(
        context,
        grouping,
        (
            (
                1,
                "running_mean",
                "($stash)input_mean[input_mean_base] * momentum"
                " + mean * ($one - momentum)",
            ),
            (
                2,
                "running_var",
                "($stash)input_var[input_var_base] * momentum"
                " + variance * ($one - momentum)",
            ),
        ),
    )
    blended = {1: ("input_mean", 3), 2: ("input_var", 4)}
    for statistic in statistics:
        name, index = blended[statistic.index]
        operands.append(
            _Operand(
                name, _per_channel(context, index, channels), strides, per_group=True
            )
        )

    stash = _widest(
        data.elem_type,
        context.require_input(1).elem_type,
        context.require_input(3).elem_type,
    )
    scalars = [_Scalar("epsilon", stash, context.float_attribute("epsilon"))]
    if statistics:
        # `momentum` blends the running statistics and is read nowhere else, so a node
        # that reports neither takes an argument its kernel never touches — which the
        # artifact's own `-Werror=unused-parameter` build refuses.
        scalars.append(_Scalar("momentum", stash, context.float_attribute("momentum")))
    return _standardize(
        context,
        data=data,
        result=result,
        grouping=grouping,
        stash=stash,
        factor="sqrt$f(variance + epsilon)",
        formula=(
            "($stash)scale[scale_position] * centred / factor"
            " + ($stash)bias[bias_position]"
        ),
        operands=operands,
        statistics=statistics,
        scalars=scalars,
    )


def _lp_normalization(context: NodeContext) -> NodeEmission:
    """Divide each row along one axis by its own Lp norm.

    The norm sums absolute values, which is what ONNX defines and what the corpus's own
    expectations compute; the reference evaluator raises the elements to the power `p`
    without taking their absolute value, so for `p` = 1 it is an oracle on non-negative
    operands only.
    """
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    order = context.int_attribute("p")
    if order not in _SUPPORTED_ORDERS:
        raise CompileError(
            f"Node `{context.label}`: ONNX defines LpNormalization for `p` in "
            f"{list(_SUPPORTED_ORDERS)}, but this node asks for {order}."
        )
    axis = normalize_axis(context, context.int_attribute("axis"), len(data.shape))
    grouping = group_axes(data.shape, (axis,))

    offset = offset_helper(context.prefix)
    elem_type = result.elem_type
    suffix = math_suffix(elem_type)
    name = _kernel_name(context, f"l{order}", (elem_type,), "")
    definition = _LP_TEMPLATE.substitute(
        name=name,
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        element_offset=(
            f"{offset.name}(index, reduced_rank, reduced_shape, reduced_strides)"
        ),
        element=c_type(elem_type),
        zero=scalar_literal(0, elem_type),
        term=f"fabs{suffix}(x)" if order == 1 else "x * x",
        norm="total" if order == 1 else f"sqrt{suffix}(total)",
    )
    return NodeEmission(
        functions=(offset, CFunction(name, definition)),
        statements=(call_kernel(name, [result.expr, data.expr, *grouping.arguments]),),
    )


def _local_response_normalization(context: NodeContext) -> NodeEmission:
    """Divide each element by a power of the squared sum of its channel neighbourhood."""
    data = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, data, result)
    _require_channel_axis(context, data)
    size = context.int_attribute("size")
    if size <= 0:
        raise CompileError(
            f"Node `{context.label}`: LRN sums over a window of `size` channels, which "
            f"this node gives as {size}."
        )
    elem_type = result.elem_type
    name = _kernel_name(context, "", (elem_type,), "")
    definition = _LRN_TEMPLATE.substitute(
        name=name,
        element=c_type(elem_type),
        zero=scalar_literal(0, elem_type),
        f=math_suffix(elem_type),
    )
    arguments = [
        result.expr,
        data.expr,
        f"{data.shape[0]}u",
        f"{_channels(data)}u",
        f"{math.prod(data.shape[2:])}u",
        # The window reaches `floor((size - 1) / 2)` channels back and
        # `ceil((size - 1) / 2)` channels on, both clamped to the tensor.
        f"{(size - 1) // 2}u",
        f"{size // 2}u",
        scalar_literal(context.float_attribute("bias"), elem_type),
        scalar_literal(context.float_attribute("alpha") / size, elem_type),
        scalar_literal(context.float_attribute("beta"), elem_type),
    ]
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


# --------------------------------------------------------------------------------------
# Operands, types and shapes the family shares
# --------------------------------------------------------------------------------------


def _stash_element(context: NodeContext) -> int:
    """The element type stage one computes in, as the node's `stash_type` names it."""
    stash = context.int_attribute("stash_type")
    if stash not in FLOAT_TYPES:
        raise CompileError(
            f"Node `{context.label}`: `stash_type` names element type "
            f"`{element_type_name(stash)}`, which the C compiler cannot take statistics "
            "in; only FLOAT and DOUBLE are supported."
        )
    return stash


def _statistics(
    context: NodeContext,
    grouping: Grouping,
    entries: Sequence[tuple[int, str, str]],
) -> list[_Statistic]:
    """The per-group outputs this node actually asks for.

    An output ONNX declares optional may be left out entirely or named as the empty string,
    and one the node omits is not computed at all.
    """
    statistics = []
    for index, name, value in entries:
        ref = context.outputs[index] if index < len(context.outputs) else None
        if ref is None:
            continue
        if ref.elem_count != grouping.group_count:
            # The groups are counted from the operand's shape and the axes the node
            # normalizes over, while the buffer is sized from the shape ONNX inferred; a
            # disagreement is a compiler bug, and this is where it stops rather than where
            # it corrupts memory.
            raise CompileError(
                f"Node `{context.label}`: normalizing leaves {grouping.group_count} "
                f"group(s), but its output `{ref.name}` holds {ref.elem_count} element(s)."
            )
        statistics.append(_Statistic(index, name, ref, value))
    return statistics


def _require_channel_axis(context: NodeContext, data: TensorRef) -> int:
    """The data's rank, refusing one that has no channel axis to normalize per."""
    rank = len(data.shape)
    if rank < 2:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{data.name}` as "
            "instances by channels by spatial axes, so it needs a rank of at least 2; this "
            f"one has shape {list(data.shape)}."
        )
    return rank


def _channels(data: TensorRef) -> int:
    """The extent of the channel axis; a tensor of rank 1 is a single channel to ONNX."""
    return data.shape[1] if len(data.shape) > 1 else 1


def _channel_strides(rank: int) -> tuple[int, ...]:
    """Strides addressing a per-channel operand from the data's coordinates."""
    return tuple(1 if axis == 1 else 0 for axis in range(rank))


def _per_channel(context: NodeContext, index: int, channels: int) -> TensorRef:
    """An operand ONNX gives one value per channel, checked against that shape."""
    operand = context.require_input(index)
    if operand.shape != (channels,):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes `{operand.name}` per "
            f"channel, so ONNX gives it shape [{channels}]; this model gives it "
            f"{list(operand.shape)}."
        )
    return operand


def _channel_operands(
    context: NodeContext, data: TensorRef, entries: Sequence[tuple[str, int]]
) -> list[_Operand]:
    channels = _channels(data)
    strides = _channel_strides(len(data.shape))
    return [
        _Operand(name, _per_channel(context, index, channels), strides)
        for name, index in entries
    ]


def _broadcast_operand(
    context: NodeContext, name: str, operand: TensorRef, data: TensorRef
) -> _Operand:
    return _Operand(
        name, operand, broadcast_strides(operand, data.shape, node_label=context.label)
    )


def _widest(*elem_types: int) -> int:
    """The element type numpy's promotion would compute these operands in."""
    return TensorProto.DOUBLE if TensorProto.DOUBLE in elem_types else TensorProto.FLOAT


register_kernel("", "BatchNormalization", _BATCH_VERSIONS, _batch_normalization)
register_kernel("", "LayerNormalization", _LAYER_VERSIONS, _layer_normalization)
register_kernel("", "RMSNormalization", _RMS_VERSIONS, _rms_normalization)
register_kernel(
    "", "InstanceNormalization", _INSTANCE_VERSIONS, _instance_normalization
)
register_kernel("", "GroupNormalization", _GROUP_VERSIONS, _group_normalization)
register_kernel("", "LpNormalization", _LP_VERSIONS, _lp_normalization)
register_kernel(
    "", "MeanVarianceNormalization", _MVN_VERSIONS, _mean_variance_normalization
)
register_kernel("", "LRN", _LRN_VERSIONS, _local_response_normalization)
