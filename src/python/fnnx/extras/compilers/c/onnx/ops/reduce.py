"""The reductions: the Reduce* family over any set of axes, and ArgMax/ArgMin.

Every one of them folds the elements of a group into a single value, so they share one loop
nest — the group loop and the element loop the named axes describe — and differ only in what
the fold starts from, what it does per element, and what it makes of the accumulator. ONNX
moved `axes` from an attribute to an input partway through the family's history; both
conventions reach the same emitter once the axes are resolved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from string import Template

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    UNSIGNED_TYPES,
    c_type,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    GROUP_PARAMETERS,
    call_kernel,
    group_axes,
    kernel_name,
    normalize_axes,
    normalize_axis,
    offset_helper,
    verify_group_count,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import combiner, expand, math_suffix

# The revisions that take `axes` as an attribute, and the ones that take it as an input.
# ReduceSum moved at 13 and the rest of the family at 18; the revisions listed alongside each
# move only widened type constraints — 12 added the int8 families to ReduceMax/ReduceMin and
# 20 the boolean ones — which leaves the emitted code unchanged.
_ATTRIBUTE_VERSIONS = (1, 11, 13)
_INPUT_VERSIONS = (18,)
_SUM_ATTRIBUTE_VERSIONS = (1, 11)
_SUM_INPUT_VERSIONS = (13,)
_EXTREMUM_ATTRIBUTE_VERSIONS = (1, 11, 12, 13)
_EXTREMUM_INPUT_VERSIONS = (18, 20)

# ArgMax/ArgMin-12 added `select_last_index`, whose default is what the earlier revisions
# compute; 11 allowed a negative axis and 13 widened the type constraints.
_ARG_VERSIONS = (1, 11, 12, 13)

_REDUCE_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element total = $identity;
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)];
            total = ($element)($combine);
        }
        out[group] = $result;
    }
}""")

# LogSumExp is the one reduction that reads its group twice: the largest element is
# subtracted from every exponent so that no term overflows, and added back afterwards, which
# is how the reference evaluator computes it too.
_LOG_SUM_EXP_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element largest = $identity;
        $accumulator total = $accumulator_zero;
        for (index = 0; index < group_size; ++index) {
            largest = ($element)$maximum(largest, in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)]);
        }
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)];
            total += exp$suffix(($accumulator)(x - largest));
        }
        out[group] = ($element)(log$suffix(total) + ($accumulator)largest);
    }
}""")

_ARG_TEMPLATE = Template("""\
static void $name(
    int64_t* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element best = $zero;
        int64_t chosen = 0;
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)];
            if (index == 0 || ($better)) {
                best = x;
                chosen = (int64_t)index;
            }
        }
        out[group] = chosen;
    }
}""")


@dataclass(frozen=True)
class Fold:
    """How one reduction folds a group, as C over `total`, `x` and `group_size`.

    `identity` is where the fold starts, which is also what a group with no elements at all
    yields — the value the reference evaluator fills such a reduction with. All three
    expressions are `$`-templated over the element type.
    """

    identity: str
    combine: str
    result: str = "total"
    helpers: tuple[CFunction, ...] = ()


# What a kernel builder hands back: the kernel to call, and the functions it calls in turn.
Kernel = tuple[CFunction, tuple[CFunction, ...]]
KernelBuilder = Callable[[NodeContext], Kernel]
Recipe = Callable[[NodeContext], Fold]


def _sum_fold(context: NodeContext) -> Fold:
    return Fold(identity="$zero", combine="total + x")


def _product_fold(context: NodeContext) -> Fold:
    return Fold(identity="$one", combine="total * x")


def _mean_fold(context: NodeContext) -> Fold:
    """The sum over the group's size, taken in the element type for the float families.

    numpy divides an integer sum in double and casts back, truncating toward zero; a group
    with no elements makes that a 0/0 whose cast numpy leaves undefined, so the integer form
    yields zero there rather than emitting a conversion C does not define either.
    """
    if context.require_output(0).elem_type in FLOAT_TYPES:
        return Fold(
            identity="$zero",
            combine="total + x",
            result="total / ($element)group_size",
        )
    return Fold(
        identity="$zero",
        combine="total + x",
        result="($element)(group_size ? (double)total / (double)group_size : 0.0)",
    )


def _absolute_sum_fold(context: NodeContext) -> Fold:
    return Fold(identity="$zero", combine=f"total + {_absolute(context)}")


def _square_sum_fold(context: NodeContext) -> Fold:
    return Fold(identity="$zero", combine="total + x * x")


def _euclidean_fold(context: NodeContext) -> Fold:
    return Fold(
        identity="$zero", combine="total + x * x", result=_libm(context, "sqrt")
    )


def _log_sum_fold(context: NodeContext) -> Fold:
    return Fold(identity="$zero", combine="total + x", result=_libm(context, "log"))


def _extremum_fold(context: NodeContext, *, largest: bool) -> Fold:
    elem_type = context.require_output(0).elem_type
    helper = combiner(context, elem_type, largest=largest)
    return Fold(
        identity=extremum_identity(elem_type, largest=largest),
        combine=f"{helper.name}(total, x)",
        helpers=(helper,),
    )


def _absolute(context: NodeContext) -> str:
    """`|x|`, which the integer families take off a comparison rather than from libm."""
    elem_type = context.require_output(0).elem_type
    if elem_type in FLOAT_TYPES:
        return "fabs$f(x)"
    if elem_type in UNSIGNED_TYPES:
        return "x"
    return "((x < $zero) ? ($element)-x : x)"


def _libm(context: NodeContext, function: str) -> str:
    """A libm call on the accumulator, taken in double for the integer families.

    numpy evaluates these in floating point whatever the tensor holds and casts the result
    back, so an integer reduction rounds once, on the way out.
    """
    elem_type = context.require_output(0).elem_type
    if elem_type in FLOAT_TYPES:
        return f"{function}{math_suffix(elem_type)}(total)"
    return f"($element){function}((double)total)"


def extremum_identity(elem_type: int, *, largest: bool) -> str:
    """The neutral element of a max or min fold at this element type.

    Shared with the poolings, whose window is a max fold over part of a tensor.
    """
    if elem_type in FLOAT_TYPES:
        return "-INFINITY" if largest else "INFINITY"
    if elem_type == TensorProto.BOOL:
        return "0" if largest else "1"
    info = np.iinfo(numpy_dtype_name(elem_type))
    return scalar_literal(info.min if largest else info.max, elem_type)


def _fold_kernel(context: NodeContext, *, recipe: Recipe) -> Kernel:
    result = context.require_output(0)
    fold = recipe(context)
    offset = offset_helper(context.prefix)
    name = kernel_name(context, numpy_dtype_name(result.elem_type))
    definition = _REDUCE_TEMPLATE.substitute(
        name=name,
        element=c_type(result.elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        identity=expand(fold.identity, result.elem_type),
        combine=expand(fold.combine, result.elem_type),
        result=expand(fold.result, result.elem_type),
    )
    return CFunction(name, definition), (offset, *fold.helpers)


def _log_sum_exp_kernel(context: NodeContext) -> Kernel:
    elem_type = context.require_output(0).elem_type
    floating = elem_type in FLOAT_TYPES
    accumulator = c_type(elem_type) if floating else "double"
    largest = combiner(context, elem_type, largest=True)
    offset = offset_helper(context.prefix)
    name = kernel_name(context, numpy_dtype_name(elem_type))
    definition = _LOG_SUM_EXP_TEMPLATE.substitute(
        name=name,
        element=c_type(elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        identity=extremum_identity(elem_type, largest=True),
        maximum=largest.name,
        accumulator=accumulator,
        accumulator_zero=scalar_literal(
            0, elem_type if floating else TensorProto.DOUBLE
        ),
        suffix=math_suffix(elem_type) if floating else "",
    )
    return CFunction(name, definition), (offset, largest)


def _from_attribute(context: NodeContext, *, build: KernelBuilder) -> NodeEmission:
    """A Reduce* revision whose axes are an attribute; without them it reduces every axis."""
    axes = context.attribute("axes", None)
    return _reduce(context, build, tuple(axes) if axes else None)


def _from_input(context: NodeContext, *, build: KernelBuilder) -> NodeEmission:
    """A Reduce* revision whose axes are an input, which it may also name none through.

    Without axes the op reduces every one of them, or — under `noop_with_empty_axes` — none.
    Reducing none is not the identity: every element becomes a group of its own and still
    goes through the fold, so ReduceL1 over no axes is an absolute value.
    """
    axes = _axes_operand(context)
    if axes is None and context.int_attribute("noop_with_empty_axes"):
        axes = ()
    return _reduce(context, build, axes)


def _axes_operand(context: NodeContext) -> tuple[int, ...] | None:
    """The axes the node's second operand names, None when it names none at all.

    An operand with no elements names no axes whatever it holds at run time, which is what
    makes the corpus's `default_axes` models compilable; one carrying values the graph does
    not fix is rejected by the frontend before any kernel is reached.
    """
    operand = context.optional_input(1)
    if operand is None or operand.elem_count == 0:
        return None
    values = context.constant_input(1)
    if values is None:
        raise CompileError(
            f"Node `{context.label}`: the axes of `{context.node.op_type}` come from "
            f"`{operand.name}`, which is not known at compile time; the shape of the "
            "result then depends on input data, which the C compiler cannot compile."
        )
    return tuple(int(axis) for axis in values.reshape(-1))


def _reduce(
    context: NodeContext, build: KernelBuilder, axes: Sequence[int] | None
) -> NodeEmission:
    """Emit the fold over `axes`, where None stands for every axis and `()` for none."""
    source = context.require_input(0)
    result = context.require_output(0)
    rank = len(source.shape)
    selected = (
        tuple(range(rank)) if axes is None else normalize_axes(context, axes, rank)
    )
    grouping = group_axes(source.shape, selected)
    verify_group_count(context, grouping, result)

    kernel, helpers = build(context)
    return NodeEmission(
        functions=(*helpers, kernel),
        statements=(
            call_kernel(kernel.name, [result.expr, source.expr, *grouping.arguments]),
        ),
    )


def _arg_extremum(context: NodeContext, *, largest: bool) -> NodeEmission:
    """ArgMax or ArgMin: the index of a group's extreme element, as numpy chooses it.

    A NaN is the extreme of any group it appears in, and the first one there wins, since
    nothing that follows can better it; `select_last_index` reverses which of several equal
    extremes is reported, exactly as the reference evaluator's own flip does.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    last = bool(context.attribute("select_last_index", 0))
    axis = normalize_axis(context, context.int_attribute("axis"), len(source.shape))
    grouping = group_axes(source.shape, (axis,))
    verify_group_count(context, grouping, result)

    offset = offset_helper(context.prefix)
    name = kernel_name(
        context, "last" if last else "first", numpy_dtype_name(source.elem_type)
    )
    definition = _ARG_TEMPLATE.substitute(
        name=name,
        element=c_type(source.elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        zero=scalar_literal(0, source.elem_type),
        better=extremum_test(source.elem_type, largest=largest, last=last),
    )
    return NodeEmission(
        functions=(offset, CFunction(name, definition)),
        statements=(
            call_kernel(name, [result.expr, source.expr, *grouping.arguments]),
        ),
    )


def extremum_test(elem_type: int, *, largest: bool, last: bool) -> str:
    """Whether `x` replaces `best`, under numpy's NaN-aware ordering.

    Shared with Hardmax, which is an ArgMax that writes a one-hot group rather than an index.
    """
    comparison = (">" if largest else "<") + ("=" if last else "")
    if elem_type not in FLOAT_TYPES:
        return f"x {comparison} best"
    if last:
        # A NaN betters anything, itself included, so the last of them is what is reported.
        return f"isnan(x) || (!isnan(best) && x {comparison} best)"
    return f"!isnan(best) && (isnan(x) || x {comparison} best)"


def _register_reduction(
    op_type: str,
    build: KernelBuilder,
    attribute_versions: tuple[int, ...],
    input_versions: tuple[int, ...],
) -> None:
    """Both axes conventions of one reduction, each at the revisions that take it."""
    register_kernel(
        "", op_type, attribute_versions, partial(_from_attribute, build=build)
    )
    register_kernel("", op_type, input_versions, partial(_from_input, build=build))


# The family: how each op folds a group, and the revisions of both axes conventions.
_REDUCTIONS: tuple[tuple[str, Recipe, tuple[int, ...], tuple[int, ...]], ...] = (
    ("ReduceSum", _sum_fold, _SUM_ATTRIBUTE_VERSIONS, _SUM_INPUT_VERSIONS),
    ("ReduceMean", _mean_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    ("ReduceProd", _product_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    ("ReduceL1", _absolute_sum_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    ("ReduceL2", _euclidean_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    ("ReduceLogSum", _log_sum_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    ("ReduceSumSquare", _square_sum_fold, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS),
    (
        "ReduceMax",
        partial(_extremum_fold, largest=True),
        _EXTREMUM_ATTRIBUTE_VERSIONS,
        _EXTREMUM_INPUT_VERSIONS,
    ),
    (
        "ReduceMin",
        partial(_extremum_fold, largest=False),
        _EXTREMUM_ATTRIBUTE_VERSIONS,
        _EXTREMUM_INPUT_VERSIONS,
    ),
)

for _op_type, _recipe, _attribute_versions, _input_versions in _REDUCTIONS:
    _register_reduction(
        _op_type,
        partial(_fold_kernel, recipe=_recipe),
        _attribute_versions,
        _input_versions,
    )
_register_reduction(
    "ReduceLogSumExp", _log_sum_exp_kernel, _ATTRIBUTE_VERSIONS, _INPUT_VERSIONS
)
register_kernel("", "ArgMax", _ARG_VERSIONS, partial(_arg_extremum, largest=True))
register_kernel("", "ArgMin", _ARG_VERSIONS, partial(_arg_extremum, largest=False))
