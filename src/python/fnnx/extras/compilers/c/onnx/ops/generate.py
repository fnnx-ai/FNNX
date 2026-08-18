"""The ops whose result is a function of its own coordinates.

OneHot, EyeLike and Trilu each decide an element from where it sits: whether the coordinate
along one axis is the index an operand names, whether two coordinates are a fixed distance
apart, whether one is above the other. ReverseSequence is the same idea one step on — the
coordinate along the time axis is mapped to another coordinate, per row, by a length the
caller supplies. None of them needs an index into a buffer, only the loop that produces the
coordinates, so each is one kernel over the result's own shape.

`ConstantOfShape` and `Range` belong to the same family and need no kernel at all. Both read
their entire result — its shape included — out of their operands, so the compiler accepts
them only where the graph fixes those operands, and where it does the folding pass resolves
them into an initializer before dispatch is ever reached.
"""

from __future__ import annotations

import math
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import FLOAT_TYPES, UNSIGNED_TYPES, c_type
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    kernel_name,
    normalize_axis,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import expand, math_suffix

# OneHot-11 only clarified how a negative axis is counted; EyeLike-22 and Trilu's single
# revision widened nothing this compiler compiles differently. ReverseSequence has one
# revision of its own.
_ONE_HOT_VERSIONS = (9, 11)
_EYE_LIKE_VERSIONS = (9, 22)
_TRILU_VERSIONS = (14,)
_REVERSE_SEQUENCE_VERSIONS = (10,)

_ONE_HOT_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $index* indices,
    const $element* values,
    size_t outer,
    size_t depth,
    size_t inner)
{
    size_t before, position, after;
    for (before = 0; before < outer; ++before) {
        for (position = 0; position < depth; ++position) {
            for (after = 0; after < inner; ++after) {
                const $index folded =
                    $fold(indices[before * inner + after], ($index)depth);
                out[(before * depth + position) * inner + after] =
                    (folded == ($index)position) ? values[1] : values[0];
            }
        }
    }
}""")

# ONNX folds an index into the range the depth allows, which is what the reference evaluator
# computes with numpy's `mod`: the result takes the sign of the depth, where C's `%` takes
# the sign of the index. An index of a floating-point type folds the same way and then
# matches no position unless it is a whole number, exactly as comparing the two would.
_FOLD_TEMPLATE = Template("""\
static $element $name($element value, $element depth)
{
    const $element folded = $modulo;
    return $adjust;
}""")

_EYE_LIKE_TEMPLATE = Template("""\
static void $name($element* out, size_t rows, size_t columns, int64_t offset)
{
    size_t row, column;
    for (row = 0; row < rows; ++row) {
        for (column = 0; column < columns; ++column) {
            out[row * columns + column] =
                ((int64_t)column - (int64_t)row == offset) ? $one : $zero;
        }
    }
}""")

_TRILU_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t batch,
    size_t rows,
    size_t columns,
    int64_t offset)
{
    size_t matrix, row, column;
    for (matrix = 0; matrix < batch; ++matrix) {
        for (row = 0; row < rows; ++row) {
            for (column = 0; column < columns; ++column) {
                const size_t position = (matrix * rows + row) * columns + column;
                out[position] =
                    ((int64_t)column - (int64_t)row $keep offset) ? in[position] : $zero;
            }
        }
    }
}""")

_REVERSE_SEQUENCE_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const int64_t* lengths,
    size_t batch,
    size_t time,
    size_t inner,
    size_t batch_stride,
    size_t time_stride)
{
    size_t row, step;
    for (row = 0; row < batch; ++row) {
        const int64_t length = lengths[row];
        if (length < 1 || (size_t)length > time) {
            return 1;
        }
        for (step = 0; step < time; ++step) {
            const size_t source =
                (step < (size_t)length) ? (size_t)length - 1 - step : step;
            memcpy(
                out + row * batch_stride + step * time_stride,
                in + row * batch_stride + source * time_stride,
                inner * sizeof(*out));
        }
    }
    return 0;
}""")


def _one_hot(context: NodeContext) -> NodeEmission:
    """OneHot: the depth axis set at the position each index names, elsewhere the off value.

    `depth` decides the extent of that axis, so it has to be fixed at compile time; the two
    values it selects between decide nothing about any shape and are read at run time.
    """
    indices = context.require_input(0)
    values = context.require_input(2)
    result = context.require_output(0)
    rank = len(indices.shape)
    axis = normalize_axis(context, context.int_attribute("axis"), rank + 1)
    depth = _depth(context)
    verify_shape(context, result, (*indices.shape[:axis], depth, *indices.shape[axis:]))
    if values.elem_count != 2:
        raise CompileError(
            f"Node `{context.label}`: `OneHot` takes its off and on values from "
            f"`{values.name}`, which holds {values.elem_count}; ONNX defines it as two."
        )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    fold = _fold(context, indices.elem_type)
    name = kernel_name(context, c_type(result.elem_type), c_type(indices.elem_type))
    return NodeEmission(
        functions=(
            fold,
            CFunction(
                name,
                _ONE_HOT_TEMPLATE.substitute(
                    name=name,
                    element=c_type(result.elem_type),
                    index=c_type(indices.elem_type),
                    fold=fold.name,
                ),
            ),
        ),
        statements=(
            call_kernel(
                name,
                [
                    result.expr,
                    indices.expr,
                    values.expr,
                    f"{math.prod(indices.shape[:axis])}u",
                    f"{depth}u",
                    f"{math.prod(indices.shape[axis:])}u",
                ],
            ),
        ),
    )


def _depth(context: NodeContext) -> int:
    operand = context.require_input(1)
    fixed = context.constant_input(1)
    if fixed is None or fixed.size != 1:
        raise CompileError(
            f"Node `{context.label}`: `OneHot` takes its depth from `{operand.name}`, "
            f"which holds {'no single value' if fixed is not None else 'no value'} known "
            "at compile time; the shape of the result then depends on input data, which "
            "the C compiler cannot compile."
        )
    return int(fixed.reshape(-1)[0])


def _fold(context: NodeContext, elem_type: int) -> CFunction:
    """An index folded into `[0, depth)`, at whatever type the indices are given in."""
    element = c_type(elem_type)
    modulo = (
        f"fmod{math_suffix(elem_type)}(value, depth)"
        if elem_type in FLOAT_TYPES
        else "value % depth"
    )
    # An unsigned index is already inside the range, and comparing one against zero is a
    # diagnostic the artifact's `-Werror` build turns into a failure.
    adjust = (
        "folded"
        if elem_type in UNSIGNED_TYPES
        else "(folded < 0) ? folded + depth : folded"
    )
    name = f"{context.prefix}_onehot_fold_{element}"
    return CFunction(
        name,
        _FOLD_TEMPLATE.substitute(
            name=name, element=element, modulo=modulo, adjust=adjust
        ),
    )


def _eye_like(context: NodeContext) -> NodeEmission:
    """EyeLike: ones on one diagonal of a matrix the operand's shape describes.

    Nothing of the operand but its shape is read, which is why the operand itself does not
    appear in the emitted call at all.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    if len(source.shape) != 2:
        raise CompileError(
            f"Node `{context.label}`: `EyeLike` takes the shape of `{source.name}`, which "
            f"is {list(source.shape)}; ONNX defines the op over 2-D tensors."
        )
    verify_shape(context, result, source.shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    rows, columns = result.shape
    name = kernel_name(context, c_type(result.elem_type))
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _EYE_LIKE_TEMPLATE.substitute(
                    name=name,
                    element=c_type(result.elem_type),
                    one=expand("$one", result.elem_type),
                    zero=expand("$zero", result.elem_type),
                ),
            ),
        ),
        statements=(
            call_kernel(
                name,
                [
                    result.expr,
                    f"{rows}u",
                    f"{columns}u",
                    str(context.int_attribute("k")),
                ],
            ),
        ),
    )


def _trilu(context: NodeContext) -> NodeEmission:
    """Trilu: one triangle of each matrix kept, the other zeroed.

    The diagonal to cut along is an operand, and one that decides nothing about any shape,
    so a graph computing it at run time still compiles: the kernel reads it as a value.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    if len(source.shape) < 2:
        raise CompileError(
            f"Node `{context.label}`: `Trilu` cuts each matrix of `{source.name}`, which "
            f"has shape {list(source.shape)}; ONNX defines the op from rank 2 up."
        )
    verify_shape(context, result, source.shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    upper = bool(context.int_attribute("upper"))
    rows, columns = source.shape[-2:]
    name = kernel_name(context, "upper" if upper else "lower", c_type(result.elem_type))
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _TRILU_TEMPLATE.substitute(
                    name=name,
                    element=c_type(result.elem_type),
                    keep=">=" if upper else "<=",
                    zero=expand("$zero", result.elem_type),
                ),
            ),
        ),
        statements=(
            call_kernel(
                name,
                [
                    result.expr,
                    source.expr,
                    f"{math.prod(source.shape[:-2])}u",
                    f"{rows}u",
                    f"{columns}u",
                    _offset(context),
                ],
            ),
        ),
    )


def _offset(context: NodeContext) -> str:
    """Which diagonal Trilu cuts along, as a C expression of type `int64_t`."""
    operand = context.optional_input(1)
    if operand is None:
        return "0"
    if operand.elem_count != 1:
        raise CompileError(
            f"Node `{context.label}`: `Trilu` takes the diagonal from `{operand.name}`, "
            f"which holds {operand.elem_count} values; ONNX defines it as a single one."
        )
    return f"{operand.expr}[0]"


def _reverse_sequence(context: NodeContext) -> NodeEmission:
    """ReverseSequence: the first `sequence_lens[b]` steps of each batch reversed.

    The lengths are values, not shapes, so they are read at run time — and validated there:
    ONNX defines them only within the time axis, and a longer one would read past the buffer.
    """
    source = context.require_input(0)
    lengths = context.require_input(1)
    result = context.require_output(0)
    rank = len(source.shape)
    if rank < 2:
        raise CompileError(
            f"Node `{context.label}`: `ReverseSequence` reverses along the time axis of "
            f"`{source.name}`, which has shape {list(source.shape)}; ONNX defines the op "
            "from rank 2 up."
        )
    batch_axis = context.int_attribute("batch_axis")
    time_axis = context.int_attribute("time_axis")
    if {batch_axis, time_axis} != {0, 1}:
        raise CompileError(
            f"Node `{context.label}`: `ReverseSequence` runs over batch axis {batch_axis} "
            f"and time axis {time_axis}; ONNX defines the two as 0 and 1 in either order."
        )
    verify_shape(context, result, source.shape)
    batch = source.shape[batch_axis]
    if lengths.shape != (batch,):
        raise CompileError(
            f"Node `{context.label}`: `ReverseSequence` takes its lengths from "
            f"`{lengths.name}` of shape {list(lengths.shape)}; ONNX defines one per batch, "
            f"of which axis {batch_axis} of `{source.name}` has {batch}."
        )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    inner = math.prod(source.shape[2:])
    strides = (source.shape[1] * inner, inner)
    name = kernel_name(context, c_type(result.elem_type))
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _REVERSE_SEQUENCE_TEMPLATE.substitute(
                    name=name, element=c_type(result.elem_type)
                ),
            ),
        ),
        statements=(
            checked_call(
                context,
                name,
                [
                    result.expr,
                    source.expr,
                    lengths.expr,
                    f"{batch}u",
                    f"{source.shape[time_axis]}u",
                    f"{inner}u",
                    f"{strides[batch_axis]}u",
                    f"{strides[time_axis]}u",
                ],
            ),
        ),
    )


register_kernel("", "OneHot", _ONE_HOT_VERSIONS, _one_hot)
register_kernel("", "EyeLike", _EYE_LIKE_VERSIONS, _eye_like)
register_kernel("", "Trilu", _TRILU_VERSIONS, _trilu)
register_kernel("", "ReverseSequence", _REVERSE_SEQUENCE_VERSIONS, _reverse_sequence)
