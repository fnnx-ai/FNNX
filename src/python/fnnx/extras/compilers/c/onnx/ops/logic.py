"""The ops whose result is a decision: comparisons, logic, bit manipulation, selection."""

from __future__ import annotations

from functools import partial

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_size
from fnnx.extras.compilers.c.onnx.kernels import (
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import elementwise, expand, pointwise

# Predicates over two operands, with the schema revisions each expression covers. Revision 1
# of the older ones carried the legacy broadcast attributes and is left unregistered; the
# revisions listed after the first only widened the type constraints. And, Or and Xor take
# boolean operands, whose emitted bytes are 0 or 1, so `!=` is exclusive-or on them.
_BINARY_PREDICATES: dict[str, tuple[tuple[int, ...], str]] = {
    "And": ((7,), "x0 && x1"),
    "Equal": ((7, 11, 13, 19), "x0 == x1"),
    "Greater": ((7, 9, 13), "x0 > x1"),
    "GreaterOrEqual": ((12, 16), "x0 >= x1"),
    "Less": ((7, 9, 13), "x0 < x1"),
    "LessOrEqual": ((12, 16), "x0 <= x1"),
    "Or": ((7,), "x0 || x1"),
    "Xor": ((7,), "x0 != x1"),
}

_NOT_VERSIONS = (1,)
_BITWISE_VERSIONS = (18,)
_BITWISE_OPERATORS = {"BitwiseAnd": "&", "BitwiseOr": "|", "BitwiseXor": "^"}
_BIT_SHIFT_VERSIONS = (11,)
# IsInf-10 and IsNaN-9/13 predate the float8 types the later revisions accept; neither op's
# behaviour on the compilable types has changed.
_IS_INF_VERSIONS = (10, 20)
_IS_NAN_VERSIONS = (9, 13, 20)
# Where-9 already broadcasts all three operands; 16 only widened the type constraints.
_WHERE_VERSIONS = (9, 16)

_SHIFT_OPERATORS = {"LEFT": "<<", "RIGHT": ">>"}


def _binary(context: NodeContext, *, expression: str) -> NodeEmission:
    """A two-operand kernel computing `expression` over `x0` and `x1`.

    The result is cast to the op's own output type: boolean for the predicates, the
    operands' type for the bitwise family, both of which `$element` resolves to.
    """
    result = context.require_output(0)
    return elementwise(
        context,
        expression=expand(f"($element)({expression})", result.elem_type),
        operands=(context.require_input(0), context.require_input(1)),
        result=result,
    )


def _bit_shift(context: NodeContext) -> NodeEmission:
    """BitShift, whose operands are unsigned, so both directions shift in zeros.

    A shift by the operand's own width or more is undefined in C and unstated by ONNX; numpy
    — the spec's executable form, and this compiler's oracle — yields zero, which the guard
    reproduces rather than leaving to whatever the target's shift instruction does.
    """
    result = context.require_output(0)
    direction = _shift_direction(context)
    width = 8 * element_size(result.elem_type)
    # The narrow types promote to `int`, where a left shift can overflow into the sign bit;
    # shifting in the widest unsigned type instead keeps every intermediate defined.
    promoted = c_type(TensorProto.UINT64 if width == 64 else TensorProto.UINT32)
    shift = f"(({promoted})x0 {_SHIFT_OPERATORS[direction]} x1)"
    return elementwise(
        context,
        expression=expand(
            f"($element)(x1 >= {width}u ? 0u : {shift})", result.elem_type
        ),
        operands=(context.require_input(0), context.require_input(1)),
        result=result,
        variant=f"_{direction.lower()}",
    )


def _shift_direction(context: NodeContext) -> str:
    value = context.attribute("direction", b"")
    direction = value.decode() if isinstance(value, bytes) else str(value)
    if direction not in _SHIFT_OPERATORS:
        raise CompileError(
            f"Node `{context.label}`: BitShift's `direction` attribute is "
            f"`{direction}`, but ONNX defines only "
            f"{' and '.join(f'`{name}`' for name in _SHIFT_OPERATORS)}."
        )
    return direction


def _is_inf(context: NodeContext) -> NodeEmission:
    """IsInf, whose two attributes select which of the infinities count as one."""
    result = context.require_output(0)
    positive = bool(context.int_attribute("detect_positive"))
    negative = bool(context.int_attribute("detect_negative"))
    if not positive and not negative:
        # No operand: nothing is detected, so reading one would leave the kernel with an
        # unused local, which the artifact's `-Werror` build contract does not allow.
        return elementwise(
            context,
            expression=expand("$zero", result.elem_type),
            operands=(),
            result=result,
            variant="_never",
        )
    if positive and negative:
        return pointwise(context, "($element)(isinf(x0) != 0)", variant="_any")
    sign = ">" if positive else "<"
    return pointwise(
        context,
        f"($element)(isinf(x0) && x0 {sign} 0)",
        variant="_positive" if positive else "_negative",
    )


def _where(context: NodeContext) -> NodeEmission:
    return elementwise(
        context,
        expression="x0 ? x1 : x2",
        operands=tuple(context.require_input(index) for index in range(3)),
        result=context.require_output(0),
    )


for _op_type, (_versions, _expression) in _BINARY_PREDICATES.items():
    register_kernel("", _op_type, _versions, partial(_binary, expression=_expression))
for _op_type, _operator in _BITWISE_OPERATORS.items():
    register_kernel(
        "",
        _op_type,
        _BITWISE_VERSIONS,
        partial(_binary, expression=f"x0 {_operator} x1"),
    )
register_kernel("", "Not", _NOT_VERSIONS, partial(pointwise, template="($element)!x0"))
register_kernel(
    "",
    "BitwiseNot",
    _BITWISE_VERSIONS,
    partial(pointwise, template="($element)~x0"),
)
register_kernel("", "BitShift", _BIT_SHIFT_VERSIONS, _bit_shift)
register_kernel("", "IsInf", _IS_INF_VERSIONS, _is_inf)
register_kernel(
    "",
    "IsNaN",
    _IS_NAN_VERSIONS,
    partial(pointwise, template="($element)(isnan(x0) != 0)"),
)
register_kernel("", "Where", _WHERE_VERSIONS, _where)
