"""Casts: converting a tensor's values to another element type, or reading its bits as one."""

from __future__ import annotations

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_type_name
from fnnx.extras.compilers.c.onnx.kernels import (
    NodeContext,
    NodeEmission,
    copy_tensor,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import elementwise

# Cast-1 took `to` as a type name rather than the type id it has been since 6; every
# revision after that widened the type constraints alone (9 strings, 13 bfloat16, 19 float8
# and the `saturate` attribute that goes with it, 21 int4, 23 float4, 24 and 25 more of the
# same). None of those types are compilable, so one generator serves every revision — and
# the target type is read off the graph rather than off `to`, which shape inference has
# already applied.
_CAST_VERSIONS = (6, 9, 13, 19, 21, 23, 24, 25)
_BITCAST_VERSIONS = (26,)


def _cast(context: NodeContext) -> NodeEmission:
    """Cast between the compilable element types, which is C's own conversion.

    That is what ONNX specifies for all but the boolean target: truncation toward zero out
    of the floating-point families — undefined out of the target's range, as ONNX leaves it
    too — and, between the integer families, the modular reinterpretation of the low bits
    that every two's-complement target performs.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    if source.elem_type == result.elem_type:
        return copy_tensor(source, result)
    element = c_type(result.elem_type)
    # A boolean target is the one rule of ONNX's own: every nonzero value is true, NaN
    # included. It also needs a variant of its own, since `bool` and `uint8` are one C type:
    # without it a model casting in both directions would name two formulas alike.
    to_bool = result.elem_type == TensorProto.BOOL
    return elementwise(
        context,
        expression=f"({element})(x0 != 0)" if to_bool else f"({element})x0",
        operands=(source,),
        result=result,
        variant="_to_bool" if to_bool else "",
    )


def _bitcast(context: NodeContext) -> NodeEmission:
    """BitCast: the same bytes, read at another element type.

    ONNX defines the op only between types of equal width — its own type inference rejects
    anything else before a kernel is reached, and a revision that relaxed that would not be
    served by this generator — so moving the bytes is the whole operation.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    if result.elem_type == TensorProto.BOOL:
        raise CompileError(
            f"Node `{context.label}`: BitCast to `BOOL` is not supported by the C "
            "compiler; a boolean tensor is emitted as bytes holding 0 or 1, which the "
            f"bits of a `{element_type_name(source.elem_type)}` need not be."
        )
    return copy_tensor(source, result)


register_kernel("", "Cast", _CAST_VERSIONS, _cast)
register_kernel("", "BitCast", _BITCAST_VERSIONS, _bitcast)
