"""ONNX element types the C compiler supports, and the C types they are emitted as."""

from __future__ import annotations

from onnx import TensorProto, helper

from fnnx.extras.compilers.c.errors import CompileError

# bool becomes a byte holding 0/1 rather than C99's `_Bool`: every tensor buffer then has
# an explicit, fixed-width layout that is identical across compilers and ABIs.
C_TYPES: dict[int, str] = {
    TensorProto.FLOAT: "float",
    TensorProto.DOUBLE: "double",
    TensorProto.INT8: "int8_t",
    TensorProto.INT16: "int16_t",
    TensorProto.INT32: "int32_t",
    TensorProto.INT64: "int64_t",
    TensorProto.UINT8: "uint8_t",
    TensorProto.UINT16: "uint16_t",
    TensorProto.UINT32: "uint32_t",
    TensorProto.UINT64: "uint64_t",
    TensorProto.BOOL: "uint8_t",
}

# The floating-point element types: the ones whose kernels have to reckon with NaN and
# signed zero, and whose arithmetic differs from the integer families'.
FLOAT_TYPES = frozenset({TensorProto.FLOAT, TensorProto.DOUBLE})

# The unsigned integer element types. A kernel branching on `value < 0` is not merely dead
# code for these — it is a `-Wtype-limits` diagnostic, which the artifact's `-Werror` build
# contract turns into a failure — so their kernels drop the negative branch instead.
UNSIGNED_TYPES = frozenset(
    {
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }
)


def element_type_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN({elem_type})"


def is_supported(elem_type: int) -> bool:
    return elem_type in C_TYPES


def numpy_dtype_name(elem_type: int) -> str:
    """The numpy name (`float32`, `bool`, ...) callers bind this element type to."""
    return helper.tensor_dtype_to_np_dtype(elem_type).name


def element_size(elem_type: int) -> int:
    return helper.tensor_dtype_to_np_dtype(elem_type).itemsize


def c_type(elem_type: int) -> str:
    try:
        return C_TYPES[elem_type]
    except KeyError:
        raise CompileError(
            f"Element type `{element_type_name(elem_type)}` is not supported by the C "
            f"compiler; supported types are "
            f"{', '.join(sorted(element_type_name(t) for t in C_TYPES))}."
        ) from None
