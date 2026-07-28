"""Quantization: the affine grid a low-precision tensor stands on, and the ops over it.

A quantized tensor is an integer grid standing for the reals `(value - zero_point) * scale`.
`DequantizeLinear` applies that map and `QuantizeLinear` inverts it — dividing by the scale,
rounding halves to even, and saturating to the target type's range, which is where the
information a low-precision tensor cannot carry is actually lost. Both read their scale and
zero point at one of the three granularities ONNX defines, and all three come to the same
addressing: a stride per axis into those operands' buffers, plus a divisor on the axis a
blocked scale is repeated along.

The other four ops fold the map into a product. `MatMulInteger` and `ConvInteger` subtract the
zero points and accumulate in `int32`, leaving the result on a grid whose scale is the product
of the operands' — which is why they take no scale at all. `QLinearConv` and `QLinearMatMul`
run the same accumulation and then requantize onto a grid of their own, by the one factor
`a_scale * b_scale / y_scale` that product comes to. So the walk over the operands is the
convolution's and the matrix product's, taken from the kernels that run them unquantized, and
only the accumulation type, the zero-point offsets and the store differ.

What is not served: a scale or zero point per row or per column of a matrix product. In the
form ONNX's own text describes it — an `M`-element vector against an `[M, K]` operand — its
reference evaluator stretches that vector along numpy's trailing axis instead, so no oracle
says what a kernel should compute there; the products are served at per-tensor granularity
alone rather than read one way in that form and another in the `[M, 1]` one. The
convolutions' per-output-channel `w_scale` and `w_zero_point` are served: on those the
evaluator and the backend corpus agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from string import Template

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    broadcast_strides,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    kernel_name,
    normalize_axis,
    row_major_strides,
    verify_same_shape,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents, math_suffix
from fnnx.extras.compilers.c.onnx.ops.conv import (
    WINDOW_PARAMETERS,
    convolution_geometry,
    verify_bias,
)
from fnnx.extras.compilers.c.onnx.ops.gemm import PRODUCT_PARAMETERS, matrix_product

# The grids ONNX quantizes onto, of the ones the compiler's element types cover: a
# `QuantizeLinear` saturates to the range of one of these, and every quantized product reads
# its operands from one. `DequantizeLinear` reads `int32` as well — where an accumulated bias
# sits, which nothing ever quantizes *to*.
_GRID_TYPES = (
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
)

# QuantizeLinear arrived at 10, gained per-axis quantization at 13, float8 and `saturate` at
# 19, blocked quantization and the 4-bit types at 21, `precision` and float4 at 23, and more
# types at 24 and 25. Every revision but 13 is claimed: the reference evaluator is
# version-faithful for each of those — and the corpus's own tests import 11, which selects
# 10, and 25 — while nothing can vouch for 13, whose semantics the evaluator does not
# distinguish and which no corpus test imports. DequantizeLinear's history runs alongside it,
# vouched for from 19 on.
_QUANTIZE_VERSIONS = (10, 19, 21, 23, 24, 25)
_DEQUANTIZE_VERSIONS = (19, 21, 23, 24, 25)

# QLinearMatMul is claimed at 21, the revision that widened its scales to a type parameter and
# the one the evaluator distinguishes; the integer products and QLinearConv have had one
# revision each, the one they arrived at.
_QLINEAR_MATMUL_VERSIONS = (21,)
_INTEGER_VERSIONS = (10,)


# --------------------------------------------------------------------------------------
# Rounding onto a grid
# --------------------------------------------------------------------------------------

_SATURATE_TEMPLATE = Template("""\
static $result $name(double value)
{
    /* The nearest integer, halves to even, saturated to the grid's range. A value that is
       not a number has no nearest integer at all: ONNX leaves it undefined, and it lands at
       the low end, which is at least the same end the reference evaluator's own conversion
       leaves it at. */
    if (!(value > $low)) {
        return ($result)$low;
    }
    if (value > $high) {
        return ($result)$high;
    }
    return ($result)rint(value);
}""")


def _saturating_cast(context: NodeContext, elem_type: int) -> CFunction:
    """The rounding-and-saturating store every op that writes a quantized grid ends in."""
    info = np.iinfo(np.dtype(numpy_dtype_name(elem_type)))
    element = c_type(elem_type)
    name = f"{context.prefix}_saturate_{element}"
    return CFunction(
        name,
        _SATURATE_TEMPLATE.substitute(
            name=name,
            result=element,
            low=f"{int(info.min)}.0",
            high=f"{int(info.max)}.0",
        ),
    )


def _verify_grid_type(
    context: NodeContext,
    operand: TensorRef,
    role: str,
    allowed: tuple[int, ...] = _GRID_TYPES,
) -> None:
    """Refuse a tensor the op reads or writes as quantized whose type is no integer grid.

    A quantized tensor's saturation range is its own type's, so a kernel emitted for a type
    ONNX does not quantize onto would round onto a grid that is not there — and a floating
    one would be truncated into the accumulation without a word.
    """
    if operand.elem_type not in allowed:
        names = ", ".join(element_type_name(elem_type) for elem_type in allowed)
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` quantizes onto the integer "
            f"grids {names}, but its {role} `{operand.name}` is "
            f"`{element_type_name(operand.elem_type)}`."
        )


def _verify_zero_point(
    context: NodeContext, zero_point: TensorRef | None, grid: TensorRef
) -> None:
    """Refuse a zero point that does not sit on the same grid as the tensor it shifts."""
    if zero_point is not None and zero_point.elem_type != grid.elem_type:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{zero_point.name}` "
            f"as `{element_type_name(zero_point.elem_type)}` against a quantized tensor of "
            f"`{element_type_name(grid.elem_type)}`; ONNX defines the two as one type."
        )


# --------------------------------------------------------------------------------------
# The affine map: QuantizeLinear and DequantizeLinear
# --------------------------------------------------------------------------------------

_AFFINE_TEMPLATE = Template("""\
static void $name(
    $result* out,
    const $source* in,
    const $scale* scale,
    const $zero* zero_point,
    size_t count,
    int rank,
    const size_t* shape,
    int block_axis,
    size_t block_size,
    const size_t* scale_strides,
    const size_t* zero_strides)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        size_t scale_offset = 0;
        size_t zero_offset = 0;
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            size_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            /* A blocked scale holds one value per `block_size` elements of this axis: the
               repetition ONNX defines it by, read backwards. */
            if (axis == block_axis) {
                coordinate /= block_size;
            }
            scale_offset += coordinate * scale_strides[axis];
            zero_offset += coordinate * zero_strides[axis];
        }
$body
    }
}""")

_QUANTIZE_BODY = Template("""\
        {
            const $compute quotient =
                ($compute)in[index] / ($compute)scale[scale_offset];
            const double zero =
                (zero_point != NULL) ? (double)zero_point[zero_offset] : 0.0;
            /* The quotient is rounded before the zero point shifts it, as ONNX rounds the
               division alone: shifting first would send a half to the other neighbour. */
            out[index] = $saturate((double)$round(quotient) + zero);
        }""")

_DEQUANTIZE_BODY = Template("""\
        {
            const $compute zero = (zero_point != NULL)
                ? ($compute)zero_point[zero_offset]
                : ($compute)0;
            /* The grid is read at single precision whatever its width, which is where the
               reference evaluator converts it too. */
            out[index] = ($result)(((float)in[index] - zero)
                * ($compute)scale[scale_offset]);
        }""")


@dataclass(frozen=True)
class _Granularity:
    """How a scale and a zero point are addressed while the data's own shape is walked."""

    shape: tuple[int, ...]
    scale_strides: tuple[int, ...]
    zero_strides: tuple[int, ...]
    block_axis: int
    block_size: int

    @property
    def arguments(self) -> list[str]:
        """Call-site literals for the addressing parameters an affine map's kernel takes."""
        return [
            f"{math.prod(self.shape)}u",
            str(len(self.shape)),
            extents(self.shape),
            str(self.block_axis),
            f"{self.block_size}u",
            extents(self.scale_strides),
            extents(self.zero_strides),
        ]


def _quantize_linear(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    scale = context.require_input(1)
    zero_point = context.optional_input(2)
    result = context.require_output(0)
    verify_same_shape(context, source, result)
    _verify_grid_type(context, result, "output")
    _verify_zero_point(context, zero_point, result)
    granularity = _granularity(context, source, scale, zero_point)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    compute = _division_type(context, source, scale)
    saturate = _saturating_cast(context, result.elem_type)
    return _affine_emission(
        context,
        source=source,
        scale=scale,
        zero_point=zero_point,
        grid=result,
        result=result,
        compute=compute,
        granularity=granularity,
        body=_QUANTIZE_BODY.substitute(
            compute=c_type(compute),
            round=f"rint{math_suffix(compute)}",
            saturate=saturate.name,
        ),
        helpers=(saturate,),
    )


def _dequantize_linear(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    scale = context.require_input(1)
    zero_point = context.optional_input(2)
    result = context.require_output(0)
    verify_same_shape(context, source, result)
    # `int32` alone is read back from a grid without ever being quantized onto one: it is
    # where an accumulated bias sits, whose scale is the product of the ones it is added to.
    _verify_grid_type(context, source, "input", (*_GRID_TYPES, TensorProto.INT32))
    if result.elem_type not in FLOAT_TYPES:
        raise CompileError(
            f"Node `{context.label}`: `DequantizeLinear` reads a grid back as the reals it "
            f"stands for, but its output `{result.name}` is "
            f"`{element_type_name(result.elem_type)}`."
        )
    _verify_zero_point(context, zero_point, source)
    granularity = _granularity(context, source, scale, zero_point)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    compute = _difference_type(source, zero_point)
    return _affine_emission(
        context,
        source=source,
        scale=scale,
        zero_point=zero_point,
        grid=source,
        result=result,
        compute=compute,
        granularity=granularity,
        body=_DEQUANTIZE_BODY.substitute(
            compute=c_type(compute), result=c_type(result.elem_type)
        ),
    )


def _affine_emission(
    context: NodeContext,
    *,
    source: TensorRef,
    scale: TensorRef,
    zero_point: TensorRef | None,
    grid: TensorRef,
    result: TensorRef,
    compute: int,
    granularity: _Granularity,
    body: str,
    helpers: tuple[CFunction, ...] = (),
) -> NodeEmission:
    """The kernel and call site both affine maps share; only `body` differs between them.

    `grid` is the tensor the zero point sits on -- the result of a quantization and the
    operand of a dequantization -- whose type the parameter keeps even where the node omits
    the operand and the kernel is passed nothing at all.
    """
    name = kernel_name(
        context,
        c_type(source.elem_type),
        c_type(scale.elem_type),
        c_type(result.elem_type),
        c_type(compute),
        *(("zp",) if zero_point is not None else ()),
    )
    definition = _AFFINE_TEMPLATE.substitute(
        name=name,
        result=c_type(result.elem_type),
        source=c_type(source.elem_type),
        scale=c_type(scale.elem_type),
        zero=c_type(grid.elem_type),
        body=body,
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            scale.expr,
            "NULL" if zero_point is None else zero_point.expr,
            *granularity.arguments,
        ],
    )
    return NodeEmission(
        functions=(*helpers, CFunction(name, definition)), statements=(call,)
    )


def _division_type(context: NodeContext, source: TensorRef, scale: TensorRef) -> int:
    """The type `x / y_scale` is computed at.

    ONNX reads it off `y_scale`'s type, while its reference evaluator divides the two arrays
    as numpy does — which promotes an `int32` operand against a `float32` one to `float64`.
    The evaluator is what both suites compare against, so its promotion is what is emitted;
    the two agree wherever both operands are floats, which is every model that quantizes one.
    """
    precision = int(context.attribute("precision", 0))
    if precision:
        raise CompileError(
            f"Node `{context.label}`: `QuantizeLinear` states `precision` "
            f"`{element_type_name(precision)}`, which the C compiler does not serve: the "
            "newest revision ONNX's reference evaluator implements predates the attribute "
            "and refuses a node carrying it, so nothing can vouch for what a kernel "
            "dividing at that precision should produce. Drop the attribute to divide at "
            "the type `y_scale` carries."
        )
    single = {source.elem_type, scale.elem_type} == {TensorProto.FLOAT}
    return TensorProto.FLOAT if single else TensorProto.DOUBLE


def _difference_type(source: TensorRef, zero_point: TensorRef | None) -> int:
    """The type `x - x_zero_point` is computed at.

    The reference converts the grid to `float32` first and subtracts the zero point from
    that, which numpy promotes to `float64` for an `int32` zero point and nothing narrower.
    """
    if zero_point is None or source.elem_type != TensorProto.INT32:
        return TensorProto.FLOAT
    return TensorProto.DOUBLE


def _granularity(
    context: NodeContext,
    source: TensorRef,
    scale: TensorRef,
    zero_point: TensorRef | None,
) -> _Granularity:
    block_size = int(context.attribute("block_size", 0))
    rank = len(source.shape)
    scale_strides, blocked = _grid_strides(context, source, scale, block_size)
    if (
        zero_point is not None
        and zero_point.elem_count != 1
        and zero_point.shape != scale.shape
    ):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{zero_point.name}` "
            f"of shape {list(zero_point.shape)} against a scale of {list(scale.shape)}; "
            "ONNX defines the two as one shape, which is what fixes the granularity."
        )
    zero_strides = (
        (0,) * rank
        if zero_point is None or zero_point.elem_count == 1
        else _grid_strides(context, source, zero_point, block_size)[0]
    )
    return _Granularity(
        shape=source.shape,
        scale_strides=scale_strides,
        zero_strides=zero_strides,
        block_axis=_quantization_axis(context, source) if blocked else -1,
        block_size=block_size if blocked else 1,
    )


def _grid_strides(
    context: NodeContext, source: TensorRef, operand: TensorRef, block_size: int
) -> tuple[tuple[int, ...], bool]:
    """Strides addressing `operand` as the data's coordinates are walked, and if it blocks.

    The three granularities are one addressing: a single-element scale is read at stride zero
    on every axis, a per-axis vector at the stride its one axis carries, and a blocked tensor
    at its own row-major strides — with the coordinate on the quantization axis divided by
    the block size, which is the repetition ONNX defines blocking by.
    """
    rank = len(source.shape)
    if operand.elem_count == 1:
        return (0,) * rank, False
    axis = _quantization_axis(context, source)
    if not block_size:
        if len(operand.shape) != 1:
            raise CompileError(
                f"Node `{context.label}`: `{context.node.op_type}` reads `{operand.name}` "
                f"of shape {list(operand.shape)}; with no `block_size`, ONNX defines a "
                "scale of more than one element as one value per slice along the "
                "quantization axis — a 1-D tensor."
            )
        spread = tuple(
            operand.shape[0] if position == axis else 1 for position in range(rank)
        )
        strides = broadcast_strides(
            replace(operand, shape=spread), source.shape, node_label=context.label
        )
        return strides, False
    _verify_blocks(context, source, operand, axis, block_size)
    return row_major_strides(operand.shape), True


def _quantization_axis(context: NodeContext, source: TensorRef) -> int:
    """The axis a per-axis or blocked scale runs along.

    `QuantizeLinear`-10 predates per-axis quantization and declares no `axis` attribute at
    all; ONNX's reference reads the absent one as 1, which is the default every revision
    that does declare it carries.
    """
    return normalize_axis(context, int(context.attribute("axis", 1)), len(source.shape))


def _verify_blocks(
    context: NodeContext,
    source: TensorRef,
    operand: TensorRef,
    axis: int,
    block_size: int,
) -> None:
    """Refuse a blocked scale the emitted addressing would read outside of.

    A blocked scale carries the data's own shape but for the quantization axis, where it
    holds one value per block of `block_size` elements — so its extent there is what ONNX
    states it: the block count the data's own extent comes to.
    """
    if block_size < 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` states `block_size` "
            f"{block_size}; ONNX defines it as a positive count of elements."
        )
    blocks = -(-source.shape[axis] // block_size)
    expected = tuple(
        blocks if position == axis else extent
        for position, extent in enumerate(source.shape)
    )
    if operand.shape != expected:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` blocks `{source.name}` of "
            f"shape {list(source.shape)} by {block_size} along axis {axis}, which takes a "
            f"scale of {list(expected)}, but `{operand.name}` has shape "
            f"{list(operand.shape)}."
        )


register_kernel("", "QuantizeLinear", _QUANTIZE_VERSIONS, _quantize_linear)
register_kernel("", "DequantizeLinear", _DEQUANTIZE_VERSIONS, _dequantize_linear)


# --------------------------------------------------------------------------------------
# The quantized matrix products
# --------------------------------------------------------------------------------------

_PRODUCT_TEMPLATE = Template("""\
static void $name(
$parameters)
{
$locals
    size_t batch, row, column, index;
    for (batch = 0; batch < batch_count; ++batch) {
        size_t left_base = 0;
        size_t right_base = 0;
        size_t remainder = batch;
        int axis;
        for (axis = batch_rank - 1; axis >= 0; --axis) {
            const size_t coordinate = remainder % batch_shape[axis];
            remainder /= batch_shape[axis];
            left_base += coordinate * left_batch_strides[axis];
            right_base += coordinate * right_batch_strides[axis];
        }
        for (row = 0; row < rows; ++row) {
            for (column = 0; column < columns; ++column) {
                int32_t sum = 0;
                for (index = 0; index < inner; ++index) {
                    sum += ((int32_t)left[left_base + row * inner + index] - left_zero)
                         * ((int32_t)right[right_base + index * columns + column]
                            - right_zero);
                }
$store
            }
        }
    }
}""")

_PRODUCT_STORE = "                out[(batch * rows + row) * columns + column] ="

_ZERO_LOCALS = Template("""\
    const int32_t left_zero =
        (left_zero_point != NULL) ? (int32_t)left_zero_point[0] : 0;
    const int32_t right_zero =
        (right_zero_point != NULL) ? (int32_t)right_zero_point[0] : 0;""")


def _matmul_integer(context: NodeContext) -> NodeEmission:
    left = context.require_input(0)
    right = context.require_input(1)
    left_zero = context.optional_input(2)
    right_zero = context.optional_input(3)
    result = context.require_output(0)
    product = matrix_product(context, left, right)
    verify_shape(context, result, product.result_shape)
    _verify_per_tensor(context, left_zero, "zero point")
    _verify_per_tensor(context, right_zero, "zero point")
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    name = kernel_name(context, c_type(left.elem_type), c_type(right.elem_type))
    definition = _PRODUCT_TEMPLATE.substitute(
        name=name,
        parameters=",\n".join(
            [
                f"    {c_type(result.elem_type)}* out",
                *_product_operands(left, right),
                PRODUCT_PARAMETERS,
            ]
        ),
        locals=_ZERO_LOCALS.template,
        store=f"{_PRODUCT_STORE} sum;",
    )
    call = call_kernel(
        name,
        [
            result.expr,
            left.expr,
            right.expr,
            _pointer(left_zero),
            _pointer(right_zero),
            *product.arguments,
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _qlinear_matmul(context: NodeContext) -> NodeEmission:
    left = context.require_input(0)
    left_scale = context.require_input(1)
    left_zero = context.require_input(2)
    right = context.require_input(3)
    right_scale = context.require_input(4)
    right_zero = context.require_input(5)
    result_scale = context.require_input(6)
    result_zero = context.require_input(7)
    result = context.require_output(0)
    product = matrix_product(context, left, right)
    verify_shape(context, result, product.result_shape)
    for operand in (left, right, result):
        _verify_grid_type(context, operand, "operand")
    _verify_zero_point(context, left_zero, left)
    _verify_zero_point(context, right_zero, right)
    _verify_zero_point(context, result_zero, result)
    for operand, role in (
        (left_scale, "scale"),
        (left_zero, "zero point"),
        (right_scale, "scale"),
        (right_zero, "zero point"),
        (result_scale, "scale"),
        (result_zero, "zero point"),
    ):
        _verify_per_tensor(context, operand, role)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    saturate = _saturating_cast(context, result.elem_type)
    name = kernel_name(
        context,
        c_type(left.elem_type),
        c_type(right.elem_type),
        c_type(result.elem_type),
        c_type(result_scale.elem_type),
    )
    definition = _PRODUCT_TEMPLATE.substitute(
        name=name,
        parameters=",\n".join(
            [
                f"    {c_type(result.elem_type)}* out",
                *_product_operands(left, right),
                f"    const {c_type(left_scale.elem_type)}* left_scale",
                f"    const {c_type(right_scale.elem_type)}* right_scale",
                f"    const {c_type(result_scale.elem_type)}* result_scale",
                f"    const {c_type(result.elem_type)}* result_zero_point",
                PRODUCT_PARAMETERS,
            ]
        ),
        locals="\n".join(
            [
                _ZERO_LOCALS.template,
                # The product of two grids stands on the product of their scales, so
                # requantizing onto a third is this one factor, taken at the scales' own
                # precision as ONNX's reference takes it.
                f"    const {c_type(result_scale.elem_type)} factor =",
                "        left_scale[0] * right_scale[0] / result_scale[0];",
                "    const double result_zero = (double)result_zero_point[0];",
            ]
        ),
        store=(
            f"{_PRODUCT_STORE}\n                    "
            f"{saturate.name}((double)sum * (double)factor + result_zero);"
        ),
    )
    call = call_kernel(
        name,
        [
            result.expr,
            left.expr,
            right.expr,
            left_zero.expr,
            right_zero.expr,
            left_scale.expr,
            right_scale.expr,
            result_scale.expr,
            result_zero.expr,
            *product.arguments,
        ],
    )
    return NodeEmission(
        functions=(saturate, CFunction(name, definition)), statements=(call,)
    )


def _product_operands(left: TensorRef, right: TensorRef) -> list[str]:
    """The operands and zero points both quantized products read, in call order."""
    return [
        f"    const {c_type(left.elem_type)}* left",
        f"    const {c_type(right.elem_type)}* right",
        f"    const {c_type(left.elem_type)}* left_zero_point",
        f"    const {c_type(right.elem_type)}* right_zero_point",
    ]


register_kernel("", "MatMulInteger", _INTEGER_VERSIONS, _matmul_integer)
register_kernel("", "QLinearMatMul", _QLINEAR_MATMUL_VERSIONS, _qlinear_matmul)


# --------------------------------------------------------------------------------------
# The quantized convolutions
# --------------------------------------------------------------------------------------

_CONVOLUTION_TEMPLATE = Template("""\
static void $name(
$parameters)
{
$locals
    size_t batch, group, filter, position, tap, channel;
    for (batch = 0; batch < batch_count; ++batch) {
        for (group = 0; group < groups; ++group) {
            const $source* plane =
                in + (batch * groups + group) * group_channels * input_size;
            for (filter = 0; filter < group_filters; ++filter) {
                const size_t channel_index = group * group_filters + filter;
                const $weight* window =
                    weights + channel_index * group_channels * window_size;
                $result* result =
                    out + (batch * groups * group_filters + channel_index) * output_size;
$channel
                for (position = 0; position < output_size; ++position) {
                    int32_t sum = $initial;
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
                                sum += ((int32_t)plane[channel * input_size + offset]
                                            - input_zero)
                                     * ((int32_t)window[channel * window_size + tap]
                                            - filter_zero);
                            }
                        }
                    }
$store
                }
            }
        }
    }
}""")

_INPUT_ZERO_LOCAL = """\
    const int32_t input_zero =
        (input_zero_point != NULL) ? (int32_t)input_zero_point[0] : 0;"""

# The filter's zero point carries one value per output channel or one for the whole filter,
# which is a stride of one or of zero; either way it is read where the channel is known.
_FILTER_ZERO_CHANNEL = """\
                const int32_t filter_zero = (weight_zero_point != NULL)
                    ? (int32_t)weight_zero_point[channel_index * weight_zero_stride]
                    : 0;"""


def _conv_integer(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    weights = context.require_input(1)
    source_zero = context.optional_input(2)
    weight_zero = context.optional_input(3)
    result = context.require_output(0)
    geometry = convolution_geometry(context, source, weights)
    channels = geometry.groups * geometry.group_filters
    verify_shape(context, result, geometry.result_shape)
    _verify_per_tensor(context, source_zero, "zero point")
    weight_zero_stride = _channel_stride(context, weight_zero, channels, "zero point")
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    name = kernel_name(context, c_type(source.elem_type), c_type(weights.elem_type))
    definition = _CONVOLUTION_TEMPLATE.substitute(
        name=name,
        source=c_type(source.elem_type),
        weight=c_type(weights.elem_type),
        result=c_type(result.elem_type),
        parameters=",\n".join(
            [
                f"    {c_type(result.elem_type)}* out",
                *_convolution_operands(source, weights),
                WINDOW_PARAMETERS,
            ]
        ),
        locals=_INPUT_ZERO_LOCAL,
        channel=_FILTER_ZERO_CHANNEL,
        initial="0",
        store="                    result[position] = sum;",
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            weights.expr,
            _pointer(source_zero),
            _pointer(weight_zero),
            f"{weight_zero_stride}u",
            *geometry.arguments,
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _qlinear_conv(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    source_scale = context.require_input(1)
    source_zero = context.require_input(2)
    weights = context.require_input(3)
    weight_scale = context.require_input(4)
    weight_zero = context.require_input(5)
    result_scale = context.require_input(6)
    result_zero = context.require_input(7)
    bias = context.optional_input(8)
    result = context.require_output(0)
    geometry = convolution_geometry(context, source, weights)
    channels = geometry.groups * geometry.group_filters
    verify_shape(context, result, geometry.result_shape)
    verify_bias(context, bias, channels)
    for operand in (source, weights, result):
        _verify_grid_type(context, operand, "operand")
    _verify_zero_point(context, source_zero, source)
    _verify_zero_point(context, weight_zero, weights)
    _verify_zero_point(context, result_zero, result)
    _verify_per_tensor(context, source_scale, "scale")
    _verify_per_tensor(context, source_zero, "zero point")
    _verify_per_tensor(context, result_scale, "scale")
    _verify_per_tensor(context, result_zero, "zero point")
    weight_scale_stride = _channel_stride(context, weight_scale, channels, "scale")
    weight_zero_stride = _channel_stride(context, weight_zero, channels, "zero point")
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    saturate = _saturating_cast(context, result.elem_type)
    name = kernel_name(
        context,
        c_type(source.elem_type),
        c_type(weights.elem_type),
        c_type(result.elem_type),
        c_type(result_scale.elem_type),
    )
    definition = _CONVOLUTION_TEMPLATE.substitute(
        name=name,
        source=c_type(source.elem_type),
        weight=c_type(weights.elem_type),
        result=c_type(result.elem_type),
        parameters=",\n".join(
            [
                f"    {c_type(result.elem_type)}* out",
                *_convolution_operands(source, weights),
                f"    const {c_type(source_scale.elem_type)}* input_scale",
                f"    const {c_type(weight_scale.elem_type)}* weight_scale",
                "    size_t weight_scale_stride",
                f"    const {c_type(result_scale.elem_type)}* result_scale",
                f"    const {c_type(result.elem_type)}* result_zero_point",
                "    const int32_t* bias",
                WINDOW_PARAMETERS,
            ]
        ),
        locals="\n".join(
            [
                _INPUT_ZERO_LOCAL,
                "    const double result_zero = (double)result_zero_point[0];",
            ]
        ),
        channel="\n".join(
            [
                _FILTER_ZERO_CHANNEL,
                # One factor per output channel: the scales of the two grids the products
                # come from, over the scale of the grid the result is written on.
                f"                const {c_type(result_scale.elem_type)} factor =",
                "                    input_scale[0]",
                "                        * weight_scale[channel_index"
                " * weight_scale_stride]",
                "                        / result_scale[0];",
            ]
        ),
        initial="(bias != NULL) ? bias[channel_index] : 0",
        store=(
            "                    result[position] =\n"
            f"                        {saturate.name}("
            "(double)sum * (double)factor + result_zero);"
        ),
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            weights.expr,
            source_zero.expr,
            weight_zero.expr,
            f"{weight_zero_stride}u",
            source_scale.expr,
            weight_scale.expr,
            f"{weight_scale_stride}u",
            result_scale.expr,
            result_zero.expr,
            _pointer(bias),
            *geometry.arguments,
        ],
    )
    return NodeEmission(
        functions=(saturate, CFunction(name, definition)), statements=(call,)
    )


def _convolution_operands(source: TensorRef, weights: TensorRef) -> list[str]:
    """The operands and zero points both quantized convolutions read, in call order."""
    return [
        f"    const {c_type(source.elem_type)}* in",
        f"    const {c_type(weights.elem_type)}* weights",
        f"    const {c_type(source.elem_type)}* input_zero_point",
        f"    const {c_type(weights.elem_type)}* weight_zero_point",
        "    size_t weight_zero_stride",
    ]


def _verify_per_tensor(
    context: NodeContext, operand: TensorRef | None, role: str
) -> None:
    """Refuse a scale or zero point this op reads as the whole tensor's and that is not one.

    ONNX's reference evaluator stretches the per-row vector its own text describes along
    numpy's trailing axis rather than the axis that text names, so nothing can vouch for what
    a kernel should compute for it, and the whole granularity is left unserved rather than
    read one way in that form and another in the one numpy does broadcast as written.
    """
    if operand is not None and operand.elem_count != 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{operand.name}` as "
            f"the one {role} of the whole tensor, but it has shape "
            f"{list(operand.shape)}; the C compiler serves this op at per-tensor "
            "granularity only."
        )


def _channel_stride(
    context: NodeContext, operand: TensorRef | None, channels: int, role: str
) -> int:
    """The stride reading `operand` per output channel: one per channel, or one for all."""
    if operand is None or operand.elem_count == 1:
        return 0
    if operand.shape != (channels,):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{operand.name}` as "
            f"one {role} per output channel — a 1-D tensor of {channels} — or as one for "
            f"the whole filter, but it has shape {list(operand.shape)}."
        )
    return 1


def _pointer(operand: TensorRef | None) -> str:
    return "NULL" if operand is None else operand.expr


register_kernel("", "ConvInteger", _INTEGER_VERSIONS, _conv_integer)
register_kernel("", "QLinearConv", _INTEGER_VERSIONS, _qlinear_conv)
