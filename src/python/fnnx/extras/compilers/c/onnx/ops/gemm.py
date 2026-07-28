"""The dense matrix kernels: Gemm, MatMul and Det.

All three walk a row-major buffer as a stack of matrices, and all three sum a product over an
inner axis in an order the spec leaves open. What separates them is how the matrices are
addressed: Gemm's two transposes and its broadcast bias become strides, MatMul's batch axes
broadcast against each other, and Det walks one matrix at a time down a copy of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import FLOAT_TYPES, c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    broadcast_strides,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    kernel_name,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents, math_suffix

# Both transposes and C's broadcast become strides, so one kernel per element type covers
# every attribute combination.
_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* left,
    const $element* right,
    const $element* bias,
    size_t rows,
    size_t columns,
    size_t inner,
    size_t left_row_stride,
    size_t left_inner_stride,
    size_t right_inner_stride,
    size_t right_column_stride,
    size_t bias_row_stride,
    size_t bias_column_stride,
    $scalar alpha,
    $scalar beta)
{
    size_t row, column, index;
    for (row = 0; row < rows; ++row) {
        for (column = 0; column < columns; ++column) {
            $element sum = $zero;
            for (index = 0; index < inner; ++index) {
                sum += left[row * left_row_stride + index * left_inner_stride]
                     * right[index * right_inner_stride + column * right_column_stride];
            }
$store
        }
    }
}""")

# The floating-point families scale in the element type, as the reference does when it
# multiplies a float array by alpha.
_FLOAT_STORE = Template("""\
            sum *= alpha;
            if (bias != NULL) {
                sum += beta * bias[row * bias_row_stride + column * bias_column_stride];
            }
            out[row * columns + column] = sum;""")

# The integer families scale in double and truncate on the way back, as the reference does
# when the float alpha promotes an integer dot product to float64.
_INTEGER_STORE = Template("""\
            {
                double scaled = (double)sum * alpha;
                if (bias != NULL) {
                    scaled += beta
                        * (double)bias[row * bias_row_stride
                                       + column * bias_column_stride];
                }
                out[row * columns + column] = ($element)scaled;
            }""")

# Numpy-style broadcasting of C arrived at opset 7; 9 and 13 widened the types, and 11 made
# C optional, which the kernel already handles.
_GEMM_VERSIONS = (7, 9, 11, 13)


def _gemm(context: NodeContext) -> NodeEmission:
    left = context.require_input(0)
    right = context.require_input(1)
    bias = context.optional_input(2)
    result = context.require_output(0)
    alpha = float(context.attribute("alpha", 1.0))
    beta = float(context.attribute("beta", 1.0))
    transpose_left = bool(context.attribute("transA", 0))
    transpose_right = bool(context.attribute("transB", 0))

    rows, inner = _oriented_shape(context, left, transpose_left)
    right_inner, columns = _oriented_shape(context, right, transpose_right)
    if inner != right_inner:
        raise CompileError(
            f"Node `{context.label}`: Gemm operands `{left.name}` and `{right.name}` do "
            f"not share an inner dimension ({inner} against {right_inner})."
        )
    left_strides = _oriented_strides(left.shape, transpose_left)
    right_strides = _oriented_strides(right.shape, transpose_right)
    bias_strides: tuple[int, ...]
    if bias is None or beta == 0.0:
        # The reference drops C whenever beta is zero, so `0 * inf` never reaches the sum.
        bias_expr, bias_strides = "NULL", (0, 0)
    else:
        bias_expr = bias.expr
        bias_strides = broadcast_strides(
            bias, (rows, columns), node_label=context.label
        )

    element = c_type(result.elem_type)
    scales_in_element_type = result.elem_type in FLOAT_TYPES
    scalar_type = result.elem_type if scales_in_element_type else TensorProto.DOUBLE
    name = f"{context.prefix}_gemm_{element}"
    store = _FLOAT_STORE if scales_in_element_type else _INTEGER_STORE
    definition = _TEMPLATE.substitute(
        name=name,
        element=element,
        scalar=c_type(scalar_type),
        zero=scalar_literal(0, result.elem_type),
        store=store.substitute(element=element),
    )
    call = "\n".join(
        [
            f"{name}(",
            f"    {result.expr}, {left.expr}, {right.expr}, {bias_expr},",
            f"    {rows}u, {columns}u, {inner}u,",
            f"    {left_strides[0]}u, {left_strides[1]}u, "
            f"{right_strides[0]}u, {right_strides[1]}u,",
            f"    {bias_strides[0]}u, {bias_strides[1]}u,",
            f"    {scalar_literal(alpha, scalar_type)}, "
            f"{scalar_literal(beta, scalar_type)});",
        ]
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _oriented_shape(
    context: NodeContext, operand: TensorRef, transposed: bool
) -> tuple[int, int]:
    """The operand's shape as the multiplication sees it, after its transpose attribute."""
    if len(operand.shape) != 2:
        raise CompileError(
            f"Node `{context.label}`: Gemm takes 2-D operands, but `{operand.name}` has "
            f"shape {list(operand.shape)}."
        )
    rows, columns = operand.shape
    return (columns, rows) if transposed else (rows, columns)


def _oriented_strides(shape: tuple[int, ...], transposed: bool) -> tuple[int, int]:
    """Strides along the two axes of the oriented operand, into its row-major buffer."""
    return (1, shape[1]) if transposed else (shape[1], 1)


register_kernel("", "Gemm", _GEMM_VERSIONS, _gemm)


# MatMul is numpy's `matmul`: everything before the last two axes is a batch the two operands
# broadcast against each other, so the batch coordinate becomes an offset into each of them
# and the matrix product itself is the same three loops Gemm runs.
# The geometry of a batched product, in the order `MatrixProduct.arguments` fills it; the
# quantized products take the same block after their own operands.
PRODUCT_PARAMETERS = """\
    size_t batch_count,
    int batch_rank,
    const size_t* batch_shape,
    const size_t* left_batch_strides,
    const size_t* right_batch_strides,
    size_t rows,
    size_t columns,
    size_t inner"""

_MATMUL_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* left,
    const $element* right,
$parameters)
{
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
                $element sum = $zero;
                for (index = 0; index < inner; ++index) {
                    sum += left[left_base + row * inner + index]
                         * right[right_base + index * columns + column];
                }
                out[(batch * rows + row) * columns + column] = sum;
            }
        }
    }
}""")

# MatMul's semantics have not changed since opset 1: 9 added the integer families and 13
# bfloat16. Only 13 is claimed all the same, because it is the only revision anything can
# vouch for — the reference evaluator is version-faithful there and the corpus's own MatMul
# tests import it, while nothing checks 1 or 9. A model importing one of those gets the
# unsupported-version error rather than a kernel no oracle has ever seen.
_MATMUL_VERSIONS = (13,)


@dataclass(frozen=True)
class MatrixProduct:
    """Where every matrix of a batched product sits, and the shape the product comes to.

    The quantized products walk the same operands as `MatMul`, so the addressing is resolved
    once here and each kernel differs only in what it accumulates and how it stores it.
    """

    rows: int
    columns: int
    inner: int
    batch_shape: tuple[int, ...]
    left_batch_strides: tuple[int, ...]
    right_batch_strides: tuple[int, ...]
    result_shape: tuple[int, ...]

    @property
    def arguments(self) -> list[str]:
        """Call-site literals for the geometry parameters a batched product's kernel takes."""
        return [
            f"{math.prod(self.batch_shape)}u",
            str(len(self.batch_shape)),
            extents(self.batch_shape),
            extents(self.left_batch_strides),
            extents(self.right_batch_strides),
            f"{self.rows}u",
            f"{self.columns}u",
            f"{self.inner}u",
        ]


def matrix_product(
    context: NodeContext, left: TensorRef, right: TensorRef
) -> MatrixProduct:
    (rows, inner), left_batch = _matrix_stack(context, left, column_vector=False)
    (right_inner, columns), right_batch = _matrix_stack(
        context, right, column_vector=True
    )
    if inner != right_inner:
        raise CompileError(
            f"Node `{context.label}`: {context.node.op_type} operands `{left.name}` and "
            f"`{right.name}` do not share an inner dimension ({inner} against "
            f"{right_inner})."
        )
    batch_shape = _broadcast_shape(context, left_batch, right_batch)
    return MatrixProduct(
        rows=rows,
        columns=columns,
        inner=inner,
        batch_shape=batch_shape,
        left_batch_strides=_batch_strides(
            context, left, left_batch, batch_shape, rows * inner
        ),
        right_batch_strides=_batch_strides(
            context, right, right_batch, batch_shape, inner * columns
        ),
        # A promoted rank-1 operand contributes an axis of extent 1 that ONNX drops from
        # the result again, which leaves the row-major layout — and so the addressing —
        # unchanged.
        result_shape=(
            *batch_shape,
            *((rows,) if len(left.shape) > 1 else ()),
            *((columns,) if len(right.shape) > 1 else ()),
        ),
    )


def _matmul(context: NodeContext) -> NodeEmission:
    left = context.require_input(0)
    right = context.require_input(1)
    result = context.require_output(0)
    product = matrix_product(context, left, right)
    verify_shape(context, result, product.result_shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    definition = _MATMUL_TEMPLATE.substitute(
        name=name,
        element=element,
        zero=scalar_literal(0, result.elem_type),
        parameters=PRODUCT_PARAMETERS,
    )
    call = call_kernel(name, [result.expr, left.expr, right.expr, *product.arguments])
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _matrix_stack(
    context: NodeContext, operand: TensorRef, *, column_vector: bool
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """The operand's trailing matrix and the batch axes in front of it.

    numpy — and so ONNX — reads a rank-1 operand as the single row or column that makes the
    product defined, and drops that axis from the result again.
    """
    if not operand.shape:
        raise CompileError(
            f"Node `{context.label}`: {context.node.op_type} multiplies matrices, but "
            f"`{operand.name}` is a scalar."
        )
    if len(operand.shape) == 1:
        (extent,) = operand.shape
        return ((extent, 1) if column_vector else (1, extent)), ()
    return (operand.shape[-2], operand.shape[-1]), operand.shape[:-2]


def _broadcast_shape(
    context: NodeContext, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """The shape two batches broadcast onto, numpy-style, aligned at their trailing axes."""
    rank = max(len(left), len(right))
    padded_left = (1,) * (rank - len(left)) + left
    padded_right = (1,) * (rank - len(right)) + right
    shape = []
    for one, other in zip(padded_left, padded_right):
        if one != other and 1 not in (one, other):
            raise CompileError(
                f"Node `{context.label}`: the batch shapes {list(left)} and {list(right)} "
                "of its operands do not broadcast against each other."
            )
        # An axis of 1 stretches to whatever the other side is, and that includes 0: a batch
        # of no matrices against a batch of one is still a batch of no matrices, so `max`
        # would be wrong exactly where a zero-element operand is involved.
        shape.append(other if one == 1 else one)
    return tuple(shape)


def _batch_strides(
    context: NodeContext,
    operand: TensorRef,
    batch: tuple[int, ...],
    batch_shape: tuple[int, ...],
    matrix_size: int,
) -> tuple[int, ...]:
    """Strides addressing the operand's matrices while iterating the broadcast batch."""
    return tuple(
        stride * matrix_size
        for stride in broadcast_strides(
            replace(operand, shape=batch), batch_shape, node_label=context.label
        )
    )


register_kernel("", "MatMul", _MATMUL_VERSIONS, _matmul)


# The determinant, by the same LU factorization with partial pivoting LAPACK runs, so that
# the pivots multiplied together here are the ones the reference's `numpy.linalg.det`
# multiplies. Elimination is destructive, hence the copy: the operand is read-only, and the
# artifact has nowhere else to put a working matrix.
_DET_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    $element* work,
    size_t batch_count,
    size_t order)
{
    size_t batch, step, row, column, pivot;
    for (batch = 0; batch < batch_count; ++batch) {
        $element determinant = $one;
        memcpy(work, in + batch * order * order, order * order * sizeof(*work));
        for (step = 0; step < order; ++step) {
            pivot = step;
            for (row = step + 1; row < order; ++row) {
                if ($absolute(work[row * order + step])
                        > $absolute(work[pivot * order + step])) {
                    pivot = row;
                }
            }
            if (pivot != step) {
                for (column = step; column < order; ++column) {
                    const $element swapped = work[step * order + column];
                    work[step * order + column] = work[pivot * order + column];
                    work[pivot * order + column] = swapped;
                }
                determinant = -determinant;
            }
            determinant *= work[step * order + step];
            if (work[step * order + step] == $zero) {
                break;
            }
            for (row = step + 1; row < order; ++row) {
                const $element factor =
                    work[row * order + step] / work[step * order + step];
                for (column = step + 1; column < order; ++column) {
                    work[row * order + column] -=
                        factor * work[step * order + column];
                }
            }
        }
        out[batch] = determinant;
    }
}""")

# Det arrived at 11 and 22 added bfloat16. Only 22 is claimed, for the reason MatMul claims
# only 13: it is the revision the reference evaluator is faithful for and the one both corpus
# tests import, and nothing vouches for 11.
_DET_VERSIONS = (22,)


def _det(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    if len(source.shape) < 2 or source.shape[-1] != source.shape[-2]:
        raise CompileError(
            f"Node `{context.label}`: Det takes square matrices, but `{source.name}` has "
            f"shape {list(source.shape)}."
        )
    verify_shape(context, result, source.shape[:-2])
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    order = source.shape[-1]
    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    definition = _DET_TEMPLATE.substitute(
        name=name,
        element=element,
        absolute=f"fabs{math_suffix(result.elem_type)}",
        one=scalar_literal(1, result.elem_type),
        zero=scalar_literal(0, result.elem_type),
    )
    work = ScratchBuffer(f"{name}_work", result.elem_type, order * order)
    call = call_kernel(
        name,
        [result.expr, source.expr, work.symbol, f"{result.elem_count}u", f"{order}u"],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),), statements=(call,), scratch=(work,)
    )


register_kernel("", "Det", _DET_VERSIONS, _det)
