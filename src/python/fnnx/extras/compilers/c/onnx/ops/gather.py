"""The ops that read a tensor at positions decided at run time.

Gather, GatherElements and GatherND all address `data` through an index operand, and TopK
through a ranking it computes; either way the positions are values rather than shapes, so
the addressing is a loop instead of the compile-time strides the views are emitted from.
What stays static is the result's shape, which follows from the operands' shapes alone.

An index operand comes from the caller, so every one of them is normalized the way ONNX
defines it — a negative index counted back from the end of the axis — and then bounds
checked: a kernel returns nonzero for an index outside its axis and the entrypoint passes
that on as an argument error, rather than reading past a buffer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import FLOAT_TYPES, c_type
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
    kernel_name,
    normalize_axis,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import expand, extents

# Gather-11 defined the negative index Gather-1 left undefined, and 13 added bfloat16;
# GatherElements arrived at 11 and gained bfloat16 at 13. Accepting a negative index at
# Gather-1 serves more than that revision defines, never something else in its place.
_GATHER_VERSIONS = (1, 11, 13)
_GATHER_ELEMENTS_VERSIONS = (11, 13)

# GatherND-12 added `batch_dims`, which the generator reads as 0 where the schema has none;
# 13 added bfloat16.
_GATHER_ND_VERSIONS = (11, 12, 13)

# TopK moved `k` from an attribute to an operand at 10 and gained `largest` and `sorted` at
# 11; 24 added bfloat16. The attribute form is the same selection read from another place,
# so one generator serves it, told where to look.
_TOP_K_ATTRIBUTE_VERSIONS = (1,)
_TOP_K_OPERAND_VERSIONS = (10, 11, 24)

_GATHER_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $index* indices,
    size_t outer,
    size_t extent,
    size_t inner,
    size_t index_count)
{
    size_t before, chosen;
    for (before = 0; before < outer; ++before) {
        for (chosen = 0; chosen < index_count; ++chosen) {
            ptrdiff_t position = (ptrdiff_t)indices[chosen];
            if (position < 0) {
                position += (ptrdiff_t)extent;
            }
            if (position < 0 || position >= (ptrdiff_t)extent) {
                return 1;
            }
            memcpy(
                out + (before * index_count + chosen) * inner,
                in + (before * extent + (size_t)position) * inner,
                inner * sizeof(*out));
        }
    }
    return 0;
}""")

_GATHER_ELEMENTS_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $index* indices,
    size_t count,
    int rank,
    const size_t* shape,
    const size_t* strides,
    int axis,
    size_t extent)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        size_t source = 0;
        int walked;
        ptrdiff_t position = (ptrdiff_t)indices[index];
        for (walked = rank - 1; walked >= 0; --walked) {
            const size_t coordinate = remainder % shape[walked];
            remainder /= shape[walked];
            if (walked != axis) {
                source += coordinate * strides[walked];
            }
        }
        if (position < 0) {
            position += (ptrdiff_t)extent;
        }
        if (position < 0 || position >= (ptrdiff_t)extent) {
            return 1;
        }
        out[index] = in[source + (size_t)position * strides[axis]];
    }
    return 0;
}""")

_GATHER_ND_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $index* indices,
    size_t batch,
    size_t outer,
    size_t depth,
    size_t slice_size,
    size_t batch_stride,
    const size_t* extents,
    const size_t* strides)
{
    size_t batch_index, row, level;
    for (batch_index = 0; batch_index < batch; ++batch_index) {
        for (row = 0; row < outer; ++row) {
            const size_t block = batch_index * outer + row;
            size_t source = batch_index * batch_stride;
            for (level = 0; level < depth; ++level) {
                ptrdiff_t position = (ptrdiff_t)indices[block * depth + level];
                if (position < 0) {
                    position += (ptrdiff_t)extents[level];
                }
                if (position < 0 || position >= (ptrdiff_t)extents[level]) {
                    return 1;
                }
                source += (size_t)position * strides[level];
            }
            memcpy(out + block * slice_size, in + source, slice_size * sizeof(*out));
        }
    }
    return 0;
}""")

# The selection is a partial sort: each pass takes the element that comes first among those
# the passes before it left, which is one scan of the group per result. `$precedes` is a
# strict total order on (value, position) pairs, so "what the passes before left" is
# everything the last selection precedes — no marker array, and no allocation.
_TOP_K_TEMPLATE = Template("""\
static void $name(
    $element* values,
    int64_t* indices,
    const $element* in,
    size_t outer,
    size_t extent,
    size_t inner,
    size_t wanted)
{
    size_t before, after, rank, position;
    for (before = 0; before < outer; ++before) {
        for (after = 0; after < inner; ++after) {
            $element taken = $zero;
            ptrdiff_t taken_at = -1;
            for (rank = 0; rank < wanted; ++rank) {
                $element best = $zero;
                ptrdiff_t best_at = -1;
                for (position = 0; position < extent; ++position) {
                    const $element candidate =
                        in[(before * extent + position) * inner + after];
                    if (taken_at >= 0 &&
                        !$precedes(taken, (size_t)taken_at, candidate, position)) {
                        continue;
                    }
                    if (best_at < 0 ||
                        $precedes(candidate, position, best, (size_t)best_at)) {
                        best = candidate;
                        best_at = (ptrdiff_t)position;
                    }
                }
                values[(before * wanted + rank) * inner + after] = best;
                indices[(before * wanted + rank) * inner + after] = (int64_t)best_at;
                taken = best;
                taken_at = best_at;
            }
        }
    }
}""")

# Ties are broken by position, which is what makes the order strict and total; the value
# comparison itself is numpy's, since that is what the reference evaluator sorts with — a
# NaN counts as larger than every number, at either end of the ranking.
_PRECEDES_TEMPLATE = Template("""\
static int $name($element left, size_t left_at, $element right, size_t right_at)
{
    if ($better) {
        return 1;
    }
    if ($worse) {
        return 0;
    }
    return left_at < right_at;
}""")


def _gather(context: NodeContext) -> NodeEmission:
    """Gather: the operand sliced along one axis at each index the operand names."""
    data = context.require_input(0)
    indices = context.require_input(1)
    result = context.require_output(0)
    rank = len(data.shape)
    if rank == 0:
        raise CompileError(
            f"Node `{context.label}`: `Gather` reads along an axis of `{data.name}`, "
            "which is a scalar and has none."
        )
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    verify_shape(
        context,
        result,
        (*data.shape[:axis], *indices.shape, *data.shape[axis + 1 :]),
    )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    outer, extent, inner = _split_at(data.shape, axis)
    name = _indexed_name(context, data, indices)
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _GATHER_TEMPLATE.substitute(
                    name=name,
                    element=c_type(data.elem_type),
                    index=c_type(indices.elem_type),
                ),
            ),
        ),
        statements=(
            checked_call(
                context,
                name,
                [
                    result.expr,
                    data.expr,
                    indices.expr,
                    f"{outer}u",
                    f"{extent}u",
                    f"{inner}u",
                    f"{indices.elem_count}u",
                ],
            ),
        ),
    )


def _gather_elements(context: NodeContext) -> NodeEmission:
    """GatherElements: one element of the operand per index, at the index's own coordinates."""
    data = context.require_input(0)
    indices = context.require_input(1)
    result = context.require_output(0)
    rank = len(data.shape)
    if rank == 0 or len(indices.shape) != rank:
        raise CompileError(
            f"Node `{context.label}`: `GatherElements` reads `{data.name}` of rank {rank} "
            f"through `{indices.name}` of rank {len(indices.shape)}; ONNX defines the two "
            "as having the same rank, and at least one axis."
        )
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    for other in range(rank):
        if other != axis and indices.shape[other] != data.shape[other]:
            raise CompileError(
                f"Node `{context.label}`: `GatherElements` gathers along axis {axis}, so "
                f"`{indices.name}` of shape {list(indices.shape)} and `{data.name}` of "
                f"shape {list(data.shape)} have to agree on every other axis."
            )
    verify_shape(context, result, indices.shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    name = _indexed_name(context, data, indices)
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _GATHER_ELEMENTS_TEMPLATE.substitute(
                    name=name,
                    element=c_type(data.elem_type),
                    index=c_type(indices.elem_type),
                ),
            ),
        ),
        statements=(
            checked_call(
                context,
                name,
                [
                    result.expr,
                    data.expr,
                    indices.expr,
                    f"{result.elem_count}u",
                    str(rank),
                    extents(indices.shape),
                    extents(row_major_strides(data.shape)),
                    str(axis),
                    f"{data.shape[axis]}u",
                ],
            ),
        ),
    )


def _gather_nd(context: NodeContext) -> NodeEmission:
    """GatherND: a slice of the operand per index tuple, the leading axes shared as batches."""
    data = context.require_input(0)
    indices = context.require_input(1)
    result = context.require_output(0)
    # GatherND-11 has no `batch_dims`, so the default is read here rather than off the
    # schema, which carries no entry to read it from at that revision.
    batch_dims = int(context.attribute("batch_dims", 0))
    rank = len(data.shape)
    if not indices.shape:
        raise CompileError(
            f"Node `{context.label}`: `GatherND` takes its index tuples from the last axis "
            f"of `{indices.name}`, which is a scalar and has none."
        )
    depth = indices.shape[-1]
    if not 0 <= batch_dims < min(rank, len(indices.shape)):
        raise CompileError(
            f"Node `{context.label}`: `GatherND` shares {batch_dims} batch dimension(s) "
            f"between `{data.name}` of rank {rank} and `{indices.name}` of rank "
            f"{len(indices.shape)}; ONNX defines it as fewer than either rank."
        )
    if data.shape[:batch_dims] != indices.shape[:batch_dims]:
        raise CompileError(
            f"Node `{context.label}`: `GatherND` shares {batch_dims} batch dimension(s), "
            f"but `{data.name}` of shape {list(data.shape)} and `{indices.name}` of shape "
            f"{list(indices.shape)} disagree on them."
        )
    if depth > rank - batch_dims:
        raise CompileError(
            f"Node `{context.label}`: `GatherND` indexes {depth} dimension(s) of "
            f"`{data.name}`, which has {rank - batch_dims} left after its batch dimensions."
        )
    verify_shape(
        context,
        result,
        (*indices.shape[:-1], *data.shape[batch_dims + depth :]),
    )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    batch = math.prod(data.shape[:batch_dims])
    strides = row_major_strides(data.shape)
    slice_size = math.prod(data.shape[batch_dims + depth :])
    name = _indexed_name(context, data, indices)
    return NodeEmission(
        functions=(
            CFunction(
                name,
                _GATHER_ND_TEMPLATE.substitute(
                    name=name,
                    element=c_type(data.elem_type),
                    index=c_type(indices.elem_type),
                ),
            ),
        ),
        statements=(
            checked_call(
                context,
                name,
                [
                    result.expr,
                    data.expr,
                    indices.expr,
                    f"{batch}u",
                    f"{math.prod(indices.shape[batch_dims:-1])}u",
                    f"{depth}u",
                    f"{slice_size}u",
                    f"{math.prod(data.shape[batch_dims:])}u",
                    extents(data.shape[batch_dims : batch_dims + depth]),
                    extents(strides[batch_dims : batch_dims + depth]),
                ],
            ),
        ),
    )


def _top_k(context: NodeContext, *, from_attribute: bool) -> NodeEmission:
    """TopK: the `k` first elements along one axis under the ranking `largest` selects.

    `k` has to be fixed at compile time — it is the extent of the result's axis, and a value
    the graph only computes at run time would make that shape depend on input data. The
    ranking itself is the reference evaluator's: values first, and among equal values the
    smaller position, so the selection is defined even where a group holds duplicates.
    """
    source = context.require_input(0)
    values = context.require_output(0)
    indices = context.require_output(1)
    rank = len(source.shape)
    if rank == 0:
        raise CompileError(
            f"Node `{context.label}`: `TopK` ranks along an axis of `{source.name}`, which "
            "is a scalar and has none."
        )
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    wanted = _wanted(context, from_attribute=from_attribute)
    if not 0 <= wanted <= source.shape[axis]:
        raise CompileError(
            f"Node `{context.label}`: `TopK` asks for {wanted} element(s) along axis "
            f"{axis} of `{source.name}`, which holds {source.shape[axis]}."
        )
    expected = (*source.shape[:axis], wanted, *source.shape[axis + 1 :])
    verify_shape(context, values, expected)
    verify_shape(context, indices, expected)
    if values.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    # `largest` and `sorted` arrived at 11; the revisions before it always take the largest
    # in sorted order, which is what the two attributes default to. Sorted output answers
    # `sorted=0` as well, which asks for the same elements in no particular order.
    largest = bool(int(context.attribute("largest", 1)))
    precedes = _precedes(context, source.elem_type, largest=largest)
    outer, extent, inner = _split_at(source.shape, axis)
    name = kernel_name(
        context, "largest" if largest else "smallest", c_type(source.elem_type)
    )
    return NodeEmission(
        functions=(
            precedes,
            CFunction(
                name,
                _TOP_K_TEMPLATE.substitute(
                    name=name,
                    element=c_type(source.elem_type),
                    zero=expand("$zero", source.elem_type),
                    precedes=precedes.name,
                ),
            ),
        ),
        statements=(
            call_kernel(
                name,
                [
                    values.expr,
                    indices.expr,
                    source.expr,
                    f"{outer}u",
                    f"{extent}u",
                    f"{inner}u",
                    f"{wanted}u",
                ],
            ),
        ),
    )


def _wanted(context: NodeContext, *, from_attribute: bool) -> int:
    """How many elements TopK selects, from wherever this revision states it."""
    if from_attribute:
        return context.int_attribute("k")
    operand = context.require_input(1)
    fixed = context.constant_input(1)
    if fixed is None or fixed.size != 1:
        raise CompileError(
            f"Node `{context.label}`: `TopK` takes `k` from `{operand.name}`, which holds "
            f"{'no single value' if fixed is not None else 'no value'} known at compile "
            "time; the shape of the result then depends on input data, which the C "
            "compiler cannot compile."
        )
    return int(fixed.reshape(-1)[0])


def _precedes(context: NodeContext, elem_type: int, *, largest: bool) -> CFunction:
    """The order TopK ranks by, as a function of two values and their positions."""
    comparison = ">" if largest else "<"
    better = f"left {comparison} right"
    worse = f"right {comparison} left"
    if elem_type in FLOAT_TYPES:
        # numpy sorts a NaN above every number, which puts it first when the largest come
        # first and last when the smallest do.
        outranks = "isnan(left) && !isnan(right)"
        outranked = "isnan(right) && !isnan(left)"
        if not largest:
            outranks, outranked = outranked, outranks
        better = f"{better} || ({outranks})"
        worse = f"{worse} || ({outranked})"
    name = kernel_name(
        context, "before", "largest" if largest else "smallest", c_type(elem_type)
    )
    return CFunction(
        name,
        _PRECEDES_TEMPLATE.substitute(
            name=name, element=c_type(elem_type), better=better, worse=worse
        ),
    )


def _split_at(shape: Sequence[int], axis: int) -> tuple[int, int, int]:
    """`shape` as the three factors an axis-wise kernel walks: before it, it, and after."""
    return math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1 :])


def _indexed_name(context: NodeContext, data: TensorRef, indices: TensorRef) -> str:
    return kernel_name(context, c_type(data.elem_type), c_type(indices.elem_type))


register_kernel("", "Gather", _GATHER_VERSIONS, _gather)
register_kernel("", "GatherElements", _GATHER_ELEMENTS_VERSIONS, _gather_elements)
register_kernel("", "GatherND", _GATHER_ND_VERSIONS, _gather_nd)
register_kernel(
    "", "TopK", _TOP_K_ATTRIBUTE_VERSIONS, partial(_top_k, from_attribute=True)
)
register_kernel(
    "", "TopK", _TOP_K_OPERAND_VERSIONS, partial(_top_k, from_attribute=False)
)
