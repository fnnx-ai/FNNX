"""Grouping a tensor by axes: the addressing every axis-wise kernel is emitted from.

A reduction, a softmax and a cumulative sum all walk the same two nested loops: one over the
groups the axes an op does *not* name leave behind, and one over the elements of a group.
This module turns a static shape and a set of axes into the compile-time literals those loops
take — an extent and a stride per axis — and emits the one helper that turns a loop counter
back into an offset into the tensor's row-major buffer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.emit import INVALID_ARGUMENT_STATUS
from fnnx.extras.compilers.c.onnx.kernels import CFunction, NodeContext, TensorRef
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents

# The parameters every axis-wise kernel takes after its buffers: the two loop bounds, and the
# extents and strides that address each loop's coordinates.
GROUP_PARAMETERS = """\
    size_t group_count,
    size_t group_size,
    int kept_rank,
    const size_t* kept_shape,
    const size_t* kept_strides,
    int reduced_rank,
    const size_t* reduced_shape,
    const size_t* reduced_strides"""

# The offset of a group's first element, and of an element within a group; both decompose a
# linear counter into coordinates, which is what `$offset` is for.
GROUP_BASE = "$offset(group, kept_rank, kept_shape, kept_strides)"
GROUP_ELEMENT = "$offset($index, reduced_rank, reduced_shape, reduced_strides)"

_OFFSET_TEMPLATE = Template("""\
static size_t $name(
    size_t index,
    int rank,
    const size_t* shape,
    const size_t* strides)
{
    size_t offset = 0;
    int axis;
    for (axis = rank - 1; axis >= 0; --axis) {
        offset += (index % shape[axis]) * strides[axis];
        index /= shape[axis];
    }
    return offset;
}""")


@dataclass(frozen=True)
class Grouping:
    """A tensor's axes split into the ones an op runs across and the ones it runs along.

    Each entry is the axis's extent and its stride into the tensor's row-major buffer, so a
    kernel addresses the original tensor without any of it being copied or transposed.
    `kept_axes` and `reduced_axes` are the positions those entries came from, which is what
    an operand addressed from the same coordinates — a normalization's scale — is split by.
    """

    kept: tuple[tuple[int, int], ...]
    reduced: tuple[tuple[int, int], ...]
    kept_axes: tuple[int, ...]
    reduced_axes: tuple[int, ...]

    @property
    def group_count(self) -> int:
        return math.prod(extent for extent, _ in self.kept)

    @property
    def group_size(self) -> int:
        return math.prod(extent for extent, _ in self.reduced)

    @property
    def arguments(self) -> list[str]:
        """Call-site literals for `GROUP_PARAMETERS`, in order."""
        return [
            f"{self.group_count}u",
            f"{self.group_size}u",
            *_axis_arguments(self.kept),
            *_axis_arguments(self.reduced),
        ]


def group_axes(shape: Sequence[int], axes: Sequence[int]) -> Grouping:
    """Split `shape` into the axes `axes` names and the ones it leaves."""
    strides = row_major_strides(shape)
    named = frozenset(axes)
    kept_axes = tuple(axis for axis in range(len(shape)) if axis not in named)
    return Grouping(
        kept=tuple((shape[axis], strides[axis]) for axis in kept_axes),
        reduced=tuple((shape[axis], strides[axis]) for axis in axes),
        kept_axes=kept_axes,
        reduced_axes=tuple(axes),
    )


def row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def normalize_axis(context: NodeContext, axis: int, rank: int) -> int:
    """An ONNX axis, which may count from the end, as an index into a rank-`rank` tensor."""
    resolved = axis + rank if axis < 0 else axis
    if not 0 <= resolved < rank:
        raise CompileError(
            f"Node `{context.label}`: axis {axis} is out of range for the rank-{rank} "
            f"tensor `{context.require_input(0).name}`."
        )
    return resolved


def normalize_axes(
    context: NodeContext, axes: Sequence[int], rank: int
) -> tuple[int, ...]:
    """The axes an op names, resolved and sorted so a kernel walks them in memory order."""
    resolved = tuple(normalize_axis(context, int(axis), rank) for axis in axes)
    if len(set(resolved)) != len(resolved):
        raise CompileError(
            f"Node `{context.label}`: axes {[int(axis) for axis in axes]} name the same "
            "dimension more than once."
        )
    return tuple(sorted(resolved))


def verify_group_count(
    context: NodeContext, grouping: Grouping, result: TensorRef
) -> None:
    """Refuse to emit a kernel that would write past the result buffer.

    The groups are counted from the operand's shape and the axes the node names, while the
    buffer is sized from the shape ONNX inferred for the result; a disagreement between the
    two is a compiler bug, and this is where it stops rather than where it corrupts memory.
    """
    if grouping.group_count != result.elem_count:
        raise CompileError(
            f"Node `{context.label}`: the axes it names leave {grouping.group_count} "
            f"group(s), but its output `{result.name}` holds {result.elem_count} element(s)."
        )


def verify_shape(
    context: NodeContext, result: TensorRef, expected: Sequence[int]
) -> None:
    """Refuse to emit a kernel whose addressing disagrees with the buffer it writes.

    The extents are derived from the operands and the node's own attributes, while the buffer
    is sized from the shape ONNX inferred for the result; a disagreement between the two is a
    compiler bug, and this is where it stops rather than where it corrupts memory.
    """
    if result.shape != tuple(expected):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` addresses a result of shape "
            f"{list(expected)}, but its output `{result.name}` holds "
            f"{list(result.shape)}."
        )


def verify_same_shape(
    context: NodeContext, source: TensorRef, result: TensorRef
) -> None:
    """Refuse to emit an axis-wise kernel whose two buffers are not laid out alike.

    Softmax and the cumulative folds write a result of the operand's own shape, so one
    grouping addresses both; a disagreement is a compiler bug, and this is where it stops.
    """
    if source.shape != result.shape:
        raise CompileError(
            f"Node `{context.label}`: `{source.name}` has shape {list(source.shape)} but "
            f"its result `{result.name}` has shape {list(result.shape)}; this op leaves "
            "the shape alone."
        )


def offset_helper(prefix: str) -> CFunction:
    """The shared index-to-offset function every axis-wise kernel calls."""
    name = f"{prefix}_axis_offset"
    return CFunction(name, _OFFSET_TEMPLATE.substitute(name=name))


def kernel_name(context: NodeContext, *parts: str) -> str:
    """A kernel name encoding the op and everything else its code depends on."""
    return "_".join((context.prefix, context.node.op_type.lower(), *parts))


def call_kernel(name: str, arguments: Sequence[str]) -> str:
    return f"{name}(\n    " + ",\n    ".join(arguments) + ");"


def checked_call(context: NodeContext, name: str, arguments: Sequence[str]) -> str:
    """A call site for a kernel that validates an operand's values at run time.

    Such a kernel returns nonzero for a value ONNX leaves undefined — an index outside the
    axis it addresses — and the entrypoint passes that on as the argument error the status
    enum exists for, rather than reading past a buffer.
    """
    call = call_kernel(name, arguments).rstrip(";")
    return "\n".join(
        [
            f"if ({call} != 0) {{",
            f"    return {context.prefix.upper()}_{INVALID_ARGUMENT_STATUS};",
            "}",
        ]
    )


def _axis_arguments(axes: Sequence[tuple[int, int]]) -> list[str]:
    return [
        str(len(axes)),
        extents([extent for extent, _ in axes]),
        extents([stride for _, stride in axes]),
    ]
