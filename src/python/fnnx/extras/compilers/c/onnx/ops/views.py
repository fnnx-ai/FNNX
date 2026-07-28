"""The views: the ops that rearrange a tensor's elements without computing any.

Reshape, Flatten, Squeeze and Unsqueeze only relabel the axes of a row-major buffer, so each
of them is a copy of it. Transpose, Concat, Split, Slice, Expand, Tile and the two block
shuffles — DepthToSpace and SpaceToDepth — do reorder the elements, and all of them the same
way: every element of the result is read from one element of the operand, at an offset that
is a fixed base plus a stride per axis times that axis's coordinate. One shared kernel walks
that addressing, and the ops differ only in the strides and bases they hand it — compile-time
literals, all of them. Where both sides come out contiguous the kernel is skipped and the
move is a single `memcpy`.

`Identity` is emitted alongside the elementwise family, and `Shape` and `Size` need no kernel
at all: their output follows from a shape the compiler has already made static, so constant
folding resolves them long before dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    broadcast_strides,
    copy_tensor,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    normalize_axis,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents

# None of the ops below read the operand that decides their result's shape — the shape ONNX
# inferred is what every buffer and every stride is derived from — so a revision that only
# moved that operand between an attribute and an input, or widened the types it accepts,
# emits the identical code and is served by the same generator.
#
# Reshape moved `shape` to an input at 5 and gained `allowzero` at 14; Squeeze and Unsqueeze
# moved `axes` to an input at 13; Flatten, Transpose and Concat kept their attributes
# throughout. Concat-1 is the one revision left out: it defaulted `axis` to 1 while its
# schema reports no default at all, so the value a node omitting the attribute means cannot
# be read off ONNX itself. From 4 on the attribute is required.
_RESHAPE_VERSIONS = (1, 5, 13, 14, 19, 21, 23, 24, 25)
_FLATTEN_VERSIONS = (1, 9, 11, 13, 21, 23, 24, 25)
_SQUEEZE_VERSIONS = (1, 11, 13, 21, 23, 24, 25)
_UNSQUEEZE_VERSIONS = (1, 11, 13, 21, 23, 24, 25)
_TRANSPOSE_VERSIONS = (1, 13, 21, 23, 24, 25)
_CONCAT_VERSIONS = (4, 11, 13)
_SPLIT_VERSIONS = (1, 2, 11, 13, 18)
_EXPAND_VERSIONS = (8, 13)

# Slice-1 takes its bounds as attributes and has no `steps`; from 10 on they are all
# operands. That is two ways of reading the same slice, so it is two generators.
_SLICE_ATTRIBUTE_VERSIONS = (1,)
_SLICE_OPERAND_VERSIONS = (10, 11, 13)

# Tile-1 is deliberately absent: its second and third operands are a repeat count and the
# single axis to apply it to, not the per-axis repeat vector every revision since 6 takes.
_TILE_VERSIONS = (6, 13)

# DepthToSpace gained its `mode` at 11 and SpaceToDepth never changed; 13 widened the types
# of both. Only 13 is claimed for either: it is the revision the reference evaluator is
# version-faithful for and the one both corpus tests import, so it is the only one anything
# can vouch for. A model importing an older one gets the unsupported-version error.
_BLOCK_VERSIONS = (13,)

_COPY_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t count,
    int rank,
    const size_t* shape,
    const ptrdiff_t* out_strides,
    const ptrdiff_t* in_strides,
    ptrdiff_t out_base,
    ptrdiff_t in_base)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        ptrdiff_t source = in_base;
        ptrdiff_t target = out_base;
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            const size_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            source += (ptrdiff_t)coordinate * in_strides[axis];
            target += (ptrdiff_t)coordinate * out_strides[axis];
        }
        out[target] = in[source];
    }
}""")


def copy_elements(
    context: NodeContext,
    *,
    source: TensorRef,
    result: TensorRef,
    shape: Sequence[int],
    source_strides: Sequence[int],
    result_strides: Sequence[int] | None = None,
    source_base: int = 0,
    result_base: int = 0,
) -> NodeEmission:
    """Move a block of `shape` elements, each side addressed by its own strides and base.

    `result_strides` defaults to the row-major strides of `shape`, which is what an op
    writing the whole of its result in order needs; Concat, which writes a slice of one,
    passes the result's own strides instead. A move that comes out contiguous on both sides
    is emitted as a `memcpy`, since the kernel would then walk it element by element to no
    end.
    """
    count = math.prod(shape)
    if count == 0:
        return NodeEmission(functions=(), statements=())
    strides = row_major_strides(shape) if result_strides is None else result_strides
    if _is_contiguous(shape, source_strides) and _is_contiguous(shape, strides):
        return NodeEmission(
            functions=(),
            statements=(
                f"memcpy({_at(result.expr, result_base)}, "
                f"{_at(source.expr, source_base)}, "
                f"{count}u * sizeof(*{result.expr}));",
            ),
        )
    element = c_type(result.elem_type)
    name = f"{context.prefix}_copy_{element}"
    return NodeEmission(
        functions=(
            CFunction(name, _COPY_TEMPLATE.substitute(name=name, element=element)),
        ),
        statements=(
            call_kernel(
                name,
                [
                    result.expr,
                    source.expr,
                    f"{count}u",
                    str(len(shape)),
                    extents(shape),
                    _offsets(strides),
                    _offsets(source_strides),
                    str(result_base),
                    str(source_base),
                ],
            ),
        ),
    )


def _is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether walking `shape` under `strides` visits one unbroken run, in order.

    An axis of a single element is skipped: its stride multiplies a coordinate that is only
    ever zero, so whatever it holds cannot break the run.
    """
    expected = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent != 1 and stride != expected:
            return False
        expected *= extent
    return True


def _at(expr: str, offset: int) -> str:
    return expr if offset == 0 else f"{expr} + {offset}"


def _offsets(values: Sequence[int]) -> str:
    """Strides as a compound literal; they are signed, since a slice may walk backwards."""
    literals = ", ".join(str(value) for value in values) or "0"
    return f"(const ptrdiff_t[]){{{literals}}}"


def _combined(emissions: Sequence[NodeEmission]) -> NodeEmission:
    """One emission out of the several an op writing block by block contributes."""
    functions = {
        function.name: function
        for emission in emissions
        for function in emission.functions
    }
    return NodeEmission(
        functions=tuple(functions.values()),
        statements=tuple(
            statement for emission in emissions for statement in emission.statements
        ),
    )


# --------------------------------------------------------------------------------------
# The ops
# --------------------------------------------------------------------------------------


def _relabel(context: NodeContext) -> NodeEmission:
    """Reshape, Flatten, Squeeze, Unsqueeze: the same row-major buffer under other axes."""
    source = context.require_input(0)
    result = context.require_output(0)
    if source.elem_count != result.elem_count:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads "
            f"{source.elem_count} element(s) from `{source.name}` but writes "
            f"{result.elem_count} to `{result.name}`; this op only relabels axes."
        )
    return copy_tensor(source, result)


def _transpose(context: NodeContext) -> NodeEmission:
    """Transpose: the operand read along permuted axes, written out in order."""
    source = context.require_input(0)
    result = context.require_output(0)
    rank = len(source.shape)
    declared = context.attribute("perm", None)
    perm = (
        tuple(reversed(range(rank)))
        if declared is None
        else tuple(int(axis) for axis in declared)
    )
    if sorted(perm) != list(range(rank)):
        raise CompileError(
            f"Node `{context.label}`: `perm` {list(perm)} is not a permutation of the "
            f"{rank} axes of `{source.name}`."
        )
    verify_shape(context, result, [source.shape[axis] for axis in perm])
    strides = row_major_strides(source.shape)
    return copy_elements(
        context,
        source=source,
        result=result,
        shape=result.shape,
        source_strides=[strides[axis] for axis in perm],
    )


def _concat(context: NodeContext) -> NodeEmission:
    """Concat: each operand written into its own band of the result along one axis."""
    result = context.require_output(0)
    operands = [
        context.require_input(index) for index in range(len(context.node.input))
    ]
    axis = normalize_axis(context, context.int_attribute("axis"), len(result.shape))
    _verify_bands(context, operands, result, axis)
    strides = row_major_strides(result.shape)
    emissions = []
    offset = 0
    for operand in operands:
        emissions.append(
            copy_elements(
                context,
                source=operand,
                result=result,
                shape=operand.shape,
                source_strides=row_major_strides(operand.shape),
                result_strides=strides,
                result_base=offset * strides[axis],
            )
        )
        offset += operand.shape[axis]
    return _combined(emissions)


def _split(context: NodeContext) -> NodeEmission:
    """Split: consecutive bands of the operand along one axis, each its own result.

    How wide each band is comes from the shape ONNX inferred for it — which is what the
    `split` operand, the `num_outputs` attribute and the equal division of neither all
    ultimately say — so every revision of the op is read the same way here.
    """
    source = context.require_input(0)
    results = [
        context.require_output(index) for index in range(len(context.node.output))
    ]
    axis = normalize_axis(context, context.int_attribute("axis"), len(source.shape))
    _verify_bands(context, results, source, axis)
    strides = row_major_strides(source.shape)
    emissions = []
    offset = 0
    for result in results:
        emissions.append(
            copy_elements(
                context,
                source=source,
                result=result,
                shape=result.shape,
                source_strides=strides,
                source_base=offset * strides[axis],
            )
        )
        offset += result.shape[axis]
    return _combined(emissions)


def _verify_bands(
    context: NodeContext,
    bands: Sequence[TensorRef],
    joined: TensorRef,
    axis: int,
) -> None:
    """Refuse to emit a split or a join whose bands do not tile the whole tensor.

    Every band has to match the joined tensor on every axis but `axis`, and their extents
    along `axis` have to add up to its own; anything else is a compiler bug that would read
    or write outside a buffer.
    """
    for band in bands:
        expected = list(joined.shape)
        if len(band.shape) == len(expected):
            expected[axis] = band.shape[axis]
        if band.shape != tuple(expected):
            raise CompileError(
                f"Node `{context.label}`: `{context.node.op_type}` joins `{band.name}` of "
                f"shape {list(band.shape)} along axis {axis} of a tensor of shape "
                f"{list(joined.shape)}; the two have to agree on every other axis."
            )
    total = sum(band.shape[axis] for band in bands)
    if total != joined.shape[axis]:
        raise CompileError(
            f"Node `{context.label}`: the bands of `{context.node.op_type}` measure "
            f"{total} along axis {axis}, but `{joined.name}` measures "
            f"{joined.shape[axis]}."
        )


# What a Slice revision reads its bounds from: the starts, ends, axes and steps, where None
# stands for a list the node leaves out.
SliceBounds = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, tuple[int, ...] | None
]


def _attribute_bounds(context: NodeContext) -> SliceBounds:
    """Slice-1, whose bounds are attributes and which has no steps at all."""
    return (
        _required_attribute(context, "starts"),
        _required_attribute(context, "ends"),
        _optional_attribute(context, "axes"),
        None,
    )


def _operand_bounds(context: NodeContext) -> SliceBounds:
    """Slice-10 and later, whose bounds are operands the graph has to fix."""
    return (
        _constant_operand(context, 1),
        _constant_operand(context, 2),
        _optional_operand(context, 3),
        _optional_operand(context, 4),
    )


def _required_attribute(context: NodeContext, name: str) -> tuple[int, ...]:
    values = context.attribute(name, None)
    if values is None:
        raise CompileError(
            f"Node `{context.label}`: `Slice` requires the `{name}` attribute at opset "
            f"version {context.since_version}."
        )
    return tuple(int(value) for value in values)


def _optional_attribute(context: NodeContext, name: str) -> tuple[int, ...] | None:
    values = context.attribute(name, None)
    return None if values is None else tuple(int(value) for value in values)


def _constant_operand(context: NodeContext, index: int) -> tuple[int, ...]:
    operand = context.require_input(index)
    values = context.constant_input(index)
    if values is None:
        raise CompileError(
            f"Node `{context.label}`: `Slice` takes its bounds from `{operand.name}`, "
            "which is not known at compile time; the shape of the result then depends on "
            "input data, which the C compiler cannot compile."
        )
    return tuple(int(value) for value in values.reshape(-1))


def _optional_operand(context: NodeContext, index: int) -> tuple[int, ...] | None:
    operand = context.optional_input(index)
    return None if operand is None else _constant_operand(context, index)


def _slice(
    context: NodeContext, *, bounds: Callable[[NodeContext], SliceBounds]
) -> NodeEmission:
    """Slice: the operand walked from a per-axis start, by a per-axis step.

    A step is a stride multiplier and a start an offset into the operand, so the whole of the
    op is addressing. The bounds are clamped the way Python's own slice does it, which is
    what numpy — and through it the ONNX reference evaluator — applies; the extents that come
    out are checked against the shape ONNX inferred rather than trusted.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    rank = len(source.shape)
    starts, ends, axes, steps = bounds(context)
    if axes is None:
        axes = tuple(range(len(starts)))
    if steps is None:
        steps = (1,) * len(starts)
    if not len(starts) == len(ends) == len(axes) == len(steps):
        raise CompileError(
            f"Node `{context.label}`: `Slice` was given {len(starts)} start(s), "
            f"{len(ends)} end(s), {len(axes)} axis/axes and {len(steps)} step(s); ONNX "
            "defines one of each per sliced axis."
        )

    strides = row_major_strides(source.shape)
    walk = list(strides)
    sliced = list(source.shape)
    base = 0
    seen: set[int] = set()
    for start, end, step, axis in zip(starts, ends, steps, axes):
        resolved = normalize_axis(context, axis, rank)
        if resolved in seen:
            raise CompileError(
                f"Node `{context.label}`: `Slice` names axis {resolved} of "
                f"`{source.name}` more than once."
            )
        seen.add(resolved)
        if step == 0:
            raise CompileError(
                f"Node `{context.label}`: `Slice` steps by 0 along axis {resolved}, which "
                "ONNX does not define."
            )
        first, stop, stride = slice(start, end, step).indices(source.shape[resolved])
        sliced[resolved] = len(range(first, stop, stride))
        walk[resolved] = strides[resolved] * stride
        base += first * strides[resolved]

    verify_shape(context, result, sliced)
    return copy_elements(
        context,
        source=source,
        result=result,
        shape=result.shape,
        source_strides=walk,
        source_base=base,
    )


def _expand(context: NodeContext) -> NodeEmission:
    """Expand: the operand stretched onto the shape it broadcasts with.

    A stretched axis is a stride of zero, so every coordinate along it reads the same
    element; which axes those are follows from the shape ONNX inferred for the result, the
    same shape the `shape` operand had to be constant to produce.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    return copy_elements(
        context,
        source=source,
        result=result,
        shape=result.shape,
        source_strides=broadcast_strides(
            source, result.shape, node_label=context.label
        ),
    )


def _tile(context: NodeContext) -> NodeEmission:
    """Tile: the operand repeated a given number of times along each of its axes.

    A tiling is a broadcast in disguise. Splitting every result axis into its repeat count
    and the operand's own extent gives a tensor of rank 2n whose row-major order is exactly
    the result's, and over which the operand is simply stretched along the repeat axes — so
    the same strided move serves, with a stride of zero on each of them.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    repeats = context.constant_input(1)
    if repeats is None:
        raise CompileError(
            f"Node `{context.label}`: `Tile` takes its repeat counts from "
            f"`{context.require_input(1).name}`, which is not known at compile time; the "
            "shape of the result then depends on input data, which the C compiler cannot "
            "compile."
        )
    counts = tuple(int(count) for count in repeats.reshape(-1))
    if len(counts) != len(source.shape):
        raise CompileError(
            f"Node `{context.label}`: `Tile` was given {len(counts)} repeat count(s) for "
            f"the {len(source.shape)} axes of `{source.name}`; ONNX defines one per axis."
        )
    verify_shape(
        context,
        result,
        [extent * count for extent, count in zip(source.shape, counts)],
    )

    strides = row_major_strides(source.shape)
    interleaved: list[int] = []
    source_strides: list[int] = []
    for axis, count in enumerate(counts):
        interleaved += [count, source.shape[axis]]
        source_strides += [0, strides[axis]]
    return copy_elements(
        context,
        source=source,
        result=result,
        shape=interleaved,
        source_strides=source_strides,
    )


def _block_shuffle(
    context: NodeContext, view: Sequence[int], perm: Sequence[int]
) -> NodeEmission:
    """The operand split into blocks by `view`, then transposed by `perm`.

    Both `DepthToSpace` and `SpaceToDepth` are defined as exactly that — a reshape, a
    transpose and a reshape back — and the two reshapes are free: `view`'s row-major order is
    the operand's own, and the transposed extents' row-major order is the result's. So only
    the transpose is emitted, through the same strided move every view op runs.
    """
    return copy_elements(
        context,
        source=context.require_input(0),
        result=context.require_output(0),
        shape=[view[axis] for axis in perm],
        source_strides=[row_major_strides(view)[axis] for axis in perm],
    )


def _depth_to_space(context: NodeContext) -> NodeEmission:
    """DepthToSpace: each channel of a block spread over one position of a spatial block.

    `mode` says how the channels are grouped before they are spread: `DCR` reads the block's
    rows and columns as the outermost channel axes and the surviving depth as the innermost,
    `CRD` the other way round.
    """
    batch, channels, rows, columns = _image(context)
    block = _blocksize(context)
    if channels % (block * block) != 0:
        raise CompileError(
            f"Node `{context.label}`: `DepthToSpace` spreads {channels} channel(s) over "
            f"{block}x{block} positions, which does not divide them evenly."
        )
    depth = channels // (block * block)
    verify_shape(
        context,
        context.require_output(0),
        (batch, depth, rows * block, columns * block),
    )
    mode = context.attribute("mode", b"DCR")
    mode = mode.decode() if isinstance(mode, bytes) else str(mode)
    if mode not in ("DCR", "CRD"):
        raise CompileError(
            f"Node `{context.label}`: `DepthToSpace` asks for `mode` `{mode}`, which is "
            "not one of the modes ONNX defines (`DCR`, `CRD`)."
        )
    if mode == "DCR":
        return _block_shuffle(
            context, (batch, block, block, depth, rows, columns), (0, 3, 4, 1, 5, 2)
        )
    return _block_shuffle(
        context, (batch, depth, block, block, rows, columns), (0, 1, 4, 2, 5, 3)
    )


def _space_to_depth(context: NodeContext) -> NodeEmission:
    """SpaceToDepth: each position of a spatial block moved into a channel of its own."""
    batch, channels, rows, columns = _image(context)
    block = _blocksize(context)
    if rows % block != 0 or columns % block != 0:
        raise CompileError(
            f"Node `{context.label}`: `SpaceToDepth` splits a {rows}x{columns} image into "
            f"{block}x{block} blocks, which do not tile it evenly."
        )
    verify_shape(
        context,
        context.require_output(0),
        (batch, channels * block * block, rows // block, columns // block),
    )
    return _block_shuffle(
        context,
        (batch, channels, rows // block, block, columns // block, block),
        (0, 3, 5, 1, 2, 4),
    )


def _image(context: NodeContext) -> tuple[int, int, int, int]:
    source = context.require_input(0)
    if len(source.shape) != 4:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes a batch of images — a "
            f"tensor of rank 4 — but `{source.name}` has shape {list(source.shape)}."
        )
    batch, channels, rows, columns = source.shape
    return batch, channels, rows, columns


def _blocksize(context: NodeContext) -> int:
    block = context.attribute("blocksize", None)
    if block is None:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` states no `blocksize`, "
            "which ONNX defines as a required attribute."
        )
    if int(block) < 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` was given `blocksize` "
            f"{int(block)}; ONNX defines it as the extent of a block, which is positive."
        )
    return int(block)


register_kernel("", "Reshape", _RESHAPE_VERSIONS, _relabel)
register_kernel("", "Flatten", _FLATTEN_VERSIONS, _relabel)
register_kernel("", "Squeeze", _SQUEEZE_VERSIONS, _relabel)
register_kernel("", "Unsqueeze", _UNSQUEEZE_VERSIONS, _relabel)
register_kernel("", "Transpose", _TRANSPOSE_VERSIONS, _transpose)
register_kernel("", "Concat", _CONCAT_VERSIONS, _concat)
register_kernel("", "Split", _SPLIT_VERSIONS, _split)
register_kernel(
    "", "Slice", _SLICE_ATTRIBUTE_VERSIONS, partial(_slice, bounds=_attribute_bounds)
)
register_kernel(
    "", "Slice", _SLICE_OPERAND_VERSIONS, partial(_slice, bounds=_operand_bounds)
)
register_kernel("", "Expand", _EXPAND_VERSIONS, _expand)
register_kernel("", "Tile", _TILE_VERSIONS, _tile)
register_kernel("", "DepthToSpace", _BLOCK_VERSIONS, _depth_to_space)
register_kernel("", "SpaceToDepth", _BLOCK_VERSIONS, _space_to_depth)
