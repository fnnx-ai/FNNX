"""The sliding window's geometry, shared by every op that slides one.

A convolution and a pooling place the same window over the same axes: `strides` move it,
`dilations` spread its taps, and `pads` or `auto_pad` place it against the operand's edges.
So the attributes are read and resolved once here — into the extents, steps and pads a kernel
walks with — and each family layers on what only it has: a filter's shape and the backward
walk for the convolutions, a `kernel_shape` and a `ceil_mode` for the poolings.
"""

from __future__ import annotations

from collections.abc import Sequence

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.kernels import NodeContext

AUTO_PAD_MODES = ("NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID")


def auto_pad_mode(context: NodeContext) -> str:
    value = context.attribute("auto_pad", b"NOTSET")
    mode = value.decode() if isinstance(value, bytes) else str(value)
    if mode not in AUTO_PAD_MODES:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` `auto_pad` `{mode}` is not "
            f"one of the modes ONNX defines ({', '.join(AUTO_PAD_MODES)})."
        )
    return mode


def spatial_extents(
    context: NodeContext, name: str, rank: int, *, minimum: int = 1
) -> tuple[int, ...] | None:
    """The node's `name` attribute as one value per spatial axis, or None when absent."""
    declared = context.attribute(name, None)
    if declared is None:
        return None
    values = tuple(int(value) for value in declared)
    if len(values) != rank:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` was given {len(values)} "
            f"`{name}` for {rank} spatial axis/axes."
        )
    if any(value < minimum for value in values):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` was given `{name}` "
            f"{list(values)}; ONNX defines them as "
            f"{'positive' if minimum > 0 else 'nonnegative'}."
        )
    return values


def spatial_attribute(
    context: NodeContext, name: str, rank: int, default: int, *, minimum: int = 1
) -> tuple[int, ...]:
    values = spatial_extents(context, name, rank, minimum=minimum)
    return (default,) * rank if values is None else values


def declared_pads(
    context: NodeContext, rank: int, mode: str
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """The `pads` attribute split into begins and ends, or None when the node omits it."""
    declared = context.attribute("pads", None)
    if declared is None:
        return None
    if mode != "NOTSET":
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` states both `auto_pad` "
            f"`{mode}` and explicit `pads`, which ONNX defines as mutually exclusive."
        )
    values = tuple(int(value) for value in declared)
    if len(values) != 2 * rank:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` was given {len(values)} "
            f"pad(s) for {rank} spatial axis/axes; ONNX defines two — a begin and an end "
            "— per axis."
        )
    return values[:rank], values[rank:]


def resolve_pads(
    context: NodeContext,
    input_shape: Sequence[int],
    window_shape: Sequence[int],
    dilations: Sequence[int],
    strides: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The pad before and after each spatial axis, after resolving `auto_pad`."""
    mode = auto_pad_mode(context)
    rank = len(input_shape)
    declared = declared_pads(context, rank, mode)
    if declared is not None:
        return declared
    if mode in ("NOTSET", "VALID"):
        return (0,) * rank, (0,) * rank

    begins, ends = [], []
    for extent, window, dilation, stride in zip(
        input_shape, window_shape, dilations, strides
    ):
        # ONNX pads so the result measures `ceil(extent / stride)`, which puts the last
        # window's start one stride before the end of the axis — or `extent % stride`
        # before it, where the stride does not divide the extent — and pads whatever of
        # the window's dilated reach then hangs off the end.
        residual = extent % stride
        reach = (window - 1) * dilation + 1
        total = max(reach - (stride if residual == 0 else residual), 0)
        smaller, larger = total // 2, total - total // 2
        begins.append(smaller if mode == "SAME_UPPER" else larger)
        ends.append(larger if mode == "SAME_UPPER" else smaller)
    return tuple(begins), tuple(ends)


def output_extents(
    input_shape: Sequence[int],
    window_shape: Sequence[int],
    dilations: Sequence[int],
    strides: Sequence[int],
    begins: Sequence[int],
    ends: Sequence[int],
) -> tuple[int, ...]:
    """Result extents of a forward walk: the window positions that fit each axis."""
    return tuple(
        (extent + begin + end - (window - 1) * dilation - 1) // stride + 1
        for extent, begin, end, window, dilation, stride in zip(
            input_shape, begins, ends, window_shape, dilations, strides
        )
    )


def offsets(values: Sequence[int]) -> str:
    """The per-axis pads as a compound literal; they are signed, since a pad may crop."""
    literals = ", ".join(str(value) for value in values) or "0"
    return f"(const ptrdiff_t[]){{{literals}}}"
