"""Pad: the operand placed inside a larger result, and what fills the rest of it.

Every mode is the same walk — each element of the result maps to the coordinate `i - begin`
of the operand along each axis — and they differ only in what that means once the coordinate
leaves the operand: a constant value, the nearest edge, the reflection back inside, or the
wrap around to the other end. So there is one kernel per mode and element type, taking the
per-axis pads as compile-time literals.

The pads themselves have to be fixed at compile time: they place the operand inside the
result, which no shape can state on their behalf. A negative pad, which ONNX defines as
cropping instead, needs nothing of its own — it is the same coordinate map, shifted the
other way.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    kernel_name,
    normalize_axis,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents

# Pad-2 takes the pads as an attribute; from 11 they are an operand, 18 added the `axes`
# operand that says which axes they apply to, and 19 the `wrap` mode. The later revisions
# only widened the element types. Every form reads as the same per-axis pad vector, so one
# generator serves them all, told where to read the pads from.
#
# Pad-1, which spells the attribute `paddings`, is deliberately left out: ONNX's own shape
# inference derives nothing for that revision, so a node of it has no result shape to
# compile against and no oracle to prove one against either.
_ATTRIBUTE_VERSIONS = (2,)
_OPERAND_VERSIONS = (11, 13, 18, 19, 21, 23, 24, 25)

# What each mode does with a coordinate that falls outside the operand. `constant` is not
# here: it reads nothing at all, so it is a template of its own.
_MAPPINGS = {
    "edge": """\
            if (coordinate < 0) {
                coordinate = 0;
            } else if (coordinate >= extent) {
                coordinate = extent - 1;
            }""",
    "reflect": """\
            if (extent > 1) {
                const ptrdiff_t period = 2 * extent - 2;
                coordinate = ((coordinate % period) + period) % period;
                if (coordinate >= extent) {
                    coordinate = period - coordinate;
                }
            } else {
                coordinate = 0;
            }""",
    "wrap": """\
            coordinate = ((coordinate % extent) + extent) % extent;""",
}

_CONSTANT_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    $element value,
    size_t count,
    int rank,
    const size_t* shape,
    const size_t* limits,
    const ptrdiff_t* pads,
    const size_t* strides)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        size_t source = 0;
        int inside = 1;
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            const ptrdiff_t coordinate =
                (ptrdiff_t)(remainder % shape[axis]) - pads[axis];
            remainder /= shape[axis];
            if (coordinate < 0 || coordinate >= (ptrdiff_t)limits[axis]) {
                inside = 0;
            } else {
                source += (size_t)coordinate * strides[axis];
            }
        }
        out[index] = inside ? in[source] : value;
    }
}""")

_MAPPED_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t count,
    int rank,
    const size_t* shape,
    const size_t* limits,
    const ptrdiff_t* pads,
    const size_t* strides)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        size_t source = 0;
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            const ptrdiff_t extent = (ptrdiff_t)limits[axis];
            ptrdiff_t coordinate = (ptrdiff_t)(remainder % shape[axis]) - pads[axis];
            remainder /= shape[axis];
$mapping
            source += (size_t)coordinate * strides[axis];
        }
        out[index] = in[source];
    }
}""")


def _pad(context: NodeContext, *, attribute: str | None) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    mode = _mode(context)
    begins, ends = _pads(context, attribute, rank=len(source.shape))
    verify_shape(
        context,
        result,
        [
            extent + begin + end
            for extent, begin, end in zip(source.shape, begins, ends)
        ],
    )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())
    if mode != "constant":
        _verify_readable(context, source.shape, result.shape, mode)

    element = c_type(result.elem_type)
    name = kernel_name(context, mode, element)
    arguments = [
        result.expr,
        source.expr,
        *([_fill(context, attribute)] if mode == "constant" else []),
        f"{result.elem_count}u",
        str(len(result.shape)),
        extents(result.shape),
        extents(source.shape),
        _offsets(begins),
        extents(row_major_strides(source.shape)),
    ]
    definition = (
        _CONSTANT_TEMPLATE.substitute(name=name, element=element)
        if mode == "constant"
        else _MAPPED_TEMPLATE.substitute(
            name=name, element=element, mapping=_MAPPINGS[mode]
        )
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def _mode(context: NodeContext) -> str:
    value = context.attribute("mode", b"constant")
    mode = value.decode() if isinstance(value, bytes) else str(value)
    if mode == "constant" or mode in _MAPPINGS:
        return mode
    raise CompileError(
        f"Node `{context.label}`: `Pad` mode `{mode}` is not one of the modes ONNX "
        f"defines ({', '.join(['constant', *sorted(_MAPPINGS)])})."
    )


def _pads(
    context: NodeContext, attribute: str | None, *, rank: int
) -> tuple[list[int], list[int]]:
    """The pad before and after each axis, zero for the axes the node leaves out."""
    values = _pad_values(context, attribute)
    axes = _axes(context, rank)
    if len(values) != 2 * len(axes):
        raise CompileError(
            f"Node `{context.label}`: `Pad` was given {len(values)} pad(s) for "
            f"{len(axes)} axis/axes; ONNX defines two — a begin and an end — per axis."
        )
    begins = [0] * rank
    ends = [0] * rank
    seen: set[int] = set()
    for position, axis in enumerate(axes):
        if axis in seen:
            raise CompileError(
                f"Node `{context.label}`: `Pad` names axis {axis} of its operand more "
                "than once."
            )
        seen.add(axis)
        begins[axis] = values[position]
        ends[axis] = values[len(axes) + position]
    return begins, ends


def _pad_values(context: NodeContext, attribute: str | None) -> list[int]:
    if attribute is not None:
        declared = context.attribute(attribute, None)
        if declared is None:
            raise CompileError(
                f"Node `{context.label}`: `Pad` requires the `{attribute}` attribute at "
                f"opset version {context.since_version}."
            )
        return [int(value) for value in declared]
    return _constant_operand(context, 1, "pads")


def _axes(context: NodeContext, rank: int) -> list[int]:
    """Which axes the pads apply to: the ones the node names, or every one of them."""
    if context.optional_input(3) is None:
        return list(range(rank))
    return [
        normalize_axis(context, axis, rank)
        for axis in _constant_operand(context, 3, "axes")
    ]


def _fill(context: NodeContext, attribute: str | None) -> str:
    """What a constant pad is filled with, as a C expression.

    From 11 on it is an operand rather than an attribute, and one whose value decides
    nothing about any shape — so it is read at run time and needs no folding.
    """
    result = context.require_output(0)
    if attribute is not None:
        return scalar_literal(context.float_attribute("value"), result.elem_type)
    operand = context.optional_input(2)
    if operand is None:
        return scalar_literal(0, result.elem_type)
    if operand.elem_count != 1:
        raise CompileError(
            f"Node `{context.label}`: `Pad` fills with `{operand.name}`, which holds "
            f"{operand.elem_count} values; ONNX defines it as a single one."
        )
    return f"{operand.expr}[0]"


def _constant_operand(context: NodeContext, index: int, role: str) -> list[int]:
    operand = context.require_input(index)
    values = context.constant_input(index)
    if values is None:
        raise CompileError(
            f"Node `{context.label}`: `Pad` takes its {role} from `{operand.name}`, which "
            "is not known at compile time; where the operand sits inside the result — and "
            "the shape of the result itself — then depends on input data, which the C "
            "compiler cannot compile."
        )
    return [int(value) for value in values.reshape(-1)]


def _verify_readable(
    context: NodeContext, source: Sequence[int], result: Sequence[int], mode: str
) -> None:
    """Refuse a mode that would have to read a value from an axis that holds none.

    Every mode but `constant` fills the result out of the operand, so an axis the operand is
    empty along has nothing to fill a wider result with — as ONNX's own reference refuses too.
    """
    for axis, (extent, wanted) in enumerate(zip(source, result)):
        if extent == 0 and wanted > 0:
            raise CompileError(
                f"Node `{context.label}`: `Pad` in mode `{mode}` fills axis {axis} of its "
                f"result, which measures {wanted}, from an operand that is empty along it."
            )


def _offsets(values: Sequence[int]) -> str:
    """The per-axis pads as a compound literal; they are signed, since a pad may crop."""
    literals = ", ".join(str(value) for value in values) or "0"
    return f"(const ptrdiff_t[]){{{literals}}}"


register_kernel("", "Pad", _ATTRIBUTE_VERSIONS, partial(_pad, attribute="pads"))
register_kernel("", "Pad", _OPERAND_VERSIONS, partial(_pad, attribute=None))
