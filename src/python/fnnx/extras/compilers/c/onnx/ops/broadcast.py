"""The loop every elementwise kernel is emitted from.

An elementwise op is a C expression over one scalar per operand: this module turns that
expression into a shared `static` kernel and the call site invoking it. Operands broadcast
onto the result's shape numpy-style, addressed through strides that are zero on every axis
they are stretched along; when every operand already has the result's shape — the common
case — the same expression is emitted into a flat loop that pays no index arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.onnx.dtypes import FLOAT_TYPES, c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    broadcast_strides,
)

_POINTWISE_TEMPLATE = Template("""\
static void $name(
$parameters,
    size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index) {
$reads
        out[index] = $expression;
    }
}""")

_BROADCAST_TEMPLATE = Template("""\
static void $name(
$parameters,
    size_t count,
    int rank,
    const size_t* shape,
$stride_parameters)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
$offsets
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            size_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
$accumulate
        }
$reads
        out[index] = $expression;
    }
}""")

# numpy's `minimum`/`maximum`, which ONNX's Min, Max, Clip, HardSigmoid and Celu are all
# defined through: NaN wins over anything, and everything else that is not strictly better
# than the second operand yields the second operand — which is what makes `minimum(-0, 0)`
# come out `+0` and `minimum(0, -0)` come out `-0`. C's `fmin`/`fmax` do the opposite with
# NaN, so they are never used.
_COMBINE_TEMPLATE = Template("""\
static $element $name($element left, $element right)
{
    return ($test) ? left : right;
}""")


@dataclass(frozen=True)
class Scalar:
    """An attribute value a kernel reads as a parameter rather than as an inlined literal.

    Kernels are shared across nodes, so the values that differ between two nodes running the
    same op — Elu's alpha, Shrink's bias — have to arrive as arguments, or every distinct
    attribute value would emit a kernel of its own.
    """

    name: str
    elem_type: int
    value: float | int


def elementwise(
    context: NodeContext,
    *,
    expression: str,
    operands: Sequence[TensorRef],
    result: TensorRef,
    scalars: Sequence[Scalar] = (),
    helpers: Sequence[CFunction] = (),
    variant: str = "",
) -> NodeEmission:
    """A kernel writing `expression` into every element of `result`.

    `expression` is C over the locals `x0`, `x1`, ... — one per operand, of that operand's
    element type — and over the `scalars`' names. `variant` distinguishes kernels whose code
    differs for a reason the operand types do not capture, such as an attribute that selects
    between two formulas; `helpers` are functions the expression calls, emitted first.
    """
    aligned = all(operand.shape == result.shape for operand in operands)
    name = _kernel_name(context, operands, result, variant, aligned)
    parameters = [f"    {c_type(result.elem_type)}* out"]
    parameters += [
        f"    const {c_type(operand.elem_type)}* in{index}"
        for index, operand in enumerate(operands)
    ]
    parameters += [
        f"    {c_type(scalar.elem_type)} {scalar.name}" for scalar in scalars
    ]
    arguments = [result.expr, *(operand.expr for operand in operands)]
    arguments += [scalar_literal(scalar.value, scalar.elem_type) for scalar in scalars]

    if aligned:
        definition = _POINTWISE_TEMPLATE.substitute(
            name=name,
            parameters=",\n".join(parameters),
            reads=_reads(operands, lambda index: "index"),
            expression=expression,
        )
        call = _call(name, [*arguments, f"{result.elem_count}u"])
    else:
        strides = [
            broadcast_strides(operand, result.shape, node_label=context.label)
            for operand in operands
        ]
        definition = _BROADCAST_TEMPLATE.substitute(
            name=name,
            parameters=",\n".join(parameters),
            stride_parameters=",\n".join(
                f"    const size_t* strides{index}" for index in range(len(operands))
            ),
            offsets="\n".join(
                f"        size_t offset{index} = 0;" for index in range(len(operands))
            ),
            accumulate="\n".join(
                f"            offset{index} += coordinate * strides{index}[axis];"
                for index in range(len(operands))
            ),
            reads=_reads(operands, lambda index: f"offset{index}"),
            expression=expression,
        )
        call = _call(
            name,
            [
                *arguments,
                f"{result.elem_count}u",
                str(len(result.shape)),
                extents(result.shape),
                *(extents(stride) for stride in strides),
            ],
        )
    return NodeEmission(
        functions=(*helpers, CFunction(name, definition)), statements=(call,)
    )


def pointwise(
    context: NodeContext,
    template: str,
    *,
    scalars: Sequence[Scalar] = (),
    helpers: Sequence[CFunction] = (),
    variant: str = "",
) -> NodeEmission:
    """A kernel for a one-operand op, whose formula is `template` over the local `x0`."""
    result = context.require_output(0)
    return elementwise(
        context,
        expression=expand(template, result.elem_type),
        operands=(context.require_input(0),),
        result=result,
        scalars=scalars,
        helpers=helpers,
        variant=variant,
    )


def expand(template: str, elem_type: int) -> str:
    """Fill in what a kernel expression takes from its element type.

    `$f` is the libm suffix, `$one` and `$zero` are literals of that type, and `$element`
    is its C type.
    """
    return Template(template).substitute(
        f=math_suffix(elem_type),
        one=scalar_literal(1, elem_type),
        zero=scalar_literal(0, elem_type),
        element=c_type(elem_type),
    )


def combiner(context: NodeContext, elem_type: int, *, largest: bool) -> CFunction:
    """numpy's `minimum` or `maximum` at `elem_type`, as a function to call per element."""
    comparison = ">" if largest else "<"
    test = f"left {comparison} right"
    if elem_type in FLOAT_TYPES:
        test = f"{test} || isnan(left)"
    name = f"{context.prefix}_{'maximum' if largest else 'minimum'}_{c_type(elem_type)}"
    return CFunction(
        name,
        _COMBINE_TEMPLATE.substitute(name=name, element=c_type(elem_type), test=test),
    )


def extents(values: Sequence[int]) -> str:
    """Shapes and strides as a compound literal; rank 0 gets an unread placeholder."""
    literals = ", ".join(f"{value}u" for value in values) or "0u"
    return f"(const size_t[]){{{literals}}}"


def math_suffix(elem_type: int) -> str:
    """The libm suffix selecting the overload for this element type: `sinf` against `sin`."""
    return "" if elem_type == TensorProto.DOUBLE else "f"


def _reads(operands: Sequence[TensorRef], offset: Callable[[int], str]) -> str:
    return "\n".join(
        f"        const {c_type(operand.elem_type)} x{index} = "
        f"in{index}[{offset(index)}];"
        for index, operand in enumerate(operands)
    )


def _call(name: str, arguments: Sequence[str]) -> str:
    return f"{name}(\n    " + ",\n    ".join(arguments) + ");"


def _kernel_name(
    context: NodeContext,
    operands: Sequence[TensorRef],
    result: TensorRef,
    variant: str,
    aligned: bool,
) -> str:
    """A name encoding everything the emitted code depends on, and nothing else.

    Two nodes running the same op reach the same kernel — and so share one definition — only
    when their operand types, arity, broadcasting and formula all agree; anything else would
    be two kernels colliding on one name.
    """
    types = [c_type(operand.elem_type) for operand in operands]
    types.append(c_type(result.elem_type))
    tag = f"{len(operands)}_{types[0]}" if len(set(types)) == 1 else "_".join(types)
    form = "" if aligned else "_bcast"
    return f"{context.prefix}_{context.node.op_type.lower()}{variant}{form}_{tag}"
