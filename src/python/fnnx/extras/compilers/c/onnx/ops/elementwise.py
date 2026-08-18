"""Elementwise math: broadcasting arithmetic, the variadic families, and pointwise math."""

from __future__ import annotations

from functools import partial
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    UNSIGNED_TYPES,
    c_type,
    element_type_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    copy_tensor,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import (
    Scalar,
    combiner,
    elementwise,
    expand,
    math_suffix,
    pointwise,
)

# Numpy-style multidirectional broadcasting arrived at opset 7; 13 and 14 only widened the
# type constraints.
_ARITHMETIC_VERSIONS = (7, 13, 14)
_ARITHMETIC_OPERATORS = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}

# The variadic families took equal shapes until opset 8 introduced broadcasting; the later
# revisions widened types. Sum and Mean were never defined for the integer families.
_EXTREMUM_VERSIONS = (8, 12, 13)
_ACCUMULATE_VERSIONS = (8, 13)

# Pow-7 introduced broadcasting, 12 let the exponent carry its own type, and 13 and 15 only
# widened the type constraints.
_POW_VERSIONS = (7, 12, 13, 15)
_MOD_VERSIONS = (10, 13)
# Identity has never changed for tensors; every revision listed adds element or container
# types, and the ones this compiler does not support are rejected before dispatch.
_IDENTITY_VERSIONS = (1, 13, 14, 16, 19, 21, 23, 24, 25)
# Dropout-7 is inference-only; 10 made the mask output boolean; 12 added the ratio and
# training_mode inputs; 13 and 22 only changed which element types are allowed. The kernel
# takes the mask's element type from the graph, so the same code serves all of them.
_DROPOUT_VERSIONS = (7, 10, 12, 13, 22)
# Clip-6 carries its bounds as attributes and 11 moved them into optional inputs; 12 and 13
# widened the types.
_CLIP_ATTRIBUTE_VERSIONS = (6,)
_CLIP_INPUT_VERSIONS = (11, 12, 13)

# Ops with one operand whose result is a libm call or a short expression over it, each with
# the schema revisions the formula covers. The legacy revision 1 of the older ones carried a
# `consumed_inputs` attribute and is left unregistered.
_UNARY_MATH: dict[str, tuple[tuple[int, ...], str]] = {
    "Acos": ((7, 22), "acos$f(x0)"),
    "Acosh": ((9, 22), "acosh$f(x0)"),
    "Asin": ((7, 22), "asin$f(x0)"),
    "Asinh": ((9, 22), "asinh$f(x0)"),
    "Atan": ((7, 22), "atan$f(x0)"),
    "Atanh": ((9, 22), "atanh$f(x0)"),
    "Ceil": ((6, 13), "ceil$f(x0)"),
    "Cos": ((7, 22), "cos$f(x0)"),
    "Cosh": ((9, 22), "cosh$f(x0)"),
    "Erf": ((9, 13), "erf$f(x0)"),
    "Exp": ((6, 13), "exp$f(x0)"),
    "Floor": ((6, 13), "floor$f(x0)"),
    "Log": ((6, 13), "log$f(x0)"),
    "Reciprocal": ((6, 13), "$one / x0"),
    # ONNX rounds halves to even, which `rint` does under C's default rounding mode.
    "Round": ((11, 22), "rint$f(x0)"),
    "Sin": ((7, 22), "sin$f(x0)"),
    "Sinh": ((9, 22), "sinh$f(x0)"),
    "Sqrt": ((6, 13), "sqrt$f(x0)"),
    "Tan": ((7, 22), "tan$f(x0)"),
    "Tanh": ((6, 13), "tanh$f(x0)"),
}

# Abs, Neg and Sign serve the integer families as well, where the sign has to be read off a
# comparison rather than from libm.
_SIGN_VERSIONS = (9, 13)
_ABS_VERSIONS = _NEG_VERSIONS = (6, 13)

_FLOORED_MOD_TEMPLATE = Template("""\
static $element $name($element left, $element right)
{
    $element remainder = left % right;
    /* C truncates toward zero; ONNX's fmod=0 takes the divisor's sign, as numpy does. */
    if (remainder != 0 && ((remainder < 0) != (right < 0))) {
        remainder += right;
    }
    return remainder;
}""")

# Integer exponentiation, which numpy performs exactly rather than through `pow`. The
# squaring runs in the unsigned counterpart of the element type so that an overflow wraps —
# defined behaviour, and the same result numpy's integer power gives — instead of being
# undefined signed overflow.
_INTEGER_POW_TEMPLATE = Template("""\
static $element $name($element base, $exponent exponent)
{
$guard    {
        $unsigned accumulator = 1u;
        $unsigned factor = ($unsigned)base;
        uint64_t remaining = (uint64_t)exponent;
        while (remaining > 0) {
            if (remaining & 1) {
                accumulator = ($unsigned)(accumulator * factor);
            }
            factor = ($unsigned)(factor * factor);
            remaining >>= 1;
        }
        return ($element)accumulator;
    }
}""")

# ONNX leaves a negative integer exponent undefined and numpy refuses to evaluate it, so
# there is no behaviour to match here — only one that has to be defined.
_NEGATIVE_EXPONENT_GUARD = """\
    if (exponent < 0) {
        return 0;
    }
"""

_UNSIGNED_COUNTERPARTS: dict[int, int] = {
    TensorProto.INT8: TensorProto.UINT8,
    TensorProto.INT16: TensorProto.UINT16,
    TensorProto.INT32: TensorProto.UINT32,
    TensorProto.INT64: TensorProto.UINT64,
}


def _arithmetic(context: NodeContext, *, operator: str) -> NodeEmission:
    return elementwise(
        context,
        expression=f"x0 {operator} x1",
        operands=(context.require_input(0), context.require_input(1)),
        result=context.require_output(0),
    )


def _extremum(context: NodeContext, *, largest: bool) -> NodeEmission:
    """Min or Max over any number of operands, folded left as the reference folds them."""
    operands = tuple(
        context.require_input(index) for index in range(len(context.inputs))
    )
    result = context.require_output(0)
    expression = "x0"
    helpers: tuple[CFunction, ...] = ()
    if len(operands) > 1:
        helper = combiner(context, result.elem_type, largest=largest)
        helpers = (helper,)
        for index in range(1, len(operands)):
            expression = f"{helper.name}({expression}, x{index})"
    return elementwise(
        context,
        expression=expression,
        operands=operands,
        result=result,
        helpers=helpers,
    )


def _accumulate(context: NodeContext, *, average: bool) -> NodeEmission:
    """Sum or Mean: the operands added in order, and for Mean divided by how many there are."""
    operands = tuple(
        context.require_input(index) for index in range(len(context.inputs))
    )
    result = context.require_output(0)
    total = " + ".join(f"x{index}" for index in range(len(operands)))
    expression = (
        f"({total}) / {scalar_literal(len(operands), result.elem_type)}"
        if average
        else total
    )
    return elementwise(context, expression=expression, operands=operands, result=result)


def _mod(context: NodeContext) -> NodeEmission:
    left = context.require_input(0)
    right = context.require_input(1)
    result = context.require_output(0)
    truncated = bool(context.attribute("fmod", 0))
    if result.elem_type in FLOAT_TYPES and not truncated:
        raise CompileError(
            f"Node `{context.label}`: Mod on `{element_type_name(result.elem_type)}` "
            "tensors requires the `fmod` attribute to be 1, as the ONNX spec does."
        )
    helpers: tuple[CFunction, ...] = ()
    if result.elem_type in FLOAT_TYPES:
        expression = f"fmod{math_suffix(result.elem_type)}(x0, x1)"
        variant = "_truncated"
    elif truncated or result.elem_type in UNSIGNED_TYPES:
        # For the unsigned families both definitions agree: no remainder is ever negative.
        expression = "x0 % x1"
        variant = "_truncated"
    else:
        name = f"{context.prefix}_floored_mod_{c_type(result.elem_type)}"
        helpers = (
            CFunction(
                name,
                _FLOORED_MOD_TEMPLATE.substitute(
                    name=name, element=c_type(result.elem_type)
                ),
            ),
        )
        expression = f"{name}(x0, x1)"
        variant = "_floored"
    return elementwise(
        context,
        expression=expression,
        operands=(left, right),
        result=result,
        helpers=helpers,
        variant=variant,
    )


def _pow(context: NodeContext) -> NodeEmission:
    """Pow, whose exponent carries an element type of its own from opset 12 on."""
    base = context.require_input(0)
    exponent = context.require_input(1)
    result = context.require_output(0)
    helpers: tuple[CFunction, ...] = ()
    if result.elem_type in FLOAT_TYPES and exponent.elem_type == result.elem_type:
        expression = f"pow{math_suffix(result.elem_type)}(x0, x1)"
    elif result.elem_type in FLOAT_TYPES or exponent.elem_type in FLOAT_TYPES:
        # numpy promotes a mixed pair to float64 and casts the result back to the base's
        # type, so the double-precision call is what has to be matched here.
        expression = f"({c_type(result.elem_type)})pow((double)x0, (double)x1)"
    else:
        helper = _integer_pow(context, result.elem_type, exponent.elem_type)
        helpers = (helper,)
        expression = f"{helper.name}(x0, x1)"
    return elementwise(
        context,
        expression=expression,
        operands=(base, exponent),
        result=result,
        helpers=helpers,
    )


def _integer_pow(context: NodeContext, elem_type: int, exponent_type: int) -> CFunction:
    name = f"{context.prefix}_integer_pow_{c_type(elem_type)}_{c_type(exponent_type)}"
    return CFunction(
        name,
        _INTEGER_POW_TEMPLATE.substitute(
            name=name,
            element=c_type(elem_type),
            exponent=c_type(exponent_type),
            unsigned=c_type(_UNSIGNED_COUNTERPARTS[elem_type]),
            guard="" if exponent_type in UNSIGNED_TYPES else _NEGATIVE_EXPONENT_GUARD,
        ),
    )


def _abs(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    if result.elem_type in FLOAT_TYPES:
        return pointwise(context, "fabs$f(x0)")
    if result.elem_type in UNSIGNED_TYPES:
        return pointwise(context, "x0")
    return pointwise(context, "(x0 < $zero) ? ($element)-x0 : x0")


def _sign(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    if result.elem_type in UNSIGNED_TYPES:
        return pointwise(context, "($element)(x0 > $zero)")
    # numpy's sign leaves NaN alone; every zero comes out `+0` whatever sign it went in
    # with, which is what the difference of the two comparisons already gives.
    negative = "($element)((x0 > $zero) - (x0 < $zero))"
    if result.elem_type in FLOAT_TYPES:
        return pointwise(context, f"isnan(x0) ? x0 : {negative}")
    return pointwise(context, negative)


def _clip_from_attributes(context: NodeContext) -> NodeEmission:
    """Clip up to opset 10, whose bounds are attributes defaulting to the float32 extremes."""
    result = context.require_output(0)
    low = float(context.attribute("min", _FLOAT_MIN))
    high = float(context.attribute("max", _FLOAT_MAX))
    largest = combiner(context, result.elem_type, largest=True)
    smallest = combiner(context, result.elem_type, largest=False)
    return pointwise(
        context,
        f"{smallest.name}(high, {largest.name}(low, x0))",
        scalars=(
            Scalar("low", result.elem_type, low),
            Scalar("high", result.elem_type, high),
        ),
        helpers=(largest, smallest),
    )


def _clip_from_inputs(context: NodeContext) -> NodeEmission:
    """Clip from opset 11 on, whose bounds are optional scalar inputs.

    ONNX defines the result as numpy's `clip`, which applies the lower bound first: a lower
    bound above the upper one yields the upper one, and a NaN bound wins outright. Which
    operand a *tie* yields — all that separates `+0` from `-0` here — differs between the
    forms numpy evaluates: with both bounds it is `minimum(max, maximum(min, x))`, keeping
    the data's zero, while a single bound goes through plain `maximum(x, min)`, keeping the
    bound's. Both are reproduced here rather than unified.
    """
    result = context.require_output(0)
    source = context.require_input(0)
    largest = combiner(context, result.elem_type, largest=True)
    smallest = combiner(context, result.elem_type, largest=False)
    bounds = {
        tag: (bound, helper)
        for tag, index, helper in (("lo", 1, largest), ("hi", 2, smallest))
        if (bound := context.optional_input(index)) is not None
    }
    if not bounds:
        return copy_tensor(source, result)
    if len(bounds) == 2:
        expression = f"{smallest.name}(x2, {largest.name}(x1, x0))"
    else:
        ((_, helper),) = bounds.values()
        expression = f"{helper.name}(x0, x1)"
    return elementwise(
        context,
        expression=expression,
        operands=(source, *(bound for bound, _ in bounds.values())),
        result=result,
        helpers=tuple(helper for _, helper in bounds.values()),
        variant=f"_{''.join(bounds)}",
    )


def _dropout(context: NodeContext) -> NodeEmission:
    """Dropout in inference mode: the data unchanged, and a mask of ones where asked for.

    Training mode samples a mask, which no compiled artifact can reproduce, so it is a
    compile error unless the graph proves the mode off.
    """
    training_mode = context.optional_input(2)
    mode = context.constant_input(2)
    if training_mode is not None and (mode is None or mode.any()):
        raise CompileError(
            f"Node `{context.label}`: Dropout is supported in inference mode only, but "
            f"`{training_mode.name}` is not a compile-time false; the training-mode mask "
            "is drawn at random and cannot be compiled."
        )
    result = context.require_output(0)
    emission = copy_tensor(context.require_input(0), result)
    mask = context.outputs[1] if len(context.outputs) > 1 else None
    if mask is None:
        return emission
    ones = elementwise(
        context,
        expression=expand("$one", mask.elem_type),
        operands=(),
        result=mask,
        variant="_mask",
    )
    return NodeEmission(
        functions=emission.functions + ones.functions,
        statements=emission.statements + ones.statements,
    )


def _identity(context: NodeContext) -> NodeEmission:
    return copy_tensor(context.require_input(0), context.require_output(0))


# The float32 extremes ONNX gives as Clip-6's default bounds.
_FLOAT_MAX = 3.4028234663852886e38
_FLOAT_MIN = -_FLOAT_MAX


for _op_type, _operator in _ARITHMETIC_OPERATORS.items():
    register_kernel(
        "", _op_type, _ARITHMETIC_VERSIONS, partial(_arithmetic, operator=_operator)
    )
for _op_type, (_versions, _template) in _UNARY_MATH.items():
    register_kernel("", _op_type, _versions, partial(pointwise, template=_template))
register_kernel("", "Min", _EXTREMUM_VERSIONS, partial(_extremum, largest=False))
register_kernel("", "Max", _EXTREMUM_VERSIONS, partial(_extremum, largest=True))
register_kernel("", "Sum", _ACCUMULATE_VERSIONS, partial(_accumulate, average=False))
register_kernel("", "Mean", _ACCUMULATE_VERSIONS, partial(_accumulate, average=True))
register_kernel("", "Mod", _MOD_VERSIONS, _mod)
register_kernel("", "Pow", _POW_VERSIONS, _pow)
register_kernel("", "Abs", _ABS_VERSIONS, _abs)
register_kernel("", "Neg", _NEG_VERSIONS, partial(pointwise, template="-x0"))
register_kernel("", "Sign", _SIGN_VERSIONS, _sign)
register_kernel("", "Clip", _CLIP_ATTRIBUTE_VERSIONS, _clip_from_attributes)
register_kernel("", "Clip", _CLIP_INPUT_VERSIONS, _clip_from_inputs)
register_kernel("", "Dropout", _DROPOUT_VERSIONS, _dropout)
register_kernel("", "Identity", _IDENTITY_VERSIONS, _identity)
