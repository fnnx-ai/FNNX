"""Activation kernels, each emitted as the formula the ONNX spec defines it by."""

from __future__ import annotations

import math
from functools import partial
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import FLOAT_TYPES, c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import (
    Scalar,
    combiner,
    elementwise,
    expand,
    pointwise,
)

# Revision 1 of the older activations carried the legacy `consumed_inputs` attribute; the
# revisions listed after the first are the ones that only widened type constraints.
_RELU_VERSIONS = (6, 13, 14)
_SIGMOID_VERSIONS = (6, 13)
_LEAKY_RELU_VERSIONS = (6, 16)
# PRelu-7 replaced the legacy broadcast attributes with numpy-style broadcasting of `slope`.
_PRELU_VERSIONS = (7, 9, 16)
_ELU_VERSIONS = (6, 22)
_SELU_VERSIONS = (6, 22)
_CELU_VERSIONS = (12,)
_HARD_SIGMOID_VERSIONS = (6, 22)
_SOFTPLUS_VERSIONS = (1, 22)
_SOFTSIGN_VERSIONS = (1, 22)
_SHRINK_VERSIONS = (9,)
_THRESHOLDED_RELU_VERSIONS = (10, 22)
_GELU_VERSIONS = (20,)

# The constants of Gelu's two formulas, which ONNX writes as `sqrt(2)`, `sqrt(2/pi)` and
# `0.044715`; the square roots are taken in the element type, as the op's function body
# takes them.
_TWO_OVER_PI = 2.0 / math.pi
_GELU_CUBIC = 0.044715
_GELU_MODES = ("none", "tanh")

# The reference's numerically stable form: whichever branch is taken, the exponent is of a
# negative number, so it underflows to zero instead of overflowing to infinity.
_SIGMOID_TEMPLATE = Template("""\
static $element $name($element x)
{
    return x > $zero
        ? $one / ($one + exp$f(-x))
        : exp$f(x) / ($one + exp$f(x));
}""")

# Activations whose formula is a plain expression over the operand and its attributes.
# `$element`, `$zero`, `$one` and `$f` come from the element type; the attribute names are
# passed to the kernel as parameters, so one kernel serves every node running the op.
_ATTRIBUTE_ACTIVATIONS: dict[str, tuple[tuple[int, ...], tuple[str, ...], str]] = {
    "Elu": (
        _ELU_VERSIONS,
        ("alpha",),
        "(x0 > $zero) ? x0 : alpha * (exp$f(x0) - $one)",
    ),
    "LeakyRelu": (_LEAKY_RELU_VERSIONS, ("alpha",), "(x0 > $zero) ? x0 : x0 * alpha"),
    "Selu": (
        _SELU_VERSIONS,
        ("alpha", "gamma"),
        "((x0 > $zero) ? x0 : exp$f(x0) * alpha - alpha) * gamma",
    ),
    "ThresholdedRelu": (
        _THRESHOLDED_RELU_VERSIONS,
        ("alpha",),
        "(x0 > alpha) ? x0 : $zero",
    ),
}

# The same, without attributes.
_PLAIN_ACTIVATIONS: dict[str, tuple[tuple[int, ...], str]] = {
    "Softplus": (_SOFTPLUS_VERSIONS, "log$f(exp$f(x0) + $one)"),
    "Softsign": (_SOFTSIGN_VERSIONS, "x0 / (fabs$f(x0) + $one)"),
}


def _relu(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    # The spec is `max(0, x)`, evaluated as numpy's `maximum`: NaN propagates and -0 comes
    # out as +0, both of which a plain `value > 0 ? value : 0` would get wrong.
    guard = (
        "x0 > $zero || isnan(x0)" if result.elem_type in FLOAT_TYPES else "x0 > $zero"
    )
    return pointwise(context, f"({guard}) ? x0 : $zero")


def _sigmoid(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    name = f"{context.prefix}_sigmoid_{c_type(result.elem_type)}"
    helper = CFunction(
        name,
        expand(_SIGMOID_TEMPLATE.safe_substitute(name=name), result.elem_type),
    )
    return pointwise(context, f"{name}(x0)", helpers=(helper,))


def _attribute_activation(
    context: NodeContext, *, names: tuple[str, ...], template: str
) -> NodeEmission:
    result = context.require_output(0)
    return pointwise(
        context,
        template,
        scalars=tuple(
            Scalar(name, result.elem_type, context.float_attribute(name))
            for name in names
        ),
    )


def _prelu(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    return elementwise(
        context,
        expression=expand("(x0 > $zero) ? x0 : x0 * x1", result.elem_type),
        operands=(context.require_input(0), context.require_input(1)),
        result=result,
    )


def _celu(context: NodeContext) -> NodeEmission:
    """Celu: `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`, as ONNX defines it."""
    result = context.require_output(0)
    largest = combiner(context, result.elem_type, largest=True)
    smallest = combiner(context, result.elem_type, largest=False)
    return pointwise(
        context,
        f"{largest.name}($zero, x0) + "
        f"{smallest.name}($zero, alpha * (exp$f(x0 / alpha) - $one))",
        scalars=(Scalar("alpha", result.elem_type, context.float_attribute("alpha")),),
        helpers=(largest, smallest),
    )


def _hard_sigmoid(context: NodeContext) -> NodeEmission:
    result = context.require_output(0)
    largest = combiner(context, result.elem_type, largest=True)
    smallest = combiner(context, result.elem_type, largest=False)
    return pointwise(
        context,
        f"{largest.name}($zero, {smallest.name}($one, x0 * alpha + beta))",
        scalars=tuple(
            Scalar(name, result.elem_type, context.float_attribute(name))
            for name in ("alpha", "beta")
        ),
        helpers=(largest, smallest),
    )


def _gelu(context: NodeContext) -> NodeEmission:
    """Gelu in whichever of its two forms the `approximate` attribute selects.

    Grouped as `(0.5 * x) * (1 + phi)`, the order of the function body ONNX defines the op
    by: at a large negative x the second factor underflows to zero, and only this grouping
    leaves the sign on the zero that comes out of it.
    """
    result = context.require_output(0)
    constant = partial(scalar_literal, elem_type=result.elem_type)
    if _approximate(context) == "tanh":
        phi = (
            f"tanh$f(sqrt$f({constant(_TWO_OVER_PI)}) * "
            f"(x0 + {constant(_GELU_CUBIC)} * pow$f(x0, {constant(3.0)})))"
        )
        variant = "_tanh"
    else:
        phi = f"erf$f(x0 / sqrt$f({constant(2.0)}))"
        variant = "_erf"
    return pointwise(
        context, f"({constant(0.5)} * x0) * ($one + {phi})", variant=variant
    )


def _approximate(context: NodeContext) -> str:
    value = context.attribute("approximate", b"none")
    mode = value.decode() if isinstance(value, bytes) else str(value)
    if mode not in _GELU_MODES:
        raise CompileError(
            f"Node `{context.label}`: Gelu's `approximate` attribute is `{mode}`, but "
            f"ONNX defines only {' and '.join(f'`{name}`' for name in _GELU_MODES)}."
        )
    return mode


def _shrink(context: NodeContext) -> NodeEmission:
    """Shrink, which the reference evaluates in double for the integer element types too.

    Adding the bias in double and rounding once on the way back gives the element type's own
    arithmetic for the floating-point families, and numpy's promotion for the integer ones.
    """
    return pointwise(
        context,
        "($element)((x0 < -lambd) ? (x0 + bias) : ((x0 > lambd) ? (x0 - bias) : 0))",
        scalars=tuple(
            Scalar(name, TensorProto.DOUBLE, context.float_attribute(name))
            for name in ("lambd", "bias")
        ),
    )


for _op_type, (_versions, _names, _template) in _ATTRIBUTE_ACTIVATIONS.items():
    register_kernel(
        "",
        _op_type,
        _versions,
        partial(_attribute_activation, names=_names, template=_template),
    )
for _op_type, (_versions, _template) in _PLAIN_ACTIVATIONS.items():
    register_kernel("", _op_type, _versions, partial(pointwise, template=_template))
register_kernel("", "Relu", _RELU_VERSIONS, _relu)
register_kernel("", "Sigmoid", _SIGMOID_VERSIONS, _sigmoid)
register_kernel("", "PRelu", _PRELU_VERSIONS, _prelu)
register_kernel("", "Celu", _CELU_VERSIONS, _celu)
register_kernel("", "HardSigmoid", _HARD_SIGMOID_VERSIONS, _hard_sigmoid)
register_kernel("", "Gelu", _GELU_VERSIONS, _gelu)
register_kernel("", "Shrink", _SHRINK_VERSIONS, _shrink)
