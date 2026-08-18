"""What an ONNX-ML predictor does with a row of scores once it has computed one.

The tree ensembles, the support vector machines and the linear models differ entirely in how
they arrive at those scores and not at all in what they do next: the five `post_transform`
values ONNX-ML defines, the second column a single-score binary classifier is paired with,
and the `argmax` that turns a row into a class label. All three are emitted from here, so a
graph running several kinds of predictor shares one kernel per element type rather than one
per op.

Each is the ONNX reference implementation's own arithmetic rather than the textbook form:
`SOFTMAX_ZERO`'s threshold, `PROBIT`'s rational approximation of `erfinv` and the constants
either side of it, and the `LOGISTIC` that folds a negative argument onto a positive one are
what `onnx.reference.ops.aionnxml._common_classifier` computes, and that is the oracle every
predictor here is compared against.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from string import Template
from typing import TypeVar

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_type_name
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    constant_data,
)
from fnnx.extras.compilers.c.onnx.ops.axes import call_kernel
from fnnx.extras.compilers.c.onnx.ops.broadcast import math_suffix

# How an op names the transform or the aggregation it applies: the ONNX-ML ops of opset 1
# spell them out as strings, the opset-5 `TreeEnsemble` numbers them.
Declared = TypeVar("Declared", str, int)

# The score transforms, in the numbering opset 5 gave them; every op that names them as
# strings names these same five.
NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT = range(5)
NAMED_TRANSFORMS = {
    "NONE": NONE,
    "SOFTMAX": SOFTMAX,
    "LOGISTIC": LOGISTIC,
    "SOFTMAX_ZERO": SOFTMAX_ZERO,
    "PROBIT": PROBIT,
}
_TRANSFORM_NAMES = {
    SOFTMAX: "softmax",
    LOGISTIC: "logistic",
    SOFTMAX_ZERO: "softmax_zero",
    PROBIT: "probit",
}

# What a classifier makes of the winning column of a row. Naming the class that column stands
# for is what every predictor here does bar one: `SVMClassifier` carries two readings of its
# own, which `set_score_svm` applies to an ensemble whose `rho` is a single value — the second
# class outright once the winning vote reaches one half, and the sign of the winning value as
# a label of its own where the ensemble does not have exactly two classes.
ARGMAX_LABEL, POSITIVE_CLASS, SIGN_LABEL = range(3)
# The transforms that read a whole row at once, against the ones that map value to value.
_ROW_TRANSFORMS = (SOFTMAX, SOFTMAX_ZERO)

# `SOFTMAX_ZERO` leaves a value this close to zero out of the exponential and scales it
# instead; the constant is the reference implementation's own.
_ZERO_THRESHOLD = 1e-7

# `PROBIT` is `sqrt(2) * erfinv(2p - 1)` with `erfinv` the rational approximation the
# reference implementation spells out; these are the two constants that approximation is
# built from, either side of `0.5 * log((1 - x) * (1 + x))`.
_PROBIT_FIRST = 2.0 / (math.pi * 0.147)
_PROBIT_SECOND = 1.0 / 0.147
_PROBIT_SCALE = 1.41421356

_SOFTMAX_TEMPLATE = Template("""\
static void $name($result* scores, size_t rows, size_t columns)
{
    size_t row, index;
    if (columns == 0) {
        return;
    }
    for (row = 0; row < rows; ++row) {
        $result* out = scores + row * columns;
        $result largest = out[0];
        $result total = ($result)0;
        for (index = 1; index < columns; ++index) {
            if (out[index] > largest || isnan(out[index])) {
                largest = out[index];
            }
        }
        for (index = 0; index < columns; ++index) {
            out[index] = exp$suffix(out[index] - largest);
            total += out[index];
        }
        for (index = 0; index < columns; ++index) {
            out[index] /= total;
        }
    }
}""")

_SOFTMAX_ZERO_TEMPLATE = Template("""\
static void $name($result* scores, size_t rows, size_t columns)
{
    size_t row, index;
    if (columns == 0) {
        return;
    }
    for (row = 0; row < rows; ++row) {
        $result* out = scores + row * columns;
        $result largest = out[0];
        $result total = ($result)0;
        $result scale;
        for (index = 1; index < columns; ++index) {
            if (out[index] > largest || isnan(out[index])) {
                largest = out[index];
            }
        }
        scale = exp$suffix(-largest);
        for (index = 0; index < columns; ++index) {
            const $result value = out[index];
            out[index] = (value > $threshold || value < -$threshold)
                ? exp$suffix(value - largest)
                : value * scale;
            total += out[index];
        }
        for (index = 0; index < columns; ++index) {
            out[index] = (total == ($result)0) ? ($result)0.5 : out[index] / total;
        }
    }
}""")

_LOGISTIC_TEMPLATE = Template("""\
static void $name($result* scores, size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        const $result value = scores[index];
        const $result mapped =
            ($result)1 / (($result)1 + exp$suffix(-fabs$suffix(value)));
        scores[index] = (value < 0) ? (($result)1 - mapped) : mapped;
    }
}""")

_PROBIT_TEMPLATE = Template("""\
static void $name($result* scores, size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        const $result value = scores[index] * ($result)2 - ($result)1;
        const $result inner = (($result)1 - value) * (($result)1 + value);
        $result mapped = ($result)0;
        if (inner != ($result)0) {
            const $result logarithm = log$suffix(inner);
            const $result first = $constant + ($result)0.5 * logarithm;
            const $result second = $reciprocal * logarithm;
            const $result root = -first + sqrt$suffix(first * first - second);
            mapped = ((value < 0) ? ($result)-1 : ($result)1) * sqrt$suffix(root);
        }
        scores[index] = $scale * mapped;
    }
}""")

_BINARY_TEMPLATE = Template("""\
static void $name($result* scores, size_t rows, size_t columns, int complement)
{
    size_t row;
    for (row = 0; row < rows; ++row) {
        $result* out = scores + row * columns;
        out[1] = out[0];
        out[0] = complement ? (($result)1 - out[1]) : -out[1];
    }
}""")

_ARGMAX_TEMPLATE = Template("""\
static void $name(
    int64_t* labels,
    const $result* values,
    const int64_t* classes,
    size_t rows,
    size_t columns,
    int rule)
{
    size_t row, index;
    for (row = 0; row < rows; ++row) {
        const $result* out = values + row * columns;
        size_t best = 0;
        for (index = 1; index < columns; ++index) {
            /* The column carrying the first value that is not a number wins outright, which
               is what the `argmax` this stands for returns. */
            if (isnan(out[best])) {
                break;
            }
            if (out[index] > out[best] || isnan(out[index])) {
                best = index;
            }
        }
        if (rule == $sign) {
            labels[row] = (out[best] > ($result)0) ? 1 : 0;
        } else if (rule == $positive && out[best] >= ($result)0.5) {
            labels[row] = classes[1];
        } else {
            labels[row] = classes[best];
        }
    }
}""")

_TRANSFORM_TEMPLATES = {
    SOFTMAX: _SOFTMAX_TEMPLATE,
    SOFTMAX_ZERO: _SOFTMAX_ZERO_TEMPLATE,
    LOGISTIC: _LOGISTIC_TEMPLATE,
    PROBIT: _PROBIT_TEMPLATE,
}


def named_transform(context: NodeContext) -> int:
    """The `post_transform` an op names as a string, as one of the five constants above."""
    return choice(
        context,
        "post_transform",
        context.string_attribute("post_transform"),
        NAMED_TRANSFORMS,
    )


def post_transform(
    context: NodeContext, scores: TensorRef, transform: int, rows: int, columns: int
) -> NodeEmission | None:
    """The transform the node names, applied over the scores in place."""
    template = _TRANSFORM_TEMPLATES.get(transform)
    if template is None:
        return None
    element = c_type(scores.elem_type)
    name = f"{context.prefix}_ml_{_TRANSFORM_NAMES[transform]}_{element}"
    definition = template.substitute(
        name=name,
        result=element,
        suffix=math_suffix(scores.elem_type),
        threshold=scalar_literal(_ZERO_THRESHOLD, scores.elem_type),
        constant=scalar_literal(_PROBIT_FIRST, scores.elem_type),
        reciprocal=scalar_literal(_PROBIT_SECOND, scores.elem_type),
        scale=scalar_literal(_PROBIT_SCALE, scores.elem_type),
    )
    arguments = (
        [scores.expr, f"{rows}u", f"{columns}u"]
        if transform in _ROW_TRANSFORMS
        else [scores.expr, f"{rows * columns}u"]
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def binary_scores(
    context: NodeContext,
    scores: TensorRef,
    rows: int,
    columns: int,
    *,
    complement: bool,
) -> NodeEmission:
    """The second column a classifier scoring one value per row pairs that value with.

    The score is expected in the first column of each row of `scores`, which the caller has
    already sized for both. Which second column it gets is the classifier's own rule: the
    complement of the score, or its negation.
    """
    element = c_type(scores.elem_type)
    name = f"{context.prefix}_ml_binary_{element}"
    definition = _BINARY_TEMPLATE.substitute(name=name, result=element)
    call = call_kernel(
        name, [scores.expr, f"{rows}u", f"{columns}u", str(int(complement))]
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def argmax_labels(
    context: NodeContext,
    labels: TensorRef,
    values: str,
    elem_type: int,
    classes: Sequence[int],
    rows: int,
    columns: int,
    rule: int = ARGMAX_LABEL,
) -> NodeEmission:
    """Each row labelled from its winning column of `values`, by the rule the op applies.

    `values` is what the classes are ranked by, which is the scores themselves for every
    predictor but the support-vector one, where the classes are ranked by their votes.
    """
    element = c_type(elem_type)
    name = f"{context.prefix}_ml_argmax_{element}"
    definition = _ARGMAX_TEMPLATE.substitute(
        name=name, result=element, sign=SIGN_LABEL, positive=POSITIVE_CLASS
    )
    data, symbol = constant_data(context, "classes", np.array(classes, np.int64))
    call = call_kernel(
        name, [labels.expr, values, symbol, f"{rows}u", f"{columns}u", str(rule)]
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(data,),
    )


def extend(emission: NodeEmission, addition: NodeEmission | None) -> NodeEmission:
    if addition is None:
        return emission
    return NodeEmission(
        functions=emission.functions + addition.functions,
        statements=emission.statements + addition.statements,
        scratch=emission.scratch + addition.scratch,
        constants=emission.constants + addition.constants,
    )


def choice(
    context: NodeContext,
    name: str,
    value: Declared,
    choices: Mapping[Declared, int],
) -> int:
    if value not in choices:
        raise CompileError(
            f"Node `{context.label}`: `{name}` is `{value}`, which is none of the values "
            f"ONNX defines for it ({', '.join(str(key) for key in choices)})."
        )
    return choices[value]


def float_output(context: NodeContext, index: int) -> TensorRef:
    """The scores, checked to be the float32 tensor the ONNX-ML schemas declare."""
    result = context.require_output(index)
    if result.elem_type != TensorProto.FLOAT:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` scores in `float`, but its "
            f"output `{result.name}` is declared "
            f"`{element_type_name(result.elem_type)}`."
        )
    return result


def label_output(context: NodeContext, index: int) -> TensorRef:
    """The predicted classes, checked to be the `int64` tensor a numeric label table needs."""
    result = context.require_output(index)
    if result.elem_type != TensorProto.INT64:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` labels each row with an "
            f"`int64` class value, but its output `{result.name}` is declared "
            f"`{element_type_name(result.elem_type)}`."
        )
    return result
