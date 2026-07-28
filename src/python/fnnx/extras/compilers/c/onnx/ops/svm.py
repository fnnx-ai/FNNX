"""Support vector machines and the linear models beside them.

`LinearRegressor`, `LinearClassifier`, `SVMRegressor` and `SVMClassifier` are ONNX-ML's four
non-forest predictors, and three of the four are one computation: a row of `X` is scored
against a coefficient per feature per output, offset, and — for a classifier — turned into a
label. Only `SVMClassifier`'s support-vector mode differs, and that is libsvm's one-against-one
scheme: every pair of classes votes, and the winner of the vote names the row's class. All
four carry their parameters as *attributes*, so every table below is emitted as `static const`
data a shared kernel reads through a pointer, and each scores in float32 whatever the element
type of `X` — which is what their schemas declare `Y` and `Z` to be.

Each op is at revision 1 of its schema and has been since ONNX-ML opset 1.

Where this follows ONNX's reference implementation rather than the prose, because the
reference is the oracle these ops are compared against:

* A `LinearRegressor` or `LinearClassifier` with no `intercepts` is refused. The reference
  reads the missing attribute as a NaN and adds it to every score; onnxruntime treats it as
  zero. Nothing can vouch for either, so the model is refused rather than silently taking a
  side.
* A `LinearClassifier` scoring one column against two class labels pairs the score with its
  own negation — `[-s, s]`, the *whole* score including the intercept, which is where the
  reference and onnxruntime part company; converters emit one coefficient row per class, so
  the case does not arise for them.
* `SVMClassifier`'s label is not always the class its winning column names: with a single
  `rho`, two class labels, no probabilities and no negative coefficient, a winning vote of at
  least 0.5 names the *second* class outright, and with a single `rho` and any other number
  of labels the label is 1 or 0 by the sign of the winning value. Both are `set_score_svm`'s.
* A row of one score is widened to two only when a second class is called for, and is left
  *untransformed* when it is not — `write_scores` returns such a row before it reaches the
  transform, `PROBIT` alone excepted.

What the compiler refuses: `prob_a`/`prob_b` on an ensemble of more than two classes. The
pairwise probabilities are coupled by an iterative solver whose matrix the reference builds
with a broadcast where libsvm — and onnxruntime with it — writes one entry per pair, so the
two disagree for three classes and up; they agree exactly for two, which is the case
scikit-learn's binary `SVC(probability=True)` emits.
"""

from __future__ import annotations

from string import Template
from typing import NamedTuple

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    constant_data,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.loader import ML_DOMAIN
from fnnx.extras.compilers.c.onnx.ops.axes import call_kernel, verify_shape
from fnnx.extras.compilers.c.onnx.ops.scores import (
    ARGMAX_LABEL,
    NONE,
    POSITIVE_CLASS,
    PROBIT,
    SIGN_LABEL,
    argmax_labels,
    binary_scores,
    choice,
    extend,
    float_output,
    label_output,
    named_transform,
    post_transform,
)

_VERSIONS = (1,)

# The kernel functions a support vector machine compares a row against, numbered for the one
# switch the emitted helper is written as. ONNX names them as strings, which the reference
# implementation lowercases before it matches — so the compiler does too.
_LINEAR, _POLY, _RBF, _SIGMOID = range(4)
_KERNEL_TYPES = {"linear": _LINEAR, "poly": _POLY, "rbf": _RBF, "sigmoid": _SIGMOID}

# `gamma`, `coef0` and `degree`, which ONNX packs into one attribute in that order.
_KERNEL_PARAMETERS = 3

# What a pairwise probability is clamped to either side, and the iteration that couples the
# pairs: the tolerance, and the cap on the number of passes. All four are the reference
# implementation's own, with the tolerance divided by the two classes this compiler serves.
_SMALLEST_PROBABILITY = 1e-7
_LARGEST_PROBABILITY = 1 - 1e-7
_COUPLING_TOLERANCE = 0.005 / 2
_COUPLING_ITERATIONS = 100

_KERNEL_TEMPLATE = Template("""\
static float $name(
    const $element* row,
    const float* support,
    size_t width,
    int kind,
    float gamma,
    float coef0,
    float degree)
{
    size_t index;
    float total = 0.0f;
    if (kind == $rbf) {
        for (index = 0; index < width; ++index) {
            const float difference = (float)row[index] - support[index];
            total += difference * difference;
        }
        return expf(-gamma * total);
    }
    for (index = 0; index < width; ++index) {
        total += (float)row[index] * support[index];
    }
    if (kind == $poly) {
        return powf(total * gamma + coef0, degree);
    }
    if (kind == $sigmoid) {
        return tanhf(total * gamma + coef0);
    }
    return total;
}""")

_SCORES_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    const float* coefficients,
    const float* bias,
    size_t rows,
    size_t width,
    size_t columns,
    size_t stride,
    size_t bias_stride)
{
    size_t row, column, feature;
    for (row = 0; row < rows; ++row) {
        for (column = 0; column < columns; ++column) {
            float total = 0.0f;
            for (feature = 0; feature < width; ++feature) {
                total += (float)in[row * width + feature]
                    * coefficients[column * width + feature];
            }
            out[row * stride + column] = total + bias[column * bias_stride];
        }
    }
}""")

_SUPPORT_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    const float* support_vectors,
    const float* coefficients,
    size_t rows,
    size_t width,
    size_t supports,
    int kind,
    float gamma,
    float coef0,
    float degree,
    float rho)
{
    size_t row, index;
    for (row = 0; row < rows; ++row) {
        float total = 0.0f;
        for (index = 0; index < supports; ++index) {
            total += coefficients[index] * $kernel(
                in + row * width,
                support_vectors + index * width,
                width,
                kind,
                gamma,
                coef0,
                degree);
        }
        out[row] = total + rho;
    }
}""")

_PAIRWISE_TEMPLATE = Template("""\
static void $name(
    float* out,
    float* votes,
    float* values,
    const $element* in,
    const float* support_vectors,
    const float* coefficients,
    const float* rho,
    const int32_t* starts,
    const int32_t* counts,
    size_t rows,
    size_t width,
    size_t vectors,
    size_t classes,
    size_t stride,
    int kind,
    float gamma,
    float coef0,
    float degree)
{
    size_t row, index, first, second, taken, evaluated;
    for (row = 0; row < rows; ++row) {
        for (index = 0; index < vectors; ++index) {
            values[index] = $kernel(
                in + row * width,
                support_vectors + index * width,
                width,
                kind,
                gamma,
                coef0,
                degree);
        }
        for (index = 0; index < classes; ++index) {
            votes[row * classes + index] = 0.0f;
        }
        evaluated = 0;
        for (first = 0; first < classes; ++first) {
            for (second = first + 1; second < classes; ++second) {
                float total = rho[evaluated];
                float side = 0.0f;
                for (taken = 0; taken < (size_t)counts[first]; ++taken) {
                    const size_t at = (size_t)starts[first] + taken;
                    side += coefficients[(second - 1) * vectors + at] * values[at];
                }
                total += side;
                side = 0.0f;
                for (taken = 0; taken < (size_t)counts[second]; ++taken) {
                    const size_t at = (size_t)starts[second] + taken;
                    side += coefficients[first * vectors + at] * values[at];
                }
                total += side;
                out[row * stride + evaluated] = total;
                votes[row * classes + ((total > 0.0f) ? first : second)] += 1.0f;
                ++evaluated;
            }
        }
    }
}""")

# The two-class case of libsvm's pairwise coupling, which is what turns one decision value
# into a pair of probabilities. `probability` starts uniform and the loop drives the residual
# `Q*p - p'Qp` to zero; comparing that residual and the clamp bounds in double is what the
# reference's own mixed float32/Python-float arithmetic does.
_PROBABILITY_TEMPLATE = Template("""\
static void $name(
    float* out,
    const float* scores,
    size_t rows,
    float prob_a,
    float prob_b)
{
    size_t row, iteration, first, second;
    for (row = 0; row < rows; ++row) {
        const float raw = scores[row] * prob_a + prob_b;
        const float mapped = 1.0f / (1.0f + expf(-fabsf(raw)));
        float pair = 1.0f - ((raw < 0) ? (1.0f - mapped) : mapped);
        float coupling[2][2];
        float probability[2];
        float product[2];
        float complement, total, largest;
        if ((double)pair < $smallest_test) {
            pair = $smallest;
        }
        if ((double)pair > $largest_test) {
            pair = $largest;
        }
        complement = 1.0f - pair;
        coupling[0][0] = complement * complement;
        coupling[0][1] = -complement * pair;
        coupling[1][0] = coupling[0][1];
        coupling[1][1] = pair * pair;
        probability[0] = 0.5f;
        probability[1] = 0.5f;
        for (iteration = 0; iteration < $iterations; ++iteration) {
            for (first = 0; first < 2; ++first) {
                product[first] = coupling[first][0] * probability[0]
                    + coupling[first][1] * probability[1];
            }
            total = probability[0] * product[0] + probability[1] * product[1];
            largest = 0.0f;
            for (first = 0; first < 2; ++first) {
                const float error = fabsf(product[first] - total);
                /* `max(error, largest)` the way Python's own `max` takes it, which keeps a
                   value that is not a number rather than discarding it. */
                largest = (largest > error) ? largest : error;
            }
            if ((double)largest < $tolerance) {
                break;
            }
            for (first = 0; first < 2; ++first) {
                const float step =
                    (-product[first] + total) / coupling[first][first];
                const float scale = 1.0f + step;
                probability[first] += step;
                total = (total
                    + step * (step * coupling[first][first] + 2.0f * product[first]))
                    / (scale * scale);
                probability[0] /= scale;
                probability[1] /= scale;
                for (second = 0; second < 2; ++second) {
                    product[second] =
                        (product[second] + step * coupling[first][second]) / scale;
                }
            }
        }
        out[row * 2] = probability[0];
        out[row * 2 + 1] = probability[1];
    }
}""")

_ONE_CLASS_TEMPLATE = Template("""\
static void $name(float* scores, size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        scores[index] = (scores[index] > 0.0f) ? 1.0f : -1.0f;
    }
}""")

_THRESHOLD_TEMPLATE = Template("""\
static void $name(
    int64_t* labels,
    const float* scores,
    size_t rows,
    int64_t positive,
    float threshold)
{
    size_t row;
    for (row = 0; row < rows; ++row) {
        labels[row] = (scores[row] >= threshold) ? positive : 0;
    }
}""")


# --------------------------------------------------------------------------------------
# The four ops
# --------------------------------------------------------------------------------------


def _linear_regressor(context: NodeContext) -> NodeEmission:
    """`Y = X * coefficients' + intercepts`, one column per target."""
    source = context.require_input(0)
    result = float_output(context, 0)
    rows, width = _rows_and_width(context, source)
    targets = context.int_attribute("targets")
    if targets < 1:
        raise CompileError(
            f"Node `{context.label}`: LinearRegressor scores {targets} target(s); it takes "
            "at least one."
        )
    verify_shape(context, result, (rows, targets))

    coefficients = _coefficient_matrix(context, targets, width)
    emission = _scores(
        context,
        source,
        result.expr,
        coefficients,
        _intercepts(context, targets),
        rows=rows,
        width=width,
        columns=targets,
        stride=targets,
    )
    transform = named_transform(context)
    return extend(emission, post_transform(context, result, transform, rows, targets))


def _linear_classifier(context: NodeContext) -> NodeEmission:
    """The same scores, transformed, and the class the winning column names.

    A single score column against two class labels is the binary case: the score is paired
    with its own negation before the transform, and the pair is then read like any other row.
    A single column that is *not* paired names its class by a threshold instead — zero on
    untransformed scores, one half on anything the transform has mapped into `[0, 1]`.
    """
    source = context.require_input(0)
    labels = label_output(context, 0)
    scores = float_output(context, 1)
    rows, width = _rows_and_width(context, source)
    classes = _class_labels(context, required=False)
    coefficients = _required_floats(context, "coefficients")
    if width == 0 or len(coefficients) % width:
        raise CompileError(
            f"Node `{context.label}`: LinearClassifier holds {len(coefficients)} "
            f"coefficient(s) for an input of {width} feature(s); it takes one per feature "
            "per class."
        )
    produced = len(coefficients) // width
    columns = 2 if produced == 1 and len(classes) == 2 else produced
    # Read before the buffers are checked: ONNX's own inference sizes `Z` from the intercepts
    # as much as from the class labels, so a node that sets none reaches this first.
    intercepts = _intercepts(context, produced)
    verify_shape(context, labels, (rows,))
    verify_shape(context, scores, (rows, columns))

    transform = named_transform(context)
    emission = _scores(
        context,
        source,
        scores.expr,
        coefficients.reshape(produced, width),
        intercepts,
        rows=rows,
        width=width,
        columns=produced,
        stride=columns,
    )
    if columns != produced:
        emission = extend(
            emission, binary_scores(context, scores, rows, columns, complement=False)
        )
    emission = extend(
        emission, post_transform(context, scores, transform, rows, columns)
    )
    if columns > 1:
        if classes and len(classes) != columns:
            raise CompileError(
                f"Node `{context.label}`: LinearClassifier scores {columns} column(s) "
                f"against {len(classes)} class label(s); the winning column names a class, "
                "so it takes one label per column."
            )
        return extend(
            emission,
            argmax_labels(
                context,
                labels,
                scores.expr,
                scores.elem_type,
                classes or tuple(range(columns)),
                rows,
                columns,
            ),
        )
    return extend(
        emission, _threshold_labels(context, labels, scores, classes, transform, rows)
    )


def _svm_regressor(context: NodeContext) -> NodeEmission:
    """One score per row: a kernel against every support vector, or one plain dot product."""
    source = context.require_input(0)
    result = float_output(context, 0)
    rows, width = _rows_and_width(context, source)
    verify_shape(context, result, (rows, 1))

    coefficients = _required_floats(context, "coefficients")
    rho = _required_floats(context, "rho")
    supports = context.int_attribute("n_supports")
    if supports > 0:
        emission = _support_scores(
            context,
            source,
            result.expr,
            _support_vectors(context, supports, width),
            # The reference reads one coefficient per support vector and ignores the rest.
            coefficients[:supports],
            rho[0],
            rows=rows,
            width=width,
            supports=supports,
        )
    else:
        if len(coefficients) != width:
            raise CompileError(
                f"Node `{context.label}`: SVMRegressor holds {len(coefficients)} "
                f"coefficient(s) for an input of {width} feature(s); with no support "
                "vectors it scores one plain dot product, which takes one per feature."
            )
        emission = _scores(
            context,
            source,
            result.expr,
            coefficients.reshape(1, width),
            rho[:1],
            rows=rows,
            width=width,
            columns=1,
            stride=1,
        )
    if context.int_attribute("one_class"):
        emission = extend(emission, _one_class(context, result, rows))
    transform = named_transform(context)
    return extend(emission, post_transform(context, result, transform, rows, 1))


def _svm_classifier(context: NodeContext) -> NodeEmission:
    """Pairwise votes over support vectors, or one score per class, and then a label."""
    source = context.require_input(0)
    labels = label_output(context, 0)
    scores = float_output(context, 1)
    rows, width = _rows_and_width(context, source)
    classes = _class_labels(context, required=True)
    coefficients = _required_floats(context, "coefficients")
    rho = _required_floats(context, "rho")
    counts = _vector_counts(context)
    vectors = sum(counts)
    transform = named_transform(context)
    probabilities = _probability_pair(context, classes) if vectors > 0 else None

    # One score per class in the linear mode and one per class *pair* over support vectors,
    # unless those pairs are coupled into a probability per class.
    produced = (
        len(classes)
        if vectors == 0 or probabilities is not None
        else _pair_count(len(classes))
    )
    # A row of one score is paired with a second only where a second class is called for:
    # one `rho`, two class labels, and a transform that is not the one `write_scores` answers
    # before it ever reaches the pairing.
    paired = (
        produced == 1 and len(rho) == 1 and len(classes) == 2 and transform != PROBIT
    )
    columns = 2 if paired else produced
    verify_shape(context, labels, (rows,))
    verify_shape(context, scores, (rows, columns))

    emission, ranking = _svm_scores(
        context,
        source,
        scores,
        classes,
        coefficients,
        rho,
        counts,
        probabilities,
        rows=rows,
        width=width,
        vectors=vectors,
        columns=columns,
    )
    emission = extend(
        emission,
        argmax_labels(
            context,
            labels,
            ranking.expr,
            TensorProto.FLOAT,
            classes,
            rows,
            ranking.columns,
            _label_rule(classes, rho, coefficients, probabilities is not None),
        ),
    )
    if paired:
        emission = extend(
            emission, binary_scores(context, scores, rows, columns, complement=False)
        )
    # A single score the pairing left alone never reaches the transform at all, which is
    # where `write_scores` returns it — unless the transform is the one it answers first.
    if columns > 1 or transform == PROBIT:
        emission = extend(
            emission, post_transform(context, scores, transform, rows, columns)
        )
    return emission


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------


class _Ranking(NamedTuple):
    """The values a classifier's label is decided from, and how wide a row of them is.

    The support-vector mode ranks the classes by their votes and the linear mode by the
    scores themselves, so this is a votes buffer in the first case and the score tensor in
    the second — read before anything transforms it, which is the order the reference
    computes the two in.
    """

    expr: str
    columns: int


def _scores(
    context: NodeContext,
    source: TensorRef,
    destination: str,
    coefficients: np.ndarray,
    bias: np.ndarray,
    *,
    rows: int,
    width: int,
    columns: int,
    stride: int,
) -> NodeEmission:
    """One dot product per output column, offset by the bias that column carries.

    A bias of one value covers every column, which is how the reference broadcasts a single
    intercept — or the single `rho` a support vector machine shares — over the score matrix.
    """
    element = c_type(source.elem_type)
    name = f"{context.prefix}_ml_scores_{element}"
    definition = _SCORES_TEMPLATE.substitute(name=name, element=element)
    coefficient_data, coefficient_symbol = constant_data(
        context, "coefficients", coefficients
    )
    bias_data, bias_symbol = constant_data(context, "bias", bias)
    call = call_kernel(
        name,
        [
            destination,
            source.expr,
            coefficient_symbol,
            bias_symbol,
            f"{rows}u",
            f"{width}u",
            f"{columns}u",
            f"{stride}u",
            f"{int(len(bias) > 1)}u",
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(coefficient_data, bias_data),
    )


def _support_scores(
    context: NodeContext,
    source: TensorRef,
    destination: str,
    vectors: np.ndarray,
    coefficients: np.ndarray,
    rho: float,
    *,
    rows: int,
    width: int,
    supports: int,
) -> NodeEmission:
    """Every support vector's kernel value against the row, weighted and summed."""
    element = c_type(source.elem_type)
    kernel = _kernel_function(context, element)
    name = f"{context.prefix}_svm_supports_{element}"
    definition = _SUPPORT_TEMPLATE.substitute(
        name=name, element=element, kernel=kernel.name
    )
    vector_data, vector_symbol = constant_data(context, "support_vectors", vectors)
    coefficient_data, coefficient_symbol = constant_data(
        context, "coefficients", coefficients
    )
    call = call_kernel(
        name,
        [
            destination,
            source.expr,
            vector_symbol,
            coefficient_symbol,
            f"{rows}u",
            f"{width}u",
            f"{supports}u",
            *_kernel_arguments(context),
            scalar_literal(rho, TensorProto.FLOAT),
        ],
    )
    return NodeEmission(
        functions=(kernel, CFunction(name, definition)),
        statements=(call,),
        constants=(vector_data, coefficient_data),
    )


def _svm_scores(
    context: NodeContext,
    source: TensorRef,
    scores: TensorRef,
    classes: tuple[int, ...],
    coefficients: np.ndarray,
    rho: np.ndarray,
    counts: list[int],
    probabilities: tuple[float, float] | None,
    *,
    rows: int,
    width: int,
    vectors: int,
    columns: int,
) -> tuple[NodeEmission, _Ranking]:
    """The scores a classifier's row holds, and what its label is then ranked from."""
    if vectors == 0:
        if len(coefficients) != len(classes) * width:
            raise CompileError(
                f"Node `{context.label}`: SVMClassifier holds {len(coefficients)} "
                f"coefficient(s) for {len(classes)} class(es) over {width} feature(s); with "
                "no support vectors it scores one dot product per class, which takes one "
                "coefficient per feature per class."
            )
        emission = _scores(
            context,
            source,
            scores.expr,
            coefficients.reshape(len(classes), width),
            rho[:1],
            rows=rows,
            width=width,
            columns=len(classes),
            stride=columns,
        )
        return emission, _Ranking(scores.expr, len(classes))

    votes = ScratchBuffer(
        f"{context.prefix}_svm_votes", TensorProto.FLOAT, rows * len(classes)
    )
    # A node computing probabilities scores the class pairs into working storage first: the
    # result holds one column per class rather than one per pair.
    pairs = (
        None
        if probabilities is None
        else ScratchBuffer(f"{context.prefix}_svm_pairs", TensorProto.FLOAT, rows)
    )
    emission = _pairwise_scores(
        context,
        source,
        scores.expr if pairs is None else pairs.symbol,
        classes,
        coefficients,
        rho,
        counts,
        votes,
        rows=rows,
        width=width,
        vectors=vectors,
        stride=columns if pairs is None else 1,
    )
    if probabilities is not None and pairs is not None:
        emission = extend(
            emission, _probabilities(context, scores, pairs, probabilities, rows)
        )
    return emission, _Ranking(votes.symbol, len(classes))


def _pairwise_scores(
    context: NodeContext,
    source: TensorRef,
    destination: str,
    classes: tuple[int, ...],
    coefficients: np.ndarray,
    rho: np.ndarray,
    counts: list[int],
    votes: ScratchBuffer,
    *,
    rows: int,
    width: int,
    vectors: int,
    stride: int,
) -> NodeEmission:
    """libsvm's one-against-one scoring: a decision value and a vote for every class pair."""
    if len(classes) < 2:
        raise CompileError(
            f"Node `{context.label}`: SVMClassifier declares {len(classes)} class(es) over "
            "support vectors; the pairwise scheme it scores them with takes at least two, "
            "and ONNX's own reference implementation refuses anything less."
        )
    if len(counts) < len(classes):
        raise CompileError(
            f"Node `{context.label}`: `vectors_per_class` holds {len(counts)} entry(s) for "
            f"{len(classes)} class(es); it takes one per class."
        )
    if len(coefficients) % vectors:
        raise CompileError(
            f"Node `{context.label}`: SVMClassifier holds {len(coefficients)} "
            f"coefficient(s) over {vectors} support vector(s); it takes a whole number of "
            "rows of them."
        )
    if len(coefficients) // vectors < len(classes) - 1:
        raise CompileError(
            f"Node `{context.label}`: SVMClassifier holds {len(coefficients) // vectors} "
            f"row(s) of coefficients for {len(classes)} class(es); the pairwise scheme "
            f"reads {len(classes) - 1} of them."
        )
    pairs = _pair_count(len(classes))
    if len(rho) < pairs:
        raise CompileError(
            f"Node `{context.label}`: `rho` holds {len(rho)} value(s) for the {pairs} class "
            "pair(s) this node scores; it takes one per pair."
        )

    element = c_type(source.elem_type)
    kernel = _kernel_function(context, element)
    name = f"{context.prefix}_svm_pairwise_{element}"
    definition = _PAIRWISE_TEMPLATE.substitute(
        name=name, element=element, kernel=kernel.name
    )
    values = ScratchBuffer(f"{context.prefix}_svm_values", TensorProto.FLOAT, vectors)
    starts = np.cumsum([0, *counts[: len(classes) - 1]], dtype=np.int32)
    tables = [
        constant_data(
            context, "support_vectors", _support_vectors(context, vectors, width)
        ),
        constant_data(context, "coefficients", coefficients),
        constant_data(context, "rho", rho[:pairs]),
        constant_data(context, "starts", starts),
        constant_data(context, "counts", np.array(counts[: len(classes)], np.int32)),
    ]
    call = call_kernel(
        name,
        [
            destination,
            votes.symbol,
            values.symbol,
            source.expr,
            *(symbol for _, symbol in tables),
            f"{rows}u",
            f"{width}u",
            f"{vectors}u",
            f"{len(classes)}u",
            f"{stride}u",
            *_kernel_arguments(context),
        ],
    )
    return NodeEmission(
        functions=(kernel, CFunction(name, definition)),
        statements=(call,),
        scratch=(votes, values),
        constants=tuple(data for data, _ in tables),
    )


def _probabilities(
    context: NodeContext,
    scores: TensorRef,
    pairs: ScratchBuffer,
    pair: tuple[float, float],
    rows: int,
) -> NodeEmission:
    """The two class probabilities libsvm's Platt scaling turns one decision value into.

    `pairs` is the working storage the pairwise scoring wrote its one decision value per row
    into, which is where this reads them from.
    """
    name = f"{context.prefix}_svm_probabilities"
    definition = _PROBABILITY_TEMPLATE.substitute(
        name=name,
        smallest_test=scalar_literal(_SMALLEST_PROBABILITY, TensorProto.DOUBLE),
        smallest=scalar_literal(_SMALLEST_PROBABILITY, TensorProto.FLOAT),
        largest_test=scalar_literal(_LARGEST_PROBABILITY, TensorProto.DOUBLE),
        largest=scalar_literal(_LARGEST_PROBABILITY, TensorProto.FLOAT),
        tolerance=scalar_literal(_COUPLING_TOLERANCE, TensorProto.DOUBLE),
        iterations=f"{_COUPLING_ITERATIONS}u",
    )
    call = call_kernel(
        name,
        [
            scores.expr,
            pairs.symbol,
            f"{rows}u",
            scalar_literal(pair[0], TensorProto.FLOAT),
            scalar_literal(pair[1], TensorProto.FLOAT),
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        scratch=(pairs,),
    )


def _one_class(context: NodeContext, result: TensorRef, rows: int) -> NodeEmission:
    """Each score replaced by which side of zero it falls on."""
    name = f"{context.prefix}_svm_one_class"
    definition = _ONE_CLASS_TEMPLATE.substitute(name=name)
    call = call_kernel(name, [result.expr, f"{rows}u"])
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _threshold_labels(
    context: NodeContext,
    labels: TensorRef,
    scores: TensorRef,
    classes: tuple[int, ...],
    transform: int,
    rows: int,
) -> NodeEmission:
    """A single score column labelled by which side of a threshold it falls on.

    The threshold is zero while the scores are the raw ones and one half once a transform has
    mapped them onto a probability, and a row below it is labelled 0 whatever the class table
    says — both of which are the reference implementation's.
    """
    name = f"{context.prefix}_ml_threshold"
    definition = _THRESHOLD_TEMPLATE.substitute(name=name)
    call = call_kernel(
        name,
        [
            labels.expr,
            scores.expr,
            f"{rows}u",
            scalar_literal(classes[0] if classes else 1, TensorProto.INT64),
            scalar_literal(0.0 if transform == NONE else 0.5, TensorProto.FLOAT),
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _kernel_function(context: NodeContext, element: str) -> CFunction:
    name = f"{context.prefix}_svm_kernel_{element}"
    return CFunction(
        name,
        _KERNEL_TEMPLATE.substitute(
            name=name, element=element, rbf=_RBF, poly=_POLY, sigmoid=_SIGMOID
        ),
    )


def _kernel_arguments(context: NodeContext) -> list[str]:
    """The kernel function the node names and the three parameters it reads."""
    declared = context.string_attribute("kernel_type")
    kind = choice(context, "kernel_type", declared.lower(), _KERNEL_TYPES)
    parameters = [float(value) for value in context.attribute("kernel_params", [])]
    if parameters and len(parameters) < _KERNEL_PARAMETERS:
        raise CompileError(
            f"Node `{context.label}`: `kernel_params` holds {len(parameters)} value(s); it "
            f"takes the {_KERNEL_PARAMETERS} its reference implementation reads — gamma, "
            "coef0 and degree — or none at all."
        )
    # No `kernel_params` leaves all three at zero, which is what the reference falls back on.
    gamma, coef0, degree = parameters[:_KERNEL_PARAMETERS] or [0.0, 0.0, 0.0]
    return [
        str(kind),
        scalar_literal(gamma, TensorProto.FLOAT),
        scalar_literal(coef0, TensorProto.FLOAT),
        # The exponent is read as a whole number and applied as one.
        scalar_literal(float(int(degree)), TensorProto.FLOAT),
    ]


# --------------------------------------------------------------------------------------
# Reading the attribute tables
# --------------------------------------------------------------------------------------


def _pair_count(classes: int) -> int:
    return classes * (classes - 1) // 2


def _label_rule(
    classes: tuple[int, ...],
    rho: np.ndarray,
    coefficients: np.ndarray,
    probabilities: bool,
) -> int:
    """Which of `set_score_svm`'s three readings of the winning column this node takes."""
    if len(rho) != 1:
        return ARGMAX_LABEL
    if len(classes) != 2:
        return SIGN_LABEL
    positive = bool(coefficients.size) and float(coefficients.min()) >= 0
    return POSITIVE_CLASS if positive and not probabilities else ARGMAX_LABEL


def _probability_pair(
    context: NodeContext, classes: tuple[int, ...]
) -> tuple[float, float] | None:
    """`prob_a`/`prob_b`, or None where the node carries no probabilities.

    Only the two-class ensemble is served: the coupling the reference solves for three
    classes and up is built from a matrix it fills by broadcast where libsvm writes one
    entry per pair, so its answer and onnxruntime's differ and neither can vouch for a
    kernel. The two agree exactly for a single pair.
    """
    first = [float(value) for value in context.attribute("prob_a", [])]
    second = [float(value) for value in context.attribute("prob_b", [])]
    if not first:
        return None
    if len(classes) != 2:
        raise CompileError(
            f"Node `{context.label}`: `prob_a` couples the {_pair_count(len(classes))} "
            f"pairwise score(s) of {len(classes)} classes into probabilities, which the C "
            "compiler supports for two classes only — ONNX's own reference implementation "
            "and onnxruntime disagree on the coupling beyond that, so nothing can vouch for "
            "a kernel; re-export the model without `prob_a`/`prob_b`."
        )
    if not second:
        raise CompileError(
            f"Node `{context.label}`: SVMClassifier sets `prob_a` and no `prob_b`; the "
            "probability of a class pair is read from one of each."
        )
    return first[0], second[0]


def _vector_counts(context: NodeContext) -> list[int]:
    """`vectors_per_class`: how many of the support vectors each class brought with it.

    Every count is a length the emitted loops run to and an offset they start from, so a
    negative one would send them off both ends of the tables. The reference implementation
    reads it as an empty slice and scores the pair as zero, which is not a reading anything
    can vouch for.
    """
    counts = [int(value) for value in context.attribute("vectors_per_class", [])]
    if any(count < 0 for count in counts):
        raise CompileError(
            f"Node `{context.label}`: `vectors_per_class` holds a negative count "
            f"({', '.join(str(count) for count in counts)}); each entry is how many of the "
            "support vectors belong to one class."
        )
    return counts


def _class_labels(context: NodeContext, *, required: bool) -> tuple[int, ...]:
    """The class values a classifier labels its rows with."""
    if list(context.attribute("classlabels_strings", [])):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` labels its rows with the "
            "strings in `classlabels_strings`, and a tensor of STRING at run time is not "
            "something the C compiler supports."
        )
    integers = tuple(int(value) for value in context.attribute("classlabels_ints", []))
    if not integers and required:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` sets no `classlabels_ints`; "
            "it labels each row with one of them."
        )
    return integers


def _coefficient_matrix(context: NodeContext, columns: int, width: int) -> np.ndarray:
    """`coefficients` as the `[columns, width]` matrix the op reads it as."""
    values = _required_floats(context, "coefficients")
    if len(values) != columns * width:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` holds {len(values)} "
            f"coefficient(s) for {columns} target(s) over {width} feature(s); it takes one "
            "per feature per target."
        )
    return values.reshape(columns, width)


def _intercepts(context: NodeContext, columns: int) -> np.ndarray:
    """`intercepts`, which every one of these models has to carry.

    ONNX's own reference implementation reads a missing `intercepts` as a NaN and adds it to
    every score, while onnxruntime reads it as no offset at all; a model that sets none is
    refused rather than compiled to one of the two.
    """
    values = [float(value) for value in context.attribute("intercepts", [])]
    if not values:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` sets no `intercepts`. ONNX's "
            "own reference implementation scores every element of such a model as NaN and "
            "onnxruntime ignores the attribute, so nothing can vouch for a kernel built "
            "without them; re-export the model with explicit intercepts."
        )
    if len(values) not in (1, columns):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` holds {len(values)} "
            f"intercept(s) for {columns} score column(s); it takes either one per column or "
            "a single one for all of them."
        )
    return np.array(values, np.float32)


def _support_vectors(context: NodeContext, supports: int, width: int) -> np.ndarray:
    """`support_vectors` as the `[supports, width]` matrix the kernel compares against."""
    values = _required_floats(context, "support_vectors")
    if len(values) != supports * width:
        raise CompileError(
            f"Node `{context.label}`: `support_vectors` holds {len(values)} value(s) for "
            f"{supports} support vector(s) over {width} feature(s); it takes one per "
            "feature per vector."
        )
    return values.reshape(supports, width)


def _required_floats(context: NodeContext, name: str) -> np.ndarray:
    values = [float(value) for value in context.attribute(name, [])]
    if not values:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` requires the `{name}` "
            "attribute."
        )
    return np.array(values, np.float32)


def _rows_and_width(context: NodeContext, source: TensorRef) -> tuple[int, int]:
    """How many rows the node scores, and how many features each of them holds."""
    if len(source.shape) != 2:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads an `[N, F]` matrix of "
            f"features, and `{source.name}` has shape {list(source.shape)}."
        )
    return source.shape


register_kernel(ML_DOMAIN, "LinearRegressor", _VERSIONS, _linear_regressor)
register_kernel(ML_DOMAIN, "LinearClassifier", _VERSIONS, _linear_classifier)
register_kernel(ML_DOMAIN, "SVMRegressor", _VERSIONS, _svm_regressor)
register_kernel(ML_DOMAIN, "SVMClassifier", _VERSIONS, _svm_classifier)
