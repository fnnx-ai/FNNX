"""SoftmaxCrossEntropyLoss: a log-softmax along the class axis, read at the labels.

The op reads its operand as instances by classes by everything else — `[N, C, D...]`, which a
row-major buffer already lays out as `N` blocks of `C` planes of `D` elements — so the class
axis is walked by a stride and no reshape is ever emitted. Each `(instance, position)` pair
takes one pass for the largest logit, one for the exponentials, and one read at the label:
the loss of an entry is the negated log-softmax at its own class, and the optional `log_prob`
output is that same log-softmax written out in full.

The logarithm is taken of the *normalized probability* — `log(exp(x - max) / total)` rather
than the algebraically equal `x - max - log(total)` — because that is what ONNX's own
LogSoftmax and the reference evaluator compute, and the two differ where the exponential
underflows: the first answers a logit far below the largest with `-inf`, the second with a
large finite number.

`reduction`, the optional `weights` operand and whether `ignore_index` is set at all decide
how the per-entry losses are folded, and each combination is a kernel of its own rather than a
run-time branch — an entry ignored contributes nothing to either side of the weighted mean,
and a kernel that takes no weights must not carry a parameter it never reads past the
artifact's `-Werror` build. The index's *value* is an ordinary argument, as every other
attribute here is: it changes what a shared kernel skips, not the code that skips it.

Labels are validated at run time, since a label outside the class axis is an out-of-bounds
read that no static check can rule out, and the kernel returns the artifact's
invalid-argument status for one. ONNX defines the labels as class indices, so that is the
whole of the range; the reference raises on one at or past the class count but reads a
negative one from the end of the axis the way numpy indexes, which is an artifact of how it
gathers rather than anything the op is defined to compute.
"""

from __future__ import annotations

import math
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
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
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import math_suffix

# SoftmaxCrossEntropyLoss-12 differs from 13 only in the type constraints it allows; nothing
# vouches for the older revision, whose backend tests the corpus does not carry, so only the
# current one is served and an older import gets the standard unsupported-version error.
_VERSIONS = (13,)

_REDUCTIONS = ("none", "mean", "sum")

# The three passes over one entry's class axis, shared by both kernels below: the largest
# logit, the sum of the exponentials around it, and the probability of a single class.
_SOFTMAX_PASSES = """\
            $element largest = -INFINITY;
            $element total = $zero;
            for (cls = 0; cls < classes; ++cls) {
                const $element logit = in[base + cls * inner];
                if (logit > largest || isnan(logit)) {
                    largest = logit;
                }
            }
            for (cls = 0; cls < classes; ++cls) {
                total += exp$f(in[base + cls * inner] - largest);
            }"""

_LOG_PROB_TEMPLATE = f"""\
static void $name(
    $element* out,
    const $element* in,
    size_t rows,
    size_t classes,
    size_t inner)
{{
    size_t row, position, cls;
    for (row = 0; row < rows; ++row) {{
        for (position = 0; position < inner; ++position) {{
            const size_t base = row * classes * inner + position;
{_SOFTMAX_PASSES}
            for (cls = 0; cls < classes; ++cls) {{
                out[base + cls * inner] =
                    log$f(exp$f(in[base + cls * inner] - largest) / total);
            }}
        }}
    }}
}}"""

# `$ignored` skips an entry the node's `ignore_index` names, `$weight` is what the entry
# contributes to the weighted mean's denominator, and `$write` folds it: the three parts the
# attribute combination decides. The label check runs before the class axis is read, so a
# label the reference would raise on never addresses the buffer.
_LOSS_TEMPLATE = f"""\
static int $name(
    $element* out,
    const $element* in,
    const $label* labels,
$weight_parameter\
$ignore_parameter\
    size_t rows,
    size_t classes,
    size_t inner)
{{
$accumulators\
    size_t row, position, cls;
    for (row = 0; row < rows; ++row) {{
        for (position = 0; position < inner; ++position) {{
            const size_t entry = row * inner + position;
            const size_t base = row * classes * inner + position;
            const int64_t label = (int64_t)labels[entry];
$ignored\
            if (label < 0 || label >= (int64_t)classes) {{
                return 1;
            }}
{_SOFTMAX_PASSES}
            {{
                const $element chosen = exp$f(
                    in[base + (size_t)label * inner] - largest) / total;
                const $element weight = $weight;
                const $element value = weight * -log$f(chosen);
$write\
            }}
        }}
    }}
$finish\
    return 0;
}}"""

_IGNORED_NONE = """\
            if (label == ignore_index) {
                out[entry] = $zero;
                continue;
            }
"""

_IGNORED_REDUCED = """\
            if (label == ignore_index) {
                continue;
            }
"""

_ACCUMULATORS = """\
    $element loss_total = $zero;
    $element weight_total = $zero;
"""

_WRITE_ELEMENT = """\
                out[entry] = value;
"""

_WRITE_ACCUMULATED = """\
                loss_total += value;
                weight_total += weight;
"""

# The weighted mean divides by the weights it actually summed, not by the number of entries:
# an ignored one contributes to neither side, and one whose weight is zero contributes to
# neither either. A denominator of zero divides zero by zero, which is the NaN the reference
# answers with rather than a guess at what it should be.
_FINISH_MEAN = """\
    out[0] = loss_total / weight_total;
"""

_FINISH_SUM = """\
    out[0] = loss_total;
    (void)weight_total;
"""


def _softmax_cross_entropy_loss(context: NodeContext) -> NodeEmission:
    scores = context.require_input(0)
    labels = context.require_input(1)
    result = context.require_output(0)
    weights = context.optional_input(2)
    reduction = _reduction(context)
    rows, classes, inner = _partition(context, scores, labels)

    if scores.elem_type not in FLOAT_TYPES:
        raise CompileError(
            f"Node `{context.label}`: SoftmaxCrossEntropyLoss takes `{scores.name}` as "
            f"logits, so it must be a floating-point tensor; this one is "
            f"`{element_type_name(scores.elem_type)}`."
        )
    if weights is not None and weights.shape != (classes,):
        raise CompileError(
            f"Node `{context.label}`: `{weights.name}` gives one weight per class, so ONNX "
            f"gives it shape [{classes}]; this model gives it {list(weights.shape)}."
        )
    verify_shape(context, result, labels.shape if reduction == "none" else ())

    functions = []
    statements = []
    log_prob = context.outputs[1] if len(context.outputs) > 1 else None
    if log_prob is not None:
        verify_shape(context, log_prob, scores.shape)
        functions.append(_log_prob_kernel(context, scores))
        statements.append(
            call_kernel(
                functions[-1].name,
                [log_prob.expr, scores.expr, f"{rows}u", f"{classes}u", f"{inner}u"],
            )
        )

    kernel = _loss_kernel(context, scores, labels, weights, reduction)
    arguments = [result.expr, scores.expr, labels.expr]
    if weights is not None:
        arguments.append(weights.expr)
    ignore_index = context.attribute("ignore_index", None)
    if ignore_index is not None:
        arguments.append(scalar_literal(int(ignore_index), TensorProto.INT64))
    arguments += [f"{rows}u", f"{classes}u", f"{inner}u"]
    functions.append(kernel)
    statements.append(checked_call(context, kernel.name, arguments))
    return NodeEmission(functions=tuple(functions), statements=tuple(statements))


def _log_prob_kernel(context: NodeContext, scores: TensorRef) -> CFunction:
    name = kernel_name(context, "log_prob", numpy_dtype_name(scores.elem_type))
    return CFunction(
        name, _fill(_LOG_PROB_TEMPLATE, name=name, elem_type=scores.elem_type)
    )


def _loss_kernel(
    context: NodeContext,
    scores: TensorRef,
    labels: TensorRef,
    weights: TensorRef | None,
    reduction: str,
) -> CFunction:
    ignore_index = context.attribute("ignore_index", None)
    elem_type = scores.elem_type
    ignored = ""
    if ignore_index is not None:
        template = _IGNORED_NONE if reduction == "none" else _IGNORED_REDUCED
        ignored = _fill(template, elem_type=elem_type)
    name = kernel_name(
        context,
        reduction,
        "weighted" if weights is not None else "plain",
        "ignoring" if ignore_index is not None else "all",
        numpy_dtype_name(elem_type),
        numpy_dtype_name(labels.elem_type),
    )
    definition = _fill(
        _LOSS_TEMPLATE,
        name=name,
        elem_type=elem_type,
        label=c_type(labels.elem_type),
        weight_parameter=(
            f"    const {c_type(elem_type)}* weights,\n" if weights is not None else ""
        ),
        ignore_parameter=(
            "    int64_t ignore_index,\n" if ignore_index is not None else ""
        ),
        ignored=ignored,
        accumulators=(
            "" if reduction == "none" else _fill(_ACCUMULATORS, elem_type=elem_type)
        ),
        weight=(
            "weights[(size_t)label]"
            if weights is not None
            else scalar_literal(1, elem_type)
        ),
        write=_WRITE_ELEMENT if reduction == "none" else _WRITE_ACCUMULATED,
        finish={"none": "", "mean": _FINISH_MEAN, "sum": _FINISH_SUM}[reduction],
    )
    return CFunction(name, definition)


def _fill(template: str, *, elem_type: int, **fields: str) -> str:
    """Substitute the element-type placeholders every template here shares, then `fields`."""
    return Template(template).substitute(
        element=c_type(elem_type),
        f=math_suffix(elem_type),
        zero=scalar_literal(0, elem_type),
        **fields,
    )


def _reduction(context: NodeContext) -> str:
    reduction = context.attribute("reduction", b"mean")
    name = reduction.decode() if isinstance(reduction, bytes) else str(reduction)
    if name not in _REDUCTIONS:
        raise CompileError(
            f"Node `{context.label}`: `reduction` names `{name}`, which is not one of "
            f"{', '.join(_REDUCTIONS)}."
        )
    return name


def _partition(
    context: NodeContext, scores: TensorRef, labels: TensorRef
) -> tuple[int, int, int]:
    """The operand read as instances by classes by positions, checked against the labels."""
    if len(scores.shape) < 2:
        raise CompileError(
            f"Node `{context.label}`: SoftmaxCrossEntropyLoss reads `{scores.name}` as "
            "instances by classes by any further axes, so it needs a rank of at least 2; "
            f"this one has shape {list(scores.shape)}."
        )
    expected = (scores.shape[0], *scores.shape[2:])
    if labels.shape != expected:
        raise CompileError(
            f"Node `{context.label}`: `{labels.name}` names one class per entry of "
            f"`{scores.name}`, so ONNX gives it shape {list(expected)}; this model gives "
            f"it {list(labels.shape)}."
        )
    return scores.shape[0], scores.shape[1], math.prod(scores.shape[2:])


register_kernel("", "SoftmaxCrossEntropyLoss", _VERSIONS, _softmax_cross_entropy_loss)
