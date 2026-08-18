"""Einsum: the equation read as addressing, and the sum it leaves behind.

ONNX defines Einsum as numpy's, and numpy's is a statement about coordinates: every label in
the equation names an extent, every operand's elements are addressed by the labels its term
carries, and the result is the sum — over the labels the output term leaves out — of the
operands multiplied together. The equation is therefore a compile-time object, one stride per
operand per label, and what is left for the kernel is two loops: one over the result's
elements, one over the labels being summed. Every equation of a given arity and element type
comes out as the same kernel, told apart only by the extents and strides its call site passes.

A label repeated inside one term is a diagonal, and needs no code of its own: the strides of
the axes it names add up, so stepping that one coordinate steps along the diagonal. A label a
term does not carry contributes a zero stride, which is what makes an outer product, a summed
axis and a stretched operand the same loop.

numpy stretches a labelled axis of extent 1 against another operand's, and ONNX's own shape
inference does not. Where that disagreement reaches the result — the equation computes a
wider tensor than the buffer ONNX sized — the shape check refuses the node rather than
writing past it; where it stays inside a summed label, the sum is numpy's, which is what the
reference evaluator computes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
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
    kernel_name,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents

# Einsum has had a single revision since it arrived at opset 12.
_VERSIONS = (12,)

# A term is letters with at most one `...` among them; numpy reads nothing else, and neither
# does this.
_TERM = re.compile(r"([a-zA-Z]*)(\.\.\.)?([a-zA-Z]*)")

# What each axis an ellipsis covers is labelled with. The dots make it unspellable in an
# equation, so it can never collide with a label the equation names itself.
_BROADCAST_LABEL = "...{}"

_TEMPLATE = Template("""\
static void $name(
    $element* out,
$operands,
    size_t count,
    int rank,
    const size_t* shape,
$result_strides,
    size_t summed_count,
    int summed_rank,
    const size_t* summed_shape,
$summed_strides)
{
    size_t index, term;
    for (index = 0; index < count; ++index) {
        size_t remainder = index;
        $element total = $zero;
$bases
        int axis;
        for (axis = rank - 1; axis >= 0; --axis) {
            const size_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
$walk_result
        }
        for (term = 0; term < summed_count; ++term) {
            size_t rest = term;
$offsets
            int summed;
            for (summed = summed_rank - 1; summed >= 0; --summed) {
                const size_t coordinate = rest % summed_shape[summed];
                rest /= summed_shape[summed];
$walk_summed
            }
            total += $product;
        }
        out[index] = total;
    }
}""")


@dataclass(frozen=True)
class _Equation:
    """The equation as addressing: one label per axis of each operand and of the result."""

    terms: tuple[tuple[str, ...], ...]
    result: tuple[str, ...]
    extents: Mapping[str, int]

    @property
    def summed(self) -> tuple[str, ...]:
        """The labels the result leaves out, in the order the terms first name them."""
        kept = frozenset(self.result)
        ordered = dict.fromkeys(label for term in self.terms for label in term)
        return tuple(label for label in ordered if label not in kept)


def _einsum(context: NodeContext) -> NodeEmission:
    operands = tuple(
        context.require_input(index) for index in range(len(context.node.input))
    )
    result = context.require_output(0)
    equation = _parse(context, operands)
    verify_shape(
        context, result, [equation.extents[label] for label in equation.result]
    )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    summed = equation.summed
    strides = [
        _label_strides(term, operand.shape, equation.extents)
        for term, operand in zip(equation.terms, operands)
    ]
    element = c_type(result.elem_type)
    name = kernel_name(context, str(len(operands)), element)
    arguments = [
        result.expr,
        *(operand.expr for operand in operands),
        f"{result.elem_count}u",
        str(len(equation.result)),
        extents([equation.extents[label] for label in equation.result]),
        *(_stride_literal(stride, equation.result) for stride in strides),
        f"{math.prod(equation.extents[label] for label in summed)}u",
        str(len(summed)),
        extents([equation.extents[label] for label in summed]),
        *(_stride_literal(stride, summed) for stride in strides),
    ]
    definition = _definition(name, element, len(operands), result.elem_type)
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def _definition(name: str, element: str, arity: int, elem_type: int) -> str:
    """The kernel for `arity` operands of `element`, which every equation shares."""
    operands = range(arity)
    return _TEMPLATE.substitute(
        name=name,
        element=element,
        zero=scalar_literal(0, elem_type),
        operands=",\n".join(f"    const {element}* in{index}" for index in operands),
        result_strides=",\n".join(
            f"    const size_t* result_strides{index}" for index in operands
        ),
        summed_strides=",\n".join(
            f"    const size_t* summed_strides{index}" for index in operands
        ),
        bases="\n".join(f"        size_t base{index} = 0;" for index in operands),
        walk_result="\n".join(
            f"            base{index} += coordinate * result_strides{index}[axis];"
            for index in operands
        ),
        offsets="\n".join(
            f"            size_t offset{index} = base{index};" for index in operands
        ),
        walk_summed="\n".join(
            f"                offset{index} += coordinate * summed_strides{index}[summed];"
            for index in operands
        ),
        product=" * ".join(f"in{index}[offset{index}]" for index in operands),
    )


def _stride_literal(strides: Mapping[str, int], labels: Sequence[str]) -> str:
    return extents([strides.get(label, 0) for label in labels])


def _parse(context: NodeContext, operands: Sequence[TensorRef]) -> _Equation:
    """The equation, with every ellipsis expanded against the operands' own ranks.

    A term that does not address the operand it is paired with — one term too few, or a rank
    the term does not cover — is refused here as well as by ONNX's own shape inference, which
    is what a model reaches first: it would otherwise be a stride addressing a buffer by
    another operand's shape, and this is where that stops rather than where it reads past one.
    """
    text = _equation_text(context)
    term_texts, result_text = _split(context, text)
    if len(term_texts) != len(operands):
        raise CompileError(
            f"Node `{context.label}`: the equation `{text}` states {len(term_texts)} "
            f"term(s) for {len(operands)} operand(s); Einsum takes one term per operand."
        )
    terms = [_split_term(context, term) for term in term_texts]
    covered = [
        _ellipsis_rank(context, text, term, operand)
        for text, term, operand in zip(term_texts, terms, operands)
    ]
    broadcast = tuple(
        _BROADCAST_LABEL.format(axis) for axis in range(max(covered, default=0))
    )
    expanded = tuple(
        (*head, *broadcast[len(broadcast) - count :], *tail)
        for (head, _, tail), count in zip(terms, covered)
    )
    return _Equation(
        terms=expanded,
        result=_result_labels(context, result_text, expanded, broadcast),
        extents=_measure(context, expanded, operands),
    )


def _equation_text(context: NodeContext) -> str:
    """The `equation` attribute, with the spaces numpy ignores taken out.

    An equation of nothing at all is refused rather than read as the scalar term numpy takes
    it for: ONNX's own reference implementation rejects it outright, so nothing states what
    such a node computes.
    """
    value = context.attribute("equation", b"")
    text = value.decode() if isinstance(value, bytes) else str(value)
    stripped = text.strip().replace(" ", "")
    if not stripped:
        raise CompileError(
            f"Node `{context.label}`: its `equation` is empty, which ONNX's Einsum does "
            "not define."
        )
    return stripped


def _split(context: NodeContext, text: str) -> tuple[list[str], str | None]:
    """The equation's input terms, and its output term where it states one."""
    parts = text.split("->")
    if len(parts) > 2:
        raise CompileError(
            f"Node `{context.label}`: the equation `{text}` states its output more than "
            "once; ONNX's Einsum writes `->` at most once."
        )
    return parts[0].split(","), parts[1] if len(parts) == 2 else None


def _split_term(
    context: NodeContext, text: str
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    """One term as the labels before its ellipsis, whether it has one, and the ones after."""
    match = _TERM.fullmatch(text)
    if match is None:
        raise CompileError(
            f"Node `{context.label}`: `{text}` is not a term ONNX's Einsum defines: a term "
            "is written as letters, with at most one `...` among them."
        )
    head, ellipsis, tail = match.groups()
    return tuple(head), ellipsis is not None, tuple(tail)


def _ellipsis_rank(
    context: NodeContext,
    text: str,
    term: tuple[tuple[str, ...], bool, tuple[str, ...]],
    operand: TensorRef,
) -> int:
    """How many of the operand's axes this term's ellipsis stands for."""
    head, ellipsis, tail = term
    named = len(head) + len(tail)
    if named > len(operand.shape) or (not ellipsis and named != len(operand.shape)):
        raise CompileError(
            f"Node `{context.label}`: the term `{text}` names {named} label(s) for "
            f"`{operand.name}`, which has shape {list(operand.shape)}; a term names one "
            "label per axis, unless it carries `...` for the rest."
        )
    return len(operand.shape) - named if ellipsis else 0


def _result_labels(
    context: NodeContext,
    text: str | None,
    terms: Sequence[tuple[str, ...]],
    broadcast: tuple[str, ...],
) -> tuple[str, ...]:
    """The result's axes: the output term's labels, or the ones implicit mode leaves.

    Implicit mode is numpy's: the axes an ellipsis covers come first, then every label
    exactly one axis of the whole equation carries, in alphabetical order.
    """
    carried = Counter(label for term in terms for label in term)
    if text is None:
        return (
            *broadcast,
            *sorted(
                label
                for label, count in carried.items()
                if count == 1 and label not in broadcast
            ),
        )
    head, ellipsis, tail = _split_term(context, text)
    labels = (*head, *(broadcast if ellipsis else ()), *tail)
    for position, label in enumerate(labels):
        if label in labels[:position]:
            raise CompileError(
                f"Node `{context.label}`: the output term `{text}` names label `{label}` "
                "more than once; each axis of the result is one label."
            )
        if label not in carried:
            raise CompileError(
                f"Node `{context.label}`: the output term `{text}` names label `{label}`, "
                "which no operand's term carries."
            )
    return labels


def _measure(
    context: NodeContext,
    terms: Sequence[tuple[str, ...]],
    operands: Sequence[TensorRef],
) -> dict[str, int]:
    """Every label's extent, over each axis of each operand that carries it.

    numpy reads the two positions a label can repeat in differently: inside one term it is a
    diagonal, which is defined only over axes measuring alike, while across two operands it
    broadcasts, an extent of 1 stretching to the other's.
    """
    sizes: dict[str, int] = {}
    for term, operand in zip(terms, operands):
        carried: dict[str, int] = {}
        for label, extent in zip(term, operand.shape):
            diagonal = carried.setdefault(label, extent)
            if diagonal != extent:
                raise CompileError(
                    f"Node `{context.label}`: the term for `{operand.name}` names label "
                    f"`{label}` on two axes of shape {list(operand.shape)} measuring "
                    f"{diagonal} and {extent}; a repeated label is a diagonal, which ONNX "
                    "defines only over axes of equal extent."
                )
        for label, extent in carried.items():
            sizes[label] = _stretched(context, sizes.get(label), extent, label)
    return sizes


def _stretched(
    context: NodeContext, measured: int | None, extent: int, label: str
) -> int:
    """One label's extent so far against another operand's, which a 1 stretches to."""
    if measured is None or measured in (extent, 1):
        return extent
    if extent == 1:
        return measured
    raise CompileError(
        f"Node `{context.label}`: label `{label}` measures {measured} on one operand and "
        f"{extent} on another; ONNX's Einsum stretches an extent of 1 against another "
        "operand's, and defines nothing for two extents that differ otherwise."
    )


def _label_strides(
    term: Sequence[str], shape: Sequence[int], sizes: Mapping[str, int]
) -> dict[str, int]:
    """How far one step along each label moves through this operand's row-major buffer.

    The strides of every axis a label names add up, which is what walks a diagonal; an axis
    the operand is stretched along contributes nothing, which is what broadcasts it.
    """
    strides: dict[str, int] = {}
    for label, stride, extent in zip(term, row_major_strides(shape), shape):
        if extent == sizes[label]:
            strides[label] = strides.get(label, 0) + stride
    return strides


register_kernel("", "Einsum", _VERSIONS, _einsum)
