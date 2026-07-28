"""Attention and RotaryEmbedding: the transformer primitives, as one loop nest each.

ONNX defines both as functions, and the compiler's default for a function is to inline its
body. That was tried and does not work here: both bodies compute the shapes they reshape,
slice and expand to through `Shape`/`Size`/`Concat`/`Where` chains written at opset 23, and
constant folding refuses to execute a node whose (op, opset) the reference evaluator is not
version-faithful for — which, at the opset those bodies are written in, is every one of
them. The shape operands stay run-time values, the static-shape verifier rejects the body,
and nothing is compiled. So these two get native kernels, which is exactly the case the
"native kernels only where expansion proves insufficient" rule is for.

An attention head is one loop nest: for each (batch item, query head, query position) a row
of scores over the key positions, softmaxed, and read back as a weighted sum of the value
rows. Everything that differs between two nodes is geometry rather than code — how the
operands are laid out, how long the sequences are, which optional operands are present — so
one shared kernel per (element type, softmax precision, mask flavour) serves every node,
with the strides, the extents and the attribute values as arguments.

The two layouts ONNX allows — `(batch, head, sequence, size)` and
`(batch, sequence, head * size)` — are one tensor at two sets of strides, so they do not
fork the kernel: the call site passes the batch, head and sequence strides of whichever it
has, and the head size is contiguous in both.

**The arithmetic is `double` whatever the tensors hold.** The reference evaluator scales `Q`
and `K` by a numpy float64 scalar, which promotes everything from the QK product through the
softmax to float64, and rounds back to the tensor's own type only at the very end; a kernel
computing a float32 model in float32 would be a different operator. `softmax_precision` is
the one place ONNX narrows that back, and it is what the softmax type is read from.

`past_key`/`past_value` are never concatenated into working storage: the kernel reads a key
position out of the past cache or out of `K` depending on where it falls. The concatenation
ONNX calls `present_key`/`present_value` is written by a small kernel of its own, and only
when the node asks for it.

RotaryEmbedding rotates `rotary_embedding_dim / 2` coordinate pairs per (batch item,
position, head) and copies the tail beyond them through unchanged. Which two lanes a pair
is, and where the rotated pair is written back, is the whole of what `interleaved` changes,
so that too is an argument rather than a second kernel.

Where the reference evaluator and the schema prose disagree, the reference is what is
compiled — both test suites take their expected values from it — and every place that
happens is commented below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    c_type,
    element_type_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    broadcast_strides,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    kernel_name,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import combiner, extents, math_suffix

# Attention arrived at opset 23 and gained `nonpad_kv_seqlen` at 24; the kernel serves both,
# reading the operand as absent at the older one. RotaryEmbedding has had one revision.
_ATTENTION_VERSIONS = (23, 24)
_ROTARY_VERSIONS = (23,)

_PAST_KEY_INPUT = 4
_PAST_VALUE_INPUT = 5
_NONPAD_INPUT = 6

# The four intermediates `qk_matmul_output` may carry. ONNX's text distinguishes 0 ("the
# output of qk matmul") from 1 ("the output after the softcap operation"), but the reference
# evaluator softcaps the tensor it reports before it ever looks at the mode, so the two are
# one and the same value; the reference is what is compiled.
_OUTPUT_MODES = (0, 1, 2, 3)


# --------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------

# The bias added to one score. `attn_mask`'s last axis is the key positions, and ONNX pads it
# out to the total key length rather than broadcasting it, so it is addressed here while
# every axis before it arrives as a broadcast stride. The causal offset is the strict upper
# triangle of the columns from `past_seq` on, which is where the reference's `np.triu(..., 1)`
# lands, and `nonpad_kv_seqlen` masks off every column at or past a batch item's own length.
# `causal_stride` is which row of that triangle a query row reads: see `_causal_row_stride`.
_BIAS_TEMPLATE = Template("""\
static double $name(
    const $mask* mask,
    const size_t* strides,
    size_t mask_len,
    const int64_t* nonpad,
    size_t item,
    size_t head,
    size_t row,
    size_t column,
    size_t past_seq,
    int causal,
    size_t causal_stride)
{
    double bias = 0.0;
    if (mask != NULL) {
        const size_t base =
            item * strides[0] + head * strides[1] + row * strides[2];
$read
    }
    if (causal && column >= past_seq
        && column - past_seq > row * causal_stride) {
        bias += -INFINITY;
    }
    if (nonpad != NULL && (int64_t)column >= nonpad[item]) {
        bias += -INFINITY;
    }
    return bias;
}""")

# A float mask is the bias itself, and the columns past its own length are the negative
# infinity ONNX pads it with.
_ADDITIVE_MASK = """\
        bias = column < mask_len ? (double)mask[base + column] : -INFINITY;"""

# A boolean mask marks the entries that take part. The reference evaluator writes the two
# cases as different expressions — `(1 - mask) * -inf` under `is_causal`, `(1 - mask)` with
# its ones replaced by -inf otherwise — and the first turns an entry that *does* take part
# into `0 * -inf`, which is a NaN that then poisons its whole softmax row. That asymmetry is
# not in the prose; it is what the oracle computes, so it is what is compiled.
_BOOLEAN_MASK = """\
        const int taken = column < mask_len && mask[base + column] != 0;
        bias = taken ? (causal ? (double)NAN : 0.0) : -INFINITY;"""

# One query row against every key position: the scaled dot products, the softcap, the bias,
# the softmax, and the value rows read back through it. `scale` is already the square root
# the reference takes of it, and it multiplies `Q` and `K` separately rather than the product
# once — an algebraically equal rearrangement would round differently.
_ATTENTION_TEMPLATE = Template("""\
static void $name(
    $element* y,
    $element* qk_out,
    const $element* q,
    const $element* k,
    const $element* v,
    const $element* past_k,
    const $element* past_v,
    const $mask* mask,
    const int64_t* nonpad,
    $soft* scores,
    const size_t* q_strides,
    const size_t* k_strides,
    const size_t* v_strides,
    const size_t* y_strides,
    const size_t* mask_strides,
    size_t batch,
    size_t q_heads,
    size_t kv_heads,
    size_t q_seq,
    size_t kv_seq,
    size_t past_seq,
    size_t head_size,
    size_t v_head_size,
    size_t repeats,
    size_t mask_len,
    double scale,
    double softcap,
    int causal,
    size_t causal_stride,
    int mode)
{
    const size_t total = past_seq + kv_seq;
    size_t item, head, row, column, lane;
    for (item = 0; item < batch; ++item) {
        for (head = 0; head < q_heads; ++head) {
            /* Grouped-query attention repeats each key/value head `repeats` times over
               adjacent query heads, which is what `np.repeat` interleaves them as. */
            const size_t kv_head = head / repeats;
            const size_t past_base = (item * kv_heads + kv_head) * past_seq;
            const $element* q_head = q + item * q_strides[0] + head * q_strides[1];
            const $element* k_head = k + item * k_strides[0] + kv_head * k_strides[1];
            const $element* v_head = v + item * v_strides[0] + kv_head * v_strides[1];
            $element* y_head = y + item * y_strides[0] + head * y_strides[1];
            for (row = 0; row < q_seq; ++row) {
                const $element* q_row = q_head + row * q_strides[2];
                const size_t reported =
                    ((item * q_heads + head) * q_seq + row) * total;
                $soft largest = -INFINITY;
                $soft weight = $soft_zero;
                for (column = 0; column < total; ++column) {
                    const $element* k_row = column < past_seq
                        ? past_k + (past_base + column) * head_size
                        : k_head + (column - past_seq) * k_strides[2];
                    double score = 0.0;
                    for (lane = 0; lane < head_size; ++lane) {
                        score += ((double)q_row[lane] * scale)
                            * ((double)k_row[lane] * scale);
                    }
                    if (softcap > 0.0) {
                        score = tanh(score / softcap) * softcap;
                    }
                    if (qk_out != NULL && mode <= 1) {
                        qk_out[reported + column] = ($element)score;
                    }
                    score += $bias(mask, mask_strides, mask_len, nonpad,
                        item, head, row, column, past_seq, causal, causal_stride);
                    if (qk_out != NULL && mode == 2) {
                        qk_out[reported + column] = ($element)score;
                    }
                    scores[column] = ($soft)score;
                }
                /* Max-subtracted, as the reference's softmax is: a row that is entirely
                   -inf therefore leaves `-inf - -inf`, and comes out NaN. */
                for (column = 0; column < total; ++column) {
                    largest = $maximum(largest, scores[column]);
                }
                for (column = 0; column < total; ++column) {
                    scores[column] = exp$soft_suffix(scores[column] - largest);
                    weight += scores[column];
                }
                for (column = 0; column < total; ++column) {
                    scores[column] /= weight;
                    if (qk_out != NULL && mode == 3) {
                        qk_out[reported + column] = ($element)scores[column];
                    }
                }
                for (lane = 0; lane < v_head_size; ++lane) {
                    $product weighted = $product_zero;
                    for (column = 0; column < total; ++column) {
                        const $element* v_row = column < past_seq
                            ? past_v + (past_base + column) * v_head_size
                            : v_head + (column - past_seq) * v_strides[2];
                        weighted +=
                            ($product)scores[column] * ($product)v_row[lane];
                    }
                    y_head[row * y_strides[2] + lane] = ($element)weighted;
                }
            }
        }
    }
}""")

# `present_key` and `present_value` are the past cache and the incoming keys or values
# concatenated along the sequence axis — always as the 4-D cache layout, whichever layout the
# incoming operand itself has, which is what the strides are for.
_PRESENT_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* past,
    const $element* current,
    const size_t* strides,
    size_t batch,
    size_t heads,
    size_t past_seq,
    size_t kv_seq,
    size_t head_size)
{
    const size_t total = past_seq + kv_seq;
    size_t item, head, step, lane;
    for (item = 0; item < batch; ++item) {
        for (head = 0; head < heads; ++head) {
            const $element* source =
                current + item * strides[0] + head * strides[1];
            $element* written = out + (item * heads + head) * total * head_size;
            for (step = 0; step < past_seq; ++step) {
                const size_t cached =
                    ((item * heads + head) * past_seq + step) * head_size;
                for (lane = 0; lane < head_size; ++lane) {
                    written[step * head_size + lane] = past[cached + lane];
                }
            }
            for (step = 0; step < kv_seq; ++step) {
                for (lane = 0; lane < head_size; ++lane) {
                    written[(past_seq + step) * head_size + lane] =
                        source[step * strides[2] + lane];
                }
            }
        }
    }
}""")


@dataclass(frozen=True)
class _Geometry:
    """A node's extents, and where each operand's elements sit under its layout.

    Rank 4 packs `(batch, head, sequence, size)` and rank 3 `(batch, sequence, head * size)`;
    the kernel walks both through one batch, head and sequence stride, since the head size is
    contiguous either way. The caches and `qk_matmul_output` are 4-D whichever rank the node
    reads, which is what `cache_shape` and `qk_shape` state.
    """

    rank: int
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    past_seq: int
    head_size: int
    v_head_size: int

    @property
    def total_seq(self) -> int:
        return self.past_seq + self.kv_seq

    @property
    def repeats(self) -> int:
        """How many query heads share one key/value head, at least one."""
        return max(1, self.q_heads // self.kv_heads) if self.kv_heads else 1

    @property
    def rows(self) -> int:
        """How many score rows the node computes; nothing is emitted for none at all."""
        return self.batch * self.q_heads * self.q_seq

    def strides(self, heads: int, seq: int, size: int) -> tuple[int, int, int]:
        """`(batch, head, sequence)` strides of an operand at this node's layout."""
        if self.rank == 4:
            return (heads * seq * size, seq * size, size)
        return (seq * heads * size, size, heads * size)

    @property
    def y_shape(self) -> tuple[int, ...]:
        if self.rank == 4:
            return (self.batch, self.q_heads, self.q_seq, self.v_head_size)
        return (self.batch, self.q_seq, self.q_heads * self.v_head_size)

    @property
    def qk_shape(self) -> tuple[int, ...]:
        return (self.batch, self.q_heads, self.q_seq, self.total_seq)

    def cache_shape(self, size: int) -> tuple[int, ...]:
        return (self.batch, self.kv_heads, self.total_seq, size)


def _attention(context: NodeContext) -> NodeEmission:
    geometry = _geometry(context)
    element = _element_type(context)
    results = tuple(
        context.outputs[index] if index < len(context.outputs) else None
        for index in range(4)
    )
    verify_shape(context, context.require_output(0), geometry.y_shape)
    present = _present_emission(context, geometry, element, results)
    scoring = _scoring_emission(context, geometry, element, results)
    return NodeEmission(
        functions=present.functions + scoring.functions,
        statements=present.statements + scoring.statements,
        scratch=scoring.scratch,
    )


def _present_emission(
    context: NodeContext,
    geometry: _Geometry,
    element: int,
    results: tuple[TensorRef | None, ...],
) -> NodeEmission:
    """The key and value caches the node asks to have written back."""
    name = kernel_name(context, "present", c_type(element))
    # `present_key` is output 1 and reads `K`, which is input 1; `present_value` is output 2
    # and reads `V`, which is input 2 — the same index on either side.
    caches = (
        (1, _PAST_KEY_INPUT, geometry.head_size),
        (2, _PAST_VALUE_INPUT, geometry.v_head_size),
    )
    statements = []
    for slot, past_index, size in caches:
        result = results[slot]
        if result is None:
            continue
        verify_shape(context, result, geometry.cache_shape(size))
        if result.elem_count == 0:
            continue
        current = context.require_input(slot)
        past = context.optional_input(past_index)
        statements.append(
            call_kernel(
                name,
                [
                    result.expr,
                    "NULL" if past is None else past.expr,
                    current.expr,
                    extents(geometry.strides(geometry.kv_heads, geometry.kv_seq, size)),
                    f"{geometry.batch}u",
                    f"{geometry.kv_heads}u",
                    f"{geometry.past_seq}u",
                    f"{geometry.kv_seq}u",
                    f"{size}u",
                ],
            )
        )
    if not statements:
        return NodeEmission(functions=(), statements=())
    definition = _PRESENT_TEMPLATE.substitute(name=name, element=c_type(element))
    return NodeEmission(
        functions=(CFunction(name, definition),), statements=tuple(statements)
    )


def _scoring_emission(
    context: NodeContext,
    geometry: _Geometry,
    element: int,
    results: tuple[TensorRef | None, ...],
) -> NodeEmission:
    """The attention itself: the kernel, the row of scores it works in, and the call."""
    mask = context.optional_input(3)
    mask_type = _mask_type(context, mask, element)
    soft = _softmax_type(context)
    product = _product_type(soft, element)
    mode = _output_mode(context)
    reported = results[3]
    if reported is not None:
        verify_shape(context, reported, geometry.qk_shape)
    if geometry.rows == 0:
        return NodeEmission(functions=(), statements=())

    bias = _bias_function(context, mask_type)
    largest = combiner(context, soft, largest=True)
    name = kernel_name(
        context, c_type(element), f"soft{c_type(soft)}", f"mask{c_type(mask_type)}"
    )
    definition = _ATTENTION_TEMPLATE.substitute(
        name=name,
        element=c_type(element),
        mask=c_type(mask_type),
        soft=c_type(soft),
        product=c_type(product),
        soft_zero=scalar_literal(0, soft),
        product_zero=scalar_literal(0, product),
        soft_suffix=math_suffix(soft),
        maximum=largest.name,
        bias=bias.name,
    )
    scratch = ScratchBuffer(
        kernel_name(context, "scores", c_type(soft)), soft, geometry.total_seq
    )
    arguments = _scoring_arguments(context, geometry, mask, reported, scratch, mode)
    return NodeEmission(
        functions=(bias, largest, CFunction(name, definition)),
        statements=(call_kernel(name, arguments),),
        scratch=(scratch,),
    )


def _scoring_arguments(
    context: NodeContext,
    geometry: _Geometry,
    mask: TensorRef | None,
    reported: TensorRef | None,
    scratch: ScratchBuffer,
    mode: int,
) -> list[str]:
    optional = {
        index: context.optional_input(index)
        for index in (_PAST_KEY_INPUT, _PAST_VALUE_INPUT, _NONPAD_INPUT)
    }
    causal = int(context.int_attribute("is_causal") != 0)
    return [
        context.require_output(0).expr,
        "NULL" if reported is None else reported.expr,
        *(context.require_input(index).expr for index in range(3)),
        *(
            "NULL" if operand is None else operand.expr
            for operand in (
                optional[_PAST_KEY_INPUT],
                optional[_PAST_VALUE_INPUT],
                mask,
                optional[_NONPAD_INPUT],
            )
        ),
        scratch.symbol,
        extents(geometry.strides(geometry.q_heads, geometry.q_seq, geometry.head_size)),
        extents(
            geometry.strides(geometry.kv_heads, geometry.kv_seq, geometry.head_size)
        ),
        extents(
            geometry.strides(geometry.kv_heads, geometry.kv_seq, geometry.v_head_size)
        ),
        extents(
            geometry.strides(geometry.q_heads, geometry.q_seq, geometry.v_head_size)
        ),
        extents(_mask_strides(context, mask, geometry)),
        f"{geometry.batch}u",
        f"{geometry.q_heads}u",
        f"{geometry.kv_heads}u",
        f"{geometry.q_seq}u",
        f"{geometry.kv_seq}u",
        f"{geometry.past_seq}u",
        f"{geometry.head_size}u",
        f"{geometry.v_head_size}u",
        f"{geometry.repeats}u",
        f"{0 if mask is None else mask.shape[-1]}u",
        scalar_literal(_scale(context, geometry.head_size), TensorProto.DOUBLE),
        scalar_literal(context.float_attribute("softcap"), TensorProto.DOUBLE),
        str(causal),
        f"{_causal_row_stride(context, mask, causal)}u",
        str(mode),
    ]


def _bias_function(context: NodeContext, mask_type: int) -> CFunction:
    name = kernel_name(context, "bias", c_type(mask_type))
    read = _BOOLEAN_MASK if mask_type == TensorProto.BOOL else _ADDITIVE_MASK
    return CFunction(
        name,
        _BIAS_TEMPLATE.substitute(name=name, mask=c_type(mask_type), read=read),
    )


# --------------------------------------------------------------------------------------
# Reading an Attention node's geometry and types
# --------------------------------------------------------------------------------------


def _geometry(context: NodeContext) -> _Geometry:
    query, key, value = (context.require_input(index) for index in range(3))
    rank = len(query.shape)
    if rank not in (3, 4) or len(key.shape) != rank or len(value.shape) != rank:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `Q`, `K` and `V` as three tensors "
            f"of one rank — 4 for `(batch, head, sequence, size)`, 3 for "
            f"`(batch, sequence, head * size)` — but they have shapes "
            f"{list(query.shape)}, {list(key.shape)} and {list(value.shape)}."
        )
    if rank == 4:
        batch, q_heads, q_seq, head_size = query.shape
        kv_heads, kv_seq, key_size = key.shape[1:]
        value_heads, value_seq, v_head_size = value.shape[1:]
        _head_count(context, "q_num_heads", q_heads)
        _head_count(context, "kv_num_heads", kv_heads)
    else:
        q_heads = _head_count(context, "q_num_heads", None)
        kv_heads = _head_count(context, "kv_num_heads", None)
        batch, q_seq, q_hidden = query.shape
        kv_seq, key_hidden = key.shape[1:]
        value_seq, value_hidden = value.shape[1:]
        head_size = _head_size(context, query, q_hidden, q_heads)
        key_size = _head_size(context, key, key_hidden, kv_heads)
        v_head_size = _head_size(context, value, value_hidden, kv_heads)
        value_heads = kv_heads
    _verify_operands(
        context,
        batch=batch,
        heads=(q_heads, kv_heads, value_heads),
        sizes=(head_size, key_size),
        sequences=(kv_seq, value_seq),
    )
    past_seq = _past_length(context, batch, kv_heads, head_size, v_head_size)
    _verify_nonpad(context, batch)
    return _Geometry(
        rank=rank,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        past_seq=past_seq,
        head_size=head_size,
        v_head_size=v_head_size,
    )


def _head_count(context: NodeContext, name: str, inferred: int | None) -> int:
    """The head count for one side of the node, from its shapes or from its attributes.

    ONNX defines the two attributes for the 3-D layout, where nothing else says how the
    hidden axis splits. The 4-D layout carries the count in the shape and the reference
    evaluator reads it there; a node stating a different one describes two tensors, and there
    is no telling which of them it meant.
    """
    declared = context.attribute(name, None)
    if inferred is None:
        if declared is None:
            raise CompileError(
                f"Node `{context.label}`: `Attention` reads `Q`, `K` and `V` as rank 3, "
                f"where only `{name}` says how the hidden axis splits into heads, and this "
                "node leaves it out."
            )
        return int(declared)
    if declared is not None and int(declared) != inferred:
        raise CompileError(
            f"Node `{context.label}`: `Attention` states `{name}` {int(declared)}, but its "
            f"rank-4 operands carry {inferred} head(s) on the axis ONNX reads that count "
            "off."
        )
    return inferred


def _head_size(
    context: NodeContext, operand: TensorRef, hidden: int, heads: int
) -> int:
    if heads <= 0 or hidden % heads:
        raise CompileError(
            f"Node `{context.label}`: `Attention` splits `{operand.name}`'s hidden axis of "
            f"{hidden} into {heads} head(s), which does not divide it."
        )
    return hidden // heads


def _verify_operands(
    context: NodeContext,
    *,
    batch: int,
    heads: tuple[int, int, int],
    sizes: tuple[int, int],
    sequences: tuple[int, int],
) -> None:
    """Refuse to emit a kernel whose addressing disagrees with the operands it is handed."""
    query, key, value = (context.require_input(index) for index in range(3))
    if key.shape[0] != batch or value.shape[0] != batch:
        raise CompileError(
            f"Node `{context.label}`: `Attention` attends one batch, but `{query.name}`, "
            f"`{key.name}` and `{value.name}` carry {batch}, {key.shape[0]} and "
            f"{value.shape[0]} item(s)."
        )
    if sizes[0] != sizes[1]:
        raise CompileError(
            f"Node `{context.label}`: `Attention` contracts `{query.name}` against "
            f"`{key.name}` over the head size, but theirs are {sizes[0]} and {sizes[1]}."
        )
    q_heads, kv_heads, value_heads = heads
    if kv_heads != value_heads or sequences[0] != sequences[1]:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{key.name}` and `{value.name}` at "
            f"the same key positions, but they carry {kv_heads} head(s) of "
            f"{sequences[0]} against {value_heads} of {sequences[1]}."
        )
    if q_heads % kv_heads if kv_heads else q_heads:
        raise CompileError(
            f"Node `{context.label}`: `Attention` shares each of the {kv_heads} key/value "
            f"head(s) between the query heads that follow it, which needs {q_heads} query "
            f"head(s) to be a multiple of {kv_heads}."
        )


def _past_length(
    context: NodeContext, batch: int, kv_heads: int, head_size: int, v_head_size: int
) -> int:
    """How many cached key positions precede `K`, with both caches checked against them."""
    past_key = context.optional_input(_PAST_KEY_INPUT)
    past_value = context.optional_input(_PAST_VALUE_INPUT)
    given = [cache for cache in (past_key, past_value) if cache is not None]
    if not given:
        return 0
    if past_key is None or past_value is None:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `past_key` and `past_value` as one "
            "cache, so ONNX defines them as used together; this node reads only "
            f"`{given[0].name}`."
        )
    if len(past_key.shape) != 4:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{past_key.name}` as "
            "`(batch, kv_num_heads, past_sequence_length, head_size)`, but it has shape "
            f"{list(past_key.shape)}."
        )
    past_seq = past_key.shape[2]
    # Pairs rather than a mapping: a node may name one tensor for both caches, and two
    # `TensorRef`s of one tensor are equal, so a mapping would drop one of the two checks.
    expected = (
        (past_key, (batch, kv_heads, past_seq, head_size)),
        (past_value, (batch, kv_heads, past_seq, v_head_size)),
    )
    for operand, shape in expected:
        if operand.shape != shape:
            raise CompileError(
                f"Node `{context.label}`: `Attention` reads `{operand.name}` as a cache of "
                f"shape {list(shape)}, but it has shape {list(operand.shape)}."
            )
    return past_seq


def _verify_nonpad(context: NodeContext, batch: int) -> None:
    nonpad = context.optional_input(_NONPAD_INPUT)
    if nonpad is not None and nonpad.shape != (batch,):
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{nonpad.name}` as one key length "
            f"per batch item — a tensor of shape [{batch}] — but it has shape "
            f"{list(nonpad.shape)}."
        )


def _element_type(context: NodeContext) -> int:
    """The element type `Q`, `K`, `V` and the caches share.

    ONNX constrains the query/key side and the value side separately, so a model may state
    two different floating-point types for them. The reference evaluator then computes in
    whatever numpy promotes the pair to, at every step of the chain; rather than reproduce
    that promotion lattice on a combination nothing in the corpus or the sweep exercises,
    the compiler serves the case where the operands agree.
    """
    query = context.require_input(0)
    element = query.elem_type
    for index in (1, 2, _PAST_KEY_INPUT, _PAST_VALUE_INPUT):
        operand = context.optional_input(index)
        if operand is not None and operand.elem_type != element:
            raise CompileError(
                f"Node `{context.label}`: `Attention` reads `{query.name}` as "
                f"`{element_type_name(element)}` and `{operand.name}` as "
                f"`{element_type_name(operand.elem_type)}`; this compiler attends one "
                "element type, and the two have to agree."
            )
    return element


def _mask_type(context: NodeContext, mask: TensorRef | None, element: int) -> int:
    """The element type the mask is read at, which is the flavour of bias the kernel adds.

    ONNX's type constraint admits every numeric type, while its own text defines the operand
    as "a boolean mask ... or a float mask of the same type as query, key, value" — and the
    reference evaluator cannot evaluate an integer mask under `is_causal` at all, since an
    integer array holds no -inf. Only the two forms the text names are compiled. A node
    without a mask takes the additive kernel with a null pointer.
    """
    if mask is None:
        return element
    if mask.elem_type in (TensorProto.BOOL, element):
        return mask.elem_type
    raise CompileError(
        f"Node `{context.label}`: `Attention` reads `{mask.name}` as "
        f"`{element_type_name(mask.elem_type)}`, but ONNX defines `attn_mask` as a boolean "
        f"mask or a float mask of the operands' own type, which is "
        f"`{element_type_name(element)}` here."
    )


def _softmax_type(context: NodeContext) -> int:
    """The element type the softmax runs in.

    `double` by default, and not the tensors' own type: the reference evaluator's scaling of
    `Q` and `K` is by a numpy float64 scalar, so everything the softmax reads has already
    been promoted to float64 whatever the model holds. `softmax_precision` is what narrows
    it back.
    """
    declared = context.attribute("softmax_precision", None)
    if declared is None:
        return TensorProto.DOUBLE
    precision = int(declared)
    if precision not in FLOAT_TYPES:
        raise CompileError(
            f"Node `{context.label}`: `Attention` states a `softmax_precision` of "
            f"`{element_type_name(precision)}`, but ONNX defines the attribute as the "
            "floating-point precision the softmax runs in, and this compiler computes in "
            f"{' and '.join(sorted(element_type_name(t) for t in FLOAT_TYPES))}."
        )
    return precision


def _product_type(soft: int, element: int) -> int:
    """The type the softmax weights are read back against the values in.

    numpy promotes the two operands of that product, so it is the wider of them.
    """
    if soft == TensorProto.DOUBLE or element == TensorProto.DOUBLE:
        return TensorProto.DOUBLE
    return TensorProto.FLOAT


def _output_mode(context: NodeContext) -> int:
    mode = context.int_attribute("qk_matmul_output_mode")
    if mode not in _OUTPUT_MODES:
        raise CompileError(
            f"Node `{context.label}`: `Attention` states a `qk_matmul_output_mode` of "
            f"{mode}, but ONNX defines only "
            f"{', '.join(str(value) for value in _OUTPUT_MODES)}."
        )
    return mode


def _scale(context: NodeContext, head_size: int) -> float:
    """What `Q` and `K` are each multiplied by before their product.

    ONNX's `scale` scales the product; the reference evaluator takes its square root and
    applies that to both operands, so that is what the kernel is handed. Both edges follow
    numpy rather than Python: the square root of a negative is a NaN, and the default of a
    head of no elements is an infinity that no dot product ever reads.
    """
    declared = context.attribute("scale", None)
    if declared is None:
        value = math.inf if head_size == 0 else 1.0 / math.sqrt(head_size)
    else:
        value = float(declared)
    return math.sqrt(value) if value >= 0.0 else math.nan


def _mask_strides(
    context: NodeContext, mask: TensorRef | None, geometry: _Geometry
) -> tuple[int, ...]:
    """Strides addressing `attn_mask` while walking batch item, query head and query row.

    Its last axis is the key positions, which ONNX pads out to the total key length rather
    than broadcasting, so the kernel addresses that one itself; everything before it
    broadcasts numpy-style onto the three axes the loops walk.
    """
    if mask is None:
        return (0, 0, 0)
    walked = (geometry.batch, geometry.q_heads, geometry.q_seq)
    if not 1 <= len(mask.shape) <= len(walked) + 1:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{mask.name}` of shape "
            f"{list(mask.shape)}, but ONNX defines `attn_mask` as broadcastable to "
            f"`(batch, q_num_heads, q_sequence_length, total_sequence_length)`."
        )
    if mask.shape[-1] > geometry.total_seq:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{mask.name}` along "
            f"{mask.shape[-1]} key position(s), but the node attends "
            f"{geometry.total_seq}; the reference evaluator pads a shorter mask out to "
            "the key axis and refuses a longer one outright, so nothing says what the "
            "columns past the end would mean."
        )
    leading = replace(mask, shape=mask.shape[:-1])
    stretched = broadcast_strides(leading, walked, node_label=context.label)
    # `broadcast_strides` reads the strides off the shape it is given, which is the mask
    # without its key axis; the real ones carry that axis' extent as a factor.
    return tuple(stride * mask.shape[-1] for stride in stretched)


def _causal_row_stride(
    context: NodeContext, mask: TensorRef | None, causal: int
) -> int:
    """Whether the causal triangle advances with the query row, or one row serves them all.

    The reference evaluator adds the triangle *into the mask* — `np.triu` over the extents it
    reads off `attn_mask.shape[-2:]` — and only then broadcasts the sum onto the scores. So a
    mask carrying one row on the query axis, which is the `(batch, 1, 1, total)` padding mask
    a decoder passes alongside `is_causal`, gets the triangle's first row and masks every
    query position alike rather than by its own place in the sequence. The schema's prose
    describes the other reading; the oracle both suites compare against computes this one.
    """
    if not causal or mask is None:
        return 1
    if len(mask.shape) < 2:
        raise CompileError(
            f"Node `{context.label}`: `Attention` reads `{mask.name}` of shape "
            f"{list(mask.shape)} under `is_causal`, where the reference evaluator takes the "
            "triangle's extent from the mask's own query axis — a mask of rank 1 has none, "
            "and the reference refuses such a node outright, so nothing vouches for what "
            "this one would compute."
        )
    return 0 if mask.shape[-2] == 1 else 1


# --------------------------------------------------------------------------------------
# RotaryEmbedding
# --------------------------------------------------------------------------------------

# One rotation per (batch item, position, head, pair). `interleaved` picks the two lanes a
# pair is made of — adjacent ones rather than the two halves — and writes the rotated pair
# back into those same lanes, so it changes addressing and nothing else. The lanes past
# `rotary_embedding_dim` are copied through untouched, which is what makes a partial rotation
# partial.
_ROTARY_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $element* cosine,
    const $element* sine,
    const int64_t* positions,
    const size_t* strides,
    const size_t* cache_strides,
    const size_t* position_strides,
    size_t batch,
    size_t seq,
    size_t heads,
    size_t head_size,
    size_t rotary_half,
    size_t cache_rows,
    size_t cache_row,
    int interleaved)
{
    size_t item, step, head, pair, lane;
    for (item = 0; item < batch; ++item) {
        for (step = 0; step < seq; ++step) {
            size_t cache_base =
                item * cache_strides[0] + step * cache_strides[1];
            if (positions != NULL) {
                ptrdiff_t position = (ptrdiff_t)positions[
                    item * position_strides[0] + step * position_strides[1]];
                /* numpy's own indexing, which is what the reference gathers with: a
                   negative index counts from the end, anything else is out of range. */
                if (position < 0) {
                    position += (ptrdiff_t)cache_rows;
                }
                if (position < 0 || position >= (ptrdiff_t)cache_rows) {
                    return 1;
                }
                cache_base = (size_t)position * cache_row;
            }
            for (head = 0; head < heads; ++head) {
                const size_t base =
                    item * strides[0] + step * strides[1] + head * strides[2];
                const size_t angles = cache_base + head * cache_strides[2];
                for (pair = 0; pair < rotary_half; ++pair) {
                    const size_t low = interleaved ? 2 * pair : pair;
                    const size_t high =
                        interleaved ? 2 * pair + 1 : pair + rotary_half;
                    const $element c = cosine[angles + pair];
                    const $element s = sine[angles + pair];
                    const $element x1 = in[base + low];
                    const $element x2 = in[base + high];
                    out[base + low] = c * x1 - s * x2;
                    out[base + high] = s * x1 + c * x2;
                }
                for (lane = 2 * rotary_half; lane < head_size; ++lane) {
                    out[base + lane] = in[base + lane];
                }
            }
        }
    }
    return 0;
}""")


def _rotary_embedding(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    verify_shape(context, result, source.shape)
    batch, seq, heads, head_size = _rotary_layout(context, source)
    rotary = context.int_attribute("rotary_embedding_dim") or head_size
    if rotary % 2 or not 0 <= rotary <= head_size:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` rotates the first {rotary} lane(s) "
            f"of a head of {head_size}, which ONNX splits into pairs — so it has to be even "
            "and no wider than the head."
        )
    positions = context.optional_input(3)
    cache_strides = _cache_strides(context, positions, (batch, seq, heads), rotary // 2)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = _rotary_element_type(context)
    name = kernel_name(context, c_type(element))
    definition = _ROTARY_TEMPLATE.substitute(name=name, element=c_type(element))
    cosine = context.require_input(1)
    arguments = [
        result.expr,
        source.expr,
        cosine.expr,
        context.require_input(2).expr,
        "NULL" if positions is None else positions.expr,
        extents(_rotary_strides(len(source.shape), seq, heads, head_size)),
        extents(cache_strides),
        extents(
            (0, 0)
            if positions is None
            else broadcast_strides(positions, (batch, seq), node_label=context.label)
        ),
        f"{batch}u",
        f"{seq}u",
        f"{heads}u",
        f"{head_size}u",
        f"{rotary // 2}u",
        f"{cosine.shape[0]}u",
        f"{cosine.shape[-1]}u",
        str(int(context.int_attribute("interleaved") != 0)),
    ]
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(checked_call(context, name, arguments),),
    )


def _rotary_layout(
    context: NodeContext, source: TensorRef
) -> tuple[int, int, int, int]:
    """`(batch, sequence, heads, head size)`, whichever of the two layouts the node reads."""
    if len(source.shape) == 4:
        batch, heads, seq, head_size = source.shape
        declared = context.attribute("num_heads", None)
        if declared is not None and int(declared) != heads:
            raise CompileError(
                f"Node `{context.label}`: `RotaryEmbedding` states `num_heads` "
                f"{int(declared)}, but its rank-4 `{source.name}` carries {heads} head(s) "
                "on the axis ONNX reads that count off."
            )
        return batch, seq, heads, head_size
    if len(source.shape) != 3:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` reads `{source.name}` as "
            "`(batch, head, sequence, size)` or `(batch, sequence, head * size)`, but it "
            f"has shape {list(source.shape)}."
        )
    heads = context.attribute("num_heads", None)
    if heads is None or int(heads) <= 0:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` reads `{source.name}` as rank 3, "
            "where only `num_heads` says how the hidden axis splits into heads, and this "
            f"node states {'none' if heads is None else int(heads)}."
        )
    batch, seq, hidden = source.shape
    if hidden % int(heads):
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` splits `{source.name}`'s hidden "
            f"axis of {hidden} into {int(heads)} head(s), which does not divide it."
        )
    return batch, seq, int(heads), hidden // int(heads)


def _rotary_strides(
    rank: int, seq: int, heads: int, head_size: int
) -> tuple[int, int, int]:
    """`(batch, sequence, head)` strides; the result carries the operand's own layout back."""
    if rank == 4:
        return (heads * seq * head_size, head_size, seq * head_size)
    return (seq * heads * head_size, heads * head_size, head_size)


def _cache_strides(
    context: NodeContext,
    positions: TensorRef | None,
    walked: tuple[int, int, int],
    half: int,
) -> tuple[int, ...]:
    """Strides addressing the sine and cosine caches over batch item, position and head.

    With `position_ids` the caches are gathered by row, which the kernel addresses itself, so
    it walks none of these axes. Without it they carry the batch and the position and are
    stretched over the heads, exactly as the reference's `expand_dims` at the head axis does.
    """
    cosine = context.require_input(1)
    sine = context.require_input(2)
    if cosine.shape != sine.shape:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` reads `{cosine.name}` and "
            f"`{sine.name}` as one pair of caches, but they have shapes "
            f"{list(cosine.shape)} and {list(sine.shape)}."
        )
    if not cosine.shape or cosine.shape[-1] != half:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` rotates {half} pair(s) per head, so "
            f"ONNX ends its caches on that axis, but `{cosine.name}` has shape "
            f"{list(cosine.shape)}."
        )
    expected = 2 if positions is not None else 3
    if len(cosine.shape) != expected:
        raise CompileError(
            f"Node `{context.label}`: `RotaryEmbedding` reads `{cosine.name}` as rank "
            f"{expected} — `(max_position_id_plus_1, rotary_embedding_dim / 2)` when "
            "`position_ids` gathers it, `(batch, sequence, rotary_embedding_dim / 2)` when "
            f"it stands as it is — but it has shape {list(cosine.shape)}."
        )
    if positions is not None:
        # Exactly rank 2, not "at most": the reference gathers the caches by these indices
        # and then inserts the head axis with `expand_dims(..., 2)`, which for a lower rank
        # lands past the angles instead of before them and rotates by a transposed cache.
        if len(positions.shape) != 2:
            raise CompileError(
                f"Node `{context.label}`: `RotaryEmbedding` reads `{positions.name}` as "
                f"`(batch, sequence)`, but it has shape {list(positions.shape)}."
            )
        return (0, 0, 0)
    stretched = replace(cosine, shape=(*cosine.shape[:-1], 1, half))
    return broadcast_strides(stretched, (*walked, half), node_label=context.label)[:3]


def _rotary_element_type(context: NodeContext) -> int:
    element = context.require_input(0).elem_type
    for index in (1, 2):
        operand = context.require_input(index)
        if operand.elem_type != element:
            raise CompileError(
                f"Node `{context.label}`: `RotaryEmbedding` rotates "
                f"`{element_type_name(element)}` values by `{operand.name}`, which is "
                f"`{element_type_name(operand.elem_type)}`; ONNX defines the operand and "
                "its caches as one type."
            )
    return element


register_kernel("", "Attention", _ATTENTION_VERSIONS, _attention)
register_kernel("", "RotaryEmbedding", _ROTARY_VERSIONS, _rotary_embedding)
