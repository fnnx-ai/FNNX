"""`TfIdfVectorizer`: counting the n-grams of a pool in a sequence of tokens.

Everything the op matches against — the pool, which slice of it holds the n-grams of each
length, and the column each n-gram counts into — arrives in attributes, so the search
structure is built at compile time and emitted as `static const` tables. The reference
implementation walks a trie of pool entries, one level per token of an n-gram; the same trie
is flattened here into a node table (the column a node counts into, and the slice of the edge
table its outgoing edges occupy) and an edge table (the token an edge is taken on, and the
node it leads to). The kernel is then the reference's own walk over those tables, at the skip
distances and gram lengths the node's attributes ask for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from string import Template

import numpy as np
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_type_name
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    constant_data,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    kernel_name,
    verify_shape,
)

_VERSIONS = (9,)

# What each weighting mode makes of a column's count, given the weights the node carries.
# ONNX defines the three over the same counts, differing in this expression alone, so the
# mode is a compile-time choice rather than a branch the kernel takes per element. Without
# weights `TFIDF` is the count itself, which is what the reference computes for it.
_WEIGHTINGS = {
    ("TF", False): "(float)count",
    ("TF", True): "(float)count",
    ("IDF", False): "count > 0 ? 1.0f : 0.0f",
    ("IDF", True): "count > 0 ? weights[column] : 0.0f",
    ("TFIDF", False): "(float)count",
    ("TFIDF", True): "weights[column] * (float)count",
}

_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    int64_t* counts,
    size_t rows,
    size_t columns,
    size_t width,
    const int64_t* tokens,
    const int32_t* targets,
    const int32_t* first_edge,
    const int32_t* edge_count,
    const int32_t* counted_column,
$weights    int min_gram,
    int max_gram,
    int max_skip)
{
    size_t row, column, index;
    for (index = 0; index < rows * width; ++index) {
        counts[index] = 0;
    }
    for (row = 0; row < rows; ++row) {
        int start_gram = min_gram;
        int skip;
        for (skip = 1; skip <= max_skip + 1; ++skip) {
            size_t start;
            for (start = 0; start < columns; ++start) {
                int32_t node = 0;
                int gram = 1;
                size_t item = start;
                if (start + (size_t)skip * (size_t)(start_gram - 1) >= columns) {
                    break;
                }
                while (edge_count[node] > 0 && gram <= max_gram && item < columns) {
                    const int64_t token = (int64_t)in[row * columns + item];
                    const int32_t last = first_edge[node] + edge_count[node];
                    int32_t edge, taken = -1;
                    for (edge = first_edge[node]; edge < last; ++edge) {
                        if (tokens[edge] == token) {
                            taken = targets[edge];
                            break;
                        }
                    }
                    if (taken < 0) {
                        break;
                    }
                    if (gram >= start_gram && counted_column[taken] >= 0) {
                        counts[row * width + (size_t)counted_column[taken]] += 1;
                    }
                    node = taken;
                    ++gram;
                    item += skip;
                }
            }
            if (start_gram == 1) {
                start_gram = 2;
                if (start_gram > max_gram) {
                    break;
                }
            }
        }
    }
    for (row = 0; row < rows; ++row) {
        for (column = 0; column < width; ++column) {
            const int64_t count = counts[row * width + column];
            out[row * width + column] = $weighting;
        }
    }
}""")


@dataclass
class _Trie:
    """The pool's n-grams as a trie, flattened into the tables the kernel indexes.

    One entry per node in `first_edge`, `edge_count` and `counted_column`; one per edge in
    `tokens` and `targets`. `counted_column` is -1 for a node no pool n-gram ends at, which
    is how a prefix that is only on the way to a longer n-gram counts nothing of its own.
    Node 0 is the root, and exists even for a pool the node registers nothing from.
    """

    counted_column: list[int] = field(default_factory=lambda: [-1])
    edges: list[dict[int, int]] = field(default_factory=lambda: [{}])

    def child(self, node: int, token: int) -> int:
        """The node `token` leads to from `node`, added if this is the first n-gram to use it."""
        existing = self.edges[node].get(token)
        if existing is not None:
            return existing
        self.edges[node][token] = len(self.edges)
        self.edges.append({})
        self.counted_column.append(-1)
        return len(self.edges) - 1

    @property
    def tokens(self) -> list[int]:
        return [token for edges in self.edges for token in edges]

    @property
    def targets(self) -> list[int]:
        return [target for edges in self.edges for target in edges.values()]

    @property
    def first_edge(self) -> list[int]:
        starts, total = [], 0
        for edges in self.edges:
            starts.append(total)
            total += len(edges)
        return starts

    @property
    def edge_count(self) -> list[int]:
        return [len(edges) for edges in self.edges]


def _tf_idf_vectorizer(context: NodeContext) -> NodeEmission:
    """One count per pool n-gram per row, weighted as the node's `mode` prescribes.

    The op reads its input as a batch of token sequences — a matrix is one sequence per row
    and anything of lower rank a single sequence — which is the reading the shape of its
    result follows from.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    if source.elem_type not in (TensorProto.INT32, TensorProto.INT64):
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer takes `int32` or `int64` tokens, not "
            f"`{element_type_name(source.elem_type)}`; a string pool is matched against a "
            "run-time string tensor, which the C compiler does not support."
        )
    if len(source.shape) > 2:
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer takes one token sequence or a batch "
            f"of them, but `{source.name}` has shape {list(source.shape)}."
        )
    # A scalar is one sequence of one token, its shape being the empty product; a vector is
    # one sequence of as many tokens as it holds, an empty one included.
    batched = len(source.shape) == 2
    rows = source.shape[0] if batched else 1
    columns = source.shape[1] if batched else math.prod(source.shape)

    indexes = [int(value) for value in context.attribute("ngram_indexes", [])]
    if not indexes or min(indexes) < 0:
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer needs a non-negative `ngram_indexes` "
            "entry for every n-gram of its pool."
        )
    width = max(indexes) + 1
    verify_shape(context, result, (rows, width) if batched else (width,))

    weights = [float(value) for value in context.attribute("weights", [])]
    if weights and len(weights) != width:
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer carries {len(weights)} weight(s) for "
            f"{width} output column(s); the op weights one per column."
        )
    return _emit(
        context, source, result, _trie(context, indexes), weights, rows, columns, width
    )


def _trie(context: NodeContext, indexes: Sequence[int]) -> _Trie:
    """The pool's n-grams, level by level, in the order ONNX numbers them.

    `ngram_counts` splits the pool by n-gram length: entry `i` is where the n-grams of length
    `i + 1` start, and they run to the next entry. Every n-gram of the pool takes the next
    identifier whether or not its length is one this node counts, so the identifiers — and
    with them the `ngram_indexes` entry each n-gram counts into — stay aligned to the pool
    however `min_gram_length` and `max_gram_length` narrow it.
    """
    minimum = context.int_attribute("min_gram_length")
    maximum = context.int_attribute("max_gram_length")
    if minimum < 1 or maximum < minimum:
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer needs 1 <= `min_gram_length` <= "
            f"`max_gram_length`, but they are {minimum} and {maximum}."
        )
    if context.attribute("pool_strings", []):
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer sets `pool_strings`, which matches "
            "against a run-time string tensor; the C compiler supports the `pool_int64s` "
            "form only."
        )
    pool = [int(value) for value in context.attribute("pool_int64s", [])]
    counts = [int(value) for value in context.attribute("ngram_counts", [])]

    trie = _Trie()
    identifier = 1
    for length, start in enumerate(counts, start=1):
        end = counts[length] if length < len(counts) else len(pool)
        available = (end - start) // length if end > start else 0
        if minimum <= length <= maximum:
            identifier = _register(
                context, trie, pool[start:end], length, available, identifier, indexes
            )
        else:
            identifier += available
    return trie


def _register(
    context: NodeContext,
    trie: _Trie,
    entries: Sequence[int],
    length: int,
    available: int,
    identifier: int,
    indexes: Sequence[int],
) -> int:
    """Add `available` n-grams of `length` consecutive tokens, returning the next identifier.

    An n-gram the pool lists twice keeps the last of its `ngram_indexes` entries, which is
    what re-walking the same path and overwriting the column at its end leaves behind.
    """
    position = 0
    for _ in range(available):
        node = 0
        for taken in range(1, length + 1):
            if position >= len(entries):
                break
            node = trie.child(node, entries[position])
            position += 1
            if taken == length:
                if identifier > len(indexes):
                    raise CompileError(
                        f"Node `{context.label}`: TfIdfVectorizer's pool holds more n-grams "
                        f"than its {len(indexes)} `ngram_indexes` entries account for."
                    )
                trie.counted_column[node] = indexes[identifier - 1]
                identifier += 1
    return identifier


def _emit(
    context: NodeContext,
    source: TensorRef,
    result: TensorRef,
    trie: _Trie,
    weights: Sequence[float],
    rows: int,
    columns: int,
    width: int,
) -> NodeEmission:
    mode = context.string_attribute("mode")
    weighting = _WEIGHTINGS.get((mode, bool(weights)))
    if weighting is None:
        raise CompileError(
            f"Node `{context.label}`: TfIdfVectorizer sets `mode` to `{mode}`; ONNX defines "
            "`TF`, `IDF` and `TFIDF`."
        )
    # `TF` ignores the weights a node may still carry, and so does `TFIDF` where there are
    # none to apply; passing a table the expression never reads would be an unused parameter,
    # which the artifact's `-Werror` build contract refuses.
    weighted = "weights[" in weighting
    element = c_type(source.elem_type)
    name = kernel_name(
        context, mode.lower(), "weighted" if weighted else "flat", element
    )
    definition = _TEMPLATE.substitute(
        name=name,
        element=element,
        weights="    const float* weights,\n" if weighted else "",
        weighting=weighting,
    )

    tables = [
        constant_data(context, role, np.array(values, dtype=dtype))
        for role, values, dtype in (
            ("tokens", trie.tokens, np.int64),
            ("targets", trie.targets, np.int32),
            ("first_edge", trie.first_edge, np.int32),
            ("edge_count", trie.edge_count, np.int32),
            ("counted_column", trie.counted_column, np.int32),
            *((("weights", weights, np.float32),) if weighted else ()),
        )
    ]
    counts = ScratchBuffer(f"{name}_counts", TensorProto.INT64, rows * width)
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            counts.symbol,
            f"{rows}u",
            f"{columns}u",
            f"{width}u",
            *(symbol for _, symbol in tables),
            str(context.int_attribute("min_gram_length")),
            str(context.int_attribute("max_gram_length")),
            str(context.int_attribute("max_skip_count")),
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        scratch=(counts,),
        constants=tuple(data for data, _ in tables),
    )


register_kernel("", "TfIdfVectorizer", _VERSIONS, _tf_idf_vectorizer)
