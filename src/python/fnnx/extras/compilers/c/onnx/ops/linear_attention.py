"""LinearAttention: the recurrence ONNX drives with a `Scan`, walked one token at a time.

The op carries one state matrix of shape `(d_k, d_v)` per batch item and per key/value head,
and walks the sequence forward through it. A step optionally decays the state by `exp(g_t)`
along the key dimension, optionally corrects the token's value by what the state already
answers for its key — `v_t <- beta_t * (v_t - S^T k_t)`, the delta rule — writes the outer
product `k_t (x) v_t` into the state, and then reads the *updated* state back through the
query: `o_t = scale * q_t^T S`. Nothing crosses a batch item or a key/value head, so those
two are the outer loops and everything inside them is sequential.

**Why a kernel and not the function body.** The rest of this op family compiles through the
body ONNX itself defines; this one cannot. `LinearAttention`'s body drives the recurrence
with a `Scan` over the time axis, and a `Scan` whose trip count is a run-time tensor is not
something constant folding can resolve away — it lands on the v1 unsupported surface
(`verify.py`'s `CONTROL_FLOW_OPS`), which is what the corpus's own `..._expanded` models,
the pre-inlined bodies, are refused for. SPEC's "native kernels only where expansion proves
insufficient" is exactly this case.

**What varies between call sites is geometry, not code.** Every extent and stride is a
kernel argument, so one shared `static` function per element type serves every node.
`update_rule` reaches the kernel as nothing at all: it decides only which of `decay` and
`beta` a node passes, and the reference refuses every other combination outright, so the two
operands being NULL or not *is* the rule. The two packings ONNX allows each of them — a
per-head scalar or a per-key-dimension vector for the decay, a per-head or a whole-batch
scalar for beta — are likewise two strides rather than two kernels, and grouped-query
attention is one more loop bound: `np.repeat` along the head axis makes the query heads one
key/value head serves consecutive.

**Where the recurrence accumulates.** The reference converts every operand to float32 and
runs the whole recurrence there whatever the tensors hold, casting the result back to the
query's type and `present_state` to `past_state`'s. The op's type constraints admit float16,
bfloat16 and float, of which this compiler supports only the last, so the accumulator and
the tensors' own type coincide — which is what lets the running state live directly in the
`present_state` output buffer, updated in place across the whole sequence, with no scratch
and no rounding between steps. The kernel still names the accumulator type in its own right,
and the C compiler rejects the call site if the two ever stop coinciding.

`chunk_size` is read here by nothing: ONNX documents it as a tuning hint for a
chunk-parallel implementation, and its own reference implementation ignores it outright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from string import Template

import onnx.defs
from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
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
from fnnx.extras.compilers.c.onnx.ops.axes import call_kernel, kernel_name, verify_shape
from fnnx.extras.compilers.c.onnx.ops.broadcast import math_suffix

# LinearAttention arrived at opset 27 and has had one revision.
_VERSIONS = (27,)

# The reference evaluator converts every operand to float32 and runs the recurrence there
# regardless of what the tensors hold; float is the one of the op's three element types this
# compiler supports, so the two coincide, and the kernel says which is which.
_ACCUMULATOR = TensorProto.FLOAT

_QUERY, _KEY, _VALUE, _PAST_STATE, _DECAY, _BETA = range(6)
_OUTPUT, _PRESENT_STATE = range(2)

# Which of the two optional gates each rule reads. The reference raises both ways — an
# operand a rule does not read is as much an error as one it needs and does not get — so
# these two flags are the whole of what `update_rule` decides.
_UPDATE_RULES: dict[str, tuple[bool, bool]] = {
    "linear": (False, False),
    "gated": (True, False),
    "delta": (False, True),
    "gated_delta": (True, True),
}

# The recurrence, per batch item and key/value head. Three properties shape the loops:
# the state's columns are independent — the delta correction of column `m` reads and then
# rewrites only column `m` — so no copy of the pre-write state is needed anywhere; the read
# happens after the write, which is what makes `o_t` a function of `S_t` rather than
# `S_{t-1}`; and the state matrix is `present_state` itself, so the sequence ends with the
# answer already in place.
_KERNEL_TEMPLATE = Template("""\
static void $name(
    $element* out,
    $accumulate* state,
    const $element* query,
    const $element* key,
    const $element* value,
    const $element* past,
    const $element* decay,
    const $element* beta,
    size_t batch,
    size_t steps,
    size_t q_heads,
    size_t kv_heads,
    size_t group,
    size_t d_k,
    size_t d_v,
    size_t decay_row,
    size_t decay_head_stride,
    size_t decay_dim_stride,
    size_t beta_row,
    size_t beta_head_stride,
    double scale)
{
    const size_t query_row = q_heads * d_k;
    const size_t key_row = kv_heads * d_k;
    const size_t value_row = kv_heads * d_v;
    const size_t out_row = q_heads * d_v;
    const size_t cells = d_k * d_v;
    size_t item, head, step, share, row, column;
    for (item = 0; item < batch; ++item) {
        for (head = 0; head < kv_heads; ++head) {
            const size_t origin = (item * kv_heads + head) * cells;
            $accumulate* carried = state + origin;
            for (row = 0; row < cells; ++row) {
                carried[row] = past == NULL ? $zero : ($accumulate)past[origin + row];
            }
            for (step = 0; step < steps; ++step) {
                const size_t token = item * steps + step;
                const $element* k_t = key + token * key_row + head * d_k;
                const $element* v_t = value + token * value_row + head * d_v;
                if (decay != NULL) {
                    /* The gate is in log space and broadcasts along the value dimension;
                       a per-head scalar reaches every key dimension through a zero
                       stride. */
                    const $element* g_t =
                        decay + token * decay_row + head * decay_head_stride;
                    for (row = 0; row < d_k; ++row) {
                        $accumulate* line = carried + row * d_v;
                        const $accumulate factor =
                            exp$f(($accumulate)g_t[row * decay_dim_stride]);
                        for (column = 0; column < d_v; ++column) {
                            line[column] *= factor;
                        }
                    }
                }
                for (column = 0; column < d_v; ++column) {
                    $accumulate written = ($accumulate)v_t[column];
                    if (beta != NULL) {
                        /* The delta rule writes only what this key is not already answered
                           with, at the rate beta names; the state it reads is the decayed
                           one, which is why the gate runs first. */
                        $accumulate retrieved = $zero;
                        for (row = 0; row < d_k; ++row) {
                            retrieved +=
                                carried[row * d_v + column] * ($accumulate)k_t[row];
                        }
                        written = ($accumulate)beta[token * beta_row
                            + head * beta_head_stride] * (written - retrieved);
                    }
                    for (row = 0; row < d_k; ++row) {
                        carried[row * d_v + column] += ($accumulate)k_t[row] * written;
                    }
                }
                for (share = 0; share < group; ++share) {
                    /* Grouped-query attention repeats the state along the head axis, so
                       the query heads this one serves are `group` consecutive ones. */
                    const size_t reader = head * group + share;
                    const $element* q_t = query + token * query_row + reader * d_k;
                    $element* answer = out + token * out_row + reader * d_v;
                    for (column = 0; column < d_v; ++column) {
                        $accumulate total = $zero;
                        for (row = 0; row < d_k; ++row) {
                            total += ($accumulate)q_t[row] * carried[row * d_v + column];
                        }
                        /* The derived scale is a numpy float64 in the reference, so the
                           product widens and rounds once on the way into the result; a
                           scale the node states outright is a Python float, which numpy
                           weakens to the sum's own type. The two differ by that rounding
                           alone, and this takes the wider of them. */
                        answer[column] = ($element)(scale * (double)total);
                    }
                }
            }
        }
    }
}""")


@dataclass(frozen=True)
class _Packing:
    """Where one of the two optional gates keeps the value a `(token, head, key dim)` reads.

    ONNX packs both into `(B, T, L)` and lets `L` say which granularity they carry, so the
    difference between them is three strides rather than two kernels: `row` is one token's
    worth, and a granularity coarser than the axis it feeds addresses it with a zero stride.
    """

    row: int
    head_stride: int
    dim_stride: int

    @staticmethod
    def absent_if(packing: _Packing | None) -> _Packing:
        return _Packing(0, 0, 0) if packing is None else packing

    @property
    def arguments(self) -> list[str]:
        return [f"{self.row}u", f"{self.head_stride}u", f"{self.dim_stride}u"]


@dataclass(frozen=True)
class _Geometry:
    """A node's shape, read off its operands and the two head counts it declares."""

    batch: int
    steps: int
    q_heads: int
    kv_heads: int
    d_k: int
    d_v: int

    @property
    def group(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.batch, self.steps, self.q_heads * self.d_v)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.batch, self.kv_heads, self.d_k, self.d_v)

    @property
    def arguments(self) -> list[str]:
        return [
            f"{self.batch}u",
            f"{self.steps}u",
            f"{self.q_heads}u",
            f"{self.kv_heads}u",
            f"{self.group}u",
            f"{self.d_k}u",
            f"{self.d_v}u",
        ]


def _linear_attention(context: NodeContext) -> NodeEmission:
    gating, delta = _update_rule(context)
    geometry = _geometry(context)
    decay = _decay_packing(context, geometry) if gating else None
    beta = _beta_packing(context, geometry) if delta else None
    operands = tuple(context.optional_input(index) for index in range(_BETA + 1))
    results = (context.require_output(_OUTPUT), context.require_output(_PRESENT_STATE))
    _verify_element_types(context, operands, results)
    verify_shape(context, results[_OUTPUT], geometry.output_shape)
    verify_shape(context, results[_PRESENT_STATE], geometry.state_shape)

    elem_type = context.require_input(_QUERY).elem_type
    name = kernel_name(context, numpy_dtype_name(elem_type))
    definition = _KERNEL_TEMPLATE.substitute(
        name=name,
        element=c_type(elem_type),
        accumulate=c_type(_ACCUMULATOR),
        f=math_suffix(_ACCUMULATOR),
        zero=scalar_literal(0, _ACCUMULATOR),
    )
    arguments = [
        results[_OUTPUT].expr,
        results[_PRESENT_STATE].expr,
        *(_operand(ref) for ref in operands),
        *geometry.arguments,
        # An absent gate reaches the kernel as the NULL pointer its branch reads, so the
        # strides that would have placed it are zeros nothing ever addresses through.
        *_Packing.absent_if(decay).arguments,
        # Beta is one value per `(token, head)`, so it has no key dimension to stride along.
        *_Packing.absent_if(beta).arguments[:2],
        scalar_literal(_scale(context, geometry), TensorProto.DOUBLE),
    ]
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def _operand(ref: TensorRef | None) -> str:
    return "NULL" if ref is None else ref.expr


def _update_rule(context: NodeContext) -> tuple[bool, bool]:
    """The rule's `(gating, delta correction)`, checked against the operands the node passes.

    ONNX ties the two optional operands to the rule in both directions: `decay` is required
    by `gated` and `gated_delta` and forbidden by the other two, `beta` by `delta` and
    `gated_delta`. The reference raises on every other combination rather than ignoring the
    stray operand, so there is nothing for a kernel to compute for one.
    """
    schema = onnx.defs.get_schema(
        context.node.op_type, context.since_version, context.domain
    )
    default = schema.attributes["update_rule"].default_value.s
    rule = context.attribute("update_rule", default).decode()
    if rule not in _UPDATE_RULES:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention`'s `update_rule` is `{rule}`, which "
            f"is not one of the recurrences ONNX defines: {', '.join(_UPDATE_RULES)}."
        )
    reads = _UPDATE_RULES[rule]
    for index, name, needed in zip((_DECAY, _BETA), ("decay", "beta"), reads):
        given = context.optional_input(index) is not None
        if given != needed:
            raise CompileError(
                f"Node `{context.label}`: `LinearAttention`'s `update_rule` is `{rule}`, "
                f"which {'requires' if needed else 'forbids'} the `{name}` input, but this "
                f"node {'leaves it out' if needed else 'passes one'}."
            )
    return reads


def _geometry(context: NodeContext) -> _Geometry:
    """The extents the kernel walks, from the packed operands and the declared head counts.

    `d_k` is the query's own head width and `d_v` the value's; the key carries `d_k` too,
    which is what makes the outer product it writes the shape of the state.
    """
    query, key, value = (
        context.require_input(index) for index in (_QUERY, _KEY, _VALUE)
    )
    for ref in (query, key, value):
        _verify_rank(context, ref, 3, "packed as (B, T, H * D)")
    q_heads = context.int_attribute("q_num_heads")
    kv_heads = context.int_attribute("kv_num_heads")
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` shares each of its {kv_heads} "
            f"key/value head(s) between an equal number of its {q_heads} query head(s), so "
            "`q_num_heads` has to be a positive multiple of `kv_num_heads`."
        )
    d_k = _head_width(context, query, q_heads, "query")
    d_v = _head_width(context, value, kv_heads, "value")
    geometry = _Geometry(
        batch=query.shape[0],
        steps=query.shape[1],
        q_heads=q_heads,
        kv_heads=kv_heads,
        d_k=d_k,
        d_v=d_v,
    )
    _verify_packed(context, key, (geometry.batch, geometry.steps, kv_heads * d_k))
    _verify_packed(context, value, (geometry.batch, geometry.steps, kv_heads * d_v))
    past = context.optional_input(_PAST_STATE)
    if past is not None and past.shape != geometry.state_shape:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` carries a state of shape "
            f"{list(geometry.state_shape)}, but its `past_state` `{past.name}` has shape "
            f"{list(past.shape)}."
        )
    return geometry


def _head_width(context: NodeContext, ref: TensorRef, heads: int, role: str) -> int:
    """How wide one head's slice of a packed operand is, which has to divide evenly."""
    packed = ref.shape[2]
    if packed % heads != 0:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` packs {heads} head(s) into the "
            f"last dimension of its {role} `{ref.name}`, which holds {packed} element(s) "
            "and is not divisible by that."
        )
    return packed // heads


def _decay_packing(context: NodeContext, geometry: _Geometry) -> _Packing:
    """Where the decay gate of a `(token, head, key dim)` sits in a `(B, T, L)` operand.

    `L` says the granularity: one value per head, broadcast across the key dimensions, or
    one per key dimension. The per-head reading is tried first, as the reference does, so
    the two agree where a `d_k` of one makes both apply and mean the same thing.
    """
    decay = context.require_input(_DECAY)
    _verify_rank(context, decay, 3, "packed as (B, T, H_kv) or (B, T, H_kv * d_k)")
    heads, packed = geometry.kv_heads, decay.shape[2]
    if packed == heads:
        packing = _Packing(row=heads, head_stride=1, dim_stride=0)
    elif packed == heads * geometry.d_k:
        packing = _Packing(row=packed, head_stride=geometry.d_k, dim_stride=1)
    else:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` reads its `decay` `{decay.name}` "
            f"either per head — {heads} value(s) — or per key dimension — "
            f"{heads * geometry.d_k} — but its last dimension holds {packed}."
        )
    _verify_packed(context, decay, (geometry.batch, geometry.steps, packed))
    return packing


def _beta_packing(context: NodeContext, geometry: _Geometry) -> _Packing:
    """Where the update rate of a `(token, head)` sits in a `(B, T, L)` operand.

    One value per head, or a single one the whole batch item shares — which the reference
    reaches by broadcasting the head axis, and this by a zero stride.
    """
    beta = context.require_input(_BETA)
    _verify_rank(context, beta, 3, "packed as (B, T, H_kv) or (B, T, 1)")
    heads, packed = geometry.kv_heads, beta.shape[2]
    if packed not in (heads, 1):
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` reads its `beta` `{beta.name}` "
            f"either per head — {heads} value(s) — or as one the heads share, but its last "
            f"dimension holds {packed}."
        )
    _verify_packed(context, beta, (geometry.batch, geometry.steps, packed))
    return _Packing(row=packed, head_stride=1 if packed == heads else 0, dim_stride=0)


def _scale(context: NodeContext, geometry: _Geometry) -> float:
    """What the read is scaled by: the node's own factor, or `1/sqrt(d_k)` for its default.

    ONNX states the default as the attribute value 0, which no model could mean literally —
    it would answer every query with zero — and the reference reads it as the request for
    the derived factor. A `d_k` of zero leaves that an infinity, which is what numpy's own
    division by `sqrt(0)` yields and what the whole result then rests on: with no key
    dimension to sum over, every answer is `inf * 0`, a NaN, in the reference and here alike.
    """
    scale = context.float_attribute("scale")
    if scale != 0.0:
        return scale
    return math.inf if geometry.d_k == 0 else 1.0 / math.sqrt(geometry.d_k)


def _verify_rank(context: NodeContext, ref: TensorRef, rank: int, form: str) -> None:
    if len(ref.shape) != rank:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` reads `{ref.name}` as a rank-"
            f"{rank} tensor {form}, but it has shape {list(ref.shape)}."
        )


def _verify_packed(
    context: NodeContext, ref: TensorRef, expected: tuple[int, ...]
) -> None:
    """Refuse to emit a kernel whose addressing disagrees with an operand it is handed.

    Every packed operand shares the batch and the sequence with the query, and its last
    dimension is fixed by the head counts the node declares; a graph that states another
    shape describes a different op, and this is where that stops rather than where it reads
    past a buffer.
    """
    if ref.shape != expected:
        raise CompileError(
            f"Node `{context.label}`: `LinearAttention` addresses `{ref.name}` as a tensor "
            f"of shape {list(expected)}, but it holds {list(ref.shape)}."
        )


def _verify_element_types(
    context: NodeContext,
    operands: tuple[TensorRef | None, ...],
    results: tuple[TensorRef, ...],
) -> None:
    """Refuse every element type but the one the recurrence accumulates in.

    ONNX types the state independently of the activations, and the reference accumulates in
    float32 whatever either of them holds. The running state lives in the `present_state`
    buffer here, which is sound exactly while that buffer holds the accumulator's own type;
    the compiler supports no other of the op's three types anyway, so the two never part.
    """
    for ref in (*operands, *results):
        if ref is not None and ref.elem_type != _ACCUMULATOR:
            raise CompileError(
                f"Node `{context.label}`: `LinearAttention` accumulates in "
                f"`{element_type_name(_ACCUMULATOR)}`, which is the only one of its element "
                f"types the C compiler serves, but `{ref.name}` is "
                f"`{element_type_name(ref.elem_type)}`."
            )


register_kernel("", "LinearAttention", _VERSIONS, _linear_attention)
