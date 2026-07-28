"""The recurrent layers: one time-stepped kernel per op, driven by call-site geometry.

An LSTM, a GRU and an RNN all walk a batch of sequences one time step at a time, carrying a
hidden state `H` — and, for the LSTM, a cell state `C` — from step to step. Everything that
varies between two nodes of the same op — how many steps, how wide the state is, which way
time runs, where the operands sit in memory — reaches the kernel as arguments, so one shared
`static` function per op and element type serves every node, and the states themselves live
in static scratch sized at compile time.

Two things shape the emission. A **direction** is an independent pass over the sequence with
its own weights, so the kernel computes one direction and a bidirectional node calls it twice,
with every operand offset onto that direction's slice. And **layout** only permutes where the
operands' elements sit, so it never reaches the kernel as a flag: the call site passes the
strides that place a time step and a batch item, which layout 1 simply reorders.

The three ops differ only in what one step computes, so they share a frame — the batch loop,
the per-item sequence length, the padding ONNX reports past a sequence's end, and the state
outputs — and each supplies the recurrence that runs inside it, along with the signature that
recurrence reads. That is what `_Layer` collects.

The activations ONNX lets a node choose — `f` over the gates, `g` over the cell or hidden
candidate, `h` over the LSTM's cell state on the way out — are emitted as one small function
each, taking the alpha and beta that parameterize them, and reach the kernel as function
pointers. Only the ones a node actually names are emitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from string import Template

import onnx.defs

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    ScratchBuffer,
    TensorRef,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import checked_call, verify_shape
from fnnx.extras.compilers.c.onnx.ops.broadcast import expand

# The recurrent ops arrived at opset 1 and were revised repeatedly since — 7 dropped the
# legacy broadcast attributes, 14 added `layout`, 22 widened the element types. Only 22 is
# claimed: it is the revision every recurrent test in the backend corpus imports, and the one
# the reference evaluator is version-faithful for across the whole family, so it is the only
# one anything can vouch for. A model importing an older one gets the unsupported-version
# error.
_VERSIONS = (22,)

_DIRECTIONS = {"forward": 1, "reverse": 1, "bidirectional": 2}

# The LSTM's peephole vector `P` carries only the three gates that read the cell state —
# input, output, forget — where `W`, `R` and `B` carry all four.
_PEEPHOLES = 3


@dataclass(frozen=True)
class _Activation:
    """One of the activations ONNX's recurrent ops may be told to run.

    `expression` is C over `x`, `alpha` and `beta`, as ONNX's LSTM specification writes the
    function. `schema` names the ONNX operator whose own defaults for alpha and beta apply,
    which is how a node that selects an activation without parameterizing it gets the values
    ONNX's operator would have applied; the two activations ONNX defines no operator for have
    no such default, so a node naming one has to state the parameters itself.
    """

    expression: str
    alpha: bool = False
    beta: bool = False
    schema: str | None = None


# The activation functions ONNX's recurrent documentation lists, each as the formula written
# there. Sigmoid takes the numerically stable form the Sigmoid kernel uses — whichever branch
# is taken, the exponent is of a negative number, so it underflows to zero rather than
# overflowing — and Relu propagates NaN the way numpy's `maximum`, which ONNX's own Relu is
# defined through, does.
_ACTIVATIONS: dict[str, _Activation] = {
    "Affine": _Activation("alpha * x + beta", alpha=True, beta=True),
    "Elu": _Activation(
        "(x >= $zero) ? x : alpha * (exp$f(x) - $one)", alpha=True, schema="Elu"
    ),
    "HardSigmoid": _Activation(
        "fmin$f(fmax$f(alpha * x + beta, $zero), $one)",
        alpha=True,
        beta=True,
        schema="HardSigmoid",
    ),
    "LeakyRelu": _Activation(
        "(x >= $zero) ? x : alpha * x", alpha=True, schema="LeakyRelu"
    ),
    "Relu": _Activation("(x > $zero || isnan(x)) ? x : $zero"),
    "ScaledTanh": _Activation("alpha * tanh$f(beta * x)", alpha=True, beta=True),
    "Sigmoid": _Activation(
        "x > $zero ? $one / ($one + exp$f(-x)) : exp$f(x) / ($one + exp$f(x))"
    ),
    "Softplus": _Activation("log$f($one + exp$f(x))"),
    "Softsign": _Activation("x / ($one + fabs$f(x))"),
    "Tanh": _Activation("tanh$f(x)"),
    "ThresholdedRelu": _Activation(
        "(x >= alpha) ? x : $zero", alpha=True, schema="ThresholdedRelu"
    ),
}

_ACTIVATION_TEMPLATE = Template("""\
static $element $name($element x, $element alpha, $element beta)
{
    (void)alpha;
    (void)beta;
    return $expression;
}""")

# The cell clip, as two comparisons rather than `fmin`/`fmax` so that a NaN — which fails both
# — reaches the activation instead of being replaced by a bound.
_CLIP_TEMPLATE = Template("""\
static $element $name($element x, $element clip)
{
    return x < -clip ? -clip : (x > clip ? clip : x);
}""")


# --------------------------------------------------------------------------------------
# The frame every recurrent kernel runs inside
# --------------------------------------------------------------------------------------

# One direction of one layer. The recurrence is per batch item — a sequence's gates read only
# its own step and its own state — so the batch is the outer loop and the state buffers hold
# one item's worth, which is also what lets each item stop at its own sequence length. What
# the op computes from a step sits in `$recurrence`, which reads `values` and the current
# `hidden` and leaves the new state in `hidden`.
_FRAME = Template("""\
static int $name(
$parameters)
{
    const size_t gate_count = $gates * hidden_size;
    size_t item, step, unit, term;
    for (item = 0; item < batch_size; ++item) {
        const $element* sequence = x + item * x_batch_stride;
        const size_t state_base = item * state_batch_stride;
        size_t length = seq_length;
        if (lengths != NULL) {
            /* ONNX defines a length as a position in the padded sequence; anything else
               names a step that is not there. */
            if (lengths[item] < 0 || (size_t)lengths[item] > seq_length) {
                return 1;
            }
            length = (size_t)lengths[item];
        }
        for (unit = 0; unit < hidden_size; ++unit) {
$carried_in
        }
        for (step = 0; step < length; ++step) {
            const size_t time = reverse ? length - 1 - step : step;
            const $element* values = sequence + time * x_time_stride;
$recurrence
            if (y != NULL) {
                $element* written = y + item * y_batch_stride + time * y_time_stride;
                for (unit = 0; unit < hidden_size; ++unit) {
                    written[unit] = hidden[unit];
                }
            }
        }
        if (y != NULL) {
            /* The steps past this sequence's own end are padding, which ONNX reports as
               zeros rather than as state. */
            for (step = length; step < seq_length; ++step) {
                $element* written = y + item * y_batch_stride + step * y_time_stride;
                for (unit = 0; unit < hidden_size; ++unit) {
                    written[unit] = $zero;
                }
            }
        }
        for (unit = 0; unit < hidden_size; ++unit) {
            /* A sequence of no steps carries no state to report: ONNX says nothing about
               one, and this is what onnxruntime returns for it. */
$carried_out
        }
    }
    return 0;
}""")

# The hidden state is the whole of what a GRU or an RNN carries between steps; the LSTM adds
# its cell state alongside, read and reported the same way.
_HIDDEN_IN = """\
            hidden[unit] = initial_h == NULL ? $zero : initial_h[state_base + unit];"""

_HIDDEN_OUT = """\
            if (y_h != NULL) {
                y_h[state_base + unit] = length == 0 ? $zero : hidden[unit];
            }"""

_LSTM_CARRIED_IN = (
    _HIDDEN_IN
    + """
            cell[unit] = initial_c == NULL ? $zero : initial_c[state_base + unit];"""
)

_LSTM_CARRIED_OUT = (
    _HIDDEN_OUT
    + """
            if (y_c != NULL) {
                y_c[state_base + unit] = length == 0 ? $zero : cell[unit];
            }"""
)


def _frame(
    parameters: str, carried_in: str, carried_out: str, recurrence: str
) -> Template:
    """The frame with one op's pieces spliced in, still holding the per-node placeholders.

    Two passes: the pieces carry `$element` and `$zero` of their own, which a single
    substitution would leave untouched inside the text it inserts.
    """
    return Template(
        _FRAME.safe_substitute(
            parameters=parameters,
            carried_in=carried_in,
            carried_out=carried_out,
            recurrence=recurrence,
        )
    )


# --------------------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------------------

_LSTM_PARAMETERS = """\
    $element* y,
    $element* y_h,
    $element* y_c,
    const $element* x,
    const $element* w,
    const $element* r,
    const $element* bias,
    const int32_t* lengths,
    const $element* initial_h,
    const $element* initial_c,
    const $element* peepholes,
    $element* hidden,
    $element* cell,
    $element* gates,
    size_t seq_length,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    size_t x_time_stride,
    size_t x_batch_stride,
    size_t y_time_stride,
    size_t y_batch_stride,
    size_t state_batch_stride,
    int reverse,
    int coupled,
    int clipped,
    $element clip,
    $element (*act_f)($element, $element, $element),
    $element alpha_f,
    $element beta_f,
    $element (*act_g)($element, $element, $element),
    $element alpha_g,
    $element beta_g,
    $element (*act_h)($element, $element, $element),
    $element alpha_h,
    $element beta_h"""

# The four gates in the order ONNX concatenates them along `W`, `R` and `B`: input, output,
# forget, cell. Each unit's update reads only its own gates and its own cell, so the new
# hidden state can be written in place as it goes.
_LSTM_RECURRENCE = """\
            for (unit = 0; unit < gate_count; ++unit) {
                $element total =
                    bias == NULL ? $zero : bias[unit] + bias[gate_count + unit];
                for (term = 0; term < input_size; ++term) {
                    total += w[unit * input_size + term] * values[term];
                }
                for (term = 0; term < hidden_size; ++term) {
                    total += r[unit * hidden_size + term] * hidden[term];
                }
                gates[unit] = total;
            }
            for (unit = 0; unit < hidden_size; ++unit) {
                const $element state = cell[unit];
                $element input_gate = gates[unit];
                $element forget_gate = gates[2 * hidden_size + unit];
                $element candidate = gates[3 * hidden_size + unit];
                $element output_gate = gates[hidden_size + unit];
                $element updated;
                if (peepholes != NULL) {
                    input_gate += peepholes[unit] * state;
                    forget_gate += peepholes[2 * hidden_size + unit] * state;
                }
                if (clipped) {
                    input_gate = $clip(input_gate, clip);
                    forget_gate = $clip(forget_gate, clip);
                    candidate = $clip(candidate, clip);
                }
                input_gate = act_f(input_gate, alpha_f, beta_f);
                forget_gate = coupled
                    ? $one - input_gate
                    : act_f(forget_gate, alpha_f, beta_f);
                updated = forget_gate * state
                    + input_gate * act_g(candidate, alpha_g, beta_g);
                cell[unit] = updated;
                if (peepholes != NULL) {
                    output_gate += peepholes[hidden_size + unit] * updated;
                }
                if (clipped) {
                    output_gate = $clip(output_gate, clip);
                }
                hidden[unit] = act_f(output_gate, alpha_f, beta_f)
                    * act_h(updated, alpha_h, beta_h);
            }"""


# --------------------------------------------------------------------------------------
# GRU
# --------------------------------------------------------------------------------------

_GRU_PARAMETERS = """\
    $element* y,
    $element* y_h,
    const $element* x,
    const $element* w,
    const $element* r,
    const $element* bias,
    const int32_t* lengths,
    const $element* initial_h,
    $element* hidden,
    $element* gates,
    size_t seq_length,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    size_t x_time_stride,
    size_t x_batch_stride,
    size_t y_time_stride,
    size_t y_batch_stride,
    size_t state_batch_stride,
    int reverse,
    int linear_before_reset,
    int clipped,
    $element clip,
    $element (*act_f)($element, $element, $element),
    $element alpha_f,
    $element beta_f,
    $element (*act_g)($element, $element, $element),
    $element alpha_g,
    $element beta_g"""

# The three gates in ONNX's order: update, reset, hidden candidate. The candidate reads the
# reset gate, so it cannot share the first loop; and it reads the whole previous state
# through the recurrence weights, so the state is replaced only once every gate has read it.
# The candidate is parked in the third gate slot, which nothing else uses.
_GRU_RECURRENCE = """\
            for (unit = 0; unit < 2 * hidden_size; ++unit) {
                $element total =
                    bias == NULL ? $zero : bias[unit] + bias[gate_count + unit];
                for (term = 0; term < input_size; ++term) {
                    total += w[unit * input_size + term] * values[term];
                }
                for (term = 0; term < hidden_size; ++term) {
                    total += r[unit * hidden_size + term] * hidden[term];
                }
                if (clipped) {
                    total = $clip(total, clip);
                }
                gates[unit] = act_f(total, alpha_f, beta_f);
            }
            for (unit = 0; unit < hidden_size; ++unit) {
                const size_t gate = 2 * hidden_size + unit;
                $element total = bias == NULL ? $zero : bias[gate];
                for (term = 0; term < input_size; ++term) {
                    total += w[gate * input_size + term] * values[term];
                }
                if (linear_before_reset) {
                    /* This unit's own reset gate scales the whole recurrence, its bias
                       included. */
                    $element carried = bias == NULL ? $zero : bias[gate_count + gate];
                    for (term = 0; term < hidden_size; ++term) {
                        carried += r[gate * hidden_size + term] * hidden[term];
                    }
                    total += gates[hidden_size + unit] * carried;
                } else {
                    /* The reset gate scales the state, before the recurrence weights read
                       it, so each term is scaled by the gate of the unit it belongs to. */
                    if (bias != NULL) {
                        total += bias[gate_count + gate];
                    }
                    for (term = 0; term < hidden_size; ++term) {
                        total += r[gate * hidden_size + term]
                            * (gates[hidden_size + term] * hidden[term]);
                    }
                }
                if (clipped) {
                    total = $clip(total, clip);
                }
                gates[gate] = act_g(total, alpha_g, beta_g);
            }
            for (unit = 0; unit < hidden_size; ++unit) {
                const $element update = gates[unit];
                hidden[unit] = ($one - update) * gates[2 * hidden_size + unit]
                    + update * hidden[unit];
            }"""


# --------------------------------------------------------------------------------------
# RNN
# --------------------------------------------------------------------------------------

_RNN_PARAMETERS = """\
    $element* y,
    $element* y_h,
    const $element* x,
    const $element* w,
    const $element* r,
    const $element* bias,
    const int32_t* lengths,
    const $element* initial_h,
    $element* hidden,
    $element* gates,
    size_t seq_length,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size,
    size_t x_time_stride,
    size_t x_batch_stride,
    size_t y_time_stride,
    size_t y_batch_stride,
    size_t state_batch_stride,
    int reverse,
    int clipped,
    $element clip,
    $element (*act_f)($element, $element, $element),
    $element alpha_f,
    $element beta_f"""

# One gate, whose activation is the whole of the new state. Every unit reads the whole
# previous state, so the state is replaced only once every gate has been accumulated.
_RNN_RECURRENCE = """\
            for (unit = 0; unit < hidden_size; ++unit) {
                $element total =
                    bias == NULL ? $zero : bias[unit] + bias[gate_count + unit];
                for (term = 0; term < input_size; ++term) {
                    total += w[unit * input_size + term] * values[term];
                }
                for (term = 0; term < hidden_size; ++term) {
                    total += r[unit * hidden_size + term] * hidden[term];
                }
                if (clipped) {
                    total = $clip(total, clip);
                }
                gates[unit] = total;
            }
            for (unit = 0; unit < hidden_size; ++unit) {
                hidden[unit] = act_f(gates[unit], alpha_f, beta_f);
            }"""


@dataclass(frozen=True)
class _Layer:
    """What distinguishes one of ONNX's three recurrent ops from the other two.

    `gates` is how many gate rows `W`, `R` and `B` carry per direction. `defaults` is the
    `(f, g, h)` — as many as the op runs — that a node naming no activations gets, which also
    fixes how many a node that does name them has to name. `mode` is the single integer
    attribute the op turns into a branch inside the kernel. `cell` marks the LSTM: the only
    one carrying a second state between steps, and so the only one with `initial_c`, `Y_c`
    and peephole weights.
    """

    op_type: str
    gates: int
    defaults: tuple[str, ...]
    template: Template
    scratch: tuple[str, ...]
    mode: str | None = None
    cell: bool = False

    @property
    def symbol(self) -> str:
        return self.op_type.lower()

    @property
    def results(self) -> int:
        return 3 if self.cell else 2


_LAYERS = (
    _Layer(
        op_type="LSTM",
        gates=4,
        defaults=("Sigmoid", "Tanh", "Tanh"),
        template=_frame(
            _LSTM_PARAMETERS, _LSTM_CARRIED_IN, _LSTM_CARRIED_OUT, _LSTM_RECURRENCE
        ),
        scratch=("hidden", "cell", "gates"),
        mode="input_forget",
        cell=True,
    ),
    _Layer(
        op_type="GRU",
        gates=3,
        defaults=("Sigmoid", "Tanh"),
        template=_frame(_GRU_PARAMETERS, _HIDDEN_IN, _HIDDEN_OUT, _GRU_RECURRENCE),
        scratch=("hidden", "gates"),
        mode="linear_before_reset",
    ),
    _Layer(
        op_type="RNN",
        gates=1,
        defaults=("Tanh",),
        template=_frame(_RNN_PARAMETERS, _HIDDEN_IN, _HIDDEN_OUT, _RNN_RECURRENCE),
        scratch=("hidden", "gates"),
    ),
)


@dataclass(frozen=True)
class _Geometry:
    """A node's shape, and where each operand's elements sit under its layout.

    The strides are in elements. Layout 1 packs the batch outermost rather than time, which
    changes nothing the kernel computes — only how far apart two steps of one sequence are.
    """

    seq_length: int
    batch_size: int
    input_size: int
    hidden_size: int
    direction: str
    layout: int

    @property
    def directions(self) -> int:
        return _DIRECTIONS[self.direction]

    @property
    def x_shape(self) -> tuple[int, ...]:
        if self.layout:
            return (self.batch_size, self.seq_length, self.input_size)
        return (self.seq_length, self.batch_size, self.input_size)

    @property
    def y_shape(self) -> tuple[int, ...]:
        if self.layout:
            return (self.batch_size, self.seq_length, self.directions, self.hidden_size)
        return (self.seq_length, self.directions, self.batch_size, self.hidden_size)

    @property
    def state_shape(self) -> tuple[int, ...]:
        if self.layout:
            return (self.batch_size, self.directions, self.hidden_size)
        return (self.directions, self.batch_size, self.hidden_size)

    @property
    def strides(self) -> tuple[int, ...]:
        """`(x_time, x_batch, y_time, y_batch, state_batch)`, in elements."""
        if self.layout:
            return (
                self.input_size,
                self.seq_length * self.input_size,
                self.directions * self.hidden_size,
                self.seq_length * self.directions * self.hidden_size,
                self.directions * self.hidden_size,
            )
        return (
            self.batch_size * self.input_size,
            self.input_size,
            self.directions * self.batch_size * self.hidden_size,
            self.hidden_size,
            self.hidden_size,
        )

    def state_offset(self, direction: int) -> int:
        """Where this direction's slice of `Y`, of a state output and of an initial one begins.

        One offset serves all of them: under either layout the direction axis sits directly
        outside the hidden units in every one.
        """
        if self.layout:
            return direction * self.hidden_size
        return direction * self.batch_size * self.hidden_size

    def runs_backwards(self, direction: int) -> bool:
        return self.direction == "reverse" or (
            self.direction == "bidirectional" and direction == 1
        )


def _recurrent(context: NodeContext, layer: _Layer) -> NodeEmission:
    geometry = _geometry(context, layer)
    activations = _activations(context, layer, geometry.directions)
    _verify_operands(context, layer, geometry)
    outputs = tuple(
        context.outputs[index] if index < len(context.outputs) else None
        for index in range(layer.results)
    )
    if all(result is None or result.elem_count == 0 for result in outputs):
        return NodeEmission(functions=(), statements=())

    elem_type = context.require_input(0).elem_type
    element = c_type(elem_type)
    name = f"{context.prefix}_{layer.symbol}_{element}"
    bound = _clip_function(context, elem_type)
    definition = layer.template.substitute(
        name=name,
        element=element,
        gates=layer.gates,
        clip=bound.name,
        one=scalar_literal(1, elem_type),
        zero=scalar_literal(0, elem_type),
    )
    scratch = _scratch(context, layer, geometry, elem_type)
    helpers = {
        function.name: function
        for selection in activations
        for function in selection.functions(context, elem_type)
    }
    return NodeEmission(
        functions=(bound, *helpers.values(), CFunction(name, definition)),
        statements=tuple(
            checked_call(
                context,
                name,
                _arguments(
                    context, layer, geometry, outputs, scratch, activations, direction
                ),
            )
            for direction in range(geometry.directions)
        ),
        scratch=scratch,
    )


# --------------------------------------------------------------------------------------
# The activations a node selects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Selection:
    """The activations one direction runs, each with the alpha and beta it is given."""

    names: tuple[str, ...]
    alphas: tuple[float, ...]
    betas: tuple[float, ...]

    def functions(self, context: NodeContext, elem_type: int) -> list[CFunction]:
        return [_activation_function(context, name, elem_type) for name in self.names]

    def arguments(self, context: NodeContext, elem_type: int) -> list[str]:
        """The function pointer and the two parameters each activation is called through."""
        arguments = []
        for name, alpha, beta in zip(self.names, self.alphas, self.betas):
            arguments.append(_activation_function(context, name, elem_type).name)
            arguments.append(scalar_literal(alpha, elem_type))
            arguments.append(scalar_literal(beta, elem_type))
        return arguments


def _activations(
    context: NodeContext, layer: _Layer, directions: int
) -> tuple[_Selection, ...]:
    """The activations each direction runs, defaulting to the op's own.

    ONNX takes the names as one flat list per direction, and their parameters as two more
    lists consumed in the same order.
    """
    count = len(layer.defaults)
    names = [name.decode() for name in context.attribute("activations", [])]
    if not names:
        names = list(layer.defaults) * directions
    if len(names) != count * directions:
        expected = count * directions
        raise CompileError(
            f"Node `{context.label}`: a `{_direction(context, layer)}` `{layer.op_type}` "
            f"runs {expected} activation{'' if expected == 1 else 's'} — {count} per "
            f"direction — but this node names {len(names)}: {', '.join(names) or 'none'}."
        )
    alphas = _parameters(context, layer, names, "alpha")
    betas = _parameters(context, layer, names, "beta")
    return tuple(
        _Selection(
            names=tuple(names[count * index : count * (index + 1)]),
            alphas=alphas[count * index : count * (index + 1)],
            betas=betas[count * index : count * (index + 1)],
        )
        for index in range(directions)
    )


def _parameters(
    context: NodeContext, layer: _Layer, names: Sequence[str], role: str
) -> tuple[float, ...]:
    """The alpha or beta each named activation runs with, else ONNX's own default.

    ONNX's list carries a value for the activations that take one and for no others: they
    consume it in the order the node names them, over both directions of a bidirectional
    node together, so a parameterized activation reads the next value rather than the one at
    its own position.
    """
    given = [float(value) for value in context.attribute(f"activation_{role}", [])]
    resolved: list[float] = []
    consumed = 0
    for name in names:
        activation = _activation(context, layer, name)
        if not getattr(activation, role):
            resolved.append(0.0)
            continue
        if consumed < len(given):
            resolved.append(given[consumed])
        elif activation.schema is None:
            raise CompileError(
                f"Node `{context.label}`: `{layer.op_type}` runs `{name}`, which ONNX "
                f"defines no operator for, so there is no default `activation_{role}` to "
                f"fall back on; the attribute carries one value per activation that takes "
                f"one, and `{name}` reads number {consumed + 1}."
            )
        else:
            resolved.append(
                float(
                    onnx.defs.get_schema(activation.schema)
                    .attributes[role]
                    .default_value.f
                )
            )
        consumed += 1
    return tuple(resolved)


def _activation(context: NodeContext, layer: _Layer, name: str) -> _Activation:
    activation = _ACTIVATIONS.get(name)
    if activation is None:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}` names the activation `{name}`, "
            f"which is not one of the functions ONNX defines it over: "
            f"{', '.join(_ACTIVATIONS)}."
        )
    return activation


def _activation_function(context: NodeContext, name: str, elem_type: int) -> CFunction:
    symbol = f"{context.prefix}_rnnact_{name.lower()}_{c_type(elem_type)}"
    return CFunction(
        symbol,
        _ACTIVATION_TEMPLATE.substitute(
            name=symbol,
            element=c_type(elem_type),
            expression=expand(_ACTIVATIONS[name].expression, elem_type),
        ),
    )


def _clip_function(context: NodeContext, elem_type: int) -> CFunction:
    name = f"{context.prefix}_rnnclip_{c_type(elem_type)}"
    return CFunction(
        name,
        _CLIP_TEMPLATE.substitute(name=name, element=c_type(elem_type)),
    )


# --------------------------------------------------------------------------------------
# Reading the geometry off the node, and placing the call
# --------------------------------------------------------------------------------------


def _geometry(context: NodeContext, layer: _Layer) -> _Geometry:
    source = context.require_input(0)
    recurrence = context.require_input(2)
    if len(source.shape) != 3:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}` reads a batch of sequences — a "
            f"tensor of rank 3 — but `{source.name}` has shape {list(source.shape)}."
        )
    if len(recurrence.shape) != 3:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}` reads its recurrence weights as "
            f"rank 3, but `{recurrence.name}` has shape {list(recurrence.shape)}."
        )
    layout = _layout(context, layer)
    steps, items = source.shape[0], source.shape[1]
    return _Geometry(
        seq_length=items if layout else steps,
        batch_size=steps if layout else items,
        input_size=source.shape[2],
        hidden_size=_hidden_size(context, layer, recurrence),
        direction=_direction(context, layer),
        layout=layout,
    )


def _layout(context: NodeContext, layer: _Layer) -> int:
    layout = context.int_attribute("layout")
    if layout not in (0, 1):
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}`'s `layout` is {layout}, but ONNX "
            "defines only 0 (time first) and 1 (batch first)."
        )
    return layout


def _direction(context: NodeContext, layer: _Layer) -> str:
    name = context.attribute("direction", b"forward").decode()
    if name not in _DIRECTIONS:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}`'s `direction` is `{name}`, which "
            f"is not one of the directions ONNX defines: {', '.join(_DIRECTIONS)}."
        )
    return name


def _hidden_size(context: NodeContext, layer: _Layer, recurrence: TensorRef) -> int:
    """The width of the state, which the recurrence weights carry.

    ONNX states it as an attribute too; a node whose attribute disagrees with the weights it
    is handed describes two different layers, and there is no telling which one it meant.
    """
    hidden_size = recurrence.shape[2]
    declared = context.attribute("hidden_size", None)
    if declared is not None and int(declared) != hidden_size:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}` states a `hidden_size` of "
            f"{int(declared)}, but its recurrence weights `{recurrence.name}` are shaped "
            f"for {hidden_size}."
        )
    return hidden_size


def _operands(
    layer: _Layer, geometry: _Geometry
) -> dict[int, tuple[str, tuple[int, ...]]]:
    """Each optional operand past `X`, by input index: what it is, and the shape it must be."""
    hidden, directions = geometry.hidden_size, geometry.directions
    row = layer.gates * hidden
    operands = {
        1: ("its input weights", (directions, row, geometry.input_size)),
        2: ("its recurrence weights", (directions, row, hidden)),
        3: ("its biases", (directions, 2 * row)),
        4: ("its sequence lengths", (geometry.batch_size,)),
        5: ("an initial hidden state", geometry.state_shape),
    }
    if layer.cell:
        operands[6] = ("an initial cell state", geometry.state_shape)
        operands[7] = ("its peephole weights", (directions, _PEEPHOLES * hidden))
    return operands


def _verify_operands(context: NodeContext, layer: _Layer, geometry: _Geometry) -> None:
    """Refuse to emit a kernel whose addressing disagrees with the buffers it is handed.

    ONNX's own shape inference derives most of this, but a graph may declare the shapes it
    would infer rather than have them inferred, so the extents the kernel walks are checked
    here against the buffers themselves.
    """
    for index, (role, shape) in _operands(layer, geometry).items():
        operand = context.optional_input(index)
        if operand is not None and operand.shape != shape:
            raise CompileError(
                f"Node `{context.label}`: `{layer.op_type}` reads `{operand.name}` as "
                f"{role} of shape {list(shape)}, but it has shape {list(operand.shape)}."
            )
    results = (geometry.y_shape, *(geometry.state_shape,) * (layer.results - 1))
    for index, shape in enumerate(results):
        result = context.outputs[index] if index < len(context.outputs) else None
        if result is not None:
            verify_shape(context, result, shape)


def _scratch(
    context: NodeContext, layer: _Layer, geometry: _Geometry, elem_type: int
) -> tuple[ScratchBuffer, ...]:
    """The state of one batch item, and the gate values of one step, reserved statically.

    A step reads the whole previous state while writing the new one, so neither state can be
    computed in place in a caller's buffer; the artifact allocates nothing, so the space is
    reserved at compile time and counted in the reported footprint like every other buffer.
    """
    element = c_type(elem_type)
    counts = {
        "hidden": geometry.hidden_size,
        "cell": geometry.hidden_size,
        "gates": layer.gates * geometry.hidden_size,
    }
    return tuple(
        ScratchBuffer(
            f"{context.prefix}_{layer.symbol}_{role}_{element}", elem_type, counts[role]
        )
        for role in layer.scratch
    )


def _arguments(
    context: NodeContext,
    layer: _Layer,
    geometry: _Geometry,
    outputs: tuple[TensorRef | None, ...],
    scratch: tuple[ScratchBuffer, ...],
    activations: tuple[_Selection, ...],
    direction: int,
) -> list[str]:
    """The call site for one direction: every operand offset onto that direction's slice."""
    elem_type = context.require_input(0).elem_type
    hidden, state = geometry.hidden_size, geometry.state_offset(direction)
    row = layer.gates * hidden
    offsets = {
        1: direction * row * geometry.input_size,
        2: direction * row * hidden,
        3: direction * 2 * row,
        4: 0,
        5: state,
        6: state,
        7: direction * _PEEPHOLES * hidden,
    }
    clip = context.attribute("clip", None)
    arguments = [_operand(result, state) for result in outputs]
    arguments.append(_operand(context.require_input(0), 0))
    arguments += [
        _operand(context.optional_input(index), offsets[index])
        for index in sorted(_operands(layer, geometry))
    ]
    arguments += [buffer.symbol for buffer in scratch]
    arguments += [
        f"{geometry.seq_length}u",
        f"{geometry.batch_size}u",
        f"{geometry.input_size}u",
        f"{hidden}u",
        *(f"{stride}u" for stride in geometry.strides),
        str(int(geometry.runs_backwards(direction))),
    ]
    if layer.mode is not None:
        arguments.append(str(int(context.int_attribute(layer.mode) != 0)))
    arguments += [
        str(int(clip is not None)),
        scalar_literal(_clip_threshold(context, layer, clip), elem_type),
        *activations[direction].arguments(context, elem_type),
    ]
    return arguments


def _operand(ref: TensorRef | None, offset: int) -> str:
    if ref is None:
        return "NULL"
    return ref.expr if offset == 0 else f"{ref.expr} + {offset}"


def _clip_threshold(context: NodeContext, layer: _Layer, clip: float | None) -> float:
    if clip is None:
        return 0.0
    if not clip >= 0.0:
        raise CompileError(
            f"Node `{context.label}`: `{layer.op_type}`'s `clip` is {clip}, but ONNX "
            "defines it as the threshold a cell is bounded to, which is not negative."
        )
    return float(clip)


for _layer in _LAYERS:
    register_kernel("", _layer.op_type, _VERSIONS, partial(_recurrent, layer=_layer))
