"""DFT and STFT: the transform written out as the sum ONNX defines it to be.

A discrete Fourier transform is one sum per output bin over the samples of one axis, and a
short-time transform is that same sum over a window slid along the signal. Both are emitted
as exactly that — no factorization, no twiddle tables — so the code is the definition and its
size does not depend on the transform's length. The transform is evaluated in `double`
whatever the tensor holds, which is what numpy's own FFT does before rounding back to the
input's type, so a `float` model is not compared against an oracle computed to a different
precision.

What varies between calls is addressing, not code: the transformed axis is read as an outer
block count, an inner stride and an extent, so one kernel per element type serves every axis,
rank and length. The axis itself may be named at run time — ONNX passes it as an operand from
opset 20 — but only where it cannot change the shape of the result; `onesided` and
`dft_length` both resize the axis they land on, and which axis that is has to be known before
a buffer can be sized. The window functions and MelWeightMatrix have no kernels at all: their
operands are their result's own shape, so a model that fixes them is folded away through the
reference evaluator before dispatch, and one that does not is refused by the frontend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.emit import INVALID_ARGUMENT_STATUS
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import call_kernel, kernel_name, verify_shape

# DFT-20 moved the axis from an attribute to an operand; the two revisions are otherwise the
# same transform. STFT has had a single revision since opset 17.
_AXIS_AS_OPERAND = 20
_DFT_VERSIONS = (17, _AXIS_AS_OPERAND)
_STFT_VERSIONS = (17,)

# Where that operand sits, and what the node transforms when it is left out: the last signal
# axis, the one before the real/imaginary pair. Revision 17's attribute defaults to the first
# signal axis instead, which its schema states and `int_attribute` reads.
_AXIS_INPUT = 2
_DEFAULT_AXIS = -2

# The last axis of a signal tensor holds a real value alone or a real and an imaginary part.
_SIGNAL_COMPONENTS = (1, 2)

# 2*pi to more digits than a double carries, so the constant is the nearest representable
# one however the C compiler parses it.
_TURN = "6.283185307179586476925286766559"

_DFT_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    size_t outer,
    size_t inner,
    size_t samples,
    size_t bins,
    size_t length,
    size_t components,
    size_t written,
    int inverse,
    int mirrored)
{
    const double turn = $turn;
    const size_t terms = mirrored ? length : (samples < length ? samples : length);
    size_t lead, trail, bin, term;
    for (lead = 0; lead < outer; ++lead) {
        for (trail = 0; trail < inner; ++trail) {
            const size_t from = (lead * samples * inner + trail) * components;
            const size_t to = (lead * bins * inner + trail) * written;
            for (bin = 0; bin < bins; ++bin) {
                double real = 0.0;
                double imaginary = 0.0;
                for (term = 0; term < terms; ++term) {
                    const int conjugated = mirrored && term > length / 2;
                    const size_t source = conjugated ? length - term : term;
                    if (source < samples) {
                        const size_t at = from + source * inner * components;
                        const double angle =
                            turn * (double)(term * bin % length) / (double)length;
                        const double cosine = cos(angle);
                        const double sine = inverse ? sin(angle) : -sin(angle);
                        const double re = (double)in[at];
                        double im = components == 2 ? (double)in[at + 1] : 0.0;
                        if (conjugated) {
                            im = -im;
                        }
                        real += re * cosine - im * sine;
                        imaginary += re * sine + im * cosine;
                    }
                }
                if (inverse) {
                    real /= (double)length;
                    imaginary /= (double)length;
                }
                out[to + bin * inner * written] = ($element)real;
                if (written == 2) {
                    out[to + bin * inner * written + 1] = ($element)imaginary;
                }
            }
        }
    }
}""")

_STFT_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* window,
    size_t batch,
    size_t signal,
    size_t components,
    size_t frames,
    size_t step,
    size_t length,
    size_t bins)
{
    const double turn = $turn;
    size_t index, frame, bin, term;
    for (index = 0; index < batch; ++index) {
        for (frame = 0; frame < frames; ++frame) {
            for (bin = 0; bin < bins; ++bin) {
                double real = 0.0;
                double imaginary = 0.0;
                for (term = 0; term < length; ++term) {
                    const size_t sample = frame * step + term;
                    if (sample < signal) {
                        const size_t at = (index * signal + sample) * components;
                        const double weight =
                            window != NULL ? (double)window[term] : 1.0;
                        const double angle =
                            turn * (double)(term * bin % length) / (double)length;
                        const double cosine = cos(angle);
                        const double sine = -sin(angle);
                        const double re = weight * (double)in[at];
                        const double im =
                            components == 2 ? weight * (double)in[at + 1] : 0.0;
                        real += re * cosine - im * sine;
                        imaginary += re * sine + im * cosine;
                    }
                }
                out[((index * frames + frame) * bins + bin) * 2] = ($element)real;
                out[((index * frames + frame) * bins + bin) * 2 + 1] =
                    ($element)imaginary;
            }
        }
    }
}""")

_NORMALIZE_TEMPLATE = Template("""\
static int64_t $name(int64_t axis, int64_t rank)
{
    return (axis < 0) ? (axis + rank) : axis;
}""")


@dataclass(frozen=True)
class _Transform:
    """One DFT call site: the shape it writes, and the addressing it walks to write it.

    `leading` and `trailing` are the operand's extents before and after the transformed axis,
    the trailing ones less the real/imaginary axis: a block of transforms per coordinate of
    the first, one stride apart for each coordinate of the second. `mirrored` marks the
    inverse one-sided transform, whose operand holds only the non-redundant half of a
    spectrum the conjugate symmetry fills back in — which is also the one transform writing
    a real result rather than a complex one.
    """

    leading: tuple[int, ...]
    trailing: tuple[int, ...]
    samples: int
    bins: int
    length: int
    components: int
    mirrored: bool
    inverse: bool

    @property
    def written(self) -> int:
        return 1 if self.mirrored else 2

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self.leading, self.bins, *self.trailing, self.written)

    def arguments(self, result: TensorRef, source: TensorRef) -> list[str]:
        return [
            result.expr,
            source.expr,
            f"{math.prod(self.leading)}u",
            f"{math.prod(self.trailing)}u",
            f"{self.samples}u",
            f"{self.bins}u",
            f"{self.length}u",
            f"{self.components}u",
            f"{self.written}u",
            str(int(self.inverse)),
            str(int(self.mirrored)),
        ]


def _dft(context: NodeContext) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    rank = _signal_rank(context, source)
    inverse = context.int_attribute("inverse") != 0
    onesided = context.int_attribute("onesided") != 0
    length = _dft_length(context)

    fixed = _fixed_axis(context, rank)
    axes = (
        (fixed,)
        if fixed is not None
        else _runtime_axes(context, rank, onesided, length)
    )
    plans = {
        axis: _plan(
            source.shape, axis, inverse=inverse, onesided=onesided, length=length
        )
        for axis in axes
    }
    for plan in plans.values():
        verify_shape(context, result, plan.shape)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())
    for plan in plans.values():
        _verify_length(context, plan.length)

    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    kernel = CFunction(
        name, _DFT_TEMPLATE.substitute(name=name, element=element, turn=_TURN)
    )
    calls = {
        axis: call_kernel(name, plan.arguments(result, source))
        for axis, plan in plans.items()
    }
    if fixed is not None:
        return NodeEmission(functions=(kernel,), statements=(calls[fixed],))
    normalize = _normalize_helper(context.prefix)
    operand = context.require_input(_AXIS_INPUT)
    return NodeEmission(
        functions=(normalize, kernel),
        statements=(_dispatch(context, operand, rank, calls, normalize),),
    )


def _stft(context: NodeContext) -> NodeEmission:
    signal = context.require_input(0)
    result = context.require_output(0)
    if len(signal.shape) != 3:
        raise CompileError(
            f"Node `{context.label}`: `STFT` was given `{signal.name}` of shape "
            f"{list(signal.shape)}; ONNX defines its signal as "
            "[batch_size][signal_length][1 or 2]."
        )
    batch, samples, components = signal.shape
    _verify_components(context, signal, components)
    window = context.optional_input(2)
    step = _frame_step(context)
    length = _frame_length(context, window)
    _verify_length(context, length)
    onesided = context.int_attribute("onesided") != 0
    bins = length // 2 + 1 if onesided else length
    frames = 1 + (samples - length) // step
    if frames < 0:
        raise CompileError(
            f"Node `{context.label}`: `STFT` frames of {length} sample(s) do not fit a "
            f"signal of {samples}; not one whole frame can be taken."
        )
    verify_shape(context, result, (batch, frames, bins, 2))
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    name = kernel_name(context, element)
    definition = _STFT_TEMPLATE.substitute(name=name, element=element, turn=_TURN)
    arguments = [
        result.expr,
        signal.expr,
        "NULL" if window is None else window.expr,
        f"{batch}u",
        f"{samples}u",
        f"{components}u",
        f"{frames}u",
        f"{step}u",
        f"{length}u",
        f"{bins}u",
    ]
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call_kernel(name, arguments),),
    )


def _plan(
    shape: Sequence[int],
    axis: int,
    *,
    inverse: bool,
    onesided: bool,
    length: int | None,
) -> _Transform:
    """How the transform along `axis` addresses the tensor, and the result it writes.

    An absent `dft_length` is the extent of the axis itself, except for the inverse one-sided
    transform, whose operand holds only the non-redundant half of an even-length spectrum.
    """
    samples = shape[axis]
    mirrored = inverse and onesided
    if length is None:
        length = 2 * (samples - 1) if mirrored else samples
    return _Transform(
        leading=tuple(shape[:axis]),
        trailing=tuple(shape[axis + 1 : -1]),
        samples=samples,
        bins=length // 2 + 1 if onesided and not inverse else length,
        length=length,
        components=shape[-1],
        mirrored=mirrored,
        inverse=inverse,
    )


def _signal_rank(context: NodeContext, source: TensorRef) -> int:
    """The operand's rank, once it is one a transform is defined over."""
    rank = len(source.shape)
    if rank < 2:
        raise CompileError(
            f"Node `{context.label}`: `DFT` was given `{source.name}` of shape "
            f"{list(source.shape)}; ONNX defines its input as at least one signal axis "
            "followed by the axis holding the real and imaginary parts."
        )
    _verify_components(context, source, source.shape[-1])
    return rank


def _verify_components(
    context: NodeContext, source: TensorRef, components: int
) -> None:
    if components not in _SIGNAL_COMPONENTS:
        raise CompileError(
            f"Node `{context.label}`: the last axis of `{source.name}` measures "
            f"{components}; ONNX defines it as 1 for a real signal and 2 for a complex one."
        )


def _verify_length(context: NodeContext, length: int) -> None:
    if length < 1:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` was given a transform "
            f"length of {length}; ONNX defines nothing for a transform over no samples "
            "at all."
        )


def _fixed_axis(context: NodeContext, rank: int) -> int | None:
    """The axis this node transforms, or None where it names one only at run time."""
    if context.since_version < _AXIS_AS_OPERAND:
        return _checked_axis(context, context.int_attribute("axis"), rank)
    if context.optional_input(_AXIS_INPUT) is None:
        return _checked_axis(context, _DEFAULT_AXIS, rank)
    fixed = context.constant_input(_AXIS_INPUT)
    if fixed is None:
        return None
    if fixed.size != 1:
        raise CompileError(
            f"Node `{context.label}`: the axis of `DFT` comes from "
            f"`{context.require_input(_AXIS_INPUT).name}`, which holds {fixed.size} "
            "values; ONNX defines it as a single one."
        )
    return _checked_axis(context, int(fixed.reshape(-1)[0]), rank)


def _checked_axis(context: NodeContext, axis: int, rank: int) -> int:
    """An ONNX DFT axis, which may count from the end, as an index into the operand.

    The last axis holds the real and imaginary parts rather than a signal, so it is the one
    axis of the operand no transform may be taken along.
    """
    if not -rank <= axis < rank - 1 or axis == -1:
        raise CompileError(
            f"Node `{context.label}`: `DFT` transforms axis {axis} of a rank-{rank} "
            "operand; ONNX defines the axis over [-rank, -2] and [0, rank - 2], the last "
            "axis being the real and imaginary parts rather than a signal."
        )
    return axis + rank if axis < 0 else axis


def _runtime_axes(
    context: NodeContext, rank: int, onesided: bool, length: int | None
) -> tuple[int, ...]:
    """Every axis a run-time operand could name, where naming one cannot resize the result.

    A one-sided transform returns half its length and a stated `dft_length` replaces it, so
    either makes the extent of the transformed axis differ from the operand's — and which
    axis that is then has to be known to size a buffer at all. With neither, every axis
    leaves the operand's own extents, so the result is one shape whichever the operand names
    and the choice is a run-time switch over compile-time call sites.
    """
    if onesided or length is not None:
        raise CompileError(
            f"Node `{context.label}`: `DFT` takes its axis from "
            f"`{context.require_input(_AXIS_INPUT).name}`, which no initializer or constant "
            f"folding fixes, while {'`onesided`' if onesided else '`dft_length`'} resizes "
            "the axis it transforms; the shape of the result then depends on input data, "
            "which the C compiler requires to be known at compile time."
        )
    return tuple(range(rank - 1))


def _dft_length(context: NodeContext) -> int | None:
    """The transform's length where the node states one, as a compile-time value."""
    operand = context.optional_input(1)
    if operand is None:
        return None
    fixed = context.constant_input(1)
    if fixed is None or fixed.size != 1:
        raise CompileError(
            f"Node `{context.label}`: `DFT` takes its length from `{operand.name}`, which "
            f"holds {'no single value' if fixed is not None else 'no value'} known at "
            "compile time; the shape of the result then depends on input data, which the C "
            "compiler cannot compile."
        )
    return int(fixed.reshape(-1)[0])


def _frame_step(context: NodeContext) -> int:
    operand = context.require_input(1)
    fixed = context.constant_input(1)
    if fixed is None or fixed.size != 1:
        raise CompileError(
            f"Node `{context.label}`: `STFT` takes its frame step from `{operand.name}`, "
            f"which holds {'no single value' if fixed is not None else 'no value'} known "
            "at compile time; how many frames the signal yields then depends on input "
            "data, which the C compiler cannot compile."
        )
    step = int(fixed.reshape(-1)[0])
    if step < 1:
        raise CompileError(
            f"Node `{context.label}`: `STFT` steps {step} sample(s) between frames; ONNX "
            "defines the step as the samples to advance by, which is positive."
        )
    return step


def _frame_length(context: NodeContext, window: TensorRef | None) -> int:
    """How many samples one frame holds, from whichever operand the node states it with.

    ONNX takes it from `frame_length`, and from the window's own extent where the node
    passes a window instead; a node passing both states the same number twice.
    """
    stated = context.optional_input(3)
    fixed = context.constant_input(3) if stated is not None else None
    if stated is not None and (fixed is None or fixed.size != 1):
        raise CompileError(
            f"Node `{context.label}`: `STFT` takes its frame length from `{stated.name}`, "
            f"which holds {'no single value' if fixed is not None else 'no value'} known "
            "at compile time; the shape of the result then depends on input data, which "
            "the C compiler cannot compile."
        )
    spanned = _window_span(context, window)
    if fixed is None:
        if spanned is None:
            raise CompileError(
                f"Node `{context.label}`: `STFT` states neither a window nor a frame "
                "length; ONNX defines the frame from one of the two."
            )
        return spanned
    length = int(fixed.reshape(-1)[0])
    if spanned is not None and spanned != length:
        raise CompileError(
            f"Node `{context.label}`: `STFT` was given a window of {spanned} sample(s) and "
            f"a frame length of {length}; ONNX defines a node stating both as stating the "
            "same number twice."
        )
    return length


def _window_span(context: NodeContext, window: TensorRef | None) -> int | None:
    if window is None:
        return None
    if len(window.shape) != 1:
        raise CompileError(
            f"Node `{context.label}`: `STFT` was given the window `{window.name}` of shape "
            f"{list(window.shape)}; ONNX defines it as a single sequence of weights."
        )
    return window.shape[0]


def _normalize_helper(prefix: str) -> CFunction:
    """An axis counted from the end, resolved against a rank, both known only as values."""
    name = f"{prefix}_normalized_axis"
    return CFunction(name, _NORMALIZE_TEMPLATE.substitute(name=name))


def _dispatch(
    context: NodeContext,
    operand: TensorRef,
    rank: int,
    calls: dict[int, str],
    normalize: CFunction,
) -> str:
    """The transform for whichever axis the operand names at run time, or an argument error.

    The axis is resolved through a function rather than into a local, so that the statement
    introduces no identifier of its own — one would shadow the entrypoint parameter a tensor
    of the same name is emitted as.
    """
    cases = []
    for axis, call in sorted(calls.items()):
        cases.append(f"case {axis}:")
        cases.extend(f"    {line}" if line else "" for line in call.splitlines())
        cases.append("    break;")
    return "\n".join(
        [
            f"switch ({normalize.name}((int64_t){operand.expr}[0], {rank})) {{",
            *cases,
            "default:",
            f"    return {context.prefix.upper()}_{INVALID_ARGUMENT_STATUS};",
            "}",
        ]
    )


register_kernel("", "DFT", _DFT_VERSIONS, _dft)
register_kernel("", "STFT", _STFT_VERSIONS, _stft)
