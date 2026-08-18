"""Compiling one artifact for a whole family of shapes, not for a single one.

A runtime dimension is compiled by compiling the model *repeatedly*, at a spread of values
for that dimension, and reading the family off the results: a tensor axis that is the same
at every value is a constant, one that is the dimension's value times a fixed factor is that
multiple, and anything else is a shape the artifact cannot be one piece of code for. The
same reading turns the compile-time literals in the emitted call sites into expressions over
the dimension's entrypoint parameter, so kernels loop to the size a call actually asks for
while every buffer stays sized for the maximum.

Probing rather than symbolic inference is deliberate: the frontend that derives the shapes,
and the kernels that turn them into code, stay exactly the ones a fixed-shape compilation
uses — there is no second, symbolic implementation of either that could disagree with it.
What the probes do not agree on is rejected, never guessed at, so the artifact is either
correct across the whole family or refused at compile time.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.codegen import (
    IOTensor,
    NodeEntry,
    Program,
    StaticBuffer,
)
from fnnx.extras.compilers.c.onnx.runtime_dims import RuntimeDim, ShapeTerm

Builder = Callable[[Mapping[str, int]], Program]

# A decimal integer standing on its own: not part of an identifier, and not the fraction or
# the exponent of a floating-point literal.
_LITERAL = re.compile(r"(?<![A-Za-z0-9_.])(\d+)[uU]?(?![A-Za-z0-9_.])")

# `INT64_C(3)` pastes its argument onto a suffix, so no expression can take its place; a
# literal in that position has to be the same at every probe or the model is rejected.
_MACRO_ARGUMENT = re.compile(r"[A-Za-z0-9_]\($")

# Lifted literals are evaluated in C's `int`, which the artifact's contract already bounds:
# a count past this would be a buffer no static allocation could hold anyway.
_INT_MAX = 2**31 - 1

# Below this maximum, a dimension has fewer than three sizes above 1 to read the emitted
# code at, which is too few to check an affine reading against a size it was not derived
# from; `_liftable_probes` then has to read the dimension at 1 as well.
_SMALLEST_GENERAL_MAXIMUM = 4


def specialize(
    build: Builder,
    dims: Sequence[RuntimeDim],
    *,
    dim_bindings: Mapping[str, int],
) -> Program:
    """Compile through `build` for every size in the runtime dimensions' ranges.

    `build` compiles the model at one concrete set of dimension values, exactly as a
    fixed-shape compilation would. The returned program is the one built at the maxima —
    so every buffer is sized for the capacity — with its emitted code rewritten to work at
    the value each call passes.
    """
    probes = _probe_bindings(dims)
    programs = [_build_at(build, dim_bindings, probe, dims) for probe in probes]
    liftable = _liftable_probes(probes, dims)
    base = programs[0]

    _check_signatures(programs, dims)
    _check_definitions(programs, liftable, dims)
    nodes = tuple(
        replace(
            entry,
            inputs=_shaped(
                entry.inputs,
                [p.nodes[index].inputs for p in programs],
                probes,
                dims,
                entry.id,
            ),
            outputs=_shaped(
                entry.outputs,
                [p.nodes[index].outputs for p in programs],
                probes,
                dims,
                entry.id,
            ),
            body=_lift_body(
                [p.nodes[index] for p in programs], liftable, probes, dims, entry.id
            ),
        )
        for index, entry in enumerate(base.nodes)
    )
    return replace(
        base,
        inputs=_shaped(base.inputs, [p.inputs for p in programs], probes, dims, ""),
        outputs=_shaped(base.outputs, [p.outputs for p in programs], probes, dims, ""),
        body=_lift_body(programs, liftable, probes, dims, ""),
        nodes=nodes,
        runtime_dims=tuple(dims),
        # The probe values are how the family was explored, not bindings the artifact was
        # compiled under; only the dimensions genuinely fixed at compile time are reported.
        dim_bindings={
            name: value
            for name, value in base.dim_bindings.items()
            if name not in {dim.name for dim in dims}
        },
    )


# --------------------------------------------------------------------------------------
# The values the model is compiled at
# --------------------------------------------------------------------------------------


def _build_at(
    build: Builder,
    dim_bindings: Mapping[str, int],
    probe: Mapping[str, int],
    dims: Sequence[RuntimeDim],
) -> Program:
    """Compile at one probe, reporting a failure as one of the whole family.

    A model that only compiles at some sizes — a reshape into a fixed extent the smallest
    size cannot fill, say — is one this artifact cannot be, and the size it broke at is the
    most useful thing to say about it.
    """
    try:
        return build({**dim_bindings, **probe})
    except CompileError as error:
        raise _untrackable(
            f"compiling it at {_probe_label(probe, dims)} failed: {error}", "", dims
        ) from error


def probe_values(dim: RuntimeDim) -> tuple[int, ...]:
    """The sizes one dimension is probed at.

    1, 2 and 3 pin down the small end of the range — where a shape that clamps, saturates
    or rounds parts company with a linear one — and the two largest values anchor the other
    end, so that a fit through the small values has to hold at the capacity as well.
    """
    candidates = {1, 2, 3, dim.maximum - 1, dim.maximum}
    return tuple(sorted(value for value in candidates if 1 <= value <= dim.maximum))


def _probe_bindings(dims: Sequence[RuntimeDim]) -> tuple[dict[str, int], ...]:
    """One probe per (dimension, size), every other dimension held at its maximum.

    The first probe holds every dimension at its maximum: it is the one the artifact's
    buffers, macros and reported footprint are taken from. A last probe moves every
    dimension away from its maximum at once, because the ones above move one at a time — a
    size that is the *product* of two dimensions agrees with a linear reading along each of
    them separately, and parts company with it only where both move together.
    """
    maxima = {dim.name: dim.maximum for dim in dims}
    probes = [dict(maxima)]
    probes += [
        {**maxima, dim.name: value}
        for dim in dims
        for value in probe_values(dim)
        if value != dim.maximum
    ]
    if len(dims) > 1:
        probes.append({dim.name: min(2, dim.maximum) for dim in dims})
    seen: set[tuple[tuple[str, int], ...]] = set()
    distinct = []
    for probe in probes:
        key = tuple(sorted(probe.items()))
        if key not in seen:
            seen.add(key)
            distinct.append(probe)
    return tuple(distinct)


def _liftable_probes(
    probes: Sequence[Mapping[str, int]], dims: Sequence[RuntimeDim]
) -> tuple[int, ...]:
    """The probes whose emitted code is compared against each other, by index.

    A dimension of 1 makes a tensor's axis vanish: an operand stops broadcasting, a
    concatenation becomes contiguous, and kernels legitimately emit a different — faster,
    equally correct — form for it. Comparing those forms against the general one would
    reject a model the general form serves perfectly well, so the code is read off the
    sizes above 1 while the *shapes* are still checked at 1 as well.

    Leaving 1 out is only safe while the sizes that remain outnumber what an affine reading
    of a literal has free parameters: two sizes determine a slope and an intercept, so a
    quantity that is quadratic in the dimension would fit them and be lifted wrongly. A
    dimension whose maximum is too small to spare three sizes above 1 is therefore read at
    1 as well — its whole range then appears among the probes, which makes the reading exact
    rather than fitted, at the price of rejecting a model whose kernels take a size-1 form.
    """
    return tuple(
        index
        for index, probe in enumerate(probes)
        if all(
            probe[dim.name] > 1
            for dim in dims
            if dim.maximum >= _SMALLEST_GENERAL_MAXIMUM
        )
    )


# --------------------------------------------------------------------------------------
# Shape families
# --------------------------------------------------------------------------------------


def _shaped(
    tensors: Sequence[IOTensor],
    across: Sequence[Sequence[IOTensor]],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    owner: str,
) -> tuple[IOTensor, ...]:
    return tuple(
        replace(
            tensor,
            runtime_shape=_shape_terms(
                [variant[index].shape for variant in across],
                probes,
                dims,
                subject=f"tensor `{tensor.name}`",
                owner=tensor.owner or owner,
            ),
        )
        for index, tensor in enumerate(tensors)
    )


def _shape_terms(
    shapes: Sequence[tuple[int, ...]],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    *,
    subject: str,
    owner: str,
) -> tuple[ShapeTerm, ...]:
    """Read one tensor's shape family off the shapes it took at each probe."""
    base = shapes[0]
    if any(len(shape) != len(base) for shape in shapes):
        raise _untrackable(
            f"the rank of {subject} changes with the dimension's value", owner, dims
        )
    return tuple(
        _axis_term(
            [shape[axis] for shape in shapes],
            probes,
            dims,
            subject=subject,
            subject_owner=owner,
            axis=axis,
        )
        for axis in range(len(base))
    )


def _axis_term(
    sizes: Sequence[int],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    *,
    subject: str,
    subject_owner: str,
    axis: int,
) -> ShapeTerm:
    """One axis as a constant or as a fixed multiple of one runtime dimension."""
    if len(set(sizes)) == 1:
        return ShapeTerm(sizes[0])
    for dim in dims:
        coefficient, remainder = divmod(sizes[0], dim.maximum)
        if remainder or coefficient < 1:
            continue
        if all(
            size == coefficient * probe[dim.name] for size, probe in zip(sizes, probes)
        ):
            return ShapeTerm(sizes[0], dim.name, coefficient)
    raise _untrackable(
        f"axis {axis} of {subject} is neither a constant nor a constant multiple of a "
        f"runtime dimension ({_observations(probes, sizes, dims)})",
        subject_owner,
        dims,
    )


# --------------------------------------------------------------------------------------
# Lifting the emitted code
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    """A literal read as `intercept + sum(slope * dimension value)`."""

    intercept: int
    slopes: tuple[int, ...]

    def at(self, probe: Mapping[str, int], dims: Sequence[RuntimeDim]) -> int:
        return self.intercept + sum(
            slope * probe[dim.name] for slope, dim in zip(self.slopes, dims)
        )

    def bounds(self, dims: Sequence[RuntimeDim]) -> tuple[int, int]:
        """The smallest and largest values the fit takes anywhere in the dimensions' ranges."""
        low = high = self.intercept
        for slope, dim in zip(self.slopes, dims):
            ends = (slope, slope * dim.maximum)
            low += min(ends)
            high += max(ends)
        return low, high

    def render(self, dims: Sequence[RuntimeDim]) -> str:
        text = ""
        for slope, dim in zip(self.slopes, dims):
            if slope:
                text += f" - {-slope}" if slope < 0 else f" + {slope}"
                text += f" * {dim.c_name}"
        if self.intercept or not text:
            text += (
                f" - {-self.intercept}"
                if self.intercept < 0
                else f" + {self.intercept}"
            )
        return "(" + text.removeprefix(" + ").lstrip() + ")"


def _lift_body(
    programs: Sequence[Program | NodeEntry],
    liftable: Sequence[int],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    entry: str,
) -> tuple[str, ...]:
    """One entrypoint's statements, with every size-dependent literal made an expression."""
    bodies = [programs[index].body for index in liftable]
    at = [probes[index] for index in liftable]
    owners = programs[0].body_owners
    where = f"node entrypoint `{entry}`" if entry else "the model entrypoint"
    if len({len(body) for body in bodies}) != 1:
        raise _untrackable(
            f"the code emitted for {where} changes shape with the dimension's value",
            "",
            dims,
        )
    return tuple(
        _lift_statement(
            [body[index] for body in bodies],
            at,
            dims,
            owner=owners[index] if index < len(owners) else "",
        )
        for index in range(len(bodies[0]))
    )


def _lift_statement(
    variants: Sequence[str],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    *,
    owner: str,
) -> str:
    separators, tokens = zip(*(_split(variant) for variant in variants))
    if len(set(separators)) != 1:
        raise _untrackable(
            "the code emitted for it is not the same at every size", owner, dims
        )
    pieces = list(separators[0])
    lifted = [
        _lift_literal(
            [token[index] for token in tokens],
            probes,
            dims,
            owner=owner,
            # A literal an expression cannot stand in for: a macro pastes its argument onto
            # a suffix, and one inside a string literal is text rather than a size.
            pasted=bool(_MACRO_ARGUMENT.search(pieces[index]))
            or '"' in "".join(pieces[: index + 1]),
        )
        for index in range(len(tokens[0]))
    ]
    text = pieces[0]
    for literal, piece in zip(lifted, pieces[1:]):
        text += literal + piece
    return text


def _split(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`text` as the runs between its integer literals, and the literals themselves."""
    separators, literals, position = [], [], 0
    for match in _LITERAL.finditer(text):
        separators.append(text[position : match.start()])
        literals.append(match.group(0))
        position = match.end()
    separators.append(text[position:])
    return tuple(separators), tuple(literals)


def _lift_literal(
    literals: Sequence[str],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
    *,
    owner: str,
    pasted: bool,
) -> str:
    if len(set(literals)) == 1:
        return literals[0]
    if pasted:
        raise _untrackable(
            "a value it pastes into a macro or a string depends on the dimension's value",
            owner,
            dims,
        )
    values = [int(literal.rstrip("uU")) for literal in literals]
    fit = _fit(values, probes, dims)
    if fit is None:
        raise _untrackable(
            "a size in the code emitted for it does not scale linearly with the "
            f"dimension's value ({_observations(probes, values, dims)})",
            owner,
            dims,
        )
    low, high = fit.bounds(dims)
    if low < 0 or high > _INT_MAX:
        raise _untrackable(
            f"a size in the code emitted for it runs to {low}..{high} across the "
            "dimension's range, which no buffer of the artifact could be indexed by",
            owner,
            dims,
        )
    return fit.render(dims)


def _fit(
    values: Sequence[int],
    probes: Sequence[Mapping[str, int]],
    dims: Sequence[RuntimeDim],
) -> _Fit | None:
    """The one affine reading of `values`, or None where they do not admit one.

    Every probe holds all but one dimension at its maximum, so each slope follows from the
    pair of probes that differ in that dimension alone; the reading is then checked against
    every probe, which is what rejects a size that is quadratic in a dimension, that
    saturates, or that mixes two dimensions into a product.
    """
    slopes = []
    for dim in dims:
        pair = next(
            (
                index
                for index, probe in enumerate(probes)
                if probe[dim.name] != dim.maximum
                and all(
                    probe[other.name] == other.maximum
                    for other in dims
                    if other is not dim
                )
            ),
            None,
        )
        if pair is None:
            slopes.append(0)
            continue
        rise = values[0] - values[pair]
        run = dim.maximum - probes[pair][dim.name]
        if rise % run:
            return None
        slopes.append(rise // run)
    fit = _Fit(
        intercept=values[0] - sum(s * d.maximum for s, d in zip(slopes, dims)),
        slopes=tuple(slopes),
    )
    if any(fit.at(probe, dims) != value for probe, value in zip(probes, values)):
        return None
    return fit


# --------------------------------------------------------------------------------------
# What every probe has to agree on
# --------------------------------------------------------------------------------------


def _check_signatures(programs: Sequence[Program], dims: Sequence[RuntimeDim]) -> None:
    """Which entrypoints the artifact publishes, and what each takes, is size-independent.

    Read at *every* probe, the degenerate ones included: the shapes checked against these
    signatures are the model's own semantics, not a form some kernel happened to emit.
    """
    base = programs[0]
    shapes = [
        (len(program.inputs), len(program.outputs))
        + tuple(
            (entry.id, entry.symbol, len(entry.inputs), len(entry.outputs))
            for entry in program.nodes
        )
        for program in programs
    ]
    if len(set(shapes)) != 1:
        raise _untrackable(
            f"the entrypoints of `{base.prefix}` change with the dimension's value",
            "",
            dims,
        )


def _check_definitions(
    programs: Sequence[Program], liftable: Sequence[int], dims: Sequence[RuntimeDim]
) -> None:
    """Everything outside the entrypoint bodies has to be identical at every size.

    Weights and constant tables are the compile-time values the graph fixes, and a value
    that moves with a dimension is a computation folded away that the artifact would have to
    do at run time. Those are read at *every* probe, size 1 included: nothing about a folded
    value is a form the emitter chose, and size 1 is where a clamped or degenerate fold most
    often parts company with the rest of the range — exactly the probe the code comparison
    leaves out. Which kernels get emitted, in contrast, *is* such a form, so it is read
    where the code is, alongside the buffers those kernels ask for.
    """
    base = programs[0]
    for other in programs[1:]:
        # Tensor by tensor first, so that the one that moved can be named — it is the only
        # thread back to the computation the fold took out of the graph.
        for one, another in zip(base.weights, other.weights):
            _same(one, another, f"the constant `{one.name}` it embeds", dims)
        _same(base.weights, other.weights, "the constant data it embeds", dims)
        _same(
            base.labels,
            other.labels,
            "the set of class-label tables it publishes",
            dims,
        )
    for index in liftable[1:]:
        other = programs[index]
        _same(base.functions, other.functions, "the set of kernels it emits", dims)
        _check_capacity(base.scratch, other.scratch, dims)


def _same(
    base: object, other: object, subject: str, dims: Sequence[RuntimeDim]
) -> None:
    if base != other:
        raise _untrackable(f"{subject} changes with the dimension's value", "", dims)


def _check_capacity(
    base: Sequence[StaticBuffer],
    other: Sequence[StaticBuffer],
    dims: Sequence[RuntimeDim],
) -> None:
    """The buffers planned at the maxima have to hold every smaller size as well."""
    sized = {buffer.symbol: buffer.declared_count for buffer in base}
    for buffer in other:
        capacity = sized.get(buffer.symbol)
        if capacity is None:
            raise _untrackable(
                f"the buffer `{buffer.symbol}` it reserves depends on the dimension's "
                "value",
                "",
                dims,
            )
        if capacity < buffer.declared_count:
            raise _untrackable(
                f"the buffer `{buffer.symbol}` needs {buffer.declared_count} elements at "
                f"a smaller size than the {capacity} it is given at the maximum",
                "",
                dims,
            )


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def _untrackable(detail: str, owner: str, dims: Sequence[RuntimeDim]) -> CompileError:
    named = ", ".join(f"`{dim.name}`" for dim in dims)
    where = f"Node `{owner}`: " if owner else ""
    # With such a maximum, size 1 is one of the sizes the emitted code is read at (see
    # `_liftable_probes`), and a kernel emits a different — still correct — form there,
    # which is a failure of the schedule rather than of the model.
    hint = (
        f" A maximum below {_SMALLEST_GENERAL_MAXIMUM} leaves too few sizes above 1 to "
        "read the code at; try a larger one."
        if any(dim.maximum < _SMALLEST_GENERAL_MAXIMUM for dim in dims)
        else ""
    )
    return CompileError(
        f"{where}{detail}. Runtime dimension(s) {named} cannot be tracked through this "
        f"model; pin them via `dim_bindings` to compile it.{hint}"
    )


def _probe_label(probe: Mapping[str, int], dims: Sequence[RuntimeDim]) -> str:
    return ", ".join(f"{dim.name}={probe[dim.name]}" for dim in dims)


def _observations(
    probes: Sequence[Mapping[str, int]],
    values: Sequence[int],
    dims: Sequence[RuntimeDim],
) -> str:
    return ", ".join(
        f"{_probe_label(probe, dims)} -> {value}"
        for probe, value in zip(probes, values)
    )
