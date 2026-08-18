"""Differential sweep: every registered kernel against the ONNX reference evaluator.

The backend corpus is example-based. It does not cover every attribute combination, every
dtype a schema allows, or the numerical edges, so this suite generates single-node models
systematically and takes every expected value from `onnx.reference.ReferenceEvaluator` --
the executable form of the spec. Nothing here decides what an op should compute.

**Oracle validity.** The evaluator carries a versioned implementation class only for the
revisions whose semantics it distinguishes; to every other opset it silently applies the
newest semantics it knows. A case is therefore generated only for an (op, version) pair the
evaluator is version-faithful for -- the same mechanical check the compiler's folding pass
applies, never an assumption about which revisions "did not really change". Registered
revisions outside that set rest on the backend corpus's frozen old-opset tests, whose
expected outputs are stored rather than computed.

**Acceptance rule.** An op counts as implemented only when both suites pass for it. The
tests at the bottom fail if the kernel registry serves an op this sweep does not execute a
case for, or one that no test in the conformance pass list exercises.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import ModelProto, TensorProto, helper, numpy_helper  # noqa: E402
from onnx.backend.test.loader import load_model_tests  # noqa: E402
from onnx.backend.test.runner import Runner  # noqa: E402
from onnx.defs import OpSchema, get_schema  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.onnx.dtypes import (  # noqa: E402
    C_TYPES,
    FLOAT_TYPES,
    UNSIGNED_TYPES,
    c_type,
    element_size,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.folding import (  # noqa: E402
    evaluator_is_version_faithful,
)
from fnnx.extras.compilers.c.onnx.kernels import (  # noqa: E402
    KERNELS,
    CFunction,
    NodeContext,
    NodeEmission,
)
from fnnx.extras.compilers.c.onnx.loader import (  # noqa: E402
    ML_DOMAIN,
    display_domain,
    normalize_domain,
)
from fnnx.extras.compilers.c.onnx.registry import KernelSpec  # noqa: E402

# Every draw below comes from `numpy.random.default_rng([SEED, operand index])`, so a
# reported case reproduces exactly.
SEED = 20260726

# ONNX's own backend-test defaults. The compiler is compared under one tolerance policy, and
# that policy is ONNX's; a kernel that needs more than the conformance suite's bounded
# per-op overrides allow is a wrong kernel, here as much as there.
RTOL = 1e-3
ATOL = 1e-7

# The conformance suite's checked-in record of which corpus tests pass, and the `onnx`
# release it -- and the corpus it names -- are keyed to.
RATCHET_PATH = Path(__file__).parent / "conformance" / "passing.txt"
LEDGER_PATH = Path(__file__).parent / "conformance" / "ledger.json"
PINNED_ONNX = "1.22"

pytestmark = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


# --------------------------------------------------------------------------------------
# What the sweep covers
# --------------------------------------------------------------------------------------


class Kind(Enum):
    """What an op does to its operands, which is what bounds the values it can be fed.

    The oracle is the reference evaluator either way; this only decides which inputs a
    disagreement with it would be attributable to the kernel.
    """

    # Nothing the op computes can leave the dtype's range, so it sees every special value
    # the dtype has, including its extremes.
    POINTWISE = "pointwise"
    # Elementwise arithmetic: IEEE pins the float edges (Inf, NaN, signed zero, overflow to
    # Inf) exactly, so floats still sweep them, but integers stay small -- ONNX does not
    # define integer overflow and C's is undefined for the signed families, so a wrap-around
    # difference would not be a divergence from the spec.
    ARITHMETIC = "arithmetic"
    # Sums many products, in an order the spec does not fix: with Inf or dtype extremes in
    # one dot product the reference's summation order and the kernel's legitimately
    # disagree. These cases run on finite values, and their edge behaviour rests on the
    # backend corpus.
    ACCUMULATING = "accumulating"


class Domain(Enum):
    """A restriction on an operand's values, for what ONNX itself leaves undefined."""

    # Integer division and remainder by zero: undefined by ONNX, undefined in C, and a trap
    # on the platforms this compiler targets. Integer operands only -- the floating-point
    # case is pinned by IEEE, so those operands keep their full range of special values.
    NONZERO = "nonzero"
    # A negative integer exponent, which numpy refuses to evaluate at all, and one large
    # enough to overflow the dtype, which ONNX does not define. Integer operands only.
    SMALL_EXPONENT = "small_exponent"
    # A floating-point operand converted to an integer type, which ONNX and C alike leave
    # undefined outside the target's range: the special values that cannot survive the
    # conversion -- NaN, the infinities, the dtype's extremes -- are pulled into a range
    # every signed integer type holds. Integer operands convert for every value, so they are
    # left alone.
    CONVERTIBLE = "convertible"
    # The same, for an unsigned target, which holds no negative value however small.
    CONVERTIBLE_UNSIGNED = "convertible_unsigned"
    # A factor of a product, running or reduced: many of them multiply together, so integer
    # operands are pulled into {-1, 0, 1}, where no product can leave the dtype's range —
    # ONNX does not define integer overflow and C's is undefined for the signed families.
    SMALL_FACTOR = "small_factor"
    # An operand a square root or a fractional power is taken of. A negative one puts the
    # whole result at NaN, which compares equal to itself and would leave the case asserting
    # nothing at all; the reference evaluator's LpNormalization needs it for a second reason,
    # recorded where it is asked for.
    NONNEGATIVE = "nonnegative"


@dataclass(frozen=True)
class Variant:
    """One attribute combination and the operand shapes it applies to.

    `None` leaves an optional operand out. A variant is generated only at the versions whose
    schema takes the operands it declares, so one table serves every revision of an op;
    `versions` and `elem_types` narrow that further, for attributes a revision does not have
    and combinations a dtype does not allow. `values` pins an operand a variant treats as a
    parameter rather than as data — Clip's bounds, a reduction's axes — where a seeded draw
    would make the case's coverage an accident of the seed; a single value fills the operand
    and a sequence gives it element by element. `domains` restricts an operand for this
    variant alone, where what ONNX leaves undefined depends on the attributes — Cast's target
    type. `operand_types` does the same for an operand's element type, where a schema leaves
    a choice the sweep's own type operand does not range over — the integer width of a
    gathering op's indices. `outputs` asks for the optional results an op computes alongside
    its first — LayerNormalization's mean, BatchNormalization's running statistics — which
    are then compared like any other output.
    """

    label: str
    shapes: tuple[tuple[int, ...] | None, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    values: Mapping[int, float | Sequence[float]] = field(default_factory=dict)
    domains: Mapping[int, Domain] = field(default_factory=dict)
    operand_types: Mapping[int, int] = field(default_factory=dict)
    versions: tuple[int, ...] | None = None
    elem_types: tuple[int, ...] | None = None
    outputs: int = 1


@dataclass(frozen=True)
class Sweep:
    """Everything generated for one op: its variants, and how its operands are fed.

    `type_operand` is the operand whose schema type constraint enumerates the element types
    swept — the first, unless the op takes something else there (Where's condition).
    `operand_types` fixes the element type of the operands the schema does not leave free.
    `constant_operands` are carried in the model as initializers rather than fed at run time:
    an op that reads an operand as configuration — a reduction's axes — needs it at compile
    time, and a model that computes it is a model the compiler refuses by design.
    `equivalent_model` builds what the oracle is run on instead, for an op ONNX defines as
    equal to another one and implements either nowhere or not faithfully; the expected values
    still come from the oracle, on the model ONNX's own specification says computes the same
    thing. `oracle` replaces the reference evaluator itself, for the one op ONNX ships no
    reference implementation for at all -- see the MaxRoiPool entry, which is the only user.
    """

    kind: Kind
    variants: tuple[Variant, ...]
    operand_domains: Mapping[int, Domain] = field(default_factory=dict)
    type_operand: int = 0
    operand_types: Mapping[int, int] = field(default_factory=dict)
    constant_operands: tuple[int, ...] = ()
    equivalent_model: Callable[[Case], ModelProto] | None = None
    oracle: Callable[[ModelProto, Mapping[str, Any]], list[Any]] | None = None


# Broadcasting cases: equal shapes, trailing-axis alignment, both-ways broadcasting, a
# scalar operand, mismatched ranks, rank 0, and zero-element tensors on either axis. `wide`
# is the only one larger than the special-value list on both sides, so it is what carries
# every dtype edge into both operands; the rest are shape coverage.
_BROADCAST_VARIANTS = (
    Variant("wide", ((4, 8), (4, 8))),
    Variant("same", ((2, 3), (2, 3))),
    Variant("trailing_axis", ((2, 3), (3,))),
    Variant("both_ways", ((1, 3), (2, 1))),
    Variant("scalar_operand", ((2, 3), ())),
    Variant("mixed_rank", ((2, 1, 3), (4, 3))),
    Variant("rank_0", ((), ())),
    Variant("empty_rows", ((0, 3), (3,))),
    Variant("empty_axis", ((2, 0, 3), (3,))),
)

# `wide` is larger than the special-value list, so seeded random draws reach these ops too.
_UNARY_VARIANTS = (
    Variant("matrix", ((2, 3),)),
    Variant("rank_0", ((),)),
    Variant("empty", ((0, 3),)),
    Variant("wide", ((4, 8),)),
)

# PRelu broadcasts its slope onto the data unidirectionally: the result keeps the data's
# shape, so only the second operand may be stretched.
_UNIDIRECTIONAL_VARIANTS = (
    Variant("wide", ((4, 8), (4, 8))),
    Variant("trailing_axis", ((2, 3), (3,))),
    Variant("scalar_operand", ((2, 3), ())),
    Variant("rank_0", ((), ())),
    Variant("empty_rows", ((0, 3), (3,))),
)

# The variadic families take one operand or many, each broadcasting against the others. The
# broadcast case gives the first operand the result's own shape: the reference's Mean
# accumulates into a copy of it, so it cannot evaluate a case where a later operand widens
# the result, and there is no oracle for one.
_VARIADIC_VARIANTS = (
    Variant("wide", ((4, 8), (4, 8), (4, 8))),
    Variant("single", ((2, 3),)),
    Variant("pair", ((2, 3), (2, 3))),
    Variant("broadcast", ((2, 4, 3), (4, 3), ())),
    Variant("rank_0", ((), ())),
    Variant("empty_rows", ((0, 3), (3,))),
)

_FLOAT_ELEM_TYPES = tuple(sorted(FLOAT_TYPES))

# Clip's bounds are scalars the op treats as parameters, so they are pinned rather than
# drawn: an inverted pair (numpy applies the lower bound first, so the upper one wins) and a
# NaN bound (which wins outright) are edges a random draw would only reach by luck.
_CLIP_BOUND_VALUES = (
    ("both", {1: -1.0, 2: 1.0}, None),
    ("inverted", {1: 1.0, 2: -1.0}, None),
    ("nan_low", {1: float("nan"), 2: 1.0}, _FLOAT_ELEM_TYPES),
)
_CLIP_VARIANTS = (
    Variant("unbounded", ((4, 8),)),
    Variant("unbounded_rank_0", ((),)),
    Variant("unbounded_empty", ((0, 3),)),
    Variant("low_only", ((4, 8), ()), values={1: 0.0}),
    Variant("high_only", ((4, 8), None, ()), values={2: 0.0}),
    *(
        Variant(label, ((4, 8), (), ()), values=values, elem_types=elem_types)
        for label, values, elem_types in _CLIP_BOUND_VALUES
    ),
    # Up to opset 10 the bounds are attributes instead, which is a kernel of its own.
    Variant("attributes", ((4, 8),), {"min": -1.0, "max": 1.0}, versions=(6,)),
    Variant("attribute_low", ((4, 8),), {"min": 0.0}, versions=(6,)),
    Variant("attribute_high", ((4, 8),), {"max": 0.0}, versions=(6,)),
)

# Dropout is the identity in inference mode, whatever ratio it is handed — including one
# that is not a number at all.
_DROPOUT_VARIANTS = (
    Variant("data", ((2, 3),)),
    Variant("wide", ((4, 8),)),
    Variant("empty", ((0, 3),)),
    Variant("ratio", ((4, 8), ()), values={1: 0.75}),
    Variant("ratio_zero", ((4, 8), ()), values={1: 0.0}),
    Variant("ratio_nan", ((4, 8), ()), values={1: float("nan")}),
)

# ONNX defines Mod's fmod=0 for the integer families only, so the floored formula is swept
# there alone; the broadcasting it shares with fmod=1 is covered by the shape family above.
_INTEGER_TYPES = tuple(
    elem_type
    for elem_type in sorted(C_TYPES)
    if elem_type not in FLOAT_TYPES and elem_type != TensorProto.BOOL
)
_MOD_VARIANTS = tuple(
    replace(variant, label=f"{variant.label}_fmod", attributes={"fmod": 1})
    for variant in _BROADCAST_VARIANTS
) + tuple(
    replace(variant, label=f"{variant.label}_floored", elem_types=_INTEGER_TYPES)
    for variant in _BROADCAST_VARIANTS
    if variant.label in ("wide", "trailing_axis")
)

# BitShift's direction is a required attribute, so every shape carries one of the two.
_BIT_SHIFT_VARIANTS = tuple(
    replace(
        variant,
        label=f"{variant.label}_{direction.lower()}",
        attributes={"direction": direction},
    )
    for direction in ("LEFT", "RIGHT")
    for variant in _BROADCAST_VARIANTS
)

# IsInf's attributes decide which infinities count, down to neither of them, where the
# operand goes unread.
_IS_INF_ATTRIBUTES: Mapping[str, Mapping[str, Any]] = {
    "positive_only": {"detect_negative": 0},
    "negative_only": {"detect_positive": 0},
    "neither": {"detect_positive": 0, "detect_negative": 0},
}

# Where broadcasts all three operands against each other; the condition is boolean whatever
# the branches carry, which is what the sweep's `operand_types` pins.
_SELECT_VARIANTS = (
    Variant("wide", ((4, 8), (4, 8), (4, 8))),
    Variant("same", ((2, 3), (2, 3), (2, 3))),
    Variant("condition_broadcast", ((2, 1), (2, 3), (2, 3))),
    Variant("branch_broadcast", ((2, 3), (3,), ())),
    Variant("every_operand_stretched", ((2, 1, 1), (1, 3, 1), (1, 1, 4))),
    Variant("rank_0", ((), (), ())),
    Variant("empty_rows", ((0, 3), (3,), ())),
)


def _cast_variants() -> tuple[Variant, ...]:
    """Every supported target type, plus the shapes a conversion does not vary over.

    The source types come from the schema, so each variant is generated once per source: the
    pair matrix is the cross product of the two, the identity conversions included.
    """
    variants = [
        Variant(
            f"to_{numpy_dtype_name(target)}",
            ((4, 8),),
            {"to": target},
            # Only the conversions to another floating-point type, or to bool, are defined
            # for every value a float operand can hold.
            domains=(
                {}
                if target in FLOAT_TYPES or target == TensorProto.BOOL
                else {
                    0: (
                        Domain.CONVERTIBLE_UNSIGNED
                        if target in UNSIGNED_TYPES
                        else Domain.CONVERTIBLE
                    )
                }
            ),
        )
        for target in sorted(C_TYPES)
    ]
    variants += [
        Variant(label, (shape,), {"to": TensorProto.DOUBLE})
        for label, shape in (("matrix", (2, 3)), ("rank_0", ()), ("empty", (0, 3)))
    ]
    return tuple(variants)


def _bitcast_variants() -> tuple[Variant, ...]:
    """One variant per target type, offered to the source types of that same width.

    ONNX defines BitCast only between types of equal width, and this compiler refuses a
    boolean target outright — its bytes are contractually 0 or 1, which arbitrary bits are
    not — so neither is generated here; both are error-path tests instead.
    """
    by_width: dict[int, list[int]] = {}
    for elem_type in sorted(C_TYPES):
        by_width.setdefault(element_size(elem_type), []).append(elem_type)
    variants = [
        Variant(
            f"to_{numpy_dtype_name(target)}",
            ((4, 8),),
            {"to": target},
            elem_types=tuple(members),
        )
        for members in by_width.values()
        for target in members
        if target != TensorProto.BOOL
    ]
    variants += [
        Variant(
            label,
            (shape,),
            {"to": TensorProto.INT32},
            elem_types=(TensorProto.FLOAT,),
        )
        for label, shape in (("rank_0", ()), ("empty", (0, 3)))
    ]
    return tuple(variants)


# Gemm's attributes are swept as a full cross product rather than a chosen handful: which
# combinations interact is exactly what a hand-picked list would be guessing at. The bias
# shapes cover every way C broadcasts onto the result, including leaving it out.
_GEMM_BIASES = (
    ("vector", (4,)),
    ("matrix", (2, 4)),
    ("row", (1, 4)),
    ("scalar", ()),
    ("absent", None),
)


def _gemm_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(
            f"transA{transpose_left}_transB{transpose_right}"
            f"_alpha{alpha}_beta{beta}_bias_{bias_label}",
            (
                (3, 2) if transpose_left else (2, 3),
                (4, 3) if transpose_right else (3, 4),
                bias,
            ),
            {
                "transA": transpose_left,
                "transB": transpose_right,
                "alpha": alpha,
                "beta": beta,
            },
        )
        for transpose_left in (0, 1)
        for transpose_right in (0, 1)
        for alpha, beta in ((1.0, 1.0), (0.5, 2.0), (1.5, 0.0))
        for bias_label, bias in _GEMM_BIASES
    ]
    # Zero-element operands, which no attribute combination reaches: an empty result, and an
    # empty contraction whose sum is over nothing at all.
    variants += [
        Variant("empty_rows", ((0, 3), (3, 4), (4,))),
        Variant("empty_inner", ((2, 0), (0, 4), (4,))),
    ]
    return tuple(variants)


_GEMM_VARIANTS = _gemm_variants()

# MatMul is numpy's `matmul`, so the sweep is the shapes that convention distinguishes: a
# rank-1 operand on either side, which is promoted to the row or column that makes the product
# defined and dropped from the result again, and batch axes that stretch against each other or
# are absent from one operand entirely. The zero-element cases cover an empty result and an
# empty contraction, whose sum is over nothing at all.
_MATMUL_VARIANTS = (
    Variant("wide", ((4, 8), (8, 4))),
    Variant("matrix", ((2, 3), (3, 4))),
    Variant("vector_vector", ((3,), (3,))),
    Variant("vector_matrix", ((3,), (3, 4))),
    Variant("matrix_vector", ((2, 3), (3,))),
    Variant("batched", ((2, 3, 4), (2, 4, 3))),
    Variant("batched_vector", ((2, 3, 4), (4,))),
    Variant("vector_batched", ((4,), (2, 4, 3))),
    Variant("unbatched_left", ((2, 3), (2, 3, 4))),
    Variant("unbatched_right", ((2, 3, 4), (4, 3))),
    Variant("both_batches_stretched", ((3, 1, 2, 4), (1, 2, 4, 3))),
    Variant("rank_4", ((1, 2, 3, 4), (1, 2, 4, 3))),
    Variant("empty_rows", ((0, 3), (3, 4))),
    Variant("empty_columns", ((2, 3), (3, 0))),
    Variant("empty_inner", ((2, 0), (0, 4))),
    Variant("empty_batch", ((0, 2, 3), (0, 3, 4))),
    # An empty batch axis stretched against a batch of one, from either side: broadcasting a
    # 1 against a 0 yields 0, which is the one place the rule is not "take the larger".
    Variant("empty_batch_over_one", ((0, 2, 3), (1, 3, 4))),
    Variant("one_batch_over_empty", ((1, 2, 3), (0, 3, 4))),
)

# Det factorizes each of the trailing square matrices. Orders 1 through 4 cover the pivoting —
# the seeded draws put the largest element of a column off the diagonal often enough to swap
# rows — and the batch axes are what carries more than one matrix through the same kernel. A
# matrix of order 0 is square too, and its determinant is the empty product.
_DET_VARIANTS = (
    Variant("order_2", ((2, 2),)),
    Variant("order_3", ((3, 3),)),
    Variant("order_4", ((4, 4),)),
    Variant("order_1", ((1, 1),)),
    Variant("batched", ((3, 2, 2),)),
    Variant("batched_rank_4", ((2, 3, 3, 3),)),
    Variant("empty_batch", ((0, 2, 2),)),
    Variant("order_0", ((2, 0, 0),)),
)

# Einsum's surface is its equation, so the sweep is every reading an equation has and the
# shapes each one addresses: an output stated and one left implicit (where the labels are
# ordered alphabetically rather than as written), a label repeated inside a term at each
# position it can repeat in — which is a diagonal — a label two terms share, which is summed,
# an ellipsis standing for a leading, inner, trailing or absent block of axes, and the spaces
# numpy takes out before reading any of it. The zero-element cases cover an empty result and
# an empty contraction, whose sum is over nothing at all.
#
# The ellipsis broadcasting swept here is the one ONNX's shape inference derives: equal
# ellipsis ranks, an extent of 1 stretching against another operand's. An ellipsis standing
# for a different *number* of axes on two operands is deliberately absent — the pinned `onnx`
# release crashes outright inferring some of those models, so there is no compiling one to
# compare against anything.
_EINSUM_VARIANTS = (
    Variant("matmul", ((2, 3), (3, 4)), {"equation": "ij,jk->ik"}),
    Variant("matmul_implicit", ((2, 3), (3, 4)), {"equation": "ij,jk"}),
    Variant("batch_matmul", ((2, 3, 4), (2, 4, 5)), {"equation": "bij,bjk->bik"}),
    Variant(
        "spaced_terms", ((2, 3, 4), (2, 4, 5)), {"equation": "b i j, b j k -> b i k"}
    ),
    Variant("transpose", ((2, 3),), {"equation": "ij->ji"}),
    Variant("identity_implicit", ((2, 3),), {"equation": "ij"}),
    Variant("reordered_implicit", ((2, 3),), {"equation": "ji"}),
    Variant("summed_axis", ((2, 3),), {"equation": "ij->i"}),
    Variant("summed_all", ((2, 3),), {"equation": "ij->"}),
    Variant("diagonal", ((4, 4),), {"equation": "ii->i"}),
    Variant("trace_implicit", ((4, 4),), {"equation": "ii"}),
    Variant("interleaved_diagonal", ((3, 4, 3),), {"equation": "iji->ij"}),
    Variant("batch_diagonal", ((2, 4, 4),), {"equation": "...ii ->...i"}),
    Variant("inner_product", ((5,), (5,)), {"equation": "i,i"}),
    Variant("outer_product", ((3,), (4,)), {"equation": "i,j->ij"}),
    Variant("hadamard", ((2, 3), (2, 3)), {"equation": "ij,ij->ij"}),
    Variant("scalar", ((),), {"equation": "->"}),
    Variant("scaled_by_scalar", ((2, 3), ()), {"equation": "ij,->ij"}),
    Variant("three_terms", ((2, 3), (3, 4), (4, 5)), {"equation": "ij,jk,kl->il"}),
    Variant(
        "leading_ellipsis", ((2, 3, 4), (2, 4, 5)), {"equation": "...ij,...jk->...ik"}
    ),
    Variant(
        "stretched_ellipsis", ((1, 2, 3), (5, 3, 4)), {"equation": "...ij,...jk->...ik"}
    ),
    Variant("inner_ellipsis", ((2, 3, 4, 5),), {"equation": "i...j->ji"}),
    Variant("summed_ellipsis", ((2, 3),), {"equation": "...i->..."}),
    Variant("dropped_ellipsis", ((2, 3),), {"equation": "...i->i"}),
    Variant("implicit_ellipsis", ((2, 3),), {"equation": "...i"}),
    Variant("stretched_label", ((1, 3), (2, 3)), {"equation": "ij,ij->j"}),
    Variant("empty_contraction", ((2, 0), (0, 3)), {"equation": "ij,jk->ik"}),
    Variant("empty_result", ((0, 3),), {"equation": "ij->ji"}),
    Variant("empty_batch", ((0, 2, 3), (0, 3, 4)), {"equation": "bij,bjk->bik"}),
)

# DFT is one sum per output bin, so the sweep is everything that decides which samples a bin
# runs over and what it writes: each of the four `inverse`/`onesided` combinations — the
# forward transform, the RFFT that keeps the non-redundant half of a real signal's spectrum,
# the inverse, and the IRFFT that mirrors such a half back into a real signal — over real and
# complex operands, at even and odd lengths, with a `dft_length` that leaves the axis alone,
# truncates it or zero-pads it. The axis is swept as the attribute revision 17 reads and as
# the operand revision 20 takes, stated, counted from the end, and left out — the two
# revisions default it differently, which a rank-4 operand tells apart. The transformed axis
# itself is never empty: numpy refuses a transform of no points at all, so there is no oracle
# for one, and an empty batch carries the zero-element case instead.
_DFT_SIGNALS: tuple[tuple[str, tuple[int, ...], Mapping[str, Any]], ...] = (
    ("real", (3, 8, 1), {}),
    ("complex", (3, 8, 2), {}),
    ("inverse", (3, 8, 2), {"inverse": 1}),
    ("rfft", (3, 8, 1), {"onesided": 1}),
    ("irfft", (3, 6, 2), {"inverse": 1, "onesided": 1}),
    ("real_odd", (3, 7, 1), {}),
    ("complex_odd", (3, 7, 2), {}),
    ("rfft_odd", (3, 7, 1), {"onesided": 1}),
    ("irfft_odd", (3, 5, 2), {"inverse": 1, "onesided": 1}),
    ("empty_batch", (0, 8, 1), {}),
    ("rank_4", (2, 3, 4, 1), {}),
)

# What a stated `dft_length` does to an 8-sample axis, against leaving it out.
_DFT_LENGTHS = (("truncated", 5), ("padded", 12), ("same", 8))


def _dft_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(label, (shape,), attributes)
        for label, shape, attributes in _DFT_SIGNALS
    ]
    variants += [
        Variant(
            f"{label}_{combination}",
            ((3, 8, 1 if onesided and not inverse else 2), ()),
            {"inverse": inverse, "onesided": onesided},
            values={1: length},
        )
        for label, length in _DFT_LENGTHS
        for combination, inverse, onesided in (
            ("forward", 0, 0),
            ("rfft", 0, 1),
            ("inverse", 1, 0),
            ("irfft", 1, 1),
        )
    ]
    # The axis as revision 17's attribute and as revision 20's operand, each at the positions
    # ONNX defines: a leading axis, a trailing signal axis, and the same two counted from the
    # end. A rank-2 operand has exactly one axis a transform is defined over, and neither
    # revision's default names it, so it is reached only by stating the axis.
    for label, shape, axis in (
        ("leading", (3, 8, 4, 1), 0),
        ("trailing", (3, 8, 4, 1), 2),
        ("from_the_end", (3, 8, 4, 1), -2),
        ("far_from_the_end", (3, 8, 4, 1), -4),
        ("rank_2", (8, 1), 0),
    ):
        variants.append(
            Variant(f"axis_{label}_attribute", (shape,), {"axis": axis}, versions=(17,))
        )
        variants.append(
            Variant(f"axis_{label}_operand", (shape, None, ()), values={2: axis})
        )
    return tuple(variants)


_DFT_VARIANTS = _dft_variants()

# STFT is a DFT per frame of a window slid along the signal, so the sweep is how the frames
# are laid out — the step against the frame length, from a dense overlap to none at all — and
# where the length comes from: the `frame_length` operand, the window's own extent, or both
# stating it. The window is fed at run time rather than folded in, since its values reach the
# kernel and nothing else about it does. `onesided` is swept stated both ways and left out:
# ONNX's shape inference reads a default of 0 there where the schema declares 1, so the
# omitted case is the one that proves the compiler sizes the result the way the op computes it.
_STFT_VARIANTS = (
    Variant("frame_length", ((2, 32, 1), (), None, ()), values={1: 8, 3: 16}),
    Variant("window", ((2, 32, 1), (), (16,)), values={1: 8}),
    Variant(
        "window_and_frame_length", ((2, 32, 1), (), (16,), ()), values={1: 8, 3: 16}
    ),
    Variant(
        "onesided",
        ((2, 32, 1), (), None, ()),
        {"onesided": 1},
        values={1: 8, 3: 16},
    ),
    Variant(
        "twosided",
        ((2, 32, 1), (), None, ()),
        {"onesided": 0},
        values={1: 8, 3: 16},
    ),
    Variant(
        "complex_signal",
        ((2, 32, 2), (), None, ()),
        {"onesided": 0},
        values={1: 8, 3: 16},
    ),
    Variant("dense_overlap", ((1, 16, 1), (), None, ()), values={1: 1, 3: 8}),
    Variant("no_overlap", ((1, 16, 1), (), None, ()), values={1: 8, 3: 8}),
    Variant("one_frame", ((1, 16, 1), (), None, ()), values={1: 8, 3: 16}),
    Variant("odd_frame_length", ((1, 16, 1), (), None, ()), values={1: 4, 3: 7}),
    Variant("empty_batch", ((0, 32, 1), (), None, ()), values={1: 8, 3: 16}),
)

# Conv slides a filter over the spatial axes of a batch of multi-channel signals, so the sweep
# is its geometry: rank 1 through 3, the attributes that move the window (strides, dilations)
# and the ones that place it (pads, auto_pad), the channel groups that split the operand into
# independent stacks, and the bias.
#
# `auto_pad` is swept only at shapes whose batch and channel extents repeat the spatial ones.
# The reference resolves the mode by reading `X.shape[i]` where the spec reads `X.shape[i+2]`,
# so it computes the padding ONNX defines exactly when those coincide, and is no oracle
# elsewhere. `VALID` goes further: the reference puts it through the same SAME-style formula,
# so the only geometries it agrees with the spec on are the ones where SAME pads nothing
# either -- hence the stride-2 2x2 window here. That the compiler does *not* read `VALID` as
# SAME is settled instead by ONNX's own shape inference, in the kernel tests.
#
# No zero-element variant: the reference raises outright on an empty batch and on a filter
# with no input channels, so neither has an oracle here. That a Conv writing no elements emits
# no code at all is an emission-contract assertion, and lives with the kernel tests.
_CONV_VARIANTS = (
    Variant("spatial_1d", ((2, 2, 7), (3, 2, 3))),
    Variant("spatial_1d_strided", ((2, 2, 7), (3, 2, 3)), {"strides": [2]}),
    Variant("spatial_1d_padded", ((2, 2, 7), (3, 2, 3)), {"pads": [2, 1]}),
    Variant("spatial_1d_dilated", ((2, 2, 7), (3, 2, 3)), {"dilations": [2]}),
    Variant("spatial_2d", ((2, 3, 5, 4), (2, 3, 3, 2))),
    Variant("pads", ((2, 3, 5, 4), (2, 3, 3, 2)), {"pads": [1, 1, 1, 1]}),
    Variant("asymmetric_pads", ((2, 3, 5, 4), (2, 3, 3, 2)), {"pads": [2, 0, 1, 1]}),
    Variant("strides", ((2, 3, 5, 4), (2, 3, 3, 2)), {"strides": [2, 2]}),
    Variant("dilations", ((2, 3, 5, 4), (2, 3, 3, 2)), {"dilations": [2, 1]}),
    Variant(
        "strided_dilated_padded",
        ((2, 3, 5, 4), (2, 3, 3, 2)),
        {"strides": [2, 1], "dilations": [1, 2], "pads": [1, 2, 0, 1]},
    ),
    Variant("bias", ((2, 3, 5, 4), (2, 3, 3, 2), (2,))),
    Variant("unit_window", ((2, 3, 5, 4), (2, 3, 1, 1))),
    # A window wider than the signal, which only the padding makes fit at all.
    Variant("window_wider_than_input", ((1, 1, 2, 2), (1, 1, 3, 3)), {"pads": [2] * 4}),
    Variant("groups", ((2, 4, 5, 4), (6, 2, 3, 2)), {"group": 2}),
    Variant("depthwise", ((2, 4, 5, 4), (4, 1, 3, 2), (4,)), {"group": 4}),
    Variant("spatial_3d", ((1, 2, 4, 3, 3), (2, 2, 2, 2, 2))),
    Variant(
        "spatial_3d_strided_padded",
        ((1, 2, 4, 3, 3), (2, 2, 2, 2, 2)),
        {"strides": [2, 1, 2], "pads": [1, 0, 1, 0, 1, 0]},
    ),
    # An odd total pad, which is what tells the two SAME modes apart: the extra one goes at
    # the end for SAME_UPPER and at the beginning for SAME_LOWER.
    Variant(
        "auto_pad_same_upper",
        ((3, 4, 3, 4), (2, 4, 2, 3)),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_lower",
        ((3, 4, 3, 4), (2, 4, 2, 3)),
        {"auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_upper_unit_stride",
        ((3, 4, 3, 4), (2, 4, 2, 3)),
        {"auto_pad": "SAME_UPPER"},
    ),
    Variant(
        "auto_pad_same_lower_unit_stride",
        ((3, 4, 3, 4), (2, 4, 2, 3)),
        {"auto_pad": "SAME_LOWER"},
    ),
    # SAME pads for the window's *dilated* reach, not its tap count. Shape inference only
    # pins the total -- at stride 2 a whole family of totals rounds to the same result shape
    # -- and says nothing at all about which end the odd one goes to, so the split under
    # dilation is a value question, and only the evaluator answers it. The first axis takes
    # a 2-tap window dilated to a reach of 4 over an extent of 5 at stride 2: three pads,
    # which the two modes divide the other way round from each other.
    Variant(
        "auto_pad_same_upper_dilated",
        ((5, 4, 5, 4), (2, 4, 2, 2)),
        {"auto_pad": "SAME_UPPER", "dilations": [3, 2], "strides": [2, 1]},
    ),
    Variant(
        "auto_pad_same_lower_dilated",
        ((5, 4, 5, 4), (2, 4, 2, 2)),
        {"auto_pad": "SAME_LOWER", "dilations": [3, 2], "strides": [2, 1]},
    ),
    Variant(
        "auto_pad_valid",
        ((4, 4, 4, 4), (2, 4, 2, 2)),
        {"auto_pad": "VALID", "strides": [2, 2]},
    ),
)

# ConvTranspose runs the same window backwards, so the sweep is Conv's geometry again --
# rank, strides, dilations, pads -- plus what only the backward walk has: `output_padding`
# and `output_shape`, which name the result the stride leaves ambiguous, and a filter laid
# out per input channel, (C, M/group, ...).
#
# Two restrictions come from the oracle. The reference resolves an `output_shape` under
# NOTSET by padding nothing and then reading the operand back through a column count it
# derives from that shape instead of from the operand -- which only agree where the padding
# the spec's own equation gives is zero anyway, as it is in the corpus's own `output_shape`
# tests. The shapes here are those. And its grouped path slices `W` by output rather than
# input channels and hands every group the whole bias, so it can only evaluate a group at
# all where each holds exactly one channel of each -- hence the depthwise-shaped `groups`
# variants below, with no bias among them. What the general case computes is settled
# instead by decomposing it into per-group transposed convolutions the reference *can*
# evaluate, in the kernel tests.
_CONV_TRANSPOSE_VARIANTS = (
    Variant("spatial_1d", ((2, 2, 7), (2, 3, 3))),
    Variant("spatial_1d_strided", ((2, 2, 7), (2, 3, 3)), {"strides": [2]}),
    Variant("spatial_1d_padded", ((2, 2, 7), (2, 3, 3)), {"pads": [2, 1]}),
    Variant("spatial_1d_dilated", ((2, 2, 7), (2, 3, 3)), {"dilations": [2]}),
    Variant("spatial_2d", ((2, 3, 5, 4), (3, 2, 3, 2))),
    Variant("pads", ((2, 3, 5, 4), (3, 2, 3, 2)), {"pads": [1, 1, 1, 1]}),
    Variant("asymmetric_pads", ((2, 3, 5, 4), (3, 2, 3, 2)), {"pads": [2, 0, 1, 1]}),
    Variant("strides", ((2, 3, 5, 4), (3, 2, 3, 2)), {"strides": [2, 2]}),
    Variant("dilations", ((2, 3, 5, 4), (3, 2, 3, 2)), {"dilations": [2, 1]}),
    Variant(
        "strided_dilated_padded",
        ((2, 3, 5, 4), (3, 2, 3, 2)),
        {"strides": [3, 2], "dilations": [1, 2], "pads": [1, 2, 0, 1]},
    ),
    Variant("bias", ((2, 3, 5, 4), (3, 2, 3, 2), (2,))),
    Variant("unit_window", ((2, 3, 5, 4), (3, 2, 1, 1))),
    # Padding wider than the operand's own reach, which crops the result to nothing but
    # the overlap.
    Variant("cropping_pads", ((1, 1, 3, 3), (1, 1, 2, 2)), {"pads": [1, 1, 1, 1]}),
    Variant("groups", ((2, 3, 5, 4), (3, 1, 3, 2)), {"group": 3}),
    Variant(
        "groups_strided_padded",
        ((2, 3, 5, 4), (3, 1, 3, 2)),
        {"group": 3, "strides": [2, 1], "pads": [1, 0, 0, 1]},
    ),
    # `output_padding` extends the result past the window's reach: the positions it adds
    # take no tap at all, and carry the bias alone.
    Variant(
        "output_padding",
        ((2, 3, 5, 4), (3, 2, 3, 2), (2,)),
        {"strides": [3, 2], "output_padding": [1, 1]},
    ),
    Variant(
        "output_shape",
        ((1, 1, 3, 3), (1, 2, 3, 3)),
        {"strides": [3, 2], "output_shape": [10, 8]},
    ),
    Variant(
        "output_shape_below_the_reach",
        ((1, 1, 3, 3), (1, 2, 3, 3)),
        {"strides": [3, 2], "output_shape": [9, 7]},
    ),
    Variant(
        "output_shape_with_output_padding",
        ((1, 1, 3, 3), (1, 2, 3, 3)),
        {"strides": [3, 2], "output_shape": [10, 8], "output_padding": [1, 1]},
    ),
    # An odd total pad, which is what tells the two SAME modes apart. Here a 3-tap window
    # at stride 2 leaves one pad over; the modes put it at opposite ends.
    Variant(
        "auto_pad_same_upper",
        ((2, 3, 5, 4), (3, 2, 3, 2)),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_lower",
        ((2, 3, 5, 4), (3, 2, 3, 2)),
        {"auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_upper_dilated",
        ((2, 3, 5, 4), (3, 2, 3, 2)),
        {"auto_pad": "SAME_UPPER", "strides": [3, 2], "dilations": [2, 1]},
    ),
    Variant(
        "auto_pad_same_lower_dilated",
        ((2, 3, 5, 4), (3, 2, 3, 2)),
        {"auto_pad": "SAME_LOWER", "strides": [3, 2], "dilations": [2, 1]},
    ),
    # A SAME mode against an explicit `output_shape`: the mode no longer decides the result,
    # only which end of it the odd pad goes to.
    Variant(
        "auto_pad_same_upper_output_shape",
        ((1, 1, 3, 3), (1, 2, 3, 3)),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2], "output_shape": [6, 6]},
    ),
    Variant(
        "auto_pad_same_lower_output_shape",
        ((1, 1, 3, 3), (1, 2, 3, 3)),
        {"auto_pad": "SAME_LOWER", "strides": [2, 2], "output_shape": [6, 6]},
    ),
    Variant("auto_pad_valid", ((2, 3, 5, 4), (3, 2, 3, 2)), {"auto_pad": "VALID"}),
    Variant("spatial_3d", ((1, 2, 4, 3, 3), (2, 2, 2, 2, 2))),
    Variant(
        "spatial_3d_strided_padded",
        ((1, 2, 4, 3, 3), (2, 2, 2, 2, 2)),
        {"strides": [2, 1, 2], "pads": [1, 0, 1, 0, 1, 0]},
    ),
)

# DeformConv shifts every tap by an offset it reads at run time, so its window lands between
# elements and is interpolated. The sweep is the geometry again — the offsets themselves are
# a seeded draw of ordinary magnitude, which puts sampling points inside the operand, across
# its border and well outside it — plus what only this op has: the `offset_group`s that
# choose which offsets a channel follows, and the `mask` that weights each tap.
#
# Every variant is two-dimensional: the reference evaluator raises outright on any other
# rank, and the compiler emits a kernel for no other rank either, for that reason. The
# `offset` shape is the geometry restated, two coordinates per tap per offset group at every
# result position, and `mask` is the same without the coordinate pair.
_DEFORM_CONV_VARIANTS = (
    Variant("basic", ((1, 1, 4, 4), (1, 1, 2, 2), (1, 8, 3, 3))),
    Variant(
        "pads",
        ((1, 1, 4, 4), (1, 1, 2, 2), (1, 8, 5, 5)),
        {"pads": [1, 1, 1, 1]},
    ),
    Variant(
        "asymmetric_pads",
        ((1, 1, 4, 4), (1, 1, 2, 2), (1, 8, 4, 5)),
        {"pads": [1, 1, 0, 1]},
    ),
    Variant(
        "strides",
        ((2, 1, 5, 4), (3, 1, 2, 3), (2, 12, 2, 2)),
        {"strides": [2, 1]},
    ),
    Variant(
        "dilations",
        ((2, 2, 5, 4), (3, 2, 2, 3), (2, 12, 3, 2)),
        {"dilations": [2, 1]},
    ),
    Variant(
        "strided_dilated_padded",
        ((1, 1, 6, 5), (1, 1, 3, 2), (1, 12, 2, 3), (1,), (1, 6, 2, 3)),
        {"strides": [2, 2], "dilations": [2, 1], "pads": [1, 1, 1, 1]},
    ),
    Variant("bias", ((1, 1, 4, 4), (2, 1, 2, 2), (1, 8, 3, 3), (2,))),
    Variant(
        "mask",
        ((1, 1, 4, 4), (2, 1, 2, 2), (1, 8, 3, 3), (2,), (1, 4, 3, 3)),
    ),
    # A mask without a bias: the operand ONNX puts between them is left out.
    Variant(
        "mask_without_bias",
        ((1, 1, 4, 4), (2, 1, 2, 2), (1, 8, 3, 3), None, (1, 4, 3, 3)),
    ),
    Variant("unit_window", ((1, 1, 4, 4), (1, 1, 1, 1), (1, 2, 4, 4))),
    Variant("groups", ((1, 2, 5, 5), (2, 1, 2, 2), (1, 8, 4, 4)), {"group": 2}),
    Variant(
        "offset_groups",
        ((1, 2, 5, 5), (1, 2, 2, 2), (1, 16, 4, 4)),
        {"offset_group": 2},
    ),
    Variant(
        "groups_and_offset_groups",
        ((1, 4, 5, 5), (2, 2, 2, 2), (1, 16, 4, 4), (2,), (1, 8, 4, 4)),
        {"group": 2, "offset_group": 2},
    ),
    # One offset group per channel, which is the most deformation the op allows.
    Variant(
        "offset_group_per_channel",
        ((1, 4, 5, 5), (2, 2, 2, 2), (1, 32, 4, 4)),
        {"group": 2, "offset_group": 4},
    ),
)


# The quantization family. What the two affine maps have to be swept over is the granularity
# of their scale -- one per tensor, one per slice along the quantization axis, one per block
# of such a slice -- at each revision that has each, and the grid types they round onto,
# which is where saturation lives. The revisions that declare `axis`, `block_size` and
# `output_dtype` are named per variant, since a node cannot carry an attribute a revision
# does not declare. `precision`, which opset 23 added, is swept nowhere: the newest
# QuantizeLinear the reference evaluator implements is revision 21, which refuses a node
# carrying the attribute outright, so there is no oracle for it -- and the compiler refuses
# it for that reason, which the kernel tests assert.
_AXIS_VERSIONS = (19, 21, 23, 24, 25)
_BLOCKED_VERSIONS = (21, 23, 24, 25)

# One scale per slice, a mix of exactly representable ones and ones that are not: a scale of
# 0.75 puts quotients on the halves that the round-to-even rule is the whole of.
_PER_AXIS_SCALES = (
    0.5, 0.25, 2.0, 1.0, 0.125, 4.0, 0.75, 1.5,
    0.0625, 8.0, 3.0, 0.3125, 0.5, 16.0, 0.875, 2.5,
)  # fmt: skip

# QuantizeLinear's scale is pinned rather than drawn wherever the quotient it divides out has
# to stay where the oracle is defined: the reference converts `rint(x / y_scale)` to `int32`
# before it clips, which numpy leaves undefined outside that range, and a drawn scale near
# zero would put it there. `x` is pulled into the same range for the same reason, which is
# what `Domain.CONVERTIBLE` is. The zero point *is* drawn, so every grid's own extremes reach
# the addition that follows -- and that is what saturates the result at either end.
_QUANTIZE_VARIANTS = (
    Variant("per_tensor", ((4, 8), (), ()), values={1: 0.5}),
    # Small enough that every grid's range is left at both ends, uint16's included.
    Variant("saturating", ((4, 8), (), ()), values={1: 1e-4}),
    # Halves throughout, which is the rule that rounds them to the even neighbour.
    Variant("halves", ((4, 8), (), ()), values={0: 1.5, 1: 1.0}),
    Variant("rank_0", ((), (), ()), values={1: 0.5}),
    Variant("empty", ((0, 3), (), ()), values={1: 0.5}),
    Variant("single_element_scale", ((4, 8), (1,), (1,)), values={1: 0.25}),
    # No zero point: the grid is uint8 unless `output_dtype` names another, which is the
    # variant below, so both run at that one type.
    Variant(
        "no_zero_point",
        ((4, 8), ()),
        values={1: 0.5},
        elem_types=(TensorProto.UINT8,),
    ),
    Variant(
        "output_dtype",
        ((4, 8), ()),
        {"output_dtype": TensorProto.INT16},
        values={1: 0.5},
        elem_types=(TensorProto.UINT8,),
        versions=_BLOCKED_VERSIONS,
    ),
    # An `int32` operand, which ONNX allows alongside the float ones and numpy promotes
    # against a float scale. Only at the revisions where `y_scale` is a type of its own:
    # 19 and 21 tie it to the operand's.
    Variant(
        "int32_data",
        ((4, 8), (), ()),
        values={1: 0.5},
        operand_types={0: TensorProto.INT32},
        elem_types=(TensorProto.UINT8,),
        versions=(10, 23, 24, 25),
    ),
    Variant("per_axis", ((4, 16), (16,), (16,)), values={1: _PER_AXIS_SCALES}),
    Variant(
        "per_axis_first",
        ((16, 4), (16,), (16,)),
        {"axis": 0},
        values={1: _PER_AXIS_SCALES},
        versions=_AXIS_VERSIONS,
    ),
    Variant(
        "per_axis_negative",
        ((4, 16), (16,), (16,)),
        {"axis": -1},
        values={1: _PER_AXIS_SCALES},
        versions=_AXIS_VERSIONS,
    ),
    Variant(
        "blocked",
        ((4, 8), (4, 2), (4, 2)),
        {"axis": 1, "block_size": 4},
        values={1: (0.5, 0.25, 2.0, 1.0, 0.125, 4.0, 0.75, 1.5)},
        versions=_BLOCKED_VERSIONS,
    ),
    # A block the axis does not divide, whose last block covers what is left of it.
    Variant(
        "blocked_remainder",
        ((4, 7), (4, 2), (4, 2)),
        {"axis": 1, "block_size": 4},
        values={1: (0.5, 0.25, 2.0, 1.0, 0.125, 4.0, 0.75, 1.5)},
        versions=_BLOCKED_VERSIONS,
    ),
    Variant(
        "blocked_first_axis",
        ((8, 3), (2, 3), (2, 3)),
        {"axis": 0, "block_size": 4},
        values={1: (0.5, 0.25, 2.0, 1.0, 0.125, 4.0)},
        versions=_BLOCKED_VERSIONS,
    ),
)

# DequantizeLinear only multiplies, so nothing it computes can leave the type it computes in:
# its scale is drawn like any other operand, special values and all, and the grid it reads
# sweeps its own extremes. The granularities are QuantizeLinear's, at the revisions that have
# them -- every revision claimed here declares `axis`.
_DEQUANTIZE_VARIANTS = (
    Variant("per_tensor", ((4, 8), (), ())),
    Variant("rank_0", ((), (), ())),
    Variant("empty", ((0, 3), (), ())),
    Variant("single_element_scale", ((4, 8), (1,), (1,))),
    Variant("no_zero_point", ((4, 8), ())),
    Variant("per_axis", ((4, 16), (16,), (16,))),
    Variant("per_axis_first", ((16, 4), (16,), (16,)), {"axis": 0}),
    Variant("per_axis_negative", ((4, 16), (16,), (16,)), {"axis": -1}),
    Variant(
        "blocked",
        ((4, 8), (4, 2), (4, 2)),
        {"axis": 1, "block_size": 4},
        versions=_BLOCKED_VERSIONS,
    ),
    Variant(
        "blocked_remainder",
        ((4, 7), (4, 2), (4, 2)),
        {"axis": 1, "block_size": 4},
        versions=_BLOCKED_VERSIONS,
    ),
    Variant(
        "blocked_first_axis",
        ((8, 3), (2, 3), (2, 3)),
        {"axis": 0, "block_size": 4},
        versions=_BLOCKED_VERSIONS,
    ),
)

# The quantized products walk their operands exactly as MatMul and Conv do, so the shapes
# swept here are those tables' own, thinned to what the walk distinguishes; what is added is
# the zero points, present and absent, and the grids, whose types are free of one another.
_MATMUL_INTEGER_VARIANTS = (
    Variant("matrix", ((2, 3), (3, 4), (), ())),
    Variant("wide", ((4, 8), (8, 4), (), ())),
    Variant("no_zero_points", ((2, 3), (3, 4))),
    Variant("left_zero_point_only", ((2, 3), (3, 4), ())),
    Variant("single_element_zero_points", ((2, 3), (3, 4), (1,), (1,))),
    Variant("batched", ((2, 3, 4), (2, 4, 3), (), ())),
    Variant("unbatched_right", ((2, 3, 4), (4, 3), (), ())),
    Variant("vector_matrix", ((3,), (3, 4), (), ())),
    Variant("matrix_vector", ((2, 3), (3,), (), ())),
    Variant("empty_rows", ((0, 3), (3, 4), (), ())),
    Variant("empty_inner", ((2, 0), (0, 4), (), ())),
    Variant(
        "mixed_grids",
        ((2, 3), (3, 4), (), ()),
        operand_types={1: TensorProto.INT8, 3: TensorProto.INT8},
    ),
)

# The scales of a quantized product are parameters, not data: the one factor the requantization
# comes to is `a_scale * b_scale / y_scale`, and pinning them puts the products it scales where
# both saturation and rounding are reached rather than leaving that to the draw.
# Both ops take them at the same three positions, so one table serves them both.
_PRODUCT_SCALES = {1: 0.02, 4: 0.03, 6: 0.01}

# Every grid the second operand and the result can carry; the first operand's is the type the
# sweep itself ranges over, so the pairs below cover all eight combinations of the three.
_GRID_PAIRS = (
    ("int8_int8", TensorProto.INT8, TensorProto.INT8),
    ("int8_uint8", TensorProto.INT8, TensorProto.UINT8),
    ("uint8_int8", TensorProto.UINT8, TensorProto.INT8),
    ("uint8_uint8", TensorProto.UINT8, TensorProto.UINT8),
)

_QLINEAR_MATMUL_SHAPES = (
    ("matrix", (2, 3), (3, 4)),
    ("wide", (4, 8), (8, 4)),
    ("batched", (2, 3, 4), (2, 4, 3)),
    ("unbatched_right", (2, 3, 4), (4, 3)),
    ("vector_matrix", (3,), (3, 4)),
    ("matrix_vector", (2, 3), (3,)),
    ("empty_rows", (0, 3), (3, 4)),
    ("empty_inner", (2, 0), (0, 4)),
)


def _qlinear_matmul_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(label, (left, (), (), right, (), (), (), ()), values=_PRODUCT_SCALES)
        for label, left, right in _QLINEAR_MATMUL_SHAPES
    ]
    variants += [
        Variant(
            f"grids_{label}",
            ((2, 3), (), (), (3, 4), (), (), (), ()),
            values=_PRODUCT_SCALES,
            operand_types={3: right, 5: right, 7: result},
        )
        for label, right, result in _GRID_PAIRS
    ]
    variants.append(
        Variant(
            "single_element_parameters",
            ((2, 3), (1,), (1,), (3, 4), (1,), (1,), (1,), (1,)),
            values=_PRODUCT_SCALES,
        )
    )
    return tuple(variants)


# The geometry both quantized convolutions are swept over: Conv's own table, thinned to the
# cases the walk distinguishes, and under `auto_pad` restricted to the shapes the reference
# resolves the mode correctly for -- the ones whose batch and channel extents repeat the
# spatial ones, for the reason Conv's table records.
_QUANTIZED_CONV_GEOMETRY: tuple[
    tuple[str, tuple[int, ...], tuple[int, ...], Mapping[str, Any]], ...
] = (
    ("spatial_1d", (2, 2, 7), (3, 2, 3), {}),
    ("spatial_2d", (2, 3, 5, 4), (2, 3, 3, 2), {}),
    ("pads", (2, 3, 5, 4), (2, 3, 3, 2), {"pads": [1, 1, 1, 1]}),
    ("asymmetric_pads", (2, 3, 5, 4), (2, 3, 3, 2), {"pads": [2, 0, 1, 1]}),
    ("strides", (2, 3, 5, 4), (2, 3, 3, 2), {"strides": [2, 2]}),
    ("dilations", (2, 3, 5, 4), (2, 3, 3, 2), {"dilations": [2, 1]}),
    ("groups", (2, 4, 5, 4), (6, 2, 3, 2), {"group": 2}),
    ("depthwise", (2, 4, 5, 4), (4, 1, 3, 2), {"group": 4}),
    ("spatial_3d", (1, 2, 4, 3, 3), (2, 2, 2, 2, 2), {}),
    (
        "auto_pad_same_upper",
        (3, 4, 3, 4),
        (2, 4, 2, 3),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    (
        "auto_pad_same_lower",
        (3, 4, 3, 4),
        (2, 4, 2, 3),
        {"auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
)


def _conv_integer_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(label, (source, weights, (), ()), attributes)
        for label, source, weights, attributes in _QUANTIZED_CONV_GEOMETRY
    ]
    variants += [
        Variant("no_zero_points", ((2, 3, 5, 4), (2, 3, 3, 2))),
        Variant("input_zero_point_only", ((2, 3, 5, 4), (2, 3, 3, 2), ())),
        # One zero point per output channel. The reference stretches a filter's zero point
        # over four axes whatever the operand's rank, so it is an oracle for this at two
        # spatial axes and nowhere else; that the kernel addresses it by output channel at
        # any rank is settled in the kernel tests.
        Variant("per_channel_zero_point", ((2, 3, 5, 4), (2, 3, 3, 2), (), (2,))),
        Variant(
            "mixed_grids",
            ((2, 3, 5, 4), (2, 3, 3, 2), (), ()),
            operand_types={1: TensorProto.INT8, 3: TensorProto.INT8},
        ),
    ]
    return tuple(variants)


def _qlinear_conv_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(
            label,
            (source, (), (), weights, (), (), (), ()),
            attributes,
            values=_PRODUCT_SCALES,
        )
        for label, source, weights, attributes in _QUANTIZED_CONV_GEOMETRY
    ]
    variants += [
        Variant(
            "bias",
            ((2, 3, 5, 4), (), (), (2, 3, 3, 2), (), (), (), (), (2,)),
            values=_PRODUCT_SCALES,
        ),
        # One scale and one zero point per output channel, at the two spatial axes the
        # reference stretches them over.
        Variant(
            "per_channel",
            ((2, 3, 5, 4), (), (), (2, 3, 3, 2), (2,), (2,), (), ()),
            values={**_PRODUCT_SCALES, 4: (0.03, 0.05)},
        ),
        Variant(
            "single_element_parameters",
            ((2, 3, 5, 4), (1,), (1,), (2, 3, 3, 2), (1,), (1,), (1,), (1,)),
            values=_PRODUCT_SCALES,
        ),
    ]
    variants += [
        Variant(
            f"grids_{label}",
            ((2, 3, 5, 4), (), (), (2, 3, 3, 2), (), (), (), ()),
            values=_PRODUCT_SCALES,
            operand_types={3: weights, 5: weights, 7: result},
        )
        for label, weights, result in _GRID_PAIRS
    ]
    return tuple(variants)


# A pooling slides the same window as a convolution, so the sweep is that geometry again --
# rank 1 through 3, the attributes that move the window and the ones that place it -- plus
# what only a pooling has: `ceil_mode`, which lets the last window hang off the end, and (for
# AveragePool) `count_include_pad`, which decides whether the padded positions are part of the
# divisor. The window is deliberately never wider than what the padding fills, since a window
# covering nothing at all averages over an empty array, which the reference raises on.
#
# Two restrictions come from the oracle, both about `auto_pad`. The reference resolves it
# through the *undilated* kernel, so the pads it derives are the ones ONNX defines only where
# the dilations are 1; and it refuses `ceil_mode` alongside it outright. What the compiler
# derives for the dilated case is settled instead by ONNX's own shape inference, in the kernel
# tests: a result shape the pads do not imply fails to compile at all.
_AVERAGE_POOL_VARIANTS = (
    Variant("spatial_1d", ((2, 2, 7),), {"kernel_shape": [3]}),
    Variant("spatial_1d_strided", ((2, 2, 7),), {"kernel_shape": [3], "strides": [2]}),
    Variant("spatial_1d_padded", ((2, 2, 7),), {"kernel_shape": [3], "pads": [2, 1]}),
    Variant(
        "spatial_1d_dilated", ((2, 2, 7),), {"kernel_shape": [3], "dilations": [2]}
    ),
    Variant("spatial_2d", ((2, 3, 5, 4),), {"kernel_shape": [3, 2]}),
    Variant("pads", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "pads": [1, 1, 1, 1]}),
    Variant(
        "asymmetric_pads",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [2, 0, 1, 1]},
    ),
    Variant("strides", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "strides": [2, 2]}),
    Variant(
        "dilations", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "dilations": [2, 1]}
    ),
    Variant(
        "strided_dilated_padded",
        ((2, 3, 5, 4),),
        {
            "kernel_shape": [3, 2],
            "strides": [2, 1],
            "dilations": [1, 2],
            "pads": [1, 2, 0, 1],
        },
    ),
    # The padded positions counted, and not: the only thing that tells the two apart is a
    # window the padding reaches into, so every one of these pads.
    Variant(
        "count_include_pad",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [1, 1, 1, 1], "count_include_pad": 1},
    ),
    Variant(
        "count_include_pad_asymmetric",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [2, 0, 1, 1], "count_include_pad": 1},
    ),
    Variant(
        "count_include_pad_strided",
        ((2, 3, 5, 4),),
        {
            "kernel_shape": [3, 2],
            "pads": [1, 1, 1, 1],
            "strides": [2, 2],
            "count_include_pad": 1,
        },
    ),
    # `ceil_mode` keeps a last window the floor would have dropped; the taps it reaches past
    # the operand's own padding are read by nothing.
    Variant(
        "ceil_mode",
        ((1, 1, 4, 4),),
        {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1},
    ),
    Variant(
        "ceil_mode_dilated",
        ((1, 1, 4, 4),),
        {
            "kernel_shape": [2, 2],
            "strides": [1, 1],
            "dilations": [2, 2],
            "ceil_mode": 1,
        },
    ),
    # The position ONNX drops again: rounding up puts the last window's own start beyond the
    # padding, where it would cover nothing but pad.
    Variant(
        "ceil_mode_last_window_starts_on_pad",
        ((1, 3, 2, 2),),
        {
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [3, 3],
            "ceil_mode": 1,
            "count_include_pad": 1,
        },
    ),
    Variant("unit_window", ((2, 3, 5, 4),), {"kernel_shape": [1, 1]}),
    Variant("spatial_3d", ((1, 2, 4, 3, 3),), {"kernel_shape": [2, 2, 2]}),
    Variant(
        "spatial_3d_strided_padded",
        ((1, 2, 4, 3, 3),),
        {
            "kernel_shape": [2, 2, 2],
            "strides": [2, 1, 2],
            "pads": [1, 0, 1, 0, 1, 0],
        },
    ),
    # An odd total pad, which is what tells the two SAME modes apart: the extra one goes at
    # the end for SAME_UPPER and at the beginning for SAME_LOWER.
    Variant(
        "auto_pad_same_upper",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_lower",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_upper_unit_stride",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_UPPER"},
    ),
    Variant(
        "auto_pad_same_lower_unit_stride",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_LOWER"},
    ),
    Variant(
        "auto_pad_valid",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "VALID", "strides": [2, 2]},
    ),
    Variant("empty_batch", ((0, 2, 5, 4),), {"kernel_shape": [2, 2]}),
)

# MaxPool is the same geometry, folded by comparison instead. The reference evaluator carries
# two implementations of it and picks between them by the attributes: a general one whenever
# any stride or dilation is other than 1, and — at unit strides and dilations — a padding-based
# one that re-pads the operand itself. That second path is no oracle for much: it unpacks 2-D
# `pads` in the wrong order, ignores them outright at any other rank, rounds `ceil_mode` up
# without dropping the window that then starts beyond the padding, and derives the `Indices`
# output through the wrong extents. So everything but the plainest unit-stride window carries a
# stride or a dilation, which is what puts the case on the general path.
#
# `auto_pad` narrows it once more: the general path reads SAME_LOWER as flooring the result and
# then splits the pad the SAME_UPPER way, which is neither of the two modes ONNX defines. The
# corpus's own `same_lower` tests are what vouch for that mode.
_MAX_POOL_VARIANTS = (
    # The one unit-stride, unit-dilation case, on the second path — and floating-point only,
    # since that path pads the operand with a NaN no integer dtype can hold.
    Variant(
        "spatial_2d",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2]},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant("spatial_1d_strided", ((2, 2, 7),), {"kernel_shape": [3], "strides": [2]}),
    Variant(
        "spatial_1d_padded",
        ((2, 2, 7),),
        {"kernel_shape": [3], "pads": [2, 1], "strides": [2]},
    ),
    Variant(
        "spatial_1d_dilated", ((2, 2, 7),), {"kernel_shape": [3], "dilations": [2]}
    ),
    Variant("strides", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "strides": [2, 2]}),
    Variant(
        "pads",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [1, 1, 1, 1], "strides": [2, 2]},
    ),
    Variant(
        "asymmetric_pads",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [2, 0, 1, 1], "strides": [2, 2]},
    ),
    Variant(
        "dilations", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "dilations": [2, 1]}
    ),
    Variant(
        "strided_dilated_padded",
        ((2, 3, 5, 4),),
        {
            "kernel_shape": [3, 2],
            "strides": [2, 1],
            "dilations": [1, 2],
            "pads": [1, 2, 0, 1],
        },
    ),
    Variant(
        "ceil_mode",
        ((1, 1, 4, 4),),
        {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1},
    ),
    Variant(
        "ceil_mode_last_window_starts_on_pad",
        ((1, 3, 2, 2),),
        {
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [3, 3],
            "ceil_mode": 1,
        },
    ),
    Variant(
        "unit_window", ((2, 3, 5, 4),), {"kernel_shape": [1, 1], "strides": [2, 2]}
    ),
    Variant(
        "spatial_3d",
        ((1, 2, 4, 3, 3),),
        {"kernel_shape": [2, 2, 2], "strides": [2, 1, 2]},
    ),
    Variant(
        "auto_pad_same_upper",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_valid",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "VALID", "strides": [2, 2]},
    ),
    Variant(
        "empty_batch", ((0, 2, 5, 4),), {"kernel_shape": [2, 2], "strides": [2, 2]}
    ),
    # `Indices` reports where each maximum was read as one flat index into the whole operand,
    # which `storage_order` lays out row-major or column-major.
    Variant(
        "indices",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "strides": [2, 2]},
        outputs=2,
    ),
    Variant(
        "indices_column_major",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "strides": [2, 2], "storage_order": 1},
        outputs=2,
    ),
    Variant(
        "indices_padded",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [1, 1, 1, 1], "strides": [2, 2]},
        outputs=2,
    ),
    Variant(
        "indices_1d",
        ((2, 2, 7),),
        {"kernel_shape": [3], "strides": [2]},
        outputs=2,
    ),
    Variant(
        "indices_3d_column_major",
        ((1, 2, 4, 3, 3),),
        {"kernel_shape": [2, 2, 2], "strides": [2, 1, 2], "storage_order": 1},
        outputs=2,
    ),
)

# LpPool is the same window under a norm. `ceil_mode` is left out of its sweep: for a window
# that rounding up clipped, the reference averages over the taps the window really holds and
# scales the result back up by the whole kernel's tap count, which its own source records as a
# computation borrowed from elsewhere that differs from the spec's -- and which the corpus's
# stored LpPool outputs, taken as the plain norm, disagree with. Only rounding up can clip a
# window, so everything below is geometry the two agree on.
_LP_POOL_VARIANTS = (
    Variant("spatial_1d", ((2, 2, 7),), {"kernel_shape": [3], "p": 3}),
    Variant("spatial_2d", ((2, 3, 5, 4),), {"kernel_shape": [3, 2]}),
    Variant("order_1", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "p": 1}),
    Variant("order_3", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "p": 3}),
    Variant(
        "pads", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "pads": [1, 1, 1, 1], "p": 3}
    ),
    Variant(
        "asymmetric_pads",
        ((2, 3, 5, 4),),
        {"kernel_shape": [3, 2], "pads": [2, 0, 1, 1]},
    ),
    Variant("strides", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "strides": [2, 2]}),
    Variant(
        "dilations", ((2, 3, 5, 4),), {"kernel_shape": [3, 2], "dilations": [2, 1]}
    ),
    Variant(
        "strided_dilated_padded",
        ((2, 3, 5, 4),),
        {
            "kernel_shape": [3, 2],
            "strides": [2, 1],
            "dilations": [1, 2],
            "pads": [1, 2, 0, 1],
            "p": 3,
        },
    ),
    Variant("unit_window", ((2, 3, 5, 4),), {"kernel_shape": [1, 1]}),
    Variant("spatial_3d", ((1, 2, 4, 3, 3),), {"kernel_shape": [2, 2, 2], "p": 1}),
    Variant(
        "auto_pad_same_upper",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_same_lower",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
    Variant(
        "auto_pad_valid",
        ((2, 3, 5, 4),),
        {"kernel_shape": [2, 3], "auto_pad": "VALID", "strides": [2, 2]},
    ),
    Variant("empty_batch", ((0, 2, 5, 4),), {"kernel_shape": [2, 2]}),
)

# The global poolings take one window the size of the operand's spatial extent, so there is
# no geometry to sweep — only the rank they take it over, and an operand with nothing in it.
# The mean of an empty batch is one the reference cannot take: it divides by a count it reads
# off the array, which is why only the two folds below carry that variant.
_GLOBAL_VARIANTS = (
    Variant("spatial_1d", ((2, 3, 7),)),
    Variant("spatial_2d", ((2, 3, 5, 4),)),
    Variant("spatial_3d", ((1, 2, 4, 3, 3),)),
    Variant("single_position", ((2, 3, 1, 1),)),
)
# GlobalMaxPool and GlobalLpPool are compared against the windowed poolings they are defined
# to equal, which fold over the taps of a window rather than over an array — so the empty
# batch has an oracle here.
_GLOBAL_MAX_VARIANTS = (*_GLOBAL_VARIANTS, Variant("empty_batch", ((0, 2, 5, 4),)))
# The order of the norm is worth sweeping for GlobalLpPool, as it is for LpPool itself.
_GLOBAL_LP_VARIANTS = (
    *_GLOBAL_MAX_VARIANTS,
    Variant("order_1", ((2, 3, 5, 4),), {"p": 1}),
    Variant("order_3", ((2, 3, 5, 4),), {"p": 3}),
)


def _as_windowed_pooling(op_type: str) -> Callable[[Case], ModelProto]:
    """The case's `Global*` pooling written as the windowed op its schema defines it to equal.

    Both schemas say it in the same words — "equivalent to [MaxPool/LpPool] with kernel size
    equal to the spatial dimension of input tensor" — so the equivalent is that op over one
    window covering the operand's whole spatial extent.
    """

    def build(case: Case) -> ModelProto:
        spatial = (case.variant.shapes[0] or ())[2:]
        return _model(
            replace(
                case,
                op_type=op_type,
                variant=replace(
                    case.variant,
                    attributes={
                        **case.variant.attributes,
                        "kernel_shape": list(spatial),
                    },
                ),
            )
        )

    return build


# MaxUnpool scatters rather than folds, so every special value its dtype has goes in and has
# to come back out unchanged. Its indices are the op's own addressing — a draw would land
# outside the result, which ONNX leaves undefined and the artifact reports as an error status
# — so every variant pins them, including one that names the same position twice, where the
# last value written is what stays.
_MAX_UNPOOL_VARIANTS = (
    Variant(
        "basic",
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        values={1: (0, 3, 12, 15)},
    ),
    Variant(
        "wide",
        ((1, 1, 4, 4), (1, 1, 4, 4)),
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        values={1: tuple(range(0, 64, 4))},
    ),
    Variant(
        "duplicate_positions",
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        values={1: (5, 5, 5, 5)},
    ),
    Variant(
        "channels",
        ((2, 3, 2, 2), (2, 3, 2, 2)),
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        values={1: tuple(range(0, 96, 4))},
    ),
    Variant(
        "padded",
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        values={1: (0, 1, 2, 3)},
    ),
    Variant(
        "unit_stride",
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        {"kernel_shape": [2, 2], "strides": [1, 1]},
        values={1: (0, 2, 6, 8)},
    ),
    Variant(
        "spatial_1d",
        ((1, 1, 3), (1, 1, 3)),
        {"kernel_shape": [2], "strides": [2]},
        values={1: (0, 3, 5)},
    ),
    Variant(
        "spatial_3d",
        ((1, 1, 2, 2, 2), (1, 1, 2, 2, 2)),
        {"kernel_shape": [2, 2, 2], "strides": [2, 2, 2]},
        values={1: (0, 9, 18, 27, 36, 45, 54, 63)},
    ),
)


# The reductions sweep both of ONNX's axes conventions: the attribute the family started
# with, and the input it moved to (ReduceSum at 13, the rest at 18). A variant is generated
# only at the revisions that take the form it declares. Axes are the op's configuration, so
# the input convention pins them — which is also what makes them compile-time constant, the
# only form the compiler can specialize a shape to.
_ATTRIBUTE_AXES_VARIANTS = (
    Variant("axis_1", ((2, 3, 4),), {"axes": [1]}),
    Variant("axes_0_2_no_keepdims", ((2, 3, 4),), {"axes": [0, 2], "keepdims": 0}),
    Variant("all_axes", ((2, 3, 4),)),
    Variant("rank_1", ((5,),), {"axes": [0], "keepdims": 0}),
    Variant("empty_result", ((0, 3),), {"axes": [1]}),
)
_INPUT_AXES_VARIANTS = (
    Variant("input_axis_1", ((2, 3, 4), (1,)), values={1: 1}),
    Variant(
        "input_axes_0_2_no_keepdims",
        ((2, 3, 4), (2,)),
        {"keepdims": 0},
        values={1: (0, 2)},
    ),
    Variant("input_negative_axis", ((2, 3, 4), (1,)), values={1: -1}),
    Variant("input_absent_axes", ((2, 3, 4),)),
    Variant("input_empty_axes", ((2, 3, 4), (0,)), values={1: 0}),
    Variant(
        "input_noop_empty_axes",
        ((2, 3, 4), (0,)),
        {"noop_with_empty_axes": 1},
        values={1: 0},
    ),
    Variant("input_rank_1", ((5,), (1,)), {"keepdims": 0}, values={1: 0}),
    Variant("input_empty_result", ((0, 3), (1,)), values={1: 1}),
)
# Reducing a group with no elements at all, which every reduction answers with its identity.
_EMPTY_GROUP_VARIANTS = (
    Variant("empty_group", ((0, 3),), {"axes": [0]}),
    Variant("input_empty_group", ((0, 3), (1,)), values={1: 0}),
)

# The reference fills a reduction over no elements from ±inf, which `np.full(..., bool)`
# turns into True for a minimum as much as for a maximum — not a value to compare a kernel
# against, so the boolean families sit that one variant out.
_NON_BOOL_TYPES = tuple(
    elem_type for elem_type in sorted(C_TYPES) if elem_type != TensorProto.BOOL
)


# The revisions of each convention: ReduceSum moved `axes` to an input at 13 and the rest of
# the family at 18, with ReduceMax and ReduceMin picking up the int8 families at 12 and the
# boolean ones at 20.
_REDUCTION_VERSIONS = ((1, 11, 13), (18,))
_SUM_VERSIONS = ((1, 11), (13,))
_EXTREMUM_VERSIONS = ((1, 11, 12, 13), (18, 20))


def _reduction_sweep(
    kind: Kind,
    attribute_versions: tuple[int, ...],
    input_versions: tuple[int, ...],
    *,
    elem_types: tuple[int, ...] | None = None,
    empty_group_types: tuple[int, ...] | None = None,
    factors: bool = False,
) -> Sweep:
    """One reduction's shape and axes family, in both of ONNX's conventions.

    `elem_types` narrows the whole sweep and `empty_group_types` only the variant reducing a
    group with no elements — each where the reference evaluator stops being an oracle.
    `factors` marks a reduction that multiplies its group rather than adding it up.
    """
    empty_group_types = empty_group_types if empty_group_types else elem_types
    attribute_empty, given_empty = _EMPTY_GROUP_VARIANTS
    return Sweep(
        kind,
        (
            *(
                replace(variant, versions=attribute_versions, elem_types=elem_types)
                for variant in _ATTRIBUTE_AXES_VARIANTS
            ),
            *(
                replace(variant, versions=input_versions, elem_types=elem_types)
                for variant in _INPUT_AXES_VARIANTS
            ),
            replace(
                attribute_empty,
                versions=attribute_versions,
                elem_types=empty_group_types,
            ),
            replace(given_empty, versions=input_versions, elem_types=empty_group_types),
        ),
        operand_domains={0: Domain.SMALL_FACTOR} if factors else {},
        operand_types={1: TensorProto.INT64},
        constant_operands=(1,),
    )


# ArgMax and ArgMin take their axis as an attribute throughout; negative axes arrived at
# opset 11 and `select_last_index` at 12.
_ARG_VARIANTS = (
    Variant("axis_0", ((2, 3, 4),), {"axis": 0}),
    Variant("axis_1_no_keepdims", ((2, 3, 4),), {"axis": 1, "keepdims": 0}),
    Variant("rank_1", ((5,),), {"axis": 0}),
    Variant("empty_result", ((0, 3),), {"axis": 1}),
    Variant("negative_axis", ((2, 3, 4),), {"axis": -1}, versions=(11, 12, 13)),
    Variant(
        "select_last",
        ((2, 3, 4),),
        {"axis": 1, "select_last_index": 1},
        versions=(12, 13),
    ),
    Variant(
        "select_last_negative_axis",
        ((2, 3, 4),),
        {"axis": -1, "select_last_index": 1},
        versions=(12, 13),
    ),
)

# Softmax, LogSoftmax and Hardmax normalize along one axis, leaving the shape alone.
_ALONG_AXIS_VARIANTS = (
    Variant("axis_0", ((2, 3, 4),), {"axis": 0}),
    Variant("axis_1", ((2, 3, 4),), {"axis": 1}),
    Variant("default_axis", ((2, 3, 4),)),
    Variant("negative_axis", ((2, 3, 4),), {"axis": -2}),
    Variant("rank_1", ((5,),), {"axis": 0}),
    Variant("empty_groups", ((0, 3),), {"axis": 1}),
    Variant("empty_group", ((0, 3),), {"axis": 0}),
)

# The cumulative folds take their axis as an operand, which models — the backend corpus among
# them — pass at run time, so the sweep feeds it rather than pinning it into the model.
_CUMULATIVE_VARIANTS = (
    Variant("axis_0", ((2, 3, 4), ()), values={1: 0}),
    Variant("axis_1_exclusive", ((2, 3, 4), ()), {"exclusive": 1}, values={1: 1}),
    Variant("axis_2_reverse", ((2, 3, 4), ()), {"reverse": 1}, values={1: 2}),
    Variant(
        "negative_axis_exclusive_reverse",
        ((2, 3, 4), ()),
        {"exclusive": 1, "reverse": 1},
        values={1: -1},
    ),
    Variant("rank_1", ((5,), ()), values={1: 0}),
    Variant("empty", ((0, 3), ()), values={1: 1}),
)


# The normalizations. Each reduces a group of elements to a mean and a variance, so all of
# them carry ACCUMULATING: the summation order the spec leaves open is what a special value
# would make the comparison depend on.
#
# BatchNormalization's five operands are the data and four per-channel vectors; the variance
# it is handed at inference goes under a square root, so it is drawn non-negative. Training
# mode reduces every axis but the channel and reports the running statistics, which the
# `outputs` variants ask for.
_BATCH_DATA = (2, 3, 4, 5)
_BATCH_PARAMETER = (3,)


def _batch_variants() -> tuple[Variant, ...]:
    def shapes(data: tuple[int, ...], channels: int) -> tuple[tuple[int, ...], ...]:
        return (data, *((channels,),) * 4)

    ranks = (
        ("spatial_2d", _BATCH_DATA, 3),
        ("spatial_1d", (2, 3, 4), 3),
        ("no_spatial", (2, 3), 3),
        # ONNX reads a tensor of rank 1 as a single channel of N instances.
        ("rank_1", (4,), 1),
        ("empty_instances", (0, 3, 2), 3),
        ("empty_channels", (2, 0, 3), 0),
        ("empty_spatial", (2, 3, 0), 3),
    )
    variants = [
        Variant(label, shapes(data, channels)) for label, data, channels in ranks
    ]
    variants += [
        Variant("epsilon", shapes(_BATCH_DATA, 3), {"epsilon": 0.01}),
        # ONNX's own shape inference refuses a training-mode node with anything but the
        # three outputs, so every one of these reports the running statistics.
        Variant("training", shapes(_BATCH_DATA, 3), {"training_mode": 1}, outputs=3),
        Variant(
            "training_momentum",
            shapes(_BATCH_DATA, 3),
            {"training_mode": 1, "momentum": 0.25, "epsilon": 0.01},
            outputs=3,
        ),
        Variant(
            "training_empty_group",
            shapes((0, 3, 2), 3),
            {"training_mode": 1},
            outputs=3,
        ),
        Variant(
            "training_empty_channels",
            shapes((2, 0, 3), 0),
            {"training_mode": 1},
            outputs=3,
        ),
    ]
    return tuple(variants)


# LayerNormalization standardizes every row from `axis` on. The reference evaluator computes
# stage one in the tensor's own dtype and reports the mean and inverse deviation in it too,
# while ONNX derives both from `stash_type` — the two coincide at float32, which is therefore
# the only element type it is an oracle for here.
_LAYER_DATA = (2, 3, 4, 5)


def _layer_variants() -> tuple[Variant, ...]:
    variants = []
    for axis in (0, 1, 2, 3, -1, -2, -4):
        normalized = _LAYER_DATA[axis if axis >= 0 else axis + len(_LAYER_DATA) :]
        label = f"axis_{axis}".replace("-", "negative_")
        variants.append(
            Variant(label, (_LAYER_DATA, normalized, normalized), {"axis": axis})
        )
    trailing = _LAYER_DATA[-1:]
    variants += [
        Variant("default_axis", (_LAYER_DATA, trailing, trailing)),
        Variant("no_bias", (_LAYER_DATA, trailing)),
        Variant("epsilon", (_LAYER_DATA, trailing, trailing), {"epsilon": 0.01}),
        Variant("stash_float", (_LAYER_DATA, trailing, trailing), {"stash_type": 1}),
        Variant("statistics", (_LAYER_DATA, trailing, trailing), outputs=3),
        Variant("statistics_no_bias", (_LAYER_DATA, trailing), outputs=3),
        Variant("statistics_mean_only", (_LAYER_DATA, trailing, trailing), outputs=2),
        # ONNX applies the scale and bias by broadcasting, so they need not carry the
        # normalized shape at all.
        Variant("broadcast_scale", (_LAYER_DATA, (), ()), {"axis": 3}),
        Variant("rank_1", ((5,), (5,), (5,)), {"axis": 0}, outputs=3),
        Variant("empty_groups", ((0, 3), (3,), (3,)), {"axis": 1}, outputs=3),
        Variant("empty_group", ((2, 0), (0,), (0,)), {"axis": 1}, outputs=3),
    ]
    return tuple(
        replace(variant, elem_types=(TensorProto.FLOAT,)) for variant in variants
    )


# InstanceNormalization standardizes each channel of each instance over its spatial axes,
# and GroupNormalization each group of channels; both take a per-channel scale and bias.
_INSTANCE_VARIANTS = (
    Variant("spatial_2d", ((2, 3, 4, 5), (3,), (3,))),
    Variant("spatial_1d", ((2, 3, 4), (3,), (3,))),
    Variant("no_spatial", ((2, 3), (3,), (3,))),
    Variant("epsilon", ((2, 3, 4, 5), (3,), (3,)), {"epsilon": 0.01}),
    Variant("empty_instances", ((0, 3, 2), (3,), (3,))),
    Variant("empty_channels", ((2, 0, 3), (0,), (0,))),
    Variant("empty_spatial", ((2, 3, 0), (3,), (3,))),
)

_GROUP_DATA = (3, 4, 2, 2)
_GROUP_VARIANTS = (
    Variant("one_group", (_GROUP_DATA, (4,), (4,)), {"num_groups": 1}),
    Variant("two_groups", (_GROUP_DATA, (4,), (4,)), {"num_groups": 2}),
    # A group per channel, where the op is InstanceNormalization.
    Variant("channel_groups", (_GROUP_DATA, (4,), (4,)), {"num_groups": 4}),
    Variant("epsilon", (_GROUP_DATA, (4,), (4,)), {"num_groups": 2, "epsilon": 0.01}),
    Variant(
        "stash_float", (_GROUP_DATA, (4,), (4,)), {"num_groups": 2, "stash_type": 1}
    ),
    Variant("spatial_1d", ((2, 4, 3), (4,), (4,)), {"num_groups": 2}),
    Variant("no_spatial", ((2, 4), (4,), (4,)), {"num_groups": 2}),
    # No zero-element variant: the reference evaluates GroupNormalization's function body,
    # whose `Reshape` to [0, 0, -1] numpy refuses outright on an empty tensor, so there is
    # no oracle for one. The loops it would exercise are the ones the other members of this
    # family carry empty variants for.
)


# RMSNormalization scales each row from `axis` on by the reciprocal root of its own mean
# square. `stash_type` is swept at its default alone: the reference refuses every other value
# outright, so there would be no oracle for one.
_RMS_DATA = (2, 3, 4, 5)


def _rms_variants() -> tuple[Variant, ...]:
    variants = []
    for axis in (0, 1, 2, 3, -1, -2, -4):
        normalized = _RMS_DATA[axis if axis >= 0 else axis + len(_RMS_DATA) :]
        label = f"axis_{axis}".replace("-", "negative_")
        variants.append(Variant(label, (_RMS_DATA, normalized), {"axis": axis}))
    trailing = _RMS_DATA[-1:]
    return (
        *variants,
        Variant("default_axis", (_RMS_DATA, trailing)),
        Variant("epsilon", (_RMS_DATA, trailing), {"epsilon": 0.01}),
        Variant("stash_float", (_RMS_DATA, trailing), {"stash_type": 1}),
        # ONNX applies the scale by broadcasting, so it need not carry the normalized shape.
        Variant("broadcast_scale", (_RMS_DATA, ()), {"axis": 3}),
        Variant("rank_1", ((5,), (5,)), {"axis": 0}),
        Variant("empty_groups", ((0, 3), (3,)), {"axis": 1}),
        Variant("empty_group", ((2, 0), (0,)), {"axis": 1}),
    )


# SoftmaxCrossEntropyLoss reads its logits as instances by classes by any further axes. The
# labels are a parameter rather than data — one outside the class axis is a read ONNX leaves
# undefined and the reference raises on — so every variant pins them, the `ignore_index` ones
# to values that exercise both the skipped and the counted branch. The weights are drawn
# non-negative so that the weighted mean's denominator cannot land on the zero that would make
# every expectation a NaN.
_SCE_2D = ((3, 5), (3,))
_SCE_3D = ((3, 5, 2), (3, 2))
_SCE_4D = ((2, 3, 2, 2), (2, 2, 2))
_SCE_WEIGHTS = (5,)
_SCE_LABELS = {
    _SCE_2D: (0, 4, 2),
    _SCE_3D: (0, 1, 2, 3, 4, 0),
    _SCE_4D: (0, 1, 2, 0, 2, 1, 0, 1),
}
_SCE_IGNORED = (0, -1, 2)


def _sce_variants() -> tuple[Variant, ...]:
    def shaped(
        label: str,
        shapes: tuple[tuple[int, ...], ...],
        attributes: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> Variant:
        return Variant(
            label,
            shapes,
            attributes or {},
            values={1: _SCE_LABELS[(shapes[0], shapes[1])]},
            domains={2: Domain.NONNEGATIVE},
            **fields,
        )

    variants = [
        shaped(f"{reduction}_{rank}d", shapes, {"reduction": reduction})
        for reduction in ("mean", "sum", "none")
        for rank, shapes in ((2, _SCE_2D), (3, _SCE_3D), (4, _SCE_4D))
    ]
    weighted = (*_SCE_2D, _SCE_WEIGHTS)
    variants += [
        shaped("default_reduction", _SCE_2D),
        shaped("weighted_mean", weighted, {"reduction": "mean"}),
        shaped("weighted_sum", weighted, {"reduction": "sum"}),
        shaped("weighted_none", weighted, {"reduction": "none"}),
        shaped("weighted_3d", (*_SCE_3D, _SCE_WEIGHTS)),
        shaped("log_prob", _SCE_2D, outputs=2),
        shaped("log_prob_weighted", weighted, outputs=2),
        shaped("log_prob_none", _SCE_2D, {"reduction": "none"}, outputs=2),
        shaped("int32_labels", _SCE_2D, operand_types={1: TensorProto.INT32}),
        # No zero-instance variant: the reference reshapes its log-softmax to `(N, C, -1)`,
        # which numpy refuses on an empty array — it cannot infer the free axis from no
        # elements at all — so there is no oracle for one. The kernel's own empty case is
        # the `ignored_all` variant below, where nothing reaches either fold.
    ]
    variants += [
        Variant(
            f"ignored_{label}",
            shapes,
            {"ignore_index": -1, **(attributes or {})},
            values={1: labels},
            domains={2: Domain.NONNEGATIVE},
            outputs=outputs,
        )
        for label, shapes, labels, attributes, outputs in (
            ("mean", _SCE_2D, _SCE_IGNORED, None, 1),
            ("sum", _SCE_2D, _SCE_IGNORED, {"reduction": "sum"}, 1),
            ("none", _SCE_2D, _SCE_IGNORED, {"reduction": "none"}, 1),
            ("weighted", weighted, _SCE_IGNORED, None, 1),
            ("log_prob", _SCE_2D, _SCE_IGNORED, None, 2),
            ("3d", _SCE_3D, (0, -1, 2, -1, 4, 0), None, 1),
            # Every entry skipped: the weighted mean divides zero by zero.
            ("all", _SCE_2D, (-1, -1, -1), None, 1),
        )
    ]
    return tuple(variants)


# LpNormalization divides each row along one axis by its own norm. The reference raises the
# elements to the power `p` without taking their absolute value, so at `p` = 1 it computes a
# signed sum rather than the norm ONNX defines: those variants feed non-negative operands,
# where the two agree. The zero-filled variants are the 0/0 the op answers with zero.
def _lp_variants() -> tuple[Variant, ...]:
    shapes = (
        ("axis_0", (2, 3, 4), 0),
        ("axis_1", (2, 3, 4), 1),
        ("axis_negative_1", (2, 3, 4), -1),
        ("rank_1", (5,), 0),
        ("empty_rows", (0, 3), 1),
        ("empty_group", (2, 0), 1),
    )
    variants = [
        Variant(f"l2_{label}", (shape,), {"p": 2, "axis": axis})
        for label, shape, axis in shapes
    ]
    variants += [
        Variant(
            f"l1_{label}",
            (shape,),
            {"p": 1, "axis": axis},
            domains={0: Domain.NONNEGATIVE},
        )
        for label, shape, axis in shapes
    ]
    variants += [
        Variant("l2_default", ((2, 3, 4),)),
        Variant("l2_zeros", ((2, 3, 4),), {"p": 2, "axis": 1}, values={0: 0.0}),
        Variant("l1_zeros", ((2, 3, 4),), {"p": 1, "axis": 1}, values={0: 0.0}),
    ]
    return tuple(variants)


# MeanVarianceNormalization is defined as an ONNX function, and the reference evaluates that
# body — whose epsilon constant is a float32, which makes the body itself ill-typed for a
# float64 tensor. float32 is therefore the only element type it can vouch for.
def _mvn_variants() -> tuple[Variant, ...]:
    variants = [
        Variant("default_axes", ((2, 3, 4, 5),)),
        Variant("axes_0", ((2, 3, 4, 5),), {"axes": [0]}),
        Variant("axes_1", ((2, 3, 4, 5),), {"axes": [1]}),
        Variant("axes_0_1", ((2, 3, 4, 5),), {"axes": [0, 1]}),
        Variant("all_axes", ((2, 3, 4, 5),), {"axes": [0, 1, 2, 3]}),
        Variant("negative_axes", ((2, 3, 4, 5),), {"axes": [-1]}),
        Variant("rank_2", ((4, 3),), {"axes": [0]}),
        Variant("empty_group", ((0, 3),), {"axes": [0]}),
        Variant("empty_result", ((2, 0),), {"axes": [0]}),
    ]
    return tuple(
        replace(variant, elem_types=(TensorProto.FLOAT,)) for variant in variants
    )


# LRN sums the squares of a window of channels around each element. The reference's channel
# loop is bounded by the *batch* extent rather than by the channel one, so it computes the
# whole result only where the two are equal; shapes that differ have no oracle here and rest
# on the corpus's own 5x5x5x5 expectations.
_LRN_VARIANTS = (
    Variant("window_3", ((3, 3, 2, 2),), {"size": 3}),
    Variant("window_1", ((3, 3, 2, 2),), {"size": 1}),
    # An even window reaches one further forward than back, and one wider than the tensor
    # clamps at both ends.
    Variant("window_2", ((3, 3, 2, 2),), {"size": 2}),
    Variant("window_wider_than_channels", ((3, 3, 2, 2),), {"size": 5}),
    Variant(
        "scaled",
        ((3, 3, 2, 2),),
        {"size": 3, "alpha": 0.0002, "beta": 0.75, "bias": 2.0},
    ),
    Variant("beta_one", ((3, 3, 2, 2),), {"size": 3, "beta": 1.0}),
    Variant("singleton_spatial", ((2, 2, 1, 1),), {"size": 2}),
    Variant("empty_spatial", ((2, 2, 0, 3),), {"size": 3}),
    Variant("empty_channels", ((0, 0, 2, 2),), {"size": 3}),
)


# The views -- the ops that rearrange elements rather than compute them. Every one of them
# takes the shape of its result from an operand or an attribute, so the interesting axis is
# that description and not the values: the shapes and attributes below are the sweep, and the
# operand carrying them is pinned rather than drawn, since a random draw would be a shape.
#
# What a view kernel emits depends on the element type only through the C type it copies, so
# `_typed_at` sweeps the shape family at float32 and one variant per op at every type the
# schema allows -- crossing the two would repeat the same addressing once per dtype.
_FLOAT_ONLY = (TensorProto.FLOAT,)


def _typed_at(variants: tuple[Variant, ...], *labels: str) -> tuple[Variant, ...]:
    """The variants named by `labels` at every element type, and the rest at float32."""
    return tuple(
        variant
        if variant.label in labels
        else replace(variant, elem_types=variant.elem_types or _FLOAT_ONLY)
        for variant in variants
    )


# Squeeze and Unsqueeze moved `axes` from an attribute to an operand at 13; a variant naming
# it one way does not apply to the revisions that take the other.
_AXES_ATTRIBUTE_VERSIONS = (1, 11)
_AXES_OPERAND_VERSIONS = (13, 21, 23, 24, 25)

# Reshape's `allowzero` arrived at 14. Without it a zero in the shape copies the input's own
# extent; with it the zero is the extent.
_RESHAPE_ALLOWZERO_VERSIONS = (14, 19, 21, 23, 24, 25)

_RESHAPE_VARIANTS = _typed_at(
    (
        Variant("merge_dims", ((2, 3, 4), (2,)), values={1: (6, 4)}),
        Variant("flatten", ((2, 3, 4), (1,)), values={1: 24}),
        Variant("split_dims", ((2, 3, 4), (4,)), values={1: (2, 3, 2, 2)}),
        Variant("negative_dim", ((2, 3, 4), (2,)), values={1: (2, -1)}),
        # A zero copies the extent of the input's axis of that position.
        Variant("zero_dim", ((2, 3, 4), (3,)), values={1: (2, 0, 4)}),
        Variant("zero_and_negative_dim", ((2, 3, 4), (3,)), values={1: (2, 0, -1)}),
        Variant("to_scalar", ((1,), (0,)), values={1: ()}),
        Variant("from_scalar", ((), (1,)), values={1: 1}),
        # A zero in the shape copies the input's own extent, so an empty result is asked for
        # through the axis that carries it, or through the inferred one.
        Variant("empty", ((0, 3), (2,)), values={1: (3, -1)}),
        Variant(
            "allowzero",
            ((0, 3, 4), (3,)),
            {"allowzero": 1},
            values={1: (3, 4, 0)},
            versions=_RESHAPE_ALLOWZERO_VERSIONS,
        ),
    ),
    "merge_dims",
)

_FLATTEN_VARIANTS = _typed_at(
    (
        Variant("axis_2", ((2, 3, 4),), {"axis": 2}),
        Variant("axis_0", ((2, 3, 4),), {"axis": 0}),
        Variant("axis_1", ((2, 3, 4),), {"axis": 1}),
        # An axis equal to the rank leaves every dimension in the first factor.
        Variant("axis_rank", ((2, 3, 4),), {"axis": 3}),
        Variant("default_axis", ((2, 3, 4),)),
        Variant("negative_axis_1", ((2, 3, 4),), {"axis": -1}),
        Variant("negative_axis_3", ((2, 3, 4),), {"axis": -3}),
        Variant("rank_1", ((5,),), {"axis": 0}),
        # The empty axis sits after `axis`: the reference reshapes to `(prod(shape[:axis]),
        # -1)`, which numpy refuses outright when the first factor is itself zero, so there
        # is no oracle for an empty dimension before it.
        Variant("empty", ((2, 0, 4),), {"axis": 1}),
    ),
    "axis_2",
)

# A variant naming no axes at all applies to both conventions: leaving the attribute out and
# leaving the operand out are the same instruction, and both mean every single dimension.
_SQUEEZE_VARIANTS = _typed_at(
    (
        Variant("all_single_dims", ((1, 4, 1, 4),)),
        Variant("nothing_to_squeeze", ((2, 3),)),
        Variant("scalar", ((1,),)),
        Variant(
            "attribute_axis_0",
            ((1, 3, 1, 4),),
            {"axes": [0]},
            versions=_AXES_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_axes_0_2",
            ((1, 3, 1, 4),),
            {"axes": [0, 2]},
            versions=_AXES_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "operand_axis_0",
            ((1, 3, 1, 4), (1,)),
            values={1: 0},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_axes_0_2",
            ((1, 3, 1, 4), (2,)),
            values={1: (0, 2)},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_negative_axis",
            ((1, 3, 1, 4), (1,)),
            values={1: -2},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_no_axes",
            ((2, 1, 3), (0,)),
            values={1: ()},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "empty", ((1, 0, 3),), {"axes": [0]}, versions=_AXES_ATTRIBUTE_VERSIONS
        ),
        Variant(
            "operand_empty",
            ((1, 0, 3), (1,)),
            values={1: 0},
            versions=_AXES_OPERAND_VERSIONS,
        ),
    ),
    "all_single_dims",
)

_UNSQUEEZE_VARIANTS = _typed_at(
    (
        Variant(
            "attribute_axis_0",
            ((3, 4),),
            {"axes": [0]},
            versions=_AXES_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_axes_0_3",
            ((3, 4),),
            {"axes": [0, 3]},
            versions=_AXES_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_scalar",
            ((),),
            {"axes": [0]},
            versions=_AXES_ATTRIBUTE_VERSIONS,
        ),
        Variant("attribute_empty", ((0, 3),), {"axes": [1]}, versions=(11,)),
        Variant(
            "operand_axis_0",
            ((4, 4), (1,)),
            values={1: 0},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_axes_0_3",
            ((3, 4), (2,)),
            values={1: (0, 3)},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        # ONNX resolves every axis against the *output's* rank, so an unsorted pair inserts
        # the same dimensions a sorted one does.
        Variant(
            "operand_unsorted_axes",
            ((3, 4), (2,)),
            values={1: (3, 0)},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_negative_axes",
            ((3, 4), (2,)),
            values={1: (-1, -4)},
            versions=_AXES_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_scalar", ((), (1,)), values={1: 0}, versions=_AXES_OPERAND_VERSIONS
        ),
        Variant(
            "operand_empty",
            ((0, 3), (1,)),
            values={1: 1},
            versions=_AXES_OPERAND_VERSIONS,
        ),
    ),
    "operand_axis_0",
)

_TRANSPOSE_VARIANTS = _typed_at(
    (
        Variant("perm_2_0_1", ((2, 3, 4),), {"perm": [2, 0, 1]}),
        Variant("perm_0_2_1", ((2, 3, 4),), {"perm": [0, 2, 1]}),
        Variant("perm_1_0_2", ((2, 3, 4),), {"perm": [1, 0, 2]}),
        Variant("identity_perm", ((2, 3, 4),), {"perm": [0, 1, 2]}),
        Variant("default_perm", ((2, 3, 4),)),
        Variant("rank_1", ((5,),), {"perm": [0]}),
        Variant("rank_0", ((),)),
        Variant("empty", ((0, 3, 4),), {"perm": [2, 0, 1]}),
    ),
    "perm_2_0_1",
)

_CONCAT_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (4, 8), (4, 8)), {"axis": 0}),
        Variant("axis_0", ((2, 3), (4, 3)), {"axis": 0}),
        Variant("axis_1", ((2, 3), (2, 4), (2, 1)), {"axis": 1}),
        Variant("negative_axis", ((2, 3, 4), (2, 3, 2)), {"axis": -1}),
        Variant("inner_axis", ((2, 3, 4), (2, 1, 4)), {"axis": 1}),
        Variant("single_operand", ((4, 8),), {"axis": 0}),
        Variant("rank_1", ((5,), (3,)), {"axis": 0}),
        Variant("empty_operand", ((0, 3), (2, 3)), {"axis": 0}),
        Variant("empty_axis", ((2, 0), (2, 0)), {"axis": 1}),
    ),
    "wide",
)

# Split's bands come from a `split` attribute up to 11, from an operand at 13, and from
# `num_outputs` -- or from an equal division of neither -- at 18.
_SPLIT_ATTRIBUTE_VERSIONS = (2, 11)
_SPLIT_OPERAND_VERSIONS = (13, 18)
_SPLIT_VARIANTS = _typed_at(
    (
        Variant(
            "operand_even",
            ((6, 4), (2,)),
            values={1: (3, 3)},
            outputs=2,
            versions=_SPLIT_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_uneven",
            ((6, 4), (3,)),
            values={1: (1, 2, 3)},
            outputs=3,
            versions=_SPLIT_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_zero_size_band",
            ((6, 4), (3,)),
            values={1: (0, 3, 3)},
            outputs=3,
            versions=_SPLIT_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_axis_1",
            ((4, 6), (2,)),
            {"axis": 1},
            values={1: (2, 4)},
            outputs=2,
            versions=_SPLIT_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_empty",
            ((0, 4), (2,)),
            values={1: (0, 0)},
            outputs=2,
            versions=_SPLIT_OPERAND_VERSIONS,
        ),
        Variant(
            "attribute_even",
            ((6, 4),),
            {"split": [3, 3]},
            outputs=2,
            versions=_SPLIT_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_uneven",
            ((6, 4),),
            {"split": [1, 2, 3]},
            outputs=3,
            versions=_SPLIT_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_axis_1",
            ((4, 6),),
            {"axis": 1, "split": [2, 4]},
            outputs=2,
            versions=_SPLIT_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_negative_axis",
            ((4, 6),),
            {"axis": -1, "split": [2, 4]},
            outputs=2,
            versions=(11,),
        ),
        # Neither form: the axis is divided equally among the outputs the node declares.
        Variant("equal_parts", ((6, 4),), outputs=2, versions=(2, 11, 13)),
        Variant(
            "num_outputs_even", ((6, 4),), {"num_outputs": 3}, outputs=3, versions=(18,)
        ),
        # An uneven division under `num_outputs` leaves the remainder in the last band.
        Variant(
            "num_outputs_uneven",
            ((5, 4),),
            {"num_outputs": 2},
            outputs=2,
            versions=(18,),
        ),
    ),
    "operand_even",
    "attribute_even",
)

# Slice-1 takes its bounds as attributes and has no steps; from 10 on they are operands.
_SLICE_ATTRIBUTE_VERSIONS = (1,)
_SLICE_OPERAND_VERSIONS = (10, 11, 13)
_SLICE_NEGATIVE_AXIS_VERSIONS = (11, 13)
_SLICE_VARIANTS = _typed_at(
    (
        Variant(
            "operand_default_axes",
            ((2, 3, 4), (3,), (3,)),
            values={1: (0, 1, 1), 2: (2, 3, 3)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_axes",
            ((2, 3, 4), (2,), (2,), (2,)),
            values={1: (1, 0), 2: (3, 3), 3: (1, 2)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_steps",
            ((2, 3, 4), (2,), (2,), (2,), (2,)),
            values={1: (0, 0), 2: (3, 4), 3: (1, 2), 4: (2, 3)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_negative_bounds",
            ((2, 3, 4), (2,), (2,), (2,)),
            values={1: (-2, -3), 2: (-1, -1), 3: (1, 2)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_negative_steps",
            ((2, 3, 4), (2,), (2,), (2,), (2,)),
            values={1: (-1, -1), 2: (-4, -5), 3: (1, 2), 4: (-1, -2)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        # Bounds beyond the extent clamp to it, in both directions.
        Variant(
            "operand_out_of_bounds",
            ((2, 3, 4), (2,), (2,), (2,)),
            values={1: (-1000, 1000), 2: (1000, -1000), 3: (1, 2)},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_empty_result",
            ((2, 3, 4), (1,), (1,), (1,)),
            values={1: 2, 2: 2, 3: 1},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_empty_source",
            ((0, 3), (1,), (1,), (1,)),
            values={1: 0, 2: 2, 3: 1},
            versions=_SLICE_OPERAND_VERSIONS,
        ),
        Variant(
            "operand_negative_axes",
            ((2, 3, 4), (1,), (1,), (1,)),
            values={1: 1, 2: 3, 3: -2},
            versions=_SLICE_NEGATIVE_AXIS_VERSIONS,
        ),
        Variant(
            "attribute_default_axes",
            ((2, 3, 4),),
            {"starts": [0, 1], "ends": [2, 3]},
            versions=_SLICE_ATTRIBUTE_VERSIONS,
        ),
        Variant(
            "attribute_axes",
            ((2, 3, 4),),
            {"starts": [1, 0], "ends": [3, 3], "axes": [1, 2]},
            versions=_SLICE_ATTRIBUTE_VERSIONS,
        ),
        # Only the end runs out of bounds here: ONNX's own shape inference gives the
        # attribute form no static shape once the *start* leaves the extent, so a model
        # with one is a compile error rather than a case. The operand form above sweeps
        # both ends of the clamping.
        Variant(
            "attribute_out_of_bounds_end",
            ((2, 3, 4),),
            {"starts": [0], "ends": [1000], "axes": [1]},
            versions=_SLICE_ATTRIBUTE_VERSIONS,
        ),
    ),
    "operand_default_axes",
    "attribute_default_axes",
)

_EXPAND_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (2,)), values={1: (4, 8)}),
        Variant("stretch_axis", ((3, 1), (2,)), values={1: (3, 4)}),
        Variant("new_axes", ((3, 1), (3,)), values={1: (2, 3, 6)}),
        # The shape broadcasts against the operand rather than replacing it, so it may be
        # shorter than the operand's own rank, and may stretch it on either side.
        Variant("broadcast_shape", ((3, 1), (3,)), values={1: (1, 3, 4)}),
        Variant("shorter_shape", ((2, 3, 1), (2,)), values={1: (3, 4)}),
        Variant("scalar_source", ((), (2,)), values={1: (2, 3)}),
        Variant("empty", ((0, 1), (2,)), values={1: (0, 4)}),
    ),
    "wide",
)

_TILE_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (2,)), values={1: (2, 1)}),
        Variant("every_axis", ((2, 3), (2,)), values={1: (2, 3)}),
        Variant("one_axis", ((2, 3), (2,)), values={1: (1, 3)}),
        Variant("no_repeats", ((2, 3), (2,)), values={1: (1, 1)}),
        Variant("rank_3", ((2, 3, 4), (3,)), values={1: (2, 1, 2)}),
        Variant("zero_repeat", ((2, 3), (2,)), values={1: (2, 0)}),
        Variant("empty_source", ((0, 3), (2,)), values={1: (2, 2)}),
    ),
    "wide",
)


# --------------------------------------------------------------------------------------
# Reading through an index, padding, and the ops built from their own coordinates
# --------------------------------------------------------------------------------------


def _cycled(count: int, extent: int) -> tuple[int, ...]:
    """`count` indices inside `[0, extent)`, striding so that every position is reached."""
    return tuple((step * 3) % extent for step in range(count))


# Every index operand below is pinned rather than drawn. An index is a position into another
# operand, so a seeded draw would spend every case on the out-of-range value ONNX leaves
# undefined — which the artifact answers with an argument error and the reference with an
# exception. What is swept is what ONNX does define: positions at both ends of the axis, and
# the negative index counted back from it.
#
# An index tensor with no elements is left out for the same reason a wrong expectation would
# be: the reference returns it as shape `(0,)` whatever the axes around it measure, which is
# not the shape ONNX's own inference derives, so there is no oracle for that case.
_GATHER_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (3,)), {"axis": 0}, values={1: (3, 0, 2)}),
        Variant("axis_1", ((2, 3, 4), (2,)), {"axis": 1}, values={1: (2, 0)}),
        Variant("negative_axis", ((2, 3, 4), (2,)), {"axis": -1}, values={1: (3, 1)}),
        Variant("default_axis", ((3, 4), (2,)), values={1: (2, 0)}),
        Variant(
            "negative_indices", ((2, 3, 4), (2,)), {"axis": 1}, values={1: (-1, -3)}
        ),
        Variant(
            "index_matrix", ((3, 4), (2, 2)), {"axis": 0}, values={1: (2, 0, 1, 1)}
        ),
        Variant("scalar_index", ((3, 4), ()), {"axis": 1}, values={1: 2}),
        Variant("rank_1", ((5,), (2,)), {"axis": 0}, values={1: (4, 0)}),
        Variant(
            "int32_indices",
            ((3, 4), (2,)),
            {"axis": 0},
            values={1: (2, 0)},
            operand_types={1: TensorProto.INT32},
        ),
        # An axis with no elements is gathered along another one, which is a shape ONNX
        # defines and a buffer the kernel must leave alone.
        Variant("empty_source_axis", ((0, 3), (2,)), {"axis": 1}, values={1: (0, 2)}),
    ),
    "wide",
)

_GATHER_ELEMENTS_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (4, 8)), {"axis": 1}, values={1: _cycled(32, 8)}),
        Variant("axis_0", ((3, 4), (2, 4)), {"axis": 0}, values={1: _cycled(8, 3)}),
        Variant("axis_1", ((3, 4), (3, 2)), {"axis": 1}, values={1: _cycled(6, 4)}),
        # The reference resolves a negative axis only where both operands have the same
        # shape: its own cross-section check reads `shape[dim + 1:]` unnormalized, and
        # differing extents make it refuse the case rather than answer it.
        Variant(
            "negative_axis", ((3, 4), (3, 4)), {"axis": -1}, values={1: _cycled(12, 4)}
        ),
        Variant("negative_indices", ((3, 4), (3, 4)), {"axis": 1}, values={1: -1}),
        Variant("rank_1", ((5,), (3,)), {"axis": 0}, values={1: (4, 0, 2)}),
        Variant(
            "rank_3", ((2, 3, 4), (2, 3, 2)), {"axis": 2}, values={1: _cycled(12, 4)}
        ),
        Variant(
            "int32_indices",
            ((3, 4), (3, 4)),
            {"axis": 1},
            values={1: _cycled(12, 4)},
            operand_types={1: TensorProto.INT32},
        ),
    ),
    "wide",
)

# GatherND's `batch_dims` arrived at 12.
_GATHER_ND_BATCH_VERSIONS = (12, 13)
_GATHER_ND_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (3, 1)), values={1: (0, 3, 2)}),
        Variant("full_depth", ((3, 4), (2, 2)), values={1: (0, 1, 2, 3)}),
        Variant("slices", ((2, 3, 4), (2, 2)), values={1: (0, 1, 1, 2)}),
        Variant("index_rank_3", ((3, 4), (2, 2, 1)), values={1: (0, 1, 2, 0)}),
        Variant("negative_indices", ((3, 4), (1, 2)), values={1: (-1, -2)}),
        Variant(
            "batch_dims_1",
            ((2, 3, 4), (2, 2, 1)),
            {"batch_dims": 1},
            values={1: (0, 2, 1, 1)},
            versions=_GATHER_ND_BATCH_VERSIONS,
        ),
        # Unlike Gather's, GatherND's reference reshapes an empty gather onto the shape ONNX
        # infers for it, so the case has an oracle.
        Variant("empty_indices", ((4, 8), (0, 1)), values={1: ()}),
    ),
    "wide",
)

# Pad's pads are its configuration — where the operand sits inside the result — so they are
# carried as an initializer and pinned per variant, as are the axes they apply to. The fill
# value is pinned for the reason Clip's bounds are: one scalar cannot carry a dtype's edges,
# and which one it carried would be an accident of the seed.
#
# A negative pad, which ONNX defines as cropping, is absent: `np.pad` refuses one outright,
# so the reference evaluator is no oracle for it.
_PAD_ATTRIBUTE_MODES = ("constant", "reflect")
_PAD_AXES_VERSIONS = (18, 19, 21, 23, 24, 25)
_PAD_WRAP_VERSIONS = (19, 21, 23, 24, 25)
_PAD_MODE_VERSIONS = {"wrap": _PAD_WRAP_VERSIONS}
_PAD_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (4,), ()), values={1: (1, 2, 3, 0), 2: 5.0}),
        *(
            Variant(
                mode,
                ((2, 3, 4), (6,), ()),
                {"mode": mode},
                values={1: (1, 0, 2, 1, 2, 0), 2: 3.5},
                versions=_PAD_MODE_VERSIONS.get(mode),
            )
            for mode in ("constant", "edge", "reflect", "wrap")
        ),
        Variant("no_pads", ((2, 3), (4,), ()), values={1: (0, 0, 0, 0), 2: 1.0}),
        Variant("one_sided", ((2, 3), (4,), ()), values={1: (0, 2, 3, 0), 2: 1.0}),
        Variant("rank_1", ((5,), (2,), ()), values={1: (2, 3), 2: 1.0}),
        Variant(
            "fill_is_not_a_number",
            ((2, 3), (4,), ()),
            values={1: (1, 1, 1, 1), 2: float("nan")},
            elem_types=_FLOAT_ELEM_TYPES,
        ),
        # An operand with no elements has nothing to reflect or repeat, but a constant pad
        # fills the whole result from the value alone.
        Variant("empty_source", ((0, 3), (4,), ()), values={1: (1, 0, 1, 0), 2: 7.0}),
        Variant(
            "axes",
            ((2, 3, 4), (2,), (), (1,)),
            values={1: (1, 2), 2: 1.0, 3: (1,)},
            versions=_PAD_AXES_VERSIONS,
        ),
        Variant(
            "negative_axes",
            ((2, 3, 4), (4,), (), (2,)),
            values={1: (1, 2, 0, 1), 2: 1.0, 3: (-1, -3)},
            versions=_PAD_AXES_VERSIONS,
        ),
        Variant(
            "wrap_axes",
            ((2, 3, 4), (2,), (), (1,)),
            {"mode": "wrap"},
            values={1: (2, 3), 2: 1.0, 3: (2,)},
            versions=_PAD_WRAP_VERSIONS,
        ),
        # Up to opset 10 the pads and the fill are attributes instead, which is a kernel of
        # its own.
        *(
            Variant(
                f"attribute_{mode}",
                ((2, 3, 4),),
                {"pads": [1, 0, 2, 1, 2, 0], "value": 2.5, "mode": mode},
                versions=(2,),
            )
            for mode in _PAD_ATTRIBUTE_MODES
        ),
    ),
    "wide",
)

# OneHot's depth is the extent of the axis it inserts, so it is an initializer; the two
# values it selects between are parameters, pinned for the reason Clip's bounds are — and
# pinned to ordinary numbers, since the reference reaches them through `off + (on - off) * m`
# rather than by selecting, which is the same value for every pair but an infinite one and no
# value at all for a boolean one, where numpy refuses to subtract. ONNX's own shape inference
# refuses indices of rank 0, so the smallest case here is rank 1.
_ONE_HOT_VALUES = (2, 7)
_ONE_HOT_VARIANTS = _typed_at(
    (
        Variant(
            "wide",
            ((4, 8), (), (2,)),
            values={0: _cycled(32, 5), 1: 5, 2: _ONE_HOT_VALUES},
            elem_types=_NON_BOOL_TYPES,
        ),
        Variant(
            "default_axis",
            ((3,), (), (2,)),
            values={0: (0, 2, 1), 1: 3, 2: _ONE_HOT_VALUES},
        ),
        Variant(
            "axis_0",
            ((3, 2), (), (2,)),
            {"axis": 0},
            values={0: _cycled(6, 4), 1: 4, 2: _ONE_HOT_VALUES},
        ),
        Variant(
            "axis_1",
            ((3, 2), (), (2,)),
            {"axis": 1},
            values={0: _cycled(6, 4), 1: 4, 2: _ONE_HOT_VALUES},
        ),
        Variant(
            "negative_axis",
            ((3, 2), (), (2,)),
            {"axis": -2},
            values={0: _cycled(6, 4), 1: 4, 2: _ONE_HOT_VALUES},
        ),
        Variant(
            "negative_indices",
            ((4,), (), (2,)),
            values={0: (-1, -5, -3, 0), 1: 5, 2: _ONE_HOT_VALUES},
        ),
        Variant(
            "empty_indices",
            ((0, 3), (), (2,)),
            values={0: (), 1: 4, 2: _ONE_HOT_VALUES},
        ),
        # ONNX types the indices and the depth as any numeric tensor, and folds an index into
        # the depth whichever it is: a fractional one then matches no position at all.
        Variant(
            "float_indices",
            ((4,), (), (2,)),
            values={0: (1.5, 2.0, -1.0, 0.0), 1: 4, 2: _ONE_HOT_VALUES},
            operand_types={0: TensorProto.FLOAT},
        ),
        Variant(
            "int32_indices",
            ((3,), (), (2,)),
            values={0: (0, 2, 1), 1: 3, 2: _ONE_HOT_VALUES},
            operand_types={0: TensorProto.INT32},
        ),
        Variant(
            "float_depth",
            ((3,), (), (2,)),
            values={0: (0, 2, 1), 1: 3.0, 2: _ONE_HOT_VALUES},
            operand_types={1: TensorProto.FLOAT},
        ),
    ),
    "wide",
)

_EYE_LIKE_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8),)),
        Variant("square", ((5, 5),)),
        Variant("tall", ((5, 3),)),
        Variant("above_the_diagonal", ((4, 5),), {"k": 2}),
        Variant("below_the_diagonal", ((4, 5),), {"k": -2}),
        Variant("beyond_the_matrix", ((3, 3),), {"k": 5}),
        Variant("to_double", ((3, 4),), {"dtype": TensorProto.DOUBLE}),
        Variant("to_int64", ((3, 4),), {"dtype": TensorProto.INT64}),
        Variant("empty", ((0, 3),)),
    ),
    "wide",
)

# Trilu's diagonal decides no shape, so it stays a run-time operand — pinned all the same,
# since a drawn one would put every case on the far side of the matrix.
_TRILU_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8),)),
        Variant("upper_default", ((4, 5),)),
        Variant("lower", ((4, 5),), {"upper": 0}),
        Variant("upper_offset", ((4, 5), ()), {"upper": 1}, values={1: 2}),
        Variant("lower_offset", ((4, 5), ()), {"upper": 0}, values={1: -1}),
        Variant("offset_zero", ((4, 5), ()), values={1: 0}),
        Variant("offset_beyond_the_matrix", ((4, 5), ()), values={1: 9}),
        Variant("batched", ((2, 3, 4),), {"upper": 0}),
        Variant("one_row", ((1, 5),)),
        Variant("one_column", ((5, 1),), {"upper": 0}),
        Variant("empty", ((0, 3),)),
    ),
    "wide",
)

# ONNX defines a sequence length as being in [1, s], and the artifact refuses anything else
# at run time rather than reading past its buffers, so the lengths are pinned inside it.
_REVERSE_SEQUENCE_VARIANTS = _typed_at(
    (
        Variant(
            "wide",
            ((4, 8), (8,)),
            {"batch_axis": 1, "time_axis": 0},
            values={1: (1, 2, 3, 4, 4, 1, 2, 3)},
        ),
        Variant(
            "batch_major",
            ((3, 5), (3,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: (1, 5, 3)},
        ),
        Variant(
            "time_major",
            ((5, 3), (3,)),
            {"batch_axis": 1, "time_axis": 0},
            values={1: (5, 1, 3)},
        ),
        Variant("default_axes", ((4, 3), (3,)), values={1: (4, 2, 1)}),
        Variant(
            "full_length",
            ((3, 4), (3,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: 4},
        ),
        Variant(
            "single_step",
            ((3, 4), (3,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: 1},
        ),
        Variant(
            "rank_3",
            ((2, 3, 4), (2,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: (3, 2)},
        ),
        Variant(
            "empty_features",
            ((2, 3, 0), (2,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: (3, 1)},
        ),
        Variant(
            "empty_batch",
            ((0, 3), (0,)),
            {"batch_axis": 0, "time_axis": 1},
            values={1: ()},
        ),
    ),
    "wide",
)

# `largest` and `sorted` arrived at 11, and `k` moved from an attribute to an operand at 10.
_TOP_K_RANKED_VERSIONS = (11, 24)
_TOP_K_VARIANTS = _typed_at(
    (
        Variant("wide", ((4, 8), (1,)), {"axis": 1}, values={1: 3}, outputs=2),
        Variant("axis_0", ((4, 3), (1,)), {"axis": 0}, values={1: 2}, outputs=2),
        Variant("default_axis", ((3, 5), (1,)), values={1: 2}, outputs=2),
        Variant(
            "negative_axis", ((2, 3, 4), (1,)), {"axis": -2}, values={1: 2}, outputs=2
        ),
        Variant("every_element", ((3, 4), (1,)), {"axis": 1}, values={1: 4}, outputs=2),
        Variant("one_element", ((3, 4), (1,)), {"axis": 1}, values={1: 1}, outputs=2),
        Variant("rank_1", ((5,), (1,)), {"axis": 0}, values={1: 3}, outputs=2),
        Variant("empty_rows", ((0, 4), (1,)), {"axis": 1}, values={1: 2}, outputs=2),
        # A group of equal values is what pins the tie-break: ONNX ranks the smaller index
        # first, at either end of the ranking.
        Variant("ties", ((3, 4), (1,)), {"axis": 1}, values={0: 1, 1: 2}, outputs=2),
        Variant(
            "smallest",
            ((4, 8), (1,)),
            {"axis": 1, "largest": 0},
            values={1: 3},
            outputs=2,
            versions=_TOP_K_RANKED_VERSIONS,
        ),
        Variant(
            "smallest_ties",
            ((3, 4), (1,)),
            {"axis": 1, "largest": 0},
            values={0: 1, 1: 2},
            outputs=2,
            versions=_TOP_K_RANKED_VERSIONS,
        ),
        Variant(
            "attribute_k", ((4, 8),), {"axis": 1, "k": 3}, outputs=2, versions=(1,)
        ),
    ),
    "wide",
)


# The reference evaluator is an oracle for a slice of each recurrent op, and the slice is what
# is swept. All three implement one direction only (they raise outright for a bidirectional
# weight tensor), ignore `clip` and `sequence_lens`, and return two outputs at most. The LSTM
# additionally drops `activations` and `input_forget` on the floor -- its own source says
# "TODO: support overridden attributes" -- so `Y_c` never comes back either; the GRU does read
# `linear_before_reset`; and the RNN honours its activation, of the two it knows. Every case
# below therefore runs one forward direction, with any sequence length equal to the padded one.
# What the rest of the family computes is settled against onnxruntime in the kernel suite, and
# against the corpus's stored outputs in the conformance suite.
def _recurrent_shapes(
    seq: int,
    batch: int,
    input_size: int,
    *,
    gates: int,
    hidden: int,
    through: int,
) -> tuple[tuple[int, ...] | None, ...]:
    """`X, W, R` and the optional operands up to `through`, at one hidden size.

    The last two are the LSTM's alone: no other layer of the family carries a cell state, so
    no other one reaches past `initial_h`.
    """
    rows = gates * hidden
    shapes: list[tuple[int, ...] | None] = [
        (seq, batch, input_size),
        (1, rows, input_size),
        (1, rows, hidden),
        (1, 2 * rows),
        (batch,),
        (1, batch, hidden),
        (1, batch, hidden),
        (1, 3 * hidden),
    ]
    return tuple(shapes[: through + 1])


_LSTM_HIDDEN = 5
_LSTM_GATES = 4 * _LSTM_HIDDEN

_lstm_shapes = partial(_recurrent_shapes, gates=4, hidden=_LSTM_HIDDEN)

_LSTM_ATTRIBUTES: Mapping[str, Any] = {"hidden_size": _LSTM_HIDDEN}

_LSTM_VARIANTS = (
    Variant("minimal", _lstm_shapes(3, 2, 4, through=2), _LSTM_ATTRIBUTES, outputs=2),
    Variant("bias", _lstm_shapes(3, 2, 4, through=3), _LSTM_ATTRIBUTES, outputs=2),
    # Every sequence runs to the padded length: the reference reads the operand and then
    # ignores it, so it is an oracle for no other value.
    Variant(
        "sequence_lens",
        _lstm_shapes(3, 2, 4, through=4),
        _LSTM_ATTRIBUTES,
        values={4: 3},
        outputs=2,
    ),
    Variant(
        "initial_state",
        (
            *_lstm_shapes(3, 2, 4, through=3),
            None,
            (1, 2, _LSTM_HIDDEN),
            (1, 2, _LSTM_HIDDEN),
        ),
        _LSTM_ATTRIBUTES,
        outputs=2,
    ),
    Variant(
        "peepholes",
        _lstm_shapes(3, 2, 4, through=7),
        _LSTM_ATTRIBUTES,
        values={4: 3},
        outputs=2,
    ),
    # `Y` alone, which is what the node's first output slot carries.
    Variant("states_dropped", _lstm_shapes(3, 2, 4, through=2), _LSTM_ATTRIBUTES),
    Variant(
        "one_step",
        _lstm_shapes(1, 3, 4, through=7),
        _LSTM_ATTRIBUTES,
        values={4: 1},
        outputs=2,
    ),
    Variant(
        "one_item",
        _lstm_shapes(4, 1, 1, through=7),
        _LSTM_ATTRIBUTES,
        values={4: 4},
        outputs=2,
    ),
    Variant(
        "empty_batch", _lstm_shapes(3, 0, 4, through=2), _LSTM_ATTRIBUTES, outputs=2
    ),
    # Layout 1 packs the batch outermost, so `X` reads as `[batch, seq, input]`. The
    # reference's own layout path holds only for a single step -- it takes the batch size off
    # the operand before transposing it, so a longer sequence broadcasts the initial state
    # against the wrong axis, and it reports `Y_h` at a shape of its own -- which is also the
    # shape the corpus's one batchwise test runs at.
    Variant(
        "batchwise",
        ((3, 1, 4), (1, _LSTM_GATES, 4), (1, _LSTM_GATES, _LSTM_HIDDEN)),
        {**_LSTM_ATTRIBUTES, "layout": 1},
        outputs=2,
    ),
)

_GRU_HIDDEN = 5
_GRU_GATES = 3 * _GRU_HIDDEN

_gru_shapes = partial(_recurrent_shapes, gates=3, hidden=_GRU_HIDDEN)

_GRU_ATTRIBUTES: Mapping[str, Any] = {"hidden_size": _GRU_HIDDEN}

_GRU_VARIANTS = (
    Variant("minimal", _gru_shapes(3, 2, 4, through=2), _GRU_ATTRIBUTES, outputs=2),
    Variant("bias", _gru_shapes(3, 2, 4, through=3), _GRU_ATTRIBUTES, outputs=2),
    # Every sequence runs to the padded length: the reference reads the operand and then
    # ignores it, so it is an oracle for no other value. It also squeezes the operand's only
    # axis before ignoring it, which is a batch of one or nothing at all.
    Variant(
        "sequence_lens",
        _gru_shapes(3, 1, 4, through=4),
        _GRU_ATTRIBUTES,
        values={4: 3},
        outputs=2,
    ),
    Variant(
        "initial_state",
        (*_gru_shapes(3, 2, 4, through=3), None, (1, 2, _GRU_HIDDEN)),
        _GRU_ATTRIBUTES,
        outputs=2,
    ),
    # `linear_before_reset` is the one attribute of the family's the reference does read, and
    # it moves both what the reset gate scales and where the candidate's recurrent bias is
    # added -- which is why it is swept with a bias and without one.
    Variant(
        "linear_before_reset",
        _gru_shapes(3, 2, 4, through=3),
        {**_GRU_ATTRIBUTES, "linear_before_reset": 1},
        outputs=2,
    ),
    Variant(
        "linear_before_reset_unbiased",
        _gru_shapes(3, 2, 4, through=2),
        {**_GRU_ATTRIBUTES, "linear_before_reset": 1},
        outputs=2,
    ),
    # `Y` alone, which is what the node's first output slot carries.
    Variant("states_dropped", _gru_shapes(3, 2, 4, through=2), _GRU_ATTRIBUTES),
    Variant("one_step", _gru_shapes(1, 3, 4, through=3), _GRU_ATTRIBUTES, outputs=2),
    Variant(
        "one_item",
        _gru_shapes(4, 1, 1, through=5),
        _GRU_ATTRIBUTES,
        values={4: 4},
        outputs=2,
    ),
    Variant("empty_batch", _gru_shapes(3, 0, 4, through=2), _GRU_ATTRIBUTES, outputs=2),
    # Layout 1 packs the batch outermost, so `X` reads as `[batch, seq, input]`. The
    # reference's own layout path holds only for a single step, for the reason recorded on
    # the LSTM's own batchwise case.
    Variant(
        "batchwise",
        ((3, 1, 4), (1, _GRU_GATES, 4), (1, _GRU_GATES, _GRU_HIDDEN)),
        {**_GRU_ATTRIBUTES, "layout": 1},
        outputs=2,
    ),
)

_RNN_HIDDEN = 5

_rnn_shapes = partial(_recurrent_shapes, gates=1, hidden=_RNN_HIDDEN)

_RNN_ATTRIBUTES: Mapping[str, Any] = {"hidden_size": _RNN_HIDDEN}

_RNN_VARIANTS = (
    Variant("minimal", _rnn_shapes(3, 2, 4, through=2), _RNN_ATTRIBUTES, outputs=2),
    Variant("bias", _rnn_shapes(3, 2, 4, through=3), _RNN_ATTRIBUTES, outputs=2),
    Variant(
        "sequence_lens",
        _rnn_shapes(3, 1, 4, through=4),
        _RNN_ATTRIBUTES,
        values={4: 3},
        outputs=2,
    ),
    Variant(
        "initial_state",
        (*_rnn_shapes(3, 2, 4, through=3), None, (1, 2, _RNN_HIDDEN)),
        _RNN_ATTRIBUTES,
        outputs=2,
    ),
    # The RNN is the one member of the family whose activation the reference reads, and
    # `Affine` the one parameterized function it knows -- so it is also the only oracle in
    # the suite for `activation_alpha` and `activation_beta` reaching the kernel at all.
    Variant(
        "affine",
        _rnn_shapes(3, 2, 4, through=3),
        {
            **_RNN_ATTRIBUTES,
            "activations": ["Affine"],
            "activation_alpha": [0.5],
            "activation_beta": [-0.25],
        },
        outputs=2,
    ),
    Variant("states_dropped", _rnn_shapes(3, 2, 4, through=2), _RNN_ATTRIBUTES),
    Variant("one_step", _rnn_shapes(1, 3, 4, through=3), _RNN_ATTRIBUTES, outputs=2),
    Variant(
        "one_item",
        _rnn_shapes(4, 1, 1, through=5),
        _RNN_ATTRIBUTES,
        values={4: 4},
        outputs=2,
    ),
    Variant("empty_batch", _rnn_shapes(3, 0, 4, through=2), _RNN_ATTRIBUTES, outputs=2),
    Variant(
        "batchwise",
        ((3, 1, 4), (1, _RNN_HIDDEN, 4), (1, _RNN_HIDDEN, _RNN_HIDDEN)),
        {**_RNN_ATTRIBUTES, "layout": 1},
        outputs=2,
    ),
)


# Resize maps every output position to a source coordinate and weights the elements around
# it, so the sweep is the three things that decide those two: `mode`, which chooses the
# weights, `coordinate_transformation_mode`, which chooses the map, and the shape family the
# scales themselves describe -- growing an axis, shrinking one, leaving one alone, and doing
# each at once. `mode` and the transformation are swept as a full cross product: which
# combinations interact is exactly what a hand-picked list would be guessing at.
#
# Its three operands are carried in the model as initializers rather than fed: they decide
# the shape of the result, and a model that computes one is a model the compiler refuses by
# design. The corpus's own Resize tests, which pass all three at run time against a declared
# result shape, are what exercise the other half -- the kernel reads them from their buffers
# either way.
_RESIZE_DATA = (1, 2, 4, 5)
_RESIZE_SCALES = (1.0, 1.0, 0.6, 2.5)

# The element types the reference evaluator can be asked about: it casts its result through
# `saturate_cast`, which refuses a boolean outright, and the compiler refuses one too.
_RESIZE_ELEM_TYPES = tuple(
    elem_type for elem_type in sorted(C_TYPES) if elem_type != TensorProto.BOOL
)

# A region of interest per axis: the pair ONNX reads as the fraction of each axis the
# result is taken from, and one that reaches past the operand so that the extrapolation
# value is what lands there.
_RESIZE_ROI = (0.0, 0.0, 0.2, 0.1, 1.0, 1.0, 0.9, 0.8)
_RESIZE_ROI_OUTSIDE = (0.0, 0.0, -0.3, 0.1, 1.0, 1.0, 1.4, 0.8)

_RESIZE_TRANSFORMS = (
    "half_pixel",
    "half_pixel_symmetric",
    "pytorch_half_pixel",
    "align_corners",
    "asymmetric",
)


def _resize_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(
            f"{mode}_{transform}",
            (_RESIZE_DATA, None, (4,)),
            {"mode": mode, "coordinate_transformation_mode": transform},
            values={2: _RESIZE_SCALES},
        )
        for mode in ("nearest", "linear", "cubic")
        for transform in _RESIZE_TRANSFORMS
    ]
    # The one transformation that reads the region of interest, and so takes one.
    variants += [
        Variant(
            f"{mode}_tf_crop_and_resize",
            (_RESIZE_DATA, (8,), (4,)),
            {"mode": mode, "coordinate_transformation_mode": "tf_crop_and_resize"},
            values={1: _RESIZE_ROI, 2: _RESIZE_SCALES},
        )
        for mode in ("nearest", "linear", "cubic")
    ]
    # The shape family: an axis grown, shrunk, left alone, and every rank the op serves.
    variants += [
        Variant(
            label,
            (shape, None, (len(shape),)),
            {"mode": "linear"},
            values={2: scales},
        )
        for label, shape, scales in (
            ("upsampled", _RESIZE_DATA, (1.0, 1.0, 2.0, 3.0)),
            ("downsampled", _RESIZE_DATA, (1.0, 1.0, 0.6, 0.5)),
            ("unchanged", _RESIZE_DATA, (1.0, 1.0, 1.0, 1.0)),
            ("spatial_1d", (2, 3, 7), (1.0, 1.0, 1.7)),
            ("spatial_3d", (1, 2, 3, 2, 4), (1.0, 1.0, 2.0, 0.5, 1.5)),
            ("rank_1", (5,), (2.5,)),
            ("single_element", (1, 1, 1, 1), (1.0, 1.0, 3.0, 1.0)),
            # A scale that shrinks an axis to nothing at all, and an operand holding
            # nothing to begin with.
            ("empty_result", _RESIZE_DATA, (1.0, 1.0, 0.2, 1.0)),
            ("empty_batch", (0, 2, 4, 5), (1.0, 1.0, 2.0, 0.5)),
        )
    ]
    # `sizes` states the result's extents instead, and a policy other than `stretch` takes
    # one scale for every axis it names -- the smallest or the largest the sizes ask for.
    variants += [
        Variant(
            label,
            (_RESIZE_DATA, None, None, (len(sizes),)),
            {"mode": "linear", **attributes},
            values={3: sizes},
        )
        for label, sizes, attributes in (
            ("sizes", (1, 2, 3, 7), {}),
            ("sizes_axes", (3, 7), {"axes": [2, 3]}),
            ("sizes_axes_reversed", (7, 3), {"axes": [3, 2]}),
            (
                "sizes_not_larger",
                (3, 7),
                {"axes": [2, 3], "keep_aspect_ratio_policy": "not_larger"},
            ),
            (
                "sizes_not_smaller",
                (3, 7),
                {"axes": [2, 3], "keep_aspect_ratio_policy": "not_smaller"},
            ),
            (
                "sizes_not_larger_every_axis",
                (1, 2, 3, 7),
                {"keep_aspect_ratio_policy": "not_larger"},
            ),
        )
    ]
    variants += [
        # `axes` names which axes the operands describe, in the order they describe them.
        Variant(
            "scales_axes",
            (_RESIZE_DATA, None, (2,)),
            {"mode": "cubic", "axes": [2, 3]},
            values={2: (0.6, 2.5)},
        ),
        Variant(
            "scales_axes_reversed",
            (_RESIZE_DATA, None, (2,)),
            {"mode": "cubic", "axes": [3, 2]},
            values={2: (2.5, 0.6)},
        ),
        Variant(
            "roi_axes",
            (_RESIZE_DATA, (4,), None, (2,)),
            {
                "mode": "linear",
                "axes": [2, 3],
                "coordinate_transformation_mode": "tf_crop_and_resize",
            },
            values={1: (0.2, 0.1, 0.9, 0.8), 3: (3, 7)},
        ),
        # A region reaching past the operand, where the extrapolation value lands instead.
        Variant(
            "extrapolation",
            (_RESIZE_DATA, (8,), (4,)),
            {
                "mode": "linear",
                "coordinate_transformation_mode": "tf_crop_and_resize",
                "extrapolation_value": 7.5,
            },
            values={1: _RESIZE_ROI_OUTSIDE, 2: _RESIZE_SCALES},
            elem_types=_FLOAT_ELEM_TYPES,
        ),
        # ONNX allows the region in any floating-point type, and the arithmetic that reads
        # it is that type's own.
        Variant(
            "roi_double",
            (_RESIZE_DATA, (8,), (4,)),
            {"mode": "linear", "coordinate_transformation_mode": "tf_crop_and_resize"},
            values={1: _RESIZE_ROI, 2: _RESIZE_SCALES},
            operand_types={1: TensorProto.DOUBLE},
        ),
        # Every element type the schema allows and the oracle can be asked about, at one
        # shape: what a resize does per element differs by type only in the narrowing at
        # the end, and crossing the two would repeat the same walk once per type.
        Variant(
            "typed",
            (_RESIZE_DATA, None, (4,)),
            {"mode": "linear"},
            values={2: _RESIZE_SCALES},
            elem_types=_RESIZE_ELEM_TYPES,
        ),
    ]
    # The neighbour a `nearest` resize rounds to, on scales that put coordinates on both
    # sides of a half and exactly on an element.
    variants += [
        Variant(
            f"nearest_{nearest_mode}",
            (_RESIZE_DATA, None, (4,)),
            {"mode": "nearest", "nearest_mode": nearest_mode},
            values={2: scales},
        )
        for nearest_mode in ("round_prefer_floor", "round_prefer_ceil", "floor", "ceil")
        for scales in (_RESIZE_SCALES,)
    ]
    variants += [
        Variant(
            f"nearest_{nearest_mode}_asymmetric",
            (_RESIZE_DATA, None, (4,)),
            {
                "mode": "nearest",
                "nearest_mode": nearest_mode,
                "coordinate_transformation_mode": "asymmetric",
            },
            values={2: (1.0, 1.0, 3.0, 1.5)},
        )
        for nearest_mode in ("round_prefer_floor", "round_prefer_ceil", "floor", "ceil")
    ]
    # `antialias` widens the filter over the elements a shrinking axis merges, and
    # `exclude_outside` drops the taps that fall off the end and renormalizes the rest.
    variants += [
        Variant(
            f"{mode}{'_antialias' if antialias else ''}"
            f"{'_exclude_outside' if exclude_outside else ''}_{label}",
            (_RESIZE_DATA, None, (4,)),
            {
                "mode": mode,
                "antialias": antialias,
                "exclude_outside": exclude_outside,
            },
            values={2: scales},
        )
        for mode in ("linear", "cubic")
        for antialias in (0, 1)
        for exclude_outside in (0, 1)
        for label, scales in (
            ("downsampled", (1.0, 1.0, 0.3, 0.5)),
            ("upsampled", (1.0, 1.0, 2.0, 3.0)),
        )
        if antialias or exclude_outside
    ]
    # The shape of the cubic filter itself, which `cubic_coeff_a` states.
    variants += [
        Variant(
            f"cubic_coeff_a_{index}",
            (_RESIZE_DATA, None, (4,)),
            {"mode": "cubic", "cubic_coeff_a": coefficient, "antialias": antialias},
            values={2: (1.0, 1.0, 0.4, 2.3)},
        )
        for index, (coefficient, antialias) in enumerate(
            ((-0.5, 0), (-1.0, 0), (-0.5, 1))
        )
    ]
    return _typed_at(tuple(variants), "typed")


_RESIZE_VARIANTS = _resize_variants()

# Upsample is Resize's predecessor at a fixed set of its settings, so the sweep is the
# shapes and the two modes it carries; everything else about the walk is Resize's above.
_UPSAMPLE_VARIANTS = _typed_at(
    tuple(
        Variant(
            f"{mode}_{label}",
            (shape, (len(shape),)),
            {"mode": mode},
            values={1: scales},
        )
        for mode in ("nearest", "linear")
        for label, shape, scales in (
            ("integer_scales", (1, 2, 4, 5), (1.0, 1.0, 2.0, 3.0)),
            ("fractional_scales", (1, 2, 4, 5), (1.0, 1.0, 1.7, 2.5)),
            ("unit_scales", (1, 2, 4, 5), (1.0, 1.0, 1.0, 1.0)),
            ("spatial_1d", (2, 3, 7), (1.0, 1.0, 1.5)),
            ("rank_1", (5,), (2.5,)),
            ("empty_batch", (0, 2, 4, 5), (1.0, 1.0, 2.0, 2.0)),
        )
    )
    + (
        Variant(
            "typed",
            ((1, 2, 4, 5), (4,)),
            {"mode": "nearest"},
            values={1: (1.0, 1.0, 2.0, 3.0)},
            elem_types=_RESIZE_ELEM_TYPES,
        ),
    ),
    "typed",
)


def _as_resize(case: Case) -> ModelProto:
    """The case's Upsample written as the Resize its successor's own spec defines it to equal.

    Resize's specification records the equivalence in as many words: `asymmetric` is
    described there as the coordinate mapping "used by Resize-10 and Upsample", and `floor`
    as the neighbour that revision takes. The reference evaluator's own Upsample is no
    oracle -- it implements integer-scaled `nearest` and raises on everything else.
    """
    variant = case.variant
    return _model(
        replace(
            case,
            op_type="Resize",
            version=19,
            variant=replace(
                variant,
                shapes=(variant.shapes[0], None, variant.shapes[1]),
                values={2: variant.values[1]},
                attributes={
                    **variant.attributes,
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            ),
        )
    )


# The two block shuffles move elements without computing any: their attributes decide the
# addressing and nothing else, so the sweep is the shape family crossed with the block, at
# every mode DepthToSpace groups its channels by.
_DEPTH_TO_SPACE_VARIANTS = _typed_at(
    tuple(
        Variant(f"{mode.lower()}_{label}", (shape,), {"blocksize": block, "mode": mode})
        for mode in ("DCR", "CRD")
        for label, shape, block in (
            ("wide", (2, 8, 2, 3), 2),
            # A block of one leaves every element where it is.
            ("unit_block", (2, 3, 4, 5), 1),
            ("cubed", (1, 9, 2, 2), 3),
            ("single_position", (1, 4, 1, 1), 2),
            ("empty_batch", (0, 8, 2, 3), 2),
            ("empty_spatial", (2, 8, 0, 3), 2),
        )
    )
    + (Variant("typed", ((2, 8, 2, 3),), {"blocksize": 2}),),
    "typed",
)

_SPACE_TO_DEPTH_VARIANTS = _typed_at(
    tuple(
        Variant(label, (shape,), {"blocksize": block})
        for label, shape, block in (
            ("wide", (2, 2, 6, 4), 2),
            ("unit_block", (2, 3, 4, 5), 1),
            ("cubed", (1, 2, 3, 6), 3),
            ("single_position", (1, 3, 2, 2), 2),
            ("empty_batch", (0, 2, 6, 4), 2),
            ("empty_channels", (2, 0, 6, 4), 2),
        )
    )
    + (Variant("typed", ((2, 2, 6, 4),), {"blocksize": 2}),),
    "typed",
)

# Col2Im folds a stack of blocks back into an image, so the sweep is the geometry that
# decides where each block sat -- the block's own extents, and the strides, dilations and
# pads that place it -- plus the overlap that makes an image position sum more than one.
# The extents are carried in the model: they are the shape of the result, and a model that
# computes one is a model the compiler refuses by design.
#
# ONNX's own reference returns nothing at all for an empty batch or an empty channel axis
# (its accumulator is never built), so there is no oracle for one; the emitted code's
# handling of those is asserted in the kernel tests instead.
# Summing truth values has no defined result, which is why the compiler refuses a boolean
# Col2Im outright; the error path is a kernel test of its own.
_COL2IM_ELEM_TYPES = tuple(
    elem_type for elem_type in sorted(C_TYPES) if elem_type != TensorProto.BOOL
)

_COL2IM_VARIANTS = _typed_at(
    tuple(
        Variant(
            label,
            (shape, (len(image),), (len(block),)),
            attributes,
            values={1: image, 2: block},
        )
        for label, shape, image, block, attributes in (
            ("blocks", (1, 5, 5), (5, 5), (1, 5), {}),
            # Every interior position is reached by four blocks, which it sums.
            ("overlapping", (1, 4, 16), (5, 5), (2, 2), {}),
            ("pads", (1, 5, 15), (5, 5), (1, 5), {"pads": [0, 1, 0, 1]}),
            # A pad that differs from the one at the other end of its own axis, and a stride
            # that differs from the one on the other axis: only the beginnings shift where a
            # block sat, so a geometry that read the ends instead computes this one wrong.
            (
                "asymmetric",
                (1, 6, 24),
                (6, 5),
                (3, 2),
                {"strides": [1, 2], "pads": [2, 1, 0, 2]},
            ),
            ("strides", (1, 9, 4), (5, 5), (3, 3), {"strides": [2, 2]}),
            ("dilations", (1, 4, 5), (6, 6), (2, 2), {"dilations": [1, 5]}),
            ("channels", (2, 12, 9), (4, 4), (2, 2), {}),
            ("signal", (1, 3, 2), (6,), (3,), {"strides": [2]}),
            ("volume", (1, 10, 12), (3, 4, 5), (1, 1, 5), {}),
        )
    )
    + (
        Variant(
            "typed",
            ((1, 4, 16), (2,), (2,)),
            values={1: (5, 5), 2: (2, 2)},
            elem_types=_COL2IM_ELEM_TYPES,
        ),
    ),
    "typed",
)

# GridSample is decided by three things: `mode`, which chooses the weights around a
# coordinate, `padding_mode`, which decides what a coordinate outside the operand reads, and
# `align_corners`, which decides what [-1, 1] spans. All three are swept as a full cross
# product: which combinations interact is exactly what a hand-picked list would be guessing
# at. The grid itself is drawn rather than pinned -- a standard normal puts roughly a third
# of its coordinates outside [-1, 1], which is what exercises the padding.
#
# The grid is float whatever the data holds: ONNX types it separately, and the reference
# denormalizes every coordinate through a float32 array whatever type it arrives in, so that
# is the width it is an oracle at.
_GRID_SAMPLE_DATA = (1, 2, 4, 5)
_GRID_SAMPLE_GRID = (1, 3, 6, 2)

# `nearest` reads one element and computes nothing, so it serves every type the schema
# allows; the interpolating modes are floating-point only, and refused otherwise.
_GRID_SAMPLE_ELEM_TYPES = tuple(sorted(C_TYPES))


def _grid_sample_variants() -> tuple[Variant, ...]:
    variants = [
        Variant(
            f"{mode}_{padding}_align{align}",
            (_GRID_SAMPLE_DATA, _GRID_SAMPLE_GRID),
            {"mode": mode, "padding_mode": padding, "align_corners": align},
        )
        for mode in ("nearest", "linear", "cubic")
        for padding in ("zeros", "border", "reflection")
        for align in (0, 1)
    ]
    # The rank family, and the shapes a sampling does not vary over. An empty result is
    # absent because ONNX's reference returns a bare array for one rather than the tuple its
    # own runner expects, which leaves nothing to compare against.
    variants += [
        Variant(label, (data, grid), {"mode": mode})
        for label, data, grid, mode in (
            ("signal", (2, 3, 7), (2, 4, 1), "linear"),
            ("volume", (1, 2, 3, 4, 5), (1, 2, 2, 2, 3), "linear"),
            ("volume_cubic", (1, 1, 4, 4, 4), (1, 2, 2, 2, 3), "cubic"),
            ("grown", (1, 2, 3, 3), (1, 7, 8, 2), "linear"),
            ("shrunk", (1, 2, 7, 8), (1, 2, 2, 2), "cubic"),
            ("single_element", (1, 1, 1, 1), (1, 2, 2, 2), "linear"),
            # An axis with no elements at all: every coordinate falls outside it.
            ("empty_spatial", (1, 1, 0, 4), (1, 2, 2, 2), "linear"),
        )
    ]
    return _typed_at(
        tuple(variants)
        + (
            Variant(
                "typed",
                (_GRID_SAMPLE_DATA, _GRID_SAMPLE_GRID),
                {"mode": "nearest"},
                elem_types=_GRID_SAMPLE_ELEM_TYPES,
            ),
        ),
        "typed",
    )


_GRID_SAMPLE_VARIANTS = _grid_sample_variants()

# AffineGrid maps a regular grid through one transform per batch, so the sweep is the shape
# of that grid and the two spacings `align_corners` chooses between. `size` is carried in
# the model: it is the shape of the result.
#
# float32 alone: the reference casts its result to float32 whatever type the transform
# arrives in, so it is no oracle for a wider one. The double kernel is the same code at
# another type, and stands on the kernel tests' second oracle.
_AFFINE_GRID_VARIANTS = tuple(
    Variant(
        f"{label}_align{align}",
        ((size[0], len(size) - 2, len(size) - 1), (len(size),)),
        {"align_corners": align},
        values={1: size},
        elem_types=_FLOAT_ONLY,
    )
    for align in (0, 1)
    for label, size in (
        ("image", (2, 3, 5, 6)),
        ("volume", (2, 1, 4, 5, 6)),
        ("single_row", (1, 3, 1, 4)),
        ("smallest", (3, 2, 2, 2)),
        ("empty_batch", (0, 3, 5, 6)),
    )
)

# RoiAlign divides each region into bins and folds a grid of bilinear samples in each, so
# the sweep is what decides the region (`spatial_scale`, the coordinate transformation), what
# decides the bins (`output_height`/`output_width`) and what decides the samples
# (`sampling_ratio`, `mode`), crossed. The regions themselves are drawn: a standard normal
# against a small feature map puts them inside it, across its edge and inverted.
#
# The batch index each region reads is pinned rather than drawn -- it is a choice of plane,
# not data, and one outside the batch is an argument the artifact rejects at run time.
_ROI_ALIGN_DATA = (2, 3, 6, 5)
_ROI_ALIGN_BATCHES = (0, 1, 1, 0)


def _roi_align_variants() -> tuple[Variant, ...]:
    shapes = (_ROI_ALIGN_DATA, (len(_ROI_ALIGN_BATCHES), 4), (len(_ROI_ALIGN_BATCHES),))
    variants = [
        Variant(
            f"{mode}_{transform}_ratio{ratio}",
            shapes,
            {
                "mode": mode,
                "coordinate_transformation_mode": transform,
                "output_height": 2,
                "output_width": 3,
                "sampling_ratio": ratio,
            },
            values={2: _ROI_ALIGN_BATCHES},
        )
        for mode in ("avg", "max")
        for transform in ("half_pixel", "output_half_pixel")
        # A ratio of zero takes as many samples per bin as the bin spans elements.
        for ratio in (0, 1, 2, 3)
    ]
    variants += [
        Variant(
            label,
            (data, (rois, 4), (rois,)),
            {**attributes, "sampling_ratio": 2},
            values={2: _ROI_ALIGN_BATCHES[:rois]},
        )
        for label, data, rois, attributes in (
            ("scaled_down", _ROI_ALIGN_DATA, 4, {"spatial_scale": 0.5}),
            ("scaled_up", _ROI_ALIGN_DATA, 4, {"spatial_scale": 2.0}),
            ("single_bin", _ROI_ALIGN_DATA, 4, {}),
            ("tall_bins", _ROI_ALIGN_DATA, 4, {"output_height": 5, "output_width": 1}),
            ("single_region", _ROI_ALIGN_DATA, 1, {"output_height": 3}),
            ("no_regions", _ROI_ALIGN_DATA, 0, {"output_height": 2}),
            ("empty_channels", (2, 0, 6, 5), 4, {"output_height": 2}),
        )
    ]
    return tuple(variants)


_ROI_ALIGN_VARIANTS = _roi_align_variants()

# MaxRoiPool rounds each region to whole elements and takes the largest in each bin, so
# nothing it computes can leave the dtype's range and every special value goes in. Its
# regions are pinned: the first of the five columns is the plane the region is read from,
# which a draw would put outside the batch. The table covers a region inside the image, one
# reaching past it, one before it, an inverted one and one that names a single element.
_MAX_ROI_POOL_DATA = (2, 2, 6, 5)
_MAX_ROI_POOL_REGIONS = (
    (0, 0, 0, 3, 3),
    (1, 1.5, 0.5, 4.5, 3.5),
    (0, -3, -2, 2, 1),
    (1, 2, 4, 8, 9),
    (1, 3, 2, 1, 1),
    (0, 2, 2, 2, 2),
)

# float32 alone: onnxruntime, the only implementation ONNX has for this op and so the
# sweep's oracle, registers it for that type only. The double kernel is the same code at
# another type, and stands on the kernel tests' equivalence between the two.
_MAX_ROI_POOL_VARIANTS = tuple(
    Variant(
        label,
        (data, (len(_MAX_ROI_POOL_REGIONS), 5)),
        {"pooled_shape": pooled, **attributes},
        values={
            1: tuple(value for region in _MAX_ROI_POOL_REGIONS for value in region)
        },
        elem_types=_FLOAT_ONLY,
    )
    for label, data, pooled, attributes in (
        ("bins", _MAX_ROI_POOL_DATA, [2, 2], {}),
        ("single_bin", _MAX_ROI_POOL_DATA, [1, 1], {}),
        ("tall_bins", _MAX_ROI_POOL_DATA, [5, 1], {}),
        # More bins than the region has elements, so some of them pool nothing at all.
        ("finer_than_the_region", _MAX_ROI_POOL_DATA, [7, 6], {}),
        ("scaled_down", _MAX_ROI_POOL_DATA, [2, 2], {"spatial_scale": 0.5}),
        ("scaled_up", _MAX_ROI_POOL_DATA, [2, 2], {"spatial_scale": 2.0}),
        ("empty_channels", (2, 0, 6, 5), [2, 2], {}),
    )
)

# The opset onnxruntime serves MaxRoiPool at. ONNX revised the op once, at 22, and only to
# widen its type constraints -- which `test_the_maxroipool_oracle_runs_the_same_op` checks
# against the schemas themselves rather than taking on trust.
_MAX_ROI_POOL_ORACLE_VERSION = 21


def _onnxruntime_outputs(model: ModelProto, feeds: Mapping[str, Any]) -> list[Any]:
    """The second oracle the compiler's parity test already stands on, run on `model`."""
    runtime = pytest.importorskip("onnxruntime")
    runtime.set_default_logger_severity(3)
    session = runtime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return list(session.run(None, dict(feeds)))


def _at_the_oracle_opset(case: Case) -> ModelProto:
    """The case's node at the newest opset its oracle implements it at."""
    return _model(replace(case, version=_MAX_ROI_POOL_ORACLE_VERSION))


# --------------------------------------------------------------------------------------
# Writing through an index
# --------------------------------------------------------------------------------------

# Every index operand below is pinned rather than drawn, for the reason the gathering ops'
# are: a seeded draw would spend every case on the out-of-range value ONNX leaves undefined.
# What is swept is what ONNX does define — positions at both ends of the axis, the negative
# index counted back from it, and the duplicates a `reduction` folds together.
#
# The rank family stops at 4: ONNX's reference implements ScatterElements for indices of
# rank 1 through 4 and raises outright for anything deeper, so there is no oracle past it.
_SCATTER_ELEMENTS_SHAPES = (
    Variant("wide", ((4, 8), (4, 8), (4, 8)), {"axis": 1}, values={1: _cycled(32, 8)}),
    # `indices` may be shorter than `data` on the axes it does not write along, which leaves
    # the elements past it alone.
    Variant("axis_0", ((3, 4), (2, 4), (2, 4)), {"axis": 0}, values={1: _cycled(8, 3)}),
    Variant("default_axis", ((3, 4), (2, 4), (2, 4)), values={1: _cycled(8, 3)}),
    Variant(
        "negative_axis",
        ((3, 4), (3, 2), (3, 2)),
        {"axis": -1},
        values={1: _cycled(6, 4)},
    ),
    Variant("negative_indices", ((3, 4), (3, 4), (3, 4)), {"axis": 1}, values={1: -1}),
    Variant("rank_1", ((5,), (3,), (3,)), values={1: (4, 0, 2)}),
    Variant(
        "rank_3",
        ((2, 3, 4), (2, 3, 2), (2, 3, 2)),
        {"axis": 2},
        values={1: _cycled(12, 4)},
    ),
    Variant(
        "rank_4",
        ((2, 2, 3, 4), (2, 2, 3, 2), (2, 2, 3, 2)),
        {"axis": 3},
        values={1: _cycled(24, 4)},
    ),
    Variant(
        "int32_indices",
        ((3, 4), (3, 4), (3, 4)),
        {"axis": 1},
        values={1: _cycled(12, 4)},
        operand_types={1: TensorProto.INT32},
    ),
    # Nothing to write at all, which still has to leave the operand's own values behind.
    Variant("no_updates", ((3, 4), (0, 4), (0, 4)), values={1: ()}),
)

# The folds, on indices that name the same element more than once — which is what tells a
# fold from a plain write, and what ONNX defines `reduction` for. Every element type goes
# through each of them: the fold is emitted per type, and two of them are the type's own
# (a boolean sum is a disjunction, an integer extremum has no NaN to reckon with).
#
# `add` and `mul` pull their integer operands into the range no sum or product of a handful
# of them can leave: ONNX does not define integer overflow and C's is undefined for the
# signed families, so a wrap-around difference would not be a divergence from the spec.
_FOLD_DOMAINS = {0: Domain.SMALL_FACTOR, 2: Domain.SMALL_FACTOR}
_SCATTER_ELEMENTS_REDUCTIONS = tuple(
    Variant(
        f"reduction_{reduction}",
        ((4, 8), (4, 8), (4, 8)),
        {"axis": 1, "reduction": reduction},
        values={1: _cycled(32, 4)},
        domains=_FOLD_DOMAINS if reduction in ("add", "mul") else {},
    )
    for reduction in ("add", "mul", "max", "min")
)

_SCATTER_ELEMENTS_VARIANTS = (
    _typed_at(_SCATTER_ELEMENTS_SHAPES, "wide") + _SCATTER_ELEMENTS_REDUCTIONS
)

# Scatter is compiled by the generator its successor is, so what its own sweep has to show
# is that the deprecated op dispatches and computes what ONNX replaced it with; the surface
# above is not swept a second time.
_SCATTER_VARIANTS = _typed_at(
    tuple(
        variant
        for variant in _SCATTER_ELEMENTS_SHAPES
        if variant.label in ("wide", "axis_0", "negative_indices", "rank_1")
    ),
    "wide",
)


def _as_scatter_elements(case: Case) -> ModelProto:
    """The case's Scatter written as the ScatterElements ONNX replaced it with.

    Scatter's own document says it in as many words: "This operator is deprecated. Please use
    ScatterElements, which provides the same functionality." The reference evaluator carries
    no implementation of Scatter at all, so the successor's — on the same operands — is the
    oracle.
    """
    return _model(replace(case, op_type="ScatterElements", version=18))


# ScatterND writes a slice per index tuple, so its sweep is the depth of those tuples — from
# one axis of the operand to all of them — crossed with the shapes around it.
_SCATTER_ND_SHAPES = (
    Variant("wide", ((4, 8), (3, 1), (3, 8)), values={1: (0, 3, 2)}),
    Variant("full_depth", ((3, 4), (2, 2), (2,)), values={1: (0, 1, 2, 3)}),
    Variant("slices", ((2, 3, 4), (2, 2), (2, 4)), values={1: (0, 1, 1, 2)}),
    Variant("index_rank_3", ((3, 4), (2, 2, 1), (2, 2, 4)), values={1: (0, 1, 2, 0)}),
    Variant("negative_indices", ((3, 4), (1, 2), (1,)), values={1: (-1, -2)}),
    Variant("rank_1", ((5,), (2, 1), (2,)), values={1: (4, 0)}),
    Variant("no_updates", ((4, 8), (0, 1), (0, 8)), values={1: ()}),
)

# The folds again, on tuples naming the same slice twice. They stay at depth 1: the
# reference's `max` and `min` branch indexes its output with the index array rather than the
# tuple built from it, which numpy answers with the wrong shape from depth 2 on and refuses
# to store — so it is an oracle for those two at depth 1 only, and the four are kept together
# rather than sweeping the same fold at two depths for half of them.
_SCATTER_ND_REDUCTIONS = tuple(
    Variant(
        f"reduction_{reduction}",
        ((4, 8), (4, 1), (4, 8)),
        {"reduction": reduction},
        values={1: (0, 2, 0, 2)},
        domains=_FOLD_DOMAINS if reduction in ("add", "mul") else {},
    )
    for reduction in ("add", "mul", "max", "min")
)

_SCATTER_ND_VARIANTS = (
    _typed_at(_SCATTER_ND_SHAPES, "wide")
    + _SCATTER_ND_REDUCTIONS
    # A fold onto single elements rather than onto slices.
    + (
        Variant(
            "reduction_add_full_depth",
            ((3, 4), (4, 2), (4,)),
            {"reduction": "add"},
            values={1: (0, 1, 0, 1, 2, 3, 2, 3)},
            domains=_FOLD_DOMAINS,
            elem_types=_FLOAT_ONLY,
        ),
    )
)

# TensorScatter writes each sample's update into that sample's own place in a cache, so its
# sweep is what decides that place: the axis the sequence runs along, the index the write
# starts at, whether the op is given those indices at all, and the mode that says what
# happens once a write runs past the end of the axis.
_TENSOR_SCATTER_CACHE = (2, 1, 4, 5)
_TENSOR_SCATTER_UPDATE = (2, 1, 2, 5)

_TENSOR_SCATTER_VARIANTS = _typed_at(
    (
        Variant(
            "wide",
            (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_UPDATE, (2,)),
            values={2: (1, 2)},
        ),
        # Without the operand every sample is written from the start of the axis.
        Variant("appended", (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_UPDATE, None)),
        Variant(
            "circular",
            (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_UPDATE, (2,)),
            {"mode": "circular"},
            values={2: (3, 2)},
        ),
        Variant(
            "circular_negative",
            (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_UPDATE, (2,)),
            {"mode": "circular"},
            values={2: (-1, -3)},
        ),
        Variant(
            "circular_appended",
            (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_UPDATE, None),
            {"mode": "circular"},
        ),
        Variant("rank_3", ((3, 4, 5), (3, 2, 5), (3,)), values={2: (0, 1, 2)}),
        Variant(
            "axis_1", ((3, 4, 5), (3, 2, 5), (3,)), {"axis": 1}, values={2: (2, 0, 1)}
        ),
        # The sequence axis is the last one, so each write moves a single element.
        Variant(
            "last_axis", ((3, 4), (3, 2), (3,)), {"axis": 1}, values={2: (0, 2, 1)}
        ),
        Variant(
            "full_sequence",
            (_TENSOR_SCATTER_CACHE, _TENSOR_SCATTER_CACHE, (2,)),
            values={2: (0, 0)},
        ),
        Variant(
            "single_step",
            (_TENSOR_SCATTER_CACHE, (2, 1, 1, 5), (2,)),
            values={2: (3, 0)},
        ),
        # A batch wider than the axis the mode wraps against: ONNX takes the whole cache
        # coordinate modulo the capacity, so the sample being written wraps along with the
        # position inside it.
        Variant(
            "batch_wraps",
            ((5, 3, 1), (5, 1, 1), (5,)),
            {"mode": "circular"},
            values={2: (2, 2, 2, 2, 2)},
        ),
        Variant("empty_batch", ((0, 1, 4, 5), (0, 1, 2, 5), (0,)), values={2: ()}),
    ),
    "wide",
)


def _with_attributes(
    base: tuple[Variant, ...], combinations: Mapping[str, Mapping[str, Any]]
) -> tuple[Variant, ...]:
    """The shape family at the op's default attributes, plus each combination at `wide`.

    An attribute changes what a kernel computes per element, never how it walks its
    operands, so crossing the two would only repeat the shape family.
    """
    return base + tuple(
        Variant(label, ((4, 8),), attributes)
        for label, attributes in sorted(combinations.items())
    )


# Attention. `B` batch items, `H` query heads over `KV` key/value heads, `Q` query positions
# against `KV_LEN` incoming key positions, at head size `D` and value head size `DV`. The 3-D
# layout packs the heads into the last axis and names their count in an attribute, which is
# what `_attention_3d` builds; the two layouts are the same node at two sets of strides, so
# every family below is swept at whichever one is not redundant for it.
_ATTENTION_BATCH = 2
_ATTENTION_HEADS = 2
_ATTENTION_Q = 2
_ATTENTION_KV = 3
_ATTENTION_SIZE = 4

_ATTENTION_3D_HEADS: Mapping[str, Any] = {
    "q_num_heads": _ATTENTION_HEADS,
    "kv_num_heads": _ATTENTION_HEADS,
}


def _attention_4d(
    *,
    batch: int = _ATTENTION_BATCH,
    q_heads: int = _ATTENTION_HEADS,
    kv_heads: int = _ATTENTION_HEADS,
    q_seq: int = _ATTENTION_Q,
    kv_seq: int = _ATTENTION_KV,
    size: int = _ATTENTION_SIZE,
    value_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """`Q, K, V` at the `(batch, head, sequence, size)` layout."""
    return (
        (batch, q_heads, q_seq, size),
        (batch, kv_heads, kv_seq, size),
        (batch, kv_heads, kv_seq, value_size if value_size is not None else size),
    )


def _attention_3d(
    *,
    batch: int = _ATTENTION_BATCH,
    q_heads: int = _ATTENTION_HEADS,
    kv_heads: int = _ATTENTION_HEADS,
    q_seq: int = _ATTENTION_Q,
    kv_seq: int = _ATTENTION_KV,
    size: int = _ATTENTION_SIZE,
    value_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """The same node at the `(batch, sequence, head * size)` layout."""
    return (
        (batch, q_seq, q_heads * size),
        (batch, kv_seq, kv_heads * size),
        (batch, kv_seq, kv_heads * (value_size if value_size is not None else size)),
    )


# The cache a node that carries one reads, and the total key length it leaves.
_ATTENTION_PAST = (_ATTENTION_BATCH, _ATTENTION_HEADS, 2, _ATTENTION_SIZE)
_ATTENTION_TOTAL = _ATTENTION_PAST[2] + _ATTENTION_KV

# A mask row that is entirely -inf, which max-subtracting leaves as `-inf - -inf`: the
# reference's softmax reports the whole row as NaN, and so must the kernel. The other row
# mixes a finite bias with a single masked-out column.
_ATTENTION_MASK_VALUES = (0.5, -math.inf, 0.0, -math.inf, -math.inf, -math.inf)

_ATTENTION_SHAPE_VARIANTS = (
    Variant("wide", _attention_4d()),
    Variant("value_head_size", _attention_4d(value_size=6)),
    Variant("grouped_query", _attention_4d(q_heads=4)),
    Variant("multi_query", _attention_4d(q_heads=4, kv_heads=1)),
    Variant("single", _attention_4d(batch=1, q_heads=1, kv_heads=1, q_seq=1, kv_seq=1)),
    Variant("empty_batch", _attention_4d(batch=0)),
    Variant("empty_head_size", _attention_4d(size=0, value_size=4)),
    Variant("empty_query", _attention_4d(q_seq=0)),
    Variant("wide_3d", _attention_3d(), _ATTENTION_3D_HEADS),
    Variant("value_head_size_3d", _attention_3d(value_size=6), _ATTENTION_3D_HEADS),
    Variant(
        "grouped_query_3d",
        _attention_3d(q_heads=4),
        {**_ATTENTION_3D_HEADS, "q_num_heads": 4},
    ),
)

# `scale` scales the product, and the reference applies its square root to each operand; the
# negative one is the edge numpy answers with a NaN rather than an error. `softcap` is only
# applied when it is positive, which is what the ignored one pins.
_ATTENTION_ATTRIBUTE_VARIANTS = (
    Variant("causal", _attention_4d(), {"is_causal": 1}),
    Variant("causal_3d", _attention_3d(), {**_ATTENTION_3D_HEADS, "is_causal": 1}),
    Variant("scaled", _attention_4d(), {"scale": 0.25}),
    Variant("scale_negative", _attention_4d(), {"scale": -1.0}),
    Variant("softcap", _attention_4d(), {"softcap": 2.0}),
    Variant("softcap_ignored", _attention_4d(), {"softcap": -1.0}),
    Variant("softcap_causal", _attention_4d(), {"softcap": 0.5, "is_causal": 1}),
    # The softmax runs in float64 unless this narrows it, whatever the tensors hold.
    Variant(
        "softmax_precision_float",
        _attention_4d(),
        {"softmax_precision": TensorProto.FLOAT},
    ),
    Variant(
        "softmax_precision_double",
        _attention_4d(),
        {"softmax_precision": TensorProto.DOUBLE},
    ),
)

# The mask families. A mask shorter than the key axis is padded out with -inf rather than
# broadcast, and every axis before that one broadcasts; a boolean mask is a different
# expression again, and a different one under `is_causal` than without it.
_ATTENTION_MASK_VARIANTS = (
    Variant("mask_2d", (*_attention_4d(), (_ATTENTION_Q, _ATTENTION_KV))),
    Variant("mask_padded", (*_attention_4d(), (_ATTENTION_Q, 2))),
    Variant(
        "mask_3d", (*_attention_4d(), (_ATTENTION_HEADS, _ATTENTION_Q, _ATTENTION_KV))
    ),
    Variant(
        "mask_4d",
        (
            *_attention_4d(),
            (_ATTENTION_BATCH, _ATTENTION_HEADS, _ATTENTION_Q, _ATTENTION_KV),
        ),
    ),
    Variant("mask_stretched", (*_attention_4d(), (1, 1, 1, _ATTENTION_KV))),
    Variant(
        "mask_infinite",
        (*_attention_4d(), (_ATTENTION_Q, _ATTENTION_KV)),
        values={3: _ATTENTION_MASK_VALUES},
    ),
    Variant(
        "mask_causal",
        (*_attention_4d(), (_ATTENTION_Q, _ATTENTION_KV)),
        {"is_causal": 1},
    ),
    Variant(
        "mask_bool",
        (*_attention_4d(), (_ATTENTION_Q, _ATTENTION_KV)),
        operand_types={3: TensorProto.BOOL},
    ),
    Variant(
        "mask_bool_4d",
        (
            *_attention_4d(),
            (_ATTENTION_BATCH, _ATTENTION_HEADS, _ATTENTION_Q, _ATTENTION_KV),
        ),
        operand_types={3: TensorProto.BOOL},
    ),
    # A True entry becomes `0 * -inf` here, which is the NaN that poisons its whole row.
    Variant(
        "mask_bool_causal",
        (*_attention_4d(), (_ATTENTION_Q, _ATTENTION_KV)),
        {"is_causal": 1},
        operand_types={3: TensorProto.BOOL},
    ),
    # The reference adds the causal triangle *into* the mask before broadcasting it, so a
    # mask carrying one row on the query axis — the padding mask a decoder passes alongside
    # `is_causal` — takes the triangle's first row for every query position rather than one
    # row each. Every rank that can carry a singleton there is swept, since which axis the
    # 1 sits on is what the addressing has to read it off.
    Variant(
        "mask_causal_one_row_2d",
        (*_attention_4d(), (1, _ATTENTION_KV)),
        {"is_causal": 1},
    ),
    Variant(
        "mask_causal_one_row_3d",
        (*_attention_4d(), (1, 1, _ATTENTION_KV)),
        {"is_causal": 1},
    ),
    Variant(
        "mask_causal_one_row_4d",
        (*_attention_4d(), (_ATTENTION_BATCH, 1, 1, _ATTENTION_KV)),
        {"is_causal": 1},
    ),
    Variant(
        "mask_causal_one_row_per_head",
        (*_attention_4d(), (_ATTENTION_BATCH, _ATTENTION_HEADS, 1, _ATTENTION_KV)),
        {"is_causal": 1},
    ),
    Variant(
        "mask_bool_causal_one_row",
        (*_attention_4d(), (_ATTENTION_BATCH, 1, 1, _ATTENTION_KV)),
        {"is_causal": 1},
        operand_types={3: TensorProto.BOOL},
    ),
    # The same singleton against a cache, where the triangle's columns start at `past_seq`.
    Variant(
        "mask_causal_one_row_past",
        (
            *_attention_4d(),
            (_ATTENTION_BATCH, 1, 1, _ATTENTION_TOTAL),
            _ATTENTION_PAST,
            _ATTENTION_PAST,
        ),
        {"is_causal": 1},
    ),
    Variant(
        "mask_3d_layout",
        (*_attention_3d(), (_ATTENTION_Q, _ATTENTION_KV)),
        _ATTENTION_3D_HEADS,
    ),
)

# The cache families. `present_key`/`present_value` are the concatenation of the cache and
# the incoming keys or values, which is what the extra outputs carry; `is_causal` counts its
# diagonal from the cached length rather than from the start of the row. Every variant asking
# for them carries a cache, and one with incoming keys: ONNX's own shape inference reports
# nothing at all for the two present outputs otherwise — no `past_key`, or an empty incoming
# `K` — and a node whose result has no inferable shape is one the compiler refuses by design.
_ATTENTION_CACHE_VARIANTS = (
    Variant("past", (*_attention_4d(), None, _ATTENTION_PAST, _ATTENTION_PAST)),
    Variant(
        "past_present",
        (*_attention_4d(), None, _ATTENTION_PAST, _ATTENTION_PAST),
        outputs=3,
    ),
    Variant(
        "past_present_causal",
        (*_attention_4d(), None, _ATTENTION_PAST, _ATTENTION_PAST),
        {"is_causal": 1},
        outputs=3,
    ),
    Variant(
        "past_present_mask",
        (
            *_attention_4d(),
            (_ATTENTION_Q, _ATTENTION_TOTAL),
            _ATTENTION_PAST,
            _ATTENTION_PAST,
        ),
        outputs=3,
    ),
    Variant(
        "past_present_3d",
        (*_attention_3d(), None, _ATTENTION_PAST, _ATTENTION_PAST),
        _ATTENTION_3D_HEADS,
        outputs=3,
    ),
    # Nothing incoming: the whole of the keys and values is what the cache already holds.
    Variant(
        "past_only",
        (*_attention_4d(kv_seq=0), None, _ATTENTION_PAST, _ATTENTION_PAST),
    ),
)

# The fourth output, and the four intermediates it may carry, each against a mask shorter
# than the key axis so that the padding is in the reported tensor too. Modes 0 and 1 are
# swept with a softcap because that is where ONNX's text distinguishes them and the
# reference does not.
_ATTENTION_REPORTED_VARIANTS = tuple(
    Variant(
        f"qk_mode_{mode}{label}",
        (
            *_attention_4d(),
            (_ATTENTION_Q, _ATTENTION_KV),
            _ATTENTION_PAST,
            _ATTENTION_PAST,
        ),
        {"qk_matmul_output_mode": mode, **attributes},
        outputs=4,
    )
    for mode in (0, 1, 2, 3)
    for label, attributes in (("", {}), ("_softcap", {"softcap": 2.0}))
)

# `nonpad_kv_seqlen` masks off every key position at or past a batch item's own length. The
# reference adds that mask into an attention bias it has already reshaped to rank 4, and the
# addition is in place, so it can only evaluate a node whose bias carries the batch axis
# already — one batch item, or a 4-D mask. Those are the two forms swept.
_ATTENTION_NONPAD_VARIANTS = (
    Variant(
        "nonpad_single_batch",
        (*_attention_4d(batch=1), None, None, None, (1,)),
        values={6: (2,)},
    ),
    Variant(
        "nonpad_mask_4d",
        (
            *_attention_4d(),
            (_ATTENTION_BATCH, _ATTENTION_HEADS, _ATTENTION_Q, _ATTENTION_KV),
            None,
            None,
            (_ATTENTION_BATCH,),
        ),
        values={6: (1, 3)},
    ),
    Variant(
        "nonpad_causal",
        (
            *_attention_4d(batch=1),
            (1, _ATTENTION_HEADS, _ATTENTION_Q, _ATTENTION_KV),
            None,
            None,
            (1,),
        ),
        {"is_causal": 1},
        values={6: (2,)},
    ),
)

_ATTENTION_VARIANTS = (
    *_ATTENTION_SHAPE_VARIANTS,
    *_ATTENTION_ATTRIBUTE_VARIANTS,
    *_ATTENTION_MASK_VARIANTS,
    *_ATTENTION_CACHE_VARIANTS,
    *_ATTENTION_REPORTED_VARIANTS,
    *_ATTENTION_NONPAD_VARIANTS,
)

# RotaryEmbedding. `position_ids` indexes the caches, so it is pinned rather than drawn: a
# seeded draw would name rows that are not there, which ONNX leaves undefined. The negative
# one is numpy's own indexing, which the reference gathers with.
_ROTARY_POSITIONS = (0, 2, 1, 3, 7, 4)
_ROTARY_NEGATIVE_POSITIONS = (-1, -8, 0, 5, -3, 2)
# Wide enough that a cache of one angle per position holds the whole special-value list.
_ROTARY_ROWS = 8

_ROTARY_VARIANTS = (
    Variant(
        "wide",
        ((2, 2, 3, 4), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (2, 3)),
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "interleaved",
        ((2, 2, 3, 4), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (2, 3)),
        {"interleaved": 1},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "negative_positions",
        ((2, 2, 3, 4), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (2, 3)),
        values={3: _ROTARY_NEGATIVE_POSITIONS},
    ),
    # A partial rotation: the lanes past `rotary_embedding_dim` are copied through untouched.
    Variant(
        "rotary_dim",
        ((2, 2, 3, 4), (_ROTARY_ROWS, 1), (_ROTARY_ROWS, 1), (2, 3)),
        {"rotary_embedding_dim": 2},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "rotary_dim_interleaved",
        ((2, 2, 3, 4), (_ROTARY_ROWS, 1), (_ROTARY_ROWS, 1), (2, 3)),
        {"rotary_embedding_dim": 2, "interleaved": 1},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "hidden_axis",
        ((2, 3, 8), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (2, 3)),
        {"num_heads": 2},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "hidden_axis_interleaved",
        ((2, 3, 8), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (2, 3)),
        {"num_heads": 2, "interleaved": 1},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "hidden_axis_rotary_dim",
        ((2, 3, 8), (_ROTARY_ROWS, 1), (_ROTARY_ROWS, 1), (2, 3)),
        {"num_heads": 2, "rotary_embedding_dim": 2},
        values={3: _ROTARY_POSITIONS},
    ),
    Variant(
        "empty_batch",
        ((0, 2, 3, 4), (_ROTARY_ROWS, 2), (_ROTARY_ROWS, 2), (0, 3)),
        values={3: ()},
    ),
    Variant(
        "one_position",
        ((1, 1, 1, 2), (1, 1), (1, 1), (1, 1)),
        values={3: (0,)},
    ),
    # Without `position_ids` the caches stand as they are, carrying the batch and the
    # position themselves and stretching over the heads.
    Variant("cached", ((2, 2, 3, 4), (2, 3, 2), (2, 3, 2))),
    Variant(
        "cached_interleaved", ((2, 2, 3, 4), (2, 3, 2), (2, 3, 2)), {"interleaved": 1}
    ),
    Variant(
        "cached_rotary_dim",
        ((2, 2, 3, 4), (2, 3, 1), (2, 3, 1)),
        {"rotary_embedding_dim": 2},
    ),
    Variant("cached_stretched", ((2, 2, 3, 4), (1, 3, 2), (1, 3, 2))),
    Variant("cached_hidden_axis", ((2, 3, 8), (2, 3, 2), (2, 3, 2)), {"num_heads": 2}),
)


# Ops with no attributes and one operand, whose whole surface is the shape family.
_POINTWISE_UNARY_OPS = (
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Cos",
    "Cosh",
    "Erf",
    "Exp",
    "Floor",
    "Identity",
    "Log",
    "Reciprocal",
    "Relu",
    "Round",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
)

# The zero-valued alpha the reference's `alpha or self.alpha` would silently replace with
# the schema default is deliberately absent from every combination below; a case built on it
# would be comparing against a value the reference does not claim to compute.
_ACTIVATION_ATTRIBUTES: dict[str, Mapping[str, Mapping[str, Any]]] = {
    "Celu": {"alpha_half": {"alpha": 0.5}, "alpha_two": {"alpha": 2.0}},
    "Elu": {"alpha_quarter": {"alpha": 0.25}, "alpha_two": {"alpha": 2.0}},
    # `approximate` selects between two formulas rather than scaling one, so both are swept
    # even though `none` is the default the shape family already runs.
    "Gelu": {"erf": {"approximate": "none"}, "tanh": {"approximate": "tanh"}},
    "HardSigmoid": {
        "shifted": {"alpha": 0.5, "beta": 0.25},
        "steep": {"alpha": 2.0, "beta": -1.0},
    },
    "LeakyRelu": {"steep": {"alpha": 0.5}, "negative": {"alpha": -0.25}},
    "Selu": {
        "unit": {"alpha": 1.0, "gamma": 1.0},
        "scaled": {"alpha": 0.5, "gamma": 2.0},
    },
    "ThresholdedRelu": {"high": {"alpha": 2.0}, "negative": {"alpha": -1.0}},
}

# Shrink's bias and lambd stay non-negative: with either negative, an integer result can
# leave the dtype's range, where the reference's own float64-to-integer cast is undefined.
_SHRINK_ATTRIBUTES: Mapping[str, Mapping[str, Any]] = {
    "hard": {"lambd": 1.5, "bias": 0.0},
    "soft": {"lambd": 1.5, "bias": 1.5},
    "wide_band": {"lambd": 3.0, "bias": 0.5},
}


def _linear_attention_variant(
    label: str,
    *,
    rule: str | None = None,
    batch: int = 2,
    steps: int = 3,
    q_heads: int = 2,
    kv_heads: int = 2,
    d_k: int = 3,
    d_v: int = 2,
    past: bool = False,
    decay: str | None = None,
    beta: str | None = None,
    **attributes: Any,
) -> Variant:
    """One LinearAttention case, from the head counts and head widths it runs at.

    Every 3-D operand packs `H * D` into its last axis, so the shapes follow from those four
    numbers. `decay` and `beta` name the granularity ONNX packs each of them at -- one value
    per key dimension or one per head for the decay, one per head or one the heads share for
    beta -- and leaving either out is how `update_rule` reaches the model: the rule forbids
    the operand it does not read, so which operands a node passes is the rule.
    """
    decay_width = kv_heads * (1 if decay == "per_head" else d_k)
    beta_width = kv_heads if beta == "per_head" else 1
    shapes: list[tuple[int, ...] | None] = [
        (batch, steps, q_heads * d_k),
        (batch, steps, kv_heads * d_k),
        (batch, steps, kv_heads * d_v),
        (batch, kv_heads, d_k, d_v) if past else None,
        None if decay is None else (batch, steps, decay_width),
        None if beta is None else (batch, steps, beta_width),
    ]
    while shapes and shapes[-1] is None:
        shapes.pop()
    return Variant(
        label,
        tuple(shapes),
        {
            "q_num_heads": q_heads,
            "kv_num_heads": kv_heads,
            **({} if rule is None else {"update_rule": rule}),
            **attributes,
        },
        outputs=2,
    )


_LINEAR_ATTENTION_VARIANTS = (
    # The four recurrences, each with exactly the gates it reads.
    _linear_attention_variant("linear", rule="linear"),
    _linear_attention_variant("gated", rule="gated", decay="per_key_dim"),
    _linear_attention_variant("gated_per_head_decay", rule="gated", decay="per_head"),
    _linear_attention_variant("delta", rule="delta", beta="per_head"),
    _linear_attention_variant("delta_shared_beta", rule="delta", beta="shared"),
    _linear_attention_variant(
        "gated_delta", rule="gated_delta", decay="per_key_dim", beta="per_head"
    ),
    # The same rule, left to the schema's own default rather than named.
    _linear_attention_variant("default_rule", decay="per_key_dim", beta="per_head"),
    _linear_attention_variant(
        "gated_delta_per_head_decay", decay="per_head", beta="shared"
    ),
    # `past_state` seeds the state the sequence starts from, which each rule carries forward
    # its own way.
    _linear_attention_variant("linear_with_past", rule="linear", past=True),
    _linear_attention_variant(
        "gated_with_past", rule="gated", decay="per_key_dim", past=True
    ),
    _linear_attention_variant(
        "delta_with_past", rule="delta", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "gated_delta_with_past", decay="per_key_dim", beta="per_head", past=True
    ),
    # Grouped-query attention: one KV head, and the one state it carries, answers several
    # query heads. A `kv_num_heads` of 1 is the multi-query special case.
    _linear_attention_variant(
        "grouped_query", q_heads=4, decay="per_key_dim", beta="per_head"
    ),
    _linear_attention_variant(
        "multi_query",
        q_heads=4,
        kv_heads=1,
        decay="per_key_dim",
        beta="per_head",
        past=True,
    ),
    # A `scale` of 0 -- the schema's default, which every variant above runs -- asks for
    # `1/sqrt(d_k)`; anything else is taken as given.
    _linear_attention_variant(
        "explicit_scale", decay="per_key_dim", beta="per_head", scale=0.25
    ),
    _linear_attention_variant(
        "negative_scale", decay="per_key_dim", beta="per_head", scale=-1.5
    ),
    # `chunk_size` is documented as a tuning hint that does not affect the output, which is
    # only worth asserting where one chunk would not span the whole sequence anyway.
    _linear_attention_variant(
        "chunk_size_hint", decay="per_key_dim", beta="per_head", chunk_size=2
    ),
    # One decode step, which is what the state inputs exist for, and the two head widths the
    # other way round.
    _linear_attention_variant(
        "decode_step", steps=1, decay="per_key_dim", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "wide_values", d_k=2, d_v=5, decay="per_key_dim", beta="per_head"
    ),
    # A key width of one packs the decay's two granularities identically; ONNX reads the
    # per-head one first, and here the two mean the same thing.
    _linear_attention_variant(
        "scalar_head",
        q_heads=1,
        kv_heads=1,
        d_k=1,
        d_v=1,
        decay="per_key_dim",
        beta="per_head",
    ),
    # The zero-element shapes the op admits. A key width of zero is the sharp one: the
    # derived scale is `1/sqrt(0)`, so every answer is that infinity times an empty sum -- a
    # NaN in the reference and here alike -- while an explicit scale leaves it a zero.
    _linear_attention_variant(
        "empty_batch", batch=0, decay="per_key_dim", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "empty_sequence", steps=0, decay="per_key_dim", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "empty_value_width", d_v=0, decay="per_key_dim", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "empty_key_width", d_k=0, decay="per_key_dim", beta="per_head", past=True
    ),
    _linear_attention_variant(
        "empty_key_width_scaled",
        d_k=0,
        decay="per_key_dim",
        beta="per_head",
        scale=0.5,
    ),
)

# --------------------------------------------------------------------------------------
# Counting n-grams
# --------------------------------------------------------------------------------------

# TfIdfVectorizer matches its input against a pool of n-grams its attributes carry, so what a
# case covers is decided by the pool and the tokens together: a draw over the whole of an
# integer dtype matches nothing at all and would compare one zero tensor against another. Most
# variants therefore pin the token sequence to values built out of the pool, and the widest
# draws freely -- which is what carries the dtype's extremes into an operand every value of
# which is compared against the pool rather than computed with.
#
# Each pool below splits differently by n-gram length: `ngram_counts` says where the entries
# of each length start, and the identifiers ONNX numbers them by run across all of them, which
# is what `ngram_indexes` is indexed by. `level_empty` is a pool whose unigram level holds
# nothing, `repeated` one listing the same n-gram twice -- the case ONNX defines as taking the
# last of the two -- and `unreached` one holding lengths outside the gram range every variant
# asks for, whose identifiers still have to advance past them.
_TFIDF_POOLS: Mapping[str, Mapping[str, Any]] = {
    "unigrams": {
        "ngram_counts": [0],
        "ngram_indexes": [0, 1, 2, 3],
        "pool_int64s": [2, 3, 5, 4],
    },
    "level_empty": {
        "ngram_counts": [0, 0],
        "ngram_indexes": [0, 1, 2],
        "pool_int64s": [5, 6, 7, 8, 6, 7],
    },
    "mixed": {
        "ngram_counts": [0, 4],
        "ngram_indexes": [0, 1, 2, 3, 4, 5, 6],
        "pool_int64s": [2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
    },
    "trigrams": {
        "ngram_counts": [0, 2, 6],
        "ngram_indexes": [0, 1, 2, 3, 4, 5],
        "pool_int64s": [2, 3, 5, 4, 5, 6, 2, 3, 4, 4, 5, 6],
    },
    "repeated": {
        "ngram_counts": [0],
        "ngram_indexes": [2, 0, 1],
        "pool_int64s": [3, 3, 5],
    },
    "unreached": {
        "ngram_counts": [0, 1, 3],
        "ngram_indexes": [0, 1, 2, 3],
        "pool_int64s": [9, 5, 6, 2, 3, 5],
    },
}

# A token sequence holding several of the pools' n-grams, adjacent and spread apart, so that
# both the skip distances and the gram lengths have something to find.
_TFIDF_TOKENS = (2, 3, 5, 4, 5, 6, 7, 8, 6, 7, 2, 3)

_TFIDF_WEIGHTS = (0.5, 1.5, -2.0, 0.25, 3.0, 1.0, 0.125)

_TFIDF_GRAM_RANGES = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3))


def _tfidf_variant(
    label: str,
    pool: str,
    *,
    shape: tuple[int, ...] = (12,),
    tokens: Sequence[int] | None = _TFIDF_TOKENS,
    mode: str = "TF",
    grams: tuple[int, int] = (1, 2),
    skip: int = 1,
    weighted: bool = False,
) -> Variant:
    attributes = {
        **_TFIDF_POOLS[pool],
        "mode": mode,
        "min_gram_length": grams[0],
        "max_gram_length": grams[1],
        "max_skip_count": skip,
    }
    if weighted:
        width = max(attributes["ngram_indexes"]) + 1
        attributes["weights"] = list(_TFIDF_WEIGHTS[:width])
    values = (
        {}
        if tokens is None
        else {
            0: tuple(tokens[index % len(tokens)] for index in range(math.prod(shape)))
        }
    )
    return Variant(f"{label}_{pool}", (shape,), attributes, values=values)


def _with_unit_weights(case: Case) -> ModelProto:
    """The case's node, with the weights it leaves out set to the identity of their product.

    The reference reads an absent `weights` as `None` rather than as an empty list, and every
    mode that reaches for it -- `IDF`, which multiplies a truncated count by the weight of its
    n-gram, and `TFIDF`, which multiplies the count itself -- raises before reaching the
    branch its own code carries for a node that sets none. Running the oracle with those
    weights set to 1 is that multiplication left as it was, and is exactly what the branch the
    raise blocks computes.
    """
    model = _model(case)
    node = model.graph.node[0]
    attributes = {entry.name for entry in node.attribute}
    if "weights" in attributes or case.variant.attributes["mode"] == "TF":
        return model
    width = max(case.variant.attributes["ngram_indexes"]) + 1
    node.attribute.append(helper.make_attribute("weights", [1.0] * width))
    return model


_TFIDF_VARIANTS = (
    tuple(
        _tfidf_variant(
            f"{mode.lower()}{'_weighted' if weighted else ''}",
            pool,
            mode=mode,
            weighted=weighted,
        )
        for pool in _TFIDF_POOLS
        for mode in ("TF", "IDF", "TFIDF")
        for weighted in (False, True)
    )
    + tuple(
        _tfidf_variant(
            f"grams_{low}_{high}_skip_{skip}", pool, grams=(low, high), skip=skip
        )
        for pool in ("mixed", "trigrams")
        for low, high in _TFIDF_GRAM_RANGES
        for skip in (0, 1, 5)
    )
    + (
        _tfidf_variant("batch", "mixed", shape=(2, 6)),
        _tfidf_variant("batch_skipped", "mixed", shape=(3, 5), skip=3, grams=(2, 3)),
        _tfidf_variant("single_row", "mixed", shape=(1, 12)),
        _tfidf_variant("single_token", "unigrams", shape=(1,)),
        # The zero-element shapes the reference reads: a sequence of no tokens, and a batch
        # of them. A batch of *no sequences* is not one of them — it refuses any `[N, C]`
        # with `N` below 1 outright — so that shape has no oracle here.
        _tfidf_variant("empty_sequence", "mixed", shape=(0,)),
        _tfidf_variant("empty_rows", "mixed", shape=(2, 0)),
        # The one variant whose tokens are drawn rather than pinned, and the only one wide
        # enough to carry the whole special-value list into the operand.
        _tfidf_variant("wide", "mixed", shape=(4, 8), tokens=None),
    )
)


# --------------------------------------------------------------------------------------
# ONNX-ML preprocessing
# --------------------------------------------------------------------------------------

# The standard-domain opset the helper nodes an ONNX-ML oracle is built from are imported at;
# any revision serves, since none of them changed what a cast or a softmax computes here.
_ML_STANDARD_OPSET = 21


def _with_float_inputs(case: Case) -> ModelProto:
    """The case's node with every operand cast to float32 in front of it.

    `Scaler`, `Normalizer`, `FeatureVectorizer` and the four predictors below are each
    declared by their own schema to produce a `tensor(float)` whatever they are handed, while
    the reference evaluator returns a result of the *input's* element type — so for anything
    but a float32 input its output contradicts the op's stated output type, and comparing
    against it would be comparing against the wrong type. The oracle is therefore run on the
    model whose input already is that float, where the two agree; for a float32 case the cast
    is an identity and this is the compiled model itself.
    """
    return _cast_inputs_to_float(_model(case))


def _cast_inputs_to_float(model: ModelProto) -> ModelProto:
    """`model` with a Cast to float32 in front of every operand it is fed."""
    graph = model.graph
    fed = [entry.name for entry in graph.input]
    casts = [
        helper.make_node("Cast", [name], [f"{name}_float"], to=TensorProto.FLOAT)
        for name in fed
    ]
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in fed:
                node.input[index] = f"{name}_float"
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return _importing_standard_ops(model)


def _importing_standard_ops(model: ModelProto) -> ModelProto:
    """`model` importing the standard domain, which its helper nodes are defined in."""
    if not any(entry.domain in ("", "ai.onnx") for entry in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", _ML_STANDARD_OPSET))
    return model


# Scaler addresses its coefficients along the last axis, which is what the shapes sweep: one
# coefficient per feature and a single one shared by all, at each rank. A rank-0 input is
# deliberately absent — the reference broadcasts the scalar against the coefficient list and
# returns the rank-1 result of that, which is not the shape the op's own schema gives it.
_SCALER_COEFFICIENTS = (
    ("per_feature", [1.0, -2.0, 0.5], [0.5, 2.0, -1.0]),
    ("shared", [1.5], [-0.25]),
    ("shared_offset", [0.0], [2.0, -3.0, 0.5]),
    ("shared_scale", [1.0, -1.0, 0.0], [4.0]),
)
_SCALER_VARIANTS = tuple(
    Variant(f"{label}_{shape_label}", (shape,), {"offset": offset, "scale": scale})
    for label, offset, scale in _SCALER_COEFFICIENTS
    for shape_label, shape in (
        ("matrix", (4, 3)),
        ("vector", (3,)),
        ("rank_3", (2, 2, 3)),
    )
) + (
    Variant("wide", ((4, 8),), {"offset": [1.0] * 8, "scale": [0.5] * 8}),
    Variant("empty_rows", ((0, 3),), {"offset": [1.0, 2.0, 3.0], "scale": [1.0] * 3}),
    Variant("empty_features", ((2, 0),), {"offset": [1.0], "scale": [2.0]}),
)

# Normalizer divides each row by a norm of that row, so every shape is the `[N,C]` matrix its
# schema describes — the reference reduces along axis 1 and has no other reading of the
# input, and it reshapes that reduction in a way no zero-element matrix survives, so those
# shapes have no oracle here. The pinned rows are the edges a draw would not reach: a row
# whose norm is zero, and hence divided by the floor the reference falls back on rather than
# by nothing; a row carrying a NaN, which its `max` and its sum both propagate; and one
# carrying an infinity, which leaves every element of the row at NaN or zero.
_NORMALIZER_EDGE_ROWS = (
    0.0,
    -0.0,
    0.0,
    float("nan"),
    1.0,
    2.0,
    float("inf"),
    1.0,
    2.0,
    -3.0,
    0.5,
    -0.25,
)
_NORMALIZER_VARIANTS = (
    tuple(
        Variant(f"{label}_{norm.lower()}", (shape,), {"norm": norm})
        for norm in ("MAX", "L1", "L2")
        for label, shape in (
            ("matrix", (4, 3)),
            ("wide", (4, 8)),
            ("single_row", (1, 5)),
            ("single_column", (4, 1)),
        )
    )
    + tuple(
        Variant(
            f"edge_rows_{norm.lower()}",
            ((4, 3),),
            {"norm": norm},
            values={0: _NORMALIZER_EDGE_ROWS},
            elem_types=_FLOAT_ELEM_TYPES,
        )
        for norm in ("MAX", "L1", "L2")
    )
    + (
        # The default `norm`, which the schema declares rather than the node.
        Variant("default_norm", ((4, 3),)),
    )
)

# Imputer's two attribute families are the floating-point and the integer marker; ONNX pairs
# each with the element types it describes, and the reference reads the float one first
# whenever it is set. Its shapes are the matrices the reference accepts — it refuses any other
# rank outright — with a value per column or a single one shared by all of them.
_INTEGER_ML_TYPES = (TensorProto.INT32, TensorProto.INT64)
_IMPUTER_VARIANTS = (
    Variant(
        "per_column",
        ((4, 3),),
        {"imputed_value_floats": [1.5, -2.5, 0.0], "replaced_value_float": 0.0},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant(
        "shared_value",
        ((4, 3),),
        {"imputed_value_floats": [9.0], "replaced_value_float": 1.0},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant(
        "nan_marker",
        ((4, 3),),
        {"imputed_value_floats": [7.0], "replaced_value_float": float("nan")},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant(
        "wide",
        ((4, 8),),
        {"imputed_value_floats": [0.25] * 8, "replaced_value_float": -1.0},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant(
        "empty_rows",
        ((0, 3),),
        {"imputed_value_floats": [1.0], "replaced_value_float": 0.0},
        elem_types=_FLOAT_ELEM_TYPES,
    ),
    Variant(
        "integer_per_column",
        ((4, 3),),
        {"imputed_value_int64s": [7, -8, 9], "replaced_value_int64": 0},
        elem_types=_INTEGER_ML_TYPES,
    ),
    Variant(
        "integer_shared",
        ((4, 3),),
        {"imputed_value_int64s": [5], "replaced_value_int64": 1},
        elem_types=_INTEGER_ML_TYPES,
    ),
)

# Binarizer is a comparison against a threshold, so its shapes are the unary family and its
# thresholds include the schema's own default, which the node leaves out.
_BINARIZER_VARIANTS = tuple(
    replace(
        variant,
        label=f"{variant.label}_{label}",
        attributes={} if threshold is None else {"threshold": threshold},
    )
    for label, threshold in (
        ("default", None),
        ("zero", 0.0),
        ("positive", 1.5),
        ("negative", -2.0),
    )
    for variant in _UNARY_VARIANTS
)

# OneHotEncoder's categories are the integer list, which is what a numeric input is matched
# against; the reference reads rank 1 and rank 2 and refuses anything deeper. `zeros` cleared
# makes a value in no category the failure the schema prescribes, so those cases pin every
# element to a category rather than drawing one.
#
# A category list with a repeat is deliberately absent: the reference sizes its result from
# the *distinct* categories while ONNX's own shape inference sizes it from the list, so the
# reference indexes past its own buffer and there is nothing to compare a kernel against.
_ENCODER_CATEGORIES = [0, 1, -1]
_IN_CATEGORY = tuple(_ENCODER_CATEGORIES[index % 3] for index in range(12))
_ONE_HOT_ENCODER_VARIANTS = (
    Variant("matrix", ((4, 3),), {"cats_int64s": _ENCODER_CATEGORIES, "zeros": 1}),
    Variant("wide", ((4, 4),), {"cats_int64s": _ENCODER_CATEGORIES, "zeros": 1}),
    Variant("vector", ((5,),), {"cats_int64s": _ENCODER_CATEGORIES, "zeros": 1}),
    Variant("single_category", ((4, 3),), {"cats_int64s": [1], "zeros": 1}),
    Variant("empty_rows", ((0, 3),), {"cats_int64s": _ENCODER_CATEGORIES, "zeros": 1}),
    Variant(
        "strict",
        ((4, 3),),
        {"cats_int64s": _ENCODER_CATEGORIES, "zeros": 0},
        values={0: _IN_CATEGORY},
    ),
)

# LabelEncoder pairs a key family with a value family. ONNX's own type inference rejects a
# key element type differing from the input's, so the `keys_int64s` and `keys_floats` families
# pair with exactly one input type each and the `keys_tensor` family is what reaches the rest.
# The keys themselves are values the generator is certain to feed — the specials every dtype's
# sweep starts with — so that hits and misses both occur.
_LABEL_KEYS_INT = [0, 1, -1]
_LABEL_KEYS_FLOAT = [0.0, 1.0, -1.0]
_LABEL_INT_MAPPING = {
    "keys_int64s": _LABEL_KEYS_INT,
    "values_int64s": [10, 20, 30],
    "default_int64": -99,
}


def _label_tensor_mapping(elem_type: int) -> dict[str, Any]:
    """A `keys_tensor`/`values_tensor` mapping, the one family that names its own types."""
    keys = np.array(_LABEL_KEYS_INT, numpy_dtype_name(elem_type))
    return {
        "keys_tensor": numpy_helper.from_array(keys, "keys"),
        "values_tensor": numpy_helper.from_array(
            np.array([3, -4, 5], np.int32), "values"
        ),
        "default_tensor": numpy_helper.from_array(np.array([-77], np.int32), "default"),
    }


_LABEL_ENCODER_VARIANTS = (
    Variant(
        "int_keys_int_values",
        ((4, 3),),
        _LABEL_INT_MAPPING,
        elem_types=(TensorProto.INT64,),
    ),
    Variant(
        "int_keys_float_values",
        ((4, 3),),
        {
            "keys_int64s": _LABEL_KEYS_INT,
            "values_floats": [1.5, -2.5, 0.0],
            "default_float": -0.5,
        },
        elem_types=(TensorProto.INT64,),
    ),
    Variant(
        "float_keys_float_values",
        ((4, 3),),
        {
            "keys_floats": _LABEL_KEYS_FLOAT,
            "values_floats": [1.5, -2.5, 0.0],
            "default_float": -0.5,
        },
        elem_types=(TensorProto.FLOAT,),
    ),
    Variant(
        "float_keys_int_values",
        ((4, 3),),
        {
            "keys_floats": _LABEL_KEYS_FLOAT,
            "values_int64s": [10, 20, 30],
            "default_int64": -99,
        },
        elem_types=(TensorProto.FLOAT,),
    ),
    # A repeated key takes its last occurrence, as the schema states in as many words.
    Variant(
        "repeated_key",
        ((4, 3),),
        {
            "keys_int64s": [1, 0, 1],
            "values_int64s": [10, 20, 30],
            "default_int64": -99,
        },
        elem_types=(TensorProto.INT64,),
    ),
    Variant(
        "default_only",
        ((4, 3),),
        {"keys_int64s": [7], "values_int64s": [8], "default_int64": 42},
        elem_types=(TensorProto.INT64,),
    ),
    *(
        Variant(label, (shape,), _LABEL_INT_MAPPING, elem_types=(TensorProto.INT64,))
        for label, shape in (("wide", (4, 8)), ("rank_0", ()), ("empty", (0, 3)))
    ),
    # The tensor family, which is where the element types the other two cannot describe are
    # swept, and which brings its own default along.
    *(
        Variant(
            f"tensor_pair_{numpy_dtype_name(elem_type)}",
            ((4, 3),),
            _label_tensor_mapping(elem_type),
            elem_types=(elem_type,),
        )
        for elem_type in (
            TensorProto.DOUBLE,
            TensorProto.FLOAT,
            TensorProto.INT16,
            TensorProto.INT32,
        )
    ),
)

# ArrayFeatureExtractor takes the columns its index operand names, so the indices are pinned
# rather than drawn, for the reason the gathering ops' are: a seeded draw would spend every
# case on the out-of-range value ONNX leaves undefined. A vector `X` is the one case the
# reference documents as following onnxruntime rather than the specification, returning the
# single row a one-row matrix would have.
_EXTRACTOR_VARIANTS = (
    Variant("matrix", ((3, 4), (2,)), values={1: (3, 0)}),
    Variant("vector", ((4,), (2,)), values={1: (1, 3)}),
    Variant("rank_3", ((2, 3, 4), (2,)), values={1: (0, 2)}),
    Variant("row_of_indices", ((3, 4), (1, 2)), values={1: (2, 1)}),
    Variant("negative_index", ((3, 4), (2,)), values={1: (-1, -4)}),
    Variant("repeated_index", ((3, 4), (3,)), values={1: (1, 1, 2)}),
    Variant("every_column", ((3, 4), (4,)), values={1: (0, 1, 2, 3)}),
    Variant("wide", ((4, 8), (2,)), values={1: (7, 0)}),
    Variant("empty_rows", ((0, 4), (2,)), values={1: (0, 3)}),
    Variant("no_indices", ((3, 4), (0,)), values={1: ()}),
)

# FeatureVectorizer lays its inputs side by side, each cut or zero-padded to its declared
# width, so the sweep is every relation a width can have to the input it describes.
_VECTORIZER_VARIANTS = (
    Variant("exact", ((4, 3), (4, 2)), {"inputdimensions": [3, 2]}),
    Variant("padded", ((4, 3), (4, 2)), {"inputdimensions": [5, 2]}),
    Variant("truncated", ((4, 3), (4, 2)), {"inputdimensions": [2, 1]}),
    Variant("single", ((4, 3),), {"inputdimensions": [3]}),
    Variant("vector_input", ((4,), (4, 2)), {"inputdimensions": [1, 2]}),
    Variant("vector_padded", ((4,),), {"inputdimensions": [3]}),
    Variant("three_inputs", ((2, 2), (2, 1), (2, 3)), {"inputdimensions": [2, 1, 3]}),
    Variant("zero_width", ((4, 3), (4, 2)), {"inputdimensions": [0, 2]}),
    Variant("empty_rows", ((0, 3), (0, 2)), {"inputdimensions": [3, 2]}),
    # Wide enough on every input for the dtype's whole special-value list to reach it.
    Variant("wide", ((4, 4), (4, 4), (4, 4)), {"inputdimensions": [4, 4, 4]}),
)

# --------------------------------------------------------------------------------------
# Tree ensembles
# --------------------------------------------------------------------------------------

# The forests below are written as trees and flattened into the two encodings ONNX-ML uses,
# so that one description covers the legacy `(tree, node)` families and opset 5's separate
# node and leaf families -- and so that what a variant is actually testing stays readable.


class _Split(NamedTuple):
    """One interior node: the test it applies, and the two subtrees it chooses between.

    `members` holds the set a `MEMBER` test matches against, which only opset 5 defines. A
    branch is either another `_Split` or a leaf, written as a list of `(target, weight)`
    pairs -- more than one of which only the legacy encoding can express.
    """

    feature: int
    test: str
    value: float
    true_branch: Any
    false_branch: Any
    missing: int = 0
    members: tuple[float, ...] = ()


# The branch tests, named without the `BRANCH_` the legacy families spell them with; opset 5
# numbers them by their position here.
_TESTS = ("LEQ", "LT", "GTE", "GT", "EQ", "NEQ", "MEMBER")

_LEGACY_NODE_FAMILIES = (
    "treeids",
    "nodeids",
    "featureids",
    "modes",
    "values",
    "truenodeids",
    "falsenodeids",
    "missing_value_tracks_true",
)
_LEGACY_LEAF_FAMILIES = ("treeids", "nodeids", "ids", "weights")
_ENSEMBLE_NODE_FAMILIES = (
    "featureids",
    "truenodeids",
    "falsenodeids",
    "trueleafs",
    "falseleafs",
    "missing_value_tracks_true",
)


def _legacy_attributes(trees: Sequence[Any], role: str, **extra: Any) -> dict[str, Any]:
    """A forest in the `(tree, node)`-keyed families the legacy ensembles are encoded in."""
    nodes: dict[str, list] = {name: [] for name in _LEGACY_NODE_FAMILIES}
    leaves: dict[str, list] = {name: [] for name in _LEGACY_LEAF_FAMILIES}
    for tree_id, tree in enumerate(trees):
        _append_legacy(tree, tree_id, nodes, leaves)
    return {
        **{f"nodes_{name}": values for name, values in nodes.items()},
        **{f"{role}_{name}": values for name, values in leaves.items()},
        **extra,
    }


def _append_legacy(
    node: Any, tree_id: int, nodes: dict[str, list], leaves: dict[str, list]
) -> int:
    """Append `node` and its subtrees, returning the node id it was given in its tree."""
    node_id = sum(1 for value in nodes["treeids"] if value == tree_id)
    position = len(nodes["treeids"])
    nodes["treeids"].append(tree_id)
    nodes["nodeids"].append(node_id)
    split = node if isinstance(node, _Split) else None
    nodes["featureids"].append(split.feature if split else 0)
    nodes["modes"].append(f"BRANCH_{split.test}" if split else "LEAF")
    nodes["values"].append(split.value if split else 0.0)
    nodes["missing_value_tracks_true"].append(split.missing if split else 0)
    nodes["truenodeids"].append(0)
    nodes["falsenodeids"].append(0)
    if split is None:
        for target, weight in node:
            leaves["treeids"].append(tree_id)
            leaves["nodeids"].append(node_id)
            leaves["ids"].append(target)
            leaves["weights"].append(weight)
        return node_id
    nodes["truenodeids"][position] = _append_legacy(
        split.true_branch, tree_id, nodes, leaves
    )
    nodes["falsenodeids"][position] = _append_legacy(
        split.false_branch, tree_id, nodes, leaves
    )
    return node_id


def _ensemble_attributes(
    trees: Sequence[Any], dtype: str = "float32", **extra: Any
) -> dict[str, Any]:
    """The same forest in opset 5's families, where leaves are indexed apart from nodes."""
    nodes: dict[str, list] = {name: [] for name in _ENSEMBLE_NODE_FAMILIES}
    tests: list[int] = []
    splits: list[float] = []
    targets: list[int] = []
    weights: list[float] = []
    members: list[float] = []
    roots = []
    for tree in trees:
        index, leaf = _append_ensemble(
            tree, nodes, tests, splits, targets, weights, members
        )
        # A tree that is a single leaf is encoded at a position in *both* families at once,
        # which is what `_bare_leaf_attributes` is written out for.
        assert not leaf, "a bare-leaf tree cannot be built from a leaf alone"
        roots.append(index)
    attributes = {
        **{f"nodes_{name}": values for name, values in nodes.items()},
        "nodes_modes": numpy_helper.from_array(
            np.array(tests, np.uint8), "nodes_modes"
        ),
        "nodes_splits": numpy_helper.from_array(
            np.array(splits, dtype), "nodes_splits"
        ),
        "leaf_targetids": targets,
        "leaf_weights": numpy_helper.from_array(
            np.array(weights, dtype), "leaf_weights"
        ),
        "tree_roots": roots,
        **extra,
    }
    if members:
        attributes["membership_values"] = numpy_helper.from_array(
            np.array(members, dtype), "membership_values"
        )
    return attributes


def _append_ensemble(
    node: Any,
    nodes: dict[str, list],
    tests: list[int],
    splits: list[float],
    targets: list[int],
    weights: list[float],
    members: list[float],
) -> tuple[int, int]:
    """Append `node`, returning its index and whether it landed in the leaf families."""
    if not isinstance(node, _Split):
        ((target, weight),) = node
        targets.append(target)
        weights.append(weight)
        return len(targets) - 1, 1
    index = len(tests)
    tests.append(_TESTS.index(node.test))
    splits.append(node.value)
    nodes["featureids"].append(node.feature)
    nodes["missing_value_tracks_true"].append(node.missing)
    for family in ("truenodeids", "falsenodeids", "trueleafs", "falseleafs"):
        nodes[family].append(0)
    # The sets are read in the order the reference builds the trees in, so they are appended
    # where that traversal reaches them: before either branch.
    if node.test == "MEMBER":
        members.extend([*node.members, float("nan")])
    for branch, child, leaf in (
        (node.true_branch, "truenodeids", "trueleafs"),
        (node.false_branch, "falsenodeids", "falseleafs"),
    ):
        nodes[child][index], nodes[leaf][index] = _append_ensemble(
            branch, nodes, tests, splits, targets, weights, members
        )
    return index, 0


# One stump per branch test, so both branches of each are taken. The ordering tests split at
# 0.5, which the seeded draws fall either side of; the equality tests split at zero, which the
# special-value list carries as both of its signs.
_STUMPS = {
    test: [
        _Split(0, test, 0.0 if test in ("EQ", "NEQ") else 0.5, [(0, 1.5)], [(0, -2.5)])
    ]
    for test in _TESTS[:6]
}
# A tree deep enough that a row's path depends on more than one feature, and one whose
# branches route a missing feature the way the flag names rather than the test.
_DEEP = [
    _Split(
        0,
        "LEQ",
        0.5,
        _Split(1, "GT", -1.0, [(0, 1.0)], [(1, 2.0)]),
        _Split(2, "LT", 2.0, [(1, -1.0)], [(0, 0.25)]),
    )
]
_MISSING = [
    _Split(
        0,
        "LEQ",
        0.5,
        [(0, 1.0)],
        _Split(1, "GTE", 0.0, [(1, 2.0)], [(0, -3.0)], missing=1),
        missing=1,
    )
]
_FOREST = [
    _Split(0, "LEQ", 0.5, [(0, 1.0)], [(1, 2.0)]),
    _Split(1, "GT", 0.0, [(1, -0.5)], [(0, 3.0)]),
    _Split(2, "LEQ", -1.0, [(0, 0.75)], [(1, -2.25)]),
]
# A leaf weighting two targets at once, which only the legacy encoding can express.
_MULTI_TARGET = [_Split(0, "LEQ", 0.5, [(0, 1.0), (1, -1.0)], [(1, 2.0)])]
# A forest whose every leaf weights both classes, so that no row's score reaches 0 or 1 —
# the two values a probit is not defined at, and the two the reference implementation's own
# `numpy.vectorize` returns a Python `int` for, which makes it return `float64` scores
# instead of the `tensor(float)` the schema declares.
_PROBABILITY_FOREST = [
    _Split(0, "LEQ", 0.5, [(0, 0.1), (1, 0.25)], [(0, 0.3), (1, 0.05)]),
    _Split(1, "GT", 0.0, [(0, 0.2), (1, 0.15)], [(0, 0.05), (1, 0.3)]),
    _Split(2, "LEQ", -1.0, [(0, 0.25), (1, 0.1)], [(0, 0.15), (1, 0.2)]),
]
# Set tests, which only opset 5 defines. The second set carries a zero, which the reference's
# own loop reads as the end of the set rather than as a member of it.
_MEMBERSHIP = [
    _Split(
        0,
        "MEMBER",
        0.0,
        [(0, 1.0)],
        _Split(0, "MEMBER", 0.0, [(1, 2.0)], [(0, -3.0)], members=(0.0, 1.0)),
        members=(1.0, -1.0, 2.0),
    )
]

# `wide` is the only shape larger than the special-value list, so it is what carries every
# float edge -- NaN and the infinities included -- into the features a branch tests.
_LEGACY_SHAPES = (
    ("matrix", (4, 3)),
    ("wide", (5, 3)),
    ("vector", (3,)),
    ("empty_rows", (0, 3)),
)


def _regressor_variants() -> tuple[Variant, ...]:
    """`TreeEnsembleRegressor` over every branch test, aggregation and score transform.

    Only float32 inputs are swept: the reference implementation scores in the element type of
    its *input*, so for a double `X` its result is not the `tensor(float)` the schema declares
    and for an integer one it refuses to accumulate at all -- neither is an oracle for what
    the op's own contract says the result should be.
    """
    forests = [
        (f"stump_{test.lower()}", trees, 1) for test, trees in _STUMPS.items()
    ] + [
        ("deep", _DEEP, 2),
        ("missing", _MISSING, 2),
        ("forest", _FOREST, 2),
        ("multi_target", _MULTI_TARGET, 2),
    ]
    variants = [
        Variant(
            label,
            ((4, 3),),
            _legacy_attributes(trees, "target", n_targets=targets),
            elem_types=_FLOAT_ONLY,
        )
        for label, trees, targets in forests
    ]
    variants += [
        Variant(
            f"forest_{label}",
            (shape,),
            _legacy_attributes(_FOREST, "target", n_targets=2, **attributes),
            elem_types=_FLOAT_ONLY,
        )
        for label, shape, attributes in (
            *((label, shape, {}) for label, shape in _LEGACY_SHAPES[1:]),
            ("sum", (4, 3), {"aggregate_function": "SUM"}),
            ("average", (4, 3), {"aggregate_function": "AVERAGE"}),
            ("minimum", (4, 3), {"aggregate_function": "MIN"}),
            ("maximum", (4, 3), {"aggregate_function": "MAX"}),
            ("base_values", (4, 3), {"base_values": [0.25, -0.5]}),
            ("shared_base_value", (4, 3), {"base_values": [1.5]}),
            (
                "base_values_average",
                (4, 3),
                {"base_values": [0.25, -0.5], "aggregate_function": "AVERAGE"},
            ),
            (
                "base_values_minimum",
                (4, 3),
                {"base_values": [0.25, -0.5], "aggregate_function": "MIN"},
            ),
            ("softmax", (4, 3), {"post_transform": "SOFTMAX"}),
            ("explicit_none", (4, 3), {"post_transform": "NONE"}),
        )
    ]
    return tuple(variants)


def _classifier_variants() -> tuple[Variant, ...]:
    """`TreeEnsembleClassifier` over the same forests, its binary rule and all five transforms.

    The scores are float32 and the labels int64 whatever `X` holds, so float32 and double are
    both swept; the integer input types are not, since the reference rounds those to float32
    before it compares -- which is neither what the op's text says nor what the regressor's
    own reference does with them.
    """
    scored = (TensorProto.FLOAT, TensorProto.DOUBLE)
    forests = [
        (f"stump_{test.lower()}", trees, [0, 1]) for test, trees in _STUMPS.items()
    ] + [
        ("deep", _DEEP, [10, 20]),
        ("missing", _MISSING, [10, 20]),
        ("forest", _FOREST, [10, 20]),
        ("three_classes", _MULTI_TARGET, [1, 2, 3]),
    ]
    variants = [
        Variant(
            label,
            ((4, 3),),
            _legacy_attributes(trees, "class", classlabels_int64s=classes),
            elem_types=scored,
            outputs=2,
        )
        for label, trees, classes in forests
    ]
    # An ensemble whose leaves all weight one class is the binary case: the reference pairs
    # the score with a second column derived from it, which the transform then decides.
    binary = [_Split(0, "LEQ", 0.5, [(0, 0.25)], [(0, 0.75)])]
    variants += [
        Variant(
            f"binary_{label}",
            ((4, 3),),
            _legacy_attributes(
                binary, "class", classlabels_int64s=classes, **attributes
            ),
            elem_types=scored,
            outputs=2,
        )
        for label, classes, attributes in (
            ("pair", [0, 1], {}),
            ("softmax", [0, 1], {"post_transform": "SOFTMAX"}),
            ("logistic", [0, 1], {"post_transform": "LOGISTIC"}),
            ("softmax_zero", [0, 1], {"post_transform": "SOFTMAX_ZERO"}),
            ("probit", [0, 1], {"post_transform": "PROBIT"}),
            ("three_labels", [7, 8, 9], {}),
            # A single class label, where the reference widens the scores to two columns.
            ("single_label", [1], {}),
            ("single_label_logistic", [1], {"post_transform": "LOGISTIC"}),
        )
    ]
    variants += [
        Variant(
            f"forest_{label}",
            (shape,),
            _legacy_attributes(
                _FOREST, "class", classlabels_int64s=[10, 20], **attributes
            ),
            elem_types=scored,
            outputs=2,
        )
        for label, shape, attributes in (
            *((label, shape, {}) for label, shape in _LEGACY_SHAPES[1:]),
            ("softmax", (4, 3), {"post_transform": "SOFTMAX"}),
            ("logistic", (4, 3), {"post_transform": "LOGISTIC"}),
            ("softmax_zero", (4, 3), {"post_transform": "SOFTMAX_ZERO"}),
            ("base_values", (4, 3), {"base_values": [0.25, -0.5]}),
            (
                "base_values_softmax",
                (4, 3),
                {"base_values": [0.25, -0.5], "post_transform": "SOFTMAX"},
            ),
        )
    ]
    # A probit maps a probability, so it is swept over the forest whose scores are ones.
    variants += [
        Variant(
            f"probabilities_{label}",
            ((4, 3),),
            _legacy_attributes(
                _PROBABILITY_FOREST,
                "class",
                classlabels_int64s=[10, 20],
                **attributes,
            ),
            elem_types=scored,
            outputs=2,
        )
        for label, attributes in (
            ("probit", {"post_transform": "PROBIT"}),
            ("none", {}),
        )
    ]
    return tuple(variants)


def _bare_leaf_attributes(dtype: str) -> dict[str, Any]:
    """A tree that is a single leaf, in the one shape its reference implementation reads.

    The root of such a tree indexes the *leaf* families rather than the node ones, so it is
    only expressible where the two indices coincide -- which is why this one is written out
    rather than built from a tree.
    """
    return {
        "nodes_featureids": [0],
        "nodes_truenodeids": [0],
        "nodes_falsenodeids": [0],
        "nodes_trueleafs": [1],
        "nodes_falseleafs": [1],
        "nodes_modes": numpy_helper.from_array(np.array([0], np.uint8), "nodes_modes"),
        "nodes_splits": numpy_helper.from_array(np.array([0.5], dtype), "nodes_splits"),
        "leaf_targetids": [1],
        "leaf_weights": numpy_helper.from_array(np.array([4.5], dtype), "leaf_weights"),
        "tree_roots": [0],
        "n_targets": 2,
    }


def _tree_ensemble_variants() -> tuple[Variant, ...]:
    """`TreeEnsemble` over the branch tests, the set tests, and every aggregation.

    Every case is generated once per element type rather than once for both: ONNX's own type
    inference requires the splits, the weights and the set members to carry the element type
    of `X`, so a forest's tables belong to the dtype they are swept at.

    A zero-row `X` is deliberately absent. The reference implementation drops each row through
    the trees with `numpy.apply_along_axis`, which refuses an empty axis outright, so there is
    no oracle for one; the legacy pair, which loop over the rows themselves, do sweep it.
    """
    forests = [
        (f"stump_{test.lower()}", trees, 1) for test, trees in _STUMPS.items()
    ] + [
        ("deep", _DEEP, 2),
        ("missing", _MISSING, 2),
        ("forest", _FOREST, 2),
        ("membership", _MEMBERSHIP, 2),
    ]
    aggregations = (
        # Wider than the special-value list, so every float edge reaches the features.
        ("wide", (5, 3), {}),
        ("average", (4, 3), {"aggregate_function": 0}),
        ("sum", (4, 3), {"aggregate_function": 1}),
        ("minimum", (4, 3), {"aggregate_function": 2}),
        ("maximum", (4, 3), {"aggregate_function": 3}),
        ("softmax", (4, 3), {"post_transform": 1}),
        ("explicit_none", (4, 3), {"post_transform": 0}),
    )
    variants: list[Variant] = []
    for elem_type in (TensorProto.FLOAT, TensorProto.DOUBLE):
        dtype = numpy_dtype_name(elem_type)
        variants += [
            Variant(
                f"{label}_{dtype}",
                ((4, 3),),
                _ensemble_attributes(trees, dtype, n_targets=targets),
                elem_types=(elem_type,),
            )
            for label, trees, targets in forests
        ]
        variants += [
            Variant(
                f"forest_{label}_{dtype}",
                (shape,),
                _ensemble_attributes(_FOREST, dtype, n_targets=2, **attributes),
                elem_types=(elem_type,),
            )
            for label, shape, attributes in aggregations
        ]
        variants.append(
            Variant(
                f"bare_leaf_{dtype}",
                ((4, 3),),
                _bare_leaf_attributes(dtype),
                elem_types=(elem_type,),
            )
        )
    return tuple(variants)


# What ONNX's own standard-domain ops say a score transform computes, for the predictors
# whose reference implementation cannot apply one.
_TRANSFORM_EQUIVALENTS: Mapping[Any, tuple[str, Mapping[str, Any]]] = {
    "SOFTMAX": ("Softmax", {"axis": 1}),
    1: ("Softmax", {"axis": 1}),
}


def _as_transformed_scores(case: Case) -> ModelProto:
    """The case's predictor with its score transform spelled out as the op ONNX defines.

    `TreeEnsembleRegressor`'s reference implementation raises for every transform but `NONE`,
    `TreeEnsemble`'s ignores the attribute outright, and the two regressors below raise like
    the first, so none of them can be the oracle for a transformed model. Each is compared
    instead against the untransformed op followed by the standard-domain op the transform is
    defined to be -- which is also what the transform the classifiers' references *do*
    implement computes, and those are compared against the reference directly.
    """
    equivalent = _TRANSFORM_EQUIVALENTS.get(
        case.variant.attributes.get("post_transform")
    )
    if equivalent is None:
        return _model(case)
    untransformed = {
        name: value
        for name, value in case.variant.attributes.items()
        if name != "post_transform"
    }
    model = _model(
        replace(case, variant=replace(case.variant, attributes=untransformed))
    )
    op_type, attributes = equivalent
    model.graph.node[0].output[0] = "scores"
    model.graph.node.append(
        helper.make_node(op_type, ["scores"], ["out0"], name="transform", **attributes)
    )
    return _importing_standard_ops(model)


# --------------------------------------------------------------------------------------
# Support vector machines and linear models
# --------------------------------------------------------------------------------------

# Every one of the four scores dot products, so their sweeps are `ACCUMULATING` for the reason
# Gemm's is: the summation order the reference takes is numpy's and the kernel's is a loop,
# and with an infinity or a dtype extreme among the products the two legitimately disagree.
# What the variants sweep instead is the *shape* of each model -- how many coefficient rows,
# how many class labels, which kernel function, and which of the readings of a row of scores
# each combination lands on.

_PREDICTOR_SHAPES = (("matrix", (4, 3)), ("single_row", (1, 3)), ("empty_rows", (0, 3)))
# One coefficient per feature, and a second row of them for the two-output cases.
_FIRST_ROW = [1.0, 0.0, -1.0]
_SECOND_ROW = [-0.5, 0.5, 0.25]
_TWO_ROWS = _FIRST_ROW + _SECOND_ROW
# Three support vectors over the same three features, and the gamma/coef0/degree triple every
# kernel function reads from.
_SUPPORT_VECTORS = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, -1.0, 0.5, 2.0]
_KERNEL_PARAMS = [0.5, 1.0, 3.0]
_KERNEL_TYPES = ("LINEAR", "POLY", "RBF", "SIGMOID")
# Coefficients small enough, against an intercept of one half, that every score lands inside
# the `[0, 1]` a probit is defined over -- outside it the transform is NaN, which compares
# equal to itself and would leave the case asserting nothing about the arithmetic.
_SMALL_ROW = [0.01, 0.02, -0.01]


def _linear_regressor_variants() -> tuple[Variant, ...]:
    """`LinearRegressor` over one and several targets, and over both intercept layouts."""
    single = {"coefficients": _FIRST_ROW, "intercepts": [0.5]}
    two = {"coefficients": _TWO_ROWS, "intercepts": [0.5, -0.25], "targets": 2}
    return (
        *(Variant(label, (shape,), single) for label, shape in _PREDICTOR_SHAPES),
        Variant("two_targets", ((4, 3),), two),
        Variant(
            "shared_intercept",
            ((4, 3),),
            {"coefficients": _TWO_ROWS, "intercepts": [0.5], "targets": 2},
        ),
        Variant("wide", ((4, 8),), {"coefficients": [0.25] * 8, "intercepts": [0.5]}),
        Variant("explicit_none", ((4, 3),), {**single, "post_transform": "NONE"}),
        Variant("softmax", ((4, 3),), {**two, "post_transform": "SOFTMAX"}),
    )


def _linear_classifier_variants() -> tuple[Variant, ...]:
    """`LinearClassifier` over every reading of a row of scores its reference has.

    One coefficient row per class is what converters emit and what the transforms are swept
    over; a single row against two labels is the paired case, where the score is set against
    its own negation, and a single row against anything else is the thresholded one, where
    the label is decided by which side of zero -- or of one half, once a transform has mapped
    the score onto a probability -- the single column falls.
    """
    binary = {"coefficients": _TWO_ROWS, "intercepts": [0.5, -0.25]}
    labelled = {**binary, "classlabels_ints": [3, 7]}
    single = {"coefficients": _FIRST_ROW, "intercepts": [0.5]}
    three = {
        "coefficients": _TWO_ROWS + [0.25, -0.75, 1.0],
        "intercepts": [0.5, -0.25, 0.0],
        "classlabels_ints": [1, 2, 3],
    }
    probabilities = {
        "coefficients": _SMALL_ROW * 3,
        "intercepts": [0.5, 0.5, 0.5],
        "classlabels_ints": [1, 2, 3],
    }
    return (
        *(
            Variant(label, (shape,), labelled, outputs=2)
            for label, shape in _PREDICTOR_SHAPES
        ),
        *(
            Variant(
                f"binary_{transform.lower()}",
                ((4, 3),),
                {**labelled, "post_transform": transform},
                outputs=2,
            )
            for transform in ("NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO")
        ),
        Variant(
            "shared_intercept",
            ((4, 3),),
            {**labelled, "intercepts": [0.5]},
            outputs=2,
        ),
        Variant(
            "wide", ((4, 8),), {**labelled, "coefficients": [0.25] * 16}, outputs=2
        ),
        Variant("three_classes", ((4, 3),), three, outputs=2),
        Variant(
            "three_classes_softmax",
            ((4, 3),),
            {**three, "post_transform": "SOFTMAX"},
            outputs=2,
        ),
        Variant(
            "probit",
            ((4, 3),),
            {**probabilities, "post_transform": "PROBIT"},
            outputs=2,
        ),
        # `multi_class` is an attribute the reference reads and then ignores.
        Variant("multi_class", ((4, 3),), {**three, "multi_class": 1}, outputs=2),
        Variant("paired", ((4, 3),), {**single, "classlabels_ints": [3, 7]}, outputs=2),
        Variant(
            "paired_logistic",
            ((4, 3),),
            {**single, "classlabels_ints": [3, 7], "post_transform": "LOGISTIC"},
            outputs=2,
        ),
        Variant(
            "single_class", ((4, 3),), {**single, "classlabels_ints": [7]}, outputs=2
        ),
        Variant(
            "single_class_logistic",
            ((4, 3),),
            {**single, "classlabels_ints": [7], "post_transform": "LOGISTIC"},
            outputs=2,
        ),
        # No class labels at all, where a thresholded row is labelled 1 or 0.
        Variant("no_labels", ((4, 3),), single, outputs=2),
    )


def _svm_regressor_variants() -> tuple[Variant, ...]:
    """`SVMRegressor` in both its modes, over every kernel function ONNX defines."""
    linear = {"coefficients": _FIRST_ROW, "rho": [0.25]}
    supports = {
        "coefficients": [1.0, -0.5, 0.25],
        "rho": [0.25],
        "n_supports": 3,
        "support_vectors": _SUPPORT_VECTORS,
        "kernel_params": _KERNEL_PARAMS,
    }
    return (
        *(Variant(label, (shape,), linear) for label, shape in _PREDICTOR_SHAPES),
        Variant("wide", ((4, 8),), {"coefficients": [0.25] * 8, "rho": [0.25]}),
        Variant("one_class", ((4, 3),), {**linear, "one_class": 1}),
        Variant("softmax", ((4, 3),), {**linear, "post_transform": "SOFTMAX"}),
        *(
            Variant(
                f"supports_{kernel.lower()}",
                ((4, 3),),
                {**supports, "kernel_type": kernel},
            )
            for kernel in _KERNEL_TYPES
        ),
        # No `kernel_params` leaves gamma, coef0 and the degree at zero.
        Variant(
            "supports_no_params",
            ((4, 3),),
            {
                name: value
                for name, value in supports.items()
                if name != "kernel_params"
            },
        ),
        Variant("supports_one_class", ((4, 3),), {**supports, "one_class": 1}),
        Variant("supports_empty_rows", ((0, 3),), supports),
        # The reference reads one coefficient per support vector and ignores the rest.
        Variant(
            "supports_spare_coefficients",
            ((4, 3),),
            {**supports, "coefficients": [1.0, -0.5, 0.25, 9.0]},
        ),
    )


def _svm_classifier_variants() -> tuple[Variant, ...]:
    """`SVMClassifier` over both modes, all three label rules and the probability coupling.

    A zero-row `X` is deliberately absent: the reference sizes its score matrix from the first
    row it computes, so with no rows it returns `None` for the scores and there is nothing to
    compare against. The other three ops sweep that shape, over the same two kernels this one
    writes its scores with.
    """
    labels = {"classlabels_ints": [3, 7]}
    linear = {"coefficients": _TWO_ROWS, "rho": [0.25], **labels}
    three = {
        "coefficients": _TWO_ROWS + [0.25, -0.75, 1.0],
        "rho": [0.25],
        "classlabels_ints": [1, 2, 3],
    }
    single = {"coefficients": _FIRST_ROW, "rho": [0.25], "classlabels_ints": [7]}
    supports = {
        "coefficients": [0.5, -0.25, 0.75],
        "rho": [0.25],
        "vectors_per_class": [2, 1],
        "support_vectors": _SUPPORT_VECTORS,
        "kernel_params": _KERNEL_PARAMS,
        **labels,
    }
    trio = {
        "coefficients": [0.5, -0.25, 0.75, 0.1, 0.2, -0.3],
        "rho": [0.25, 0.1, -0.2],
        "vectors_per_class": [1, 1, 1],
        "support_vectors": _SUPPORT_VECTORS,
        "kernel_params": _KERNEL_PARAMS,
        "kernel_type": "LINEAR",
        "classlabels_ints": [1, 2, 3],
    }
    return (
        Variant("linear_binary", ((4, 3),), linear, outputs=2),
        Variant("linear_single_row", ((1, 3),), linear, outputs=2),
        Variant(
            "linear_wide", ((4, 8),), {**linear, "coefficients": [0.25] * 16}, outputs=2
        ),
        *(
            Variant(
                f"linear_binary_{transform.lower()}",
                ((4, 3),),
                {**linear, "post_transform": transform},
                outputs=2,
            )
            for transform in ("NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT")
        ),
        Variant("linear_three_classes", ((4, 3),), three, outputs=2),
        # More than one `rho` takes the plain reading of the winning column, whatever the
        # class labels say.
        Variant(
            "linear_three_rho",
            ((4, 3),),
            {**three, "rho": [0.25, 0.1, -0.2]},
            outputs=2,
        ),
        # A single score against a single class label: the label is the sign of that score,
        # and the row is returned before it ever reaches the transform.
        Variant("linear_single_class", ((4, 3),), single, outputs=2),
        Variant(
            "linear_single_class_logistic",
            ((4, 3),),
            {**single, "post_transform": "LOGISTIC"},
            outputs=2,
        ),
        Variant(
            "linear_single_class_probit",
            ((4, 3),),
            {**single, "post_transform": "PROBIT"},
            outputs=2,
        ),
        *(
            Variant(
                f"supports_{kernel.lower()}",
                ((4, 3),),
                {**supports, "kernel_type": kernel},
                outputs=2,
            )
            for kernel in _KERNEL_TYPES
        ),
        Variant("supports_single_row", ((1, 3),), supports, outputs=2),
        # No coefficient below zero is the one case where a winning vote of at least one half
        # names the second class outright.
        Variant(
            "supports_all_positive",
            ((4, 3),),
            {**supports, "coefficients": [0.5, 0.25, 0.75]},
            outputs=2,
        ),
        *(
            Variant(
                f"supports_{transform.lower()}",
                ((4, 3),),
                {**supports, "post_transform": transform},
                outputs=2,
            )
            for transform in ("LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT")
        ),
        Variant("supports_three_classes", ((4, 3),), trio, outputs=2),
        Variant(
            "supports_three_classes_softmax",
            ((4, 3),),
            {**trio, "post_transform": "SOFTMAX"},
            outputs=2,
        ),
        # Platt scaling, which turns the single decision value of a class pair into the two
        # probabilities the row is then scored with.
        Variant(
            "supports_probabilities",
            ((4, 3),),
            {**supports, "prob_a": [-1.5], "prob_b": [0.25]},
            outputs=2,
        ),
        Variant(
            "supports_probabilities_softmax",
            ((4, 3),),
            {
                **supports,
                "prob_a": [-1.5],
                "prob_b": [0.25],
                "post_transform": "SOFTMAX",
            },
            outputs=2,
        ),
    )


def _as_float_predictor(case: Case) -> ModelProto:
    """A predictor's oracle: float32 operands, and a transform its reference cannot apply."""
    return _cast_inputs_to_float(_as_transformed_scores(case))


SWEEP: dict[tuple[str, str], Sweep] = {
    **{
        ("", op_type): Sweep(Kind.POINTWISE, _UNARY_VARIANTS)
        for op_type in _POINTWISE_UNARY_OPS
    },
    # The reference divides into a 0-d output array, which numpy refuses outright, so a
    # rank-0 Softsign has no oracle to be compared against.
    ("", "Softsign"): Sweep(
        Kind.POINTWISE,
        tuple(variant for variant in _UNARY_VARIANTS if variant.label != "rank_0"),
    ),
    # Negation and the bias are arithmetic, so the integer extremes — where C's overflow is
    # undefined and ONNX's is unstated — stay out of these.
    **{
        ("", op_type): Sweep(Kind.ARITHMETIC, _UNARY_VARIANTS)
        for op_type in ("Abs", "Neg")
    },
    **{
        ("", op_type): Sweep(Kind.ARITHMETIC, _BROADCAST_VARIANTS)
        for op_type in ("Add", "Sub", "Mul")
    },
    **{
        ("", op_type): Sweep(Kind.POINTWISE, _VARIADIC_VARIANTS)
        for op_type in ("Max", "Min")
    },
    **{
        ("", op_type): Sweep(Kind.ARITHMETIC, _VARIADIC_VARIANTS)
        for op_type in ("Mean", "Sum")
    },
    **{
        ("", op_type): Sweep(
            Kind.POINTWISE, _with_attributes(_UNARY_VARIANTS, combinations)
        )
        for op_type, combinations in _ACTIVATION_ATTRIBUTES.items()
    },
    # Comparisons and boolean logic: nothing they compute can leave a dtype's range, so
    # every operand sweeps its extremes.
    **{
        ("", op_type): Sweep(Kind.POINTWISE, _BROADCAST_VARIANTS)
        for op_type in (
            "And",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
            "Or",
            "Xor",
        )
    },
    **{
        ("", op_type): Sweep(Kind.POINTWISE, _UNARY_VARIANTS)
        for op_type in ("BitwiseNot", "IsNaN", "Not")
    },
    ("", "BitShift"): Sweep(Kind.POINTWISE, _BIT_SHIFT_VARIANTS),
    ("", "BitCast"): Sweep(Kind.POINTWISE, _bitcast_variants()),
    ("", "Cast"): Sweep(Kind.POINTWISE, _cast_variants()),
    ("", "IsInf"): Sweep(
        Kind.POINTWISE, _with_attributes(_UNARY_VARIANTS, _IS_INF_ATTRIBUTES)
    ),
    ("", "Where"): Sweep(
        Kind.POINTWISE,
        _SELECT_VARIANTS,
        type_operand=1,
        operand_types={0: TensorProto.BOOL},
    ),
    ("", "Div"): Sweep(Kind.ARITHMETIC, _BROADCAST_VARIANTS, {1: Domain.NONZERO}),
    ("", "Mod"): Sweep(Kind.ARITHMETIC, _MOD_VARIANTS, {1: Domain.NONZERO}),
    ("", "Pow"): Sweep(
        Kind.ARITHMETIC, _BROADCAST_VARIANTS, {1: Domain.SMALL_EXPONENT}
    ),
    ("", "PRelu"): Sweep(Kind.ARITHMETIC, _UNIDIRECTIONAL_VARIANTS),
    ("", "Shrink"): Sweep(
        Kind.ARITHMETIC, _with_attributes(_UNARY_VARIANTS, _SHRINK_ATTRIBUTES)
    ),
    ("", "Clip"): Sweep(Kind.POINTWISE, _CLIP_VARIANTS),
    ("", "Dropout"): Sweep(Kind.POINTWISE, _DROPOUT_VARIANTS),
    ("", "Gemm"): Sweep(Kind.ACCUMULATING, _GEMM_VARIANTS),
    ("", "MatMul"): Sweep(Kind.ACCUMULATING, _MATMUL_VARIANTS),
    # Det multiplies the pivots of a factorization, which is an accumulation of its own: a
    # special value anywhere in a matrix decides the whole determinant, and by which pivots
    # were chosen rather than by the spec.
    ("", "Det"): Sweep(Kind.ACCUMULATING, _DET_VARIANTS),
    # Einsum sums a product per result element in an order the equation does not fix, so it
    # accumulates for the reason Gemm does.
    ("", "Einsum"): Sweep(Kind.ACCUMULATING, _EINSUM_VARIANTS),
    # The transforms sum every sample of an axis into every bin, so they accumulate for the
    # reason Gemm does — and more so: the reference runs numpy's FFT at the operand's own
    # precision, whose butterfly leaves a residue of its own where a bin cancels to zero.
    # What each reads as configuration rather than as data — the length, the axis, the step —
    # is carried as an initializer, since a compiler with no run-time shapes needs it fixed.
    ("", "DFT"): Sweep(
        Kind.ACCUMULATING,
        _DFT_VARIANTS,
        operand_types={1: TensorProto.INT64, 2: TensorProto.INT64},
        constant_operands=(1, 2),
    ),
    ("", "STFT"): Sweep(
        Kind.ACCUMULATING,
        _STFT_VARIANTS,
        operand_types={1: TensorProto.INT64, 3: TensorProto.INT64},
        constant_operands=(1, 3),
    ),
    # A convolution sums a whole window of products per output element, in an order the spec
    # leaves open, so it accumulates for the reason Gemm does.
    ("", "Conv"): Sweep(Kind.ACCUMULATING, _CONV_VARIANTS),
    ("", "ConvTranspose"): Sweep(Kind.ACCUMULATING, _CONV_TRANSPOSE_VARIANTS),
    ("", "DeformConv"): Sweep(Kind.ACCUMULATING, _DEFORM_CONV_VARIANTS),
    # The affine maps are elementwise arithmetic, so they sweep the float edges; what bounds
    # them is only what the reference leaves undefined, which is the conversion at the end of
    # a quantization -- hence the restriction on its operand and the pinned scales its table
    # records. The type swept is the grid rather than the operand: for QuantizeLinear that is
    # the zero point's, which is what decides the result's, and its operand and scale are
    # pinned to the one float type ONNX allows them both to carry at every revision claimed.
    ("", "QuantizeLinear"): Sweep(
        Kind.ARITHMETIC,
        _QUANTIZE_VARIANTS,
        {0: Domain.CONVERTIBLE},
        type_operand=2,
        operand_types={0: TensorProto.FLOAT, 1: TensorProto.FLOAT},
    ),
    ("", "DequantizeLinear"): Sweep(
        Kind.POINTWISE,
        _DEQUANTIZE_VARIANTS,
        operand_types={1: TensorProto.FLOAT},
    ),
    # The quantized products sum a window or a row of products, so they accumulate for the
    # reason Gemm does. Their bias is `int32` whatever grid the operands stand on.
    ("", "MatMulInteger"): Sweep(Kind.ACCUMULATING, _MATMUL_INTEGER_VARIANTS),
    ("", "QLinearMatMul"): Sweep(
        Kind.ACCUMULATING,
        _qlinear_matmul_variants(),
        operand_types={
            1: TensorProto.FLOAT,
            4: TensorProto.FLOAT,
            6: TensorProto.FLOAT,
        },
    ),
    ("", "ConvInteger"): Sweep(Kind.ACCUMULATING, _conv_integer_variants()),
    ("", "QLinearConv"): Sweep(
        Kind.ACCUMULATING,
        _qlinear_conv_variants(),
        operand_types={
            1: TensorProto.FLOAT,
            4: TensorProto.FLOAT,
            6: TensorProto.FLOAT,
            8: TensorProto.INT32,
        },
    ),
    # AveragePool and LpPool sum a window, so they accumulate for the reason Gemm does.
    # MaxPool and GlobalMaxPool only compare — but the reference evaluator runs them through
    # whichever of its two pooling implementations the attributes select, and the two disagree
    # about a NaN in the window (one drops it, the other lets it win where it comes first) and
    # about which of a window's equal maxima is reported, which is what tells -0.0 and 0.0
    # apart. So the whole family is fed the finite values every path agrees on, and the
    # special ones rest on the backend corpus.
    ("", "AveragePool"): Sweep(Kind.ACCUMULATING, _AVERAGE_POOL_VARIANTS),
    ("", "LpPool"): Sweep(Kind.ACCUMULATING, _LP_POOL_VARIANTS),
    ("", "MaxPool"): Sweep(Kind.ACCUMULATING, _MAX_POOL_VARIANTS),
    ("", "GlobalAveragePool"): Sweep(Kind.ACCUMULATING, _GLOBAL_VARIANTS),
    # The other two global folds are run against the windowed op each is defined to equal.
    # GlobalLpPool has to be: the reference evaluator implements it nowhere. GlobalMaxPool it
    # has one for, but that one reduces `range(rank - 2, rank)` — the spatial axes only for a
    # 4-D operand, and the wrong axes for every other rank — so it is no oracle for the ranks
    # this kernel equally serves, and nothing else in the corpus is either.
    ("", "GlobalMaxPool"): Sweep(
        Kind.ACCUMULATING,
        _GLOBAL_MAX_VARIANTS,
        equivalent_model=_as_windowed_pooling("MaxPool"),
    ),
    ("", "GlobalLpPool"): Sweep(
        Kind.ACCUMULATING,
        _GLOBAL_LP_VARIANTS,
        equivalent_model=_as_windowed_pooling("LpPool"),
    ),
    # MaxUnpool moves elements without computing any, so every special value goes in and has
    # to come back out unchanged; the positions it moves them to are int64 whatever they hold.
    ("", "MaxUnpool"): Sweep(
        Kind.POINTWISE, _MAX_UNPOOL_VARIANTS, operand_types={1: TensorProto.INT64}
    ),
    # The reductions and the running folds accumulate a whole group in an order the spec does
    # not fix, so they carry ACCUMULATING for the same reason Gemm does. ReduceMax, ReduceMin,
    # ArgMax, ArgMin and Hardmax compare rather than accumulate — no order changes what they
    # select — so those sweep every special value their dtype has.
    ("", "ReduceSum"): _reduction_sweep(Kind.ACCUMULATING, *_SUM_VERSIONS),
    ("", "ReduceMean"): _reduction_sweep(
        Kind.ACCUMULATING,
        *_REDUCTION_VERSIONS,
        # numpy's mean of nothing is a 0/0 it casts unsafely to the element type, which is a
        # value only the floating-point families have.
        empty_group_types=_FLOAT_ELEM_TYPES,
    ),
    ("", "ReduceProd"): _reduction_sweep(
        Kind.ACCUMULATING, *_REDUCTION_VERSIONS, factors=True
    ),
    ("", "ReduceL1"): _reduction_sweep(Kind.ACCUMULATING, *_REDUCTION_VERSIONS),
    ("", "ReduceL2"): _reduction_sweep(Kind.ACCUMULATING, *_REDUCTION_VERSIONS),
    # The reference evaluator is no oracle for these three on the integer families: it raises
    # outright for the two logarithmic ones, and returns a ReduceSumSquare whose dtype
    # disagrees with ONNX's own type inference. Their integer kernels are emitted the way the
    # op is defined; the sweep stops where the oracle does.
    ("", "ReduceLogSum"): _reduction_sweep(
        Kind.ACCUMULATING, *_REDUCTION_VERSIONS, elem_types=_FLOAT_ELEM_TYPES
    ),
    ("", "ReduceLogSumExp"): _reduction_sweep(
        Kind.ACCUMULATING, *_REDUCTION_VERSIONS, elem_types=_FLOAT_ELEM_TYPES
    ),
    ("", "ReduceSumSquare"): _reduction_sweep(
        Kind.ACCUMULATING, *_REDUCTION_VERSIONS, elem_types=_FLOAT_ELEM_TYPES
    ),
    ("", "ReduceMax"): _reduction_sweep(
        Kind.POINTWISE, *_EXTREMUM_VERSIONS, empty_group_types=_NON_BOOL_TYPES
    ),
    ("", "ReduceMin"): _reduction_sweep(
        Kind.POINTWISE, *_EXTREMUM_VERSIONS, empty_group_types=_NON_BOOL_TYPES
    ),
    **{
        ("", op_type): Sweep(Kind.POINTWISE, _ARG_VARIANTS)
        for op_type in ("ArgMax", "ArgMin")
    },
    **{
        ("", op_type): Sweep(Kind.ACCUMULATING, _ALONG_AXIS_VARIANTS)
        for op_type in ("LogSoftmax", "Softmax")
    },
    ("", "Hardmax"): Sweep(Kind.POINTWISE, _ALONG_AXIS_VARIANTS),
    ("", "CumSum"): Sweep(
        Kind.ACCUMULATING,
        _CUMULATIVE_VARIANTS,
        operand_types={1: TensorProto.INT64},
    ),
    ("", "CumProd"): Sweep(
        Kind.ACCUMULATING,
        _CUMULATIVE_VARIANTS,
        {0: Domain.SMALL_FACTOR},
        operand_types={1: TensorProto.INT64},
    ),
    ("", "BatchNormalization"): Sweep(
        Kind.ACCUMULATING, _batch_variants(), {4: Domain.NONNEGATIVE}
    ),
    ("", "LayerNormalization"): Sweep(Kind.ACCUMULATING, _layer_variants()),
    ("", "RMSNormalization"): Sweep(Kind.ACCUMULATING, _rms_variants()),
    # The labels index the class axis, so they do not range over the element type the logits
    # are swept at; the weights do, being read as one coefficient per class.
    ("", "SoftmaxCrossEntropyLoss"): Sweep(
        Kind.ACCUMULATING, _sce_variants(), operand_types={1: TensorProto.INT64}
    ),
    ("", "InstanceNormalization"): Sweep(Kind.ACCUMULATING, _INSTANCE_VARIANTS),
    ("", "GroupNormalization"): Sweep(Kind.ACCUMULATING, _GROUP_VARIANTS),
    ("", "LpNormalization"): Sweep(Kind.ACCUMULATING, _lp_variants()),
    ("", "MeanVarianceNormalization"): Sweep(Kind.ACCUMULATING, _mvn_variants()),
    ("", "LRN"): Sweep(Kind.ACCUMULATING, _LRN_VARIANTS),
    # The views move elements without computing any, so every special value their dtype has
    # goes in and has to come back out unchanged, signed zeros included.
    ("", "Transpose"): Sweep(Kind.POINTWISE, _TRANSPOSE_VARIANTS),
    ("", "Concat"): Sweep(Kind.POINTWISE, _CONCAT_VARIANTS),
    ("", "Flatten"): Sweep(Kind.POINTWISE, _FLATTEN_VARIANTS),
    # The operand describing the result's shape is carried in the model as an initializer:
    # one a graph computes at run time makes the result's shape depend on input data, which
    # the compiler refuses by design.
    **{
        ("", op_type): Sweep(
            Kind.POINTWISE,
            variants,
            operand_types={1: TensorProto.INT64},
            constant_operands=(1,),
        )
        for op_type, variants in (
            ("Reshape", _RESHAPE_VARIANTS),
            ("Squeeze", _SQUEEZE_VARIANTS),
            ("Unsqueeze", _UNSQUEEZE_VARIANTS),
            ("Split", _SPLIT_VARIANTS),
            ("Expand", _EXPAND_VARIANTS),
            ("Tile", _TILE_VARIANTS),
        )
    },
    ("", "Slice"): Sweep(
        Kind.POINTWISE,
        _SLICE_VARIANTS,
        operand_types=dict.fromkeys((1, 2, 3, 4), TensorProto.INT64),
        constant_operands=(1, 2, 3, 4),
    ),
    # The gathering ops move elements without computing any, so every special value their
    # dtype has goes in and has to come back out unchanged. Their indices are read at run
    # time — the result's shape follows from the operands' shapes alone — so they are fed
    # rather than carried in the model.
    **{
        ("", op_type): Sweep(
            Kind.POINTWISE,
            variants,
            operand_types={1: TensorProto.INT64},
        )
        for op_type, variants in (
            ("Gather", _GATHER_VARIANTS),
            ("GatherElements", _GATHER_ELEMENTS_VARIANTS),
            ("GatherND", _GATHER_ND_VARIANTS),
        )
    },
    # The scattering ops write elements without computing any, so every special value their
    # dtype has goes in and has to come back out unchanged — except where `reduction` folds
    # an update into the element already there, which the variants carrying one restrict for.
    # Their indices are read at run time: the result's shape is the operand's own, whatever
    # they hold.
    **{
        ("", op_type): Sweep(
            Kind.POINTWISE,
            variants,
            operand_types={1: TensorProto.INT64},
        )
        for op_type, variants in (
            ("ScatterElements", _SCATTER_ELEMENTS_VARIANTS),
            ("ScatterND", _SCATTER_ND_VARIANTS),
        )
    },
    ("", "Scatter"): Sweep(
        Kind.POINTWISE,
        _SCATTER_VARIANTS,
        operand_types={1: TensorProto.INT64},
        equivalent_model=_as_scatter_elements,
    ),
    ("", "TensorScatter"): Sweep(
        Kind.POINTWISE,
        _TENSOR_SCATTER_VARIANTS,
        operand_types={2: TensorProto.INT64},
    ),
    ("", "Pad"): Sweep(
        Kind.POINTWISE,
        _PAD_VARIANTS,
        operand_types={1: TensorProto.INT64, 3: TensorProto.INT64},
        constant_operands=(1, 3),
    ),
    ("", "OneHot"): Sweep(
        Kind.POINTWISE,
        _ONE_HOT_VARIANTS,
        # The output — and so the sweep's element types — is the type of the two values the
        # op selects between; the indices and the depth are typed independently of it.
        type_operand=2,
        operand_types={0: TensorProto.INT64, 1: TensorProto.INT64},
        constant_operands=(1,),
    ),
    ("", "EyeLike"): Sweep(Kind.POINTWISE, _EYE_LIKE_VARIANTS),
    ("", "Trilu"): Sweep(
        Kind.POINTWISE, _TRILU_VARIANTS, operand_types={1: TensorProto.INT64}
    ),
    ("", "ReverseSequence"): Sweep(
        Kind.POINTWISE,
        _REVERSE_SEQUENCE_VARIANTS,
        operand_types={1: TensorProto.INT64},
    ),
    # A recurrent layer sums a whole row of products per gate and then carries the result
    # forward through every remaining step, so it accumulates for the reason Gemm does,
    # several times over. Its lengths are int32 by ONNX's own type constraint whatever the
    # sweep's element type is.
    ("", "LSTM"): Sweep(
        Kind.ACCUMULATING, _LSTM_VARIANTS, operand_types={4: TensorProto.INT32}
    ),
    ("", "GRU"): Sweep(
        Kind.ACCUMULATING, _GRU_VARIANTS, operand_types={4: TensorProto.INT32}
    ),
    ("", "RNN"): Sweep(
        Kind.ACCUMULATING, _RNN_VARIANTS, operand_types={4: TensorProto.INT32}
    ),
    # LinearAttention sums a whole key dimension per state cell and per answer, and carries
    # the state forward through every remaining token, so it accumulates for the reason the
    # recurrent layers do. Its state operand is typed independently of its activations by
    # ONNX, but the compiler serves the op at one element type and the sweep only has that
    # one to offer either of them.
    ("", "LinearAttention"): Sweep(Kind.ACCUMULATING, _LINEAR_ATTENTION_VARIANTS),
    # A resize sums a filter's worth of products per output element, in an order the spec
    # leaves open, so it accumulates for the reason Gemm does. Its three operands are typed
    # by ONNX rather than by the sweep: the scales are float and the sizes int64 whatever
    # the data holds, and the region defaults to float where a variant does not ask for
    # another floating-point type.
    ("", "Resize"): Sweep(
        Kind.ACCUMULATING,
        _RESIZE_VARIANTS,
        operand_types={
            1: TensorProto.FLOAT,
            2: TensorProto.FLOAT,
            3: TensorProto.INT64,
        },
        constant_operands=(1, 2, 3),
    ),
    ("", "Upsample"): Sweep(
        Kind.ACCUMULATING,
        _UPSAMPLE_VARIANTS,
        operand_types={1: TensorProto.FLOAT},
        constant_operands=(1,),
        equivalent_model=_as_resize,
    ),
    # The block shuffles move elements without computing any, so every special value their
    # dtype has goes in and has to come back out unchanged, signed zeros included.
    ("", "DepthToSpace"): Sweep(Kind.POINTWISE, _DEPTH_TO_SPACE_VARIANTS),
    ("", "SpaceToDepth"): Sweep(Kind.POINTWISE, _SPACE_TO_DEPTH_VARIANTS),
    # Col2Im sums the blocks that reach an image position, in the order ONNX's own reference
    # accumulates them, so the float edges are compared rather than left out: arithmetic, not
    # accumulation. Its two extent operands are int64 whatever the data holds.
    ("", "Col2Im"): Sweep(
        Kind.ARITHMETIC,
        _COL2IM_VARIANTS,
        operand_types={1: TensorProto.INT64, 2: TensorProto.INT64},
        constant_operands=(1, 2),
    ),
    # The samplers weight the elements around a coordinate and sum them, in an order the
    # spec leaves open, so they accumulate for the reason Gemm does.
    ("", "GridSample"): Sweep(
        Kind.ACCUMULATING,
        _GRID_SAMPLE_VARIANTS,
        operand_types={1: TensorProto.FLOAT},
    ),
    ("", "AffineGrid"): Sweep(
        Kind.ACCUMULATING,
        _AFFINE_GRID_VARIANTS,
        operand_types={1: TensorProto.INT64},
        constant_operands=(1,),
    ),
    ("", "RoiAlign"): Sweep(
        Kind.ACCUMULATING,
        _ROI_ALIGN_VARIANTS,
        operand_types={2: TensorProto.INT64},
    ),
    # MaxRoiPool compares rather than accumulates, so it sweeps every special value its dtype
    # has. ONNX ships neither a reference implementation nor a node test for it — it is the
    # one registered op with no ONNX-published oracle at all — so the expected values come
    # from onnxruntime, the second oracle the compiler's parity test already stands on, run
    # on the same node at the newest opset onnxruntime serves it at.
    ("", "MaxRoiPool"): Sweep(
        Kind.POINTWISE,
        _MAX_ROI_POOL_VARIANTS,
        equivalent_model=_at_the_oracle_opset,
        oracle=_onnxruntime_outputs,
    ),
    # TopK compares rather than accumulates — no summation order changes what it selects —
    # so it sweeps every special value its dtype has, NaN included. `k` is the extent of the
    # result's axis, which makes it configuration the model has to carry.
    ("", "TopK"): Sweep(
        Kind.POINTWISE,
        _TOP_K_VARIANTS,
        operand_types={1: TensorProto.INT64},
        constant_operands=(1,),
    ),
    # Attention sums a head's worth of products per score and a whole key row per output
    # element, so it accumulates for the reason Gemm does; the special values its mask paths
    # turn on -- the -inf that masks a column out, and the NaN a boolean mask under
    # `is_causal` poisons a row with -- are pinned by the variants that need them rather than
    # drawn. Only revision 24 is generated: the evaluator is version-faithful for the newest
    # revision alone, and 23 rests on the 63 corpus models that import it. Its key lengths
    # are int64 by ONNX's own type constraint whatever the tensors hold.
    ("", "Attention"): Sweep(
        Kind.ACCUMULATING,
        _ATTENTION_VARIANTS,
        operand_types={6: TensorProto.INT64},
    ),
    # RotaryEmbedding computes two products and one sum per rotated pair, in the order the
    # reference writes them, so the float edges are compared rather than left out: arithmetic,
    # not accumulation. Its positions index the caches, so they are pinned per variant.
    ("", "RotaryEmbedding"): Sweep(
        Kind.ARITHMETIC,
        _ROTARY_VARIANTS,
        operand_types={3: TensorProto.INT64},
    ),
    # TfIdfVectorizer compares each token against a pool and counts the matches, so no value
    # it reads is computed with and every dtype extreme goes into the one variant that draws.
    ("", "TfIdfVectorizer"): Sweep(
        Kind.POINTWISE, _TFIDF_VARIANTS, equivalent_model=_with_unit_weights
    ),
    # The ONNX-ML preprocessing ops. The three whose schema declares a `tensor(float)` result
    # take the oracle on the model that casts their input to that float, for the reason
    # `_with_float_inputs` records; the rest are compared against the evaluator directly.
    #
    # Scaler subtracts and multiplies once per element, so the float edges are compared rather
    # than left out; no integer operand can overflow, since every one of them is converted to
    # float before anything is computed with it.
    (ML_DOMAIN, "Scaler"): Sweep(
        Kind.POINTWISE, _SCALER_VARIANTS, equivalent_model=_with_float_inputs
    ),
    # Normalizer sums a row before dividing by it, which is the accumulation Gemm's sweep
    # keeps the extremes out of; the rows that carry the special values are pinned instead.
    (ML_DOMAIN, "Normalizer"): Sweep(
        Kind.ACCUMULATING, _NORMALIZER_VARIANTS, equivalent_model=_with_float_inputs
    ),
    # FeatureVectorizer copies and pads, so nothing it computes can leave a dtype's range.
    (ML_DOMAIN, "FeatureVectorizer"): Sweep(
        Kind.POINTWISE, _VECTORIZER_VARIANTS, equivalent_model=_with_float_inputs
    ),
    **{
        (ML_DOMAIN, op_type): Sweep(Kind.POINTWISE, variants)
        for op_type, variants in (
            ("Binarizer", _BINARIZER_VARIANTS),
            ("Imputer", _IMPUTER_VARIANTS),
            ("LabelEncoder", _LABEL_ENCODER_VARIANTS),
            ("OneHotEncoder", _ONE_HOT_ENCODER_VARIANTS),
        )
    },
    # The index operand is int64 by ONNX's own type constraint whatever the data holds.
    (ML_DOMAIN, "ArrayFeatureExtractor"): Sweep(
        Kind.POINTWISE, _EXTRACTOR_VARIANTS, operand_types={1: TensorProto.INT64}
    ),
    # The tree ensembles. Nothing they compute is a function of the input's magnitude -- a
    # feature is compared against a split and the scores are sums of the weights the
    # attributes carry -- so the whole special-value list goes into `X`, NaN included, where
    # the missing-value flag decides which branch it takes.
    (ML_DOMAIN, "TreeEnsembleRegressor"): Sweep(
        Kind.POINTWISE,
        _regressor_variants(),
        equivalent_model=_as_transformed_scores,
    ),
    (ML_DOMAIN, "TreeEnsembleClassifier"): Sweep(
        Kind.POINTWISE, _classifier_variants()
    ),
    (ML_DOMAIN, "TreeEnsemble"): Sweep(
        Kind.POINTWISE,
        _tree_ensemble_variants(),
        equivalent_model=_as_transformed_scores,
    ),
    # The support vector machines and the linear models, which score in float32 whatever they
    # are handed and are therefore compared on the model whose operands already are that
    # float. The two regressors' references raise for any transform but `NONE`, so theirs is
    # spelled out as the standard-domain op it is defined to be; the classifiers' references
    # apply the transforms themselves — including the one case where a row is deliberately
    # returned without one — and are compared directly.
    **{
        (ML_DOMAIN, op_type): Sweep(
            Kind.ACCUMULATING, variants, equivalent_model=_as_float_predictor
        )
        for op_type, variants in (
            ("LinearRegressor", _linear_regressor_variants()),
            ("SVMRegressor", _svm_regressor_variants()),
        )
    },
    **{
        (ML_DOMAIN, op_type): Sweep(
            Kind.ACCUMULATING, variants, equivalent_model=_with_float_inputs
        )
        for op_type, variants in (
            ("LinearClassifier", _linear_classifier_variants()),
            ("SVMClassifier", _svm_classifier_variants()),
        )
    },
}


@dataclass(frozen=True)
class Case:
    """One generated model: an op at one revision, one element type, one variant."""

    domain: str
    op_type: str
    version: int
    elem_type: int
    kind: Kind
    variant: Variant

    @property
    def dtype(self) -> Any:
        return np.dtype(numpy_dtype_name(self.elem_type))

    @property
    def id(self) -> str:
        return f"{self.op_type}-{self.version}-{self.dtype.name}-{self.variant.label}"

    def __str__(self) -> str:
        shapes = ", ".join(
            "omitted" if shape is None else str(list(shape))
            for shape in self.variant.shapes
        )
        attributes = (
            ", ".join(
                f"{name}={value}"
                for name, value in sorted(self.variant.attributes.items())
            )
            or "none"
        )
        return (
            f"`{self.op_type}` (domain `{display_domain(self.domain)}`) at opset version "
            f"{self.version}, dtype `{self.dtype.name}`, variant `{self.variant.label}`, "
            f"operand shapes {shapes}, attributes {attributes}, seed {SEED}"
        )


def _faithful_revisions(domain: str, op_type: str) -> tuple[int, ...]:
    """Registered revisions of the op the reference evaluator is a valid oracle for."""
    return tuple(
        version
        for version in KERNELS.registered_versions(domain, op_type)
        if evaluator_is_version_faithful(domain, op_type, version)
    )


@cache
def _cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for domain, op_type in KERNELS.registered_ops():
        sweep = SWEEP.get((domain, op_type))
        if sweep is None:
            # The acceptance-rule test reports this; skipping keeps that one failure from
            # multiplying into a collection error per missing case.
            continue
        for version in _faithful_revisions(domain, op_type):
            schema = get_schema(op_type, version, domain)
            for elem_type in _swept_element_types(schema, sweep.type_operand):
                cases.extend(
                    Case(domain, op_type, version, elem_type, sweep.kind, variant)
                    for variant in sweep.variants
                    if _variant_applies(schema, version, elem_type, variant)
                )
    return tuple(cases)


def _variant_applies(
    schema: OpSchema, version: int, elem_type: int, variant: Variant
) -> bool:
    if variant.versions is not None and version not in variant.versions:
        return False
    if variant.elem_types is not None and elem_type not in variant.elem_types:
        return False
    return _schema_takes(schema, variant)


def _swept_element_types(schema: OpSchema, operand: int) -> tuple[int, ...]:
    """Element types the schema allows for the sweep's type operand and the compiler supports.

    Read off the schema's type constraints rather than listed by hand, so a dtype ONNX adds
    to an op is swept from the moment the installed package defines it.
    """
    allowed = _allowed_type_strings(schema, operand)
    return tuple(
        elem_type for elem_type in sorted(C_TYPES) if _type_string(elem_type) in allowed
    )


def _allowed_type_strings(schema: OpSchema, operand: int) -> frozenset[str]:
    type_str = schema.inputs[operand].type_str
    for constraint in schema.type_constraints:
        if constraint.type_param_str == type_str:
            return frozenset(constraint.allowed_type_strs)
    return frozenset({type_str})


def _type_string(elem_type: int) -> str:
    return f"tensor({TensorProto.DataType.Name(elem_type).lower()})"


def _schema_takes(schema: OpSchema, variant: Variant) -> bool:
    """Whether the schema at this version takes exactly the operands the variant declares."""
    variadic = (
        schema.inputs
        and schema.inputs[-1].option == OpSchema.FormalParameterOption.Variadic
    )
    if len(variant.shapes) > len(schema.inputs) and not variadic:
        return False
    provided = {
        index for index, shape in enumerate(variant.shapes) if shape is not None
    }
    return all(
        index in provided
        for index, formal in enumerate(schema.inputs)
        if formal.option == OpSchema.FormalParameterOption.Single
    )


# --------------------------------------------------------------------------------------
# Generating a case: the model, and the values fed to it
# --------------------------------------------------------------------------------------


def _model(case: Case) -> ModelProto:
    """A single-node model importing exactly the opset the case's kernel revision claims.

    The output carries no declared type: shape inference derives it, so the generated model
    states nothing about the op's result that could disagree with the oracle.
    """
    names = [
        "" if shape is None else f"in{index}"
        for index, shape in enumerate(case.variant.shapes)
    ]
    while names and not names[-1]:
        names.pop()
    results = [f"out{index}" for index in range(case.variant.outputs)]
    node = helper.make_node(
        case.op_type,
        names,
        results,
        name="node",
        domain=case.domain,
        **dict(case.variant.attributes),
    )
    constants = SWEEP[(case.domain, case.op_type)].constant_operands
    graph = helper.make_graph(
        [node],
        "sweep",
        [
            helper.make_tensor_value_info(name, _operand_type(case, index), list(shape))
            for index, (name, shape) in enumerate(zip(names, case.variant.shapes))
            if shape is not None and index not in constants
        ],
        [helper.make_empty_tensor_value_info(name) for name in results],
        initializer=[
            numpy_helper.from_array(_operand(case, index, shape), name)
            for index, (name, shape) in enumerate(zip(names, case.variant.shapes))
            if shape is not None and index in constants
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid(case.domain, case.version)]
    )


def _operand_type(case: Case, index: int) -> int:
    """The element type of operand `index`: the case's, unless the variant or sweep pins it."""
    pinned = SWEEP[(case.domain, case.op_type)].operand_types.get(index, case.elem_type)
    return case.variant.operand_types.get(index, pinned)


def _feeds(case: Case) -> dict[str, Any]:
    constants = SWEEP[(case.domain, case.op_type)].constant_operands
    return {
        f"in{index}": _operand(case, index, shape)
        for index, shape in enumerate(case.variant.shapes)
        if shape is not None and index not in constants
    }


def _operand(case: Case, index: int, shape: tuple[int, ...]) -> Any:
    """What operand `index` holds: the value the variant pins, or a seeded draw."""
    sweep = SWEEP[(case.domain, case.op_type)]
    elem_type = _operand_type(case, index)
    pinned = case.variant.values.get(index)
    if pinned is None:
        domains = {**sweep.operand_domains, **case.variant.domains}
        return _values(shape, elem_type, case.kind, index, domains.get(index))
    dtype = numpy_dtype_name(elem_type)
    if isinstance(pinned, (int, float)):
        return np.full(shape, pinned, dtype)
    return np.array(pinned, dtype).reshape(shape)


def _values(
    shape: tuple[int, ...],
    elem_type: int,
    kind: Kind,
    operand: int,
    domain: Domain | None = None,
) -> Any:
    """The operand's values: its dtype's special values first, then seeded random draws.

    The specials are permuted per operand, so a binary op sees them paired up differently
    (NaN against a finite value, +Inf against -Inf) rather than always against themselves.
    """
    generator = np.random.default_rng([SEED, operand])
    dtype = np.dtype(numpy_dtype_name(elem_type))
    size = math.prod(shape)
    specials = generator.permutation(_special_values(dtype, kind))[:size]
    filler = _random_values(generator, size - len(specials), dtype, kind)
    values = np.concatenate([specials, filler]).astype(dtype)
    return _restrict(values, domain).reshape(shape)


def _restrict(values: Any, domain: Domain | None) -> Any:
    """Move an operand into the range the op is defined over."""
    if domain is None:
        return values
    if domain in (Domain.CONVERTIBLE, Domain.CONVERTIBLE_UNSIGNED):
        if values.dtype.kind != "f":
            return values
        # Every integer type of the target's signedness holds this range, so the clipped
        # values convert to any of them; the fractions, the signed zeros and the subnormals
        # the draw carries survive it.
        smallest = -8 if domain is Domain.CONVERTIBLE else 0
        return np.clip(np.nan_to_num(values), smallest, 8)
    if domain is Domain.NONNEGATIVE:
        return np.abs(values)
    if values.dtype.kind not in "iu":
        return values
    if domain is Domain.NONZERO:
        return np.where(values == 0, values.dtype.type(1), values)
    if domain is Domain.SMALL_FACTOR:
        # `% 3 - 1` on an unsigned dtype would wrap a zero into the dtype's maximum.
        return values % 2 if values.dtype.kind == "u" else values % 3 - 1
    return np.abs(values) % 4


def _special_values(dtype: Any, kind: Kind) -> Any:
    if kind is Kind.ACCUMULATING:
        return np.empty(0, dtype=dtype)
    if dtype.kind == "f":
        info = np.finfo(dtype)
        return np.array(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                np.nan,
                np.inf,
                -np.inf,
                info.max,
                -info.max,
                info.tiny,
                -info.tiny,
                info.smallest_subnormal,
                -info.smallest_subnormal,
            ],
            dtype=dtype,
        )
    if dtype == np.bool_:
        return np.array([False, True])
    info = np.iinfo(dtype)
    values = [0, 1, 2] if info.min == 0 else [0, 1, -1]
    if kind is Kind.POINTWISE:
        values += [info.min, info.max, info.min + 1, info.max - 1]
    return np.array(values, dtype=dtype)


def _random_values(generator: Any, count: int, dtype: Any, kind: Kind) -> Any:
    if count <= 0:
        return np.empty(0, dtype=dtype)
    if dtype.kind == "f":
        return generator.normal(size=count).astype(dtype)
    if dtype == np.bool_:
        return generator.integers(0, 2, size=count).astype(dtype)
    info = np.iinfo(dtype)
    low, high = info.min, info.max
    if kind is not Kind.POINTWISE:
        # Bounded so that neither the sum nor the product of two draws can leave this
        # dtype's range.
        limit = min(100, math.isqrt(info.max))
        low, high = max(low, -limit), min(high, limit)
    return generator.integers(low, high, size=count, endpoint=True, dtype=dtype)


# --------------------------------------------------------------------------------------
# Running a case and comparing it against the evaluator
# --------------------------------------------------------------------------------------


def _execute(case: Case, directory: Path) -> tuple[list[Any], list[Any]]:
    """Compile, build and run the case, and run the same model through the evaluator.

    The same model, unless the op's sweep names an equivalent one: an op the evaluator cannot
    be trusted on is run as the op ONNX defines it to be equal to, on the same operands. An
    op the evaluator does not implement at all takes the oracle its sweep names instead.
    """
    model = _model(case)
    feeds = _feeds(case)
    sweep = SWEEP[(case.domain, case.op_type)]
    oracle = model if sweep.equivalent_model is None else sweep.equivalent_model(case)
    with np.errstate(all="ignore"):
        expected = (
            list(ReferenceEvaluator(oracle).run(None, feeds))
            if sweep.oracle is None
            else sweep.oracle(oracle, feeds)
        )
    outputs = compile_onnx(model, directory).load().run(feeds)
    return [outputs[entry.name] for entry in model.graph.output], expected


def _assert_matches(
    case: Case, outputs: Sequence[Any], expected: Sequence[Any]
) -> None:
    try:
        Runner.assert_similar_outputs(expected, outputs, rtol=RTOL, atol=ATOL)
        for got, want in zip(outputs, expected):
            _assert_zero_signs_match(case, got, want)
    except AssertionError as error:
        raise AssertionError(
            f"{case} diverges from the ONNX reference evaluator.\n{error}"
        ) from None


def _assert_zero_signs_match(case: Case, got: Any, want: Any) -> None:
    """-0.0 and 0.0 are `allclose`, so the sign of every zero is compared separately.

    A signed zero is one of the values the sweep feeds in, and a pointwise kernel applies
    the same IEEE operation to the same operand as the reference, so it must come out with
    the same sign. Accumulating kernels are exempt: 0 + (-0) is +0, which makes a sum's zero
    sign a function of the summation order the spec does not fix.
    """
    if case.kind is Kind.ACCUMULATING or want.dtype.kind != "f":
        return
    zeros = want == 0
    np.testing.assert_array_equal(
        np.signbit(got[zeros]),
        np.signbit(want[zeros]),
        err_msg="the sign of a zero differs from the reference",
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.id)
def test_the_kernel_matches_the_reference_evaluator(case, tmp_path):
    _assert_matches(case, *_execute(case, tmp_path))


# --------------------------------------------------------------------------------------
# The suite's own teeth
# --------------------------------------------------------------------------------------


def _relu_that_drops_nan(context: NodeContext) -> NodeEmission:
    """A Relu that is right everywhere except the edge: `x > 0 ? x : 0` sends NaN to zero."""
    source = context.require_input(0)
    result = context.require_output(0)
    element = c_type(result.elem_type)
    name = f"{context.prefix}_diverging_relu_{element}"
    definition = "\n".join(
        [
            f"static void {name}({element}* out, const {element}* in, size_t count)",
            "{",
            "    size_t index;",
            "    for (index = 0; index < count; ++index) {",
            f"        out[index] = in[index] > 0 ? in[index] : ({element})0;",
            "    }",
            "}",
        ]
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(f"{name}({result.expr}, {source.expr}, {result.elem_count}u);",),
    )


def test_a_kernel_that_diverges_on_an_edge_input_is_reported(tmp_path, monkeypatch):
    """Divergence on NaN alone still fails, and the report names what to reproduce it with."""
    case = Case(
        domain="",
        op_type="Relu",
        version=14,
        elem_type=TensorProto.FLOAT,
        kind=Kind.POINTWISE,
        variant=Variant("wide", ((4, 8),)),
    )
    select = KERNELS.select
    monkeypatch.setattr(
        KERNELS,
        "select",
        lambda domain, op_type, version: (
            KernelSpec(domain, op_type, version, _relu_that_drops_nan)
            if op_type == "Relu"
            else select(domain, op_type, version)
        ),
    )

    with pytest.raises(AssertionError) as error:
        _assert_matches(case, *_execute(case, tmp_path))

    message = str(error.value)
    assert "`Relu`" in message
    assert "opset version 14" in message
    assert "float32" in message
    assert str(SEED) in message


def test_a_kernel_that_loses_a_zero_sign_is_reported(tmp_path):
    """The comparison sees -0.0 against 0.0, which `assert_allclose` alone would not."""
    case = Case(
        domain="",
        op_type="Relu",
        version=14,
        elem_type=TensorProto.FLOAT,
        kind=Kind.POINTWISE,
        variant=Variant("wide", ((4, 8),)),
    )
    outputs, expected = _execute(case, tmp_path)
    flipped = [np.where(value == 0, np.float32(-0.0), value) for value in expected]

    with pytest.raises(AssertionError, match="sign of a zero"):
        _assert_matches(case, outputs, flipped)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_generator_feeds_every_float_edge(dtype):
    values = _values((4, 8), _elem_type(dtype), Kind.POINTWISE, 0)
    info = np.finfo(dtype)

    assert np.isnan(values).any()
    assert (values == np.inf).any() and (values == -np.inf).any()
    assert (np.signbit(values) & (values == 0)).any()
    assert (~np.signbit(values) & (values == 0)).any()
    assert (values == info.max).any() and (values == -info.max).any()
    assert (np.abs(values) == info.smallest_subnormal).any()
    assert (np.abs(values) == info.tiny).any()


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint8", "uint64"])
def test_the_generator_feeds_the_integer_extremes(dtype):
    values = _values((4, 8), _elem_type(dtype), Kind.POINTWISE, 0)
    info = np.iinfo(dtype)

    assert (values == info.min).any()
    assert (values == info.max).any()


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int64"])
def test_arithmetic_operands_cannot_overflow_the_dtype(dtype):
    """Integer overflow is undefined in C and undefined by ONNX, so it is never generated."""
    values = _values((8, 8), _elem_type(dtype), Kind.ARITHMETIC, 0).astype(np.int64)
    info = np.iinfo(dtype)

    assert int(np.abs(values).max()) ** 2 <= info.max
    assert int(np.abs(values).max()) * 2 <= info.max


def test_accumulating_operands_are_finite():
    values = _values((4, 8), TensorProto.FLOAT, Kind.ACCUMULATING, 0)

    assert np.isfinite(values).all()


def test_the_generator_reproduces_from_the_seed_and_varies_by_operand():
    """A reported case has to reproduce, and a binary op must not see two identical sides."""
    first = _values((4, 8), TensorProto.FLOAT, Kind.POINTWISE, 0)
    again = _values((4, 8), TensorProto.FLOAT, Kind.POINTWISE, 0)
    second = _values((4, 8), TensorProto.FLOAT, Kind.POINTWISE, 1)

    assert np.array_equal(first, again, equal_nan=True)
    assert not np.array_equal(first, second, equal_nan=True)


def test_zero_element_shapes_generate_empty_operands():
    values = _values((0, 3), TensorProto.FLOAT, Kind.POINTWISE, 0)

    assert values.shape == (0, 3)
    assert values.size == 0


def test_every_operand_of_every_swept_op_is_fed_the_whole_special_set():
    """Specials are sliced to the operand's element count, so a family of shapes that are
    all smaller than the list feeds only part of it -- and which part is an accident of the
    seed. Each operand needs one variant at least as wide as the list, or the dtype edges
    the sweep exists to cover reach one side of a binary op and not the other."""
    problems = []
    for (domain, op_type), sweep in sorted(SWEEP.items()):
        # float carries the longest list, so a variant wide enough for it is wide enough
        # for every dtype the op takes.
        required = len(_special_values(np.dtype("float64"), sweep.kind))
        arity = max(len(variant.shapes) for variant in sweep.variants)
        for operand in range(arity):
            drawn = [
                variant.shapes[operand]
                for variant in sweep.variants
                if operand < len(variant.shapes)
                and variant.shapes[operand] is not None
                and operand not in variant.values
            ]
            # An operand no variant both draws and shapes — Clip's bounds, Dropout's ratio —
            # has no room for the list to begin with: it is a parameter, and the values that
            # matter for it are the ones its variants pin.
            if not any(drawn):
                continue
            widest = max(math.prod(shape) for shape in drawn)
            if widest < required:
                problems.append(
                    f"`{op_type}` (domain `{display_domain(domain)}`) operand {operand}: "
                    f"its widest variant holds {widest} elements, short of the "
                    f"{required} special values, so the rest are never fed to it."
                )

    assert not problems, "\n".join(problems)


def _elem_type(dtype: str) -> int:
    return helper.np_dtype_to_tensor_dtype(np.dtype(dtype))


# --------------------------------------------------------------------------------------
# The acceptance rule
# --------------------------------------------------------------------------------------


def test_every_registered_op_is_swept_here():
    """Half the acceptance rule: a kernel with no differential coverage is not implemented."""
    covered = {(case.domain, case.op_type) for case in _cases()}
    missing = [
        f"`{op_type}` (domain `{display_domain(domain)}`)"
        for domain, op_type in KERNELS.registered_ops()
        if (domain, op_type) not in covered
    ]

    assert not missing, (
        f"The kernel registry serves {', '.join(missing)}, which this sweep executes no "
        "case for; add the op to SWEEP with the attribute combinations its kernel reads."
    )


def test_the_sweep_claims_exactly_the_revisions_the_evaluator_can_vouch_for():
    """The oracle-validity restriction, in both directions.

    Sweeping a revision the evaluator is not faithful for would compare a kernel against the
    wrong semantics; dropping one it is faithful for would quietly lose coverage.
    """
    swept = {(case.domain, case.op_type, case.version) for case in _cases()}
    provable = {
        (domain, op_type, version)
        for domain, op_type in KERNELS.registered_ops()
        if (domain, op_type) in SWEEP
        for version in _faithful_revisions(domain, op_type)
    }

    assert swept == provable


def _signature(schema: OpSchema) -> Any:
    """Everything about an op's interface a revision could have changed."""
    return (
        schema.doc,
        [(formal.name, formal.type_str, formal.option) for formal in schema.inputs],
        [(formal.name, formal.type_str, formal.option) for formal in schema.outputs],
        {
            name: (attribute.type, attribute.required, str(attribute.default_value))
            for name, attribute in schema.attributes.items()
        },
    )


def test_the_maxroipool_oracle_runs_the_same_op():
    """The one sweep whose oracle is handed a revision other than the kernel's own.

    onnxruntime implements `MaxRoiPool` up to opset 21 and the kernel is registered at the
    revision after it, so running the oracle at 21 only proves anything if ONNX changed
    nothing between the two but the element types it accepts. That is read off the schemas
    rather than taken on trust — and the one type the newer revision adds is one this
    compiler supports at neither.
    """
    (claimed,) = KERNELS.registered_versions("", "MaxRoiPool")
    newer = get_schema("MaxRoiPool", claimed, "")
    older = get_schema("MaxRoiPool", _MAX_ROI_POOL_ORACLE_VERSION, "")

    assert older.since_version < newer.since_version
    assert _signature(newer) == _signature(older)
    added = _allowed_type_strings(newer, 0) - _allowed_type_strings(older, 0)
    assert added == {"tensor(bfloat16)"}
    assert not _allowed_type_strings(older, 0) - _allowed_type_strings(newer, 0)
    assert not added & {_type_string(elem_type) for elem_type in C_TYPES}


def test_the_two_scatter_revisions_are_one_op():
    """The other op claimed at a revision this sweep does not run, and why that is sound.

    Scatter is registered at 9 and at 11, the revision that deprecated it. Only 11 is swept —
    the evaluator is version-faithful for it — while the corpus's own Scatter tests import
    opset 10, which selects 9. Claiming both from one generator is only sound if ONNX changed
    nothing but the deprecation between them, which is read off the schemas rather than taken
    on trust; and the deprecation notice is what points at the op this sweep's oracle runs.
    """
    assert KERNELS.registered_versions("", "Scatter") == [9, 11]
    older = get_schema("Scatter", 9, "")
    newer = get_schema("Scatter", 11, "")

    assert not older.deprecated and newer.deprecated
    # Everything but the document, which gained the notice quoted below.
    assert _signature(older)[1:] == _signature(newer)[1:]
    assert _allowed_type_strings(older, 0) == _allowed_type_strings(newer, 0)
    assert "Please use ScatterElements" in newer.doc


@pytest.mark.parametrize("op_type", ["TreeEnsembleRegressor", "TreeEnsembleClassifier"])
def test_the_legacy_ensemble_revisions_are_one_op(op_type):
    """The remaining ops claimed at revisions this sweep does not run, and why that is sound.

    Both are registered at 1, 3 and 5 while only 5 -- the revision that deprecated them --
    is swept, the evaluator being version-faithful for no earlier one. Opset 1 is what
    scikit-learn's own converter emits, which is where the compiler meets these ops in
    practice and what the parity tests in `test_extra_compiler_trees.py` run at, with
    onnxruntime as their oracle.

    Claiming all three from one generator is only sound if the revisions differ in nothing
    the emitted code reads. That is read off the schemas rather than taken on trust: 3 added
    the `*_as_tensor` attribute families, which the compiler refuses outright, and 5 added a
    deprecation notice.
    """
    assert KERNELS.registered_versions(ML_DOMAIN, op_type) == [1, 3, 5]
    schemas = {
        version: get_schema(op_type, version, ML_DOMAIN) for version in (1, 3, 5)
    }

    assert not schemas[1].deprecated and not schemas[3].deprecated
    assert schemas[5].deprecated
    # The documents differ by those two notices alone, and the interfaces not at all.
    assert _signature(schemas[3])[1:3] == _signature(schemas[1])[1:3]
    assert _signature(schemas[5])[1:] == _signature(schemas[3])[1:]
    assert _allowed_type_strings(schemas[1], 0) == _allowed_type_strings(schemas[5], 0)
    added = set(schemas[3].attributes) - set(schemas[1].attributes)
    assert added and all(name.endswith("_as_tensor") for name in added)
    assert {
        name: attribute
        for name, attribute in _signature(schemas[3])[3].items()
        if name not in added
    } == _signature(schemas[1])[3]


@pytest.mark.skipif(
    not onnx.__version__.startswith(f"{PINNED_ONNX}."),
    reason=(
        f"the conformance corpus and pass list are pinned to onnx {PINNED_ONNX}.*, "
        f"but onnx {onnx.__version__} is installed"
    ),
)
def test_every_registered_op_passes_the_backend_suite_too():
    """The other half: an op no ratcheted corpus test exercises is not implemented either.

    Unless the corpus holds no test of the op that could ever run: every node test for
    `Expand`, `Tile`, `Pad`, `OneHot` and `TopK` hands the op the repeats, shape, pads,
    depth or `k` that decide the shape of its result as a *run-time input*, which makes
    those models uncompilable whatever the kernel does, and every one of them is ledgered
    for exactly that. Or no test of the op at all, as for `GlobalLpPool`. An op in either
    position has no backend evidence to offer either way and rests on the sweep above. Both
    exemptions are derived from the corpus and the ledger rather than listed, so an op leaves
    them the moment a test of it exists and compiles -- at which point the pass list has to
    cover it.
    """
    exercised, ledgered_only, mentioned = _corpus_op_types()
    missing = [
        f"`{op_type}` (domain `{display_domain(domain)}`)"
        for domain, op_type in KERNELS.registered_ops()
        if (domain, op_type) in mentioned
        and (domain, op_type) not in exercised | ledgered_only
    ]

    assert not missing, (
        f"The kernel registry serves {', '.join(missing)}, which no test in "
        f"`{RATCHET_PATH.name}` exercises; the backend conformance suite has to pass for "
        "an op before it counts as implemented."
    )


@pytest.mark.skipif(
    not onnx.__version__.startswith(f"{PINNED_ONNX}."),
    reason=(
        f"the conformance corpus and pass list are pinned to onnx {PINNED_ONNX}.*, "
        f"but onnx {onnx.__version__} is installed"
    ),
)
def test_the_ops_resting_on_the_sweep_alone_are_spelled_out():
    """Which registered ops the backend suite has no passable test for, written down.

    The exemptions above are derived, so nothing would otherwise show when they grew. These
    lists are the record: an op joins the first only because every corpus test of it is
    ledgered and the second only because the corpus has no test of it at all, and leaves as
    soon as a test of it compiles -- at which point the pass list has to cover the op and
    this expectation shrinks in the same change.

    `AffineGrid`, `Col2Im` and `STFT` join the first for the same reason the rest of it is
    there: every node test of them hands the op the extents that decide the shape of its
    result as a run-time input — for `STFT`, the frame step every frame of the result is one
    of, and the frame length each of those transforms. `LabelEncoder` joins it for a reason
    of its own: every node test of it maps to or from a string tensor, which is a run-time
    string whatever the kernel does, and all four are ledgered as exactly that.

    The second list is where the ONNX-ML preprocessing ops, the two legacy tree ensembles and
    the four support-vector and linear predictors sit, alongside `GlobalLpPool` and
    `MaxRoiPool`: the corpus carries a node test for three of the fifteen ONNX-ML ops this
    compiler serves and none at all for the rest, which is the thinness their own targeted
    tests make up for. `GlobalLpPool` and `MaxRoiPool` are the two ops ONNX ships neither a
    node test nor a reference implementation for, and their sweeps above are the whole of what
    covers them: `GlobalLpPool` against the LpPool its own schema defines it to equal, and
    `MaxRoiPool` against onnxruntime.
    """
    exercised, ledgered_only, mentioned = _corpus_op_types()

    assert sorted(set(KERNELS.registered_ops()) & ledgered_only) == [
        ("", "AffineGrid"),
        ("", "Col2Im"),
        ("", "Expand"),
        ("", "OneHot"),
        ("", "Pad"),
        ("", "STFT"),
        ("", "Tile"),
        ("", "TopK"),
        (ML_DOMAIN, "LabelEncoder"),
    ]
    assert sorted(set(KERNELS.registered_ops()) - mentioned) == [
        ("", "GlobalLpPool"),
        ("", "MaxRoiPool"),
        (ML_DOMAIN, "FeatureVectorizer"),
        (ML_DOMAIN, "Imputer"),
        (ML_DOMAIN, "LinearClassifier"),
        (ML_DOMAIN, "LinearRegressor"),
        (ML_DOMAIN, "Normalizer"),
        (ML_DOMAIN, "OneHotEncoder"),
        (ML_DOMAIN, "SVMClassifier"),
        (ML_DOMAIN, "SVMRegressor"),
        (ML_DOMAIN, "Scaler"),
        (ML_DOMAIN, "TreeEnsembleClassifier"),
        (ML_DOMAIN, "TreeEnsembleRegressor"),
    ]
    assert ("", "Add") in exercised
    assert (ML_DOMAIN, "Binarizer") in exercised
    assert (ML_DOMAIN, "TreeEnsemble") in exercised


@cache
def _corpus_op_types() -> tuple[
    frozenset[tuple[str, str]], frozenset[tuple[str, str]], frozenset[tuple[str, str]]
]:
    """The ops the ratcheted corpus tests run, the ops only ledgered tests mention, and
    every op the corpus names at all."""
    ledger = set(json.loads(LEDGER_PATH.read_text(encoding="utf-8")))
    ratchet = {
        line.strip()
        for line in RATCHET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    corpus = {case.name: case for case in load_model_tests(kind="node")}
    assert not ratchet - set(corpus), (
        "the pass list names tests the corpus does not have"
    )

    exercised: set[tuple[str, str]] = set()
    ledgered: set[tuple[str, str]] = set()
    for name, case in corpus.items():
        if name not in ratchet and name not in ledger:
            continue
        assert case.model_dir is not None, f"the corpus test `{name}` ships no model"
        model = onnx.load(Path(case.model_dir) / "model.onnx")
        op_types = {
            (normalize_domain(node.domain), node.op_type) for node in model.graph.node
        }
        (exercised if name in ratchet else ledgered).update(op_types)
    return (
        frozenset(exercised),
        frozenset(ledgered - exercised),
        frozenset(exercised | ledgered),
    )
