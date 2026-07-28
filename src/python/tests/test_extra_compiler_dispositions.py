"""Op-coverage closure: every op of the supported domains is dispositioned, and provably so.

`conformance/dispositions.json` classifies every op schema the pinned `onnx` package defines
for `ai.onnx` and `ai.onnx.ml` as exactly one of

* **native-kernel** — a generator in the kernel registry serves it,
* **function-expansion** — no kernel does, and the compiler inlines the function body ONNX
  defines for it,
* **folding-or-graph-pass** — no kernel does, and a compiler pass resolves the node away
  before dispatch: constant folding, or the ZipMap removal pass,
* **unsupported** — with a reason drawn from the structural part of the conformance
  ledger's own closed set of categories.

Coverage is then a closed property rather than an aspiration: an op the table does not name
— one an `onnx` upgrade adds, say — fails this suite, and so does a table entry the compiler
itself contradicts. Nothing here is taken on the table's word. A kernel claim is checked
against the registry, an expansion claim against the schema's function body, a reason against
the very set the compiler rejects that family of ops from, and every claim to serve an op
without a kernel against evidence that it compiles: a corpus test in the conformance pass
list, or a model this module compiles and runs against the ONNX reference evaluator.

The table and the ledger have to agree, in both directions: an op the table calls unsupported
cannot appear in a corpus test that passes, and a corpus test excluded as `op-not-implemented`
has to hold an op the table does not claim a kernel or a function body for.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import ModelProto, TensorProto, helper, numpy_helper  # noqa: E402
from onnx.backend.test.loader import load_model_tests  # noqa: E402
from onnx.backend.test.runner import Runner  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.errors import CompileError  # noqa: E402
from fnnx.extras.compilers.c.onnx.folding import NONDETERMINISTIC_OPS  # noqa: E402
from fnnx.extras.compilers.c.onnx.kernels import KERNELS  # noqa: E402
from fnnx.extras.compilers.c.onnx.loader import (  # noqa: E402
    ML_DOMAIN,
    SUPPORTED_DOMAINS,
    display_domain,
    max_supported_opset,
    normalize_domain,
)
from fnnx.extras.compilers.c.onnx.verify import (  # noqa: E402
    CONTROL_FLOW_OPS,
    DATA_DEPENDENT_SHAPE_OPS,
)
from fnnx.extras.compilers.c.onnx.zipmap import ZIP_MAP  # noqa: E402

# The ledger's categories, the corpus and the schema set are all one release's; this module
# reads the conformance suite's own definitions rather than restating them, so the two files
# cannot drift into disagreeing about what a category is.
from test_extra_compiler_conformance import (  # noqa: E402
    CODEC_OPS,
    LEDGER_CATEGORIES,
    LEDGER_PATH,
    PINNED_ONNX,
    RATCHET_PATH,
)

TABLE_PATH = Path(__file__).parent / "conformance" / "dispositions.json"

DISPOSITIONS = (
    "native-kernel",
    "function-expansion",
    "folding-or-graph-pass",
    "unsupported",
)

# The ledger categories that cannot be a reason an *op* is unsupported. Two describe a test
# rather than an op: a corpus test is out of scope because of the domain it imports and
# dtype-limited because of the tensors it carries, neither of which is a property of an op the
# supported domains define. The third, `op-not-implemented`, is the milestone category the
# kernel tasks drive down, and admitting it here would reopen as a *disposition* the very
# "not implemented yet" excuse the closed category set exists to refuse: a straggler is to be
# implemented or refused for a structural reason, never parked in the table. What remains is
# exactly the reasons this module re-derives from the compiler or from the op's own schema.
_NOT_OP_REASONS = ("out-of-scope-domain", "unsupported-dtype", "op-not-implemented")
UNSUPPORTED_REASONS = tuple(
    category for category in LEDGER_CATEGORIES if category not in _NOT_OP_REASONS
)

# Where a reason names a family the compiler itself decides membership of, the set it decides
# it from. A table cannot declare that an op is control flow, or a draw: it can only record
# what the compiler already rejects it as.
_REASON_FAMILIES: Mapping[str, frozenset[str]] = {
    "control-flow": CONTROL_FLOW_OPS,
    "data-dependent-shape": DATA_DEPENDENT_SHAPE_OPS,
    "random-op": NONDETERMINISTIC_OPS,
    "external-codec": CODEC_OPS,
}

# The dispositions that claim the compiler serves an op whatever a model does with it, as
# against `folding-or-graph-pass`, which serves the forms a pass can resolve and refuses the
# rest.
_UNCONDITIONAL = ("native-kernel", "function-expansion")

pytestmark = [
    pytest.mark.skipif(
        not onnx.__version__.startswith(f"{PINNED_ONNX}."),
        reason=(
            f"the disposition table is pinned to onnx {PINNED_ONNX}.*, "
            f"but onnx {onnx.__version__} is installed"
        ),
    ),
    pytest.mark.skipif(
        not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
        reason="no system C compiler available",
    ),
]


# --------------------------------------------------------------------------------------
# The table, the schemas it has to cover, and the corpus it has to agree with
# --------------------------------------------------------------------------------------


def _table() -> dict[tuple[str, str], dict[str, Any]]:
    """The checked-in table, keyed by the normalized `(domain, op_type)` the compiler uses."""
    raw = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    return {
        (normalize_domain(domain), op_type): entry
        for domain, entries in raw.items()
        for op_type, entry in entries.items()
    }


@cache
def _schemas() -> dict[tuple[str, str], Any]:
    """Every op the pinned package defines for the supported domains, at its newest schema.

    Enumerated from the package rather than listed, so an op an upgrade adds is one this
    module already asks the table about.
    """
    names = {
        (schema.domain, schema.name)
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.domain in SUPPORTED_DOMAINS
    }
    return {
        (domain, op_type): onnx.defs.get_schema(
            op_type, max_supported_opset(domain), domain
        )
        for domain, op_type in names
    }


def _defines_a_function(domain: str, op_type: str) -> bool:
    """Whether ONNX defines the op as a function body the compiler could inline."""
    schema = _schemas()[(domain, op_type)]
    return bool(
        schema.has_function  # type: ignore[attr-defined]
        or schema.has_context_dependent_function  # type: ignore[attr-defined]
    )


@cache
def _corpus_ops() -> tuple[
    frozenset[tuple[str, str]], dict[str, frozenset[tuple[str, str]]]
]:
    """The ops every ratcheted test runs, and the ops of each `op-not-implemented` exclusion."""
    ledger: dict[str, str] = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    ratchet = {
        line.strip()
        for line in RATCHET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    exercised: set[tuple[str, str]] = set()
    unimplemented: dict[str, frozenset[tuple[str, str]]] = {}
    for case in load_model_tests(kind="node"):
        wanted = case.name in ratchet or ledger.get(case.name) == "op-not-implemented"
        if not wanted:
            continue
        assert case.model_dir is not None, (
            f"the corpus test `{case.name}` ships no model"
        )
        model = onnx.load(Path(case.model_dir) / "model.onnx")
        ops = frozenset(
            (normalize_domain(node.domain), node.op_type) for node in _nodes(model)
        )
        if case.name in ratchet:
            exercised |= ops
        else:
            unimplemented[case.name] = ops
    return frozenset(exercised), unimplemented


def _nodes(model: ModelProto) -> Iterator[Any]:
    def walk(graph: Any) -> Iterator[Any]:
        for node in graph.node:
            yield node
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    yield from walk(attribute.g)
                for subgraph in attribute.graphs:
                    yield from walk(subgraph)

    yield from walk(model.graph)


# --------------------------------------------------------------------------------------
# What is wrong with a table
# --------------------------------------------------------------------------------------


def _problems(table: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    """Everything the compiler, the corpus and the ledger contradict in `table`."""
    return [
        *_coverage_problems(table),
        *_claim_problems(table),
        *_ledger_problems(table),
    ]


def _coverage_problems(table: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    problems = []
    for domain, op_type in sorted(set(_schemas()) - set(table)):
        problems.append(
            f"`{op_type}` (domain `{display_domain(domain)}`) is defined by the installed "
            "`onnx` package but has no disposition."
        )
    for domain, op_type in sorted(set(table) - set(_schemas())):
        problems.append(
            f"`{op_type}` (domain `{display_domain(domain)}`) is dispositioned but the "
            "installed `onnx` package defines no such op in a supported domain."
        )
    return problems


def _claim_problems(table: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    """Every entry against what the compiler itself does with the op."""
    problems = []
    for (domain, op_type), entry in sorted(table.items()):
        if (domain, op_type) not in _schemas():
            continue
        label = f"`{op_type}` (domain `{display_domain(domain)}`)"
        disposition = entry.get("disposition")
        served_by_kernel = bool(KERNELS.registered_versions(domain, op_type))
        if disposition not in DISPOSITIONS:
            problems.append(
                f"{label} is dispositioned `{disposition}`, which is not one of "
                f"{', '.join(DISPOSITIONS)}."
            )
            continue
        if served_by_kernel != (disposition == "native-kernel"):
            problems.append(
                f"{label} is dispositioned `{disposition}` while the kernel registry "
                + ("serves it." if served_by_kernel else "does not serve it.")
            )
        if disposition == "function-expansion" and not _defines_a_function(
            domain, op_type
        ):
            problems.append(
                f"{label} is dispositioned `function-expansion`, but ONNX defines no "
                "function body for it to be expanded into."
            )
        if (
            disposition in ("folding-or-graph-pass", "unsupported")
            and not str(entry.get("note", "")).strip()
        ):
            problems.append(f"{label} is dispositioned `{disposition}` with no note.")
        if disposition == "unsupported":
            problems.extend(_reason_problems(label, domain, op_type, entry))
    return problems


def _reason_problems(
    label: str, domain: str, op_type: str, entry: Mapping[str, Any]
) -> list[str]:
    """An unsupported entry against the closed reason set and the schema's own types."""
    reason = entry.get("reason")
    if reason not in UNSUPPORTED_REASONS:
        return [
            f"{label} is unsupported for reason `{reason}`, which is not one of "
            f"{', '.join(UNSUPPORTED_REASONS)}."
        ]
    family = _REASON_FAMILIES.get(reason)
    if family is not None and op_type not in family:
        return [
            f"{label} is unsupported as `{reason}`, but the compiler does not count it "
            f"as one: its `{reason}` family is {', '.join(sorted(family))}."
        ]
    if reason == "non-tensor-io" and not _admits(
        domain, op_type, ("seq(", "map(", "optional(")
    ):
        return [
            f"{label} is unsupported as `non-tensor-io`, but no operand of its schema "
            "takes or produces a sequence, a map or an optional."
        ]
    if reason == "runtime-strings" and not _admits(
        domain, op_type, ("tensor(string)",)
    ):
        return [
            f"{label} is unsupported as `runtime-strings`, but no operand of its schema "
            "takes or produces a string tensor."
        ]
    return []


def _admits(domain: str, op_type: str, kinds: Sequence[str]) -> bool:
    """Whether any operand of the op's newest schema admits one of these type forms."""
    schema = _schemas()[(domain, op_type)]
    constraints = {
        constraint.type_param_str: set(constraint.allowed_type_strs)
        for constraint in schema.type_constraints
    }
    return any(
        allowed.startswith(tuple(kinds))
        for formal in (*schema.inputs, *schema.outputs)
        for allowed in constraints.get(formal.type_str, {formal.type_str})
    )


def _ledger_problems(table: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    """The two directions the table and the conformance ledger have to agree in."""
    exercised, unimplemented = _corpus_ops()
    problems = []
    for domain, op_type in sorted(exercised):
        entry = table.get((domain, op_type), {})
        if entry.get("disposition") == "unsupported":
            problems.append(
                f"`{op_type}` (domain `{display_domain(domain)}`) is dispositioned "
                f"unsupported as `{entry.get('reason')}`, but a corpus test in the "
                "conformance pass list compiles and runs it."
            )
    for name, ops in sorted(unimplemented.items()):
        served = [
            op_type
            for domain, op_type in sorted(ops)
            if table.get((domain, op_type), {}).get("disposition") in _UNCONDITIONAL
        ]
        if len(served) == len(ops):
            problems.append(
                f"`{name}` is ledgered as `op-not-implemented`, but every op it runs "
                f"({', '.join(served)}) is dispositioned as served by a kernel or a "
                "function body."
            )
    return problems


# --------------------------------------------------------------------------------------
# Evidence that an op served without a kernel really compiles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Proof:
    """A model an op's disposition is proven on, and what the oracle is run on.

    `oracle` is the model the reference evaluator computes the expected outputs from, which is
    the compiled model itself except where ONNX ships no reference implementation for the op
    at all -- `ZipMap`, whose entry says what stands in for it.
    """

    model: ModelProto
    feeds: dict[str, Any]
    oracle: ModelProto | None = None


def _windowed(op_type: str) -> Proof:
    """A window of a fixed length: the one operand the graph has to fix for it to compile."""
    return Proof(
        _proof_model(
            [helper.make_node(op_type, ["size"], ["Y"])],
            initializer=[_constant("size", np.array(8, dtype=np.int64))],
        ),
        {},
    )


def _mel_weight_matrix() -> Proof:
    return Proof(
        _proof_model(
            [
                helper.make_node(
                    "MelWeightMatrix",
                    ["bins", "dft_length", "sample_rate", "lower", "upper"],
                    ["Y"],
                )
            ],
            initializer=[
                _constant("bins", np.array(4, dtype=np.int64)),
                _constant("dft_length", np.array(16, dtype=np.int64)),
                _constant("sample_rate", np.array(16000, dtype=np.int64)),
                _constant("lower", np.array(0.0, dtype=np.float32)),
                _constant("upper", np.array(8000.0, dtype=np.float32)),
            ],
        ),
        {},
    )


def _range() -> Proof:
    return Proof(
        _proof_model(
            [helper.make_node("Range", ["start", "limit", "delta"], ["Y"])],
            initializer=[
                _constant("start", np.array(1.0, dtype=np.float32)),
                _constant("limit", np.array(6.0, dtype=np.float32)),
                _constant("delta", np.array(0.5, dtype=np.float32)),
            ],
        ),
        {},
    )


def _center_crop_pad() -> Proof:
    """A crop on one axis and a pad on the other, through the function body ONNX defines."""
    return Proof(
        _proof_model(
            [helper.make_node("CenterCropPad", ["X", "shape"], ["Y"])],
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 6])],
            initializer=[_constant("shape", np.array([2, 8], dtype=np.int64))],
        ),
        {"X": np.arange(24, dtype=np.float32).reshape(4, 6)},
    )


def _zip_map() -> Proof:
    """A scaled tensor keyed into a map output, which the pass removes to leave the tensor.

    ONNX ships no reference implementation for `ZipMap` at all, so the oracle runs the same
    graph with the node already gone: what the pass promises is that the tensor `ZipMap` read
    reaches the caller in its place, which is exactly what that graph computes. The pairing of
    label to column — the part this cannot show — is covered in `test_extra_compiler_ml.py`,
    against onnxruntime.
    """
    scaler = helper.make_node(
        "Scaler",
        ["X"],
        ["scores"],
        domain=ML_DOMAIN,
        offset=[1.0, 0.0, -1.0],
        scale=[0.5, 0.25, 2.0],
    )
    keyed = helper.make_node(
        ZIP_MAP, ["scores"], ["Z"], domain=ML_DOMAIN, classlabels_int64s=[7, 9, 11]
    )
    maps = helper.make_value_info(
        "Z",
        helper.make_sequence_type_proto(
            helper.make_map_type_proto(
                TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, [])
            )
        ),
    )
    fed = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])]
    return Proof(
        _proof_model([scaler, keyed], inputs=fed, outputs=[maps], ml=True),
        {"X": np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=np.float32)},
        oracle=_proof_model(
            [scaler],
            inputs=fed,
            outputs=[helper.make_empty_tensor_value_info("scores")],
            ml=True,
        ),
    )


def _proof_model(
    nodes: Sequence[Any],
    *,
    inputs: Sequence[Any] = (),
    outputs: Sequence[Any] | None = None,
    initializer: Sequence[Any] = (),
    ml: bool = False,
) -> ModelProto:
    """A single-node model at the pinned package's newest opset, typed by inference alone."""
    graph = helper.make_graph(
        list(nodes),
        "proof",
        list(inputs),
        list(outputs)
        if outputs is not None
        else [helper.make_empty_tensor_value_info("Y")],
        initializer=list(initializer),
    )
    imports = [helper.make_opsetid("", max_supported_opset(""))]
    if ml:
        imports.append(helper.make_opsetid(ML_DOMAIN, max_supported_opset(ML_DOMAIN)))
    return helper.make_model(graph, opset_imports=imports)


def _constant(name: str, value: Any) -> Any:
    return numpy_helper.from_array(value, name)


PROOFS = {
    ("", "BlackmanWindow"): lambda: _windowed("BlackmanWindow"),
    ("", "CenterCropPad"): _center_crop_pad,
    ("", "HammingWindow"): lambda: _windowed("HammingWindow"),
    ("", "HannWindow"): lambda: _windowed("HannWindow"),
    ("", "MelWeightMatrix"): _mel_weight_matrix,
    ("", "Range"): _range,
    (ML_DOMAIN, ZIP_MAP): _zip_map,
}


def _served_without_a_kernel(
    table: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[tuple[str, str]]:
    return sorted(
        key
        for key, entry in table.items()
        if entry.get("disposition") in ("function-expansion", "folding-or-graph-pass")
    )


def _unevidenced(table: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[str]:
    """Ops claimed to be served without a kernel that neither the corpus nor a proof covers."""
    exercised, _ = _corpus_ops()
    return [
        f"`{op_type}` (domain `{display_domain(domain)}`)"
        for domain, op_type in _served_without_a_kernel(table)
        if (domain, op_type) not in exercised and (domain, op_type) not in PROOFS
    ]


# --------------------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------------------


def test_every_op_in_the_supported_domains_is_dispositioned():
    problems = _problems(_table())

    assert not problems, "\n".join(problems)


def test_an_op_the_table_does_not_name_fails_the_suite():
    """What an `onnx` upgrade looks like here: a new op is a missing disposition."""
    table = {key: entry for key, entry in _table().items() if key != ("", "Add")}

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`Add`" in problem and "has no disposition" in problem
    ]


def test_a_disposition_the_registry_contradicts_fails_the_suite():
    """Both ways round: a kernel claimed for an op with none, and none for an op with one."""
    table = {
        **_table(),
        ("", "NonZero"): {"disposition": "native-kernel"},
        ("", "Add"): {
            "disposition": "unsupported",
            "reason": "data-dependent-shape",
            "note": "invented",
        },
    }

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`NonZero`" in problem and "does not serve it" in problem
    ]
    assert [
        problem for problem in problems if "`Add`" in problem and "serves it" in problem
    ]


def test_an_expansion_claim_without_a_function_body_fails_the_suite():
    table = {**_table(), ("", "NonZero"): {"disposition": "function-expansion"}}

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`NonZero`" in problem and "no function body" in problem
    ]


def test_a_reason_outside_the_ledger_categories_fails_the_suite():
    table = {
        **_table(),
        ("", "NonZero"): {
            "disposition": "unsupported",
            "reason": "too difficult",
            "note": "invented",
        },
    }

    problems = _problems(table)

    assert [problem for problem in problems if "which is not one of" in problem]


def test_an_op_parked_as_not_yet_implemented_fails_the_suite():
    """The ledger's milestone category is not a disposition: closure admits no backlog."""
    table = {
        **_table(),
        ("", "NonZero"): {
            "disposition": "unsupported",
            "reason": "op-not-implemented",
            "note": "invented",
        },
    }

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`NonZero`" in problem and "which is not one of" in problem
    ]


def test_a_reason_the_compiler_does_not_apply_to_the_op_fails_the_suite():
    """A table cannot declare that an op is control flow, or a string op, or a draw."""
    table = {
        **_table(),
        ("", "NonZero"): {
            "disposition": "unsupported",
            "reason": "control-flow",
            "note": "invented",
        },
        (ML_DOMAIN, "CastMap"): {
            "disposition": "unsupported",
            "reason": "runtime-strings",
            "note": "invented",
        },
        ("", "Compress"): {
            "disposition": "unsupported",
            "reason": "non-tensor-io",
            "note": "invented",
        },
    }

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`NonZero`" in problem and "does not count it as one" in problem
    ]
    assert [
        problem
        for problem in problems
        if "`Compress`" in problem and "no operand of its schema" in problem
    ]
    # CastMap really does produce a string tensor for one of its output types, so the claim
    # that strings reach it is the one the schema cannot refute; the note has to carry the
    # rest of the story.
    assert not [
        problem
        for problem in problems
        if "`CastMap`" in problem and "no operand of its schema" in problem
    ]


def test_an_undocumented_unsupported_op_fails_the_suite():
    table = {
        **_table(),
        ("", "NonZero"): {
            "disposition": "unsupported",
            "reason": "data-dependent-shape",
            "note": "  ",
        },
    }

    problems = _problems(table)

    assert [problem for problem in problems if "with no note" in problem]


def test_calling_an_op_the_corpus_runs_unsupported_fails_the_suite():
    """The ratchet is the table's second opinion: a passing test disproves `unsupported`."""
    table = {
        **_table(),
        ("", "Relu"): {
            "disposition": "unsupported",
            "reason": "data-dependent-shape",
            "note": "invented",
        },
    }

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if "`Relu`" in problem
        and "conformance pass list compiles and runs it" in problem
    ]


def test_an_exclusion_the_table_says_is_served_fails_the_suite():
    """`op-not-implemented` cannot be claimed for a graph of ops the table says are served."""
    _, unimplemented = _corpus_ops()
    assert unimplemented, "the ledger excludes nothing as `op-not-implemented`"
    name, ops = sorted(unimplemented.items())[0]
    table = {
        **_table(),
        **{key: {"disposition": "native-kernel"} for key in ops},
    }

    problems = _problems(table)

    assert [
        problem
        for problem in problems
        if f"`{name}`" in problem
        and "is dispositioned as served by a kernel" in problem
    ]


def test_every_op_served_without_a_kernel_has_evidence():
    """No claim to serve an op through a pass or a function body rests on the table alone."""
    unevidenced = _unevidenced(_table())

    assert not unevidenced, (
        f"{', '.join(unevidenced)} are dispositioned as served without a kernel, which no "
        "test in the conformance pass list exercises and no proof model here compiles; add "
        "one to PROOFS."
    )


def test_an_unproven_claim_to_serve_an_op_fails_the_suite():
    table = {
        **_table(),
        ("", "NonZero"): {
            "disposition": "folding-or-graph-pass",
            "note": "invented",
        },
    }

    unevidenced = _unevidenced(table)

    assert [entry for entry in unevidenced if "`NonZero`" in entry]


@pytest.mark.parametrize(
    "domain,op_type", sorted(PROOFS), ids=lambda value: value or "ai.onnx"
)
def test_an_op_served_by_a_pass_or_a_function_body_compiles(domain, op_type, tmp_path):
    """The evidence itself: the compiler serves the op, and computes what ONNX says it does.

    Every expected value comes from the ONNX reference evaluator, never from this module.
    """
    proof = PROOFS[(domain, op_type)]()

    compiled = compile_onnx(proof.model, tmp_path).load()
    outputs = compiled.run(proof.feeds)

    expected = ReferenceEvaluator(proof.oracle or proof.model).run(None, proof.feeds)
    Runner.assert_similar_outputs(
        list(expected),
        [outputs[spec.name] for spec in compiled.outputs],
        rtol=1e-3,
        atol=1e-7,
    )


def test_a_range_below_the_revision_the_evaluator_can_be_vouched_for_is_refused(
    tmp_path,
):
    """The other half of `Range`'s entry: what the folding pass declines is refused outright.

    ONNX revised the op at opset 27, so at any older one the evaluator folding runs the node
    through implements a revision the compiler cannot vouch for. The rule the whole compiler
    is built on — never serve semantics nothing can confirm — makes that a compile error
    naming the op, not a folded value, and that is what the ledger's remaining
    `op-not-implemented` exclusion is.
    """
    model = _range().model
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 24))

    with pytest.raises(CompileError, match="`Range`"):
        compile_onnx(model, tmp_path)
