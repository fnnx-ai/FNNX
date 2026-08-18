"""ONNX backend conformance: the shipped node corpus is the oracle.

Every expected value comes from the test data the `onnx` package ships, and every
comparison goes through ONNX's own `Runner.assert_similar_outputs` under the tolerances
ONNX records per test. Nothing in this module decides what an op should compute.

The suite is fail-closed. Each enumerated corpus test is either

* **ratcheted** — compiled, built, executed and compared, where any error in that chain is
  a failure, or
* **ledgered** — excluded for a reason the governance check re-derives from the model
  itself, so an entry cannot be invented to silence a failing test,

and never neither nor both. The only skip is environmental and takes the whole module with
it: no `onnx`, no C compiler, or an `onnx` other than the pinned one whose schema set and
corpus the ledger and pass list are keyed to.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

onnx = pytest.importorskip("onnx")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import (  # noqa: E402
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    helper,
    numpy_helper,
)
from onnx.backend.test.loader import load_model_tests  # noqa: E402
from onnx.backend.test.runner import Runner  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.errors import CompileError  # noqa: E402
from fnnx.extras.compilers.c.onnx.dtypes import C_TYPES  # noqa: E402
from fnnx.extras.compilers.c.onnx.folding import fold_constants  # noqa: E402
from fnnx.extras.compilers.c.onnx.functions import (  # noqa: E402
    MAX_EXPANSION_DEPTH,
    function_model,
)
from fnnx.extras.compilers.c.onnx.kernels import KERNELS  # noqa: E402
from fnnx.extras.compilers.c.onnx.loader import (  # noqa: E402
    SUPPORTED_DOMAINS,
    normalize_domain,
    resolve_opsets,
)
from fnnx.extras.compilers.c.onnx.shapes import (  # noqa: E402
    infer_shapes,
    runtime_shape_operand,
    tensor_types,
)
from fnnx.extras.compilers.c.onnx.verify import (  # noqa: E402
    CONTROL_FLOW_OPS,
    DATA_DEPENDENT_SHAPE_OPS,
)

# The ledger, the pass list and the disposition of every op are keyed to one exact `onnx`
# release: it defines both the schema set the compiler dispatches against and the corpus
# this suite enumerates. Running under another release is not a weaker run of this suite,
# it is a run of a different suite, so the module steps aside rather than reporting on it.
PINNED_ONNX = "1.22"

CONFORMANCE_DIR = Path(__file__).parent / "conformance"
LEDGER_PATH = CONFORMANCE_DIR / "ledger.json"
RATCHET_PATH = CONFORMANCE_DIR / "passing.txt"
TOLERANCES_PATH = CONFORMANCE_DIR / "tolerances.json"

# The closed set of reasons a corpus test may be excluded, in the order `_classify` applies
# them. All but the last are structural — properties of the model that no amount of kernel
# work changes — and cover the compiler's v1 unsupported surface. `op-not-implemented` is
# the milestone category: the kernel tasks drive it down, and the governance check keeps it
# honest by refusing it for any op the kernel registry serves. What is left of it is
# accounted for op by op in `conformance/dispositions.json`, whose own suite holds the table
# to these same categories and refuses an entry that would exempt an op the compiler serves.
LEDGER_CATEGORIES = (
    "out-of-scope-domain",
    "non-tensor-io",
    "runtime-strings",
    "unsupported-dtype",
    "control-flow",
    "data-dependent-shape",
    "random-op",
    "external-codec",
    "op-not-implemented",
)

RANDOM_OPS = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)
CODEC_OPS = frozenset({"ImageDecoder"})

# What `onnx.backend.test.loader.load_model_tests` applies to a test that ships no
# `data.json`; a test asserts these are still the corpus-wide defaults.
ONNX_DEFAULT_RTOL = 1e-3
ONNX_DEFAULT_ATOL = 1e-7

# How far a per-op override may loosen those defaults. Two orders of magnitude covers what
# summation order and libm accuracy can cost a float32 kernel — `atol` reaches 1e-5, still
# well inside float32's ~1e-7 epsilon times a few thousand accumulations. Anything needing
# more is a wrong kernel, not a tolerance problem.
MAX_TOLERANCE_FACTOR = 100

pytestmark = [
    pytest.mark.skipif(
        not onnx.__version__.startswith(f"{PINNED_ONNX}."),
        reason=(
            f"the conformance corpus and ledger are pinned to onnx {PINNED_ONNX}.*, "
            f"but onnx {onnx.__version__} is installed"
        ),
    ),
    pytest.mark.skipif(
        not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
        reason="no system C compiler available",
    ),
]


# --------------------------------------------------------------------------------------
# The corpus, and the category a model may be ledgered under
# --------------------------------------------------------------------------------------


@cache
def _corpus() -> dict[str, Any]:
    """Every node test the pinned `onnx` package ships, by name."""
    return {case.name: case for case in load_model_tests(kind="node")}


def _subgraphs(node: NodeProto) -> Iterator[GraphProto]:
    for attribute in node.attribute:
        if attribute.HasField("g"):
            yield attribute.g
        yield from attribute.graphs


def _nodes(graph: GraphProto) -> Iterator[NodeProto]:
    for node in graph.node:
        yield node
        for subgraph in _subgraphs(node):
            yield from _nodes(subgraph)


def _value_infos(graph: GraphProto) -> Iterator[ValueInfoProto]:
    yield from graph.input
    yield from graph.output
    yield from graph.value_info
    for node in graph.node:
        for subgraph in _subgraphs(node):
            yield from _value_infos(subgraph)


def _classify(model: ModelProto) -> str | None:
    """The category this model may be ledgered under, or None if it has to compile and run.

    Everything is read off the model proto, the compiler's folding pass and the kernel
    registry, never off a compilation attempt: a compiler bug must surface as a failing
    test, not as a test that quietly becomes ledgerable.
    """
    graph = model.graph
    nodes = list(_nodes(graph))
    domains = {normalize_domain(imported.domain) for imported in model.opset_import}
    domains |= {normalize_domain(node.domain) for node in nodes}
    if not domains <= set(SUPPORTED_DOMAINS):
        return "out-of-scope-domain"

    infos = list(_value_infos(graph))
    if graph.sparse_initializer or any(
        info.type.WhichOneof("value") not in (None, "tensor_type") for info in infos
    ):
        return "non-tensor-io"

    declared = {
        info.type.tensor_type.elem_type
        for info in infos
        if info.type.WhichOneof("value") == "tensor_type"
    }
    declared |= {initializer.data_type for initializer in graph.initializer}
    if TensorProto.STRING in declared:
        return "runtime-strings"
    # UNDEFINED is a tensor whose element type the model does not state, not one whose
    # element type is unsupported; the compiler's own verification rejects it by name.
    if not declared <= set(C_TYPES) | {TensorProto.UNDEFINED}:
        return "unsupported-dtype"

    op_types = {node.op_type for node in nodes}
    for category, family in (
        ("control-flow", CONTROL_FLOW_OPS),
        ("data-dependent-shape", DATA_DEPENDENT_SHAPE_OPS),
    ):
        if op_types & family:
            return category
    if any(_draws_at_random(node, graph) for node in nodes):
        return "random-op"
    if op_types & CODEC_OPS:
        return "external-codec"

    try:
        opsets = resolve_opsets(model)
    except CompileError:
        return "op-not-implemented"
    folded = _folded_graph(model)
    types = tensor_types(folded)
    constants = {initializer.name for initializer in folded.initializer}
    # An op that takes its output shape from an operand's values — a reduction's axes — has a
    # data-dependent output shape unless the graph fixes that operand, whatever kernel serves
    # the op. That is structural, so it is derived before the milestone category below.
    if any(
        runtime_shape_operand(node, constants, types) is not None
        for node in _nodes(folded)
    ):
        return "data-dependent-shape"
    if any(not _serviceable(node, opsets, types) for node in _nodes(folded)):
        return "op-not-implemented"
    return None


def _draws_at_random(node: NodeProto, graph: GraphProto) -> bool:
    """Whether the node's output is a draw rather than a function of its inputs.

    `Dropout` is one only in training mode: without a `training_mode` operand, or with one
    the graph pins to false, it passes its input through.
    """
    if node.op_type in RANDOM_OPS:
        return True
    if node.op_type != "Dropout" or len(node.input) < 3 or not node.input[2]:
        return False
    for initializer in graph.initializer:
        if initializer.name == node.input[2]:
            return bool(numpy_helper.to_array(initializer).any())
    return True


def _serviceable(
    node: NodeProto,
    opsets: Mapping[str, int],
    types: Mapping[str, TypeProto],
    depth: int = 0,
) -> bool:
    """Whether the compiler has a way to compile this node, short of trying.

    A registered kernel, or — the fallback dispatch takes — the ONNX function body defining
    the op, provided the compiler can serve what that body is made of in turn. Deliberately
    "has a kernel at all" rather than "has one at this model's opset": a version the registry
    cannot vouch for is a gap to close, not a reason to exempt.

    The body is built and folded exactly as `codegen` builds and folds it, through the
    compiler's own passes, so that the nodes weighed here are the ones dispatch would really
    reach: a `Shape` the body computes its result from resolves by folding rather than by a
    kernel, and a context-dependent body — `CastLike`'s — only exists once its operands carry
    types. What is deliberately *not* run is the static verification: whether the body would
    compile is the question a failing test answers, not one a ledger entry may.
    """
    domain = normalize_domain(node.domain)
    if KERNELS.registered_versions(domain, node.op_type):
        return True
    opset_version = opsets.get(domain)
    if opset_version is None or depth >= MAX_EXPANSION_DEPTH:
        return False
    try:
        built = function_model(
            node, domain, opset_version, [types.get(name) for name in node.input]
        )
    except CompileError:
        return False
    if built is None or not built.model.graph.node:
        return False
    body_opsets = {
        normalize_domain(imported.domain): imported.version
        for imported in built.model.opset_import
    }
    folded = _folded_graph(built.model)
    body_types = tensor_types(folded)
    return all(
        _serviceable(inner, body_opsets, body_types, depth + 1)
        for inner in _nodes(folded)
    )


def _folded_graph(model: ModelProto) -> GraphProto:
    """What is left for kernels once the compiler's own folding pass has run.

    `Shape`, `Size` and constant subgraphs are resolved by folding rather than by a kernel —
    a disposition of its own — so a test of one is not a test of an unimplemented op, and
    ledgering it would hide a test that passes today. The pass run here is the compiler's,
    never a reimplementation of it. When it cannot run at all (ONNX's own shape inference
    rejects a handful of corpus models outright) nothing folds, which is what the unfolded
    graph already says.
    """
    folded = ModelProto()
    folded.CopyFrom(model)
    try:
        opsets = resolve_opsets(folded)
        folded = infer_shapes(folded)
        while fold_constants(folded, opsets):
            folded = infer_shapes(folded)
    except CompileError:
        return model.graph
    return folded.graph


@cache
def _derived_categories() -> dict[str, str | None]:
    return {
        name: _classify(onnx.load(Path(case.model_dir) / "model.onnx"))
        for name, case in _corpus().items()
    }


# --------------------------------------------------------------------------------------
# The checked-in ledger, pass list and tolerance overrides
# --------------------------------------------------------------------------------------


def _ledger() -> dict[str, str]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _ratchet() -> tuple[str, ...]:
    lines = RATCHET_PATH.read_text(encoding="utf-8").splitlines()
    return tuple(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )


def _overrides() -> dict[str, dict[str, Any]]:
    return json.loads(TOLERANCES_PATH.read_text(encoding="utf-8"))["overrides"]


def _governance_problems(
    ledger: Mapping[str, str], ratchet: Sequence[str]
) -> list[str]:
    """Everything wrong with the ledger and pass list as a pair, worst case all of it."""
    corpus = _corpus()
    derived = _derived_categories()
    problems = []
    for name, category in sorted(ledger.items()):
        if name not in corpus:
            problems.append(f"`{name}` is ledgered but the corpus has no such test.")
        elif category not in LEDGER_CATEGORIES:
            problems.append(
                f"`{name}` is ledgered as `{category}`, which is not one of the "
                f"categories {', '.join(LEDGER_CATEGORIES)}."
            )
        elif derived[name] != category:
            actual = derived[name]
            problems.append(
                f"`{name}` is ledgered as `{category}`, but its model is "
                + (
                    f"`{actual}`."
                    if actual is not None
                    else "compilable: a supported op's test cannot be ledgered."
                )
            )
    for name in sorted(set(ratchet) - set(corpus)):
        problems.append(
            f"`{name}` is in the pass list but the corpus has no such test."
        )
    for name in sorted(set(ratchet) & set(ledger)):
        problems.append(f"`{name}` is both ledgered and in the pass list.")
    for name in sorted(set(corpus) - set(ledger) - set(ratchet)):
        problems.append(
            f"`{name}` is in neither the ledger nor the pass list; it has to pass, "
            f"or be ledgered as `{derived[name]}`."
        )
    if list(ratchet) != sorted(set(ratchet)):
        problems.append("The pass list is not sorted, or lists a test twice.")
    return problems


def _tolerance_problems(overrides: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems = []
    for op_type, override in sorted(overrides.items()):
        if not str(override.get("justification", "")).strip():
            problems.append(
                f"The `{op_type}` tolerance override carries no written justification."
            )
        for field, default in (
            ("rtol", ONNX_DEFAULT_RTOL),
            ("atol", ONNX_DEFAULT_ATOL),
        ):
            value = override.get(field)
            if value is None:
                problems.append(f"The `{op_type}` tolerance override has no `{field}`.")
            elif value > default * MAX_TOLERANCE_FACTOR:
                problems.append(
                    f"The `{op_type}` tolerance override loosens `{field}` to {value}, "
                    f"beyond {MAX_TOLERANCE_FACTOR}x the ONNX default {default}."
                )
    return problems


# --------------------------------------------------------------------------------------
# Compiling, executing and comparing one corpus test
# --------------------------------------------------------------------------------------


def _read_tensor(path: Path) -> Any:
    tensor = TensorProto()
    tensor.ParseFromString(path.read_bytes())
    return numpy_helper.to_array(tensor)


def _load_data_set(directory: Path, model: ModelProto) -> tuple[dict[str, Any], list]:
    """Inputs by graph position, outputs in graph order — as ONNX's own runner reads them."""
    feeds = {}
    for index, entry in enumerate(model.graph.input):
        path = directory / f"input_{index}.pb"
        if path.is_file():
            feeds[entry.name] = _read_tensor(path)
    expected = [
        _read_tensor(directory / f"output_{index}.pb")
        for index in range(len(model.graph.output))
    ]
    return feeds, expected


def _tolerances(model: ModelProto, rtol: float, atol: float) -> tuple[float, float]:
    op_types = {node.op_type for node in _nodes(model.graph)}
    for op_type, override in sorted(_overrides().items()):
        if op_type in op_types:
            rtol = max(rtol, override["rtol"])
            atol = max(atol, override["atol"])
    return rtol, atol


def _execute(model_dir: Path, *, rtol: float, atol: float) -> None:
    """Compile a corpus test, run every data set it ships, compare against its outputs."""
    model_path = model_dir / "model.onnx"
    model = onnx.load(model_path)
    rtol, atol = _tolerances(model, rtol, atol)
    data_sets = sorted(
        path for path in model_dir.iterdir() if path.name.startswith("test_data_set")
    )
    assert data_sets, f"`{model_dir}` ships no test data set."

    with tempfile.TemporaryDirectory() as artifact_dir:
        compiled = compile_onnx(model_path, artifact_dir).load()
        for data_set in data_sets:
            feeds, expected = _load_data_set(data_set, model)
            outputs = compiled.run(
                {spec.name: feeds[spec.name] for spec in compiled.inputs}
            )
            Runner.assert_similar_outputs(
                expected,
                [outputs[entry.name] for entry in model.graph.output],
                rtol=rtol,
                atol=atol,
                model_dir=str(model_dir),
            )


# --------------------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ratchet())
def test_ratcheted_conformance_test_passes(name):
    """Every test the ratchet records as passing still compiles, runs and matches."""
    case = _corpus().get(name)
    assert case is not None, f"the corpus has no test `{name}`"

    _execute(Path(case.model_dir), rtol=case.rtol, atol=case.atol)


def test_every_corpus_test_is_ledgered_or_ratcheted():
    problems = _governance_problems(_ledger(), _ratchet())

    assert not problems, "\n".join(problems)


def test_the_ledger_cannot_exempt_a_supported_op():
    """A ledger entry for an op the registry serves is caught, whatever reason it claims."""
    ledger = {**_ledger(), "test_add": "op-not-implemented"}
    ratchet = [name for name in _ratchet() if name != "test_add"]

    problems = _governance_problems(ledger, ratchet)

    assert [
        problem
        for problem in problems
        if "test_add" in problem and "cannot be ledgered" in problem
    ]


def test_the_ledger_cannot_exempt_an_op_folding_resolves():
    """`Shape` never needs a kernel, so `op-not-implemented` cannot be claimed for it."""
    ledger = {**_ledger(), "test_shape": "op-not-implemented"}
    ratchet = [name for name in _ratchet() if name != "test_shape"]

    problems = _governance_problems(ledger, ratchet)

    assert [
        problem
        for problem in problems
        if "`test_shape`" in problem and "cannot be ledgered" in problem
    ]


def test_the_ledger_cannot_exempt_an_op_function_expansion_serves():
    """No kernel serves `HardSwish`; the body ONNX defines for it compiles all the same."""
    assert not KERNELS.registered_versions("", "HardSwish")
    ledger = {**_ledger(), "test_hardswish": "op-not-implemented"}
    ratchet = [name for name in _ratchet() if name != "test_hardswish"]

    problems = _governance_problems(ledger, ratchet)

    assert [
        problem
        for problem in problems
        if "`test_hardswish`" in problem and "cannot be ledgered" in problem
    ]


def test_ledgering_a_ratcheted_test_fails_the_suite():
    """The ratchet only grows: a passing test cannot be silenced by ledgering it."""
    ledger = {**_ledger(), "test_relu": "unsupported-dtype"}

    problems = _governance_problems(ledger, _ratchet())

    assert [
        problem
        for problem in problems
        if "test_relu" in problem and "both ledgered and in the pass list" in problem
    ]


def test_an_unaccounted_corpus_test_fails_the_suite():
    """A corpus test in neither file — an `onnx` upgrade's new tests — is not silence."""
    ledger = dict(_ledger())
    dropped = sorted(set(ledger) - set(_ratchet()))[0]
    del ledger[dropped]

    problems = _governance_problems(ledger, _ratchet())

    assert [
        problem
        for problem in problems
        if dropped in problem and "neither the ledger nor the pass list" in problem
    ]


def test_a_reason_outside_the_closed_set_is_rejected():
    ledger = {**_ledger(), sorted(_ledger())[0]: "not yet implemented"}

    problems = _governance_problems(ledger, _ratchet())

    assert [problem for problem in problems if "not one of the categories" in problem]


def test_the_recorded_tolerance_defaults_match_the_corpus():
    """The defaults the override bound rests on are ONNX's, not this suite's."""
    defaults = {(case.rtol, case.atol) for case in _corpus().values()}

    assert (ONNX_DEFAULT_RTOL, ONNX_DEFAULT_ATOL) in defaults


def test_the_tolerance_overrides_are_bounded_and_justified():
    problems = _tolerance_problems(_overrides())

    assert not problems, "\n".join(problems)


def test_an_unjustified_or_unbounded_override_is_rejected():
    problems = _tolerance_problems(
        {
            "Conv": {"rtol": ONNX_DEFAULT_RTOL, "atol": ONNX_DEFAULT_ATOL},
            "Gemm": {
                "rtol": ONNX_DEFAULT_RTOL,
                "atol": ONNX_DEFAULT_ATOL * MAX_TOLERANCE_FACTOR * 10,
                "justification": "summation order",
            },
        }
    )

    assert len(problems) == 2
    assert any("Conv" in problem and "justification" in problem for problem in problems)
    assert any("Gemm" in problem and "beyond" in problem for problem in problems)


def test_a_wrong_expectation_fails_the_case(tmp_path):
    """Fail-closed: the comparison is real, on the corpus's own data."""
    copied = tmp_path / "case"
    shutil.copytree(_corpus()["test_add"].model_dir, copied)
    output = copied / "test_data_set_0" / "output_0.pb"
    output.write_bytes(
        numpy_helper.from_array(_read_tensor(output) + 1.0).SerializeToString()
    )

    with pytest.raises(AssertionError):
        _execute(copied, rtol=ONNX_DEFAULT_RTOL, atol=ONNX_DEFAULT_ATOL)


def test_a_model_the_compiler_rejects_fails_the_case(tmp_path):
    """Fail-closed: a compile error is a failure, never a skip."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("NonZero", ["x"], ["y"], name="nonzero")],
            "rejected",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [2, 6])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    case = tmp_path / "case"
    (case / "test_data_set_0").mkdir(parents=True)
    onnx.save_model(model, case / "model.onnx")

    with pytest.raises(CompileError, match="NonZero"):
        _execute(case, rtol=ONNX_DEFAULT_RTOL, atol=ONNX_DEFAULT_ATOL)
