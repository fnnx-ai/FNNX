"""The load-and-run harness, and the starter kernels driven end to end through it.

Every expected value comes from the ONNX reference evaluator — the executable form of the
spec — never from a hand-written expectation. The only hand-written C here is the stub
artifact used to cover harness behaviour (status codes, per-node entrypoints, the strict
build flags) that the emitted artifacts cannot yet exercise.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fnnx.extras.compilers.c.errors import CompileError, HarnessError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
codegen = pytest.importorskip("fnnx.extras.compilers.c.onnx.codegen")
frontend = pytest.importorskip("fnnx.extras.compilers.c.onnx.frontend")
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from fnnx.extras.compilers.c import compile_onnx, load_compiled  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

OPSET = 21
SEED = 20260725

# Building the artifact is the point of this module, so the whole file needs a compiler.
pytestmark = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


def _tensor(name: str, elem_type: int, shape) -> Any:
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _model(nodes, inputs, outputs, *, initializer=(), name="graph", opset=OPSET):
    graph = helper.make_graph(
        nodes, name, list(inputs), list(outputs), initializer=list(initializer)
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _elem_type(dtype) -> int:
    return helper.np_dtype_to_tensor_dtype(np.dtype(dtype))


def _values(shape, dtype, *, seed: int = SEED):
    """Seeded inputs; integers stay small so C's undefined signed overflow is never hit."""
    generator = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.floating):
        return generator.normal(size=shape).astype(dtype)
    if np.dtype(dtype) == np.bool_:
        return generator.integers(0, 2, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    low, high = max(info.min, -100), min(info.max, 100)
    return generator.integers(low, high, size=shape, endpoint=True).astype(dtype)


def _reference(model, feeds: dict[str, Any]) -> dict[str, Any]:
    outputs = ReferenceEvaluator(model).run(None, dict(feeds))
    return dict(zip([entry.name for entry in model.graph.output], outputs))


def _assert_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert sorted(actual) == sorted(expected)
    for name, want in expected.items():
        got = actual[name]
        assert got.dtype == want.dtype, name
        assert got.shape == want.shape, name
        if np.issubdtype(want.dtype, np.floating):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)
        else:
            np.testing.assert_array_equal(got, want, err_msg=name)


def _run_against_reference(model, feeds, tmp_path):
    compiled = compile_onnx(model, tmp_path).load()
    _assert_matches(compiled.run(feeds), _reference(model, feeds))
    return compiled


def _pipeline_model(name="demo"):
    """Gemm -> Add -> Relu -> Identity: every starter kernel in one graph."""
    weights = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5.0) / 7.0
    intercept = np.array([0.25, -0.5, 1.0, 2.0], dtype=np.float32)
    offset = np.array([[0.1], [-0.2]], dtype=np.float32)
    return _model(
        [
            helper.make_node(
                "Gemm", ["x", "w", "b"], ["h"], name="gemm", alpha=0.5, transB=1
            ),
            helper.make_node("Add", ["h", "offset"], ["s"], name="add"),
            helper.make_node("Relu", ["s"], ["r"], name="relu"),
            helper.make_node("Identity", ["r"], ["y"], name="identity"),
        ],
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [_tensor("y", TensorProto.FLOAT, [2, 4])],
        initializer=[
            onnx.numpy_helper.from_array(weights, "w"),
            onnx.numpy_helper.from_array(intercept, "b"),
            onnx.numpy_helper.from_array(offset, "offset"),
        ],
        name=name,
    )


def _symbolic_batch_model():
    """The same layer over a symbolic batch, so a binding decides every buffer's extent."""
    weights = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5.0) / 7.0
    return _model(
        [
            helper.make_node("Gemm", ["x", "w"], ["h"], name="gemm", transB=1),
            helper.make_node("Relu", ["h"], ["y"], name="relu"),
        ],
        [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
        [_tensor("y", TensorProto.FLOAT, ["batch", 4])],
        initializer=[onnx.numpy_helper.from_array(weights, "w")],
    )


def _external_weight_model(weights, location: str):
    """A layer whose weight lives in a side file next to the model rather than inside it."""
    tensor = onnx.numpy_helper.from_array(weights, "w")
    onnx.external_data_helper.set_external_data(tensor, location=location)
    tensor.ClearField("raw_data")
    return _model(
        [helper.make_node("Gemm", ["x", "w"], ["y"], name="gemm", transB=1)],
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [_tensor("y", TensorProto.FLOAT, [2, 4])],
        initializer=[tensor],
        name="external",
    )


# --------------------------------------------------------------------------------------
# Loading and running an emitted artifact
# --------------------------------------------------------------------------------------


def test_repeated_runs_match_the_reference_and_carry_no_state(tmp_path):
    model = _pipeline_model()
    compiled = compile_onnx(model, tmp_path).load()
    first = {"x": _values((2, 3), np.float32, seed=1)}
    second = {"x": _values((2, 3), np.float32, seed=2)}

    first_outputs = compiled.run(first)
    second_outputs = compiled.run(second)
    repeat_outputs = compiled.run(first)

    _assert_matches(first_outputs, _reference(model, first))
    _assert_matches(second_outputs, _reference(model, second))
    _assert_matches(repeat_outputs, _reference(model, first))


def test_a_bound_dimension_sizes_the_code_the_artifact_runs(tmp_path):
    """A binding is not only report metadata: the emitted code has to compute at that size.

    The report and the macros are checked where they are emitted; what is checked here is
    the half a buffer sized from the pre-binding shape would pass — that the artifact run at
    the bound size produces what the spec says it should.
    """
    model = _symbolic_batch_model()
    result = compile_onnx(model, tmp_path, dim_bindings={"batch": 4})
    feeds = {"x": _values((4, 3), np.float32)}

    outputs = result.load().run(feeds)

    assert result.report["entrypoint"]["inputs"][0]["shape"] == [4, 3]
    _assert_matches(outputs, _reference(model, feeds))


def test_an_external_data_model_runs_without_its_side_file(tmp_path):
    """External tensors are resolved at compile time, so the artifact never reads them again.

    The side file is deleted after the build and before the run, which is what makes "no
    runtime file access" an assertion rather than a claim.
    """
    weights = _values((4, 3), np.float32)
    side_file = tmp_path / "w.bin"
    side_file.write_bytes(weights.tobytes())
    model_path = tmp_path / "model.onnx"
    onnx.save_model(_external_weight_model(weights, side_file.name), str(model_path))
    feeds = {"x": _values((2, 3), np.float32, seed=SEED + 1)}
    expected = _reference(onnx.load(str(model_path)), feeds)

    compiled = compile_onnx(model_path, tmp_path / "out").load()
    side_file.unlink()

    _assert_matches(compiled.run(feeds), expected)


def test_load_compiled_accepts_the_header_or_the_report(tmp_path):
    result = compile_onnx(_pipeline_model(), tmp_path)
    feeds = {"x": _values((2, 3), np.float32)}

    from_header = load_compiled(result.header_path).run(feeds)
    from_report = load_compiled(result.report_path).run(feeds)

    np.testing.assert_array_equal(from_header["y"], from_report["y"])


def test_the_artifact_directory_keeps_only_the_emitted_files(tmp_path):
    result = compile_onnx(_pipeline_model(), tmp_path)
    result.load()

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "demo.h",
        "demo_report.json",
    ]


def test_compiling_a_kernel_graph_twice_is_byte_identical(tmp_path):
    """Kernel dedup and registry lookup are the new orderings determinism rests on."""
    first = compile_onnx(_pipeline_model(), tmp_path / "first")
    second = compile_onnx(_pipeline_model(), tmp_path / "second")

    assert first.header_path.read_bytes() == second.header_path.read_bytes()


def test_the_starter_kernels_allocate_nothing(tmp_path):
    """Every other test builds under `-Werror=vla`; the tokens are checked here."""
    header = compile_onnx(_pipeline_model(), tmp_path).header_path.read_text()

    for token in ("malloc", "calloc", "realloc", "free", "alloca"):
        assert not re.search(rf"\b{token}\b", header), token


def test_metadata_comes_from_the_compile_report(tmp_path):
    compiled = compile_onnx(_pipeline_model(), tmp_path).load()

    assert compiled.inputs == (harness.TensorSpec("x", np.dtype("float32"), (2, 3)),)
    assert compiled.outputs == (harness.TensorSpec("y", np.dtype("float32"), (2, 4)),)
    assert compiled.node_ids == ()


def test_inputs_may_be_named_or_passed_as_a_mapping(tmp_path):
    """Tensor names are not always Python identifiers, so both forms have to work."""
    model = _model(
        [helper.make_node("Relu", ["in.1"], ["out.1"], name="relu")],
        [_tensor("in.1", TensorProto.FLOAT, [3])],
        [_tensor("out.1", TensorProto.FLOAT, [3])],
    )
    compiled = compile_onnx(model, tmp_path).load()
    values = _values((3,), np.float32)

    _assert_matches(compiled.run({"in.1": values}), _reference(model, {"in.1": values}))
    with pytest.raises(HarnessError, match="unexpected `x`"):
        compiled.run(x=values)


def test_a_name_given_twice_is_rejected(tmp_path):
    compiled = compile_onnx(_pipeline_model(), tmp_path).load()
    values = _values((2, 3), np.float32)

    with pytest.raises(HarnessError, match="both in the mapping and as keyword"):
        compiled.run({"x": values}, x=values)


def test_non_contiguous_inputs_are_accepted(tmp_path):
    model = _model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [_tensor("y", TensorProto.FLOAT, [2, 3])],
    )
    compiled = compile_onnx(model, tmp_path).load()
    view = _values((3, 2), np.float32).T

    assert not view.flags["C_CONTIGUOUS"]
    _assert_matches(compiled.run(x=view), _reference(model, {"x": view}))


def _forbid_the_c_call(compiled, monkeypatch) -> None:
    def explode(*arguments):
        raise AssertionError("the C entrypoint must not be called")

    monkeypatch.setattr(compiled._entry, "call", explode)


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        (
            np.zeros((2, 3), dtype=np.float64),
            r"input `x` has dtype `float64`.*`float32`",
        ),
        (np.zeros((3, 3), dtype=np.float32), r"input `x` has shape \(3, 3\).*\(2, 3\)"),
        (np.zeros((2, 3), dtype=np.int32), r"input `x` has dtype `int32`.*`float32`"),
    ],
)
def test_mismatched_inputs_are_rejected_before_the_c_call(
    tmp_path, monkeypatch, value, expected_message
):
    compiled = compile_onnx(_pipeline_model(), tmp_path).load()
    _forbid_the_c_call(compiled, monkeypatch)

    with pytest.raises(HarnessError, match=expected_message):
        compiled.run(x=value)


def test_missing_and_unexpected_inputs_are_named(tmp_path, monkeypatch):
    compiled = compile_onnx(_pipeline_model(), tmp_path).load()
    _forbid_the_c_call(compiled, monkeypatch)

    with pytest.raises(HarnessError, match="missing `x` and unexpected `wrong`"):
        compiled.run(wrong=_values((2, 3), np.float32))


# --------------------------------------------------------------------------------------
# Harness behaviour the emitted artifacts cannot exercise yet
# --------------------------------------------------------------------------------------

_STUB_HEADER = """\
#ifndef STUB_H_INCLUDED
#define STUB_H_INCLUDED

int stub_run(const float* x, float* y);
int stub_node_double_run(const float* x, float* y);

#endif /* STUB_H_INCLUDED */

#ifdef STUB_IMPLEMENTATION

int stub_run(const float* x, float* y)
{
    (void)x;
    (void)y;
    return 7;
}

int stub_node_double_run(const float* x, float* y)
{
    y[0] = x[0] * 2.0f;
    return 0;
}

#endif /* STUB_IMPLEMENTATION */
"""


def _stub_artifact(tmp_path: Path, *, header: str = _STUB_HEADER, **overrides) -> Path:
    """A hand-written artifact standing in for one the bundle layer will emit later."""
    scalar = [{"name": "x", "dtype": "float32", "shape": [1]}]
    result = [{"name": "y", "dtype": "float32", "shape": [1]}]
    report = {
        "prefix": "stub",
        "header": "stub.h",
        "entrypoint": {"symbol": "stub_run", "inputs": scalar, "outputs": result},
        "nodes": [
            {
                "id": "double",
                "symbol": "stub_node_double_run",
                "inputs": scalar,
                "outputs": result,
            }
        ],
    }
    report.update(overrides)
    (tmp_path / "stub.h").write_text(header, encoding="utf-8")
    report_path = tmp_path / "stub_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_a_nonzero_status_becomes_an_exception(tmp_path):
    compiled = load_compiled(_stub_artifact(tmp_path))

    with pytest.raises(HarnessError, match="`stub_run` returned status 7"):
        compiled.run(x=np.ones(1, dtype=np.float32))


def test_node_entrypoints_are_bound_from_the_report(tmp_path):
    compiled = load_compiled(_stub_artifact(tmp_path))

    outputs = compiled.run_node("double", {"x": np.array([1.5], dtype=np.float32)})

    assert compiled.node_ids == ("double",)
    np.testing.assert_array_equal(outputs["y"], np.array([3.0], dtype=np.float32))


def test_node_inputs_are_validated_like_the_model_inputs(tmp_path):
    compiled = load_compiled(_stub_artifact(tmp_path))

    with pytest.raises(
        HarnessError, match=r"Node `double`: input `x` has shape \(2,\)"
    ):
        compiled.run_node("double", {"x": np.ones(2, dtype=np.float32)})


def test_an_unknown_node_id_lists_the_available_ones(tmp_path):
    compiled = load_compiled(_stub_artifact(tmp_path))

    with pytest.raises(
        HarnessError, match="no entrypoint for node `missing`.*`double`"
    ):
        compiled.run_node("missing", {})


def test_an_artifact_without_node_entrypoints_says_so(tmp_path):
    compiled = compile_onnx(_pipeline_model(), tmp_path).load()

    with pytest.raises(HarnessError, match="no entrypoint for node `gemm`.*none"):
        compiled.run_node("gemm", {})


def test_a_symbol_the_library_lacks_is_reported(tmp_path):
    report_path = _stub_artifact(
        tmp_path,
        entrypoint={
            "symbol": "stub_absent_run",
            "inputs": [],
            "outputs": [],
        },
    )

    with pytest.raises(HarnessError, match="exports no symbol `stub_absent_run`"):
        load_compiled(report_path)


def test_the_artifact_is_built_under_the_strict_flags(tmp_path):
    """An unused parameter only fails the build because of `-Wextra -Werror`."""
    lax_header = _STUB_HEADER.replace("    (void)x;\n", "")

    with pytest.raises(HarnessError, match="unused parameter"):
        load_compiled(_stub_artifact(tmp_path, header=lax_header))


def test_a_missing_header_is_reported(tmp_path):
    report_path = _stub_artifact(tmp_path)
    (tmp_path / "stub.h").unlink()

    with pytest.raises(
        HarnessError, match="stub.h` the compile report names is missing"
    ):
        load_compiled(report_path)


@pytest.mark.parametrize(
    ("name", "expected_message"),
    [
        ("model.onnx", "neither a generated header"),
        ("absent_report.json", "Compile report not found"),
        ("absent.h", "Compile report not found"),
    ],
)
def test_paths_that_are_not_an_artifact_are_rejected(tmp_path, name, expected_message):
    with pytest.raises(HarnessError, match=expected_message):
        load_compiled(tmp_path / name)


def test_a_report_that_is_not_ours_is_rejected(tmp_path):
    report_path = tmp_path / "other_report.json"
    report_path.write_text(json.dumps({"prefix": "other"}), encoding="utf-8")

    with pytest.raises(HarnessError, match="missing the `header`, `entrypoint` field"):
        load_compiled(report_path)


def test_an_unknown_compiler_is_reported(tmp_path):
    result = compile_onnx(_pipeline_model(), tmp_path)

    with pytest.raises(HarnessError, match="`not-a-real-compiler` was not found"):
        result.load(compiler="not-a-real-compiler")


def test_a_missing_compiler_is_reported(tmp_path, monkeypatch):
    result = compile_onnx(_pipeline_model(), tmp_path)
    monkeypatch.setattr(harness.shutil, "which", lambda name: None)

    with pytest.raises(HarnessError, match="No system C compiler was found"):
        result.load()


def test_a_missing_numpy_raises_an_actionable_error(monkeypatch):
    """A `ModuleNotFoundError` keeps `pytest.importorskip` skipping rather than erroring."""
    module = "fnnx.extras.compilers.c.harness"
    monkeypatch.delitem(sys.modules, module)

    with mock.patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ModuleNotFoundError, match=r"numpy.*fnnx\[core\]"):
            importlib.import_module(module)


def test_the_cc_environment_variable_is_preferred(tmp_path, monkeypatch):
    result = compile_onnx(_pipeline_model(), tmp_path)
    monkeypatch.setenv("CC", "cc-from-the-environment")
    monkeypatch.setattr(
        harness.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if "env" in name else None,
    )

    with pytest.raises(HarnessError, match="cc-from-the-environment"):
        result.load()


# --------------------------------------------------------------------------------------
# Starter kernels
# --------------------------------------------------------------------------------------

_BINARY_SHAPES = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((1, 3), (2, 1)),
    ((2, 3), ()),
    ((2, 1, 3), (4, 3)),
    ((), ()),
    ((0, 3), (3,)),
]


@pytest.mark.parametrize("op_type", ["Add", "Mul"])
@pytest.mark.parametrize(("left_shape", "right_shape"), _BINARY_SHAPES)
def test_binary_ops_broadcast_like_the_reference(
    tmp_path, op_type, left_shape, right_shape
):
    output_shape = np.broadcast_shapes(left_shape, right_shape)
    model = _model(
        [helper.make_node(op_type, ["a", "b"], ["y"], name="op")],
        [
            _tensor("a", TensorProto.FLOAT, left_shape),
            _tensor("b", TensorProto.FLOAT, right_shape),
        ],
        [_tensor("y", TensorProto.FLOAT, output_shape)],
    )
    feeds = {
        "a": _values(left_shape, np.float32, seed=1),
        "b": _values(right_shape, np.float32, seed=2),
    }

    _run_against_reference(model, feeds, tmp_path)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8, np.int8]
)
def test_binary_ops_cover_every_supported_family(tmp_path, dtype):
    elem_type = _elem_type(dtype)
    model = _model(
        [helper.make_node("Add", ["a", "b"], ["y"], name="op")],
        [_tensor("a", elem_type, (2, 3)), _tensor("b", elem_type, (3,))],
        [_tensor("y", elem_type, (2, 3))],
    )
    feeds = {
        "a": _values((2, 3), dtype, seed=1),
        "b": _values((3,), dtype, seed=2),
    }

    _run_against_reference(model, feeds, tmp_path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_relu_handles_nan_infinity_and_signed_zero(tmp_path, dtype):
    info = np.finfo(dtype)
    values = np.array(
        [
            0.0,
            -0.0,
            np.nan,
            np.inf,
            -np.inf,
            info.max,
            -info.max,
            info.tiny,
            -info.tiny,
            info.smallest_subnormal,
            -1.5,
            1.5,
        ],
        dtype=dtype,
    )
    elem_type = _elem_type(dtype)
    model = _model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        [_tensor("x", elem_type, values.shape)],
        [_tensor("y", elem_type, values.shape)],
    )

    compiled = _run_against_reference(model, {"x": values}, tmp_path)
    outputs = compiled.run(x=values)

    # `allclose` reads -0.0 and 0.0 as equal, so the sign of every zero is checked too.
    expected = _reference(model, {"x": values})["y"]
    assert np.array_equal(np.signbit(outputs["y"]), np.signbit(expected))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.int8])
def test_relu_clamps_integers(tmp_path, dtype):
    elem_type = _elem_type(dtype)
    values = _values((3, 4), dtype)
    model = _model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        [_tensor("x", elem_type, values.shape)],
        [_tensor("y", elem_type, values.shape)],
    )

    _run_against_reference(model, {"x": values}, tmp_path)


@pytest.mark.parametrize("transpose_a", [0, 1])
@pytest.mark.parametrize("transpose_b", [0, 1])
@pytest.mark.parametrize(
    ("alpha", "beta", "bias_shape"),
    [
        (1.0, 1.0, (4,)),
        (0.5, 2.0, (2, 4)),
        (1.5, 0.0, (2, 4)),
        (0.25, 3.0, ()),
        (2.0, 1.0, None),
    ],
)
def test_gemm_covers_transposes_scaling_and_bias(
    tmp_path, transpose_a, transpose_b, alpha, beta, bias_shape
):
    left_shape = (3, 2) if transpose_a else (2, 3)
    right_shape = (4, 3) if transpose_b else (3, 4)
    inputs = ["a", "b"] if bias_shape is None else ["a", "b", "c"]
    model = _model(
        [
            helper.make_node(
                "Gemm",
                inputs,
                ["y"],
                name="gemm",
                alpha=alpha,
                beta=beta,
                transA=transpose_a,
                transB=transpose_b,
            )
        ],
        [
            _tensor("a", TensorProto.FLOAT, left_shape),
            _tensor("b", TensorProto.FLOAT, right_shape),
            *(
                []
                if bias_shape is None
                else [_tensor("c", TensorProto.FLOAT, bias_shape)]
            ),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 4))],
    )
    feeds = {
        "a": _values(left_shape, np.float32, seed=1),
        "b": _values(right_shape, np.float32, seed=2),
    }
    if bias_shape is not None:
        feeds["c"] = _values(bias_shape, np.float32, seed=3)

    _run_against_reference(model, feeds, tmp_path)


def test_gemm_scales_integer_operands_like_the_reference(tmp_path):
    """Integer Gemm scales through float64 and truncates, as the reference does."""
    model = _model(
        [
            helper.make_node(
                "Gemm", ["a", "b", "c"], ["y"], name="gemm", alpha=0.5, beta=1.5
            )
        ],
        [
            _tensor("a", TensorProto.INT32, (2, 3)),
            _tensor("b", TensorProto.INT32, (3, 4)),
            _tensor("c", TensorProto.INT32, (4,)),
        ],
        [_tensor("y", TensorProto.INT32, (2, 4))],
    )
    feeds = {
        "a": _values((2, 3), np.int32, seed=1),
        "b": _values((3, 4), np.int32, seed=2),
        "c": _values((4,), np.int32, seed=3),
    }

    _run_against_reference(model, feeds, tmp_path)


def test_gemm_rejects_operands_that_are_not_matrices():
    """ONNX shape inference rejects such a graph first, so codegen is driven directly.

    The kernel must still refuse rather than fail on an unpacking error.
    """
    model = _model(
        [helper.make_node("Gemm", ["a", "b"], ["y"], name="gemm")],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3, 4)),
            _tensor("b", TensorProto.FLOAT, (3, 4)),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 4))],
    )
    prepared = frontend.PreparedModel(model=model, opsets={"": OPSET}, dim_bindings={})

    with pytest.raises(CompileError, match="Gemm takes 2-D operands"):
        codegen.build_program(prepared)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, np.bool_])
def test_identity_round_trips_every_element_family(tmp_path, dtype):
    elem_type = _elem_type(dtype)
    values = _values((2, 3), dtype)
    model = _model(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        [_tensor("x", elem_type, values.shape)],
        [_tensor("y", elem_type, values.shape)],
    )

    _run_against_reference(model, {"x": values}, tmp_path)


def test_zero_element_tensors_flow_through_the_kernels(tmp_path):
    model = _model(
        [
            helper.make_node("Add", ["x", "b"], ["s"], name="add"),
            helper.make_node("Relu", ["s"], ["r"], name="relu"),
            helper.make_node("Identity", ["r"], ["y"], name="identity"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (0, 3)),
            _tensor("b", TensorProto.FLOAT, (3,)),
        ],
        [_tensor("y", TensorProto.FLOAT, (0, 3))],
    )
    feeds = {
        "x": np.zeros((0, 3), dtype=np.float32),
        "b": _values((3,), np.float32),
    }

    _run_against_reference(model, feeds, tmp_path)


def test_a_zero_element_intermediate_no_statement_names_still_builds(tmp_path):
    """The chained copies emit nothing at all, leaving `t` named by no statement.

    An intermediate the implementation declares but never mentions is a `static` the C
    compiler rejects under the strict flags, so it must not be declared either.
    """
    model = _model(
        [
            helper.make_node("Identity", ["x"], ["t"], name="first"),
            helper.make_node("Identity", ["t"], ["y"], name="second"),
        ],
        [_tensor("x", TensorProto.FLOAT, (0, 3))],
        [_tensor("y", TensorProto.FLOAT, (0, 3))],
    )
    feeds = {"x": np.zeros((0, 3), dtype=np.float32)}

    compiled = _run_against_reference(model, feeds, tmp_path)

    assert compiled.report["memory"]["arena_bytes"] == 0


@pytest.mark.parametrize("opset", [7, 13, 14, OPSET])
def test_kernels_dispatch_at_every_registered_revision(tmp_path, opset):
    model = _model(
        [helper.make_node("Add", ["a", "b"], ["y"], name="op")],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (3,)),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 3))],
        opset=opset,
    )
    feeds = {
        "a": _values((2, 3), np.float32, seed=1),
        "b": _values((3,), np.float32, seed=2),
    }

    _run_against_reference(model, feeds, tmp_path)


def test_an_opset_below_the_registered_semantics_is_refused(tmp_path):
    """Add-6 broadcast through attributes; its semantics are not what the kernel implements."""
    model = _model(
        [helper.make_node("Add", ["a", "b"], ["y"], name="op")],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (2, 3)),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 3))],
        opset=6,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path / "out")

    assert "`Add`" in str(error.value)
    assert "Nearest supported version: 7." in str(error.value)
    assert not (tmp_path / "out").exists()
