"""The command-line entry `python -m fnnx.extras.compilers.c` and the report it summarizes.

What a compiled artifact computes is the business of the conformance, differential and
bundle suites; what is pinned here is the command line — which files land where, which
options reach the compiler, what the summary says, and what a failure looks like on stderr.
The one execution check compares against the ONNX reference evaluator, never against a
hand-written expectation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from fnnx.extras.compilers.c.__main__ import main  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

OPSET = 21
SEED = 20260726

MODELS = Path(__file__).parent / "models"
PIPELINE_BUNDLE = MODELS / "onnx_pipeline.fnnx"

needs_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


def _relu_model(
    path: Path, name: str = "tiny", shape: tuple[str | int, ...] = ("batch", 3)
) -> Path:
    """A one-node model with a symbolic batch dimension, saved as a `.onnx` file."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.save(model, path)
    return path


def _nonzero_model(path: Path) -> Path:
    """`NonZero`'s output shape depends on the data, so this can never compile."""
    graph = helper.make_graph(
        [helper.make_node("NonZero", ["x"], ["y"], name="find")],
        "dynamic",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1, "n"])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)]), path
    )
    return path


def _summary_fields(stdout: str) -> dict[str, str]:
    """The summary's `label: value` lines, keyed by label; the heading line dropped."""
    return {
        label.strip(): value.strip()
        for label, _, value in (line.partition(":") for line in stdout.splitlines()[1:])
    }


def _report(directory: Path, prefix: str) -> dict[str, Any]:
    return json.loads((directory / f"{prefix}_report.json").read_text())


def test_the_module_entry_compiles_a_bundle(tmp_path):
    """The advertised invocation, run as its own process: `python -m ...`."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fnnx.extras.compilers.c",
            str(PIPELINE_BUNDLE),
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fnnx_model.h",
        "fnnx_model_report.json",
    ]
    assert str(tmp_path / "fnnx_model.h") in completed.stdout
    assert str(tmp_path / "fnnx_model_report.json") in completed.stdout


@needs_c_compiler
def test_a_raw_onnx_model_compiles_and_runs(tmp_path):
    model_path = _relu_model(tmp_path / "tiny.onnx")

    assert main([str(model_path), "-o", str(tmp_path / "out"), "--dim", "batch=2"]) == 0

    compiled = harness.load_compiled(tmp_path / "out" / "tiny_report.json")
    feeds = {"x": np.random.default_rng(SEED).normal(size=(2, 3)).astype(np.float32)}
    reference = ReferenceEvaluator(str(model_path))
    expected = dict(zip(reference.output_names, reference.run(None, feeds)))
    np.testing.assert_allclose(
        compiled.run(feeds)["y"], expected["y"], rtol=1e-6, atol=1e-6
    )


def test_the_dim_flag_binds_symbolic_dimensions(tmp_path):
    assert main([str(PIPELINE_BUNDLE), "-o", str(tmp_path), "--dim", "batch=4"]) == 0

    report = _report(tmp_path, "fnnx_model")
    assert report["options"]["dim_bindings"] == {"batch": 4}
    assert report["dim_bindings"] == {"batch": 4}
    assert report["entrypoint"]["inputs"][0]["shape"] == [4, 3]
    assert (
        "#define FNNX_MODEL_INPUT_X_DIM_0 4" in (tmp_path / "fnnx_model.h").read_text()
    )


def test_the_dim_flag_repeats_once_per_symbolic_dimension(tmp_path):
    model_path = _relu_model(tmp_path / "pair.onnx", "pair", ("batch", "features"))
    output = tmp_path / "out"
    flags = ["--dim", "batch=2", "--dim", "features=5"]

    assert main([str(model_path), "-o", str(output), *flags]) == 0

    report = _report(output, "pair")
    assert report["dim_bindings"] == {"batch": 2, "features": 5}
    assert report["entrypoint"]["inputs"][0]["shape"] == [2, 5]


def test_unbound_dimensions_default_to_one(tmp_path, capsys):
    assert main([str(PIPELINE_BUNDLE), "-o", str(tmp_path)]) == 0

    assert _report(tmp_path, "fnnx_model")["dim_bindings"] == {"batch": 1}
    assert _summary_fields(capsys.readouterr().out)["dimensions"] == "batch=1"


def test_the_prefix_flag_names_the_files_and_the_symbols(tmp_path):
    model_path = _relu_model(tmp_path / "tiny.onnx")
    output = tmp_path / "out"

    assert main([str(model_path), "-o", str(output), "--prefix", "my model"]) == 0

    assert sorted(path.name for path in output.iterdir()) == [
        "my_model.h",
        "my_model_report.json",
    ]
    assert "int my_model_run(" in (output / "my_model.h").read_text()
    assert _report(output, "my_model")["options"]["prefix"] == "my model"


def test_the_summary_reports_footprint_kernels_opsets_and_bindings(tmp_path, capsys):
    assert main([str(PIPELINE_BUNDLE), "-o", str(tmp_path), "--dim", "batch=2"]) == 0

    fields = _summary_fields(capsys.readouterr().out)
    report = _report(tmp_path, "fnnx_model")
    memory = report["memory"]
    assert fields["header"] == str(tmp_path / "fnnx_model.h")
    assert fields["report"] == str(tmp_path / "fnnx_model_report.json")
    assert fields["entrypoint"] == f"{report['entrypoint']['symbol']}()"
    assert fields["opsets"] == "ai.onnx=21, ai.onnx.ml=1"
    assert fields["opsets"] == ", ".join(
        f"{domain}={version}" for domain, version in report["opsets"].items()
    )
    assert fields["dimensions"] == "batch=2"
    assert fields["kernels"] == str(len(report["kernels"]))
    assert fields["static memory"] == (
        f"{memory['static_bytes']} bytes "
        f"(weights {memory['weights_bytes']}, arena {memory['arena_bytes']})"
    )
    assert memory["static_bytes"] == memory["weights_bytes"] + memory["arena_bytes"]


@pytest.mark.parametrize("flag", ["batch", "batch=four", "batch=", "=4", "batch=4.0"])
def test_a_malformed_dim_flag_is_rejected_before_compiling(tmp_path, flag, capsys):
    output = tmp_path / "out"

    with pytest.raises(SystemExit) as failure:
        main([str(PIPELINE_BUNDLE), "-o", str(output), "--dim", flag])

    assert failure.value.code == 2
    assert "--dim" in capsys.readouterr().err
    assert not output.exists()


def test_the_output_directory_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit) as failure:
        main([str(PIPELINE_BUNDLE)])

    assert failure.value.code == 2
    assert "--output-dir" in capsys.readouterr().err


def test_a_negative_dim_binding_is_reported_as_a_compile_error(tmp_path, capsys):
    output = tmp_path / "out"

    assert main([str(PIPELINE_BUNDLE), "-o", str(output), "--dim", "batch=-1"]) == 1

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err.startswith("error: ")
    assert "batch" in streams.err
    assert not output.exists()


def test_an_unsupported_model_fails_without_writing_anything(tmp_path, capsys):
    model_path = _nonzero_model(tmp_path / "dynamic.onnx")
    output = tmp_path / "out"

    assert main([str(model_path), "-o", str(output)]) == 1

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err.startswith("error: ")
    assert "find" in streams.err and "NonZero" in streams.err
    assert not output.exists()


def test_an_output_path_that_is_not_a_directory_is_reported_cleanly(tmp_path, capsys):
    occupied = tmp_path / "taken"
    occupied.write_text("not a directory")

    assert main([str(PIPELINE_BUNDLE), "-o", str(occupied)]) == 1

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err.startswith("error: ")
    assert str(occupied) in streams.err
    assert "Traceback" not in streams.err
    assert occupied.read_text() == "not a directory"


@pytest.mark.parametrize("missing", ["absent.fnnx", "absent.onnx"])
def test_a_missing_source_is_reported_cleanly(tmp_path, missing, capsys):
    output = tmp_path / "out"

    assert main([str(tmp_path / missing), "-o", str(output)]) == 1

    streams = capsys.readouterr()
    assert streams.err.startswith("error: ")
    assert missing in streams.err
    assert "Traceback" not in streams.err
    assert not output.exists()
