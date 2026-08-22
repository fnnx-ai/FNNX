"""Hands-on demo: torch model -> ONNX -> FNNX bundle -> self-contained C99 header.

Builds a two-stage torch model (feature standardization, then a small MLP), exports each
stage to ONNX, wires the two into an FNNX ``pipeline`` bundle, and compiles that bundle
with ``fnnx.extras.compilers.c`` into a single C header with no dependencies beyond libm.
The artifact is then built into a shared library and its outputs are checked against the
FNNX runtime (onnxruntime) and against torch itself.

Usage (from src/python, with torch, onnx, onnxruntime, numpy and a C compiler installed):

    python examples/torch_to_c.py

Everything is written under ./_fnnx_c_demo/ so it can be inspected afterwards:

    _fnnx_c_demo/onnx/{scale,mlp}.onnx   the exported stages
    _fnnx_c_demo/scoring.fnnx/           the FNNX pipeline bundle
    _fnnx_c_demo/c/scoring.h             the compiled pipeline + its compile report
    _fnnx_c_demo/c/main.c                a C program that includes it, built with cc
    _fnnx_c_demo/c_mlp/mlp.h             the `mlp` stage compiled straight from ONNX

The compiler has two entrypoints, both shown below:
  * compile_bundle(...) -- an FNNX pipeline bundle, one C entrypoint per node plus the
    pipeline glue that runs them in order
  * compile_onnx(...)   -- a bare `.onnx` model, no FNNX packaging involved
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from fnnx.extras.compilers.c import compile_bundle, compile_onnx
from fnnx.runtime import Runtime

OUT_DIR = Path.cwd() / "_fnnx_c_demo"
FEATURES = 3
OPSET = 17
MAX_BATCH = 8
SEED = 20260727


# --------------------------------------------------------------------------------------
# 1. The torch model, as two stages
# --------------------------------------------------------------------------------------


class Standardize(nn.Module):
    """`(x - mean) / std`, with the statistics baked in as buffers."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MLP(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_stages() -> tuple[Standardize, MLP]:
    torch.manual_seed(SEED)
    scale = Standardize(
        mean=torch.tensor([0.5, -1.0, 2.0]), std=torch.tensor([1.5, 0.25, 3.0])
    )
    return scale.eval(), MLP(FEATURES, hidden=8).eval()


# --------------------------------------------------------------------------------------
# 2. torch -> ONNX
# --------------------------------------------------------------------------------------


def export_onnx(
    module: nn.Module, path: Path, *, input_name: str, output_name: str
) -> Path:
    """Export one stage with a symbolic batch axis named `batch`.

    The dimension name matters downstream: the compiler either fixes `batch` at compile
    time or turns it into a per-call argument, and it is addressed by this name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (torch.zeros(1, FEATURES),),
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=OPSET,
    )
    return path


# --------------------------------------------------------------------------------------
# 3. ONNX -> FNNX pipeline bundle
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One pipeline node: an ONNX model, the edges it reads and writes, and its shapes."""

    id: str
    model_path: Path
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_shapes: tuple[tuple[int | str, ...], ...]
    output_shapes: tuple[tuple[int | str, ...], ...]


def _tensor_spec(shape: tuple[int | str, ...]) -> dict[str, Any]:
    return {"dtype": "Array[float32]", "shape": list(shape)}


def _manifest_tensor(name: str, shape: tuple[int | str, ...]) -> dict[str, Any]:
    return {"name": name, "content_type": "NDJSON", **_tensor_spec(shape)}


def _op_attributes(model_path: Path) -> dict[str, Any]:
    """Read back what the exporter actually produced, rather than restating it."""
    model = onnx.load(str(model_path), load_external_data=False)
    return {
        "opsets": [
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ],
        "has_external_data": False,
        "onnx_ir_version": model.ir_version,
    }


def write_bundle(
    directory: Path,
    stages: list[Stage],
    *,
    name: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> Path:
    """Write an unpacked FNNX `pipeline` bundle: the manifest, the ops, and the wiring."""
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)

    manifest = {
        "variant": "pipeline",
        "name": name,
        "version": "1.0.0",
        "description": "Standardize features, then score them with a small MLP.",
        "producer_name": "fnnx-examples",
        "producer_version": "1.0.0",
        "producer_tags": ["torch", "demo"],
        "inputs": inputs,
        "outputs": outputs,
        "dynamic_attributes": [],
        "env_vars": [],
    }
    ops = [
        {
            "id": stage.id,
            "op": "ONNX_v1",
            "inputs": [_tensor_spec(shape) for shape in stage.input_shapes],
            "outputs": [_tensor_spec(shape) for shape in stage.output_shapes],
            "attributes": _op_attributes(stage.model_path),
            "dynamic_attributes": {},
        }
        for stage in stages
    ]
    variant_config = {
        "nodes": [
            {
                "op_instance_id": stage.id,
                "inputs": list(stage.inputs),
                "outputs": list(stage.outputs),
                "extra_dynattrs": {},
            }
            for stage in stages
        ]
    }

    for filename, document in (
        ("manifest.json", manifest),
        ("ops.json", ops),
        ("variant_config.json", variant_config),
        ("dtypes.json", {}),
        ("env.json", {}),
        ("meta.json", []),
    ):
        (directory / filename).write_text(json.dumps(document, indent=2) + "\n")

    for stage in stages:
        artifacts = directory / "ops_artifacts" / stage.id
        artifacts.mkdir(parents=True)
        shutil.copyfile(stage.model_path, artifacts / "model.onnx")
    return directory


# --------------------------------------------------------------------------------------
# 4. Compile, build, run
# --------------------------------------------------------------------------------------


def print_report(result: Any) -> None:
    report = result.report
    memory = report["memory"]
    runtime_dims = (
        ", ".join(f"{dim['name']}<={dim['max']}" for dim in report["runtime_dims"])
        or "none"
    )
    nodes = ", ".join(f"{node['symbol']}()" for node in report["nodes"]) or "none"
    print(
        f"  header:       {result.header_path} ({result.header_path.stat().st_size} B)"
    )
    print(f"  report:       {result.report_path}")
    print(f"  entrypoint:   {report['entrypoint']['symbol']}()")
    print(f"  node entries: {nodes}")
    print(f"  opsets:       {report['opsets']}")
    print(f"  fixed dims:   {report['dim_bindings'] or 'none'}")
    print(f"  runtime dims: {runtime_dims}")
    print(f"  kernels:      {len(report['kernels'])} ({', '.join(report['kernels'])})")
    print(
        f"  static mem:   {memory['static_bytes']} B "
        f"(weights {memory['weights_bytes']}, arena {memory['arena_bytes']})"
    )


def compare(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    print(f"  {label:<28} max |diff| = {np.abs(actual - expected).max():.3e}  OK")


# --------------------------------------------------------------------------------------
# 5. The artifact as a C program: what the header is actually for
# --------------------------------------------------------------------------------------

C_MAIN = """\
#define SCORING_IMPLEMENTATION
#include "scoring.h"
#include <stdio.h>

int main(void)
{{
    /* Buffers are sized by the macros the header states, i.e. for the maximum
       batch; this call passes {batch} rows and only those are read and written. */
    float x[SCORING_INPUT_X_COUNT] = {{{values}}};
    float score[SCORING_OUTPUT_SCORE_COUNT];

    if (scoring_run({batch}, x, score) != SCORING_OK) {{
        return 1;
    }}
    for (int row = 0; row < {batch}; ++row) {{
        printf("%.7f\\n", score[row]);
    }}
    return 0;
}}
"""


def run_as_c_program(directory: Path, x: np.ndarray) -> np.ndarray:
    """Compile a C program against the emitted header and return what it prints.

    Nothing of FNNX is involved past this point: the header, a C compiler, and libm.
    """
    values = ", ".join(f"{value:.9g}f" for value in x.ravel())
    source = directory / "main.c"
    binary = directory / "main"
    source.write_text(C_MAIN.format(batch=x.shape[0], values=values))
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            str(source),
            "-o",
            str(binary),
            "-lm",
        ],
        check=True,
    )
    output = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    return np.array([[float(line)] for line in output.stdout.split()], dtype=np.float32)


def main() -> None:
    scale, mlp = build_stages()
    onnx_dir = OUT_DIR / "onnx"
    scale_path = export_onnx(
        scale, onnx_dir / "scale.onnx", input_name="x", output_name="z"
    )
    mlp_path = export_onnx(mlp, onnx_dir / "mlp.onnx", input_name="z", output_name="s")
    print(f"exported {scale_path} and {mlp_path}")

    row = ("batch", FEATURES)
    score = ("batch", 1)
    bundle = write_bundle(
        OUT_DIR / "scoring.fnnx",
        [
            Stage("scale", scale_path, ("x",), ("z",), (row,), (row,)),
            Stage("mlp", mlp_path, ("z",), ("score",), (row,), (score,)),
        ],
        name="scoring",
        inputs=[_manifest_tensor("x", row)],
        outputs=[_manifest_tensor("score", score)],
    )
    print(f"wrote FNNX pipeline bundle {bundle}")

    x = np.random.default_rng(SEED).normal(size=(4, FEATURES)).astype(np.float32)
    with torch.no_grad():
        torch_score = mlp(scale(torch.from_numpy(x))).numpy()
    runtime_score = Runtime(str(bundle)).compute({"x": x}, {})["score"]

    # `batch` stays a per-call argument: buffers are sized for MAX_BATCH and every
    # entrypoint takes the actual size. Use `dim_bindings={"batch": 4}` instead to bake a
    # single size in; unbound symbolic dimensions default to 1.
    print("\n=== compile_bundle: the whole pipeline as one header ===")
    result = compile_bundle(bundle, OUT_DIR / "c", runtime_dims={"batch": MAX_BATCH})
    print_report(result)

    # Builds the header under `-std=c99 -Wall -Wextra -Werror` and binds it via ctypes.
    compiled = result.load()
    print("\n  running the compiled artifact:")
    compare("pipeline vs torch", compiled.run({"x": x})["score"], torch_score)
    compare("pipeline vs fnnx runtime", compiled.run({"x": x})["score"], runtime_score)
    # Every node is callable on its own, by the id `ops.json` gives it.
    compare(
        "node `scale` vs torch",
        compiled.run_node("scale", {"x": x})["z"],
        scale(torch.from_numpy(x)).detach().numpy(),
    )
    other = (
        np.random.default_rng(SEED + 1).normal(size=(7, FEATURES)).astype(np.float32)
    )
    with torch.no_grad():
        expected = mlp(scale(torch.from_numpy(other))).numpy()
    compare("same artifact, batch of 7", compiled.run({"x": other})["score"], expected)

    # Without a bundle there is no manifest name to take the symbol prefix from, so the
    # graph's own name is used unless `prefix` says otherwise.
    print("\n=== compile_onnx: one ONNX model, no bundle ===")
    onnx_result = compile_onnx(
        mlp_path, OUT_DIR / "c_mlp", dim_bindings={"batch": 4}, prefix="mlp"
    )
    print_report(onnx_result)
    with torch.no_grad():
        expected_mlp = mlp(torch.from_numpy(x)).numpy()
    print()
    compare("mlp vs torch", onnx_result.load().run({"z": x})["s"], expected_mlp)

    print("\n=== the header on its own: a C program, cc, and libm ===")
    compare(
        "compiled C binary vs torch", run_as_c_program(OUT_DIR / "c", x), torch_score
    )

    print(
        "\nThe same two compilations from the command line:\n"
        f"  python -m fnnx.extras.compilers.c {bundle} "
        f"-o {OUT_DIR / 'c'} --runtime-dim batch={MAX_BATCH}\n"
        f"  python -m fnnx.extras.compilers.c {mlp_path} "
        f"-o {OUT_DIR / 'c_mlp'} --dim batch=4 --prefix mlp"
    )


if __name__ == "__main__":
    main()
