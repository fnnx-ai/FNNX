"""The standalone ONNX-to-C entrypoint and the compile report it writes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from onnx import ModelProto, TensorProto

from fnnx import __version__
from fnnx.extras.compilers.c.onnx.codegen import IOTensor, Program, build_program
from fnnx.extras.compilers.c.onnx.dtypes import c_type, element_size, numpy_dtype_name
from fnnx.extras.compilers.c.onnx.frontend import prepare_model
from fnnx.extras.compilers.c.onnx.header import buffer_bytes, render_header
from fnnx.extras.compilers.c.onnx.loader import display_domain, load_model
from fnnx.extras.compilers.c.onnx.runtime_dims import ShapeTerm, resolve_runtime_dims
from fnnx.extras.compilers.c.onnx.specialize import specialize
from fnnx.extras.compilers.c.result import CompileResult

COMPILER = "fnnx.extras.compilers.c"


def compile_onnx(
    source: str | os.PathLike[str] | ModelProto,
    output_dir: str | os.PathLike[str],
    *,
    dim_bindings: Mapping[str, int] | None = None,
    runtime_dims: Mapping[str, int] | None = None,
    prefix: str | None = None,
) -> CompileResult:
    """Compile an ONNX model into a single self-contained C99 header plus a report.

    `source` is a path to a `.onnx` file or an in-memory `ModelProto`. `prefix` defaults to
    the graph name, sanitized to a C identifier. `runtime_dims` maps a symbolic dimension to
    the largest size the artifact must serve, leaving the actual size to each call; every
    other symbolic dimension is fixed at compile time. Compilation is all-or-nothing:
    nothing is written unless the whole model compiles.
    """
    dims = resolve_runtime_dims(runtime_dims, dim_bindings)
    loaded = load_model(source)

    def build(bindings: Mapping[str, int]) -> Program:
        return build_program(
            prepare_model(loaded, dim_bindings=bindings),
            prefix=prefix,
            runtime_dims=dims,
        )

    program = (
        specialize(build, dims, dim_bindings=dim_bindings or {})
        if dims
        else build(dim_bindings or {})
    )
    return write_artifact(
        program,
        output_dir,
        options={
            "prefix": prefix,
            "dim_bindings": dict(sorted((dim_bindings or {}).items())),
            "runtime_dims": dict((runtime_dims or {}).items()),
        },
    )


def write_artifact(
    program: Program, output_dir: str | os.PathLike[str], *, options: dict[str, Any]
) -> CompileResult:
    """Render `program` into its output directory, creating the directory if needed.

    The last step of every compilation, and the only one that writes: a model that fails to
    compile leaves no partial artifact behind.
    """
    report = build_report(program, options=options)
    header = render_header(program)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    header_path = directory / report["header"]
    report_path = directory / f"{program.prefix}_report.json"
    header_path.write_text(header, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CompileResult(
        header_path=header_path, report_path=report_path, report=report
    )


def build_report(program: Program, *, options: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable description of the artifact, for tooling and for the harness."""
    weights_bytes = buffer_bytes(program.weights)
    arena_bytes = buffer_bytes(program.scratch)
    return {
        "fnnx_version": __version__,
        "compiler": COMPILER,
        "prefix": program.prefix,
        "header": f"{program.prefix}.h",
        "graph": program.graph_name,
        "options": options,
        "dim_bindings": dict(sorted(program.dim_bindings.items())),
        "runtime_dims": [
            {
                "name": dim.name,
                "max": dim.maximum,
                "parameter": dim.c_name,
                "macro": dim.macro(program.prefix),
            }
            for dim in program.runtime_dims
        ],
        "opsets": {
            display_domain(domain): version
            for domain, version in sorted(program.opsets.items())
        },
        "kernels": [function.name for function in program.functions],
        "class_labels": [
            {
                "tensor": table.tensor,
                "symbol": table.symbol,
                "macro": table.macro,
                "dtype": "str" if table.elem_type == TensorProto.STRING else "int64",
                "values": list(table.values),
            }
            for table in program.labels
        ],
        "memory": {
            "weights_bytes": weights_bytes,
            "arena_bytes": arena_bytes,
            "static_bytes": weights_bytes + arena_bytes,
        },
        "entrypoint": {
            "symbol": f"{program.prefix}_run",
            "inputs": [_tensor_report(tensor) for tensor in program.inputs],
            "outputs": [_tensor_report(tensor) for tensor in program.outputs],
        },
        "nodes": [
            {
                "id": node.id,
                "symbol": node.symbol,
                "inputs": [_tensor_report(tensor) for tensor in node.inputs],
                "outputs": [_tensor_report(tensor) for tensor in node.outputs],
            }
            for node in program.nodes
        ],
    }


def _tensor_report(tensor: IOTensor) -> dict[str, Any]:
    """One tensor of an entrypoint's signature; `shape` is the buffer's capacity.

    With runtime dimensions in play the buffer is sized for their maxima while a call works
    at whatever sizes it passes, so `runtime_shape` says how each axis follows from them —
    which is what the load-and-run harness sizes its arrays by.
    """
    report = {
        "name": tensor.name,
        "c_name": tensor.c_name,
        "macro": tensor.macro,
        "dtype": numpy_dtype_name(tensor.elem_type),
        "c_type": c_type(tensor.elem_type),
        "shape": list(tensor.shape),
        "elem_count": tensor.elem_count,
        "bytes": tensor.elem_count * element_size(tensor.elem_type),
    }
    if tensor.runtime_shape:
        report["runtime_shape"] = [_axis_report(term) for term in tensor.runtime_shape]
    return report


def _axis_report(term: ShapeTerm) -> Any:
    if term.dim is None:
        return term.size
    return {"dim": term.dim, "coefficient": term.coefficient}
