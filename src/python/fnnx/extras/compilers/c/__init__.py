"""Compiler from FNNX bundles and ONNX models to a self-contained C99 header."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from fnnx.extras.compilers.c.errors import CompileError, HarnessError
from fnnx.extras.compilers.c.result import CompileResult

if TYPE_CHECKING:
    from fnnx.extras.compilers.c.bundle import compile_bundle
    from fnnx.extras.compilers.c.harness import CompiledModel, load_compiled
    from fnnx.extras.compilers.c.onnx.api import compile_onnx

__all__ = [
    "CompileError",
    "CompileResult",
    "CompiledModel",
    "HarnessError",
    "compile_bundle",
    "compile_onnx",
    "load_compiled",
]

# Compiling needs the optional `onnx` package and the load-and-run harness needs numpy and
# a system C compiler; importing them lazily keeps the error types importable without
# either, and leaves the missing-dependency messages intact.
_LAZY = {
    "compile_bundle": "fnnx.extras.compilers.c.bundle",
    "compile_onnx": "fnnx.extras.compilers.c.onnx.api",
    "CompiledModel": "fnnx.extras.compilers.c.harness",
    "load_compiled": "fnnx.extras.compilers.c.harness",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
