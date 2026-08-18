"""Model loading, external-data resolution, and per-domain opset resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import onnx
import onnx.defs
from onnx import ModelProto

# onnx exposes no public walker over every tensor of a model (initializers plus the
# ones nested in subgraph attributes), which is what external-data detection needs.
from onnx.external_data_helper import (
    _get_all_tensors,
    load_external_data_for_model,
    uses_external_data,
)

from fnnx.extras.compilers.c.errors import CompileError

STANDARD_DOMAIN = ""
ML_DOMAIN = "ai.onnx.ml"
SUPPORTED_DOMAINS = (STANDARD_DOMAIN, ML_DOMAIN)

_DOMAIN_ALIASES = {"ai.onnx": STANDARD_DOMAIN}


def normalize_domain(domain: str) -> str:
    return _DOMAIN_ALIASES.get(domain, domain)


def display_domain(domain: str) -> str:
    normalized = normalize_domain(domain)
    return "ai.onnx" if normalized == STANDARD_DOMAIN else normalized


@lru_cache(maxsize=None)
def max_supported_opset(domain: str) -> int:
    """Highest opset version the installed `onnx` package defines for `domain`."""
    normalized = normalize_domain(domain)
    if normalized == STANDARD_DOMAIN:
        return onnx.defs.onnx_opset_version()
    versions = [
        schema.since_version
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.domain == normalized
    ]
    if not versions:
        raise ValueError(
            f"The installed `onnx` package defines no schemas for domain `{normalized}`."
        )
    return max(versions)


@dataclass(frozen=True)
class LoadedModel:
    model: ModelProto
    opsets: dict[str, int]

    def opset_for(self, domain: str) -> int:
        normalized = normalize_domain(domain)
        version = self.opsets.get(normalized)
        if version is None:
            raise CompileError(
                f"Model imports no opset for domain `{display_domain(normalized)}`."
            )
        return version


def load_model(
    source: str | os.PathLike[str] | ModelProto,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> LoadedModel:
    """Load an ONNX model, embed its external data, and resolve its opset imports.

    `base_dir` is where external tensor files are looked up; it defaults to the directory
    holding `source` when a path is given, and is required for an in-memory proto whose
    tensors live in external files. The caller's proto is never mutated.
    """
    if isinstance(source, ModelProto):
        model = ModelProto()
        model.CopyFrom(source)
        data_dir = Path(base_dir) if base_dir is not None else None
    else:
        path = Path(source)
        if not path.is_file():
            raise CompileError(f"ONNX model file not found: `{path}`.")
        try:
            model = onnx.load_model(os.fspath(path), load_external_data=False)
        except Exception as exc:
            raise CompileError(f"Failed to parse ONNX model `{path}`: {exc}") from exc
        data_dir = Path(base_dir) if base_dir is not None else path.parent

    if model.ir_version > onnx.IR_VERSION:
        raise CompileError(
            f"Model IR version {model.ir_version} is newer than the installed `onnx` package "
            f"({onnx.__version__}) supports (at most {onnx.IR_VERSION}); "
            "upgrade `onnx` to compile this model."
        )
    _embed_external_data(model, data_dir)
    return LoadedModel(model=model, opsets=resolve_opsets(model))


def resolve_opsets(model: ModelProto) -> dict[str, int]:
    """Map every imported domain, normalized, to the opset version the model requests."""
    opsets: dict[str, int] = {}
    for imported in model.opset_import:
        domain = normalize_domain(imported.domain)
        if domain not in SUPPORTED_DOMAINS:
            supported = " and ".join(display_domain(d) for d in SUPPORTED_DOMAINS)
            raise CompileError(
                f"Model imports unsupported opset domain `{imported.domain}`; "
                f"the C compiler supports only {supported}."
            )
        if imported.version < 1:
            raise CompileError(
                f"Model imports invalid opset version {imported.version} "
                f"for domain `{display_domain(domain)}`."
            )
        maximum = max_supported_opset(domain)
        if imported.version > maximum:
            raise CompileError(
                f"Model imports opset version {imported.version} for domain "
                f"`{display_domain(domain)}`, but the installed `onnx` package "
                f"({onnx.__version__}) defines at most version {maximum} for that domain; "
                "upgrade `onnx` to compile this model."
            )
        previous = opsets.get(domain)
        if previous is not None and previous != imported.version:
            raise CompileError(
                f"Model imports conflicting opset versions {previous} and {imported.version} "
                f"for domain `{display_domain(domain)}`."
            )
        opsets[domain] = imported.version
    if not opsets:
        raise CompileError("Model imports no opsets.")
    return opsets


def _embed_external_data(model: ModelProto, data_dir: Path | None) -> None:
    external = [
        tensor for tensor in _get_all_tensors(model) if uses_external_data(tensor)
    ]
    if not external:
        return
    if data_dir is None:
        names = ", ".join(f"`{tensor.name}`" for tensor in external[:3])
        raise CompileError(
            f"Model stores tensors ({names}) in external files, but no directory to resolve "
            "them against is known; pass `base_dir` when compiling an in-memory model."
        )
    try:
        load_external_data_for_model(model, os.fspath(data_dir))
    except Exception as exc:
        raise CompileError(
            f"Failed to load external tensor data from `{data_dir}`: {exc}"
        ) from exc
