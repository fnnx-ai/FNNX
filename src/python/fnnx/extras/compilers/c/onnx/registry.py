"""Kernel registry and opset-driven dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import onnx.defs

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.loader import display_domain, normalize_domain

G = TypeVar("G")


@dataclass(frozen=True)
class KernelSpec(Generic[G]):
    domain: str
    op_type: str
    since_version: int
    generator: G


def latest_semantic_revision(
    domain: str, op_type: str, opset_version: int
) -> int | None:
    """`since_version` of the ONNX schema in effect for the op at `opset_version`.

    None when the installed `onnx` package defines no schema for the op at that version.
    """
    try:
        schema = onnx.defs.get_schema(op_type, opset_version, normalize_domain(domain))
    except onnx.defs.SchemaError:
        return None
    return schema.since_version


class KernelRegistry(Generic[G]):
    """Maps `(domain, op_type)` to version-keyed generators, mirroring ONNX's op versioning."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], dict[int, KernelSpec[G]]] = {}

    def register(
        self, domain: str, op_type: str, since_version: int, generator: G
    ) -> None:
        normalized = normalize_domain(domain)
        if latest_semantic_revision(normalized, op_type, since_version) is None:
            raise ValueError(
                f"ONNX defines no schema for `{op_type}` "
                f"(domain `{display_domain(normalized)}`) at opset version {since_version}."
            )
        versions = self._specs.setdefault((normalized, op_type), {})
        if since_version in versions:
            raise ValueError(
                f"A kernel for `{op_type}` (domain `{display_domain(normalized)}`) is "
                f"already registered at since_version {since_version}."
            )
        versions[since_version] = KernelSpec(
            domain=normalized,
            op_type=op_type,
            since_version=since_version,
            generator=generator,
        )

    def registered_ops(self) -> list[tuple[str, str]]:
        """Every `(domain, op_type)` a kernel is registered for, in a stable order."""
        return sorted(self._specs)

    def registered_versions(self, domain: str, op_type: str) -> list[int]:
        return sorted(self._specs.get((normalize_domain(domain), op_type), {}))

    def select(
        self, domain: str, op_type: str, opset_version: int
    ) -> KernelSpec[G] | None:
        """Highest-versioned kernel valid at `opset_version`, or None if none can be vouched for.

        The semantic-revision guard rejects an otherwise applicable kernel when ONNX revised
        the op's spec after the kernel's `since_version` and at or below the requested
        version: old semantics are never silently applied to a newer opset.
        """
        normalized = normalize_domain(domain)
        versions = self._specs.get((normalized, op_type), {})
        applicable = [version for version in versions if version <= opset_version]
        if not applicable:
            return None
        spec = versions[max(applicable)]
        revision = latest_semantic_revision(normalized, op_type, opset_version)
        if revision is None or revision > spec.since_version:
            return None
        return spec

    def unsupported_op_error(
        self,
        domain: str,
        op_type: str,
        opset_version: int,
        *,
        node_name: str | None = None,
    ) -> CompileError:
        """Build — but do not raise — the error for an op no registered kernel can serve.

        Callers fall through to function expansion before raising it.
        """
        normalized = normalize_domain(domain)
        versions = self.registered_versions(normalized, op_type)
        prefix = f"Node `{node_name}`: " if node_name else ""
        message = (
            f"{prefix}op `{op_type}` (domain `{display_domain(normalized)}`) is not "
            f"supported at opset version {opset_version}"
        )
        if not versions:
            return CompileError(f"{message}: no kernel is registered for this op.")
        nearest = min(
            versions, key=lambda version: (abs(version - opset_version), version)
        )
        if all(version > opset_version for version in versions):
            reason = "every registered kernel targets a newer opset version"
        else:
            revision = latest_semantic_revision(normalized, op_type, opset_version)
            if revision is None:
                reason = "ONNX defines no schema for this op at that version"
            else:
                reason = (
                    f"ONNX revised this op at opset version {revision} and no "
                    "registered kernel covers that revision"
                )
        return CompileError(
            f"{message}: {reason}. Nearest supported version: {nearest}."
        )
