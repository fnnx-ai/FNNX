"""Dimensions the caller sizes per call, within a maximum fixed at compile time."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.emit import sanitize_identifier


@dataclass(frozen=True)
class RuntimeDim:
    """A symbolic dimension compiled for the whole family of sizes `[1, maximum]`.

    `identifier` is the dimension's name sanitized to a C identifier; the entrypoint
    parameter carrying the actual value and the macro publishing the maximum are both
    derived from it, so that two dimensions can never name the same one.
    """

    name: str
    maximum: int
    identifier: str

    @property
    def c_name(self) -> str:
        return f"dim_{self.identifier}"

    def macro(self, prefix: str) -> str:
        return f"{prefix.upper()}_DIM_{self.identifier.upper()}_MAX"


@dataclass(frozen=True)
class ShapeTerm:
    """One axis of a tensor: a constant extent, or a multiple of a runtime dimension.

    `size` is the extent at the dimension's maximum — the capacity the artifact's buffers
    and macros are sized for — while `extent` gives the one a particular call works at.
    """

    size: int
    dim: str | None = None
    coefficient: int = 0

    def extent(self, values: Mapping[str, int]) -> int:
        return self.size if self.dim is None else self.coefficient * values[self.dim]


def resolve_runtime_dims(
    runtime_dims: Mapping[str, int] | None, dim_bindings: Mapping[str, int] | None
) -> tuple[RuntimeDim, ...]:
    """Validate the requested runtime dimensions, in the order they were declared."""
    identifiers: dict[str, str] = {}
    resolved = []
    for name, maximum in (runtime_dims or {}).items():
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise CompileError(
                f"Runtime dimension `{name}` needs a maximum of at least 1, got "
                f"{maximum!r}."
            )
        if name in (dim_bindings or {}):
            raise CompileError(
                f"Dimension `{name}` is both bound to "
                f"{(dim_bindings or {})[name]} and declared runtime; a dimension is "
                "either fixed at compile time or sized per call."
            )
        identifier = sanitize_identifier(name, fallback="")
        if not identifier:
            raise CompileError(
                f"Runtime dimension `{name}` has no C identifier to derive its "
                "entrypoint parameter from; rename it."
            )
        if identifier in identifiers:
            raise CompileError(
                f"Runtime dimensions `{identifiers[identifier]}` and `{name}` both "
                f"sanitize to the C identifier `{identifier}`; rename one of them."
            )
        identifiers[identifier] = name
        resolved.append(RuntimeDim(name, maximum, identifier))
    return tuple(resolved)
