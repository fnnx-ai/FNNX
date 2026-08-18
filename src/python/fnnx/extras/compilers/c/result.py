"""The object every `compile_*` entrypoint returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fnnx.extras.compilers.c.harness import CompiledModel


@dataclass(frozen=True)
class CompileResult:
    header_path: Path
    report_path: Path
    report: dict[str, Any]

    def load(self, *, compiler: str | None = None) -> CompiledModel:
        """Build this artifact into a shared library and bind it for execution."""
        # Imported here: driving an artifact needs numpy and a C compiler, which merely
        # compiling one does not.
        from fnnx.extras.compilers.c.harness import load_compiled

        return load_compiled(self.report_path, compiler=compiler)
