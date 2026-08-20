import os
from typing import Any

from fnnx.envs._common import run_cmd
from fnnx.envs._common import select_pip_deps
from fnnx.console import console
from fnnx.utils import get_python_version
from fnnx.envs._common import which
from fnnx.envs.base import BaseEnvManager


class UvEnvManager(BaseEnvManager):
    """
    Ephemeral uv-based environment manager.
    """

    def __init__(
        self, env_spec: dict[str, Any], accelerator: str | None = None
    ) -> None:
        self.env_spec = env_spec
        self.accelerator = ("cpu" if accelerator is None else accelerator).lower()

        build_deps: list[str] = list(env_spec.get("build_dependencies", []))
        if len(build_deps) > 0:
            console.warn("UvEnvManager does not support build dependencies. Ignoring.")
        self.python_version = (
            env_spec["python_version"]
            if "python_version" in env_spec
            else get_python_version(micro=False)
        )
        deps: list[dict[str, Any]] = list(env_spec.get("dependencies", []))
        if not deps:
            deps.append({"package": "fnnx[core]"})

        self.selected_pip_deps = select_pip_deps(deps, self.accelerator)

        self._uv_exe = self._get_uv_exe()
        self._env_dir: str | None = None
        self._cmd_prefix = [
            self._uv_exe,
            "run",
            "--no-project",
            "--isolated",
            "--python",
            self.python_version,
        ]
        for dep in self.selected_pip_deps:
            self._cmd_prefix.extend(["--with", dep["package"]])

    def _get_uv_exe(self) -> str:
        exe = os.environ.get("FNNX_UV_EXE") or which("uv")
        if not exe:
            raise RuntimeError(
                "uv executable not found. Install uv or set FNNX_UV_EXE."
            )
        return exe

    def ensure(self):
        """Pre-warm the uv environment cache by running a no-op command."""
        console.info("Preparing uv environment with dependencies...")
        warmup_cmd = self._cmd_prefix + [
            "python",
            "-c",
            "import sys; sys.exit(0)",
        ]

        try:
            run_cmd(warmup_cmd, capture=False)
            console.info("Environment ready.")
        except RuntimeError:
            console.error("Failed to prepare uv environment.")
            raise

    def python_cmd(self, argv: list[str]) -> list[str]:
        deps = []
        for dep in self.selected_pip_deps:
            deps.extend(["--with", dep["package"]])

        return self._cmd_prefix + [
            "python",
            "-u",
            *argv,
        ]
