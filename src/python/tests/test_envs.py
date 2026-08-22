import json
from pathlib import Path
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from fnnx.device import DeviceMap
from fnnx.envs._common import select_pip_deps
from fnnx.envs.base import BaseEnvManager
from fnnx.envs.conda import CondaLikeEnvManager
from fnnx.envs.uv import UvEnvManager
from fnnx.handlers.stdio import StdIOHandler, StdIOHandlerConfig


def _write_artifact(root: Path, environment: dict[str, Any]) -> Path:
    root.mkdir()
    documents: dict[str, Any] = {
        "manifest.json": {
            "variant": "pipeline",
            "producer_name": "tests",
            "producer_version": "1.0",
            "producer_tags": [],
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        },
        "ops.json": [],
        "variant_config.json": {"nodes": []},
        "dtypes.json": {},
        "env.json": environment,
    }
    for filename, document in documents.items():
        (root / filename).write_text(json.dumps(document), encoding="utf-8")
    return root


class _UnexpectedEnvManager(BaseEnvManager):
    def __init__(
        self, env_spec: dict[str, Any], accelerator: str | None = None
    ) -> None:
        raise AssertionError("unsupported environments must not be provisioned")

    def ensure(self) -> None:
        raise AssertionError

    def python_cmd(self, argv: list[str]) -> list[str]:
        raise AssertionError


class TestEnvironmentKindSelection(unittest.TestCase):
    def test_stdio_rejects_and_names_unsupported_environment_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = _write_artifact(
                Path(temporary_directory) / "unsupported.fnnx",
                {"vendor::environment": {"setting": "value"}},
            )

            with self.assertRaisesRegex(
                RuntimeError, "unsupported.*vendor::environment"
            ):
                StdIOHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    StdIOHandlerConfig(
                        auto_cleanup=False, env_manager=_UnexpectedEnvManager
                    ),
                )

    def test_stdio_rejects_an_empty_environment_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = _write_artifact(Path(temporary_directory) / "missing.fnnx", {})

            with self.assertRaisesRegex(RuntimeError, "unsupported.*none"):
                StdIOHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    StdIOHandlerConfig(
                        auto_cleanup=False, env_manager=_UnexpectedEnvManager
                    ),
                )


class TestEnvironmentConditions(unittest.TestCase):
    def test_platform_matching_requires_exact_case_insensitive_membership(self) -> None:
        dependencies = [
            {"package": "prefix", "condition": {"platform": ["x86"]}},
            {"package": "exact", "condition": {"platform": ["X86_64"]}},
            {"package": "unrestricted", "condition": {"platform": []}},
        ]

        with (
            patch("fnnx.envs._common.platform.machine", return_value="x86_64"),
            patch("fnnx.envs._common.platform.system", return_value="Linux"),
        ):
            selected = select_pip_deps(dependencies, accelerator="cpu")

        self.assertEqual(
            [dependency["package"] for dependency in selected],
            ["exact", "unrestricted"],
        )

    def test_accelerator_defaults_only_when_absent(self) -> None:
        dependencies = [
            {"package": "cpu-only", "condition": {"accelerator": ["cpu"]}}
        ]

        self.assertEqual(
            [
                dependency["package"]
                for dependency in select_pip_deps(dependencies, accelerator=None)
            ],
            ["cpu-only"],
        )
        self.assertEqual(select_pip_deps(dependencies, accelerator=""), [])


class TestEnvironmentManagerDefaults(unittest.TestCase):
    def test_conda_manager_defaults_only_when_fields_are_absent(self) -> None:
        environment = {
            "python_version": "",
            "build_dependencies": [],
            "dependencies": [{"package": "example==1"}],
            "conda_channels": [],
        }
        with (
            patch.object(CondaLikeEnvManager, "_get_exe", return_value="conda"),
            patch("fnnx.envs.conda.get_python_version") as get_default_version,
        ):
            manager = CondaLikeEnvManager(environment, accelerator="")

        get_default_version.assert_not_called()
        self.assertEqual(manager.accelerator, "")
        self.assertEqual(manager.python_version, "")
        self.assertEqual(manager.channels, [])

    def test_uv_manager_defaults_python_only_when_field_is_absent(self) -> None:
        environment = {
            "python_version": "",
            "build_dependencies": [],
            "dependencies": [{"package": "example==1"}],
        }
        with (
            patch.object(UvEnvManager, "_get_uv_exe", return_value="uv"),
            patch("fnnx.envs.uv.get_python_version") as get_default_version,
        ):
            manager = UvEnvManager(environment, accelerator="")

        get_default_version.assert_not_called()
        self.assertEqual(manager.accelerator, "")
        self.assertEqual(manager.python_version, "")


if __name__ == "__main__":
    unittest.main()
