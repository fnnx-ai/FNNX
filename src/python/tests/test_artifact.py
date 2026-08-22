"""Root-file reading that every consumer shares: manifest patches, optional root files,
unique IO names, and what a tar member may put on disk."""

import json
import os
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any

from fnnx.device import DeviceMap
from fnnx.envs.base import BaseEnvManager
from fnnx.handlers._common import unpack_model
from fnnx.handlers.local import LocalHandler, LocalHandlerConfig
from fnnx.handlers.stdio import StdIOHandler, StdIOHandlerConfig

OPTIONAL_ROOT_FILES = ("dtypes.json", "env.json", "meta.json")


def _io_entry(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "content_type": "NDJSON",
        "dtype": "Array[float32]",
        "shape": [],
    }


def _manifest(
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "variant": "pipeline",
        "name": "base",
        "producer_name": "tests",
        "producer_version": "1.0",
        "producer_tags": [],
        "inputs": inputs if inputs is not None else [_io_entry("x")],
        "outputs": outputs if outputs is not None else [],
        "dynamic_attributes": [],
        "env_vars": [],
    }


def _write_artifact(
    root: Path,
    manifest: dict[str, Any] | None = None,
    patches: dict[str, Any] | None = None,
    optional_root_files: bool = True,
) -> Path:
    root.mkdir(parents=True)
    documents: dict[str, Any] = {
        "manifest.json": manifest if manifest is not None else _manifest(),
        "ops.json": [],
        "variant_config.json": {"nodes": []},
    }
    if optional_root_files:
        documents.update({"dtypes.json": {}, "env.json": {}, "meta.json": []})
    documents.update(patches or {})
    for filename, document in documents.items():
        (root / filename).write_text(json.dumps(document), encoding="utf-8")
    return root


def _load_local(artifact: Path) -> LocalHandler:
    return LocalHandler(
        str(artifact),
        DeviceMap(accelerator="cpu", node_device_map={}),
        LocalHandlerConfig(auto_cleanup=False),
    )


class _UnexpectedEnvManager(BaseEnvManager):
    def __init__(self, env_spec: dict[str, Any], accelerator: str | None = None):
        raise AssertionError("the manifest must be read before the environment")

    def ensure(self) -> None:
        raise AssertionError

    def python_cmd(self, argv: list[str]) -> list[str]:
        raise AssertionError


class TestEffectiveManifest(unittest.TestCase):
    def test_local_handler_reads_the_patched_manifest(self) -> None:
        patches = {
            "manifest-0001.patch.json": [
                {"op": "replace", "path": "/name", "value": "patched"},
                {"op": "add", "path": "/inputs/-", "value": _io_entry("z")},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "patched.fnnx", patches=patches)
            handler = _load_local(artifact)
            try:
                self.assertEqual(handler.manifest["name"], "patched")
                self.assertIn("z", handler.input_specs)
            finally:
                handler.executor.shutdown()
                handler.op_executor.shutdown()

    def test_patches_apply_in_ascending_byte_order(self) -> None:
        patches = {
            "manifest-b.patch.json": [
                {"op": "replace", "path": "/name", "value": "second"}
            ],
            "manifest-a.patch.json": [
                {"op": "replace", "path": "/name", "value": "first"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "ordered.fnnx", patches=patches)
            handler = _load_local(artifact)
            try:
                self.assertEqual(handler.manifest["name"], "second")
            finally:
                handler.executor.shutdown()
                handler.op_executor.shutdown()

    def test_patch_with_a_forbidden_operation_is_rejected(self) -> None:
        patches = {
            "manifest-0001.patch.json": [{"op": "remove", "path": "/name"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "removing.fnnx", patches=patches)
            with self.assertRaisesRegex(ValueError, "remove"):
                _load_local(artifact)

    def test_stdio_handler_reads_the_patched_manifest(self) -> None:
        json_entry = {"name": "y", "content_type": "JSON", "dtype": "ext::record"}
        patches = {
            "manifest-0001.patch.json": [
                {"op": "add", "path": "/outputs/-", "value": json_entry}
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(
                Path(directory) / "patched.fnnx", patches=patches
            )
            with self.assertRaises(ValueError) as error:
                StdIOHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    StdIOHandlerConfig(
                        auto_cleanup=False, env_manager=_UnexpectedEnvManager
                    ),
                )
        self.assertIn("y", str(error.exception))


class TestOptionalRootFiles(unittest.TestCase):
    def test_missing_optional_root_files_are_read_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(
                Path(directory) / "sparse.fnnx", optional_root_files=False
            )
            for filename in OPTIONAL_ROOT_FILES:
                self.assertFalse((artifact / filename).exists())
            handler = _load_local(artifact)
            try:
                self.assertEqual(handler.dtypes_manager.dtypes, {})
            finally:
                handler.executor.shutdown()
                handler.op_executor.shutdown()

    def test_stdio_reports_unsupported_environment_without_env_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(
                Path(directory) / "sparse.fnnx", optional_root_files=False
            )
            with self.assertRaisesRegex(RuntimeError, "environment kind"):
                StdIOHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    StdIOHandlerConfig(
                        auto_cleanup=False, env_manager=_UnexpectedEnvManager
                    ),
                )


class TestUniqueIONames(unittest.TestCase):
    def test_duplicate_output_names_are_rejected(self) -> None:
        manifest = _manifest(outputs=[_io_entry("y"), _io_entry("y")])
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "duplicate.fnnx", manifest)
            with self.assertRaisesRegex(ValueError, "output `y`"):
                _load_local(artifact)

    def test_duplicate_input_names_are_rejected(self) -> None:
        manifest = _manifest(inputs=[_io_entry("x"), _io_entry("x")])
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "duplicate.fnnx", manifest)
            with self.assertRaisesRegex(ValueError, "input `x`"):
                _load_local(artifact)

    def test_stdio_rejects_duplicate_output_names(self) -> None:
        manifest = _manifest(outputs=[_io_entry("y"), _io_entry("y")])
        with tempfile.TemporaryDirectory() as directory:
            artifact = _write_artifact(Path(directory) / "duplicate.fnnx", manifest)
            with self.assertRaisesRegex(ValueError, "output `y`"):
                StdIOHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    StdIOHandlerConfig(
                        auto_cleanup=False, env_manager=_UnexpectedEnvManager
                    ),
                )


def _add_file(tar: tarfile.TarFile, name: str, content: object) -> None:
    data = json.dumps(content).encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, fileobj=BytesIO(data))


def _add_special(tar: tarfile.TarFile, name: str, kind: bytes, target: str = "") -> None:
    info = tarfile.TarInfo(name=name)
    info.type = kind
    info.linkname = target
    if kind in (tarfile.CHRTYPE, tarfile.BLKTYPE):
        info.devmajor = 1
        info.devminor = 3
    tar.addfile(info)


class TestTarExtraction(unittest.TestCase):
    def _write_tar(self, path: Path) -> Path:
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
            _add_file(tar, "manifest.json", _manifest())
            _add_file(tar, "ops.json", [])
            _add_file(tar, "variant_config.json", {"nodes": []})
            _add_special(tar, "symlinked.json", tarfile.SYMTYPE, "manifest.json")
            _add_special(tar, "hardlinked.json", tarfile.LNKTYPE, "manifest.json")
            _add_special(tar, "device", tarfile.CHRTYPE)
            _add_file(tar, "/absolute.json", {"absolute": True})
            # A doubled leading slash is a second absolute form, with its own POSIX root.
            _add_file(tar, "//double-absolute.json", {"absolute": True})
            _add_file(tar, "../escaped.json", {"escaped": True})
        return path

    def test_extraction_ignores_links_devices_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._write_tar(Path(directory) / "artifact.fnnx.tar")
            unpacked, temporary = unpack_model(str(archive))
            try:
                self.assertTrue(temporary)
                written = {
                    os.path.relpath(os.path.join(root, name), unpacked)
                    for root, _, names in os.walk(unpacked)
                    for name in names
                }
                self.assertEqual(
                    written, {"manifest.json", "ops.json", "variant_config.json"}
                )
                self.assertFalse(
                    os.path.exists(os.path.join(os.path.dirname(unpacked), "escaped.json"))
                )
            finally:
                for root, _, names in os.walk(unpacked, topdown=False):
                    for name in names:
                        os.unlink(os.path.join(root, name))
                    os.rmdir(root)

    def test_a_tar_artifact_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._write_tar(Path(directory) / "artifact.fnnx.tar")
            handler = LocalHandler(
                str(archive),
                DeviceMap(accelerator="cpu", node_device_map={}),
                LocalHandlerConfig(auto_cleanup=True),
            )
            try:
                self.assertEqual(handler.manifest["name"], "base")
            finally:
                handler.executor.shutdown()
                handler.op_executor.shutdown()


if __name__ == "__main__":
    unittest.main()
