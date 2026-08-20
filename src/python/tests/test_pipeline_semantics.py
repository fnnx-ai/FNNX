import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, cast

import numpy as np

from fnnx.device import DeviceMap
from fnnx.envs.base import BaseEnvManager
from fnnx.handlers.local import LocalHandler, LocalHandlerConfig
from fnnx.handlers.stdio import StdIOHandler, StdIOHandlerConfig
from fnnx.handlers.stdio.worker import _dynamic_attributes_from_body


def _io_entry(name: str, content_type: str = "NDJSON") -> dict[str, Any]:
    if content_type == "JSON":
        return {"name": name, "content_type": content_type, "dtype": "ext::record"}
    return {
        "name": name,
        "content_type": content_type,
        "dtype": "Array[float32]",
        "shape": [],
    }


def _manifest() -> dict[str, Any]:
    return {
        "variant": "pipeline",
        "producer_name": "tests",
        "producer_version": "1.0",
        "producer_tags": [],
        "inputs": [_io_entry("x")],
        "outputs": [_io_entry("y")],
        "dynamic_attributes": [],
        "env_vars": [],
    }


def _op_instance(
    op_instance_id: str = "op", input_count: int = 1, output_count: int = 1
) -> dict[str, Any]:
    io_spec = {"dtype": "Array[float32]", "shape": []}
    return {
        "id": op_instance_id,
        "op": "ONNX_v1",
        "inputs": [deepcopy(io_spec) for _ in range(input_count)],
        "outputs": [deepcopy(io_spec) for _ in range(output_count)],
        "attributes": {
            "opsets": [{"domain": "ai.onnx", "version": 12}],
            "has_external_data": False,
            "onnx_ir_version": 7,
        },
        "dynamic_attributes": {},
    }


def _node(
    op_instance_id: str = "op",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "op_instance_id": op_instance_id,
        "inputs": inputs if inputs is not None else ["x"],
        "outputs": outputs if outputs is not None else ["y"],
        "extra_dynattrs": {},
    }


def _write_pipeline_artifact(
    root: Path,
    manifest: dict[str, Any] | None = None,
    ops: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> Path:
    root.mkdir()
    documents = {
        "manifest.json": manifest if manifest is not None else _manifest(),
        "ops.json": ops if ops is not None else [],
        "variant_config.json": {"nodes": nodes if nodes is not None else []},
        "dtypes.json": {},
    }
    for filename, document in documents.items():
        (root / filename).write_text(json.dumps(document), encoding="utf-8")
    return root


class _UnexpectedEnvManager(BaseEnvManager):
    def __init__(self, env_spec: dict[str, Any], accelerator: str | None = None):
        raise AssertionError(
            "environment setup must not run before pipeline validation"
        )

    def ensure(self) -> None:
        raise AssertionError

    def python_cmd(self, argv: list[str]) -> list[str]:
        raise AssertionError


class _StaticClient:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.last_body: dict[str, Any] | None = None

    def request(
        self,
        handler: str,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.last_body = body
        return self.response

    def close(self) -> None:
        pass


class TestPipelineValidation(unittest.TestCase):
    def _assert_local_load_rejected(
        self,
        expected_identifier: str,
        manifest: dict[str, Any] | None = None,
        ops: list[dict[str, Any]] | None = None,
        nodes: list[dict[str, Any]] | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = _write_pipeline_artifact(
                Path(temporary_directory) / "invalid.fnnx", manifest, ops, nodes
            )
            with self.assertRaises(ValueError) as error:
                LocalHandler(
                    str(artifact),
                    DeviceMap(accelerator="cpu", node_device_map={}),
                    LocalHandlerConfig(auto_cleanup=False),
                )
        self.assertIn(expected_identifier, str(error.exception))

    def test_rejects_undeclared_op_instance(self) -> None:
        self._assert_local_load_rejected("missing", nodes=[_node("missing")])

    def test_rejects_input_arity_mismatch(self) -> None:
        self._assert_local_load_rejected(
            "op", ops=[_op_instance()], nodes=[_node(inputs=[])]
        )

    def test_rejects_output_arity_mismatch(self) -> None:
        self._assert_local_load_rejected(
            "op", ops=[_op_instance()], nodes=[_node(outputs=[])]
        )

    def test_rejects_names_bound_more_than_once(self) -> None:
        duplicate_inputs = _manifest()
        duplicate_inputs["inputs"] = [_io_entry("x"), _io_entry("x")]
        cases = [
            (duplicate_inputs, [], [], "x"),
            (_manifest(), [_op_instance()], [_node(outputs=["x"])], "x"),
            (
                _manifest(),
                [_op_instance(output_count=2)],
                [_node(outputs=["y", "y"])],
                "y",
            ),
        ]
        for manifest, ops, nodes, name in cases:
            with self.subTest(name=name, nodes=nodes):
                self._assert_local_load_rejected(name, manifest, ops, nodes)

    def test_rejects_input_not_bound_by_an_earlier_value(self) -> None:
        self._assert_local_load_rejected(
            "unbound", ops=[_op_instance()], nodes=[_node(inputs=["unbound"])]
        )

    def test_rejects_json_inputs_and_outputs(self) -> None:
        for entry_kind, entry_name in (("inputs", "x"), ("outputs", "y")):
            manifest = _manifest()
            manifest[entry_kind] = [_io_entry(entry_name, "JSON")]
            with self.subTest(entry_kind=entry_kind):
                self._assert_local_load_rejected(entry_name, manifest)

    def test_stdio_rejects_invalid_pipeline_before_environment_setup(self) -> None:
        manifest = _manifest()
        manifest["outputs"] = [_io_entry("y", "JSON")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = _write_pipeline_artifact(
                Path(temporary_directory) / "invalid.fnnx", manifest
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


class TestMissingDeclaredOutputs(unittest.TestCase):
    def test_local_sync_and_async_computation_reject_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = _write_pipeline_artifact(
                Path(temporary_directory) / "missing-output.fnnx"
            )
            handler = LocalHandler(
                str(artifact),
                DeviceMap(accelerator="cpu", node_device_map={}),
                LocalHandlerConfig(auto_cleanup=False),
            )
            try:
                with self.assertRaisesRegex(ValueError, "y"):
                    handler.compute({"x": np.asarray(1, dtype=np.float32)}, {})
                with self.assertRaisesRegex(ValueError, "y"):
                    asyncio.run(
                        handler.compute_async(
                            {"x": np.asarray(1, dtype=np.float32)}, {}
                        )
                    )
            finally:
                handler.executor.shutdown()
                handler.op_executor.shutdown()

    def test_stdio_sync_and_async_computation_reject_missing_output(self) -> None:
        handler = object.__new__(StdIOHandler)
        handler.input_specs = {}
        handler.output_specs = {
            "y": {"name": "y", "content_type": "JSON", "dtype": "ext::record"}
        }
        handler._client = cast(Any, _StaticClient({}))
        handler._executor = ThreadPoolExecutor(max_workers=1)
        try:
            with self.assertRaisesRegex(ValueError, "y"):
                handler.compute({}, {})
            with self.assertRaisesRegex(ValueError, "y"):
                asyncio.run(handler.compute_async({}, {}))
        finally:
            handler._client.close()
            handler._executor.shutdown()

    def test_stdio_preserves_dynamic_attribute_strings(self) -> None:
        handler = object.__new__(StdIOHandler)
        handler.input_specs = {}
        handler.output_specs = {}
        client = _StaticClient({})
        handler._client = cast(Any, client)
        handler._executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.assertEqual(handler.compute({}, {"n": ""}), {})
        finally:
            handler._client.close()
            handler._executor.shutdown()

        self.assertEqual(
            client.last_body,
            {"inputs": {}, "dynamic_attributes": {"n": ""}},
        )

    def test_stdio_worker_preserves_strings_and_rejects_other_values(self) -> None:
        wire_body = json.loads(json.dumps({"dynamic_attributes": {"n": ""}}))

        self.assertEqual(_dynamic_attributes_from_body(wire_body), {"n": ""})
        with self.assertRaisesRegex(TypeError, "strings"):
            _dynamic_attributes_from_body({"dynamic_attributes": {"n": 1}})


if __name__ == "__main__":
    unittest.main()
