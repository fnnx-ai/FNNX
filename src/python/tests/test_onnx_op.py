import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from fnnx.device import DeviceConfig
from fnnx.ops import onnx as onnx_module
from fnnx.ops.onnx import OnnxOp_V1


class TestOnnxOpV1(unittest.TestCase):
    def _make_op(
        self,
        artifact_path: Path,
        opsets: list[dict[str, str | int]],
        input_specs: list[dict[str, Any]] | None = None,
        output_specs: list[dict[str, Any]] | None = None,
    ) -> OnnxOp_V1:
        op = OnnxOp_V1(
            str(artifact_path),
            attributes={"opsets": opsets},
            dynamic_attribute_map={},
            device_config=DeviceConfig(accelerator="cpu", device_config=None),
            input_specs=input_specs or [],
            output_specs=output_specs or [],
            dtypes_manager=Mock(),
            executor=ThreadPoolExecutor(max_workers=1),
        )
        self.addCleanup(op.executor.shutdown)
        return op

    def test_unsupported_domain_declines_before_session_creation(self) -> None:
        runtime = Mock()
        op = self._make_op(
            Path("artifact/ops_artifacts/custom_domain_op"),
            [{"domain": "example.custom", "version": 1}],
        )

        with (
            patch.object(onnx_module, "ort", runtime),
            self.assertRaisesRegex(
                RuntimeError,
                "example.custom.*custom_domain_op|custom_domain_op.*example.custom",
            ),
        ):
            op.warmup()

        runtime.SessionOptions.assert_not_called()
        runtime.InferenceSession.assert_not_called()

    def test_session_creation_error_names_the_op_instance(self) -> None:
        session_error = ValueError("invalid ONNX model")
        runtime = Mock()
        runtime.InferenceSession.side_effect = session_error
        op = self._make_op(
            Path("artifact/ops_artifacts/broken_op"),
            [{"domain": "ai.onnx", "version": 21}],
        )

        with (
            patch.object(onnx_module, "ort", runtime),
            self.assertRaisesRegex(RuntimeError, "broken_op") as raised,
        ):
            op.warmup()

        self.assertIs(raised.exception.__cause__, session_error)

    def _runtime_with_graph(self, input_count: int, output_count: int) -> Mock:
        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name=f"in{index}") for index in range(input_count)
        ]
        session.get_outputs.return_value = [
            SimpleNamespace(name=f"out{index}") for index in range(output_count)
        ]
        runtime = Mock()
        runtime.InferenceSession.return_value = session
        return runtime

    def test_declared_arity_disagreeing_with_the_model_is_rejected(self) -> None:
        tensor: dict[str, Any] = {"dtype": "Array[float32]", "shape": []}
        cases = (
            ("input", [tensor], [tensor], self._runtime_with_graph(2, 1)),
            ("output", [tensor], [tensor], self._runtime_with_graph(1, 3)),
        )
        for role, input_specs, output_specs, runtime in cases:
            with self.subTest(role=role):
                op = self._make_op(
                    Path("artifact/ops_artifacts/arity_op"),
                    [{"domain": "ai.onnx", "version": 12}],
                    input_specs=input_specs,
                    output_specs=output_specs,
                )
                with (
                    patch.object(onnx_module, "ort", runtime),
                    self.assertRaisesRegex(RuntimeError, f"arity_op.*{role}"),
                ):
                    op.warmup()

    def test_matching_arity_warms_up(self) -> None:
        tensor: dict[str, Any] = {"dtype": "Array[float32]", "shape": []}
        op = self._make_op(
            Path("artifact/ops_artifacts/arity_op"),
            [{"domain": "ai.onnx", "version": 12}],
            input_specs=[tensor, tensor],
            output_specs=[tensor],
        )

        with patch.object(onnx_module, "ort", self._runtime_with_graph(2, 1)):
            op.warmup()

        self.assertTrue(op._warmed_up)


if __name__ == "__main__":
    unittest.main()
