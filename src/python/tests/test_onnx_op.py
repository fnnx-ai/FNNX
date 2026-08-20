import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from fnnx.device import DeviceConfig
from fnnx.ops import onnx as onnx_module
from fnnx.ops.onnx import OnnxOp_V1


class TestOnnxOpV1(unittest.TestCase):
    def _make_op(
        self, artifact_path: Path, opsets: list[dict[str, str | int]]
    ) -> OnnxOp_V1:
        op = OnnxOp_V1(
            str(artifact_path),
            attributes={"opsets": opsets},
            dynamic_attribute_map={},
            device_config=DeviceConfig(accelerator="cpu", device_config=None),
            input_specs=[],
            output_specs=[],
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


if __name__ == "__main__":
    unittest.main()
