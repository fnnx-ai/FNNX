from __future__ import annotations

from os.path import basename, normpath
from os.path import join as pjoin
from typing import Any

from fnnx.ops._base import BaseOp, OpOutput

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    ort = None

from fnnx.utils import to_thread

CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
SUPPORTED_OPSET_DOMAINS = frozenset({"ai.onnx", "ai.onnx.ml"})


class OnnxOp_V1(BaseOp):
    def warmup(
        self,
    ) -> OnnxOp_V1:
        self.model_path = pjoin(self.artifact_path, "model.onnx")
        op_instance_id = basename(normpath(self.artifact_path))
        for opset in self.attributes.get("opsets", []):
            domain = opset.get("domain")
            if domain not in SUPPORTED_OPSET_DOMAINS:
                raise RuntimeError(
                    f"ONNX op instance `{op_instance_id}` declares unsupported "
                    f"opset domain `{domain}`"
                )
        if not ort:
            raise ImportError("onnxruntime is not installed")
        if self._device_config.accelerator == "cuda":
            execution_providers = [CUDA_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER]
        else:
            execution_providers = [CPU_EXECUTION_PROVIDER]
        session_options = ort.SessionOptions()
        try:
            self._sess = ort.InferenceSession(
                self.model_path,
                providers=execution_providers,
                sess_options=session_options,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not create ONNX Runtime session for op instance "
                f"`{op_instance_id}`: {error}"
            ) from error
        self._ort_inputs = [i.name for i in self._sess.get_inputs()]
        self._ort_outputs = [o.name for o in self._sess.get_outputs()]
        self._verify_arity(op_instance_id)
        self._warmed_up = True
        return self

    def _verify_arity(self, op_instance_id: str) -> None:
        for role, declared, graph in (
            ("input", self.input_specs, self._ort_inputs),
            ("output", self.output_specs, self._ort_outputs),
        ):
            if len(declared) != len(graph):
                raise RuntimeError(
                    f"ONNX op instance `{op_instance_id}` declares {len(declared)} "
                    f"{role}(s), but its model has {len(graph)}"
                )

    def compute(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        if not self._warmed_up:
            raise RuntimeError("Op is not warmed up")
        outputs = self._sess.run(self._ort_outputs, dict(zip(self._ort_inputs, inputs)))
        return OpOutput(value=list(outputs), metadata={})

    async def compute_async(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        return await to_thread(self.executor, self.compute, inputs, dynamic_attributes)
