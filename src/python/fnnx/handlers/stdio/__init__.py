from __future__ import annotations

import atexit
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import abspath
from os.path import join as pjoin
from shutil import rmtree
from typing import Any

from fnnx.device import DeviceMap
from fnnx.dtypes import (
    BUILTINS,
    DtypesManager,
    NDContainer,
    decode_nonfinite_float_strings,
    encode_nonfinite_floats,
)
from fnnx.envs.base import BaseEnvManager
from fnnx.envs.conda import CondaLikeEnvManager
from fnnx.handlers._base import BaseHandler, BaseHandlerConfig
from fnnx.handlers._common import unpack_model
from fnnx.handlers.stdio.client import StdIOClient
from fnnx.utils import to_thread
from fnnx.validators.model_schema import (
    validate_manifest,
    validate_op_instances,
    validate_variant,
)
from fnnx.variants.pipeline import validate_pipeline

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

WORKER_PATH = pjoin(abspath(os.path.dirname(__file__)), "worker.py")


@dataclass
class StdIOHandlerConfig(BaseHandlerConfig):
    request_timeout_s: float | None = None
    auto_cleanup: bool = True
    worker_num_threads: int = 1
    env_manager: type[BaseEnvManager] | str = CondaLikeEnvManager  # if str -> import


class StdIOHandler(BaseHandler):
    def __init__(
        self,
        model_path: str,
        device_map: DeviceMap,
        handler_config: StdIOHandlerConfig | None = None,
        **kwargs,
    ):
        self.handler_config = handler_config or StdIOHandlerConfig()

        if not isinstance(device_map, DeviceMap):
            raise ValueError("device_map must be an instance of DeviceMap")

        model_path, cleanup = unpack_model(model_path)
        self.model_path = abspath(model_path)
        self.cleanup = self.handler_config.auto_cleanup and cleanup
        if self.cleanup:
            atexit.register(lambda: _cleanup(self.model_path))

        with open(pjoin(self.model_path, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
            validate_manifest(self.manifest)
            self.input_specs = {i["name"]: i for i in self.manifest["inputs"]}
            self.output_specs = {o["name"]: o for o in self.manifest["outputs"]}

        with open(pjoin(self.model_path, "ops.json"), "r") as f:
            self.ops = json.load(f)
            validate_op_instances(self.ops)

        with open(pjoin(self.model_path, "variant_config.json"), "r") as f:
            self.variant_config = json.load(f)
            validate_variant(self.manifest["variant"], self.variant_config)
        if self.manifest["variant"] == "pipeline":
            validate_pipeline(self.manifest, self.ops, self.variant_config)

        with open(pjoin(self.model_path, "dtypes.json"), "r") as f:
            external_dtypes = json.load(f)
            self.dtypes_manager = DtypesManager(external_dtypes, BUILTINS)

        env_spec_path = pjoin(self.model_path, "env.json")
        raw_env_spec: dict = {}
        if os.path.exists(env_spec_path):
            with open(env_spec_path, "r") as f:
                raw_env_spec = json.load(f)
        conda_like_spec = raw_env_spec.get("python3::conda_pip", raw_env_spec or {})

        if isinstance(self.handler_config.env_manager, str):
            import importlib

            module_name, class_name = self.handler_config.env_manager.rsplit(".", 1)
            module = importlib.import_module(module_name)
            EnvMngrClass: type[BaseEnvManager] = getattr(module, class_name)
        else:
            EnvMngrClass = self.handler_config.env_manager

        if not issubclass(EnvMngrClass, BaseEnvManager):
            raise ValueError("env_manager must be a subclass of BaseEnvManager")

        mngr = EnvMngrClass(conda_like_spec, accelerator=device_map.accelerator)

        mngr.ensure()
        device_map_payload = {
            "accelerator": device_map.accelerator,
            "node_device_map": device_map.node_device_map,
            "variant_device_config": device_map.variant_device_config,
        }

        cmd = mngr.python_cmd(
            [
                WORKER_PATH,
                "--model",
                self.model_path,
                "--device-map",
                json.dumps(device_map_payload),
                "--worker-num-threads",
                str(self.handler_config.worker_num_threads),
            ]
        )

        self._client = StdIOClient(cmd)

        self._executor = ThreadPoolExecutor(max_workers=1)

    def _as_np(self, data: Any, spec: dict[str, Any]) -> Any:
        if np is None:
            raise RuntimeError("You must have numpy installed to use Array dtype")
        dtype = spec["dtype"][6:-1]
        if dtype == "string":
            return np.asarray(data).astype(np.str_)
        if dtype in {"float32", "float64"}:
            if isinstance(data, np.ndarray):
                shape = data.shape
                data = decode_nonfinite_float_strings(data.tolist())
                return np.asarray(data).reshape(shape).astype(dtype)
            data = decode_nonfinite_float_strings(data)
        return np.asarray(data).astype(dtype)

    def _inputs_to_wire(self, inputs: dict) -> dict:
        out = {}
        for name, val in inputs.items():
            spec = self.input_specs[name]
            if spec["content_type"] == "NDJSON":
                if "Array[" in spec["dtype"]:
                    if np is not None and isinstance(val, np.ndarray):
                        val = val.tolist()
                    out[name] = encode_nonfinite_floats(val)
                elif "NDContainer[" in spec["dtype"]:
                    if isinstance(val, NDContainer):
                        val = val.data
                    out[name] = encode_nonfinite_floats(val)
                else:
                    raise ValueError(f"Unknown dtype {spec['dtype']}")
            elif spec["content_type"] == "JSON":
                out[name] = val
            else:
                raise ValueError(f"Unknown input content_type {spec['content_type']}")
        return out

    def _outputs_from_wire(self, outputs: dict[str, Any]) -> dict[str, Any]:
        for name in self.output_specs:
            if name not in outputs:
                raise ValueError(f"Missing declared output `{name}`")

        prepared = {}
        for name, spec in self.output_specs.items():
            raw = outputs[name]
            if spec["content_type"] == "NDJSON":
                if "Array[" in spec["dtype"]:
                    prepared[name] = self._as_np(raw, spec)
                elif "NDContainer[" in spec["dtype"]:
                    if spec["dtype"] == "NDContainer[float]":
                        raw = decode_nonfinite_float_strings(raw)
                    prepared[name] = NDContainer(
                        raw, dtype=spec["dtype"], dtypes_manager=self.dtypes_manager
                    )
                else:
                    raise ValueError(f"Unknown dtype {spec['dtype']}")
            elif spec["content_type"] == "JSON":
                prepared[name] = raw
            else:
                raise ValueError(f"Unknown output content_type {spec['content_type']}")
        return prepared

    def compute(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        wire_inputs = self._inputs_to_wire(inputs)
        res = self._client.request(
            "compute",
            {"inputs": wire_inputs, "dynamic_attributes": dynamic_attributes},
            # timeout=self._request_timeout,
        )
        return self._outputs_from_wire(res)

    async def compute_async(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        def _call():
            wire_inputs = self._inputs_to_wire(inputs)
            res = self._client.request(
                "compute_async",
                {"inputs": wire_inputs, "dynamic_attributes": dynamic_attributes},
                # timeout=self._request_timeout,
            )
            return self._outputs_from_wire(res)

        return await to_thread(self._executor, _call)

    def __del__(self):
        try:
            self._client.close()
            self._executor.shutdown()
        except Exception:
            pass


def _cleanup(model_path: str):
    rmtree(model_path, ignore_errors=True)
