try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore[assignment]
from os.path import join as pjoin
from shutil import rmtree
import atexit
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from fnnx.device import DeviceMap
from fnnx.dtypes import DtypesManager, BUILTINS, NDContainer
from fnnx.variants.pipeline import Pipeline, validate_pipeline
from fnnx.handlers._base import BaseHandler, BaseHandlerConfig
from fnnx.handlers._common import unpack_model
from fnnx.registry import Registry
from fnnx.variants.pyfunc import PyFuncVariant
from fnnx.validators.model_schema import (
    validate_manifest,
    validate_op_instances,
    validate_variant,
)
from fnnx.ops._base import BaseOp
from fnnx.variants._base import BaseVariant


_VARIANT_CLASSES = {"pipeline": Pipeline, "pyfunc": PyFuncVariant}


@dataclass
class LocalHandlerConfig(BaseHandlerConfig):
    n_workers: int = 1
    n_workers_node: int = 1
    auto_cleanup: bool = True
    extra_ops: dict[str, type[BaseOp]] | None = None


class LocalHandler(BaseHandler):
    def __init__(
        self,
        model_path: str,
        device_map: DeviceMap,
        handler_config: LocalHandlerConfig | None = None,
        **kwargs,
    ):
        if handler_config is None:
            handler_config = LocalHandlerConfig()

        if not isinstance(device_map, DeviceMap):
            raise ValueError("device_map must be an instance of DeviceMap")

        model_path, cleanup = unpack_model(model_path)

        self.model_path = model_path
        self.cleanup = handler_config.auto_cleanup and cleanup

        # should this be done on exit or on delete?
        if self.cleanup:
            # passing model_path and not self.model_path to avoid reference on self
            atexit.register(lambda: _cleanup(model_path))

        self.manifest = self._load_json_config("manifest.json")
        validate_manifest(self.manifest)
        self.input_specs = {spec["name"]: spec for spec in self.manifest["inputs"]}
        self.output_specs = {spec["name"]: spec for spec in self.manifest["outputs"]}

        self.ops = self._load_json_config("ops.json")
        validate_op_instances(self.ops)

        self.variant_config = self._load_json_config("variant_config.json")
        validate_variant(self.manifest["variant"], self.variant_config)
        if self.manifest["variant"] == "pipeline":
            validate_pipeline(self.manifest, self.ops, self.variant_config)

        external_dtypes = self._load_json_config("dtypes.json")
        self.dtypes_manager = DtypesManager(external_dtypes, BUILTINS)

        self.variant = self.manifest.get("variant")

        registry = Registry()
        registry.register_default_ops()
        if handler_config.extra_ops:
            for op_name, op in handler_config.extra_ops.items():
                registry.register_op(op, op_name)

        vcls = _VARIANT_CLASSES.get(self.variant)
        if vcls is None:
            raise ValueError(f"Unknown variant: {self.variant}")

        self.executor = ThreadPoolExecutor(max_workers=handler_config.n_workers)
        self.op_executor = ThreadPoolExecutor(max_workers=handler_config.n_workers_node)
        self.vrt: BaseVariant = vcls(
            self.model_path,
            self.ops,
            self.variant_config,
            registry=registry,
            device_map=device_map,
            dtypes_manager=self.dtypes_manager,
            executor=self.executor,
            op_executor=self.op_executor,
        ).warmup()

    def _load_json_config(self, filename: str):
        """Load and return JSON config from model path."""
        with open(pjoin(self.model_path, filename), "r") as f:
            return json.load(f)

    def _as_np(self, data, spec):
        if np is None:
            raise RuntimeError("You must have numpy installed to use Array dtype")
        dtype = spec["dtype"][6:-1]
        if dtype == "string":
            return np.asarray(data).astype(np.str_)
        return np.asarray(data).astype(dtype)

    def _prepare_ndjson_input(self, input, spec):
        if "NDContainer[" in spec["dtype"]:
            if not isinstance(input, NDContainer):
                return NDContainer(
                    input,
                    dtype=spec["dtype"],
                    dtypes_manager=self.dtypes_manager,
                )
            return input
        elif "Array[" in spec["dtype"]:
            return self._as_np(input, spec)
        else:
            raise ValueError(f"Unknown dtype {spec['dtype']}")

    def _prepare_json_input(self, input, spec):
        if self.variant == "pipeline":
            raise ValueError("Pipeline variant does not support JSON inputs")
        dtype = spec["dtype"]
        self.dtypes_manager.validate_jsonschema(dtype, input)
        return input

    def _prepare_inputs(self, inputs):
        prepared_inputs = {}
        for name, input in inputs.items():
            spec = self.input_specs[name]
            if spec["content_type"] == "NDJSON":
                prepared_inputs[name] = self._prepare_ndjson_input(input, spec)
            elif spec["content_type"] == "JSON":
                prepared_inputs[name] = self._prepare_json_input(input, spec)
            else:
                raise ValueError(f"Unknown input type {spec['content_type']}")
        return prepared_inputs

    def _prepare_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        for name in self.output_specs:
            if name not in outputs:
                raise ValueError(f"Missing declared output `{name}`")
        return {name: outputs[name] for name in self.output_specs}

    def compute(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        res = self.vrt.compute(
            self._prepare_inputs(inputs),
            dynamic_attributes=dynamic_attributes,
        )
        return self._prepare_outputs(res)

    async def compute_async(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        res = await self.vrt.compute_async(
            self._prepare_inputs(inputs),
            dynamic_attributes=dynamic_attributes,
        )
        return self._prepare_outputs(res)

    def __del__(self):
        try:
            self.executor.shutdown()
            self.op_executor.shutdown()
        except Exception:
            pass


def _cleanup(model_path):
    # print("Cleaning up temporary model files at", model_path)
    rmtree(model_path)
