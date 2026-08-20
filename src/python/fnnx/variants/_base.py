from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures._base import Executor
from os.path import abspath, join as pjoin
from typing import Any

from fnnx.device import DeviceConfig, DeviceMap
from fnnx.dtypes import DtypesManager
from fnnx.node_instance import OpInstance
from fnnx.registry import Registry


class BaseVariant(ABC):
    def __init__(
        self,
        model_path: str,
        op_instances: list[dict[str, Any]],
        variant_config: dict[str, Any],
        registry: Registry,
        device_map: DeviceMap,
        dtypes_manager: DtypesManager,
        executor: Executor,
        op_executor: Executor,
    ) -> None:
        self.model_path = abspath(model_path)
        self.registry = registry
        self.dtypes_manager = dtypes_manager
        self.op_instances: dict[str, OpInstance] = {}
        self.variant_config = variant_config
        self.executor = executor
        self.op_executor = op_executor
        self.device_map = device_map

        for op_instance in op_instances:
            op = self.registry.get_op(op_instance["op"])
            device = DeviceConfig(
                accelerator=device_map.accelerator,
                device_config=device_map.node_device_map.get(op_instance["id"], None),
            )
            artifact_path = pjoin(self.model_path, "ops_artifacts", op_instance["id"])
            operator = op(
                artifact_path,
                attributes=op_instance.get("attributes", {}),
                dynamic_attribute_map=op_instance.get("dynamic_attributes", {}),
                device_config=device,
                input_specs=op_instance["inputs"],
                output_specs=op_instance["outputs"],
                dtypes_manager=self.dtypes_manager,
                executor=self.op_executor,
            )

            self.op_instances[op_instance["id"]] = OpInstance(
                operator=operator,
                input_specs=op_instance["inputs"],
                output_specs=op_instance["outputs"],
            )
        self._post_init()

    @abstractmethod
    def _post_init(self) -> None:
        pass

    def warmup(self) -> BaseVariant:
        for instance in self.op_instances.values():
            instance.operator.warmup()
        return self

    @abstractmethod
    def compute(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    async def compute_async(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        pass
