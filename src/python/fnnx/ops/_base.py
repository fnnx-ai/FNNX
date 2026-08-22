from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures._base import Executor
from dataclasses import dataclass
from typing import Any

from fnnx.device import DeviceConfig
from fnnx.dtypes import DtypesManager


class BaseOp(ABC):
    supported_dynamic_attributes: list[str] = []
    required_dynamic_attributes: list[str] = []

    def __init__(
        self,
        artifact_path: str,
        *args: Any,
        attributes: dict[str, Any],
        dynamic_attribute_map: dict[str, dict[str, str]],
        device_config: DeviceConfig,
        input_specs: list[dict[str, Any]],
        output_specs: list[dict[str, Any]],
        dtypes_manager: DtypesManager,
        executor: Executor,
        **kwargs: Any,
    ) -> None:
        self.dynamic_attribute_map = dynamic_attribute_map
        self._warmed_up = False
        self.artifact_path = artifact_path
        self._device_config: DeviceConfig = device_config
        self.attributes = attributes
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.dtypes_manager = dtypes_manager
        self.executor = executor

    @abstractmethod
    def warmup(self, *args: Any, **kwargs: Any) -> BaseOp:
        pass

    @abstractmethod
    def compute(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        pass

    @abstractmethod
    async def compute_async(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        pass

    def extract_dynamic_attribute(
        self, dynamic_attributes: dict[str, str]
    ) -> dict[str, str]:
        extracted: dict[str, str] = {}
        for key, value in self.dynamic_attribute_map.items():
            source_name = value["name"]
            if source_name in dynamic_attributes:
                extracted[key] = dynamic_attributes[source_name]
            else:
                extracted[key] = value["default_value"]
        return extracted

    def verify_required_dynamic_attributes(
        self, dynamic_attributes_map: dict[str, str]
    ) -> None:
        for key in self.required_dynamic_attributes:
            if key not in dynamic_attributes_map:
                raise ValueError(f"Missing required dynamic attribute: {key}")


@dataclass
class OpOutput:
    value: list[Any]
    metadata: dict[str, Any] | None = None
