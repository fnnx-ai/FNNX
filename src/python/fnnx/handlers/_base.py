from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fnnx.device import DeviceMap


@dataclass
class BaseHandlerConfig:
    pass


class BaseHandler(ABC):
    @abstractmethod
    def __init__(
        self,
        model_path: str,
        device_map: DeviceMap,
        handler_config: BaseHandlerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        pass

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
