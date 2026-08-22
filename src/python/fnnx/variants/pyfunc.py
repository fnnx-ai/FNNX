from fnnx.variants._base import BaseVariant, OpInstance
from fnnx.variants._common.validators import validating_op_instance
from abc import ABC, abstractmethod
from contextlib import contextmanager
import sys
from os.path import join as pjoin, relpath
from os import walk
from threading import Lock
import importlib.util
import uuid
from typing import Any

_pyfunc_lock = Lock()


@contextmanager
def temp_sys_path(extra_modules_path: str, lock):
    with lock:
        sys.path.insert(0, extra_modules_path)
        original_modules = set(sys.modules.keys())
        try:
            yield
        finally:
            for name in set(sys.modules.keys()) - original_modules:
                mod = sys.modules.get(name)
                if mod and (getattr(mod, "__file__", "") or "").startswith(
                    extra_modules_path
                ):
                    sys.modules.pop(name, None)
            sys.path.pop(0)


class Context:
    def __init__(
        self,
        model_path: str,
        ops: dict[str, OpInstance],
        executor,
        accelerator,
        values: dict | None = None,
        device_config: dict | str | None = None,
    ):
        self._values = values or {}
        self._ops = ops
        self.executor = executor
        self.accelerator = accelerator
        self.device_config = device_config

        # make an index of files and their absolute paths
        self.files: dict[str, str] = {}
        self.dir: dict[str, str] = {}
        self._scan(model_path)

    @property
    def device(self):
        return self.accelerator

    def _scan(self, model_path):
        scan_path = pjoin(model_path, "variant_artifacts", "extra_files")
        for root, dirs, files in walk(scan_path):
            for file in files:
                full_path = pjoin(root, file)
                relative_path = relpath(full_path, scan_path)
                self.files[relative_path] = full_path
            for dir in dirs:
                full_path = pjoin(root, dir)
                relative_path = relpath(full_path, scan_path)
                self.dir[relative_path] = full_path

    def get_filepath(self, file: str):
        return self.files.get(file, None)

    def get_dirpath(self, dir: str):
        return self.dir.get(dir, None)

    def get_op_instance(self, node_id: str):
        return self._ops.get(node_id, None)

    def get_value(self, key: str):
        return self._values.get(key, None)


class PyFunc(ABC):
    def __init__(self, context: Context):
        self.fnnx_context = context

    @abstractmethod
    def warmup(self):
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


class PyFuncVariant(BaseVariant):
    def _post_init(self):
        self.context = Context(
            self.model_path,
            {
                op_id: validating_op_instance(instance)
                for op_id, instance in self.op_instances.items()
            },
            self.executor,
            self.device_map,
            values=self.variant_config.get("extra_values", None),
        )
        self.pyfunc_file_path = pjoin(
            self.model_path, "variant_artifacts", "__pyfunc__.py"
        )
        self.pyfunc_classname = self.variant_config["pyfunc_classname"]
        cls = self.get_pyfunc()
        self.pyfunc = cls(self.context)

    def warmup(self):
        super().warmup()
        with temp_sys_path(
            pjoin(self.model_path, "variant_artifacts", "extra_modules"), _pyfunc_lock
        ):
            self.pyfunc.warmup()
        return self

    def get_pyfunc(self) -> type[PyFunc]:
        unique_module_name = f"temp_module_{uuid.uuid4().hex}"
        with temp_sys_path(
            pjoin(self.model_path, "variant_artifacts", "extra_modules"), _pyfunc_lock
        ):
            spec = importlib.util.spec_from_file_location(
                unique_module_name, self.pyfunc_file_path
            )
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load {self.pyfunc_file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = getattr(module, self.pyfunc_classname)

        if not issubclass(cls, PyFunc):
            raise ValueError(f"Class {cls} is not a subclass of PyFunc")
        return cls

    def compute(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        return self.pyfunc.compute(inputs, dynamic_attributes)

    async def compute_async(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        return await self.pyfunc.compute_async(inputs, dynamic_attributes)
