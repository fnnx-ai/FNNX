import tarfile
import io
from typing import Type, Callable
from fnnx.variants.pyfunc import PyFunc
import json
from dataclasses import dataclass
from dataclasses import asdict as dataclass_asdict
import inspect
import sys

from pydantic import BaseModel as PydanticBaseModel

from fnnx import __version__ as fnnx_version

from fnnx.extras.pydantic_models.manifest import Manifest, NDJSON, JSON, Var
from fnnx.extras.pydantic_models.variants.pyfunc import PyFuncVariant
from fnnx.extras.pydantic_models.envs import Python3_CondaPip, PipDependency


def asdict(obj):
    if isinstance(obj, PydanticBaseModel):
        return obj.model_dump()
    return dataclass_asdict(obj)


def asjson(obj):
    if isinstance(obj, PydanticBaseModel):
        return obj.model_dump_json(indent=4)
    return json.dumps(asdict(obj), indent=4)


PYTHON_VERSION = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)


@dataclass
class PyFuncSpec:
    filepath: str
    class_name: str


class PyfuncBuilder:
    def __init__(
        self,
        pyfunc: Type[PyFunc] | PyFuncSpec,
        model_name: str | None = None,
        model_version: str | None = None,
        model_description: str | None = None,
        create_meta_callback: Callable | None = None,
        python_version: str | None = None,
    ) -> None:
        self._inputs: list[NDJSON | JSON] = []
        self._outputs: list[NDJSON | JSON] = []
        self._dynamic_attributes: list[Var] = []
        self._env_vars: list[Var] = []

        self._producer_name: str = "fnnx.ai"
        self._producer_version: str = fnnx_version
        self._producer_tags: list[str] = []

        self._extra_dtypes: dict = {}

        self._name = model_name
        self._version = model_version
        self._description = model_description

        self._extra_modules: list[str] = []
        self._extra_files: list[tuple[str, str]] = []
        self._extra_values: dict | None = None

        self._build_dependencies: list[str] = []
        self._rt_dependencies: list[str] = []

        self._python_version = python_version

        self.create_meta_callback = create_meta_callback

        if isinstance(pyfunc, PyFuncSpec):
            self.pyfunc_name = pyfunc.class_name
            pyfunc_file = pyfunc.filepath
        elif issubclass(pyfunc, PyFunc):
            self.pyfunc_name = pyfunc.__name__
            pyfunc_file = inspect.getfile(pyfunc)
        else:
            raise TypeError(
                "Pyfunc must be a subclass of PyFunc or an instance of PyFuncSpec"
            )

        with open(pyfunc_file) as f:
            self.pyfunc_content = f.read()

    def add_input(self, input_spec: NDJSON | JSON) -> None:
        if not isinstance(input_spec, (NDJSON, JSON)):
            raise TypeError("input_spec must be NDJSON or JSON instance")
        if input_spec.name in [x.name for x in self._inputs]:
            raise ValueError(f"input with name {input_spec.name} already exists")
        if (
            input_spec.dtype.startswith("ext::")
            and input_spec.dtype not in self._extra_dtypes
        ):
            raise ValueError(f"extra dtype with name {input_spec.dtype} not defined")
        self._inputs.append(input_spec)

    def add_output(self, output_spec: NDJSON | JSON) -> None:
        if not isinstance(output_spec, (NDJSON, JSON)):
            raise TypeError("output_spec must be NDJSON or JSON instance")
        self._outputs.append(output_spec)

    def add_dynamic_attribute(self, dynamic_attribute: Var) -> None:
        if not isinstance(dynamic_attribute, Var):
            raise TypeError("dynamic_attribute must be Var instance")
        if dynamic_attribute.name in [x.name for x in self._dynamic_attributes]:
            raise ValueError(
                f"dynamic_attribute with name {dynamic_attribute.name} already exists"
            )
        self._dynamic_attributes.append(dynamic_attribute)

    def add_env_var(self, env_var: Var) -> None:
        if not isinstance(env_var, Var):
            raise TypeError("env_var must be Var instance")
        if env_var.name in [x.name for x in self._env_vars]:
            raise ValueError(f"env_var with name {env_var.name} already exists")
        self._env_vars.append(env_var)

    def set_extra_values(self, values: dict) -> None:
        self._extra_values = values.copy()

    def define_dtype(
        self, name: str, dtype: Type[PydanticBaseModel] | dict
    ) -> None:
        if not name.startswith("ext::"):
            raise ValueError("dtype name must start with 'ext::'")
        if isinstance(dtype, dict):
            self._extra_dtypes[name] = dtype
        else:
            self._extra_dtypes[name] = dtype.model_json_schema()

    def set_producer_info(
        self, name: str, version: str, tags: list[str] | None = None
    ) -> None:
        self._producer_name = name
        self._producer_version = version
        self._producer_tags = tags or []

    def add_module(self, module_path: str) -> None:
        module_name = module_path.split("/")[-1]
        if module_name in [x.split("/")[-1] for x in self._extra_modules]:
            raise ValueError(f"module with name {module_name} already exists")
        self._extra_modules.append(module_path)

    def add_file(self, file_path: str, target_path: str) -> None:
        if target_path in [x[1] for x in self._extra_files]:
            raise ValueError(f"file with name {target_path} already exists")
        self._extra_files.append((file_path, target_path))

    def save(self, path: str) -> None:
        manifest = Manifest(
            variant="pyfunc",
            producer_name=self._producer_name,
            producer_version=self._producer_version,
            producer_tags=self._producer_tags,
            inputs=self._inputs,
            outputs=self._outputs,
            dynamic_attributes=self._dynamic_attributes,
            env_vars=self._env_vars,
            name=self._name,
            version=self._version,
            description=self._description,
        )

        f = File(path)
        f.create_file("manifest.json", asjson(manifest))
        f.create_file("dtypes.json", json.dumps(self._extra_dtypes))
        f.create_file("env.json", json.dumps(self._make_env()))
        f.create_file("variant_artifacts/__pyfunc__.py", self.pyfunc_content)
        f.create_file(
            "variant_config.json",
            asjson(
                PyFuncVariant(
                    pyfunc_classname=self.pyfunc_name, extra_values=self._extra_values
                )
            ),
        )
        f.create_file("ops.json", "[]")
        f.make_artifacts_folders()
        if self.create_meta_callback:
            self.create_meta_callback(f)
        else:
            f.create_file("meta.json", "[]")

        for module in self._extra_modules:
            f.copy(
                module,
                f"variant_artifacts/extra_modules/{module.split('/')[-1]}",
                should_exclude_pycache=True,
            )

        for file_path, target_path in self._extra_files:
            f.copy(file_path, f"variant_artifacts/extra_files/{target_path}")

        try:
            pass
        except Exception as e:
            raise e
        finally:
            f.close()

    def add_default_build_dependencies(self) -> None:
        import subprocess
        import re

        def get_version(
            command: str, idx: int = 0, regex: str = r"^\d+(\.\d+)*([a-zA-Z]+\d*)?$"
        ) -> str:
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                output = result.stdout.strip()

                version_parts = output.split()
                version = version_parts[idx]
                if not version or not re.match(regex, version):
                    raise ValueError(f"Invalid version format: ```{version}```")
                return version
            except (subprocess.CalledProcessError, IndexError, ValueError) as e:
                print(f"Error retrieving version for '{command}': {e}")
                return "unknown"

        try:
            pip_version = get_version("pip --version", 1)
            self.add_build_dependency(f"pip=={pip_version}")
        except Exception as e:
            print(f"Error adding pip version to build dependencies: {e}")

        try:
            setuptools_version = get_version(
                "pip show setuptools | grep Version | awk '{print $2}'"
            )
            self.add_build_dependency(f"setuptools=={setuptools_version}")
        except Exception as e:
            print(f"Error adding setuptools version to build dependencies: {e}")

        try:
            wheel_version = get_version(
                "pip show wheel | grep Version | awk '{print $2}'"
            )
            self.add_build_dependency(f"wheel=={wheel_version}")
        except Exception as e:
            print(f"Error adding wheel version to build dependencies: {e}")

    def add_build_dependency(self, dep: str) -> None:
        self._build_dependencies.append(dep)

    def add_runtime_dependency(self, dep: str) -> None:
        # TODO add conditions for runtime dependencies
        self._rt_dependencies.append(dep)

    def add_fnnx_runtime_dependency(self, core: bool = False) -> None:
        import fnnx

        fnnx_version = fnnx.__version__
        dependency_name = "fnnx[core]" if core else "fnnx"
        self.add_runtime_dependency(f"{dependency_name}=={fnnx_version}")

    def _make_env(self):
        python_version = self._python_version or PYTHON_VERSION
        return {
            "python3::conda_pip": asdict(
                Python3_CondaPip(
                    python_version=python_version,
                    build_dependencies=self._build_dependencies,
                    dependencies=[PipDependency(package=p) for p in self._rt_dependencies],
                )
            )
        }


class File:
    def __init__(self, path):
        self.tar = tarfile.open(path, "w")

    def make_artifacts_folders(self):
        self.create_file("meta_artifacts/.keep", ".keep")
        self.create_file("ops_artifacts/.keep", ".keep")
        self.create_file("variant_artifacts/extra_modules/.keep", ".keep")
        self.create_file("variant_artifacts/extra_files/.keep", ".keep")

    def create_file(self, path, content: str):
        data = io.BytesIO(content.encode())
        info = tarfile.TarInfo(path)
        info.size = len(data.getvalue())
        self.tar.addfile(info, data)

    def copy(self, src, dst, should_exclude_pycache: bool = False):
        def exclude_pycache(tarinfo):
            if "__pycache__" in tarinfo.name and should_exclude_pycache:
                return None
            return tarinfo

        self.tar.add(src, dst, filter=exclude_pycache)

    def close(self):
        self.tar.close()
