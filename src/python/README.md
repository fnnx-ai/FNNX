# FNNX for Python

The Python package runs, inspects, and creates FNNX artifacts. The [FNNX specification](https://github.com/BeastByteAI/FNNX/tree/main/spec) defines the format and its execution semantics.

## Installation

FNNX requires Python 3.10 or later. Add the base package with `uv`.

```console
uv add fnnx
```

The package provides extras for features with additional dependencies.

| Extra | Support |
| --- | --- |
| `core` | NumPy arrays and `ONNX_v1` execution with [ONNX Runtime](https://onnxruntime.ai/docs/). |
| `extras` | `Reader`, `PyfuncBuilder`, and local MLflow model conversion. |
| `mlflow` | Remote MLflow URI resolution and conversion verification. |
| `compiler` | Compilation of ONNX models and FNNX pipeline artifacts to C99. |

Combine extras when one application needs several features.

```console
uv add "fnnx[core,extras]"
uv add "fnnx[extras,mlflow]"
uv add "fnnx[compiler]"
```

## Running an artifact

`Runtime` accepts an unpacked artifact directory or an uncompressed tar artifact. It uses `LocalHandler` unless you select another handler.

This example assumes `model.fnnx` declares `x` and `y` as `Array[float32]` values.

```python
import numpy as np

from fnnx.stable.v1 import LocalHandler, LocalHandlerConfig, Runtime

runtime = Runtime(
    "model.fnnx",
    handler=LocalHandler,
    handler_config=LocalHandlerConfig(n_workers=2, n_workers_node=2),
    device_map="cpu",
)
inputs = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
outputs = runtime.compute(inputs, {})
print(outputs["y"])
```

The input and output names must match the artifact manifest. Pass dynamic attributes as the second mapping to `compute`.

Every dynamic attribute value must be a string. Call `compute_async` from an async function when the caller must not block.

`n_workers` sets the artifact worker-thread count. `n_workers_node` sets the operation worker-thread count.

The `extra_ops` field maps operation names to custom operation classes.

## Inspecting an artifact

`Reader` reads a tar artifact without executing it. It exposes the effective manifest, ordered metadata, and raw environment document.

```python
from fnnx.extras.reader import Reader

reader = Reader("model.fnnx")
print(reader.manifest.model_dump())
print([entry.model_dump() for entry in reader.metadata])
print(reader.env)
```

`reader.pyenv` contains the parsed `python3::conda_pip` environment when the artifact declares one. It is `None` for other environment kinds.

## Running in an artifact environment

`StdIOHandler` starts a worker in the environment declared by the artifact. It supports the `python3::conda_pip` environment kind.

The default `CondaLikeEnvManager` searches for `micromamba`, `mamba`, or `conda`. Set `FNNX_CONDA_EXE` to select another executable path.

Use `UvEnvManager` to create the worker command with `uv`.

```python
from fnnx.envs.uv import UvEnvManager
from fnnx.handlers.stdio import StdIOHandler, StdIOHandlerConfig
from fnnx.stable.v1 import Runtime

runtime = Runtime(
    "model.fnnx",
    handler=StdIOHandler,
    handler_config=StdIOHandlerConfig(env_manager=UvEnvManager),
    device_map="cpu",
)
outputs = runtime.compute({"x": [[1.0, 2.0]]}, {})
print(outputs["y"])
```

`UvEnvManager` requires `uv` on `PATH`, or a path in `FNNX_UV_EXE`. It ignores declared build dependencies.

`LocalHandler` does not provision the artifact environment. Use `StdIOHandler` when the worker needs the dependencies from `env.json`.

## Creating a pyfunc artifact

A pyfunc artifact stores a `PyFunc` subclass. `PyfuncBuilder` reads that class from its Python source file and writes a tar artifact.

The following file builds and runs an echo artifact.

```python
from __future__ import annotations

from typing import Any

from fnnx.variants.pyfunc import PyFunc


class Echo(PyFunc):
    def warmup(self) -> None:
        pass

    def compute(
        self,
        inputs: dict[str, Any],
        dynamic_attributes: dict[str, str],
    ) -> dict[str, Any]:
        return {"echo": inputs["message"]}

    async def compute_async(
        self,
        inputs: dict[str, Any],
        dynamic_attributes: dict[str, str],
    ) -> dict[str, Any]:
        return self.compute(inputs, dynamic_attributes)


if __name__ == "__main__":
    from fnnx.extras.builder import PyfuncBuilder
    from fnnx.extras.pydantic_models.manifest import NDJSON
    from fnnx.stable.v1 import Runtime

    builder = PyfuncBuilder(Echo, model_name="echo", model_version="1")
    builder.add_input(
        NDJSON(
            name="message",
            content_type="NDJSON",
            dtype="NDContainer[string]",
            shape=["batch"],
        )
    )
    builder.add_output(
        NDJSON(
            name="echo",
            content_type="NDJSON",
            dtype="NDContainer[string]",
            shape=["batch"],
        )
    )
    builder.add_fnnx_runtime_dependency()
    builder.save("echo.fnnx")

    result = Runtime("echo.fnnx").compute({"message": ["hello"]}, {})
    print(result["echo"].data)
```

Use `add_runtime_dependency` for imports that the stored class needs. Use `add_file` or `add_module` to include local resources.

The [pyfunc variant specification](https://github.com/BeastByteAI/FNNX/blob/main/spec/variants/pyfunc.md) defines the stored entry point and its context.

## Converting an MLflow model

`package_mlflow_model` converts a local MLflow model directory or an MLflow URI to a pyfunc artifact. It derives inputs from the MLflow signature.

```python
from fnnx.extras.mlflow import package_mlflow_model

package_mlflow_model(
    "mlflow-model",
    "forecast.fnnx",
    name="forecast",
    verify=True,
)
```

Remote URIs and `verify=True` require the `mlflow` extra. Verification loads the artifact and uses its saved input example when one exists.

Use `input_specs` or `output_specs` when the inferred interface is not suitable. The converter stores the source MLflow model inside the artifact.

## Compiling ONNX to C

The compiler accepts a standalone [ONNX](https://onnx.ai/onnx/) model or an FNNX pipeline artifact. It writes one C99 header and one JSON report.

```console
uv run python -m fnnx.extras.compilers.c model.onnx \
    --output-dir build/model-c \
    --runtime-dim batch=64 \
    --prefix model
```

Use `--dim NAME=VALUE` to fix a symbolic dimension. Symbolic dimensions without a binding default to `1`.

Use `--runtime-dim NAME=MAX` to set a per-call dimension with a fixed maximum. The compiler rejects operations and types it cannot emit.
