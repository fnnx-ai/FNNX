"""Static runtime wrapper for FNNX packages produced by ``fnnx.extras.mlflow``.

Every converted MLflow model uses this exact class as its pyfunc entry point;
per-model behavior is driven by ``extra_values["fnnx_mlflow"]`` plus the
manifest. This module imports only ``fnnx``; ``mlflow`` is imported lazily in
``warmup`` and ``numpy``/``pandas`` are probed at call time so this file stays
importable in environments that lack them.
"""

from contextlib import contextmanager
from dataclasses import asdict as dataclass_asdict, is_dataclass
import os
import sys
from typing import Any

from fnnx.variants.pyfunc import PyFunc


_SINGLE_TENSOR_SENTINEL = "__single__"

# mlflow reads this at model-load time to choose the prediction device for
# device-aware flavors (e.g. pytorch). We bridge FNNX's device_map onto it.
_MLFLOW_DEVICE_ENV = "MLFLOW_DEFAULT_PREDICTION_DEVICE"


def _fnnx_device_to_mlflow(accelerator) -> str | None:
    """Map an FNNX ``device_map`` accelerator to an mlflow prediction device.

    Accepts the ``DeviceMap`` carried by ``Context.device`` or a bare string.
    Returns ``None`` when there is no actionable device hint (``"auto"`` or an
    unknown accelerator), leaving mlflow's own auto-detection in charge.
    """
    if hasattr(accelerator, "accelerator"):  # a DeviceMap flowed through
        accelerator = accelerator.accelerator
    if not isinstance(accelerator, str):
        return None
    a = accelerator.strip().lower()
    if a == "cpu":
        return "cpu"
    if a in ("cuda", "gpu"):
        return "cuda"
    if a.startswith("cuda:"):
        return accelerator  # preserve an explicit index, e.g. "cuda:1"
    return None  # auto / mps / unknown -> let mlflow decide


@contextmanager
def _prediction_device(device: str | None):
    """Scope ``MLFLOW_DEFAULT_PREDICTION_DEVICE`` to a model load.

    A device already present in the environment wins (an explicit runtime
    override), and the previous value is always restored so the process
    environment is never mutated permanently.
    """
    if not device or os.environ.get(_MLFLOW_DEVICE_ENV):
        yield
        return
    os.environ[_MLFLOW_DEVICE_ENV] = device
    try:
        yield
    finally:
        os.environ.pop(_MLFLOW_DEVICE_ENV, None)


def _to_jsonable(x):
    """Normalize predict() output into JSON-safe data.

    Handles numpy scalars/ndarrays, pandas DataFrame/Series, pydantic models,
    dataclasses, dicts, lists/tuples; falls back to ``str(x)`` for anything
    else. numpy/pandas/pydantic are detected via ``sys.modules`` so this helper
    works without them installed.
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()

    pd = sys.modules.get("pandas")
    if pd is not None:
        if isinstance(x, pd.DataFrame):
            return x.to_dict(orient="records")
        if isinstance(x, pd.Series):
            return x.tolist()

    pydantic = sys.modules.get("pydantic")
    if pydantic is not None:
        base_model = getattr(pydantic, "BaseModel", None)
        if base_model is not None and isinstance(x, base_model):
            return x.model_dump()

    if is_dataclass(x) and not isinstance(x, type):
        return dataclass_asdict(x)

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    return str(x)


class MLflowModel(PyFunc):
    """Single static pyfunc wrapper for any converted MLflow model."""

    def warmup(self):
        import mlflow.pyfunc  # type: ignore[import-not-found]

        cfg = self.fnnx_context.get_value("fnnx_mlflow") or {}
        self._cfg = cfg
        model_dir = self.fnnx_context.get_dirpath(cfg["model_dir"])
        if model_dir is None:
            raise FileNotFoundError(
                f"MLflow model directory '{cfg['model_dir']}' not found in package"
            )
        # Bridge the FNNX runtime's device_map onto mlflow's load-time device
        # selection (honored by device-aware flavors such as pytorch). The
        # default device_map accelerator is "cpu", so a converted model runs on
        # CPU unless the runtime is given device_map="cuda" (or an explicit
        # MLFLOW_DEFAULT_PREDICTION_DEVICE env var, which still wins).
        device = _fnnx_device_to_mlflow(getattr(self.fnnx_context, "device", None))
        with _prediction_device(device):
            self.model = mlflow.pyfunc.load_model(model_dir)

    def _build_payload(self, inputs: dict):
        cfg = self._cfg
        mode = cfg.get("input_mode", "passthrough")

        if mode == "columns":
            import pandas as pd  # type: ignore[import-untyped]

            column_order = cfg.get("column_order") or list(inputs.keys())
            return pd.DataFrame({c: inputs[c] for c in column_order})

        if mode == "tensor":
            tensor_names = cfg.get("tensor_names") or []
            if tensor_names == [_SINGLE_TENSOR_SENTINEL]:
                return inputs["input"]
            return {n: inputs[n] for n in tensor_names}

        return inputs["data"]

    def compute(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        payload = self._build_payload(inputs)

        param_names = self._cfg.get("param_names") or []
        params = {
            n: dynamic_attributes[n] for n in param_names if n in dynamic_attributes
        }

        if params:
            result = self.model.predict(payload, params=params)
        else:
            result = self.model.predict(payload)

        return {"predictions": _to_jsonable(result)}

    async def compute_async(
        self, inputs: dict[str, Any], dynamic_attributes: dict[str, str]
    ) -> dict[str, Any]:
        return self.compute(inputs, dynamic_attributes)
