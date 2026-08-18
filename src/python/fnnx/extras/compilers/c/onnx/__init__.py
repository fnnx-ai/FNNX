"""FNNX-agnostic core of the C compiler: ONNX model in, C out."""

from importlib.util import find_spec

if find_spec("onnx") is None:
    # ModuleNotFoundError rather than a bare ImportError so that callers (and
    # `pytest.importorskip`) can tell a missing optional dependency from a broken one.
    raise ModuleNotFoundError(
        "The FNNX C compiler requires the `onnx` package. "
        'Install it with `pip install "fnnx[compiler]"`.',
        name="onnx",
    )
