"""Building a compiled artifact into a shared library and driving it from Python.

This is tooling around the artifact, not part of it: nothing here influences the generated
C. It needs numpy and a system C compiler, and nothing else — in particular not `onnx`,
so an artifact can be exercised wherever it was copied to.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fnnx.extras.compilers.c.errors import HarnessError

if find_spec("numpy") is None:
    # ModuleNotFoundError rather than a bare ImportError so that callers (and
    # `pytest.importorskip`) can tell a missing optional dependency from a broken one.
    raise ModuleNotFoundError(
        "The FNNX C load-and-run harness requires numpy. "
        'Install it with `pip install "fnnx[core]"`.',
        name="numpy",
    )

import numpy  # noqa: E402

# The artifact's build contract: what the generated header must compile cleanly under.
STRICT_FLAGS = ("-std=c99", "-Wall", "-Wextra", "-Werror", "-Werror=vla")
SHARED_FLAGS = ("-fPIC", "-shared")

COMPILER_CANDIDATES = ("cc", "gcc", "clang")

_REQUIRED_REPORT_FIELDS = ("prefix", "header", "entrypoint")


def load_compiled(
    path: str | os.PathLike[str], *, compiler: str | None = None
) -> CompiledModel:
    """Build a compiled artifact into a shared library and bind its entrypoints.

    `path` is the emitted header or its compile report, which sits beside it; the report is
    what drives the binding. `compiler` overrides the detected system C compiler.
    """
    report_path = _resolve_report_path(Path(path))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HarnessError(
            f"Could not read the compile report `{report_path}`: {error}"
        ) from error
    missing = [field for field in _REQUIRED_REPORT_FIELDS if field not in report]
    if missing:
        raise HarnessError(
            f"The compile report `{report_path}` is missing the "
            f"{', '.join(f'`{field}`' for field in missing)} field(s); it does not "
            "describe an artifact of this compiler."
        )

    build_dir = Path(tempfile.mkdtemp(prefix="fnnx_c_harness_"))
    try:
        library_path = _build_shared_library(
            report_path.parent, report, build_dir, compiler=compiler
        )
        model = CompiledModel(report, library_path)
    except BaseException:
        _discard(build_dir)
        raise
    # The library stays mapped once loaded, so the build directory only has to outlive the
    # load itself; tying it to the model keeps the artifact directory free of build output.
    weakref.finalize(model, _discard, build_dir)
    return model


@dataclass(frozen=True)
class TensorSpec:
    """One tensor of an entrypoint's signature, as the compile report describes it.

    `shape` is the buffer's capacity. Where the artifact has runtime dimensions, `axes`
    says how each extent follows from them — `(None, size)` for a fixed axis, and
    `(dimension, factor)` for one that scales — so the shape a particular call works at is
    `shape_at`.
    """

    name: str
    dtype: numpy.dtype
    shape: tuple[int, ...]
    axes: tuple[tuple[str | None, int], ...] = ()

    def shape_at(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        if not self.axes:
            return self.shape
        return tuple(
            size if dim is None else size * dims[dim] for dim, size in self.axes
        )


@dataclass(frozen=True)
class RuntimeDimSpec:
    """A dimension the caller sizes per call, and the maximum it was compiled for."""

    name: str
    maximum: int


class CompiledModel:
    """A compiled artifact, built into a shared library and bound through ctypes.

    Not reentrant, following the artifact's own contract: one in-flight call per model.
    """

    def __init__(self, report: Mapping[str, Any], library_path: Path) -> None:
        self.report = dict(report)
        self.library_path = library_path
        try:
            self._library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise HarnessError(
                f"Could not load the built shared library `{library_path}`: {error}"
            ) from error
        self._dims = tuple(
            RuntimeDimSpec(str(dim["name"]), int(dim["max"]))
            for dim in report.get("runtime_dims", ())
        )
        self._entry = self._bind(report["entrypoint"], "The compiled model")
        self._nodes = {
            str(node["id"]): self._bind(node, f"Node `{node['id']}`")
            for node in report.get("nodes", ())
        }

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return self._entry.inputs

    @property
    def outputs(self) -> tuple[TensorSpec, ...]:
        return self._entry.outputs

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    @property
    def runtime_dims(self) -> tuple[RuntimeDimSpec, ...]:
        return self._dims

    def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        dims: Mapping[str, int] | None = None,
        **named: Any,
    ) -> dict[str, numpy.ndarray]:
        """Run the whole graph on named numpy arrays, returning its named outputs.

        Inputs may be passed as a mapping — tensor names need not be Python identifiers —
        or as keyword arguments, and are validated against the compiled shapes and dtypes
        before the C entrypoint is called. `dims` gives the size of each runtime dimension
        for this call; one left out is read off the inputs that scale with it.
        """
        return self._entry.run(_merge_inputs(inputs, named), dims or {})

    def run_node(
        self,
        node_id: str,
        inputs: Mapping[str, Any] | None = None,
        *,
        dims: Mapping[str, int] | None = None,
        **named: Any,
    ) -> dict[str, numpy.ndarray]:
        """Run a single node's entrypoint, by the node id the compile report lists."""
        entry = self._nodes.get(str(node_id))
        if entry is None:
            available = ", ".join(f"`{name}`" for name in self._nodes) or "none"
            raise HarnessError(
                f"The compiled artifact exposes no entrypoint for node `{node_id}`; "
                f"it exposes: {available}."
            )
        return entry.run(_merge_inputs(inputs, named), dims or {})

    def _bind(self, description: Mapping[str, Any], label: str) -> _Entrypoint:
        symbol = description["symbol"]
        try:
            function = getattr(self._library, symbol)
        except AttributeError:
            raise HarnessError(
                f"The shared library built from `{self.report['header']}` exports no "
                f"symbol `{symbol}`, which the compile report names as an entrypoint."
            ) from None
        inputs = _tensor_specs(description["inputs"])
        outputs = _tensor_specs(description["outputs"])
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_int32] * len(self._dims) + [ctypes.c_void_p] * (
            len(inputs) + len(outputs)
        )
        return _Entrypoint(label, symbol, inputs, outputs, function, self._dims)


@dataclass
class _Entrypoint:
    """A bound C entrypoint: the tensors it takes and the callable behind its symbol."""

    label: str
    symbol: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    call: Callable[..., int]
    dims: tuple[RuntimeDimSpec, ...] = ()

    def run(
        self, values: Mapping[str, Any], dims: Mapping[str, int]
    ) -> dict[str, numpy.ndarray]:
        self._check_names(values)
        sizes = self._resolve_dims(values, dims)
        arguments = tuple(
            self._checked(spec, values[spec.name], sizes) for spec in self.inputs
        )
        results = {
            spec.name: numpy.empty(spec.shape_at(sizes), dtype=spec.dtype)
            for spec in self.outputs
        }
        buffers = (*arguments, *results.values())
        status = self.call(
            *[sizes[dim.name] for dim in self.dims],
            *[buffer.ctypes.data for buffer in buffers],
        )
        if status != 0:
            raise HarnessError(
                f"{self.label}: `{self.symbol}` returned status {status}."
            )
        return results

    def _check_names(self, values: Mapping[str, Any]) -> None:
        expected = {spec.name for spec in self.inputs}
        missing = sorted(expected - set(values))
        unexpected = sorted(set(values) - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {_quoted(missing)}")
            if unexpected:
                details.append(f"unexpected {_quoted(unexpected)}")
            raise HarnessError(
                f"{self.label}: {' and '.join(details)}; it takes "
                f"{_quoted(spec.name for spec in self.inputs) or 'no inputs'}."
            )

    def _resolve_dims(
        self, values: Mapping[str, Any], given: Mapping[str, int]
    ) -> dict[str, int]:
        """The size of every runtime dimension for this call, stated or read off the inputs."""
        unknown = sorted(set(given) - {dim.name for dim in self.dims})
        if unknown:
            named = _quoted(dim.name for dim in self.dims) or "none"
            raise HarnessError(
                f"{self.label}: {_quoted(unknown)} is not a runtime dimension of this "
                f"artifact; it has {named}."
            )
        sizes = {}
        for dim in self.dims:
            size = given.get(dim.name)
            if size is None:
                size = self._infer_dim(dim, values)
            if not isinstance(size, (int, numpy.integer)) or isinstance(size, bool):
                raise HarnessError(
                    f"{self.label}: runtime dimension `{dim.name}` needs an integer "
                    f"size, got {size!r}."
                )
            if not 1 <= size <= dim.maximum:
                raise HarnessError(
                    f"{self.label}: runtime dimension `{dim.name}` is {size}, outside "
                    f"the [1, {dim.maximum}] the artifact was compiled for."
                )
            sizes[dim.name] = int(size)
        return sizes

    def _infer_dim(self, dim: RuntimeDimSpec, values: Mapping[str, Any]) -> int:
        for spec in self.inputs:
            for axis, (name, factor) in enumerate(spec.axes):
                if name != dim.name:
                    continue
                extent = numpy.shape(values[spec.name])
                if axis >= len(extent):
                    continue
                size, remainder = divmod(extent[axis], factor)
                if remainder:
                    raise HarnessError(
                        f"{self.label}: input `{spec.name}` is {extent[axis]} long on "
                        f"axis {axis}, which is not {factor} times a size of runtime "
                        f"dimension `{dim.name}`."
                    )
                return size
        raise HarnessError(
            f"{self.label}: no input's shape depends on runtime dimension "
            f"`{dim.name}`, so its size has to be passed as `dims={{'{dim.name}': ...}}`."
        )

    def _checked(
        self, spec: TensorSpec, value: Any, dims: Mapping[str, int]
    ) -> numpy.ndarray:
        array = numpy.asarray(value)
        if array.dtype != spec.dtype:
            raise HarnessError(
                f"{self.label}: input `{spec.name}` has dtype `{array.dtype}`, but the "
                f"artifact was compiled for `{spec.dtype}`."
            )
        expected = spec.shape_at(dims)
        if array.shape != expected:
            raise HarnessError(
                f"{self.label}: input `{spec.name}` has shape {array.shape}, but the "
                f"artifact was compiled for {expected}."
            )
        return numpy.ascontiguousarray(array)


def _tensor_specs(
    descriptions: Sequence[Mapping[str, Any]],
) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=description["name"],
            dtype=numpy.dtype(description["dtype"]),
            shape=tuple(description["shape"]),
            axes=_axes(description.get("runtime_shape")),
        )
        for description in descriptions
    )


def _axes(
    runtime_shape: Sequence[Any] | None,
) -> tuple[tuple[str | None, int], ...]:
    if not runtime_shape:
        return ()
    return tuple(
        (None, int(axis))
        if isinstance(axis, int)
        else (str(axis["dim"]), int(axis["coefficient"]))
        for axis in runtime_shape
    )


def _merge_inputs(
    mapping: Mapping[str, Any] | None, named: dict[str, Any]
) -> dict[str, Any]:
    values = dict(mapping) if mapping is not None else {}
    duplicates = sorted(set(values) & set(named))
    if duplicates:
        raise HarnessError(
            f"Input(s) {_quoted(duplicates)} were given both in the mapping and as "
            "keyword arguments."
        )
    values.update(named)
    return values


def _resolve_report_path(path: Path) -> Path:
    if path.suffix == ".json":
        report = path
    elif path.suffix == ".h":
        report = path.with_name(f"{path.stem}_report.json")
    else:
        raise HarnessError(
            f"`{path}` is neither a generated header (`.h`) nor a compile report "
            "(`.json`); pass one of the two files a compilation emitted."
        )
    if not report.is_file():
        raise HarnessError(f"Compile report not found: `{report}`.")
    return report


def _build_shared_library(
    artifact_dir: Path,
    report: Mapping[str, Any],
    build_dir: Path,
    *,
    compiler: str | None,
) -> Path:
    header = artifact_dir / report["header"]
    if not header.is_file():
        raise HarnessError(
            f"The header `{header}` the compile report names is missing."
        )
    unit = build_dir / "implementation.c"
    unit.write_text(
        f"#define {report['prefix'].upper()}_IMPLEMENTATION\n"
        f'#include "{header.name}"\n',
        encoding="utf-8",
    )
    library = build_dir / f"{report['prefix']}.so"
    command = [
        _find_compiler(compiler),
        *STRICT_FLAGS,
        *SHARED_FLAGS,
        f"-I{artifact_dir}",
        str(unit),
        "-o",
        str(library),
        "-lm",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        raise HarnessError(
            f"Could not run the C compiler `{command[0]}`: {error}"
        ) from error
    if process.returncode != 0:
        raise HarnessError(
            f"Building `{header.name}` as a shared library failed "
            f"(`{' '.join(command)}`):\n{process.stderr.strip()}"
        )
    return library


def _find_compiler(requested: str | None) -> str:
    if requested is not None:
        if shutil.which(requested) is None:
            raise HarnessError(f"The requested C compiler `{requested}` was not found.")
        return requested
    candidates = [os.environ.get("CC"), *COMPILER_CANDIDATES]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise HarnessError(
        "No system C compiler was found; the load-and-run harness needs one of "
        f"{', '.join(COMPILER_CANDIDATES)} on PATH, the `CC` environment variable set, "
        "or the `compiler` argument."
    )


def _discard(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)
