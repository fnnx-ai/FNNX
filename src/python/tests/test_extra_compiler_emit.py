"""C emission: the single-header artifact, its metadata, and its determinism.

What is asserted here is structural — symbols, macros, buffers, byte-identity — and the
values that reach the header are whatever the ONNX reference evaluator folded, never
hand-written. Op semantics belong to the conformance and differential suites.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from fnnx.extras.compilers.c.errors import CompileError

if TYPE_CHECKING:
    import numpy

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
codegen = pytest.importorskip("fnnx.extras.compilers.c.onnx.codegen")
dtypes = pytest.importorskip("fnnx.extras.compilers.c.onnx.dtypes")
emit = pytest.importorskip("fnnx.extras.compilers.c.onnx.emit")
frontend = pytest.importorskip("fnnx.extras.compilers.c.onnx.frontend")
kernels = pytest.importorskip("fnnx.extras.compilers.c.onnx.kernels")
registry = pytest.importorskip("fnnx.extras.compilers.c.onnx.registry")

from fnnx import __version__  # noqa: E402
from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

OPSET = 21
STRICT_FLAGS = ("-std=c99", "-Wall", "-Wextra", "-Werror", "-Werror=vla")
C_COMPILERS = [name for name in ("gcc", "clang") if shutil.which(name)]
ALLOCATION_TOKENS = ("malloc", "calloc", "realloc", "free", "alloca")
SEED = 20260725


def _model(nodes, inputs, outputs, *, initializer=(), name="graph", opset=OPSET):
    graph = helper.make_graph(
        nodes, name, list(inputs), list(outputs), initializer=list(initializer)
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _tensor(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _constant_node(name, array):
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=onnx.numpy_helper.from_array(array, f"{name}_value"),
    )


def _alias_model(name="demo"):
    """A folded constant output, an aliased input, and an input nothing reads."""
    left = onnx.numpy_helper.from_array(np.array([1.5, 2.5], dtype=np.float32), "left")
    right = onnx.numpy_helper.from_array(
        np.array([1.0, 4.0], dtype=np.float32), "right"
    )
    return _model(
        [helper.make_node("Add", ["left", "right"], ["y"], name="add")],
        [
            _tensor("x", TensorProto.FLOAT, ["batch", 3]),
            _tensor("unread", TensorProto.FLOAT, [2]),
        ],
        [
            _tensor("y", TensorProto.FLOAT, [2]),
            _tensor("x", TensorProto.FLOAT, ["batch", 3]),
        ],
        initializer=[left, right],
        name=name,
    )


def _sample_values(elem_type: int) -> numpy.ndarray:
    """Special values plus seeded random ones, at `elem_type`."""
    dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
    generator = np.random.default_rng(SEED)
    if elem_type == TensorProto.BOOL:
        return np.array([True, False, True], dtype=dtype)
    if elem_type in (TensorProto.FLOAT, TensorProto.DOUBLE):
        info = np.finfo(dtype)
        special = [
            0.0,
            -0.0,
            np.nan,
            np.inf,
            -np.inf,
            info.max,
            -info.max,
            info.tiny,
            info.smallest_subnormal,
            info.eps,
        ]
        random = generator.uniform(-1e6, 1e6, size=8)
        return np.array([*special, *random], dtype=dtype)
    info = np.iinfo(dtype)
    random = generator.integers(info.min, info.max, size=8, endpoint=True, dtype=dtype)
    return np.array([info.min, info.max, 0, 1, *random], dtype=dtype)


def _parse_c_literal(text: str) -> float | int:
    """Read a literal `scalar_literal` produced back into Python."""
    expression = re.sub(r"\b(?:INT64_C|UINT64_C)\((-?\d+)\)", r"\1", text.strip())
    if expression == "NAN":
        return math.nan
    if expression in ("INFINITY", "-INFINITY"):
        return math.inf if expression == "INFINITY" else -math.inf
    compound = re.fullmatch(r"\((-\d+) - (\d+)\)", expression)
    if compound:
        return int(compound.group(1)) - int(compound.group(2))
    expression = re.sub(r"[fu]$", "", expression)
    if re.search(r"[.eE]", expression):
        return float(expression)
    return int(expression)


def _same_value(parsed: float | int, expected: float | int, dtype: numpy.dtype) -> bool:
    """Whether a literal, read back at its own precision, reproduces the source value."""
    if not np.issubdtype(dtype, np.floating):
        return int(parsed) == int(expected)
    read = dtype.type(parsed)
    if math.isnan(expected):
        return bool(np.isnan(read))
    if expected == 0.0:
        return read == 0.0 and math.copysign(1.0, float(read)) == math.copysign(
            1.0, expected
        )
    return bool(read == expected)


def _weight_arrays(header: str) -> dict[str, list[float | int]]:
    """Every embedded weight in the header, parsed back into Python values."""
    blocks = re.findall(
        r"static const \w+ (\w+)\[\d+\] = \{(.*?)\};", header, flags=re.DOTALL
    )
    return {
        symbol: [_parse_c_literal(piece) for piece in body.split(",") if piece.strip()]
        for symbol, body in blocks
    }


def _driver(report: dict) -> str:
    """A declarations-only unit that calls the entrypoint with correctly sized buffers."""
    entrypoint = report["entrypoint"]
    lines = [f'#include "{report["header"]}"', "", "int main(void)", "{"]
    arguments = []
    for index, tensor in enumerate([*entrypoint["inputs"], *entrypoint["outputs"]]):
        buffer = f"buffer_{index}"
        lines.append(
            f"    static {tensor['c_type']} {buffer}[{max(1, tensor['elem_count'])}];"
        )
        arguments.append(buffer)
    lines += [f"    return {entrypoint['symbol']}({', '.join(arguments)});", "}", ""]
    return "\n".join(lines)


def _build(compiler: str, directory: Path, sources: list[Path]) -> Path:
    binary = directory / f"{compiler}_artifact"
    result = subprocess.run(
        [
            compiler,
            *STRICT_FLAGS,
            f"-I{directory}",
            *[str(source) for source in sources],
            "-o",
            str(binary),
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _build_and_run(compiler: str, result) -> None:
    """Build the artifact from two units — one asking for the implementation — and run it."""
    directory = result.header_path.parent
    main = directory / "main.c"
    main.write_text(_driver(result.report), encoding="utf-8")
    unit = directory / "implementation.c"
    unit.write_text(
        f"#define {result.report['prefix'].upper()}_IMPLEMENTATION\n"
        f'#include "{result.report["header"]}"\n',
        encoding="utf-8",
    )
    binary = _build(compiler, directory, [main, unit])
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


needs_c_compiler = pytest.mark.skipif(
    not C_COMPILERS, reason="no system C compiler available"
)


def test_compile_writes_header_and_report(tmp_path):
    result = compile_onnx(_alias_model(), tmp_path)

    assert result.header_path == tmp_path / "demo.h"
    assert result.report_path == tmp_path / "demo_report.json"
    assert json.loads(result.report_path.read_text()) == result.report
    assert "int demo_run(" in result.header_path.read_text()


def test_declarations_and_implementation_are_separated(tmp_path):
    header = compile_onnx(_alias_model(), tmp_path).header_path.read_text()
    declarations, _, implementation = header.partition("#ifdef DEMO_IMPLEMENTATION")

    assert (
        "int demo_run(const float* x, const float* unread, float* y, float* x_2);"
        in declarations
    )
    assert "static const float demo_w_y[2]" not in declarations
    assert "static const float demo_w_y[2]" in implementation
    assert "#ifndef DEMO_H_INCLUDED" in declarations


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_header_builds_and_runs_under_strict_flags(tmp_path, compiler):
    _build_and_run(compiler, compile_onnx(_alias_model(), tmp_path))


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_header_is_include_guarded(tmp_path, compiler):
    compile_onnx(_alias_model(), tmp_path)
    source = tmp_path / "twice.c"
    source.write_text(
        "#define DEMO_IMPLEMENTATION\n"
        '#include "demo.h"\n#include "demo.h"\n'
        "int main(void) { return 0; }\n",
        encoding="utf-8",
    )
    _build(compiler, tmp_path, [source])


def _comment_hostile_model():
    """Names carrying both block-comment delimiters, which reach the header's comments."""
    return _model(
        [],
        [_tensor("in/*x", TensorProto.FLOAT, [2])],
        [
            _tensor("in/*x", TensorProto.FLOAT, [2]),
            _tensor("w*/z", TensorProto.FLOAT, [2]),
        ],
        initializer=[
            onnx.numpy_helper.from_array(np.ones(2, dtype=np.float32), "w*/z")
        ],
        name="hostile/*name*/",
    )


def test_names_never_nest_a_block_comment(tmp_path):
    header = compile_onnx(_comment_hostile_model(), tmp_path).header_path.read_text()

    for comment in re.findall(r"/\*.*?\*/", header, flags=re.DOTALL):
        assert "/*" not in comment[2:] and "*/" not in comment[:-2], comment


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_comment_hostile_names_build_under_strict_flags(tmp_path, compiler):
    _build_and_run(compiler, compile_onnx(_comment_hostile_model(), tmp_path))


def test_generated_source_has_no_allocation_calls(tmp_path):
    header = compile_onnx(_alias_model(), tmp_path).header_path.read_text()

    for token in ALLOCATION_TOKENS:
        assert not re.search(rf"\b{token}\b", header), token
    assert "#include" in header
    assert set(re.findall(r"#include <(\w+\.h)>", header)) <= {
        "stdint.h",
        "stddef.h",
        "string.h",
        "math.h",
    }


def test_repeated_compiles_are_byte_identical(tmp_path):
    first = compile_onnx(_alias_model(), tmp_path / "first", dim_bindings={"batch": 3})
    second = compile_onnx(
        _alias_model(), tmp_path / "second", dim_bindings={"batch": 3}
    )

    assert first.header_path.read_bytes() == second.header_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()


def test_report_records_options_bindings_opsets_and_footprint(tmp_path):
    result = compile_onnx(_alias_model(), tmp_path, dim_bindings={"batch": 4})
    report = result.report

    assert report["fnnx_version"] == __version__
    assert report["options"] == {
        "prefix": None,
        "dim_bindings": {"batch": 4},
        "runtime_dims": {},
    }
    assert report["dim_bindings"] == {"batch": 4}
    assert report["runtime_dims"] == []
    assert report["opsets"] == {"ai.onnx": OPSET}
    assert report["kernels"] == []
    assert report["memory"] == {
        "weights_bytes": 8,
        "arena_bytes": 0,
        "static_bytes": 8,
    }
    assert [tensor["name"] for tensor in report["entrypoint"]["inputs"]] == [
        "x",
        "unread",
    ]
    assert report["entrypoint"]["inputs"][0] == {
        "name": "x",
        "c_name": "x",
        "macro": "DEMO_INPUT_X",
        "dtype": "float32",
        "c_type": "float",
        "shape": [4, 3],
        "elem_count": 12,
        "bytes": 48,
    }


def test_unbound_dimensions_default_to_one(tmp_path):
    report = compile_onnx(_alias_model(), tmp_path).report

    assert report["dim_bindings"] == {"batch": 1}
    assert report["entrypoint"]["inputs"][0]["shape"] == [1, 3]
    assert "Dimension bindings: batch=1" in (tmp_path / "demo.h").read_text()


def test_the_preamble_documents_usage_reentrancy_bindings_and_footprint(tmp_path):
    result = compile_onnx(_alias_model(), tmp_path, dim_bindings={"batch": 4})
    preamble = result.header_path.read_text().split("*/", 1)[0]
    memory = result.report["memory"]

    assert f"#define {result.report['prefix'].upper()}_IMPLEMENTATION" in preamble
    assert f'#include "{result.report["header"]}"' in preamble
    assert f"{result.report['entrypoint']['symbol']}` runs the whole model" in preamble
    assert "DEMO_OK on success" in preamble
    assert " ".join(STRICT_FLAGS) in preamble
    assert "Not reentrant" in preamble
    assert (
        f"Static memory: {memory['static_bytes']} bytes "
        f"({memory['weights_bytes']} of weights, {memory['arena_bytes']} of scratch)"
        in preamble
    )
    assert f"Opset imports: ai.onnx={OPSET}" in preamble
    assert "Dimension bindings: batch=4" in preamble


def test_metadata_macros_describe_every_tensor(tmp_path):
    result = compile_onnx(_alias_model(), tmp_path, dim_bindings={"batch": 4})
    header = result.header_path.read_text()
    defined = dict(re.findall(r"#define (\w+) (\S+)", header))

    for tensor in (
        *result.report["entrypoint"]["inputs"],
        *result.report["entrypoint"]["outputs"],
    ):
        macro = tensor["macro"]
        assert defined[f"{macro}_RANK"] == str(len(tensor["shape"]))
        assert defined[f"{macro}_COUNT"] == str(tensor["elem_count"])
        for axis, size in enumerate(tensor["shape"]):
            assert defined[f"{macro}_DIM_{axis}"] == str(size)
    assert defined["DEMO_ARENA_BYTES"] == "0"
    assert defined["DEMO_WEIGHTS_BYTES"] == "8"
    assert defined["DEMO_STATIC_BYTES"] == "8"


def test_every_public_name_carries_the_prefix(tmp_path):
    header = compile_onnx(
        _alias_model(), tmp_path, prefix="my model.v2"
    ).header_path.read_text()

    assert "int my_model_v2_run(" in header
    for macro in re.findall(r"#define (\w+)", header):
        assert macro.startswith("MY_MODEL_V2_"), macro
    for line in header.splitlines():
        declaration = re.match(r"(?:static )?[A-Za-z_][\w ]*?[ *](\w+)\(", line)
        if declaration:
            assert declaration.group(1).startswith("my_model_v2_"), line


def test_prefix_falls_back_when_the_graph_name_is_unusable(tmp_path):
    result = compile_onnx(_alias_model(name="***"), tmp_path)

    assert result.report["prefix"] == codegen.DEFAULT_PREFIX
    assert result.header_path.name == f"{codegen.DEFAULT_PREFIX}.h"


def test_colliding_tensor_names_get_distinct_identifiers(tmp_path):
    model = _model(
        [],
        [
            _tensor("x.1", TensorProto.FLOAT, [2]),
            _tensor("x-1", TensorProto.FLOAT, [2]),
        ],
        [
            _tensor("x.1", TensorProto.FLOAT, [2]),
            _tensor("x-1", TensorProto.FLOAT, [2]),
        ],
    )
    report = compile_onnx(model, tmp_path).report
    names = [tensor["c_name"] for tensor in report["entrypoint"]["inputs"]]
    macros = [tensor["macro"] for tensor in report["entrypoint"]["outputs"]]

    assert names == ["x_1", "x_1_2"]
    assert len(set(macros)) == 2


def test_a_parameter_never_shadows_a_static_buffer(tmp_path):
    """An input named like a weight's symbol would hide it inside the entrypoint."""
    left = onnx.numpy_helper.from_array(np.array([1.5, 2.5], dtype=np.float32), "left")
    right = onnx.numpy_helper.from_array(
        np.array([1.0, 4.0], dtype=np.float32), "right"
    )
    model = _model(
        [helper.make_node("Add", ["left", "right"], ["y"], name="add")],
        [_tensor("demo_w_y", TensorProto.FLOAT, [2])],
        [
            _tensor("y", TensorProto.FLOAT, [2]),
            _tensor("demo_w_y", TensorProto.FLOAT, [2]),
        ],
        initializer=[left, right],
        name="demo",
    )
    header = compile_onnx(model, tmp_path).header_path.read_text()
    weights = re.findall(r"static const float (\w+)\[2\]", header)
    signature = re.findall(r"int demo_run\(([^)]*)\)", header)
    parameters = re.findall(r"\*\s*(\w+)", signature[0])

    assert len(weights) == 1
    assert weights[0] not in parameters
    assert f"memcpy(y, {weights[0]}, 2u * sizeof(*y));" in header


@pytest.mark.parametrize("elem_type", sorted(dtypes.C_TYPES))
def test_scalar_literals_round_trip(elem_type):
    values = _sample_values(elem_type)
    for value in values.tolist():
        literal = emit.scalar_literal(value, elem_type)
        parsed = _parse_c_literal(literal)
        expected = int(value) if isinstance(value, bool) else value
        assert _same_value(parsed, expected, values.dtype), literal


def _external_data_model(values, location: str):
    """A weight whose bytes live in a side file next to the model."""
    tensor = onnx.numpy_helper.from_array(values, "w")
    onnx.external_data_helper.set_external_data(tensor, location=location)
    tensor.ClearField("raw_data")
    return _model(
        [],
        [],
        [_tensor("w", TensorProto.FLOAT, list(values.shape))],
        initializer=[tensor],
        name="external",
    )


def test_external_data_weights_are_embedded_in_the_header(tmp_path):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    (tmp_path / "w.bin").write_bytes(values.tobytes())
    model_path = tmp_path / "model.onnx"
    onnx.save_model(_external_data_model(values, "w.bin"), str(model_path))

    result = compile_onnx(model_path, tmp_path / "out")
    emitted = _weight_arrays(result.header_path.read_text())

    assert emitted["external_w_w"] == values.reshape(-1).tolist()


def test_embedded_weights_preserve_every_supported_dtype(tmp_path):
    arrays = {
        dtypes.numpy_dtype_name(elem_type): _sample_values(elem_type)
        for elem_type in sorted(dtypes.C_TYPES)
    }
    model = _model(
        [_constant_node(name, array) for name, array in arrays.items()],
        [],
        [
            _tensor(
                name, onnx.helper.np_dtype_to_tensor_dtype(array.dtype), array.shape
            )
            for name, array in arrays.items()
        ],
        name="weights",
    )
    result = compile_onnx(model, tmp_path)
    emitted = _weight_arrays(result.header_path.read_text())

    for name, array in arrays.items():
        parsed = emitted[f"weights_w_{name}"]
        expected = [
            int(value) if isinstance(value, bool) else value for value in array.tolist()
        ]
        assert len(parsed) == len(expected)
        assert all(
            _same_value(read, source, array.dtype)
            for read, source in zip(parsed, expected)
        ), name


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_every_supported_dtype_builds(tmp_path, compiler):
    arrays = {
        dtypes.numpy_dtype_name(elem_type): _sample_values(elem_type)
        for elem_type in sorted(dtypes.C_TYPES)
    }
    model = _model(
        [_constant_node(name, array) for name, array in arrays.items()],
        [],
        [
            _tensor(
                name, onnx.helper.np_dtype_to_tensor_dtype(array.dtype), array.shape
            )
            for name, array in arrays.items()
        ],
        name="weights",
    )
    _build_and_run(compiler, compile_onnx(model, tmp_path))


def _zero_element_model():
    empty = np.zeros((0,), dtype=np.float32)
    return _model(
        [_constant_node("empty", empty)],
        [_tensor("x", TensorProto.FLOAT, [0, 3])],
        [
            _tensor("empty", TensorProto.FLOAT, [0]),
            _tensor("x", TensorProto.FLOAT, [0, 3]),
        ],
        name="zero",
    )


def test_zero_element_tensors_declare_no_empty_arrays(tmp_path):
    result = compile_onnx(_zero_element_model(), tmp_path)
    header = result.header_path.read_text()

    assert "[0]" not in header.split("#ifdef ZERO_IMPLEMENTATION")[1]
    assert "memcpy" not in header
    assert result.report["entrypoint"]["outputs"][0]["elem_count"] == 0
    assert result.report["memory"]["static_bytes"] == 0


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_zero_element_artifact_builds_and_runs(tmp_path, compiler):
    _build_and_run(compiler, compile_onnx(_zero_element_model(), tmp_path))


def _relu_generator(context):
    """A stand-in kernel: the emitter's contract, not an ONNX-conformant Relu."""
    source, target = context.inputs[0], context.outputs[0]
    c_type = dtypes.c_type(source.elem_type)
    name = f"{context.prefix}_relu_{c_type}"
    definition = "\n".join(
        [
            f"static void {name}({c_type}* out, const {c_type}* in, size_t count)",
            "{",
            "    size_t index;",
            "    for (index = 0; index < count; ++index) {",
            "        out[index] = in[index] > 0 ? in[index] : 0;",
            "    }",
            "}",
        ]
    )
    return kernels.NodeEmission(
        functions=(kernels.CFunction(name, definition),),
        statements=(f"{name}({target.expr}, {source.expr}, {target.elem_count}u);",),
    )


@pytest.fixture
def relu_registry(monkeypatch):
    stub = registry.KernelRegistry()
    stub.register("", "Relu", 14, _relu_generator)
    monkeypatch.setattr(codegen, "KERNELS", stub)
    return stub


def _chain_model():
    """Two chained Relus on float and one on double: a shared kernel and a private one."""
    return _model(
        [
            helper.make_node("Relu", ["x"], ["hidden"], name="first"),
            helper.make_node("Relu", ["hidden"], ["y"], name="second"),
            helper.make_node("Relu", ["xd"], ["yd"], name="third"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, [2, 3]),
            _tensor("xd", TensorProto.DOUBLE, [4]),
        ],
        [
            _tensor("y", TensorProto.FLOAT, [2, 3]),
            _tensor("yd", TensorProto.DOUBLE, [4]),
        ],
        name="chain",
    )


def test_kernels_are_shared_and_intermediates_are_static(tmp_path, relu_registry):
    result = compile_onnx(_chain_model(), tmp_path)
    header = result.header_path.read_text()

    assert result.report["kernels"] == ["chain_relu_float", "chain_relu_double"]
    assert header.count("static void chain_relu_float(") == 1
    assert header.count("chain_relu_float(") == 3
    assert "static float chain_t_hidden[6];" in header
    assert result.report["memory"] == {
        "weights_bytes": 0,
        "arena_bytes": 24,
        "static_bytes": 24,
    }
    calls = re.findall(r"chain_relu_\w+\([^;]+\);", header)
    assert calls == [
        "chain_relu_float(chain_t_hidden, x, 6u);",
        "chain_relu_float(y, chain_t_hidden, 6u);",
        "chain_relu_double(yd, xd, 4u);",
    ]


@needs_c_compiler
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_kernel_artifact_builds_and_runs(tmp_path, compiler, relu_registry):
    _build_and_run(compiler, compile_onnx(_chain_model(), tmp_path))


def test_fan_out_output_is_read_from_the_caller_buffer(tmp_path, relu_registry):
    """A node output that is both a graph output and a downstream input."""
    model = _model(
        [
            helper.make_node("Relu", ["x"], ["shared"], name="first"),
            helper.make_node("Relu", ["shared"], ["y"], name="second"),
        ],
        [_tensor("x", TensorProto.FLOAT, [2])],
        [
            _tensor("shared", TensorProto.FLOAT, [2]),
            _tensor("y", TensorProto.FLOAT, [2]),
        ],
        name="fanout",
    )
    header = compile_onnx(model, tmp_path).header_path.read_text()
    calls = re.findall(r"fanout_relu_\w+\([^;]+\);", header)

    assert calls == [
        "fanout_relu_float(shared, x, 2u);",
        "fanout_relu_float(y, shared, 2u);",
    ]
    assert "fanout_t_shared" not in header


def test_a_kernel_colliding_with_a_tensor_name_is_rejected(tmp_path, relu_registry):
    """Such a tensor would shadow the kernel inside the entrypoint, breaking the build."""
    model = _model(
        [helper.make_node("Relu", ["demo_relu_float"], ["y"], name="relu")],
        [_tensor("demo_relu_float", TensorProto.FLOAT, [2])],
        [_tensor("y", TensorProto.FLOAT, [2])],
        name="demo",
    )

    with pytest.raises(CompileError, match="`demo_relu_float` collides with"):
        compile_onnx(model, tmp_path)

    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_kernel_scratch_colliding_with_a_tensor_name_is_rejected(tmp_path):
    """A kernel's working buffer is named like the kernel, and shadowed the same way."""
    model = _model(
        [helper.make_node("Det", ["demo_det_float_work"], ["y"], name="det")],
        [_tensor("demo_det_float_work", TensorProto.FLOAT, [2, 2])],
        [_tensor("y", TensorProto.FLOAT, [])],
        name="demo",
        opset=22,
    )

    with pytest.raises(CompileError, match="`demo_det_float_work` collides with"):
        compile_onnx(model, tmp_path)

    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_kernels_sharing_a_name_must_share_a_definition(tmp_path, monkeypatch):
    def clashing(context):
        target = context.outputs[0]
        definition = f"static void clash(void) {{ /* {context.node.name} */ }}"
        return kernels.NodeEmission(
            functions=(kernels.CFunction("clash", definition),),
            statements=(f"(void){target.expr};",),
        )

    stub = registry.KernelRegistry()
    stub.register("", "Relu", 14, clashing)
    monkeypatch.setattr(codegen, "KERNELS", stub)
    model = _model(
        [
            helper.make_node("Relu", ["x"], ["hidden"], name="first"),
            helper.make_node("Relu", ["hidden"], ["y"], name="second"),
        ],
        [_tensor("x", TensorProto.FLOAT, [2])],
        [_tensor("y", TensorProto.FLOAT, [2])],
    )
    with pytest.raises(CompileError, match="emitted twice with different definitions"):
        compile_onnx(model, tmp_path)


def test_unsupported_op_writes_no_files(tmp_path, monkeypatch):
    """An empty registry stands in for any op no kernel covers, whatever is implemented."""
    monkeypatch.setattr(codegen, "KERNELS", registry.KernelRegistry())
    model = _model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        [_tensor("x", TensorProto.FLOAT, [2])],
        [_tensor("y", TensorProto.FLOAT, [2])],
    )
    output_dir = tmp_path / "out"
    with pytest.raises(CompileError) as error:
        compile_onnx(model, output_dir)

    message = str(error.value)
    assert "`relu`" in message and "`Relu`" in message
    assert "ai.onnx" in message and str(OPSET) in message
    assert not output_dir.exists()


def test_data_dependent_shape_op_writes_no_files(tmp_path):
    model = _model(
        [helper.make_node("NonZero", ["x"], ["y"], name="nonzero")],
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [_tensor("y", TensorProto.INT64, [2, "n"])],
    )
    output_dir = tmp_path / "out"
    with pytest.raises(CompileError) as error:
        compile_onnx(model, output_dir)

    message = str(error.value)
    assert "`nonzero`" in message and "`NonZero`" in message
    assert "depends on input data" in message
    assert not output_dir.exists()


def test_output_without_a_producer_is_rejected(tmp_path):
    model = _model(
        [],
        [_tensor("x", TensorProto.FLOAT, [2])],
        [_tensor("missing", TensorProto.FLOAT, [2])],
    )
    with pytest.raises(CompileError, match="`missing` is not produced"):
        compile_onnx(model, tmp_path)


def test_node_reading_an_undefined_tensor_is_rejected(relu_registry):
    """Nodes out of topological order, or reading a tensor nothing defines.

    Shape inference rejects most such graphs first, so codegen is driven directly here:
    the emitter must still refuse rather than reference an undeclared C symbol.
    """
    model = _model(
        [
            helper.make_node("Relu", ["hidden"], ["y"], name="second"),
            helper.make_node("Relu", ["x"], ["hidden"], name="first"),
        ],
        [_tensor("x", TensorProto.FLOAT, [2])],
        [_tensor("y", TensorProto.FLOAT, [2])],
    )
    prepared = frontend.PreparedModel(model=model, opsets={"": OPSET}, dim_bindings={})

    with pytest.raises(CompileError, match="reads tensor `hidden`"):
        codegen.build_program(prepared)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("plain", "plain"),
        ("with.dots-and spaces", "with_dots_and_spaces"),
        ("2fast", "v_2fast"),
        ("int", "int_"),
        ("", "fallback"),
        ("***", "fallback"),
    ],
)
def test_sanitize_identifier(name, expected):
    assert emit.sanitize_identifier(name, fallback="fallback") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("ends*/here", "ends* /here"),
        ("opens/*here", "opens/ *here"),
        ("/**/", "/ ** /"),
    ],
)
def test_comment_safe_neutralizes_both_delimiters(text, expected):
    assert emit.comment_safe(text) == expected


def test_unique_names_disambiguate_case_insensitively():
    names = emit.UniqueNames()
    assigned = [names.assign(name, fallback="v") for name in ("a", "A", "a.", "b")]

    assert assigned == ["a", "A_2", "a_", "b"]
