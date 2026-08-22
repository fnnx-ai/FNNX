"""The FNNX bundle layer: reading a pipeline bundle, node entrypoints, and pipeline glue.

The oracle for a compiled pipeline is the FNNX `Runtime` — onnxruntime executing the same
bundle — and, for a single node, the ONNX reference evaluator. Nothing here states an
expected output of its own.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")
bundle_module = pytest.importorskip("fnnx.extras.compilers.c.bundle")
codegen = pytest.importorskip("fnnx.extras.compilers.c.onnx.codegen")
kernels = pytest.importorskip("fnnx.extras.compilers.c.onnx.kernels")

from fnnx.extras.compilers.c import compile_bundle  # noqa: E402
from fnnx.runtime import Runtime  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

OPSET = 21
SEED = 20260726

MODELS = Path(__file__).parent / "models"
PIPELINE_BUNDLE = MODELS / "onnx_pipeline.fnnx"
PIPELINE_TAR = MODELS / "onnx_pipeline.fnnx.tar"

# Compiling is only half of what these tests assert; the other half runs the artifact.
pytestmark = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


# --------------------------------------------------------------------------------------
# Bundle fixtures
# --------------------------------------------------------------------------------------


def _values(shape, *, seed: int = SEED):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _spec(shape, dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": f"Array[{dtype}]", "shape": list(shape)}


def _manifest_tensor(name: str, shape=(), dtype: str = "float32") -> dict[str, Any]:
    return {
        "name": name,
        "content_type": "NDJSON",
        "dtype": f"Array[{dtype}]",
        "shape": list(shape),
    }


@dataclass
class _Node:
    """One pipeline node of a bundle written for a test."""

    id: str
    model: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_specs: tuple[dict[str, Any], ...]
    output_specs: tuple[dict[str, Any], ...]
    dynamic_attributes: dict[str, Any] = field(default_factory=dict)
    extra_dynattrs: dict[str, str] = field(default_factory=dict)
    op: str = "ONNX_v1"
    opset: int = OPSET


def _affine_model(weight: np.ndarray, bias: float, *, name: str):
    """`y = x * weight + bias` over a symbolic batch, with weights of its own."""
    width = int(weight.size)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Mul", ["x", "w"], ["scaled"], name="mul"),
                helper.make_node("Add", ["scaled", "b"], ["y"], name="add"),
            ],
            name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", width])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", width])],
            initializer=[
                onnx.numpy_helper.from_array(weight.astype(np.float32), "w"),
                onnx.numpy_helper.from_array(np.array([bias], dtype=np.float32), "b"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", OPSET)],
    )


def _difference_model(name: str):
    """`y = a - b` over a symbolic batch: the join of a diamond.

    Subtraction rather than addition so that the two operands are not interchangeable: a
    join wired to its inputs in the wrong order has to show up in the result.
    """
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Sub", ["a", "b"], ["y"], name="sub")],
            name,
            [
                helper.make_tensor_value_info("a", TensorProto.FLOAT, ["batch", 3]),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, ["batch", 3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
        ),
        opset_imports=[helper.make_opsetid("", OPSET)],
    )


def _write_bundle(
    directory: Path,
    nodes,
    inputs,
    outputs,
    *,
    variant: str = "pipeline",
    manifest_dynamic_attributes=(),
) -> Path:
    """Write a bundle the runtime can load and the compiler can read."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": variant,
        "producer_name": "tests",
        "producer_version": "0.0.0",
        "producer_tags": [],
        "inputs": list(inputs),
        "outputs": list(outputs),
        "dynamic_attributes": list(manifest_dynamic_attributes),
        "env_vars": [],
    }
    ops = [
        {
            "id": node.id,
            "op": node.op,
            "inputs": list(node.input_specs),
            "outputs": list(node.output_specs),
            "attributes": {
                "opsets": [{"domain": "ai.onnx", "version": node.opset}],
                "has_external_data": False,
                "onnx_ir_version": 10,
            },
            "dynamic_attributes": dict(node.dynamic_attributes),
        }
        for node in nodes
    ]
    variant_config = {
        "nodes": [
            {
                "op_instance_id": node.id,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "extra_dynattrs": dict(node.extra_dynattrs),
            }
            for node in nodes
        ]
    }
    for name, document in (
        ("manifest.json", manifest),
        ("ops.json", ops),
        ("variant_config.json", variant_config),
        ("dtypes.json", {}),
    ):
        (directory / name).write_text(json.dumps(document, indent=2), encoding="utf-8")
    for node in nodes:
        artifacts = directory / "ops_artifacts" / node.id
        artifacts.mkdir(parents=True, exist_ok=True)
        if node.model is not None:
            onnx.save_model(node.model, str(artifacts / "model.onnx"))
    return directory


def _diamond_nodes():
    """`head` fans out to `left` and `right`, which `join` sums; `head` is also an output.

    Every node carries weights of its own, so a fan-out edge routed to the wrong buffer —
    or a weight shared between nodes that should not share one — changes the result.
    """
    tensor = _spec(["batch", 3])
    return [
        _Node(
            id="head",
            model=_affine_model(np.array([1.0, 2.0, 3.0]), 0.5, name="head"),
            inputs=("x",),
            outputs=("h",),
            input_specs=(tensor,),
            output_specs=(tensor,),
        ),
        _Node(
            id="left",
            model=_affine_model(np.array([-1.0, 0.25, 4.0]), -2.0, name="left"),
            inputs=("h",),
            outputs=("l",),
            input_specs=(tensor,),
            output_specs=(tensor,),
        ),
        _Node(
            id="right",
            model=_affine_model(np.array([7.0, -0.5, 0.125]), 3.0, name="right"),
            inputs=("h",),
            outputs=("r",),
            input_specs=(tensor,),
            output_specs=(tensor,),
        ),
        _Node(
            id="join",
            model=_difference_model("join"),
            inputs=("l", "r"),
            outputs=("out",),
            input_specs=(tensor, tensor),
            output_specs=(tensor,),
        ),
    ]


@pytest.fixture
def diamond_bundle(tmp_path):
    return _write_bundle(
        tmp_path / "diamond.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("h"), _manifest_tensor("out")],
    )


def _runtime_outputs(bundle: Path, feeds: dict[str, Any]) -> dict[str, Any]:
    return Runtime(str(bundle)).compute(dict(feeds), {})


def _assert_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert sorted(actual) == sorted(expected)
    for name, want in expected.items():
        got = actual[name]
        assert got.dtype == want.dtype, name
        assert got.shape == want.shape, name
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)


# --------------------------------------------------------------------------------------
# Compiling the pipeline test bundle
# --------------------------------------------------------------------------------------


def test_the_pipeline_bundle_compiles_to_one_header_and_report(tmp_path):
    result = compile_bundle(PIPELINE_BUNDLE, tmp_path)
    header = result.header_path.read_text()

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fnnx_model.h",
        "fnnx_model_report.json",
    ]
    assert "int fnnx_model_run(" in header
    for node_id in ("linreg", "linreg2", "linreg3", "concat_reduce"):
        assert f"int fnnx_model_node_{node_id}_run(" in header
    # The preamble points a reader at the per-node entrypoints the pipeline exposes.
    assert "fnnx_model_node_<id>_run" in header.split("*/", 1)[0]
    assert result.report["dim_bindings"] == {"batch": 1}
    assert [node["id"] for node in result.report["nodes"]] == [
        "linreg",
        "linreg2",
        "linreg3",
        "concat_reduce",
    ]
    # Builds the header under `-std=c99 -Wall -Wextra -Werror -Werror=vla`.
    result.load()


def test_the_compiled_pipeline_matches_the_python_runtime(tmp_path):
    compiled = compile_bundle(PIPELINE_BUNDLE, tmp_path).load()
    feeds = {"x": _values((1, 3))}

    _assert_matches(compiled.run(feeds), _runtime_outputs(PIPELINE_BUNDLE, feeds))


def test_a_dimension_binding_sizes_every_buffer_of_the_pipeline(tmp_path):
    result = compile_bundle(PIPELINE_BUNDLE, tmp_path, dim_bindings={"batch": 4})
    header = result.header_path.read_text()
    feeds = {"x": _values((4, 3))}

    outputs = result.load().run(feeds)

    assert result.report["dim_bindings"] == {"batch": 4}
    assert result.report["options"]["dim_bindings"] == {"batch": 4}
    assert result.report["entrypoint"]["inputs"][0]["shape"] == [4, 3]
    assert "#define FNNX_MODEL_INPUT_X_DIM_0 4" in header
    assert "#define FNNX_MODEL_OUTPUT_Y4_DIM_0 4" in header
    assert "#define FNNX_MODEL_NODE_LINREG_INPUT_FLOAT_INPUT_DIM_0 4" in header
    _assert_matches(outputs, _runtime_outputs(PIPELINE_BUNDLE, feeds))


def test_a_tar_packaged_bundle_compiles_like_its_directory(tmp_path):
    from_directory = compile_bundle(PIPELINE_BUNDLE, tmp_path / "directory")
    from_tar = compile_bundle(PIPELINE_TAR, tmp_path / "tar")
    feeds = {"x": _values((1, 3))}

    outputs = from_tar.load().run(feeds)

    # The two artifacts differ only in the file the preamble names as their source.
    for key in ("prefix", "entrypoint", "nodes", "memory", "dim_bindings"):
        assert from_tar.report[key] == from_directory.report[key], key
    _assert_matches(outputs, _runtime_outputs(PIPELINE_BUNDLE, feeds))


def test_repeated_runs_carry_no_state_between_them(tmp_path):
    compiled = compile_bundle(PIPELINE_BUNDLE, tmp_path).load()
    first = {"x": _values((1, 3), seed=1)}
    second = {"x": _values((1, 3), seed=2)}

    first_outputs = compiled.run(first)
    second_outputs = compiled.run(second)
    repeated = compiled.run(first)

    _assert_matches(first_outputs, _runtime_outputs(PIPELINE_BUNDLE, first))
    _assert_matches(second_outputs, _runtime_outputs(PIPELINE_BUNDLE, second))
    _assert_matches(repeated, _runtime_outputs(PIPELINE_BUNDLE, first))


def test_a_node_entrypoint_computes_what_the_node_computes(tmp_path):
    """The `linreg` entrypoint alone, against the reference evaluator on its own model."""
    compiled = compile_bundle(PIPELINE_BUNDLE, tmp_path).load()
    model = onnx.load(str(PIPELINE_BUNDLE / "ops_artifacts" / "linreg" / "model.onnx"))
    feeds = {"float_input": _values((1, 3))}

    outputs = compiled.run_node("linreg", feeds)

    expected = ReferenceEvaluator(model).run(None, feeds)
    _assert_matches(outputs, {"variable": expected[0]})


def test_compiling_a_bundle_twice_is_byte_identical(tmp_path):
    first = compile_bundle(PIPELINE_BUNDLE, tmp_path / "first")
    second = compile_bundle(PIPELINE_BUNDLE, tmp_path / "second")

    assert first.header_path.read_bytes() == second.header_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()


def test_the_pipeline_glue_allocates_nothing(tmp_path):
    header = compile_bundle(PIPELINE_BUNDLE, tmp_path).header_path.read_text()

    for token in ("malloc", "calloc", "realloc", "free", "alloca"):
        assert not re.search(rf"\b{token}\b", header), token


def test_a_kernel_two_nodes_share_is_emitted_once(tmp_path):
    """`linreg`, `linreg2` and `linreg3` are the same op at the same types."""
    result = compile_bundle(PIPELINE_BUNDLE, tmp_path)
    header = result.header_path.read_text()

    kernels = result.report["kernels"]

    assert len(kernels) == len(set(kernels))
    scorers = [name for name in kernels if "ml_scores" in name]
    assert len(scorers) == 1
    assert header.count(f"static void {scorers[0]}(") == 1


def test_the_prefix_defaults_to_the_manifest_name(tmp_path, diamond_bundle):
    """`onnx_pipeline.fnnx` carries no name, so only a bundle that does shows this."""
    manifest = json.loads((diamond_bundle / "manifest.json").read_text())
    manifest["name"] = "my diamond"
    (diamond_bundle / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = compile_bundle(diamond_bundle, tmp_path / "out")

    assert result.report["prefix"] == "my_diamond"
    assert result.header_path.name == "my_diamond.h"
    assert "int my_diamond_run(" in result.header_path.read_text()
    assert result.report["nodes"][0]["symbol"] == "my_diamond_node_head_run"
    result.load()


def test_an_explicit_prefix_renames_every_entrypoint(tmp_path):
    result = compile_bundle(PIPELINE_BUNDLE, tmp_path, prefix="my model")
    header = result.header_path.read_text()

    assert result.report["prefix"] == "my_model"
    assert "int my_model_run(" in header
    assert "int my_model_node_linreg_run(" in header
    result.load()


# --------------------------------------------------------------------------------------
# Pipeline wiring
# --------------------------------------------------------------------------------------


def test_a_fan_out_edge_that_is_also_an_output_reaches_every_consumer(
    tmp_path, diamond_bundle
):
    compiled = compile_bundle(diamond_bundle, tmp_path / "out").load()
    feeds = {"x": _values((1, 3))}

    outputs = compiled.run(feeds)

    _assert_matches(outputs, _runtime_outputs(diamond_bundle, feeds))


def test_a_node_entrypoint_matches_what_the_pipeline_computes(tmp_path, diamond_bundle):
    """`head`'s output is a manifest output too, so both routes are observable."""
    compiled = compile_bundle(diamond_bundle, tmp_path / "out").load()
    feeds = {"x": _values((1, 3))}

    inside = compiled.run(feeds)["h"]
    alone = compiled.run_node("head", {"x": feeds["x"]})["y"]

    np.testing.assert_array_equal(alone, inside)


def test_nodes_are_ordered_by_their_edges_not_by_their_declaration(
    tmp_path, diamond_bundle
):
    """Declared back to front, the nodes still compile — and run — in dependency order."""
    nodes = _diamond_nodes()
    shuffled = _write_bundle(
        tmp_path / "shuffled.fnnx",
        [nodes[3], nodes[2], nodes[1], nodes[0]],
        [_manifest_tensor("x")],
        [_manifest_tensor("h"), _manifest_tensor("out")],
    )
    feeds = {"x": _values((1, 3))}

    result = compile_bundle(shuffled, tmp_path / "out")

    assert [node["id"] for node in result.report["nodes"]] == [
        "head",
        "right",
        "left",
        "join",
    ]
    _assert_matches(result.load().run(feeds), _runtime_outputs(diamond_bundle, feeds))


def test_a_manifest_shape_that_agrees_with_the_nodes_is_accepted(tmp_path):
    bundle = _write_bundle(
        tmp_path / "declared.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x", ["batch", 3])],
        [_manifest_tensor("h", ["batch", 3]), _manifest_tensor("out", ["batch", 3])],
    )
    feeds = {"x": _values((2, 3))}

    result = compile_bundle(bundle, tmp_path / "out", dim_bindings={"batch": 2})

    _assert_matches(result.load().run(feeds), _runtime_outputs(bundle, feeds))


def test_one_op_instance_may_run_at_two_places_in_the_pipeline(tmp_path):
    """Two pipeline nodes on one op instance share its entrypoint, called twice."""
    tensor = _spec(["batch", 3])
    scale = _Node(
        id="scale",
        model=_affine_model(np.array([2.0, 3.0, 4.0]), 1.0, name="scale"),
        inputs=("x",),
        outputs=("once",),
        input_specs=(tensor,),
        output_specs=(tensor,),
    )
    again = _Node(
        id="scale",
        model=None,
        inputs=("once",),
        outputs=("twice",),
        input_specs=(tensor,),
        output_specs=(tensor,),
    )
    bundle = _write_bundle(
        tmp_path / "twice.fnnx",
        [scale, again],
        [_manifest_tensor("x")],
        [_manifest_tensor("twice")],
    )
    # `ops.json` holds the instance once; the variant config wires it twice.
    ops = json.loads((bundle / "ops.json").read_text())
    (bundle / "ops.json").write_text(json.dumps(ops[:1]), encoding="utf-8")
    feeds = {"x": _values((1, 3))}

    result = compile_bundle(bundle, tmp_path / "out")

    assert [node["id"] for node in result.report["nodes"]] == ["scale"]
    _assert_matches(result.load().run(feeds), _runtime_outputs(bundle, feeds))


DET_OPSET = 22


def _determinant_model(order: int, *, name: str):
    """`Det`, whose kernel works on scratch storage its call sites share."""
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Det", ["x"], ["y"], name="det")],
            name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, order, order])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        ),
        opset_imports=[helper.make_opsetid("", DET_OPSET)],
    )


def test_kernel_scratch_two_nodes_share_is_sized_for_the_larger(tmp_path):
    """Two `Det` nodes share one kernel, and its working storage has to fit both.

    The smaller node is compiled first, so a merge that kept the first claim would leave
    the larger node writing past the end of the buffer.
    """
    nodes = [
        _Node(
            id="small",
            model=_determinant_model(2, name="small"),
            inputs=("a",),
            outputs=("da",),
            input_specs=(_spec([1, 2, 2]),),
            output_specs=(_spec([1]),),
            opset=DET_OPSET,
        ),
        _Node(
            id="large",
            model=_determinant_model(5, name="large"),
            inputs=("b",),
            outputs=("db",),
            input_specs=(_spec([1, 5, 5]),),
            output_specs=(_spec([1]),),
            opset=DET_OPSET,
        ),
    ]
    bundle = _write_bundle(
        tmp_path / "dets.fnnx",
        nodes,
        [_manifest_tensor("a"), _manifest_tensor("b")],
        [_manifest_tensor("da"), _manifest_tensor("db")],
    )
    feeds = {"a": _values((1, 2, 2)), "b": _values((1, 5, 5))}

    result = compile_bundle(bundle, tmp_path / "out")

    scratch = re.findall(
        r"static float \w+_work\[(\d+)\];", result.header_path.read_text()
    )
    assert scratch == ["25"]
    _assert_matches(result.load().run(feeds), _runtime_outputs(bundle, feeds))


def test_one_kernel_name_with_two_definitions_is_refused():
    """The invariant kernel sharing rests on: a name encodes everything its code needs.

    Within one graph the emitter enforces it; between the nodes of a pipeline, the merge
    has to, or one node would silently run the other's code.
    """

    def program(definition: str) -> Any:
        return codegen.Program(
            prefix="p",
            graph_name="g",
            source="test",
            opsets={},
            dim_bindings={},
            inputs=(),
            outputs=(),
            weights=(),
            scratch=(),
            functions=(kernels.CFunction("p_kernel", definition),),
            body=(),
        )

    with pytest.raises(CompileError, match="emitted twice with different definitions"):
        bundle_module._merged_functions(
            [program("void a(void);"), program("void b(void);")]
        )


def test_a_node_weight_in_a_side_file_is_embedded_at_compile_time(tmp_path):
    """`has_external_data` nodes: the side file is read while compiling, never afterwards."""
    nodes = _diamond_nodes()
    weight = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    nodes[0].model = _affine_model(weight, 0.5, name="head")
    for tensor in nodes[0].model.graph.initializer:
        if tensor.name == "w":
            onnx.external_data_helper.set_external_data(tensor, location="head_w.bin")
            tensor.ClearField("raw_data")
    bundle = _write_bundle(
        tmp_path / "external.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    side_file = bundle / "ops_artifacts" / "head" / "head_w.bin"
    side_file.write_bytes(weight.tobytes())
    feeds = {"x": _values((1, 3))}
    expected = _runtime_outputs(bundle, feeds)

    compiled = compile_bundle(bundle, tmp_path / "out").load()
    side_file.unlink()

    _assert_matches(compiled.run(feeds), expected)


def test_an_edge_no_node_reads_still_gets_a_buffer(tmp_path):
    """`left` writes `l`, which nothing downstream reads and the manifest does not expose."""
    nodes = _diamond_nodes()
    nodes[3].inputs = ("h", "r")
    bundle = _write_bundle(
        tmp_path / "deadend.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    feeds = {"x": _values((1, 3))}

    result = compile_bundle(bundle, tmp_path / "out")

    _assert_matches(result.load().run(feeds), _runtime_outputs(bundle, feeds))


def test_a_pipeline_of_zero_element_tensors_runs(tmp_path, diamond_bundle):
    """Binding the batch to 0 makes every edge empty; the buffers still have addresses."""
    result = compile_bundle(diamond_bundle, tmp_path / "out", dim_bindings={"batch": 0})
    feeds = {"x": np.zeros((0, 3), dtype=np.float32)}

    outputs = result.load().run(feeds)

    assert result.report["entrypoint"]["inputs"][0]["shape"] == [0, 3]
    _assert_matches(outputs, _runtime_outputs(diamond_bundle, feeds))


def test_an_input_no_node_reads_is_still_a_parameter(tmp_path):
    """An unused pipeline input keeps its place in the signature and builds warning-free."""
    nodes = _diamond_nodes()
    bundle = _write_bundle(
        tmp_path / "unused.fnnx",
        nodes,
        [_manifest_tensor("x"), _manifest_tensor("spare", ["batch", 2])],
        [_manifest_tensor("out")],
    )

    result = compile_bundle(bundle, tmp_path / "out")

    assert [tensor["name"] for tensor in result.report["entrypoint"]["inputs"]] == [
        "x",
        "spare",
    ]
    assert "(void)spare;" in result.header_path.read_text()
    result.load()


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def _assert_nothing_written(output_dir: Path) -> None:
    assert not output_dir.exists() or not list(output_dir.iterdir())


def test_a_non_pipeline_bundle_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "pyfunc.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
        variant="pyfunc",
    )
    output = tmp_path / "out"

    with pytest.raises(CompileError, match="compiles `pipeline` bundles.*`pyfunc`"):
        compile_bundle(bundle, output)
    _assert_nothing_written(output)


def test_a_node_op_the_compiler_cannot_compile_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[1].op = "PyFunc_v1"
    bundle = _write_bundle(
        tmp_path / "custom.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    output = tmp_path / "out"

    with pytest.raises(
        CompileError, match="`left` runs op `PyFunc_v1`.*no node compiler.*`ONNX_v1`"
    ):
        compile_bundle(bundle, output)
    _assert_nothing_written(output)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda nodes: nodes[1].dynamic_attributes.update(
                {"scale": {"name": "scale", "default_value": "1"}}
            ),
            id="op-instance",
        ),
        pytest.param(
            lambda nodes: nodes[2].extra_dynattrs.update({"scale": "other"}),
            id="pipeline-node",
        ),
    ],
)
def test_declared_dynamic_attributes_are_ignored(tmp_path, mutate):
    """`ONNX_v1` defines no dynamic attributes, so a consumer ignores whatever is declared.

    The runtime op ignores them too, which is why it stays the oracle here.
    """
    nodes = _diamond_nodes()
    mutate(nodes)
    bundle = _write_bundle(
        tmp_path / "dynattrs.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("h"), _manifest_tensor("out")],
    )
    feeds = {"x": _values((1, 3))}

    compiled = compile_bundle(bundle, tmp_path / "out").load()

    _assert_matches(compiled.run(feeds), _runtime_outputs(bundle, feeds))


def test_manifest_dynamic_attributes_do_not_gate_compilation(tmp_path):
    """The manifest documents the attributes a caller may pass; it never gates execution.

    Only the op instance and node declarations say that the computation reads one.
    """
    bundle = _write_bundle(
        tmp_path / "manifest-dynattrs.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
        manifest_dynamic_attributes=[{"name": "temperature", "description": "how hot"}],
    )
    output = tmp_path / "out"

    result = compile_bundle(bundle, output)

    assert result.header_path.is_file()


@pytest.mark.parametrize(
    ("tensor", "expected"),
    [
        pytest.param(
            {"name": "x", "content_type": "JSON", "dtype": "MyRecord"},
            "content type `JSON`",
            id="json-content-type",
        ),
        pytest.param(
            _manifest_tensor("x") | {"dtype": "NDContainer[MyRecord]"},
            r"dtype `NDContainer\[MyRecord\]`.*`Array\[\.\.\.\]`",
            id="ndcontainer",
        ),
        pytest.param(
            _manifest_tensor("x") | {"dtype": "Array[string]"},
            "element type `string`.*does not support",
            id="runtime-strings",
        ),
        pytest.param(
            _manifest_tensor("x") | {"dtype": "Array[float16]"},
            "element type `float16`.*does not support",
            id="float16",
        ),
    ],
)
def test_non_tensor_pipeline_io_is_rejected(tmp_path, tensor, expected):
    bundle = _write_bundle(
        tmp_path / "io.fnnx",
        _diamond_nodes(),
        [tensor],
        [_manifest_tensor("out")],
    )
    output = tmp_path / "out"

    with pytest.raises(CompileError, match=f"Manifest input `x`.*{expected}"):
        compile_bundle(bundle, output)
    _assert_nothing_written(output)


def test_two_manifest_entries_sharing_a_name_are_rejected(tmp_path):
    """Two parameters of one name: the second is wired, the first reaches nothing."""
    bundle = _write_bundle(
        tmp_path / "duplicate.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x"), _manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    output = tmp_path / "out"

    with pytest.raises(CompileError, match="Manifest input `x` is declared twice"):
        compile_bundle(bundle, output)
    _assert_nothing_written(output)


def test_a_dimension_written_as_minus_one_is_rejected(tmp_path):
    """`-1` is how ONNX writes an unknown dimension; here it names nothing to bind."""
    nodes = _diamond_nodes()
    nodes[0].input_specs = (_spec([-1, 3]),)
    bundle = _write_bundle(
        tmp_path / "negative.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="`head` input 0 has shape entry -1, which is neither"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_spec_dtype_the_compiler_cannot_hold_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[1].input_specs = (_spec(["batch", 3], dtype="string"),)
    bundle = _write_bundle(
        tmp_path / "strings.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="Op instance `left` input 0 has element type `string`"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_an_edge_no_node_produces_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[3].inputs = ("l", "missing")
    bundle = _write_bundle(
        tmp_path / "dangling.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError,
        match="`join` reads `missing`, which no manifest input and no node",
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_cycle_between_nodes_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[1].inputs = ("out",)
    bundle = _write_bundle(
        tmp_path / "cycle.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(CompileError, match="cycle.*`left`.*`join`"):
        compile_bundle(bundle, tmp_path / "out")


def test_two_nodes_writing_one_edge_are_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[2].outputs = ("l",)
    bundle = _write_bundle(
        tmp_path / "double.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(CompileError, match="`left` and `right` both write `l`"):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_writing_a_manifest_input_is_rejected(tmp_path):
    """The caller's input buffer is `const`, and overwriting it would lose what it holds."""
    nodes = _diamond_nodes()
    nodes[1].outputs = ("x",)
    bundle = _write_bundle(
        tmp_path / "overwrite.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="`left` writes `x`, which is also a manifest input"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_an_output_no_node_produces_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "unproduced.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out"), _manifest_tensor("elsewhere")],
    )

    with pytest.raises(
        CompileError, match="output `elsewhere` is produced by no pipeline node"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_an_output_that_names_a_pipeline_input_is_rejected(tmp_path):
    """A pass-through output would have the caller's output buffer take the input's name."""
    bundle = _write_bundle(
        tmp_path / "passthrough.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out"), _manifest_tensor("x")],
    )

    with pytest.raises(
        CompileError, match="output `x` is produced by no pipeline node"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_manifest_shape_that_contradicts_the_nodes_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "contradiction.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x", ["batch", 5])],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match=r"input `x` declares shape \[1, 5\].*takes \[1, 3\]"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_manifest_dtype_that_contradicts_the_nodes_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "dtype.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x", dtype="int64")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError,
        match=r"input `x` is declared `Array\[int64\]`.*takes `Array\[float32\]`",
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_an_input_nothing_reads_and_nothing_sizes_is_rejected(tmp_path):
    """Without a consumer and without a declared shape, nothing says how big `spare` is."""
    bundle = _write_bundle(
        tmp_path / "unsized.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x"), _manifest_tensor("spare")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError,
        match="input `spare` is read by no pipeline node and declares no shape",
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_an_edge_the_two_ends_disagree_about_is_rejected(tmp_path):
    """One buffer cannot be [batch, 3] where it is written and [batch, 2] where it is read.

    Each node here agrees with its own ONNX graph; what disagrees is the two ends of `h`.
    """
    nodes = _diamond_nodes()
    nodes[1].model = _affine_model(np.array([1.0, 2.0]), 0.0, name="left")
    nodes[1].input_specs = (_spec(["batch", 2]),)
    nodes[1].output_specs = (_spec(["batch", 2]),)
    nodes[3].inputs = ("r", "r")
    bundle = _write_bundle(
        tmp_path / "mismatch.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError,
        match=r"edge `h` is float32\[1, 3\] as output 0 of node `head`, but "
        r"float32\[1, 2\] as input 0 of node `left`",
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_spec_that_contradicts_its_onnx_graph_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[0].input_specs = (_spec(["batch", 3]), _spec(["batch", 3]))
    nodes[0].inputs = ("x", "x")
    bundle = _write_bundle(
        tmp_path / "arity.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="`head`: its op spec declares 2 input.*ONNX graph has 1"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_spec_dimension_that_contradicts_its_graph_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[0].input_specs = (_spec(["batch", 4]),)
    bundle = _write_bundle(
        tmp_path / "dims.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="input 0 .*has size 3 on axis 1 in the ONNX graph, but 4"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_spec_dtype_that_contradicts_its_graph_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[0].input_specs = (_spec(["batch", 3], dtype="int64"),)
    bundle = _write_bundle(
        tmp_path / "elemtype.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError,
        match="`head`: input 0 .*is `float32` in the ONNX graph, but `int64` in the op",
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_wired_to_the_wrong_number_of_edges_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    nodes[3].inputs = ("l",)
    bundle = _write_bundle(
        tmp_path / "wiring.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(
        CompileError, match="`join` is wired to 1 input.*op spec declares 2"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_missing_node_model_is_rejected(tmp_path):
    nodes = _diamond_nodes()
    bundle = _write_bundle(
        tmp_path / "incomplete.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    (bundle / "ops_artifacts" / "left" / "model.onnx").unlink()

    with pytest.raises(CompileError, match="`left`: `model.onnx` is missing"):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_the_ops_file_does_not_define_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "unknown.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    config = json.loads((bundle / "variant_config.json").read_text())
    config["nodes"][1]["op_instance_id"] = "ghost"
    (bundle / "variant_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        CompileError, match="op instance `ghost`, which `ops.json` does not define"
    ):
        compile_bundle(bundle, tmp_path / "out")


def test_a_bundle_path_that_does_not_exist_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="FNNX bundle not found"):
        compile_bundle(tmp_path / "nowhere.fnnx", tmp_path / "out")


def test_a_malformed_bundle_file_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "broken.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    (bundle / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(CompileError, match="Could not read `manifest.json`"):
        compile_bundle(bundle, tmp_path / "out")


def test_a_manifest_missing_a_required_field_is_rejected(tmp_path):
    bundle = _write_bundle(
        tmp_path / "invalid.fnnx",
        _diamond_nodes(),
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    del manifest["producer_name"]
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CompileError, match="`manifest.json` is not valid"):
        compile_bundle(bundle, tmp_path / "out")


def test_a_node_the_onnx_core_rejects_names_the_op_instance(tmp_path):
    """A failure inside a node's graph still points at the node it came from."""
    nodes = _diamond_nodes()
    nodes[1].model = helper.make_model(
        helper.make_graph(
            [helper.make_node("NonZero", ["x"], ["y"], name="nonzero")],
            "left",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [2, "count"])],
        ),
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    nodes[1].output_specs = (_spec([2, 3], dtype="int64"),)
    nodes[2].inputs = ("h",)
    nodes[3].input_specs = (_spec([2, 3], dtype="int64"), _spec(["batch", 3]))
    bundle = _write_bundle(
        tmp_path / "nonzero.fnnx",
        nodes,
        [_manifest_tensor("x")],
        [_manifest_tensor("out")],
    )

    with pytest.raises(CompileError, match="Op instance `left`.*`nonzero`.*NonZero"):
        compile_bundle(bundle, tmp_path / "out")


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------


def test_a_tar_bundle_leaves_no_temporary_directory_behind(tmp_path, monkeypatch):
    """The bundle is unpacked to compile it; what is unpacked has to be cleaned up."""
    unpacked: list[Path] = []
    original = bundle_module.unpack_model

    def record(path):
        directory, temporary = original(path)
        unpacked.append(Path(directory))
        return directory, temporary

    monkeypatch.setattr(bundle_module, "unpack_model", record)
    packed = tmp_path / "packed.tar"
    with tarfile.open(packed, "w") as archive:
        for entry in PIPELINE_BUNDLE.iterdir():
            archive.add(entry, arcname=entry.name)

    compile_bundle(packed, tmp_path / "out")

    assert unpacked and not any(directory.exists() for directory in unpacked)
