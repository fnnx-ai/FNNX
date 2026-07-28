"""Bounded runtime dimensions: one artifact for a whole family of sizes.

The oracles are the same two the rest of the compiler suite uses — the FNNX `Runtime`
(onnxruntime) for bundles and `onnx.reference.ReferenceEvaluator` for single-node models —
run at each size the compiled artifact is then executed at. Nothing here states an expected
output of its own; what *is* asserted directly is the artifact's own contract: buffers sized
for the maximum, a status code for a size outside the range, and a compile error wherever
the dimension cannot be tracked.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fnnx.extras.compilers.c.errors import CompileError, HarnessError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")
specialize = pytest.importorskip("fnnx.extras.compilers.c.onnx.specialize")

from fnnx.extras.compilers.c import compile_bundle, compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.__main__ import main as cli_main  # noqa: E402
from fnnx.runtime import Runtime  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

OPSET = 21
SEED = 20260727
MAX_BATCH = 8
STRICT_FLAGS = ("-std=c99", "-Wall", "-Wextra", "-Werror", "-Werror=vla")
C_COMPILERS = [name for name in ("gcc", "clang") if shutil.which(name)]
ALLOCATION_TOKENS = ("malloc", "calloc", "realloc", "free", "alloca")

PIPELINE_BUNDLE = Path(__file__).parent / "models" / "onnx_pipeline.fnnx"

pytestmark = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _values(shape, *, seed: int = SEED):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _tensor(name, shape, elem_type=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _model(nodes, inputs, outputs, *, initializer=(), opset=OPSET, name="graph"):
    return helper.make_model(
        helper.make_graph(nodes, name, list(inputs), list(outputs), list(initializer)),
        opset_imports=[helper.make_opsetid("", opset)],
    )


def _affine_model():
    """`relu(x @ W + b)` over a symbolic batch: a matmul, a broadcast add, an activation."""
    weight = _values((3, 4), seed=SEED + 1)
    bias = _values((1, 4), seed=SEED + 2)
    return _model(
        [
            helper.make_node("MatMul", ["x", "w"], ["scored"], name="matmul"),
            helper.make_node("Add", ["scored", "b"], ["biased"], name="add"),
            helper.make_node("Relu", ["biased"], ["y"], name="relu"),
        ],
        [_tensor("x", ["batch", 3])],
        [_tensor("y", [None, 4])],
        initializer=[
            numpy_helper.from_array(weight, "w"),
            numpy_helper.from_array(bias, "b"),
        ],
    )


def _pipeline_outputs(feeds):
    return Runtime(str(PIPELINE_BUNDLE)).compute(dict(feeds), {})


@pytest.fixture(scope="module")
def pipeline_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("runtime_dim_pipeline")
    return compile_bundle(
        PIPELINE_BUNDLE, directory, runtime_dims={"batch": MAX_BATCH}, prefix="rtb"
    )


# --------------------------------------------------------------------------------------
# Scenario: Bounded runtime batch dimension
# --------------------------------------------------------------------------------------


def test_a_pipeline_compiled_at_a_maximum_runs_at_every_smaller_batch(
    pipeline_artifact,
):
    model = pipeline_artifact.load()

    for batch in (1, 5, MAX_BATCH):
        inputs = {"x": _values((batch, 3), seed=SEED + batch)}
        computed = model.run(inputs)["y4"]
        expected = np.asarray(_pipeline_outputs(inputs)["y4"])
        assert computed.shape == expected.shape
        np.testing.assert_allclose(computed, expected, rtol=1e-5, atol=1e-5)


def test_buffers_and_the_arena_are_sized_for_the_maximum(pipeline_artifact, tmp_path):
    at_max = compile_bundle(
        PIPELINE_BUNDLE, tmp_path / "pinned", dim_bindings={"batch": MAX_BATCH}
    )

    assert pipeline_artifact.report["memory"] == at_max.report["memory"]
    for tensor in pipeline_artifact.report["entrypoint"]["inputs"]:
        assert tensor["shape"][0] == MAX_BATCH


def test_the_header_publishes_the_maximum_of_each_runtime_dimension(pipeline_artifact):
    header = pipeline_artifact.header_path.read_text(encoding="utf-8")

    assert f"#define RTB_DIM_BATCH_MAX {MAX_BATCH}" in header
    assert pipeline_artifact.report["runtime_dims"] == [
        {
            "name": "batch",
            "max": MAX_BATCH,
            "parameter": "dim_batch",
            "macro": "RTB_DIM_BATCH_MAX",
        }
    ]


def test_entrypoints_take_the_dimension_value_ahead_of_their_buffers(pipeline_artifact):
    header = pipeline_artifact.header_path.read_text(encoding="utf-8")

    assert "int rtb_run(int32_t dim_batch, const float* x, float* y4);" in header
    assert (
        "int rtb_node_linreg_run(int32_t dim_batch, const float* float_input, "
        "float* variable);" in header
    )


def test_the_report_describes_which_axes_scale_with_which_dimension(pipeline_artifact):
    entry = pipeline_artifact.report["entrypoint"]

    assert entry["inputs"][0]["runtime_shape"] == [
        {"dim": "batch", "coefficient": 1},
        3,
    ]
    assert entry["outputs"][0]["runtime_shape"] == [
        {"dim": "batch", "coefficient": 1},
        1,
    ]


def test_a_node_entrypoint_runs_at_a_smaller_batch_than_the_maximum(pipeline_artifact):
    model = pipeline_artifact.load()
    inputs = {"x": _values((3, 3))}

    inside = np.asarray(_pipeline_outputs(inputs)["y4"])
    alone = model.run_node("linreg", {"float_input": inputs["x"]})["variable"]

    assert alone.shape == (3, 1)
    # `linreg` is one of the three regressors the pipeline sums, so its own output is not
    # the pipeline's; what is asserted is that the node entry works at a partial batch and
    # that the whole pipeline agrees with the runtime at the same one.
    np.testing.assert_allclose(model.run(inputs)["y4"], inside, rtol=1e-5, atol=1e-5)


def test_a_call_below_the_maximum_touches_only_the_size_it_asked_for(pipeline_artifact):
    """Compute scales with the value passed, not with the capacity the buffers hold.

    Handed a buffer sized for the maximum, the artifact must write the leading rows of the
    shape this call's dimension value gives and leave the rest of the capacity alone — the
    same property that keeps it inside a caller's exactly-sized buffer.
    """
    model = pipeline_artifact.load()
    library = ctypes.CDLL(str(model.library_path))
    entry = library.rtb_run
    entry.restype = ctypes.c_int
    entry.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]

    values = _values((MAX_BATCH, 3))
    outputs = np.full((MAX_BATCH, 1), 1234.5, dtype=np.float32)
    assert entry(2, values.ctypes.data, outputs.ctypes.data) == 0

    expected = np.asarray(_pipeline_outputs({"x": values[:2]})["y4"])
    np.testing.assert_allclose(outputs[:2], expected, rtol=1e-5, atol=1e-5)
    assert np.all(outputs[2:] == np.float32(1234.5))


# --------------------------------------------------------------------------------------
# Scenario: Out-of-range runtime dim value
# --------------------------------------------------------------------------------------


def test_the_c_entrypoint_rejects_a_dimension_outside_the_compiled_range(
    pipeline_artifact,
):
    """The artifact's own contract, checked below the harness that also validates."""
    model = pipeline_artifact.load()
    library = ctypes.CDLL(str(model.library_path))
    entry = library.rtb_run
    entry.restype = ctypes.c_int
    entry.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]

    values = _values((MAX_BATCH, 3))
    for batch in (0, MAX_BATCH + 1, -1):
        outputs = np.full((MAX_BATCH, 1), 1234.5, dtype=np.float32)
        status = entry(batch, values.ctypes.data, outputs.ctypes.data)
        assert status != 0
        assert np.all(outputs == np.float32(1234.5))


def test_the_harness_rejects_a_dimension_outside_the_compiled_range(pipeline_artifact):
    model = pipeline_artifact.load()

    with pytest.raises(HarnessError, match=r"`batch` is 9, outside the \[1, 8\]"):
        model.run({"x": _values((1, 3))}, dims={"batch": 9})
    with pytest.raises(HarnessError, match=r"`batch` is 0, outside the \[1, 8\]"):
        model.run({"x": _values((1, 3))}, dims={"batch": 0})


def test_the_harness_rejects_an_input_that_disagrees_with_the_stated_dimension(
    pipeline_artifact,
):
    model = pipeline_artifact.load()

    with pytest.raises(HarnessError, match=r"input `x` has shape \(5, 3\)"):
        model.run({"x": _values((5, 3))}, dims={"batch": 4})


def test_the_harness_rejects_a_dimension_the_artifact_does_not_have(pipeline_artifact):
    model = pipeline_artifact.load()

    with pytest.raises(HarnessError, match="`width` is not a runtime dimension"):
        model.run({"x": _values((2, 3))}, dims={"width": 2})


# --------------------------------------------------------------------------------------
# Scenario: Runtime dim that cannot be tracked
# --------------------------------------------------------------------------------------


def test_a_reshape_folding_the_batch_into_a_fixed_dim_is_rejected(tmp_path):
    model = _model(
        [helper.make_node("Reshape", ["x", "shape"], ["y"], name="fold")],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [MAX_BATCH, None])],
        initializer=[
            numpy_helper.from_array(np.array([MAX_BATCH, -1], dtype=np.int64), "shape")
        ],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})

    assert "fold" in str(error.value)
    assert "pin them via `dim_bindings`" in str(error.value)
    assert not list(tmp_path.iterdir())


def test_a_concat_along_the_dimension_with_a_constant_is_rejected(tmp_path):
    model = _model(
        [helper.make_node("Concat", ["x", "pad"], ["y"], axis=0, name="join")],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [None, 4])],
        initializer=[numpy_helper.from_array(np.zeros((2, 4), np.float32), "pad")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})

    assert "Node `join`" in str(error.value)
    assert "constant multiple of a runtime dimension" in str(error.value)
    assert "pin them via `dim_bindings`" in str(error.value)


def test_a_broadcast_the_dimension_cannot_be_proven_against_is_rejected(tmp_path):
    """`[batch, 4] + [3, 4]` broadcasts at batch 3 and at nothing else in the range."""
    model = _model(
        [helper.make_node("Add", ["x", "other"], ["y"], name="combine")],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [None, 4])],
        initializer=[numpy_helper.from_array(np.ones((3, 4), np.float32), "other")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})

    assert "combine" in str(error.value)
    assert "pin them via `dim_bindings`" in str(error.value)


def test_a_slice_clamping_the_dimension_is_rejected(tmp_path):
    """`x[:3]` is the batch below 3 and 3 above it — correct code for neither family."""
    model = _model(
        [helper.make_node("Slice", ["x", "s", "e", "a"], ["y"], name="cut")],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [None, 4])],
        initializer=[
            numpy_helper.from_array(np.array([0], np.int64), "s"),
            numpy_helper.from_array(np.array([3], np.int64), "e"),
            numpy_helper.from_array(np.array([0], np.int64), "a"),
        ],
    )

    with pytest.raises(CompileError, match="pin them via `dim_bindings`"):
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})


def test_a_size_that_grows_faster_than_the_dimension_is_rejected(tmp_path):
    """A `[batch, batch]` intermediate: its element count is the dimension squared."""
    model = _model(
        [
            helper.make_node("MatMul", ["x", "z"], ["t"], name="matmul"),
            helper.make_node("Relu", ["t"], ["y"], name="relu"),
        ],
        [_tensor("x", ["batch", 4]), _tensor("z", [4, "batch"])],
        [_tensor("y", [None, None])],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})

    assert "Node `relu`" in str(error.value)
    assert "does not scale linearly" in str(error.value)


@pytest.mark.parametrize("maximum", [3, 4, MAX_BATCH])
def test_a_size_quadratic_in_the_dimension_is_rejected_at_every_maximum(
    tmp_path, maximum
):
    """A `[batch+1, batch+1]` intermediate, whose element count is the dimension squared.

    Two sizes are exactly what an affine reading of a literal has free parameters, so a
    schedule offering only two would fit this count and lift it wrongly — code correct at
    the sizes probed and silently wrong at the rest of the range.
    """
    model = _model(
        [
            helper.make_node("Concat", ["x", "row"], ["wide"], axis=0),
            helper.make_node("Concat", ["z", "column"], ["tall"], axis=1),
            helper.make_node("MatMul", ["wide", "tall"], ["square"], name="matmul"),
            helper.make_node("ReduceSum", ["square"], ["y"], keepdims=0, name="total"),
        ],
        [_tensor("x", ["batch", 4]), _tensor("z", [4, "batch"])],
        [_tensor("y", [])],
        initializer=[
            numpy_helper.from_array(np.ones((1, 4), np.float32), "row"),
            numpy_helper.from_array(np.ones((4, 1), np.float32), "column"),
        ],
    )

    with pytest.raises(CompileError, match="does not scale linearly"):
        compile_onnx(model, tmp_path, runtime_dims={"batch": maximum})


def test_a_quadratic_size_is_rejected_where_only_its_own_dimension_is_small(tmp_path):
    """The same count, in a dimension with two sizes above 1 beside one with four.

    Whether the emitted code may be read off the sizes above 1 is a question per dimension:
    `rows` has enough of them to check a reading against, `cols` does not, so `cols` has to
    be read at 1 as well however much room `rows` has to spare.
    """
    model = _model(
        [
            helper.make_node("Concat", ["w", "row"], ["wide"], axis=0),
            helper.make_node("Concat", ["z", "column"], ["tall"], axis=1),
            helper.make_node("MatMul", ["wide", "tall"], ["square"], name="matmul"),
            helper.make_node("ReduceSum", ["square"], ["total"], keepdims=0),
            helper.make_node("Mul", ["x", "total"], ["y"], name="scale"),
        ],
        [
            _tensor("x", ["rows", 4]),
            _tensor("z", [4, "cols"]),
            _tensor("w", ["cols", 4]),
        ],
        [_tensor("y", [None, 4])],
        initializer=[
            numpy_helper.from_array(np.ones((1, 4), np.float32), "row"),
            numpy_helper.from_array(np.ones((4, 1), np.float32), "column"),
        ],
    )

    with pytest.raises(CompileError, match="does not scale linearly"):
        compile_onnx(model, tmp_path, runtime_dims={"rows": 6, "cols": 3})


def test_a_value_folded_from_the_dimension_is_rejected(tmp_path):
    """`Shape(x)` reaching the data makes a weight the artifact would have to recompute."""
    model = _model(
        [
            helper.make_node("Shape", ["x"], ["extent"], name="shape"),
            helper.make_node("Cast", ["extent"], ["sized"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["sized"], ["total"], keepdims=0),
            helper.make_node("Mul", ["x", "total"], ["y"], name="scale"),
        ],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [None, 4])],
    )

    with pytest.raises(CompileError, match="pin them via `dim_bindings`"):
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})


def test_a_folded_value_that_only_moves_at_size_one_is_rejected(tmp_path):
    """`min(batch, 2)` folded into a weight: the same at every size in range but 1.

    Constant data is what a fold took out of the graph, so it is read at every probe rather
    than only where the emitted code is — a clamp is flat across most of the range and the
    size it bends at is the one the code comparison leaves out.
    """
    model = _model(
        [
            helper.make_node("Shape", ["x"], ["extent"], name="shape"),
            helper.make_node("Gather", ["extent", "first"], ["rows"], axis=0),
            helper.make_node("Min", ["rows", "ceiling"], ["clamped"], name="clamp"),
            helper.make_node("Cast", ["clamped"], ["scale"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["x", "scale"], ["y"], name="apply"),
        ],
        [_tensor("x", ["batch", 4])],
        [_tensor("y", [None, 4])],
        initializer=[
            numpy_helper.from_array(np.array([0], np.int64), "first"),
            numpy_helper.from_array(np.array([2], np.int64), "ceiling"),
        ],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"batch": MAX_BATCH})

    assert "`clamped`" in str(error.value)
    assert "pin them via `dim_bindings`" in str(error.value)


# --------------------------------------------------------------------------------------
# Option validation
# --------------------------------------------------------------------------------------


def test_a_dimension_cannot_be_both_bound_and_runtime(tmp_path):
    with pytest.raises(CompileError, match="both bound to 4 and declared runtime"):
        compile_bundle(
            PIPELINE_BUNDLE,
            tmp_path,
            dim_bindings={"batch": 4},
            runtime_dims={"batch": MAX_BATCH},
        )
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


@pytest.mark.parametrize("maximum", [0, -3, True, 2.5])
def test_a_runtime_dimension_needs_a_positive_integer_maximum(tmp_path, maximum):
    with pytest.raises(CompileError, match="needs a maximum of at least 1"):
        compile_bundle(PIPELINE_BUNDLE, tmp_path, runtime_dims={"batch": maximum})


def test_two_runtime_dimensions_may_not_share_a_c_identifier(tmp_path):
    with pytest.raises(CompileError, match="sanitize to the C identifier"):
        compile_bundle(PIPELINE_BUNDLE, tmp_path, runtime_dims={"a.b": 4, "a-b": 4})


def test_a_maximum_of_one_leaves_the_artifact_fully_static(tmp_path):
    """The only size in range is 1, so every extent is a constant the buffers are cut to."""
    pinned = compile_bundle(PIPELINE_BUNDLE, tmp_path / "pinned", dim_bindings={})
    result = compile_bundle(
        PIPELINE_BUNDLE, tmp_path / "one", runtime_dims={"batch": 1}
    )
    compiled = result.load()
    inputs = {"x": _values((1, 3))}

    assert result.report["memory"] == pinned.report["memory"]
    np.testing.assert_allclose(
        compiled.run(inputs, dims={"batch": 1})["y4"],
        np.asarray(_pipeline_outputs(inputs)["y4"]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("maximum", [2, 3])
def test_a_small_maximum_says_so_when_the_code_cannot_be_read(tmp_path, maximum):
    """Size 1 is then one of the sizes the emitted code is read at, and it reads apart."""
    with pytest.raises(CompileError) as error:
        compile_bundle(PIPELINE_BUNDLE, tmp_path, runtime_dims={"batch": maximum})

    assert "A maximum below 4 leaves too few sizes above 1" in str(error.value)


# --------------------------------------------------------------------------------------
# Against the reference evaluator, at several sizes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 5, MAX_BATCH])
def test_a_compiled_graph_matches_the_reference_evaluator_at_every_size(
    tmp_path, batch
):
    model = _affine_model()
    compiled = compile_onnx(
        model, tmp_path, runtime_dims={"batch": MAX_BATCH}, prefix="affine"
    ).load()
    values = _values((batch, 3), seed=SEED + batch)

    computed = compiled.run(x=values)["y"]
    expected = ReferenceEvaluator(model).run(None, {"x": values})[0]

    assert computed.shape == expected.shape
    np.testing.assert_allclose(computed, expected, rtol=1e-5, atol=1e-6)


def test_repeated_calls_at_different_sizes_carry_no_state(tmp_path):
    model = _affine_model()
    compiled = compile_onnx(
        model, tmp_path, runtime_dims={"batch": MAX_BATCH}, prefix="affine"
    ).load()
    evaluator = ReferenceEvaluator(model)

    sizes = (MAX_BATCH, 1, 3, MAX_BATCH, 2)
    for batch in sizes:
        values = _values((batch, 3), seed=SEED + batch)
        np.testing.assert_allclose(
            compiled.run(x=values)["y"],
            evaluator.run(None, {"x": values})[0],
            rtol=1e-5,
            atol=1e-6,
        )


def test_an_axis_that_is_a_multiple_of_the_dimension_is_tracked(tmp_path):
    """`Concat` along the batch of two batch-sized tensors: an axis of `2 * batch`."""
    model = _model(
        [helper.make_node("Concat", ["x", "x"], ["y"], axis=0, name="stack")],
        [_tensor("x", ["batch", 3])],
        [_tensor("y", [None, 3])],
    )
    result = compile_onnx(
        model, tmp_path, runtime_dims={"batch": MAX_BATCH}, prefix="stack"
    )

    assert result.report["entrypoint"]["outputs"][0]["runtime_shape"] == [
        {"dim": "batch", "coefficient": 2},
        3,
    ]
    compiled = result.load()
    for batch in (1, 3, MAX_BATCH):
        values = _values((batch, 3), seed=SEED + batch)
        np.testing.assert_array_equal(
            compiled.run(x=values)["y"],
            ReferenceEvaluator(model).run(None, {"x": values})[0],
        )


def test_two_runtime_dimensions_are_tracked_independently(tmp_path):
    model = _model(
        [helper.make_node("MatMul", ["x", "z"], ["y"], name="matmul")],
        [_tensor("x", ["rows", 4]), _tensor("z", [4, "cols"])],
        [_tensor("y", [None, None])],
    )
    compiled = compile_onnx(
        model, tmp_path, runtime_dims={"rows": 6, "cols": 5}, prefix="pair"
    ).load()
    evaluator = ReferenceEvaluator(model)

    for rows, columns in ((1, 1), (6, 5), (4, 2), (2, 5)):
        feeds = {
            "x": _values((rows, 4), seed=SEED + rows),
            "z": _values((4, columns), seed=SEED + columns),
        }
        np.testing.assert_allclose(
            compiled.run(feeds)["y"],
            evaluator.run(None, feeds)[0],
            rtol=1e-5,
            atol=1e-6,
        )


def test_a_size_that_is_the_product_of_two_dimensions_is_rejected(tmp_path):
    """An elementwise loop over `[rows, cols]` counts `rows * cols` — linear in neither.

    The per-dimension probes move one dimension at a time, and a product agrees with a
    linear reading along each of them on its own; only the probe that moves both at once
    tells them apart.
    """
    model = _model(
        [helper.make_node("Add", ["x", "x"], ["y"], name="twice")],
        [_tensor("x", ["rows", "cols"])],
        [_tensor("y", [None, None])],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path, runtime_dims={"rows": 6, "cols": 5})

    assert "Node `twice`" in str(error.value)
    assert "does not scale linearly" in str(error.value)


def test_a_dimension_no_input_carries_has_to_be_passed_explicitly(tmp_path):
    """`ConstantOfShape` has nothing to read the size off, so the caller states it."""
    model = _model(
        [helper.make_node("Add", ["x", "x"], ["y"], name="twice")],
        [_tensor("x", [2, 3])],
        [_tensor("y", [2, 3])],
    )
    compiled = compile_onnx(
        model, tmp_path, runtime_dims={"unused": 4}, prefix="lonely"
    ).load()
    values = _values((2, 3))

    with pytest.raises(HarnessError, match="has to be passed as"):
        compiled.run(x=values)
    np.testing.assert_array_equal(
        compiled.run({"x": values}, dims={"unused": 3})["y"], values + values
    )


# --------------------------------------------------------------------------------------
# The artifact's build contract
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not C_COMPILERS, reason="no system C compiler available")
@pytest.mark.parametrize("compiler", C_COMPILERS)
def test_the_artifact_builds_under_the_strict_flags(
    pipeline_artifact, tmp_path, compiler
):
    unit = tmp_path / "implementation.c"
    unit.write_text(
        f"#define {pipeline_artifact.report['prefix'].upper()}_IMPLEMENTATION\n"
        f'#include "{pipeline_artifact.report["header"]}"\n',
        encoding="utf-8",
    )
    build = subprocess.run(
        [
            compiler,
            *STRICT_FLAGS,
            "-c",
            f"-I{pipeline_artifact.header_path.parent}",
            str(unit),
            "-o",
            str(tmp_path / "artifact.o"),
        ],
        capture_output=True,
        text=True,
    )

    assert build.returncode == 0, build.stderr


def test_the_artifact_allocates_nothing_and_compiles_deterministically(tmp_path):
    first = compile_bundle(
        PIPELINE_BUNDLE, tmp_path / "first", runtime_dims={"batch": MAX_BATCH}
    )
    second = compile_bundle(
        PIPELINE_BUNDLE, tmp_path / "second", runtime_dims={"batch": MAX_BATCH}
    )

    source = first.header_path.read_text(encoding="utf-8")
    for token in ALLOCATION_TOKENS:
        assert not re.search(rf"\b{token}\b", source), token
    assert first.header_path.read_bytes() == second.header_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()


def test_compiling_without_runtime_dimensions_is_unchanged(tmp_path):
    """The default contract stays exactly what it was: no parameter, no macro, no field."""
    result = compile_bundle(PIPELINE_BUNDLE, tmp_path, prefix="plain")
    header = result.header_path.read_text(encoding="utf-8")

    assert "int plain_run(const float* x, float* y4);" in header
    assert "int32_t" not in header
    assert "_DIM_BATCH_MAX" not in header
    assert result.report["runtime_dims"] == []
    assert "runtime_shape" not in result.report["entrypoint"]["inputs"][0]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_the_cli_compiles_with_a_runtime_dimension(tmp_path, capsys):
    status = cli_main(
        [
            str(PIPELINE_BUNDLE),
            "-o",
            str(tmp_path),
            "--runtime-dim",
            f"batch={MAX_BATCH}",
            "--prefix",
            "cli",
        ]
    )
    output = capsys.readouterr().out

    assert status == 0
    assert f"batch<={MAX_BATCH}" in output
    assert f"#define CLI_DIM_BATCH_MAX {MAX_BATCH}" in (tmp_path / "cli.h").read_text(
        encoding="utf-8"
    )


def test_the_cli_rejects_a_dimension_that_is_both_bound_and_runtime(tmp_path, capsys):
    status = cli_main(
        [
            str(PIPELINE_BUNDLE),
            "-o",
            str(tmp_path),
            "--dim",
            "batch=2",
            "--runtime-dim",
            "batch=4",
        ]
    )

    assert status == 1
    assert "declared runtime" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# The probe schedule itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (1, (1,)),
        (2, (1, 2)),
        (3, (1, 2, 3)),
        (4, (1, 2, 3, 4)),
        (8, (1, 2, 3, 7, 8)),
        (32, (1, 2, 3, 31, 32)),
    ],
)
def test_the_probe_schedule_spans_both_ends_of_the_range(maximum, expected):
    dim = specialize.RuntimeDim("batch", maximum, "batch")

    assert specialize.probe_values(dim) == expected
