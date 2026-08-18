"""The elementwise kernel family: the loop it is emitted from, and what it refuses.

What an op computes is settled by the conformance and differential suites, against ONNX's
own corpus and reference evaluator. What is asserted here is the emission contract — one
shared kernel per op, operand types and loop form; the flat loop when nothing broadcasts —
and the errors for the combinations the compiler will not compile at all.
"""

from __future__ import annotations

import re
import shutil

import pytest

from fnnx.extras.compilers.c.errors import CompileError, HarnessError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.onnx.kernels import (  # noqa: E402
    KERNELS,
    NodeContext,
    TensorRef,
)

OPSET = 22

requires_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


def _tensor(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _model(nodes, inputs, outputs, *, initializer=(), opset=OPSET):
    graph = helper.make_graph(
        nodes, "kernels", list(inputs), list(outputs), initializer=list(initializer)
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _compile(model, tmp_path):
    result = compile_onnx(model, tmp_path)
    return result.report, result.header_path.read_text(encoding="utf-8")


def _add_model(left_shape, right_shape, *, elem_type=TensorProto.FLOAT):
    return _model(
        [helper.make_node("Add", ["a", "b"], ["y"], name="add")],
        [_tensor("a", elem_type, left_shape), _tensor("b", elem_type, right_shape)],
        [helper.make_empty_tensor_value_info("y")],
    )


def _kernels(report, op_type: str) -> list[str]:
    prefix = f"{report['prefix']}_{op_type}"
    return [name for name in report["kernels"] if name.startswith(prefix)]


# --------------------------------------------------------------------------------------
# The shared loop
# --------------------------------------------------------------------------------------


def test_operands_of_the_results_shape_take_the_flat_loop(tmp_path):
    """Nothing broadcasts, so the kernel indexes straight into its operands."""
    report, header = _compile(_add_model((2, 3), (2, 3)), tmp_path)

    (kernel,) = _kernels(report, "add")
    assert "strides" not in header
    assert f"{kernel}(\n        y,\n        a,\n        b,\n        6u);" in header


def test_a_broadcasting_operand_takes_the_strided_loop(tmp_path):
    """The stretched axis is a zero stride, and the shape and strides are call-site literals."""
    report, header = _compile(_add_model((2, 3), (3,)), tmp_path)

    (kernel,) = _kernels(report, "add")
    assert "const size_t* strides0" in header
    assert "offset1 += coordinate * strides1[axis];" in header
    assert (
        f"{kernel}(\n        y,\n        a,\n        b,\n        6u,\n        2,\n"
        "        (const size_t[]){2u, 3u},\n"
        "        (const size_t[]){3u, 1u},\n"
        "        (const size_t[]){0u, 1u});" in header
    )


def test_nodes_running_one_op_at_one_type_share_a_kernel(tmp_path):
    """Kernels are shared statics, not code inlined per node."""
    model = _model(
        [
            helper.make_node("Add", ["a", "b"], ["h"], name="first"),
            helper.make_node("Add", ["h", "b"], ["y"], name="second"),
        ],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (2, 3)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "add")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_kernel_is_emitted_per_element_type_and_per_loop_form(tmp_path):
    """A kernel name has to encode everything the emitted code depends on."""
    model = _model(
        [
            helper.make_node("Add", ["a", "b"], ["y"], name="aligned"),
            helper.make_node("Add", ["a", "c"], ["z"], name="broadcasting"),
            helper.make_node("Add", ["i", "i"], ["w"], name="integer"),
        ],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (2, 3)),
            _tensor("c", TensorProto.FLOAT, (3,)),
            _tensor("i", TensorProto.INT32, (2, 3)),
        ],
        [
            helper.make_empty_tensor_value_info("y"),
            helper.make_empty_tensor_value_info("z"),
            helper.make_empty_tensor_value_info("w"),
        ],
    )

    report, _ = _compile(model, tmp_path)

    assert len(set(_kernels(report, "add"))) == 3


def test_an_attribute_value_is_a_kernel_argument_rather_than_a_kernel(tmp_path):
    """Two nodes differing only in an attribute value are one kernel called twice."""
    model = _model(
        [
            helper.make_node("Elu", ["x"], ["h"], name="gentle", alpha=0.5),
            helper.make_node("Elu", ["h"], ["y"], name="steep", alpha=2.0),
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "elu")
    assert header.count(f"static void {kernel}(") == 1
    assert "0.5f" in header and "2.0f" in header


# --------------------------------------------------------------------------------------
# Moving bytes rather than computing them
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op_type", "source", "target"),
    [
        ("Cast", TensorProto.FLOAT, TensorProto.FLOAT),
        ("BitCast", TensorProto.FLOAT, TensorProto.INT32),
    ],
)
def test_a_conversion_that_changes_no_bits_is_a_copy(tmp_path, op_type, source, target):
    """Neither op has anything to compute per element, so neither emits a kernel at all."""
    model = _model(
        [helper.make_node(op_type, ["x"], ["y"], name="convert", to=target)],
        [_tensor("x", source, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        opset=26,
    )

    report, header = _compile(model, tmp_path)

    assert not _kernels(report, op_type.lower())
    assert "memcpy(y, x, 6u * sizeof(*y));" in header


def test_a_cast_kernel_is_emitted_per_target_type(tmp_path):
    """The target type is what a cast's code depends on, so it has to name the kernel."""
    model = _model(
        [
            helper.make_node("Cast", ["x"], ["a"], name="narrow", to=TensorProto.INT16),
            helper.make_node("Cast", ["x"], ["b"], name="wide", to=TensorProto.INT32),
            helper.make_node("Cast", ["x"], ["c"], name="again", to=TensorProto.INT32),
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info(name) for name in ("a", "b", "c")],
    )

    report, header = _compile(model, tmp_path)

    assert len(set(_kernels(report, "cast"))) == 2
    assert "(int16_t)x0" in header and "(int32_t)x0" in header


@requires_c_compiler
def test_casting_to_and_from_bool_emits_a_kernel_for_each_direction(tmp_path):
    """`bool` and `uint8` are one C type, and the two directions are not one formula.

    A byte casts to true when it is nonzero, while a boolean carries the value it already
    holds, so a kernel named after its C types alone would have the two collide.
    """
    model = _model(
        [
            helper.make_node("Cast", ["x"], ["b"], name="to_bool", to=TensorProto.BOOL),
            helper.make_node(
                "Cast", ["b"], ["y"], name="to_byte", to=TensorProto.UINT8
            ),
        ],
        [_tensor("x", TensorProto.UINT8, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )
    values = np.array([[0, 1, 2], [3, 200, 255]], dtype=np.uint8)

    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"x": values})

    assert len(set(_kernels(result.report, "cast"))) == 2
    expected = ReferenceEvaluator(model).run(None, {"x": values})
    np.testing.assert_array_equal(outputs["y"], expected[0])


@requires_c_compiler
def test_isinf_detecting_neither_infinity_never_reads_its_operand(tmp_path):
    """Nothing is detected, so the result is false everywhere.

    A kernel that still read the operand would leave the local unused, which the artifact's
    `-Werror` build contract turns into a failure rather than a warning.
    """
    model = _model(
        [
            helper.make_node(
                "IsInf",
                ["x"],
                ["y"],
                name="never",
                detect_positive=0,
                detect_negative=0,
            )
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"x": np.full((2, 3), np.inf, dtype=np.float32)})

    (kernel,) = _kernels(result.report, "isinf")
    assert f"{kernel}(\n        y,\n        6u);" in result.header_path.read_text()
    assert not outputs["y"].any()


# --------------------------------------------------------------------------------------
# Rearranging elements rather than computing them
# --------------------------------------------------------------------------------------


def _int64(name, values):
    return onnx.numpy_helper.from_array(np.array(values, dtype=np.int64), name)


@pytest.mark.parametrize(
    ("op_type", "shape", "initializer"),
    [
        ("Reshape", (2, 3), [_int64("p", [6])]),
        ("Flatten", (2, 3), []),
        ("Squeeze", (1, 2, 3), [_int64("p", [0])]),
        ("Unsqueeze", (2, 3), [_int64("p", [0])]),
    ],
)
def test_an_op_that_only_relabels_axes_is_a_copy(tmp_path, op_type, shape, initializer):
    """None of them moves an element: the row-major buffer is the same, under other axes."""
    inputs = ["x", "p"] if initializer else ["x"]
    model = _model(
        [helper.make_node(op_type, inputs, ["y"], name="view")],
        [_tensor("x", TensorProto.FLOAT, shape)],
        [helper.make_empty_tensor_value_info("y")],
        initializer=initializer,
    )

    report, header = _compile(model, tmp_path)

    assert not _kernels(report, "copy")
    assert "memcpy(y, x, 6u * sizeof(*y));" in header


def test_a_move_that_stays_contiguous_is_a_memcpy(tmp_path):
    """Concatenating along the outermost axis lays each operand down in one unbroken run."""
    model = _model(
        [helper.make_node("Concat", ["a", "b"], ["y"], name="join", axis=0)],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (4, 3)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    report, header = _compile(model, tmp_path)

    assert not _kernels(report, "copy")
    assert "memcpy(y, a, 6u * sizeof(*y));" in header
    assert "memcpy(y + 6, b, 12u * sizeof(*y));" in header


def test_a_reordering_move_passes_its_strides_as_call_site_literals(tmp_path):
    """A transpose is the operand read along permuted strides, written out in order."""
    model = _model(
        [helper.make_node("Transpose", ["x"], ["y"], name="swap", perm=[1, 0])],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "copy")
    assert (
        f"{kernel}(\n        y,\n        x,\n        6u,\n        2,\n"
        "        (const size_t[]){3u, 2u},\n"
        "        (const ptrdiff_t[]){2, 1},\n"
        "        (const ptrdiff_t[]){1, 3},\n"
        "        0,\n"
        "        0);" in header
    )


def test_the_views_share_one_move_kernel_per_element_type(tmp_path):
    """Every one of them walks the same addressing, so they are one shared static."""
    model = _model(
        [
            helper.make_node("Transpose", ["x"], ["t"], name="swap", perm=[1, 0]),
            helper.make_node("Tile", ["t", "r"], ["u"], name="repeat"),
            helper.make_node("Slice", ["u", "s", "e"], ["y"], name="cut"),
            helper.make_node("Transpose", ["i"], ["z"], name="swap_ints", perm=[1, 0]),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("i", TensorProto.INT32, (2, 3)),
        ],
        [
            helper.make_empty_tensor_value_info("y"),
            helper.make_empty_tensor_value_info("z"),
        ],
        initializer=[_int64("r", [2, 2]), _int64("s", [1, 1]), _int64("e", [5, 3])],
    )

    report, header = _compile(model, tmp_path)

    kernels = set(_kernels(report, "copy"))
    assert len(kernels) == 2
    for kernel in kernels:
        assert header.count(f"static void {kernel}(") == 1
    assert f"{report['prefix']}_copy_float" in kernels
    assert f"{report['prefix']}_copy_int32_t" in kernels


@requires_c_compiler
def test_a_view_of_a_zero_element_tensor_emits_no_move(tmp_path):
    """There is nothing to move, and an empty `memcpy` would still need a valid pointer."""
    model = _model(
        [helper.make_node("Tile", ["x", "r"], ["y"], name="repeat")],
        [_tensor("x", TensorProto.FLOAT, (2, 0))],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[_int64("r", [2, 3])],
    )

    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"x": np.zeros((2, 0), dtype=np.float32)})

    assert "memcpy" not in result.header_path.read_text()
    assert outputs["y"].shape == (4, 0)


# --------------------------------------------------------------------------------------
# Walking a tensor by axes
# --------------------------------------------------------------------------------------


def test_reductions_of_one_op_and_type_share_a_kernel(tmp_path):
    """Extents, strides and group counts are arguments, so the axes do not name a kernel."""
    model = _model(
        [
            helper.make_node("ReduceSum", ["x"], ["a"], name="rows", axes=[0]),
            helper.make_node(
                "ReduceSum", ["x"], ["b"], name="columns", axes=[1], keepdims=0
            ),
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info(name) for name in ("a", "b")],
        opset=11,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "reducesum")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_scan_axis_the_graph_fixes_compiles_to_a_single_call(tmp_path):
    model = _model(
        [helper.make_node("CumSum", ["x", "axis"], ["y"], name="scan")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[onnx.numpy_helper.from_array(np.array(1, dtype=np.int64), "axis")],
        opset=14,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "cumsum")
    assert "switch" not in header
    assert header.count(f"{kernel}(\n") == 2


@requires_c_compiler
def test_a_runtime_scan_axis_compiles_to_a_call_per_axis(tmp_path):
    """The axis decides which elements a scan visits and no shape at all, so a graph that
    computes it still compiles: every value it can take is a call site of its own."""
    model = _model(
        [helper.make_node("CumSum", ["x", "axis"], ["y"], name="scan")],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("axis", TensorProto.INT32, ()),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )
    values = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = compile_onnx(model, tmp_path)
    compiled = result.load()

    header = result.header_path.read_text(encoding="utf-8")
    assert header.count("case 0:") == 1 and header.count("case 1:") == 1
    for axis in (0, 1, -1, -2):
        feeds = {"x": values, "axis": np.array(axis, dtype=np.int32)}
        expected = ReferenceEvaluator(model).run(None, feeds)
        np.testing.assert_array_equal(compiled.run(feeds)["y"], expected[0])


@requires_c_compiler
@pytest.mark.parametrize("axis", [2, -3])
def test_a_scan_axis_outside_the_rank_returns_a_nonzero_status(tmp_path, axis):
    """The status enum is what a run-time-checked operand reports through."""
    model = _model(
        [helper.make_node("CumSum", ["x", "axis"], ["y"], name="scan")],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("axis", TensorProto.INT32, ()),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )

    compiled = compile_onnx(model, tmp_path).load()

    with pytest.raises(HarnessError, match="status 1"):
        compiled.run(
            {
                "x": np.zeros((2, 3), dtype=np.float32),
                "axis": np.array(axis, dtype=np.int32),
            }
        )


@pytest.mark.parametrize("axis", [2, -3])
def test_a_scan_axis_the_graph_fixes_outside_the_rank_is_rejected(tmp_path, axis):
    """An axis the graph pins is checked where it is known: at compile time, not at run."""
    model = _model(
        [helper.make_node("CumSum", ["x", "axis"], ["y"], name="scan")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[
            onnx.numpy_helper.from_array(np.array(axis, dtype=np.int64), "axis")
        ],
        opset=14,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`scan`" in message
    assert f"axis {axis}" in message
    assert "rank-2" in message


def test_a_scan_of_a_scalar_is_rejected(tmp_path):
    model = _model(
        [helper.make_node("CumSum", ["x", "axis"], ["y"], name="scan")],
        [
            _tensor("x", TensorProto.FLOAT, ()),
            _tensor("axis", TensorProto.INT32, ()),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`scan`" in message
    assert "`x`" in message


def test_axes_naming_one_dimension_twice_are_rejected(tmp_path):
    """Reducing an axis twice has no meaning; numpy refuses it and so does the compiler."""
    model = _model(
        [helper.make_node("ReduceSum", ["x"], ["y"], name="total", axes=[1, 1])],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        opset=11,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`total`" in message
    assert "[1, 1]" in message


# --------------------------------------------------------------------------------------
# Normalizing by a group's own statistics
# --------------------------------------------------------------------------------------


def _instance_norm_node(name, data, **attributes):
    return helper.make_node(
        "InstanceNormalization",
        [data, "s", "b"],
        [f"y_{name}"],
        name=name,
        **attributes,
    )


def test_normalizations_of_one_op_and_type_share_a_kernel(tmp_path):
    """Extents, strides and the epsilon are arguments, so neither names a kernel."""
    model = _model(
        [
            _instance_norm_node("wide", "x"),
            _instance_norm_node("narrow", "z", epsilon=0.01),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3, 4, 5)),
            _tensor("z", TensorProto.FLOAT, (2, 3, 4)),
            _tensor("s", TensorProto.FLOAT, (3,)),
            _tensor("b", TensorProto.FLOAT, (3,)),
        ],
        [
            helper.make_empty_tensor_value_info(f"y_{name}")
            for name in ("wide", "narrow")
        ],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "instancenormalization")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_kernel_is_emitted_per_set_of_statistics_asked_for(tmp_path):
    """The mean and the inverse deviation are buffers the kernel writes into.

    Two nodes that report different ones do not run the same code, so a name that did not
    say which they are would have the two collide.
    """
    model = _model(
        [
            helper.make_node("LayerNormalization", ["x", "s"], ["y"], name="plain"),
            helper.make_node(
                "LayerNormalization", ["x", "s"], ["z", "m", "d"], name="reporting"
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("s", TensorProto.FLOAT, (3,)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z", "m", "d")],
        opset=17,
    )

    report, header = _compile(model, tmp_path)

    kernels = set(_kernels(report, "layernormalization"))
    assert len(kernels) == 2
    for kernel in kernels:
        assert header.count(f"static void {kernel}(") == 1


@requires_c_compiler
def test_a_running_statistic_the_node_skips_is_not_computed(tmp_path):
    """ONNX lets a node drop an optional output by naming it the empty string."""
    model = _model(
        [
            helper.make_node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y", "", "running_var"],
                name="norm",
                training_mode=1,
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3, 2)),
            _tensor("s", TensorProto.FLOAT, (3,)),
            _tensor("b", TensorProto.FLOAT, (3,)),
            _tensor("m", TensorProto.FLOAT, (3,)),
            _tensor("v", TensorProto.FLOAT, (3,)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "running_var")],
        opset=15,
    )
    feeds = {
        "x": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "s": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "m": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "v": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    expected = ReferenceEvaluator(model).run(None, feeds)
    for name, want in zip(("y", "running_var"), expected):
        np.testing.assert_allclose(outputs[name], want, rtol=1e-6, atol=1e-6)


@requires_c_compiler
def test_a_training_node_reporting_no_statistic_still_builds(tmp_path):
    """`momentum` blends the running statistics and is read nowhere else.

    A node that reports neither is one ONNX's own shape inference refuses, but the
    reference evaluator computes its `Y` all the same, so the kernel is emitted — and has
    to leave the argument out rather than take one it never reads, which the artifact's
    own `-Werror=unused-parameter` build would refuse. The result shape is declared
    because inference derives none for a node it will not vouch for.
    """
    model = _model(
        [
            helper.make_node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y"],
                name="norm",
                training_mode=1,
                momentum=0.7,
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3, 2)),
            *(_tensor(name, TensorProto.FLOAT, (3,)) for name in ("s", "b", "m", "v")),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 3, 2))],
        opset=15,
    )
    feeds = {
        "x": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "s": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "m": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "v": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    (expected,) = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_allclose(outputs["y"], expected, rtol=1e-6, atol=1e-6)


def test_a_stash_type_that_takes_no_statistics_is_rejected(tmp_path):
    """`stash_type` names the precision stage one runs in, which has to be a float one."""
    model = _model(
        [
            helper.make_node(
                "LayerNormalization",
                ["x", "s"],
                ["y"],
                name="norm",
                stash_type=TensorProto.INT32,
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("s", TensorProto.FLOAT, (3,)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=17,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`norm`" in message
    assert "INT32" in message


def test_groups_that_do_not_divide_the_channels_are_rejected(tmp_path):
    model = _model(
        [
            helper.make_node(
                "GroupNormalization", ["x", "s", "b"], ["y"], name="norm", num_groups=2
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 5, 3)),
            _tensor("s", TensorProto.FLOAT, (5,)),
            _tensor("b", TensorProto.FLOAT, (5,)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=21,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`norm`" in message
    assert "5 channel(s)" in message


def test_an_lp_order_onnx_does_not_define_is_rejected(tmp_path):
    """`p` selects the norm, and ONNX defines the op for the first two only."""
    model = _model(
        [helper.make_node("LpNormalization", ["x"], ["y"], name="norm", p=3)],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`norm`" in message
    assert "[1, 2]" in message


@requires_c_compiler
def test_the_l1_norm_sums_absolute_values(tmp_path):
    """The one place onnxruntime is the oracle, because ONNX's own two are blind here.

    The reference evaluator raises the elements to the power `p` without taking their
    absolute value, so at `p` = 1 it computes a signed sum rather than a norm, and the
    corpus's `l1normalization` tests all carry non-negative data — nothing else in this
    suite can tell the two apart. onnxruntime, the second oracle the compiler's parity
    testing rests on, computes the norm ONNX defines.
    """
    ort = pytest.importorskip("onnxruntime")
    model = _model(
        [helper.make_node("LpNormalization", ["x"], ["y"], name="norm", p=1, axis=1)],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        # onnxruntime implements the op's first revision only, which is the one every
        # model below opset 22 -- and so this one -- dispatches to.
        opset=21,
    )
    feeds = {"x": np.array([[1.0, -2.0, 3.0], [-1.0, -1.0, 2.0]], dtype=np.float32)}

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    path = tmp_path / "lpnormalization.onnx"
    onnx.save(model, str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = session.run(None, feeds)[0]
    np.testing.assert_allclose(outputs["y"], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("op_type", "inputs", "shapes"),
    [
        ("InstanceNormalization", ["x", "s", "b"], [(4,), (1,), (1,)]),
        ("LRN", ["x"], [(4,)]),
    ],
)
def test_a_normalization_without_a_channel_axis_is_rejected(
    tmp_path, op_type, inputs, shapes
):
    """Both read their operand as instances by channels; a vector has no channels."""
    attributes = {"size": 3} if op_type == "LRN" else {}
    model = _model(
        [helper.make_node(op_type, inputs, ["y"], name="norm", **attributes)],
        [
            _tensor(name, TensorProto.FLOAT, shape)
            for name, shape in zip(inputs, shapes)
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`norm`" in message
    assert "rank of at least 2" in message


def test_batch_normalization_at_inference_refuses_the_training_outputs(tmp_path):
    """ONNX leaves the running statistics undefined outside training mode.

    Its own shape inference refuses the node too, but only where it can derive the extra
    outputs' types; a model that declares them itself gets this far, and is stopped here
    rather than handed a buffer nothing writes.
    """
    model = _model(
        [
            helper.make_node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                ["y", "rm", "rv"],
                name="norm",
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            *(_tensor(name, TensorProto.FLOAT, (3,)) for name in ("s", "b", "m", "v")),
        ],
        [
            _tensor("y", TensorProto.FLOAT, (2, 3)),
            _tensor("rm", TensorProto.FLOAT, (3,)),
            _tensor("rv", TensorProto.FLOAT, (3,)),
        ],
        opset=15,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`norm`" in message
    assert "training mode only" in message


# --------------------------------------------------------------------------------------
# Reading through an index
# --------------------------------------------------------------------------------------


def _gather_model(op_type, data_shape, index_shape, *, index_type=TensorProto.INT64):
    return _model(
        [helper.make_node(op_type, ["x", "i"], ["y"], name="pick", axis=0)],
        [
            _tensor("x", TensorProto.FLOAT, data_shape),
            _tensor("i", index_type, index_shape),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )


def test_gathering_at_one_element_and_index_type_shares_a_kernel(tmp_path):
    """Two nodes reading the same types read them through the same emitted loop."""
    model = _model(
        [
            helper.make_node("Gather", ["x", "i"], ["a"], name="first", axis=0),
            helper.make_node("Gather", ["x", "j"], ["b"], name="second", axis=1),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (3, 4)),
            _tensor("i", TensorProto.INT64, (2,)),
            _tensor("j", TensorProto.INT64, (3,)),
        ],
        [
            helper.make_empty_tensor_value_info("a"),
            helper.make_empty_tensor_value_info("b"),
        ],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "gather")
    assert header.count(f"static int {kernel}(") == 1
    assert header.count(f"if ({kernel}(") == 2


def test_a_gather_kernel_is_emitted_per_index_type(tmp_path):
    """The index type is part of the loop's signature, so int32 and int64 are two kernels."""
    model = _model(
        [
            helper.make_node("Gather", ["x", "i"], ["a"], name="wide", axis=0),
            helper.make_node("Gather", ["x", "j"], ["b"], name="narrow", axis=0),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (3, 4)),
            _tensor("i", TensorProto.INT64, (2,)),
            _tensor("j", TensorProto.INT32, (2,)),
        ],
        [
            helper.make_empty_tensor_value_info("a"),
            helper.make_empty_tensor_value_info("b"),
        ],
    )

    report, _ = _compile(model, tmp_path)

    assert sorted(_kernels(report, "gather")) == [
        f"{report['prefix']}_gather_float_int32_t",
        f"{report['prefix']}_gather_float_int64_t",
    ]


@requires_c_compiler
@pytest.mark.parametrize(
    ("op_type", "index"),
    [("Gather", 3), ("Gather", -4), ("GatherElements", 3), ("GatherND", 3)],
)
def test_an_index_outside_its_axis_returns_a_nonzero_status(tmp_path, op_type, index):
    """An index operand comes from the caller, so the artifact refuses one it cannot serve.

    ONNX defines an index only within its axis, counted from either end; anything else would
    read past the buffer, which is what the status enum exists to report instead.
    """
    index_shape = {"GatherND": (2, 1), "GatherElements": (2, 4)}.get(op_type, (2,))
    compiled = compile_onnx(
        _gather_model(op_type, (3, 4), index_shape), tmp_path
    ).load()

    with pytest.raises(HarnessError, match="status 1"):
        compiled.run(
            {
                "x": np.zeros((3, 4), dtype=np.float32),
                "i": np.full(index_shape, index, dtype=np.int64),
            }
        )


@requires_c_compiler
def test_a_sequence_length_outside_the_time_axis_returns_a_nonzero_status(tmp_path):
    """ONNX defines a length as being in `[1, s]`; the reversal of anything else is nothing."""
    model = _model(
        [
            helper.make_node(
                "ReverseSequence",
                ["x", "lens"],
                ["y"],
                name="reverse",
                batch_axis=0,
                time_axis=1,
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("lens", TensorProto.INT64, (2,)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    compiled = compile_onnx(model, tmp_path).load()

    for lengths in ([0, 2], [2, 4]):
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run(
                {
                    "x": np.zeros((2, 3), dtype=np.float32),
                    "lens": np.array(lengths, dtype=np.int64),
                }
            )


def test_an_eye_reads_nothing_but_the_shape_of_its_operand(tmp_path):
    """EyeLike is a function of coordinates, so the operand it is shaped like goes unread."""
    model = _model(
        [helper.make_node("EyeLike", ["x"], ["y"], name="eye")],
        [_tensor("x", TensorProto.FLOAT, (3, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "eyelike")
    entrypoint = header.split(f"int {report['prefix']}_run(")[-1]
    assert "(void)x;" in entrypoint
    assert "x" not in entrypoint.split(f"{kernel}(")[1].split(");")[0]


def test_a_pad_kernel_is_emitted_per_mode(tmp_path):
    """What a mode does with a coordinate outside the operand is the kernel's whole body."""
    model = _model(
        [
            helper.make_node("Pad", ["x", "p"], ["a"], name="fill", mode="constant"),
            helper.make_node("Pad", ["x", "p"], ["b"], name="mirror", mode="reflect"),
            helper.make_node("Pad", ["x", "p"], ["c"], name="repeat", mode="edge"),
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info(name) for name in ("a", "b", "c")],
        initializer=[_int64("p", [1, 0, 1, 0])],
    )

    report, _ = _compile(model, tmp_path)

    assert sorted(_kernels(report, "pad")) == [
        f"{report['prefix']}_pad_{mode}_float"
        for mode in ("constant", "edge", "reflect")
    ]


# --------------------------------------------------------------------------------------
# The matrix products and the determinant
# --------------------------------------------------------------------------------------


def test_matmuls_of_one_element_type_share_a_kernel(tmp_path):
    """Shapes reach the kernel as call-site literals, so batching is not a kernel of its own."""
    model = _model(
        [
            helper.make_node("MatMul", ["a", "b"], ["p"], name="plain"),
            helper.make_node("MatMul", ["c", "d"], ["q"], name="batched"),
        ],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (3, 4)),
            _tensor("c", TensorProto.FLOAT, (5, 2, 3)),
            _tensor("d", TensorProto.FLOAT, (3,)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "matmul")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_determinants_share_one_working_matrix_sized_for_the_largest(tmp_path):
    """Elimination needs a copy of its matrix, which the artifact reserves at compile time.

    Both nodes run the same kernel, so both eliminate in the same static buffer — they run one
    after the other — and it is large enough for the bigger of the two.
    """
    model = _model(
        [
            helper.make_node("Det", ["x"], ["a"], name="small"),
            helper.make_node("Det", ["z"], ["b"], name="large"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (3, 2, 2)),
            _tensor("z", TensorProto.FLOAT, (4, 4)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("a", "b")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "det")
    assert f"static float {kernel}_work[16];" in header
    assert header.count(f"{kernel}_work,") == 2
    assert report["memory"]["arena_bytes"] == 16 * 4


def test_a_determinant_of_another_element_type_gets_its_own_working_matrix(tmp_path):
    """The buffer a kernel eliminates in is typed like the kernel, so the two do not share."""
    model = _model(
        [
            helper.make_node("Det", ["x"], ["a"], name="single"),
            helper.make_node("Det", ["z"], ["b"], name="wide"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 2)),
            _tensor("z", TensorProto.DOUBLE, (2, 2)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("a", "b")],
    )

    report, header = _compile(model, tmp_path)

    prefix = report["prefix"]
    assert f"static float {prefix}_det_float_work[4];" in header
    assert f"static double {prefix}_det_double_work[4];" in header
    assert report["memory"]["arena_bytes"] == 4 * 4 + 4 * 8


# --------------------------------------------------------------------------------------
# The convolutions
# --------------------------------------------------------------------------------------


def _conv_model(
    x_shape,
    w_shape,
    *,
    op_type="Conv",
    bias=None,
    output=None,
    opset=OPSET,
    **attributes,
):
    names = ["x", "w"] + (["b"] if bias else [])
    inputs = [
        _tensor("x", TensorProto.FLOAT, x_shape),
        _tensor("w", TensorProto.FLOAT, w_shape),
    ]
    if bias:
        inputs.append(_tensor("b", TensorProto.FLOAT, bias))
    return _model(
        [helper.make_node(op_type, names, ["y"], name="conv", **attributes)],
        inputs,
        [
            helper.make_empty_tensor_value_info("y")
            if output is None
            else _tensor("y", TensorProto.FLOAT, output)
        ],
        opset=opset,
    )


def test_convolutions_of_one_element_type_share_a_kernel(tmp_path):
    """The geometry reaches the kernel as call-site literals, so rank is not a kernel of its
    own: a 1-D and a grouped 2-D convolution run the same code at different arguments."""
    model = _model(
        [
            helper.make_node("Conv", ["x", "w"], ["p"], name="signal"),
            helper.make_node("Conv", ["i", "k"], ["q"], name="image", group=2),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 6)),
            _tensor("w", TensorProto.FLOAT, (3, 2, 3)),
            _tensor("i", TensorProto.FLOAT, (2, 4, 5, 5)),
            _tensor("k", TensorProto.FLOAT, (6, 2, 3, 3)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "conv")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_the_resolved_padding_reaches_the_kernel_as_literals(tmp_path):
    """`auto_pad` is resolved at compile time: the kernel only ever sees concrete pads.

    A 4-wide window at stride 2 over a 5-wide axis needs three pads, and SAME_LOWER puts the
    odd one at the beginning — two here against SAME_UPPER's one. The second axis is padded
    for the reach of a 2-dilated window instead, which splits evenly.
    """
    model = _conv_model(
        (1, 1, 5, 5),
        (1, 1, 4, 2),
        auto_pad="SAME_LOWER",
        strides=[2, 1],
        dilations=[1, 2],
    )

    _, header = _compile(model, tmp_path)

    assert "(const ptrdiff_t[]){2, 1}" in header


def test_a_valid_convolution_pads_nothing(tmp_path):
    """`VALID` is not `SAME`: it drops the positions a full window does not fit in.

    ONNX's own shape inference arbitrates this. It derives the result's shape from the pads
    the mode implies, and the kernel refuses to be emitted against a buffer its addressing
    disagrees with — so a compiler reading `VALID` as `SAME` would fail to compile this
    model rather than quietly pad it out to the 5x5 SAME shape.
    """
    report, header = _compile(
        _conv_model((1, 1, 5, 5), (1, 1, 3, 3), auto_pad="VALID"), tmp_path
    )

    assert report["entrypoint"]["outputs"][0]["shape"] == [1, 1, 3, 3]
    assert "(const ptrdiff_t[]){0, 0}" in header


def test_a_convolution_writing_no_elements_emits_no_call(tmp_path):
    """An empty batch leaves nothing to convolve, and no loop that could read past a buffer."""
    report, header = _compile(_conv_model((0, 2, 5, 5), (3, 2, 3, 3)), tmp_path)

    assert not _kernels(report, "conv")
    assert "conv" not in header.split(f"int {report['prefix']}_run(")[-1]


@requires_c_compiler
def test_a_grouped_convolution_reads_only_its_own_channels(tmp_path):
    """Each group convolves its own slice of the channels, which zeroed weights expose.

    Zeroing the second group's filters leaves the first group's output untouched, so a
    kernel addressing the whole channel stack per filter would change the answer. What the
    surviving group *computes* is settled by the conformance and differential suites.
    """
    model = _conv_model((1, 4, 5, 5), (4, 2, 3, 3), group=2)
    compiled = compile_onnx(model, tmp_path).load()
    x = np.arange(100, dtype=np.float32).reshape(1, 4, 5, 5) / 100
    w = np.ones((4, 2, 3, 3), dtype=np.float32)
    masked = w.copy()
    masked[2:] = 0.0

    full = compiled.run({"x": x, "w": w})["y"]
    partial = compiled.run({"x": x, "w": masked})["y"]

    np.testing.assert_array_equal(full[:, :2], partial[:, :2])
    np.testing.assert_array_equal(partial[:, 2:], np.zeros_like(partial[:, 2:]))


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"auto_pad": "SAME_UPPER", "pads": [1, 1, 1, 1]}, "mutually exclusive"),
        ({"kernel_shape": [2, 2]}, "the filter it is handed measures [3, 3]"),
        ({"auto_pad": "SAME"}, "is not one of the modes ONNX defines"),
        ({"group": 3}, "3 group(s) takes"),
        ({"strides": [0, 1]}, "ONNX defines them as positive"),
        ({"dilations": [1, 0]}, "ONNX defines them as positive"),
        ({"group": 0}, "positive count"),
    ],
)
def test_a_convolution_the_compiler_cannot_place_is_rejected(
    tmp_path, attributes, message
):
    model = _conv_model((1, 2, 5, 5), (2, 2, 3, 3), output=(1, 2, 3, 3), **attributes)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "output", "message"),
    [
        ((1, 3), (2, 3), (1, 2), "rank 3 or more"),
        ((1, 1, 5, 5), (1, 1, 3), (1, 1, 3, 3), "ONNX defines both as rank 4"),
    ],
)
def test_a_convolution_of_mismatched_ranks_is_rejected(
    tmp_path, x_shape, w_shape, output, message
):
    """Declared shapes get past ONNX's own inference, so the kernel checks them itself."""
    model = _conv_model(x_shape, w_shape, output=output)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_a_convolution_bias_is_one_value_per_output_channel(tmp_path):
    model = _conv_model((1, 2, 5, 5), (2, 2, 3, 3), bias=(3,))

    with pytest.raises(CompileError, match="one bias per output channel"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The transposed convolution
# --------------------------------------------------------------------------------------


def _transpose_model(x_shape, w_shape, **kwargs):
    return _conv_model(x_shape, w_shape, op_type="ConvTranspose", **kwargs)


def test_transposed_convolutions_of_one_element_type_share_a_kernel(tmp_path):
    """One kernel for the op, as with Conv: the geometry is what the call sites differ in."""
    model = _model(
        [
            helper.make_node("ConvTranspose", ["x", "w"], ["p"], name="signal"),
            helper.make_node("ConvTranspose", ["i", "k"], ["q"], name="image", group=2),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 6)),
            _tensor("w", TensorProto.FLOAT, (2, 3, 3)),
            _tensor("i", TensorProto.FLOAT, (2, 4, 5, 5)),
            _tensor("k", TensorProto.FLOAT, (4, 3, 3, 3)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "convtranspose")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_transposed_convolution_is_not_a_convolution(tmp_path):
    """The two walk the same geometry in opposite directions, so neither may serve for the
    other: one kernel each, however alike their arguments look."""
    model = _model(
        [
            helper.make_node("Conv", ["x", "w"], ["p"], name="forward"),
            helper.make_node("ConvTranspose", ["x", "k"], ["q"], name="backward"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 5, 5)),
            _tensor("w", TensorProto.FLOAT, (2, 2, 3, 3)),
            _tensor("k", TensorProto.FLOAT, (2, 2, 3, 3)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, _ = _compile(model, tmp_path)

    assert len(_kernels(report, "conv")) == 2


def test_the_padding_a_same_mode_implies_reaches_the_kernel_as_literals(tmp_path):
    """`auto_pad` is resolved at compile time: the kernel only ever sees concrete pads.

    A transposed convolution pads so that the result measures `extent * stride`, cropping
    the reach the window has beyond it. A 3-tap window reaches two past a stride of 2 and
    nothing past a stride of 3, and SAME_LOWER puts the odd pad of the first at the front.
    """
    model = _transpose_model(
        (1, 1, 3, 3), (1, 1, 3, 3), auto_pad="SAME_LOWER", strides=[2, 3]
    )

    report, header = _compile(model, tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [1, 1, 6, 9]
    assert "(const ptrdiff_t[]){1, 0}" in header


def test_a_result_wider_than_the_window_reaches_is_not_padded(tmp_path):
    """`output_shape` may name a result the taps do not cover; the rest of it is bias alone.

    The pads are what crops the reach down to the result, so a result the reach falls short
    of needs none — the positions past it are simply ones no tap contributes to.
    """
    model = _transpose_model(
        (1, 1, 3, 3), (1, 2, 3, 3), strides=[3, 2], output_shape=[10, 8]
    )

    report, header = _compile(model, tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [1, 2, 10, 8]
    assert "(const ptrdiff_t[]){0, 0}" in header


def test_a_transposed_convolution_writing_no_elements_emits_no_call(tmp_path):
    report, header = _compile(_transpose_model((0, 2, 5, 5), (2, 3, 3, 3)), tmp_path)

    assert not _kernels(report, "convtranspose")
    assert "convtranspose" not in header.split(f"int {report['prefix']}_run(")[-1]


@requires_c_compiler
def test_a_grouped_transposed_convolution_is_its_groups_run_separately(tmp_path):
    """`group` splits both channel stacks into independent transposed convolutions.

    That is ONNX's definition of the attribute, and here it is also the only oracle for the
    general case: the reference evaluator's own grouped path slices `W` by output rather
    than input channels and hands every group the whole bias, so it can evaluate a grouped
    node only where each group holds exactly one channel of each — which is what the
    conformance corpus and the differential sweep are left covering. Splitting the operands
    here and running each group through the evaluator ungrouped puts the general case back
    within reach of the same oracle.
    """
    x_shape, w_shape, groups = (2, 4, 4, 3), (4, 3, 3, 2), 2
    attributes = {"strides": [2, 1], "pads": [1, 0, 0, 1], "dilations": [1, 2]}
    generator = np.random.default_rng(20260726)
    x = generator.normal(size=x_shape).astype(np.float32)
    w = generator.normal(size=w_shape).astype(np.float32)
    compiled = compile_onnx(
        _transpose_model(x_shape, w_shape, group=groups, **attributes), tmp_path
    ).load()

    got = compiled.run({"x": x, "w": w})["y"]

    channels, filters = x_shape[1] // groups, w_shape[1]
    expected = np.concatenate(
        [
            ReferenceEvaluator(
                _transpose_model(
                    (x_shape[0], channels, *x_shape[2:]),
                    (channels, filters, *w_shape[2:]),
                    **attributes,
                )
            ).run(
                None,
                {
                    "x": x[:, group * channels : (group + 1) * channels],
                    "w": w[group * channels : (group + 1) * channels],
                },
            )[0]
            for group in range(groups)
        ],
        axis=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"auto_pad": "SAME_UPPER", "pads": [1, 1, 1, 1]}, "mutually exclusive"),
        ({"kernel_shape": [2, 2]}, "the filter it is handed measures [3, 3]"),
        ({"auto_pad": "SAME"}, "is not one of the modes ONNX defines"),
        ({"group": 3}, "3 group(s) takes a filter"),
        ({"strides": [0, 1]}, "ONNX defines them as positive"),
        ({"output_padding": [-1, 0]}, "ONNX defines them as nonnegative"),
        ({"output_shape": [4]}, "was given 1 `output_shape` for 2 spatial axis/axes"),
        ({"pads": [1, 1]}, "was given 2 pad(s) for 2 spatial axis/axes"),
    ],
)
def test_a_transposed_convolution_the_compiler_cannot_place_is_rejected(
    tmp_path, attributes, message
):
    model = _transpose_model(
        (1, 2, 5, 5), (2, 2, 3, 3), output=(1, 2, 7, 7), **attributes
    )

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_a_transposed_filter_is_indexed_by_input_channel(tmp_path):
    """`W` is (C, M/group, ...) here, not Conv's (M, C/group, ...); a node that hands over
    the other layout is convolving something other than what it declares."""
    model = _transpose_model((1, 2, 5, 5), (3, 2, 3, 3), output=(1, 2, 7, 7))

    with pytest.raises(CompileError, match="one stack per input channel"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The deformable convolution
# --------------------------------------------------------------------------------------


def _deform_model(
    x_shape,
    w_shape,
    offset_shape,
    *,
    bias=None,
    mask=None,
    output=None,
    **attributes,
):
    operands = [("x", x_shape), ("w", w_shape), ("offset", offset_shape)]
    operands.append(("b" if bias else "", bias))
    if mask:
        operands.append(("mask", mask))
    names = [name for name, _ in operands]
    while names and not names[-1]:
        names.pop()
    return _model(
        [helper.make_node("DeformConv", names, ["y"], name="deform", **attributes)],
        [
            _tensor(name, TensorProto.FLOAT, shape)
            for name, shape in operands
            if name and shape
        ],
        [
            helper.make_empty_tensor_value_info("y")
            if output is None
            else _tensor("y", TensorProto.FLOAT, output)
        ],
    )


def test_a_deformable_convolution_emits_one_sampler_per_element_type(tmp_path):
    """The interpolation is a shared static of its own, so nodes reading the same element
    type share it however their geometries differ."""
    model = _model(
        [
            helper.make_node("DeformConv", ["x", "w", "o"], ["p"], name="small"),
            helper.make_node("DeformConv", ["x", "k", "n"], ["q"], name="wide"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 1, 4, 4)),
            _tensor("w", TensorProto.FLOAT, (1, 1, 2, 2)),
            _tensor("o", TensorProto.FLOAT, (1, 8, 3, 3)),
            _tensor("k", TensorProto.FLOAT, (2, 1, 3, 3)),
            _tensor("n", TensorProto.FLOAT, (1, 18, 2, 2)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, header = _compile(model, tmp_path)

    (sampler,) = _kernels(report, "bilinear")
    assert header.count(f"static float {sampler}(") == 1
    (kernel,) = _kernels(report, "deformconv")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_deformable_convolution_writing_no_elements_emits_no_call(tmp_path):
    """An empty batch leaves nothing to sample, and no sampler either."""
    report, header = _compile(
        _deform_model((0, 1, 4, 4), (1, 1, 2, 2), (0, 8, 3, 3)), tmp_path
    )

    assert not _kernels(report, "deformconv")
    assert not _kernels(report, "bilinear")
    assert "deformconv" not in header.split(f"int {report['prefix']}_run(")[-1]


@requires_c_compiler
def test_a_deformation_that_leaves_the_operand_samples_nothing(tmp_path):
    """A sampling point outside the operand contributes nothing — and one that is not a
    number is not a point at all, so it cannot be floored into an index.

    Neither is reachable through the suites that settle what this op computes: the corpus
    exercises ordinary offsets and the reference evaluator raises outright on a coordinate
    it cannot floor. They are asserted here because the artifact has to stay defined on
    whatever the caller passes, and what is left when every tap misses is the bias alone.
    """
    compiled = compile_onnx(
        _deform_model((1, 1, 4, 4), (1, 1, 2, 2), (1, 8, 3, 3), bias=(1,)), tmp_path
    ).load()
    operands = {
        "x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4),
        "w": np.ones((1, 1, 2, 2), dtype=np.float32),
        "b": np.array([0.5], dtype=np.float32),
    }
    bias = np.full((1, 1, 3, 3), 0.5, dtype=np.float32)

    for offset in (1e30, -1e30, np.nan):
        outside = compiled.run(
            {**operands, "offset": np.full((1, 8, 3, 3), offset, dtype=np.float32)}
        )["y"]

        np.testing.assert_array_equal(outside, bias)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset_shape": (1, 8, 4, 4)}, "addresses `offset` as [1, 8, 3, 3]"),
        ({"offset_shape": (1, 6, 3, 3)}, "addresses `offset` as [1, 8, 3, 3]"),
        (
            {"offset_shape": (1, 8, 3, 3), "mask": (1, 3, 3, 3)},
            "addresses `mask` as [1, 4, 3, 3]",
        ),
        ({"offset_shape": (1, 8, 3, 3), "bias": (2,)}, "one bias per output channel"),
        (
            {"offset_shape": (1, 16, 3, 3), "offset_group": 3},
            "ONNX defines `offset_group` as a positive count that divides them",
        ),
    ],
)
def test_a_deformation_shaped_for_another_geometry_is_rejected(
    tmp_path, kwargs, message
):
    """ONNX's own shape inference reads none of these operands, so the kernel checks them
    itself rather than addressing past the end of one."""
    model = _deform_model((1, 2, 4, 4), (1, 2, 2, 2), **kwargs)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_a_deformable_convolution_of_another_rank_is_rejected(tmp_path):
    """ONNX defines the op for any rank; nothing can vouch for a sampler of another one."""
    model = _deform_model((1, 1, 4, 4, 4), (1, 1, 2, 2, 2), (1, 24, 3, 3, 3))

    with pytest.raises(CompileError, match="compiled for 2 spatial axes"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The poolings
# --------------------------------------------------------------------------------------


def _pool_model(
    x_shape,
    *,
    op_type="AveragePool",
    elem_type=TensorProto.FLOAT,
    outputs=1,
    output=None,
    **attributes,
):
    names = ["y"] + [f"y{index}" for index in range(1, outputs)]
    declared = [
        helper.make_empty_tensor_value_info(name) if output is None else output
        for name in names
    ]
    return _model(
        [helper.make_node(op_type, ["x"], names, name="pool", **attributes)],
        [_tensor("x", elem_type, x_shape)],
        declared,
    )


def test_poolings_of_one_fold_and_element_type_share_a_kernel(tmp_path):
    """The geometry reaches the kernel as call-site literals, so neither rank nor a window
    the size of the whole operand is a kernel of its own: a 1-D average pooling, a 2-D one
    and a GlobalAveragePool all run the same code at different arguments."""
    model = _model(
        [
            helper.make_node(
                "AveragePool", ["x"], ["p"], name="signal", kernel_shape=[3]
            ),
            helper.make_node(
                "AveragePool", ["i"], ["q"], name="image", kernel_shape=[3, 3]
            ),
            helper.make_node("GlobalAveragePool", ["i"], ["r"], name="whole"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 6)),
            _tensor("i", TensorProto.FLOAT, (2, 4, 5, 5)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r")],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "pool")
    assert kernel.endswith("_pool_average_float")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 4


def test_a_global_pooling_takes_the_whole_spatial_extent_as_one_window(tmp_path):
    """One window per channel, covering every position, which is what ONNX defines it as.

    Six planes of twenty positions each, folded into one position by a window of twenty.
    """
    report, header = _compile(
        _pool_model((2, 3, 5, 4), op_type="GlobalAveragePool"), tmp_path
    )

    (kernel,) = _kernels(report, "pool")
    call = header.split(f"{kernel}(\n")[-1].splitlines()[:6]
    assert [line.strip(" ,);") for line in call] == ["y", "x", "6u", "20u", "1u", "20u"]
    assert report["entrypoint"]["outputs"][0]["shape"] == [2, 3, 1, 1]


def test_count_include_pad_is_a_kernel_argument_rather_than_a_kernel(tmp_path):
    """Two poolings differing only in what they divide by share one kernel, at 0u and 1u."""
    model = _model(
        [
            helper.make_node(
                "AveragePool", ["x"], ["p"], name="bare", kernel_shape=[3], pads=[1, 1]
            ),
            helper.make_node(
                "AveragePool",
                ["x"],
                ["q"],
                name="padded",
                kernel_shape=[3],
                pads=[1, 1],
                count_include_pad=1,
            ),
        ],
        [_tensor("x", TensorProto.FLOAT, (1, 2, 6))],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
    )

    report, header = _compile(model, tmp_path)

    assert len(_kernels(report, "pool")) == 1
    assert header.count("    0u);") == 1
    assert header.count("    1u);") == 1


def test_the_padding_a_dilated_same_mode_implies_reaches_the_kernel_as_literals(
    tmp_path,
):
    """`auto_pad` is resolved at compile time, for the window's *dilated* reach.

    A 2-tap window dilated to a reach of 4 over a 5-wide axis at stride 2 needs three pads,
    and SAME_LOWER puts the odd one at the beginning — two here against SAME_UPPER's one. The
    second axis takes an undilated 2-tap window at unit stride, which needs one.
    """
    model = _pool_model(
        (1, 1, 5, 5),
        kernel_shape=[2, 2],
        auto_pad="SAME_LOWER",
        strides=[2, 1],
        dilations=[3, 1],
    )

    _, header = _compile(model, tmp_path)

    assert "(const ptrdiff_t[]){2, 1}" in header


def test_a_valid_pooling_pads_nothing(tmp_path):
    """`VALID` is not `SAME`: it drops the positions a full window does not fit in.

    ONNX's own shape inference arbitrates this, and the dilated reach with it. It derives the
    result's shape from the pads the mode implies, and the kernel refuses to be emitted
    against a buffer its addressing disagrees with — so a compiler reading `VALID` as `SAME`,
    or measuring the window by its tap count rather than its reach, would fail to compile
    this model rather than quietly pool the wrong positions.
    """
    report, header = _compile(
        _pool_model(
            (1, 1, 7, 7), kernel_shape=[3, 3], auto_pad="VALID", dilations=[2, 1]
        ),
        tmp_path,
    )

    assert report["entrypoint"]["outputs"][0]["shape"] == [1, 1, 3, 5]
    assert "(const ptrdiff_t[]){0, 0}" in header


@pytest.mark.parametrize(
    ("extent", "attributes", "shape"),
    [
        # A 3-tap window at stride 2 fits a 4-wide axis one and a half times: rounding down
        # drops the half window, rounding up keeps it and reads the two taps of it that land
        # on the operand.
        (4, {"kernel_shape": [3, 3], "strides": [2, 2]}, [1, 3, 1, 1]),
        (4, {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1}, [1, 3, 2, 2]),
        # Rounding up here would put a second window's own start at 3, level with the end of
        # the padded operand, where it would cover nothing but pad: ONNX drops it again.
        (
            2,
            {
                "kernel_shape": [3, 3],
                "strides": [3, 3],
                "pads": [1, 1, 1, 1],
                "ceil_mode": 1,
            },
            [1, 3, 1, 1],
        ),
    ],
)
def test_ceil_mode_decides_how_many_windows_fit(tmp_path, extent, attributes, shape):
    """What the compiler counts is checked against what ONNX's shape inference counted."""
    report, _ = _compile(_pool_model((1, 3, extent, extent), **attributes), tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == shape


def test_a_pooling_writing_no_elements_emits_no_call(tmp_path):
    """An empty batch leaves nothing to pool, and no loop that could read past a buffer."""
    report, header = _compile(_pool_model((0, 2, 5, 5), kernel_shape=[3, 3]), tmp_path)

    assert not _kernels(report, "pool")
    assert "pool" not in header.split(f"int {report['prefix']}_run(")[-1]


def test_the_indexed_kernel_is_emitted_only_for_a_node_that_asks_for_indices(tmp_path):
    """Reporting where each maximum came from is a kernel of its own; the plain fold is not
    burdened with it, and the two do not share a name."""
    plain, _ = _compile(
        _pool_model((1, 2, 5, 5), op_type="MaxPool", kernel_shape=[2, 2]), tmp_path
    )
    indexed, _ = _compile(
        _pool_model((1, 2, 5, 5), op_type="MaxPool", kernel_shape=[2, 2], outputs=2),
        tmp_path / "indexed",
    )

    assert [name.split("_pool_")[-1] for name in _kernels(plain, "pool")] == [
        "max_float"
    ]
    assert [name.split("_pool_")[-1] for name in _kernels(indexed, "pool")] == [
        "max_indexed_float"
    ]


@pytest.mark.parametrize(
    ("storage_order", "strides", "rejected"),
    [
        (0, "(const size_t[]){5u, 1u}", "(const size_t[]){1u, 5u}"),
        (1, "(const size_t[]){1u, 5u}", "(const size_t[]){5u, 1u}"),
    ],
)
def test_storage_order_chooses_the_strides_the_indices_are_reported_in(
    tmp_path, storage_order, strides, rejected
):
    """ONNX defines the second output as one flat index per maximum, laid out row-major or
    column-major; both are the same walk over the operand at different strides."""
    model = _pool_model(
        (1, 2, 5, 5),
        op_type="MaxPool",
        kernel_shape=[2, 2],
        outputs=2,
        storage_order=storage_order,
    )

    _, header = _compile(model, tmp_path)

    assert strides in header
    assert rejected not in header


def _positions(*values):
    return np.array(values, np.int64).reshape(1, 1, 2, 2)


@requires_c_compiler
def test_an_unpooled_position_outside_the_result_returns_a_nonzero_status(tmp_path):
    """ONNX leaves an index past the end of the result undefined; the artifact reports it."""
    model = _model(
        [
            helper.make_node(
                "MaxUnpool",
                ["x", "i"],
                ["y"],
                name="unpool",
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 1, 2, 2)),
            _tensor("i", TensorProto.INT64, (1, 1, 2, 2)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )
    compiled = compile_onnx(model, tmp_path).load()
    x = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

    inside = compiled.run({"x": x, "i": _positions(0, 5, 10, 15)})["y"]
    with pytest.raises(HarnessError, match="status"):
        compiled.run({"x": x, "i": _positions(0, 5, 10, 16)})

    assert inside.reshape(-1).tolist() == [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        3,
    ]


@pytest.mark.parametrize(
    ("op_type", "attributes", "message"),
    [
        ("AveragePool", {}, "states no `kernel_shape`"),
        ("MaxPool", {}, "states no `kernel_shape`"),
        ("LpPool", {}, "states no `kernel_shape`"),
        (
            "AveragePool",
            {"kernel_shape": [2]},
            "was given 1 `kernel_shape` for 2 spatial axis/axes",
        ),
        (
            "AveragePool",
            {"kernel_shape": [2, 2], "auto_pad": "SAME_UPPER", "pads": [1, 1, 1, 1]},
            "mutually exclusive",
        ),
        (
            "AveragePool",
            {"kernel_shape": [2, 2], "auto_pad": "SAME"},
            "is not one of the modes ONNX defines",
        ),
        (
            "AveragePool",
            {"kernel_shape": [2, 2], "strides": [0, 1]},
            "ONNX defines them as positive",
        ),
        (
            "MaxPool",
            {"kernel_shape": [2, 2], "dilations": [1, 0]},
            "ONNX defines them as positive",
        ),
        (
            "AveragePool",
            {"kernel_shape": [2, 2], "pads": [1, 1]},
            "was given 2 pad(s) for 2 spatial axis/axes",
        ),
        ("LpPool", {"kernel_shape": [2, 2], "p": 0}, "which is positive"),
    ],
)
def test_a_pooling_the_compiler_cannot_place_is_rejected(
    tmp_path, op_type, attributes, message
):
    model = _pool_model(
        (1, 2, 5, 5),
        op_type=op_type,
        output=_tensor("y", TensorProto.FLOAT, (1, 2, 4, 4)),
        **attributes,
    )

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_a_pooling_of_a_tensor_with_no_channel_axis_is_rejected(tmp_path):
    model = _pool_model(
        (2, 5),
        kernel_shape=[2],
        output=_tensor("y", TensorProto.FLOAT, (2, 4)),
    )

    with pytest.raises(CompileError, match="rank 3 or more"):
        compile_onnx(model, tmp_path)


def test_unpooling_one_index_per_value_is_required(tmp_path):
    """ONNX's shape inference reads neither operand's extent against the other's."""
    model = _model(
        [
            helper.make_node(
                "MaxUnpool", ["x", "i"], ["y"], name="unpool", kernel_shape=[2, 2]
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 1, 2, 2)),
            _tensor("i", TensorProto.INT64, (1, 1, 3, 3)),
        ],
        [_tensor("y", TensorProto.FLOAT, (1, 1, 3, 3))],
    )

    with pytest.raises(CompileError, match="one index per value"):
        compile_onnx(model, tmp_path)


def test_an_unpooling_shaped_at_run_time_is_rejected(tmp_path):
    """`output_shape` is an operand, so a graph may compute it; one it does not fix makes the
    result's shape a function of input data, which no binding can make static."""
    model = _model(
        [
            helper.make_node(
                "MaxUnpool",
                ["x", "i", "s"],
                ["y"],
                name="unpool",
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 1, 2, 2)),
            _tensor("i", TensorProto.INT64, (1, 1, 2, 2)),
            _tensor("s", TensorProto.INT64, (4,)),
        ],
        [_tensor("y", TensorProto.FLOAT, (1, 1, 4, 4))],
    )

    with pytest.raises(CompileError, match="takes the shape of its `MaxUnpool` output"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The recurrent layers
# --------------------------------------------------------------------------------------

# What separates the three layers: how many gate rows the weights carry, and the two operands
# and third result only the LSTM's cell state brings. Everything else — the batch loop, the
# per-item sequence length, the padding past a sequence's end, the state outputs — is one
# shared frame, which is why most of what follows is parametrized over the family.
_LAYERS = ("LSTM", "GRU", "RNN")
_GATES = {"LSTM": 4, "GRU": 3, "RNN": 1}
_ALL_OPERANDS = ("X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P")
_ALL_RESULTS = ("Y", "Y_h", "Y_c")


def _layer_operands(op):
    return _ALL_OPERANDS if op == "LSTM" else _ALL_OPERANDS[:6]


def _layer_results(op):
    return _ALL_RESULTS if op == "LSTM" else _ALL_RESULTS[:2]


def _recurrent_feeds(
    op,
    *,
    hidden=3,
    seq=4,
    batch=2,
    inputs=3,
    direction="forward",
    optional=(),
    lengths=None,
    layout=0,
    seed=11,
):
    """Seeded operands for one recurrent node; `optional` names those past `X`, `W`, `R`."""
    generator = np.random.default_rng(seed)
    rows = _GATES[op] * hidden
    directions = 2 if direction == "bidirectional" else 1
    sequences = (batch, seq, inputs) if layout else (seq, batch, inputs)
    state = (batch, directions, hidden) if layout else (directions, batch, hidden)

    def draw(shape, scale=1.0):
        return (generator.normal(size=shape) * scale).astype(np.float32)

    feeds = {
        "X": draw(sequences),
        "W": draw((directions, rows, inputs), 0.3),
        "R": draw((directions, rows, hidden), 0.3),
    }
    available = {
        "B": lambda: draw((directions, 2 * rows), 0.2),
        "sequence_lens": lambda: np.array(lengths, dtype=np.int32),
        "initial_h": lambda: draw(state),
        "initial_c": lambda: draw(state),
        "P": lambda: draw((directions, 3 * hidden), 0.4),
    }
    for name in _layer_operands(op):
        if name in optional:
            feeds[name] = available[name]()
    return feeds


def _recurrent_model(op, feeds, *, outputs=None, hidden=3, **attributes):
    outputs = _layer_results(op) if outputs is None else outputs
    names = [name if name in feeds else "" for name in _layer_operands(op)]
    while names and not names[-1]:
        names.pop()
    node = helper.make_node(
        op,
        names,
        list(outputs),
        name=op.lower(),
        **{"hidden_size": hidden, **attributes},
    )
    return _model(
        [node],
        [
            _tensor(
                name,
                TensorProto.INT32 if name == "sequence_lens" else TensorProto.FLOAT,
                value.shape,
            )
            for name, value in feeds.items()
        ],
        [helper.make_empty_tensor_value_info(name) for name in outputs if name],
    )


def _recurrent_error_model(op, *, outputs=("Y",), shapes=None, **attributes):
    """A model whose every tensor is declared, so ONNX's own inference derives none of them.

    Which is what leaves the kernel's own checks reachable: a shape ONNX would have inferred,
    and refused to, is one the compiler has to refuse itself.
    """
    rows = 3 * _GATES[op]
    declared = {
        "X": (4, 2, 3),
        "W": (1, rows, 3),
        "R": (1, rows, 3),
        "B": (1, 2 * rows),
        "sequence_lens": (2,),
        "initial_h": (1, 2, 3),
        "initial_c": (1, 2, 3),
        "P": (1, 9),
    }
    operands = {name: declared[name] for name in _layer_operands(op)}
    operands.update(shapes or {})
    results = {"Y": (4, 1, 2, 3), "Y_h": (1, 2, 3), "Y_c": (1, 2, 3)}
    return _model(
        [
            helper.make_node(
                op,
                list(operands),
                list(outputs),
                name=op.lower(),
                **{"hidden_size": 3, **attributes},
            )
        ],
        [
            _tensor(
                name,
                TensorProto.INT32 if name == "sequence_lens" else TensorProto.FLOAT,
                shape,
            )
            for name, shape in operands.items()
        ],
        [_tensor(name, TensorProto.FLOAT, results[name]) for name in outputs],
    )


# Where a call site's argument list carries what, in the kernel's own parameter order: the
# operand pointers and scratch buffers, the four extents, the five strides that place them,
# and the flags the attributes become. The LSTM's pointers run five longer — a cell state to
# read, one to report and peephole weights, plus the buffer it carries the cell in — so
# everything after them sits further along its list.
_STRIDES = {"LSTM": slice(18, 23), "GRU": slice(14, 19), "RNN": slice(14, 19)}
_FLAGS = {"LSTM": slice(23, 27), "GRU": slice(19, 23), "RNN": slice(19, 22)}


def _recurrent_call(header, kernel, index=0):
    """One emitted call site's arguments, in order."""
    body = header.split(f"if ({kernel}(\n")[index + 1].split(") != 0)")[0]
    return [line.strip().rstrip(",") for line in body.splitlines()]


def _recurrent_kernel(report, op):
    """The op's own kernel, which the `rnnact_`/`rnnclip_` helpers must not be mistaken for."""
    (kernel,) = _kernels(report, f"{op.lower()}_float")
    return kernel


def _assert_matches_onnxruntime(tmp_path, model, feeds):
    """Run the compiled artifact and onnxruntime on the same node, and compare.

    The reference evaluator implements a slice of each recurrent op only -- one forward
    direction, with `clip` and `sequence_lens` ignored outright, and for the LSTM the
    activations too -- so the differential sweep covers that slice and everything else rests
    on the second oracle the compiler's parity test already stands on: onnxruntime, which is
    neither this compiler nor the evaluator. Nothing here states an expected value of its own.
    """
    runtime = pytest.importorskip("onnxruntime")
    runtime.set_default_logger_severity(3)
    session = runtime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = session.run(None, feeds)

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    for entry, want in zip(model.graph.output, expected):
        np.testing.assert_allclose(
            outputs[entry.name], want, rtol=1e-5, atol=1e-6, err_msg=entry.name
        )


@pytest.mark.parametrize("op", _LAYERS)
def test_recurrent_nodes_of_one_element_type_share_a_kernel(tmp_path, op):
    """Sequence length, batch, width and layout are call-site literals, not kernels."""
    rows = _GATES[op]
    model = _model(
        [
            helper.make_node(op, ["x", "w", "r"], ["y"], name="wide", hidden_size=3),
            helper.make_node(
                op, ["s", "v", "q"], ["z"], name="narrow", hidden_size=2, layout=1
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (4, 2, 3)),
            _tensor("w", TensorProto.FLOAT, (1, 3 * rows, 3)),
            _tensor("r", TensorProto.FLOAT, (1, 3 * rows, 3)),
            _tensor("s", TensorProto.FLOAT, (2, 5, 1)),
            _tensor("v", TensorProto.FLOAT, (1, 2 * rows, 1)),
            _tensor("q", TensorProto.FLOAT, (1, 2 * rows, 2)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z")],
    )

    report, header = _compile(model, tmp_path)

    kernel = _recurrent_kernel(report, op)
    assert header.count(f"static int {kernel}(") == 1
    assert header.count(f"if ({kernel}(") == 2
    # The wider node's state decides the shared scratch; the narrower one reuses it.
    symbol = f"{report['prefix']}_{op.lower()}"
    assert f"static float {symbol}_hidden_float[3];" in header
    assert f"static float {symbol}_gates_float[{3 * rows}];" in header


@pytest.mark.parametrize(
    ("op", "weights"),
    [
        ("LSTM", ("W + 48", "R + 36")),
        ("GRU", ("W + 36", "R + 27")),
        ("RNN", ("W + 12", "R + 9")),
    ],
)
def test_a_bidirectional_layer_is_its_two_passes_over_one_kernel(tmp_path, op, weights):
    """One direction per call, differing only in what the second direction reads and writes.

    Its own half of the weights, its own slice of the results — one direction's worth of
    hidden units into each — and the flag that walks time backwards; everything else is the
    same call.
    """
    feeds = _recurrent_feeds(
        op, direction="bidirectional", hidden=3, seq=4, batch=2, inputs=4
    )
    model = _recurrent_model(op, feeds, outputs=("Y", "Y_h"), direction="bidirectional")

    report, header = _compile(model, tmp_path)

    kernel = _recurrent_kernel(report, op)
    assert header.count(f"if ({kernel}(") == 2
    forward = _recurrent_call(header, kernel)
    backward = _recurrent_call(header, kernel, 1)
    assert len(forward) == len(backward)
    assert [pair for pair in zip(forward, backward) if pair[0] != pair[1]] == [
        ("Y", "Y + 6"),
        ("Y_h", "Y_h + 6"),
        ("W", weights[0]),
        ("R", weights[1]),
        ("0", "1"),
    ]


@pytest.mark.parametrize("op", _LAYERS)
def test_the_layout_reaches_the_kernel_as_strides(tmp_path, op):
    """Layout 1 packs the batch outermost, which changes only how far apart two steps are."""
    emissions = {}
    for layout in (0, 1):
        feeds = _recurrent_feeds(op, hidden=3, seq=4, batch=2, inputs=3, layout=layout)
        emissions[layout] = _compile(
            _recurrent_model(op, feeds, layout=layout), tmp_path / str(layout)
        )

    # Time first: one step of a sequence is a whole batch of input rows away (2 * 3) and one
    # batch item is a single row (3). Batch first: a step is one row away and an item a whole
    # sequence of them (4 * 3). The results follow the same reordering.
    kernel = _recurrent_kernel(emissions[0][0], op)
    assert _recurrent_kernel(emissions[1][0], op) == kernel
    strides = {
        layout: _recurrent_call(header, kernel)[_STRIDES[op]]
        for layout, (_, header) in emissions.items()
    }
    assert strides[0] == ["6u", "3u", "6u", "3u", "3u"]
    assert strides[1] == ["3u", "12u", "3u", "12u", "3u"]


@pytest.mark.parametrize(
    ("op", "attributes", "flags"),
    [
        ("LSTM", {}, ["0", "0", "0", "0.0f"]),
        ("LSTM", {"direction": "reverse"}, ["1", "0", "0", "0.0f"]),
        ("LSTM", {"input_forget": 1}, ["0", "1", "0", "0.0f"]),
        ("LSTM", {"clip": 0.5}, ["0", "0", "1", "0.5f"]),
        ("GRU", {}, ["0", "0", "0", "0.0f"]),
        ("GRU", {"direction": "reverse"}, ["1", "0", "0", "0.0f"]),
        ("GRU", {"linear_before_reset": 1}, ["0", "1", "0", "0.0f"]),
        ("GRU", {"clip": 0.5}, ["0", "0", "1", "0.5f"]),
        # An RNN has no second mode to switch, so its flags are the direction and the clip.
        ("RNN", {}, ["0", "0", "0.0f"]),
        ("RNN", {"direction": "reverse"}, ["1", "0", "0.0f"]),
        ("RNN", {"clip": 0.5}, ["0", "1", "0.5f"]),
    ],
)
def test_the_attributes_that_only_switch_a_branch_are_kernel_arguments(
    tmp_path, op, attributes, flags
):
    """Direction, the op's own mode and the cell clip pick a branch: one kernel serves all."""
    feeds = _recurrent_feeds(op)
    report, header = _compile(_recurrent_model(op, feeds, **attributes), tmp_path)

    kernel = _recurrent_kernel(report, op)
    assert _recurrent_call(header, kernel)[_FLAGS[op]] == flags


@pytest.mark.parametrize("op", _LAYERS)
def test_an_output_the_node_drops_is_never_written(tmp_path, op):
    """The optional results are pointers the kernel checks, so a dropped one costs no buffer."""
    feeds = _recurrent_feeds(op)
    report, header = _compile(
        _recurrent_model(op, feeds, outputs=("", "Y_h")), tmp_path
    )

    kernel = _recurrent_kernel(report, op)
    dropped = ["NULL", "Y_h"] + ["NULL"] * (len(_layer_results(op)) - 2)
    assert _recurrent_call(header, kernel)[: len(dropped)] == dropped
    assert [entry["name"] for entry in report["entrypoint"]["outputs"]] == ["Y_h"]


@pytest.mark.parametrize(
    ("op", "chosen", "dropped"),
    [
        ("LSTM", ["Relu", "Softsign", "Tanh"], "sigmoid"),
        ("GRU", ["Relu", "Softsign"], "sigmoid"),
        # An RNN runs `Tanh` by default and nothing else, so naming another drops it too.
        ("RNN", ["Relu"], "tanh"),
    ],
)
def test_a_recurrent_layer_emits_only_the_activations_it_names(
    tmp_path, op, chosen, dropped
):
    """One function per activation, shared by the gates that run it and by other nodes."""
    feeds = _recurrent_feeds(op)
    report, default = _compile(_recurrent_model(op, feeds), tmp_path / "default")
    prefix = report["prefix"]
    picked = _compile(
        _recurrent_model(op, feeds, activations=chosen), tmp_path / "chosen"
    )[1]

    assert f"static float {prefix}_rnnact_{dropped}_float(" in default
    assert f"{prefix}_rnnact_relu_float" not in default
    assert f"static float {prefix}_rnnact_relu_float(" in picked
    assert f"{prefix}_rnnact_{dropped}_float" not in picked


@pytest.mark.parametrize(
    ("op", "case", "feeds", "attributes"),
    [
        (
            "LSTM",
            "reverse",
            {"optional": ("B", "initial_h", "initial_c", "P")},
            {"direction": "reverse"},
        ),
        (
            "LSTM",
            "bidirectional",
            {
                "direction": "bidirectional",
                "optional": ("B", "initial_h", "initial_c", "P"),
            },
            {"direction": "bidirectional"},
        ),
        ("LSTM", "clip", {"optional": ("B",)}, {"clip": 0.4}),
        # The clip bounds a gate's whole pre-activation, its peephole term included, and the
        # output gate's after the cell that term reads has been updated.
        ("LSTM", "clip_with_peepholes", {"optional": ("B", "P")}, {"clip": 0.4}),
        ("LSTM", "input_forget", {"optional": ("B", "P")}, {"input_forget": 1}),
        (
            "LSTM",
            "activations",
            {"optional": ("B",)},
            {"activations": ["Relu", "Softsign", "HardSigmoid"]},
        ),
        (
            "LSTM",
            "parameterized_activations",
            {"optional": ("B",)},
            {
                "activations": ["LeakyRelu", "LeakyRelu", "LeakyRelu"],
                "activation_alpha": [0.3, 0.4, 0.5],
            },
        ),
        # `activation_alpha` and `activation_beta` carry a value for the activations that
        # take one and for no others, so `Sigmoid` here consumes neither: the `LeakyRelu`
        # reads the first alpha and the `HardSigmoid` the second, along with the only beta.
        (
            "LSTM",
            "activation_parameters_consumed_where_they_are_taken",
            {"optional": ("B",)},
            {
                "activations": ["Sigmoid", "LeakyRelu", "HardSigmoid"],
                "activation_alpha": [0.3, 0.9],
                "activation_beta": [0.7],
            },
        ),
        # And they are consumed over both directions together, not restarted per direction.
        (
            "LSTM",
            "activation_parameters_across_directions",
            {"direction": "bidirectional", "optional": ("B",)},
            {
                "direction": "bidirectional",
                "activations": ["LeakyRelu", "Tanh", "Tanh"] * 2,
                "activation_alpha": [0.3, 0.8],
            },
        ),
        (
            "LSTM",
            "short_sequences",
            {
                "optional": ("B", "sequence_lens", "initial_h", "initial_c"),
                "lengths": [4, 2],
            },
            {},
        ),
        (
            "LSTM",
            "empty_sequence",
            {
                "optional": ("B", "sequence_lens", "initial_h", "initial_c"),
                "lengths": [3, 0],
            },
            {},
        ),
        (
            "LSTM",
            "backwards_over_short_sequences",
            {"optional": ("B", "sequence_lens"), "lengths": [1, 3]},
            {"direction": "reverse"},
        ),
        (
            "GRU",
            "reverse",
            {"optional": ("B", "initial_h")},
            {"direction": "reverse"},
        ),
        (
            "GRU",
            "bidirectional",
            {"direction": "bidirectional", "optional": ("B", "initial_h")},
            {"direction": "bidirectional"},
        ),
        ("GRU", "clip", {"optional": ("B",)}, {"clip": 0.4}),
        # `linear_before_reset` moves the candidate's recurrent bias inside the term the
        # reset gate scales, so the two branches differ only where a bias is present — and
        # differ in what the reset gate multiplies whether one is or not.
        (
            "GRU",
            "linear_before_reset",
            {"optional": ("B",)},
            {"linear_before_reset": 1},
        ),
        ("GRU", "linear_before_reset_unbiased", {}, {"linear_before_reset": 1}),
        (
            "GRU",
            "linear_before_reset_clipped",
            {"optional": ("B",)},
            {"linear_before_reset": 1, "clip": 0.3},
        ),
        (
            "GRU",
            "activations",
            {"optional": ("B",)},
            {"activations": ["Relu", "Softsign"]},
        ),
        (
            "GRU",
            "parameterized_activations",
            {"optional": ("B",)},
            {
                "activations": ["LeakyRelu", "HardSigmoid"],
                "activation_alpha": [0.3, 0.9],
                "activation_beta": [0.7],
            },
        ),
        (
            "GRU",
            "activation_parameters_across_directions",
            {"direction": "bidirectional", "optional": ("B",)},
            {
                "direction": "bidirectional",
                "activations": ["LeakyRelu", "Tanh"] * 2,
                "activation_alpha": [0.3, 0.8],
            },
        ),
        (
            "GRU",
            "short_sequences",
            {"optional": ("B", "sequence_lens", "initial_h"), "lengths": [4, 2]},
            {},
        ),
        (
            "GRU",
            "empty_sequence",
            {"optional": ("B", "sequence_lens", "initial_h"), "lengths": [3, 0]},
            {},
        ),
        (
            "GRU",
            "backwards_over_short_sequences",
            {"optional": ("B", "sequence_lens"), "lengths": [1, 3]},
            {"direction": "reverse"},
        ),
        (
            "RNN",
            "reverse",
            {"optional": ("B", "initial_h")},
            {"direction": "reverse"},
        ),
        (
            "RNN",
            "bidirectional",
            {"direction": "bidirectional", "optional": ("B", "initial_h")},
            {"direction": "bidirectional"},
        ),
        ("RNN", "clip", {"optional": ("B",)}, {"clip": 0.4}),
        ("RNN", "activations", {"optional": ("B",)}, {"activations": ["Relu"]}),
        (
            "RNN",
            "parameterized_activations",
            {"optional": ("B",)},
            {
                "activations": ["HardSigmoid"],
                "activation_alpha": [0.3],
                "activation_beta": [0.7],
            },
        ),
        # onnxruntime reads this attribute per direction rather than per parameterized
        # activation, and walks off the end of its own vector when given fewer values than
        # it has directions -- so the case that separates the two readings is the LSTM's.
        (
            "RNN",
            "activation_parameters_across_directions",
            {"direction": "bidirectional", "optional": ("B",)},
            {
                "direction": "bidirectional",
                "activations": ["LeakyRelu", "LeakyRelu"],
                "activation_alpha": [0.3, 0.8],
            },
        ),
        (
            "RNN",
            "short_sequences",
            {"optional": ("B", "sequence_lens", "initial_h"), "lengths": [4, 2]},
            {},
        ),
        (
            "RNN",
            "backwards_over_short_sequences",
            {"optional": ("B", "sequence_lens"), "lengths": [1, 3]},
            {"direction": "reverse"},
        ),
    ],
)
@requires_c_compiler
def test_the_recurrent_surface_the_evaluator_cannot_vouch_for(
    tmp_path, op, case, feeds, attributes
):
    """Every attribute ONNX's reference evaluator drops, against onnxruntime instead.

    An empty sequence is asked of the LSTM and the GRU only: onnxruntime leaves an RNN's
    state for a zero-length item uninitialized, so there is nothing there to compare against.
    """
    operands = _recurrent_feeds(op, **feeds)

    _assert_matches_onnxruntime(
        tmp_path, _recurrent_model(op, operands, **attributes), operands
    )


@requires_c_compiler
def test_the_lstm_cell_state_output_matches_onnxruntime(tmp_path):
    """`Y_c` is the third output, which the reference evaluator never returns at all."""
    feeds = _recurrent_feeds("LSTM", optional=("B", "initial_h", "initial_c", "P"))

    _assert_matches_onnxruntime(
        tmp_path, _recurrent_model("LSTM", feeds, outputs=("", "", "Y_c")), feeds
    )


@pytest.mark.parametrize("op", _LAYERS)
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
@requires_c_compiler
def test_a_batchwise_layer_runs_what_the_time_first_one_does(tmp_path, op, direction):
    """Layout 1 over several steps, against the layout its two oracles cover.

    onnxruntime refuses `layout` 1 outright and the reference evaluator's own layout path
    holds for a single step only, which leaves the corpus's one batchwise test per op -- also
    a single step -- as the whole of the direct evidence. But layout 1 means only that the
    operands are packed batch-outermost, so transposing them onto a time-first node of the
    same layer has to reproduce the run down to the bit: same arithmetic in the same order,
    reaching the same elements from elsewhere in memory.
    """
    packed = ("X", "initial_h", "initial_c")
    optional = ("B", "sequence_lens", "initial_h", "initial_c", "P")
    flat = _recurrent_feeds(op, direction=direction, optional=optional, lengths=[4, 2])
    batchwise = {
        name: value.transpose(1, 0, 2).copy() if name in packed else value
        for name, value in flat.items()
    }

    time_first = (
        compile_onnx(_recurrent_model(op, flat, direction=direction), tmp_path / "time")
        .load()
        .run(flat)
    )
    batch_first = (
        compile_onnx(
            _recurrent_model(op, batchwise, direction=direction, layout=1),
            tmp_path / "batch",
        )
        .load()
        .run(batchwise)
    )

    assert np.array_equal(batch_first["Y"], time_first["Y"].transpose(2, 0, 1, 3))
    for state in _layer_results(op)[1:]:
        assert np.array_equal(batch_first[state], time_first[state].transpose(1, 0, 2))


@pytest.mark.parametrize("op", _LAYERS)
@requires_c_compiler
def test_a_sequence_length_past_the_padded_end_returns_a_nonzero_status(tmp_path, op):
    """A length names a step of the padded sequence; anything else names one that is absent."""
    feeds = _recurrent_feeds(op, optional=("sequence_lens",), lengths=[4, 4])
    compiled = compile_onnx(_recurrent_model(op, feeds), tmp_path).load()

    for lengths in ([4, 5], [-1, 2]):
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run({**feeds, "sequence_lens": np.array(lengths, dtype=np.int32)})


@pytest.mark.parametrize("op", _LAYERS)
@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"direction": "backward"}, "not one of the directions ONNX defines"),
        ({"layout": 2}, "ONNX defines only 0 (time first) and 1 (batch first)"),
        ({"clip": -1.0}, "which is not negative"),
        ({"hidden_size": 4}, "states a `hidden_size` of 4"),
    ],
)
def test_a_recurrent_node_the_compiler_cannot_place_is_rejected(
    tmp_path, op, attributes, message
):
    model = _recurrent_error_model(op, **attributes)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("op", "attributes", "message"),
    [
        (
            "LSTM",
            {"activations": ["Sigmoid", "Tanh"]},
            "runs 3 activations — 3 per direction — but this node names 2",
        ),
        (
            "GRU",
            {"activations": ["Sigmoid", "Tanh", "Tanh"]},
            "runs 2 activations — 2 per direction — but this node names 3",
        ),
        (
            "RNN",
            {"activations": ["Tanh", "Tanh"]},
            "runs 1 activation — 1 per direction — but this node names 2",
        ),
        (
            "RNN",
            {"direction": "bidirectional", "activations": ["Tanh"]},
            "runs 2 activations — 1 per direction — but this node names 1",
        ),
        ("RNN", {"activations": ["Swish"]}, "names the activation `Swish`"),
        (
            "GRU",
            {"activations": ["Affine", "Tanh"]},
            "there is no default `activation_alpha`",
        ),
        (
            "RNN",
            {"activations": ["ScaledTanh"], "activation_alpha": [2.0]},
            "there is no default `activation_beta`",
        ),
    ],
)
def test_an_activation_a_layer_cannot_run_is_rejected(
    tmp_path, op, attributes, message
):
    """How many activations an op runs is the op's own, and so is what it may name."""
    model = _recurrent_error_model(op, **attributes)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("op", "name", "shape", "message"),
    [
        ("LSTM", "X", (4, 2), "a tensor of rank 3"),
        ("LSTM", "R", (12, 3), "recurrence weights as rank 3"),
        ("LSTM", "W", (1, 12, 2), "reads `W` as its input weights of shape [1, 12, 3]"),
        ("LSTM", "B", (1, 12), "reads `B` as its biases of shape [1, 24]"),
        ("LSTM", "P", (1, 12), "reads `P` as its peephole weights of shape [1, 9]"),
        (
            "LSTM",
            "initial_h",
            (1, 3, 3),
            "as an initial hidden state of shape [1, 2, 3]",
        ),
        ("LSTM", "sequence_lens", (3,), "as its sequence lengths of shape [2]"),
        (
            "GRU",
            "W",
            (1, 12, 3),
            "`GRU` reads `W` as its input weights of shape [1, 9, 3]",
        ),
        ("GRU", "B", (1, 24), "`GRU` reads `B` as its biases of shape [1, 18]"),
        (
            "GRU",
            "initial_h",
            (1, 3, 3),
            "as an initial hidden state of shape [1, 2, 3]",
        ),
        (
            "RNN",
            "W",
            (1, 12, 3),
            "`RNN` reads `W` as its input weights of shape [1, 3, 3]",
        ),
        ("RNN", "B", (1, 24), "`RNN` reads `B` as its biases of shape [1, 6]"),
        ("RNN", "X", (4, 2), "a tensor of rank 3"),
    ],
)
def test_a_recurrent_operand_shaped_for_another_layer_is_rejected(
    tmp_path, op, name, shape, message
):
    """Declared shapes get past ONNX's own inference, so the kernel checks them itself."""
    model = _recurrent_error_model(op, shapes={name: shape})

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize("op", _LAYERS)
@pytest.mark.parametrize("output", ["Y", "Y_h"])
def test_a_recurrent_result_shaped_for_another_layer_is_rejected(tmp_path, op, output):
    """The same, for the buffers it writes: a declared result is checked, never trusted."""
    model = _recurrent_error_model(op, outputs=(output,))
    entry = model.graph.output[0]
    entry.type.tensor_type.shape.dim[0].dim_value = 7

    with pytest.raises(CompileError, match="addresses a result of shape"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The resizes
# --------------------------------------------------------------------------------------

# What a resize reads its geometry from — the scales, the sizes and the region of interest —
# is passed as operands rather than attributes, so every test below has to say where each of
# them comes from. Carried in the model, they let ONNX's own shape inference derive the
# result; fed at run time, they leave the declared result shape as the only static one, which
# is the form the backend corpus's own Resize tests take.
_RESIZE_OPSET = 19

_RESIZE_OPERANDS = (
    ("roi", TensorProto.FLOAT),
    ("scales", TensorProto.FLOAT),
    ("sizes", TensorProto.INT64),
)


def _resize_model(
    x_shape,
    *,
    op_type="Resize",
    elem_type=TensorProto.FLOAT,
    roi=None,
    scales=None,
    sizes=None,
    runtime=(),
    output=None,
    opset=_RESIZE_OPSET,
    name="resize",
    **attributes,
):
    """One resize node, its operands carried in the model unless `runtime` names them."""
    values = {"roi": roi, "scales": scales, "sizes": sizes}
    inputs = [_tensor("x", elem_type, x_shape)]
    initializer = []
    names = ["x"]
    for operand, operand_type in _RESIZE_OPERANDS:
        given = values[operand]
        names.append("" if given is None else operand)
        if given is None:
            continue
        array = np.array(given, dtype=helper.tensor_dtype_to_np_dtype(operand_type))
        if operand in runtime:
            inputs.append(_tensor(operand, operand_type, array.shape))
        else:
            initializer.append(onnx.numpy_helper.from_array(array, operand))
    if op_type == "Upsample":
        names = ["x", "scales"]
    while names and not names[-1]:
        names.pop()
    declared = (
        helper.make_empty_tensor_value_info("y")
        if output is None
        else _tensor("y", elem_type, output)
    )
    return _model(
        [helper.make_node(op_type, names, ["y"], name=name, **attributes)],
        inputs,
        [declared],
        initializer=initializer,
        opset=opset,
    )


def test_resizes_of_one_element_type_share_a_kernel(tmp_path):
    """Every setting a resize reads is a kernel argument, not a kernel of its own.

    Three nodes interpolating differently, over different ranks and in both directions, run
    the same code at different literals — the mode, the mapping and the rounding rule among
    them.
    """
    model = _model(
        [
            helper.make_node(
                "Resize", ["x", "", "up"], ["p"], name="grown", mode="linear"
            ),
            helper.make_node(
                "Resize",
                ["x", "", "down"],
                ["q"],
                name="shrunk",
                mode="cubic",
                antialias=1,
            ),
            helper.make_node(
                "Resize",
                ["s", "", "signal"],
                ["r"],
                name="signal",
                mode="nearest",
                nearest_mode="ceil",
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 4, 5)),
            _tensor("s", TensorProto.FLOAT, (2, 3, 7)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r")],
        initializer=[
            onnx.numpy_helper.from_array(np.array(values, dtype=np.float32), name)
            for name, values in (
                ("up", [1.0, 1.0, 2.0, 3.0]),
                ("down", [1.0, 1.0, 0.5, 0.5]),
                ("signal", [1.0, 1.0, 1.5]),
            )
        ],
        opset=_RESIZE_OPSET,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "resize_float")
    assert kernel.endswith("_resize_float_float")
    assert header.count(f"static int {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 4


def test_the_geometry_reaches_the_kernel_as_call_site_literals(tmp_path):
    """The shapes, the axes and the settings are compile-time constants at the call site."""
    model = _resize_model(
        (1, 2, 4, 5),
        scales=[0.5, 2.0],
        axes=[3, 2],
        mode="cubic",
        coordinate_transformation_mode="align_corners",
        exclude_outside=1,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "resize_float")
    call = header.split(f"{kernel}(\n")[-1].splitlines()
    assert [line.strip(" ,);") for line in call[:14]] == [
        "y",
        "x",
        "NULL",
        f"{report['prefix']}_w_scales",
        "NULL",
        f"{report['prefix']}_resize_work",
        f"{report['prefix']}_resize_spare",
        "40u",
        "32u",
        "4",
        "(const size_t[]){1u, 2u, 4u, 5u}",
        "(const size_t[]){1u, 2u, 8u, 2u}",
        "2",
        "(const size_t[]){3u, 2u}",
    ]
    # mode `cubic`, the default rounding, `align_corners`, no antialias, exclude outside.
    assert [line.strip(" ,);") for line in call[14:20]] == [
        "2",
        "0",
        "3",
        "0",
        "1",
        "0",
    ]


def test_an_upsample_is_the_resize_its_successor_defines_it_to_be(tmp_path):
    """ONNX deprecated Upsample in favour of Resize's asymmetric mapping at the floor.

    Compiling the two ops on the same operands emits the same call to the same kernel — the
    deprecated op is not a walk of its own, it is that one at the settings its successor's
    specification spells out for it.
    """
    emissions = [
        _compile(
            _resize_model(
                (1, 1, 4, 4),
                op_type=op_type,
                scales=[1.0, 1.0, 2.0, 1.5],
                opset=opset,
                **attributes,
            ),
            tmp_path / op_type,
        )
        for op_type, opset, attributes in (
            ("Upsample", 9, {}),
            (
                "Resize",
                _RESIZE_OPSET,
                {
                    "mode": "nearest",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            ),
        )
    ]

    (kernel,) = _kernels(emissions[0][0], "resize_float")
    assert _kernels(emissions[1][0], "resize_float") == [kernel]
    settings = [
        [
            line.strip(" ,);")
            for line in header.split(f"{kernel}(\n")[-1].splitlines()[14:20]
        ]
        for _, header in emissions
    ]
    # `nearest` at the floor of an asymmetrically mapped coordinate, no antialias, no
    # exclusion and no aspect-ratio policy.
    assert settings[0] == ["0", "2", "4", "0", "0", "0"]
    assert settings[1] == settings[0]


def test_the_working_buffers_are_static_and_sized_for_the_widest_pass(tmp_path):
    """A pass reads the whole result of the one before it, so two buffers are reserved.

    They are shared by every resize in the model and sized for the largest — each axis at
    the larger of the two extents it carries — which is what the reported footprint holds.
    """
    model = _model(
        [
            helper.make_node("Resize", ["x", "", "up"], ["p"], name="grown"),
            helper.make_node("Resize", ["x", "", "down"], ["q"], name="shrunk"),
        ],
        [_tensor("x", TensorProto.FLOAT, (1, 2, 4, 5))],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q")],
        initializer=[
            onnx.numpy_helper.from_array(np.array(values, dtype=np.float32), name)
            for name, values in (
                ("up", [1.0, 1.0, 3.0, 2.0]),
                ("down", [1.0, 1.0, 0.5, 0.5]),
            )
        ],
        opset=_RESIZE_OPSET,
    )

    report, header = _compile(model, tmp_path)

    widest = 1 * 2 * 12 * 10
    for role in ("work", "spare"):
        assert f"static double {report['prefix']}_resize_{role}[{widest}];" in header
    # The two buffers, plus the results of the two nodes and the embedded scales.
    assert report["memory"]["arena_bytes"] >= 2 * widest * 8


@requires_c_compiler
def test_a_run_time_scale_the_artifact_was_not_compiled_for_is_reported(tmp_path):
    """The result's extent follows from operand *values*, which only the caller has.

    Compiled against the shape the model declares, the artifact still derives the extents
    the operands ask for and refuses, through the status enum, to write a result of any
    other shape — so a scale it was not compiled for is an error rather than a buffer
    written past.
    """
    model = _resize_model(
        (1, 1, 2, 2),
        scales=[1.0, 1.0, 2.0, 3.0],
        runtime=("scales",),
        output=(1, 1, 4, 6),
        mode="nearest",
    )
    values = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

    compiled = compile_onnx(model, tmp_path).load()

    compiled_output = compiled.run(
        {"x": values, "scales": np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)}
    )["y"]
    expected = ReferenceEvaluator(model).run(
        None, {"x": values, "scales": np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)}
    )[0]
    np.testing.assert_array_equal(compiled_output, expected)

    with pytest.raises(HarnessError, match="status"):
        compiled.run(
            {"x": values, "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)}
        )


@requires_c_compiler
def test_a_resize_writing_nothing_emits_no_kernel_at_all(tmp_path):
    """A scale that shrinks an axis to nothing leaves a result with no elements in it."""
    model = _resize_model((1, 2, 4, 5), scales=[1.0, 1.0, 0.2, 1.0], mode="linear")

    report, header = _compile(model, tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [1, 2, 0, 5]
    assert not _kernels(report, "resize")
    assert f"{report['prefix']}_resize" not in header


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of `scales` and `sizes`"),
        # An operand holding nothing at all is one left out, which is how an exporter
        # passes `sizes` without leaving the `scales` position blank.
        ({"scales": []}, "exactly one of `scales` and `sizes`"),
        (
            {"scales": [1.0, 1.0, 2.0, 2.0], "sizes": [1, 2, 8, 10]},
            "exactly one of `scales` and `sizes`",
        ),
        ({"scales": [1.0, 2.0]}, "takes `scales` as 4 value(s)"),
        (
            {
                "scales": [1.0, 1.0, 2.0, 2.0],
                "roi": [0.0, 1.0],
                "coordinate_transformation_mode": "tf_crop_and_resize",
            },
            "takes `roi` as 8 value(s)",
        ),
        (
            {"scales": [1.0, 1.0, 2.0, 2.0], "mode": "bilinear"},
            "asks for `mode` `bilinear`",
        ),
        (
            {
                "scales": [1.0, 1.0, 2.0, 2.0],
                "coordinate_transformation_mode": "corners",
            },
            "asks for `coordinate_transformation_mode` `corners`",
        ),
        (
            {"scales": [1.0, 1.0, 2.0, 2.0], "nearest_mode": "nearest"},
            "asks for `nearest_mode` `nearest`",
        ),
        (
            {"sizes": [1, 2, 8, 10], "keep_aspect_ratio_policy": "squash"},
            "asks for `keep_aspect_ratio_policy` `squash`",
        ),
        (
            {"scales": [1.0, 1.0, 2.0, 2.0], "antialias": 1},
            "`antialias` in `nearest` mode",
        ),
        (
            {"scales": [2.0, 2.0], "axes": [2, 2]},
            "names the same dimension more than once",
        ),
        ({"scales": [2.0, 2.0], "axes": [2, 4]}, "axis 4 is out of range"),
    ],
)
def test_a_resize_the_compiler_cannot_serve_is_rejected(tmp_path, kwargs, message):
    """Everything about a resize that its operands' shapes and its attributes settle.

    The operands are fed at run time here, which is what leaves these to the compiler at
    all: given their values, ONNX's own inference rejects most of these models first.
    """
    model = _resize_model(
        (1, 2, 4, 5), runtime=("roi", "scales", "sizes"), output=(1, 2, 8, 10), **kwargs
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`resize`" in str(error.value)
    assert message in str(error.value)


def test_a_resize_of_a_boolean_tensor_is_rejected(tmp_path):
    """ONNX allows one; there is no value between two truth values to interpolate to."""
    model = _resize_model(
        (1, 2, 4, 5),
        elem_type=TensorProto.BOOL,
        scales=[1.0, 1.0, 2.0, 2.0],
        mode="nearest",
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`resize`" in str(error.value)
    assert "BOOL" in str(error.value)


@pytest.mark.parametrize(
    ("x_shape", "output", "message"),
    [
        ((1, 2, 4, 5), (2, 2, 8, 10), "does not resize axis 0"),
        ((1, 2, 0, 5), (1, 2, 3, 10), "which holds no elements"),
    ],
)
def test_a_declared_result_the_axes_do_not_allow_is_rejected(
    tmp_path, x_shape, output, message
):
    """A result shape the model declares rather than computes is checked, never trusted.

    With the scales fed at run time, nothing but the declaration says what shape the result
    has — so the axes the node does not resize have to carry the operand's own extents, and
    an axis holding nothing has nothing to interpolate into one that does.
    """
    model = _resize_model(
        x_shape,
        scales=[2.0, 2.0],
        axes=[2, 3],
        runtime=("scales",),
        output=output,
        mode="nearest",
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`resize`" in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    ("op_type", "opset", "nearest"),
    [("Resize", 18, 19), ("Upsample", 7, 9)],
)
def test_a_resize_below_its_supported_revision_names_the_nearest_version(
    tmp_path, op_type, opset, nearest
):
    """Resize was revised at 19 and Upsample took its scales as an attribute up to 7.

    Neither older revision has an oracle — the reference evaluator applies today's
    implementation to them and the backend corpus has no test at one — so no kernel claims
    them, and dispatch says so rather than serving the current walk under the old op's name.
    """
    scales = [1.0, 1.0, 2.0, 2.0]
    if op_type == "Upsample":
        # The revision that took its scales as an attribute rather than as an operand.
        model = _model(
            [helper.make_node(op_type, ["x"], ["y"], name="node", scales=scales)],
            [_tensor("x", TensorProto.FLOAT, (1, 1, 4, 4))],
            [helper.make_empty_tensor_value_info("y")],
            opset=opset,
        )
    else:
        model = _resize_model(
            (1, 1, 4, 4), op_type=op_type, scales=scales, opset=opset, name="node"
        )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert f"opset version {opset}" in message
    assert f"Nearest supported version: {nearest}" in message


# --------------------------------------------------------------------------------------
# The samplers
# --------------------------------------------------------------------------------------

# What a sampler reads at run time is a coordinate, not an index, so what is asserted here is
# what the sweeps cannot reach: that the mode, the padding and the geometry are call-site
# literals rather than kernels of their own; that a coordinate no conversion is defined for
# reads nothing instead of reading past a buffer; and the errors for the combinations the
# compiler refuses outright. The two block shuffles are neither sampled nor computed — they
# are the strided move every view op runs — which is what their tests say.

_AFFINE_GRID_OPSET = 20
_COL2IM_OPSET = 18
_BLOCK_OPSET = 13


def _grid_sample_model(
    x_shape,
    grid_shape,
    *,
    elem_type=TensorProto.FLOAT,
    grid_type=TensorProto.FLOAT,
    name="sample",
    **attributes,
):
    return _model(
        [helper.make_node("GridSample", ["x", "g"], ["y"], name=name, **attributes)],
        [_tensor("x", elem_type, x_shape), _tensor("g", grid_type, grid_shape)],
        [helper.make_empty_tensor_value_info("y")],
    )


def _roi_model(
    op_type,
    x_shape,
    roi_count,
    *,
    elem_type=TensorProto.FLOAT,
    columns=None,
    indices=None,
    name="roi",
    **attributes,
):
    """One region-of-interest pooling; MaxRoiPool carries its batch in the region itself."""
    columns = (4 if op_type == "RoiAlign" else 5) if columns is None else columns
    names = ["x", "rois"]
    inputs = [
        _tensor("x", elem_type, x_shape),
        _tensor("rois", elem_type, (roi_count, columns)),
    ]
    if op_type == "RoiAlign":
        names.append("batch_indices")
        inputs.append(
            _tensor(
                "batch_indices",
                TensorProto.INT64,
                (roi_count,) if indices is None else indices,
            )
        )
    return _model(
        [helper.make_node(op_type, names, ["y"], name=name, **attributes)],
        inputs,
        [helper.make_empty_tensor_value_info("y")],
    )


def _affine_grid_model(
    size,
    *,
    theta_shape=None,
    elem_type=TensorProto.FLOAT,
    runtime=False,
    name="grid",
    **attributes,
):
    """One AffineGrid, its size carried in the model unless `runtime` feeds it instead."""
    rank = len(size) - 2
    theta_shape = (size[0], rank, rank + 1) if theta_shape is None else theta_shape
    values = np.array(size, dtype=np.int64)
    inputs = [_tensor("theta", elem_type, theta_shape)]
    initializer = []
    if runtime:
        inputs.append(_tensor("size", TensorProto.INT64, values.shape))
    else:
        initializer.append(onnx.numpy_helper.from_array(values, "size"))
    return _model(
        [
            helper.make_node(
                "AffineGrid", ["theta", "size"], ["y"], name=name, **attributes
            )
        ],
        inputs,
        [helper.make_empty_tensor_value_info("y")],
        initializer=initializer,
        opset=_AFFINE_GRID_OPSET,
    )


def _col2im_model(
    x_shape,
    image,
    block,
    *,
    elem_type=TensorProto.FLOAT,
    runtime=(),
    output=None,
    name="fold",
    **attributes,
):
    """One Col2Im, its extents carried in the model unless `runtime` names them."""
    inputs = [_tensor("x", elem_type, x_shape)]
    initializer = []
    for operand, extents in (("image_shape", image), ("block_shape", block)):
        values = np.array(extents, dtype=np.int64)
        if operand in runtime:
            inputs.append(_tensor(operand, TensorProto.INT64, values.shape))
        else:
            initializer.append(onnx.numpy_helper.from_array(values, operand))
    declared = (
        helper.make_empty_tensor_value_info("y")
        if output is None
        else _tensor("y", elem_type, output)
    )
    return _model(
        [
            helper.make_node(
                "Col2Im",
                ["x", "image_shape", "block_shape"],
                ["y"],
                name=name,
                **attributes,
            )
        ],
        inputs,
        [declared],
        initializer=initializer,
        opset=_COL2IM_OPSET,
    )


def _block_model(
    op_type, x_shape, *, elem_type=TensorProto.FLOAT, name="block", **attributes
):
    return _model(
        [helper.make_node(op_type, ["x"], ["y"], name=name, **attributes)],
        [_tensor("x", elem_type, x_shape)],
        [helper.make_empty_tensor_value_info("y")],
        opset=_BLOCK_OPSET,
    )


def _samplers_model(name="sample"):
    """Four samplers over one operand: what they share and what they do not."""
    return _model(
        [
            helper.make_node("GridSample", ["x", "g"], ["p"], name=name, mode="linear"),
            helper.make_node(
                "GridSample",
                ["x", "g"],
                ["q"],
                name="cubic",
                mode="cubic",
                padding_mode="border",
                align_corners=1,
            ),
            helper.make_node(
                "GridSample", ["x", "g"], ["r"], name="near", mode="nearest"
            ),
            helper.make_node(
                "RoiAlign",
                ["x", "rois", "batch_indices"],
                ["s"],
                name="align",
                output_height=2,
                output_width=2,
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 4, 5)),
            _tensor("g", TensorProto.FLOAT, (1, 3, 3, 2)),
            _tensor("rois", TensorProto.FLOAT, (2, 4)),
            _tensor("batch_indices", TensorProto.INT64, (2,)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r", "s")],
    )


def test_interpolating_grid_samples_of_one_element_type_share_a_kernel(tmp_path):
    """The mode, the padding and the corner convention are arguments, not kernels.

    Two nodes interpolating differently — one cubic against a clamped border, one linear
    against zeros — run the same code at different literals. `nearest` is the one that is
    not: it reads a single element and computes nothing, which is what lets it serve the
    element types no interpolation is defined for.
    """
    report, header = _compile(_samplers_model(), tmp_path)

    interpolating, nearest = sorted(_kernels(report, "gridsample"))
    assert interpolating.endswith("_gridsample_float_float")
    assert nearest.endswith("_gridsample_nearest_float_float")
    assert header.count(f"static void {interpolating}(") == 1
    assert header.count(f"{interpolating}(\n") == 3


def test_every_sampler_shares_one_padding_resolution(tmp_path):
    """What a coordinate outside the operand reads is one decision, made in one place."""
    report, header = _compile(_samplers_model(), tmp_path)

    for helper_name, returns in (
        ("sample_reflect", "double"),
        ("sample_index", "ptrdiff_t"),
        ("sample_locate_float", "float"),
        ("sample_coefficient_float", "float"),
    ):
        (shared,) = _kernels(report, helper_name)
        assert header.count(f"static {returns} {shared}(") == 1


def test_the_grid_sample_geometry_reaches_the_kernel_as_call_site_literals(tmp_path):
    """The extents, the mode and the padding are compile-time constants at the call site."""
    model = _grid_sample_model(
        (2, 3, 4, 5, 6),
        (2, 2, 3, 4, 3),
        mode="cubic",
        padding_mode="reflection",
        align_corners=1,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "gridsample")
    call = header.split(f"{kernel}(\n")[-1].splitlines()
    assert [line.strip(" ,);") for line in call[:12]] == [
        "y",
        "x",
        "g",
        "2u",
        "3u",
        "120u",
        "24u",
        "3",
        "(const size_t[]){4u, 5u, 6u}",
        # `reflection` padding, aligned corners, and the cubic mode.
        "2",
        "1",
        "2",
    ]


@requires_c_compiler
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_a_grid_coordinate_no_index_could_hold_reads_nothing(tmp_path, padding_mode):
    """A coordinate arrives at run time, and not every one of them names an element.

    ONNX says nothing about a grid holding infinities or values that are not numbers at all —
    its own reference raises on them — so what is asserted is only that the artifact answers
    with a value from the operand or with zero, rather than reading wherever an out-of-range
    conversion would point.
    """
    model = _grid_sample_model(
        (1, 1, 4, 4), (1, 2, 3, 2), mode="nearest", padding_mode=padding_mode
    )
    grid = np.array(
        [
            [np.nan, 0.0],
            [np.inf, 0.5],
            [-np.inf, -0.5],
            [1e30, 0.0],
            [-1e30, 0.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    ).reshape(1, 2, 3, 2)
    values = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

    outputs = compile_onnx(model, tmp_path).load().run({"x": values, "g": grid})

    assert np.isin(outputs["y"], np.append(values, 0.0)).all()


@requires_c_compiler
def test_a_grid_sample_writing_nothing_emits_no_kernel_at_all(tmp_path):
    """An empty batch leaves no position to sample, and no loop that could read a buffer."""
    model = _grid_sample_model((0, 2, 4, 5), (0, 3, 3, 2))

    report, header = _compile(model, tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [0, 2, 3, 3]
    assert not _kernels(report, "gridsample")
    assert f"{report['prefix']}_gridsample" not in header


@requires_c_compiler
def test_a_nearest_grid_sample_carries_every_value_through_unchanged(tmp_path):
    """It selects rather than interpolates, which is why it serves the integer types too."""
    model = _grid_sample_model(
        (1, 1, 2, 2),
        (1, 2, 2, 2),
        elem_type=TensorProto.INT64,
        mode="nearest",
        align_corners=1,
    )
    values = np.array([[[[-(2**62), 2**62 - 1], [7, -7]]]], dtype=np.int64)
    grid = np.array(
        [[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]], np.float32
    )

    outputs = compile_onnx(model, tmp_path).load().run({"x": values, "g": grid})

    np.testing.assert_array_equal(outputs["y"], values)


@requires_c_compiler
def test_the_regions_a_roi_align_reads_are_checked_against_the_batch(tmp_path):
    """A batch index outside the operand is an argument the artifact reports, not a read."""
    model = _roi_model("RoiAlign", (2, 1, 4, 4), 1, output_height=2, output_width=2)
    compiled = compile_onnx(model, tmp_path).load()
    feeds = {
        "x": np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4),
        "rois": np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
    }

    inside = compiled.run({**feeds, "batch_indices": np.array([1], dtype=np.int64)})

    assert inside["y"].shape == (1, 1, 2, 2)
    for outside in (2, -1):
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run(
                {**feeds, "batch_indices": np.array([outside], dtype=np.int64)}
            )


@requires_c_compiler
def test_the_batch_a_max_roi_pool_reads_is_checked_against_the_operand(tmp_path):
    """MaxRoiPool carries the batch in the region's first column, and checks it there."""
    model = _roi_model("MaxRoiPool", (2, 1, 4, 4), 1, pooled_shape=[2, 2])
    compiled = compile_onnx(model, tmp_path).load()
    values = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)

    inside = compiled.run(
        {"x": values, "rois": np.array([[1.0, 0.0, 0.0, 3.0, 3.0]], np.float32)}
    )

    assert inside["y"].shape == (1, 1, 2, 2)
    for region in (
        [2.0, 0.0, 0.0, 3.0, 3.0],
        [-1.0, 0.0, 0.0, 3.0, 3.0],
        [0.0, 0.0, 0.0, 1e30, 3.0],
    ):
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run({"x": values, "rois": np.array([region], np.float32)})


@requires_c_compiler
def test_a_max_roi_pool_pools_the_same_regions_at_either_precision(tmp_path):
    """The double kernel is the float one at another type, which is what grounds it.

    onnxruntime — the only implementation ONNX has for this op, and so the differential
    sweep's oracle for it — is registered for float alone. On data every value of which both
    types hold exactly, the two kernels select from the same elements, so the wider one is
    pinned to the narrower one the sweep covers.
    """
    values = np.arange(-24, 24, dtype=np.float64).reshape(2, 2, 4, 3)
    regions = np.array(
        [[0.0, 0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 0.0, 3.0, 2.0]], dtype=np.float64
    )
    outputs = {}
    for elem_type, dtype in (
        (TensorProto.FLOAT, np.float32),
        (TensorProto.DOUBLE, np.float64),
    ):
        model = _roi_model(
            "MaxRoiPool",
            values.shape,
            len(regions),
            elem_type=elem_type,
            pooled_shape=[2, 2],
        )
        compiled = compile_onnx(model, tmp_path / dtype.__name__).load()
        outputs[dtype] = compiled.run(
            {"x": values.astype(dtype), "rois": regions.astype(dtype)}
        )["y"]

    np.testing.assert_array_equal(
        outputs[np.float64], outputs[np.float32].astype(np.float64)
    )


@requires_c_compiler
def test_an_affine_grid_maps_the_same_coordinates_at_either_precision(tmp_path):
    """The one place onnxruntime is the oracle, because ONNX's own is blind to the type.

    The reference evaluator casts its result to float32 whatever type the transform arrives
    in, so it cannot tell a double AffineGrid from a float one — the differential sweep runs
    it at float32 for that reason. onnxruntime, the second oracle the compiler's parity
    testing rests on, computes the grid in the type ONNX defines for it — save for the
    spacing between two positions, which it rounds to float32 whatever the type. The extents
    here are the ones whose spacing that rounding leaves exact, so the two are compared
    element for element.
    """
    runtime = pytest.importorskip("onnxruntime")
    runtime.set_default_logger_severity(3)
    model = _affine_grid_model(
        (2, 3, 5, 5), elem_type=TensorProto.DOUBLE, align_corners=1
    )
    theta = np.array(
        [[[0.7, -0.3, 0.1], [0.25, 0.9, -0.2]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]],
        dtype=np.float64,
    )

    outputs = compile_onnx(model, tmp_path).load().run({"theta": theta})

    session = runtime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = session.run(None, {"theta": theta})[0]
    assert outputs["y"].dtype == np.float64
    np.testing.assert_array_equal(outputs["y"], expected)


@requires_c_compiler
def test_a_col2im_folding_nothing_emits_no_call(tmp_path):
    """An empty batch leaves no image to fold into, and no loop over one."""
    report, header = _compile(_col2im_model((0, 5, 5), (5, 5), (1, 5)), tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [0, 1, 5, 5]
    assert not _kernels(report, "col2im")
    assert f"{report['prefix']}_col2im" not in header


def test_the_col2im_geometry_reaches_the_kernel_as_call_site_literals(tmp_path):
    """The block, the image and the positions the blocks sat at are all literals."""
    model = _col2im_model(
        (2, 12, 9), (4, 4), (2, 2), dilations=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "col2im")
    call = header.split(f"{kernel}(\n")[-1].splitlines()
    assert [line.strip(" ,);") for line in call[:13]] == [
        "y",
        "x",
        "6u",
        "9u",
        "16u",
        "4u",
        "2",
        "(const size_t[]){4u, 4u}",
        "(const size_t[]){2u, 2u}",
        "(const size_t[]){3u, 3u}",
        "(const size_t[]){1u, 1u}",
        "(const size_t[]){1u, 1u}",
        "(const ptrdiff_t[]){0, 0}",
    ]


@pytest.mark.parametrize(
    ("op_type", "attributes", "shape", "strides"),
    [
        # A block shuffle is a transpose of the operand read as blocks, so what it emits is
        # the shared strided move at the strides that transpose says.
        (
            "DepthToSpace",
            {"blocksize": 2, "mode": "DCR"},
            "(const size_t[]){1u, 2u, 2u, 2u, 3u, 2u}",
            "(const ptrdiff_t[]){48, 6, 3, 24, 1, 12}",
        ),
        (
            "DepthToSpace",
            {"blocksize": 2, "mode": "CRD"},
            "(const size_t[]){1u, 2u, 2u, 2u, 3u, 2u}",
            "(const ptrdiff_t[]){48, 24, 3, 12, 1, 6}",
        ),
        (
            "SpaceToDepth",
            {"blocksize": 2},
            "(const size_t[]){1u, 2u, 2u, 2u, 2u, 3u}",
            "(const ptrdiff_t[]){48, 6, 1, 24, 12, 2}",
        ),
    ],
)
def test_a_block_shuffle_is_emitted_as_the_shared_strided_move(
    tmp_path, op_type, attributes, shape, strides
):
    x_shape = (1, 8, 2, 3) if op_type == "DepthToSpace" else (1, 2, 4, 6)
    result_strides = (
        "(const ptrdiff_t[]){48, 24, 12, 6, 2, 1}"
        if op_type == "DepthToSpace"
        else "(const ptrdiff_t[]){48, 24, 12, 6, 3, 1}"
    )
    model = _block_model(op_type, x_shape, **attributes)

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "copy")
    assert not _kernels(report, op_type.lower())
    call = header.split(f"{kernel}(\n")[-1].splitlines()
    assert [line.strip(" ,);") for line in call[:7]] == [
        "y",
        "x",
        "48u",
        "6",
        shape,
        result_strides,
        strides,
    ]


@requires_c_compiler
@pytest.mark.parametrize("op_type", ["DepthToSpace", "SpaceToDepth"])
def test_a_block_shuffle_of_one_element_moves_nothing(tmp_path, op_type):
    """A block of one leaves every element where it is, which is one `memcpy`."""
    model = _block_model(op_type, (2, 3, 4, 5), blocksize=1)

    report, header = _compile(model, tmp_path)

    assert not _kernels(report, "copy")
    assert "memcpy(y, x, 120u * sizeof(*y));" in header


@requires_c_compiler
def test_a_block_shuffle_writing_nothing_emits_no_move(tmp_path):
    report, header = _compile(
        _block_model("SpaceToDepth", (0, 2, 6, 4), blocksize=2), tmp_path
    )

    assert report["entrypoint"]["outputs"][0]["shape"] == [0, 8, 3, 2]
    assert not _kernels(report, "copy")
    assert "memcpy" not in header.split(f"int {report['prefix']}_run(")[-1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        # Everything a grid sample settles that ONNX's own inference does not: given the
        # shapes, inference rejects a grid of the wrong rank or width before this does.
        (
            {"elem_type": TensorProto.INT32, "mode": "linear"},
            "weights the elements around each coordinate",
        ),
        (
            {"elem_type": TensorProto.BOOL, "mode": "cubic"},
            "weights the elements around each coordinate",
        ),
        ({"mode": "bilinear"}, "is not one of the values ONNX defines for it"),
        ({"padding_mode": "wrap"}, "is not one of the values ONNX defines for it"),
    ],
)
def test_a_grid_sample_the_compiler_cannot_serve_is_rejected(tmp_path, kwargs, message):
    x_shape = kwargs.pop("x_shape", (1, 2, 4, 5))
    grid_shape = kwargs.pop("grid_shape", (1, 3, 3, 2))

    with pytest.raises(CompileError) as error:
        compile_onnx(_grid_sample_model(x_shape, grid_shape, **kwargs), tmp_path)

    assert "`sample`" in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    ("op_type", "kwargs", "message"),
    [
        ("RoiAlign", {"columns": 5}, "one row of 4 value(s) per region"),
        ("MaxRoiPool", {"columns": 4}, "one row of 5 value(s) per region"),
        ("RoiAlign", {"x_shape": (1, 2, 0, 4)}, "holds no elements to sample"),
        ("MaxRoiPool", {"x_shape": (1, 2, 4, 0)}, "holds no elements to sample"),
        (
            "RoiAlign",
            {"mode": "median"},
            "is not one of the values ONNX defines for it",
        ),
        (
            "RoiAlign",
            {"coordinate_transformation_mode": "asymmetric"},
            "is not one of the values ONNX defines for it",
        ),
    ],
)
def test_a_region_pooling_the_compiler_cannot_serve_is_rejected(
    tmp_path, op_type, kwargs, message
):
    defaults = (
        {"output_height": 2, "output_width": 2}
        if op_type == "RoiAlign"
        else {"pooled_shape": [2, 2]}
    )
    x_shape = kwargs.pop("x_shape", (2, 3, 4, 5))

    with pytest.raises(CompileError) as error:
        compile_onnx(
            _roi_model(op_type, x_shape, 2, **{**defaults, **kwargs}), tmp_path
        )

    assert "`roi`" in str(error.value)
    assert message in str(error.value)


def test_a_max_roi_pool_without_a_pooled_shape_is_rejected(tmp_path):
    """ONNX defines the attribute as required, so its absence is not a default."""
    model = _model(
        [helper.make_node("MaxRoiPool", ["x", "rois"], ["y"], name="roi")],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 4, 5)),
            _tensor("rois", TensorProto.FLOAT, (2, 5)),
        ],
        [_tensor("y", TensorProto.FLOAT, (2, 2, 2, 2))],
    )

    with pytest.raises(CompileError, match=re.escape("states no `pooled_shape`")):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"theta_shape": (2, 2, 2)}, "ONNX defines as a transform of shape [2, 2, 3]"),
        ({"theta_shape": (1, 2, 3)}, "ONNX defines as a transform of shape [2, 2, 3]"),
    ],
)
def test_an_affine_grid_the_compiler_cannot_serve_is_rejected(
    tmp_path, kwargs, message
):
    size = kwargs.pop("size", (2, 3, 4, 5))
    # A theta the size disagrees with leaves ONNX's own inference nothing to derive, so the
    # result's shape is declared rather than inferred.
    model = _affine_grid_model(size, **kwargs)

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`grid`" in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    ("op_type", "runtime", "operand"),
    [
        ("AffineGrid", ("size",), "size"),
        ("Col2Im", ("image_shape",), "image_shape"),
        ("Col2Im", ("block_shape",), "block_shape"),
    ],
)
def test_an_extent_read_at_run_time_is_rejected_as_a_dynamic_shape(
    tmp_path, op_type, runtime, operand
):
    """The operands that decide these results' shapes have to be fixed by the graph.

    Which is what the corpus's own tests of both ops do not do — every one of them feeds the
    extents — so the frontend names the operand rather than letting a kernel reach for values
    that are not there.
    """
    model = (
        _affine_grid_model((2, 3, 4, 5), runtime=True)
        if op_type == "AffineGrid"
        else _col2im_model(
            (1, 5, 5), (5, 5), (1, 5), runtime=runtime, output=(1, 1, 5, 5)
        )
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert f"`{operand}`" in str(error.value)
    assert "depends on input data" in str(error.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block": (2, 2)}, "which does not divide the 5 row(s)"),
        ({"x_shape": (1, 5, 7)}, "block(s) over an image of [5, 5]"),
        ({"attributes": {"strides": [2, 2]}}, "block(s) over an image of [5, 5]"),
        ({"image": (5, 0)}, "ONNX defines them as extents, which are positive"),
        ({"block": (1, 5, 5)}, "for a 2-dimensional image"),
        ({"x_shape": (1, 1, 5, 5)}, "a tensor of rank 3"),
        (
            {"elem_type": TensorProto.BOOL},
            "summing truth values has no defined result",
        ),
    ],
)
def test_a_col2im_the_compiler_cannot_serve_is_rejected(tmp_path, kwargs, message):
    shape = kwargs.pop("x_shape", (1, 5, 5))
    image = kwargs.pop("image", (5, 5))
    block = kwargs.pop("block", (1, 5))
    elem_type = kwargs.pop("elem_type", TensorProto.FLOAT)
    # ONNX's own inference rejects most of these first, given the extents; the declared
    # result shape is what leaves them to the compiler at all.
    model = _col2im_model(
        shape,
        image,
        block,
        elem_type=elem_type,
        output=(shape[0], 1, 5, 5),
        **kwargs.pop("attributes", {}),
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`fold`" in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    ("op_type", "shape", "attributes", "message"),
    [
        # What a block shuffle settles that ONNX's own inference does not: given the shape,
        # inference rejects a non-positive blocksize and a rank other than 4 before this does.
        ("DepthToSpace", (1, 7, 2, 3), {"blocksize": 2}, "does not divide them evenly"),
        ("SpaceToDepth", (1, 2, 5, 4), {"blocksize": 2}, "do not tile it evenly"),
        (
            "DepthToSpace",
            (1, 8, 2, 3),
            {"blocksize": 2, "mode": "RCD"},
            "not one of the modes ONNX defines",
        ),
    ],
)
def test_a_block_shuffle_the_compiler_cannot_serve_is_rejected(
    tmp_path, op_type, shape, attributes, message
):
    model = _block_model(op_type, shape, **attributes)

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`block`" in str(error.value)
    assert message in str(error.value)


def test_a_block_shuffle_without_a_blocksize_is_rejected(tmp_path):
    """ONNX defines the attribute as required, so its absence is not a default."""
    model = _model(
        [helper.make_node("DepthToSpace", ["x"], ["y"], name="block")],
        [_tensor("x", TensorProto.FLOAT, (1, 8, 2, 3))],
        [_tensor("y", TensorProto.FLOAT, (1, 2, 4, 6))],
        opset=_BLOCK_OPSET,
    )

    with pytest.raises(CompileError, match=re.escape("states no `blocksize`")):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("op_type", "opset", "nearest"),
    [
        ("GridSample", 16, 22),
        ("GridSample", 20, 22),
        ("RoiAlign", 10, 22),
        ("RoiAlign", 16, 22),
        ("MaxRoiPool", 21, 22),
        ("DepthToSpace", 11, 13),
        ("SpaceToDepth", 1, 13),
    ],
)
def test_a_sampler_below_its_supported_revision_names_the_nearest_version(
    tmp_path, op_type, opset, nearest
):
    """None of the older revisions has an oracle — the reference evaluator applies today's
    implementation to them and the backend corpus tests none of them — so no kernel claims
    them, and dispatch says so rather than serving the current walk under an older name."""
    models = {
        "GridSample": lambda: _grid_sample_model(
            (1, 2, 4, 5), (1, 3, 3, 2), name="node"
        ),
        "RoiAlign": lambda: _roi_model(
            "RoiAlign", (1, 2, 4, 5), 2, name="node", output_height=2, output_width=2
        ),
        "MaxRoiPool": lambda: _roi_model(
            "MaxRoiPool", (1, 2, 4, 5), 2, name="node", pooled_shape=[2, 2]
        ),
        "DepthToSpace": lambda: _block_model(
            "DepthToSpace", (1, 8, 2, 3), name="node", blocksize=2
        ),
        "SpaceToDepth": lambda: _block_model(
            "SpaceToDepth", (1, 2, 6, 4), name="node", blocksize=2
        ),
    }
    model = models[op_type]()
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", opset))

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert f"opset version {opset}" in message
    assert f"Nearest supported version: {nearest}" in message


# --------------------------------------------------------------------------------------
# The scatters
# --------------------------------------------------------------------------------------

# What a scatter computes is settled by the conformance and differential suites. What is
# asserted here is what those cannot reach: that the result is the operand copied and then
# written into, that the geometry and the fold are compile-time literals rather than kernels
# of their own, that an index no element sits at is reported instead of written past, and the
# errors for the shapes and attributes the compiler refuses outright.

_SCATTER_OPSET = 18
_TENSOR_SCATTER_OPSET = 24


def _scatter_model(
    op_type,
    data_shape,
    indices_shape,
    *,
    updates_shape=None,
    elem_type=TensorProto.FLOAT,
    index_type=TensorProto.INT64,
    name="scatter",
    opset=_SCATTER_OPSET,
    **attributes,
):
    """One scatter; ScatterElements takes one update per index, ScatterND one per tuple."""
    return _model(
        [
            helper.make_node(
                op_type,
                ["data", "indices", "updates"],
                ["y"],
                name=name,
                **attributes,
            )
        ],
        [
            _tensor("data", elem_type, data_shape),
            _tensor("indices", index_type, indices_shape),
            _tensor(
                "updates",
                elem_type,
                indices_shape if updates_shape is None else updates_shape,
            ),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=opset,
    )


def _tensor_scatter_model(
    cache_shape,
    update_shape,
    *,
    elem_type=TensorProto.FLOAT,
    indices_shape=(),
    name="cache",
    **attributes,
):
    """One TensorScatter; `indices_shape` of None leaves the write indices out."""
    names = ["past", "update"]
    inputs = [
        _tensor("past", elem_type, cache_shape),
        _tensor("update", elem_type, update_shape),
    ]
    if indices_shape is not None:
        names.append("written_at")
        inputs.append(
            _tensor(
                "written_at",
                TensorProto.INT64,
                cache_shape[:1] if indices_shape == () else indices_shape,
            )
        )
    return _model(
        [helper.make_node("TensorScatter", names, ["y"], name=name, **attributes)],
        inputs,
        [helper.make_empty_tensor_value_info("y")],
        opset=_TENSOR_SCATTER_OPSET,
    )


def _reference(model, feeds):
    """What ONNX's own evaluator computes for the model, as the oracle for a kernel test."""
    return ReferenceEvaluator(model).run(None, feeds)


def test_scatters_of_one_fold_and_type_share_a_kernel(tmp_path):
    """A kernel name encodes the fold and the types, and nothing else it does not depend on."""
    model = _model(
        [
            helper.make_node("ScatterElements", ["data", "i", "u"], ["h"], name="one"),
            helper.make_node("ScatterElements", ["h", "i", "u"], ["g"], name="two"),
            helper.make_node(
                "ScatterElements",
                ["g", "i", "u"],
                ["y"],
                name="folded",
                reduction="add",
            ),
        ],
        [
            _tensor("data", TensorProto.FLOAT, (3, 4)),
            _tensor("i", TensorProto.INT64, (2, 4)),
            _tensor("u", TensorProto.FLOAT, (2, 4)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=_SCATTER_OPSET,
    )

    report, header = _compile(model, tmp_path)

    plain, folded = sorted(_kernels(report, "scatterelements"))
    assert plain.endswith("_add_float_int64_t") and folded.endswith(
        "_none_float_int64_t"
    )
    assert header.count(f"static int {folded}(") == 1
    assert header.count(f"{folded}(\n") == 3
    assert "out[offset] = updates[index];" in header
    assert "out[offset] = out[offset] + updates[index];" in header


def test_the_scatter_geometry_reaches_the_kernel_as_call_site_literals(tmp_path):
    """The result is the operand copied, and then written into through its own strides.

    The updates are walked by their own shape and addressed by the operand's strides, which
    is what lets an index tensor cover only part of the axes it does not write along.
    """
    model = _scatter_model("ScatterElements", (3, 4), (2, 3), axis=1)

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "scatterelements")
    assert "memcpy(y, data, 12u * sizeof(*y));" in header
    assert (
        f"if ({kernel}(\n        y,\n        updates,\n        indices,\n"
        "        6u,\n"
        "        2,\n"
        "        (const size_t[]){2u, 3u},\n"
        "        (const size_t[]){4u, 1u},\n"
        "        1,\n"
        "        4u) != 0) {" in header
    )


def test_a_scatter_nd_writes_a_slice_per_index_tuple(tmp_path):
    """The tuple's depth decides how much of the operand one update replaces."""
    model = _scatter_model("ScatterND", (4, 2, 3), (2, 1), updates_shape=(2, 2, 3))

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "scatternd")
    assert (
        f"if ({kernel}(\n        y,\n        updates,\n        indices,\n"
        "        2u,\n"
        "        1u,\n"
        "        6u,\n"
        "        (const size_t[]){4u},\n"
        "        (const size_t[]){6u}) != 0) {" in header
    )


def test_a_scatter_with_no_updates_is_the_copy_alone(tmp_path):
    """Nothing is written, and no loop is emitted that could read an empty buffer."""
    model = _scatter_model("ScatterElements", (3, 4), (0, 4))

    report, header = _compile(model, tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [3, 4]
    assert not _kernels(report, "scatterelements")
    assert "memcpy(y, data, 12u * sizeof(*y));" in header


def test_the_two_families_fold_an_extremum_as_their_own_references_do(tmp_path):
    """`max` means one thing in ScatterElements and another in ScatterND — on a NaN.

    ONNX documents both as folding with `max`, and its two reference implementations disagree
    about the case neither document mentions: ScatterElements folds with Python's own `max`,
    which keeps the element already in the result when a comparison against a NaN comes out
    false, while ScatterND folds with `np.maximum`, which propagates a NaN from either side.
    Each kernel follows the reference of its own op, which is what this pins.
    """
    headers = {
        op_type: _compile(
            _scatter_model(
                op_type,
                (3, 4),
                (2, 4) if op_type == "ScatterElements" else (2, 1),
                updates_shape=None if op_type == "ScatterElements" else (2, 4),
                reduction="max",
            ),
            tmp_path / op_type,
        )[1]
        for op_type in ("ScatterElements", "ScatterND")
    }

    assert (
        "out[offset] = (updates[index] > out[offset]) ? updates[index] : out[offset];"
        in headers["ScatterElements"]
    )
    assert "isnan" not in headers["ScatterElements"]
    assert "_maximum_float(out[offset + element]," in headers["ScatterND"]
    assert (
        "return (left > right || isnan(left)) ? left : right;" in headers["ScatterND"]
    )


@requires_c_compiler
@pytest.mark.parametrize(
    ("op_type", "indices_shape", "updates_shape", "outside"),
    [
        ("ScatterElements", (2, 4), None, (4, -5)),
        ("Scatter", (2, 4), None, (4, -5)),
        ("ScatterND", (2, 1), (2, 4), (3, -4)),
    ],
)
def test_an_index_no_element_sits_at_is_reported(
    tmp_path, op_type, indices_shape, updates_shape, outside
):
    """An index past either end of the axis is an argument the artifact reports."""
    model = _scatter_model(
        op_type, (3, 4), indices_shape, updates_shape=updates_shape, axis=0
    )
    compiled = compile_onnx(model, tmp_path).load()
    feeds = {
        "data": np.zeros((3, 4), np.float32),
        "updates": np.ones(
            indices_shape if updates_shape is None else updates_shape, np.float32
        ),
    }

    inside = compiled.run(
        {**feeds, "indices": np.full(indices_shape, -1, dtype=np.int64)}
    )

    np.testing.assert_array_equal(inside["y"][-1], np.ones(4, np.float32))
    for index in outside:
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run(
                {**feeds, "indices": np.full(indices_shape, index, dtype=np.int64)}
            )


@requires_c_compiler
def test_a_deprecated_scatter_computes_what_its_successor_does(tmp_path):
    """Scatter dispatches at the revisions before the deprecating one too.

    Only the deprecating revision has an oracle in the differential sweep — the corpus's own
    Scatter tests import opset 10 — so what this adds is that every revision the kernel claims
    reaches it, against the ScatterElements ONNX's own document says computes the same thing.
    """
    feeds = {
        "data": np.arange(12, dtype=np.float32).reshape(3, 4),
        "indices": np.array([[0, 2], [1, 0]], np.int64),
        "updates": np.array([[10.0, 20.0], [30.0, 40.0]], np.float32),
    }
    successor = _scatter_model("ScatterElements", (3, 4), (2, 2), axis=1)
    (expected,) = _reference(successor, feeds)

    for opset in (9, 10, 11):
        model = _scatter_model("Scatter", (3, 4), (2, 2), axis=1, opset=opset)
        outputs = compile_onnx(model, tmp_path / str(opset)).load().run(feeds)

        np.testing.assert_array_equal(outputs["y"], expected)


def test_a_tensor_scatter_without_write_indices_reads_none(tmp_path):
    """The operand is optional, and a kernel that took it anyway would read a buffer that
    is not there; leaving it out is a kernel of its own, writing from the start of the axis."""
    report, header = _compile(
        _tensor_scatter_model((2, 1, 4, 5), (2, 1, 2, 5), indices_shape=None), tmp_path
    )

    (kernel,) = _kernels(report, "tensorscatter")
    assert kernel.endswith("_linear_appended_float")
    assert "write_indices" not in header
    assert "const ptrdiff_t written_at = 0;" in header


@requires_c_compiler
def test_a_linear_write_past_the_end_of_the_cache_is_reported(tmp_path):
    """ONNX's own reference answers a write running off the axis with an exception; the
    artifact answers it with the argument error the status enum exists for, having read
    nothing outside the buffer."""
    model = _tensor_scatter_model((2, 1, 4, 5), (2, 1, 2, 5))
    compiled = compile_onnx(model, tmp_path).load()
    feeds = {
        "past": np.zeros((2, 1, 4, 5), np.float32),
        "update": np.ones((2, 1, 2, 5), np.float32),
    }

    inside = compiled.run({**feeds, "written_at": np.array([0, 2], np.int64)})

    assert inside["y"][1, 0, 3].tolist() == [1.0] * 5
    for written_at in ((0, 3), (-1, 0), (4, 0)):
        with pytest.raises(HarnessError, match="status 1"):
            compiled.run({**feeds, "written_at": np.array(written_at, np.int64)})


@requires_c_compiler
def test_a_circular_write_wraps_where_a_linear_one_is_refused(tmp_path):
    """The mode is what decides whether running off the end is an error or a wrap."""
    written_at = np.array([3, 0], np.int64)
    feeds = {
        "past": np.zeros((2, 1, 4, 5), np.float32),
        "update": np.arange(20, dtype=np.float32).reshape(2, 1, 2, 5),
        "written_at": written_at,
    }
    model = _tensor_scatter_model((2, 1, 4, 5), (2, 1, 2, 5), mode="circular")
    (expected,) = _reference(model, feeds)

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    np.testing.assert_array_equal(outputs["y"], expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"op_type": "ScatterElements", "indices_shape": (2, 4, 1)},
            "the same rank",
        ),
        (
            {
                "op_type": "ScatterElements",
                "indices_shape": (2, 4),
                "updates_shape": (2, 3),
            },
            "one update per index",
        ),
        (
            {"op_type": "ScatterElements", "indices_shape": (4, 4), "axis": 1},
            "reaches past it on axis 0",
        ),
        (
            {
                "op_type": "ScatterElements",
                "indices_shape": (2, 4),
                "reduction": "mean",
            },
            "not one of the reductions ONNX defines",
        ),
        (
            {"op_type": "Scatter", "indices_shape": (2, 4), "reduction": "add"},
            "not one of the reductions ONNX defines",
        ),
        (
            {"op_type": "ScatterND", "indices_shape": (2, 3), "updates_shape": (2,)},
            "addresses 3 dimension(s)",
        ),
        (
            {"op_type": "ScatterND", "indices_shape": (2, 1), "updates_shape": (2, 3)},
            "writes one slice per index tuple",
        ),
    ],
)
def test_a_scatter_the_compiler_cannot_serve_is_rejected(tmp_path, kwargs, message):
    op_type = kwargs.pop("op_type")

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(_scatter_model(op_type, (3, 4), **kwargs), tmp_path)


def test_a_tensor_scatter_mode_onnx_does_not_define_is_rejected(tmp_path):
    """The one thing about a TensorScatter that ONNX's own shape inference does not check."""
    model = _tensor_scatter_model((2, 1, 4, 5), (2, 1, 2, 5), mode="rolling")

    with pytest.raises(CompileError, match="not one of the modes ONNX defines"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        # The sequence axis is never the batch the write indices are read by — ONNX's own
        # inference calls axis 0 out of range for any rank, which a rank-2 cache reaches
        # through the default axis alone.
        {"axis": 0},
        {"cache_shape": (3, 4), "update_shape": (3, 2)},
        # An update longer than the cache along the sequence axis, one differing on another
        # axis, one of another rank, and a write index per something other than the batch.
        {"update_shape": (2, 1, 5, 5)},
        {"update_shape": (2, 1, 2, 4)},
        {"update_shape": (2, 1, 2)},
        {"indices_shape": (3,)},
    ],
)
def test_a_tensor_scatter_of_disagreeing_shapes_is_rejected(tmp_path, kwargs):
    """Every one of these is refused before a kernel is asked for it.

    ONNX's own shape inference relates a TensorScatter's operands, so a model that puts them
    at odds has no inferred result to compile against; the compiler reports that against the
    node rather than emitting a kernel whose addressing would run off a buffer. The kernel
    generator checks the same relations again, which is where a shape that reached it another
    way would stop.
    """
    model = _tensor_scatter_model(
        kwargs.pop("cache_shape", (2, 1, 4, 5)),
        kwargs.pop("update_shape", (2, 1, 2, 5)),
        **kwargs,
    )

    with pytest.raises(CompileError, match="cache"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("op_type", "opset"),
    [("ScatterElements", 11), ("ScatterElements", 16), ("ScatterND", 13)],
)
def test_a_scatter_below_its_supported_revision_names_the_nearest_version(
    tmp_path, op_type, opset
):
    """`reduction` arrived at 16 and grew at 18, so the older revisions are a different op.

    The reference evaluator applies today's implementation to all of them and the backend
    corpus tests none of them, so nothing can vouch for one; dispatch says so rather than
    serving the current fold under an older revision's name.
    """
    model = _scatter_model(
        op_type,
        (3, 4),
        (2, 4) if op_type == "ScatterElements" else (2, 1),
        updates_shape=None if op_type == "ScatterElements" else (2, 4),
        name="node",
        opset=opset,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert f"opset version {opset}" in message
    assert "Nearest supported version: 18" in message


# --------------------------------------------------------------------------------------
# The einsum contraction
# --------------------------------------------------------------------------------------

# What an equation computes is settled by the conformance and differential suites, against
# ONNX's corpus and reference evaluator. What is asserted here is what those cannot reach:
# that the equation is read at compile time into extents and strides — so that a diagonal is
# addressing rather than code, and every equation of one arity and element type is one shared
# kernel — and the errors for the equations the compiler refuses outright.

_EINSUM_OPSET = 12


def _einsum_model(
    equation,
    shapes,
    *,
    names=None,
    elem_type=TensorProto.FLOAT,
    result_shape=None,
    name="node",
):
    """One Einsum; `result_shape` declares the result rather than leaving it to inference."""
    names = list(names or [f"in{index}" for index in range(len(shapes))])
    result = (
        helper.make_empty_tensor_value_info("y")
        if result_shape is None
        else _tensor("y", elem_type, result_shape)
    )
    return _model(
        [helper.make_node("Einsum", names, ["y"], name=name, equation=equation)],
        [_tensor(operand, elem_type, shape) for operand, shape in zip(names, shapes)],
        [result],
        opset=_EINSUM_OPSET,
    )


def test_equations_of_one_arity_and_element_type_share_a_kernel(tmp_path):
    """The equation is addressing, so only the operand count and type reach the loop."""
    model = _model(
        [
            helper.make_node(
                "Einsum", ["a", "b"], ["p"], name="product", equation="ij,jk->ik"
            ),
            helper.make_node(
                "Einsum", ["a", "a"], ["q"], name="hadamard", equation="ij,ij->ij"
            ),
            helper.make_node("Einsum", ["a"], ["r"], name="total", equation="ij->"),
        ],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (3, 4)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r")],
        opset=_EINSUM_OPSET,
    )

    report, header = _compile(model, tmp_path)

    unary, binary = sorted(_kernels(report, "einsum"))
    assert unary.endswith("_1_float") and binary.endswith("_2_float")
    assert header.count(f"static void {binary}(") == 1
    assert header.count(f"{binary}(\n") == 3


def test_the_equation_reaches_the_kernel_as_call_site_literals(tmp_path):
    """A label the result keeps is a stride per operand; one it drops is the summed loop."""
    model = _einsum_model("ij,jk->ik", [(2, 3), (3, 4)], names=["a", "b"])

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "einsum")
    assert (
        f"{kernel}(\n        y,\n        a,\n        b,\n"
        "        8u,\n"
        "        2,\n"
        "        (const size_t[]){2u, 4u},\n"
        "        (const size_t[]){3u, 0u},\n"
        "        (const size_t[]){0u, 1u},\n"
        "        3u,\n"
        "        1,\n"
        "        (const size_t[]){3u},\n"
        "        (const size_t[]){1u},\n"
        "        (const size_t[]){4u});" in header
    )


def test_a_label_repeated_in_a_term_is_a_stride_down_the_diagonal(tmp_path):
    """The strides of the axes one label names add up: 3 + 1 walks a 3x3 diagonally."""
    model = _einsum_model("...ii->...i", [(2, 3, 3)], names=["a"])

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "einsum")
    assert (
        f"{kernel}(\n        y,\n        a,\n"
        "        6u,\n"
        "        2,\n"
        "        (const size_t[]){2u, 3u},\n"
        "        (const size_t[]){9u, 4u},\n"
        "        1u,\n"
        "        0,\n"
        "        (const size_t[]){0u},\n"
        "        (const size_t[]){0u});" in header
    )


def test_an_operand_stretched_along_a_label_reads_it_at_a_zero_stride(tmp_path):
    """numpy stretches an extent of 1 against another operand's, which is a stride of 0."""
    model = _einsum_model("ij,ij->j", [(1, 3), (2, 3)], names=["a", "b"])

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "einsum")
    assert (
        f"{kernel}(\n        y,\n        a,\n        b,\n"
        "        3u,\n"
        "        1,\n"
        "        (const size_t[]){3u},\n"
        "        (const size_t[]){1u},\n"
        "        (const size_t[]){1u},\n"
        "        2u,\n"
        "        1,\n"
        "        (const size_t[]){2u},\n"
        "        (const size_t[]){0u},\n"
        "        (const size_t[]){3u});" in header
    )


def test_an_equation_with_no_result_to_write_emits_no_loop(tmp_path):
    """Nothing to write, so no kernel is emitted that could read an empty buffer."""
    report, _ = _compile(_einsum_model("ij->ji", [(0, 3)]), tmp_path)

    assert report["entrypoint"]["outputs"][0]["shape"] == [3, 0]
    assert not _kernels(report, "einsum")


@requires_c_compiler
def test_a_contraction_over_an_empty_axis_sums_over_nothing(tmp_path):
    """The result is written, from a sum of no terms at all, reading neither operand."""
    model = _einsum_model("ij,jk->ik", [(2, 0), (0, 3)])
    feeds = {
        "in0": np.zeros((2, 0), dtype=np.float32),
        "in1": np.zeros((0, 3), dtype=np.float32),
    }

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    np.testing.assert_array_equal(outputs["y"], _reference(model, feeds)[0])


@pytest.mark.parametrize(
    ("label", "equation", "shapes", "result_shape", "expected"),
    [
        ("empty", "", ((),), (), "`equation` is empty"),
        ("not_a_label", "i1,j->ij", ((2,), (3,)), None, "is not a term"),
        ("two_ellipses", "...i...->i", ((2, 3),), None, "is not a term"),
        (
            "two_outputs",
            "ij->i->j",
            ((2, 3),),
            None,
            "states its output more than once",
        ),
        ("uneven_diagonal", "ii->i", ((2, 3),), None, "only over axes of equal extent"),
        (
            "disagreeing_label",
            "ij,jk->ik",
            ((2, 3), (4, 5)),
            None,
            "measures 3 on one operand",
        ),
        ("repeated_output", "i->ii", ((3,),), None, "more than once; each axis"),
        ("unknown_output", "i->ij", ((3,),), None, "which no operand's term carries"),
        (
            "term_per_operand",
            "ij,jk->ik",
            ((2, 3),),
            (2, 4),
            "states 2 term(s) for 1 operand(s)",
        ),
        (
            "label_per_axis",
            "ijk->ij",
            ((2, 3),),
            (2, 3),
            "names 3 label(s) for `in0`",
        ),
    ],
)
def test_an_equation_onnx_does_not_define_is_rejected(
    tmp_path, label, equation, shapes, result_shape, expected
):
    """Every reading ONNX's Einsum leaves undefined is refused by name, never guessed at.

    The empty equation is one numpy would take for a scalar term and ONNX's own reference
    implementation rejects outright; it is refused here for that reason. Declaring a result
    is what reaches these readings at all: left underived, a node ONNX cannot infer a shape
    for is stopped one step earlier, by shape inference rather than by the equation.
    """
    model = _einsum_model(equation, shapes, result_shape=result_shape)

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert expected in message


def test_an_equation_wider_than_the_shape_onnx_inferred_is_rejected(tmp_path):
    """numpy stretches a labelled axis onto the result and ONNX's shape inference does not.

    The two disagree about this equation's result — numpy computes [2, 3] and the buffer ONNX
    sized holds [1, 3] — so the node is refused rather than written past.
    """
    model = _einsum_model("ij,ij->ij", [(1, 3), (2, 3)])

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert "[2, 3]" in message and "[1, 3]" in message


def test_an_einsum_below_its_supported_revision_names_the_nearest_version(tmp_path):
    """Einsum arrived at opset 12; nothing defines it below that."""
    model = _model(
        [helper.make_node("Einsum", ["x"], ["y"], name="node", equation="ij->ji")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (3, 2))],
        opset=11,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`Einsum`" in message
    assert "opset version 11" in message
    assert "Nearest supported version: 12" in message


# --------------------------------------------------------------------------------------
# Opset-dependent dispatch
# --------------------------------------------------------------------------------------


def test_softmax_below_its_supported_revision_names_the_nearest_version(tmp_path):
    """Up to opset 12 Softmax flattened the axes from `axis` on, which nothing can vouch for.

    The reference evaluator applies the current semantics to those revisions and the backend
    corpus has no test at one, so no kernel claims them — and dispatch says so rather than
    serving the current formula under the old op's name.
    """
    model = _model(
        [helper.make_node("Softmax", ["x"], ["y"], name="soft", axis=1)],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        opset=12,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`soft`" in message
    assert "opset version 12" in message
    assert "Nearest supported version: 13" in message


@pytest.mark.parametrize(
    ("op_type", "shapes", "opset", "nearest"),
    [
        ("MatMul", ((2, 3), (3, 4)), 12, 13),
        ("Det", ((2, 2),), 21, 22),
        ("Conv", ((1, 1, 5, 5), (1, 1, 3, 3)), 21, 22),
        ("ConvTranspose", ((1, 1, 5, 5), (1, 1, 3, 3)), 21, 22),
    ],
)
def test_a_matrix_op_below_its_supported_revision_names_the_nearest_version(
    tmp_path, op_type, shapes, opset, nearest
):
    """None of these ops changed semantics at its claimed revision — but nothing vouches for
    the older ones either: the reference evaluator applies today's implementation to them and
    the backend corpus has no test at one, so no kernel claims them.
    """
    names = ["x", "z"][: len(shapes)]
    model = _model(
        [helper.make_node(op_type, names, ["y"], name="node")],
        [_tensor(name, TensorProto.FLOAT, shape) for name, shape in zip(names, shapes)],
        [helper.make_empty_tensor_value_info("y")],
        opset=opset,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`node`" in message
    assert f"opset version {opset}" in message
    assert f"Nearest supported version: {nearest}" in message


@requires_c_compiler
def test_each_opset_gets_the_semantics_of_its_own_revision(tmp_path):
    """`Clip` reads its bounds from attributes up to opset 10 and from inputs after it.

    The same node is two different ops across that boundary: at 6 the attributes bound the
    result, at 13 they are not attributes at all and an unbounded `Clip` is the identity.
    Each model is compared against the reference evaluator reading that same model.
    """
    values = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    results = {}
    for opset, attributes in ((6, {"min": -1.0, "max": 1.0}), (13, {})):
        model = _model(
            [helper.make_node("Clip", ["x"], ["y"], name="clip", **attributes)],
            [_tensor("x", TensorProto.FLOAT, (2, 3))],
            [helper.make_empty_tensor_value_info("y")],
            opset=opset,
        )

        results[opset] = (
            compile_onnx(model, tmp_path / str(opset)).load().run({"x": values})["y"]
        )

        expected = ReferenceEvaluator(model).run(None, {"x": values})[0]
        np.testing.assert_array_equal(results[opset], expected)
    assert not np.array_equal(results[6], results[13])


# --------------------------------------------------------------------------------------
# What the kernels refuse
# --------------------------------------------------------------------------------------


def test_an_undefined_gelu_approximation_is_rejected(tmp_path):
    """`approximate` selects a formula, so an unknown one has no code to emit."""
    model = _model(
        [helper.make_node("Gelu", ["x"], ["y"], name="gelu", approximate="sigmoid")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`gelu`" in message
    assert "`sigmoid`" in message
    assert "`tanh`" in message


def test_a_bitcast_to_bool_is_rejected(tmp_path):
    """A boolean tensor is emitted as bytes holding 0 or 1; arbitrary bits are neither."""
    model = _model(
        [helper.make_node("BitCast", ["x"], ["y"], name="bits", to=TensorProto.BOOL)],
        [_tensor("x", TensorProto.INT8, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        opset=26,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`bits`" in message
    assert "BOOL" in message
    assert "INT8" in message


@pytest.mark.parametrize("direction", ["SIDEWAYS", None])
def test_a_bit_shift_in_no_defined_direction_is_rejected(tmp_path, direction):
    """`direction` selects the operator, so an unknown one has no code to emit."""
    attributes = {} if direction is None else {"direction": direction}
    model = _model(
        [helper.make_node("BitShift", ["a", "b"], ["y"], name="shift", **attributes)],
        [
            _tensor("a", TensorProto.UINT8, (2, 3)),
            _tensor("b", TensorProto.UINT8, (2, 3)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`shift`" in message
    assert "`LEFT`" in message and "`RIGHT`" in message


def test_mod_on_floats_without_fmod_is_rejected(tmp_path):
    """ONNX requires `fmod=1` for the floating-point families; the compiler says so."""
    model = _model(
        [helper.make_node("Mod", ["a", "b"], ["y"], name="remainder")],
        [
            _tensor("a", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.FLOAT, (2, 3)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`remainder`" in message
    assert "FLOAT" in message
    assert "fmod" in message


@pytest.mark.parametrize("training_mode", [None, True])
def test_dropout_that_is_not_provably_inference_is_rejected(tmp_path, training_mode):
    """Training mode samples a mask, which no static artifact can reproduce."""
    initializer = (
        []
        if training_mode is None
        else [onnx.numpy_helper.from_array(np.array(True), "mode")]
    )
    inputs = [_tensor("x", TensorProto.FLOAT, (2, 3))]
    if training_mode is None:
        inputs.append(_tensor("mode", TensorProto.BOOL, ()))
    model = _model(
        [helper.make_node("Dropout", ["x", "", "mode"], ["y"], name="drop")],
        inputs,
        [helper.make_empty_tensor_value_info("y")],
        initializer=initializer,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`drop`" in message
    assert "`mode`" in message
    assert "inference mode" in message


@requires_c_compiler
def test_dropout_pinned_to_inference_passes_its_input_and_mask_through(tmp_path):
    """A `training_mode` the graph fixes to false is compiled as the identity it is."""
    model = _model(
        [helper.make_node("Dropout", ["x", "", "mode"], ["y", "mask"], name="drop")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [
            helper.make_empty_tensor_value_info("y"),
            helper.make_empty_tensor_value_info("mask"),
        ],
        initializer=[onnx.numpy_helper.from_array(np.array(False), "mode")],
    )
    values = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)

    outputs = compile_onnx(model, tmp_path).load().run({"x": values})

    expected = ReferenceEvaluator(model).run(None, {"x": values})
    np.testing.assert_array_equal(outputs["y"], expected[0])
    np.testing.assert_array_equal(outputs["mask"], expected[1])


def test_a_perm_that_is_not_a_permutation_is_rejected(tmp_path):
    """`perm` says where each axis comes from, so it has to name each of them once."""
    model = _model(
        [helper.make_node("Transpose", ["x"], ["y"], name="swap", perm=[0, 0])],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (2, 2))],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`swap`" in message
    assert "[0, 0]" in message


@pytest.mark.parametrize(
    ("label", "starts", "ends", "axes", "steps", "expected"),
    [
        ("zero_step", [0], [2], [0], [0], "steps by 0"),
        ("repeated_axis", [0, 0], [2, 2], [1, 1], [1, 1], "more than once"),
        ("mismatched_bounds", [0, 0], [2], [0], [1], "one of each"),
    ],
)
def test_slice_bounds_onnx_does_not_define_are_rejected(
    tmp_path, label, starts, ends, axes, steps, expected
):
    model = _model(
        [helper.make_node("Slice", ["x", "s", "e", "a", "t"], ["y"], name="cut")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (2, 3))],
        initializer=[
            _int64("s", starts),
            _int64("e", ends),
            _int64("a", axes),
            _int64("t", steps),
        ],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`cut`" in message
    assert expected in message


def test_a_repeat_count_per_axis_is_required(tmp_path):
    """ONNX defines `repeats` as one count per axis; anything else addresses nothing."""
    model = _model(
        [helper.make_node("Tile", ["x", "r"], ["y"], name="repeat")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (4, 3))],
        initializer=[_int64("r", [2])],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`repeat`" in message
    assert "1 repeat count(s)" in message


def test_a_pad_mode_onnx_does_not_define_is_rejected(tmp_path):
    model = _model(
        [helper.make_node("Pad", ["x", "p"], ["y"], name="fill", mode="mirror")],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[_int64("p", [1, 0, 1, 0])],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`fill`" in message
    assert "mirror" in message
    assert "constant, edge, reflect, wrap" in message


@pytest.mark.parametrize("mode", ["edge", "reflect", "wrap"])
def test_padding_an_empty_axis_from_the_operand_is_rejected(tmp_path, mode):
    """Only a constant pad can widen an axis the operand has no values along."""
    model = _model(
        [helper.make_node("Pad", ["x", "p"], ["y"], name="fill", mode=mode)],
        [_tensor("x", TensorProto.FLOAT, (0, 3))],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[_int64("p", [1, 0, 1, 0])],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`fill`" in message
    assert f"`{mode}`" in message
    assert "empty along it" in message


def test_gathering_elements_across_disagreeing_axes_is_rejected(tmp_path):
    """Every axis but the gathered one addresses both operands, so both must measure alike."""
    model = _model(
        [helper.make_node("GatherElements", ["x", "i"], ["y"], name="pick", axis=0)],
        [
            _tensor("x", TensorProto.FLOAT, (3, 4)),
            _tensor("i", TensorProto.INT64, (2, 5)),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`pick`" in message
    assert "[2, 5]" in message and "[3, 4]" in message


@pytest.mark.parametrize(
    ("label", "attributes", "lengths_shape", "expected"),
    [
        ("axes", {"batch_axis": 0, "time_axis": 2}, (2,), "0 and 1 in either order"),
        ("lengths", {"batch_axis": 0, "time_axis": 1}, (4,), "one per batch"),
    ],
)
def test_reversing_what_onnx_does_not_define_is_rejected(
    tmp_path, label, attributes, lengths_shape, expected
):
    model = _model(
        [
            helper.make_node(
                "ReverseSequence", ["x", "l"], ["y"], name="reverse", **attributes
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3, 4)),
            _tensor("l", TensorProto.INT64, lengths_shape),
        ],
        [helper.make_empty_tensor_value_info("y")],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`reverse`" in message
    assert expected in message


def test_an_op_only_folding_serves_is_refused_where_folding_declines(tmp_path):
    """`ConstantOfShape` carries no kernel: a graph that fixes its shape folds it away.

    That rests on the reference evaluator being a valid oracle for the revision, which it is
    only from 25 on. Below it the node survives folding and dispatch refuses it by name,
    rather than a kernel serving semantics nothing can vouch for.
    """
    model = _model(
        [
            helper.make_node(
                "ConstantOfShape",
                ["s"],
                ["y"],
                name="fill",
                value=helper.make_tensor("v", TensorProto.FLOAT, [1], [1.5]),
            )
        ],
        [],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[_int64("s", [2, 3])],
        opset=21,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`ConstantOfShape`" in message
    assert "opset version 21" in message
    assert "no kernel is registered" in message


def test_the_pad_revision_onnx_infers_no_shape_for_names_the_nearest_version(tmp_path):
    """Pad-1 spells its pads `paddings`, and ONNX derives no shape for a node of it.

    A revision whose result the compiler can only take on the model's word is one it cannot
    prove anything about, so dispatch refuses it by name instead of reading the attribute
    under another spelling.
    """
    model = _model(
        [
            helper.make_node(
                "Pad", ["x"], ["y"], name="fill", paddings=[1, 0, 1, 0], mode="constant"
            )
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (4, 3))],
        opset=1,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`Pad`" in message
    assert "opset version 1" in message
    assert "Nearest supported version: 2" in message


@pytest.mark.parametrize(
    ("op_type", "opset", "nearest"),
    [("Tile", 5, 6), ("Concat", 1, 4)],
)
def test_a_view_revision_no_kernel_claims_names_the_nearest_version(
    tmp_path, op_type, opset, nearest
):
    """Tile-1 repeats along one named axis, and Concat-1's `axis` default is unreadable.

    Neither is the op the generator implements, and neither has an oracle to prove one
    against, so dispatch says so rather than serving the current semantics under the old
    revision's name.
    """
    # Tile-1 takes a repeat count and the single axis to apply it to, both as operands of
    # the tensor's own element type; Concat takes any number of operands and an axis.
    tiling = op_type == "Tile"
    model = _model(
        [
            helper.make_node(
                op_type,
                ["x", "n", "a"] if tiling else ["x", "x"],
                ["y"],
                name="view",
                **({} if tiling else {"axis": 0}),
            )
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 3))],
        [_tensor("y", TensorProto.FLOAT, (4, 3))],
        initializer=[
            onnx.numpy_helper.from_array(np.array([value], dtype=np.float32), name)
            for name, value in (("n", 2.0), ("a", 0.0))
        ]
        if tiling
        else [],
        opset=opset,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert f"`{op_type}`" in message
    assert f"opset version {opset}" in message
    assert f"Nearest supported version: {nearest}" in message


# --------------------------------------------------------------------------------------
# The Fourier transforms
# --------------------------------------------------------------------------------------

# What a transform computes is settled by the conformance and differential suites, against
# ONNX's corpus and reference evaluator. What is asserted here is what those cannot reach:
# that the axis, the length and the frame geometry are read at compile time into the
# addressing one shared kernel walks, that an axis named only at run time becomes a switch
# over those call sites where it cannot change the result's shape, and the errors for the
# models the compiler refuses outright.

_DFT_OPSET = 20
_DFT_ATTRIBUTE_OPSET = 19
_STFT_OPSET = 17


def _dft_model(
    shape,
    *,
    result_shape=None,
    inputs=("x",),
    initializer=(),
    opset=_DFT_OPSET,
    elem_type=TensorProto.FLOAT,
    **attributes,
):
    result = (
        helper.make_empty_tensor_value_info("y")
        if result_shape is None
        else _tensor("y", elem_type, result_shape)
    )
    declared = [_tensor("x", elem_type, shape)]
    declared += [
        _tensor(name, TensorProto.INT64, ())
        for name in inputs[1:]
        if name and name not in {entry.name for entry in initializer}
    ]
    return _model(
        [helper.make_node("DFT", list(inputs), ["y"], name="dft", **attributes)],
        declared,
        [result],
        initializer=initializer,
        opset=opset,
    )


def _stft_model(
    shape,
    *,
    inputs=("x", "step"),
    initializer=(),
    window_shape=None,
    **attributes,
):
    declared = [_tensor("x", TensorProto.FLOAT, shape)]
    if window_shape is not None:
        declared.append(_tensor("window", TensorProto.FLOAT, window_shape))
    return _model(
        [helper.make_node("STFT", list(inputs), ["y"], name="stft", **attributes)],
        declared,
        [helper.make_empty_tensor_value_info("y")],
        initializer=initializer,
        opset=_STFT_OPSET,
    )


def test_every_transform_of_one_element_type_shares_a_kernel(tmp_path):
    """Axis, length and mode are addressing, so only the element type reaches the code."""
    model = _model(
        [
            helper.make_node("DFT", ["x", "", "first"], ["p"], name="first"),
            helper.make_node("DFT", ["x", "long", "last"], ["q"], name="padded"),
            helper.make_node("DFT", ["x"], ["r"], name="reversed", inverse=1),
        ],
        [_tensor("x", TensorProto.FLOAT, (2, 4, 2))],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r")],
        initializer=[_int64("first", 0), _int64("last", 1), _int64("long", 6)],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "dft")
    assert header.count(f"static void {kernel}(") == 1
    # The call sites, told from the definition by the indent their arguments carry.
    assert header.count(f"{kernel}(\n        ") == 3


def test_the_transform_reaches_the_kernel_as_call_site_literals(tmp_path):
    """One block per leading coordinate, one stride per trailing one, and the bin count."""
    model = _dft_model(
        (2, 6, 3, 1),
        inputs=("x", "", "axis"),
        initializer=[_int64("axis", 1)],
        onesided=1,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "dft")
    assert (
        f"{kernel}(\n        y,\n        x,\n"
        "        2u,\n"
        "        3u,\n"
        "        6u,\n"
        "        4u,\n"
        "        6u,\n"
        "        1u,\n"
        "        2u,\n"
        "        0,\n"
        "        0);" in header
    )


def test_the_inverse_one_sided_transform_writes_a_real_result(tmp_path):
    """It is the one transform whose result has no imaginary part, and the one that
    mirrors its operand: the length it writes is twice its own extent, less two."""
    model = _dft_model((1, 5, 2), inverse=1, onesided=1)

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "dft")
    assert (
        f"{kernel}(\n        y,\n        x,\n"
        "        1u,\n"
        "        1u,\n"
        "        5u,\n"
        "        8u,\n"
        "        8u,\n"
        "        2u,\n"
        "        1u,\n"
        "        1,\n"
        "        1);" in header
    )


def test_the_two_revisions_read_their_default_axis_differently(tmp_path):
    """Revision 17 states the axis as an attribute defaulting to 1, revision 20 as an
    operand defaulting to the last signal axis; a rank-4 operand tells the two apart."""
    older = _dft_model((2, 3, 4, 1), opset=_DFT_ATTRIBUTE_OPSET)
    newer = _dft_model((2, 3, 4, 1))

    (report, header), (newer_report, newer_header) = (
        _compile(older, tmp_path / "older"),
        _compile(newer, tmp_path / "newer"),
    )

    (kernel,) = _kernels(report, "dft")
    (newer_kernel,) = _kernels(newer_report, "dft")
    # The transformed axis is the operand's, so the blocks before it and the strides after
    # it are what the default moves: axis 1 against axis 2.
    assert f"{kernel}(\n        y,\n        x,\n        2u,\n        4u,\n" in header
    assert (
        f"{newer_kernel}(\n        y,\n        x,\n        6u,\n        1u,\n"
        in newer_header
    )


@requires_c_compiler
def test_an_axis_named_at_run_time_switches_over_the_axes_it_could_name(tmp_path):
    """The result keeps the operand's extents whichever axis is transformed, so every axis
    is a call site of its own and the operand only chooses between them."""
    model = _dft_model((2, 4, 3, 1), inputs=("x", "", "axis"))
    signal = np.arange(24, dtype=np.float32).reshape(2, 4, 3, 1)

    result = compile_onnx(model, tmp_path)
    loaded = result.load()

    (kernel,) = _kernels(result.report, "dft")
    header = result.header_path.read_text(encoding="utf-8")
    prefix = result.report["prefix"].upper()
    assert header.count(f"static void {kernel}(") == 1
    assert (
        f"switch ({result.report['prefix']}_normalized_axis((int64_t)axis[0], 4))"
        in (header)
    )
    assert [f"case {axis}:" in header for axis in range(4)] == [True, True, True, False]
    assert f"default:\n        return {prefix}_ERROR_INVALID_ARGUMENT;" in header
    for axis in (0, 1, 2, -2, -3, -4):
        outputs = loaded.run({"x": signal, "axis": np.array(axis, dtype=np.int64)})
        expected = ReferenceEvaluator(model).run(
            None, {"x": signal, "axis": np.array(axis, dtype=np.int64)}
        )
        np.testing.assert_allclose(outputs["y"], expected[0], rtol=1e-3, atol=1e-6)


@requires_c_compiler
def test_an_axis_outside_the_operands_rank_returns_the_argument_error(tmp_path):
    """The last axis holds the real and imaginary parts, so it is one no transform names."""
    model = _dft_model((2, 4, 3, 1), inputs=("x", "", "axis"))

    loaded = compile_onnx(model, tmp_path).load()

    with pytest.raises(HarnessError, match="status 1"):
        loaded.run(
            {
                "x": np.zeros((2, 4, 3, 1), dtype=np.float32),
                "axis": np.array(3, dtype=np.int64),
            }
        )


@pytest.mark.parametrize(
    ("label", "attributes", "initializer"),
    [
        ("onesided", {"onesided": 1}, []),
        ("dft_length", {}, [_int64("length", 4)]),
    ],
)
def test_an_axis_named_at_run_time_that_resizes_the_result_is_refused(
    tmp_path, label, attributes, initializer
):
    """Both resize the axis they land on, so which axis that is decides the result's shape,
    and no buffer can be sized before the operand is read."""
    model = _dft_model(
        (2, 4, 3, 1),
        inputs=("x", "length" if initializer else "", "axis"),
        initializer=initializer,
        **attributes,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`dft`" in message and "`axis`" in message
    assert "depends on input data" in message


def test_a_transform_length_named_at_run_time_is_refused(tmp_path):
    model = _dft_model((2, 4, 1), inputs=("x", "length"))

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`length`" in str(error.value)
    assert "depends on input data" in str(error.value)


def test_a_transform_over_no_samples_at_all_is_refused(tmp_path):
    """A one-sided transform of an empty axis still states a bin, which nothing defines:
    numpy refuses the transform outright, so there is no such thing to compile."""
    model = _dft_model((2, 0, 1), result_shape=(2, 1, 2), onesided=1)

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "transform length of 0" in str(error.value)


def test_a_transform_of_an_operand_that_is_not_a_signal_is_refused(tmp_path):
    """The last axis is the real and imaginary parts, so it measures 1 or 2 and nothing
    else; ONNX's own inference does not check it."""
    model = _dft_model((2, 4, 3), result_shape=(2, 4, 2))

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "measures 3" in str(error.value)


@requires_c_compiler
def test_a_window_is_read_at_run_time(tmp_path):
    """Its values reach the kernel and nothing else about it does, so a window the graph
    does not fix is a pointer the call site passes rather than a compile error."""
    model = _stft_model(
        (1, 16, 1),
        inputs=("x", "step", "window"),
        initializer=[_int64("step", 4)],
        window_shape=(8,),
    )
    signal = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    window = np.hanning(8).astype(np.float32)

    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"x": signal, "window": window})

    (kernel,) = _kernels(result.report, "stft")
    assert f"{kernel}(\n        y,\n        x,\n        window,\n" in (
        result.header_path.read_text(encoding="utf-8")
    )
    expected = ReferenceEvaluator(model).run(None, {"x": signal, "window": window})
    np.testing.assert_allclose(outputs["y"], expected[0], rtol=1e-3, atol=1e-6)


def test_a_transform_with_no_window_passes_none_in_its_place(tmp_path):
    """Every sample weighs the same, which the kernel reads as no window at all."""
    model = _stft_model(
        (1, 16, 1),
        inputs=("x", "step", "", "length"),
        initializer=[_int64("step", 4), _int64("length", 8)],
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "stft")
    assert (
        f"{kernel}(\n        y,\n        x,\n        NULL,\n"
        "        1u,\n"
        "        16u,\n"
        "        1u,\n"
        "        3u,\n"
        "        4u,\n"
        "        8u,\n"
        "        5u);" in header
    )


def test_the_frame_step_must_be_known_at_compile_time(tmp_path):
    model = _stft_model(
        (1, 16, 1),
        inputs=("x", "step", "", "length"),
        initializer=[_int64("length", 8)],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`step`" in str(error.value)
    assert "depends on input data" in str(error.value)


@pytest.mark.parametrize(
    ("label", "initializer"),
    [
        ("no_frame", [_int64("step", 4)]),
        ("no_step", [_int64("step", 0), _int64("length", 8)]),
    ],
)
def test_a_frame_layout_that_states_no_frames_is_refused(tmp_path, label, initializer):
    """ONNX reads the frame from a window or a `frame_length`, and counts the frames by
    dividing the signal by the step; without the first its own inference stops before a
    shape and with a step of nothing it divides by zero, so neither reaches an artifact."""
    stated = ("x", "step") if label == "no_frame" else ("x", "step", "", "length")
    model = _stft_model((1, 16, 1), inputs=stated, initializer=initializer)

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "`stft`" in str(error.value)


def test_a_signal_shorter_than_one_frame_is_refused(tmp_path):
    """Not one whole frame fits, and ONNX's own two readings of that disagree: its shape
    inference truncates the frame count towards zero where the reference floors it."""
    model = _stft_model(
        (1, 4, 1),
        inputs=("x", "step", "", "length"),
        initializer=[_int64("step", 3), _int64("length", 8)],
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "do not fit a signal of 4" in str(error.value)


# --------------------------------------------------------------------------------------
# The quantization family
# --------------------------------------------------------------------------------------

# What these compute is settled by the conformance and differential suites, against ONNX's
# corpus and reference evaluator. What is asserted here is what those cannot reach: that the
# granularity of a scale — per tensor, per axis, per block — is resolved at compile time into
# the strides one shared kernel walks, that a filter parameter per output channel is addressed
# by channel at a rank the reference evaluator cannot evaluate at all, and the errors for the
# models the compiler refuses outright.

_QUANTIZE_OPSET = 25
_INTEGER_OPSET = 10
_QLINEAR_MATMUL_OPSET = 21


def _quantize_model(
    shape,
    scale_shape,
    *,
    zero_shape=(),
    grid=TensorProto.UINT8,
    opset=_QUANTIZE_OPSET,
    output=None,
    name="quantize",
    **attributes,
):
    declared = [
        _tensor("x", TensorProto.FLOAT, shape),
        _tensor("y_scale", TensorProto.FLOAT, scale_shape),
    ]
    inputs = ["x", "y_scale"]
    if zero_shape is not None:
        declared.append(_tensor("y_zero_point", grid, zero_shape))
        inputs.append("y_zero_point")
    return _model(
        [helper.make_node("QuantizeLinear", inputs, ["y"], name=name, **attributes)],
        declared,
        [
            helper.make_empty_tensor_value_info("y")
            if output is None
            else _tensor("y", output, shape)
        ],
        opset=opset,
    )


def _dequantize_model(
    shape, scale_shape, *, grid=TensorProto.UINT8, opset=_QUANTIZE_OPSET, **attributes
):
    return _model(
        [
            helper.make_node(
                "DequantizeLinear",
                ["x", "x_scale", "x_zero_point"],
                ["y"],
                name="dequantize",
                **attributes,
            )
        ],
        [
            _tensor("x", grid, shape),
            _tensor("x_scale", TensorProto.FLOAT, scale_shape),
            _tensor("x_zero_point", grid, scale_shape),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=opset,
    )


def _conv_integer_model(
    shape,
    filter_shape,
    *,
    zero_shape=(),
    filter_zero_shape=None,
    grid=TensorProto.UINT8,
    **attributes,
):
    declared = [_tensor("x", grid, shape), _tensor("w", grid, filter_shape)]
    inputs = ["x", "w"]
    for name, operand_shape in (("x_zp", zero_shape), ("w_zp", filter_zero_shape)):
        if operand_shape is None:
            break
        declared.append(_tensor(name, grid, operand_shape))
        inputs.append(name)
    return _model(
        [helper.make_node("ConvInteger", inputs, ["y"], name="conv", **attributes)],
        declared,
        [helper.make_empty_tensor_value_info("y")],
        opset=_INTEGER_OPSET,
    )


def _qlinear_conv_model(
    shape,
    filter_shape,
    *,
    filter_scale_shape=(),
    filter_zero_shape=(),
    scale_shape=(),
    grid=TensorProto.UINT8,
    bias=False,
    **attributes,
):
    operands = (
        ("x", grid, shape),
        ("x_scale", TensorProto.FLOAT, scale_shape),
        ("x_zero_point", grid, scale_shape),
        ("w", grid, filter_shape),
        ("w_scale", TensorProto.FLOAT, filter_scale_shape),
        ("w_zero_point", grid, filter_zero_shape),
        ("y_scale", TensorProto.FLOAT, scale_shape),
        ("y_zero_point", grid, scale_shape),
    )
    declared = [_tensor(*operand) for operand in operands]
    if bias:
        declared.append(_tensor("b", TensorProto.INT32, (filter_shape[0],)))
    return _model(
        [
            helper.make_node(
                "QLinearConv",
                [entry.name for entry in declared],
                ["y"],
                name="qconv",
                **attributes,
            )
        ],
        declared,
        [helper.make_empty_tensor_value_info("y")],
        opset=_INTEGER_OPSET,
    )


def _qlinear_matmul_model(left_shape, right_shape, *, parameter_shape=()):
    operands = (
        ("a", TensorProto.UINT8, left_shape),
        ("a_scale", TensorProto.FLOAT, parameter_shape),
        ("a_zero_point", TensorProto.UINT8, parameter_shape),
        ("b", TensorProto.UINT8, right_shape),
        ("b_scale", TensorProto.FLOAT, parameter_shape),
        ("b_zero_point", TensorProto.UINT8, parameter_shape),
        ("y_scale", TensorProto.FLOAT, parameter_shape),
        ("y_zero_point", TensorProto.UINT8, parameter_shape),
    )
    declared = [_tensor(*operand) for operand in operands]
    return _model(
        [
            helper.make_node(
                "QLinearMatMul",
                [entry.name for entry in declared],
                ["y"],
                name="qmatmul",
            )
        ],
        declared,
        [helper.make_empty_tensor_value_info("y")],
        opset=_QLINEAR_MATMUL_OPSET,
    )


@pytest.mark.parametrize(
    ("label", "kwargs", "block", "strides"),
    [
        ("per_tensor", {"scale_shape": ()}, "        -1,\n        1u,", "{0u, 0u}"),
        (
            "single_element",
            {"scale_shape": (1,)},
            "        -1,\n        1u,",
            "{0u, 0u}",
        ),
        ("per_axis", {"scale_shape": (8,)}, "        -1,\n        1u,", "{0u, 1u}"),
        (
            "per_axis_first",
            {"scale_shape": (4,), "axis": 0},
            "        -1,\n        1u,",
            "{1u, 0u}",
        ),
        (
            "blocked",
            {"scale_shape": (4, 2), "axis": 1, "block_size": 4},
            "        1,\n        4u,",
            "{2u, 1u}",
        ),
        (
            "blocked_first_axis",
            {"scale_shape": (2, 8), "axis": 0, "block_size": 2},
            "        0,\n        2u,",
            "{8u, 1u}",
        ),
    ],
)
def test_the_granularity_of_a_scale_reaches_the_kernel_as_call_site_literals(
    tmp_path, label, kwargs, block, strides
):
    """All three granularities are one addressing: a stride per axis, and a divisor on the
    axis a blocked scale repeats along. Which one a node states is settled at compile time."""
    scale_shape = kwargs.pop("scale_shape")
    model = _quantize_model((4, 8), scale_shape, zero_shape=scale_shape, **kwargs)

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "quantizelinear")
    assert (
        f"{kernel}(\n        y,\n        x,\n        y_scale,\n        y_zero_point,\n"
        "        32u,\n        2,\n        (const size_t[]){4u, 8u},\n"
        f"{block}\n"
        f"        (const size_t[]){strides},\n"
        f"        (const size_t[]){strides});" in header
    )


def test_maps_of_one_grid_and_precision_share_a_kernel(tmp_path):
    """Granularity is addressing, so only the types the kernel reads and writes reach the
    code — and whether a zero point shifts the grid at all, which is a formula of its own."""
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["x", "s", "z"], ["p"], name="tensor"),
            helper.make_node(
                "QuantizeLinear", ["x", "v", "w"], ["q"], name="axis", axis=1
            ),
            helper.make_node("QuantizeLinear", ["x", "s"], ["r"], name="bare"),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (4, 8)),
            _tensor("s", TensorProto.FLOAT, ()),
            _tensor("z", TensorProto.UINT8, ()),
            _tensor("v", TensorProto.FLOAT, (8,)),
            _tensor("w", TensorProto.UINT8, (8,)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("p", "q", "r")],
        opset=_QUANTIZE_OPSET,
    )

    report, header = _compile(model, tmp_path)

    # The one that reads a zero point carries a suffix saying so, and so is the longer name.
    bare, shifted = sorted(_kernels(report, "quantizelinear"), key=len)
    assert header.count(f"static void {bare}(") == 1
    assert header.count(f"static void {shifted}(") == 1
    # The two that state a zero point share one definition and call it from both sites.
    assert header.count(f"{shifted}(\n        ") == 2
    assert f"{bare}(\n        r,\n        x,\n        s,\n        NULL," in header


def test_the_rounding_store_is_one_helper_shared_across_the_family(tmp_path):
    """Every op that writes a grid saturates onto it the same way, so the store is one
    function per grid rather than one per op."""
    model = _model(
        [
            helper.make_node("QuantizeLinear", ["x", "s", "z"], ["a"], name="quantize"),
            helper.make_node(
                "QLinearMatMul",
                ["a", "s", "z", "b", "s", "z", "s", "z"],
                ["y"],
                name="qmatmul",
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (4, 8)),
            _tensor("s", TensorProto.FLOAT, ()),
            _tensor("z", TensorProto.UINT8, ()),
            _tensor("b", TensorProto.UINT8, (8, 4)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=_QLINEAR_MATMUL_OPSET,
    )

    report, header = _compile(model, tmp_path)

    (saturate,) = [name for name in report["kernels"] if "saturate" in name]
    assert saturate.endswith("_uint8_t")
    assert header.count(f"static uint8_t {saturate}(") == 1
    assert header.count(f"{saturate}(") == 3
    assert "return (uint8_t)rint(value);" in header


@pytest.mark.parametrize(
    ("filter_scale_shape", "filter_zero_shape", "scale_stride", "zero_stride"),
    [((), (), "0u", "0u"), ((2,), (2,), "1u", "1u"), ((2,), (), "1u", "0u")],
)
def test_a_filter_parameter_per_output_channel_reaches_the_kernel_as_a_stride(
    tmp_path, filter_scale_shape, filter_zero_shape, scale_stride, zero_stride
):
    """A filter's scale and zero point are one for the whole filter or one per output
    channel, which is the difference between a stride of zero and a stride of one."""
    model = _qlinear_conv_model(
        (1, 1, 5, 5),
        (2, 1, 2, 2),
        filter_scale_shape=filter_scale_shape,
        filter_zero_shape=filter_zero_shape,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "qlinearconv")
    assert (
        f"{kernel}(\n        y,\n        x,\n        w,\n        x_zero_point,\n"
        f"        w_zero_point,\n        {zero_stride},\n        x_scale,\n"
        f"        w_scale,\n        {scale_stride},\n        y_scale,\n"
        "        y_zero_point,\n        NULL," in header
    )


@requires_c_compiler
def test_a_filter_zero_point_per_channel_is_read_by_channel_at_any_rank(tmp_path):
    """ONNX defines it as one zero point per output channel, whatever the filter's rank.

    Its reference evaluator stretches the operand over four axes regardless, so it can
    evaluate a per-channel zero point at two spatial axes and nowhere else — which is what
    the differential sweep is left covering. Splitting the filter here and running each
    output channel through the evaluator with a zero point of its own puts the one-spatial-
    axis case back within reach of the same oracle.
    """
    shape, filter_shape = (2, 2, 7), (3, 2, 3)
    generator = np.random.default_rng(20260726)
    x = generator.integers(0, 256, size=shape, dtype=np.uint8)
    w = generator.integers(0, 256, size=filter_shape, dtype=np.uint8)
    x_zero = np.uint8(37)
    w_zero = np.array([3, 130, 255], np.uint8)
    compiled = compile_onnx(
        _conv_integer_model(shape, filter_shape, filter_zero_shape=(3,)), tmp_path
    ).load()

    got = compiled.run({"x": x, "w": w, "x_zp": x_zero, "w_zp": w_zero})["y"]

    expected = np.concatenate(
        [
            ReferenceEvaluator(
                _conv_integer_model(shape, (1, *filter_shape[1:]), filter_zero_shape=())
            ).run(
                None,
                {
                    "x": x,
                    "w": w[channel : channel + 1],
                    "x_zp": x_zero,
                    "w_zp": w_zero[channel],
                },
            )[0]
            for channel in range(filter_shape[0])
        ],
        axis=1,
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision": TensorProto.DOUBLE}, "states `precision` `DOUBLE`"),
        (
            {"scale_shape": (4, 2)},
            "with no `block_size`, ONNX defines a scale of more than one element",
        ),
        (
            {"scale_shape": (4, 3), "axis": 1, "block_size": 4},
            "takes a scale of [4, 2], but `y_scale` has shape [4, 3]",
        ),
        (
            {"scale_shape": (4, 2), "axis": 1, "block_size": 0 - 2},
            "states `block_size` -2",
        ),
        ({"scale_shape": (4,)}, "does not broadcast to [4, 8]"),
        ({"scale_shape": (8,), "axis": 3}, "axis 3 is out of range"),
    ],
)
def test_a_granularity_the_compiler_cannot_address_is_rejected(
    tmp_path, kwargs, message
):
    scale_shape = kwargs.pop("scale_shape", ())
    model = _quantize_model((4, 8), scale_shape, zero_shape=scale_shape, **kwargs)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_a_zero_point_of_another_shape_than_its_scale_is_rejected(tmp_path):
    """The two together fix the granularity, so ONNX defines them as one shape; a pair that
    disagrees would be read at two granularities at once."""
    model = _quantize_model((4, 8), (8,), zero_shape=(4,))

    with pytest.raises(CompileError, match="ONNX defines the two as one shape"):
        compile_onnx(model, tmp_path)


def test_a_zero_point_off_the_grid_it_shifts_is_rejected(tmp_path):
    """`y_zero_point` is what states the grid, so a result declared as another type is two
    answers to what this node quantizes onto."""
    model = _quantize_model((4, 8), (), output=TensorProto.INT16)

    with pytest.raises(CompileError, match="ONNX defines the two as one type"):
        compile_onnx(model, tmp_path)


def test_quantizing_onto_a_type_that_is_no_grid_is_rejected(tmp_path):
    """The saturation range is the grid's own, so there is none to round onto here."""
    model = _quantize_model((4, 8), (), zero_shape=None, output=TensorProto.INT32)

    with pytest.raises(CompileError, match="quantizes onto the integer grids"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("parameter_shape", "message"),
    [((2, 1), "[2, 1]"), ((2,), "[2]")],
)
def test_a_matrix_product_quantized_per_row_is_rejected(
    tmp_path, parameter_shape, message
):
    """ONNX reads these per row of `A` and per column of `B`. In the form its own text
    describes — an `M`-element vector against an `[M, K]` operand — the reference evaluator
    stretches that vector along numpy's trailing axis instead, so nothing can vouch for what
    a kernel should compute; the granularity goes unserved as a whole rather than be read one
    way there and another in the `[M, 1]` form numpy does broadcast as written."""
    model = _qlinear_matmul_model((2, 4), (4, 3), parameter_shape=parameter_shape)

    with pytest.raises(CompileError, match=re.escape(message)) as error:
        compile_onnx(model, tmp_path)

    assert "per-tensor granularity only" in str(error.value)


def test_a_filter_parameter_of_neither_granularity_is_rejected(tmp_path):
    model = _qlinear_conv_model(
        (1, 1, 5, 5), (2, 1, 2, 2), filter_scale_shape=(3,), filter_zero_shape=(2,)
    )

    with pytest.raises(CompileError, match=re.escape("a 1-D tensor of 2")):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("op_type", "nearest"), [("QuantizeLinear", 10), ("DequantizeLinear", 19)]
)
def test_a_map_below_its_supported_revision_names_the_nearest_version(
    tmp_path, op_type, nearest
):
    """Opset 13 revised both, and no oracle covers that revision: the reference evaluator
    does not distinguish it and no corpus test imports it, so it is not claimed at all."""
    model = (
        _quantize_model((4, 8), (), opset=13)
        if op_type == "QuantizeLinear"
        else _dequantize_model((4, 8), (), opset=13)
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert f"`{op_type}`" in str(error.value)
    assert f"Nearest supported version: {nearest}." in str(error.value)


@requires_c_compiler
def test_a_quantized_model_stays_on_the_grid_between_its_ops(tmp_path):
    """The pipeline these ops exist for: quantize, convolve on the grid, read it back.

    Every tensor between the first op and the last is an integer one, which is the whole
    point of compiling a quantized model — nothing widens back to float in between — and the
    result is the reference evaluator's for the same chain.
    """
    shape, filter_shape = (1, 1, 5, 5), (2, 1, 3, 3)
    parameters = (
        ("x_scale", TensorProto.FLOAT, ()),
        ("x_zero_point", TensorProto.UINT8, ()),
        ("w", TensorProto.UINT8, filter_shape),
        ("w_scale", TensorProto.FLOAT, (2,)),
        ("w_zero_point", TensorProto.UINT8, (2,)),
        ("y_scale", TensorProto.FLOAT, ()),
        ("y_zero_point", TensorProto.UINT8, ()),
    )
    model = _model(
        [
            helper.make_node(
                "QuantizeLinear",
                ["x", "x_scale", "x_zero_point"],
                ["q"],
                name="quantize",
            ),
            helper.make_node(
                "QLinearConv",
                [
                    "q",
                    "x_scale",
                    "x_zero_point",
                    *[name for name, _, _ in parameters[2:]],
                ],
                ["p"],
                name="qconv",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["p", "y_scale", "y_zero_point"],
                ["y"],
                name="dequantize",
            ),
        ],
        [_tensor("x", TensorProto.FLOAT, shape), *(_tensor(*p) for p in parameters)],
        [helper.make_empty_tensor_value_info("y")],
        opset=_QUANTIZE_OPSET,
    )
    generator = np.random.default_rng(20260726)
    feeds = {
        "x": generator.normal(size=shape).astype(np.float32),
        "x_scale": np.float32(0.017),
        "x_zero_point": np.uint8(128),
        "w": generator.integers(0, 256, size=filter_shape, dtype=np.uint8),
        "w_scale": np.array([0.011, 0.023], np.float32),
        "w_zero_point": np.array([127, 130], np.uint8),
        "y_scale": np.float32(0.09),
        "y_zero_point": np.uint8(64),
    }
    result = compile_onnx(model, tmp_path)

    got = result.load().run(feeds)["y"]

    header = result.header_path.read_text(encoding="utf-8")
    assert "static uint8_t" in header
    assert not re.search(r"^static (float|double)", header, re.MULTILINE)
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-7)


# --------------------------------------------------------------------------------------
# The normalization by a root mean square, and the cross-entropy loss
# --------------------------------------------------------------------------------------

_RMS_OPSET = 23
_SCE_OPSET = 13


def _rms_model(shape=(2, 3), *, opset=_RMS_OPSET, **attributes):
    return _model(
        [
            helper.make_node(
                "RMSNormalization", ["x", "s"], ["y"], name="rms", **attributes
            )
        ],
        [
            _tensor("x", TensorProto.FLOAT, shape),
            _tensor("s", TensorProto.FLOAT, shape[-1:]),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=opset,
    )


def _sce_model(
    scores_shape=(3, 5),
    *,
    weighted=False,
    log_prob=False,
    labels_type=TensorProto.INT64,
    **attributes,
):
    inputs = ["scores", "labels"] + (["weights"] if weighted else [])
    outputs = ["loss"] + (["log_prob"] if log_prob else [])
    labels_shape = (scores_shape[0], *scores_shape[2:])
    return _model(
        [
            helper.make_node(
                "SoftmaxCrossEntropyLoss", inputs, outputs, name="sce", **attributes
            )
        ],
        [
            _tensor("scores", TensorProto.FLOAT, scores_shape),
            _tensor("labels", labels_type, labels_shape),
            *(
                [_tensor("weights", TensorProto.FLOAT, scores_shape[1:2])]
                if weighted
                else []
            ),
        ],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        opset=_SCE_OPSET,
    )


def test_a_stash_type_the_reference_refuses_is_a_compile_error(tmp_path):
    """The reference evaluator computes RMSNormalization in the data's own type and raises
    on any `stash_type` but its default, so nothing vouches for another one."""
    with pytest.raises(CompileError, match="stash_type"):
        compile_onnx(_rms_model(stash_type=TensorProto.DOUBLE), tmp_path)


@requires_c_compiler
def test_nodes_normalizing_at_one_element_type_share_one_kernel(tmp_path):
    """The extents and the epsilon are call-site literals, so the axis does not fork it."""
    model = _model(
        [
            helper.make_node(
                "RMSNormalization", ["x", "s"], ["h"], name="first", axis=1
            ),
            helper.make_node(
                "RMSNormalization", ["h", "s"], ["y"], name="second", axis=-1
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (2, 3)),
            _tensor("s", TensorProto.FLOAT, (3,)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=_RMS_OPSET,
    )

    report, header = _compile(model, tmp_path)

    assert len(_kernels(report, "rmsnormalization")) == 1
    assert header.count("static void kernels_rmsnormalization") == 1


@requires_c_compiler
@pytest.mark.parametrize(
    ("shape", "weighted", "log_prob", "attributes"),
    [
        ((3, 5), False, False, {}),
        ((3, 5), True, True, {"reduction": "sum"}),
        ((3, 5, 2), True, False, {"reduction": "none", "ignore_index": -1}),
        ((2, 3, 2, 2), False, True, {"ignore_index": 1}),
    ],
)
def test_the_loss_matches_the_reference_across_its_operand_combinations(
    tmp_path, shape, weighted, log_prob, attributes
):
    model = _sce_model(shape, weighted=weighted, log_prob=log_prob, **attributes)
    generator = np.random.default_rng(20260726)
    labels_shape = (shape[0], *shape[2:])
    feeds = {
        "scores": generator.normal(size=shape).astype(np.float32),
        "labels": generator.integers(-1, shape[1], size=labels_shape).astype(np.int64),
    }
    if weighted:
        feeds["weights"] = np.abs(generator.normal(size=shape[1])).astype(np.float32)
    # A label ONNX does not ignore has to name a class; the draw above reaches -1, which
    # only the variants naming it as `ignore_index` may see.
    if attributes.get("ignore_index") != -1:
        feeds["labels"] = np.abs(feeds["labels"]) % shape[1]

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    expected = ReferenceEvaluator(model).run(None, feeds)
    for name, reference in zip(["loss", "log_prob"], expected):
        np.testing.assert_allclose(outputs[name], reference, rtol=1e-3, atol=1e-7)


@requires_c_compiler
def test_a_label_outside_the_class_axis_is_an_argument_error(tmp_path):
    """The reference raises on one; the artifact reads nothing and reports the status."""
    compiled = compile_onnx(_sce_model(), tmp_path).load()

    with pytest.raises(HarnessError, match="status 1"):
        compiled.run(
            {
                "scores": np.zeros((3, 5), dtype=np.float32),
                "labels": np.array([0, 5, 1], dtype=np.int64),
            }
        )


@requires_c_compiler
def test_a_label_the_node_ignores_may_sit_outside_the_class_axis(tmp_path):
    """`ignore_index` is checked first, so the entry it names is skipped rather than refused."""
    model = _sce_model(ignore_index=-1)
    feeds = {
        "scores": np.arange(15, dtype=np.float32).reshape(3, 5),
        "labels": np.array([0, -1, 4], dtype=np.int64),
    }

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    expected = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_allclose(outputs["loss"], expected[0], rtol=1e-3, atol=1e-7)


@requires_c_compiler
def test_the_unweighted_loss_takes_no_weight_operand(tmp_path):
    """A parameter the kernel never reads is what the artifact's `-Werror` build refuses,
    so the operand combination is part of the kernel's identity rather than a branch."""
    plain, _ = _compile(_sce_model(), tmp_path / "plain")
    weighted, header = _compile(_sce_model(weighted=True), tmp_path / "weighted")

    (plain_kernel,) = _kernels(plain, "softmaxcrossentropyloss")
    (weighted_kernel,) = _kernels(weighted, "softmaxcrossentropyloss")
    assert plain_kernel != weighted_kernel
    assert "const float* weights" in header


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reduction": "median"}, "reduction"),
        ({"scores_shape": (5,)}, "rank of at least 2"),
    ],
)
def test_the_loss_refuses_what_it_cannot_group(tmp_path, kwargs, message):
    with pytest.raises(CompileError, match=message):
        compile_onnx(_sce_model(**kwargs), tmp_path)


# --------------------------------------------------------------------------------------
# LinearAttention
# --------------------------------------------------------------------------------------

_LINEAR_ATTENTION_OPSET = 27

# The operands in schema order; the ones a node leaves out reach it as empty names.
_LINEAR_ATTENTION_INPUTS = ("query", "key", "value", "past_state", "decay", "beta")

# Where the call site carries what, in the kernel's own parameter order: eight pointers,
# seven extents, then the strides that place each of the two optional gates and the scale.
_LINEAR_ATTENTION_DECAY_STRIDES = slice(15, 18)
_LINEAR_ATTENTION_BETA_STRIDES = slice(18, 20)


def _linear_attention_operands(
    *,
    batch=2,
    steps=3,
    q_heads=2,
    kv_heads=2,
    d_k=3,
    d_v=2,
    past=False,
    decay=None,
    beta=None,
):
    """The shape of every operand a node passes, by name, in schema order.

    Every 3-D operand packs `H * D` into its last axis, so the shapes follow from the two
    head counts and the two head widths. `decay` and `beta` name the granularity ONNX packs
    each of them at -- one value per key dimension or one per head for the decay, one per
    head or one the heads share for beta -- and leaving either out is how `update_rule`
    reaches a node: the rule forbids the operand it does not read.
    """
    operands = {
        "query": (batch, steps, q_heads * d_k),
        "key": (batch, steps, kv_heads * d_k),
        "value": (batch, steps, kv_heads * d_v),
    }
    if past:
        operands["past_state"] = (batch, kv_heads, d_k, d_v)
    if decay is not None:
        operands["decay"] = (
            batch,
            steps,
            kv_heads * (1 if decay == "per_head" else d_k),
        )
    if beta is not None:
        operands["beta"] = (batch, steps, kv_heads if beta == "per_head" else 1)
    return operands


def _linear_attention_model(
    *, q_heads=2, kv_heads=2, state_shape=None, attributes=None, **geometry
):
    """One LinearAttention node, with `state_shape` declaring its second output by hand."""
    operands = _linear_attention_operands(
        q_heads=q_heads, kv_heads=kv_heads, **geometry
    )
    names = [name if name in operands else "" for name in _LINEAR_ATTENTION_INPUTS]
    while names and not names[-1]:
        names.pop()
    node = helper.make_node(
        "LinearAttention",
        names,
        ["output", "present_state"],
        name="attn",
        q_num_heads=q_heads,
        kv_num_heads=kv_heads,
        **(attributes or {}),
    )
    state = (
        helper.make_empty_tensor_value_info("present_state")
        if state_shape is None
        else _tensor("present_state", TensorProto.FLOAT, state_shape)
    )
    return _model(
        [node],
        [_tensor(name, TensorProto.FLOAT, shape) for name, shape in operands.items()],
        [helper.make_empty_tensor_value_info("output"), state],
        opset=_LINEAR_ATTENTION_OPSET,
    )


def _linear_attention_feeds(model, seed=20260726):
    return {
        entry.name: np.random.default_rng([seed, index])
        .normal(size=[dim.dim_value for dim in entry.type.tensor_type.shape.dim])
        .astype(np.float32)
        for index, entry in enumerate(model.graph.input)
    }


def _linear_attention_emission(operands, results, **attributes):
    """The registered kernel generator, run on one node directly.

    ONNX's own shape inference vets this op thoroughly: it rejects every operand combination
    the reference evaluator refuses, and does so before dispatch ever asks for a kernel, so a
    model is no way to reach the generator's own refusals. They are the compiler's last word
    for a node that arrives without inference having vetted it, and this is where they are
    read.
    """
    node = helper.make_node(
        "LinearAttention",
        [name if shape is not None else "" for name, shape in operands],
        [name for name, _ in results],
        name="attn",
        **{"q_num_heads": 2, "kv_num_heads": 2, **attributes},
    )
    spec = KERNELS.select("", "LinearAttention", _LINEAR_ATTENTION_OPSET)
    context = NodeContext(
        node=node,
        domain="",
        opset_version=_LINEAR_ATTENTION_OPSET,
        since_version=spec.since_version,
        prefix="attn",
        inputs=tuple(
            None if shape is None else TensorRef(name, TensorProto.FLOAT, shape, name)
            for name, shape in operands
        ),
        outputs=tuple(
            TensorRef(name, TensorProto.FLOAT, shape, name) for name, shape in results
        ),
    )
    return spec.generator(context)


def _linear_attention_node(*, decay=None, beta=None):
    """One node's operands and results as `_linear_attention_emission` takes them."""
    shapes = _linear_attention_operands(decay=decay, beta=beta)
    return (
        [(name, shapes.get(name)) for name in _LINEAR_ATTENTION_INPUTS],
        [("output", (2, 3, 4)), ("present_state", (2, 2, 3, 2))],
    )


def _linear_attention_call(header, kernel, index=0):
    """One emitted call site's arguments, in order.

    The kernel's own definition opens the same way, so the split's first piece is its
    parameter list and the call sites follow.
    """
    body = header.split(f"{kernel}(\n")[index + 2].split(");")[0]
    return [line.strip().rstrip(",") for line in body.splitlines()]


# Every combination of a rule and the two optional gates ONNX refuses, with the operand its
# refusal is about: `gated` reads the decay and forbids beta, `delta` the other way round,
# `gated_delta` reads both and `linear` neither. The reference raises for a stray operand as
# readily as for a missing one, so there is nothing for a kernel to compute for any of these.
_LINEAR_ATTENTION_MISMATCHES = (
    ("linear", "per_key_dim", None, "decay"),
    ("linear", None, "per_head", "beta"),
    ("gated", None, None, "decay"),
    ("gated", "per_key_dim", "per_head", "beta"),
    ("delta", "per_key_dim", "per_head", "decay"),
    ("delta", None, None, "beta"),
    ("gated_delta", None, "per_head", "decay"),
    ("gated_delta", "per_key_dim", None, "beta"),
)


@pytest.mark.parametrize(
    ("rule", "decay", "beta", "operand"), _LINEAR_ATTENTION_MISMATCHES
)
def test_a_rule_and_the_gates_it_reads_have_to_agree(
    tmp_path, rule, decay, beta, operand
):
    """The compiler refuses exactly the models the reference does, and for the same reason."""
    model = _linear_attention_model(
        decay=decay, beta=beta, attributes={"update_rule": rule}
    )

    with pytest.raises(CompileError):
        compile_onnx(model, tmp_path)

    with pytest.raises(ValueError, match=f"'{rule}' (requires|forbids) {operand}"):
        ReferenceEvaluator(model).run(None, _linear_attention_feeds(model))


@pytest.mark.parametrize(
    ("rule", "decay", "beta", "operand"), _LINEAR_ATTENTION_MISMATCHES
)
def test_the_kernel_refuses_a_rule_its_operands_contradict(rule, decay, beta, operand):
    """The same refusal as the kernel's own, naming the rule and the operand it is about."""
    operands, results = _linear_attention_node(decay=decay, beta=beta)

    with pytest.raises(CompileError, match=f"`{rule}`") as error:
        _linear_attention_emission(operands, results, update_rule=rule)

    assert f"`{operand}`" in str(error.value)


def test_an_update_rule_onnx_does_not_define_is_refused(tmp_path):
    """The four recurrences are the whole of what the attribute may name."""
    model = _linear_attention_model(
        decay="per_key_dim", beta="per_head", attributes={"update_rule": "chunked"}
    )
    operands, results = _linear_attention_node(decay="per_key_dim", beta="per_head")

    with pytest.raises(CompileError):
        compile_onnx(model, tmp_path)
    with pytest.raises(ValueError, match="chunked"):
        ReferenceEvaluator(model).run(None, _linear_attention_feeds(model))

    with pytest.raises(CompileError, match="`chunked`") as error:
        _linear_attention_emission(operands, results, update_rule="chunked")

    assert "linear, gated, delta, gated_delta" in str(error.value)


def test_the_op_is_served_by_a_kernel_rather_than_by_its_function_body(tmp_path):
    """Why this op has a kernel at all, written down.

    ONNX defines a function body for `LinearAttention` and the compiler prefers one over a
    kernel wherever it can -- but this body drives the recurrence with a `Scan` over the
    sequence, whose trip count is a run-time tensor rather than anything constant folding can
    resolve, which puts it on the v1 unsupported surface. The registry claiming the op is
    what keeps every model of it off that path.
    """
    assert onnx.defs.get_schema(
        "LinearAttention", _LINEAR_ATTENTION_OPSET, ""
    ).has_context_dependent_function
    assert KERNELS.registered_versions("", "LinearAttention") == [
        _LINEAR_ATTENTION_OPSET
    ]

    report, _ = _compile(
        _linear_attention_model(decay="per_key_dim", beta="per_head"), tmp_path
    )

    assert _kernels(report, "linearattention")


def test_linear_attention_nodes_of_one_element_type_share_a_kernel(tmp_path):
    """Batch, sequence, head counts and head widths are call-site literals, not kernels."""
    model = _model(
        [
            helper.make_node(
                "LinearAttention",
                ["q", "k", "v"],
                ["y", "s"],
                name="wide",
                q_num_heads=2,
                kv_num_heads=2,
                update_rule="linear",
            ),
            helper.make_node(
                "LinearAttention",
                ["q2", "k2", "v2"],
                ["y2", "s2"],
                name="narrow",
                q_num_heads=3,
                kv_num_heads=1,
                update_rule="linear",
            ),
        ],
        [
            _tensor("q", TensorProto.FLOAT, (2, 3, 6)),
            _tensor("k", TensorProto.FLOAT, (2, 3, 6)),
            _tensor("v", TensorProto.FLOAT, (2, 3, 4)),
            _tensor("q2", TensorProto.FLOAT, (1, 5, 6)),
            _tensor("k2", TensorProto.FLOAT, (1, 5, 2)),
            _tensor("v2", TensorProto.FLOAT, (1, 5, 7)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "s", "y2", "s2")],
        opset=_LINEAR_ATTENTION_OPSET,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "linearattention")
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_the_running_state_is_the_present_state_buffer(tmp_path):
    """`present_state` is a required output, so the recurrence needs no storage of its own:
    it updates that buffer in place and the sequence ends with the answer already there."""
    model = _linear_attention_model(decay="per_key_dim", beta="per_head", past=True)

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "linearattention")
    assert _linear_attention_call(header, kernel)[:3] == [
        "output",
        "present_state",
        "query",
    ]
    assert f"static float {report['prefix']}_linearattention" not in header


@pytest.mark.parametrize(
    ("decay", "strides"),
    [("per_head", ["2u", "1u", "0u"]), ("per_key_dim", ["6u", "3u", "1u"])],
)
def test_the_decay_granularity_reaches_the_kernel_as_strides(tmp_path, decay, strides):
    """Which granularity the gate is packed at is where its elements sit, not what the
    kernel computes: the per-head packing reaches every key dimension through a zero
    stride."""
    model = _linear_attention_model(decay=decay, attributes={"update_rule": "gated"})

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "linearattention")
    arguments = _linear_attention_call(header, kernel)
    assert arguments[_LINEAR_ATTENTION_DECAY_STRIDES] == strides


@pytest.mark.parametrize(
    ("beta", "strides"), [("per_head", ["2u", "1u"]), ("shared", ["1u", "0u"])]
)
def test_the_beta_granularity_reaches_the_kernel_as_strides(tmp_path, beta, strides):
    """A beta the heads share is one the head axis is addressed with a zero stride."""
    model = _linear_attention_model(beta=beta, attributes={"update_rule": "delta"})

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "linearattention")
    arguments = _linear_attention_call(header, kernel)
    assert arguments[_LINEAR_ATTENTION_BETA_STRIDES] == strides


def test_the_default_scale_is_the_one_the_schema_derives(tmp_path):
    """A `scale` of 0 -- the attribute's own default -- asks for `1/sqrt(d_k)`, which no
    model could mean literally: a zero factor answers every query with zero. The derived
    factor and the same number stated outright have to reach the kernel alike."""
    geometry = {"d_k": 4, "decay": "per_key_dim", "beta": "per_head"}
    report, derived = _compile(
        _linear_attention_model(**geometry), tmp_path / "derived"
    )
    _, stated = _compile(
        _linear_attention_model(attributes={"scale": 0.5}, **geometry),
        tmp_path / "stated",
    )

    (kernel,) = _kernels(report, "linearattention")
    assert _linear_attention_call(derived, kernel)[-1] != "0.0"
    assert _linear_attention_call(derived, kernel) == _linear_attention_call(
        stated, kernel
    )


def test_a_state_output_shaped_otherwise_is_refused(tmp_path):
    """The state's shape follows from the head counts and the two head widths; a graph that
    declares another one describes a different op, and this is where that stops rather than
    where the kernel writes past the buffer."""
    model = _linear_attention_model(
        decay="per_key_dim", beta="per_head", state_shape=(2, 2, 2, 3)
    )

    with pytest.raises(CompileError, match=re.escape("[2, 2, 3, 2]")) as error:
        compile_onnx(model, tmp_path)

    assert "present_state" in str(error.value)


@requires_c_compiler
def test_a_sequence_decoded_one_token_at_a_time_matches_the_prefill(tmp_path):
    """What the two state operands exist for: a prefill over a whole sequence and one decode
    step per token, each handed the state the last one reported, are the same recurrence.

    Neither suite reaches this -- the sweep and the corpus both run a single node once -- and
    the expected values are the reference evaluator's, on the prefill the chain is supposed
    to equal.
    """
    steps = 4
    packing = {"decay": "per_key_dim", "beta": "per_head", "past": True}
    prefill = _linear_attention_model(steps=steps, **packing)
    feeds = _linear_attention_feeds(prefill)
    expected = ReferenceEvaluator(prefill).run(None, feeds)

    compiled = compile_onnx(
        _linear_attention_model(steps=1, **packing), tmp_path
    ).load()
    state = feeds["past_state"]
    answers = []
    for step in range(steps):
        result = compiled.run(
            {
                name: state if name == "past_state" else feeds[name][:, step : step + 1]
                for name in feeds
            }
        )
        answers.append(result["output"])
        state = result["present_state"]

    np.testing.assert_allclose(
        np.concatenate(answers, axis=1), expected[0], rtol=1e-3, atol=1e-6
    )
    np.testing.assert_allclose(state, expected[1], rtol=1e-3, atol=1e-6)


# --------------------------------------------------------------------------------------
# Attention and RotaryEmbedding
# --------------------------------------------------------------------------------------

_TRANSFORMER_OPSET = 24

# The operands and the results in schema order; the ones a node leaves out reach it as
# empty names, and the ones it does not ask for are simply not there.
_ATTENTION_INPUTS = (
    "Q",
    "K",
    "V",
    "attn_mask",
    "past_key",
    "past_value",
    "nonpad_kv_seqlen",
)
_ATTENTION_OUTPUTS = ("Y", "present_key", "present_value", "qk_matmul_output")
_ROTARY_INPUTS = ("X", "cos_cache", "sin_cache", "position_ids")

# Attention emits its scorer and two helpers under one `<prefix>_attention_` family; these
# are the tokens that tell the helpers apart from the scorer itself.
_ATTENTION_HELPERS = ("bias", "present")

_ATTENTION_OPERANDS = {
    "Q": (TensorProto.FLOAT, (1, 2, 2, 4)),
    "K": (TensorProto.FLOAT, (1, 2, 3, 4)),
    "V": (TensorProto.FLOAT, (1, 2, 3, 4)),
}

_ROTARY_OPERANDS = {
    "X": (TensorProto.FLOAT, (1, 2, 3, 4)),
    "cos_cache": (TensorProto.FLOAT, (5, 2)),
    "sin_cache": (TensorProto.FLOAT, (5, 2)),
    "position_ids": (TensorProto.INT64, (1, 3)),
}


def _transformer_model(op_type, order, operands, results, **attributes):
    names = [entry if entry in operands else "" for entry in order]
    while names and not names[-1]:
        names.pop()
    node = helper.make_node(op_type, names, list(results), name="node", **attributes)
    return _model(
        [node],
        [_tensor(entry, *operands[entry]) for entry in order if entry in operands],
        [helper.make_empty_tensor_value_info(name) for name in results],
        opset=_TRANSFORMER_OPSET,
    )


def _attention_model(operands, *, outputs=1, **attributes):
    return _transformer_model(
        "Attention",
        _ATTENTION_INPUTS,
        operands,
        _ATTENTION_OUTPUTS[:outputs],
        **attributes,
    )


def _rotary_model(operands, **attributes):
    return _transformer_model(
        "RotaryEmbedding", _ROTARY_INPUTS, operands, ("Y",), **attributes
    )


def _attention_kernels(report, role=None):
    """Attention's emitted kernels: the scorer itself, or one of its named helpers."""
    start = len(f"{report['prefix']}_attention_")
    return [
        name
        for name in _kernels(report, "attention")
        if name[start:].split("_")[0] == role
        or (role is None and name[start:].split("_")[0] not in _ATTENTION_HELPERS)
    ]


def _feeds(model, seed=20260726):
    """A seeded value for every input the model declares, at its own dtype and shape."""
    generator = np.random.default_rng(seed)
    drawn = {}
    for entry in model.graph.input:
        tensor = entry.type.tensor_type
        shape = tuple(dim.dim_value for dim in tensor.shape.dim)
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        drawn[entry.name] = (
            generator.normal(size=shape).astype(dtype)
            if dtype.kind == "f"
            else generator.integers(0, 2, size=shape).astype(dtype)
        )
    return drawn


def test_both_attention_layouts_share_one_kernel(tmp_path):
    """`(batch, head, sequence, size)` and `(batch, sequence, head * size)` are one tensor at
    two sets of strides, so the loop nest is the same code and the layout arrives as
    arguments."""
    model = _model(
        [
            helper.make_node("Attention", ["q4", "k4", "v4"], ["y4"], name="packed"),
            helper.make_node(
                "Attention",
                ["q3", "k3", "v3"],
                ["y3"],
                name="flat",
                q_num_heads=2,
                kv_num_heads=2,
            ),
        ],
        [
            _tensor("q4", TensorProto.FLOAT, (1, 2, 2, 4)),
            _tensor("k4", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("v4", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("q3", TensorProto.FLOAT, (1, 2, 8)),
            _tensor("k3", TensorProto.FLOAT, (1, 3, 8)),
            _tensor("v3", TensorProto.FLOAT, (1, 3, 8)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y4", "y3")],
        opset=_TRANSFORMER_OPSET,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _attention_kernels(report)
    assert header.count(f"static void {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


def test_a_narrowed_softmax_precision_emits_a_scorer_of_its_own(tmp_path):
    """The type the softmax runs in is part of the emitted code, so it is part of the name."""
    model = _model(
        [
            helper.make_node("Attention", ["q", "k", "v"], ["y"], name="wide"),
            helper.make_node(
                "Attention",
                ["q", "k", "v"],
                ["z"],
                name="narrow",
                softmax_precision=TensorProto.FLOAT,
            ),
        ],
        [
            _tensor("q", TensorProto.FLOAT, (1, 2, 2, 4)),
            _tensor("k", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("v", TensorProto.FLOAT, (1, 2, 3, 4)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z")],
        opset=_TRANSFORMER_OPSET,
    )

    report, _ = _compile(model, tmp_path)

    kernels = sorted(_attention_kernels(report))
    assert len(kernels) == 2
    assert [name.split("_soft")[1] for name in kernels] == [
        "double_maskfloat",
        "float_maskfloat",
    ]


def test_a_boolean_mask_emits_a_bias_of_its_own(tmp_path):
    """The two mask forms are different expressions over a differently typed operand."""
    model = _model(
        [
            helper.make_node("Attention", ["q", "k", "v", "f"], ["y"], name="additive"),
            helper.make_node("Attention", ["q", "k", "v", "b"], ["z"], name="boolean"),
        ],
        [
            _tensor("q", TensorProto.FLOAT, (1, 2, 2, 4)),
            _tensor("k", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("v", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("f", TensorProto.FLOAT, (2, 3)),
            _tensor("b", TensorProto.BOOL, (2, 3)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z")],
        opset=_TRANSFORMER_OPSET,
    )

    report, header = _compile(model, tmp_path)

    assert len(_attention_kernels(report)) == 2
    assert len(_attention_kernels(report, "bias")) == 2
    # The NaN a boolean mask turns an entry that does take part into under `is_causal`;
    # the additive form has no such value anywhere in it.
    assert header.count("(causal ? (double)NAN : 0.0) : -INFINITY;") == 1


def test_the_caches_are_written_only_where_the_node_asks_for_them(tmp_path):
    """`present_key` and `present_value` are a concatenation nothing else needs."""
    past = (TensorProto.FLOAT, (1, 2, 2, 4))
    operands = {**_ATTENTION_OPERANDS, "past_key": past, "past_value": past}

    report, _ = _compile(_attention_model(operands), tmp_path)
    assert not _attention_kernels(report, "present")

    report, header = _compile(_attention_model(operands, outputs=3), tmp_path)
    (present,) = _attention_kernels(report, "present")
    assert header.count(f"{present}(\n") == 3


def test_the_row_of_scores_is_one_buffer_sized_for_the_longest_row(tmp_path):
    """The softmax needs the whole row before it can normalize any of it, and the artifact
    allocates nothing; nodes sharing the kernel share that row, sized for the largest."""
    model = _model(
        [
            helper.make_node("Attention", ["q", "k", "v"], ["y"], name="short"),
            helper.make_node("Attention", ["q", "k2", "v2"], ["z"], name="long"),
        ],
        [
            _tensor("q", TensorProto.FLOAT, (1, 2, 2, 4)),
            _tensor("k", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("v", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("k2", TensorProto.FLOAT, (1, 2, 7, 4)),
            _tensor("v2", TensorProto.FLOAT, (1, 2, 7, 4)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z")],
        opset=_TRANSFORMER_OPSET,
    )

    report, header = _compile(model, tmp_path)

    assert (
        header.count(f"static double {report['prefix']}_attention_scores_double[") == 1
    )
    assert f"{report['prefix']}_attention_scores_double[7];" in header


@requires_c_compiler
def test_a_mask_of_one_axis_reaches_every_query_row(tmp_path):
    """ONNX defines `attn_mask` as broadcastable to the whole score tensor, which a single
    key axis is; the corpus only ever ships the 2-D and 4-D forms."""
    model = _attention_model(
        {**_ATTENTION_OPERANDS, "attn_mask": (TensorProto.FLOAT, (3,))}
    )
    feeds = _feeds(model)

    got = compile_onnx(model, tmp_path).load().run(feeds)["Y"]

    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-7)


def test_both_rotation_patterns_share_one_kernel(tmp_path):
    """`interleaved` picks which two lanes a pair is made of and nothing else, so it reaches
    the loop nest as an argument rather than forking it."""
    model = _model(
        [
            helper.make_node(
                "RotaryEmbedding", ["x", "cos", "sin"], ["y"], name="halves"
            ),
            helper.make_node(
                "RotaryEmbedding",
                ["x", "cos", "sin"],
                ["z"],
                name="interleaved",
                interleaved=1,
            ),
        ],
        [
            _tensor("x", TensorProto.FLOAT, (1, 2, 3, 4)),
            _tensor("cos", TensorProto.FLOAT, (1, 3, 2)),
            _tensor("sin", TensorProto.FLOAT, (1, 3, 2)),
        ],
        [helper.make_empty_tensor_value_info(name) for name in ("y", "z")],
        opset=_TRANSFORMER_OPSET,
    )

    report, header = _compile(model, tmp_path)

    (kernel,) = _kernels(report, "rotaryembedding")
    assert header.count(f"static int {kernel}(") == 1
    assert header.count(f"{kernel}(\n") == 3


@requires_c_compiler
def test_a_position_the_cache_has_no_row_for_is_reported(tmp_path):
    """`position_ids` is read at run time, so an index outside the cache is the argument
    error the status enum exists for rather than a read past the buffer."""
    compiled = compile_onnx(_rotary_model(_ROTARY_OPERANDS), tmp_path).load()
    feeds = _feeds(_rotary_model(_ROTARY_OPERANDS))
    feeds["position_ids"] = np.array([[0, 1, 5]], np.int64)

    with pytest.raises(HarnessError, match="status 1"):
        compiled.run(feeds)


@pytest.mark.parametrize(
    ("operands", "attributes", "message"),
    [
        (
            {"V": (TensorProto.DOUBLE, (1, 2, 3, 4))},
            {},
            "this compiler attends one element type",
        ),
        (
            {"attn_mask": (TensorProto.INT32, (2, 3))},
            {},
            "a boolean mask or a float mask of the operands' own type",
        ),
        (
            {},
            {"softmax_precision": TensorProto.FLOAT16},
            "`softmax_precision` of `FLOAT16`",
        ),
        (
            {},
            {"softmax_precision": TensorProto.INT32},
            "`softmax_precision` of `INT32`",
        ),
        ({}, {"qk_matmul_output_mode": 4}, "`qk_matmul_output_mode` of 4"),
        (
            {
                "Q": (TensorProto.FLOAT, (1, 2, 9)),
                "K": (TensorProto.FLOAT, (1, 3, 8)),
                "V": (TensorProto.FLOAT, (1, 3, 8)),
            },
            {"q_num_heads": 2, "kv_num_heads": 2},
            "hidden axis of 9 into 2 head(s), which does not divide it",
        ),
        ({}, {"q_num_heads": 3}, "states `q_num_heads` 3"),
        (
            {"K": (TensorProto.FLOAT, (1, 2, 3, 6))},
            {},
            "contracts `Q` against `K` over the head size",
        ),
        (
            {"V": (TensorProto.FLOAT, (1, 2, 5, 4))},
            {},
            "reads `K` and `V` at the same key positions",
        ),
        (
            {"Q": (TensorProto.FLOAT, (1, 3, 2, 4))},
            {},
            "needs 3 query head(s) to be a multiple of 2",
        ),
        (
            {"past_key": (TensorProto.FLOAT, (1, 2, 2, 4))},
            {},
            "ONNX defines them as used together",
        ),
        (
            {
                "past_key": (TensorProto.FLOAT, (1, 2, 2, 4)),
                "past_value": (TensorProto.FLOAT, (1, 2, 3, 4)),
            },
            {},
            "as a cache of shape [1, 2, 2, 4]",
        ),
        (
            {"nonpad_kv_seqlen": (TensorProto.INT64, (2,))},
            {},
            "one key length per batch item",
        ),
        (
            {"attn_mask": (TensorProto.FLOAT, (5, 3))},
            {},
            "does not broadcast",
        ),
        # Shorter than the key axis is padded out with -inf; longer is what the reference
        # raises on, so there is nothing for the columns past the end to mean.
        (
            {"attn_mask": (TensorProto.FLOAT, (2, 4))},
            {},
            "reads `attn_mask` along 4 key position(s), but the node attends 3",
        ),
        # The reference takes the triangle's extent from the mask's own query axis, so a
        # mask with no such axis is a node it cannot evaluate at all.
        (
            {"attn_mask": (TensorProto.FLOAT, (3,))},
            {"is_causal": 1},
            "the triangle's extent from the mask's own query axis",
        ),
    ],
)
def test_an_attention_node_the_compiler_cannot_serve_is_rejected(
    tmp_path, operands, attributes, message
):
    model = _attention_model({**_ATTENTION_OPERANDS, **operands}, **attributes)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("operands", "attributes", "message"),
    [
        ({}, {"rotary_embedding_dim": 3}, "rotates the first 3 lane(s) of a head of 4"),
        ({}, {"rotary_embedding_dim": 6}, "rotates the first 6 lane(s) of a head of 4"),
        ({}, {"num_heads": 3}, "states `num_heads` 3"),
        (
            {"X": (TensorProto.FLOAT, (1, 3, 9))},
            {"num_heads": 2},
            "hidden axis of 9 into 2 head(s), which does not divide it",
        ),
        (
            {"sin_cache": (TensorProto.FLOAT, (5, 1))},
            {},
            "as one pair of caches",
        ),
        (
            {
                "cos_cache": (TensorProto.FLOAT, (5, 1)),
                "sin_cache": (TensorProto.FLOAT, (5, 1)),
            },
            {},
            "rotates 2 pair(s) per head",
        ),
        (
            {
                "cos_cache": (TensorProto.FLOAT, (1, 3, 2)),
                "sin_cache": (TensorProto.FLOAT, (1, 3, 2)),
            },
            {},
            "reads `cos_cache` as rank 2",
        ),
        (
            {"position_ids": (TensorProto.INT64, (1, 1, 3))},
            {},
            "reads `position_ids` as `(batch, sequence)`",
        ),
        # Rank 1 is refused for the same reason rank 3 is: the reference inserts the head
        # axis at position 2 of whatever it gathered, which for one axis fewer lands past
        # the angles rather than before them.
        (
            {"position_ids": (TensorProto.INT64, (3,))},
            {},
            "reads `position_ids` as `(batch, sequence)`",
        ),
    ],
)
def test_a_rotary_node_the_compiler_cannot_serve_is_rejected(
    tmp_path, operands, attributes, message
):
    model = _rotary_model({**_ROTARY_OPERANDS, **operands}, **attributes)

    with pytest.raises(CompileError, match=re.escape(message)):
        compile_onnx(model, tmp_path)


def test_one_tensor_named_for_both_caches_is_checked_against_both(tmp_path):
    """`past_key` and `past_value` are checked separately even when they are one tensor.

    A node may name the same tensor for both, and the two are then the same value — so a
    check that pairs each operand with its expected shape has to keep the pair, not key on
    the operand. `head_size` and `v_head_size` differ here, so at most one of the two can
    hold, and the kernel would otherwise address 8 lanes of a cache laid out at 2.
    """
    operands = {
        "Q": (TensorProto.FLOAT, (1, 1, 2, 8)),
        "K": (TensorProto.FLOAT, (1, 1, 2, 8)),
        "V": (TensorProto.FLOAT, (1, 1, 2, 2)),
    }
    node = helper.make_node(
        "Attention", ["Q", "K", "V", "", "cache", "cache"], ["Y"], name="node"
    )
    model = _model(
        [node],
        [_tensor(name, *spec) for name, spec in operands.items()]
        + [_tensor("cache", TensorProto.FLOAT, (1, 1, 3, 2))],
        [helper.make_empty_tensor_value_info("Y")],
        opset=_TRANSFORMER_OPSET,
    )

    with pytest.raises(
        CompileError, match=re.escape("as a cache of shape [1, 1, 3, 8]")
    ):
        compile_onnx(model, tmp_path)


@requires_c_compiler
def test_two_losses_ignoring_different_labels_share_one_kernel(tmp_path):
    """`ignore_index` decides what a kernel skips, not the code that skips it.

    Which of `reduction`, `weights` and the presence of the attribute a node carries forks
    the kernel's body; the index itself is a call-site literal like every other attribute
    value here, so two nodes that differ only in it share one emitted function instead of
    claiming one name for two definitions.
    """
    nodes = [
        helper.make_node(
            "SoftmaxCrossEntropyLoss",
            ["x", "t"],
            [f"l{index}"],
            name=f"loss{index}",
            reduction="sum",
            ignore_index=index,
        )
        for index in (1, 2)
    ]
    nodes.append(helper.make_node("Add", ["l1", "l2"], ["y"], name="total"))
    model = _model(
        nodes,
        [
            _tensor("x", TensorProto.FLOAT, (3, 5)),
            _tensor("t", TensorProto.INT64, (3,)),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=_SCE_OPSET,
    )
    feeds = {
        "x": np.arange(15, dtype=np.float32).reshape(3, 5) / 3,
        "t": np.array([0, 1, 2], dtype=np.int64),
    }

    report, header = _compile(model, tmp_path)

    assert len([name for name in report["kernels"] if "crossentropy" in name]) == 1
    outputs = compile_onnx(model, tmp_path / "run").load().run(feeds)
    expected = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_allclose(outputs["y"], expected[0], rtol=1e-6, atol=1e-6)


def test_a_cache_nothing_gathers_carries_its_own_positions(tmp_path):
    """Without `position_ids` the caches stand as they are, which ONNX shapes
    `(batch, sequence, rotary_embedding_dim / 2)` rather than by position."""
    operands = {
        name: value
        for name, value in _ROTARY_OPERANDS.items()
        if name != "position_ids"
    }

    with pytest.raises(CompileError, match="reads `cos_cache` as rank 3"):
        compile_onnx(_rotary_model(operands), tmp_path)


# --------------------------------------------------------------------------------------
# TfIdfVectorizer
# --------------------------------------------------------------------------------------

# What the op counts is settled by the conformance and differential suites, against ONNX's
# own corpus and reference evaluator. What is asserted here is that the pool becomes one
# shared kernel over `static const` tables rather than code per node, and the refusals for
# the nodes the compiler will not emit at all.
_TFIDF_OPSET = 9
_TFIDF_POOL = {
    "ngram_counts": [0, 4],
    "ngram_indexes": [0, 1, 2, 3, 4, 5, 6],
    "pool_int64s": [2, 3, 5, 4, 5, 6, 7, 8, 6, 7],
}


def _tfidf_model(*, shape=(12,), elem_type=TensorProto.INT64, nodes=1, **attributes):
    counted = {
        **_TFIDF_POOL,
        "mode": "TF",
        "min_gram_length": 1,
        "max_gram_length": 2,
        "max_skip_count": 0,
        **attributes,
    }
    graph = [
        helper.make_node(
            "TfIdfVectorizer", ["x"], [f"y{index}"], name=f"count{index}", **counted
        )
        for index in range(nodes)
    ]
    return _model(
        graph,
        [_tensor("x", elem_type, shape)],
        [helper.make_empty_tensor_value_info(f"y{index}") for index in range(nodes)],
        opset=_TFIDF_OPSET,
    )


def test_the_pool_is_emitted_once_for_the_nodes_that_share_it(tmp_path):
    """Two nodes over one pool: one kernel, and one copy of each table it walks."""
    report, header = _compile(_tfidf_model(nodes=2), tmp_path)

    assert len(_kernels(report, "tfidfvectorizer")) == 1
    for role in ("tokens", "targets", "first_edge", "edge_count", "counted_column"):
        assert len(re.findall(rf"tfidfvectorizer_{role}_\w+\[\d+\] = ", header)) == 1


def test_the_modes_that_read_no_weights_share_one_kernel(tmp_path):
    """`TF` ignores the weights a node carries, so it emits the code a node without them does."""
    weighted = _tfidf_model(weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    report, header = _compile(weighted, tmp_path)

    (kernel,) = _kernels(report, "tfidfvectorizer")
    assert kernel.endswith("_tf_flat_int64_t")
    # Past the preamble, whose footprint summary names weights as a memory category.
    assert "weights" not in header.split("*/", 1)[1]


def test_a_string_pool_is_rejected(tmp_path):
    """A string pool is matched against a string tensor, which the artifact cannot hold."""
    model = _tfidf_model(pool_strings=["a", "b", "c", "d"])
    del model.graph.node[0].attribute[
        [entry.name for entry in model.graph.node[0].attribute].index("pool_int64s")
    ]

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`count0`" in message
    assert "pool_strings" in message


@pytest.mark.parametrize(
    ("attributes", "expected"),
    (
        ({"mode": "COUNT"}, "`TF`, `IDF` and `TFIDF`"),
        ({"min_gram_length": 0}, "`min_gram_length`"),
        ({"min_gram_length": 3, "max_gram_length": 2}, "`min_gram_length`"),
        ({"weights": [1.0, 2.0]}, "weight(s) for"),
        # Every n-gram of the pool takes an identifier, and every identifier reads an index.
        ({"ngram_indexes": [0, 1, 2]}, "`ngram_indexes` entries account for"),
    ),
)
def test_a_node_the_compiler_cannot_count_from_is_rejected(
    attributes, expected, tmp_path
):
    with pytest.raises(CompileError) as error:
        compile_onnx(_tfidf_model(**attributes), tmp_path)

    message = str(error.value)
    assert "`count0`" in message
    assert expected in message


def _tfidf_emission(shape, elem_type, width=7, **attributes):
    """The registered kernel generator, run on one node directly.

    ONNX's own inference reads the rank and the index list before dispatch asks for a kernel,
    and refuses what it cannot type: these are the generator's last word for a node that
    arrives without inference having vetted it, and this is where they are read.
    """
    node = helper.make_node(
        "TfIdfVectorizer",
        ["x"],
        ["y"],
        name="count0",
        **{
            **_TFIDF_POOL,
            "mode": "TF",
            "min_gram_length": 1,
            "max_gram_length": 2,
            "max_skip_count": 0,
            **attributes,
        },
    )
    spec = KERNELS.select("", "TfIdfVectorizer", _TFIDF_OPSET)
    context = NodeContext(
        node=node,
        domain="",
        opset_version=_TFIDF_OPSET,
        since_version=spec.since_version,
        prefix="count",
        inputs=(TensorRef("x", elem_type, shape, "x"),),
        outputs=(TensorRef("y", TensorProto.FLOAT, (*shape[:-1], width), "y"),),
    )
    return spec.generator(context)


@pytest.mark.parametrize(
    ("shape", "elem_type", "attributes", "expected"),
    (
        ((2, 3, 4), TensorProto.INT64, {}, "one token sequence or a batch"),
        ((12,), TensorProto.FLOAT, {}, "int32` or `int64` tokens"),
        ((12,), TensorProto.INT64, {"ngram_indexes": [0, -1]}, "`ngram_indexes` entry"),
    ),
)
def test_a_node_inference_did_not_vet_is_still_refused(
    shape, elem_type, attributes, expected
):
    with pytest.raises(CompileError, match=re.escape(expected)):
        _tfidf_emission(shape, elem_type, **attributes)
