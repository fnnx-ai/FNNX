"""Function expansion: compiling an op through the body ONNX defines for it.

Dispatch is native kernel → function expansion → compile error, and this module covers the
middle step. Nothing here decides what an op computes: the bodies come from the `onnx`
schema registry, the values a body is specialized with come from the model's own node, and
what an expanded artifact must produce comes from `onnx.reference.ReferenceEvaluator`.
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402
from fnnx.extras.compilers.c.onnx import codegen  # noqa: E402
from fnnx.extras.compilers.c.onnx.frontend import prepare_model  # noqa: E402
from fnnx.extras.compilers.c.onnx.functions import (  # noqa: E402
    Expansion,
    expand_function,
    function_body,
)
from fnnx.extras.compilers.c.onnx.kernels import KERNELS  # noqa: E402
from fnnx.extras.compilers.c.onnx.loader import LoadedModel, resolve_opsets  # noqa: E402

# `HardSwish` at 22 is the fallback's smallest complete case: no native kernel serves it,
# ONNX defines it by a function body, and that body — one `HardSigmoid` and one `Mul` — is
# made of ops this compiler does serve.
HARD_SWISH_OPSET = 22
# `Bernoulli` at 22 is the other side of it: a body built on `RandomUniformLike`, which
# draws at random and can never be compiled into a static artifact — so it stays the
# fallback's failing case however far kernel coverage grows.
BERNOULLI_OPSET = 22
# `Clip`'s body is context-dependent, which is what the specialization tests need; dispatch
# never reaches it, since a native kernel serves the op.
CLIP_OPSET = 13
LAYER_NORM_OPSET = 17
# `CenterCropPad` at 18 is the case for an operand's *value* reaching a body: its own
# reads the extents it crops to out of a tensor, and no kernel serves the op.
CENTER_CROP_PAD_OPSET = 18
# Older than every registered `Add` kernel, and an opset at which ONNX defines no body for
# the op — so dispatch runs out of options with kernels registered for the op all the same.
LEGACY_ADD_OPSET = 6

requires_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


def _model(node, inputs, outputs, *, opset, name="expansion"):
    graph = helper.make_graph([node], name, list(inputs), list(outputs))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _tensor(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _type(elem_type, shape) -> Any:
    return helper.make_tensor_type_proto(elem_type, list(shape))


def _hard_swish(elem_type=TensorProto.FLOAT, shape=(2, 3)):
    return _model(
        helper.make_node("HardSwish", ["x"], ["y"], name="swish"),
        [_tensor("x", elem_type, list(shape))],
        [helper.make_empty_tensor_value_info("y")],
        opset=HARD_SWISH_OPSET,
    )


def _layer_norm_node():
    """`LayerNormalization` asking only for `Y`, leaving `Mean` and `InvStdDev` out."""
    return helper.make_node("LayerNormalization", ["x", "scale"], ["y"], axis=-1)


_LAYER_NORM_TYPES = (_type(TensorProto.FLOAT, (2, 3)), _type(TensorProto.FLOAT, (3,)))


# --------------------------------------------------------------------------------------
# Dispatch: kernel, then body, then error
# --------------------------------------------------------------------------------------


@requires_c_compiler
@pytest.mark.parametrize(
    ("elem_type", "dtype"),
    [(TensorProto.FLOAT, "float32"), (TensorProto.DOUBLE, "float64")],
)
def test_a_function_defined_op_compiles_and_matches_the_reference(
    tmp_path, elem_type, dtype
):
    """The scenario itself: no kernel, a body, and output the reference agrees with."""
    model = _hard_swish(elem_type)
    values = np.arange(-3, 3, dtype=dtype).reshape(2, 3)

    compiled = compile_onnx(model, tmp_path).load()
    outputs = compiled.run({"x": values})

    expected = ReferenceEvaluator(model).run(None, {"x": values})
    np.testing.assert_array_equal(outputs["y"], expected[0])


@requires_c_compiler
def test_a_context_dependent_body_carries_the_type_castlike_reads_off_its_operand(
    tmp_path,
):
    """`CastLike` has no kernel of its own, and could not have a fixed one: the type it
    converts to is the second operand's, which ONNX writes into the body it builds per call
    site. Compiling that body is what serves the op."""
    assert not KERNELS.registered_versions("", "CastLike")
    model = _model(
        helper.make_node("CastLike", ["x", "like"], ["y"], name="convert"),
        [
            _tensor("x", TensorProto.FLOAT, [2, 3]),
            _tensor("like", TensorProto.INT16, [1]),
        ],
        [helper.make_empty_tensor_value_info("y")],
        opset=21,
    )
    feeds = {
        "x": np.array([[1.5, -2.5, 0.0], [3.9, -4.9, 6.0]], dtype="float32"),
        "like": np.zeros(1, dtype="int16"),
    }

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    expected = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_array_equal(outputs["y"], expected[0])


def test_a_native_kernel_takes_precedence_over_the_function_body(tmp_path):
    """`Relu` has both; the body needs ops no kernel serves, so using it would fail."""
    model = _model(
        helper.make_node("Relu", ["x"], ["y"], name="relu"),
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )
    assert onnx.defs.get_schema("Relu", 14, "").has_function

    report = compile_onnx(model, tmp_path).report

    assert [
        kernel
        for kernel in report["kernels"]
        if kernel.startswith(f"{report['prefix']}_relu_")
    ] == report["kernels"]


def test_a_kernel_the_registry_cannot_vouch_for_falls_through_to_the_body(
    tmp_path, monkeypatch
):
    """A selection the semantic-revision guard rejects is not the end of dispatch.

    `Relu` has both a kernel and a body, so a registry that selects nothing for it is what
    a guard rejection looks like from the emitter's side.
    """
    monkeypatch.setattr(KERNELS, "select", lambda domain, op_type, version: None)
    model = _model(
        helper.make_node("Relu", ["x"], ["y"], name="relu"),
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    assert "function body of `Relu`" in str(error.value)


def test_an_op_with_neither_a_kernel_nor_a_body_is_rejected(tmp_path):
    """`RandomUniform` draws rather than computes, which is off the v1 supported surface."""
    model = _model(
        helper.make_node("RandomUniform", [], ["y"], name="draw", shape=[2, 3]),
        [],
        [helper.make_empty_tensor_value_info("y")],
        opset=14,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`draw`" in message
    assert "`RandomUniform`" in message
    assert "ai.onnx" in message
    assert "14" in message


def test_a_version_no_kernel_and_no_body_covers_keeps_the_dispatch_guidance(tmp_path):
    """Falling through to expansion must not cost the error its nearest-version guidance.

    Dispatch declines twice here — no kernel is valid at this opset and ONNX defines no
    body — and what surfaces has to be the registry's own error, not the fallback's.
    """
    nearest = min(KERNELS.registered_versions("", "Add"))
    assert nearest > LEGACY_ADD_OPSET
    model = _model(
        helper.make_node("Add", ["a", "b"], ["y"], name="adder"),
        [_tensor("a", TensorProto.FLOAT, [2]), _tensor("b", TensorProto.FLOAT, [2])],
        [helper.make_empty_tensor_value_info("y")],
        opset=LEGACY_ADD_OPSET,
    )
    output_dir = tmp_path / "out"

    with pytest.raises(CompileError) as error:
        compile_onnx(model, output_dir)

    message = str(error.value)
    assert "`adder`" in message
    assert "`Add`" in message
    assert "ai.onnx" in message
    assert f"opset version {LEGACY_ADD_OPSET}" in message
    assert f"Nearest supported version: {nearest}" in message
    assert not output_dir.exists()


def test_an_op_whose_body_needs_a_missing_kernel_is_rejected(tmp_path):
    """The error names the model's own node and the primitive the body ran aground on."""
    model = _model(
        helper.make_node("Bernoulli", ["x"], ["y"], name="draw"),
        [_tensor("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_empty_tensor_value_info("y")],
        opset=BERNOULLI_OPSET,
    )

    with pytest.raises(CompileError) as error:
        compile_onnx(model, tmp_path)

    message = str(error.value)
    assert "`draw`" in message
    assert "`Bernoulli`" in message
    assert "`RandomUniformLike`" in message


@requires_c_compiler
def test_a_body_made_of_function_defined_ops_expands_recursively(tmp_path, monkeypatch):
    """`HardSwish`'s body calls `HardSigmoid`, which is function-defined in turn.

    With no kernel serving `HardSigmoid` — what a semantic-revision guard rejection looks
    like — the inner node has to expand through its own body, so the artifact is built out
    of that body's primitives rather than out of a `HardSigmoid` kernel, and still has to
    compute what the reference computes.
    """
    select = KERNELS.select
    monkeypatch.setattr(
        KERNELS,
        "select",
        lambda domain, op_type, version: (
            None if op_type == "HardSigmoid" else select(domain, op_type, version)
        ),
    )
    model = _hard_swish()
    values = np.arange(-3, 3, dtype="float32").reshape(2, 3)

    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"x": values})

    assert not [
        kernel for kernel in result.report["kernels"] if "hardsigmoid" in kernel
    ]
    expected = ReferenceEvaluator(model).run(None, {"x": values})
    np.testing.assert_array_equal(outputs["y"], expected[0])


def test_expansion_stops_at_a_bounded_nesting_depth(tmp_path, monkeypatch):
    """A body that never bottoms out is a compile error, not a stack overflow."""
    monkeypatch.setattr(codegen, "MAX_EXPANSION_DEPTH", 0)

    with pytest.raises(CompileError, match="nested more than"):
        compile_onnx(_hard_swish(), tmp_path)


# --------------------------------------------------------------------------------------
# The body a node expands to
# --------------------------------------------------------------------------------------


def test_the_body_is_specialized_to_the_operands_the_node_passes():
    """A context-dependent body sees which optional operands the node actually has."""
    float_type = _type(TensorProto.FLOAT, (2, 3))
    scalar = _type(TensorProto.FLOAT, ())
    unbounded = helper.make_node("Clip", ["x", "", ""], ["y"])
    bounded = helper.make_node("Clip", ["x", "lo", "hi"], ["y"])

    without = expand_function(unbounded, "", CLIP_OPSET, [float_type, None, None])
    with_bounds = expand_function(bounded, "", CLIP_OPSET, [float_type, scalar, scalar])

    assert [node.op_type for node in without.prepared.model.graph.node] == ["Identity"]
    assert without.inputs == (("input", 0),)
    assert with_bounds.inputs == (("input", 0), ("min", 1), ("max", 2))
    assert len(with_bounds.prepared.model.graph.node) > 1


def test_an_attribute_the_node_sets_reaches_the_body():
    """`LeakyRelu`'s body reads `alpha` through the caller's node."""
    node = helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.25)

    expansion = expand_function(node, "", 16, [_type(TensorProto.FLOAT, (2, 3))])

    assert 0.25 in _folded_values(expansion)


def test_an_attribute_the_node_omits_falls_back_on_the_schema_default():
    """Without the default filled in, the body's `Constant` would have no value at all."""
    node = helper.make_node("LeakyRelu", ["x"], ["y"])
    default = (
        onnx.defs.get_schema("LeakyRelu", 16, "").attributes["alpha"].default_value
    )

    expansion = expand_function(node, "", 16, [_type(TensorProto.FLOAT, (2, 3))])

    assert np.float32(default.f) in _folded_values(expansion)


def test_the_body_of_an_omitted_optional_output_is_not_compiled():
    """A body computes every output its op declares; the caller pays only for the ones it
    asked for. Asserted as the property pruning establishes: nothing left in the compiled
    body fails to reach an output the node wants."""
    graph = expand_function(
        _layer_norm_node(), "", LAYER_NORM_OPSET, list(_LAYER_NORM_TYPES)
    ).prepared.model.graph

    assert [entry.name for entry in graph.output] == ["Y"]
    live = {entry.name for entry in graph.output}
    for node in reversed(graph.node):
        assert any(name in live for name in node.output), (
            f"`{node.op_type}` computes nothing the expanded node asked for"
        )
        live |= {name for name in node.input if name}


def test_the_unpruned_body_does_compute_the_omitted_outputs():
    """Teeth for the test above: ONNX's body really is bigger than what gets compiled."""
    node = _layer_norm_node()

    body = function_body(node, "", LAYER_NORM_OPSET, list(_LAYER_NORM_TYPES))
    compiled = expand_function(
        node, "", LAYER_NORM_OPSET, list(_LAYER_NORM_TYPES)
    ).prepared.model.graph

    assert list(body.output) == ["Y", "Mean", "InvStdDev"]
    assert len(compiled.node) < len(
        [entry for entry in body.node if entry.op_type != "Constant"]
    )


def test_an_op_onnx_defines_no_body_for_has_no_expansion():
    node = helper.make_node("Sub", ["a", "b"], ["y"])

    assert function_body(node, "", 14, [None, None]) is None
    assert expand_function(node, "", 14, [None, None]) is None


def test_an_op_the_installed_onnx_does_not_define_has_no_expansion():
    node = helper.make_node("NotAnOnnxOp", ["a"], ["y"])

    assert expand_function(node, "", 14, [None]) is None


def _folded_values(expansion: Expansion) -> list[Any]:
    """Every scalar the prepared body holds as constant data."""
    return [
        onnx.numpy_helper.to_array(initializer).reshape(-1)[0]
        for initializer in expansion.prepared.model.graph.initializer
        if onnx.numpy_helper.to_array(initializer).size == 1
    ]


# --------------------------------------------------------------------------------------
# Splicing a body into the caller's buffers
# --------------------------------------------------------------------------------------


@requires_c_compiler
def test_a_body_output_no_node_writes_still_reaches_the_callers_buffer(
    tmp_path, monkeypatch
):
    """A body output that folding resolves to a constant is written by nothing; the value
    still has to land in the buffer the caller passed for it."""
    values = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    body = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["output"],
                    value=onnx.numpy_helper.from_array(values, "value"),
                )
            ],
            "constant_body",
            [_tensor("input", TensorProto.FLOAT, [2, 3])],
            [helper.make_empty_tensor_value_info("output")],
        ),
        opset_imports=[helper.make_opsetid("", CLIP_OPSET)],
    )
    expansion = Expansion(
        prepared=prepare_model(LoadedModel(model=body, opsets=resolve_opsets(body))),
        inputs=(("input", 0),),
        outputs=(("output", 0),),
    )
    monkeypatch.setattr(codegen, "expand_function", lambda *_: expansion)

    compiled = compile_onnx(_hard_swish(), tmp_path).load()
    outputs = compiled.run({"x": np.zeros((2, 3), dtype=np.float32)})

    np.testing.assert_array_equal(outputs["y"], values)


@requires_c_compiler
def test_an_operand_the_graph_fixes_reaches_the_body(tmp_path):
    """A body that computes its own result shape from an operand needs the operand itself.

    `CenterCropPad` is defined by a body that pads and then slices by extents it derives from
    the `shape` tensor; with only that tensor's type, the `Pad` inside takes a shape no
    folding can settle and the body is refused. The values the caller's graph fixes therefore
    travel with the types, and here they are what makes the op compilable at all.
    """
    extents = np.array([9, 5], dtype=np.int64)
    node = helper.make_node(
        "CenterCropPad", ["x", "shape"], ["y"], name="crop", axes=[0, 1]
    )
    graph = helper.make_graph(
        [node],
        "center_crop_pad",
        [_tensor("x", TensorProto.FLOAT, [10, 7, 3])],
        [helper.make_empty_tensor_value_info("y")],
        initializer=[onnx.numpy_helper.from_array(extents, "shape")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", CENTER_CROP_PAD_OPSET)]
    )
    feeds = {"x": np.arange(210, dtype=np.float32).reshape(10, 7, 3)}

    outputs = compile_onnx(model, tmp_path).load().run(feeds)

    assert not KERNELS.registered_versions("", "CenterCropPad")
    expected = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_array_equal(outputs["y"], expected[0])
