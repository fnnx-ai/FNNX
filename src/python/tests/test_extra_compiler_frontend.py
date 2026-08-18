"""Dimension binding, constant folding, and static verification in the C compiler frontend.

Every expected value comes from the ONNX reference evaluator — the executable form of the
spec — or from the ONNX schema registry; none is hand-written.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
loader = pytest.importorskip("fnnx.extras.compilers.c.onnx.loader")
dtypes = pytest.importorskip("fnnx.extras.compilers.c.onnx.dtypes")
folding = pytest.importorskip("fnnx.extras.compilers.c.onnx.folding")
frontend = pytest.importorskip("fnnx.extras.compilers.c.onnx.frontend")
shapes = pytest.importorskip("fnnx.extras.compilers.c.onnx.shapes")
verify = pytest.importorskip("fnnx.extras.compilers.c.onnx.verify")

from onnx import ModelProto, TensorProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"
OPS_ARTIFACTS = MODELS_DIR / "onnx_pipeline.fnnx" / "ops_artifacts"

MAX_OPSET = onnx.defs.onnx_opset_version()
# The opset the bundle's own node models import; folding must work at a real model's
# version, not only at the newest one the installed `onnx` package defines.
BUNDLE_OPSET = 21


def _model(
    nodes,
    inputs,
    outputs,
    *,
    initializer=(),
    sparse_initializer=(),
    opset: int = BUNDLE_OPSET,
    ir_version: int | None = None,
):
    graph = helper.make_graph(
        nodes,
        "g",
        inputs,
        outputs,
        initializer=list(initializer),
        sparse_initializer=list(sparse_initializer),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if ir_version is not None:
        model.ir_version = ir_version
    return model


def _prepare(model, **kwargs):
    return frontend.prepare_model(loader.load_model(model), **kwargs)


def _tensor(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _unshaped(name, elem_type=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, elem_type, None)


def _constant_node(name, values, dtype=TensorProto.INT64):
    array = np.asarray(values)
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=helper.make_tensor(
            f"{name}_value", dtype, list(array.shape), array.flatten().tolist()
        ),
    )


def _constant_branch(name, values):
    """A subgraph that produces a constant, for use as an `If` branch."""
    return helper.make_graph(
        [_constant_node(name, values, dtype=TensorProto.FLOAT)],
        name,
        [],
        [_tensor(name, TensorProto.FLOAT, [len(values)])],
    )


def _random_branch(name):
    """A subgraph that samples a distribution, for use as an `If` branch."""
    return helper.make_graph(
        [
            helper.make_node(
                "RandomNormal", [], [name], dtype=TensorProto.FLOAT, shape=[2]
            )
        ],
        name,
        [],
        [_tensor(name, TensorProto.FLOAT, [2])],
    )


def _random_op_models() -> dict[str, ModelProto]:
    """Models whose single node samples a distribution, with every input a constant.

    No `seed` attribute is set, so each evaluation of these nodes draws different values.
    """
    probabilities = onnx.numpy_helper.from_array(
        np.full((4, 4), 0.5, dtype=np.float32), "p"
    )
    return {
        "RandomNormal": _model(
            [
                helper.make_node(
                    "RandomNormal",
                    [],
                    ["y"],
                    name="draw",
                    dtype=TensorProto.FLOAT,
                    shape=[2, 3],
                )
            ],
            [],
            [_unshaped("y")],
            opset=MAX_OPSET,
        ),
        "Bernoulli": _model(
            [helper.make_node("Bernoulli", ["p"], ["y"], name="draw")],
            [],
            [_unshaped("y")],
            initializer=[probabilities],
            opset=MAX_OPSET,
        ),
        # Dropout samples a mask in training mode; only its inference-mode identity is in scope.
        "Dropout": _model(
            [
                helper.make_node(
                    "Dropout", ["p", "ratio", "training"], ["y"], name="drop"
                )
            ],
            [],
            [_unshaped("y")],
            initializer=[
                probabilities,
                onnx.numpy_helper.from_array(np.float32(0.5), "ratio"),
                onnx.numpy_helper.from_array(np.array(True), "training"),
            ],
            opset=MAX_OPSET,
        ),
    }


def _op_types(prepared) -> list[str]:
    return [node.op_type for node in prepared.model.graph.node]


def _initializer(prepared, name):
    for initializer in prepared.model.graph.initializer:
        if initializer.name == name:
            return onnx.numpy_helper.to_array(initializer)
    raise AssertionError(f"no initializer named `{name}`")


def _output_shape(prepared, name) -> tuple[int, ...]:
    for value_info in prepared.model.graph.output:
        if value_info.name == name:
            return shapes.static_shape(value_info.type)
    raise AssertionError(f"no graph output named `{name}`")


def _schema_revisions(op_type: str, domain: str = "") -> list[int]:
    return sorted(
        schema.since_version
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.name == op_type and schema.domain == domain
    )


def _reference_outputs(model, feeds: dict) -> list:
    """What ONNX's own evaluator — the executable spec — computes for `model`."""
    return list(ReferenceEvaluator(model).run(None, feeds))


def _assert_matches_reference(test: unittest.TestCase, original, prepared, feeds: dict):
    """The prepared graph computes what the original did, at the shapes it claims."""
    expected = _reference_outputs(original, feeds)
    actual = _reference_outputs(prepared.model, feeds)
    test.assertEqual(len(expected), len(actual))
    for value_info, want, got in zip(prepared.model.graph.output, expected, actual):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        test.assertEqual(
            _output_shape(prepared, value_info.name),
            tuple(np.asarray(got).shape),
            f"declared shape of `{value_info.name}` does not match the computed one",
        )


class BindDimsTest(unittest.TestCase):
    def _identity_model(self, shape, elem_type=TensorProto.FLOAT):
        return _model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [_tensor("x", elem_type, shape)],
            [_unshaped("y", elem_type)],
        )

    def test_unbound_symbolic_dim_defaults_to_one_and_is_recorded(self):
        prepared = _prepare(self._identity_model(["batch", 3]))

        self.assertEqual(prepared.dim_bindings, {"batch": 1})
        self.assertEqual(_output_shape(prepared, "y"), (1, 3))

    def test_explicit_binding_reaches_inputs_and_inferred_outputs(self):
        original = self._identity_model(["batch", 3])

        prepared = _prepare(original, dim_bindings={"batch": 4})

        self.assertEqual(prepared.dim_bindings, {"batch": 4})
        self.assertEqual(
            shapes.static_shape(prepared.model.graph.input[0].type), (4, 3)
        )
        _assert_matches_reference(
            self, original, prepared, {"x": np.zeros((4, 3), dtype=np.float32)}
        )

    def test_unnamed_unknown_dim_defaults_to_one(self):
        model = self._identity_model([None, 3])

        prepared = _prepare(model, dim_bindings={"batch": 4})

        self.assertEqual(prepared.dim_bindings, {})
        self.assertEqual(_output_shape(prepared, "y"), (1, 3))

    def test_binding_for_an_absent_dim_is_ignored(self):
        prepared = _prepare(self._identity_model([2, 3]), dim_bindings={"unused": 8})

        self.assertEqual(prepared.dim_bindings, {})
        self.assertEqual(_output_shape(prepared, "y"), (2, 3))

    def test_one_name_binds_every_occurrence(self):
        original = _model(
            [helper.make_node("Add", ["x", "z"], ["y"])],
            [
                _tensor("x", TensorProto.FLOAT, ["batch", 3]),
                _tensor("z", TensorProto.FLOAT, ["batch", 3]),
            ],
            [_unshaped("y")],
        )

        prepared = _prepare(original, dim_bindings={"batch": 5})

        self.assertEqual(prepared.dim_bindings, {"batch": 5})
        feeds = {
            "x": np.arange(15, dtype=np.float32).reshape(5, 3),
            "z": np.ones((5, 3), dtype=np.float32),
        }
        _assert_matches_reference(self, original, prepared, feeds)

    def test_zero_is_a_valid_binding(self):
        original = self._identity_model(["batch", 3])

        prepared = _prepare(original, dim_bindings={"batch": 0})

        self.assertEqual(_output_shape(prepared, "y"), (0, 3))
        _assert_matches_reference(
            self, original, prepared, {"x": np.zeros((0, 3), dtype=np.float32)}
        )

    def test_invalid_binding_values_are_rejected(self):
        for value in (-1, "4", 2.0, True, None):
            with self.subTest(value=value):
                with self.assertRaises(CompileError) as ctx:
                    _prepare(
                        self._identity_model(["batch", 3]),
                        dim_bindings={"batch": value},
                    )
                self.assertIn("batch", str(ctx.exception))

    def test_initializer_shadowing_an_input_becomes_a_weight(self):
        """Pre-IR-4 models declare initializers as inputs; they compile as static weights."""
        weights = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        original = _model(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            [
                _tensor("x", TensorProto.FLOAT, ["batch", 3]),
                _tensor("w", TensorProto.FLOAT, [3, 1]),
            ],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(weights, "w")],
            ir_version=3,
        )

        prepared = _prepare(original, dim_bindings={"batch": 2})

        self.assertEqual([entry.name for entry in prepared.model.graph.input], ["x"])
        self.assertEqual(
            [entry.name for entry in prepared.model.graph.initializer], ["w"]
        )
        self.assertEqual(_output_shape(prepared, "y"), (2, 1))

    def test_an_output_the_graph_does_not_compute_aliases_its_input(self):
        """Echoing an input as an output must not cost it the shape the binding gave it."""
        original = _model(
            [helper.make_node("Relu", ["x"], ["y"])],
            [_tensor("x", TensorProto.FLOAT, ["batch", 2])],
            [_tensor("x", TensorProto.FLOAT, ["batch", 2]), _unshaped("y")],
        )

        prepared = _prepare(original, dim_bindings={"batch": 3})

        self.assertEqual(_output_shape(prepared, "x"), (3, 2))
        _assert_matches_reference(
            self,
            original,
            prepared,
            {"x": np.arange(6, dtype=np.float32).reshape(3, 2) - 2},
        )

    def test_binding_reaches_a_declared_ml_output_shape(self):
        """`ai.onnx.ml` inference drops the batch dim; the binding reaches it through the
        output declaration, which is the only static description of it there is."""
        graph = helper.make_graph(
            [
                helper.make_node(
                    "LinearRegressor",
                    ["x"],
                    ["y"],
                    domain="ai.onnx.ml",
                    coefficients=[1.0, 2.0, 3.0],
                    intercepts=[0.5],
                    targets=1,
                )
            ],
            "g",
            [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
            [_tensor("y", TensorProto.FLOAT, ["batch", 1])],
        )
        original = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", BUNDLE_OPSET),
                helper.make_opsetid("ai.onnx.ml", 1),
            ],
        )

        prepared = _prepare(original, dim_bindings={"batch": 4})

        self.assertEqual(prepared.dim_bindings, {"batch": 4})
        self.assertEqual(_output_shape(prepared, "y"), (4, 1))
        _assert_matches_reference(
            self,
            original,
            prepared,
            {"x": np.arange(12, dtype=np.float32).reshape(4, 3)},
        )


class ConstantFoldingTest(unittest.TestCase):
    def _shape_chain_model(self, opset):
        """Flatten x by computing its element count from its own shape."""
        return _model(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                _constant_node("i0", [0]),
                _constant_node("i1", [1]),
                helper.make_node("Gather", ["s", "i0"], ["rows"]),
                helper.make_node("Gather", ["s", "i1"], ["cols"]),
                helper.make_node("Mul", ["rows", "cols"], ["count"]),
                helper.make_node("Reshape", ["x", "count"], ["y"]),
            ],
            [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
            [_unshaped("y")],
            opset=opset,
        )

    def test_shape_computation_chain_folds_to_a_constant(self):
        for opset in (BUNDLE_OPSET, MAX_OPSET):
            with self.subTest(opset=opset):
                original = self._shape_chain_model(opset)

                prepared = _prepare(original, dim_bindings={"batch": 4})

                self.assertEqual(_op_types(prepared), ["Reshape"])
                np.testing.assert_array_equal(_initializer(prepared, "count"), [12])
                self.assertEqual(_output_shape(prepared, "y"), (12,))
                _assert_matches_reference(
                    self,
                    original,
                    prepared,
                    {"x": np.arange(12, dtype=np.float32).reshape(4, 3)},
                )

    def test_folding_reaches_a_fixpoint_through_chained_constants(self):
        original = _model(
            [
                _constant_node("a", [1, 2, 3]),
                _constant_node("b", [10, 20, 30]),
                helper.make_node("Add", ["a", "b"], ["c"]),
                helper.make_node("Mul", ["c", "b"], ["y"]),
            ],
            [],
            [_unshaped("y", TensorProto.INT64)],
        )

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), [])
        np.testing.assert_array_equal(
            _initializer(prepared, "y"), _reference_outputs(original, {})[0]
        )

    def test_a_node_reading_one_constant_twice_folds(self):
        original = _model(
            [helper.make_node("Mul", ["c", "c"], ["y"])],
            [],
            [_unshaped("y", TensorProto.INT64)],
            initializer=[
                onnx.numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "c")
            ],
        )

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), [])
        np.testing.assert_array_equal(
            _initializer(prepared, "y"), _reference_outputs(original, {})[0]
        )

    def _generated_tensor_model(self, opset):
        """A tensor built out of nothing but the operands describing it."""
        return _model(
            [
                _constant_node("shape", [2, 3]),
                helper.make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["filled"],
                    value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.5]),
                ),
                _constant_node("start", 0, dtype=TensorProto.FLOAT),
                _constant_node("limit", 3, dtype=TensorProto.FLOAT),
                _constant_node("step", 1, dtype=TensorProto.FLOAT),
                helper.make_node("Range", ["start", "limit", "step"], ["counted"]),
                helper.make_node("Add", ["filled", "counted"], ["y"]),
            ],
            [],
            [_unshaped("y")],
            opset=opset,
        )

    def test_a_tensor_generated_out_of_fixed_operands_is_resolved_by_folding(self):
        """`ConstantOfShape` and `Range` state their whole result through their operands.

        Nothing of either is left once the graph fixes those operands — which is the only
        form the compiler accepts them in at all, since the result's shape is a function of
        their values — so neither op carries a kernel of its own.
        """
        original = self._generated_tensor_model(MAX_OPSET)

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), [])
        self.assertEqual(_output_shape(prepared, "y"), (2, 3))
        np.testing.assert_array_equal(
            _initializer(prepared, "y"), _reference_outputs(original, {})[0]
        )

    def test_a_window_and_a_mel_matrix_are_resolved_by_folding(self):
        """A window and a mel matrix are functions of their operands alone.

        Nothing of either is left once the graph fixes those operands — which is the only
        form the compiler accepts them in, their operands being their result's own shape —
        so neither carries a kernel, and the values come from the evaluator rather than from
        a reimplementation of the formulas.
        """
        original = _model(
            [
                _constant_node("size", 8, dtype=TensorProto.INT32),
                _constant_node("bins", 4, dtype=TensorProto.INT32),
                _constant_node("rate", 16000, dtype=TensorProto.INT32),
                _constant_node("low", 0.0, dtype=TensorProto.FLOAT),
                _constant_node("high", 8000.0, dtype=TensorProto.FLOAT),
                helper.make_node("HannWindow", ["size"], ["hann"]),
                helper.make_node("HammingWindow", ["size"], ["hamming"]),
                helper.make_node("BlackmanWindow", ["size"], ["blackman"], periodic=0),
                helper.make_node("Add", ["hann", "hamming"], ["summed"]),
                helper.make_node("Add", ["summed", "blackman"], ["windows"]),
                helper.make_node(
                    "MelWeightMatrix",
                    ["bins", "size", "rate", "low", "high"],
                    ["mel"],
                ),
            ],
            [],
            [_unshaped("windows"), _unshaped("mel")],
            opset=MAX_OPSET,
        )

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), [])
        self.assertEqual(_output_shape(prepared, "windows"), (8,))
        self.assertEqual(_output_shape(prepared, "mel"), (5, 4))
        expected = _reference_outputs(original, {})
        for name, want in zip(("windows", "mel"), expected):
            np.testing.assert_allclose(_initializer(prepared, name), want, rtol=1e-6)

    def test_a_generated_tensor_at_an_unvouchable_revision_is_left_for_dispatch(self):
        """Resting on folding means resting on where the evaluator is a valid oracle.

        Below the revision it implements, folding declines rather than applying semantics
        nothing can vouch for — and the node reaches dispatch, which has no kernel for it.
        """
        prepared = _prepare(self._generated_tensor_model(BUNDLE_OPSET))

        self.assertEqual(_op_types(prepared), ["ConstantOfShape", "Range", "Add"])

    def test_nodes_reading_runtime_values_are_left_alone(self):
        original = _model(
            [
                _constant_node("w", [1.0, 2.0, 3.0], dtype=TensorProto.FLOAT),
                helper.make_node("Add", ["x", "w"], ["y"]),
            ],
            [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
            [_unshaped("y")],
        )

        prepared = _prepare(original, dim_bindings={"batch": 2})

        self.assertEqual(_op_types(prepared), ["Add"])
        _assert_matches_reference(
            self, original, prepared, {"x": np.ones((2, 3), dtype=np.float32)}
        )

    def test_op_the_evaluator_cannot_vouch_for_is_not_folded(self):
        """Opset-7 `Add` carries `broadcast`/`axis`; only its modern semantics are implemented."""
        constants = [
            onnx.numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "a"),
            onnx.numpy_helper.from_array(np.array([10, 20, 30], dtype=np.int64), "b"),
        ]
        add = [helper.make_node("Add", ["a", "b"], ["y"])]
        output = [_unshaped("y", TensorProto.INT64)]

        stale = _prepare(_model(add, [], output, initializer=constants, opset=7))
        current = _prepare(
            _model(
                add,
                [],
                output,
                initializer=constants,
                opset=max(_schema_revisions("Add")),
            )
        )

        self.assertEqual(_op_types(stale), ["Add"])
        self.assertEqual(_op_types(current), [])

    def test_constant_condition_selects_a_branch(self):
        original = _model(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    name="branch",
                    then_branch=_constant_branch("then", [1.0, 2.0]),
                    else_branch=_constant_branch("else", [3.0, 4.0]),
                )
            ],
            [],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(np.array(True), "cond")],
            opset=MAX_OPSET,
        )

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), [])
        np.testing.assert_array_equal(
            _initializer(prepared, "y"), _reference_outputs(original, {})[0]
        )

    def test_subgraph_reading_a_runtime_value_keeps_the_node(self):
        """An `If` whose branch reads a runtime tensor cannot be folded, so it is rejected."""
        passthrough = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["t"])], "then", [], [_unshaped("t")]
        )
        original = _model(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    name="branch",
                    then_branch=passthrough,
                    else_branch=_constant_branch("else", [0.0, 0.0]),
                )
            ],
            [_tensor("x", TensorProto.FLOAT, [2])],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(np.array(True), "cond")],
            opset=MAX_OPSET,
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(original)
        self.assertIn("branch", str(ctx.exception))

    def test_initializers_left_unused_by_folding_are_dropped(self):
        original = _model(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Reshape", ["x", "s"], ["y"]),
            ],
            [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
            [_unshaped("y")],
            initializer=[
                onnx.numpy_helper.from_array(np.zeros(4, dtype=np.float32), "unused")
            ],
        )

        prepared = _prepare(original, dim_bindings={"batch": 2})

        self.assertEqual(
            [entry.name for entry in prepared.model.graph.initializer], ["s"]
        )
        self.assertEqual(_op_types(prepared), ["Reshape"])

    def test_shapes_of_folded_away_tensors_are_dropped(self):
        """Stale `value_info` would otherwise offer the emitter buffers for absent tensors."""
        prepared = _prepare(
            self._shape_chain_model(BUNDLE_OPSET), dim_bindings={"batch": 4}
        )

        self.assertEqual([entry.name for entry in prepared.model.graph.value_info], [])

    def test_random_draws_are_never_folded(self):
        """A draw is not a constant; baking one in would compile an unsupported op wrongly."""
        for op_type, model in _random_op_models().items():
            with self.subTest(op_type=op_type):
                prepared = _prepare(model)

                self.assertEqual(_op_types(prepared), [op_type])

    def test_a_random_draw_inside_a_branch_blocks_folding(self):
        """Folding the enclosing `If` would bake the draw in; it is refused instead."""
        original = _model(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    name="branch",
                    then_branch=_random_branch("then"),
                    else_branch=_constant_branch("else", [0.0, 0.0]),
                )
            ],
            [],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(np.array(True), "cond")],
            opset=MAX_OPSET,
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(original)

        message = str(ctx.exception)
        self.assertIn("branch", message)
        self.assertIn("control flow", message)

    def test_a_random_op_does_not_make_preparing_non_deterministic(self):
        loaded = loader.load_model(_random_op_models()["RandomNormal"])

        first = frontend.prepare_model(loaded)
        second = frontend.prepare_model(loaded)

        self.assertEqual(
            first.model.SerializeToString(), second.model.SerializeToString()
        )


class EvaluatorFaithfulnessTest(unittest.TestCase):
    def test_latest_revision_is_faithful(self):
        for op_type in ("Add", "Cast", "Concat", "Shape", "Reshape"):
            with self.subTest(op_type=op_type):
                self.assertTrue(
                    folding.evaluator_is_version_faithful("", op_type, MAX_OPSET)
                )

    def test_superseded_semantics_are_refused(self):
        """`Add` gained numpy broadcasting at opset 7 and dropped `axis`; only one is implemented."""
        self.assertFalse(folding.evaluator_is_version_faithful("", "Add", 6))
        self.assertFalse(folding.evaluator_is_version_faithful("", "Add", 7))
        self.assertTrue(
            folding.evaluator_is_version_faithful(
                "", "Add", max(_schema_revisions("Add"))
            )
        )

    def test_a_versioned_implementation_covers_its_own_revision(self):
        """`Cast` has implementations at revisions 1 and 19; opset 13 lies between them."""
        self.assertTrue(folding.evaluator_is_version_faithful("", "Cast", 19))
        self.assertFalse(folding.evaluator_is_version_faithful("", "Cast", 13))

    def test_domain_alias_is_accepted(self):
        self.assertEqual(
            folding.evaluator_is_version_faithful("ai.onnx", "Concat", MAX_OPSET),
            folding.evaluator_is_version_faithful("", "Concat", MAX_OPSET),
        )

    def test_unknown_op_or_domain_is_not_faithful(self):
        self.assertFalse(
            folding.evaluator_is_version_faithful("", "NoSuchOp", MAX_OPSET)
        )
        self.assertFalse(
            folding.evaluator_is_version_faithful("com.example", "Add", MAX_OPSET)
        )


class StaticVerificationTest(unittest.TestCase):
    def test_data_dependent_output_shape_names_the_node_and_op(self):
        model = _model(
            [helper.make_node("NonZero", ["x"], ["y"], name="find")],
            [_tensor("x", TensorProto.FLOAT, ["batch", 3])],
            [_unshaped("y", TensorProto.INT64)],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("find", message)
        self.assertIn("NonZero", message)
        self.assertIn("depends on input data", message)

    def test_shape_that_only_runtime_data_determines_is_rejected(self):
        """A declared symbolic output shape never substitutes for one the graph must compute.

        Binding the names the model declares its result under is not enough: the shape the
        graph actually computes comes from an operand nothing fixes, and the error names it.
        """
        model = _model(
            [helper.make_node("Reshape", ["x", "s"], ["y"], name="reshape")],
            [
                _tensor("x", TensorProto.FLOAT, ["batch", 3]),
                _tensor("s", TensorProto.INT64, [2]),
            ],
            [_tensor("y", TensorProto.FLOAT, ["rows", "cols"])],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model, dim_bindings={"batch": 2, "rows": 2, "cols": 3})

        message = str(ctx.exception)
        self.assertIn("reshape", message)
        self.assertIn("`s`", message)
        self.assertIn("depends on input data", message)

    def test_every_operand_a_view_takes_its_shape_from_is_checked(self):
        """Each of these ops reads a shape, an axis list or a repeat count as data.

        Whichever operand carries it, the result's shape is a function of its *values*, so a
        model that computes one at run time is rejected by name rather than compiled for
        whatever the operand happened to hold.
        """
        cases = (
            ("Reshape", ["x", "p"], {}),
            ("Expand", ["x", "p"], {}),
            ("Tile", ["x", "p"], {}),
            ("Squeeze", ["x", "p"], {}),
            ("Unsqueeze", ["x", "p"], {}),
            ("Split", ["x", "p"], {"axis": 0}),
            ("Slice", ["x", "p", "p"], {}),
        )
        for op_type, inputs, attributes in cases:
            with self.subTest(op_type=op_type):
                model = _model(
                    [
                        helper.make_node(
                            op_type, inputs, ["y"], name="view", **attributes
                        )
                    ],
                    [
                        _tensor("x", TensorProto.FLOAT, [2, 3]),
                        _tensor("p", TensorProto.INT64, [2]),
                    ],
                    [_unshaped("y")],
                )

                with self.assertRaises(CompileError) as ctx:
                    _prepare(model)

                message = str(ctx.exception)
                self.assertIn("view", message)
                self.assertIn("`p`", message)
                self.assertIn("depends on input data", message)

    def test_every_operand_that_sizes_a_result_is_checked(self):
        """These ops read their result's own extents out of an operand's values.

        A padding, a depth, a `k`, the shape a tensor is generated at: none of them is a
        shape the graph states, so an operand nothing fixes leaves the result's size a
        function of input data — and a buffer the compiler cannot size.
        """
        float_result = {"y": TensorProto.FLOAT}
        cases = (
            ("ConstantOfShape", ["p"], float_result, {}),
            ("Range", ["s", "s", "s"], {"y": TensorProto.INT64}, {}),
            ("OneHot", ["i", "s", "v"], float_result, {}),
            ("Pad", ["x", "p"], float_result, {}),
            ("CenterCropPad", ["x", "p"], float_result, {}),
            (
                "TopK",
                ["x", "p"],
                {"y": TensorProto.FLOAT, "z": TensorProto.INT64},
                {"axis": 0},
            ),
        )
        for op_type, inputs, outputs, attributes in cases:
            with self.subTest(op_type=op_type):
                model = _model(
                    [
                        helper.make_node(
                            op_type, inputs, list(outputs), name="size", **attributes
                        )
                    ],
                    [
                        _tensor("x", TensorProto.FLOAT, [2, 3]),
                        _tensor("i", TensorProto.INT64, [2]),
                        _tensor("p", TensorProto.INT64, [2]),
                        _tensor("s", TensorProto.INT64, []),
                        _tensor("v", TensorProto.FLOAT, [2]),
                    ],
                    [_unshaped(name, elem_type) for name, elem_type in outputs.items()],
                )

                with self.assertRaises(CompileError) as ctx:
                    _prepare(model)

                message = str(ctx.exception)
                self.assertIn("size", message)
                self.assertIn("depends on input data", message)

    def test_every_operand_a_signal_op_takes_its_extents_from_is_checked(self):
        """The transforms and the windows state their result's shape through operands too.

        A window is `size` samples and nothing else, a mel matrix is its bin counts, a
        short-time transform is one frame per step of `frame_step` each `frame_length`
        long, and a `dft_length` is the extent of the axis it transforms — so an operand
        nothing fixes leaves a buffer the compiler cannot size. Each is declared with the
        result shape it would have had, since ONNX's own inference stops at the same
        operands and would otherwise report an unknown shape instead.
        """
        cases = (
            ("HannWindow", ["n"], [_tensor("y", TensorProto.FLOAT, [8])], {}),
            ("HammingWindow", ["n"], [_tensor("y", TensorProto.FLOAT, [8])], {}),
            ("BlackmanWindow", ["n"], [_tensor("y", TensorProto.FLOAT, [8])], {}),
            (
                "MelWeightMatrix",
                ["n", "n", "n", "f", "f"],
                [_tensor("y", TensorProto.FLOAT, [5, 8])],
                {},
            ),
            (
                "STFT",
                ["signal", "s", "", "s"],
                [_tensor("y", TensorProto.FLOAT, [1, 3, 5, 2])],
                {},
            ),
            ("DFT", ["signal", "s"], [_tensor("y", TensorProto.FLOAT, [1, 16, 2])], {}),
        )
        for op_type, inputs, outputs, attributes in cases:
            with self.subTest(op_type=op_type):
                model = _model(
                    [
                        helper.make_node(
                            op_type,
                            inputs,
                            [entry.name for entry in outputs],
                            name="sized",
                            **attributes,
                        )
                    ],
                    [
                        _tensor("signal", TensorProto.FLOAT, [1, 16, 1]),
                        _tensor("n", TensorProto.INT32, []),
                        _tensor("s", TensorProto.INT64, []),
                        _tensor("f", TensorProto.FLOAT, []),
                    ],
                    outputs,
                    opset=17,
                )

                with self.assertRaises(CompileError) as ctx:
                    _prepare(model)

                message = str(ctx.exception)
                self.assertIn("sized", message)
                self.assertIn("depends on input data", message)

    def test_a_transform_axis_is_a_shape_operand_only_where_it_resizes_an_axis(self):
        """`onesided` halves the axis it lands on, which makes which axis that is part of
        the result's shape; without it the result carries the operand's own extents
        whichever axis is transformed, and a run-time axis is a value, not a shape."""

        def transform(**attributes):
            return _model(
                [
                    helper.make_node(
                        "DFT", ["signal", "", "axis"], ["y"], name="dft", **attributes
                    )
                ],
                [
                    _tensor("signal", TensorProto.FLOAT, [1, 8, 1]),
                    _tensor("axis", TensorProto.INT64, []),
                ],
                [_unshaped("y")],
                opset=MAX_OPSET,
            )

        prepared = _prepare(transform())

        self.assertEqual(_op_types(prepared), ["DFT"])
        self.assertEqual(_output_shape(prepared, "y"), (1, 8, 2))
        with self.assertRaises(CompileError) as ctx:
            _prepare(transform(onesided=1))
        self.assertIn("depends on input data", str(ctx.exception))

    def test_a_view_shape_an_initializer_fixes_is_accepted(self):
        model = _model(
            [helper.make_node("Tile", ["x", "repeats"], ["y"], name="tile")],
            [_tensor("x", TensorProto.FLOAT, [2, 3])],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(np.array([2, 3]), "repeats")],
        )

        prepared = _prepare(model)

        self.assertEqual(
            shapes.static_shape(shapes.tensor_types(prepared.model.graph)["y"]), (4, 9)
        )

    def test_a_squeeze_axes_operand_with_no_elements_is_kept(self):
        """An empty axes list squeezes nothing, where an absent one squeezes everything.

        The reductions are the one family ONNX defines the two the same way for, so they are
        the only one whose empty operand the frontend drops; dropping this one would turn a
        `[1, 3]` result into a `[3]` one.
        """
        model = _model(
            [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="squeeze")],
            [_tensor("x", TensorProto.FLOAT, [1, 3])],
            [_unshaped("y")],
            initializer=[
                onnx.numpy_helper.from_array(np.array([], dtype=np.int64), "axes")
            ],
        )

        prepared = _prepare(model)

        self.assertEqual(list(prepared.model.graph.node[0].input), ["x", "axes"])
        self.assertEqual(
            shapes.static_shape(shapes.tensor_types(prepared.model.graph)["y"]), (1, 3)
        )

    def test_reduction_axes_that_only_runtime_data_names_is_rejected(self):
        """Which axes a reduction removes decides its output shape, so they must be static."""
        model = _model(
            [helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="total")],
            [
                _tensor("x", TensorProto.FLOAT, [2, 3]),
                _tensor("axes", TensorProto.INT64, [1]),
            ],
            [_unshaped("y")],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("total", message)
        self.assertIn("`axes`", message)
        self.assertIn("depends on input data", message)

    def test_reduction_axes_an_initializer_fixes_are_accepted(self):
        model = _model(
            [helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="total")],
            [_tensor("x", TensorProto.FLOAT, [2, 3])],
            [_unshaped("y")],
            initializer=[onnx.numpy_helper.from_array(np.array([1]), "axes")],
        )

        prepared = _prepare(model)

        self.assertEqual(
            shapes.static_shape(shapes.tensor_types(prepared.model.graph)["y"]), (2, 1)
        )

    def test_an_axes_operand_with_no_elements_is_left_out(self):
        """An empty axes tensor names no axes, which is what passing none at all means.

        ONNX's shape inference does not reason about the values of an operand it cannot see,
        so it leaves the result of such a reduction untyped; dropping the operand is what
        makes the shape follow from the graph.
        """
        model = _model(
            [helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="total")],
            [
                _tensor("x", TensorProto.FLOAT, [2, 3]),
                _tensor("axes", TensorProto.INT64, [0]),
            ],
            [_unshaped("y")],
        )

        prepared = _prepare(model)

        self.assertEqual(list(prepared.model.graph.node[0].input), ["x", ""])
        _assert_matches_reference(
            self,
            model,
            prepared,
            {
                "x": np.arange(6, dtype=np.float32).reshape(2, 3),
                "axes": np.array([], dtype=np.int64),
            },
        )

    def test_a_short_time_transform_states_the_onesided_default_it_is_read_under(self):
        """ONNX's shape inference reads a default for `onesided` its schema does not declare.

        The schema's default is 1 and the reference evaluator applies it, so the op returns
        the non-redundant half of each frame's spectrum; inference falls back to 0 and would
        size the result at the whole frame length. Stating the schema's own default leaves
        what the node computes alone and makes the two agree.
        """
        model = _model(
            [helper.make_node("STFT", ["x", "step", "", "length"], ["y"], name="stft")],
            [_tensor("x", TensorProto.FLOAT, [1, 16, 1])],
            [_unshaped("y")],
            initializer=[
                onnx.numpy_helper.from_array(np.array(value, dtype=np.int64), name)
                for name, value in (("step", 4), ("length", 8))
            ],
            opset=17,
        )
        declared = onnx.defs.get_schema("STFT").attributes["onesided"].default_value.i

        prepared = _prepare(model)

        (stated,) = prepared.model.graph.node[0].attribute
        self.assertEqual((stated.name, stated.i), ("onesided", declared))
        self.assertEqual(_output_shape(prepared, "y"), (1, 3, 5, 2))
        _assert_matches_reference(
            self,
            model,
            prepared,
            {"x": np.arange(16, dtype=np.float32).reshape(1, 16, 1)},
        )

    def test_shape_inference_failure_names_the_graph_and_node(self):
        model = _model(
            [helper.make_node("Add", ["x", "z"], ["y"], name="mismatch")],
            [
                _tensor("x", TensorProto.FLOAT, [2, 3]),
                _tensor("z", TensorProto.FLOAT, [4, 5]),
            ],
            [_unshaped("y")],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("shape inference failed", message)
        self.assertIn("mismatch", message)

    def test_tensor_without_an_inferred_type_is_reported(self):
        """`verify_static` refuses a tensor nothing described, rather than assuming one."""
        model = _model(
            [helper.make_node("Identity", ["x"], ["y"], name="copy")],
            [_tensor("x", TensorProto.FLOAT, [2, 2])],
            [_unshaped("y")],
        )
        model.graph.output[0].ClearField("type")

        with self.assertRaises(CompileError) as ctx:
            verify.verify_static(model)

        message = str(ctx.exception)
        self.assertIn("copy", message)
        self.assertIn("`y`", message)

    def test_unsupported_element_types_name_the_tensor_and_type(self):
        for elem_type in (
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.STRING,
            TensorProto.COMPLEX64,
        ):
            with self.subTest(elem_type=elem_type):
                model = _model(
                    [helper.make_node("Identity", ["x"], ["y"])],
                    [_tensor("x", elem_type, [2, 2])],
                    [_unshaped("y", elem_type)],
                )

                with self.assertRaises(CompileError) as ctx:
                    _prepare(model)

                message = str(ctx.exception)
                self.assertIn("`x`", message)
                self.assertIn(onnx.TensorProto.DataType.Name(elem_type), message)

    def test_unsupported_weight_type_names_the_initializer(self):
        model = _model(
            [
                helper.make_node("CastLike", ["w", "x"], ["wf"], name="cast"),
                helper.make_node("Add", ["x", "wf"], ["y"], name="add"),
            ],
            [_tensor("x", TensorProto.FLOAT, [2, 2])],
            [_unshaped("y")],
            initializer=[
                onnx.numpy_helper.from_array(np.ones((2, 2), dtype=np.float16), "w")
            ],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("initializer `w`", message)
        self.assertIn("FLOAT16", message)

    def test_non_tensor_io_is_rejected(self):
        sequence = helper.make_value_info(
            "s",
            helper.make_sequence_type_proto(
                helper.make_tensor_type_proto(TensorProto.FLOAT, [2])
            ),
        )
        model = _model(
            [helper.make_node("ConcatFromSequence", ["s"], ["y"], axis=0)],
            [sequence],
            [_unshaped("y")],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("`s`", message)
        self.assertIn("sequence", message)

    def test_sparse_initializers_are_rejected(self):
        sparse = helper.make_sparse_tensor(
            helper.make_tensor("sv", TensorProto.FLOAT, [1], [1.0]),
            helper.make_tensor("si", TensorProto.INT64, [1, 2], [0, 0]),
            [2, 2],
        )
        model = _model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [_tensor("x", TensorProto.FLOAT, [2, 2])],
            [_unshaped("y")],
            sparse_initializer=[sparse],
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)
        self.assertIn("sparse", str(ctx.exception))

    def test_control_flow_that_survives_folding_is_rejected(self):
        model = _model(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    name="branch",
                    then_branch=_constant_branch("then", [1.0]),
                    else_branch=_constant_branch("else", [2.0]),
                )
            ],
            [_tensor("cond", TensorProto.BOOL, [])],
            [_unshaped("y")],
            opset=MAX_OPSET,
        )

        with self.assertRaises(CompileError) as ctx:
            _prepare(model)

        message = str(ctx.exception)
        self.assertIn("branch", message)
        self.assertIn("control flow", message)

    def test_zero_element_tensors_are_static(self):
        original = _model(
            [helper.make_node("Concat", ["a", "b"], ["y"], axis=0)],
            [
                _tensor("a", TensorProto.FLOAT, [0, 3]),
                _tensor("b", TensorProto.FLOAT, ["batch", 3]),
            ],
            [_unshaped("y")],
        )

        prepared = _prepare(original, dim_bindings={"batch": 2})

        self.assertEqual(_output_shape(prepared, "y"), (2, 3))
        _assert_matches_reference(
            self,
            original,
            prepared,
            {
                "a": np.zeros((0, 3), dtype=np.float32),
                "b": np.ones((2, 3), dtype=np.float32),
            },
        )


class ShapeInferenceGapTest(unittest.TestCase):
    """Where ONNX's own inference stops short of a shape it nonetheless guarantees."""

    def test_a_body_it_cannot_infer_through_is_retried_without_strict_mode(self):
        """Strict inference recurses into the function body ONNX defines for the op.

        MeanVarianceNormalization's builds its `axes` from a Constant that carries nothing
        at all unless the node sets the attribute, and inference raises on it — while the
        shapes it derives without strict mode are complete, so the model still compiles.
        """
        model = _model(
            [helper.make_node("MeanVarianceNormalization", ["x"], ["y"], name="mvn")],
            [_tensor("x", TensorProto.FLOAT, [2, 3, 2, 2])],
            [_unshaped("y")],
            opset=13,
        )

        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.shape_inference.infer_shapes(model, strict_mode=True)
        prepared = _prepare(model)

        self.assertEqual(_output_shape(prepared, "y"), (2, 3, 2, 2))

    def test_group_normalization_gives_its_result_the_shape_of_its_operand(self):
        """Inference stops inside GroupNormalization's body, which reshapes through shapes
        it computes, leaving its result a rank it never states — and every tensor after it
        none either, though the schema says `Y` has the shape of `X`."""
        model = _model(
            [
                helper.make_node(
                    "GroupNormalization",
                    ["x", "scale", "bias"],
                    ["h"],
                    name="norm",
                    num_groups=2,
                ),
                helper.make_node("Relu", ["h"], ["y"], name="relu"),
            ],
            [
                _tensor("x", TensorProto.FLOAT, [2, 4, 3]),
                _tensor("scale", TensorProto.FLOAT, [4]),
                _tensor("bias", TensorProto.FLOAT, [4]),
            ],
            [_unshaped("y")],
        )

        prepared = _prepare(model)

        self.assertEqual(_output_shape(prepared, "y"), (2, 4, 3))
        _assert_matches_reference(
            self,
            model,
            prepared,
            {
                "x": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
                "scale": np.arange(4, dtype=np.float32),
                "bias": np.ones(4, dtype=np.float32),
            },
        )

    def test_a_shape_only_folding_makes_derivable_is_not_shadowed(self):
        """A tensor inference could not size on the first pass takes the shape of the last.

        Folding is what gives inference the operand values it was missing, so the two run in
        turn — and ONNX leaves behind an entry naming only the element type for every tensor
        it stopped at. Those entries outlive the round that produced them and are read before
        the graph's own outputs, so a stale one would hide the shape the next round derives.
        """
        model = _model(
            [
                helper.make_node("Shape", ["x"], ["s"], name="shape"),
                helper.make_node("Sub", ["s", "k"], ["ends"], name="sub"),
                helper.make_node("Slice", ["x", "zeros", "ends"], ["y"], name="slice"),
            ],
            [_tensor("x", TensorProto.FLOAT, [2, 4, 8])],
            [_unshaped("y")],
            initializer=[
                onnx.numpy_helper.from_array(np.array([0, 1, 3], np.int64), "k"),
                onnx.numpy_helper.from_array(np.zeros(3, np.int64), "zeros"),
            ],
        )

        prepared = _prepare(model)

        self.assertEqual(_output_shape(prepared, "y"), (2, 3, 5))
        _assert_matches_reference(
            self,
            model,
            prepared,
            {"x": np.arange(64, dtype=np.float32).reshape(2, 4, 8)},
        )


class PrepareBundleModelTest(unittest.TestCase):
    def test_ml_node_model_takes_its_shape_from_the_declared_output(self):
        """`ai.onnx.ml` inference stops at the batch dimension; the declaration carries it."""
        original = onnx.load(str(OPS_ARTIFACTS / "linreg" / "model.onnx"))

        prepared = _prepare(original)

        self.assertEqual(
            shapes.static_shape(prepared.model.graph.input[0].type), (1, 3)
        )
        self.assertEqual(_output_shape(prepared, "variable"), (1, 1))
        _assert_matches_reference(
            self,
            original,
            prepared,
            {"float_input": np.ones((1, 3), dtype=np.float32)},
        )

    def test_standard_node_model_folds_its_constant_node(self):
        original = onnx.load(str(OPS_ARTIFACTS / "concat_reduce" / "model.onnx"))

        prepared = _prepare(original)

        self.assertEqual(_op_types(prepared), ["Concat", "ReduceSum"])
        _assert_matches_reference(
            self,
            original,
            prepared,
            {
                name: np.full((1, 1), index + 1, dtype=np.float32)
                for index, name in enumerate(["input1", "input2", "input3"])
            },
        )

    def test_preparing_does_not_mutate_the_loaded_model(self):
        loaded = loader.load_model(OPS_ARTIFACTS / "concat_reduce" / "model.onnx")
        before = loaded.model.SerializeToString()

        frontend.prepare_model(loaded, dim_bindings={"batch": 3})

        self.assertEqual(loaded.model.SerializeToString(), before)

    def test_preparing_twice_produces_identical_models(self):
        loaded = loader.load_model(OPS_ARTIFACTS / "linreg" / "model.onnx")

        first = frontend.prepare_model(loaded, dim_bindings={"batch": 2})
        second = frontend.prepare_model(loaded, dim_bindings={"batch": 2})

        self.assertEqual(
            first.model.SerializeToString(), second.model.SerializeToString()
        )


class ElementTypeTest(unittest.TestCase):
    def test_every_supported_type_has_a_fixed_width_c_type(self):
        expected = {
            TensorProto.FLOAT: "float",
            TensorProto.DOUBLE: "double",
            TensorProto.INT8: "int8_t",
            TensorProto.INT16: "int16_t",
            TensorProto.INT32: "int32_t",
            TensorProto.INT64: "int64_t",
            TensorProto.UINT8: "uint8_t",
            TensorProto.UINT16: "uint16_t",
            TensorProto.UINT32: "uint32_t",
            TensorProto.UINT64: "uint64_t",
            TensorProto.BOOL: "uint8_t",
        }
        self.assertEqual(dtypes.C_TYPES, expected)
        for elem_type, c_type in expected.items():
            self.assertTrue(dtypes.is_supported(elem_type))
            self.assertEqual(dtypes.c_type(elem_type), c_type)

    def test_unsupported_type_raises_and_names_the_type(self):
        self.assertFalse(dtypes.is_supported(TensorProto.FLOAT16))
        with self.assertRaises(CompileError) as ctx:
            dtypes.c_type(TensorProto.FLOAT16)
        self.assertIn("FLOAT16", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
