"""The ONNX-ML tree ensembles: converted forests, and the models the compiler refuses.

What each op computes on a single node is settled by the conformance suite (opset 5's two
corpus tests) and by the differential sweep against the reference evaluator. Neither reaches
where these ops actually turn up: a scikit-learn converter emits `ai.onnx.ml` opset **1**,
which the reference evaluator is not version-faithful for and therefore cannot be the oracle
of. The parity tests below run those converted forests against onnxruntime instead — the
second oracle, independent of both the compiler and the reference evaluator — and the rest of
the module covers the error contracts, which no sweep asserts.
"""

from __future__ import annotations

import shutil

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import TensorProto, helper  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402

ML_OPSET = 5

requires_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)

_SAMPLES = 32
_FEATURES = 4


def _model(nodes, inputs, outputs, ml=ML_OPSET):
    """A model whose intermediates carry no `value_info`, as a converter's output does."""
    model = helper.make_model(
        helper.make_graph(nodes, "trees", list(inputs), list(outputs)),
        opset_imports=[helper.make_opsetid("ai.onnx.ml", ml)],
    )
    model.ir_version = 10
    return model


def _regressor(**attributes):
    """A one-stump, one-target regressor, with `attributes` overriding what it declares."""
    declared = {
        "n_targets": 1,
        "nodes_treeids": [0, 0, 0],
        "nodes_nodeids": [0, 1, 2],
        "nodes_featureids": [0, 0, 0],
        "nodes_modes": ["BRANCH_LEQ", "LEAF", "LEAF"],
        "nodes_values": [0.5, 0.0, 0.0],
        "nodes_truenodeids": [1, 0, 0],
        "nodes_falsenodeids": [2, 0, 0],
        "target_treeids": [0, 0],
        "target_nodeids": [1, 2],
        "target_ids": [0, 0],
        "target_weights": [1.5, -2.5],
        **attributes,
    }
    return _model(
        [
            helper.make_node(
                "TreeEnsembleRegressor",
                ["X"],
                ["Y"],
                name="ensemble",
                domain="ai.onnx.ml",
                **declared,
            )
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])],
        [helper.make_empty_tensor_value_info("Y")],
    )


def _classifier(**attributes):
    """A one-stump, two-class classifier, with `attributes` overriding what it declares."""
    declared = {
        "classlabels_int64s": [10, 20],
        "nodes_treeids": [0, 0, 0],
        "nodes_nodeids": [0, 1, 2],
        "nodes_featureids": [0, 0, 0],
        "nodes_modes": ["BRANCH_LEQ", "LEAF", "LEAF"],
        "nodes_values": [0.5, 0.0, 0.0],
        "nodes_truenodeids": [1, 0, 0],
        "nodes_falsenodeids": [2, 0, 0],
        "class_treeids": [0, 0],
        "class_nodeids": [1, 2],
        "class_ids": [0, 1],
        "class_weights": [1.0, 1.0],
        **attributes,
    }
    return _model(
        [
            helper.make_node(
                "TreeEnsembleClassifier",
                ["X"],
                ["Y", "Z"],
                name="ensemble",
                domain="ai.onnx.ml",
                **declared,
            )
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])],
        [
            helper.make_empty_tensor_value_info("Y"),
            helper.make_empty_tensor_value_info("Z"),
        ],
    )


# --------------------------------------------------------------------------------------
# Converted scikit-learn forests, against onnxruntime
# --------------------------------------------------------------------------------------


def _converted(estimator, data, target, **options):
    """The ONNX a scikit-learn converter emits for `estimator`, at `data`'s exact shape."""
    skl2onnx = pytest.importorskip("skl2onnx")
    data_types = pytest.importorskip("skl2onnx.common.data_types")
    return skl2onnx.convert_sklearn(
        estimator.fit(data, target),
        initial_types=[("X", data_types.FloatTensorType(list(data.shape)))],
        **options,
    )


def _session(model):
    runtime = pytest.importorskip("onnxruntime")
    runtime.set_default_logger_severity(3)
    return runtime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _matches_onnxruntime(model, feeds, tmp_path):
    """Every output of the compiled artifact against the one onnxruntime computes for it."""
    compiled = compile_onnx(model, tmp_path).load()
    outputs = compiled.run(feeds)
    expected = _session(model).run(None, dict(feeds))
    assert [spec.name for spec in compiled.outputs] == [
        entry.name for entry in model.graph.output
    ]
    for entry, want in zip(model.graph.output, expected):
        got, want = outputs[entry.name], np.asarray(want)
        assert got.dtype == want.dtype, entry.name
        assert got.shape == want.shape, entry.name
        if want.dtype.kind == "f":
            np.testing.assert_allclose(
                got, want, rtol=1e-5, atol=1e-6, err_msg=entry.name
            )
        else:
            np.testing.assert_array_equal(got, want, err_msg=entry.name)


def _fitted_data(seed=0):
    generator = np.random.default_rng(seed)
    return generator.normal(size=(_SAMPLES, _FEATURES)).astype(np.float32)


def _regression_target(data):
    return (2 * data[:, 0] + data[:, 1] - data[:, 3]).astype(np.float64)


@requires_c_compiler
@pytest.mark.parametrize(
    ("module", "name", "arguments"),
    [
        ("sklearn.tree", "DecisionTreeRegressor", {"max_depth": 4}),
        (
            "sklearn.ensemble",
            "RandomForestRegressor",
            {"n_estimators": 5, "max_depth": 3},
        ),
        (
            "sklearn.ensemble",
            "GradientBoostingRegressor",
            {"n_estimators": 5, "max_depth": 3},
        ),
        (
            "sklearn.ensemble",
            "ExtraTreesRegressor",
            {"n_estimators": 4, "max_depth": 3},
        ),
    ],
)
def test_a_converted_regressor_matches_onnxruntime(tmp_path, module, name, arguments):
    """The `AVERAGE` a forest aggregates with and the `base_values` a boosted one offsets."""
    estimators = pytest.importorskip(module)
    data = _fitted_data()

    model = _converted(
        getattr(estimators, name)(random_state=0, **arguments),
        data,
        _regression_target(data),
    )

    assert [node.op_type for node in model.graph.node] == ["TreeEnsembleRegressor"]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_multi_target_regressor_matches_onnxruntime(tmp_path):
    """One leaf weighting several targets, which is what the flat leaf ranges are for.

    The converter declares a one-column result for a forest whose `n_targets` is two, which
    onnxruntime pays no attention to; the compiler refuses the disagreement rather than
    writing two columns into a buffer the header would size for one, and compiles the same
    model once that declaration is dropped.
    """
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=1)
    targets = np.stack(
        [_regression_target(data), data[:, 2].astype(np.float64)], axis=1
    )
    model = _converted(
        ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, random_state=0),
        data,
        targets,
    )

    with pytest.raises(CompileError, match="addresses a result of shape"):
        compile_onnx(model, tmp_path / "declared")
    model.graph.output[0].ClearField("type")

    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
@pytest.mark.parametrize("classes", [2, 3], ids=["binary", "multiclass"])
def test_a_converted_classifier_matches_onnxruntime(tmp_path, classes):
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=2)
    labels = np.digitize(data[:, 0], np.linspace(-1, 1, classes - 1)).astype(np.int64)

    model = _converted(
        ensemble.RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0),
        data,
        labels,
        options={"zipmap": False},
    )

    assert [node.op_type for node in model.graph.node] == ["TreeEnsembleClassifier"]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_boosted_classifier_matches_onnxruntime(tmp_path):
    """Gradient boosting is where a classifier carries `base_values` and a transform."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=3)
    labels = (data[:, 0] + data[:, 1] > 0).astype(np.int64)

    model = _converted(
        ensemble.GradientBoostingClassifier(
            n_estimators=5, max_depth=3, random_state=0
        ),
        data,
        labels,
        options={"zipmap": False},
    )
    (node,) = [
        entry for entry in model.graph.node if entry.op_type == "TreeEnsembleClassifier"
    ]

    assert {entry.name for entry in node.attribute} >= {"base_values", "post_transform"}
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_classifier_keeps_its_probabilities_through_the_zipmap_pass(
    tmp_path,
):
    """The converter's default output is a map, which the graph pass turns back into scores.

    onnxruntime is the oracle for the pairing itself: which label names which column is the
    one thing a run of the graph with `ZipMap` already removed cannot show.
    """
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=4)
    labels = np.digitize(data[:, 1], [-0.5, 0.5]).astype(np.int64)
    model = _converted(
        ensemble.RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0),
        data,
        labels,
    )

    assert "ZipMap" in [node.op_type for node in model.graph.node]
    result = compile_onnx(model, tmp_path)
    outputs = result.load().run({"X": data})
    predicted, rows = _session(model).run(None, {"X": data})
    (table,) = result.report["class_labels"]

    assert table["dtype"] == "int64"
    np.testing.assert_array_equal(outputs["output_label"], predicted)
    np.testing.assert_allclose(
        outputs[table["tensor"]],
        np.array(
            [[row[label] for label in table["values"]] for row in rows], np.float32
        ),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_c_compiler
def test_a_converted_pipeline_of_scaler_and_forest_matches_onnxruntime(tmp_path):
    """The ensemble downstream of a preprocessor, whose intermediates carry no declared type."""
    pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=5)

    model = _converted(
        pipeline.make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, random_state=0),
        ),
        data,
        _regression_target(data),
    )

    assert [node.op_type for node in model.graph.node] == [
        "Scaler",
        "TreeEnsembleRegressor",
    ]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_two_ensembles_of_one_shape_share_a_single_walker(tmp_path):
    """The tree walk is one shared kernel per element type, however many nodes run it."""
    model = _regressor()
    second = onnx.NodeProto()
    second.CopyFrom(model.graph.node[0])
    second.name = "second"
    second.input[0] = "Y"
    second.output[0] = "Y2"
    model.graph.node.append(second)
    del model.graph.output[:]
    model.graph.output.append(helper.make_empty_tensor_value_info("Y2"))
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1

    result = compile_onnx(model, tmp_path)

    assert [name for name in result.report["kernels"] if "tree" in name] == [
        "trees_tree_aggregate_float_float"
    ]


# --------------------------------------------------------------------------------------
# What the compiler refuses
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    ["base_values_as_tensor", "nodes_values_as_tensor", "target_weights_as_tensor"],
)
def test_a_double_precision_table_is_rejected(tmp_path, family):
    """The `*_as_tensor` families, which ONNX's own reference implementation never reads."""
    table = helper.make_tensor(family, TensorProto.DOUBLE, [1], [0.5])

    with pytest.raises(CompileError, match="_as_tensor"):
        compile_onnx(_regressor(**{family: table}), tmp_path)


def test_a_string_labelled_classifier_is_rejected(tmp_path):
    """A label tensor of strings is a run-time string whatever the ensemble computes."""
    model = _classifier(classlabels_int64s=None, classlabels_strings=["a", "b"])

    with pytest.raises(CompileError, match="STRING"):
        compile_onnx(model, tmp_path)


@requires_c_compiler
def test_a_converted_classifier_over_string_classes_is_rejected(tmp_path):
    """Which is what a converter emits for a model fitted on string labels."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    data = _fitted_data(seed=6)
    labels = np.where(data[:, 0] > 0, "yes", "no")

    model = _converted(
        ensemble.RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0),
        data,
        labels,
        options={"zipmap": False},
    )

    with pytest.raises(CompileError, match="STRING"):
        compile_onnx(model, tmp_path)


def test_a_classifier_setting_both_label_families_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="classlabels_int64s"):
        compile_onnx(_classifier(classlabels_strings=["a", "b"]), tmp_path)


def test_a_classifier_declaring_no_classes_is_rejected(tmp_path):
    """Nothing then says what element type its labels would even be, let alone how many."""
    with pytest.raises(CompileError, match="no type"):
        compile_onnx(_classifier(classlabels_int64s=None), tmp_path)


def test_a_single_class_label_other_than_one_is_rejected(tmp_path):
    """The case ONNX's own reference implementation raises on, so nothing can vouch for it."""
    model = _classifier(classlabels_int64s=[7], class_ids=[0, 0])

    with pytest.raises(CompileError, match="single class"):
        compile_onnx(model, tmp_path)


def test_a_branch_test_onnx_does_not_define_is_rejected(tmp_path):
    model = _regressor(nodes_modes=["BRANCH_APPROX", "LEAF", "LEAF"])

    with pytest.raises(CompileError, match="BRANCH_APPROX"):
        compile_onnx(model, tmp_path)


def test_a_feature_outside_the_input_is_rejected(tmp_path):
    """A feature id the walker would read past the end of a row with."""
    model = _regressor(nodes_featureids=[7, 0, 0])

    with pytest.raises(CompileError, match="feature 7"):
        compile_onnx(model, tmp_path)


def test_a_target_outside_the_scored_targets_is_rejected(tmp_path):
    model = _regressor(target_ids=[0, 3])

    with pytest.raises(CompileError, match="target_ids"):
        compile_onnx(model, tmp_path)


def test_a_child_no_node_defines_is_rejected(tmp_path):
    model = _regressor(nodes_truenodeids=[9, 0, 0])

    with pytest.raises(CompileError, match="node 9 of tree 0"):
        compile_onnx(model, tmp_path)


def test_a_cycle_between_nodes_is_rejected(tmp_path):
    """Which the emitted walker would otherwise loop on forever."""
    model = _regressor(
        nodes_modes=["BRANCH_LEQ", "BRANCH_LEQ", "LEAF"],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 2, 0],
        target_treeids=[0],
        target_nodeids=[2],
        target_ids=[0],
        target_weights=[1.5],
    )

    with pytest.raises(CompileError, match="reachable more than once"):
        compile_onnx(model, tmp_path)


def test_families_that_disagree_on_their_length_are_rejected(tmp_path):
    model = _regressor(nodes_featureids=[0, 0])

    with pytest.raises(CompileError, match="nodes_featureids"):
        compile_onnx(model, tmp_path)


def test_a_base_value_per_nothing_is_rejected(tmp_path):
    model = _regressor(base_values=[1.0, 2.0])

    with pytest.raises(CompileError, match="base_values"):
        compile_onnx(model, tmp_path)


def test_a_transform_onnx_does_not_define_is_rejected(tmp_path):
    model = _regressor(post_transform="LOGIT")

    with pytest.raises(CompileError, match="post_transform"):
        compile_onnx(model, tmp_path)


def test_an_aggregation_onnx_does_not_define_is_rejected(tmp_path):
    model = _regressor(aggregate_function="MEDIAN")

    with pytest.raises(CompileError, match="aggregate_function"):
        compile_onnx(model, tmp_path)


def test_a_rank_3_input_is_rejected(tmp_path):
    """An ensemble reads a matrix of rows; the result is declared so that it is the rank
    the kernel objects to rather than a shape nothing could infer."""
    model = _regressor()
    model.graph.input[0].type.CopyFrom(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 2, 3])
    )
    model.graph.output[0].type.CopyFrom(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 2, 1])
    )

    with pytest.raises(CompileError, match=r"\[N, F\]"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# The opset-5 encoding, whose corpus tests cover the happy path
# --------------------------------------------------------------------------------------


def _ensemble(**attributes):
    declared = {
        "n_targets": 1,
        "tree_roots": [0],
        "nodes_featureids": [0],
        "nodes_truenodeids": [0],
        "nodes_falsenodeids": [1],
        "nodes_trueleafs": [1],
        "nodes_falseleafs": [1],
        "nodes_modes": helper.make_tensor("nodes_modes", TensorProto.UINT8, [1], [0]),
        "nodes_splits": helper.make_tensor(
            "nodes_splits", TensorProto.FLOAT, [1], [0.5]
        ),
        "leaf_targetids": [0, 0],
        "leaf_weights": helper.make_tensor(
            "leaf_weights", TensorProto.FLOAT, [2], [1.5, -2.5]
        ),
        **attributes,
    }
    return _model(
        [
            helper.make_node(
                "TreeEnsemble",
                ["X"],
                ["Y"],
                name="ensemble",
                domain="ai.onnx.ml",
                **declared,
            )
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])],
        [helper.make_empty_tensor_value_info("Y")],
    )


@requires_c_compiler
def test_the_opset_5_encoding_runs_the_stump_it_describes(tmp_path):
    compiled = compile_onnx(_ensemble(), tmp_path).load()

    scores = compiled.run({"X": np.array([[0.0, 0, 0], [1, 0, 0]] * 2, np.float32)})[
        "Y"
    ]

    np.testing.assert_array_equal(scores, np.array([[1.5], [-2.5]] * 2, np.float32))


def test_a_set_test_without_members_is_rejected(tmp_path):
    model = _ensemble(
        nodes_modes=helper.make_tensor("nodes_modes", TensorProto.UINT8, [1], [6])
    )

    with pytest.raises(CompileError, match="membership_values"):
        compile_onnx(model, tmp_path)


def test_a_membership_list_that_names_too_few_sets_is_rejected(tmp_path):
    model = _ensemble(
        nodes_modes=helper.make_tensor("nodes_modes", TensorProto.UINT8, [1], [6]),
        membership_values=helper.make_tensor(
            "membership_values", TensorProto.FLOAT, [2], [1.0, 2.0]
        ),
    )

    with pytest.raises(CompileError, match="NaN-terminated"):
        compile_onnx(model, tmp_path)


def test_a_branch_number_onnx_does_not_define_is_rejected(tmp_path):
    model = _ensemble(
        nodes_modes=helper.make_tensor("nodes_modes", TensorProto.UINT8, [1], [9])
    )

    with pytest.raises(CompileError, match="nodes_modes"):
        compile_onnx(model, tmp_path)


def test_a_root_outside_the_nodes_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="tree_roots"):
        compile_onnx(_ensemble(tree_roots=[3]), tmp_path)


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"nodes_falsenodeids": [5], "nodes_falseleafs": [0]}, "node 5 as a child"),
        ({"nodes_truenodeids": [9]}, "leaf 9 as a child"),
    ],
    ids=["node", "leaf"],
)
def test_a_child_outside_the_family_it_addresses_is_rejected(
    tmp_path, attributes, message
):
    """An interior child is reached by the set-member traversal before the walker's tables
    are built, so it is that traversal which has to refuse the ones no node defines."""
    with pytest.raises(CompileError, match=message):
        compile_onnx(_ensemble(**attributes), tmp_path)


def test_a_cycle_between_opset_5_nodes_is_rejected(tmp_path):
    """The traversal that lays out the set members walks these too, and stops here."""
    model = _ensemble(
        nodes_featureids=[0, 0],
        nodes_truenodeids=[1, 0],
        nodes_falsenodeids=[0, 1],
        nodes_trueleafs=[0, 0],
        nodes_falseleafs=[1, 1],
        nodes_modes=helper.make_tensor("nodes_modes", TensorProto.UINT8, [2], [0, 0]),
        nodes_splits=helper.make_tensor(
            "nodes_splits", TensorProto.FLOAT, [2], [0.5, 0.25]
        ),
    )

    with pytest.raises(CompileError, match="reachable more than once"):
        compile_onnx(model, tmp_path)


def test_a_missing_required_table_is_rejected(tmp_path):
    model = _ensemble()
    (node,) = model.graph.node
    del node.attribute[
        next(
            index
            for index, entry in enumerate(node.attribute)
            if entry.name == "leaf_weights"
        )
    ]

    with pytest.raises(CompileError, match="leaf_weights"):
        compile_onnx(model, tmp_path)
