"""The ONNX-ML support vector machines and linear models: converted models, and refusals.

What each op computes on a single node is settled by the differential sweep against the
reference evaluator; the corpus carries no node test for any of the four. Neither reaches
where these ops actually turn up — a scikit-learn converter emits `ai.onnx.ml` opset 1 and
wraps a multi-class `SVC` in a graph of a further thirty nodes — so the parity tests below run
those converted models against onnxruntime, the second oracle, independent of both the
compiler and the reference evaluator. Neither reaches the float edges either: all four sweep
finite operands, since a dot product's summation order is the reference's and not the
kernel's, and an op with no node test has no corpus to cover the edges in its place — so the
edge cases here feed them directly. The rest of the module covers the error contracts, which
no sweep asserts.
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
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402

ML_OPSET = 5

requires_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)

_SAMPLES = 32
_FEATURES = 4

# Three support vectors over three features, and the gamma/coef0/degree triple the kernel
# functions read.
_SUPPORT_VECTORS = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, -1.0, 0.5, 2.0]
_KERNEL_PARAMS = [0.5, 1.0, 3.0]


def _model(nodes, inputs, outputs, ml=ML_OPSET):
    """A model whose intermediates carry no `value_info`, as a converter's output does."""
    model = helper.make_model(
        helper.make_graph(nodes, "predictors", list(inputs), list(outputs)),
        opset_imports=[helper.make_opsetid("ai.onnx.ml", ml)],
    )
    model.ir_version = 10
    return model


def _node(op_type, outputs, attributes, results=None):
    """A single-node model over a `[4, 3]` input, with `attributes` overriding the defaults."""
    return _model(
        [
            helper.make_node(
                op_type,
                ["X"],
                list(outputs),
                name="predictor",
                domain="ai.onnx.ml",
                **attributes,
            )
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])],
        results or [helper.make_empty_tensor_value_info(name) for name in outputs],
    )


def _linear_regressor(**attributes):
    declared = {"coefficients": [1.0, 0.0, -1.0], "intercepts": [0.5], **attributes}
    return _node("LinearRegressor", ["Y"], declared)


def _linear_classifier(results=None, **attributes):
    declared = {
        "coefficients": [1.0, 0.0, -1.0, -0.5, 0.5, 0.25],
        "intercepts": [0.5, -0.25],
        "classlabels_ints": [3, 7],
        **attributes,
    }
    return _node("LinearClassifier", ["Y", "Z"], declared, results)


def _svm_regressor(**attributes):
    declared = {"coefficients": [1.0, 0.0, -1.0], "rho": [0.25], **attributes}
    return _node("SVMRegressor", ["Y"], declared)


def _dropping(model, *names):
    """`model` with those attributes taken off its node, which is the only way to leave an
    empty one: ONNX's own builder refuses to infer a list attribute's type from no values."""
    node = model.graph.node[0]
    kept = [entry for entry in node.attribute if entry.name not in names]
    del node.attribute[:]
    node.attribute.extend(kept)
    return model


def _svm_classifier(**attributes):
    declared = {
        "coefficients": [0.5, -0.25, 0.75],
        "rho": [0.25],
        "classlabels_ints": [3, 7],
        "vectors_per_class": [2, 1],
        "support_vectors": _SUPPORT_VECTORS,
        "kernel_type": "RBF",
        "kernel_params": _KERNEL_PARAMS,
        **attributes,
    }
    return _node("SVMClassifier", ["Y", "Z"], declared)


# --------------------------------------------------------------------------------------
# Converted scikit-learn models, against onnxruntime
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
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_a_converted_support_vector_regressor_matches_onnxruntime(tmp_path, kernel):
    """Every kernel function ONNX defines, as scikit-learn's own converter encodes it."""
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data()

    model = _converted(svm.SVR(kernel=kernel, degree=3), data, _regression_target(data))

    assert "SVMRegressor" in [node.op_type for node in model.graph.node]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_one_class_support_vector_machine_matches_onnxruntime(tmp_path):
    """The novelty detector, whose converter reads the score's sign downstream of the op."""
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data(seed=1)

    model = _converted(svm.OneClassSVM(), data, None)

    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
@pytest.mark.parametrize(
    ("module", "name", "arguments"),
    [
        ("sklearn.linear_model", "LinearRegression", {}),
        ("sklearn.linear_model", "Ridge", {"alpha": 0.5}),
        ("sklearn.svm", "LinearSVR", {"max_iter": 2000}),
    ],
)
def test_a_converted_linear_regressor_matches_onnxruntime(
    tmp_path, module, name, arguments
):
    estimators = pytest.importorskip(module)
    data = _fitted_data(seed=2)

    model = _converted(
        getattr(estimators, name)(**arguments), data, _regression_target(data)
    )

    assert [node.op_type for node in model.graph.node] == ["LinearRegressor"]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_multi_target_linear_regressor_matches_onnxruntime(tmp_path):
    """Several targets at once, which is what the coefficient matrix's rows are."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    data = _fitted_data(seed=3)
    targets = np.stack(
        [_regression_target(data), data[:, 2].astype(np.float64)], axis=1
    )

    model = _converted(linear_model.Ridge(alpha=0.5), data, targets)

    (node,) = model.graph.node
    assert {entry.name for entry in node.attribute} >= {"targets"}
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
@pytest.mark.parametrize("classes", [2, 3], ids=["binary", "multiclass"])
def test_a_converted_logistic_regression_matches_onnxruntime(tmp_path, classes):
    """`LinearClassifier` with the two transforms scikit-learn's converter emits."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    data = _fitted_data(seed=4)
    labels = np.digitize(data[:, 0], np.linspace(-1, 1, classes - 1)).astype(np.int64)

    model = _converted(
        linear_model.LogisticRegression(max_iter=500),
        data,
        labels,
        options={"zipmap": False},
    )

    (node,) = [
        entry for entry in model.graph.node if entry.op_type == "LinearClassifier"
    ]
    (transform,) = [
        entry.s for entry in node.attribute if entry.name == "post_transform"
    ]
    assert transform in (b"LOGISTIC", b"SOFTMAX")
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
@pytest.mark.parametrize("classes", [2, 3], ids=["binary", "multiclass"])
def test_a_converted_support_vector_classifier_matches_onnxruntime(tmp_path, classes):
    """The pairwise scheme, and the graph the converter wraps its votes in above two classes."""
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data(seed=5)
    labels = np.digitize(data[:, 0], np.linspace(-1, 1, classes - 1)).astype(np.int64)

    model = _converted(
        svm.SVC(kernel="rbf", random_state=0),
        data,
        labels,
        options={"zipmap": False},
    )

    (node,) = [entry for entry in model.graph.node if entry.op_type == "SVMClassifier"]
    assert {entry.name for entry in node.attribute} >= {"vectors_per_class"}
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_probability_classifier_matches_onnxruntime(tmp_path):
    """Platt scaling, which is the one thing `prob_a` and `prob_b` are there for."""
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data(seed=6)
    labels = (data[:, 0] + data[:, 1] > 0).astype(np.int64)

    model = _converted(
        svm.SVC(kernel="rbf", probability=True, random_state=0),
        data,
        labels,
        options={"zipmap": False},
    )

    (node,) = [entry for entry in model.graph.node if entry.op_type == "SVMClassifier"]
    assert {entry.name for entry in node.attribute} >= {"prob_a", "prob_b"}
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_pipeline_of_scaler_and_classifier_matches_onnxruntime(tmp_path):
    """The classifier downstream of a preprocessor, whose intermediates carry no type."""
    pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data(seed=7)
    labels = (data[:, 1] > 0).astype(np.int64)

    model = _converted(
        pipeline.make_pipeline(
            preprocessing.StandardScaler(), svm.SVC(kernel="rbf", random_state=0)
        ),
        data,
        labels,
        options={"zipmap": False},
    )

    assert [node.op_type for node in model.graph.node][:2] == [
        "Scaler",
        "SVMClassifier",
    ]
    _matches_onnxruntime(model, {"X": data}, tmp_path)


@requires_c_compiler
def test_a_converted_classifier_keeps_its_probabilities_through_the_zipmap_pass(
    tmp_path,
):
    """The converter's default output is a map, which the graph pass turns back into scores.

    onnxruntime is the oracle for the pairing itself: which label names which column is the
    one thing a run of the graph with `ZipMap` already removed cannot show.
    """
    svm = pytest.importorskip("sklearn.svm")
    data = _fitted_data(seed=8)
    labels = np.digitize(data[:, 0], [-0.5, 0.5]).astype(np.int64)
    model = _converted(svm.SVC(kernel="rbf", random_state=0), data, labels)

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
def test_two_linear_models_of_one_shape_share_a_single_scoring_kernel(tmp_path):
    """The dot product is one shared kernel per element type, however many nodes run it."""
    model = _linear_regressor(coefficients=[2.0])
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
    second = onnx.NodeProto()
    second.CopyFrom(model.graph.node[0])
    second.name = "second"
    second.input[0] = "Y"
    second.output[0] = "Y2"
    model.graph.node.append(second)
    del model.graph.output[:]
    model.graph.output.append(helper.make_empty_tensor_value_info("Y2"))

    result = compile_onnx(model, tmp_path)

    assert [name for name in result.report["kernels"] if "scores" in name] == [
        "predictors_ml_scores_float"
    ]


# --------------------------------------------------------------------------------------
# The float edges, against the reference evaluator
# --------------------------------------------------------------------------------------

_INFO = np.finfo(np.float32)
_EDGE_VALUES = (
    0.0,
    -0.0,
    np.inf,
    -np.inf,
    np.nan,
    _INFO.max,
    -_INFO.max,
    _INFO.tiny,
    _INFO.smallest_subnormal,
)


def _edge_rows(features=3):
    """One special value per row, in a column that rotates, and zero everywhere else.

    A row holding a single value is what makes its score independent of the order the
    products of that row are summed in — the one thing about these ops the reference cannot
    be held to, and the reason their sweeps run on finite operands. Everything else about an
    edge value is the arithmetic itself: what an infinity does to a kernel function, what a
    value that is not a number does to the winning column, and which side of a threshold an
    overflow lands on.
    """
    values = np.array(_EDGE_VALUES, np.float32)
    rows = np.zeros((len(values), features), np.float32)
    rows[np.arange(len(values)), np.arange(len(values)) % features] = values
    return rows


def _with_rows(model, rows):
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = rows
    return model


def _svm_classifier_over_classes():
    """The support vector classifier's other mode: one score per class, no support vectors."""
    return _dropping(
        _svm_classifier(coefficients=[1.0, 0.0, -1.0, -0.5, 0.5, 0.25]),
        "vectors_per_class",
        "support_vectors",
    )


@requires_c_compiler
@pytest.mark.parametrize(
    ("label", "builder"),
    [
        ("linear_regressor", _linear_regressor),
        ("linear_classifier", _linear_classifier),
        (
            "linear_classifier_paired",
            lambda: _linear_classifier(
                coefficients=[1.0, 0.0, -1.0],
                intercepts=[0.5],
                post_transform="LOGISTIC",
            ),
        ),
        ("svm_regressor", _svm_regressor),
        *(
            (
                f"svm_regressor_{kernel.lower()}",
                lambda kernel=kernel: _svm_regressor(
                    n_supports=3,
                    support_vectors=_SUPPORT_VECTORS,
                    kernel_params=_KERNEL_PARAMS,
                    kernel_type=kernel,
                ),
            )
            for kernel in ("LINEAR", "POLY", "RBF", "SIGMOID")
        ),
        ("svm_classifier", _svm_classifier),
        ("svm_classifier_linear", _svm_classifier_over_classes),
        (
            "svm_classifier_probabilities",
            lambda: _svm_classifier(prob_a=[-1.5], prob_b=[0.25]),
        ),
    ],
)
def test_a_predictor_matches_the_reference_on_the_float_edges(tmp_path, label, builder):
    data = _edge_rows()
    model = _with_rows(builder(), len(data))

    outputs = compile_onnx(model, tmp_path).load().run({"X": data})
    with np.errstate(all="ignore"):
        expected = ReferenceEvaluator(model).run(None, {"X": data})

    for entry, want in zip(model.graph.output, expected):
        got, want = outputs[entry.name], np.asarray(want)
        assert got.dtype == want.dtype, entry.name
        assert got.shape == want.shape, entry.name
        if want.dtype.kind == "f":
            np.testing.assert_allclose(
                got, want, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=entry.name
            )
        else:
            np.testing.assert_array_equal(got, want, err_msg=entry.name)


# --------------------------------------------------------------------------------------
# What the compiler refuses
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "attributes"),
    [
        (_linear_regressor, {}),
        (_linear_classifier, {}),
        (_svm_regressor, {}),
        (_svm_classifier, {}),
    ],
    ids=["linear_regressor", "linear_classifier", "svm_regressor", "svm_classifier"],
)
@pytest.mark.parametrize("shape", [[3], [2, 2, 3]], ids=["vector", "rank_3"])
def test_an_input_that_is_not_a_matrix_is_rejected(
    tmp_path, builder, attributes, shape
):
    """These ops read `[N, F]`; every other rank is a model the reference cannot run either."""
    model = builder(**attributes)
    del model.graph.input[0].type.tensor_type.shape.dim[:]
    for extent in shape:
        model.graph.input[0].type.tensor_type.shape.dim.add().dim_value = extent

    with pytest.raises(CompileError):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    "builder", [_linear_regressor, _linear_classifier], ids=["regressor", "classifier"]
)
def test_a_linear_model_without_intercepts_is_rejected(tmp_path, builder):
    """The one attribute the reference and onnxruntime read differently when it is absent."""
    with pytest.raises(CompileError, match="sets no `intercepts`"):
        compile_onnx(_dropping(builder(), "intercepts"), tmp_path)


def test_a_linear_model_with_an_intercept_per_nothing_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="intercept"):
        compile_onnx(
            _linear_regressor(
                coefficients=[1.0, 0.0, -1.0, 0.5, 0.5, 0.5],
                intercepts=[0.5, 0.25, 1.0],
                targets=2,
            ),
            tmp_path,
        )


def test_a_coefficient_per_nothing_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="coefficient"):
        compile_onnx(_linear_regressor(coefficients=[1.0, 0.0]), tmp_path)


def test_a_linear_regressor_scoring_no_targets_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="target"):
        compile_onnx(_linear_regressor(targets=0), tmp_path)


def test_a_linear_classifier_with_fewer_labels_than_columns_is_rejected(tmp_path):
    """A declared result wide enough to hide the disagreement is still refused."""
    model = _linear_classifier(
        results=[
            helper.make_tensor_value_info("Y", TensorProto.INT64, [4]),
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4, 3]),
        ],
        coefficients=[1.0] * 9,
        intercepts=[0.5, 0.0, -0.5],
    )

    with pytest.raises(CompileError, match="class label"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    "builder", [_linear_classifier, _svm_classifier], ids=["linear", "svm"]
)
def test_a_string_labelled_classifier_is_rejected(tmp_path, builder):
    model = _dropping(builder(classlabels_strings=["a", "b"]), "classlabels_ints")

    with pytest.raises(CompileError, match="STRING"):
        compile_onnx(model, tmp_path)


def test_a_support_vector_classifier_without_class_labels_is_rejected(tmp_path):
    """A declared result, since with no labels at all nothing derives one to begin with."""
    model = _dropping(_svm_classifier(), "classlabels_ints")
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("Y", TensorProto.INT64, [4])
    )
    model.graph.output[1].CopyFrom(
        helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4, 2])
    )

    with pytest.raises(CompileError, match="classlabels_ints"):
        compile_onnx(model, tmp_path)


def test_a_support_vector_machine_without_rho_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="`rho`"):
        compile_onnx(_dropping(_svm_regressor(), "rho"), tmp_path)


def test_a_kernel_onnx_does_not_define_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="kernel_type"):
        compile_onnx(_svm_classifier(kernel_type="COSINE"), tmp_path)


def test_a_transform_onnx_does_not_define_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="post_transform"):
        compile_onnx(_linear_regressor(post_transform="SIGMOID"), tmp_path)


def test_kernel_parameters_that_describe_less_than_a_kernel_are_rejected(tmp_path):
    with pytest.raises(CompileError, match="kernel_params"):
        compile_onnx(_svm_classifier(kernel_params=[0.5]), tmp_path)


def test_support_vectors_that_do_not_fill_the_matrix_are_rejected(tmp_path):
    with pytest.raises(CompileError, match="support_vectors"):
        compile_onnx(
            _svm_regressor(
                n_supports=3,
                support_vectors=_SUPPORT_VECTORS[:-1],
                kernel_params=_KERNEL_PARAMS,
            ),
            tmp_path,
        )


def test_a_support_vector_regressor_with_a_coefficient_per_nothing_is_rejected(
    tmp_path,
):
    with pytest.raises(CompileError, match="coefficient"):
        compile_onnx(_svm_regressor(coefficients=[1.0, 0.5]), tmp_path)


def test_a_single_class_over_support_vectors_is_rejected(tmp_path):
    """The pairwise scheme has no pairs to score, which its own reference refuses outright."""
    with pytest.raises(CompileError, match="at least two"):
        compile_onnx(
            _svm_classifier(classlabels_ints=[7], vectors_per_class=[3]), tmp_path
        )


def test_fewer_vector_counts_than_classes_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="vectors_per_class"):
        compile_onnx(_svm_classifier(vectors_per_class=[3]), tmp_path)


@pytest.mark.parametrize("counts", [[-1, 4], [-4, 1]], ids=["scored", "unscored"])
def test_a_negative_count_of_support_vectors_is_rejected(tmp_path, counts):
    """Each count is a length the pairwise loops run to, and a negative one walks off the
    tables they read; the reference scores such a pair as zero, which nothing can vouch for."""
    with pytest.raises(CompileError, match="negative count"):
        compile_onnx(_svm_classifier(vectors_per_class=counts), tmp_path)


def test_too_few_coefficient_rows_for_the_class_pairs_is_rejected(tmp_path):
    model = _svm_classifier(
        classlabels_ints=[1, 2, 3],
        vectors_per_class=[1, 1, 1],
        rho=[0.25, 0.1, -0.2],
    )

    with pytest.raises(CompileError, match="row"):
        compile_onnx(model, tmp_path)


def test_fewer_rho_than_class_pairs_is_rejected(tmp_path):
    model = _svm_classifier(
        classlabels_ints=[1, 2, 3],
        vectors_per_class=[1, 1, 1],
        coefficients=[0.5, -0.25, 0.75, 0.1, 0.2, -0.3],
    )

    with pytest.raises(CompileError, match="`rho` holds"):
        compile_onnx(model, tmp_path)


def test_probabilities_over_more_than_two_classes_are_rejected(tmp_path):
    """The one attribute combination whose two oracles disagree."""
    model = _svm_classifier(
        classlabels_ints=[1, 2, 3],
        vectors_per_class=[1, 1, 1],
        coefficients=[0.5, -0.25, 0.75, 0.1, 0.2, -0.3],
        rho=[0.25, 0.1, -0.2],
        prob_a=[-1.5, 0.5, 1.0],
        prob_b=[0.25, 0.0, -0.5],
    )

    with pytest.raises(CompileError, match="two classes only"):
        compile_onnx(model, tmp_path)


def test_a_probability_without_its_pair_is_rejected(tmp_path):
    with pytest.raises(CompileError, match="prob_b"):
        compile_onnx(_svm_classifier(prob_a=[-1.5]), tmp_path)
