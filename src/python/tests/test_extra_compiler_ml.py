"""The `ai.onnx.ml` preprocessing surface: the ZipMap pass, its metadata, and the pipelines.

What each op computes is settled by the conformance and differential suites, against ONNX's
own corpus and reference evaluator. The corpus is thin here — it carries a node test for two
of the nine ONNX-ML ops this compiler serves — so what this module adds is the coverage a
single-node sweep cannot reach: whole converted pipelines, whose expected values still come
from the reference evaluator; the graph pass and header metadata that have no op of their
own; and the errors for the models the compiler will not compile at all.
"""

from __future__ import annotations

import ctypes
import shutil

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
# The harness refuses to import without numpy, so this covers both dependencies.
harness = pytest.importorskip("fnnx.extras.compilers.c.harness")

from onnx import TensorProto, ValueInfoProto, helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from fnnx.extras.compilers.c import compile_onnx  # noqa: E402

ML_OPSET = 5
STANDARD_OPSET = 21

requires_c_compiler = pytest.mark.skipif(
    not any(shutil.which(name) for name in harness.COMPILER_CANDIDATES),
    reason="no system C compiler available",
)


def _tensor(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _ml_node(op_type, inputs, outputs, **attributes):
    return helper.make_node(
        op_type,
        list(inputs),
        list(outputs),
        name=outputs[0],
        domain="ai.onnx.ml",
        **attributes,
    )


def _model(nodes, inputs, outputs):
    """A model whose intermediates carry no `value_info`, as a converter's output does."""
    graph = helper.make_graph(nodes, "ml", list(inputs), list(outputs))
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", STANDARD_OPSET),
            helper.make_opsetid("ai.onnx.ml", ML_OPSET),
        ],
    )
    model.ir_version = 9
    return model


def _untyped(name):
    return helper.make_empty_tensor_value_info(name)


def _map_output(name):
    """A `ZipMap` result: the sequence of maps its schema declares, not a tensor."""
    entry = ValueInfoProto()
    entry.name = name
    entry.type.CopyFrom(
        helper.make_sequence_type_proto(
            helper.make_map_type_proto(
                TensorProto.STRING,
                helper.make_tensor_type_proto(TensorProto.FLOAT, []),
            )
        )
    )
    return entry


def _matches_reference(model, feeds, tmp_path):
    """Every output of the compiled artifact against the reference evaluator's own."""
    compiled = compile_onnx(model, tmp_path).load()
    outputs = compiled.run(feeds)
    expected = ReferenceEvaluator(model).run(None, feeds)
    assert [spec.name for spec in compiled.outputs] == [
        entry.name for entry in model.graph.output
    ]
    for entry, want in zip(model.graph.output, expected):
        got = outputs[entry.name]
        assert got.dtype == want.dtype, entry.name
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, err_msg=entry.name)


# --------------------------------------------------------------------------------------
# Pipelines, which is what an ONNX-ML model is made of
# --------------------------------------------------------------------------------------


@requires_c_compiler
def test_a_chain_of_preprocessors_compiles_and_matches_the_reference(tmp_path):
    """The shape and type of every intermediate come from the compiler's own inference.

    ONNX ships no inference function for `Imputer`, `Scaler` or `Normalizer`, and a converted
    model states nothing about its intermediates, so nothing but that inference gives these
    tensors a type at all.
    """
    model = _model(
        [
            _ml_node(
                "Imputer",
                ["x"],
                ["filled"],
                imputed_value_floats=[0.5, -1.0, 2.0],
                replaced_value_float=float("nan"),
            ),
            _ml_node(
                "Scaler",
                ["filled"],
                ["scaled"],
                offset=[1.0, 0.0, -1.0],
                scale=[0.5, 2.0, 1.0],
            ),
            _ml_node("Normalizer", ["scaled"], ["y"], norm="L2"),
        ],
        [_tensor("x", TensorProto.FLOAT, [4, 3])],
        [_untyped("y")],
    )
    feeds = {
        "x": np.array(
            [[1.0, np.nan, 3.0], [np.nan, 0.0, -2.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
    }

    _matches_reference(model, feeds, tmp_path)


@requires_c_compiler
def test_a_feature_union_of_encoded_and_scaled_columns_matches_the_reference(tmp_path):
    """The fan-out a converted `ColumnTransformer` produces: two branches, then a union."""
    model = _model(
        [
            _ml_node(
                "ArrayFeatureExtractor", ["x", "categorical"], ["category_column"]
            ),
            _ml_node(
                "OneHotEncoder",
                ["category_column"],
                ["encoded"],
                cats_int64s=[0, 1, 2],
                zeros=1,
            ),
            helper.make_node(
                "Reshape", ["encoded", "flat"], ["encoded_rows"], name="reshape"
            ),
            _ml_node("ArrayFeatureExtractor", ["x", "numeric"], ["numeric_columns"]),
            _ml_node(
                "Scaler", ["numeric_columns"], ["scaled"], offset=[1.0], scale=[0.25]
            ),
            _ml_node(
                "FeatureVectorizer",
                ["encoded_rows", "scaled"],
                ["y"],
                inputdimensions=[3, 2],
            ),
        ],
        [_tensor("x", TensorProto.FLOAT, [4, 3])],
        [_untyped("y")],
    )
    model.graph.initializer.extend(
        [
            helper.make_tensor("categorical", TensorProto.INT64, [1], [0]),
            helper.make_tensor("numeric", TensorProto.INT64, [2], [1, 2]),
            helper.make_tensor("flat", TensorProto.INT64, [2], [4, 3]),
        ]
    )
    feeds = {
        "x": np.array(
            [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0], [2.0, 5.0, 6.0], [9.0, 7.0, 8.0]],
            dtype=np.float32,
        )
    }

    _matches_reference(model, feeds, tmp_path)


@requires_c_compiler
def test_a_label_encoder_over_integer_classes_matches_the_reference(tmp_path):
    model = _model(
        [
            _ml_node(
                "LabelEncoder",
                ["x"],
                ["y"],
                keys_int64s=[0, 1, 2],
                values_floats=[0.25, -0.5, 1.0],
                default_float=-9.0,
            )
        ],
        [_tensor("x", TensorProto.INT64, [2, 4])],
        [_untyped("y")],
    )
    feeds = {"x": np.array([[0, 1, 2, 3], [-1, 2, 1, 0]], dtype=np.int64)}

    _matches_reference(model, feeds, tmp_path)


@requires_c_compiler
def test_repeated_preprocessors_share_one_kernel_and_one_table(tmp_path):
    """Two nodes of the same op and element type are one kernel; equal tables are one array."""
    model = _model(
        [
            _ml_node("Scaler", ["x"], ["a"], offset=[1.0, 2.0], scale=[0.5, 0.5]),
            _ml_node("Scaler", ["a"], ["b"], offset=[1.0, 2.0], scale=[0.5, 0.5]),
            _ml_node("Scaler", ["b"], ["y"], offset=[3.0, 4.0], scale=[0.5, 0.5]),
        ],
        [_tensor("x", TensorProto.FLOAT, [2, 2])],
        [_untyped("y")],
    )

    result = compile_onnx(model, tmp_path)
    header = result.header_path.read_text(encoding="utf-8")

    assert [name for name in result.report["kernels"] if "scaler" in name] == [
        "ml_scaler_float"
    ]
    # Two distinct offset tables and one shared scale table, each defined exactly once.
    tables = sorted(
        {
            line.split()[3].split("[")[0]
            for line in header.splitlines()
            if line.startswith("static const float ml_scaler_")
        }
    )
    assert len(tables) == 3
    assert result.report["memory"]["weights_bytes"] == 3 * 2 * 4


@requires_c_compiler
def test_an_out_of_range_index_returns_a_nonzero_status(tmp_path):
    """`ArrayFeatureExtractor` reads its columns from a run-time operand, so it checks them."""
    model = _model(
        [_ml_node("ArrayFeatureExtractor", ["x", "i"], ["y"])],
        [
            _tensor("x", TensorProto.FLOAT, [2, 3]),
            _tensor("i", TensorProto.INT64, [2]),
        ],
        [_untyped("y")],
    )
    compiled = compile_onnx(model, tmp_path).load()

    with pytest.raises(harness.HarnessError, match="status"):
        compiled.run(
            {
                "x": np.zeros((2, 3), np.float32),
                "i": np.array([0, 3], dtype=np.int64),
            }
        )


@requires_c_compiler
def test_a_value_in_no_category_returns_a_nonzero_status(tmp_path):
    """`zeros` cleared makes an unknown category the failure the schema prescribes."""
    model = _model(
        [_ml_node("OneHotEncoder", ["x"], ["y"], cats_int64s=[1, 2], zeros=0)],
        [_tensor("x", TensorProto.INT64, [3])],
        [_untyped("y")],
    )
    compiled = compile_onnx(model, tmp_path).load()

    assert compiled.run({"x": np.array([1, 2, 1], np.int64)})["y"].shape == (3, 2)
    with pytest.raises(harness.HarnessError, match="status"):
        compiled.run({"x": np.array([1, 2, 7], np.int64)})


# --------------------------------------------------------------------------------------
# The ZipMap pass and the class-label metadata it produces
# --------------------------------------------------------------------------------------

_LABELS = ["setosa", 'versi"colo\\r', "vir/*ginica"]


def _zipmap_model(labels_attribute, labels, *, extra_output=False):
    """A stand-in classifier: a probability tensor, keyed by `ZipMap` into a map output."""
    nodes = [
        _ml_node("Scaler", ["x"], ["scores"], offset=[1.0, 0.0, -1.0], scale=[0.5] * 3),
        _ml_node("Normalizer", ["scores"], ["probabilities"], norm="L1"),
        _ml_node(
            "ZipMap",
            ["probabilities"],
            ["output_probability"],
            **{labels_attribute: labels},
        ),
    ]
    outputs = [_map_output("output_probability")]
    if extra_output:
        nodes.insert(0, helper.make_node("Identity", ["x"], ["passthrough"], name="id"))
        outputs.append(_untyped("passthrough"))
    return _model(nodes, [_tensor("x", TensorProto.FLOAT, [2, 3])], outputs)


def test_a_trailing_zipmap_is_replaced_by_the_tensor_it_reads(tmp_path):
    result = compile_onnx(_zipmap_model("classlabels_strings", _LABELS), tmp_path)

    assert [entry["name"] for entry in result.report["entrypoint"]["outputs"]] == [
        "probabilities"
    ]
    (labels,) = result.report["class_labels"]
    assert labels["tensor"] == "probabilities"
    assert labels["dtype"] == "str"
    assert labels["values"] == _LABELS
    assert "ZipMap" not in result.header_path.read_text(encoding="utf-8")


def test_repeated_compiles_of_an_ml_model_are_byte_identical(tmp_path):
    """A table an ONNX-ML kernel embeds is named after its contents, not after its node."""
    model = _zipmap_model("classlabels_strings", _LABELS)

    first = compile_onnx(model, tmp_path / "first")
    second = compile_onnx(model, tmp_path / "second")

    assert first.header_path.read_bytes() == second.header_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()


def test_the_promoted_tensor_keeps_the_output_position_zipmap_held(tmp_path):
    result = compile_onnx(
        _zipmap_model("classlabels_strings", _LABELS, extra_output=True), tmp_path
    )

    assert [entry["name"] for entry in result.report["entrypoint"]["outputs"]] == [
        "probabilities",
        "passthrough",
    ]


def test_the_class_label_table_is_declared_with_a_count_macro(tmp_path):
    result = compile_onnx(_zipmap_model("classlabels_int64s", [7, 8, 9]), tmp_path)
    header = result.header_path.read_text(encoding="utf-8")
    (labels,) = result.report["class_labels"]

    assert labels["dtype"] == "int64"
    assert labels["values"] == [7, 8, 9]
    assert f"#define {labels['macro']} 3" in header
    assert f"extern const int64_t {labels['symbol']}[{labels['macro']}];" in header


def test_two_label_tables_on_one_output_get_their_own_count_macro(tmp_path):
    """Each macro is derived from its table's own symbol, not from the tensor they share.

    Two tables keying one tensor would otherwise define one macro name twice, with the two
    lengths — which is the header failing the `-Werror` build its own contract states.
    """
    model = _zipmap_model("classlabels_strings", _LABELS)
    model.graph.node.append(
        _ml_node(
            "ZipMap", ["probabilities"], ["output_label"], classlabels_int64s=[7, 8]
        )
    )
    model.graph.output.append(_map_output("output_label"))

    result = compile_onnx(model, tmp_path)
    defines = result.header_path.read_text(encoding="utf-8").splitlines()
    strings, integers = result.report["class_labels"]

    assert strings["tensor"] == integers["tensor"] == "probabilities"
    assert [
        line for line in defines if line.startswith(f"#define {strings['macro']} ")
    ] == [f"#define {strings['macro']} 3"]
    assert [
        line for line in defines if line.startswith(f"#define {integers['macro']} ")
    ] == [f"#define {integers['macro']} 2"]


@requires_c_compiler
def test_the_promoted_tensor_holds_the_probabilities_the_labels_name(tmp_path):
    """The pairing `ZipMap` stood for, read back off the tensor and the table replacing it.

    onnxruntime is the oracle here rather than the reference evaluator, which has no `ZipMap`
    implementation at all: which label names which column is the one thing a run of the graph
    with the node already removed cannot show.
    """
    runtime = pytest.importorskip("onnxruntime")
    model = _zipmap_model("classlabels_strings", _LABELS)
    feeds = {"x": np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)}

    result = compile_onnx(model, tmp_path)
    probabilities = result.load().run(feeds)["probabilities"]
    (rows,) = runtime.InferenceSession(model.SerializeToString()).run(None, feeds)
    (labels,) = result.report["class_labels"]

    expected = np.array(
        [[row[label] for label in labels["values"]] for row in rows], dtype=np.float32
    )
    assert probabilities.shape == expected.shape
    np.testing.assert_allclose(probabilities, expected, rtol=1e-6, atol=1e-7)


@requires_c_compiler
def test_the_emitted_string_table_holds_the_labels_byte_for_byte(tmp_path):
    """Read back through the built library, which is what proves the escaping is right."""
    result = compile_onnx(_zipmap_model("classlabels_strings", _LABELS), tmp_path)
    compiled = result.load()
    (labels,) = result.report["class_labels"]

    library = ctypes.CDLL(str(compiled.library_path))
    table = (ctypes.c_char_p * len(_LABELS)).in_dll(library, labels["symbol"])

    assert [entry.decode("utf-8") for entry in table] == _LABELS


@requires_c_compiler
def test_the_int64_table_holds_the_labels_the_model_declared(tmp_path):
    values = [-1, 0, 2**62]
    result = compile_onnx(_zipmap_model("classlabels_int64s", values), tmp_path)
    compiled = result.load()
    (labels,) = result.report["class_labels"]

    library = ctypes.CDLL(str(compiled.library_path))
    table = (ctypes.c_int64 * len(values)).in_dll(library, labels["symbol"])

    assert list(table) == values


def test_a_zipmap_whose_result_is_not_a_graph_output_is_rejected(tmp_path):
    model = _zipmap_model("classlabels_strings", _LABELS)
    del model.graph.output[:]
    model.graph.output.append(_untyped("probabilities"))

    with pytest.raises(CompileError, match="sequence of maps"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    "attributes",
    [{}, {"classlabels_strings": _LABELS, "classlabels_int64s": [1, 2, 3]}],
    ids=["neither", "both"],
)
def test_a_zipmap_without_exactly_one_label_list_is_rejected(tmp_path, attributes):
    model = _zipmap_model("classlabels_strings", _LABELS)
    (zipmap,) = [node for node in model.graph.node if node.op_type == "ZipMap"]
    del zipmap.attribute[:]
    for name, values in attributes.items():
        zipmap.attribute.append(helper.make_attribute(name, values))

    with pytest.raises(CompileError, match="classlabels_strings"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# What the compiler refuses
# --------------------------------------------------------------------------------------


def test_a_string_tensor_between_ml_ops_is_rejected(tmp_path):
    """`CategoryMapper` maps to or from strings whichever way it is pointed."""
    model = _model(
        [
            _ml_node(
                "CategoryMapper",
                ["x"],
                ["y"],
                cats_int64s=[1, 2],
                cats_strings=["a", "b"],
            )
        ],
        [_tensor("x", TensorProto.INT64, [3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="STRING"):
        compile_onnx(model, tmp_path)


def test_string_categories_against_a_numeric_input_are_rejected(tmp_path):
    model = _model(
        [_ml_node("OneHotEncoder", ["x"], ["y"], cats_strings=["a", "b"])],
        [_tensor("x", TensorProto.FLOAT, [3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="cats_int64s"):
        compile_onnx(model, tmp_path)


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"keys_strings": ["a", "b"], "values_int64s": [1, 2]}, "LabelEncoder"),
        ({"keys_int64s": [1, 2], "values_strings": ["a", "b"]}, "STRING"),
    ],
    ids=["string_keys", "string_values"],
)
def test_a_label_encoder_over_strings_is_rejected(tmp_path, attributes, message):
    """Either side of the mapping being strings puts it out of reach, for its own reason."""
    model = _model(
        [_ml_node("LabelEncoder", ["x"], ["y"], **attributes)],
        [_tensor("x", TensorProto.INT64, [3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match=message):
        compile_onnx(model, tmp_path)


def test_a_coefficient_list_that_fits_no_feature_axis_is_rejected(tmp_path):
    model = _model(
        [_ml_node("Scaler", ["x"], ["y"], offset=[1.0, 2.0], scale=[1.0, 2.0])],
        [_tensor("x", TensorProto.FLOAT, [4, 3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="2 `offset` value"):
        compile_onnx(model, tmp_path)


def test_a_norm_onnx_does_not_define_is_rejected(tmp_path):
    model = _model(
        [_ml_node("Normalizer", ["x"], ["y"], norm="L3")],
        [_tensor("x", TensorProto.FLOAT, [4, 3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="L1, L2, MAX"):
        compile_onnx(model, tmp_path)


def test_imputer_setting_both_value_families_is_rejected(tmp_path):
    model = _model(
        [
            _ml_node(
                "Imputer",
                ["x"],
                ["y"],
                imputed_value_floats=[1.0],
                imputed_value_int64s=[1],
            )
        ],
        [_tensor("x", TensorProto.FLOAT, [4, 3])],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="exactly one of"):
        compile_onnx(model, tmp_path)


def test_a_width_per_input_is_required(tmp_path):
    model = _model(
        [_ml_node("FeatureVectorizer", ["a", "b"], ["y"], inputdimensions=[2])],
        [
            _tensor("a", TensorProto.FLOAT, [4, 3]),
            _tensor("b", TensorProto.FLOAT, [4, 2]),
        ],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="one width per input"):
        compile_onnx(model, tmp_path)


def test_inputs_that_disagree_on_their_row_count_are_rejected(tmp_path):
    model = _model(
        [_ml_node("FeatureVectorizer", ["a", "b"], ["y"], inputdimensions=[3, 2])],
        [
            _tensor("a", TensorProto.FLOAT, [4, 3]),
            _tensor("b", TensorProto.FLOAT, [2, 2]),
        ],
        [_untyped("y")],
    )

    with pytest.raises(CompileError, match="first dimension"):
        compile_onnx(model, tmp_path)


# --------------------------------------------------------------------------------------
# Converted scikit-learn pipelines, where the backend corpus has nothing
# --------------------------------------------------------------------------------------

_SAMPLES = 24
_FEATURES = 4


def _converted(estimator, data):
    """The ONNX a scikit-learn converter emits for `estimator`, at `data`'s exact shape.

    The shape is stated rather than taken from a sample row: a converter leaves the batch
    dimension open otherwise, and a compiled artifact is specialized to one concrete shape.
    """
    skl2onnx = pytest.importorskip("skl2onnx")
    data_types = pytest.importorskip("skl2onnx.common.data_types")
    tensor_type = (
        data_types.Int64TensorType
        if data.dtype == np.int64
        else data_types.FloatTensorType
    )
    return skl2onnx.convert_sklearn(
        estimator.fit(data), initial_types=[("X", tensor_type(list(data.shape)))]
    )


def _fitted_data(seed=0):
    generator = np.random.default_rng(seed)
    return generator.normal(size=(_SAMPLES, _FEATURES)).astype(np.float32)


@requires_c_compiler
@pytest.mark.parametrize(
    "name",
    ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer"],
)
def test_a_converted_scaler_matches_the_reference(tmp_path, name):
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    data = _fitted_data()

    model = _converted(getattr(preprocessing, name)(), data)

    _matches_reference(model, {model.graph.input[0].name: data}, tmp_path)


@requires_c_compiler
def test_a_converted_imputer_matches_the_reference(tmp_path):
    impute = pytest.importorskip("sklearn.impute")
    data = _fitted_data()
    data[3, 1] = np.nan
    data[7, 2] = np.nan

    model = _converted(impute.SimpleImputer(strategy="mean"), data)

    _matches_reference(model, {model.graph.input[0].name: data}, tmp_path)


@requires_c_compiler
def test_a_converted_binarizer_matches_the_reference(tmp_path):
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    data = _fitted_data()

    model = _converted(preprocessing.Binarizer(threshold=0.25), data)

    _matches_reference(model, {model.graph.input[0].name: data}, tmp_path)


@requires_c_compiler
def test_a_converted_preprocessing_pipeline_matches_the_reference(tmp_path):
    pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    impute = pytest.importorskip("sklearn.impute")
    data = _fitted_data(seed=1)
    data[2, 0] = np.nan

    model = _converted(
        pipeline.make_pipeline(
            impute.SimpleImputer(strategy="median"),
            preprocessing.StandardScaler(),
            preprocessing.Normalizer(norm="l2"),
        ),
        data,
    )

    _matches_reference(model, {model.graph.input[0].name: data}, tmp_path)


@requires_c_compiler
def test_a_converted_one_hot_encoder_matches_the_reference(tmp_path):
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    generator = np.random.default_rng(2)
    data = generator.integers(0, 3, size=(_SAMPLES, 2)).astype(np.int64)

    model = _converted(preprocessing.OneHotEncoder(sparse_output=False), data)

    _matches_reference(model, {model.graph.input[0].name: data}, tmp_path)
