"""Symbolic-dimension binding, shape inference, and tensor type/shape lookup."""

from __future__ import annotations

import math
from collections.abc import Container, Iterator, Mapping

import onnx.defs
import onnx.shape_inference
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
    helper,
)

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.loader import (
    ML_DOMAIN,
    STANDARD_DOMAIN,
    normalize_domain,
)

UNBOUND_DIM_DEFAULT = 1

# Standard-domain ops whose ONNX schema states the output carries the input's shape and
# whose shape ONNX's own inference nevertheless does not derive; see
# `_propagate_preserved_shapes`.
SHAPE_PRESERVING_OPS = frozenset({"GroupNormalization"})

# `ai.onnx.ml` ops the installed `onnx` package ships no type-and-shape inference function
# for at all, plus the ones it has a function for that stop short of a case; see
# `_infer_ml_types`.
ML_UNINFERRED_OPS = frozenset(
    {
        "ArrayFeatureExtractor",
        "FeatureVectorizer",
        "Imputer",
        "LinearRegressor",
        "Normalizer",
        "SVMClassifier",
        "SVMRegressor",
        "Scaler",
        "TreeEnsembleClassifier",
        "TreeEnsembleRegressor",
    }
)

# The Reduce* family, whose `axes` operand ONNX defines as naming every axis when it is
# absent — the one reading of an empty operand `drop_empty_shape_operands` rests on.
REDUCTIONS = (
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
)

# Standard-domain ops with operands whose *values* decide the shape of their output, by the
# positions of those operands. ONNX passes them as tensors rather than as attributes, so a
# graph may compute one at run time; folding resolves every one a model fixes, and what is
# left makes the output shape a function of input data, which no binding can make static.
SHAPE_DEFINING_INPUTS: Mapping[str, tuple[int, ...]] = {
    # `size` is the result's shape, spatial axes and all, less its trailing coordinate axis.
    "AffineGrid": (1,),
    # A window is `size` samples long and nothing else; a window op is a compile-time value
    # rather than a kernel, so a `size` the graph does not fix leaves nothing to compile.
    "BlackmanWindow": (0,),
    # `shape` is the extent of every axis `axes` names, cropped or padded to reach it.
    "CenterCropPad": (1,),
    # `image_shape` is the result's spatial extent and `block_shape` decides how many
    # channels the columns hold.
    "Col2Im": (1, 2),
    "ConstantOfShape": (0,),
    # `dft_length` is the extent of the transformed axis, and `axis` says which axis takes
    # it — the second only where the transform resizes an axis; see `_shape_defining_inputs`.
    "DFT": (1, 2),
    "Expand": (1,),
    "HammingWindow": (0,),
    "HannWindow": (0,),
    "MaxUnpool": (2,),
    # `num_mel_bins` and `dft_length` are the two extents of the matrix.
    "MelWeightMatrix": (0, 1),
    "OneHot": (1,),
    # `pads` and `axes`: the two together say how much longer each axis of the result is.
    "Pad": (1, 3),
    "Range": (0, 1, 2),
    "Reshape": (1,),
    "Slice": (1, 2, 3, 4),
    "Split": (1,),
    "Squeeze": (1,),
    # `frame_step` decides how many frames the signal yields and `frame_length` how long
    # each transform is; the window is read for its values alone, so it may stay run-time.
    "STFT": (1, 3),
    "Tile": (1,),
    "TopK": (1,),
    "Unsqueeze": (1,),
    **{op_type: (1,) for op_type in REDUCTIONS},
}


def runtime_shape_operand(
    node: NodeProto,
    constants: Container[str],
    types: Mapping[str, TypeProto],
) -> str | None:
    """The first shape-deciding operand this node reads at run time, if it has any.

    An operand with no elements does not count: it names nothing whatever the data holds, so
    a reduction over an empty axes tensor is as static as one with no axes operand at all.
    """
    if normalize_domain(node.domain) != STANDARD_DOMAIN:
        return None
    indices = _shape_defining_inputs(node)
    if indices is None:
        return None
    for index in indices:
        name = node.input[index] if index < len(node.input) else ""
        if not name or name in constants:
            continue
        shape = static_shape(types.get(name))
        if shape is None or math.prod(shape) != 0:
            return name
    return None


def _shape_defining_inputs(node: NodeProto) -> tuple[int, ...] | None:
    """Which of the op's shape-deciding operands decide *this* node's result shape.

    DFT is the one op whose own attributes settle that: a one-sided transform returns half
    the length of the axis it lands on, and a stated `dft_length` replaces that extent
    outright, so with either of them which axis is transformed is part of the result's
    shape. With neither, the result carries the operand's own extents whichever axis the
    transform takes, and the axis is a value a kernel can switch on at run time.
    """
    indices = SHAPE_DEFINING_INPUTS.get(node.op_type)
    if indices is None or node.op_type != "DFT":
        return indices
    onesided = any(
        attribute.name == "onesided" and attribute.i for attribute in node.attribute
    )
    resized = onesided or (len(node.input) > 1 and bool(node.input[1]))
    return indices if resized else indices[:1]


def drop_empty_shape_operands(model: ModelProto) -> None:
    """Leave out a reduction's `axes` operand when it holds no elements.

    An empty axes tensor names no axes, which is what leaving the operand out means — ONNX
    defines the two the same way — but ONNX's shape inference does not reason about the
    values of an operand it cannot see, so it types the result of a reduction over an empty
    `axes` *input* as being of unknown rank. Dropping the operand keeps the meaning and lets
    inference derive the shape; the tensor stays a graph input, unread.

    The reductions alone: every other op reading an axis list defines an empty one as naming
    nothing rather than everything, so dropping it there would change what the node computes.
    """
    graph = model.graph
    types = tensor_types(graph)
    for node in graph.node:
        if node.op_type not in REDUCTIONS:
            continue
        if normalize_domain(node.domain) != STANDARD_DOMAIN:
            continue
        name = node.input[1] if len(node.input) > 1 else ""
        shape = static_shape(types.get(name)) if name else None
        if shape is not None and math.prod(shape) == 0:
            node.input[1] = ""


def state_stft_onesided(model: ModelProto) -> None:
    """Write STFT's own default for `onesided` into the nodes that leave it out.

    ONNX's schema declares the default as 1 — and the reference evaluator applies it, so a
    one-sided spectrum is what the op computes — while ONNX's shape inference falls back to
    0 and sizes the result at the whole frame length rather than at its non-redundant half.
    Stating the default the schema itself declares changes nothing about what the node
    computes and leaves inference deriving the shape the op actually produces.
    """
    for node in model.graph.node:
        if node.op_type != "STFT" or normalize_domain(node.domain) != STANDARD_DOMAIN:
            continue
        if any(attribute.name == "onesided" for attribute in node.attribute):
            continue
        declared = onnx.defs.get_schema("STFT").attributes["onesided"].default_value
        node.attribute.append(helper.make_attribute("onesided", declared.i))


def bind_dims(model: ModelProto, dim_bindings: Mapping[str, int]) -> dict[str, int]:
    """Give every graph input a concrete shape and return the bindings that were applied.

    Symbolic (and unnamed unknown) input dimensions take their value from `dim_bindings`,
    defaulting to `UNBOUND_DIM_DEFAULT`. Symbolic dimensions on graph outputs are dropped
    rather than bound, and stale intermediate shapes are discarded, so that every shape
    downstream follows from the computation instead of from a declaration the graph does
    not actually guarantee. An output the graph does not compute is the exception: it just
    aliases the input of that name, whose bound shape it therefore takes. Bindings naming a
    dimension the model does not use are ignored; dimension names are global across a
    bundle, and a node need not use all of them.
    """
    for name, size in dim_bindings.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CompileError(
                f"Dimension binding `{name}` must be a non-negative integer, got {size!r}."
            )

    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    applied: dict[str, int] = {}
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        for dim in _dims(value_info):
            _bind_dim(dim, dim_bindings, applied)

    bound_inputs = {value_info.name: value_info.type for value_info in graph.input}
    for value_info in graph.output:
        aliased = bound_inputs.get(value_info.name)
        if aliased is not None:
            value_info.type.CopyFrom(aliased)
            continue
        for dim in _dims(value_info):
            if dim.WhichOneof("value") == "dim_param":
                dim.ClearField("dim_param")
    del graph.value_info[:]
    return applied


def declared_output_types(graph: GraphProto) -> dict[str, TypeProto]:
    """Snapshot of the graph outputs' declared types, taken before `bind_dims` clears them."""
    declared = {}
    for value_info in graph.output:
        if value_info.HasField("type"):
            copied = TypeProto()
            copied.CopyFrom(value_info.type)
            declared[value_info.name] = copied
    return declared


def apply_declared_output_shapes(
    model: ModelProto,
    declared: Mapping[str, TypeProto],
    dim_bindings: Mapping[str, int],
) -> dict[str, int]:
    """Fall back to a graph output's declared shape where inference could not derive one.

    Only outputs produced outside the standard domain qualify: `ai.onnx.ml` inference does
    not propagate batch dimensions, so a classifier's declared output shape is the only
    static description of it there is. A standard-domain op that inference cannot resolve is
    one whose shape depends on runtime data, and its declaration is deliberately not
    trusted — compilation fails instead of emitting code for a shape the graph never
    guarantees. Returns the bindings the accepted declarations applied.
    """
    producers = {
        name: node for node in model.graph.node for name in node.output if name
    }
    applied: dict[str, int] = {}
    for value_info in model.graph.output:
        if static_shape(value_info.type) is not None:
            continue
        producer = producers.get(value_info.name)
        declared_type = declared.get(value_info.name)
        if producer is None or declared_type is None:
            continue
        if normalize_domain(producer.domain) == STANDARD_DOMAIN:
            continue
        candidate = TypeProto()
        candidate.CopyFrom(declared_type)
        bound: dict[str, int] = {}
        for dim in _shape_dims(candidate):
            _bind_dim(dim, dim_bindings, bound)
        if static_shape(candidate) is None:
            continue
        value_info.type.tensor_type.shape.CopyFrom(candidate.tensor_type.shape)
        applied.update(bound)
    return applied


def drop_shadowed_inputs(model: ModelProto) -> None:
    """Remove graph inputs that an initializer also defines.

    Pre-IR-4 models list every initializer as an input with the initializer as its default.
    The C compiler embeds initializers as static weights, so those entries are dropped and
    the remaining inputs are exactly the tensors a caller must provide. That is IR 4's rule,
    and shape inference applies the older one literally — ignoring initializers that are not
    inputs — so the model moves to the IR version whose semantics it is now compiled under.
    """
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    kept = [entry for entry in graph.input if entry.name not in initializer_names]
    if len(kept) == len(graph.input):
        return
    del graph.input[:]
    graph.input.extend(kept)
    model.ir_version = max(model.ir_version, 4)


def infer_shapes(model: ModelProto) -> ModelProto:
    """Infer every tensor's type and shape, strictly wherever ONNX can be strict.

    Strict mode turns an inference error into a compile error naming the node, rather than
    into a tensor the compiler discovers has no type much later. It also recurses into the
    bodies ONNX defines for function ops, and a few of those raise on a node the model is
    not answerable for — MeanVarianceNormalization's body builds its `axes` from a Constant
    that carries nothing at all unless the node sets the attribute. A relaxed retry is
    therefore accepted, but only when it leaves every tensor typed; anything less and the
    strict diagnostic is the one worth reporting.
    """
    try:
        inferred = _infer(model, strict=True)
    except CompileError as strict_error:
        try:
            inferred = _infer(model, strict=False)
        except CompileError:
            raise strict_error from None
        if _has_untyped_tensor(inferred.graph):
            raise strict_error from None
    return inferred


def _infer(model: ModelProto, *, strict: bool) -> ModelProto:
    """Run ONNX's inference, filling in the shapes it stops at until it stops adding any.

    Each round can only give a shape to a tensor that had none, so the loop shrinks a finite
    set and ends; a graph with nothing to fill in runs inference exactly once. Entries left
    shapeless by a previous run are discarded first: folding is what gives inference the
    values it was missing, and a stale entry would shadow the shape this run derives. The
    ONNX-ML results ONNX derives nothing for are filled in before the first round too, since
    strict inference stops at the first node whose operand it cannot type.
    """
    _drop_shapeless_value_info(model.graph)
    _infer_ml_types(model.graph)
    inferred = _run_inference(model, strict=strict)
    while _fill_underived_shapes(inferred.graph):
        _drop_shapeless_value_info(inferred.graph)
        inferred = _run_inference(inferred, strict=strict)
    return inferred


def _fill_underived_shapes(graph: GraphProto) -> bool:
    """Both passes over what ONNX's inference left untyped, neither short-circuiting."""
    preserved = _propagate_preserved_shapes(graph)
    return _infer_ml_types(graph) or preserved


def _drop_shapeless_value_info(graph: GraphProto) -> None:
    """Discard the intermediate entries inference could not give a shape.

    They state an element type and nothing more, and ONNX leaves them behind when it runs
    over a graph it has already seen — where a re-run derives the shape, the stale entry
    would shadow it, since `tensor_types` reads `value_info` after the graph's outputs.
    """
    kept = [entry for entry in graph.value_info if static_shape(entry.type) is not None]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)


def _run_inference(model: ModelProto, *, strict: bool) -> ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=strict, data_prop=True
        )
    except Exception as exc:
        raise CompileError(
            f"ONNX shape inference failed for graph `{graph_label(model.graph)}`: {exc}"
        ) from exc


def _propagate_preserved_shapes(graph: GraphProto) -> bool:
    """Give an output the shape its op's ONNX schema states it takes from its input.

    ONNX's own inference derives a shape for every op this compiler serves but one:
    GroupNormalization is defined as a function whose body reshapes through shapes it
    computes, and inference stops at the first of them, leaving a rank it never states —
    though the schema says in as many words that `Y` has the shape of `X`. Without this the
    op would be compilable only in a model that declares that shape itself, and every tensor
    downstream of one would lose its own.
    """
    filled = False
    types = tensor_types(graph)
    for node in graph.node:
        if node.op_type not in SHAPE_PRESERVING_OPS:
            continue
        if normalize_domain(node.domain) != STANDARD_DOMAIN:
            continue
        source = types.get(node.input[0]) if node.input else None
        produced = node.output[0] if node.output else ""
        if source is None or static_shape(source) is None or not produced:
            continue
        declared = types.get(produced)
        if declared is None or declared.WhichOneof("value") != "tensor_type":
            continue
        if static_shape(declared) is not None:
            continue
        # Every entry the tensor has, since inference leaves a graph output described in
        # both `output` and `value_info` and a stale one would shadow the other.
        for entry in (*graph.output, *graph.value_info):
            if entry.name == produced:
                entry.type.tensor_type.shape.CopyFrom(source.tensor_type.shape)
                filled = True
    return filled


def _infer_ml_types(graph: GraphProto) -> bool:
    """Type the `ai.onnx.ml` results ONNX's own inference derives nothing for.

    The installed `onnx` package registers no inference function for `Scaler`, `Normalizer`,
    `Imputer` or `FeatureVectorizer`; the one it registers for `ArrayFeatureExtractor` stops
    short of a rank-1 `X`; and the ones it registers for the two tree ensembles derive an
    element type but no shape at all. Every rule below is the op's own schema read literally —
    the element type its results are declared as, and the shape its documentation states — so
    a model whose intermediates carry no `value_info`, which is most of what ONNX-ML
    converters emit, still has a static type for every tensor. Anything derived wrongly here
    is a wrong result shape, which the differential sweep compares against the reference
    evaluator.
    """
    types = tensor_types(graph)
    filled = False
    for node in graph.node:
        if normalize_domain(node.domain) != ML_DOMAIN:
            continue
        if node.op_type not in ML_UNINFERRED_OPS or not node.output:
            continue
        derived = _ml_result_types(node, types)
        for produced, type_proto in zip(node.output, derived):
            if not produced or type_proto is None:
                continue
            if static_shape(types.get(produced)) is not None:
                continue
            _set_tensor_type(graph, produced, type_proto)
            filled = True
    return filled


def _ml_result_types(
    node: NodeProto, types: Mapping[str, TypeProto]
) -> tuple[TypeProto | None, ...]:
    """The types of `node`'s results, or nothing while an operand of its own has none."""
    operands = [types.get(name) for name in node.input]
    shapes = [static_shape(operand) for operand in operands]
    if not operands or any(shape is None for shape in shapes):
        return ()
    source, shape = operands[0], shapes[0]
    assert source is not None and shape is not None
    if node.op_type == "Imputer":
        return (
            helper.make_tensor_type_proto(source.tensor_type.elem_type, list(shape)),
        )
    if node.op_type in ("Normalizer", "Scaler"):
        return (helper.make_tensor_type_proto(TensorProto.FLOAT, list(shape)),)
    if node.op_type == "FeatureVectorizer":
        return (_feature_vectorizer_type(node, shape),)
    if node.op_type == "TreeEnsembleRegressor":
        return (_tree_ensemble_type(node, shape),)
    if node.op_type == "TreeEnsembleClassifier":
        return _tree_ensemble_classifier_types(node, shape)
    if node.op_type == "LinearRegressor":
        return (_linear_regressor_type(node, shape),)
    if node.op_type == "SVMRegressor":
        return (_svm_regressor_type(shape),)
    if node.op_type == "SVMClassifier":
        return _svm_classifier_types(node, shape)
    # ArrayFeatureExtractor, which takes as many columns as its index operand holds elements
    # — one for a scalar, which is the case ONNX's own inference leaves as an unknown
    # dimension. A vector `X` is the other case it stops short of, and its reference
    # implementation documents that one as following onnxruntime rather than the
    # specification: the result is the single row a matrix of one row would have.
    indices = shapes[1]
    if not shape or indices is None:
        return ()
    taken = math.prod(indices)
    return (
        helper.make_tensor_type_proto(
            source.tensor_type.elem_type,
            [1, taken] if len(shape) == 1 else [*shape[:-1], taken],
        ),
    )


def _tree_ensemble_rows(shape: tuple[int, ...]) -> int | None:
    """How many rows an ensemble scores: one per row of `X`, a vector being a single row."""
    if len(shape) == 2:
        return shape[0]
    return 1 if len(shape) == 1 else None


def _tree_ensemble_type(node: NodeProto, shape: tuple[int, ...]) -> TypeProto | None:
    """`TreeEnsembleRegressor`'s `Y`: one float score per target per row."""
    rows = _tree_ensemble_rows(shape)
    targets = _attribute(node, "n_targets")
    if rows is None or targets is None:
        return None
    return helper.make_tensor_type_proto(TensorProto.FLOAT, [rows, targets.i])


def _tree_ensemble_classifier_types(
    node: NodeProto, shape: tuple[int, ...]
) -> tuple[TypeProto | None, ...]:
    """`TreeEnsembleClassifier`'s label and score outputs.

    An ensemble whose leaves all weight one class scores a *pair* of columns whatever its one
    class label says, which is the binary rule its reference implementation applies; that is
    what decides the width of `Z` and cannot be read off the class labels alone.
    """
    rows = _tree_ensemble_rows(shape)
    integers = _attribute(node, "classlabels_int64s")
    strings = _attribute(node, "classlabels_strings")
    if rows is None or (integers is None and strings is None):
        return ()
    classes = max(
        len(integers.ints) if integers else 0, len(strings.strings) if strings else 0
    )
    identifiers = _attribute(node, "class_ids")
    binary = (
        len({int(value) for value in identifiers.ints}) == 1 if identifiers else False
    )
    return (
        helper.make_tensor_type_proto(
            TensorProto.INT64 if integers else TensorProto.STRING, [rows]
        ),
        helper.make_tensor_type_proto(
            TensorProto.FLOAT, [rows, 2 if binary and classes == 1 else classes]
        ),
    )


def _linear_regressor_type(node: NodeProto, shape: tuple[int, ...]) -> TypeProto | None:
    """`LinearRegressor`'s `Y`: one float score per target per row."""
    if len(shape) != 2:
        return None
    targets = _attribute(node, "targets")
    return helper.make_tensor_type_proto(
        TensorProto.FLOAT, [shape[0], targets.i if targets else 1]
    )


def _svm_regressor_type(shape: tuple[int, ...]) -> TypeProto | None:
    """`SVMRegressor`'s `Y`: the single score a row is given, as a column of its own."""
    if len(shape) != 2:
        return None
    return helper.make_tensor_type_proto(TensorProto.FLOAT, [shape[0], 1])


def _svm_classifier_types(
    node: NodeProto, shape: tuple[int, ...]
) -> tuple[TypeProto | None, ...]:
    """`SVMClassifier`'s label and score outputs.

    How wide a row of `Z` is depends on the whole shape of the node: an ensemble over support
    vectors scores one value per *pair* of classes unless it couples them into probabilities,
    a linear one scores one per class, and a lone score is paired with a second where a second
    class is called for. The kernel derives the same width from the same attributes and
    refuses to write a buffer that disagrees with it.
    """
    if len(shape) != 2:
        return ()
    integers = _attribute(node, "classlabels_ints")
    strings = _attribute(node, "classlabels_strings")
    if integers is None and strings is None:
        return ()
    classes = max(
        len(integers.ints) if integers else 0, len(strings.strings) if strings else 0
    )
    counts = _attribute(node, "vectors_per_class")
    vectors = sum(counts.ints) if counts else 0
    coupled = vectors > 0 and bool(_floats(node, "prob_a"))
    scored = (
        max(classes, 1) if vectors == 0 or coupled else classes * (classes - 1) // 2
    )
    transform = _attribute(node, "post_transform")
    paired = (
        scored == 1
        and classes == 2
        and len(_floats(node, "rho")) == 1
        and (transform is None or transform.s != b"PROBIT")
    )
    return (
        helper.make_tensor_type_proto(
            TensorProto.INT64 if integers else TensorProto.STRING, [shape[0]]
        ),
        helper.make_tensor_type_proto(
            TensorProto.FLOAT, [shape[0], 2 if paired else scored]
        ),
    )


def _floats(node: NodeProto, name: str) -> list[float]:
    attribute = _attribute(node, name)
    return list(attribute.floats) if attribute else []


def _attribute(node: NodeProto, name: str) -> AttributeProto | None:
    return next(
        (entry for entry in node.attribute if entry.name == name),
        None,
    )


def _feature_vectorizer_type(
    node: NodeProto, first: tuple[int, ...]
) -> TypeProto | None:
    """`FeatureVectorizer`'s result: one row per input row, `inputdimensions` wide in total."""
    widths = _attribute(node, "inputdimensions")
    if widths is None or len(first) not in (1, 2):
        return None
    return helper.make_tensor_type_proto(
        TensorProto.FLOAT, [first[0], sum(widths.ints)]
    )


def _set_tensor_type(graph: GraphProto, name: str, type_proto: TypeProto) -> None:
    """Record `name`'s type, on every entry describing it and on a new one if there is none."""
    entries = [
        entry for entry in (*graph.output, *graph.value_info) if entry.name == name
    ]
    if not entries:
        entries = [graph.value_info.add()]
        entries[0].name = name
    for entry in entries:
        entry.type.CopyFrom(type_proto)


def _has_untyped_tensor(graph: GraphProto) -> bool:
    types = tensor_types(graph)
    produced = (name for node in graph.node for name in node.output if name)
    return any(
        static_shape(types.get(name)) is None
        for name in (*produced, *(entry.name for entry in graph.output))
    )


def tensor_types(graph: GraphProto) -> dict[str, TypeProto]:
    """Type of every tensor the graph names, initializers included."""
    types = {
        initializer.name: helper.make_tensor_type_proto(
            initializer.data_type, list(initializer.dims)
        )
        for initializer in graph.initializer
    }
    for value_info in (*graph.input, *graph.output, *graph.value_info):
        if value_info.HasField("type"):
            types[value_info.name] = value_info.type
    return types


def static_shape(type_proto: TypeProto | None) -> tuple[int, ...] | None:
    """Concrete dimensions of a tensor type, or None if any of them is not static.

    A zero-sized dimension is static: zero-element tensors are legal in ONNX.
    """
    if type_proto is None or type_proto.WhichOneof("value") != "tensor_type":
        return None
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.WhichOneof("value") != "dim_value" or dim.dim_value < 0:
            return None
        dims.append(dim.dim_value)
    return tuple(dims)


def graph_label(graph: GraphProto) -> str:
    return graph.name or "<unnamed>"


def _bind_dim(
    dim: TensorShapeProto.Dimension,
    dim_bindings: Mapping[str, int],
    applied: dict[str, int],
) -> None:
    if dim.WhichOneof("value") == "dim_value":
        return
    name = dim.dim_param
    dim.dim_value = dim_bindings.get(name, UNBOUND_DIM_DEFAULT)
    if name:
        applied[name] = dim.dim_value


def _dims(value_info: ValueInfoProto) -> Iterator[TensorShapeProto.Dimension]:
    yield from _shape_dims(value_info.type)


def _shape_dims(type_proto: TypeProto) -> Iterator[TensorShapeProto.Dimension]:
    if type_proto.WhichOneof("value") != "tensor_type":
        return
    yield from type_proto.tensor_type.shape.dim
