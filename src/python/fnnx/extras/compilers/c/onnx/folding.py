"""Compile-time constant folding through the official ONNX reference evaluator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import lru_cache

import numpy as np
import onnx.defs
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
    helper,
)
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator

# The evaluator's versioned implementation classes are the mechanical record of where it
# distinguishes an op's historical semantics; onnx exposes them only through this builder.
from onnx.reference.ops._op_list import (
    _build_registered_operators as _standard_implementations,
)
from onnx.reference.ops.aionnxml._op_list import (
    _build_registered_operators as _ml_implementations,
)

from fnnx.extras.compilers.c.onnx.loader import (
    ML_DOMAIN,
    STANDARD_DOMAIN,
    normalize_domain,
)
from fnnx.extras.compilers.c.onnx.shapes import static_shape, tensor_types

# Ops whose output is a function of its input's shape alone, so a statically-shaped input
# makes them constant even though its values are not known until run time.
_SHAPE_ONLY_OPS = frozenset({"Shape", "Size"})

# Ops that draw from a random distribution: their output is not a function of their inputs,
# so folding one would bake a single draw into the artifact — silently turning an
# unsupported op into a wrong-but-compiling constant, and making the output non-deterministic
# across compiles. `Dropout` is listed because it samples a mask in training mode; the
# inference-mode identity it also serves is compiled as a kernel instead.
NONDETERMINISTIC_OPS = frozenset(
    {
        "Bernoulli",
        "Dropout",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)


def evaluator_is_version_faithful(
    domain: str, op_type: str, opset_version: int
) -> bool:
    """Whether `onnx.reference` implements the op exactly as `opset_version` defines it.

    The evaluator carries a versioned implementation class per revision whose semantics it
    distinguishes, and applies the newest one to every later opset; an op it has no class
    for at all it runs by expanding the ONNX function body of the schema the imported opset
    selects, which is the newest revision's body from that revision on. Either way it is
    faithful when the requested opset selects a schema revision a versioned class
    implements, or one at or after the newest revision the evaluator distinguishes — the
    schema history's own newest, where there are no versioned classes to go by. Anything
    older gets the wrong semantics silently, and is refused here rather than folded.
    """
    normalized = normalize_domain(domain)
    revision = _schema_revision(normalized, op_type, opset_version)
    if revision is None:
        return False
    implemented = _implemented_revisions(normalized, op_type)
    if implemented and revision in implemented:
        return True
    newest = max(implemented) if implemented else _latest_revision(normalized, op_type)
    return newest is not None and revision >= newest


def fold_constants(model: ModelProto, opsets: Mapping[str, int]) -> int:
    """Replace every node with compile-time-constant inputs by its computed value.

    Returns the number of nodes folded. Nodes the evaluator cannot vouch for or cannot
    execute are left in place for kernel compilation: folding is what makes shape
    computations static, never a correctness requirement.
    """
    graph = model.graph
    constants = {initializer.name: initializer for initializer in graph.initializer}
    types = tensor_types(graph)
    kept: list[NodeProto] = []
    folded: list[TensorProto] = []

    for node in graph.node:
        inputs = _constant_inputs(node, constants, types, opsets)
        results = None if inputs is None else _evaluate(node, inputs, opsets)
        if results is None:
            kept.append(node)
            continue
        for value in results:
            constants[value.name] = value
            folded.append(value)

    if not folded:
        return 0
    node_count = len(graph.node)
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(folded)
    return node_count - len(kept)


def prune_unused_initializers(graph: GraphProto) -> None:
    referenced = _referenced_names(graph)
    kept = [
        initializer
        for initializer in graph.initializer
        if initializer.name in referenced
    ]
    if len(kept) != len(graph.initializer):
        del graph.initializer[:]
        graph.initializer.extend(kept)


def prune_stale_value_info(graph: GraphProto) -> None:
    """Drop intermediate shapes for tensors no node produces any more."""
    produced = {name for node in graph.node for name in node.output if name}
    kept = [entry for entry in graph.value_info if entry.name in produced]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)


def _constant_inputs(
    node: NodeProto,
    constants: Mapping[str, TensorProto],
    types: Mapping[str, TypeProto],
    opsets: Mapping[str, int],
) -> list[TensorProto] | None:
    """Constant tensors the node reads, or None when it cannot be folded."""
    domain = normalize_domain(node.domain)
    if _draws_at_random(node):
        return None
    opset_version = opsets.get(domain)
    if opset_version is None or not evaluator_is_version_faithful(
        domain, node.op_type, opset_version
    ):
        return None

    inputs: dict[str, TensorProto] = {}
    for name in (*node.input, *sorted(_outer_scope_names(node))):
        if not name or name in inputs:
            continue
        constant = constants.get(name)
        if constant is None and domain == STANDARD_DOMAIN:
            constant = _shape_only_placeholder(node, name, types)
        if constant is None:
            return None
        inputs[name] = constant
    return list(inputs.values())


def _draws_at_random(node: NodeProto) -> bool:
    """Whether the node — or any node its subgraphs would run — samples a distribution."""
    if (
        normalize_domain(node.domain) == STANDARD_DOMAIN
        and node.op_type in NONDETERMINISTIC_OPS
    ):
        return True
    return any(
        _draws_at_random(inner) for graph in _subgraphs(node) for inner in graph.node
    )


def _shape_only_placeholder(
    node: NodeProto, name: str, types: Mapping[str, TypeProto]
) -> TensorProto | None:
    """A zero tensor standing in for a `Shape`/`Size` input of known static shape."""
    if node.op_type not in _SHAPE_ONLY_OPS:
        return None
    shape = static_shape(types.get(name))
    if shape is None:
        return None
    try:
        dtype = helper.tensor_dtype_to_np_dtype(types[name].tensor_type.elem_type)
        return from_array(np.zeros(shape, dtype=dtype), name)
    except Exception:
        return None


def _evaluate(
    node: NodeProto, inputs: list[TensorProto], opsets: Mapping[str, int]
) -> list[TensorProto] | None:
    outputs = [name for name in node.output if name]
    graph = helper.make_graph(
        [node],
        "constant_fold",
        [],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        initializer=inputs,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(domain, version) for domain, version in opsets.items()
        ],
    )
    try:
        results = ReferenceEvaluator(model).run(None, {})
        folded = [
            from_array(np.asarray(value), name) for name, value in zip(outputs, results)
        ]
    except Exception:
        return None
    return folded


def _outer_scope_names(node: NodeProto) -> set[str]:
    """Names a node's subgraphs read from the enclosing graph."""
    names: set[str] = set()
    for subgraph in _subgraphs(node):
        names |= _free_names(subgraph)
    return names


def _free_names(graph: GraphProto) -> set[str]:
    bound = {entry.name for entry in graph.input} | {
        initializer.name for initializer in graph.initializer
    }
    free: set[str] = set()
    for node in graph.node:
        free |= {name for name in node.input if name and name not in bound}
        free |= {name for name in _outer_scope_names(node) if name not in bound}
        bound |= {name for name in node.output if name}
    return free


def _referenced_names(graph: GraphProto) -> set[str]:
    names = {entry.name for entry in graph.output}
    for node in graph.node:
        names |= {name for name in node.input if name}
        names |= _outer_scope_names(node)
    return names


def _subgraphs(node: NodeProto) -> Iterator[GraphProto]:
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            yield attribute.g
        elif attribute.type == AttributeProto.GRAPHS:
            yield from attribute.graphs


def _schema_revision(domain: str, op_type: str, opset_version: int) -> int | None:
    try:
        return onnx.defs.get_schema(op_type, opset_version, domain).since_version
    except onnx.defs.SchemaError:
        return None


@lru_cache(maxsize=None)
def _latest_revision(domain: str, op_type: str) -> int | None:
    revisions = [
        schema.since_version
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.name == op_type and schema.domain == domain
    ]
    return max(revisions) if revisions else None


@lru_cache(maxsize=None)
def _implemented_revisions(domain: str, op_type: str) -> frozenset[int] | None:
    """Opset versions the reference evaluator implements separately, or None if it has no
    implementation for the op at all (it may still expand a function body, whose semantics
    this check cannot vouch for)."""
    if domain == STANDARD_DOMAIN:
        registry = _standard_implementations()
    elif domain == ML_DOMAIN:
        registry = _ml_implementations()
    else:
        return None
    implementations = registry.get(op_type)
    if implementations is None:
        return None
    return frozenset(version for version in implementations if isinstance(version, int))
