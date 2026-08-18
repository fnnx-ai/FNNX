"""ONNX function bodies as compilable sub-models: what dispatch falls back to.

An op no registered kernel serves is compiled through the body ONNX itself defines for it —
the very body `onnx.reference` expands, so the compiler and the oracle it is tested against
work from one definition of the op.

The body is prepared as a **standalone model under its own opset imports** rather than
inlined into the caller's graph. ONNX writes a body against whatever opset suits it, which
need not be the one importing the op — `Relu`'s body is written at opset 18 for a schema
introduced at 14 — so inlining would either interpret body nodes at an opset they were not
written for, or produce a graph (opset 14 importing a node added at 15) that ONNX's own
shape inference rejects. Keeping the body separate means each half is compiled at exactly
the opset it was written for; only the emitted C is spliced together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import onnx.defs
from onnx import (
    AttributeProto,
    FunctionProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
    helper,
)

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.frontend import PreparedModel, prepare_model
from fnnx.extras.compilers.c.onnx.loader import (
    LoadedModel,
    display_domain,
    resolve_opsets,
)

# A body is compiled like any other model, so an op that is function-defined inside a
# function body expands in turn. ONNX's own bodies nest a few levels deep; the cap turns a
# body that expanded into itself into a compile error rather than a stack overflow.
MAX_EXPANSION_DEPTH = 8


@dataclass(frozen=True)
class Binding:
    """How a function body's graph I/O binds to the expanded node's.

    `inputs` and `outputs` pair a name in the body's graph with the index of the node
    operand it stands for; operands the node leaves out have no entry.
    """

    inputs: tuple[tuple[str, int], ...]
    outputs: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FunctionModel(Binding):
    """A function body as a standalone model, before the frontend's passes have run."""

    model: ModelProto


@dataclass(frozen=True)
class Expansion(Binding):
    """The same body once the frontend has made it static."""

    prepared: PreparedModel


def function_body(
    node: NodeProto,
    domain: str,
    opset_version: int,
    input_types: Sequence[TypeProto | None],
) -> FunctionProto | None:
    """The body ONNX defines for the node's op at `opset_version`, or None if it defines none.

    `input_types` holds the static type of each of the node's inputs, None where the node
    leaves an optional operand out. Context-dependent bodies are built from the node and
    those types, which is how an op like `Clip` collapses to just the comparisons its
    optional inputs call for.
    """
    schema = _schema(node.op_type, domain, opset_version)
    return None if schema is None else _body(schema, node, input_types)


def function_model(
    node: NodeProto,
    domain: str,
    opset_version: int,
    input_types: Sequence[TypeProto | None],
    input_values: Mapping[int, TensorProto] = MappingProxyType({}),
) -> FunctionModel | None:
    """The op's ONNX function body as a standalone model, or None if ONNX defines none.

    The body becomes a model under its own opset imports, with the node's input types
    (`input_types`, one per node input) as its graph inputs and the node's attributes
    substituted into it. `input_values` carries the operands the caller's graph already
    fixes, by position: they become initializers as well as inputs, which is what lets a body
    compute its own result shape from one — `CenterCropPad` reads the extents it crops to as
    a tensor, and without the value the body's `Pad` would take a shape no folding can settle.
    """
    schema = _schema(node.op_type, domain, opset_version)
    if schema is None:
        return None
    body = _body(schema, node, input_types)
    if body is None:
        return None

    provided = _provided_types(node, input_types)
    inputs = tuple(
        (formal, index) for index, formal in enumerate(body.input) if index in provided
    )
    outputs = tuple(
        (formal, index)
        for index, formal in enumerate(body.output)
        if index < len(node.output) and node.output[index]
    )
    return FunctionModel(
        inputs=inputs,
        outputs=outputs,
        model=_body_model(node, body, schema, provided, inputs, outputs, input_values),
    )


def expand_function(
    node: NodeProto,
    domain: str,
    opset_version: int,
    input_types: Sequence[TypeProto | None],
    input_values: Mapping[int, TensorProto] = MappingProxyType({}),
) -> Expansion | None:
    """Prepare the op's ONNX function body as a standalone static model, or None if it has none.

    The body goes through the same frontend as a top-level model — shape inference,
    constant folding, static verification — under its own opset imports.
    """
    built = function_model(node, domain, opset_version, input_types, input_values)
    if built is None:
        return None
    model = built.model
    return Expansion(
        prepared=prepare_model(LoadedModel(model=model, opsets=resolve_opsets(model))),
        inputs=built.inputs,
        outputs=built.outputs,
    )


def _schema(op_type: str, domain: str, opset_version: int) -> onnx.defs.OpSchema | None:
    try:
        return onnx.defs.get_schema(op_type, opset_version, domain)
    except onnx.defs.SchemaError:
        return None


def _body(
    schema: onnx.defs.OpSchema,
    node: NodeProto,
    input_types: Sequence[TypeProto | None],
) -> FunctionProto | None:
    # `onnx`'s stubs declare only part of `OpSchema`'s pybind11 surface; these three members
    # exist at runtime, and are the same ones `onnx.reference` dispatches a function on.
    if schema.has_function:  # type: ignore[attr-defined]
        return schema.function_body
    if not schema.has_context_dependent_function:  # type: ignore[attr-defined]
        return None
    types = [entry if entry is not None else TypeProto() for entry in input_types]
    try:
        payload = schema.get_context_dependent_function(  # type: ignore[attr-defined]
            node.SerializeToString(), [entry.SerializeToString() for entry in types]
        )
    except Exception as error:
        raise CompileError(
            f"ONNX could not build the function body of `{node.op_type}` (domain "
            f"`{display_domain(schema.domain)}`) for this node: {error}"
        ) from error
    body = FunctionProto()
    body.ParseFromString(payload)
    return body


def _provided_types(
    node: NodeProto, input_types: Sequence[TypeProto | None]
) -> dict[int, TypeProto]:
    """Type of every operand the node actually passes, by position."""
    provided = {}
    for index, name in enumerate(node.input):
        if not name:
            continue
        type_proto = input_types[index] if index < len(input_types) else None
        if type_proto is None:
            raise CompileError(f"the type of operand `{name}` is not known.")
        provided[index] = type_proto
    return provided


def _body_model(
    node: NodeProto,
    body: FunctionProto,
    schema: onnx.defs.OpSchema,
    provided: Mapping[int, TypeProto],
    inputs: tuple[tuple[str, int], ...],
    outputs: tuple[tuple[str, int], ...],
    values: Mapping[int, TensorProto],
) -> ModelProto:
    """The body as a model computing the node's outputs from the node's input types."""
    omitted = {
        formal: "" for index, formal in enumerate(body.input) if index not in provided
    }
    attributes = _attributes(node, schema)
    nodes = _reachable(
        [_resolved(entry, attributes, omitted) for entry in body.node],
        {name for name, _ in outputs},
    )
    initializers = []
    for formal, index in inputs:
        value = values.get(index)
        if value is None:
            continue
        renamed = TensorProto()
        renamed.CopyFrom(value)
        renamed.name = formal
        initializers.append(renamed)
    graph = helper.make_graph(
        nodes,
        f"{node.op_type}_function_body",
        [helper.make_value_info(formal, provided[index]) for formal, index in inputs],
        [helper.make_empty_tensor_value_info(formal) for formal, _ in outputs],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=list(body.opset_import))


def _attributes(
    node: NodeProto, schema: onnx.defs.OpSchema
) -> dict[str, AttributeProto]:
    """What the body's attribute references resolve against: the node's, then the defaults.

    A body reads the caller's attributes by name, so an attribute the node leaves at its
    default has to be filled in from the schema — otherwise the body node referencing it
    (typically a `Constant` holding the value) is left without one.
    """
    attributes = {entry.name: entry for entry in node.attribute}
    for name, formal in schema.attributes.items():
        if (
            name not in attributes
            and formal.default_value.type != AttributeProto.UNDEFINED
        ):
            attributes[name] = formal.default_value
    return attributes


def _resolved(
    node: NodeProto,
    attributes: Mapping[str, AttributeProto],
    omitted: Mapping[str, str],
) -> NodeProto:
    """A body node with its attribute references substituted and omitted operands blanked."""
    resolved = NodeProto()
    resolved.CopyFrom(node)
    del resolved.input[:]
    resolved.input.extend(omitted.get(name, name) for name in node.input)
    del resolved.attribute[:]
    for attribute in node.attribute:
        if not attribute.ref_attr_name:
            resolved.attribute.append(attribute)
            continue
        supplied = attributes.get(attribute.ref_attr_name)
        if supplied is None:
            # Neither the node nor the schema gives the attribute a value, so the body
            # node keeps none either and falls back on its own op's default.
            continue
        substituted = resolved.attribute.add()
        substituted.CopyFrom(supplied)
        substituted.name = attribute.name
    return resolved


def _reachable(nodes: Sequence[NodeProto], wanted: set[str]) -> list[NodeProto]:
    """The body nodes that contribute to the outputs the caller asked for.

    A body computes every output its op declares; a caller that omits an optional one must
    not have to compile the ops that would have produced it. Function bodies are in
    topological order, so one backwards pass suffices.
    """
    live = set(wanted)
    kept = []
    for node in reversed(nodes):
        if not any(name in live for name in node.output if name):
            continue
        kept.append(node)
        live |= {name for name in node.input if name}
    kept.reverse()
    return kept
