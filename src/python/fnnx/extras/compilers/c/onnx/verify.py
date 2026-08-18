"""Verification that a prepared graph is fully static and uses only supported types."""

from __future__ import annotations

from collections.abc import Container, Mapping

from onnx import GraphProto, ModelProto, NodeProto, TypeProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import element_type_name, is_supported
from fnnx.extras.compilers.c.onnx.loader import display_domain, normalize_domain
from fnnx.extras.compilers.c.onnx.shapes import (
    graph_label,
    runtime_shape_operand,
    static_shape,
    tensor_types,
)

CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})

# Ops whose output shape is a function of input values, not of input shapes: no binding or
# folding can make them static, so they are rejected by name rather than through the
# generic shape check, which could only report the symptom.
DATA_DEPENDENT_SHAPE_OPS = frozenset(
    {"NonZero", "Unique", "Compress", "NonMaxSuppression"}
)


def verify_static(model: ModelProto) -> None:
    """Raise a `CompileError` unless every tensor of the graph is statically compilable."""
    graph = model.graph
    types = tensor_types(graph)
    constants = {initializer.name for initializer in graph.initializer}
    for node in graph.node:
        _verify_node(graph, node, constants, types)
    if graph.sparse_initializer:
        names = ", ".join(
            f"`{sparse.values.name}`" for sparse in graph.sparse_initializer
        )
        raise CompileError(
            f"Graph `{graph_label(graph)}`: sparse initializers ({names}) are not "
            "supported by the C compiler."
        )

    for entry in graph.input:
        _verify_tensor(
            graph, entry.name, types.get(entry.name), f"input `{entry.name}`"
        )
    for initializer in graph.initializer:
        _verify_tensor(
            graph,
            initializer.name,
            types.get(initializer.name),
            f"initializer `{initializer.name}`",
        )
    for node in graph.node:
        for index, name in enumerate(node.output):
            if not name:
                continue
            _verify_tensor(
                graph,
                name,
                types.get(name),
                f"output {index} of node `{_node_label(node)}`",
            )


def _verify_node(
    graph: GraphProto,
    node: NodeProto,
    constants: Container[str],
    types: Mapping[str, TypeProto],
) -> None:
    domain = normalize_domain(node.domain)
    if domain != "":
        return
    operand = runtime_shape_operand(node, constants, types)
    if operand is not None:
        raise CompileError(
            f"Graph `{graph_label(graph)}`: node `{_node_label(node)}` takes the shape of "
            f"its `{node.op_type}` output from `{operand}`, which no initializer or "
            "constant folding fixes; that shape then depends on input data, which the C "
            "compiler requires to be known at compile time."
        )
    if node.op_type in CONTROL_FLOW_OPS:
        raise CompileError(
            f"Graph `{graph_label(graph)}`: node `{_node_label(node)}` uses control flow "
            f"op `{node.op_type}`, which the C compiler supports only when constant "
            "folding can resolve it away; that needs its inputs and everything its "
            "subgraphs read from this graph to be known at compile time."
        )
    if node.op_type in DATA_DEPENDENT_SHAPE_OPS:
        raise CompileError(
            f"Graph `{graph_label(graph)}`: node `{_node_label(node)}` uses op "
            f"`{node.op_type}` (domain `{display_domain(domain)}`), whose output shape "
            "depends on input data; the C compiler requires every shape to be known at "
            "compile time."
        )


def _verify_tensor(
    graph: GraphProto, name: str, type_proto: TypeProto | None, role: str
) -> None:
    label = f"Graph `{graph_label(graph)}`: {role}"
    if type_proto is None or not type_proto.WhichOneof("value"):
        raise CompileError(
            f"{label} has no type; the C compiler could not infer one for tensor `{name}`."
        )
    kind = type_proto.WhichOneof("value")
    if kind != "tensor_type":
        raise CompileError(
            f"{label} has type `{kind}`, which the C compiler does not support; only "
            "tensors can be compiled."
        )
    elem_type = type_proto.tensor_type.elem_type
    if not is_supported(elem_type):
        raise CompileError(
            f"{label} has element type `{element_type_name(elem_type)}`, which the C "
            "compiler does not support."
        )
    if static_shape(type_proto) is None:
        raise CompileError(
            f"{label} has shape `{_shape_label(type_proto)}`, which is not static; bind "
            "its symbolic dimensions with `dim_bindings` or remove the data-dependent "
            "computation that produces it."
        )


def _shape_label(type_proto: TypeProto) -> str:
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return "<unknown rank>"
    dims = []
    for dim in tensor_type.shape.dim:
        kind = dim.WhichOneof("value")
        dims.append(str(dim.dim_value) if kind == "dim_value" else dim.dim_param or "?")
    return f"[{', '.join(dims)}]"


def _node_label(node: NodeProto) -> str:
    return node.name or f"<unnamed {node.op_type}>"
