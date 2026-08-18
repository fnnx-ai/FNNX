"""The graph pass that turns an ONNX-ML classifier's map output back into a tensor.

`ZipMap` exists only to pair a classifier's probability tensor with its class names, and its
result is a sequence of maps — not a tensor, so not something the C compiler can hand a
caller a buffer for. The pass removes a trailing one, promotes the probability tensor it was
reading to the graph output in its place, and hands the class names on as metadata the header
publishes alongside that output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.loader import ML_DOMAIN, normalize_domain
from fnnx.extras.compilers.c.onnx.shapes import graph_label

ZIP_MAP = "ZipMap"


@dataclass(frozen=True)
class ClassLabels:
    """The class names a removed `ZipMap` was keying its map output by.

    `tensor` is the graph output the labels describe, one label per element of its trailing
    axis. `elem_type` is `STRING` or `INT64` — the two key types ONNX-ML defines.
    """

    tensor: str
    elem_type: int
    values: tuple[str, ...] | tuple[int, ...]


def remove_zipmap(model: ModelProto) -> tuple[ClassLabels, ...]:
    """Drop every trailing `ZipMap` node, returning the class labels each one carried.

    A `ZipMap` anywhere but at the end of the graph is a compile error: its result is a
    sequence of maps, and nothing this compiler emits can consume or produce one.
    """
    graph = model.graph
    consumed = {name for node in graph.node for name in node.input if name}
    outputs = {entry.name for entry in graph.output}
    labels: list[ClassLabels] = []
    kept: list[NodeProto] = []
    for node in graph.node:
        if node.op_type != ZIP_MAP or normalize_domain(node.domain) != ML_DOMAIN:
            kept.append(node)
            continue
        produced = node.output[0] if node.output else ""
        source = node.input[0] if node.input else ""
        if (
            not produced
            or not source
            or produced in consumed
            or produced not in outputs
        ):
            raise CompileError(
                f"Graph `{graph_label(graph)}`: node `{_label(node)}` produces a sequence "
                "of maps, which the C compiler supports only as the last node on a graph "
                "output, where it is removed and its class labels are published as header "
                "metadata."
            )
        elem_type, values = _class_labels(graph, node)
        labels.append(ClassLabels(source, elem_type, values))
        _promote(graph, produced, source)
    if len(kept) != len(graph.node):
        del graph.node[:]
        graph.node.extend(kept)
    return tuple(labels)


def _class_labels(
    graph: GraphProto, node: NodeProto
) -> tuple[int, tuple[str, ...] | tuple[int, ...]]:
    strings = _attribute(node, "classlabels_strings") or ()
    integers = _attribute(node, "classlabels_int64s") or ()
    if bool(strings) == bool(integers):
        raise CompileError(
            f"Graph `{graph_label(graph)}`: node `{_label(node)}` must set exactly one of "
            "`classlabels_strings` and `classlabels_int64s`."
        )
    if strings:
        return TensorProto.STRING, tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in strings
        )
    return TensorProto.INT64, tuple(int(value) for value in integers)


def _promote(graph: GraphProto, produced: str, source: str) -> None:
    """Make `source` the graph output `produced` was.

    The declared type goes with the node: it described a sequence of maps, while the tensor
    taking its place is typed by shape inference from the classifier that produces it. An
    output the graph already exposes under its own name keeps that single entry.
    """
    replacement = ValueInfoProto()
    replacement.name = source
    already_exposed = any(entry.name == source for entry in graph.output)
    kept = [entry for entry in graph.output if entry.name != produced]
    if not already_exposed:
        kept.insert(_position(graph, produced), replacement)
    del graph.output[:]
    graph.output.extend(kept)


def _position(graph: GraphProto, name: str) -> int:
    """Where `name` sits among the graph outputs, so the replacement keeps its place."""
    return next(index for index, entry in enumerate(graph.output) if entry.name == name)


def _attribute(node: NodeProto, name: str) -> Any:
    for entry in node.attribute:
        if entry.name == name:
            return helper.get_attribute_value(entry)
    return None


def _label(node: NodeProto) -> str:
    return node.name or f"<unnamed {node.op_type}>"
