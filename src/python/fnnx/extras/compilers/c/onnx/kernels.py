"""The kernel-generator convention and the registry ONNX kernels register into."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx.defs
from onnx import NodeProto, TensorProto, helper
from onnx.numpy_helper import from_array, to_array

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.registry import (
    KernelRegistry,
    latest_semantic_revision,
)


@dataclass(frozen=True)
class TensorRef:
    """A tensor a kernel reads or writes, with the C expression naming its buffer."""

    name: str
    elem_type: int
    shape: tuple[int, ...]
    expr: str

    @property
    def elem_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class CFunction:
    """A shared `static` kernel definition; `name` is what it is deduplicated by."""

    name: str
    definition: str


@dataclass(frozen=True)
class NodeContext:
    """Everything a kernel generator needs to emit code for one node.

    `inputs`/`outputs` hold None where the node omits an optional operand. `prefix` is the
    artifact-wide symbol prefix: kernel names build on it so that every static definition in
    the emitted header is unique, while kernels shared between nodes still deduplicate.
    `constants` holds the graph's compile-time values, for the operands an op reads as
    configuration rather than as data.
    """

    node: NodeProto
    domain: str
    opset_version: int
    since_version: int
    prefix: str
    inputs: tuple[TensorRef | None, ...]
    outputs: tuple[TensorRef | None, ...]
    constants: Mapping[str, TensorProto] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.node.name or f"<unnamed {self.node.op_type}>"

    def require_input(self, index: int) -> TensorRef:
        return self._require(self.inputs, index, "input")

    def optional_input(self, index: int) -> TensorRef | None:
        return self.inputs[index] if index < len(self.inputs) else None

    def require_output(self, index: int) -> TensorRef:
        return self._require(self.outputs, index, "output")

    def attribute(self, name: str, default: Any) -> Any:
        """The node's `name` attribute as a Python value, or `default` when absent."""
        for entry in self.node.attribute:
            if entry.name == name:
                return helper.get_attribute_value(entry)
        return default

    def float_attribute(self, name: str) -> float:
        """The node's `name` attribute, defaulting to the one the op's schema declares.

        Reading the default off the schema rather than restating it keeps a kernel from
        drifting from the value ONNX's own tooling — and the reference evaluator — applies.
        """
        return float(
            self.attribute(name, self._schema().attributes[name].default_value.f)
        )

    def int_attribute(self, name: str) -> int:
        """The node's `name` integer attribute, defaulting to the schema's own default."""
        return int(
            self.attribute(name, self._schema().attributes[name].default_value.i)
        )

    def string_attribute(self, name: str) -> str:
        """The node's `name` string attribute, defaulting to the schema's own default."""
        value = self.attribute(name, self._schema().attributes[name].default_value.s)
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def constant_input(self, index: int) -> np.ndarray | None:
        """The compile-time value of input `index`, or None when it has none.

        Only an operand every path through the graph fixes has one: an initializer, or a
        tensor constant folding resolved. A runtime value reads as None.
        """
        operand = self.inputs[index] if index < len(self.inputs) else None
        if operand is None:
            return None
        tensor = self.constants.get(operand.name)
        return None if tensor is None else to_array(tensor)

    def _schema(self) -> onnx.defs.OpSchema:
        return onnx.defs.get_schema(self.node.op_type, self.since_version, self.domain)

    def _require(
        self, operands: tuple[TensorRef | None, ...], index: int, role: str
    ) -> TensorRef:
        operand = operands[index] if index < len(operands) else None
        if operand is None:
            raise CompileError(
                f"Node `{self.label}`: op `{self.node.op_type}` requires {role} {index}, "
                "which this node leaves out."
            )
        return operand


@dataclass(frozen=True)
class ScratchBuffer:
    """Static working storage a kernel needs beyond the tensors it reads and writes.

    A few kernels cannot compute in place — a determinant eliminates on a copy of its matrix
    — and the artifact allocates nothing, so the space is reserved at compile time and counted
    in the reported footprint like every other buffer. `symbol` is what it is deduplicated by:
    the nodes that share a kernel share its buffer, sized for the largest of them, which is
    safe under the artifact's one-call-at-a-time contract.
    """

    symbol: str
    elem_type: int
    elem_count: int


@dataclass(frozen=True)
class ConstantData:
    """A table a kernel reads from `static const` storage rather than from an operand.

    ONNX-ML carries an op's parameters in attributes rather than in inputs — a scaler's
    per-feature offsets, an encoder's categories, an ensemble's nodes — and those tables run
    long enough that passing them as compound literals would put them on the stack and leave
    them out of the reported footprint. `symbol` is what the data is deduplicated by, so it
    has to encode the contents; `constant_data` builds one that does.
    """

    symbol: str
    tensor: TensorProto


@dataclass(frozen=True)
class NodeEmission:
    """What a kernel generator contributes: shared functions plus the call site."""

    functions: tuple[CFunction, ...]
    statements: tuple[str, ...]
    scratch: tuple[ScratchBuffer, ...] = ()
    constants: tuple[ConstantData, ...] = ()


KernelGenerator = Callable[[NodeContext], NodeEmission]

KERNELS: KernelRegistry[KernelGenerator] = KernelRegistry()


def register_kernel(
    domain: str, op_type: str, versions: Iterable[int], generator: KernelGenerator
) -> None:
    """Register `generator` at each listed schema revision the installed `onnx` defines.

    ONNX bumps an op's `since_version` on every spec change, including the type-constraint
    additions that leave the emitted code identical, so one generator usually covers several
    revisions — and each has to be registered, or the semantic-revision guard rejects the
    kernel at the newer opset. `versions` is therefore the explicit claim of which revisions
    this generator implements: ones the installed `onnx` package does not define are skipped
    (keeping kernels installable across the supported `onnx` range), and ones that are not
    listed are left to the guard rather than silently served with older semantics.
    """
    for version in versions:
        if latest_semantic_revision(domain, op_type, version) == version:
            KERNELS.register(domain, op_type, version, generator)


def copy_tensor(source: TensorRef, result: TensorRef) -> NodeEmission:
    """`source`'s elements written into `result`, where no per-element code is needed.

    A straight copy: an identity, or a reinterpretation of the same bytes at another element
    type of the same width. A shared kernel would not earn its keep over one `memcpy`.
    """
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())
    return NodeEmission(
        functions=(),
        statements=(
            f"memcpy({result.expr}, {source.expr}, "
            f"{result.elem_count}u * sizeof(*{result.expr}));",
        ),
    )


def constant_data(
    context: NodeContext, role: str, values: np.ndarray
) -> tuple[ConstantData, str]:
    """`values` as static constant data, and the C expression naming it.

    The symbol carries a digest of the contents, so two nodes reading the same table share
    one definition and two different tables can never collide on a name.
    """
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(
        f"{array.dtype.str}{array.shape}".encode() + array.tobytes()
    )
    symbol = (
        f"{context.prefix}_{context.node.op_type.lower()}_{role}_"
        f"{digest.hexdigest()[:12]}"
    )
    return ConstantData(symbol, from_array(array, symbol)), symbol


def broadcast_strides(
    source: TensorRef, shape: tuple[int, ...], *, node_label: str
) -> tuple[int, ...]:
    """Row-major strides addressing `source` while iterating a tensor of `shape`.

    A stride is zero on every axis `source` is broadcast along, so the same element is read
    for every coordinate on that axis; `source` is aligned to the trailing axes, as ONNX's
    broadcasting rules prescribe.
    """
    if len(source.shape) > len(shape):
        raise _broadcast_error(source, shape, node_label)
    padded = (1,) * (len(shape) - len(source.shape)) + source.shape
    strides = []
    stride = 1
    for size, target in zip(reversed(padded), reversed(shape)):
        if size == target:
            strides.append(stride)
        elif size == 1:
            strides.append(0)
        else:
            raise _broadcast_error(source, shape, node_label)
        stride *= size
    return tuple(reversed(strides))


def _broadcast_error(
    source: TensorRef, shape: tuple[int, ...], node_label: str
) -> CompileError:
    return CompileError(
        f"Node `{node_label}`: tensor `{source.name}` of shape {list(source.shape)} does "
        f"not broadcast to {list(shape)}."
    )


# Imported for the side effect of registering the kernels; the import sits at the bottom
# because every op module builds on the definitions above.
from fnnx.extras.compilers.c.onnx import ops  # noqa: E402, F401
