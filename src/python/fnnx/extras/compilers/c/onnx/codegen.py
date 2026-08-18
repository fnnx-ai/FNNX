"""Planning a prepared graph into C: buffers, kernel dispatch, and the entrypoint body."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from onnx import GraphProto, NodeProto, TensorProto, TypeProto, helper

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import element_type_name
from fnnx.extras.compilers.c.onnx.emit import UniqueNames, sanitize_identifier
from fnnx.extras.compilers.c.onnx.frontend import PreparedModel
from fnnx.extras.compilers.c.onnx.functions import (
    MAX_EXPANSION_DEPTH,
    Expansion,
    expand_function,
)
from fnnx.extras.compilers.c.onnx.kernels import (
    KERNELS,
    CFunction,
    ConstantData,
    KernelGenerator,
    NodeContext,
    ScratchBuffer,
    TensorRef,
)
from fnnx.extras.compilers.c.onnx.loader import display_domain, normalize_domain
from fnnx.extras.compilers.c.onnx.registry import KernelSpec
from fnnx.extras.compilers.c.onnx.runtime_dims import RuntimeDim, ShapeTerm
from fnnx.extras.compilers.c.onnx.shapes import graph_label, static_shape, tensor_types

DEFAULT_PREFIX = "fnnx_model"


@dataclass(frozen=True)
class IOTensor:
    """A tensor the caller provides a buffer for: a graph input or output.

    `shape` is the buffer's capacity — the extents at the runtime dimensions' maxima — and
    `runtime_shape`, present only where the artifact has runtime dimensions, says which of
    those extents scale with which dimension.
    """

    name: str
    c_name: str
    macro: str
    elem_type: int
    shape: tuple[int, ...]
    runtime_shape: tuple[ShapeTerm, ...] = ()
    owner: str = ""

    @property
    def elem_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class StaticBuffer:
    """A `static` array the implementation owns: an embedded weight, or scratch space."""

    name: str
    symbol: str
    elem_type: int
    shape: tuple[int, ...]
    tensor: TensorProto | None

    @property
    def elem_count(self) -> int:
        return math.prod(self.shape)

    @property
    def declared_count(self) -> int:
        """C99 has no zero-length arrays, yet a zero-element tensor still needs an address."""
        return max(1, self.elem_count)


@dataclass(frozen=True)
class LabelTable:
    """Class names the header publishes alongside the output tensor they describe."""

    tensor: str
    symbol: str
    macro: str
    elem_type: int
    values: tuple[str, ...] | tuple[int, ...]


@dataclass(frozen=True)
class NodeEntry:
    """A per-node entrypoint the artifact publishes beside the whole-model one.

    `id` is the FNNX op-instance id the caller knows the node by; `symbol` is what that id
    was sanitized to, which is what the header actually declares.
    """

    id: str
    symbol: str
    inputs: tuple[IOTensor, ...]
    outputs: tuple[IOTensor, ...]
    body: tuple[str, ...]
    body_owners: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactScope:
    """What a graph shares with the other graphs compiled into the same header.

    `prefix` is the artifact-wide prefix kernels, their scratch and their constant tables
    are named from, so that graphs using the same kernel share one definition; `symbols`
    hands out every other identifier, keeping the graphs' own definitions distinct.
    """

    prefix: str
    symbols: UniqueNames


@dataclass(frozen=True)
class Program:
    """Everything the header renderer needs, with every ordering already fixed.

    `body_owners` names, per statement, the node the statement was emitted for; it is what
    lets a failure to compile a whole family of shapes point back at a node of the model.
    """

    prefix: str
    graph_name: str
    source: str
    opsets: dict[str, int]
    dim_bindings: dict[str, int]
    inputs: tuple[IOTensor, ...]
    outputs: tuple[IOTensor, ...]
    weights: tuple[StaticBuffer, ...]
    scratch: tuple[StaticBuffer, ...]
    functions: tuple[CFunction, ...]
    body: tuple[str, ...]
    labels: tuple[LabelTable, ...] = ()
    nodes: tuple[NodeEntry, ...] = ()
    runtime_dims: tuple[RuntimeDim, ...] = ()
    body_owners: tuple[str, ...] = ()


def build_program(
    prepared: PreparedModel,
    *,
    prefix: str | None = None,
    scope: ArtifactScope | None = None,
    runtime_dims: tuple[RuntimeDim, ...] = (),
) -> Program:
    """Plan the C code for a prepared graph.

    `prefix` defaults to the graph name, sanitized to a C identifier, and to `fnnx_model`
    when that leaves nothing. `scope` compiles this graph as one of several sharing a
    header — kernels deduplicated between them, identifiers kept apart — and defaults to a
    scope of the graph's own; a caller passing one owns reserving `runtime_dims`' parameter
    names in it, so that no tensor is emitted under a name a parameter already carries.
    """
    return _ProgramBuilder(prepared, prefix, scope, runtime_dims).build()


def reserve_dim_parameters(
    names: UniqueNames, runtime_dims: Sequence[RuntimeDim]
) -> None:
    """Claim the entrypoint parameters the runtime dimensions take, before any tensor does."""
    for dim in runtime_dims:
        taken = names.assign(dim.c_name, fallback=dim.c_name)
        assert taken == dim.c_name


@dataclass
class _Slot:
    """Where a tensor's data lives, as a C expression, plus its static type."""

    expr: str
    elem_type: int
    shape: tuple[int, ...]


def _slot_for(ref: TensorRef) -> _Slot:
    return _Slot(ref.expr, ref.elem_type, ref.shape)


def _initializers(graph: GraphProto) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _bound(ref: TensorRef | None) -> TensorRef:
    """An operand an expansion binds; it only ever binds ones the node actually passes."""
    if ref is None:
        raise CompileError("the function body binds an operand the node does not pass.")
    return ref


@dataclass
class _ProgramBuilder:
    prepared: PreparedModel
    requested_prefix: str | None
    requested_scope: ArtifactScope | None = None
    runtime_dims: tuple[RuntimeDim, ...] = ()

    prefix: str = field(init=False)
    scope: ArtifactScope = field(init=False)
    types: dict[str, TypeProto] = field(init=False)
    constants: dict[str, TensorProto] = field(init=False)
    slots: dict[str, _Slot] = field(default_factory=dict, init=False)
    depth: int = field(default=0, init=False)
    referenced: set[str] = field(default_factory=set, init=False)
    weights: list[StaticBuffer] = field(default_factory=list, init=False)
    scratch: list[StaticBuffer] = field(default_factory=list, init=False)
    kernel_scratch: dict[str, StaticBuffer] = field(default_factory=dict, init=False)
    constants_data: dict[str, TensorProto] = field(default_factory=dict, init=False)
    functions: dict[str, CFunction] = field(default_factory=dict, init=False)
    statements: list[str] = field(default_factory=list, init=False)
    statement_owners: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        graph = self.prepared.model.graph
        source = (
            self.requested_prefix if self.requested_prefix is not None else graph.name
        )
        self.prefix = sanitize_identifier(source, fallback=DEFAULT_PREFIX)
        if self.requested_scope is not None:
            self.scope = self.requested_scope
        else:
            self.scope = ArtifactScope(self.prefix, UniqueNames())
            reserve_dim_parameters(self.scope.symbols, self.runtime_dims)
        self.types = tensor_types(graph)
        self.constants = _initializers(graph)

    @property
    def graph(self) -> GraphProto:
        return self.prepared.model.graph

    @property
    def names(self) -> UniqueNames:
        """One namespace for parameters and static buffers alike, across every graph.

        A parameter that happened to match a buffer symbol would shadow it inside the
        entrypoint; a buffer two graphs of one artifact both named would be defined twice.
        """
        return self.scope.symbols

    def _emit(self, statements: Iterable[str], owner: str) -> None:
        for statement in statements:
            self.statements.append(statement)
            self.statement_owners.append(owner)

    def build(self) -> Program:
        inputs = self._plan_inputs()
        self._plan_weights()
        outputs, copies = self._plan_outputs()
        labels = self._plan_labels(outputs)
        for node in self.graph.node:
            self._emit_node(node)
        self._emit(copies, "")
        unused = [
            f"(void){tensor.c_name};"
            for tensor in (*inputs, *outputs)
            if tensor.c_name not in self.referenced
        ]
        return Program(
            prefix=self.prefix,
            graph_name=self.graph.name,
            source=f"ONNX graph `{self.graph.name}`",
            opsets=dict(self.prepared.opsets),
            dim_bindings=dict(self.prepared.dim_bindings),
            inputs=inputs,
            outputs=outputs,
            # A buffer no emitted statement names — a weight nothing reads, or the
            # intermediate of a zero-element copy that compiles to nothing — would be a
            # `static` the C compiler warns about, so only the referenced ones are emitted.
            weights=tuple(
                weight for weight in self.weights if weight.symbol in self.referenced
            ),
            scratch=tuple(
                buffer for buffer in self.scratch if buffer.symbol in self.referenced
            ),
            functions=tuple(self.functions.values()),
            body=tuple(unused + self.statements),
            labels=labels,
            runtime_dims=self.runtime_dims,
            body_owners=tuple([""] * len(unused) + self.statement_owners),
        )

    def _plan_labels(self, outputs: tuple[IOTensor, ...]) -> tuple[LabelTable, ...]:
        """Give every class-label table a symbol and a macro, keyed to its output tensor."""
        by_name = {tensor.name: tensor for tensor in outputs}
        tables = []
        for labels in self.prepared.class_labels:
            tensor = by_name.get(labels.tensor)
            if tensor is None:
                raise CompileError(
                    f"Graph `{graph_label(self.graph)}`: the class labels of "
                    f"`{labels.tensor}` describe a tensor the graph does not output."
                )
            symbol = self.names.assign(
                f"{self.prefix}_classlabels_{tensor.c_name}",
                fallback=f"{self.prefix}_classlabels",
            )
            tables.append(
                LabelTable(
                    tensor=tensor.name,
                    symbol=symbol,
                    # Derived from the symbol rather than from the output's own macro family,
                    # because the symbol is the name `self.names` has already made unique:
                    # two tables keying one tensor would otherwise share a macro and define
                    # it twice with the two lengths.
                    macro=f"{symbol.upper()}_COUNT",
                    elem_type=labels.elem_type,
                    values=labels.values,
                )
            )
        return tuple(tables)

    def _plan_inputs(self) -> tuple[IOTensor, ...]:
        inputs = []
        for entry in self.graph.input:
            elem_type, shape = self._tensor_type(entry.name, f"input `{entry.name}`")
            c_name = self.names.assign(entry.name, fallback="input")
            self.slots[entry.name] = _Slot(c_name, elem_type, shape)
            inputs.append(
                self._io_tensor(entry.name, c_name, "INPUT", elem_type, shape)
            )
        return tuple(inputs)

    def _plan_weights(self) -> None:
        """Embed the initializers the emitted code actually reads, in graph order.

        One already bound to a buffer is left alone: a function body takes the operands its
        caller fixes as initializers so that folding can read them, and the caller's own
        buffer already holds those bytes.
        """
        referenced = {name for node in self.graph.node for name in node.input if name}
        referenced |= {entry.name for entry in self.graph.output}
        for initializer in self.graph.initializer:
            if initializer.name not in referenced or initializer.name in self.slots:
                continue
            symbol = self._storage_symbol("w", initializer.name)
            shape = tuple(initializer.dims)
            self.slots[initializer.name] = _Slot(symbol, initializer.data_type, shape)
            self.weights.append(
                StaticBuffer(
                    initializer.name, symbol, initializer.data_type, shape, initializer
                )
            )

    def _plan_outputs(self) -> tuple[tuple[IOTensor, ...], list[str]]:
        """Give every graph output a parameter, copying into it where it aliases a buffer.

        An output a node computes is written straight into the caller's buffer, and every
        downstream consumer reads it from there. One that aliases an input, a folded
        constant or an earlier output is copied instead, after all nodes have run.
        """
        produced = {
            name: node.name or f"<unnamed {node.op_type}>"
            for node in self.graph.node
            for name in node.output
            if name
        }
        outputs = []
        copies = []
        for entry in self.graph.output:
            source = self.slots.get(entry.name)
            c_name = self.names.assign(entry.name, fallback="output")
            if source is None:
                if entry.name not in produced:
                    raise CompileError(
                        f"Graph `{graph_label(self.graph)}`: output `{entry.name}` is not "
                        "produced by any node, input or initializer."
                    )
                elem_type, shape = self._tensor_type(
                    entry.name, f"output `{entry.name}`"
                )
                self.slots[entry.name] = _Slot(c_name, elem_type, shape)
            else:
                elem_type, shape = source.elem_type, source.shape
                count = math.prod(shape)
                if count:
                    self.referenced.add(c_name)
                    self._mark_used(source)
                    copies.append(
                        f"memcpy({c_name}, {source.expr}, "
                        f"{count}u * sizeof(*{c_name}));"
                    )
            outputs.append(
                self._io_tensor(
                    entry.name,
                    c_name,
                    "OUTPUT",
                    elem_type,
                    shape,
                    owner=produced.get(entry.name, ""),
                )
            )
        return tuple(outputs), copies

    def _emit_node(self, node: NodeProto) -> None:
        """Compile one node: its native kernel, else its ONNX function body, else an error."""
        label = node.name or f"<unnamed {node.op_type}>"
        domain = normalize_domain(node.domain)
        opset_version = self.prepared.opsets.get(domain)
        if opset_version is None:
            raise CompileError(
                f"Graph `{graph_label(self.graph)}`: node `{label}` uses domain "
                f"`{display_domain(domain)}`, which the model does not import an opset for."
            )
        inputs = tuple(self._read_ref(name, label) for name in node.input)
        outputs = tuple(self._write_ref(name) for name in node.output)
        spec = KERNELS.select(domain, node.op_type, opset_version)
        if spec is not None:
            self._emit_kernel(spec, node, domain, opset_version, inputs, outputs, label)
        elif not self._expand_node(node, domain, opset_version, label, inputs, outputs):
            raise KERNELS.unsupported_op_error(
                domain, node.op_type, opset_version, node_name=label
            )

    def _emit_kernel(
        self,
        spec: KernelSpec[KernelGenerator],
        node: NodeProto,
        domain: str,
        opset_version: int,
        inputs: tuple[TensorRef | None, ...],
        outputs: tuple[TensorRef | None, ...],
        label: str,
    ) -> None:
        context = NodeContext(
            node=node,
            domain=domain,
            opset_version=opset_version,
            since_version=spec.since_version,
            prefix=self.scope.prefix,
            inputs=inputs,
            outputs=outputs,
            constants=self.constants,
        )
        emission = spec.generator(context)
        for function in emission.functions:
            self._add_function(function)
        for constant in emission.constants:
            self._embed_constant(constant)
        for buffer in emission.scratch:
            self._reserve_scratch(buffer)
        self._mark_emitted(context, emission.statements)
        self._emit(emission.statements, label)

    def _expand_node(
        self,
        node: NodeProto,
        domain: str,
        opset_version: int,
        label: str,
        inputs: tuple[TensorRef | None, ...],
        outputs: tuple[TensorRef | None, ...],
    ) -> bool:
        """Compile the node through the function body ONNX defines for its op.

        False means ONNX defines no body, leaving the node unsupported. Anything the body
        itself cannot compile is an error naming the node it was expanded for, so a failure
        several expansions deep still points back at the model's own node.
        """
        if self.depth >= MAX_EXPANSION_DEPTH:
            raise CompileError(
                f"Node `{label}`: ONNX function bodies nested more than "
                f"{MAX_EXPANSION_DEPTH} levels deep; `{node.op_type}` (domain "
                f"`{display_domain(domain)}`) does not expand into primitive ops."
            )
        input_types = tuple(
            None
            if ref is None
            else helper.make_tensor_type_proto(ref.elem_type, list(ref.shape))
            for ref in inputs
        )
        # The values the graph fixes go with the types: a body that computes its own result
        # shape from an operand needs the operand, not just its extents.
        input_values = {
            index: self.constants[ref.name]
            for index, ref in enumerate(inputs)
            if ref is not None and ref.name in self.constants
        }
        try:
            expansion = expand_function(
                node, domain, opset_version, input_types, input_values
            )
            if expansion is None:
                return False
            self._emit_expansion(expansion, inputs, outputs, label)
        except CompileError as error:
            raise CompileError(
                f"Node `{label}`: compiling the ONNX function body of `{node.op_type}` "
                f"(domain `{display_domain(domain)}`, opset version {opset_version}) "
                f"failed: {error}"
            ) from error
        return True

    def _emit_expansion(
        self,
        expansion: Expansion,
        inputs: tuple[TensorRef | None, ...],
        outputs: tuple[TensorRef | None, ...],
        label: str,
    ) -> None:
        """Emit a prepared function body against the expanded node's own buffers.

        The body is a graph of its own — its tensor names, types and opsets are unrelated to
        the enclosing graph's — so it is compiled in a scope of its own, with only the
        caller's buffers shared: the body writes its outputs straight into them.
        """
        outer = (self.prepared, self.types, self.constants, self.slots)
        graph = expansion.prepared.model.graph
        self.prepared = expansion.prepared
        self.types = tensor_types(graph)
        self.constants = _initializers(graph)
        self.slots = {}
        self.depth += 1
        try:
            for name, index in expansion.inputs:
                self.slots[name] = _slot_for(_bound(inputs[index]))
            self._plan_weights()
            for name, index in expansion.outputs:
                self._bind_body_output(name, _bound(outputs[index]))
            for node in graph.node:
                self._emit_node(node)
            for name, index in expansion.outputs:
                self._copy_body_output(name, _bound(outputs[index]), label)
        finally:
            self.prepared, self.types, self.constants, self.slots = outer
            self.depth -= 1

    def _bind_body_output(self, name: str, ref: TensorRef) -> None:
        """Point a body output at the caller's buffer, so the body writes straight into it.

        A name the body already binds — an input it passes through, or a constant folding
        resolved it to — keeps that binding and is copied out afterwards instead.
        """
        declared = self._tensor_type(name, f"output `{name}`")
        if declared != (ref.elem_type, ref.shape):
            raise CompileError(
                f"the body computes `{name}` as "
                f"{element_type_name(declared[0])}{list(declared[1])}, but the node's "
                f"output is {element_type_name(ref.elem_type)}{list(ref.shape)}."
            )
        self.slots.setdefault(name, _slot_for(ref))

    def _copy_body_output(self, name: str, ref: TensorRef, label: str) -> None:
        slot = self.slots[name]
        count = math.prod(slot.shape)
        if slot.expr == ref.expr or not count:
            return
        self.referenced.add(ref.expr)
        self._mark_used(slot)
        self._emit(
            (f"memcpy({ref.expr}, {slot.expr}, {count}u * sizeof(*{ref.expr}));",),
            label,
        )

    def _mark_emitted(self, context: NodeContext, statements: tuple[str, ...]) -> None:
        """Record the buffers the emitted call sites actually name.

        An operand the kernel drops — Gemm's `C` when beta is zero, say — must not keep a
        weight or an entrypoint parameter alive, or the artifact stops building under
        `-Wunused-const-variable` and `-Wunused-parameter`.
        """
        text = "\n".join(statements)
        for ref in (*context.inputs, *context.outputs):
            if ref is not None and re.search(rf"\b{re.escape(ref.expr)}\b", text):
                self.referenced.add(ref.expr)

    def _read_ref(self, name: str, node_label: str) -> TensorRef | None:
        if not name:
            return None
        slot = self.slots.get(name)
        if slot is None:
            raise CompileError(
                f"Graph `{graph_label(self.graph)}`: node `{node_label}` reads tensor "
                f"`{name}`, which no input, initializer or preceding node defines."
            )
        return TensorRef(name, slot.elem_type, slot.shape, slot.expr)

    def _write_ref(self, name: str) -> TensorRef | None:
        if not name:
            return None
        slot = self.slots.get(name)
        if slot is None:
            elem_type, shape = self._tensor_type(name, f"tensor `{name}`")
            symbol = self._storage_symbol("t", name)
            slot = _Slot(symbol, elem_type, shape)
            self.slots[name] = slot
            self.scratch.append(StaticBuffer(name, symbol, elem_type, shape, None))
        return TensorRef(name, slot.elem_type, slot.shape, slot.expr)

    def _add_function(self, function: CFunction) -> None:
        existing = self.functions.get(function.name)
        if existing is not None:
            if existing.definition != function.definition:
                raise CompileError(
                    f"Kernel `{function.name}` was emitted twice with different "
                    "definitions; a kernel name must encode everything its code "
                    "depends on."
                )
            return
        if self.names.is_taken(function.name):
            # Kernel names are composed from the prefix rather than handed out by
            # `self.names`, so that nodes sharing a kernel agree on it; a tensor that
            # sanitizes to the same identifier would shadow the function inside the
            # entrypoint, and the build would fail on an opaque C diagnostic.
            raise CompileError(
                f"Kernel `{function.name}` collides with the identifier already emitted "
                f"for a tensor of that name in graph `{graph_label(self.graph)}`; "
                "compile with an explicit `prefix` to keep the two apart."
            )
        self.functions[function.name] = function

    def _embed_constant(self, constant: ConstantData) -> None:
        """Embed a table a kernel reads from its attributes, once per distinct table.

        It is a weight in every way that matters — `static const` data the call site names,
        counted in the reported footprint — so it is planned as one; what differs is only
        that it comes from an attribute rather than from an initializer.
        """
        existing = self.constants_data.get(constant.symbol)
        if existing is not None:
            if existing.SerializeToString() != constant.tensor.SerializeToString():
                raise CompileError(
                    f"Constant table `{constant.symbol}` was emitted twice with different "
                    "contents; its symbol must encode everything the data depends on."
                )
        else:
            if self.names.is_taken(constant.symbol):
                # Composed from the prefix rather than handed out by `self.names`, so that
                # nodes reading the same table agree on it; a tensor sanitizing to the same
                # identifier would shadow the data inside the entrypoint.
                raise CompileError(
                    f"Constant table `{constant.symbol}` collides with the identifier "
                    f"already emitted for a tensor of that name in graph "
                    f"`{graph_label(self.graph)}`; compile with an explicit `prefix` to "
                    "keep the two apart."
                )
            self.constants_data[constant.symbol] = constant.tensor
            self.weights.append(
                StaticBuffer(
                    constant.symbol,
                    constant.symbol,
                    constant.tensor.data_type,
                    tuple(constant.tensor.dims),
                    constant.tensor,
                )
            )
        self.referenced.add(constant.symbol)

    def _reserve_scratch(self, buffer: ScratchBuffer) -> None:
        """Set aside the working storage a kernel asked for, as one static buffer.

        Nodes calling the same kernel at different shapes ask for different amounts, so the
        buffer grows to the largest of them rather than being reserved once per node; they
        run one after another, which is what makes sharing it safe.
        """
        reserved = self.kernel_scratch.get(buffer.symbol)
        if reserved is not None and reserved.elem_type != buffer.elem_type:
            raise CompileError(
                f"Kernel scratch `{buffer.symbol}` was reserved at two element types; a "
                "scratch symbol must encode everything its storage depends on."
            )
        if reserved is None and self.names.is_taken(buffer.symbol):
            # Composed from the prefix rather than handed out by `self.names`, for the
            # reason kernel names are; a tensor sanitizing to the same identifier would
            # shadow the buffer inside the entrypoint.
            raise CompileError(
                f"Kernel scratch `{buffer.symbol}` collides with the identifier already "
                f"emitted for a tensor of that name in graph `{graph_label(self.graph)}`; "
                "compile with an explicit `prefix` to keep the two apart."
            )
        if reserved is None or reserved.elem_count < buffer.elem_count:
            grown = StaticBuffer(
                buffer.symbol,
                buffer.symbol,
                buffer.elem_type,
                (buffer.elem_count,),
                None,
            )
            if reserved is None:
                self.scratch.append(grown)
            else:
                self.scratch[self.scratch.index(reserved)] = grown
            self.kernel_scratch[buffer.symbol] = grown
        self.referenced.add(buffer.symbol)

    def _mark_used(self, slot: _Slot) -> None:
        self.referenced.add(slot.expr)

    def _io_tensor(
        self,
        name: str,
        c_name: str,
        role: str,
        elem_type: int,
        shape: tuple[int, ...],
        owner: str = "",
    ) -> IOTensor:
        macro = f"{self.prefix.upper()}_{role}_{c_name.upper()}"
        return IOTensor(name, c_name, macro, elem_type, shape, owner=owner)

    def _storage_symbol(self, kind: str, name: str) -> str:
        return self.names.assign(
            f"{self.prefix}_{kind}_{name}", fallback=f"{self.prefix}_{kind}"
        )

    def _tensor_type(self, name: str, role: str) -> tuple[int, tuple[int, ...]]:
        type_proto = self.types.get(name)
        shape = static_shape(type_proto)
        if type_proto is None or shape is None:
            raise CompileError(
                f"Graph `{graph_label(self.graph)}`: {role} has no static tensor type."
            )
        return type_proto.tensor_type.elem_type, shape
