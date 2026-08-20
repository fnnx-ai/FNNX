"""The FNNX bundle layer: reading a pipeline bundle and compiling it to one header.

The ONNX core knows nothing about FNNX. This module holds everything that does: reading and
validating the bundle, handing each node to a node compiler, and emitting the pipeline glue
that calls the compiled nodes in order.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from onnx import ModelProto, TypeProto, ValueInfoProto, helper

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.api import write_artifact
from fnnx.extras.compilers.c.onnx.codegen import (
    DEFAULT_PREFIX,
    ArtifactScope,
    IOTensor,
    NodeEntry,
    Program,
    StaticBuffer,
    build_program,
    reserve_dim_parameters,
)
from fnnx.extras.compilers.c.onnx.dtypes import C_TYPES, numpy_dtype_name
from fnnx.extras.compilers.c.onnx.emit import UniqueNames, sanitize_identifier
from fnnx.extras.compilers.c.onnx.frontend import prepare_model
from fnnx.extras.compilers.c.onnx.kernels import CFunction
from fnnx.extras.compilers.c.onnx.loader import load_model
from fnnx.extras.compilers.c.onnx.runtime_dims import RuntimeDim, resolve_runtime_dims
from fnnx.extras.compilers.c.onnx.shapes import (
    UNBOUND_DIM_DEFAULT,
    drop_shadowed_inputs,
)
from fnnx.extras.compilers.c.onnx.specialize import specialize
from fnnx.extras.compilers.c.result import CompileResult
from fnnx.handlers._common import unpack_model
from fnnx.validators.model_schema import (
    validate_manifest,
    validate_op_instances,
    validate_variant,
)

PIPELINE_VARIANT = "pipeline"
ONNX_OP = "ONNX_v1"

MANIFEST_FILE = "manifest.json"
OPS_FILE = "ops.json"
VARIANT_FILE = "variant_config.json"
ARTIFACTS_DIR = "ops_artifacts"
ONNX_MODEL_FILE = "model.onnx"

NDJSON_CONTENT_TYPE = "NDJSON"

_ARRAY_DTYPE = re.compile(r"^Array\[(.+)\]$")

# The FNNX element names the artifact has a C type for. An `Array[...]` of anything else —
# a float16, a runtime string — is a compile error naming the entry that asks for it.
ELEMENT_TYPES = {numpy_dtype_name(elem_type): elem_type for elem_type in C_TYPES}


def compile_bundle(
    bundle_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    dim_bindings: Mapping[str, int] | None = None,
    runtime_dims: Mapping[str, int] | None = None,
    prefix: str | None = None,
) -> CompileResult:
    """Compile an FNNX pipeline bundle into a single self-contained C99 header.

    `bundle_path` is a bundle directory or a tar-packaged bundle, the two forms the runtime
    accepts. `prefix` defaults to the manifest's name, sanitized to a C identifier.
    `runtime_dims` maps a symbolic dimension to the largest size the artifact must serve,
    leaving the actual size to each call. Compilation is all-or-nothing: nothing is written
    unless the whole bundle compiles.
    """
    dims = resolve_runtime_dims(runtime_dims, dim_bindings)
    source = Path(bundle_path)
    if not source.exists():
        raise CompileError(f"FNNX bundle not found: `{source}`.")
    directory, temporary = _unpack(source)
    try:
        bundle = read_bundle(directory, source_name=source.name)

        def build(bindings: Mapping[str, int]) -> Program:
            return _PipelineBuilder(bundle, dict(bindings), prefix, dims).build()

        program = (
            specialize(build, dims, dim_bindings=dim_bindings or {})
            if dims
            else build(dim_bindings or {})
        )
    finally:
        if temporary:
            shutil.rmtree(directory, ignore_errors=True)
    return write_artifact(
        program,
        output_dir,
        options={
            "prefix": prefix,
            "dim_bindings": dict(sorted((dim_bindings or {}).items())),
            "runtime_dims": dict((runtime_dims or {}).items()),
        },
    )


# --------------------------------------------------------------------------------------
# The bundle, as the compiler needs it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IOSpec:
    """One tensor of an FNNX op spec: its element type and its (partly symbolic) shape."""

    elem_type: int
    shape: tuple[int | str, ...]

    def bind(self, dim_bindings: Mapping[str, int]) -> tuple[int, ...]:
        return tuple(
            size
            if isinstance(size, int)
            else dim_bindings.get(size, UNBOUND_DIM_DEFAULT)
            for size in self.shape
        )


@dataclass(frozen=True)
class ManifestTensor:
    """A pipeline input or output as the manifest declares it.

    An empty `shape` leaves the extent unconstrained: it then follows from the nodes the
    tensor is wired to, and a non-empty one has to agree with what those nodes say.
    """

    name: str
    elem_type: int
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class OpInstance:
    """One entry of `ops.json`, with the directory holding its artifacts."""

    id: str
    op: str
    inputs: tuple[IOSpec, ...]
    outputs: tuple[IOSpec, ...]
    attributes: Mapping[str, Any]
    artifact_dir: Path


@dataclass(frozen=True)
class PipelineNode:
    """One `variant_config` node: the op instance it runs and the edges it is wired to."""

    instance: OpInstance
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Bundle:
    """A pipeline bundle, read.

    `name` is the manifest's own name, which the artifact's prefix defaults to and which is
    empty for the many bundles that carry none; `label` is what the header calls the bundle
    when it says where it came from, and falls back to the file it was read from.
    """

    name: str
    label: str
    inputs: tuple[ManifestTensor, ...]
    outputs: tuple[ManifestTensor, ...]
    nodes: tuple[PipelineNode, ...]


def read_bundle(directory: Path, *, source_name: str) -> Bundle:
    """Read an unpacked bundle directory, validating it against the C compiler's contract."""
    manifest = _read_json(directory / MANIFEST_FILE)
    ops = _read_json(directory / OPS_FILE)
    variant_config = _read_json(directory / VARIANT_FILE)

    _validate(validate_manifest, manifest, MANIFEST_FILE)
    variant = manifest.get("variant")
    if variant != PIPELINE_VARIANT:
        raise CompileError(
            f"The C compiler compiles `{PIPELINE_VARIANT}` bundles; this one is "
            f"`{variant}`."
        )
    _validate(
        lambda config: validate_variant(variant, config), variant_config, VARIANT_FILE
    )
    _reject_unknown_ops(ops)
    _validate(validate_op_instances, ops, OPS_FILE)
    _reject_dynamic_attributes(manifest, ops, variant_config)

    instances = {
        instance["id"]: _read_op_instance(instance, directory) for instance in ops
    }
    name = manifest.get("name") or ""
    return Bundle(
        name=name,
        label=name or source_name,
        inputs=_read_manifest_tensors(manifest["inputs"], "input"),
        outputs=_read_manifest_tensors(manifest["outputs"], "output"),
        nodes=tuple(
            _read_pipeline_node(node, instances) for node in variant_config["nodes"]
        ),
    )


def _read_manifest_tensors(
    entries: Sequence[Mapping[str, Any]], role: str
) -> tuple[ManifestTensor, ...]:
    tensors = []
    seen: set[str] = set()
    for entry in entries:
        label = f"Manifest {role} `{entry['name']}`"
        content_type = entry.get("content_type")
        if content_type != NDJSON_CONTENT_TYPE:
            raise CompileError(
                f"{label} has content type `{content_type}`; the C compiler compiles only "
                f"`{NDJSON_CONTENT_TYPE}` tensors."
            )
        if entry["name"] in seen:
            # The entrypoint takes one parameter per manifest tensor, and the edge of that
            # name can only be wired to one of them; the other would be a parameter nothing
            # reads, which the artifact's own `-Werror` build contract refuses.
            raise CompileError(
                f"{label} is declared twice; every {role} needs its own name."
            )
        seen.add(entry["name"])
        tensors.append(
            ManifestTensor(
                name=entry["name"],
                elem_type=_element_type(entry["dtype"], label),
                shape=_read_shape(entry["shape"], label),
            )
        )
    return tuple(tensors)


def _read_op_instance(entry: Mapping[str, Any], directory: Path) -> OpInstance:
    label = f"Op instance `{entry['id']}`"
    return OpInstance(
        id=entry["id"],
        op=entry["op"],
        inputs=_read_specs(entry["inputs"], f"{label} input"),
        outputs=_read_specs(entry["outputs"], f"{label} output"),
        attributes=entry["attributes"],
        artifact_dir=directory / ARTIFACTS_DIR / entry["id"],
    )


def _read_specs(entries: Sequence[Mapping[str, Any]], role: str) -> tuple[IOSpec, ...]:
    return tuple(
        IOSpec(
            elem_type=_element_type(entry["dtype"], f"{role} {index}"),
            shape=_read_shape(entry["shape"], f"{role} {index}"),
        )
        for index, entry in enumerate(entries)
    )


def _read_pipeline_node(
    node: Mapping[str, Any], instances: Mapping[str, OpInstance]
) -> PipelineNode:
    instance = instances.get(node["op_instance_id"])
    if instance is None:
        raise CompileError(
            f"Pipeline node references op instance `{node['op_instance_id']}`, which "
            f"`{OPS_FILE}` does not define."
        )
    inputs = tuple(node["inputs"])
    outputs = tuple(node["outputs"])
    for role, wired, specs in (
        ("input", inputs, instance.inputs),
        ("output", outputs, instance.outputs),
    ):
        if len(wired) != len(specs):
            raise CompileError(
                f"Pipeline node `{instance.id}` is wired to {len(wired)} {role}(s), but "
                f"its op spec declares {len(specs)}."
            )
    return PipelineNode(instance=instance, inputs=inputs, outputs=outputs)


def _read_shape(shape: Sequence[Any], label: str) -> tuple[int | str, ...]:
    dims: list[int | str] = []
    for size in shape:
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            dims.append(size)
        elif isinstance(size, str):
            dims.append(size)
        else:
            raise CompileError(
                f"{label} has shape entry {size!r}, which is neither a non-negative "
                "dimension size nor a dimension name."
            )
    return tuple(dims)


def _element_type(dtype: str, label: str) -> int:
    match = _ARRAY_DTYPE.match(dtype)
    if match is None:
        raise CompileError(
            f"{label} has dtype `{dtype}`; the C compiler compiles only `Array[...]` "
            "tensors."
        )
    elem_type = ELEMENT_TYPES.get(match.group(1))
    if elem_type is None:
        raise CompileError(
            f"{label} has element type `{match.group(1)}`, which the C compiler does not "
            f"support; supported types are {', '.join(sorted(ELEMENT_TYPES))}."
        )
    return elem_type


def _reject_unknown_ops(ops: Any) -> None:
    """Dispatch every op instance through the node-compiler registry, before anything else
    reads `ops.json`: an op the C compiler has no compiler for is the more useful error."""
    if not isinstance(ops, list) or any(not isinstance(entry, dict) for entry in ops):
        raise CompileError(f"Bundle `{OPS_FILE}` must hold a list of op instances.")
    for instance in ops:
        if instance.get("op") not in NODE_COMPILERS:
            raise CompileError(
                f"Op instance `{instance.get('id')}` runs op `{instance.get('op')}`, "
                "which the C compiler has no node compiler for; it compiles "
                f"{', '.join(f'`{name}`' for name in NODE_COMPILERS)}."
            )


def _reject_dynamic_attributes(
    manifest: Mapping[str, Any],
    ops: Sequence[Mapping[str, Any]],
    variant_config: Mapping[str, Any],
) -> None:
    """Refuse a bundle that takes attribute values per call.

    Everything the artifact does is fixed at compile time, so a dynamic attribute cannot be
    honoured; compiling as if it were absent would silently ignore what the caller passes.
    """
    declared: list[tuple[str, Iterable[Any]]] = [
        ("the manifest", [entry["name"] for entry in manifest["dynamic_attributes"]])
    ]
    declared += [
        (f"op instance `{instance['id']}`", instance["dynamic_attributes"])
        for instance in ops
    ]
    declared += [
        (f"pipeline node `{node['op_instance_id']}`", node["extra_dynattrs"])
        for node in variant_config["nodes"]
    ]
    for owner, names in declared:
        listed = ", ".join(f"`{name}`" for name in sorted(names))
        if listed:
            raise CompileError(
                f"The C compiler does not support dynamic attributes, and {owner} "
                f"declares {listed}."
            )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CompileError(
            f"Could not read `{path.name}` from the bundle: {error}"
        ) from error


def _validate(validator: Callable[[Any], None], document: Any, filename: str) -> None:
    try:
        validator(document)
    except Exception as error:
        raise CompileError(f"Bundle `{filename}` is not valid: {error}") from error


def _unpack(source: Path) -> tuple[Path, bool]:
    try:
        directory, temporary = unpack_model(os.fspath(source))
    except Exception as error:
        raise CompileError(f"Could not open FNNX bundle `{source}`: {error}") from error
    return Path(directory), temporary


# --------------------------------------------------------------------------------------
# Node compilers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeCompilation:
    """What a node compiler is handed, and the scope it emits its definitions into.

    A compiler returns a `Program` whose inputs, outputs and body are the node's entry
    function — one pointer parameter per op-spec tensor, in spec order — and whose weights,
    scratch and functions are the definitions that entry needs. Pipeline codegen depends on
    nothing else, so a future FNNX op type plugs in here.
    """

    instance: OpInstance
    prefix: str
    scope: ArtifactScope
    dim_bindings: Mapping[str, int]
    runtime_dims: tuple[RuntimeDim, ...] = ()


NodeCompiler = Callable[[NodeCompilation], Program]


def compile_onnx_node(compilation: NodeCompilation) -> Program:
    """Compile an `ONNX_v1` node: its `model.onnx`, specialized to its op spec's shapes.

    Everything that can fail runs under one handler, so a failure anywhere — in the file
    layout, in the ONNX core, in the agreement between spec and graph — names the op
    instance the pipeline knows the node by.
    """
    instance = compilation.instance
    try:
        program = _compile_onnx_model(compilation)
    except CompileError as error:
        raise CompileError(f"Op instance `{instance.id}`: {error}") from error
    return program


def _compile_onnx_model(compilation: NodeCompilation) -> Program:
    instance = compilation.instance
    path = instance.artifact_dir / ONNX_MODEL_FILE
    if not path.is_file():
        raise CompileError(
            f"`{ONNX_MODEL_FILE}` is missing from `{instance.artifact_dir}`."
        )
    loaded = load_model(path)
    _declare_spec_types(loaded.model, instance)
    prepared = prepare_model(loaded, dim_bindings=compilation.dim_bindings)
    program = build_program(
        prepared,
        prefix=compilation.prefix,
        scope=compilation.scope,
        runtime_dims=compilation.runtime_dims,
    )
    _verify_signature(program, instance, compilation.dim_bindings)
    return program


NODE_COMPILERS: dict[str, NodeCompiler] = {ONNX_OP: compile_onnx_node}


def _declare_spec_types(model: ModelProto, instance: OpInstance) -> None:
    """Give the ONNX graph the I/O types the FNNX op spec declares.

    The spec is the contract the pipeline is wired on, and it is where the symbolic
    dimension names live — a converter usually leaves the graph's own batch dimension
    nameless — so it is what the graph gets specialized to. Whatever the graph states
    concretely has to agree with it.
    """
    drop_shadowed_inputs(model)
    _declare_side(model.graph.input, instance.inputs, "input")
    _declare_side(model.graph.output, instance.outputs, "output")


def _declare_side(
    entries: Sequence[ValueInfoProto], specs: Sequence[IOSpec], role: str
) -> None:
    if len(entries) != len(specs):
        raise CompileError(
            f"its op spec declares {len(specs)} {role}(s), but its ONNX graph has "
            f"{len(entries)}."
        )
    for index, (entry, spec) in enumerate(zip(entries, specs)):
        label = f"{role} {index} (`{entry.name}`)"
        _check_declared_type(entry.type, spec, label)
        entry.type.CopyFrom(
            helper.make_tensor_type_proto(spec.elem_type, list(spec.shape))
        )


def _check_declared_type(declared: TypeProto, spec: IOSpec, label: str) -> None:
    kind = declared.WhichOneof("value")
    if kind is not None and kind != "tensor_type":
        raise CompileError(
            f"{label} is a `{kind}` in the ONNX graph, which the C compiler does not "
            "support; only tensors can be compiled."
        )
    tensor_type = declared.tensor_type
    if tensor_type.elem_type and tensor_type.elem_type != spec.elem_type:
        raise CompileError(
            f"{label} is `{numpy_dtype_name(tensor_type.elem_type)}` in the ONNX graph, "
            f"but `{numpy_dtype_name(spec.elem_type)}` in the op spec."
        )
    if not tensor_type.HasField("shape"):
        return
    dims = tensor_type.shape.dim
    if len(dims) != len(spec.shape):
        raise CompileError(
            f"{label} has rank {len(dims)} in the ONNX graph, but rank "
            f"{len(spec.shape)} in the op spec."
        )
    for axis, (dim, size) in enumerate(zip(dims, spec.shape)):
        if dim.WhichOneof("value") != "dim_value" or not isinstance(size, int):
            continue
        if dim.dim_value != size:
            raise CompileError(
                f"{label} has size {dim.dim_value} on axis {axis} in the ONNX graph, but "
                f"{size} in the op spec."
            )


def _verify_signature(
    program: Program, instance: OpInstance, dim_bindings: Mapping[str, int]
) -> None:
    """Check the compiled entry against the op spec the pipeline wires the node by."""
    for role, tensors, specs in (
        ("input", program.inputs, instance.inputs),
        ("output", program.outputs, instance.outputs),
    ):
        if len(tensors) != len(specs):
            raise CompileError(
                f"its op spec declares {len(specs)} {role}(s), but the compiled graph "
                f"takes {len(tensors)}."
            )
        for index, (tensor, spec) in enumerate(zip(tensors, specs)):
            expected = spec.bind(dim_bindings)
            if (tensor.elem_type, tensor.shape) != (spec.elem_type, expected):
                raise CompileError(
                    f"{role} {index} (`{tensor.name}`) compiles to "
                    f"{numpy_dtype_name(tensor.elem_type)}{list(tensor.shape)}, but its "
                    f"op spec declares {numpy_dtype_name(spec.elem_type)}"
                    f"{list(expected)}."
                )


# --------------------------------------------------------------------------------------
# Pipeline codegen
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Edge:
    """Where a pipeline edge's data lives, as a C expression, plus the type it holds."""

    expr: str
    elem_type: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Port:
    """One end of an edge: a node's input or output, and the tensor it compiled to."""

    label: str
    tensor: IOTensor


@dataclass
class _PipelineBuilder:
    bundle: Bundle
    dim_bindings: Mapping[str, int]
    requested_prefix: str | None
    runtime_dims: tuple[RuntimeDim, ...] = ()

    prefix: str = field(init=False)
    scope: ArtifactScope = field(init=False)
    names: UniqueNames = field(default_factory=UniqueNames, init=False)
    produced: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        source = (
            self.requested_prefix
            if self.requested_prefix is not None
            else self.bundle.name
        )
        self.prefix = sanitize_identifier(source, fallback=DEFAULT_PREFIX)
        self.scope = ArtifactScope(self.prefix, self.names)
        reserve_dim_parameters(self.names, self.runtime_dims)

    def build(self) -> Program:
        # Handed out before any node is compiled, so that the pipeline entrypoint's
        # parameters read as the manifest names rather than as whatever a node claimed.
        parameters = [
            self.names.assign(tensor.name, fallback=role)
            for role, tensors in (
                ("input", self.bundle.inputs),
                ("output", self.bundle.outputs),
            )
            for tensor in tensors
        ]
        ordered = _order_nodes(self.bundle)
        programs = self._compile_nodes(ordered)
        entries = tuple(
            NodeEntry(
                id=instance_id,
                symbol=f"{program.prefix}_run",
                inputs=program.inputs,
                outputs=program.outputs,
                body=program.body,
                body_owners=program.body_owners,
            )
            for instance_id, program in programs.items()
        )

        ports = _collect_ports(ordered, programs)
        self.produced = {
            name: node.instance.id for node in ordered for name in node.outputs
        }
        split = len(self.bundle.inputs)
        inputs = self._pipeline_tensors(
            self.bundle.inputs, parameters[:split], "INPUT", ports
        )
        outputs = self._pipeline_tensors(
            self.bundle.outputs, parameters[split:], "OUTPUT", ports
        )
        edges, buffers = self._plan_edges(ordered, ports, (*inputs, *outputs))
        body, body_owners = self._emit_body(ordered, entries, edges, inputs)
        return Program(
            prefix=self.prefix,
            graph_name=self.bundle.label,
            source=f"FNNX bundle `{self.bundle.label}`",
            opsets=_merged_opsets(programs.values()),
            dim_bindings=_merged_bindings(programs.values()),
            inputs=inputs,
            outputs=outputs,
            weights=_merged_buffers(program.weights for program in programs.values()),
            scratch=_merged_buffers(
                [buffers, *(program.scratch for program in programs.values())]
            ),
            functions=_merged_functions(programs.values()),
            body=body,
            labels=tuple(
                table for program in programs.values() for table in program.labels
            ),
            nodes=entries,
            runtime_dims=self.runtime_dims,
            body_owners=body_owners,
        )

    def _compile_nodes(self, ordered: Sequence[PipelineNode]) -> dict[str, Program]:
        """Compile each op instance once, in the order the pipeline runs them.

        Two pipeline nodes may run the same op instance; they then share one entrypoint,
        called twice on different buffers.
        """
        programs: dict[str, Program] = {}
        for node in ordered:
            instance = node.instance
            if instance.id in programs:
                continue
            prefix = self.names.assign(
                f"{self.prefix}_node_{instance.id}", fallback=f"{self.prefix}_node"
            )
            programs[instance.id] = NODE_COMPILERS[instance.op](
                NodeCompilation(
                    instance=instance,
                    prefix=prefix,
                    scope=self.scope,
                    dim_bindings=self.dim_bindings,
                    runtime_dims=self.runtime_dims,
                )
            )
        return programs

    def _pipeline_tensors(
        self,
        declared: Sequence[ManifestTensor],
        parameters: Sequence[str],
        role: str,
        ports: Mapping[str, list[_Port]],
    ) -> tuple[IOTensor, ...]:
        """The manifest tensors as the artifact exposes them, sized from the nodes they
        are wired to and checked against whatever the manifest itself declares."""
        tensors = []
        for tensor, c_name in zip(declared, parameters):
            elem_type, shape = self._derived_type(tensor, role, ports)
            if elem_type != tensor.elem_type:
                raise CompileError(
                    f"Manifest {role.lower()} `{tensor.name}` is declared "
                    f"`Array[{numpy_dtype_name(tensor.elem_type)}]`, but the node it is "
                    f"wired to takes `Array[{numpy_dtype_name(elem_type)}]`."
                )
            if tensor.shape:
                expected = IOSpec(elem_type, tensor.shape).bind(self.dim_bindings)
                if expected != shape:
                    raise CompileError(
                        f"Manifest {role.lower()} `{tensor.name}` declares shape "
                        f"{list(expected)}, but the node it is wired to takes "
                        f"{list(shape)}."
                    )
            tensors.append(
                IOTensor(
                    name=tensor.name,
                    c_name=c_name,
                    macro=f"{self.prefix.upper()}_{role}_{c_name.upper()}",
                    elem_type=elem_type,
                    shape=shape,
                    owner=self.produced.get(tensor.name, ""),
                )
            )
        return tuple(tensors)

    def _derived_type(
        self, declared: ManifestTensor, role: str, ports: Mapping[str, list[_Port]]
    ) -> tuple[int, tuple[int, ...]]:
        # An output only counts as wired where a node *writes* it: one that merely appears
        # as some node's input is a name the caller's output buffer would take over, and
        # the node would then read the buffer it was supposed to be read into.
        if role == "OUTPUT" and declared.name not in self.produced:
            raise CompileError(
                f"Manifest output `{declared.name}` is produced by no pipeline node."
            )
        used = ports.get(declared.name)
        if used:
            tensor = _agreed_tensor(declared.name, used)
            return tensor.elem_type, tensor.shape
        if not declared.shape:
            raise CompileError(
                f"Manifest input `{declared.name}` is read by no pipeline node and "
                "declares no shape, so the buffer it needs cannot be sized."
            )
        return declared.elem_type, IOSpec(declared.elem_type, declared.shape).bind(
            self.dim_bindings
        )

    def _plan_edges(
        self,
        ordered: Sequence[PipelineNode],
        ports: Mapping[str, list[_Port]],
        exposed: Sequence[IOTensor],
    ) -> tuple[dict[str, _Edge], tuple[StaticBuffer, ...]]:
        """Give every edge a buffer: the caller's, or a static one of the pipeline's own.

        An edge the manifest exposes is the caller's buffer, written there by its producer
        and read from there by every downstream node, so a fan-out that is also a pipeline
        output needs no copy.
        """
        edges = {
            tensor.name: _Edge(tensor.c_name, tensor.elem_type, tensor.shape)
            for tensor in exposed
        }
        buffers = []
        for node in ordered:
            for name in node.outputs:
                if name in edges:
                    continue
                tensor = _agreed_tensor(name, ports[name])
                symbol = self.names.assign(
                    f"{self.prefix}_e_{name}", fallback=f"{self.prefix}_e"
                )
                edges[name] = _Edge(symbol, tensor.elem_type, tensor.shape)
                buffers.append(
                    StaticBuffer(name, symbol, tensor.elem_type, tensor.shape, None)
                )
        return edges, tuple(buffers)

    def _emit_body(
        self,
        ordered: Sequence[PipelineNode],
        entries: Sequence[NodeEntry],
        edges: Mapping[str, _Edge],
        inputs: Sequence[IOTensor],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        symbols = {entry.id: entry.symbol for entry in entries}
        read = {name for node in ordered for name in node.inputs}
        dims = [dim.c_name for dim in self.runtime_dims]
        body = [
            f"(void){tensor.c_name};" for tensor in inputs if tensor.name not in read
        ]
        owners = [""] * len(body)
        if ordered:
            body.append("int status;")
            owners.append("")
        for node in ordered:
            arguments = ", ".join(
                dims + [edges[name].expr for name in (*node.inputs, *node.outputs)]
            )
            body.append(f"status = {symbols[node.instance.id]}({arguments});")
            body.append(
                f"if (status != {self.prefix.upper()}_OK) {{\n    return status;\n}}"
            )
            owners += [node.instance.id] * 2
        return tuple(body), tuple(owners)


def _order_nodes(bundle: Bundle) -> tuple[PipelineNode, ...]:
    """Topological order of the pipeline nodes, stable in the order they are declared."""
    available = {tensor.name for tensor in bundle.inputs}
    producers: dict[str, PipelineNode] = {}
    for node in bundle.nodes:
        for name in node.outputs:
            if name in available:
                raise CompileError(
                    f"Pipeline node `{node.instance.id}` writes `{name}`, which is also a "
                    "manifest input."
                )
            if name in producers:
                raise CompileError(
                    f"Pipeline nodes `{producers[name].instance.id}` and "
                    f"`{node.instance.id}` both write `{name}`."
                )
            producers[name] = node
    for node in bundle.nodes:
        for name in node.inputs:
            if name not in available and name not in producers:
                raise CompileError(
                    f"Pipeline node `{node.instance.id}` reads `{name}`, which no "
                    "manifest input and no node produces."
                )

    remaining = list(bundle.nodes)
    ordered: list[PipelineNode] = []
    while remaining:
        for index, node in enumerate(remaining):
            if all(name in available for name in node.inputs):
                del remaining[index]
                ordered.append(node)
                available.update(node.outputs)
                break
        else:
            blocked = ", ".join(f"`{node.instance.id}`" for node in remaining)
            raise CompileError(
                f"The pipeline has a cycle: nodes {blocked} each wait on another's output."
            )
    return tuple(ordered)


def _collect_ports(
    ordered: Sequence[PipelineNode], programs: Mapping[str, Program]
) -> dict[str, list[_Port]]:
    """Every use of every edge, so that the ends of an edge can be checked against each
    other: one buffer cannot hold two shapes."""
    ports: dict[str, list[_Port]] = {}
    for node in ordered:
        program = programs[node.instance.id]
        for role, names, tensors in (
            ("input", node.inputs, program.inputs),
            ("output", node.outputs, program.outputs),
        ):
            for index, (name, tensor) in enumerate(zip(names, tensors)):
                label = f"{role} {index} of node `{node.instance.id}`"
                ports.setdefault(name, []).append(_Port(label, tensor))
    return ports


def _agreed_tensor(name: str, ports: Sequence[_Port]) -> IOTensor:
    first = ports[0]
    for other in ports[1:]:
        if (other.tensor.elem_type, other.tensor.shape) != (
            first.tensor.elem_type,
            first.tensor.shape,
        ):
            raise CompileError(
                f"Pipeline edge `{name}` is {numpy_dtype_name(first.tensor.elem_type)}"
                f"{list(first.tensor.shape)} as {first.label}, but "
                f"{numpy_dtype_name(other.tensor.elem_type)}"
                f"{list(other.tensor.shape)} as {other.label}."
            )
    return first.tensor


def _merged_opsets(programs: Iterable[Program]) -> dict[str, int]:
    """The opset each domain is compiled at, the highest winning where nodes differ.

    Nodes are compiled independently and may import different versions of a domain; the
    merged map is reported metadata, not something dispatch reads back.
    """
    merged: dict[str, int] = {}
    for program in programs:
        for domain, version in program.opsets.items():
            merged[domain] = max(merged.get(domain, version), version)
    return merged


def _merged_bindings(programs: Iterable[Program]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for program in programs:
        merged.update(program.dim_bindings)
    return merged


def _merged_functions(programs: Iterable[Program]) -> tuple[CFunction, ...]:
    """Every kernel the nodes emitted, each once: they are named from the artifact-wide
    prefix, so nodes using the same kernel at the same types share one definition."""
    merged: dict[str, CFunction] = {}
    for program in programs:
        for function in program.functions:
            existing = merged.setdefault(function.name, function)
            if existing.definition != function.definition:
                raise CompileError(
                    f"Kernel `{function.name}` was emitted twice with different "
                    "definitions; a kernel name must encode everything its code "
                    "depends on."
                )
    return tuple(merged.values())


def _merged_buffers(
    groups: Iterable[Sequence[StaticBuffer]],
) -> tuple[StaticBuffer, ...]:
    """Every static buffer the nodes reserved, each once, sized for the largest claim.

    Only kernel scratch is ever claimed twice — its symbol is shared between the nodes that
    call the kernel — and sharing it stays safe under the artifact's one-call-at-a-time
    contract, exactly as it is between the nodes of a single graph.
    """
    merged: dict[str, StaticBuffer] = {}
    for group in groups:
        for buffer in group:
            reserved = merged.get(buffer.symbol)
            if reserved is None or reserved.elem_count < buffer.elem_count:
                merged[buffer.symbol] = buffer
    return tuple(merged.values())
