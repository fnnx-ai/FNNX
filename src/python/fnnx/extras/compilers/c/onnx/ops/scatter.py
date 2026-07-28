"""The ops that write into a copy of a tensor at positions decided at run time.

ScatterElements, ScatterND and TensorScatter all do the same shape of work: the result is the
first operand copied, and then some of its elements are overwritten out of a second operand,
at positions a third names. The positions are values rather than shapes, so the addressing is
a loop; what stays static is the result's shape, which is the copied operand's own. Each op
is emitted as a `memcpy` of the whole operand followed by one kernel walking the updates.

An index comes from the caller, so every one of them is normalized the way ONNX defines it —
a negative index counted back from the end of the axis — and then bounds checked: a kernel
returns nonzero for an index outside its axis and the entrypoint passes that on as an
argument error, rather than writing past a buffer.

`reduction` says what an update does to the element already in the result, and ONNX's two
families disagree about one case of it. `ScatterElements` is defined by a reference that
folds with Python's own `max`/`min`, which keep the value already in the result whenever a
comparison against a NaN comes out false; `ScatterND`'s folds with `np.maximum`/`np.minimum`,
which propagate a NaN from either side. Both op documents say only "max" and "min", so each
kernel follows the reference implementation of its own op — the only thing that states what
these two do with a NaN at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from string import Template

from onnx import TensorProto

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    copy_tensor,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    kernel_name,
    normalize_axis,
    row_major_strides,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import combiner, extents

# Only the newest revision of ScatterElements and ScatterND is claimed: it is the one the
# reference evaluator is version-faithful for and the one every corpus test of them imports,
# so it is the only one anything can vouch for. Both arrived at 11, took `reduction` with
# `add` and `mul` at 16 and gained `max` and `min` at 18. TensorScatter has had a single
# revision since it arrived at 24. A model importing an older one gets the
# unsupported-version error.
_SCATTER_ELEMENTS_VERSIONS = (18,)
_SCATTER_ND_VERSIONS = (18,)
_TENSOR_SCATTER_VERSIONS = (24,)

# Scatter, which ONNX deprecated in favour of ScatterElements, is that op before `reduction`
# was added to it, so the same generator serves it. 9 is the revision the corpus's own tests
# select — they import opset 10 — and 11 is the deprecating revision, which changed nothing
# else about the op; `test_the_two_scatter_revisions_are_one_op` compares the two schemas
# rather than taking that on trust.
_SCATTER_VERSIONS = (9, 11)

# The opset each `reduction` value arrived at, by the revision that added it.
_REDUCTION_VERSIONS = ((16, ("add", "mul")), (18, ("max", "min")))

_TENSOR_SCATTER_MODES = ("linear", "circular")


@dataclass(frozen=True)
class _Fold:
    """How an update is combined with the element already in the result."""

    expression: str
    helpers: tuple[CFunction, ...] = ()


_SCATTER_ELEMENTS_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* updates,
    const $index* indices,
    size_t count,
    int rank,
    const size_t* shape,
    const size_t* strides,
    int axis,
    size_t extent)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        ptrdiff_t position = (ptrdiff_t)indices[index];
        size_t remainder = index;
        size_t offset = 0;
        int walked;
        for (walked = rank - 1; walked >= 0; --walked) {
            const size_t coordinate = remainder % shape[walked];
            remainder /= shape[walked];
            if (walked != axis) {
                offset += coordinate * strides[walked];
            }
        }
        if (position < 0) {
            position += (ptrdiff_t)extent;
        }
        if (position < 0 || position >= (ptrdiff_t)extent) {
            return 1;
        }
        offset += (size_t)position * strides[axis];
        out[offset] = $fold;
    }
    return 0;
}""")

_SCATTER_ND_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* updates,
    const $index* indices,
    size_t rows,
    size_t depth,
    size_t slice_size,
    const size_t* extents,
    const size_t* strides)
{
    size_t row, level, element;
    for (row = 0; row < rows; ++row) {
        size_t offset = 0;
        for (level = 0; level < depth; ++level) {
            ptrdiff_t position = (ptrdiff_t)indices[row * depth + level];
            if (position < 0) {
                position += (ptrdiff_t)extents[level];
            }
            if (position < 0 || position >= (ptrdiff_t)extents[level]) {
                return 1;
            }
            offset += (size_t)position * strides[level];
        }
        for (element = 0; element < slice_size; ++element) {
            out[offset + element] = $fold;
        }
    }
    return 0;
}""")

# The write index is validated once per sample rather than per step: with it inside the axis,
# no `written_at + step` the loop forms can leave `ptrdiff_t`, and running off the end of the
# axis part-way through the update is what the per-step check catches.
_TENSOR_SCATTER_LINEAR_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* update,$index_parameters
    size_t prefix_count,
    size_t sequence_length,
    size_t max_sequence_length,
    size_t block)
{
    size_t prefix, step;
    for (prefix = 0; prefix < prefix_count; ++prefix) {
        const ptrdiff_t written_at = $written_at;
        if (written_at < 0 || written_at >= (ptrdiff_t)max_sequence_length) {
            return 1;
        }
        for (step = 0; step < sequence_length; ++step) {
            const size_t position = (size_t)written_at + step;
            if (position >= max_sequence_length) {
                return 1;
            }
            memcpy(
                out + (prefix * max_sequence_length + position) * block,
                update + (prefix * sequence_length + step) * block,
                block * sizeof(*out));
        }
    }
    return 0;
}""")

# ONNX defines the circular mode by taking the whole cache coordinate modulo the sequence
# capacity — `np.mod(np.asarray(cache_idx), max_sequence_length)` in the op's own pseudocode,
# which is what its reference implementation runs — so the coordinates before the sequence
# axis wrap along with the write index. Hence the destination is recomposed from the wrapped
# coordinates rather than being the sample's own base offset. Wrapping only ever lowers a
# coordinate, so it always lands inside the axis it addresses and nothing is checked here.
_TENSOR_SCATTER_CIRCULAR_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* update,$index_parameters
    size_t prefix_count,
    size_t sequence_length,
    size_t max_sequence_length,
    size_t block,
    int prefix_rank,
    const size_t* prefix_shape,
    const size_t* prefix_strides)
{
    size_t prefix, step;
    for (prefix = 0; prefix < prefix_count; ++prefix) {
        const ptrdiff_t written_at = $written_at;
        const ptrdiff_t capacity = (ptrdiff_t)max_sequence_length;
        const size_t start = (size_t)(((written_at % capacity) + capacity) % capacity);
        size_t remainder = prefix;
        size_t base = 0;
        int walked;
        for (walked = prefix_rank - 1; walked >= 0; --walked) {
            const size_t coordinate = remainder % prefix_shape[walked];
            remainder /= prefix_shape[walked];
            base += (coordinate % max_sequence_length) * prefix_strides[walked];
        }
        for (step = 0; step < sequence_length; ++step) {
            memcpy(
                out + base + ((start + step) % max_sequence_length) * block,
                update + (prefix * sequence_length + step) * block,
                block * sizeof(*out));
        }
    }
}""")


def _scatter_elements(context: NodeContext) -> NodeEmission:
    """ScatterElements: one element written per update, at the index's own coordinates."""
    data = context.require_input(0)
    indices = context.require_input(1)
    updates = context.require_input(2)
    result = context.require_output(0)
    rank = len(data.shape)
    verify_shape(context, result, data.shape)
    if rank == 0 or len(indices.shape) != rank:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` writes into `{data.name}` "
            f"of rank {rank} through `{indices.name}` of rank {len(indices.shape)}; ONNX "
            "defines the two as having the same rank, and at least one axis."
        )
    if updates.shape != indices.shape:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes one update per index, "
            f"but `{updates.name}` has shape {list(updates.shape)} and `{indices.name}` "
            f"has shape {list(indices.shape)}."
        )
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    for other in range(rank):
        if other != axis and indices.shape[other] > data.shape[other]:
            raise CompileError(
                f"Node `{context.label}`: `{context.node.op_type}` writes along axis "
                f"{axis}, so `{indices.name}` of shape {list(indices.shape)} addresses "
                f"`{data.name}` of shape {list(data.shape)} on every other axis, and "
                f"reaches past it on axis {other}."
            )

    statements = list(copy_tensor(data, result).statements)
    if updates.elem_count == 0:
        return NodeEmission(functions=(), statements=tuple(statements))

    reduction = _reduction(context)
    fold = _fold(
        context,
        reduction,
        data.elem_type,
        current="out[offset]",
        update="updates[index]",
        numpy_extremum=False,
    )
    name = kernel_name(
        context, reduction, c_type(data.elem_type), c_type(indices.elem_type)
    )
    statements.append(
        checked_call(
            context,
            name,
            [
                result.expr,
                updates.expr,
                indices.expr,
                f"{updates.elem_count}u",
                str(rank),
                extents(indices.shape),
                extents(row_major_strides(data.shape)),
                str(axis),
                f"{data.shape[axis]}u",
            ],
        )
    )
    return NodeEmission(
        functions=(
            *fold.helpers,
            CFunction(
                name,
                _SCATTER_ELEMENTS_TEMPLATE.substitute(
                    name=name,
                    element=c_type(data.elem_type),
                    index=c_type(indices.elem_type),
                    fold=fold.expression,
                ),
            ),
        ),
        statements=tuple(statements),
    )


def _scatter_nd(context: NodeContext) -> NodeEmission:
    """ScatterND: a slice written per index tuple, into the axes the tuple names."""
    data = context.require_input(0)
    indices = context.require_input(1)
    updates = context.require_input(2)
    result = context.require_output(0)
    rank = len(data.shape)
    verify_shape(context, result, data.shape)
    if not indices.shape:
        raise CompileError(
            f"Node `{context.label}`: `ScatterND` takes its index tuples from the last "
            f"axis of `{indices.name}`, which is a scalar and has none."
        )
    depth = indices.shape[-1]
    if depth > rank:
        raise CompileError(
            f"Node `{context.label}`: `ScatterND` addresses {depth} dimension(s) of "
            f"`{data.name}`, which has {rank}."
        )
    expected = (*indices.shape[:-1], *data.shape[depth:])
    if updates.shape != expected:
        raise CompileError(
            f"Node `{context.label}`: `ScatterND` writes one slice per index tuple, so "
            f"`{updates.name}` has shape {list(expected)}; it has "
            f"{list(updates.shape)}."
        )

    statements = list(copy_tensor(data, result).statements)
    if updates.elem_count == 0:
        return NodeEmission(functions=(), statements=tuple(statements))

    slice_size = math.prod(data.shape[depth:])
    reduction = _reduction(context)
    fold = _fold(
        context,
        reduction,
        data.elem_type,
        current="out[offset + element]",
        update="updates[row * slice_size + element]",
        numpy_extremum=True,
    )
    name = kernel_name(
        context, reduction, c_type(data.elem_type), c_type(indices.elem_type)
    )
    statements.append(
        checked_call(
            context,
            name,
            [
                result.expr,
                updates.expr,
                indices.expr,
                f"{math.prod(indices.shape[:-1])}u",
                f"{depth}u",
                f"{slice_size}u",
                extents(data.shape[:depth]),
                extents(row_major_strides(data.shape)[:depth]),
            ],
        )
    )
    return NodeEmission(
        functions=(
            *fold.helpers,
            CFunction(
                name,
                _SCATTER_ND_TEMPLATE.substitute(
                    name=name,
                    element=c_type(data.elem_type),
                    index=c_type(indices.elem_type),
                    fold=fold.expression,
                ),
            ),
        ),
        statements=tuple(statements),
    )


def _tensor_scatter(context: NodeContext) -> NodeEmission:
    """TensorScatter: the update written into each sample's cache at that sample's index."""
    cache = context.require_input(0)
    update = context.require_input(1)
    written_at = context.optional_input(2)
    result = context.require_output(0)
    rank = len(cache.shape)
    verify_shape(context, result, cache.shape)
    axis = _sequence_axis(context, rank)
    if len(update.shape) != rank or any(
        extent != cached
        for position, (extent, cached) in enumerate(zip(update.shape, cache.shape))
        if position != axis
    ):
        raise CompileError(
            f"Node `{context.label}`: `TensorScatter` writes `{update.name}` of shape "
            f"{list(update.shape)} into `{cache.name}` of shape {list(cache.shape)}; ONNX "
            f"defines the two as differing on axis {axis} alone."
        )
    if update.shape[axis] > cache.shape[axis]:
        raise CompileError(
            f"Node `{context.label}`: `TensorScatter` writes {update.shape[axis]} "
            f"position(s) into axis {axis} of `{cache.name}`, which holds "
            f"{cache.shape[axis]}."
        )
    if written_at is not None and written_at.shape != (cache.shape[0],):
        raise CompileError(
            f"Node `{context.label}`: `TensorScatter` reads one write index per sample of "
            f"the batch, so `{written_at.name}` has shape {[cache.shape[0]]}; it has "
            f"{list(written_at.shape)}."
        )

    statements = list(copy_tensor(cache, result).statements)
    if update.elem_count == 0:
        return NodeEmission(functions=(), statements=tuple(statements))

    indexed = written_at is not None
    circular = _mode(context) == "circular"
    prefix_shape = cache.shape[:axis]
    block = math.prod(cache.shape[axis + 1 :])
    arguments = [result.expr, update.expr]
    if written_at is not None:
        arguments += [written_at.expr, f"{math.prod(prefix_shape[1:])}u"]
    arguments += [
        f"{math.prod(prefix_shape)}u",
        f"{update.shape[axis]}u",
        f"{cache.shape[axis]}u",
        f"{block}u",
    ]
    if circular:
        arguments += [
            str(axis),
            extents(prefix_shape),
            extents(row_major_strides(cache.shape)[:axis]),
        ]
    name = kernel_name(
        context,
        "circular" if circular else "linear",
        "indexed" if indexed else "appended",
        c_type(cache.elem_type),
    )
    template = (
        _TENSOR_SCATTER_CIRCULAR_TEMPLATE
        if circular
        else _TENSOR_SCATTER_LINEAR_TEMPLATE
    )
    definition = template.substitute(
        name=name,
        element=c_type(cache.elem_type),
        index_parameters=(
            f"\n    const {c_type(written_at.elem_type)}* write_indices,"
            "\n    size_t batch_stride,"
            if written_at is not None
            else ""
        ),
        written_at=(
            "(ptrdiff_t)write_indices[prefix / batch_stride]" if indexed else "0"
        ),
    )
    statements.append(
        call_kernel(name, arguments)
        if circular
        else checked_call(context, name, arguments)
    )
    return NodeEmission(
        functions=(CFunction(name, definition),), statements=tuple(statements)
    )


def _fold(
    context: NodeContext,
    reduction: str,
    elem_type: int,
    *,
    current: str,
    update: str,
    numpy_extremum: bool,
) -> _Fold:
    """The C expression writing `update` over `current`, as `reduction` combines the two.

    `numpy_extremum` selects between the two readings of `max` and `min` the ONNX reference
    implementations of these ops carry — see the module docstring.
    """
    if reduction == "none":
        return _Fold(update)
    if reduction == "add":
        # numpy adds two booleans as their disjunction, which is what the reference folds
        # with; a `+` on the byte a boolean is emitted as would leave a 2 behind.
        operator = "|" if elem_type == TensorProto.BOOL else "+"
        return _Fold(f"{current} {operator} {update}")
    if reduction == "mul":
        return _Fold(f"{current} * {update}")
    largest = reduction == "max"
    if numpy_extremum:
        helper = combiner(context, elem_type, largest=largest)
        return _Fold(f"{helper.name}({current}, {update})", (helper,))
    comparison = ">" if largest else "<"
    return _Fold(f"({update} {comparison} {current}) ? {update} : {current}")


def _reduction(context: NodeContext) -> str:
    """How the node folds an update into the result, of the values its revision defines."""
    allowed = ["none"]
    for version, added in _REDUCTION_VERSIONS:
        if context.since_version >= version:
            allowed += added
    value = context.attribute("reduction", b"none")
    reduction = value.decode() if isinstance(value, bytes) else str(value)
    if reduction not in allowed:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reduction `{reduction}` is "
            f"not one of the reductions ONNX defines at opset version "
            f"{context.since_version} ({', '.join(allowed)})."
        )
    return reduction


def _mode(context: NodeContext) -> str:
    value = context.attribute("mode", b"linear")
    mode = value.decode() if isinstance(value, bytes) else str(value)
    if mode not in _TENSOR_SCATTER_MODES:
        raise CompileError(
            f"Node `{context.label}`: `TensorScatter` mode `{mode}` is not one of the "
            f"modes ONNX defines ({', '.join(_TENSOR_SCATTER_MODES)})."
        )
    return mode


def _sequence_axis(context: NodeContext, rank: int) -> int:
    """The axis TensorScatter writes along, which is never the batch it reads indices by."""
    axis = normalize_axis(context, context.int_attribute("axis"), rank)
    if axis == 0:
        raise CompileError(
            f"Node `{context.label}`: `TensorScatter` writes along axis 0 of "
            f"`{context.require_input(0).name}`, which is the batch it takes one write "
            "index per sample of; ONNX defines the sequence axis as a later one."
        )
    return axis


register_kernel("", "Scatter", _SCATTER_VERSIONS, _scatter_elements)
register_kernel("", "ScatterElements", _SCATTER_ELEMENTS_VERSIONS, _scatter_elements)
register_kernel("", "ScatterND", _SCATTER_ND_VERSIONS, _scatter_nd)
register_kernel("", "TensorScatter", _TENSOR_SCATTER_VERSIONS, _tensor_scatter)
