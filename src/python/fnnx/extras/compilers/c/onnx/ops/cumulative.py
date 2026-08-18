"""CumSum and CumProd: the running fold along one axis.

The axis is an operand rather than an attribute, and models — the ONNX backend corpus among
them — do pass it at run time. It decides which elements the scan visits but nothing about
any shape, so a run-time axis still compiles to fully static code: one call site per axis the
operand's rank allows, chosen by a switch, with an out-of-range value returning the argument
error the status enum exists for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from string import Template

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import c_type, numpy_dtype_name
from fnnx.extras.compilers.c.onnx.emit import INVALID_ARGUMENT_STATUS
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    GROUP_PARAMETERS,
    call_kernel,
    group_axes,
    kernel_name,
    normalize_axis,
    offset_helper,
    verify_same_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import expand

# CumSum-14 only added bfloat16 to CumSum-11's type constraints; CumProd arrived at 26.
_CUM_SUM_VERSIONS = (11, 14)
_CUM_PROD_VERSIONS = (26,)

# `exclusive` and `reverse` are parameters rather than four kernels: the call site passes
# literals, so a C compiler folds the branches away and the artifact still carries one scan.
_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters,
    int exclusive,
    int reverse)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element total = $identity;
        for (index = 0; index < group_size; ++index) {
            const size_t position = base + $offset(
                reverse ? group_size - 1 - index : index,
                reduced_rank, reduced_shape, reduced_strides);
            const $element x = in[position];
            if (exclusive) {
                out[position] = total;
            }
            total = ($element)($combine);
            if (!exclusive) {
                out[position] = total;
            }
        }
    }
}""")


def _cumulative(context: NodeContext, *, identity: str, combine: str) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    axis_operand = context.require_input(1)
    verify_same_shape(context, source, result)
    rank = len(source.shape)
    if rank == 0:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` scans along an axis, but "
            f"`{source.name}` is a scalar and has none."
        )
    if axis_operand.elem_count != 1:
        raise CompileError(
            f"Node `{context.label}`: the axis of `{context.node.op_type}` comes from "
            f"`{axis_operand.name}`, which holds {axis_operand.elem_count} values; ONNX "
            "defines it as a single one."
        )

    offset = offset_helper(context.prefix)
    name = kernel_name(context, numpy_dtype_name(result.elem_type))
    definition = _TEMPLATE.substitute(
        name=name,
        element=c_type(result.elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        identity=expand(identity, result.elem_type),
        combine=expand(combine, result.elem_type),
    )

    def call(axis: int) -> str:
        grouping = group_axes(source.shape, (axis,))
        return call_kernel(
            name,
            [
                result.expr,
                source.expr,
                *grouping.arguments,
                str(context.int_attribute("exclusive")),
                str(context.int_attribute("reverse")),
            ],
        )

    fixed = context.constant_input(1)
    if fixed is not None:
        return NodeEmission(
            functions=(offset, CFunction(name, definition)),
            statements=(
                call(normalize_axis(context, int(fixed.reshape(-1)[0]), rank)),
            ),
        )
    normalize = _normalize_helper(context.prefix)
    return NodeEmission(
        functions=(offset, normalize, CFunction(name, definition)),
        statements=(_dispatch(context, axis_operand.expr, rank, call, normalize),),
    )


def _normalize_helper(prefix: str) -> CFunction:
    """An axis counted from the end, resolved against a rank, both known only as values."""
    name = f"{prefix}_normalized_axis"
    return CFunction(
        name,
        "\n".join(
            [
                f"static int64_t {name}(int64_t axis, int64_t rank)",
                "{",
                "    return (axis < 0) ? (axis + rank) : axis;",
                "}",
            ]
        ),
    )


def _dispatch(
    context: NodeContext,
    operand: str,
    rank: int,
    call: Callable[[int], str],
    normalize: CFunction,
) -> str:
    """The scan for whichever axis the operand names at run time, or an argument error.

    The axis is resolved through a function rather than into a local, so that the statement
    introduces no identifier of its own — one would shadow the entrypoint parameter a tensor
    of the same name is emitted as.
    """
    cases = []
    for axis in range(rank):
        cases.append(f"case {axis}:")
        cases.extend(_indented(call(axis), "    "))
        cases.append("    break;")
    return "\n".join(
        [
            f"switch ({normalize.name}((int64_t){operand}[0], {rank})) {{",
            *cases,
            "default:",
            f"    return {context.prefix.upper()}_{INVALID_ARGUMENT_STATUS};",
            "}",
        ]
    )


def _indented(statement: str, indent: str) -> Sequence[str]:
    return [f"{indent}{line}" if line else "" for line in statement.splitlines()]


register_kernel(
    "",
    "CumSum",
    _CUM_SUM_VERSIONS,
    partial(_cumulative, identity="$zero", combine="total + x"),
)
register_kernel(
    "",
    "CumProd",
    _CUM_PROD_VERSIONS,
    partial(_cumulative, identity="$one", combine="total * x"),
)
