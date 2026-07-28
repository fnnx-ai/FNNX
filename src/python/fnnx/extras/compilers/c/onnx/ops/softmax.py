"""Softmax, LogSoftmax and Hardmax: the normalizations that run along one axis.

Each writes a group in place of itself — the result carries the operand's shape — so one
grouping addresses both buffers. ONNX revised all three at opset 13: up to then they
flattened every axis from `axis` on into a single one, and nothing can vouch for those
revisions (the reference evaluator applies the current semantics to them, and the backend
corpus has no test at an older opset), so only the current revision is served and an older
import gets the standard unsupported-version error.
"""

from __future__ import annotations

from functools import partial
from string import Template

from fnnx.extras.compilers.c.onnx.dtypes import c_type, numpy_dtype_name
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
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
from fnnx.extras.compilers.c.onnx.ops.broadcast import combiner, expand, math_suffix
from fnnx.extras.compilers.c.onnx.ops.reduce import extremum_test

_VERSIONS = (13,)

# The group's largest element is subtracted from every exponent, so nothing overflows and
# the result is unchanged; the reference evaluator normalizes the same way, and LogSoftmax
# is the logarithm of what Softmax computes rather than a formula of its own — which is what
# ONNX defines it as, down to where the underflow to zero puts an infinity.
_SOFTMAX_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element largest = -INFINITY;
        $element total = $zero;
        for (index = 0; index < group_size; ++index) {
            largest = ($element)$maximum(largest, in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)]);
        }
        for (index = 0; index < group_size; ++index) {
            total += exp$f(in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)]
                - largest);
        }
        for (index = 0; index < group_size; ++index) {
            const size_t position = base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides);
            out[position] = $result;
        }
    }
}""")

_HARDMAX_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
$parameters)
{
    size_t group, index;
    for (group = 0; group < group_count; ++group) {
        const size_t base = $offset(group, kept_rank, kept_shape, kept_strides);
        $element best = $zero;
        int64_t chosen = 0;
        for (index = 0; index < group_size; ++index) {
            const $element x = in[base
                + $offset(index, reduced_rank, reduced_shape, reduced_strides)];
            if (index == 0 || ($better)) {
                best = x;
                chosen = (int64_t)index;
            }
        }
        for (index = 0; index < group_size; ++index) {
            out[base + $offset(index, reduced_rank, reduced_shape, reduced_strides)] =
                (index == (size_t)chosen) ? $one : $zero;
        }
    }
}""")

_NORMALIZED = "exp$f(in[position] - largest) / total"


def _softmax(context: NodeContext, *, logarithmic: bool) -> NodeEmission:
    elem_type = context.require_output(0).elem_type
    largest = combiner(context, elem_type, largest=True)
    offset = offset_helper(context.prefix)
    name = kernel_name(context, numpy_dtype_name(elem_type))
    result = f"log$f({_NORMALIZED})" if logarithmic else _NORMALIZED
    definition = _SOFTMAX_TEMPLATE.substitute(
        name=name,
        element=c_type(elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        maximum=largest.name,
        zero=scalar_literal(0, elem_type),
        f=math_suffix(elem_type),
        result=expand(result, elem_type),
    )
    return _emit(context, CFunction(name, definition), (offset, largest))


def _hardmax(context: NodeContext) -> NodeEmission:
    """One where the group's largest element is, zero everywhere else, ties going first."""
    elem_type = context.require_output(0).elem_type
    offset = offset_helper(context.prefix)
    name = kernel_name(context, numpy_dtype_name(elem_type))
    definition = _HARDMAX_TEMPLATE.substitute(
        name=name,
        element=c_type(elem_type),
        parameters=GROUP_PARAMETERS,
        offset=offset.name,
        zero=scalar_literal(0, elem_type),
        one=scalar_literal(1, elem_type),
        better=extremum_test(elem_type, largest=True, last=False),
    )
    return _emit(context, CFunction(name, definition), (offset,))


def _emit(
    context: NodeContext, kernel: CFunction, helpers: tuple[CFunction, ...]
) -> NodeEmission:
    source = context.require_input(0)
    result = context.require_output(0)
    verify_same_shape(context, source, result)
    axis = normalize_axis(context, context.int_attribute("axis"), len(source.shape))
    grouping = group_axes(source.shape, (axis,))
    return NodeEmission(
        functions=(*helpers, kernel),
        statements=(
            call_kernel(kernel.name, [result.expr, source.expr, *grouping.arguments]),
        ),
    )


register_kernel("", "Softmax", _VERSIONS, partial(_softmax, logarithmic=False))
register_kernel("", "LogSoftmax", _VERSIONS, partial(_softmax, logarithmic=True))
register_kernel("", "Hardmax", _VERSIONS, _hardmax)
