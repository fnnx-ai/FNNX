"""The ONNX-ML preprocessing ops: the feature transforms a fitted pipeline is made of.

They differ from the standard domain in where their parameters live: a scaler's offsets, an
encoder's categories and a label encoder's key/value pairs arrive as *attributes* rather than
as operands, so each one is emitted as `static const` data the shared kernel reads through a
pointer — which keeps one kernel per (op, element types) however many nodes run it, and keeps
the tables in the artifact's reported footprint rather than on the stack.

Every one of these ops is at revision 1 of its schema and has been since ONNX-ML opset 1,
except `LabelEncoder`, whose revision 4 is the only one the reference evaluator implements
faithfully — the older two are left to the standard unsupported-version error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from string import Template

import numpy as np
from onnx import TensorProto
from onnx.numpy_helper import to_array

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    FLOAT_TYPES,
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    NodeContext,
    NodeEmission,
    TensorRef,
    constant_data,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.loader import ML_DOMAIN
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    kernel_name,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import Scalar, pointwise

_INITIAL_VERSIONS = (1,)
_LABEL_ENCODER_VERSIONS = (4,)

# What `Normalizer`'s reference implementation divides by when a row's norm underflows to
# zero, so that an all-zero row comes out as itself rather than as NaN. It is not in the
# op's prose, but it is what the executable form of the specification computes.
_NORM_FLOOR = "1e-30f"

# The two lines each norm contributes to the row loop: how a value joins the running total,
# and what turns that total into the divisor. The largest magnitude is taken the way numpy's
# own `max` takes it, so one NaN in a row carries through to every element of it.
_NORMS = {
    "MAX": (
        "            const float magnitude = fabsf(value);\n"
        "            if (magnitude > norm || isnan(magnitude)) {\n"
        "                norm = magnitude;\n"
        "            }",
        "",
    ),
    "L1": ("            norm += fabsf(value);", ""),
    "L2": ("            norm += value * value;", "        norm = sqrtf(norm);\n"),
}

# What `OneHotEncoder` does about a value in no category, by whether `zeros` is set: leave
# the row at zero, or report the failure the schema prescribes through the status enum.
_ONE_HOT_MISS = "        if (chosen < 0) {\n            return 1;\n        }\n"

_SCALER_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    const float* offset,
    const float* scale,
    size_t count,
    size_t channels,
    size_t offset_stride,
    size_t scale_stride)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        const size_t channel = index % channels;
        out[index] = ((float)in[index] - offset[channel * offset_stride])
            * scale[channel * scale_stride];
    }
}""")

_NORMALIZER_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    size_t rows,
    size_t columns)
{
    size_t row, column;
    for (row = 0; row < rows; ++row) {
        const size_t base = row * columns;
        float norm = 0.0f;
        for (column = 0; column < columns; ++column) {
            const float value = (float)in[base + column];
$accumulate
        }
$finish        if (norm < $floor) {
            norm = $floor;
        }
        for (column = 0; column < columns; ++column) {
            out[base + column] = (float)in[base + column] / norm;
        }
    }
}""")

_IMPUTER_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $element* imputed,
    size_t count,
    size_t channels,
    size_t imputed_stride$marker)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        const $element value = in[index];
        out[index] = ($test)
            ? imputed[(index % channels) * imputed_stride]
            : value;
    }
}""")

_ONE_HOT_TEMPLATE = Template("""\
static $status $name(
    float* out,
    const $element* in,
    const $element* categories,
    size_t count,
    size_t category_count)
{
    size_t index, category;
    for (index = 0; index < count; ++index) {
        const $element value = in[index];
        ptrdiff_t chosen = -1;
        for (category = 0; category < category_count; ++category) {
            if (value == categories[category]) {
                chosen = (ptrdiff_t)category;
            }
        }
$missing        for (category = 0; category < category_count; ++category) {
            out[index * category_count + category] =
                (chosen == (ptrdiff_t)category) ? 1.0f : 0.0f;
        }
    }
$result}""")

_LABEL_ENCODER_TEMPLATE = Template("""\
static void $name(
    $result* out,
    const $element* in,
    const $element* keys,
    const $result* values,
    size_t count,
    size_t pair_count,
    $result fallback)
{
    size_t index, pair;
    for (index = 0; index < count; ++index) {
        const $element value = in[index];
        $result mapped = fallback;
        for (pair = 0; pair < pair_count; ++pair) {
            if ($match) {
                mapped = values[pair];
            }
        }
        out[index] = mapped;
    }
}""")

_EXTRACTOR_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const int64_t* indices,
    size_t rows,
    size_t extent,
    size_t index_count)
{
    size_t row, chosen;
    for (row = 0; row < rows; ++row) {
        for (chosen = 0; chosen < index_count; ++chosen) {
            ptrdiff_t position = (ptrdiff_t)indices[chosen];
            if (position < 0) {
                position += (ptrdiff_t)extent;
            }
            if (position < 0 || position >= (ptrdiff_t)extent) {
                return 1;
            }
            out[row * index_count + chosen] = in[row * extent + (size_t)position];
        }
    }
    return 0;
}""")

_VECTORIZER_TEMPLATE = Template("""\
static void $name(
    float* out,
    const $element* in,
    size_t rows,
    size_t columns,
    size_t taken,
    size_t width,
    size_t stride,
    size_t offset)
{
    size_t row, column;
    for (row = 0; row < rows; ++row) {
        for (column = 0; column < width; ++column) {
            out[row * stride + offset + column] =
                (column < taken) ? (float)in[row * columns + column] : 0.0f;
        }
    }
}""")


def _scaler(context: NodeContext) -> NodeEmission:
    """`Y = (X - offset) * scale`, per feature, as the float32 the schema declares `Y`.

    Both coefficient lists are either one value shared by every feature or one per feature,
    which is what their stride at the call site says.
    """
    source = context.require_input(0)
    result = _float_result(context, source.shape)
    channels = _trailing_extent(source.shape)
    offset = _coefficients(context, "offset", channels)
    scale = _coefficients(context, "scale", channels)

    name = kernel_name(context, c_type(source.elem_type))
    definition = _SCALER_TEMPLATE.substitute(
        name=name, element=c_type(source.elem_type)
    )
    offset_data, offset_symbol = constant_data(context, "offset", offset)
    scale_data, scale_symbol = constant_data(context, "scale", scale)
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            offset_symbol,
            scale_symbol,
            f"{result.elem_count}u",
            f"{max(1, channels)}u",
            f"{int(len(offset) > 1)}u",
            f"{int(len(scale) > 1)}u",
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(offset_data, scale_data),
    )


def _normalizer(context: NodeContext) -> NodeEmission:
    """Each row divided by its own norm: the largest magnitude in it, its L1 or its L2."""
    source = context.require_input(0)
    result = _float_result(context, source.shape)
    if not source.shape:
        raise CompileError(
            f"Node `{context.label}`: Normalizer runs along the last axis of a `[C]` or "
            "`[N,C]` tensor, and its input is a scalar."
        )
    declared = context.attribute("norm", b"MAX")
    norm = declared.decode() if isinstance(declared, bytes) else str(declared)
    accumulate, finish = _NORMS.get(norm, (None, None))
    if accumulate is None:
        raise CompileError(
            f"Node `{context.label}`: `norm` is `{norm}`, which is none of the modes ONNX "
            f"defines ({', '.join(sorted(_NORMS))})."
        )

    columns = source.shape[-1]
    name = kernel_name(context, norm.lower(), c_type(source.elem_type))
    definition = _NORMALIZER_TEMPLATE.substitute(
        name=name,
        element=c_type(source.elem_type),
        accumulate=accumulate,
        finish=finish,
        floor=_NORM_FLOOR,
    )
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            f"{math.prod(source.shape[:-1])}u",
            f"{columns}u",
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _imputer(context: NodeContext) -> NodeEmission:
    """Every element equal to the marked value replaced by this feature's imputed one.

    ONNX carries the pair as two attribute families, one for the floating-point element
    types and one for the integer ones; the float family is the one in force whenever it is
    set, which is also how the reference evaluator picks between them. A marker that is NaN
    is a test for NaN rather than a comparison, since nothing compares equal to it.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    verify_shape(context, result, source.shape)
    if result.elem_type != source.elem_type:
        raise CompileError(
            f"Node `{context.label}`: Imputer leaves the element type alone, but its "
            f"input is `{element_type_name(source.elem_type)}` and its output "
            f"`{element_type_name(result.elem_type)}`."
        )

    floats = [float(value) for value in context.attribute("imputed_value_floats", [])]
    integers = [int(value) for value in context.attribute("imputed_value_int64s", [])]
    if bool(floats) == bool(integers):
        raise CompileError(
            f"Node `{context.label}`: Imputer must set exactly one of "
            "`imputed_value_floats` and `imputed_value_int64s`."
        )
    channels = _trailing_extent(source.shape)
    values = np.array(floats or integers).astype(numpy_dtype_name(source.elem_type))
    if len(values) not in (1, channels):
        raise CompileError(
            f"Node `{context.label}`: Imputer was given {len(values)} imputed value(s) "
            f"for a tensor whose last axis holds {channels}; it takes either one value "
            "for every feature or a single value for all of them."
        )

    # The float family compares in double, which is what promoting an integer tensor to the
    # attribute's own type does; the integer family compares in the tensor's own type, where
    # a value too large for a double to hold exactly still has to match itself.
    if floats:
        marker, compare = context.float_attribute("replaced_value_float"), "double"
    else:
        marker, compare = (
            context.int_attribute("replaced_value_int64"),
            c_type(source.elem_type),
        )
    # Nothing compares equal to NaN, so a NaN marker is a test for it — and the marker then
    # stops being a value the kernel is handed at all.
    matches_nan = compare == "double" and math.isnan(marker)
    test = "isnan((double)value)" if matches_nan else f"({compare})value == replaced"
    name = kernel_name(
        context, "nan" if matches_nan else compare, c_type(source.elem_type)
    )
    definition = _IMPUTER_TEMPLATE.substitute(
        name=name,
        element=c_type(source.elem_type),
        marker="" if matches_nan else f",\n    {compare} replaced",
        test=test,
    )
    data, symbol = constant_data(context, "imputed", values)
    arguments = [
        result.expr,
        source.expr,
        symbol,
        f"{result.elem_count}u",
        f"{max(1, channels)}u",
        f"{int(len(values) > 1)}u",
    ]
    if not matches_nan:
        arguments.append(
            scalar_literal(
                marker, TensorProto.DOUBLE if compare == "double" else source.elem_type
            )
        )
    call = call_kernel(name, arguments)
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(data,),
    )


def _binarizer(context: NodeContext) -> NodeEmission:
    """One where the value is strictly above the threshold, zero elsewhere.

    The comparison runs in double because that is what numpy's own promotion of the
    tensor against the threshold — a value of the attribute's floating-point type — does.
    """
    source = context.require_input(0)
    verify_shape(context, context.require_output(0), source.shape)
    return pointwise(
        context,
        "($element)((double)x0 > threshold)",
        scalars=(
            Scalar(
                "threshold", TensorProto.DOUBLE, context.float_attribute("threshold")
            ),
        ),
    )


def _one_hot_encoder(context: NodeContext) -> NodeEmission:
    """A row of indicators per element, one column per category, all zero for a miss.

    A repeated category is the last of its occurrences, which is the entry a mapping built
    from the list in order ends up holding. With `zeros` cleared, a value in no category is
    the failure the op's schema prescribes, reported through the status enum.
    """
    source = context.require_input(0)
    categories = [int(value) for value in context.attribute("cats_int64s", [])]
    if not categories:
        raise CompileError(
            f"Node `{context.label}`: OneHotEncoder sets no `cats_int64s`; a numeric input "
            "is encoded against the integer category list, `cats_strings` being reachable "
            "only from the string input this compiler does not support."
        )
    result = _float_result(context, (*source.shape, len(categories)))

    zeros = bool(context.int_attribute("zeros"))
    name = kernel_name(
        context, "zeros" if zeros else "strict", c_type(source.elem_type)
    )
    definition = _ONE_HOT_TEMPLATE.substitute(
        name=name,
        element=c_type(source.elem_type),
        status="void" if zeros else "int",
        missing="" if zeros else _ONE_HOT_MISS,
        result="" if zeros else "    return 0;\n",
    )
    data, symbol = constant_data(
        context,
        "categories",
        np.array(categories).astype(numpy_dtype_name(source.elem_type)),
    )
    arguments = [
        result.expr,
        source.expr,
        symbol,
        f"{source.elem_count}u",
        f"{len(categories)}u",
    ]
    call = (
        call_kernel(name, arguments)
        if zeros
        else checked_call(context, name, arguments)
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(data,),
    )


def _label_encoder(context: NodeContext) -> NodeEmission:
    """Each element looked up in the key list and replaced by the value beside it.

    A repeated key takes its last occurrence, as the op's schema states in as many words —
    which is why the loop runs to the end rather than stopping at the first hit. The same
    paragraph makes a NaN key match any NaN input, so a floating-point key list tests for
    that as well as for equality.
    """
    source = context.require_input(0)
    result = context.require_output(0)
    verify_shape(context, result, source.shape)
    keys, values, fallback = _label_pairs(context, source, result)

    floating = source.elem_type in FLOAT_TYPES
    match = (
        "value == keys[pair] || (isnan((double)value) && isnan((double)keys[pair]))"
        if floating
        else "value == keys[pair]"
    )
    name = kernel_name(context, c_type(source.elem_type), c_type(result.elem_type))
    definition = _LABEL_ENCODER_TEMPLATE.substitute(
        name=name,
        element=c_type(source.elem_type),
        result=c_type(result.elem_type),
        match=match,
    )
    key_data, key_symbol = constant_data(context, "keys", keys)
    value_data, value_symbol = constant_data(context, "values", values)
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            key_symbol,
            value_symbol,
            f"{result.elem_count}u",
            f"{len(keys)}u",
            scalar_literal(fallback, result.elem_type),
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=(key_data, value_data),
    )


def _array_feature_extractor(context: NodeContext) -> NodeEmission:
    """The columns the index operand names, taken from every row of the last axis.

    The indices come from the caller, so each one is normalized the way numpy's own
    indexing — which the reference evaluator uses — defines it, and then bounds checked.
    """
    source = context.require_input(0)
    indices = context.require_input(1)
    result = context.require_output(0)
    if not source.shape:
        raise CompileError(
            f"Node `{context.label}`: ArrayFeatureExtractor selects along the last axis of "
            "its input, and its input is a scalar."
        )
    if indices.elem_type != TensorProto.INT64:
        raise CompileError(
            f"Node `{context.label}`: ArrayFeatureExtractor takes `int64` indices, not "
            f"`{element_type_name(indices.elem_type)}`."
        )
    rows = math.prod(source.shape[:-1])
    verify_shape(
        context,
        result,
        (1, indices.elem_count)
        if len(source.shape) == 1
        else (*source.shape[:-1], indices.elem_count),
    )

    name = kernel_name(context, c_type(source.elem_type))
    definition = _EXTRACTOR_TEMPLATE.substitute(
        name=name, element=c_type(source.elem_type)
    )
    call = checked_call(
        context,
        name,
        [
            result.expr,
            source.expr,
            indices.expr,
            f"{rows}u",
            f"{source.shape[-1]}u",
            f"{indices.elem_count}u",
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _feature_vectorizer(context: NodeContext) -> NodeEmission:
    """Every input cut or zero-padded to its declared width, laid side by side as float32."""
    sources = [context.require_input(index) for index in range(len(context.node.input))]
    result = _float_result(context, None)
    widths = [int(value) for value in context.attribute("inputdimensions", [])]
    if len(widths) != len(sources):
        raise CompileError(
            f"Node `{context.label}`: FeatureVectorizer declares {len(widths)} entry(s) in "
            f"`inputdimensions` for {len(sources)} input(s); it takes one width per input."
        )
    rows, columns = _vectorizer_layout(context, sources)
    verify_shape(context, result, (rows, sum(widths)))

    functions: list[CFunction] = []
    statements: list[str] = []
    offset = 0
    for source, width, count in zip(sources, widths, columns):
        name = kernel_name(context, c_type(source.elem_type))
        functions.append(
            CFunction(
                name,
                _VECTORIZER_TEMPLATE.substitute(
                    name=name, element=c_type(source.elem_type)
                ),
            )
        )
        statements.append(
            call_kernel(
                name,
                [
                    result.expr,
                    source.expr,
                    f"{rows}u",
                    f"{count}u",
                    f"{min(count, width)}u",
                    f"{width}u",
                    f"{sum(widths)}u",
                    f"{offset}u",
                ],
            )
        )
        offset += width
    return NodeEmission(functions=tuple(functions), statements=tuple(statements))


def _vectorizer_layout(
    context: NodeContext, sources: Sequence[TensorRef]
) -> tuple[int, tuple[int, ...]]:
    """The shared row count and each input's own column count.

    A vector input is read as the single column a matrix of one column would be, which is
    the shape the reference evaluator gives it before concatenating.
    """
    layouts = []
    for source in sources:
        if len(source.shape) not in (1, 2):
            raise CompileError(
                f"Node `{context.label}`: FeatureVectorizer takes inputs of rank 1 or 2, "
                f"but `{source.name}` has shape {list(source.shape)}."
            )
        layouts.append(
            (source.shape[0], source.shape[1] if len(source.shape) == 2 else 1)
        )
    rows = {row for row, _ in layouts}
    if len(rows) != 1:
        raise CompileError(
            f"Node `{context.label}`: FeatureVectorizer lays its inputs side by side, so "
            f"they must agree on their first dimension; these hold {sorted(rows)} rows."
        )
    return layouts[0][0], tuple(count for _, count in layouts)


def _label_pairs(
    context: NodeContext, source: TensorRef, result: TensorRef
) -> tuple[np.ndarray, np.ndarray, float | int]:
    """`LabelEncoder`'s key/value tables and its default, in the attribute family in force.

    ONNX carries the mapping in one of three families and the schema picks whichever is set,
    in the order the reference evaluator reads them; the pairs stop at the shorter of the two
    lists. The string family never reaches here — ONNX's own type inference refuses a key type
    that differs from the input's, and a string result is refused by the static verification —
    so it is the numeric families or nothing.
    """
    keys = _label_table(context, "keys", source.elem_type)
    values = _label_table(context, "values", result.elem_type)
    if keys is None or values is None:
        raise CompileError(
            f"Node `{context.label}`: LabelEncoder sets no numeric `keys_*`/`values_*` "
            f"pair to map `{element_type_name(source.elem_type)}` through."
        )
    pairs = min(len(keys), len(values))
    return keys[:pairs], values[:pairs], _label_default(context, result.elem_type)


def _label_table(context: NodeContext, role: str, elem_type: int) -> np.ndarray | None:
    """The `role` list of the first attribute family the node sets, at `elem_type`."""
    for family in (f"{role}_floats", f"{role}_int64s"):
        values = context.attribute(family, [])
        if values:
            return np.array(list(values)).astype(numpy_dtype_name(elem_type))
    tensor = context.attribute(f"{role}_tensor", None)
    if tensor is not None:
        return to_array(tensor).reshape(-1).astype(numpy_dtype_name(elem_type))
    return None


def _label_default(context: NodeContext, elem_type: int) -> float | int:
    """The value an element matching no key takes, from the family the values came from."""
    tensor = context.attribute("default_tensor", None)
    if tensor is not None:
        return to_array(tensor).reshape(-1).tolist()[0]
    if context.attribute("values_floats", []):
        return context.float_attribute("default_float")
    if context.attribute("values_int64s", []):
        return context.int_attribute("default_int64")
    # `values_tensor` with no `default_tensor`: the schema documents the default as -1 for an
    # integral value type and -0 for a floating-point one, which are the two `default_*`
    # attributes' own declared defaults.
    return (
        context.float_attribute("default_float")
        if elem_type in FLOAT_TYPES
        else context.int_attribute("default_int64")
    )


def _coefficients(context: NodeContext, name: str, channels: int) -> np.ndarray:
    """A `Scaler` coefficient list: one value per feature, or a single shared one."""
    values = [float(value) for value in context.attribute(name, [])]
    if not values:
        raise CompileError(
            f"Node `{context.label}`: Scaler sets no `{name}`; it takes one value per "
            "feature, or a single value for all of them."
        )
    if len(values) not in (1, channels):
        raise CompileError(
            f"Node `{context.label}`: Scaler was given {len(values)} `{name}` value(s) for "
            f"a tensor whose last axis holds {channels}; it takes either one value for "
            "every feature or a single value for all of them."
        )
    return np.array(values, dtype=np.float32)


def _float_result(context: NodeContext, shape: tuple[int, ...] | None) -> TensorRef:
    """The node's result, checked to be the float32 tensor its ONNX schema declares."""
    result = context.require_output(0)
    if result.elem_type != TensorProto.FLOAT:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` produces a `float` tensor, "
            f"but its output `{result.name}` is declared "
            f"`{element_type_name(result.elem_type)}`."
        )
    if shape is not None:
        verify_shape(context, result, shape)
    return result


def _trailing_extent(shape: tuple[int, ...]) -> int:
    """How many features a tensor holds per row: its last axis, or one for a scalar."""
    return shape[-1] if shape else 1


register_kernel(ML_DOMAIN, "Scaler", _INITIAL_VERSIONS, _scaler)
register_kernel(ML_DOMAIN, "Normalizer", _INITIAL_VERSIONS, _normalizer)
register_kernel(ML_DOMAIN, "Imputer", _INITIAL_VERSIONS, _imputer)
register_kernel(ML_DOMAIN, "Binarizer", _INITIAL_VERSIONS, _binarizer)
register_kernel(ML_DOMAIN, "OneHotEncoder", _INITIAL_VERSIONS, _one_hot_encoder)
register_kernel(ML_DOMAIN, "LabelEncoder", _LABEL_ENCODER_VERSIONS, _label_encoder)
register_kernel(
    ML_DOMAIN, "ArrayFeatureExtractor", _INITIAL_VERSIONS, _array_feature_extractor
)
register_kernel(ML_DOMAIN, "FeatureVectorizer", _INITIAL_VERSIONS, _feature_vectorizer)
