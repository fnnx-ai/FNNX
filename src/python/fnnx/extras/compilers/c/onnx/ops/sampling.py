"""The samplers: the ops that read an image at positions that fall between its elements.

`GridSample` is the general one — every output position carries its own sampling coordinate,
normalized to [-1, 1] — and the other two are it at fixed geometries. `RoiAlign` samples a
regular grid inside each region of interest and folds what it reads into one value per bin;
`MaxRoiPool`, its predecessor, rounds each region to whole elements instead and takes the
largest in each bin, so it interpolates nothing. `AffineGrid` computes no samples at all: it
builds the coordinates a `GridSample` is then fed, which is why it sits here.

Three things are shared, and emitted once per artifact whatever mixture of the four a model
holds: the reflection that folds a coordinate back between two borders, the resolution of one
tap's index under the padding mode, and the cubic weights. Everything else is per element
type, with the geometry reaching the kernels as call-site literals.

A coordinate arrives at run time, so every conversion of one into an index is bounded first:
a value far outside the operand — or one that is not a number at all — is pulled to where the
sampling reads nothing rather than converted to whatever an out-of-range cast would give.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from string import Template

import numpy as np
from onnx import TensorProto

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
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.ops.axes import (
    call_kernel,
    checked_call,
    verify_shape,
)
from fnnx.extras.compilers.c.onnx.ops.broadcast import extents, math_suffix

# GridSample arrived at 16, 20 generalized it past two spatial axes and renamed its modes,
# and 22 widened the element types; RoiAlign arrived at 10, 16 added the coordinate
# transformation and 22 widened the types; MaxRoiPool has 1 and 22. Only the newest revision
# of each is claimed: it is the one the reference evaluator is version-faithful for and the
# one every corpus test of these ops imports, so it is the only one anything can vouch for.
# A model importing an older one gets the unsupported-version error. AffineGrid has had a
# single revision since it arrived.
_VERSIONS = (22,)
_AFFINE_GRID_VERSIONS = (20,)

# The C values the kernels switch on, in the order their branches are written.
_MODES = {"nearest": 0, "linear": 1, "cubic": 2}
_PADDING_MODES = {"zeros": 0, "border": 1, "reflection": 2}
_ROI_TRANSFORMS = {"half_pixel": 0, "output_half_pixel": 1}
_ROI_MODES = {"avg": 0, "max": 1}

# How far a coordinate read at run time may reach before a kernel refuses it. Comfortably
# past any extent an artifact can hold, and small enough that the products the geometry
# forms out of it stay inside a 64-bit `ptrdiff_t`.
_COORDINATE_LIMIT = 1.0e9

_REFLECT_TEMPLATE = Template("""\
static double $name(double value, double lower, double upper)
{
    double range = upper - lower;
    double excess, periods, remainder;
    /* A border with no room between its ends -- one element, measured corner to corner --
       leaves the single position it names. */
    if (!(range > 0.0)) {
        return lower;
    }
    if (value < lower) {
        excess = lower - value;
    } else if (value > upper) {
        excess = value - upper;
    } else {
        return value;
    }
    /* Every whole range crossed flips the direction the excess is measured in; the count is
       kept in floating point so that a coordinate of any magnitude folds without a
       conversion that would not be defined for it. */
    periods = floor(excess / range);
    remainder = excess - periods * range;
    if (fmod(periods, 2.0) != 0.0) {
        return (value < lower) ? upper - remainder : lower + remainder;
    }
    return (value < lower) ? lower + remainder : upper - remainder;
}""")

# Where one tap lands, once the padding mode has had its say. -1 is the answer `zeros` gives
# for a tap outside the operand: it reads nothing and contributes nothing.
_INDEX_TEMPLATE = Template("""\
static ptrdiff_t $name(
    ptrdiff_t index,
    size_t extent,
    int padding_mode,
    int align_corners)
{
    /* An axis with no elements has nothing to clamp or reflect onto, whatever the mode. */
    if (extent == 0) {
        return -1;
    }
    if (padding_mode == $zeros) {
        return (index >= 0 && index < (ptrdiff_t)extent) ? index : -1;
    }
    if (padding_mode == $reflection) {
        index = (ptrdiff_t)$reflect(
            (double)index,
            align_corners ? 0.0 : -0.5,
            align_corners ? (double)extent - 1.0 : (double)extent - 0.5);
    }
    /* What `border` does, and what a reflected index needs anyway: the borders it folds
       between reach half an element past the operand when the corners are not aligned. */
    if (index < 0) {
        return 0;
    }
    return (index >= (ptrdiff_t)extent) ? (ptrdiff_t)extent - 1 : index;
}""")

# The sampling coordinate one normalized grid value names, in the operand's own units.
_LOCATE_TEMPLATE = Template("""\
static $coord $name(
    $coord value,
    size_t extent,
    int align_corners,
    int padding_mode,
    int nearest)
{
    const double lower = align_corners ? 0.0 : -0.5;
    const double upper =
        align_corners ? (double)extent - 1.0 : (double)extent - 0.5;
    const $coord reach = ($coord)extent + ($coord)8;
    /* [-1, 1] spans the whole axis either corner to corner or edge to edge. */
    $coord x = align_corners
        ? (value + $one) / ($coord)2 * (($coord)extent - $one)
        : ((value + $one) * ($coord)extent - $one) / ($coord)2;
    if (nearest) {
        x = rint$f(x);
    }
    if ((double)x < lower || (double)x > upper) {
        if (padding_mode == $border) {
            x = (x < $zero) ? $zero : x;
            x = (x > ($coord)extent - $one) ? ($coord)extent - $one : x;
        } else if (padding_mode == $reflection) {
            x = ($coord)$reflect((double)x, lower, upper);
        }
    }
    /* `zeros` leaves the coordinate wherever it fell. Every tap around one this far out is
       outside the operand whatever its exact value, so it is pulled to where that is still
       true and the conversion to an index is defined -- as is a coordinate that is not a
       number, which no comparison above holds for. */
    if (!(x > -reach)) {
        return -reach;
    }
    return (x < reach) ? x : reach;
}""")

# Keys' cubic convolution at a = -0.75, which GridSample fixes where Resize takes it as an
# attribute. The taps sit at -1, 0, 1 and 2 around the coordinate's floor.
_COEFFICIENT_TEMPLATE = Template("""\
static $coord $name(int mode, $coord ratio, int tap)
{
    $coord x;
    if (mode == $linear) {
        return (tap == 0) ? $one - ratio : ratio;
    }
    switch (tap) {
    case 0:
        x = ratio + $one;
        return ((($coord)-0.75 * x + ($coord)3.75) * x - ($coord)6) * x + ($coord)3;
    case 1:
        return (($coord)1.25 * ratio - ($coord)2.25) * ratio * ratio + $one;
    case 2:
        x = $one - ratio;
        return (($coord)1.25 * x - ($coord)2.25) * x * x + $one;
    default:
        x = ($coord)2 - ratio;
        return ((($coord)-0.75 * x + ($coord)3.75) * x - ($coord)6) * x + ($coord)3;
    }
}""")

# The parameters both GridSample kernels take after their buffers.
_GRID_PARAMETERS = """\
    size_t batch_count,
    size_t channels,
    size_t input_size,
    size_t output_size,
    int spatial_rank,
    const size_t* input_shape,
    int padding_mode,
    int align_corners"""

# Reading the grid at one output position: the coordinates are stored in the reverse of the
# operand's axis order, which is what the innermost subscript undoes.
_GRID_COORDINATE = """\
                        const size_t extent = input_shape[axis];
                        const $coord x = $locate(
                            coordinates[position * (size_t)spatial_rank
                                + (size_t)(spatial_rank - 1 - axis)],
                            extent,
                            align_corners,
                            padding_mode,
                            $nearest);"""

_GRID_NEAREST_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $coord* grid,
$parameters)
{
    size_t batch, channel, position;
    int axis;
    for (batch = 0; batch < batch_count; ++batch) {
        const $coord* coordinates =
            grid + batch * output_size * (size_t)spatial_rank;
        for (channel = 0; channel < channels; ++channel) {
            const $element* plane = in + (batch * channels + channel) * input_size;
            $element* result = out + (batch * channels + channel) * output_size;
            for (position = 0; position < output_size; ++position) {
                size_t offset = 0;
                size_t stride = 1;
                int outside = 0;
                for (axis = spatial_rank - 1; axis >= 0; --axis) {
$coordinate
                    const ptrdiff_t resolved =
                        $index((ptrdiff_t)x, extent, padding_mode, align_corners);
                    if (resolved < 0) {
                        outside = 1;
                    } else {
                        offset += (size_t)resolved * stride;
                    }
                    stride *= extent;
                }
                result[position] = outside ? $zero : plane[offset];
            }
        }
    }
}""")

_GRID_INTERPOLATE_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* in,
    const $coord* grid,
$parameters,
    int mode)
{
    const int taps_per_axis = (mode == $linear) ? 2 : 4;
    size_t batch, channel, position, tap, tap_count = 1;
    int axis;
    for (axis = 0; axis < spatial_rank; ++axis) {
        tap_count *= (size_t)taps_per_axis;
    }
    for (batch = 0; batch < batch_count; ++batch) {
        const $coord* coordinates =
            grid + batch * output_size * (size_t)spatial_rank;
        for (channel = 0; channel < channels; ++channel) {
            const $element* plane = in + (batch * channels + channel) * input_size;
            $element* result = out + (batch * channels + channel) * output_size;
            for (position = 0; position < output_size; ++position) {
                $element total = $zero;
                for (tap = 0; tap < tap_count; ++tap) {
                    size_t remaining = tap;
                    size_t offset = 0;
                    size_t stride = 1;
                    $coord weight = $unit;
                    int outside = 0;
                    for (axis = spatial_rank - 1; axis >= 0; --axis) {
$coordinate
                        const $coord base = floor$f(x);
                        const int within = (int)(remaining % (size_t)taps_per_axis);
                        const ptrdiff_t resolved = $index(
                            (ptrdiff_t)base + within - ((taps_per_axis == 4) ? 1 : 0),
                            extent,
                            padding_mode,
                            align_corners);
                        remaining /= (size_t)taps_per_axis;
                        weight *= $coefficient(mode, x - base, within);
                        if (resolved < 0) {
                            outside = 1;
                        } else {
                            offset += (size_t)resolved * stride;
                        }
                        stride *= extent;
                    }
                    if (!outside) {
                        total += ($element)weight * plane[offset];
                    }
                }
                result[position] = total;
            }
        }
    }
}""")

_AFFINE_COORDINATE_TEMPLATE = Template("""\
static double $name(size_t index, size_t extent, int align_corners)
{
    double step;
    if (align_corners) {
        /* One position covers the whole axis, so there is no step to take along it. */
        if (extent < 2) {
            return -1.0;
        }
        return -1.0 + (double)index * (2.0 / (double)(extent - 1));
    }
    step = 2.0 / (double)extent;
    return (-1.0 + step / 2.0) + (double)index * step;
}""")

_AFFINE_GRID_TEMPLATE = Template("""\
static void $name(
    $element* out,
    const $element* theta,
    size_t batch_count,
    size_t output_size,
    int spatial_rank,
    const size_t* spatial_shape,
    int align_corners)
{
    size_t batch, position;
    int axis, row;
    for (batch = 0; batch < batch_count; ++batch) {
        for (position = 0; position < output_size; ++position) {
            for (row = 0; row < spatial_rank; ++row) {
                const $element* weights =
                    theta + ((batch * (size_t)spatial_rank) + (size_t)row)
                        * (size_t)(spatial_rank + 1);
                size_t remainder = position;
                double sum = 0.0;
                /* The homogeneous coordinate runs the axes backwards -- the last spatial
                   axis first -- and closes with the constant one. */
                for (axis = spatial_rank - 1; axis >= 0; --axis) {
                    const size_t extent = spatial_shape[axis];
                    const size_t index = remainder % extent;
                    remainder /= extent;
                    sum += (double)weights[spatial_rank - 1 - axis]
                         * $coordinate(index, extent, align_corners);
                }
                sum += (double)weights[spatial_rank];
                out[(batch * output_size + position) * (size_t)spatial_rank + (size_t)row] =
                    ($element)sum;
            }
        }
    }
}""")

_ROI_ALIGN_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $element* rois,
    const int64_t* batch_indices,
    size_t roi_count,
    size_t batch_count,
    size_t channels,
    size_t height,
    size_t width,
    size_t pooled_height,
    size_t pooled_width,
    int sampling_ratio,
    int half_pixel,
    int max_mode,
    $element spatial_scale)
{
    const size_t plane_size = height * width;
    size_t roi, channel, row, column;
    for (roi = 0; roi < roi_count; ++roi) {
        const $element* box = rois + roi * 4;
        const int64_t batch = batch_indices[roi];
        const $element shift = half_pixel ? ($element)0.5 : $zero;
        const $element start_column = box[0] * spatial_scale - shift;
        const $element start_row = box[1] * spatial_scale - shift;
        $element box_width = box[2] * spatial_scale - shift - start_column;
        $element box_height = box[3] * spatial_scale - shift - start_row;
        $element bin_width, bin_height, count;
        ptrdiff_t grid_rows, grid_columns, down, right;
        if (batch < 0 || (uint64_t)batch >= (uint64_t)batch_count) {
            return 1;
        }
        if (!half_pixel) {
            /* A region that measures less than one element is widened to one. */
            box_width = (box_width < $one) ? $one : box_width;
            box_height = (box_height < $one) ? $one : box_height;
        }
        bin_height = box_height / ($element)pooled_height;
        bin_width = box_width / ($element)pooled_width;
        if (sampling_ratio > 0) {
            grid_rows = sampling_ratio;
            grid_columns = sampling_ratio;
        } else {
            /* Enough samples per bin to cover every element it spans. */
            const double rows = ceil((double)bin_height);
            const double columns = ceil((double)bin_width);
            if (!(rows >= -$limit && rows <= $limit)
                    || !(columns >= -$limit && columns <= $limit)) {
                return 1;
            }
            grid_rows = (ptrdiff_t)rows;
            grid_columns = (ptrdiff_t)columns;
        }
        count = ($element)((grid_rows * grid_columns > 1) ? grid_rows * grid_columns : 1);
        for (channel = 0; channel < channels; ++channel) {
            const $element* plane =
                in + ((size_t)batch * channels + channel) * plane_size;
            $element* result =
                out + (roi * channels + channel) * pooled_height * pooled_width;
            for (row = 0; row < pooled_height; ++row) {
                for (column = 0; column < pooled_width; ++column) {
                    $element total = $zero;
                    int seen = 0;
                    for (down = 0; down < grid_rows; ++down) {
                        const $element y = start_row + ($element)row * bin_height
                            + (($element)down + ($element)0.5) * bin_height
                                / ($element)grid_rows;
                        for (right = 0; right < grid_columns; ++right) {
                            const $element x = start_column + ($element)column * bin_width
                                + (($element)right + ($element)0.5) * bin_width
                                    / ($element)grid_columns;
                            ptrdiff_t low_row = 0, high_row = 0;
                            ptrdiff_t low_column = 0, high_column = 0;
                            $element w1 = $zero, w2 = $zero, w3 = $zero, w4 = $zero;
                            $element p1, p2, p3, p4;
                            /* A sample more than one element outside the feature map reads
                               nothing; so does one that is not a number at all. */
                            if (y >= -$one && y <= ($element)height
                                    && x >= -$one && x <= ($element)width) {
                                $element sampled_row = (y < $zero) ? $zero : y;
                                $element sampled_column = (x < $zero) ? $zero : x;
                                $element ly, lx;
                                low_row = (ptrdiff_t)sampled_row;
                                low_column = (ptrdiff_t)sampled_column;
                                if (low_row >= (ptrdiff_t)height - 1) {
                                    high_row = low_row = (ptrdiff_t)height - 1;
                                    sampled_row = ($element)low_row;
                                } else {
                                    high_row = low_row + 1;
                                }
                                if (low_column >= (ptrdiff_t)width - 1) {
                                    high_column = low_column = (ptrdiff_t)width - 1;
                                    sampled_column = ($element)low_column;
                                } else {
                                    high_column = low_column + 1;
                                }
                                ly = sampled_row - ($element)low_row;
                                lx = sampled_column - ($element)low_column;
                                w1 = ($one - ly) * ($one - lx);
                                w2 = ($one - ly) * lx;
                                w3 = ly * ($one - lx);
                                w4 = ly * lx;
                            }
                            p1 = w1 * plane[(size_t)low_row * width + (size_t)low_column];
                            p2 = w2 * plane[(size_t)low_row * width + (size_t)high_column];
                            p3 = w3 * plane[(size_t)high_row * width + (size_t)low_column];
                            p4 = w4 * plane[(size_t)high_row * width + (size_t)high_column];
                            if (max_mode) {
                                $element best = p1;
                                best = (p2 > best) ? p2 : best;
                                best = (p3 > best) ? p3 : best;
                                best = (p4 > best) ? p4 : best;
                                if (!seen || best > total) {
                                    total = best;
                                    seen = 1;
                                }
                            } else {
                                total += p1 + p2 + p3 + p4;
                            }
                        }
                    }
                    result[row * pooled_width + column] =
                        max_mode ? total : total / count;
                }
            }
        }
    }
    return 0;
}""")

_MAX_ROI_POOL_TEMPLATE = Template("""\
static int $name(
    $element* out,
    const $element* in,
    const $element* rois,
    size_t roi_count,
    size_t batch_count,
    size_t channels,
    size_t height,
    size_t width,
    size_t pooled_height,
    size_t pooled_width,
    $element spatial_scale)
{
    const size_t plane_size = height * width;
    size_t roi, channel, row, column;
    for (roi = 0; roi < roi_count; ++roi) {
        const $element* box = rois + roi * 5;
        const double batch = (double)box[0];
        double corners[4];
        ptrdiff_t start_column, start_row, box_width, box_height;
        $element bin_height, bin_width;
        int corner;
        if (!(batch >= 0.0 && batch < (double)batch_count)) {
            return 1;
        }
        /* The region is rounded to whole elements before anything is pooled, which is what
           separates this op from the RoiAlign that replaced it. */
        for (corner = 0; corner < 4; ++corner) {
            corners[corner] = (double)round$f(box[corner + 1] * spatial_scale);
            if (!(corners[corner] >= -$limit && corners[corner] <= $limit)) {
                return 1;
            }
        }
        start_column = (ptrdiff_t)corners[0];
        start_row = (ptrdiff_t)corners[1];
        box_width = (ptrdiff_t)corners[2] - start_column + 1;
        box_height = (ptrdiff_t)corners[3] - start_row + 1;
        box_width = (box_width > 1) ? box_width : 1;
        box_height = (box_height > 1) ? box_height : 1;
        bin_height = ($element)box_height / ($element)pooled_height;
        bin_width = ($element)box_width / ($element)pooled_width;
        for (channel = 0; channel < channels; ++channel) {
            const $element* plane =
                in + ((size_t)batch * channels + channel) * plane_size;
            $element* result =
                out + (roi * channels + channel) * pooled_height * pooled_width;
            for (row = 0; row < pooled_height; ++row) {
                for (column = 0; column < pooled_width; ++column) {
                    const ptrdiff_t first_row = $clamp(
                        (ptrdiff_t)floor$f(($element)row * bin_height) + start_row,
                        (ptrdiff_t)height);
                    const ptrdiff_t last_row = $clamp(
                        (ptrdiff_t)ceil$f(($element)(row + 1) * bin_height) + start_row,
                        (ptrdiff_t)height);
                    const ptrdiff_t first_column = $clamp(
                        (ptrdiff_t)floor$f(($element)column * bin_width) + start_column,
                        (ptrdiff_t)width);
                    const ptrdiff_t last_column = $clamp(
                        (ptrdiff_t)ceil$f(($element)(column + 1) * bin_width)
                            + start_column,
                        (ptrdiff_t)width);
                    /* A bin the region does not reach pools nothing and stays at zero. One
                       that does starts from the lowest finite value of its type rather than
                       from negative infinity: that is the floor Caffe's ROI pooling -- which
                       is what ONNX inherited this op from, and what onnxruntime, its only
                       implementation, still computes -- pools down to. */
                    $element best = $lowest;
                    ptrdiff_t sampled_row, sampled_column;
                    if (last_row <= first_row || last_column <= first_column) {
                        result[row * pooled_width + column] = $zero;
                        continue;
                    }
                    for (sampled_row = first_row; sampled_row < last_row; ++sampled_row) {
                        for (sampled_column = first_column;
                                sampled_column < last_column;
                                ++sampled_column) {
                            const $element value =
                                plane[(size_t)sampled_row * width
                                    + (size_t)sampled_column];
                            /* The later of the two wins unless it is strictly smaller,
                               which is what `std::max` -- and so onnxruntime, the only
                               implementation ONNX has for this op -- folds a window with.
                               It is observable wherever one holds a NaN, which compares
                               smaller than nothing. */
                            best = (value < best) ? best : value;
                        }
                    }
                    result[row * pooled_width + column] = best;
                }
            }
        }
    }
    return 0;
}""")

_CLAMP_TEMPLATE = Template("""\
static ptrdiff_t $name(ptrdiff_t value, ptrdiff_t extent)
{
    if (value < 0) {
        return 0;
    }
    return (value > extent) ? extent : value;
}""")


# --------------------------------------------------------------------------------------
# GridSample
# --------------------------------------------------------------------------------------


def _grid_sample(context: NodeContext) -> NodeEmission:
    """GridSample: every output position read at the coordinate the grid names for it."""
    source = context.require_input(0)
    grid = context.require_input(1)
    result = context.require_output(0)
    rank = _grid_geometry(context, source, grid, result)
    mode = _mode(context)
    padding_mode = _choice(context, "padding_mode", "zeros", _PADDING_MODES)
    align_corners = int(context.int_attribute("align_corners") != 0)
    _require_float(context, grid)
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    coord = c_type(grid.elem_type)
    padding = _padding_helpers(context, grid.elem_type)
    arguments = [
        result.expr,
        source.expr,
        grid.expr,
        f"{source.shape[0]}u",
        f"{source.shape[1]}u",
        f"{math.prod(source.shape[2:])}u",
        f"{math.prod(result.shape[2:])}u",
        str(rank),
        extents(source.shape[2:]),
        str(padding_mode),
        str(align_corners),
    ]
    if mode == _MODES["nearest"]:
        name = f"{context.prefix}_gridsample_nearest_{element}_{coord}"
        definition = _GRID_NEAREST_TEMPLATE.substitute(
            name=name,
            element=element,
            coord=coord,
            parameters=_GRID_PARAMETERS,
            coordinate=padding.coordinate(nearest=1),
            index=padding.index.name,
            zero=scalar_literal(0, result.elem_type),
        )
        return NodeEmission(
            functions=(*padding.functions, CFunction(name, definition)),
            statements=(call_kernel(name, arguments),),
        )

    coefficient = _coefficient(context, grid.elem_type)
    name = f"{context.prefix}_gridsample_{element}_{coord}"
    definition = _GRID_INTERPOLATE_TEMPLATE.substitute(
        name=name,
        element=element,
        coord=coord,
        parameters=_GRID_PARAMETERS,
        coordinate=padding.coordinate(nearest=0),
        index=padding.index.name,
        coefficient=coefficient.name,
        linear=_MODES["linear"],
        f=math_suffix(grid.elem_type),
        zero=scalar_literal(0, result.elem_type),
        unit=scalar_literal(1, grid.elem_type),
    )
    return NodeEmission(
        functions=(*padding.functions, coefficient, CFunction(name, definition)),
        statements=(call_kernel(name, [*arguments, str(mode)]),),
    )


def _mode(context: NodeContext) -> int:
    """The interpolation GridSample asks for, refused where its element type cannot hold it.

    `nearest` reads one element and computes nothing, so it serves every type the schema
    allows; the other two weight the elements around a coordinate, which is arithmetic an
    integer tensor cannot carry — ONNX's own reference truncates the weights to integers
    there rather than defining anything usable.
    """
    mode = _choice(context, "mode", "linear", _MODES)
    elem_type = context.require_output(0).elem_type
    if mode == _MODES["nearest"] or elem_type in FLOAT_TYPES:
        return mode
    name = next(choice for choice, value in _MODES.items() if value == mode)
    raise CompileError(
        f"Node `{context.label}`: `GridSample` in `{name}` mode weights the elements "
        f"around each coordinate, which a `{element_type_name(elem_type)}` tensor cannot "
        "hold; only `nearest` mode is supported for it."
    )


def _grid_geometry(
    context: NodeContext, source: TensorRef, grid: TensorRef, result: TensorRef
) -> int:
    """The number of axes the grid samples along, once every operand agrees on it."""
    rank = len(source.shape) - 2
    if rank < 1:
        raise CompileError(
            f"Node `{context.label}`: `GridSample` samples a batch of multi-channel "
            f"signals — a tensor of rank 3 or more — but `{source.name}` has shape "
            f"{list(source.shape)}."
        )
    if len(grid.shape) != rank + 2 or grid.shape[0] != source.shape[0]:
        raise CompileError(
            f"Node `{context.label}`: `GridSample` reads `{grid.name}` of shape "
            f"{list(grid.shape)} as one coordinate per sampled position of "
            f"`{source.name}` of shape {list(source.shape)}; ONNX defines it as the batch "
            f"of `{source.name}`, one axis per sampled position, and a trailing axis of "
            f"{rank} coordinate(s)."
        )
    if grid.shape[-1] != rank:
        raise CompileError(
            f"Node `{context.label}`: `GridSample` samples {rank} spatial axis/axes of "
            f"`{source.name}`, but `{grid.name}` carries {grid.shape[-1]} coordinate(s) "
            "per position."
        )
    verify_shape(context, result, (*source.shape[:2], *grid.shape[1:-1]))
    return rank


# --------------------------------------------------------------------------------------
# AffineGrid
# --------------------------------------------------------------------------------------


def _affine_grid(context: NodeContext) -> NodeEmission:
    """AffineGrid: the coordinates an affine transform maps a regular grid onto."""
    theta = context.require_input(0)
    result = context.require_output(0)
    size = context.constant_input(1)
    if size is None:
        raise CompileError(
            f"Node `{context.label}`: `AffineGrid` takes the grid's shape from "
            f"`{context.require_input(1).name}`, which is not known at compile time; the "
            "shape of the result then depends on input data, which the C compiler cannot "
            "compile."
        )
    _require_float(context, theta)
    spatial = tuple(int(extent) for extent in size.reshape(-1))[2:]
    rank = len(spatial)
    if rank < 1 or any(extent < 0 for extent in spatial):
        raise CompileError(
            f"Node `{context.label}`: `AffineGrid` was given a size of "
            f"{[int(extent) for extent in size.reshape(-1)]}; ONNX defines it as a batch, "
            "a channel count and one nonnegative extent per spatial axis."
        )
    batch = int(size.reshape(-1)[0])
    if theta.shape != (batch, rank, rank + 1):
        raise CompileError(
            f"Node `{context.label}`: `AffineGrid` maps {rank} spatial axis/axes of a "
            f"batch of {batch}, which ONNX defines as a transform of shape "
            f"{[batch, rank, rank + 1]}, but `{theta.name}` has shape "
            f"{list(theta.shape)}."
        )
    verify_shape(context, result, (batch, *spatial, rank))
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())

    element = c_type(result.elem_type)
    coordinate = CFunction(
        f"{context.prefix}_affine_coordinate",
        _AFFINE_COORDINATE_TEMPLATE.substitute(
            name=f"{context.prefix}_affine_coordinate"
        ),
    )
    name = f"{context.prefix}_affinegrid_{element}"
    definition = _AFFINE_GRID_TEMPLATE.substitute(
        name=name, element=element, coordinate=coordinate.name
    )
    call = call_kernel(
        name,
        [
            result.expr,
            theta.expr,
            f"{batch}u",
            f"{math.prod(spatial)}u",
            str(rank),
            extents(spatial),
            str(int(context.int_attribute("align_corners") != 0)),
        ],
    )
    return NodeEmission(
        functions=(coordinate, CFunction(name, definition)), statements=(call,)
    )


# --------------------------------------------------------------------------------------
# The region-of-interest poolings
# --------------------------------------------------------------------------------------


def _roi_align(context: NodeContext) -> NodeEmission:
    """RoiAlign: each region divided into bins, each bin folding a grid of samples."""
    source = context.require_input(0)
    rois = context.require_input(1)
    indices = context.require_input(2)
    result = context.require_output(0)
    height, width = _feature_map(context, source)
    pooled = (
        context.int_attribute("output_height"),
        context.int_attribute("output_width"),
    )
    _require_float(context, rois)
    _verify_regions(context, rois, indices, columns=4)
    verify_shape(context, result, (rois.shape[0], source.shape[1], *pooled))
    sampling_ratio = context.int_attribute("sampling_ratio")
    mode = _choice(context, "mode", "avg", _ROI_MODES)
    transform = _choice(
        context, "coordinate_transformation_mode", "half_pixel", _ROI_TRANSFORMS
    )
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())
    _require_samples(context, source)

    element = c_type(result.elem_type)
    name = f"{context.prefix}_roialign_{element}"
    definition = _ROI_ALIGN_TEMPLATE.substitute(
        name=name,
        element=element,
        one=scalar_literal(1, result.elem_type),
        zero=scalar_literal(0, result.elem_type),
        limit=scalar_literal(_COORDINATE_LIMIT, TensorProto.DOUBLE),
    )
    call = checked_call(
        context,
        name,
        [
            result.expr,
            source.expr,
            rois.expr,
            indices.expr,
            f"{rois.shape[0]}u",
            f"{source.shape[0]}u",
            f"{source.shape[1]}u",
            f"{height}u",
            f"{width}u",
            f"{pooled[0]}u",
            f"{pooled[1]}u",
            str(sampling_ratio),
            str(int(transform == _ROI_TRANSFORMS["half_pixel"])),
            str(mode),
            scalar_literal(context.float_attribute("spatial_scale"), result.elem_type),
        ],
    )
    return NodeEmission(functions=(CFunction(name, definition),), statements=(call,))


def _max_roi_pool(context: NodeContext) -> NodeEmission:
    """MaxRoiPool: each region rounded to whole elements, then max-pooled into its bins."""
    source = context.require_input(0)
    rois = context.require_input(1)
    result = context.require_output(0)
    height, width = _feature_map(context, source)
    pooled = context.attribute("pooled_shape", None)
    if pooled is None:
        raise CompileError(
            f"Node `{context.label}`: `MaxRoiPool` states no `pooled_shape`, which ONNX "
            "defines as a required attribute."
        )
    pooled = tuple(int(extent) for extent in pooled)
    if len(pooled) != 2 or any(extent < 1 for extent in pooled):
        raise CompileError(
            f"Node `{context.label}`: `MaxRoiPool` was given `pooled_shape` "
            f"{list(pooled)}; ONNX defines it as a positive height and width."
        )
    _require_float(context, rois)
    _verify_regions(context, rois, None, columns=5)
    verify_shape(context, result, (rois.shape[0], source.shape[1], *pooled))
    if result.elem_count == 0:
        return NodeEmission(functions=(), statements=())
    _require_samples(context, source)

    element = c_type(result.elem_type)
    clamp = CFunction(
        f"{context.prefix}_roi_clamp",
        _CLAMP_TEMPLATE.substitute(name=f"{context.prefix}_roi_clamp"),
    )
    name = f"{context.prefix}_maxroipool_{element}"
    definition = _MAX_ROI_POOL_TEMPLATE.substitute(
        name=name,
        element=element,
        clamp=clamp.name,
        f=math_suffix(result.elem_type),
        zero=scalar_literal(0, result.elem_type),
        lowest=scalar_literal(
            -float(np.finfo(numpy_dtype_name(result.elem_type)).max), result.elem_type
        ),
        limit=scalar_literal(_COORDINATE_LIMIT, TensorProto.DOUBLE),
    )
    call = checked_call(
        context,
        name,
        [
            result.expr,
            source.expr,
            rois.expr,
            f"{rois.shape[0]}u",
            f"{source.shape[0]}u",
            f"{source.shape[1]}u",
            f"{height}u",
            f"{width}u",
            f"{pooled[0]}u",
            f"{pooled[1]}u",
            scalar_literal(context.float_attribute("spatial_scale"), result.elem_type),
        ],
    )
    return NodeEmission(
        functions=(clamp, CFunction(name, definition)), statements=(call,)
    )


def _feature_map(context: NodeContext, source: TensorRef) -> tuple[int, int]:
    if len(source.shape) != 4:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` pools regions of a batch of "
            f"images — a tensor of rank 4 — but `{source.name}` has shape "
            f"{list(source.shape)}."
        )
    return source.shape[2], source.shape[3]


def _verify_regions(
    context: NodeContext, rois: TensorRef, indices: TensorRef | None, *, columns: int
) -> None:
    if len(rois.shape) != 2 or rois.shape[1] != columns:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{rois.name}` of "
            f"shape {list(rois.shape)} as its regions; ONNX defines them as one row of "
            f"{columns} value(s) per region."
        )
    if indices is not None and indices.shape != (rois.shape[0],):
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` takes one batch index per "
            f"region, but `{indices.name}` has shape {list(indices.shape)} against "
            f"{rois.shape[0]} region(s)."
        )


def _require_samples(context: NodeContext, source: TensorRef) -> None:
    """Refuse to pool regions of a feature map that holds no elements to sample.

    Every region samples the map whatever it holds, and ONNX's own reference implementation
    indexes past the end of an empty one rather than defining a value for it, so there is
    nothing to compile against.
    """
    if math.prod(source.shape[2:]) == 0:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` pools regions of "
            f"`{source.name}` of shape {list(source.shape)}, which holds no elements to "
            "sample."
        )


# --------------------------------------------------------------------------------------
# Shared emission
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Padding:
    """What turns a grid value into an element of the operand, under the padding mode."""

    reflect: CFunction
    locate: CFunction
    index: CFunction
    coord: str

    @property
    def functions(self) -> tuple[CFunction, ...]:
        return (self.reflect, self.locate, self.index)

    def coordinate(self, *, nearest: int) -> str:
        """Reading one output position's coordinate along one axis, as kernel body text."""
        return Template(_GRID_COORDINATE).substitute(
            locate=self.locate.name, nearest=nearest, coord=self.coord
        )


def _padding_helpers(context: NodeContext, elem_type: int) -> _Padding:
    coord = c_type(elem_type)
    reflect = CFunction(
        f"{context.prefix}_sample_reflect",
        _REFLECT_TEMPLATE.substitute(name=f"{context.prefix}_sample_reflect"),
    )
    locate = f"{context.prefix}_sample_locate_{coord}"
    index = f"{context.prefix}_sample_index"
    return _Padding(
        reflect=reflect,
        locate=CFunction(
            locate,
            _LOCATE_TEMPLATE.substitute(
                name=locate,
                coord=coord,
                reflect=reflect.name,
                border=_PADDING_MODES["border"],
                reflection=_PADDING_MODES["reflection"],
                f=math_suffix(elem_type),
                one=scalar_literal(1, elem_type),
                zero=scalar_literal(0, elem_type),
            ),
        ),
        index=CFunction(
            index,
            _INDEX_TEMPLATE.substitute(
                name=index,
                reflect=reflect.name,
                zeros=_PADDING_MODES["zeros"],
                reflection=_PADDING_MODES["reflection"],
            ),
        ),
        coord=coord,
    )


def _coefficient(context: NodeContext, elem_type: int) -> CFunction:
    name = f"{context.prefix}_sample_coefficient_{c_type(elem_type)}"
    return CFunction(
        name,
        _COEFFICIENT_TEMPLATE.substitute(
            name=name,
            coord=c_type(elem_type),
            linear=_MODES["linear"],
            one=scalar_literal(1, elem_type),
        ),
    )


def _require_float(context: NodeContext, operand: TensorRef) -> None:
    """Refuse an operand of coordinates the emitted arithmetic would not compute in.

    ONNX defines every one of these as floating-point; a model declaring one otherwise is
    rejected rather than served with integer arithmetic that would silently truncate.
    """
    if operand.elem_type not in FLOAT_TYPES:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` reads `{operand.name}` as "
            f"coordinates, which ONNX defines as floating-point, but it holds "
            f"`{element_type_name(operand.elem_type)}`."
        )


def _choice(
    context: NodeContext, attribute: str, default: str, choices: dict[str, int]
) -> int:
    value = context.attribute(attribute, default)
    name = value.decode() if isinstance(value, bytes) else str(value)
    if name not in choices:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` asks for `{attribute}` "
            f"`{name}`, which is not one of the values ONNX defines for it "
            f"({', '.join(f'`{choice}`' for choice in choices)})."
        )
    return choices[name]


register_kernel("", "GridSample", _VERSIONS, _grid_sample)
register_kernel("", "AffineGrid", _AFFINE_GRID_VERSIONS, _affine_grid)
register_kernel("", "RoiAlign", _VERSIONS, _roi_align)
register_kernel("", "MaxRoiPool", _VERSIONS, _max_roi_pool)
