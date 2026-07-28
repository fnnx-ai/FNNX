"""The compiler frontend: from a loaded model to a fully static, verified graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from onnx import ModelProto

from fnnx.extras.compilers.c.onnx import folding, shapes
from fnnx.extras.compilers.c.onnx.loader import LoadedModel
from fnnx.extras.compilers.c.onnx.verify import verify_static
from fnnx.extras.compilers.c.onnx.zipmap import ClassLabels, remove_zipmap


@dataclass(frozen=True)
class PreparedModel:
    """A graph whose every tensor has a static shape and a supported element type."""

    model: ModelProto
    opsets: dict[str, int]
    dim_bindings: dict[str, int]
    class_labels: tuple[ClassLabels, ...] = ()


def prepare_model(
    loaded: LoadedModel, *, dim_bindings: Mapping[str, int] | None = None
) -> PreparedModel:
    """Bind symbolic dimensions, fold constants to fixpoint, and verify the result."""
    model = ModelProto()
    model.CopyFrom(loaded.model)
    bindings = dim_bindings or {}

    shapes.drop_shadowed_inputs(model)
    # Before the declared types are snapshotted: the entry a removed `ZipMap` leaves behind
    # describes a sequence of maps, which is nothing the tensor taking its place could fall
    # back on.
    class_labels = remove_zipmap(model)
    declared_outputs = shapes.declared_output_types(model.graph)
    applied = shapes.bind_dims(model, bindings)
    shapes.drop_empty_shape_operands(model)
    shapes.state_stft_onesided(model)
    model = shapes.infer_shapes(model)
    while folding.fold_constants(model, loaded.opsets):
        model = shapes.infer_shapes(model)
    applied.update(
        shapes.apply_declared_output_shapes(model, declared_outputs, bindings)
    )
    folding.prune_unused_initializers(model.graph)
    folding.prune_stale_value_info(model.graph)

    verify_static(model)
    return PreparedModel(
        model=model,
        opsets=dict(loaded.opsets),
        dim_bindings=applied,
        class_labels=class_labels,
    )
