from types import ModuleType
from typing import Any

from fnnx.dtypes import NDContainer
from fnnx.node_instance import OpInstance
from fnnx.ops._base import BaseOp, OpOutput

np: ModuleType | None
try:
    import numpy as np
except ImportError:
    np = None


def validate_inputs(inputs, input_specs):
    if len(inputs) != len(input_specs):
        raise ValueError(f"Expected {len(input_specs)} inputs, got {len(inputs)}")
    for input_spec, input_val in zip(input_specs, inputs):
        # The dtype check comes first: it is what establishes that the value carries a
        # shape at all.
        _validate_dtype(input_spec, input_val)
        _validate_shape(input_spec, input_val)


def _validate_dtype(input_spec, input_val) -> None:
    if input_spec["dtype"].startswith("Array["):
        if np is None:
            raise RuntimeError("You must have numpy installed to use Array dtype")
        spec_dtype = input_spec["dtype"][6:-1]
        if not isinstance(input_val, np.ndarray):
            raise ValueError(
                f"Expected input dtype {input_spec['dtype']}, got {type(input_val)}"
            )
        if not input_val.dtype == spec_dtype:
            if not (spec_dtype == "string" and np.issubdtype(input_val.dtype, np.str_)):
                raise ValueError(
                    f"Expected input dtype {input_spec['dtype']}, got Array[{input_val.dtype}]"
                )
    elif input_spec["dtype"].startswith("NDContainer["):
        spec_dtype = input_spec["dtype"][12:-1]
        if not isinstance(input_val, NDContainer):
            raise ValueError(
                f"Expected input dtype {input_spec['dtype']}, got {type(input_val)}"
            )
        if not input_val._dtype == spec_dtype:
            raise ValueError(
                f"Expected input dtype {input_spec['dtype']}, got NDContainer[{input_val._dtype}]"
            )
    else:
        raise ValueError(f"Unknown dtype {input_spec['dtype']}")


def _validate_shape(input_spec, input_val) -> None:
    input_shape = input_val.shape
    if len(input_spec["shape"]) != len(input_shape):
        raise ValueError(
            f"Expected input shape {input_spec['shape']}, got {input_shape}"
        )
    for spec_dim, input_dim in zip(input_spec["shape"], input_shape):
        if (not isinstance(spec_dim, str)) and spec_dim != input_dim:
            raise ValueError(
                f"Expected input shape {input_spec['shape']}, got {input_shape}"
            )


class ValidatingOperator(BaseOp):
    """An operator that validates its inputs against the op instance's declarations."""

    def __init__(self, operator: BaseOp, input_specs: Any) -> None:
        self._operator = operator
        self._input_specs = input_specs

    def warmup(self, *args: Any, **kwargs: Any) -> BaseOp:
        # Returns the wrapper, not the wrapped op: `warmup().compute(...)` must still
        # validate.
        self._operator.warmup(*args, **kwargs)
        return self

    def compute(
        self, inputs: list[Any], dynamic_attributes: dict[str, str], **kwargs: Any
    ) -> OpOutput:
        validate_inputs(inputs, self._input_specs)
        return self._operator.compute(inputs, dynamic_attributes, **kwargs)

    async def compute_async(
        self, inputs: list[Any], dynamic_attributes: dict[str, str], **kwargs: Any
    ) -> OpOutput:
        validate_inputs(inputs, self._input_specs)
        return await self._operator.compute_async(inputs, dynamic_attributes, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._operator, name)


def validating_op_instance(instance: OpInstance) -> OpInstance:
    return OpInstance(
        operator=ValidatingOperator(instance.operator, instance.input_specs),
        input_specs=instance.input_specs,
        output_specs=instance.output_specs,
    )
