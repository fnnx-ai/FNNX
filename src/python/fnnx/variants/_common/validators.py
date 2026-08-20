from types import ModuleType

from fnnx.dtypes import NDContainer

np: ModuleType | None
try:
    import numpy as np
except ImportError:
    np = None


def validate_inputs(inputs, input_specs):
    for input_spec, input_val in zip(input_specs, inputs):
        # validate shape
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
        # validate dtype
        if input_spec["dtype"].startswith("Array["):
            if np is None:
                raise RuntimeError("You must have numpy installed to use Array dtype")
            spec_dtype = input_spec["dtype"][6:-1]
            if not isinstance(input_val, np.ndarray):
                raise ValueError(
                    f"Expected input dtype {input_spec['dtype']}, got {type(input_val)}"
                )
            if not input_val.dtype == spec_dtype:
                if not (
                    spec_dtype == "string" and np.issubdtype(input_val.dtype, np.str_)
                ):
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
