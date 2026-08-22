import math
from copy import deepcopy
from typing import Any

from fnnx.validators.jsonschema import validate_jsonschema

RESERVED_DTYPE_NAMES: frozenset[str] = frozenset(
    {
        "string",
        "integer",
        "float",
        "boolean",
        "Array",
        "NDContainer",
        "float32",
        "float64",
        "int32",
        "int64",
        "bool",
    }
)
_NONFINITE_FLOATS: dict[str, float] = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


def encode_nonfinite_floats(data: Any) -> Any:
    if isinstance(data, (list, tuple)):
        return [encode_nonfinite_floats(value) for value in data]
    if isinstance(data, float) and not math.isfinite(data):
        if math.isnan(data):
            return "NaN"
        return "Infinity" if data > 0 else "-Infinity"
    return data


def decode_nonfinite_float_strings(data: Any) -> Any:
    if isinstance(data, (list, tuple)):
        return [decode_nonfinite_float_strings(value) for value in data]
    if isinstance(data, str):
        if data not in _NONFINITE_FLOATS:
            raise ValueError(
                f"Invalid float array string {data!r}; expected a non-finite float marker"
            )
        return _NONFINITE_FLOATS[data]
    return data


class DtypesManager:
    def __init__(
        self, external_dtypes: dict[str, Any], builtins: dict[str, Any]
    ) -> None:
        self.dtypes: dict[str, Any] = deepcopy(external_dtypes)
        self.dtypes.update(deepcopy(builtins))
        for dtype in self.dtypes:
            if "[" in dtype:
                raise ValueError(f"Invalid dtype name: {dtype}")
        for reserved_type in RESERVED_DTYPE_NAMES:
            if reserved_type in self.dtypes:
                raise ValueError(f"Invalid dtype name: {reserved_type}")

    def get_dtype(self, dtype_name: str) -> dict[str, Any]:
        if dtype_name not in self.dtypes:
            raise ValueError(f"Unknown dtype: {dtype_name}")
        return self.dtypes[dtype_name]

    def validate_jsonschema(self, dtype_name: str, data: Any) -> None:
        schema = self.get_dtype(dtype_name)
        validate_jsonschema(data, schema)

    def validate_dtype(self, dtype_name: str, data: Any) -> None:
        if isinstance(data, list):
            for d in data:
                self.validate_dtype(dtype_name, d)
        elif isinstance(data, dict):
            self.validate_jsonschema(dtype_name, data)
        elif isinstance(data, str):
            if dtype_name != "string":
                raise TypeError(
                    f"Invalid data type, expected `string`, got `{dtype_name}`"
                )
        elif isinstance(data, bool):
            if dtype_name != "boolean":
                raise TypeError(
                    f"Invalid data type, expected `boolean`, got `{dtype_name}`"
                )
        elif isinstance(data, int):
            # A whole JSON number also satisfies a declared `float`: some JSON
            # parsers erase the 2.0/2 distinction, and core.md permits the leniency.
            if dtype_name not in ("integer", "float"):
                raise TypeError(
                    f"Invalid data type, expected `integer`, got `{dtype_name}`"
                )
        elif isinstance(data, float):
            if dtype_name != "float":
                raise TypeError(
                    f"Invalid data type, expected `float`, got `{dtype_name}`"
                )
        else:
            raise TypeError(f"Invalid data type: {type(data)}")


class NDContainer:
    def __init__(self, data, dtype, dtypes_manager: DtypesManager):
        if dtype.startswith("Array["):
            raise ValueError("NDContainer does not support Array dtype")
        elif dtype.startswith("NDContainer["):
            dtype = dtype[12:-1]
            if dtype.startswith("Array["):
                raise ValueError(
                    f"NDContainer inner dtype `{dtype}` must not be an Array[...] form"
                )

        self.data = deepcopy(data if isinstance(data, list) else [data])

        if dtypes_manager:
            dtypes_manager.validate_dtype(dtype, self.data)
        self.dtypes_manager = dtypes_manager
        self._dtype = dtype
        self.shape = tuple(self._compute_shape(self.data))

    def _compute_shape(self, data):
        if not isinstance(data, list) or not data:
            return []
        sub_shape = self._compute_shape(data[0])
        return [len(data)] + sub_shape

    def __getitem__(self, index):
        if isinstance(index, tuple):
            result = self.data
            for idx in index:
                result = result[idx]
            return result
        return self.data[index]

    def reshape(self, *new_shape):
        new_dimensions = (
            new_shape[0]
            if isinstance(new_shape[0], (tuple, list))
            else new_shape
        )
        # Check if the total number of elements matches
        if self._product(new_dimensions) != self._product(self.shape):
            raise ValueError(
                "Cannot reshape array of size {} into shape {}".format(
                    self._product(self.shape), new_dimensions
                )
            )
        flat_list = self.flatten(self.data)
        if self._product(new_dimensions) != self._product(self.shape):
            raise ValueError(
                "Cannot reshape array of size {} into shape {}".format(
                    self._product(self.shape), new_dimensions
                )
            )
        reshaped = self._reshape_helper(flat_list, list(new_dimensions))
        return NDContainer(
            data=reshaped, dtype=self.dtype, dtypes_manager=self.dtypes_manager
        )

    def flatten(self, data: list | None = None):
        return NDContainer(
            data=self._flatten(data or self.data),
            dtype=self.dtype,
            dtypes_manager=self.dtypes_manager,
        )

    def _flatten(self, data) -> list:
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(self._flatten(item))
            else:
                result.append(item)
        return result

    def _reshape_helper(self, flat_list, shape):
        if len(shape) == 1:
            return flat_list[: shape[0]]
        step = self._product(shape[1:])
        return [
            self._reshape_helper(flat_list[i * step : (i + 1) * step], shape[1:])
            for i in range(shape[0])
        ]

    def _product(self, shape):
        product = 1
        for dim in shape:
            product *= dim
        return product

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        raise AttributeError("Cannot modify immutable attribute dtype")

    def __repr__(self) -> str:
        return f"NDContainer(shape={self.shape}, dtype={self._dtype}, data={self.data})"


BUILTINS: dict[str, Any] = {}
