import json
import math
import unittest

import numpy as np
import numpy.typing as npt

from fnnx.dtypes import BUILTINS, DtypesManager
from fnnx.handlers.local import LocalHandler
from fnnx.handlers.stdio import StdIOHandler
from fnnx.handlers.stdio.worker import _to_jsonable


def _ndjson_spec(dtype: str) -> dict[str, object]:
    return {"name": "value", "content_type": "NDJSON", "dtype": dtype, "shape": [4]}


class TestStdIONumericWire(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = object.__new__(StdIOHandler)
        self.handler.dtypes_manager = DtypesManager({}, BUILTINS)

    def test_non_finite_float_array_round_trip_uses_strings(self) -> None:
        spec = _ndjson_spec("Array[float64]")
        self.handler.input_specs = {"value": spec}
        self.handler.output_specs = {"value": spec}
        values: npt.NDArray[np.float64] = np.asarray(
            [math.nan, math.inf, -math.inf, 1.5], dtype=np.float64
        )

        request_body = self.handler._inputs_to_wire({"value": values})
        request_json = json.dumps(request_body, allow_nan=False)

        self.assertEqual(
            request_json,
            '{"value": ["NaN", "Infinity", "-Infinity", 1.5]}',
        )

        local_handler = object.__new__(LocalHandler)
        worker_value = local_handler._as_np(json.loads(request_json)["value"], spec)
        response_body = _to_jsonable({"value": worker_value})
        response_json = json.dumps(response_body, allow_nan=False)
        result = self.handler._outputs_from_wire(json.loads(response_json))["value"]

        self.assertTrue(math.isnan(result[0]))
        self.assertEqual(result[1], math.inf)
        self.assertEqual(result[2], -math.inf)
        self.assertEqual(result[3], 1.5)

    def test_float_array_rejects_unrecognized_string(self) -> None:
        spec = _ndjson_spec("Array[float32]")
        local_handler = object.__new__(LocalHandler)

        with self.assertRaisesRegex(ValueError, "finite float"):
            local_handler._as_np(["1.5"], spec)

        self.handler.output_specs = {"value": spec}
        with self.assertRaisesRegex(ValueError, "finite float"):
            self.handler._outputs_from_wire({"value": ["not-a-float"]})

    def test_float_marshalling_preserves_empty_ndarray_shape(self) -> None:
        handlers: tuple[LocalHandler | StdIOHandler, ...] = (
            object.__new__(LocalHandler),
            object.__new__(StdIOHandler),
        )
        for handler in handlers:
            for dtype in ("float32", "float64"):
                with self.subTest(handler=type(handler).__name__, dtype=dtype):
                    values: npt.NDArray[np.float32 | np.float64] = np.empty(
                        (0, 3), dtype=dtype
                    )

                    result = handler._as_np(values, _ndjson_spec(f"Array[{dtype}]"))

                    self.assertEqual(result.shape, (0, 3))
                    self.assertEqual(result.dtype, values.dtype)

    def test_int64_round_trip_preserves_exact_value(self) -> None:
        spec = _ndjson_spec("Array[int64]")
        self.handler.input_specs = {"value": spec}
        self.handler.output_specs = {"value": spec}
        expected = 2**53 + 1
        values: npt.NDArray[np.int64] = np.asarray([expected], dtype=np.int64)

        request_json = json.dumps(
            self.handler._inputs_to_wire({"value": values}), allow_nan=False
        )
        worker_payload = json.loads(request_json)
        self.assertIsInstance(worker_payload["value"][0], int)
        self.assertEqual(worker_payload["value"][0], expected)

        local_handler = object.__new__(LocalHandler)
        worker_value = local_handler._as_np(worker_payload["value"], spec)
        response_json = json.dumps(
            _to_jsonable({"value": worker_value}), allow_nan=False
        )
        result = self.handler._outputs_from_wire(json.loads(response_json))["value"]

        self.assertEqual(int(result[0]), expected)


if __name__ == "__main__":
    unittest.main()
