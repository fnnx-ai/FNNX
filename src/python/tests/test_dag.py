import unittest
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Any
from fnnx.variants._common.dag import (
    dag_compute_async,
    dag_compute,
    DagComponent as Component,
)
from fnnx.variants._common.validators import validate_inputs as _validate_inputs


class TestDagFunctions(unittest.TestCase):
    def test_validate_inputs_valid(self):
        input_specs = [{"shape": (2, 2), "dtype": "Array[float64]"}]
        inputs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        try:
            _validate_inputs(inputs, input_specs)
        except Exception as e:
            self.fail(f"_validate_inputs raised an exception {e}")

    def test_validate_inputs_invalid_shape(self):
        input_specs = [{"shape": (2, 2), "dtype": "Array[float64]"}]
        inputs = [np.array([1.0, 2.0])]
        with self.assertRaises(ValueError):
            _validate_inputs(inputs, input_specs)

    def test_validate_inputs_invalid_dtype(self):
        input_specs = [{"shape": (2, 2), "dtype": "Array[int32]"}]
        inputs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        with self.assertRaises(ValueError):
            _validate_inputs(inputs, input_specs)

    async def async_compute_fn(self, component, inputs, **kwargs):
        await asyncio.sleep(0.1)
        return [np.sum(inputs[0])]

    def as_val(self, result):
        return result

    def test_dag_compute_async(self):

        inputs = {"x": np.array([1, 2, 3])}
        components = [
            Component(
                inputs=["x"],
                outputs=["y"],
                extra_dynattrs={},
            )
        ]
        components_passthrough: dict[str, Any] = {}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            dag_compute_async(
                inputs,
                components,
                self.async_compute_fn,
                self.as_val,
                components_passthrough,
            )
        )
        self.assertIn("y", result)
        self.assertEqual(result["y"], 6)

    def test_dag_compute_async_does_not_leak_extra_dynamic_attributes(self) -> None:
        async def compute_dynamic_attribute(
            component: Component, inputs: list[Any], **kwargs: Any
        ) -> list[str]:
            return [kwargs["dynamic_attributes"]["n"]]

        inputs = {"x": np.array([1])}
        components = [
            Component(inputs=["x"], outputs=["y1"], extra_dynattrs={"n": "pinned"}),
            Component(inputs=["x"], outputs=["y2"], extra_dynattrs={}),
        ]
        passthrough = {"dynamic_attributes": {"n": "caller"}}

        result = asyncio.run(
            dag_compute_async(
                inputs,
                components,
                compute_dynamic_attribute,
                self.as_val,
                passthrough,
            )
        )

        self.assertEqual(result["y1"], "pinned")
        self.assertEqual(result["y2"], "caller")
        self.assertEqual(passthrough["dynamic_attributes"], {"n": "caller"})

    def compute_fn(self, component, inputs, **kwargs):
        return [np.prod(inputs[0])]

    def test_dag_compute(self):

        inputs = {"x": np.array([1, 2, 3, 4])}
        components = [
            Component(
                inputs=["x"],
                outputs=["y"],
                extra_dynattrs={},
            )
        ]
        components_passthrough: dict[str, Any] = {}
        graph_executor = ThreadPoolExecutor(max_workers=2)
        result = dag_compute(
            inputs,
            components,
            graph_executor,
            self.compute_fn,
            self.as_val,
            components_passthrough,
        )
        self.assertIn("y", result)
        self.assertEqual(result["y"], 24)

    def test_dag_compute_does_not_leak_extra_dynamic_attributes(self) -> None:
        def compute_dynamic_attribute(
            component: Component, inputs: list[Any], **kwargs: Any
        ) -> list[str]:
            return [kwargs["dynamic_attributes"]["n"]]

        inputs = {"x": np.array([1])}
        components = [
            Component(inputs=["x"], outputs=["y1"], extra_dynattrs={"n": "pinned"}),
            Component(inputs=["x"], outputs=["y2"], extra_dynattrs={}),
        ]
        passthrough = {"dynamic_attributes": {"n": "caller"}}

        with ThreadPoolExecutor(max_workers=2) as graph_executor:
            result = dag_compute(
                inputs,
                components,
                graph_executor,
                compute_dynamic_attribute,
                self.as_val,
                passthrough,
            )

        self.assertEqual(result["y1"], "pinned")
        self.assertEqual(result["y2"], "caller")
        self.assertEqual(passthrough["dynamic_attributes"], {"n": "caller"})


if __name__ == "__main__":
    unittest.main()
