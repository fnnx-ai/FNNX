import os
import unittest
from typing import Any, cast
from fnnx.ops._base import BaseOp, OpOutput
from fnnx.utils import to_thread
from fnnx.dtypes import NDContainer
from fnnx.runtime import Runtime
from fnnx.handlers.local import LocalHandlerConfig, LocalHandler
from asyncio import run
import numpy as np

MODELS_BASE_PATH = os.path.join(os.path.dirname(__file__), "models")


class _TestOP(BaseOp):
    def warmup(
        self,
    ) -> "_TestOP":
        self._warmed_up = True
        return self

    def compute(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        return OpOutput(
            [
                NDContainer(
                    data={"hello": "from inside the op"},
                    dtype=self.output_specs[0]["dtype"],
                    dtypes_manager=self.dtypes_manager,
                )
            ]
        )

    async def compute_async(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        executor = kwargs.get("op_executor")
        return await to_thread(
            executor, self.compute, inputs, dynamic_attributes=dynamic_attributes
        )


class _TestOpWithDynattrs(_TestOP):
    supported_dynamic_attributes = ["testop::value"]
    required_dynamic_attributes = ["testop::value"]

    def compute(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        **kwargs: Any,
    ) -> OpOutput:
        dynattrs = self.extract_dynamic_attribute(dynamic_attributes)
        self.verify_required_dynamic_attributes(dynattrs)

        return OpOutput(
            [
                NDContainer(
                    data={"hello": dynattrs["testop::value"]},
                    dtype=self.output_specs[0]["dtype"],
                    dtypes_manager=self.dtypes_manager,
                )
            ]
        )

    async def compute_async(
        self,
        inputs: list[Any],
        dynamic_attributes: dict[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> OpOutput:
        return self.compute(inputs, dynamic_attributes, **kwargs)


_testop = {
    "$defs": {
        "Empty": {
            "additionalProperties": False,
            "properties": {},
            "title": "Empty",
            "type": "object",
        },
        "OpDynamicAttribute": {
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "default_value": {"title": "Default Value", "type": "string"},
            },
            "required": ["name", "default_value"],
            "title": "OpDynamicAttribute",
            "type": "object",
        },
        "NodeIO": {
            "properties": {
                "dtype": {"title": "Dtype", "type": "string"},
                "shape": {
                    "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "title": "Shape",
                    "type": "array",
                },
            },
            "required": ["dtype", "shape"],
            "title": "NodeIO",
            "type": "object",
        },
        "Opset": {
            "properties": {
                "domain": {"title": "Domain", "type": "string"},
                "version": {"title": "Version", "type": "integer"},
            },
            "required": ["domain", "version"],
            "title": "Opset",
            "type": "object",
        },
    },
    "properties": {
        "id": {"pattern": "^[a-zA-Z0-9_]+$", "title": "Id", "type": "string"},
        "op": {
            "const": "TestOp",
            "enum": ["TestOp"],
            "title": "Op",
            "type": "string",
        },
        "inputs": {
            "items": {"$ref": "#/$defs/NodeIO"},
            "title": "Inputs",
            "type": "array",
        },
        "outputs": {
            "items": {"$ref": "#/$defs/NodeIO"},
            "title": "Outputs",
            "type": "array",
        },
        "attributes": {"$ref": "#/$defs/Empty"},
        "dynamic_attributes": {
            "additionalProperties": {"$ref": "#/$defs/OpDynamicAttribute"},
            "title": "Dynamic Attributes",
            "type": "object",
        },
    },
    "required": [
        "id",
        "op",
        "inputs",
        "outputs",
        "attributes",
        "dynamic_attributes",
    ],
    "title": "ONNX_v1",
    "type": "object",
}


class Test_TestOP(unittest.TestCase):
    def setUp(self) -> None:
        from fnnx.spec import schema

        ops_schema = cast(dict[str, Any], schema["ops"])
        ops_schema["TestOp"] = _testop
        self.runtime = Runtime(
            os.path.join(MODELS_BASE_PATH, "model_testop.fnnx"),
            handler=LocalHandler,
            handler_config=LocalHandlerConfig(extra_ops={"TestOp": _TestOP}),
        )
        ops_schema.pop("TestOp")

        self.inputs = {"x": [{"hello": "world"}]}

    def _check_output(self, res):
        self.assertIn("y1", res.keys())
        self.assertEqual(res["y1"].data, [{"hello": "from inside the op"}])
        self.assertEqual(res["y1"].dtype, "ext::hello")

    def test_compute(self):
        res = self.runtime.compute(self.inputs, {})
        self._check_output(res)

    def test_compute_async(self):
        res = run(self.runtime.compute_async(self.inputs, {}))
        self._check_output(res)


class Test_TestOP_Dynattrs(unittest.TestCase):
    def setUp(self) -> None:
        from fnnx.spec import schema

        ops_schema = cast(dict[str, Any], schema["ops"])
        ops_schema["TestOp"] = _testop
        self.runtime = Runtime(
            os.path.join(MODELS_BASE_PATH, "model_testop_dynattrs.fnnx"),
            handler=LocalHandler,
            handler_config=LocalHandlerConfig(
                extra_ops={"TestOp": _TestOpWithDynattrs}
            ),
        )

        self.runtime2 = Runtime(
            os.path.join(
                MODELS_BASE_PATH, "model_testop_dynattrs_pipeline_provided.fnnx"
            ),
            handler=LocalHandler,
            handler_config=LocalHandlerConfig(
                extra_ops={"TestOp": _TestOpWithDynattrs}
            ),
        )
        ops_schema.pop("TestOp")

        self.inputs = {"x": [{"hello": "world"}]}

    def _check_output(self, res, expected_value):
        self.assertIn("y1", res.keys())
        self.assertEqual(res["y1"].data, [{"hello": expected_value}])
        self.assertEqual(res["y1"].dtype, "ext::hello")

    def test_compute(self):
        expected = "from external source"
        res = self.runtime.compute(self.inputs, {"testdynattr": expected})
        self._check_output(res, expected)
        expected = ""
        res = self.runtime.compute(self.inputs, {"testdynattr": expected})
        self._check_output(res, expected)
        expected = "default_value"
        res = self.runtime.compute(self.inputs, {})
        self._check_output(res, expected)

        expected = "pipeline"
        res = self.runtime2.compute(self.inputs, {})
        self._check_output(res, expected)
        res = self.runtime2.compute(self.inputs, {"testdynattr": "abcdedf"})
        self._check_output(res, expected)

    def test_compute_async(self):
        expected = "from external source"
        res = run(self.runtime.compute_async(self.inputs, {"testdynattr": expected}))
        self._check_output(res, expected)
        expected = ""
        res = run(self.runtime.compute_async(self.inputs, {"testdynattr": expected}))
        self._check_output(res, expected)
        expected = "default_value"
        res = run(self.runtime.compute_async(self.inputs, {}))
        self._check_output(res, expected)

        expected = "pipeline"
        res = self.runtime2.compute(self.inputs, {})
        self._check_output(res, expected)
        res = self.runtime2.compute(self.inputs, {"testdynattr": "abcdedf"})
        self._check_output(res, expected)


class Test_OnnxV1Pipeline(unittest.TestCase):

    def setUp(self) -> None:

        self.runtime = Runtime(
            os.path.join(MODELS_BASE_PATH, "onnx_pipeline.fnnx"),
        )

        self.inputs = np.asarray([[0.80334168, 0.66020722, 0.15638698]]).astype(
            np.float32
        )

        self.runtime_from_tar = Runtime(
            os.path.join(MODELS_BASE_PATH, "onnx_pipeline.fnnx.tar")
        )

    def _check_output(self, res):
        self.assertIn("y4", res.keys())
        self.assertIsInstance(res["y4"], np.ndarray)
        self.assertTrue(np.allclose(res["y4"], [[86.54103]]))
        self.assertEqual(res["y4"].dtype, np.float32)

    def test_compute(self):
        res = self.runtime.compute({"x": self.inputs}, {})
        self._check_output(res)
        res = self.runtime_from_tar.compute({"x": self.inputs}, {})
        self._check_output(res)

    def test_compute_async(self):
        res = run(self.runtime.compute_async({"x": self.inputs}, {}))
        self._check_output(res)
        res = run(self.runtime_from_tar.compute_async({"x": self.inputs}, {}))
        self._check_output(res)


class Test_PyFunc(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime1 = Runtime(
            os.path.join(MODELS_BASE_PATH, "pyfunc.fnnx"),
        )
        self.runtime2 = Runtime(
            os.path.join(MODELS_BASE_PATH, "pyfunc2.fnnx"),
        )

        self.runtime3 = Runtime(
            os.path.join(MODELS_BASE_PATH, "pyfunc_with_extra.fnnx"),
        )

        self.inputs = {"x": np.asarray([1, 2, 3]).astype(np.int64)}

        self.out1 = np.asarray([1, 4, 9]).astype(np.int64)
        self.out2 = np.asarray([2, 5, 10]).astype(np.int64)
        self.out3 = np.asarray([9, 12, 15]).astype(np.int64)

    def test_alternating_invokations(self):
        res = self.runtime1.compute(self.inputs, {})
        self.assertTrue(np.all(res["y"] == self.out1))
        res = self.runtime2.compute(self.inputs, {})
        self.assertTrue(np.all(res["y"] == self.out2))
        res = self.runtime1.compute(self.inputs, {})
        self.assertTrue(np.all(res["y"] == self.out1))
        res = self.runtime2.compute(self.inputs, {})
        self.assertTrue(np.all(res["y"] == self.out2))

    def test_model_with_extras(self):
        res = self.runtime3.compute(self.inputs, {})
        self.assertTrue(np.all(res["y"] == self.out3))
