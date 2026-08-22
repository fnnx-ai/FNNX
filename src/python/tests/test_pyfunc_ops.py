"""An op instance handed to a `pyfunc` validates the values it is given, exactly as a
pipeline node does."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np

from fnnx.device import DeviceMap
from fnnx.dtypes import BUILTINS, DtypesManager
from fnnx.ops._base import BaseOp, OpOutput
from fnnx.registry import Registry
from fnnx.variants.pyfunc import PyFuncVariant

_ENTRY_MODULE = dedent(
    '''
    from fnnx.variants.pyfunc import PyFunc


    class Doubler(PyFunc):
        def warmup(self):
            pass

        def compute(self, inputs, dynamic_attributes):
            instance = self.fnnx_context.get_op_instance("doubler")
            output = instance.operator.compute([inputs["x"]], dynamic_attributes)
            return {"y": output.value[0]}

        async def compute_async(self, inputs, dynamic_attributes):
            return self.compute(inputs, dynamic_attributes)
    '''
)


class _DoublerOp(BaseOp):
    def warmup(self) -> "_DoublerOp":
        self._warmed_up = True
        return self

    def compute(
        self, inputs: list[Any], dynamic_attributes: dict[str, str], **kwargs: Any
    ) -> OpOutput:
        return OpOutput(value=[inputs[0] * 2], metadata={})

    async def compute_async(
        self, inputs: list[Any], dynamic_attributes: dict[str, str], **kwargs: Any
    ) -> OpOutput:
        return self.compute(inputs, dynamic_attributes, **kwargs)


def _op_instance() -> dict[str, Any]:
    tensor = {"dtype": "Array[float32]", "shape": [3]}
    return {
        "id": "doubler",
        "op": "Doubler",
        "inputs": [tensor],
        "outputs": [tensor],
        "attributes": {},
        "dynamic_attributes": {},
    }


class TestPyFuncOpInstanceValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name) / "pyfunc.fnnx"
        (root / "variant_artifacts").mkdir(parents=True)
        (root / "variant_artifacts" / "__pyfunc__.py").write_text(
            _ENTRY_MODULE, encoding="utf-8"
        )

        registry = Registry()
        registry.register_op(_DoublerOp, "Doubler")
        executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown)
        self.variant = PyFuncVariant(
            str(root),
            [_op_instance()],
            {"pyfunc_classname": "Doubler"},
            registry=registry,
            device_map=DeviceMap(accelerator="cpu", node_device_map={}),
            dtypes_manager=DtypesManager({}, BUILTINS),
            executor=executor,
            op_executor=executor,
        ).warmup()

    def test_conforming_input_reaches_the_op_instance(self) -> None:
        inputs = {"x": np.asarray([1, 2, 3], dtype=np.float32)}
        result = self.variant.compute(inputs, {})
        np.testing.assert_allclose(result["y"], [2, 4, 6])

    def test_input_that_disagrees_with_the_declaration_is_rejected(self) -> None:
        for name, value in (
            ("shape", np.asarray([1, 2], dtype=np.float32)),
            ("dtype", np.asarray([1, 2, 3], dtype=np.float64)),
        ):
            with self.subTest(declaration=name):
                with self.assertRaises(ValueError):
                    self.variant.compute({"x": value}, {})

    def test_a_value_that_is_no_array_at_all_is_rejected(self) -> None:
        # A plain list has no `.shape`; the dtype check has to reject it first.
        with self.assertRaisesRegex(ValueError, "Array\\[float32\\]"):
            self.variant.compute({"x": [1.0, 2.0, 3.0]}, {})

    def test_the_wrong_number_of_inputs_is_rejected(self) -> None:
        instance = self.variant.context.get_op_instance("doubler")
        value = np.asarray([1, 2, 3], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Expected 1 inputs, got 2"):
            instance.operator.compute([value, value], {})

    def test_warmup_does_not_hand_out_the_unvalidated_operator(self) -> None:
        instance = self.variant.context.get_op_instance("doubler")
        wrong_shape = np.asarray([1, 2], dtype=np.float32)
        with self.assertRaises(ValueError):
            instance.operator.warmup().compute([wrong_shape], {})


if __name__ == "__main__":
    unittest.main()
