"""Model loading, external-data resolution, and opset resolution for the C compiler."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
loader = pytest.importorskip("fnnx.extras.compilers.c.onnx.loader")

MODELS_DIR = Path(__file__).parent / "models"
LINREG_MODEL = (
    MODELS_DIR / "onnx_pipeline.fnnx" / "ops_artifacts" / "linreg" / "model.onnx"
)


def _onnx_domain_maximum(domain: str) -> int:
    """Highest opset ONNX itself reports for `domain`, independent of the loader."""
    return onnx.defs.C.schema_version_map()[domain][1]


def _identity_model(*opset_imports: tuple[str, int], ir_version: int | None = None):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid(domain, version)
            for domain, version in opset_imports
        ],
    )
    if ir_version is not None:
        model.ir_version = ir_version
    return model


def _external_data_model(values, location: str):
    tensor = onnx.numpy_helper.from_array(values, "w")
    onnx.external_data_helper.set_external_data(tensor, location=location)
    tensor.ClearField("raw_data")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["w"], ["y"])],
        "g",
        [],
        [
            onnx.helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, list(values.shape)
            )
        ],
        initializer=[tensor],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )


class LoadModelTest(unittest.TestCase):
    def test_loads_bundle_node_model_with_both_domains(self):
        imported = {
            entry.domain: entry.version
            for entry in onnx.load(str(LINREG_MODEL)).opset_import
        }
        self.assertEqual(set(imported), {"", "ai.onnx.ml"})

        loaded = loader.load_model(LINREG_MODEL)

        self.assertEqual(loaded.opsets, imported)
        self.assertEqual(
            loaded.opset_for("ai.onnx"), loaded.opset_for(loader.STANDARD_DOMAIN)
        )

    def test_accepts_in_memory_proto(self):
        loaded = loader.load_model(_identity_model(("", 21)))
        self.assertEqual(loaded.opsets, {loader.STANDARD_DOMAIN: 21})

    def test_missing_file_names_the_path(self):
        missing = MODELS_DIR / "does_not_exist.onnx"
        with self.assertRaises(CompileError) as ctx:
            loader.load_model(missing)
        self.assertIn(str(missing), str(ctx.exception))

    def test_unparseable_file_is_a_compile_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.onnx"
            path.write_bytes(b"this is not a protobuf")
            with self.assertRaises(CompileError) as ctx:
                loader.load_model(path)
            self.assertIn(str(path), str(ctx.exception))

    def test_ir_version_newer_than_installed_onnx(self):
        model = _identity_model(("", 21), ir_version=onnx.IR_VERSION + 1)
        with self.assertRaises(CompileError) as ctx:
            loader.load_model(model)
        message = str(ctx.exception)
        self.assertIn(f"IR version {onnx.IR_VERSION + 1}", message)
        self.assertIn(f"at most {onnx.IR_VERSION}", message)
        self.assertIn("upgrade", message.lower())


class ResolveOpsetsTest(unittest.TestCase):
    def test_domain_alias_is_normalized(self):
        opsets = loader.resolve_opsets(_identity_model(("ai.onnx", 17)))
        self.assertEqual(opsets, {loader.STANDARD_DOMAIN: 17})

    def test_ml_domain_is_supported(self):
        opsets = loader.resolve_opsets(_identity_model(("", 21), (loader.ML_DOMAIN, 1)))
        self.assertEqual(opsets[loader.ML_DOMAIN], 1)

    def test_repeated_consistent_import_is_accepted(self):
        opsets = loader.resolve_opsets(_identity_model(("", 17), ("ai.onnx", 17)))
        self.assertEqual(opsets, {loader.STANDARD_DOMAIN: 17})

    def test_conflicting_imports_for_one_domain(self):
        with self.assertRaises(CompileError) as ctx:
            loader.resolve_opsets(_identity_model(("", 17), ("ai.onnx", 18)))
        message = str(ctx.exception)
        self.assertIn("17", message)
        self.assertIn("18", message)

    def test_custom_domain_is_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            loader.resolve_opsets(_identity_model(("", 21), ("com.example.ops", 1)))
        self.assertIn("com.example.ops", str(ctx.exception))

    def test_opset_newer_than_installed_onnx(self):
        maximum = _onnx_domain_maximum(loader.STANDARD_DOMAIN)
        with self.assertRaises(CompileError) as ctx:
            loader.resolve_opsets(_identity_model(("", maximum + 1)))
        message = str(ctx.exception)
        self.assertIn(f"imports opset version {maximum + 1}", message)
        self.assertIn(f"at most version {maximum}", message)
        self.assertIn("upgrade", message.lower())

    def test_ml_opset_newer_than_installed_onnx(self):
        maximum = _onnx_domain_maximum(loader.ML_DOMAIN)
        with self.assertRaises(CompileError) as ctx:
            loader.resolve_opsets(
                _identity_model(("", 21), (loader.ML_DOMAIN, maximum + 1))
            )
        message = str(ctx.exception)
        self.assertIn(loader.ML_DOMAIN, message)
        self.assertIn(f"imports opset version {maximum + 1}", message)
        self.assertIn(f"at most version {maximum}", message)

    def test_invalid_opset_version(self):
        with self.assertRaises(CompileError):
            loader.resolve_opsets(_identity_model(("", 0)))

    def test_model_without_opset_imports(self):
        model = _identity_model(("", 21))
        del model.opset_import[:]
        with self.assertRaises(CompileError):
            loader.resolve_opsets(model)

    def test_opset_for_unimported_domain(self):
        loaded = loader.load_model(_identity_model(("", 21)))
        with self.assertRaises(CompileError) as ctx:
            loaded.opset_for(loader.ML_DOMAIN)
        self.assertIn(loader.ML_DOMAIN, str(ctx.exception))

    def test_max_supported_opset_matches_onnx_domain_version_map(self):
        for domain in loader.SUPPORTED_DOMAINS:
            self.assertEqual(
                loader.max_supported_opset(domain), _onnx_domain_maximum(domain)
            )


class ExternalDataTest(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(6, dtype=np.float32).reshape(2, 3)

    def test_external_tensor_is_embedded_from_the_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "w.bin").write_bytes(self.values.tobytes())
            path = Path(tmp) / "model.onnx"
            onnx.save_model(_external_data_model(self.values, "w.bin"), str(path))

            loaded = loader.load_model(path)

            initializer = loaded.model.graph.initializer[0]
            self.assertFalse(onnx.external_data_helper.uses_external_data(initializer))
            np.testing.assert_array_equal(
                onnx.numpy_helper.to_array(initializer), self.values
            )

    def test_base_dir_overrides_the_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights"
            weights.mkdir()
            (weights / "w.bin").write_bytes(self.values.tobytes())
            path = Path(tmp) / "model.onnx"
            onnx.save_model(_external_data_model(self.values, "w.bin"), str(path))

            loaded = loader.load_model(path, base_dir=weights)

            np.testing.assert_array_equal(
                onnx.numpy_helper.to_array(loaded.model.graph.initializer[0]),
                self.values,
            )

    def test_in_memory_proto_without_base_dir_names_the_tensor(self):
        with self.assertRaises(CompileError) as ctx:
            loader.load_model(_external_data_model(self.values, "w.bin"))
        self.assertIn("`w`", str(ctx.exception))

    def test_missing_external_file_names_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CompileError) as ctx:
                loader.load_model(
                    _external_data_model(self.values, "w.bin"), base_dir=tmp
                )
            self.assertIn(tmp, str(ctx.exception))

    def test_external_path_escaping_the_base_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "secret.bin").write_bytes(self.values.tobytes())
            inner = Path(tmp) / "artifacts"
            inner.mkdir()
            with self.assertRaises(CompileError):
                loader.load_model(
                    _external_data_model(self.values, "../secret.bin"), base_dir=inner
                )

    def test_source_proto_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "w.bin").write_bytes(self.values.tobytes())
            model = _external_data_model(self.values, "w.bin")

            loaded = loader.load_model(model, base_dir=tmp)

            self.assertTrue(
                onnx.external_data_helper.uses_external_data(model.graph.initializer[0])
            )
            self.assertFalse(
                onnx.external_data_helper.uses_external_data(
                    loaded.model.graph.initializer[0]
                )
            )


class OptionalDependencyTest(unittest.TestCase):
    def test_missing_onnx_package_raises_an_actionable_error(self):
        """A `ModuleNotFoundError` keeps `pytest.importorskip` skipping rather than erroring."""
        package = "fnnx.extras.compilers.c.onnx"
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == package or name.startswith(f"{package}.")
        }
        try:
            for name in saved:
                del sys.modules[name]
            with mock.patch("importlib.util.find_spec", return_value=None):
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    importlib.import_module(package)
            self.assertIn("fnnx[compiler]", str(ctx.exception))
        finally:
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
