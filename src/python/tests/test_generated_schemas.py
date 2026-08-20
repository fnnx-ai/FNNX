import importlib
import json
import tarfile
import unittest
from pathlib import Path

from fnnx.spec import schema

PYTHON_ROOT = Path(__file__).parents[1]
PYDANTIC_MODELS_ROOT = PYTHON_ROOT / "fnnx" / "extras" / "pydantic_models"
MODEL_FIXTURES_ROOT = Path(__file__).parent / "models"
# Build the names so the source scan does not match its own targets.
REMOVED_ATTRIBUTES = (
    "_".join(("requires", "ort", "extensions")),  # noqa: FLY002
    "_".join(("use", "onnxruntime", "extensions")),  # noqa: FLY002
)


class TestGeneratedSchemas(unittest.TestCase):
    def test_every_copied_pydantic_module_imports(self) -> None:
        for path in sorted(PYDANTIC_MODELS_ROOT.rglob("*.py")):
            relative_module = path.relative_to(PYTHON_ROOT).with_suffix("")
            module_name = ".".join(relative_module.parts)
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_schema_and_fixtures_do_not_use_removed_attributes(self) -> None:
        self.assertEqual(schema["version"], "0.1.0")

        roots = (
            PYTHON_ROOT / "fnnx",
            PYTHON_ROOT / "examples",
            Path(__file__).parent,
        )
        for root in roots:
            for path in root.rglob("*"):
                if path.suffix not in {".json", ".py"}:
                    continue
                contents = path.read_text(encoding="utf-8")
                for attribute in REMOVED_ATTRIBUTES:
                    with self.subTest(path=path, attribute=attribute):
                        self.assertNotIn(attribute, contents)

        tar_path = MODEL_FIXTURES_ROOT / "onnx_pipeline.fnnx.tar"
        with tarfile.open(tar_path, "r") as archive:
            ops_file = archive.extractfile("ops.json")
            if ops_file is None:
                self.fail("The tar fixture does not contain ops.json")
            ops = json.load(ops_file)

        for op in ops:
            for attribute in REMOVED_ATTRIBUTES:
                with self.subTest(op=op["id"], attribute=attribute):
                    self.assertNotIn(attribute, op["attributes"])


if __name__ == "__main__":
    unittest.main()
