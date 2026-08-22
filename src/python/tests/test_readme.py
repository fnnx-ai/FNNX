import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


README_PATH = Path(__file__).parents[1] / "README.md"
PYTHON_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


class TestReadme(unittest.TestCase):
    def test_python_examples_compile(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        examples = PYTHON_FENCE.findall(readme)

        self.assertTrue(examples)
        for index, source in enumerate(examples, start=1):
            compile(source, f"README.md example {index}", "exec")

    def test_pyfunc_example_runs(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        example = next(
            source
            for source in PYTHON_FENCE.findall(readme)
            if "class Echo(PyFunc)" in source
        )

        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "build_echo.py"
            script.write_text(example, encoding="utf-8")
            environment = os.environ.copy()
            python_paths = [str(README_PATH.parent)]
            if existing_path := environment.get("PYTHONPATH"):
                python_paths.append(existing_path)
            environment["PYTHONPATH"] = os.pathsep.join(python_paths)
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=directory,
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "['hello']")
