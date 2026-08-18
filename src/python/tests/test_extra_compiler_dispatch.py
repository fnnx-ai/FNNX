"""Kernel-registry dispatch, the semantic-revision guard, and the unsupported-op error.

Expected dispatch behaviour is derived from the ONNX schema registry itself, so the
tests stay valid as the installed `onnx` package (and its op revisions) change.
"""

from __future__ import annotations

import unittest

import pytest

from fnnx.extras.compilers.c.errors import CompileError

onnx = pytest.importorskip("onnx")
registry_module = pytest.importorskip("fnnx.extras.compilers.c.onnx.registry")

KernelRegistry = registry_module.KernelRegistry
latest_semantic_revision = registry_module.latest_semantic_revision

ML_DOMAIN = "ai.onnx.ml"


def _revisions(op_type: str, domain: str = "") -> list[int]:
    """Opset versions at which ONNX revised `op_type`, oldest first."""
    return sorted(
        schema.since_version
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.name == op_type and schema.domain == domain
    )


ADD_REVISIONS = _revisions("Add")
PREVIOUS_ADD_REVISION, LATEST_ADD_REVISION = ADD_REVISIONS[-2:]


class SelectTest(unittest.TestCase):
    def setUp(self):
        self.registry: KernelRegistry[str] = KernelRegistry()

    def test_selects_highest_version_at_or_below_the_requested_opset(self):
        self.registry.register("", "Add", PREVIOUS_ADD_REVISION, "old")
        self.registry.register("", "Add", LATEST_ADD_REVISION, "new")

        selected = self.registry.select("", "Add", LATEST_ADD_REVISION)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.since_version, LATEST_ADD_REVISION)
        self.assertEqual(selected.generator, "new")

    def test_older_kernel_serves_opsets_before_the_next_revision(self):
        self.registry.register("", "Add", PREVIOUS_ADD_REVISION, "old")
        self.registry.register("", "Add", LATEST_ADD_REVISION, "new")

        selected = self.registry.select("", "Add", LATEST_ADD_REVISION - 1)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.since_version, PREVIOUS_ADD_REVISION)

    def test_semantic_revision_guard_rejects_stale_kernel(self):
        self.registry.register("", "Add", PREVIOUS_ADD_REVISION, "old")

        self.assertIsNone(self.registry.select("", "Add", LATEST_ADD_REVISION))

    def test_unregistered_op_selects_nothing(self):
        self.assertIsNone(self.registry.select("", "Add", LATEST_ADD_REVISION))

    def test_kernel_newer_than_the_requested_opset_selects_nothing(self):
        self.registry.register("", "Add", LATEST_ADD_REVISION, "new")

        self.assertIsNone(self.registry.select("", "Add", LATEST_ADD_REVISION - 1))

    def test_domain_alias_is_normalized_on_both_sides(self):
        self.registry.register("ai.onnx", "Add", LATEST_ADD_REVISION, "new")

        by_empty = self.registry.select("", "Add", LATEST_ADD_REVISION)
        by_alias = self.registry.select("ai.onnx", "Add", LATEST_ADD_REVISION)

        self.assertEqual(by_empty, by_alias)
        self.assertIsNotNone(by_empty)
        self.assertEqual(
            self.registry.registered_versions("", "Add"), [LATEST_ADD_REVISION]
        )

    def test_ml_domain_dispatch(self):
        since = _revisions("Scaler", ML_DOMAIN)[-1]
        self.registry.register(ML_DOMAIN, "Scaler", since, "scaler")

        selected = self.registry.select(ML_DOMAIN, "Scaler", since)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.generator, "scaler")
        self.assertIsNone(self.registry.select("", "Scaler", since))

    def test_registered_versions_are_sorted(self):
        for version in reversed(ADD_REVISIONS):
            self.registry.register("", "Add", version, f"add{version}")

        self.assertEqual(self.registry.registered_versions("", "Add"), ADD_REVISIONS)


class RegisterTest(unittest.TestCase):
    def setUp(self):
        self.registry: KernelRegistry[str] = KernelRegistry()

    def test_unknown_op_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register("", "NotAnOnnxOp", 1, "kernel")

    def test_version_before_the_op_existed_is_rejected(self):
        introduced = _revisions("Add")[0]
        with self.assertRaises(ValueError):
            self.registry.register("", "Add", introduced - 1, "kernel")

    def test_duplicate_registration_is_rejected(self):
        self.registry.register("", "Add", LATEST_ADD_REVISION, "first")
        with self.assertRaises(ValueError):
            self.registry.register("ai.onnx", "Add", LATEST_ADD_REVISION, "second")


class LatestSemanticRevisionTest(unittest.TestCase):
    def test_matches_the_schema_history(self):
        for version in range(ADD_REVISIONS[0], LATEST_ADD_REVISION + 2):
            expected = max((r for r in ADD_REVISIONS if r <= version), default=None)
            self.assertEqual(latest_semantic_revision("", "Add", version), expected)

    def test_unknown_op_has_no_revision(self):
        self.assertIsNone(latest_semantic_revision("", "NotAnOnnxOp", 1))


class UnsupportedOpErrorTest(unittest.TestCase):
    def setUp(self):
        self.registry: KernelRegistry[str] = KernelRegistry()

    def test_names_op_domain_version_and_nearest_supported_version(self):
        self.registry.register("", "Add", PREVIOUS_ADD_REVISION, "old")

        error = self.registry.unsupported_op_error("", "Add", LATEST_ADD_REVISION)

        self.assertIsInstance(error, CompileError)
        message = str(error)
        self.assertIn("Add", message)
        self.assertIn("ai.onnx", message)
        self.assertIn(str(LATEST_ADD_REVISION), message)
        self.assertIn(f"Nearest supported version: {PREVIOUS_ADD_REVISION}", message)

    def test_nearest_supported_version_can_be_above_the_requested_opset(self):
        self.registry.register("", "Add", ADD_REVISIONS[0], "oldest")
        self.registry.register("", "Add", LATEST_ADD_REVISION, "newest")

        message = str(
            self.registry.unsupported_op_error("", "Add", PREVIOUS_ADD_REVISION)
        )

        self.assertIn(f"Nearest supported version: {LATEST_ADD_REVISION}", message)

    def test_nearest_supported_version_can_be_below_the_requested_opset(self):
        older_revision, requested = ADD_REVISIONS[1], ADD_REVISIONS[2]
        self.registry.register("", "Add", older_revision, "older")
        self.registry.register("", "Add", LATEST_ADD_REVISION, "newest")

        message = str(self.registry.unsupported_op_error("", "Add", requested))

        self.assertIn(f"Nearest supported version: {older_revision}", message)

    def test_reports_when_no_kernel_is_registered(self):
        message = str(
            self.registry.unsupported_op_error("", "Add", LATEST_ADD_REVISION)
        )

        self.assertIn("Add", message)
        self.assertIn("ai.onnx", message)
        self.assertIn(str(LATEST_ADD_REVISION), message)
        self.assertIn("no kernel is registered", message)

    def test_reports_kernels_that_are_all_newer(self):
        self.registry.register("", "Add", LATEST_ADD_REVISION, "new")

        message = str(self.registry.unsupported_op_error("", "Add", ADD_REVISIONS[0]))

        self.assertIn("newer opset version", message)
        self.assertIn(f"Nearest supported version: {LATEST_ADD_REVISION}", message)

    def test_names_the_node(self):
        message = str(
            self.registry.unsupported_op_error(
                "", "Add", LATEST_ADD_REVISION, node_name="adder"
            )
        )

        self.assertIn("adder", message)


if __name__ == "__main__":
    unittest.main()
