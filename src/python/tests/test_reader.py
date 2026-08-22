import unittest
import tarfile
import json
import tempfile
import os
from io import BytesIO

from fnnx.extras.reader import Reader


def _add_tar_member(tar: tarfile.TarFile, name: str, content: object) -> None:
    data = (
        content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
    )
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, fileobj=BytesIO(data))


def _create_tar_with_members(
    manifest: dict[str, object], members: list[tuple[str, object]]
) -> str:
    fd, tar_path = tempfile.mkstemp(suffix=".tar")
    os.close(fd)

    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "manifest.json", manifest)
        _add_tar_member(tar, "env.json", {})
        for name, content in members:
            _add_tar_member(tar, name, content)

    return tar_path


class TestReaderManifestPatching(unittest.TestCase):
    def _create_test_tar(self, manifest, patches=None, env=None, metadata=None):
        """Helper to create a test tar file with manifest and optional patches."""
        if env is None:
            env = {}
        if metadata is None:
            metadata = []

        # Create a temporary tar file
        fd, tar_path = tempfile.mkstemp(suffix=".tar")
        os.close(fd)

        with tarfile.open(tar_path, "w") as tar:
            # Add manifest.json
            manifest_info = tarfile.TarInfo(name="manifest.json")
            manifest_bytes = json.dumps(manifest).encode("utf-8")
            manifest_info.size = len(manifest_bytes)
            tar.addfile(manifest_info, fileobj=BytesIO(manifest_bytes))

            # Add patch files if provided
            if patches:
                for i, patch in enumerate(patches):
                    patch_name = f"manifest-{i:04d}.patch.json"
                    patch_info = tarfile.TarInfo(name=patch_name)
                    patch_bytes = json.dumps(patch).encode("utf-8")
                    patch_info.size = len(patch_bytes)
                    tar.addfile(patch_info, fileobj=BytesIO(patch_bytes))

            # Add env.json
            env_info = tarfile.TarInfo(name="env.json")
            env_bytes = json.dumps(env).encode("utf-8")
            env_info.size = len(env_bytes)
            tar.addfile(env_info, fileobj=BytesIO(env_bytes))

            # Add metadata if provided
            if metadata:
                meta_info = tarfile.TarInfo(name="meta-0.json")
                meta_bytes = json.dumps(metadata).encode("utf-8")
                meta_info.size = len(meta_bytes)
                tar.addfile(meta_info, fileobj=BytesIO(meta_bytes))

        return tar_path

    def _base_manifest(self):
        """Create a minimal valid manifest for testing."""
        return {
            "variant": "test_variant",
            "name": "test_model",
            "version": "1.0.0",
            "description": "Test description",
            "producer_name": "test_producer",
            "producer_version": "1.0",
            "producer_tags": ["test"],
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        }

    def test_load_manifest_without_patches(self):
        manifest = self._base_manifest()
        tar_path = self._create_test_tar(manifest)

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.name, "test_model")
            self.assertEqual(reader.manifest.version, "1.0.0")
            self.assertEqual(reader.manifest.variant, "test_variant")
        finally:
            os.unlink(tar_path)

    def test_load_manifest_with_single_patch(self):
        manifest = self._base_manifest()
        patches = [[{"op": "replace", "path": "/version", "value": "2.0.0"}]]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.version, "2.0.0")
            self.assertEqual(reader.manifest.name, "test_model")
        finally:
            os.unlink(tar_path)

    def test_load_manifest_with_multiple_patches(self):
        manifest = self._base_manifest()
        patches = [
            [{"op": "replace", "path": "/version", "value": "2.0.0"}],
            [{"op": "replace", "path": "/description", "value": "Updated description"}],
            [{"op": "add", "path": "/producer_tags/-", "value": "patched"}],
        ]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.version, "2.0.0")
            self.assertEqual(reader.manifest.description, "Updated description")
            self.assertIn("patched", reader.manifest.producer_tags)
        finally:
            os.unlink(tar_path)

    def test_patches_applied_in_order(self):
        manifest = self._base_manifest()
        patches = [
            [{"op": "replace", "path": "/version", "value": "2.0.0"}],
            [{"op": "replace", "path": "/version", "value": "3.0.0"}],
        ]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            # Second patch should win
            self.assertEqual(reader.manifest.version, "3.0.0")
        finally:
            os.unlink(tar_path)

    def test_patches_apply_in_byte_order_not_tar_order(self) -> None:
        manifest = self._base_manifest()
        tar_path = _create_tar_with_members(
            manifest,
            [
                (
                    "manifest-b.patch.json",
                    [{"op": "replace", "path": "/version", "value": "from-b"}],
                ),
                (
                    "manifest-a.patch.json",
                    [{"op": "replace", "path": "/version", "value": "from-a"}],
                ),
            ],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.version, "from-b")
        finally:
            os.unlink(tar_path)

    def test_repeated_manifest_and_patch_use_only_last_occurrence(self) -> None:
        first_manifest = self._base_manifest()
        first_manifest["name"] = "first"
        last_manifest = self._base_manifest()
        last_manifest["name"] = "last"
        tar_path = _create_tar_with_members(
            first_manifest,
            [
                (
                    "manifest-repeated.patch.json",
                    [{"op": "add", "path": "/producer_tags/-", "value": "first"}],
                ),
                ("manifest.json", last_manifest),
                (
                    "manifest-repeated.patch.json",
                    [{"op": "add", "path": "/producer_tags/-", "value": "last"}],
                ),
            ],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.name, "last")
            self.assertEqual(reader.manifest.producer_tags, ["test", "last"])
        finally:
            os.unlink(tar_path)

    def test_patch_adds_new_field(self):
        manifest = self._base_manifest()
        manifest["name"] = None  # Start with None
        patches = [[{"op": "add", "path": "/name", "value": "patched_name"}]]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.name, "patched_name")
        finally:
            os.unlink(tar_path)

    def test_patch_modifies_array(self):
        manifest = self._base_manifest()
        patches = [
            [{"op": "add", "path": "/producer_tags/-", "value": "tag1"}],
            [{"op": "add", "path": "/producer_tags/-", "value": "tag2"}],
        ]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            self.assertIn("test", reader.manifest.producer_tags)
            self.assertIn("tag1", reader.manifest.producer_tags)
            self.assertIn("tag2", reader.manifest.producer_tags)
            self.assertEqual(len(reader.manifest.producer_tags), 3)
        finally:
            os.unlink(tar_path)

    def test_multiple_operations_in_single_patch(self):
        manifest = self._base_manifest()
        patches = [
            [
                {"op": "replace", "path": "/version", "value": "2.0.0"},
                {"op": "replace", "path": "/description", "value": "Multi-op patch"},
                {"op": "add", "path": "/producer_tags/-", "value": "multi"},
            ]
        ]

        tar_path = self._create_test_tar(manifest, patches)

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.version, "2.0.0")
            self.assertEqual(reader.manifest.description, "Multi-op patch")
            self.assertIn("multi", reader.manifest.producer_tags)
        finally:
            os.unlink(tar_path)


class TestReaderMetadataLoading(unittest.TestCase):
    def _create_test_tar_with_files(self, manifest, files_dict, env=None):
        """Helper to create a tar file with arbitrary files.

        Args:
            manifest: The manifest dict
            files_dict: Dict mapping filename -> content (as dict for JSON files)
            env: Optional env dict
        """
        if env is None:
            env = {}

        fd, tar_path = tempfile.mkstemp(suffix=".tar")
        os.close(fd)

        with tarfile.open(tar_path, "w") as tar:
            # Add manifest.json
            manifest_info = tarfile.TarInfo(name="manifest.json")
            manifest_bytes = json.dumps(manifest).encode("utf-8")
            manifest_info.size = len(manifest_bytes)
            tar.addfile(manifest_info, fileobj=BytesIO(manifest_bytes))

            # Add env.json
            env_info = tarfile.TarInfo(name="env.json")
            env_bytes = json.dumps(env).encode("utf-8")
            env_info.size = len(env_bytes)
            tar.addfile(env_info, fileobj=BytesIO(env_bytes))

            # Add custom files
            for filename, content in files_dict.items():
                file_info = tarfile.TarInfo(name=filename)
                file_bytes = json.dumps(content).encode("utf-8")
                file_info.size = len(file_bytes)
                tar.addfile(file_info, fileobj=BytesIO(file_bytes))

        return tar_path

    def _base_manifest(self):
        """Create a minimal valid manifest for testing."""
        return {
            "variant": "test_variant",
            "name": "test_model",
            "version": "1.0.0",
            "description": "Test description",
            "producer_name": "test_producer",
            "producer_version": "1.0",
            "producer_tags": ["test"],
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        }

    def _meta_entry(self, entry_id, payload_key):
        """Helper to create a valid MetaEntry dict."""
        return {
            "id": entry_id,
            "producer": "test_producer",
            "producer_version": "1.0",
            "producer_tags": ["test"],
            "payload": {"key": payload_key},
        }

    def test_load_meta_json(self):
        manifest = self._base_manifest()
        files = {
            "meta.json": [
                self._meta_entry("entry1", "value1"),
                self._meta_entry("entry2", "value2"),
            ]
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            self.assertEqual(len(reader.metadata), 2)
        finally:
            os.unlink(tar_path)

    def test_load_meta_with_uid(self):
        manifest = self._base_manifest()
        files = {
            "meta-abc123.json": [self._meta_entry("entry1", "value1")],
            "meta-xyz789.json": [self._meta_entry("entry2", "value2")],
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            self.assertEqual(len(reader.metadata), 2)
        finally:
            os.unlink(tar_path)

    def test_ignore_meta_subdirectory_files(self):
        manifest = self._base_manifest()
        files = {
            "meta.json": [self._meta_entry("valid", "valid")],
            "meta_artifacts/file.json": [self._meta_entry("ignored", "should_ignore")],
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            # Should only load from meta.json, not from meta_artifacts/file.json
            self.assertEqual(len(reader.metadata), 1)
            self.assertEqual(reader.metadata[0].id, "valid")
        finally:
            os.unlink(tar_path)

    def test_ignore_metadata_json(self):
        manifest = self._base_manifest()
        files = {
            "meta.json": [self._meta_entry("valid", "valid")],
            "metadata.json": [self._meta_entry("ignored", "should_ignore")],
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            # Should only load from meta.json, not metadata.json
            self.assertEqual(len(reader.metadata), 1)
            self.assertEqual(reader.metadata[0].id, "valid")
        finally:
            os.unlink(tar_path)

    def test_ignore_meta_with_wrong_extension(self):
        manifest = self._base_manifest()
        files = {
            "meta.json": [self._meta_entry("valid", "valid")],
            "meta.txt": [self._meta_entry("ignored", "should_ignore")],
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            # Should only load .json files
            self.assertEqual(len(reader.metadata), 1)
            self.assertEqual(reader.metadata[0].id, "valid")
        finally:
            os.unlink(tar_path)

    def test_load_multiple_meta_files(self):
        manifest = self._base_manifest()
        files = {
            "meta.json": [self._meta_entry("from_base", "from_base")],
            "meta-uid1.json": [
                self._meta_entry("from_uid1_a", "from_uid1_a"),
                self._meta_entry("from_uid1_b", "from_uid1_b"),
            ],
            "meta-uid2.json": [self._meta_entry("from_uid2", "from_uid2")],
        }
        tar_path = self._create_test_tar_with_files(manifest, files)

        try:
            reader = Reader(tar_path)
            # Should load all entries from all valid meta files
            self.assertEqual(len(reader.metadata), 4)
            ids = {m.id for m in reader.metadata}
            self.assertEqual(
                ids, {"from_base", "from_uid1_a", "from_uid1_b", "from_uid2"}
            )
        finally:
            os.unlink(tar_path)

    def test_metadata_uses_defined_file_order(self) -> None:
        manifest = self._base_manifest()
        tar_path = _create_tar_with_members(
            manifest,
            [
                ("meta-b.json", [self._meta_entry("from_b", "from_b")]),
                ("meta.json", [self._meta_entry("from_base", "from_base")]),
                ("meta-a.json", [self._meta_entry("from_a", "from_a")]),
            ],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual(
                [entry.id for entry in reader.metadata],
                ["from_base", "from_a", "from_b"],
            )
        finally:
            os.unlink(tar_path)

    def test_repeated_metadata_member_uses_only_last_occurrence(self) -> None:
        manifest = self._base_manifest()
        tar_path = _create_tar_with_members(
            manifest,
            [
                ("meta-x.json", [self._meta_entry("first", "first")]),
                ("meta-x.json", [self._meta_entry("last", "last")]),
            ],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual([entry.id for entry in reader.metadata], ["last"])
        finally:
            os.unlink(tar_path)

    def test_bad_metadata_does_not_abort_remaining_entries_or_files(self) -> None:
        manifest = self._base_manifest()
        missing_producer = self._meta_entry("invalid", "invalid")
        del missing_producer["producer"]
        tar_path = _create_tar_with_members(
            manifest,
            [
                (
                    "meta-a.json",
                    [missing_producer, self._meta_entry("valid_a", "valid_a")],
                ),
                ("meta-b.json", b"not valid JSON"),
                ("meta-c.json", [self._meta_entry("valid_c", "valid_c")]),
            ],
        )

        try:
            with self.assertWarnsRegex(UserWarning, "meta-b.json"):
                reader = Reader(tar_path)
            self.assertEqual(
                [entry.id for entry in reader.metadata], ["valid_a", "valid_c"]
            )
        finally:
            os.unlink(tar_path)

    def test_unknown_entry_keys_are_preserved(self) -> None:
        manifest = self._base_manifest()
        entry = self._meta_entry("with_extras", "value")
        entry["seen_by_a_later_revision"] = {"kept": True}
        tar_path = self._create_test_tar_with_files(manifest, {"meta.json": [entry]})

        try:
            reader = Reader(tar_path)
            self.assertEqual(
                reader.metadata[0].model_dump()["seen_by_a_later_revision"],
                {"kept": True},
            )
        finally:
            os.unlink(tar_path)


class TestReaderMemberSelection(unittest.TestCase):
    def _base_manifest(self):
        return {
            "variant": "test_variant",
            "name": "test_model",
            "producer_name": "test_producer",
            "producer_version": "1.0",
            "producer_tags": [],
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        }

    def _meta_entry(self, entry_id: str) -> dict[str, object]:
        return {
            "id": entry_id,
            "producer": "test_producer",
            "producer_version": "1.0",
            "producer_tags": [],
            "payload": {},
        }

    def _create_tar(self, members: list[tuple[str, object]], links=()) -> str:
        fd, tar_path = tempfile.mkstemp(suffix=".tar")
        os.close(fd)
        with tarfile.open(tar_path, "w") as tar:
            _add_tar_member(tar, "manifest.json", self._base_manifest())
            for name, content in members:
                _add_tar_member(tar, name, content)
            for name, target in links:
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
        return tar_path

    def test_missing_env_json_is_read_as_empty(self) -> None:
        tar_path = self._create_tar([])

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.env, {})
            self.assertIsNone(reader.pyenv)
        finally:
            os.unlink(tar_path)

    def test_metadata_declared_as_a_link_is_ignored(self) -> None:
        tar_path = self._create_tar(
            [("meta.json", [self._meta_entry("from_meta")])],
            links=[("meta-linked.json", "meta.json")],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual([entry.id for entry in reader.metadata], ["from_meta"])
        finally:
            os.unlink(tar_path)

    def test_manifest_patch_declared_as_a_link_is_ignored(self) -> None:
        tar_path = self._create_tar(
            [("patch-source.json", [{"op": "replace", "path": "/name", "value": "x"}])],
            links=[("manifest-linked.patch.json", "patch-source.json")],
        )

        try:
            reader = Reader(tar_path)
            self.assertEqual(reader.manifest.name, "test_model")
        finally:
            os.unlink(tar_path)


if __name__ == "__main__":
    unittest.main()
