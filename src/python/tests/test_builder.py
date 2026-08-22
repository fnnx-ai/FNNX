"""What the producer writes into the archive: an uncompressed ustar tar, with no header
member of its own."""

import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from fnnx.extras.builder import File

USTAR_MAGIC = b"ustar\x0000"


class TestBuilderTarFormat(unittest.TestCase):
    def _build(self, root: Path) -> Path:
        source = root / "payload"
        source.mkdir()
        (source / "model.bin").write_bytes(b"weights")
        # A fractional mtime is what a PAX writer carries in an extended header.
        os.utime(source / "model.bin", (1_700_000_000.5, 1_700_000_000.5))

        archive = root / "model.fnnx"
        tar = File(str(archive))
        tar.create_file("manifest.json", "{}")
        tar.copy(str(source), "ops_artifacts/op")
        tar.close()
        return archive

    def test_archive_is_ustar_without_pax_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._build(Path(directory))

            with open(archive, "rb") as handle:
                self.assertEqual(handle.read(512)[257:265], USTAR_MAGIC)

            with tarfile.open(archive, "r") as tar:
                names = tar.getnames()
                mtimes = [member.mtime for member in tar.getmembers()]

            self.assertNotIn("PaxHeader", " ".join(names))
            self.assertIn("ops_artifacts/op/model.bin", names)
            self.assertTrue(all(isinstance(mtime, int) for mtime in mtimes))

    def test_a_name_too_long_for_ustar_falls_back_to_a_pax_record(self) -> None:
        # A component over 100 bytes fits neither the name field nor a name/prefix split.
        long_name = "ops_artifacts/" + "n" * 120 + ".bin"
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "model.fnnx"
            tar = File(str(archive))
            tar.create_file(long_name, "weights")
            tar.close()

            with tarfile.open(archive, "r") as handle:
                self.assertIn(long_name, handle.getnames())


if __name__ == "__main__":
    unittest.main()
