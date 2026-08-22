import json
import re
import tarfile
import warnings

from pydantic import ValidationError

from fnnx.artifact import (
    MANIFEST_FILE,
    MANIFEST_PATCH_PATTERN,
    manifest_patch_names,
)
from fnnx.extras.pydantic_models.envs import Python3_CondaPip
from fnnx.extras.pydantic_models.manifest import Manifest
from fnnx.extras.pydantic_models.meta import MetaEntry
from fnnx.jsonpatcher import JsonObject, JsonPatch, apply_patches


class Reader:
    def __init__(self, model_path: str) -> None:
        with tarfile.open(model_path, "r:*") as tar:
            self.manifest: Manifest = Manifest(**self._load_manifest(tar))
            self.metadata: list[MetaEntry] = self._load_metadata(tar)
            env = self._read_member(tar, "env.json")
            self.env: JsonObject = json.loads(env) if env is not None else {}

        self.pyenv: Python3_CondaPip | None = None
        if "python3::conda_pip" in self.env:
            self.pyenv = Python3_CondaPip(**self.env["python3::conda_pip"])

    def _load_manifest(self, tar: tarfile.TarFile) -> JsonObject:
        manifest_data: JsonObject = json.loads(self._get_file(tar, MANIFEST_FILE))

        patch_members = self._matching_members(tar, MANIFEST_PATCH_PATTERN)
        patch_names = manifest_patch_names(patch_members)
        if patch_names:
            patches: list[JsonPatch] = [
                json.loads(self._get_file(tar, patch_members[name]))
                for name in patch_names
            ]
            manifest_data = apply_patches(manifest_data, patches)

        return manifest_data

    def _load_metadata(self, tar: tarfile.TarFile) -> list[MetaEntry]:
        meta_pattern = re.compile(r"^meta(-[^/]+)?\.json$")
        meta_members = self._matching_members(tar, meta_pattern)
        sidecar_names = sorted(
            (name for name in meta_members if name != "meta.json"),
            key=lambda filename: filename.encode("utf-8"),
        )
        meta_names = (
            ["meta.json"] if "meta.json" in meta_members else []
        ) + sidecar_names

        metadata: list[MetaEntry] = []
        for name in meta_names:
            try:
                entries = json.loads(self._get_file(tar, meta_members[name]))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                warnings.warn(
                    f"Ignoring unparseable metadata file `{name}`: {error}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if not isinstance(entries, list):
                warnings.warn(
                    f"Ignoring metadata file `{name}` because it is not a JSON array",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for entry in entries:
                try:
                    metadata.append(MetaEntry.model_validate(entry))
                except ValidationError:
                    continue

        return metadata

    def _matching_members(
        self, tar: tarfile.TarFile, pattern: re.Pattern[str]
    ) -> dict[str, tarfile.TarInfo]:
        """The last regular-file member under each matching name.

        Links are never read through, and a repeated name resolves to its last occurrence.
        """
        return {
            member.name: member
            for member in tar.getmembers()
            if member.isfile() and pattern.fullmatch(member.name)
        }

    def _find_member(self, tar: tarfile.TarFile, name: str) -> tarfile.TarInfo | None:
        found: tarfile.TarInfo | None = None
        for member in tar.getmembers():
            if member.name == name and member.isfile():
                found = member
        return found

    def _read_member(self, tar: tarfile.TarFile, name: str) -> str | None:
        member = self._find_member(tar, name)
        return None if member is None else self._get_file(tar, member)

    def _get_file(self, tar: tarfile.TarFile, target: str | tarfile.TarInfo) -> str:
        if isinstance(target, str):
            member = self._find_member(tar, target)
            if member is None:
                raise ValueError(f"Could not read `{target}`")
        else:
            member = target
        f = tar.extractfile(member)
        if not f:
            raise ValueError(f"Could not read `{member.name}`")
        return f.read().decode("utf-8")
