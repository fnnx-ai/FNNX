import json
import re
import tarfile
import warnings

from pydantic import ValidationError

from fnnx.extras.jsonpatcher import JsonObject, JsonPatch, apply_patches
from fnnx.extras.pydantic_models.envs import Python3_CondaPip
from fnnx.extras.pydantic_models.manifest import Manifest
from fnnx.extras.pydantic_models.meta import MetaEntry


class Reader:
    def __init__(self, model_path: str) -> None:
        with tarfile.open(model_path, "r:*") as tar:
            self.manifest: Manifest = Manifest(**self._load_manifest(tar))
            self.metadata: list[MetaEntry] = self._load_metadata(tar)
            self.env: JsonObject = json.loads(self._get_file(tar, "env.json"))

        self.pyenv: Python3_CondaPip | None = None
        if "python3::conda_pip" in self.env:
            self.pyenv = Python3_CondaPip(**self.env["python3::conda_pip"])

    def _load_manifest(self, tar: tarfile.TarFile) -> JsonObject:
        manifest_data: JsonObject = json.loads(self._get_file(tar, "manifest.json"))

        patch_pattern = re.compile(r"^manifest-[^/]+\.patch\.json$")
        patch_members = self._last_matching_members(tar, patch_pattern)
        patch_names = sorted(
            patch_members, key=lambda filename: filename.encode("utf-8")
        )
        if patch_names:
            patches: list[JsonPatch] = [
                json.loads(self._get_file(tar, patch_members[name]))
                for name in patch_names
            ]
            manifest_data = apply_patches(manifest_data, patches)

        return manifest_data

    def _load_metadata(self, tar: tarfile.TarFile) -> list[MetaEntry]:
        meta_pattern = re.compile(r"^meta(-[^/]+)?\.json$")
        meta_members = self._last_matching_members(tar, meta_pattern)
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

    def _last_matching_members(
        self, tar: tarfile.TarFile, pattern: re.Pattern[str]
    ) -> dict[str, tarfile.TarInfo]:
        return {
            member.name: member
            for member in tar.getmembers()
            if pattern.fullmatch(member.name)
        }

    def _get_file(self, tar: tarfile.TarFile, target: str | tarfile.TarInfo) -> str:
        member = tar.getmember(target) if isinstance(target, str) else target
        f = tar.extractfile(member)
        if not f:
            raise ValueError(f"Could not read `{member.name}`")
        return f.read().decode("utf-8")
