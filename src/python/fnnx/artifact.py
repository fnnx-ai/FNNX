from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from os.path import join as pjoin
from typing import Any, Iterable

from fnnx.jsonpatcher import JsonObject, JsonPatch, apply_patches

MANIFEST_FILE = "manifest.json"
MANIFEST_PATCH_PATTERN = re.compile(r"^manifest-[^/]+\.patch\.json$")

# Root files a consumer may read as their empty value when they are absent.
OPTIONAL_ROOT_FILES: dict[str, Any] = {
    "dtypes.json": {},
    "env.json": {},
    "meta.json": [],
}


def manifest_patch_names(names: Iterable[str]) -> list[str]:
    """The manifest patch files among `names`, in the order the spec applies them."""
    return sorted(
        {name for name in names if MANIFEST_PATCH_PATTERN.fullmatch(name)},
        key=lambda name: name.encode("utf-8"),
    )


def load_root_json(root: str, filename: str) -> Any:
    path = pjoin(root, filename)
    if filename in OPTIONAL_ROOT_FILES and not os.path.isfile(path):
        return deepcopy(OPTIONAL_ROOT_FILES[filename])
    with open(path, "r") as f:
        return json.load(f)


def load_effective_manifest(root: str) -> JsonObject:
    """The manifest of an extracted artifact, with every root manifest patch applied."""
    manifest: JsonObject = load_root_json(root, MANIFEST_FILE)
    patch_names = manifest_patch_names(os.listdir(root))
    if not patch_names:
        return manifest
    patches: list[JsonPatch] = [load_root_json(root, name) for name in patch_names]
    return apply_patches(manifest, patches)


def io_specs_by_name(
    entries: list[dict[str, Any]], role: str
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry["name"]
        if name in specs:
            raise ValueError(
                f"Manifest declares {role} `{name}` more than once; "
                f"{role} names must be unique"
            )
        specs[name] = entry
    return specs
