from __future__ import annotations

import copy
from typing import Any, Iterable


JsonObject = dict[str, Any]
JsonPatchOp = dict[str, Any]
JsonPatch = list[JsonPatchOp]


def apply_patches(
    document: JsonObject, patch_documents: Iterable[JsonPatch]
) -> JsonObject:
    """
    Apply a sequence of RFC 6902 JSON Patch documents to a JSON object.

    Only "add" and "replace" operations are supported.

    Returns a deep-copied, patched document.
    """
    result = copy.deepcopy(document)
    for patch in patch_documents:
        _apply_patch(result, patch)
    return result


def _apply_patch(document: JsonObject, patch: JsonPatch) -> None:
    for op in patch:
        op_type = op.get("op")
        path = op.get("path")
        if op_type is None or path is None:
            raise ValueError(f"Invalid JSON Patch operation: {op!r}")

        if op_type == "add":
            _op_add(document, path, op.get("value"))
        elif op_type == "replace":
            _op_replace(document, path, op.get("value"))
        else:
            raise ValueError(
                f"Unsupported JSON Patch op: {op_type!r} (only 'add' and 'replace' are allowed)"
            )


def _decode_pointer_token(token: str) -> str:
    # RFC 6901: "~1" → "/", "~0" → "~"
    return token.replace("~1", "/").replace("~0", "~")


def _split_pointer(path: str) -> list[str]:
    if path == "":
        raise ValueError("Empty JSON Pointer path is not supported")
    if not path.startswith("/"):
        raise ValueError(
            f"Only absolute JSON Pointer paths are supported, got: {path!r}"
        )

    raw_tokens = path.lstrip("/").split("/")
    return [_decode_pointer_token(t) for t in raw_tokens if t != ""]


def _traverse_to_parent(doc: Any, path: str) -> tuple[Any, str]:
    tokens = _split_pointer(path)
    if not tokens:
        raise ValueError(f"Path {path!r} does not point to a child of the root")

    parent = doc
    for token in tokens[:-1]:
        if isinstance(parent, list):
            idx = _parse_index(token, len(parent))
            parent = parent[idx]
        elif isinstance(parent, dict):
            if token not in parent:
                raise KeyError(
                    f"Path segment {token!r} not found while traversing {path!r}"
                )
            parent = parent[token]
        else:
            raise TypeError(
                f"Cannot traverse into non-container type at segment {token!r}"
            )
    return parent, tokens[-1]


def _parse_index(token: str, max_len: int) -> int:
    try:
        idx = int(token)
    except ValueError:
        raise ValueError(f"Array index must be an integer, got {token!r}") from None
    if idx < 0 or idx >= max_len:
        raise IndexError(f"Array index {idx} out of range (len={max_len})")
    return idx


def _parse_index_for_add(token: str, max_len: int) -> int:
    try:
        idx = int(token)
    except ValueError:
        raise ValueError(f"Array index must be an integer, got {token!r}") from None
    if idx < 0 or idx > max_len:
        raise IndexError(f"Array index {idx} out of range for add (len={max_len})")
    return idx


def _op_add(doc: Any, path: str, value: Any) -> None:
    parent, token = _traverse_to_parent(doc, path)

    if isinstance(parent, list):
        if token == "-":
            parent.append(value)
            return
        idx = _parse_index_for_add(token, len(parent))
        parent.insert(idx, value)
        return

    if isinstance(parent, dict):
        parent[token] = value
        return

    raise TypeError(f"Cannot apply 'add' at {path!r}: parent is not a container")


def _op_replace(doc: Any, path: str, value: Any) -> None:
    parent, token = _traverse_to_parent(doc, path)

    if isinstance(parent, list):
        idx = _parse_index(token, len(parent))
        parent[idx] = value
        return

    if isinstance(parent, dict):
        if token not in parent:
            raise KeyError(
                f"Cannot 'replace' non-existent member {token!r} at {path!r}"
            )
        parent[token] = value
        return

    raise TypeError(f"Cannot apply 'replace' at {path!r}: parent is not a container")
