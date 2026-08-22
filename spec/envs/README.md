# Environment kinds

An environment kind declares how to reconstruct an execution environment for an artifact.

| Key | Document | Summary |
| --- | --- | --- |
| `python3::conda_pip` | [python3_conda_pip.md](python3_conda_pip.md) | A Python environment from a Python version, conda build dependencies and pip dependencies. |

## Adding an environment kind

A new environment kind gets a new key. An existing key never changes meaning ([`../compatibility.md`](../compatibility.md)). Name the document after the key, with `::` replaced by `_`: `python3::conda_pip` → `python3_conda_pip.md`.

Open the document with the header table below. Then define, at minimum: the declaration fields, any selection or matching rules, what providers may or may not honour, and an example.

```markdown
# The `<key>` environment kind

| Key | Added in | Updated in | Schema |
| --- | --- | --- | --- |
| `<key>` | <spec version> | <spec version> | [`env.json`](../schemas/env.json) |
```

Also: add the kind to the table above, and add its schema to [`env.json`](../schemas/env.json), which maps each kind to its schema.
