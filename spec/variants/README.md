# Variants

A variant defines the execution semantics of an artifact. `manifest.variant` names it, and [`../core.md`](../core.md) defines both the dispatch rule and what a variant owns.

| Key | Document | Summary | Classification |
| --- | --- | --- | --- |
| `pipeline` | [pipeline.md](pipeline.md) | A directed acyclic graph of first-party operations. | Strong portability, Strong execution durability |
| `pyfunc` | [pyfunc.md](pyfunc.md) | A packaged Python function. | Weak portability, Weak execution durability |

## Adding a variant

A new variant gets a new key. An existing key never changes meaning ([`../compatibility.md`](../compatibility.md)). Name the document after the key: `<key>.md`.

Open the document with the header table below. Then define, at minimum: the `variant_config.json` schema, the artifact layout under `variant_artifacts/`, the execution semantics, the permitted content types, and an example.

```markdown
# The `<key>` variant

| Key | Added in | Updated in | Classification | Operations | Content types | Schema |
| --- | --- | --- | --- | --- | --- | --- |
| `<key>` | <spec version> | <spec version> | <portability>, <durability> | No \| ALL \| <list of admitted operations> | <list of permitted content types> | [`variant_<key>.json`](../schemas/variant_<key>.json) |
```

Also: add the variant to the table above, add its schema under [`../schemas/`](../schemas/), and add its classification to [`../compatibility.md`](../compatibility.md).
