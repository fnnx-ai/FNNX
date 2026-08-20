# Operations

An operation type defines the execution contract of an op instance.

| Key | Document | Summary |
| --- | --- | --- |
| `ONNX_v1` | [onnx_v1.md](onnx_v1.md) | Executes an ONNX model.|

## Adding an operation

[`../compatibility.md`](../compatibility.md) states the requirements a new first-party operation must meet. Do not add an operation for coverage or symmetry. Name the document after the key, lowercased: `<key_lowercase>.md`.

Open the document with the header table below. Then define, at minimum: the layout under `ops_artifacts/<op_instance_id>/`, the `attributes` fields, the binding of instance inputs and outputs, the dynamic attributes (or their absence), and an example.

```markdown
# The `<Key>` operation

| Key | Added in | Updated in | Schema |
| --- | --- | --- | --- |
| `<Key>` | <spec version> | <spec version> | [`op_<key_lowercase>.json`](../schemas/op_<key_lowercase>.json) |
```

Also: add the operation to the table above, and add its schema under [`../schemas/`](../schemas/).
