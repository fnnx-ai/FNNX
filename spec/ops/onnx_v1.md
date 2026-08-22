# The `ONNX_v1` operation

| Key | Added in | Updated in | Schema |
| --- | --- | --- | --- |
| `ONNX_v1` | 0.1.0 | 0.1.0 | [`op_onnx_v1.json`](../schemas/op_onnx_v1.json) |

`ONNX_v1` executes an ONNX model. The operation adds no computational semantics of its own. The [ONNX specification](https://onnx.ai/onnx/) is the execution contract, and an `ONNX_v1` op instance means what the referenced ONNX model means under that contract.

## Artifact layout

The ONNX model of an op instance is stored at a fixed path derived from the instance's `id`:

```
ops_artifacts/<op_instance_id>/model.onnx
```

The file MUST be a serialized ONNX `ModelProto`. The name `model.onnx` is fixed. Consumers locate the model by this path, and MUST NOT search the directory for an alternative. Additional files MAY be present in the same directory. See external data below.

## Attributes

The op instance's `attributes` object carries the operation's compatibility declarations. A consumer reads them to decide whether it can execute the op instance, without parsing the ONNX model.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `opsets` | array of objects | yes | The operator sets the model imports. Each entry has `domain` (string) and `version` (integer), mirroring the model's opset imports. The default ONNX domain is written as `ai.onnx`. |
| `has_external_data` | boolean | yes | Whether tensor data is stored outside `model.onnx`. |
| `onnx_ir_version` | integer | yes | The `ir_version` of the serialized `ModelProto`. |
| `used_operators` | object or null | no | Map from domain to the list of operator type names the model uses. Absent or `null` when not declared. |

Producers MUST write these declarations to agree with the model file. A consumer that cannot satisfy a declaration MUST decline the artifact with an identifiable reason. It MUST NOT attempt execution. Unsatisfiable declarations include an unsupported opset version, an unimplemented IR version, and a missing operator domain. `used_operators` is optional because deriving it requires a graph walk. When present, it lets a consumer check operator coverage from the declaration alone.

## Operator set domains

The standard domains are `ai.onnx` and `ai.onnx.ml`. The ONNX specification defines their operators.

A producer MAY declare another domain in `opsets`. The owner of that domain defines its operator semantics. Consumer support for a non-standard domain is implementation-specific. Producers SHOULD NOT declare one unless the model cannot be expressed without it. [`../compatibility.md`](../compatibility.md) states the effect on the classification.

## Tensor binding

The op instance's `inputs` and `outputs` entries correspond to the inputs and outputs of the ONNX graph in order. The *i*-th entry describes the *i*-th graph input or output. The tensor names inside the ONNX graph take no part in binding. A consumer MUST bind by position and MUST NOT match on names.

The number of entries in `inputs` MUST equal the number of graph inputs the model requires at execution. The number of entries in `outputs` MUST equal the number of graph outputs. A consumer MUST reject an op instance whose arity disagrees with the model.

*Note: ONNX permits a graph input to also appear as an initializer. The value is then optional, and the initializer supplies it when nothing is fed. Such inputs are not part of the bound sequence unless the producer intends them to be supplied at execution.*

## Declared dtypes and shapes

Each entry in `inputs` and `outputs` declares a `dtype` and a `shape`, using the FNNX data model defined in [`../core.md`](../core.md). Both MUST be consistent with the ONNX graph. The declared element type MUST correspond to the tensor element type of that graph input or output. The length of the declared shape MUST equal the tensor's rank. Each integer dimension MUST equal the corresponding fixed dimension in the graph. A dimension that is symbolic in the ONNX graph is written as a string in the declared shape.

## Dynamic attributes

`ONNX_v1` defines no dynamic attributes. Producers SHOULD write the op instance's `dynamic_attributes` as an empty object. Consumers MUST ignore any entries it contains, and MUST ignore any attribute value a caller supplies for the instance.

## External data

ONNX allows a model to store its tensor data outside the `ModelProto`, in separate files referenced by relative location. When an op instance sets `has_external_data` to `true`, those files sit alongside `model.onnx` under `ops_artifacts/<op_instance_id>/`. A consumer MUST resolve external data locations relative to that directory. Producers MUST NOT reference external data outside the op instance's own directory. Producers MUST NOT use absolute paths or parent-directory segments in the references. [`../core.md`](../core.md) forbids both in artifact member paths.

## Declaration example

An op instance named `linreg`, whose model is at `ops_artifacts/linreg/model.onnx`:

```json
{
    "id": "linreg",
    "op": "ONNX_v1",
    "inputs": [
        {"dtype": "Array[float32]", "shape": ["batch", 3]}
    ],
    "outputs": [
        {"dtype": "Array[float32]", "shape": ["batch", 1]}
    ],
    "attributes": {
        "opsets": [{"domain": "ai.onnx", "version": 12}],
        "has_external_data": false,
        "onnx_ir_version": 7
    },
    "dynamic_attributes": {}
}
```

The declaration alone states that the model imports version 12 of the default ONNX operator set, and holds all its tensor data. A consumer can decide whether it supports this op instance before it opens the model file.
