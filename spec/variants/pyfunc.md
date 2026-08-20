# The `pyfunc` variant

| Key | Added in | Updated in | Classification | Operations | Schema |
| --- | --- | --- | --- | --- | --- |
| `pyfunc` | 0.0.4 | 0.1.0 | Weak portability, Weak execution durability | ALL | [`variant_pyfunc.json`](../schemas/variant_pyfunc.json) |

The `pyfunc` variant packages a Python function as an FNNX artifact. A consumer imports the entry module and constructs the class it defines. It then warms the instance up, and calls it with the artifact's declared inputs.

`pyfunc` carries models that a graph of first-party operations cannot express. It pays for that with Weak portability and Weak execution durability. A consumer must reconstruct a compatible Python interpreter and package environment, and FNNX does not commit to reconstructing an obsolete one. [`../compatibility.md`](../compatibility.md) defines both grades.

## Variant configuration

`variant_config.json` holds a single JSON object.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `pyfunc_classname` | string | yes | Name of the class, defined by the entry module, that implements the PyFunc contract |
| `extra_values` | object \| null | no | Producer-defined values exposed to the function through the execution context, looked up by key |

Consumers MUST ignore additional keys.

## Artifact layout

Variant-specific content lives under `variant_artifacts/`.

```
variant_artifacts/
    __pyfunc__.py
    extra_modules/
        compute.py
    extra_files/
        val1.json
        subdir/val2.json
```

`variant_artifacts/__pyfunc__.py` is the entry module and is REQUIRED. Producers MUST write the entry module at exactly this path, and consumers MUST locate it there.

`variant_artifacts/extra_modules/` is an importable directory of Python modules and packages. The entry module imports the module at `variant_artifacts/extra_modules/<name>.py` as `<name>`. The directory is not a package, and its name is never part of an import path. The directory MAY be absent.

`variant_artifacts/extra_files/` is a payload tree for data the function needs at runtime: weights, tokenizer files, configuration, or an embedded model directory from another format. FNNX imposes no structure on it. The function addresses its contents by path relative to `extra_files/` through the execution context, never by absolute path. The directory MAY be absent.

## The PyFunc contract

The entry module MUST define a class whose name is the value of `pyfunc_classname`. A consumer constructs exactly one instance of that class. It passes the execution context as the only constructor argument. It keeps the instance for the lifetime of the loaded artifact.

The contract has three members.

- `warmup(self)` prepares the function before its first computation: loading weights, opening sessions, importing heavy dependencies. A consumer calls it once, before any computation.
- `compute(self, inputs, dynamic_attributes)` computes synchronously. `inputs` maps each declared input name to the marshalled input value. `dynamic_attributes` maps each caller-supplied attribute name to its value. The return value maps each declared output name to an output value.
- `compute_async(self, inputs, dynamic_attributes)` is the asynchronous form of the same computation. It takes the same arguments and returns the same mapping.

All three MUST be implemented. A producer whose model has no asynchronous path implements `compute_async` by delegating to `compute`.

The interface is normative, and its host language is Python. A conforming implementation of `pyfunc` is therefore a Python implementation. This is the concrete form of the variant's Weak portability.

A consumer MAY verify the contract by any means that establishes that the three members are present and callable with the signatures above. A consumer MAY also supply the contract as a base class and require the entry class to derive from it.

Consumers MUST call `warmup` before the first `compute` or `compute_async`.

## Execution context

The context object passed to the constructor is the function's only channel to the rest of the artifact. It exposes three lookups.

Files under `variant_artifacts/extra_files/` resolve by their path relative to that directory: `subdir/val2.json` names the file at `variant_artifacts/extra_files/subdir/val2.json`. Resolution yields a filesystem path the function can open, or a null result when no such file is bundled. Directories resolve the same way. Producers MUST use forward slashes as the path separator in these lookups. The function MUST NOT assume any particular location for the unpacked artifact, and MUST NOT construct paths into it by other means.

Values from `extra_values` resolve by key. An absent key yields a null result, including the case where `extra_values` itself is absent or null. `extra_values` is producer-defined configuration. This specification assigns no meaning to any key.

Declared op instances resolve by their `id`. An id that names no declared instance yields a null result. The function is the only caller of an op instance, and invokes it from `compute` or `compute_async`.

The context also carries the accelerator and device configuration the consumer selected, and an executor the function MAY use to run work concurrently. These describe the host, not the artifact.

## Inputs and outputs

A `pyfunc` artifact may declare inputs and outputs of either content type. The consumer marshals an `NDJSON` entry into the array or container form that its `dtype` names. It validates a `JSON` entry against the schema of the dtype it names, and passes the document through unchanged. A producer that accepts or returns structured non-tensor data uses `JSON` IO, and defines the corresponding dtypes with the `ext::` prefix.

Beyond that, the variant validates no IO. It takes output values from the returned mapping without any check against the declared dtypes or shapes. The producer is responsible for matching its own outputs to the manifest.

The computation MUST produce every output declared in the manifest. A returned mapping that omits a declared output is an error. Consumers ignore additional keys that correspond to no declared output.

Dynamic attributes reach the function as the caller supplied them: the variant applies no filtering and no defaults. An op instance invoked through the context applies its own mapping and defaults.

## Module loading and isolation

Consumers MUST load `variant_artifacts/__pyfunc__.py` under a module name that cannot collide with another module in the host process. Consumers MUST NOT make the entry module importable by a fixed, predictable name. Two artifacts loaded in one process are independent.

`variant_artifacts/extra_modules/` is importable while the entry module is imported and while `warmup` runs. Producers MUST import bundled modules during one of those two phases: at module import time, at class definition time, or inside `warmup`.

*Note: a consumer MAY make `extra_modules` importable during those two phases only. A bundled import deferred to `compute` then fails, or resolves against an unrelated installed module of the same name.*

The entry module may import packages installed in the environment at any time. The two-phase requirement covers only modules bundled inside the artifact.

## Environment coupling

A `pyfunc` artifact runs only in a Python environment that satisfies the imports of its entry module. Producers SHOULD describe that environment in `env.json`, under an [environment kind](../envs/README.md) that describes a Python environment. The description SHOULD be complete enough to reconstruct the environment from it alone. It is the only durable statement of what the packaged function needs.

## Example

The following artifact squares a float array.

`variant_config.json`:

```json
{
    "pyfunc_classname": "TestFunc"
}
```

`manifest.json` declares one input and one output:

```json
{
    "variant": "pyfunc",
    "producer_name": "producer",
    "producer_version": "1.0.0",
    "producer_tags": [],
    "inputs": [
        {"name": "x", "content_type": "NDJSON", "dtype": "Array[float32]", "shape": []}
    ],
    "outputs": [
        {"name": "y", "content_type": "NDJSON", "dtype": "Array[float32]", "shape": []}
    ],
    "dynamic_attributes": [],
    "env_vars": []
}
```

`variant_artifacts/__pyfunc__.py` defines the class, and imports a bundled module at import time:

```python
from compute import compute


class TestFunc:
    def __init__(self, context):
        self.context = context

    def warmup(self):
        pass

    def compute(self, inputs, dynamic_attributes):
        return {"y": compute(inputs["x"])}

    async def compute_async(self, inputs, dynamic_attributes):
        return {"y": compute(inputs["x"])}
```

`variant_artifacts/extra_modules/compute.py` supplies that module:

```python
def compute(x):
    return x**2
```

`ops.json` is the empty array `[]`, `dtypes.json` and `env.json` are `{}`, and `meta.json` is `[]`.
