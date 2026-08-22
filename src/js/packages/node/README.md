# `@fnnx-ai/node`

`@fnnx-ai/node` loads and executes FNNX artifacts in Node.js. It uses `onnxruntime-node` for `ONNX_v1` operation instances.

The [FNNX specification](../../../../spec/) defines artifact structure and execution semantics.

## Installation

Install the package with npm.

```bash
npm install @fnnx-ai/node
```

## Loading and compute

`Model.fromPath()` accepts an artifact directory or an uncompressed tar archive. Call `warmup()` before the first computation. Pass an object whose keys match the input names in the manifest.

Use `NDArray` for an `Array[...]` input. Its shape and `ArrayDType` must match the manifest declaration. Use `NDContainer` or nested JavaScript values for an `NDContainer[...]` input.

```js
import { ArrayDType, Model, NDArray } from "@fnnx-ai/node";

const model = await Model.fromPath("model.fnnx");

try {
    await model.warmup();

    const x = new NDArray([1, 3], new Float32Array([1, 2, 3]), ArrayDType.Float32);
    const outputs = await model.compute({ x });
    const y4 = outputs.y4;

    if (!(y4 instanceof NDArray)) {
        throw new TypeError("Expected output y4 to be an NDArray");
    }

    console.log(y4.toArray());
} finally {
    model.cleanup();
}
```

The optional second argument to `compute()` is a string-to-string dynamic attribute map. The manifest defines the artifact-specific input, output, and attribute names.

## Buffer loading

`Model.fromBuffer()` accepts a Node.js `Buffer` or an `ArrayBuffer` that contains an uncompressed tar artifact.

```js
import { readFile } from "node:fs/promises";
import { Model } from "@fnnx-ai/node";

const bytes = await readFile("model.fnnx");
const model = await Model.fromBuffer(bytes);

try {
    console.log(model.getManifest());
} finally {
    model.cleanup();
}
```

The model extracts tar data to a temporary directory. Call `cleanup()` when the model is no longer needed. The method does not remove an artifact directory passed to `fromPath()`.

## Inspection and limits

Use `getManifest()` and `getMetadata()` to inspect the main declarations. Use `getDtypes()` and `getEnv()` to inspect the remaining documents. Each method returns a copy.

The package executes the `pipeline` variant with `ONNX_v1` operations. It does not execute the `pyfunc` variant or other operation types. It ignores `env.json` during execution because it runs in the current process.
