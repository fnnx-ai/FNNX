# `@fnnx-ai/web`

`@fnnx-ai/web` loads and executes FNNX artifacts in a browser. It uses `onnxruntime-web` for `ONNX_v1` operation instances.

The [FNNX specification](../../../../spec/) defines artifact structure and execution semantics.

## Installation

Install the package with npm.

```bash
npm install @fnnx-ai/web
```

## Fetching and compute

`Model.fromPath()` fetches an uncompressed tar artifact from a relative or absolute URL. Call `warmup()` before the first computation. Pass an object whose keys match the input names in the manifest.

Use `NDArray` for an `Array[...]` input. Its shape and `ArrayDType` must match the manifest declaration. Use `NDContainer` or nested JavaScript values for an `NDContainer[...]` input.

```js
import { ArrayDType, Model, NDArray } from "@fnnx-ai/web";

const model = await Model.fromPath("/models/model.fnnx");
await model.warmup();

const x = new NDArray([1, 3], new Float32Array([1, 2, 3]), ArrayDType.Float32);
const outputs = await model.compute({ x });
const y4 = outputs.y4;

if (!(y4 instanceof NDArray)) {
    throw new TypeError("Expected output y4 to be an NDArray");
}

console.log(y4.toArray());
```

The optional second argument to `compute()` is a string-to-string dynamic attribute map. The manifest defines the artifact-specific input, output, and attribute names.

## Buffer loading

Use `Model.fromBuffer()` when application code fetches the artifact. The method accepts an `ArrayBuffer` that contains an uncompressed tar artifact.

```js
import { Model } from "@fnnx-ai/web";

const response = await fetch("/models/model.fnnx");

if (!response.ok) {
    throw new Error(`Could not fetch artifact: HTTP ${response.status}`);
}

const model = await Model.fromBuffer(await response.arrayBuffer());
console.log(model.getManifest());
```

## Inspection and limits

Use `getManifest()` and `getMetadata()` to inspect the main declarations. Use `getDtypes()` and `getEnv()` to inspect the remaining documents. Each method returns a copy.

The package executes the `pipeline` variant with `ONNX_v1` operations. It does not execute the `pyfunc` variant or other operation types. It ignores `env.json` during execution because it runs in the current page.

The web backend does not load ONNX external data files. It rejects an operation instance that declares external data.
