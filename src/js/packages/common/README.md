# `@fnnx-ai/common`

`@fnnx-ai/common` contains the runtime-independent core for JavaScript FNNX consumers. It loads artifact declarations through an `ArtifactSource`. It executes the `pipeline` variant through registered operation classes.

The package exports `NDArray`, `NDContainer`, and the NDJSON codec. It also exports dtype validation, typed errors, and wire-format interfaces. It has no storage or network integration. It does not extract tar archives or create ONNX Runtime sessions.

Use `@fnnx-ai/node` to execute an artifact in Node.js. Use `@fnnx-ai/web` to execute an artifact in a browser. Depend on this package directly when you build another host or use its data and validation APIs without a host.

The [FNNX specification](../../../../spec/) defines artifact structure and execution semantics.

## Installation

Install the package with npm.

```bash
npm install @fnnx-ai/common
```

## Data values

`NDArray` stores values for an `Array[...]` dtype. `NDContainer` stores values for an `NDContainer[...]` dtype. The codec converts between nested NDJSON values and `NDArray` instances.

```js
import { ArrayDType, decodeNDJSON, encodeNDJSON } from "@fnnx-ai/common";

const value = decodeNDJSON("[[1,2,3]]", ArrayDType.Float32, [1, 3]);

console.log(value.getShape());
console.log(encodeNDJSON(value));
```

## Host integration

A custom host implements `ArtifactSource` for its storage system. It registers a `BaseOp` subclass for each supported operation type. The shared `Model` loads declarations and dispatches pipelines. It also handles validation, warmup, and compute.

This package has no operation backend. A host must supply the backend before the shared `Model` can execute an artifact.
