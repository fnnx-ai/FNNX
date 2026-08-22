<p align="center">
<picture>
  <img alt="FNNX logo" src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/9043db455d5fbde91d030063720a96bbed01fcaf/fnnx.svg" height = "250">
</picture>
</p>

# FNNX: A Universal Machine Learning Packaging Format

<p align="center">
<a href="https://pypi.org/project/fnnx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fnnx?label=pypi%20%7C%20fnnx"></a>
<a href="https://www.npmjs.com/package/@fnnx-ai/node"><img alt="npm" src="https://img.shields.io/npm/v/%40fnnx-ai%2Fnode?label=npm%20%7C%20%40fnnx-ai%2Fnode"></a>
<a href="spec/"><img alt="Specification" src="https://img.shields.io/badge/spec-0.1.0-blue"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

FNNX is a packaging format for machine learning models. An FNNX artifact holds a model, a statement of how to execute it, and the names and types of its inputs and outputs. A model here is any computation with a declared interface: a single network, a preprocessing chain in front of one, or several models composed together. 

The [specification](spec/) defines the format in full.

At the moment the format has two variants. 
 - A [`pipeline`](spec/variants/pipeline.md) is a graph whose nodes are standardized, portable operations. A runtime for it can be written in any language from the specification alone. 
 - A [`pyfunc`](spec/variants/pyfunc.md) is a Python function packaged with a description of the [environment](spec/envs/python3_conda_pip.md) it needs. It can express anything Python can express, but running it means recreating that environment. The specification recommends a `pipeline` whenever the model can be expressed as one.

## Python

The `fnnx` package runs, inspects, and creates artifacts.

```console
uv add "fnnx[core]"
```

```python
import numpy as np

from fnnx.stable.v1 import Runtime

runtime = Runtime("model.fnnx")
outputs = runtime.compute({"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}, {})
print(outputs["y"])
```

The package also converts MLflow models to artifacts and compiles pipeline artifacts to C99. [`src/python/README.md`](src/python/README.md) documents it in full.

## JavaScript

[`@fnnx-ai/node`](src/js/packages/node/README.md) runs artifacts in Node.js. [`@fnnx-ai/web`](src/js/packages/web/README.md) runs them in the browser. Both execute the `pipeline` variant only; running a `pyfunc` requires Python.

```console
npm install @fnnx-ai/node
```

```js
import { ArrayDType, Model, NDArray } from "@fnnx-ai/node";

const model = await Model.fromPath("model.fnnx");
await model.warmup();

const x = new NDArray([1, 2], new Float32Array([1, 2]), ArrayDType.Float32);
const outputs = await model.compute({ x });

console.log(outputs.y.toArray());
model.cleanup();
```

## License

FNNX is released under the [MIT license](LICENSE).

⭐ If you liked the project, please support us with a star!
