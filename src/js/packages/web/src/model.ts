import { Model as CommonModel } from "@fnnx-ai/common";
import { ONNXOpV1 } from "./ops.js";
import { WebArtifactSource } from "./source.js";

const OP_IMPLEMENTATIONS = { ONNX_v1: ONNXOpV1 };

export class Model extends CommonModel {
    private constructor(modelData: ArrayBuffer) {
        super(new WebArtifactSource(modelData), { operators: OP_IMPLEMENTATIONS });
    }

    static async fromPath(modelPath: string): Promise<Model> {
        const response = await fetch(modelPath);
        if (!response.ok) {
            throw new Error(`Could not fetch artifact: HTTP ${response.status}`);
        }
        return new Model(await response.arrayBuffer());
    }

    static async fromBuffer(modelData: ArrayBuffer): Promise<Model> {
        return new Model(modelData);
    }
}
