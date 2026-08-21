import {
    ArrayDType,
    ArtifactFile,
    MissingArtifactFileError,
    NDArray,
    ONNXOpBase,
    ONNXSession,
} from "@fnnx-ai/common";
import * as ort from "onnxruntime-node";

export class ONNXOpV1 extends ONNXOpBase {
    protected canLoadExternalData(): boolean {
        return true;
    }

    protected async createSession(
        modelFile: ArtifactFile,
        _artifacts: ArtifactFile[]
    ): Promise<ONNXSession> {
        if (!modelFile.filesystemPath) {
            throw new MissingArtifactFileError(modelFile.path);
        }
        const session = await ort.InferenceSession.create(modelFile.filesystemPath);
        return {
            run: async (inputs: NDArray[]): Promise<NDArray[]> => {
                const feeds = Object.fromEntries(
                    session.inputNames.map((name, index) => {
                        const input = inputs[index];
                        return [
                            name,
                            new ort.Tensor(
                                input.dtype as ort.Tensor.Type,
                                input.toArray() as ort.Tensor.DataType,
                                input.shape
                            ),
                        ];
                    })
                );
                const outputs = await session.run(feeds);
                return session.outputNames.map((name) => {
                    const output = outputs[name];
                    if (!output) {
                        throw new Error(`ONNX output \`${name}\` was not returned`);
                    }
                    return new NDArray([...output.dims], output.data, output.type as ArrayDType);
                });
            },
        };
    }
}
