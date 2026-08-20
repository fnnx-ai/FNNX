import { BaseOp, interfaces, DtypesManager, NDArray, ArrayDType } from "@fnnx-ai/common";
import * as ort from "onnxruntime-web";



export class ONNXOpV1 extends BaseOp {
    private modelFile: interfaces.TarFileContent;
    private inferenceSession: ort.InferenceSession | null = null;
    private inputNames: string[] = [];
    private outputNames: string[] = [];

    constructor(
        artifacts: interfaces.TarFileContent[],
        config: {
            attributes: Record<string, any>;
            dynamicAttributeMap: Record<string, interfaces.OpDynamicAttribute>;
            deviceConfig: interfaces.DeviceConfig;
            inputSpecs: interfaces.OpIO[];
            outputSpecs: interfaces.OpIO[];
            dtypesManager: DtypesManager;
        }) {
        super(artifacts, config);


        const modelFile = artifacts.find(f => f.relpath.endsWith('model.onnx'));
        if (!modelFile) {
            throw new Error('Model file not found');
        }
        this.modelFile = modelFile;

    }

    async warmup(..._args: any[]): Promise<this> {
        if (!this.modelFile.content) {
            throw new Error('Model file not found');
        }
        this.inferenceSession = await ort.InferenceSession.create(this.modelFile.content);
        this.inputNames = [...this.inferenceSession.inputNames];
        this.outputNames = [...this.inferenceSession.outputNames];

        return this;
    }

    async compute(inputs: any, _dynamicAttributes: any) {
        if (!this.inferenceSession) {
            throw new Error('Model not loaded');
        }
        const feeds = Object.fromEntries(
            this.inputNames.map((name, i) => {
                const inp = inputs[i].toArray();
                const dtype = inputs[i].dtype;
                const shape = inputs[i].shape;
                return [name, new ort.Tensor(dtype, inp, shape)];
            })
        );

        const outputs = await this.inferenceSession.run(feeds);
        const results = this.outputNames.map(name => {
            const output = outputs[name];
            if (!output) {
                throw new Error(`Output ${name} not found`);
            }
            const data = output.data;
            const shape = [...output.dims];
            const dtype = output.type;
            return new NDArray(shape, <any[]>data, <ArrayDType>dtype);
        })
        return { value: results, metadata: {} };
    }
}
