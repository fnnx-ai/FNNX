
import { BaseVariant, ConcreteVariant } from './variants/base';
import { ConcreteOp } from './ops/base';
import Registry from './registry';
import { NDArray, ArrayDType, DtypesManager, NDContainer } from './ndarray';
import { Manifest, OpInstanceConfig, PipelineVariant, DeviceMap, JSONI, NDJSON, TarFileContent } from './interfaces';
import { Pipeline } from './variants/pipeline';


export type DynamicAttributes = Record<string, any>;
export type Inputs = Record<string, any>;
export type Outputs = Record<string, any>;

interface HandlerConfig {
    operators: Record<string, ConcreteOp>;
}


export class LocalHandler {
    private manifest: Manifest;
    private ops: OpInstanceConfig[];
    private variantConfig: Record<string, any>;
    private dtypesManager: DtypesManager;
    private variant: string;
    private vrt: BaseVariant;
    private inputSpecs: Record<string, NDJSON | JSONI>;
    private outputSpecs: Record<string, NDJSON | JSONI>;
    private modelContent: TarFileContent[];

    constructor(
        modelContent: TarFileContent[],
        manifest: Manifest,
        ops: OpInstanceConfig[],
        variantConfig: PipelineVariant,
        dtypesManager: DtypesManager,
        deviceMap: DeviceMap,
        handlerConfig: HandlerConfig
    ) {
        this.modelContent = modelContent;
        this.manifest = manifest;
        this.ops = ops;
        this.variantConfig = variantConfig;
        this.dtypesManager = dtypesManager;
        this.variant = manifest.variant;

        // Create lookup maps for specs
        this.inputSpecs = Object.fromEntries(
            manifest.inputs.map(spec => [spec.name, spec])
        );
        this.outputSpecs = Object.fromEntries(
            manifest.outputs.map(spec => [spec.name, spec])
        );

        const registry = new Registry(handlerConfig.operators);

        let VariantClass: ConcreteVariant;
        if (this.variant === 'pipeline') {
            VariantClass = Pipeline;
        } else if (this.variant === 'pyfunc') {
            throw new Error('pyfunc variant is not supported in JavaScript');
        } else {
            throw new Error(`Unknown variant: ${this.variant}`);
        }

        this.vrt = new VariantClass(
            this.modelContent,
            ops,
            variantConfig,
            {
                registry,
                deviceMap,
                dtypesManager
            }
        )
    }

    public async warmup(): Promise<void> {
        await this.vrt.warmup();
    }

    private parseArrayDtype(dtype: string): ArrayDType {
        // Extract dtype from Array[dtype] format
        const match = dtype.match(/^Array\[(.+)\]$/);
        if (!match) {
            throw new Error(`Invalid Array dtype format: ${dtype}`);
        }

        const dtypeStr = match[1];
        switch (dtypeStr) {
            case 'float32':
                return ArrayDType.Float32;
            case 'int32':
                return ArrayDType.Int32;
            case 'int64':
                return ArrayDType.Int64;
            case 'string':
                return ArrayDType.String;
            case 'bool':
                return ArrayDType.Bool;
            default:
                throw new Error(`Unsupported Array dtype: ${dtypeStr}`);
        }
    }

    private prepareInputs(inputs: Inputs): Inputs {
        const preparedInputs: Inputs = {};

        for (const [name, input] of Object.entries(inputs)) {
            const spec = this.inputSpecs[name];
            if (!spec) {
                throw new Error(`Unknown input: ${name}`);
            }

            if (spec.content_type === 'NDJSON') {
                if (spec.dtype.startsWith('NDContainer[')) {
                    if (input instanceof NDContainer) {
                        preparedInputs[name] = input;
                    } else {
                        preparedInputs[name] = new NDContainer(
                            input,
                            spec.dtype,
                            this.dtypesManager
                        );
                    }
                } else if (spec.dtype.startsWith('Array[')) {
                    if (!(input instanceof NDArray)) {
                        throw new Error(`Input ${name} must be an NDArray`);
                    } else {
                        const arrayDtype = this.parseArrayDtype(spec.dtype);
                        if (input.getDType() !== arrayDtype) {
                            throw new Error(`Input dtype mismatch for ${name}. Expected ${arrayDtype}, got ${input.getDType()}`);
                        }
                        preparedInputs[name] = input;
                    }
                } else {
                    throw new Error(`Invalid NDJSON dtype: ${spec.dtype}. Must be Array[...] or NDContainer[...]`);
                }
            } else if (spec.content_type === 'JSON') {
                if (this.variant === 'pipeline') {
                    throw new Error('Pipeline variant does not support JSON inputs');
                }
                const dtype = spec.dtype;
                this.dtypesManager.validateJsonSchema(dtype, input);
                preparedInputs[name] = input;
            } else {
                throw new Error(`Unknown content type: ${spec["content_type"]}`);
            }
        }

        return preparedInputs;
    }

    private prepareOutputs(outputs: Outputs): Outputs {
        return Object.fromEntries(
            Object.keys(this.outputSpecs)
                .map(key => [key, outputs[key]])
                .filter(([_, value]) => value !== undefined)
        );
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes = {}): Promise<Outputs> {
        const res = await this.vrt.compute(
            this.prepareInputs(inputs),
            dynamicAttributes
        );
        return this.prepareOutputs(res);
    }
}
