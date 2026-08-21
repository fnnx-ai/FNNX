import { ArtifactSource } from "./artifact.js";
import { DtypesManager } from "./dtypes.js";
import { UnsupportedVariantError } from "./errors.js";
import { JSONIO, Manifest, NDJSONIO, OpInstanceConfig } from "./interfaces.js";
import { ArrayDType, NDArray } from "./ndarray.js";
import { NDContainer } from "./ndcontainer.js";
import { ConcreteOp } from "./ops/base.js";
import Registry from "./registry.js";
import { BaseVariant, ConcreteVariant } from "./variants/base.js";
import { Pipeline } from "./variants/pipeline.js";

export type DynamicAttributes = Record<string, string>;
export type Inputs = Record<string, unknown>;
export type Outputs = Record<string, unknown>;

export interface HandlerConfig {
    operators: Record<string, ConcreteOp>;
}

export class LocalHandler {
    private readonly dtypesManager: DtypesManager;
    private readonly variant: string;
    private readonly runtimeVariant: BaseVariant;
    private readonly inputSpecs: Record<string, NDJSONIO | JSONIO>;
    private readonly outputSpecs: Record<string, NDJSONIO | JSONIO>;

    constructor(
        source: ArtifactSource,
        manifest: Manifest,
        ops: OpInstanceConfig[],
        variantConfig: Record<string, unknown>,
        dtypesManager: DtypesManager,
        handlerConfig: HandlerConfig
    ) {
        this.dtypesManager = dtypesManager;
        this.variant = manifest.variant;
        this.inputSpecs = Object.fromEntries(manifest.inputs.map((spec) => [spec.name, spec]));
        this.outputSpecs = Object.fromEntries(manifest.outputs.map((spec) => [spec.name, spec]));

        let VariantClass: ConcreteVariant;
        if (this.variant === "pipeline") {
            VariantClass = Pipeline;
        } else {
            throw new UnsupportedVariantError(this.variant);
        }
        this.runtimeVariant = new VariantClass(
            source,
            ops,
            variantConfig,
            new Registry(handlerConfig.operators),
            dtypesManager
        );
    }

    warmup(): Promise<BaseVariant> {
        return this.runtimeVariant.warmup();
    }

    private parseArrayDtype(dtype: string): ArrayDType {
        const match = dtype.match(/^Array\[(.+)\]$/);
        if (!match) {
            throw new Error(`Invalid Array dtype format: ${dtype}`);
        }
        const dtypes: Record<string, ArrayDType> = {
            float32: ArrayDType.Float32,
            float64: ArrayDType.Float64,
            int32: ArrayDType.Int32,
            int64: ArrayDType.Int64,
            string: ArrayDType.String,
            bool: ArrayDType.Bool,
        };
        const arrayDtype = dtypes[match[1]];
        if (!arrayDtype) {
            throw new Error(`Unsupported Array dtype: ${match[1]}`);
        }
        return arrayDtype;
    }

    private prepareInputs(inputs: Inputs): Inputs {
        const preparedInputs: Inputs = {};
        for (const [name, input] of Object.entries(inputs)) {
            const spec = this.inputSpecs[name];
            if (!spec) {
                throw new Error(`Unknown input: ${name}`);
            }
            if (spec.content_type === "NDJSON") {
                if (spec.dtype.startsWith("NDContainer[")) {
                    preparedInputs[name] =
                        input instanceof NDContainer
                            ? input
                            : new NDContainer(input, spec.dtype, this.dtypesManager);
                } else if (spec.dtype.startsWith("Array[")) {
                    if (!(input instanceof NDArray)) {
                        throw new Error(`Input ${name} must be an NDArray`);
                    }
                    const expectedDtype = this.parseArrayDtype(spec.dtype);
                    if (input.getDType() !== expectedDtype) {
                        throw new Error(
                            `Input dtype mismatch for ${name}. Expected ${expectedDtype}, got ${input.getDType()}`
                        );
                    }
                    preparedInputs[name] = input;
                } else {
                    throw new Error(`Invalid NDJSON dtype: ${spec.dtype}`);
                }
            } else if (spec.content_type === "JSON") {
                if (this.variant === "pipeline") {
                    throw new Error("Pipeline variant does not support JSON inputs");
                }
                this.dtypesManager.validateJsonSchema(spec.dtype, input);
                preparedInputs[name] = input;
            }
        }
        return preparedInputs;
    }

    private prepareOutputs(outputs: Outputs): Outputs {
        return Object.fromEntries(
            Object.keys(this.outputSpecs)
                .map((name) => [name, outputs[name]])
                .filter((entry) => entry[1] !== undefined)
        );
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes = {}): Promise<Outputs> {
        const result = await this.runtimeVariant.compute(
            this.prepareInputs(inputs),
            dynamicAttributes
        );
        return this.prepareOutputs(result);
    }
}
