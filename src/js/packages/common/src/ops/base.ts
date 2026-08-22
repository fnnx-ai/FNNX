import { ArtifactFile } from "../artifact.js";
import { DtypesManager } from "../dtypes.js";
import { OpDynamicAttribute, OpIO } from "../interfaces.js";

export interface OpOutput {
    value: unknown[];
    metadata?: Record<string, unknown>;
}

export interface OpRuntimeConfig {
    op_instance_id: string;
    attributes: Record<string, unknown>;
    dynamic_attributes: Record<string, OpDynamicAttribute>;
    inputs: OpIO[];
    outputs: OpIO[];
    dtypes_manager: DtypesManager;
}

export abstract class BaseOp {
    protected static requiredDynamicAttributes: string[] = [];

    protected readonly artifacts: ArtifactFile[];
    protected readonly attributes: Record<string, unknown>;
    protected readonly inputSpecs: OpIO[];
    protected readonly outputSpecs: OpIO[];
    protected readonly dtypesManager: DtypesManager;
    protected readonly opInstanceId: string;
    private readonly dynamicAttributeMap: Record<string, OpDynamicAttribute>;
    private warmedUp = false;

    constructor(artifacts: ArtifactFile[], config: OpRuntimeConfig) {
        this.artifacts = artifacts;
        this.attributes = config.attributes;
        this.dynamicAttributeMap = config.dynamic_attributes;
        this.inputSpecs = config.inputs;
        this.outputSpecs = config.outputs;
        this.dtypesManager = config.dtypes_manager;
        this.opInstanceId = config.op_instance_id;
    }

    abstract warmup(): Promise<this>;

    async compute(inputs: unknown[], dynamicAttributes: Record<string, string>): Promise<OpOutput> {
        const resolvedAttributes = this.resolveDynamicAttributes(dynamicAttributes);
        this.verifyRequiredDynamicAttributes(resolvedAttributes);
        return this.run(inputs, resolvedAttributes);
    }

    protected abstract run(
        inputs: unknown[],
        dynamicAttributes: Record<string, string>
    ): Promise<OpOutput>;

    private resolveDynamicAttributes(
        dynamicAttributes: Record<string, string>
    ): Record<string, string> {
        const resolved: Record<string, string> = {};
        for (const [internalName, declaration] of Object.entries(this.dynamicAttributeMap)) {
            resolved[internalName] = Object.prototype.hasOwnProperty.call(
                dynamicAttributes,
                declaration.name
            )
                ? dynamicAttributes[declaration.name]
                : declaration.default_value;
        }
        return resolved;
    }

    private verifyRequiredDynamicAttributes(dynamicAttributes: Record<string, string>): void {
        for (const name of (this.constructor as typeof BaseOp).requiredDynamicAttributes) {
            if (!(name in dynamicAttributes) || dynamicAttributes[name] === undefined) {
                throw new Error(`Missing required dynamic attribute: ${name}`);
            }
        }
    }

    protected isWarmedUp(): boolean {
        return this.warmedUp;
    }

    protected setWarmedUp(value: boolean): void {
        this.warmedUp = value;
    }
}

export type ConcreteOp = new (artifacts: ArtifactFile[], config: OpRuntimeConfig) => BaseOp;
