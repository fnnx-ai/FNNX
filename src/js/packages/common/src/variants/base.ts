import { ArtifactSource } from "../artifact.js";
import { DtypesManager } from "../dtypes.js";
import { UnsupportedOperationError } from "../errors.js";
import { OpInstanceConfig, OpIO } from "../interfaces.js";
import { BaseOp } from "../ops/base.js";
import Registry from "../registry.js";

interface OpInstance {
    operator: BaseOp;
    inputSpecs: OpIO[];
    outputSpecs: OpIO[];
}

export abstract class BaseVariant {
    protected readonly dtypesManager: DtypesManager;
    protected readonly opInstances = new Map<string, OpInstance>();
    protected readonly variantConfig: Record<string, unknown>;

    constructor(
        source: ArtifactSource,
        ops: OpInstanceConfig[],
        variantConfig: Record<string, unknown>,
        registry: Registry,
        dtypesManager: DtypesManager
    ) {
        this.dtypesManager = dtypesManager;
        this.variantConfig = variantConfig;

        for (const opInstance of ops) {
            const OpClass = registry.getOp(opInstance.op);
            if (!OpClass) {
                throw new UnsupportedOperationError(opInstance.op, opInstance.id);
            }
            const operator = new OpClass(source.resolveOpArtifacts(opInstance.id), {
                op_instance_id: opInstance.id,
                attributes: opInstance.attributes ?? {},
                dynamic_attributes: opInstance.dynamic_attributes ?? {},
                inputs: opInstance.inputs,
                outputs: opInstance.outputs,
                dtypes_manager: dtypesManager,
            });
            this.opInstances.set(opInstance.id, {
                operator,
                inputSpecs: opInstance.inputs,
                outputSpecs: opInstance.outputs,
            });
        }
    }

    async warmup(): Promise<this> {
        for (const instance of this.opInstances.values()) {
            await instance.operator.warmup();
        }
        return this;
    }

    abstract compute(
        inputs: Record<string, unknown>,
        dynamicAttributes: Record<string, string>
    ): Promise<Record<string, unknown>>;
}

export type ConcreteVariant = new (
    source: ArtifactSource,
    ops: OpInstanceConfig[],
    variantConfig: Record<string, unknown>,
    registry: Registry,
    dtypesManager: DtypesManager
) => BaseVariant;
