
import Registry from '../registry';
import { BaseOp } from '../ops/base';
import { DeviceMap, OpIO, TarFileContent, OpInstanceConfig } from '../interfaces';
import { DtypesManager } from '../ndarray';



interface OpInstance {
    operator: BaseOp;
    inputSpecs: OpIO[];
    outputSpecs: OpIO[];
}

export abstract class BaseVariant {
    protected modelContent: TarFileContent[];
    protected registry: Registry;
    protected dtypesManager: DtypesManager;
    protected opInstances: Map<string, OpInstance>;
    protected variantConfig: Record<string, any>;
    protected deviceMap: DeviceMap;

    constructor(
        modelContent: TarFileContent[],
        ops: OpInstanceConfig[],
        variantConfig: Record<string, any>,
        config: {
            registry: Registry;
            deviceMap: DeviceMap;
            dtypesManager: DtypesManager;
        }
    ) {
        this.registry = config.registry;
        this.dtypesManager = config.dtypesManager;
        this.opInstances = new Map();
        this.variantConfig = variantConfig;
        this.deviceMap = config.deviceMap;
        this.modelContent = modelContent;

        for (const opInstance of ops) {
            const OpClass = this.registry.getOp(opInstance.op);

            const device = {
                accelerator: config.deviceMap.accelerator,
                device_config: config.deviceMap.node_device_map[opInstance.id]
            };


            const operator = new OpClass(
                this.filterContent(modelContent, opInstance.id),
                {
                    attributes: opInstance.attributes || {},
                    dynamicAttributeMap: opInstance.dynamicAttributes || {},
                    deviceConfig: device,
                    inputSpecs: opInstance.inputs,
                    outputSpecs: opInstance.outputs,
                    dtypesManager: this.dtypesManager
                }
            );

            this.opInstances.set(
                opInstance.id,
                {
                    operator: operator,
                    inputSpecs: opInstance.inputs,
                    outputSpecs: opInstance.outputs
                }
            );
        }

   
    }

    protected filterContent(content: TarFileContent[], opId: string): TarFileContent[] {
        return content.filter((c) => c.relpath.startsWith(`ops_artifacts/${opId}/`) && c.relpath !== `ops_artifacts/${opId}/`);
    }

    protected abstract postInit(): void;

    async warmup(): Promise<this> {
        for (const instance of this.opInstances.values()) {
            await instance.operator.warmup();
        }
        return this;
    }

    abstract compute(
        inputs: Record<string, any>,
        dynamicAttributes: Record<string, any>
    ): Promise<Record<string, any>>;

}

export type ConcreteVariant = new (modelContent: TarFileContent[],
    ops: OpInstanceConfig[],
    variantConfig: Record<string, any>,
    config: {
        registry: Registry;
        deviceMap: DeviceMap;
        dtypesManager: DtypesManager;
    }) => BaseVariant;
