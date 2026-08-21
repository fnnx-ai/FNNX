import { ArtifactSource } from "../artifact.js";
import { DtypesManager } from "../dtypes.js";
import { OperationInstanceError } from "../errors.js";
import { OpInstanceConfig, OpIO, PipelineVariant } from "../interfaces.js";
import { BaseOp } from "../ops/base.js";
import Registry from "../registry.js";
import { BaseVariant } from "./base.js";
import { DagComponent, dagCompute } from "./common/dag.js";

class PipelineNodeInstance implements DagComponent {
    readonly operator: BaseOp;
    readonly inputs: string[];
    readonly outputs: string[];
    readonly inputSpecs: OpIO[];
    readonly outputSpecs: OpIO[];
    readonly extra_dynattrs: Record<string, string>;

    constructor(config: {
        operator: BaseOp;
        inputs: string[];
        outputs: string[];
        inputSpecs: OpIO[];
        outputSpecs: OpIO[];
        extra_dynattrs: Record<string, string>;
    }) {
        this.operator = config.operator;
        this.inputs = config.inputs;
        this.outputs = config.outputs;
        this.inputSpecs = config.inputSpecs;
        this.outputSpecs = config.outputSpecs;
        this.extra_dynattrs = config.extra_dynattrs;
    }
}

export class Pipeline extends BaseVariant {
    private readonly pipelineNodeInstances: PipelineNodeInstance[];

    constructor(
        source: ArtifactSource,
        ops: OpInstanceConfig[],
        variantConfig: Record<string, unknown>,
        registry: Registry,
        dtypesManager: DtypesManager
    ) {
        super(source, ops, variantConfig, registry, dtypesManager);
        const config = variantConfig as unknown as PipelineVariant;
        this.pipelineNodeInstances = config.nodes.map((node) => {
            const opInstance = this.opInstances.get(node.op_instance_id);
            if (!opInstance) {
                throw new OperationInstanceError(node.op_instance_id, "instance is not declared");
            }
            return new PipelineNodeInstance({
                operator: opInstance.operator,
                inputs: node.inputs,
                outputs: node.outputs,
                inputSpecs: opInstance.inputSpecs,
                outputSpecs: opInstance.outputSpecs,
                extra_dynattrs: node.extra_dynattrs ?? {},
            });
        });
    }

    private validateInputs(inputs: unknown[], inputSpecs: OpIO[]): void {
        for (let index = 0; index < inputSpecs.length; index++) {
            const spec = inputSpecs[index];
            const input = inputs[index] as { shape?: number[] } | null | undefined;
            if (input === undefined || input === null || !input.shape || spec.shape.length === 0) {
                continue;
            }
            if (
                spec.shape.length !== input.shape.length ||
                spec.shape.some(
                    (extent, dimension) =>
                        typeof extent === "number" && extent !== input.shape?.[dimension]
                )
            ) {
                throw new Error(`Expected input shape [${spec.shape}], got [${input.shape}]`);
            }
        }
    }

    private async nodeCompute(
        nodeInstance: PipelineNodeInstance,
        nodeInputs: unknown[],
        passthrough: Record<string, unknown>
    ): Promise<unknown> {
        this.validateInputs(nodeInputs, nodeInstance.inputSpecs);
        return nodeInstance.operator.compute(
            nodeInputs,
            passthrough.dynamic_attributes as Record<string, string>
        );
    }

    async compute(
        inputs: Record<string, unknown>,
        dynamicAttributes: Record<string, string>
    ): Promise<Record<string, unknown>> {
        return dagCompute(
            inputs,
            this.pipelineNodeInstances,
            (component, values, passthrough) =>
                this.nodeCompute(
                    component as PipelineNodeInstance,
                    values,
                    passthrough as Record<string, unknown>
                ),
            (result) => (result as { value: unknown[] }).value,
            { dynamic_attributes: dynamicAttributes }
        );
    }
}
