import { ArtifactSource } from "../artifact.js";
import { DtypesManager } from "../dtypes.js";
import { InvalidArtifactFileError, OperationInstanceError } from "../errors.js";
import { Manifest, OpInstanceConfig, OpIO, PipelineVariant } from "../interfaces.js";
import { BaseOp } from "../ops/base.js";
import Registry from "../registry.js";
import { BaseVariant } from "./base.js";
import { DagComponent, dagCompute } from "./common/dag.js";
import { validateInputs } from "./common/validators.js";

export function validatePipeline(
    manifest: Manifest,
    ops: OpInstanceConfig[],
    variantConfig: Record<string, unknown>
): void {
    for (const [entryKind, entries] of [
        ["input", manifest.inputs],
        ["output", manifest.outputs],
    ] as const) {
        for (const entry of entries) {
            if (entry.content_type !== "NDJSON") {
                throw new InvalidArtifactFileError(
                    "manifest.json",
                    `Pipeline ${entryKind} \`${entry.name}\` requires the NDJSON content type, ` +
                        `got \`${entry.content_type}\``
                );
            }
        }
    }

    if (!Array.isArray(variantConfig.nodes)) {
        throw new InvalidArtifactFileError("variant_config.json", "`nodes` must be an array");
    }

    const opInstances = new Map(ops.map((instance) => [instance.id, instance]));
    const boundNames = new Set<string>();
    for (const input of manifest.inputs) {
        if (boundNames.has(input.name)) {
            throw new Error(`Pipeline input \`${input.name}\` binds a value more than once`);
        }
        boundNames.add(input.name);
    }

    const config = variantConfig as unknown as PipelineVariant;
    for (const [nodeIndex, node] of config.nodes.entries()) {
        const nodeName = `Pipeline node ${nodeIndex} (\`${node.op_instance_id}\`)`;
        const opInstance = opInstances.get(node.op_instance_id);
        if (!opInstance) {
            throw new Error(
                `${nodeName} references undeclared op instance \`${node.op_instance_id}\``
            );
        }

        for (const ioKind of ["inputs", "outputs"] as const) {
            const nodeArity = node[ioKind].length;
            const opArity = opInstance[ioKind].length;
            if (nodeArity !== opArity) {
                throw new Error(
                    `${nodeName} has ${ioKind.slice(0, -1)} arity ${nodeArity}, but op ` +
                        `instance \`${node.op_instance_id}\` declares ${opArity}`
                );
            }
        }

        for (const inputName of node.inputs) {
            if (!boundNames.has(inputName)) {
                throw new Error(`${nodeName} consumes unbound input \`${inputName}\``);
            }
        }
        for (const outputName of node.outputs) {
            if (boundNames.has(outputName)) {
                throw new Error(`${nodeName} binds value \`${outputName}\` more than once`);
            }
            boundNames.add(outputName);
        }
    }
}

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

    private async nodeCompute(
        nodeInstance: PipelineNodeInstance,
        nodeInputs: unknown[],
        passthrough: Record<string, unknown>
    ): Promise<unknown> {
        validateInputs(nodeInputs, nodeInstance.inputSpecs);
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
