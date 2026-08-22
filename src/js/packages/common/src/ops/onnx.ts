import { ArtifactFile } from "../artifact.js";
import {
    InvalidOperationDeclarationError,
    FNNXError,
    MissingArtifactFileError,
    OperationInstanceError,
    UnsupportedExternalDataError,
    UnsupportedONNXDomainError,
} from "../errors.js";
import { NDArray } from "../ndarray.js";
import { ONNXAttributes, ONNXOpset } from "../interfaces.js";
import { BaseOp, OpOutput, OpRuntimeConfig } from "./base.js";

const STANDARD_ONNX_DOMAINS = new Set(["ai.onnx", "ai.onnx.ml"]);

export interface ONNXSession {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    run(inputs: NDArray[]): Promise<NDArray[]>;
}

export abstract class ONNXOpBase extends BaseOp {
    private readonly modelFile: ArtifactFile;
    private session: ONNXSession | null = null;

    constructor(artifacts: ArtifactFile[], config: OpRuntimeConfig) {
        super(artifacts, config);
        const expectedPath = `ops_artifacts/${config.op_instance_id}/model.onnx`;
        const modelFile = artifacts.find((artifact) => artifact.path === expectedPath);
        if (!modelFile) {
            throw new MissingArtifactFileError(expectedPath);
        }
        this.modelFile = modelFile;

        const attributes = this.validateAttributes(config.attributes);
        for (const opset of attributes.opsets) {
            if (!STANDARD_ONNX_DOMAINS.has(opset.domain)) {
                throw new UnsupportedONNXDomainError(config.op_instance_id, opset.domain);
            }
        }
        if (attributes.has_external_data && !this.canLoadExternalData()) {
            throw new UnsupportedExternalDataError(config.op_instance_id);
        }
    }

    async warmup(): Promise<this> {
        let session: ONNXSession;
        try {
            session = await this.createSession(this.modelFile, this.artifacts);
        } catch (error) {
            if (error instanceof FNNXError) {
                throw error;
            }
            const reason = error instanceof Error ? error.message : String(error);
            throw new OperationInstanceError(this.opInstanceId, reason);
        }
        this.validateArity(session);
        this.session = session;
        this.setWarmedUp(true);
        return this;
    }

    private validateArity(session: ONNXSession): void {
        const arities = [
            ["inputs", this.inputSpecs.length, session.inputNames.length],
            ["outputs", this.outputSpecs.length, session.outputNames.length],
        ] as const;
        for (const [field, declared, actual] of arities) {
            if (declared !== actual) {
                throw new InvalidOperationDeclarationError(
                    this.opInstanceId,
                    field,
                    `declares ${declared} ${field}, but the ONNX model has ${actual}`
                );
            }
        }
    }

    protected async run(
        inputs: unknown[],
        _dynamicAttributes: Record<string, string>
    ): Promise<OpOutput> {
        if (!this.session || !this.isWarmedUp()) {
            throw new OperationInstanceError(this.opInstanceId, "ONNX session is not initialized");
        }
        if (!inputs.every((input) => input instanceof NDArray)) {
            throw new OperationInstanceError(this.opInstanceId, "ONNX inputs must be NDArrays");
        }
        try {
            return { value: await this.session.run(inputs as NDArray[]) };
        } catch (error) {
            if (error instanceof FNNXError) {
                throw error;
            }
            const reason = error instanceof Error ? error.message : String(error);
            throw new OperationInstanceError(this.opInstanceId, reason);
        }
    }

    protected abstract createSession(
        modelFile: ArtifactFile,
        artifacts: ArtifactFile[]
    ): Promise<ONNXSession>;

    protected abstract canLoadExternalData(): boolean;

    private validateAttributes(attributes: Record<string, unknown>): ONNXAttributes {
        if (!Array.isArray(attributes.opsets)) {
            throw new InvalidOperationDeclarationError(
                this.opInstanceId,
                "opsets",
                "expected an array"
            );
        }
        const opsets = attributes.opsets.map((value) => this.validateOpset(value));
        if (typeof attributes.has_external_data !== "boolean") {
            throw new InvalidOperationDeclarationError(
                this.opInstanceId,
                "has_external_data",
                "expected a boolean"
            );
        }
        if (!Number.isInteger(attributes.onnx_ir_version)) {
            throw new InvalidOperationDeclarationError(
                this.opInstanceId,
                "onnx_ir_version",
                "expected an integer"
            );
        }
        return {
            ...attributes,
            opsets,
            has_external_data: attributes.has_external_data,
            onnx_ir_version: attributes.onnx_ir_version as number,
        } as ONNXAttributes;
    }

    private validateOpset(value: unknown): ONNXOpset {
        if (typeof value !== "object" || value === null) {
            throw new InvalidOperationDeclarationError(
                this.opInstanceId,
                "opsets",
                "expected objects with domain and version"
            );
        }
        const opset = value as Record<string, unknown>;
        if (typeof opset.domain !== "string" || !Number.isInteger(opset.version)) {
            throw new InvalidOperationDeclarationError(
                this.opInstanceId,
                "opsets",
                "expected string domains and integer versions"
            );
        }
        return { domain: opset.domain, version: opset.version as number };
    }
}
