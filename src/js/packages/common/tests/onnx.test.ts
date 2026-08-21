import { describe, expect, it } from "vitest";
import {
    ArtifactFile,
    DtypesManager,
    InvalidOperationDeclarationError,
    MissingArtifactFileError,
    ONNXOpBase,
    ONNXSession,
    OpRuntimeConfig,
    UnsupportedExternalDataError,
    UnsupportedONNXDomainError,
    OperationInstanceError,
} from "../src/index.js";

class MemoryFile implements ArtifactFile {
    constructor(public readonly path: string) {}

    read(): Uint8Array {
        return new Uint8Array();
    }
}

class TestONNXOp extends ONNXOpBase {
    protected canLoadExternalData(): boolean {
        return false;
    }

    protected async createSession(
        _modelFile: ArtifactFile,
        _artifacts: ArtifactFile[]
    ): Promise<ONNXSession> {
        return { run: async () => [] };
    }
}

class ExternalDataONNXOp extends TestONNXOp {
    protected canLoadExternalData(): boolean {
        return true;
    }
}

class FailingONNXOp extends TestONNXOp {
    protected async createSession(
        _modelFile: ArtifactFile,
        _artifacts: ArtifactFile[]
    ): Promise<ONNXSession> {
        return {
            run: async () => {
                throw new Error("backend failure");
            },
        };
    }
}

function config(attributes: Record<string, unknown> = {}): OpRuntimeConfig {
    return {
        op_instance_id: "predict",
        inputs: [],
        outputs: [],
        dynamic_attributes: {},
        dtypes_manager: new DtypesManager(),
        attributes: {
            opsets: [{ domain: "ai.onnx", version: 18 }],
            has_external_data: false,
            onnx_ir_version: 9,
            ...attributes,
        },
    };
}

const MODEL_FILE = new MemoryFile("ops_artifacts/predict/model.onnx");

describe("shared ONNX operation", () => {
    it("requires model.onnx at the exact op artifact path", () => {
        const artifacts = [
            new MemoryFile("ops_artifacts/predict/sub/model.onnx"),
            new MemoryFile("ops_artifacts/predict/xmodel.onnx"),
        ];

        expect(() => new TestONNXOp(artifacts, config())).toThrowError(MissingArtifactFileError);
        try {
            new TestONNXOp(artifacts, config());
        } catch (error) {
            expect(error).toMatchObject({ filePath: "ops_artifacts/predict/model.onnx" });
        }
    });

    it("declines a non-standard domain with its domain and instance", () => {
        expect(
            () =>
                new TestONNXOp(
                    [MODEL_FILE],
                    config({ opsets: [{ domain: "example.org", version: 1 }] })
                )
        ).toThrowError(UnsupportedONNXDomainError);
        try {
            new TestONNXOp(
                [MODEL_FILE],
                config({ opsets: [{ domain: "example.org", version: 1 }] })
            );
        } catch (error) {
            expect(error).toMatchObject({ domain: "example.org", opInstanceId: "predict" });
        }
    });

    it("declines external data when the backend cannot load it", () => {
        expect(
            () => new TestONNXOp([MODEL_FILE], config({ has_external_data: true }))
        ).toThrowError(UnsupportedExternalDataError);
    });

    it("accepts external data when the backend supports it", () => {
        expect(
            () => new ExternalDataONNXOp([MODEL_FILE], config({ has_external_data: true }))
        ).not.toThrow();
    });

    it("validates required compatibility declarations", () => {
        expect(() => new TestONNXOp([MODEL_FILE], config({ opsets: "invalid" }))).toThrowError(
            InvalidOperationDeclarationError
        );
    });

    it("wraps backend errors with the op instance id", async () => {
        const operation = new FailingONNXOp([MODEL_FILE], config());
        await operation.warmup();

        await expect(operation.compute([], {})).rejects.toThrowError(OperationInstanceError);
        await operation.compute([], {}).catch((error: unknown) => {
            expect(error).toMatchObject({ opInstanceId: "predict" });
        });
    });
});
