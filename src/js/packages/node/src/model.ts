import { mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Model as CommonModel } from "@fnnx-ai/common";
import { ONNXOpV1 } from "./ops.js";
import {
    extractTarBufferToDirectory,
    extractTarFileToDirectory,
    NodeArtifactSource,
} from "./source.js";

const OP_IMPLEMENTATIONS = { ONNX_v1: ONNXOpV1 };

export class Model extends CommonModel {
    private cleanupDirectory: string | null;
    private exitCleanup: (() => void) | null = null;

    private constructor(modelDirectory: string, cleanupDirectory: string | null) {
        super(new NodeArtifactSource(modelDirectory), { operators: OP_IMPLEMENTATIONS });
        this.cleanupDirectory = cleanupDirectory;
        if (cleanupDirectory) {
            this.exitCleanup = () => {
                rmSync(cleanupDirectory, { recursive: true, force: true });
            };
            process.once("exit", this.exitCleanup);
        }
    }

    static async fromPath(modelPath: string): Promise<Model> {
        if (statSync(modelPath).isDirectory()) {
            return new Model(modelPath, null);
        }
        const temporaryDirectory = mkdtempSync(path.join(tmpdir(), "fnnx-"));
        try {
            await extractTarFileToDirectory(modelPath, temporaryDirectory);
            return new Model(temporaryDirectory, temporaryDirectory);
        } catch (error) {
            rmSync(temporaryDirectory, { recursive: true, force: true });
            throw error;
        }
    }

    static async fromBuffer(modelData: ArrayBuffer | Buffer): Promise<Model> {
        const temporaryDirectory = mkdtempSync(path.join(tmpdir(), "fnnx-"));
        const buffer = Buffer.isBuffer(modelData) ? modelData : Buffer.from(modelData);
        try {
            await extractTarBufferToDirectory(buffer, temporaryDirectory);
            return new Model(temporaryDirectory, temporaryDirectory);
        } catch (error) {
            rmSync(temporaryDirectory, { recursive: true, force: true });
            throw error;
        }
    }

    cleanup(): void {
        if (this.cleanupDirectory) {
            rmSync(this.cleanupDirectory, { recursive: true, force: true });
            this.cleanupDirectory = null;
        }
        if (this.exitCleanup) {
            process.off("exit", this.exitCleanup);
            this.exitCleanup = null;
        }
    }
}
