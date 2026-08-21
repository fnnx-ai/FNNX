import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { Model } from "../../src/index";

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const modelPath = path.resolve(
    testDirectory,
    "../../../../../python/tests/models/onnx_pipeline.fnnx.tar"
);

describe("web Model", () => {
    it("loads the shared cross-language fixture from an ArrayBuffer", async () => {
        const bytes = readFileSync(modelPath);
        const buffer = bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength
        ) as ArrayBuffer;

        const model = await Model.fromBuffer(buffer);

        expect(model.getManifest().variant).toBe("pipeline");
        expect(model.getDtypes()).toEqual({});
        expect(model.getEnv()).toEqual({});
    });
});
