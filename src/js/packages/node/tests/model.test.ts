import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { readFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { extract as tarExtract } from "tar";
import { Model, NDArray } from "../src/index";
import { InvalidOperationDeclarationError, ModelNotWarmedUpError } from "@fnnx-ai/common";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MODEL_PATH = path.resolve(__dirname, "../../../../python/tests/models/onnx_pipeline.fnnx.tar");

async function extractModelToDir(): Promise<string> {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'fnnx-test-'));
    await tarExtract({ file: MODEL_PATH, C: tempDir });
    return tempDir;
}

describe("Model", () => {
    describe("loading from tar file", () => {
        it("should load from tar path", async () => {
            const model = await Model.fromPath(MODEL_PATH);
            expect(model).toBeInstanceOf(Model);
            model.cleanup();
        });

        it("should load from buffer", async () => {
            const buffer = readFileSync(MODEL_PATH);
            const model = await Model.fromBuffer(buffer);
            expect(model).toBeInstanceOf(Model);
            model.cleanup();
        });

        it("should throw error for invalid path", async () => {
            await expect(Model.fromPath("/nonexistent/model.tar")).rejects.toThrow();
        });
    });

    describe("loading from directory", () => {
        let modelDir: string;

        beforeAll(async () => {
            modelDir = await extractModelToDir();
        });

        it("should load from directory path", async () => {
            const model = await Model.fromPath(modelDir);
            expect(model).toBeInstanceOf(Model);
        });

        it("should warmup and compute from directory", async () => {
            const model = await Model.fromPath(modelDir);
            await model.warmup();

            const input = new NDArray([1, 3], new Float32Array([1.0, 2.0, 3.0]), "float32");
            const output = await model.compute({ x: input }, {});

            const manifest = model.getManifest();
            const outputName = manifest.outputs[0].name;
            expect(output[outputName]).toBeInstanceOf(NDArray);
            expect(output[outputName].shape).toEqual([1, 1]);
        });
    });

    describe("cleanup", () => {
        it("should clean up temp directory for tar loads", async () => {
            const model = await Model.fromPath(MODEL_PATH);
            const manifest = model.getManifest();
            expect(manifest.variant).toBe("pipeline");

            model.cleanup();
        });

        it("should not fail on double cleanup", async () => {
            const model = await Model.fromPath(MODEL_PATH);
            model.cleanup();
            expect(() => model.cleanup()).not.toThrow();
        });
    });

    describe("warmup", () => {
        it("should warmup successfully", async () => {
            const model = await Model.fromPath(MODEL_PATH);
            await expect(model.warmup()).resolves.not.toThrow();
            model.cleanup();
        });
    });

    describe("compute", () => {
        let model: Model;

        beforeAll(async () => {
            model = await Model.fromPath(MODEL_PATH);
            await model.warmup();
        });

        afterAll(() => {
            model.cleanup();
        });

        it("should throw error when computing before warmup", async () => {
            const freshModel = await Model.fromPath(MODEL_PATH);
            const input = new NDArray([1, 3], new Float32Array([1.0, 2.0, 3.0]), "float32");
            await expect(freshModel.compute({ x: input }, {})).rejects.toThrowError(
                ModelNotWarmedUpError
            );
            freshModel.cleanup();
        });

        it("should compute single batch input with output shape [1,1]", async () => {
            const input = new NDArray([1, 3], new Float32Array([1.0, 2.0, 3.0]), "float32");
            const output = await model.compute({ x: input }, {});

            const manifest = model.getManifest();
            const outputName = manifest.outputs[0].name;
            const result = output[outputName];

            expect(result).toBeInstanceOf(NDArray);
            expect(result.shape).toEqual([1, 1]);
        });

        it("should compute multiple batch inputs with output shape [3,1]", async () => {
            const inputData = new Float32Array([
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            ]);
            const input = new NDArray([3, 3], inputData, "float32");
            const output = await model.compute({ x: input }, {});

            const manifest = model.getManifest();
            const outputName = manifest.outputs[0].name;
            const result = output[outputName];

            expect(result).toBeInstanceOf(NDArray);
            expect(result.shape).toEqual([3, 1]);
            expect(result.toArray().length).toBe(3);
        });

        it("should produce deterministic outputs", async () => {
            const input = new NDArray([1, 3], new Float32Array([1.0, 2.0, 3.0]), "float32");

            const manifest = model.getManifest();
            const outputName = manifest.outputs[0].name;

            const output1 = await model.compute({ x: input }, {});
            const output2 = await model.compute({ x: input }, {});

            const data1 = output1[outputName].toArray();
            const data2 = output2[outputName].toArray();

            expect(data1).toEqual(data2);
        });

        it("should produce different outputs for different inputs", async () => {
            const manifest = model.getManifest();
            const outputName = manifest.outputs[0].name;

            const input1 = new NDArray([1, 3], new Float32Array([1.0, 2.0, 3.0]), "float32");
            const input2 = new NDArray([1, 3], new Float32Array([4.0, 5.0, 6.0]), "float32");

            const output1 = await model.compute({ x: input1 }, {});
            const output2 = await model.compute({ x: input2 }, {});

            const data1 = output1[outputName].toArray();
            const data2 = output2[outputName].toArray();

            expect(JSON.stringify(data1)).not.toBe(JSON.stringify(data2));
        });
    });

    describe("metadata", () => {
        let model: Model;

        beforeAll(async () => {
            model = await Model.fromPath(MODEL_PATH);
        });

        afterAll(() => {
            model.cleanup();
        });

        it("should return manifest data", () => {
            const manifest = model.getManifest();

            expect(manifest).toBeDefined();
            expect(manifest.variant).toBe("pipeline");
            expect(manifest.inputs).toHaveLength(1);
            expect(manifest.outputs).toHaveLength(1);
            expect(manifest.inputs[0].name).toBe("x");
        });

        it("should return a deep copy of manifest", () => {
            const m1 = model.getManifest();
            const m2 = model.getManifest();
            expect(m1).toEqual(m2);
            m1.variant = "modified";
            expect(model.getManifest().variant).toBe("pipeline");
        });

        it("should return dtypes", () => {
            const dtypes = model.getDtypes();
            expect(dtypes).toBeDefined();
            expect(typeof dtypes).toBe("object");
        });
    });

    describe("ONNX arity", () => {
        it("should reject an op instance whose arity differs from its ONNX model", async () => {
            const directory = await extractModelToDir();
            const ops = JSON.parse(
                readFileSync(path.join(directory, "ops.json"), "utf-8")
            ) as Array<{ id: string; inputs: unknown[] }>;
            const linreg = ops.find((instance) => instance.id === "linreg")!;
            linreg.inputs = [...linreg.inputs, { ...(linreg.inputs[0] as object) }];
            writeFileSync(path.join(directory, "ops.json"), JSON.stringify(ops));

            const variantConfig = JSON.parse(
                readFileSync(path.join(directory, "variant_config.json"), "utf-8")
            ) as { nodes: Array<{ op_instance_id: string; inputs: string[] }> };
            variantConfig.nodes[0].inputs = ["x", "x"];
            writeFileSync(
                path.join(directory, "variant_config.json"),
                JSON.stringify(variantConfig)
            );

            const model = await Model.fromPath(directory);
            try {
                await expect(model.warmup()).rejects.toThrowError(
                    InvalidOperationDeclarationError
                );
                await model.warmup().catch((error: unknown) => {
                    expect(error).toMatchObject({ opInstanceId: "linreg", field: "inputs" });
                });
            } finally {
                rmSync(directory, { recursive: true, force: true });
            }
        });
    });

    describe("compressed archives", () => {
        // Consumers MAY accept a transparently compressed stream; this host does.
        it("should load a gzip-compressed archive from a buffer", async () => {
            const model = await Model.fromBuffer(gzipSync(readFileSync(MODEL_PATH)));

            expect(model.getManifest().variant).toBe("pipeline");
            model.cleanup();
        });

        it("should load a gzip-compressed archive from a path", async () => {
            const directory = mkdtempSync(path.join(tmpdir(), "fnnx-gzip-"));
            const compressedPath = path.join(directory, "model.fnnx.tar.gz");
            writeFileSync(compressedPath, gzipSync(readFileSync(MODEL_PATH)));
            try {
                const model = await Model.fromPath(compressedPath);

                expect(model.getManifest().variant).toBe("pipeline");
                model.cleanup();
            } finally {
                rmSync(directory, { recursive: true, force: true });
            }
        });
    });
});
