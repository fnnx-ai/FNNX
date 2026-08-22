import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MissingArtifactFileError } from "@fnnx-ai/common";
import { NodeArtifactSource } from "../src/source";

const directories: string[] = [];

function makeArtifact(): string {
    const directory = mkdtempSync(path.join(tmpdir(), "fnnx-source-test-"));
    directories.push(directory);
    mkdirSync(path.join(directory, "ops_artifacts", "predict", "nested"), { recursive: true });
    writeFileSync(path.join(directory, "manifest.json"), "{}");
    writeFileSync(path.join(directory, "ops_artifacts", "predict", "model.onnx"), "model");
    writeFileSync(path.join(directory, "ops_artifacts", "predict", "nested", "data.bin"), "data");
    return directory;
}

afterEach(() => {
    for (const directory of directories) {
        rmSync(directory, { recursive: true, force: true });
    }
    directories.length = 0;
});

describe("NodeArtifactSource", () => {
    it("lists root files and resolves one operation's artifacts", () => {
        const source = new NodeArtifactSource(makeArtifact());

        expect(source.listRootMembers()).toEqual(["manifest.json"]);
        expect(source.resolveOpArtifacts("predict").map((file) => file.path).sort()).toEqual([
            "ops_artifacts/predict/model.onnx",
            "ops_artifacts/predict/nested/data.bin",
        ]);
        expect(new TextDecoder().decode(source.readFile("manifest.json").read())).toBe("{}");
    });

    it("throws a typed error for a missing file", () => {
        const source = new NodeArtifactSource(makeArtifact());

        expect(() => source.readFile("ops.json")).toThrowError(MissingArtifactFileError);
    });
});
