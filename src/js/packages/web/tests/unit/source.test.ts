import { describe, expect, it } from "vitest";
import { MissingArtifactFileError, Model } from "@fnnx-ai/common";
import { WebArtifactSource } from "../../src/source";

function tarEntry(path: string, content: string): Uint8Array {
    const encoder = new TextEncoder();
    const contentBytes = encoder.encode(content);
    const entry = new Uint8Array(512 + Math.ceil(contentBytes.length / 512) * 512);
    entry.set(encoder.encode(path), 0);
    entry.set(encoder.encode("0000644"), 100);
    entry.set(encoder.encode("0000000"), 108);
    entry.set(encoder.encode("0000000"), 116);
    entry.set(encoder.encode(contentBytes.length.toString(8).padStart(11, "0")), 124);
    entry.set(encoder.encode("00000000000"), 136);
    entry.set(encoder.encode("        "), 148);
    entry.set(encoder.encode("0"), 156);
    let checksum = 0;
    for (let index = 0; index < 512; index++) {
        checksum += entry[index];
    }
    entry.set(encoder.encode(checksum.toString(8).padStart(6, "0") + "\0 "), 148);
    entry.set(contentBytes, 512);
    return entry;
}

function tarBuffer(entries: Uint8Array[]): ArrayBuffer {
    const length = entries.reduce((total, entry) => total + entry.length, 1024);
    const buffer = new Uint8Array(length);
    let offset = 0;
    for (const entry of entries) {
        buffer.set(entry, offset);
        offset += entry.length;
    }
    return buffer.buffer;
}

describe("WebArtifactSource", () => {
    it("lists root files, resolves op artifacts, and keeps repeated members last", () => {
        const firstManifest = {
            variant: "pipeline",
            description: "first",
            producer_name: "tests",
            producer_version: "1",
            producer_tags: [],
            inputs: [],
            outputs: [],
            dynamic_attributes: [],
            env_vars: [],
        };
        const lastManifest = { ...firstManifest, description: "last" };
        const metadata = (id: string): object[] => [
            {
                id,
                producer: "tests",
                producer_version: "1",
                producer_tags: [],
                payload: {},
            },
        ];
        const buffer = tarBuffer([
            tarEntry("manifest.json", JSON.stringify(firstManifest)),
            tarEntry("ops.json", "[]"),
            tarEntry("variant_config.json", '{"nodes":[]}'),
            tarEntry("dtypes.json", "{}"),
            tarEntry("env.json", "{}"),
            tarEntry("meta.json", "[]"),
            tarEntry("meta-x.json", JSON.stringify(metadata("first"))),
            tarEntry("ops_artifacts/predict/model.onnx", "model"),
            tarEntry("manifest.json", JSON.stringify(lastManifest)),
            tarEntry("meta-x.json", JSON.stringify(metadata("last"))),
        ]);
        const source = new WebArtifactSource(buffer);

        expect(source.listRootMembers().sort()).toEqual(
            [
                "dtypes.json",
                "env.json",
                "manifest.json",
                "meta-x.json",
                "meta.json",
                "ops.json",
                "variant_config.json",
            ].sort()
        );
        expect(JSON.parse(new TextDecoder().decode(source.readFile("manifest.json").read()))).toEqual(
            lastManifest
        );
        expect(source.resolveOpArtifacts("predict").map((file) => file.path)).toEqual([
            "ops_artifacts/predict/model.onnx",
        ]);

        const model = new Model(source, { operators: {} });
        expect(model.getManifest().description).toBe("last");
        expect(model.getMetadata().map((entry) => entry.id)).toEqual(["last"]);
    });

    it("throws a typed error for a missing file", () => {
        const source = new WebArtifactSource(tarBuffer([]));

        expect(() => source.readFile("manifest.json")).toThrowError(MissingArtifactFileError);
    });

    it("ignores unknown root files and unknown directories while loading", () => {
        const manifest = {
            variant: "pipeline",
            description: "kept",
            producer_name: "tests",
            producer_version: "1",
            producer_tags: [],
            inputs: [],
            outputs: [],
            dynamic_attributes: [],
            env_vars: [],
        };
        const entry = {
            id: "base",
            producer: "tests",
            producer_version: "1",
            producer_tags: [],
            payload: {},
        };
        const source = new WebArtifactSource(
            tarBuffer([
                tarEntry("manifest.json", JSON.stringify(manifest)),
                tarEntry("ops.json", "[]"),
                tarEntry("variant_config.json", '{"nodes":[]}'),
                tarEntry("meta.json", JSON.stringify([entry])),
                tarEntry("README.md", "not part of the spec"),
                tarEntry("unknown.json", '{"ignored":true}'),
                tarEntry("future_directory/payload.bin", "binary"),
                tarEntry("meta_artifacts/orphan/data.json", "{}"),
            ])
        );

        expect(source.listRootMembers().sort()).toEqual([
            "README.md",
            "manifest.json",
            "meta.json",
            "ops.json",
            "unknown.json",
            "variant_config.json",
        ]);

        const model = new Model(source, { operators: {} });
        expect(model.getManifest().description).toBe("kept");
        expect(model.getMetadata().map((item) => item.id)).toEqual(["base"]);
        expect(model.getDtypes()).toEqual({});
        expect(model.getEnv()).toEqual({});
    });
});
