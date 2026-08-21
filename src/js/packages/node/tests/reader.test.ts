import { describe, it, expect, afterEach, vi } from "vitest";
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Model } from "../src/model";

function createTarEntry(fileName: string, content: string): Uint8Array {
    const contentBytes = new TextEncoder().encode(content);
    const alignedSize = Math.ceil(contentBytes.length / 512) * 512;
    const block = new Uint8Array(512 + alignedSize);

    const encoder = new TextEncoder();
    block.set(encoder.encode(fileName), 0);
    block.set(encoder.encode("0000644"), 100);
    block.set(encoder.encode("0000000"), 108);
    block.set(encoder.encode("0000000"), 116);

    const sizeOctal = contentBytes.length.toString(8).padStart(11, "0");
    block.set(encoder.encode(sizeOctal), 124);

    block.set(encoder.encode("00000000000"), 136);
    block.set(encoder.encode("        "), 148);
    block.set(encoder.encode("0"), 156);

    let checksum = 0;
    for (let i = 0; i < 512; i++) {
        checksum += block[i];
    }
    const checksumStr = checksum.toString(8).padStart(6, "0") + "\0 ";
    block.set(encoder.encode(checksumStr), 148);

    block.set(contentBytes, 512);

    return block;
}

function createMultiFileTar(files: Array<{ name: string; content: string }>): ArrayBuffer {
    const entries = files.map((f) => createTarEntry(f.name, f.content));
    const totalSize = entries.reduce((sum, e) => sum + e.length, 0) + 1024;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const entry of entries) {
        view.set(entry, offset);
        offset += entry.length;
    }
    return buffer;
}

const BASE_MANIFEST = {
    variant: "pipeline",
    description: "test model",
    producer_name: "test",
    producer_version: "1.0.0",
    producer_tags: ["tag1"],
    inputs: [
        { name: "x", content_type: "NDJSON", dtype: "Array[float32]", shape: [] },
    ],
    outputs: [
        { name: "y", content_type: "NDJSON", dtype: "Array[float32]", shape: [] },
    ],
    dynamic_attributes: [],
    env_vars: [],
};

const BASE_META: Array<{ id: string; producer: string; producer_version: string; producer_tags: string[]; payload: Record<string, any> }> = [
    {
        id: "entry1",
        producer: "test",
        producer_version: "1.0.0",
        producer_tags: ["tag1"],
        payload: { key: "value1" },
    },
];

function makeBaseFiles(overrides?: {
    manifest?: object;
    meta?: any;
    extraFiles?: Array<{ name: string; content: string }>;
}): Array<{ name: string; content: string }> {
    const manifest = overrides?.manifest ?? BASE_MANIFEST;
    const meta = overrides?.meta ?? BASE_META;
    const files = [
        { name: "manifest.json", content: JSON.stringify(manifest) },
        { name: "ops.json", content: "[]" },
        { name: "variant_config.json", content: JSON.stringify({ nodes: [] }) },
        { name: "meta.json", content: JSON.stringify(meta) },
        { name: "dtypes.json", content: "{}" },
        { name: "env.json", content: "{}" },
    ];
    if (overrides?.extraFiles) {
        files.push(...overrides.extraFiles);
    }
    return files;
}

describe("Model reader: manifest patches", () => {
    const tempDirs: string[] = [];

    function makeDirFromFiles(files: Array<{ name: string; content: string }>): string {
        const dir = mkdtempSync(path.join(tmpdir(), "fnnx-reader-test-"));
        tempDirs.push(dir);
        for (const f of files) {
            const filePath = path.join(dir, f.name);
            mkdirSync(path.dirname(filePath), { recursive: true });
            writeFileSync(filePath, f.content);
        }
        return dir;
    }

    afterEach(() => {
        for (const dir of tempDirs) {
            try { rmSync(dir, { recursive: true, force: true }); } catch {}
        }
        tempDirs.length = 0;
    });

    it("should load manifest without patches (no patch files)", async () => {
        const dir = makeDirFromFiles(makeBaseFiles());
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.description).toBe("test model");
        expect(manifest.producer_version).toBe("1.0.0");
    });

    it("should apply a single manifest patch", async () => {
        const patch = [
            { op: "replace", path: "/description", value: "patched model" },
            { op: "replace", path: "/producer_version", value: "2.0.0" },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "manifest-001.patch.json", content: JSON.stringify(patch) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.description).toBe("patched model");
        expect(manifest.producer_version).toBe("2.0.0");
    });

    it("should apply multiple manifest patches in sorted order", async () => {
        const patch1 = [
            { op: "replace", path: "/description", value: "first patch" },
        ];
        const patch2 = [
            { op: "replace", path: "/description", value: "second patch" },
            { op: "add", path: "/producer_tags/-", value: "tag2" },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "manifest-002.patch.json", content: JSON.stringify(patch2) },
                { name: "manifest-001.patch.json", content: JSON.stringify(patch1) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.description).toBe("second patch");
        expect(manifest.producer_tags).toEqual(["tag1", "tag2"]);
    });

    it("should apply patch that adds a new field", async () => {
        const patch = [
            { op: "add", path: "/name", value: "my_model" },
            { op: "add", path: "/version", value: "3.0.0" },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "manifest-abc.patch.json", content: JSON.stringify(patch) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.name).toBe("my_model");
        expect(manifest.version).toBe("3.0.0");
    });

    it("should not treat non-matching files as patches", async () => {
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "manifest.patch.json", content: JSON.stringify([{ op: "replace", path: "/description", value: "nope" }]) },
                { name: "manifest_extra.json", content: JSON.stringify({ bad: true }) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.description).toBe("test model");
    });

    it("should apply manifest patches from buffer", async () => {
        const patch = [
            { op: "replace", path: "/description", value: "buffer patched" },
        ];
        const tar = createMultiFileTar(makeBaseFiles({
            extraFiles: [
                { name: "manifest-001.patch.json", content: JSON.stringify(patch) },
            ],
        }));
        const model = await Model.fromBuffer(Buffer.from(tar));
        const manifest = model.getManifest();
        expect(manifest.description).toBe("buffer patched");
        model.cleanup();
    });

    it("keeps only the last repeated manifest and metadata members", async () => {
        const replacement = { ...BASE_MANIFEST, description: "last manifest" };
        const firstMetadata = [
            {
                id: "first",
                producer: "tests",
                producer_version: "1",
                producer_tags: [],
                payload: {},
            },
        ];
        const lastMetadata = [
            {
                id: "last",
                producer: "tests",
                producer_version: "1",
                producer_tags: [],
                payload: {},
            },
        ];
        const tar = createMultiFileTar([
            ...makeBaseFiles({ meta: [] }),
            { name: "meta-x.json", content: JSON.stringify(firstMetadata) },
            { name: "manifest.json", content: JSON.stringify(replacement) },
            { name: "meta-x.json", content: JSON.stringify(lastMetadata) },
        ]);

        const model = await Model.fromBuffer(Buffer.from(tar));

        expect(model.getManifest().description).toBe("last manifest");
        expect(model.getMetadata().map((entry) => entry.id)).toEqual(["last"]);
        model.cleanup();
    });

    it("applies ordered patches and tolerates malformed metadata in a local tar fixture", async () => {
        const patchB = [{ op: "replace", path: "/description", value: "patch-b" }];
        const patchA = [{ op: "replace", path: "/description", value: "patch-a" }];
        const metadata = (id: string): object => ({
            id,
            producer: "tests",
            producer_version: "1",
            producer_tags: [],
            payload: {},
        });
        const tar = createMultiFileTar([
            ...makeBaseFiles({ meta: [metadata("base")] }),
            { name: "manifest-b.patch.json", content: JSON.stringify(patchB) },
            { name: "manifest-a.patch.json", content: JSON.stringify(patchA) },
            { name: "meta-b.json", content: JSON.stringify([metadata("b")]) },
            {
                name: "meta-a.json",
                content: JSON.stringify([
                    {
                        id: "invalid",
                        producer_version: "1",
                        producer_tags: [],
                        payload: {},
                    },
                    metadata("a"),
                ]),
            },
            { name: "meta-bad.json", content: "{" },
        ]);
        const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        let model: Model | undefined;
        try {
            model = await Model.fromBuffer(Buffer.from(tar));

            expect(model.getManifest().description).toBe("patch-b");
            expect(model.getMetadata().map((entry) => entry.id)).toEqual(["base", "a", "b"]);
            expect(warning).toHaveBeenCalledWith(expect.stringContaining("meta-bad.json"));
        } finally {
            model?.cleanup();
            warning.mockRestore();
        }
    });

    it("should replace a nested value in inputs via patch", async () => {
        const patch = [
            { op: "replace", path: "/inputs/0/dtype", value: "Array[int32]" },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "manifest-dtype.patch.json", content: JSON.stringify(patch) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const manifest = model.getManifest();
        expect(manifest.inputs[0].dtype).toBe("Array[int32]");
    });
});

describe("Model reader: multi-file metadata", () => {
    const tempDirs: string[] = [];

    function makeDirFromFiles(files: Array<{ name: string; content: string }>): string {
        const dir = mkdtempSync(path.join(tmpdir(), "fnnx-reader-test-"));
        tempDirs.push(dir);
        for (const f of files) {
            const filePath = path.join(dir, f.name);
            mkdirSync(path.dirname(filePath), { recursive: true });
            writeFileSync(filePath, f.content);
        }
        return dir;
    }

    afterEach(() => {
        for (const dir of tempDirs) {
            try { rmSync(dir, { recursive: true, force: true }); } catch {}
        }
        tempDirs.length = 0;
    });

    it("should load metadata from single meta.json", async () => {
        const dir = makeDirFromFiles(makeBaseFiles());
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(1);
        expect(metadata[0].id).toBe("entry1");
        expect(metadata[0].payload).toEqual({ key: "value1" });
    });

    it("should combine entries from meta.json and meta-{uid}.json", async () => {
        const extraMeta = [
            {
                id: "entry2",
                producer: "test2",
                producer_version: "2.0.0",
                producer_tags: ["extra"],
                payload: { key: "value2" },
            },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "meta-extra1.json", content: JSON.stringify(extraMeta) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(2);
        const ids = metadata.map((m) => m.id);
        expect(ids).toContain("entry1");
        expect(ids).toContain("entry2");
    });

    it("should combine entries from multiple meta-{uid}.json files", async () => {
        const meta2 = [
            {
                id: "entry2",
                producer: "p2",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: {},
            },
        ];
        const meta3 = [
            {
                id: "entry3",
                producer: "p3",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: { nested: { data: true } },
            },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            extraFiles: [
                { name: "meta-uid1.json", content: JSON.stringify(meta2) },
                { name: "meta-uid2.json", content: JSON.stringify(meta3) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(3);
        const ids = metadata.map((m) => m.id);
        expect(ids).toContain("entry1");
        expect(ids).toContain("entry2");
        expect(ids).toContain("entry3");
    });

    it("should handle meta.json with multiple entries per file", async () => {
        const multiEntryMeta = [
            {
                id: "a",
                producer: "p",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: {},
            },
            {
                id: "b",
                producer: "p",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: {},
            },
        ];
        const dir = makeDirFromFiles(makeBaseFiles({
            meta: multiEntryMeta,
        }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(2);
        expect(metadata.map((m) => m.id)).toEqual(["a", "b"]);
    });

    it("should return empty array when meta.json is empty object", async () => {
        const dir = makeDirFromFiles(makeBaseFiles({ meta: {} }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toEqual([]);
    });

    it("should return empty array when meta.json is empty array", async () => {
        const dir = makeDirFromFiles(makeBaseFiles({ meta: [] }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toEqual([]);
    });

    it("should not match metadata.json or meta_stuff.json", async () => {
        const dir = makeDirFromFiles(makeBaseFiles({
            meta: [],
            extraFiles: [
                { name: "metadata.json", content: JSON.stringify([{ id: "bad", producer: "x", producer_version: "1", producer_tags: [], payload: {} }]) },
                { name: "meta_stuff.json", content: JSON.stringify([{ id: "bad2", producer: "x", producer_version: "1", producer_tags: [], payload: {} }]) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toEqual([]);
    });

    it("should not match meta files in subdirectories", async () => {
        const dir = makeDirFromFiles(makeBaseFiles({
            meta: [],
            extraFiles: [
                { name: "subdir/meta-nested.json", content: JSON.stringify([{ id: "nested", producer: "x", producer_version: "1", producer_tags: [], payload: {} }]) },
            ],
        }));
        const model = await Model.fromPath(dir);
        const metadata = model.getMetadata();
        expect(metadata).toEqual([]);
    });

    it("should load multi-file metadata from buffer", async () => {
        const extraMeta = [
            {
                id: "buf_entry",
                producer: "buf_producer",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: { from: "buffer" },
            },
        ];
        const tar = createMultiFileTar(makeBaseFiles({
            extraFiles: [
                { name: "meta-buffer.json", content: JSON.stringify(extraMeta) },
            ],
        }));
        const model = await Model.fromBuffer(Buffer.from(tar));
        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(2);
        const ids = metadata.map((m) => m.id);
        expect(ids).toContain("entry1");
        expect(ids).toContain("buf_entry");
        model.cleanup();
    });
});

describe("Model reader: combined manifest patches + multi-file metadata", () => {
    it("should handle both patches and multi-file metadata from buffer", async () => {
        const patch = [
            { op: "replace", path: "/description", value: "fully patched" },
        ];
        const extraMeta = [
            {
                id: "combined_entry",
                producer: "combined",
                producer_version: "1.0.0",
                producer_tags: [],
                payload: {},
            },
        ];
        const tar = createMultiFileTar(makeBaseFiles({
            extraFiles: [
                { name: "manifest-combined.patch.json", content: JSON.stringify(patch) },
                { name: "meta-combined.json", content: JSON.stringify(extraMeta) },
            ],
        }));
        const model = await Model.fromBuffer(Buffer.from(tar));

        const manifest = model.getManifest();
        expect(manifest.description).toBe("fully patched");

        const metadata = model.getMetadata();
        expect(metadata).toHaveLength(2);
        expect(metadata.map((m) => m.id)).toContain("combined_entry");

        model.cleanup();
    });
});
