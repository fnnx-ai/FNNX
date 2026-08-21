import { describe, expect, it } from "vitest";
import {
    ArtifactFile,
    ArtifactSource,
    BaseOp,
    DtypesManager,
    MissingArtifactFileError,
    Model,
    ModelNotWarmedUpError,
    OpOutput,
    OpRuntimeConfig,
    UnsupportedOperationError,
    UnsupportedVariantError,
} from "../src/index.js";

class MemoryFile implements ArtifactFile {
    constructor(
        public readonly path: string,
        private readonly bytes: Uint8Array
    ) {}

    read(): Uint8Array {
        return this.bytes.slice();
    }
}

class MemorySource implements ArtifactSource {
    private readonly files: Map<string, MemoryFile>;

    constructor(documents: Record<string, unknown>) {
        const encoder = new TextEncoder();
        this.files = new Map(
            Object.entries(documents).map(([path, value]) => [
                path,
                new MemoryFile(
                    path,
                    value instanceof Uint8Array ? value : encoder.encode(JSON.stringify(value))
                ),
            ])
        );
    }

    listRootMembers(): string[] {
        return [...this.files.keys()].filter((path) => !path.includes("/"));
    }

    readFile(path: string): ArtifactFile {
        const file = this.files.get(path);
        if (!file) {
            throw new MissingArtifactFileError(path);
        }
        return file;
    }

    resolveOpArtifacts(opInstanceId: string): ArtifactFile[] {
        const prefix = `ops_artifacts/${opInstanceId}/`;
        return [...this.files.values()].filter((file) => file.path.startsWith(prefix));
    }
}

const MANIFEST = {
    variant: "pipeline",
    description: "base",
    producer_name: "tests",
    producer_version: "1",
    producer_tags: [],
    inputs: [],
    outputs: [],
    dynamic_attributes: [],
    env_vars: [],
};

function modelDocuments(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
        "manifest.json": MANIFEST,
        "ops.json": [],
        "variant_config.json": { nodes: [] },
        "dtypes.json": { "ext::record": { type: "object" } },
        "env.json": { "example::runtime": { version: "1" } },
        "meta.json": [
            {
                id: "base",
                producer: "tests",
                producer_version: "1",
                producer_tags: [],
                payload: {},
            },
        ],
        ...overrides,
    };
}

describe("shared Model core", () => {
    it("loads patches, metadata, dtypes, and the raw environment", () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest-change.patch.json": [
                        { op: "replace", path: "/description", value: "patched" },
                    ],
                    "meta-extra.json": [
                        {
                            id: "extra",
                            producer: "tests",
                            producer_version: "1",
                            producer_tags: [],
                            payload: { source: "sidecar" },
                        },
                    ],
                })
            ),
            { operators: {} }
        );

        expect(model.getManifest().description).toBe("patched");
        expect(model.getMetadata().map((entry) => entry.id)).toEqual(["base", "extra"]);
        expect(model.getDtypes()).toEqual({ "ext::record": { type: "object" } });
        expect(model.getEnv()).toEqual({ "example::runtime": { version: "1" } });
    });

    it("returns copies from document accessors", () => {
        const model = new Model(new MemorySource(modelDocuments()), { operators: {} });
        const manifest = model.getManifest();
        manifest.description = "changed";
        const environment = model.getEnv();
        environment.changed = true;

        expect(model.getManifest().description).toBe("base");
        expect(model.getEnv()).not.toHaveProperty("changed");
    });

    it("dispatches unsupported variants with a typed error", () => {
        const source = new MemorySource(
            modelDocuments({ "manifest.json": { ...MANIFEST, variant: "pyfunc" } })
        );

        expect(() => new Model(source, { operators: {} })).toThrowError(UnsupportedVariantError);
        try {
            new Model(source, { operators: {} });
        } catch (error) {
            expect(error).toMatchObject({ variant: "pyfunc" });
        }
    });

    it("identifies a missing required file", () => {
        const documents = modelDocuments();
        delete documents["ops.json"];

        expect(() => new Model(new MemorySource(documents), { operators: {} })).toThrowError(
            MissingArtifactFileError
        );
    });

    it("identifies an unsupported operation and its instance", () => {
        const source = new MemorySource(
            modelDocuments({
                "ops.json": [
                    {
                        id: "custom",
                        op: "example::op_v1",
                        inputs: [],
                        outputs: [],
                        attributes: {},
                        dynamic_attributes: {},
                    },
                ],
            })
        );

        expect(() => new Model(source, { operators: {} })).toThrowError(UnsupportedOperationError);
        try {
            new Model(source, { operators: {} });
        } catch (error) {
            expect(error).toMatchObject({ opType: "example::op_v1", opInstanceId: "custom" });
        }
    });

    it("requires warmup before compute", async () => {
        const model = new Model(new MemorySource(modelDocuments()), { operators: {} });

        await expect(model.compute({})).rejects.toThrowError(ModelNotWarmedUpError);
        await model.warmup();
        await expect(model.compute({})).resolves.toEqual({});
    });
});

class DynamicAttributeOp extends BaseOp {
    static received: Record<string, string>[] = [];

    async warmup(): Promise<this> {
        this.setWarmedUp(true);
        return this;
    }

    protected async run(
        _inputs: unknown[],
        dynamicAttributes: Record<string, string>
    ): Promise<OpOutput> {
        DynamicAttributeOp.received.push(dynamicAttributes);
        return { value: [] };
    }
}

describe("wire-shaped operation declarations", () => {
    it("uses dynamic_attributes and default_value fields", async () => {
        DynamicAttributeOp.received = [];
        const config: OpRuntimeConfig = {
            op_instance_id: "test",
            attributes: {},
            dynamic_attributes: { internal: { name: "external", default_value: "default" } },
            inputs: [],
            outputs: [],
            dtypes_manager: new DtypesManager(),
        };
        const operation = new DynamicAttributeOp([], config);

        await operation.compute([], { external: "" });
        expect(DynamicAttributeOp.received.at(-1)).toEqual({ internal: "" });
        await operation.compute([], {});
        expect(DynamicAttributeOp.received.at(-1)).toEqual({ internal: "default" });
    });

    it("does not leak extra_dynattrs between pipeline nodes", async () => {
        DynamicAttributeOp.received = [];
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "ops.json": [
                        {
                            id: "test",
                            op: "test",
                            inputs: [],
                            outputs: [],
                            attributes: {},
                            dynamic_attributes: {
                                internal: { name: "n", default_value: "default" },
                            },
                        },
                    ],
                    "variant_config.json": {
                        nodes: [
                            {
                                op_instance_id: "test",
                                inputs: [],
                                outputs: [],
                                extra_dynattrs: { n: "pinned" },
                            },
                            {
                                op_instance_id: "test",
                                inputs: [],
                                outputs: [],
                                extra_dynattrs: {},
                            },
                        ],
                    },
                })
            ),
            { operators: { test: DynamicAttributeOp } }
        );

        await model.warmup();
        await model.compute({}, { n: "caller" });

        expect(DynamicAttributeOp.received).toEqual([
            { internal: "pinned" },
            { internal: "caller" },
        ]);
    });
});
