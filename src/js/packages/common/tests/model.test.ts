import { describe, expect, it, vi } from "vitest";
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

const ARRAY_IO = {
    content_type: "NDJSON",
    dtype: "Array[float32]",
    shape: [],
};

const OP_IO = {
    dtype: "Array[float32]",
    shape: [],
};

function modelIO(name: string, contentType: "NDJSON" | "JSON" = "NDJSON"): object {
    if (contentType === "JSON") {
        return { name, content_type: contentType, dtype: "ext::record" };
    }
    return { name, ...ARRAY_IO };
}

function opInstance(
    inputCount = 0,
    outputCount = 0,
    dynamicAttributes: Record<string, { name: string; default_value: string }> = {}
): object {
    return {
        id: "test",
        op: "test",
        inputs: Array.from({ length: inputCount }, () => ({ ...OP_IO })),
        outputs: Array.from({ length: outputCount }, () => ({ ...OP_IO })),
        attributes: {},
        dynamic_attributes: dynamicAttributes,
    };
}

function pipelineNode(inputs: string[] = [], outputs: string[] = []): object {
    return {
        op_instance_id: "test",
        inputs,
        outputs,
        extra_dynattrs: {},
    };
}

function metadataEntry(id: string): object {
    return {
        id,
        producer: "tests",
        producer_version: "1",
        producer_tags: [],
        payload: {},
    };
}

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

    it("applies manifest patches in UTF-8 byte order", () => {
        const astralSuffix = "\u{10000}";
        const privateUseSuffix = "\uE000";
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    [`manifest-${astralSuffix}.patch.json`]: [
                        { op: "replace", path: "/description", value: "astral" },
                    ],
                    [`manifest-${privateUseSuffix}.patch.json`]: [
                        { op: "replace", path: "/description", value: "private-use" },
                    ],
                })
            ),
            { operators: {} }
        );

        expect(model.getManifest().description).toBe("astral");
    });

    it("assembles metadata in filename order and skips invalid entries", () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "meta-b.json": [metadataEntry("b")],
                    "meta-a.json": [
                        {
                            id: "invalid",
                            producer_version: "1",
                            producer_tags: [],
                            payload: {},
                        },
                        metadataEntry("a"),
                    ],
                })
            ),
            { operators: {} }
        );

        expect(model.getMetadata().map((entry) => entry.id)).toEqual(["base", "a", "b"]);
    });

    it("warns about an unparseable metadata sidecar and reads the others", () => {
        const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        try {
            const model = new Model(
                new MemorySource(
                    modelDocuments({
                        "meta-bad.json": new TextEncoder().encode("{"),
                        "meta-good.json": [metadataEntry("good")],
                    })
                ),
                { operators: {} }
            );

            expect(model.getMetadata().map((entry) => entry.id)).toEqual(["base", "good"]);
            expect(warning).toHaveBeenCalledWith(expect.stringContaining("meta-bad.json"));
        } finally {
            warning.mockRestore();
        }
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

    it("rejects a reserved dtype name while loading an artifact", () => {
        const source = new MemorySource(modelDocuments({ "dtypes.json": { boolean: {} } }));

        expect(() => new Model(source, { operators: {} })).toThrow(/Invalid dtype name: boolean/);
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

class RequiredDynamicAttributeOp extends DynamicAttributeOp {
    protected static requiredDynamicAttributes = ["internal"];
}

class InputTrackingOp extends BaseOp {
    static calls = 0;

    async warmup(): Promise<this> {
        this.setWarmedUp(true);
        return this;
    }

    protected async run(inputs: unknown[]): Promise<OpOutput> {
        InputTrackingOp.calls += 1;
        return { value: [inputs[0]] };
    }
}

describe("pipeline validation", () => {
    it("rejects a node that references an undeclared op instance", () => {
        const source = new MemorySource(
            modelDocuments({
                "variant_config.json": {
                    nodes: [
                        {
                            op_instance_id: "missing",
                            inputs: [],
                            outputs: [],
                            extra_dynattrs: {},
                        },
                    ],
                },
            })
        );

        expect(() => new Model(source, { operators: {} })).toThrow(/missing/);
    });

    it.each([
        { node: pipelineNode([], []), inputCount: 1, outputCount: 0, arity: "input" },
        { node: pipelineNode([], []), inputCount: 0, outputCount: 1, arity: "output" },
    ])("rejects a node whose $arity arity differs from its op instance", (testCase) => {
        const source = new MemorySource(
            modelDocuments({
                "ops.json": [opInstance(testCase.inputCount, testCase.outputCount)],
                "variant_config.json": { nodes: [testCase.node] },
            })
        );

        expect(() => new Model(source, { operators: { test: DynamicAttributeOp } })).toThrow(
            new RegExp(`test.*${testCase.arity}|${testCase.arity}.*test`, "i")
        );
    });

    it.each([
        {
            manifest: { ...MANIFEST, inputs: [modelIO("x"), modelIO("x")] },
            ops: [],
            nodes: [],
            name: "x",
        },
        {
            manifest: { ...MANIFEST, inputs: [modelIO("x")] },
            ops: [opInstance(0, 1)],
            nodes: [pipelineNode([], ["x"])],
            name: "x",
        },
        {
            manifest: MANIFEST,
            ops: [opInstance(0, 2)],
            nodes: [pipelineNode([], ["y", "y"])],
            name: "y",
        },
    ])("rejects value name $name when it is bound twice", (testCase) => {
        const source = new MemorySource(
            modelDocuments({
                "manifest.json": testCase.manifest,
                "ops.json": testCase.ops,
                "variant_config.json": { nodes: testCase.nodes },
            })
        );

        expect(() => new Model(source, { operators: { test: DynamicAttributeOp } })).toThrow(
            new RegExp(testCase.name)
        );
    });

    it("rejects a node input that no earlier value binds", () => {
        const source = new MemorySource(
            modelDocuments({
                "ops.json": [opInstance(1, 0)],
                "variant_config.json": { nodes: [pipelineNode(["unbound"], [])] },
            })
        );

        expect(() => new Model(source, { operators: { test: DynamicAttributeOp } })).toThrow(
            /unbound/
        );
    });

    it.each(["inputs", "outputs"] as const)(
        "rejects JSON content type in manifest $entryKind",
        (entryKind) => {
            const manifest = {
                ...MANIFEST,
                [entryKind]: [modelIO(entryKind === "inputs" ? "x" : "y", "JSON")],
            };
            const source = new MemorySource(modelDocuments({ "manifest.json": manifest }));

            expect(() => new Model(source, { operators: {} })).toThrow(
                new RegExp(entryKind === "inputs" ? "x" : "y")
            );
        }
    );

    it("reports a model input omitted at compute time before invoking the node", async () => {
        InputTrackingOp.calls = 0;
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": {
                        ...MANIFEST,
                        inputs: [modelIO("x")],
                        outputs: [modelIO("y")],
                    },
                    "ops.json": [opInstance(1, 1)],
                    "variant_config.json": { nodes: [pipelineNode(["x"], ["y"])] },
                })
            ),
            { operators: { test: InputTrackingOp } }
        );
        await model.warmup();

        await expect(model.compute({})).rejects.toThrow(/x/);
        expect(InputTrackingOp.calls).toBe(0);
    });

    it("does not pass an absent upstream result to a later node", async () => {
        InputTrackingOp.calls = 0;
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": { ...MANIFEST, outputs: [modelIO("z")] },
                    "ops.json": [
                        {
                            id: "empty",
                            op: "empty",
                            inputs: [],
                            outputs: [{ ...OP_IO }],
                            attributes: {},
                            dynamic_attributes: {},
                        },
                        {
                            id: "tracking",
                            op: "tracking",
                            inputs: [{ ...OP_IO }],
                            outputs: [{ ...OP_IO }],
                            attributes: {},
                            dynamic_attributes: {},
                        },
                    ],
                    "variant_config.json": {
                        nodes: [
                            {
                                op_instance_id: "empty",
                                inputs: [],
                                outputs: ["intermediate"],
                                extra_dynattrs: {},
                            },
                            {
                                op_instance_id: "tracking",
                                inputs: ["intermediate"],
                                outputs: ["z"],
                                extra_dynattrs: {},
                            },
                        ],
                    },
                })
            ),
            { operators: { empty: DynamicAttributeOp, tracking: InputTrackingOp } }
        );
        await model.warmup();

        await expect(model.compute({})).rejects.toThrow(/intermediate/);
        expect(InputTrackingOp.calls).toBe(0);
    });

    it("reports a missing declared output without returning a partial result", async () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": { ...MANIFEST, outputs: [modelIO("y")] },
                })
            ),
            { operators: {} }
        );
        await model.warmup();

        await expect(model.compute({})).rejects.toThrow(/y/);
    });
});

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

    it("resolves supplied and default values through a pipeline invocation", async () => {
        DynamicAttributeOp.received = [];
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "ops.json": [
                        opInstance(0, 0, {
                            internal: { name: "external", default_value: "default" },
                        }),
                    ],
                    "variant_config.json": { nodes: [pipelineNode()] },
                })
            ),
            { operators: { test: RequiredDynamicAttributeOp } }
        );
        await model.warmup();

        await model.compute({}, { external: "" });
        await model.compute({});

        expect(DynamicAttributeOp.received).toEqual([{ internal: "" }, { internal: "default" }]);
    });

    it("rejects a missing dynamic attribute required by the operation", async () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "ops.json": [opInstance()],
                    "variant_config.json": { nodes: [pipelineNode()] },
                })
            ),
            { operators: { test: RequiredDynamicAttributeOp } }
        );
        await model.warmup();

        await expect(model.compute({})).rejects.toThrow(/internal/);
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
