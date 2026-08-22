import { describe, expect, it, vi } from "vitest";
import {
    ArrayDType,
    ArtifactFile,
    ArtifactSource,
    BaseOp,
    DtypesManager,
    interfaces,
    InvalidArtifactFileError,
    MissingArtifactFileError,
    Model,
    ModelNotWarmedUpError,
    NDArray,
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

    it("dispatches unsupported variants with a typed error when execution is attempted", async () => {
        const source = new MemorySource(
            modelDocuments({ "manifest.json": { ...MANIFEST, variant: "pyfunc" } })
        );
        const model = new Model(source, { operators: {} });

        await expect(model.warmup()).rejects.toThrowError(UnsupportedVariantError);
        await model.warmup().catch((error: unknown) => {
            expect(error).toMatchObject({ variant: "pyfunc" });
        });
    });

    it("reads Core-level declarations of an artifact it cannot execute", () => {
        const source = new MemorySource(
            modelDocuments({ "manifest.json": { ...MANIFEST, variant: "pyfunc" } })
        );
        const model = new Model(source, { operators: {} });

        expect(model.getManifest().variant).toBe("pyfunc");
        expect(model.getMetadata().map((entry) => entry.id)).toEqual(["base"]);
        expect(model.getDtypes()).toEqual({ "ext::record": { type: "object" } });
        expect(model.getEnv()).toEqual({ "example::runtime": { version: "1" } });
    });

    it("identifies a missing required file", () => {
        const documents = modelDocuments();
        delete documents["ops.json"];

        expect(() => new Model(new MemorySource(documents), { operators: {} })).toThrowError(
            MissingArtifactFileError
        );
    });

    it("identifies an unsupported operation and its instance", async () => {
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
        const model = new Model(source, { operators: {} });

        await expect(model.warmup()).rejects.toThrowError(UnsupportedOperationError);
        await model.warmup().catch((error: unknown) => {
            expect(error).toMatchObject({ opType: "example::op_v1", opInstanceId: "custom" });
        });
    });

    it("requires warmup before compute", async () => {
        const model = new Model(new MemorySource(modelDocuments()), { operators: {} });

        await expect(model.compute({})).rejects.toThrowError(ModelNotWarmedUpError);
        await model.warmup();
        await expect(model.compute({})).resolves.toEqual({});
    });
});

describe("effective manifest", () => {
    function manifestOf(overrides: Record<string, unknown>): interfaces.Manifest {
        return new Model(new MemorySource(modelDocuments(overrides)), {
            operators: {},
        }).getManifest();
    }

    it("applies several patch files in ascending byte order, not discovery order", () => {
        const manifest = manifestOf({
            "manifest-c.patch.json": [{ op: "replace", path: "/description", value: "c" }],
            "manifest-a.patch.json": [{ op: "replace", path: "/description", value: "a" }],
            "manifest-b.patch.json": [{ op: "replace", path: "/description", value: "b" }],
        });

        expect(manifest.description).toBe("c");
    });

    it("orders patch files by byte value, so an upper-case suffix comes first", () => {
        const manifest = manifestOf({
            "manifest-a.patch.json": [{ op: "replace", path: "/description", value: "lower" }],
            "manifest-B.patch.json": [{ op: "replace", path: "/description", value: "upper" }],
        });

        expect(manifest.description).toBe("lower");
    });

    it("applies the operations of one patch file in document order", () => {
        const manifest = manifestOf({
            "manifest-x.patch.json": [
                { op: "replace", path: "/description", value: "first" },
                { op: "replace", path: "/description", value: "second" },
            ],
        });

        expect(manifest.description).toBe("second");
    });

    it("patches array members and nested objects", () => {
        const manifest = manifestOf({
            "manifest.json": { ...MANIFEST, inputs: [modelIO("x")], producer_tags: ["base"] },
            "manifest-x.patch.json": [
                { op: "add", path: "/producer_tags/-", value: "appended" },
                { op: "add", path: "/producer_tags/0", value: "prepended" },
                { op: "replace", path: "/inputs/0/dtype", value: "Array[int32]" },
                { op: "add", path: "/inputs/0/tags", value: ["extra"] },
                { op: "add", path: "/inputs/-", value: modelIO("second") },
            ],
        });

        expect(manifest.producer_tags).toEqual(["prepended", "base", "appended"]);
        expect(manifest.inputs[0]).toMatchObject({ dtype: "Array[int32]", tags: ["extra"] });
        expect(manifest.inputs.map((entry) => entry.name)).toEqual(["x", "second"]);
    });

    it.each(["remove", "move", "copy", "test"])(
        "aborts the load on a patch document that uses %s",
        (operation) => {
            const source = new MemorySource(
                modelDocuments({
                    "manifest-x.patch.json": [
                        { op: "replace", path: "/description", value: "applied" },
                        { op: operation, path: "/description", from: "/variant", value: "base" },
                    ],
                })
            );

            expect(() => new Model(source, { operators: {} })).toThrow(
                new RegExp(`Unsupported JSON Patch op.*${operation}`)
            );
        }
    );

    it("ignores root files that only resemble a patch name", () => {
        const manifest = manifestOf({
            "manifest.patch.json": [{ op: "replace", path: "/description", value: "no uid" }],
            "manifest-x.patch.json5": [{ op: "replace", path: "/description", value: "wrong ext" }],
            "manifest-x.json": [{ op: "replace", path: "/description", value: "not a patch" }],
        });

        expect(manifest.description).toBe("base");
    });

    it("rejects a patch file that does not hold a JSON Patch array", () => {
        const source = new MemorySource(
            modelDocuments({ "manifest-x.patch.json": { op: "replace" } })
        );

        expect(() => new Model(source, { operators: {} })).toThrowError(InvalidArtifactFileError);
        expect(() => new Model(source, { operators: {} })).toThrow(/JSON Patch array/);
    });

    it("exposes an input that only a patch declares", async () => {
        const documents = modelDocuments({
            "manifest-x.patch.json": [{ op: "add", path: "/inputs/-", value: modelIO("added") }],
        });
        const patched = new Model(new MemorySource(documents), { operators: {} });
        const unpatched = new Model(new MemorySource(modelDocuments()), { operators: {} });
        await patched.warmup();
        await unpatched.warmup();
        const value = new NDArray([], [1], ArrayDType.Float32);

        await expect(patched.compute({ added: value })).resolves.toEqual({});
        await expect(unpatched.compute({ added: value })).rejects.toThrow(/Unknown input: added/);
    });

    it("drops an input that a patch replaces away", async () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": { ...MANIFEST, inputs: [modelIO("x")] },
                    "manifest-x.patch.json": [
                        { op: "replace", path: "/inputs/0", value: modelIO("z") },
                    ],
                })
            ),
            { operators: {} }
        );
        await model.warmup();
        const value = new NDArray([], [1], ArrayDType.Float32);

        expect(model.getManifest().inputs.map((entry) => entry.name)).toEqual(["z"]);
        await expect(model.compute({ z: value })).resolves.toEqual({});
        await expect(model.compute({ x: value })).rejects.toThrow(/Unknown input: x/);
    });

    it.each(["inputs", "outputs"] as const)(
        "rejects a manifest that declares the same %s name twice",
        (kind) => {
            const source = new MemorySource(
                modelDocuments({
                    "manifest.json": { ...MANIFEST, [kind]: [modelIO("dup"), modelIO("dup")] },
                })
            );

            expect(() => new Model(source, { operators: {} })).toThrowError(
                InvalidArtifactFileError
            );
            expect(() => new Model(source, { operators: {} })).toThrow(/declared more than once/);
        }
    );
});

describe("metadata assembly", () => {
    function metadataIds(overrides: Record<string, unknown>): string[] {
        return new Model(new MemorySource(modelDocuments(overrides)), { operators: {} })
            .getMetadata()
            .map((entry) => entry.id);
    }

    it("orders sidecars by byte value, so an upper-case suffix comes first", () => {
        expect(
            metadataIds({
                "meta-a.json": [metadataEntry("lower")],
                "meta-B.json": [metadataEntry("upper")],
            })
        ).toEqual(["base", "upper", "lower"]);
    });

    it("keeps meta.json first even when a sidecar sorts before it", () => {
        expect(
            metadataIds({
                "meta-A.json": [metadataEntry("sidecar")],
            })
        ).toEqual(["base", "sidecar"]);
    });

    it("reads metadata when meta.json is absent", () => {
        const documents = modelDocuments({ "meta-only.json": [metadataEntry("only")] });
        delete documents["meta.json"];

        expect(
            new Model(new MemorySource(documents), { operators: {} })
                .getMetadata()
                .map((entry) => entry.id)
        ).toEqual(["only"]);
    });

    it("skips a sidecar that is not a JSON array without dropping the others", () => {
        const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        try {
            expect(
                metadataIds({
                    "meta-a.json": { id: "object" },
                    "meta-b.json": "text",
                    "meta-c.json": [metadataEntry("c")],
                })
            ).toEqual(["base", "c"]);
            expect(warning).toHaveBeenCalledWith(expect.stringContaining("meta-a.json"));
            expect(warning).toHaveBeenCalledWith(expect.stringContaining("meta-b.json"));
        } finally {
            warning.mockRestore();
        }
    });

    it.each([
        ["a missing producer", { producer: undefined }],
        ["a non-string producer_version", { producer_version: 1 }],
        ["non-string producer_tags", { producer_tags: [1] }],
        ["producer_tags that are not an array", { producer_tags: "tag" }],
        ["a payload that is not an object", { payload: [] }],
    ])("skips an entry with %s and keeps its neighbours", (_reason, override) => {
        const malformed = { ...metadataEntry("malformed"), ...override };
        if ("producer" in override) {
            delete (malformed as Record<string, unknown>).producer;
        }

        expect(metadataIds({ "meta-a.json": [malformed, metadataEntry("kept")] })).toEqual([
            "base",
            "kept",
        ]);
    });

    it("preserves top-level keys it does not understand", () => {
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "meta-a.json": [
                        {
                            ...metadataEntry("extended"),
                            seen_by_a_later_revision: { kept: true },
                        },
                    ],
                })
            ),
            { operators: {} }
        );

        expect(model.getMetadata().at(-1)).toMatchObject({
            id: "extended",
            seen_by_a_later_revision: { kept: true },
        });
    });

    it.each([
        "metadata.json",
        "meta-.json",
        "meta_stuff.json",
        "meta.txt",
        "meta.json.bak",
        "sub/meta.json",
        "meta_artifacts/meta.json",
    ])("does not read %s as metadata", (name) => {
        expect(metadataIds({ [name]: [metadataEntry("ignored")] })).toEqual(["base"]);
    });
});

describe("declaration validation", () => {
    it.each(["inputs", "outputs"] as const)(
        "rejects an unknown content type declared on manifest %s",
        (entryKind) => {
            const name = entryKind === "inputs" ? "x" : "y";
            const source = new MemorySource(
                modelDocuments({
                    "manifest.json": {
                        ...MANIFEST,
                        [entryKind]: [
                            { name, content_type: "PARQUET", dtype: "Array[float32]", shape: [] },
                        ],
                    },
                })
            );

            expect(() => new Model(source, { operators: {} })).toThrowError(
                InvalidArtifactFileError
            );
            expect(() => new Model(source, { operators: {} })).toThrow(
                new RegExp(`${name}.*PARQUET`)
            );
        }
    );

    it.each([
        { manifest: { ...MANIFEST, variant: 1 }, reason: "variant" },
        { manifest: { ...MANIFEST, inputs: {} }, reason: "inputs" },
        { manifest: { ...MANIFEST, outputs: null }, reason: "outputs" },
        { manifest: { ...MANIFEST, producer_name: null }, reason: "producer_name" },
        { manifest: { ...MANIFEST, producer_tags: "tag" }, reason: "producer_tags" },
        {
            manifest: { ...MANIFEST, inputs: [{ name: "x", content_type: "NDJSON" }] },
            reason: "dtype",
        },
        {
            manifest: {
                ...MANIFEST,
                inputs: [{ name: "x", content_type: "NDJSON", dtype: "Array[float32]" }],
            },
            reason: "shape",
        },
    ])("reports a malformed manifest field $reason", (testCase) => {
        const source = new MemorySource(modelDocuments({ "manifest.json": testCase.manifest }));

        expect(() => new Model(source, { operators: {} })).toThrowError(InvalidArtifactFileError);
        expect(() => new Model(source, { operators: {} })).toThrow(new RegExp(testCase.reason));
    });

    it("reports a malformed variant_config", () => {
        const source = new MemorySource(modelDocuments({ "variant_config.json": {} }));

        expect(() => new Model(source, { operators: {} })).toThrowError(InvalidArtifactFileError);
        expect(() => new Model(source, { operators: {} })).toThrow(/nodes/);
    });

    it.each(["with space", "with-dash", "with/slash", "", "wi.th"])(
        "rejects the op instance id %j",
        (id) => {
            const source = new MemorySource(
                modelDocuments({ "ops.json": [{ ...opInstance(), id }] })
            );

            expect(() => new Model(source, { operators: {} })).toThrowError(
                InvalidArtifactFileError
            );
        }
    );

    it("rejects a duplicate op instance id", () => {
        const source = new MemorySource(
            modelDocuments({ "ops.json": [opInstance(), opInstance()] })
        );

        expect(() => new Model(source, { operators: {} })).toThrow(/declared more than once/);
    });

    it("accepts an op instance id of letters, digits and underscores", () => {
        const source = new MemorySource(
            modelDocuments({ "ops.json": [{ ...opInstance(), id: "Op_1" }] })
        );

        expect(() => new Model(source, { operators: {} })).not.toThrow();
    });

    it("reports a malformed op instance declaration", () => {
        const source = new MemorySource(
            modelDocuments({ "ops.json": [{ id: "test", op: "test", inputs: {}, outputs: [] }] })
        );

        expect(() => new Model(source, { operators: {} })).toThrow(/inputs/);
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

    async function computeThroughNode(
        modelIOSpec: object,
        opIOSpec: object,
        value: unknown
    ): Promise<unknown> {
        InputTrackingOp.calls = 0;
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": {
                        ...MANIFEST,
                        inputs: [{ name: "x", ...modelIOSpec }],
                        outputs: [{ name: "y", ...modelIOSpec }],
                    },
                    "ops.json": [
                        {
                            ...opInstance(1, 1),
                            inputs: [opIOSpec],
                            outputs: [opIOSpec],
                        },
                    ],
                    "variant_config.json": { nodes: [pipelineNode(["x"], ["y"])] },
                })
            ),
            { operators: { test: InputTrackingOp } }
        );
        await model.warmup();
        return model.compute({ x: value });
    }

    it("rejects a node input whose dtype differs from the op instance declaration", async () => {
        const declared = { content_type: "NDJSON", dtype: "Array[float64]", shape: ["batch", 2] };
        const opDeclared = { dtype: "Array[float32]", shape: ["batch", 2] };
        const value = new NDArray([1, 2], [1, 2], ArrayDType.Float64);

        await expect(computeThroughNode(declared, opDeclared, value)).rejects.toThrow(
            /Expected input dtype Array\[float32\], got Array\[float64\]/
        );
        expect(InputTrackingOp.calls).toBe(0);
    });

    it("rejects a node input that is not the declared container kind", async () => {
        const declared = { content_type: "NDJSON", dtype: "Array[float32]", shape: [] };
        const opDeclared = { dtype: "NDContainer[integer]", shape: [] };
        const value = new NDArray([], [1], ArrayDType.Float32);

        await expect(computeThroughNode(declared, opDeclared, value)).rejects.toThrow(
            /Expected input dtype NDContainer\[integer\], got NDArray/
        );
        expect(InputTrackingOp.calls).toBe(0);
    });

    it("validates a rank-0 op instance shape declaration", async () => {
        const declared = { content_type: "NDJSON", dtype: "Array[float32]", shape: ["batch", 2] };
        const opDeclared = { dtype: "Array[float32]", shape: [] };
        const value = new NDArray([1, 2], [1, 2], ArrayDType.Float32);

        await expect(computeThroughNode(declared, opDeclared, value)).rejects.toThrow(
            /Expected input shape \[\], got \[1,2\]/
        );
        expect(InputTrackingOp.calls).toBe(0);
    });

    it("accepts a rank-0 value for a rank-0 op instance declaration", async () => {
        const declared = { content_type: "NDJSON", dtype: "Array[float32]", shape: [] };
        const opDeclared = { dtype: "Array[float32]", shape: [] };
        const value = new NDArray([], [1.5], ArrayDType.Float32);

        await expect(computeThroughNode(declared, opDeclared, value)).resolves.toEqual({
            y: value,
        });
        expect(InputTrackingOp.calls).toBe(1);
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

    it("passes a caller attribute the manifest does not declare on to the operation", async () => {
        DynamicAttributeOp.received = [];
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": { ...MANIFEST, dynamic_attributes: [] },
                    "ops.json": [
                        opInstance(0, 0, {
                            internal: { name: "external", default_value: "default" },
                        }),
                    ],
                    "variant_config.json": { nodes: [pipelineNode()] },
                })
            ),
            { operators: { test: DynamicAttributeOp } }
        );
        await model.warmup();

        await model.compute({}, { external: "supplied", unrelated: "ignored" });

        expect(DynamicAttributeOp.received).toEqual([{ internal: "supplied" }]);
    });

    it("does not reject an invocation that omits a declared attribute", async () => {
        DynamicAttributeOp.received = [];
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "manifest.json": {
                        ...MANIFEST,
                        dynamic_attributes: [{ name: "external", description: "" }],
                    },
                    "ops.json": [
                        opInstance(0, 0, {
                            internal: { name: "external", default_value: "default" },
                        }),
                    ],
                    "variant_config.json": { nodes: [pipelineNode()] },
                })
            ),
            { operators: { test: DynamicAttributeOp } }
        );
        await model.warmup();

        await expect(model.compute({})).resolves.toEqual({});
        expect(DynamicAttributeOp.received).toEqual([{ internal: "default" }]);
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

    it("does not mutate the caller's attribute mapping", async () => {
        DynamicAttributeOp.received = [];
        const model = new Model(
            new MemorySource(
                modelDocuments({
                    "ops.json": [
                        opInstance(0, 0, { internal: { name: "n", default_value: "default" } }),
                    ],
                    "variant_config.json": {
                        nodes: [
                            {
                                op_instance_id: "test",
                                inputs: [],
                                outputs: [],
                                extra_dynattrs: { n: "pinned" },
                            },
                        ],
                    },
                })
            ),
            { operators: { test: DynamicAttributeOp } }
        );
        await model.warmup();
        const callerAttributes = { n: "caller" };

        await model.compute({}, callerAttributes);

        expect(callerAttributes).toEqual({ n: "caller" });
        expect(DynamicAttributeOp.received).toEqual([{ internal: "pinned" }]);
    });
});

class ScaleOp extends BaseOp {
    static invocations: string[] = [];

    async warmup(): Promise<this> {
        this.setWarmedUp(true);
        return this;
    }

    protected async run(inputs: unknown[]): Promise<OpOutput> {
        ScaleOp.invocations.push(this.opInstanceId);
        const total = inputs.reduce<number>(
            (sum, value) => sum + Number((value as NDArray).toArray()[0]),
            0
        );
        const factor = Number(this.attributes.factor ?? 1);
        return { value: [new NDArray([], [total * factor], ArrayDType.Float32)] };
    }
}

function scaleInstance(id: string, inputCount: number, factor: number): object {
    return {
        id,
        op: "scale",
        inputs: Array.from({ length: inputCount }, () => ({ ...OP_IO })),
        outputs: [{ ...OP_IO }],
        attributes: { factor },
        dynamic_attributes: {},
    };
}

function scaleNode(id: string, inputs: string[], outputs: string[]): object {
    return { op_instance_id: id, inputs, outputs, extra_dynattrs: {} };
}

function graphModel(
    ops: object[],
    nodes: object[],
    inputs: string[],
    outputs: string[]
): Model {
    return new Model(
        new MemorySource(
            modelDocuments({
                "manifest.json": {
                    ...MANIFEST,
                    inputs: inputs.map((name) => modelIO(name)),
                    outputs: outputs.map((name) => modelIO(name)),
                },
                "ops.json": ops,
                "variant_config.json": { nodes },
            })
        ),
        { operators: { scale: ScaleOp } }
    );
}

function scalar(value: number): NDArray {
    return new NDArray([], [value], ArrayDType.Float32);
}

describe("pipeline graph semantics", () => {
    it("computes a diamond graph through its join node", async () => {
        ScaleOp.invocations = [];
        const model = graphModel(
            [scaleInstance("left", 1, 2), scaleInstance("right", 1, 3), scaleInstance("join", 2, 1)],
            [
                scaleNode("left", ["x"], ["a"]),
                scaleNode("right", ["x"], ["b"]),
                scaleNode("join", ["a", "b"], ["y"]),
            ],
            ["x"],
            ["y"]
        );
        await model.warmup();

        const outputs = await model.compute({ x: scalar(5) });

        expect((outputs.y as NDArray).toArray()).toEqual([25]);
        expect(ScaleOp.invocations).toEqual(["left", "right", "join"]);
    });

    it("invokes one op instance once per node that references it", async () => {
        ScaleOp.invocations = [];
        const model = graphModel(
            [scaleInstance("scale", 1, 2)],
            [scaleNode("scale", ["x"], ["a"]), scaleNode("scale", ["a"], ["y"])],
            ["x"],
            ["y"]
        );
        await model.warmup();

        const outputs = await model.compute({ x: scalar(3) });

        expect((outputs.y as NDArray).toArray()).toEqual([12]);
        expect(ScaleOp.invocations).toEqual(["scale", "scale"]);
    });

    it("keeps a bound value that no declared output names internal", async () => {
        ScaleOp.invocations = [];
        const model = graphModel(
            [scaleInstance("first", 1, 2), scaleInstance("second", 1, 5)],
            [scaleNode("first", ["x"], ["internal"]), scaleNode("second", ["internal"], ["y"])],
            ["x"],
            ["y"]
        );
        await model.warmup();

        const outputs = await model.compute({ x: scalar(1) });

        expect(Object.keys(outputs)).toEqual(["y"]);
        expect((outputs.y as NDArray).toArray()).toEqual([10]);
    });

    it("returns declared outputs in manifest order regardless of binding order", async () => {
        ScaleOp.invocations = [];
        const model = graphModel(
            [scaleInstance("first", 1, 2), scaleInstance("second", 1, 3)],
            [scaleNode("first", ["x"], ["late"]), scaleNode("second", ["x"], ["early"])],
            ["x"],
            ["early", "late"]
        );
        await model.warmup();

        const outputs = await model.compute({ x: scalar(1) });

        expect(Object.keys(outputs)).toEqual(["early", "late"]);
    });

    it("binds a model input straight to a declared output when no node reads it", async () => {
        const model = graphModel([], [], ["x"], ["x"]);
        await model.warmup();
        const value = scalar(7);

        await expect(model.compute({ x: value })).resolves.toEqual({ x: value });
    });
});
