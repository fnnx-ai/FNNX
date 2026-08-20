import { describe, it, expect } from "vitest";
import { applyPatches } from "../src/index";

describe("applyPatches", () => {
    describe("add operation", () => {
        it("should add a new top-level key", () => {
            const doc = { a: 1 };
            const result = applyPatches(doc, [[{ op: "add", path: "/b", value: 2 }]]);
            expect(result).toEqual({ a: 1, b: 2 });
        });

        it("should add a nested key", () => {
            const doc = { a: { b: 1 } };
            const result = applyPatches(doc, [[{ op: "add", path: "/a/c", value: 2 }]]);
            expect(result).toEqual({ a: { b: 1, c: 2 } });
        });

        it("should overwrite an existing key via add", () => {
            const doc = { a: 1 };
            const result = applyPatches(doc, [[{ op: "add", path: "/a", value: 99 }]]);
            expect(result).toEqual({ a: 99 });
        });

        it("should append to array with '-'", () => {
            const doc = { items: [1, 2] };
            const result = applyPatches(doc, [[{ op: "add", path: "/items/-", value: 3 }]]);
            expect(result).toEqual({ items: [1, 2, 3] });
        });

        it("should insert into array at index", () => {
            const doc = { items: ["a", "c"] };
            const result = applyPatches(doc, [[{ op: "add", path: "/items/1", value: "b" }]]);
            expect(result).toEqual({ items: ["a", "b", "c"] });
        });

        it("should insert at beginning of array", () => {
            const doc = { items: ["b", "c"] };
            const result = applyPatches(doc, [[{ op: "add", path: "/items/0", value: "a" }]]);
            expect(result).toEqual({ items: ["a", "b", "c"] });
        });

        it("should add complex nested value", () => {
            const doc = { config: {} };
            const result = applyPatches(doc, [
                [{ op: "add", path: "/config/nested", value: { x: 1, y: [2, 3] } }],
            ]);
            expect(result).toEqual({ config: { nested: { x: 1, y: [2, 3] } } });
        });
    });

    describe("replace operation", () => {
        it("should replace a top-level value", () => {
            const doc = { version: "1.0.0" };
            const result = applyPatches(doc, [
                [{ op: "replace", path: "/version", value: "2.0.0" }],
            ]);
            expect(result).toEqual({ version: "2.0.0" });
        });

        it("should replace a nested value", () => {
            const doc = { a: { b: { c: 1 } } };
            const result = applyPatches(doc, [[{ op: "replace", path: "/a/b/c", value: 42 }]]);
            expect(result).toEqual({ a: { b: { c: 42 } } });
        });

        it("should replace an array element", () => {
            const doc = { items: ["a", "b", "c"] };
            const result = applyPatches(doc, [[{ op: "replace", path: "/items/1", value: "B" }]]);
            expect(result).toEqual({ items: ["a", "B", "c"] });
        });

        it("should throw when replacing non-existent key", () => {
            const doc = { a: 1 };
            expect(() =>
                applyPatches(doc, [[{ op: "replace", path: "/nonexistent", value: 2 }]])
            ).toThrow("non-existent");
        });

        it("should throw when replacing out-of-bounds array index", () => {
            const doc = { items: [1, 2] };
            expect(() =>
                applyPatches(doc, [[{ op: "replace", path: "/items/5", value: 99 }]])
            ).toThrow();
        });
    });

    describe("multiple patches", () => {
        it("should apply multiple patch documents sequentially", () => {
            const doc = { version: "1.0.0", description: "original" };
            const result = applyPatches(doc, [
                [{ op: "replace", path: "/version", value: "2.0.0" }],
                [{ op: "replace", path: "/description", value: "updated" }],
            ]);
            expect(result).toEqual({ version: "2.0.0", description: "updated" });
        });

        it("should apply multiple operations within a single patch", () => {
            const doc = { a: 1, b: 2 };
            const result = applyPatches(doc, [
                [
                    { op: "replace", path: "/a", value: 10 },
                    { op: "add", path: "/c", value: 3 },
                ],
            ]);
            expect(result).toEqual({ a: 10, b: 2, c: 3 });
        });

        it("should apply later patches on top of earlier ones", () => {
            const doc = { value: 1 };
            const result = applyPatches(doc, [
                [{ op: "replace", path: "/value", value: 2 }],
                [{ op: "replace", path: "/value", value: 3 }],
            ]);
            expect(result).toEqual({ value: 3 });
        });
    });

    describe("RFC 6901 pointer escaping", () => {
        it("should handle ~1 escape for /", () => {
            const doc = { "a/b": 1 };
            const result = applyPatches(doc, [[{ op: "replace", path: "/a~1b", value: 2 }]]);
            expect(result).toEqual({ "a/b": 2 });
        });

        it("should handle ~0 escape for ~", () => {
            const doc = { "a~b": 1 };
            const result = applyPatches(doc, [[{ op: "replace", path: "/a~0b", value: 2 }]]);
            expect(result).toEqual({ "a~b": 2 });
        });

        it("should handle combined escapes", () => {
            const doc = { "a~/b": 1 };
            const result = applyPatches(doc, [[{ op: "replace", path: "/a~0~1b", value: 2 }]]);
            expect(result).toEqual({ "a~/b": 2 });
        });
    });

    describe("error cases", () => {
        it("should throw for unsupported operations", () => {
            const doc = { a: 1 };
            expect(() => applyPatches(doc, [[{ op: "remove", path: "/a" }]])).toThrow(
                "Unsupported JSON Patch op"
            );
        });

        it("should throw for empty path", () => {
            const doc = { a: 1 };
            expect(() => applyPatches(doc, [[{ op: "add", path: "", value: 2 }]])).toThrow(
                "Empty JSON Pointer path"
            );
        });

        it("should throw for relative path", () => {
            const doc = { a: 1 };
            expect(() => applyPatches(doc, [[{ op: "add", path: "a", value: 2 }]])).toThrow(
                "Only absolute JSON Pointer paths"
            );
        });

        it("should throw for missing path segment during traversal", () => {
            const doc = { a: {} };
            expect(() => applyPatches(doc, [[{ op: "add", path: "/a/b/c", value: 1 }]])).toThrow(
                "not found while traversing"
            );
        });

        it("should throw for non-integer array index", () => {
            const doc = { items: [1, 2] };
            expect(() =>
                applyPatches(doc, [[{ op: "replace", path: "/items/abc", value: 99 }]])
            ).toThrow("Array index must be an integer");
        });

        it("should throw for missing op field", () => {
            const doc = { a: 1 };
            expect(() => applyPatches(doc, [[{ path: "/a", value: 2 } as any]])).toThrow(
                "Invalid JSON Patch operation"
            );
        });

        it("should throw for missing path field", () => {
            const doc = { a: 1 };
            expect(() => applyPatches(doc, [[{ op: "add", value: 2 } as any]])).toThrow(
                "Invalid JSON Patch operation"
            );
        });
    });

    describe("deep copy behavior", () => {
        it("should not mutate the original document", () => {
            const doc = { a: 1, b: { c: 2 } };
            const original = JSON.parse(JSON.stringify(doc));
            applyPatches(doc, [
                [{ op: "replace", path: "/a", value: 99 }],
                [{ op: "add", path: "/b/d", value: 3 }],
            ]);
            expect(doc).toEqual(original);
        });
    });

    describe("manifest-like patching", () => {
        it("should patch a realistic manifest structure", () => {
            const manifest = {
                variant: "pipeline",
                description: "original model",
                producer_name: "test",
                producer_version: "1.0.0",
                producer_tags: ["tag1"],
                inputs: [{ name: "x", content_type: "NDJSON", dtype: "Array[float32]", shape: [] }],
                outputs: [
                    { name: "y", content_type: "NDJSON", dtype: "Array[float32]", shape: [] },
                ],
                dynamic_attributes: [],
                env_vars: [],
            };

            const result = applyPatches(manifest, [
                [
                    { op: "replace", path: "/description", value: "patched model" },
                    { op: "replace", path: "/producer_version", value: "2.0.0" },
                    { op: "add", path: "/producer_tags/-", value: "tag2" },
                ],
            ]);

            expect(result.description).toBe("patched model");
            expect(result.producer_version).toBe("2.0.0");
            expect(result.producer_tags).toEqual(["tag1", "tag2"]);
            expect(result.inputs).toEqual(manifest.inputs);
        });

        it("should replace an input spec in manifest", () => {
            const manifest = {
                variant: "pipeline",
                inputs: [{ name: "x", content_type: "NDJSON", dtype: "Array[float32]", shape: [] }],
                outputs: [],
            };

            const result = applyPatches(manifest as any, [
                [
                    {
                        op: "replace",
                        path: "/inputs/0/dtype",
                        value: "Array[int32]",
                    },
                ],
            ]);

            expect(result.inputs[0].dtype).toBe("Array[int32]");
        });
    });
});
