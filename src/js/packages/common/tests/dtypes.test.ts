import { describe, expect, it } from "vitest";
import { ArrayDType, DtypesManager, NDArray, NDContainer } from "../src/index.js";

describe("array dtypes", () => {
    it.each([
        [ArrayDType.Float32, 1.25, 1.25],
        [ArrayDType.Float64, Number.MAX_VALUE, Number.MAX_VALUE],
        [ArrayDType.Int32, 42, 42],
        [ArrayDType.Int64, 42n, 42n],
        [ArrayDType.String, "value", "value"],
        [ArrayDType.Bool, true, true],
    ])("supports %s", (dtype, input, expected) => {
        expect(new NDArray([1], [input], dtype).toArray()).toEqual([expected]);
    });

    it("casts boolean strings by their value", () => {
        const value = new NDArray([4], [false, true, "false", "true"], ArrayDType.Bool);

        expect(value.toArray()).toEqual([false, true, false, true]);
        expect(() => new NDArray([1], ["not-a-boolean"], ArrayDType.Bool)).toThrow(/boolean/i);
    });
});

describe("dtype names", () => {
    it("validates all four implicit scalar dtypes distinctly", () => {
        const manager = new DtypesManager();

        expect(() => manager.validateDtype("string", "value")).not.toThrow();
        expect(() => manager.validateDtype("integer", 1)).not.toThrow();
        expect(() => manager.validateDtype("float", 1.5)).not.toThrow();
        expect(() => manager.validateDtype("boolean", true)).not.toThrow();
        expect(() => manager.validateDtype("integer", true)).toThrow(TypeError);
        expect(() => manager.validateDtype("float", true)).toThrow(TypeError);

        const booleanContainer = new NDContainer([true], "NDContainer[boolean]", manager);
        expect(booleanContainer.toArray()).toEqual([true]);
        expect(() => new NDContainer([true], "NDContainer[integer]", manager)).toThrow(TypeError);
    });

    it.each([
        "string",
        "integer",
        "float",
        "boolean",
        "Array",
        "NDContainer",
        "float32",
        "float64",
        "int32",
        "int64",
        "bool",
    ])("rejects the reserved custom dtype name %s", (name) => {
        expect(() => new DtypesManager({ [name]: {} })).toThrow(/Invalid dtype name/);
    });

    it("enforces custom schemas for NDContainer elements", () => {
        const manager = new DtypesManager({
            "ext::rec": {
                type: "object",
                required: ["a"],
                properties: { a: { type: "integer" } },
            },
        });

        expect(() => new NDContainer([{ a: 1 }], "NDContainer[ext::rec]", manager)).not.toThrow();
        expect(() => new NDContainer([{ a: true }], "NDContainer[ext::rec]", manager)).toThrow(
            /integer/i
        );
        expect(() => new NDContainer([], "NDContainer[ext::missing]", manager)).toThrow(
            /Unknown dtype/
        );
    });
});
