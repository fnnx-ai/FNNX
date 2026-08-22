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

    it("accepts a whole JSON number as a float element", () => {
        const manager = new DtypesManager();

        // JSON.parse turns `2.0` into the same value as `2`, so `float` cannot require a
        // fractional part without rejecting values other implementations accept.
        expect(() => manager.validateDtype("float", JSON.parse("2.0"))).not.toThrow();
        expect(() => new NDContainer([1, 2], "NDContainer[float]", manager)).not.toThrow();
        expect(() => manager.validateDtype("integer", 1.5)).toThrow(TypeError);
    });

    it("validates NDContainer elements at the declared rank", () => {
        const manager = new DtypesManager({
            "ext::point": { type: "array", items: { type: "integer" }, minItems: 2, maxItems: 2 },
        });

        const container = new NDContainer(
            [
                [1, 2],
                [3, 4],
            ],
            "NDContainer[ext::point]",
            manager,
            1
        );

        expect(container.shape).toEqual([2]);
        expect(container.toArray()).toEqual([
            [1, 2],
            [3, 4],
        ]);
        expect(() => new NDContainer([[1, 2, 3]], "NDContainer[ext::point]", manager, 1)).toThrow(
            /maxItems|items/i
        );
        expect(() => new NDContainer([1], "NDContainer[ext::point]", manager, 2)).toThrow(
            /nesting/
        );
    });

    it("rejects a custom dtype name that opens the parameterized form", () => {
        expect(() => new DtypesManager({ "Invalid[Name]": {} })).toThrow(/Invalid dtype name/);
        expect(() => new DtypesManager({ "ext::valid": {} })).not.toThrow();
    });

    it("reports an unknown dtype name from getDtype and validateDtype", () => {
        const manager = new DtypesManager({ "ext::known": { type: "object" } });

        expect(manager.getDtype("ext::known")).toEqual({ type: "object" });
        expect(() => manager.getDtype("ext::unknown")).toThrow(/Unknown dtype/);
        expect(() => manager.validateDtype("ext::unknown", {})).toThrow(/Unknown dtype/);
    });

    it.each([
        ["boolean", 1],
        ["boolean", "true"],
        ["boolean", null],
        ["string", 1],
        ["string", true],
        ["integer", "1"],
        ["float", "1.5"],
        ["float", null],
    ])("rejects %s data typed as %j", (name, value) => {
        expect(() => new DtypesManager().validateDtype(name, value)).toThrow(TypeError);
    });

    it("recurses through nesting when no rank bounds it", () => {
        const manager = new DtypesManager();

        expect(() => manager.validateDtype("integer", [[1, 2], [3]])).not.toThrow();
        expect(() => manager.validateDtype("integer", [[1], ["two"]])).toThrow(TypeError);
        // An empty container has no elements to disagree with the declared dtype.
        expect(() => manager.validateDtype("integer", [])).not.toThrow();
    });

    it("requires the declared rank of nesting when one is given", () => {
        const manager = new DtypesManager();

        expect(() => manager.validateDtype("integer", [[1]], 2)).not.toThrow();
        expect(() => manager.validateDtype("integer", [1], 2)).toThrow(/nesting/);
        // At rank 0 the value is an element, so a list is validated as one.
        expect(() => manager.validateDtype("integer", [1], 0)).toThrow(TypeError);
    });

    it("rejects an Array form as the inner dtype of an NDContainer", () => {
        const manager = new DtypesManager();

        expect(() => new NDContainer([[1.5]], "NDContainer[Array[float32]]", manager)).toThrow(
            /Array\[float32\]/
        );
        // The rule holds without a manager: the dtype language forbids the form itself.
        expect(() => new NDContainer([[1.5]], "NDContainer[Array[float32]]")).toThrow(
            /Array\[float32\]/
        );
        expect(() => new NDContainer([1.5], "Array[float32]")).toThrow(/Array dtype/);
    });

    it("unwraps exactly one NDContainer level from the declared dtype", () => {
        const manager = new DtypesManager({ "ext::num": { type: "integer" } });

        expect(new NDContainer([1, 2], "NDContainer[ext::num]", manager).dtype).toBe("ext::num");
        expect(new NDContainer([1, 2], "ext::num", manager).dtype).toBe("ext::num");
    });

    it("wraps a scalar into one level of nesting", () => {
        const manager = new DtypesManager();
        const scalar = new NDContainer(7, "NDContainer[integer]", manager, 0);

        expect(scalar.toArray()).toEqual([7]);
        expect(scalar.shape).toEqual([1]);
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
