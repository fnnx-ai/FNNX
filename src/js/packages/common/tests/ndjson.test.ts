import { describe, expect, it } from "vitest";
import { ArrayDType, decodeNDJSON, encodeNDJSON, NDArray } from "../src/index.js";

describe("NDJSON codec", () => {
    it.each([
        [ArrayDType.Float32, [1.25, -2.5]],
        [ArrayDType.Float64, [Number.MAX_VALUE, Number.MIN_VALUE]],
        [ArrayDType.Int32, [1, -2]],
        [ArrayDType.Int64, [1n, -2n]],
        [ArrayDType.String, ["one", "two"]],
        [ArrayDType.Bool, [true, false]],
    ])("round-trips nested %s arrays", (dtype, data) => {
        const array = new NDArray([1, 2], data, dtype);

        const decoded = decodeNDJSON(encodeNDJSON(array), dtype, [1, "items"]);

        expect(decoded.shape).toEqual([1, 2]);
        expect(decoded.toArray()).toEqual(data);
    });

    it("encodes and restores non-finite floats as strings", () => {
        const array = new NDArray(
            [4],
            [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 1.5],
            ArrayDType.Float64
        );

        const encoded = encodeNDJSON(array);

        expect(encoded).toBe('["NaN","Infinity","-Infinity",1.5]');
        expect(encoded).not.toMatch(/\[(NaN|Infinity)|,null/);
        const values = decodeNDJSON(encoded, ArrayDType.Float64, [4]).toArray();
        expect(values[0]).toBeNaN();
        expect(values[1]).toBe(Number.POSITIVE_INFINITY);
        expect(values[2]).toBe(Number.NEGATIVE_INFINITY);
        expect(values[3]).toBe(1.5);
    });

    it("rejects strings other than the three float markers", () => {
        expect(() => decodeNDJSON('["1.5"]', ArrayDType.Float32, [1])).toThrow(
            /float.*string|marker/i
        );
    });

    it("serializes int64 values as exact JSON integers", () => {
        const expected = 2n ** 53n + 1n;
        const array = new NDArray([1], [expected], ArrayDType.Int64);

        const encoded = encodeNDJSON(array);

        expect(encoded).toBe("[9007199254740993]");
        expect(decodeNDJSON(encoded, ArrayDType.Int64, [1]).toArray()).toEqual([expected]);
    });

    it("validates nesting against concrete and symbolic dimensions", () => {
        const decoded = decodeNDJSON("[[1,2],[3,4]]", ArrayDType.Int32, ["rows", 2]);
        const empty = decodeNDJSON("[]", ArrayDType.Int32, [0, 3]);

        expect(decoded.shape).toEqual([2, 2]);
        expect(empty.shape).toEqual([0, 3]);
        expect(() => decodeNDJSON("[[1,2],[3,4]]", ArrayDType.Int32, [2, 3])).toThrow(
            /shape|extent/i
        );
        expect(() => decodeNDJSON("[[1,2],[3]]", ArrayDType.Int32)).toThrow(/rectangular|shape/i);
    });

    it("supports scalar arrays and rejects wire values of the wrong element type", () => {
        const scalar = decodeNDJSON("1.5", ArrayDType.Float64, []);

        expect(scalar.shape).toEqual([]);
        expect(scalar.toArray()).toEqual([1.5]);
        expect(encodeNDJSON(scalar)).toBe("1.5");
        expect(() => decodeNDJSON('["false"]', ArrayDType.Bool, [1])).toThrow(/boolean/i);
        expect(() => decodeNDJSON("[1.5]", ArrayDType.Int64, [1])).toThrow(/integer/i);
    });
});
