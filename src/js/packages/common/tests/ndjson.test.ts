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

describe("NDJSON integer element limits", () => {
    const INT64_MIN = -(2n ** 63n);
    const INT64_MAX = 2n ** 63n - 1n;

    it("decodes int64 elements at both signed 64-bit boundaries", () => {
        const decoded = decodeNDJSON(`[${INT64_MIN},${INT64_MAX}]`, ArrayDType.Int64, [2]);

        expect(decoded.toArray()).toEqual([INT64_MIN, INT64_MAX]);
        expect(encodeNDJSON(decoded)).toBe(`[${INT64_MIN},${INT64_MAX}]`);
    });

    it.each([`${INT64_MIN - 1n}`, `${INT64_MAX + 1n}`])(
        "rejects the int64 element %s that sits just outside the range",
        (token) => {
            expect(() => decodeNDJSON(`[${token}]`, ArrayDType.Int64, [1])).toThrow(RangeError);
        }
    );

    it("decodes int32 elements at both signed 32-bit boundaries", () => {
        const decoded = decodeNDJSON("[-2147483648,2147483647]", ArrayDType.Int32, [2]);

        expect(decoded.toArray()).toEqual([-2147483648, 2147483647]);
        expect(encodeNDJSON(decoded)).toBe("[-2147483648,2147483647]");
    });

    it.each(["-2147483649", "2147483648"])(
        "rejects the int32 element %s that sits just outside the range",
        (token) => {
            expect(() => decodeNDJSON(`[${token}]`, ArrayDType.Int32, [1])).toThrow(RangeError);
        }
    );

    it("preserves an int64 magnitude that a double cannot represent exactly", () => {
        // 2^63 - 1 rounds to 2^63 as a double, so the codec must never route it through Number.
        const decoded = decodeNDJSON(`[${INT64_MAX}]`, ArrayDType.Int64, [1]);

        expect(decoded.toArray()[0]).toBe(INT64_MAX);
        expect(BigInt(JSON.parse(`${INT64_MAX}`) as number)).not.toBe(INT64_MAX);
    });

    it.each([
        ["scientific notation", "[1e3]"],
        ["a trailing fraction", "[1.0]"],
        ["a JSON string", '["1"]'],
        ["a JSON null", "[null]"],
        ["a JSON boolean", "[true]"],
    ])("rejects %s in an integer array", (_form, serialized) => {
        expect(() => decodeNDJSON(serialized, ArrayDType.Int64, [1])).toThrow(/integer/i);
        expect(() => decodeNDJSON(serialized, ArrayDType.Int32, [1])).toThrow(/integer/i);
    });
});

describe("NDJSON float element forms", () => {
    it("decodes scientific notation", () => {
        const decoded = decodeNDJSON("[1.5e2,1E-3,-2e+2]", ArrayDType.Float64, [3]);

        expect(decoded.toArray()).toEqual([150, 0.001, -200]);
    });

    it("rejects a numeric literal that overflows to a non-finite double", () => {
        // The spec carries non-finite floats as the three string markers, never as a literal.
        expect(() => decodeNDJSON("[1e400]", ArrayDType.Float64, [1])).toThrow(/non-finite/i);
    });

    it.each(["nan", "inf", "infinity", "+Infinity", "NaN ", "1.5", ""])(
        "rejects the float array string %j",
        (text) => {
            expect(() =>
                decodeNDJSON(JSON.stringify([text]), ArrayDType.Float32, [1])
            ).toThrow(/marker/i);
        }
    );

    it.each([
        ["null", "[null]"],
        ["a boolean", "[true]"],
    ])("rejects %s in a float array", (_form, serialized) => {
        expect(() => decodeNDJSON(serialized, ArrayDType.Float64, [1])).toThrow(TypeError);
    });
});

describe("NDJSON element type strictness", () => {
    it("keeps the non-finite markers as plain text in a string array", () => {
        const serialized = '["NaN","Infinity","-Infinity"]';
        const decoded = decodeNDJSON(serialized, ArrayDType.String, [3]);

        expect(decoded.toArray()).toEqual(["NaN", "Infinity", "-Infinity"]);
        expect(encodeNDJSON(decoded)).toBe(serialized);
    });

    it.each(["[1]", "[0]", '["true"]', '["TRUE"]', "[null]"])(
        "rejects the non-boolean token %s in a bool array",
        (serialized) => {
            expect(() => decodeNDJSON(serialized, ArrayDType.Bool, [1])).toThrow(/boolean/i);
        }
    );

    it.each(["[1]", "[true]", "[null]"])(
        "rejects the non-string token %s in a string array",
        (serialized) => {
            expect(() => decodeNDJSON(serialized, ArrayDType.String, [1])).toThrow(/string/i);
        }
    );
});

describe("NDJSON nesting", () => {
    it("round-trips deeply nested rank-64 data", () => {
        const rank = 64;
        const serialized = "[".repeat(rank) + "7" + "]".repeat(rank);

        const decoded = decodeNDJSON(serialized, ArrayDType.Int32, Array(rank).fill(1));

        expect(decoded.shape).toEqual(Array(rank).fill(1));
        expect(decoded.toArray()).toEqual([7]);
        expect(encodeNDJSON(decoded)).toBe(serialized);
    });

    it("decodes empty nesting and fills the trailing declared extents", () => {
        expect(decodeNDJSON("[]", ArrayDType.Int32).shape).toEqual([0]);
        expect(decodeNDJSON("[[],[]]", ArrayDType.Int32, [2, 0]).shape).toEqual([2, 0]);
        expect(decodeNDJSON("[]", ArrayDType.Int32, [0, 3, 4]).shape).toEqual([0, 3, 4]);
        expect(encodeNDJSON(decodeNDJSON("[[],[]]", ArrayDType.Int32, [2, 0]))).toBe("[[],[]]");
    });

    it("rejects ragged nesting below the outermost axis", () => {
        expect(() => decodeNDJSON("[[[1],[2]],[[3]]]", ArrayDType.Int32)).toThrow(/rectangular/i);
        expect(() => decodeNDJSON("[[1,2],[]]", ArrayDType.Int32)).toThrow(/rectangular/i);
    });

    it("tolerates insignificant JSON whitespace", () => {
        const decoded = decodeNDJSON(" [ [ 1 , 2 ] ,\n[ 3 , 4 ] ]\t", ArrayDType.Int32, [2, 2]);

        expect(decoded.toArray()).toEqual([1, 2, 3, 4]);
    });

    it("rejects content after the outermost value", () => {
        expect(() => decodeNDJSON("[1] [2]", ArrayDType.Int32)).toThrow(/Unexpected content/);
    });
});
