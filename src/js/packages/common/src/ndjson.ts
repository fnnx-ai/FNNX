import { ArrayDType, ArrayElement, NDArray } from "./ndarray.js";

export type DeclaredShape = readonly (number | string)[];

interface NumberToken {
    readonly raw: string;
}

type ParsedValue = NumberToken | string | boolean | null | ParsedValue[];

export function decodeNDJSON(
    serialized: string,
    dtype: ArrayDType,
    declaredShape?: DeclaredShape
): NDArray {
    const parsed = new NestedJsonParser(serialized).parse();
    const flattened = flatten(parsed);
    const shape =
        declaredShape === undefined
            ? flattened.shape
            : resolveDeclaredShape(flattened.shape, declaredShape);
    return new NDArray(
        shape,
        flattened.elements.map((value) => decodeElement(value, dtype)),
        dtype
    );
}

export function encodeNDJSON(value: NDArray): string {
    const data = value.toArray();
    return encodeDimension(value.shape, data, value.dtype, 0, 0);
}

class NestedJsonParser {
    private position = 0;

    constructor(private readonly source: string) {}

    parse(): ParsedValue {
        const value = this.parseValue();
        this.skipWhitespace();
        if (this.position !== this.source.length) {
            throw new Error(`Unexpected content at position ${this.position}`);
        }
        return value;
    }

    private parseValue(): ParsedValue {
        this.skipWhitespace();
        const character = this.source[this.position];
        if (character === "[") {
            return this.parseArray();
        }
        if (character === '"') {
            return this.parseString();
        }
        if (character === "t") {
            this.consumeLiteral("true");
            return true;
        }
        if (character === "f") {
            this.consumeLiteral("false");
            return false;
        }
        if (character === "n") {
            this.consumeLiteral("null");
            return null;
        }
        return this.parseNumber();
    }

    private parseArray(): ParsedValue[] {
        this.position += 1;
        this.skipWhitespace();
        if (this.source[this.position] === "]") {
            this.position += 1;
            return [];
        }

        const values: ParsedValue[] = [];
        while (true) {
            values.push(this.parseValue());
            this.skipWhitespace();
            const separator = this.source[this.position];
            this.position += 1;
            if (separator === "]") {
                return values;
            }
            if (separator !== ",") {
                throw new Error(`Expected ',' or ']' at position ${this.position - 1}`);
            }
        }
    }

    private parseString(): string {
        const start = this.position;
        this.position += 1;
        while (this.position < this.source.length) {
            const character = this.source[this.position];
            if (character === '"') {
                this.position += 1;
                return JSON.parse(this.source.slice(start, this.position)) as string;
            }
            if (character === "\\") {
                this.position += 1;
            }
            this.position += 1;
        }
        throw new Error(`Unterminated string at position ${start}`);
    }

    private parseNumber(): NumberToken {
        const token = this.source
            .slice(this.position)
            .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/)?.[0];
        if (!token) {
            throw new Error(`Expected an NDJSON value at position ${this.position}`);
        }
        this.position += token.length;
        return { raw: token };
    }

    private consumeLiteral(literal: string): void {
        if (this.source.slice(this.position, this.position + literal.length) !== literal) {
            throw new Error(`Invalid JSON literal at position ${this.position}`);
        }
        this.position += literal.length;
    }

    private skipWhitespace(): void {
        while (isJsonWhitespace(this.source[this.position])) {
            this.position += 1;
        }
    }
}

function flatten(value: ParsedValue): {
    shape: number[];
    elements: Exclude<ParsedValue, ParsedValue[]>[];
} {
    if (!Array.isArray(value)) {
        return { shape: [], elements: [value] };
    }
    if (value.length === 0) {
        return { shape: [0], elements: [] };
    }

    const children = value.map(flatten);
    const childShape = children[0].shape;
    if (children.some((child) => !sameShape(child.shape, childShape))) {
        throw new Error("NDJSON nesting must be rectangular");
    }
    return {
        shape: [value.length, ...childShape],
        elements: children.flatMap((child) => child.elements),
    };
}

function resolveDeclaredShape(actual: number[], declared: DeclaredShape): number[] {
    for (let axis = 0; axis < declared.length; axis += 1) {
        const extent = declared[axis];
        if (
            typeof extent !== "string" &&
            (typeof extent !== "number" || !Number.isSafeInteger(extent) || extent < 0)
        ) {
            throw new Error(`Invalid declared extent at axis ${axis}: ${String(extent)}`);
        }
    }

    const nestingEndedAtZero = actual.length > 0 && actual.at(-1) === 0;
    if (
        actual.length > declared.length ||
        (actual.length < declared.length && !nestingEndedAtZero)
    ) {
        throw new Error(
            `NDJSON shape rank ${actual.length} does not match declared rank ${declared.length}`
        );
    }
    for (let axis = 0; axis < declared.length; axis += 1) {
        const extent = declared[axis];
        if (axis < actual.length && typeof extent === "number" && actual[axis] !== extent) {
            throw new Error(
                `NDJSON extent ${actual[axis]} at axis ${axis} does not match declared extent ${extent}`
            );
        }
    }

    return declared.map((extent, axis) => {
        if (axis < actual.length) {
            return actual[axis];
        }
        return typeof extent === "number" ? extent : 0;
    });
}

function decodeElement(
    value: Exclude<ParsedValue, ParsedValue[]>,
    dtype: ArrayDType
): ArrayElement {
    switch (dtype) {
        case ArrayDType.Float32:
        case ArrayDType.Float64:
            return decodeFloat(value);
        case ArrayDType.Int32:
            return decodeInt32(value);
        case ArrayDType.Int64:
            return decodeInt64(value);
        case ArrayDType.String:
            if (typeof value === "string") {
                return value;
            }
            throw new TypeError("Array[string] elements must be JSON strings");
        case ArrayDType.Bool:
            if (typeof value === "boolean") {
                return value;
            }
            throw new TypeError("Array[bool] elements must be JSON booleans");
    }
}

function decodeFloat(value: Exclude<ParsedValue, ParsedValue[]>): number {
    if (isNumberToken(value)) {
        const decoded = Number(value.raw);
        if (!Number.isFinite(decoded)) {
            throw new TypeError("Non-finite float values must use an NDJSON string marker");
        }
        return decoded;
    }
    if (typeof value === "string") {
        if (value === "NaN") {
            return Number.NaN;
        }
        if (value === "Infinity") {
            return Number.POSITIVE_INFINITY;
        }
        if (value === "-Infinity") {
            return Number.NEGATIVE_INFINITY;
        }
        throw new TypeError(
            `Invalid float array string ${JSON.stringify(value)}; expected a non-finite marker`
        );
    }
    throw new TypeError("Float array elements must be JSON numbers or non-finite string markers");
}

function decodeInt32(value: Exclude<ParsedValue, ParsedValue[]>): number {
    const integer = decodeIntegerToken(value, "int32");
    if (integer < -2_147_483_648n || integer > 2_147_483_647n) {
        throw new RangeError(`int32 value is out of range: ${integer}`);
    }
    return Number(integer);
}

function decodeInt64(value: Exclude<ParsedValue, ParsedValue[]>): bigint {
    const integer = decodeIntegerToken(value, "int64");
    if (integer < -9_223_372_036_854_775_808n || integer > 9_223_372_036_854_775_807n) {
        throw new RangeError(`int64 value is out of range: ${integer}`);
    }
    return integer;
}

function decodeIntegerToken(value: Exclude<ParsedValue, ParsedValue[]>, dtype: string): bigint {
    if (!isNumberToken(value) || !/^-?(?:0|[1-9]\d*)$/.test(value.raw)) {
        throw new TypeError(`Array[${dtype}] elements must be JSON integers`);
    }
    return BigInt(value.raw);
}

function encodeDimension(
    shape: number[],
    data: ArrayElement[],
    dtype: ArrayDType,
    axis: number,
    offset: number
): string {
    if (axis === shape.length) {
        return encodeElement(data[offset], dtype);
    }

    const stride = shape.slice(axis + 1).reduce((product, extent) => product * extent, 1);
    const values = Array.from({ length: shape[axis] }, (_, index) =>
        encodeDimension(shape, data, dtype, axis + 1, offset + index * stride)
    );
    return `[${values.join(",")}]`;
}

function encodeElement(value: ArrayElement, dtype: ArrayDType): string {
    switch (dtype) {
        case ArrayDType.Float32:
        case ArrayDType.Float64:
            if (typeof value !== "number") {
                throw new TypeError(`Expected a numeric ${dtype} value`);
            }
            if (Number.isNaN(value)) {
                return '"NaN"';
            }
            if (value === Number.POSITIVE_INFINITY) {
                return '"Infinity"';
            }
            if (value === Number.NEGATIVE_INFINITY) {
                return '"-Infinity"';
            }
            return JSON.stringify(value);
        case ArrayDType.Int32:
            if (typeof value !== "number" || !Number.isInteger(value)) {
                throw new TypeError("Expected an int32 value");
            }
            return String(value);
        case ArrayDType.Int64:
            if (typeof value !== "bigint") {
                throw new TypeError("Expected an int64 bigint value");
            }
            return value.toString();
        case ArrayDType.String:
            if (typeof value !== "string") {
                throw new TypeError("Expected a string value");
            }
            return JSON.stringify(value);
        case ArrayDType.Bool:
            if (typeof value !== "boolean") {
                throw new TypeError("Expected a boolean value");
            }
            return String(value);
    }
}

function isNumberToken(value: Exclude<ParsedValue, ParsedValue[]>): value is NumberToken {
    return typeof value === "object" && value !== null;
}

function sameShape(left: number[], right: number[]): boolean {
    return left.length === right.length && left.every((extent, axis) => extent === right[axis]);
}

function isJsonWhitespace(character: string | undefined): boolean {
    return character === " " || character === "\t" || character === "\n" || character === "\r";
}
