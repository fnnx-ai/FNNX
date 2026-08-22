import { validateJsonSchema } from "./jsonschema.js";

const RESERVED_TYPES = new Set([
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
]);

const IMPLICIT_TYPES = new Set(["string", "integer", "float", "boolean"]);

export type DtypeSchema = Record<string, unknown>;

export class DtypesManager {
    private readonly dtypes: Record<string, DtypeSchema>;

    constructor(externalDtypes: Record<string, DtypeSchema> = {}) {
        this.dtypes = { ...externalDtypes };

        for (const dtype of Object.keys(this.dtypes)) {
            if (dtype.includes("[") || RESERVED_TYPES.has(dtype)) {
                throw new Error(`Invalid dtype name: ${dtype}`);
            }
        }
    }

    getDtype(name: string): DtypeSchema {
        if (!Object.prototype.hasOwnProperty.call(this.dtypes, name)) {
            throw new Error(`Unknown dtype: ${name}`);
        }
        return this.dtypes[name];
    }

    /**
     * `rank` bounds the nesting the elements sit under. Without it a custom dtype whose values
     * are JSON arrays cannot be told apart from another level of container nesting.
     */
    validateDtype(name: string, data: unknown, rank: number | null = null): void {
        const isCustom = Object.prototype.hasOwnProperty.call(this.dtypes, name);
        if (!isCustom && !IMPLICIT_TYPES.has(name)) {
            throw new Error(`Unknown dtype: ${name}`);
        }

        if (rank === null ? Array.isArray(data) : rank > 0) {
            if (!Array.isArray(data)) {
                throw new TypeError(
                    `Expected ${rank} more level(s) of nesting, got \`${implicitTypeOf(data)}\``
                );
            }
            for (const value of data) {
                this.validateDtype(name, value, rank === null ? null : rank - 1);
            }
            return;
        }

        if (isCustom) {
            this.validateJsonSchema(name, data);
            return;
        }

        if (!matchesImplicitType(name, data)) {
            throw new TypeError(
                `Invalid data type, expected \`${name}\`, got \`${implicitTypeOf(data)}\``
            );
        }
    }

    validateJsonSchema(name: string, data: unknown): void {
        validateJsonSchema(data, this.getDtype(name));
    }
}

function matchesImplicitType(name: string, data: unknown): boolean {
    // JSON.parse erases the fractional part of a whole number, so `2.0` and `2` reach us as the
    // same JS value. `float` therefore accepts any number; `integer` still requires an integer.
    if (name === "float") {
        return typeof data === "number";
    }
    return implicitTypeOf(data) === name;
}

function implicitTypeOf(data: unknown): string {
    if (typeof data === "boolean") {
        return "boolean";
    }
    if (typeof data === "string") {
        return "string";
    }
    if (typeof data === "number") {
        return Number.isInteger(data) ? "integer" : "float";
    }
    if (data === null) {
        return "null";
    }
    if (Array.isArray(data)) {
        return "array";
    }
    return typeof data;
}
