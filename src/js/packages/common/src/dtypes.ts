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

    validateDtype(name: string, data: unknown): void {
        const isCustom = Object.prototype.hasOwnProperty.call(this.dtypes, name);
        if (!isCustom && !IMPLICIT_TYPES.has(name)) {
            throw new Error(`Unknown dtype: ${name}`);
        }

        if (Array.isArray(data)) {
            for (const value of data) {
                this.validateDtype(name, value);
            }
            return;
        }

        if (isCustom) {
            this.validateJsonSchema(name, data);
            return;
        }

        const actualType = implicitTypeOf(data);
        if (actualType !== name) {
            throw new TypeError(`Invalid data type, expected \`${name}\`, got \`${actualType}\``);
        }
    }

    validateJsonSchema(name: string, data: unknown): void {
        validateJsonSchema(data, this.getDtype(name));
    }
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
