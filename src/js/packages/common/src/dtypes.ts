const RESERVED_TYPES = ["string", "integer", "float", "Array", "NDContainer"];

export class DtypesManager {
    private readonly dtypes: Record<string, object>;

    constructor(externalDtypes: Record<string, object> = {}) {
        this.dtypes = { ...externalDtypes };

        for (const dtype of Object.keys(this.dtypes)) {
            if (dtype.includes("[") || RESERVED_TYPES.includes(dtype)) {
                throw new Error(`Invalid dtype name: ${dtype}`);
            }
        }
    }

    getDtype(name: string): object {
        const dtype = this.dtypes[name];
        if (!dtype) {
            throw new Error(`Unknown dtype: ${name}`);
        }
        return dtype;
    }

    validateDtype(name: string, data: unknown): void {
        if (Array.isArray(data)) {
            for (const value of data) {
                this.validateDtype(name, value);
            }
        } else if (typeof data === "object" && data !== null) {
            this.validateJsonSchema(name, data as Record<string, unknown>);
        } else if (typeof data === "string") {
            if (name !== "string") {
                throw new TypeError(`Invalid data type, expected \`string\`, got \`${name}\``);
            }
        } else if (typeof data === "number" && Number.isInteger(data)) {
            if (name !== "integer") {
                throw new TypeError(`Invalid data type, expected \`integer\`, got \`${name}\``);
            }
        } else if (typeof data === "number") {
            if (name !== "float") {
                throw new TypeError(`Invalid data type, expected \`float\`, got \`${name}\``);
            }
        } else {
            throw new TypeError(`Invalid data type: ${typeof data}`);
        }
    }

    validateJsonSchema(name: string, data: Record<string, unknown>): void {
        const schema = this.getDtype(name) as Record<string, unknown>;
        const properties = schema.properties;
        const required = schema.required;
        if (
            schema.type === "object" &&
            typeof properties === "object" &&
            properties !== null &&
            Array.isArray(required)
        ) {
            for (const property of required) {
                if (typeof property === "string" && !(property in data)) {
                    throw new Error(`Missing required property: ${property}`);
                }
            }
        }
    }
}
