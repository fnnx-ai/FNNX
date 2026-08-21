import { describe, expect, it } from "vitest";
import { validateJsonSchema } from "../src/index.js";

function expectValid(instance: unknown, schema: Record<string, unknown>): void {
    expect(() => validateJsonSchema(instance, schema)).not.toThrow();
}

function expectInvalid(instance: unknown, schema: Record<string, unknown>): void {
    expect(() => validateJsonSchema(instance, schema)).toThrow();
}

describe("JSON Schema structure and composition", () => {
    it("supports local $defs and $ref", () => {
        const schema = {
            $defs: { positive: { type: "integer", minimum: 0 } },
            type: "object",
            properties: { value: { $ref: "#/$defs/positive" } },
        };

        expectValid({ value: 1 }, schema);
        expectInvalid({ value: -1 }, schema);
        expect(() => validateJsonSchema(1, { $ref: "https://example.com/schema" })).toThrow(
            /Unsupported \$ref/
        );
    });

    it("supports const, enum, not, anyOf, allOf, and oneOf", () => {
        expectValid("fixed", { const: "fixed" });
        expectInvalid("other", { const: "fixed" });
        expectValid("a", { enum: ["a", "b"] });
        expectInvalid("c", { enum: ["a", "b"] });
        expectValid(1, { not: { type: "string" } });
        expectInvalid("value", { not: { type: "string" } });
        expectValid("value", { anyOf: [{ type: "string" }, { type: "number" }] });
        expectInvalid(true, { anyOf: [{ type: "string" }, { type: "number" }] });
        expectValid(4, { allOf: [{ type: "integer" }, { minimum: 2 }] });
        expectInvalid(1, { allOf: [{ type: "integer" }, { minimum: 2 }] });
        expectValid("x", {
            oneOf: [
                { type: "string", maxLength: 2 },
                { type: "string", minLength: 4 },
            ],
        });
        expectInvalid("xx", {
            oneOf: [{ type: "string" }, { type: "string", minLength: 2 }],
        });
    });

    it("supports if, then, and else", () => {
        const schema = {
            if: { type: "number" },
            then: { minimum: 10 },
            else: { type: "string", minLength: 2 },
        };

        expectValid(10, schema);
        expectInvalid(5, schema);
        expectValid("ok", schema);
        expectInvalid("x", schema);
    });
});

describe("JSON Schema types", () => {
    it.each([
        [{}, "object"],
        [[], "array"],
        ["value", "string"],
        [1, "integer"],
        [1.5, "number"],
        [true, "boolean"],
        [null, "null"],
    ])("accepts %j as %s", (instance, type) => {
        expectValid(instance, { type });
    });

    it("supports type unions and keeps booleans distinct from numbers", () => {
        const union = { type: ["string", "null"] };

        expectValid("value", union);
        expectValid(null, union);
        expectInvalid(1, union);
        expectInvalid(true, { type: "integer" });
        expectInvalid(true, { type: "number" });
    });
});

describe("JSON Schema object keywords", () => {
    it("supports required, property counts, and properties", () => {
        const schema = {
            type: "object",
            required: ["name"],
            minProperties: 1,
            maxProperties: 2,
            properties: { name: { type: "string" } },
        };

        expectValid({ name: "Ada" }, schema);
        expectInvalid({}, schema);
        expectInvalid({ name: 1 }, schema);
        expectInvalid({ name: "Ada", a: 1, b: 2 }, schema);
    });

    it("supports patternProperties and additionalProperties", () => {
        const schema = {
            type: "object",
            properties: { fixed: { type: "string" } },
            patternProperties: { "^n_": { type: "number" } },
            additionalProperties: false,
        };

        expectValid({ fixed: "yes", n_value: 1 }, schema);
        expectInvalid({ fixed: "yes", n_value: "one" }, schema);
        expectInvalid({ fixed: "yes", extra: true }, schema);
        expectValid({ extra: 1 }, { additionalProperties: { type: "integer" } });
        expectInvalid({ extra: true }, { additionalProperties: { type: "integer" } });
    });

    it("supports property and schema dependencies", () => {
        const propertyDependency = { dependencies: { card: ["billing"] } };
        const schemaDependency = {
            dependencies: { card: { required: ["billing"] } },
        };

        expectValid({ card: "1", billing: "here" }, propertyDependency);
        expectInvalid({ card: "1" }, propertyDependency);
        expectValid({ card: "1", billing: "here" }, schemaDependency);
        expectInvalid({ card: "1" }, schemaDependency);
    });
});

describe("JSON Schema array keywords", () => {
    it("supports item counts and deep uniqueness", () => {
        const schema = { type: "array", minItems: 1, maxItems: 2, uniqueItems: true };

        expectValid([{ a: 1 }, { a: 2 }], schema);
        expectInvalid([], schema);
        expectInvalid([1, 2, 3], schema);
        expectInvalid([{ a: 1 }, { a: 1 }], schema);
    });

    it("supports single and tuple item schemas with additionalItems", () => {
        expectValid([1, 2], { items: { type: "integer" } });
        expectInvalid([1, true], { items: { type: "integer" } });

        const tuple = {
            items: [{ type: "integer" }, { type: "string" }],
            additionalItems: false,
        };
        expectValid([1, "two"], tuple);
        expectInvalid([1, "two", 3], tuple);
        expectValid([1, "two", true], {
            items: [{ type: "integer" }],
            additionalItems: { type: ["string", "boolean"] },
        });
        expectInvalid([1, "two", 3], {
            items: [{ type: "integer" }],
            additionalItems: { type: "string" },
        });
    });
});

describe("JSON Schema string and number keywords", () => {
    it("supports string lengths, patterns, and defined formats", () => {
        const schema = { type: "string", minLength: 2, maxLength: 4, pattern: "^[a-z]+$" };

        expectValid("abc", schema);
        expectInvalid("a", schema);
        expectInvalid("abcde", schema);
        expectInvalid("ABC", schema);
        expectValid("a@example.com", { format: "email" });
        expectInvalid("not-email", { format: "email" });
        expectValid("https://example.com", { format: "uri" });
        expectInvalid("not a uri", { format: "uri" });
    });

    it("supports numeric bounds and multiples", () => {
        const schema = {
            type: "number",
            minimum: 1,
            maximum: 10,
            exclusiveMinimum: 0,
            exclusiveMaximum: 11,
            multipleOf: 0.1,
        };

        expectValid(1.2, schema);
        expectInvalid(0, schema);
        expectInvalid(11, schema);
        expectInvalid(1.25, schema);
    });

    it("ignores unknown keywords and unknown formats", () => {
        expectValid({ any: "value" }, { unknownKeyword: false });
        expectValid("value", { type: "string", format: "unknown" });
    });
});
