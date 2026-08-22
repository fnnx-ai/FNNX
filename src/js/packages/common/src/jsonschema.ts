export type JsonSchema = Record<string, unknown> | boolean;

type Definitions = Record<string, JsonSchema>;

export function validateJsonSchema(instance: unknown, schema: JsonSchema): void {
    const definitions = collectDefinitions(schema);
    validate(instance, schema, definitions);
}

function collectDefinitions(schema: JsonSchema): Definitions {
    if (!isRecord(schema) || !isRecord(schema.$defs)) {
        return {};
    }

    return Object.fromEntries(
        Object.entries(schema.$defs).map(([name, definition]) => [
            name,
            asSchema(definition, `$defs.${name}`),
        ])
    );
}

function validate(instance: unknown, schema: JsonSchema, definitions: Definitions): void {
    if (typeof schema === "boolean") {
        if (!schema) {
            throw new Error("Value is rejected by a false schema");
        }
        return;
    }

    if ("$ref" in schema) {
        validate(instance, resolveReference(schema.$ref, definitions), definitions);
        return;
    }

    validateConstAndEnum(instance, schema);
    validateComposition(instance, schema, definitions);
    validateType(instance, schema.type);

    if (isRecord(instance)) {
        validateObject(instance, schema, definitions);
    } else if (Array.isArray(instance)) {
        validateArray(instance, schema, definitions);
    } else if (typeof instance === "string") {
        validateString(instance, schema);
    } else if (typeof instance === "number") {
        validateNumber(instance, schema);
    }
}

function resolveReference(reference: unknown, definitions: Definitions): JsonSchema {
    if (typeof reference !== "string" || !reference.startsWith("#/$defs/")) {
        throw new Error(`Unsupported $ref: ${String(reference)}`);
    }

    const name = reference.slice("#/$defs/".length);
    if (!Object.prototype.hasOwnProperty.call(definitions, name)) {
        throw new Error(`Unknown $ref: ${reference}`);
    }
    return definitions[name];
}

function validateConstAndEnum(instance: unknown, schema: Record<string, unknown>): void {
    if (
        Object.prototype.hasOwnProperty.call(schema, "const") &&
        !jsonEqual(instance, schema.const)
    ) {
        throw new Error("Value does not match const");
    }

    if ("enum" in schema) {
        if (
            !Array.isArray(schema.enum) ||
            !schema.enum.some((value) => jsonEqual(instance, value))
        ) {
            throw new Error("Value is not in enum");
        }
    }
}

function validateComposition(
    instance: unknown,
    schema: Record<string, unknown>,
    definitions: Definitions
): void {
    if ("not" in schema && matches(instance, asSchema(schema.not, "not"), definitions)) {
        throw new Error('Value must not match the schema in "not"');
    }

    if ("anyOf" in schema) {
        const schemas = asSchemaArray(schema.anyOf, "anyOf");
        if (!schemas.some((candidate) => matches(instance, candidate, definitions))) {
            throw new Error("Value does not match any schema in anyOf");
        }
    }

    if ("allOf" in schema) {
        for (const candidate of asSchemaArray(schema.allOf, "allOf")) {
            validate(instance, candidate, definitions);
        }
    }

    if ("oneOf" in schema) {
        const matchCount = asSchemaArray(schema.oneOf, "oneOf").filter((candidate) =>
            matches(instance, candidate, definitions)
        ).length;
        if (matchCount !== 1) {
            throw new Error(`Value must match exactly one schema in oneOf; matched ${matchCount}`);
        }
    }

    if ("if" in schema) {
        const branch = matches(instance, asSchema(schema.if, "if"), definitions)
            ? schema.then
            : schema.else;
        if (branch !== undefined) {
            validate(instance, asSchema(branch, "conditional branch"), definitions);
        }
    }
}

function matches(instance: unknown, schema: JsonSchema, definitions: Definitions): boolean {
    try {
        validate(instance, schema, definitions);
        return true;
    } catch {
        return false;
    }
}

function validateType(instance: unknown, typeDeclaration: unknown): void {
    if (typeDeclaration === undefined) {
        return;
    }

    const types = Array.isArray(typeDeclaration) ? typeDeclaration : [typeDeclaration];
    if (!types.every((type) => typeof type === "string")) {
        throw new Error("Schema type must be a string or an array of strings");
    }
    if (!types.some((type) => hasJsonType(instance, type as string))) {
        throw new Error(`Expected type ${types.join(" or ")}, got ${jsonTypeOf(instance)}`);
    }
}

function hasJsonType(instance: unknown, expected: string): boolean {
    switch (expected) {
        case "object":
            return isRecord(instance);
        case "array":
            return Array.isArray(instance);
        case "string":
            return typeof instance === "string";
        case "integer":
            return (
                typeof instance === "number" &&
                Number.isFinite(instance) &&
                Number.isInteger(instance)
            );
        case "number":
            return typeof instance === "number" && Number.isFinite(instance);
        case "boolean":
            return typeof instance === "boolean";
        case "null":
            return instance === null;
        default:
            return false;
    }
}

function validateObject(
    instance: Record<string, unknown>,
    schema: Record<string, unknown>,
    definitions: Definitions
): void {
    const keys = Object.keys(instance);
    if ("required" in schema) {
        for (const property of asStringArray(schema.required, "required")) {
            if (!Object.prototype.hasOwnProperty.call(instance, property)) {
                throw new Error(`Missing required property: ${property}`);
            }
        }
    }

    validateCount(keys.length, schema.minProperties, schema.maxProperties, "properties");

    const properties =
        schema.properties === undefined ? {} : asRecord(schema.properties, "properties");
    for (const [property, propertySchema] of Object.entries(properties)) {
        if (Object.prototype.hasOwnProperty.call(instance, property)) {
            validate(
                instance[property],
                asSchema(propertySchema, `properties.${property}`),
                definitions
            );
        }
    }

    const patternProperties =
        schema.patternProperties === undefined
            ? {}
            : asRecord(schema.patternProperties, "patternProperties");
    const patterns = Object.entries(patternProperties).map(
        ([pattern, patternSchema]) =>
            [new RegExp(pattern), asSchema(patternSchema, `patternProperties.${pattern}`)] as const
    );
    for (const property of keys) {
        for (const [pattern, patternSchema] of patterns) {
            if (pattern.test(property)) {
                validate(instance[property], patternSchema, definitions);
            }
        }
    }

    if ("additionalProperties" in schema) {
        const extras = keys.filter(
            (property) =>
                !Object.prototype.hasOwnProperty.call(properties, property) &&
                !patterns.some(([pattern]) => pattern.test(property))
        );
        validateAdditionalProperties(instance, extras, schema.additionalProperties, definitions);
    }

    if ("dependencies" in schema) {
        validateDependencies(instance, asRecord(schema.dependencies, "dependencies"), definitions);
    }
}

function validateAdditionalProperties(
    instance: Record<string, unknown>,
    extras: string[],
    declaration: unknown,
    definitions: Definitions
): void {
    if (declaration === false && extras.length > 0) {
        throw new Error(`Additional properties are not allowed: ${extras.join(", ")}`);
    }
    if (declaration === true) {
        return;
    }
    if (declaration !== false) {
        const additionalSchema = asSchema(declaration, "additionalProperties");
        for (const property of extras) {
            validate(instance[property], additionalSchema, definitions);
        }
    }
}

function validateDependencies(
    instance: Record<string, unknown>,
    dependencies: Record<string, unknown>,
    definitions: Definitions
): void {
    for (const [property, dependency] of Object.entries(dependencies)) {
        if (!Object.prototype.hasOwnProperty.call(instance, property)) {
            continue;
        }
        if (Array.isArray(dependency)) {
            for (const requiredProperty of asStringArray(dependency, `dependencies.${property}`)) {
                if (!Object.prototype.hasOwnProperty.call(instance, requiredProperty)) {
                    throw new Error(
                        `Property ${property} depends on missing property ${requiredProperty}`
                    );
                }
            }
        } else {
            validate(instance, asSchema(dependency, `dependencies.${property}`), definitions);
        }
    }
}

function validateArray(
    instance: unknown[],
    schema: Record<string, unknown>,
    definitions: Definitions
): void {
    validateCount(instance.length, schema.minItems, schema.maxItems, "items");

    if (schema.uniqueItems === true) {
        for (let index = 0; index < instance.length; index += 1) {
            if (instance.slice(0, index).some((value) => jsonEqual(value, instance[index]))) {
                throw new Error("Array items are not unique");
            }
        }
    }

    if (!("items" in schema)) {
        return;
    }
    if (Array.isArray(schema.items)) {
        validateTupleItems(instance, schema.items, schema.additionalItems, definitions);
        return;
    }

    const itemSchema = asSchema(schema.items, "items");
    for (const item of instance) {
        validate(item, itemSchema, definitions);
    }
}

function validateTupleItems(
    instance: unknown[],
    itemSchemas: unknown[],
    additionalItems: unknown,
    definitions: Definitions
): void {
    for (let index = 0; index < Math.min(instance.length, itemSchemas.length); index += 1) {
        validate(instance[index], asSchema(itemSchemas[index], `items.${index}`), definitions);
    }

    if (
        instance.length <= itemSchemas.length ||
        additionalItems === undefined ||
        additionalItems === true
    ) {
        return;
    }
    if (additionalItems === false) {
        throw new Error("Additional items are not allowed");
    }

    const additionalSchema = asSchema(additionalItems, "additionalItems");
    for (let index = itemSchemas.length; index < instance.length; index += 1) {
        validate(instance[index], additionalSchema, definitions);
    }
}

function validateString(instance: string, schema: Record<string, unknown>): void {
    validateCount([...instance].length, schema.minLength, schema.maxLength, "string characters");

    if ("pattern" in schema) {
        if (typeof schema.pattern !== "string") {
            throw new Error("Schema pattern must be a string");
        }
        if (!new RegExp(schema.pattern).test(instance)) {
            throw new Error(`String does not match pattern ${schema.pattern}`);
        }
    }

    if (schema.format === "email" && !/^[^@]+@[^@]+\.[^@]+$/.test(instance)) {
        throw new Error("String is not a valid email address");
    }
    if (schema.format === "uri" && !/^[A-Za-z][A-Za-z0-9+.-]*:/.test(instance)) {
        throw new Error("String is not a valid URI");
    }
}

function validateNumber(instance: number, schema: Record<string, unknown>): void {
    const minimum = optionalNumber(schema.minimum, "minimum");
    const maximum = optionalNumber(schema.maximum, "maximum");
    const exclusiveMinimum = optionalNumber(schema.exclusiveMinimum, "exclusiveMinimum");
    const exclusiveMaximum = optionalNumber(schema.exclusiveMaximum, "exclusiveMaximum");
    const multipleOf = optionalNumber(schema.multipleOf, "multipleOf");

    if (minimum !== undefined && instance < minimum) {
        throw new Error(`Value ${instance} is less than minimum ${minimum}`);
    }
    if (maximum !== undefined && instance > maximum) {
        throw new Error(`Value ${instance} is greater than maximum ${maximum}`);
    }
    if (exclusiveMinimum !== undefined && instance <= exclusiveMinimum) {
        throw new Error(
            `Value ${instance} is not greater than exclusiveMinimum ${exclusiveMinimum}`
        );
    }
    if (exclusiveMaximum !== undefined && instance >= exclusiveMaximum) {
        throw new Error(`Value ${instance} is not less than exclusiveMaximum ${exclusiveMaximum}`);
    }
    if (multipleOf !== undefined && !isMultipleOf(instance, multipleOf)) {
        throw new Error(`Value ${instance} is not a multiple of ${multipleOf}`);
    }
}

function isMultipleOf(value: number, divisor: number): boolean {
    if (divisor <= 0) {
        throw new Error("multipleOf must be greater than zero");
    }
    const quotient = value / divisor;
    const difference = Math.abs(quotient - Math.round(quotient));
    return difference <= Number.EPSILON * Math.max(1, Math.abs(quotient)) * 4;
}

function validateCount(
    count: number,
    minimumDeclaration: unknown,
    maximumDeclaration: unknown,
    unit: string
): void {
    const minimum = optionalNumber(minimumDeclaration, `minimum ${unit}`);
    const maximum = optionalNumber(maximumDeclaration, `maximum ${unit}`);
    if (minimum !== undefined && count < minimum) {
        throw new Error(`Expected at least ${minimum} ${unit}, got ${count}`);
    }
    if (maximum !== undefined && count > maximum) {
        throw new Error(`Expected at most ${maximum} ${unit}, got ${count}`);
    }
}

function optionalNumber(value: unknown, keyword: string): number | undefined {
    if (value === undefined) {
        return undefined;
    }
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${keyword} must be a finite number`);
    }
    return value;
}

function asSchema(value: unknown, keyword: string): JsonSchema {
    if (typeof value === "boolean" || isRecord(value)) {
        return value;
    }
    throw new Error(`${keyword} must contain a schema`);
}

function asSchemaArray(value: unknown, keyword: string): JsonSchema[] {
    if (!Array.isArray(value)) {
        throw new Error(`${keyword} must be an array of schemas`);
    }
    return value.map((schema, index) => asSchema(schema, `${keyword}.${index}`));
}

function asRecord(value: unknown, keyword: string): Record<string, unknown> {
    if (!isRecord(value)) {
        throw new Error(`${keyword} must be an object`);
    }
    return value;
}

function asStringArray(value: unknown, keyword: string): string[] {
    if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
        throw new Error(`${keyword} must be an array of strings`);
    }
    return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function jsonTypeOf(value: unknown): string {
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value)) {
        return "array";
    }
    return typeof value;
}

function jsonEqual(left: unknown, right: unknown): boolean {
    if (left === right) {
        return true;
    }
    if (Array.isArray(left) && Array.isArray(right)) {
        return (
            left.length === right.length &&
            left.every((value, index) => jsonEqual(value, right[index]))
        );
    }
    if (isRecord(left) && isRecord(right)) {
        const leftKeys = Object.keys(left);
        const rightKeys = Object.keys(right);
        return (
            leftKeys.length === rightKeys.length &&
            leftKeys.every(
                (key) =>
                    Object.prototype.hasOwnProperty.call(right, key) &&
                    jsonEqual(left[key], right[key])
            )
        );
    }
    return false;
}
