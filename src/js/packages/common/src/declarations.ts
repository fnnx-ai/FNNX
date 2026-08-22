import { InvalidArtifactFileError } from "./errors.js";
import { Manifest, OpInstanceConfig } from "./interfaces.js";
import { JsonObject, JsonValue } from "./jsonpatcher.js";

const CONTENT_TYPES = new Set(["NDJSON", "JSON"]);
const OP_INSTANCE_ID_PATTERN = /^[a-zA-Z0-9_]+$/;

export function validateManifest(value: JsonValue): Manifest {
    const manifest = requireObject(value, "manifest.json", "the manifest");
    for (const field of ["variant", "producer_name", "producer_version"] as const) {
        if (typeof manifest[field] !== "string") {
            throw new InvalidArtifactFileError("manifest.json", `\`${field}\` must be a string`);
        }
    }
    for (const field of ["producer_tags", "dynamic_attributes", "env_vars"] as const) {
        if (!Array.isArray(manifest[field])) {
            throw new InvalidArtifactFileError("manifest.json", `\`${field}\` must be an array`);
        }
    }
    for (const kind of ["inputs", "outputs"] as const) {
        const entries = manifest[kind];
        if (!Array.isArray(entries)) {
            throw new InvalidArtifactFileError("manifest.json", `\`${kind}\` must be an array`);
        }
        const declaredNames = new Set<string>();
        entries.forEach((entry, index) => {
            const name = validateIOEntry(entry, kind, index);
            if (declaredNames.has(name)) {
                throw new InvalidArtifactFileError(
                    "manifest.json",
                    `${kind.slice(0, -1)} \`${name}\` is declared more than once; ` +
                        `${kind} names must be unique`
                );
            }
            declaredNames.add(name);
        });
    }
    return manifest as unknown as Manifest;
}

export function validateOpDeclarations(value: JsonValue): OpInstanceConfig[] {
    if (!Array.isArray(value)) {
        throw new InvalidArtifactFileError("ops.json", "expected an array");
    }
    const declaredIds = new Set<string>();
    value.forEach((entry, index) => {
        const instance = requireObject(entry, "ops.json", `entry ${index}`);
        const id = instance.id;
        if (typeof id !== "string" || !OP_INSTANCE_ID_PATTERN.test(id)) {
            throw new InvalidArtifactFileError(
                "ops.json",
                `entry ${index} declares id \`${String(id)}\`, which does not match ` +
                    `${OP_INSTANCE_ID_PATTERN.source}`
            );
        }
        if (declaredIds.has(id)) {
            throw new InvalidArtifactFileError(
                "ops.json",
                `op instance id \`${id}\` is declared more than once`
            );
        }
        declaredIds.add(id);
        if (typeof instance.op !== "string") {
            throw new InvalidArtifactFileError(
                "ops.json",
                `op instance \`${id}\` must declare a string \`op\``
            );
        }
        for (const kind of ["inputs", "outputs"] as const) {
            if (!Array.isArray(instance[kind])) {
                throw new InvalidArtifactFileError(
                    "ops.json",
                    `op instance \`${id}\` must declare \`${kind}\` as an array`
                );
            }
        }
    });
    return value as unknown as OpInstanceConfig[];
}

function validateIOEntry(entry: JsonValue, kind: "inputs" | "outputs", index: number): string {
    const io = requireObject(entry, "manifest.json", `${kind}[${index}]`);
    const name = io.name;
    if (typeof name !== "string") {
        throw new InvalidArtifactFileError(
            "manifest.json",
            `${kind}[${index}] must declare a string \`name\``
        );
    }
    const contentType = io.content_type;
    if (typeof contentType !== "string" || !CONTENT_TYPES.has(contentType)) {
        throw new InvalidArtifactFileError(
            "manifest.json",
            `${kind.slice(0, -1)} \`${name}\` declares unsupported content type ` +
                `\`${String(contentType)}\``
        );
    }
    if (typeof io.dtype !== "string") {
        throw new InvalidArtifactFileError(
            "manifest.json",
            `${kind.slice(0, -1)} \`${name}\` must declare a string \`dtype\``
        );
    }
    if (contentType === "NDJSON" && !Array.isArray(io.shape)) {
        throw new InvalidArtifactFileError(
            "manifest.json",
            `NDJSON ${kind.slice(0, -1)} \`${name}\` must declare a \`shape\` array`
        );
    }
    return name;
}

function requireObject(value: JsonValue, filePath: string, label: string): JsonObject {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new InvalidArtifactFileError(filePath, `${label} must be an object`);
    }
    return value;
}
