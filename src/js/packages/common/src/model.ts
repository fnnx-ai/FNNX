import { ArtifactSource, readArtifactFile } from "./artifact.js";
import { validateManifest, validateOpDeclarations } from "./declarations.js";
import { DtypeSchema, DtypesManager } from "./dtypes.js";
import { InvalidArtifactFileError, ModelNotWarmedUpError } from "./errors.js";
import {
    DynamicAttributes,
    HandlerConfig,
    Inputs,
    LocalHandler,
    Outputs,
    validateVariantDeclarations,
} from "./handler.js";
import { Manifest, MetaEntry, OpInstanceConfig } from "./interfaces.js";
import { applyPatches, JsonObject, JsonPatch, JsonValue } from "./jsonpatcher.js";

const MANIFEST_PATCH_PATTERN = /^manifest-[^/]+\.patch\.json$/;
const META_PATTERN = /^meta(-[^/]+)?\.json$/;

export class Model {
    private readonly manifest: Manifest;
    private readonly metadata: MetaEntry[];
    private readonly dtypes: Record<string, DtypeSchema>;
    private readonly environment: Record<string, unknown>;
    private readonly source: ArtifactSource;
    private readonly ops: OpInstanceConfig[];
    private readonly variantConfig: JsonObject;
    private readonly dtypesManager: DtypesManager;
    private readonly handlerConfig: HandlerConfig;
    private handler: LocalHandler | null = null;
    private warmedUp = false;

    constructor(source: ArtifactSource, config: HandlerConfig) {
        this.manifest = validateManifest(this.loadManifest(source));
        this.ops = validateOpDeclarations(parseJsonFile(source, "ops.json"));
        const variantValue = parseJsonFile(source, "variant_config.json");
        if (!isObject(variantValue)) {
            throw new InvalidArtifactFileError("variant_config.json", "expected an object");
        }
        this.variantConfig = variantValue;
        const dtypes = this.loadOptionalObject(source, "dtypes.json");
        for (const schema of Object.values(dtypes)) {
            if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
                throw new InvalidArtifactFileError(
                    "dtypes.json",
                    "expected each dtype schema to be an object"
                );
            }
        }
        this.dtypes = dtypes as Record<string, DtypeSchema>;
        this.dtypesManager = new DtypesManager(this.dtypes);
        this.environment = this.loadOptionalObject(source, "env.json");
        this.metadata = this.loadMetadata(source);
        this.source = source;
        this.handlerConfig = config;
        validateVariantDeclarations(this.manifest, this.ops, this.variantConfig);
    }

    async warmup(): Promise<void> {
        await this.getHandler().warmup();
        this.warmedUp = true;
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes = {}): Promise<Outputs> {
        if (!this.warmedUp) {
            throw new ModelNotWarmedUpError();
        }
        return this.getHandler().compute(inputs, dynamicAttributes);
    }

    /**
     * The handler is built on first use, so an artifact whose variant or operations this
     * consumer cannot execute still exposes what Core defines.
     */
    protected getHandler(): LocalHandler {
        this.handler ??= new LocalHandler(
            this.source,
            this.manifest,
            this.ops,
            this.variantConfig,
            this.dtypesManager,
            this.handlerConfig
        );
        return this.handler;
    }

    getManifest(): Manifest {
        return cloneJson(this.manifest);
    }

    getMetadata(): MetaEntry[] {
        return cloneJson(this.metadata);
    }

    getDtypes(): Record<string, DtypeSchema> {
        return cloneJson(this.dtypes);
    }

    getEnv(): Record<string, unknown> {
        return cloneJson(this.environment);
    }

    private loadManifest(source: ArtifactSource): JsonObject {
        const value = parseJsonFile(source, "manifest.json");
        if (!isObject(value)) {
            throw new InvalidArtifactFileError("manifest.json", "expected an object");
        }
        const patchNames = source
            .listRootMembers()
            .filter((name) => MANIFEST_PATCH_PATTERN.test(name))
            .sort(compareUtf8ByteOrder);
        const patches = patchNames.map((name) => {
            const patch = parseJsonFile(source, name);
            if (!Array.isArray(patch)) {
                throw new InvalidArtifactFileError(name, "expected a JSON Patch array");
            }
            return patch as unknown as JsonPatch;
        });
        return applyPatches(value, patches);
    }

    private loadMetadata(source: ArtifactSource): MetaEntry[] {
        const entries: MetaEntry[] = [];
        const matchingNames = source.listRootMembers().filter((name) => META_PATTERN.test(name));
        const sidecarNames = matchingNames
            .filter((name) => name !== "meta.json")
            .sort(compareUtf8ByteOrder);
        const metadataNames = matchingNames.includes("meta.json")
            ? ["meta.json", ...sidecarNames]
            : sidecarNames;

        for (const name of metadataNames) {
            let value: JsonValue;
            try {
                value = parseJsonFile(source, name);
            } catch (error) {
                if (!(error instanceof InvalidArtifactFileError)) {
                    throw error;
                }
                console.warn(`Ignoring unparseable metadata file \`${name}\`: ${error.message}`);
                continue;
            }
            if (!Array.isArray(value)) {
                console.warn(`Ignoring metadata file \`${name}\` because it is not a JSON array`);
                continue;
            }
            for (const entry of value) {
                if (isMetaEntry(entry)) {
                    entries.push(entry as unknown as MetaEntry);
                }
            }
        }
        return entries;
    }

    private loadOptionalObject(source: ArtifactSource, path: string): Record<string, unknown> {
        if (!source.listRootMembers().includes(path)) {
            return {};
        }
        const value = parseJsonFile(source, path);
        if (!isObject(value)) {
            throw new InvalidArtifactFileError(path, "expected an object");
        }
        return value;
    }
}

function parseJsonFile(source: ArtifactSource, path: string): JsonValue {
    try {
        const bytes = readArtifactFile(source, path).read();
        const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
        return JSON.parse(text) as JsonValue;
    } catch (error) {
        if (error instanceof InvalidArtifactFileError) {
            throw error;
        }
        if (error instanceof SyntaxError || error instanceof TypeError) {
            throw new InvalidArtifactFileError(path, error.message);
        }
        throw error;
    }
}

function isObject(value: JsonValue): value is JsonObject {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isMetaEntry(value: JsonValue): boolean {
    return (
        isObject(value) &&
        typeof value.id === "string" &&
        typeof value.producer === "string" &&
        typeof value.producer_version === "string" &&
        Array.isArray(value.producer_tags) &&
        value.producer_tags.every((tag) => typeof tag === "string") &&
        isObject(value.payload)
    );
}

function compareUtf8ByteOrder(left: string, right: string): number {
    const encoder = new TextEncoder();
    const leftBytes = encoder.encode(left);
    const rightBytes = encoder.encode(right);
    const sharedLength = Math.min(leftBytes.length, rightBytes.length);
    for (let index = 0; index < sharedLength; index++) {
        if (leftBytes[index] !== rightBytes[index]) {
            return leftBytes[index] - rightBytes[index];
        }
    }
    return leftBytes.length - rightBytes.length;
}

function cloneJson<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}
