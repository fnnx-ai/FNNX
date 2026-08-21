import { ArtifactSource, readArtifactFile } from "./artifact.js";
import { DtypeSchema, DtypesManager } from "./dtypes.js";
import { InvalidArtifactFileError, ModelNotWarmedUpError } from "./errors.js";
import { DynamicAttributes, HandlerConfig, Inputs, LocalHandler, Outputs } from "./handler.js";
import { Manifest, MetaEntry, OpInstanceConfig } from "./interfaces.js";
import { applyPatches, JsonObject, JsonPatch, JsonValue } from "./jsonpatcher.js";

const MANIFEST_PATCH_PATTERN = /^manifest-[^/]+\.patch\.json$/;
const META_PATTERN = /^meta(-[^/]+)?\.json$/;

export class Model {
    private readonly manifest: Manifest;
    private readonly metadata: MetaEntry[];
    private readonly dtypes: Record<string, DtypeSchema>;
    private readonly environment: Record<string, unknown>;
    private readonly handler: LocalHandler;
    private warmedUp = false;

    constructor(source: ArtifactSource, config: HandlerConfig) {
        this.manifest = this.loadManifest(source);
        const opsValue = parseJsonFile(source, "ops.json");
        if (!Array.isArray(opsValue)) {
            throw new InvalidArtifactFileError("ops.json", "expected an array");
        }
        const variantValue = parseJsonFile(source, "variant_config.json");
        if (!isObject(variantValue)) {
            throw new InvalidArtifactFileError("variant_config.json", "expected an object");
        }
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
        this.environment = this.loadOptionalObject(source, "env.json");
        this.metadata = this.loadMetadata(source);
        this.handler = new LocalHandler(
            source,
            this.manifest,
            opsValue as unknown as OpInstanceConfig[],
            variantValue,
            new DtypesManager(this.dtypes),
            config
        );
    }

    async warmup(): Promise<void> {
        await this.handler.warmup();
        this.warmedUp = true;
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes = {}): Promise<Outputs> {
        if (!this.warmedUp) {
            throw new ModelNotWarmedUpError();
        }
        return this.handler.compute(inputs, dynamicAttributes);
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

    private loadManifest(source: ArtifactSource): Manifest {
        const value = parseJsonFile(source, "manifest.json");
        if (!isObject(value)) {
            throw new InvalidArtifactFileError("manifest.json", "expected an object");
        }
        const patchNames = source
            .listRootMembers()
            .filter((name) => MANIFEST_PATCH_PATTERN.test(name))
            .sort();
        const patches = patchNames.map((name) => {
            const patch = parseJsonFile(source, name);
            if (!Array.isArray(patch)) {
                throw new InvalidArtifactFileError(name, "expected a JSON Patch array");
            }
            return patch as unknown as JsonPatch;
        });
        return applyPatches(value as JsonObject, patches) as unknown as Manifest;
    }

    private loadMetadata(source: ArtifactSource): MetaEntry[] {
        const entries: MetaEntry[] = [];
        for (const name of source.listRootMembers().filter((item) => META_PATTERN.test(item))) {
            const value = parseJsonFile(source, name);
            if (Array.isArray(value)) {
                entries.push(...(value as unknown as MetaEntry[]));
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

function cloneJson<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}
