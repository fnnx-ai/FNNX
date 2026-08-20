import { readFileSync, readdirSync, statSync, rmSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { extract as tarExtract } from "tar";
import { interfaces, LocalHandler, DtypesManager, Inputs, Outputs, DynamicAttributes, applyPatches } from "@fnnx-ai/common";
import { ONNXOpV1 } from "./ops.js";

const MANIFEST_PATCH_PATTERN = /^manifest-[^/]+\.patch\.json$/;
const META_PATTERN = /^meta(-[^/]+)?\.json$/;

const op_implementations = {
    "ONNX_v1": ONNXOpV1
}

export class Model {
    private modelDir: string;
    private needsCleanup: boolean;
    private modelFiles: interfaces.TarFileContent[];
    private manifest: interfaces.Manifest;
    private handler: LocalHandler | null = null;
    private ops: interfaces.OpInstanceConfig[];
    private variantConfig: object;

    private constructor(modelDir: string, needsCleanup: boolean) {
        this.modelDir = modelDir;
        this.needsCleanup = needsCleanup;
        this.modelFiles = listModelFiles(modelDir);

        this.manifest = this.loadManifest();
        this.ops = this.readJsonFile('ops.json') as interfaces.OpInstanceConfig[];
        this.variantConfig = this.readJsonFile('variant_config.json');

        if (needsCleanup) {
            const dir = modelDir;
            process.on('exit', () => {
                try { rmSync(dir, { recursive: true, force: true }); } catch {}
            });
        }
    }

    static async fromPath(modelPath: string): Promise<Model> {
        const stat = statSync(modelPath);
        if (stat.isDirectory()) {
            return new Model(modelPath, false);
        }
        const tempDir = mkdtempSync(path.join(tmpdir(), 'fnnx-'));
        await tarExtract({ file: modelPath, C: tempDir });
        return new Model(tempDir, true);
    }

    static async fromBuffer(modelData: ArrayBuffer | Buffer): Promise<Model> {
        const tempDir = mkdtempSync(path.join(tmpdir(), 'fnnx-'));
        const buf = Buffer.isBuffer(modelData) ? modelData : Buffer.from(modelData);
        await extractToDir(buf, tempDir);
        return new Model(tempDir, true);
    }

    async compute(inputs: Inputs, dynamicAttributes: DynamicAttributes): Promise<Outputs> {
        if (this.handler === null) {
            throw new Error('Model handler is not initialized. Please call warmup() before compute().');
        }
        return await this.handler.compute(inputs, dynamicAttributes);
    }

    async warmup(): Promise<void> {
        let externalDtypes: Record<string, object> = {};
        try {
            externalDtypes = this.readJsonFile('dtypes.json') as Record<string, object>;
        } catch {
            // dtypes.json may not exist
        }
        const dtypesManager = new DtypesManager(externalDtypes);
        const handlerConfig = { operators: op_implementations };
        const deviceMap: interfaces.DeviceMap = { accelerator: 'cpu', node_device_map: {}, variant_device_config: {} };
        this.handler = new LocalHandler(this.modelFiles, this.manifest, this.ops, this.variantConfig as interfaces.PipelineVariant, dtypesManager, deviceMap, handlerConfig);
        await this.handler.warmup();
    }

    cleanup(): void {
        if (this.needsCleanup) {
            rmSync(this.modelDir, { recursive: true, force: true });
            this.needsCleanup = false;
        }
    }

    getManifest(): interfaces.Manifest {
        return JSON.parse(JSON.stringify(this.manifest));
    }

    getMetadata(): Array<interfaces.MetaEntry> {
        const rootFiles = readdirSync(this.modelDir);
        const entries: interfaces.MetaEntry[] = [];
        for (const filename of rootFiles) {
            if (META_PATTERN.test(filename)) {
                const content = this.readJsonFile(filename);
                if (Array.isArray(content)) {
                    entries.push(...(content as interfaces.MetaEntry[]));
                }
            }
        }
        return entries;
    }

    getDtypes(): Record<string, any> {
        try {
            return this.readJsonFile("dtypes.json") as Record<string, any>;
        } catch {
            return {};
        }
    }

    private loadManifest(): interfaces.Manifest {
        let manifestData = this.readJsonFile('manifest.json') as Record<string, any>;

        const rootFiles = readdirSync(this.modelDir);
        const patchFiles = rootFiles
            .filter((f) => MANIFEST_PATCH_PATTERN.test(f))
            .sort();

        if (patchFiles.length > 0) {
            const patches = patchFiles.map(
                (pf) => this.readJsonFile(pf) as Array<{ op: string; path: string; value?: any }>
            );
            manifestData = applyPatches(manifestData, patches);
        }

        return manifestData as unknown as interfaces.Manifest;
    }

    private readJsonFile(relpath: string): object {
        const filePath = path.join(this.modelDir, relpath);
        return JSON.parse(readFileSync(filePath, 'utf-8'));
    }
}

async function extractToDir(tarBuffer: Buffer, targetDir: string): Promise<void> {
    const readable = Readable.from(tarBuffer);
    await pipeline(readable, tarExtract({ C: targetDir }));
}

function listModelFiles(dir: string): interfaces.TarFileContent[] {
    const files: interfaces.TarFileContent[] = [];

    function walk(currentDir: string, relBase: string): void {
        const entries = readdirSync(currentDir, { withFileTypes: true });
        for (const entry of entries) {
            const relPath = relBase ? `${relBase}/${entry.name}` : entry.name;
            const fullPath = path.join(currentDir, entry.name);
            if (entry.isDirectory()) {
                files.push({ relpath: relPath + '/', type: 'directory', content: null, fsPath: fullPath });
                walk(fullPath, relPath);
            } else {
                files.push({ relpath: relPath, type: 'file', content: null, fsPath: fullPath });
            }
        }
    }

    walk(dir, '');
    return files;
}
