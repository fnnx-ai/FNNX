import { readdirSync, readFileSync } from "node:fs";
import type { Stats } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { extract as tarExtract } from "tar";
import type { ReadEntry } from "tar";
import {
    ArtifactFile,
    ArtifactSource,
    MissingArtifactFileError,
    assertSafeArtifactPath,
} from "@fnnx-ai/common";

const ALLOWED_TAR_TYPES = new Set(["File", "OldFile", "ContiguousFile", "Directory"]);

class NodeArtifactFile implements ArtifactFile {
    constructor(
        public readonly path: string,
        public readonly filesystemPath: string
    ) {}

    read(): Uint8Array {
        const buffer = readFileSync(this.filesystemPath);
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
}

export class NodeArtifactSource implements ArtifactSource {
    private readonly files = new Map<string, NodeArtifactFile>();

    constructor(private readonly rootDirectory: string) {
        this.collectFiles(rootDirectory, "");
    }

    listRootMembers(): string[] {
        return [...this.files.keys()].filter((name) => !name.includes("/"));
    }

    readFile(filePath: string): ArtifactFile {
        const file = this.files.get(filePath);
        if (!file) {
            throw new MissingArtifactFileError(filePath);
        }
        return file;
    }

    resolveOpArtifacts(opInstanceId: string): ArtifactFile[] {
        const prefix = `ops_artifacts/${opInstanceId}/`;
        return [...this.files.values()].filter((file) => file.path.startsWith(prefix));
    }

    private collectFiles(directory: string, relativeDirectory: string): void {
        for (const entry of readdirSync(directory, { withFileTypes: true })) {
            const relativePath = relativeDirectory
                ? `${relativeDirectory}/${entry.name}`
                : entry.name;
            const filesystemPath = path.join(directory, entry.name);
            if (entry.isDirectory()) {
                this.collectFiles(filesystemPath, relativePath);
            } else if (entry.isFile()) {
                assertSafeArtifactPath(relativePath);
                this.files.set(relativePath, new NodeArtifactFile(relativePath, filesystemPath));
            }
        }
    }
}

export async function extractTarFileToDirectory(
    tarPath: string,
    targetDirectory: string
): Promise<void> {
    await tarExtract({
        file: tarPath,
        C: targetDirectory,
        preservePaths: false,
        strict: true,
        filter: safeTarFilter,
    });
}

export async function extractTarBufferToDirectory(
    tarBuffer: Buffer,
    targetDirectory: string
): Promise<void> {
    await pipeline(
        Readable.from(tarBuffer),
        tarExtract({
            C: targetDirectory,
            preservePaths: false,
            strict: true,
            filter: safeTarFilter,
        })
    );
}

function safeTarFilter(memberPath: string, entry: Stats | ReadEntry): boolean {
    try {
        assertSafeArtifactPath(memberPath);
    } catch {
        return false;
    }
    if ("type" in entry && !ALLOWED_TAR_TYPES.has(entry.type)) {
        return false;
    }
    return true;
}
