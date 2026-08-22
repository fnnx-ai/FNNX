import { ArtifactFile, ArtifactSource, MissingArtifactFileError } from "@fnnx-ai/common";
import { TarEntry, TarExtractor } from "./tar.js";

class WebArtifactFile implements ArtifactFile {
    constructor(
        public readonly path: string,
        private readonly content: Uint8Array
    ) {}

    read(): Uint8Array {
        return this.content.slice();
    }
}

export class WebArtifactSource implements ArtifactSource {
    private readonly files = new Map<string, WebArtifactFile>();

    constructor(modelData: ArrayBuffer) {
        for (const entry of new TarExtractor(modelData).extract()) {
            this.addEntry(entry);
        }
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

    private addEntry(entry: TarEntry): void {
        if (entry.type === "file") {
            this.files.set(entry.relpath, new WebArtifactFile(entry.relpath, entry.content));
        }
    }
}
