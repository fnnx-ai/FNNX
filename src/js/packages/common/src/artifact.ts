import { MissingArtifactFileError, UnsafeTarMemberError } from "./errors.js";

export interface ArtifactFile {
    readonly path: string;
    readonly filesystemPath?: string;
    read(): Uint8Array;
}

export interface ArtifactSource {
    listRootMembers(): string[];
    readFile(path: string): ArtifactFile;
    resolveOpArtifacts(opInstanceId: string): ArtifactFile[];
}

export function readArtifactFile(source: ArtifactSource, path: string): ArtifactFile {
    const file = source.readFile(path);
    if (!file) {
        throw new MissingArtifactFileError(path);
    }
    return file;
}

export function assertSafeArtifactPath(memberPath: string): void {
    if (
        memberPath.startsWith("/") ||
        memberPath.startsWith("\\") ||
        /^[a-zA-Z]:[\\/]/.test(memberPath)
    ) {
        throw new UnsafeTarMemberError(memberPath, "absolute paths are not permitted");
    }

    if (memberPath.split(/[\\/]/).includes("..")) {
        throw new UnsafeTarMemberError(memberPath, "parent-directory segments are not permitted");
    }
}
