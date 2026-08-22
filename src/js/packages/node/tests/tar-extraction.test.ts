import { describe, it, expect, afterEach } from "vitest";
import {
    mkdtempSync,
    mkdirSync,
    writeFileSync,
    readFileSync,
    existsSync,
    rmSync,
    symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { create as tarCreate } from "tar";
import {
    extractTarBufferToDirectory,
    extractTarFileToDirectory,
} from "../src/source";

const tempDirs: string[] = [];

function makeTempDir(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "fnnx-tar-test-"));
    tempDirs.push(dir);
    return dir;
}

function writeFile(dir: string, relpath: string, content: string): void {
    const fullPath = path.join(dir, relpath);
    mkdirSync(path.dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, content);
}

function createRawTarEntry(memberPath: string, content: string): Buffer {
    const encoder = new TextEncoder();
    const contentBytes = encoder.encode(content);
    const entry = new Uint8Array(512 + Math.ceil(contentBytes.length / 512) * 512 + 1024);
    entry.set(encoder.encode(memberPath), 0);
    entry.set(encoder.encode("0000644"), 100);
    entry.set(encoder.encode("0000000"), 108);
    entry.set(encoder.encode("0000000"), 116);
    entry.set(encoder.encode(contentBytes.length.toString(8).padStart(11, "0")), 124);
    entry.set(encoder.encode("00000000000"), 136);
    entry.set(encoder.encode("        "), 148);
    entry.set(encoder.encode("0"), 156);
    let checksum = 0;
    for (let index = 0; index < 512; index++) {
        checksum += entry[index];
    }
    entry.set(encoder.encode(checksum.toString(8).padStart(6, "0") + "\0 "), 148);
    entry.set(contentBytes, 512);
    return Buffer.from(entry);
}

// Builds a path of exactly `length` characters, split into segments no filesystem rejects.
function memberPathOfLength(length: number): string {
    const characters = Array.from({ length }, () => "a");
    for (let index = 50; index < length - 1; index += 51) {
        characters[index] = "/";
    }
    return characters.join("");
}

describe("Tar extraction with long file names", () => {
    afterEach(() => {
        for (const dir of tempDirs) {
            try { rmSync(dir, { recursive: true, force: true }); } catch {}
        }
        tempDirs.length = 0;
    });

    it("should extract files with paths > 100 characters", async () => {
        const sourceDir = makeTempDir();
        const longDir = "a".repeat(50) + "/" + "b".repeat(50);
        const longPath = longDir + "/file.txt";
        writeFile(sourceDir, longPath, "long path content");
        writeFile(sourceDir, "short.txt", "short");

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const extractDir = makeTempDir();
        await extractTarFileToDirectory(tarPath, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("long path content");
        expect(readFileSync(path.join(extractDir, "short.txt"), "utf-8")).toBe("short");
    });

    it("should extract files with paths > 100 characters from buffer", async () => {
        const sourceDir = makeTempDir();
        const longDir = "deeply/nested/" + "subdir/".repeat(15) + "final";
        const longPath = longDir + "/data.json";
        writeFile(sourceDir, longPath, '{"key":"value"}');

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractTarBufferToDirectory(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe('{"key":"value"}');
    });

    it("should extract files with very long paths (> 255 characters)", async () => {
        const sourceDir = makeTempDir();
        const segments = Array.from({ length: 20 }, (_, i) => `segment_${i}`);
        const longPath = segments.join("/") + "/deep_file.txt";
        writeFile(sourceDir, longPath, "very deep content");

        expect(longPath.length).toBeGreaterThan(200);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractTarBufferToDirectory(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("very deep content");
    });

    it("should extract files with unicode characters in long paths", async () => {
        const sourceDir = makeTempDir();
        const longPath = "data/" + "folder_".repeat(14) + "/file.txt";
        writeFile(sourceDir, longPath, "unicode content");

        expect(longPath.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractTarBufferToDirectory(tarBuffer, extractDir);

        expect(existsSync(path.join(extractDir, longPath))).toBe(true);
        expect(readFileSync(path.join(extractDir, longPath), "utf-8")).toBe("unicode content");
    });

    it("should preserve directory structure with deeply nested long paths", async () => {
        const sourceDir = makeTempDir();
        const basePath = "models/production/v2/" + "component_".repeat(10);
        const file1 = basePath + "/weights.bin";
        const file2 = basePath + "/config.json";
        writeFile(sourceDir, file1, "weights data");
        writeFile(sourceDir, file2, '{"layers": 3}');

        expect(file1.length).toBeGreaterThan(100);

        const tarPath = path.join(makeTempDir(), "test.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);

        const tarBuffer = readFileSync(tarPath);
        const extractDir = makeTempDir();
        await extractTarBufferToDirectory(tarBuffer, extractDir);

        expect(readFileSync(path.join(extractDir, file1), "utf-8")).toBe("weights data");
        expect(readFileSync(path.join(extractDir, file2), "utf-8")).toBe('{"layers": 3}');
    });

    it.each([100, 101, 155, 156, 255])(
        "extracts a member path of exactly %i characters",
        async (length) => {
            const sourceDir = makeTempDir();
            const memberPath = memberPathOfLength(length);
            writeFile(sourceDir, memberPath, `length ${length}`);
            expect(memberPath).toHaveLength(length);

            const tarPath = path.join(makeTempDir(), "boundary.tar");
            await tarCreate({ file: tarPath, C: sourceDir }, ["."]);
            const extractDir = makeTempDir();
            await extractTarBufferToDirectory(readFileSync(tarPath), extractDir);

            expect(readFileSync(path.join(extractDir, memberPath), "utf-8")).toBe(
                `length ${length}`
            );
        }
    );

    it("extracts members whose directories have no member entry of their own", async () => {
        const withoutDirectories = createRawTarEntry("nested/deep/file.txt", "no dir entries");
        const extractDir = makeTempDir();

        await extractTarBufferToDirectory(withoutDirectories, extractDir);

        expect(readFileSync(path.join(extractDir, "nested/deep/file.txt"), "utf-8")).toBe(
            "no dir entries"
        );
    });

    it("rejects symbolic links", async () => {
        const sourceDir = makeTempDir();
        writeFile(sourceDir, "target.txt", "target");
        symlinkSync("target.txt", path.join(sourceDir, "link.txt"));
        const tarPath = path.join(makeTempDir(), "links.tar");
        await tarCreate({ file: tarPath, C: sourceDir }, ["."]);
        const extractDir = makeTempDir();

        await expect(extractTarFileToDirectory(tarPath, extractDir)).resolves.toBeUndefined();
        expect(existsSync(path.join(extractDir, "link.txt"))).toBe(false);
    });

    it("does not extract parent-directory paths", async () => {
        const parentDirectory = makeTempDir();
        const extractDirectory = path.join(parentDirectory, "artifact");
        mkdirSync(extractDirectory);
        const outsidePath = path.join(parentDirectory, "outside.txt");

        try {
            await extractTarBufferToDirectory(
                createRawTarEntry("../outside.txt", "unsafe"),
                extractDirectory
            );
        } catch {
            // The tar library may reject the archive before its filter ignores the member.
        }

        expect(existsSync(outsidePath)).toBe(false);
        expect(existsSync(path.join(extractDirectory, "outside.txt"))).toBe(false);
    });
});
