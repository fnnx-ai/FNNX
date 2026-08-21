import { describe, it, expect } from "vitest";
import { InvalidTarError, UnsafeTarMemberError } from "@fnnx-ai/common";
import { TarExtractor } from "../../src/tar";

function computeChecksum(view: Uint8Array): string {
    let checksum = 0;
    for (let i = 0; i < 148; i++) checksum += view[i];
    for (let i = 148; i < 156; i++) checksum += 32;
    for (let i = 156; i < 512; i++) checksum += view[i];
    return checksum.toString(8).padStart(6, "0") + "\0 ";
}

function createMockTarFile(fileName: string, content: string): ArrayBuffer {
    const buffer = new ArrayBuffer(1024);
    const view = new Uint8Array(buffer);
    const encoder = new TextEncoder();

    view.set(encoder.encode(fileName), 0);
    view.set(encoder.encode("0000644"), 100);
    view.set(encoder.encode("0000000"), 108);
    view.set(encoder.encode("0000000"), 116);

    const sizeOctal = content.length.toString(8).padStart(11, "0");
    view.set(encoder.encode(sizeOctal), 124);
    view.set(encoder.encode("00000000000"), 136);
    view.set(encoder.encode("        "), 148);
    view.set(encoder.encode("0"), 156);

    view.set(encoder.encode(computeChecksum(view)), 148);
    view.set(encoder.encode(content), 512);

    return buffer;
}

function createTarHeader(
    name: string,
    size: number,
    typeflag: number,
    prefix?: string
): Uint8Array {
    const header = new Uint8Array(512);
    const encoder = new TextEncoder();

    header.set(encoder.encode(name), 0);
    header.set(encoder.encode("0000644"), 100);
    header.set(encoder.encode("0000000"), 108);
    header.set(encoder.encode("0000000"), 116);

    const sizeOctal = size.toString(8).padStart(11, "0");
    header.set(encoder.encode(sizeOctal), 124);
    header.set(encoder.encode("00000000000"), 136);
    header.set(encoder.encode("        "), 148);
    header[156] = typeflag;

    if (prefix) {
        header.set(encoder.encode(prefix), 345);
    }

    header.set(encoder.encode(computeChecksum(header)), 148);
    return header;
}

function align512(size: number): number {
    const remainder = size % 512;
    return remainder ? size + (512 - remainder) : size;
}

function buildTarBuffer(blocks: Uint8Array[]): ArrayBuffer {
    const totalSize = blocks.reduce((sum, b) => sum + b.length, 0) + 1024;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const block of blocks) {
        view.set(block, offset);
        offset += block.length;
    }
    return buffer;
}

describe("TarExtractor", () => {
    describe("extract()", () => {
        it("should extract a simple file from TAR archive", () => {
            const fileName = "test.txt";
            const content = "Hello, World!";
            const tarBuffer = createMockTarFile(fileName, content);

            const extractor = new TarExtractor(tarBuffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe(fileName);
            expect(files[0].type).toBe("file");

            const extractedContent = new TextDecoder().decode(files[0].content!);
            expect(extractedContent).toContain("Hello, World!");
        });

        it("should return empty array for empty TAR archive", () => {
            const buffer = new ArrayBuffer(1024);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();
            expect(files).toHaveLength(0);
        });
    });

    describe("scan()", () => {
        it("should scan and return file positions", () => {
            const fileName = "test.txt";
            const content = "Hello, World!";
            const tarBuffer = createMockTarFile(fileName, content);

            const extractor = new TarExtractor(tarBuffer);
            const scanResults = extractor.scan();

            expect(scanResults.size).toBe(1);
            expect(scanResults.has(fileName)).toBe(true);

            const [offset, size] = scanResults.get(fileName)!;
            expect(offset).toBe(512);
            expect(size).toBe(content.length);
        });

        it("should return empty map for empty archive", () => {
            const buffer = new ArrayBuffer(1024);
            const extractor = new TarExtractor(buffer);
            const scanResults = extractor.scan();
            expect(scanResults.size).toBe(0);
        });
    });

    describe("Error handling", () => {
        it("should throw error for invalid checksum", () => {
            const buffer = new ArrayBuffer(1024);
            const view = new Uint8Array(buffer);
            const encoder = new TextEncoder();

            view.set(encoder.encode("test.txt"), 0);
            view.set(encoder.encode("0000644"), 100);
            view.set(encoder.encode("0000013"), 124);
            view.set(encoder.encode("9999999"), 148);
            view.set(encoder.encode("0"), 156);

            const extractor = new TarExtractor(buffer);
            expect(() => extractor.extract()).toThrowError(InvalidTarError);
        });

        it("should throw error when file extends beyond buffer", () => {
            const buffer = new ArrayBuffer(1024);
            const view = new Uint8Array(buffer);
            const encoder = new TextEncoder();

            view.set(encoder.encode("test.txt"), 0);
            view.set(encoder.encode("0000644"), 100);
            view.set(encoder.encode("7777777777"), 124);

            view.set(encoder.encode(computeChecksum(view)), 148);

            const extractor = new TarExtractor(buffer);
            expect(() => extractor.extract()).toThrow("File content extends beyond buffer");
        });
    });

    describe("Edge cases", () => {
        it("keeps only the last occurrence of a repeated member", () => {
            const firstHeader = createTarHeader("same.txt", 5, 0x30);
            const firstData = new Uint8Array(512);
            firstData.set(new TextEncoder().encode("first"));
            const secondHeader = createTarHeader("same.txt", 6, 0x30);
            const secondData = new Uint8Array(512);
            secondData.set(new TextEncoder().encode("second"));
            const extractor = new TarExtractor(
                buildTarBuffer([firstHeader, firstData, secondHeader, secondData])
            );

            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(new TextDecoder().decode(files[0].content)).toBe("second");
            expect(extractor.scan().get("same.txt")).toEqual([1536, 6]);
        });

        it("rejects parent-directory paths", () => {
            const header = createTarHeader("../outside.txt", 0, 0x30);
            const extractor = new TarExtractor(buildTarBuffer([header]));

            expect(() => extractor.extract()).toThrowError(UnsafeTarMemberError);
        });

        it("rejects symbolic links", () => {
            const header = createTarHeader("link.txt", 0, 0x32);
            const extractor = new TarExtractor(buildTarBuffer([header]));

            expect(() => extractor.extract()).toThrowError(UnsafeTarMemberError);
        });

        it("should handle files with null bytes in name", () => {
            const tarBuffer = createMockTarFile("test.txt", "content");
            const extractor = new TarExtractor(tarBuffer);
            const files = extractor.extract();

            expect(files[0].relpath).not.toContain("\0");
            expect(files[0].relpath).toBe("test.txt");
        });

        it("should align file data to 512 byte blocks", () => {
            const tarBuffer = createMockTarFile("small.txt", "A");
            const extractor = new TarExtractor(tarBuffer);
            const files = extractor.extract();
            expect(files[0].content!.length).toBe(1);
        });

        it("should handle empty file names gracefully", () => {
            const tarBuffer = createMockTarFile("", "content");
            const extractor = new TarExtractor(tarBuffer);
            const files = extractor.extract();
            expect(files[0].relpath).toBe("");
        });
    });

    describe("Directory handling", () => {
        it("should recognize directory type", () => {
            // typeflag '5' = 0x35
            const header = createTarHeader("testdir/", 0, 0x35);
            const buffer = buildTarBuffer([header]);

            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files[0].type).toBe("directory");
            expect(files[0].relpath).toBe("testdir/");
        });
    });

    describe("USTAR prefix handling", () => {
        it("should combine prefix and name in extract()", () => {
            const header = createTarHeader("file.txt", 5, 0x30, "some/long/prefix");
            const dataBlock = new Uint8Array(512);
            dataBlock.set(new TextEncoder().encode("hello"), 0);

            const buffer = buildTarBuffer([header, dataBlock]);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe("some/long/prefix/file.txt");
            expect(new TextDecoder().decode(files[0].content!)).toBe("hello");
        });

        it("should combine prefix and name in scan()", () => {
            const header = createTarHeader("file.txt", 5, 0x30, "some/long/prefix");
            const dataBlock = new Uint8Array(512);
            dataBlock.set(new TextEncoder().encode("hello"), 0);

            const buffer = buildTarBuffer([header, dataBlock]);
            const extractor = new TarExtractor(buffer);
            const scanResults = extractor.scan();

            expect(scanResults.size).toBe(1);
            expect(scanResults.has("some/long/prefix/file.txt")).toBe(true);
        });

        it("should not add prefix separator when prefix is empty", () => {
            const header = createTarHeader("plain.txt", 3, 0x30);
            const dataBlock = new Uint8Array(512);
            dataBlock.set(new TextEncoder().encode("abc"), 0);

            const buffer = buildTarBuffer([header, dataBlock]);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files[0].relpath).toBe("plain.txt");
        });
    });

    describe("PAX extended header handling", () => {
        function createPaxEntry(paxPath: string, fileContent: string): Uint8Array[] {
            const encoder = new TextEncoder();

            // Build PAX data: "NN path=<paxPath>\n"
            const keyValue = `path=${paxPath}`;
            const recordContent = ` ${keyValue}\n`;
            // Record format: "<length> <key>=<value>\n"
            // Length includes the length field itself, space, key=value, and newline
            const lenPrefix = (recordContent.length + String(recordContent.length).length).toString();
            const paxRecord = `${lenPrefix}${recordContent}`;
            const paxData = encoder.encode(paxRecord);

            // PAX header (typeflag 'x' = 0x78)
            const paxHeader = createTarHeader("PaxHeader/file", paxData.length, 0x78);
            const paxDataBlock = new Uint8Array(align512(paxData.length));
            paxDataBlock.set(paxData, 0);

            // Actual file header
            const contentBytes = encoder.encode(fileContent);
            const fileHeader = createTarHeader("short.txt", contentBytes.length, 0x30);
            const fileDataBlock = new Uint8Array(align512(contentBytes.length));
            fileDataBlock.set(contentBytes, 0);

            return [paxHeader, paxDataBlock, fileHeader, fileDataBlock];
        }

        it("should use PAX path in extract()", () => {
            const blocks = createPaxEntry("very/long/pax/path/to/file.txt", "data");
            const buffer = buildTarBuffer(blocks);

            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe("very/long/pax/path/to/file.txt");
            expect(new TextDecoder().decode(files[0].content!)).toBe("data");
        });

        it("should use PAX path in scan()", () => {
            const blocks = createPaxEntry("very/long/pax/path/to/file.txt", "data");
            const buffer = buildTarBuffer(blocks);

            const extractor = new TarExtractor(buffer);
            const scanResults = extractor.scan();

            expect(scanResults.size).toBe(1);
            expect(scanResults.has("very/long/pax/path/to/file.txt")).toBe(true);
        });

        it("should override USTAR prefix when PAX path is present", () => {
            const encoder = new TextEncoder();
            const paxPath = "pax/overridden/path.txt";
            const keyValue = `path=${paxPath}`;
            const recordContent = ` ${keyValue}\n`;
            const lenPrefix = (recordContent.length + String(recordContent.length).length).toString();
            const paxRecord = `${lenPrefix}${recordContent}`;
            const paxData = encoder.encode(paxRecord);

            const paxHeader = createTarHeader("PaxHeader/file", paxData.length, 0x78);
            const paxDataBlock = new Uint8Array(align512(paxData.length));
            paxDataBlock.set(paxData, 0);

            // File with a USTAR prefix that should be overridden by PAX
            const fileHeader = createTarHeader("name.txt", 4, 0x30, "prefix");
            const fileDataBlock = new Uint8Array(512);
            fileDataBlock.set(encoder.encode("test"), 0);

            const buffer = buildTarBuffer([paxHeader, paxDataBlock, fileHeader, fileDataBlock]);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files[0].relpath).toBe("pax/overridden/path.txt");
        });

        it("should skip PAX global headers (typeflag g)", () => {
            const encoder = new TextEncoder();
            const globalData = encoder.encode("20 comment=global\n");

            const globalHeader = createTarHeader("GlobalHead", globalData.length, 0x67);
            const globalDataBlock = new Uint8Array(align512(globalData.length));
            globalDataBlock.set(globalData, 0);

            const fileHeader = createTarHeader("actual.txt", 4, 0x30);
            const fileDataBlock = new Uint8Array(512);
            fileDataBlock.set(encoder.encode("data"), 0);

            const buffer = buildTarBuffer([globalHeader, globalDataBlock, fileHeader, fileDataBlock]);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe("actual.txt");
        });
    });

    describe("GNU LongName handling", () => {
        function createGnuLongNameEntry(longName: string, fileContent: string): Uint8Array[] {
            const encoder = new TextEncoder();
            const longNameBytes = encoder.encode(longName);

            // GNU LongName header (typeflag 'L' = 0x4c)
            const longNameHeader = createTarHeader("././@LongLink", longNameBytes.length, 0x4c);
            const longNameDataBlock = new Uint8Array(align512(longNameBytes.length));
            longNameDataBlock.set(longNameBytes, 0);

            // Actual file header (with truncated name)
            const contentBytes = encoder.encode(fileContent);
            const truncatedName = longName.slice(0, 99);
            const fileHeader = createTarHeader(truncatedName, contentBytes.length, 0x30);
            const fileDataBlock = new Uint8Array(align512(contentBytes.length));
            fileDataBlock.set(contentBytes, 0);

            return [longNameHeader, longNameDataBlock, fileHeader, fileDataBlock];
        }

        it("should use GNU LongName in extract()", () => {
            const longName = "this/is/a/very/long/path/that/exceeds/one/hundred/characters/and/needs/gnu/longname/header/support/file.txt";
            const blocks = createGnuLongNameEntry(longName, "content");
            const buffer = buildTarBuffer(blocks);

            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe(longName);
            expect(new TextDecoder().decode(files[0].content!)).toBe("content");
        });

        it("should use GNU LongName in scan()", () => {
            const longName = "this/is/a/very/long/path/that/exceeds/one/hundred/characters/and/needs/gnu/longname/header/support/file.txt";
            const blocks = createGnuLongNameEntry(longName, "content");
            const buffer = buildTarBuffer(blocks);

            const extractor = new TarExtractor(buffer);
            const scanResults = extractor.scan();

            expect(scanResults.size).toBe(1);
            expect(scanResults.has(longName)).toBe(true);
        });

        it("should prefer PAX path over GNU LongName", () => {
            const encoder = new TextEncoder();

            // PAX header first
            const paxPath = "pax/takes/priority.txt";
            const keyValue = `path=${paxPath}`;
            const recordContent = ` ${keyValue}\n`;
            const lenPrefix = (recordContent.length + String(recordContent.length).length).toString();
            const paxRecord = `${lenPrefix}${recordContent}`;
            const paxData = encoder.encode(paxRecord);

            const paxHeader = createTarHeader("PaxHeader/file", paxData.length, 0x78);
            const paxDataBlock = new Uint8Array(align512(paxData.length));
            paxDataBlock.set(paxData, 0);

            // Then GNU LongName
            const longName = "gnu/longname/path.txt";
            const longNameBytes = encoder.encode(longName);
            const longNameHeader = createTarHeader("././@LongLink", longNameBytes.length, 0x4c);
            const longNameDataBlock = new Uint8Array(align512(longNameBytes.length));
            longNameDataBlock.set(longNameBytes, 0);

            // Actual file
            const fileHeader = createTarHeader("short.txt", 4, 0x30);
            const fileDataBlock = new Uint8Array(512);
            fileDataBlock.set(encoder.encode("data"), 0);

            const buffer = buildTarBuffer([paxHeader, paxDataBlock, longNameHeader, longNameDataBlock, fileHeader, fileDataBlock]);
            const extractor = new TarExtractor(buffer);
            const files = extractor.extract();

            expect(files).toHaveLength(1);
            expect(files[0].relpath).toBe("pax/takes/priority.txt");
        });
    });

    describe("scan() filtering", () => {
        it("should only include regular files in scan results", () => {
            // Regular file (typeflag '0' = 0x30)
            const fileHeader = createTarHeader("file.txt", 4, 0x30);
            const fileData = new Uint8Array(512);
            fileData.set(new TextEncoder().encode("data"), 0);

            // Directory (typeflag '5' = 0x35)
            const dirHeader = createTarHeader("mydir/", 0, 0x35);

            const buffer = buildTarBuffer([fileHeader, fileData, dirHeader]);
            const extractor = new TarExtractor(buffer);
            const scanResults = extractor.scan();

            expect(scanResults.size).toBe(1);
            expect(scanResults.has("file.txt")).toBe(true);
            expect(scanResults.has("mydir/")).toBe(false);
        });
    });
});
