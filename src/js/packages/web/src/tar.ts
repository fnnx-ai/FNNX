import { InvalidTarError, UnsafeTarMemberError, assertSafeArtifactPath } from "@fnnx-ai/common";

export interface TarEntry {
    relpath: string;
    type: "file" | "directory";
    content: Uint8Array;
}

interface ParsedTarEntry {
    relpath: string;
    type: "file" | "directory";
    dataOffset: number;
    size: number;
}

const FORBIDDEN_TYPEFLAGS = new Set([0x31, 0x32, 0x33, 0x34, 0x36]);

export class TarExtractor {
    private readonly view: DataView;

    constructor(private readonly buffer: ArrayBuffer) {
        this.view = new DataView(buffer);
    }

    extract(): TarEntry[] {
        const files = new Map<string, TarEntry>();
        for (const entry of this.iterateEntries()) {
            const content = new Uint8Array(this.buffer, entry.dataOffset, entry.size).slice();
            files.delete(entry.relpath);
            files.set(entry.relpath, { relpath: entry.relpath, type: entry.type, content });
        }
        return [...files.values()];
    }

    scan(): Map<string, [number, number]> {
        const positions = new Map<string, [number, number]>();
        for (const entry of this.iterateEntries()) {
            positions.delete(entry.relpath);
            if (entry.type === "file") {
                positions.set(entry.relpath, [entry.dataOffset, entry.size]);
            }
        }
        return positions;
    }

    private *iterateEntries(): Generator<ParsedTarEntry> {
        let offset = 0;
        let pendingPaxPath: string | null = null;
        let pendingLongName: string | null = null;

        while (offset < this.buffer.byteLength) {
            if (offset + 512 > this.buffer.byteLength) {
                throw new InvalidTarError("Incomplete tar header");
            }
            if (this.isZeroBlock(offset)) {
                return;
            }

            const storedChecksum = this.readOctal(offset + 148, 8);
            if (storedChecksum !== this.calculateChecksum(offset)) {
                throw new InvalidTarError("Invalid tar header checksum");
            }

            const name = this.readString(offset, 100);
            const size = this.readOctal(offset + 124, 12);
            const typeflag = this.view.getUint8(offset + 156);
            const prefix = this.readString(offset + 345, 155);
            const dataOffset = offset + 512;
            const nextOffset = dataOffset + align512(size);
            if (!Number.isSafeInteger(size) || size < 0 || nextOffset > this.buffer.byteLength) {
                throw new InvalidTarError("File content extends beyond buffer");
            }

            if (typeflag === 0x78) {
                pendingPaxPath = this.readPaxPath(dataOffset, size) ?? pendingPaxPath;
                offset = nextOffset;
                continue;
            }
            if (typeflag === 0x67) {
                offset = nextOffset;
                continue;
            }
            if (typeflag === 0x4c) {
                pendingLongName = this.readUtf8String(dataOffset, size);
                offset = nextOffset;
                continue;
            }

            const relpath =
                pendingPaxPath ?? pendingLongName ?? (prefix ? `${prefix}/${name}` : name);
            pendingPaxPath = null;
            pendingLongName = null;
            assertSafeArtifactPath(relpath);

            if (FORBIDDEN_TYPEFLAGS.has(typeflag)) {
                throw new UnsafeTarMemberError(relpath, "links and device nodes are not permitted");
            }

            const type = classifyType(typeflag);
            if (type) {
                yield { relpath, type, dataOffset, size };
            }
            offset = nextOffset;
        }
    }

    private isZeroBlock(offset: number): boolean {
        for (let index = 0; index < 512; index++) {
            if (this.view.getUint8(offset + index) !== 0) {
                return false;
            }
        }
        return true;
    }

    // ustar name and prefix fields carry UTF-8 bytes, so a short non-ASCII member name has to
    // decode the same way as the PAX and GNU long-name forms.
    private readString(offset: number, length: number): string {
        const bytes = new Uint8Array(this.buffer, offset, length);
        const nullIndex = bytes.indexOf(0);
        return new TextDecoder("utf-8").decode(
            bytes.slice(0, nullIndex === -1 ? length : nullIndex)
        );
    }

    private readUtf8String(offset: number, length: number): string {
        const bytes = new Uint8Array(this.buffer, offset, length);
        const nullIndex = bytes.indexOf(0);
        return new TextDecoder("utf-8", { fatal: true }).decode(
            bytes.slice(0, nullIndex === -1 ? length : nullIndex)
        );
    }

    private readOctal(offset: number, length: number): number {
        const raw = this.readString(offset, length).trim();
        if (!raw) {
            return 0;
        }
        if (!/^[0-7]+$/.test(raw)) {
            return Number.NaN;
        }
        return Number.parseInt(raw, 8);
    }

    private calculateChecksum(headerOffset: number): number {
        let sum = 0;
        for (let index = 0; index < 512; index++) {
            sum += index >= 148 && index < 156 ? 32 : this.view.getUint8(headerOffset + index);
        }
        return sum;
    }

    private readPaxPath(offset: number, size: number): string | null {
        const data = new Uint8Array(this.buffer, offset, size);
        let cursor = 0;
        let path: string | null = null;
        while (cursor < data.length) {
            const space = data.indexOf(0x20, cursor);
            if (space === -1) {
                throw new InvalidTarError("Invalid PAX record");
            }
            const lengthText = new TextDecoder("ascii").decode(data.slice(cursor, space));
            const recordLength = Number.parseInt(lengthText, 10);
            const recordEnd = cursor + recordLength;
            if (
                !Number.isSafeInteger(recordLength) ||
                recordLength <= 0 ||
                recordEnd > data.length
            ) {
                throw new InvalidTarError("Invalid PAX record length");
            }
            const record = new TextDecoder("utf-8", { fatal: true }).decode(
                data.slice(space + 1, recordEnd - 1)
            );
            const equals = record.indexOf("=");
            if (equals !== -1 && record.slice(0, equals) === "path") {
                path = record.slice(equals + 1);
            }
            cursor = recordEnd;
        }
        return path;
    }
}

function align512(size: number): number {
    return Math.ceil(size / 512) * 512;
}

function classifyType(typeflag: number): "file" | "directory" | null {
    if (typeflag === 0 || typeflag === 0x30 || typeflag === 0x37) {
        return "file";
    }
    if (typeflag === 0x35) {
        return "directory";
    }
    return null;
}
