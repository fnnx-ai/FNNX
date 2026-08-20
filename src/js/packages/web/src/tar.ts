import { interfaces } from "@fnnx-ai/common";

export class TarExtractor {
    private offset = 0;
    private view: DataView;
    private pendingPaxPath: string | null = null;
    private pendingLongName: string | null = null;

    constructor(private buffer: ArrayBuffer) {
        this.view = new DataView(buffer);
    }

    private readString(length: number): string {
        const bytes = new Uint8Array(this.buffer, this.offset, length);
        const nullIndex = bytes.indexOf(0);
        const strLength = nullIndex !== -1 ? nullIndex : length;
        const str = new TextDecoder('ascii').decode(bytes.slice(0, strLength));
        this.offset += length;
        return str;
    }

    private readAsciiAt(offset: number, length: number): string {
        const bytes = new Uint8Array(this.buffer, offset, length);
        const nullIndex = bytes.indexOf(0);
        const strLength = nullIndex !== -1 ? nullIndex : length;
        return new TextDecoder('ascii').decode(bytes.slice(0, strLength));
    }

    private readOctal(length: number): number {
        const str = this.readString(length).trim();
        return str ? parseInt(str, 8) : 0;
    }

    private readOctalAt(offset: number, length: number): number {
        const raw = new TextDecoder('ascii').decode(new Uint8Array(this.buffer, offset, length));
        const trimmed = raw.replace(/\0/g, '').trim();
        return trimmed ? parseInt(trimmed, 8) : 0;
    }

    private align512(size: number): number {
        const remainder = size % 512;
        return remainder ? size + (512 - remainder) : size;
    }

    private calculateChecksum(headerStart: number): number {
        let sum = 0;
        for (let i = 0; i < 512; i++) {
            if (i >= 148 && i < 156) {
                sum += 32;
            } else {
                sum += this.view.getUint8(headerStart + i);
            }
        }
        return sum;
    }

    private parsePaxData(dataStart: number, size: number): void {
        const view = new Uint8Array(this.buffer, dataStart, size);
        let i = 0;
        while (i < view.length) {
            const recStartDigits = i;
            let lenStr = '';
            while (i < view.length && view[i] !== 0x20) {
                lenStr += String.fromCharCode(view[i++]);
            }
            if (i >= view.length) break;
            const recLen = parseInt(lenStr, 10);
            i++;
            const recTotalEnd = recStartDigits + recLen;
            const recContentEnd = recTotalEnd - 1;
            if (recTotalEnd > view.length || recContentEnd < i) break;
            const line = new TextDecoder('utf-8').decode(view.slice(i, recContentEnd));
            const eq = line.indexOf('=');
            if (eq !== -1) {
                const key = line.slice(0, eq);
                const val = line.slice(eq + 1);
                if (key === 'path') {
                    this.pendingPaxPath = val;
                }
            }
            i = recTotalEnd;
        }
    }

    private parseHeader(): interfaces.TarFileContent | null {
        while (true) {
            if (this.offset >= this.buffer.byteLength) {
                return null;
            }

            let isZeroBlock = true;
            for (let i = 0; i < 512; i++) {
                if (this.view.getUint8(this.offset + i) !== 0) {
                    isZeroBlock = false;
                    break;
                }
            }
            if (isZeroBlock) {
                return null;
            }

            const originalOffset = this.offset;
            const name = this.readString(100);
            this.readOctal(8);
            this.readOctal(8);
            this.readOctal(8);
            const size = this.readOctal(12);
            this.readOctal(12);
            const checksum = this.readOctal(8);
            const type = this.readString(1);
            this.readString(100);

            const prefix = this.readAsciiAt(originalOffset + 345, 155);
            const typeflag = this.view.getUint8(originalOffset + 156);

            const calculatedChecksum = this.calculateChecksum(originalOffset);
            if (checksum !== calculatedChecksum) {
                throw new Error(`Invalid header`);
            }

            if (size < 0) {
                throw new Error('Invalid file size in tar header');
            }

            const dataStart = originalOffset + 512;
            const dataSpan = this.align512(size);

            // PAX extended header (typeflag 'x' = 0x78)
            if (typeflag === 0x78) {
                this.parsePaxData(dataStart, size);
                this.offset = dataStart + dataSpan;
                continue;
            }

            // PAX global header (typeflag 'g' = 0x67) - skip
            if (typeflag === 0x67) {
                this.offset = dataStart + dataSpan;
                continue;
            }

            // GNU LongName header (typeflag 'L' = 0x4c)
            if (typeflag === 0x4c) {
                const bytes = new Uint8Array(this.buffer, dataStart, size);
                const nul = bytes.indexOf(0);
                this.pendingLongName = new TextDecoder('utf-8').decode(
                    bytes.slice(0, nul === -1 ? bytes.length : nul)
                );
                this.offset = dataStart + dataSpan;
                continue;
            }

            if (dataStart + size > this.buffer.byteLength) {
                throw new Error('File content extends beyond buffer');
            }

            let fullPath: string;
            if (this.pendingPaxPath) {
                fullPath = this.pendingPaxPath;
                this.pendingPaxPath = null;
            } else if (this.pendingLongName) {
                fullPath = this.pendingLongName;
                this.pendingLongName = null;
            } else if (prefix) {
                fullPath = `${prefix}/${name}`;
            } else {
                fullPath = name;
            }

            this.offset = dataStart;
            const data = new Uint8Array(this.buffer, this.offset, size);
            this.offset += dataSpan;

            return {
                relpath: fullPath.replace(/\0/g, ''),
                content: data,
                type: type === '5' ? 'directory' : 'file',
                fsPath: null
            };
        }
    }

    extract(): interfaces.TarFileContent[] {
        const files: interfaces.TarFileContent[] = [];
        this.pendingPaxPath = null;
        this.pendingLongName = null;
        let file: interfaces.TarFileContent | null;
        while ((file = this.parseHeader()) !== null) {
            files.push(file);
        }
        return files;
    }

    scan(): Map<string, [number, number]> {
        const results = new Map<string, [number, number]>();
        let scanOffset = 0;

        let pendingPaxPath: string | null = null;
        let pendingLongName: string | null = null;

        const isZeroBlock = (o: number) => {
            for (let i = 0; i < 512; i++) {
                if (this.view.getUint8(o + i) !== 0) return false;
            }
            return true;
        };

        while (scanOffset + 512 <= this.buffer.byteLength) {
            if (isZeroBlock(scanOffset)) break;

            const name = this.readAsciiAt(scanOffset + 0, 100);
            const size = this.readOctalAt(scanOffset + 124, 12);
            const typeflag = this.view.getUint8(scanOffset + 156);
            const prefix = this.readAsciiAt(scanOffset + 345, 155);

            const dataStart = scanOffset + 512;
            const dataSpan = this.align512(size);

            // PAX extended header (typeflag 'x' = 0x78)
            if (typeflag === 0x78) {
                const view = new Uint8Array(this.buffer, dataStart, size);
                let i = 0;
                while (i < view.length) {
                    const recStartDigits = i;
                    let lenStr = '';
                    while (i < view.length && view[i] !== 0x20) {
                        lenStr += String.fromCharCode(view[i++]);
                    }
                    if (i >= view.length) break;
                    const recLen = parseInt(lenStr, 10);
                    i++;
                    const recTotalEnd = recStartDigits + recLen;
                    const recContentEnd = recTotalEnd - 1;
                    if (recTotalEnd > view.length || recContentEnd < i) break;
                    const line = new TextDecoder('utf-8').decode(view.slice(i, recContentEnd));
                    const eq = line.indexOf('=');
                    if (eq !== -1) {
                        const key = line.slice(0, eq);
                        const val = line.slice(eq + 1);
                        if (key === 'path') pendingPaxPath = val;
                    }
                    i = recTotalEnd;
                }
                scanOffset = dataStart + dataSpan;
                continue;
            }

            // PAX global header (typeflag 'g' = 0x67) - skip
            if (typeflag === 0x67) {
                scanOffset = dataStart + dataSpan;
                continue;
            }

            // GNU LongName header (typeflag 'L' = 0x4c)
            if (typeflag === 0x4c) {
                const bytes = new Uint8Array(this.buffer, dataStart, size);
                const nul = bytes.indexOf(0);
                pendingLongName = new TextDecoder('utf-8').decode(
                    bytes.slice(0, nul === -1 ? bytes.length : nul)
                );
                scanOffset = dataStart + dataSpan;
                continue;
            }

            let fullPath: string;
            if (pendingPaxPath) {
                fullPath = pendingPaxPath;
                pendingPaxPath = null;
            } else if (pendingLongName) {
                fullPath = pendingLongName;
                pendingLongName = null;
            } else if (prefix) {
                fullPath = `${prefix}/${name}`;
            } else {
                fullPath = name;
            }

            const isRegular = typeflag === 0x30 || typeflag === 0x00;
            if (isRegular) {
                results.set(fullPath.replace(/\0/g, ''), [dataStart, size]);
            }

            scanOffset = dataStart + dataSpan;
        }

        return results;
    }
}
