import { DtypesManager } from "./dtypes.js";

export class NDContainer {
    public readonly shape: number[];
    public readonly data: unknown[];
    private readonly innerDtype: string;

    constructor(data: unknown, dtype: string, dtypesManager: DtypesManager | null = null) {
        if (dtype.startsWith("Array[")) {
            throw new Error("NDContainer does not support Array dtype");
        }
        if (dtype.startsWith("NDContainer[")) {
            dtype = dtype.slice(12, -1);
        }

        this.data = Array.isArray(data) ? structuredClone(data) : [structuredClone(data)];
        dtypesManager?.validateDtype(dtype, this.data);
        this.innerDtype = dtype;
        this.shape = this.computeShape(this.data);
    }

    get dtype(): string {
        return this.innerDtype;
    }

    private computeShape(data: unknown): number[] {
        if (!Array.isArray(data) || data.length === 0) {
            return [];
        }
        return [data.length, ...this.computeShape(data[0])];
    }

    toArray(): unknown[] {
        return structuredClone(this.data);
    }
}
