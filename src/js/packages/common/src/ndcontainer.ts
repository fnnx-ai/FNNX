import { DtypesManager } from "./dtypes.js";

export class NDContainer {
    public readonly shape: number[];
    public readonly data: unknown[];
    private readonly innerDtype: string;

    constructor(
        data: unknown,
        dtype: string,
        dtypesManager: DtypesManager | null = null,
        rank: number | null = null
    ) {
        if (dtype.startsWith("Array[")) {
            throw new Error("NDContainer does not support Array dtype");
        }
        if (dtype.startsWith("NDContainer[")) {
            dtype = dtype.slice(12, -1);
            if (dtype.startsWith("Array[")) {
                throw new Error(
                    `NDContainer inner dtype \`${dtype}\` must not be an Array[...] form`
                );
            }
        }

        const wrapped = !Array.isArray(data);
        this.data = wrapped ? [structuredClone(data)] : (structuredClone(data) as unknown[]);
        // A scalar is wrapped in a one-element list, which always adds one level of nesting.
        const depth = rank === null ? null : wrapped ? Math.max(rank, 1) : rank;
        dtypesManager?.validateDtype(dtype, this.data, depth);
        this.innerDtype = dtype;
        this.shape = this.computeShape(this.data, depth);
    }

    get dtype(): string {
        return this.innerDtype;
    }

    private computeShape(data: unknown, depth: number | null): number[] {
        if (depth === 0 || !Array.isArray(data) || data.length === 0) {
            return [];
        }
        return [data.length, ...this.computeShape(data[0], depth === null ? null : depth - 1)];
    }

    toArray(): unknown[] {
        return structuredClone(this.data);
    }
}
