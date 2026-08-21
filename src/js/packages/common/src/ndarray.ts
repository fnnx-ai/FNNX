export enum ArrayDType {
    Float32 = "float32",
    Float64 = "float64",
    Int32 = "int32",
    Int64 = "int64",
    String = "string",
    Bool = "bool",
}

export type ArrayElement = number | bigint | string | boolean;

export class NDArray {
    public readonly shape: number[];
    public readonly dtype: ArrayDType;
    private data: ArrayElement[];

    constructor(shape: number[], data: ArrayLike<unknown>, dtype: ArrayDType) {
        this.shape = [...shape];
        this.dtype = dtype;

        if (!Object.values(ArrayDType).includes(dtype)) {
            throw new Error(`Unsupported dtype: ${dtype}`);
        }

        if (shape.some((extent) => !Number.isSafeInteger(extent) || extent < 0)) {
            throw new Error(`Invalid shape: ${JSON.stringify(shape)}`);
        }

        const size = shape.reduce((product, extent) => product * extent, 1);
        if (data.length !== size) {
            throw new Error(
                `Data length (${data.length}) does not match the size of the shape (${size})`
            );
        }
        this.data = Array.from(data, (item) => this.castToDType(item));
    }

    private castToDType(value: unknown): ArrayElement {
        switch (this.dtype) {
            case ArrayDType.Float32:
            case ArrayDType.Float64:
                return castFloat(value);
            case ArrayDType.Int32:
                return castInt32(value);
            case ArrayDType.Int64:
                return castInt64(value);
            case ArrayDType.String:
                return String(value);
            case ArrayDType.Bool:
                return castBoolean(value);
        }
    }

    private computeIndex(indices: number[]): number {
        if (indices.length !== this.shape.length) {
            throw new Error(`Expected ${this.shape.length} indices, got ${indices.length}`);
        }
        let index = 0;
        let stride = 1;
        for (let dimension = this.shape.length - 1; dimension >= 0; dimension--) {
            if (indices[dimension] < 0 || indices[dimension] >= this.shape[dimension]) {
                throw new Error(
                    `Index ${indices[dimension]} out of bounds for dimension ${dimension}`
                );
            }
            index += indices[dimension] * stride;
            stride *= this.shape[dimension];
        }
        return index;
    }

    get(indices: number[]): ArrayElement {
        return this.data[this.computeIndex(indices)];
    }

    set(indices: number[], value: unknown): void {
        this.data[this.computeIndex(indices)] = this.castToDType(value);
    }

    getShape(): number[] {
        return [...this.shape];
    }

    getDType(): ArrayDType {
        return this.dtype;
    }

    toArray(): ArrayElement[] {
        return [...this.data];
    }

    astype(newDType: ArrayDType): NDArray {
        return new NDArray(this.shape, this.data, newDType);
    }
}

function castFloat(value: unknown): number {
    if (typeof value === "number") {
        return value;
    }
    if (typeof value === "bigint") {
        return Number(value);
    }
    if (typeof value === "string") {
        if (value === "NaN") {
            return Number.NaN;
        }
        if (value === "Infinity") {
            return Number.POSITIVE_INFINITY;
        }
        if (value === "-Infinity") {
            return Number.NEGATIVE_INFINITY;
        }
        const converted = Number(value);
        if (value.trim() !== "" && Number.isFinite(converted)) {
            return converted;
        }
    }
    throw new TypeError(`Cannot cast ${String(value)} to a float`);
}

function castInt32(value: unknown): number {
    const converted = Number(value);
    if (!Number.isInteger(converted) || converted < -2_147_483_648 || converted > 2_147_483_647) {
        throw new TypeError(`Cannot cast ${String(value)} to int32`);
    }
    return converted;
}

function castInt64(value: unknown): bigint {
    if (typeof value === "number" && (!Number.isSafeInteger(value) || !Number.isFinite(value))) {
        throw new TypeError(
            "Cannot cast an unsafe number to int64; use bigint or an integer string"
        );
    }
    let converted: bigint;
    try {
        converted = BigInt(value as string | number | bigint | boolean);
    } catch {
        throw new TypeError(`Cannot cast ${String(value)} to int64`);
    }
    if (converted < -9_223_372_036_854_775_808n || converted > 9_223_372_036_854_775_807n) {
        throw new RangeError(`int64 value is out of range: ${converted}`);
    }
    return converted;
}

function castBoolean(value: unknown): boolean {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "string") {
        const normalized = value.toLowerCase();
        if (normalized === "true") {
            return true;
        }
        if (normalized === "false") {
            return false;
        }
    }
    if (typeof value === "number" && Number.isFinite(value)) {
        return value !== 0;
    }
    if (typeof value === "bigint") {
        return value !== 0n;
    }
    throw new TypeError(`Cannot cast ${String(value)} to a boolean`);
}
