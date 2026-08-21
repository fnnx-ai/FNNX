export enum ArrayDType {
    Float32 = "float32",
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
                return Number.parseFloat(String(value));
            case ArrayDType.Int32:
                return Number.parseInt(String(value), 10);
            case ArrayDType.Int64:
                return BigInt(value as string | number | bigint | boolean);
            case ArrayDType.String:
                return String(value);
            case ArrayDType.Bool:
                return Boolean(value);
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
