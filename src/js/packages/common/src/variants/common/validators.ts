import { OpIO } from "../../interfaces.js";
import { NDArray } from "../../ndarray.js";
import { NDContainer } from "../../ndcontainer.js";

export function validateInputs(inputs: unknown[], inputSpecs: OpIO[]): void {
    inputSpecs.forEach((spec, index) => {
        const value = inputs[index];
        if (spec.dtype.startsWith("Array[")) {
            if (!(value instanceof NDArray)) {
                throw new TypeError(
                    `Expected input dtype ${spec.dtype}, got ${describeValue(value)}`
                );
            }
            if (value.getDType() !== spec.dtype.slice(6, -1)) {
                throw new TypeError(
                    `Expected input dtype ${spec.dtype}, got Array[${value.getDType()}]`
                );
            }
            validateShape(value.getShape(), spec.shape);
        } else if (spec.dtype.startsWith("NDContainer[")) {
            if (!(value instanceof NDContainer)) {
                throw new TypeError(
                    `Expected input dtype ${spec.dtype}, got ${describeValue(value)}`
                );
            }
            if (value.dtype !== spec.dtype.slice(12, -1)) {
                throw new TypeError(
                    `Expected input dtype ${spec.dtype}, got NDContainer[${value.dtype}]`
                );
            }
            validateShape(value.shape, spec.shape);
        } else {
            throw new TypeError(`Unknown dtype ${spec.dtype}`);
        }
    });
}

function validateShape(actual: readonly number[], declared: (number | string)[]): void {
    const conforms =
        declared.length === actual.length &&
        declared.every((extent, axis) => typeof extent === "string" || extent === actual[axis]);
    if (!conforms) {
        throw new Error(`Expected input shape [${declared}], got [${actual}]`);
    }
}

function describeValue(value: unknown): string {
    if (value instanceof NDContainer) {
        return `NDContainer[${value.dtype}]`;
    }
    if (value === null) {
        return "null";
    }
    if (typeof value !== "object") {
        return typeof value;
    }
    return value.constructor?.name ?? "object";
}
