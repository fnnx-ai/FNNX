import { describe, expect, it } from "vitest";
import { ArrayDType, Model, NDArray, NDContainer } from "../../src/index";

describe("web package exports", () => {
    it("exports the input construction types", () => {
        const array = new NDArray([1], [1], ArrayDType.Float32);
        const container = new NDContainer(["value"], "string");

        expect(array).toBeInstanceOf(NDArray);
        expect(container).toBeInstanceOf(NDContainer);
        expect(Model).toBeTypeOf("function");
    });
});
