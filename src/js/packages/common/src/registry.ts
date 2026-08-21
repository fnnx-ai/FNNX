import { ConcreteOp } from "./ops/base.js";

export default class Registry {
    constructor(private readonly ops: Record<string, ConcreteOp>) {}

    getOp(name: string): ConcreteOp | undefined {
        return this.ops[name];
    }
}
