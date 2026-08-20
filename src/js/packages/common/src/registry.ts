import { ConcreteOp } from './ops/base';

export default class Registry {
    private ops: Record<string, ConcreteOp>;

    constructor(ops: Record<string, ConcreteOp>) {
        this.ops = ops;
    }

    public getOp(name: string): ConcreteOp {
        return this.ops[name];
    }
}
