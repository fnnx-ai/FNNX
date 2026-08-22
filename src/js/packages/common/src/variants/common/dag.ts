export interface DagComponent {
    inputs: string[];
    outputs: string[];
    extra_dynattrs: Record<string, string>;
}

interface DelayedResponse {
    promise: Promise<unknown>;
    index: number;
}

type ComputeFn = (
    component: DagComponent,
    inputs: unknown[],
    passthrough: Record<string, unknown>
) => Promise<unknown>;
type AsValFn = (result: unknown) => unknown[];

export async function dagCompute(
    inputs: Record<string, unknown>,
    components: DagComponent[],
    computeFn: ComputeFn,
    asVal: AsValFn,
    componentsPassthrough: Record<string, unknown>
): Promise<Record<string, unknown>> {
    const state = new Map<string, unknown>();
    const componentPromises: Promise<unknown>[] = [];
    Object.entries(inputs).forEach(([key, value]) => {
        state.set(key, value);
    });

    for (const component of components) {
        const inputPromises: Promise<unknown>[] = [];
        const inputKeys: string[] = [];

        for (const key of component.inputs) {
            if (!state.has(key)) {
                throw new Error(`Pipeline input \`${key}\` is unbound at compute time`);
            }
            const stateValue = state.get(key);
            if (isDelayedResponse(stateValue)) {
                inputPromises.push(stateValue.promise);
                inputKeys.push(key);
            }
        }

        if (inputPromises.length > 0) {
            const results = await Promise.all(inputPromises);
            results.forEach((result, index) => {
                const key = inputKeys[index];
                const stateValue = state.get(key) as DelayedResponse;
                state.set(key, asVal(result)[stateValue.index]);
            });
        }

        const componentInputs = component.inputs.map((key) => {
            const value = state.get(key);
            if (value === undefined) {
                throw new Error(`Pipeline input \`${key}\` is unbound at compute time`);
            }
            return value;
        });

        const passthroughCopy = { ...componentsPassthrough };
        if ("dynamic_attributes" in passthroughCopy) {
            passthroughCopy.dynamic_attributes = {
                ...(passthroughCopy.dynamic_attributes as Record<string, string>),
                ...component.extra_dynattrs,
            };
        }

        const componentPromise = computeFn(component, componentInputs, passthroughCopy);
        componentPromises.push(componentPromise);

        component.outputs.forEach((outputKey, index) => {
            state.set(outputKey, {
                promise: componentPromise,
                index,
            });
        });
    }

    await Promise.all(componentPromises);

    const finalState: Record<string, unknown> = {};
    for (const [key, value] of state.entries()) {
        if (isDelayedResponse(value)) {
            const result = await value.promise;
            finalState[key] = asVal(result)[value.index];
        } else {
            finalState[key] = value;
        }
    }

    return finalState;
}

function isDelayedResponse(value: unknown): value is DelayedResponse {
    return typeof value === "object" && value !== null && "promise" in value;
}
