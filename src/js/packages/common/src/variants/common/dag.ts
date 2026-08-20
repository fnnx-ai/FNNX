export interface DagComponent {
    inputs: string[];
    outputs: string[];
    extra_dynattrs: Record<string, string>;
}

interface DelayedResponse<T> {
    promise: Promise<T>;
    index: number;
}

type StateValue<T> = DelayedResponse<T> | any;
type ComputeFn = (component: DagComponent, inputs: any[], passthrough: any) => Promise<any>;
type AsValFn = (result: any) => any[];

export async function dagCompute<T>(
    inputs: Record<string, any>,
    components: DagComponent[],
    computeFn: ComputeFn,
    asVal: AsValFn,
    componentsPassthrough: Record<string, any>
): Promise<Record<string, any>> {
    const state = new Map<string, StateValue<T>>();
    Object.entries(inputs).forEach(([key, value]) => {
        state.set(key, value);
    });

    for (const component of components) {
        const inputPromises: Promise<any>[] = [];
        const inputKeys: string[] = [];

        for (const key of component.inputs) {
            const stateValue = state.get(key);
            if (stateValue && 'promise' in stateValue) {
                inputPromises.push(stateValue.promise);
                inputKeys.push(key);
            }
        }

        // Await any pending input promises
        if (inputPromises.length > 0) {
            const results = await Promise.all(inputPromises);
            results.forEach((result, idx) => {
                const key = inputKeys[idx];
                const stateValue = state.get(key) as DelayedResponse<T>;
                state.set(key, asVal(result)[stateValue.index]);
            });
        }


        const componentInputs = component.inputs.map(key => state.get(key));

        const passthroughCopy = { ...componentsPassthrough };
        if ('dynamic_attributes' in passthroughCopy) {
            passthroughCopy.dynamic_attributes = {
                ...passthroughCopy.dynamic_attributes,
                ...component.extra_dynattrs
            };
        }

        const componentPromise = computeFn(component, componentInputs, passthroughCopy);


        component.outputs.forEach((outputKey, index) => {
            state.set(outputKey, {
                promise: componentPromise,
                index
            });
        });
    }

    const finalState: Record<string, any> = {};
    for (const [key, value] of state.entries()) {
        if (value && 'promise' in value) {
            const result = await value.promise;
            finalState[key] = asVal(result)[value.index];
        } else {
            finalState[key] = value;
        }
    }

    return finalState;
}
