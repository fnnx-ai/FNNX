import asyncio
from concurrent.futures import Future
from concurrent.futures._base import Executor
from copy import copy as shallow_copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class AsyncDelayedResponse:
    task: asyncio.Task[Any]
    index: int


@dataclass
class DelayedResponse:
    task: Future[Any]
    index: int


@dataclass
class DagComponent:
    inputs: list[str]
    outputs: list[str]
    extra_dynattrs: dict[str, str]


async def dag_compute_async(
    inputs: dict[str, Any],
    components: Sequence[DagComponent],
    compute_fn: Callable[..., Any],
    as_val: Callable[[Any], Any],
    components_passthrough: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, AsyncDelayedResponse | Any] = shallow_copy(inputs)
    for component in components:
        tasks = []
        keys = []
        for k in component.inputs:
            if isinstance(state[k], AsyncDelayedResponse):
                tasks.append(state[k].task)
                keys.append(k)

        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            state[keys[i]] = as_val(result)[state[keys[i]].index]

        component_inputs = [state[k] for k in component.inputs]

        component_passthrough = components_passthrough
        if "dynamic_attributes" in components_passthrough:
            component_passthrough = components_passthrough | {
                "dynamic_attributes": components_passthrough["dynamic_attributes"]
                | component.extra_dynattrs
            }
        component_compute_output = asyncio.create_task(
            compute_fn(
                component,
                component_inputs,
                **component_passthrough,
            )
        )

        for i, output_key in enumerate(component.outputs):
            state[output_key] = AsyncDelayedResponse(
                task=component_compute_output,
                index=i,
            )

    # TODO better parallelization
    for k, v in state.items():
        if isinstance(v, AsyncDelayedResponse):
            state[k] = as_val(await v.task)[v.index]

    return state


def dag_compute(
    inputs: dict[str, Any],
    components: Sequence[DagComponent],
    graph_executor: Executor,
    compute_fn: Callable[..., Any],
    as_val: Callable[[Any], Any],
    components_passthrough: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, DelayedResponse | Any] = shallow_copy(inputs)
    for component in components:
        component_inputs = []
        for k in component.inputs:
            if isinstance(state[k], DelayedResponse):
                index = state[k].index
                result = as_val(state[k].task.result())[index]
                state[k] = result
            component_inputs.append(state[k])

        component_passthrough = components_passthrough
        if "dynamic_attributes" in components_passthrough:
            component_passthrough = components_passthrough | {
                "dynamic_attributes": components_passthrough["dynamic_attributes"]
                | component.extra_dynattrs
            }

        component_output = graph_executor.submit(
            compute_fn,
            component,
            component_inputs,
            graph_executor=graph_executor,
            **component_passthrough,
        )
        for i, output_key in enumerate(component.outputs):
            state[output_key] = DelayedResponse(
                task=component_output,
                index=i,
            )

    for k, v in state.items():
        if isinstance(v, DelayedResponse):
            state[k] = as_val(v.task.result())[v.index]

    return state
