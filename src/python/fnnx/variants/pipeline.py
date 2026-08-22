from dataclasses import dataclass
from typing import Any

from fnnx.node_instance import OpInstance
from fnnx.variants._base import BaseVariant
from fnnx.variants._common.dag import DagComponent, dag_compute, dag_compute_async
from fnnx.variants._common.validators import validate_inputs


@dataclass
class PipelineNodeInstance(OpInstance, DagComponent):
    pass


def validate_pipeline(
    manifest: dict[str, Any],
    op_instances: list[dict[str, Any]],
    variant_config: dict[str, Any],
) -> None:
    for entry_kind in ("input", "output"):
        for entry in manifest[f"{entry_kind}s"]:
            if entry["content_type"] == "JSON":
                raise ValueError(
                    f"Pipeline {entry_kind} `{entry['name']}` cannot use JSON content type"
                )

    instances_by_id = {instance["id"]: instance for instance in op_instances}
    bound_names: set[str] = set()
    for input_entry in manifest["inputs"]:
        input_name = input_entry["name"]
        if input_name in bound_names:
            raise ValueError(
                f"Pipeline input `{input_name}` binds a value more than once"
            )
        bound_names.add(input_name)

    for node_index, node in enumerate(variant_config["nodes"]):
        op_instance_id = node["op_instance_id"]
        node_name = f"Pipeline node {node_index} (`{op_instance_id}`)"
        op_instance = instances_by_id.get(op_instance_id)
        if op_instance is None:
            raise ValueError(
                f"{node_name} references undeclared op instance `{op_instance_id}`"
            )

        for io_kind in ("inputs", "outputs"):
            node_arity = len(node[io_kind])
            op_arity = len(op_instance[io_kind])
            if node_arity != op_arity:
                raise ValueError(
                    f"{node_name} has {io_kind[:-1]} arity {node_arity}, but op "
                    f"instance `{op_instance_id}` declares {op_arity}"
                )

        for input_name in node["inputs"]:
            if input_name not in bound_names:
                raise ValueError(f"{node_name} consumes unbound input `{input_name}`")

        for output_name in node["outputs"]:
            if output_name in bound_names:
                raise ValueError(
                    f"{node_name} binds value `{output_name}` more than once"
                )
            bound_names.add(output_name)


class Pipeline(BaseVariant):
    def _post_init(
        self,
    ) -> None:
        self.pipeline_node_instances: list[PipelineNodeInstance] = []
        for node in self.variant_config["nodes"]:
            op_instance = self.op_instances[node["op_instance_id"]]
            self.pipeline_node_instances.append(
                PipelineNodeInstance(
                    operator=op_instance.operator,
                    inputs=node["inputs"],
                    outputs=node["outputs"],
                    input_specs=op_instance.input_specs,
                    output_specs=op_instance.output_specs,
                    extra_dynattrs=node.get("extra_dynattrs", {}),
                )
            )

    async def _node_compute_async(
        self,
        node_instance: PipelineNodeInstance,
        node_inputs: list[Any],
        **node_passtrhough: Any,
    ) -> Any:
        validate_inputs(node_inputs, node_instance.input_specs)
        return await node_instance.operator.compute_async(
            node_inputs, **node_passtrhough
        )

    def _node_compute(
        self,
        node_instance: PipelineNodeInstance,
        node_inputs: list[Any],
        **node_passtrhough: Any,
    ) -> Any:
        validate_inputs(node_inputs, node_instance.input_specs)
        return node_instance.operator.compute(node_inputs, **node_passtrhough)

    async def compute_async(
        self,
        inputs: dict[str, Any],
        dynamic_attributes: dict[str, str],
    ) -> dict[str, Any]:
        passthrough = {
            "op_executor": self.op_executor,
            "dynamic_attributes": dynamic_attributes,
        }

        return await dag_compute_async(
            inputs,
            self.pipeline_node_instances,
            self._node_compute_async,
            as_val=lambda res: res.value,
            components_passthrough=passthrough,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        dynamic_attributes: dict[str, str],
    ) -> dict[str, Any]:
        passthrough = {
            "op_executor": self.op_executor,
            "dynamic_attributes": dynamic_attributes,
        }

        return dag_compute(
            inputs,
            self.pipeline_node_instances,
            self.executor,
            self._node_compute,
            as_val=lambda res: res.value,
            components_passthrough=passthrough,
        )
