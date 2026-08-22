from typing import Any, cast

from fnnx.spec import schema
from fnnx.validators.jsonschema import validate_jsonschema


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate_jsonschema(manifest, schema["manifest"])


def validate_op_instances(instances: list[dict[str, Any]]) -> None:
    if not isinstance(instances, list):
        raise ValueError("Nodes must be a list")
    op_schemas = cast(dict[str, dict[str, Any]], schema["ops"])
    for instance in instances:
        op_type = instance["op"]
        if op_type not in op_schemas:
            raise ValueError(f"Unknown op type: {op_type}")
        validate_jsonschema(instance, op_schemas[op_type])


def validate_variant(variant: str, config: dict[str, Any]) -> None:
    variant_schemas = cast(dict[str, dict[str, Any]], schema["variants"])
    if variant not in variant_schemas:
        raise ValueError(f"Unknown variant: {variant}")
    validate_jsonschema(config, variant_schemas[variant])
