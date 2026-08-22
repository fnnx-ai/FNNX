# This file is auto generated and must not be modified manually!
schema = {
    "version": "0.1.0",
    "manifest": {
        "$defs": {
            "JSON": {
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "content_type": {
                        "const": "JSON",
                        "title": "Content Type",
                        "type": "string",
                    },
                    "dtype": {"title": "Dtype", "type": "string"},
                    "tags": {
                        "anyOf": [
                            {"items": {"type": "string"}, "type": "array"},
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Tags",
                    },
                },
                "required": ["name", "content_type", "dtype"],
                "title": "JSON",
                "type": "object",
            },
            "NDJSON": {
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "content_type": {
                        "const": "NDJSON",
                        "title": "Content Type",
                        "type": "string",
                    },
                    "dtype": {
                        "description": "Must be in format 'Array[...]' or 'NDContainer[...]'",
                        "pattern": "^(Array\\[.+\\]|NDContainer\\[.+\\])$",
                        "title": "Dtype",
                        "type": "string",
                    },
                    "tags": {
                        "anyOf": [
                            {"items": {"type": "string"}, "type": "array"},
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Tags",
                    },
                    "shape": {
                        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                        "title": "Shape",
                        "type": "array",
                    },
                },
                "required": ["name", "content_type", "dtype", "shape"],
                "title": "NDJSON",
                "type": "object",
            },
            "Var": {
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "description": {"title": "Description", "type": "string"},
                    "tags": {
                        "anyOf": [
                            {"items": {"type": "string"}, "type": "array"},
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Tags",
                    },
                },
                "required": ["name", "description"],
                "title": "Var",
                "type": "object",
            },
        },
        "properties": {
            "variant": {"title": "Variant", "type": "string"},
            "name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Name",
            },
            "version": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Version",
            },
            "description": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Description",
            },
            "producer_name": {"title": "Producer Name", "type": "string"},
            "producer_version": {"title": "Producer Version", "type": "string"},
            "producer_tags": {
                "items": {"type": "string"},
                "title": "Producer Tags",
                "type": "array",
            },
            "inputs": {
                "items": {
                    "anyOf": [{"$ref": "#/$defs/NDJSON"}, {"$ref": "#/$defs/JSON"}]
                },
                "title": "Inputs",
                "type": "array",
            },
            "outputs": {
                "items": {
                    "anyOf": [{"$ref": "#/$defs/NDJSON"}, {"$ref": "#/$defs/JSON"}]
                },
                "title": "Outputs",
                "type": "array",
            },
            "dynamic_attributes": {
                "items": {"$ref": "#/$defs/Var"},
                "title": "Dynamic Attributes",
                "type": "array",
            },
            "env_vars": {
                "items": {"$ref": "#/$defs/Var"},
                "title": "Env Vars",
                "type": "array",
            },
        },
        "required": [
            "variant",
            "producer_name",
            "producer_version",
            "producer_tags",
            "inputs",
            "outputs",
            "dynamic_attributes",
            "env_vars",
        ],
        "title": "Manifest",
        "type": "object",
    },
    "ops_entries": {
        "$defs": {
            "OpDynamicAttribute": {
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "default_value": {"title": "Default Value", "type": "string"},
                },
                "required": ["name", "default_value"],
                "title": "OpDynamicAttribute",
                "type": "object",
            },
            "OpIO": {
                "properties": {
                    "dtype": {"title": "Dtype", "type": "string"},
                    "shape": {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "title": "Shape",
                        "type": "array",
                    },
                },
                "required": ["dtype", "shape"],
                "title": "OpIO",
                "type": "object",
            },
            "OpInstance": {
                "properties": {
                    "id": {
                        "pattern": "^[a-zA-Z0-9_]+$",
                        "title": "Id",
                        "type": "string",
                    },
                    "op": {"title": "Op", "type": "string"},
                    "inputs": {
                        "items": {"$ref": "#/$defs/OpIO"},
                        "title": "Inputs",
                        "type": "array",
                    },
                    "outputs": {
                        "items": {"$ref": "#/$defs/OpIO"},
                        "title": "Outputs",
                        "type": "array",
                    },
                    "attributes": {
                        "additionalProperties": True,
                        "title": "Attributes",
                        "type": "object",
                    },
                    "dynamic_attributes": {
                        "additionalProperties": {"$ref": "#/$defs/OpDynamicAttribute"},
                        "title": "Dynamic Attributes",
                        "type": "object",
                    },
                },
                "required": [
                    "id",
                    "op",
                    "inputs",
                    "outputs",
                    "attributes",
                    "dynamic_attributes",
                ],
                "title": "OpInstance",
                "type": "object",
            },
        },
        "properties": {
            "ops": {
                "items": {"$ref": "#/$defs/OpInstance"},
                "title": "Ops",
                "type": "array",
            }
        },
        "required": ["ops"],
        "title": "OpInstances",
        "type": "object",
    },
    "meta_entry": {
        "additionalProperties": True,
        "properties": {
            "id": {"title": "Id", "type": "string"},
            "producer": {"title": "Producer", "type": "string"},
            "producer_version": {"title": "Producer Version", "type": "string"},
            "producer_tags": {
                "items": {"type": "string"},
                "title": "Producer Tags",
                "type": "array",
            },
            "payload": {
                "additionalProperties": True,
                "title": "Payload",
                "type": "object",
            },
        },
        "required": ["id", "producer", "producer_version", "producer_tags", "payload"],
        "title": "MetaEntry",
        "type": "object",
    },
    "envs": {
        "python3::conda_pip": {
            "$defs": {
                "PipCondition": {
                    "properties": {
                        "platform": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Platform",
                        },
                        "os": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Os",
                        },
                        "accelerator": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Accelerator",
                        },
                    },
                    "title": "PipCondition",
                    "type": "object",
                },
                "PipDependency": {
                    "properties": {
                        "package": {"title": "Package", "type": "string"},
                        "extra_pip_args": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Extra Pip Args",
                        },
                        "condition": {
                            "anyOf": [
                                {"$ref": "#/$defs/PipCondition"},
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                    },
                    "required": ["package"],
                    "title": "PipDependency",
                    "type": "object",
                },
            },
            "properties": {
                "python_version": {"title": "Python Version", "type": "string"},
                "build_dependencies": {
                    "items": {"type": "string"},
                    "title": "Build Dependencies",
                    "type": "array",
                },
                "dependencies": {
                    "items": {"$ref": "#/$defs/PipDependency"},
                    "title": "Dependencies",
                    "type": "array",
                },
                "conda_channels": {
                    "anyOf": [
                        {"items": {"type": "string"}, "type": "array"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Conda Channels",
                },
            },
            "required": ["python_version", "build_dependencies", "dependencies"],
            "title": "Python3_CondaPip",
            "type": "object",
        }
    },
    "ops": {
        "ONNX_v1": {
            "$defs": {
                "ONNXAttributes": {
                    "properties": {
                        "opsets": {
                            "items": {"$ref": "#/$defs/Opset"},
                            "title": "Opsets",
                            "type": "array",
                        },
                        "has_external_data": {
                            "title": "Has External Data",
                            "type": "boolean",
                        },
                        "onnx_ir_version": {
                            "title": "Onnx Ir Version",
                            "type": "integer",
                        },
                        "used_operators": {
                            "anyOf": [
                                {
                                    "additionalProperties": {
                                        "items": {"type": "string"},
                                        "type": "array",
                                    },
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Used Operators",
                        },
                    },
                    "required": ["opsets", "has_external_data", "onnx_ir_version"],
                    "title": "ONNXAttributes",
                    "type": "object",
                },
                "OpDynamicAttribute": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "default_value": {"title": "Default Value", "type": "string"},
                    },
                    "required": ["name", "default_value"],
                    "title": "OpDynamicAttribute",
                    "type": "object",
                },
                "OpIO": {
                    "properties": {
                        "dtype": {"title": "Dtype", "type": "string"},
                        "shape": {
                            "items": {
                                "anyOf": [{"type": "integer"}, {"type": "string"}]
                            },
                            "title": "Shape",
                            "type": "array",
                        },
                    },
                    "required": ["dtype", "shape"],
                    "title": "OpIO",
                    "type": "object",
                },
                "Opset": {
                    "properties": {
                        "domain": {"title": "Domain", "type": "string"},
                        "version": {"title": "Version", "type": "integer"},
                    },
                    "required": ["domain", "version"],
                    "title": "Opset",
                    "type": "object",
                },
            },
            "properties": {
                "id": {"pattern": "^[a-zA-Z0-9_]+$", "title": "Id", "type": "string"},
                "op": {"const": "ONNX_v1", "title": "Op", "type": "string"},
                "inputs": {
                    "items": {"$ref": "#/$defs/OpIO"},
                    "title": "Inputs",
                    "type": "array",
                },
                "outputs": {
                    "items": {"$ref": "#/$defs/OpIO"},
                    "title": "Outputs",
                    "type": "array",
                },
                "attributes": {"$ref": "#/$defs/ONNXAttributes"},
                "dynamic_attributes": {
                    "additionalProperties": {"$ref": "#/$defs/OpDynamicAttribute"},
                    "title": "Dynamic Attributes",
                    "type": "object",
                },
            },
            "required": [
                "id",
                "op",
                "inputs",
                "outputs",
                "attributes",
                "dynamic_attributes",
            ],
            "title": "ONNX_v1",
            "type": "object",
        }
    },
    "variants": {
        "pipeline": {
            "$defs": {
                "PipelineNode": {
                    "properties": {
                        "op_instance_id": {"title": "Op Instance Id", "type": "string"},
                        "inputs": {
                            "items": {"type": "string"},
                            "title": "Inputs",
                            "type": "array",
                        },
                        "outputs": {
                            "items": {"type": "string"},
                            "title": "Outputs",
                            "type": "array",
                        },
                        "extra_dynattrs": {
                            "additionalProperties": {"type": "string"},
                            "title": "Extra Dynattrs",
                            "type": "object",
                        },
                    },
                    "required": [
                        "op_instance_id",
                        "inputs",
                        "outputs",
                        "extra_dynattrs",
                    ],
                    "title": "PipelineNode",
                    "type": "object",
                }
            },
            "properties": {
                "nodes": {
                    "items": {"$ref": "#/$defs/PipelineNode"},
                    "title": "Nodes",
                    "type": "array",
                }
            },
            "required": ["nodes"],
            "title": "PipelineVariant",
            "type": "object",
        },
        "pyfunc": {
            "properties": {
                "pyfunc_classname": {"title": "Pyfunc Classname", "type": "string"},
                "extra_values": {
                    "anyOf": [
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Extra Values",
                },
            },
            "required": ["pyfunc_classname"],
            "title": "PyFuncVariant",
            "type": "object",
        },
    },
}
