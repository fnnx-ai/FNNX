export interface ModelIO {
    name: string;
    content_type: string;
    dtype: string;
    tags?: string[];
}

export interface JSONIO extends ModelIO {
    content_type: "JSON";
}

export interface NDJSONIO extends ModelIO {
    content_type: "NDJSON";
    shape: (string | number)[];
}

export interface Var {
    name: string;
    description: string;
    tags?: string[];
}

export interface Manifest {
    variant: string;
    name?: string | null;
    version?: string | null;
    description?: string | null;
    producer_name: string;
    producer_version: string;
    producer_tags: string[];
    inputs: (NDJSONIO | JSONIO)[];
    outputs: (NDJSONIO | JSONIO)[];
    dynamic_attributes: Var[];
    env_vars: Var[];
}

export interface PipelineNode {
    op_instance_id: string;
    inputs: string[];
    outputs: string[];
    extra_dynattrs: Record<string, string>;
}

export interface PipelineVariant {
    nodes: PipelineNode[];
}

export interface OpIO {
    dtype: string;
    shape: (number | string)[];
}

export interface OpDynamicAttribute {
    name: string;
    default_value: string;
}

export interface OpInstanceConfig {
    id: string;
    op: string;
    inputs: OpIO[];
    outputs: OpIO[];
    attributes: Record<string, unknown>;
    dynamic_attributes: Record<string, OpDynamicAttribute>;
}

export interface ONNXOpset {
    domain: string;
    version: number;
}

export interface ONNXAttributes {
    opsets: ONNXOpset[];
    has_external_data: boolean;
    onnx_ir_version: number;
    used_operators?: Record<string, string[]> | null;
}

export interface MetaEntry {
    id: string;
    producer: string;
    producer_version: string;
    producer_tags: string[];
    payload: Record<string, unknown>;
}
