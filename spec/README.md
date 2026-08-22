# FNNX Specification

Version 0.1.0.

FNNX is a packaging format for machine learning models. An FNNX artifact carries a model, an explicit statement of how to execute it, and the compatibility expectations that apply to it. FNNX does not serialize every possible model representation. It assigns explicit compatibility expectations to the representations it standardizes.

A model here is any computation with a declared interface. A neural network qualifies. So does a preprocessing pipeline, a compound multi-model system, or an agentic pipeline.

One artifact is one model. An artifact contains a manifest, a variant, op instance declarations, metadata entries, and an optional environment specification. The manifest describes the model's identity and its input and output interface. The variant determines the execution semantics. The op instance declarations name the operations the variant may invoke. The metadata entries carry producer-defined information. The environment specification describes the runtime environment the artifact expects. Two variants exist: `pipeline` describes a graph of first-party operations, and `pyfunc` describes a Python entry point. The rest of the artifact is interpreted relative to its variant.

## Specification version

This document set is version 0.1.0. The version matches the `SPEC_VERSION` constant used to generate the machine-readable schemas in [`schemas/`](schemas/). It governs the whole document set: the core artifact model, the variants, the operations, the environment kinds, and the generated schemas. A change to any of them advances the specification version.

The specification version is not a compatibility handle for artifacts. Artifacts carry no global specification-version field. Constructs are versioned by name instead. A consumer interprets an artifact through the construct names it carries: the variant name, the op names in its op instance declarations, and the environment kind keys in its environment specification. `ONNX_v1` is a distinct operation from any future `ONNX_v2`, and the two may coexist in one artifact. A consumer that does not recognize a name knows which construct it is missing. A specification-version field would add a second and coarser compatibility axis over these names.

## Documents

- [`compatibility.md`](compatibility.md) — the FNNX compatibility model: maturity states, portability and execution durability grades, external dependencies, composition closure, the current classification of implemented constructs, and the operation-set policy.
- [`core.md`](core.md) — FNNX Core: the artifact model, container encodings, root files, the manifest, the data model, op instance declarations, the metadata system, manifest patches, and the append and extension model.
- [`variants/`](variants/README.md) — the variant collection: [`pipeline`](variants/pipeline.md), [`pyfunc`](variants/pyfunc.md).
- [`ops/`](ops/README.md) — the operation collection: [`ONNX_v1`](ops/onnx_v1.md).
- [`envs/`](envs/README.md) — the environment kind collection: [`python3::conda_pip`](envs/python3_conda_pip.md).
- [`schemas/`](schemas/) — machine-readable JSON Schema definitions, generated from the specification's source models. `schemas/combined.json` aggregates the individual schema files. Where a schema and the prose of this specification disagree, the prose is normative. The relevant document records the divergence in a note.

Each collection directory has a README that indexes its documents and holds the template for new ones. Each construct document opens with a header table. The table gives the construct key, the specification version that introduced the construct (Added in), the version of its last normative change (Updated in), its classification where one applies, and its machine-readable schema. A variant's table also names the operations it admits.

## Notation

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY and OPTIONAL in this specification are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) and [RFC 8174](https://www.rfc-editor.org/rfc/rfc8174) when, and only when, they appear in all capitals.

Normative requirements are addressed to three roles. A **producer** writes an artifact. A **consumer** reads an artifact, and in most cases executes it. An **appender** extends an existing artifact without rewriting what is already in it. One piece of software may act in more than one role. A requirement stated for one role does not implicitly apply to the others.

Material marked with an italic *Note:* is non-normative. It records rationale, history or clarification, and imposes no requirements.

## Maturity states

Every construct — the core format, the variants, the operations and the environment kinds — carries a maturity state. An **Experimental** construct may change incompatibly or be removed. A **Stable** construct has an immutable normative meaning within its published version. A **Deprecated** construct keeps its guarantees, but SHOULD NOT be used in new artifacts.

[`compatibility.md`](compatibility.md) defines the states and their normative force in full, together with the portability and execution durability classifications of the defined constructs.

## Conformance

A conforming artifact satisfies the requirements of [`core.md`](core.md), of the variant it names, and of every operation and environment kind it references. A conforming consumer correctly interprets the constructs it claims to support. It rejects artifacts that require constructs it does not support, and states an identifiable reason. Partial support is acceptable and expected: a consumer that implements only the `pipeline` variant is conforming if it declines `pyfunc` artifacts explicitly instead of misinterpreting them.

A conformance test corpus is a planned companion of this specification. It is not part of this document set.
