# Proposals

FNNX specification 0.1.0 (`spec/`) is the first written definition of the format. Both reference implementations predate it. The Python implementation (`src/python/fnnx`) was the de-facto definition and now diverges from the written rules in known places: it still carries the removed `requires_ort_extensions` attribute (and reads it under a wrong name), applies manifest patches and metadata sidecars in tar order instead of lexicographic filename order, lets falsy dynamic-attribute values trigger defaults, leaks per-node `extra_dynattrs` into later pipeline nodes, silently degrades missing declared outputs, lacks the `boolean` implicit scalar and non-finite float encoding, and ships generated schemas frozen at spec 0.0.4. The JavaScript implementation (`src/js`) is thinner — pipeline + `ONNX_v1` only — and diverges further: the node and web packages carry two hand-copied `Model` classes whose copies already disagree (byte-order vs locale-dependent patch ordering), the dynamic-attribute path is dead code due to a key-naming mismatch, the web tar reader takes the first occurrence of a repeated member instead of the last, `float64` is unsupported, and the `common` package has no tests at all.

The agreed roadmap makes conformance the gate before the spec is used as outward-facing material. This work brings both implementations up to spec 0.1.0 and documents them.

The proposal has three parts. First, align the Python implementation: regenerate the schema artifacts from the 0.1.0 source models, fix every divergence from the written rules, and add regression tests for each fixed rule. Second, align the JavaScript implementation, and use the alignment as the occasion to restructure it: the copy-paste duplication between the node and web packages is the direct cause of at least one divergence, so shared behavior moves into `@fnnx-ai/common` behind a file-access abstraction, dead code is removed, lint/format tooling is set up, and the `common` package gets a real test suite. Third, write user-facing documentation for both implementations in a defined Simplified-Technical-English style, so the docs match the register of the spec itself.

Alignment is chosen over rewriting because both implementations are small (~1.4k LOC Python core, ~1.1k LOC JS) and structurally sound at the module level; the defects are rule-level, not architectural. The JS refactoring is folded into this effort rather than deferred because fixing the divergences twice — once in each copied `Model` — would entrench the duplication that produced them.

# Design

## Ground rules

The spec prose in `spec/` is the sole authority. Where an implementation and the spec disagree, the implementation changes. This effort does not modify `spec/*.md`; the only permitted change under `spec/` is to the schema generator tooling in `spec/schemas/src/` where a generator bug blocks correct output (see below).

Every fixed rule gets a regression test that fails on the old behavior. Python tests live in `src/python/tests/` (unittest-style classes run under pytest, the existing convention). JS tests live per package under `packages/<pkg>/tests/` with Vitest.

Out of scope: pyfunc support in JS (the spec defines pyfunc as Python-hosted; JS must decline it identifiably), the conformance corpus and validator (next roadmap item), signing/integrity, new pipeline operations, and any change to what the spec says.

## Python alignment

**Schema regeneration.** Rerun `spec/schemas/src/generate.py` so `fnnx/spec.py` and `fnnx/extras/pydantic_models/` reflect spec 0.1.0 (`requires_ort_extensions` gone, version string updated). The generator copies pydantic model files without rewriting their import paths, which leaves `fnnx/extras/pydantic_models/ops/onnx.py` unimportable (`from pydantic_models.op_instances import ...`); fix the generator's copy step so the copied package imports correctly, and add a test that imports every copied module.

**ONNX_v1 extensions attribute.** Remove all handling of `requires_ort_extensions` and the misspelled `use_onnxruntime_extensions`: the ort-extensions registration branch in `fnnx/ops/onnx.py`, the rejection check in `fnnx/extras/compilers/c/bundle.py`, the attribute in test fixtures (`tests/models/onnx_pipeline.fnnx/ops.json` and the tar form), examples, and compiler tests. Support for non-standard opset domains is now signaled by `opsets` alone: when an op instance declares a domain the runtime cannot satisfy, the consumer declines with an error naming the domain and the op instance, before attempting execution. Runtime errors from onnxruntime during session creation are wrapped so the failing op instance is identifiable.

**Reading order and metadata.** `extras/reader.py` collects manifest patches and metadata sidecars by iterating tar members. Change both to the spec's rules: patch files apply in ascending lexicographic (byte order) filename order; effective metadata is `meta.json` first, then sidecars in the same order. A repeated member name contributes only its last occurrence. A metadata entry missing a required key is skipped without aborting the rest; an unparseable sidecar likewise must not abort the others (surface a warning, not an exception).

**Dynamic attributes.** In `fnnx/ops/_base.py`, a default applies only when the external name is absent from the caller-supplied mapping; a supplied empty string passes through. In `fnnx/variants/_common/dag.py`, a node's `extra_dynattrs` merge applies to that node's invocation only — the current code rebinds the shared mapping and leaks merged values into later nodes; fix both sync and async paths. Attribute values are strings end-to-end; type annotations state this, and the stdio wire preserves it.

**Pipeline validation and outputs.** The pipeline variant validates its graph at load, before any compute: every `op_instance_id` resolves to a declared op instance; node input/output arity equals the op instance's declared arity; no name is bound twice (model inputs and node outputs share one namespace); every node input is bound by a model input or an earlier node's output; and no manifest IO entry of a pipeline artifact uses the `JSON` content type (inputs and outputs alike). Each violation rejects the artifact with an error naming the offending node, name, or entry. At compute time, a declared output missing from the result is an error naming the output — in `LocalHandler`, and in `StdIOHandler`, whose current `outputs.get(name)` path silently fabricates values from `None`.

**Boolean and non-finite floats.** `fnnx/dtypes.py` gains `boolean` as the fourth implicit scalar: it validates a JSON boolean and nothing else, a boolean never validates as `integer` or `float`, and `boolean` joins the reserved names (together with the six `Array[...]` element tokens, which are also reserved by the spec). NDJSON float arrays decode the strings `"NaN"`, `"Infinity"`, `"-Infinity"` to their IEEE 754 values and reject any other string; the stdio wire encodes non-finite floats as those strings and never emits bare `NaN`/`Infinity` JSON tokens. Integer elements cross the wire as JSON integers with exact round-trip.

**Environment handling.** `StdIOHandler` currently falls back to treating the whole `env.json` as a `python3::conda_pip` spec when that key is absent; instead, a consumer that needs an environment and finds no kind it implements reports the artifact as unsupported, naming the kinds it found. Condition matching in `fnnx/envs/_common.py` uses substring matching for `platform`; the spec defines case-insensitive membership, i.e. equality against the declared array. Defaults in the env managers trigger on absence, not falsiness.

**Quality gates.** Add ruff and mypy configuration to `src/python/pyproject.toml`, scoped to exclude `fnnx/extras/compilers/` and the `test_extra_compiler_*` / `test_extra_mlflow_*` test files. Everything in scope passes both tools. CI runs them.

## JavaScript alignment and restructure

The workspace keeps its three packages: `@fnnx-ai/common` (dependency-free core), `@fnnx-ai/node`, `@fnnx-ai/web`. Breaking API changes are allowed; the three packages move in lockstep to version 0.1.0.

**Shared model core.** `common` gains an artifact-source abstraction: the operations the model core needs from a container (list root members, read a file, resolve an op instance's artifact files), with last-occurrence-wins semantics for repeated names built into the contract. One shared `Model` core in `common` implements everything above the source: manifest loading with patches, metadata assembly, dtypes, variant dispatch, warmup, compute, and read accessors for manifest, metadata, dtypes, and the raw `env.json` (JS never provisions environments — an in-process consumer ignores `env.json` per the spec — but callers can inspect it). `node` and `web` each contribute a source (filesystem/tar-extraction; in-memory `TarExtractor`) plus their ONNX backend, and re-export the pieces a caller needs to construct inputs (`web` currently exports only `Model`, which leaves browser callers without `NDArray`).

**Ordering and container rules.** Filename ordering for patches and sidecars is byte-order comparison, implemented once in `common` (no `localeCompare`). Metadata assembly is `meta.json` first, then sidecars, tolerant of malformed entries as on the Python side. The web `TarExtractor` honors last-occurrence-wins, rejects unsafe members (absolute paths, `..` segments, links, device nodes), and its two duplicated header-parsing paths (`extract` and `scan`) collapse into one header iterator. The node path passes an equivalent safety filter to its tar extraction.

**ONNX op.** One shared ONNX op base in `common` with an abstract session factory; `node` and `web` supply their onnxruntime backends. The op locates the model at exactly `ops_artifacts/<id>/model.onnx` (no suffix search). It reads the `attributes` declarations and declines identifiably anything it cannot satisfy: a non-standard opset domain, and `has_external_data` where the backend cannot resolve external data (node resolves files alongside `model.onnx`; if `onnxruntime-web` cannot load external data, `web` declines such instances with a named reason rather than failing obscurely).

**Wire-shaped types.** All interfaces describing on-disk documents use the wire field names exactly (`dynamic_attributes`, `default_value`, `extra_dynattrs`, ...). This removes the camelCase mismatch that currently makes the whole dynamic-attribute resolution path dead code, and the fixed path then follows the spec rule: default only on absence, values are strings.

**Dtypes and NDJSON codec.** The dtype layer supports all six `Array[...]` element types (adding `float64`) and all four implicit scalars (adding `boolean`, which is not an integer); the reserved-name list matches the spec. A nested↔flat codec in `common` converts between the spec's nested-JSON-array carriage and `NDArray`, validating shape against nesting. Float decoding accepts exactly the three non-finite strings and rejects other strings; encoding emits them. Integer elements round-trip exactly — `int64` values must serialize as JSON integers, not throw on `BigInt` or pass through a lossy double. Boolean casting must not turn the string `"false"` into `true`.

**Custom dtype validation.** `common` gets a real FNNX JSON Schema subset validator with the keyword set `core.md` defines, replacing the current `required`-only stub. Port the semantics of the Python `fnnx/validators/jsonschema.py` (including "a boolean is not an integer and not a number", local `$defs`/`$ref` only, unknown keywords ignored). This is reachable in a pipeline-only consumer through `NDContainer[<custom dtype>]` elements.

**Pipeline validation and errors.** The same load-time graph validation as the Python side: op-instance resolution, arity, duplicate binding, bound-before-use, `JSON` content type rejected for pipeline artifacts. At compute: an unbound node input is an error, never `undefined`; a declared output missing from the result is an error naming it — the current `prepareOutputs` filter that silently drops them goes away. Errors throughout become typed error classes carrying the identifying construct (variant name, op type, op instance id, file path), so "unsupported variant `pyfunc`" is distinguishable from a missing file without string matching.

**Tooling and tests.** ESLint (flat config) and Prettier configs at the workspace root with working `lint`/`format` scripts; dead code and unused imports removed (`verifyRequiredDynamicAttributes` call sites are added, not deleted — the spec requires the required-attribute check). `common` gets a Vitest config and a test suite covering the model core, codec, dtypes, validator, pipeline validation, and jsonpatcher (replacing the two identical copies in `node` and `web`); `node`/`web` keep integration-level tests. Root `pnpm test` runs all three packages; CI runs lint, unit tests, and the web Playwright e2e suite. Package manifests gain `license`, `repository`, and `exports` fields.

**JS-owned fixtures.** JS tests currently load only a fixture from `src/python/tests/models/`. Cross-language fixtures stay (they are a de-facto conformance check), but JS also gets small local fixtures for rule-level tests (duplicate members, patch ordering, malformed metadata) built in-test with the `tar` package or raw buffers.

## Documentation

Each package gets a README shipped with it: `src/python/README.md` (already referenced by pyproject, currently missing), `src/js/packages/common/README.md`, `src/js/packages/node/README.md`, `src/js/packages/web/README.md`. The Python README covers installation with extras, running artifacts (`Runtime`/`LocalHandler`, `StdIOHandler` with the env managers), inspecting artifacts (`Reader`), authoring (`PyfuncBuilder`, the MLflow converter), and the ONNX-to-C compiler CLI. The `node` and `web` READMEs cover install, loading a model (path/buffer; fetch), constructing inputs, and compute; `common`'s README states what it is (the runtime-independent core) and when to depend on it directly. READMEs link to `spec/` for format semantics instead of restating them.

### Documentation style

All READMEs follow this style. It applies to documentation prose, not to code identifiers or output samples.

Sentence level, following ASD-STE100 Simplified Technical English:
- Short sentences, at most ~20–25 words. One instruction or one fact per sentence.
- Active voice with a named actor ("The runtime loads...", "Call `warmup` before...").
- One term per concept throughout all documents; never alternate synonyms (pick "artifact", not sometimes "package"/"bundle"/"model file").
- Simple verbs. Prefer a period over a semicolon, a dash, or a colon-plus-list-in-a-sentence.
- Flag and rewrite any sentence over ~25 words or containing three or more commas. Do not join two requirements with "and".

Document level:
- Neutral, factual voice. No promotion or value judgments: never "powerful", "elegant", "simple", "easy", "seamless", "remarkably", "sophisticated". State what the software does and let facts speak.
- No meta-commentary: do not open with "This document covers..." and do not end with a summary; begin with content, stop when done.
- No preambles ("It is worth noting that...") and no restatements. State each fact once, in the document that owns it, and link elsewhere.
- Continuous prose over bullet lists. Use a list only for genuinely parallel enumerable items, kept short, without bold labels. Use a table only for reference data, not to restate prose.
- Descriptive unnumbered headings. No "What is X?" / "How it works" fragment subheadings. No horizontal rules between sections.
- Side information (definitions, history, caveats) goes in an italicized `*Note: ...*` block, keeping the main narrative clean.
- Link to external documentation (the spec, ONNX, onnxruntime) rather than re-explaining it.
- Acknowledge real limitations plainly (e.g. what the JS implementation does not support).
- Code blocks contain real, runnable code only. Verify names against the codebase before writing them.

# Scenarios

Unless marked Python-only or JS-only, each scenario applies to both implementations and gets a test in each.

## Scenario: Manifest patches apply in filename order
**Given** an artifact whose tar contains `manifest-b.patch.json` before `manifest-a.patch.json`, both replacing the same manifest field
**When** the effective manifest is read
**Then** the value from `manifest-b.patch.json` wins, because patches apply in ascending byte-order filename order regardless of member order

## Scenario: Repeated member name keeps only the last occurrence
**Given** a tar artifact where `meta-x.json` appears twice with different entries, and `manifest.json` appears twice
**When** the artifact is read
**Then** metadata contains only the second `meta-x.json`'s entries exactly once, and the manifest is the second `manifest.json`

## Scenario: Metadata assembles in the defined order and tolerates bad entries
**Given** an artifact with `meta.json`, `meta-b.json`, and `meta-a.json`, where `meta-a.json` contains one entry missing `producer` and one valid entry
**When** metadata is read
**Then** entries appear in the order `meta.json`, `meta-a.json`, `meta-b.json`; the non-conforming entry is skipped; the valid entries of the same file are still returned; no exception aborts the read

## Scenario: Dynamic attribute default applies only on absence
**Given** an op instance mapping internal key `k` to external name `n` with `default_value` `"d"`
**When** the caller supplies `{"n": ""}` and, separately, supplies no `n`
**Then** the resolved value is `""` in the first case and `"d"` in the second

## Scenario: extra_dynattrs do not leak across nodes
**Given** a pipeline where node 1 declares `extra_dynattrs {"n": "pinned"}` and node 2 declares none, and both nodes' op instances read external name `n`
**When** the caller supplies `{"n": "caller"}`
**Then** node 1 resolves `"pinned"`, node 2 resolves `"caller"`

## Scenario: Missing declared output is an error
**Given** a loaded artifact whose computation returns a mapping without the declared output `y`
**When** compute runs
**Then** the consumer raises an error that names `y` and returns no partial result (Python: both `LocalHandler` and `StdIOHandler`; JS: `Model.compute`)

## Scenario: Pipeline graph is validated at load
**Given** pipeline artifacts that respectively (a) reference an undeclared `op_instance_id`, (b) declare node inputs whose count differs from the op instance's arity, (c) bind the same value name twice, (d) consume a name no earlier node or model input binds
**When** the artifact loads
**Then** each is rejected before any compute, with an error naming the offending node, name, or instance

## Scenario: JSON content type rejected for pipeline
**Given** a pipeline artifact declaring a manifest output with `content_type` `JSON`
**When** the artifact loads
**Then** it is rejected with an error naming the entry

## Scenario: Boolean is a distinct implicit scalar
**Given** a value of dtype `NDContainer[boolean]` containing `true`, and a `dtypes.json` attempting to redefine `boolean`
**When** the value is validated and the artifact is loaded
**Then** `true` validates as `boolean`, `true` does not validate as `integer`, and the redefinition is rejected as a reserved name

## Scenario: Non-finite floats cross the boundary as strings
**Given** a float array value containing `NaN` and `Infinity`
**When** the value is encoded to NDJSON and decoded back
**Then** the encoded JSON contains the strings `"NaN"` and `"Infinity"` (never bare tokens or `null`), and decoding restores the IEEE 754 values; decoding a float array containing any other string fails with an error

## Scenario: Integers round-trip exactly
**Given** an `Array[int64]` value containing `2^53 + 1`
**When** it crosses the wire (Python StdIO; JS codec serialization)
**Then** the decoded value equals `2^53 + 1` exactly

## Scenario: ONNX op declines undeclared-domain instances identifiably
**Given** an op instance whose `opsets` declare a non-standard domain the runtime does not support
**When** the artifact is executed
**Then** the consumer declines before execution with an error naming the domain and the op instance

## Scenario: Removed extensions attribute no longer exists (Python-only)
**Given** the regenerated schemas and fixtures
**When** grep runs over `src/python` for `requires_ort_extensions` and `use_onnxruntime_extensions`
**Then** there are no code hits (fixtures, examples, compiler, runtime), and `fnnx/spec.py` carries version `0.1.0`

## Scenario: Unsupported environment kind is not approximated (Python-only)
**Given** an artifact whose `env.json` contains only a kind key the handler does not implement
**When** `StdIOHandler` loads it
**Then** it reports the artifact unsupported and names the offered kinds, instead of interpreting the object as a `python3::conda_pip` spec

## Scenario: Env condition platform matching is exact (Python-only)
**Given** a dependency with condition `platform: ["x86"]` evaluated on machine `x86_64`
**When** dependencies are selected
**Then** the dependency does not match (membership means case-insensitive equality, not substring)

## Scenario: Web reader honors container safety rules (JS-only)
**Given** a tar containing a member with a `..` path segment and a symlink member
**When** the web `TarExtractor` (and the node extraction path) processes it
**Then** the unsafe members are rejected or ignored and never written or surfaced as content

## Scenario: ONNX model located only at its fixed path (JS-only)
**Given** an op instance directory containing `sub/model.onnx` and `xmodel.onnx` but no `model.onnx`
**When** the op loads
**Then** loading fails with an error naming the expected path, instead of picking a near-match

## Scenario: pyfunc declined identifiably (JS-only)
**Given** an artifact with `manifest.variant` `pyfunc`
**When** it is loaded in JS
**Then** a typed error reports the variant as unsupported and names `pyfunc`

## Scenario: Dynamic attributes reach the ONNX-adjacent test op via wire field names (JS-only)
**Given** an op instance declaring `dynamic_attributes` with `default_value` on disk
**When** the artifact is loaded and computed with and without the attribute supplied
**Then** the resolution path runs (not dead code): supplied values arrive, defaults apply on absence, and required attributes missing at resolution raise the required-attribute error

## Scenario: Custom dtype validation enforces the schema subset (JS-only)
**Given** an `NDContainer[ext::rec]` where `ext::rec` requires `{"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}`
**When** an element `{"a": true}` is validated
**Then** validation fails (boolean is not an integer), and an element `{"a": 1}` passes

# Tasks

- [x] Regenerate Python schemas and drop the ONNX extensions attribute
  - [x] Fix the generator's pydantic-model copy step so copied imports resolve, in `spec/schemas/src/generate.py`
  - [x] Regenerate `fnnx/spec.py` and `fnnx/extras/pydantic_models/` at spec version 0.1.0
  - [x] Remove `use_onnxruntime_extensions` handling from `fnnx/ops/onnx.py` and `requires_ort_extensions` from the compiler bundle check, fixtures, examples, and tests
  - [x] Add the non-standard-domain decline and identifiable ORT error wrapping in `fnnx/ops/onnx.py`
  - [x] Tests: pydantic package imports, fixture cleanliness, domain decline
- [x] Fix Python artifact reading order and metadata handling
  - [x] Byte-order patch application and meta assembly in `fnnx/extras/reader.py`
  - [x] Last-occurrence-wins for repeated member names in patches and metadata
  - [x] Malformed-entry and unparseable-sidecar tolerance
  - [x] Tests: ordering, duplicates, tolerance (extend `tests/test_reader.py`)
- [x] Fix Python dynamic attributes and pipeline semantics
  - [x] Absence-only defaults in `fnnx/ops/_base.py`; string-typed values end-to-end
  - [x] Fix `extra_dynattrs` leakage in `fnnx/variants/_common/dag.py` (sync and async)
  - [x] Load-time pipeline graph validation (resolution, arity, double-bind, bound-before-use, JSON content type)
  - [x] Missing-declared-output errors in `fnnx/handlers/local.py` and `fnnx/handlers/stdio/`
  - [x] Tests for each rule
- [ ] Add boolean scalar and non-finite float handling to Python
  - [ ] `boolean` implicit scalar and full reserved-name list in `fnnx/dtypes.py`
  - [ ] Non-finite float string encoding/decoding on the stdio wire and input marshalling; reject other strings in float arrays
  - [ ] Integer exactness across the wire
  - [ ] Tests: boolean vs integer, round-trips, rejection cases
- [ ] Fix Python environment kind handling
  - [ ] Remove the whole-`env.json` fallback in `fnnx/handlers/stdio/__init__.py`; report unsupported kinds by name
  - [ ] Exact membership matching for platform conditions in `fnnx/envs/_common.py`; absence-based defaults in the env managers
  - [ ] Tests: unsupported kind, condition matching
- [ ] Add ruff and mypy gates to the Python package
  - [ ] Configs in `src/python/pyproject.toml`, excluding `fnnx/extras/compilers/` and compiler/mlflow test files
  - [ ] Fix all findings in scope; wire both tools into CI
- [ ] Set up JS lint, format, and test tooling
  - [ ] ESLint flat config and Prettier at the workspace root; working `lint`/`format` scripts in every package
  - [ ] Vitest config and test scaffolding for `@fnnx-ai/common`; root `pnpm test` includes it
  - [ ] Move the duplicated jsonpatcher tests into `common`
  - [ ] CI runs lint, all unit tests, and the web Playwright e2e suite
- [ ] Restructure JS packages around a shared model core
  - [ ] Artifact-source abstraction with last-occurrence-wins in `common`; node and web sources
  - [ ] One shared `Model` core (manifest+patches, metadata, dtypes, variant dispatch, env accessor); thin node/web wrappers
  - [ ] Shared ONNX op base with abstract session factory; fixed `model.onnx` path; attribute declaration checks
  - [ ] Wire-shaped interface types; typed error classes; remove dead code and the device-map dead plumbing
  - [ ] Split `ndarray.ts` and `tar.ts` by concern; unify the tar header parsers; tar safety rules
  - [ ] Re-export input-construction types from node and web entry points; add `license`/`repository`/`exports`; lockstep versions at 0.1.0
  - [ ] Migrate existing tests; add model-core tests in `common`
- [ ] Align JS dtypes and NDJSON handling with the spec
  - [ ] `float64` and `boolean` support; full reserved-name list; safe boolean casting
  - [ ] Nested↔flat NDJSON codec with shape validation, non-finite string handling, exact integers (including `int64` serialization)
  - [ ] FNNX JSON Schema subset validator replacing the `required`-only stub
  - [ ] Tests: all six element types, four scalars, codec round-trips, validator keyword coverage
- [ ] Align JS pipeline and reading semantics with the spec
  - [ ] Load-time pipeline graph validation; unbound-input and missing-output errors; JSON content type rejection
  - [ ] Byte-order filename sorting for patches and sidecars; metadata order and tolerance
  - [ ] Live dynamic-attribute resolution (wire field names) with required-attribute verification
  - [ ] JS-owned rule-level fixtures; tests for each rule
- [ ] Write the Python package README
  - [ ] `src/python/README.md` per the documentation style: install/extras, runtime usage, Reader, StdIO + env managers, PyfuncBuilder, MLflow converter, compiler CLI
- [ ] Write the JS package READMEs
  - [ ] `common`, `node`, and `web` READMEs per the documentation style; bump versions if not yet done in the restructure task
