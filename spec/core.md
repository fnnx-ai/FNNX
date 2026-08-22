# FNNX Core

FNNX Core defines what every FNNX artifact carries, whatever computation it holds. It defines the package format, the root files, the declared interface, the data types, the operation instance and environment declarations, and the metadata. It also defines how a consumer decides whether it can handle an artifact. The [variants](variants/README.md) define what differs between kinds of computation. [compatibility.md](compatibility.md) defines the compatibility expectations for each construct.

## The artifact model

An FNNX artifact is a tree of files that describes itself. The files declare the external interface. They declare the operation instances, and hold the payload data those instances need. They describe how the pieces compose, in a form the variant defines. They can also describe the execution environment and carry metadata.

[README.md](README.md) defines the three roles, and each requirement here names the role it addresses. Core divides consumers into two kinds. An inspection consumer reads the declarations only. An execution consumer also runs the computation.

An artifact carries no global format-version field, and consumers MUST NOT expect one. Meaning comes from the construct names the artifact itself carries: `manifest.variant`, the operation type names in `ops.json`, the environment kind keys in `env.json`, and the dtype names. A consumer checks which of those names it implements. It never compares version numbers.

Core itself has no versioned name. An artifact cannot state which revision of Core it targets. A Stable Core rule therefore never changes incompatibly. An incompatible revision is published as a new named construct, not as a change to an established rule.

*Note: the version in [schemas/combined.json](schemas/combined.json) identifies that schema bundle and the specification revision it came from. It is not part of an artifact. Where a schema and this document disagree, this document is normative.*

## Container encodings

A conforming artifact is a directory tree or an uncompressed POSIX tar archive. The two encodings are equivalent: tar member names are exactly the relative paths of the directory form, with `/` as the separator. Every requirement here applies to both.

Producers MUST write tar archives in the ustar format. A member path too long for the ustar name and prefix fields is carried in a PAX `path` record or a GNU long-name entry. Consumers SHOULD accept both. Directory member entries are OPTIONAL. Consumers MUST tolerate their presence and their absence, and MUST NOT require a directory entry before accepting members beneath it.

Producers MUST NOT compress the archive. Appending, byte-range access and streaming member scans all depend on an uncompressed contiguous archive. Consumers MAY accept a transparently compressed stream. A compressed file is still not an FNNX artifact, and appenders MUST refuse to extend one.

Member paths MUST be relative to the artifact root. Members MUST NOT use absolute paths or `..` segments. Members MUST NOT be symbolic links, hard links or device nodes. Consumers MUST reject or ignore a member that breaks these rules, and MUST NOT write it to disk during extraction.

`manifest.json` SHOULD be the first member of a tar archive, so a consumer can read the declarations from a stream head or one ranged request.

Member names MAY repeat, because artifacts are append-extensible. The last occurrence of a name wins, and consumers MUST ignore the earlier ones.

The `.fnnx` extension is conventional, not required. Consumers recognize artifacts by structure and content, and MUST NOT make behaviour depend on the file extension.

A typical artifact has the layout below. Only the root files are mandatory.

```
model.fnnx/
├── manifest.json
├── ops.json
├── variant_config.json
├── dtypes.json
├── env.json
├── meta.json
├── meta-8f3a1c2e4b6d47f0a9c8e5b3d1f70246.json
├── manifest-8f3a1c2e4b6d47f0a9c8e5b3d1f70246.patch.json
├── ops_artifacts/
│   ├── linreg/
│   │   └── model.onnx
│   ├── linreg2/
│   │   └── model.onnx
│   ├── linreg3/
│   │   └── model.onnx
│   └── concat_reduce/
│       └── model.onnx
├── variant_artifacts/
└── meta_artifacts/
    └── 8f3a1c2e4b6d47f0a9c8e5b3d1f70246/
        └── report.json
```

## Root files

Six files at the artifact root are REQUIRED. Producers MUST write all six, even when the content is empty.

| File | Type | Empty form | Meaning |
| --- | --- | --- | --- |
| `manifest.json` | object | — | The artifact's interface and identity. |
| `ops.json` | array | `[]` | Declarations of the embedded operation instances. |
| `variant_config.json` | object | `{}` | Configuration whose shape is defined by the variant. |
| `dtypes.json` | object | `{}` | Custom data type definitions. |
| `env.json` | object | `{}` | Execution environment descriptions, keyed by environment kind. |
| `meta.json` | array | `[]` | Metadata entries. |

Consumers MAY treat a missing `dtypes.json`, `env.json` or `meta.json` as the empty value above. A missing `manifest.json`, `ops.json` or `variant_config.json` is an error.

Two further families of root files are OPTIONAL: metadata sidecars matching `^meta(-[^/]+)?\.json$`, and manifest patches matching `^manifest-[^/]+\.patch\.json$`.

Three directories are reserved. `ops_artifacts/` holds one subdirectory per operation instance, named by the instance id. `variant_artifacts/` holds files whose layout the variant defines. `meta_artifacts/` holds one subdirectory per metadata entry that has auxiliary files, named by the entry id. Any other member is unknown content.

## The manifest

`manifest.json` is a JSON object. It states what the artifact is, what it accepts, and what it returns.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `variant` | string | yes | Name of the variant that defines the artifact's execution model. |
| `name` | string or null | no | Human-readable model name. |
| `version` | string or null | no | Version of the model itself. Unrelated to any specification version. |
| `description` | string or null | no | Human-readable description. |
| `producer_name` | string | yes | Identifier of the tool or system that produced the artifact. |
| `producer_version` | string | yes | Version of that producer. |
| `producer_tags` | array of string | yes | Tags describing the artifact, possibly empty. |
| `inputs` | array of IO entries | yes | Ordered declaration of the computation's inputs. |
| `outputs` | array of IO entries | yes | Ordered declaration of the computation's outputs. |
| `dynamic_attributes` | array of Var | yes | Declared per-invocation attributes, possibly empty. |
| `env_vars` | array of Var | yes | Declared environment variables the computation reads, possibly empty. |

An IO entry declares one named value crossing the artifact boundary. Every entry carries these fields.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | The name callers use for this value. Names MUST be unique within `inputs` and within `outputs`. |
| `content_type` | string | yes | The content type of the value. |
| `tags` | array of string | no | Tags describing this value. |

The content type decides the remaining fields of the entry. [Content types](#content-types) defines the two content types and the fields each one adds.

A Var declares an out-of-band parameter of the computation.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | The name of the dynamic attribute or environment variable. |
| `description` | string | yes | Human-readable description, possibly empty. |
| `tags` | array of string | no | Tags describing this parameter. |

`dynamic_attributes` and `env_vars` document which parameters exist and what they mean. They do not gate execution. Attribute and environment variable values are strings. A caller MAY pass an attribute that `manifest.dynamic_attributes` does not declare, and consumers MUST pass it on. A caller MAY omit a declared attribute, and consumers MUST NOT reject the invocation for that reason. The operation instance declarations or the variant decide whether an attribute is required.

A minimal manifest for a pipeline artifact:

```json
{
    "variant": "pipeline",
    "name": "linreg_ensemble",
    "version": "1.0.0",
    "description": "Three linear regressions combined into one score.",
    "producer_name": "example-exporter",
    "producer_version": "1.0.0",
    "producer_tags": ["example.org::regression:v1"],
    "inputs": [
        {
            "name": "x",
            "content_type": "NDJSON",
            "dtype": "Array[float32]",
            "shape": ["batch", 3]
        }
    ],
    "outputs": [
        {
            "name": "y4",
            "content_type": "NDJSON",
            "dtype": "Array[float32]",
            "shape": ["batch", 1]
        }
    ],
    "dynamic_attributes": [],
    "env_vars": []
}
```

## Content types

A content type states how a value crosses the artifact boundary. It decides the encoding of the value, and the fields the IO entry carries beyond the common ones. The dtype of the entry states what the value is. Each content type names the dtype forms it permits, and the declared dtype MUST have one of those forms.

This specification defines two content types: `NDJSON` and `JSON`. No other value of `content_type` is permitted. The variant decides which of the two an artifact may use. A content type whose meaning changes incompatibly gets a new name. A new content type brings its own field set. It does not add a flag to an existing one.

### NDJSON

`NDJSON` denotes an n-dimensional value: an array of a single element type, or a container of typed elements. An `NDJSON` entry carries two further fields.

*Note: `NDJSON` is short for n-dimensional JSON. It is not newline-delimited JSON (ndjson.org). The value is one nested JSON array, not a stream of JSON lines.*

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `dtype` | string | yes | An `Array[...]` or `NDContainer[...]` form, see [the dtype language](#the-dtype-language). |
| `shape` | array of integer or string | yes | Declared shape, see [Shapes](#shapes). |

The value is carried as nested JSON arrays. The element type comes from the declared dtype, not from the JSON encoding.

JSON has no token for a non-finite number. A float element that is not finite is carried as one of the strings `"NaN"`, `"Infinity"` and `"-Infinity"`. Producers MUST NOT put any other string in a float array. Consumers MUST decode these three strings to the IEEE 754 values.

An integer element is carried as a JSON integer. Consumers MUST preserve its exact value, and MUST NOT pass it through a lossy floating-point representation.

### JSON

`JSON` denotes a single JSON document. A `JSON` entry carries one further field.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `dtype` | string | yes | A custom dtype defined in `dtypes.json`, see [Custom dtypes](#custom-dtypes). |

The value is carried as the JSON document itself, and MUST validate against the schema of the named dtype. A `JSON` entry has no rank and no axes. Producers MUST NOT declare a `shape` on one, and consumers MUST ignore a `shape` they find there.

## The dtype language

A dtype names the data type of a value. Two forms are parameterized, written with square brackets. The rest are plain names. Dtype names carry no version: a form whose meaning changes incompatibly gets a new name.

### `Array[<element>]`

`Array[<element>]` denotes a dense rectangular array whose elements all have the primitive type named by `<element>`. The element token MUST be one of `float32`, `float64`, `int32`, `int64`, `bool` and `string`. A consumer that supports `Array` dtypes at all MUST support all six. Any other element name is outside this specification: producers SHOULD keep to the six, and consumers MAY reject anything else.

### `NDContainer[<dtype>]`

`NDContainer[<dtype>]` denotes an n-dimensional container. Its elements are values of the named inner dtype. The inner dtype MUST be an implicit scalar dtype, or a custom dtype defined in `dtypes.json`. It MUST NOT be an `Array[...]` form.

The value is carried as nested JSON arrays. Each level of nesting is one dimension, outermost first, and the deepest level holds the elements. Every element MUST validate against the inner dtype. Nesting MUST be rectangular. Consumers need not detect ragged nesting, and producers MUST NOT rely on ragged data being accepted or rejected in any particular way.

### Implicit scalar dtypes

Four implicit scalar dtypes are always available, and producers MUST NOT declare them in `dtypes.json`. `string` matches a JSON string. `integer` matches a JSON integer. `float` matches a JSON number with a fractional part, and producers MUST write the fractional part, as in `2.0`. Consumers MAY also accept a whole JSON number where `float` is declared, because some JSON parsers do not preserve the distinction between `2.0` and `2`. `boolean` matches a JSON boolean. A boolean is not an integer.

### Custom dtypes

`dtypes.json` maps a custom dtype name to a JSON Schema for values of that dtype. A custom dtype may be the `dtype` of a `JSON` IO entry, or the inner dtype of an `NDContainer[...]`.

The character `[` opens a parameterized form. A dtype name MUST NOT contain it. The names `string`, `integer`, `float`, `boolean`, `Array` and `NDContainer` are reserved and MUST NOT be redefined in `dtypes.json`. The six element tokens of `Array[...]` are also reserved. Custom dtype names SHOULD carry the `ext::` prefix, as in `ext::customer_record`. Names intended to be recognized outside the producing system SHOULD be namespaced beyond that prefix.

Consumers validating `JSON` values and `NDContainer` elements support the FNNX JSON Schema subset. The subset holds the following keywords.

Structure and composition: `$defs`, `$ref` (only local references of the form `#/$defs/<name>`), `const`, `enum`, `not`, `anyOf`, `allOf`, `oneOf`, and `if`/`then`/`else`.

Types: `type`, with the values `object`, `array`, `string`, `integer`, `number`, `boolean` and `null`. A boolean is not an integer and not a number.

Objects: `required`, `minProperties`, `maxProperties`, `properties`, `patternProperties`, `additionalProperties` and `dependencies`.

Arrays: `minItems`, `maxItems`, `uniqueItems`, `items` (both the single-schema and the tuple form) and `additionalItems`.

Strings: `minLength`, `maxLength`, `pattern` and `format`. Only the `format` values `email` and `uri` carry validation.

Numbers: `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` and `multipleOf`.

Consumers MUST ignore keywords outside this subset, including annotations such as `$schema`, `title` and `description`. An unknown keyword is not an error. Producers MUST NOT depend on a constraint outside the subset being enforced.

*Note: `pattern` and `patternProperties` values are regular expressions, and this specification does not fix a dialect. Producers SHOULD keep patterns within a conservative common subset.*

## Shapes

A shape is an array with one entry per axis, so its length fixes the rank of the value. An integer entry declares an exact extent, and a value whose corresponding axis has a different extent does not conform. A string entry declares a symbolic extent and places no constraint on that axis. A shape MAY mix the two forms.

Symbolic names are documentation. A consumer need not check that axes with the same name are equal, and producers MUST NOT rely on such a check. Using the same name for axes that are in fact equal remains RECOMMENDED. The variant and the operation instance declarations decide where a shape is validated.

## Operation instance declarations

`ops.json` is a bare JSON array of operation instance objects. An artifact with no operations contains `[]`.

The variant decides whether an artifact may declare op instances at all, which operation types it admits, and how the computation invokes them. A variant that admits no operations requires `ops.json` to be `[]`.

*Note: the generated schemas wrap this array in an object with a single `ops` key: [schemas/ops.json](schemas/ops.json), and the `ops_entries` entry in [schemas/combined.json](schemas/combined.json). The wrapper is an artifact of schema generation, and producers MUST NOT emit it.*

Each element declares one instance of an operation type.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string matching `^[a-zA-Z0-9_]+$` | yes | Identifier of this instance, unique within the artifact. |
| `op` | string | yes | The versioned operation type name, for example `ONNX_v1`. |
| `inputs` | array of `{dtype, shape}` | yes | Ordered declaration of the instance's inputs. |
| `outputs` | array of `{dtype, shape}` | yes | Ordered declaration of the instance's outputs. |
| `attributes` | object | yes | Build-time attributes whose keys and meaning the operation type defines. |
| `dynamic_attributes` | object | yes | Mapping from internal attribute names to their external source, see below. |

The entries of `inputs` and `outputs` are ordinal: the *i*-th entry describes the *i*-th value the instance consumes or produces. Entries carry no names, and variants bind values by position. Execution consumers MUST validate the values they pass to an instance against these declarations.

The operation type defines the contents of `attributes`, and the [operations](ops/README.md) specify them. Attributes are fixed when the artifact is produced.

`dynamic_attributes` maps an internal attribute name, meaningful to the operation type, to an object with the REQUIRED keys `name` and `default_value`. Resolution runs per internal key. If the caller-supplied attribute map contains the external name `name`, the consumer uses that value. If it does not, the consumer uses the string `default_value`. Only absence triggers the default. A supplied empty string MUST be passed through unchanged. An operation type MAY declare attributes it requires. If a required attribute resolves to nothing, execution consumers MUST report an error and MUST NOT proceed.

An instance's payload files live under `ops_artifacts/<id>/`, so the id pattern allows only path-safe characters. The operation type defines the layout inside that directory. An instance whose operation type needs no files has no directory.

A consumer that does not implement a named operation type cannot execute the artifact. It MUST report the type as unsupported and name it. It MUST NOT substitute another operation, ignore the instance, or guess semantics from the name.

## The environment slot

`env.json` maps environment-kind identifiers to descriptions of how to reconstruct an execution environment for the computation. The [environment kinds](envs/README.md) define the identifiers and the descriptions. An empty object means the artifact carries no environment description.

One artifact MAY describe the same environment under several kinds, and a consumer picks a kind it implements. Consumers MUST ignore keys they do not recognize. A consumer that needs an environment description and implements none of the offered kinds MUST report the artifact as unsupported.

Only consumers that provision environments read this slot, such as those that use a separate interpreter or a container. An in-process consumer ignores `env.json`. An environment description is a means of obtaining a runtime. The variant determines how far the meaning of the computation depends on the environment.

## Metadata

Metadata is descriptive information attached by whoever produced or later annotated the artifact. It never affects the computation: consumers MUST NOT let the presence, absence or content of metadata change execution semantics.

Metadata lives in root files matching `^meta(-[^/]+)?\.json$`, each holding a JSON array of metadata entries. The effective metadata is the concatenation of those files. `meta.json` comes first. The sidecars follow, in ascending lexicographic (byte order) filename order. Each file keeps its own entry order.

An entry is a JSON object.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Identifier of the entry. SHOULD be unique within the artifact. |
| `producer` | string | yes | Identifier of the tool or system that wrote the entry. |
| `producer_version` | string | yes | Version of that producer. |
| `producer_tags` | array of string | yes | Tags describing the entry's payload, possibly empty. |
| `payload` | object | yes | The metadata itself, whose shape is defined by whoever writes it. |

Additional top-level keys are permitted. Consumers MUST ignore keys they do not understand, and MUST preserve them when they rewrite an entry. An entry that lacks a required key does not conform. Consumers MAY ignore such an entry. They MUST NOT let it abort the reading of the remaining metadata.

*Note: the manifest names its producer with `producer_name`, a metadata entry with `producer`. The inconsistency is historical.*

An entry MAY have auxiliary files under `meta_artifacts/<entry-id>/`, where `<entry-id>` is that entry's `id`. An id used this way MUST be unique within the artifact. The id becomes a path segment, so it MUST be path-safe. It MUST NOT contain `/`. It MUST NOT be `.` or `..`. It MUST NOT be empty. Consumers MUST ignore a `meta_artifacts/` subdirectory with no corresponding entry.

Ids are often derived from a tag or another identifier that contains path-unsafe characters. The convention replaces `:` with `~c~` and `/` with `~s~`. It then joins the escaped prefix to a unique suffix with the separator `~~et~~`, giving `<escaped-prefix>~~et~~<unique-suffix>`. The suffix is typically 32 hexadecimal characters from a UUID.

The embedded prefix is a discovery aid. An artifact can carry many metadata files, and an entry's auxiliary files live under a directory named by its id. A reader that lists `meta_artifacts/` can find an entry from the directory name alone, without opening any metadata file.

*Note: this escaping is not reversible in general, because `~` itself is not escaped. Consumers SHOULD treat entry ids as opaque identifiers and select entries by inspecting the entry object, in particular its `producer_tags`.*

## Tags

Tags appear in `manifest.producer_tags`, in the `tags` field of manifest IO entries and of Vars, and in `producer_tags` on metadata entries. A producer uses them to mark content as belonging to a category that some consumer knows how to handle.

A tag is an opaque string compared by exact equality. Consumers MUST NOT attach meaning to a prefix, a suffix, a substring or a separator inside a tag. Consumers MUST NOT case-fold before comparing. Consumers MUST ignore tags they do not recognize.

The RECOMMENDED grammar for tags intended to be recognized outside the producing system is `<namespace>::<name>`, as in `example.org::model_card:v1`. `<namespace>` names a web resource that the tag's definer controls. The resource can be a domain. It can also be a path under a shared host, such as a `github.com/<owner>/<repo>` repository, which is easier to obtain than a domain and is still uniquely controlled. `<name>` identifies the meaning. Qualifiers separated by `:` MAY follow the name. A version such as `v<N>` is the suggested first qualifier. Tags are compared exactly, so a changed meaning needs a new tag rather than a redefinition.

This specification standardizes the grammar and the comparison rule only. It never assigns meaning to a particular tag.

## Manifest patches

A manifest can be amended after production, without rewriting `manifest.json`. Root files matching `^manifest-[^/]+\.patch\.json$` each hold a patch document as defined by [RFC 6902](https://www.rfc-editor.org/rfc/rfc6902), that is a JSON array of operation objects.

Only the `add` and `replace` operations are permitted. Producers and appenders MUST NOT emit `remove`, `move`, `copy` or `test`. Consumers MUST reject a patch document that contains them, rather than skip the offending operation. Pointers MUST be absolute JSON Pointers into the manifest document, beginning with `/`.

The effective manifest is `manifest.json` with every patch applied. Consumers apply the patch files in ascending lexicographic (byte order) filename order. Within one file, they apply the operations in document order. The result MUST be a valid manifest. Every consumer that reads the manifest reads the effective manifest.

## Extending an artifact

Append-extensibility lets a system annotate a model it did not produce. An appender MUST only add new members at the end of the archive. An appender MUST NOT modify, remove or reorder existing members. A directory-form artifact is extended by adding files, under the same rule.

If an appended member repeats an existing member name, the last occurrence wins. This is the only way an append can supersede earlier content. Appenders SHOULD add new members rather than shadow existing ones, so that the original content stays recoverable.

Appending metadata is the canonical extension mechanism. An appender adds a root sidecar named `meta-<suffix>.json` holding a JSON array of entries, plus any auxiliary blobs under `meta_artifacts/<entry-id>/`. `<suffix>` SHOULD be globally unique. The established choice is 32 hexadecimal characters from a UUID. Appenders MUST NOT rewrite `meta.json` or an existing sidecar. An appender amends the manifest the same way, by appending a `manifest-<suffix>.patch.json` file.

*Note: every append changes the bytes of the artifact file and invalidates any digest over the whole file. A system that identifies artifacts by content hash must hash after the last append.*

## Unknown content

Consumers MUST ignore members that this specification, the artifact's variant and its operation types do not define, rather than treat them as errors. This covers extra root files, extra directories, and extra files inside reserved directories. Consumers SHOULD likewise ignore object keys they do not understand instead of rejecting the containing document, except where a construct's definition states otherwise.

An extension takes one of two forms. It mints a new versioned name: a variant, an operation type, an environment kind, a content type, a dtype or a tag. Or it adds members and metadata to an artifact. An extension MUST NOT add flags to an established construct, and MUST NOT reinterpret an existing file. A consumer that predates an extension can still read the artifact, but cannot act on the extension.

## Variant dispatch

`manifest.variant` names the variant. The variant decides everything Core leaves open:

- the schema and meaning of `variant_config.json`
- the layout of `variant_artifacts/`
- whether the artifact may declare operation instances, and how it invokes them
- how manifest inputs and outputs are bound
- which content types are permitted

The [variants](variants/README.md) index the defined variants.

A consumer that does not implement the named variant MUST report the artifact as unsupported and MUST name the variant it did not recognize. It MUST NOT execute the artifact on a guess, fall back to another variant, or infer an execution model from the artifact's contents. It MAY still read what Core defines, because the effective manifest, the dtypes, the metadata and the tags are variant-independent.

The other named constructs differ in role only. An unrecognized operation type makes the artifact unexecutable. An unrecognized `env.json` key is one alternative among several, and the consumer ignores it. A consumer MUST NOT approximate a construct it does not implement. An artifact that a consumer cannot execute correctly MUST fail visibly instead of producing results of unknown meaning.
