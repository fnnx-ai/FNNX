# FNNX Compatibility Model

FNNX distinguishes three separate properties of every published construct. Semantic stability is whether the normative meaning of a construct can change. Portability is how realistically an independent implementation can support it. Execution durability is how strongly FNNX expects historical artifacts to remain executable.

The three properties are orthogonal. A construct can be precisely specified and still be impractical to reimplement. It can be portable and still depend on an external service. It can stay understandable long after runtimes stop executing it. Each construct therefore carries an explicit judgment on each property that applies to it.

A stable FNNX artifact MUST NOT become ambiguous. A future runtime can lose the ability to execute an artifact. Even then, a consumer MUST be able to determine what the artifact means, which versioned constructs it requires, and why the runtime cannot execute it. Loss of execution support is acceptable. Loss of meaning is not. Constructs therefore carry versioned names, and consumers reject unrecognized constructs explicitly instead of approximating them.

## Specification maturity

Every FNNX construct carries an explicit maturity state. The state determines what the specification guarantees about the construct.

An **Experimental** construct MAY change incompatibly, and MAY be removed. The guarantees that apply to stable constructs do not cover artifacts that depend on experimental semantics. The state lets a new variant, operation or representation change before FNNX commits to permanent semantics.

A **Stable** construct has a normative meaning that is immutable within its published version. Any incompatible semantic change MUST be published as a new versioned construct under a different name. Once Stable, `ONNX_v1` never acquires different semantics. A stable construct MUST NOT normatively depend on experimental semantics.

A **Deprecated** construct remains semantically stable and retains the guarantees it had while Stable. Producers SHOULD NOT emit it in new artifacts. Deprecation changes the recommendation only. It MUST NOT redefine the construct.

A construct that has not been designated Stable MAY change in place, under its existing name. Such a change advances the specification version and the construct's Updated in entry.

## Portability

Portability describes whether different implementations can support a construct from an established execution contract. The test is whether an independent consumer that is correct but slow and unoptimized is reasonably implementable. Optimization is outside the test. Reproducing a production system such as ONNX Runtime, a set of CUDA kernels or a graph optimizer is also outside it.

**Strong** portability requires two things. A mature formal specification, or a stable and widely accepted de-facto standard, must back the execution contract. A correct unoptimized independent implementation must also be achievable without a particular opaque implementation. Multiple complete implementations are not required.

**Medium** portability applies where independent implementations are realistic, but the execution contract lacks broad standardization. The semantics are spread across de-facto conventions, or they rely on a reference implementation, or no canonical semantic contract exists.

**Weak** portability applies where correct execution depends on a particular implementation or environment, or where a compatible independent consumer is impractical to implement.

*Note: ONNX is Strong. ONNX itself is the shared contract, even though ONNX Runtime is its most complete implementation. An independent minimal executor for a bounded set of ONNX operators is reasonably implementable.*

*Note: A rigorously specified programming language can still be Weak. Python is well documented, but an FNNX consumer cannot be required to implement a compatible Python interpreter, object model, runtime and surrounding package environment. An arbitrary packaged Python function is therefore not Strongly portable.*

## Execution durability

Execution durability describes FNNX's intended long-term support for historical artifacts. It states an intent about the specification's own evolution, not a prediction that any implementation, service, hardware platform, dependency or company will exist indefinitely.

**Strong** execution durability means historical artifacts are intended to remain executable indefinitely. Future conforming implementations are expected to preserve execution directly, or through semantics-preserving migration or conversion. This is the intended contract for the `pipeline` variant.

**Medium** execution durability means historical artifacts remain permanently well defined and compatibility remains explicitly decidable, but future runtimes MAY retire execution support. A runtime that declines to execute an artifact for an identifiable, stated reason is an acceptable outcome under this grade.

**Weak** execution durability means execution is expected only while a compatible implementation or execution environment remains available or reproducible. Future implementations are not expected to reconstruct an obsolete environment. This grade fits `pyfunc` and implementation-oriented constructs generally.

*Note: FNNX does not promise that ONNX Runtime will exist forever. The Strong durability intent for `pipeline` is narrower. Historical stable operations stay part of the long-term execution contract. Whichever implementation satisfies that contract at the time executes them.*

## External dependencies

An external dependency is a resource that execution requires outside the artifact and the runtime: an HTTP service, a file, a database, a versioned remote API. This dimension is orthogonal to portability and execution durability. A construct MAY be Strong on both axes and still depend on external state. Failure of the external resource is an execution failure, and the artifact and the runtime can describe it. Such a failure does not make the operation non-portable or non-durable.

Stable FNNX semantics MUST reference a frozen contract. An operation MUST NOT be defined to mean whatever an external API happens to mean at the time it is called. Where a construct depends on an external interface, the version or revision of that interface is part of the construct's normative definition. The meaning of the artifact then stays determinable, whatever the resource does now.

## Composition and guarantee closure

A composite construct cannot provide stronger guarantees than the normative dependencies required to execute it. Guarantees close downward over composition, and the closure rules are normative.

A Strong construct composed only of Strong dependencies MAY remain Strong. A Strong construct that normatively depends on a Medium construct is at most Medium. A Strong construct that normatively depends on a Weak construct is at most Weak. A construct that normatively depends on an Experimental construct MUST NOT be Stable.

A construct MUST NOT be published as Strong when its implementation hides a weaker execution contract. An operation can be specified in portable terms and still have one realistic execution path through a particular opaque implementation. Its portability grade is then the grade of that path, not the grade of the description.

## Current classification

| Variant | Portability | Execution durability |
| --- | --- | --- |
| `pipeline` | Strong | Strong |
| `pyfunc` | Weak | Weak |

The table omits semantic stability. Semantic stability is mandatory and immutable for every Stable construct, so constructs do not differ on it.

`pyfunc` packages an implementation instead of a specified computation. Its FNNX packaging semantics are stable, but its portability stays Weak: an FNNX consumer cannot be required to recreate a compatible Python execution environment. Its execution durability depends on how far that environment can be reconstructed, which makes it Weak rather than Medium. Producers that can express a model as a `pipeline` SHOULD do so.

*Note: The diagonal in this table is not a rule. An operation can call an external service and still be Strong on both axes. It must be mature, and the contract of the service must be frozen.*

## The operation set

FNNX defines its operations in one first-party set, indexed in [`ops/`](ops/README.md). An operation is defined independently of any variant. A variant declares which operations it admits.

The set holds exactly one operation. The set is not a registry that awaits new entries. FNNX does not add an operation for feature coverage or for symmetry with other formats. It adds one only for a stated need.

A stable first-party operation MUST have stable, immutable, versioned semantics. It MUST have Strong portability and Strong execution durability. A mature formal or de-facto execution contract MUST back it. It MUST admit a realistic path to a correct unoptimized independent implementation. It MUST NOT normatively depend on any Medium, Weak or Experimental construct. `ONNX_v1` satisfies these conditions, and it is the only initial operation.

The Strong grades of `ONNX_v1` cover op instances that declare only the standard operator set domains. [`ops/onnx_v1.md`](ops/onnx_v1.md) names them. An op instance that declares another domain is outside these guarantees.

FNNX does not duplicate computation that ONNX already represents. First-party operations equivalent to ONNX operators such as Cast, Concat, MatMul, Resize or Softmax MUST NOT be added because they are easy to define. Such duplication creates several representations of one computation, and it enlarges the compatibility surface permanently. A new operation addresses an abstraction or an execution contract that the existing Strong operations cannot express.

The `pipeline` variant composes these operations into a graph, and admits every operation in the set. Its Strong grades therefore rest on the whole set staying Strong. The variant is not named `onnx`, because ONNX is one operation inside the graph and not the identity of the variant. The composition semantics do not depend on ONNX: the value namespace and the binding rules are defined in [`variants/pipeline.md`](variants/pipeline.md).

## Glossary

| Term | Definition |
| --- | --- |
| Stable semantics | A published version never changes its normative meaning incompatibly. |
| Strong portability | A mature formal or de-facto execution contract exists, and a correct unoptimized independent consumer is reasonably implementable. |
| Medium portability | Independent implementations are realistic, but no mature or commonly agreed execution contract exists yet. |
| Weak portability | Correct execution depends on a particular implementation or environment, or a compatible independent consumer is impractical to implement. |
| Strong execution durability | Preserving execution of historical artifacts is an explicit long-term compatibility goal. |
| Medium execution durability | Historical artifacts remain unambiguous and compatibility remains explicitly decidable, but runtime execution support may be retired. |
| Weak execution durability | Execution is expected only while a compatible implementation or environment remains available or reproducible. |
| External dependency | Execution requires a resource or service outside the artifact and the runtime; orthogonal to portability and execution durability. |
