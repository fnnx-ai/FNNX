# The `python3::conda_pip` environment kind

| Key | Added in | Updated in | Schema |
| --- | --- | --- | --- |
| `python3::conda_pip` | 0.1.0 | 0.1.0 | [`env.json`](../schemas/env.json) |

`python3::conda_pip` declares a reconstructible Python environment: a Python version, and the packages that must be present before execution starts. It serves artifacts that need an environment the consumer does not already have. A provider is a consumer that reconstructs the environment from this declaration.

## Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `python_version` | string | yes | The Python version the artifact expects, as a version string such as `"3.11"` |
| `build_dependencies` | array of string | yes | Package specifications the environment provider installs before the pip dependencies |
| `dependencies` | array of PipDependency | yes | The pip dependencies of the artifact |
| `conda_channels` | array of string \| null | no | Package channels the provider draws `build_dependencies` from |

`build_dependencies` entries are opaque, conda-oriented package specification strings. A provider installs them alongside the interpreter, before any pip installation. A provider that supports channels draws them from `conda_channels`. The typical content is build backends such as `pip`, `setuptools` and `wheel`, pinned to the versions the artifact was produced with. Producers MUST write the array. It MAY be empty.

A PipDependency object describes one pip requirement.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `package` | string | yes | A pip requirement string, such as `"scikit-learn==1.5.0"` |
| `extra_pip_args` | string \| null | no | Additional command-line arguments to pass to pip for this requirement |
| `condition` | PipCondition \| null | no | Restricts the requirement to matching target environments |

A PipCondition object restricts a dependency to a subset of target environments. All three of its fields are optional arrays of strings.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `platform` | array of string \| null | no | Matched against the machine architecture of the target environment |
| `os` | array of string \| null | no | Matched against the operating system of the target environment: `linux`, `darwin` or `windows` |
| `accelerator` | array of string \| null | no | Matched against the accelerator selected for execution, defaulting to `cpu` |

## Condition matching

The accelerator is the execution hardware the consumer selects, named by an opaque string. The conditions on a dependency are ANDed: a provider installs the dependency only when every condition it declares matches the target environment. A condition that is absent, null or an empty array imposes no restriction. Matching is case-insensitive on both sides, and a value matches by membership in the declared array. Producers SHOULD declare the full architecture string rather than a prefix of one.

## Provisioning

Provisioning is provider-defined. A provider obtains an interpreter of the declared `python_version`. It installs `build_dependencies` from `conda_channels`. It then installs the matching `dependencies` with pip. How it names, caches or isolates the resulting environment is its own concern.

A provider MAY be unable to honour parts of a declaration. A provider that does not use conda has no equivalent of `build_dependencies` or `conda_channels`. A provider that installs dependencies as one set may have nowhere to apply a per-requirement `extra_pip_args`. A provider that cannot honour a field SHOULD report that to the caller instead of proceeding silently.

A provider that finds an absent or empty `dependencies` array SHOULD install nothing beyond what it needs to run the artifact itself. A provider that finds an absent `python_version`, which only a non-conforming artifact presents, SHOULD use the version of the interpreter it runs under. Neither substitution permits a producer to omit the field.

Producers SHOULD pin their dependencies exactly. Exact pins keep an artifact reconstructible longer than open ranges.

## Example

```json
{
    "python3::conda_pip": {
        "python_version": "3.11",
        "build_dependencies": [
            "pip==24.0",
            "setuptools==69.5.1",
            "wheel==0.43.0"
        ],
        "dependencies": [
            {"package": "scikit-learn==1.5.0"},
            {
                "package": "torch==2.3.0",
                "extra_pip_args": "--extra-index-url https://download.pytorch.org/whl/cu121",
                "condition": {
                    "os": ["linux"],
                    "platform": ["x86_64"],
                    "accelerator": ["cuda"]
                }
            },
            {
                "package": "torch==2.3.0",
                "condition": {"accelerator": ["cpu"]}
            }
        ],
        "conda_channels": ["conda-forge"]
    }
}
```

The two `torch` entries are mutually exclusive. A provider installs the first only on a Linux x86-64 target with a CUDA accelerator, and the second only when the selected accelerator is `cpu`.
