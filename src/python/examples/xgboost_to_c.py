"""Hands-on demo: XGBoost model -> ONNX-ML -> self-contained C99 header.

Trains a small XGBoost regressor and classifier, converts both to ONNX with onnxmltools,
and compiles them with ``fnnx.extras.compilers.c`` into single C headers -- gradient
boosted trees as straight-line C, with no runtime, no allocation and no ONNX at inference
time. Each artifact is then built into a shared library and checked against XGBoost's own
predictions, and the classifier is additionally driven from a plain C program.

Usage (from src/python; needs xgboost, onnxmltools, onnx, numpy and a C compiler):

    pip install onnxmltools
    python examples/xgboost_to_c.py

Everything is written under ./_fnnx_xgb_demo/:

    _fnnx_xgb_demo/reg/xgb_reg.h    the regressor, `xgb_reg_run()`
    _fnnx_xgb_demo/reg5/xgb_reg5.h  the same regressor, re-encoded for ai.onnx.ml 5
    _fnnx_xgb_demo/clf/xgb_clf.h    the classifier, `xgb_clf_run()`
    _fnnx_xgb_demo/clf/main.c       a C program that includes it, built with cc

Both models compile to `ai.onnx.ml` kernels: `TreeEnsembleRegressor` and
`TreeEnsembleClassifier` become a table-driven tree walk, plus the post-transform the
classifier needs. The converters still emit the opset-1 encoding, so section 3 rewrites the
regressor into the opset-5 `TreeEnsemble` that replaced the pair, and checks that both
compile to the same thing. To package a model as an FNNX bundle instead of a bare `.onnx`
file -- and to get a per-node entrypoint per stage -- see `torch_to_c.py`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import xgboost
from onnx import ModelProto, NodeProto, TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from fnnx.extras.compilers.c import compile_onnx

ML_DOMAIN = "ai.onnx.ml"

OUT_DIR = Path.cwd() / "_fnnx_xgb_demo"
FEATURES = 4
SAMPLES = 400
TREES = 12
MAX_BATCH = 32
SEED = 20260727


# --------------------------------------------------------------------------------------
# 1. The models
# --------------------------------------------------------------------------------------


def training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(SAMPLES, FEATURES)).astype(np.float32)
    target = 2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
    y = (target + rng.normal(scale=0.1, size=SAMPLES)).astype(np.float32)
    return x, y, (y > 0).astype(np.int64)


def train() -> tuple[xgboost.XGBRegressor, xgboost.XGBClassifier, np.ndarray]:
    x, y, labels = training_data()
    settings = {"n_estimators": TREES, "max_depth": 3, "random_state": SEED}
    regressor = xgboost.XGBRegressor(**settings).fit(x, y)
    classifier = xgboost.XGBClassifier(**settings).fit(x, labels)
    return regressor, classifier, x


# --------------------------------------------------------------------------------------
# 2. XGBoost -> ONNX
# --------------------------------------------------------------------------------------


def to_onnx(model: Any) -> ModelProto:
    """Convert one fitted XGBoost model, with the row axis exported under a name.

    `FloatTensorType([None, FEATURES])` -- the usual spelling -- leaves that axis
    *anonymous*, and the compiler has no name to bind: it pins the axis to 1 and the
    artifact serves single rows only. A string dimension is what makes `batch` addressable
    by `runtime_dims` / `dim_bindings` further down.
    """
    return convert_xgboost(
        model, initial_types=[("x", FloatTensorType(["batch", FEATURES]))]
    )


def rename_tensor(model: ModelProto, old: str, new: str) -> ModelProto:
    """Rename a graph tensor in place; the converter calls every output `variable`."""
    for node in model.graph.node:
        node.input[:] = [new if name == old else name for name in node.input]
        node.output[:] = [new if name == old else name for name in node.output]
    for value in model.graph.output:
        if value.name == old:
            value.name = new
    return model


# --------------------------------------------------------------------------------------
# 3. ai.onnx.ml 1 -> ai.onnx.ml 5, by hand
# --------------------------------------------------------------------------------------

# Opset 5 replaced the `TreeEnsembleRegressor`/`TreeEnsembleClassifier` pair with a single
# `TreeEnsemble`, re-encoded around the walk rather than around the trees:
#
#   * interior nodes and leaves are two index spaces, not one. The legacy families
#     interleave `LEAF` entries among the branch entries and key everything by
#     (`nodes_treeids`, `nodes_nodeids`); v5 drops both id families, keeps only interior
#     nodes in `nodes_*`, and marks each child with `nodes_trueleafs`/`nodes_falseleafs`
#     to say which space its index addresses.
#   * `tree_roots` replaces the tree ids: one entry per tree, an index into `nodes_*`.
#   * leaf scores move from the (`target_treeids`, `target_nodeids`, `target_ids`,
#     `target_weights`) quadruple to `leaf_targetids` + `leaf_weights`, one entry per leaf.
#   * `nodes_modes` becomes a uint8 tensor of enumerators instead of strings, and
#     `nodes_values` becomes the `nodes_splits` tensor -- typed attributes, so a float64
#     ensemble no longer needs the `*_as_tensor` twins opset 3 had added.
#   * `base_values` is gone, as are `classlabels_*`; a classifier is now just a
#     `TreeEnsemble` with several targets and whatever post-transform the caller wants.
#
# The compiler reads both, and its own tables are close to the v5 form -- which is why
# `ops/tree.py` converts the legacy encoding into it rather than the other way round.

_MODES = {
    "BRANCH_LEQ": 0,
    "BRANCH_LT": 1,
    "BRANCH_GTE": 2,
    "BRANCH_GT": 3,
    "BRANCH_EQ": 4,
    "BRANCH_NEQ": 5,
}
_AGGREGATES = {"AVERAGE": 0, "SUM": 1, "MIN": 2, "MAX": 3}
_SUM = 1
_LEAF = "LEAF"


def node_attributes(node: NodeProto) -> dict[str, Any]:
    return {entry.name: helper.get_attribute_value(entry) for entry in node.attribute}


def to_opset5(model: ModelProto) -> ModelProto:
    """Re-encode a legacy `TreeEnsembleRegressor` graph as an opset-5 `TreeEnsemble`.

    Only what this example produces is handled: one regressor node, `NONE` post-transform,
    and no set-membership tests (the legacy encoding has none). Anything else raises rather
    than emitting an ensemble that scores differently.
    """
    legacy = model.graph.node[0]
    attributes = node_attributes(legacy)
    aggregate = _AGGREGATES[attributes.get("aggregate_function", b"SUM").decode()]
    modes = [mode.decode() for mode in attributes["nodes_modes"]]
    tree_ids = list(attributes["nodes_treeids"])
    node_ids = list(attributes["nodes_nodeids"])
    missing = list(attributes.get("nodes_missing_value_tracks_true", [0] * len(modes)))

    # The two index spaces, assigned in the order the legacy families list their entries.
    branches: dict[tuple[int, int], int] = {}
    leaves: dict[tuple[int, int], int] = {}
    for index, mode in enumerate(modes):
        space = leaves if mode == _LEAF else branches
        space[tree_ids[index], node_ids[index]] = len(space)

    def child(tree: int, node_id: int) -> tuple[int, int]:
        """The child's index, and whether that index addresses the leaf space."""
        key = (tree, node_id)
        return (leaves[key], 1) if key in leaves else (branches[key], 0)

    features: list[int] = []
    node_modes: list[int] = []
    splits: list[float] = []
    true_ids: list[int] = []
    false_ids: list[int] = []
    true_leafs: list[int] = []
    false_leafs: list[int] = []
    tracks: list[int] = []
    for index, mode in enumerate(modes):
        if mode == _LEAF:
            continue
        tree = tree_ids[index]
        true_id, true_leaf = child(tree, attributes["nodes_truenodeids"][index])
        false_id, false_leaf = child(tree, attributes["nodes_falsenodeids"][index])
        features.append(attributes["nodes_featureids"][index])
        node_modes.append(_MODES[mode])
        splits.append(attributes["nodes_values"][index])
        true_ids.append(true_id)
        false_ids.append(false_id)
        true_leafs.append(true_leaf)
        false_leafs.append(false_leaf)
        tracks.append(missing[index])

    weights, targets = _leaf_scores(attributes, leaves)
    _fold_base_values(attributes, aggregate, leaves, weights, targets)

    roots = []
    for tree in sorted(set(tree_ids)):
        # The root is the tree's first entry, which is how the compiler reads it too.
        first = tree_ids.index(tree)
        if modes[first] == _LEAF:
            raise ValueError(
                f"Tree {tree} is a single leaf; v5 spells that as a root whose two "
                "children are the same leaf, which this example does not emit."
            )
        roots.append(branches[tree, node_ids[first]])

    ensemble = helper.make_node(
        "TreeEnsemble",
        [legacy.input[0]],
        [legacy.output[0]],
        domain=ML_DOMAIN,
        n_targets=int(attributes["n_targets"]),
        aggregate_function=aggregate,
        post_transform=0,
        tree_roots=roots,
        leaf_targetids=targets,
        leaf_weights=_float_tensor("leaf_weights", weights),
        nodes_featureids=features,
        nodes_truenodeids=true_ids,
        nodes_falsenodeids=false_ids,
        nodes_trueleafs=true_leafs,
        nodes_falseleafs=false_leafs,
        nodes_missing_value_tracks_true=tracks,
        nodes_splits=_float_tensor("nodes_splits", splits),
        nodes_modes=helper.make_tensor(
            "nodes_modes", TensorProto.UINT8, [len(node_modes)], node_modes
        ),
    )
    return helper.make_model(
        helper.make_graph(
            [ensemble],
            model.graph.name,
            list(model.graph.input),
            list(model.graph.output),
        ),
        opset_imports=[helper.make_opsetid(ML_DOMAIN, 5)],
    )


def _leaf_scores(
    attributes: dict[str, Any], leaves: dict[tuple[int, int], int]
) -> tuple[list[float], list[int]]:
    """`leaf_weights` and `leaf_targetids`, from the legacy `target_*` quadruple."""
    weights = [0.0] * len(leaves)
    targets = [0] * len(leaves)
    for tree, node_id, target, weight in zip(
        attributes["target_treeids"],
        attributes["target_nodeids"],
        attributes["target_ids"],
        attributes["target_weights"],
    ):
        index = leaves[tree, node_id]
        weights[index] = weight
        targets[index] = target
    return weights, targets


def _fold_base_values(
    attributes: dict[str, Any],
    aggregate: int,
    leaves: dict[tuple[int, int], int],
    weights: list[float],
    targets: list[int],
) -> None:
    """Push the dropped `base_values` into the first tree's leaves.

    v5 has no such attribute. Under SUM the fold is exact -- and marginally *more* accurate
    than the legacy artifact, which adds the base once at the end in float32 while this
    carries it through the accumulation, exactly as XGBoost does.
    """
    base = list(attributes.get("base_values", []))
    if not any(base):
        return
    if aggregate != _SUM:
        raise ValueError(
            "`base_values` can only be folded into the leaves under SUM aggregation."
        )
    first = min(tree for tree, _ in leaves)
    for (tree, _), index in leaves.items():
        if tree == first:
            weights[index] += base[targets[index]]


def _float_tensor(name: str, values: list[float]) -> TensorProto:
    return helper.make_tensor(name, TensorProto.FLOAT, [len(values)], values)


# --------------------------------------------------------------------------------------
# 3. Compile, build, run
# --------------------------------------------------------------------------------------


def print_report(result: Any) -> None:
    report = result.report
    memory = report["memory"]
    runtime_dims = (
        ", ".join(f"{dim['name']}<={dim['max']}" for dim in report["runtime_dims"])
        or "none"
    )
    signature = ", ".join(
        f"{tensor['dtype']}{list(tensor['shape'])} {tensor['name']}"
        for tensor in report["entrypoint"]["inputs"] + report["entrypoint"]["outputs"]
    )
    print(
        f"  header:       {result.header_path} ({result.header_path.stat().st_size} B)"
    )
    print(f"  entrypoint:   {report['entrypoint']['symbol']}({signature})")
    print(f"  opsets:       {report['opsets']}")
    print(f"  runtime dims: {runtime_dims}")
    print(f"  kernels:      {', '.join(report['kernels'])}")
    print(
        f"  static mem:   {memory['static_bytes']} B "
        f"(weights {memory['weights_bytes']}, arena {memory['arena_bytes']})"
    )


def compare(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    print(f"  {label:<30} max |diff| = {np.abs(actual - expected).max():.3e}  OK")


def compare_exact(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)
    print(f"  {label:<30} {len(actual)} rows identical  OK")


# --------------------------------------------------------------------------------------
# 4. The classifier as a C program
# --------------------------------------------------------------------------------------

C_MAIN = """\
#define XGB_CLF_IMPLEMENTATION
#include "xgb_clf.h"
#include <stdio.h>

int main(void)
{{
    float x[XGB_CLF_INPUT_X_COUNT] = {{{values}}};
    int64_t label[XGB_CLF_OUTPUT_LABEL_COUNT];
    float probabilities[XGB_CLF_OUTPUT_PROBABILITIES_COUNT];

    if (xgb_clf_run({rows}, x, label, probabilities) != XGB_CLF_OK) {{
        return 1;
    }}
    for (int row = 0; row < {rows}; ++row) {{
        printf("%lld %.7f\\n", (long long) label[row], probabilities[row * 2 + 1]);
    }}
    return 0;
}}
"""


def run_as_c_program(directory: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a C program against the emitted header; returns what it prints per row.

    Nothing of FNNX, ONNX or XGBoost is involved past this point: the header, a C
    compiler, and libm for the logistic post-transform.
    """
    source = directory / "main.c"
    source.write_text(
        C_MAIN.format(
            rows=x.shape[0], values=", ".join(f"{value:.9g}f" for value in x.ravel())
        )
    )
    binary = directory / "main"
    flags = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-O2"]
    subprocess.run(["cc", *flags, str(source), "-o", str(binary), "-lm"], check=True)
    printed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    ).stdout.split("\n")
    rows = [line.split() for line in printed if line]
    return (
        np.array([int(row[0]) for row in rows], dtype=np.int64),
        np.array([float(row[1]) for row in rows], dtype=np.float32),
    )


def main() -> None:
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    regressor, classifier, x = train()
    batch = x[:MAX_BATCH]
    print(f"trained {TREES} trees per model on {SAMPLES}x{FEATURES} rows")

    print("\n=== XGBRegressor -> TreeEnsembleRegressor -> C ===")
    model = rename_tensor(to_onnx(regressor), "variable", "score")
    result = compile_onnx(
        model, OUT_DIR / "reg", runtime_dims={"batch": MAX_BATCH}, prefix="xgb_reg"
    )
    print_report(result)
    compiled = result.load()
    legacy_scores = compiled.run({"x": batch})["score"].ravel()
    print()
    compare("compiled vs xgboost", legacy_scores, regressor.predict(batch))
    # The same artifact serves any batch up to the maximum it was compiled for.
    compare(
        "same artifact, 5 rows",
        compiled.run({"x": x[:5]})["score"].ravel(),
        regressor.predict(x[:5]),
    )

    print("\n=== the same regressor re-encoded as ai.onnx.ml 5 `TreeEnsemble` ===")
    modern = to_opset5(model)
    modern_result = compile_onnx(
        modern, OUT_DIR / "reg5", runtime_dims={"batch": MAX_BATCH}, prefix="xgb_reg5"
    )
    print_report(modern_result)
    modern_scores = modern_result.load().run({"x": batch})["score"].ravel()
    print()
    # The reference evaluator checks the re-encoding itself, not just what the compiler
    # makes of it: a v5 graph this compiler mis-read would still have to score the same.
    evaluated = cast(
        list[np.ndarray], ReferenceEvaluator(modern).run(None, {"x": batch})
    )
    reference = evaluated[0].ravel()
    compare("opset 5 vs onnx reference", modern_scores, reference)
    compare("opset 5 vs xgboost", modern_scores, regressor.predict(batch))
    compare("opset 5 vs opset 1 artifact", modern_scores, legacy_scores)
    print(
        f"  {'same tables either way':<30} "
        f"{modern_result.report['memory']['static_bytes']} B vs "
        f"{result.report['memory']['static_bytes']} B, kernel "
        f"{modern_result.report['kernels'][0].split('_', 3)[-1]}"
    )

    print("\n=== XGBClassifier -> TreeEnsembleClassifier -> C ===")
    result = compile_onnx(
        to_onnx(classifier),
        OUT_DIR / "clf",
        runtime_dims={"batch": MAX_BATCH},
        prefix="xgb_clf",
    )
    print_report(result)
    compiled = result.load()
    outputs = compiled.run({"x": batch})
    print()
    compare_exact("labels vs xgboost", outputs["label"], classifier.predict(batch))
    compare(
        "probabilities vs xgboost",
        outputs["probabilities"],
        classifier.predict_proba(batch),
    )

    print("\n=== the classifier header on its own: a C program, cc, and libm ===")
    labels, positive = run_as_c_program(OUT_DIR / "clf", batch)
    compare_exact("C binary labels vs xgboost", labels, classifier.predict(batch))
    compare("C binary p(class 1) vs xgboost", positive, outputs["probabilities"][:, 1])

    print(
        "\nOn the command line, once the `.onnx` files are on disk:\n"
        f"  python -m fnnx.extras.compilers.c model.onnx -o {OUT_DIR / 'reg'} "
        f"--runtime-dim batch={MAX_BATCH} --prefix xgb_reg"
    )


if __name__ == "__main__":
    main()
