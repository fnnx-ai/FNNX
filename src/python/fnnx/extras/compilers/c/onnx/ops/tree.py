"""Decision forests: ONNX-ML's two legacy ensembles and the opset-5 op that supersedes them.

`TreeEnsembleRegressor`, `TreeEnsembleClassifier` and `TreeEnsemble` are three encodings of
one computation: every row of `X` falls through a set of binary trees, and the weights of the
leaves it lands in are aggregated per target. What differs is the layout. The legacy pair name
a node by a `(tree, node)` pair, mark leaves with a `LEAF` mode among the branch tests, and
carry the leaf weights in a second family of attributes keyed by that same pair; opset 5
splits interior nodes from leaves into two families indexed directly, and moves the tests, the
splits and the weights into tensors. Both are normalized here into one flat form — nodes,
leaves, `(target, weight)` entries and roots — so a single walker serves all three ops, and
everything it reads is `static const` data laid out at compile time.

Three places the emitted code follows ONNX's reference implementation rather than the prose,
because the reference is the only oracle these ops have:

* A `BRANCH_NEQ` node of a *legacy* ensemble sends a NaN feature down the branch its
  `missing_value_tracks_true` flag names, while opset 5's `NEQ` sends it down the true branch
  outright — `NaN != split` being true. The two differ by one table entry, so the legacy test
  is emitted as a branch test of its own rather than the walker taking a flag.
* A set test reads its members off `membership_values` in the order the reference builds the
  trees in — depth first from each root, true branch before false — and a member that is zero
  ends the set early, which is what its `while (m := next(it)) and not isnan(m)` does.
* `TreeEnsembleClassifier` widens a one-class ensemble's scores to two columns and derives the
  first from the second; that binary rule, and the `argmax` over the result, are the
  reference's.

What the compiler refuses: the `*_as_tensor` attribute families opset 3 added for
double-precision tables. The reference evaluator stores them and then reads the float32
families anyway, so nothing could vouch for what a kernel built from them computes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Template

import numpy as np
from onnx.numpy_helper import to_array

from fnnx.extras.compilers.c.errors import CompileError
from fnnx.extras.compilers.c.onnx.dtypes import (
    c_type,
    element_type_name,
    numpy_dtype_name,
)
from fnnx.extras.compilers.c.onnx.emit import scalar_literal
from fnnx.extras.compilers.c.onnx.kernels import (
    CFunction,
    ConstantData,
    NodeContext,
    NodeEmission,
    TensorRef,
    constant_data,
    register_kernel,
)
from fnnx.extras.compilers.c.onnx.loader import ML_DOMAIN
from fnnx.extras.compilers.c.onnx.ops.axes import call_kernel, verify_shape
from fnnx.extras.compilers.c.onnx.ops.scores import (
    NONE,
    PROBIT,
    argmax_labels,
    binary_scores,
    choice,
    extend,
    float_output,
    label_output,
    named_transform,
    post_transform,
)

# The legacy pair changed twice after opset 1: at 3, which added the `*_as_tensor` families,
# and at 5, which deprecated them in favour of `TreeEnsemble` and changed nothing else. One
# generator therefore claims all three revisions, a claim
# `test_the_legacy_ensemble_revisions_are_one_op` holds to the schemas themselves.
_LEGACY_VERSIONS = (1, 3, 5)
_ENSEMBLE_VERSIONS = (5,)

# The branch tests, in the encoding opset 5 numbers them with. The legacy families name them
# as strings and add `LEAF`, which this compiler turns into a leaf rather than a test.
_BRANCH_TESTS = {
    "BRANCH_LEQ": 0,
    "BRANCH_LT": 1,
    "BRANCH_GTE": 2,
    "BRANCH_GT": 3,
    "BRANCH_EQ": 4,
    "BRANCH_NEQ": 5,
}
_LEAF = "LEAF"
_MEMBER_TEST = 6
# `BRANCH_NEQ` under the legacy NaN rule; see the module docstring.
_LEGACY_NEQ_TEST = 7

# Columns of one row of the emitted node table: the feature it tests, the test, the flags
# below, the two children, and the range of `membership_values` a set test reads.
_NODE_FIELDS = 7
_TRUE_IS_LEAF = 1
_FALSE_IS_LEAF = 2
_MISSING_TRACKS_TRUE = 4

# What the walker does with a leaf's weight. AVERAGE differs from SUM only in what is divided
# by the number of trees, which the two divisors carry.
_ACCUMULATE = 0
_MINIMUM = 1
_MAXIMUM = 2
_LEGACY_AGGREGATES = {
    "SUM": _ACCUMULATE,
    "AVERAGE": _ACCUMULATE,
    "MIN": _MINIMUM,
    "MAX": _MAXIMUM,
}
_ENSEMBLE_AGGREGATES = {0: _ACCUMULATE, 1: _ACCUMULATE, 2: _MINIMUM, 3: _MAXIMUM}

# Opset 5 numbers the transforms the way `ops.scores` does, so its lookup exists to reject a
# value ONNX does not define rather than to translate one.
_ENSEMBLE_TRANSFORMS = {value: value for value in range(5)}

# What is added to every target once the aggregation is done, where the node offsets none.
# Negative zero rather than positive is what makes the addition an identity for *every* value:
# `x + (-0.0) == x` holds for the negative zero a MIN fold can leave behind, where `x + 0.0`
# would flip its sign.
_NO_OFFSET = -0.0

_AGGREGATE_TEMPLATE = Template("""\
static void $name(
    $result* scores,
    const $element* features,
    const int32_t* nodes,
    const double* splits,
    const double* members,
    const int32_t* leaves,
    const int32_t* leaf_targets,
    const $result* leaf_weights,
    const int32_t* roots,
    const $result* initial,
    const $result* offset,
    size_t rows,
    size_t width,
    size_t tree_count,
    size_t target_count,
    int aggregate,
    $result weight_divisor,
    $result total_divisor)
{
    size_t row, tree, target;
    for (row = 0; row < rows; ++row) {
        $result* out = scores + row * target_count;
        const $element* in = features + row * width;
        for (target = 0; target < target_count; ++target) {
            out[target] = initial[target];
        }
        for (tree = 0; tree < tree_count; ++tree) {
            size_t index = (size_t)roots[tree * 2];
            int is_leaf = roots[tree * 2 + 1];
            size_t entry, last, member;
            while (!is_leaf) {
                const int32_t* node = nodes + index * $fields;
                const double value = (double)in[node[0]];
                const double split = splits[index];
                int taken = 0;
                switch (node[1]) {
                case 0: taken = value <= split; break;
                case 1: taken = value < split; break;
                case 2: taken = value >= split; break;
                case 3: taken = value > split; break;
                case 4: taken = value == split; break;
                case 5: taken = value != split; break;
                case $member:
                    for (member = 0; member < (size_t)node[6]; ++member) {
                        if (value == members[(size_t)node[5] + member]) {
                            taken = 1;
                        }
                    }
                    break;
                default:
                    /* The legacy encoding's BRANCH_NEQ, the one test that leaves a value
                       that is not a number to the missing-value rule below. */
                    taken = !isnan(value) && value != split;
                    break;
                }
                if (!taken && (node[2] & $missing) != 0 && isnan(value)) {
                    taken = 1;
                }
                is_leaf = taken ? (node[2] & $true_leaf) : (node[2] & $false_leaf);
                index = (size_t)(taken ? node[3] : node[4]);
            }
            entry = (size_t)leaves[index * 2];
            last = entry + (size_t)leaves[index * 2 + 1];
            for (; entry < last; ++entry) {
                const size_t chosen = (size_t)leaf_targets[entry];
                const double weight = (double)(leaf_weights[entry] / weight_divisor);
                if (aggregate == $accumulate) {
                    out[chosen] = ($result)((double)out[chosen] + weight);
                } else if (aggregate == $minimum) {
                    if (weight < (double)out[chosen]) {
                        out[chosen] = ($result)weight;
                    }
                } else if (weight > (double)out[chosen]) {
                    out[chosen] = ($result)weight;
                }
            }
        }
        for (target = 0; target < target_count; ++target) {
            out[target] = out[target] / total_divisor + offset[target];
        }
    }
}""")


@dataclass(frozen=True)
class _Ensemble:
    """A forest in the one flat form the walker reads, whatever encoding it arrived in.

    `nodes` holds `_NODE_FIELDS` columns per interior node and `roots` an `(index, is_leaf)`
    pair per tree; a child index addresses `nodes` or `leaves` according to the node's flags.
    Each leaf names a range of the `targets`/`weights` pair list, which is what lets one leaf
    of a multi-target legacy ensemble contribute to several targets at once.
    """

    nodes: np.ndarray
    splits: np.ndarray
    members: np.ndarray
    leaves: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    roots: np.ndarray

    @property
    def tree_count(self) -> int:
        return len(self.roots)


@dataclass(frozen=True)
class _Aggregation:
    """How the walker folds the leaves a row reaches into the score of each target."""

    mode: int
    weight_divisor: float
    total_divisor: float
    initial: np.ndarray
    offset: np.ndarray


# --------------------------------------------------------------------------------------
# The three ops
# --------------------------------------------------------------------------------------


def _tree_ensemble_regressor(context: NodeContext) -> NodeEmission:
    """Leaf weights aggregated per target, then averaged, offset and transformed."""
    source = context.require_input(0)
    result = float_output(context, 0)
    rows, width = _rows_and_width(context, source)
    targets = context.int_attribute("n_targets")
    verify_shape(context, result, (rows, targets))
    _refuse_tensor_tables(context, ("base_values", "nodes_values", "target_weights"))

    ensemble = _legacy_ensemble(context, "target", width, targets)
    declared = context.string_attribute("aggregate_function")
    mode = choice(context, "aggregate_function", declared, _LEGACY_AGGREGATES)
    base = _base_values(context, targets, result.elem_type)
    aggregation = _Aggregation(
        mode=mode,
        weight_divisor=1.0,
        # The reference divides the accumulated total by the number of trees and adds the
        # base values only afterwards, which is why they offset rather than seed the scores.
        total_divisor=float(ensemble.tree_count) if declared == "AVERAGE" else 1.0,
        initial=_seed(mode, targets, result.elem_type),
        offset=_filled(targets, _NO_OFFSET, result.elem_type) if base is None else base,
    )
    emission = _aggregate(context, source, result, ensemble, aggregation, rows, width)
    transform = named_transform(context)
    return extend(emission, post_transform(context, result, transform, rows, targets))


def _tree_ensemble_classifier(context: NodeContext) -> NodeEmission:
    """The same aggregation, then the class the winning column names."""
    source = context.require_input(0)
    labels = label_output(context, 0)
    scores = float_output(context, 1)
    rows, width = _rows_and_width(context, source)
    classes = _class_labels(context)
    binary = len({int(value) for value in context.attribute("class_ids", [])}) == 1
    # A one-class ensemble's scores are widened to the two columns the binary rule fills, and
    # the winning column is then the label itself.
    columns = 2 if binary and len(classes) == 1 else len(classes)
    table = (0, 1)[:columns] if len(classes) == 1 else classes
    verify_shape(context, labels, (rows,))
    verify_shape(context, scores, (rows, columns))
    _refuse_tensor_tables(context, ("base_values", "nodes_values", "class_weights"))

    ensemble = _legacy_ensemble(context, "class", width, len(classes))
    base = _base_values(context, len(classes), scores.elem_type)
    initial = _filled(columns, 0.0, scores.elem_type)
    if base is not None:
        initial[: len(classes)] = base
    aggregation = _Aggregation(
        mode=_ACCUMULATE,
        weight_divisor=1.0,
        total_divisor=1.0,
        initial=initial,
        offset=_filled(columns, _NO_OFFSET, scores.elem_type),
    )
    transform = named_transform(context)
    emission = _aggregate(context, source, scores, ensemble, aggregation, rows, width)
    if binary:
        emission = extend(
            emission, _binary_scores(context, scores, transform, rows, columns)
        )
    emission = extend(
        emission, post_transform(context, scores, transform, rows, columns)
    )
    return extend(
        emission,
        argmax_labels(
            context, labels, scores.expr, scores.elem_type, table, rows, columns
        ),
    )


def _tree_ensemble(context: NodeContext) -> NodeEmission:
    """The opset-5 op: the same forest, with interior nodes and leaves indexed apart."""
    source = context.require_input(0)
    result = context.require_output(0)
    rows, width = _rows_and_width(context, source)
    targets = context.int_attribute("n_targets")
    if result.elem_type != source.elem_type:
        raise CompileError(
            f"Node `{context.label}`: TreeEnsemble scores in the element type of its input, "
            f"but `{source.name}` is `{element_type_name(source.elem_type)}` and its output "
            f"`{result.name}` is `{element_type_name(result.elem_type)}`."
        )
    verify_shape(context, result, (rows, targets))

    ensemble = _opset5_ensemble(context, width, targets)
    declared = context.int_attribute("aggregate_function")
    mode = choice(context, "aggregate_function", declared, _ENSEMBLE_AGGREGATES)
    aggregation = _Aggregation(
        mode=mode,
        # Each weight is divided by the number of trees before it is added, which does not
        # round the way dividing the total once would.
        weight_divisor=float(ensemble.tree_count) if declared == 0 else 1.0,
        total_divisor=1.0,
        initial=_seed(mode, targets, result.elem_type),
        offset=_filled(targets, _NO_OFFSET, result.elem_type),
    )
    emission = _aggregate(context, source, result, ensemble, aggregation, rows, width)
    transform = choice(
        context,
        "post_transform",
        context.int_attribute("post_transform"),
        _ENSEMBLE_TRANSFORMS,
    )
    return extend(emission, post_transform(context, result, transform, rows, targets))


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------


def _aggregate(
    context: NodeContext,
    source: TensorRef,
    result: TensorRef,
    ensemble: _Ensemble,
    aggregation: _Aggregation,
    rows: int,
    width: int,
) -> NodeEmission:
    """The one call that walks every tree and folds the leaves it reaches into `result`."""
    element, value = c_type(source.elem_type), c_type(result.elem_type)
    name = f"{context.prefix}_tree_aggregate_{element}_{value}"
    definition = _AGGREGATE_TEMPLATE.substitute(
        name=name,
        element=element,
        result=value,
        fields=_NODE_FIELDS,
        member=_MEMBER_TEST,
        missing=_MISSING_TRACKS_TRUE,
        true_leaf=_TRUE_IS_LEAF,
        false_leaf=_FALSE_IS_LEAF,
        accumulate=_ACCUMULATE,
        minimum=_MINIMUM,
    )
    tables, symbols = _tables(context, ensemble, aggregation, result.elem_type)
    call = call_kernel(
        name,
        [
            result.expr,
            source.expr,
            *symbols,
            f"{rows}u",
            f"{width}u",
            f"{ensemble.tree_count}u",
            f"{len(aggregation.initial)}u",
            str(aggregation.mode),
            scalar_literal(aggregation.weight_divisor, result.elem_type),
            scalar_literal(aggregation.total_divisor, result.elem_type),
        ],
    )
    return NodeEmission(
        functions=(CFunction(name, definition),),
        statements=(call,),
        constants=tables,
    )


def _tables(
    context: NodeContext,
    ensemble: _Ensemble,
    aggregation: _Aggregation,
    elem_type: int,
) -> tuple[tuple[ConstantData, ...], tuple[str, ...]]:
    """Every table the walker reads, as static constant data, in call-site order.

    The weights are accumulated in the element type the scores are held in, so that is what
    they are laid out as. Nothing is lost by it: the legacy families carry float32 weights
    into a float32 result, and opset 5's are required by ONNX's own type inference to carry
    the element type of `X`, which is what its result is scored in.
    """
    built = [
        constant_data(context, role, values)
        for role, values in (
            ("nodes", ensemble.nodes),
            ("splits", ensemble.splits),
            ("members", ensemble.members),
            ("leaves", ensemble.leaves),
            ("leaf_targets", ensemble.targets),
            ("leaf_weights", ensemble.weights.astype(numpy_dtype_name(elem_type))),
            ("roots", ensemble.roots),
            ("initial", aggregation.initial),
            ("offset", aggregation.offset),
        )
    ]
    return tuple(data for data, _ in built), tuple(symbol for _, symbol in built)


def _binary_scores(
    context: NodeContext, scores: TensorRef, transform: int, rows: int, columns: int
) -> NodeEmission:
    """The second column a single-class-weight ensemble's score is paired with.

    Which one it gets is the reference implementation's rule: the complement of the score
    where the transform leaves the scale of a probability alone, its negation otherwise.
    """
    return binary_scores(
        context, scores, rows, columns, complement=transform in (NONE, PROBIT)
    )


# --------------------------------------------------------------------------------------
# Reading a legacy ensemble
# --------------------------------------------------------------------------------------


def _legacy_ensemble(
    context: NodeContext, role: str, width: int, targets: int
) -> _Ensemble:
    """The `(tree, node)`-keyed encoding, flattened.

    A node is addressed by its position in the `nodes_*` families, while `nodes_truenodeids`
    names a node *id* within the same tree, which is resolved here into that position; a
    tree's root is its first node in those families rather than the node whose id is zero.
    Leaves are given an index space of their own, and the `role_*` families — `target_*` for
    the regressor, `class_*` for the classifier — say what each of them contributes.
    """
    tests = [
        _branch_test(context, value) for value in context.attribute("nodes_modes", [])
    ]
    count = len(tests)
    tree_ids = _integers(context, "nodes_treeids", count)
    node_ids = _integers(context, "nodes_nodeids", count)
    features = _integers(context, "nodes_featureids", count)
    true_ids = _integers(context, "nodes_truenodeids", count)
    false_ids = _integers(context, "nodes_falsenodeids", count)
    missing = _integers(
        context, "nodes_missing_value_tracks_true", count, optional=True
    )
    splits = _floats(context, "nodes_values", count, optional=True)

    positions = {pair: index for index, pair in enumerate(zip(tree_ids, node_ids))}
    slots = _index_spaces(tests)
    entries = _leaf_entries(context, role, targets)

    nodes = []
    for index, test in enumerate(tests):
        if test == _LEAF:
            continue
        true_child, true_leaf = _resolve(
            context, positions, tests, slots, tree_ids[index], true_ids[index]
        )
        false_child, false_leaf = _resolve(
            context, positions, tests, slots, tree_ids[index], false_ids[index]
        )
        nodes.append(
            (
                _feature(context, features[index], width),
                _LEGACY_NEQ_TEST if test == "BRANCH_NEQ" else _BRANCH_TESTS[test],
                (_TRUE_IS_LEAF if true_leaf else 0)
                | (_FALSE_IS_LEAF if false_leaf else 0)
                | (_MISSING_TRACKS_TRUE if missing[index] else 0),
                true_child,
                false_child,
                0,
                0,
            )
        )

    leaves: list[tuple[int, int]] = []
    leaf_targets: list[int] = []
    leaf_weights: list[float] = []
    for index, test in enumerate(tests):
        if test != _LEAF:
            continue
        pairs = entries.get((tree_ids[index], node_ids[index]), ())
        leaves.append((len(leaf_targets), len(pairs)))
        leaf_targets.extend(target for target, _ in pairs)
        leaf_weights.extend(weight for _, weight in pairs)

    roots = []
    for tree_id in sorted(set(tree_ids)):
        position = tree_ids.index(tree_id)
        leaf = tests[position] == _LEAF
        roots.append((slots[position], int(leaf)))

    ensemble = _Ensemble(
        nodes=_table(nodes, np.int32, _NODE_FIELDS),
        splits=np.array(
            [splits[index] for index, test in enumerate(tests) if test != _LEAF],
            np.float64,
        ),
        members=np.zeros(0, np.float64),
        leaves=_table(leaves, np.int32, 2),
        targets=np.array(leaf_targets, np.int32),
        # The reference reads these tables as the float32 the attribute stores them in.
        weights=np.array(leaf_weights, np.float32),
        roots=_table(roots, np.int32, 2),
    )
    _verify_acyclic(context, ensemble)
    return ensemble


def _index_spaces(tests: Sequence[str]) -> list[int]:
    """Where each flat node lands once interior nodes and leaves are indexed apart."""
    slots = []
    interior = leaves = 0
    for test in tests:
        if test == _LEAF:
            slots.append(leaves)
            leaves += 1
        else:
            slots.append(interior)
            interior += 1
    return slots


def _resolve(
    context: NodeContext,
    positions: Mapping[tuple[int, int], int],
    tests: Sequence[str],
    slots: Sequence[int],
    tree_id: int,
    node_id: int,
) -> tuple[int, bool]:
    position = positions.get((tree_id, node_id))
    if position is None:
        raise CompileError(
            f"Node `{context.label}`: `{context.node.op_type}` names node {node_id} of tree "
            f"{tree_id} as a child, which none of its `nodes_*` entries defines."
        )
    return slots[position], tests[position] == _LEAF


def _leaf_entries(
    context: NodeContext, role: str, targets: int
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """The `(target, weight)` pairs each leaf contributes, keyed by `(tree, node)`."""
    tree_ids = _integers(context, f"{role}_treeids", None)
    node_ids = _integers(context, f"{role}_nodeids", len(tree_ids))
    ids = _integers(context, f"{role}_ids", len(tree_ids))
    weights = _floats(context, f"{role}_weights", len(tree_ids))
    entries: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for tree_id, node_id, target, weight in zip(tree_ids, node_ids, ids, weights):
        if not 0 <= target < targets:
            raise CompileError(
                f"Node `{context.label}`: `{role}_ids` names {target}, which is outside the "
                f"{targets} target(s) this node scores."
            )
        entries.setdefault((tree_id, node_id), []).append((target, weight))
    return entries


# --------------------------------------------------------------------------------------
# Reading an opset-5 ensemble
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Opset5Nodes:
    """The interior-node families of an opset-5 ensemble, read and length-checked."""

    tests: list[int]
    features: list[int]
    true_ids: list[int]
    false_ids: list[int]
    true_leafs: list[int]
    false_leafs: list[int]
    missing: list[int]
    splits: np.ndarray

    def __len__(self) -> int:
        return len(self.tests)

    def is_bare_leaf(self, root: int) -> bool:
        """Whether the tree rooted here is a single leaf, as the reference reads that.

        Both children being leaves *and* being the same one is what marks the degenerate
        tree, and the root's own position is then read as an index into the leaf families.
        """
        return bool(
            self.true_leafs[root]
            and self.false_leafs[root]
            and self.true_ids[root] == self.false_ids[root]
        )


def _opset5_ensemble(context: NodeContext, width: int, targets: int) -> _Ensemble:
    """The opset-5 encoding, which is already the flat form bar the membership ranges."""
    parsed = _opset5_nodes(context)
    weights = _required_tensor(context, "leaf_weights")
    leaf_targets = _integers(context, "leaf_targetids", len(weights))
    for target in leaf_targets:
        if not 0 <= target < targets:
            raise CompileError(
                f"Node `{context.label}`: `leaf_targetids` names {target}, which is outside "
                f"the {targets} target(s) this node scores."
            )
    roots = _roots(context, parsed, len(weights))
    members = _membership(context, parsed.tests)
    ranges = _membership_ranges(context, parsed, roots, members, len(weights))

    nodes = [
        (
            _feature(context, parsed.features[index], width),
            parsed.tests[index],
            (_TRUE_IS_LEAF if parsed.true_leafs[index] else 0)
            | (_FALSE_IS_LEAF if parsed.false_leafs[index] else 0)
            | (_MISSING_TRACKS_TRUE if parsed.missing[index] else 0),
            _child(
                context,
                parsed.true_ids[index],
                parsed.true_leafs[index],
                parsed,
                len(weights),
            ),
            _child(
                context,
                parsed.false_ids[index],
                parsed.false_leafs[index],
                parsed,
                len(weights),
            ),
            *ranges.get(index, (0, 0)),
        )
        for index in range(len(parsed))
    ]
    ensemble = _Ensemble(
        nodes=_table(nodes, np.int32, _NODE_FIELDS),
        splits=parsed.splits.astype(np.float64),
        members=np.array(members, np.float64),
        leaves=_table([(index, 1) for index in range(len(weights))], np.int32, 2),
        targets=np.array(leaf_targets, np.int32),
        weights=weights,
        roots=_table(roots, np.int32, 2),
    )
    _verify_acyclic(context, ensemble)
    return ensemble


def _opset5_nodes(context: NodeContext) -> _Opset5Nodes:
    splits = _required_tensor(context, "nodes_splits")
    tests = [
        _numbered_test(context, int(value))
        for value in _required_tensor(context, "nodes_modes")
    ]
    count = len(tests)
    if len(splits) != count:
        raise CompileError(
            f"Node `{context.label}`: `nodes_splits` holds {len(splits)} entry(s) where "
            f"{count} are described by the attributes beside it."
        )
    return _Opset5Nodes(
        tests=tests,
        features=_integers(context, "nodes_featureids", count),
        true_ids=_integers(context, "nodes_truenodeids", count),
        false_ids=_integers(context, "nodes_falsenodeids", count),
        true_leafs=_integers(context, "nodes_trueleafs", count),
        false_leafs=_integers(context, "nodes_falseleafs", count),
        missing=_integers(
            context, "nodes_missing_value_tracks_true", count, optional=True
        ),
        splits=splits,
    )


def _roots(
    context: NodeContext, parsed: _Opset5Nodes, leaves: int
) -> list[tuple[int, int]]:
    """Each tree's root, and whether it addresses the leaf families rather than the nodes."""
    resolved = []
    for root in _integers(context, "tree_roots", None):
        if not 0 <= root < len(parsed):
            raise CompileError(
                f"Node `{context.label}`: `tree_roots` names node {root}, which is outside "
                f"the {len(parsed)} node(s) this ensemble defines."
            )
        leaf = parsed.is_bare_leaf(root)
        if leaf and root >= leaves:
            raise CompileError(
                f"Node `{context.label}`: the tree rooted at node {root} is a single leaf, "
                f"which ONNX reads at that same position among the leaves — and this "
                f"ensemble defines {leaves} of them."
            )
        resolved.append((root, int(leaf)))
    return resolved


def _child(
    context: NodeContext, child: int, leaf: int, parsed: _Opset5Nodes, leaves: int
) -> int:
    limit = leaves if leaf else len(parsed)
    if not 0 <= child < limit:
        raise CompileError(
            f"Node `{context.label}`: TreeEnsemble names {'leaf' if leaf else 'node'} "
            f"{child} as a child, which is outside the {limit} it defines."
        )
    return child


def _membership(context: NodeContext, tests: Sequence[int]) -> list[float]:
    """`membership_values`, checked against the set tests it is supposed to describe."""
    tensor = context.attribute("membership_values", None)
    sets = sum(1 for test in tests if test == _MEMBER_TEST)
    if tensor is None:
        if sets:
            raise CompileError(
                f"Node `{context.label}`: TreeEnsemble has {sets} set test(s) and no "
                "`membership_values` saying what they test against."
            )
        return []
    values = [float(value) for value in to_array(tensor).reshape(-1)]
    terminators = sum(1 for value in values if math.isnan(value))
    if terminators != sets:
        raise CompileError(
            f"Node `{context.label}`: `membership_values` holds {terminators} "
            f"NaN-terminated set(s) for {sets} set test(s)."
        )
    return values


def _membership_ranges(
    context: NodeContext,
    parsed: _Opset5Nodes,
    roots: Sequence[tuple[int, int]],
    members: Sequence[float],
    leaves: int,
) -> dict[int, tuple[int, int]]:
    """Which slice of `membership_values` each set test reads.

    The sets are laid out in the order the reference implementation builds the trees in —
    depth first from each root, true branch before false — rather than in node order, so that
    is the traversal here. A set ends at the first NaN *or zero*, which is where the
    reference's own loop condition stops.

    This is the traversal `_verify_acyclic` makes over the normalized form, and it runs first,
    so it carries the same guards: without them a cycle would be an endless walk here, and a
    child naming a node the ensemble does not define an `IndexError`, rather than the compile
    errors they are once the nodes below are built.
    """
    ranges: dict[int, tuple[int, int]] = {}
    consumed = 0
    visited: set[int] = set()
    for root, leaf in roots:
        stack = [] if leaf else [root]
        while stack:
            index = stack.pop()
            if index in visited:
                raise _revisited(context, index)
            visited.add(index)
            if parsed.tests[index] == _MEMBER_TEST:
                start = consumed
                consumed = _end_of_set(context, members, consumed, index)
                ranges[index] = (start, consumed - start - 1)
            if not parsed.false_leafs[index]:
                stack.append(
                    _child(context, parsed.false_ids[index], 0, parsed, leaves)
                )
            if not parsed.true_leafs[index]:
                stack.append(_child(context, parsed.true_ids[index], 0, parsed, leaves))
    return ranges


def _end_of_set(
    context: NodeContext, members: Sequence[float], start: int, index: int
) -> int:
    """One past the terminator of the set beginning at `start`."""
    position = start
    while True:
        if position >= len(members):
            raise CompileError(
                f"Node `{context.label}`: `membership_values` runs out before the set node "
                f"{index} tests against is terminated."
            )
        value = members[position]
        position += 1
        if value == 0.0 or math.isnan(value):
            return position


# --------------------------------------------------------------------------------------
# Shared reading and validation
# --------------------------------------------------------------------------------------


def _verify_acyclic(context: NodeContext, ensemble: _Ensemble) -> None:
    """Refuse a forest whose nodes are not a tree, which the emitted walker would loop on.

    A node two parents reach is refused along with one that reaches itself: the walk would be
    the same, but the membership ranges an opset-5 node carries would not, since the reference
    reads a set of its own every time it builds that node.
    """
    nodes = ensemble.nodes.reshape(-1, _NODE_FIELDS).tolist()
    visited: set[int] = set()
    for root, leaf in ensemble.roots.reshape(-1, 2).tolist():
        stack = [] if leaf else [root]
        while stack:
            index = stack.pop()
            if index in visited:
                raise _revisited(context, index)
            visited.add(index)
            row = nodes[index]
            for child, flag in ((row[3], _TRUE_IS_LEAF), (row[4], _FALSE_IS_LEAF)):
                if not row[2] & flag:
                    stack.append(child)


def _revisited(context: NodeContext, index: int) -> CompileError:
    return CompileError(
        f"Node `{context.label}`: node {index} is reachable more than once; the C compiler "
        "serves ensembles whose nodes form a tree."
    )


def _feature(context: NodeContext, feature: int, width: int) -> int:
    if not 0 <= feature < width:
        raise CompileError(
            f"Node `{context.label}`: `nodes_featureids` names feature {feature}, which is "
            f"outside the {width} column(s) of its input."
        )
    return feature


def _class_labels(context: NodeContext) -> tuple[int, ...]:
    """The class values a classifier labels its rows with, of the two families it may set."""
    integers = tuple(
        int(value) for value in context.attribute("classlabels_int64s", [])
    )
    strings = list(context.attribute("classlabels_strings", []))
    if bool(integers) == bool(strings):
        raise CompileError(
            f"Node `{context.label}`: TreeEnsembleClassifier must set exactly one of "
            "`classlabels_int64s` and `classlabels_strings`."
        )
    if strings:
        raise CompileError(
            f"Node `{context.label}`: TreeEnsembleClassifier labels its rows with the "
            "strings in `classlabels_strings`, and a tensor of STRING at run time is not "
            "something the C compiler supports."
        )
    if integers == (1,):
        return integers
    if len(integers) == 1:
        raise CompileError(
            f"Node `{context.label}`: TreeEnsembleClassifier declares the single class "
            f"{integers[0]}, which ONNX's own reference implementation refuses for any value "
            "but 1; nothing says what such a row should be labelled."
        )
    return integers


def _base_values(
    context: NodeContext, targets: int, elem_type: int
) -> np.ndarray | None:
    """`base_values` stretched over the targets it applies to, or None where none are set.

    A single value covers every target, which is how the reference broadcasts the attribute
    over the score matrix.
    """
    values = [float(value) for value in context.attribute("base_values", [])]
    if not values:
        return None
    if len(values) not in (1, targets):
        raise CompileError(
            f"Node `{context.label}`: `base_values` holds {len(values)} value(s) for "
            f"{targets} target(s); it takes either one value per target or a single value "
            "for all of them."
        )
    dtype = numpy_dtype_name(elem_type)
    return np.broadcast_to(np.array(values, dtype), (targets,)).astype(dtype)


def _seed(mode: int, targets: int, elem_type: int) -> np.ndarray:
    """What each target's score starts at: zero, or the extreme a fold runs down from."""
    if mode == _ACCUMULATE:
        return _filled(targets, 0.0, elem_type)
    info = np.finfo(numpy_dtype_name(elem_type))
    return _filled(
        targets, float(info.max if mode == _MINIMUM else info.min), elem_type
    )


def _filled(count: int, value: float, elem_type: int) -> np.ndarray:
    return np.full(count, value, numpy_dtype_name(elem_type))


def _refuse_tensor_tables(context: NodeContext, families: Sequence[str]) -> None:
    """Refuse the `*_as_tensor` families, which ONNX's reference implementation never reads.

    Opset 3 added them so that a double-precision ensemble need not round its tables to
    float32. The reference evaluator stores them and goes on reading the float32 families, so
    a kernel built from them would be answerable to nothing; the model is refused instead.
    """
    named = [
        family
        for family in families
        if context.attribute(f"{family}_as_tensor", None) is not None
    ]
    if named:
        raise CompileError(
            f"Node `{context.label}`: `{named[0]}_as_tensor` is not supported by the C "
            "compiler — ONNX's own reference implementation ignores the `*_as_tensor` "
            f"families and reads `{named[0]}` instead, so nothing can vouch for a kernel "
            "built from them; re-export the model with the float32 tables."
        )


def _rows_and_width(context: NodeContext, source: TensorRef) -> tuple[int, int]:
    """How many rows an ensemble reads from `X`, and how many features each of them holds."""
    if len(source.shape) == 2:
        return source.shape[0], source.shape[1]
    if len(source.shape) == 1:
        return 1, source.shape[0]
    raise CompileError(
        f"Node `{context.label}`: `{context.node.op_type}` reads an `[N, F]` matrix of "
        f"features, and `{source.name}` has shape {list(source.shape)}."
    )


def _branch_test(context: NodeContext, value: object) -> str:
    test = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if test != _LEAF and test not in _BRANCH_TESTS:
        raise CompileError(
            f"Node `{context.label}`: `nodes_modes` holds `{test}`, which is none of the "
            f"branch tests ONNX defines ({', '.join(sorted(_BRANCH_TESTS))}, {_LEAF})."
        )
    return test


def _numbered_test(context: NodeContext, test: int) -> int:
    if not 0 <= test <= _MEMBER_TEST:
        raise CompileError(
            f"Node `{context.label}`: `nodes_modes` holds {test}, which is none of the "
            f"branch tests ONNX numbers 0 to {_MEMBER_TEST}."
        )
    return test


def _integers(
    context: NodeContext, name: str, count: int | None, *, optional: bool = False
) -> list[int]:
    values = [int(value) for value in context.attribute(name, [])]
    if optional and not values:
        return [0] * (count or 0)
    if count is not None and len(values) != count:
        raise CompileError(
            f"Node `{context.label}`: `{name}` holds {len(values)} entry(s) where {count} "
            "are described by the attributes beside it."
        )
    return values


def _floats(
    context: NodeContext, name: str, count: int, *, optional: bool = False
) -> list[float]:
    values = [float(value) for value in context.attribute(name, [])]
    if optional and not values:
        return [0.0] * count
    if len(values) != count:
        raise CompileError(
            f"Node `{context.label}`: `{name}` holds {len(values)} entry(s) where {count} "
            "are described by the attributes beside it."
        )
    return values


def _required_tensor(context: NodeContext, name: str) -> np.ndarray:
    tensor = context.attribute(name, None)
    if tensor is None:
        raise CompileError(
            f"Node `{context.label}`: TreeEnsemble requires the `{name}` attribute."
        )
    return np.ascontiguousarray(to_array(tensor).reshape(-1))


def _table(rows: Sequence[Sequence[int]], dtype: type, columns: int) -> np.ndarray:
    return np.array(list(rows), dtype).reshape(len(rows), columns)


register_kernel(
    ML_DOMAIN, "TreeEnsembleRegressor", _LEGACY_VERSIONS, _tree_ensemble_regressor
)
register_kernel(
    ML_DOMAIN, "TreeEnsembleClassifier", _LEGACY_VERSIONS, _tree_ensemble_classifier
)
register_kernel(ML_DOMAIN, "TreeEnsemble", _ENSEMBLE_VERSIONS, _tree_ensemble)
