"""Weighted hierarchical cross-entropy (WHXE) loss (CLAUDE.md §4, §6 step 7:
"taxonomy.py + WHXE loss <- swap head, keep everything else").

Villar, de Soto & Gagliano 2023, "Hierarchical Cross-entropy Loss for
Classification of Astrophysical Transients," arXiv:2312.02266 (NeurIPS 2023),
Section 2, eqs. 1-5. A class `c` is factorized along its root-to-leaf path as
a product of conditional probabilities, one per level, each a softmax over
that level's sibling group (eq. 2-3). The loss is a weighted sum of the
per-level negative log conditional probabilities (eq. 4):

    L_WHXE(p, c) = -sum_{h=0}^{H-1} W(c^(h)) * lambda(c^(h)) * log p(c^(h)|c^(h+1))

`c^(0)` is the leaf, `c^(H)` is the root; the root's own probability is fixed
at 1 ("every event is an astrophysical object") and excluded from the sum --
exactly `taxonomy.ROOT_ID`'s exclusion, already noted in `taxonomy.parents()`'s
own docstring before this module existed.

W(c) = N_all / (N_labels * N_c) reweights each node by its class fraction
(eq. 4's text); N_c for an internal node is its subtree total, which is why
"parent classes have a lower weighting than leaf nodes" (paper, Sec 2) --
see `compute_node_class_weights`.

lambda(c) = exp(-alpha * depth(c)) is a second, structural weighting that
emphasizes different tree levels (eq. 5, attributed to Bertinetto et al.
2020).

**Depth convention -- settled, with justification, not an open assumption.**
Eq. 2 defines c^(0) as the leaf and c^(H) as the root. Read eq. 5's h(c) the
same way (0 at the leaf, increasing toward the root) and
lambda(c^(h))=exp(-alpha*h) holds constant at 1 for the leaf-level term
while shrinking for terms closer to the root as alpha grows -- the opposite
of the paper's own surrounding claim that larger alpha weights the top of
the hierarchy more strongly, with alpha>1 said to greatly favor parent
classification. Those two do not describe the same behaviour under the
leaf-indexed reading.

This is resolved, not merely noted, by which of the two could plausibly be
wrong. An equation's index convention can be a typo or a copy-paste slip; a
sentence describing how a hyperparameter's qualitative behaviour changes
across its stated range (alpha=0 treats every level equally; alpha>1
"greatly favors parent classification") cannot be a typo -- that is a
description of what the authors observed their own model do, across a
range of values they say they tried. If the equation and that description
disagree, the equation's indexing is the more likely error. A structural
check agrees independently: the leaf-indexed reading produces a loss that
weights fine-grained (leaf-level) distinctions *most* heavily and coarse
distinctions least -- a hierarchical loss with no purpose over a plain flat
softmax, since the entire reason to build one is to penalize coarse
mistakes (supernova vs artifact) more than fine ones (Ia vs Ib/c), per
CLAUDE.md §4. Nobody designs a hierarchical loss to behave like a worse
flat one.

This module therefore uses depth **from the root** (root=0, increasing
toward leaves): lambda stays near 1 for coarse, near-root terms and shrinks
for fine-grained, leaf-level terms as alpha grows, matching both the paper's
stated intent and CLAUDE.md §4's plain-English restatement ("confusing Ia
with Ib/c is a small error; confusing a supernova with an artifact is a
large one"). At alpha=0.5 (the value used here) this makes an SN-like vs
Fast/Long confusion (depth 3, lambda=0.223) weigh ~1.65x more than an Ia vs
Ib/c confusion (depth 4, lambda=0.135); the leaf-indexed reading would
invert that same ratio the other way. Follow-up, not yet done: the paper
links a reference implementation (github.com/villrv/hxe-for-tda/) that
would settle this by inspection rather than argument -- check it once this
run's result is in hand.

alpha is fixed at the paper's own value, 0.5 ("somewhat favors parent
classification over fine-grained") -- no hyperparameter search (Day-6
instruction: "one run").
"""

import math
from functools import lru_cache
from typing import Sequence

import torch

from src import taxonomy

ALPHA = 0.5

_ORDERED_IDS: tuple[int, ...] = tuple(taxonomy.ordered_class_ids())
_INDEX_OF: dict[int, int] = {cid: i for i, cid in enumerate(_ORDERED_IDS)}
NUM_TAXONOMY_NODES: int = len(_ORDERED_IDS)


def node_index(class_id: int) -> int:
    """Position of `class_id` in the `NUM_TAXONOMY_NODES`-wide logit vector
    (`ordered_class_ids()` order -- stable, per `taxonomy.py`)."""
    return _INDEX_OF[class_id]


@lru_cache(maxsize=None)
def depth_from_root(class_id: int) -> int:
    """Edges from `ROOT_ID` down to `class_id` (0 for the root itself). See
    module docstring for why this, not the paper's literal leaf-indexed `h`,
    is the convention `_lambda` uses."""
    return len(taxonomy.parents(class_id))


def _lambda(class_id: int) -> float:
    return math.exp(-ALPHA * depth_from_root(class_id))


@lru_cache(maxsize=None)
def ancestor_path(leaf_class_id: int) -> tuple[int, ...]:
    """`(leaf_class_id, parent(leaf), ..., topmost non-root ancestor)` --
    `taxonomy.parents()` with `ROOT_ID` excluded (taxonomy.py's own note:
    `ROOT_ID` carries no information and must be excluded from per-node loss
    terms; matches the paper's eq. 2, P(root)=1, eliminated)."""
    return (leaf_class_id,) + tuple(
        a for a in taxonomy.parents(leaf_class_id) if a != taxonomy.ROOT_ID
    )


def compute_node_class_weights(leaf_counts: dict[int, int]) -> dict[int, float]:
    """W(c) = N_all / (N_labels * N_c) for every node on any leaf's ancestor
    path (eq. 4's text). `N_c` for an internal (non-leaf) node is its subtree
    total: the sum of every descendant leaf's count, computed bottom-up from
    `leaf_counts` -- this is why parent nodes end up with a lower weight than
    leaf nodes whenever more than one leaf shares that parent.

    Args:
        leaf_counts: `{leaf_class_id: count}` for every leaf class actually
            present in the training set, e.g. `{2222: 172484, 2223: 240359,
            2224: 430774}` (`N_labels = len(leaf_counts)`, `N_all =
            sum(leaf_counts.values())`).

    Returns:
        `{node_id: W(node_id)}` for every node reachable from some leaf's
        ancestor path (every leaf plus every internal ancestor any leaf
        shares) -- never `ROOT_ID`, never an untouched node.
    """
    n_all = sum(leaf_counts.values())
    n_labels = len(leaf_counts)

    subtree_count: dict[int, int] = dict(leaf_counts)
    for leaf_id, count in leaf_counts.items():
        for ancestor in ancestor_path(leaf_id)[1:]:  # skip the leaf itself
            subtree_count[ancestor] = subtree_count.get(ancestor, 0) + count

    return {node_id: n_all / (n_labels * n_c) for node_id, n_c in subtree_count.items()}


def whxe_loss(
    logits: torch.Tensor,
    class_index: torch.Tensor,
    mask: torch.Tensor,
    leaf_class_ids: Sequence[int],
    node_weights: dict[int, float],
) -> torch.Tensor:
    """Masked WHXE loss (eq. 4) over every real (unpadded) timestep, mean-
    reduced like `src.train.pretrain.masked_cross_entropy` (one term per
    level of the ancestor path shared by every class in `leaf_class_ids`).

    Args:
        logits: `(B, T, NUM_TAXONOMY_NODES)` raw per-timestep logits.
        class_index: `(B,)` integer index into `leaf_class_ids` -- same
            convention as `src.train.dataset.Example.class_index`.
        mask: `(B, T)` bool; `True` = real (unpadded) timestep.
        leaf_class_ids: the target leaf classes, in the order `class_index`
            indexes into (e.g. `taxonomy.TARGET_CLASS_IDS`).
        node_weights: `{node_id: W(node_id)}` from
            `compute_node_class_weights`.

    Raises:
        ValueError: if the leaves in `leaf_class_ids` are not all at the
            same tree depth, or don't all share one sibling group at some
            level -- this implementation assumes a single, shared
            root-to-leaf structure across every target leaf (true for
            CLAUDE.md §4's Ia/Ib-c/II target set, where all three sit under
            2220 SN-like at depth 4; not assumed true for an arbitrary
            future leaf set without re-checking).
    """
    paths = [ancestor_path(c) for c in leaf_class_ids]
    depth = len(paths[0])
    if any(len(p) != depth for p in paths):
        raise ValueError(
            "whxe_loss assumes every target leaf shares one ancestor-path depth; "
            f"got depths {[len(p) for p in paths]} for {list(leaf_class_ids)}"
        )

    device = logits.device
    total = torch.zeros((), device=device)

    for h in range(depth):
        nodes_at_h = [p[h] for p in paths]
        parents_at_h = {taxonomy.parent(n) for n in nodes_at_h}
        if len(parents_at_h) != 1:
            raise ValueError(
                f"whxe_loss assumes every target leaf's level-{h} ancestor shares "
                f"one parent (one shared sibling group); got nodes {nodes_at_h}"
            )
        parent = next(iter(parents_at_h))
        siblings = taxonomy.children(parent) if parent is not None else nodes_at_h
        sibling_indices = torch.tensor(
            [node_index(s) for s in siblings], device=device, dtype=torch.long
        )

        level_log_probs = torch.log_softmax(
            logits.index_select(-1, sibling_indices), dim=-1
        )  # (B, T, |siblings|)

        own_position = torch.tensor(
            [siblings.index(n) for n in nodes_at_h], device=device, dtype=torch.long
        )  # (K,) K = len(leaf_class_ids)
        position_per_example = own_position[class_index]  # (B,)
        index = position_per_example.view(-1, 1, 1).expand(-1, logits.shape[1], 1)
        log_p_mine = level_log_probs.gather(-1, index).squeeze(-1)  # (B, T)

        weight_per_leaf = torch.tensor(
            [node_weights[n] * _lambda(n) for n in nodes_at_h], device=device
        )  # (K,)
        weight_per_example = weight_per_leaf[class_index]  # (B,)

        total = total + (-(weight_per_example.unsqueeze(1) * log_p_mine))[mask].sum()

    n_valid = mask.sum().clamp(min=1)
    return total / n_valid


def leaf_probabilities(logits: torch.Tensor, leaf_class_ids: Sequence[int]) -> torch.Tensor:
    """Renormalized softmax over just `leaf_class_ids`' own logits (ignoring
    every other taxonomy node, including non-target siblings such as 2225
    Iax/2226 91bg for CLAUDE.md §4's 3-class leaf set).

    This is the fair, apples-to-apples probability triple for comparing
    against the flat model's own 3-class softmax through `src.eval.report`:
    the full, un-renormalized hierarchical marginal P(leaf) sums to P(2220)
    (< 1 in general, since some probability mass legitimately belongs to
    sibling leaves this run's data never labels), which is not directly
    comparable to a flat 3-class softmax that always sums to 1.
    """
    indices = torch.tensor(
        [node_index(c) for c in leaf_class_ids], device=logits.device, dtype=torch.long
    )
    return torch.softmax(logits.index_select(-1, indices), dim=-1)


def predicted_node_matches(logits: torch.Tensor, target_node_id: int) -> torch.Tensor:
    """Per-example bool: does the cascading top-down argmax prediction --
    decide among ROOT's children, then among the winner's children, and so
    on -- resolve to exactly `target_node_id`?

    This is "accuracy at the `target_node_id` node" (CLAUDE.md Day-6 step 7:
    report accuracy at 2220 without requiring the leaf-level Ia/Ib-c/II
    decision to also be right -- the graceful-degradation property a
    hierarchical head exists for: an object with a wrong or uncertain leaf
    call can still cascade correctly to "definitely a supernova").

    Args:
        logits: `(..., NUM_TAXONOMY_NODES)`.
        target_node_id: the node to check reachability to, e.g. 2220.

    Returns:
        `(...,)` bool tensor (one fewer dim than `logits`).
    """
    current = taxonomy.ROOT_ID
    reached = torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)
    path_to_target = () if target_node_id == taxonomy.ROOT_ID else ancestor_path(target_node_id)

    for node_id in reversed(path_to_target):
        siblings = taxonomy.children(current)
        sibling_indices = torch.tensor(
            [node_index(s) for s in siblings], device=logits.device, dtype=torch.long
        )
        winner = logits.index_select(-1, sibling_indices).argmax(dim=-1)
        reached = reached & (winner == siblings.index(node_id))
        current = node_id

    return reached
