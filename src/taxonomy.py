"""ELAsTiCC2 taxonomy tree (CLAUDE.md §4). NOT frozen — §2.1 lists only
schema.py and src/eval/ as frozen; this file was never on that list. It
becomes freezable once WHXE loss (§6 step 7) has landed without needing
further changes here. `load_gentype_of_classid`/`load_classid_of_gentype`
were verified against the real CSVs on Day 2 (data/raw/elasticc2/ landed);
see TestLoadGentypeOfClassidRealFile etc. in tests/test_taxonomy.py.

The taxonomy is a 4-digit id where the hierarchy is encoded positionally:
zeroing the rightmost nonzero digit of a node's id gives its parent's id, all
the way up to the implicit root, `ROOT_ID`. E.g. 2222 -> 2220 -> 2200 -> 2000
-> ROOT_ID. This module exploits that structure instead of hardcoding a
parent map, per CLAUDE.md §4.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# The synthetic root of the tree CLAUDE.md §4 defines (a trimmed subset of the
# full ELAsTiCC2 taxonomy: only the Variable/Non-Recurring/SN-like branch and
# siblings we actually use). This must NOT be 0: in the real, untrimmed
# ELAsTiCC2 taxonomy, id 0 is the "Meta" branch (100 Meta/Other, 200 Residual,
# 300 NotClassified), which we deliberately left out of `_NODES` below but
# whose id is still taken. Using 0 here would silently collide with it if
# that branch is ever added back. -1 is guaranteed not to collide with any
# real taxonomy id (all real ids are >= 0).
ROOT_ID = -1


class TargetClass(str, Enum):
    """The three-plus-anomaly target used by the classifier head."""

    IA = "Ia"
    IBC = "Ibc"
    II = "II"
    OTHER = "other"


@dataclass(frozen=True)
class _Node:
    class_id: int
    name: str


# The tree from CLAUDE.md §4, transcribed verbatim (id, display name). This is
# the single source of truth; parents/children/leaves are all derived from it
# below rather than hand-maintained separately.
_NODES: tuple[_Node, ...] = (
    _Node(ROOT_ID, "Alert"),
    _Node(1000, "Static"),
    _Node(2000, "Variable"),
    _Node(2200, "Non-Recurring"),
    _Node(2220, "SN-like"),
    _Node(2222, "Ia"),
    _Node(2223, "Ib/c"),
    _Node(2224, "II"),
    _Node(2225, "Iax"),
    _Node(2226, "91bg"),
    _Node(2230, "Fast"),
    _Node(2232, "KN"),
    _Node(2233, "M-dwarf Flare"),
    _Node(2234, "Dwarf Novae"),
    _Node(2235, "uLens"),
    _Node(2240, "Long"),
    _Node(2242, "SLSN"),
    _Node(2243, "TDE"),
    _Node(2244, "ILOT"),
    _Node(2245, "CART"),
    _Node(2246, "PISN"),
    _Node(2300, "Recurring"),
    _Node(2320, "Periodic"),
    _Node(2322, "Cepheid"),
    _Node(2323, "RR Lyrae"),
    _Node(2324, "Delta Scuti"),
    _Node(2325, "EB"),
    _Node(2326, "LPV/Mira"),
    _Node(2330, "Non-Periodic"),
    _Node(2332, "AGN"),
)

_NAME_OF: dict[int, str] = {n.class_id: n.name for n in _NODES}
_VALID_IDS: frozenset[int] = frozenset(_NAME_OF)

# The three SN subtypes the classifier targets directly (CLAUDE.md §4: "Ib and
# Ic are merged into 2223. Separating them requires spectroscopy.").
_TARGET_LEAF_OF: dict[int, TargetClass] = {
    2222: TargetClass.IA,
    2223: TargetClass.IBC,
    2224: TargetClass.II,
}

# Public, ordered (Ia, Ib/c, II) tuple of the flat 3-class head's target ids
# (CLAUDE.md §6 step 6). Single source of truth for "class index i corresponds
# to which taxonomy id" -- training/eval code should index into this rather
# than re-deriving or hardcoding the mapping locally.
TARGET_CLASS_IDS: tuple[int, int, int] = (2222, 2223, 2224)


def _check_valid(class_id: int) -> None:
    if class_id not in _VALID_IDS:
        raise ValueError(f"{class_id} is not a valid ELAsTiCC2 taxonomy id")


def parent(class_id: int) -> int | None:
    """Return the immediate parent id, or None if `class_id` is the root.

    Implemented by zeroing the rightmost nonzero digit of the (zero-padded,
    4-digit) id — see module docstring. A fully-zeroed result (i.e. the top
    of the digit hierarchy) maps to `ROOT_ID`, not to the literal int 0 —
    see the `ROOT_ID` comment on why 0 itself must never be used here.
    """
    _check_valid(class_id)
    if class_id == ROOT_ID:
        return None
    digits = list(f"{class_id:04d}")
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] != "0":
            digits[i] = "0"
            break
    result = int("".join(digits))
    return ROOT_ID if result == 0 else result


def parents(class_id: int) -> list[int]:
    """Return the chain of ancestor ids from `class_id`'s immediate parent up
    to and including `ROOT_ID`. Empty list if `class_id` is already the root.

    `ROOT_ID` is included so every non-root id has a stable, non-empty
    ancestor chain ending the same way — useful for consistent indexing. It
    carries no discriminating information (every object sits beneath it, so
    it is a constant for every example): WHXE loss (§6 step 7) must exclude
    `ROOT_ID` from its per-node loss terms, or it will contribute a useless
    constant to every example's loss. Documented here so step 7 doesn't have
    to rediscover it.
    """
    _check_valid(class_id)
    chain: list[int] = []
    current = parent(class_id)
    while current is not None:
        chain.append(current)
        current = parent(current)
    return chain


def children(class_id: int) -> list[int]:
    """Return the immediate children of `class_id`, ascending by id."""
    _check_valid(class_id)
    return sorted(cid for cid in _VALID_IDS if parent(cid) == class_id)


def is_leaf(class_id: int) -> bool:
    """True if `class_id` has no children in the tree."""
    _check_valid(class_id)
    return len(children(class_id)) == 0


def name(class_id: int) -> str:
    """Human-readable display name for `class_id`, e.g. 2222 -> 'Ia'."""
    _check_valid(class_id)
    return _NAME_OF[class_id]


def to_target_class(class_id: int) -> TargetClass:
    """Map any taxonomy id to one of {Ia, Ibc, II, other}.

    Walks the ancestor chain (including `class_id` itself): if 2222/2223/2224
    is self or an ancestor, returns the matching target class; otherwise
    OTHER. This means a hypothetical future subtype nested under e.g. 2222
    would still map to Ia without this function changing — only the tree
    above needs to grow.
    """
    _check_valid(class_id)
    for node_id in [class_id, *parents(class_id)]:
        if node_id in _TARGET_LEAF_OF:
            return _TARGET_LEAF_OF[node_id]
    return TargetClass.OTHER


def ordered_class_ids() -> list[int]:
    """Stable pre-order traversal of every node id (including `ROOT_ID`), for
    indexing into model outputs. Deterministic across runs (children are
    visited in ascending id order), so an index computed today stays valid
    tomorrow.

    Includes `ROOT_ID` at index 0 for a complete, stable index space, but see
    the note on `parents()`: `ROOT_ID` carries no information and WHXE loss
    (§6 step 7) must exclude its index from per-node loss terms.
    """

    order: list[int] = []

    def _visit(node_id: int) -> None:
        order.append(node_id)
        for child_id in children(node_id):
            _visit(child_id)

    _visit(ROOT_ID)
    return order


def class_index(class_id: int) -> int:
    """Position of `class_id` in `ordered_class_ids()`."""
    _check_valid(class_id)
    return ordered_class_ids().index(class_id)


# Confirmed by the user directly, not guessed: both CSVs have columns named
# 'gentype' and 'classid'. Read by name — not positionally — and fail loudly
# if either is absent. Verified against the real files on Day 2 (both also
# carry extra columns we ignore: 'description' in elasticc2_gentypeofclassid,
# 'description'/'exactmatch'/'categorymatch'/'broadmatch'/'generalmatch' in
# elasticc2_classidofgentype — harmless, since we read by name).
_GENTYPE_COL = "gentype"
_CLASSID_COL = "classid"


def _require_gentype_classid_columns(df, path: str) -> None:
    missing = [c for c in (_GENTYPE_COL, _CLASSID_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing expected column(s) {missing}; found columns "
            f"{df.columns}. Expected '{_GENTYPE_COL}' and '{_CLASSID_COL}' "
            "(CLAUDE.md §4)."
        )


def load_gentype_of_classid(path: str) -> dict[int, int]:
    """Load `elasticc2_gentypeofclassid.csv`. Where `gentype` is non-blank,
    it is unique across rows. Answers "which specific classid corresponds to
    this gentype?"

    Blank `gentype` is legitimate and expected here, exactly as in
    `load_classid_of_gentype`: every category-level classId (2000, 2200,
    2220, 2230, 2240, 2300, 2320, 2330 — every internal/category node in the
    tree this project uses) has no single corresponding gentype, because
    every SNANA-simulated object has come from one specific model/type — a
    category can't be "the" gentype for anything. Blank rows are excluded
    from the returned mapping (a blank can't be a dict key) and the excluded
    count is logged via `logging` at INFO level, per CLAUDE.md §2.5 — never
    silently dropped, and never raised on, since this is documented/expected
    shape, not an anomaly.

    Args:
        path: path to `elasticc2_gentypeofclassid.csv`.

    Returns:
        Mapping from SNANA gentype (int) to ELAsTiCC2 taxonomy `class_id`
        (int), covering only rows with a non-blank gentype.

    Raises:
        ValueError: if the `gentype`/`classid` columns are missing, or if two
            non-blank `gentype` values collide (a violation of this file's
            documented uniqueness-when-present property).
    """
    import polars as pl

    df = pl.read_csv(path)
    _require_gentype_classid_columns(df, path)

    blank_mask = df[_GENTYPE_COL].is_null()
    n_blank = int(blank_mask.sum())
    if n_blank > 0:
        logger.info(
            "%s: skipped %d of %d row(s) with blank '%s' (category-level "
            "classIds, which have no single corresponding gentype)",
            path,
            n_blank,
            df.height,
            _GENTYPE_COL,
        )

    kept = df.filter(~blank_mask)
    n_duplicates = kept.height - kept[_GENTYPE_COL].n_unique()
    if n_duplicates > 0:
        raise ValueError(
            f"{path}: expected unique non-blank '{_GENTYPE_COL}' values but "
            f"found {n_duplicates} duplicate row(s)."
        )

    return dict(
        zip(
            kept[_GENTYPE_COL].cast(pl.Int64).to_list(),
            kept[_CLASSID_COL].cast(pl.Int64).to_list(),
        )
    )


def load_classid_of_gentype(path: str) -> dict[int, list[int]]:
    """Load `elasticc2_classidofgentype.csv`. Unlike `load_gentype_of_classid`
    (unique when non-blank), `gentype` here repeats across rows because of
    the taxonomy hierarchy — e.g. every SN gentype has a row with
    `classid=2220` (SN-like) in addition to rows for its more specific
    classid(s). That repetition is intentional, not a data error. Answers
    "which classids is this gentype consistent with?"

    Blank `gentype` is legitimate and expected here too, for the same reason
    as `load_gentype_of_classid`: category-level classIds (2000, 2200, 2220,
    2230, 2240, 2300, 2320, 2330 — every internal/category node in the tree
    this project uses) have no single corresponding gentype, since every
    SNANA-simulated object has come from one specific model/type. Blank rows
    are excluded from the returned mapping (a blank cannot be a dict key) and
    the excluded count is logged via the stdlib `logging` module at INFO
    level — not silently dropped, per CLAUDE.md §2.5, and not raised on,
    because this is documented/expected shape, not an anomaly.

    Args:
        path: path to `elasticc2_classidofgentype.csv`.

    Returns:
        Mapping from gentype to the list of classids it is consistent with,
        in file order, duplicates preserved as-is.

    Raises:
        ValueError: if the `gentype`/`classid` columns are missing.
    """
    import polars as pl

    df = pl.read_csv(path)
    _require_gentype_classid_columns(df, path)

    blank_mask = df[_GENTYPE_COL].is_null()
    n_blank = int(blank_mask.sum())
    if n_blank > 0:
        logger.info(
            "%s: skipped %d of %d row(s) with blank '%s' (category-level "
            "classIds, which have no single corresponding gentype)",
            path,
            n_blank,
            df.height,
            _GENTYPE_COL,
        )

    kept = df.filter(~blank_mask)
    mapping: dict[int, list[int]] = {}
    for gentype, classid in zip(
        kept[_GENTYPE_COL].cast(pl.Int64).to_list(),
        kept[_CLASSID_COL].cast(pl.Int64).to_list(),
    ):
        mapping.setdefault(gentype, []).append(classid)
    return mapping
