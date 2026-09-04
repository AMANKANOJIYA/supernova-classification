"""ELAsTiCC2 taxonomy tree (CLAUDE.md §4). NOT frozen — WHXE loss (§6 step 7)
will extend this module, but the tree structure and mappings below should not
change without a reason as strong as an actual taxonomy revision from the
ELAsTiCC2 project.

The taxonomy is a 4-digit id where the hierarchy is encoded positionally:
zeroing the rightmost nonzero digit of a node's id gives its parent's id, all
the way up to the implicit root (id 0, "Alert"). E.g. 2222 -> 2220 -> 2200 ->
2000 -> 0. This module exploits that structure instead of hardcoding a parent
map, per CLAUDE.md §4.
"""

from dataclasses import dataclass
from enum import Enum

ROOT_ID = 0


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


def _check_valid(class_id: int) -> None:
    if class_id not in _VALID_IDS:
        raise ValueError(f"{class_id} is not a valid ELAsTiCC2 taxonomy id")


def parent(class_id: int) -> int | None:
    """Return the immediate parent id, or None if `class_id` is the root.

    Implemented by zeroing the rightmost nonzero digit of the (zero-padded,
    4-digit) id — see module docstring.
    """
    _check_valid(class_id)
    if class_id == ROOT_ID:
        return None
    digits = list(f"{class_id:04d}")
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] != "0":
            digits[i] = "0"
            break
    return int("".join(digits))


def parents(class_id: int) -> list[int]:
    """Return the chain of ancestor ids from `class_id`'s immediate parent up
    to and including the root (id 0, "Alert"). Empty list if `class_id` is
    already the root.
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
    """Stable pre-order traversal of every node id, for indexing into model
    outputs. Deterministic across runs (children are visited in ascending id
    order), so an index computed today stays valid tomorrow.
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


def load_gentype_mapping(
    gentype_of_classid_path: str,
    classid_of_gentype_path: str,
) -> dict[int, int]:
    """Load the SNANA gentype <-> ELAsTiCC2 taxonomy id mapping.

    CLAUDE.md §4: "Truth labels in the SNANA files are 'gentypes', not
    taxonomy ids. Map them using `elasticc2_gentypeofclassid.csv` and
    `elasticc2_classidofgentype.csv`."

    ASSUMPTION (flagged, not silently picked): CLAUDE.md does not document
    the exact column names in these two CSVs, and they are not present on
    disk as of this session (only `data/raw/elasticc2/` exists, empty). This
    function reads whichever of the two files maps gentype -> classid
    (`classid_of_gentype_path`, by the naming convention `<x>_of_<y>` = a
    table keyed by y giving x) and expects exactly two integer-castable
    columns, taking the first as the key (gentype) and the second as the
    value (classid) — positionally, not by a guessed header name. This should
    be verified against the real file when it lands (see CLAUDE.md §5.1 on
    A_FORMAT.TXT) and this docstring updated with the confirmed column names.
    `gentype_of_classid_path` (the inverse table) is accepted for symmetry
    with the two documented filenames but is currently unused; both tables
    are redundant with each other for this function's purpose.

    Returns:
        Mapping from SNANA gentype (int) to ELAsTiCC2 taxonomy class_id (int).

    Raises:
        FileNotFoundError: if `classid_of_gentype_path` does not exist.
    """
    import polars as pl

    df = pl.read_csv(classid_of_gentype_path)
    if df.width < 2:
        raise ValueError(
            f"{classid_of_gentype_path} has {df.width} column(s); expected at "
            "least 2 (gentype, classid)"
        )
    gentype_col, classid_col = df.columns[0], df.columns[1]
    return dict(
        zip(
            df[gentype_col].cast(pl.Int64).to_list(),
            df[classid_col].cast(pl.Int64).to_list(),
        )
    )
