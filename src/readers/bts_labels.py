"""BTS Sample Explorer spectroscopic-type fetch + taxonomy join (Day 8
session, deliberately separate from `src/readers/btsbot.py`).

`data/raw/btsbot/metadata_v10/` (Zenodo) carries no spectroscopic subtype at
all — only BTSbot's own binary bright-transient/reject triage
(`label`/`is_SN`/`source_set`). The real Ia/Ib/Ic/II spectroscopic type for
each object lives in the external BTS Sample Explorer (Caltech,
https://sites.astro.caltech.edu/ztf/bts/explorer.php), keyed by ZTF object
id ("ZTFID" in its export == `objectId` in metadata_v10 — same string
format, confirmed empirically this session: 5200/5203 (99.94%) of
metadata_v10's `source_set=='trues'` (BTS-confirmed) objects join cleanly;
see chat report for the 3 unmatched and the `rejects`/`dims`/`vars` objects,
which mostly don't match because they were never spectroscopically
classified in the first place — genuine absence, not an id-format bug).

This module is the ONLY place in the reader stack that reaches the network.
`src/readers/btsbot.py` never imports it and always emits `class_id=None` —
the network dependency lives here, isolated and cacheable, so a slow/broken
connection can't break the offline metadata parse. Joining this module's
output onto `btsbot.py`'s `CanonicalAlert.object_id` is a Day 9
fine-tuning-time step, not done here.

-----------------------------------------------------------------------------
Fetch
-----------------------------------------------------------------------------
`fetch_bts_sample()` pulls ALL classified + candidate BTS objects in one CSV
export (`subsample=all&format=csv`, ~20,600 rows as of this session) rather
than querying per-object — verified this single export covers the full
sample rather than one particular subsample/classification slice. Cached to
`data/raw/btsbot/bts_sample_explorer_raw.csv` plus a sidecar
`bts_sample_explorer_raw.meta.json` recording the fetch date/URL/row count,
since BTS is a live, growing sample and this join's reproducibility depends
on knowing when it was pulled — re-running `fetch_bts_sample(force=True)`
will get a larger sample as BTS classifies more objects over time.

-----------------------------------------------------------------------------
Type-string -> taxonomy id mapping — finalized Day 8, with two exceptions
-----------------------------------------------------------------------------
41 distinct raw `type` strings were found across the full BTS sample this
session (see chat report for the complete list with counts). The user
reviewed the proposal against that list and made these calls:

  ACCEPTED (now finalized, no longer "flag for review"):
    "SN Ia-91T"  -> 2222 (Ia). Overluminous, slow-declining, but still a
                    normal thermonuclear explosion; §4 has no dedicated
                    91T node, so it folds into the parent Ia leaf.
    "SN IIb"     -> 2224 (II). §4 explicit: IIb is II, not Ib/c.
    "SLSN-II"    -> 2242 (SLSN). Rare, and §4's single undifferentiated SLSN
                    leaf is understood to model SLSN-I specifically, but
                    there is no better home, so it maps here anyway.

  OVERTURNED:
    "SN Ibn" / "SN Icn" no longer map to 2223 (Ib/c) -- see
    EXCLUDED_CSM_INTERACTING below. Narrow-lined (circumstellar-interacting)
    stripped-envelope SNe are spectroscopically stripped-envelope, but
    photometrically CSM-interaction-powered: brighter, faster-rising, and
    shaped by the same physical mechanism as IIn, not by the stripped
    envelope alone. The "n" means the light curve is shaped by interaction,
    not by the progenitor class the rest of the name implies. Ib/c is the
    class this project is most sensitive on (§11's 0.40-0.60 recall band
    already reflects real scarcity); admitting objects whose photometry
    plausibly belongs nearer the II/CSM family would contaminate exactly the
    class least able to absorb it. Excluded from the 3-class set entirely
    (class_id=None), not reassigned to 2224 -- the point is contamination
    risk, not "II is the truer label," and their photometric behavior
    relative to ordinary IIn was never actually checked this session.

  RESOLVED, from the 549-confirmed-non-SN review:
    "Ca-rich" / "SN Ca-rich-Ca" -> 2245 (CaRT). §4 has a dedicated node;
    these were sitting in NO_TREE_NODE only because no one had checked
    whether CaRT (Cool/Calcium-rich Transient) was the right read on it --
    it is, so they moved out of that bucket into MAPPED.

  STILL PROPOSED, not finalized (do not train on these yet):
    CV, nova, LBV, LRN remain in NO_TREE_NODE, unmapped. Tentative reading
    for a future review, recorded here rather than silently decided:
      - LBV and LRN are both plausibly ILOT-family (2244): giant eruptions
        and stellar-merger transients are commonly grouped under the
        intermediate-luminosity-optical-transient umbrella in the literature
        alongside ILRT (already mapped to 2244 above). Lower confidence than
        the ILRT mapping itself, since LBV/LRN are debated as impostors
        rather than genuine ILOTs by some authors.
      - CV and nova have no defensible §4 home at all, tentative or
        otherwise. §4's only accretion-related leaf is 2234 Dwarf Novae, a
        specific CV subtype (disk instability) -- generic "CV" spans
        several physically distinct subtypes (dwarf novae, novae, polars)
        that don't share one photometric signature, and "nova" (a full
        thermonuclear surface event) is not a Dwarf Novae at all. Forcing
        either into 2234 would misrepresent the distinction §4 itself draws
        by giving Dwarf Novae its own node separate from a generic CV
        catch-all. Leave both unmapped unless/until a broader CV branch is
        added to the tree.

Five buckets, deliberately NOT collapsed into one "else -> None":
  MAPPED                    -- clear taxonomy-id match (may or may not be
                                one of the three §4 SN target leaves
                                2222/2223/2224)
  UNCLASSIFIED               -- '-' / 'Other' / 'other': BTS has not
                                obtained/assigned a spectroscopic class yet.
                                This is MISSING DATA, not a confirmed non-SN
                                negative -- must not be conflated with the
                                categories below when this feeds WHXE (§11)
                                or the anomaly branch (§11) at a later
                                session.
  UNCERTAIN                  -- '?'-suffixed tentative classifications (CV?,
                                AGN?): BTS itself flags these as
                                not-yet-confirmed.
  AMBIGUOUS                  -- "-pec"/CSM/SC-tagged: real spectra were taken
                                and showed something unusual, but the string
                                alone doesn't specify a single family
                                unambiguously enough to bucket automatically.
                                Reported, never silently assigned.
  NO_TREE_NODE                -- a confirmed, unambiguous real class (CV,
                                nova, LBV, LRN) that CLAUDE.md §4's tree
                                simply doesn't have a leaf for. Not "unmapped
                                because ambiguous" -- unmapped because no
                                matching node exists (or, for LBV/LRN, no
                                node the user has signed off on yet), a
                                different failure mode worth distinguishing.
  EXCLUDED_CSM_INTERACTING     -- Ibn/Icn: a confirmed, unambiguous
                                spectroscopic class that DOES have a
                                superficially matching tree node (2223,
                                Ib/c) but is deliberately kept out of it
                                because its photometry is judged CSM-driven
                                rather than representative of ordinary
                                stripped-envelope SNe. Distinct from
                                NO_TREE_NODE (no matching node exists at
                                all) and from AMBIGUOUS (string alone
                                doesn't specify a family) -- here the family
                                is unambiguous, the tree has a node for it,
                                and it is still excluded on purpose.
"""

import dataclasses
import datetime
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import polars as pl

from src.schema import CanonicalAlert

logger = logging.getLogger(__name__)

_EXPLORER_URL = (
    "https://sites.astro.caltech.edu/ztf/bts/explorer.php"
    "?f=s&subsample=all&format=csv&sort=ztfid"
)

_DEFAULT_CACHE_CSV = "data/raw/btsbot/bts_sample_explorer_raw.csv"
_DEFAULT_CACHE_META = "data/raw/btsbot/bts_sample_explorer_raw.meta.json"

# ---------------------------------------------------------------------------
# Taxonomy-id mapping (§4 ids: 2222 Ia, 2223 Ib/c, 2224 II, 2225 Iax,
# 2226 91bg, 2242 SLSN, 2243 TDE, 2244 ILOT, 2332 AGN).
# ---------------------------------------------------------------------------

MAPPED: dict[str, int] = {
    # -- Ia family --
    "SN Ia": 2222,
    # 91T-like: overluminous, slow-declining, but still a normal thermonuclear
    # explosion -- unlike 91bg/Iax, ELAsTiCC2's own taxonomy has no dedicated
    # node for it, so there is nowhere else for it to go without inventing a
    # node §4 doesn't have. Proposed reading: fold into 2222. Flag for review.
    "SN Ia-91T": 2222,
    "SN Ia-91bg": 2226,  # dedicated node -- must NOT fold into 2222
    "SN Iax": 2225,  # dedicated node -- must NOT fold into 2222
    # -- Ib/c family (merged per §4) --
    "SN Ib": 2223,
    "SN Ic": 2223,
    "SN Ic-BL": 2223,  # explicit instruction: Ic-BL merges into 2223
    "SN Ib/c": 2223,
    # -- II family --
    "SN II": 2224,
    "SN IIn": 2224,
    "SN IIP": 2224,
    "SN IIL": 2224,
    "SN IIb": 2224,  # §4 explicit clarification: IIb is II, not Ib/c
    # -- non-3-class, but §4 has a leaf --
    "AGN": 2332,
    "SLSN-I": 2242,
    # SLSN-II (hydrogen-rich) is rare and its relationship to §4's single,
    # undifferentiated "SLSN" leaf is genuinely unclear -- ELAsTiCC2's own
    # SLSN sim class is understood to model SLSN-I specifically. Accepted
    # anyway (single leaf, no better home) -- Day 8 user decision.
    "SLSN-II": 2242,
    "TDE": 2243,
    "TDE-H-He": 2243,
    "TDE-He": 2243,
    "TDE-featureless": 2243,
    "ILRT": 2244,  # ILRT ~= ILOT in the literature
    # Confirmed non-SN with a dedicated §4 node (CaRT) -- moved here from
    # NO_TREE_NODE, Day 8 review of the 549 confirmed-non-SN set.
    "Ca-rich": 2245,
    "SN Ca-rich-Ca": 2245,
}

# Confirmed, unambiguous spectroscopic class that DOES have a superficially
# matching §4 node (2223, Ib/c) but is deliberately excluded from it --
# photometrically CSM/interaction-driven, judged nearer IIn than ordinary
# stripped-envelope SNe. See module docstring "OVERTURNED" note. Excluded
# from the 3-class set outright (class_id=None), not reassigned to 2224.
EXCLUDED_CSM_INTERACTING: frozenset[str] = frozenset({"SN Ibn", "SN Icn"})

# No spectroscopic classification obtained/assigned yet -- MISSING DATA, not
# a confirmed non-SN negative.
UNCLASSIFIED: frozenset[str] = frozenset({"-", "Other", "other"})

# BTS's own '?' suffix: tentative/candidate, not yet confirmed.
UNCERTAIN: frozenset[str] = frozenset({"CV?", "AGN?"})

# Real spectra, ambiguous family membership from the string alone -- report,
# never bucket automatically.
AMBIGUOUS: frozenset[str] = frozenset(
    {
        "SN Ia-pec",
        "SN II-pec",
        "SN Ib-pec",
        "SN Ic-pec",
        "SN IIn-pec",
        "SN Ia-CSM",
        "SN Ia-SC",
    }
)

# Confirmed, unambiguous real class, but CLAUDE.md §4's tree has no leaf for
# it at all (not a merge-candidate, not a naming variant of something that
# does have a leaf). CV/nova/LBV/LRN only -- Ca-rich moved to MAPPED (2245)
# Day 8, see module docstring.
NO_TREE_NODE: frozenset[str] = frozenset({"CV", "nova", "LBV", "LRN"})

_ALL_KNOWN_TYPES: frozenset[str] = (
    frozenset(MAPPED)
    | UNCLASSIFIED
    | UNCERTAIN
    | AMBIGUOUS
    | NO_TREE_NODE
    | EXCLUDED_CSM_INTERACTING
)


@dataclass(frozen=True)
class TypeClassification:
    class_id: Optional[int]
    category: str  # "mapped" | "unclassified" | "uncertain" | "ambiguous" | "no_tree_node" | "excluded_csm_interacting" | "unrecognized"


def classify_bts_type(type_str: str) -> TypeClassification:
    """Classify one raw BTS `type` string. Never silently drops an
    unrecognized string (CLAUDE.md §2.5's spirit applied to labels, not just
    rows): a string not in any of the six known buckets above is logged and
    returned as `category="unrecognized"`, `class_id=None` -- a safety net
    for when BTS's live sample grows a new type string this module hasn't
    seen yet, not an expected path for the 41 strings checked this session.
    """
    if type_str in MAPPED:
        return TypeClassification(MAPPED[type_str], "mapped")
    if type_str in UNCLASSIFIED:
        return TypeClassification(None, "unclassified")
    if type_str in UNCERTAIN:
        return TypeClassification(None, "uncertain")
    if type_str in AMBIGUOUS:
        return TypeClassification(None, "ambiguous")
    if type_str in NO_TREE_NODE:
        return TypeClassification(None, "no_tree_node")
    if type_str in EXCLUDED_CSM_INTERACTING:
        return TypeClassification(None, "excluded_csm_interacting")
    logger.warning(
        "bts_labels: unrecognized BTS type string %r -- not in any known "
        "bucket (mapped/unclassified/uncertain/ambiguous/no_tree_node/"
        "excluded_csm_interacting). Returning class_id=None; this string "
        "was not part of the 41 seen in the Day 8 session and needs review "
        "before being bucketed.",
        type_str,
    )
    return TypeClassification(None, "unrecognized")


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


def fetch_bts_sample(
    cache_csv: str = _DEFAULT_CACHE_CSV,
    cache_meta: str = _DEFAULT_CACHE_META,
    force: bool = False,
) -> pl.DataFrame:
    """Return the full BTS Sample Explorer export as a Polars DataFrame,
    fetching from the network only if `cache_csv` doesn't already exist (or
    `force=True`). Records the fetch date/URL/row-count in `cache_meta` --
    BTS is a live, growing sample, so a future re-fetch will return more
    rows than this session's snapshot.
    """
    if force or not os.path.exists(cache_csv):
        logger.info("bts_labels: fetching BTS Sample Explorer sample from %s", _EXPLORER_URL)
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        request = urllib.request.Request(_EXPLORER_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        with open(cache_csv, "wb") as f:
            f.write(data)

        df = pl.read_csv(cache_csv)
        meta = {
            "source_url": _EXPLORER_URL,
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_rows": df.height,
        }
        with open(cache_meta, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("bts_labels: fetched %d rows, cached to %s", df.height, cache_csv)
        return df

    logger.info("bts_labels: using cached BTS Sample Explorer sample at %s", cache_csv)
    return pl.read_csv(cache_csv)


def build_object_id_to_classification(
    bts_sample: pl.DataFrame,
) -> dict[str, TypeClassification]:
    """One `TypeClassification` per unique `ZTFID` in `bts_sample` (the raw
    export from `fetch_bts_sample`)."""
    result: dict[str, TypeClassification] = {}
    for ztfid, type_str in bts_sample.select(["ZTFID", "type"]).iter_rows():
        result[ztfid] = classify_bts_type(type_str)
    return result


def join_object_ids(
    object_ids: list[str], bts_sample: pl.DataFrame
) -> dict[str, Optional[TypeClassification]]:
    """Join a list of `CanonicalAlert.object_id`s (from
    `btsbot.read_candidates_csv`) onto BTS Sample Explorer types. An
    `object_id` with no entry in `bts_sample` at all (never spectroscopically
    followed up, or not yet in this snapshot) maps to `None` -- distinct from
    `TypeClassification(class_id=None, category="unclassified")`, which
    means BTS DOES have a row for this object but its own `type` is '-' /
    'Other'. Both ultimately mean "no usable class_id", but the distinction
    matters for auditing join coverage (CLAUDE.md §2.5's spirit: know why
    something is missing, don't collapse different reasons together).
    """
    by_id = build_object_id_to_classification(bts_sample)
    return {oid: by_id.get(oid) for oid in object_ids}


def apply_classifications_to_alerts(
    alerts: list[CanonicalAlert], bts_sample: pl.DataFrame
) -> tuple[list[CanonicalAlert], dict[str, int]]:
    """Join BTS Sample Explorer types onto `alerts` (typically
    `src.readers.btsbot.read_candidates_csv` output, which always emits
    `class_id=None` -- see that module's docstring) and return new alerts
    with `class_id` filled in wherever the join produced a usable one, plus
    a per-outcome count (CLAUDE.md §2.5's spirit applied to labels, not just
    rows: every object's fate is accounted for, none silently dropped).

    `class_id` is set only for `category="mapped"` results -- everything
    else (`unclassified`/`uncertain`/`ambiguous`/`no_tree_node`/
    `excluded_csm_interacting`/`unrecognized`, or no BTS row at all) leaves
    `class_id=None`, exactly as `src.readers.btsbot` already does for every
    object before this join runs. Downstream 3-class training code (e.g.
    `src.train.dataset_static.LightCurveStaticDataset`) already filters to
    `class_id in taxonomy.TARGET_CLASS_IDS`, so a non-3-class `mapped`
    result (AGN, SLSN, TDE, 91bg, Iax, CaRT, ...) still gets a real
    `class_id` here -- useful for anomaly/held-out-class work later -- and
    is excluded from 3-class training downstream, not by this function.

    Returns:
        `(new_alerts, stats)`. `new_alerts` is a new list of `CanonicalAlert`
        (via `dataclasses.replace`, since the dataclass is frozen); alert
        order and every other field is unchanged. `stats` counts every
        input alert exactly once, keyed by outcome: `"mapped_ia"` /
        `"mapped_ibc"` / `"mapped_ii"` / `"mapped_other_leaf"` (a real,
        non-3-class taxonomy id), `"unclassified"`, `"uncertain"`,
        `"ambiguous"`, `"no_tree_node"`, `"excluded_csm_interacting"`,
        `"unrecognized"`, `"not_in_bts_sample"` (no BTS row for this
        object_id at all).
    """
    by_id = build_object_id_to_classification(bts_sample)
    stats: dict[str, int] = {}

    def _bump(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    new_alerts: list[CanonicalAlert] = []
    for alert in alerts:
        classification = by_id.get(alert.object_id)
        if classification is None:
            _bump("not_in_bts_sample")
            new_alerts.append(alert)
            continue

        if classification.category != "mapped":
            _bump(classification.category)
            new_alerts.append(alert)
            continue

        class_id = classification.class_id
        if class_id == 2222:
            _bump("mapped_ia")
        elif class_id == 2223:
            _bump("mapped_ibc")
        elif class_id == 2224:
            _bump("mapped_ii")
        else:
            _bump("mapped_other_leaf")
        new_alerts.append(dataclasses.replace(alert, class_id=class_id))

    logger.info("bts_labels: join outcome counts across %d alerts: %s", len(alerts), stats)
    return new_alerts, stats
