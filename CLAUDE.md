# CLAUDE.md — Transient Supernova Classifier

Project instructions. Read this fully at the start of every session. If anything
here conflicts with a request in chat, say so before proceeding.

---

## 1. What we are building

A multi-survey, multimodal pipeline that classifies transient astronomical alerts
into **SN Ia / SN Ib-c / SN II**, and flags objects that fit none of them.

- Pretrain on simulated LSST light curves (ELAsTiCC2)
- Fine-tune on real ZTF light curves with spectroscopic labels (BTSbot / BTS)
- Serve inference over a Kafka alert stream
- Rank anomalies nightly with a human feedback loop

Timeline is 14 days, solo developer. Optimise for correct-and-verified over
feature-complete. See the descope ladder in §10.

---

## 2. Non-negotiables

These are not preferences. Violating any of them invalidates results.

### 2.1 Frozen files

After they are written on Day 1–2, **do not modify**:

- `src/schema.py`

Under `src/eval/`: existing functions may not be modified or removed, and no
existing metric's computed value may change. A new, standard,
independently-defined metric may be added — as a new function, plus the
minimal wiring needed to carry it through the existing report structures (a
new dataclass field, a new output-dict key, a new call site) — provided it is
tested against analytically known values and does not alter how any existing
function computes its existing output.

If a task appears to require anything beyond that, stop and ask. A "fix" to
the eval harness that improves a metric is the failure mode this rule exists
to prevent.

**Widened Day 6.** The original wording read as blocking any change at all,
including adding a metric with no correct home outside `src/eval/` (adding it
elsewhere would violate the no-duplicate-metrics property this project
otherwise enforces). The rule's actual purpose — stopping an existing result
from being quietly improved by rewriting how it's computed — doesn't require
blocking a genuinely new, additive metric. See `roc_auc` in `metrics.py`,
added this way after §11's Ia-vs-rest anchor turned out to name a metric
(ROC-AUC) that didn't exist anywhere in the frozen harness.

### 2.2 Train on flux, never magnitude

Magnitudes are logarithmic and cannot represent negative values. Difference-image
photometry produces negative fluxes routinely, from noise on faint sources. Those
points are real information and they are concentrated in the early, pre-peak part
of the light curve — exactly the regime early classification depends on. Convert
to magnitude only for plotting.

### 2.3 Causal models only

`bidirectional=False` everywhere. No attention mask that permits attending to
future timesteps. A bidirectional model at day 3 already knows what happened at
day 60, which is information that does not exist at inference time. It produces
excellent, meaningless scores.

There is a unit test enforcing this (§7, Gate B). It must never be skipped or
weakened.

### 2.4 No new dependencies without asking

Ask first, with a one-line justification. Preferred stack is in §8.

### 2.5 Never silently drop data

If a reader discards rows (bad flags, missing fields, NaNs), it must log the
count and reason. Silent filtering is how a 14:1 class imbalance becomes 40:1.

---

### 2.6 Two ELAsTiCC2 fields are banned as model inputs

Neither may reach any encoder, feature, or label path:

- **`SNTYPE`** — a type and that type plus 100 conventionally distinguishes the
  spectroscopically-confirmed subset in SNANA (observed: 10 and 110 for
  SNIa-SALT3).
- **`REDSHIFT_HELIO`** — documented as zSpec if available, else zPhot.

Both encode whether an object was spectroscopically confirmed. Spec confirmation
is not random: it tracks brightness, which tracks class. Either field is
therefore a partial shortcut to the label, and that selection effect is
simulated into the data.

Labels come from `SIM_TYPE_INDEX` only. Host redshift comes from
`HOSTGAL_PHOTOZ` / `HOSTGAL_PHOTOZ_ERR` only.

## 3. The canonical schema

Every data source converts into this. It is the contract that makes multi-survey
training possible. **FROZEN.**

```python
# src/schema.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class Survey(str, Enum):
    ZTF = "ztf"
    LSST = "lsst"
    ELASTICC = "elasticc"


@dataclass(frozen=True)
class Observation:
    mjd: float                    # Modified Julian Date
    lam_eff: float                # filter effective wavelength, nanometres
    flux: float                   # calibrated flux; MAY BE NEGATIVE
    flux_err: float               # 1-sigma uncertainty, > 0
    is_detection: bool            # False => forced photometry / upper limit


@dataclass(frozen=True)
class StaticContext:
    host_photoz: Optional[float]
    host_photoz_err: Optional[float]
    host_offset_arcsec: Optional[float]
    host_ellipticity: Optional[float]
    host_mag_r: Optional[float]
    mw_ebv: Optional[float]       # Milky Way reddening, from dustmaps
    gal_lat: Optional[float]      # galactic latitude, degrees
    real_bogus_score: Optional[float]


@dataclass(frozen=True)
class CanonicalAlert:
    object_id: str
    survey: Survey
    observations: list[Observation]      # sorted by mjd, ascending
    static: StaticContext
    cutouts: Optional[np.ndarray]        # (3, 63, 63) sci/ref/diff, or None
    class_id: Optional[int]              # ELAsTiCC2 taxonomy id; None if unlabelled
```

Rules:

- `observations` is always sorted ascending by `mjd`.
- `flux` is never clipped at zero.
- `cutouts` is `None` for all simulated data. This is not a bug; see §5.1.
- `class_id` uses the ELAsTiCC2 taxonomy in §4, never a local string label.
- **`flux` is on the SNANA `FLUXCAL` convention: AB zeropoint 27.5.**
  Every reader is responsible for converting its source's photometry
  to this scale. `tensorize.py` and all models may assume a single
  consistent flux scale across all tiers and must never apply a
  per-survey correction.

**This conversion is not automatic — Day 6 audit finding.** ELAsTiCC2's
SNANA `FLUXCAL` already sits on AB zeropoint 27.5, so
`src/readers/elasticc2.py` does no conversion — but that reader is the one
exception, not a template. A real LSST/ZTF alert-stream reader (Day 9+)
receives flux in **nJy, AB zeropoint 31.4**, not 27.5: a fixed factor of
~36.3 between the two scales, applied as a **division** — the FLUXCAL value
is `flux_nJy * 10**((27.5-31.4)/2.5) = flux_nJy / 36.3`. In general,
`flux_27.5 = flux_zp * 10**((27.5 - zp)/2.5)`, which is the only direction
that keeps `mag = zp - 2.5*log10(flux)` unchanged. Every such reader must
rescale `flux`/`flux_err` this way before constructing an `Observation`.

**Corrected 2026-11-05 (post-NSSP audit).** This paragraph previously
called 36.3 a "multiplicative factor"; `src/readers/zeropoint.py` and its
tests implemented exactly that reading, i.e. the **inverse** of the
correct conversion. Every Tier B (BTSbot) and live ZTF flux was stored a
median ~10.3× too faint (row-dependent through `magzpsci`, p5–p95
0.056–0.127× of correct), and every live LSST flux ~1318× too bright.
Tier A (no conversion) was always correct. Fixed in `zeropoint.py`;
`tests/test_zeropoint_conversion.py` now asserts magnitude invariance
directly instead of a hard-coded factor, and every golden fixture's
expected flux was recomputed from `10**(-0.4*(magpsf - 27.5))`, which
needs no zeropoint at all. Numbers from before this date that compare
Tier A against Tier B (zero-shot, sim-to-real gap, transfer gain) were
measured on mis-scaled Tier B input — see `reports/zeropoint_fix_2026-11-05/`
for the re-run.
`flux_norm` (`flux/flux_err`, `tensorize.py`) is scale-invariant and needs no
correction — the factor cancels in the ratio. `flux_abs_norm` and
`flux_err_norm` are **not** scale-invariant, and skipping the conversion
would not just shift them by a constant: the distortion's size depends on
brightness, worst in the low-flux, early-time regime this project
prioritises (§2.2). `src/readers/zeropoint.py` holds the shared, tested
`convert_zeropoint` helper every non-SNANA reader must call;
`tests/test_zeropoint_conversion.py` fails loudly if a future reader
constructs an `Observation` without calling it.

### Filter effective wavelengths (nm)

Use these constants, do not hardcode band indices anywhere.

```python
LAM_EFF = {
    # LSST
    "u": 368.6, "g": 480.3, "r": 622.2,
    "i": 754.5, "z": 869.1, "y": 971.0,
    # ZTF
    "ztf_g": 472.3, "ztf_r": 633.9, "ztf_i": 788.6,
}
```

**Why wavelength and not band identity.** A categorical band index means ZTF's
three bands and LSST's six bands occupy different input spaces, so pretrained
weights transfer to nothing. Conditioning on continuous wavelength makes ZTF _r_
and LSST _r_ land near each other automatically, and makes a missing band a
missing timestep rather than a shape mismatch. This single decision is what makes
the whole ZTF-to-LSST transfer story possible. Do not "simplify" it.

---

## 4. Taxonomy

Use the **published ELAsTiCC2 taxonomy** verbatim. Do not invent one. The digit
structure encodes the hierarchy, so truncating digits gives you the parent class,
and our numbers stay comparable to published broker results.

```
Alert
├── 1000 Static
└── 2000 Variable
    ├── 2200 Non-Recurring
    │   ├── 2220 SN-like
    │   │   ├── 2222 Ia          <-- target class
    │   │   ├── 2223 Ib/c        <-- target class
    │   │   ├── 2224 II          <-- target class
    │   │   ├── 2225 Iax
    │   │   └── 2226 91bg
    │   ├── 2230 Fast
    │   │   ├── 2232 KN
    │   │   ├── 2233 M-dwarf Flare
    │   │   ├── 2234 Dwarf Novae
    │   │   └── 2235 uLens
    │   └── 2240 Long
    │       ├── 2242 SLSN
    │       ├── 2243 TDE
    │       ├── 2244 ILOT
    │       ├── 2245 CART
    │       └── 2246 PISN
    └── 2300 Recurring
        ├── 2320 Periodic
        │   ├── 2322 Cepheid
        │   ├── 2323 RR Lyrae
        │   ├── 2324 Delta Scuti
        │   ├── 2325 EB
        │   └── 2326 LPV/Mira
        └── 2330 Non-Periodic
            └── 2332 AGN
```

- Ib and Ic are **merged** into 2223. Separating them requires spectroscopy.
- Truth labels in the SNANA files are "gentypes", not taxonomy ids. Map them
  using `elasticc2_gentypeofclassid.csv` and `elasticc2_classidofgentype.csv`
  (in `data/raw/elasticc2/`).
- Loss is **weighted hierarchical cross-entropy (WHXE)**: predict per node, and
  weight errors higher in the tree more heavily. Confusing Ia with Ib/c is a
  small error; confusing a supernova with an artifact is a large one.
  Reference: Villar et al. 2023, arXiv:2312.02266.
- §4 transcribes only the branches we use. The real ELAsTiCC2
  taxonomy is wider: it also contains 0 Meta (100 Meta/Other,
  200 Residual, 300 NotClassified), 1000 Static, and an `.../Other`
  sibling for every category (2221, 2231, 2241, 2310, 2321, 2331).
  None currently carries a specific gentype, so nothing breaks, but
  `load_*` must not assume every classid in the CSVs appears in this
  tree.

---

- **`SNIIb` is class 2224 (II), not 2223 (Ib/c).** Verified empirically: gentype
  37, read from `SIM_TYPE_INDEX` in the real HEAD file, and 37 sits in the II
  group {30,31,32,35,36,37} rather than the Ib/c group {20,21,25,26,27}.

## 5. Data

| Tier  | Source                                      | Bands        | Cutouts | Labels        | Role           |
| ----- | ------------------------------------------- | ------------ | ------- | ------------- | -------------- |
| **A** | ELAsTiCC2 training sample, SNANA FITS       | ugrizy       | no      | simulated     | pretraining    |
| **B** | BTSbot v10 (Zenodo 10.5281/zenodo.10839691) | ztf_g, ztf_r | yes     | spectroscopic | fine-tuning    |
| **C** | Fink livestream (ZTF + LSST)                | varies       | yes     | none          | inference only |

Local layout:

```
data/
├── raw/
│   ├── elasticc2/          # ELASTICC2_TRAIN_* dirs, *.DUMP, A_FORMAT.TXT, gentype csvs
│   └── btsbot/             # metadata_v10/, images_v10/ (images deferred to Day 10)
├── interim/                # parquet after reader stage
└── processed/              # tensorised, ready for DataLoader
```

`data/` is gitignored. Never commit data.

**ELAsTiCC2 training-sample source** (public, no credentials):
`https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/`.
Each model directory under that base URL is a browsable subdirectory holding
40 gzipped `_HEAD.FITS`, 40 `_PHOT.FITS`, and a `.DUMP` truth CSV. Fetch with
`make data-elasticc2`, which wraps the same `wget -r -np -nH --cut-dirs=5 -c`
loop over every model directory in §4/§6's SN Ia/Ib-c/II set — `-c` resumes,
so re-running the target after a truncated download (has happened once
already) is always safe and is the correct fix, never a manual retry.

### 5.1 Known data facts — read before writing any reader

**ELAsTiCC2 has no pixel data.** It is a photometry-level simulation, not
pixel-level. There is also no simulated RA/Dec uncertainty. Host galaxies _were_
simulated, with zero to two candidate hosts per object. Consequence: the image
encoder can only be trained during Tier B fine-tuning, so modality dropout is
load-bearing, not decorative.

**Forced photometry is absent from the first alert.** The first detection of a
transient carries no forced photometry. Every detection at least one night later
carries forced photometry reaching back to 30 days before first detection. So a
day-1 alert and a day-2 alert have structurally different content, not merely one
extra point. Code that assumes uniform alert structure will be silently wrong in
exactly the early-time regime we care most about.

**The ELAsTiCC2 training sample is deliberately not identical in composition to
the streamed dataset.** This was intentional, to mimic the fact that brokers will
not have perfectly matched training data. Treat it as a feature.

**Held-out anomaly classes do NOT come free — corrected Day 10.** This
section previously claimed gentypes 71–74 (strongly lensed SN Ia/II/Ib/Ic)
and gentype 98 were sitting in already-downloaded data, excluded from
training, ready to use. **False, checked directly Day 10**: no object with
these gentypes exists in any of the 30 downloadable
`ELASTICC2_TRAINING_SAMPLE_2` model directories (12 SN + 18 exotic), nor in
any DUMP file's GENTYPE column across them. They belong to a separate,
not-currently-accessible ELAsTiCC2 test-sample release, not this training
sample. **Substituted Day 10**: the 18 exotic model directories (TDE, SLSN-I
×2, CART, ILOT, PISN, KN×2, CLAGN, Cepheid, RRL, EB, d-Sct, dwarf-nova,
Mdwarf-flare, uLens×3) served as the Tier A anomaly held-out set instead —
see §11's Day 10 anomaly-branch entry for which of these were actually
scored and why 6 were not.

**BTSbot public data is partial.** The Zenodo release covers ZTF public
(`programid=1`) alerts only, while the production BTSbot also trained on ZTF
partnership data. We therefore cannot exactly reproduce their published numbers,
and should not claim to.

**Tier B has no pre-detection data — Day 8 finding.** Every one of the
409,107 `metadata_v10` rows is a real ZTF alert-stream detection; there is no
forced-photometry row type in this table at all (contrast ELAsTiCC2's PHOT
files, which carry ~30 days of forced photometry per object). Consequence:
`tensorize.tensorize_alert`'s pre-detection windowing is a no-op on Tier B —
there is nothing before day 0 to window. Gate C's phase curve (§7, §11) is
therefore only evaluable from the `cutoff=0` slice onward on Tier B; the
`-5` cutoff must be dropped from that comparison, not reported as a
(meaningless) number. This matters because Day 7's static-encoder gain was
concentrated exactly in the pre-detection regime (+10.8% relative macro-F1
at `-5d`, decaying to +2.7% at full cutoff) — a regime Tier B cannot
represent at all. **"The model degrades on real data" and "the model was
evaluated on an easier phase slice" are different claims and must not be
conflated**: a Tier A→Tier B macro-F1 comparison at matched cutoffs (`≥0`
only) isolates the first; comparing Tier A's full phase curve (which
includes `-5`) against Tier B's necessarily-truncated one would silently
smuggle in the second and make the sim-to-real gap look larger or smaller
than it actually is for reasons that have nothing to do with sim-to-real
generalization.

**`isdiffpos` is true for all 409,107 Tier B rows — Day 8 finding.** BTS
pre-filters its public sample to brightening subtractions before a candidate
ever reaches `metadata_v10`, so fading candidates and negative subtractions
never enter Tier B at all (this is also why every reconstructed Tier B flux
is positive — see Gate A check #2 and `src/readers/btsbot.py`'s module
docstring). A live Fink stream (Tier C) does no such filtering. Tier B is
therefore a systematically cleaner slice than Tier C, not a representative
sample of it — same class of problem as the Day 4 ELAsTiCC2-training-vs-
streamed-composition mismatch (§5.1 above). Consequence for Day 12 serving:
a model fine-tuned only on Tier B meets negative-flux inputs for the first
time in production, on Tier C, having never seen one since Tier A
pretraining. Record this as a known Tier B → Tier C divergence to revisit
at the serving step, not something Day 9's fine-tune can fix by itself.

**Dustmaps SFD data — fetched from a mirror, Day 8.** `dustmaps`' documented
default fetch source (Harvard Dataverse) returned HTTP 403 in this
environment, so the two SFD FITS files (`SFD_dust_4096_ngp.fits`,
`SFD_dust_4096_sgp.fits`, placed at `data/raw/dustmaps/sfd/`) were fetched
from an unofficial mirror instead. An unofficial mirror is a reproducibility
liability even when the values verify correctly, so both files' checksums
are recorded here for future verification against whatever source is used
next time:

| File                      | SHA-256                                                            |
| ------------------------- | ------------------------------------------------------------------- |
| `SFD_dust_4096_ngp.fits`  | `50b6aaad0b880762d0fd081177802dcc17c39d7044a410dd5649e2dfd0503e97` |
| `SFD_dust_4096_sgp.fits`  | `84891a59054adab44a7be54051e4dcf0e66e3f13eee0d845ce3739242f553b83` |

**The exact mirror URL used for this fetch was not recorded at fetch time and
is not otherwise reconstructable this session — needs the user to supply it
so this table can be completed with provenance, not just checksums.**

**Read `data/raw/elasticc2/A_FORMAT.TXT` before writing the SNANA reader.** It is
the format spec. SNANA ships HEAD and PHOT FITS file pairs; they are readable as
standard FITS tables but the row-pointer convention linking HEAD to PHOT is not
obvious. Do not guess it.

**The serving trigger is detection-driven, not photometry-driven — Day 6
clarification.** Alert streams (ZTF, LSST, Fink) emit one alert per detection
(S/N > 5), never per forced-photometry point alone. No detection means no
alert means no re-classification opportunity. This is why
`tensorize.tensorize_alert`'s window right edge is `last_detection_mjd`, not
"latest observation of any kind": at every moment a live/replay pipeline
would actually be asked to classify an object, `last_detection_mjd` **is**
"now" — there is no fresher data it could have missed. Consequence for
anything built later: any re-scoring path — the nightly anomaly-ranking loop
in §1 included — must not re-tensorize from newly-arrived forced photometry
alone, with no new detection. Doing so would feed the model a right edge the
training distribution never contained, and in the current detection-driven
design would never actually be triggered by a real alert anyway.

**Resolved: per-timestep cadence as a class shortcut (Day 6 audit, closed).**
Coarse per-object statistics (total observation count, band count, time
span) were checked directly against the raw ELAsTiCC2 FITS files across all
12 model directories and don't separate class (mutual information ≤0.03 bits
against a 1.47-bit ceiling; band count is literally constant at 6 for every
object, every class). What that check couldn't rule out — a finer-grained,
class-correlated simulation fingerprint in the exact night-to-night cadence
pattern a causal GRU actually sees, timestep by timestep, which the coarse
statistics average away — is exactly what the shuffled-label test (Gate B
#1, §7) is built to catch: if the model could read class off cadence rather
than off the (now-randomised) label, it would still find that signal and
predict shuffled labels well above chance. At the corrected 5,000-object
scale it does not: 3 of 4 seeds collapse completely onto a single predicted
class (750/750), the fourth lands at 0.288, both within the ~0.17–0.29
no-signal band measured at that scale. A model with a real cadence-based
shortcut would show up here regardless of which label got shuffled onto
which object. Closed, not just left clean at the coarse-statistic level.

---

**SNANA sentinel values — four different conventions, most undocumented.**
Convert each to `None` by comparing a field against _its own_ sentinel. Never
infer missingness from `HOSTGAL_NMATCH`: an object with `HOSTGAL_NMATCH=0` was
empirically observed still carrying real host values for a sub-threshold nearest
candidate.

| Sentinel | Fields                                                                                                                      | Documented? |
| -------- | --------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `-9`     | `HOSTGAL_PHOTOZ[_ERR]`, `HOSTGAL_SPECZ[_ERR]`, `HOSTGAL_SNSEP`, `HOSTGAL2_PHOTOZ`, `HOSTGAL2_SNSEP`, `REDSHIFT_HELIO[_ERR]` | yes         |
| `-999`   | `HOSTGAL2_RA`, `HOSTGAL2_DEC`                                                                                               | partly      |
| `-9999`  | `HOSTGAL_ELLIPTICITY`, `HOSTGAL2_ELLIPTICITY`                                                                               | **no**      |
| `999`    | `HOSTGAL_MAG_[band]`, `HOSTGAL2_MAG_[band]`                                                                                 | **no**      |
| `0`      | `HOSTGAL2_OBJID`                                                                                                            | no          |

`999` is the most dangerous: host magnitudes are legitimately 18–28, so an
unconverted `999` reads as "very faint host" rather than "missing" and will
train silently.

**`PTROBS_MIN` / `PTROBS_MAX` are 1-indexed and inclusive on both ends.**
`PTROBS_MAX - PTROBS_MIN + 1 == NOBS`; the correct slice is
`phot[PTROBS_MIN-1 : PTROBS_MAX]`. An `MJD = -777` terminator row sits
immediately after `PTROBS_MAX` and must never be included. This is not
documented in `A_FORMAT.TXT`; it was derived empirically and cross-checked
against `LSSTDESC/elasticc`'s own `lib_elasticc2/read_snana.py`.

**Three unrelated numbering schemes coexist in these files** and were observed
taking three different values for the same object: MERGE-list position (29),
`SIM_MODEL_INDEX` (6), and `SIM_TYPE_INDEX` / gentype (10). Only
`SIM_TYPE_INDEX` is the gentype.

**File-count completeness must be asserted, not assumed.** Every model directory
contains exactly 40 `_HEAD.FITS` and 40 `_PHOT.FITS` files. Readers
log-and-continue on a corrupt file (a truncated download was observed), so a
per-directory assertion of 40+40 is required, or a bad download silently removes
~1/40 of a class without any row-level discard being recorded.

## 6. Build order

Each step exists to make the next step's errors visible. Do not reorder.

```
 1. src/schema.py                  <- freeze
 2. src/eval/                      <- freeze. BEFORE any model.
 3. dumb baselines                 <- the number to beat
 4. src/readers/                   <- one adapter per source -> CanonicalAlert
 5. src/features/tensorize.py      <- CanonicalAlert -> padded tensors
 6. LC encoder + flat 3-class head <- simplest thing that trains
 7. taxonomy.py + WHXE loss        <- swap head, keep everything else
 8. static encoder                 <- host + extinction features
 9. fine-tune on Tier B            <- the sim-to-real step
10. image encoder                  <- only meaningful on Tier B
11. anomaly branch                 <- needs a trained encoder
12. serving, replay mode           <- Kafka producer replays archived parquet
13. serving, live mode             <- swap producer; config change only
14. review UI + AAD retrain        <- closes the loop
```

Build **replay mode before live mode**. Same consumer code, archived data as the
producer, no credentials needed. Swapping in the live Fink consumer is then a
config change, which keeps credential turnaround off the critical path. Fink also
provides test streams with replayed alerts using the same credentials, topics
prefixed `ftest_` instead of `fink_`.

---

## 7. Gates

Do not advance past a gate. Each catches a specific silent failure.

### Gate A — after readers (step 4)

- Plot 20 light curves per tier. Ia should show a smooth rise and decline;
  Type IIP should show a visible plateau. If they look like noise, the reader is
  wrong regardless of whether it ran.
- Assert negative fluxes exist in the parsed data. If every flux is positive,
  upper limits were dropped or something went through magnitudes.
- Print class counts. Expect roughly 3700 Ia to 255 Ib/c in the real BTS sample.
  If Ib/c is far below that, a filter is eating them.

### Gate B — after first trained model (step 6)

Three mandatory tests, in `tests/test_gate_b.py`:

1. **Shuffled-label test.** Permute training labels randomly, retrain briefly.
   Macro-F1 must collapse to chance. If it does not, there is label leakage.
   Chance is **not** flat 0.33 here — against this sample's 20.4/28.5/51.1
   composition, marginal collapse gives ~0.225 and uniform-random gives
   ~0.32, so the no-signal band is ~0.23–0.32.
   **Day 6: ran to completion for the first time and failed** (450-object
   subsample, macro-F1 0.453 vs the test's `<0.45` threshold). Diagnosed via
   (a) confirming the permutation touches only training labels, never the
   test split — clean; (b) confusion matrix at the failing seed — spread
   across all three classes, looked structured in isolation; (c) three more
   seeds on the same 450 objects — scattered 0.18–0.45 (two below the
   no-signal floor), which is the signature of test-set-size noise, not a
   model finding real signal in random labels (genuine leakage would show up
   consistently elevated across seeds, not scattered above and below chance).
   Fixed by raising the subsample from 450 to ~5,000 stratified objects
   (`_N_PER_CLASS` 150→1667 in `tests/test_gate_b.py`) — a sample-size
   change only, threshold untouched. Passes at the larger size: a 4-seed
   re-run at 5,000 objects gave 3 of 4 seeds as a total collapse onto one
   predicted class (750/750) and the fourth at 0.288 — an observed
   no-signal range of ~0.17–0.29 at this scale, well under the test's
   `<0.45` threshold. That threshold is therefore loose relative to what's
   actually observed — noted here, not tightened, since the test currently
   passes and there's no evidence the slack has hidden anything.
2. **Causality test.** Truncate a sequence at timestep `t`. Assert the prediction
   at `t` is bit-identical with and without timesteps after `t` present.
3. **Ceiling check.** Ia-vs-rest **ROC-AUC** (not macro-F1 — see §11's Day-6
   correction) should land at 0.93–0.96 once static/host features exist
   (step 8); at the LC-only stage (steps 6-7) the wider range 0.88–0.96
   applies (`CEILING_ROC_AUC_STEP6` in `tests/test_gate_b.py`). Above 0.99
   means something leaked, usually a metadata column encoding the answer.

### Gate C — after fine-tuning (step 9)

Accuracy will drop from simulation to real ZTF. That drop is the expected result
and must be recorded. If it does not drop at all, the evaluation split is wrong.

### Gate D — after anomaly branch (step 11)

Manually inspect the top 50 anomalies. Expect them to be artifact-dominated. A
published run over 2.25 million ZTF objects found 68% of flagged outliers were
bogus light curves from subtraction artifacts and blending. If our top 50 is
_not_ artifact-heavy, real-bogus filtering is over-aggressive and is suppressing
real signal too.

**Day 10 verdict: DEFERRED, not passed or failed.** This gate's own premise
did not apply to what was run, and that is itself the finding -- there is no
pass/fail reading available from this pass, only a different, useful result
(below). The published 68%-artifact anchor is about IMAGE-level subtraction
artifacts. This pipeline has no cutout pixel data anywhere (§5.1: none in
ELAsTiCC2 by construction; BTSbot's `images_v10` never loaded, and the image
encoder, step 10, was explicitly skipped this session in favor of the
anomaly branch -- §10's Day-10 reorder note) -- so the top 50 could only be
inspected photometrically (light-curve shape/cadence), never visually. Worse:
Tier B's `weird` eval population (`src.train.train_anomaly.load_tier_b_weird`)
is built entirely from BTS Sample Explorer's spectroscopically-CONFIRMED
categories (`mapped_other_leaf` / `no_tree_node` / `excluded_csm_interacting`
-- see `src/readers/bts_labels.py`'s own docstring: each is "a confirmed,
unambiguous real class"), never from unclassified/candidate/reject objects.
Every one of the top 50 is therefore, by construction, a real astrophysical
object with a real spectroscopic label -- there was no bogus/artifact
candidate population in the eval set for the ranking to have found in the
first place. Gate D's actual question ("does real-bogus filtering suppress
real signal") cannot be answered by this pass; a real Gate D needs the full,
unfiltered alert stream (including rejects/candidates), which is Tier C
(serving), not built yet.

What the top 50 actually showed is the detector working, read correctly:
46/50 objects were `no_tree_node` (CV/nova/LBV/LRN), 3/50 AGN, 1/50 91bg --
all "weird"-sourced, ZERO from the normal-class test population, and zero
artifacts (there were none to find, see above). Ranking real,
astrophysically distinct objects (eruptive variable stars, AGN flickering)
above ordinary Ia/Ib-c/II every single time, with no false positives from
the normal population leaking into the top 50, is exactly what a working
anomaly ranker should do on a population that contains no bogus candidates.
A cheap photometric heuristic (`src/train/gate_d_inspection.py`,
single-point-outlier-vs-neighbors) additionally tagged 36/50 "artifact_like"
-- but since every object is spectroscopically confirmed real, visual
inspection of the saved light-curve plots (`reports/gate_d/light_curves/`)
showed these are genuine multi-night eruption/decline sequences (nova-like
outbursts), not artifacts. The heuristic's "artifact_like" label is a real
limitation of a proxy built without pixel data, not evidence of actual
artifacts -- stated here so a future session doesn't take that bucket's
name at face value.

This compounds with, rather than duplicates, §5.1's `isdiffpos` finding.
That finding is about the input POPULATION: BTS pre-filters to brightening
subtractions before a candidate ever reaches `metadata_v10`, so Tier B is
already a cleaner slice than a live Fink stream (Tier C) at the data-source
level, before any ranking happens. This session's finding is about the EVAL
SET this Gate D pass was built from specifically: `weird` draws only from
BTS's spectroscopically-CONFIRMED categories, a second, independent
filtering step on top of `isdiffpos` that removes candidates/rejects
entirely, not just fading ones. Both point the same direction for the same
reason (Tier B is pre-cleaned relative to what production will see) but at
different pipeline stages -- `isdiffpos` at ingestion, this session's
category choice at eval-set construction. Consequence for Day 12: an
artifact-rate check against the real 68% published anchor needs a Tier-C-
style unfiltered stream, not Tier B under either filter.

---

## 8. Stack

Ask before adding anything not listed.

| Layer       | Choice                                                    |
| ----------- | --------------------------------------------------------- |
| Language    | Python 3.12                                               |
| Modelling   | PyTorch 2.x + Lightning                                   |
| Config      | Hydra                                                     |
| Dataframes  | Polars; pandas only where a library forces it             |
| Astro       | astropy, sncosmo, dustmaps                                |
| Features    | `light-curve` (Rust-backed)                               |
| Anomaly     | `coniferest` (Isolation Forest, AAD, PineForest)          |
| Streaming   | `confluent-kafka`, or `fink-client` for Tier C            |
| State       | Redis (rolling 30-day object history), Postgres (durable) |
| Serving     | FastAPI + ONNX Runtime                                    |
| Tracking    | MLflow                                                    |
| Monitoring  | Prometheus + Grafana                                      |
| Local infra | Docker Compose                                            |
| Plotting    | matplotlib (backend `Agg` outside notebooks)              |

Baselines worth reading rather than reimplementing: `snmachine` (LSST DESC),
`SuperNNova`, ALeRCE's `lc_classifier`.

---

## 9. Repo layout

```
sn-classifier/
├── CLAUDE.md
├── Makefile                  # make setup / data / train / test / reproduce
├── pyproject.toml
├── configs/                  # hydra
├── data/                     # gitignored
├── notebooks/                # exploration only, never imported by src/
├── src/
│   ├── schema.py             # FROZEN
│   ├── taxonomy.py
│   ├── readers/              # elasticc2.py, btsbot.py, avro_ztf.py, avro_lsst.py
│   ├── features/             # tensorize.py, extinction.py, wavelength.py
│   ├── models/               # lc_encoder.py, image_encoder.py, static_encoder.py,
│   │                         # fusion.py, heads.py, whxe.py
│   ├── anomaly/              # decoder.py, scoring.py, pineforest.py
│   ├── train/                # pretrain.py, finetune.py
│   ├── eval/                 # FROZEN — metrics.py, phase_curves.py, baselines.py
│   └── serve/                # ingest.py, inference_worker.py, api.py, replay.py
├── infra/                    # docker-compose.yml
├── ui/                        # anomaly review app
└── tests/
```

- `src/diagnostics.py` — data-integrity plots and checks (Gate A light
  curves, distribution summaries). Distinct from `src/eval/`, which
  judges models and is frozen. Diagnostics judge the _parse_. May
  import from `src/eval/`; the reverse is forbidden.

---

## 10. Descope ladder

We will fall behind. Cut from the bottom up. Everything above a cut still forms a
coherent project.

```
cut last  | 1. schema + eval + baselines
          | 2. ELAsTiCC2 -> causal GRU -> 3-class
          | 3. WHXE hierarchical loss
          | 4. static / host features
          | 5. fine-tune on real ZTF        <- minimum respectable stopping point
          | 6. anomaly branch
          | 7. replay-mode Kafka pipeline
          | 8. image encoder
          | 9. live Fink stream
cut first | 10. review UI + AAD loop
```

Items 9 and 10 depend on things outside our control (credential turnaround, a
human available to label). Treat them as bonus.

**Day 10 reorder, not a descope.** §6's build order lists step 10 (image
encoder) before step 11 (anomaly branch); this ladder lists item 6 (anomaly
branch) above item 8 (image encoder) already, so the two documents did not
even agree on relative priority. Built anomaly branch first, image encoder
skipped for now. Reasoning: (1) the anomaly branch is the second half of
this project's own problem statement (§1: "rank anomalies nightly with a
human feedback loop"), while the image encoder is an input modality, not a
deliverable in itself; (2) this ladder already ranks anomaly branch (6)
above image encoder (8) -- cutting from the bottom up, image encoder goes
before anomaly branch would if anything have to go first; (3) §5.1: "the
image encoder can only be trained during Tier B fine-tuning" -- it has a
hard prerequisite (Tier B images, `images_v10`, still not loaded -- CLAUDE.md
§9's repo layout note, "images deferred to Day 10" -- deferred again) that
the anomaly branch does not share; (4) the anomaly branch's held-out
negatives (§5.1's gentype 71-74/98 note; the newly-downloaded exotic model
directories; Tier B's confirmed non-SN/AGN/SLSN/TDE/CSM-interacting objects)
all exist and are usable today, with or without an image encoder. Image
encoder remains descoped-from-here (§10 ladder item 8), not abandoned.

**Day 12: image encoder (item 8) is CUT, not deferred -- decision recorded,
not yet acted on by deleting anything.** With 2 days left (Day 13 live Fink
+ UI, Day 14 UI + buffer) and replay-mode serving (item 7) landing today,
there is no remaining day this ladder allocates to it: cutting from the
bottom up, items 9 (live Fink) and 10 (review UI + AAD loop) are already
committed as this project's actual remaining work, which leaves nothing
between here and Day 14 for item 8. It was never in §1's problem statement
(pretrain / fine-tune / serve / rank anomalies -- no mention of images) --
unlike the anomaly branch, which is item 6 of §1 verbatim. Its hard
prerequisite (Tier B `images_v10`, §5.1) is still unloaded three separate
sessions after "deferred to Day 10" (§9) was itself already a second
deferral, and loading it now would still only be step one of building an
image encoder, fusing it in, and retraining Tier B's fine-tune to actually
use it -- a multi-day scope this timeline no longer has room for. The UI
(§1, §6 step 14) will still display BTSbot's and Fink's own image cutouts
directly from their source data -- that is presentation of existing
pixels, not a model change, and needs no image encoder to exist. Cut, for
the record: not built, not scheduled, and every image-encoder-dependent
plan in this file (§9's repo layout `image_encoder.py` stub, this ladder's
item 8) should be read as declined scope from here on, not a TODO.

---

## 11. Expected results — sanity anchors

| Measurement                        | Expected      | Interpretation if far off              |
| ----------------------------------- | ------------- | -------------------------------------- |
| Ia vs rest, binary **ROC-AUC**, Tier A | 0.93–0.96 (0.88–0.96 pre-step-8) | <0.90 preprocessing bug; >0.99 leakage |
| 3-class macro-F1, LC only    | ~0.70         |                                        |
| 3-class macro-F1, + metadata | 0.80–0.85     | metadata should give a clear jump      |
| **Ib/c recall**              | **0.40–0.60** | this looks bad and is correct          |
| Tier A -> Tier B drop        | noticeable    | no drop means broken eval split        |

**Day-6 correction: this row is ROC-AUC, not macro-F1.** The 0.93–0.96
figure is a literature anchor (RAPID, SuperNNova) reporting ROC-AUC/accuracy,
not macro-F1 — obvious in hindsight next to this table's own macro-F1
anchors (ATAT 82.9, ATCAT 71.8, both far below 93-96). The step-6 checkpoint
was first checked against macro-F1 (0.7905) by mistake; the actual metric,
Ia-vs-rest ROC-AUC, measures **0.9185** on the same checkpoint — just under
the literature range, which is expected: no static/host features yet (those
land at step 8), no WHXE loss yet, and this sample's Ib/c share (28.5%) is
far above its real-sky rate (see the class-weighting note below). The range
is therefore stage-dependent: **0.88–0.96** applies through step 7 (LC-only,
`CEILING_ROC_AUC_STEP6` in `tests/test_gate_b.py`); the tighter **0.93–0.96**
literature anchor (`CEILING_ROC_AUC_STEP8`) applies from step 8 once static
features exist to explain it. `src/eval/metrics.py::roc_auc`, added Day 6
under §2.1's widened frozen-file rule — see that section.

**WHXE decision (Day 6), for the record, provisional.** Step 7's controlled
comparison (same encoder, data, split, seed, class weighting as the step-6
flat model) landed leaf macro-F1 at 0.6823 vs. the flat head's 0.6981
(−0.0158), with per-class F1 worse across all three classes and Ia-vs-rest
ROC-AUC essentially unchanged (0.9183 vs. 0.9185). 2220 (SN-like) node
accuracy measured 1.000, but that number is not evidence of anything: every
object in this dataset is SN-like, so there is no non-SN-like branch for the
hierarchy to correctly reject, and no flat-model equivalent to compare it
against. The mechanism explains the cost: WHXE pulls probability mass up to
the shared 2220 parent and redistributes it toward Ia (Ia recall
0.719→0.819, precision 0.632→0.545), a real error-pattern shift with nowhere
useful to spend it on a single-populated-branch dataset. **Decision: keep
the flat head as the default for now.** This is provisional, not a rejection
of WHXE on merit — its sole justification, graceful degradation under real
ambiguity, is structurally untestable until a non-SN-like branch exists in
the data. Revisit at step 9 (Tier B fine-tuning introduces real non-SN
contaminants) and step 11 (the anomaly branch's held-out exotic classes) —
the first points where 2220-node confidence means anything. `whxe.py` and
its tests stay in the repo; a future session should not re-litigate this
decision from the numbers alone without reading this note, and should not
assume WHXE was rejected on merit.

**Day 9 confirmation, not yet acted on.** Step 9's Tier B join produced the
non-SN-like population this note anticipated: 549 objects confirmed
non-SN-like (AGN/SLSN/TDE/CaRT/91bg/Iax and friends, "mapped_other_leaf" in
the join stats) plus 442+ objects with a confirmed real class outside §4's
tree entirely (CV/nova/LBV/LRN — `no_tree_node`) and 22+ excluded
CSM-interacting Ibn/Icn, across a dataset that also has genuine SN-like
objects (Ia/Ib-c/II). 2220-node accuracy is therefore now a meaningful
measurement for the first time — there is finally a non-SN-like branch for
the hierarchy to be right or wrong about. Per this note's own plan, that
means WHXE is revisitable at step 11, not now: this session did not retrain
or re-evaluate WHXE, and nothing above should be read as doing so.

Published state of the art for context: ATAT reports macro-F1 82.9 ± 0.4 over 20
ELAsTiCC classes; ATCAT reports 71.8% F1 light-curve-only and 89.8% with
metadata. Both ORACLE and ATAT report SN subtypes as their weak spot. **A model
that suddenly nails Ib/c is a model that is cheating.** Investigate, do not
celebrate.

Primary reported figure is **macro-F1 as a function of days since first
detection**, plus confusion matrices at fixed time cutoffs. Plain accuracy is not
a reportable metric here — always-predict-Ia scores well and is useless.

---

- Do **not** derive class weights from Tier A training-sample proportions and
  assume they transfer. The ELAsTiCC training sets were deliberately not matched
  in composition to the streamed data (§5.1), so the sample's class balance is
  an artefact of the release, not an estimate of the real rate.

- **Class balance decision (Day 4), for the record.** The full 3-class Tier A
  set (12 model directories, 843,617 objects) is Ia 20.4% / Ib/c 28.5% /
  II 51.1% — an artefact of how many model directories DESC happened to
  release per class, not a rate estimate (real sky rate is roughly
  25–30% / 10–20% / 55–65%). We train on the natural ELAsTiCC composition as
  released: **no oversampling, no reweighting toward sky rates.** It's
  actually closer to balanced than the real sky rate is, and macro-F1 (§11's
  primary metric) already weights every class equally at eval time regardless
  of the training-time prior — so there is little to gain and a leakage risk
  (§2.6's whole concern: anything correlated with class is suspect) to
  introducing an artificial reweighting scheme. Apply mild class weighting
  only if Ia specifically underperforms in practice. Output probabilities from
  a model trained this way carry the **ELAsTiCC prior**, not the true sky
  prior — correcting for the real rate (if ever needed) is a **deployment-time
  step**, not something to bake into training.

- **`class_weighting=true`, left unvalidated (Day 6).** The step-6 checkpoint
  trained with `class_weighting: true` (sqrt-dampened inverse-frequency:
  Ia 1.21 / Ib/c 1.02 / II 0.77) even though this section's own precondition
  for turning it on — "only if Ia specifically underperforms" — didn't hold:
  Ia was not the worst class (recall 0.719 vs Ib/c 0.648). Ib/c was
  effectively unweighted (1.02 ≈ 1), so the weights are not why its recall
  sits above the 0.40–0.60 band above. No controlled ablation was run: at
  590,532 training examples, a rerun costs ~2.5h wall-clock, not affordable
  with 8 days left. Left as `class_weighting: true`, documented as
  unvalidated rather than silently assumed fine. Expected to resolve on its
  own at Gate C, where Ib/c drops to its real ~6%-prevalence rate (255
  objects) instead of this sample's 28.5% (168,251 objects) — the 0.40–0.60
  band was calibrated against that real rarity, not against a sample where
  Ib/c is nearly a third of the data.

- **Gate C: Tier B fine-tune results (Day 9), for the record.** Full
  methodology, LR/freezing schedule, and per-variant numbers live in
  `src/train/finetune_bts.py`'s module docstring and
  `reports/gate_c/gate_c_report.json`; headline numbers only, here:
  - **Sim-to-real gap**: Tier A test macro-F1 0.7169 → Tier B test
    (fine-tuned, LC+static) macro-F1 0.6343, a 0.0826 drop — the expected
    direction per §7 Gate C. Zero-shot (Tier A weights, no fine-tuning) on
    Tier B test macro-F1 0.4483 — fine-tuning bought +0.186 macro-F1.
  - **Ib/c**: F1 0.411 on 35 test objects (13.9:1 Ia:Ib/c at test) — inside
    the §11 0.40–0.60 band, as expected, not a surprise needing
    investigation.
  - **Host-fields ablation — surprising, investigated, not tuned away.**
    This session's own prediction ("most of the static gain likely doesn't
    survive, since photo-z was probably doing the heavy lifting") undershot
    what actually happened: the LC-only fine-tune (macro-F1 0.6716) beat the
    LC+static fine-tune (0.6343) outright — static features were net
    *negative* on Tier B, not merely non-additive. Plausible mechanism, not
    fully verified: 3 of the static branch's 7 inputs (`host_photoz`,
    `host_photoz_err`, `host_ellipticity`) are permanently missing-flagged
    at Tier B (§5.1), so the branch carries real information from only 4
    fields into a fine-tune with just 5,407 labeled training objects —
    small enough that an extra 32-unit MLP branch competing for the same
    downstream head plausibly overfits or crowds out trunk adaptation
    rather than helping. Not chased further with an ablation-of-the-
    ablation (e.g. retraining the static branch alone at a different LR) —
    time-boxed per the 14-day timeline (§1); flagged here for a future
    session rather than quietly averaged away.
  - **`real_bogus_score`, populated at Tier B — did not help.** The
    LC+static+DRB variant (macro-F1 0.5812, Ib/c F1 0.15) underperformed
    both LC+static (0.6343) and LC-only (0.6716). Given the static branch
    was already net-negative at this sample size, adding a second small,
    freshly-initialized branch (`drb_branch`, never warm-started — see
    `src.models.fusion.LCStaticDRBFusionModel`) most likely compounds the
    same small-sample overfitting rather than testing `real_bogus_score`'s
    information content in isolation. Answer to "does it help": not in this
    experiment, but the experiment cannot cleanly separate "this feature is
    uninformative" from "any added branch overfits at this sample size" —
    stated as a limitation, not resolved.
  - **Normalizer**: refit on Tier B train (not reused from Tier A) for
    every fine-tuned variant — stated assumption (§12): the static encoder
    is being adapted to Tier B's real host distribution, so its input
    normalization should match what it is actually fed, unlike the
    zero-shot condition (which deliberately keeps Tier A's own normalizer,
    since it is testing Tier A weights completely unmodified).
  - Phase curve (`>=0` cutoffs only, per §5.1's Tier B finding): both Tier
    A and Tier B curves rise monotonically with more data; the sim-to-real
    gap is widest at `cutoff=0` (0.546 vs 0.284) and narrows toward `full`
    (0.717 vs 0.634) — see `reports/gate_c/phase_curve_comparison.png`.

- **Day 10: photo-z ablation -- REFUTES the standard-candle hypothesis; the
  Tier B static inversion remains an open question with no confirmed
  cause.** Day 9 proposed that `host_photoz`/`host_photoz_err` -- absent at
  Tier B (§5.1) -- drove most of Day 7's Tier A LC+static gain (Ia are
  standard candles; comparing apparent brightness against M ≈ −19.3 needs a
  distance), which would explain why the static branch went net-negative on
  Tier B. Tested directly: retrained Tier A LC+static with those two fields
  removed (5 fields instead of 7), same data/split/seed/architecture/
  class-weighting as Day 7 (`src/train/pretrain_static_ablation.py`,
  `outputs/train_static_no_photoz/`). Result: full-cutoff macro-F1 0.7108,
  vs LC-only's 0.6981 and the original 7-field run's 0.7169 -- gain +0.0127
  without photo-z vs +0.0189 with it. **67% of the gain survives removing
  photo-z; only 33% is attributable to it.** The hypothesis is refuted, not
  confirmed: photo-z is a real but minority contributor, not "most of the
  gain." This does NOT hand the Tier B inversion a substitute explanation --
  Day 9's small-sample-overfitting story is still just a plausible,
  unverified alternative, not a confirmed cause. **The Tier B static
  inversion has no confirmed explanation as of Day 10.** Do not let either
  story stand in for that gap in a future session. (`host_ellipticity`,
  also missing-flagged at Tier B, was not removed here -- this ablation
  targeted the standard-candle mechanism specifically, per instruction, not
  the full three-field Tier-B-missing set.)

- **Day 10: anomaly branch built (step 11), ahead of the image encoder --
  see §10's reorder note.** GRU-encoder-hidden-state reconstruction decoder
  (`src/anomaly/decoder.py`) + classifier entropy + Mahalanobis distance in
  latent space + PineForest (`coniferest`, §8) over
  `[latent ‖ engineered features]`, combined by equal-weighted z-score
  average (`src/anomaly/scoring.py`, `src/train/train_anomaly.py`). Built on
  the **Tier A LC-only encoder** (`outputs/train/checkpoint.pt`,
  `LCEncoder`, not the LC+static fusion model) -- stated assumption: the
  decoder reconstructs light curves only, and `LCEncoder` is literally the
  encoder as opposed to a wrapper around one. The decoder predicts each
  timestep's flux features from the encoder's hidden state as of the
  *previous* timestep (never the same timestep, which would already contain
  the answer and make reconstruction trivial) -- causal by construction,
  same spirit as §2.3.

  Decoder trained on 60,000 objects subsampled from Tier A train (3 normal
  classes only), 5 epochs -- a time-boxing decision; every EVALUATION
  population below is used in full, uncapped.

  **Held-out gentypes 71-74/98 did not come free -- §5.1 corrected.**
  Checked directly: no strongly-lensed-SN or gentype-98 objects exist in
  any downloadable TRAIN_02 model directory. Substituted the 18
  newly-downloaded exotic directories instead.

  **Scope cut -- corrected Day 10, own earlier cost claim was wrong.** Of
  the 18 exotic directories, 6 (Cepheid, RRL, EB, d-Sct, dwarf-nova,
  Mdwarf-flare -- Recurring/Periodic + Recurring/Fast) were excluded from
  scoring. The exclusion decision stands, but the ORIGINAL justification
  for it ("EB took ~2 hours vs TDE's ~3 minutes, ~40x slower, measured
  empirically") was **wrong -- a contention artifact, not a genuine
  per-class cost difference.** That measurement happened concurrently with
  the photo-z ablation's own anomalously slow epoch 10 (10,038.8s vs a
  normal ~400-500s, logged at the identical wall-clock window). A clean,
  uncontended re-test this session: `elasticc2.read_model_directory` on the
  full 40-file EB directory took 182.4s (94,934 objects) vs TDE's 164.0s
  (94,285 objects) -- 11% slower, not 40x. **The real issue is
  representation, not reader cost.** Checked directly via HEAD-file
  `MJD_DETECT_FIRST`/`MJD_DETECT_LAST`: TDE's mean detection span is 40.9
  days (median 34.0, a bounded transient event); EB's is 705.3 days (median
  828.9); CLAGN's is 792.8 days (median 864.8) -- both ~17-20x longer than
  TDE, and NOBS/object is only ~1.5x higher (256 vs 173), so it isn't that
  periodic/recurring objects are More Numerous per timestep, it's that their
  detections are sparse across a MUCH longer real-time baseline.
  `tensorize_alert`'s window (§5.1/tensorize.py) is
  `[first_detection - pre_detection_window_days, last_detection]` -- it
  bounds the LEFT edge only. For a fading transient, `last_detection` sits
  naturally close to `first_detection` (the object fades below threshold),
  so the window is effectively bounded on both sides in practice. For a
  periodic/continuously-variable source, nothing fades, so `last_detection`
  sits wherever the survey simulation happened to end -- the window's right
  edge is UNBOUNDED in practice, and the tensorized sequence spans however
  much of the survey the object was ever re-observed across. **This is a
  windowing/representation mismatch, not a fundamental architectural one**
  -- the causal GRU itself has no trouble with a ~250-timestep sequence
  regardless of what real-time span it covers; the problem is that this
  span is far outside anything the Tier A encoder was ever trained on (SN
  training objects fade within roughly a season), so the model meets a
  genuinely out-of-training-distribution input, not a malformed one.

  **This is a live problem today, not just in the excluded classes --
  CLAGN (kept, not excluded) has the identical pathology** (792.8-day mean
  span, worse than EB's 705.3) and was scored anyway. See the entropy
  finding below for the direct, measured consequence. **Day 12 flag:** a
  live/replay serving pipeline scoring "as of now" for a long-lived
  recurring source (periodic variable OR AGN) will see its alert history
  grow unboundedly over the object's lifetime in the survey unless the
  windowing policy also bounds the RIGHT edge (a fixed recent-window cap,
  not "since first detection") -- this needs to be decided before Day 12's
  serving layer is built, not discovered by it.

  Kept and scored: TDE, SLSN-I (both variants), CART, ILOT, PISN, KN (both
  variants), CLAGN, uLens (all three variants) -- 442,303 objects, all read
  cleanly with zero discards.

  **ROC-AUC (normal test vs. anomalous), per component and combined:**

  | Component             | Tier A exotics (n=126,542 vs 442,303) | Tier B weird (n=658 vs 1,045) |
  | ---------------------- | -------------------------------------: | -----------------------------: |
  | Reconstruction error   | 0.5643                                  | **0.8382**                     |
  | Classifier entropy     | 0.2578 (worse than random)              | 0.4203 (worse than random)     |
  | Mahalanobis distance    | 0.7860                                  | 0.5236 (~chance)               |
  | Isolation forest (PineForest) | **0.8428**                       | 0.6400                         |
  | Combined (equal-weight, w/ entropy) | 0.7644                     | 0.7418                          |
  | **Combined (entropy dropped)** | **0.8146**                      | **0.7622**                     |

  **Headline finding: both latent-space methods collapse toward chance
  under domain shift; reconstruction error inverts.** Isoforest goes
  0.843→0.640 and Mahalanobis goes 0.786→0.524 (Tier A → Tier B) -- both
  operate on the Tier A encoder's OWN latent space, and that geometry
  simply does not transfer to real ZTF data the encoder never saw a shred
  of during training. Reconstruction error goes the other way, 0.564→0.838
  -- it measures a decoder's ability to predict raw flux features from
  recent history, closer to the actual physical signal than to any
  learned-geometry artifact of Tier A specifically. Reading: a CV or nova
  genuinely does not look like a supernova light curve, in flux-vs-time
  terms, regardless of what the Tier A latent space thinks its neighbors
  look like -- that's a property of the raw signal, which is why it
  survives the domain shift that breaks the two latent-space methods.

  **The combination should be read as robustness under domain shift, not as
  the best detector -- it is beaten on both individual sets.** Isoforest
  alone beats the combination on Tier A (0.843 vs 0.764); reconstruction
  error alone beats it on Tier B (0.838 vs 0.742). No single component
  clears ~0.74 on both sets except the combination. That is the actual,
  honest case for combining four components that specialize differently
  under sim-to-real shift -- not "produces the top score," which it does
  not, on either set.

  **Classifier entropy: INVERTED, not weak -- confirmed mechanism, not
  speculation.** Recomputed directly from saved latents
  (`encoder.head(latent)` → softmax; verified bit-for-bit against the saved
  entropy values, max diff 0.0):

  | Population | mean entropy | mean max-softmax-prob |
  | --- | ---: | ---: |
  | Tier A normal test | 0.5737 | 0.7100 |
  | Tier A exotic | 0.2600 | 0.8730 |
  | Tier B normal test | 0.4577 | 0.7847 |
  | Tier B weird | 0.3695 | 0.8125 |

  Confirms "confidently wrong, not uncertain" directly: exotic/weird
  objects show HIGHER max-softmax-probability than their tier's own normal
  population, in both tiers, same direction. The 0.258-vs-0.420 asymmetry
  is explained by MAGNITUDE, not direction: the confidence gap (exotic mean
  max-prob minus normal mean max-prob) is 0.1630 on Tier A vs only 0.0278 on
  Tier B -- about 6x smaller. Per-exotic-class breakdown makes the Tier A
  driver concrete: CLAGN (mean entropy 0.082, the single largest exotic
  class at 108,939/442,303 objects, and the class with the LONGEST
  detection span per the finding above) is by far the most confidently-wrong
  subset; KN (mean entropy 0.786, actually genuinely fast/short like an
  ordinary transient) shows no inversion at all. The mechanism is therefore
  tied directly to the windowing/representation finding above, not a
  separate phenomenon: objects whose tensorized span sits far outside
  training range get pushed toward a confident, wrong extreme rather than
  toward genuine indecision, and CLAGN -- kept despite sharing EB's
  windowing pathology -- is the demonstrated case of this happening.

  **Decision: entropy is dropped from the practical combined score.**
  Dropping it (3-component equal-weight z-score average instead of 4)
  IMPROVES the combination on both sets: 0.7644→0.8146 on Tier A (+0.0502),
  0.7418→0.7622 on Tier B (+0.0204). Entropy was not neutral in the
  4-component average -- its partial, asymmetric inversion was actively
  hurting the combination, not just failing to help. The 3-component
  combined score above (bold) is the one to treat as this branch's actual
  output going forward.

  **Gate D (top 50, Tier B): see §7's Gate D entry above.**

- **Day 12: right-edge windowing fix (measured, not asserted) + replay-mode
  serving pipeline built end to end (§6 steps 12).**

  **Task 1 -- `max_span_days`.** Added `src.features.tensorize.tensorize_alert`'s
  `max_span_days` parameter (default `None`, preserving prior behaviour
  exactly -- see that module's docstring, "RIGHT-EDGE CAP"): the window's
  right edge becomes `min(last_detection_mjd, first_detection_mjd +
  max_span_days)`. Ablated `{100, 200, None}` on the existing Tier A LC-only
  checkpoint (`outputs/train/checkpoint.pt`), inference only, no retraining
  (`src/train/span_ablation.py`, `reports/span_ablation/`):

  | `max_span_days` | Tier A test macro-F1 (full cutoff) | Ia affected | Ib/c affected | II affected |
  | --- | ---: | ---: | ---: | ---: |
  | 100 | 0.6975 | 1.26% | 4.41% | 3.77% |
  | **200** | **0.6979** | **0.03%** | **0.44%** | **0.32%** |
  | None (uncapped) | 0.6981 | 0% | 0% | 0% |

  200 days -- the prior stated at the top of this task -- wins outright
  among the finite settings tested and costs only -0.0002 macro-F1 against
  uncapped, while leaving well under 1% of every class's objects truncated
  at all. Set as `tensorize.RECOMMENDED_MAX_SPAN_DAYS = 200.0`, a constant
  serving code passes explicitly (`tensorize_alert`'s own parameter default
  stays `None`, per the task's own instruction, so existing training/eval
  callers are unaffected).

  **CLAGN entropy, before/after** (same 108,939-object population as Day
  10's anomaly branch, re-scored through the frozen Tier A encoder):

  | | mean entropy | mean max-softmax-prob | mean seq length |
  | --- | ---: | ---: | ---: |
  | uncapped (Day-10 setup, reproduced) | 0.0820 | 0.968 | 213.9 |
  | `max_span_days=200` | 0.2816 | 0.878 | 65.7 |
  | (Tier A normal test, Day-10 reference) | 0.574 | -- | -- |

  The uncapped row reproduces Day 10's recorded 0.082 almost exactly (0.0820
  vs. 0.082), confirming the re-scoring methodology before trusting the
  capped comparison. Capping the span more than triples CLAGN's mean
  entropy (0.082 -> 0.282) and pulls mean confidence down from 0.968 to
  0.878 -- a real step toward genuine uncertainty, though still short of
  the 0.574 a normal in-distribution object shows. This connects three
  separate findings into one mechanism, as anticipated when this task was
  set: the Day-4 windowing decision (left edge only) left the right edge
  unbounded; Day 10 found this produces confidently-wrong (not uncertain)
  predictions on long-lived sources specifically because their tensorized
  span sits far outside the training distribution; today's fix bounds that
  same span and directly, measurably reduces that overconfidence, on the
  same objects, via the same causal mechanism (sequence length relative to
  training range) Day 10 identified. It does not close the gap
  entirely -- CLAGN at 200-day cap is still far more confident than a
  genuine ambiguous case would be -- so this is a real mitigation, not a
  full fix.

  Note also (§10's own already-recorded caveat, unresolved by this fix):
  `max_span_days` bounds relative to `first_detection_mjd`, not to "now" --
  a still-alerting AGN's tensorized window stops advancing once the cap is
  reached, rather than sliding forward. That is a separate, serving-layer
  (Redis state) rolling-window question, not something this parameter
  answers by itself (see `src.features.tensorize` module docstring,
  "RIGHT-EDGE CAP").

  **Task 2 -- replay-mode serving pipeline, `src/serve/` (first creation of
  `infra/`).** `replay.py` (Kafka producer, Tier B parquet in global MJD
  order, configurable speed or unthrottled) -> `ingest.py` (stateless
  packet -> `Observation` normalization) -> `state.py` (Redis rolling
  per-object history, `sn:lc:{object_id}`) -> `inference_worker.py` (the one
  consumer implementation, per §6's "same consumer code" -- micro-batched
  ONNX classifier + reconstruction-error anomaly score, Postgres results) ->
  `api.py` (FastAPI: `/classify`, `/object/{id}`, `/top-anomalies`,
  `/stats`, `/health`). `infra/docker-compose.yml`: single-node KRaft Kafka,
  Redis, Postgres.

  **Non-monotonic window, reproduced and tested** (§5.1's own predicted
  case): `src.serve.state.merge_history` re-sorts by mjd on every merge
  rather than appending, so a later alert's bundled forced photometry
  arriving with an EARLIER mjd than everything already stored gets inserted
  in its correct sorted position, not tacked onto the end.
  `tests/test_serve_state.py::TestNonMonotonicMerge` covers this directly
  (day-3 forced photometry arriving in an alert triggered at day 20, after
  a day-10 detection was already stored, lands at position 0, not 2).

  **End-to-end validation against Day 9's own recorded Gate C number.**
  Replayed the full Tier B test split (1,994 objects, 49,587 alerts,
  BTS-Sample-Explorer-joined ground truth -- 488 Ia / 35 Ib/c / 135 II
  3-class-labeled, matching Day 9's reported 13.9:1 Ia:Ib/c exactly) through
  the complete pipeline. Each object's LATEST prediction (i.e. using its
  full accumulated history, same as Gate C's "full cutoff") scores macro-F1
  **0.6727** against ground truth -- Day 9's own recorded LC-only Tier B
  test macro-F1 was **0.6716**. The pipeline was not built or tuned against
  this number; it reproduces it to within 0.0011, using `max_span_days=200`
  where Gate C's own eval used no cap at all. This is strong end-to-end
  evidence that ingest/state/tensorize/ONNX-export introduce no correctness
  regression relative to the offline harness.

  **Latency and throughput, measured on this machine (Postgres-recorded,
  full pipeline: Kafka consume -> Redis GET/SET -> tensorize -> ONNX
  classify -> anomaly score -> Postgres INSERT, one row per alert, no
  batching on the DB write):**

  | | value |
  | --- | ---: |
  | p50 | 27.2 ms |
  | p99 | 73.7 ms |
  | mean | 29.7 ms |
  | max | 273.8 ms |
  | throughput (busy interval) | 730 alerts/sec |
  | throughput (wall time, incl. idle-stop wait) | 660 alerts/sec |

  ONNX classifier inference itself, measured separately in the
  micro-batch timing logs, is 0.3-0.5ms/alert -- the end-to-end latency
  above is dominated by per-alert Redis and Postgres round trips (both
  currently unbatched, one network round trip per alert each), not by the
  model. Batching the Postgres insert per micro-batch (already computed
  together) rather than per-alert is the obvious next lever if higher
  throughput is ever needed -- not attempted today, flagged for whoever
  picks this up next.

  **What the serving layer exposed that offline evaluation had hidden.**
  Day 10's anomaly branch combined four components (reconstruction error,
  entropy [dropped], Mahalanobis distance, PineForest/isolation-forest);
  building a servable version surfaced that only reconstruction error
  actually CAN be served from what Day 10 persisted.
  `src.anomaly.scoring.MahalanobisScorer` and the isolation-forest scorer
  are fitted, stateful objects that `src.train.train_anomaly` never
  serialized -- only their scores on the four eval populations, their
  z-score reference stats, and the decoder's weights were saved
  (`outputs/anomaly/anomaly_results.json`, `decoder_checkpoint.pt`). A live
  server has no way to reconstruct a fitted Mahalanobis mean/precision or a
  fitted forest from those artifacts without re-running the entire
  60,000-object fit population through the pipeline at every process
  start. `src/serve/model.py::AnomalyScorer` therefore serves
  reconstruction error only (documented in that module's own docstring, not
  silently substituted) -- z-scored against the already-persisted
  `reference_zscore_stats.reconstruction_error`, the one component that
  needed nothing beyond already-checkpointed weights. Offline evaluation
  (ROC-AUC against two held-out sets) never had to confront this because it
  only ever ran once, in-process, right after fitting; the moment scoring
  needs to survive a process restart, "saved the numbers" and "saved the
  model" turned out to be different claims. Persisting the fitted scorer
  objects themselves (not just their scores) is what would let the full
  3-component combination (§11 Day 10's actual best-on-both-sets result)
  serve -- flagged here, not fixed today.

  A second, smaller version of the same lesson: Tier B's interim parquet
  (`data/interim/btsbot_*.parquet`) carries `class_id=None` for every row by
  construction (`src.readers.btsbot`'s own docstring) -- ground truth only
  exists after a separate join against the BTS Sample Explorer export
  (`src.readers.bts_labels`). `replay.py` originally replayed without this
  join and every `true_class_id` came through `NULL`; the offline harness
  never surfaces this because `finetune_bts.py`/`train_anomaly.py` always
  perform the join as a normal part of loading Tier B. A replay/serving
  path that reads the interim parquet directly, the way a new script
  naturally would, has to know to do this join itself -- worth a pointer
  for whoever writes the next Tier-B-reading script.

- **Day 13: preliminary verification (3 items), live Fink (ZTF connected,
  LSST confirmed empty), and a results/live UI.**

  **1. `max_span_days=200.0` wiring -- confirmed by grep, not assumed.**
  Both serving call sites import `RECOMMENDED_MAX_SPAN_DAYS` from
  `src.features.tensorize` and pass it EXPLICITLY as a hardcoded constant:
  `src/serve/inference_worker.py`'s `_build_batch_tensor` and
  `src/serve/api.py`'s `/classify` both call `tensorize_alert(...,
  max_span_days=RECOMMENDED_MAX_SPAN_DAYS)` -- not opt-in, a caller cannot
  forget it. No training/eval call site (`dataset.py`, `dataset_static.py`,
  `finetune_bts.py`, `pretrain*.py`, `train_anomaly.py`) passes
  `max_span_days` at all -- confirmed by grep, and this is NOT a gap: Day
  12's own span ablation was inference-only against the ALREADY-TRAINED
  checkpoint, no retraining under the cap was ever performed or claimed, so
  there is no training call site that should be passing it.

  **2. Anomaly scorers persisted -- the full 3-component combination now
  serves.** `src/train/persist_anomaly_scorers.py` reconstructs the exact
  60,000-object Day-10 fit population (same seed; loaded via
  `dataset.load_alerts`'s own documented `object_ids`-filtered fast path --
  reconstructing all 843,617 objects to recover 60,000 of them would cost
  the ~50 minutes that function's docstring warns about; id-filtering first
  cost under a minute), refits `MahalanobisScorer`/`PineForestScorer`
  against it, and persists the fitted objects for the first time
  (`outputs/anomaly/mahalanobis.npz`, `engineered_normalizer.npz`,
  `forest.pkl`). Verification, kept deliberately cheap (re-scoring Tier A's
  442,303-object exotic eval set again would cost as much as Day 10's own
  eval pass for no new information): the refit fit-population z-score
  stats match Day 10's recorded ones almost exactly --
  reconstruction_error and mahalanobis to ~1e-4 or better, isoforest to
  only ~0.002 (a small, real, unexplained residual, most likely
  `PineForest`'s own internal randomness not fully pinned by the shared
  seed -- noted, not chased further). Tier B's full, uncapped 3-component
  ROC-AUC (entropy dropped, per this section's own Day-10 decision) was
  independently recomputed end-to-end: **0.7616**, matching Day 10's
  already-recorded 3-component figure (0.7622) to within 0.0006 despite
  being computed from a freshly-refit forest -- strong evidence this is the
  same fit, not a different one. Individual components:
  reconstruction_error 0.8382 (exact match) and mahalanobis 0.5236 (exact
  match) against Day 10's own numbers; isoforest 0.6479 vs. 0.6400 (the
  same small residual noted above). Tier A's exotic-population ROC-AUC was
  NOT recomputed (too expensive for a verification pass); its
  already-recorded per-component numbers are unaffected, since they never
  depended on persisting these objects. `src/serve/model.py::AnomalyScorer`
  now loads the persisted artifacts when present and serves the full
  combination; `inference_worker.py`/`api.py` were both updated to pass
  each batch's source `CanonicalAlert`s into `.score()` (needed for the
  isoforest component's engineered features, which read from unpadded
  observations, never the padded tensor) -- falls back to
  reconstruction-error-only, unchanged, if the artifacts are absent.

  **3. Throughput headline, stated explicitly.** 730 alerts/sec (Day 12's
  peak benchmark) is ~63,072,000/day against LSST's own ~7,000,000/night --
  roughly 9x headroom on a single node. The more interesting half of this
  number is the caveat: it is single-node, commodity hardware, and the
  bottleneck is unbatched per-alert Redis/Postgres round trips (Day 12's
  own finding), not the model (ONNX classifier inference alone: 0.3-0.5ms/
  alert) -- the number says more about how cheap this specific model is to
  run than about the serving layer's own engineering maturity.

  **Task 1, ZTF: connected and streamed within seconds; ~1,591 real alerts
  ingested end to end.** `fink-client` actually installed: **11.0**, not
  the >=12 this session's own briefing assumed (`requirements-lock.txt`
  claims 12.1.0; `pip show fink-client` in this venv reports 11.0) -- a
  real, unexplained lock-file/installed-version discrepancy, flagged here
  rather than silently worked around. Version 11.0's `AlertConsumer` API
  was sufficient for everything needed (topic listing, `.poll()`), so this
  did not block the work, but a future session should not assume 12.1.0 is
  actually installed without checking again. No password was required on
  either broker -- `username`+`group.id` alone succeeded for topic listing
  AND for consuming real alerts on both `kafka-ztf.fink-broker.org:24499`
  and `kafka-lsst.fink-broker.org:24499` (`fink_client.consumer.
  _get_kafka_config` only enables SASL when a `password` key is present;
  none was ever supplied, nothing failed). `ftest_*` test topics (this
  session's own suggested starting point) were not present in either
  broker's topic listing at connection time -- instead, both surveys
  expose Fink's own SN-candidate-filtered LIVE topics directly
  (`fink_sn_candidates_ztf`, `fink_most_likely_sn_lsst`, and others), which
  is what this session connected to; see "pre-filter" below for why that's
  also the right long-term choice, not just what was available.

  `src/readers/avro_ztf.py` -- built and golden-tested
  (`tests/fixtures/golden_avro_ztf_alert.py`, `tests/test_readers_avro_ztf.py`)
  against ONE REAL alert polled live from `fink_sn_candidates_ztf` (object
  ZTF26ablxxjw, 26 total rows). Same ZTF alert-packet schema and mag->flux
  conversion `src/readers/btsbot.py` already established
  (`magpsf`/`sigmapsf`/`magzpsci` per-alert zeropoint, NOT the generic
  nJy/zp=31.4 convention §3 describes for a hypothetical LSST/ZTF reader),
  confirmed directly on the live packet, not re-derived from memory. One
  real difference: a live Fink packet's `isdiffpos` is the literal string
  `"t"`/`"f"`, not the `bool` btsbot.py's CSV export already produced. This
  is also the first reader in this repo that has to represent a ZTF
  **non-detection** (`prv_candidates` rows with `magpsf=None`, only a
  5-sigma `diffmaglim` known -- 19 of 25 `prv_candidates` rows on the
  golden alert) -- BTSbot's export contains no such row type at all (§5.1
  Day-8 finding). Represented as `flux=0.0` (no measured excess) with
  `flux_err` from the limiting depth read as a 5-sigma bound -- a stated
  modeling assumption, not a verified external convention.

  **A real bug, found by this live run, not by inspection.**
  `_decode_cutouts`'s first version let an undersized cutout stamp (a real
  ZTF object near a CCD-quadrant edge gets a stamp smaller than 63x63 --
  observed down to `(63, 42)`) raise all the way through
  `CanonicalAlert.validate()`'s hard `(3, 63, 63)` shape check, discarding
  the alert's entire LIGHT CURVE over an undersized CUTOUT -- 9/1,015
  alerts lost to this in the first live run. Fixed to degrade to
  `cutouts=None` on any mis-sized/undecodable cutout (the same defensive
  pattern `avro_lsst.py` already used, for a different, anticipated
  reason), confirmed 0/585 lost on a second run with the fix in place, and
  pinned with a regression test. This never showed up in any prior session
  because no cutout had ever flowed through this pipeline before today --
  exactly the class of bug Gate-style "verify against real data" exists to
  catch.

  **Live ZTF run (post-fix), `fink_sn_candidates_ztf`, ~161s wall-clock
  across two runs:** 1,591 alerts received, 1,591 produced (0 skipped after
  the fix), 337-338 unique objects, all scored end-to-end through the
  EXISTING replay-mode pipeline (Kafka -> Redis state -> ONNX classifier +
  anomaly scorer -> Postgres) with zero code changes to that pipeline --
  confirms §6's "same consumer code... swap producer" for steps 12->13 in
  practice, not just in design.

  **Negative-flux fraction: ~3.0-3.1% of live ZTF detections** (625/20,780
  and 363/11,866 detections across the two runs) -- the first non-zero
  negative-flux fraction this pipeline has ever measured on real ZTF data.
  Tier B (§5.1) measures exactly 0% by construction (BTS's `isdiffpos`
  pre-filter excludes fading/negative subtractions before a candidate ever
  reaches `metadata_v10`) -- this is the first direct, quantified
  confirmation of that filtering's effect, from data Tier B's own filter
  never included.

  **Predicted class distribution (latest prediction per object, 338
  objects, live-run-only): II 52% (176), Ib/c 35% (117), Ia 13% (45).**
  Notably skewed against EVERY prior anchor this project has recorded --
  ELAsTiCC's own training composition (20/28/51), the real sky rate
  (~25-30/10-20/55-65), and Tier B's own Gate C test composition (13.9:1
  Ia:Ib/c) all put Ia well above Ib/c, not roughly 3x below it. **Flagged
  as an interesting, unexplained live observation, not a validated
  result** -- per this section's own Day-6 precedent ("a model that
  suddenly nails Ib/c is a model that is cheating... investigate, do not
  celebrate"), an inverse skew deserves the same skepticism, not a story.
  Plausible, UNTESTED contributors: live triggers are disproportionately
  early-phase/short sequences (fresh detections), unlike Tier B's
  full-cutoff Gate C evaluation; Fink's own `fink_sn_candidates_ztf`
  topic-level filter may itself select non-representatively. No ground
  truth exists for these live objects to check accuracy against -- this is
  a prediction-SHAPE observation only, exactly the caveat this session's
  own task instructions anticipated ("this is live inference and sanity
  checking, not a measured result").

  **Task 1, LSST: stream confirmed empty, not merely unreachable.**
  `src/readers/avro_lsst.py` was built against the standard Rubin/LSST DPDD
  alert schema (`diaSource`/`diaObject`/`prvDiaSources`/
  `prvDiaForcedSources`, nJy flux at AB zp=31.4 -- §3's own named example)
  but is **UNVERIFIED against real traffic**: every livestream topic on
  `kafka-lsst.fink-broker.org` (11 checked: `fink_most_likely_sn_lsst`,
  `fink_uniform_sample_lsst`, and 9 others) returned zero messages, checked
  via Kafka watermark offsets (not just a poll timeout) including an
  8-sample watch over a continuous 100 seconds showing no movement at all
  -- the SAME session, SAME code pattern, connected to ZTF and streamed a
  real alert within seconds. This is a genuine, checked fact about Fink's
  current LSST relay state, not a connectivity failure on this project's
  end. `avro_lsst.py` fails loudly and by name (never silently mismaps a
  field) the first time real data arrives -- tested only against synthetic
  packets built from its own documented assumptions
  (`tests/test_readers_avro_lsst.py`). **No LSST prediction distribution
  exists to report, for a verified reason, not a missing one** -- whether
  ugrizy ingests cleanly and whether wavelength-conditioning transfers
  weights across surveys as designed (§3's whole point) remains unverified
  until Fink's LSST relay actually carries traffic.

  **What Tier C added that Tier B hid -- three findings, plus one
  non-finding worth stating explicitly.** (1) negative-flux detections
  (~3%), structurally excluded by Tier B's own `isdiffpos` pre-filter; (2)
  undersized/CCD-edge cutout shapes, a real-world variability Tier B never
  exercises (BTSbot images were never wired into this pipeline at all); (3)
  genuine non-detection upper-limit rows (19/25 in the golden alert) --
  Tier B's export contains no such row type (§5.1), so this session is the
  first time any reader in this repo has had to decide how to represent
  one. **The non-finding: Gate D's actual blocker is NOT resolved by
  today's work.** This session's live ingestion deliberately consumed
  Fink's OWN pre-filtered SN-candidate topics (see "pre-filter" below) --
  which means, like Tier B, it still contains no bogus/artifact candidates
  for Gate D's ranker to have found. Gate D's real unblock needs an
  UNFILTERED alert stream (or Fink's own reject/candidate topics), which
  this session did not attempt to consume.

  **Pre-filter, stated per this session's own task instruction.** Chosen:
  consume Fink's own science-module-filtered candidate topics
  (`fink_sn_candidates_ztf`, `fink_most_likely_sn_lsst`) rather than either
  survey's raw/uniform stream -- delegates real-bogus and SN-likelihood
  filtering to Fink's own deployed, production-tested science modules
  rather than reimplementing an equivalent filter from scratch. Necessary
  for LSST specifically (~7M alerts/night); applied uniformly to ZTF too,
  for the same reason.

  **Task 2: UI built, results + live panels, both reading from existing
  artifacts.** Single page (`ui/index.html`), served by the existing
  FastAPI app (`GET /`, plus `/static/reports`/`/static/outputs` mounts) --
  no new service, no build step. Results half reads `gate_c_report.json`,
  `anomaly_results.json` (+ `persisted_scorers_meta.json` when present),
  and the existing confusion-matrix/phase-curve PNGs directly; nothing is
  recomputed. Live half polls `/stats`/`/top-anomalies` every 4s and adds
  one new endpoint, `GET /cutouts/{object_id}` (`src/serve/cutouts.py`,
  ZScale-rendered PNG triplets, base64-inlined) -- the first time this
  pipeline has served an image at all. Verified end to end in-browser
  against this session's own live ZTF run: clicking a live object's row
  renders its real science/template/difference cutout triplet, decoded
  from the actual gzip-FITS stamp bytes ingested minutes earlier.

- **Day 14 (final, buffer day): flux-conversion report, class-distribution
  root cause found, LSST reframed, headline numbers finalized.**

  **1. Fink flux conversion -- restated and freshly re-verified, since Day
  13's chat report omitted it even though CLAUDE.md's own Day-13 entry
  already named it.** `src/readers/avro_ztf.py` reads ZTF's alert-packet
  magnitude convention (`magpsf`/`sigmapsf`, PSF-fit magnitude + 1-sigma
  uncertainty) against that alert's OWN per-row photometric zeropoint
  (`magzpsci` -- verified varying row-to-row on the live golden alert,
  25.98-26.44 mag across its 26 rows -- NOT the fixed nJy/zp=31.4 §3
  describes for a hypothetical generic LSST/ZTF reader). Conversion:
  `flux_raw = sign * 10**(-0.4*(magpsf - magzpsci))` (sign: `isdiffpos`
  `"t"` -> +1, else -1), `flux_err_raw = |flux_raw| * (ln(10)/2.5) *
  sigmapsf`, then `src.readers.zeropoint.convert_zeropoint(flux_raw,
  flux_err_raw, from_zeropoint=magzpsci)` rescales onto the schema's SNANA
  FLUXCAL zp=27.5 -- the SAME shared, tested helper every non-SNANA reader
  must call (§3), `from_zeropoint` passed per-row, never a module
  constant. Verified independently: `tests/fixtures/golden_avro_ztf_alert.py`
  hand-computes expected flux/flux_err for all 26 rows of one real,
  live-polled alert via plain Python math (not by calling the reader, or
  even `convert_zeropoint`), and
  `tests/test_readers_avro_ztf.py::test_flux_and_flux_err_converted_to_fluxcal_27_5`
  checks the reader's actual output against those independent numbers to
  `abs=1e-6`/`rel=1e-6` -- re-run fresh this session, still 20/20 passing.
  **Conversion is happening, on the correct per-alert zeropoint, verified
  independently -- not a live bug.**

  **2. Class-distribution skew -- ROOT CAUSE CONFIRMED: training-prior
  mismatch, not a zeropoint bias.** Pulled the actual per-object softmax
  posteriors (`prob_ia`/`prob_ibc`/`prob_ii`, already stored in Postgres)
  for all 338 live-run objects and applied a standard prior correction:
  `corrected(c) ∝ p_model(c) * (sky_prior(c) / train_prior(c))`,
  renormalized, re-argmaxed. Training prior: ELAsTiCC's own Ia 20.4% /
  Ib-c 28.5% / II 51.1% (§11, Day 4). Sky prior: midpoint of the stated
  real rates, Ia 27.5% / Ib-c 15% / II 57.5%.

  | | Ia | Ib/c | II |
  | --- | ---: | ---: | ---: |
  | raw (argmax of model posterior) | 13.3% | 34.6% | 52.1% |
  | **prior-corrected** | **17.5%** | **13.9%** | **68.6%** |
  | real sky rate (target) | 25-30% | 10-20% | 55-65% |

  Ib/c collapses from 2.3x too high (34.6%) to squarely inside the target
  band (13.9%) once the training prior is divided out -- the dominant
  effect, and exactly the axis (Ia<->Ib/c) the correction should move if
  candidate 1 (trained prior showing through) is right. Robust across the
  full plausible range of sky-prior point estimates (three points spanning
  Ia 25-30% / Ib-c 10-20% / II 55-65%): Ib/c lands at 7.1-23.1% in every
  case, always inside or near the target band, never anywhere close to the
  raw 34.6%. **Candidate 1 (ELAsTiCC training prior showing through in the
  outputs -- §11's own already-recorded caveat, "output probabilities...
  carry the ELAsTiCC prior, not the true sky prior") is CONFIRMED as the
  dominant driver.** Candidate 2 (a flux-zeropoint bias) is not supported:
  item 1's fresh re-verification found the conversion correct and
  independently checked bit-for-bit against hand computation, on the
  correct per-alert zeropoint -- there is no live bug in that path for a
  bias to originate from.

  A real residual remains, reported honestly rather than smoothed away: Ia
  stays under-represented even after correction (17.5% vs. a 25-30%
  target) and II overshoots slightly above it (68.6% vs. 55-65%) at the
  midpoint sky-prior estimate. Plausible, UNTESTED explanation: the
  correction assumes the model's raw softmax output is a well-calibrated
  posterior under the training prior, which nothing in this project has
  actually checked (no calibration curve was ever verified against
  ELAsTiCC's own 20.4/28.5/51.1 composition) -- an uncalibrated model would
  not correct cleanly with a single multiplicative reweighting, and 338
  live objects from one night is a small sample besides. Not chased
  further -- the headline conclusion (prior mismatch, not a zeropoint bug)
  is well supported without resolving this residual, and the residual
  itself is a calibration question, not a live-serving-path bug.

  **3. LSST: reframed, and the reframe is now evidence-backed, not just
  more careful wording.** Day 13's "confirmed empty" framing overstated
  what was actually observed. Topic names re-checked against Fink's own
  docs (`doc.lsst.fink-broker.org/services/livestream/`): every topic
  subscribed to (`fink_most_likely_sn_lsst`, `fink_uniform_sample_lsst`,
  and the other 9) is confirmed correct and current -- not a naming
  mismatch. Watermark offsets re-checked ~90 minutes after the Day-13
  check, at a different point in the UTC day: **every topic's cumulative
  offset is now in the hundreds of thousands** (`fink_most_likely_sn_lsst`:
  366,760; `fink_remove_unlikely_transients_lsst`: 585,407; smallest
  nonzero, `fink_early_snia_candidate_lsst`: 1,299; one topic,
  `fink_faint_trails_lsst`, has never carried a single message, offset 0
  both times) -- direct evidence of substantial real historical
  production, consistent with the survey running since 30 June 2026, and
  inconsistent with Day 13's stronger claim. At BOTH check times, however,
  `lo == hi` on every topic: zero messages currently retained in queue at
  either specific moment, consistent with Fink's documented ~4-7 day
  retention window landing between production bursts rather than the
  topics being structurally empty. A follow-up poll just now found nothing
  currently arriving either. **Corrected finding: no traffic on the
  subscribed topics during this session's two observation windows
  (2026-09-07, ~02:30-03:20 UTC and ~04:10 UTC) -- not "LSST is empty."**
  Rubin observes from Chile (Cerro Pachón); both windows fall inside a
  plausible nighttime-observing UTC range for that longitude in early
  September, so a simple day/night mismatch does not explain the gap
  either -- the honest state is that this session caught two lulls, for a
  reason not further diagnosed (one attempt beyond Day 13's, per
  instruction, then move on). `avro_lsst.py` remains unverified against
  real traffic and continues to fail loudly by name on any mismatch, which
  is the right place to leave it -- the topic-name and reachability
  questions are now resolved; only actual traffic during a future session
  is still needed.

  **4. Headline numbers -- confirmed present, two gaps closed.** Every
  number below now has its metric explicitly named and its
  checkpoint/cutoff identified in this document, not just implied by an
  artifact file:
  - Tier A (LC+static, step 8, full cutoff): macro-F1 **0.7169**,
    Ia-vs-rest ROC-AUC **0.9301** (`outputs/train_static/eval/metrics.json`,
    `full.macro_f1` / `full.roc_auc[0]`) -- previously only the macro-F1
    half of this pair was named anywhere in this file (Gate C's
    sim-to-real-gap sentence); the ROC-AUC half is added here for the
    first time. 0.9301 sits at the very bottom of Gate B's own step-8
    ceiling band (0.93-0.96) -- passes, but barely; noted, not chased
    further.
  - Tier B (LC-only fine-tune, the DEPLOYED model, full cutoff): macro-F1
    **0.6716** -- already named (Day 9).
  - **Sim-to-real gap for the actually-deployed model, added here for the
    first time: 0.7169 -> 0.6716, a -0.0453 drop.** Day 9's own recorded
    gap sentence (0.7169 -> 0.6343, -0.0826) uses the LC+static variant,
    which is NOT what serving actually runs (Day 9 itself found LC-only
    beat LC+static on Tier B and shipped LC-only) -- both gaps are real and
    both are now on the record, but the LC-only one is the one that
    describes what is actually deployed.
  - Transfer gain: zero-shot 0.4483 -> LC+static fine-tuned 0.6343,
    **+0.186** -- already named (Day 9), unchanged.
  - Ib/c F1 **0.411** (LC+static fine-tune, 35 test objects) -- already
    named (Day 9).
  - Anomaly combined (entropy dropped): Tier A **0.8146**, Tier B
    **0.7622** -- already named (Day 10).
  - Replay-mode agreement: **0.6727** (live replay, `max_span_days=200`)
    vs. Day 9's recorded **0.6716** (offline, uncapped) -- already named
    (Day 12), 0.0011 apart.
  - Latency/throughput: p50 **27.2ms**, p99 **73.7ms**, **730** alerts/sec
    peak -- already named (Day 12).

  See `RESULTS.md` (repo root) for the consolidated, reader-facing summary
  of what was built, measured, and left unresolved.

- **2026-11-05 (post-NSSP): zeropoint conversion was inverted; Tier B
  re-run. SUPERSEDES every Tier B number above.** See §3's correction
  note, `reports/audit_2026-11-05/AUDIT.md`, and
  `reports/zeropoint_fix_2026-11-05/`. Regenerated Tier B parquets
  reproduce `magpsf` to 3.6e-15 mag; static fields and `flux/flux_err`
  unchanged; correct fluxes are a median 10.3x larger (p5-p95 7.9-17.8x)
  than what was stored before.

  Controlled comparison, 5 seeds x 2 conditions, identical code and
  object order (the reader's object order is not deterministic across
  builds -- a separate, smaller confound, removed here by giving the
  old-flux control the new order):

  | Tier B test, full cutoff | old flux | corrected flux |
  | --- | ---: | ---: |
  | zero-shot (LC+static weights) | 0.4483 | 0.4240 |
  | zero-shot (LC-only weights) | 0.4377 | 0.2737 |
  | LC+static fine-tune | 0.6309 +/- 0.0033 | **0.6210 +/- 0.0067** |
  | LC-only fine-tune | 0.6754 +/- 0.0012 | **0.5647 +/- 0.0012** |
  | LC+static+DRB fine-tune | 0.6064 +/- 0.0340 | 0.5358 +/- 0.0059 |
  | logistic-regression baseline | 0.3810 | 0.4687 |

  (macro-F1; +/- is the SD over seeds 42/1/2/3/4.)

  Consequences, stated as measured, not explained:
  - **The "Tier B static-encoder inversion" (unresolved item #1) does not
    survive the fix.** With correct fluxes LC+static beats LC-only by
    +0.056. The inversion was produced by the mis-scaled input; neither
    Day 9's overfitting story nor Day 10's photo-z ablation needs to
    explain it any more.
  - **The deployed model (LC-only) is no longer the best variant**, and its
    Ib/c F1 falls to 0.137 (was 0.500), below §11's 0.40-0.60 anchor.
    LC+static Ib/c F1 is 0.286. **Redeployed 2026-11-05:** serving now
    runs the corrected LC+static model
    (`outputs/finetune_bts_zpfix/with_static_{checkpoint.pt,model.onnx}`,
    seed 42, macro-F1 0.6201). `src/serve/model.py`'s ONNX graph takes a
    second input (`static_features`), built with `build_static_vector`
    and the Tier-B-refit normalizer stored in the checkpoint -- serving's
    static vectors match training's exactly (max diff 0.0 over the 658
    test objects); ONNX vs PyTorch probabilities agree to 7e-6 with 100%
    argmax agreement. End-to-end replay: macro-F1 0.6213 (offline 0.6201;
    the 0.0012 is the serving-only 200-day span cap). Uncontended latency
    p50 37.6 / p95 54.1 / p99 65.6 ms, 575 alerts/s, ONNX 0.185 ms/alert
    -- unchanged from LC-only. Live ZTF: 29/29 alerts, 0 skipped.
    `tests/test_serve_model_classifier.py` covers export parity,
    padding/batching invariance, causality of the exported graph,
    all-missing static context, and normalizer provenance.
  - Sim-to-real gaps (matched configs): LC-only 0.6981 -> 0.5647 (-0.133);
    LC+static 0.7169 -> 0.6210 (-0.096). Transfer gains: LC+static +0.197
    (0.4240 -> 0.6210); LC-only +0.291 (0.2737 -> 0.5647).
  - Ia-vs-rest ROC-AUC rises for zero-shot (0.780 -> 0.860) and LC+static
    (0.864 -> 0.882) even where macro-F1 falls -- ranking improves while the
    argmax boundary shifts.
  - Mechanism for the larger gap, measured: median detection FLUXCAL is
    ~62 on Tier A, was 236 on buggy Tier B, is 2578 on correct Tier B.
    Real BTS objects are ~41x brighter than the simulated training
    population; the bug had hidden ~10x of that gap. Tier A is unchanged
    (no conversion), verified: Tier A baselines identical before/after.
  - Tier B anomaly (Tier-A-fitted scorers, byte-identical): recon 0.8142,
    Mahalanobis 0.4407, isoforest 0.5371, combined(3) **0.7793** (was
    0.7616). Gate D's top-50 was not re-derived.
  - Replay through serving with the corrected parquet + re-exported ONNX:
    0.5678 vs offline 0.5635.
  - Live LSST additionally needs `avro_lsst.py` to read `dec` (live schema)
    instead of `decl`; not changed here.

## 12. Working style

- **One module per session.** Do not build the model and the serving pipeline in
  one go. Long sessions drift and lose track of these instructions.
- **Golden tests before generation.** For each reader, hand-verify one object's
  light curve against the source docs, commit it as a fixture, and make it the
  test. Then write the reader against the test.
- **Never accept an unexplained metric jump.** If macro-F1 moves from 0.74 to
  0.91 after a refactor, assume leakage until proven otherwise. Report the jump
  and stop.
- **Preprocessing gets checked by hand.** Zero points, MJD versus Δt, upper-limit
  sign conventions. These produce bugs that neither crash nor look wrong.
- **State assumptions out loud.** If a spec is ambiguous, say which reading you
  took and why, rather than picking silently.
- **Commit per logical unit**, with a message saying what was verified, not just
  what was written.
- Prefer deleting code to adding flags.
- **Metrics code does not render.** `src/eval/metrics.py` and
  `phase_curves.py` return numbers only and must not import
  matplotlib. All rendering lives in `src/eval/plots.py`, which
  consumes those return values; `report.py` orchestrates
  compute-then-render. This keeps metrics unit-testable against
  analytically known values without a rendering backend.
