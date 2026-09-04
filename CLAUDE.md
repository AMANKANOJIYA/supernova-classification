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
- anything under `src/eval/`

If a task appears to require changing them, stop and ask. A "fix" to the eval
harness that improves a metric is the failure mode this rule exists to prevent.

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

---

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

**Held-out anomaly classes come free.** SNANA gentypes 71–74 (strongly lensed SN
Ia / II / Ib / Ic) and gentype 98 were deliberately excluded from the ELAsTiCC
training set. Use them as a genuinely unseen evaluation set for the anomaly
branch. Do not let them leak into training.

**BTSbot public data is partial.** The Zenodo release covers ZTF public
(`programid=1`) alerts only, while the production BTSbot also trained on ZTF
partnership data. We therefore cannot exactly reproduce their published numbers,
and should not claim to.

**Read `data/raw/elasticc2/A_FORMAT.TXT` before writing the SNANA reader.** It is
the format spec. SNANA ships HEAD and PHOT FITS file pairs; they are readable as
standard FITS tables but the row-pointer convention linking HEAD to PHOT is not
obvious. Do not guess it.

---

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
   Macro-F1 must collapse to chance (~0.33). If it does not, there is label
   leakage.
2. **Causality test.** Truncate a sequence at timestep `t`. Assert the prediction
   at `t` is bit-identical with and without timesteps after `t` present.
3. **Ceiling check.** Binary Ia-vs-rest should land at 0.93–0.96 on simulations.
   Above 0.99 means something leaked, usually a metadata column encoding the
   answer.

### Gate C — after fine-tuning (step 9)

Accuracy will drop from simulation to real ZTF. That drop is the expected result
and must be recorded. If it does not drop at all, the evaluation split is wrong.

### Gate D — after anomaly branch (step 11)

Manually inspect the top 50 anomalies. Expect them to be artifact-dominated. A
published run over 2.25 million ZTF objects found 68% of flagged outliers were
bogus light curves from subtraction artifacts and blending. If our top 50 is
_not_ artifact-heavy, real-bogus filtering is over-aggressive and is suppressing
real signal too.

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

---

## 11. Expected results — sanity anchors

| Measurement                  | Expected      | Interpretation if far off              |
| ---------------------------- | ------------- | -------------------------------------- |
| Ia vs rest, binary, Tier A   | 0.93–0.96     | <0.90 preprocessing bug; >0.99 leakage |
| 3-class macro-F1, LC only    | ~0.70         |                                        |
| 3-class macro-F1, + metadata | 0.80–0.85     | metadata should give a clear jump      |
| **Ib/c recall**              | **0.40–0.60** | this looks bad and is correct          |
| Tier A -> Tier B drop        | noticeable    | no drop means broken eval split        |

Published state of the art for context: ATAT reports macro-F1 82.9 ± 0.4 over 20
ELAsTiCC classes; ATCAT reports 71.8% F1 light-curve-only and 89.8% with
metadata. Both ORACLE and ATAT report SN subtypes as their weak spot. **A model
that suddenly nails Ib/c is a model that is cheating.** Investigate, do not
celebrate.

Primary reported figure is **macro-F1 as a function of days since first
detection**, plus confusion matrices at fixed time cutoffs. Plain accuracy is not
a reportable metric here — always-predict-Ia scores well and is useless.

---

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
