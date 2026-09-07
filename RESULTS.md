# Results — Transient Supernova Classifier

14-day solo build. Full narrative, day-by-day, with every measurement and
its context lives in `CLAUDE.md` §11 — this document is the short version:
what got built, what got measured, what the gates showed, and what is
still open. If a number here and a number in `CLAUDE.md` ever disagree,
`CLAUDE.md` is the source of truth; this file is a summary of it, not a
replacement.

## What was built

A multi-survey pipeline that classifies transient alerts into SN Ia / Ib-c
/ II from light curves (+ host/static features), pretrained on simulated
LSST data and fine-tuned on real ZTF data, served over a live alert stream
with an anomaly-ranking side branch and a results/monitoring UI.

- **`src/schema.py`** — the frozen `CanonicalAlert` contract every survey
  converts into (flux on a single shared zeropoint, wavelength not band
  index, always-sorted observations).
- **Readers** (`src/readers/`) — ELAsTiCC2 (SNANA FITS, Tier A/pretrain),
  BTSbot (real ZTF + spectroscopic labels, Tier B/fine-tune), and, as of
  Day 13, live Fink alerts for ZTF and LSST (Tier C/serving).
- **Model** — causal GRU light-curve encoder + optional static/host branch,
  WHXE hierarchical loss built but not adopted (see Unresolved).
- **Anomaly branch** — reconstruction-error decoder + Mahalanobis distance
  + isolation forest (PineForest), combined by z-score average; the fitted
  scorers were persisted for the first time on Day 13, so serving now runs
  the full combination, not reconstruction error alone.
- **Serving** — Kafka → Redis (rolling per-object state) → ONNX classifier
  + anomaly scorer → Postgres, in replay mode (Day 12, archived Tier B
  data as producer) and live mode (Day 13, real Fink alerts as producer) —
  the same consumer code either way, exactly as planned.
- **UI** (`ui/`) — single page, served by the existing FastAPI app: a
  results panel reading the existing report JSONs/figures, and a live
  panel polling `/stats`/`/top-anomalies` with per-object cutout images.

## What was measured

| Measurement | Value | Context |
| --- | --- | --- |
| Tier A, Ia-vs-rest ROC-AUC (LC+static, step 8, full cutoff) | **0.9301** | Gate B ceiling band 0.93–0.96 — passes, at the low end |
| Tier A, 3-class macro-F1 (LC+static, full cutoff) | **0.7169** | anchor: ~0.70 (LC-only), 0.80–0.85 (+metadata) |
| Tier B, 3-class macro-F1 (LC-only fine-tune, **deployed**) | **0.6716** | |
| Sim-to-real gap, deployed model | **0.7169 → 0.6716 (−0.0453)** | expected direction; a *different*, larger gap (−0.0826) holds for the LC+static variant, which was not shipped |
| Zero-shot → fine-tuned transfer gain (LC+static) | **+0.186** macro-F1 | fine-tuning is doing real work |
| Ib/c F1 (Tier B, LC+static, 35 test objects) | **0.411** | inside the 0.40–0.60 anchor band — expected to look bad, correctly does |
| Anomaly branch, combined score (entropy dropped) | Tier A **0.8146** / Tier B **0.7622** | best-on-both-sets combination; no single component wins both |
| Replay-mode vs. offline agreement | **0.6727** (live) vs. **0.6716** (offline) | 0.0011 apart — end-to-end pipeline reproduces the offline number |
| Serving latency | p50 **27.2 ms** / p99 **73.7 ms** | full pipeline, Kafka→Postgres, one alert at a time |
| Serving throughput | **730** alerts/sec peak | ≈63M/day vs. LSST's ≈7M/night — bottlenecked on unbatched Redis/Postgres round trips, not the model |
| Live ZTF ingestion (Day 13) | 1,591 real alerts, 338 objects, 0 dropped | via Fink, through the *unmodified* replay-mode pipeline |
| Live negative-flux fraction | ~3.0–3.1% | Tier B measures exactly 0% by construction (§5.1) — first real confirmation of that filter's effect |

## What the gates showed

- **Gate A (readers):** passed. Negative fluxes present in parsed data
  (e.g. one ELAsTiCC2 model directory measured a 35.6% negative-flux
  fraction). Real BTS class counts (Ia 5,024 / Ib/c 364 / II 1,287,
  train+val+test) land at a 13.8:1 Ia:Ib/c ratio, in line with the ~14.5:1
  anchor.
- **Gate B (causal model):** passed, at a corrected scale. The shuffled-
  label test failed at first (450 objects, too small a sample — see
  `CLAUDE.md` §7) and passed once raised to 5,000 stratified objects.
  Causality (truncation-invariance) and the ROC-AUC ceiling both hold.
- **Gate C (fine-tune):** passed. Sim-to-real accuracy drop is present and
  in the expected direction (see table above) — the alternative, no drop
  at all, would have meant a broken eval split.
- **Gate D (anomaly inspection):** **deferred, not failed.** The only eval
  population available (Tier B "weird" objects) is spectroscopically
  confirmed real objects only — there was no bogus/artifact candidate in
  the pool for the ranker to have found, so the gate's actual question
  ("does real-bogus filtering suppress real signal") was never actually
  testable this project. Live Fink ingestion (Day 13) does not change
  this: it deliberately consumed Fink's own pre-filtered SN-candidate
  topics, which has the same property. Still open — see below.

## What remains unresolved

This is deliberately the most detailed section. Everything here is a real,
open question, not a hedge.

1. **Tier B static-encoder inversion — no confirmed cause.** Adding host/
   static features helped substantially on Tier A (simulated) but made
   Tier B (real ZTF fine-tune) measurably *worse* (LC-only 0.6716 vs.
   LC+static 0.6343) — not just non-additive, net negative. The leading
   hypothesis (host photo-z, absent at Tier B, was doing most of the
   Tier-A gain as a distance proxy for Ia standard-candle brightness) was
   tested directly with a photo-z-ablated retrain and **refuted**: only
   33% of the Tier A gain is attributable to photo-z, not "most of it."
   No alternative hypothesis has been confirmed. A small-sample-overfitting
   story remains plausible (Tier B fine-tuning has only 5,407 labeled
   training objects, and 3 of the static branch's 7 inputs are
   permanently missing there) but was never isolated with its own ablation.
   The currently-deployed model (LC-only) sidesteps this by not using
   static features at all — this is a real limitation of the static
   branch specifically, not of the deployed model.

2. **Gate D's real question is still unasked.** See above — it needs an
   unfiltered alert stream (or Fink's own reject/candidate topics), which
   this project never consumed. Everything built (Tier B "weird" eval,
   Day 13's live Fink ingestion) has, by construction or by choice,
   excluded bogus/artifact candidates from the population the ranker
   ever saw.

3. **LSST is unverified against real traffic.** `src/readers/avro_lsst.py`
   is built against the standard Rubin/LSST DPDD schema and is defensive
   (fails loudly by name on any field mismatch rather than silently
   mis-mapping), but no real LSST alert has ever been decoded by it. Two
   separate observation windows this project attempted (Day 13 and Day 14,
   ~90 minutes apart, both inside a plausible Chilean nighttime-observing
   UTC range) found zero currently-queued messages on every subscribed
   topic, even though cumulative topic offsets (hundreds of thousands)
   confirm the survey has genuinely been producing data. The honest state:
   this project caught two lulls, for a reason not further diagnosed —
   whether ugrizy ingests cleanly and whether wavelength-conditioning
   actually transfers weights across surveys (the whole point of that
   design decision, §3) remains an open, unanswered question.

4. **Class-distribution skew — resolved as to cause, not as to residual.**
   Day 13's live ZTF predictions were skewed against every real-rate
   anchor (II 52% / Ib-c 35% / Ia 13%, vs. a real ~55-65/10-20/25-30).
   Day 14 confirmed the cause: dividing out the ELAsTiCC training prior
   (Ia 20.4/Ib-c 28.5/II 51.1%) and re-normalizing moves Ib/c from 34.6%
   to 13.9% — inside the target band, robust across the whole plausible
   range of real-rate estimates. This is expected, documented behavior
   (§11: model outputs carry the training prior, not the sky prior), not
   a bug — but a real residual remains even after correction (Ia still
   under, II still slightly over), most likely because the model's own
   calibration under the training prior was never independently checked.
   Prior correction was applied post-hoc for this analysis only; it is
   not wired into the serving path, and probably should be before these
   probabilities are used for anything beyond monitoring.

5. **Smaller, still-real open items, for completeness:**
   - `fink-client` installed (11.0) does not match the pinned lock file
     (12.1.0) — unexplained, did not block this session, not reconciled.
   - The SFD dustmaps mirror used to fetch two FITS files (Day 8) has
     checksums on record but the actual mirror URL was never captured —
     a reproducibility gap, not a correctness one.
   - `real_bogus_score` as a Tier B static input did not help, but the
     experiment that tested it (a second small, freshly-initialized
     branch on an already-small fine-tuning set) cannot cleanly separate
     "this feature is uninformative" from "any added branch overfits at
     this sample size."
   - WHXE (hierarchical loss) was built but is not the shipped loss —
     Day 6 found it worse than a flat head on a dataset with no non-SN-like
     branch to be meaningfully hierarchical about. Day 9 confirmed Tier B
     finally has that branch (549+ confirmed non-SN-like objects); WHXE
     was never revisited against it. Still a provisional decision, not a
     final one.
   - The image encoder was explicitly cut (Day 12), not deferred — Tier B
     image data (`images_v10`) was never loaded across three separate
     sessions, and there was no remaining day to build, fuse, and retrain
     around it. The UI shows Fink's and BTSbot's own image cutouts
     directly; no model ever sees them.

## Where things live

Full derivations, every number's provenance, and the day-by-day reasoning
behind every decision above: `CLAUDE.md` §11 ("Expected results — sanity
anchors"). Report JSONs and figures the UI reads directly:
`reports/gate_c/gate_c_report.json`, `outputs/anomaly/anomaly_results.json`,
`reports/span_ablation/span_ablation_report.json`, and the confusion-matrix/
phase-curve PNGs under `outputs/*/eval/`.
