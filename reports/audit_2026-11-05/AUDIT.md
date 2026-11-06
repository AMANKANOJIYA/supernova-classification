# State audit and re-measurement — 2026-11-05

No model changes and no retraining, except the three fine-tune runs asked for in Q4.
Those ran in an isolated clone, so nothing under `outputs/` was overwritten.
Every artifact this audit produced is a new file under this folder.

## 0. Read this first: three findings that affect the paper

1. **The flux zeropoint conversion is inverted (`src/readers/zeropoint.py`).**
   `convert_zeropoint` multiplies by `10**((from_zp − 27.5)/2.5)`; the correct factor is `10**((27.5 − from_zp)/2.5)`.
   Checked numerically against magnitude invariance:
   - An LSST source at 21.64 mag comes out at 13.84 mag (1318× too bright).
   - A ZTF point at mag 19.0 (zp 26) comes out at 22.0 (15.8× too faint).

   Every Tier B row goes through it (`btsbot.py`), and so does every live ZTF/LSST row.
   Across all 407,711 BTSbot rows, stored Tier B flux is a median **10.3× too faint**, and the error varies by row (p5–p95: 0.056–0.127× of the correct value) because `magzpsci` varies.
   Tier A (ELAsTiCC, no conversion) is correct.

   Consequences:
   - Tier B fine-tune, test and serving all use the same wrong scale, so Tier B numbers are internally consistent.
   - Every **Tier A → Tier B** comparison feeds the model input on a different scale from training: zero-shot, sim-to-real gap and transfer gain. The affected features are `flux_abs_norm` and `flux_err_norm`; `flux_norm` is scale-invariant.
   - The bug comes from CLAUDE.md §3's wording ("multiplicative factor ≈ 36.3"). `tests/test_zeropoint_conversion.py` asserts that same factor, and no test checks magnitude invariance. That is why all 414 tests pass.
   - The Day-14 claim that the conversion was "independently verified" is circular.
2. **Two cells in the paper are wrong** (Q1, Q3 below).
3. **The 0.0011 "non-determinism" is not run-to-run noise** (Q4).

## Part A — reproducibility

All paths are relative to the repo root. "exists" means present and readable on disk today.

### Tier A (test n = 126,542), `outputs/<run>/eval/metrics.json` — all exist, original (Sep 5–6)

| Config | macro-F1 full | Ia ROC-AUC full | F1 Ia / Ib-c / II | path |
|---|---|---|---|---|
| LC-only (step 6) | 0.6981 | **0.9185** | 0.6726 / 0.6315 / 0.7900 | outputs/train/eval/metrics.json |
| LC+static (step 8) | 0.7169 | **0.9301** | 0.7056 / 0.6524 / 0.7928 | outputs/train_static/eval/metrics.json |
| LC+static, no photo-z | 0.7108 | 0.9255 | 0.6913 / 0.6480 / 0.7931 | outputs/train_static_no_photoz/eval/metrics.json |
| WHXE | 0.6823 | 0.9183 | 0.6542 / 0.6254 / 0.7673 | outputs/train_whxe/eval/metrics.json |

Phase curves (macro-F1 at −5 / 0 / 5 / 15 / 30 / full), from the same files. Raw and normalised confusion matrices are in the same files at every cutoff.

| Config | −5 | 0 | 5 | 15 | 30 | full |
|---|---|---|---|---|---|---|
| LC-only | .4304 | .5185 | .5912 | .6483 | .6831 | .6981 |
| LC+static | .4769 | .5467 | .6139 | .6681 | .7021 | .7169 |
| no photo-z | .4644 | .5394 | .6076 | .6616 | .6956 | .7108 |
| WHXE | .4409 | .5042 | .5739 | .6296 | .6661 | .6823 |

### Tier B (test n = 658; 488 Ia / 35 Ib-c / 135 II)

Original numbers come from `reports/gate_c/gate_c_report.json` (git-tracked, exists).
The files under `outputs/finetune_bts/*/eval/` are **not** the originals: they were overwritten on Sep 7 11:43.

| Config | macro-F1 full | Ia ROC-AUC | F1 Ia / Ib-c / II | Original still reproducible? |
|---|---|---|---|---|
| Zero-shot (**LC+static** Tier A weights) | 0.4483 | 0.7802 | 0.6057 / 0.3294 / 0.4098 | yes, deterministic (inference only), identical today |
| Zero-shot, **LC-only** Tier A weights (new) | 0.4377 | 0.7764 | 0.5321 / 0.3826 / 0.3985 | new: inference_only_evals/zero_shot_lc_only |
| LC-only FT (**deployed**) | 0.6716 | 0.8822 | 0.8843 / 0.5000 / 0.6304 | **yes**: the serving ONNX holds the original weights and reproduces 0.6715774 / 0.8821962 exactly (inference_only_evals/deployed_onnx_lc_only) |
| LC+static FT | 0.6343 | 0.8641 | 0.8755 / 0.4110 / 0.6165 | **no**: weights overwritten; today's code gives 0.6354 |
| LC+static+DRB FT | 0.5812 | 0.8704 | 0.9024 / 0.1500 / 0.6912 | **no**: weights overwritten; today's code gives 0.5841 |

Tier B phase curves (macro-F1 at 0 / 5 / 15 / 30 / full; −5 is undefined on Tier B):

| Config | 0 | 5 | 15 | 30 | full | Source |
|---|---|---|---|---|---|---|
| Zero-shot (LC+static) | .1665 | .2714 | .3863 | .4045 | .4483 | outputs/finetune_bts/zero_shot/eval/metrics.json (deterministic, valid) |
| LC-only deployed | .2839 | .3182 | .4519 | .6018 | .6716 | **regenerated** from the deployed ONNX, with confusions |
| LC+static FT (original) | .2839 | .2947 | .4631 | .5510 | .6343 | gate_c_report.json; **original confusion matrices lost** |
| LC+static+DRB FT (original) | — | — | — | — | .5812 | full-cutoff value only; **phase curve and confusions lost** |

### Anomaly ROC-AUC

Source: `outputs/anomaly/anomaly_results.json` (exists). The 3-component combinations were recomputed today from `scores_*.json` and match exactly; see `anomaly_recomputed_from_scores.json`.

| | recon | entropy | Mahalanobis | isoforest | combined (4) | combined (3, entropy dropped) |
|---|---|---|---|---|---|---|
| Tier A (126,542 vs 442,303) | 0.5643 | 0.2578 | 0.7860 | 0.8428 | 0.7644 | **0.8146** |
| Tier B (658 vs 1,045) | 0.8382 | 0.4203 | 0.5236 | 0.6400 (Day-10 forest) / 0.6479 (persisted forest) | 0.7418 | **0.7622** (Day-10) / 0.7616 (persisted) |

The persisted forest's refit is **byte-identical** across runs (reproduced today). CLAUDE.md attributes the 0.0079 isoforest gap to "PineForest randomness", which is wrong: the Day-10 forest and the persisted Day-13 forest are simply different fits.

### The four questions

**Q1. Table 4.1's 0.8822 is a transcription (macro) error.**
`paper/sections/results.tex:30` uses `\LCOnlyRocAucIa` (the Tier B macro) in the Tier A LC-only row.
- Tier A LC-only is **0.9185**: `\TierLCOnlyRocAucIaFull` already exists in `numbers.tex:187`.
- Tier B LC-only is 0.8822 (0.882196), confirmed independently by re-running the deployed ONNX.
- Tier A LC-only at the 15-day cutoff happens to be 0.8821, which may be how the coincidence went unnoticed.

**Q2. The zero-shot baseline used the LC+static checkpoint** (`finetune_bts.py:575-592`).
- +0.1860 (0.4483 → 0.6343) is a correct, like-for-like LC+static gain.
- +0.2233 (0.4483 → 0.6716) mixes two configurations and should not be used.
- The like-for-like gain for the deployed LC-only model, measured today with the Tier A LC-only checkpoint (inference only), is **0.4377 → 0.6716 = +0.2338**.
- Both gains are subject to the zeropoint finding in §0.

**Q3.** The correct Ia ROC-AUC for Tier A LC+static is **0.9301** (`outputs/train_static/eval/metrics.json` → `full.roc_auc[0]`); the abstract's `\LCOnlyRocAucIa` must be `\TierLCStaticRocAucIaFull`.
The same mixing affects the "deployed" sim-to-real gap. 0.7169 → 0.6716 compares LC+static on Tier A with LC-only on Tier B; matched LC-only is **0.6981 → 0.6716 = −0.0265**.
There is also a minor rounding mismatch: the gap is 0.0454 in `numbers.tex` and 0.0453 in RESULTS.md (exact value 0.04537).

**Q4. Determinism.** Three runs of `finetune_bts.py`, identical config and seed, same data (isolated clone):
- **The spread is zero.** All four variants matched to the last digit, and the LC-only checkpoint md5 was identical in all three runs.
- LC-only 0.6727254, LC+static 0.6354493, DRB 0.5840691, zero-shot 0.4483002.
- Running the original Day-9 commit (d58bdd9) against the same inputs gives the same numbers, so the code changes since Day 9 are not the cause.
- All inputs (Tier A checkpoints, BTSbot parquets, BTS label CSV) have mtimes **before** the original Sep 6 12:34 run.
- The difference is therefore a one-time environment change between Sep 6 12:34 and Sep 7 11:42, not run-to-run noise.
- Leading explanation: `requirements-lock.txt`, written 13:19 on Sep 6, about 45 minutes after the original run, records torch 2.14.0 / numpy 2.5.2 / polars 1.44.1. The Sep 7 environment (and today's) has torch 2.5.1 / numpy 2.1.3 / polars 1.12.0.
- **Not verifiable**: that venv was replaced on Sep 22 and no longer exists.
- Separately, the Day-12 replay macro-F1 (0.6727) reproduces exactly from the rows still in Postgres. It also reproduced in today's benchmark (0.6727254).

## Part B — regenerated and persisted

**Baselines** (`baselines/`): the frozen `src/eval/baselines.py` and `report.py`, the model's own splits, and the logistic regression fit on full-light-curve train features. The protocol is in the report JSON.

| | Tier A macro-F1 | Tier A Ia ROC | Tier B macro-F1 | Tier B Ia ROC |
|---|---|---|---|---|
| Majority class | 0.2253 (always II) | 0.5000 | 0.2839 (always Ia) | 0.5000 |
| Logistic regression (5 features) | 0.3093 | 0.5401 | 0.3810 | 0.7513 |

Per-class F1 and confusion matrices are in `baselines/baseline_report.json`, with phase cutoffs in `baselines/tier_*/*/eval/metrics.json`. The Tier B LR uses mis-scaled fluxes (§0), but its features are scale-dependent only through `peak_flux`.

**Serving benchmark** (`serving_benchmark/`): Tier B test replay, 49,587 alerts, 0 dropped, on 2026-11-05.
- Latency (Kafka receipt → just before the Postgres INSERT): **p50 38.3 ms, p95 55.7 ms, p99 68.1 ms**, max 144 ms.
- Throughput: **559 alerts/s** over the busy interval, 411 alerts/s over wall time.
- ONNX classifier alone: **0.18 ms/alert** at batch size 1, 0.24 ms/alert at batch 32. ONNX plus the anomaly scorer: 0.61 ms/alert median.
- The Day-12 prose figures (27.2 / 73.7 ms, 730/s) don't match any of the 14 runs still in Postgres (`postgres_history/db_existing_runs.json`). They should be treated as unverifiable.

**Day-14 prior correction, re-measured** (`postgres_history/live_prior_correction_from_db.json`): it reproduces exactly from the original 338-object live rows still in Postgres.
- Raw: Ia 13.3 / Ib-c 34.6 / II 52.1 %. Corrected at the midpoint sky prior: **17.5 / 13.9 / 68.6 %**.
- Robustness range for Ib/c: **6.8–23.1 %**. CLAUDE.md says 7.1–23.1; the low end differs.

## Part C — live, 2026-11-05 ~03:30–03:45 UTC

**ZTF** (`fink_sn_candidates_ztf`, 3 min): **runs unmodified.**
- 466 received, 466 produced, 0 skipped; 459 cutouts stored.
- Negative-flux fraction **5.35%** (356 of 6,656 detections).
- 297 objects. Raw distribution Ia 9.4 / Ib-c 25.6 / II 65.0 %; corrected (midpoint) 13.8 / 10.4 / 75.8 %; Ib/c range 4.4–15.8 %.
- The Ia under-prediction left after correction is larger than on Day 14.

**LSST: does not run unmodified.**
- Traffic is flowing: the broker answered, and `fink_most_likely_sn_lsst` held 74 retained alerts and `fink_remove_unlikely_transients_lsst` 696.
- Some topic names are new since the last attempt (`fink_extragalactic_lt20mag_candidate_lsst`, `fink_sn_near_galaxy_candidate_lsst`); the full list is in `live/fink_lsst_topics_watermarks.json`.
- **802 alerts received, 0 produced, 802 skipped**, every one with `diaSource missing ra/decl`.
- The live schema uses `dec`, and `avro_lsst.py:234` reads `decl`. Everything else the reader requires (`midpointMjdTai`, `band`, `psfFlux`, `psfFluxErr`) is present.
- Even after a one-key fix, the zeropoint bug would store a 21.9-mag source as about 14 mag.
- The ZTF broker's topic listing returned nothing within 150 s, but consuming worked.
- **Undocumented earlier run:** Postgres holds 805 LSST predictions (740 objects) from **2026-10-31 03:25 UTC**, plus a 194-object live ZTF run and three replays from Oct 30–31. No doc or report mentions them.
  - On Oct 31 the reader evidently decoded packets: the schema then still had `decl`, or the code differed.
  - Those LSST predictions (Ia 39.5 / Ib-c 0 / II 60.5 %, near-constant probabilities) ran on fluxes inflated about 1318× and should not be reported.
- **Continuous listener:** started a raw-packet archiver, not the pipeline, since the pipeline rejects every packet today.
  - Process: PID in `~/.local/share/sn-classifier/lsst_listener/pid`.
  - Output: `data/raw/fink_lsst_archive/`, hard cap 2 GB. It saved 604 alerts in its first minute.
  - Stop it with `kill $(cat ~/.local/share/sn-classifier/lsst_listener/pid)`.

## Part D — current state

- **iCloud.** The repo is on an iCloud-synced Desktop and 429 files are evicted ("dataless"); they can't be downloaded ("Operation canceled"), so `git log` hangs. Affected:
  - 265 git objects;
  - all 11 Hydra configs under `configs/`;
  - `reports/live_fink/{ztf_run2_fixed,lsst_run1}.json`;
  - `outputs/anomaly/engineered_normalizer.npz`;
  - `outputs/train_whxe/eval/node_accuracy.json`;
  - frontend `package.json`, `next.config.ts`, `tsconfig.json` and 77 `.next` files.

  Every tracked file can be recovered from GitHub (remote `main` = local HEAD 3117138). This audit ran from a clean clone for that reason. `engineered_normalizer.npz` was regenerated in the clone: the refit is deterministic (forest and Mahalanobis are byte-identical to the originals). It was **not** written back to the repo.
- **Serving in place is broken.** `AnomalyScorer()` hangs on the evicted `.npz` (killed after 120 s), so the worker and API can't start from the repo. The frontend can't start because `package.json` is evicted. From the clean clone the whole serving path works (benchmark above). The Next.js dashboard was not tested from the clone.
- **Tests:** 414 passed, 0 skipped, 0 failed (76.8 s, clean clone). The suite does not catch the zeropoint bug (§0).
- **Lost to overwrites (Sep 7 11:43):**
  - `outputs/finetune_bts/{lc_only,with_static,with_static_drb}_checkpoint.pt` and their `eval/` directories (both now hold re-run versions);
  - the original LC+static and DRB weights, confusion matrices and DRB phase curve.

  The original LC-only weights survive in `lc_only_model.onnx`.
- **Stale:**
  - CLAUDE.md §3's zeropoint wording, and the Day-13 "PineForest randomness" note.
  - `requirements-lock.txt`.
  - `avro_lsst.py` (`decl`) and `zeropoint.py`.
  - GUIDE §7's "could not re-verify" notes: the replay figure now is verified.
  - The docs target fink-client 12; the environment has 11.0. It still works.
- **Disk:** the volume is 98% full (12 GiB free before the listener). `data/raw` is 26 GB, of which `images_v10.zip` (19.6 GB) is read by nothing.
- **Docker:** `sn-classifier-{kafka,redis,postgres}` were started by this audit and are still running. The Postgres volume holds the full history (backed up to scratch before any writes).
  - The audit added the benchmark replay and live rows (scored_at ≥ 2026-11-05 03:22 UTC).
  - Redis was backed up; the Tier B test objects' keys were cleared and rebuilt by the benchmark replay.
