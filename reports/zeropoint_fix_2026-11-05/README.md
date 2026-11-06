# Zeropoint fix and Tier B re-run — 2026-11-05

This supersedes every Tier B number in the NSSP-2026 paper, RESULTS.md and earlier CLAUDE.md entries.
Tier A numbers are unaffected: ELAsTiCC needs no zeropoint conversion, and the Tier A baselines are identical before and after the fix.

## The fix

`src/readers/zeropoint.py` used the reciprocal of the magnitude-preserving factor.
- It now computes `flux_27.5 = flux_zp * 10**((27.5 - zp)/2.5)`.
- The tests now assert AB-magnitude invariance directly; 31 tests failed before the fix and pass after it. The full suite is 445 passed, 0 skipped.
- The golden fixtures were regenerated from `10**(-0.4*(magpsf - 27.5))`, which needs no zeropoint. Every old value was first confirmed to equal the inverted formula.
- Code is on branch `fix/zeropoint-direction` (commits 7955ad2 and e64c07d, in a scratch clone, not pushed). The same files have been copied into this working tree.

## Data

`data/interim/btsbot_{train,val,test}.parquet` were regenerated; the old files are kept beside them as `*.pre_zpfix`.
- Fluxes reproduce `magpsf` to 3.6e-15 mag.
- Object counts are unchanged: 15,804 / 1,713 / 1,994.
- Static fields and `flux/flux_err` are unchanged.
- Correct flux is a median 10.3× the old stored value (p5–p95: 7.9–17.8×).

## Results (Tier B test, n = 658, full cutoff)

`seed_sweep.json` and `seed_sweep_summary.json` hold 5 seeds per condition. Both conditions use identical code and the same object order, so the only difference is the flux.

| Configuration | Macro-F1, old flux | Macro-F1, corrected | Ia ROC-AUC, corrected | Ib/c F1, corrected |
|---|---|---|---|---|
| Zero-shot, LC+static weights | 0.4483 | 0.4240 | 0.8599 | 0.162 |
| Zero-shot, LC-only weights | 0.4377 | 0.2737 | 0.7951 | 0.122 |
| **LC+static fine-tune** | 0.6309 ± 0.0033 | **0.6210 ± 0.0067** | 0.8823 | 0.286 |
| LC-only fine-tune (currently deployed) | 0.6754 ± 0.0012 | 0.5647 ± 0.0012 | 0.8685 | 0.137 |
| LC+static+DRB fine-tune | 0.6064 ± 0.0340 | 0.5358 ± 0.0059 | 0.8798 | 0.000 |
| Majority-class baseline | 0.2839 | 0.2839 | 0.5000 | 0.000 |
| Logistic regression (5 features) | 0.3810 | 0.4687 | 0.7550 | 0.200 |

The "old flux" column comes from the control runs, which give old fluxes the new object order. The original paper values (0.6716 / 0.6343 / 0.5812) differ from it by the environment and order effects documented in `../audit_2026-11-05/AUDIT.md`. Those effects are at most 0.004 for LC-only.

**Derived figures:**
- Sim-to-real gap, matched configurations:
  - LC+static 0.7169 → 0.6210 (−0.096)
  - LC-only 0.6981 → 0.5647 (−0.133)
- Transfer gain:
  - LC+static +0.197
  - LC-only +0.291
- Tier B anomaly ROC-AUC (Tier-A-fitted scorers, unchanged):
  - reconstruction error 0.8142, Mahalanobis 0.4407, isoforest 0.5371
  - combined (3 components) **0.7793**, up from 0.7616
- Replay through serving (corrected parquet, re-exported ONNX): 0.5678, against 0.5635 offline.

## What changes in the paper's story

1. **The static-feature inversion was an artifact of the bug.** With correct fluxes, LC+static beats LC-only on real data by +0.056. The "net-negative host features" open finding should be withdrawn.
2. **The deployed choice (LC-only) no longer holds.** Its Ib/c F1 is 0.137, below the 0.40–0.60 anchor. **Serving was switched to the corrected LC+static model the same day** (`outputs/finetune_bts_zpfix/with_static_{checkpoint.pt,model.onnx}`, seed 42):
   - end-to-end replay macro-F1 0.6213 (offline 0.6201);
   - Ib/c F1 0.286;
   - p50 / p95 / p99 latency 37.6 / 54.1 / 65.6 ms, 575 alerts/s;
   - live ZTF: 29 of 29 alerts processed.

   Raw numbers are in `serving_lc_static/`.
3. **The real sim-to-real gap is larger than reported.** The median detection FLUXCAL is about 62 on Tier A and 2,578 on correct Tier B: real BTS objects are about 41× brighter than the simulated population. The bug hid roughly 10× of that. The frozen-trunk, low-learning-rate fine-tune schedule was never exposed to that range.
4. Ia ROC-AUC improves for zero-shot and LC+static even where macro-F1 falls. Ranking improves while the argmax decision boundary shifts.

## Not done

- Gate D's top-50 was not re-derived.
- Live ZTF was not re-run with the fix.
- Serving was not redeployed.
- `avro_lsst.py` still reads `decl`; the live LSST schema uses `dec`.
- Tier A anomaly numbers need no re-run.
