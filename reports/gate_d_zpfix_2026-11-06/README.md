# Gate D re-derived on zeropoint-corrected Tier B — 2026-11-06

`scripts/rederive_gate_d.py` ranks the pooled Tier B evaluation set (658 normal
SNe + 1,045 confirmed non-SN "weird" objects) by the deployed 3-component
anomaly score, using the persisted scorers exactly as serving loads them (no
refit). It refuses to rank unless the combined ROC-AUC reproduces the fix
pass's 0.7793; it does, exactly, and so do all three components.

| | pre-fix (`reports/gate_d/`) | corrected |
|---|---|---|
| CV / nova / LBV / LRN (`no_tree_node`) | 46 | 42 |
| AGN | 3 | 6 |
| 91bg | 1 | 1 |
| TDE | 0 | 1 |
| from the normal SN test set | 0 | 0 |
| heuristic `artifact_like` / `genuine_anomaly` / `short_sparse` | 36 / 13 / 1 | 38 / 11 / 1 |

35 of 50 objects are in both lists. All 50 light curves were inspected
(`contact_sheet_{1,2}.png`, `light_curves/`): multi-season recurrent outbursts
and stochastic AGN variability; no isolated single-epoch spikes. The
`artifact_like` tag is the heuristic's known false label, as on Day 10.

Verdict unchanged: **deferred**. Every object is spectroscopically confirmed
real because the evaluation population is pre-filtered; the gate's question
about real-bogus filtering needs an unfiltered stream.
