# Live Fink runs with the corrected, deployed LC+static model — 2026-11-06

Both runs used the unmodified serving pipeline (`src.serve.live_fink` →
Kafka → `src.serve.inference_worker` → Postgres), Fink username `aman`.

| | ZTF (`fink_sn_candidates_ztf`) | LSST (`fink_most_likely_sn_lsst`) |
|---|---|---|
| window (UTC) | 08:11–08:17 | 08:19–08:25 |
| alerts received / produced / skipped | 578 / 578 / 0 | 1817 / 1817 / 0 |
| objects | 330 | 853 |
| negative-flux detections | 5.7 % | 19.1 % |
| cutouts stored | 571 | 0 (by design, see below) |
| raw class distribution (latest per object) Ia / Ib-c / II | 28.2 / 3.9 / 67.9 % | 27.0 / 3.4 / 69.6 % |
| ELAsTiCC-prior corrected, midpoint (Ib/c range) | 32.7 / 0.3 / 67.0 % (0.3–0.6) | 37.3 / 0.4 / 62.4 % (0.1–1.6) |
| Tier-B-effective-prior corrected, midpoint | 2.7 / 0.3 / 97.0 % | 5.3 / 0.7 / 94.0 % |

**LSST alerts are unlabelled: no accuracy claim is possible.** The result is
that LSST photometry now flows end to end on the corrected flux scale.

Prior correction (`scripts/live_classdist.py`) is reported two ways: with the
ELAsTiCC prior (comparable to Day 14 and the 2026-11-05 audit) and with the
prior the deployed model was actually trained under (Tier B counts × the
sqrt-dampened class weights, ≈ 56.8 / 14.6 / 28.6 %). Neither gives a
plausible sky distribution, so a single multiplicative correction does not
fit this model's outputs; they need calibration.

`scripts/lsst_archive_reader_check.py` runs the fixed reader over the 2,033
packets archived on 2026-11-05: all convert, triggering-source AB magnitude
preserved to 4e-15 mag. Live LSST cutouts are raw 4-HDU FITS, 30–102 px
(6/2033 are 63 px); the frozen schema's (3, 63, 63) cutout cannot hold them.
