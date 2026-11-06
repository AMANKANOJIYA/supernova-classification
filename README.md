# Transient Supernova Classifier

A multi-survey pipeline that classifies astronomical transient alerts as
**SN Ia**, **SN Ib/c** or **SN II** from their light curves. It also flags
objects that fit none of those classes.

- **Pretrain** on simulated LSST light curves (ELAsTiCC2).
- **Fine-tune** on real ZTF light curves with spectroscopic labels (BTSbot / BTS).
- **Serve** predictions over a Kafka alert stream, either replaying archived
  data or consuming the live Fink broker.
- **Rank anomalies**, meaning alerts that don't look like an ordinary supernova.

The project was a 14-day solo build for the NSSP 2026 summer school. It
favours results that are correct and verified over feature count.

---

## Documentation map

| File | What it is for |
| --- | --- |
| [`README.md`](README.md) | This page: overview, quick start, resources |
| [`GUIDE.md`](GUIDE.md) | Operational guide for newcomers (no astronomy background needed): domain, pipeline, every command, how to read results |
| [`RESULTS.md`](RESULTS.md) | Short summary of what was built and measured |
| [`CLAUDE.md`](CLAUDE.md) | Project rules, design decisions and the full day-by-day log of every measurement. **Source of truth** when numbers disagree |
| [`reports/zeropoint_fix_2026-11-05/`](reports/zeropoint_fix_2026-11-05/README.md) | The flux-zeropoint fix and the corrected real-data results |
| [`reports/audit_2026-11-05/AUDIT.md`](reports/audit_2026-11-05/AUDIT.md) | Audit that found the zeropoint bug |
| [`paper/`](paper/) | LaTeX report (`main.tex`), figures and presentation |

> **Heads-up: some docs predate a bug fix.** On 2026-11-05 the flux-zeropoint
> conversion for real ZTF data was found to be inverted, and it has been fixed.
> Every real-data (Tier B) number in `RESULTS.md`, `GUIDE.md` and the paper
> was measured on mis-scaled input and is superseded. The corrected numbers are
> [below](#results) and in `reports/zeropoint_fix_2026-11-05/`. Simulated-data
> (Tier A) numbers are unaffected.

---

## How it works

```
 ELAsTiCC2 (sim LSST)   BTSbot (real ZTF)   Fink live stream (ZTF/LSST)
        │                     │                       │
        └──── src/readers/ ───┴───────────────────────┘
                     │   every source → CanonicalAlert (src/schema.py)
                     ▼
        src/features/tensorize.py  → padded tensors (flux, not magnitude;
                     │               wavelength, not band index)
                     ▼
   causal GRU light-curve encoder  +  static/host-galaxy branch
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   3-class head          anomaly branch (reconstruction error +
   Ia / Ib-c / II        Mahalanobis + isolation forest)
          │                     │
          └──── serving: Kafka → Redis → ONNX → Postgres → FastAPI → dashboard
```

Key design decisions. The reasoning for each is in `CLAUDE.md` §2–§3.

- **One canonical schema.** Every survey converts into `CanonicalAlert`
  (`src/schema.py`). Flux is on a single scale, SNANA FLUXCAL with AB
  zeropoint 27.5.
- **Flux, never magnitude.** Difference imaging produces negative fluxes.
  They carry real early-time information, and magnitudes cannot represent them.
- **Filter wavelength, not band index.** ZTF *r* and LSST *r* land near each
  other in input space, which is what makes pretrained weights transfer
  across surveys.
- **Causal models only.** Every prediction uses only data available at that
  moment. A test enforces this.
- **Published ELAsTiCC2 taxonomy** (Ia = 2222, Ib/c = 2223, II = 2224), so
  results are comparable to other brokers.

### Data tiers

| Tier | Source | Bands | Labels | Role |
| --- | --- | --- | --- | --- |
| A | ELAsTiCC2 training sample (simulated LSST) | ugrizy | simulated | pretraining |
| B | BTSbot v10 (real ZTF) | ztf_g, ztf_r | spectroscopic | fine-tuning |
| C | Fink livestream (ZTF + LSST) | varies | none | inference only |

---

## Results

All numbers are 3-class **macro-F1** at full light-curve cutoff unless stated
otherwise. Plain accuracy is not reported because always predicting Ia scores
well and is useless. Real-data (Tier B) numbers are **post-fix**: the mean
± SD over 5 seeds, on 658 test objects.

| Measurement | Value |
| --- | --- |
| Tier A (sim), light curve + host features | **0.7169**, Ia-vs-rest ROC-AUC **0.9301** |
| Tier A (sim), light curve only | 0.6981 |
| Tier B (real ZTF), light curve + host, fine-tuned — **deployed** | **0.6210 ± 0.0067**, Ia-vs-rest ROC-AUC 0.882 |
| Tier B (real ZTF), light curve only, fine-tuned | 0.5647 ± 0.0012 |
| Tier B zero-shot (Tier A weights, no fine-tuning) | 0.4240 |
| Logistic-regression baseline / majority class | 0.4687 / 0.2839 |
| Sim-to-real gap (light curve + host) | 0.7169 → 0.6210 (−0.096) |
| Ib/c F1, deployed model | 0.286 (below the 0.40–0.60 literature anchor) |
| Anomaly detection, combined ROC-AUC | Tier A 0.8146 / Tier B 0.7793 |
| Replay through the full serving stack vs offline | 0.6213 vs 0.6201 |
| Serving latency / throughput (single node) | p50 37.6 ms, p99 65.6 ms, 575 alerts/s |

A drop from simulated to real data is the expected result. Real BTS objects
are about 41× brighter than the simulated training population. Details and
open questions are in `CLAUDE.md` §11.

---

## Quick start

### Prerequisites

- Python **3.12** (pinned to `>=3.12,<3.13`)
- Docker, needed only for the serving stack
- Node.js 20+ and npm, needed only for the dashboard
- About 30 GB of free disk space if you download all the raw data

### Install and test

```bash
make setup     # creates .venv and runs pip install -e ".[dev]"
make test      # pytest
```

Use `make setup`, not `requirements-lock.txt`. The lockfile is known to be
stale (see `GUIDE.md` §4).

Most tests run on checked-in fixtures. A few need data on disk:

- **SFD dust maps.** Tests that build static features need the two dust-map
  FITS files described under [Data](#data).
- **Raw ELAsTiCC2 files.** About 28 tests skip unless these are present.

Once both are in place, expect all tests to pass.

### Data

`data/` is gitignored, so a fresh clone has none of it.

| What | How | Size |
| --- | --- | --- |
| ELAsTiCC2 (Tier A) | `make data-elasticc2`, then `make data-elasticc2-verify` (slow, about 1 h) | 7.3 GB |
| BTSbot `metadata_v10/` (Tier B) | download by hand from [Zenodo 10.5281/zenodo.10839691](https://doi.org/10.5281/zenodo.10839691) | 476 MB |
| BTS spectroscopic labels | fetched and cached automatically by `src/readers/bts_labels.py` | small |
| SFD dust maps | see below | 2 × 64 MB |

The default `dustmaps` source (Harvard Dataverse) returns HTTP 403. Fetch the
files from the mirror instead and check their hashes:

```bash
mkdir -p data/raw/dustmaps/sfd && cd data/raw/dustmaps/sfd
for f in ngp sgp; do
  curl -fLO https://github.com/kbarbary/sfddata/raw/master/SFD_dust_4096_$f.fits
done
shasum -a 256 *.fits
# 50b6aaad0b880762d0fd081177802dcc17c39d7044a410dd5649e2dfd0503e97  SFD_dust_4096_ngp.fits
# 84891a59054adab44a7be54051e4dcf0e66e3f13eee0d845ce3739242f553b83  SFD_dust_4096_sgp.fits
```

BTSbot's `images_v10.zip` (18 GB) is not needed. No code path reads it.

### Train

```bash
python -m src.train.pretrain           # Tier A, light curve only
python -m src.train.pretrain_static    # Tier A, light curve + host features
python -m src.train.finetune_bts       # Tier B fine-tune (all variants, ~40 s)
python -m src.train.train_anomaly      # anomaly branch (expensive)
python -m src.train.persist_anomaly_scorers
```

Training is configured with Hydra (`configs/`). You can override any setting
with `key=value`. Full Tier A runs take 2–4 hours.

**These scripts have no `--help`.** Running one with any argument starts a
real run and overwrites its outputs.

Checkpoints are written to `outputs/`, which is gitignored. Serving loads
`outputs/finetune_bts_zpfix/with_static_model.onnx`.

### Serve

```bash
make infra-up                    # Kafka :9092, Redis :6379, Postgres :5432
make worker                      # consumer: Kafka → Redis → ONNX → Postgres
make replay SPLIT=test           # producer: replay archived Tier B alerts
make api                         # FastAPI on :8010
npm --prefix frontend install && make frontend   # dashboard on :3000
```

For live alerts, replace the replay producer with:

```bash
python -m src.serve.live_fink --survey ztf --username <user> --group-id <group>
```

The consumer code is identical in replay and live mode.

---

## Repository layout

```
src/
├── schema.py        # CanonicalAlert contract (frozen)
├── taxonomy.py      # ELAsTiCC2 class hierarchy
├── readers/         # elasticc2, btsbot, bts_labels, avro_ztf, avro_lsst, zeropoint
├── features/        # tensorize, extinction, wavelength
├── models/          # GRU encoder, static encoder, fusion, heads, WHXE loss
├── anomaly/         # decoder, scoring (Mahalanobis, PineForest)
├── train/           # pretrain*, finetune_bts, train_anomaly, ablations
├── eval/            # metrics, phase curves, baselines (frozen)
└── serve/           # replay, live_fink, ingest, state, inference_worker, api
configs/             # Hydra configs
infra/               # docker-compose.yml
frontend/            # Next.js dashboard
ui/                  # older single-page viewer (served at :8010/)
tests/               # pytest suite + golden fixtures
reports/             # gate reports, audits, figures
paper/               # LaTeX report and presentation
```

---

## Known limitations

- **Ib/c is the weak class.** After the fix its F1 sits below the literature
  band.
- **LSST live ingestion is unverified.** No live LSST traffic was captured.
  `avro_lsst.py` also needs to read `dec` instead of `decl` to match the live
  schema.
- **Raw outputs carry the training prior.** Predicted probabilities reflect
  ELAsTiCC's class mix, not real sky rates. Correcting for that is a
  deployment-time step.
- **Gate D is deferred.** Artifact-rate inspection of top anomalies needs an
  unfiltered alert stream, which hasn't been ingested yet.
- **The image encoder was cut from scope.** Cutouts are displayed in the UI
  but not used by the model.

The full list is in `RESULTS.md` and `CLAUDE.md` §11.

---

## Resources

**Data**
- ELAsTiCC2 training sample: <https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/>
- BTSbot v10 dataset (Zenodo): <https://doi.org/10.5281/zenodo.10839691>
- ZTF Bright Transient Survey Sample Explorer: <https://sites.astro.caltech.edu/ztf/bts/explorer.php>
- SFD dust-map mirror: <https://github.com/kbarbary/sfddata>

**Brokers and streams**
- Fink broker docs (ZTF): <https://doc.ztf.fink-broker.org/>
- Fink broker docs (LSST livestream): <https://doc.lsst.fink-broker.org/services/livestream/>
- `fink-client`: <https://github.com/astrolabsoftware/fink-client>

**Papers and related work**
- Villar et al. 2023, weighted hierarchical cross-entropy (WHXE): [arXiv:2312.02266](https://arxiv.org/abs/2312.02266)
- BTSbot (Rehemtulla et al. 2024): [arXiv:2401.15167](https://arxiv.org/abs/2401.15167)
- Baselines worth reading: [SuperNNova](https://github.com/supernnova/SuperNNova),
  [snmachine](https://github.com/LSSTDESC/snmachine),
  [ALeRCE lc_classifier](https://github.com/alercebroker/lc_classifier)
- ELAsTiCC challenge code: <https://github.com/LSSTDESC/elasticc>

**Tooling**
- [`coniferest`](https://github.com/snad-space/coniferest) (isolation forest / PineForest / AAD)
- [`light-curve`](https://github.com/light-curve/light-curve-python) feature extraction
- [`dustmaps`](https://github.com/gregreen/dustmaps), [`sncosmo`](https://sncosmo.readthedocs.io/)
