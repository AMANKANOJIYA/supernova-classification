"""Gate B (CLAUDE.md §7): the three mandatory tests gating the first trained
model (step 6). None may be skipped or weakened.

Tests 1 and 2 are self-contained (real small subset for #1, hand-built
fixtures for #2) and always run. Test 3 (ceiling check) needs the real,
fully-trained production checkpoint from `python -m src.train.pretrain`
(`outputs/train/checkpoint.pt` + `outputs/train/eval/metrics.json`) --
skipped if that has not been produced yet, mirroring the existing golden-test
skip pattern in `tests/test_readers_elasticc2.py`.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.eval.metrics import macro_f1, roc_auc
from src.eval.phase_curves import select_at_cutoff
from src.features import tensorize
from src.models.fusion import LCStaticFusionModel
from src.models.lc_encoder import LCEncoder
from src.models.static_encoder import StaticFeatureNormalizer
from src.schema import CanonicalAlert, Observation, StaticContext, Survey
from src.train.dataset import (
    LightCurveDataset,
    collate_examples,
    load_alerts,
    load_object_class_ids,
    split_by_object_id,
)
from src.train.dataset_static import LightCurveStaticDataset
from src.train.pretrain import (
    collect_phase_predictions,
    compute_class_weights,
    masked_cross_entropy,
)
from src.train.pretrain_static import collect_phase_predictions_static

PARQUET = Path("data/interim/elasticc2/train_02_full_3class.parquet")
# Day 7: Gate B's ceiling check now targets the current (step-8, LC+static)
# checkpoint rather than the step-6 flat baseline -- Gate B gates the most
# advanced model at any point, not a permanently frozen step-6 snapshot.
CHECKPOINT = Path("outputs/train_static/checkpoint.pt")
METRICS_JSON = Path("outputs/train_static/eval/metrics.json")

_pytestmark_parquet = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="cached ELAsTiCC2 parquet not present on disk (data/interim/elasticc2/train_02_full_3class.parquet)",
)


def _static():
    return StaticContext(
        host_photoz=0.1,
        host_photoz_err=0.01,
        host_offset_arcsec=0.5,
        host_ellipticity=0.2,
        host_mag_r=20.0,
        mw_ebv=0.05,
        gal_lat=30.0,
        real_bogus_score=None,
    )


def _alert(observations, object_id="obj", class_id=2222):
    return CanonicalAlert(
        object_id=object_id,
        survey=Survey.LSST,
        observations=observations,
        static=_static(),
        cutouts=None,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# Gate B #1 -- shuffled-label test.
# ---------------------------------------------------------------------------

_SEED = 0
# Day 6: raised from 150 to 1667 (~5,000 objects total, ~0.6% of the corpus,
# vs the original 450 / 0.05%). At 150/class the test failed its own
# threshold (macro-F1 0.453 vs <0.45) on its first-ever completed run;
# diagnostics showed this was sampling variance, not leakage (a 4-seed sweep
# on the same 450 objects scattered 0.18-0.45, only 1 of 4 above 0.44, two
# below the calculated no-signal floor) -- but a test meant to catch leakage
# has no business running at a sample size where its own noise floor is
# comparable to its pass/fail margin. This is a sample-size increase only;
# the <0.45 threshold is unchanged.
_N_PER_CLASS = 1667  # "small" per CLAUDE.md's "retrain briefly", not tiny


def _small_real_alert_subset():
    """Day 6: was `load_alerts(str(PARQUET))` then filtered down to 450
    objects after paying the cost of reconstructing all 843,617 -- the exact
    inefficiency that made this suite impractical to run to completion (see
    CLAUDE.md audit notes). Filtering by class first via the cheap
    id-only read, then reconstructing only the ~450 objects actually needed,
    is the same fix applied to `TestCeilingCheck` below."""
    lightweight = load_object_class_ids(str(PARQUET))
    by_class: dict = {}
    for item in lightweight:
        by_class.setdefault(item.class_id, []).append(item.object_id)
    subset_ids: set = set()
    for class_id, object_ids in by_class.items():
        subset_ids.update(sorted(object_ids)[:_N_PER_CLASS])
    return load_alerts(str(PARQUET), object_ids=frozenset(subset_ids))


def _train_briefly(train_ds, num_epochs=8, hidden_size=32, lr=1e-2, seed=0, class_weighting=False):
    torch.manual_seed(seed)
    model = LCEncoder(
        input_size=tensorize.NUM_FEATURES, hidden_size=hidden_size, num_layers=2, dropout=0.1, num_classes=3
    )
    class_weights = compute_class_weights(train_ds, 3, enabled=class_weighting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_examples)
    for _ in range(num_epochs):
        model.train()
        for batch in loader:
            logits = model(batch.features, batch.lengths)
            loss = masked_cross_entropy(logits, batch.class_index, batch.mask, class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _full_cutoff_macro_f1(model, test_ds):
    predictions = collect_phase_predictions(model, test_ds, torch.device("cpu"), batch_size=64)
    selected, _ = select_at_cutoff(predictions, cutoff=None)
    y_true = np.array([p.true_class for p in selected])
    y_prob = np.stack([p.probs for p in selected])
    y_pred = np.argmax(y_prob, axis=1)
    return macro_f1(y_true, y_pred, num_classes=3)


@_pytestmark_parquet
class TestShuffledLabelTest:
    """CLAUDE.md §7 Gate B #1: "Permute training labels randomly, retrain
    briefly. Macro-F1 must collapse to chance (~0.33). If it does not, there
    is label leakage." """

    def test_macro_f1_collapses_to_chance_with_shuffled_labels(self):
        alerts = _small_real_alert_subset()
        split = split_by_object_id(alerts, seed=_SEED, train_frac=0.7, val_frac=0.15)

        train_ds = LightCurveDataset(alerts, split.train, pre_detection_window_days=30.0)
        test_ds = LightCurveDataset(alerts, split.test, pre_detection_window_days=30.0)

        rng = np.random.RandomState(_SEED)
        permutation = rng.permutation(len(train_ds))
        original = [ex.class_index for ex in train_ds._examples]
        shuffled = [original[i] for i in permutation]
        import dataclasses

        train_ds._examples = [
            dataclasses.replace(ex, class_index=new_idx)
            for ex, new_idx in zip(train_ds._examples, shuffled)
        ]

        model = _train_briefly(train_ds, num_epochs=8)
        score = _full_cutoff_macro_f1(model, test_ds)
        assert score < 0.45, (
            f"shuffled-label macro-F1={score:.3f}, expected near chance (~0.33) "
            "-- possible label leakage"
        )


# ---------------------------------------------------------------------------
# Gate B #2 -- causality test, composed with Day-4 windowing.
# ---------------------------------------------------------------------------


class TestCausalityTest:
    """CLAUDE.md §7 Gate B #2: "Truncate a sequence at timestep t. Assert
    the prediction at t is bit-identical with and without timesteps after t
    present." Composed with Day-4 windowing per this session's instruction:
    the window is recomputed fresh from whatever observations are actually
    passed to `tensorize_alert`, so truncating the raw observation list (not
    slicing an already-tensorized/windowed output) is the correct
    composition -- verified below, including the one genuine subtlety this
    composition introduces (see the second test)."""

    def test_prediction_at_t_is_bit_identical_with_and_without_future_observations(self):
        """Clean case: every observation in the shorter run already sits
        inside its own window (verified by the shape assertion below), so
        appending strictly-later observations cannot retroactively change
        which of the shorter run's own rows survive windowing -- isolating
        the property CLAUDE.md's wording describes."""
        short_observations = [
            Observation(mjd=60000.0, lam_eff=622.2, flux=5.0, flux_err=1.0, is_detection=True),
            Observation(mjd=60003.0, lam_eff=633.9, flux=80.0, flux_err=2.0, is_detection=True),
            Observation(mjd=60010.0, lam_eff=622.2, flux=250.0, flux_err=3.0, is_detection=True),
            Observation(mjd=60020.0, lam_eff=633.9, flux=120.0, flux_err=2.0, is_detection=True),
        ]
        long_observations = short_observations + [
            Observation(mjd=60035.0, lam_eff=622.2, flux=40.0, flux_err=2.0, is_detection=True),
            Observation(mjd=60300.0, lam_eff=633.9, flux=2.0, flux_err=1.0, is_detection=False),
        ]

        alert_short = _alert(short_observations, object_id="short")
        alert_long = _alert(long_observations, object_id="long")

        torch.manual_seed(0)
        model = LCEncoder(
            input_size=tensorize.NUM_FEATURES, hidden_size=16, num_layers=2, dropout=0.0, num_classes=3
        )
        model.eval()

        short_t = tensorize.tensorize_alert(alert_short)
        long_t = tensorize.tensorize_alert(alert_long)

        # Confirms this test isolates the intended property -- see docstring.
        assert short_t.features.shape[0] == len(short_observations)

        with torch.no_grad():
            short_logits = model(
                short_t.features.unsqueeze(0), torch.tensor([short_t.features.shape[0]])
            )
            long_logits = model(long_t.features.unsqueeze(0), torch.tensor([long_t.features.shape[0]]))

        n_short = short_t.features.shape[0]
        assert torch.equal(short_logits[0], long_logits[0, :n_short]), (
            "prediction at t changed when future observations were appended -- causality violated"
        )

    def test_windowing_can_retroactively_include_a_previously_excluded_point(self):
        """Documents a real, non-obvious interaction this session's Day-5
        instruction asked to verify: forced photometry sitting between an
        early vantage point's own last detection and a later detection's
        time can be *excluded* at the early vantage point (the window isn't
        wide enough yet) and *included* once that later detection arrives
        and widens the window rightward. This is NOT a causality violation:
        at the early vantage point the model has no way to know a later
        detection is coming, so correctly leaving that point out is the
        causally sound choice at that point in time -- it is simply
        re-evaluated, not retroactively changed, once genuinely new
        information (the later detection) arrives.
        """
        observations = [
            Observation(mjd=60000.0, lam_eff=622.2, flux=100.0, flux_err=2.0, is_detection=True),
            Observation(mjd=60025.0, lam_eff=633.9, flux=3.0, flux_err=1.0, is_detection=False),
            Observation(mjd=60050.0, lam_eff=622.2, flux=90.0, flux_err=2.0, is_detection=True),
        ]
        early_alert = _alert(observations[:2], object_id="early")
        full_alert = _alert(observations, object_id="full")

        early_t = tensorize.tensorize_alert(early_alert)
        full_t = tensorize.tensorize_alert(full_alert)

        # last_detection_mjd=60000 at the early vantage point -> the forced-phot
        # point at 60025 (10 days past window end) is excluded.
        assert early_t.features.shape[0] == 1
        # last_detection_mjd=60050 once the later detection arrives -> the
        # window widens to include the previously-excluded point.
        assert full_t.features.shape[0] == 3


# ---------------------------------------------------------------------------
# Gate B #3 -- ceiling check.
# ---------------------------------------------------------------------------

# Day 6 correction: CLAUDE.md §11's Ia-vs-rest anchor names ROC-AUC (the
# RAPID/SuperNNova literature figure), not macro-F1 -- the two were
# conflated in the original test, which asserted macro-F1 (0.7905 on the real
# checkpoint) against a range calibrated for a different metric. Ranges are
# stage-dependent rather than widened in place (moving a threshold to fit an
# observed number is exactly what §2.1 exists to prevent): a LC-only
# baseline with no static/host features (this step, and step 7's WHXE head
# swap) is checked against the wider STEP6 range; the tighter literature
# anchor applies once static features (step 8) exist to explain it.
CEILING_ROC_AUC_STEP6 = (0.88, 0.96)  # LC-only baseline (Gate B) -- no static features yet
CEILING_ROC_AUC_STEP8 = (0.93, 0.96)  # with static features -- the RAPID/SuperNNova literature anchor


@pytest.mark.skipif(
    not (CHECKPOINT.exists() and METRICS_JSON.exists()),
    reason="production checkpoint/eval report not present (run `python -m src.train.pretrain_static` first)",
)
class TestCeilingCheck:
    """CLAUDE.md §7 Gate B #3, Day-7 corrected: Ia-vs-rest ROC-AUC should
    land in CEILING_ROC_AUC_STEP8 now that static/host features exist
    (step 8) -- the first live test of the literature anchor itself,
    reported whatever it is, not tuned to reach it. Uses the real
    production (LC+static) checkpoint's test split (same seed/config,
    reloaded from the checkpoint itself so this can never silently drift
    from what was actually trained), reconstructed via the fast
    id-filtered path (Day 6: loading the full 843,617-object corpus just to
    use its 126,542-object test slice made this suite impractical to run to
    completion)."""

    def test_ia_vs_rest_roc_auc_in_expected_range(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]

        model = LCStaticFusionModel(
            lc_input_size=cfg["model"]["lc_input_size"],
            lc_hidden_size=cfg["model"]["lc_hidden_size"],
            lc_num_layers=cfg["model"]["lc_num_layers"],
            lc_dropout=cfg["model"]["lc_dropout"],
            static_hidden_size=cfg["model"]["static_hidden_size"],
            static_output_size=cfg["model"]["static_output_size"],
            num_classes=cfg["model"]["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        normalizer = StaticFeatureNormalizer(
            mean=np.array(checkpoint["static_normalizer_mean"]),
            std=np.array(checkpoint["static_normalizer_std"]),
        )

        lightweight = load_object_class_ids(cfg["train"]["data_parquet"])
        split = split_by_object_id(
            lightweight,
            seed=cfg["train"]["seed"],
            train_frac=cfg["train"]["split"]["train_frac"],
            val_frac=cfg["train"]["split"]["val_frac"],
        )
        test_alerts = load_alerts(cfg["train"]["data_parquet"], object_ids=split.test)
        test_ds = LightCurveStaticDataset(
            test_alerts,
            split.test,
            pre_detection_window_days=cfg["train"]["pre_detection_window_days"],
            normalizer=normalizer,
        )

        predictions = collect_phase_predictions_static(model, test_ds, torch.device("cpu"))
        selected, _ = select_at_cutoff(predictions, cutoff=None)
        y_true = np.array([p.true_class for p in selected])
        y_prob = np.stack([p.probs for p in selected])

        # class index 0 == Ia (src.taxonomy.TARGET_CLASS_IDS[0] == 2222).
        score = roc_auc(y_true, y_prob, num_classes=3)[0]
        lo, hi = CEILING_ROC_AUC_STEP8
        assert lo <= score <= hi, (
            f"Ia-vs-rest ROC-AUC={score:.4f}, expected [{lo},{hi}] "
            "(literature anchor, first live test at step 8 -- not tuned to reach it)"
        )
