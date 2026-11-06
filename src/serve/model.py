"""Model loading for serving: ONNX-exported Tier B classifier + the frozen
Tier A anomaly-reconstruction component (CLAUDE.md §6 step 12,
"inference_worker.py -- ONNX export of the Tier B model").

-----------------------------------------------------------------------------
Why ONNX only wraps the CLASSIFIER, not the anomaly score
-----------------------------------------------------------------------------
The task is specifically "ONNX export of the Tier B model" -- the shipped
classifier. Since 2026-11-05 that is the LC+static fine-tune on
zeropoint-corrected Tier B data
(`outputs/finetune_bts_zpfix/with_static_checkpoint.pt`, Tier B test
macro-F1 0.6201; CLAUDE.md §11 2026-11-05 entry). Before that it was the
LC-only fine-tune (0.6716), chosen on Day 9 on mis-scaled input -- with
correct fluxes LC-only drops to ~0.565 and LC+static wins. The anomaly branch (CLAUDE.md §11 Day 10) is a separate system
built on a DIFFERENT, frozen checkpoint (`outputs/train/checkpoint.pt`, the
Tier A LC-only encoder -- never fine-tuned, still sees the negative-flux
distribution Tier B's `isdiffpos` filter never included) plus a small
decoder head. It stays plain PyTorch here: it is not the model this task
asked to export, and `LCEncoder.forward`'s `pack_padded_sequence` is a
training-time efficiency device, not something ONNX export gains anything
by fighting for a component that already runs in milliseconds per
micro-batch on CPU.

-----------------------------------------------------------------------------
Serving-time anomaly score: reconstruction error only, THEN the full
3-component combination once its fitted scorers were persisted (Day 13)
-----------------------------------------------------------------------------
Day 10's anomaly branch combined FOUR components (reconstruction error,
entropy [dropped, CLAUDE.md §11], Mahalanobis distance, PineForest/
isolation-forest) via equal-weight z-scoring. Building the Day-12 serving
path surfaced something offline evaluation never had to confront:
`MahalanobisScorer` and the isolation-forest scorer are fitted, stateful
objects (`src.anomaly.scoring.MahalanobisScorer`, `src.anomaly.pineforest.
PineForestScorer`/`IsolationForestScorer`) that `src.train.train_anomaly`
never persisted -- only their SCORES on the four eval populations, their
z-score reference statistics, and the trained decoder's weights were saved.
Day 12 shipped reconstruction-error-only as a flagged, stated gap, not a
silent workaround.

Day 13 (`src/train/persist_anomaly_scorers.py`) closed that gap: the frozen
Tier A encoder + already-trained decoder reconstruct the SAME 60,000-object
fit population (same seed), refit `MahalanobisScorer`/`PineForestScorer`
against it (deterministic given the same inputs -- verified by diffing the
refit z-score reference stats against `anomaly_results.json`'s recorded
ones), and persist the fitted objects themselves (`outputs/anomaly/
mahalanobis.npz`, `engineered_normalizer.npz`, `forest.pkl`). `AnomalyScorer`
below loads these if present and serves the full 3-component combination
(entropy dropped, per CLAUDE.md §11 Day 10's own decision that dropping it
IMPROVES the combination on both eval sets); if absent, it degrades to
reconstruction-error-only exactly as before -- never a hard dependency.

The isoforest component needs `src.anomaly.pineforest.engineered_features`,
which reads directly from a `CanonicalAlert`'s own (unpadded) observations
(pineforest.py's own docstring: "not from the padded/windowed tensorized
form"). `AnomalyScorer.score` therefore takes the batch's source
`CanonicalAlert` objects as well as the padded tensors -- both
`inference_worker.py` and `api.py` already hold these per pending item, so
this is a signature addition, not new state either caller has to compute.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from src.anomaly.decoder import (
    ReconstructionDecoder,
    encoder_trunk_hidden_states,
    per_object_reconstruction_error,
    shift_right,
)
from src.anomaly.pineforest import EngineeredFeatureNormalizer, engineered_features
from src.anomaly.scoring import MahalanobisScorer, combine_equal_weight, zscore
from src.features.tensorize import NUM_FEATURES
from src.models.fusion import LCStaticFusionModel
from src.models.lc_encoder import LCEncoder
from src.models.static_encoder import StaticFeatureNormalizer, build_static_vector
from src.schema import CanonicalAlert, StaticContext

logger = logging.getLogger(__name__)

# Deployed classifier: Tier B LC+static fine-tune, retrained 2026-11-05 on
# zeropoint-corrected Tier B data (CLAUDE.md §3 correction, §11 2026-11-05
# entry). It replaced the LC-only model: with correct fluxes LC+static wins
# (macro-F1 0.6210 vs 0.5647, 5-seed means), reversing the Day-9 choice,
# which had been made on mis-scaled input.
TIER_B_CHECKPOINT = Path("outputs/finetune_bts_zpfix/with_static_checkpoint.pt")
TIER_B_ONNX_PATH = Path("outputs/finetune_bts_zpfix/with_static_model.onnx")
TIER_A_ENCODER_CHECKPOINT = Path("outputs/train/checkpoint.pt")
ANOMALY_DECODER_CHECKPOINT = Path("outputs/anomaly/decoder_checkpoint.pt")
ANOMALY_RESULTS_JSON = Path("outputs/anomaly/anomaly_results.json")
# Persisted Day 13 (src/train/persist_anomaly_scorers.py) -- optional; see
# AnomalyScorer.__init__ for the reconstruction-error-only fallback when
# these don't exist.
MAHALANOBIS_NPZ = Path("outputs/anomaly/mahalanobis.npz")
ENGINEERED_NORMALIZER_NPZ = Path("outputs/anomaly/engineered_normalizer.npz")
FOREST_PKL = Path("outputs/anomaly/forest.pkl")

CLASS_NAMES = ("Ia", "Ib/c", "II")  # src.taxonomy.TARGET_CLASS_IDS order


class _ONNXExportableLCStatic(nn.Module):
    """Same computation as `LCStaticFusionModel.forward` in eval mode, minus
    `pack_padded_sequence`/`pad_packed_sequence`. Those ops exist purely to
    skip wasted compute on padded timesteps during TRAINING; at inference
    they change nothing about the values at real timesteps (a causal GRU's
    output at t depends only on inputs at <=t, so zero-padded steps after a
    sequence's real length cannot affect any real timestep -- they only
    produce outputs at positions the caller must not read). Dropping them is
    what makes this exportable. Dropout is the identity in eval mode and is
    omitted. The static branch is per-object, broadcast to every timestep,
    exactly as in training."""

    def __init__(self, model: LCStaticFusionModel) -> None:
        super().__init__()
        self.gru = model.gru
        self.layer_norm = model.layer_norm
        self.static_encoder = model.static_encoder
        self.head = model.head

    def forward(self, features: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        lc_out, _ = self.gru(features)
        lc_out = self.layer_norm(lc_out)
        static_out = self.static_encoder(static_features)
        static_broadcast = static_out.unsqueeze(1).expand(-1, lc_out.shape[1], -1)
        return self.head(torch.cat([lc_out, static_broadcast], dim=-1))


def load_lc_static_checkpoint(checkpoint_path: Path) -> tuple[LCStaticFusionModel, StaticFeatureNormalizer]:
    """Rebuild the fine-tuned LC+static model and the Tier-B-refit static
    normalizer it was trained with. Layer sizes are read from the weight
    shapes, so this does not depend on any Tier A config file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    model = LCStaticFusionModel(
        lc_input_size=NUM_FEATURES,
        lc_hidden_size=sd["layer_norm.weight"].shape[0],
        lc_num_layers=sum(1 for k in sd if k.startswith("gru.weight_ih_l")),
        static_hidden_size=sd["static_encoder.mlp.0.weight"].shape[0],
        static_output_size=sd["static_encoder.mlp.3.weight"].shape[0],
        num_classes=sd["head.weight"].shape[0],
        num_static_fields=sd["static_encoder.mlp.0.weight"].shape[1] // 2,
    )
    model.load_state_dict(sd)
    model.eval()
    normalizer = StaticFeatureNormalizer(
        mean=np.array(ckpt["static_normalizer_mean"]), std=np.array(ckpt["static_normalizer_std"])
    )
    return model, normalizer


def export_tier_b_classifier_to_onnx(
    checkpoint_path: Path = TIER_B_CHECKPOINT, onnx_path: Path = TIER_B_ONNX_PATH
) -> None:
    """One-time export. Idempotent -- `ONNXClassifier` calls this only if
    `onnx_path` doesn't already exist, so the export cost is paid once per
    checkpoint, not once per worker process start."""
    model, _ = load_lc_static_checkpoint(checkpoint_path)
    export_model = _ONNXExportableLCStatic(model)
    export_model.eval()

    n_static_inputs = model.static_encoder.mlp[0].in_features
    dummy = (torch.zeros(1, 5, NUM_FEATURES), torch.zeros(1, n_static_inputs))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["features", "static_features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "static_features": {0: "batch"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    logger.info("Exported Tier B LC+static classifier to %s", onnx_path)


class ONNXClassifier:
    """Wraps an `onnxruntime.InferenceSession` for the Tier B LC+static
    classifier. Takes a zero-padded `(B, T, NUM_FEATURES)` batch, each
    sequence's real `lengths`, and each object's `StaticContext`; returns
    softmax probabilities at each sequence's own last real timestep -- the
    same "predict as of the object's own last detection" convention
    `src.train.pretrain.collect_phase_predictions` uses.

    Static vectors are built with the same `build_static_vector` and the
    same Tier-B-refit normalizer the model was fine-tuned with (stored in
    the checkpoint), so serving cannot drift from training here."""

    def __init__(
        self, onnx_path: Path = TIER_B_ONNX_PATH, checkpoint_path: Path = TIER_B_CHECKPOINT
    ) -> None:
        if not onnx_path.exists():
            logger.info("%s not found -- exporting now", onnx_path)
            export_tier_b_classifier_to_onnx(checkpoint_path=checkpoint_path, onnx_path=onnx_path)
        _, self.normalizer = load_lc_static_checkpoint(checkpoint_path)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def static_matrix(self, statics: Sequence[StaticContext]) -> np.ndarray:
        return np.stack([build_static_vector(s, self.normalizer) for s in statics]).astype(np.float32)

    def predict_probs(
        self, features: np.ndarray, lengths: np.ndarray, statics: Sequence[StaticContext]
    ) -> np.ndarray:
        """`features`: `(B, T, NUM_FEATURES)` float32. `lengths`: `(B,)` int.
        `statics`: `B` `StaticContext`s, one per row.
        Returns `(B, len(CLASS_NAMES))` float32 softmax probabilities."""
        if len(statics) != features.shape[0]:
            raise ValueError(f"got {len(statics)} StaticContexts for a batch of {features.shape[0]}")
        (logits,) = self.session.run(
            ["logits"],
            {"features": features.astype(np.float32), "static_features": self.static_matrix(statics)},
        )
        idx = np.clip(lengths - 1, 0, None)
        last_logits = logits[np.arange(logits.shape[0]), idx]
        exp = np.exp(last_logits - last_logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class AnomalyScorer:
    """Full 3-component combination (reconstruction_error + mahalanobis +
    isoforest, entropy dropped) when the Day-13-persisted scorer artifacts
    are present; degrades to reconstruction-error-only otherwise -- see
    module docstring."""

    def __init__(
        self,
        encoder_checkpoint: Path = TIER_A_ENCODER_CHECKPOINT,
        decoder_checkpoint: Path = ANOMALY_DECODER_CHECKPOINT,
        results_json: Path = ANOMALY_RESULTS_JSON,
        mahalanobis_npz: Path = MAHALANOBIS_NPZ,
        engineered_normalizer_npz: Path = ENGINEERED_NORMALIZER_NPZ,
        forest_pkl: Path = FOREST_PKL,
    ) -> None:
        enc_ckpt = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        self.encoder = LCEncoder(input_size=NUM_FEATURES, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3)
        self.encoder.load_state_dict(enc_ckpt["model_state_dict"])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        dec_ckpt = torch.load(decoder_checkpoint, map_location="cpu", weights_only=False)
        dec_cfg = dec_ckpt["config"]
        self.decoder = ReconstructionDecoder(
            encoder_hidden_size=dec_cfg["encoder_hidden_size"],
            hidden_size=dec_cfg["decoder_hidden_size"],
            dropout=dec_cfg["dropout"],
        )
        self.decoder.load_state_dict(dec_ckpt["decoder_state_dict"])
        self.decoder.eval()

        with open(results_json) as f:
            results = json.load(f)
        self._ref_stats = results["reference_zscore_stats"]

        self.full_combination = (
            mahalanobis_npz.exists() and engineered_normalizer_npz.exists() and forest_pkl.exists()
        )
        if self.full_combination:
            maha_npz = np.load(mahalanobis_npz)
            self._mahalanobis = MahalanobisScorer(mean=maha_npz["mean"], precision=maha_npz["precision"])
            eng_npz = np.load(engineered_normalizer_npz)
            self._engineered_normalizer = EngineeredFeatureNormalizer(mean=eng_npz["mean"], std=eng_npz["std"])
            with open(forest_pkl, "rb") as f:
                self._forest = pickle.load(f)
            logger.info(
                "AnomalyScorer: persisted mahalanobis/isoforest scorers found -- "
                "serving the full 3-component combination"
            )
        else:
            logger.info(
                "AnomalyScorer: %s not found -- serving reconstruction_error only "
                "(run `python -m src.train.persist_anomaly_scorers` to enable the "
                "full combination)",
                forest_pkl,
            )

    @torch.no_grad()
    def score(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        alerts: Optional[list[CanonicalAlert]] = None,
    ) -> np.ndarray:
        """Higher = more anomalous. `alerts` (this batch's source
        `CanonicalAlert`s, same order as `features`) is required to compute
        the isoforest component's engineered features when
        `self.full_combination` is True -- omit it (or run without the
        persisted artifacts) to always get reconstruction-error-only."""
        raw_recon = per_object_reconstruction_error(self.encoder, self.decoder, features, lengths, mask)
        recon_z = zscore(raw_recon.numpy(), *self._ref_stats["reconstruction_error"])

        if not self.full_combination or alerts is None:
            return recon_z

        hidden = encoder_trunk_hidden_states(self.encoder, features, lengths)
        idx = (lengths - 1).clamp(min=0)
        latent = hidden[torch.arange(hidden.shape[0]), idx].numpy()

        maha_raw = self._mahalanobis.score(latent)
        engineered = np.stack([engineered_features(a) for a in alerts], axis=0)
        pine_x = np.concatenate([latent, self._engineered_normalizer.transform(engineered)], axis=1)
        iso_raw = self._forest.score(pine_x)

        maha_z = zscore(maha_raw, *self._ref_stats["mahalanobis"])
        iso_z = zscore(iso_raw, *self._ref_stats["isoforest"])
        return combine_equal_weight([recon_z, maha_z, iso_z])

    @torch.no_grad()
    def score_with_components(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        alerts: list[CanonicalAlert],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Same computation as `score()`, but additionally returns each
        z-scored component that was averaged into the combined score --
        added for the UI's "anomaly score with components broken out"
        panel (CLAUDE.md Day-13-follow-up task), which needs the
        breakdown `score()`'s single-array return throws away.
        `src.serve.inference_worker`/`api.py`'s `/classify` keep calling
        plain `score()` unchanged; this is a read-only addition for the
        object-detail view, not a replacement."""
        raw_recon = per_object_reconstruction_error(self.encoder, self.decoder, features, lengths, mask)
        recon_z = zscore(raw_recon.numpy(), *self._ref_stats["reconstruction_error"])

        if not self.full_combination:
            return recon_z, {"reconstruction_error": recon_z}

        hidden = encoder_trunk_hidden_states(self.encoder, features, lengths)
        idx = (lengths - 1).clamp(min=0)
        latent = hidden[torch.arange(hidden.shape[0]), idx].numpy()

        maha_raw = self._mahalanobis.score(latent)
        engineered = np.stack([engineered_features(a) for a in alerts], axis=0)
        pine_x = np.concatenate([latent, self._engineered_normalizer.transform(engineered)], axis=1)
        iso_raw = self._forest.score(pine_x)

        maha_z = zscore(maha_raw, *self._ref_stats["mahalanobis"])
        iso_z = zscore(iso_raw, *self._ref_stats["isoforest"])
        combined = combine_equal_weight([recon_z, maha_z, iso_z])
        return combined, {"reconstruction_error": recon_z, "mahalanobis": maha_z, "isoforest": iso_z}
