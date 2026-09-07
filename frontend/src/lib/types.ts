// Types mirror the FastAPI serving layer (src/serve/api.py, src/serve/db.py)
// and the on-disk report JSONs under reports/ and outputs/*/eval/. Class
// order is always ["Ia", "Ib/c", "II"] -- src.serve.model.CLASS_NAMES.

export type ClassName = "Ia" | "Ib/c" | "II";
export const CLASS_NAMES: ClassName[] = ["Ia", "Ib/c", "II"];

export interface PredictionRow {
  id: number;
  object_id: string;
  survey: string;
  alert_mjd: number;
  scored_at_utc: string;
  predicted_class: ClassName;
  prob_ia: number;
  prob_ibc: number;
  prob_ii: number;
  anomaly_score: number;
  true_class_id: number | null;
  seq_length: number;
  inference_latency_ms: number;
  true_class_name?: ClassName | null;
  correct?: boolean | null;
}

export interface ObjectsResponse {
  total: number;
  limit: number;
  offset: number;
  results: PredictionRow[];
}

export interface RecentAlertsResponse {
  n: number;
  results: PredictionRow[];
}

export interface TopAnomaliesResponse {
  n: number;
  results: PredictionRow[];
}

export interface AlertsRateBucket {
  minute: string;
  n: number;
}

export interface AlertsRateResponse {
  minutes: number;
  buckets: AlertsRateBucket[];
}

export interface StatsResponse {
  n_predictions: number;
  n_objects: number;
  class_distribution: Partial<Record<ClassName, number>>;
  inference_latency_ms: {
    p50_ms: number;
    p99_ms: number;
    mean_ms: number;
    max_ms: number;
  };
}

export interface ObservationOut {
  mjd: number;
  lam_eff: number;
  flux: number;
  flux_err: number;
  is_detection: boolean;
}

export interface ObjectDetailResponse {
  object_id: string;
  survey: string;
  n_observations: number;
  observations: ObservationOut[];
  class_id: number | null;
  true_class_name: ClassName | null;
  correct: boolean | null;
  latest_prediction: PredictionRow | null;
}

export interface ObjectAnomalyResponse {
  object_id: string;
  combined: number;
  components: {
    reconstruction_error: number;
    mahalanobis: number;
    isoforest: number;
    entropy?: number;
  };
  full_combination: boolean;
}

export interface CutoutsResponse {
  object_id: string;
  images: {
    science: string;
    template: string;
    difference: string;
  };
}

export interface HealthResponse {
  status: "ok" | "degraded";
  redis: boolean;
  postgres: boolean;
}

// ---- Report JSON shapes (reports/, outputs/*/eval/metrics.json) ----

export interface PerClassMetrics {
  precision: number[];
  recall: number[];
  f1: number[];
  support: number[];
}

export interface ConfusionBlock {
  raw: number[][];
  normalized: number[][];
}

export interface CutoffMetrics {
  cutoff: number | null;
  n_objects: number;
  n_excluded: number;
  macro_f1: number;
  per_class: PerClassMetrics;
  confusion: ConfusionBlock;
  pr_auc: number[];
  roc_auc: number[];
  calibration: {
    ece: number;
    bin_edges: number[];
    bin_confidence: (number | null)[];
    bin_accuracy: (number | null)[];
    bin_count: number[];
  };
}

// keys are cutoff labels: "-5" | "0" | "5" | "15" | "30" | "full"
export type EvalMetrics = Record<string, CutoffMetrics>;

export interface GateCReport {
  join_stats: Record<string, Record<string, number>>;
  n_3class_labeled: Record<string, number>;
  zero_shot: {
    macro_f1_full: number;
    ia_vs_rest_roc_auc: number;
    per_class_f1_full: number[];
    per_class_support_full: number[];
  };
  with_static: {
    macro_f1_full: number;
    ia_vs_rest_roc_auc: number;
    per_class_f1_full: number[];
    per_class_support_full: number[];
    phase_curve: Record<string, number>;
  };
  lc_only: {
    macro_f1_full: number;
    ia_vs_rest_roc_auc: number;
    per_class_f1_full: number[];
    per_class_support_full: number[];
  };
  with_static_drb: {
    macro_f1_full: number;
    ia_vs_rest_roc_auc: number;
    per_class_f1_full: number[];
    per_class_support_full: number[];
  };
  tier_a_test_macro_f1_full: number;
  sim_to_real_gap: number;
}

export interface AnomalyTierRocAuc {
  reconstruction_error: number;
  entropy: number;
  mahalanobis: number;
  isoforest: number;
  combined: number;
  n_normal: number;
  n_anomalous: number;
}

export interface AnomalyResults {
  forest_kind: string;
  decoder_fit_subsample: number;
  decoder_num_epochs: number;
  reference_zscore_stats: Record<string, [number, number]>;
  tier_a_roc_auc: AnomalyTierRocAuc;
  tier_b_roc_auc: AnomalyTierRocAuc;
  tier_b_category_counts: Record<string, number>;
}

export interface PersistedScorersMeta {
  forest_kind: string;
  fit_zscore_stats_diff: Record<string, { mean_diff: number; std_diff: number }>;
  verification_tier_b_roc_auc: {
    reconstruction_error: number;
    mahalanobis: number;
    isoforest: number;
    combined_3component: number;
    n_normal: number;
    n_anomalous: number;
  };
  prior_tier_b_roc_auc: AnomalyTierRocAuc;
  tier_a_note: string;
  combined_definition: string;
}

export interface GateDObject {
  object_id: string;
  combined: number;
  source: string;
  label: string;
  category: "artifact_like" | "genuine_anomaly" | "short_sparse" | "not_found";
  n_obs: number;
}

export interface GateDReport {
  breakdown: Record<string, number>;
  objects: GateDObject[];
}

export interface SpanSetting {
  max_span_days: number | null;
  macro_f1_full: number;
  n_objects: number;
  n_excluded: number;
  fraction_affected_per_class: Record<
    string,
    { n_total: number; n_affected: number; fraction_affected: number }
  >;
  elapsed_seconds: number;
}

export interface SpanAblationReport {
  span_settings: Record<string, SpanSetting>;
  chosen_default: number;
  chosen_default_reason: string;
}
