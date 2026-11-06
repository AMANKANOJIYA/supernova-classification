"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import { useStaticJson } from "@/hooks/use-static-json";
import type { AnomalyResults, GateCReport, PersistedScorersMeta } from "@/lib/types";
import { PageHeader } from "@/components/page-header";
import { StatCard } from "@/components/stat-card";
import { GateBadge } from "@/components/gate-badge";
import { CardSkeleton } from "@/components/panel-skeleton";
import { SourceCite } from "@/components/source-cite";
import { Sparkline } from "@/components/charts/sparkline";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fmtCompact, fmtMs } from "@/lib/format";

export default function OverviewPage() {
  const { data: gateC } = useStaticJson<GateCReport>("reports/zeropoint_fix_2026-11-05/finetune_seed42/gate_c_report.json");
  const { data: anomaly } = useStaticJson<AnomalyResults>("outputs/anomaly/anomaly_results.json");
  const { data: persisted } = useStaticJson<PersistedScorersMeta>(
    "reports/zeropoint_fix_2026-11-05/anomaly_tier_b/persisted_scorers_meta.json",
  );
  const { data: stats } = useSWR("stats-overview", () => api.stats(), { refreshInterval: 15000 });
  const { data: rate } = useSWR("rate-overview", () => api.alertsRate(60), { refreshInterval: 20000 });

  const loading = !gateC || !anomaly || !persisted || !stats;

  const tierAF1 = gateC?.tier_a_test_macro_f1_full;
  const tierBF1 = gateC?.with_static.macro_f1_full;
  const gap = tierAF1 !== undefined && tierBF1 !== undefined ? tierAF1 - tierBF1 : undefined;
  const anomalyRocAuc = persisted?.verification_tier_b_roc_auc.combined_3component;

  return (
    <div className="flex flex-col gap-10">
      <PageHeader
        title="Overview"
        description="A multi-survey, multimodal pipeline classifying transient alerts into SN Ia / Ib-c / II, pretrained on simulated LSST light curves and fine-tuned on real ZTF alerts with spectroscopic labels. Anomalies are ranked nightly against a held-out exotic-object population."
      />

      <section>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-5">
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} lines={2} />)
          ) : (
            <>
              <StatCard
                label="Tier A macro-F1 (LC + static)"
                value={tierAF1 !== undefined ? tierAF1.toFixed(4) : "—"}
                caption="ELAsTiCC2 sim, full cutoff"
                accent="primary"
              >
                <SourceCite path="finetune_seed42/gate_c_report.json:tier_a_test_macro_f1_full" className="mt-1" />
              </StatCard>
              <StatCard
                label="Tier B macro-F1 (deployed)"
                value={tierBF1 !== undefined ? tierBF1.toFixed(4) : "—"}
                caption="Real ZTF, LC+static fine-tune (seed 42)"
                accent="ii"
              >
                <SourceCite path="finetune_seed42/gate_c_report.json:with_static.macro_f1_full" className="mt-1" />
              </StatCard>
              <StatCard
                label="Sim-to-real gap"
                value={gap !== undefined ? `−${gap.toFixed(4)}` : "—"}
                caption="LC+static vs LC+static (Gate C)"
                accent="ibc"
              >
                <SourceCite path="finetune_seed42/gate_c_report.json (A − B, LC+static)" className="mt-1" />
              </StatCard>
              <StatCard
                label="Anomaly ROC-AUC"
                value={anomalyRocAuc !== undefined ? anomalyRocAuc.toFixed(4) : "—"}
                caption="Tier B, 3-component, corrected flux"
                accent="ia"
              >
                <SourceCite path="zeropoint_fix_2026-11-05/anomaly_tier_b/persisted_scorers_meta.json" className="mt-1" />
              </StatCard>
              <StatCard
                label="Serving latency (p50)"
                value={stats ? fmtMs(stats.inference_latency_ms.p50_ms) : "—"}
                caption={stats ? `p99 ${fmtMs(stats.inference_latency_ms.p99_ms)} · live` : ""}
              >
                <SourceCite path="GET /stats" className="mt-1" />
              </StatCard>
            </>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-section-heading mb-4">Gate status</h2>
        <div className="flex flex-wrap gap-4">
          <GateBadge
            label="Gate A — readers"
            status="pass"
            detail="Light curves plausible, negative fluxes present, class counts sane"
          />
          <GateBadge
            label="Gate B — first model"
            status="pass"
            detail="Shuffled-label collapse, causal-mask, ROC-AUC ceiling all pass"
          />
          <GateBadge
            label="Gate C — fine-tune"
            status="pass"
            detail="Tier A → B macro-F1 drop recorded, expected direction"
          />
          <GateBadge
            label="Gate D — anomaly"
            status="deferred"
            detail="Eval set had no bogus candidates — needs unfiltered stream"
          />
        </div>
      </section>

      <section>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="py-6 lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-card-title">What this pipeline does</CardTitle>
            </CardHeader>
            <CardContent className="text-body prose-relaxed text-muted-foreground">
              <p>
                Light curves are trained on calibrated <strong className="text-foreground">flux</strong>, never
                magnitude — difference-image photometry produces real, informative negative fluxes concentrated
                in the early, pre-peak regime this project prioritizes. Every model is strictly causal
                (<code className="text-foreground">bidirectional=False</code>), so a day-3 prediction never sees
                day-60 data. A held-out population of exotic ELAsTiCC2 classes and confirmed non-SN BTS objects
                backs the nightly anomaly ranker below.
              </p>
            </CardContent>
          </Card>
          <Card className="py-6">
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle className="text-card-title">Live throughput</CardTitle>
              <span className="text-caption tabular-nums-all">
                {stats ? `${fmtCompact(stats.n_predictions)} scored` : "—"}
              </span>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <Sparkline
                data={(rate?.buckets ?? []).map((b) => ({ minute: b.minute, n: b.n }))}
                color="var(--color-ia)"
                height={72}
              />
              <p className="text-caption">
                Alerts/minute, trailing 60 min · stored benchmark 575 alerts/sec on one node, p50 37.6 ms
                (Tier B test replay, deployed LC+static model, 2026-11-05).
                <SourceCite path="reports/zeropoint_fix_2026-11-05/serving_lc_static/replay_benchmark_uncontended.json" className="ml-1" />
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {anomaly ? (
        <p className="text-caption">
          Class distribution shown across the sidebar and Objects tab reflects{" "}
          <strong className="text-foreground">
            {stats ? fmtCompact(stats.n_objects) : "—"}
          </strong>{" "}
          scored objects. Ib/c recall in the 0.40–0.60 band is expected, not a bug — see Results.
        </p>
      ) : null}
    </div>
  );
}
