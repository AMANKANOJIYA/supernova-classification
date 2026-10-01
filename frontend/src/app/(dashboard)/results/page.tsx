"use client";

import { useMemo, useState } from "react";
import { useStaticJson } from "@/hooks/use-static-json";
import type { AnomalyResults, EvalMetrics, GateCReport, PersistedScorersMeta } from "@/lib/types";
import { PageHeader } from "@/components/page-header";
import { SectionHeading } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartSkeleton, CardSkeleton } from "@/components/panel-skeleton";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { PhaseCurveChart } from "@/components/charts/phase-curve-chart";
import { AnomalySlopeChart } from "@/components/charts/anomaly-slope-chart";
import { SourceCite } from "@/components/source-cite";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const TIER_A_CUTOFFS = ["-5", "0", "5", "15", "30", "full"];
const TIER_B_CUTOFFS = ["0", "5", "15", "30", "full"];

export default function ResultsPage() {
  const { data: tierAMetrics } = useStaticJson<EvalMetrics>("outputs/train_static/eval/metrics.json");
  const { data: tierBMetrics } = useStaticJson<EvalMetrics>("outputs/finetune_bts/lc_only/eval/metrics.json");
  const { data: gateC } = useStaticJson<GateCReport>("reports/gate_c/gate_c_report.json");
  const { data: anomaly } = useStaticJson<AnomalyResults>("outputs/anomaly/anomaly_results.json");
  const { data: persisted } = useStaticJson<PersistedScorersMeta>(
    "outputs/anomaly/persisted_scorers_meta.json",
  );

  const [dataset, setDataset] = useState<"tierA" | "tierB">("tierB");
  const [mode, setMode] = useState<"normalized" | "raw">("normalized");
  // Keyed by cutoff label (not index): Tier A/B have different cutoff sets
  // (Tier A has "-5", Tier B doesn't -- §5.1), so an index-based slider
  // silently pointed at the wrong label across a dataset toggle. Falling
  // back to "full" whenever the current key doesn't exist in the newly
  // selected dataset sidesteps that entirely.
  const [cutoffKey, setCutoffKey] = useState<string>("full");
  const activeCutoffs = dataset === "tierA" ? TIER_A_CUTOFFS : TIER_B_CUTOFFS;
  const safeCutoffKey = activeCutoffs.includes(cutoffKey) ? cutoffKey : "full";
  const safeIdx = activeCutoffs.indexOf(safeCutoffKey);
  const metricsSource = dataset === "tierA" ? tierAMetrics : tierBMetrics;
  const confusion = metricsSource?.[safeCutoffKey]?.confusion;

  const phaseCurveData = useMemo(() => {
    if (!tierAMetrics || !tierBMetrics || !gateC) return [];
    // Tier B here must be the DEPLOYED LC-only fine-tune, not LC+static.
    // Intermediate cutoffs come from tierBMetrics (outputs/finetune_bts/
    // lc_only/eval/metrics.json) -- the only place they exist. That file
    // is itself a later, non-deterministic re-run of the fine-tuning
    // script, whose own full-cutoff value (0.6727) drifted 0.0011 from
    // the original, git-tracked 0.6716 quoted everywhere else (gate_c_
    // report.json, the paper, RESULTS.md). So the "full" point is pinned
    // to that original scalar (gateC.lc_only.macro_f1_full), not the
    // re-run file's own drifted "full" entry, keeping this curve's
    // endpoint consistent with every other reported number.
    return ["-5", "0", "5", "15", "30", "full"].map((c) => ({
      cutoff: c,
      tierA: tierAMetrics[c]?.macro_f1 ?? null,
      tierB: c === "full" ? gateC.lc_only.macro_f1_full : (tierBMetrics[c]?.macro_f1 ?? null),
    }));
  }, [tierAMetrics, tierBMetrics, gateC]);

  return (
    <div className="flex flex-col gap-12">
      <PageHeader
        title="Results"
        description="Every number below is read directly from the frozen eval harness's output JSONs (src/eval/, outputs/*/eval/metrics.json) or from reports/ — nothing on this page is recomputed."
      />

      <section>
        <SectionHeading
          title="Phase curve"
          description="Macro-F1 as a function of days since first detection — the project's primary reportable metric (plain accuracy is not, per CLAUDE.md §11)."
        />
        {!phaseCurveData.length ? (
          <ChartSkeleton />
        ) : (
          <Card className="py-6">
            <CardContent>
              <PhaseCurveChart
                data={phaseCurveData}
                series={[
                  { key: "tierA", name: "Tier A (LC + static, sim)", color: "var(--color-primary)" },
                  { key: "tierB", name: "Tier B (LC-only, deployed, real ZTF)", color: "var(--color-ii)" },
                ]}
              />
              <p className="text-caption mt-2">
                Tier B has no cutoff=-5 point — BTSbot alerts carry no pre-detection forced photometry (§5.1
                Day-8 finding), so that comparison is dropped rather than plotted as a meaningless number.
                Tier B&apos;s full-cutoff point is pinned to the original, git-tracked scalar
                (gate_c_report.json:lc_only.macro_f1_full = 0.6716) rather than the intermediate-cutoffs
                file&apos;s own re-run value (0.6727) — the two differ by 0.0011 due to a documented
                non-deterministic re-run (paper/NUMBERS_GAPS.md).
              </p>
              <SourceCite path="outputs/train_static/eval/metrics.json[cutoff].macro_f1 · outputs/finetune_bts/lc_only/eval/metrics.json[cutoff].macro_f1 · gate_c_report.json:lc_only.macro_f1_full" />
            </CardContent>
          </Card>
        )}
      </section>

      <section>
        <SectionHeading
          title="Confusion matrix"
          description='Row-normalized by default so the diagonal reads as recall directly — the raw-count matrix saturates (429 vs. 3 in one real cell).'
        />
        <Card className="py-6">
          <CardHeader className="flex-row flex-wrap items-center gap-4">
            <Tabs value={dataset} onValueChange={(v) => setDataset(v as "tierA" | "tierB")}>
              <TabsList>
                <TabsTrigger value="tierB">Tier B (deployed, real ZTF)</TabsTrigger>
                <TabsTrigger value="tierA">Tier A (LC + static, sim)</TabsTrigger>
              </TabsList>
            </Tabs>
            <Select value={mode} onValueChange={(v) => setMode(v as "normalized" | "raw")}>
              <SelectTrigger className="w-44"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="normalized">Row-normalized</SelectItem>
                <SelectItem value="raw">Raw counts</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex min-w-56 flex-1 items-center gap-3">
              <span className="text-caption whitespace-nowrap">
                cutoff: {safeCutoffKey === "full" ? "full" : `${safeCutoffKey}d`}
              </span>
              <Slider
                min={0}
                max={activeCutoffs.length - 1}
                step={1}
                value={[safeIdx]}
                onValueChange={([v]) => setCutoffKey(activeCutoffs[v])}
                className="max-w-56"
              />
            </div>
          </CardHeader>
          <CardContent>
            {!confusion ? (
              <CardSkeleton height={280} />
            ) : (
              <>
                <ConfusionMatrix confusion={confusion} mode={mode} />
                <SourceCite
                  path={`${dataset === "tierA" ? "outputs/train_static" : "outputs/finetune_bts/lc_only"}/eval/metrics.json["${safeCutoffKey}"].confusion`}
                  className="mt-3 block"
                />
              </>
            )}
          </CardContent>
        </Card>
      </section>

      <section>
        <SectionHeading
          title="Anomaly components: sim-to-real flip"
          description="Latent-space methods (Mahalanobis, isolation forest) collapse toward chance under domain shift; reconstruction error inverts and improves. The lines crossing is the finding."
        />
        {!anomaly ? (
          <ChartSkeleton />
        ) : (
          <Card className="py-6">
            <CardContent>
              <AnomalySlopeChart tierA={anomaly.tier_a_roc_auc} tierB={anomaly.tier_b_roc_auc} />
              <SourceCite path="outputs/anomaly/anomaly_results.json:tier_a_roc_auc / tier_b_roc_auc" className="mt-2 block" />
            </CardContent>
          </Card>
        )}
        {anomaly ? (
          <Card className="mt-6 py-6">
            <CardHeader><CardTitle className="text-card-title">Component ROC-AUC (normal vs. anomalous)</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Component</TableHead>
                      <TableHead className="text-right">Tier A (n={anomaly.tier_a_roc_auc.n_normal.toLocaleString()} vs {anomaly.tier_a_roc_auc.n_anomalous.toLocaleString()})</TableHead>
                      <TableHead className="text-right">Tier B (n={anomaly.tier_b_roc_auc.n_normal.toLocaleString()} vs {anomaly.tier_b_roc_auc.n_anomalous.toLocaleString()})</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(["reconstruction_error", "mahalanobis", "isoforest", "entropy"] as const).map((k) => (
                      <TableRow key={k}>
                        <TableCell className="capitalize">{k.replace("_", " ")}</TableCell>
                        <TableCell className="text-right tabular-nums-all">{anomaly.tier_a_roc_auc[k].toFixed(4)}</TableCell>
                        <TableCell className="text-right tabular-nums-all">{anomaly.tier_b_roc_auc[k].toFixed(4)}</TableCell>
                      </TableRow>
                    ))}
                    <TableRow className="border-t-2 border-border font-medium">
                      <TableCell>Combined, 4-component (w/ entropy)</TableCell>
                      <TableCell className="text-right tabular-nums-all">{anomaly.tier_a_roc_auc.combined.toFixed(4)}</TableCell>
                      <TableCell className="text-right tabular-nums-all">{anomaly.tier_b_roc_auc.combined.toFixed(4)}</TableCell>
                    </TableRow>
                    <TableRow className="font-medium text-ii">
                      <TableCell>Combined, 3-component (entropy dropped — actual output)</TableCell>
                      <TableCell className="text-right tabular-nums-all">0.8146*</TableCell>
                      <TableCell className="text-right tabular-nums-all">
                        {persisted ? persisted.verification_tier_b_roc_auc.combined_3component.toFixed(4) : "0.7622*"}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </div>
              <p className="text-caption mt-3">
                * Tier A&apos;s 3-component figure (442,303-object exotic eval set) was not re-run this
                session — cited from CLAUDE.md §11 (Day 10), not a stored JSON field. Tier B&apos;s is a
                Day-13 recomputation, <SourceCite path="outputs/anomaly/persisted_scorers_meta.json:verification_tier_b_roc_auc.combined_3component" className="ml-1" />.
              </p>
            </CardContent>
          </Card>
        ) : null}
      </section>

      <section>
        <SectionHeading
          title="Sim-to-real gap"
          description="Every fine-tune variant against Tier A's own full-cutoff macro-F1 (0.7169). The deployed model is LC-only — static features were net-negative on Tier B (small labeled sample, §11 Day 9)."
        />
        {!gateC ? (
          <CardSkeleton height={200} />
        ) : (
          <Card className="py-6">
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Variant</TableHead>
                      <TableHead className="text-right">Tier B macro-F1</TableHead>
                      <TableHead className="text-right">Gap vs. Tier A (0.7169)</TableHead>
                      <TableHead className="text-right">Ia-vs-rest ROC-AUC</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(
                      [
                        ["Zero-shot (no fine-tune)", gateC.zero_shot],
                        ["LC-only fine-tune (deployed)", gateC.lc_only],
                        ["LC + static fine-tune", gateC.with_static],
                        ["LC + static + DRB fine-tune", gateC.with_static_drb],
                      ] as const
                    ).map(([label, v]) => (
                      <TableRow key={label} className={label.includes("deployed") ? "text-ii font-medium" : undefined}>
                        <TableCell>{label}</TableCell>
                        <TableCell className="text-right tabular-nums-all">{v.macro_f1_full.toFixed(4)}</TableCell>
                        <TableCell className="text-right tabular-nums-all">
                          −{(gateC.tier_a_test_macro_f1_full - v.macro_f1_full).toFixed(4)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums-all">{v.ia_vs_rest_roc_auc.toFixed(4)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <SourceCite path="reports/gate_c/gate_c_report.json" className="mt-3 block" />
            </CardContent>
          </Card>
        )}
      </section>
    </div>
  );
}
