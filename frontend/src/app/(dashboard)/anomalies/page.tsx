"use client";

import { useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useStaticJson } from "@/hooks/use-static-json";
import type { GateDReport } from "@/lib/types";
import { PageHeader, SectionHeading } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ClassBadge } from "@/components/class-badge";
import { ObjectDetailSheet } from "@/components/object-detail-sheet";
import { TableSkeleton, CardSkeleton } from "@/components/panel-skeleton";
import { SourceCite } from "@/components/source-cite";
import { GateBadge } from "@/components/gate-badge";

const COMPONENT_DOCS = [
  {
    name: "Reconstruction error",
    detail:
      "A causal GRU decoder predicts each timestep's flux from the encoder's hidden state as of the previous timestep. High error means the light curve doesn't look like ordinary supernova photometry — the component that survives sim-to-real shift best (Tier B ROC-AUC 0.838).",
  },
  {
    name: "Mahalanobis distance",
    detail:
      "Distance from the Tier A encoder's own latent-space mean, in covariance-normalized units. Strong on Tier A (0.786) but collapses toward chance on Tier B (0.524) — the learned geometry doesn't transfer to real ZTF.",
  },
  {
    name: "Isolation forest",
    detail:
      "PineForest (coniferest) over [latent ‖ engineered features]. Best single Tier A component (0.843), but also degrades under domain shift (0.640 on Tier B).",
  },
  {
    name: "Classifier entropy (dropped)",
    detail:
      "Softmax entropy of the 3-class head. Inverted, not just weak: exotic/weird objects show HIGHER confidence than normal objects, not lower — confidently wrong rather than uncertain. Dropped from the combined score (CLAUDE.md §11, Day 10).",
  },
];

export default function AnomaliesPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const { data: top } = useSWR("top-anomalies", () => api.topAnomalies(50), { refreshInterval: 30000 });
  const { data: gateD } = useStaticJson<GateDReport>("reports/gate_d/gate_d_top50.json");

  return (
    <div className="flex flex-col gap-10">
      <PageHeader
        title="Anomalies"
        description="Reconstruction error, Mahalanobis distance, and an isolation forest, combined by equal-weight z-score average, rank objects the classifier's own training distribution doesn't explain well."
      />

      <section>
        <SectionHeading title="What the four components measure" />
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-4">
          {COMPONENT_DOCS.map((c) => (
            <Card key={c.name} className="py-6">
              <CardHeader><CardTitle className="text-card-title">{c.name}</CardTitle></CardHeader>
              <CardContent>
                <p className="text-body text-muted-foreground">{c.detail}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <SectionHeading
          title="Gate D — manual inspection of the top 50"
          description="Deferred, not passed or failed: the eval set (BTS spectroscopically-confirmed 'weird' categories) had no bogus/artifact candidates to find in the first place."
        />
        <div className="flex flex-wrap items-center gap-6">
          <GateBadge
            label="Gate D status"
            status="deferred"
            detail="Needs an unfiltered alert stream (Tier C) to actually test artifact suppression"
          />
          {gateD ? (
            <div className="flex flex-wrap gap-4">
              {Object.entries(gateD.breakdown).map(([k, v]) => (
                <div key={k} className="flex flex-col items-center rounded-lg border border-border px-4 py-2.5">
                  <span className="text-card-title tabular-nums-all font-semibold">{v}</span>
                  <span className="text-caption capitalize">{k.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          ) : (
            <CardSkeleton lines={1} />
          )}
        </div>
        {gateD ? <SourceCite path="reports/gate_d/gate_d_top50.json" className="mt-3 block" /> : null}
      </section>

      <section>
        <SectionHeading
          title="Top-ranked objects"
          description="Latest prediction per object, ranked by combined anomaly score. Click a row for the full breakdown."
        />
        {!top ? (
          <TableSkeleton />
        ) : (
          <div className="scrollbar-thin overflow-x-auto rounded-lg border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Object</TableHead>
                  <TableHead>Predicted</TableHead>
                  <TableHead>True</TableHead>
                  <TableHead className="text-right">Anomaly score</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {top.results.map((r, i) => (
                  <TableRow key={r.id} className="cursor-pointer hover:bg-accent/60" onClick={() => setSelected(r.object_id)}>
                    <TableCell className="text-muted-foreground tabular-nums-all">{i + 1}</TableCell>
                    <TableCell className="font-mono text-sm">{r.object_id}</TableCell>
                    <TableCell><ClassBadge name={r.predicted_class} /></TableCell>
                    <TableCell><ClassBadge name={r.true_class_name} /></TableCell>
                    <TableCell className="text-right font-medium tabular-nums-all">{r.anomaly_score.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </section>

      <ObjectDetailSheet objectId={selected} open={!!selected} onOpenChange={(o) => !o && setSelected(null)} />
    </div>
  );
}
