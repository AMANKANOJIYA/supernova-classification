"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ClassBadge } from "@/components/class-badge";
import { EmptyState } from "@/components/empty-state";
import { ChartSkeleton, TableSkeleton } from "@/components/panel-skeleton";
import { ObjectDetailSheet } from "@/components/object-detail-sheet";
import { makeChartTooltip } from "@/components/charts/chart-tooltip";
import { AXIS_TEXT, NEUTRAL_GRID } from "@/lib/colors";
import { fmtMs, relTime } from "@/lib/format";
import { Radio, Terminal } from "lucide-react";

export default function LiveMonitorPage() {
  const [survey, setSurvey] = useState<string>("all");
  const [predictedClass, setPredictedClass] = useState<string>("all");
  const [selected, setSelected] = useState<string | null>(null);

  const { data: recent } = useSWR(
    ["alerts-recent", survey, predictedClass],
    () =>
      api.recentAlerts({
        limit: 150,
        survey: survey === "all" ? undefined : survey,
        predicted_class: predictedClass === "all" ? undefined : predictedClass,
      }),
    { refreshInterval: 5000 },
  );
  const { data: rate } = useSWR("alerts-rate-live", () => api.alertsRate(30), {
    refreshInterval: 8000,
  });

  const chartData = useMemo(
    () =>
      (rate?.buckets ?? []).map((b) => ({
        minute: new Date(b.minute).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        n: b.n,
      })),
    [rate],
  );

  // `alerts_per_minute` (src/serve/db.py) omits empty buckets entirely, so
  // the LAST bucket is just "the most recent minute that had any alerts" --
  // it could be from hours ago. Checking its timestamp against wall-clock
  // now (not just "is the array non-empty") is what actually answers
  // "is something streaming right now".
  const lastBucketAt = rate?.buckets.length ? new Date(rate.buckets[rate.buckets.length - 1].minute).getTime() : 0;
  const isCurrentlyLive = lastBucketAt > 0 && Date.now() - lastBucketAt < 90_000;

  const TooltipContent = makeChartTooltip((payload, label) => ({
    title: String(label),
    rows: [{ label: "alerts", value: String((payload[0] as { value: number }).value), color: "var(--color-primary)" }],
  }));

  const noData = recent && recent.n === 0;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Live monitor"
        description="Alerts scored by the serving pipeline (Kafka -> Redis state -> ONNX classifier + anomaly scorer -> Postgres), replay or live Fink."
      />

      {!rate ? (
        <ChartSkeleton />
      ) : (
        <Card className="py-6">
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="text-card-title flex items-center gap-2">
              <Radio className={`size-4 ${isCurrentlyLive ? "text-ii" : "text-muted-foreground"}`} />
              Alerts per minute
            </CardTitle>
            <span className="text-caption">{isCurrentlyLive ? "streaming now" : "no traffic in the last 2 min"}</span>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData} margin={{ top: 8, right: 20, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={NEUTRAL_GRID} vertical={false} />
                <XAxis
                  dataKey="minute"
                  tick={{ fill: AXIS_TEXT, fontSize: 14 }}
                  tickLine={false}
                  axisLine={{ stroke: NEUTRAL_GRID }}
                  label={{ value: "time (UTC)", position: "insideBottom", offset: -6, fill: AXIS_TEXT, fontSize: 14 }}
                />
                <YAxis
                  tick={{ fill: AXIS_TEXT, fontSize: 14 }}
                  tickLine={false}
                  axisLine={{ stroke: NEUTRAL_GRID }}
                  width={40}
                  label={{ value: "alerts / min", angle: -90, position: "insideLeft", fill: AXIS_TEXT, fontSize: 14 }}
                />
                <Tooltip content={TooltipContent} cursor={{ fill: NEUTRAL_GRID }} />
                <Legend
                  verticalAlign="bottom"
                  height={32}
                  wrapperStyle={{ fontSize: 14, paddingTop: 8 }}
                  formatter={() => "alerts / minute"}
                />
                <Bar dataKey="n" fill="var(--color-primary)" radius={[3, 3, 0, 0]} isAnimationActive animationDuration={500} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <section className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <Select value={survey} onValueChange={setSurvey}>
            <SelectTrigger className="w-40"><SelectValue placeholder="Survey" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All surveys</SelectItem>
              <SelectItem value="ztf">ZTF</SelectItem>
              <SelectItem value="lsst">LSST</SelectItem>
              <SelectItem value="elasticc">ELAsTiCC2</SelectItem>
            </SelectContent>
          </Select>
          <Select value={predictedClass} onValueChange={setPredictedClass}>
            <SelectTrigger className="w-44"><SelectValue placeholder="Predicted class" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All predicted classes</SelectItem>
              <SelectItem value="Ia">Ia</SelectItem>
              <SelectItem value="Ib/c">Ib/c</SelectItem>
              <SelectItem value="II">II</SelectItem>
            </SelectContent>
          </Select>
          {recent ? <span className="text-caption ml-auto">{recent.n} alert(s) shown</span> : null}
        </div>

        {!recent ? (
          <TableSkeleton />
        ) : noData ? (
          <EmptyState
            icon={Terminal}
            title="Nothing streaming yet"
            description={
              <>
                Start the replay producer and worker to populate this feed:
                <br />
                <code className="mt-2 block rounded bg-muted px-2 py-1 font-mono text-foreground">
                  make replay
                </code>
                <code className="mt-1 block rounded bg-muted px-2 py-1 font-mono text-foreground">
                  make worker
                </code>
              </>
            }
          />
        ) : (
          <div className="scrollbar-thin overflow-x-auto rounded-lg border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Object</TableHead>
                  <TableHead>Survey</TableHead>
                  <TableHead>Predicted</TableHead>
                  <TableHead>True</TableHead>
                  <TableHead className="text-right">Confidence</TableHead>
                  <TableHead className="text-right">Anomaly</TableHead>
                  <TableHead className="text-right">Latency</TableHead>
                  <TableHead className="text-right">Scored</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recent.results.map((r) => {
                  const conf = Math.max(r.prob_ia, r.prob_ibc, r.prob_ii);
                  return (
                    <TableRow
                      key={r.id}
                      className="cursor-pointer hover:bg-accent/60"
                      onClick={() => setSelected(r.object_id)}
                    >
                      <TableCell className="font-mono text-sm">{r.object_id}</TableCell>
                      <TableCell className="uppercase text-muted-foreground">{r.survey}</TableCell>
                      <TableCell><ClassBadge name={r.predicted_class} /></TableCell>
                      <TableCell><ClassBadge name={r.true_class_name} /></TableCell>
                      <TableCell className="text-right tabular-nums-all">{(conf * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-right tabular-nums-all">{r.anomaly_score.toFixed(2)}</TableCell>
                      <TableCell className="text-right tabular-nums-all">{fmtMs(r.inference_latency_ms)}</TableCell>
                      <TableCell className="text-right text-muted-foreground">{relTime(r.scored_at_utc)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </section>

      <ObjectDetailSheet objectId={selected} open={!!selected} onOpenChange={(o) => !o && setSelected(null)} />
    </div>
  );
}
