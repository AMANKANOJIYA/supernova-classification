"use client";

import {
  CartesianGrid,
  ReferenceLine,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";
import { useMemo } from "react";
import type { ObservationOut } from "@/lib/types";
import { nearestBandName, wavelengthToColor } from "@/lib/colors";
import { makeChartTooltip } from "./chart-tooltip";
import { NEUTRAL_GRID, AXIS_TEXT } from "@/lib/colors";

interface Point {
  day: number;
  flux: number;
  flux_err: number;
  is_detection: boolean;
  band: string;
  mjd: number;
}

function DetectionDot(props: {
  cx?: number;
  cy?: number;
  fill?: string;
  payload?: Point;
}) {
  const { cx, cy, fill, payload } = props;
  if (cx === undefined || cy === undefined) return null;
  const isDet = payload?.is_detection ?? true;
  if (isDet) {
    return <circle cx={cx} cy={cy} r={4} fill={fill} stroke="none" fillOpacity={0.92} />;
  }
  return <circle cx={cx} cy={cy} r={3.5} fill="none" stroke={fill} strokeWidth={1.4} fillOpacity={0.5} />;
}

export function LightCurveChart({ observations }: { observations: ObservationOut[] }) {
  const { series } = useMemo(() => {
    const dets = observations.filter((o) => o.is_detection);
    const firstDet = dets.length ? Math.min(...dets.map((o) => o.mjd)) : observations[0]?.mjd ?? 0;
    const byBand = new Map<string, Point[]>();
    for (const o of observations) {
      const band = nearestBandName(o.lam_eff);
      const arr = byBand.get(band) ?? [];
      arr.push({
        day: o.mjd - firstDet,
        flux: o.flux,
        flux_err: o.flux_err,
        is_detection: o.is_detection,
        band,
        mjd: o.mjd,
      });
      byBand.set(band, arr);
    }
    const s = Array.from(byBand.entries()).map(([band, points]) => {
      const repLamEff = observations.find((o) => nearestBandName(o.lam_eff) === band)!.lam_eff;
      return { band, color: wavelengthToColor(repLamEff), points };
    });
    return { series: s };
  }, [observations]);

  const TooltipContent = makeChartTooltip((payload) => {
    const p = (payload[0] as { payload: Point })?.payload;
    if (!p) return null;
    return {
      title: `${p.band} · day ${p.day.toFixed(1)}`,
      rows: [
        { label: "flux", value: p.flux.toFixed(1), color: "var(--color-primary)" },
        { label: "flux_err", value: `± ${p.flux_err.toFixed(1)}`, color: "var(--color-muted-foreground)" },
        { label: p.is_detection ? "detection" : "non-detection", value: p.mjd.toFixed(3), color: "var(--color-muted-foreground)" },
      ],
    };
  });

  return (
    <div>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 12, right: 20, bottom: 24, left: 8 }}>
          <CartesianGrid stroke={NEUTRAL_GRID} vertical={false} />
          <XAxis
            type="number"
            dataKey="day"
            name="days since first detection"
            tick={{ fill: AXIS_TEXT, fontSize: 14 }}
            tickLine={false}
            axisLine={{ stroke: NEUTRAL_GRID }}
            label={{
              value: "days since first detection",
              position: "insideBottom",
              offset: -10,
              fill: AXIS_TEXT,
              fontSize: 14,
            }}
          />
          <YAxis
            type="number"
            dataKey="flux"
            name="flux"
            tick={{ fill: AXIS_TEXT, fontSize: 14 }}
            tickLine={false}
            axisLine={{ stroke: NEUTRAL_GRID }}
            width={64}
            label={{
              value: "flux (FLUXCAL, zp=27.5)",
              angle: -90,
              position: "insideLeft",
              fill: AXIS_TEXT,
              fontSize: 14,
            }}
          />
          <ReferenceLine y={0} stroke={NEUTRAL_GRID} strokeDasharray="4 4" />
          <Tooltip content={TooltipContent} cursor={{ stroke: NEUTRAL_GRID }} />
          {series.map((s) => (
            <Scatter
              key={s.band}
              name={s.band}
              data={s.points}
              fill={s.color}
              shape={(p: unknown) => <DetectionDot {...(p as object)} fill={s.color} />}
              isAnimationActive
              animationDuration={500}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
        {series.map((s) => (
          <div key={s.band} className="flex items-center gap-1.5 text-caption">
            <span className="inline-block size-2 rounded-full" style={{ background: s.color }} />
            {s.band}
          </div>
        ))}
        <div className="flex items-center gap-1.5 text-caption">
          <span className="inline-block size-2 rounded-full border border-muted-foreground" />
          non-detection / forced photometry
        </div>
      </div>
    </div>
  );
}
