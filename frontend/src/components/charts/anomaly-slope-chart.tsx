"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TEXT, NEUTRAL_GRID } from "@/lib/colors";
import { makeChartTooltip } from "./chart-tooltip";
import { useLegendToggle } from "./legend-toggle";
import type { AnomalyTierRocAuc } from "@/lib/types";

// Recharts silently DROPS a category tick's <text> from the DOM (not just
// clips it visually) when the default centered anchor would push it past
// the plot's left/right bound -- exactly what happens with only 2 wide
// category labels ("Tier A (ELAsTiCC2 exotics)" centered at the leftmost
// tick position overflows left, past the Y-axis). Anchoring the two edge
// ticks to "start"/"end" instead of "middle" keeps both on-canvas.
function EdgeAwareTick(props: { x?: number; y?: number; payload?: { value: string }; index?: number }) {
  const { x = 0, y = 0, payload, index } = props;
  const anchor = index === 0 ? "start" : "end";
  return (
    <text x={x} y={y + 16} textAnchor={anchor} fill={AXIS_TEXT} fontSize={14}>
      {payload?.value}
    </text>
  );
}

const COMPONENT_META: Array<{ key: keyof AnomalyTierRocAuc; name: string; color: string; dashed?: boolean }> = [
  { key: "reconstruction_error", name: "Reconstruction error", color: "#8b6cf0" },
  { key: "mahalanobis", name: "Mahalanobis distance", color: "#38bdf8" },
  { key: "isoforest", name: "Isolation forest", color: "#f43f5e" },
  { key: "entropy", name: "Classifier entropy (dropped)", color: "#71717a", dashed: true },
];

export function AnomalySlopeChart({
  tierA,
  tierB,
}: {
  tierA: AnomalyTierRocAuc;
  tierB: AnomalyTierRocAuc;
}) {
  const data = [
    { dataset: "Tier A (ELAsTiCC2 exotics)", ...tierA },
    { dataset: "Tier B (BTS weird)", ...tierB },
  ];
  const { onLegendClick, isHidden } = useLegendToggle();

  const TooltipContent = makeChartTooltip((payload, label) => ({
    title: String(label),
    rows: payload.map((p) => {
      const pt = p as { name: string; value: number; color: string };
      return { label: pt.name, value: pt.value.toFixed(4), color: pt.color };
    }),
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 12, right: 24, bottom: 24, left: 8 }}>
        <CartesianGrid stroke={NEUTRAL_GRID} vertical={false} />
        <XAxis
          dataKey="dataset"
          tick={<EdgeAwareTick />}
          tickLine={false}
          axisLine={{ stroke: NEUTRAL_GRID }}
          interval={0}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fill: AXIS_TEXT, fontSize: 14 }}
          tickLine={false}
          axisLine={{ stroke: NEUTRAL_GRID }}
          width={44}
          label={{ value: "ROC-AUC", angle: -90, position: "insideLeft", fill: AXIS_TEXT, fontSize: 14 }}
        />
        <Tooltip content={TooltipContent} cursor={{ stroke: NEUTRAL_GRID }} />
        <Legend
          verticalAlign="bottom"
          height={48}
          wrapperStyle={{ fontSize: 14, paddingTop: 12, cursor: "pointer" }}
          onClick={onLegendClick}
          formatter={(value, entry) => {
            const key = String((entry as { dataKey?: unknown })?.dataKey ?? "");
            return <span style={{ opacity: isHidden(key) ? 0.4 : 1 }}>{value}</span>;
          }}
        />
        {COMPONENT_META.map((c) => (
          <Line
            key={c.key}
            type="linear"
            dataKey={c.key}
            name={c.name}
            stroke={c.color}
            strokeWidth={2.5}
            strokeDasharray={c.dashed ? "5 4" : undefined}
            dot={{ r: 5, fill: c.color, strokeWidth: 0 }}
            activeDot={{ r: 7 }}
            hide={isHidden(c.key)}
            isAnimationActive
            animationDuration={600}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
