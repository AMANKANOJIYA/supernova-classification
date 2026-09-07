"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_TEXT, NEUTRAL_GRID } from "@/lib/colors";
import { makeChartTooltip } from "./chart-tooltip";
import { useLegendToggle } from "./legend-toggle";

export interface PhaseCurveSeries {
  key: string;
  name: string;
  color: string;
}

export function PhaseCurveChart({
  data,
  series,
}: {
  data: Array<Record<string, number | string | null>>;
  series: PhaseCurveSeries[];
}) {
  const { onLegendClick, isHidden } = useLegendToggle();
  const TooltipContent = makeChartTooltip((payload, label) => ({
    title: `cutoff: ${label}`,
    rows: payload
      .filter((p) => (p as { value: unknown }).value !== null && (p as { value: unknown }).value !== undefined)
      .map((p) => {
        const pt = p as { name: string; value: number; color: string };
        return { label: pt.name, value: pt.value.toFixed(4), color: pt.color };
      }),
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 12, right: 24, bottom: 24, left: 8 }}>
        <CartesianGrid stroke={NEUTRAL_GRID} vertical={false} />
        <XAxis
          dataKey="cutoff"
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
          domain={[0, 1]}
          tick={{ fill: AXIS_TEXT, fontSize: 14 }}
          tickLine={false}
          axisLine={{ stroke: NEUTRAL_GRID }}
          width={44}
          label={{ value: "macro-F1", angle: -90, position: "insideLeft", fill: AXIS_TEXT, fontSize: 14 }}
        />
        <Tooltip content={TooltipContent} cursor={{ stroke: NEUTRAL_GRID }} />
        <Legend
          verticalAlign="bottom"
          height={40}
          wrapperStyle={{ fontSize: 14, paddingTop: 12, cursor: "pointer" }}
          onClick={onLegendClick}
          formatter={(value, entry) => {
            const key = String((entry as { dataKey?: unknown })?.dataKey ?? "");
            return <span style={{ opacity: isHidden(key) ? 0.4 : 1 }}>{value}</span>;
          }}
        />
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color}
            strokeWidth={2.5}
            dot={{ r: 4, fill: s.color, strokeWidth: 0 }}
            activeDot={{ r: 6 }}
            connectNulls={false}
            hide={isHidden(s.key)}
            isAnimationActive
            animationDuration={600}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
