import { Card } from "@/components/ui/card";

export interface TooltipRow {
  label: string;
  value: string;
  color: string;
}

export function ChartTooltipCard({
  title,
  rows,
}: {
  title?: string;
  rows: TooltipRow[];
}) {
  return (
    <Card className="gap-1 rounded-md border-border/80 bg-popover px-3 py-2.5 shadow-lg">
      {title ? <div className="text-caption font-medium text-foreground">{title}</div> : null}
      <div className="flex flex-col gap-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-sm">
            <span className="inline-block size-2 shrink-0 rounded-full" style={{ background: r.color }} />
            <span className="text-muted-foreground">{r.label}</span>
            <span className="ml-auto pl-3 font-medium tabular-nums-all">{r.value}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// Generic Recharts-compatible tooltip renderer. `formatRows` maps the
// payload Recharts hands us into display rows.
export function makeChartTooltip(
  formatRows: (
    payload: readonly unknown[],
    label?: string | number,
  ) => { title?: string; rows: TooltipRow[] } | null,
) {
  function TooltipContent(props: { active?: boolean; payload?: readonly unknown[]; label?: string | number }) {
    if (!props.active || !props.payload || props.payload.length === 0) return null;
    const content = formatRows(props.payload, props.label);
    if (!content) return null;
    return <ChartTooltipCard title={content.title} rows={content.rows} />;
  }
  return TooltipContent;
}
