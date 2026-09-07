const LABELS: Record<string, string> = {
  reconstruction_error: "Reconstruction error",
  mahalanobis: "Mahalanobis distance",
  isoforest: "Isolation forest",
  entropy: "Classifier entropy",
};

const CLAMP = 4;

export function AnomalyBreakdown({
  combined,
  components,
}: {
  combined: number;
  components: Record<string, number>;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-caption">Combined (z-score avg.)</span>
        <span className="text-card-title tabular-nums-all font-semibold">{combined.toFixed(2)}</span>
      </div>
      <div className="flex flex-col gap-2.5">
        {Object.entries(components).map(([key, value]) => {
          const t = Math.max(-CLAMP, Math.min(CLAMP, value)) / CLAMP; // -1..1
          const widthPct = Math.abs(t) * 50;
          const isPositive = t >= 0;
          return (
            <div key={key} className="flex items-center gap-3">
              <span className="w-40 shrink-0 text-sm text-muted-foreground">{LABELS[key] ?? key}</span>
              <div className="relative h-2 flex-1 rounded-full bg-muted">
                <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                <div
                  className="absolute inset-y-0 rounded-full bg-primary"
                  style={
                    isPositive
                      ? { left: "50%", width: `${widthPct}%` }
                      : { right: "50%", width: `${widthPct}%` }
                  }
                />
              </div>
              <span className="w-12 shrink-0 text-right text-sm tabular-nums-all">{value.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
      <p className="text-caption">
        Higher = more anomalous, in z-scored units against the fit population. Positive and negative both occur
        by construction.
      </p>
    </div>
  );
}
