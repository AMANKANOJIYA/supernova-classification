const STEPS = [
  { n: 1, title: "Schema + eval", detail: "Frozen contract & metrics, before any model exists" },
  { n: 2, title: "Dumb baselines", detail: "The number every later model must beat" },
  { n: 3, title: "Readers", detail: "ELAsTiCC2, BTSbot, Fink → CanonicalAlert" },
  { n: 4, title: "Tensorize", detail: "Padded, causal, wavelength-conditioned tensors" },
  { n: 5, title: "LC encoder", detail: "Causal GRU + flat 3-class head, then WHXE" },
  { n: 6, title: "Static + fine-tune", detail: "Host features, then real ZTF (Tier B)" },
  { n: 7, title: "Anomaly branch", detail: "Reconstruction + Mahalanobis + isoforest" },
  { n: 8, title: "Serving + UI", detail: "Replay → live Fink, anomaly review loop" },
];

export function PipelineDiagram() {
  return (
    <div className="scrollbar-thin overflow-x-auto pb-2">
      <div className="flex min-w-[900px] items-start gap-0">
        {STEPS.map((s, i) => (
          <div key={s.n} className="flex flex-1 items-start">
            <div className="flex flex-col items-center gap-2 px-2 text-center">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-full border-2 border-primary bg-primary/10 text-sm font-semibold tabular-nums-all">
                {s.n}
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-sm font-medium leading-tight">{s.title}</span>
                <span className="text-caption max-w-[9rem] leading-snug">{s.detail}</span>
              </div>
            </div>
            {i < STEPS.length - 1 ? (
              <div className="mt-4.5 h-0.5 flex-1 bg-gradient-to-r from-primary/50 to-primary/10" />
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}
