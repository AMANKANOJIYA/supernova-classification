import { CLASS_NAMES } from "@/lib/types";
import { CLASS_HEX_STRONG } from "@/lib/colors";
import { fmtPercent } from "@/lib/format";

export function ProbabilityBars({
  probs,
}: {
  probs: { Ia: number; "Ib/c": number; II: number };
}) {
  return (
    <div className="flex flex-col gap-2.5">
      {CLASS_NAMES.map((name) => {
        const p = probs[name] ?? 0;
        return (
          <div key={name} className="flex items-center gap-3">
            <span className="w-12 shrink-0 text-sm font-medium">{name}</span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full transition-[width] duration-300"
                style={{ width: `${Math.max(2, p * 100)}%`, background: CLASS_HEX_STRONG[name] }}
              />
            </div>
            <span className="w-14 shrink-0 text-right text-sm tabular-nums-all text-muted-foreground">
              {fmtPercent(p, 1)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
