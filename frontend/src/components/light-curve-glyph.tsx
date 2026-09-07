import { CLASS_HEX_STRONG } from "@/lib/colors";
import type { ClassName } from "@/lib/types";

// Small illustrative sparklines of each class's characteristic light-curve
// shape -- schematic, not real data (real examples live on Objects/Results).
const PATHS: Record<ClassName, string> = {
  // Ia: fast rise, smooth asymmetric decline (slower than the rise).
  Ia: "M2,58 C 18,56 28,20 38,8 C 46,-2 52,4 58,16 C 68,34 78,46 118,52",
  // Ib/c: fast rise, fast and irregular/jagged decline.
  "Ib/c": "M2,58 C 14,50 22,14 34,8 C 40,20 44,26 50,24 C 56,22 58,34 64,32 C 72,30 70,44 78,42 C 90,40 100,48 118,54",
  // II: rise to a plateau, hold, then drop off.
  II: "M2,58 C 14,40 22,14 34,10 L 74,14 C 86,16 92,26 98,42 C 102,52 108,56 118,58",
};

export function LightCurveGlyph({ shape, className }: { shape: ClassName; className?: string }) {
  const color = CLASS_HEX_STRONG[shape];
  return (
    <svg viewBox="0 0 120 64" className={className} aria-hidden>
      <line x1="0" y1="58" x2="120" y2="58" stroke="currentColor" strokeOpacity={0.15} strokeWidth={1} />
      <path d={PATHS[shape]} fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" />
    </svg>
  );
}
