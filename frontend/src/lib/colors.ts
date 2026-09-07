import type { ClassName } from "./types";

// Semantic class colors -- single source of truth. Mirrors the CSS tokens in
// globals.css (--ia / --ibc / --ii). Recharts needs literal color strings
// (SVG fill/stroke), so these are the resolved hex values of the same hues,
// not CSS vars -- keep the two in sync if the palette ever changes.
export const CLASS_HEX: Record<ClassName, string> = {
  Ia: "#eab86b", // amber
  "Ib/c": "#5b9bf0", // blue
  II: "#4fc79a", // emerald
};

export const CLASS_HEX_STRONG: Record<ClassName, string> = {
  Ia: "#f0b03e",
  "Ib/c": "#3b82f6",
  II: "#22c99a",
};

export const CLASS_BG_CLASS: Record<ClassName, string> = {
  Ia: "bg-ia/15 text-ia border-ia/30",
  "Ib/c": "bg-ibc/15 text-ibc border-ibc/30",
  II: "bg-ii/15 text-ii border-ii/30",
};

export const ACCENT_HEX = "#8b6cf0";
export const NEUTRAL_GRID = "rgba(255,255,255,0.08)";
export const AXIS_TEXT = "rgba(228,228,231,0.65)";

export function classHex(name: string | null | undefined): string {
  if (name === "Ia" || name === "Ib/c" || name === "II") return CLASS_HEX[name];
  return "#71717a";
}

// Sequential single-hue ramp for the confusion-matrix heatmap: dark card
// background -> accent, real lightness range (not saturated blue-on-blue).
export function heatmapColor(value: number): string {
  const t = Math.max(0, Math.min(1, value));
  // Interpolate in a simple RGB mix between the dark card tone and the accent.
  const from = { r: 35, g: 33, b: 46 }; // ~--card-strong
  const to = { r: 139, g: 108, b: 240 }; // accent
  const r = Math.round(from.r + (to.r - from.r) * t);
  const g = Math.round(from.g + (to.g - from.g) * t);
  const b = Math.round(from.b + (to.b - from.b) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function heatmapTextColor(value: number): string {
  return value > 0.55 ? "#0e0b17" : "rgba(244,244,245,0.92)";
}

// LAM_EFF (schema.py / CLAUDE.md §3), nanometres -- for labeling light-curve
// points by band. Continuous wavelength conditioning is the whole point of
// the schema (u/g/r/i/z/y + ztf_g/r/i share one axis), so we label the
// nearest named band rather than plotting a bare "filter index."
const NAMED_BANDS: Array<{ name: string; nm: number }> = [
  { name: "u", nm: 368.6 },
  { name: "ztf_g", nm: 472.3 },
  { name: "g", nm: 480.3 },
  { name: "r", nm: 622.2 },
  { name: "ztf_r", nm: 633.9 },
  { name: "i", nm: 754.5 },
  { name: "ztf_i", nm: 788.6 },
  { name: "z", nm: 869.1 },
  { name: "y", nm: 971.0 },
];

export function nearestBandName(lamEff: number): string {
  let best = NAMED_BANDS[0];
  let bestDiff = Infinity;
  for (const b of NAMED_BANDS) {
    const diff = Math.abs(b.nm - lamEff);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = b;
    }
  }
  return bestDiff <= 15 ? best.name : `${lamEff.toFixed(0)}nm`;
}

// Continuous wavelength -> perceptual color, ~360-1000nm (near-UV through
// near-IR), so a light curve's band axis reads visually the way the schema
// treats it: continuous, not categorical.
export function wavelengthToColor(nm: number): string {
  const stops: Array<[number, [number, number, number]]> = [
    [360, [129, 87, 235]], // violet/u
    [480, [91, 155, 240]], // blue/g
    [630, [79, 199, 154]], // green/r
    [790, [234, 184, 107]], // amber/i
    [870, [230, 126, 84]], // orange/z
    [970, [204, 84, 92]], // red/y
  ];
  if (nm <= stops[0][0]) return `rgb(${stops[0][1].join(",")})`;
  if (nm >= stops[stops.length - 1][0]) return `rgb(${stops[stops.length - 1][1].join(",")})`;
  for (let i = 0; i < stops.length - 1; i++) {
    const [n0, c0] = stops[i];
    const [n1, c1] = stops[i + 1];
    if (nm >= n0 && nm <= n1) {
      const t = (nm - n0) / (n1 - n0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(160,160,170)";
}
