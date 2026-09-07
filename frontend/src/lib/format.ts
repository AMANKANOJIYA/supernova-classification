export function fmtNumber(n: number, digits = 2): string {
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPercent(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtMs(n: number): string {
  return `${n.toFixed(1)} ms`;
}

export function fmtCompact(n: number): string {
  return n.toLocaleString("en-US");
}

export function fmtMjd(mjd: number): string {
  // MJD -> UTC calendar date. Epoch: MJD 0 = 1858-11-17.
  const ms = (mjd - 40587) * 86400 * 1000;
  return new Date(ms).toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export function relTime(iso: string): string {
  const d = new Date(iso).getTime();
  const diffSec = Math.round((Date.now() - d) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.round(diffSec / 3600)}h ago`;
  return `${Math.round(diffSec / 86400)}d ago`;
}
