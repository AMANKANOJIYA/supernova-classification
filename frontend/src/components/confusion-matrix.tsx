"use client";

import { CLASS_NAMES } from "@/lib/types";
import type { ConfusionBlock } from "@/lib/types";
import { heatmapColor, heatmapTextColor } from "@/lib/colors";
import { cn } from "@/lib/utils";

export function ConfusionMatrix({
  confusion,
  mode,
}: {
  confusion: ConfusionBlock;
  mode: "normalized" | "raw";
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1.5">
        <thead>
          <tr>
            <th className="w-28" />
            <th colSpan={3} className="pb-1 text-center text-caption font-normal">
              predicted
            </th>
          </tr>
          <tr>
            <th className="text-right text-caption font-normal pr-2">true \ pred</th>
            {CLASS_NAMES.map((c) => (
              <th key={c} className="w-24 pb-1 text-center text-sm font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CLASS_NAMES.map((rowName, i) => (
            <tr key={rowName}>
              <th className="whitespace-nowrap pr-2 text-right text-sm font-medium">{rowName}</th>
              {CLASS_NAMES.map((_, j) => {
                const norm = confusion.normalized[i][j];
                const raw = confusion.raw[i][j];
                const bg = heatmapColor(norm);
                const fg = heatmapTextColor(norm);
                const isDiagonal = i === j;
                return (
                  <td
                    key={j}
                    className={cn(
                      "h-20 w-24 rounded-md text-center align-middle transition-colors duration-300",
                      isDiagonal && "ring-1 ring-inset ring-primary/40",
                    )}
                    style={{ background: bg, color: fg }}
                  >
                    <div className="flex flex-col items-center justify-center gap-0.5">
                      <span className="text-lg font-semibold tabular-nums-all">
                        {mode === "normalized" ? `${(norm * 100).toFixed(1)}%` : raw.toLocaleString()}
                      </span>
                      <span className="text-sm tabular-nums-all opacity-70">
                        {mode === "normalized" ? raw.toLocaleString() : `${(norm * 100).toFixed(1)}%`}
                      </span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-caption mt-3">
        Rows sum to 1 when row-normalized, so the diagonal reads directly as recall. Ring highlight marks the
        diagonal (correct predictions).
      </p>
    </div>
  );
}
