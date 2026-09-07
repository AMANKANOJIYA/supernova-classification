"use client";

import { useState, useCallback } from "react";

// Recharts' <Legend> renders labels but doesn't hide series on click by
// itself -- this wires that up: click a legend entry to toggle its series,
// click again to bring it back. `dataKey` is what Recharts' legend payload
// carries per entry (matches each <Line>/<Bar>'s own `dataKey`).
export function useLegendToggle() {
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const onLegendClick = useCallback((entry: { dataKey?: unknown }) => {
    const key = String(entry.dataKey ?? "");
    if (!key) return;
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const isHidden = useCallback((key: string) => hidden.has(key), [hidden]);

  return { onLegendClick, isHidden };
}
