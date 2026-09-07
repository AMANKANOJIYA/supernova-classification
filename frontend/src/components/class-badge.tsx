import { cn } from "@/lib/utils";
import { CLASS_BG_CLASS } from "@/lib/colors";
import type { ClassName } from "@/lib/types";

export function ClassBadge({
  name,
  className,
}: {
  name: ClassName | string | null | undefined;
  className?: string;
}) {
  if (!name) {
    return (
      <span
        className={cn(
          "inline-flex items-center rounded-md border border-border px-2 py-0.5 text-sm font-medium text-muted-foreground",
          className,
        )}
      >
        unlabeled
      </span>
    );
  }
  const cls = CLASS_BG_CLASS[name as ClassName] ?? "bg-muted text-muted-foreground border-border";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-sm font-medium tabular-nums-all",
        cls,
        className,
      )}
    >
      {name}
    </span>
  );
}
