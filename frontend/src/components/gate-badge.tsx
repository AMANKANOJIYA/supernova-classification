import { cn } from "@/lib/utils";
import { CheckCircle2, CircleSlash, HelpCircle } from "lucide-react";

export type GateStatus = "pass" | "fail" | "deferred";

const STYLES: Record<GateStatus, string> = {
  pass: "bg-ii/15 text-ii border-ii/30",
  fail: "bg-destructive/15 text-destructive border-destructive/30",
  deferred: "bg-muted text-muted-foreground border-border",
};

const ICONS: Record<GateStatus, typeof CheckCircle2> = {
  pass: CheckCircle2,
  fail: CircleSlash,
  deferred: HelpCircle,
};

export function GateBadge({
  label,
  status,
  detail,
}: {
  label: string;
  status: GateStatus;
  detail?: string;
}) {
  const Icon = ICONS[status];
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-lg border px-4 py-3 min-w-[11rem]",
        STYLES[status],
      )}
      title={detail}
    >
      <div className="flex items-center gap-1.5 text-sm font-medium">
        <Icon className="size-4 shrink-0" />
        <span>{label}</span>
      </div>
      {detail ? <span className="text-caption leading-snug">{detail}</span> : null}
    </div>
  );
}
