import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

export function StatCard({
  label,
  value,
  caption,
  accent,
  className,
  children,
}: {
  label: string;
  value: string;
  caption?: string;
  accent?: "ia" | "ibc" | "ii" | "primary";
  className?: string;
  children?: React.ReactNode;
}) {
  const accentClass = accent
    ? {
        ia: "before:bg-ia",
        ibc: "before:bg-ibc",
        ii: "before:bg-ii",
        primary: "before:bg-primary",
      }[accent]
    : "before:bg-primary";

  return (
    <Card
      className={cn(
        "relative overflow-hidden py-6 before:absolute before:inset-x-0 before:top-0 before:h-1",
        accentClass,
        className,
      )}
    >
      <CardContent className="flex flex-col gap-2 px-6">
        <span className="text-caption">{label}</span>
        <span className="text-[32px] font-semibold leading-none tracking-tight tabular-nums-all">
          {value}
        </span>
        {caption ? <span className="text-caption">{caption}</span> : null}
        {children}
      </CardContent>
    </Card>
  );
}
