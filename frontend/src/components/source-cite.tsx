import { cn } from "@/lib/utils";

export function SourceCite({ path, className }: { path: string; className?: string }) {
  return (
    <code
      className={cn(
        "text-sm leading-snug text-muted-foreground/70 font-mono break-all",
        className,
      )}
    >
      {path}
    </code>
  );
}
