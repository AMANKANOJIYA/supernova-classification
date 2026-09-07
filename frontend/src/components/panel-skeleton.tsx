import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export function CardSkeleton({ lines = 3, height }: { lines?: number; height?: number }) {
  return (
    <Card className="py-6">
      <CardHeader>
        <Skeleton className="h-4 w-32" />
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {height ? (
          <Skeleton style={{ height }} className="w-full" />
        ) : (
          Array.from({ length: lines }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" style={{ width: `${90 - i * 12}%` }} />
          ))
        )}
      </CardContent>
    </Card>
  );
}

export function ChartSkeleton({ height = 400 }: { height?: number }) {
  return (
    <Card className="py-6">
      <CardHeader className="flex-row items-center justify-between">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-4 w-24" />
      </CardHeader>
      <CardContent>
        <Skeleton className="w-full" style={{ height }} />
      </CardContent>
    </Card>
  );
}

export function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-2">
      <Skeleton className="h-9 w-full" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-11 w-full" />
      ))}
    </div>
  );
}

export function GridCardSkeleton() {
  return (
    <Card className="py-0 overflow-hidden">
      <Skeleton className="h-40 w-full rounded-none" />
      <CardContent className="flex flex-col gap-2 p-4">
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-4 w-1/3" />
      </CardContent>
    </Card>
  );
}
