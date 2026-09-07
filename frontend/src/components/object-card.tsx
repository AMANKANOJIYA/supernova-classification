import { Card } from "@/components/ui/card";
import { ClassBadge } from "@/components/class-badge";
import { ObjectThumb } from "@/components/object-thumb";
import { CheckCircle2, XCircle } from "lucide-react";
import type { PredictionRow } from "@/lib/types";

export function ObjectCard({ row, onClick }: { row: PredictionRow; onClick: () => void }) {
  const confidence = Math.max(row.prob_ia, row.prob_ibc, row.prob_ii);
  return (
    <Card
      className="group cursor-pointer gap-0 overflow-hidden py-0 transition-colors hover:border-primary/50"
      onClick={onClick}
    >
      <div className="relative">
        <ObjectThumb objectId={row.object_id} />
        <div className="absolute right-2 top-2">
          {row.true_class_name ? (
            row.correct ? (
              <CheckCircle2 className="size-5 text-ii drop-shadow" />
            ) : (
              <XCircle className="size-5 text-destructive drop-shadow" />
            )
          ) : null}
        </div>
      </div>
      <div className="flex flex-col gap-2 p-4">
        <span className="truncate font-mono text-sm text-muted-foreground">{row.object_id}</span>
        <div className="flex items-center gap-2">
          <ClassBadge name={row.predicted_class} />
          {row.true_class_name ? (
            <>
              <span className="text-caption">true</span>
              <ClassBadge name={row.true_class_name} />
            </>
          ) : (
            <span className="text-caption">unlabeled</span>
          )}
        </div>
        <span className="text-caption tabular-nums-all">confidence {(confidence * 100).toFixed(1)}%</span>
      </div>
    </Card>
  );
}
