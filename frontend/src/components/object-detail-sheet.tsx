"use client";

import useSWR from "swr";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ClassBadge } from "@/components/class-badge";
import { LightCurveChart } from "@/components/charts/light-curve-chart";
import { CutoutViewer } from "@/components/cutout-viewer";
import { ProbabilityBars } from "@/components/probability-bars";
import { AnomalyBreakdown } from "@/components/anomaly-breakdown";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import { fmtMjd } from "@/lib/format";
import { CheckCircle2, XCircle } from "lucide-react";

// A takeover Dialog, not a side Sheet: this is the primary way people
// inspect one object (light curve + cutouts + probabilities + anomaly
// breakdown all at once), and a narrow side panel left most of that
// cramped into a sliver of the screen -- this needs the room.
export function ObjectDetailSheet({
  objectId,
  open,
  onOpenChange,
}: {
  objectId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: detail, error: detailError } = useSWR(
    objectId ? ["object-detail", objectId] : null,
    () => api.objectDetail(objectId as string),
    { shouldRetryOnError: false },
  );
  const { data: anomaly, error: anomalyError } = useSWR(
    objectId ? ["object-anomaly", objectId] : null,
    () => api.objectAnomaly(objectId as string),
    { shouldRetryOnError: false },
  );
  const noRollingState = detailError instanceof ApiError && detailError.status === 404;
  const { data: cutouts } = useSWR(
    objectId ? ["object-cutouts", objectId] : null,
    async () => {
      try {
        return await api.cutouts(objectId as string);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    { shouldRetryOnError: false },
  );

  const pred = detail?.latest_prediction;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton
        className="flex h-[92vh] max-h-[960px] w-[96vw] max-w-6xl flex-col gap-0 rounded-xl p-0 sm:max-w-6xl"
      >
        <DialogHeader className="shrink-0 border-b border-border px-6 py-5">
          <DialogTitle className="flex items-center gap-2 font-mono text-xl">
            {objectId}
          </DialogTitle>
          {/* Plain div, not DialogDescription: Radix renders that as a <p>,
              and the badges below render block content that's invalid
              inside <p> -- causes a hydration mismatch. */}
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            {detail ? (
              <>
                <span className="uppercase">{detail.survey}</span>
                <span>·</span>
                <span>predicted</span>
                <ClassBadge name={pred?.predicted_class} />
                {detail.true_class_name ? (
                  <>
                    <span>true</span>
                    <ClassBadge name={detail.true_class_name} />
                    {detail.correct ? (
                      <CheckCircle2 className="size-4 text-ii" />
                    ) : (
                      <XCircle className="size-4 text-destructive" />
                    )}
                  </>
                ) : (
                  <span>no ground truth</span>
                )}
              </>
            ) : (
              <Skeleton className="h-4 w-48" />
            )}
          </div>
        </DialogHeader>
        <ScrollArea className="min-h-0 flex-1">
          <div className="p-6">
            {noRollingState ? (
              <div className="rounded-lg border border-dashed border-border px-4 py-10 text-center">
                <p className="text-card-title mb-1">No live rolling-state history</p>
                <p className="text-caption mx-auto max-w-md">
                  Redis holds each object&apos;s accumulated light curve only while its rolling state is
                  populated (src/serve/state.py) — it&apos;s cleared on a container restart. This object&apos;s
                  past predictions still exist in Postgres, but its observation history isn&apos;t currently
                  loaded. Re-run <code className="text-foreground">make replay</code> +{" "}
                  <code className="text-foreground">make worker</code> to repopulate it.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
                <div className="flex flex-col gap-8 lg:col-span-2">
                  <section>
                    <h3 className="text-card-title mb-3">Light curve</h3>
                    {detail ? (
                      <>
                        <LightCurveChart observations={detail.observations} />
                        <p className="text-caption mt-1">
                          {detail.n_observations} observation(s) · latest alert{" "}
                          {pred ? fmtMjd(pred.alert_mjd) : "—"}
                        </p>
                      </>
                    ) : (
                      <Skeleton className="h-[340px] w-full" />
                    )}
                  </section>

                  <section>
                    <h3 className="text-card-title mb-3">Cutouts</h3>
                    {cutouts !== undefined ? (
                      <CutoutViewer images={cutouts?.images ?? null} />
                    ) : (
                      <Skeleton className="h-48 w-full" />
                    )}
                  </section>
                </div>

                <div className="flex flex-col gap-6">
                  <Card className="py-6">
                    <CardHeader>
                      <CardTitle className="text-card-title">Class probabilities</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {pred ? (
                        <ProbabilityBars probs={{ Ia: pred.prob_ia, "Ib/c": pred.prob_ibc, II: pred.prob_ii }} />
                      ) : (
                        <Skeleton className="h-24 w-full" />
                      )}
                    </CardContent>
                  </Card>
                  <Card className="py-6">
                    <CardHeader>
                      <CardTitle className="text-card-title">Anomaly breakdown</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {anomaly ? (
                        <AnomalyBreakdown combined={anomaly.combined} components={anomaly.components} />
                      ) : anomalyError ? (
                        <p className="text-caption">Unavailable for this object.</p>
                      ) : (
                        <Skeleton className="h-24 w-full" />
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
