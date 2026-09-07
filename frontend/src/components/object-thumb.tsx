"use client";

import useSWR from "swr";
import { ImageOff } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError } from "@/lib/api";

export function ObjectThumb({ objectId }: { objectId: string }) {
  const { data, error } = useSWR(
    ["thumb", objectId],
    async () => {
      try {
        return await api.cutouts(objectId);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    { shouldRetryOnError: false, revalidateOnFocus: false },
  );

  if (data === undefined && !error) {
    return <Skeleton className="aspect-square w-full rounded-none" />;
  }
  if (!data) {
    return (
      <div className="flex aspect-square w-full items-center justify-center bg-muted/60">
        <ImageOff className="size-6 text-muted-foreground/50" />
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`data:image/png;base64,${data.images.science}`}
      alt=""
      className="aspect-square w-full bg-black object-cover [image-rendering:pixelated]"
    />
  );
}
