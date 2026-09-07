import { EmptyState } from "@/components/empty-state";
import { ImageOff } from "lucide-react";

const LABELS: Record<string, string> = {
  science: "Science",
  template: "Template",
  difference: "Difference",
};

export function CutoutViewer({ images }: { images: { science: string; template: string; difference: string } | null }) {
  if (!images) {
    return (
      <EmptyState
        icon={ImageOff}
        title="No cutouts available"
        description="ELAsTiCC2 (Tier A) never has pixel cutouts by construction -- it's a photometry-level simulation. Real ZTF/Fink objects (Tier B/C) should have a science/template/difference triplet here."
        className="py-10"
      />
    );
  }
  return (
    <div className="grid grid-cols-3 gap-3">
      {(["science", "template", "difference"] as const).map((key) => (
        <figure key={key} className="flex flex-col gap-1.5">
          <div className="overflow-hidden rounded-lg border border-border bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${images[key]}`}
              alt={`${LABELS[key]} cutout`}
              className="aspect-square w-full [image-rendering:pixelated]"
            />
          </div>
          <figcaption className="text-center text-caption">{LABELS[key]}</figcaption>
        </figure>
      ))}
    </div>
  );
}
