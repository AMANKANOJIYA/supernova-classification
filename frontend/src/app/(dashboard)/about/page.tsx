import Image from "next/image";
import { PageHeader, SectionHeading } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LightCurveGlyph } from "@/components/light-curve-glyph";
import { PipelineDiagram } from "@/components/pipeline-diagram";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ClassBadge } from "@/components/class-badge";

const SN_TYPES = [
  {
    name: "Ia" as const,
    title: "Type Ia",
    subtitle: "Thermonuclear white dwarf",
    image: "/images/sn-ia-kepler.jpg",
    imageAlt: "Kepler's Supernova Remnant, a Type Ia remnant",
    credit: "NASA, ESA, R. Sankrit & W. Blair (Johns Hopkins University)",
    description:
      "A white dwarf accreting past the Chandrasekhar limit (or merging with a companion) detonates completely. Reliably bright standard candles — the light curve shows a fast rise and a smooth, slightly asymmetric decline.",
  },
  {
    name: "Ib/c" as const,
    title: "Type Ib/c",
    subtitle: "Stripped-envelope core collapse",
    image: "/images/sn-ibc-wr124.png",
    imageAlt: "Wolf-Rayet star WR 124, a stripped-envelope progenitor analog",
    credit: "NASA, ESA, CSA, STScI, Webb ERO Production Team",
    description:
      "A massive star that has shed its outer hydrogen (Ib) and helium (Ic) layers — often via Wolf-Rayet winds, like WR 124 pictured here — collapses. Merged into one class in this taxonomy since Ib/Ic separation needs spectroscopy. Fast, irregular light curves with no plateau; recall in the 0.40–0.60 band is expected, not a bug.",
  },
  {
    name: "II" as const,
    title: "Type II",
    subtitle: "Hydrogen-rich core collapse",
    image: "/images/sn-ii-1987a.png",
    imageAlt: "Supernova 1987A remnant in the Large Magellanic Cloud",
    credit: "NASA, ESA, A. Angelich (NRAO/AUI/NSF), R. Kirshner (CfA)",
    description:
      "A massive star with its hydrogen envelope intact collapses — SN 1987A, pictured here, is the closest and best-studied example. The retained hydrogen envelope produces the class's signature: a distinctive plateau as recombination temporarily halts the decline.",
  },
];

const DATA_TIERS = [
  {
    tier: "A",
    name: "ELAsTiCC2 training sample",
    role: "Pretraining",
    bands: "ugrizy (LSST)",
    cutouts: false,
    labels: "Simulated",
    detail:
      "SNANA-simulated LSST light curves, 12 SN model directories (843,617 objects). No pixel data by construction — photometry-level simulation only.",
  },
  {
    tier: "B",
    name: "BTSbot v10",
    role: "Fine-tuning",
    bands: "ztf_g, ztf_r",
    cutouts: true,
    labels: "Spectroscopic",
    detail:
      "Real ZTF public-stream alerts with spectroscopic labels from the Bright Transient Survey. Pre-filtered to brightening subtractions — 0% negative flux, a cleaner slice than a live stream.",
  },
  {
    tier: "C",
    name: "Fink livestream",
    role: "Inference only",
    bands: "varies (ZTF + LSST)",
    cutouts: true,
    labels: "None",
    detail:
      "Live ZTF and LSST alerts via Fink's own SN-candidate-filtered topics. LSST relay confirmed reachable but showed no traffic during this session's observation windows.",
  },
];

const UNRESOLVED = [
  {
    title: "Tier B static-feature inversion has no confirmed cause",
    body: "LC+static fine-tuning underperforms LC-only on real ZTF (0.634 vs 0.672 macro-F1) — the opposite of Tier A. A photo-z ablation refuted the leading hypothesis (standard-candle distance information): only 33% of Tier A's static gain survives removing photo-z, not the 'most of it' originally guessed. Small-sample overfitting (5,407 labeled training objects, 3 of 7 static fields permanently missing at Tier B) remains plausible but unverified.",
  },
  {
    title: "WHXE hierarchical loss: provisional, not rejected",
    body: "A flat 3-class head currently beats WHXE (0.698 vs 0.682 macro-F1) on a dataset where every object is SN-like — WHXE's whole justification (graceful degradation under real ambiguity) is untestable until a non-SN-like branch exists. Revisit once the anomaly branch's held-out exotics are used for retraining, not this decision made from these numbers alone.",
  },
  {
    title: "class_weighting=true, left unvalidated",
    body: "Enabled without meeting its own stated precondition (Ia underperforming) — Ib/c was actually the worst class at the time. No controlled ablation was run (a rerun costs ~2.5h). Documented as unvalidated rather than silently assumed fine.",
  },
  {
    title: "LSST relay: reachable, but no traffic observed",
    body: "Every subscribed Fink LSST topic showed real historical throughput (hundreds of thousands of cumulative messages) but zero currently-queued messages across two observation windows. avro_lsst.py remains unverified against real traffic — it fails loudly by name on any schema mismatch, which is the right place to leave it.",
  },
  {
    title: "Gate D's real question is still unanswered",
    body: "The anomaly ranker's top-50 inspection used an eval set (BTS 'weird' categories) containing zero bogus/artifact candidates by construction — so it could only confirm the ranker finds genuine astrophysical outliers, never test whether real-bogus filtering suppresses real signal. That needs an unfiltered Tier-C stream, not built yet.",
  },
  {
    title: "Anomaly scorer persistence gap (now closed) and its lesson",
    body: "Mahalanobis/isoforest scorers were originally only ever fit in-process and never serialized — 'saved the numbers' and 'saved the model' turned out to be different claims the moment scoring needed to survive a server restart. Fixed by persisting the fitted objects (outputs/anomaly/mahalanobis.npz, forest.pkl); flagged here as a pattern worth watching for in any future stateful-scorer addition.",
  },
];

export default function AboutPage() {
  return (
    <div className="flex flex-col gap-12">
      <PageHeader
        title="About"
        description="What this pipeline classifies, how the data is organized, and what's still open."
      />

      <section>
        <SectionHeading title="The three target classes" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {SN_TYPES.map((t) => (
            <Card key={t.name} className="gap-0 overflow-hidden py-0">
              <div className="relative h-48 w-full bg-black">
                <Image src={t.image} alt={t.imageAlt} fill className="object-cover" sizes="(max-width: 1024px) 100vw, 33vw" />
              </div>
              <CardHeader className="pt-5">
                <div className="flex items-center gap-2">
                  <CardTitle className="text-card-title">{t.title}</CardTitle>
                  <ClassBadge name={t.name} />
                </div>
                <p className="text-caption">{t.subtitle}</p>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <LightCurveGlyph shape={t.name} className="h-16 w-full text-muted-foreground" />
                <p className="text-body text-muted-foreground">{t.description}</p>
                <p className="text-caption">Image: {t.credit}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <SectionHeading
          title="Pipeline"
          description="Each step exists to make the next step's errors visible — the build order is not arbitrary (CLAUDE.md §6)."
        />
        <Card className="py-6">
          <CardContent>
            <PipelineDiagram />
          </CardContent>
        </Card>
      </section>

      <section>
        <SectionHeading title="Data tiers" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {DATA_TIERS.map((t) => (
            <Card key={t.tier} className="py-6">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <span className="flex size-7 items-center justify-center rounded-md bg-primary/15 text-sm font-semibold text-primary">
                    {t.tier}
                  </span>
                  <CardTitle className="text-card-title">{t.name}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
                  <dt className="text-muted-foreground">Role</dt>
                  <dd className="text-right">{t.role}</dd>
                  <dt className="text-muted-foreground">Bands</dt>
                  <dd className="text-right">{t.bands}</dd>
                  <dt className="text-muted-foreground">Cutouts</dt>
                  <dd className="text-right">{t.cutouts ? "yes" : "no"}</dd>
                  <dt className="text-muted-foreground">Labels</dt>
                  <dd className="text-right">{t.labels}</dd>
                </dl>
                <p className="text-caption">{t.detail}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <SectionHeading
          title="Open questions"
          description="Recorded rather than smoothed over — collapsed by default."
        />
        <Card className="py-2">
          <CardContent className="px-2">
            <Accordion type="single" collapsible className="w-full">
              {UNRESOLVED.map((u, i) => (
                <AccordionItem key={i} value={`item-${i}`}>
                  <AccordionTrigger className="px-4 text-base">{u.title}</AccordionTrigger>
                  <AccordionContent className="px-4 text-body text-muted-foreground">
                    {u.body}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
