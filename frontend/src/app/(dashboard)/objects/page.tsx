"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Search as SearchIcon, Orbit } from "lucide-react";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/page-header";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { ObjectCard } from "@/components/object-card";
import { ObjectDetailSheet } from "@/components/object-detail-sheet";
import { GridCardSkeleton } from "@/components/panel-skeleton";
import { EmptyState } from "@/components/empty-state";

const PAGE_SIZE = 24;

export default function ObjectsPage() {
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [trueClass, setTrueClass] = useState("all");
  const [predictedClass, setPredictedClass] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      setDebounced(search);
      setPage(0);
    }, 300);
    return () => clearTimeout(t);
  }, [search]);

  function selectFilter(setter: (v: string) => void) {
    return (v: string) => {
      setter(v);
      setPage(0);
    };
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const { data } = useSWR(
    ["objects", debounced, trueClass, predictedClass, labelFilter, page],
    () =>
      api.objects({
        search: debounced || undefined,
        true_class: trueClass === "all" ? undefined : trueClass,
        predicted_class: predictedClass === "all" ? undefined : predictedClass,
        label: labelFilter === "all" ? undefined : labelFilter,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    { keepPreviousData: true },
  );

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Objects"
        description="Every BTSbot (Tier B / real ZTF) and live-Fink object scored so far. Browse, filter, and open any object for its full light curve, cutouts, and anomaly breakdown."
      />

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative w-64">
          <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            ref={searchRef}
            placeholder="Search object ID…  (press /)"
            className="pl-8"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Select value={trueClass} onValueChange={selectFilter(setTrueClass)}>
          <SelectTrigger className="w-40"><SelectValue placeholder="True class" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Any true class</SelectItem>
            <SelectItem value="Ia">Ia</SelectItem>
            <SelectItem value="Ib/c">Ib/c</SelectItem>
            <SelectItem value="II">II</SelectItem>
          </SelectContent>
        </Select>
        <Select value={predictedClass} onValueChange={selectFilter(setPredictedClass)}>
          <SelectTrigger className="w-44"><SelectValue placeholder="Predicted class" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Any predicted class</SelectItem>
            <SelectItem value="Ia">Ia</SelectItem>
            <SelectItem value="Ib/c">Ib/c</SelectItem>
            <SelectItem value="II">II</SelectItem>
          </SelectContent>
        </Select>
        <Select value={labelFilter} onValueChange={selectFilter(setLabelFilter)}>
          <SelectTrigger className="w-40"><SelectValue placeholder="Label" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Any label</SelectItem>
            <SelectItem value="correct">Correct</SelectItem>
            <SelectItem value="incorrect">Incorrect</SelectItem>
            <SelectItem value="labeled">Labeled</SelectItem>
            <SelectItem value="unlabeled">Unlabeled</SelectItem>
          </SelectContent>
        </Select>
        {data ? <span className="text-caption ml-auto">{data.total.toLocaleString()} object(s)</span> : null}
      </div>

      {!data ? (
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
          {Array.from({ length: 12 }).map((_, i) => (
            <GridCardSkeleton key={i} />
          ))}
        </div>
      ) : data.results.length === 0 ? (
        <EmptyState
          icon={Orbit}
          title="No objects match these filters"
          description="Try loosening the class or label filter, or clearing the search."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
            {data.results.map((row) => (
              <ObjectCard key={row.id} row={row} onClick={() => setSelected(row.object_id)} />
            ))}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-caption">
              Page {page + 1} of {totalPages}
            </span>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}

      <ObjectDetailSheet objectId={selected} open={!!selected} onOpenChange={(o) => !o && setSelected(null)} />
    </div>
  );
}
