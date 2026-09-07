"use client";

import useSWR from "swr";
import { getStaticJson } from "@/lib/api";

export function useStaticJson<T>(relPath: string) {
  return useSWR<T>(["static", relPath], () => getStaticJson<T>(relPath), {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
}
