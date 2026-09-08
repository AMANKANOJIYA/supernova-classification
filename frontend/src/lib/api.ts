import type {
  AlertsRateResponse,
  CutoutsResponse,
  HealthResponse,
  ObjectAnomalyResponse,
  ObjectDetailResponse,
  ObjectsResponse,
  RecentAlertsResponse,
  StatsResponse,
  TopAnomaliesResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, API_BASE);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} for ${url.pathname}`);
  }
  return res.json() as Promise<T>;
}

export { ApiError };

export const api = {
  health: () => getJson<HealthResponse>("/health"),
  stats: () => getJson<StatsResponse>("/stats"),
  objects: (params: {
    survey?: string;
    predicted_class?: string;
    true_class?: string;
    label?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) => getJson<ObjectsResponse>("/objects", params),
  objectDetail: (objectId: string) =>
    getJson<ObjectDetailResponse>(`/object/${encodeURIComponent(objectId)}`),
  objectAnomaly: (objectId: string) =>
    getJson<ObjectAnomalyResponse>(`/object/${encodeURIComponent(objectId)}/anomaly`),
  cutouts: (objectId: string) =>
    getJson<CutoutsResponse>(`/cutouts/${encodeURIComponent(objectId)}`),
  recentAlerts: (params: { limit?: number; offset?: number; survey?: string; predicted_class?: string }) =>
    getJson<RecentAlertsResponse>("/alerts/recent", params),
  alertsRate: (minutes = 30) => getJson<AlertsRateResponse>("/alerts/rate", { minutes }),
  topAnomalies: (limit = 50) => getJson<TopAnomaliesResponse>("/top-anomalies", { limit }),
};

// Static report JSONs (reports/, outputs/) mounted read-only by the API at
// /static/reports and /static/outputs (src/serve/api.py) -- fetched, never
// recomputed, per the task's own "presentation only" constraint.
export function staticReportUrl(relPath: string): string {
  return new URL(`/static/${relPath}`, API_BASE).toString();
}

export async function getStaticJson<T>(relPath: string): Promise<T> {
  const res = await fetch(staticReportUrl(relPath), { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} for ${relPath}`);
  }
  return res.json() as Promise<T>;
}
