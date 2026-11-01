import type { components } from "./schema";

type Schemas = components["schemas"];
export type IncidentSummary = Schemas["IncidentSummary"];
export type IncidentPage = Schemas["IncidentPage"];
export type IncidentDetail = Schemas["IncidentDetail"];
export type IncidentCreate = Schemas["IncidentCreate"];
export type IncidentGraph = Schemas["IncidentGraph"];
export type IncidentTimeline = Schemas["IncidentTimeline"];
export type IncidentCoverage = Schemas["IncidentCoverage"];
export type ServiceEvidence = Schemas["ServiceEvidence"];
export type WhatIfRequest = Schemas["WhatIfRequest"];
export type WhatIfResult = Schemas["WhatIfResult"];
export type FeedbackCreate = Schemas["FeedbackCreate"];
export type FeedbackLabel = Schemas["FeedbackLabel"];
export type HistoryReport = Schemas["HistoryReport"];
export type ModelReport = Schemas["ModelReport"];
export type SettingsUpdate = Schemas["SettingsUpdate"];
export type SettingsView = Schemas["SettingsView"];
export type IntegrationList = Schemas["IntegrationList"];

const BASE = "/api/v1";
const MESSAGES: Record<string, string> = {
  network_error: "The service could not be reached. Check that it is running.",
  incident_not_found: "This incident does not exist.",
  service_not_found: "That service is not part of this incident.",
  whatif_busy: "What-if is busy. Try again shortly.",
  whatif_timeout: "The scenario took too long and was stopped.",
  whatif_rate_limited: "Too many what-if runs. Wait a minute and try again.",
  body_too_large: "That case is larger than the service accepts.",
  internal_error: "The service could not complete that request.",
};

/** A failed request, with a message that is safe to show. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    serverMessage: string,
  ) {
    super(MESSAGES[code] ?? serverMessage);
    this.name = "ApiError";
  }
}

async function failure(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as Schemas["ErrorResponse"];
    return new ApiError(response.status, body.error.code, body.error.message);
  } catch {
    return new ApiError(response.status, "internal_error", "");
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "network_error", "");
  }
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

function send<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method,
    signal,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export interface IncidentQuery {
  status?: string;
  severity?: string;
  service?: string;
  since?: number;
  limit?: number;
  cursor?: string;
}

function queryString(values: object): string {
  const entries = Object.entries(values).filter(([, value]) => value !== undefined && value !== "");
  const text = new URLSearchParams(entries.map(([key, value]) => [key, String(value)])).toString();
  return text ? `?${text}` : "";
}

const incident = (id: string) => `/incidents/${encodeURIComponent(id)}`;

export const api = {
  incidents: (query: IncidentQuery = {}, signal?: AbortSignal) =>
    request<IncidentPage>(`/incidents${queryString(query)}`, { signal }),
  incident: (id: string, signal?: AbortSignal) => request<IncidentDetail>(incident(id), { signal }),
  createIncident: (body: IncidentCreate) => send<IncidentDetail>("POST", "/incidents", body),
  rerun: (id: string) => send<IncidentDetail>("POST", `${incident(id)}/rerun`),
  graph: (id: string, signal?: AbortSignal) =>
    request<IncidentGraph>(`${incident(id)}/graph`, { signal }),
  timeline: (id: string, signal?: AbortSignal) =>
    request<IncidentTimeline>(`${incident(id)}/timeline`, { signal }),
  coverage: (id: string, signal?: AbortSignal) =>
    request<IncidentCoverage>(`${incident(id)}/coverage`, { signal }),
  evidence: (id: string, service: string, signal?: AbortSignal) =>
    request<ServiceEvidence>(`${incident(id)}/evidence/${encodeURIComponent(service)}`, { signal }),
  whatIf: (id: string, body: WhatIfRequest, signal?: AbortSignal) =>
    send<WhatIfResult>("POST", `${incident(id)}/whatif`, body, signal),
  feedback: (id: string, body: FeedbackCreate) =>
    send<FeedbackLabel>("POST", `${incident(id)}/feedback`, body),
  history: (since?: number, signal?: AbortSignal) =>
    request<HistoryReport>(`/history${queryString({ since })}`, { signal }),
  model: (signal?: AbortSignal) => request<ModelReport>("/model", { signal }),
  settings: (signal?: AbortSignal) => request<SettingsView>("/settings", { signal }),
  saveSettings: (body: SettingsUpdate) => send<SettingsView>("PUT", "/settings", body),
  integrations: (signal?: AbortSignal) => request<IntegrationList>("/integrations", { signal }),
};

export const queryKeys = {
  incidents: (query: IncidentQuery = {}) => ["incidents", query] as const,
  incident: (id: string) => ["incident", id] as const,
  graph: (id: string) => ["incident", id, "graph"] as const,
  timeline: (id: string) => ["incident", id, "timeline"] as const,
  coverage: (id: string) => ["incident", id, "coverage"] as const,
  evidence: (id: string, service: string) => ["incident", id, "evidence", service] as const,
  history: ["history"] as const,
  model: ["model"] as const,
  settings: ["settings"] as const,
  integrations: ["integrations"] as const,
};
