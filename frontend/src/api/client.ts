import type {
  HealthResponse,
  KnowledgeBase,
  KnowledgeBaseCreateRequest,
  KnowledgeBaseListResponse,
  KnowledgeBaseUpdateRequest,
  MetaResponse,
} from "./types";

const DEFAULT_API_BASE_URL = "/api";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(
  /\/$/,
  "",
);

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }

  return (await response.json()) as T;
}

async function requestNoContent(path: string, options: RequestInit = {}): Promise<void> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/v1/health");
}

export function fetchMeta(): Promise<MetaResponse> {
  return request<MetaResponse>("/v1/meta");
}

export interface KnowledgeBaseListParams {
  limit?: number;
  offset?: number;
}

export function fetchKnowledgeBases({
  limit = 20,
  offset = 0,
}: KnowledgeBaseListParams = {}): Promise<KnowledgeBaseListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<KnowledgeBaseListResponse>(`/v1/knowledge-bases?${searchParams.toString()}`);
}

export function fetchKnowledgeBase(id: string): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`/v1/knowledge-bases/${encodeURIComponent(id)}`);
}

export function createKnowledgeBase(
  payload: KnowledgeBaseCreateRequest,
): Promise<KnowledgeBase> {
  return request<KnowledgeBase>("/v1/knowledge-bases", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateKnowledgeBase(
  id: string,
  payload: KnowledgeBaseUpdateRequest,
): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`/v1/knowledge-bases/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteKnowledgeBase(id: string): Promise<void> {
  return requestNoContent(`/v1/knowledge-bases/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}
