import type {
  HealthResponse,
  KnowledgeBase,
  KnowledgeBaseCreateRequest,
  KnowledgeBaseListResponse,
  KnowledgeBaseUpdateRequest,
  MetaResponse,
  Document,
  DocumentListResponse,
} from "./types";

const DEFAULT_API_BASE_URL = "/api";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, statusText: string, detail?: string) {
    super(detail || `API request failed: ${status} ${statusText}`);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorDetail(response: Response): Promise<string | undefined> {
  try {
    const payload: unknown = await response.json();
    if (typeof payload === "object" && payload !== null && "detail" in payload) {
      const detail = payload.detail;
      if (typeof detail === "string") {
        return detail;
      }
      if (Array.isArray(detail)) {
        const messages = detail.flatMap((item) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            return typeof item.msg === "string" ? [item.msg] : [];
          }
          return [];
        });
        return messages.length > 0 ? messages.join("；") : undefined;
      }
    }
  } catch {
    return undefined;
  }
  return undefined;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (
    options.body !== undefined &&
    options.body !== null &&
    !(options.body instanceof FormData) &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    throw new ApiError(
      response.status,
      response.statusText,
      await readErrorDetail(response),
    );
  }

  return (await response.json()) as T;
}

async function requestNoContent(path: string, options: RequestInit = {}): Promise<void> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    throw new ApiError(
      response.status,
      response.statusText,
      await readErrorDetail(response),
    );
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

export interface DocumentListParams {
  limit?: number;
  offset?: number;
}

export function fetchDocuments(
  knowledgeBaseId: string,
  { limit = 10, offset = 0 }: DocumentListParams = {},
): Promise<DocumentListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<DocumentListResponse>(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents?${searchParams.toString()}`,
  );
}

export function uploadDocument(knowledgeBaseId: string, file: File): Promise<Document> {
  const formData = new FormData();
  formData.append("file", file);
  return request<Document>(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents`,
    {
      method: "POST",
      body: formData,
    },
  );
}

export function deleteDocument(knowledgeBaseId: string, documentId: string): Promise<void> {
  return requestNoContent(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}
