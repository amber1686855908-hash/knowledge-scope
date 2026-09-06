export interface HealthResponse {
  status: "ok";
  project_name: string;
  version: string;
  python_version: string;
  config_status: "ok";
  environment: string;
  log_level: string;
  data_dir: string;
}

export interface MetaResponse {
  project_name: string;
  version: string;
  phase: "A1.3";
  status: "foundation";
  config_status: "ok";
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeBaseListResponse {
  items: KnowledgeBase[];
  total: number;
  limit: number;
  offset: number;
}

export interface KnowledgeBaseCreateRequest {
  name: string;
  description?: string | null;
}

export interface KnowledgeBaseUpdateRequest {
  name?: string;
  description?: string | null;
}

export interface Document {
  id: string;
  knowledge_base_id: string;
  original_filename: string;
  media_type: "application/pdf";
  size_bytes: number;
  sha256: string;
  status: "uploaded";
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: Document[];
  total: number;
  limit: number;
  offset: number;
}
