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
  phase: "A0.5";
  status: "foundation";
  config_status: "ok";
}
