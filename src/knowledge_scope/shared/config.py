"""Validated application configuration loaded from environment variables."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Runtime settings for the KnowledgeScope application."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KNOWLEDGE_SCOPE_",
        env_ignore_empty=True,
        extra="ignore",
    )

    project_name: str = Field(default="KnowledgeScope", min_length=1)
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    max_upload_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = (
        "postgresql+asyncpg://knowledgescope:knowledgescope@127.0.0.1:5433/knowledgescope"
    )
    mineru_command: str = Field(default="mineru", min_length=1)
    mineru_timeout_seconds: int = Field(default=1800, ge=1)
    qdrant_url: str = Field(default="http://127.0.0.1:6333", min_length=1)
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: int = Field(default=10, ge=1)
    qdrant_upsert_batch_size: int = Field(default=128, ge=1, le=2_000)
    qdrant_collection_name: str = Field(
        default="knowledgescope_chunks_v1",
        pattern=r"^[a-z0-9][a-z0-9_-]{2,62}$",
    )
    embedding_device: str = Field(default="cuda", min_length=1)
    embedding_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_max_seq_length: int = Field(default=512, ge=1)
    embedding_model_revision: str = Field(
        default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        min_length=1,
    )
    reranker_model_key: str = Field(default="qwen3-reranker-0.6b", min_length=1)
    reranker_device: str = Field(default="cuda", min_length=1)
    reranker_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    reranker_batch_size: int = Field(default=4, ge=1)
    reranker_max_seq_length: int = Field(default=512, ge=1)
    reranker_model_revision: str | None = Field(default=None, min_length=1)
    llm_provider: Literal["deepseek"] = "deepseek"
    llm_base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    llm_api_key: SecretStr | None = None
    llm_model: str = Field(default="deepseek-chat", min_length=1)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=0, ge=0, le=2)
    llm_input_cost_per_1k_tokens: Decimal | None = Field(default=None, ge=0)
    llm_output_cost_per_1k_tokens: Decimal | None = Field(default=None, ge=0)
    rag_candidate_limit: int = Field(default=10, ge=1, le=100)
    rag_rerank_limit: int = Field(default=5, ge=1, le=100)
    rag_context_budget_chars: int = Field(default=6_000, ge=1)
    rag_max_tokens: int = Field(default=512, ge=1)


def get_settings() -> Settings:
    """Load and validate settings for the current process."""
    return Settings()
