"""Validated application configuration loaded from environment variables."""

from __future__ import annotations

from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
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
    neo4j_uri: str = Field(default="bolt://127.0.0.1:7687", min_length=1)
    neo4j_username: str = Field(default="neo4j", min_length=1)
    neo4j_password: SecretStr | None = None
    neo4j_database: str = Field(default="neo4j", min_length=1)
    neo4j_timeout_seconds: float = Field(default=10.0, gt=0)
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
    graph_extraction_max_tokens: int = Field(default=1024, ge=1)
    graph_extraction_max_parse_retries: int = Field(default=1, ge=0, le=1)
    # The first budget is derived from graph_extraction_max_tokens when this
    # setting is omitted.  An explicit tuple makes the bounded escalation
    # policy part of the runtime configuration and its pipeline fingerprint.
    graph_extraction_truncation_budgets: tuple[int, ...] = ()
    rag_candidate_limit: int = Field(default=10, ge=1, le=100)
    rag_rerank_limit: int = Field(default=5, ge=1, le=100)
    rag_context_budget_chars: int = Field(default=6_000, ge=1)
    rag_max_tokens: int = Field(default=512, ge=1)
    graph_retrieval_max_seed_entities: int = Field(default=5, ge=1, le=50)
    graph_retrieval_max_hops: int = Field(default=2, ge=1, le=2)
    graph_retrieval_max_neighbors: int = Field(default=20, ge=1, le=100)
    graph_retrieval_max_relations: int = Field(default=100, ge=1, le=1_000)
    graph_retrieval_max_evidence: int = Field(default=50, ge=1, le=500)
    graph_retrieval_max_entity_scan: int = Field(default=10_000, ge=1, le=100_000)
    graph_retrieval_lexical_threshold: float = Field(default=0.78, ge=0.7, le=1)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=10_000)
    hybrid_vector_candidate_limit: int = Field(default=20, ge=1, le=100)
    hybrid_vector_rerank_limit: int = Field(default=10, ge=1, le=100)
    hybrid_graph_result_limit: int = Field(default=20, ge=1, le=500)
    hybrid_result_limit: int = Field(default=20, ge=1, le=500)
    hybrid_failure_mode: Literal["strict", "degraded"] = "degraded"

    @model_validator(mode="after")
    def validate_graph_extraction_truncation_policy(self) -> Self:
        """Keep truncation escalation finite, ordered, and tied to the initial budget."""

        budgets = self.graph_extraction_truncation_budgets
        if not budgets:
            budgets = (
                self.graph_extraction_max_tokens,
                self.graph_extraction_max_tokens * 2,
                self.graph_extraction_max_tokens * 4,
            )
            self.graph_extraction_truncation_budgets = budgets
        if len(budgets) > 4:
            raise ValueError("graph extraction truncation budgets must contain at most four values")
        if any(value < 1 for value in budgets):
            raise ValueError("graph extraction truncation budgets must be positive")
        if budgets[0] != self.graph_extraction_max_tokens:
            raise ValueError(
                "the first graph extraction truncation budget must equal "
                "graph_extraction_max_tokens"
            )
        if any(current >= following for current, following in pairwise(budgets)):
            raise ValueError("graph extraction truncation budgets must be strictly increasing")
        return self

    @property
    def graph_extraction_max_truncation_retries(self) -> int:
        """Return the maximum number of bounded budget escalations."""

        return len(self.graph_extraction_truncation_budgets) - 1

    @property
    def graph_extraction_truncation_ceiling(self) -> int:
        """Return the last output budget used by truncation escalation."""

        return self.graph_extraction_truncation_budgets[-1]


def get_settings() -> Settings:
    """Load and validate settings for the current process."""
    return Settings()
