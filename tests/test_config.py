from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import _resolve_test_base_url
from knowledge_scope.shared.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.project_name == "KnowledgeScope"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.max_upload_size_bytes == 50 * 1024 * 1024
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.database_url.endswith("/knowledgescope")
    assert settings.mineru_command == "mineru"
    assert settings.mineru_timeout_seconds == 1800
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection_name == "knowledgescope_chunks_v1"
    assert settings.qdrant_representation_collection_name == "knowledgescope_representations_v1"
    assert settings.neo4j_uri == "bolt://127.0.0.1:7687"
    assert settings.neo4j_username == "neo4j"
    assert settings.neo4j_password is None
    assert settings.neo4j_database == "neo4j"
    assert settings.neo4j_timeout_seconds == 10.0
    assert settings.embedding_model_revision
    assert settings.reranker_model_key == "qwen3-reranker-0.6b"
    assert settings.reranker_device == "cuda"
    assert settings.reranker_dtype == "float16"
    assert settings.reranker_batch_size == 4
    assert settings.reranker_max_seq_length == 512
    assert settings.llm_provider == "deepseek"
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_api_key is None
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_timeout_seconds == 60.0
    assert settings.llm_max_retries == 0
    assert settings.llm_input_cost_per_1k_tokens is None
    assert settings.llm_output_cost_per_1k_tokens is None
    assert settings.graph_extraction_max_tokens == 1024
    assert settings.graph_extraction_max_parse_retries == 1
    assert settings.graph_extraction_truncation_budgets == (1024, 2048, 4096)
    assert settings.graph_extraction_max_truncation_retries == 2
    assert settings.graph_extraction_truncation_ceiling == 4096
    assert settings.rag_candidate_limit == 10
    assert settings.rag_rerank_limit == 5
    assert settings.rag_context_budget_chars == 6_000
    assert settings.rag_max_tokens == 512
    assert settings.chatbi_max_rows == 1_000
    assert settings.chatbi_max_result_bytes == 4_000_000
    assert settings.chatbi_max_cell_bytes == 1_000_000
    assert settings.chatbi_max_nested_value_depth == 32
    assert settings.chatbi_max_collection_items == 10_000
    assert settings.chatbi_statement_timeout_ms == 30_000
    assert settings.chatbi_schema_connection_timeout_seconds == 10.0
    assert settings.chatbi_schema_context_max_chars == 24_000
    assert settings.chatbi_allowed_schemas == ["public"]
    assert settings.chatbi_allow_views is False
    assert settings.chatbi_nl2sql_max_tokens == 512
    assert settings.chatbi_agent_max_sql_attempts == 2
    assert settings.chatbi_agent_max_repair_attempts == 1
    assert settings.chatbi_agent_max_steps == 6
    assert settings.chatbi_agent_max_llm_calls == 3
    assert settings.chatbi_analysis_max_tokens == 512
    assert settings.sparse_index_path == Path("data/evaluation/a4-3/sparse.sqlite3")
    assert settings.sparse_bm25_k1 == 1.2
    assert settings.sparse_bm25_b == 0.75


def test_settings_load_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_SCOPE_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_DATA_DIR", "/tmp/knowledgescope-data")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MAX_UPLOAD_SIZE_BYTES", "1024")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MINERU_COMMAND", "/opt/mineru/bin/mineru")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_QDRANT_URL", "http://localhost:6334")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_QDRANT_UPSERT_BATCH_SIZE", "64")
    monkeypatch.setenv(
        "KNOWLEDGE_SCOPE_QDRANT_REPRESENTATION_COLLECTION_NAME",
        "representations_test_v1",
    )
    monkeypatch.setenv("KNOWLEDGE_SCOPE_NEO4J_URI", "bolt://localhost:17687")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_NEO4J_USERNAME", "graph-user")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_NEO4J_PASSWORD", "test-graph-secret")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_NEO4J_DATABASE", "graph-test")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_NEO4J_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_MODEL_KEY", "bge-reranker-v2-m3")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_DTYPE", "float32")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_BATCH_SIZE", "2")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_MAX_SEQ_LENGTH", "256")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_MODEL", "custom-model")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_INPUT_COST_PER_1K_TOKENS", "0.12")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LLM_OUTPUT_COST_PER_1K_TOKENS", "0.34")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_GRAPH_EXTRACTION_MAX_TOKENS", "768")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_GRAPH_EXTRACTION_MAX_PARSE_RETRIES", "1")
    monkeypatch.setenv(
        "KNOWLEDGE_SCOPE_GRAPH_EXTRACTION_TRUNCATION_BUDGETS",
        "[768,1536,3072]",
    )
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RAG_CANDIDATE_LIMIT", "8")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RAG_RERANK_LIMIT", "4")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS", "4000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RAG_MAX_TOKENS", "256")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_MAX_ROWS", "250")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_MAX_RESULT_BYTES", "2000000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_MAX_CELL_BYTES", "500000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_MAX_NESTED_VALUE_DEPTH", "16")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_MAX_COLLECTION_ITEMS", "5000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_STATEMENT_TIMEOUT_MS", "5000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_SCHEMA_CONNECTION_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_SCHEMA_CONTEXT_MAX_CHARS", "12000")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_ALLOWED_SCHEMAS", '["analytics", "public"]')
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_ALLOW_VIEWS", "true")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_NL2SQL_MAX_TOKENS", "1024")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_AGENT_MAX_SQL_ATTEMPTS", "3")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_AGENT_MAX_REPAIR_ATTEMPTS", "2")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_AGENT_MAX_STEPS", "8")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_AGENT_MAX_LLM_CALLS", "4")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_CHATBI_ANALYSIS_MAX_TOKENS", "768")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_SPARSE_INDEX_PATH", "/tmp/sparse.sqlite3")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_SPARSE_BM25_K1", "1.4")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_SPARSE_BM25_B", "0.6")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("/tmp/knowledgescope-data")
    assert settings.max_upload_size_bytes == 1024
    assert settings.mineru_command == "/opt/mineru/bin/mineru"
    assert settings.mineru_timeout_seconds == 600
    assert settings.qdrant_url == "http://localhost:6334"
    assert settings.qdrant_upsert_batch_size == 64
    assert settings.qdrant_representation_collection_name == "representations_test_v1"
    assert settings.neo4j_uri == "bolt://localhost:17687"
    assert settings.neo4j_username == "graph-user"
    assert settings.neo4j_password is not None
    assert settings.neo4j_password.get_secret_value() == "test-graph-secret"
    assert settings.neo4j_database == "graph-test"
    assert settings.neo4j_timeout_seconds == 2.5
    assert settings.reranker_model_key == "bge-reranker-v2-m3"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_dtype == "float32"
    assert settings.reranker_batch_size == 2
    assert settings.reranker_max_seq_length == 256
    assert settings.llm_base_url == "https://llm.example.test/v1"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "test-secret"
    assert settings.llm_model == "custom-model"
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_max_retries == 2
    assert settings.llm_input_cost_per_1k_tokens == Decimal("0.12")
    assert settings.llm_output_cost_per_1k_tokens == Decimal("0.34")
    assert settings.graph_extraction_max_tokens == 768
    assert settings.graph_extraction_max_parse_retries == 1
    assert settings.graph_extraction_truncation_budgets == (768, 1536, 3072)
    assert settings.graph_extraction_max_truncation_retries == 2
    assert settings.graph_extraction_truncation_ceiling == 3072
    assert settings.rag_candidate_limit == 8
    assert settings.rag_rerank_limit == 4
    assert settings.rag_context_budget_chars == 4000
    assert settings.rag_max_tokens == 256
    assert settings.chatbi_max_rows == 250
    assert settings.chatbi_max_result_bytes == 2_000_000
    assert settings.chatbi_max_cell_bytes == 500_000
    assert settings.chatbi_max_nested_value_depth == 16
    assert settings.chatbi_max_collection_items == 5_000
    assert settings.chatbi_statement_timeout_ms == 5_000
    assert settings.chatbi_schema_connection_timeout_seconds == 2.5
    assert settings.chatbi_schema_context_max_chars == 12_000
    assert settings.chatbi_allowed_schemas == ["analytics", "public"]
    assert settings.chatbi_allow_views is True
    assert settings.chatbi_nl2sql_max_tokens == 1024
    assert settings.chatbi_agent_max_sql_attempts == 3
    assert settings.chatbi_agent_max_repair_attempts == 2
    assert settings.chatbi_agent_max_steps == 8
    assert settings.chatbi_agent_max_llm_calls == 4
    assert settings.chatbi_analysis_max_tokens == 768
    assert settings.sparse_index_path == Path("/tmp/sparse.sqlite3")
    assert settings.sparse_bm25_k1 == 1.4
    assert settings.sparse_bm25_b == 0.6


def test_extraction_retry_setting_allows_at_most_one_corrective_retry() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, graph_extraction_max_parse_retries=2)


def test_truncation_budget_policy_is_bounded_and_ordered() -> None:
    with pytest.raises(ValidationError, match="first graph extraction truncation budget"):
        Settings(_env_file=None, graph_extraction_truncation_budgets=(2048, 4096))
    with pytest.raises(ValidationError, match="strictly increasing"):
        Settings(_env_file=None, graph_extraction_truncation_budgets=(1024, 1024))
    with pytest.raises(ValidationError, match="at most four"):
        Settings(
            _env_file=None,
            graph_extraction_truncation_budgets=(1024, 2048, 4096, 8192, 16384),
        )


def test_settings_load_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KNOWLEDGE_SCOPE_PROJECT_NAME=ConfiguredProject\n"
        "KNOWLEDGE_SCOPE_ENVIRONMENT=test\n"
        'KNOWLEDGE_SCOPE_CORS_ORIGINS=["http://localhost:4173"]\n'
        "KNOWLEDGE_SCOPE_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/example\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.project_name == "ConfiguredProject"
    assert settings.environment == "test"
    assert settings.cors_origins == ["http://localhost:4173"]
    assert settings.database_url.endswith("/example")


def test_env_example_loads_with_optional_values_empty() -> None:
    settings = Settings(_env_file=Path(__file__).resolve().parents[1] / ".env.example")

    assert settings.llm_api_key is None
    assert settings.llm_input_cost_per_1k_tokens is None
    assert settings.llm_output_cost_per_1k_tokens is None
    assert settings.reranker_model_revision is None
    assert settings.neo4j_password is None


def test_settings_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="staging")


def test_test_database_guard_rejects_remote_application_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KNOWLEDGE_SCOPE_TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "KNOWLEDGE_SCOPE_DATABASE_URL",
        "postgresql+asyncpg://user:pass@staging.example.com:5432/knowledgescope",
    )

    with pytest.raises(pytest.UsageError, match="KNOWLEDGE_SCOPE_TEST_DATABASE_URL"):
        _resolve_test_base_url()


def test_test_database_guard_prefers_explicit_test_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "KNOWLEDGE_SCOPE_DATABASE_URL",
        "postgresql+asyncpg://user:pass@staging.example.com:5432/knowledgescope",
    )
    monkeypatch.setenv(
        "KNOWLEDGE_SCOPE_TEST_DATABASE_URL",
        "postgresql+asyncpg://test:pass@test-db.example.com:5432/postgres",
    )

    assert _resolve_test_base_url().host == "test-db.example.com"
