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
    assert settings.embedding_model_revision
    assert settings.reranker_model_key == "qwen3-reranker-0.6b"
    assert settings.reranker_device == "cuda"
    assert settings.reranker_dtype == "float16"
    assert settings.reranker_batch_size == 4
    assert settings.reranker_max_seq_length == 512


def test_settings_load_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_SCOPE_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_DATA_DIR", "/tmp/knowledgescope-data")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MAX_UPLOAD_SIZE_BYTES", "1024")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MINERU_COMMAND", "/opt/mineru/bin/mineru")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_QDRANT_URL", "http://localhost:6334")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_QDRANT_UPSERT_BATCH_SIZE", "64")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_MODEL_KEY", "bge-reranker-v2-m3")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_DTYPE", "float32")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_BATCH_SIZE", "2")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_RERANKER_MAX_SEQ_LENGTH", "256")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("/tmp/knowledgescope-data")
    assert settings.max_upload_size_bytes == 1024
    assert settings.mineru_command == "/opt/mineru/bin/mineru"
    assert settings.mineru_timeout_seconds == 600
    assert settings.qdrant_url == "http://localhost:6334"
    assert settings.qdrant_upsert_batch_size == 64
    assert settings.reranker_model_key == "bge-reranker-v2-m3"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_dtype == "float32"
    assert settings.reranker_batch_size == 2
    assert settings.reranker_max_seq_length == 256


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
