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
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.database_url.endswith("/knowledgescope")


def test_settings_load_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_SCOPE_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_DATA_DIR", "/tmp/knowledgescope-data")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("/tmp/knowledgescope-data")


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
