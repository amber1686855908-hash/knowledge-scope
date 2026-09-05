from pathlib import Path

import pytest
from pydantic import ValidationError

from knowledge_scope.shared.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.project_name == "KnowledgeScope"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")


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
        "KNOWLEDGE_SCOPE_PROJECT_NAME=ConfiguredProject\nKNOWLEDGE_SCOPE_ENVIRONMENT=test\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.project_name == "ConfiguredProject"
    assert settings.environment == "test"


def test_settings_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="staging")
