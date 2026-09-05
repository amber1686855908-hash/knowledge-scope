"""Validated application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
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
        extra="ignore",
    )

    project_name: str = Field(default="KnowledgeScope", min_length=1)
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = (
        "postgresql+asyncpg://knowledgescope:knowledgescope@127.0.0.1:5433/knowledgescope"
    )


def get_settings() -> Settings:
    """Load and validate settings for the current process."""
    return Settings()
