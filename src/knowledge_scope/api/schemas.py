"""Explicit response models for the KnowledgeScope API."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Non-sensitive runtime and configuration health information."""

    status: Literal["ok"] = "ok"
    project_name: str
    version: str
    python_version: str
    config_status: Literal["ok"]
    environment: str
    log_level: str
    data_dir: str


class MetaResponse(BaseModel):
    """Non-sensitive metadata about the current project foundation."""

    project_name: str
    version: str
    phase: Literal["A0.5"]
    status: Literal["foundation"]
    config_status: Literal["ok"]
