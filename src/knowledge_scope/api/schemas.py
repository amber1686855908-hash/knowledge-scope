"""Explicit request and response models for the KnowledgeScope API."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_scope.documents.models import (
    DOCUMENT_FILENAME_MAX_LENGTH,
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_STATUS_UPLOADED,
)
from knowledge_scope.knowledge_bases.models import (
    KNOWLEDGE_BASE_DESCRIPTION_MAX_LENGTH,
    KNOWLEDGE_BASE_NAME_MAX_LENGTH,
)


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
    phase: Literal["A1.4"]
    status: Literal["foundation"]
    config_status: Literal["ok"]


def _normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("name must contain at least one non-whitespace character")
    return normalized


def _normalize_description(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class KnowledgeBaseCreate(BaseModel):
    """Validated input for creating a knowledge base."""

    name: str = Field(..., max_length=KNOWLEDGE_BASE_NAME_MAX_LENGTH)
    description: str | None = Field(default=None, max_length=KNOWLEDGE_BASE_DESCRIPTION_MAX_LENGTH)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_name(value)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return _normalize_description(value)


class KnowledgeBaseUpdate(BaseModel):
    """Validated partial input for updating a knowledge base."""

    name: str | None = Field(default=None, max_length=KNOWLEDGE_BASE_NAME_MAX_LENGTH)
    description: str | None = Field(default=None, max_length=KNOWLEDGE_BASE_DESCRIPTION_MAX_LENGTH)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        return _normalize_name(value) if value is not None else None

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return _normalize_description(value)

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        return self


class KnowledgeBaseResponse(BaseModel):
    """Public representation of a persisted knowledge base."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    """Bounded list response for knowledge bases."""

    items: list[KnowledgeBaseResponse]
    total: int
    limit: int
    offset: int


class DocumentResponse(BaseModel):
    """Public metadata for one uploaded document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    original_filename: str = Field(..., max_length=DOCUMENT_FILENAME_MAX_LENGTH)
    media_type: Literal[DOCUMENT_MEDIA_TYPE_PDF]
    size_bytes: int = Field(..., gt=0)
    sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    status: Literal[DOCUMENT_STATUS_UPLOADED]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Bounded list response for documents in one knowledge base."""

    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int
