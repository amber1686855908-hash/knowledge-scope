"""Strict request, citation, and server-sent event models for RAG QA."""

from __future__ import annotations

from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RAGEventName = Literal["answer_delta", "citations", "complete", "error"]
RAGCompletionStatus = Literal["completed", "insufficient_evidence", "error"]


class RAGQueryRequest(BaseModel):
    """Public input for one streamed retrieval-augmented question."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class RAGCitation(BaseModel):
    """Application-generated metadata for one context source."""

    model_config = ConfigDict(extra="forbid")

    marker: str = Field(pattern=r"^C[1-9][0-9]*$")
    document_id: UUID
    knowledge_base_id: UUID | None = None
    chunk_id: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    section_title: str | None = None

    @field_validator("page_end")
    @classmethod
    def validate_page_range(cls, value: int, info: Any) -> int:
        page_start = info.data.get("page_start")
        if page_start is not None and value < page_start:
            raise ValueError("page_end must not be less than page_start")
        return value

    @field_validator("source_block_ids")
    @classmethod
    def validate_source_blocks(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("source_block_ids must not contain blank values")
        if len(set(values)) != len(values):
            raise ValueError("source_block_ids must be unique")
        return values


class RAGAnswerDeltaData(BaseModel):
    """Typed payload for one answer text increment."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class RAGCitationsData(BaseModel):
    """Typed payload for application-controlled citation metadata."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: str = Field(min_length=1)
    items: list[RAGCitation]

    @field_validator("items")
    @classmethod
    def validate_markers(cls, values: list[RAGCitation]) -> list[RAGCitation]:
        markers = [item.marker for item in values]
        if len(set(markers)) != len(markers):
            raise ValueError("citation markers must be unique")
        return values


class RAGCompleteData(BaseModel):
    """Typed terminal status and timing/usage payload."""

    model_config = ConfigDict(extra="forbid")

    status: RAGCompletionStatus
    prompt_version: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    retrieval_latency_ms: float | None = Field(default=None, ge=0)
    llm_latency_ms: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)


class RAGErrorData(BaseModel):
    """Typed, sanitized error payload for the SSE stream."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RAGStreamEvent(BaseModel):
    """One JSON payload carried by the RAG SSE endpoint."""

    model_config = ConfigDict(extra="forbid")

    event: RAGEventName
    data: dict[str, Any]

    @model_validator(mode="after")
    def validate_event_data(self) -> Self:
        event_models = {
            "answer_delta": RAGAnswerDeltaData,
            "citations": RAGCitationsData,
            "complete": RAGCompleteData,
            "error": RAGErrorData,
        }
        event_models[self.event].model_validate(self.data)
        return self


__all__ = [
    "RAGAnswerDeltaData",
    "RAGCitation",
    "RAGCitationsData",
    "RAGCompleteData",
    "RAGCompletionStatus",
    "RAGErrorData",
    "RAGEventName",
    "RAGQueryRequest",
    "RAGStreamEvent",
]
