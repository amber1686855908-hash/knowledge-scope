"""Strict request, citation, and server-sent event models for RAG QA."""

from __future__ import annotations

import math
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_scope.graph.retrieval import GraphPath

RAGEventName = Literal["answer_delta", "citations", "complete", "error"]
RAGCompletionStatus = Literal["completed", "insufficient_evidence", "error"]
RAGRetrievalMode = Literal["dense", "unified"]
RAGBranchStatus = Literal["success", "empty", "failed", "timed_out", "cancelled"]
RAGCitationModality = Literal["text", "image", "table", "formula"]
RAGCitationBranchName = Literal["dense", "sparse", "graph", "multimodal"]


class RAGQueryRequest(BaseModel):
    """Public input for one streamed retrieval-augmented question."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None
    retrieval_mode: RAGRetrievalMode = "dense"

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized

    @model_validator(mode="after")
    def validate_retrieval_mode_scope(self) -> Self:
        if self.retrieval_mode == "unified" and self.knowledge_base_id is None:
            raise ValueError("knowledge_base_id is required for unified retrieval")
        return self


class RAGCitationBranch(BaseModel):
    """Auditable contribution from one unified retrieval branch."""

    model_config = ConfigDict(extra="forbid")

    branch: RAGCitationBranchName
    rank: int = Field(ge=1)
    score: float
    graph_seed_entity_id: str | None = None
    graph_retrieval_reason: str | None = None
    graph_paths: list[GraphPath] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("citation branch score must be finite")
        return value

    @model_validator(mode="after")
    def validate_branch_metadata(self) -> Self:
        if self.branch == "graph":
            if self.graph_seed_entity_id is None or self.graph_retrieval_reason is None:
                raise ValueError("graph citation metadata requires seed and reason")
        elif (
            self.graph_seed_entity_id is not None
            or self.graph_retrieval_reason is not None
            or self.graph_paths
        ):
            raise ValueError("graph citation metadata belongs only to graph contributions")
        return self


class RAGCitation(BaseModel):
    """Application-generated metadata for one context source."""

    model_config = ConfigDict(extra="forbid")

    marker: str = Field(pattern=r"^C[1-9][0-9]*$")
    document_id: UUID
    knowledge_base_id: UUID | None = None
    candidate_kind: Literal["chunk", "evidence"] = "chunk"
    chunk_id: str | None = None
    evidence_id: str | None = None
    modality: RAGCitationModality | None = None
    representation_ids: list[str] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    section_title: str | None = None
    final_rank: int | None = Field(default=None, ge=1)
    final_reranker_score: float | None = None
    branch_provenance: list[RAGCitationBranch] = Field(default_factory=list)

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

    @field_validator("representation_ids", "asset_refs")
    @classmethod
    def validate_reference_lists(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("citation references must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("citation references must be unique")
        return values

    @field_validator("final_reranker_score")
    @classmethod
    def validate_final_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("final reranker score must be finite")
        return value

    @field_validator("branch_provenance")
    @classmethod
    def validate_branch_provenance(
        cls,
        values: list[RAGCitationBranch],
    ) -> list[RAGCitationBranch]:
        branches = [value.branch for value in values]
        if len(branches) != len(set(branches)):
            raise ValueError("citation branch provenance must be unique")
        return values

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> Self:
        if self.candidate_kind == "chunk":
            if self.chunk_id is None or not self.chunk_id.strip():
                raise ValueError("chunk citations require chunk_id")
            if self.evidence_id is not None or self.modality is not None:
                raise ValueError("chunk citations must not claim Evidence identity")
        else:
            if self.chunk_id is not None:
                raise ValueError("Evidence citations must not expose chunk_id")
            if self.evidence_id is None or not self.evidence_id.strip():
                raise ValueError("Evidence citations require evidence_id")
            if self.knowledge_base_id is None:
                raise ValueError("Evidence citations require knowledge_base_id")
            if self.modality is None or not self.representation_ids:
                raise ValueError("Evidence citations require modality and representations")
        if (self.final_rank is None) != (self.final_reranker_score is None):
            raise ValueError("final rank and score must be provided together")
        return self


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
    retrieval_mode: RAGRetrievalMode = "dense"
    retrieval_degraded: bool | None = None
    retrieval_branch_statuses: dict[str, RAGBranchStatus] | None = None
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
    "RAGBranchStatus",
    "RAGCitation",
    "RAGCitationBranch",
    "RAGCitationBranchName",
    "RAGCitationModality",
    "RAGCitationsData",
    "RAGCompleteData",
    "RAGCompletionStatus",
    "RAGErrorData",
    "RAGEventName",
    "RAGQueryRequest",
    "RAGRetrievalMode",
    "RAGStreamEvent",
]
