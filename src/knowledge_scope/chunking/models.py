"""Parser-independent semantic chunk models."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

CHUNK_SCHEMA_VERSION = "1.0"
CHUNKER_VERSION = "1.0"
SPLITTING_POLICY_VERSION = "section-block-v2"

ContentType = Literal["title", "text", "table", "formula", "image"]


class _ChunkBaseModel(BaseModel):
    """Shared validation settings for chunking data."""

    model_config = ConfigDict(extra="forbid")


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must contain at least one non-whitespace character")
    return value


class ChunkingConfig(_ChunkBaseModel):
    """Character-based structural chunking policy for A1.6."""

    target_chars: StrictInt = Field(default=1200, ge=1)
    max_chars: StrictInt = Field(default=1600, ge=1)
    min_chars: StrictInt = Field(default=240, ge=0)

    @model_validator(mode="after")
    def validate_budget_order(self) -> Self:
        if self.min_chars > self.target_chars:
            raise ValueError("min_chars must not exceed target_chars")
        if self.target_chars > self.max_chars:
            raise ValueError("target_chars must not exceed max_chars")
        return self


def chunking_config_fingerprint(config: ChunkingConfig) -> str:
    """Return a stable fingerprint for all policy inputs that affect chunk output."""
    payload = {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "splitting_policy_version": SPLITTING_POLICY_VERSION,
        "target_chars": config.target_chars,
        "max_chars": config.max_chars,
        "min_chars": config.min_chars,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Chunk(_ChunkBaseModel):
    """One deterministic, lineage-preserving chunk."""

    schema_version: Literal["1.0"] = CHUNK_SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_id: UUID
    ordinal: StrictInt = Field(ge=0)
    text: str = ""
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    content_types: list[ContentType] = Field(min_length=1)
    asset_refs: list[str] = Field(default_factory=list)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        return _require_non_blank(value, "chunk_id")

    @field_validator("source_block_ids")
    @classmethod
    def validate_source_block_ids(cls, value: list[str]) -> list[str]:
        if any(not block_id.strip() for block_id in value):
            raise ValueError("source_block_ids must contain non-blank IDs")
        if len(value) != len(set(value)):
            raise ValueError("source_block_ids must be unique within a chunk")
        return value

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, value: list[str]) -> list[str]:
        if any(not title.strip() for title in value):
            raise ValueError("section_path must contain non-blank titles")
        return value

    @field_validator("content_types")
    @classmethod
    def validate_content_types(cls, value: list[ContentType]) -> list[ContentType]:
        if len(value) != len(set(value)):
            raise ValueError("content_types must be unique within a chunk")
        return value

    @field_validator("asset_refs")
    @classmethod
    def validate_asset_refs(cls, value: list[str]) -> list[str]:
        if any(not asset_ref.strip() for asset_ref in value):
            raise ValueError("asset_refs must contain non-blank references")
        if len(value) != len(set(value)):
            raise ValueError("asset_refs must be unique within a chunk")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_start > self.page_end:
            raise ValueError("page_start must not exceed page_end")
        return self

    @model_validator(mode="after")
    def validate_meaningful_content(self) -> Self:
        if not self.text.strip() and not self.asset_refs:
            raise ValueError("chunk must contain text or at least one asset_ref")
        return self


class ChunkedDocument(_ChunkBaseModel):
    """Chunk artifact envelope for one canonical document."""

    schema_version: Literal["1.0"] = CHUNK_SCHEMA_VERSION
    document_id: UUID
    page_count: StrictInt = Field(ge=1)
    chunker_version: Literal["1.0"] = CHUNKER_VERSION
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: list[Chunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_document_chunks(self) -> Self:
        ordinals = [chunk.ordinal for chunk in self.chunks]
        if ordinals != list(range(len(self.chunks))):
            raise ValueError("chunk ordinals must be contiguous from zero")

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id must be unique within a document")

        for chunk in self.chunks:
            if chunk.document_id != self.document_id:
                raise ValueError("chunk document_id must match the artifact document_id")
            if chunk.page_end > self.page_count:
                raise ValueError("chunk page range exceeds the document page count")
        return self


__all__ = [
    "CHUNKER_VERSION",
    "CHUNK_SCHEMA_VERSION",
    "SPLITTING_POLICY_VERSION",
    "Chunk",
    "ChunkedDocument",
    "ChunkingConfig",
    "ContentType",
    "chunking_config_fingerprint",
]
