"""Provider-independent canonical models for extracted graph facts."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from typing import Final, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

GRAPH_SCHEMA_VERSION: Final = "1.0"
GRAPH_ID_VERSION: Final = "v2"
EntityId = str
RelationId = str


class _GraphBaseModel(BaseModel):
    """Shared strict validation settings for graph data."""

    model_config = ConfigDict(extra="forbid")


def _require_non_blank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one non-whitespace character")
    return normalized


def _identity_text(value: str, field_name: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", _require_non_blank(value, field_name)).casefold().split()
    )


def canonical_identity_json(kind: str, payload: Mapping[str, object]) -> str:
    """Serialize an identity envelope deterministically and without delimiters."""

    envelope = {
        "identity_schema_version": GRAPH_ID_VERSION,
        "kind": kind,
        "payload": dict(payload),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest_id(kind: str, payload: Mapping[str, object]) -> str:
    encoded = canonical_identity_json(kind, payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{kind}_{GRAPH_ID_VERSION}_{digest}"


class ExtractionProvenance(_GraphBaseModel):
    """Provenance for how one evidence-supported fact was extracted or authored."""

    method: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1)

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        return _require_non_blank(value, "method")


class GraphProvenance(_GraphBaseModel):
    """A canonical evidence reference for one graph fact."""

    document_id: UUID
    knowledge_base_id: UUID
    chunk_id: str = Field(min_length=1)
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    extraction_provenance: ExtractionProvenance | None = None

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        return _require_non_blank(value, "chunk_id")

    @field_validator("source_block_ids")
    @classmethod
    def validate_source_block_ids(cls, values: list[str]) -> list[str]:
        normalized = [_require_non_blank(value, "source_block_id") for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_block_ids must be unique")
        return normalized

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, values: list[str]) -> list[str]:
        return [_require_non_blank(value, "section_path item") for value in values]

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_start > self.page_end:
            raise ValueError("page_end must not be less than page_start")
        return self


def entity_id_for(
    canonical_name: str,
    entity_type: str,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> EntityId:
    """Build a deterministic provisional identity for one document extraction.

    This intentionally performs only Unicode/whitespace/case normalization. It is
    not an entity-linking or synonym-resolution algorithm. The structured payload
    keeps arbitrary valid field contents unambiguous.
    """

    normalized_name = _identity_text(canonical_name, "canonical_name")
    normalized_type = _identity_text(entity_type, "entity_type")
    return _digest_id(
        "entity",
        {
            "knowledge_base_id": str(knowledge_base_id),
            "document_id": str(document_id),
            "canonical_name": normalized_name,
            "entity_type": normalized_type,
        },
    )


def relation_id_for(
    source_entity_id: EntityId,
    target_entity_id: EntityId,
    relation_type: str,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> RelationId:
    """Build a deterministic, directed relation identity in one local scope."""

    source = _require_non_blank(source_entity_id, "source_entity_id")
    target = _require_non_blank(target_entity_id, "target_entity_id")
    normalized_type = _identity_text(relation_type, "relation_type")
    return _digest_id(
        "relation",
        {
            "knowledge_base_id": str(knowledge_base_id),
            "document_id": str(document_id),
            "source_entity_id": source,
            "target_entity_id": target,
            "relation_type": normalized_type,
        },
    )


def evidence_id_for(provenance: GraphProvenance) -> str:
    """Build a deterministic ID for a reusable lineage evidence node."""

    payload = provenance.model_dump(mode="json")
    return _digest_id("evidence", payload)


def _validate_entity_id(value: str) -> str:
    value = _require_non_blank(value, "entity_id")
    prefix = f"entity_{GRAPH_ID_VERSION}_"
    if not (value.startswith(prefix) and len(value) == len(prefix) + 64):
        raise ValueError("entity_id must be generated by entity_id_for")
    try:
        int(value.removeprefix(prefix), 16)
    except ValueError as error:
        raise ValueError("entity_id must contain a hexadecimal digest") from error
    return value


def _validate_relation_id(value: str) -> str:
    value = _require_non_blank(value, "relation_id")
    prefix = f"relation_{GRAPH_ID_VERSION}_"
    if not (value.startswith(prefix) and len(value) == len(prefix) + 64):
        raise ValueError("relation_id must be generated by relation_id_for")
    try:
        int(value.removeprefix(prefix), 16)
    except ValueError as error:
        raise ValueError("relation_id must contain a hexadecimal digest") from error
    return value


class GraphEntity(_GraphBaseModel):
    """One provisional entity with at least one source lineage."""

    schema_version: Literal["1.0"] = GRAPH_SCHEMA_VERSION
    entity_id: EntityId
    knowledge_base_id: UUID
    document_id: UUID
    canonical_name: str = Field(min_length=1, max_length=500)
    entity_type: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list)
    provenance: list[GraphProvenance] = Field(min_length=1)

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, value: str) -> str:
        return _validate_entity_id(value)

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, value: str) -> str:
        return _require_non_blank(value, "canonical_name")

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, value: str) -> str:
        return _require_non_blank(value, "entity_type")

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        normalized = [_require_non_blank(value, "alias") for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("aliases must be unique")
        return sorted(normalized)

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, values: list[GraphProvenance]) -> list[GraphProvenance]:
        keys = [evidence_id_for(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("provenance entries must be unique")
        return sorted(values, key=evidence_id_for)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if any(value.knowledge_base_id != self.knowledge_base_id for value in self.provenance):
            raise ValueError("all entity provenance must use the entity knowledge_base_id")
        if any(value.document_id != self.document_id for value in self.provenance):
            raise ValueError("all entity provenance must use the entity document_id")
        if self.entity_id != entity_id_for(
            self.canonical_name,
            self.entity_type,
            knowledge_base_id=self.knowledge_base_id,
            document_id=self.document_id,
        ):
            raise ValueError("entity_id does not match the local entity identity")
        return self


class GraphRelation(_GraphBaseModel):
    """One directed provisional relation with source evidence."""

    schema_version: Literal["1.0"] = GRAPH_SCHEMA_VERSION
    relation_id: RelationId
    knowledge_base_id: UUID
    document_id: UUID
    source_entity_id: EntityId
    target_entity_id: EntityId
    relation_type: str = Field(min_length=1, max_length=200)
    provenance: list[GraphProvenance] = Field(min_length=1)

    @field_validator("relation_id", "source_entity_id", "target_entity_id")
    @classmethod
    def validate_entity_references(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        if field_name == "relation_id":
            return _validate_relation_id(value)
        return _validate_entity_id(value)

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, value: str) -> str:
        return _require_non_blank(value, "relation_type")

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, values: list[GraphProvenance]) -> list[GraphProvenance]:
        keys = [evidence_id_for(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("provenance entries must be unique")
        return sorted(values, key=evidence_id_for)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if any(value.knowledge_base_id != self.knowledge_base_id for value in self.provenance):
            raise ValueError("all relation provenance must use the relation knowledge_base_id")
        if any(value.document_id != self.document_id for value in self.provenance):
            raise ValueError("all relation provenance must use the relation document_id")
        if self.relation_id != relation_id_for(
            self.source_entity_id,
            self.target_entity_id,
            self.relation_type,
            knowledge_base_id=self.knowledge_base_id,
            document_id=self.document_id,
        ):
            raise ValueError("relation_id does not match the local relation identity")
        return self


__all__ = [
    "GRAPH_ID_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "EntityId",
    "ExtractionProvenance",
    "GraphEntity",
    "GraphProvenance",
    "GraphRelation",
    "RelationId",
    "canonical_identity_json",
    "entity_id_for",
    "evidence_id_for",
    "relation_id_for",
]
