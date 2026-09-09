"""Typed models and deterministic ranking helpers for graph retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from knowledge_scope.linking.models import normalize_linking_label

from .models import (
    ExtractionProvenance,
    GraphProvenance,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)

GRAPH_RETRIEVAL_SCHEMA_VERSION = "1.0"

SeedResolutionMethod = Literal[
    "exact_local_name",
    "exact_local_alias",
    "exact_canonical_name",
    "exact_canonical_alias",
    "contains_local_name",
    "contains_local_alias",
    "contains_canonical_name",
    "contains_canonical_alias",
    "lexical_local_name",
    "lexical_local_alias",
    "lexical_canonical_name",
    "lexical_canonical_alias",
]
GraphPathDirection = Literal["forward", "reverse", "canonical_bridge"]
GraphPathKind = Literal["seed", "relation", "canonical_bridge", "mixed"]


class _GraphRetrievalModel(BaseModel):
    """Strict base for graph-retrieval records crossing module boundaries."""

    model_config = ConfigDict(extra="forbid")


def _require_non_blank(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned


def _sorted_unique(values: Iterable[str], field_name: str) -> list[str]:
    cleaned = [_require_non_blank(value, field_name) for value in values]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field_name} must be unique")
    return sorted(cleaned)


def _unique_in_order(values: Iterable[str], field_name: str) -> list[str]:
    cleaned = [_require_non_blank(value, field_name) for value in values]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field_name} must be unique")
    return cleaned


class GraphEvidence(_GraphRetrievalModel):
    """A stored evidence record that can be resolved to a source chunk."""

    evidence_id: str = Field(pattern=r"^evidence_v2_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
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
        # A3.1's evidence identity includes this list in its original order.
        return _unique_in_order(values, "source_block_id")

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, values: list[str]) -> list[str]:
        return [_require_non_blank(value, "section_path item") for value in values]

    @model_validator(mode="after")
    def validate_page_range(self) -> GraphEvidence:
        if self.page_start > self.page_end:
            raise ValueError("page_end must not be less than page_start")
        return self

    @model_validator(mode="after")
    def validate_identity(self) -> GraphEvidence:
        provenance = GraphProvenance(
            document_id=self.document_id,
            knowledge_base_id=self.knowledge_base_id,
            chunk_id=self.chunk_id,
            page_start=self.page_start,
            page_end=self.page_end,
            source_block_ids=self.source_block_ids,
            section_path=self.section_path,
            extraction_provenance=self.extraction_provenance,
        )
        if self.evidence_id != evidence_id_for(provenance):
            raise ValueError("evidence_id does not match the evidence lineage")
        return self


class GraphEntityReference(_GraphRetrievalModel):
    """The safe, read-only portion of a local graph entity."""

    entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    canonical_name: str = Field(min_length=1, max_length=500)
    entity_type: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("canonical_name", "entity_type")
    @classmethod
    def validate_labels(cls, value: str) -> str:
        return _require_non_blank(value, "entity label")

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        return _sorted_unique(values, "alias")

    @model_validator(mode="after")
    def validate_identity(self) -> GraphEntityReference:
        expected = entity_id_for(
            self.canonical_name,
            self.entity_type,
            knowledge_base_id=self.knowledge_base_id,
            document_id=self.document_id,
        )
        if self.entity_id != expected:
            raise ValueError("entity_id does not match the local entity identity")
        return self


class GraphCanonicalReference(_GraphRetrievalModel):
    """A canonical node used only as an explicit local-entity bridge."""

    canonical_entity_id: str = Field(pattern=r"^canonical_entity_v1_[0-9a-f]{32}$")
    knowledge_base_id: UUID
    canonical_name: str = Field(min_length=1, max_length=500)
    entity_type: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("canonical_name", "entity_type")
    @classmethod
    def validate_labels(cls, value: str) -> str:
        return _require_non_blank(value, "canonical label")

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        return _sorted_unique(values, "canonical alias")


class GraphRelationReference(_GraphRetrievalModel):
    """The safe, read-only portion of a local graph relation."""

    relation_id: str = Field(pattern=r"^relation_v2_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    source_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    target_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    relation_type: str = Field(min_length=1, max_length=200)

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, value: str) -> str:
        return _require_non_blank(value, "relation_type")

    @model_validator(mode="after")
    def validate_identity(self) -> GraphRelationReference:
        expected = relation_id_for(
            self.source_entity_id,
            self.target_entity_id,
            self.relation_type,
            knowledge_base_id=self.knowledge_base_id,
            document_id=self.document_id,
        )
        if self.relation_id != expected:
            raise ValueError("relation_id does not match the local relation identity")
        return self


class GraphEntitySnapshot(_GraphRetrievalModel):
    """One local entity with its current canonical bridge and source evidence."""

    entity: GraphEntityReference
    evidence: list[GraphEvidence] = Field(default_factory=list)
    canonical_entities: list[GraphCanonicalReference] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, values: list[GraphEvidence]) -> list[GraphEvidence]:
        ids = [value.evidence_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence must be unique")
        return sorted(values, key=lambda value: value.evidence_id)

    @field_validator("canonical_entities")
    @classmethod
    def validate_canonical_entities(
        cls,
        values: list[GraphCanonicalReference],
    ) -> list[GraphCanonicalReference]:
        ids = [value.canonical_entity_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("canonical_entities must be unique")
        return sorted(values, key=lambda value: value.canonical_entity_id)

    @model_validator(mode="after")
    def validate_scope(self) -> GraphEntitySnapshot:
        if any(
            value.knowledge_base_id != self.entity.knowledge_base_id
            or value.document_id != self.entity.document_id
            for value in self.evidence
        ):
            raise ValueError("entity evidence must use the local entity scope")
        if any(
            value.knowledge_base_id != self.entity.knowledge_base_id
            for value in self.canonical_entities
        ):
            raise ValueError("canonical bridge must use the local entity KB")
        return self


class GraphSeedCandidate(_GraphRetrievalModel):
    """An application-scored local seed candidate."""

    seed_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    matched_term: str = Field(min_length=1)
    method: SeedResolutionMethod
    score: float = Field(ge=0, le=1)
    matched_canonical_entity_id: str | None = Field(
        default=None,
        pattern=r"^canonical_entity_v1_[0-9a-f]{32}$",
    )


class GraphNeighbor(_GraphRetrievalModel):
    """One bounded, evidence-bearing local-entity edge returned by Neo4j."""

    seed_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    neighbor: GraphEntityReference
    relation: GraphRelationReference | None = None
    canonical: GraphCanonicalReference | None = None
    direction: GraphPathDirection
    evidence: list[GraphEvidence] = Field(default_factory=list)
    neighbor_evidence: list[GraphEvidence] = Field(default_factory=list)
    relation_evidence: list[GraphEvidence] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, values: list[GraphEvidence]) -> list[GraphEvidence]:
        ids = [value.evidence_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("neighbor evidence must be unique")
        return sorted(values, key=lambda value: value.evidence_id)

    @field_validator("neighbor_evidence", "relation_evidence")
    @classmethod
    def validate_support_evidence(cls, values: list[GraphEvidence]) -> list[GraphEvidence]:
        ids = [value.evidence_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("support evidence must be unique")
        return sorted(values, key=lambda value: value.evidence_id)

    @model_validator(mode="after")
    def validate_edge(self) -> GraphNeighbor:
        if self.neighbor.entity_id == self.seed_entity_id:
            raise ValueError("neighbor must differ from seed")
        if self.relation is None and self.canonical is None:
            raise ValueError("neighbor must contain a relation or canonical bridge")
        if self.relation is not None:
            if self.canonical is not None or self.direction == "canonical_bridge":
                raise ValueError("relation and canonical bridge cannot be combined")
            if self.relation.knowledge_base_id != self.neighbor.knowledge_base_id:
                raise ValueError("relation and neighbor must use one KB")
            if self.relation.document_id != self.neighbor.document_id:
                raise ValueError("relation and neighbor must use one document")
            if self.direction == "forward":
                expected = (self.seed_entity_id, self.neighbor.entity_id)
            else:
                expected = (self.neighbor.entity_id, self.seed_entity_id)
            if (self.relation.source_entity_id, self.relation.target_entity_id) != expected:
                raise ValueError("relation direction does not match the neighbor edge")
            if not self.relation_evidence:
                raise ValueError("relation must have current support evidence")
            if any(
                value.knowledge_base_id != self.relation.knowledge_base_id
                or value.document_id != self.relation.document_id
                for value in self.relation_evidence
            ):
                raise ValueError("relation support evidence must use the relation scope")
        if self.canonical is not None:
            if self.relation is not None or self.direction != "canonical_bridge":
                raise ValueError("canonical edge must be a canonical bridge")
            if self.canonical.knowledge_base_id != self.neighbor.knowledge_base_id:
                raise ValueError("canonical bridge must use one KB")
            if self.relation_evidence:
                raise ValueError("canonical bridge cannot contain relation support evidence")
        evidence_ids = {value.evidence_id for value in self.evidence}
        if not {
            value.evidence_id for value in (*self.neighbor_evidence, *self.relation_evidence)
        }.issubset(evidence_ids):
            raise ValueError("support evidence must be included in the neighbor evidence")
        if not self.neighbor_evidence:
            raise ValueError("neighbor must have current support evidence")
        if any(
            value.knowledge_base_id != self.neighbor.knowledge_base_id
            or value.document_id != self.neighbor.document_id
            for value in self.neighbor_evidence
        ):
            raise ValueError("neighbor support evidence must use the neighbor scope")
        if self.relation is None and self.relation_evidence:
            raise ValueError("relation support evidence requires a relation")
        if self.relation is not None and any(
            value.document_id != self.relation.document_id for value in self.evidence
        ):
            raise ValueError("relation evidence must use the relation document")
        if any(
            value.knowledge_base_id != self.neighbor.knowledge_base_id for value in self.evidence
        ):
            raise ValueError("neighbor evidence must use the requested KB")
        return self


class GraphPath(_GraphRetrievalModel):
    """A deterministic explanation of how an evidence item was reached."""

    seed_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    entity_ids: list[str] = Field(min_length=1)
    relation_ids: list[str] = Field(default_factory=list)
    canonical_entity_ids: list[str] = Field(default_factory=list)
    directions: list[GraphPathDirection] = Field(default_factory=list)
    hop_distance: StrictInt = Field(ge=0, le=2)
    kind: GraphPathKind

    @field_validator("entity_ids")
    @classmethod
    def validate_entity_ids(cls, values: list[str]) -> list[str]:
        return [_require_non_blank(value, "entity_id") for value in values]

    @field_validator("relation_ids")
    @classmethod
    def validate_relation_ids(cls, values: list[str]) -> list[str]:
        return _unique_in_order(values, "relation_id")

    @field_validator("canonical_entity_ids")
    @classmethod
    def validate_canonical_entity_ids(cls, values: list[str]) -> list[str]:
        return _unique_in_order(values, "canonical_entity_id")

    @model_validator(mode="after")
    def validate_path_shape(self) -> GraphPath:
        if self.entity_ids[0] != self.seed_entity_id:
            raise ValueError("path must start at its seed entity")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("path entity IDs must be unique")
        if len(self.entity_ids) != self.hop_distance + 1:
            raise ValueError("path entity IDs must match hop distance")
        if len(self.directions) != self.hop_distance:
            raise ValueError("path directions must match hop distance")
        if len(self.relation_ids) + len(self.canonical_entity_ids) != self.hop_distance:
            raise ValueError("path edge IDs must match hop distance")
        if self.hop_distance == 0 and self.kind != "seed":
            raise ValueError("zero-hop path must be a seed path")
        if self.hop_distance == 1 and self.kind not in {"relation", "canonical_bridge"}:
            raise ValueError("one-hop path kind is invalid")
        if self.hop_distance == 2 and self.kind != "mixed":
            raise ValueError("two-hop path must describe a mixed path")
        return self


class GraphEvidenceResult(_GraphRetrievalModel):
    """One unique evidence item with its best graph explanation."""

    evidence: GraphEvidence
    score: float = Field(ge=0, le=1)
    seed_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    retrieval_reason: str = Field(min_length=1, max_length=200)
    paths: list[GraphPath] = Field(min_length=1)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: list[GraphPath]) -> list[GraphPath]:
        return sorted(
            values,
            key=lambda value: (
                value.hop_distance,
                value.kind,
                value.seed_entity_id,
                tuple(value.entity_ids),
                tuple(value.relation_ids),
                tuple(value.canonical_entity_ids),
            ),
        )

    @model_validator(mode="after")
    def validate_result(self) -> GraphEvidenceResult:
        if not any(path.seed_entity_id == self.seed_entity_id for path in self.paths):
            raise ValueError("result seed must be represented by one of its paths")
        return self


class GraphRetrievalConfig(_GraphRetrievalModel):
    """Small safety bounds for query resolution and graph traversal."""

    max_seed_entities: StrictInt = Field(default=5, ge=1, le=50)
    max_hops: StrictInt = Field(default=2, ge=1, le=2)
    max_neighbors: StrictInt = Field(default=20, ge=1, le=100)
    max_relations: StrictInt = Field(default=100, ge=1, le=1_000)
    max_evidence: StrictInt = Field(default=50, ge=1, le=500)
    max_entity_scan: StrictInt = Field(default=10_000, ge=1, le=100_000)
    lexical_threshold: float = Field(default=0.78, ge=0.7, le=1)
    min_lexical_term_length: StrictInt = Field(default=3, ge=2, le=20)


class GraphRetrievalResult(_GraphRetrievalModel):
    """Typed graph retrieval response; no raw Neo4j values escape."""

    schema_version: Literal["1.0"] = GRAPH_RETRIEVAL_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    seeds: list[GraphSeedCandidate] = Field(default_factory=list)
    items: list[GraphEvidenceResult] = Field(default_factory=list)

    @field_validator("seeds")
    @classmethod
    def validate_seed_ids(cls, values: list[GraphSeedCandidate]) -> list[GraphSeedCandidate]:
        ids = [value.seed_entity_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("seeds must be unique")
        return values

    @field_validator("items")
    @classmethod
    def validate_evidence_ids(cls, values: list[GraphEvidenceResult]) -> list[GraphEvidenceResult]:
        ids = [value.evidence.evidence_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence results must be unique")
        return values

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return _require_non_blank(value, "query")

    @model_validator(mode="after")
    def validate_result_scope(self) -> GraphRetrievalResult:
        if any(seed.knowledge_base_id != self.knowledge_base_id for seed in self.seeds):
            raise ValueError("seed scope does not match the request KB")
        if any(item.evidence.knowledge_base_id != self.knowledge_base_id for item in self.items):
            raise ValueError("evidence scope does not match the request KB")
        return self


def seed_method_priority(method: SeedResolutionMethod) -> int:
    """Return a stable priority for equivalent resolution scores."""

    return {
        "exact_local_name": 0,
        "exact_local_alias": 1,
        "exact_canonical_name": 2,
        "exact_canonical_alias": 3,
        "contains_local_name": 4,
        "contains_local_alias": 5,
        "contains_canonical_name": 6,
        "contains_canonical_alias": 7,
        "lexical_local_name": 8,
        "lexical_local_alias": 9,
        "lexical_canonical_name": 10,
        "lexical_canonical_alias": 11,
    }[method]


def graph_score(seed_score: float, hop_distance: int, kind: GraphPathKind) -> float:
    """Score only explainable graph signals; higher is better."""

    distance = 1 / (1 + hop_distance)
    path_bonus = {
        "seed": 0.08,
        "relation": 0.12,
        "canonical_bridge": 0.06,
        "mixed": 0.02,
    }[kind]
    return round(min(1.0, 0.72 * seed_score + 0.2 * distance + path_bonus), 6)


__all__ = [
    "GRAPH_RETRIEVAL_SCHEMA_VERSION",
    "GraphCanonicalReference",
    "GraphEntityReference",
    "GraphEntitySnapshot",
    "GraphEvidence",
    "GraphEvidenceResult",
    "GraphNeighbor",
    "GraphPath",
    "GraphRelationReference",
    "GraphRetrievalConfig",
    "GraphRetrievalResult",
    "GraphSeedCandidate",
    "SeedResolutionMethod",
    "graph_score",
    "normalize_linking_label",
    "seed_method_priority",
]
