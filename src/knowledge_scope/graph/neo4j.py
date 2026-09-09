"""Small Neo4j adapter for the canonical KnowledgeScope graph schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from knowledge_scope.linking.models import (
    CanonicalEntity,
    EntityCanonicalLink,
    EntityLinkDecision,
    canonical_link_id_for,
    link_decision_id_for,
    normalize_linking_label,
)
from knowledge_scope.shared.config import Settings

from .models import (
    GRAPH_SCHEMA_VERSION,
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)

_T = TypeVar("_T")

NEO4J_ENTITY_LABEL = "KnowledgeEntity"
NEO4J_RELATION_LABEL = "KnowledgeRelation"
NEO4J_EVIDENCE_LABEL = "KnowledgeEvidence"
NEO4J_CANONICAL_ENTITY_LABEL = "CanonicalEntity"
NEO4J_LINK_PAIR_LABEL = "KnowledgeLinkPair"
NEO4J_LINK_DECISION_LABEL = "KnowledgeLinkDecision"


_LIST_LEGACY_ENTITY_TYPES_QUERY = """
MATCH (entity:KnowledgeEntity)
WHERE entity.entity_type_normalized IS NULL
RETURN entity.entity_id AS entity_id, entity.entity_type AS entity_type
ORDER BY entity.entity_id
""".strip()

_BACKFILL_ENTITY_TYPE_QUERY = """
MATCH (entity:KnowledgeEntity {entity_id: $entity_id})
WHERE entity.entity_type_normalized IS NULL
SET entity.entity_type_normalized = $entity_type_normalized
""".strip()


class GraphStoreError(RuntimeError):
    """Raised when a Neo4j operation cannot complete safely."""


class Neo4jReadiness(BaseModel):
    """Non-sensitive Neo4j connectivity information."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "unavailable"]
    database: str
    schema_version: Literal["1.0"] = GRAPH_SCHEMA_VERSION
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GraphDeleteResult:
    """Counts returned after removing one document's graph evidence."""

    document_id: UUID
    evidence_count: int
    relation_count: int
    entity_count: int


@dataclass(frozen=True, slots=True)
class GraphUpsertResult:
    """Counts returned after one extraction batch transaction."""

    entity_count: int
    relation_count: int


@dataclass(frozen=True, slots=True)
class GraphLinkUpsertResult:
    """Counts returned after one canonical-linking transaction."""

    canonical_entity_count: int
    decision_count: int
    mapping_count: int


@dataclass(frozen=True, slots=True)
class GraphLinkDeleteResult:
    """Counts returned after removing one document's linking records."""

    knowledge_base_id: UUID
    document_id: UUID
    decision_count: int
    mapping_count: int
    canonical_entity_count: int


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE CONSTRAINT knowledgescope_entity_id_unique IF NOT EXISTS
    FOR (entity:KnowledgeEntity) REQUIRE entity.entity_id IS UNIQUE
    """.strip(),
    """
    CREATE CONSTRAINT knowledgescope_relation_id_unique IF NOT EXISTS
    FOR (relation:KnowledgeRelation) REQUIRE relation.relation_id IS UNIQUE
    """.strip(),
    """
    CREATE CONSTRAINT knowledgescope_evidence_id_unique IF NOT EXISTS
    FOR (evidence:KnowledgeEvidence) REQUIRE evidence.evidence_id IS UNIQUE
    """.strip(),
    """
    CREATE INDEX knowledgescope_entity_type IF NOT EXISTS
    FOR (entity:KnowledgeEntity) ON (entity.entity_type)
    """.strip(),
    """
    CREATE INDEX knowledgescope_relation_type IF NOT EXISTS
    FOR (relation:KnowledgeRelation) ON (relation.relation_type)
    """.strip(),
    """
    CREATE INDEX knowledgescope_evidence_document IF NOT EXISTS
    FOR (evidence:KnowledgeEvidence) ON (evidence.document_id)
    """.strip(),
    """
    CREATE CONSTRAINT knowledgescope_canonical_entity_id_unique IF NOT EXISTS
    FOR (canonical:CanonicalEntity) REQUIRE canonical.canonical_entity_id IS UNIQUE
    """.strip(),
    """
    CREATE CONSTRAINT knowledgescope_link_decision_id_unique IF NOT EXISTS
    FOR (decision:KnowledgeLinkDecision) REQUIRE decision.link_decision_id IS UNIQUE
    """.strip(),
    """
    CREATE CONSTRAINT knowledgescope_link_pair_id_unique IF NOT EXISTS
    FOR (pair:KnowledgeLinkPair) REQUIRE pair.link_pair_id IS UNIQUE
    """.strip(),
    """
    CREATE INDEX knowledgescope_canonical_entity_kb IF NOT EXISTS
    FOR (canonical:CanonicalEntity) ON (canonical.knowledge_base_id)
    """.strip(),
    """
    CREATE INDEX knowledgescope_link_decision_kb IF NOT EXISTS
    FOR (decision:KnowledgeLinkDecision) ON (decision.knowledge_base_id)
    """.strip(),
)

_UPSERT_ENTITY_QUERY = """
MERGE (entity:KnowledgeEntity {entity_id: $entity_id})
ON CREATE SET entity.schema_version = $schema_version,
              entity.knowledge_base_id = $knowledge_base_id,
              entity.document_id = $document_id,
              entity.canonical_name = $canonical_name,
              entity.entity_type = $entity_type,
              entity.entity_type_normalized = $entity_type_normalized,
              entity.aliases = []
SET entity.entity_type_normalized = $entity_type_normalized,
    entity._alias_merge_revision = coalesce(entity._alias_merge_revision, 0) + 1
WITH entity, coalesce(entity.aliases, []) + $aliases AS alias_values
CALL {
    WITH alias_values
    UNWIND alias_values AS alias
    WITH DISTINCT alias
    ORDER BY alias
    RETURN collect(alias) AS merged_aliases
}
SET entity.aliases = merged_aliases
REMOVE entity._alias_merge_revision
RETURN entity.entity_id AS entity_id
""".strip()

_UPSERT_ENTITY_EVIDENCE_QUERY = """
MATCH (entity:KnowledgeEntity {entity_id: $entity_id})
UNWIND $provenance AS evidence_input
MERGE (evidence:KnowledgeEvidence {evidence_id: evidence_input.evidence_id})
ON CREATE SET evidence.schema_version = $schema_version,
              evidence.document_id = evidence_input.document_id,
              evidence.knowledge_base_id = evidence_input.knowledge_base_id,
              evidence.chunk_id = evidence_input.chunk_id,
              evidence.page_start = evidence_input.page_start,
              evidence.page_end = evidence_input.page_end,
              evidence.source_block_ids = evidence_input.source_block_ids,
              evidence.section_path = evidence_input.section_path,
              evidence.extraction_provenance_json = evidence_input.extraction_provenance_json
MERGE (entity)-[:SUPPORTED_BY]->(evidence)
""".strip()

_UPSERT_RELATION_QUERY = """
MATCH (source:KnowledgeEntity {
    entity_id: $source_entity_id,
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})
MATCH (target:KnowledgeEntity {
    entity_id: $target_entity_id,
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})
MERGE (relation:KnowledgeRelation {relation_id: $relation_id})
ON CREATE SET relation.schema_version = $schema_version,
              relation.knowledge_base_id = $knowledge_base_id,
              relation.document_id = $document_id,
              relation.source_entity_id = $source_entity_id,
              relation.target_entity_id = $target_entity_id,
              relation.relation_type = $relation_type
MERGE (source)-[:SOURCE_OF]->(relation)
MERGE (relation)-[:TARGET_OF]->(target)
RETURN relation.relation_id AS relation_id
""".strip()

_UPSERT_RELATION_EVIDENCE_QUERY = """
MATCH (source:KnowledgeEntity {
    entity_id: $source_entity_id,
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})
MATCH (target:KnowledgeEntity {
    entity_id: $target_entity_id,
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})
MATCH (relation:KnowledgeRelation {relation_id: $relation_id})
UNWIND $provenance AS evidence_input
MERGE (evidence:KnowledgeEvidence {evidence_id: evidence_input.evidence_id})
ON CREATE SET evidence.schema_version = $schema_version,
              evidence.document_id = evidence_input.document_id,
              evidence.knowledge_base_id = evidence_input.knowledge_base_id,
              evidence.chunk_id = evidence_input.chunk_id,
              evidence.page_start = evidence_input.page_start,
              evidence.page_end = evidence_input.page_end,
              evidence.source_block_ids = evidence_input.source_block_ids,
              evidence.section_path = evidence_input.section_path,
              evidence.extraction_provenance_json = evidence_input.extraction_provenance_json
MERGE (relation)-[:SUPPORTED_BY]->(evidence)
MERGE (source)-[:SUPPORTED_BY]->(evidence)
MERGE (target)-[:SUPPORTED_BY]->(evidence)
""".strip()

_GET_ENTITY_QUERY = """
MATCH (entity:KnowledgeEntity {entity_id: $entity_id})
OPTIONAL MATCH (entity)-[:SUPPORTED_BY]->(evidence:KnowledgeEvidence)
RETURN properties(entity) AS entity,
       collect(CASE WHEN evidence IS NULL THEN null ELSE properties(evidence) END) AS provenance
""".strip()

_GET_RELATION_QUERY = """
MATCH (relation:KnowledgeRelation {relation_id: $relation_id})
OPTIONAL MATCH (relation)-[:SUPPORTED_BY]->(evidence:KnowledgeEvidence)
RETURN properties(relation) AS relation,
       collect(CASE WHEN evidence IS NULL THEN null ELSE properties(evidence) END) AS provenance
""".strip()

_COUNT_DOCUMENT_EVIDENCE_QUERY = """
MATCH (evidence:KnowledgeEvidence {document_id: $document_id})
RETURN count(evidence) AS count
""".strip()

_DELETE_DOCUMENT_EVIDENCE_QUERY = """
MATCH (evidence:KnowledgeEvidence {document_id: $document_id})
DETACH DELETE evidence
""".strip()

_COUNT_ORPHAN_RELATIONS_QUERY = """
MATCH (relation:KnowledgeRelation)
WHERE NOT (relation)-[:SUPPORTED_BY]->()
RETURN count(relation) AS count
""".strip()

_DELETE_ORPHAN_RELATIONS_QUERY = """
MATCH (relation:KnowledgeRelation)
WHERE NOT (relation)-[:SUPPORTED_BY]->()
DETACH DELETE relation
""".strip()

_COUNT_ORPHAN_ENTITIES_QUERY = """
MATCH (entity:KnowledgeEntity)
WHERE NOT (entity)-[:SUPPORTED_BY]->()
RETURN count(entity) AS count
""".strip()

_DELETE_ORPHAN_ENTITIES_QUERY = """
MATCH (entity:KnowledgeEntity)
WHERE NOT (entity)-[:SUPPORTED_BY]->()
DETACH DELETE entity
""".strip()

_UPSERT_CANONICAL_ENTITY_QUERY = """
MERGE (canonical:CanonicalEntity {canonical_entity_id: $canonical_entity_id})
ON CREATE SET canonical.schema_version = $schema_version,
              canonical.knowledge_base_id = $knowledge_base_id,
              canonical.canonical_name = $canonical_name,
              canonical.canonical_name_normalized = $canonical_name_normalized,
              canonical.entity_type = $entity_type,
              canonical.entity_type_normalized = $entity_type_normalized,
              canonical.aliases = []
WITH canonical
WHERE canonical.knowledge_base_id = $knowledge_base_id
  AND canonical.entity_type_normalized = $entity_type_normalized
SET canonical._alias_merge_revision = coalesce(canonical._alias_merge_revision, 0) + 1
WITH canonical, coalesce(canonical.aliases, []) + $aliases AS alias_values
CALL {
    WITH alias_values
    UNWIND alias_values AS alias
    WITH DISTINCT alias
    ORDER BY alias
    RETURN collect(alias) AS merged_aliases
}
SET canonical.aliases = merged_aliases,
    canonical.canonical_name = $canonical_name,
    canonical.canonical_name_normalized = $canonical_name_normalized
REMOVE canonical._alias_merge_revision
REMOVE canonical.anchor_local_entity_id
RETURN canonical.canonical_entity_id AS canonical_entity_id
""".strip()

_RECONCILE_CURRENT_LINK_DECISION_QUERY = """
MATCH (pair:KnowledgeLinkPair {
    link_pair_id: $link_pair_id,
    knowledge_base_id: $knowledge_base_id
})
OPTIONAL MATCH (pair)-[current:CURRENT_DECISION]->(old:KnowledgeLinkDecision)
DELETE current
WITH pair, old
OPTIONAL MATCH ()-[membership:CANONICAL_MEMBER_OF]->()
WHERE old IS NOT NULL
  AND old.link_decision_id IN coalesce(
      membership.decision_ids,
      CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
  )
WITH pair, old, collect(DISTINCT membership) AS memberships
FOREACH (membership IN memberships |
    SET membership.decision_ids = [
        decision_id IN coalesce(
            membership.decision_ids,
            CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
        )
        WHERE decision_id <> old.link_decision_id
    ]
)
WITH pair, old, memberships
FOREACH (membership IN memberships |
    FOREACH (_ IN CASE
        WHEN size(coalesce(
            membership.decision_ids,
            CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
        )) = 0 THEN [1]
        ELSE []
    END | DELETE membership)
)
WITH pair, old
OPTIONAL MATCH (old)-[support:DECIDES_CANONICAL]->(canonical:CanonicalEntity)
WHERE old IS NOT NULL
  AND NOT EXISTS {
      MATCH ()-[membership:CANONICAL_MEMBER_OF]->(canonical)
      WHERE old.link_decision_id IN coalesce(
          membership.decision_ids,
          CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
      )
  }
DELETE support
RETURN pair.link_pair_id AS link_pair_id
""".strip()

_UPSERT_LINK_DECISION_QUERY = """
MATCH (a:KnowledgeEntity {
    entity_id: $local_entity_a_id,
    knowledge_base_id: $knowledge_base_id
})
MATCH (b:KnowledgeEntity {
    entity_id: $local_entity_b_id,
    knowledge_base_id: $knowledge_base_id
})
WHERE ($decision <> 'LINK' OR a.entity_type_normalized = b.entity_type_normalized)
  AND EXISTS { MATCH (a)-[:SUPPORTED_BY]->(:KnowledgeEvidence) }
  AND EXISTS { MATCH (b)-[:SUPPORTED_BY]->(:KnowledgeEvidence) }
MERGE (pair:KnowledgeLinkPair {link_pair_id: $link_pair_id})
ON CREATE SET pair.schema_version = $schema_version,
              pair.knowledge_base_id = $knowledge_base_id,
              pair.local_entity_a_id = $local_entity_a_id,
              pair.local_entity_b_id = $local_entity_b_id
WITH a, b, pair
WHERE pair.knowledge_base_id = $knowledge_base_id
  AND pair.local_entity_a_id = $local_entity_a_id
  AND pair.local_entity_b_id = $local_entity_b_id
MERGE (decision:KnowledgeLinkDecision {link_decision_id: $link_decision_id})
ON CREATE SET decision.schema_version = $schema_version,
              decision.link_pair_id = $link_pair_id,
              decision.run_id = $run_id,
              decision.knowledge_base_id = $knowledge_base_id,
              decision.local_entity_a_id = $local_entity_a_id,
              decision.local_entity_b_id = $local_entity_b_id,
              decision.decision = $decision,
              decision.method = $method,
              decision.confidence = $confidence,
              decision.reason = $reason,
              decision.created_at = $created_at,
              decision.candidate_signals_json = $candidate_signals_json,
              decision.provider = $provider,
              decision.model = $model,
              decision.prompt_version = $prompt_version,
              decision.decision_fingerprint = $decision_fingerprint
WITH a, b, pair, decision
WHERE decision.schema_version = $schema_version
  AND decision.link_pair_id = $link_pair_id
  AND decision.run_id = $run_id
  AND decision.knowledge_base_id = $knowledge_base_id
  AND decision.local_entity_a_id = $local_entity_a_id
  AND decision.local_entity_b_id = $local_entity_b_id
  AND decision.decision_fingerprint = $decision_fingerprint
MERGE (a)-[:HAS_LINK_DECISION]->(decision)
MERGE (b)-[:HAS_LINK_DECISION]->(decision)
MERGE (pair)-[:CURRENT_DECISION]->(decision)
RETURN decision.link_decision_id AS link_decision_id
""".strip()

_UPSERT_CANONICAL_LINK_QUERY = """
MATCH (local:KnowledgeEntity {
    entity_id: $local_entity_id,
    knowledge_base_id: $knowledge_base_id
})
MATCH (canonical:CanonicalEntity {
    canonical_entity_id: $canonical_entity_id,
    knowledge_base_id: $knowledge_base_id
})
MATCH (decision:KnowledgeLinkDecision {
    link_decision_id: $primary_decision_id,
    knowledge_base_id: $knowledge_base_id
})
WHERE decision.decision = 'LINK'
  AND (decision.local_entity_a_id = $local_entity_id
       OR decision.local_entity_b_id = $local_entity_id)
  AND canonical.entity_type_normalized = local.entity_type_normalized
  AND EXISTS { MATCH (local)-[:SUPPORTED_BY]->(:KnowledgeEvidence) }
WITH local, canonical, decision
SET local._canonical_link_merge_revision =
    coalesce(local._canonical_link_merge_revision, 0) + 1
WITH local, canonical, decision
OPTIONAL MATCH (local)-[existing:CANONICAL_MEMBER_OF]->(other:CanonicalEntity)
WHERE existing.link_id <> $link_id
WITH local, canonical, decision, count(existing) AS conflicting_memberships
WHERE conflicting_memberships = 0
MERGE (local)-[membership:CANONICAL_MEMBER_OF {link_id: $link_id}]->(canonical)
SET membership.schema_version = $schema_version,
    membership.knowledge_base_id = $knowledge_base_id,
    membership.local_entity_id = $local_entity_id,
    membership.canonical_entity_id = $canonical_entity_id,
    membership.primary_decision_id = $primary_decision_id,
    membership.decision_ids = reduce(
        merged = [],
        decision_id IN (
            coalesce(
                membership.decision_ids,
                CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
            ) + $supporting_decision_ids
        ) |
        CASE
            WHEN decision_id IN merged THEN merged
            ELSE merged + decision_id
        END
    ),
    membership.method = $method,
    membership.confidence = $confidence,
    membership.reason = $reason,
    membership.candidate_signals_json = $candidate_signals_json,
    membership.provider = $provider,
    membership.model = $model,
    membership.prompt_version = $prompt_version
REMOVE local._canonical_link_merge_revision, membership.decision_id
WITH local, canonical, membership
UNWIND $supporting_decision_ids AS supporting_decision_id
MATCH (supporting_decision:KnowledgeLinkDecision {
    link_decision_id: supporting_decision_id,
    knowledge_base_id: $knowledge_base_id
})
WHERE supporting_decision.decision = 'LINK'
  AND (
      supporting_decision.local_entity_a_id = $local_entity_id
      OR supporting_decision.local_entity_b_id = $local_entity_id
  )
MERGE (supporting_decision)-[:DECIDES_CANONICAL]->(canonical)
RETURN membership.link_id AS link_id
""".strip()

_REPAIR_CANONICAL_MEMBERSHIP_QUERY = """
MATCH (local:KnowledgeEntity {
    knowledge_base_id: $knowledge_base_id
})-[membership:CANONICAL_MEMBER_OF]->(canonical:CanonicalEntity {
    knowledge_base_id: $knowledge_base_id
})
WHERE size(coalesce(
    membership.decision_ids,
    CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
)) > 0
UNWIND coalesce(
    membership.decision_ids,
    CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
) AS decision_id
WITH membership, decision_id
ORDER BY decision_id
WITH membership, collect(decision_id) AS sorted_decision_ids
MATCH (primary:KnowledgeLinkDecision {
    link_decision_id: sorted_decision_ids[0],
    knowledge_base_id: $knowledge_base_id
})
WHERE primary.decision = 'LINK'
SET membership.decision_ids = sorted_decision_ids,
    membership.primary_decision_id = sorted_decision_ids[0],
    membership.method = primary.method,
    membership.confidence = primary.confidence,
    membership.reason = primary.reason,
    membership.candidate_signals_json = primary.candidate_signals_json,
    membership.provider = primary.provider,
    membership.model = primary.model,
    membership.prompt_version = primary.prompt_version
REMOVE membership.decision_id
RETURN count(membership) AS count
""".strip()

_GET_CANONICAL_ENTITY_QUERY = """
MATCH (canonical:CanonicalEntity {canonical_entity_id: $canonical_entity_id})
RETURN properties(canonical) AS canonical
""".strip()

_LIST_DOCUMENT_LOCAL_ENTITY_IDS_QUERY = """
MATCH (entity:KnowledgeEntity {
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})
RETURN collect(entity.entity_id) AS entity_ids
""".strip()

_COUNT_DOCUMENT_CANONICAL_LINKS_QUERY = """
MATCH (local:KnowledgeEntity {
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})-[membership:CANONICAL_MEMBER_OF]->(:CanonicalEntity)
RETURN count(membership) AS count
""".strip()

_DELETE_DOCUMENT_CANONICAL_LINKS_QUERY = """
MATCH (local:KnowledgeEntity {
    knowledge_base_id: $knowledge_base_id,
    document_id: $document_id
})-[membership:CANONICAL_MEMBER_OF]->(:CanonicalEntity)
DELETE membership
""".strip()

_DELETE_DOCUMENT_LINK_DECISIONS_QUERY = """
MATCH (decision:KnowledgeLinkDecision {knowledge_base_id: $knowledge_base_id})
WHERE decision.local_entity_a_id IN $entity_ids
   OR decision.local_entity_b_id IN $entity_ids
OPTIONAL MATCH (pair:KnowledgeLinkPair {
    link_pair_id: decision.link_pair_id,
    knowledge_base_id: $knowledge_base_id
})-[current:CURRENT_DECISION]->(decision)
DELETE current
WITH collect(DISTINCT decision) AS decisions
UNWIND decisions AS decision
OPTIONAL MATCH ()-[membership:CANONICAL_MEMBER_OF]->()
WHERE decision.link_decision_id IN coalesce(
    membership.decision_ids,
    CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
)
WITH decision, collect(DISTINCT membership) AS memberships
FOREACH (membership IN memberships |
    SET membership.decision_ids = [
        decision_id IN coalesce(
            membership.decision_ids,
            CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
        )
        WHERE decision_id <> decision.link_decision_id
    ]
)
WITH decision, memberships
FOREACH (membership IN memberships |
    FOREACH (_ IN CASE
        WHEN size(coalesce(
            membership.decision_ids,
            CASE WHEN membership.decision_id IS NULL THEN [] ELSE [membership.decision_id] END
        )) = 0 THEN [1]
        ELSE []
    END | DELETE membership)
)
DETACH DELETE decision
RETURN count(decision) AS count
""".strip()

_DELETE_DOCUMENT_LINK_PAIRS_QUERY = """
MATCH (pair:KnowledgeLinkPair {knowledge_base_id: $knowledge_base_id})
WHERE pair.local_entity_a_id IN $entity_ids
   OR pair.local_entity_b_id IN $entity_ids
DETACH DELETE pair
RETURN count(pair) AS count
""".strip()

_COUNT_ORPHAN_CANONICAL_ENTITIES_QUERY = """
MATCH (canonical:CanonicalEntity {knowledge_base_id: $knowledge_base_id})
WHERE NOT (canonical)<-[:CANONICAL_MEMBER_OF]-()
RETURN count(canonical) AS count
""".strip()

_DELETE_ORPHAN_CANONICAL_ENTITIES_QUERY = """
MATCH (canonical:CanonicalEntity {knowledge_base_id: $knowledge_base_id})
WHERE NOT (canonical)<-[:CANONICAL_MEMBER_OF]-()
DETACH DELETE canonical
""".strip()


def _record_value(record: Any, key: str) -> Any:
    """Read a Neo4j record or a small test-double record consistently."""

    try:
        return record[key]
    except (KeyError, IndexError, TypeError):
        return None


def _single_or_none(result: Any) -> Any | None:
    record = result.single()
    return record


def _extraction_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _provenance_parameters(values: Sequence[GraphProvenance]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": evidence_id_for(value),
            "document_id": str(value.document_id),
            "knowledge_base_id": str(value.knowledge_base_id),
            "chunk_id": value.chunk_id,
            "page_start": value.page_start,
            "page_end": value.page_end,
            "source_block_ids": list(value.source_block_ids),
            "section_path": list(value.section_path),
            "extraction_provenance_json": _extraction_json(value.extraction_provenance),
        }
        for value in values
    ]


def _entity_parameters(entity: GraphEntity) -> dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "knowledge_base_id": str(entity.knowledge_base_id),
        "document_id": str(entity.document_id),
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "entity_type_normalized": normalize_linking_label(entity.entity_type),
        "aliases": list(entity.aliases),
        "provenance": _provenance_parameters(entity.provenance),
    }


def _relation_parameters(relation: GraphRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "source_entity_id": relation.source_entity_id,
        "target_entity_id": relation.target_entity_id,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "knowledge_base_id": str(relation.knowledge_base_id),
        "document_id": str(relation.document_id),
        "relation_type": relation.relation_type,
        "provenance": _provenance_parameters(relation.provenance),
    }


def _canonical_entity_parameters(entity: CanonicalEntity) -> dict[str, Any]:
    return {
        "canonical_entity_id": entity.canonical_entity_id,
        "schema_version": entity.schema_version,
        "knowledge_base_id": str(entity.knowledge_base_id),
        "canonical_name": entity.canonical_name,
        "canonical_name_normalized": normalize_linking_label(entity.canonical_name),
        "entity_type": entity.entity_type,
        "entity_type_normalized": normalize_linking_label(entity.entity_type),
        "aliases": list(entity.aliases),
    }


def _decision_fingerprint(decision: EntityLinkDecision) -> str:
    serialized = json.dumps(
        decision.model_dump(mode="json", exclude={"created_at"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _link_decision_parameters(decision: EntityLinkDecision) -> dict[str, Any]:
    return {
        "link_decision_id": decision.decision_id,
        "link_pair_id": decision.link_pair_id,
        "run_id": str(decision.run_id),
        "schema_version": decision.schema_version,
        "knowledge_base_id": str(decision.knowledge_base_id),
        "local_entity_a_id": decision.local_entity_a_id,
        "local_entity_b_id": decision.local_entity_b_id,
        "decision": decision.decision,
        "method": decision.method,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "created_at": decision.created_at.isoformat(),
        "candidate_signals_json": json.dumps(
            decision.candidate_signals.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "provider": decision.provider,
        "model": decision.model,
        "prompt_version": decision.prompt_version,
        "decision_fingerprint": _decision_fingerprint(decision),
    }


def _canonical_link_parameters(mapping: EntityCanonicalLink) -> dict[str, Any]:
    return {
        "link_id": mapping.link_id,
        "schema_version": mapping.schema_version,
        "knowledge_base_id": str(mapping.knowledge_base_id),
        "local_entity_id": mapping.local_entity_id,
        "canonical_entity_id": mapping.canonical_entity_id,
        "primary_decision_id": mapping.decision_id,
        "supporting_decision_ids": list(mapping.supporting_decision_ids),
        "method": mapping.method,
        "confidence": mapping.confidence,
        "reason": mapping.reason,
        "candidate_signals_json": json.dumps(
            mapping.candidate_signals.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "provider": mapping.provider,
        "model": mapping.model,
        "prompt_version": mapping.prompt_version,
    }


def _provenance_from_properties(values: Any) -> list[GraphProvenance]:
    if not isinstance(values, list):
        return []
    allowed = set(GraphProvenance.model_fields)
    parsed: list[GraphProvenance] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        payload = {key: item for key, item in value.items() if key in allowed}
        extraction = value.get("extraction_provenance_json")
        if extraction:
            try:
                payload["extraction_provenance"] = json.loads(extraction)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise GraphStoreError("Neo4j returned invalid extraction provenance") from error
        parsed.append(GraphProvenance.model_validate(payload))
    return parsed


def _entity_from_record(record: Any) -> GraphEntity:
    properties = _record_value(record, "entity")
    if not isinstance(properties, dict):
        raise GraphStoreError("Neo4j returned malformed entity data")
    payload = {key: value for key, value in properties.items() if key in GraphEntity.model_fields}
    payload["provenance"] = _provenance_from_properties(_record_value(record, "provenance"))
    try:
        return GraphEntity.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise GraphStoreError("Neo4j returned invalid entity data") from error


def _relation_from_record(record: Any) -> GraphRelation:
    properties = _record_value(record, "relation")
    if not isinstance(properties, dict):
        raise GraphStoreError("Neo4j returned malformed relation data")
    payload = dict(properties)
    payload["provenance"] = _provenance_from_properties(_record_value(record, "provenance"))
    try:
        return GraphRelation.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise GraphStoreError("Neo4j returned invalid relation data") from error


class Neo4jGraphStore:
    """Synchronous Neo4j adapter used by developer workflows and future jobs."""

    def __init__(self, settings: Settings, *, driver: Any | None = None) -> None:
        self.settings = settings
        self._driver = driver

    def _get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        password = self.settings.neo4j_password
        if password is None or not password.get_secret_value():
            raise GraphStoreError("Neo4j password is not configured")
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, password.get_secret_value()),
                connection_timeout=self.settings.neo4j_timeout_seconds,
            )
        except ImportError as error:
            raise GraphStoreError("Neo4j dependency is not installed") from error
        except Exception as error:
            raise GraphStoreError("Neo4j driver could not be created") from error
        return self._driver

    def close(self) -> None:
        """Close the owned or injected driver without exposing driver errors."""

        if self._driver is not None and hasattr(self._driver, "close"):
            self._driver.close()

    def _read(self, work: Callable[[Any], _T]) -> _T:
        try:
            with self._get_driver().session(database=self.settings.neo4j_database) as session:
                return work(session)
        except GraphStoreError:
            raise
        except Exception as error:
            raise GraphStoreError("Neo4j read operation failed") from error

    def _write(self, work: Callable[[Any], _T]) -> _T:
        try:
            with self._get_driver().session(database=self.settings.neo4j_database) as session:
                return session.execute_write(work)
        except GraphStoreError:
            raise
        except Exception as error:
            raise GraphStoreError("Neo4j write operation failed") from error

    def readiness(self) -> Neo4jReadiness:
        """Check connectivity only; schema creation remains an explicit operation."""

        try:

            def ping(session: Any) -> int:
                record = _single_or_none(session.run("RETURN 1 AS ok"))
                return int(_record_value(record, "ok")) if record is not None else 0

            result = self._read(ping)
            if result != 1:
                raise GraphStoreError("Neo4j ping returned an invalid result")
        except GraphStoreError:
            return Neo4jReadiness(
                status="unavailable",
                database=self.settings.neo4j_database,
                error="Neo4j is not reachable",
            )
        return Neo4jReadiness(
            status="ready",
            database=self.settings.neo4j_database,
        )

    def ensure_schema(self) -> Neo4jReadiness:
        """Create schema and backfill legacy entity type labels safely.

        A3.1 nodes may not have ``entity_type_normalized``.  The compatibility
        pass validates every missing value with the same normalization helper
        used for current writes before changing any node.  The update query
        checks the property again, so reruns and concurrent initialization do
        not overwrite an existing normalized value.
        """

        def initialize(session: Any) -> None:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement).consume()
            legacy_updates: list[tuple[str, str]] = []
            for record in session.run(_LIST_LEGACY_ENTITY_TYPES_QUERY):
                entity_id = _record_value(record, "entity_id")
                entity_type = _record_value(record, "entity_type")
                if not isinstance(entity_id, str) or not entity_id.strip():
                    raise GraphStoreError("legacy KnowledgeEntity has an invalid entity_id")
                if not isinstance(entity_type, str):
                    raise GraphStoreError(
                        f"legacy KnowledgeEntity {entity_id} has an invalid entity_type"
                    )
                normalized_type = normalize_linking_label(entity_type)
                if not normalized_type:
                    raise GraphStoreError(
                        f"legacy KnowledgeEntity {entity_id} has an unrecognized entity_type"
                    )
                legacy_updates.append((entity_id, normalized_type))
            for entity_id, normalized_type in legacy_updates:
                session.run(
                    _BACKFILL_ENTITY_TYPE_QUERY,
                    entity_id=entity_id,
                    entity_type_normalized=normalized_type,
                ).consume()

        self._read(initialize)
        return Neo4jReadiness(status="ready", database=self.settings.neo4j_database)

    @staticmethod
    def _validate_entity(entity: GraphEntity) -> GraphEntity:
        """Revalidate mutable model state before it crosses the storage boundary."""

        try:
            validated = GraphEntity.model_validate(entity.model_dump(mode="json"))
        except (TypeError, ValueError) as error:
            raise GraphStoreError("entity payload failed canonical validation") from error
        if validated.entity_id != entity_id_for(
            validated.canonical_name,
            validated.entity_type,
            knowledge_base_id=validated.knowledge_base_id,
            document_id=validated.document_id,
        ):
            raise GraphStoreError("entity_id does not match canonical entity identity")
        return validated

    @staticmethod
    def _validate_relation(relation: GraphRelation) -> GraphRelation:
        """Revalidate mutable model state before it crosses the storage boundary."""

        try:
            validated = GraphRelation.model_validate(relation.model_dump(mode="json"))
        except (TypeError, ValueError) as error:
            raise GraphStoreError("relation payload failed canonical validation") from error
        if validated.relation_id != relation_id_for(
            validated.source_entity_id,
            validated.target_entity_id,
            validated.relation_type,
            knowledge_base_id=validated.knowledge_base_id,
            document_id=validated.document_id,
        ):
            raise GraphStoreError("relation_id does not match canonical relation identity")
        return validated

    @staticmethod
    def _validate_canonical_entity(entity: CanonicalEntity) -> CanonicalEntity:
        """Revalidate a mutable canonical entity before any Neo4j write."""

        try:
            initial_id = getattr(entity, "_initial_canonical_entity_id", None)
            initial_kb = getattr(entity, "_initial_knowledge_base_id", None)
            initial_type = getattr(entity, "_initial_entity_type", None)
            if initial_id is not None and initial_id != entity.canonical_entity_id:
                raise GraphStoreError("canonical_entity_id is immutable")
            if initial_kb is not None and initial_kb != entity.knowledge_base_id:
                raise GraphStoreError("canonical entity knowledge_base_id is immutable")
            if initial_type is not None and initial_type != entity.entity_type:
                raise GraphStoreError("canonical entity_type is immutable")
            validated = CanonicalEntity.model_validate(entity.model_dump(mode="json"))
        except GraphStoreError:
            raise
        except (TypeError, ValueError) as error:
            raise GraphStoreError("canonical entity payload failed validation") from error
        return validated

    @staticmethod
    def _validate_link_decision(decision: EntityLinkDecision) -> EntityLinkDecision:
        """Revalidate a mutable linking decision before any Neo4j write."""

        try:
            validated = EntityLinkDecision.model_validate(decision.model_dump(mode="json"))
        except (TypeError, ValueError) as error:
            raise GraphStoreError("link decision payload failed validation") from error
        if validated.decision_id != link_decision_id_for(
            validated.knowledge_base_id,
            validated.local_entity_a_id,
            validated.local_entity_b_id,
            validated.run_id,
        ):
            raise GraphStoreError("decision_id does not match linking identity")
        return validated

    @staticmethod
    def _validate_canonical_link(mapping: EntityCanonicalLink) -> EntityCanonicalLink:
        """Revalidate a mutable local-to-canonical membership before writing."""

        try:
            validated = EntityCanonicalLink.model_validate(mapping.model_dump(mode="json"))
        except (TypeError, ValueError) as error:
            raise GraphStoreError("canonical link payload failed validation") from error
        if validated.link_id != canonical_link_id_for(
            validated.knowledge_base_id,
            validated.local_entity_id,
            validated.canonical_entity_id,
        ):
            raise GraphStoreError("link_id does not match membership identity")
        return validated

    @classmethod
    def _validate_linking_payload(
        cls,
        canonical_entities: Sequence[CanonicalEntity],
        decisions: Sequence[EntityLinkDecision],
        mappings: Sequence[EntityCanonicalLink],
    ) -> tuple[
        tuple[CanonicalEntity, ...],
        tuple[EntityLinkDecision, ...],
        tuple[EntityCanonicalLink, ...],
    ]:
        validated_canonical = tuple(
            cls._validate_canonical_entity(value) for value in canonical_entities
        )
        validated_decisions = tuple(cls._validate_link_decision(value) for value in decisions)
        validated_mappings = tuple(cls._validate_canonical_link(value) for value in mappings)

        def ensure_unique[T](values: Sequence[T], key: Callable[[T], str], label: str) -> None:
            identifiers = [key(value) for value in values]
            if len(identifiers) != len(set(identifiers)):
                raise GraphStoreError(f"{label} contains duplicate identities")

        ensure_unique(
            validated_canonical,
            lambda value: value.canonical_entity_id,
            "canonical entities",
        )
        ensure_unique(validated_decisions, lambda value: value.decision_id, "link decisions")
        ensure_unique(validated_mappings, lambda value: value.link_id, "canonical links")

        scopes = {
            value.knowledge_base_id
            for value in (*validated_canonical, *validated_decisions, *validated_mappings)
        }
        if len(scopes) > 1:
            raise GraphStoreError("one linking transaction cannot mix knowledge bases")
        decision_by_id = {value.decision_id: value for value in validated_decisions}
        canonical_by_id = {value.canonical_entity_id: value for value in validated_canonical}
        pair_ids = [value.link_pair_id for value in validated_decisions]
        if len(pair_ids) != len(set(pair_ids)):
            raise GraphStoreError(
                "one linking transaction cannot contain multiple decisions for a pair"
            )
        local_to_canonical: dict[str, str] = {}
        canonical_members: dict[str, set[str]] = {}
        decision_canonicals: dict[str, set[str]] = {}
        for decision in validated_decisions:
            if decision.decision != "LINK":
                continue
        for mapping in validated_mappings:
            decision = decision_by_id.get(mapping.decision_id)
            if decision is None or decision.decision != "LINK":
                raise GraphStoreError("canonical link requires a persisted LINK decision")
            if not decision.candidate_signals.compatible_entity_type:
                raise GraphStoreError("incompatible entity types cannot produce a canonical link")
            if mapping.local_entity_id not in {
                decision.local_entity_a_id,
                decision.local_entity_b_id,
            }:
                raise GraphStoreError("canonical link local entity is not in its decision")
            for supporting_id in mapping.supporting_decision_ids:
                supporting = decision_by_id.get(supporting_id)
                if supporting is None or supporting.decision != "LINK":
                    raise GraphStoreError(
                        "canonical link has an unknown or non-LINK support decision"
                    )
                if mapping.local_entity_id not in {
                    supporting.local_entity_a_id,
                    supporting.local_entity_b_id,
                }:
                    raise GraphStoreError(
                        "canonical link support decision does not contain its local entity"
                    )
            if mapping.canonical_entity_id not in canonical_by_id:
                raise GraphStoreError("canonical link references an unknown canonical entity")
            if (
                mapping.method != decision.method
                or mapping.confidence != decision.confidence
                or mapping.reason != decision.reason
                or mapping.candidate_signals != decision.candidate_signals
                or mapping.provider != decision.provider
                or mapping.model != decision.model
                or mapping.prompt_version != decision.prompt_version
            ):
                raise GraphStoreError("canonical link audit fields must match its LINK decision")
            previous_canonical = local_to_canonical.setdefault(
                mapping.local_entity_id,
                mapping.canonical_entity_id,
            )
            if previous_canonical != mapping.canonical_entity_id:
                raise GraphStoreError("one local entity cannot map to multiple canonical entities")
            canonical_members.setdefault(mapping.canonical_entity_id, set()).add(
                mapping.local_entity_id
            )
            for supporting_id in mapping.supporting_decision_ids:
                decision_canonicals.setdefault(supporting_id, set()).add(
                    mapping.canonical_entity_id
                )
        for decision in validated_decisions:
            if decision.decision == "LINK":
                supported_canonicals = decision_canonicals.get(decision.decision_id, set())
                endpoint_canonicals = {
                    local_to_canonical.get(decision.local_entity_a_id),
                    local_to_canonical.get(decision.local_entity_b_id),
                }
                if None in endpoint_canonicals or len(endpoint_canonicals) != 1:
                    raise GraphStoreError(
                        "each LINK decision endpoint must share one canonical membership"
                    )
                if supported_canonicals != endpoint_canonicals:
                    raise GraphStoreError(
                        "each LINK decision must support exactly one canonical entity"
                    )
        mapping_canonical_ids = {value.canonical_entity_id for value in validated_mappings}
        for canonical in validated_canonical:
            if canonical.canonical_entity_id not in mapping_canonical_ids:
                raise GraphStoreError("canonical entity must have at least one local membership")
            members = canonical_members.get(canonical.canonical_entity_id, set())
            if len(members) < 2:
                raise GraphStoreError("canonical entity must have at least two local memberships")
        return validated_canonical, validated_decisions, validated_mappings

    def upsert_entity(self, entity: GraphEntity) -> GraphEntity:
        """MERGE one entity and its evidence, without creating provenance-free facts."""

        entity = self._validate_entity(entity)
        params = _entity_parameters(entity)

        def write(tx: Any) -> GraphEntity:
            record = _single_or_none(tx.run(_UPSERT_ENTITY_QUERY, **params))
            if record is None:
                raise GraphStoreError("Neo4j did not create the entity")
            tx.run(_UPSERT_ENTITY_EVIDENCE_QUERY, **params).consume()
            return entity

        return self._write(write)

    def upsert_relation(self, relation: GraphRelation) -> GraphRelation:
        """MERGE one directed relation and require both endpoint entities first."""

        relation = self._validate_relation(relation)
        params = _relation_parameters(relation)

        def write(tx: Any) -> GraphRelation:
            record = _single_or_none(tx.run(_UPSERT_RELATION_QUERY, **params))
            if record is None:
                raise GraphStoreError("relation endpoints must be upserted before the relation")
            tx.run(_UPSERT_RELATION_EVIDENCE_QUERY, **params).consume()
            return relation

        return self._write(write)

    def upsert_extraction(
        self,
        entities: Sequence[GraphEntity],
        relations: Sequence[GraphRelation],
    ) -> GraphUpsertResult:
        """Upsert one validated chunk extraction in one managed Neo4j transaction.

        All objects are validated before opening a session.  Neo4j rolls back the
        complete managed transaction if an endpoint or any write fails, so a
        chunk cannot be left half-written by this method.
        """

        validated_entities = tuple(self._validate_entity(entity) for entity in entities)
        validated_relations = tuple(self._validate_relation(relation) for relation in relations)
        entity_params = tuple(_entity_parameters(entity) for entity in validated_entities)
        relation_params = tuple(_relation_parameters(relation) for relation in validated_relations)

        def write(tx: Any) -> GraphUpsertResult:
            for params in entity_params:
                record = _single_or_none(tx.run(_UPSERT_ENTITY_QUERY, **params))
                if record is None:
                    raise GraphStoreError("Neo4j did not create the entity")
                tx.run(_UPSERT_ENTITY_EVIDENCE_QUERY, **params).consume()
            for params in relation_params:
                record = _single_or_none(tx.run(_UPSERT_RELATION_QUERY, **params))
                if record is None:
                    raise GraphStoreError("relation endpoints must be upserted before the relation")
                tx.run(_UPSERT_RELATION_EVIDENCE_QUERY, **params).consume()
            return GraphUpsertResult(
                entity_count=len(entity_params),
                relation_count=len(relation_params),
            )

        return self._write(write)

    def upsert_linking_result(
        self,
        canonical_entities: Sequence[CanonicalEntity],
        decisions: Sequence[EntityLinkDecision],
        mappings: Sequence[EntityCanonicalLink],
    ) -> GraphLinkUpsertResult:
        """Persist one linking plan in a single Neo4j transaction.

        Local A3.1 entities and their evidence are only matched, never changed.
        A missing local endpoint, stale identity, or conflicting membership
        aborts the managed transaction before the plan is reported as stored.
        """

        validated_canonical, validated_decisions, validated_mappings = (
            self._validate_linking_payload(canonical_entities, decisions, mappings)
        )
        canonical_params = tuple(
            _canonical_entity_parameters(value) for value in validated_canonical
        )
        decision_params = tuple(_link_decision_parameters(value) for value in validated_decisions)
        mapping_params = tuple(_canonical_link_parameters(value) for value in validated_mappings)

        def write(tx: Any) -> GraphLinkUpsertResult:
            for params in canonical_params:
                record = _single_or_none(tx.run(_UPSERT_CANONICAL_ENTITY_QUERY, **params))
                if record is None:
                    raise GraphStoreError("canonical entity identity or scope conflicts")
            for params in decision_params:
                tx.run(_RECONCILE_CURRENT_LINK_DECISION_QUERY, **params).consume()
                record = _single_or_none(tx.run(_UPSERT_LINK_DECISION_QUERY, **params))
                if record is None:
                    raise GraphStoreError(
                        "link decision local entities, scope, or historical identity are invalid"
                    )
            for params in mapping_params:
                record = _single_or_none(tx.run(_UPSERT_CANONICAL_LINK_QUERY, **params))
                if record is None:
                    raise GraphStoreError("canonical membership conflicts or endpoints are missing")
            scope_params = decision_params or mapping_params
            if scope_params:
                knowledge_base_value = scope_params[0]["knowledge_base_id"]
                tx.run(
                    _REPAIR_CANONICAL_MEMBERSHIP_QUERY,
                    knowledge_base_id=knowledge_base_value,
                ).consume()
                tx.run(
                    _DELETE_ORPHAN_CANONICAL_ENTITIES_QUERY,
                    knowledge_base_id=knowledge_base_value,
                ).consume()
            return GraphLinkUpsertResult(
                canonical_entity_count=len(canonical_params),
                decision_count=len(decision_params),
                mapping_count=len(mapping_params),
            )

        return self._write(write)

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Return one entity with its stored source evidence, if present."""

        def read(session: Any) -> GraphEntity | None:
            record = _single_or_none(session.run(_GET_ENTITY_QUERY, entity_id=entity_id))
            return _entity_from_record(record) if record is not None else None

        return self._read(read)

    def get_relation(self, relation_id: str) -> GraphRelation | None:
        """Return one relation with its stored source evidence, if present."""

        def read(session: Any) -> GraphRelation | None:
            record = _single_or_none(session.run(_GET_RELATION_QUERY, relation_id=relation_id))
            return _relation_from_record(record) if record is not None else None

        return self._read(read)

    def get_canonical_entity(self, canonical_entity_id: str) -> CanonicalEntity | None:
        """Return one canonical entity without exposing local storage details."""

        def read(session: Any) -> CanonicalEntity | None:
            record = _single_or_none(
                session.run(
                    _GET_CANONICAL_ENTITY_QUERY,
                    canonical_entity_id=canonical_entity_id,
                )
            )
            if record is None:
                return None
            properties = _record_value(record, "canonical")
            if not isinstance(properties, dict):
                raise GraphStoreError("Neo4j returned malformed canonical entity data")
            try:
                payload = {
                    key: value
                    for key, value in properties.items()
                    if key in CanonicalEntity.model_fields
                }
                return CanonicalEntity.model_validate(payload)
            except (TypeError, ValueError) as error:
                raise GraphStoreError("Neo4j returned invalid canonical entity data") from error

        return self._read(read)

    def delete_document_links(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> GraphLinkDeleteResult:
        """Remove only A3.3 records attached to one local document.

        A3.1 local entities, relations, and evidence are intentionally untouched.
        Historical decisions are removed only when their local endpoints belong to
        this document; memberships supported by another LINK decision remain.
        """

        params = {
            "knowledge_base_id": str(knowledge_base_id),
            "document_id": str(document_id),
        }

        def write(tx: Any) -> GraphLinkDeleteResult:
            entity_ids = self._document_local_entity_ids(tx, params)
            mapping_count, decision_count, canonical_count = self._delete_document_linking_state(
                tx,
                params,
                entity_ids,
            )
            return GraphLinkDeleteResult(
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                decision_count=decision_count,
                mapping_count=mapping_count,
                canonical_entity_count=canonical_count,
            )

        return self._write(write)

    @staticmethod
    def _document_local_entity_ids(tx: Any, params: dict[str, Any]) -> list[str]:
        entity_record = _single_or_none(tx.run(_LIST_DOCUMENT_LOCAL_ENTITY_IDS_QUERY, **params))
        raw_ids = _record_value(entity_record, "entity_ids") if entity_record else []
        return [str(value) for value in raw_ids] if isinstance(raw_ids, list) else []

    @staticmethod
    def _delete_document_linking_state(
        tx: Any,
        params: dict[str, Any],
        entity_ids: Sequence[str],
    ) -> tuple[int, int, int]:
        """Remove current linking state while preserving independently supported data."""

        mapping_record = _single_or_none(tx.run(_COUNT_DOCUMENT_CANONICAL_LINKS_QUERY, **params))
        mapping_count = int(_record_value(mapping_record, "count") or 0)
        count_params = {**params, "entity_ids": list(entity_ids)}
        deleted_decision_record = _single_or_none(
            tx.run(_DELETE_DOCUMENT_LINK_DECISIONS_QUERY, **count_params)
        )
        decision_count = int(_record_value(deleted_decision_record, "count") or 0)
        tx.run(
            _REPAIR_CANONICAL_MEMBERSHIP_QUERY,
            knowledge_base_id=params["knowledge_base_id"],
        ).consume()
        tx.run(_DELETE_DOCUMENT_LINK_PAIRS_QUERY, **count_params).consume()
        tx.run(_DELETE_DOCUMENT_CANONICAL_LINKS_QUERY, **params).consume()
        orphan_record = _single_or_none(
            tx.run(
                _COUNT_ORPHAN_CANONICAL_ENTITIES_QUERY,
                knowledge_base_id=params["knowledge_base_id"],
            )
        )
        canonical_count = int(_record_value(orphan_record, "count") or 0)
        tx.run(
            _DELETE_ORPHAN_CANONICAL_ENTITIES_QUERY,
            knowledge_base_id=params["knowledge_base_id"],
        ).consume()
        return mapping_count, decision_count, canonical_count

    def delete_document(
        self,
        document_id: UUID,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> GraphDeleteResult:
        """Remove one document's A3.1 and, when scoped, A3.3 graph state.

        The Neo4j writes are atomic within this transaction.  PostgreSQL, files,
        Qdrant, and Neo4j remain separate systems, so document deletion as a
        whole is compensating/non-atomic across systems.
        """

        document_value = str(document_id)

        def write(tx: Any) -> GraphDeleteResult:
            if knowledge_base_id is not None:
                params = {
                    "knowledge_base_id": str(knowledge_base_id),
                    "document_id": document_value,
                }
                entity_ids = self._document_local_entity_ids(tx, params)
                self._delete_document_linking_state(tx, params, entity_ids)

            evidence_record = _single_or_none(
                tx.run(_COUNT_DOCUMENT_EVIDENCE_QUERY, document_id=document_value)
            )
            evidence_count = int(_record_value(evidence_record, "count") or 0)
            tx.run(_DELETE_DOCUMENT_EVIDENCE_QUERY, document_id=document_value).consume()

            relation_record = _single_or_none(tx.run(_COUNT_ORPHAN_RELATIONS_QUERY))
            relation_count = int(_record_value(relation_record, "count") or 0)
            tx.run(_DELETE_ORPHAN_RELATIONS_QUERY).consume()

            entity_record = _single_or_none(tx.run(_COUNT_ORPHAN_ENTITIES_QUERY))
            entity_count = int(_record_value(entity_record, "count") or 0)
            tx.run(_DELETE_ORPHAN_ENTITIES_QUERY).consume()
            return GraphDeleteResult(
                document_id=document_id,
                evidence_count=evidence_count,
                relation_count=relation_count,
                entity_count=entity_count,
            )

        return self._write(write)


__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "NEO4J_CANONICAL_ENTITY_LABEL",
    "NEO4J_ENTITY_LABEL",
    "NEO4J_EVIDENCE_LABEL",
    "NEO4J_LINK_DECISION_LABEL",
    "NEO4J_LINK_PAIR_LABEL",
    "NEO4J_RELATION_LABEL",
    "SCHEMA_STATEMENTS",
    "GraphDeleteResult",
    "GraphLinkDeleteResult",
    "GraphLinkUpsertResult",
    "GraphStoreError",
    "GraphUpsertResult",
    "Neo4jGraphStore",
    "Neo4jReadiness",
]
