"""Small Neo4j adapter for the canonical KnowledgeScope graph schema."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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
)

_UPSERT_ENTITY_QUERY = """
MERGE (entity:KnowledgeEntity {entity_id: $entity_id})
ON CREATE SET entity.schema_version = $schema_version,
              entity.knowledge_base_id = $knowledge_base_id,
              entity.document_id = $document_id,
              entity.canonical_name = $canonical_name,
              entity.entity_type = $entity_type,
              entity.aliases = []
SET entity._alias_merge_revision = coalesce(entity._alias_merge_revision, 0) + 1
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
    payload = dict(properties)
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
        """Create the versioned labels' constraints and lookup indexes idempotently."""

        def initialize(session: Any) -> None:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement).consume()

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

    def delete_document(self, document_id: UUID) -> GraphDeleteResult:
        """Remove one document's evidence and then prune unsupported graph facts."""

        document_value = str(document_id)

        def write(tx: Any) -> GraphDeleteResult:
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
    "NEO4J_ENTITY_LABEL",
    "NEO4J_EVIDENCE_LABEL",
    "NEO4J_RELATION_LABEL",
    "SCHEMA_STATEMENTS",
    "GraphDeleteResult",
    "GraphStoreError",
    "GraphUpsertResult",
    "Neo4jGraphStore",
    "Neo4jReadiness",
]
