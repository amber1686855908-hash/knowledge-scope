"""Provider-independent graph models and the Neo4j infrastructure adapter."""

from .models import (
    GRAPH_ID_VERSION,
    GRAPH_SCHEMA_VERSION,
    EntityId,
    ExtractionProvenance,
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    RelationId,
    canonical_identity_json,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)
from .neo4j import (
    GraphDeleteResult,
    GraphStoreError,
    GraphUpsertResult,
    Neo4jGraphStore,
    Neo4jReadiness,
)

__all__ = [
    "GRAPH_ID_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "EntityId",
    "ExtractionProvenance",
    "GraphDeleteResult",
    "GraphEntity",
    "GraphProvenance",
    "GraphRelation",
    "GraphStoreError",
    "GraphUpsertResult",
    "Neo4jGraphStore",
    "Neo4jReadiness",
    "RelationId",
    "canonical_identity_json",
    "entity_id_for",
    "evidence_id_for",
    "relation_id_for",
]
