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
    NEO4J_LINK_PAIR_LABEL,
    GraphDeleteResult,
    GraphLinkDeleteResult,
    GraphLinkUpsertResult,
    GraphStoreError,
    GraphUpsertResult,
    Neo4jGraphStore,
    Neo4jReadiness,
)

__all__ = [
    "GRAPH_ID_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "NEO4J_LINK_PAIR_LABEL",
    "EntityId",
    "ExtractionProvenance",
    "GraphDeleteResult",
    "GraphEntity",
    "GraphLinkDeleteResult",
    "GraphLinkUpsertResult",
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
