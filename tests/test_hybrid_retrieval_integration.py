from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from knowledge_scope.graph.models import GraphEntity, GraphProvenance, entity_id_for
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.graph.retrieval import GraphRetrievalConfig
from knowledge_scope.graph.retrieval_service import GraphRetrievalService
from knowledge_scope.retrieval.hybrid import HybridRetrievalService
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_VECTOR_DIMENSION,
    ChunkVectorPayload,
    QdrantVectorStore,
    VectorPoint,
    point_id_for_chunk,
)
from knowledge_scope.retrieval.reranking import RerankingService
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.shared.config import get_settings


class _IntegrationEncoder:
    model_id = "integration-encoder"

    def encode_query(self, _query: str) -> list[float]:
        return [1.0, *([0.0] * (QDRANT_VECTOR_DIMENSION - 1))]


class _IntegrationReranker:
    model_id = "integration-reranker"

    def score_pairs(self, _query: str, passages: list[str]) -> list[float]:
        return [1.0] * len(passages)


@pytest.mark.integration
def test_real_qdrant_and_neo4j_hybrid_round_trip() -> None:
    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_HYBRID_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_HYBRID_INTEGRATION=1 to run hybrid integration")

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    collection_name = f"knowledgescope_hybrid_{uuid4().hex[:20]}"
    settings = settings.model_copy(update={"qdrant_collection_name": collection_name})
    knowledge_base_id = uuid4()
    document_id = uuid4()
    chunk_id = f"hybrid-integration-{uuid4().hex}"
    block_id = f"{chunk_id}-block"
    payload = ChunkVectorPayload(
        collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
        chunk_id=chunk_id,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["A3.5 集成测试"],
        content_types=["text"],
        asset_refs=[],
        text="混合检索集成测试正文",
        chunking_config_fingerprint="a" * 64,
        embedding_model="integration-encoder",
        embedding_model_revision="test",
        embedding_config_fingerprint="b" * 64,
    )
    point = VectorPoint(
        point_id=point_id_for_chunk(chunk_id),
        vector=(1.0, *([0.0] * (QDRANT_VECTOR_DIMENSION - 1))),
        payload=payload,
    )
    provenance = GraphProvenance(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["A3.5 集成测试"],
    )
    entity = GraphEntity(
        entity_id=entity_id_for(
            "混合样本",
            "概念",
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        ),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        canonical_name="混合样本",
        entity_type="概念",
        provenance=[provenance],
    )
    qdrant_store = QdrantVectorStore(settings)
    neo4j_store = Neo4jGraphStore(settings)
    try:
        qdrant_store.ensure_collection()
        qdrant_store.replace_document([point])
        neo4j_store.ensure_schema()
        neo4j_store.upsert_entity(entity)
        service = HybridRetrievalService(
            DenseRetrievalService(qdrant_store, _IntegrationEncoder()),
            RerankingService(_IntegrationReranker()),
            GraphRetrievalService(
                neo4j_store,
                config=GraphRetrievalConfig(max_hops=1),
            ),
        )

        result = asyncio.run(service.search("混合样本", knowledge_base_id))

        assert result.vector_status == "success"
        assert result.graph_status == "success"
        assert result.source_distribution == {"vector": 0, "graph": 0, "both": 1}
        assert result.items[0].chunk_id == chunk_id
    finally:
        try:
            neo4j_store.delete_document(document_id, knowledge_base_id=knowledge_base_id)
        finally:
            qdrant_store.delete_document(document_id)
            client = qdrant_store._get_client()
            client.delete_collection(collection_name)
            qdrant_store.close()
            neo4j_store.close()
