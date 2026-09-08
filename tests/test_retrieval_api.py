from __future__ import annotations

from uuid import uuid4

import anyio
import httpx

from knowledge_scope.api.app import create_app
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    QdrantReadiness,
    RetrievedChunk,
)
from knowledge_scope.shared.config import Settings


class _FakeEmbeddingModel:
    model_id = "Qwen/Qwen3-Embedding-0.6B"

    def encode_query(self, _query: str) -> list[float]:
        return [0.0] * 1023 + [1.0]


class _FakeVectorStore:
    collection_name = "knowledgescope_chunks_v1"

    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def readiness(self) -> QdrantReadiness:
        if self.available:
            return QdrantReadiness(
                status="ready",
                collection_name=self.collection_name,
                collection_exists=True,
                vector_dimension=1024,
            )
        return QdrantReadiness(
            status="unavailable",
            collection_name=self.collection_name,
            collection_exists=False,
            error="Qdrant is not reachable",
        )

    def search(self, _vector: list[float], **_filters: object) -> list[RetrievedChunk]:
        document_id = uuid4()
        return [
            RetrievedChunk(
                point_id=uuid4(),
                score=0.98,
                payload=ChunkVectorPayload(
                    collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
                    chunk_id="chunk-1",
                    document_id=document_id,
                    knowledge_base_id=uuid4(),
                    page_start=2,
                    page_end=2,
                    source_block_ids=["p2-b1"],
                    section_path=["章节"],
                    content_types=["text"],
                    asset_refs=[],
                    text="答案正文",
                    chunking_config_fingerprint="a" * 64,
                    embedding_model="Qwen/Qwen3-Embedding-0.6B",
                    embedding_model_revision="revision",
                    embedding_config_fingerprint="b" * 64,
                ),
            )
        ]

    def close(self) -> None:
        return None


def _request(path: str, *, store: _FakeVectorStore) -> httpx.Response:
    application = create_app(
        Settings(_env_file=None, environment="test"),
        vector_store=store,
        embedding_model=_FakeEmbeddingModel(),
    )

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            if path.endswith("health/qdrant"):
                return await client.get(path)
            return await client.post(path, json={"query": "如何理解?", "limit": 3})

    return anyio.run(run)


def test_qdrant_readiness_endpoint_reports_ready_state() -> None:
    response = _request("/api/v1/health/qdrant", store=_FakeVectorStore())

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["vector_dimension"] == 1024


def test_retrieval_endpoint_returns_citation_ready_metadata() -> None:
    response = _request("/api/v1/retrieval/search", store=_FakeVectorStore())

    assert response.status_code == 200
    assert response.json()["items"][0]["chunk_id"] == "chunk-1"
    assert response.json()["items"][0]["source_block_ids"] == ["p2-b1"]
    assert response.json()["items"][0]["text"] == "答案正文"


def test_retrieval_endpoint_reports_qdrant_unavailability() -> None:
    response = _request(
        "/api/v1/health/qdrant",
        store=_FakeVectorStore(available=False),
    )

    assert response.status_code == 503
    assert "Qdrant" in response.json()["detail"]
