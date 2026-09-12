from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from qdrant_client import models

from knowledge_scope.chunking.models import Chunk, ChunkedDocument
from knowledge_scope.retrieval.embedding import embedding_config_fingerprint
from knowledge_scope.retrieval.indexing import (
    IndexingError,
    build_vector_points,
    index_chunk_artifact,
)
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_VECTOR_DIMENSION,
    ChunkVectorPayload,
    CollectionConfigurationError,
    QdrantVectorStore,
    VectorPoint,
    VectorStoreError,
    point_id_for_chunk,
)
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.shared.config import Settings


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * (QDRANT_VECTOR_DIMENSION - 1)


def _payload(document_id: UUID, chunk_id: str, knowledge_base_id: UUID) -> ChunkVectorPayload:
    return ChunkVectorPayload(
        collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
        chunk_id=chunk_id,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["测试章节"],
        content_types=["text"],
        asset_refs=[],
        text=f"文本 {chunk_id}",
        chunking_config_fingerprint="a" * 64,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_model_revision="revision",
        embedding_config_fingerprint="b" * 64,
    )


def _point(document_id: UUID, chunk_id: str, knowledge_base_id: UUID) -> VectorPoint:
    return VectorPoint(
        point_id=point_id_for_chunk(chunk_id),
        vector=_vector(1.0),
        payload=_payload(document_id, chunk_id, knowledge_base_id),
    )


@dataclass
class _FakeRecord:
    id: UUID
    vector: list[float] | None
    payload: dict[str, object] | None


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: dict[str, _FakeRecord] = {}
        self.fail_upsert = False
        self.fail_upsert_once = False
        self.fail_delete = False
        self._info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=models.VectorParams(size=QDRANT_VECTOR_DIMENSION, distance="Cosine")
                )
            )
        )

    def collection_exists(self, _collection_name: str) -> bool:
        return True

    def create_collection(self, **_kwargs: object) -> bool:
        return True

    def get_collection(self, _collection_name: str) -> object:
        return self._info

    def close(self) -> None:
        return None

    def scroll(
        self,
        *,
        scroll_filter: models.Filter | None = None,
        limit: int,
        offset: object = None,
        **_kwargs: object,
    ):
        records = list(self.points.values())
        if scroll_filter is not None:
            document_id = scroll_filter.must[0].match.value
            records = [
                record
                for record in records
                if record.payload and record.payload["document_id"] == document_id
            ]
        start = int(offset or 0)
        page = records[start : start + limit]
        next_offset = start + limit if start + limit < len(records) else None
        return [
            SimpleNamespace(id=record.id, payload=record.payload) for record in page
        ], next_offset

    def retrieve(self, *, ids: list[UUID], **_kwargs: object):
        return [
            SimpleNamespace(
                id=point_id,
                vector=self.points[str(point_id)].vector,
                payload=self.points[str(point_id)].payload,
            )
            for point_id in ids
            if str(point_id) in self.points
        ]

    def upsert(self, *, points: list[models.PointStruct], **_kwargs: object) -> None:
        if self.fail_upsert or self.fail_upsert_once:
            self.fail_upsert_once = False
            raise RuntimeError("simulated upsert failure")
        for point in points:
            self.points[str(point.id)] = _FakeRecord(
                id=UUID(str(point.id)),
                vector=[float(value) for value in point.vector],
                payload=point.payload,
            )

    def delete(self, *, points_selector: models.PointIdsList, **_kwargs: object) -> None:
        if self.fail_delete:
            raise RuntimeError("simulated delete failure")
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)

    def set_payload(
        self,
        *,
        payload: dict[str, object],
        points: models.PointIdsList,
        **_kwargs: object,
    ) -> None:
        for point_id in points.points:
            record = self.points[str(point_id)]
            assert record.payload is not None
            record.payload.update(payload)

    def query_points(
        self,
        *,
        query: list[float],
        query_filter: models.Filter | None,
        limit: int,
        **_kwargs: object,
    ):
        records = list(self.points.values())
        if query_filter is not None:
            for condition in query_filter.must:
                records = [
                    record
                    for record in records
                    if record.payload and record.payload[condition.key] == condition.match.value
                ]
        scored = []
        for record in records:
            score = sum(
                left * right for left, right in zip(query, record.vector or (), strict=True)
            )
            scored.append(SimpleNamespace(id=record.id, score=score, payload=record.payload))
        scored.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])


class _FakeEncoder:
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    model_revision = "revision"
    config_fingerprint = "b" * 64

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(_vector(float(index + 1))) for index, _ in enumerate(texts)]

    def encode_query(self, _query: str) -> list[float]:
        return list(_vector(1.0))


def _chunked_document(document_id: UUID) -> ChunkedDocument:
    return ChunkedDocument(
        document_id=document_id,
        page_count=1,
        config_fingerprint="a" * 64,
        chunks=[
            Chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                ordinal=0,
                text="正文",
                page_start=1,
                page_end=1,
                source_block_ids=["p1-b1"],
                section_path=["章节"],
                content_types=["text"],
                asset_refs=[],
            ),
            Chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                ordinal=1,
                text="第二段",
                page_start=1,
                page_end=1,
                source_block_ids=["p1-b2"],
                section_path=["章节"],
                content_types=["text"],
                asset_refs=[],
            ),
        ],
    )


def test_point_ids_and_embedding_config_are_deterministic() -> None:
    settings = Settings(_env_file=None)

    assert point_id_for_chunk("chunk-1") == point_id_for_chunk("chunk-1")
    assert point_id_for_chunk("chunk-1") != point_id_for_chunk("chunk-2")
    assert len(embedding_config_fingerprint(settings)) == 64


def test_build_vector_points_preserves_chunk_lineage_and_payload() -> None:
    document_id = uuid4()
    knowledge_base_id = uuid4()
    points = build_vector_points(
        _chunked_document(document_id),
        knowledge_base_id=knowledge_base_id,
        embedder=_FakeEncoder(),
    )

    assert len(points) == 2
    assert points[0].payload.document_id == document_id
    assert points[0].payload.knowledge_base_id == knowledge_base_id
    assert points[0].payload.source_block_ids == ["p1-b1"]
    assert points[0].payload.text == "正文"
    assert len(points[0].vector) == QDRANT_VECTOR_DIMENSION


def test_index_chunk_artifact_rejects_a_document_id_mismatch(tmp_path: Path) -> None:
    artifact_document_id = uuid4()
    requested_document_id = uuid4()
    artifact_path = tmp_path / "chunks.json"
    artifact_path.write_text(
        _chunked_document(artifact_document_id).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(IndexingError, match="document ID does not match"):
        index_chunk_artifact(
            artifact_path,
            document_id=requested_document_id,
            knowledge_base_id=uuid4(),
            store=QdrantVectorStore(Settings(_env_file=None), client=_FakeQdrantClient()),
            embedder=_FakeEncoder(),
        )


def test_qdrant_replace_reindex_delete_and_filtered_search() -> None:
    settings = Settings(_env_file=None, qdrant_upsert_batch_size=1)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    knowledge_base_id = uuid4()

    first = store.replace_document(
        [
            _point(document_id, "chunk-1", knowledge_base_id),
            _point(document_id, "chunk-2", knowledge_base_id),
        ]
    )
    assert first.indexed_count == 2
    assert first.removed_stale_count == 0

    replacement = store.replace_document([_point(document_id, "chunk-3", knowledge_base_id)])
    assert replacement.removed_stale_count == 2
    results = store.search(
        _vector(1.0),
        limit=5,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )
    assert [result.payload.chunk_id for result in results] == ["chunk-3"]

    assert store.delete_document(document_id) == 1
    assert store.delete_document(document_id) == 0


def test_qdrant_rejects_incompatible_collection_schema() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    client._info.config.params.vectors = models.VectorParams(size=768, distance="Cosine")

    with pytest.raises(CollectionConfigurationError, match="incompatible vector schema"):
        QdrantVectorStore(settings, client=client).ensure_collection()


def test_qdrant_rejects_a_point_missing_the_collection_schema_version() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    store.replace_document([_point(document_id, "chunk-1", knowledge_base_id)])
    next(iter(client.points.values())).payload.pop("collection_schema_version")

    with pytest.raises(VectorStoreError, match="Qdrant dense search failed"):
        store.search(
            _vector(1.0),
            limit=1,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )


def test_qdrant_failed_replacement_restores_previous_points() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    store.replace_document([_point(document_id, "chunk-1", knowledge_base_id)])
    old_ids = set(client.points)
    client.fail_upsert_once = True

    with pytest.raises(VectorStoreError, match="previous set was restored"):
        store.replace_document([_point(document_id, "chunk-2", knowledge_base_id)])

    assert set(client.points) == old_ids


def test_dense_retrieval_service_passes_filters_and_returns_ranked_items() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    store.replace_document([_point(document_id, "chunk-1", knowledge_base_id)])

    result = DenseRetrievalService(store, _FakeEncoder()).search(
        "查询",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )

    assert result.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert result.items[0].payload.chunk_id == "chunk-1"


def test_qdrant_payload_attribution_preserves_point_and_vector_identity() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    original_kb = uuid4()
    repaired_kb = uuid4()
    store.replace_document([_point(document_id, "chunk-1", original_kb)])
    point_id = next(iter(client.points))
    original = client.points[point_id]
    original_vector = list(original.vector or [])

    metadata = store.list_point_metadata()
    assert metadata[0].point_id == UUID(point_id)
    assert metadata[0].chunk_id == "chunk-1"
    assert metadata[0].knowledge_base_id == original_kb

    store.set_point_knowledge_base_ids((metadata[0].point_id,), repaired_kb)
    repaired = client.points[point_id]
    assert repaired.payload is not None
    assert repaired.payload["knowledge_base_id"] == str(repaired_kb)
    assert repaired.vector == original_vector
    assert repaired.id == original.id
    assert repaired.payload["document_id"] == str(document_id)
    assert repaired.payload["chunk_id"] == "chunk-1"


def test_chunk_payload_lookup_preserves_requested_scope() -> None:
    settings = Settings(_env_file=None)
    client = _FakeQdrantClient()
    store = QdrantVectorStore(settings, client=client)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    store.replace_document([_point(document_id, "chunk-1", knowledge_base_id)])

    payload = store.get_chunk_payload(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="chunk-1",
    )
    assert payload is not None
    assert payload.chunk_id == "chunk-1"
    assert payload.knowledge_base_id == knowledge_base_id
    with pytest.raises(VectorStoreError, match="outside the requested scope"):
        store.get_chunk_payload(
            knowledge_base_id=uuid4(),
            document_id=document_id,
            chunk_id="chunk-1",
        )
    assert (
        store.get_chunk_payload(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            chunk_id="missing",
        )
        is None
    )


@pytest.mark.integration
def test_real_qdrant_contract_round_trip() -> None:
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip(
            "set KNOWLEDGE_SCOPE_RUN_QDRANT_INTEGRATION=1 to run the Qdrant integration test"
        )
    settings = Settings(_env_file=None)
    store = QdrantVectorStore(settings)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    try:
        store.ensure_collection()
        store.replace_document([_point(document_id, "integration-chunk", knowledge_base_id)])
        results = store.search(
            _vector(1.0),
            limit=1,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        assert results[0].payload.chunk_id == "integration-chunk"
    finally:
        store.delete_document(document_id)
        store.close()
