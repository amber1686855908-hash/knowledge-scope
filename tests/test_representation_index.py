from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from knowledge_scope.evidence.lifecycle import assert_evidence_artifact_current
from knowledge_scope.evidence.service import representation_index_payloads
from knowledge_scope.parsing.models import (
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)
from knowledge_scope.retrieval.embedding import embedding_config_fingerprint
from knowledge_scope.retrieval.representation_index import (
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    IndexedRepresentationPayload,
    MultimodalRepresentationRetrievalService,
    QdrantRepresentationStore,
    RepresentationCollectionConfigurationError,
    RepresentationIndexError,
    RepresentationVisibilitySnapshot,
    audit_representation_index,
    build_indexable_evidence,
    build_representation_points,
    index_canonical_document,
    point_id_for_chunk,
    point_id_for_representation,
    representation_collection_fingerprint,
    validate_representation_collection_role,
)
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
FIRST_KB = UUID("22222222-2222-2222-2222-222222222222")
SECOND_KB = UUID("33333333-3333-3333-3333-333333333333")


def _document() -> CanonicalDocument:
    return CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="title-1", reading_order=0, text="设备维护"),
                    TextBlock(
                        block_id="text-1",
                        reading_order=1,
                        text="设备需要定期维护, 温度应保持稳定。",
                    ),
                    ImageBlock(
                        block_id="image-1",
                        reading_order=2,
                        asset_ref="images/maintenance.png",
                    ),
                    TableBlock(
                        block_id="table-1",
                        reading_order=3,
                        markdown="| 项目 | 状态 |\n| --- | --- |\n| 温度 | 正常 |",
                        caption="检查项目表",
                        asset_ref="tables/check.png",
                    ),
                    FormulaBlock(block_id="formula-1", reading_order=4, latex="E = mc^2"),
                ],
            ),
            Page(page_number=2),
        ],
    )


class _DirectionalEncoder:
    model_id = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self) -> None:
        settings = Settings(_env_file=None)
        self.model_revision = settings.embedding_model_revision
        self.config_fingerprint = embedding_config_fingerprint(settings)

    @staticmethod
    def _vector(axis: int) -> list[float]:
        values = [0.0] * QDRANT_VECTOR_DIMENSION
        values[axis] = 1.0
        return values

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(self._axis(text)) for text in texts]

    def encode_query(self, query: str) -> list[float]:
        return self._vector(self._axis(query))

    @staticmethod
    def _axis(text: str) -> int:
        if "公式" in text or "mc" in text:
            return 3
        if "表" in text or "温度" in text or "项目" in text:
            return 2
        if "维护" in text or "设备" in text:
            return 1
        return 0


def _store(settings: Settings | None = None) -> QdrantRepresentationStore:
    return QdrantRepresentationStore(
        settings or Settings(_env_file=None),
        client=QdrantClient(":memory:"),
    )


def test_representation_collection_role_is_fail_closed_even_for_mutated_settings() -> None:
    settings = Settings(_env_file=None)
    unsafe_settings = settings.model_copy(
        update={"qdrant_representation_collection_name": settings.qdrant_collection_name}
    )

    with pytest.raises(RepresentationCollectionConfigurationError):
        validate_representation_collection_role(unsafe_settings)
    with pytest.raises(RepresentationCollectionConfigurationError):
        QdrantRepresentationStore(unsafe_settings, client=QdrantClient(":memory:"))

    with pytest.raises(ValidationError, match="must differ"):
        Settings(
            _env_file=None,
            qdrant_collection_name="same_collection",
            qdrant_representation_collection_name="same_collection",
        )


def test_representation_collection_cannot_target_frozen_name_when_chunk_setting_changes() -> None:
    settings = Settings(_env_file=None)
    unsafe_settings = settings.model_copy(
        update={
            "qdrant_collection_name": "some_other_collection",
            "qdrant_representation_collection_name": "knowledgescope_chunks_v1",
        }
    )

    with pytest.raises(RepresentationCollectionConfigurationError, match="protected"):
        validate_representation_collection_role(unsafe_settings)
    with pytest.raises(RepresentationCollectionConfigurationError, match="protected"):
        QdrantRepresentationStore(unsafe_settings, client=QdrantClient(":memory:"))


def test_existing_renamed_chunk_collection_is_protected_before_mutation() -> None:
    settings = Settings(
        _env_file=None,
        qdrant_collection_name="some_other_collection",
        qdrant_representation_collection_name="renamed_chunks_v1",
    )
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_representation_collection_name,
        vectors_config=models.VectorParams(
            size=QDRANT_VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )
    chunk_payload = {
        "collection_schema_version": "1.0",
        "chunk_id": "a2-chunk",
        "document_id": str(DOCUMENT_ID),
        "knowledge_base_id": str(FIRST_KB),
        "page_start": 1,
        "page_end": 1,
        "source_block_ids": ["text-1"],
        "section_path": [],
        "content_types": ["text"],
        "asset_refs": [],
        "text": "冻结 chunk",
        "chunking_config_fingerprint": "a" * 64,
        "embedding_model": QWEN_EMBEDDING_MODEL_ID,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_config_fingerprint": embedding_config_fingerprint(settings),
    }
    client.upsert(
        collection_name=settings.qdrant_representation_collection_name,
        points=[
            models.PointStruct(
                id=str(point_id_for_chunk("a2-chunk")),
                vector=[0.0] * QDRANT_VECTOR_DIMENSION,
                payload=chunk_payload,
            )
        ],
    )
    store = QdrantRepresentationStore(settings, client=client)

    with pytest.raises(RepresentationCollectionConfigurationError, match="chunk index"):
        store.ensure_collection()
    valid_point = build_representation_points(
        build_indexable_evidence(_document(), FIRST_KB),
        settings=settings,
        embedder=_DirectionalEncoder(),
    )[0]
    with pytest.raises(RepresentationCollectionConfigurationError, match="chunk index"):
        store.replace_document(DOCUMENT_ID, FIRST_KB, [valid_point])
    with pytest.raises(RepresentationCollectionConfigurationError, match="chunk index"):
        store._delete_ids([point_id_for_chunk("a2-chunk")])
    assert client.retrieve(
        settings.qdrant_representation_collection_name,
        [str(point_id_for_chunk("a2-chunk"))],
    )
    store.close()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("embedding_model", "another/model"),
        ("embedding_model_revision", "another-revision"),
        ("embedding_config_fingerprint", "b" * 64),
        ("collection_fingerprint", "c" * 64),
        ("representation_schema_version", "0.9"),
    ],
)
def test_replace_document_rejects_incompatible_point_contract_before_qdrant_write(
    field_name: str,
    bad_value: str,
) -> None:
    settings = Settings(_env_file=None)
    client = QdrantClient(":memory:")
    store = QdrantRepresentationStore(settings, client=client)
    artifact = build_indexable_evidence(_document(), FIRST_KB)
    point = build_representation_points(
        artifact,
        settings=settings,
        embedder=_DirectionalEncoder(),
    )[0]
    invalid_payload = point.payload.model_copy(update={field_name: bad_value})

    with pytest.raises(RepresentationCollectionConfigurationError, match="active index contract"):
        store.replace_document(
            DOCUMENT_ID,
            FIRST_KB,
            [replace(point, payload=invalid_payload)],
        )
    assert not client.collection_exists(settings.qdrant_representation_collection_name)
    store.close()


def test_replace_document_rejects_wrong_vector_dimension_before_qdrant_write() -> None:
    settings = Settings(_env_file=None)
    client = QdrantClient(":memory:")
    store = QdrantRepresentationStore(settings, client=client)
    point = build_representation_points(
        build_indexable_evidence(_document(), FIRST_KB),
        settings=settings,
        embedder=_DirectionalEncoder(),
    )[0]

    with pytest.raises(RepresentationIndexError, match="1024 dimensions"):
        store.replace_document(
            DOCUMENT_ID,
            FIRST_KB,
            [replace(point, vector=(0.0,) * (QDRANT_VECTOR_DIMENSION - 1))],
        )
    assert not client.collection_exists(settings.qdrant_representation_collection_name)
    store.close()


def test_replace_document_rejects_non_normalized_vector_before_qdrant_write() -> None:
    settings = Settings(_env_file=None)
    client = QdrantClient(":memory:")
    store = QdrantRepresentationStore(settings, client=client)
    point = build_representation_points(
        build_indexable_evidence(_document(), FIRST_KB),
        settings=settings,
        embedder=_DirectionalEncoder(),
    )[0]

    with pytest.raises(RepresentationIndexError, match="L2-normalized"):
        store.replace_document(
            DOCUMENT_ID,
            FIRST_KB,
            [replace(point, vector=(0.5,) + (0.0,) * (QDRANT_VECTOR_DIMENSION - 1))],
        )
    assert not client.collection_exists(settings.qdrant_representation_collection_name)
    store.close()


def test_existing_wrong_collection_payload_fails_before_replacement() -> None:
    settings = Settings(_env_file=None)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_representation_collection_name,
        vectors_config=models.VectorParams(
            size=QDRANT_VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )
    client.upsert(
        collection_name=settings.qdrant_representation_collection_name,
        points=[
            models.PointStruct(
                id=str(UUID("44444444-4444-4444-4444-444444444444")),
                vector=[0.0] * QDRANT_VECTOR_DIMENSION,
                payload={"document_id": str(DOCUMENT_ID), "chunk_id": "a2-chunk"},
            )
        ],
    )
    store = QdrantRepresentationStore(settings, client=client)

    with pytest.raises(RepresentationIndexError, match="could not be created or validated"):
        store.ensure_collection()
    assert (
        len(
            client.retrieve(
                settings.qdrant_representation_collection_name,
                ["44444444-4444-4444-4444-444444444444"],
            )
        )
        == 1
    )


def test_materialization_preserves_source_lineage_and_structural_context(
    tmp_path: Path,
) -> None:
    document = _document()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    artifact = build_indexable_evidence(document, FIRST_KB, data_dir=settings.data_dir)

    by_block = {item.lineage.source_block_ids[0]: item for item in artifact.evidence}
    assert by_block["image-1"].modality == "image"
    assert {rep.representation_type for rep in by_block["image-1"].representations} == {
        "asset_ref",
        "context",
    }
    image_context = next(
        rep.content
        for rep in by_block["image-1"].representations
        if rep.representation_type == "context"
    )
    assert image_context is not None
    assert "设备需要定期维护" in image_context
    text_representation_types = {
        rep.representation_type for rep in by_block["text-1"].representations
    }
    assert text_representation_types == {"text", "context"}
    text_context = next(
        rep.content
        for rep in by_block["text-1"].representations
        if rep.representation_type == "context"
    )
    assert text_context is not None
    assert "章节: 设备维护" in text_context
    assert by_block["image-1"].lineage.asset_refs == ["images/maintenance.png"]
    assert by_block["table-1"].lineage.source_block_ids == ["table-1"]
    assert by_block["formula-1"].representations[0].content == "E = mc^2"

    stored = assert_evidence_artifact_current(settings.data_dir, document, FIRST_KB)
    assert stored.model_dump(mode="json") == artifact.model_dump(mode="json")


def test_representation_ids_and_collection_fingerprint_are_stable_and_independent() -> None:
    settings = Settings(_env_file=None)
    artifact = build_indexable_evidence(_document(), FIRST_KB)
    points = build_representation_points(
        artifact, settings=settings, embedder=_DirectionalEncoder()
    )

    assert points
    assert point_id_for_representation(points[0].payload.representation_id) == points[0].point_id
    assert representation_collection_fingerprint(settings) == representation_collection_fingerprint(
        Settings(_env_file=None)
    )
    assert settings.qdrant_collection_name != settings.qdrant_representation_collection_name
    assert all(point.payload.collection_schema_version == "1.0" for point in points)
    assert all(point.payload.evidence_schema_version == "1.0" for point in points)
    assert all(point.payload.representation_schema_version == "1.0" for point in points)
    assert all(point.payload.embedding_model == _DirectionalEncoder.model_id for point in points)


def test_index_replacement_is_idempotent_removes_stale_points_and_supports_empty_documents() -> (
    None
):
    settings = Settings(_env_file=None)
    store = _store(settings)
    artifact = build_indexable_evidence(_document(), FIRST_KB)
    points = build_representation_points(
        artifact, settings=settings, embedder=_DirectionalEncoder()
    )

    first = store.replace_document(DOCUMENT_ID, FIRST_KB, points)
    second = store.replace_document(DOCUMENT_ID, FIRST_KB, points)
    assert first.indexed_count == second.indexed_count == len(points)
    assert second.removed_stale_count == 0

    replacement = store.replace_document(DOCUMENT_ID, FIRST_KB, points[:2])
    assert replacement.removed_stale_count == len(points) - 2
    assert len(store.list_payloads(knowledge_base_id=FIRST_KB)) == 2

    cleared = store.replace_document(DOCUMENT_ID, FIRST_KB, [])
    assert cleared.indexed_count == 0
    assert cleared.removed_stale_count == 2
    assert store.delete_document(DOCUMENT_ID, knowledge_base_id=FIRST_KB) == 0
    store.close()


def test_kb_isolation_and_modality_filter_are_enforced() -> None:
    settings = Settings(_env_file=None)
    store = _store(settings)
    artifact_one = build_indexable_evidence(_document(), FIRST_KB)
    artifact_two = build_indexable_evidence(_document(), SECOND_KB)
    encoder = _DirectionalEncoder()
    store.replace_document(
        DOCUMENT_ID,
        FIRST_KB,
        build_representation_points(artifact_one, settings=settings, embedder=encoder),
    )
    store.replace_document(
        DOCUMENT_ID,
        SECOND_KB,
        build_representation_points(artifact_two, settings=settings, embedder=encoder),
    )

    image_hits = store.search(
        encoder.encode_query("设备维护图片"),
        limit=10,
        knowledge_base_id=FIRST_KB,
        modality="image",
    )
    assert image_hits
    assert all(hit.payload.knowledge_base_id == FIRST_KB for hit in image_hits)
    assert all(hit.payload.modality == "image" for hit in image_hits)
    assert store.delete_document(DOCUMENT_ID, knowledge_base_id=FIRST_KB) > 0
    assert store.list_payloads(knowledge_base_id=SECOND_KB)
    store.close()


def test_search_requires_searchable_and_excludes_quarantined_points(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    store = _store(settings)
    encoder = _DirectionalEncoder()
    artifact = build_indexable_evidence(_document(), FIRST_KB, data_dir=settings.data_dir)
    points = build_representation_points(artifact, settings=settings, embedder=encoder)
    store.replace_document(DOCUMENT_ID, FIRST_KB, points)

    opaque_payload = next(
        payload
        for payload in representation_index_payloads(artifact, searchable_only=False)
        if not payload.searchable
    )
    indexed_opaque_payload = IndexedRepresentationPayload(
        **opaque_payload.model_dump(mode="python"),
        collection_fingerprint=representation_collection_fingerprint(settings),
        embedding_model=encoder.model_id,
        embedding_model_revision=encoder.model_revision,
        embedding_config_fingerprint=encoder.config_fingerprint,
    )
    client = store._get_client()
    opaque_point_id = point_id_for_representation(indexed_opaque_payload.representation_id)
    client.upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(opaque_point_id),
                vector=encoder._vector(1),
                payload=indexed_opaque_payload.model_dump(mode="json"),
            )
        ],
    )

    query_vector = encoder.encode_query("设备维护图片")
    before = store.search(query_vector, limit=10, knowledge_base_id=FIRST_KB)
    assert all(hit.payload.searchable for hit in before)
    assert all(
        hit.payload.representation_id != indexed_opaque_payload.representation_id for hit in before
    )

    snapshot = store.quarantine_document(DOCUMENT_ID, knowledge_base_id=FIRST_KB)
    assert snapshot.point_states
    assert store.search(query_vector, limit=10, knowledge_base_id=FIRST_KB) == []
    store.restore_document_visibility(snapshot)
    after = store.search(query_vector, limit=10, knowledge_base_id=FIRST_KB)
    assert after
    assert all(hit.payload.searchable for hit in after)
    assert all(
        hit.payload.representation_id != indexed_opaque_payload.representation_id for hit in after
    )
    store.close()


def test_retrieval_deduplicates_multiple_representations_at_evidence_level(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    store = _store(settings)
    encoder = _DirectionalEncoder()
    artifact = build_indexable_evidence(_document(), FIRST_KB, data_dir=settings.data_dir)
    points = build_representation_points(artifact, settings=settings, embedder=encoder)
    store.replace_document(DOCUMENT_ID, FIRST_KB, points)

    result = MultimodalRepresentationRetrievalService(store, encoder).search(
        "温度表",
        knowledge_base_id=FIRST_KB,
        top_k=10,
        modality="table",
    )
    table_items = [item for item in result.items if item.modality == "table"]
    assert len(table_items) == 1
    assert len(table_items[0].representations) >= 2
    assert len(result.raw_hits) >= len(table_items[0].representations)
    assert len({item.evidence_id for item in result.items}) == len(result.items)
    assert [item.rank for item in result.items] == list(range(1, len(result.items) + 1))
    store.close()


def test_index_payload_rejects_mismatched_authoritative_lineage() -> None:
    settings = Settings(_env_file=None)
    artifact = build_indexable_evidence(_document(), FIRST_KB)
    payload = build_representation_points(
        artifact,
        settings=settings,
        embedder=_DirectionalEncoder(),
    )[0].payload.model_dump(mode="python")

    with pytest.raises(ValidationError, match="evidence_id"):
        IndexedRepresentationPayload(
            **payload | {"document_id": UUID("44444444-4444-4444-4444-444444444444")}
        )


def test_retrieval_rejects_payload_that_disagrees_with_authoritative_section_path(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    store = _store(settings)
    encoder = _DirectionalEncoder()
    artifact = build_indexable_evidence(_document(), FIRST_KB, data_dir=settings.data_dir)
    points = build_representation_points(artifact, settings=settings, embedder=encoder)
    store.replace_document(DOCUMENT_ID, FIRST_KB, points)

    client = store._get_client()
    record = client.retrieve(
        store.collection_name,
        [str(points[0].point_id)],
        with_payload=True,
        with_vectors=True,
    )[0]
    tampered_payload = dict(record.payload)
    tampered_payload["section_path"] = ["伪造章节"]
    client.upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(record.id),
                vector=record.vector,
                payload=tampered_payload,
            )
        ],
    )

    with pytest.raises(RepresentationIndexError, match="authoritative"):
        MultimodalRepresentationRetrievalService(store, encoder).search(
            "设备维护",
            knowledge_base_id=FIRST_KB,
            top_k=10,
        )
    store.close()


def test_failed_representation_replacement_restores_previous_evidence_and_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(_document().model_dump_json(), encoding="utf-8")
    store = _store(settings)
    encoder = _DirectionalEncoder()

    index_canonical_document(
        canonical_path,
        knowledge_base_id=FIRST_KB,
        settings=settings,
        store=store,
        embedder=encoder,
    )
    evidence_path = settings.data_dir / "evidence" / str(DOCUMENT_ID) / "evidence.json"
    evidence_before = evidence_path.read_bytes()
    points_before = store.list_payloads(knowledge_base_id=FIRST_KB)

    def fail_replacement(*_: object, **__: object) -> object:
        raise RepresentationIndexError("injected replacement failure")

    monkeypatch.setattr(store, "replace_document", fail_replacement)
    with pytest.raises(RepresentationIndexError, match="injected replacement failure"):
        index_canonical_document(
            canonical_path,
            knowledge_base_id=FIRST_KB,
            settings=settings,
            store=store,
            embedder=encoder,
        )

    assert evidence_path.read_bytes() == evidence_before
    assert store.list_payloads(knowledge_base_id=FIRST_KB) == points_before
    store.close()


def test_replacement_fails_closed_when_points_have_no_authoritative_evidence(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(_document().model_dump_json(), encoding="utf-8")
    store = _store(settings)
    encoder = _DirectionalEncoder()
    points = build_representation_points(
        build_indexable_evidence(_document(), FIRST_KB),
        settings=settings,
        embedder=encoder,
    )
    store.replace_document(DOCUMENT_ID, FIRST_KB, points)

    with pytest.raises(RepresentationIndexError, match="no matching evidence artifact"):
        index_canonical_document(
            canonical_path,
            knowledge_base_id=FIRST_KB,
            settings=settings,
            store=store,
            embedder=encoder,
        )

    assert len(store.list_payloads(knowledge_base_id=FIRST_KB)) == len(points)
    store.close()


def test_visibility_restore_fails_closed_when_a_captured_point_disappears() -> None:
    settings = Settings(_env_file=None)
    store = _store(settings)
    encoder = _DirectionalEncoder()
    artifact = build_indexable_evidence(_document(), FIRST_KB)
    points = build_representation_points(artifact, settings=settings, embedder=encoder)
    store.replace_document(DOCUMENT_ID, FIRST_KB, points)
    snapshot = store.quarantine_document(DOCUMENT_ID, knowledge_base_id=FIRST_KB)
    missing_point_id = snapshot.point_states[0][0]
    store._get_client().delete(
        collection_name=store.collection_name,
        points_selector=models.PointIdsList(points=[str(missing_point_id)]),
        wait=True,
    )

    with pytest.raises(RepresentationIndexError, match="visibility snapshot is missing"):
        store.restore_document_visibility(
            RepresentationVisibilitySnapshot(
                document_id=DOCUMENT_ID,
                knowledge_base_id=FIRST_KB,
                point_states=snapshot.point_states,
            )
        )
    store.close()


def test_audit_distinguishes_non_searchable_asset_references(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(_document().model_dump_json(), encoding="utf-8")
    store = _store(settings)
    index_canonical_document(
        canonical_path,
        knowledge_base_id=FIRST_KB,
        settings=settings,
        store=store,
        embedder=_DirectionalEncoder(),
    )

    root = tmp_path / "canonical"
    root.mkdir()
    (root / "document.json").write_text(
        canonical_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    audit = audit_representation_index(
        root,
        knowledge_base_id=FIRST_KB,
        settings=settings,
        store=store,
    )
    assert audit.source_evidence_count == 5
    assert audit.representation_count > audit.searchable_representation_count
    assert audit.indexed_count == audit.searchable_representation_count
    asset_buckets = [
        bucket for bucket in audit.buckets if bucket.representation_type == "asset_ref"
    ]
    assert sum(bucket.representation_count for bucket in asset_buckets) == 2
    assert sum(bucket.searchable_representation_count for bucket in asset_buckets) == 0
    assert sum(bucket.indexed_count for bucket in asset_buckets) == 0
    assert audit.missing_count == audit.stale_count == 0
    store.close()


@pytest.mark.integration
def test_real_representation_qdrant_contract_round_trip(tmp_path: Path) -> None:
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip(
            "set KNOWLEDGE_SCOPE_RUN_QDRANT_INTEGRATION=1 to run the Qdrant integration test"
        )
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    store = QdrantRepresentationStore(settings)
    encoder = _DirectionalEncoder()
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(_document().model_dump_json(), encoding="utf-8")
    try:
        result = index_canonical_document(
            canonical_path,
            knowledge_base_id=FIRST_KB,
            settings=settings,
            store=store,
            embedder=encoder,
        )
        assert result.indexed_count > 0
        search_result = MultimodalRepresentationRetrievalService(store, encoder).search(
            "温度表",
            knowledge_base_id=FIRST_KB,
            top_k=5,
            modality="table",
        )
        assert search_result.items
        assert search_result.items[0].knowledge_base_id == FIRST_KB
        assert search_result.items[0].document_id == DOCUMENT_ID
    finally:
        store.delete_document(DOCUMENT_ID, knowledge_base_id=FIRST_KB)
        store.close()
