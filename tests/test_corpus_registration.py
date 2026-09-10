from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from knowledge_scope.documents.models import (
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_STATUS_UPLOADED,
    DOCUMENT_STORAGE_KIND_MANAGED,
    Document,
)
from knowledge_scope.documents.registration import (
    DEFAULT_EVAL_DATASET,
    CorpusRegistrationError,
    build_frozen_eval_kb_mapping,
    build_registration_spec,
    register_corpus,
)
from knowledge_scope.evaluation.embedding_benchmark import IndexedChunk
from knowledge_scope.knowledge_bases.models import KnowledgeBase
from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock


def _write_corpus_fixture(
    root: Path,
    *,
    document_count: int = 2,
    duplicate_first: bool = False,
) -> tuple[Path, Path, Path, tuple[UUID, ...]]:
    canonical_root = root / "canonical"
    canonical_root.mkdir()
    manifest_path = root / "corpus-manifest.jsonl"
    chunk_index_path = root / "chunk-index.jsonl"
    document_ids = tuple(uuid4() for _ in range(document_count))
    manifest_rows: list[dict[str, object]] = []
    chunks: list[IndexedChunk] = []
    for index, document_id in enumerate(document_ids):
        block_id = f"p1-b{index + 1}"
        item_id = f"item-{index + 1}"
        canonical = CanonicalDocument(
            document_id=document_id,
            pages=[
                Page(
                    page_number=1,
                    blocks=[
                        TextBlock(block_id=block_id, reading_order=0, text=f"内容 {index + 1}")
                    ],
                )
            ],
        )
        (canonical_root / f"{item_id}.json").write_text(
            canonical.model_dump_json(), encoding="utf-8"
        )
        metadata = {
            "basename": f"document-{index + 1}.pdf",
            "benchmark_document_uuid": str(document_id),
            "benchmark_item_id": item_id,
            "inventory_status": "ready",
            "relative_path": f"part_{index + 1}/document-{index + 1}.pdf",
            "sha256": f"{index + 1:064x}",
            "size_bytes": index + 1,
        }
        manifest_rows.append(metadata)
        if duplicate_first and index == 0:
            manifest_rows.append(
                {
                    **metadata,
                    "basename": "document-1-copy.pdf",
                    "benchmark_item_id": "item-1-copy",
                    "relative_path": "part_1/document-1-copy.pdf",
                }
            )
        chunks.append(
            IndexedChunk(
                schema_version="1.0",
                chunk_id=f"chunk-{index + 1}",
                document_id=document_id,
                ordinal=0,
                text=f"内容 {index + 1}",
                page_start=1,
                page_end=1,
                source_block_ids=[block_id],
                section_path=[],
                content_types=["text"],
                asset_refs=[],
                config_fingerprint="a" * 64,
            )
        )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    chunk_index_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    return manifest_path, canonical_root, chunk_index_path, document_ids


async def _cleanup_registration_rows(
    factory: async_sessionmaker[AsyncSession],
    knowledge_base_ids: tuple[UUID, ...],
) -> None:
    async with factory() as session:
        await session.execute(
            delete(Document).where(Document.knowledge_base_id.in_(knowledge_base_ids))
        )
        await session.execute(delete(KnowledgeBase).where(KnowledgeBase.id.in_(knowledge_base_ids)))
        await session.commit()


def test_registration_spec_deduplicates_identical_manifest_document_ids(tmp_path: Path) -> None:
    manifest, canonical_root, chunk_index, document_ids = _write_corpus_fixture(
        tmp_path, duplicate_first=True
    )

    spec = build_registration_spec(
        UUID(int=100),
        corpus_manifest=manifest,
        canonical_root=canonical_root,
        chunk_index=chunk_index,
        expected_document_count=2,
        expected_chunk_count=2,
    )

    assert {document.document_id for document in spec.documents} == set(document_ids)
    assert spec.manifest_rows == 3
    assert spec.duplicate_manifest_rows == 1
    assert spec.chunk_count == 2


def test_registration_spec_rejects_chunk_documents_outside_canonical_set(tmp_path: Path) -> None:
    manifest, canonical_root, chunk_index, _ = _write_corpus_fixture(tmp_path)
    records = chunk_index.read_text(encoding="utf-8").splitlines()
    invalid = json.loads(records[0])
    invalid["document_id"] = str(UUID(int=999))
    chunk_index.write_text(json.dumps(invalid) + "\n" + records[1] + "\n", encoding="utf-8")

    with pytest.raises(CorpusRegistrationError, match="document IDs"):
        build_registration_spec(
            UUID(int=100),
            corpus_manifest=manifest,
            canonical_root=canonical_root,
            chunk_index=chunk_index,
            expected_document_count=2,
            expected_chunk_count=2,
        )


def test_registration_spec_rejects_missing_canonical_artifact(tmp_path: Path) -> None:
    manifest, canonical_root, chunk_index, _ = _write_corpus_fixture(tmp_path)
    (canonical_root / "item-1.json").unlink()

    with pytest.raises(CorpusRegistrationError, match="canonical artifact"):
        build_registration_spec(
            UUID(int=100),
            corpus_manifest=manifest,
            canonical_root=canonical_root,
            chunk_index=chunk_index,
            expected_document_count=2,
            expected_chunk_count=2,
        )


def test_registration_spec_rejects_missing_chunk_index(tmp_path: Path) -> None:
    manifest, canonical_root, chunk_index, _ = _write_corpus_fixture(tmp_path)
    chunk_index.unlink()

    with pytest.raises(CorpusRegistrationError, match="chunk index"):
        build_registration_spec(
            UUID(int=100),
            corpus_manifest=manifest,
            canonical_root=canonical_root,
            chunk_index=chunk_index,
            expected_document_count=2,
            expected_chunk_count=2,
        )


def test_registration_spec_rejects_duplicate_content_for_distinct_documents(tmp_path: Path) -> None:
    manifest, canonical_root, chunk_index, _ = _write_corpus_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records[1]["sha256"] = records[0]["sha256"]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(CorpusRegistrationError, match="share a SHA-256"):
        build_registration_spec(
            UUID(int=100),
            corpus_manifest=manifest,
            canonical_root=canonical_root,
            chunk_index=chunk_index,
            expected_document_count=2,
            expected_chunk_count=2,
        )


def test_frozen_eval_mapping_keeps_labels_unchanged_and_requires_registered_documents() -> None:
    records = [
        json.loads(line)
        for line in DEFAULT_EVAL_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evidence_document_ids = {
        UUID(location["document_id"])
        for record in records
        for location in record["item"]["evidence"]
    }
    target_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    assignments = build_frozen_eval_kb_mapping(
        DEFAULT_EVAL_DATASET,
        registered_document_ids=evidence_document_ids,
        knowledge_base_id=target_id,
    )

    assert len(assignments) == 108
    assert {assignment.split for assignment in assignments} == {"dev", "test"}
    assert {assignment.knowledge_base_id for assignment in assignments} == {target_id}
    assert [assignment.item_id for assignment in assignments] == [
        record["item"]["item_id"] for record in records
    ]
    with pytest.raises(CorpusRegistrationError, match="unregistered document"):
        build_frozen_eval_kb_mapping(
            DEFAULT_EVAL_DATASET,
            registered_document_ids=set(),
            knowledge_base_id=target_id,
        )


@pytest.mark.anyio
async def test_registration_is_idempotent_and_external_reference_backed(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    manifest, canonical_root, chunk_index, document_ids = _write_corpus_fixture(tmp_path)
    target_id = uuid4()
    spec = build_registration_spec(
        target_id,
        corpus_manifest=manifest,
        canonical_root=canonical_root,
        chunk_index=chunk_index,
        expected_document_count=2,
        expected_chunk_count=2,
    )
    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with factory() as session:
        session.add(KnowledgeBase(id=target_id, name="benchmark"))
        await session.commit()

    async with factory() as session:
        first = await register_corpus(session, spec)
    async with factory() as session:
        second = await register_corpus(session, spec)

    assert first.documents_inserted == 2
    assert first.documents_already_valid == 0
    assert second.documents_inserted == 0
    assert second.documents_already_valid == 2
    async with factory() as session:
        documents = list(
            (
                await session.scalars(
                    select(Document).where(Document.knowledge_base_id == target_id)
                )
            ).all()
        )
    assert {document.id for document in documents} == set(document_ids)
    assert all(document.storage_key is None for document in documents)
    assert all(document.storage_kind == "external_reference" for document in documents)
    assert all(document.status == "registered" for document in documents)
    assert all(document.source_ref.startswith("benchmark/a1-5/") for document in documents)
    await _cleanup_registration_rows(factory, (target_id,))


@pytest.mark.anyio
async def test_registration_conflicting_document_owner_does_not_partially_insert(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    manifest, canonical_root, chunk_index, document_ids = _write_corpus_fixture(tmp_path)
    target_id = uuid4()
    other_id = uuid4()
    spec = build_registration_spec(
        target_id,
        corpus_manifest=manifest,
        canonical_root=canonical_root,
        chunk_index=chunk_index,
        expected_document_count=2,
        expected_chunk_count=2,
    )
    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with factory() as session:
        session.add_all(
            [KnowledgeBase(id=target_id, name="target"), KnowledgeBase(id=other_id, name="other")]
        )
        await session.commit()
        session.add(
            Document(
                id=document_ids[0],
                knowledge_base_id=other_id,
                original_filename="already.pdf",
                storage_key="documents/other/document/original.pdf",
                storage_kind=DOCUMENT_STORAGE_KIND_MANAGED,
                source_ref=None,
                media_type=DOCUMENT_MEDIA_TYPE_PDF,
                size_bytes=1,
                sha256="b" * 64,
                status=DOCUMENT_STATUS_UPLOADED,
            )
        )
        await session.commit()

    with pytest.raises(CorpusRegistrationError, match="preflight"):
        async with factory() as session:
            await register_corpus(session, spec)

    async with factory() as session:
        target_count = await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.knowledge_base_id == target_id)
        )
    assert target_count == 0
    await _cleanup_registration_rows(factory, (target_id, other_id))


@pytest.mark.anyio
async def test_registration_flush_failure_rolls_back_all_rows(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, canonical_root, chunk_index, _ = _write_corpus_fixture(tmp_path)
    target_id = uuid4()
    spec = build_registration_spec(
        target_id,
        corpus_manifest=manifest,
        canonical_root=canonical_root,
        chunk_index=chunk_index,
        expected_document_count=2,
        expected_chunk_count=2,
    )
    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with factory() as session:
        session.add(KnowledgeBase(id=target_id, name="rollback-target"))
        await session.commit()

    async def fail_flush(_session: AsyncSession, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SQLAlchemyError("injected flush failure")

    monkeypatch.setattr(AsyncSession, "flush", fail_flush)
    with pytest.raises(CorpusRegistrationError, match="rolled back"):
        async with factory() as session:
            await register_corpus(session, spec)

    async with factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.knowledge_base_id == target_id)
        )
    assert count == 0
    await _cleanup_registration_rows(factory, (target_id,))
