from __future__ import annotations

# ruff: noqa: RUF001
import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from knowledge_scope.chunking.models import Chunk
from knowledge_scope.evaluation.embedding_benchmark import IndexedChunk
from knowledge_scope.evaluation.graph_corpus_build import (
    CheckpointStore,
    ChunkCheckpoint,
    CorpusDocumentChunks,
    CorpusDocumentMetadata,
    FailedProviderAttemptSummary,
    GraphCorpusBuildError,
    GraphCorpusRunner,
    build_a21_graph_coverage,
    build_corpus_input_snapshot,
    build_pipeline_fingerprint,
    estimate_full_run,
    iter_indexed_chunks,
    load_corpus_metadata,
    maximum_extraction_provider_attempts,
    summarize_failed_provider_attempts,
    validate_corpus_input_snapshot,
)
from knowledge_scope.evaluation.retrieval_eval import (
    FinalDatasetItem,
    RetrievalEvidence,
    make_eval_item,
)
from knowledge_scope.extraction.service import (
    ChunkExtractionResult,
    ExtractionAttempt,
    ExtractionError,
    ExtractionStats,
)
from knowledge_scope.graph.models import GraphEntity, GraphProvenance, entity_id_for
from knowledge_scope.graph.neo4j import GraphStoreError
from knowledge_scope.llm.errors import LLMProviderError
from knowledge_scope.llm.schemas import LLMResult
from knowledge_scope.shared.config import Settings

KNOWLEDGE_BASE_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_A = UUID("22222222-2222-4222-8222-222222222222")
DOCUMENT_B = UUID("33333333-3333-4333-8333-333333333333")


def _chunk(document_id: UUID, ordinal: int = 0) -> IndexedChunk:
    return IndexedChunk(
        schema_version="1.0",
        chunk_id=f"chunk-{document_id}-{ordinal}",
        document_id=document_id,
        ordinal=ordinal,
        text=f"文档 {document_id} 的第 {ordinal} 个事实。",
        page_start=1,
        page_end=1,
        source_block_ids=[f"block-{ordinal}"],
        section_path=["测试章节"],
        content_types=["text"],
        asset_refs=[],
        config_fingerprint="a" * 64,
    )


def _document(document_id: UUID, *, chunk_count: int = 1) -> CorpusDocumentChunks:
    return CorpusDocumentChunks(
        metadata=CorpusDocumentMetadata(
            benchmark_item_id=f"item-{document_id}",
            document_id=document_id,
            subject="测试",
            relative_path="corpus/example.pdf",
        ),
        chunks=tuple(_chunk(document_id, ordinal) for ordinal in range(chunk_count)),
    )


def _entity(document_id: UUID, chunk_id: str) -> GraphEntity:
    provenance = GraphProvenance(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=["block-0"],
        section_path=["测试章节"],
    )
    return GraphEntity(
        entity_id=entity_id_for(
            f"实体-{document_id}",
            "概念",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=document_id,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=document_id,
        canonical_name=f"实体-{document_id}",
        entity_type="概念",
        provenance=[provenance],
    )


def _result(chunk: IndexedChunk | Chunk, *, with_usage: bool = False) -> ChunkExtractionResult:
    entity = _entity(chunk.document_id, chunk.chunk_id)
    return ChunkExtractionResult(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        status="accepted",
        entities=(entity,),
        relations=(),
        stats=ExtractionStats(
            attempts=1,
            parse_failures=0,
            schema_failures=0,
            grounding_rejections=0,
            duplicate_entities=0,
            duplicate_relations=0,
        ),
        llm_results=(
            LLMResult(
                text="{}",
                provider="fake",
                model="fake-model",
                input_tokens=100,
                output_tokens=20,
                latency_ms=1.0,
            ),
        )
        if with_usage
        else (),
    )


class _FakeExtraction:
    def __init__(
        self,
        *,
        fail_once: set[str] | None = None,
        cancel: bool = False,
        with_usage: bool = False,
    ) -> None:
        self.fail_once = set(fail_once or ())
        self.cancel = cancel
        self.with_usage = with_usage
        self.calls: list[Chunk] = []

    async def extract_chunk(
        self, chunk: Chunk, *, knowledge_base_id: UUID
    ) -> ChunkExtractionResult:
        assert knowledge_base_id == KNOWLEDGE_BASE_ID
        self.calls.append(chunk)
        if self.cancel:
            raise asyncio.CancelledError
        if chunk.chunk_id in self.fail_once:
            self.fail_once.remove(chunk.chunk_id)
            raise ExtractionError("temporary extraction failure", category="timeout")
        return _result(chunk, with_usage=self.with_usage)


class _ProviderFailureExtraction:
    """Emit a safe provider failure without making a network request."""

    def __init__(self, *, status_code: int, retryable: bool, successes: int = 0) -> None:
        self.status_code = status_code
        self.retryable = retryable
        self.successes = successes
        self.calls: list[Chunk] = []

    async def extract_chunk(
        self, chunk: Chunk, *, knowledge_base_id: UUID
    ) -> ChunkExtractionResult:
        assert knowledge_base_id == KNOWLEDGE_BASE_ID
        self.calls.append(chunk)
        if self.successes:
            self.successes -= 1
            return _result(chunk)
        try:
            raise LLMProviderError(
                "api",
                "safe fake provider failure",
                retryable=self.retryable,
                status_code=self.status_code,
            )
        except LLMProviderError as provider_error:
            raise ExtractionError(
                "LLM extraction request failed",
                category="api",
                attempts=(
                    ExtractionAttempt(
                        number=1,
                        category="api",
                        provider_attempts=1,
                    ),
                ),
            ) from provider_error


class _CancelAfterFirstExtraction(_FakeExtraction):
    def __init__(self) -> None:
        super().__init__()
        self._completed = False

    async def extract_chunk(
        self, chunk: Chunk, *, knowledge_base_id: UUID
    ) -> ChunkExtractionResult:
        if self._completed:
            raise asyncio.CancelledError
        self._completed = True
        return await super().extract_chunk(chunk, knowledge_base_id=knowledge_base_id)


class _FakeStore:
    def __init__(
        self,
        *,
        fail_upsert: bool = False,
        fail_link_upsert: bool = False,
        fail_link_delete: bool = False,
        fail_entity_listing_once: bool = False,
    ) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.deleted: list[UUID] = []
        self.deleted_link_documents: list[UUID] = []
        self.link_calls = 0
        self.fail_upsert = fail_upsert
        self.fail_link_upsert = fail_link_upsert
        self.fail_link_delete = fail_link_delete
        self.fail_entity_listing_once = fail_entity_listing_once

    def upsert_extraction(
        self,
        entities: Sequence[GraphEntity],
        relations: Sequence[object],
    ) -> None:
        del relations
        if self.fail_upsert:
            raise GraphStoreError("fake persistence failure")
        self.entities.update({entity.entity_id: entity for entity in entities})

    def list_entities_for_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> tuple[GraphEntity, ...]:
        if self.fail_entity_listing_once:
            self.fail_entity_listing_once = False
            raise GraphStoreError("fake entity listing failure")
        return tuple(
            entity
            for entity in self.entities.values()
            if entity.knowledge_base_id == knowledge_base_id and entity.document_id == document_id
        )

    def upsert_linking_result(
        self,
        canonical_entities: Sequence[object],
        decisions: Sequence[object],
        mappings: Sequence[object],
    ) -> None:
        del canonical_entities, decisions, mappings
        if self.fail_link_upsert:
            raise GraphStoreError("fake linking persistence failure")
        self.link_calls += 1

    def has_document_state(self, knowledge_base_id: UUID, document_id: UUID) -> bool:
        return any(
            entity.knowledge_base_id == knowledge_base_id and entity.document_id == document_id
            for entity in self.entities.values()
        )

    def delete_documents_links(
        self,
        knowledge_base_id: UUID,
        document_ids: Sequence[UUID],
    ) -> None:
        if self.fail_link_delete:
            raise GraphStoreError("fake linking cleanup failure")
        del knowledge_base_id
        self.deleted_link_documents.extend(document_ids)

    def delete_document(self, document_id: UUID, *, knowledge_base_id: UUID | None = None) -> None:
        if knowledge_base_id is not None:
            self.entities = {
                entity_id: entity
                for entity_id, entity in self.entities.items()
                if not (
                    entity.document_id == document_id
                    and entity.knowledge_base_id == knowledge_base_id
                )
            }
        else:
            self.entities = {
                entity_id: entity
                for entity_id, entity in self.entities.items()
                if entity.document_id != document_id
            }
        self.deleted.append(document_id)


def _settings() -> Settings:
    return Settings(_env_file=None, environment="test")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_runner_persists_and_resumes_without_reloading_indexed_chunk_as_wrong_type(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A, chunk_count=2)
    second_document = _document(DOCUMENT_B)
    store = _FakeStore()
    first = _FakeExtraction()
    manifest = _run(
        GraphCorpusRunner(
            first,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            link_batch_documents=2,
        ).run([document, second_document])
    )

    assert manifest.status == "completed"
    assert manifest.counters.chunks_attempted == 3
    assert manifest.counters.chunks_completed == 3
    assert all(isinstance(chunk, Chunk) for chunk in first.calls)
    assert store.link_calls == 1

    second = _FakeExtraction()
    resumed = _run(
        GraphCorpusRunner(
            second,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            link_batch_documents=2,
        ).run([document, second_document])
    )

    assert resumed.status == "completed"
    assert second.calls == []
    assert resumed.counters.documents_skipped == 2
    assert resumed.counters.chunks_skipped == 3


def test_retry_failed_retries_only_failed_chunk_and_keeps_successful_graph_state(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A, chunk_count=2)
    failed_chunk_id = document.chunks[0].chunk_id
    store = _FakeStore()
    first = _FakeExtraction(fail_once={failed_chunk_id})
    partial = _run(
        GraphCorpusRunner(
            first,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )
    assert partial.status == "partial_failure"
    assert failed_chunk_id in partial.failed_chunk_ids

    retry = _FakeExtraction()
    complete = _run(
        GraphCorpusRunner(
            retry,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            retry_failed=True,
        ).run([document])
    )

    assert complete.status == "completed"
    assert [chunk.chunk_id for chunk in retry.calls] == [failed_chunk_id]
    assert store.deleted == []


def test_completed_document_with_missing_chunk_checkpoint_is_rebuilt(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A, chunk_count=2)
    store = _FakeStore()
    first = _FakeExtraction()
    _run(
        GraphCorpusRunner(
            first,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    checkpoint_lines = (tmp_path / "chunk-checkpoints.jsonl").read_text().splitlines()
    (tmp_path / "chunk-checkpoints.jsonl").write_text(
        checkpoint_lines[0] + "\n",
        encoding="utf-8",
    )

    repaired = _FakeExtraction()
    manifest = _run(
        GraphCorpusRunner(
            repaired,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert manifest.status == "completed"
    assert store.deleted == [DOCUMENT_A]
    assert [chunk.chunk_id for chunk in repaired.calls] == [
        chunk.chunk_id for chunk in document.chunks
    ]


def test_invalid_utf8_chunk_index_fails_with_bounded_error(tmp_path: Path) -> None:
    chunk_index = tmp_path / "chunks.jsonl"
    chunk_index.write_bytes(b"{\xff\n")

    with pytest.raises(GraphCorpusBuildError, match="valid UTF-8"):
        tuple(iter_indexed_chunks(chunk_index))


def test_graph_persistence_failure_keeps_provider_usage_in_checkpoint_and_manifest(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore(fail_upsert=True)
    manifest = _run(
        GraphCorpusRunner(
            _FakeExtraction(with_usage=True),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert manifest.status == "partial_failure"
    assert manifest.counters.provider_calls == 1
    assert manifest.counters.input_tokens == 100
    assert manifest.counters.output_tokens == 20
    checkpoint = CheckpointStore(tmp_path).get_chunk(document.chunks[0].chunk_id)
    assert checkpoint is not None
    assert checkpoint.attempts == 1
    assert checkpoint.extraction_status == "accepted"
    assert checkpoint.error_category == "persistence"


def test_reprocess_forces_linking_after_document_graph_cleanup(tmp_path: Path) -> None:
    documents = [_document(DOCUMENT_A), _document(DOCUMENT_B)]
    store = _FakeStore()
    _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            link_batch_documents=2,
        ).run(documents)
    )
    assert store.link_calls == 1

    _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            link_batch_documents=2,
            reprocess=True,
        ).run(documents)
    )

    assert store.link_calls == 2
    assert store.deleted == [DOCUMENT_A, DOCUMENT_B]


def test_cancellation_is_durable_and_does_not_look_completed(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A)
    with pytest.raises(asyncio.CancelledError):
        _run(
            GraphCorpusRunner(
                _FakeExtraction(cancel=True),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path,
            ).run([document])
        )

    checkpoints = CheckpointStore(tmp_path)
    assert checkpoints.get_chunk(document.chunks[0].chunk_id).status == "cancelled"
    manifest = json.loads((tmp_path / "run-manifest.json").read_text())
    assert manifest["status"] == "cancelled"


def test_fatal_provider_block_preserves_completed_and_untouched_chunks(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A, chunk_count=5)
    extraction = _ProviderFailureExtraction(status_code=402, retryable=False, successes=1)

    manifest = _run(
        GraphCorpusRunner(
            extraction,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            max_concurrency=1,
        ).run([document])
    )

    assert manifest.status == "provider_blocked"
    assert manifest.provider_blocked is True
    assert manifest.provider_block_reason == ("repeated non-retryable provider API error HTTP 402")
    assert [chunk.chunk_id for chunk in extraction.calls] == [
        document.chunks[index].chunk_id for index in range(4)
    ]
    assert manifest.processed_chunk_ids == [chunk.chunk_id for chunk in document.chunks[:4]]
    assert manifest.failed_chunk_ids == [chunk.chunk_id for chunk in document.chunks[1:4]]

    checkpoints = CheckpointStore(tmp_path)
    document_checkpoint = checkpoints.get_document(DOCUMENT_A)
    assert document_checkpoint is not None
    assert document_checkpoint.status == "provider_blocked"
    assert document_checkpoint.completed_chunk_count == 1
    assert document_checkpoint.failed_chunk_count == 3
    assert document_checkpoint.deferred_chunk_count == 1
    assert document_checkpoint.extraction_complete is False
    assert len(checkpoints._chunks) == 4
    assert checkpoints.get_chunk(document.chunks[0].chunk_id).status == "completed"
    assert checkpoints.get_chunk(document.chunks[4].chunk_id) is None


def test_transient_provider_failure_does_not_trip_fatal_circuit(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A, chunk_count=4)
    extraction = _ProviderFailureExtraction(status_code=503, retryable=True)

    manifest = _run(
        GraphCorpusRunner(
            extraction,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            max_concurrency=1,
        ).run([document])
    )

    assert manifest.status == "partial_failure"
    assert manifest.provider_blocked is False
    assert len(extraction.calls) == 4
    assert len(manifest.failed_chunk_ids) == 4
    assert all(
        CheckpointStore(tmp_path).get_chunk(chunk.chunk_id).error_category == "api"
        for chunk in document.chunks
    )


def test_provider_blocked_run_resumes_failed_and_deferred_work_without_recalling_success(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A, chunk_count=5)
    first = _ProviderFailureExtraction(status_code=403, retryable=False, successes=1)
    blocked = _run(
        GraphCorpusRunner(
            first,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            max_concurrency=1,
        ).run([document])
    )
    assert blocked.status == "provider_blocked"

    recovery = _FakeExtraction()
    resumed = _run(
        GraphCorpusRunner(
            recovery,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            max_concurrency=1,
            retry_failed=True,
        ).run([document])
    )

    assert resumed.status == "completed"
    assert resumed.provider_blocked is False
    assert [chunk.chunk_id for chunk in recovery.calls] == [
        chunk.chunk_id for chunk in document.chunks[1:]
    ]
    checkpoints = CheckpointStore(tmp_path)
    assert all(
        checkpoints.get_chunk(chunk.chunk_id).status == "completed" for chunk in document.chunks
    )


def test_a21_graph_coverage_counts_supported_chunks_from_canonical_blocks(tmp_path: Path) -> None:
    from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock

    canonical = CanonicalDocument(
        document_id=DOCUMENT_A,
        pages=[
            Page(
                page_number=1,
                blocks=[TextBlock(block_id="answer", reading_order=0, text="答案内容")],
            )
        ],
    )
    item = make_eval_item(
        query="这个事实是什么？",
        subject="测试",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=DOCUMENT_A,
                page_number=1,
                source_block_ids=["answer"],
            )
        ],
        document_lookup={DOCUMENT_A: canonical},
        verification_status="verified",
    )
    record = FinalDatasetItem(
        split="dev",
        leakage_group_id="group-1",
        item=item,
    )
    # The function reads JSONL, so use a temporary file without retaining corpus text.
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    coverage = build_a21_graph_coverage(
        dataset_path,
        source_block_to_chunks={(str(DOCUMENT_A), "answer"): ["chunk-answer"]},
        graph_supported_chunk_ids=["chunk-answer"],
    )
    assert coverage.item_count == 1
    assert coverage.queries_with_complete_graph_coverage == 1
    assert coverage.queries_with_no_graph_coverage == 0


def test_full_run_estimate_is_transparent_and_cost_is_unknown_without_pricing(
    tmp_path: Path,
) -> None:
    summary_paths: list[Path] = []
    for index, (chunks, calls) in enumerate(((2, 3), (1, 1))):
        path = tmp_path / f"sample-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "attempted_chunks": chunks,
                    "provider_calls": calls,
                    "input_tokens": chunks * 100,
                    "output_tokens": chunks * 20,
                    "run_elapsed_ms": chunks * 1_000,
                }
            ),
            encoding="utf-8",
        )
        summary_paths.append(path)

    estimate = estimate_full_run(
        summary_paths,
        target_chunks=6,
        settings=_settings(),
        max_concurrency=2,
    )

    assert estimate.measured_sample_chunks == 3
    assert estimate.measured_provider_calls == 4
    assert estimate.measured_retry_calls == 1
    assert estimate.expected_provider_calls_at_observed_rate == 8
    assert estimate.estimated_cost_at_observed_rate is None
    assert estimate.bounded_wall_clock_seconds_at_max_concurrency == 3.0


def test_manifest_deduplicates_same_content_and_rejects_conflicting_content(
    tmp_path: Path,
) -> None:
    document_id = UUID("44444444-4444-4444-8444-444444444444")
    common = {
        "inventory_status": "ready",
        "benchmark_document_uuid": str(document_id),
        "subject": "测试",
        "size_bytes": 10,
        "sha256": "a" * 64,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        **common,
                        "benchmark_item_id": "item-b",
                        "relative_path": "corpus/b.pdf",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        **common,
                        "benchmark_item_id": "item-a",
                        "relative_path": "corpus/a.pdf",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = load_corpus_metadata(manifest)
    assert len(metadata) == 1
    assert metadata[0].benchmark_item_id == "item-a"
    assert metadata[0].relative_path == "corpus/a.pdf"

    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        **common,
                        "benchmark_item_id": "item-a",
                        "relative_path": "corpus/a.pdf",
                    }
                ),
                json.dumps(
                    {
                        **common,
                        "benchmark_item_id": "item-c",
                        "relative_path": "corpus/c.pdf",
                        "size_bytes": 11,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GraphCorpusBuildError, match="conflicting content"):
        load_corpus_metadata(manifest)


def test_fresh_checkpoint_directory_cleans_existing_scoped_graph_state(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore()
    _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path / "first",
            store=store,
        ).run([document])
    )
    assert store.has_document_state(KNOWLEDGE_BASE_ID, DOCUMENT_A)

    _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path / "fresh",
            store=store,
            reprocess=True,
        ).run([document])
    )

    assert store.deleted == [DOCUMENT_A]


def test_fresh_checkpoint_directory_requires_explicit_reprocess(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore()
    _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path / "first",
            store=store,
        ).run([document])
    )

    with pytest.raises(GraphCorpusBuildError, match="use --reprocess"):
        _run(
            GraphCorpusRunner(
                _FakeExtraction(),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path / "fresh",
                store=store,
            ).run([document])
        )

    assert store.deleted == []


def test_cancelled_document_is_durable_and_safe_to_resume_with_cleanup(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A, chunk_count=2)
    store = _FakeStore()
    with pytest.raises(asyncio.CancelledError):
        _run(
            GraphCorpusRunner(
                _CancelAfterFirstExtraction(),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path,
                store=store,
            ).run([document])
        )

    cancelled = CheckpointStore(tmp_path).get_document(DOCUMENT_A)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.extraction_complete is False
    assert store.has_document_state(KNOWLEDGE_BASE_ID, DOCUMENT_A)

    manifest = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert manifest.status == "completed"
    assert store.deleted == [DOCUMENT_A]


def test_checkpoint_write_failure_requires_explicit_reprocess_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore()
    original_save_chunk = CheckpointStore.save_chunk
    failure_budget = 2

    def fail_checkpoint_write(
        checkpoint_store: CheckpointStore,
        record: object,
    ) -> None:
        nonlocal failure_budget
        if checkpoint_store.output_dir == tmp_path and failure_budget:
            failure_budget -= 1
            raise GraphCorpusBuildError("simulated checkpoint failure")
        original_save_chunk(checkpoint_store, record)  # type: ignore[arg-type]

    monkeypatch.setattr(CheckpointStore, "save_chunk", fail_checkpoint_write)
    with pytest.raises(GraphCorpusBuildError):
        _run(
            GraphCorpusRunner(
                _FakeExtraction(),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path,
                store=store,
            ).run([document])
        )

    assert store.has_document_state(KNOWLEDGE_BASE_ID, DOCUMENT_A)
    with pytest.raises(GraphCorpusBuildError, match="use --reprocess"):
        _run(
            GraphCorpusRunner(
                _FakeExtraction(),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path,
                store=store,
            ).run([document])
        )

    recovered = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            reprocess=True,
        ).run([document])
    )
    assert recovered.status == "completed"
    assert store.deleted == [DOCUMENT_A]


def test_transient_checkpoint_failure_preserves_extraction_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore()
    original_save_chunk = CheckpointStore.save_chunk
    failure_budget = 1

    def fail_once(
        checkpoint_store: CheckpointStore,
        record: object,
    ) -> None:
        nonlocal failure_budget
        if checkpoint_store.output_dir == tmp_path and failure_budget:
            failure_budget -= 1
            raise GraphCorpusBuildError("simulated transient checkpoint failure")
        original_save_chunk(checkpoint_store, record)  # type: ignore[arg-type]

    monkeypatch.setattr(CheckpointStore, "save_chunk", fail_once)
    manifest = _run(
        GraphCorpusRunner(
            _FakeExtraction(with_usage=True),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert manifest.status == "partial_failure"
    assert manifest.counters.provider_calls == 1
    assert manifest.counters.input_tokens == 100
    assert manifest.counters.output_tokens == 20
    checkpoint = CheckpointStore(tmp_path).get_chunk(document.chunks[0].chunk_id)
    assert checkpoint is not None
    assert checkpoint.provider_calls == 1
    assert checkpoint.input_tokens == 100
    assert checkpoint.output_tokens == 20


def test_manifest_write_failure_resumes_from_durable_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore()
    first = GraphCorpusRunner(
        _FakeExtraction(),
        _settings(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        output_dir=tmp_path,
        store=store,
    )
    original_save_manifest = first.manifest_store.save
    save_calls = 0

    def fail_second_manifest_save(manifest: object) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise GraphCorpusBuildError("simulated manifest failure")
        original_save_manifest(manifest)  # type: ignore[arg-type]

    monkeypatch.setattr(first.manifest_store, "save", fail_second_manifest_save)
    with pytest.raises(GraphCorpusBuildError, match="simulated manifest failure"):
        _run(first.run([document]))

    resumed_extraction = _FakeExtraction()
    resumed = _run(
        GraphCorpusRunner(
            resumed_extraction,
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert resumed.status == "completed"
    assert resumed_extraction.calls == []


def test_recovered_entity_listing_replaces_stale_link_failure_batch(
    tmp_path: Path,
) -> None:
    document = _document(DOCUMENT_A)
    store = _FakeStore(fail_entity_listing_once=True)
    first = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )
    assert first.status == "partial_failure"
    assert first.failed_link_batch_keys

    recovered = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    assert recovered.status == "completed"
    assert recovered.failed_link_batch_keys == []


def test_schema_rejected_chunk_blocks_document_linking(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A)

    class _SchemaRejectedExtraction:
        async def extract_chunk(
            self, chunk: Chunk, *, knowledge_base_id: UUID
        ) -> ChunkExtractionResult:
            assert knowledge_base_id == KNOWLEDGE_BASE_ID
            return ChunkExtractionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                status="schema_rejected",
                entities=(),
                relations=(),
                stats=ExtractionStats(
                    attempts=1,
                    parse_failures=0,
                    schema_failures=1,
                    grounding_rejections=0,
                    duplicate_entities=0,
                    duplicate_relations=0,
                ),
                attempts=(
                    ExtractionAttempt(
                        number=1,
                        category="schema_validation",
                        details=("schema_validation", "entities[0].entity_type: missing"),
                    ),
                ),
            )

    store = _FakeStore()
    manifest = _run(
        GraphCorpusRunner(
            _SchemaRejectedExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    checkpoint = CheckpointStore(tmp_path).get_document(DOCUMENT_A)
    assert checkpoint is not None
    assert checkpoint.extraction_complete is False
    assert checkpoint.extraction_partial is True
    assert checkpoint.linking_complete is False
    assert checkpoint.linking_blocked is True
    chunk_checkpoint = CheckpointStore(tmp_path).get_chunk(document.chunks[0].chunk_id)
    assert chunk_checkpoint is not None
    assert chunk_checkpoint.error_category == "schema_validation"
    assert chunk_checkpoint.error == "schema_validation; entities[0].entity_type: missing"
    assert manifest.status == "partial_failure"
    assert store.link_calls == 0


def test_runner_accounts_truncation_and_corrective_retries_separately(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A)

    class _RetryAccountingExtraction:
        async def extract_chunk(
            self, chunk: Chunk, *, knowledge_base_id: UUID
        ) -> ChunkExtractionResult:
            base = _result(chunk, with_usage=True)
            first, second = base.llm_results[0], base.llm_results[0]
            return replace(
                base,
                stats=replace(base.stats, attempts=3, truncation_retries=2),
                llm_results=(first, second, first),
                attempts=(
                    ExtractionAttempt(
                        number=1,
                        category="truncated_response",
                        max_tokens=1024,
                        finish_reason="length",
                        output_tokens=1024,
                    ),
                    ExtractionAttempt(
                        number=2,
                        category="truncated_response",
                        max_tokens=2048,
                        finish_reason="length",
                        output_tokens=2048,
                    ),
                    ExtractionAttempt(
                        number=3,
                        category="valid_extraction",
                        max_tokens=4096,
                        finish_reason="stop",
                        output_tokens=20,
                    ),
                ),
            )

    manifest = _run(
        GraphCorpusRunner(
            _RetryAccountingExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=_FakeStore(),
        ).run([document])
    )

    checkpoint = CheckpointStore(tmp_path).get_chunk(_chunk(DOCUMENT_A).chunk_id)
    assert checkpoint is not None
    assert checkpoint.provider_calls == 3
    assert checkpoint.retry_count == 2
    assert checkpoint.extraction_initial_calls == 1
    assert checkpoint.extraction_truncation_retry_calls == 2
    assert checkpoint.extraction_corrective_retry_calls == 0
    assert checkpoint.extraction_provider_retry_calls == 0
    assert manifest.counters.extraction_truncation_retry_calls == 2
    assert manifest.counters.extraction_corrective_retry_calls == 0


def test_failed_linking_requires_explicit_retry_failed(tmp_path: Path) -> None:
    documents = [_document(DOCUMENT_A), _document(DOCUMENT_B)]
    store = _FakeStore(fail_link_upsert=True)
    first = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run(documents)
    )
    assert first.status == "partial_failure"

    store.fail_link_upsert = False
    not_retried = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run(documents)
    )
    assert not_retried.status == "partial_failure"
    assert store.link_calls == 0

    retried = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            retry_failed=True,
        ).run(documents)
    )
    assert retried.status == "completed"
    assert store.link_calls == 1


def test_link_cleanup_failure_marks_documents_linking_blocked(tmp_path: Path) -> None:
    documents = [_document(DOCUMENT_A), _document(DOCUMENT_B)]
    store = _FakeStore(fail_link_upsert=True)
    first = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run(documents)
    )
    assert first.status == "partial_failure"

    store.fail_link_upsert = False
    store.fail_link_delete = True
    blocked = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
            retry_failed=True,
        ).run(documents)
    )

    checkpoint = CheckpointStore(tmp_path).get_document(DOCUMENT_A)
    assert checkpoint is not None
    assert blocked.status == "partial_failure"
    assert checkpoint.linking_complete is False
    assert checkpoint.linking_blocked is True
    assert store.link_calls == 0


def test_failed_chunk_marks_document_partial_and_blocks_linking(tmp_path: Path) -> None:
    document = _document(DOCUMENT_A, chunk_count=2)
    store = _FakeStore()
    failed_chunk_id = document.chunks[0].chunk_id
    manifest = _run(
        GraphCorpusRunner(
            _FakeExtraction(fail_once={failed_chunk_id}),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=store,
        ).run([document])
    )

    checkpoint = CheckpointStore(tmp_path).get_document(DOCUMENT_A)
    assert checkpoint is not None
    assert checkpoint.extraction_complete is False
    assert checkpoint.extraction_partial is True
    assert checkpoint.linking_complete is False
    assert checkpoint.linking_blocked is True
    assert manifest.status == "partial_failure"
    assert store.link_calls == 0
    assert store.deleted_link_documents == [DOCUMENT_A]


def test_fresh_run_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    first = GraphCorpusRunner(
        _FakeExtraction(),
        _settings(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        output_dir=tmp_path,
    )
    try:
        with pytest.raises(GraphCorpusBuildError, match="owns this checkpoint directory"):
            GraphCorpusRunner(
                _FakeExtraction(),
                _settings(),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                output_dir=tmp_path,
            )
    finally:
        first.close()


def test_complete_input_snapshot_rejects_truncated_selection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    manifest_path.write_text("manifest", encoding="utf-8")
    chunk_path.write_text("chunks", encoding="utf-8")
    documents = [_document(DOCUMENT_A), _document(DOCUMENT_B)]
    expected = build_corpus_input_snapshot(
        documents,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        corpus_manifest=manifest_path,
        chunk_index=chunk_path,
    )
    actual = build_corpus_input_snapshot(
        documents[:1],
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        corpus_manifest=manifest_path,
        chunk_index=chunk_path,
    )

    with pytest.raises(GraphCorpusBuildError, match="incomplete"):
        validate_corpus_input_snapshot(actual, expected, require_full=True)


def test_pipeline_fingerprint_includes_endpoint_timeout_and_linking_policy() -> None:
    settings = _settings()
    base = build_pipeline_fingerprint(
        settings,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        input_fingerprint="a" * 64,
        linking_mode="llm_adjudication",
    )
    changed_timeout = settings.model_copy(update={"llm_timeout_seconds": 31.0})
    changed_endpoint = settings.model_copy(update={"llm_base_url": "https://example.test/v1"})
    changed_truncation = Settings(
        _env_file=None,
        graph_extraction_truncation_budgets=(1024, 2048, 8192),
    )

    assert base != build_pipeline_fingerprint(
        changed_timeout,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        input_fingerprint="a" * 64,
        linking_mode="llm_adjudication",
    )
    assert base != build_pipeline_fingerprint(
        changed_endpoint,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        input_fingerprint="a" * 64,
        linking_mode="llm_adjudication",
    )
    assert base != build_pipeline_fingerprint(
        changed_truncation,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        input_fingerprint="a" * 64,
        linking_mode="llm_adjudication",
    )


def test_prompt_layout_change_cannot_silently_resume_with_same_pipeline_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_scope.evaluation import graph_corpus_build

    runner = GraphCorpusRunner(
        _FakeExtraction(),
        _settings(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        output_dir=tmp_path,
    )
    manifest = _run(runner.run([_document(DOCUMENT_A)]))
    assert manifest.prompt_layout_version == "legacy-v1"

    manifest_path = tmp_path / "run-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_fingerprint = payload["pipeline_fingerprint"]
    payload.pop("prompt_layout_version")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = GraphCorpusRunner(
        _FakeExtraction(),
        _settings(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        output_dir=tmp_path,
    )
    try:
        assert resumed.manifest.prompt_layout_version is None
        assert stored_fingerprint == build_pipeline_fingerprint(
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            max_concurrency=1,
            link_batch_documents=8,
            linking_mode="manual_review_only",
        )
    finally:
        resumed.close()

    monkeypatch.setattr(graph_corpus_build, "EXTRACTION_PROMPT_LAYOUT_VERSION", "cache-v2")
    with pytest.raises(GraphCorpusBuildError, match="prompt serialization layout differs"):
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
        )


def test_failed_provider_attempt_summary_separates_chunks_attempts_and_buckets() -> None:
    def failed(chunk_id: str, timestamp: datetime, provider_calls: int) -> ChunkCheckpoint:
        return ChunkCheckpoint(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_A,
            chunk_id=chunk_id,
            input_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
            pipeline_fingerprint="c" * 64,
            status="failed",
            attempts=1,
            retry_count=0,
            provider_calls=provider_calls,
            error_category="api",
            updated_at=timestamp,
        )

    summary = summarize_failed_provider_attempts(
        [
            failed("chunk-a", datetime(2026, 9, 10, 14, 34, 27, tzinfo=UTC), 3),
            failed("chunk-a", datetime(2026, 9, 10, 14, 36, 2, tzinfo=UTC), 1),
            failed("chunk-b", datetime(2026, 9, 10, 14, 39, 9, tzinfo=UTC), 2),
            ChunkCheckpoint(
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                document_id=DOCUMENT_A,
                chunk_id="completed",
                input_fingerprint="a" * 64,
                config_fingerprint="b" * 64,
                pipeline_fingerprint="c" * 64,
                status="completed",
                attempts=1,
                retry_count=0,
                provider_calls=1,
            ),
        ],
    )

    assert isinstance(summary, FailedProviderAttemptSummary)
    assert summary.unique_failed_chunks == 2
    assert summary.failed_provider_attempts == 6
    assert dict(summary.failures_by_time_bucket) == {
        "2026-09-10T14:30:00Z": 3,
        "2026-09-10T14:35:00Z": 3,
    }


def test_maximum_extraction_attempt_formula_includes_all_retry_categories() -> None:
    settings = Settings(
        _env_file=None,
        llm_max_retries=2,
        graph_extraction_max_parse_retries=1,
        graph_extraction_truncation_budgets=(1024, 2048, 4096),
    )

    assert maximum_extraction_provider_attempts(settings) == 12
    assert maximum_extraction_provider_attempts(settings, corpus_retry=True) == 24


def test_resume_rejects_a_changed_truncation_policy(tmp_path: Path) -> None:
    first = GraphCorpusRunner(
        _FakeExtraction(),
        _settings(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        output_dir=tmp_path,
    )
    try:
        _run(first.run([_document(DOCUMENT_A)]))
    finally:
        first.close()

    changed_settings = Settings(
        _env_file=None,
        environment="test",
        graph_extraction_truncation_budgets=(1024, 2048, 8192),
    )
    with pytest.raises(GraphCorpusBuildError, match="pipeline fingerprint differs"):
        GraphCorpusRunner(
            _FakeExtraction(),
            changed_settings,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
        )


def test_corpus_input_fingerprint_is_independent_of_target_knowledge_base() -> None:
    first = build_corpus_input_snapshot(
        [_document(DOCUMENT_A)],
        knowledge_base_id=KNOWLEDGE_BASE_ID,
    )
    second = build_corpus_input_snapshot(
        [_document(DOCUMENT_A)],
        knowledge_base_id=UUID("55555555-5555-4555-8555-555555555555"),
    )

    assert first.knowledge_base_id != second.knowledge_base_id
    assert first.input_fingerprint == second.input_fingerprint


def test_linking_usage_is_counted_when_neo4j_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_scope.evaluation import graph_corpus_build

    stats = SimpleNamespace(
        candidate_count=1,
        link_count=1,
        no_link_count=0,
        uncertain_count=0,
        llm_adjudications=2,
        provider_attempts=3,
        provider_retry_calls=1,
        llm_failures=1,
        input_tokens=100,
        output_tokens=20,
        estimated_cost=None,
    )
    linking_run = SimpleNamespace(
        stats=stats,
        decisions=[object(), object()],
        canonical_entities=[object()],
        mappings=[object()],
    )

    async def fake_link_entities(*_args: Any, **_kwargs: Any) -> Any:
        return linking_run

    monkeypatch.setattr(graph_corpus_build, "link_entities", fake_link_entities)
    manifest = _run(
        GraphCorpusRunner(
            _FakeExtraction(),
            _settings(),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            output_dir=tmp_path,
            store=_FakeStore(fail_link_upsert=True),
            linking_gateway=object(),
            link_batch_documents=2,
        ).run([_document(DOCUMENT_A), _document(DOCUMENT_B)])
    )

    assert manifest.status == "partial_failure"
    # Two extraction calls plus three linking provider attempts are all
    # reflected in the run-level total; the linking-specific fields below
    # isolate the latter.
    assert manifest.counters.provider_calls == 5
    assert manifest.counters.linking_initial_calls == 2
    assert manifest.counters.linking_provider_retry_calls == 1
    assert manifest.counters.linking_failed_attempts == 1
    link_checkpoint = next(iter(CheckpointStore(tmp_path)._links.values()))
    assert link_checkpoint.initial_calls == 2
    assert link_checkpoint.provider_retry_calls == 1
