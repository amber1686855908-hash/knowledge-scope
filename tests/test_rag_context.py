from __future__ import annotations

from uuid import UUID, uuid4

from knowledge_scope.rag.context import assemble_context
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk


def _ranked(chunk_id: str, text: str, source_block_ids: list[str], rank: int) -> RerankedChunk:
    document_id = UUID("11111111-1111-4111-8111-111111111111")
    return RerankedChunk(
        chunk=RetrievedChunk(
            point_id=uuid4(),
            score=1.0 / rank,
            payload=ChunkVectorPayload(
                collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
                chunk_id=chunk_id,
                document_id=document_id,
                knowledge_base_id=None,
                page_start=rank,
                page_end=rank,
                source_block_ids=source_block_ids,
                section_path=["章节", chunk_id],
                content_types=["text"],
                asset_refs=[],
                text=text,
                chunking_config_fingerprint="a" * 64,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_model_revision="revision",
                embedding_config_fingerprint="b" * 64,
            ),
        ),
        dense_rank=rank,
        reranker_score=float(10 - rank),
    )


def test_context_suppresses_exact_duplicate_source_block_text() -> None:
    selection = assemble_context(
        [
            _ranked("chunk-1", "第一段答案", ["block-1", "block-2"], 1),
            _ranked("chunk-2", "第一段答案", ["block-2"], 2),
            _ranked("chunk-3", "第三段答案", ["block-3"], 3),
        ],
        budget_chars=100,
    )

    assert [item.citation.chunk_id for item in selection.items] == ["chunk-1", "chunk-3"]
    assert [item.citation.marker for item in selection.items] == ["C1", "C2"]
    assert selection.items[0].citation.source_block_ids == ["block-1", "block-2"]
    assert selection.render().count("第一段答案") == 1


def test_context_keeps_distinct_chunks_from_one_split_source_block() -> None:
    selection = assemble_context(
        [
            _ranked("chunk-1", "同一正文的前半段", ["block-1"], 1),
            _ranked("chunk-2", "同一正文的后半段", ["block-1"], 2),
        ],
        budget_chars=100,
    )

    assert [item.citation.chunk_id for item in selection.items] == ["chunk-1", "chunk-2"]
    assert selection.render().count("同一正文") == 2


def test_context_suppresses_exact_duplicate_text_with_overlapping_lineage() -> None:
    selection = assemble_context(
        [
            _ranked("chunk-1", "重复正文", ["block-1"], 1),
            _ranked("chunk-2", "  重复正文  ", ["block-1", "block-2"], 2),
            _ranked("chunk-3", "新的正文", ["block-2"], 3),
        ],
        budget_chars=100,
    )

    assert [item.citation.chunk_id for item in selection.items] == ["chunk-1", "chunk-3"]


def test_context_skips_asset_only_and_oversized_chunks() -> None:
    selection = assemble_context(
        [
            _ranked("asset-only", "", ["asset-block"], 1),
            _ranked("chunk-1", "abcdefghij", ["text-block"], 2),
            _ranked("chunk-2", "后续", ["later-block"], 3),
        ],
        budget_chars=4,
    )

    assert [item.citation.chunk_id for item in selection.items] == ["chunk-2"]
    assert selection.character_count == 2
    assert selection.items[0].text == "后续"
    assert selection.items[0].truncated is False


def test_context_returns_no_citation_when_every_chunk_exceeds_budget() -> None:
    selection = assemble_context(
        [_ranked("chunk-1", "abcdefghij", ["text-block"], 1)],
        budget_chars=4,
    )

    assert selection.items == ()
    assert selection.character_count == 0
