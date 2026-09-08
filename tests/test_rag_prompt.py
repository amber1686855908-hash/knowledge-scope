from __future__ import annotations

from uuid import UUID, uuid4

from knowledge_scope.rag.context import assemble_context
from knowledge_scope.rag.prompt import RAG_PROMPT_VERSION, build_rag_messages
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk


def _context(text: str) -> object:
    chunk = RerankedChunk(
        chunk=RetrievedChunk(
            point_id=uuid4(),
            score=1.0,
            payload=ChunkVectorPayload(
                collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
                chunk_id="chunk-1",
                document_id=UUID("22222222-2222-4222-8222-222222222222"),
                page_start=1,
                page_end=1,
                source_block_ids=["block-1"],
                section_path=["第一章", "概念"],
                content_types=["text"],
                asset_refs=[],
                text=text,
                chunking_config_fingerprint="a" * 64,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_model_revision="revision",
                embedding_config_fingerprint="b" * 64,
            ),
        ),
        dense_rank=1,
        reranker_score=1.0,
    )
    return assemble_context([chunk], budget_chars=100)


def test_rag_prompt_is_versioned_and_contains_only_application_markers() -> None:
    messages = build_rag_messages("什么是概念?", _context("概念是对事物本质特征的概括。"))

    assert len(messages) == 2
    assert RAG_PROMPT_VERSION in messages[0].content
    assert "只能依据" in messages[0].content
    assert "[C1]" in messages[1].content
    assert "document_id=22222222-2222-4222-8222-222222222222" in messages[1].content
    assert "什么是概念?" in messages[1].content


def test_empty_context_prompt_requires_insufficient_evidence_response() -> None:
    messages = build_rag_messages("问题", _context(""))

    assert "没有检索到可用的文本资料" in messages[1].content
    assert "当前检索到的资料不足以回答该问题" in messages[0].content
