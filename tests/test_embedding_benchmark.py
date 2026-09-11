"""Focused tests for the A2.2 benchmark protocol and frozen-input handling."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.evaluation.embedding_benchmark import (
    MODEL_SPECS,
    EmbeddingBenchmarkError,
    EmbeddingBenchmarkProtocol,
    _derive_relevant_chunks,
    _query_texts,
    load_frozen_chunk_index,
    load_frozen_eval_cases,
)
from knowledge_scope.evaluation.retrieval_eval import (
    IndexedChunk,
    RetrievalEvalItem,
    RetrievalEvidence,
    deterministic_item_id,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _item() -> RetrievalEvalItem:
    evidence = [
        RetrievalEvidence(
            document_id=DOCUMENT_ID,
            page_number=1,
            source_block_ids=["p1-b2"],
        )
    ]
    query = "答案是什么\uff1f"
    fingerprint = "a" * 64
    return RetrievalEvalItem(
        item_id=deterministic_item_id(query, "数学", "factual", evidence, fingerprint),
        query=query,
        subject="数学",
        query_type="factual",
        verification_status="verified",
        evidence=evidence,
        query_source=[
            RetrievalEvidence(
                document_id=DOCUMENT_ID,
                page_number=1,
                source_block_ids=["p1-b1"],
            )
        ],
        evidence_fingerprint=fingerprint,
    )


def test_model_specs_keep_official_query_conventions() -> None:
    assert MODEL_SPECS["qwen3-embedding-0.6b"].query_mode == "qwen_prompt_name"
    assert MODEL_SPECS["qwen3-embedding-4b"].query_mode == "qwen_prompt_name"
    assert MODEL_SPECS["bge-m3"].query_mode == "none"
    assert MODEL_SPECS["multilingual-e5-large-instruct"].query_mode == "e5_instruction"

    raw, qwen_detail = _query_texts(MODEL_SPECS["qwen3-embedding-0.6b"], ["问题"])
    assert raw == ["问题"]
    assert "prompt_name=query" in qwen_detail

    e5, e5_detail = _query_texts(MODEL_SPECS["multilingual-e5-large-instruct"], ["问题"])
    assert e5 == [
        "Instruct: Given a textbook question, retrieve passages that contain its answer\n"
        "Query: 问题"
    ]
    assert "Instruct + Query" in e5_detail

    bge, bge_detail = _query_texts(MODEL_SPECS["bge-m3"], ["问题"])
    assert bge == ["问题"]
    assert "no query instruction" in bge_detail


def test_protocol_keeps_existing_a21_metric_cutoffs() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        EmbeddingBenchmarkProtocol(top_k=(1, 5))


def test_relevant_chunk_derivation_excludes_query_source_chunks() -> None:
    item = _item()
    chunks = {
        "chunk-with-question": IndexedChunk(
            schema_version="1.0",
            chunk_id="chunk-with-question",
            document_id=DOCUMENT_ID,
            ordinal=0,
            text="问题和答案",
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1", "p1-b2"],
            section_path=[],
            content_types=["text"],
            asset_refs=[],
            config_fingerprint="b" * 64,
        ),
        "answer-only": IndexedChunk(
            schema_version="1.0",
            chunk_id="answer-only",
            document_id=DOCUMENT_ID,
            ordinal=1,
            text="答案",
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b2"],
            section_path=[],
            content_types=["text"],
            asset_refs=[],
            config_fingerprint="b" * 64,
        ),
    }

    assert _derive_relevant_chunks(item, chunks) == frozenset({"answer-only"})


def test_chunk_index_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    chunk = IndexedChunk(
        schema_version="1.0",
        chunk_id="duplicate",
        document_id=DOCUMENT_ID,
        ordinal=0,
        text="正文",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        section_path=[],
        content_types=["text"],
        asset_refs=[],
        config_fingerprint="c" * 64,
    )
    path = tmp_path / "chunks.jsonl"
    payload = chunk.model_dump_json() + "\n" + chunk.model_dump_json() + "\n"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate chunk IDs"):
        load_frozen_chunk_index(path)


def test_frozen_loader_rejects_duplicate_materialized_items(tmp_path: Path) -> None:
    source = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
    lines = source.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "materialized.jsonl"
    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(EmbeddingBenchmarkError, match="duplicate item ID"):
        load_frozen_eval_cases("dev", materialized_path=path)
