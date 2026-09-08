from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.evaluation.embedding_benchmark import FrozenEvalCase
from knowledge_scope.evaluation.retrieval_eval import (
    IndexedChunk,
    RetrievalEvalItem,
    RetrievalEvidence,
    deterministic_item_id,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    BGE_RERANKER_MODEL_ID,
    RetrievalSystemBenchmarkProtocol,
    classify_bad_case,
    compare_dense_and_qdrant_rankings,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _case(*, query_type: str = "factual", gold: tuple[str, ...] = ("answer",)) -> FrozenEvalCase:
    evidence = RetrievalEvidence(
        document_id=DOCUMENT_ID,
        page_number=1,
        source_block_ids=["p1-b2"],
    )
    query = "问题是什么?"
    item = RetrievalEvalItem(
        item_id=deterministic_item_id(query, "数学", query_type, [evidence], "a" * 64),
        query=query,
        subject="数学",
        query_type=query_type,
        verification_status="verified",
        evidence=[evidence],
        query_source=[],
        evidence_fingerprint="a" * 64,
    )
    return FrozenEvalCase(
        item=item,
        relevant_chunk_ids=frozenset(gold),
        gold_source_blocks=frozenset({(str(DOCUMENT_ID), "p1-b2")}),
    )


def _chunks(*chunk_ids: str) -> dict[str, IndexedChunk]:
    return {
        chunk_id: IndexedChunk(
            schema_version="1.0",
            chunk_id=chunk_id,
            document_id=DOCUMENT_ID,
            ordinal=index,
            text=f"正文 {chunk_id}",
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b2"],
            section_path=["章节"],
            content_types=["text"],
            asset_refs=[],
            config_fingerprint="b" * 64,
        )
        for index, chunk_id in enumerate(chunk_ids)
    }


def test_protocol_is_fixed_to_the_strict_top10_profile() -> None:
    assert RetrievalSystemBenchmarkProtocol().dense_top_k == 10
    with pytest.raises(ValidationError, match="strict dense"):
        RetrievalSystemBenchmarkProtocol(dense_top_k=20)
    with pytest.raises(ValidationError, match="must remain 512"):
        RetrievalSystemBenchmarkProtocol(reranker_max_seq_length=1024)
    with pytest.raises(ValidationError):
        RetrievalSystemBenchmarkProtocol(unexpected="value")


def test_qdrant_comparison_reports_agreement_and_metric_delta() -> None:
    exact = {"one": ("answer", "other")}
    qdrant = {"one": ("answer", "other")}
    comparison = compare_dense_and_qdrant_rankings(
        "test",
        exact,
        qdrant,
        {"mrr": 1.0, "hit@1": 1.0},
        {"mrr": 1.0, "hit@1": 1.0},
        ann_mode="full_scan",
        document_filter_queries=1,
        document_filter_empty_count=0,
        document_filter_violations=0,
    )

    assert comparison.identical_order_rate == 1.0
    assert comparison.mean_top10_set_overlap == 0.2
    assert comparison.ann_related_miss_count == 0
    assert comparison.quality_delta == {"hit@1": 0.0, "mrr": 0.0}


def test_bad_case_distinguishes_dense_absence_from_reranker_ordering() -> None:
    chunks = _chunks("answer", "other")
    case = _case()

    absent = classify_bad_case(case, ["other"], ["other"], chunks)
    assert absent is not None
    assert absent.categories == ("evidence_absent_from_dense_top10",)
    assert absent.bge_top10_has_gold is False

    lowered = classify_bad_case(
        case,
        ["answer", "other"],
        ["other", "answer"],
        chunks,
        truncated_chunk_ids={"answer"},
    )
    assert lowered is not None
    assert "reranker_ranked_relevant_lower" in lowered.categories
    assert "long_or_truncated_relevant_candidate" in lowered.categories
    assert lowered.bge_top10_has_gold is True


def test_bge_model_identity_is_the_selected_a24_reranker() -> None:
    assert BGE_RERANKER_MODEL_ID == "BAAI/bge-reranker-v2-m3"
