from __future__ import annotations

import pytest

from knowledge_scope.evaluation.retrieval_metrics import (
    evidence_recall_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    ranking_metrics,
)


def test_ranking_metrics_deduplicate_retrieved_chunk_ids() -> None:
    retrieved = ["chunk-a", "chunk-a", "chunk-b", "chunk-c"]
    relevant = {"chunk-b"}
    chunk_source_blocks = {
        "chunk-a": {("doc", "block-1")},
        "chunk-b": {("doc", "block-2")},
        "chunk-c": {("doc", "block-3")},
    }

    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 3) == 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert (
        evidence_recall_at_k(
            retrieved,
            chunk_source_blocks,
            {("doc", "block-2"), ("doc", "block-3")},
            3,
        )
        == 1.0
    )

    metrics = ranking_metrics(
        retrieved,
        relevant,
        chunk_source_blocks,
        {("doc", "block-2"), ("doc", "block-3")},
    )
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["evidence_recall@1"] == 0.0
    assert metrics["evidence_recall@3"] == 1.0
    assert metrics["mrr"] == 0.5


def test_evidence_recall_counts_unique_gold_source_blocks() -> None:
    assert evidence_recall_at_k(
        ["chunk-1", "chunk-2"],
        {
            "chunk-1": {("doc", "block-1"), ("doc", "block-2")},
            "chunk-2": {("doc", "block-2"), ("doc", "block-3")},
        },
        {("doc", "block-1"), ("doc", "block-2"), ("doc", "block-3")},
        1,
    ) == pytest.approx(2 / 3)


def test_metrics_reject_invalid_cutoff_and_empty_evidence_is_zero() -> None:
    with pytest.raises(ValueError, match="at least one"):
        hit_at_k([], set(), 0)
    assert evidence_recall_at_k([], {}, set(), 1) == 0.0
