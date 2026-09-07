"""Pure ranking metrics for the A2.1 retrieval evaluation set."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

SourceBlockKey = tuple[str, str]


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least one")


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Collection[str],
    k: int,
) -> float:
    """Return binary Hit@K for a ranked list of chunk IDs."""
    _validate_k(k)
    top_k = _unique_in_order(retrieved_chunk_ids)[:k]
    return float(any(chunk_id in relevant_chunk_ids for chunk_id in top_k))


def mean_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Collection[str],
) -> float:
    """Return reciprocal rank of the first relevant chunk, or zero."""
    for rank, chunk_id in enumerate(_unique_in_order(retrieved_chunk_ids), start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evidence_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    chunk_source_blocks: Mapping[str, Collection[SourceBlockKey]],
    gold_source_blocks: Collection[SourceBlockKey],
    k: int,
) -> float:
    """Return the fraction of unique gold source blocks covered by top-K chunks."""
    _validate_k(k)
    gold = set(gold_source_blocks)
    if not gold:
        return 0.0
    covered: set[SourceBlockKey] = set()
    for chunk_id in _unique_in_order(retrieved_chunk_ids)[:k]:
        covered.update(chunk_source_blocks.get(chunk_id, ()))
    return len(covered & gold) / len(gold)


def ranking_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Collection[str],
    chunk_source_blocks: Mapping[str, Collection[SourceBlockKey]],
    gold_source_blocks: Collection[SourceBlockKey],
) -> dict[str, float]:
    """Compute metric values for the supported evaluation cutoffs."""
    metrics = {
        "mrr": mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
    }
    for k in (1, 3, 5, 10):
        metrics[f"hit@{k}"] = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
        metrics[f"evidence_recall@{k}"] = evidence_recall_at_k(
            retrieved_chunk_ids,
            chunk_source_blocks,
            gold_source_blocks,
            k,
        )
    return metrics


__all__ = [
    "SourceBlockKey",
    "evidence_recall_at_k",
    "hit_at_k",
    "mean_reciprocal_rank",
    "ranking_metrics",
]
