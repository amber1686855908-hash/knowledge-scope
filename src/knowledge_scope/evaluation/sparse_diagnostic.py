"""Small, descriptive Dense-vs-Sparse diagnostic helpers for A4.3."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DiagnosticQuery:
    """One safe diagnostic query and its existing A2.1 gold chunk IDs."""

    item_id: str
    query: str
    knowledge_base_id: UUID
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class SparseDiagnosticRecord:
    """Metadata-only comparison for one query; not a quality benchmark result."""

    item_id: str
    sparse_chunk_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    sparse_hit: bool
    dense_hit: bool
    overlap_count: int
    sparse_only_count: int
    dense_only_count: int


@dataclass(frozen=True, slots=True)
class SparseDiagnosticSummary:
    """Descriptive totals for a bounded diagnostic sample."""

    query_count: int
    sparse_hit_count: int
    dense_hit_count: int
    both_hit_count: int
    sparse_only_hit_count: int
    dense_only_hit_count: int
    mean_top_k_overlap: float


def compare_rankings(
    item_id: str,
    sparse_chunk_ids: Sequence[str],
    dense_chunk_ids: Sequence[str],
    relevant_chunk_ids: frozenset[str],
) -> SparseDiagnosticRecord:
    """Compare bounded rankings without combining incompatible score spaces."""

    sparse = tuple(sparse_chunk_ids)
    dense = tuple(dense_chunk_ids)
    sparse_set = set(sparse)
    dense_set = set(dense)
    return SparseDiagnosticRecord(
        item_id=item_id,
        sparse_chunk_ids=sparse,
        dense_chunk_ids=dense,
        sparse_hit=bool(sparse_set & relevant_chunk_ids),
        dense_hit=bool(dense_set & relevant_chunk_ids),
        overlap_count=len(sparse_set & dense_set),
        sparse_only_count=len(sparse_set - dense_set),
        dense_only_count=len(dense_set - sparse_set),
    )


def summarize(records: Sequence[SparseDiagnosticRecord], *, top_k: int) -> SparseDiagnosticSummary:
    """Summarize only hit/overlap observations from a bounded sample."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    query_count = len(records)
    sparse_hits = sum(record.sparse_hit for record in records)
    dense_hits = sum(record.dense_hit for record in records)
    both_hits = sum(record.sparse_hit and record.dense_hit for record in records)
    return SparseDiagnosticSummary(
        query_count=query_count,
        sparse_hit_count=sparse_hits,
        dense_hit_count=dense_hits,
        both_hit_count=both_hits,
        sparse_only_hit_count=sum(record.sparse_hit and not record.dense_hit for record in records),
        dense_only_hit_count=sum(record.dense_hit and not record.sparse_hit for record in records),
        mean_top_k_overlap=(
            sum(record.overlap_count for record in records) / query_count if query_count else 0.0
        ),
    )


def run_diagnostic(
    queries: Iterable[DiagnosticQuery],
    *,
    sparse_search: Callable[[str, UUID], Sequence[str]],
    dense_search: Callable[[str, UUID], Sequence[str]],
    top_k: int = 10,
) -> tuple[tuple[SparseDiagnosticRecord, ...], SparseDiagnosticSummary]:
    """Run a small caller-supplied diagnostic over existing retrieval branches."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    records = tuple(
        compare_rankings(
            query.item_id,
            sparse_search(query.query, query.knowledge_base_id)[:top_k],
            dense_search(query.query, query.knowledge_base_id)[:top_k],
            query.relevant_chunk_ids,
        )
        for query in queries
    )
    return records, summarize(records, top_k=top_k)


def write_records(path: Path, records: Iterable[SparseDiagnosticRecord]) -> None:
    """Write bounded metadata-only diagnostic records as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": record.item_id,
                    "sparse_chunk_ids": list(record.sparse_chunk_ids),
                    "dense_chunk_ids": list(record.dense_chunk_ids),
                    "sparse_hit": record.sparse_hit,
                    "dense_hit": record.dense_hit,
                    "overlap_count": record.overlap_count,
                    "sparse_only_count": record.sparse_only_count,
                    "dense_only_count": record.dense_only_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


__all__ = [
    "DiagnosticQuery",
    "SparseDiagnosticRecord",
    "SparseDiagnosticSummary",
    "compare_rankings",
    "run_diagnostic",
    "summarize",
    "write_records",
]
