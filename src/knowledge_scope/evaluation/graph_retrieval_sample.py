"""Small, read-only A3.4 graph-retrieval review sample."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from uuid import UUID

from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.graph.retrieval import GraphEvidence, GraphRetrievalResult
from knowledge_scope.graph.retrieval_service import GraphRetrievalError, GraphRetrievalService
from knowledge_scope.linking.models import normalize_linking_label
from knowledge_scope.linking.service_types import LocalEntityContext
from knowledge_scope.shared.config import Settings

from .entity_linking_sample import load_sample_entity_contexts

DEFAULT_INPUT: tuple[Path, ...] = (
    Path("data/evaluation/a3-2/debug-9/sample.jsonl"),
    Path("data/evaluation/a3-2/holdout-18/sample.jsonl"),
    Path("data/evaluation/a3-2/fresh-18/sample.jsonl"),
)
DEFAULT_OUTPUT = Path("data/evaluation/a3-4")
DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID = UUID("00000000-0000-4000-8000-000000000032")
SAMPLE_SCHEMA_VERSION = "1.0"


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _bounded_excerpt(value: str, limit: int = 280) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit] + ("…" if len(normalized) > limit else "")


def _no_seed_control_query(
    contexts: Sequence[LocalEntityContext],
    *,
    knowledge_base_id: UUID,
) -> str:
    """Create a deterministic sentinel absent from the temporary graph vocabulary."""

    vocabulary = {
        normalize_linking_label(value)
        for context in contexts
        if context.entity.knowledge_base_id == knowledge_base_id
        for value in (context.entity.canonical_name, *context.entity.aliases)
    }
    fingerprint_input = json.dumps(
        {
            "knowledge_base_id": str(knowledge_base_id),
            "vocabulary": sorted(vocabulary),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_input).hexdigest()
    for attempt in range(1_000):
        candidate = hashlib.sha256(f"{fingerprint}:{attempt}".encode("ascii")).hexdigest()
        if all(
            len(term) < 2 or (term not in candidate and candidate not in term)
            for term in vocabulary
        ):
            return candidate
    raise GraphRetrievalError("could not construct a vocabulary-free no-seed control query")


def build_review_queries(
    contexts: Sequence[LocalEntityContext],
    *,
    knowledge_base_id: UUID,
) -> tuple[dict[str, object], ...]:
    """Build deterministic entity-name, alias, and control queries from A3.2 records."""

    selected: dict[str, LocalEntityContext] = {}
    for context in sorted(
        contexts,
        key=lambda value: (
            value.subject or "",
            str(value.entity.document_id),
            value.entity.entity_id,
        ),
    ):
        if context.entity.knowledge_base_id != knowledge_base_id:
            continue
        selected.setdefault(context.subject or "未分类", context)

    queries: list[dict[str, object]] = []
    for subject, context in sorted(selected.items()):
        queries.append(
            {
                "query_id": f"subject-{len(queries):02d}",
                "kind": "direct_entity_fact",
                "subject": subject,
                "query": context.entity.canonical_name,
                "expected_entity_id": context.entity.entity_id,
                "source_excerpt": _bounded_excerpt(context.source_excerpt),
            }
        )
        if context.entity.aliases:
            queries.append(
                {
                    "query_id": f"alias-{len(queries):02d}",
                    "kind": "alias_entity_fact",
                    "subject": subject,
                    "query": context.entity.aliases[0],
                    "expected_entity_id": context.entity.entity_id,
                    "source_excerpt": _bounded_excerpt(context.source_excerpt),
                }
            )
    queries.extend(
        [
            {
                "query_id": "control-no-seed",
                "kind": "no_graph_seed",
                "subject": None,
                "query": _no_seed_control_query(
                    contexts,
                    knowledge_base_id=knowledge_base_id,
                ),
                "expected_entity_id": None,
                "source_excerpt": "",
            },
            {
                "query_id": "control-generic",
                "kind": "generic_short_name",
                "subject": None,
                "query": "的",
                "expected_entity_id": None,
                "source_excerpt": "",
            },
        ]
    )
    return tuple(queries)


def _result_record(
    query_record: dict[str, object],
    result: GraphRetrievalResult,
    *,
    elapsed_ms: float,
) -> dict[str, object]:
    valid_lineage_evidence_ids = [
        item.evidence.evidence_id
        for item in result.items
        if _has_valid_lineage(item.evidence, result.knowledge_base_id)
    ]
    invalid_lineage_evidence_ids = [
        item.evidence.evidence_id
        for item in result.items
        if item.evidence.evidence_id not in valid_lineage_evidence_ids
    ]
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "query_id": query_record["query_id"],
        "kind": query_record["kind"],
        "subject": query_record["subject"],
        "query": query_record["query"],
        "expected_entity_id": query_record["expected_entity_id"],
        "source_excerpt": query_record["source_excerpt"],
        "seeds": [seed.model_dump(mode="json") for seed in result.seeds],
        "items": [item.model_dump(mode="json") for item in result.items],
        "lineage_valid": (
            all(item.evidence.evidence_id in valid_lineage_evidence_ids for item in result.items)
            if result.items
            else None
        ),
        "valid_lineage_evidence_ids": valid_lineage_evidence_ids,
        "invalid_lineage_evidence_ids": invalid_lineage_evidence_ids,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _has_valid_lineage(evidence: GraphEvidence, knowledge_base_id: UUID) -> bool:
    """Validate lineage per returned result, including its deterministic evidence ID."""

    if evidence.knowledge_base_id != knowledge_base_id:
        return False
    try:
        GraphEvidence.model_validate(evidence.model_dump(mode="json"))
    except (TypeError, ValueError):
        return False
    return True


def run_graph_retrieval_review_sample(
    input_paths: Sequence[Path],
    *,
    settings: Settings,
    output_dir: Path = DEFAULT_OUTPUT,
    knowledge_base_id: UUID = DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID,
    store: Neo4jGraphStore | None = None,
) -> dict[str, object]:
    """Run bounded graph searches against an already-built local sample graph."""

    contexts = load_sample_entity_contexts(input_paths)
    query_records = build_review_queries(contexts, knowledge_base_id=knowledge_base_id)
    if not query_records:
        raise ValueError("A3.2 sample contains no usable graph-retrieval queries")
    graph_store = store or Neo4jGraphStore(settings)
    service = GraphRetrievalService(store=graph_store)
    records: list[dict[str, object]] = []
    started = perf_counter()
    try:
        for query_record in query_records:
            query = query_record["query"]
            if not isinstance(query, str):
                raise ValueError("graph-retrieval sample query must be text")
            query_started = perf_counter()
            result = service.search(query, knowledge_base_id)
            if query_record["kind"] == "no_graph_seed" and result.seeds:
                raise GraphRetrievalError("the no-seed control query unexpectedly resolved a seed")
            records.append(
                _result_record(
                    query_record,
                    result,
                    elapsed_ms=(perf_counter() - query_started) * 1000,
                )
            )
    finally:
        if store is None:
            graph_store.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "review.jsonl"
    review_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write_text(review_path, review_content)
    counts = Counter("resolved" if record["seeds"] else "no_seed" for record in records)
    hop_counts: Counter[str] = Counter()
    result_count = 0
    lineage_valid_count = 0
    lineage_invalid_count = 0
    for record in records:
        result_count += len(record["items"])
        valid_lineage_count = record["valid_lineage_evidence_ids"]
        invalid_lineage_count = record["invalid_lineage_evidence_ids"]
        if isinstance(valid_lineage_count, list):
            lineage_valid_count += len(valid_lineage_count)
        if isinstance(invalid_lineage_count, list):
            lineage_invalid_count += len(invalid_lineage_count)
        for item in record["items"]:
            paths = item.get("paths", [])
            if isinstance(paths, list) and paths:
                hop_counts[str(min(path["hop_distance"] for path in paths))] += 1

    summary: dict[str, object] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "knowledge_base_id": str(knowledge_base_id),
        "query_count": len(records),
        "resolved_query_count": counts["resolved"],
        "no_seed_count": counts["no_seed"],
        "total_graph_results": result_count,
        "hop_distribution": dict(sorted(hop_counts.items())),
        "valid_lineage_results": lineage_valid_count,
        "invalid_lineage_results": lineage_invalid_count,
        "run_elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "review_record_count": len(records),
        "review_sha256": hashlib.sha256(review_content.encode("utf-8")).hexdigest(),
        "input_files": [
            str(Path(path.parent.name) / path.name) for path in sorted(input_paths, key=str)
        ],
        "note": (
            "This is a bounded graph-retrieval review sample over an existing local graph; "
            "it does not measure recall, precision, or accuracy."
        ),
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return summary


__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID",
    "build_review_queries",
    "run_graph_retrieval_review_sample",
]
