"""Small, reproducible runtime sample for A3.2 graph extraction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID

from knowledge_scope.chunking.models import Chunk, ChunkingConfig
from knowledge_scope.chunking.service import chunk_document
from knowledge_scope.evaluation.retrieval_eval import CorpusDocument, load_canonical_corpus
from knowledge_scope.extraction.prompt import EXTRACTION_PROMPT_VERSION
from knowledge_scope.extraction.service import (
    ChunkExtractionResult,
    ExtractionAttempt,
    ExtractionError,
    ExtractionGateway,
    ExtractionPersistenceError,
    ExtractionService,
)
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.parsing.models import CanonicalDocument
from knowledge_scope.shared.config import Settings

DEFAULT_CANONICAL_ROOT = Path("data/benchmarks/a1-5/canonical")
DEFAULT_CORPUS_MANIFEST = Path("data/benchmarks/a1-5/corpus-manifest.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/a3-2")
DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID = UUID("00000000-0000-4000-8000-000000000032")
SAMPLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class SampleChunk:
    """A chunk selected by stable A1.5 manifest order for manual review."""

    subject: str
    benchmark_item_id: str
    config_fingerprint: str
    chunk: Chunk


def _sample_chunk_for_document(
    document: CanonicalDocument,
    config: ChunkingConfig,
) -> tuple[Chunk, str] | None:
    chunked = chunk_document(document, config)
    candidates = [chunk for chunk in chunked.chunks if chunk.text.strip()]
    if not candidates:
        return None
    substantive = [
        chunk
        for chunk in candidates
        if len(chunk.text.strip()) >= 80
        and any(
            content_type in {"text", "formula", "table"} for content_type in chunk.content_types
        )
    ]
    chosen = substantive[0] if substantive else candidates[0]
    return chosen, chunked.config_fingerprint


def select_sample_chunks(
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    *,
    sample_per_subject: int = 2,
    sample_offset: int = 0,
    config: ChunkingConfig | None = None,
) -> tuple[SampleChunk, ...]:
    """Select stable text chunks per subject without rerunning MinerU.

    ``sample_offset`` skips that many eligible chunks within every subject.
    It makes a deterministic holdout possible without changing the chunking
    implementation or overlapping the development sample.
    """

    if sample_per_subject < 1 or sample_per_subject > 5:
        raise ValueError("sample_per_subject must be between one and five")
    if sample_offset < 0 or sample_offset > 5:
        raise ValueError("sample_offset must be between zero and five")
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    chunking_config = config or ChunkingConfig()
    selected: list[SampleChunk] = []
    by_subject: dict[str, list[CorpusDocument]] = {}
    for corpus_document in corpus.values():
        by_subject.setdefault(corpus_document.subject, []).append(corpus_document)

    for subject in sorted(by_subject):
        documents: list[CorpusDocument] = sorted(
            by_subject[subject], key=lambda item: item.benchmark_item_id
        )
        candidates_for_subject: list[SampleChunk] = []
        for corpus_document in documents:
            selected_chunk = _sample_chunk_for_document(corpus_document.document, chunking_config)
            if selected_chunk is None:
                continue
            chunk, fingerprint = selected_chunk
            candidates_for_subject.append(
                SampleChunk(
                    subject=subject,
                    benchmark_item_id=corpus_document.benchmark_item_id,
                    config_fingerprint=fingerprint,
                    chunk=chunk,
                )
            )
            if len(candidates_for_subject) >= sample_offset + sample_per_subject:
                break
        chosen = candidates_for_subject[sample_offset : sample_offset + sample_per_subject]
        selected.extend(chosen)
        if not chosen:
            raise ValueError(f"subject {subject!r} has no chunk with extractable text")
    return tuple(selected)


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _bounded_excerpt(text: str, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit] + ("…" if len(normalized) > limit else "")


def _attempt_record(attempt: ExtractionAttempt) -> dict[str, object]:
    record = asdict(attempt)
    record["details"] = list(attempt.details)
    return record


def _grounding_record(result: ChunkExtractionResult) -> dict[str, object]:
    rejection_count = result.stats.grounding_rejections
    rejection_reasons = Counter(result.stats.grounding_rejection_reasons)
    if result.status == "accepted":
        decision = "accepted_with_rejections" if rejection_count else "accepted"
    elif result.status == "rejected":
        decision = "rejected"
    elif result.status == "empty":
        decision = "not_applicable"
    else:
        decision = "not_reached"
    return {
        "decision": decision,
        "rejection_count": rejection_count,
        "reasons": dict(sorted(rejection_reasons.items())),
    }


def _result_record(
    sample: SampleChunk,
    result: ChunkExtractionResult,
    *,
    settings: Settings,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "subject": sample.subject,
        "benchmark_item_id": sample.benchmark_item_id,
        "document_id": str(sample.chunk.document_id),
        "chunk_id": sample.chunk.chunk_id,
        "pages": [sample.chunk.page_start, sample.chunk.page_end],
        "source_block_ids": list(sample.chunk.source_block_ids),
        "section_path": list(sample.chunk.section_path),
        "content_types": list(sample.chunk.content_types),
        "config_fingerprint": sample.config_fingerprint,
        "text_excerpt": _bounded_excerpt(sample.chunk.text),
        "status": result.status,
        "error": result.error,
        "stats": asdict(result.stats),
        "attempts": [_attempt_record(attempt) for attempt in result.attempts],
        "grounding": _grounding_record(result),
        "entities": [entity.model_dump(mode="json") for entity in result.entities],
        "relations": [relation.model_dump(mode="json") for relation in result.relations],
        "grounded_relation_evidence": [
            asdict(evidence) for evidence in result.grounded_relation_evidence
        ],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost": result.estimated_cost(settings),
        "provider": result.llm_results[-1].provider if result.llm_results else None,
        "model": result.llm_results[-1].model if result.llm_results else None,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "latency_ms": round(result.latency_ms, 3),
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _failed_record(
    sample: SampleChunk,
    error: ExtractionError,
    *,
    settings: Settings,
    elapsed_ms: float,
    result: ChunkExtractionResult | None = None,
) -> dict[str, object]:
    attempts = result.attempts if result is not None else error.attempts
    provider_result = result.llm_results[-1] if result is not None and result.llm_results else None
    return {
        "subject": sample.subject,
        "benchmark_item_id": sample.benchmark_item_id,
        "document_id": str(sample.chunk.document_id),
        "chunk_id": sample.chunk.chunk_id,
        "pages": [sample.chunk.page_start, sample.chunk.page_end],
        "source_block_ids": list(sample.chunk.source_block_ids),
        "section_path": list(sample.chunk.section_path),
        "content_types": list(sample.chunk.content_types),
        "config_fingerprint": sample.config_fingerprint,
        "text_excerpt": _bounded_excerpt(sample.chunk.text),
        "status": "failed",
        "extraction_status": result.status if result is not None else None,
        "error_category": error.category,
        "error": str(error),
        "stats": asdict(result.stats) if result is not None else None,
        "attempts": [_attempt_record(attempt) for attempt in attempts],
        "grounding": (
            _grounding_record(result)
            if result is not None
            else {"decision": "not_reached", "rejection_count": 0}
        ),
        "entities": (
            [entity.model_dump(mode="json") for entity in result.entities]
            if result is not None
            else []
        ),
        "relations": (
            [relation.model_dump(mode="json") for relation in result.relations]
            if result is not None
            else []
        ),
        "grounded_relation_evidence": (
            [asdict(evidence) for evidence in result.grounded_relation_evidence]
            if result is not None
            else []
        ),
        "input_tokens": result.input_tokens if result is not None else None,
        "output_tokens": result.output_tokens if result is not None else None,
        "estimated_cost": result.estimated_cost(settings) if result is not None else None,
        "provider": provider_result.provider if provider_result is not None else None,
        "model": provider_result.model if provider_result is not None else None,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "latency_ms": result.latency_ms if result is not None else None,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one runtime artifact only after its complete content is written."""

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


def _sum_or_none(records: Sequence[dict[str, object]], key: str) -> int | None:
    values = [record[key] for record in records]
    if not values or any(value is None or not isinstance(value, int) for value in values):
        return None
    return sum(values)


def _cost_or_none(records: Sequence[dict[str, object]]) -> Decimal | None:
    costs = [record["estimated_cost"] for record in records]
    if not costs or any(value is None for value in costs):
        return None
    try:
        return sum((Decimal(str(value)) for value in costs), Decimal("0"))
    except (ArithmeticError, ValueError):
        return None


def _attempts_from_record(record: dict[str, object]) -> list[dict[str, object]]:
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [value for value in attempts if isinstance(value, dict)]


def _structured_attempt(category: object) -> bool:
    return category in {
        "valid_extraction",
        "empty_valid_extraction",
        "grounding_rejection",
    }


async def run_sample_evaluation(
    samples: Sequence[SampleChunk],
    *,
    gateway: ExtractionGateway,
    settings: Settings,
    output_dir: Path = DEFAULT_OUTPUT,
    knowledge_base_id: UUID = DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID,
    store: Neo4jGraphStore | None = None,
) -> dict[str, object]:
    """Run only the selected chunks and write bounded, ignored review artifacts."""

    if not samples:
        raise ValueError("at least one sample chunk is required")
    service = ExtractionService(gateway, settings)
    records: list[dict[str, object]] = []
    failure_categories: Counter[str] = Counter()
    started = perf_counter()
    for sample in samples:
        request_started = perf_counter()
        try:
            if store is None:
                result = await service.extract_chunk(
                    sample.chunk,
                    knowledge_base_id=knowledge_base_id,
                )
            else:
                result = await service.extract_and_persist(
                    sample.chunk,
                    knowledge_base_id=knowledge_base_id,
                    store=store,
                )
        except ExtractionError as error:
            failure_categories[error.category] += 1
            records.append(
                _failed_record(
                    sample,
                    error,
                    settings=settings,
                    elapsed_ms=(perf_counter() - request_started) * 1000,
                    result=(
                        error.result if isinstance(error, ExtractionPersistenceError) else None
                    ),
                )
            )
        else:
            records.append(
                _result_record(
                    sample,
                    result,
                    settings=settings,
                    elapsed_ms=(perf_counter() - request_started) * 1000,
                )
            )

    status_counts = Counter(str(record["status"]) for record in records)
    all_result_records = [
        record
        for record in records
        if record["status"] != "failed" or record.get("extraction_status") is not None
    ]
    observation_records = [
        record
        for record in all_result_records
        if record.get("extraction_status", record["status"]) in {"accepted", "empty", "rejected"}
    ]
    stats_records = [record["stats"] for record in all_result_records if record["stats"]]
    stats = [value for value in stats_records if isinstance(value, dict)]
    attempt_records = [attempt for record in records for attempt in _attempts_from_record(record)]
    first_attempts = [
        attempts[0] for record in records if (attempts := _attempts_from_record(record))
    ]
    retry_attempts = [
        attempt for record in records for attempt in _attempts_from_record(record)[1:]
    ]
    retry_recovered = sum(
        1
        for record in records
        if (attempts := _attempts_from_record(record))
        and len(attempts) > 1
        and _structured_attempt(attempts[-1].get("category"))
        and not _structured_attempt(attempts[0].get("category"))
    )
    summary: dict[str, object] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "knowledge_base_id": str(knowledge_base_id),
        "sample_size": len(samples),
        "attempted_chunks": len(samples),
        "status_counts": dict(sorted(status_counts.items())),
        "structured_parse_success": len(observation_records),
        "structured_parse_failure": status_counts.get("schema_rejected", 0),
        "parse_failure_count": sum(int(value.get("parse_failures", 0)) for value in stats),
        "provider_or_persistence_failure": status_counts.get("failed", 0),
        "provider_calls": len(attempt_records),
        "attempt_outcomes": dict(
            sorted(Counter(str(value.get("category")) for value in attempt_records).items())
        ),
        "first_attempt_outcomes": dict(
            sorted(Counter(str(value.get("category")) for value in first_attempts).items())
        ),
        "retry_attempt_outcomes": dict(
            sorted(Counter(str(value.get("category")) for value in retry_attempts).items())
        ),
        "first_attempt_structured_success": sum(
            _structured_attempt(value.get("category")) for value in first_attempts
        ),
        "retry_recovered_chunks": retry_recovered,
        "truncation_retry_count": sum(int(value.get("truncation_retries", 0)) for value in stats),
        "corrective_retry_count": sum(
            max(
                0,
                len(_attempts_from_record(record))
                - 1
                - int((record.get("stats") or {}).get("truncation_retries", 0)),
            )
            for record in records
        ),
        "finish_reasons": dict(
            sorted(
                Counter(
                    str(value.get("finish_reason"))
                    for value in attempt_records
                    if value.get("finish_reason") is not None
                ).items()
            )
        ),
        "schema_rejection_count": sum(int(value.get("schema_failures", 0)) for value in stats),
        "grounding_rejection_count": sum(
            int(value.get("grounding_rejections", 0)) for value in stats
        ),
        "grounding_rejection_reasons": dict(
            sorted(
                Counter(
                    str(reason)
                    for value in stats
                    for reason in value.get("grounding_rejection_reasons", ())
                ).items()
            )
        ),
        "empty_extraction_count": status_counts.get("empty", 0),
        "entities_produced": sum(len(record["entities"]) for record in all_result_records),
        "relations_produced": sum(len(record["relations"]) for record in all_result_records),
        "duplicate_entity_count": sum(int(value.get("duplicate_entities", 0)) for value in stats),
        "duplicate_relation_count": sum(
            int(value.get("duplicate_relations", 0)) for value in stats
        ),
        "input_tokens": _sum_or_none(all_result_records, "input_tokens"),
        "output_tokens": _sum_or_none(all_result_records, "output_tokens"),
        "estimated_cost": _cost_or_none(all_result_records),
        "latency_ms": round(sum(float(record["latency_ms"] or 0.0) for record in records), 3),
        "elapsed_ms": round(sum(float(record["elapsed_ms"] or 0.0) for record in records), 3),
        "failure_categories": dict(sorted(failure_categories.items())),
        "subject_counts": dict(sorted(Counter(sample.subject for sample in samples).items())),
        "note": (
            "This sample records schema/grounding/usage observations only; "
            "it does not measure accuracy."
        ),
        "run_elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n"
        for record in records
    )
    summary["record_count"] = len(records)
    summary["sample_sha256"] = hashlib.sha256(sample_content.encode("utf-8")).hexdigest()
    summary_content = (
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default)
        + "\n"
    )
    _atomic_write_text(output_dir / "sample.jsonl", sample_content)
    _atomic_write_text(output_dir / "summary.json", summary_content)
    return summary


__all__ = [
    "DEFAULT_CANONICAL_ROOT",
    "DEFAULT_CORPUS_MANIFEST",
    "DEFAULT_OUTPUT",
    "DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID",
    "SampleChunk",
    "run_sample_evaluation",
    "select_sample_chunks",
]
