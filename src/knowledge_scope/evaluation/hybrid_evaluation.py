"""A3.7 vector-only versus vector-plus-graph retrieval evaluation.

The evaluator is deliberately read-only with respect to the retrieval stores and
the frozen A2.1 annotations.  It runs the existing A2.3/A2.4 vector path once
for the baseline and once as the vector branch of the existing A3.5 hybrid
service, then fuses that branch with the existing A3.4 graph retriever.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from knowledge_scope.evaluation.embedding_benchmark import (
    FrozenEvalCase,
    load_frozen_chunk_index,
    load_frozen_eval_cases,
)
from knowledge_scope.evaluation.graph_corpus_build import read_graph_coverage
from knowledge_scope.evaluation.retrieval_eval import IndexedChunk
from knowledge_scope.evaluation.retrieval_metrics import SourceBlockKey, ranking_metrics
from knowledge_scope.evaluation.retrieval_system_benchmark import BGE_RERANKER_MODEL_REVISION
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.graph.retrieval import (
    GraphEvidenceResult,
    GraphRetrievalConfig,
    GraphRetrievalResult,
)
from knowledge_scope.graph.retrieval_service import GraphRetrievalService
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel, embedding_config_fingerprint
from knowledge_scope.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetrievalService,
)
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    QdrantVectorStore,
    point_id_for_chunk,
)
from knowledge_scope.retrieval.reranking import RerankingService, create_local_reranker
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.shared.config import Settings, get_settings

A37_SCHEMA_VERSION = "1.0"
A25_PROFILE_VERSION = "a2.5-frozen-v1"
A25_EMBEDDING_MODEL = QWEN_EMBEDDING_MODEL_ID
A25_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
A25_DEVICE = "cuda"
A25_DTYPE = "float16"
A25_BATCH_SIZE = 8
A25_MAX_SEQ_LENGTH = 512
A25_DENSE_TOP_K = 10
A25_RERANK_TOP_K = 10
A25_QDRANT_COLLECTION = "knowledgescope_chunks_v1"
A25_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
A25_RERANKER_REVISION = BGE_RERANKER_MODEL_REVISION
A35_GRAPH_RESULT_LIMIT = 20
A35_HYBRID_RESULT_LIMIT = 20
A35_RRF_K = 60
GRAPH_PROMPT_LAYOUT_VERSION = "legacy-v1"
GRAPH_TRUNCATION_BUDGETS = (1024, 2048, 4096)
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_MATERIALIZED = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
DEFAULT_KB_MAPPING = Path("data/evaluation/a3-5/a2-1-kb-mapping.jsonl")
DEFAULT_GRAPH_RUN_MANIFEST = Path("data/evaluation/a3-6/full-run-20260910/run-manifest.json")
DEFAULT_GRAPH_EXCLUSIONS = Path(
    "data/evaluation/a3-6/full-run-20260910/partial-document-exclusions.json"
)
DEFAULT_OUTPUT = Path("data/evaluation/a3-7")
DEFAULT_A25_RESULTS = Path("data/evaluation/a2-5/results.jsonl")
DEFAULT_A25_MANIFEST = Path("data/evaluation/a2-5/manifest.json")
DEFAULT_GRAPH_RUN_ID = "e926e1d3-9050-4911-89ef-1632ed0894c2"
DEFAULT_GRAPH_PIPELINE_FINGERPRINT = (
    "a9407046dad32e8da8f2ef4256497b60e8ffdb21771c4075a7c87a013ff70cc7"
)
EXPECTED_DOCUMENT_COUNT = 255
EXPECTED_CHUNK_COUNT = 7_524
EXPECTED_GRAPH_ELIGIBLE_DOCUMENT_COUNT = 246
EXPECTED_GRAPH_EVIDENCE_COUNT = 6_087
EXPECTED_GRAPH_ENTITY_COUNT = 55_098
EXPECTED_GRAPH_RELATION_COUNT = 27_268
EXPECTED_CANONICAL_ENTITY_COUNT = 1_241
EXPECTED_MEMBERSHIP_COUNT = 2_609
EXPECTED_GRAPH_COVERED_ITEMS = 105
EXPECTED_GRAPH_PARTIAL_ITEMS = 0
EXPECTED_GRAPH_UNCOVERED_ITEMS = 3
EXPECTED_TERMINAL_FAILED_CHUNK_COUNT = 11
DEFAULT_METRIC_KS = (1, 3, 5, 10)

SplitName = Literal["dev", "test", "all"]
CoverageClass = Literal["complete", "partial", "none"]
BranchStatus = Literal["success", "empty", "failed", "timed_out"]


class HybridEvaluationError(RuntimeError):
    """Raised when frozen inputs or the current graph/vector state is unsafe."""


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HybridEvaluationProtocol(_EvaluationModel):
    """Frozen A3.7 profile matching the accepted A2.5 Vector configuration."""

    a25_profile_version: Literal["a2.5-frozen-v1"] = A25_PROFILE_VERSION
    embedding_model: Literal["Qwen/Qwen3-Embedding-0.6B"] = A25_EMBEDDING_MODEL
    embedding_model_revision: str
    embedding_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_query_convention: Literal["SentenceTransformers prompt_name=query"] = (
        "SentenceTransformers prompt_name=query"
    )
    embedding_document_convention: Literal["no query prompt"] = "no query prompt"
    embedding_device: str
    embedding_dtype: str
    embedding_batch_size: StrictInt = Field(default=A25_BATCH_SIZE, ge=1)
    embedding_max_seq_length: StrictInt = Field(default=A25_MAX_SEQ_LENGTH, ge=1)
    reranker_model: Literal["BAAI/bge-reranker-v2-m3"] = A25_RERANKER_MODEL
    reranker_model_revision: str
    reranker_prompt_convention: Literal["raw query/passage pair"] = "raw query/passage pair"
    reranker_device: str
    reranker_dtype: str
    reranker_batch_size: StrictInt = Field(default=A25_BATCH_SIZE, ge=1)
    reranker_max_seq_length: StrictInt = Field(default=A25_MAX_SEQ_LENGTH, ge=1)
    qdrant_collection: Literal["knowledgescope_chunks_v1"] = A25_QDRANT_COLLECTION
    qdrant_schema_version: Literal["1.0"] = QDRANT_COLLECTION_SCHEMA_VERSION
    qdrant_vector_dimension: StrictInt = QDRANT_VECTOR_DIMENSION
    qdrant_distance: Literal["cosine"] = "cosine"
    qdrant_search_scope: Literal["current collection; no ANN performance claim"] = (
        "current collection; no ANN performance claim"
    )
    vector_candidate_limit: StrictInt = Field(default=A25_DENSE_TOP_K, ge=1, le=100)
    vector_rerank_limit: StrictInt = Field(default=A25_RERANK_TOP_K, ge=1, le=100)
    graph_result_limit: StrictInt = Field(default=20, ge=1, le=500)
    hybrid_result_limit: StrictInt = Field(default=20, ge=1, le=500)
    rrf_k: StrictInt = Field(default=60, ge=1, le=10_000)
    metric_ks: tuple[StrictInt, ...] = DEFAULT_METRIC_KS
    failure_mode: Literal["strict"] = "strict"
    warmup: bool = True

    @model_validator(mode="after")
    def validate_frozen_a25_profile(self) -> HybridEvaluationProtocol:
        expected = {
            "embedding_model": A25_EMBEDDING_MODEL,
            "embedding_model_revision": A25_EMBEDDING_REVISION,
            "embedding_device": A25_DEVICE,
            "embedding_dtype": A25_DTYPE,
            "embedding_batch_size": A25_BATCH_SIZE,
            "embedding_max_seq_length": A25_MAX_SEQ_LENGTH,
            "reranker_model": A25_RERANKER_MODEL,
            "reranker_model_revision": A25_RERANKER_REVISION,
            "reranker_device": A25_DEVICE,
            "reranker_dtype": A25_DTYPE,
            "reranker_batch_size": A25_BATCH_SIZE,
            "reranker_max_seq_length": A25_MAX_SEQ_LENGTH,
            "qdrant_collection": A25_QDRANT_COLLECTION,
            "vector_candidate_limit": A25_DENSE_TOP_K,
            "vector_rerank_limit": A25_RERANK_TOP_K,
            "graph_result_limit": A35_GRAPH_RESULT_LIMIT,
            "hybrid_result_limit": A35_HYBRID_RESULT_LIMIT,
            "rrf_k": A35_RRF_K,
        }
        mismatches = [
            f"{field}={getattr(self, field)!r} (expected {value!r})"
            for field, value in expected.items()
            if getattr(self, field) != value
        ]
        if mismatches:
            raise ValueError("A3.7 requires the frozen A2.5 profile: " + "; ".join(mismatches))
        return self


class HybridEvaluationPreflight(_EvaluationModel):
    """Read-only identity and coverage facts checked before scoring."""

    knowledge_base_id: UUID
    frozen_item_count: StrictInt
    dev_count: StrictInt
    test_count: StrictInt
    qdrant_point_count: StrictInt
    qdrant_collection: str
    qdrant_vector_dimension: StrictInt
    qdrant_distance: Literal["cosine"]
    qdrant_embedding_model: str
    qdrant_embedding_model_revision: str
    qdrant_embedding_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_document_count: StrictInt
    graph_evidence_count: StrictInt
    graph_supported_chunk_count: StrictInt
    graph_entity_count: StrictInt
    graph_relation_count: StrictInt
    canonical_entity_count: StrictInt
    membership_count: StrictInt
    graph_coverage_complete_count: StrictInt
    graph_coverage_partial_count: StrictInt
    graph_coverage_none_count: StrictInt
    graph_run_id: str
    graph_pipeline_fingerprint: str


class HybridQueryRecord(_EvaluationModel):
    """Bounded machine-readable evidence and ranking record for one query."""

    item_id: str
    query: str
    split: Literal["dev", "test"]
    subject: str
    query_type: str
    coverage: CoverageClass
    gold_relevant_chunk_ids: list[str]
    gold_source_blocks: list[str]
    vector_status: BranchStatus
    graph_status: BranchStatus
    vector_error: str | None = None
    graph_error: str | None = None
    vector_ranked_chunk_ids: list[str]
    graph_ranked_chunk_ids: list[str]
    graph_evidence_ids: list[str]
    hybrid_ranked_chunk_ids: list[str]
    hybrid_items: list[dict[str, Any]]
    vector_metrics: dict[str, float]
    graph_metrics: dict[str, float]
    hybrid_metrics: dict[str, float]
    vector_gold_chunk_ids_at_10: list[str]
    graph_gold_chunk_ids_at_10: list[str]
    both_gold_chunk_ids_at_10: list[str]
    vector_gold_source_blocks_at_10: list[str]
    graph_gold_source_blocks_at_10: list[str]
    both_gold_source_blocks_at_10: list[str]
    vector_missed_graph_recovered_source_blocks_at_10: list[str]
    graph_only_final_recovered_gold_chunk_ids_at_10: list[str]
    graph_only_final_recovered_source_blocks_at_10: list[str]
    vector_first_relevant_rank: StrictInt | None = Field(default=None, ge=1)
    hybrid_first_relevant_rank: StrictInt | None = Field(default=None, ge=1)
    first_relevant_rank_delta: StrictInt | None = None
    mrr_delta: float
    vector_baseline_latency_ms: float
    vector_embedding_latency_ms: float
    vector_qdrant_latency_ms: float
    vector_reranker_latency_ms: float
    hybrid_vector_latency_ms: float
    hybrid_graph_latency_ms: float
    hybrid_fusion_latency_ms: float
    hybrid_total_latency_ms: float


class HybridCutoffContribution(_EvaluationModel):
    """Query-level hit/loss counts for one ranking cutoff."""

    k: StrictInt = Field(ge=1)
    vector_found_gold_query_count: StrictInt
    hybrid_found_gold_query_count: StrictInt
    improved_query_count: StrictInt
    unchanged_query_count: StrictInt
    regressed_query_count: StrictInt
    vector_found_but_hybrid_lost_query_count: StrictInt


class HybridSplitSummary(_EvaluationModel):
    """Aggregate metrics for a split or a graph-coverage/type diagnostic slice."""

    name: str
    query_count: StrictInt
    vector_metrics: dict[str, float]
    hybrid_metrics: dict[str, float]
    absolute_delta: dict[str, float]
    relative_delta: dict[str, float | None]
    vector_latency_ms: dict[str, float | None]
    hybrid_latency_ms: dict[str, float | None]


class HybridBranchObservation(_EvaluationModel):
    """Bounded branch/result counts for one evaluation slice."""

    query_count: StrictInt
    vector_result_count: StrictInt
    graph_result_count: StrictInt
    hybrid_result_count: StrictInt
    graph_nonempty_query_count: StrictInt
    graph_empty_query_count: StrictInt
    both_branch_overlap_chunk_count: StrictInt
    both_branch_overlap_query_count: StrictInt
    source_distribution: dict[str, StrictInt]


class HybridContributionSummary(_EvaluationModel):
    """Graph contribution counts; unsuffixed query counts use the @10 cutoff."""

    cutoff_k: StrictInt = 10
    query_count: StrictInt
    vector_found_gold_query_count: StrictInt
    graph_found_gold_query_count: StrictInt
    both_found_gold_query_count: StrictInt
    graph_only_recovered_gold_chunk_count: StrictInt
    graph_only_recovered_gold_source_block_count: StrictInt
    graph_only_final_recovered_gold_chunk_count: StrictInt
    graph_only_final_recovered_source_block_count: StrictInt
    improved_query_count: StrictInt
    unchanged_query_count: StrictInt
    regressed_query_count: StrictInt
    graph_no_usable_result_query_count: StrictInt
    vector_found_but_hybrid_lost_query_count: StrictInt
    by_k: dict[str, HybridCutoffContribution]


@dataclass(slots=True)
class _StageTiming:
    embedding_ms: float = 0.0
    qdrant_ms: float = 0.0
    reranker_ms: float = 0.0


class _TimedEncoder:
    """Timing proxy that leaves the existing DenseRetrievalService unchanged."""

    def __init__(self, encoder: QwenEmbeddingModel, timing: _StageTiming) -> None:
        self.encoder = encoder
        self.timing = timing

    @property
    def model_id(self) -> str:
        return self.encoder.model_id

    def encode_query(self, query: str) -> list[float]:
        started = time.perf_counter()
        result = self.encoder.encode_query(query)
        self.timing.embedding_ms = (time.perf_counter() - started) * 1_000
        return result


class _TimedVectorStore:
    """Timing proxy for the existing Qdrant store, without a second retrieval path."""

    def __init__(self, store: QdrantVectorStore, timing: _StageTiming) -> None:
        self.store = store
        self.timing = timing

    @property
    def collection_name(self) -> str:
        return self.store.collection_name

    def search(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self.store.search(*args, **kwargs)
        self.timing.qdrant_ms = (time.perf_counter() - started) * 1_000
        return result


class _TimedReranking:
    """Timing proxy for the existing A2.4 reranking service."""

    def __init__(self, service: RerankingService, timing: _StageTiming) -> None:
        self.service = service
        self.timing = timing

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self.service.rerank(*args, **kwargs)
        self.timing.reranker_ms = (time.perf_counter() - started) * 1_000
        return result


@dataclass(slots=True)
class _Runtime:
    qdrant: QdrantVectorStore
    neo4j: Neo4jGraphStore
    baseline: HybridRetrievalService
    hybrid: HybridRetrievalService
    vector_timing: _StageTiming
    hybrid_vector_timing: _StageTiming


@dataclass(frozen=True, slots=True)
class _VectorMeasurement:
    ranked_chunk_ids: tuple[str, ...]
    total_latency_ms: float
    embedding_latency_ms: float
    qdrant_latency_ms: float
    reranker_latency_ms: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HybridEvaluationError(f"evaluation input is not readable: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise HybridEvaluationError(f"invalid JSON in {path} line {line_number}") from error
        if not isinstance(value, dict):
            raise HybridEvaluationError(f"expected JSON object in {path} line {line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise HybridEvaluationError(f"evaluation input is not readable: {path}") from error
    return digest.hexdigest()


def _gold_blocks(case: FrozenEvalCase) -> set[SourceBlockKey]:
    return set(case.gold_source_blocks)


def _chunk_blocks(
    chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, IndexedChunk],
) -> set[SourceBlockKey]:
    blocks: set[SourceBlockKey] = set()
    for chunk_id in dict.fromkeys(chunk_ids):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None:
            blocks.update((str(chunk.document_id), block_id) for block_id in chunk.source_block_ids)
    return blocks


def _unique_graph_items(result: GraphRetrievalResult) -> list[GraphEvidenceResult]:
    """Return the first graph item per chunk, matching A3.5 graph rank semantics."""
    seen_chunks: set[str] = set()
    unique_items: list[GraphEvidenceResult] = []
    for item in result.items:
        chunk_id = item.evidence.chunk_id
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        unique_items.append(item)
    return unique_items


def _graph_blocks(result: GraphRetrievalResult, *, limit: int) -> set[SourceBlockKey]:
    blocks: set[SourceBlockKey] = set()
    for item in _unique_graph_items(result)[:limit]:
        blocks.update(
            (str(item.evidence.document_id), block_id)
            for block_id in item.evidence.source_block_ids
        )
    return blocks


def _unique_chunk_ids(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _graph_chunk_ids(result: GraphRetrievalResult) -> list[str]:
    return [item.evidence.chunk_id for item in _unique_graph_items(result)]


def _validate_graph_result(
    result: GraphRetrievalResult,
    *,
    knowledge_base_id: UUID,
    chunks_by_id: Mapping[str, IndexedChunk],
) -> None:
    """Fail closed if Neo4j returns evidence outside the frozen chunk lineage."""
    if result.knowledge_base_id != knowledge_base_id:
        raise HybridEvaluationError("graph result has invalid request KB lineage")
    for item in result.items:
        evidence = item.evidence
        if evidence.knowledge_base_id != knowledge_base_id:
            raise HybridEvaluationError("graph evidence has invalid KB lineage")
        chunk = chunks_by_id.get(evidence.chunk_id)
        if chunk is None:
            raise HybridEvaluationError(
                f"graph evidence references a chunk outside the frozen index: {evidence.chunk_id}"
            )
        if chunk.document_id != evidence.document_id:
            raise HybridEvaluationError("graph evidence has invalid document lineage")
        if evidence.page_start < chunk.page_start or evidence.page_end > chunk.page_end:
            raise HybridEvaluationError("graph evidence has invalid page lineage")
        if not set(evidence.source_block_ids).issubset(chunk.source_block_ids):
            raise HybridEvaluationError("graph evidence has invalid source-block lineage")
        if evidence.section_path and evidence.section_path != chunk.section_path:
            raise HybridEvaluationError("graph evidence has invalid section lineage")


def _coverage_for_case(
    case: FrozenEvalCase,
    graph_supported_chunk_ids: Collection[str],
) -> CoverageClass:
    gold = set(case.relevant_chunk_ids)
    supported = gold & set(graph_supported_chunk_ids)
    if supported == gold:
        return "complete"
    if supported:
        return "partial"
    return "none"


def _delta(vector: Mapping[str, float], hybrid: Mapping[str, float]) -> dict[str, float]:
    return {
        key: round(hybrid.get(key, 0.0) - vector.get(key, 0.0), 8)
        for key in sorted(set(vector) | set(hybrid))
    }


def _relative_delta(
    vector: Mapping[str, float], hybrid: Mapping[str, float]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in sorted(set(vector) | set(hybrid)):
        before = vector.get(key, 0.0)
        after = hybrid.get(key, 0.0)
        result[key] = None if before == 0 else round((after - before) / before, 8)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return round(ordered[index], 3)


def _first_relevant_rank(
    ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Collection[str]
) -> int | None:
    """Return the one-based rank of the first relevant chunk, if present."""
    relevant = set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant:
            return rank
    return None


def summarize_rank_movement(records: Sequence[HybridQueryRecord]) -> dict[str, Any]:
    """Summarize first-relevant rank movement without hiding misses."""
    both_ranked = [
        record.first_relevant_rank_delta
        for record in records
        if record.first_relevant_rank_delta is not None
    ]
    return {
        "query_count": len(records),
        "both_ranked_query_count": len(both_ranked),
        "vector_miss_hybrid_hit_query_count": sum(
            record.vector_first_relevant_rank is None
            and record.hybrid_first_relevant_rank is not None
            for record in records
        ),
        "both_missed_query_count": sum(
            record.vector_first_relevant_rank is None and record.hybrid_first_relevant_rank is None
            for record in records
        ),
        "vector_hit_hybrid_miss_query_count": sum(
            record.vector_first_relevant_rank is not None
            and record.hybrid_first_relevant_rank is None
            for record in records
        ),
        "earlier_query_count": sum(value < 0 for value in both_ranked),
        "unchanged_query_count": sum(value == 0 for value in both_ranked),
        "later_query_count": sum(value > 0 for value in both_ranked),
        "delta_mean": round(statistics.fmean(both_ranked), 8) if both_ranked else None,
        "delta_p50": _percentile(both_ranked, 0.50),
        "delta_p95": _percentile(both_ranked, 0.95),
    }


def summarize_mrr_deltas(records: Sequence[HybridQueryRecord]) -> dict[str, Any]:
    """Return a descriptive per-query MRR delta distribution."""
    deltas = [record.mrr_delta for record in records]
    return {
        "query_count": len(deltas),
        "positive_query_count": sum(value > 0 for value in deltas),
        "zero_query_count": sum(value == 0 for value in deltas),
        "negative_query_count": sum(value < 0 for value in deltas),
        "delta_mean": round(statistics.fmean(deltas), 8) if deltas else None,
        "delta_min": round(min(deltas), 8) if deltas else None,
        "delta_max": round(max(deltas), 8) if deltas else None,
        "delta_p50": _percentile(deltas, 0.50),
        "delta_p95": _percentile(deltas, 0.95),
    }


def _summary(name: str, records: Sequence[HybridQueryRecord]) -> HybridSplitSummary:
    if not records:
        raise HybridEvaluationError(f"cannot summarize empty slice: {name}")

    def metric_rows(field_name: str) -> list[Mapping[str, float]]:
        return [getattr(record, field_name) for record in records]

    def average(field_name: str) -> dict[str, float]:
        rows = metric_rows(field_name)
        names = sorted({key for row in rows for key in row})
        return {
            key: round(statistics.fmean(row[key] for row in rows), 8)
            for key in names
            if all(key in row for row in rows)
        }

    vector_metrics = average("vector_metrics")
    hybrid_metrics = average("hybrid_metrics")
    vector_latency = {
        "embedding_mean_ms": round(
            statistics.fmean(record.vector_embedding_latency_ms for record in records), 3
        ),
        "qdrant_mean_ms": round(
            statistics.fmean(record.vector_qdrant_latency_ms for record in records), 3
        ),
        "reranker_mean_ms": round(
            statistics.fmean(record.vector_reranker_latency_ms for record in records), 3
        ),
        "total_mean_ms": round(
            statistics.fmean(record.vector_baseline_latency_ms for record in records), 3
        ),
        "total_p50_ms": _percentile(
            [record.vector_baseline_latency_ms for record in records], 0.50
        ),
        "total_p95_ms": _percentile(
            [record.vector_baseline_latency_ms for record in records], 0.95
        ),
    }
    hybrid_latency = {
        "vector_mean_ms": round(
            statistics.fmean(record.hybrid_vector_latency_ms for record in records), 3
        ),
        "graph_mean_ms": round(
            statistics.fmean(record.hybrid_graph_latency_ms for record in records), 3
        ),
        "fusion_mean_ms": round(
            statistics.fmean(record.hybrid_fusion_latency_ms for record in records), 3
        ),
        "total_mean_ms": round(
            statistics.fmean(record.hybrid_total_latency_ms for record in records), 3
        ),
        "total_p50_ms": _percentile([record.hybrid_total_latency_ms for record in records], 0.50),
        "total_p95_ms": _percentile([record.hybrid_total_latency_ms for record in records], 0.95),
    }
    return HybridSplitSummary(
        name=name,
        query_count=len(records),
        vector_metrics=vector_metrics,
        hybrid_metrics=hybrid_metrics,
        absolute_delta=_delta(vector_metrics, hybrid_metrics),
        relative_delta=_relative_delta(vector_metrics, hybrid_metrics),
        vector_latency_ms=vector_latency,
        hybrid_latency_ms=hybrid_latency,
    )


def summarize_contribution(records: Sequence[HybridQueryRecord]) -> HybridContributionSummary:
    """Summarize graph contribution using the same top-10 cutoff as A2.1 metrics."""
    improved = 0
    unchanged = 0
    regressed = 0
    vector_lost = 0
    vector_found = 0
    graph_found = 0
    both_found = 0
    graph_no_result = 0
    graph_recovered_chunks = 0
    graph_recovered_blocks = 0
    graph_final_recovered_chunks = 0
    graph_final_recovered_blocks = 0
    for record in records:
        vector_hit = bool(record.vector_gold_chunk_ids_at_10)
        graph_hit = bool(record.graph_gold_chunk_ids_at_10)
        hybrid_hit = bool(
            set(record.gold_relevant_chunk_ids) & set(record.hybrid_ranked_chunk_ids[:10])
        )
        vector_found += vector_hit
        graph_found += graph_hit
        both_found += vector_hit and graph_hit
        if not record.graph_ranked_chunk_ids:
            graph_no_result += 1
        graph_recovered_chunks += len(
            set(record.graph_gold_chunk_ids_at_10) - set(record.vector_gold_chunk_ids_at_10)
        )
        graph_recovered_blocks += len(record.vector_missed_graph_recovered_source_blocks_at_10)
        graph_final_recovered_chunks += len(record.graph_only_final_recovered_gold_chunk_ids_at_10)
        graph_final_recovered_blocks += len(record.graph_only_final_recovered_source_blocks_at_10)
        if hybrid_hit and not vector_hit:
            improved += 1
        elif vector_hit and not hybrid_hit:
            regressed += 1
            vector_lost += 1
        else:
            unchanged += 1
    by_k: dict[str, HybridCutoffContribution] = {}
    for k in DEFAULT_METRIC_KS:
        vector_hits = [record.vector_metrics.get(f"hit@{k}", 0.0) > 0 for record in records]
        hybrid_hits = [record.hybrid_metrics.get(f"hit@{k}", 0.0) > 0 for record in records]
        improved_at_k = sum(
            hybrid_hit and not vector_hit
            for vector_hit, hybrid_hit in zip(vector_hits, hybrid_hits, strict=True)
        )
        regressed_at_k = sum(
            vector_hit and not hybrid_hit
            for vector_hit, hybrid_hit in zip(vector_hits, hybrid_hits, strict=True)
        )
        by_k[str(k)] = HybridCutoffContribution(
            k=k,
            vector_found_gold_query_count=sum(vector_hits),
            hybrid_found_gold_query_count=sum(hybrid_hits),
            improved_query_count=improved_at_k,
            unchanged_query_count=len(records) - improved_at_k - regressed_at_k,
            regressed_query_count=regressed_at_k,
            vector_found_but_hybrid_lost_query_count=regressed_at_k,
        )
    return HybridContributionSummary(
        cutoff_k=10,
        query_count=len(records),
        vector_found_gold_query_count=vector_found,
        graph_found_gold_query_count=graph_found,
        both_found_gold_query_count=both_found,
        graph_only_recovered_gold_chunk_count=graph_recovered_chunks,
        graph_only_recovered_gold_source_block_count=graph_recovered_blocks,
        graph_only_final_recovered_gold_chunk_count=graph_final_recovered_chunks,
        graph_only_final_recovered_source_block_count=graph_final_recovered_blocks,
        improved_query_count=improved,
        unchanged_query_count=unchanged,
        regressed_query_count=regressed,
        graph_no_usable_result_query_count=graph_no_result,
        vector_found_but_hybrid_lost_query_count=vector_lost,
        by_k=by_k,
    )


def summarize_branch_observations(
    records: Sequence[HybridQueryRecord],
) -> HybridBranchObservation:
    """Summarize bounded branch sizes and fused provenance for a slice."""
    source_distribution = Counter(
        item.get("source")
        for record in records
        for item in record.hybrid_items
        if item.get("source") in {"vector", "graph", "both"}
    )
    overlap_counts = [
        len(set(record.vector_ranked_chunk_ids) & set(record.graph_ranked_chunk_ids))
        for record in records
    ]
    overlap_query_count = sum(value > 0 for value in overlap_counts)
    return HybridBranchObservation(
        query_count=len(records),
        vector_result_count=sum(len(record.vector_ranked_chunk_ids) for record in records),
        graph_result_count=sum(len(record.graph_ranked_chunk_ids) for record in records),
        hybrid_result_count=sum(len(record.hybrid_ranked_chunk_ids) for record in records),
        graph_nonempty_query_count=sum(bool(record.graph_ranked_chunk_ids) for record in records),
        graph_empty_query_count=sum(not record.graph_ranked_chunk_ids for record in records),
        both_branch_overlap_chunk_count=sum(overlap_counts),
        both_branch_overlap_query_count=overlap_query_count,
        source_distribution={
            source: source_distribution.get(source, 0) for source in ("vector", "graph", "both")
        },
    )


def evaluate_query_record(
    case: FrozenEvalCase,
    *,
    split: Literal["dev", "test"] = "dev",
    coverage: CoverageClass,
    chunks_by_id: Mapping[str, IndexedChunk],
    vector_ranked_chunk_ids: Sequence[str],
    graph_result: GraphRetrievalResult,
    hybrid_items: Sequence[Any],
    vector_status: BranchStatus = "success",
    graph_status: BranchStatus = "success",
    vector_error: str | None = None,
    graph_error: str | None = None,
    vector_baseline_latency_ms: float = 0.0,
    vector_embedding_latency_ms: float = 0.0,
    vector_qdrant_latency_ms: float = 0.0,
    vector_reranker_latency_ms: float = 0.0,
    hybrid_vector_latency_ms: float = 0.0,
    hybrid_graph_latency_ms: float = 0.0,
    hybrid_fusion_latency_ms: float = 0.0,
    hybrid_total_latency_ms: float = 0.0,
) -> HybridQueryRecord:
    """Build one record from already-produced branch outputs without any I/O."""
    vector_ids = _unique_chunk_ids(vector_ranked_chunk_ids)
    graph_ids = _graph_chunk_ids(graph_result)
    hybrid_ids = _unique_chunk_ids([item.chunk_id for item in hybrid_items])
    gold_chunks = set(case.relevant_chunk_ids)
    gold_blocks = _gold_blocks(case)
    vector_top10 = set(vector_ids[:10]) & gold_chunks
    graph_top10 = set(graph_ids[:10]) & gold_chunks
    both_chunks = vector_top10 & graph_top10
    vector_blocks = _chunk_blocks(vector_ids[:10], chunks_by_id) & gold_blocks
    graph_blocks = _graph_blocks(graph_result, limit=10) & gold_blocks
    all_graph_blocks = _graph_blocks(graph_result, limit=len(graph_result.items)) & gold_blocks
    both_blocks = vector_blocks & graph_blocks
    recovered_blocks = graph_blocks - vector_blocks
    hybrid_top10 = set(hybrid_ids[:10]) & gold_chunks
    final_recovered_chunks = (hybrid_top10 - vector_top10) & set(graph_ids)
    hybrid_blocks = _chunk_blocks(hybrid_ids[:10], chunks_by_id) & gold_blocks
    final_recovered_blocks = (hybrid_blocks - vector_blocks) & all_graph_blocks
    vector_first_rank = _first_relevant_rank(vector_ids, gold_chunks)
    hybrid_first_rank = _first_relevant_rank(hybrid_ids, gold_chunks)
    rank_delta = (
        hybrid_first_rank - vector_first_rank
        if vector_first_rank is not None and hybrid_first_rank is not None
        else None
    )
    chunk_source_blocks = {
        chunk_id: {(str(chunk.document_id), block_id) for block_id in chunk.source_block_ids}
        for chunk_id, chunk in chunks_by_id.items()
    }
    vector_metrics = ranking_metrics(vector_ids, gold_chunks, chunk_source_blocks, gold_blocks)
    graph_metrics = ranking_metrics(graph_ids, gold_chunks, chunk_source_blocks, gold_blocks)
    hybrid_metrics = ranking_metrics(hybrid_ids, gold_chunks, chunk_source_blocks, gold_blocks)
    return HybridQueryRecord(
        item_id=case.item.item_id,
        query=case.item.query,
        split=split,
        subject=case.item.subject,
        query_type=case.item.query_type,
        coverage=coverage,
        gold_relevant_chunk_ids=sorted(gold_chunks),
        gold_source_blocks=sorted(
            f"{document_id}:{block_id}" for document_id, block_id in gold_blocks
        ),
        vector_status=vector_status,
        graph_status=graph_status,
        vector_error=vector_error,
        graph_error=graph_error,
        vector_ranked_chunk_ids=vector_ids,
        graph_ranked_chunk_ids=graph_ids,
        graph_evidence_ids=_unique_chunk_ids(
            [item.evidence.evidence_id for item in graph_result.items]
        ),
        hybrid_ranked_chunk_ids=hybrid_ids,
        hybrid_items=[item.model_dump(mode="json") for item in hybrid_items],
        vector_metrics=vector_metrics,
        graph_metrics=graph_metrics,
        hybrid_metrics=hybrid_metrics,
        vector_gold_chunk_ids_at_10=sorted(vector_top10),
        graph_gold_chunk_ids_at_10=sorted(graph_top10),
        both_gold_chunk_ids_at_10=sorted(both_chunks),
        vector_gold_source_blocks_at_10=sorted(f"{doc}:{block}" for doc, block in vector_blocks),
        graph_gold_source_blocks_at_10=sorted(f"{doc}:{block}" for doc, block in graph_blocks),
        both_gold_source_blocks_at_10=sorted(f"{doc}:{block}" for doc, block in both_blocks),
        vector_missed_graph_recovered_source_blocks_at_10=sorted(
            f"{doc}:{block}" for doc, block in recovered_blocks
        ),
        graph_only_final_recovered_gold_chunk_ids_at_10=sorted(final_recovered_chunks),
        graph_only_final_recovered_source_blocks_at_10=sorted(
            f"{doc}:{block}" for doc, block in final_recovered_blocks
        ),
        vector_first_relevant_rank=vector_first_rank,
        hybrid_first_relevant_rank=hybrid_first_rank,
        first_relevant_rank_delta=rank_delta,
        mrr_delta=round(hybrid_metrics["mrr"] - vector_metrics["mrr"], 8),
        vector_baseline_latency_ms=round(vector_baseline_latency_ms, 3),
        vector_embedding_latency_ms=round(vector_embedding_latency_ms, 3),
        vector_qdrant_latency_ms=round(vector_qdrant_latency_ms, 3),
        vector_reranker_latency_ms=round(vector_reranker_latency_ms, 3),
        hybrid_vector_latency_ms=round(hybrid_vector_latency_ms, 3),
        hybrid_graph_latency_ms=round(hybrid_graph_latency_ms, 3),
        hybrid_fusion_latency_ms=round(hybrid_fusion_latency_ms, 3),
        hybrid_total_latency_ms=round(hybrid_total_latency_ms, 3),
    )


def _frozen_a25_settings(settings: Settings) -> Settings:
    """Return runtime settings for the accepted A2.5 vector profile."""

    checks = {
        "embedding_model_revision": (settings.embedding_model_revision, A25_EMBEDDING_REVISION),
        "embedding_device": (settings.embedding_device, A25_DEVICE),
        "embedding_dtype": (settings.embedding_dtype, A25_DTYPE),
        "embedding_max_seq_length": (settings.embedding_max_seq_length, A25_MAX_SEQ_LENGTH),
        "reranker_device": (settings.reranker_device, A25_DEVICE),
        "reranker_dtype": (settings.reranker_dtype, A25_DTYPE),
        "reranker_max_seq_length": (settings.reranker_max_seq_length, A25_MAX_SEQ_LENGTH),
        "qdrant_collection_name": (settings.qdrant_collection_name, A25_QDRANT_COLLECTION),
        "hybrid_graph_result_limit": (
            settings.hybrid_graph_result_limit,
            A35_GRAPH_RESULT_LIMIT,
        ),
        "hybrid_result_limit": (settings.hybrid_result_limit, A35_HYBRID_RESULT_LIMIT),
        "hybrid_rrf_k": (settings.hybrid_rrf_k, A35_RRF_K),
    }
    mismatches = [
        f"{field}={actual!r} (expected {expected!r})"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    if settings.reranker_model_revision not in {None, A25_RERANKER_REVISION}:
        mismatches.append(
            "reranker_model_revision="
            f"{settings.reranker_model_revision!r} (expected {A25_RERANKER_REVISION!r})"
        )
    if mismatches:
        raise HybridEvaluationError(
            "runtime settings are incompatible with the frozen A2.5 vector profile: "
            + "; ".join(mismatches)
        )
    return settings.model_copy(
        update={
            "embedding_batch_size": A25_BATCH_SIZE,
            "reranker_model_key": "bge-reranker-v2-m3",
            "reranker_batch_size": A25_BATCH_SIZE,
            "reranker_model_revision": A25_RERANKER_REVISION,
        }
    )


def _frozen_a25_protocol(settings: Settings) -> tuple[Settings, HybridEvaluationProtocol]:
    frozen_settings = _frozen_a25_settings(settings)
    protocol = HybridEvaluationProtocol(
        embedding_model_revision=A25_EMBEDDING_REVISION,
        embedding_config_fingerprint=embedding_config_fingerprint(frozen_settings),
        embedding_device=A25_DEVICE,
        embedding_dtype=A25_DTYPE,
        embedding_batch_size=A25_BATCH_SIZE,
        embedding_max_seq_length=A25_MAX_SEQ_LENGTH,
        reranker_model_revision=A25_RERANKER_REVISION,
        reranker_device=A25_DEVICE,
        reranker_dtype=A25_DTYPE,
        reranker_batch_size=A25_BATCH_SIZE,
        reranker_max_seq_length=A25_MAX_SEQ_LENGTH,
        qdrant_collection=A25_QDRANT_COLLECTION,
        vector_candidate_limit=A25_DENSE_TOP_K,
        vector_rerank_limit=A25_RERANK_TOP_K,
        graph_result_limit=A35_GRAPH_RESULT_LIMIT,
        hybrid_result_limit=A35_HYBRID_RESULT_LIMIT,
        rrf_k=A35_RRF_K,
    )
    expected_fingerprint = embedding_config_fingerprint(frozen_settings)
    if protocol.embedding_config_fingerprint != expected_fingerprint:
        raise HybridEvaluationError("A3.7 embedding configuration fingerprint is inconsistent")
    return frozen_settings, protocol


def _load_a25_reference_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load the ignored A2.5 baseline metrics for fail-closed reproduction."""

    if not path.is_file():
        raise HybridEvaluationError(
            f"frozen A2.5 results are required for baseline reproduction: {path}"
        )
    rows = _read_jsonl(path)
    reference: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("system") != "qdrant_dense_bge_top10":
            continue
        split = row.get("split")
        if split not in {"dev", "test"}:
            continue
        key = str(split)
        if key in reference:
            raise HybridEvaluationError(f"A2.5 baseline contains duplicate {split} results")
        if row.get("status") != "success" or row.get("candidate_pool_size") != A25_DENSE_TOP_K:
            raise HybridEvaluationError("A2.5 baseline result is not the frozen Top-10 profile")
        if row.get("dense_model_id") != A25_EMBEDDING_MODEL:
            raise HybridEvaluationError("A2.5 baseline uses an unexpected embedding model")
        if row.get("dense_model_revision") != A25_EMBEDDING_REVISION:
            raise HybridEvaluationError("A2.5 baseline uses an unexpected embedding revision")
        if row.get("reranker_model_id") != A25_RERANKER_MODEL:
            raise HybridEvaluationError("A2.5 baseline uses an unexpected reranker model")
        if row.get("reranker_model_revision") != A25_RERANKER_REVISION:
            raise HybridEvaluationError("A2.5 baseline uses an unexpected reranker revision")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in metrics.values()
        ):
            raise HybridEvaluationError("A2.5 baseline metrics are malformed")
        reference[key] = {str(name): float(value) for name, value in metrics.items()}
    if set(reference) != {"dev", "test"}:
        raise HybridEvaluationError("A2.5 baseline results must contain one dev and one test row")
    return reference


def _validate_a25_manifest(path: Path, *, dataset_path: Path) -> None:
    """Validate the non-ranking identity facts of the frozen A2.5 run."""
    manifest = _read_json_object(path)
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise HybridEvaluationError("A2.5 manifest has no valid environment record")
    expected_environment = {
        "chunk_count": EXPECTED_CHUNK_COUNT,
        "dense_embedding_model": A25_EMBEDDING_MODEL,
        "dense_embedding_revision": A25_EMBEDDING_REVISION,
        "reranker_model": A25_RERANKER_MODEL,
        "reranker_model_revision": A25_RERANKER_REVISION,
        "qdrant_collection": A25_QDRANT_COLLECTION,
        "qdrant_point_count": EXPECTED_CHUNK_COUNT,
        "qdrant_vector_dimension": QDRANT_VECTOR_DIMENSION,
        "qdrant_distance": "cosine",
    }
    mismatches = [
        f"{field}={environment.get(field)!r} (expected {expected!r})"
        for field, expected in expected_environment.items()
        if environment.get(field) != expected
    ]
    if environment.get("dataset_sha256") != _sha256(dataset_path):
        mismatches.append("dataset_sha256 does not match the frozen A2.1 dataset")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or {
        "dense_top_k": protocol.get("dense_top_k") if isinstance(protocol, dict) else None,
        "dense_max_seq_length": protocol.get("dense_max_seq_length")
        if isinstance(protocol, dict)
        else None,
        "reranker_batch_size": protocol.get("reranker_batch_size")
        if isinstance(protocol, dict)
        else None,
        "reranker_max_seq_length": protocol.get("reranker_max_seq_length")
        if isinstance(protocol, dict)
        else None,
    } != {
        "dense_top_k": A25_DENSE_TOP_K,
        "dense_max_seq_length": A25_MAX_SEQ_LENGTH,
        "reranker_batch_size": A25_BATCH_SIZE,
        "reranker_max_seq_length": A25_MAX_SEQ_LENGTH,
    }:
        mismatches.append("A2.5 manifest does not describe the frozen Top-10/512/BGE profile")
    if mismatches:
        raise HybridEvaluationError("A2.5 manifest is incompatible: " + "; ".join(mismatches))


def _aggregate_case_metrics(
    cases: Sequence[FrozenEvalCase],
    rankings: Mapping[str, Sequence[str]],
    chunks_by_id: Mapping[str, IndexedChunk],
) -> dict[str, float]:
    rows: list[Mapping[str, float]] = []
    chunk_source_blocks = {
        chunk_id: {(str(chunk.document_id), block_id) for block_id in chunk.source_block_ids}
        for chunk_id, chunk in chunks_by_id.items()
    }
    for case in cases:
        try:
            ranking = rankings[case.item.item_id]
        except KeyError as error:
            raise HybridEvaluationError(
                f"baseline ranking is missing {case.item.item_id}"
            ) from error
        rows.append(
            ranking_metrics(
                ranking,
                case.relevant_chunk_ids,
                chunk_source_blocks,
                case.gold_source_blocks,
            )
        )
    metric_names = sorted({name for row in rows for name in row})
    return {
        name: round(statistics.fmean(row[name] for row in rows), 8)
        for name in metric_names
        if all(name in row for row in rows)
    }


def _assert_a25_baseline_reproduction(
    *,
    cases_by_split: Mapping[str, Sequence[FrozenEvalCase]],
    baseline_rankings: Mapping[str, Sequence[str]],
    chunks_by_id: Mapping[str, IndexedChunk],
    reference_path: Path,
) -> dict[str, dict[str, float]]:
    reference = _load_a25_reference_metrics(reference_path)
    for split, cases in cases_by_split.items():
        actual = _aggregate_case_metrics(cases, baseline_rankings, chunks_by_id)
        expected = reference[split]
        metric_names = set(actual) | set(expected)
        mismatches = {
            name: (actual.get(name), expected.get(name))
            for name in metric_names
            if actual.get(name) is None
            or expected.get(name) is None
            or abs(actual.get(name, 0.0) - expected.get(name, 0.0)) > 1e-5
        }
        if mismatches:
            raise HybridEvaluationError(
                f"A2.5 {split} baseline cannot be reproduced within tolerance: {mismatches}"
            )
    return reference


def _validate_preflight_inputs(
    *,
    cases_by_split: Mapping[str, Sequence[FrozenEvalCase]],
    chunks_by_id: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
    qdrant: QdrantVectorStore,
    graph: Neo4jGraphStore,
    mapping_path: Path,
    graph_manifest_path: Path,
    exclusions_path: Path,
    protocol: HybridEvaluationProtocol,
) -> tuple[HybridEvaluationPreflight, set[str]]:
    if len(chunks_by_id) != EXPECTED_CHUNK_COUNT:
        raise HybridEvaluationError(
            f"A1.6 chunk index must contain {EXPECTED_CHUNK_COUNT} chunks, "
            f"found {len(chunks_by_id)}"
        )
    if len(cases_by_split.get("dev", ())) != 72 or len(cases_by_split.get("test", ())) != 36:
        raise HybridEvaluationError("frozen A2.1 split must be 72 dev and 36 test")
    frozen_document_ids = {str(chunk.document_id) for chunk in chunks_by_id.values()}
    if len(frozen_document_ids) != EXPECTED_DOCUMENT_COUNT:
        raise HybridEvaluationError(
            f"A1.6 chunk index must cover {EXPECTED_DOCUMENT_COUNT} documents, "
            f"found {len(frozen_document_ids)}"
        )
    mapping_rows = _read_jsonl(mapping_path)
    item_ids = {case.item.item_id for cases in cases_by_split.values() for case in cases}
    mapping_item_ids = [str(row.get("item_id")) for row in mapping_rows]
    if len(mapping_item_ids) != len(set(mapping_item_ids)):
        raise HybridEvaluationError("A2.1 KB mapping contains duplicate item IDs")
    if len(mapping_item_ids) != len(item_ids) or set(mapping_item_ids) != item_ids:
        raise HybridEvaluationError("A2.1 KB mapping does not cover exactly the frozen items")
    if any(str(row.get("knowledge_base_id")) != str(knowledge_base_id) for row in mapping_rows):
        raise HybridEvaluationError("A2.1 mapping is not attributed to the requested KB")

    if qdrant.collection_name != protocol.qdrant_collection:
        raise HybridEvaluationError("Qdrant collection is not the frozen A2.5 collection")
    readiness = qdrant.readiness()
    if (
        readiness.status != "ready"
        or readiness.vector_dimension != protocol.qdrant_vector_dimension
    ):
        raise HybridEvaluationError("Qdrant collection is not ready with the frozen vector schema")
    qdrant_points = qdrant.list_point_metadata()
    if len(qdrant_points) != EXPECTED_CHUNK_COUNT:
        raise HybridEvaluationError(
            f"Qdrant must contain {EXPECTED_CHUNK_COUNT} points, found {len(qdrant_points)}"
        )
    if len({point.point_id for point in qdrant_points}) != len(qdrant_points):
        raise HybridEvaluationError("Qdrant contains duplicate point IDs")
    point_chunks = {point.chunk_id for point in qdrant_points}
    if point_chunks != set(chunks_by_id):
        raise HybridEvaluationError("Qdrant points do not match the frozen chunk index")
    expected_embedding_config = protocol.embedding_config_fingerprint
    for point in qdrant_points:
        chunk = chunks_by_id[point.chunk_id]
        if point.point_id != point_id_for_chunk(point.chunk_id):
            raise HybridEvaluationError(f"Qdrant point ID is not deterministic: {point.chunk_id}")
        if point.document_id != chunk.document_id:
            raise HybridEvaluationError(f"Qdrant document lineage drift: {point.chunk_id}")
        if point.knowledge_base_id != knowledge_base_id:
            raise HybridEvaluationError("Qdrant contains a point outside the requested KB")
        if point.collection_schema_version != protocol.qdrant_schema_version:
            raise HybridEvaluationError(f"Qdrant schema metadata drift: {point.chunk_id}")
        if point.chunking_config_fingerprint != chunk.config_fingerprint:
            raise HybridEvaluationError(f"Qdrant chunk configuration drift: {point.chunk_id}")
        if point.embedding_model != protocol.embedding_model:
            raise HybridEvaluationError(f"Qdrant embedding model drift: {point.chunk_id}")
        if point.embedding_model_revision != protocol.embedding_model_revision:
            raise HybridEvaluationError(f"Qdrant embedding revision drift: {point.chunk_id}")
        if point.embedding_config_fingerprint != expected_embedding_config:
            raise HybridEvaluationError(f"Qdrant embedding configuration drift: {point.chunk_id}")

    manifest = _read_json_object(graph_manifest_path)
    if manifest.get("run_id") != DEFAULT_GRAPH_RUN_ID:
        raise HybridEvaluationError("A3.6 graph run ID does not match the audited run")
    if manifest.get("pipeline_fingerprint") != DEFAULT_GRAPH_PIPELINE_FINGERPRINT:
        raise HybridEvaluationError(
            "A3.6 graph pipeline fingerprint does not match the audited run"
        )
    _validated_graph_prompt_layout(manifest.get("prompt_layout_version"))
    if manifest.get("knowledge_base_id") != str(knowledge_base_id):
        raise HybridEvaluationError("A3.6 graph run is attributed to another KB")
    if (
        manifest.get("expected_document_count") != EXPECTED_DOCUMENT_COUNT
        or manifest.get("expected_chunk_count") != EXPECTED_CHUNK_COUNT
    ):
        raise HybridEvaluationError("A3.6 corpus snapshot is not the 255-document/7524-chunk run")
    for field_name in ("expected_document_ids", "processed_document_ids"):
        values = manifest.get(field_name)
        if (
            not isinstance(values, list)
            or len(values) != len(frozen_document_ids)
            or {str(value) for value in values} != frozen_document_ids
        ):
            raise HybridEvaluationError(
                f"A3.6 manifest {field_name} does not match the frozen corpus"
            )

    exclusions = _read_json_object(exclusions_path)
    if (
        exclusions.get("run_id") != DEFAULT_GRAPH_RUN_ID
        or exclusions.get("pipeline_fingerprint") != DEFAULT_GRAPH_PIPELINE_FINGERPRINT
        or exclusions.get("knowledge_base_id") != str(knowledge_base_id)
    ):
        raise HybridEvaluationError("A3.6 exclusion audit identity does not match the audited run")
    exclusion_rows = exclusions.get("documents")
    if not isinstance(exclusion_rows, list) or any(
        not isinstance(value, dict) for value in exclusion_rows
    ):
        raise HybridEvaluationError("A3.6 exclusion audit has invalid document records")
    excluded_documents = {str(value.get("document_id")) for value in exclusion_rows}
    if (
        len(exclusion_rows) != len(excluded_documents)
        or len(excluded_documents)
        != EXPECTED_DOCUMENT_COUNT - EXPECTED_GRAPH_ELIGIBLE_DOCUMENT_COUNT
    ):
        raise HybridEvaluationError("A3.6 exclusion audit does not cover the audited documents")
    if not excluded_documents <= frozen_document_ids:
        raise HybridEvaluationError("A3.6 exclusions contain documents outside the frozen corpus")
    manifest_failed_documents = {str(value) for value in manifest.get("failed_document_ids", [])}
    if manifest_failed_documents != excluded_documents:
        raise HybridEvaluationError("A3.6 exclusions do not match manifest failed documents")
    manifest_failed_chunks = {str(value) for value in manifest.get("failed_chunk_ids", [])}
    exclusion_failed_chunks: set[str] = set()
    chunk_document_ids = {
        chunk_id: str(chunk.document_id) for chunk_id, chunk in chunks_by_id.items()
    }
    for row in exclusion_rows:
        document_id = str(row.get("document_id"))
        failed_chunk_ids = row.get("failed_chunk_ids")
        if (
            not isinstance(failed_chunk_ids, list)
            or not failed_chunk_ids
            or any(not isinstance(value, str) or not value.strip() for value in failed_chunk_ids)
            or len(failed_chunk_ids) != len(set(failed_chunk_ids))
        ):
            raise HybridEvaluationError(f"A3.6 exclusion has no failed chunks: {document_id}")
        for chunk_id in failed_chunk_ids:
            chunk_id = str(chunk_id)
            if chunk_document_ids.get(chunk_id) != document_id:
                raise HybridEvaluationError(f"A3.6 exclusion chunk lineage drift: {chunk_id}")
            exclusion_failed_chunks.add(chunk_id)
    if exclusion_failed_chunks != manifest_failed_chunks:
        raise HybridEvaluationError("A3.6 exclusion chunks do not match manifest failed chunks")
    if len(manifest_failed_chunks) != EXPECTED_TERMINAL_FAILED_CHUNK_COUNT:
        raise HybridEvaluationError(
            "A3.6 manifest failed chunk count is not the audited final state"
        )
    coverage = read_graph_coverage(graph, knowledge_base_id)
    if len(coverage.document_ids) != EXPECTED_GRAPH_ELIGIBLE_DOCUMENT_COUNT:
        raise HybridEvaluationError(
            "current graph does not have the audited eligible document count"
        )
    if coverage.evidence_count != EXPECTED_GRAPH_EVIDENCE_COUNT:
        raise HybridEvaluationError("current graph does not have the audited evidence count")
    if coverage.entity_count != EXPECTED_GRAPH_ENTITY_COUNT:
        raise HybridEvaluationError("current graph does not have the audited entity count")
    if coverage.relation_count != EXPECTED_GRAPH_RELATION_COUNT:
        raise HybridEvaluationError("current graph does not have the audited relation count")
    if coverage.canonical_entity_count != EXPECTED_CANONICAL_ENTITY_COUNT:
        raise HybridEvaluationError("current graph does not have the audited canonical count")
    if coverage.membership_count != EXPECTED_MEMBERSHIP_COUNT:
        raise HybridEvaluationError("current graph does not have the audited membership count")
    if coverage.document_ids != frozen_document_ids - excluded_documents:
        raise HybridEvaluationError("current graph documents do not match the eligible A3.6 corpus")
    graph_chunk_sets = (
        coverage.evidence_chunk_ids,
        coverage.entity_chunk_ids,
        coverage.relation_chunk_ids,
    )
    if any(not graph_chunks <= set(chunks_by_id) for graph_chunks in graph_chunk_sets):
        raise HybridEvaluationError("current graph contains chunks outside the frozen corpus")
    if any(graph_chunks & exclusion_failed_chunks for graph_chunks in graph_chunk_sets):
        raise HybridEvaluationError(
            "excluded A3.6 documents still have retrieval-visible graph state"
        )

    graph_coverage_counts = Counter(
        _coverage_for_case(case, coverage.evidence_chunk_ids)
        for cases in cases_by_split.values()
        for case in cases
    )
    if (
        graph_coverage_counts["complete"] != EXPECTED_GRAPH_COVERED_ITEMS
        or graph_coverage_counts["partial"] != EXPECTED_GRAPH_PARTIAL_ITEMS
        or graph_coverage_counts["none"] != EXPECTED_GRAPH_UNCOVERED_ITEMS
    ):
        raise HybridEvaluationError("current graph coverage does not match the audited A2.1 split")
    return (
        HybridEvaluationPreflight(
            knowledge_base_id=knowledge_base_id,
            frozen_item_count=108,
            dev_count=72,
            test_count=36,
            qdrant_point_count=len(qdrant_points),
            qdrant_collection=qdrant.collection_name,
            qdrant_vector_dimension=QDRANT_VECTOR_DIMENSION,
            qdrant_distance="cosine",
            qdrant_embedding_model=protocol.embedding_model,
            qdrant_embedding_model_revision=protocol.embedding_model_revision,
            qdrant_embedding_config_fingerprint=expected_embedding_config,
            graph_document_count=len(coverage.document_ids),
            graph_evidence_count=coverage.evidence_count,
            graph_supported_chunk_count=len(coverage.evidence_chunk_ids),
            graph_entity_count=coverage.entity_count,
            graph_relation_count=coverage.relation_count,
            canonical_entity_count=coverage.canonical_entity_count,
            membership_count=coverage.membership_count,
            graph_coverage_complete_count=graph_coverage_counts["complete"],
            graph_coverage_partial_count=graph_coverage_counts["partial"],
            graph_coverage_none_count=graph_coverage_counts["none"],
            graph_run_id=str(manifest["run_id"]),
            graph_pipeline_fingerprint=str(manifest["pipeline_fingerprint"]),
        ),
        set(coverage.evidence_chunk_ids),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridEvaluationError(f"evaluation JSON is not readable: {path}") from error
    if not isinstance(value, dict):
        raise HybridEvaluationError(f"evaluation JSON must be an object: {path}")
    return value


def _validated_graph_prompt_layout(value: Any) -> str:
    """Apply A3.6's explicit compatibility fallback for old manifests."""
    if value is None:
        return GRAPH_PROMPT_LAYOUT_VERSION
    if value != GRAPH_PROMPT_LAYOUT_VERSION:
        raise HybridEvaluationError("A3.6 graph run uses an incompatible prompt layout")
    return value


def _build_runtime(settings: Settings, protocol: HybridEvaluationProtocol) -> _Runtime:
    qdrant = QdrantVectorStore(settings)
    neo4j = Neo4jGraphStore(settings)
    vector_timing = _StageTiming()
    hybrid_vector_timing = _StageTiming()
    embedding = QwenEmbeddingModel(settings)
    baseline_vector = DenseRetrievalService(
        _TimedVectorStore(qdrant, vector_timing),
        _TimedEncoder(embedding, vector_timing),
    )
    hybrid_vector = DenseRetrievalService(
        _TimedVectorStore(qdrant, hybrid_vector_timing),
        _TimedEncoder(embedding, hybrid_vector_timing),
    )
    reranker_settings = settings
    if reranker_settings.reranker_model_revision is None:
        reranker_settings = settings.model_copy(
            update={"reranker_model_revision": BGE_RERANKER_MODEL_REVISION}
        )
    reranker = create_local_reranker(reranker_settings, model_key="bge-reranker-v2-m3")
    baseline_reranking = _TimedReranking(RerankingService(reranker), vector_timing)
    hybrid_reranking = _TimedReranking(RerankingService(reranker), hybrid_vector_timing)
    # The standalone baseline is timed by the caller around the exact same
    # production composition; its stage proxy remains separate from hybrid.
    graph = GraphRetrievalService(
        neo4j,
        config=GraphRetrievalConfig(
            max_seed_entities=settings.graph_retrieval_max_seed_entities,
            max_hops=settings.graph_retrieval_max_hops,
            max_neighbors=settings.graph_retrieval_max_neighbors,
            max_relations=settings.graph_retrieval_max_relations,
            max_evidence=settings.graph_retrieval_max_evidence,
            max_entity_scan=settings.graph_retrieval_max_entity_scan,
            lexical_threshold=settings.graph_retrieval_lexical_threshold,
        ),
    )
    hybrid_config = HybridRetrievalConfig(
        rrf_k=protocol.rrf_k,
        vector_candidate_limit=protocol.vector_candidate_limit,
        vector_rerank_limit=protocol.vector_rerank_limit,
        graph_result_limit=protocol.graph_result_limit,
        result_limit=protocol.hybrid_result_limit,
        failure_mode=protocol.failure_mode,
    )
    baseline = HybridRetrievalService(
        baseline_vector,
        baseline_reranking,
        graph,
        config=hybrid_config,
    )
    hybrid = HybridRetrievalService(
        hybrid_vector,
        hybrid_reranking,
        graph,
        config=hybrid_config,
    )
    return _Runtime(qdrant, neo4j, baseline, hybrid, vector_timing, hybrid_vector_timing)


async def _run_one(
    runtime: _Runtime,
    case: FrozenEvalCase,
    *,
    knowledge_base_id: UUID,
    split: Literal["dev", "test"],
    coverage: CoverageClass,
    chunks_by_id: Mapping[str, IndexedChunk],
    baseline: _VectorMeasurement,
) -> HybridQueryRecord:
    runtime.hybrid_vector_timing.embedding_ms = 0.0
    runtime.hybrid_vector_timing.qdrant_ms = 0.0
    runtime.hybrid_vector_timing.reranker_ms = 0.0
    started = time.perf_counter()
    vector_execution, graph_execution = await asyncio.gather(
        runtime.hybrid._execute_vector(case.item.query, knowledge_base_id, None),
        runtime.hybrid._execute_graph(case.item.query, knowledge_base_id, None),
    )
    if vector_execution.status in {"failed", "timed_out"}:
        raise HybridEvaluationError(vector_execution.error or "vector branch failed")
    if graph_execution.status in {"failed", "timed_out"}:
        raise HybridEvaluationError(graph_execution.error or "graph branch failed")
    shared_vector = list(vector_execution.items)
    shared_ids = tuple(item.chunk.payload.chunk_id for item in shared_vector)
    if shared_ids != baseline.ranked_chunk_ids:
        raise HybridEvaluationError(
            f"vector branch changed between baseline and hybrid for {case.item.item_id}"
        )
    fusion_started = time.perf_counter()
    hybrid_items = runtime.hybrid._fuse(shared_vector, graph_execution.items)
    fusion_ms = (time.perf_counter() - fusion_started) * 1_000
    total_ms = (time.perf_counter() - started) * 1_000
    graph_result = GraphRetrievalResult(
        query=case.item.query,
        knowledge_base_id=knowledge_base_id,
        seeds=[],
        items=list(graph_execution.items),
    )
    _validate_graph_result(
        graph_result,
        knowledge_base_id=knowledge_base_id,
        chunks_by_id=chunks_by_id,
    )
    record = evaluate_query_record(
        case,
        split=split,
        coverage=coverage,
        chunks_by_id=chunks_by_id,
        vector_ranked_chunk_ids=baseline.ranked_chunk_ids,
        graph_result=graph_result,
        hybrid_items=hybrid_items,
        vector_status=vector_execution.status,
        graph_status=graph_execution.status,
        vector_error=vector_execution.error,
        graph_error=graph_execution.error,
        vector_baseline_latency_ms=baseline.total_latency_ms,
        vector_embedding_latency_ms=baseline.embedding_latency_ms,
        vector_qdrant_latency_ms=baseline.qdrant_latency_ms,
        vector_reranker_latency_ms=baseline.reranker_latency_ms,
        hybrid_vector_latency_ms=vector_execution.latency_ms,
        hybrid_graph_latency_ms=graph_execution.latency_ms,
        hybrid_fusion_latency_ms=fusion_ms,
        hybrid_total_latency_ms=total_ms,
    )
    return record


async def _measure_vector_baseline(
    runtime: _Runtime,
    case: FrozenEvalCase,
    knowledge_base_id: UUID,
) -> _VectorMeasurement:
    runtime.vector_timing.embedding_ms = 0.0
    runtime.vector_timing.qdrant_ms = 0.0
    runtime.vector_timing.reranker_ms = 0.0
    started = time.perf_counter()
    try:
        items = await asyncio.to_thread(
            runtime.baseline._search_vector_sync,
            case.item.query,
            knowledge_base_id,
            None,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise HybridEvaluationError("vector baseline evaluation failed") from error
    return _VectorMeasurement(
        ranked_chunk_ids=tuple(item.chunk.payload.chunk_id for item in items),
        total_latency_ms=(time.perf_counter() - started) * 1_000,
        embedding_latency_ms=runtime.vector_timing.embedding_ms,
        qdrant_latency_ms=runtime.vector_timing.qdrant_ms,
        reranker_latency_ms=runtime.vector_timing.reranker_ms,
    )


async def _warm_up_full_path(
    runtime: _Runtime,
    case: FrozenEvalCase,
    knowledge_base_id: UUID,
) -> None:
    """Warm both existing branches and the fusion code before measurement."""

    try:
        vector_execution, graph_execution = await asyncio.gather(
            runtime.hybrid._execute_vector(case.item.query, knowledge_base_id, None),
            runtime.hybrid._execute_graph(case.item.query, knowledge_base_id, None),
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise HybridEvaluationError("full retrieval warmup failed") from error
    if vector_execution.status in {"failed", "timed_out"}:
        raise HybridEvaluationError(vector_execution.error or "vector warmup failed")
    if graph_execution.status in {"failed", "timed_out"}:
        raise HybridEvaluationError(graph_execution.error or "graph warmup failed")
    runtime.hybrid._fuse(vector_execution.items, graph_execution.items)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


async def _run_hybrid_evaluation_async(
    *,
    knowledge_base_id: UUID,
    split: Literal["dev", "test", "both"] = "both",
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    mapping_path: Path = DEFAULT_KB_MAPPING,
    graph_manifest_path: Path = DEFAULT_GRAPH_RUN_MANIFEST,
    exclusions_path: Path = DEFAULT_GRAPH_EXCLUSIONS,
    a25_results_path: Path = DEFAULT_A25_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the read-only A3.7 comparison on the frozen A2.1 set."""
    if split not in {"dev", "test", "both"}:
        raise HybridEvaluationError(f"unsupported split: {split}")
    selected_settings = settings or get_settings()
    frozen_settings, protocol = _frozen_a25_protocol(selected_settings)
    _validate_a25_manifest(DEFAULT_A25_MANIFEST, dataset_path=dataset_path)
    chunks_by_id = load_frozen_chunk_index(chunk_index_path)
    requested = ("dev", "test") if split == "both" else (split,)
    all_cases_by_split = {
        selected: load_frozen_eval_cases(
            selected,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            chunk_index=chunks_by_id,
        )[0]
        for selected in ("dev", "test")
    }
    cases_by_split = {selected: all_cases_by_split[selected] for selected in requested}
    runtime = _build_runtime(frozen_settings, protocol)
    try:
        readiness = runtime.qdrant.readiness()
        if readiness.status != "ready":
            raise HybridEvaluationError(readiness.error or "Qdrant is not ready")
        preflight, graph_supported_chunks = _validate_preflight_inputs(
            cases_by_split=all_cases_by_split,
            chunks_by_id=chunks_by_id,
            knowledge_base_id=knowledge_base_id,
            qdrant=runtime.qdrant,
            graph=runtime.neo4j,
            mapping_path=mapping_path,
            graph_manifest_path=graph_manifest_path,
            exclusions_path=exclusions_path,
            protocol=protocol,
        )
        warmup_cases = all_cases_by_split["dev"]
        if protocol.warmup and warmup_cases:
            await _warm_up_full_path(runtime, warmup_cases[0], knowledge_base_id)

        baseline_measurements: dict[str, _VectorMeasurement] = {}
        baseline_rankings: dict[str, Sequence[str]] = {}
        for split_name in requested:
            for case in cases_by_split[split_name]:
                measurement = await _measure_vector_baseline(runtime, case, knowledge_base_id)
                baseline_measurements[case.item.item_id] = measurement
                baseline_rankings[case.item.item_id] = measurement.ranked_chunk_ids
        a25_reference = _assert_a25_baseline_reproduction(
            cases_by_split=cases_by_split,
            baseline_rankings=baseline_rankings,
            chunks_by_id=chunks_by_id,
            reference_path=a25_results_path,
        )
        records: list[HybridQueryRecord] = []
        for split_name in requested:
            for case in cases_by_split[split_name]:
                coverage = _coverage_for_case(case, graph_supported_chunks)
                records.append(
                    await _run_one(
                        runtime,
                        case,
                        knowledge_base_id=knowledge_base_id,
                        split=split_name,
                        coverage=coverage,
                        chunks_by_id=chunks_by_id,
                        baseline=baseline_measurements[case.item.item_id],
                    )
                )
        by_name = {
            name: [record for record in records if record.split == name] for name in requested
        }
        summary_groups: dict[str, list[HybridQueryRecord]] = {}
        for name, values in by_name.items():
            summary_groups[name] = values
        summary_groups["all"] = records
        for coverage in ("complete", "partial", "none"):
            values = [record for record in records if record.coverage == coverage]
            if values:
                summary_groups[f"graph_{coverage}"] = values
                for split_name in requested:
                    split_values = [record for record in values if record.split == split_name]
                    if split_values:
                        summary_groups[f"{split_name}_graph_{coverage}"] = split_values
        query_types = sorted({record.query_type for record in records})
        for split_name in requested:
            for query_type in query_types:
                values = [
                    record
                    for record in records
                    if record.split == split_name and record.query_type == query_type
                ]
                if values:
                    summary_groups[f"{split_name}_{query_type}"] = values
        summaries = [_summary(name, values) for name, values in summary_groups.items()]
        contribution_by_split = {
            name: summarize_contribution(values).model_dump(mode="json")
            for name, values in by_name.items()
        }
        contribution_by_split["all"] = summarize_contribution(records).model_dump(mode="json")
        rank_movement = {
            name: summarize_rank_movement(values) for name, values in summary_groups.items()
        }
        mrr_delta_distribution = {
            name: summarize_mrr_deltas(values) for name, values in summary_groups.items()
        }
        branch_observations = {
            name: summarize_branch_observations(values).model_dump(mode="json")
            for name, values in by_name.items()
        }
        branch_observations["all"] = summarize_branch_observations(records).model_dump(mode="json")
        output_dir.mkdir(parents=True, exist_ok=True)
        rankings_path = output_dir / "per-query.jsonl"
        _write_jsonl(rankings_path, [record.model_dump(mode="json") for record in records])
        aggregate = {
            "schema_version": A37_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": protocol.model_dump(mode="json"),
            "preflight": preflight.model_dump(mode="json"),
            "input_fingerprints": {
                "chunk_index_sha256": _sha256(chunk_index_path),
                "dataset_sha256": _sha256(dataset_path),
                "materialized_sha256": _sha256(materialized_path),
                "mapping_sha256": _sha256(mapping_path),
                "graph_manifest_sha256": _sha256(graph_manifest_path),
                "graph_exclusions_sha256": _sha256(exclusions_path),
                "a25_results_sha256": _sha256(a25_results_path),
                "a25_manifest_sha256": _sha256(DEFAULT_A25_MANIFEST),
            },
            "a25_baseline_reference": {
                "results_path": str(a25_results_path),
                "results_sha256": _sha256(a25_results_path),
                "manifest_path": str(DEFAULT_A25_MANIFEST),
                "manifest_sha256": _sha256(DEFAULT_A25_MANIFEST),
                "metrics": a25_reference,
                "profile_version": A25_PROFILE_VERSION,
            },
            "a21_provenance": {
                "item_count": 108,
                "dev_count": 72,
                "test_count": 36,
                "dataset_sha256": _sha256(dataset_path),
                "materialized_sha256": _sha256(materialized_path),
                "metric_semantics": "A2.1 ranking_metrics",
            },
            "qdrant_provenance": {
                "collection": preflight.qdrant_collection,
                "point_count": preflight.qdrant_point_count,
                "vector_dimension": preflight.qdrant_vector_dimension,
                "distance": preflight.qdrant_distance,
                "embedding_model": preflight.qdrant_embedding_model,
                "embedding_model_revision": preflight.qdrant_embedding_model_revision,
                "embedding_config_fingerprint": preflight.qdrant_embedding_config_fingerprint,
                "point_identity": "point_id_for_chunk(chunk_id)",
            },
            "a35_fusion_provenance": {
                "rrf_k": protocol.rrf_k,
                "graph_result_limit": protocol.graph_result_limit,
                "hybrid_result_limit": protocol.hybrid_result_limit,
                "implementation": "existing A3.5 RRF rank fusion",
            },
            "evaluation_split": split,
            "graph_configuration": {
                "max_seed_entities": selected_settings.graph_retrieval_max_seed_entities,
                "max_hops": selected_settings.graph_retrieval_max_hops,
                "max_neighbors": selected_settings.graph_retrieval_max_neighbors,
                "max_relations": selected_settings.graph_retrieval_max_relations,
                "max_evidence": selected_settings.graph_retrieval_max_evidence,
                "max_entity_scan": selected_settings.graph_retrieval_max_entity_scan,
                "lexical_threshold": selected_settings.graph_retrieval_lexical_threshold,
            },
            "graph_provenance": {
                "run_id": preflight.graph_run_id,
                "pipeline_fingerprint": preflight.graph_pipeline_fingerprint,
                "prompt_layout_version": GRAPH_PROMPT_LAYOUT_VERSION,
                "truncation_budgets": list(GRAPH_TRUNCATION_BUDGETS),
                "cache_layout": "cache-v2 not used",
            },
            "summaries": [summary.model_dump(mode="json") for summary in summaries],
            "contribution": contribution_by_split,
            "rank_movement": rank_movement,
            "mrr_delta_distribution": mrr_delta_distribution,
            "branch_observations": branch_observations,
            "per_query_file": str(rankings_path),
            "note": "Retrieval-only evaluation; no answer-generation LLM calls were made.",
        }
        aggregate_path = output_dir / "aggregate.json"
        aggregate_path.write_text(
            json.dumps(aggregate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return aggregate
    finally:
        runtime.qdrant.close()
        runtime.neo4j.close()


def run_hybrid_evaluation(
    *,
    knowledge_base_id: UUID,
    split: Literal["dev", "test", "both"] = "both",
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    mapping_path: Path = DEFAULT_KB_MAPPING,
    graph_manifest_path: Path = DEFAULT_GRAPH_RUN_MANIFEST,
    exclusions_path: Path = DEFAULT_GRAPH_EXCLUSIONS,
    a25_results_path: Path = DEFAULT_A25_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the asynchronous evaluator from a regular developer/CLI entry point."""
    return asyncio.run(
        _run_hybrid_evaluation_async(
            knowledge_base_id=knowledge_base_id,
            split=split,
            chunk_index_path=chunk_index_path,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            mapping_path=mapping_path,
            graph_manifest_path=graph_manifest_path,
            exclusions_path=exclusions_path,
            a25_results_path=a25_results_path,
            output_dir=output_dir,
            settings=settings,
        )
    )


__all__ = [
    "A37_SCHEMA_VERSION",
    "DEFAULT_OUTPUT",
    "HybridBranchObservation",
    "HybridContributionSummary",
    "HybridCutoffContribution",
    "HybridEvaluationError",
    "HybridEvaluationPreflight",
    "HybridEvaluationProtocol",
    "HybridQueryRecord",
    "HybridSplitSummary",
    "evaluate_query_record",
    "run_hybrid_evaluation",
    "summarize_branch_observations",
    "summarize_contribution",
]
