"""Benchmark the KnowledgeScope dense-to-rerank retrieval system.

The benchmark consumes the frozen A1.6 chunk index and A2.1 annotations.  It
does not rebuild canonical documents, change evaluation labels, or add another
retrieval implementation.  Runtime rankings and measurements are written to
the ignored ``data/evaluation/a2-5`` directory.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import platform
import statistics
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from knowledge_scope.evaluation.embedding_benchmark import (
    FrozenEvalCase,
    load_frozen_chunk_index,
    load_frozen_eval_cases,
)
from knowledge_scope.evaluation.retrieval_eval import IndexedChunk
from knowledge_scope.evaluation.retrieval_metrics import SourceBlockKey, ranking_metrics
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import (
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    QdrantVectorStore,
    RetrievedChunk,
)
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.shared.config import Settings, get_settings

RETRIEVAL_SYSTEM_BENCHMARK_SCHEMA_VERSION = "1.0"
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_MATERIALIZED = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/a2-5")
DEFAULT_DENSE_TOP_K = 10
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_SEQ_LENGTH = 512
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
SUPPORTED_SPLITS = ("dev", "test", "both")

SplitName = Literal["dev", "test"]
RunSplit = Literal["dev", "test", "both"]
RunStatus = Literal["success", "failed"]
SystemName = Literal["exact_dense", "qdrant_dense", "qdrant_dense_bge_top10"]


class RetrievalSystemBenchmarkError(RuntimeError):
    """Raised when the frozen inputs or local retrieval system is unsafe to benchmark."""


class _BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalSystemBenchmarkProtocol(_BenchmarkModel):
    """One fixed protocol for all A2.5 system comparisons."""

    dense_batch_size: StrictInt = Field(default=4, ge=1)
    reranker_batch_size: StrictInt = Field(default=DEFAULT_RERANKER_BATCH_SIZE, ge=1)
    dense_max_seq_length: StrictInt = Field(default=512, ge=1)
    reranker_max_seq_length: StrictInt = Field(default=DEFAULT_RERANKER_MAX_SEQ_LENGTH, ge=1)
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    device: str = Field(default="cuda", min_length=1)
    dense_top_k: StrictInt = Field(default=DEFAULT_DENSE_TOP_K, ge=1, le=100)
    warmup_queries: StrictInt = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_fixed_profile(self) -> Self:
        if self.dense_top_k != DEFAULT_DENSE_TOP_K:
            raise ValueError("A2.5 requires a strict dense and reranker Top-10 profile")
        if self.reranker_max_seq_length != DEFAULT_RERANKER_MAX_SEQ_LENGTH:
            raise ValueError("A2.5 reranker max sequence length must remain 512")
        return self


class RetrievalSystemBenchmarkEnvironment(_BenchmarkModel):
    """Runtime and frozen-input identity captured with benchmark results."""

    python_version: str
    platform: str
    torch_version: str
    transformers_version: str
    sentence_transformers_version: str
    accelerate_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_vram_mb: float | None
    chunk_count: StrictInt
    chunk_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_embedding_model: str
    dense_embedding_revision: str
    reranker_model: str
    reranker_model_revision: str
    qdrant_collection: str
    qdrant_point_count: StrictInt
    qdrant_vector_dimension: StrictInt
    qdrant_distance: str
    qdrant_full_scan: bool | None


class RetrievalSystemBenchmarkResult(_BenchmarkModel):
    """Metrics and performance facts for one system and split."""

    schema_version: Literal["1.0"] = RETRIEVAL_SYSTEM_BENCHMARK_SCHEMA_VERSION
    system: SystemName
    split: SplitName
    status: RunStatus
    corpus_count: StrictInt
    query_count: StrictInt
    candidate_pool_size: StrictInt
    dense_model_id: str
    dense_model_revision: str
    reranker_model_id: str | None = None
    reranker_model_revision: str | None = None
    reranker_actual_dtype: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    model_load_time_seconds: float | None = None
    corpus_encoding_time_seconds: float | None = None
    retrieval_latency_mean_ms: float | None = None
    retrieval_latency_p50_ms: float | None = None
    retrieval_latency_p95_ms: float | None = None
    reranking_latency_mean_ms: float | None = None
    reranking_latency_p50_ms: float | None = None
    reranking_latency_p95_ms: float | None = None
    total_query_latency_mean_ms: float | None = None
    total_query_latency_p50_ms: float | None = None
    total_query_latency_p95_ms: float | None = None
    retrieval_throughput_per_second: float | None = None
    reranking_throughput_per_second: float | None = None
    total_throughput_per_second: float | None = None
    peak_cuda_allocated_mb: float | None = None
    peak_cuda_reserved_mb: float | None = None
    reranker_load_time_seconds: float | None = None
    reranker_load_peak_cuda_allocated_mb: float | None = None
    reranker_load_peak_cuda_reserved_mb: float | None = None
    reranker_truncated_candidate_pairs: StrictInt | None = None
    reranker_candidate_pair_count: StrictInt | None = None
    reranker_truncated_gold_items: StrictInt | None = None
    ranking_file: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class QdrantRankingComparison(_BenchmarkModel):
    """Comparison between exact dense and Qdrant Top-10 rankings."""

    split: SplitName
    query_count: StrictInt
    identical_order_count: StrictInt
    identical_order_rate: float
    mean_top10_set_overlap: float
    quality_delta: dict[str, float] = Field(default_factory=dict)
    ann_mode: Literal["full_scan", "ann", "unknown"]
    ann_related_miss_count: StrictInt
    document_filter_queries: StrictInt
    document_filter_empty_count: StrictInt
    document_filter_violations: StrictInt


class BadCase(_BenchmarkModel):
    """A safe, metadata-only classification of one failed test query."""

    item_id: str
    categories: tuple[str, ...]
    dense_top10_has_gold: bool
    bge_top10_has_gold: bool
    gold_chunk_count: StrictInt
    relevant_gold_chunk_lengths: tuple[StrictInt, ...]


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RetrievalSystemBenchmarkError(f"benchmark input is not readable: {path}") from error
    return digest.hexdigest()


def _torch_runtime() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RetrievalSystemBenchmarkError(
            "retrieval benchmark dependencies are missing; run uv sync --group embedding-benchmark"
        ) from error
    return torch


def _cross_encoder_runtime() -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RetrievalSystemBenchmarkError(
            "reranker dependencies are missing; run uv sync --group embedding-benchmark"
        ) from error
    return CrossEncoder


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak_memory(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _memory_snapshot(torch: Any, device: str) -> tuple[float | None, float | None]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.max_memory_allocated() / (1024 * 1024),
        torch.cuda.max_memory_reserved() / (1024 * 1024),
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[position] * 1000


def _aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {
        name: statistics.fmean(row[name] for row in rows)
        for name in names
        if all(name in row for row in rows)
    }


def _passage_text(chunk: IndexedChunk) -> str:
    """Match the A2.3 indexing carrier for empty asset-only chunks."""
    if chunk.text.strip():
        return chunk.text
    section = " / ".join(chunk.section_path)
    content_types = ", ".join(chunk.content_types)
    return " ".join(part for part in (section, f"[{content_types}]") if part)


def _source_block_map(chunks: Sequence[IndexedChunk]) -> dict[str, set[SourceBlockKey]]:
    return {
        chunk.chunk_id: {(str(chunk.document_id), block_id) for block_id in chunk.source_block_ids}
        for chunk in chunks
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def compare_dense_and_qdrant_rankings(
    split: SplitName,
    exact_rankings: Mapping[str, Sequence[str]],
    qdrant_rankings: Mapping[str, Sequence[str]],
    exact_metrics: Mapping[str, float],
    qdrant_metrics: Mapping[str, float],
    *,
    ann_mode: Literal["full_scan", "ann", "unknown"],
    document_filter_queries: int,
    document_filter_empty_count: int,
    document_filter_violations: int,
) -> QdrantRankingComparison:
    """Summarize agreement without treating any ranking change as an ANN miss."""
    item_ids = sorted(exact_rankings)
    if set(item_ids) != set(qdrant_rankings):
        raise RetrievalSystemBenchmarkError("exact and Qdrant rankings cover different queries")
    identical = 0
    overlaps: list[float] = []
    ann_related_misses = 0
    for item_id in item_ids:
        exact = tuple(exact_rankings[item_id][:DEFAULT_DENSE_TOP_K])
        qdrant = tuple(qdrant_rankings[item_id][:DEFAULT_DENSE_TOP_K])
        if exact == qdrant:
            identical += 1
        exact_set = set(exact)
        qdrant_set = set(qdrant)
        overlaps.append(len(exact_set & qdrant_set) / DEFAULT_DENSE_TOP_K)
        if ann_mode == "ann" and exact_set - qdrant_set:
            ann_related_misses += 1
    quality_delta = {
        key: qdrant_metrics.get(key, 0.0) - exact_metrics.get(key, 0.0)
        for key in sorted(set(exact_metrics) | set(qdrant_metrics))
    }
    return QdrantRankingComparison(
        split=split,
        query_count=len(item_ids),
        identical_order_count=identical,
        identical_order_rate=identical / len(item_ids) if item_ids else 0.0,
        mean_top10_set_overlap=statistics.fmean(overlaps) if overlaps else 0.0,
        quality_delta=quality_delta,
        ann_mode=ann_mode,
        ann_related_miss_count=ann_related_misses,
        document_filter_queries=document_filter_queries,
        document_filter_empty_count=document_filter_empty_count,
        document_filter_violations=document_filter_violations,
    )


def classify_bad_case(
    case: FrozenEvalCase,
    dense_chunk_ids: Sequence[str],
    reranked_chunk_ids: Sequence[str],
    chunks_by_id: Mapping[str, IndexedChunk],
    *,
    truncated_chunk_ids: Collection[str] = (),
) -> BadCase | None:
    """Classify a test miss using observable retrieval facts only.

    Truncation is deliberately reported as a risk category.  A1.6 has no
    character/token offsets inside ``source_block_ids``, so this function never
    claims that a gold answer span was definitely removed.
    """
    gold = set(case.relevant_chunk_ids)
    dense = set(dense_chunk_ids)
    reranked = set(reranked_chunk_ids)
    dense_has_gold = bool(gold & dense)
    reranked_has_gold = bool(gold & reranked)
    categories: list[str] = []
    if not dense_has_gold:
        categories.append("evidence_absent_from_dense_top10")
    else:
        dense_rank = _first_rank(dense_chunk_ids, gold)
        reranked_rank = _first_rank(reranked_chunk_ids, gold)
        if dense_rank is not None and reranked_rank is not None and reranked_rank > dense_rank:
            categories.append("reranker_ranked_relevant_lower")
    relevant_truncated = gold & set(truncated_chunk_ids)
    if relevant_truncated:
        categories.append("long_or_truncated_relevant_candidate")
    if len(gold) > 1:
        categories.append("multi_chunk_or_chunking_boundary")
    if case.item.query_type in {"comparison", "cross_block"}:
        categories.append("ambiguous_or_cross_block_query")
    if not categories:
        categories.append("other_retrieval_failure")
    return BadCase(
        item_id=case.item.item_id,
        categories=tuple(categories),
        dense_top10_has_gold=dense_has_gold,
        bge_top10_has_gold=reranked_has_gold,
        gold_chunk_count=len(gold),
        relevant_gold_chunk_lengths=tuple(
            sorted(
                len(chunks_by_id[chunk_id].text) for chunk_id in gold if chunk_id in chunks_by_id
            )
        ),
    )


def _first_rank(ranking: Sequence[str], relevant: set[str]) -> int | None:
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            return rank
    return None


def _qdrant_chunk_ids(
    store: QdrantVectorStore,
    expected_chunk_ids: set[str],
) -> tuple[int, dict[str, object]]:
    """Verify that the collection contains exactly the frozen indexed chunks."""
    client = store._get_client()
    observed: set[str] = set()
    offset: Any | None = None
    while True:
        records, offset = client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for record in records:
            payload = getattr(record, "payload", None)
            chunk_id = payload.get("chunk_id") if isinstance(payload, dict) else None
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise RetrievalSystemBenchmarkError(
                    "Qdrant collection contains a point without a valid chunk_id payload"
                )
            observed.add(chunk_id)
        if offset is None:
            break
    if observed != expected_chunk_ids:
        missing = len(expected_chunk_ids - observed)
        extra = len(observed - expected_chunk_ids)
        raise RetrievalSystemBenchmarkError(
            "Qdrant collection must contain exactly the frozen A1.6 chunk IDs "
            f"(expected={len(expected_chunk_ids)}, actual={len(observed)}, "
            f"missing={missing}, extra={extra})"
        )
    info = client.get_collection(store.collection_name)
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    size = getattr(vectors, "size", None)
    distance = str(getattr(vectors, "distance", "")).casefold()
    hnsw = getattr(getattr(info, "config", None), "hnsw_config", None)
    full_scan_threshold = getattr(hnsw, "full_scan_threshold", None)
    indexed_vectors_count = getattr(info, "indexed_vectors_count", None)
    full_scan = (
        isinstance(full_scan_threshold, int)
        and len(observed) < full_scan_threshold
        and indexed_vectors_count == 0
    )
    return len(observed), {
        "vector_dimension": size,
        "distance": distance,
        "full_scan": full_scan,
        "full_scan_threshold": full_scan_threshold,
        "indexed_vectors_count": indexed_vectors_count,
    }


def _validate_retrieved_items(
    items: Sequence[RetrievedChunk],
    chunks_by_id: Mapping[str, IndexedChunk],
) -> tuple[str, ...]:
    ids = tuple(item.payload.chunk_id for item in items)
    if len(ids) != len(set(ids)):
        raise RetrievalSystemBenchmarkError("Qdrant returned duplicate chunk IDs")
    if any(chunk_id not in chunks_by_id for chunk_id in ids):
        raise RetrievalSystemBenchmarkError("Qdrant returned a chunk outside the frozen index")
    if any(left.score < right.score for left, right in itertools.pairwise(items)):
        raise RetrievalSystemBenchmarkError("Qdrant results are not ordered by descending score")
    return ids


def _reranker_model_dtype(model: Any) -> str | None:
    loaded = getattr(model, "model", model)
    parameters = getattr(loaded, "parameters", None)
    if parameters is None:
        return None
    try:
        return str(next(parameters()).dtype).removeprefix("torch.")
    except (StopIteration, TypeError):
        return None


def _reranker_model_revision(model: Any) -> str | None:
    loaded = getattr(model, "model", model)
    config = getattr(loaded, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None


class _BgeReranker:
    """Small benchmark-local adapter for the already selected A2.4 BGE model."""

    def __init__(self, protocol: RetrievalSystemBenchmarkProtocol) -> None:
        self.protocol = protocol
        self.model: Any | None = None
        self.torch: Any | None = None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        torch = _torch_runtime()
        cross_encoder = _cross_encoder_runtime()
        if self.protocol.device.startswith("cuda") and not torch.cuda.is_available():
            raise RetrievalSystemBenchmarkError(
                "CUDA reranker device was requested but is unavailable"
            )
        model_kwargs: dict[str, Any] = {}
        if self.protocol.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[self.protocol.dtype]
        try:
            self.model = cross_encoder(
                BGE_RERANKER_MODEL_ID,
                device=self.protocol.device,
                revision=BGE_RERANKER_MODEL_REVISION,
                model_kwargs=model_kwargs,
                max_length=self.protocol.reranker_max_seq_length,
            )
        except Exception as error:
            raise RetrievalSystemBenchmarkError(
                f"BGE reranker could not be loaded: {BGE_RERANKER_MODEL_ID}"
            ) from error
        self.torch = torch
        return self.model

    @property
    def actual_dtype(self) -> str | None:
        return _reranker_model_dtype(self.model) if self.model is not None else None

    @property
    def model_revision(self) -> str:
        return _reranker_model_revision(self.model) or BGE_RERANKER_MODEL_REVISION

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if not passages:
            return []
        model = self.load()
        try:
            scores = model.predict(
                [(query, passage) for passage in passages],
                batch_size=self.protocol.reranker_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as error:
            raise RetrievalSystemBenchmarkError("BGE reranker scoring failed") from error
        if hasattr(scores, "reshape"):
            values = scores.reshape(-1).tolist()
        elif isinstance(scores, (list, tuple)):
            values = scores
        else:
            values = [scores]
        try:
            result = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise RetrievalSystemBenchmarkError(
                "BGE reranker returned non-numeric scores"
            ) from error
        if len(result) != len(passages) or any(not math.isfinite(value) for value in result):
            raise RetrievalSystemBenchmarkError("BGE reranker returned invalid scores")
        return result

    def truncated_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[bool] | None:
        """Audit full tokenizer lengths without asserting where a gold span lies."""
        model = self.load()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        flags: list[bool] = []
        original_max_length = getattr(tokenizer, "model_max_length", None)
        try:
            # Tokenize without truncation for the audit, while avoiding the
            # tokenizer warning that the intentionally overlong input is
            # being passed to the model.
            if isinstance(original_max_length, int):
                tokenizer.model_max_length = 1_000_000
            for query, passage in pairs:
                encoded = tokenizer(
                    query,
                    passage,
                    truncation=False,
                    add_special_tokens=True,
                )
                flags.append(len(encoded["input_ids"]) > self.protocol.reranker_max_seq_length)
        except Exception:
            return None
        finally:
            if isinstance(original_max_length, int):
                tokenizer.model_max_length = original_max_length
        return flags


def _requested_splits(split: RunSplit) -> tuple[SplitName, ...]:
    if split == "both":
        return ("dev", "test")
    return (split,)


def _load_cases(
    split: RunSplit,
    chunk_index_path: Path,
    dataset_path: Path,
    materialized_path: Path,
) -> tuple[dict[str, IndexedChunk], dict[SplitName, list[FrozenEvalCase]]]:
    chunks_by_id = load_frozen_chunk_index(chunk_index_path)
    requested = _requested_splits(split)
    cases_by_split = {
        requested_split: load_frozen_eval_cases(
            requested_split,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            chunk_index=chunks_by_id,
        )[0]
        for requested_split in requested
    }
    return chunks_by_id, cases_by_split


def _environment(
    torch: Any,
    chunks: Sequence[IndexedChunk],
    chunk_index_path: Path,
    dataset_path: Path,
    settings: Settings,
    collection_name: str,
    qdrant_point_count: int,
    qdrant_info: Mapping[str, object],
    reranker_revision: str,
) -> RetrievalSystemBenchmarkEnvironment:
    cuda_available = bool(torch.cuda.is_available())
    return RetrievalSystemBenchmarkEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        sentence_transformers_version=_package_version("sentence-transformers"),
        accelerate_version=_package_version("accelerate"),
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        gpu_name=torch.cuda.get_device_name(0) if cuda_available else None,
        gpu_total_vram_mb=(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            if cuda_available
            else None
        ),
        chunk_count=len(chunks),
        chunk_index_sha256=_sha256(chunk_index_path),
        dataset_sha256=_sha256(dataset_path),
        dense_embedding_model=QWEN_EMBEDDING_MODEL_ID,
        dense_embedding_revision=settings.embedding_model_revision,
        reranker_model=BGE_RERANKER_MODEL_ID,
        reranker_model_revision=reranker_revision,
        qdrant_collection=collection_name,
        qdrant_point_count=qdrant_point_count,
        qdrant_vector_dimension=int(qdrant_info.get("vector_dimension") or 0),
        qdrant_distance=str(qdrant_info.get("distance") or ""),
        qdrant_full_scan=(
            bool(qdrant_info["full_scan"])
            if isinstance(qdrant_info.get("full_scan"), bool)
            else None
        ),
    )


def _result(
    *,
    system: SystemName,
    split: SplitName,
    cases: Sequence[FrozenEvalCase],
    chunks: Sequence[IndexedChunk],
    metrics: Mapping[str, float],
    dense_revision: str,
    model_load_time: float | None,
    corpus_encoding_time: float | None,
    retrieval_timings: Sequence[float],
    reranking_timings: Sequence[float] = (),
    total_timings: Sequence[float] = (),
    peak_memory: tuple[float | None, float | None] = (None, None),
    reranker: _BgeReranker | None = None,
    reranker_load_time: float | None = None,
    reranker_load_peak: tuple[float | None, float | None] = (None, None),
    truncated_pair_count: int | None = None,
    candidate_pair_count: int | None = None,
    truncated_gold_items: int | None = None,
    ranking_file: str | None = None,
) -> RetrievalSystemBenchmarkResult:
    def _rate(count: int, elapsed: float) -> float | None:
        return count / elapsed if elapsed else None

    retrieval_elapsed = sum(retrieval_timings)
    reranking_elapsed = sum(reranking_timings)
    total_elapsed = sum(total_timings) if total_timings else retrieval_elapsed
    return RetrievalSystemBenchmarkResult(
        system=system,
        split=split,
        status="success",
        corpus_count=len(chunks),
        query_count=len(cases),
        candidate_pool_size=DEFAULT_DENSE_TOP_K,
        dense_model_id=QWEN_EMBEDDING_MODEL_ID,
        dense_model_revision=dense_revision,
        reranker_model_id=BGE_RERANKER_MODEL_ID if reranker else None,
        reranker_model_revision=reranker.model_revision if reranker else None,
        reranker_actual_dtype=reranker.actual_dtype if reranker else None,
        metrics=dict(metrics),
        model_load_time_seconds=model_load_time,
        corpus_encoding_time_seconds=corpus_encoding_time,
        retrieval_latency_mean_ms=statistics.fmean(retrieval_timings) * 1000
        if retrieval_timings
        else None,
        retrieval_latency_p50_ms=_percentile(retrieval_timings, 0.50),
        retrieval_latency_p95_ms=_percentile(retrieval_timings, 0.95),
        reranking_latency_mean_ms=statistics.fmean(reranking_timings) * 1000
        if reranking_timings
        else None,
        reranking_latency_p50_ms=_percentile(reranking_timings, 0.50),
        reranking_latency_p95_ms=_percentile(reranking_timings, 0.95),
        total_query_latency_mean_ms=statistics.fmean(total_timings) * 1000
        if total_timings
        else statistics.fmean(retrieval_timings) * 1000
        if retrieval_timings
        else None,
        total_query_latency_p50_ms=_percentile(total_timings or retrieval_timings, 0.50),
        total_query_latency_p95_ms=_percentile(total_timings or retrieval_timings, 0.95),
        retrieval_throughput_per_second=_rate(len(retrieval_timings), retrieval_elapsed),
        reranking_throughput_per_second=_rate(len(reranking_timings), reranking_elapsed)
        if reranking_timings
        else None,
        total_throughput_per_second=_rate(len(cases), total_elapsed),
        peak_cuda_allocated_mb=peak_memory[0],
        peak_cuda_reserved_mb=peak_memory[1],
        reranker_load_time_seconds=reranker_load_time,
        reranker_load_peak_cuda_allocated_mb=reranker_load_peak[0] if reranker else None,
        reranker_load_peak_cuda_reserved_mb=reranker_load_peak[1] if reranker else None,
        reranker_truncated_candidate_pairs=truncated_pair_count,
        reranker_candidate_pair_count=candidate_pair_count,
        reranker_truncated_gold_items=truncated_gold_items,
        ranking_file=ranking_file,
    )


def run_retrieval_system_benchmark(
    *,
    split: RunSplit = "both",
    protocol: RetrievalSystemBenchmarkProtocol | None = None,
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    output_dir: Path = DEFAULT_OUTPUT,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run exact dense, Qdrant dense and Qdrant+BGE Top-10 on frozen inputs."""
    if split not in SUPPORTED_SPLITS:
        raise RetrievalSystemBenchmarkError(f"unsupported benchmark split: {split}")
    selected_settings = settings or get_settings()
    protocol = protocol or RetrievalSystemBenchmarkProtocol(
        dense_batch_size=selected_settings.embedding_batch_size,
        dense_max_seq_length=selected_settings.embedding_max_seq_length,
        dtype=selected_settings.embedding_dtype,
        device=selected_settings.embedding_device,
    )
    chunks_by_id, cases_by_split = _load_cases(
        split,
        chunk_index_path,
        dataset_path,
        materialized_path,
    )
    chunks = [chunks_by_id[chunk_id] for chunk_id in sorted(chunks_by_id)]
    cases = [case for split_cases in cases_by_split.values() for case in split_cases]
    if not cases:
        raise RetrievalSystemBenchmarkError("no frozen evaluation cases selected")

    store = QdrantVectorStore(selected_settings)
    try:
        readiness = store.readiness()
        if readiness.status != "ready":
            raise RetrievalSystemBenchmarkError(
                readiness.error or "Qdrant collection is not ready; run qdrant create first"
            )
        qdrant_count, qdrant_info = _qdrant_chunk_ids(store, set(chunks_by_id))
        if qdrant_info.get("vector_dimension") != QDRANT_VECTOR_DIMENSION:
            raise RetrievalSystemBenchmarkError("Qdrant vector dimension is not 1024")
        if qdrant_info.get("distance") != "cosine":
            raise RetrievalSystemBenchmarkError("Qdrant collection distance is not cosine")

        output_dir.mkdir(parents=True, exist_ok=True)
        ranking_dir = output_dir / "rankings"
        ranking_dir.mkdir(parents=True, exist_ok=True)
        torch = _torch_runtime()
        if protocol.device.startswith("cuda") and not torch.cuda.is_available():
            raise RetrievalSystemBenchmarkError("CUDA was requested but is unavailable")

        embedding_model = QwenEmbeddingModel(selected_settings)
        load_started = time.perf_counter()
        embedding_model._load()
        _synchronize(torch, protocol.device)
        model_load_time = time.perf_counter() - load_started
        dense_revision = embedding_model.model_revision

        passage_texts = [_passage_text(chunk) for chunk in chunks]
        _reset_peak_memory(torch, protocol.device)
        corpus_started = time.perf_counter()
        corpus_vectors = embedding_model.encode_documents(passage_texts)
        _synchronize(torch, protocol.device)
        corpus_encoding_time = time.perf_counter() - corpus_started
        if len(corpus_vectors) != len(chunks) or any(
            len(vector) != QDRANT_VECTOR_DIMENSION for vector in corpus_vectors
        ):
            raise RetrievalSystemBenchmarkError("Qwen corpus vectors do not match the frozen index")
        corpus_tensor = torch.as_tensor(corpus_vectors, dtype=torch.float32, device=protocol.device)
        corpus_tensor = torch.nn.functional.normalize(corpus_tensor, p=2, dim=1)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_source_blocks = _source_block_map(chunks)

        exact_rankings: dict[str, tuple[str, ...]] = {}
        exact_timings_by_split: dict[SplitName, list[float]] = {
            split_name: [] for split_name in cases_by_split
        }
        exact_metric_rows: dict[SplitName, list[dict[str, float]]] = {
            split_name: [] for split_name in cases_by_split
        }
        if protocol.warmup_queries:
            embedding_model.encode_query(cases[0].item.query)
            _synchronize(torch, protocol.device)
        _reset_peak_memory(torch, protocol.device)
        case_split = {
            case.item.item_id: split_name
            for split_name, split_cases in cases_by_split.items()
            for case in split_cases
        }
        for case in cases:
            started = time.perf_counter()
            query_vector = torch.as_tensor(
                embedding_model.encode_query(case.item.query),
                dtype=torch.float32,
                device=protocol.device,
            )
            query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=0)
            scores = corpus_tensor @ query_vector
            score_values = scores.detach().cpu().tolist()
            order = sorted(
                range(len(chunk_ids)),
                key=lambda index: (-float(score_values[index]), chunk_ids[index]),
            )[: protocol.dense_top_k]
            _synchronize(torch, protocol.device)
            exact_timings_by_split[case_split[case.item.item_id]].append(
                time.perf_counter() - started
            )
            ranking = tuple(chunk_ids[index] for index in order)
            exact_rankings[case.item.item_id] = ranking
            exact_metric_rows[case_split[case.item.item_id]].append(
                ranking_metrics(
                    ranking,
                    case.relevant_chunk_ids,
                    chunk_source_blocks,
                    case.gold_source_blocks,
                )
            )
        exact_peak = _memory_snapshot(torch, protocol.device)

        exact_result_by_split: dict[SplitName, RetrievalSystemBenchmarkResult] = {}
        for split_name, split_cases in cases_by_split.items():
            path = ranking_dir / f"exact-dense-{split_name}-top10.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "item_id": case.item.item_id,
                        "retrieved_chunk_ids": list(exact_rankings[case.item.item_id]),
                    }
                    for case in split_cases
                ],
            )
            exact_result_by_split[split_name] = _result(
                system="exact_dense",
                split=split_name,
                cases=split_cases,
                chunks=chunks,
                metrics=_aggregate_metrics(exact_metric_rows[split_name]),
                dense_revision=dense_revision,
                model_load_time=model_load_time,
                corpus_encoding_time=corpus_encoding_time,
                retrieval_timings=exact_timings_by_split[split_name],
                total_timings=exact_timings_by_split[split_name],
                peak_memory=exact_peak,
                ranking_file=str(path),
            )

        qdrant_rankings: dict[str, tuple[str, ...]] = {}
        qdrant_timings_by_split: dict[SplitName, list[float]] = {
            split_name: [] for split_name in cases_by_split
        }
        qdrant_metric_rows: dict[SplitName, list[dict[str, float]]] = {
            split_name: [] for split_name in cases_by_split
        }
        service = DenseRetrievalService(store, embedding_model)
        if protocol.warmup_queries:
            service.search(cases[0].item.query, limit=protocol.dense_top_k)
            _synchronize(torch, protocol.device)
        _reset_peak_memory(torch, protocol.device)
        for case in cases:
            split_name = case_split[case.item.item_id]
            started = time.perf_counter()
            result = service.search(case.item.query, limit=protocol.dense_top_k)
            _synchronize(torch, protocol.device)
            elapsed = time.perf_counter() - started
            dense_ids = _validate_retrieved_items(result.items, chunks_by_id)
            qdrant_rankings[case.item.item_id] = dense_ids
            qdrant_timings_by_split[split_name].append(elapsed)
            qdrant_metric_rows[split_name].append(
                ranking_metrics(
                    dense_ids,
                    case.relevant_chunk_ids,
                    chunk_source_blocks,
                    case.gold_source_blocks,
                )
            )
        qdrant_peak = _memory_snapshot(torch, protocol.device)

        reranker = _BgeReranker(protocol)
        reranker_load_started = time.perf_counter()
        _reset_peak_memory(torch, protocol.device)
        reranker.load()
        _synchronize(torch, protocol.device)
        reranker_load_time = time.perf_counter() - reranker_load_started
        reranker_load_peak = _memory_snapshot(torch, protocol.device)
        if protocol.warmup_queries:
            reranker.score(cases[0].item.query, [passage_texts[0]])
            _synchronize(torch, protocol.device)

        reranked_rankings: dict[str, tuple[str, ...]] = {}
        reranking_timings_by_split: dict[SplitName, list[float]] = {
            split_name: [] for split_name in cases_by_split
        }
        rerank_retrieval_timings_by_split: dict[SplitName, list[float]] = {
            split_name: [] for split_name in cases_by_split
        }
        total_timings_by_split: dict[SplitName, list[float]] = {
            split_name: [] for split_name in cases_by_split
        }
        reranked_metric_rows: dict[SplitName, list[dict[str, float]]] = {
            split_name: [] for split_name in cases_by_split
        }
        truncated_by_item: dict[str, set[str]] = {}
        _reset_peak_memory(torch, protocol.device)
        for case in cases:
            split_name = case_split[case.item.item_id]
            total_started = time.perf_counter()
            result = service.search(case.item.query, limit=protocol.dense_top_k)
            _synchronize(torch, protocol.device)
            rerank_retrieval_timings_by_split[split_name].append(
                time.perf_counter() - total_started
            )
            qdrant_items = result.items
            dense_ids = _validate_retrieved_items(qdrant_items, chunks_by_id)
            if dense_ids != qdrant_rankings[case.item.item_id]:
                raise RetrievalSystemBenchmarkError(
                    "Qdrant candidate list changed between dense and rerank measurements"
                )

            passages = [_passage_text(chunks_by_id[chunk_id]) for chunk_id in dense_ids]
            rerank_started = time.perf_counter()
            scores = reranker.score(case.item.query, passages)
            _synchronize(torch, protocol.device)
            rerank_elapsed = time.perf_counter() - rerank_started
            reranking_timings_by_split[split_name].append(rerank_elapsed)
            order = sorted(range(len(dense_ids)), key=lambda index: (-scores[index], index))
            reranked_ids = tuple(dense_ids[index] for index in order)
            reranked_rankings[case.item.item_id] = reranked_ids
            # Keep the total query timing limited to the retrieval request and
            # reranker.  Tokenizer inspection below is benchmark-only audit
            # work and must not inflate the production-path latency.
            total_timings_by_split[split_name].append(time.perf_counter() - total_started)
            reranked_metric_rows[split_name].append(
                ranking_metrics(
                    reranked_ids,
                    case.relevant_chunk_ids,
                    chunk_source_blocks,
                    case.gold_source_blocks,
                )
            )
            pair_flags = reranker.truncated_pairs(
                [(case.item.query, passage) for passage in passages]
            )
            if pair_flags is not None:
                truncated_by_item[case.item.item_id] = {
                    chunk_id for chunk_id, flag in zip(dense_ids, pair_flags, strict=True) if flag
                }

        combined_peak = _memory_snapshot(torch, protocol.device)

        qdrant_result_by_split: dict[SplitName, RetrievalSystemBenchmarkResult] = {}
        reranked_result_by_split: dict[SplitName, RetrievalSystemBenchmarkResult] = {}
        comparisons: list[QdrantRankingComparison] = []
        filter_stats: dict[SplitName, tuple[int, int, int]] = {}
        for split_name, split_cases in cases_by_split.items():
            filter_empty = 0
            filter_violations = 0
            for case in split_cases:
                document_id = case.item.evidence[0].document_id
                query_vector = embedding_model.encode_query(case.item.query)
                filtered = store.search(
                    query_vector,
                    limit=protocol.dense_top_k,
                    document_id=document_id,
                )
                if not filtered:
                    filter_empty += 1
                if any(item.payload.document_id != document_id for item in filtered):
                    filter_violations += 1
            filter_stats[split_name] = (len(split_cases), filter_empty, filter_violations)
            exact_metrics = _aggregate_metrics(exact_metric_rows[split_name])
            qdrant_metrics = _aggregate_metrics(qdrant_metric_rows[split_name])
            ann_mode: Literal["full_scan", "ann", "unknown"] = (
                "full_scan"
                if qdrant_info.get("full_scan") is True
                else "ann"
                if qdrant_info.get("full_scan") is False
                else "unknown"
            )
            comparisons.append(
                compare_dense_and_qdrant_rankings(
                    split_name,
                    {case.item.item_id: exact_rankings[case.item.item_id] for case in split_cases},
                    {case.item.item_id: qdrant_rankings[case.item.item_id] for case in split_cases},
                    exact_metrics,
                    qdrant_metrics,
                    ann_mode=ann_mode,
                    document_filter_queries=filter_stats[split_name][0],
                    document_filter_empty_count=filter_stats[split_name][1],
                    document_filter_violations=filter_stats[split_name][2],
                )
            )
            exact_path = ranking_dir / f"qdrant-dense-{split_name}-top10.jsonl"
            _write_jsonl(
                exact_path,
                [
                    {
                        "item_id": case.item.item_id,
                        "retrieved_chunk_ids": list(qdrant_rankings[case.item.item_id]),
                    }
                    for case in split_cases
                ],
            )
            qdrant_result_by_split[split_name] = _result(
                system="qdrant_dense",
                split=split_name,
                cases=split_cases,
                chunks=chunks,
                metrics=qdrant_metrics,
                dense_revision=dense_revision,
                model_load_time=model_load_time,
                corpus_encoding_time=corpus_encoding_time,
                retrieval_timings=qdrant_timings_by_split[split_name],
                total_timings=qdrant_timings_by_split[split_name],
                peak_memory=qdrant_peak,
                ranking_file=str(exact_path),
            )
            rerank_path = ranking_dir / f"qdrant-dense-bge-top10-{split_name}.jsonl"
            _write_jsonl(
                rerank_path,
                [
                    {
                        "item_id": case.item.item_id,
                        "dense_candidate_chunk_ids": list(qdrant_rankings[case.item.item_id]),
                        "reranked_chunk_ids": list(reranked_rankings[case.item.item_id]),
                    }
                    for case in split_cases
                ],
            )
            truncated_pairs = sum(
                len(truncated_by_item.get(case.item.item_id, ())) for case in split_cases
            )
            truncated_gold_items = sum(
                bool(set(case.relevant_chunk_ids) & truncated_by_item.get(case.item.item_id, set()))
                for case in split_cases
            )
            reranked_result_by_split[split_name] = _result(
                system="qdrant_dense_bge_top10",
                split=split_name,
                cases=split_cases,
                chunks=chunks,
                metrics=_aggregate_metrics(reranked_metric_rows[split_name]),
                dense_revision=dense_revision,
                model_load_time=model_load_time,
                corpus_encoding_time=corpus_encoding_time,
                retrieval_timings=rerank_retrieval_timings_by_split[split_name],
                reranking_timings=reranking_timings_by_split[split_name],
                total_timings=total_timings_by_split[split_name],
                peak_memory=combined_peak,
                reranker=reranker,
                reranker_load_time=reranker_load_time,
                reranker_load_peak=reranker_load_peak,
                truncated_pair_count=truncated_pairs,
                candidate_pair_count=len(split_cases) * protocol.dense_top_k,
                truncated_gold_items=truncated_gold_items,
                ranking_file=str(rerank_path),
            )

        bad_cases: list[BadCase] = []
        for case in cases_by_split.get("test", []):
            bad_case = classify_bad_case(
                case,
                qdrant_rankings[case.item.item_id],
                reranked_rankings[case.item.item_id],
                chunks_by_id,
                truncated_chunk_ids=truncated_by_item.get(case.item.item_id, set()),
            )
            dense_hit = bool(set(case.relevant_chunk_ids) & set(qdrant_rankings[case.item.item_id]))
            rerank_hit = bool(
                set(case.relevant_chunk_ids) & set(reranked_rankings[case.item.item_id])
            )
            needs_review = (
                not rerank_hit
                or not dense_hit
                or (
                    bad_case is not None and "reranker_ranked_relevant_lower" in bad_case.categories
                )
            )
            if bad_case is not None and needs_review:
                bad_cases.append(bad_case)

        results = [
            result.model_dump(mode="json")
            for result in [
                *exact_result_by_split.values(),
                *qdrant_result_by_split.values(),
                *reranked_result_by_split.values(),
            ]
        ]
        results_path = output_dir / "results.jsonl"
        _write_jsonl(results_path, results)
        comparisons_path = output_dir / "qdrant-comparison.json"
        comparisons_path.write_text(
            json.dumps(
                [comparison.model_dump(mode="json") for comparison in comparisons],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        bad_cases_path = output_dir / "bad-cases.jsonl"
        _write_jsonl(bad_cases_path, [case.model_dump(mode="json") for case in bad_cases])
        environment = _environment(
            torch,
            chunks,
            chunk_index_path,
            dataset_path,
            selected_settings,
            store.collection_name,
            qdrant_count,
            qdrant_info,
            reranker.model_revision,
        )
        manifest = {
            "schema_version": RETRIEVAL_SYSTEM_BENCHMARK_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "protocol": protocol.model_dump(mode="json"),
            "environment": environment.model_dump(mode="json"),
            "systems": [result["system"] for result in results],
            "results_file": str(results_path),
            "qdrant_comparison_file": str(comparisons_path),
            "bad_cases_file": str(bad_cases_path),
            "qdrant_comparisons": [
                comparison.model_dump(mode="json") for comparison in comparisons
            ],
            "bad_case_counts": dict(
                sorted(
                    Counter(category for case in bad_cases for category in case.categories).items()
                )
            ),
            "candidate_pool_limitation": (
                "BGE reranking only changes the Qdrant dense Top-10 order; it cannot recover "
                "gold evidence absent from that candidate pool."
            ),
            "metric_semantics": (
                "A2.1 ranking_metrics; MRR and Hit/EvidenceRecall use the supplied Top-10 list"
            ),
            "qdrant_consistency": (
                "The benchmark checks the local collection before querying; no ANN or "
                "database transaction is claimed."
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "manifest": manifest,
            "results": results,
            "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
            "bad_cases": [case.model_dump(mode="json") for case in bad_cases],
        }
    finally:
        store.close()
        gc.collect()


__all__ = [
    "BGE_RERANKER_MODEL_ID",
    "BGE_RERANKER_MODEL_REVISION",
    "DEFAULT_CHUNK_INDEX",
    "DEFAULT_DATASET",
    "DEFAULT_DENSE_TOP_K",
    "DEFAULT_MATERIALIZED",
    "DEFAULT_OUTPUT",
    "BadCase",
    "QdrantRankingComparison",
    "RetrievalSystemBenchmarkEnvironment",
    "RetrievalSystemBenchmarkError",
    "RetrievalSystemBenchmarkProtocol",
    "RetrievalSystemBenchmarkResult",
    "classify_bad_case",
    "compare_dense_and_qdrant_rankings",
    "run_retrieval_system_benchmark",
]
