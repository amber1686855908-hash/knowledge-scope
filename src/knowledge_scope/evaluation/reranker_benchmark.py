"""Benchmark local rerankers on the frozen A2.1 retrieval evaluation set."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from knowledge_scope.retrieval.qdrant import QWEN_EMBEDDING_MODEL_ID
from knowledge_scope.retrieval.reranking import (
    RERANKER_MODEL_KEYS,
    RERANKER_MODEL_SPECS,
    LocalCrossEncoderReranker,
)
from knowledge_scope.shared.config import Settings, get_settings

RERANKER_BENCHMARK_SCHEMA_VERSION = "1.0"
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_MATERIALIZED = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/a2-4")
CANDIDATE_SIZES = (10, 20, 50)
SUPPORTED_SPLITS = ("dev", "test", "both")

SplitName = Literal["dev", "test"]
RunSplit = Literal["dev", "test", "both"]
RunStatus = Literal["success", "failed"]


class RerankerBenchmarkError(RuntimeError):
    """Raised when frozen inputs or the local benchmark runtime is invalid."""


class _BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RerankerBenchmarkProtocol(_BenchmarkModel):
    """The same inference profile is applied to every reranker candidate."""

    batch_size: StrictInt = Field(default=4, ge=1)
    max_seq_length: StrictInt = Field(default=512, ge=1)
    dense_max_seq_length: StrictInt = Field(default=512, ge=1)
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    device: str = Field(default="cuda", min_length=1)
    candidate_sizes: tuple[StrictInt, ...] = CANDIDATE_SIZES
    warmup_pairs: StrictInt = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_candidate_sizes(self) -> Self:
        ordered = tuple(self.candidate_sizes)
        if not ordered or ordered != tuple(sorted(set(ordered))):
            raise ValueError("candidate_sizes must be a non-empty sorted unique sequence")
        if any(size > 100 for size in ordered):
            raise ValueError("candidate_sizes must not exceed 100")
        return self


class RerankerBenchmarkEnvironment(_BenchmarkModel):
    """Runtime and frozen-input identity captured alongside benchmark results."""

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


class DenseBaselineResult(_BenchmarkModel):
    """Dense-only metrics for the exact candidate pool used by reranking."""

    schema_version: Literal["1.0"] = RERANKER_BENCHMARK_SCHEMA_VERSION
    split: SplitName
    status: RunStatus
    model_id: str
    model_revision: str
    corpus_count: StrictInt
    query_count: StrictInt
    candidate_pool_size: StrictInt
    metrics: dict[str, float] = Field(default_factory=dict)
    ranking_file: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class RerankerBenchmarkResult(_BenchmarkModel):
    """One reranker/candidate-size/split measurement, including failures."""

    schema_version: Literal["1.0"] = RERANKER_BENCHMARK_SCHEMA_VERSION
    model_key: str
    model_id: str
    split: SplitName
    candidate_size: StrictInt
    actual_candidate_count: StrictInt
    status: RunStatus
    batch_size: StrictInt
    max_seq_length: StrictInt
    requested_dtype: str
    actual_dtype: str | None = None
    device: str
    prompt_mode: str
    prompt_detail: str
    model_revision: str | None = None
    corpus_count: StrictInt
    query_count: StrictInt
    pair_count: StrictInt
    model_load_time_seconds: float | None = None
    model_load_peak_cuda_allocated_mb: float | None = None
    model_load_peak_cuda_reserved_mb: float | None = None
    rerank_time_seconds: float | None = None
    reranker_pairs_per_second: float | None = None
    reranker_queries_per_second: float | None = None
    reranker_latency_mean_ms: float | None = None
    reranker_latency_p50_ms: float | None = None
    reranker_latency_p95_ms: float | None = None
    peak_cuda_allocated_mb: float | None = None
    peak_cuda_reserved_mb: float | None = None
    dense_candidate_metrics: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    ranking_file: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    oom: bool = False


@dataclass(frozen=True, slots=True)
class _DenseRun:
    rankings: dict[str, tuple[str, ...]]
    baseline_results: tuple[DenseBaselineResult, ...]
    load_time_seconds: float
    corpus_encoding_time_seconds: float
    query_time_seconds: float
    query_latency_mean_ms: float
    query_throughput_per_second: float
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None
    model_revision: str


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
        raise RerankerBenchmarkError(f"benchmark input is not readable: {path}") from error
    return digest.hexdigest()


def _torch_dtype(torch: Any, dtype: str) -> Any:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]


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


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[position]


def _import_embedding_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RerankerBenchmarkError(
            "benchmark dependencies are missing; run uv sync --group reranker-benchmark"
        ) from error
    return torch, SentenceTransformer


def _encode_dense(
    model: Any,
    torch: Any,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
) -> Any:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "normalize_embeddings": True,
    }
    if is_query:
        kwargs["prompt_name"] = "query"
    encoded = model.encode(list(texts), **kwargs)
    if not torch.is_tensor(encoded):
        encoded = torch.as_tensor(encoded)
    return torch.nn.functional.normalize(encoded.float(), p=2, dim=1)


def _source_block_map(chunks: Sequence[IndexedChunk]) -> dict[str, set[SourceBlockKey]]:
    return {
        chunk.chunk_id: {(str(chunk.document_id), block_id) for block_id in chunk.source_block_ids}
        for chunk in chunks
    }


def _aggregate_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {
        name: statistics.fmean(row[name] for row in rows)
        for name in names
        if all(name in row for row in rows)
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _passage_for_dense_benchmark(chunk: IndexedChunk) -> str:
    if chunk.text.strip():
        return chunk.text
    section = " / ".join(chunk.section_path)
    content_types = ", ".join(chunk.content_types)
    return " ".join(part for part in (section, f"[{content_types}]") if part)


def _build_dense_rankings(
    cases: Sequence[FrozenEvalCase],
    case_split_by_id: Mapping[str, SplitName],
    chunks: Sequence[IndexedChunk],
    protocol: RerankerBenchmarkProtocol,
    settings: Settings,
    output_dir: Path,
) -> _DenseRun:
    torch, sentence_transformer = _import_embedding_runtime()
    if protocol.device.startswith("cuda") and not torch.cuda.is_available():
        raise RerankerBenchmarkError("CUDA was requested but is not available")

    started = time.perf_counter()
    _reset_peak_memory(torch, protocol.device)
    model_kwargs: dict[str, Any] = {}
    if protocol.device.startswith("cuda"):
        model_kwargs["torch_dtype"] = _torch_dtype(torch, protocol.dtype)
    try:
        model = sentence_transformer(
            QWEN_EMBEDDING_MODEL_ID,
            device=protocol.device,
            revision=settings.embedding_model_revision,
            model_kwargs=model_kwargs,
            processor_kwargs={"padding_side": "left"},
        )
        model.max_seq_length = protocol.dense_max_seq_length
        _synchronize(torch, protocol.device)
    except Exception as error:
        raise RerankerBenchmarkError(
            f"dense baseline model could not be loaded: {QWEN_EMBEDDING_MODEL_ID}"
        ) from error
    load_time = time.perf_counter() - started
    model_revision = _model_revision(model) or settings.embedding_model_revision

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_texts = [_passage_for_dense_benchmark(chunk) for chunk in chunks]
    _synchronize(torch, protocol.device)
    corpus_started = time.perf_counter()
    corpus_embeddings = _encode_dense(
        model,
        torch,
        chunk_texts,
        is_query=False,
        batch_size=protocol.batch_size,
    )
    _synchronize(torch, protocol.device)
    corpus_elapsed = time.perf_counter() - corpus_started
    if corpus_embeddings.shape[0] != len(chunks):
        raise RerankerBenchmarkError("dense baseline returned an invalid corpus count")

    max_candidate_size = max(protocol.candidate_sizes)
    rankings: dict[str, tuple[str, ...]] = {}
    query_timings: list[float] = []
    chunk_source_blocks = _source_block_map(chunks)
    rows_by_split: dict[str, list[Mapping[str, object]]] = {"dev": [], "test": []}
    metric_rows_by_split: dict[str, list[dict[str, float]]] = {"dev": [], "test": []}
    for case in cases:
        query_started = time.perf_counter()
        query_embedding = _encode_dense(
            model,
            torch,
            [case.item.query],
            is_query=True,
            batch_size=1,
        )
        scores = query_embedding @ corpus_embeddings.T
        score_values = scores[0].detach().float().cpu().tolist()
        ordered_indices = sorted(
            range(len(chunk_ids)),
            key=lambda index: (-float(score_values[index]), chunk_ids[index]),
        )[: min(max_candidate_size, len(chunk_ids))]
        _synchronize(torch, protocol.device)
        query_timings.append(time.perf_counter() - query_started)
        retrieved = tuple(chunk_ids[index] for index in ordered_indices)
        rankings[case.item.item_id] = retrieved
        metrics = ranking_metrics(
            retrieved,
            case.relevant_chunk_ids,
            chunk_source_blocks,
            case.gold_source_blocks,
        )
        split = case_split_by_id[case.item.item_id]
        metric_rows_by_split[split].append(metrics)
        rows_by_split[split].append(
            {
                "item_id": case.item.item_id,
                "retrieved_chunk_ids": list(retrieved),
            }
        )

    ranking_dir = output_dir / "dense-rankings"
    baseline_results: list[DenseBaselineResult] = []
    for split in ("dev", "test"):
        split_cases = [case for case in cases if case_split_by_id[case.item.item_id] == split]
        if not split_cases:
            continue
        ranking_path = ranking_dir / f"dense-{split}.jsonl"
        _write_jsonl(ranking_path, rows_by_split[split])
        baseline_results.append(
            DenseBaselineResult(
                split=split,
                status="success",
                model_id=QWEN_EMBEDDING_MODEL_ID,
                model_revision=model_revision,
                corpus_count=len(chunks),
                query_count=len(split_cases),
                candidate_pool_size=max_candidate_size,
                metrics=_aggregate_metrics(metric_rows_by_split[split]),
                ranking_file=str(ranking_path),
            )
        )

    allocated, reserved = _memory_snapshot(torch, protocol.device)
    query_elapsed = sum(query_timings)
    del model, corpus_embeddings
    gc.collect()
    if protocol.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return _DenseRun(
        rankings=rankings,
        baseline_results=tuple(baseline_results),
        load_time_seconds=load_time,
        corpus_encoding_time_seconds=corpus_elapsed,
        query_time_seconds=query_elapsed,
        query_latency_mean_ms=statistics.fmean(query_timings) * 1000 if query_timings else 0.0,
        query_throughput_per_second=len(query_timings) / query_elapsed if query_elapsed else 0.0,
        peak_cuda_allocated_mb=allocated,
        peak_cuda_reserved_mb=reserved,
        model_revision=model_revision,
    )


def _model_revision(model: Any) -> str | None:
    loaded_model = getattr(model, "model", model)
    config = getattr(loaded_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None


def _is_oom(error: BaseException, torch: Any) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (isinstance(oom_type, type) and isinstance(error, oom_type)) or (
        "out of memory" in str(error).casefold()
    )


def _error_message(error: BaseException) -> str:
    return str(error).strip().replace("\n", " ")[:1000] or type(error).__name__


def _failure_result(
    model_key: str,
    split: SplitName,
    candidate_size: int,
    protocol: RerankerBenchmarkProtocol,
    *,
    corpus_count: int,
    query_count: int,
    load_time: float | None,
    model: LocalCrossEncoderReranker,
    error: BaseException,
    torch: Any,
    actual_candidate_count: int = 0,
) -> RerankerBenchmarkResult:
    return RerankerBenchmarkResult(
        model_key=model_key,
        model_id=model.model_id,
        split=split,
        candidate_size=candidate_size,
        actual_candidate_count=actual_candidate_count,
        status="failed",
        batch_size=protocol.batch_size,
        max_seq_length=protocol.max_seq_length,
        requested_dtype=protocol.dtype,
        actual_dtype=model.actual_dtype,
        device=protocol.device,
        prompt_mode=model.spec.prompt_mode,
        prompt_detail=model.prompt_detail,
        model_revision=model.model_revision,
        corpus_count=corpus_count,
        query_count=query_count,
        pair_count=0,
        model_load_time_seconds=load_time,
        error_type=type(error).__name__,
        error_message=_error_message(error),
        oom=_is_oom(error, torch),
    )


def _benchmark_candidate_size(
    model: LocalCrossEncoderReranker,
    model_key: str,
    split: SplitName,
    cases: Sequence[FrozenEvalCase],
    dense_rankings: Mapping[str, tuple[str, ...]],
    chunks: Mapping[str, IndexedChunk],
    chunk_source_blocks: Mapping[str, set[SourceBlockKey]],
    candidate_size: int,
    protocol: RerankerBenchmarkProtocol,
    output_dir: Path,
    torch: Any,
    load_time: float,
    load_peak: tuple[float | None, float | None],
) -> RerankerBenchmarkResult:
    query_timings: list[float] = []
    metric_rows: list[dict[str, float]] = []
    dense_metric_rows: list[dict[str, float]] = []
    ranking_rows: list[Mapping[str, object]] = []
    pair_count = 0
    actual_candidate_count = 0
    _reset_peak_memory(torch, protocol.device)
    try:
        for case in cases:
            dense_candidates = dense_rankings[case.item.item_id][:candidate_size]
            actual_candidate_count = max(actual_candidate_count, len(dense_candidates))
            passages = [
                _passage_for_dense_benchmark(chunks[chunk_id]) for chunk_id in dense_candidates
            ]
            started = time.perf_counter()
            scores = model.score_pairs(case.item.query, passages)
            _synchronize(torch, protocol.device)
            order = sorted(
                range(len(dense_candidates)),
                key=lambda index: (-scores[index], index),
            )
            elapsed = time.perf_counter() - started
            query_timings.append(elapsed)
            pair_count += len(dense_candidates)
            reranked = [dense_candidates[index] for index in order]
            dense_metrics = ranking_metrics(
                dense_candidates,
                case.relevant_chunk_ids,
                chunk_source_blocks,
                case.gold_source_blocks,
            )
            metrics = ranking_metrics(
                reranked,
                case.relevant_chunk_ids,
                chunk_source_blocks,
                case.gold_source_blocks,
            )
            dense_metric_rows.append(dense_metrics)
            metric_rows.append(metrics)
            ranking_rows.append(
                {
                    "item_id": case.item.item_id,
                    "dense_candidate_chunk_ids": list(dense_candidates),
                    "reranked_chunk_ids": reranked,
                    "reranker_scores": [scores[index] for index in order],
                }
            )
    except Exception as error:
        if protocol.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _failure_result(
            model_key,
            split,
            candidate_size,
            protocol,
            corpus_count=len(chunks),
            query_count=len(cases),
            load_time=load_time,
            model=model,
            error=error,
            torch=torch,
            actual_candidate_count=actual_candidate_count,
        )

    ranking_path = output_dir / "rankings" / f"{model_key}-{split}-top{candidate_size}.jsonl"
    _write_jsonl(ranking_path, ranking_rows)
    rerank_elapsed = sum(query_timings)
    allocated, reserved = _memory_snapshot(torch, protocol.device)
    return RerankerBenchmarkResult(
        model_key=model_key,
        model_id=model.model_id,
        split=split,
        candidate_size=candidate_size,
        actual_candidate_count=actual_candidate_count,
        status="success",
        batch_size=protocol.batch_size,
        max_seq_length=protocol.max_seq_length,
        requested_dtype=protocol.dtype,
        actual_dtype=model.actual_dtype,
        device=protocol.device,
        prompt_mode=model.spec.prompt_mode,
        prompt_detail=model.prompt_detail,
        model_revision=model.model_revision,
        corpus_count=len(chunks),
        query_count=len(cases),
        pair_count=pair_count,
        model_load_time_seconds=load_time,
        model_load_peak_cuda_allocated_mb=load_peak[0],
        model_load_peak_cuda_reserved_mb=load_peak[1],
        rerank_time_seconds=rerank_elapsed,
        reranker_pairs_per_second=pair_count / rerank_elapsed if rerank_elapsed else None,
        reranker_queries_per_second=len(cases) / rerank_elapsed if rerank_elapsed else None,
        reranker_latency_mean_ms=statistics.fmean(query_timings) * 1000 if query_timings else None,
        reranker_latency_p50_ms=_percentile(query_timings, 0.50) * 1000,
        reranker_latency_p95_ms=_percentile(query_timings, 0.95) * 1000,
        peak_cuda_allocated_mb=allocated,
        peak_cuda_reserved_mb=reserved,
        dense_candidate_metrics=_aggregate_metrics(dense_metric_rows),
        metrics=_aggregate_metrics(metric_rows),
        ranking_file=str(ranking_path),
    )


def _environment(
    torch: Any,
    chunks: Sequence[IndexedChunk],
    chunk_index_path: Path,
    dataset_path: Path,
    dense_revision: str,
) -> RerankerBenchmarkEnvironment:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if cuda_available else None
    )
    return RerankerBenchmarkEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        sentence_transformers_version=_package_version("sentence-transformers"),
        accelerate_version=_package_version("accelerate"),
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        gpu_name=gpu_name,
        gpu_total_vram_mb=gpu_memory,
        chunk_count=len(chunks),
        chunk_index_sha256=_sha256(chunk_index_path),
        dataset_sha256=_sha256(dataset_path),
        dense_embedding_model=QWEN_EMBEDDING_MODEL_ID,
        dense_embedding_revision=dense_revision,
    )


def _requested_splits(split: RunSplit) -> tuple[SplitName, ...]:
    if split == "both":
        return ("dev", "test")
    if split in ("dev", "test"):
        return (split,)
    raise RerankerBenchmarkError(f"unsupported benchmark split: {split}")


def run_reranker_benchmark(
    *,
    split: RunSplit = "both",
    model_keys: Sequence[str] = RERANKER_MODEL_KEYS,
    protocol: RerankerBenchmarkProtocol | None = None,
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    output_dir: Path = DEFAULT_OUTPUT,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run dense candidate generation once, then benchmark each local reranker."""
    selected_settings = settings or get_settings()
    protocol = protocol or RerankerBenchmarkProtocol(
        batch_size=selected_settings.reranker_batch_size,
        max_seq_length=selected_settings.reranker_max_seq_length,
        dense_max_seq_length=selected_settings.embedding_max_seq_length,
        dtype=selected_settings.reranker_dtype,
        device=selected_settings.reranker_device,
    )
    unknown = sorted(set(model_keys) - set(RERANKER_MODEL_SPECS))
    if unknown:
        raise RerankerBenchmarkError(f"unknown reranker model keys: {', '.join(unknown)}")
    requested = _requested_splits(split)
    chunks_by_id = load_frozen_chunk_index(chunk_index_path)
    chunks = [chunks_by_id[chunk_id] for chunk_id in sorted(chunks_by_id)]
    cases_by_split = {
        requested_split: load_frozen_eval_cases(
            requested_split,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            chunk_index=chunks_by_id,
        )[0]
        for requested_split in requested
    }
    cases = [case for requested_split in requested for case in cases_by_split[requested_split]]
    case_split_by_id = {
        case.item.item_id: requested_split
        for requested_split in requested
        for case in cases_by_split[requested_split]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dense = _build_dense_rankings(
        cases,
        case_split_by_id,
        chunks,
        protocol,
        selected_settings,
        output_dir,
    )
    torch, _ = _import_embedding_runtime()
    environment = _environment(
        torch,
        chunks,
        chunk_index_path,
        dataset_path,
        dense.model_revision,
    )
    chunk_source_blocks = _source_block_map(chunks)
    results: list[RerankerBenchmarkResult] = []
    for model_key in model_keys:
        spec = RERANKER_MODEL_SPECS[model_key]
        reranker = LocalCrossEncoderReranker(
            spec,
            device=protocol.device,
            dtype=protocol.dtype,
            batch_size=protocol.batch_size,
            max_seq_length=protocol.max_seq_length,
            revision=selected_settings.reranker_model_revision,
        )
        load_started = time.perf_counter()
        try:
            # Reset after the previous model has been released so this snapshot
            # describes this reranker's load phase only.
            _reset_peak_memory(torch, protocol.device)
            reranker.load()
            _synchronize(torch, protocol.device)
            load_time = time.perf_counter() - load_started
            load_peak = _memory_snapshot(torch, protocol.device)
            if protocol.warmup_pairs and cases:
                warmup_case = cases[0]
                warmup_ids = dense.rankings[warmup_case.item.item_id][
                    : min(protocol.warmup_pairs, 1)
                ]
                reranker.score_pairs(
                    warmup_case.item.query,
                    [
                        _passage_for_dense_benchmark(chunks_by_id[chunk_id])
                        for chunk_id in warmup_ids
                    ],
                )
                _synchronize(torch, protocol.device)
            for requested_split in requested:
                for candidate_size in protocol.candidate_sizes:
                    results.append(
                        _benchmark_candidate_size(
                            reranker,
                            model_key,
                            requested_split,
                            cases_by_split[requested_split],
                            dense.rankings,
                            chunks_by_id,
                            chunk_source_blocks,
                            candidate_size,
                            protocol,
                            output_dir,
                            torch,
                            load_time,
                            load_peak,
                        )
                    )
        except Exception as error:
            load_time = time.perf_counter() - load_started
            for requested_split in requested:
                for candidate_size in protocol.candidate_sizes:
                    results.append(
                        _failure_result(
                            model_key,
                            requested_split,
                            candidate_size,
                            protocol,
                            corpus_count=len(chunks),
                            query_count=len(cases_by_split[requested_split]),
                            load_time=load_time,
                            model=reranker,
                            error=error,
                            torch=torch,
                        )
                    )
        finally:
            del reranker
            gc.collect()
            if protocol.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    results_path = output_dir / "results.jsonl"
    _write_jsonl(results_path, [result.model_dump(mode="json") for result in results])
    baseline_path = output_dir / "dense-baseline.jsonl"
    _write_jsonl(
        baseline_path,
        [result.model_dump(mode="json") for result in dense.baseline_results],
    )
    status_counts = Counter(result.status for result in results)
    manifest = {
        "schema_version": RERANKER_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "model_keys": list(model_keys),
        "model_specs": [
            {
                "key": RERANKER_MODEL_SPECS[key].key,
                "model_id": RERANKER_MODEL_SPECS[key].model_id,
                "prompt_mode": RERANKER_MODEL_SPECS[key].prompt_mode,
                "prompt_detail": RERANKER_MODEL_SPECS[key].prompt_detail,
                "official_reference": RERANKER_MODEL_SPECS[key].official_reference,
            }
            for key in model_keys
        ],
        "protocol": protocol.model_dump(mode="json"),
        "environment": environment.model_dump(mode="json"),
        "dense_baseline": {
            "model_id": QWEN_EMBEDDING_MODEL_ID,
            "model_revision": dense.model_revision,
            "candidate_pool_size": max(protocol.candidate_sizes),
            "load_time_seconds": dense.load_time_seconds,
            "corpus_encoding_time_seconds": dense.corpus_encoding_time_seconds,
            "query_time_seconds": dense.query_time_seconds,
            "query_latency_mean_ms": dense.query_latency_mean_ms,
            "query_throughput_per_second": dense.query_throughput_per_second,
            "peak_cuda_allocated_mb": dense.peak_cuda_allocated_mb,
            "peak_cuda_reserved_mb": dense.peak_cuda_reserved_mb,
            "results_file": str(baseline_path),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "results_file": str(results_path),
        "ranking_directory": str(output_dir / "rankings"),
        "candidate_pool_limitation": (
            "Reranking only changes the order of the dense candidate pool; it cannot "
            "recover gold evidence absent from that pool."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "dense_baseline": [result.model_dump(mode="json") for result in dense.baseline_results],
        "results": [result.model_dump(mode="json") for result in results],
    }


__all__ = [
    "CANDIDATE_SIZES",
    "DEFAULT_CHUNK_INDEX",
    "DEFAULT_DATASET",
    "DEFAULT_MATERIALIZED",
    "DEFAULT_OUTPUT",
    "RERANKER_MODEL_KEYS",
    "DenseBaselineResult",
    "RerankerBenchmarkEnvironment",
    "RerankerBenchmarkError",
    "RerankerBenchmarkProtocol",
    "RerankerBenchmarkResult",
    "run_reranker_benchmark",
]
