"""Local dense embedding benchmark for the frozen A2.1 retrieval set.

This module deliberately treats A2.1 as read-only input.  It loads the
existing chunk index and final evaluation annotations, encodes them with a
local model, and writes only ignored runtime measurements and rankings.
"""

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

from knowledge_scope.evaluation.retrieval_eval import (
    FinalDatasetItem,
    IndexedChunk,
    MaterializedEvalItem,
    RetrievalEvalItem,
)
from knowledge_scope.evaluation.retrieval_metrics import SourceBlockKey, ranking_metrics

EMBEDDING_BENCHMARK_SCHEMA_VERSION = "1.0"
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_MATERIALIZED = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/a2-2")
SUPPORTED_SPLITS = ("dev", "test", "both")
FROZEN_A21_SUBJECTS = frozenset(
    {"化学", "历史", "地理", "思想政治", "数学", "物理", "生物", "英语", "语文"}
)
MODEL_KEYS = (
    "qwen3-embedding-0.6b",
    "qwen3-embedding-4b",
    "bge-m3",
    "multilingual-e5-large-instruct",
)
DEFAULT_QUERY_TASK = "Given a textbook question, retrieve passages that contain its answer"

SplitName = Literal["dev", "test"]
RunSplit = Literal["dev", "test", "both"]
RunStatus = Literal["success", "failed"]


class EmbeddingBenchmarkError(RuntimeError):
    """Raised when frozen inputs or the local benchmark runtime is invalid."""


class _BenchmarkModel(BaseModel):
    """Strict persisted benchmark contract."""

    model_config = ConfigDict(extra="forbid")


class EmbeddingBenchmarkProtocol(_BenchmarkModel):
    """Shared final protocol applied to every candidate model."""

    batch_size: StrictInt = Field(default=4, ge=1)
    max_seq_length: StrictInt = Field(default=512, ge=1)
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    device: str = "cuda"
    top_k: tuple[StrictInt, ...] = (1, 3, 5, 10)
    normalization: Literal["l2"] = "l2"
    warmup_queries: StrictInt = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_top_k(self) -> Self:
        if self.top_k != (1, 3, 5, 10):
            raise ValueError("A2.1 metrics require top_k to remain (1, 3, 5, 10)")
        return self


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    """Model identity and its official query-side convention."""

    key: str
    model_id: str
    query_mode: Literal["qwen_prompt_name", "e5_instruction", "none"]
    official_reference: str


MODEL_SPECS: Mapping[str, EmbeddingModelSpec] = {
    "qwen3-embedding-0.6b": EmbeddingModelSpec(
        key="qwen3-embedding-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        query_mode="qwen_prompt_name",
        official_reference=("https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"),
    ),
    "qwen3-embedding-4b": EmbeddingModelSpec(
        key="qwen3-embedding-4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        query_mode="qwen_prompt_name",
        official_reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    ),
    "bge-m3": EmbeddingModelSpec(
        key="bge-m3",
        model_id="BAAI/bge-m3",
        query_mode="none",
        official_reference="https://huggingface.co/BAAI/bge-m3",
    ),
    "multilingual-e5-large-instruct": EmbeddingModelSpec(
        key="multilingual-e5-large-instruct",
        model_id="intfloat/multilingual-e5-large-instruct",
        query_mode="e5_instruction",
        official_reference=("https://huggingface.co/intfloat/multilingual-e5-large-instruct"),
    ),
}


class EmbeddingBenchmarkEnvironment(_BenchmarkModel):
    """Versions and hardware captured alongside every benchmark run."""

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


class EmbeddingBenchmarkResult(_BenchmarkModel):
    """One model/split measurement, including failures without fake metrics."""

    schema_version: Literal["1.0"] = EMBEDDING_BENCHMARK_SCHEMA_VERSION
    model_key: str
    model_id: str
    split: SplitName
    status: RunStatus
    batch_size: StrictInt
    max_seq_length: StrictInt
    requested_dtype: str
    actual_dtype: str | None = None
    device: str
    query_prompt_mode: str
    query_prompt_detail: str
    embedding_dimension: StrictInt | None = None
    model_revision: str | None = None
    corpus_count: StrictInt
    query_count: StrictInt
    load_time_seconds: float | None = None
    corpus_encoding_time_seconds: float | None = None
    corpus_throughput_chunks_per_second: float | None = None
    query_encoding_time_seconds: float | None = None
    query_throughput_per_second: float | None = None
    exact_search_time_seconds: float | None = None
    query_latency_mean_ms: float | None = None
    query_latency_p50_ms: float | None = None
    query_latency_p95_ms: float | None = None
    peak_cuda_allocated_mb: float | None = None
    peak_cuda_reserved_mb: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    ranking_file: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    oom: bool = False


@dataclass(frozen=True, slots=True)
class FrozenEvalCase:
    """A final A2.1 item plus derived runtime relevance."""

    item: RetrievalEvalItem
    relevant_chunk_ids: frozenset[str]
    gold_source_blocks: frozenset[SourceBlockKey]


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
        raise EmbeddingBenchmarkError(f"cannot read benchmark input: {path}") from error
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EmbeddingBenchmarkError(f"benchmark input is not readable: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EmbeddingBenchmarkError(
                f"invalid JSON on line {line_number} of {path}"
            ) from error
        if not isinstance(record, dict):
            raise EmbeddingBenchmarkError(f"line {line_number} of {path} is not an object")
        records.append(record)
    return records


def load_frozen_chunk_index(path: Path = DEFAULT_CHUNK_INDEX) -> dict[str, IndexedChunk]:
    """Load and sanity-check the existing A1.6 chunk index without rewriting it."""
    chunks = [IndexedChunk.model_validate(record) for record in _read_jsonl(path)]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise EmbeddingBenchmarkError("A1.6 chunk index contains duplicate chunk IDs")
    if not chunks:
        raise EmbeddingBenchmarkError("A1.6 chunk index is empty")
    return by_id


def _gold_source_blocks(item: RetrievalEvalItem) -> frozenset[SourceBlockKey]:
    return frozenset(
        (str(location.document_id), block_id)
        for location in item.evidence
        for block_id in location.source_block_ids
    )


def _query_source_blocks(item: RetrievalEvalItem) -> frozenset[SourceBlockKey]:
    return frozenset(
        (str(location.document_id), block_id)
        for location in item.query_source
        for block_id in location.source_block_ids
    )


def _derive_relevant_chunks(
    item: RetrievalEvalItem,
    chunk_index: Mapping[str, IndexedChunk],
) -> frozenset[str]:
    gold = _gold_source_blocks(item)
    query_source = _query_source_blocks(item)
    relevant = {
        chunk_id
        for chunk_id, chunk in chunk_index.items()
        if any((str(chunk.document_id), block_id) in gold for block_id in chunk.source_block_ids)
        and not any(
            (str(chunk.document_id), block_id) in query_source
            for block_id in chunk.source_block_ids
        )
    }
    return frozenset(relevant)


def _validate_evidence_lineage(
    item: RetrievalEvalItem,
    chunks: Mapping[str, IndexedChunk],
) -> None:
    """Validate every frozen evidence reference against the A1.6 chunk index."""

    for location_kind, locations in (
        ("gold evidence", item.evidence),
        ("query source", item.query_source),
    ):
        for location in locations:
            for block_id in location.source_block_ids:
                matching_chunks = [
                    chunk
                    for chunk in chunks.values()
                    if chunk.document_id == location.document_id
                    and block_id in chunk.source_block_ids
                    and chunk.page_start <= location.page_number <= chunk.page_end
                ]
                if not matching_chunks:
                    raise EmbeddingBenchmarkError(
                        f"{location_kind} {location.document_id}/{block_id} is outside "
                        f"the frozen chunk lineage"
                    )


def _materialized_source_keys(
    references: Sequence[Any],
) -> set[tuple[str, int, str]]:
    return {
        (str(reference.document_id), reference.page_number, reference.source_block_id)
        for reference in references
    }


def _validate_materialized_item(
    item: RetrievalEvalItem,
    derived: MaterializedEvalItem,
    chunks: Mapping[str, IndexedChunk],
) -> frozenset[str]:
    """Reject stale or denormalized materialization instead of trusting it."""

    if derived.verification_status != item.verification_status:
        raise EmbeddingBenchmarkError(f"materialized status drift: {item.item_id}")
    if len(derived.relevant_chunk_ids) != len(set(derived.relevant_chunk_ids)):
        raise EmbeddingBenchmarkError(f"materialized chunk IDs are duplicated: {item.item_id}")
    expected_relevant = _derive_relevant_chunks(item, chunks)
    materialized_relevant = frozenset(derived.relevant_chunk_ids)
    if materialized_relevant != expected_relevant:
        raise EmbeddingBenchmarkError(
            f"materialized relevant chunks drift from frozen evidence: {item.item_id}"
        )

    expected_gold_keys = {
        (str(location.document_id), location.page_number, block_id)
        for location in item.evidence
        for block_id in location.source_block_ids
    }
    materialized_gold_keys = _materialized_source_keys(derived.gold_source_blocks)
    if len(materialized_gold_keys) != len(derived.gold_source_blocks):
        raise EmbeddingBenchmarkError(f"materialized gold evidence is duplicated: {item.item_id}")
    if materialized_gold_keys != expected_gold_keys:
        raise EmbeddingBenchmarkError(f"materialized gold evidence drift: {item.item_id}")
    covered_keys = _materialized_source_keys(derived.covered_source_blocks)
    uncovered_keys = _materialized_source_keys(derived.uncovered_source_blocks)
    if len(covered_keys) != len(derived.covered_source_blocks) or len(uncovered_keys) != len(
        derived.uncovered_source_blocks
    ):
        raise EmbeddingBenchmarkError(
            f"materialized evidence coverage is duplicated: {item.item_id}"
        )
    if covered_keys & uncovered_keys or covered_keys | uncovered_keys != expected_gold_keys:
        raise EmbeddingBenchmarkError(
            f"materialized evidence coverage is inconsistent: {item.item_id}"
        )
    if any(chunk_id not in chunks for chunk_id in materialized_relevant):
        raise EmbeddingBenchmarkError(f"item references a missing chunk: {item.item_id}")
    expected_covered = {
        (document_id, page_number, block_id)
        for document_id, page_number, block_id in expected_gold_keys
        if any(
            chunk_id in materialized_relevant
            and document_id == str(chunks[chunk_id].document_id)
            and block_id in chunks[chunk_id].source_block_ids
            and chunks[chunk_id].page_start <= page_number <= chunks[chunk_id].page_end
            for chunk_id in materialized_relevant
        )
    }
    if covered_keys != expected_covered:
        raise EmbeddingBenchmarkError(
            f"materialized covered evidence is inconsistent: {item.item_id}"
        )
    if derived.all_gold_blocks_covered != (not bool(uncovered_keys)):
        raise EmbeddingBenchmarkError(f"materialized coverage flag drift: {item.item_id}")
    return materialized_relevant


def load_frozen_eval_cases(
    split: SplitName,
    *,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    chunk_index: Mapping[str, IndexedChunk] | None = None,
) -> tuple[list[FrozenEvalCase], dict[str, int]]:
    """Load frozen A2.1 items and existing relevant-chunk derivations."""
    chunks = chunk_index or load_frozen_chunk_index()
    records = [FinalDatasetItem.model_validate(record) for record in _read_jsonl(dataset_path)]
    if len(records) != 108:
        raise EmbeddingBenchmarkError(
            f"frozen A2.1 dataset must contain 108 items, found {len(records)}"
        )
    if Counter(record.split for record in records) != {"dev": 72, "test": 36}:
        raise EmbeddingBenchmarkError("frozen A2.1 dataset does not have a 72/36 split")
    if any(record.item.verification_status != "verified" for record in records):
        raise EmbeddingBenchmarkError("frozen A2.1 dataset contains a non-verified item")
    item_ids = [record.item.item_id for record in records]
    if len(item_ids) != len(set(item_ids)):
        raise EmbeddingBenchmarkError("frozen A2.1 dataset contains duplicate item IDs")
    if set(record.item.subject for record in records) != FROZEN_A21_SUBJECTS:
        raise EmbeddingBenchmarkError("frozen A2.1 dataset does not cover the 9 subjects")
    for record in records:
        _validate_evidence_lineage(record.item, chunks)

    materialized: dict[str, MaterializedEvalItem] = {}
    if materialized_path.is_file():
        for raw_record in _read_jsonl(materialized_path):
            value = MaterializedEvalItem.model_validate(raw_record)
            if value.item_id in materialized:
                raise EmbeddingBenchmarkError(
                    f"materialized A2.1 data contains duplicate item ID: {value.item_id}"
                )
            materialized[value.item_id] = value
        if set(materialized) != set(item_ids):
            raise EmbeddingBenchmarkError("materialized A2.1 data does not cover the frozen items")

    selected: list[FrozenEvalCase] = []
    for record in records:
        if record.split != split:
            continue
        item = record.item
        derived = materialized.get(item.item_id)
        relevant = (
            _validate_materialized_item(item, derived, chunks)
            if derived is not None
            else _derive_relevant_chunks(item, chunks)
        )
        if not relevant:
            raise EmbeddingBenchmarkError(f"item has no relevant chunks: {item.item_id}")
        if any(chunk_id not in chunks for chunk_id in relevant):
            raise EmbeddingBenchmarkError(f"item references a missing chunk: {item.item_id}")
        if derived is not None and not derived.all_gold_blocks_covered:
            raise EmbeddingBenchmarkError(f"item gold evidence is not covered: {item.item_id}")
        selected.append(
            FrozenEvalCase(
                item=item,
                relevant_chunk_ids=relevant,
                gold_source_blocks=_gold_source_blocks(item),
            )
        )
    expected_count = 72 if split == "dev" else 36
    if len(selected) != expected_count:
        raise EmbeddingBenchmarkError(
            f"frozen A2.1 {split} split must contain {expected_count} items"
        )
    subject_counts = dict(sorted(Counter(case.item.subject for case in selected).items()))
    if any(count != (8 if split == "dev" else 4) for count in subject_counts.values()):
        raise EmbeddingBenchmarkError(f"frozen A2.1 {split} subject quotas are invalid")
    return selected, subject_counts


def _import_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise EmbeddingBenchmarkError(
            "embedding benchmark dependencies are missing; run uv sync --group embedding-benchmark"
        ) from error
    return torch, SentenceTransformer


def _dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _memory_snapshot(torch: Any, device: str) -> tuple[float | None, float | None]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.max_memory_allocated() / (1024 * 1024),
        torch.cuda.max_memory_reserved() / (1024 * 1024),
    )


def _query_texts(
    spec: EmbeddingModelSpec,
    queries: Sequence[str],
) -> tuple[list[str], str]:
    if spec.query_mode == "e5_instruction":
        return (
            [f"Instruct: {DEFAULT_QUERY_TASK}\nQuery: {query}" for query in queries],
            "manual official E5 template: Instruct + Query",
        )
    if spec.query_mode == "qwen_prompt_name":
        return list(queries), "SentenceTransformers official prompt_name=query"
    return list(queries), "no query instruction (official BGE-M3 convention)"


def _encode(
    model: Any,
    torch: Any,
    spec: EmbeddingModelSpec,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
) -> Any:
    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "normalize_embeddings": True,
    }
    encoded_texts = list(texts)
    if is_query and spec.query_mode == "qwen_prompt_name":
        prompts = getattr(model, "prompts", {}) or {}
        if "query" in prompts:
            encode_kwargs["prompt_name"] = "query"
        else:
            encoded_texts = [f"Instruct: {DEFAULT_QUERY_TASK}\nQuery:{query}" for query in texts]
    encoded = model.encode(encoded_texts, **encode_kwargs)
    if not torch.is_tensor(encoded):
        encoded = torch.as_tensor(encoded)
    return torch.nn.functional.normalize(encoded.float(), p=2, dim=1)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[position]


def _model_revision(model: Any) -> str | None:
    modules = getattr(model, "_modules", {})
    transformer = modules.get("0") if isinstance(modules, Mapping) else None
    auto_model = getattr(transformer, "auto_model", None)
    config = getattr(auto_model, "config", None)
    value = getattr(config, "_commit_hash", None)
    return str(value) if value else None


def _failure_result(
    spec: EmbeddingModelSpec,
    split: SplitName,
    protocol: EmbeddingBenchmarkProtocol,
    corpus_count: int,
    query_count: int,
    started: float,
    error: BaseException,
    *,
    torch: Any | None = None,
) -> EmbeddingBenchmarkResult:
    message = str(error).strip().replace("\n", " ")[:1000] or type(error).__name__
    oom = "out of memory" in message.casefold() or (
        torch is not None
        and isinstance(error, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
    )
    allocated, reserved = (
        _memory_snapshot(torch, protocol.device) if torch is not None else (None, None)
    )
    return EmbeddingBenchmarkResult(
        model_key=spec.key,
        model_id=spec.model_id,
        split=split,
        status="failed",
        batch_size=protocol.batch_size,
        max_seq_length=protocol.max_seq_length,
        requested_dtype=protocol.dtype,
        device=protocol.device,
        query_prompt_mode=spec.query_mode,
        query_prompt_detail="not completed",
        corpus_count=corpus_count,
        query_count=query_count,
        load_time_seconds=max(0.0, time.perf_counter() - started),
        peak_cuda_allocated_mb=allocated,
        peak_cuda_reserved_mb=reserved,
        error_type=type(error).__name__,
        error_message=message,
        oom=oom,
    )


def _benchmark_loaded_model(
    model: Any,
    torch: Any,
    spec: EmbeddingModelSpec,
    split: SplitName,
    cases: Sequence[FrozenEvalCase],
    chunks: Sequence[IndexedChunk],
    protocol: EmbeddingBenchmarkProtocol,
    output_dir: Path,
    load_time: float,
) -> EmbeddingBenchmarkResult:
    device = protocol.device
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_source_blocks: dict[str, set[SourceBlockKey]] = {
        chunk.chunk_id: {(str(chunk.document_id), block_id) for block_id in chunk.source_block_ids}
        for chunk in chunks
    }
    _synchronize(torch, device)
    corpus_started = time.perf_counter()
    corpus_embeddings = _encode(
        model,
        torch,
        spec,
        chunk_texts,
        is_query=False,
        batch_size=protocol.batch_size,
    )
    _synchronize(torch, device)
    corpus_elapsed = time.perf_counter() - corpus_started
    if corpus_embeddings.shape[0] != len(chunks):
        raise EmbeddingBenchmarkError("model returned an invalid corpus embedding count")

    dimension = int(corpus_embeddings.shape[1])
    queries = [case.item.query for case in cases]
    prepared_queries, prompt_detail = _query_texts(spec, queries)
    if protocol.warmup_queries:
        _encode(
            model,
            torch,
            spec,
            prepared_queries[: protocol.warmup_queries],
            is_query=spec.query_mode == "qwen_prompt_name",
            batch_size=1,
        )
        _synchronize(torch, device)

    timings: list[float] = []
    query_encode_elapsed = 0.0
    search_elapsed = 0.0
    metric_rows: list[dict[str, float]] = []
    ranking_rows: list[dict[str, object]] = []
    for case, query in zip(cases, prepared_queries, strict=True):
        query_started = time.perf_counter()
        encode_started = time.perf_counter()
        query_embedding = _encode(
            model,
            torch,
            spec,
            [query],
            is_query=spec.query_mode == "qwen_prompt_name",
            batch_size=1,
        )
        _synchronize(torch, device)
        query_encode_elapsed += time.perf_counter() - encode_started

        search_started = time.perf_counter()
        scores = query_embedding @ corpus_embeddings.T
        top_k = min(10, len(chunk_ids))
        indices = torch.topk(scores[0], k=top_k, largest=True, sorted=True).indices
        _synchronize(torch, device)
        search_elapsed += time.perf_counter() - search_started
        elapsed = time.perf_counter() - query_started
        timings.append(elapsed)
        retrieved = [chunk_ids[int(index)] for index in indices.detach().cpu().tolist()]
        metrics = ranking_metrics(
            retrieved,
            case.relevant_chunk_ids,
            chunk_source_blocks,
            case.gold_source_blocks,
        )
        metric_rows.append(metrics)
        ranking_rows.append(
            {
                "item_id": case.item.item_id,
                "query": case.item.query,
                "retrieved_chunk_ids": retrieved,
                "metrics": metrics,
            }
        )

    ranking_path = output_dir / "rankings" / f"{spec.key}-{split}.jsonl"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in ranking_rows),
        encoding="utf-8",
    )
    metric_names = metric_rows[0].keys() if metric_rows else ()
    aggregate_metrics = {
        name: statistics.fmean(row[name] for row in metric_rows) for name in metric_names
    }
    actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    allocated, reserved = _memory_snapshot(torch, device)
    query_total = query_encode_elapsed + search_elapsed
    query_prompt_detail = prompt_detail
    if spec.query_mode == "qwen_prompt_name" and not (getattr(model, "prompts", {}) or {}).get(
        "query"
    ):
        query_prompt_detail = "manual official Qwen Instruct + Query template fallback"
    return EmbeddingBenchmarkResult(
        model_key=spec.key,
        model_id=spec.model_id,
        split=split,
        status="success",
        batch_size=protocol.batch_size,
        max_seq_length=protocol.max_seq_length,
        requested_dtype=protocol.dtype,
        actual_dtype=actual_dtype,
        device=device,
        query_prompt_mode=spec.query_mode,
        query_prompt_detail=query_prompt_detail,
        embedding_dimension=dimension,
        model_revision=_model_revision(model),
        corpus_count=len(chunks),
        query_count=len(cases),
        load_time_seconds=load_time,
        corpus_encoding_time_seconds=corpus_elapsed,
        corpus_throughput_chunks_per_second=len(chunks) / corpus_elapsed
        if corpus_elapsed
        else None,
        query_encoding_time_seconds=query_encode_elapsed,
        query_throughput_per_second=len(cases) / query_total if query_total else None,
        exact_search_time_seconds=search_elapsed,
        query_latency_mean_ms=statistics.fmean(timings) * 1000 if timings else None,
        query_latency_p50_ms=_percentile(timings, 0.50) * 1000,
        query_latency_p95_ms=_percentile(timings, 0.95) * 1000,
        peak_cuda_allocated_mb=allocated,
        peak_cuda_reserved_mb=reserved,
        metrics=aggregate_metrics,
        ranking_file=str(ranking_path),
    )


def benchmark_model(
    spec: EmbeddingModelSpec,
    split: SplitName,
    cases: Sequence[FrozenEvalCase],
    chunks: Sequence[IndexedChunk],
    protocol: EmbeddingBenchmarkProtocol,
    output_dir: Path,
) -> EmbeddingBenchmarkResult:
    """Load and benchmark one model, preserving a failure record on OOM."""
    torch, sentence_transformer = _import_runtime()
    if protocol.device.startswith("cuda") and not torch.cuda.is_available():
        raise EmbeddingBenchmarkError("CUDA was requested but is not available")
    requested_dtype = _dtype(torch, protocol.dtype)
    started = time.perf_counter()
    model: Any | None = None
    try:
        model_kwargs: dict[str, Any] = {}
        if protocol.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = requested_dtype
        tokenizer_kwargs: dict[str, Any] = {}
        if spec.query_mode == "qwen_prompt_name":
            tokenizer_kwargs["padding_side"] = "left"
        if protocol.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = sentence_transformer(
            spec.model_id,
            device=protocol.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        model.max_seq_length = protocol.max_seq_length
        _synchronize(torch, protocol.device)
        load_time = time.perf_counter() - started
        return _benchmark_loaded_model(
            model,
            torch,
            spec,
            split,
            cases,
            chunks,
            protocol,
            output_dir,
            load_time,
        )
    except Exception as error:
        return _failure_result(
            spec,
            split,
            protocol,
            len(chunks),
            len(cases),
            started,
            error,
            torch=torch,
        )
    finally:
        del model
        gc.collect()
        if protocol.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _environment(
    torch: Any,
    chunks: Sequence[IndexedChunk],
    chunk_index_path: Path,
    dataset_path: Path,
) -> EmbeddingBenchmarkEnvironment:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if cuda_available else None
    )
    return EmbeddingBenchmarkEnvironment(
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
    )


def run_embedding_benchmark(
    *,
    split: RunSplit = "dev",
    model_keys: Sequence[str] = MODEL_KEYS,
    protocol: EmbeddingBenchmarkProtocol | None = None,
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Run the same dense protocol for selected models and requested splits."""
    protocol = protocol or EmbeddingBenchmarkProtocol()
    unknown = sorted(set(model_keys) - set(MODEL_SPECS))
    if unknown:
        raise EmbeddingBenchmarkError(f"unknown embedding model keys: {', '.join(unknown)}")
    if split not in SUPPORTED_SPLITS:
        raise EmbeddingBenchmarkError(f"unsupported benchmark split: {split}")
    torch, _ = _import_runtime()
    chunks_by_id = load_frozen_chunk_index(chunk_index_path)
    chunks = [chunks_by_id[chunk_id] for chunk_id in sorted(chunks_by_id)]
    splits: tuple[SplitName, ...] = ("dev", "test") if split == "both" else (split,)
    cases_by_split = {
        requested_split: load_frozen_eval_cases(
            requested_split,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            chunk_index=chunks_by_id,
        )[0]
        for requested_split in splits
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment(torch, chunks, chunk_index_path, dataset_path)
    results_path = output_dir / "results.jsonl"
    persisted_results: dict[tuple[str, SplitName], EmbeddingBenchmarkResult] = {}
    if results_path.is_file():
        for record in _read_jsonl(results_path):
            result = EmbeddingBenchmarkResult.model_validate(record)
            persisted_results[(result.model_key, result.split)] = result
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        for requested_split in splits:
            persisted_results[(model_key, requested_split)] = benchmark_model(
                spec,
                requested_split,
                cases_by_split[requested_split],
                chunks,
                protocol,
                output_dir,
            )
            results_path.write_text(
                "".join(
                    f"{json.dumps(row.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)}"
                    "\n"
                    for row in persisted_results.values()
                ),
                encoding="utf-8",
            )
    results = list(persisted_results.values())
    status_counts = Counter(result.status for result in results)
    manifest = {
        "schema_version": EMBEDDING_BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "model_keys": list(model_keys),
        "model_specs": [
            {
                "key": MODEL_SPECS[key].key,
                "model_id": MODEL_SPECS[key].model_id,
                "query_mode": MODEL_SPECS[key].query_mode,
                "official_reference": MODEL_SPECS[key].official_reference,
            }
            for key in model_keys
        ],
        "protocol": protocol.model_dump(mode="json"),
        "environment": environment.model_dump(mode="json"),
        "status_counts": dict(sorted(status_counts.items())),
        "results_file": "results.jsonl",
        "ranking_directory": "rankings",
        "test_selection_policy": (
            "Run the unchanged final protocol for test after dev protocol checks; "
            "do not tune per model against test results."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "results": [result.model_dump(mode="json") for result in results]}


__all__ = [
    "DEFAULT_CHUNK_INDEX",
    "DEFAULT_DATASET",
    "DEFAULT_MATERIALIZED",
    "DEFAULT_OUTPUT",
    "MODEL_KEYS",
    "MODEL_SPECS",
    "EmbeddingBenchmarkEnvironment",
    "EmbeddingBenchmarkError",
    "EmbeddingBenchmarkProtocol",
    "EmbeddingBenchmarkResult",
    "EmbeddingModelSpec",
    "benchmark_model",
    "load_frozen_chunk_index",
    "load_frozen_eval_cases",
    "run_embedding_benchmark",
]
