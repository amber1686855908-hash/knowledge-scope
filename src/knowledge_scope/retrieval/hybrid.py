"""Deterministic vector-plus-graph retrieval for KnowledgeScope."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from knowledge_scope.graph.retrieval import (
    GraphEvidence,
    GraphEvidenceResult,
    GraphPath,
    GraphRetrievalResult,
)
from knowledge_scope.graph.retrieval_service import GraphRetrievalError

from .qdrant import ChunkVectorPayload, RetrievedChunk
from .reranking import RerankedChunk, RerankingService
from .service import RetrievalError, RetrievalResult

HYBRID_SCHEMA_VERSION = "1.0"

HybridSource = Literal["vector", "graph", "both"]
HybridBranchStatus = Literal["success", "empty", "failed", "timed_out"]
HybridFailureMode = Literal["strict", "degraded"]


class HybridRetrievalError(RuntimeError):
    """Raised when hybrid retrieval cannot return a trustworthy result."""


class HybridRetrievalConfig(BaseModel):
    """Small, explicit bounds and fusion settings for one hybrid request."""

    model_config = ConfigDict(extra="forbid")

    rrf_k: StrictInt = Field(default=60, ge=1, le=10_000)
    vector_candidate_limit: StrictInt = Field(default=20, ge=1, le=100)
    vector_rerank_limit: StrictInt = Field(default=10, ge=1, le=100)
    graph_result_limit: StrictInt = Field(default=20, ge=1, le=500)
    result_limit: StrictInt = Field(default=20, ge=1, le=500)
    failure_mode: HybridFailureMode = "degraded"

    @model_validator(mode="after")
    def validate_candidate_bounds(self) -> HybridRetrievalConfig:
        if self.vector_rerank_limit > self.vector_candidate_limit:
            raise ValueError("vector_rerank_limit must not exceed vector_candidate_limit")
        return self


class HybridGraphContribution(BaseModel):
    """One consolidated graph contribution for a fused chunk."""

    model_config = ConfigDict(extra="forbid")

    rank: StrictInt = Field(ge=1)
    score: float = Field(ge=0, le=1)
    seed_entity_id: str = Field(pattern=r"^entity_v2_[0-9a-f]{64}$")
    retrieval_reason: str = Field(min_length=1, max_length=200)
    evidence_ids: list[str] = Field(min_length=1)
    paths: list[GraphPath] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_metadata(self) -> HybridGraphContribution:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("graph evidence IDs must be unique")
        path_keys = [path.model_dump_json() for path in self.paths]
        if len(path_keys) != len(set(path_keys)):
            raise ValueError("graph paths must be unique")
        return self


class HybridResult(BaseModel):
    """One deduplicated chunk with explainable branch contributions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = HYBRID_SCHEMA_VERSION
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str = Field(min_length=1)
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    evidence_ids: list[str] = Field(default_factory=list)
    source: HybridSource
    dense_rank: StrictInt | None = Field(default=None, ge=1)
    dense_score: float | None = None
    vector_rank: StrictInt | None = Field(default=None, ge=1)
    reranker_rank: StrictInt | None = Field(default=None, ge=1)
    reranker_score: float | None = None
    graph_rank: StrictInt | None = Field(default=None, ge=1)
    graph_score: float | None = Field(default=None, ge=0, le=1)
    vector_rrf_contribution: float = Field(ge=0)
    graph_rrf_contribution: float = Field(ge=0)
    fusion_score: float = Field(ge=0)
    graph_contribution: HybridGraphContribution | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> HybridResult:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("source_block_ids must be unique")
        has_vector = self.vector_rank is not None
        has_graph = self.graph_rank is not None
        expected_source: HybridSource
        if has_vector and has_graph:
            expected_source = "both"
        elif has_vector:
            expected_source = "vector"
        elif has_graph:
            expected_source = "graph"
        else:
            raise ValueError("hybrid result must have a vector or graph contribution")
        if self.source != expected_source:
            raise ValueError("source does not match branch contributions")
        if has_graph and self.graph_contribution is None:
            raise ValueError("graph results require graph contribution metadata")
        if not has_graph and self.graph_contribution is not None:
            raise ValueError("vector-only results cannot contain graph metadata")
        if self.evidence_ids and self.graph_contribution is None:
            raise ValueError("evidence IDs require graph contribution metadata")
        if self.graph_contribution is not None and self.evidence_ids != sorted(
            self.graph_contribution.evidence_ids
        ):
            raise ValueError("evidence IDs must match graph contribution metadata")
        return self


class HybridRetrievalResult(BaseModel):
    """A bounded hybrid response, including visible branch failure state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = HYBRID_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    items: list[HybridResult] = Field(default_factory=list)
    vector_status: HybridBranchStatus
    graph_status: HybridBranchStatus
    vector_error: str | None = None
    graph_error: str | None = None
    vector_latency_ms: float = Field(ge=0)
    graph_latency_ms: float = Field(ge=0)
    fusion_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    degraded: bool = False

    @model_validator(mode="after")
    def validate_scope(self) -> HybridRetrievalResult:
        if any(item.knowledge_base_id != self.knowledge_base_id for item in self.items):
            raise ValueError("hybrid item scope does not match the request KB")
        if self.vector_status in {"failed", "timed_out"} and not self.vector_error:
            raise ValueError("failed vector branch must expose a safe error")
        if self.graph_status in {"failed", "timed_out"} and not self.graph_error:
            raise ValueError("failed graph branch must expose a safe error")
        return self

    @property
    def source_distribution(self) -> dict[str, int]:
        """Return stable counts by branch provenance for this response."""
        counts = {"vector": 0, "graph": 0, "both": 0}
        for item in self.items:
            counts[item.source] += 1
        return counts


class VectorRetrievalProtocol(Protocol):
    """Structural contract for the existing dense retrieval service."""

    def search(
        self,
        query: str,
        *,
        limit: int,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> RetrievalResult:
        """Return bounded dense candidates."""


class GraphRetrievalProtocol(Protocol):
    """Structural contract for the existing A3.4 graph retrieval service."""

    def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> GraphRetrievalResult:
        """Return bounded, evidence-bearing graph results."""


@dataclass(slots=True)
class _BranchExecution:
    status: HybridBranchStatus
    items: tuple[RerankedChunk, ...] | tuple[GraphEvidenceResult, ...]
    error: str | None
    latency_ms: float


@dataclass(slots=True)
class _FusedItem:
    key: tuple[UUID, UUID, str]
    vector: RerankedChunk | None = None
    vector_rank: int | None = None
    graph_matches: list[tuple[int, GraphEvidenceResult]] = field(default_factory=list)


def _is_timeout(error: BaseException) -> bool:
    """Recognize direct and wrapped timeout failures without exposing details."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.casefold():
            return True
        current = current.__cause__ or current.__context__
    return False


def _safe_branch_error(branch: str, error: BaseException) -> str:
    """Normalize branch failures without returning provider or infrastructure details."""
    if _is_timeout(error):
        return f"{branch} retrieval timed out"
    if isinstance(error, (HybridRetrievalError, RetrievalError, GraphRetrievalError, ValueError)):
        message = str(error).strip()
        if message:
            return message[:240]
    return f"{branch} retrieval failed"


def _branch_failure_status(error: BaseException) -> Literal["failed", "timed_out"]:
    """Classify recoverable branch failures without converting cancellation to success."""
    return "timed_out" if _is_timeout(error) else "failed"


def _path_key(path: GraphPath) -> str:
    return path.model_dump_json()


def _append_unique(values: list[str], seen: set[str], candidates: Sequence[str]) -> None:
    """Append lineage values once while retaining their authoritative order."""
    for value in candidates:
        if value not in seen:
            seen.add(value)
            values.append(value)


def _graph_best_match(
    matches: Sequence[tuple[int, GraphEvidenceResult]],
) -> tuple[int, GraphEvidenceResult]:
    return min(
        matches,
        key=lambda pair: (
            -pair[1].score,
            min(path.hop_distance for path in pair[1].paths),
            pair[0],
            pair[1].seed_entity_id,
            pair[1].evidence.evidence_id,
        ),
    )


def _validate_cross_branch_lineage(
    payload: ChunkVectorPayload,
    evidence: GraphEvidence,
) -> None:
    """Reject graph evidence that cannot belong to the vector chunk lineage.

    Graph evidence may cover only a subset of a chunk's source blocks and page
    range, but it must not introduce lineage outside the chunk selected by the
    vector branch.  Failing closed is safer than silently combining conflicting
    records from the two stores.
    """
    if evidence.page_start < payload.page_start or evidence.page_end > payload.page_end:
        raise HybridRetrievalError("vector and graph results have conflicting page lineage")
    if not set(evidence.source_block_ids).issubset(payload.source_block_ids):
        raise HybridRetrievalError("vector and graph results have conflicting block lineage")
    if evidence.section_path and evidence.section_path != payload.section_path:
        raise HybridRetrievalError("vector and graph results have conflicting section lineage")


class HybridRetrievalService:
    """Run existing vector and graph retrievers concurrently, then fuse by RRF."""

    def __init__(
        self,
        vector_retrieval: VectorRetrievalProtocol,
        reranking: RerankingService,
        graph_retrieval: GraphRetrievalProtocol,
        *,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.vector_retrieval = vector_retrieval
        self.reranking = reranking
        self.graph_retrieval = graph_retrieval
        self.config = config or HybridRetrievalConfig()

    def _search_vector_sync(
        self,
        query: str,
        knowledge_base_id: UUID,
        document_id: UUID | None,
    ) -> tuple[RerankedChunk, ...]:
        dense = self.vector_retrieval.search(
            query,
            limit=self.config.vector_candidate_limit,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        validated_items: list[RetrievedChunk] = []
        for item in dense.items:
            try:
                item = RetrievedChunk.model_validate(item.model_dump(mode="json"))
            except (TypeError, ValueError) as error:
                raise HybridRetrievalError("vector result has invalid chunk lineage") from error
            if item.payload.knowledge_base_id != knowledge_base_id:
                raise HybridRetrievalError("vector result has invalid knowledge-base lineage")
            if document_id is not None and item.payload.document_id != document_id:
                raise HybridRetrievalError("vector result has invalid document lineage")
            if document_id is None or item.payload.document_id == document_id:
                validated_items.append(item)
        return self.reranking.rerank(
            query,
            validated_items,
            limit=self.config.vector_rerank_limit,
        )

    def _search_graph_sync(
        self,
        query: str,
        knowledge_base_id: UUID,
        document_id: UUID | None,
    ) -> tuple[GraphEvidenceResult, ...]:
        result = self.graph_retrieval.search(
            query,
            knowledge_base_id,
            document_id=document_id,
        )
        if result.knowledge_base_id != knowledge_base_id:
            raise HybridRetrievalError("graph result has invalid knowledge-base lineage")
        validated_items: list[GraphEvidenceResult] = []
        for item in result.items:
            try:
                item = GraphEvidenceResult.model_validate(item.model_dump(mode="json"))
            except (TypeError, ValueError) as error:
                raise HybridRetrievalError("graph result has invalid evidence lineage") from error
            if item.evidence.knowledge_base_id != knowledge_base_id:
                raise HybridRetrievalError("graph result has invalid knowledge-base lineage")
            if document_id is not None and item.evidence.document_id != document_id:
                continue
            validated_items.append(item)
        return tuple(validated_items[: self.config.graph_result_limit])

    async def _execute_vector(
        self,
        query: str,
        knowledge_base_id: UUID,
        document_id: UUID | None,
    ) -> _BranchExecution:
        started = perf_counter()
        try:
            items = await asyncio.to_thread(
                self._search_vector_sync,
                query,
                knowledge_base_id,
                document_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _BranchExecution(
                status=_branch_failure_status(error),
                items=(),
                error=_safe_branch_error("vector", error),
                latency_ms=(perf_counter() - started) * 1_000,
            )
        return _BranchExecution(
            status="success" if items else "empty",
            items=items,
            error=None,
            latency_ms=(perf_counter() - started) * 1_000,
        )

    async def _execute_graph(
        self,
        query: str,
        knowledge_base_id: UUID,
        document_id: UUID | None,
    ) -> _BranchExecution:
        started = perf_counter()
        try:
            items = await asyncio.to_thread(
                self._search_graph_sync,
                query,
                knowledge_base_id,
                document_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _BranchExecution(
                status=_branch_failure_status(error),
                items=(),
                error=_safe_branch_error("graph", error),
                latency_ms=(perf_counter() - started) * 1_000,
            )
        return _BranchExecution(
            status="success" if items else "empty",
            items=items,
            error=None,
            latency_ms=(perf_counter() - started) * 1_000,
        )

    def _fuse(
        self,
        vector_items: Sequence[RerankedChunk],
        graph_items: Sequence[GraphEvidenceResult],
    ) -> list[HybridResult]:
        entries: dict[tuple[UUID, UUID, str], _FusedItem] = {}
        unique_vector_items: list[RerankedChunk] = []
        vector_by_key: dict[tuple[UUID, UUID, str], RerankedChunk] = {}
        for item in vector_items:
            payload = item.chunk.payload
            key = (payload.knowledge_base_id, payload.document_id, payload.chunk_id)
            previous = vector_by_key.get(key)
            if previous is not None and previous.chunk.payload != payload:
                raise HybridRetrievalError("vector branch returned conflicting chunk lineage")
            if previous is None:
                vector_by_key[key] = item
                unique_vector_items.append(item)

        for vector_rank, item in enumerate(unique_vector_items, start=1):
            payload = item.chunk.payload
            key = (payload.knowledge_base_id, payload.document_id, payload.chunk_id)
            entries[key] = _FusedItem(key=key, vector=item, vector_rank=vector_rank)

        graph_groups: dict[tuple[UUID, UUID, str], list[GraphEvidenceResult]] = {}
        graph_keys: list[tuple[UUID, UUID, str]] = []
        for item in graph_items:
            evidence = item.evidence
            key = (evidence.knowledge_base_id, evidence.document_id, evidence.chunk_id)
            if key not in graph_groups:
                graph_keys.append(key)
                graph_groups[key] = []
            graph_groups[key].append(item)

        for graph_rank, key in enumerate(graph_keys, start=1):
            entry = entries.get(key)
            if entry is None:
                entry = _FusedItem(key=key)
                entries[key] = entry
            for item in graph_groups[key]:
                if entry.vector is not None:
                    _validate_cross_branch_lineage(entry.vector.chunk.payload, item.evidence)
                entry.graph_matches.append((graph_rank, item))

        results: list[HybridResult] = []
        for entry in entries.values():
            vector = entry.vector
            graph_matches = entry.graph_matches
            graph_contribution: HybridGraphContribution | None = None
            graph_rank: int | None = None
            graph_score: float | None = None
            evidence_ids: list[str] = []
            graph_source_blocks: list[str] = []
            graph_source_block_seen: set[str] = set()
            graph_page_ranges: list[tuple[int, int]] = []
            graph_paths: dict[str, GraphPath] = {}
            best_graph: GraphEvidenceResult | None = None
            if graph_matches:
                graph_rank = min(rank for rank, _item in graph_matches)
                _best_rank, best_graph = _graph_best_match(graph_matches)
                graph_score = max(item.score for _rank, item in graph_matches)
                evidence_ids = sorted({item.evidence.evidence_id for _rank, item in graph_matches})
                for _rank, graph_item in graph_matches:
                    _append_unique(
                        graph_source_blocks,
                        graph_source_block_seen,
                        graph_item.evidence.source_block_ids,
                    )
                    graph_page_ranges.append(
                        (graph_item.evidence.page_start, graph_item.evidence.page_end)
                    )
                    for path in graph_item.paths:
                        graph_paths.setdefault(_path_key(path), path)
                graph_contribution = HybridGraphContribution(
                    rank=graph_rank,
                    score=graph_score,
                    seed_entity_id=best_graph.seed_entity_id,
                    retrieval_reason=best_graph.retrieval_reason,
                    evidence_ids=evidence_ids,
                    paths=[graph_paths[key] for key in sorted(graph_paths)],
                )
            if vector is not None:
                vector_payload = vector.chunk.payload
                vector_page_start = vector_payload.page_start
                vector_page_end = vector_payload.page_end
                source_block_ids = list(vector_payload.source_block_ids)
                page_start = vector_page_start
                page_end = vector_page_end
                section_path = list(vector_payload.section_path)
                dense_score = vector.chunk.score
                dense_rank = vector.dense_rank
                reranker_score = vector.reranker_score
                reranker_rank = entry.vector_rank
            else:
                assert best_graph is not None
                page_start = best_graph.evidence.page_start
                page_end = best_graph.evidence.page_end
                source_block_ids = []
                section_path = list(best_graph.evidence.section_path)
                dense_score = None
                dense_rank = None
                reranker_score = None
                reranker_rank = None
            if graph_page_ranges:
                page_start = min(page_start, *(start for start, _end in graph_page_ranges))
                page_end = max(page_end, *(end for _start, end in graph_page_ranges))
            _append_unique(source_block_ids, set(source_block_ids), graph_source_blocks)
            if not source_block_ids:
                raise HybridRetrievalError("hybrid result has no source-block lineage")
            has_vector = vector is not None
            has_graph = bool(graph_matches)
            source: HybridSource = (
                "both" if has_vector and has_graph else "vector" if has_vector else "graph"
            )
            vector_rrf = (
                1 / (self.config.rrf_k + entry.vector_rank) if entry.vector_rank is not None else 0
            )
            graph_rrf = 1 / (self.config.rrf_k + graph_rank) if graph_rank is not None else 0
            fusion_score = round(vector_rrf + graph_rrf, 12)
            results.append(
                HybridResult(
                    knowledge_base_id=entry.key[0],
                    document_id=entry.key[1],
                    chunk_id=entry.key[2],
                    page_start=page_start,
                    page_end=page_end,
                    source_block_ids=source_block_ids,
                    section_path=section_path,
                    evidence_ids=evidence_ids,
                    source=source,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    vector_rank=entry.vector_rank,
                    reranker_rank=reranker_rank,
                    reranker_score=reranker_score,
                    graph_rank=graph_rank,
                    graph_score=graph_score,
                    vector_rrf_contribution=round(vector_rrf, 12),
                    graph_rrf_contribution=round(graph_rrf, 12),
                    fusion_score=fusion_score,
                    graph_contribution=graph_contribution,
                )
            )
        results.sort(
            key=lambda item: (
                -item.fusion_score,
                str(item.knowledge_base_id),
                str(item.document_id),
                item.page_start,
                item.page_end,
                item.chunk_id,
                tuple(item.source_block_ids),
                tuple(item.evidence_ids),
            )
        )
        return results[: self.config.result_limit]

    async def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
        failure_mode: HybridFailureMode | None = None,
    ) -> HybridRetrievalResult:
        """Run both branches concurrently and fuse their bounded results with RRF."""
        if not query.strip():
            raise ValueError("query must not be blank")
        mode = failure_mode or self.config.failure_mode
        if mode not in {"strict", "degraded"}:
            raise ValueError("failure_mode must be strict or degraded")
        started = perf_counter()
        tasks = (
            asyncio.create_task(self._execute_vector(query, knowledge_base_id, document_id)),
            asyncio.create_task(self._execute_graph(query, knowledge_base_id, document_id)),
        )
        try:
            vector_execution, graph_execution = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        failures = [
            execution
            for execution in (vector_execution, graph_execution)
            if execution.status in {"failed", "timed_out"}
        ]
        if failures and mode == "strict":
            branch = "vector" if vector_execution.status in {"failed", "timed_out"} else "graph"
            raise HybridRetrievalError(f"{branch} branch failed")
        if len(failures) == 2:
            raise HybridRetrievalError("both hybrid retrieval branches failed")

        fusion_started = perf_counter()
        vector_items = (
            vector_execution.items if vector_execution.status not in {"failed", "timed_out"} else ()
        )
        graph_items = (
            graph_execution.items if graph_execution.status not in {"failed", "timed_out"} else ()
        )
        items = self._fuse(vector_items, graph_items)
        fusion_latency_ms = (perf_counter() - fusion_started) * 1_000
        return HybridRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            items=items,
            vector_status=vector_execution.status,
            graph_status=graph_execution.status,
            vector_error=vector_execution.error,
            graph_error=graph_execution.error,
            vector_latency_ms=vector_execution.latency_ms,
            graph_latency_ms=graph_execution.latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            total_latency_ms=(perf_counter() - started) * 1_000,
            degraded=bool(failures),
        )


__all__ = [
    "HYBRID_SCHEMA_VERSION",
    "HybridBranchStatus",
    "HybridFailureMode",
    "HybridGraphContribution",
    "HybridResult",
    "HybridRetrievalConfig",
    "HybridRetrievalError",
    "HybridRetrievalResult",
    "HybridRetrievalService",
    "HybridSource",
]
