"""Unified multi-retriever candidates with one final local reranker.

This module composes the existing dense, sparse, graph, and multimodal
retrievers.  It deliberately owns no retrieval algorithm: branch rank and
branch score only select bounded candidates and explain provenance.  The
configured final reranker is the only authority for the returned order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from knowledge_scope.graph.retrieval import (
    GraphEvidence,
    GraphEvidenceResult,
    GraphPath,
    GraphRetrievalResult,
)

from .qdrant import ChunkVectorPayload, RetrievedChunk
from .representation_index import RepresentationRetrievalResult, RetrievedEvidence
from .reranking import RerankerProtocol
from .service import RetrievalResult
from .sparse import SparseSearchHit, SparseSearchResponse

UNIFIED_RETRIEVAL_SCHEMA_VERSION = "1.0"
UNIFIED_CANDIDATE_ID_SCHEMA_VERSION = "1.0"
UNIFIED_BRANCH_ORDER = ("dense", "sparse", "graph", "multimodal")

UnifiedBranch = Literal["dense", "sparse", "graph", "multimodal"]
UnifiedCandidateKind = Literal["chunk", "evidence"]
UnifiedBranchStatus = Literal["success", "empty", "failed", "timed_out", "cancelled"]
UnifiedFailureMode = Literal["strict", "degraded"]
UnifiedModality = Literal["text", "image", "table", "formula"]
UnifiedCandidateSource = Literal["dense", "sparse", "graph", "multimodal", "multiple"]
UnifiedCandidateKey = tuple[UnifiedCandidateKind, UUID, UUID, str]


class UnifiedRetrievalError(RuntimeError):
    """Raised when unified retrieval cannot return a trustworthy result."""


class DenseRetrievalProtocol(Protocol):
    """Existing dense chunk retrieval contract."""

    def search(
        self,
        query: str,
        *,
        limit: int,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> RetrievalResult:
        """Return bounded dense candidates."""


class SparseRetrievalProtocol(Protocol):
    """Existing sparse retrieval contract."""

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> SparseSearchResponse:
        """Return a bounded sparse response."""


class GraphRetrievalProtocol(Protocol):
    """Existing A3.4 graph retrieval contract."""

    def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> GraphRetrievalResult:
        """Return bounded, evidence-bearing graph results."""


class MultimodalRetrievalProtocol(Protocol):
    """Existing A4.2 Evidence-level representation retrieval contract."""

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> RepresentationRetrievalResult:
        """Return bounded Evidence-level representation results."""


class ChunkPayloadLookupProtocol(Protocol):
    """Resolve current authoritative chunk text for graph-only candidates."""

    def get_chunk_payload(
        self,
        *,
        knowledge_base_id: UUID,
        document_id: UUID,
        chunk_id: str,
    ) -> ChunkVectorPayload | None:
        """Return the current chunk payload, or ``None`` when it is absent."""


class UnifiedRetrievalConfig(BaseModel):
    """Explicit bounded branch, pool, and final-reranking configuration."""

    model_config = ConfigDict(extra="forbid")

    dense_candidate_limit: StrictInt = Field(default=20, ge=1, le=100)
    sparse_candidate_limit: StrictInt = Field(default=20, ge=1, le=100)
    graph_candidate_limit: StrictInt = Field(default=20, ge=1, le=500)
    multimodal_candidate_limit: StrictInt = Field(default=20, ge=1, le=100)
    candidate_pool_limit: StrictInt = Field(default=80, ge=1, le=500)
    result_limit: StrictInt = Field(default=20, ge=1, le=500)
    rerank_text_max_chars: StrictInt = Field(default=6_000, ge=1, le=20_000)
    failure_mode: UnifiedFailureMode = "degraded"
    reranker_model_id: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)

    @model_validator(mode="after")
    def validate_result_bound(self) -> UnifiedRetrievalConfig:
        if self.result_limit > self.candidate_pool_limit:
            raise ValueError("result_limit must not exceed candidate_pool_limit")
        return self

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for the unified retrieval contract."""
        payload = {
            "schema_version": UNIFIED_RETRIEVAL_SCHEMA_VERSION,
            **self.model_dump(mode="json"),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class UnifiedBranchContribution(BaseModel):
    """One branch's rank and explainable metadata for a unified candidate."""

    model_config = ConfigDict(extra="forbid")

    branch: UnifiedBranch
    rank: StrictInt = Field(ge=1)
    score: float
    point_id: UUID | None = None
    representation_ids: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    graph_seed_entity_id: str | None = None
    graph_retrieval_reason: str | None = None
    graph_paths: list[GraphPath] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("branch score must be finite")
        return value

    @field_validator("representation_ids", "matched_terms")
    @classmethod
    def validate_unique_strings(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("branch metadata must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("branch metadata must be unique")
        return sorted(values)

    @model_validator(mode="after")
    def validate_branch_metadata(self) -> UnifiedBranchContribution:
        if self.branch == "multimodal" and not self.representation_ids:
            raise ValueError("multimodal contribution must retain representation IDs")
        if self.branch == "graph":
            if self.graph_seed_entity_id is None or self.graph_retrieval_reason is None:
                raise ValueError("graph contribution must retain seed and retrieval reason")
            if not self.graph_paths:
                raise ValueError("graph contribution must retain graph paths")
        elif (
            self.graph_seed_entity_id is not None
            or self.graph_retrieval_reason is not None
            or self.graph_paths
        ):
            raise ValueError("graph metadata belongs only to graph contributions")
        if self.branch != "multimodal" and self.representation_ids:
            raise ValueError("representation IDs belong only to multimodal contributions")
        return self


def _candidate_id(
    kind: UnifiedCandidateKind,
    knowledge_base_id: UUID,
    document_id: UUID,
    local_id: str,
) -> str:
    payload = {
        "schema_version": UNIFIED_CANDIDATE_ID_SCHEMA_VERSION,
        "kind": kind,
        "knowledge_base_id": str(knowledge_base_id),
        "document_id": str(document_id),
        "local_id": local_id,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}_candidate_v1_{hashlib.sha256(encoded).hexdigest()}"


def chunk_candidate_id(knowledge_base_id: UUID, document_id: UUID, chunk_id: str) -> str:
    """Return the deterministic identity of a chunk-backed unified candidate."""
    if not chunk_id.strip():
        raise ValueError("chunk_id must not be blank")
    return _candidate_id("chunk", knowledge_base_id, document_id, chunk_id)


def evidence_candidate_id(knowledge_base_id: UUID, document_id: UUID, evidence_id: str) -> str:
    """Return the deterministic identity of an Evidence-backed candidate."""
    if not evidence_id.strip():
        raise ValueError("evidence_id must not be blank")
    return _candidate_id("evidence", knowledge_base_id, document_id, evidence_id)


class UnifiedCandidate(BaseModel):
    """One final-reranker candidate with authoritative lineage and branch provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = UNIFIED_RETRIEVAL_SCHEMA_VERSION
    candidate_id: str = Field(min_length=1)
    candidate_kind: UnifiedCandidateKind
    source: UnifiedCandidateSource
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str | None = None
    evidence_id: str | None = None
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    modality: UnifiedModality | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    representation_ids: list[str] = Field(default_factory=list)
    source_text: str | None = None
    rerank_text: str = Field(min_length=1, max_length=20_000)
    branches: list[UnifiedBranchContribution] = Field(min_length=1)
    final_rank: StrictInt | None = Field(default=None, ge=1)
    final_reranker_score: float | None = None

    @field_validator("source_block_ids", "content_types", "asset_refs", "evidence_ids")
    @classmethod
    def validate_lists(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("candidate metadata must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("candidate metadata must be unique")
        return values

    @field_validator("representation_ids")
    @classmethod
    def validate_representation_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("representation IDs must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("representation IDs must be unique")
        return sorted(values)

    @field_validator("final_reranker_score")
    @classmethod
    def validate_final_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("final reranker score must be finite")
        return value

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> UnifiedCandidate:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        branch_names = [contribution.branch for contribution in self.branches]
        if len(branch_names) != len(set(branch_names)):
            raise ValueError("candidate branch contributions must be unique")
        expected_source: UnifiedCandidateSource = (
            branch_names[0] if len(branch_names) == 1 else "multiple"
        )
        if self.source != expected_source:
            raise ValueError("candidate source does not match branch contributions")
        if self.candidate_kind == "chunk":
            if self.chunk_id is None or self.evidence_id is not None:
                raise ValueError("chunk candidates require only chunk identity")
            if self.modality is not None:
                raise ValueError("chunk candidates must not claim Evidence modality")
            if self.candidate_id != chunk_candidate_id(
                self.knowledge_base_id,
                self.document_id,
                self.chunk_id,
            ):
                raise ValueError("chunk candidate ID does not match its identity")
        else:
            if self.evidence_id is None or self.chunk_id is not None or self.modality is None:
                raise ValueError("Evidence candidates require Evidence identity and modality")
            if not self.representation_ids:
                raise ValueError("Evidence candidates require contributing representations")
            if self.candidate_id != evidence_candidate_id(
                self.knowledge_base_id,
                self.document_id,
                self.evidence_id,
            ):
                raise ValueError("Evidence candidate ID does not match its identity")
        if (self.final_rank is None) != (self.final_reranker_score is None):
            raise ValueError("final rank and score must be provided together")
        return self


class UnifiedBranchExecution(BaseModel):
    """Observable outcome of one bounded retrieval branch."""

    model_config = ConfigDict(extra="forbid")

    branch: UnifiedBranch
    status: UnifiedBranchStatus
    candidate_count: StrictInt = Field(ge=0)
    latency_ms: float = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_error_contract(self) -> UnifiedBranchExecution:
        if self.status in {"failed", "timed_out"} and not self.error:
            raise ValueError("failed branch must expose a safe error")
        if self.status in {"success", "empty"} and self.error is not None:
            raise ValueError("successful branch must not expose an error")
        return self


class UnifiedRetrievalResult(BaseModel):
    """Final bounded results and per-branch status/latency observations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = UNIFIED_RETRIEVAL_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    reranker_model_id: str = Field(min_length=1)
    branches: list[UnifiedBranchExecution] = Field(min_length=1)
    candidate_pool_size: StrictInt = Field(ge=0)
    normalization_latency_ms: float = Field(ge=0)
    rerank_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    degraded: bool = False
    items: list[UnifiedCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result_contract(self) -> UnifiedRetrievalResult:
        branch_names = [branch.branch for branch in self.branches]
        if len(branch_names) != len(set(branch_names)):
            raise ValueError("branch executions must be unique")
        if any(item.knowledge_base_id != self.knowledge_base_id for item in self.items):
            raise ValueError("candidate scope does not match the request KB")
        if self.candidate_pool_size < len(self.items):
            raise ValueError("candidate pool cannot be smaller than returned items")
        failures = any(branch.status in {"failed", "timed_out"} for branch in self.branches)
        if failures != self.degraded:
            raise ValueError("degraded flag must reflect branch failures")
        return self

    @property
    def source_distribution(self) -> dict[str, int]:
        """Return stable final candidate counts by branch provenance."""
        counts = {source: 0 for source in ("dense", "sparse", "graph", "multimodal", "multiple")}
        for item in self.items:
            counts[item.source] += 1
        return counts


@dataclass(slots=True)
class _BranchExecution:
    branch: UnifiedBranch
    status: UnifiedBranchStatus
    raw_items: tuple[object, ...]
    error: str | None
    latency_ms: float


@dataclass(slots=True)
class _CandidateBuilder:
    candidate_kind: UnifiedCandidateKind
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str | None
    evidence_id: str | None
    page_start: int
    page_end: int
    source_block_ids: list[str]
    section_path: list[str]
    content_types: list[str]
    asset_refs: list[str]
    modality: UnifiedModality | None
    evidence_ids: set[str] = field(default_factory=set)
    representation_ids: set[str] = field(default_factory=set)
    source_text: str | None = None
    rerank_text: str = ""
    contributions: dict[UnifiedBranch, UnifiedBranchContribution] = field(default_factory=dict)

    @property
    def key(self) -> UnifiedCandidateKey:
        local_id = self.chunk_id if self.candidate_kind == "chunk" else self.evidence_id
        assert local_id is not None
        return self.candidate_kind, self.knowledge_base_id, self.document_id, local_id


def _safe_error(branch: str, error: BaseException) -> str:
    """Return a bounded branch error without exposing infrastructure details."""
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold():
        return f"{branch} retrieval timed out"
    message = str(error).strip()
    return message[:240] if message else f"{branch} retrieval failed"


def _chunk_rerank_text(payload: ChunkVectorPayload, max_chars: int) -> str:
    if payload.text.strip():
        return payload.text.strip()[:max_chars]
    carrier = " / ".join(payload.section_path)
    types = ", ".join(payload.content_types)
    value = " ".join(part for part in (carrier, f"[{types}]" if types else "") if part)
    return value[:max_chars] or "[asset-only chunk]"


def _representation_rerank_text(item: RetrievedEvidence, max_chars: int) -> str:
    """Build a bounded deterministic text view without treating it as source Evidence."""
    records: set[tuple[str, str, str]] = set()
    primary_text = item.text
    primary_type = item.representation_type
    primary_id = item.representation_id
    if isinstance(primary_text, str) and primary_text.strip():
        records.add((str(primary_type), str(primary_id), primary_text.strip()))
    for representation in item.representations:
        text = representation.text
        if isinstance(text, str) and text.strip():
            records.add(
                (
                    representation.representation_type,
                    representation.representation_id,
                    text.strip(),
                )
            )
    value = "\n".join(text for _type, _id, text in sorted(records))
    if value:
        return value[:max_chars]
    section = " / ".join(item.section_path)
    modality = item.modality
    return (
        " ".join(part for part in (section, f"[{modality}]" if modality else "") if part)[
            :max_chars
        ]
        or "[evidence without searchable text]"
    )


def _validate_chunk_lineage(
    builder: _CandidateBuilder,
    *,
    document_id: UUID,
    page_start: int,
    page_end: int,
    source_block_ids: Sequence[str],
    section_path: Sequence[str],
    content_types: Sequence[str],
    asset_refs: Sequence[str],
    source_text: str,
) -> None:
    expected = (
        builder.document_id,
        builder.page_start,
        builder.page_end,
        tuple(builder.source_block_ids),
        tuple(builder.section_path),
        tuple(builder.content_types),
        tuple(builder.asset_refs),
    )
    actual = (
        document_id,
        page_start,
        page_end,
        tuple(source_block_ids),
        tuple(section_path),
        tuple(content_types),
        tuple(asset_refs),
    )
    if expected != actual:
        raise UnifiedRetrievalError("retrieval branches returned conflicting chunk lineage")
    if builder.source_text and source_text and builder.source_text != source_text:
        raise UnifiedRetrievalError("retrieval branches returned conflicting chunk text")


def _validate_graph_evidence_lineage(
    payload: ChunkVectorPayload,
    evidence: GraphEvidence,
) -> None:
    """Ensure graph Evidence is a supported slice of the current chunk."""
    if (
        evidence.knowledge_base_id != payload.knowledge_base_id
        or evidence.document_id != payload.document_id
        or evidence.chunk_id != payload.chunk_id
    ):
        raise UnifiedRetrievalError("graph evidence has conflicting chunk identity")
    if evidence.page_start < payload.page_start or evidence.page_end > payload.page_end:
        raise UnifiedRetrievalError("graph evidence has conflicting page lineage")
    if not set(evidence.source_block_ids).issubset(payload.source_block_ids):
        raise UnifiedRetrievalError("graph evidence has conflicting block lineage")
    if evidence.section_path and evidence.section_path != payload.section_path:
        raise UnifiedRetrievalError("graph evidence has conflicting section lineage")


def _record_branch_key(
    branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    branch: UnifiedBranch,
    key: UnifiedCandidateKey,
) -> None:
    """Record one candidate once for its branch, including duplicate hits."""
    if key not in branch_keys[branch]:
        branch_keys[branch].append(key)


def _validate_builder_compatibility(
    builder: _CandidateBuilder,
    incoming: _CandidateBuilder,
) -> None:
    """Check that two branches describe the same candidate lineage."""
    if builder.candidate_kind != incoming.candidate_kind:
        raise UnifiedRetrievalError("retrieval branches returned conflicting candidate kinds")
    if builder.knowledge_base_id != incoming.knowledge_base_id:
        raise UnifiedRetrievalError("retrieval branches returned another knowledge base")
    if builder.document_id != incoming.document_id:
        raise UnifiedRetrievalError("retrieval branches returned conflicting document lineage")
    if builder.page_start != incoming.page_start or builder.page_end != incoming.page_end:
        raise UnifiedRetrievalError("retrieval branches returned conflicting page lineage")
    if builder.source_block_ids != incoming.source_block_ids:
        raise UnifiedRetrievalError("retrieval branches returned conflicting block lineage")
    if builder.section_path != incoming.section_path:
        raise UnifiedRetrievalError("retrieval branches returned conflicting section lineage")
    if builder.content_types != incoming.content_types:
        raise UnifiedRetrievalError("retrieval branches returned conflicting content lineage")
    if builder.asset_refs != incoming.asset_refs:
        raise UnifiedRetrievalError("retrieval branches returned conflicting asset lineage")
    if builder.modality != incoming.modality:
        raise UnifiedRetrievalError("retrieval branches returned conflicting modality")
    if builder.source_text and incoming.source_text and builder.source_text != incoming.source_text:
        raise UnifiedRetrievalError("retrieval branches returned conflicting chunk text")


def _merge_builder(
    builders: dict[UnifiedCandidateKey, _CandidateBuilder],
    branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    branch: UnifiedBranch,
    incoming: _CandidateBuilder,
) -> None:
    """Commit one successfully normalized branch builder into the shared pool."""
    key = incoming.key
    builder = builders.get(key)
    if builder is None:
        builders[key] = incoming
        _record_branch_key(branch_keys, branch, key)
        return
    _validate_builder_compatibility(builder, incoming)
    builder.evidence_ids.update(incoming.evidence_ids)
    builder.representation_ids.update(incoming.representation_ids)
    if builder.source_text is None:
        builder.source_text = incoming.source_text
    for contribution in incoming.contributions.values():
        _merge_contribution(builder, contribution)
    _record_branch_key(branch_keys, branch, key)


def _merge_contribution(
    builder: _CandidateBuilder,
    contribution: UnifiedBranchContribution,
) -> None:
    existing = builder.contributions.get(contribution.branch)
    if existing is None:
        builder.contributions[contribution.branch] = contribution
        return
    if contribution.branch == "graph":
        path_values = {path.model_dump_json(): path for path in existing.graph_paths}
        path_values.update({path.model_dump_json(): path for path in contribution.graph_paths})
        seed_and_reason = min(
            (
                (-candidate.score, candidate.rank, candidate.graph_seed_entity_id or "", candidate)
                for candidate in (existing, contribution)
            ),
            key=lambda value: value[:3],
        )[3]
        builder.contributions[contribution.branch] = UnifiedBranchContribution(
            branch="graph",
            rank=min(existing.rank, contribution.rank),
            score=max(existing.score, contribution.score),
            graph_seed_entity_id=seed_and_reason.graph_seed_entity_id,
            graph_retrieval_reason=seed_and_reason.graph_retrieval_reason,
            graph_paths=[path_values[key] for key in sorted(path_values)],
        )
        return
    if contribution.branch == "multimodal":
        builder.contributions[contribution.branch] = UnifiedBranchContribution(
            branch="multimodal",
            rank=min(existing.rank, contribution.rank),
            score=max(existing.score, contribution.score),
            representation_ids=sorted(
                set(existing.representation_ids) | set(contribution.representation_ids)
            ),
        )
        return
    better = min(
        (existing, contribution),
        key=lambda value: (-value.score, value.rank, str(value.point_id or "")),
    )
    builder.contributions[contribution.branch] = better


def _add_chunk_candidate(
    builders: dict[UnifiedCandidateKey, _CandidateBuilder],
    branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    branch: UnifiedBranch,
    payload: ChunkVectorPayload,
    *,
    rank: int,
    score: float,
    point_id: UUID | None,
    max_chars: int,
    evidence_ids: Sequence[str] = (),
    graph_seed_entity_id: str | None = None,
    graph_retrieval_reason: str | None = None,
    graph_paths: Sequence[GraphPath] = (),
    matched_terms: Sequence[str] = (),
) -> None:
    if payload.knowledge_base_id is None:
        raise UnifiedRetrievalError("chunk candidate has no authoritative knowledge-base scope")
    key: UnifiedCandidateKey = (
        "chunk",
        payload.knowledge_base_id,
        payload.document_id,
        payload.chunk_id,
    )
    builder = builders.get(key)
    if builder is None:
        builder = _CandidateBuilder(
            candidate_kind="chunk",
            knowledge_base_id=payload.knowledge_base_id,
            document_id=payload.document_id,
            chunk_id=payload.chunk_id,
            evidence_id=None,
            page_start=payload.page_start,
            page_end=payload.page_end,
            source_block_ids=list(payload.source_block_ids),
            section_path=list(payload.section_path),
            content_types=list(payload.content_types),
            asset_refs=list(payload.asset_refs),
            modality=None,
            source_text=payload.text,
            rerank_text=_chunk_rerank_text(payload, max_chars),
        )
        builders[key] = builder
        _record_branch_key(branch_keys, branch, key)
    else:
        _validate_chunk_lineage(
            builder,
            document_id=payload.document_id,
            page_start=payload.page_start,
            page_end=payload.page_end,
            source_block_ids=payload.source_block_ids,
            section_path=payload.section_path,
            content_types=payload.content_types,
            asset_refs=payload.asset_refs,
            source_text=payload.text,
        )
        if builder.knowledge_base_id != payload.knowledge_base_id:
            raise UnifiedRetrievalError("retrieval branches returned another knowledge base")
        _record_branch_key(branch_keys, branch, key)
    builder.evidence_ids.update(evidence_ids)
    if builder.source_text is None and payload.text:
        builder.source_text = payload.text
    contribution = UnifiedBranchContribution(
        branch=branch,
        rank=rank,
        score=float(score),
        point_id=point_id,
        matched_terms=list(matched_terms),
        graph_seed_entity_id=graph_seed_entity_id,
        graph_retrieval_reason=graph_retrieval_reason,
        graph_paths=list(graph_paths),
    )
    _merge_contribution(builder, contribution)


def _add_evidence_candidate(
    builders: dict[UnifiedCandidateKey, _CandidateBuilder],
    branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    branch: UnifiedBranch,
    item: RetrievedEvidence,
    *,
    knowledge_base_id: UUID,
    max_chars: int,
) -> None:
    if item.knowledge_base_id != knowledge_base_id:
        raise UnifiedRetrievalError("multimodal candidate has invalid knowledge-base lineage")
    document_id = item.document_id
    evidence_id = item.evidence_id
    if not evidence_id.strip():
        raise UnifiedRetrievalError("multimodal candidate has no authoritative evidence ID")
    modality = item.modality
    if modality not in {"text", "image", "table", "formula"}:
        raise UnifiedRetrievalError("multimodal candidate has invalid modality")
    key: UnifiedCandidateKey = ("evidence", knowledge_base_id, document_id, evidence_id)
    representation_ids = [
        representation.representation_id for representation in item.representations
    ]
    representation_ids.append(item.representation_id)
    representation_ids = sorted(set(representation_ids))
    if not representation_ids:
        raise UnifiedRetrievalError("multimodal candidate has no representation provenance")
    builder = builders.get(key)
    if builder is None:
        builder = _CandidateBuilder(
            candidate_kind="evidence",
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            chunk_id=None,
            evidence_id=evidence_id,
            page_start=item.page_start,
            page_end=item.page_end,
            source_block_ids=list(item.source_block_ids),
            section_path=list(item.section_path),
            content_types=[],
            asset_refs=list(item.asset_refs),
            modality=modality,
            rerank_text=_representation_rerank_text(item, max_chars),
        )
        builders[key] = builder
        _record_branch_key(branch_keys, branch, key)
    else:
        if (
            builder.document_id != document_id
            or builder.page_start != item.page_start
            or builder.page_end != item.page_end
            or builder.source_block_ids != list(item.source_block_ids)
            or builder.section_path != list(item.section_path)
            or builder.asset_refs != list(item.asset_refs)
            or builder.modality != modality
        ):
            raise UnifiedRetrievalError("multimodal branches returned conflicting Evidence lineage")
        _record_branch_key(branch_keys, branch, key)
    builder.representation_ids.update(representation_ids)
    contribution = UnifiedBranchContribution(
        branch=branch,
        rank=item.rank,
        score=item.score,
        representation_ids=representation_ids,
    )
    _merge_contribution(builder, contribution)


def _to_candidate(builder: _CandidateBuilder, max_chars: int) -> UnifiedCandidate:
    branch_values = [
        builder.contributions[name]
        for name in UNIFIED_BRANCH_ORDER
        if name in builder.contributions
    ]
    source: UnifiedCandidateSource = (
        branch_values[0].branch if len(branch_values) == 1 else "multiple"
    )
    if builder.candidate_kind == "chunk":
        candidate_id = chunk_candidate_id(
            builder.knowledge_base_id,
            builder.document_id,
            builder.chunk_id or "",
        )
    else:
        candidate_id = evidence_candidate_id(
            builder.knowledge_base_id,
            builder.document_id,
            builder.evidence_id or "",
        )
    return UnifiedCandidate(
        candidate_id=candidate_id,
        candidate_kind=builder.candidate_kind,
        source=source,
        knowledge_base_id=builder.knowledge_base_id,
        document_id=builder.document_id,
        chunk_id=builder.chunk_id,
        evidence_id=builder.evidence_id,
        page_start=builder.page_start,
        page_end=builder.page_end,
        source_block_ids=list(builder.source_block_ids),
        section_path=list(builder.section_path),
        content_types=list(builder.content_types),
        asset_refs=list(builder.asset_refs),
        modality=builder.modality,
        evidence_ids=sorted(builder.evidence_ids),
        representation_ids=sorted(builder.representation_ids),
        source_text=builder.source_text,
        rerank_text=builder.rerank_text[:max_chars],
        branches=branch_values,
    )


def _fair_pool(
    builders: dict[UnifiedCandidateKey, _CandidateBuilder],
    branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    limit: int,
) -> list[_CandidateBuilder]:
    """Select a bounded pool by deterministic branch round-robin."""
    selected: list[_CandidateBuilder] = []
    selected_keys: set[UnifiedCandidateKey] = set()
    offsets = {branch: 0 for branch in UNIFIED_BRANCH_ORDER}
    while len(selected) < limit:
        progress = False
        for branch in UNIFIED_BRANCH_ORDER:
            keys = branch_keys[branch]
            offset = offsets[branch]
            if offset >= len(keys):
                continue
            offsets[branch] += 1
            progress = True
            key = keys[offset]
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(builders[key])
            if len(selected) >= limit:
                break
        if not progress:
            break
    return selected


class UnifiedRetrievalService:
    """Run all existing retrievers concurrently and apply one final BGE reranker."""

    def __init__(
        self,
        dense_retrieval: DenseRetrievalProtocol,
        sparse_retrieval: SparseRetrievalProtocol,
        graph_retrieval: GraphRetrievalProtocol,
        multimodal_retrieval: MultimodalRetrievalProtocol,
        final_reranker: RerankerProtocol,
        *,
        chunk_lookup: ChunkPayloadLookupProtocol,
        config: UnifiedRetrievalConfig | None = None,
    ) -> None:
        self.dense_retrieval = dense_retrieval
        self.sparse_retrieval = sparse_retrieval
        self.graph_retrieval = graph_retrieval
        self.multimodal_retrieval = multimodal_retrieval
        self.final_reranker = final_reranker
        self.chunk_lookup = chunk_lookup
        self.config = config or UnifiedRetrievalConfig()

    async def _execute_branch(
        self,
        branch: UnifiedBranch,
        work: Callable[[], object],
        extract_items: Callable[[object], Sequence[object]],
    ) -> _BranchExecution:
        started = perf_counter()
        try:
            response = await asyncio.to_thread(work)
            items = tuple(extract_items(response))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            status: UnifiedBranchStatus = (
                "timed_out"
                if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold()
                else "failed"
            )
            return _BranchExecution(
                branch=branch,
                status=status,
                raw_items=(),
                error=_safe_error(branch, error),
                latency_ms=(perf_counter() - started) * 1_000,
            )
        return _BranchExecution(
            branch=branch,
            status="success" if items else "empty",
            raw_items=items,
            error=None,
            latency_ms=(perf_counter() - started) * 1_000,
        )

    def _normalize_branch(
        self,
        execution: _BranchExecution,
        *,
        knowledge_base_id: UUID,
        document_id: UUID | None,
        builders: dict[UnifiedCandidateKey, _CandidateBuilder],
        branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]],
    ) -> None:
        for raw_rank, item in enumerate(execution.raw_items, start=1):
            if execution.branch == "dense":
                if not isinstance(item, RetrievedChunk):
                    raise UnifiedRetrievalError("dense branch returned an invalid chunk")
                payload = item.payload
                if payload.knowledge_base_id != knowledge_base_id:
                    raise UnifiedRetrievalError("dense branch returned another knowledge base")
                if document_id is not None and payload.document_id != document_id:
                    continue
                _add_chunk_candidate(
                    builders,
                    branch_keys,
                    "dense",
                    payload,
                    rank=raw_rank,
                    score=item.score,
                    point_id=item.point_id,
                    max_chars=self.config.rerank_text_max_chars,
                )
            elif execution.branch == "sparse":
                if not isinstance(item, SparseSearchHit):
                    raise UnifiedRetrievalError("sparse branch returned an invalid chunk")
                item_kb = item.knowledge_base_id
                item_document_id = item.document_id
                if item_kb != knowledge_base_id:
                    raise UnifiedRetrievalError("sparse branch returned another knowledge base")
                if document_id is not None and item_document_id != document_id:
                    continue
                payload = ChunkVectorPayload(
                    collection_schema_version="1.0",
                    chunk_id=item.chunk_id,
                    document_id=item_document_id,
                    knowledge_base_id=item_kb,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    source_block_ids=list(item.source_block_ids),
                    section_path=list(item.section_path),
                    content_types=list(item.content_types),
                    asset_refs=list(item.asset_refs),
                    text=item.text,
                    chunking_config_fingerprint="0" * 64,
                    embedding_model="sparse/bm25",
                    embedding_model_revision=None,
                    embedding_config_fingerprint="0" * 64,
                )
                _add_chunk_candidate(
                    builders,
                    branch_keys,
                    "sparse",
                    payload,
                    rank=item.rank if item.rank >= 1 else raw_rank,
                    score=item.bm25_score,
                    point_id=None,
                    max_chars=self.config.rerank_text_max_chars,
                    matched_terms=list(item.matched_terms),
                )
            elif execution.branch == "graph":
                if not isinstance(item, GraphEvidenceResult):
                    raise UnifiedRetrievalError("graph branch returned an invalid evidence result")
                evidence = item.evidence
                if evidence.knowledge_base_id != knowledge_base_id:
                    raise UnifiedRetrievalError("graph branch returned another knowledge base")
                if document_id is not None and evidence.document_id != document_id:
                    continue
                payload = self.chunk_lookup.get_chunk_payload(
                    knowledge_base_id=knowledge_base_id,
                    document_id=evidence.document_id,
                    chunk_id=evidence.chunk_id,
                )
                if payload is None:
                    raise UnifiedRetrievalError(
                        "graph evidence references a chunk absent from the current chunk index"
                    )
                _validate_graph_evidence_lineage(payload, evidence)
                _add_chunk_candidate(
                    builders,
                    branch_keys,
                    "graph",
                    payload,
                    rank=raw_rank,
                    score=item.score,
                    point_id=None,
                    max_chars=self.config.rerank_text_max_chars,
                    evidence_ids=[evidence.evidence_id],
                    graph_seed_entity_id=item.seed_entity_id,
                    graph_retrieval_reason=item.retrieval_reason,
                    graph_paths=item.paths,
                )
            else:
                if not isinstance(item, RetrievedEvidence):
                    raise UnifiedRetrievalError("multimodal branch returned invalid Evidence")
                if item.knowledge_base_id != knowledge_base_id:
                    raise UnifiedRetrievalError("multimodal branch returned another knowledge base")
                if document_id is not None and item.document_id != document_id:
                    continue
                _add_evidence_candidate(
                    builders,
                    branch_keys,
                    "multimodal",
                    item,
                    knowledge_base_id=knowledge_base_id,
                    max_chars=self.config.rerank_text_max_chars,
                )

    async def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
        failure_mode: UnifiedFailureMode | None = None,
    ) -> UnifiedRetrievalResult:
        """Retrieve bounded candidates concurrently and rank them with the final reranker."""
        if not query.strip():
            raise ValueError("query must not be blank")
        mode = failure_mode or self.config.failure_mode
        if mode not in {"strict", "degraded"}:
            raise ValueError("failure_mode must be strict or degraded")
        started = perf_counter()
        tasks = [
            asyncio.create_task(
                self._execute_branch(
                    "dense",
                    lambda: self.dense_retrieval.search(
                        query,
                        limit=self.config.dense_candidate_limit,
                        knowledge_base_id=knowledge_base_id,
                        document_id=document_id,
                    ),
                    lambda response: response.items,
                )
            ),
            asyncio.create_task(
                self._execute_branch(
                    "sparse",
                    lambda: self.sparse_retrieval.search(
                        query,
                        knowledge_base_id=knowledge_base_id,
                        top_k=self.config.sparse_candidate_limit,
                        document_id=document_id,
                    ),
                    lambda response: response.items,
                )
            ),
            asyncio.create_task(
                self._execute_branch(
                    "graph",
                    lambda: self.graph_retrieval.search(
                        query,
                        knowledge_base_id,
                        document_id=document_id,
                    ),
                    lambda response: response.items[: self.config.graph_candidate_limit],
                )
            ),
            asyncio.create_task(
                self._execute_branch(
                    "multimodal",
                    lambda: self.multimodal_retrieval.search(
                        query,
                        knowledge_base_id=knowledge_base_id,
                        top_k=self.config.multimodal_candidate_limit,
                        document_id=document_id,
                    ),
                    lambda response: response.items,
                )
            ),
        ]
        try:
            executions = list(await asyncio.gather(*tasks))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        failures = [
            execution for execution in executions if execution.status in {"failed", "timed_out"}
        ]
        if failures and mode == "strict":
            raise UnifiedRetrievalError(
                ", ".join(f"{execution.branch} branch failed" for execution in failures)
            )
        if failures and len(failures) == len(executions):
            raise UnifiedRetrievalError("all unified retrieval branches failed")

        normalization_started = perf_counter()
        builders: dict[UnifiedCandidateKey, _CandidateBuilder] = {}
        branch_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]] = {
            branch: [] for branch in UNIFIED_BRANCH_ORDER
        }
        normalized_executions: list[_BranchExecution] = []
        for execution in executions:
            if execution.status in {"failed", "timed_out"}:
                normalized_executions.append(execution)
                continue
            branch_builders: dict[UnifiedCandidateKey, _CandidateBuilder] = {}
            branch_candidate_keys: dict[UnifiedBranch, list[UnifiedCandidateKey]] = {
                branch: [] for branch in UNIFIED_BRANCH_ORDER
            }
            try:
                await asyncio.to_thread(
                    self._normalize_branch,
                    execution,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    builders=branch_builders,
                    branch_keys=branch_candidate_keys,
                )
                for key in branch_candidate_keys[execution.branch]:
                    existing = builders.get(key)
                    if existing is not None:
                        _validate_builder_compatibility(existing, branch_builders[key])
                for key in branch_candidate_keys[execution.branch]:
                    _merge_builder(
                        builders,
                        branch_keys,
                        execution.branch,
                        branch_builders[key],
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                normalized_executions.append(
                    _BranchExecution(
                        branch=execution.branch,
                        status="failed",
                        raw_items=execution.raw_items,
                        error=_safe_error(execution.branch, error),
                        latency_ms=execution.latency_ms,
                    )
                )
                continue
            normalized_executions.append(execution)

        normalization_latency_ms = (perf_counter() - normalization_started) * 1_000
        normalized_failures = [
            execution
            for execution in normalized_executions
            if execution.status in {"failed", "timed_out"}
        ]
        if normalized_failures and mode == "strict":
            raise UnifiedRetrievalError(
                ", ".join(f"{execution.branch} branch failed" for execution in normalized_failures)
            )
        if normalized_failures and len(normalized_failures) == len(normalized_executions):
            raise UnifiedRetrievalError("all unified retrieval branches failed validation")

        selected_builders = _fair_pool(
            builders,
            branch_keys,
            self.config.candidate_pool_limit,
        )
        candidates = [
            _to_candidate(builder, self.config.rerank_text_max_chars)
            for builder in selected_builders
        ]
        rerank_started = perf_counter()
        if candidates:
            try:
                scores = await asyncio.to_thread(
                    self.final_reranker.score_pairs,
                    query,
                    [candidate.rerank_text for candidate in candidates],
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise UnifiedRetrievalError("final reranker failed") from error
            if len(scores) != len(candidates):
                raise UnifiedRetrievalError("final reranker returned an invalid score count")
            try:
                normalized_scores = [float(score) for score in scores]
            except (TypeError, ValueError) as error:
                raise UnifiedRetrievalError("final reranker returned invalid scores") from error
            if not all(math.isfinite(score) for score in normalized_scores):
                raise UnifiedRetrievalError("final reranker returned non-finite scores")
            ranked = sorted(
                zip(candidates, normalized_scores, strict=True),
                key=lambda pair: (-float(pair[1]), pair[0].candidate_id),
            )
            final_items = [
                candidate.model_copy(
                    update={
                        "final_rank": rank,
                        "final_reranker_score": float(score),
                    }
                )
                for rank, (candidate, score) in enumerate(ranked[: self.config.result_limit], 1)
            ]
        else:
            final_items = []
        rerank_latency_ms = (perf_counter() - rerank_started) * 1_000

        result = UnifiedRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            config_fingerprint=self.config.fingerprint,
            reranker_model_id=self.final_reranker.model_id,
            branches=[
                UnifiedBranchExecution(
                    branch=execution.branch,
                    status=execution.status,
                    candidate_count=len(execution.raw_items),
                    latency_ms=execution.latency_ms,
                    error=execution.error,
                )
                for execution in normalized_executions
            ],
            candidate_pool_size=len(candidates),
            normalization_latency_ms=normalization_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            total_latency_ms=(perf_counter() - started) * 1_000,
            degraded=bool(normalized_failures),
            items=final_items,
        )
        return result


__all__ = [
    "ChunkPayloadLookupProtocol",
    "DenseRetrievalProtocol",
    "GraphRetrievalProtocol",
    "MultimodalRetrievalProtocol",
    "SparseRetrievalProtocol",
    "UnifiedBranchContribution",
    "UnifiedBranchExecution",
    "UnifiedBranchStatus",
    "UnifiedCandidate",
    "UnifiedCandidateKind",
    "UnifiedFailureMode",
    "UnifiedRetrievalConfig",
    "UnifiedRetrievalError",
    "UnifiedRetrievalResult",
    "UnifiedRetrievalService",
    "chunk_candidate_id",
    "evidence_candidate_id",
]
