"""Resumable A3.6 graph-corpus build and coverage accounting.

The runner deliberately keeps the A3.2 extraction and A3.3 linking services as
the policy owners.  This module only supplies bounded input streaming,
checkpointing, run accounting, and the orchestration needed to scale those
services over the already-produced A1.6 chunk index.

Runtime manifests and checkpoints belong below ``data/evaluation/a3-6`` and
are never a source of truth for textbook content.  They contain identifiers and
bounded operational counters, not raw provider responses or corpus text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported development platform is POSIX
    fcntl = None

from knowledge_scope.chunking.models import Chunk
from knowledge_scope.evaluation.embedding_benchmark import IndexedChunk
from knowledge_scope.evaluation.retrieval_eval import FinalDatasetItem
from knowledge_scope.extraction.models import ExtractionOutput
from knowledge_scope.extraction.prompt import (
    EXTRACTION_PROMPT_LAYOUT_VERSION,
    EXTRACTION_PROMPT_VERSION,
)
from knowledge_scope.extraction.service import (
    ChunkExtractionResult,
    ExtractionError,
    ExtractionPersistenceError,
)
from knowledge_scope.graph.models import (
    GRAPH_SCHEMA_VERSION,
    GraphEntity,
    evidence_id_for,
)
from knowledge_scope.graph.neo4j import GraphStoreError, Neo4jGraphStore
from knowledge_scope.linking.models import LINKING_SCHEMA_VERSION
from knowledge_scope.linking.prompt import (
    LINKING_PROMPT_VERSION,
    MAX_PROMPT_ALIAS_CHARS,
    MAX_PROMPT_ALIASES,
    MAX_PROMPT_EXCERPT_CHARS,
)
from knowledge_scope.linking.service import (
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_PREFIX_WINDOW,
    LinkingRun,
    link_entities,
)
from knowledge_scope.linking.service_types import LocalEntityContext
from knowledge_scope.llm.errors import LLMProviderError
from knowledge_scope.llm.usage import estimate_cost
from knowledge_scope.shared.config import Settings

GRAPH_CORPUS_BUILD_SCHEMA_VERSION = "1.0"
GRAPH_CORPUS_PIPELINE_VERSION = "a3.6.1"
DEFAULT_OUTPUT = Path("data/evaluation/a3-6")
DEFAULT_CORPUS_MANIFEST = Path("data/benchmarks/a1-5/corpus-manifest.jsonl")
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_EVAL_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_ESTIMATE_SUMMARIES: tuple[Path, ...] = (
    Path("data/evaluation/a3-2/debug-9/summary.json"),
    Path("data/evaluation/a3-2/holdout-18/summary.json"),
    Path("data/evaluation/a3-2/fresh-18/summary.json"),
)
DEFAULT_INPUT_SNAPSHOT = Path("docs/benchmarks/a3-6-corpus-input-snapshot.json")

# The current full-run manifest predates the explicit layout marker. Keep this
# fallback fixed so a future layout change cannot silently reinterpret an old
# manifest by changing the fallback along with the new implementation.
LEGACY_EXTRACTION_PROMPT_LAYOUT_VERSION: Final = "legacy-v1"

ChunkRunStatus = Literal["completed", "failed", "cancelled"]
DocumentRunStatus = Literal["completed", "failed", "cancelled", "provider_blocked"]
LinkBatchRunStatus = Literal["completed", "failed", "cancelled"]
BuildRunStatus = Literal["running", "completed", "partial_failure", "cancelled", "provider_blocked"]

DEFAULT_PROVIDER_BLOCK_THRESHOLD = 3
_NON_RETRYABLE_PROVIDER_STATUS_CODES = frozenset(set(range(400, 500)) - {408, 409, 425, 429})


class GraphCorpusBuildError(RuntimeError):
    """Raised when corpus inputs or durable runtime state are unsafe."""


def _find_provider_error(error: BaseException) -> LLMProviderError | None:
    """Find a provider error through the bounded exception-cause chain."""

    current: BaseException | None = error
    seen: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, LLMProviderError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _fatal_provider_signature(error: BaseException) -> tuple[str, int] | None:
    """Return a safe signature only for explicit non-retryable provider 4xx errors."""

    provider_error = _find_provider_error(error)
    if provider_error is None:
        return None
    status_code = provider_error.status_code
    if (
        provider_error.category != "api"
        or provider_error.retryable
        or status_code not in _NON_RETRYABLE_PROVIDER_STATUS_CODES
    ):
        return None
    return provider_error.category, status_code


class ExtractionRunner(Protocol):
    """The small extraction surface needed by the durable runner."""

    async def extract_chunk(
        self,
        chunk: Any,
        *,
        knowledge_base_id: UUID,
    ) -> ChunkExtractionResult:
        """Extract one existing chunk without rerunning parsing."""


class LinkingGateway(Protocol):
    """The optional A3.3 LLM adjudication surface."""

    async def complete(self, request: Any) -> Any:
        """Complete one linking adjudication request."""


class GraphCorpusStore(Protocol):
    """Minimal graph adapter surface used by the build workflow."""

    def upsert_extraction(
        self,
        entities: Sequence[GraphEntity],
        relations: Sequence[Any],
    ) -> Any:
        """Persist one validated chunk atomically within Neo4j."""

    def list_entities_for_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> Sequence[GraphEntity]:
        """Read current supported local entities for one document."""

    def has_document_state(self, knowledge_base_id: UUID, document_id: UUID) -> bool:
        """Return whether graph state already exists for one KB/document scope."""

    def delete_documents_links(
        self,
        knowledge_base_id: UUID,
        document_ids: Sequence[UUID],
    ) -> Any:
        """Remove stale linking state for a bounded set of documents."""

    def upsert_linking_result(
        self,
        canonical_entities: Sequence[Any],
        decisions: Sequence[Any],
        mappings: Sequence[Any],
    ) -> Any:
        """Persist one validated A3.3 plan in one Neo4j transaction."""

    def delete_document(
        self,
        document_id: UUID,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> Any:
        """Remove the current graph state for one document before a rebuild."""


class _RuntimeModel(BaseModel):
    """Strict JSON contracts for ignored operational artifacts."""

    model_config = ConfigDict(extra="forbid")


class ChunkCheckpoint(_RuntimeModel):
    """Latest durable state for one chunk; later records supersede earlier ones."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ChunkRunStatus
    extraction_status: str | None = None
    attempts: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    structured_success: bool = False
    extraction_initial_calls: int = Field(default=0, ge=0)
    extraction_provider_retry_calls: int = Field(default=0, ge=0)
    extraction_truncation_retry_calls: int = Field(default=0, ge=0)
    extraction_corrective_retry_calls: int = Field(default=0, ge=0)
    extraction_corpus_retry_calls: int = Field(default=0, ge=0)
    grounding_rejections: int = Field(default=0, ge=0)
    entity_count: int = Field(default=0, ge=0)
    relation_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = None
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentCheckpoint(_RuntimeModel):
    """Latest durable state for one document's extraction phase."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    knowledge_base_id: UUID
    document_id: UUID
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=0)
    completed_chunk_count: int = Field(ge=0)
    failed_chunk_count: int = Field(ge=0)
    deferred_chunk_count: int = Field(default=0, ge=0)
    status: DocumentRunStatus
    extraction_complete: bool = False
    extraction_partial: bool = False
    linking_complete: bool = False
    linking_blocked: bool = False
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LinkBatchCheckpoint(_RuntimeModel):
    """Latest durable state for one bounded cross-document linking window."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    knowledge_base_id: UUID
    batch_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ids: list[UUID] = Field(min_length=1)
    entity_ids: list[str]
    run_id: UUID
    status: LinkBatchRunStatus
    candidate_count: int = Field(default=0, ge=0)
    link_count: int = Field(default=0, ge=0)
    no_link_count: int = Field(default=0, ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    canonical_entity_count: int = Field(default=0, ge=0)
    membership_count: int = Field(default=0, ge=0)
    initial_calls: int = Field(default=0, ge=0)
    provider_retry_calls: int = Field(default=0, ge=0)
    failed_attempts: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BuildCounters(_RuntimeModel):
    """Cumulative operational counters, never quality claims."""

    documents_attempted: int = Field(default=0, ge=0)
    documents_completed: int = Field(default=0, ge=0)
    documents_failed: int = Field(default=0, ge=0)
    documents_skipped: int = Field(default=0, ge=0)
    chunks_attempted: int = Field(default=0, ge=0)
    chunks_completed: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    chunks_skipped: int = Field(default=0, ge=0)
    accepted_extractions: int = Field(default=0, ge=0)
    empty_extractions: int = Field(default=0, ge=0)
    rejected_extractions: int = Field(default=0, ge=0)
    schema_rejected_extractions: int = Field(default=0, ge=0)
    structured_parse_successes: int = Field(default=0, ge=0)
    grounding_rejections: int = Field(default=0, ge=0)
    provider_calls: int = Field(default=0, ge=0)
    retry_calls: int = Field(default=0, ge=0)
    extraction_initial_calls: int = Field(default=0, ge=0)
    extraction_provider_retry_calls: int = Field(default=0, ge=0)
    extraction_truncation_retry_calls: int = Field(default=0, ge=0)
    extraction_corrective_retry_calls: int = Field(default=0, ge=0)
    extraction_corpus_retry_calls: int = Field(default=0, ge=0)
    entities: int = Field(default=0, ge=0)
    relations: int = Field(default=0, ge=0)
    evidences: int = Field(default=0, ge=0)
    linking_candidates: int = Field(default=0, ge=0)
    link_decisions: int = Field(default=0, ge=0)
    link_count: int = Field(default=0, ge=0)
    no_link_count: int = Field(default=0, ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    canonical_entities: int = Field(default=0, ge=0)
    memberships: int = Field(default=0, ge=0)
    link_batches_attempted: int = Field(default=0, ge=0)
    link_batches_completed: int = Field(default=0, ge=0)
    link_batches_failed: int = Field(default=0, ge=0)
    link_batches_skipped: int = Field(default=0, ge=0)
    linking_initial_calls: int = Field(default=0, ge=0)
    linking_provider_retry_calls: int = Field(default=0, ge=0)
    linking_failed_attempts: int = Field(default=0, ge=0)
    # Zero means that no provider usage has been observed yet.  Once a provider
    # call omits usage, the corresponding total becomes null and stays unknown.
    input_tokens: int | None = Field(default=0, ge=0)
    output_tokens: int | None = Field(default=0, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    estimated_cost_complete: bool = True
    latency_ms: float = Field(default=0.0, ge=0)


class A21GraphCoverage(_RuntimeModel):
    """Query-level coverage derived from graph-supported evidence chunks."""

    item_count: int = Field(ge=0)
    evidence_document_count: int = Field(ge=0)
    relevant_chunk_count: int = Field(ge=0)
    queries_with_any_graph_coverage: int = Field(ge=0)
    queries_with_complete_graph_coverage: int = Field(ge=0)
    queries_with_partial_graph_coverage: int = Field(ge=0)
    queries_with_no_graph_coverage: int = Field(ge=0)


class CorpusAudit(_RuntimeModel):
    """Read-only corpus, graph and frozen-evaluation audit."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    knowledge_base_id: UUID
    expected_document_count: int = Field(ge=0)
    expected_chunk_count: int = Field(ge=0)
    manifest_rows: int = Field(ge=0)
    ready_manifest_rows: int = Field(ge=0)
    unique_manifest_documents: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunk_documents: int = Field(ge=0)
    subjects: list[str]
    registered_document_count: int | None = Field(default=None, ge=0)
    registered_documents_without_manifest: list[UUID]
    manifest_documents_not_registered: list[UUID]
    documents_without_chunks: list[UUID]
    chunks_without_manifest_document: list[UUID]
    graph_status: Literal["available", "unavailable", "not_requested"]
    graph_document_count: int | None = Field(default=None, ge=0)
    graph_evidence_count: int | None = Field(default=None, ge=0)
    graph_entity_count: int | None = Field(default=None, ge=0)
    graph_relation_count: int | None = Field(default=None, ge=0)
    graph_canonical_entity_count: int | None = Field(default=None, ge=0)
    graph_membership_count: int | None = Field(default=None, ge=0)
    graph_supported_chunks: int | None = Field(default=None, ge=0)
    graph_chunks_with_entities: int | None = Field(default=None, ge=0)
    graph_chunks_with_relations: int | None = Field(default=None, ge=0)
    a21_graph_coverage: A21GraphCoverage


class GraphCoverageSnapshot:
    """Internal graph coverage sets plus counts returned by Neo4j."""

    def __init__(
        self,
        *,
        document_ids: Iterable[str] = (),
        evidence_chunk_ids: Iterable[str] = (),
        entity_chunk_ids: Iterable[str] = (),
        relation_chunk_ids: Iterable[str] = (),
        entity_count: int = 0,
        relation_count: int = 0,
        evidence_count: int = 0,
        canonical_entity_count: int = 0,
        membership_count: int = 0,
    ) -> None:
        self.document_ids = frozenset(document_ids)
        self.evidence_chunk_ids = frozenset(evidence_chunk_ids)
        self.entity_chunk_ids = frozenset(entity_chunk_ids)
        self.relation_chunk_ids = frozenset(relation_chunk_ids)
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.evidence_count = evidence_count
        self.canonical_entity_count = canonical_entity_count
        self.membership_count = membership_count


class RunManifest(_RuntimeModel):
    """Durable run metadata and aggregate counters."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    run_id: UUID
    knowledge_base_id: UUID
    pipeline_version: str
    pipeline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pipeline_versions: dict[str, str]
    provider: str
    model: str
    started_at: datetime
    ended_at: datetime | None = None
    status: BuildRunStatus
    # Optional for manifests created before the prompt-layout compatibility
    # marker was introduced.
    prompt_layout_version: str | None = None
    provider_blocked: bool = False
    provider_block_reason: str | None = None
    provider_blocked_at: datetime | None = None
    processed_document_ids: list[UUID] = Field(default_factory=list)
    processed_chunk_ids: list[str] = Field(default_factory=list)
    failed_document_ids: list[UUID] = Field(default_factory=list)
    failed_chunk_ids: list[str] = Field(default_factory=list)
    failed_link_batch_keys: list[str] = Field(default_factory=list)
    input_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_document_count: int | None = Field(default=None, ge=0)
    expected_chunk_count: int | None = Field(default=None, ge=0)
    expected_document_ids: list[UUID] = Field(default_factory=list)
    observed_document_count: int = Field(default=0, ge=0)
    observed_chunk_count: int = Field(default=0, ge=0)
    input_validation_error: str | None = None
    run_error: str | None = None
    counters: BuildCounters = Field(default_factory=BuildCounters)
    note: str = (
        "Operational coverage/build record only; it does not measure "
        "extraction or linking accuracy."
    )


class CostEstimate(_RuntimeModel):
    """Measured sample and transparent extrapolation for a possible full run."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    target_chunks: int = Field(ge=1)
    sample_files: list[str]
    measured_sample_chunks: int = Field(ge=1)
    measured_provider_calls: int = Field(ge=0)
    measured_retry_calls: int = Field(ge=0)
    measured_input_tokens: int | None = Field(default=None, ge=0)
    measured_output_tokens: int | None = Field(default=None, ge=0)
    measured_wall_clock_seconds: float = Field(ge=0)
    observed_retry_rate: float = Field(ge=0, le=1)
    expected_provider_calls_at_observed_rate: int = Field(ge=0)
    expected_retry_calls_at_observed_rate: int = Field(ge=0)
    expected_input_tokens_at_observed_rate: int | None = Field(default=None, ge=0)
    expected_output_tokens_at_observed_rate: int | None = Field(default=None, ge=0)
    estimated_cost_at_observed_rate: Decimal | None = Field(default=None, ge=0)
    pricing_configured: bool
    sequential_wall_clock_seconds: float = Field(ge=0)
    bounded_wall_clock_seconds_at_max_concurrency: float = Field(ge=0)
    max_concurrency: int = Field(ge=1)
    extrapolation_note: str


@dataclass(frozen=True, slots=True)
class CorpusDocumentMetadata:
    """Safe A1.5 manifest metadata for one unique document."""

    benchmark_item_id: str
    document_id: UUID
    subject: str
    relative_path: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusDocumentChunks:
    """One bounded document window from the ordered A1.6 chunk index."""

    metadata: CorpusDocumentMetadata
    chunks: tuple[IndexedChunk, ...]


class CorpusInputDocument(_RuntimeModel):
    """Repository-safe identity for one corpus document, without document text."""

    document_id: UUID
    benchmark_item_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    size_bytes: int | None = Field(default=None, ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=1)
    chunk_ids: list[str] = Field(min_length=1)
    document_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_chunk_inventory(self) -> CorpusInputDocument:
        if self.chunk_count != len(self.chunk_ids):
            raise ValueError("chunk_count must equal the number of chunk_ids")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique within a document")
        if any(not chunk_id.strip() for chunk_id in self.chunk_ids):
            raise ValueError("chunk_ids must not be blank")
        return self


class CorpusInputSnapshot(_RuntimeModel):
    """Authoritative, repository-safe corpus/chunk-set completeness snapshot."""

    schema_version: Literal["1.0"] = GRAPH_CORPUS_BUILD_SCHEMA_VERSION
    # The repository snapshot describes corpus identity, not ownership.  The
    # selected run still fingerprints its explicit target KB separately.
    knowledge_base_id: UUID | None = None
    corpus_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    chunk_index_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    document_ids: list[UUID] = Field(default_factory=list)
    documents: list[CorpusInputDocument] = Field(default_factory=list)
    input_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_document_inventory(self) -> CorpusInputSnapshot:
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ValueError("document_ids must be unique")
        document_ids = [entry.document_id for entry in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("documents must have unique document IDs")
        if self.document_ids and self.document_count != len(self.document_ids):
            raise ValueError("document_count must equal the number of document_ids")
        if self.documents:
            if self.document_count != len(self.documents):
                raise ValueError("document_count must equal the number of documents")
            if set(self.document_ids) != set(document_ids):
                raise ValueError("document_ids and documents must describe the same set")
            if sum(entry.chunk_count for entry in self.documents) != self.chunk_count:
                raise ValueError("chunk_count must equal the document chunk inventory")
        return self


@dataclass(frozen=True, slots=True)
class _ChunkOutcome:
    chunk: IndexedChunk
    checkpoint: ChunkCheckpoint
    result: ChunkExtractionResult | None
    reused: bool = False


def _now() -> datetime:
    return datetime.now(UTC)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_input_fingerprint(entries: Sequence[CorpusInputDocument]) -> str:
    """Fingerprint corpus identity independently of the target Knowledge Base."""

    return _sha256_json(
        {
            "schema_version": GRAPH_CORPUS_BUILD_SCHEMA_VERSION,
            "documents": [
                entry.model_dump(mode="json")
                for entry in sorted(entries, key=lambda value: str(value.document_id))
            ],
        }
    )


def chunk_input_fingerprint(chunk: IndexedChunk) -> str:
    """Fingerprint all chunk fields that can change extraction input."""

    return _sha256_json(chunk.model_dump(mode="json"))


def _document_fingerprint_for(
    document: CorpusDocumentChunks,
) -> str:
    return _sha256_json(
        {
            "schema_version": GRAPH_CORPUS_BUILD_SCHEMA_VERSION,
            "document_id": str(document.metadata.document_id),
            "benchmark_item_id": document.metadata.benchmark_item_id,
            "subject": document.metadata.subject,
            "relative_path": document.metadata.relative_path,
            "size_bytes": document.metadata.size_bytes,
            "sha256": document.metadata.sha256,
            "chunks": [
                {
                    "ordinal": chunk.ordinal,
                    "chunk_id": chunk.chunk_id,
                    "input_fingerprint": chunk_input_fingerprint(chunk),
                    "config_fingerprint": chunk.config_fingerprint,
                }
                for chunk in document.chunks
            ],
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise GraphCorpusBuildError(f"input file is not readable: {path.name}") from error
    return digest.hexdigest()


def build_corpus_input_snapshot(
    documents: Iterable[CorpusDocumentChunks],
    *,
    knowledge_base_id: UUID | None,
    corpus_manifest: Path | None = None,
    chunk_index: Path | None = None,
) -> CorpusInputSnapshot:
    """Build a repository-safe identity snapshot from the selected input stream."""

    entries: list[CorpusInputDocument] = []
    seen_documents: set[UUID] = set()
    seen_chunks: set[str] = set()
    total_chunks = 0
    for document in documents:
        metadata = document.metadata
        if metadata.document_id in seen_documents:
            raise GraphCorpusBuildError("input stream contains duplicate document IDs")
        if not document.chunks:
            raise GraphCorpusBuildError("input snapshot cannot contain an empty document")
        for chunk in document.chunks:
            if chunk.document_id != metadata.document_id:
                raise GraphCorpusBuildError("chunk document ID does not match its document")
            if chunk.chunk_id in seen_chunks:
                raise GraphCorpusBuildError("input stream contains duplicate chunk IDs")
            seen_chunks.add(chunk.chunk_id)
        if [chunk.ordinal for chunk in document.chunks] != list(range(len(document.chunks))):
            raise GraphCorpusBuildError("input snapshot chunk ordinals are not contiguous")
        seen_documents.add(metadata.document_id)
        total_chunks += len(document.chunks)
        entries.append(
            CorpusInputDocument(
                document_id=metadata.document_id,
                benchmark_item_id=metadata.benchmark_item_id,
                subject=metadata.subject,
                relative_path=metadata.relative_path,
                size_bytes=metadata.size_bytes,
                sha256=metadata.sha256,
                chunk_count=len(document.chunks),
                chunk_ids=[chunk.chunk_id for chunk in document.chunks],
                document_fingerprint=_document_fingerprint_for(
                    document,
                ),
            )
        )
    entries.sort(key=lambda value: (value.subject, value.benchmark_item_id, str(value.document_id)))
    return CorpusInputSnapshot(
        knowledge_base_id=knowledge_base_id,
        corpus_manifest_sha256=(
            _sha256_file(corpus_manifest) if corpus_manifest is not None else None
        ),
        chunk_index_sha256=_sha256_file(chunk_index) if chunk_index is not None else None,
        document_count=len(entries),
        chunk_count=total_chunks,
        document_ids=sorted((entry.document_id for entry in entries), key=str),
        documents=entries,
        input_fingerprint=_corpus_input_fingerprint(entries),
    )


def load_corpus_input_snapshot(path: Path = DEFAULT_INPUT_SNAPSHOT) -> CorpusInputSnapshot:
    """Load the safe authoritative corpus snapshot."""

    try:
        return CorpusInputSnapshot.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise GraphCorpusBuildError(
            f"authoritative corpus input snapshot is invalid or unreadable: {path.name}"
        ) from error


def validate_corpus_input_snapshot(
    actual: CorpusInputSnapshot,
    expected: CorpusInputSnapshot,
    *,
    require_full: bool = False,
) -> None:
    """Fail closed when selected corpus identities differ from the authority."""

    authority_ids = set(expected.document_ids)
    expected_by_id = {entry.document_id: entry for entry in expected.documents}
    actual_ids = set(actual.document_ids)
    actual_by_id = {entry.document_id: entry for entry in actual.documents}
    if expected.document_count != len(authority_ids or expected_by_id):
        raise GraphCorpusBuildError("authoritative snapshot document count is inconsistent")
    if actual.document_count != len(actual_ids or actual_by_id):
        raise GraphCorpusBuildError("input snapshot document count is inconsistent")
    if not authority_ids:
        authority_ids = set(expected_by_id)
    if actual_ids and actual_ids != set(actual_by_id):
        raise GraphCorpusBuildError("input snapshot document IDs are inconsistent")
    if expected_by_id and authority_ids != set(expected_by_id):
        raise GraphCorpusBuildError("authoritative snapshot document IDs are inconsistent")

    if (
        expected.knowledge_base_id is not None
        and actual.knowledge_base_id != expected.knowledge_base_id
    ):
        raise GraphCorpusBuildError("corpus input snapshot Knowledge Base differs")
    if (
        expected.corpus_manifest_sha256 is not None
        and actual.corpus_manifest_sha256 != expected.corpus_manifest_sha256
    ):
        raise GraphCorpusBuildError("corpus manifest fingerprint differs from the authority")
    if (
        expected.chunk_index_sha256 is not None
        and actual.chunk_index_sha256 != expected.chunk_index_sha256
    ):
        raise GraphCorpusBuildError("chunk index fingerprint differs from the authority")
    unknown_documents = (actual_ids or set(actual_by_id)) - authority_ids
    if unknown_documents:
        raise GraphCorpusBuildError("input snapshot contains documents outside the authority")
    for document_id, actual_entry in actual_by_id.items():
        expected_entry = expected_by_id.get(document_id)
        if expected_entry is not None and actual_entry != expected_entry:
            raise GraphCorpusBuildError(f"document/chunk fingerprint differs for {document_id}")
    if require_full and (
        actual.document_count != expected.document_count
        or actual.chunk_count != expected.chunk_count
        or (
            expected.input_fingerprint is not None
            and actual.input_fingerprint != expected.input_fingerprint
        )
        or (actual_ids or set(actual_by_id)) != authority_ids
    ):
        raise GraphCorpusBuildError("full corpus input snapshot is incomplete")


def _snapshot_document_ids(snapshot: CorpusInputSnapshot) -> tuple[UUID, ...]:
    """Return the authoritative document IDs from either snapshot form."""

    values = snapshot.document_ids or [entry.document_id for entry in snapshot.documents]
    return tuple(sorted(values, key=str))


def _safe_endpoint_identity(value: str) -> dict[str, str | int | None]:
    """Return provider endpoint identity without userinfo, query, or credentials."""

    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise GraphCorpusBuildError("LLM endpoint has an invalid port") from error
    return {
        "scheme": parsed.scheme.lower(),
        "hostname": (parsed.hostname or "").lower(),
        "port": port,
        "path": parsed.path.rstrip("/"),
    }


def _run_owner_payload() -> bytes:
    return json.dumps(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": _now().isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _RunOwnershipLock:
    """A process-lifetime advisory lock for one checkpoint directory."""

    def __init__(self, path: Path) -> None:
        if fcntl is None:
            raise GraphCorpusBuildError("the checkpoint run lock requires a POSIX file-lock API")
        self.path = path
        self._descriptor: int | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, _run_owner_payload())
            os.fsync(descriptor)
        except OSError as error:
            if "descriptor" in locals():
                os.close(descriptor)
            if getattr(error, "errno", None) in {11, 13}:
                raise GraphCorpusBuildError(
                    "another graph corpus build owns this checkpoint directory"
                ) from error
            raise GraphCorpusBuildError("could not acquire the graph corpus run lock") from error
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def build_pipeline_fingerprint(
    settings: Settings,
    *,
    knowledge_base_id: UUID | None = None,
    input_fingerprint: str | None = None,
    max_concurrency: int | None = None,
    link_batch_documents: int | None = None,
    linking_mode: str | None = None,
    retry_failed: bool = False,
    reprocess: bool = False,
) -> str:
    """Fingerprint the policy/configuration that makes extraction reproducible."""

    return _sha256_json(
        {
            "pipeline_version": GRAPH_CORPUS_PIPELINE_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "graph_id_schema_version": "v2",
            "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
            "input_fingerprint": input_fingerprint,
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "extraction_schema_version": ExtractionOutput.model_json_schema(),
            "linking_schema_version": LINKING_SCHEMA_VERSION,
            "linking_prompt_version": LINKING_PROMPT_VERSION,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "endpoint": _safe_endpoint_identity(settings.llm_base_url),
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "llm_max_retries": settings.llm_max_retries,
            "graph_extraction_max_tokens": settings.graph_extraction_max_tokens,
            "graph_extraction_max_parse_retries": settings.graph_extraction_max_parse_retries,
            "graph_extraction_truncation_budgets": list(
                settings.graph_extraction_truncation_budgets
            ),
            "graph_extraction_max_truncation_retries": (
                settings.graph_extraction_max_truncation_retries
            ),
            "graph_extraction_truncation_ceiling": settings.graph_extraction_truncation_ceiling,
            "response_format": "json_object",
            "reasoning": "disabled",
            "grounding_policy": "a3.2-grounded-output-v1",
            "linking_policy": {
                "max_candidates": DEFAULT_MAX_CANDIDATES,
                "max_block_size": DEFAULT_MAX_BLOCK_SIZE,
                "prefix_window": DEFAULT_PREFIX_WINDOW,
                "prefix_similarity_threshold": 0.72,
                "prompt_max_aliases": MAX_PROMPT_ALIASES,
                "prompt_max_alias_chars": MAX_PROMPT_ALIAS_CHARS,
                "prompt_max_excerpt_chars": MAX_PROMPT_EXCERPT_CHARS,
            },
            "runner": {
                "max_concurrency": max_concurrency,
                "link_batch_documents": link_batch_documents,
                "linking_mode": linking_mode,
                "corpus_retry_mode": "explicit_retry_failed",
                "reprocess_mode": "explicit_reprocess",
            },
        }
    )


def maximum_extraction_provider_attempts(
    settings: Settings,
    *,
    corpus_retry: bool = False,
) -> int:
    """Return the bounded provider-attempt upper bound for one extraction chunk.

    Each logical extraction request can use the gateway's provider retry budget.
    The extraction layer can then issue one initial request, bounded truncation
    escalations, and bounded corrective requests. ``corpus_retry`` represents
    the runner's explicit second pass for a failed chunk.
    """

    logical_calls = (
        1
        + settings.graph_extraction_max_truncation_retries
        + settings.graph_extraction_max_parse_retries
    )
    corpus_passes = 2 if corpus_retry else 1
    return (1 + settings.llm_max_retries) * logical_calls * corpus_passes


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class CheckpointStore:
    """Append-only, fsynced checkpoint log with last-record-wins loading."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_path = output_dir / "chunk-checkpoints.jsonl"
        self.document_path = output_dir / "document-checkpoints.jsonl"
        self.link_path = output_dir / "link-batch-checkpoints.jsonl"
        self._chunks = self._load(self.chunk_path, ChunkCheckpoint)
        self._documents = self._load(self.document_path, DocumentCheckpoint)
        self._links = self._load(self.link_path, LinkBatchCheckpoint)

    @staticmethod
    def _load(path: Path, model_type: type[_RuntimeModel]) -> dict[str, _RuntimeModel]:
        if not path.is_file():
            return {}
        records: dict[str, _RuntimeModel] = {}
        try:
            with path.open(encoding="utf-8") as handle:
                for _line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = model_type.model_validate_json(line)
                    if isinstance(value, ChunkCheckpoint):
                        key = value.chunk_id
                    elif isinstance(value, DocumentCheckpoint):
                        key = str(value.document_id)
                    elif isinstance(value, LinkBatchCheckpoint):
                        key = value.batch_key
                    else:  # pragma: no cover - closed over the model union above
                        raise GraphCorpusBuildError("unknown checkpoint record type")
                    records[key] = value
        except (OSError, ValueError) as error:
            raise GraphCorpusBuildError(
                f"checkpoint file is invalid or unreadable: {path.name}"
            ) from error
        return records

    @staticmethod
    def _append(path: Path, record: _RuntimeModel) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise GraphCorpusBuildError("checkpoint could not be durably written") from error

    def save_chunk(self, record: ChunkCheckpoint) -> None:
        self._append(self.chunk_path, record)
        self._chunks[record.chunk_id] = record

    def save_document(self, record: DocumentCheckpoint) -> None:
        self._append(self.document_path, record)
        self._documents[str(record.document_id)] = record

    def save_link_batch(self, record: LinkBatchCheckpoint) -> None:
        self._append(self.link_path, record)
        self._links[record.batch_key] = record

    def has_records(self) -> bool:
        """Return whether orphan checkpoint records exist without a manifest."""

        return bool(self._chunks or self._documents or self._links)

    def get_chunk(self, chunk_id: str) -> ChunkCheckpoint | None:
        value = self._chunks.get(chunk_id)
        return value if isinstance(value, ChunkCheckpoint) else None

    def get_document(self, document_id: UUID) -> DocumentCheckpoint | None:
        value = self._documents.get(str(document_id))
        return value if isinstance(value, DocumentCheckpoint) else None

    def get_link_batch(self, batch_key: str) -> LinkBatchCheckpoint | None:
        value = self._links.get(batch_key)
        return value if isinstance(value, LinkBatchCheckpoint) else None

    def failed_link_batches_for_documents(
        self,
        document_ids: Iterable[UUID],
    ) -> tuple[str, ...]:
        """Return failed batch keys for the same document window."""

        target = frozenset(document_ids)
        return tuple(
            key
            for key, value in self._links.items()
            if isinstance(value, LinkBatchCheckpoint)
            and value.status == "failed"
            and frozenset(value.document_ids) == target
        )


@dataclass(frozen=True, slots=True)
class FailedProviderAttemptSummary:
    """Distinct failed chunks and provider API attempts from checkpoint history."""

    unique_failed_chunks: int
    failed_provider_attempts: int
    failures_by_time_bucket: tuple[tuple[str, int], ...]


def summarize_failed_provider_attempts(
    records: Iterable[ChunkCheckpoint],
    *,
    bucket_minutes: int = 5,
) -> FailedProviderAttemptSummary:
    """Separate unique API-failed chunks from their accumulated provider attempts.

    ``records`` should be the append-only checkpoint history when cumulative
    attempts are required. A latest-record view still produces a valid
    unique-chunk summary, but cannot reconstruct superseded attempts.
    """

    if not 1 <= bucket_minutes <= 60:
        raise ValueError("bucket_minutes must be between one and 60")
    failed_chunks: set[str] = set()
    failures_by_bucket: Counter[str] = Counter()
    failed_provider_attempts = 0
    for record in records:
        if record.status != "failed" or record.error_category != "api":
            continue
        failed_chunks.add(record.chunk_id)
        provider_attempts = record.provider_calls
        failed_provider_attempts += provider_attempts
        if provider_attempts:
            timestamp = record.updated_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            timestamp = timestamp.astimezone(UTC)
            bucket_minute = (timestamp.minute // bucket_minutes) * bucket_minutes
            bucket = (
                timestamp.replace(
                    minute=bucket_minute,
                    second=0,
                    microsecond=0,
                )
                .isoformat()
                .replace("+00:00", "Z")
            )
            failures_by_bucket[bucket] += provider_attempts
    return FailedProviderAttemptSummary(
        unique_failed_chunks=len(failed_chunks),
        failed_provider_attempts=failed_provider_attempts,
        failures_by_time_bucket=tuple(sorted(failures_by_bucket.items())),
    )


class _ManifestStore:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "run-manifest.json"

    def load(self) -> RunManifest | None:
        if not self.path.is_file():
            return None
        try:
            return RunManifest.model_validate_json(self.path.read_bytes())
        except (OSError, ValueError) as error:
            raise GraphCorpusBuildError("run manifest is invalid or unreadable") from error

    def save(self, manifest: RunManifest) -> None:
        _atomic_write(
            self.path,
            manifest.model_dump_json(indent=2) + "\n",
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise GraphCorpusBuildError(f"JSONL input is not readable: {path.name}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise GraphCorpusBuildError(
                f"invalid JSON on line {line_number}: {path.name}"
            ) from error
        if not isinstance(value, dict):
            raise GraphCorpusBuildError(f"JSONL line is not an object: {path.name}")
        records.append(value)
    return records


def load_corpus_metadata(
    path: Path = DEFAULT_CORPUS_MANIFEST,
) -> tuple[CorpusDocumentMetadata, ...]:
    """Load only safe manifest metadata; canonical and PDF content are not read."""

    records = _read_jsonl(path)
    candidates: dict[UUID, list[CorpusDocumentMetadata]] = {}
    content_fingerprints: dict[UUID, tuple[int, str]] = {}
    for record in records:
        if record.get("inventory_status") != "ready":
            continue
        try:
            item_id = record["benchmark_item_id"]
            document_id = UUID(str(record["benchmark_document_uuid"]))
            subject = record["subject"]
            relative_path = record["relative_path"]
            size_bytes = record["size_bytes"]
            sha256 = record["sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise GraphCorpusBuildError("A1.5 manifest has incomplete document metadata") from error
        if not all(
            isinstance(value, str) and value.strip() for value in (item_id, subject, relative_path)
        ):
            raise GraphCorpusBuildError("A1.5 manifest has blank document metadata")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise GraphCorpusBuildError("A1.5 manifest has invalid content metadata")
        current = CorpusDocumentMetadata(
            benchmark_item_id=item_id,
            document_id=document_id,
            subject=subject,
            relative_path=relative_path,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        current_fingerprint = (size_bytes, sha256)
        previous_fingerprint = content_fingerprints.get(document_id)
        if previous_fingerprint is not None and previous_fingerprint != current_fingerprint:
            raise GraphCorpusBuildError("duplicate A1.5 document IDs have conflicting content")
        content_fingerprints[document_id] = current_fingerprint
        candidates.setdefault(document_id, []).append(current)
    metadata = {
        document_id: min(
            values,
            key=lambda value: (value.relative_path, value.benchmark_item_id),
        )
        for document_id, values in candidates.items()
    }
    return tuple(
        sorted(metadata.values(), key=lambda value: (value.subject, value.benchmark_item_id))
    )


def iter_indexed_chunks(path: Path = DEFAULT_CHUNK_INDEX) -> Iterator[IndexedChunk]:
    """Stream the existing A1.6 JSONL index without retaining the corpus."""

    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise GraphCorpusBuildError(f"A1.6 chunk index is not readable: {path.name}") from error
    seen_ids: set[str] = set()
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = IndexedChunk.model_validate_json(line)
            except ValueError as error:
                raise GraphCorpusBuildError(
                    f"A1.6 chunk index is invalid on line {line_number}"
                ) from error
            if chunk.chunk_id in seen_ids:
                raise GraphCorpusBuildError("A1.6 chunk index contains duplicate chunk IDs")
            seen_ids.add(chunk.chunk_id)
            yield chunk
    except UnicodeDecodeError as error:
        raise GraphCorpusBuildError(f"A1.6 chunk index is not valid UTF-8: {path.name}") from error
    finally:
        handle.close()


def select_document_ids(
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    *,
    full: bool = False,
    sample_per_subject: int = 1,
    sample_offset: int = 0,
    max_documents: int | None = None,
) -> tuple[UUID, ...]:
    """Select all documents or a stable bounded subject-stratified subset."""

    if sample_per_subject < 1:
        raise ValueError("sample_per_subject must be at least one")
    if sample_offset < 0:
        raise ValueError("sample_offset must not be negative")
    metadata = load_corpus_metadata(corpus_manifest)
    if full:
        selected = list(metadata)
    else:
        by_subject: dict[str, list[CorpusDocumentMetadata]] = {}
        for item in metadata:
            by_subject.setdefault(item.subject, []).append(item)
        selected = [
            item
            for subject in sorted(by_subject)
            for item in by_subject[subject][sample_offset : sample_offset + sample_per_subject]
        ]
    selected = sorted(selected, key=lambda value: (value.subject, value.benchmark_item_id))
    if max_documents is not None:
        if max_documents < 1:
            raise ValueError("max_documents must be at least one")
        selected = selected[:max_documents]
    if not selected:
        raise GraphCorpusBuildError("document selection is empty")
    return tuple(item.document_id for item in selected)


def iter_corpus_documents(
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    chunk_index: Path = DEFAULT_CHUNK_INDEX,
    *,
    selected_document_ids: Sequence[UUID] | None = None,
    registered_document_ids: Iterable[UUID] | None = None,
) -> Iterator[CorpusDocumentChunks]:
    """Yield one document at a time from registered metadata and A1.6 chunks."""

    metadata = {item.document_id: item for item in load_corpus_metadata(corpus_manifest)}
    selected = set(selected_document_ids) if selected_document_ids is not None else set(metadata)
    missing_metadata = selected - set(metadata)
    if missing_metadata:
        raise GraphCorpusBuildError("selected documents are missing from the A1.5 manifest")
    if registered_document_ids is not None:
        unregistered = selected - set(registered_document_ids)
        if unregistered:
            raise GraphCorpusBuildError("selected documents are not registered in the target KB")

    current_document_id: UUID | None = None
    current_chunks: list[IndexedChunk] = []
    seen_documents: set[UUID] = set()

    def flush() -> CorpusDocumentChunks | None:
        nonlocal current_document_id, current_chunks
        if current_document_id is None:
            return None
        document_id = current_document_id
        chunks = tuple(current_chunks)
        current_document_id = None
        current_chunks = []
        if document_id in seen_documents:
            raise GraphCorpusBuildError("A1.6 chunk index is not grouped by document")
        seen_documents.add(document_id)
        if document_id not in selected:
            return None
        if not chunks:
            raise GraphCorpusBuildError("selected document has no A1.6 chunks")
        expected_ordinals = list(range(len(chunks)))
        if [chunk.ordinal for chunk in chunks] != expected_ordinals:
            raise GraphCorpusBuildError("A1.6 chunk ordinals are not contiguous within a document")
        return CorpusDocumentChunks(metadata[document_id], chunks)

    for chunk in iter_indexed_chunks(chunk_index):
        if current_document_id != chunk.document_id:
            value = flush()
            if value is not None:
                yield value
            current_document_id = chunk.document_id
        current_chunks.append(chunk)
    value = flush()
    if value is not None:
        yield value
    if selected - seen_documents:
        raise GraphCorpusBuildError("selected documents are missing from the A1.6 chunk index")


def _graph_chunk_sets_from_session(session: Any, knowledge_base_id: UUID) -> GraphCoverageSnapshot:
    """Read bounded graph coverage sets from one Neo4j read session."""

    params = {"knowledge_base_id": str(knowledge_base_id)}

    def count(query: str) -> int:
        record = session.run(query, **params).single()
        return int(record["count"]) if record is not None else 0

    def collect(query: str) -> list[str]:
        record = session.run(query, **params).single()
        if record is None:
            return []
        values = record["values"]
        return [str(value) for value in values or [] if value is not None]

    document_ids = collect(
        "MATCH (e:KnowledgeEvidence {knowledge_base_id: $knowledge_base_id}) "
        "RETURN collect(DISTINCT e.document_id) AS values"
    )
    evidence_chunks = collect(
        "MATCH (e:KnowledgeEvidence {knowledge_base_id: $knowledge_base_id}) "
        "RETURN collect(DISTINCT e.chunk_id) AS values"
    )
    entity_chunks = collect(
        "MATCH (:KnowledgeEntity {knowledge_base_id: $knowledge_base_id})-[:SUPPORTED_BY]->"
        "(e:KnowledgeEvidence {knowledge_base_id: $knowledge_base_id}) "
        "RETURN collect(DISTINCT e.chunk_id) AS values"
    )
    relation_chunks = collect(
        "MATCH (:KnowledgeRelation {knowledge_base_id: $knowledge_base_id})-[:SUPPORTED_BY]->"
        "(e:KnowledgeEvidence {knowledge_base_id: $knowledge_base_id}) "
        "RETURN collect(DISTINCT e.chunk_id) AS values"
    )
    return GraphCoverageSnapshot(
        document_ids=document_ids,
        evidence_chunk_ids=evidence_chunks,
        entity_chunk_ids=entity_chunks,
        relation_chunk_ids=relation_chunks,
        entity_count=count(
            "MATCH (n:KnowledgeEntity {knowledge_base_id: $knowledge_base_id}) "
            "RETURN count(n) AS count"
        ),
        relation_count=count(
            "MATCH (n:KnowledgeRelation {knowledge_base_id: $knowledge_base_id}) "
            "RETURN count(n) AS count"
        ),
        evidence_count=count(
            "MATCH (n:KnowledgeEvidence {knowledge_base_id: $knowledge_base_id}) "
            "RETURN count(n) AS count"
        ),
        canonical_entity_count=count(
            "MATCH (n:CanonicalEntity {knowledge_base_id: $knowledge_base_id}) "
            "RETURN count(n) AS count"
        ),
        membership_count=count(
            "MATCH (:KnowledgeEntity {knowledge_base_id: $knowledge_base_id})-"
            "[r:CANONICAL_MEMBER_OF]->(:CanonicalEntity {knowledge_base_id: $knowledge_base_id}) "
            "RETURN count(r) AS count"
        ),
    )


def read_graph_coverage(store: Neo4jGraphStore, knowledge_base_id: UUID) -> GraphCoverageSnapshot:
    """Read graph coverage without changing graph state."""

    return store._read(lambda session: _graph_chunk_sets_from_session(session, knowledge_base_id))


def build_a21_graph_coverage(
    dataset_path: Path,
    *,
    source_block_to_chunks: Mapping[tuple[str, str], Sequence[str]],
    graph_supported_chunk_ids: Iterable[str],
) -> A21GraphCoverage:
    """Compute query-level graph coverage from frozen A2.1 evidence lineage."""

    records = [FinalDatasetItem.model_validate(record) for record in _read_jsonl(dataset_path)]
    graph_chunks = set(graph_supported_chunk_ids)
    any_count = 0
    complete_count = 0
    partial_count = 0
    no_count = 0
    relevant_all: set[str] = set()
    evidence_documents: set[str] = set()
    for record in records:
        evidence_documents.update(str(location.document_id) for location in record.item.evidence)
        relevant: set[str] = set()
        for location in record.item.evidence:
            document_id = str(location.document_id)
            for block_id in location.source_block_ids:
                relevant.update(source_block_to_chunks.get((document_id, block_id), ()))
        relevant_all.update(relevant)
        covered = relevant & graph_chunks
        if covered:
            any_count += 1
            if covered == relevant:
                complete_count += 1
            else:
                partial_count += 1
        else:
            no_count += 1
    return A21GraphCoverage(
        item_count=len(records),
        evidence_document_count=len(evidence_documents),
        relevant_chunk_count=len(relevant_all),
        queries_with_any_graph_coverage=any_count,
        queries_with_complete_graph_coverage=complete_count,
        queries_with_partial_graph_coverage=partial_count,
        queries_with_no_graph_coverage=no_count,
    )


def audit_corpus_files(
    knowledge_base_id: UUID,
    *,
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    chunk_index: Path = DEFAULT_CHUNK_INDEX,
    eval_dataset: Path = DEFAULT_EVAL_DATASET,
    registered_document_ids: Iterable[UUID] | None = None,
    graph_snapshot: GraphCoverageSnapshot | None = None,
    expected_snapshot: CorpusInputSnapshot | None = None,
    expected_document_count: int | None = None,
    expected_chunk_count: int | None = None,
) -> CorpusAudit:
    """Audit files and optional current graph state without mutating anything."""

    all_manifest_records = _read_jsonl(corpus_manifest)
    metadata = load_corpus_metadata(corpus_manifest)
    metadata_by_id = {item.document_id: item for item in metadata}
    if expected_snapshot is not None:
        actual_snapshot = build_corpus_input_snapshot(
            iter_corpus_documents(corpus_manifest, chunk_index),
            knowledge_base_id=knowledge_base_id,
            corpus_manifest=corpus_manifest,
            chunk_index=chunk_index,
        )
        validate_corpus_input_snapshot(
            actual_snapshot,
            expected_snapshot,
            require_full=True,
        )
        expected_document_count = expected_snapshot.document_count
        expected_chunk_count = expected_snapshot.chunk_count
    registered_ids = set(registered_document_ids) if registered_document_ids is not None else None
    counts_by_document: Counter[UUID] = Counter()
    source_block_to_chunks: dict[tuple[str, str], list[str]] = {}
    for chunk in iter_indexed_chunks(chunk_index):
        counts_by_document[chunk.document_id] += 1
        for block_id in chunk.source_block_ids:
            source_block_to_chunks.setdefault((str(chunk.document_id), block_id), []).append(
                chunk.chunk_id
            )
    if expected_document_count is None:
        expected_document_count = len(metadata_by_id)
    if expected_chunk_count is None:
        expected_chunk_count = sum(counts_by_document.values())
    missing_chunks = sorted(set(metadata_by_id) - set(counts_by_document), key=str)
    chunks_without_manifest = sorted(set(counts_by_document) - set(metadata_by_id), key=str)
    coverage = build_a21_graph_coverage(
        eval_dataset,
        source_block_to_chunks=source_block_to_chunks,
        graph_supported_chunk_ids=(graph_snapshot.evidence_chunk_ids if graph_snapshot else ()),
    )
    return CorpusAudit(
        knowledge_base_id=knowledge_base_id,
        expected_document_count=expected_document_count,
        expected_chunk_count=expected_chunk_count,
        manifest_rows=len(all_manifest_records),
        ready_manifest_rows=sum(
            record.get("inventory_status") == "ready" for record in all_manifest_records
        ),
        unique_manifest_documents=len(metadata),
        chunk_count=sum(counts_by_document.values()),
        chunk_documents=len(counts_by_document),
        subjects=sorted({item.subject for item in metadata}),
        registered_document_count=(len(registered_ids) if registered_ids is not None else None),
        registered_documents_without_manifest=(
            sorted(registered_ids - set(metadata_by_id), key=str)
            if registered_ids is not None
            else []
        ),
        manifest_documents_not_registered=(
            sorted(set(metadata_by_id) - registered_ids, key=str)
            if registered_ids is not None
            else []
        ),
        documents_without_chunks=missing_chunks,
        chunks_without_manifest_document=chunks_without_manifest,
        graph_status=("available" if graph_snapshot is not None else "not_requested"),
        graph_document_count=(len(graph_snapshot.document_ids) if graph_snapshot else None),
        graph_evidence_count=(graph_snapshot.evidence_count if graph_snapshot else None),
        graph_entity_count=(graph_snapshot.entity_count if graph_snapshot else None),
        graph_relation_count=(graph_snapshot.relation_count if graph_snapshot else None),
        graph_canonical_entity_count=(
            graph_snapshot.canonical_entity_count if graph_snapshot else None
        ),
        graph_membership_count=(graph_snapshot.membership_count if graph_snapshot else None),
        graph_supported_chunks=(len(graph_snapshot.evidence_chunk_ids) if graph_snapshot else None),
        graph_chunks_with_entities=(
            len(graph_snapshot.entity_chunk_ids) if graph_snapshot else None
        ),
        graph_chunks_with_relations=(
            len(graph_snapshot.relation_chunk_ids) if graph_snapshot else None
        ),
        a21_graph_coverage=coverage,
    )


def estimate_full_run(
    summary_paths: Sequence[Path] = DEFAULT_ESTIMATE_SUMMARIES,
    *,
    target_chunks: int | None = None,
    settings: Settings | None = None,
    max_concurrency: int = 1,
) -> CostEstimate:
    """Extrapolate only from existing measured A3.2 sample summaries."""

    if target_chunks is None:
        raise GraphCorpusBuildError(
            "target_chunks must be supplied from the authoritative input snapshot"
        )
    if target_chunks < 1 or max_concurrency < 1:
        raise ValueError("target_chunks and max_concurrency must be at least one")
    summaries: list[dict[str, Any]] = []
    for path in summary_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GraphCorpusBuildError(
                f"sample summary is invalid or unreadable: {path.name}"
            ) from error
        if not isinstance(value, dict):
            raise GraphCorpusBuildError(f"sample summary is not an object: {path.name}")
        summaries.append(value)
    sample_chunks = sum(int(value.get("attempted_chunks", 0)) for value in summaries)
    provider_calls = sum(int(value.get("provider_calls", 0)) for value in summaries)
    retry_calls = max(0, provider_calls - sample_chunks)
    input_values = [value.get("input_tokens") for value in summaries]
    output_values = [value.get("output_tokens") for value in summaries]
    input_tokens = (
        sum(int(value) for value in input_values)
        if all(value is not None for value in input_values)
        else None
    )
    output_tokens = (
        sum(int(value) for value in output_values)
        if all(value is not None for value in output_values)
        else None
    )
    wall_values = [
        float(value.get("run_elapsed_ms", value.get("elapsed_ms", 0.0))) for value in summaries
    ]
    wall_clock_seconds = sum(wall_values) / 1000
    if sample_chunks < 1:
        raise GraphCorpusBuildError("sample summaries contain no attempted chunks")
    retry_rate = retry_calls / sample_chunks
    expected_calls = round(target_chunks * provider_calls / sample_chunks)
    expected_retries = max(0, expected_calls - target_chunks)
    expected_input = (
        round(target_chunks * input_tokens / sample_chunks) if input_tokens is not None else None
    )
    expected_output = (
        round(target_chunks * output_tokens / sample_chunks) if output_tokens is not None else None
    )
    estimated = (
        estimate_cost(expected_input, expected_output, settings)
        if settings is not None and expected_input is not None and expected_output is not None
        else None
    )
    sequential_seconds = target_chunks * wall_clock_seconds / sample_chunks
    return CostEstimate(
        target_chunks=target_chunks,
        sample_files=[path.name for path in summary_paths],
        measured_sample_chunks=sample_chunks,
        measured_provider_calls=provider_calls,
        measured_retry_calls=retry_calls,
        measured_input_tokens=input_tokens,
        measured_output_tokens=output_tokens,
        measured_wall_clock_seconds=round(wall_clock_seconds, 3),
        observed_retry_rate=round(retry_rate, 6),
        expected_provider_calls_at_observed_rate=expected_calls,
        expected_retry_calls_at_observed_rate=expected_retries,
        expected_input_tokens_at_observed_rate=expected_input,
        expected_output_tokens_at_observed_rate=expected_output,
        estimated_cost_at_observed_rate=estimated,
        pricing_configured=(
            settings is not None
            and settings.llm_input_cost_per_1k_tokens is not None
            and settings.llm_output_cost_per_1k_tokens is not None
        ),
        sequential_wall_clock_seconds=round(sequential_seconds, 3),
        bounded_wall_clock_seconds_at_max_concurrency=round(
            sequential_seconds / max_concurrency, 3
        ),
        max_concurrency=max_concurrency,
        extrapolation_note=(
            f"Measured A3.2 sample totals are extrapolated linearly to {target_chunks:,} chunks; "
            "provider rate limits, document skew, failures and Neo4j write time are not modeled."
        ),
    )


def _add_optional(current: int | None, value: int | None) -> int | None:
    """Add a known total while preserving an unknown observation."""

    if current is None or value is None:
        return None
    return current + value


def _add_cost(current: Decimal | None, value: Decimal | None) -> Decimal | None:
    """Add configured cost while leaving an unconfigured total unknown."""

    if value is None:
        return None
    return value if current is None else current + value


def _extraction_attempt_counts(
    result: ChunkExtractionResult,
) -> tuple[int, int, int, int, int]:
    """Return logical, provider, truncation, corrective, and provider-retry calls."""

    logical_calls = max(result.stats.attempts, len(result.llm_results))
    provider_calls = sum(
        max(1, getattr(value, "provider_attempts", 1)) for value in result.llm_results
    )
    if provider_calls == 0 and logical_calls:
        provider_calls = logical_calls
    truncation_calls = result.stats.truncation_retries
    corrective_calls = max(0, logical_calls - 1 - truncation_calls)
    provider_retry_calls = max(0, provider_calls - logical_calls)
    return logical_calls, provider_calls, truncation_calls, corrective_calls, provider_retry_calls


def _extraction_error_attempt_counts(
    error: BaseException,
) -> tuple[int, int, int, int, int]:
    attempts = error.attempts if isinstance(error, ExtractionError) else ()
    logical_calls = len(attempts)
    provider_calls = sum(max(0, attempt.provider_attempts) for attempt in attempts)
    if provider_calls == 0 and logical_calls:
        provider_calls = logical_calls
    return logical_calls, provider_calls, 0, 0, max(0, provider_calls - logical_calls)


def _schema_rejection_diagnostics(
    result: ChunkExtractionResult,
) -> tuple[str | None, str | None]:
    """Keep bounded, application-owned diagnostics for a terminal rejection."""

    if result.status != "schema_rejected" or not result.attempts:
        return None, None
    final_attempt = result.attempts[-1]
    details = "; ".join(final_attempt.details)[:500]
    return final_attempt.category, details or final_attempt.category


def _merge_entities(entities: Iterable[GraphEntity]) -> tuple[GraphEntity, ...]:
    grouped: dict[str, list[GraphEntity]] = {}
    for entity in entities:
        grouped.setdefault(entity.entity_id, []).append(entity)
    merged: list[GraphEntity] = []
    for entity_id in sorted(grouped):
        values = grouped[entity_id]
        first = values[0]
        if any(
            value.knowledge_base_id != first.knowledge_base_id
            or value.document_id != first.document_id
            or value.canonical_name != first.canonical_name
            or value.entity_type != first.entity_type
            for value in values[1:]
        ):
            raise GraphCorpusBuildError("one local entity ID has conflicting identity fields")
        aliases = sorted({alias for value in values for alias in value.aliases})
        provenance = {evidence_id_for(item): item for value in values for item in value.provenance}
        merged.append(
            GraphEntity(
                entity_id=first.entity_id,
                knowledge_base_id=first.knowledge_base_id,
                document_id=first.document_id,
                canonical_name=first.canonical_name,
                entity_type=first.entity_type,
                aliases=aliases,
                provenance=list(provenance.values()),
            )
        )
    return tuple(merged)


def _bounded_excerpt(value: str, limit: int = 280) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit] + ("…" if len(normalized) > limit else "")


def _extraction_chunk(chunk: IndexedChunk) -> Chunk:
    """Convert the runtime index record to the canonical extraction input type."""

    try:
        payload = chunk.model_dump(mode="json")
        payload.pop("config_fingerprint", None)
        return Chunk.model_validate(payload)
    except ValueError as error:
        raise GraphCorpusBuildError("A1.6 chunk cannot be used as extraction input") from error


class GraphCorpusRunner:
    """Run bounded extraction/linking windows with durable resume semantics."""

    def __init__(
        self,
        extraction_service: ExtractionRunner,
        settings: Settings,
        *,
        knowledge_base_id: UUID,
        output_dir: Path = DEFAULT_OUTPUT,
        store: GraphCorpusStore | None = None,
        linking_gateway: LinkingGateway | None = None,
        max_concurrency: int = 1,
        link_batch_documents: int = 8,
        resume: bool = True,
        retry_failed: bool = False,
        reprocess: bool = False,
        input_snapshot: CorpusInputSnapshot | None = None,
    ) -> None:
        if not 1 <= max_concurrency <= 8:
            raise ValueError("max_concurrency must be between one and eight")
        if not 1 <= link_batch_documents <= 64:
            raise ValueError("link_batch_documents must be between one and 64")
        self.extraction_service = extraction_service
        self.settings = settings
        self.knowledge_base_id = knowledge_base_id
        self.output_dir = output_dir
        self.store = store
        self.linking_gateway = linking_gateway
        self.max_concurrency = max_concurrency
        self.link_batch_documents = link_batch_documents
        self.retry_failed = retry_failed
        self.reprocess = reprocess
        self.input_snapshot = input_snapshot
        self._observed_document_ids: set[UUID] = set()
        self._observed_chunk_count = 0
        self._observed_document_fingerprints: dict[UUID, str] = {}
        self._observed_input_documents: dict[UUID, CorpusInputDocument] = {}
        self._provider_blocked = False
        self._fatal_provider_signature: tuple[str, int] | None = None
        self._fatal_provider_failure_count = 0
        if (
            input_snapshot is not None
            and input_snapshot.document_count > 0
            and not input_snapshot.documents
        ):
            raise GraphCorpusBuildError(
                "GraphCorpusRunner requires a materialized input snapshot; "
                "validate the repository authority before constructing the runner"
            )
        self._run_lock = _RunOwnershipLock(output_dir / "run.lock")
        try:
            self._initialize_run(resume=resume)
        except BaseException:
            self.close()
            raise

    def _initialize_run(self, *, resume: bool) -> None:
        self.checkpoints = CheckpointStore(self.output_dir)
        self.manifest_store = _ManifestStore(self.output_dir)
        self.pipeline_fingerprint = build_pipeline_fingerprint(
            self.settings,
            knowledge_base_id=self.knowledge_base_id,
            input_fingerprint=(
                self.input_snapshot.input_fingerprint if self.input_snapshot is not None else None
            ),
            max_concurrency=self.max_concurrency,
            link_batch_documents=self.link_batch_documents,
            linking_mode=(
                "llm_adjudication" if self.linking_gateway is not None else "manual_review_only"
            ),
            retry_failed=self.retry_failed,
            reprocess=self.reprocess,
        )
        existing = self.manifest_store.load()
        if (
            self.input_snapshot is not None
            and self.input_snapshot.knowledge_base_id is not None
            and self.input_snapshot.knowledge_base_id != self.knowledge_base_id
        ):
            raise GraphCorpusBuildError("input snapshot Knowledge Base does not match the run")
        if existing is None and self.checkpoints.has_records():
            raise GraphCorpusBuildError(
                "checkpoint records exist without a run manifest; use a new output directory"
            )
        if existing is not None and not resume:
            raise GraphCorpusBuildError(
                "a run already exists in this output directory; "
                "use resume or a new output directory"
            )
        if existing is not None:
            if existing.knowledge_base_id != self.knowledge_base_id:
                raise GraphCorpusBuildError("resume Knowledge Base does not match the run manifest")
            if existing.pipeline_fingerprint != self.pipeline_fingerprint:
                raise GraphCorpusBuildError(
                    "resume pipeline fingerprint differs; use a new output directory"
                )
            stored_prompt_layout = (
                existing.prompt_layout_version or LEGACY_EXTRACTION_PROMPT_LAYOUT_VERSION
            )
            if stored_prompt_layout != EXTRACTION_PROMPT_LAYOUT_VERSION:
                raise GraphCorpusBuildError(
                    "resume prompt serialization layout differs; use a new output directory"
                )
            if self.input_snapshot is not None and (
                existing.input_fingerprint != self.input_snapshot.input_fingerprint
                or existing.expected_document_ids
                != list(_snapshot_document_ids(self.input_snapshot))
                or existing.expected_document_count != self.input_snapshot.document_count
                or existing.expected_chunk_count != self.input_snapshot.chunk_count
            ):
                raise GraphCorpusBuildError(
                    "resume corpus input snapshot differs; use a new output directory"
                )
            self.manifest = existing
        else:
            self.manifest = RunManifest(
                run_id=uuid4(),
                knowledge_base_id=self.knowledge_base_id,
                pipeline_version=GRAPH_CORPUS_PIPELINE_VERSION,
                pipeline_fingerprint=self.pipeline_fingerprint,
                pipeline_versions={
                    "graph_corpus_build": GRAPH_CORPUS_PIPELINE_VERSION,
                    "graph_schema": GRAPH_SCHEMA_VERSION,
                    "extraction_prompt": EXTRACTION_PROMPT_VERSION,
                    "extraction_schema": "graph-extraction-v1.3",
                    "linking_prompt": LINKING_PROMPT_VERSION,
                    "linking_schema": LINKING_SCHEMA_VERSION,
                },
                provider=self.settings.llm_provider,
                model=self.settings.llm_model,
                started_at=_now(),
                status="running",
                prompt_layout_version=EXTRACTION_PROMPT_LAYOUT_VERSION,
                input_fingerprint=(
                    self.input_snapshot.input_fingerprint
                    if self.input_snapshot is not None
                    else None
                ),
                expected_document_count=(
                    self.input_snapshot.document_count if self.input_snapshot is not None else None
                ),
                expected_chunk_count=(
                    self.input_snapshot.chunk_count if self.input_snapshot is not None else None
                ),
                expected_document_ids=(
                    list(_snapshot_document_ids(self.input_snapshot))
                    if self.input_snapshot is not None
                    else []
                ),
            )
            self.manifest_store.save(self.manifest)

    def close(self) -> None:
        """Release the local run lock; safe to call more than once."""

        run_lock = getattr(self, "_run_lock", None)
        if run_lock is not None:
            run_lock.release()

    def _record_provider_success(self) -> None:
        """Reset consecutive fatal-provider evidence after a valid provider response."""

        if not self._provider_blocked:
            self._fatal_provider_signature = None
            self._fatal_provider_failure_count = 0

    def _record_provider_failure(self, error: BaseException) -> None:
        """Trip only on bounded, repeated, explicit non-retryable provider 4xx errors."""

        signature = _fatal_provider_signature(error)
        if signature is None or self._provider_blocked:
            if not self._provider_blocked:
                self._fatal_provider_signature = None
                self._fatal_provider_failure_count = 0
            return
        if signature == self._fatal_provider_signature:
            self._fatal_provider_failure_count += 1
        else:
            self._fatal_provider_signature = signature
            self._fatal_provider_failure_count = 1
        if self._fatal_provider_failure_count < DEFAULT_PROVIDER_BLOCK_THRESHOLD:
            return
        self._provider_blocked = True
        self.manifest.provider_blocked = True
        self.manifest.provider_block_reason = (
            f"repeated non-retryable provider API error HTTP {signature[1]}"
        )
        self.manifest.provider_blocked_at = _now()
        self.manifest.status = "provider_blocked"
        self.manifest_store.save(self.manifest)

    def _document_fingerprint(self, document: CorpusDocumentChunks) -> str:
        return _document_fingerprint_for(document)

    def _link_batch_key(
        self,
        documents: Sequence[CorpusDocumentChunks],
        entities: Sequence[GraphEntity],
    ) -> str:
        return _sha256_json(
            {
                "knowledge_base_id": str(self.knowledge_base_id),
                "document_ids": sorted(
                    str(document.metadata.document_id) for document in documents
                ),
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "aliases": sorted(entity.aliases),
                        "evidence_ids": sorted(evidence_id_for(item) for item in entity.provenance),
                    }
                    for entity in sorted(entities, key=lambda value: value.entity_id)
                ],
                "pipeline_fingerprint": self.pipeline_fingerprint,
                "linking_mode": "llm_adjudication"
                if self.linking_gateway is not None
                else "manual_review_only",
            }
        )

    def _observe_document(self, document: CorpusDocumentChunks) -> None:
        document_id = document.metadata.document_id
        if document_id in self._observed_document_ids:
            raise GraphCorpusBuildError("input stream contains a duplicate document")
        if not document.chunks:
            raise GraphCorpusBuildError("input stream contains an empty document")
        if any(chunk.document_id != document_id for chunk in document.chunks):
            raise GraphCorpusBuildError("document contains a chunk from another document")
        if [chunk.ordinal for chunk in document.chunks] != list(range(len(document.chunks))):
            raise GraphCorpusBuildError("document chunk ordinals are not contiguous")
        self._observed_document_ids.add(document_id)
        self._observed_chunk_count += len(document.chunks)
        fingerprint = self._document_fingerprint(document)
        self._observed_document_fingerprints[document_id] = fingerprint
        try:
            self._observed_input_documents[document_id] = CorpusInputDocument(
                document_id=document_id,
                benchmark_item_id=document.metadata.benchmark_item_id,
                subject=document.metadata.subject,
                relative_path=document.metadata.relative_path,
                size_bytes=document.metadata.size_bytes,
                sha256=document.metadata.sha256,
                chunk_count=len(document.chunks),
                chunk_ids=[chunk.chunk_id for chunk in document.chunks],
                document_fingerprint=fingerprint,
            )
        except ValueError as error:
            raise GraphCorpusBuildError("document input metadata is invalid") from error
        if self.input_snapshot is not None:
            expected = {entry.document_id: entry for entry in self.input_snapshot.documents}.get(
                document_id
            )
            if expected is None and document_id not in _snapshot_document_ids(self.input_snapshot):
                raise GraphCorpusBuildError(
                    f"document input is outside the supplied snapshot: {document_id}"
                )
            if expected is not None and expected.document_fingerprint != fingerprint:
                raise GraphCorpusBuildError(
                    f"document input differs from the supplied snapshot: {document_id}"
                )

    def _validate_observed_input(self) -> None:
        self.manifest.observed_document_count = len(self._observed_document_ids)
        self.manifest.observed_chunk_count = self._observed_chunk_count
        if self.input_snapshot is None:
            return
        expected_ids = set(_snapshot_document_ids(self.input_snapshot))
        observed_entries = tuple(
            self._observed_input_documents[document_id]
            for document_id in sorted(self._observed_input_documents, key=str)
        )
        if (
            self._observed_document_ids != expected_ids
            or len(self._observed_document_ids) != self.input_snapshot.document_count
            or self._observed_chunk_count != self.input_snapshot.chunk_count
        ):
            self.manifest.input_validation_error = "observed input is incomplete"
            raise GraphCorpusBuildError(
                "observed corpus does not match the supplied complete input snapshot"
            )
        if (
            self.input_snapshot.input_fingerprint is not None
            and _corpus_input_fingerprint(observed_entries) != self.input_snapshot.input_fingerprint
        ):
            self.manifest.input_validation_error = "observed input fingerprint differs"
            raise GraphCorpusBuildError(
                "observed corpus fingerprint differs from the supplied snapshot"
            )

    def _mark_processed_chunk(self, chunk_id: str, *, failed: bool = False) -> None:
        if chunk_id not in self.manifest.processed_chunk_ids:
            self.manifest.processed_chunk_ids.append(chunk_id)
        if failed:
            if chunk_id not in self.manifest.failed_chunk_ids:
                self.manifest.failed_chunk_ids.append(chunk_id)
        else:
            self.manifest.failed_chunk_ids = [
                value for value in self.manifest.failed_chunk_ids if value != chunk_id
            ]

    def _mark_processed_document(self, document_id: UUID, *, failed: bool = False) -> None:
        if document_id not in self.manifest.processed_document_ids:
            self.manifest.processed_document_ids.append(document_id)
        if failed:
            if document_id not in self.manifest.failed_document_ids:
                self.manifest.failed_document_ids.append(document_id)
        else:
            self.manifest.failed_document_ids = [
                value for value in self.manifest.failed_document_ids if value != document_id
            ]

    def _add_extraction_counters(self, result: ChunkExtractionResult) -> None:
        counters = self.manifest.counters
        (
            logical_calls,
            provider_calls,
            truncation_calls,
            corrective_calls,
            provider_retry_calls,
        ) = _extraction_attempt_counts(result)
        counters.provider_calls += provider_calls
        counters.retry_calls += truncation_calls + corrective_calls + provider_retry_calls
        counters.extraction_initial_calls += min(1, logical_calls)
        counters.extraction_provider_retry_calls += provider_retry_calls
        counters.extraction_truncation_retry_calls += truncation_calls
        counters.extraction_corrective_retry_calls += corrective_calls
        counters.entities += len(result.entities)
        counters.relations += len(result.relations)
        counters.evidences += len(
            {
                evidence_id_for(provenance)
                for fact in (*result.entities, *result.relations)
                for provenance in fact.provenance
            }
        )
        counters.grounding_rejections += result.stats.grounding_rejections
        if result.llm_results:
            counters.input_tokens = _add_optional(counters.input_tokens, result.input_tokens)
            counters.output_tokens = _add_optional(counters.output_tokens, result.output_tokens)
            cost = result.estimated_cost(self.settings)
            if cost is None:
                counters.estimated_cost_complete = False
                counters.estimated_cost = None
            elif counters.estimated_cost_complete:
                counters.estimated_cost = _add_cost(counters.estimated_cost, cost)
        counters.latency_ms += result.latency_ms
        if result.status == "accepted":
            counters.accepted_extractions += 1
            counters.structured_parse_successes += 1
        elif result.status == "empty":
            counters.empty_extractions += 1
            counters.structured_parse_successes += 1
        elif result.status == "rejected":
            counters.rejected_extractions += 1
            counters.structured_parse_successes += 1
        elif result.status == "schema_rejected":
            counters.schema_rejected_extractions += 1

    def _add_failed_extraction_counters(self, checkpoint: ChunkCheckpoint) -> None:
        counters = self.manifest.counters
        counters.provider_calls += checkpoint.provider_calls
        counters.retry_calls += checkpoint.retry_count
        counters.extraction_initial_calls += checkpoint.extraction_initial_calls
        counters.extraction_provider_retry_calls += checkpoint.extraction_provider_retry_calls
        counters.extraction_truncation_retry_calls += checkpoint.extraction_truncation_retry_calls
        counters.extraction_corrective_retry_calls += checkpoint.extraction_corrective_retry_calls
        counters.extraction_corpus_retry_calls += checkpoint.extraction_corpus_retry_calls

    def _checkpoint_from_result(
        self,
        chunk: IndexedChunk,
        result: ChunkExtractionResult,
    ) -> ChunkCheckpoint:
        (
            logical_calls,
            provider_calls,
            truncation_calls,
            corrective_calls,
            provider_retry_calls,
        ) = _extraction_attempt_counts(result)
        error_category, error = _schema_rejection_diagnostics(result)
        return ChunkCheckpoint(
            knowledge_base_id=self.knowledge_base_id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            input_fingerprint=chunk_input_fingerprint(chunk),
            config_fingerprint=chunk.config_fingerprint,
            pipeline_fingerprint=self.pipeline_fingerprint,
            status=(
                "completed" if result.status in {"accepted", "empty", "rejected"} else "failed"
            ),
            extraction_status=result.status,
            attempts=logical_calls,
            retry_count=truncation_calls + corrective_calls + provider_retry_calls,
            provider_calls=provider_calls,
            structured_success=result.status in {"accepted", "empty", "rejected"},
            extraction_initial_calls=min(1, logical_calls),
            extraction_provider_retry_calls=provider_retry_calls,
            extraction_truncation_retry_calls=truncation_calls,
            extraction_corrective_retry_calls=corrective_calls,
            grounding_rejections=result.stats.grounding_rejections,
            entity_count=len(result.entities),
            relation_count=len(result.relations),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost=result.estimated_cost(self.settings),
            latency_ms=result.latency_ms,
            error_category=error_category,
            error=error,
        )

    def _checkpoint_from_error(
        self,
        chunk: IndexedChunk,
        error: BaseException,
        result: ChunkExtractionResult | None = None,
    ) -> ChunkCheckpoint:
        if result is not None:
            (
                attempts,
                provider_calls,
                truncation_calls,
                corrective_calls,
                provider_retry_calls,
            ) = _extraction_attempt_counts(result)
        else:
            (
                attempts,
                provider_calls,
                truncation_calls,
                corrective_calls,
                provider_retry_calls,
            ) = _extraction_error_attempt_counts(error)
        error_category = error.category if isinstance(error, ExtractionError) else "persistence"
        message = str(error).strip().replace("\n", " ")[:500] or type(error).__name__
        return ChunkCheckpoint(
            knowledge_base_id=self.knowledge_base_id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            input_fingerprint=chunk_input_fingerprint(chunk),
            config_fingerprint=chunk.config_fingerprint,
            pipeline_fingerprint=self.pipeline_fingerprint,
            status="failed",
            extraction_status=result.status if result is not None else None,
            attempts=attempts,
            retry_count=truncation_calls + corrective_calls + provider_retry_calls,
            provider_calls=provider_calls,
            structured_success=(
                result.status in {"accepted", "empty", "rejected"} if result is not None else False
            ),
            extraction_initial_calls=min(1, attempts),
            extraction_provider_retry_calls=provider_retry_calls,
            extraction_truncation_retry_calls=truncation_calls,
            extraction_corrective_retry_calls=corrective_calls,
            grounding_rejections=result.stats.grounding_rejections if result is not None else 0,
            entity_count=len(result.entities) if result is not None else 0,
            relation_count=len(result.relations) if result is not None else 0,
            input_tokens=result.input_tokens if result is not None else None,
            output_tokens=result.output_tokens if result is not None else None,
            estimated_cost=result.estimated_cost(self.settings) if result is not None else None,
            latency_ms=result.latency_ms if result is not None else None,
            error_category=error_category,
            error=message,
        )

    def _with_corpus_retry(
        self,
        checkpoint: ChunkCheckpoint,
        previous: ChunkCheckpoint | None,
    ) -> ChunkCheckpoint:
        corpus_retry = int(
            self.retry_failed and previous is not None and previous.status == "failed"
        )
        return checkpoint.model_copy(
            update={
                "extraction_corpus_retry_calls": corpus_retry,
                "retry_count": checkpoint.retry_count + corpus_retry,
            }
        )

    async def _process_chunk(
        self,
        chunk: IndexedChunk,
        semaphore: asyncio.Semaphore,
        *,
        force_reprocess: bool = False,
    ) -> _ChunkOutcome:
        previous = self.checkpoints.get_chunk(chunk.chunk_id)
        input_fingerprint = chunk_input_fingerprint(chunk)
        same_inputs = (
            previous is not None
            and previous.knowledge_base_id == self.knowledge_base_id
            and previous.document_id == chunk.document_id
            and previous.input_fingerprint == input_fingerprint
            and previous.config_fingerprint == chunk.config_fingerprint
            and previous.pipeline_fingerprint == self.pipeline_fingerprint
        )
        if (
            not force_reprocess
            and same_inputs
            and previous is not None
            and (
                (previous.status == "completed" and previous.structured_success)
                or (previous.status == "failed" and not self.retry_failed)
            )
        ):
            return _ChunkOutcome(chunk=chunk, checkpoint=previous, result=None, reused=True)
        try:
            extraction_chunk = _extraction_chunk(chunk)
        except GraphCorpusBuildError as error:
            checkpoint = self._checkpoint_from_error(chunk, error)
            checkpoint.error_category = "input_validation"
            self.checkpoints.save_chunk(checkpoint)
            return _ChunkOutcome(chunk=chunk, checkpoint=checkpoint, result=None)
        async with semaphore:
            result: ChunkExtractionResult | None = None
            try:
                result = await self.extraction_service.extract_chunk(
                    extraction_chunk,
                    knowledge_base_id=self.knowledge_base_id,
                )
                self._record_provider_success()
                if self.store is not None and result.status == "accepted":
                    await asyncio.to_thread(
                        self.store.upsert_extraction,
                        result.entities,
                        result.relations,
                    )
                checkpoint = self._with_corpus_retry(
                    self._checkpoint_from_result(chunk, result),
                    previous,
                )
                self.checkpoints.save_chunk(checkpoint)
                return _ChunkOutcome(chunk=chunk, checkpoint=checkpoint, result=result)
            except asyncio.CancelledError as error:
                checkpoint = self._checkpoint_from_error(chunk, error)
                checkpoint.status = "cancelled"
                checkpoint.error_category = "cancelled"
                self.checkpoints.save_chunk(checkpoint)
                raise
            except (ExtractionError, GraphStoreError) as error:
                if isinstance(error, ExtractionError):
                    self._record_provider_failure(error)
                result = error.result if isinstance(error, ExtractionPersistenceError) else result
                checkpoint = self._with_corpus_retry(
                    self._checkpoint_from_error(chunk, error, result),
                    previous,
                )
                self.checkpoints.save_chunk(checkpoint)
                return _ChunkOutcome(chunk=chunk, checkpoint=checkpoint, result=result)
            except Exception as error:  # isolate one chunk without exposing provider payloads
                checkpoint = self._checkpoint_from_error(chunk, error, result)
                checkpoint.error_category = "unexpected"
                checkpoint.error = type(error).__name__
                self.checkpoints.save_chunk(checkpoint)
                return _ChunkOutcome(chunk=chunk, checkpoint=checkpoint, result=result)

    async def _process_document(
        self, document: CorpusDocumentChunks
    ) -> tuple[list[_ChunkOutcome], bool, bool]:
        document_id = document.metadata.document_id
        document_fingerprint = self._document_fingerprint(document)
        previous = self.checkpoints.get_document(document_id)
        same_inputs = (
            previous is not None
            and previous.knowledge_base_id == self.knowledge_base_id
            and previous.document_id == document_id
            and previous.input_fingerprint == document_fingerprint
            and previous.pipeline_fingerprint == self.pipeline_fingerprint
        )
        chunk_checkpoints = tuple(
            self.checkpoints.get_chunk(chunk.chunk_id) for chunk in document.chunks
        )
        all_chunks_reusable = all(
            checkpoint is not None
            and checkpoint.knowledge_base_id == self.knowledge_base_id
            and checkpoint.document_id == chunk.document_id
            and checkpoint.input_fingerprint == chunk_input_fingerprint(chunk)
            and checkpoint.config_fingerprint == chunk.config_fingerprint
            and checkpoint.pipeline_fingerprint == self.pipeline_fingerprint
            and checkpoint.status == "completed"
            and checkpoint.structured_success
            for chunk, checkpoint in zip(document.chunks, chunk_checkpoints, strict=True)
        )
        reusable = (
            same_inputs
            and previous is not None
            and previous.status == "completed"
            and previous.extraction_complete
            and not self.reprocess
            and all_chunks_reusable
        )
        if reusable:
            self.manifest.counters.documents_skipped += 1
            outcomes = [
                _ChunkOutcome(
                    chunk=chunk,
                    checkpoint=checkpoint,
                    result=None,
                    reused=True,
                )
                for chunk, checkpoint in zip(document.chunks, chunk_checkpoints, strict=True)
            ]
            return outcomes, False, False

        repair_completed_checkpoint = (
            previous is not None and previous.status == "completed" and not all_chunks_reusable
        )
        source_changed = previous is not None and not same_inputs
        persisted_state = False
        if self.store is not None and hasattr(self.store, "has_document_state"):
            persisted_state = await asyncio.to_thread(
                self.store.has_document_state,
                self.knowledge_base_id,
                document_id,
            )
        if previous is None and persisted_state and not self.reprocess:
            raise GraphCorpusBuildError(
                "existing graph state has no checkpoint in this directory; "
                "use --reprocess to replace it explicitly"
            )
        rebuild_graph = (
            self.reprocess
            or source_changed
            or repair_completed_checkpoint
            or (previous is None and persisted_state)
            or (previous is not None and previous.status == "cancelled" and persisted_state)
        )
        if self.store is not None and rebuild_graph:
            await asyncio.to_thread(
                self.store.delete_document,
                document_id,
                knowledge_base_id=self.knowledge_base_id,
            )
        self.manifest.counters.documents_attempted += 1
        semaphore = asyncio.Semaphore(self.max_concurrency)
        try:
            outcomes: list[_ChunkOutcome] = []
            for offset in range(0, len(document.chunks), self.max_concurrency):
                if self._provider_blocked:
                    break
                outcomes.extend(
                    await asyncio.gather(
                        *(
                            self._process_chunk(
                                chunk,
                                semaphore,
                                force_reprocess=(
                                    self.reprocess
                                    or rebuild_graph
                                    or previous is None
                                    or repair_completed_checkpoint
                                ),
                            )
                            for chunk in document.chunks[offset : offset + self.max_concurrency]
                        )
                    )
                )
                if self._provider_blocked:
                    break
        except asyncio.CancelledError:
            latest_checkpoints = tuple(
                self.checkpoints.get_chunk(chunk.chunk_id) for chunk in document.chunks
            )
            completed_count = sum(
                checkpoint is not None and checkpoint.status == "completed"
                for checkpoint in latest_checkpoints
            )
            failed_count = len(document.chunks) - completed_count
            self.checkpoints.save_document(
                DocumentCheckpoint(
                    knowledge_base_id=self.knowledge_base_id,
                    document_id=document_id,
                    input_fingerprint=document_fingerprint,
                    pipeline_fingerprint=self.pipeline_fingerprint,
                    chunk_count=len(document.chunks),
                    completed_chunk_count=completed_count,
                    failed_chunk_count=failed_count,
                    status="cancelled",
                    extraction_complete=False,
                    extraction_partial=True,
                    linking_complete=False,
                    linking_blocked=True,
                    error="document extraction cancelled",
                )
            )
            raise
        failed = any(outcome.checkpoint.status != "completed" for outcome in outcomes)
        completed_count = sum(outcome.checkpoint.status == "completed" for outcome in outcomes)
        failed_count = len(outcomes) - completed_count
        deferred_count = len(document.chunks) - len(outcomes)
        self.checkpoints.save_document(
            DocumentCheckpoint(
                knowledge_base_id=self.knowledge_base_id,
                document_id=document_id,
                input_fingerprint=document_fingerprint,
                pipeline_fingerprint=self.pipeline_fingerprint,
                chunk_count=len(document.chunks),
                completed_chunk_count=completed_count,
                failed_chunk_count=failed_count,
                deferred_chunk_count=deferred_count,
                status=(
                    "provider_blocked"
                    if self._provider_blocked
                    else "failed"
                    if failed
                    else "completed"
                ),
                extraction_complete=not failed and deferred_count == 0,
                extraction_partial=failed or deferred_count > 0,
                error=(
                    "provider blocked; remaining chunks were not scheduled"
                    if self._provider_blocked
                    else "one or more chunks failed"
                    if failed
                    else None
                ),
            )
        )
        return outcomes, failed, rebuild_graph

    async def _entities_for_document(
        self,
        document: CorpusDocumentChunks,
        outcomes: Sequence[_ChunkOutcome],
    ) -> tuple[GraphEntity, ...]:
        if self.store is not None and hasattr(self.store, "list_entities_for_document"):
            try:
                stored = await asyncio.to_thread(
                    self.store.list_entities_for_document,
                    self.knowledge_base_id,
                    document.metadata.document_id,
                )
            except GraphStoreError:
                raise
            return _merge_entities(stored)
        return _merge_entities(
            entity
            for outcome in outcomes
            if outcome.result is not None
            for entity in outcome.result.entities
        )

    async def _delete_linking_state(self, document_ids: Sequence[UUID]) -> None:
        if self.store is None or not document_ids:
            return
        if hasattr(self.store, "delete_documents_links"):
            await asyncio.to_thread(
                self.store.delete_documents_links,
                self.knowledge_base_id,
                tuple(document_ids),
            )
            return
        if hasattr(self.store, "delete_document_links"):
            for document_id in document_ids:
                await asyncio.to_thread(
                    self.store.delete_document_links,
                    self.knowledge_base_id,
                    document_id,
                )
            return
        raise GraphCorpusBuildError("graph store cannot clear stale linking state")

    def _set_document_linking_state(
        self,
        document_id: UUID,
        *,
        complete: bool,
        blocked: bool,
    ) -> None:
        checkpoint = self.checkpoints.get_document(document_id)
        if checkpoint is None:
            return
        self.checkpoints.save_document(
            checkpoint.model_copy(
                update={
                    "linking_complete": complete,
                    "linking_blocked": blocked,
                }
            )
        )

    def _add_linking_counters(self, linking_run: LinkingRun) -> None:
        """Add provider usage before persistence so failed writes remain observable."""

        counters = self.manifest.counters
        counters.linking_candidates += linking_run.stats.candidate_count
        counters.link_decisions += len(linking_run.decisions)
        counters.link_count += linking_run.stats.link_count
        counters.no_link_count += linking_run.stats.no_link_count
        counters.uncertain_count += linking_run.stats.uncertain_count
        counters.canonical_entities += len(linking_run.canonical_entities)
        counters.memberships += len(linking_run.mappings)
        counters.provider_calls += linking_run.stats.provider_attempts
        counters.retry_calls += linking_run.stats.provider_retry_calls
        counters.linking_initial_calls += linking_run.stats.llm_adjudications
        counters.linking_provider_retry_calls += linking_run.stats.provider_retry_calls
        counters.linking_failed_attempts += linking_run.stats.llm_failures
        if linking_run.stats.llm_adjudications:
            counters.input_tokens = _add_optional(
                counters.input_tokens,
                linking_run.stats.input_tokens,
            )
            counters.output_tokens = _add_optional(
                counters.output_tokens,
                linking_run.stats.output_tokens,
            )
            if linking_run.stats.estimated_cost is None:
                counters.estimated_cost_complete = False
                counters.estimated_cost = None
            elif counters.estimated_cost_complete:
                counters.estimated_cost = _add_cost(
                    counters.estimated_cost,
                    linking_run.stats.estimated_cost,
                )

    async def _link_batch(
        self,
        documents: Sequence[CorpusDocumentChunks],
        outcomes_by_document: Mapping[UUID, Sequence[_ChunkOutcome]],
        *,
        force_relink: bool = False,
    ) -> bool:
        if self.store is None:
            return True
        document_ids = sorted(
            (document.metadata.document_id for document in documents),
            key=str,
        )
        stale_failed_batch_keys = set(
            self.checkpoints.failed_link_batches_for_documents(document_ids)
        )
        if stale_failed_batch_keys:
            self.manifest.failed_link_batch_keys = [
                value
                for value in self.manifest.failed_link_batch_keys
                if value not in stale_failed_batch_keys
            ]
        started = asyncio.get_running_loop().time()
        try:
            entities: list[GraphEntity] = []
            for document in documents:
                entities.extend(
                    await self._entities_for_document(
                        document,
                        outcomes_by_document.get(document.metadata.document_id, ()),
                    )
                )
            merged_entities = _merge_entities(entities)
        except (GraphCorpusBuildError, GraphStoreError, ValueError) as error:
            batch_key = self._link_batch_key(documents, ())
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=(),
                run_id=uuid4(),
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            return False
        entity_ids = [entity.entity_id for entity in merged_entities]
        batch_key = self._link_batch_key(documents, merged_entities)
        previous = self.checkpoints.get_link_batch(batch_key)
        if previous is not None and previous.status == "completed" and not force_relink:
            self.manifest.counters.link_batches_skipped += 1
            return True
        if (
            previous is not None
            and previous.status == "failed"
            and not (self.retry_failed or force_relink)
        ):
            self.manifest.counters.link_batches_skipped += 1
            if batch_key not in self.manifest.failed_link_batch_keys:
                self.manifest.failed_link_batch_keys.append(batch_key)
            return False
        run_id = previous.run_id if previous is not None else uuid4()
        self.manifest.counters.link_batches_attempted += 1
        try:
            await self._delete_linking_state(document_ids)
        except asyncio.CancelledError:
            raise
        except (GraphCorpusBuildError, GraphStoreError, ValueError) as error:
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                count_attempt=False,
            )
            return False
        except Exception as error:  # isolate one linking window without raw provider output
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                count_attempt=False,
            )
            return False
        if len(merged_entities) < 2:
            self.checkpoints.save_link_batch(
                LinkBatchCheckpoint(
                    knowledge_base_id=self.knowledge_base_id,
                    batch_key=batch_key,
                    document_ids=document_ids,
                    entity_ids=entity_ids,
                    run_id=run_id,
                    status="completed",
                )
            )
            self.manifest.counters.link_batches_completed += 1
            return True

        chunk_text = {
            chunk.chunk_id: chunk.text for document in documents for chunk in document.chunks
        }
        contexts = tuple(
            LocalEntityContext(
                entity=entity,
                source_excerpt="\n".join(
                    _bounded_excerpt(chunk_text.get(item.chunk_id, ""))
                    for item in entity.provenance
                    if chunk_text.get(item.chunk_id, "").strip()
                ),
                subject=next(
                    (
                        document.metadata.subject
                        for document in documents
                        if document.metadata.document_id == entity.document_id
                    ),
                    None,
                ),
            )
            for entity in merged_entities
        )
        self.checkpoints.save_link_batch(
            LinkBatchCheckpoint(
                knowledge_base_id=self.knowledge_base_id,
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                status="cancelled",
                error="linking batch interrupted before completion",
            )
        )
        try:
            linking_run: LinkingRun = await link_entities(
                merged_entities,
                gateway=self.linking_gateway,
                settings=self.settings if self.linking_gateway is not None else None,
                contexts=contexts,
                run_id=run_id,
            )
        except asyncio.CancelledError:
            raise
        except (GraphStoreError, ValueError) as error:
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                count_attempt=False,
            )
            return False
        except Exception as error:  # isolate one linking window without raw provider output
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                count_attempt=False,
            )
            return False
        self._add_linking_counters(linking_run)
        try:
            await asyncio.to_thread(
                self.store.upsert_linking_result,
                linking_run.canonical_entities,
                linking_run.decisions,
                linking_run.mappings,
            )
        except asyncio.CancelledError:
            raise
        except (GraphStoreError, ValueError) as error:
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                linking_run=linking_run,
                count_attempt=False,
            )
            return False
        except Exception as error:  # isolate one linking window without raw provider output
            self._record_link_failure(
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                error=error,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
                linking_run=linking_run,
                count_attempt=False,
            )
            return False
        self.checkpoints.save_link_batch(
            LinkBatchCheckpoint(
                knowledge_base_id=self.knowledge_base_id,
                batch_key=batch_key,
                document_ids=document_ids,
                entity_ids=entity_ids,
                run_id=run_id,
                status="completed",
                candidate_count=linking_run.stats.candidate_count,
                link_count=linking_run.stats.link_count,
                no_link_count=linking_run.stats.no_link_count,
                uncertain_count=linking_run.stats.uncertain_count,
                canonical_entity_count=len(linking_run.canonical_entities),
                membership_count=len(linking_run.mappings),
                initial_calls=linking_run.stats.llm_adjudications,
                provider_retry_calls=linking_run.stats.provider_retry_calls,
                failed_attempts=linking_run.stats.llm_failures,
                input_tokens=linking_run.stats.input_tokens,
                output_tokens=linking_run.stats.output_tokens,
                estimated_cost=linking_run.stats.estimated_cost,
                latency_ms=max(0.0, (asyncio.get_running_loop().time() - started) * 1000),
            )
        )
        self.manifest.counters.link_batches_completed += 1
        self.manifest.failed_link_batch_keys = [
            value for value in self.manifest.failed_link_batch_keys if value != batch_key
        ]
        return True

    def _record_link_failure(
        self,
        *,
        batch_key: str,
        document_ids: Sequence[UUID],
        entity_ids: Sequence[str],
        run_id: UUID,
        error: BaseException,
        latency_ms: float,
        linking_run: LinkingRun | None = None,
        count_attempt: bool = True,
    ) -> None:
        if count_attempt:
            self.manifest.counters.link_batches_attempted += 1
        self.manifest.counters.link_batches_failed += 1
        if batch_key not in self.manifest.failed_link_batch_keys:
            self.manifest.failed_link_batch_keys.append(batch_key)
        self.checkpoints.save_link_batch(
            LinkBatchCheckpoint(
                knowledge_base_id=self.knowledge_base_id,
                batch_key=batch_key,
                document_ids=list(document_ids),
                entity_ids=list(entity_ids),
                run_id=run_id,
                status="failed",
                candidate_count=(linking_run.stats.candidate_count if linking_run else 0),
                link_count=(linking_run.stats.link_count if linking_run else 0),
                no_link_count=(linking_run.stats.no_link_count if linking_run else 0),
                uncertain_count=(linking_run.stats.uncertain_count if linking_run else 0),
                canonical_entity_count=(len(linking_run.canonical_entities) if linking_run else 0),
                membership_count=(len(linking_run.mappings) if linking_run else 0),
                initial_calls=(linking_run.stats.llm_adjudications if linking_run else 0),
                provider_retry_calls=(linking_run.stats.provider_retry_calls if linking_run else 0),
                failed_attempts=(linking_run.stats.llm_failures if linking_run else 0),
                input_tokens=(linking_run.stats.input_tokens if linking_run else None),
                output_tokens=(linking_run.stats.output_tokens if linking_run else None),
                estimated_cost=(linking_run.stats.estimated_cost if linking_run else None),
                error=(
                    str(error).replace("\n", " ")[:500]
                    if isinstance(error, (GraphCorpusBuildError, GraphStoreError, ExtractionError))
                    else type(error).__name__
                ),
                latency_ms=max(0.0, latency_ms),
            )
        )

    async def run(self, documents: Iterable[CorpusDocumentChunks]) -> RunManifest:
        """Process a bounded document iterator and persist progress after each unit."""

        try:
            self.manifest.status = "running"
            self._provider_blocked = False
            self._fatal_provider_signature = None
            self._fatal_provider_failure_count = 0
            self.manifest.provider_blocked = False
            self.manifest.provider_block_reason = None
            self.manifest.provider_blocked_at = None
            self.manifest.ended_at = None
            self.manifest.observed_document_count = 0
            self.manifest.observed_chunk_count = 0
            self.manifest.input_validation_error = None
            self.manifest.run_error = None
            self.manifest_store.save(self.manifest)
            saw_document = False
            batch: list[CorpusDocumentChunks] = []
            for document in documents:
                saw_document = True
                self._observe_document(document)
                batch.append(document)
                if len(batch) < self.link_batch_documents:
                    continue
                await self._run_batch(batch)
                batch = []
            if batch:
                await self._run_batch(batch)
            if not saw_document:
                raise GraphCorpusBuildError("corpus build selected no documents")
            self._validate_observed_input()
            self.manifest.status = (
                "provider_blocked"
                if self._provider_blocked
                else "partial_failure"
                if (
                    self.manifest.failed_document_ids
                    or self.manifest.failed_chunk_ids
                    or self.manifest.failed_link_batch_keys
                )
                else "completed"
            )
            self.manifest.ended_at = _now()
            self.manifest_store.save(self.manifest)
            return self.manifest
        except asyncio.CancelledError:
            self.manifest.status = "cancelled"
            self.manifest.ended_at = _now()
            self.manifest_store.save(self.manifest)
            raise
        except (GraphCorpusBuildError, GraphStoreError) as error:
            self.manifest.status = (
                "provider_blocked" if self._provider_blocked else "partial_failure"
            )
            self.manifest.ended_at = _now()
            message = str(error).replace("\n", " ")[:500]
            self.manifest.run_error = message
            if isinstance(error, GraphCorpusBuildError) and not self._provider_blocked:
                self.manifest.input_validation_error = message
            self.manifest_store.save(self.manifest)
            raise
        except Exception as error:
            self.manifest.status = "partial_failure"
            self.manifest.ended_at = _now()
            self.manifest.run_error = type(error).__name__
            self.manifest_store.save(self.manifest)
            raise
        finally:
            self.close()

    async def _run_batch(self, documents: Sequence[CorpusDocumentChunks]) -> None:
        if self._provider_blocked:
            return
        outcomes_by_document: dict[UUID, Sequence[_ChunkOutcome]] = {}
        force_relink = False
        linkable_documents: list[CorpusDocumentChunks] = []
        for document in documents:
            outcomes, failed, rebuilt = await self._process_document(document)
            force_relink = force_relink or rebuilt
            outcomes_by_document[document.metadata.document_id] = outcomes
            for outcome in outcomes:
                if outcome.result is not None:
                    self._add_extraction_counters(outcome.result)
                    self.manifest.counters.retry_calls += (
                        outcome.checkpoint.extraction_corpus_retry_calls
                    )
                    self.manifest.counters.extraction_corpus_retry_calls += (
                        outcome.checkpoint.extraction_corpus_retry_calls
                    )
                elif not outcome.reused:
                    self._add_failed_extraction_counters(outcome.checkpoint)
                if outcome.reused:
                    self.manifest.counters.chunks_skipped += 1
                elif outcome.checkpoint.status == "completed":
                    self.manifest.counters.chunks_attempted += 1
                    self.manifest.counters.chunks_completed += 1
                elif outcome.checkpoint.status == "failed":
                    self.manifest.counters.chunks_attempted += 1
                    self.manifest.counters.chunks_failed += 1
                self._mark_processed_chunk(
                    outcome.chunk.chunk_id,
                    failed=outcome.checkpoint.status != "completed",
                )
            if failed:
                self.manifest.counters.documents_failed += 1
            elif not all(outcome.reused for outcome in outcomes):
                self.manifest.counters.documents_completed += 1
            self._mark_processed_document(document.metadata.document_id, failed=failed)
            if failed:
                self._set_document_linking_state(
                    document.metadata.document_id,
                    complete=False,
                    blocked=True,
                )
                await self._delete_linking_state([document.metadata.document_id])
            else:
                linkable_documents.append(document)
            self.manifest_store.save(self.manifest)
            if self._provider_blocked:
                break
        if linkable_documents:
            linking_succeeded = await self._link_batch(
                linkable_documents,
                outcomes_by_document,
                force_relink=force_relink,
            )
            for document in linkable_documents:
                self._set_document_linking_state(
                    document.metadata.document_id,
                    complete=linking_succeeded,
                    blocked=not linking_succeeded,
                )
        self.manifest_store.save(self.manifest)


def _graph_audit_with_unavailable(
    audit: CorpusAudit,
) -> CorpusAudit:
    return audit.model_copy(update={"graph_status": "unavailable"})


__all__ = [
    "DEFAULT_CHUNK_INDEX",
    "DEFAULT_CORPUS_MANIFEST",
    "DEFAULT_ESTIMATE_SUMMARIES",
    "DEFAULT_EVAL_DATASET",
    "DEFAULT_INPUT_SNAPSHOT",
    "DEFAULT_OUTPUT",
    "A21GraphCoverage",
    "BuildCounters",
    "CheckpointStore",
    "CorpusAudit",
    "CorpusDocumentChunks",
    "CorpusDocumentMetadata",
    "CorpusInputDocument",
    "CorpusInputSnapshot",
    "CostEstimate",
    "DocumentCheckpoint",
    "FailedProviderAttemptSummary",
    "GraphCorpusBuildError",
    "GraphCorpusRunner",
    "GraphCoverageSnapshot",
    "LinkBatchCheckpoint",
    "RunManifest",
    "audit_corpus_files",
    "build_a21_graph_coverage",
    "build_corpus_input_snapshot",
    "build_pipeline_fingerprint",
    "chunk_input_fingerprint",
    "estimate_full_run",
    "iter_corpus_documents",
    "iter_indexed_chunks",
    "load_corpus_input_snapshot",
    "load_corpus_metadata",
    "maximum_extraction_provider_attempts",
    "read_graph_coverage",
    "select_document_ids",
    "summarize_failed_provider_attempts",
    "validate_corpus_input_snapshot",
]
