# ruff: noqa: RUF001
"""Final read-only retrieval ablation and multimodal evaluation for A4.5.

The evaluator composes the retrieval services already accepted in A2--A4.4.
It owns the frozen evaluation protocol, provenance-safe result accounting and
report serialization, but it does not implement another retriever or mutate a
retrieval store.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import platform
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from knowledge_scope.evaluation.embedding_benchmark import (
    FrozenEvalCase,
    load_frozen_chunk_index,
    load_frozen_eval_cases,
)
from knowledge_scope.evaluation.retrieval_eval import IndexedChunk
from knowledge_scope.evaluation.retrieval_metrics import SourceBlockKey, ranking_metrics
from knowledge_scope.evidence.models import (
    EvidenceRepresentation,
    MultimodalEvidence,
    MultimodalEvidenceDocument,
    canonical_document_fingerprint,
)
from knowledge_scope.evidence.service import validate_evidence_document
from knowledge_scope.graph.retrieval import GraphRetrievalConfig, GraphRetrievalResult
from knowledge_scope.graph.retrieval_service import GraphRetrievalService
from knowledge_scope.parsing.models import CanonicalDocument
from knowledge_scope.rag.context import assemble_context
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    ChunkVectorPayload,
    QdrantVectorStore,
    RetrievedChunk,
    point_id_for_chunk,
)
from knowledge_scope.retrieval.representation_index import (
    MultimodalRepresentationRetrievalService,
    QdrantRepresentationStore,
    RepresentationRetrievalResult,
    RetrievedEvidence,
)
from knowledge_scope.retrieval.reranking import (
    RerankedChunk,
    RerankerProtocol,
    RerankingService,
    create_local_reranker,
)
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.retrieval.sparse import (
    SparseIndexStore,
    SparseSearchHit,
    SparseSearchResponse,
    term_frequencies,
)
from knowledge_scope.retrieval.unified import (
    UnifiedCandidate,
    UnifiedRetrievalConfig,
    UnifiedRetrievalResult,
    UnifiedRetrievalService,
)
from knowledge_scope.shared.config import Settings, get_settings

A45_SCHEMA_VERSION = "1.0"
A45_MULTIMODAL_DATASET_VERSION = "a4-5-multimodal-eval-v2"
A45_MULTIMODAL_GENERATION_METHOD = "source_derived_template_assisted_v2"
A45_MULTIMODAL_QUERY_ORIGIN = "source_derived_template_assisted"
A45_MULTIMODAL_LEAKAGE_POLICY = "representation-leakage-v1"
A45_PROFILE_VERSION = "a4.5-frozen-v1"
A25_PROFILE_VERSION = "a2.5-frozen-v1"
A25_EMBEDDING_MODEL = QWEN_EMBEDDING_MODEL_ID
A25_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
A25_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
A25_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
A25_QDRANT_COLLECTION = "knowledgescope_chunks_v1"
A25_DENSE_TOP_K = 10
A25_RERANK_TOP_K = 10
A45_SPARSE_TOP_K = 20
A45_GRAPH_TOP_K = 20
A45_MULTIMODAL_TOP_K = 20
A45_CANDIDATE_POOL = 80
A45_RESULT_LIMIT = 10
A45_RERANK_TEXT_MAX_CHARS = 6_000
A45_RRF_K = 60
A36_RUN_ID = "e926e1d3-9050-4911-89ef-1632ed0894c2"
A36_GRAPH_ELIGIBLE_DOCUMENTS = 246
A36_GRAPH_EVIDENCE_CHUNKS = 6_087
A36_RUN_MANIFEST = Path("data/evaluation/a3-6/full-run-20260910/run-manifest.json")
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_MATERIALIZED = Path("data/evaluation/a2-1/retrieval-eval-v1/materialized.jsonl")
DEFAULT_A25_RESULTS = Path("data/evaluation/a2-5/results.jsonl")
DEFAULT_A25_MANIFEST = Path("data/evaluation/a2-5/manifest.json")
DEFAULT_EVIDENCE_DIR = Path("data/evidence")
DEFAULT_CORPUS_MANIFEST = Path("data/benchmarks/a1-5/corpus-manifest.jsonl")
DEFAULT_CANONICAL_ROOT = Path("data/benchmarks/a1-5/canonical")
LEGACY_MULTIMODAL_DATASET = Path("docs/benchmarks/a4-5-multimodal-eval-v1.jsonl")
LEGACY_MULTIMODAL_MANIFEST = Path("docs/benchmarks/a4-5-multimodal-eval-manifest.json")
DEFAULT_MULTIMODAL_DATASET = Path("docs/benchmarks/a4-5-multimodal-eval-v2.jsonl")
DEFAULT_MULTIMODAL_MANIFEST = Path("docs/benchmarks/a4-5-multimodal-eval-v2-manifest.json")
DEFAULT_FROZEN_STORE_MANIFEST = Path("docs/benchmarks/a4-5-frozen-store-manifest.json")
# These hashes are the repository-safe v2 artifact identity.  The builder and
# loader both enforce them so a self-consistent replacement cannot masquerade
# as the accepted frozen evaluation set.
A45_FROZEN_MULTIMODAL_DATASET_FINGERPRINT = (
    "467c7a09d5db41c8c02af3562a55cc87fae0ac9bf20be6515b63b10c010ba28c"
)
A45_FROZEN_MULTIMODAL_FILE_SHA256 = (
    "217d83f37d0e84033b3c784044ae2734f6136f6b3604a4406f9b911cff9f5359"
)
A45_FROZEN_STORE_MANIFEST_SHA256 = (
    "16df2c177aec5ff3a41b28043dbe50e8ad163ed235bd3844da1248d4924c19ac"
)
DEFAULT_OUTPUT = Path("data/evaluation/a4-5")
SUPPORTED_SPLITS = ("dev", "test", "both")
METRIC_KS = (1, 3, 5, 10)
GENERAL_SYSTEMS = (
    "dense_bge",
    "dense_sparse_bge",
    "dense_graph_bge",
    "dense_multimodal_bge",
    "unified",
)
MULTIMODAL_SYSTEMS = ("dense_bge", "multimodal_representation", "unified")
BRANCHES = ("dense", "sparse", "graph", "multimodal")
MM_MODALITIES = ("image", "table", "formula")
FROZEN_SUBJECTS = ("化学", "历史", "地理", "思想政治", "数学", "物理", "生物", "英语", "语文")
MULTIMODAL_MODALITY_QUOTAS: dict[str, dict[str, int]] = {
    "化学": {"image": 3, "table": 3, "formula": 2},
    "历史": {"image": 5, "table": 3},
    "地理": {"image": 6, "table": 2},
    "思想政治": {"image": 5, "table": 3},
    "数学": {"image": 2, "table": 3, "formula": 3},
    "物理": {"image": 3, "table": 2, "formula": 3},
    "生物": {"image": 3, "table": 3, "formula": 2},
    "英语": {"image": 5, "table": 3},
    "语文": {"image": 6, "table": 2},
}
_GENERIC_SOURCE_LABELS = frozenset(
    {
        "学习提示",
        "综合运用",
        "资料分析",
        "本章复习题",
        "练习",
        "思考题",
        "讨论题",
        "答案",
        "目录",
        "人民教育出版社",
        "出版社",
    }
)

SplitName = Literal["dev", "test", "all"]
RunSplit = Literal["dev", "test", "both"]
A45SystemName = Literal[
    "dense_bge",
    "dense_sparse_bge",
    "dense_graph_bge",
    "dense_multimodal_bge",
    "unified",
]
A45Modality = Literal["image", "table", "formula"]
MultimodalQueryType = Literal["factual", "explanation", "definition", "formula_or_table"]


class A45EvaluationError(RuntimeError):
    """Raised when frozen inputs or runtime provenance cannot be trusted."""


class _A45Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class A45Protocol(_A45Model):
    """Frozen A4.5 profile; no field is tuned from the test split."""

    profile_version: Literal["a4.5-frozen-v1"] = A45_PROFILE_VERSION
    a25_profile_version: Literal["a2.5-frozen-v1"] = A25_PROFILE_VERSION
    embedding_model: Literal["Qwen/Qwen3-Embedding-0.6B"] = A25_EMBEDDING_MODEL
    embedding_revision: str = A25_EMBEDDING_REVISION
    embedding_device: str = "cuda"
    embedding_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    embedding_batch_size: StrictInt = Field(default=4, ge=1)
    embedding_max_seq_length: StrictInt = Field(default=512, ge=1)
    embedding_query_convention: Literal["SentenceTransformers prompt_name=query"] = (
        "SentenceTransformers prompt_name=query"
    )
    embedding_document_convention: Literal["no query prompt"] = "no query prompt"
    embedding_normalization: Literal["l2"] = "l2"
    embedding_pooling: Literal["SentenceTransformer model-native pooling"] = (
        "SentenceTransformer model-native pooling"
    )
    reranker_model: Literal["BAAI/bge-reranker-v2-m3"] = A25_RERANKER_MODEL
    reranker_revision: str = A25_RERANKER_REVISION
    reranker_device: str = "cuda"
    reranker_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    reranker_batch_size: StrictInt = Field(default=8, ge=1)
    reranker_max_seq_length: StrictInt = Field(default=512, ge=1)
    qdrant_collection: Literal["knowledgescope_chunks_v1"] = A25_QDRANT_COLLECTION
    qdrant_schema_version: Literal["1.0"] = QDRANT_COLLECTION_SCHEMA_VERSION
    qdrant_dimension: StrictInt = QDRANT_VECTOR_DIMENSION
    qdrant_distance: Literal["cosine"] = "cosine"
    qdrant_scope: Literal["current collection; no ANN performance claim"] = (
        "current collection; no ANN performance claim"
    )
    dense_candidate_limit: StrictInt = Field(default=A25_DENSE_TOP_K, ge=1, le=100)
    dense_rerank_limit: StrictInt = Field(default=A25_RERANK_TOP_K, ge=1, le=100)
    sparse_candidate_limit: StrictInt = Field(default=A45_SPARSE_TOP_K, ge=1, le=100)
    graph_candidate_limit: StrictInt = Field(default=A45_GRAPH_TOP_K, ge=1, le=500)
    multimodal_candidate_limit: StrictInt = Field(default=A45_MULTIMODAL_TOP_K, ge=1, le=100)
    candidate_pool_limit: StrictInt = Field(default=A45_CANDIDATE_POOL, ge=1, le=500)
    result_limit: StrictInt = Field(default=A45_RESULT_LIMIT, ge=1, le=100)
    rerank_text_max_chars: StrictInt = Field(default=A45_RERANK_TEXT_MAX_CHARS, ge=1)
    a35_rrf_k: StrictInt = Field(default=A45_RRF_K, ge=1)
    failure_mode: Literal["strict"] = "strict"
    warmup: StrictBool = True

    @model_validator(mode="after")
    def validate_frozen_limits(self) -> A45Protocol:
        expected = {
            "embedding_revision": A25_EMBEDDING_REVISION,
            "embedding_device": "cuda",
            "embedding_dtype": "float16",
            "embedding_batch_size": 4,
            "embedding_max_seq_length": 512,
            "embedding_query_convention": "SentenceTransformers prompt_name=query",
            "embedding_document_convention": "no query prompt",
            "embedding_normalization": "l2",
            "embedding_pooling": "SentenceTransformer model-native pooling",
            "reranker_revision": A25_RERANKER_REVISION,
            "reranker_device": "cuda",
            "reranker_dtype": "float16",
            "reranker_batch_size": 8,
            "reranker_max_seq_length": 512,
            "dense_candidate_limit": A25_DENSE_TOP_K,
            "dense_rerank_limit": A25_RERANK_TOP_K,
            "sparse_candidate_limit": A45_SPARSE_TOP_K,
            "graph_candidate_limit": A45_GRAPH_TOP_K,
            "multimodal_candidate_limit": A45_MULTIMODAL_TOP_K,
            "qdrant_collection": A25_QDRANT_COLLECTION,
            "qdrant_dimension": QDRANT_VECTOR_DIMENSION,
            "candidate_pool_limit": A45_CANDIDATE_POOL,
            "result_limit": A45_RESULT_LIMIT,
            "rerank_text_max_chars": A45_RERANK_TEXT_MAX_CHARS,
            "a35_rrf_k": A45_RRF_K,
            "failure_mode": "strict",
            "warmup": True,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError("A4.5 frozen profile mismatch: " + "; ".join(mismatches))
        if self.result_limit > self.candidate_pool_limit:
            raise ValueError("result_limit must not exceed candidate_pool_limit")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class A45MultimodalEvalItem(_A45Model):
    """One frozen query targeting authoritative A4.1 Evidence."""

    schema_version: Literal["1.0"] = A45_SCHEMA_VERSION
    eval_id: str = Field(pattern=r"^a4-5-mm-[0-9a-f]{64}$")
    query: str = Field(min_length=1, max_length=500)
    subject: str = Field(min_length=1)
    query_type: MultimodalQueryType
    modality: A45Modality
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    asset_refs: list[str]
    split: Literal["dev", "test"]
    rationale: str = Field(min_length=1, max_length=240)
    generation_method: Literal["source_derived_template_assisted_v2"] = (
        A45_MULTIMODAL_GENERATION_METHOD
    )
    query_origin: Literal["source_derived_template_assisted"] = A45_MULTIMODAL_QUERY_ORIGIN
    source_representation_type: str = Field(min_length=1)
    source_representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_document_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lineage(self) -> A45MultimodalEvalItem:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        if any(not value.strip() for value in self.source_block_ids):
            raise ValueError("source_block_ids must not contain blanks")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("source_block_ids must be unique")
        if any(not value.strip() for value in self.asset_refs):
            raise ValueError("asset_refs must not contain blanks")
        if "…" in self.query or "..." in self.query:
            raise ValueError("multimodal query must not contain truncation markers")
        expected = multimodal_eval_id(
            self.query,
            self.modality,
            self.evidence_id,
            self.knowledge_base_id,
            self.document_id,
            self.page_start,
            self.page_end,
            self.source_block_ids,
            self.asset_refs,
            source_fingerprint=self.source_fingerprint,
            canonical_document_fingerprint=self.canonical_document_fingerprint,
        )
        if self.eval_id != expected:
            raise ValueError("eval_id does not match the evidence identity")
        return self


class A45MultimodalDatasetManifest(_A45Model):
    """Repository-safe identity and distribution metadata for the frozen set."""

    schema_version: Literal["1.0"] = A45_SCHEMA_VERSION
    dataset_version: Literal["a4-5-multimodal-eval-v2"] = A45_MULTIMODAL_DATASET_VERSION
    generation_method: Literal["source_derived_template_assisted_v2"] = (
        A45_MULTIMODAL_GENERATION_METHOD
    )
    query_origin: Literal["source_derived_template_assisted"] = A45_MULTIMODAL_QUERY_ORIGIN
    item_count: StrictInt = Field(ge=0)
    dev_count: StrictInt = Field(ge=0)
    test_count: StrictInt = Field(ge=0)
    modality_counts: dict[str, StrictInt]
    subject_counts: dict[str, StrictInt]
    subject_split_counts: dict[str, dict[str, StrictInt]] = Field(default_factory=dict)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_base_ids: list[UUID]
    source_evidence_schema_version: Literal["1.0"] = "1.0"
    leakage_policy: Literal["representation-leakage-v1"] = A45_MULTIMODAL_LEAKAGE_POLICY
    leakage_rejected_count: StrictInt = Field(default=0, ge=0)
    leakage_issue_counts: dict[str, StrictInt] = Field(default_factory=dict)
    leakage_checked_item_count: StrictInt = Field(default=0, ge=0)
    leakage_item_count: StrictInt = Field(default=0, ge=0)
    leakage_audit_issue_counts: dict[str, StrictInt] = Field(default_factory=dict)
    leakage_audit_passed: StrictBool = False
    duplicate_query_count: StrictInt = Field(default=0, ge=0)
    near_duplicate_query_count: StrictInt = Field(default=0, ge=0)
    repeated_evidence_count: StrictInt = Field(default=0, ge=0)
    dev_document_count: StrictInt = Field(default=0, ge=0)
    test_document_count: StrictInt = Field(default=0, ge=0)
    document_overlap_count: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_frozen_distribution(self) -> A45MultimodalDatasetManifest:
        expected_modalities = {"image": 38, "table": 24, "formula": 10}
        expected_subjects = {subject: 8 for subject in FROZEN_SUBJECTS}
        # The small model is also useful for validating a manifest round trip in
        # isolation.  The production loader/builder applies the frozen v2
        # contract unconditionally; this model-level check is therefore only
        # strict once the frozen-size manifest is being represented.
        if self.item_count != 72:
            return self
        if self.dev_count != 36 or self.test_count != 36:
            raise ValueError("A4.5 v2 requires 72 items with a 36/36 split")
        if self.modality_counts != expected_modalities:
            raise ValueError("A4.5 v2 modality quotas are invalid")
        if self.subject_counts != expected_subjects:
            raise ValueError("A4.5 v2 must contain eight items for every subject")
        if any(
            values.get("dev") != 4 or values.get("test") != 4
            for values in self.subject_split_counts.values()
        ) or set(self.subject_split_counts) != set(FROZEN_SUBJECTS):
            raise ValueError("A4.5 v2 requires a four/four subject split")
        if (
            self.leakage_checked_item_count != 72
            or self.leakage_item_count != 0
            or self.leakage_audit_issue_counts
            or not self.leakage_audit_passed
        ):
            raise ValueError("A4.5 v2 leakage audit must cover all items and pass")
        return self


class A45A21FrozenStore(_A45Model):
    """Immutable A2.1 dataset identity used by the read-only evaluator."""

    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: StrictInt = 108
    dev_count: StrictInt = 72
    test_count: StrictInt = 36
    knowledge_base_id: UUID


class A45DenseFrozenStore(_A45Model):
    """Immutable identity and critical lineage fingerprint for A2.5 Qdrant."""

    collection_name: Literal["knowledgescope_chunks_v1"] = A25_QDRANT_COLLECTION
    collection_role: Literal["a2.5-frozen-dense-chunks"] = "a2.5-frozen-dense-chunks"
    schema_version: Literal["1.0"] = QDRANT_COLLECTION_SCHEMA_VERSION
    vector_dimension: StrictInt = QDRANT_VECTOR_DIMENSION
    distance: Literal["cosine"] = "cosine"
    embedding_model: Literal["Qwen/Qwen3-Embedding-0.6B"] = A25_EMBEDDING_MODEL
    embedding_revision: str = A25_EMBEDDING_REVISION
    embedding_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    point_count: StrictInt = Field(ge=0)
    point_identity_lineage_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class A45SparseFrozenStore(_A45Model):
    """Immutable identity of the active A4.3 read-only generation."""

    index_path: str
    static_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_generation_id: str = Field(min_length=1)
    active_index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    searchable_chunk_count: StrictInt = Field(ge=0)


class A45RepresentationFrozenStore(_A45Model):
    """Immutable identity and payload fingerprint of the A4.2 collection."""

    collection_name: str = Field(min_length=1)
    collection_role: Literal["a4.2-multimodal-representations"] = "a4.2-multimodal-representations"
    collection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    point_count: StrictInt = Field(ge=0)
    identity_payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class A45GraphFrozenStore(_A45Model):
    """Immutable identity of the eligible A3.6 graph snapshot."""

    knowledge_base_id: UUID
    a36_run_id: str
    eligible_document_count: StrictInt = Field(ge=0)
    evidence_chunk_count: StrictInt = Field(ge=0)
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class A45FrozenStoreManifest(_A45Model):
    """Repository-safe expected identities for every read-only evaluation store."""

    manifest_version: Literal["a4-5-frozen-stores-v1"] = "a4-5-frozen-stores-v1"
    knowledge_base_id: UUID
    a21: A45A21FrozenStore
    dense: A45DenseFrozenStore
    sparse: A45SparseFrozenStore
    representation: A45RepresentationFrozenStore
    graph: A45GraphFrozenStore


StoreName = Literal["dense", "representation", "sparse", "graph"]


class A45StoreStateSnapshot(_A45Model):
    """Deterministic read-only state captured around one evaluation."""

    snapshot_version: Literal["a4-5-store-state-v1"] = "a4-5-store-state-v1"
    knowledge_base_id: UUID
    benchmark_inputs: A45A21FrozenStore
    dense: A45DenseFrozenStore
    representation: A45RepresentationFrozenStore
    sparse: A45SparseFrozenStore
    graph: A45GraphFrozenStore

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class A45StoreStateAudit(_A45Model):
    """Evidence-derived before/after comparison for the frozen evaluator state."""

    before: A45StoreStateSnapshot
    after: A45StoreStateSnapshot
    before_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_stores: list[StoreName]
    changed_components: list[str]
    benchmark_inputs_changed: StrictBool
    stores_mutated: StrictBool

    @model_validator(mode="after")
    def validate_derived_fields(self) -> A45StoreStateAudit:
        if self.before_fingerprint != self.before.fingerprint:
            raise ValueError("before store-state fingerprint is inconsistent")
        if self.after_fingerprint != self.after.fingerprint:
            raise ValueError("after store-state fingerprint is inconsistent")
        expected_stores = [
            name
            for name in ("dense", "representation", "sparse", "graph")
            if getattr(self.before, name) != getattr(self.after, name)
        ]
        if self.changed_stores != expected_stores:
            raise ValueError("changed_stores must be derived from the snapshots")
        expected_benchmark_change = self.before.benchmark_inputs != self.after.benchmark_inputs
        if self.benchmark_inputs_changed != expected_benchmark_change:
            raise ValueError("benchmark input change must be derived from the snapshots")
        expected_components = [*expected_stores]
        if expected_benchmark_change:
            expected_components.append("benchmark_inputs")
        if self.changed_components != expected_components:
            raise ValueError("changed_components must be derived from the snapshots")
        if self.stores_mutated != bool(expected_stores):
            raise ValueError("stores_mutated must be derived from changed stores")
        return self


class A45RankedCandidate(_A45Model):
    """Bounded per-query candidate metadata with authoritative lineage."""

    candidate_id: str = Field(min_length=1)
    candidate_kind: Literal["chunk", "evidence"]
    rank: StrictInt = Field(ge=1)
    score: float
    source: str = Field(min_length=1)
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str | None = None
    evidence_id: str | None = None
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    asset_refs: list[str]
    modality: str | None = None
    representation_ids: list[str]
    branch_ranks: dict[str, StrictInt]

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> A45RankedCandidate:
        if not math.isfinite(self.score):
            raise ValueError("candidate score must be finite")
        if self.page_end < self.page_start:
            raise ValueError("candidate page range is invalid")
        if self.candidate_kind == "chunk" and (not self.chunk_id or self.evidence_id is not None):
            raise ValueError("chunk candidate identity is invalid")
        if self.candidate_kind == "evidence" and (
            not self.evidence_id or self.chunk_id is not None or self.modality is None
        ):
            raise ValueError("Evidence candidate identity is invalid")
        return self


class A45SystemObservation(_A45Model):
    """One system's result for one query."""

    status: Literal["success", "failed"]
    ranked_candidate_ids: list[str]
    ranked_candidates: list[A45RankedCandidate]
    metrics: dict[str, float]
    branch_candidates: dict[str, list[str]]
    branch_source_blocks: dict[str, dict[str, list[str]]]
    branch_status: dict[str, str]
    branch_latency_ms: dict[str, float]
    latency_breakdown_ms: dict[str, float]
    total_latency_ms: float
    branch_audit: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> A45SystemObservation:
        if self.status == "failed" and not self.error:
            raise ValueError("failed observation needs a safe error")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful observation must not contain an error")
        if self.ranked_candidate_ids != [item.candidate_id for item in self.ranked_candidates]:
            raise ValueError("ranked candidate IDs must match candidate records")
        if self.total_latency_ms < 0 or not math.isfinite(self.total_latency_ms):
            raise ValueError("total latency must be finite and non-negative")
        return self


class A45GeneralQueryRecord(_A45Model):
    """Machine-readable general retrieval result for one frozen A2.1 item."""

    evaluation_kind: Literal["general"] = "general"
    eval_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    gold_relevant_chunk_ids: list[str]
    gold_source_blocks: list[str]
    branch_gold_recovery: dict[str, dict[str, list[str]]]
    systems: dict[str, A45SystemObservation]


class A45MultimodalQueryRecord(_A45Model):
    """Machine-readable multimodal result with Evidence-level gold semantics."""

    evaluation_kind: Literal["multimodal"] = "multimodal"
    eval_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    modality: A45Modality
    gold_evidence_id: str
    knowledge_base_id: UUID
    document_id: UUID
    systems: dict[str, A45SystemObservation]
    citation_validation: dict[str, Literal["valid", "not_applicable", "invalid"]]


class A45Preflight(_A45Model):
    """Read-only store and frozen-input facts captured before evaluation."""

    knowledge_base_id: UUID
    a21_item_count: StrictInt
    a21_dev_count: StrictInt
    a21_test_count: StrictInt
    chunk_count: StrictInt
    qdrant_collection: str
    qdrant_point_count: StrictInt
    qdrant_vector_dimension: StrictInt | None
    qdrant_distance: str | None
    representation_collection: str
    representation_point_count: StrictInt | None
    representation_collection_fingerprint: str
    sparse_status: str
    sparse_index_fingerprint: str | None
    neo4j_status: str
    a36_run_id: str
    a36_run_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    a36_graph_eligible_documents: StrictInt
    a36_graph_evidence_chunks: StrictInt
    frozen_store_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    store_state_before: A45StoreStateSnapshot


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise A45EvaluationError(f"benchmark input is not readable: {path}") from error
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise A45EvaluationError(f"benchmark input is not readable: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise A45EvaluationError(f"invalid JSON on line {line_number}: {path}") from error
        if not isinstance(record, dict):
            raise A45EvaluationError(f"line {line_number} is not an object: {path}")
        records.append(record)
    return records


def _normal_query(value: str) -> str:
    return " ".join(value.casefold().split())


def multimodal_eval_id(
    query: str,
    modality: str,
    evidence_id: str,
    knowledge_base_id: UUID,
    document_id: UUID,
    page_start: int,
    page_end: int,
    source_block_ids: Sequence[str],
    asset_refs: Sequence[str],
    *,
    source_fingerprint: str | None = None,
    canonical_document_fingerprint: str | None = None,
) -> str:
    """Create an ID from query plus the authoritative Evidence contract."""

    payload = {
        "schema_version": A45_SCHEMA_VERSION,
        "dataset_version": A45_MULTIMODAL_DATASET_VERSION,
        "query": _normal_query(query),
        "modality": modality,
        "evidence_id": evidence_id,
        "knowledge_base_id": str(knowledge_base_id),
        "document_id": str(document_id),
        "page_start": page_start,
        "page_end": page_end,
        "source_block_ids": list(source_block_ids),
        "asset_refs": list(asset_refs),
        "source_fingerprint": source_fingerprint,
        "canonical_document_fingerprint": canonical_document_fingerprint,
    }
    return f"a4-5-mm-{_sha256_json(payload)}"


def multimodal_dataset_fingerprint(items: Sequence[A45MultimodalEvalItem]) -> str:
    """Fingerprint the ordered frozen item content without filesystem metadata."""

    return _sha256_json([item.model_dump(mode="json") for item in items])


def _clean_source_fragment(value: str) -> str:
    """Normalize display context for quality checks, never for gold content."""

    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    value = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip(" \t\r\n，。；;:：")
    return value


def _is_substantive_fragment(value: str) -> bool:
    normalized = _clean_source_fragment(value)
    if len(normalized) < 8 or normalized.casefold() in _GENERIC_SOURCE_LABELS:
        return False
    if re.fullmatch(r"[\d\W_]+", normalized, flags=re.UNICODE):
        return False
    if re.fullmatch(r"(?:第?[\d一二三四五六七八九十百]+[章节题问号]?|[A-Z]?)", normalized):
        return False
    meaningful = sum(
        character.isalnum() or "\u3400" <= character <= "\u9fff" for character in normalized
    )
    return meaningful >= 5


def _query_fragment(value: str) -> str | None:
    cleaned = _clean_source_fragment(value)
    parts = [
        re.sub(r"^\s*(邻近文本|章节|上下文|说明)\s*[:：]\s*", "", part).strip()
        for part in re.split(r"[；;]+", cleaned)
    ]
    substantive_parts = [part for part in parts if _is_substantive_fragment(part)]
    if len(parts) > 1 and not substantive_parts:
        return None
    if substantive_parts:
        # Prefer the most informative source sentence over a short section or
        # exercise label that happens to precede the same context.
        cleaned = max(substantive_parts, key=lambda part: (len(part), part))
    if not _is_substantive_fragment(cleaned):
        return None
    sentence = re.split(r"[。！？；;.!?]", cleaned, maxsplit=1)[0].strip()
    if len(sentence) >= 8:
        cleaned = sentence
    return cleaned if _is_substantive_fragment(cleaned) else None


def _preferred_searchable_representation(
    evidence: MultimodalEvidence,
) -> EvidenceRepresentation | None:
    """Select only representation metadata; its content never drafts the query."""

    priorities: dict[str, tuple[str, ...]] = {
        "image": ("context", "caption"),
        "table": ("context", "markdown", "html", "caption"),
        "formula": ("context", "latex"),
    }
    by_type = {
        representation.representation_type: representation
        for representation in evidence.representations
        if representation.searchable and representation.content
    }
    for representation_type in priorities[evidence.modality]:
        representation = by_type.get(representation_type)
        if representation is not None and representation.content is not None:
            return representation
    return None


def _context_label(value: str) -> str | None:
    """Return a complete, bounded section/title label or reject it."""

    cleaned = _clean_source_fragment(value).replace('"', "“").replace('"', "”")
    if "…" in cleaned or "..." in cleaned:
        return None
    if not _is_substantive_fragment(cleaned):
        return None
    if len(cleaned) <= 96:
        return cleaned
    boundaries = [match.end() for match in re.finditer(r"[。！？；;.!?]", cleaned)]
    usable = [position for position in boundaries if 8 <= position <= 96]
    if usable:
        return cleaned[: max(usable)].strip()
    whitespace = [position for position, char in enumerate(cleaned[:97], start=1) if char.isspace()]
    if whitespace and max(whitespace) >= 8:
        return cleaned[: max(whitespace)].strip()
    return None


def _topic_from_source_context(
    evidence: MultimodalEvidence,
    canonical: CanonicalDocument,
    subject: str,
) -> str:
    """Derive a query topic from section/title metadata, not representation text."""

    for value in reversed(evidence.lineage.section_path):
        topic = _context_label(value)
        if topic is not None and topic.casefold() not in _GENERIC_SOURCE_LABELS:
            return topic
    page = next(
        (page for page in canonical.pages if page.page_number == evidence.lineage.page_start),
        None,
    )
    if page is not None:
        for block in page.blocks:
            if getattr(block, "type", None) == "title":
                topic = _context_label(getattr(block, "text", ""))
                if topic is not None and topic.casefold() not in _GENERIC_SOURCE_LABELS:
                    return topic
    return _context_label(subject) or subject


def _draft_query_from_context(
    evidence: MultimodalEvidence,
    *,
    subject: str,
    document_label: str,
    canonical: CanonicalDocument,
) -> tuple[str, MultimodalQueryType]:
    topic = _topic_from_source_context(evidence, canonical, subject)
    document = _context_label(document_label) or subject
    page = evidence.lineage.page_start
    if evidence.modality == "image":
        return f"在《{document}》第{page}页的“{topic}”部分，图像主要呈现了什么内容？", "explanation"
    if evidence.modality == "table":
        return (
            f"在《{document}》第{page}页的“{topic}”部分，表格主要展示了哪些数据或分类？",
            "factual",
        )
    return (
        f"在《{document}》第{page}页的“{topic}”部分，公式表达了哪种数量关系？",
        "formula_or_table",
    )


def _load_corpus_metadata(corpus_manifest: Path) -> dict[UUID, tuple[str, str]]:
    """Load only safe subject/document labels from the registered corpus manifest."""

    metadata: dict[UUID, tuple[str, str]] = {}
    for record in _read_jsonl(corpus_manifest):
        value = record.get("benchmark_document_uuid")
        subject = record.get("subject")
        if not isinstance(value, str) or not isinstance(subject, str) or not subject.strip():
            continue
        try:
            document_id = UUID(value)
        except ValueError:
            continue
        basename = record.get("basename")
        label = (
            Path(basename).name
            if isinstance(basename, str) and basename.strip()
            else str(document_id)
        )
        label = Path(label).stem or str(document_id)
        current = (subject.strip(), label)
        if document_id in metadata and metadata[document_id][0] != current[0]:
            raise A45EvaluationError(f"corpus manifest has conflicting subjects: {document_id}")
        metadata[document_id] = current
    if not metadata:
        raise A45EvaluationError("corpus manifest contains no benchmark document subjects")
    return metadata


def _load_subjects(corpus_manifest: Path) -> dict[UUID, str]:
    return {
        document_id: subject
        for document_id, (subject, _label) in _load_corpus_metadata(corpus_manifest).items()
    }


def _load_canonical_documents(canonical_root: Path) -> dict[UUID, CanonicalDocument]:
    documents: dict[UUID, CanonicalDocument] = {}
    paths = sorted(canonical_root.glob("*.json"))
    if not paths:
        raise A45EvaluationError(f"no canonical artifacts found under {canonical_root}")
    for path in paths:
        try:
            document = CanonicalDocument.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as error:
            raise A45EvaluationError(f"invalid canonical artifact: {path}") from error
        if document.document_id in documents:
            raise A45EvaluationError(
                f"canonical artifacts contain duplicate document_id: {document.document_id}"
            )
        documents[document.document_id] = document
    return documents


def _load_evidence_documents(
    evidence_dir: Path,
    subjects: Mapping[UUID, str],
    *,
    canonical_root: Path | None = None,
) -> list[tuple[MultimodalEvidenceDocument, str]]:
    paths = sorted(evidence_dir.glob("*/evidence.json"))
    if not paths:
        raise A45EvaluationError(f"no A4.1 evidence artifacts found under {evidence_dir}")
    canonical_documents = (
        _load_canonical_documents(canonical_root) if canonical_root is not None else None
    )
    loaded: list[tuple[MultimodalEvidenceDocument, str]] = []
    for path in paths:
        try:
            artifact = MultimodalEvidenceDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise A45EvaluationError(f"invalid A4.1 evidence artifact: {path}") from error
        subject = subjects.get(artifact.document_id)
        # data/evidence may contain local smoke artifacts in addition to the
        # registered benchmark corpus.  They are outside this evaluation and
        # must not make a safe corpus build fail.
        if subject is None:
            continue
        if canonical_documents is not None:
            canonical = canonical_documents.get(artifact.document_id)
            if canonical is None:
                raise A45EvaluationError(
                    f"evidence artifact has no matching canonical document: {artifact.document_id}"
                )
            try:
                validate_evidence_document(canonical, artifact, artifact.knowledge_base_id)
            except ValueError as error:
                raise A45EvaluationError(
                    f"evidence artifact is stale or invalid for document {artifact.document_id}"
                ) from error
        loaded.append((artifact, subject))
    if not loaded:
        raise A45EvaluationError("no evidence artifact belongs to the registered benchmark corpus")
    return loaded


def _normal_leakage_text(value: str) -> str:
    """Normalize query/representation text for conservative leakage checks."""

    return "".join(unicodedata.normalize("NFKC", html.unescape(value)).casefold().split())


def _character_ngrams(value: str, size: int = 3) -> set[str]:
    compact = _normal_leakage_text(value)
    return {compact[index : index + size] for index in range(max(0, len(compact) - size + 1))}


def _multimodal_leakage_issues(
    item: A45MultimodalEvalItem,
    evidence: MultimodalEvidence,
) -> tuple[str, ...]:
    """Return codes only; never persist source representation text in the audit."""

    query = _normal_leakage_text(item.query)
    issues: set[str] = set()
    metadata_values = {
        evidence.evidence_id,
        item.source_representation_id,
        *evidence.lineage.source_block_ids,
    }
    if any(_normal_leakage_text(value) in query for value in metadata_values if value):
        issues.add("internal_metadata_in_query")

    for representation in evidence.representations:
        if not representation.searchable or not representation.content:
            continue
        source = _normal_leakage_text(representation.content)
        if not source:
            continue
        if query == source:
            issues.add("exact_representation_copy")
        if len(source) >= 24 and (source in query or query in source):
            issues.add("long_representation_substring")
        query_ngrams = _character_ngrams(item.query)
        source_ngrams = _character_ngrams(representation.content)
        if len(query) >= 32 and len(source) >= 32 and query_ngrams and source_ngrams:
            overlap = len(query_ngrams & source_ngrams) / len(query_ngrams | source_ngrams)
            if overlap >= 0.80:
                issues.add("high_representation_ngram_overlap")
    return tuple(sorted(issues))


def _multimodal_leakage_audit(
    items: Sequence[A45MultimodalEvalItem],
    evidence_by_id: Mapping[str, MultimodalEvidence],
) -> dict[str, object]:
    issue_counts: Counter[str] = Counter()
    leaking_ids: list[str] = []
    for item in items:
        evidence = evidence_by_id.get(item.evidence_id)
        if evidence is None:
            raise A45EvaluationError(f"multimodal gold Evidence is missing: {item.evidence_id}")
        issues = _multimodal_leakage_issues(item, evidence)
        if issues:
            leaking_ids.append(item.eval_id)
            issue_counts.update(issues)
    return {
        "policy": A45_MULTIMODAL_LEAKAGE_POLICY,
        "checked_item_count": len(items),
        "leaking_item_count": len(leaking_ids),
        "leaking_eval_ids": sorted(leaking_ids),
        "issue_counts": dict(sorted(issue_counts.items())),
        "passed": not leaking_ids,
    }


def _query_near_duplicate_count(items: Sequence[A45MultimodalEvalItem]) -> int:
    normalized = [_normal_leakage_text(item.query) for item in items]
    ngrams = [_character_ngrams(item.query) for item in items]
    count = 0
    for index, left in enumerate(normalized):
        for right_index in range(index + 1, len(normalized)):
            right = normalized[right_index]
            if left == right or not ngrams[index] or not ngrams[right_index]:
                continue
            similarity = len(ngrams[index] & ngrams[right_index]) / len(
                ngrams[index] | ngrams[right_index]
            )
            if similarity >= 0.90:
                count += 1
    return count


def _multimodal_split_audit(items: Sequence[A45MultimodalEvalItem]) -> dict[str, object]:
    dev_documents = {str(item.document_id) for item in items if item.split == "dev"}
    test_documents = {str(item.document_id) for item in items if item.split == "test"}
    normalized_queries = [_normal_query(item.query) for item in items]
    evidence_ids = [item.evidence_id for item in items]
    return {
        "duplicate_query_count": len(normalized_queries) - len(set(normalized_queries)),
        "near_duplicate_query_count": _query_near_duplicate_count(items),
        "repeated_evidence_count": len(evidence_ids) - len(set(evidence_ids)),
        "dev_document_count": len(dev_documents),
        "test_document_count": len(test_documents),
        "document_overlap_count": len(dev_documents & test_documents),
    }


def _validate_multimodal_contract(items: Sequence[A45MultimodalEvalItem]) -> dict[str, object]:
    """Enforce the exact repository v2 contract before it can be loaded or written."""

    if len(items) != 72:
        raise A45EvaluationError("A4.5 v2 requires exactly 72 items")
    split_counts = Counter(item.split for item in items)
    if split_counts != Counter({"dev": 36, "test": 36}):
        raise A45EvaluationError("A4.5 v2 requires a 36/36 split")
    modality_counts = Counter(item.modality for item in items)
    if modality_counts != Counter({"image": 38, "table": 24, "formula": 10}):
        raise A45EvaluationError("A4.5 v2 modality quotas are invalid")
    subject_counts = Counter(item.subject for item in items)
    if subject_counts != Counter({subject: 8 for subject in FROZEN_SUBJECTS}):
        raise A45EvaluationError("A4.5 v2 must contain eight items for every subject")
    per_subject_split = {
        subject: Counter(item.split for item in items if item.subject == subject)
        for subject in FROZEN_SUBJECTS
    }
    if any(counts != Counter({"dev": 4, "test": 4}) for counts in per_subject_split.values()):
        raise A45EvaluationError("A4.5 v2 requires a four/four split for every subject")
    if len({item.eval_id for item in items}) != len(items):
        raise A45EvaluationError("A4.5 v2 contains duplicate eval IDs")
    normalized_queries = [_normal_query(item.query) for item in items]
    if len(set(normalized_queries)) != len(items):
        raise A45EvaluationError("A4.5 v2 contains duplicate normalized queries")
    if len({item.evidence_id for item in items}) != len(items):
        raise A45EvaluationError("A4.5 v2 assigns one gold Evidence more than once")
    if any(
        item.generation_method != A45_MULTIMODAL_GENERATION_METHOD
        or item.query_origin != A45_MULTIMODAL_QUERY_ORIGIN
        for item in items
    ):
        raise A45EvaluationError("A4.5 v2 contains an unsupported query provenance")
    return _multimodal_split_audit(items)


def _validate_frozen_multimodal_identity(
    manifest: A45MultimodalDatasetManifest,
) -> None:
    if (
        manifest.dataset_fingerprint != A45_FROZEN_MULTIMODAL_DATASET_FINGERPRINT
        or manifest.file_sha256 != A45_FROZEN_MULTIMODAL_FILE_SHA256
    ):
        raise A45EvaluationError(
            "A4.5 v2 dataset does not match the frozen repository artifact identity"
        )


def _assign_subject_splits(items: Sequence[A45MultimodalEvalItem]) -> list[A45MultimodalEvalItem]:
    """Prefer document-disjoint dev/test groups, with explicit overlap if impossible."""

    ordered = sorted(items, key=lambda item: (item.document_id, item.modality, item.eval_id))
    for dev_indexes in combinations(range(len(ordered)), 4):
        dev_index_set = set(dev_indexes)
        dev_documents = {ordered[index].document_id for index in dev_indexes}
        test_documents = {
            ordered[index].document_id
            for index in range(len(ordered))
            if index not in dev_index_set
        }
        if dev_documents.isdisjoint(test_documents):
            return [
                item.model_copy(update={"split": "dev" if index in dev_index_set else "test"})
                for index, item in enumerate(ordered)
            ]
    return [
        item.model_copy(update={"split": "dev" if index < 4 else "test"})
        for index, item in enumerate(ordered)
    ]


def _select_multimodal_drafts(
    loaded: Sequence[tuple[MultimodalEvidenceDocument, str]],
    canonical_documents: Mapping[UUID, CanonicalDocument],
    document_labels: Mapping[UUID, str],
) -> tuple[list[A45MultimodalEvalItem], int, dict[str, int]]:
    candidates: dict[str, dict[str, list[tuple[tuple[int, str], A45MultimodalEvalItem]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    rejected_count = 0
    leakage_rejections: Counter[str] = Counter()
    for artifact, subject in loaded:
        canonical = canonical_documents.get(artifact.document_id)
        if canonical is None:
            raise A45EvaluationError(f"no canonical document for {artifact.document_id}")
        for evidence in artifact.evidence:
            if evidence.modality not in MM_MODALITIES:
                continue
            preferred = _preferred_searchable_representation(evidence)
            if preferred is None:
                rejected_count += 1
                leakage_rejections["no_searchable_representation"] += 1
                continue
            lineage = evidence.lineage
            query, query_type = _draft_query_from_context(
                evidence,
                subject=subject,
                document_label=document_labels.get(artifact.document_id, str(artifact.document_id)),
                canonical=canonical,
            )
            item = A45MultimodalEvalItem(
                eval_id=multimodal_eval_id(
                    query,
                    evidence.modality,
                    evidence.evidence_id,
                    lineage.knowledge_base_id,
                    lineage.document_id,
                    lineage.page_start,
                    lineage.page_end,
                    lineage.source_block_ids,
                    lineage.asset_refs,
                    source_fingerprint=lineage.source_fingerprint,
                    canonical_document_fingerprint=artifact.canonical_document_fingerprint,
                ),
                query=query,
                subject=subject,
                query_type=query_type,
                modality=evidence.modality,
                evidence_id=evidence.evidence_id,
                knowledge_base_id=lineage.knowledge_base_id,
                document_id=lineage.document_id,
                page_start=lineage.page_start,
                page_end=lineage.page_end,
                source_block_ids=list(lineage.source_block_ids),
                asset_refs=list(lineage.asset_refs),
                split="dev",
                rationale="主题取自章节或页面标题；问题未读取表示正文，需人工核验。",
                source_representation_type=preferred.representation_type,
                source_representation_id=preferred.representation_id,
                source_fingerprint=lineage.source_fingerprint,
                canonical_document_fingerprint=artifact.canonical_document_fingerprint,
            )
            issues = _multimodal_leakage_issues(item, evidence)
            if issues:
                rejected_count += 1
                leakage_rejections.update(issues)
                continue
            quality = (
                len(_topic_from_source_context(evidence, canonical, subject)),
                item.eval_id,
            )
            candidates[subject][evidence.modality].append((quality, item))

    selected: list[A45MultimodalEvalItem] = []
    seen_queries: set[str] = set()
    for subject in FROZEN_SUBJECTS:
        subject_candidates = candidates.get(subject, {})
        slots = [
            modality
            for modality in MM_MODALITIES
            for _ in range(MULTIMODAL_MODALITY_QUOTAS[subject].get(modality, 0))
        ]
        for values in subject_candidates.values():
            values.sort(key=lambda pair: (-pair[0][0], pair[0][1]))
        slot_candidates = [subject_candidates.get(modality, []) for modality in slots]

        def choose(
            slot_index: int,
            used_evidence: set[str],
            local_queries: set[str],
            chosen: list[A45MultimodalEvalItem],
            candidates_by_slot: Sequence[Sequence[tuple[tuple[int, str], A45MultimodalEvalItem]]],
        ) -> list[A45MultimodalEvalItem] | None:
            if slot_index == len(candidates_by_slot):
                return list(chosen)
            for _quality, candidate in candidates_by_slot[slot_index][:256]:
                normalized_query = _normal_query(candidate.query)
                if candidate.evidence_id in used_evidence or normalized_query in local_queries:
                    continue
                if normalized_query in seen_queries:
                    continue
                result = choose(
                    slot_index + 1,
                    used_evidence | {candidate.evidence_id},
                    local_queries | {normalized_query},
                    [*chosen, candidate],
                    candidates_by_slot,
                )
                if result is not None:
                    return result
            return None

        subject_items = choose(0, set(), set(), [], slot_candidates)
        if subject_items is None:
            raise A45EvaluationError(f"A4.5 v2 quotas cannot be satisfied for subject {subject}")
        assigned = _assign_subject_splits(subject_items)
        selected.extend(assigned)
        seen_queries.update(_normal_query(item.query) for item in assigned)

    selected.sort(key=lambda item: (item.subject, item.split, item.eval_id))
    _validate_multimodal_contract(selected)
    if not selected:
        raise A45EvaluationError(
            "no substantive multimodal Evidence candidates survived quality gates"
        )
    return selected, rejected_count, dict(sorted(leakage_rejections.items()))


def _evidence_index(
    loaded: Sequence[tuple[MultimodalEvidenceDocument, str]],
) -> dict[str, tuple[MultimodalEvidenceDocument, MultimodalEvidence, str]]:
    indexed: dict[str, tuple[MultimodalEvidenceDocument, MultimodalEvidence, str]] = {}
    for artifact, subject in loaded:
        for evidence in artifact.evidence:
            if evidence.evidence_id in indexed:
                raise A45EvaluationError(
                    f"duplicate authoritative Evidence ID across artifacts: {evidence.evidence_id}"
                )
            indexed[evidence.evidence_id] = (artifact, evidence, subject)
    return indexed


def _validate_multimodal_authority(
    items: Sequence[A45MultimodalEvalItem],
    loaded: Sequence[tuple[MultimodalEvidenceDocument, str]],
    canonical_documents: Mapping[UUID, CanonicalDocument],
) -> dict[str, tuple[MultimodalEvidenceDocument, MultimodalEvidence, str]]:
    """Resolve every gold item against current canonical and A4.1 artifacts."""

    indexed = _evidence_index(loaded)
    for item in items:
        current = indexed.get(item.evidence_id)
        if current is None:
            raise A45EvaluationError(f"multimodal gold Evidence is missing: {item.evidence_id}")
        artifact, evidence, subject = current
        canonical = canonical_documents.get(item.document_id)
        if canonical is None:
            raise A45EvaluationError(f"multimodal gold document is missing: {item.document_id}")
        lineage = evidence.lineage
        if subject != item.subject:
            raise A45EvaluationError(f"multimodal subject is stale for {item.eval_id}")
        if (
            artifact.knowledge_base_id != item.knowledge_base_id
            or artifact.document_id != item.document_id
            or lineage.knowledge_base_id != item.knowledge_base_id
            or lineage.document_id != item.document_id
            or evidence.modality != item.modality
            or lineage.page_start != item.page_start
            or lineage.page_end != item.page_end
            or lineage.source_block_ids != item.source_block_ids
            or lineage.asset_refs != item.asset_refs
            or lineage.source_fingerprint != item.source_fingerprint
            or artifact.canonical_document_fingerprint != item.canonical_document_fingerprint
            or canonical_document_fingerprint(canonical) != item.canonical_document_fingerprint
        ):
            raise A45EvaluationError(
                f"multimodal gold lineage is stale or inconsistent: {item.eval_id}"
            )
        representations = {
            representation.representation_id: representation
            for representation in evidence.representations
        }
        representation = representations.get(item.source_representation_id)
        if representation is None:
            raise A45EvaluationError(f"gold representation is missing for {item.eval_id}")
        if (
            not representation.searchable
            or not representation.content
            or representation.representation_type != item.source_representation_type
            or representation.evidence_id != evidence.evidence_id
            or representation.modality != evidence.modality
        ):
            raise A45EvaluationError(
                f"gold representation is stale or not searchable: {item.eval_id}"
            )
    return indexed


def build_multimodal_dataset(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
    output_path: Path = DEFAULT_MULTIMODAL_DATASET,
    manifest_path: Path = DEFAULT_MULTIMODAL_MANIFEST,
) -> A45MultimodalDatasetManifest:
    """Build the small frozen set before any retrieval result is inspected."""

    corpus_metadata = _load_corpus_metadata(corpus_manifest)
    subjects = {document_id: subject for document_id, (subject, _label) in corpus_metadata.items()}
    document_labels = {
        document_id: label for document_id, (_subject, label) in corpus_metadata.items()
    }
    canonical_documents = _load_canonical_documents(canonical_root)
    loaded = _load_evidence_documents(
        evidence_dir,
        subjects,
        canonical_root=canonical_root,
    )
    items, rejected_count, leakage_issue_counts = _select_multimodal_drafts(
        loaded,
        canonical_documents,
        document_labels,
    )
    if len({item.knowledge_base_id for item in items}) != 1:
        raise A45EvaluationError("A4.5 v2 must use exactly one knowledge base")
    split_audit = _validate_multimodal_contract(items)
    authority = _validate_multimodal_authority(items, loaded, canonical_documents)
    leakage_audit = _multimodal_leakage_audit(
        items,
        {
            evidence_id: evidence
            for evidence_id, (_artifact, evidence, _subject) in authority.items()
        },
    )
    if not bool(leakage_audit["passed"]):
        raise A45EvaluationError("A4.5 v2 candidate construction failed its leakage audit")
    final_leakage_issue_counts = {
        str(key): int(value)
        for key, value in dict(leakage_audit["issue_counts"]).items()  # type: ignore[union-attr]
    }
    # The selection step is deliberately complete before writing; no runtime
    # retrieval service is constructed by this function.  Validate the bytes
    # before replacing either repository-safe artifact.
    dataset_text = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in items
    )
    dataset_bytes = dataset_text.encode("utf-8")
    fingerprint = multimodal_dataset_fingerprint(items)
    manifest = A45MultimodalDatasetManifest(
        item_count=len(items),
        dev_count=sum(item.split == "dev" for item in items),
        test_count=sum(item.split == "test" for item in items),
        modality_counts=dict(sorted(Counter(item.modality for item in items).items())),
        subject_counts=dict(sorted(Counter(item.subject for item in items).items())),
        dataset_fingerprint=fingerprint,
        file_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        knowledge_base_ids=sorted({item.knowledge_base_id for item in items}, key=str),
        subject_split_counts={
            subject: {
                "dev": sum(item.subject == subject and item.split == "dev" for item in items),
                "test": sum(item.subject == subject and item.split == "test" for item in items),
            }
            for subject in FROZEN_SUBJECTS
        },
        leakage_rejected_count=rejected_count,
        leakage_issue_counts=leakage_issue_counts,
        leakage_checked_item_count=len(items),
        leakage_item_count=int(leakage_audit["leaking_item_count"]),
        leakage_audit_issue_counts=final_leakage_issue_counts,
        leakage_audit_passed=bool(leakage_audit["passed"]),
        duplicate_query_count=int(split_audit["duplicate_query_count"]),
        near_duplicate_query_count=int(split_audit["near_duplicate_query_count"]),
        repeated_evidence_count=int(split_audit["repeated_evidence_count"]),
        dev_document_count=int(split_audit["dev_document_count"]),
        test_document_count=int(split_audit["test_document_count"]),
        document_overlap_count=int(split_audit["document_overlap_count"]),
    )
    _validate_frozen_multimodal_identity(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(dataset_bytes)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_multimodal_dataset(
    dataset_path: Path = DEFAULT_MULTIMODAL_DATASET,
    manifest_path: Path = DEFAULT_MULTIMODAL_MANIFEST,
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
) -> tuple[list[A45MultimodalEvalItem], A45MultimodalDatasetManifest]:
    """Load and verify the frozen multimodal set without rebuilding it."""

    try:
        manifest = A45MultimodalDatasetManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise A45EvaluationError("multimodal dataset manifest is invalid") from error
    items = [A45MultimodalEvalItem.model_validate(record) for record in _read_jsonl(dataset_path)]
    split_audit = _validate_multimodal_contract(items)
    _validate_frozen_multimodal_identity(manifest)
    if len(items) != manifest.item_count:
        raise A45EvaluationError("multimodal dataset item count differs from manifest")
    if sum(item.split == "dev" for item in items) != manifest.dev_count:
        raise A45EvaluationError("multimodal dev count differs from manifest")
    if sum(item.split == "test" for item in items) != manifest.test_count:
        raise A45EvaluationError("multimodal test count differs from manifest")
    if dict(sorted(Counter(item.modality for item in items).items())) != manifest.modality_counts:
        raise A45EvaluationError("multimodal modality counts differ from manifest")
    if dict(sorted(Counter(item.subject for item in items).items())) != manifest.subject_counts:
        raise A45EvaluationError("multimodal subject counts differ from manifest")
    if sorted({item.knowledge_base_id for item in items}, key=str) != manifest.knowledge_base_ids:
        raise A45EvaluationError("multimodal knowledge-base scope differs from manifest")
    if dict(manifest.subject_split_counts) != {
        subject: {
            "dev": sum(item.subject == subject and item.split == "dev" for item in items),
            "test": sum(item.subject == subject and item.split == "test" for item in items),
        }
        for subject in FROZEN_SUBJECTS
    }:
        raise A45EvaluationError("multimodal subject split counts differ from manifest")
    for field_name in (
        "duplicate_query_count",
        "near_duplicate_query_count",
        "repeated_evidence_count",
        "dev_document_count",
        "test_document_count",
        "document_overlap_count",
    ):
        if getattr(manifest, field_name) != int(split_audit[field_name]):
            raise A45EvaluationError(f"multimodal split audit differs from manifest: {field_name}")
    if multimodal_dataset_fingerprint(items) != manifest.dataset_fingerprint:
        raise A45EvaluationError("multimodal dataset fingerprint differs from manifest")
    if _sha256_file(dataset_path) != manifest.file_sha256:
        raise A45EvaluationError("multimodal dataset file hash differs from manifest")
    corpus_metadata = _load_corpus_metadata(corpus_manifest)
    subjects = {document_id: subject for document_id, (subject, _label) in corpus_metadata.items()}
    canonical_documents = _load_canonical_documents(canonical_root)
    loaded = _load_evidence_documents(
        evidence_dir,
        subjects,
        canonical_root=canonical_root,
    )
    authority = _validate_multimodal_authority(items, loaded, canonical_documents)
    leakage_audit = _multimodal_leakage_audit(
        items,
        {
            evidence_id: evidence
            for evidence_id, (_artifact, evidence, _subject) in authority.items()
        },
    )
    expected_leakage_issue_counts = {
        str(key): int(value)
        for key, value in dict(leakage_audit["issue_counts"]).items()  # type: ignore[union-attr]
    }
    if manifest.leakage_checked_item_count != len(items):
        raise A45EvaluationError("multimodal leakage audit item count differs from manifest")
    if manifest.leakage_item_count != int(leakage_audit["leaking_item_count"]):
        raise A45EvaluationError("multimodal leakage audit result differs from manifest")
    if manifest.leakage_audit_issue_counts != expected_leakage_issue_counts:
        raise A45EvaluationError("multimodal leakage audit issues differ from manifest")
    if manifest.leakage_audit_passed != bool(leakage_audit["passed"]):
        raise A45EvaluationError("multimodal leakage audit status differs from manifest")
    if not bool(leakage_audit["passed"]):
        raise A45EvaluationError("multimodal dataset has representation leakage")
    return items, manifest


def evidence_hit_at_k(
    retrieved_evidence_ids: Sequence[str],
    gold_evidence_ids: Sequence[str],
    k: int,
) -> float:
    """Evidence-level binary Hit@K with stable duplicate suppression."""

    if k < 1:
        raise ValueError("k must be at least one")
    return float(bool(set(retrieved_evidence_ids[:k]) & set(gold_evidence_ids)))


def evidence_mrr(retrieved_evidence_ids: Sequence[str], gold_evidence_ids: Sequence[str]) -> float:
    """Evidence-level reciprocal rank of the first gold Evidence."""

    gold = set(gold_evidence_ids)
    for rank, value in enumerate(dict.fromkeys(retrieved_evidence_ids), start=1):
        if value in gold:
            return 1.0 / rank
    return 0.0


def evidence_metrics(
    retrieved_evidence_ids: Sequence[str],
    gold_evidence_ids: Sequence[str],
) -> dict[str, float]:
    """Compute Evidence Hit/MRR/Recall metrics for multimodal items."""

    gold = set(gold_evidence_ids)
    metrics = {"evidence_mrr": evidence_mrr(retrieved_evidence_ids, gold_evidence_ids)}
    for k in METRIC_KS:
        hit = evidence_hit_at_k(retrieved_evidence_ids, gold_evidence_ids, k)
        metrics[f"evidence_hit@{k}"] = hit
        covered = set(dict.fromkeys(retrieved_evidence_ids[:k])) & gold
        metrics[f"evidence_recall@{k}"] = len(covered) / len(gold) if gold else 0.0
    return metrics


def evidence_metrics_for_candidates(
    candidates: Sequence[A45RankedCandidate],
    gold_evidence_ids: Sequence[str],
) -> dict[str, float]:
    """Score Evidence gold at the candidates' actual mixed-list positions."""

    gold = set(gold_evidence_ids)
    metrics: dict[str, float] = {}
    first_rank: int | None = None
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.candidate_kind == "evidence" and candidate.evidence_id in gold:
            first_rank = rank
            break
    metrics["evidence_mrr"] = 1.0 / first_rank if first_rank is not None else 0.0
    for k in METRIC_KS:
        covered = {
            candidate.evidence_id
            for candidate in candidates[:k]
            if candidate.candidate_kind == "evidence"
            and candidate.evidence_id is not None
            and candidate.evidence_id in gold
        }
        metrics[f"evidence_hit@{k}"] = float(bool(covered))
        metrics[f"evidence_recall@{k}"] = len(covered) / len(gold) if gold else 0.0
    return metrics


def aggregate_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean query-level metrics, matching A2.1 aggregation semantics."""

    names = sorted({name for row in rows for name in row})
    return {
        name: statistics.fmean(row[name] for row in rows)
        for name in names
        if all(name in row for row in rows)
    }


def compare_query_hits(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    k: int,
) -> dict[str, int]:
    """Compare binary Hit@K query outcomes without mixing metric aggregates."""

    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("baseline and candidate rows must have equal length")
    key = f"hit@{k}"
    counts = {"improved": 0, "unchanged": 0, "regressed": 0}
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        before = bool(baseline.get(key, 0.0))
        after = bool(candidate.get(key, 0.0))
        if after and not before:
            counts["improved"] += 1
        elif before and not after:
            counts["regressed"] += 1
        else:
            counts["unchanged"] += 1
    return counts


class _TimedQueryEncoder:
    """Delegate the accepted encoder while exposing per-call timing for reports."""

    def __init__(self, delegate: QwenEmbeddingModel) -> None:
        self.delegate = delegate
        self.last_query_ms = 0.0

    @property
    def model_id(self) -> str:
        return self.delegate.model_id

    @property
    def model_revision(self) -> str:
        return self.delegate.model_revision

    @property
    def config_fingerprint(self) -> str:
        return self.delegate.config_fingerprint

    def encode_query(self, query: str) -> list[float]:
        started = time.perf_counter()
        try:
            return self.delegate.encode_query(query)
        finally:
            self.last_query_ms = (time.perf_counter() - started) * 1_000

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.delegate.encode_documents(texts)


class _TimedChunkStore:
    """Duck-typed Qdrant store wrapper for query-stage timing."""

    def __init__(self, delegate: QdrantVectorStore) -> None:
        self.delegate = delegate
        self.last_search_ms = 0.0

    @property
    def collection_name(self) -> str:
        return self.delegate.collection_name

    def search(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        started = time.perf_counter()
        try:
            return self.delegate.search(*args, **kwargs)
        finally:
            self.last_search_ms = (time.perf_counter() - started) * 1_000


class _TimedReranker:
    """Delegate the accepted BGE adapter while measuring score-pair latency."""

    def __init__(self, delegate: RerankerProtocol) -> None:
        self.delegate = delegate
        self.last_score_ms = 0.0

    @property
    def model_id(self) -> str:
        return self.delegate.model_id

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        started = time.perf_counter()
        try:
            return self.delegate.score_pairs(query, passages)
        finally:
            self.last_score_ms = (time.perf_counter() - started) * 1_000


class _EmptySparse:
    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> SparseSearchResponse:
        return SparseSearchResponse(
            query=query,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            top_k=top_k,
            index_fingerprint="0" * 64,
            items=[],
        )


class _EmptyGraph:
    def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> GraphRetrievalResult:
        return GraphRetrievalResult(query=query, knowledge_base_id=knowledge_base_id, items=[])


class _EmptyMultimodal:
    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> RepresentationRetrievalResult:
        return RepresentationRetrievalResult(query=query, knowledge_base_id=knowledge_base_id)


class _RecordingBranch:
    """Record successful raw branch items without changing the branch result."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_items: tuple[object, ...] = ()

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.last_items = ()
        response = self.delegate.search(*args, **kwargs)
        self.last_items = tuple(getattr(response, "items", ()))
        return response


@dataclass(slots=True)
class _A45Runtime:
    settings: Settings
    protocol: A45Protocol
    qdrant: QdrantVectorStore
    timed_store: _TimedChunkStore
    encoder: QwenEmbeddingModel
    timed_encoder: _TimedQueryEncoder
    representation_store: QdrantRepresentationStore
    sparse: SparseIndexStore
    neo4j: Any
    dense: DenseRetrievalService
    multimodal: MultimodalRepresentationRetrievalService
    graph: GraphRetrievalService
    reranker: _TimedReranker
    reranking: RerankingService

    def close(self) -> None:
        self.sparse.close()
        self.qdrant.close()
        self.representation_store.close()
        self.neo4j.close()


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _runtime_settings(settings: Settings, protocol: A45Protocol) -> Settings:
    if settings.qdrant_collection_name != A25_QDRANT_COLLECTION:
        raise A45EvaluationError(
            "A4.5 requires the frozen Qdrant chunk collection knowledgescope_chunks_v1"
        )
    if settings.embedding_model_revision != protocol.embedding_revision:
        raise A45EvaluationError(
            "configured embedding revision differs from the frozen A2.5 profile"
        )
    return settings.model_copy(
        update={
            "embedding_device": protocol.embedding_device,
            "embedding_dtype": protocol.embedding_dtype,
            "embedding_batch_size": protocol.embedding_batch_size,
            "embedding_max_seq_length": protocol.embedding_max_seq_length,
            "reranker_device": protocol.reranker_device,
            "reranker_dtype": protocol.reranker_dtype,
            "reranker_batch_size": protocol.reranker_batch_size,
            "reranker_max_seq_length": protocol.reranker_max_seq_length,
            "reranker_model_revision": protocol.reranker_revision,
        }
    )


def _build_runtime(settings: Settings, protocol: A45Protocol) -> _A45Runtime:
    runtime_settings = _runtime_settings(settings, protocol)
    qdrant = QdrantVectorStore(runtime_settings)
    timed_store = _TimedChunkStore(qdrant)
    encoder = QwenEmbeddingModel(runtime_settings)
    timed_encoder = _TimedQueryEncoder(encoder)
    dense = DenseRetrievalService(timed_store, timed_encoder)
    representation_store = QdrantRepresentationStore(runtime_settings)
    multimodal = MultimodalRepresentationRetrievalService(representation_store, timed_encoder)
    sparse = SparseIndexStore(runtime_settings.sparse_index_path, read_only=True)
    from knowledge_scope.graph.neo4j import Neo4jGraphStore

    neo4j = Neo4jGraphStore(runtime_settings)
    graph = GraphRetrievalService(
        neo4j,
        config=GraphRetrievalConfig(
            max_evidence=protocol.graph_candidate_limit,
            max_hops=2,
        ),
    )
    reranker_delegate = create_local_reranker(runtime_settings, model_key="bge-reranker-v2-m3")
    reranker = _TimedReranker(reranker_delegate)
    return _A45Runtime(
        settings=runtime_settings,
        protocol=protocol,
        qdrant=qdrant,
        timed_store=timed_store,
        encoder=encoder,
        timed_encoder=timed_encoder,
        representation_store=representation_store,
        sparse=sparse,
        neo4j=neo4j,
        dense=dense,
        multimodal=multimodal,
        graph=graph,
        reranker=reranker,
        reranking=RerankingService(reranker),
    )


def _count_representation_points(store: QdrantRepresentationStore) -> int | None:
    try:
        result = store._get_client().count(collection_name=store.collection_name, exact=True)
        count = getattr(result, "count", None)
        return int(count) if isinstance(count, int) else None
    except Exception:
        return None


def _record_value(record: object, key: str) -> object:
    try:
        return record[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        data = getattr(record, "data", None)
        if callable(data):
            return data().get(key)
        return None


def _safe_graph_value(value: object) -> object:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_graph_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_graph_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _graph_node_identifier(properties: Mapping[str, object]) -> str:
    for key in (
        "entity_id",
        "relation_id",
        "evidence_id",
        "canonical_entity_id",
        "link_id",
        "decision_id",
        "link_pair_id",
    ):
        value = properties.get(key)
        if value is not None:
            return str(value)
    return _sha256_json(dict(properties))


def _current_graph_snapshot_fingerprint(neo4j: Any, knowledge_base_id: UUID) -> str:
    """Hash current scoped graph identity/state without issuing a write."""

    params = {"knowledge_base_id": str(knowledge_base_id)}

    def read(session: Any) -> str:
        nodes: list[dict[str, object]] = []
        for record in session.run(
            """
            MATCH (node)
            WHERE node.knowledge_base_id = $knowledge_base_id
            RETURN labels(node) AS labels, properties(node) AS properties
            """,
            **params,
        ):
            labels = _safe_graph_value(_record_value(record, "labels"))
            properties_value = _safe_graph_value(_record_value(record, "properties"))
            properties = properties_value if isinstance(properties_value, dict) else {}
            nodes.append(
                {
                    "labels": sorted(str(label) for label in (labels or [])),
                    "id": _graph_node_identifier(properties),
                    "properties": properties,
                }
            )
        relationships: list[dict[str, object]] = []
        for record in session.run(
            """
            MATCH (left)-[relationship]->(right)
            WHERE left.knowledge_base_id = $knowledge_base_id
              AND right.knowledge_base_id = $knowledge_base_id
            RETURN properties(left) AS left_properties,
                   properties(right) AS right_properties,
                   type(relationship) AS relationship_type,
                   properties(relationship) AS relationship_properties
            """,
            **params,
        ):
            left_value = _safe_graph_value(_record_value(record, "left_properties"))
            right_value = _safe_graph_value(_record_value(record, "right_properties"))
            relationship_value = _safe_graph_value(_record_value(record, "relationship_properties"))
            left_properties = left_value if isinstance(left_value, dict) else {}
            right_properties = right_value if isinstance(right_value, dict) else {}
            relationships.append(
                {
                    "source": _graph_node_identifier(left_properties),
                    "target": _graph_node_identifier(right_properties),
                    "type": str(_record_value(record, "relationship_type")),
                    "properties": relationship_value,
                }
            )
        return _sha256_json(
            {
                "schema_version": "a4-5-graph-snapshot-v1",
                "knowledge_base_id": str(knowledge_base_id),
                "nodes": sorted(nodes, key=lambda item: json.dumps(item, sort_keys=True)),
                "relationships": sorted(
                    relationships,
                    key=lambda item: json.dumps(item, sort_keys=True),
                ),
            }
        )

    return neo4j._read(read)


def _current_frozen_store_manifest(
    runtime: _A45Runtime,
    *,
    knowledge_base_id: UUID,
    chunks: Mapping[str, IndexedChunk],
    cases_by_split: Mapping[str, Sequence[FrozenEvalCase]],
    dataset_path: Path,
    chunk_index_path: Path,
) -> A45FrozenStoreManifest:
    """Capture the current read-only store identities for strict comparison."""

    qdrant_readiness = runtime.qdrant.readiness()
    if qdrant_readiness.status != "ready":
        raise A45EvaluationError("cannot snapshot frozen Qdrant collection")
    client = runtime.qdrant._get_client()
    records: list[tuple[UUID, ChunkVectorPayload]] = []
    offset: Any | None = None
    while True:
        page, offset = client.scroll(
            collection_name=runtime.qdrant.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in page:
            try:
                point_id = UUID(str(record.id))
                payload = ChunkVectorPayload.model_validate(record.payload)
            except (TypeError, ValueError) as error:
                raise A45EvaluationError("frozen Qdrant chunk payload is invalid") from error
            if payload.knowledge_base_id != knowledge_base_id:
                raise A45EvaluationError("frozen Qdrant chunk payload has another KB")
            if point_id != point_id_for_chunk(payload.chunk_id):
                raise A45EvaluationError("frozen Qdrant point ID is inconsistent")
            records.append((point_id, payload))
        if offset is None:
            break
    if len(records) != len(chunks) or {payload.chunk_id for _point, payload in records} != set(
        chunks
    ):
        raise A45EvaluationError("frozen Qdrant points do not exactly cover A2.1 chunks")
    embedding_fingerprints = {payload.embedding_config_fingerprint for _point, payload in records}
    if len(embedding_fingerprints) != 1:
        raise A45EvaluationError("frozen Qdrant points use multiple embedding configurations")
    chunk_identity = [
        {"point_id": str(point_id), **payload.model_dump(mode="json")}
        for point_id, payload in sorted(records, key=lambda pair: pair[1].chunk_id)
    ]
    dense = A45DenseFrozenStore(
        embedding_config_fingerprint=next(iter(embedding_fingerprints)),
        point_count=len(records),
        point_identity_lineage_fingerprint=_sha256_json(chunk_identity),
    )

    active = runtime.sparse._active_generation()
    if active is None:
        raise A45EvaluationError("frozen Sparse index has no active generation")
    sparse_records = runtime.sparse._active_records()
    sparse = A45SparseFrozenStore(
        index_path=str(runtime.sparse.path),
        static_fingerprint=runtime.sparse.config.static_fingerprint,
        active_generation_id=active.generation_id,
        active_index_fingerprint=active.contract.fingerprint,
        corpus_fingerprint=active.contract.corpus_fingerprint,
        searchable_chunk_count=sum(
            bool(term_frequencies(record.text)) for record in sparse_records
        ),
    )

    representation_payloads = runtime.representation_store.list_payloads(
        knowledge_base_id=knowledge_base_id,
    )
    if len({payload.representation_id for payload in representation_payloads}) != len(
        representation_payloads
    ):
        raise A45EvaluationError("frozen representation collection has duplicate IDs")
    representation = A45RepresentationFrozenStore(
        collection_name=runtime.representation_store.collection_name,
        collection_fingerprint=runtime.representation_store.collection_fingerprint,
        point_count=len(representation_payloads),
        identity_payload_fingerprint=_sha256_json(
            [
                payload.model_dump(mode="json")
                for payload in sorted(
                    representation_payloads,
                    key=lambda value: value.representation_id,
                )
            ]
        ),
    )

    a21 = A45A21FrozenStore(
        dataset_sha256=_sha256_file(dataset_path),
        chunk_index_sha256=_sha256_file(chunk_index_path),
        knowledge_base_id=knowledge_base_id,
    )
    graph = A45GraphFrozenStore(
        knowledge_base_id=knowledge_base_id,
        a36_run_id=A36_RUN_ID,
        eligible_document_count=A36_GRAPH_ELIGIBLE_DOCUMENTS,
        evidence_chunk_count=A36_GRAPH_EVIDENCE_CHUNKS,
        snapshot_fingerprint=_current_graph_snapshot_fingerprint(runtime.neo4j, knowledge_base_id),
    )
    if sum(len(values) for values in cases_by_split.values()) != 108:
        raise A45EvaluationError("A2.1 cases are not the frozen 108-item set")
    return A45FrozenStoreManifest(
        knowledge_base_id=knowledge_base_id,
        a21=a21,
        dense=dense,
        sparse=sparse,
        representation=representation,
        graph=graph,
    )


def _store_state_snapshot(manifest: A45FrozenStoreManifest) -> A45StoreStateSnapshot:
    """Separate retrieval-store state from immutable benchmark-input identity."""

    return A45StoreStateSnapshot(
        knowledge_base_id=manifest.knowledge_base_id,
        benchmark_inputs=manifest.a21,
        dense=manifest.dense,
        representation=manifest.representation,
        sparse=manifest.sparse,
        graph=manifest.graph,
    )


def _compare_store_state(
    before: A45StoreStateSnapshot,
    after: A45StoreStateSnapshot,
) -> A45StoreStateAudit:
    """Derive a material before/after mutation audit from complete snapshots."""

    changed_stores = [
        name
        for name in ("dense", "representation", "sparse", "graph")
        if getattr(before, name) != getattr(after, name)
    ]
    benchmark_inputs_changed = before.benchmark_inputs != after.benchmark_inputs
    changed_components = [*changed_stores]
    if benchmark_inputs_changed:
        changed_components.append("benchmark_inputs")
    return A45StoreStateAudit(
        before=before,
        after=after,
        before_fingerprint=before.fingerprint,
        after_fingerprint=after.fingerprint,
        changed_stores=changed_stores,
        changed_components=changed_components,
        benchmark_inputs_changed=benchmark_inputs_changed,
        stores_mutated=bool(changed_stores),
    )


def _ensure_store_state_unchanged(audit: A45StoreStateAudit) -> None:
    """Reject evaluation finalization when frozen state changed materially."""

    if audit.changed_components:
        changed = ", ".join(audit.changed_components)
        raise A45EvaluationError(
            f"A4.5 frozen state changed during evaluation; changed components: {changed}"
        )


def load_frozen_store_manifest(
    path: Path = DEFAULT_FROZEN_STORE_MANIFEST,
) -> A45FrozenStoreManifest:
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != A45_FROZEN_STORE_MANIFEST_SHA256:
            raise A45EvaluationError("A4.5 frozen-store manifest identity does not match")
        return A45FrozenStoreManifest.model_validate(json.loads(raw))
    except A45EvaluationError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise A45EvaluationError("A4.5 frozen-store manifest is invalid or missing") from error


def _preflight(
    runtime: _A45Runtime,
    *,
    knowledge_base_id: UUID,
    chunks: Mapping[str, IndexedChunk],
    cases_by_split: Mapping[str, Sequence[FrozenEvalCase]],
    dataset_path: Path,
    chunk_index_path: Path,
    frozen_manifest: A45FrozenStoreManifest,
    frozen_manifest_path: Path,
) -> A45Preflight:
    qdrant_readiness = runtime.qdrant.readiness()
    if qdrant_readiness.status != "ready":
        raise A45EvaluationError(
            qdrant_readiness.error or "frozen Qdrant collection is unavailable"
        )
    metadata = runtime.qdrant.list_point_metadata()
    point_chunk_ids = {item.chunk_id for item in metadata}
    if len(metadata) != len(chunks) or point_chunk_ids != set(chunks):
        raise A45EvaluationError("Qdrant points do not exactly cover the frozen chunk index")
    if any(item.knowledge_base_id != knowledge_base_id for item in metadata):
        raise A45EvaluationError("Qdrant chunk points contain another knowledge base")
    current_manifest = _current_frozen_store_manifest(
        runtime,
        knowledge_base_id=knowledge_base_id,
        chunks=chunks,
        cases_by_split=cases_by_split,
        dataset_path=dataset_path,
        chunk_index_path=chunk_index_path,
    )
    if current_manifest.model_dump(mode="json") != frozen_manifest.model_dump(mode="json"):
        raise A45EvaluationError("A4.5 frozen-store manifest mismatch; evaluation stopped")
    store_state_before = _store_state_snapshot(current_manifest)
    representation_readiness = runtime.representation_store.readiness()
    if representation_readiness.status != "ready":
        raise A45EvaluationError(
            representation_readiness.error or "multimodal representation collection is unavailable"
        )
    sparse_readiness = runtime.sparse.readiness()
    neo_readiness = runtime.neo4j.readiness()
    if neo_readiness.status != "ready":
        raise A45EvaluationError(neo_readiness.error or "Neo4j graph is unavailable")
    return A45Preflight(
        knowledge_base_id=knowledge_base_id,
        a21_item_count=sum(len(values) for values in cases_by_split.values()),
        a21_dev_count=len(cases_by_split.get("dev", ())),
        a21_test_count=len(cases_by_split.get("test", ())),
        chunk_count=len(chunks),
        qdrant_collection=runtime.qdrant.collection_name,
        qdrant_point_count=len(metadata),
        qdrant_vector_dimension=qdrant_readiness.vector_dimension,
        qdrant_distance="cosine",
        representation_collection=runtime.representation_store.collection_name,
        representation_point_count=_count_representation_points(runtime.representation_store),
        representation_collection_fingerprint=(runtime.representation_store.collection_fingerprint),
        sparse_status=sparse_readiness.status,
        sparse_index_fingerprint=sparse_readiness.index_fingerprint,
        neo4j_status=neo_readiness.status,
        a36_run_id=A36_RUN_ID,
        a36_run_manifest_sha256=(
            _sha256_file(A36_RUN_MANIFEST) if A36_RUN_MANIFEST.is_file() else None
        ),
        a36_graph_eligible_documents=A36_GRAPH_ELIGIBLE_DOCUMENTS,
        a36_graph_evidence_chunks=A36_GRAPH_EVIDENCE_CHUNKS,
        frozen_store_manifest_sha256=_sha256_file(frozen_manifest_path),
        store_state_before=store_state_before,
    )


def _warmup_runtime(runtime: _A45Runtime, knowledge_base_id: UUID) -> None:
    """Warm the already-selected local adapters without changing stored state."""

    if not runtime.protocol.warmup:
        return
    query = "A4.5 deterministic local warmup"
    dense = runtime.dense.search(
        query,
        limit=1,
        knowledge_base_id=knowledge_base_id,
    )
    if dense.items:
        runtime.reranking.rerank(query, dense.items, limit=1)
    runtime.sparse.search(query, knowledge_base_id=knowledge_base_id, top_k=1)
    runtime.multimodal.search(query, knowledge_base_id=knowledge_base_id, top_k=1)
    runtime.graph.search(query, knowledge_base_id)


def _source_block_strings(blocks: Sequence[SourceBlockKey]) -> list[str]:
    return sorted(f"{document_id}:{block_id}" for document_id, block_id in blocks)


def _validate_dense_items(
    items: Sequence[RetrievedChunk],
    chunks: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
) -> tuple[str, ...]:
    ids = tuple(item.payload.chunk_id for item in items)
    if len(ids) != len(set(ids)):
        raise A45EvaluationError("dense branch returned duplicate chunks")
    if any(chunk_id not in chunks for chunk_id in ids):
        raise A45EvaluationError("dense branch returned a chunk outside the frozen index")
    if any(
        item.payload.knowledge_base_id != knowledge_base_id
        or item.point_id != point_id_for_chunk(item.payload.chunk_id)
        for item in items
    ):
        raise A45EvaluationError("dense branch returned another knowledge base")
    return ids


def _chunk_ranked_candidate(item: RerankedChunk, rank: int) -> A45RankedCandidate:
    payload = item.chunk.payload
    return A45RankedCandidate(
        candidate_id=payload.chunk_id,
        candidate_kind="chunk",
        rank=rank,
        score=float(item.reranker_score),
        source="dense",
        knowledge_base_id=payload.knowledge_base_id,
        document_id=payload.document_id,
        chunk_id=payload.chunk_id,
        page_start=payload.page_start,
        page_end=payload.page_end,
        source_block_ids=list(payload.source_block_ids),
        section_path=list(payload.section_path),
        asset_refs=list(payload.asset_refs),
        modality=None,
        representation_ids=[],
        branch_ranks={"dense": item.dense_rank},
    )


def _unified_ranked_candidate(item: UnifiedCandidate, rank: int) -> A45RankedCandidate:
    branch_ranks = {contribution.branch: contribution.rank for contribution in item.branches}
    return A45RankedCandidate(
        candidate_id=item.candidate_id,
        candidate_kind=item.candidate_kind,
        rank=rank,
        score=float(item.final_reranker_score or 0.0),
        source=item.source,
        knowledge_base_id=item.knowledge_base_id,
        document_id=item.document_id,
        chunk_id=item.chunk_id,
        evidence_id=item.evidence_id,
        page_start=item.page_start,
        page_end=item.page_end,
        source_block_ids=list(item.source_block_ids),
        section_path=list(item.section_path),
        asset_refs=list(item.asset_refs),
        modality=item.modality,
        representation_ids=list(item.representation_ids),
        branch_ranks=branch_ranks,
    )


def _retrieved_evidence_candidate(item: RetrievedEvidence, rank: int) -> A45RankedCandidate:
    return A45RankedCandidate(
        candidate_id=f"evidence::{item.evidence_id}",
        candidate_kind="evidence",
        rank=rank,
        score=float(item.score),
        source="multimodal",
        knowledge_base_id=item.knowledge_base_id,
        document_id=item.document_id,
        evidence_id=item.evidence_id,
        page_start=item.page_start,
        page_end=item.page_end,
        source_block_ids=list(item.source_block_ids),
        section_path=list(item.section_path),
        asset_refs=list(item.asset_refs),
        modality=item.modality,
        representation_ids=[
            representation.representation_id for representation in item.representations
        ],
        branch_ranks={"multimodal": item.rank},
    )


def _observation_from_candidates(
    candidates: Sequence[A45RankedCandidate],
    *,
    metrics: Mapping[str, float],
    branch_candidates: Mapping[str, Sequence[str]],
    branch_source_blocks: Mapping[str, Mapping[str, Sequence[str]]],
    branch_status: Mapping[str, str],
    branch_latency_ms: Mapping[str, float],
    latency_breakdown_ms: Mapping[str, float],
    total_latency_ms: float,
    branch_audit: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> A45SystemObservation:
    return A45SystemObservation(
        status="success",
        ranked_candidate_ids=[candidate.candidate_id for candidate in candidates],
        ranked_candidates=list(candidates),
        metrics=dict(metrics),
        branch_candidates={branch: list(values) for branch, values in branch_candidates.items()},
        branch_source_blocks={
            branch: {key: list(blocks) for key, blocks in values.items()}
            for branch, values in branch_source_blocks.items()
        },
        branch_status=dict(branch_status),
        branch_latency_ms=dict(branch_latency_ms),
        latency_breakdown_ms=dict(latency_breakdown_ms),
        total_latency_ms=total_latency_ms,
        branch_audit={
            branch: [dict(record) for record in records]
            for branch, records in (branch_audit or {}).items()
        },
    )


def _general_metrics(
    candidates: Sequence[A45RankedCandidate],
    case: FrozenEvalCase,
) -> dict[str, float]:
    tokens = [candidate.chunk_id or candidate.candidate_id for candidate in candidates]
    source_blocks = {
        token: {(str(candidate.document_id), block_id) for block_id in candidate.source_block_ids}
        for token, candidate in zip(tokens, candidates, strict=True)
    }
    return ranking_metrics(tokens, case.relevant_chunk_ids, source_blocks, case.gold_source_blocks)


def _evidence_ids(candidates: Sequence[A45RankedCandidate]) -> list[str]:
    return [candidate.evidence_id for candidate in candidates if candidate.evidence_id is not None]


def _branch_raw_snapshot(
    branch: str,
    items: Sequence[object],
) -> tuple[list[str], dict[str, list[str]]]:
    ids: list[str] = []
    blocks: dict[str, list[str]] = {}
    for item in items:
        token: str | None = None
        source_block_ids: Sequence[str] = ()
        document_id: UUID | None = None
        if branch == "dense" and isinstance(item, RetrievedChunk):
            token = item.payload.chunk_id
            source_block_ids = item.payload.source_block_ids
            document_id = item.payload.document_id
        elif branch == "sparse" and isinstance(item, SparseSearchHit):
            token = item.chunk_id
            source_block_ids = item.source_block_ids
            document_id = item.document_id
        elif branch == "graph":
            evidence = getattr(item, "evidence", None)
            if evidence is not None:
                token = evidence.chunk_id
                source_block_ids = evidence.source_block_ids
                document_id = evidence.document_id
        elif branch == "multimodal" and isinstance(item, RetrievedEvidence):
            token = f"evidence::{item.evidence_id}"
            source_block_ids = item.source_block_ids
            document_id = item.document_id
        if token is None or document_id is None or token in blocks:
            continue
        ids.append(token)
        blocks[token] = [f"{document_id}:{block_id}" for block_id in source_block_ids]
    return ids, blocks


def _branch_audit_snapshot(
    branch: str,
    items: Sequence[object],
) -> list[dict[str, object]]:
    """Keep branch-native scores and metadata without persisting raw content."""

    records: list[dict[str, object]] = []
    for rank, item in enumerate(items, start=1):
        record: dict[str, object] = {"rank": rank}
        if branch == "dense" and isinstance(item, RetrievedChunk):
            record.update(
                {
                    "candidate_id": item.payload.chunk_id,
                    "native_score": float(item.score),
                    "chunk_id": item.payload.chunk_id,
                }
            )
        elif branch == "sparse" and isinstance(item, SparseSearchHit):
            record.update(
                {
                    "candidate_id": item.chunk_id,
                    "chunk_id": item.chunk_id,
                    "bm25_score": float(item.bm25_score),
                    "matched_terms": list(item.matched_terms),
                }
            )
        elif branch == "graph":
            evidence = getattr(item, "evidence", None)
            if evidence is None:
                continue
            paths = getattr(item, "paths", ())
            record.update(
                {
                    "candidate_id": evidence.chunk_id,
                    "evidence_id": evidence.evidence_id,
                    "chunk_id": evidence.chunk_id,
                    "graph_score": float(getattr(item, "score", 0.0)),
                    "seed_entity_id": getattr(item, "seed_entity_id", None),
                    "retrieval_reason": getattr(item, "retrieval_reason", None),
                    "hop": min(
                        (int(getattr(path, "hop_distance", 0)) for path in paths),
                        default=0,
                    ),
                    "path": [path.model_dump(mode="json") for path in paths],
                }
            )
        elif branch == "multimodal" and isinstance(item, RetrievedEvidence):
            representation_hits = list(item.representations)
            record.update(
                {
                    "candidate_id": f"evidence::{item.evidence_id}",
                    "evidence_id": item.evidence_id,
                    "modality": item.modality,
                    "representation_id": item.representation_id,
                    "representation_ids": [hit.representation_id for hit in representation_hits],
                    "similarity": float(item.score),
                    "representations": [
                        {
                            "representation_id": hit.representation_id,
                            "rank": hit.rank,
                            "similarity": float(hit.score),
                        }
                        for hit in representation_hits
                    ],
                }
            )
        else:
            continue
        records.append(record)
    return records


def _branch_gold_recovery(
    observation: A45SystemObservation,
    case: FrozenEvalCase,
) -> dict[str, dict[str, list[str]]]:
    gold_blocks = set(_source_block_strings(case.gold_source_blocks))
    result: dict[str, dict[str, list[str]]] = {}
    for branch in BRANCHES:
        ids = observation.branch_candidates.get(branch, [])
        chunk_ids = sorted(set(ids) & set(case.relevant_chunk_ids))
        block_ids = sorted(
            {
                block
                for token in ids
                for block in observation.branch_source_blocks.get(branch, {}).get(token, [])
                if block in gold_blocks
            }
        )
        result[branch] = {
            "candidate_ids": list(ids),
            "gold_chunk_ids": chunk_ids,
            "gold_source_blocks": block_ids,
        }
    return result


def _build_unified_service(
    runtime: _A45Runtime,
    enabled: set[str],
) -> tuple[UnifiedRetrievalService, dict[str, _RecordingBranch]]:
    recordings: dict[str, _RecordingBranch] = {}
    delegates: dict[str, Any] = {
        "dense": runtime.dense,
        "sparse": runtime.sparse,
        "graph": runtime.graph,
        "multimodal": runtime.multimodal,
    }
    empties: dict[str, Any] = {
        "sparse": _EmptySparse(),
        "graph": _EmptyGraph(),
        "multimodal": _EmptyMultimodal(),
    }
    for branch in BRANCHES:
        delegate = (
            delegates[branch] if branch in enabled else empties.get(branch, delegates[branch])
        )
        recordings[branch] = _RecordingBranch(delegate)
    config = UnifiedRetrievalConfig(
        dense_candidate_limit=runtime.protocol.dense_candidate_limit,
        sparse_candidate_limit=runtime.protocol.sparse_candidate_limit,
        graph_candidate_limit=runtime.protocol.graph_candidate_limit,
        multimodal_candidate_limit=runtime.protocol.multimodal_candidate_limit,
        candidate_pool_limit=runtime.protocol.candidate_pool_limit,
        result_limit=runtime.protocol.result_limit,
        rerank_text_max_chars=runtime.protocol.rerank_text_max_chars,
        failure_mode=runtime.protocol.failure_mode,
        reranker_model_id=runtime.protocol.reranker_model,
    )
    service = UnifiedRetrievalService(
        recordings["dense"],
        recordings["sparse"],
        recordings["graph"],
        recordings["multimodal"],
        runtime.reranker,
        chunk_lookup=runtime.qdrant,
        config=config,
    )
    return service, recordings


def _branch_observations_from_unified(
    result: UnifiedRetrievalResult,
    recordings: Mapping[str, _RecordingBranch],
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, list[str]]],
    dict[str, str],
    dict[str, float],
    dict[str, list[dict[str, object]]],
]:
    branch_candidates: dict[str, list[str]] = {}
    branch_source_blocks: dict[str, dict[str, list[str]]] = {}
    branch_audit: dict[str, list[dict[str, object]]] = {}
    for branch in BRANCHES:
        ids, blocks = _branch_raw_snapshot(branch, recordings[branch].last_items)
        branch_candidates[branch] = ids
        branch_source_blocks[branch] = blocks
        branch_audit[branch] = _branch_audit_snapshot(branch, recordings[branch].last_items)
    return (
        branch_candidates,
        branch_source_blocks,
        {branch.branch: branch.status for branch in result.branches},
        {branch.branch: branch.latency_ms for branch in result.branches},
        branch_audit,
    )


def _evaluate_dense_general_case(
    runtime: _A45Runtime,
    case: FrozenEvalCase,
    chunks: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
) -> A45SystemObservation:
    started = time.perf_counter()
    dense_result = runtime.dense.search(
        case.item.query,
        limit=runtime.protocol.dense_candidate_limit,
        knowledge_base_id=knowledge_base_id,
    )
    dense_ids = _validate_dense_items(dense_result.items, chunks, knowledge_base_id)
    dense_elapsed = (time.perf_counter() - started) * 1_000
    reranked = runtime.reranking.rerank(
        case.item.query,
        dense_result.items,
        limit=runtime.protocol.dense_rerank_limit,
    )
    total_elapsed = (time.perf_counter() - started) * 1_000
    candidates = [_chunk_ranked_candidate(item, rank) for rank, item in enumerate(reranked, 1)]
    source_blocks = {
        chunk_id: [
            f"{chunks[chunk_id].document_id}:{block_id}"
            for block_id in chunks[chunk_id].source_block_ids
        ]
        for chunk_id in dense_ids
    }
    return _observation_from_candidates(
        candidates,
        metrics=_general_metrics(candidates, case),
        branch_candidates={"dense": list(dense_ids), "sparse": [], "graph": [], "multimodal": []},
        branch_source_blocks={"dense": source_blocks, "sparse": {}, "graph": {}, "multimodal": {}},
        branch_status={
            "dense": "success",
            "sparse": "empty",
            "graph": "empty",
            "multimodal": "empty",
        },
        branch_latency_ms={"dense": dense_elapsed, "sparse": 0.0, "graph": 0.0, "multimodal": 0.0},
        latency_breakdown_ms={
            "dense": dense_elapsed,
            "final_reranker": runtime.reranker.last_score_ms,
            "total": total_elapsed,
        },
        total_latency_ms=total_elapsed,
        branch_audit={"dense": _branch_audit_snapshot("dense", dense_result.items)},
    )


async def _evaluate_unified_general_cases(
    runtime: _A45Runtime,
    cases: Sequence[FrozenEvalCase],
    chunks: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
    enabled: set[str],
) -> list[A45SystemObservation]:
    service, recordings = _build_unified_service(runtime, enabled)
    observations: list[A45SystemObservation] = []
    for case in cases:
        result = await service.search(case.item.query, knowledge_base_id, failure_mode="strict")
        candidates = [
            _unified_ranked_candidate(item, rank) for rank, item in enumerate(result.items, 1)
        ]
        branch_candidates, branch_blocks, branch_status, branch_latency, branch_audit = (
            _branch_observations_from_unified(result, recordings)
        )
        observations.append(
            _observation_from_candidates(
                candidates,
                metrics=_general_metrics(candidates, case),
                branch_candidates=branch_candidates,
                branch_source_blocks=branch_blocks,
                branch_status=branch_status,
                branch_latency_ms=branch_latency,
                latency_breakdown_ms={
                    **{branch: value for branch, value in branch_latency.items()},
                    "normalization": result.normalization_latency_ms,
                    "final_reranker": result.rerank_latency_ms,
                    "total": result.total_latency_ms,
                },
                total_latency_ms=result.total_latency_ms,
                branch_audit=branch_audit,
            )
        )
    return observations


def _multimodal_authoritative_match(
    candidate: A45RankedCandidate,
    item: A45MultimodalEvalItem,
) -> bool:
    return (
        candidate.evidence_id == item.evidence_id
        and candidate.modality == item.modality
        and candidate.knowledge_base_id == item.knowledge_base_id
        and candidate.document_id == item.document_id
        and candidate.page_start == item.page_start
        and candidate.page_end == item.page_end
        and candidate.source_block_ids == item.source_block_ids
        and candidate.asset_refs == item.asset_refs
    )


def _citation_is_valid(candidate: UnifiedCandidate, item: A45MultimodalEvalItem) -> bool:
    selection = assemble_context([candidate], budget_chars=runtime_context_budget(candidate))
    if len(selection.items) != 1:
        return False
    citation = selection.items[0].citation
    return (
        citation.candidate_kind == "evidence"
        and citation.evidence_id == item.evidence_id
        and citation.modality == item.modality
        and citation.document_id == item.document_id
        and citation.page_start == item.page_start
        and citation.page_end == item.page_end
        and citation.source_block_ids == item.source_block_ids
        and citation.asset_refs == item.asset_refs
        and bool(citation.representation_ids)
    )


def runtime_context_budget(candidate: UnifiedCandidate) -> int:
    """Use the current A4.4 bounded context policy for citation-only validation."""

    return max(6_000, len(candidate.rerank_text))


def _evaluate_dense_multimodal_case(
    runtime: _A45Runtime,
    item: A45MultimodalEvalItem,
    chunks: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
) -> A45SystemObservation:
    started = time.perf_counter()
    result = runtime.dense.search(
        item.query,
        limit=runtime.protocol.dense_candidate_limit,
        knowledge_base_id=knowledge_base_id,
    )
    _validate_dense_items(result.items, chunks, knowledge_base_id)
    elapsed = (time.perf_counter() - started) * 1_000
    reranked = runtime.reranking.rerank(
        item.query,
        result.items,
        limit=runtime.protocol.dense_rerank_limit,
    )
    total_elapsed = (time.perf_counter() - started) * 1_000
    candidates = [_chunk_ranked_candidate(value, rank) for rank, value in enumerate(reranked, 1)]
    source_blocks = {
        value.payload.chunk_id: [
            f"{chunks[value.payload.chunk_id].document_id}:{block_id}"
            for block_id in chunks[value.payload.chunk_id].source_block_ids
        ]
        for value in result.items
        if value.payload.chunk_id in chunks
    }
    return _observation_from_candidates(
        candidates,
        metrics=evidence_metrics([], [item.evidence_id]),
        branch_candidates={
            "dense": [value.payload.chunk_id for value in result.items],
            "sparse": [],
            "graph": [],
            "multimodal": [],
        },
        branch_source_blocks={
            "dense": source_blocks,
            "sparse": {},
            "graph": {},
            "multimodal": {},
        },
        branch_status={
            "dense": "success",
            "sparse": "empty",
            "graph": "empty",
            "multimodal": "empty",
        },
        branch_latency_ms={"dense": elapsed, "sparse": 0.0, "graph": 0.0, "multimodal": 0.0},
        latency_breakdown_ms={
            "dense": elapsed,
            "final_reranker": runtime.reranker.last_score_ms,
            "total": total_elapsed,
        },
        total_latency_ms=total_elapsed,
        branch_audit={"dense": _branch_audit_snapshot("dense", result.items)},
    )


async def _evaluate_multimodal_cases(
    runtime: _A45Runtime,
    items: Sequence[A45MultimodalEvalItem],
    chunks: Mapping[str, IndexedChunk],
    knowledge_base_id: UUID,
) -> list[A45MultimodalQueryRecord]:
    unified, recordings = _build_unified_service(runtime, set(BRANCHES))
    records: list[A45MultimodalQueryRecord] = []
    for item in items:
        dense_observation = _evaluate_dense_multimodal_case(
            runtime,
            item,
            chunks,
            knowledge_base_id,
        )
        started = time.perf_counter()
        multimodal_result = runtime.multimodal.search(
            item.query,
            knowledge_base_id=knowledge_base_id,
            top_k=runtime.protocol.result_limit,
        )
        multimodal_elapsed = (time.perf_counter() - started) * 1_000
        mm_candidates = [
            _retrieved_evidence_candidate(value, rank)
            for rank, value in enumerate(multimodal_result.items, 1)
        ]
        mm_observation = _observation_from_candidates(
            mm_candidates,
            metrics=evidence_metrics(_evidence_ids(mm_candidates), [item.evidence_id]),
            branch_candidates={
                "dense": [],
                "sparse": [],
                "graph": [],
                "multimodal": [candidate.candidate_id for candidate in mm_candidates],
            },
            branch_source_blocks={
                "dense": {},
                "sparse": {},
                "graph": {},
                "multimodal": {
                    candidate.candidate_id: [
                        f"{candidate.document_id}:{block}" for block in candidate.source_block_ids
                    ]
                    for candidate in mm_candidates
                },
            },
            branch_status={
                "dense": "empty",
                "sparse": "empty",
                "graph": "empty",
                "multimodal": "success",
            },
            branch_latency_ms={
                "dense": 0.0,
                "sparse": 0.0,
                "graph": 0.0,
                "multimodal": multimodal_elapsed,
            },
            latency_breakdown_ms={"multimodal": multimodal_elapsed, "total": multimodal_elapsed},
            total_latency_ms=multimodal_elapsed,
            branch_audit={
                "multimodal": _branch_audit_snapshot("multimodal", multimodal_result.items)
            },
        )
        unified_result = await unified.search(item.query, knowledge_base_id, failure_mode="strict")
        unified_candidates = [
            _unified_ranked_candidate(value, rank)
            for rank, value in enumerate(unified_result.items, 1)
        ]
        branch_candidates, branch_blocks, branch_status, branch_latency, branch_audit = (
            _branch_observations_from_unified(unified_result, recordings)
        )
        unified_observation = _observation_from_candidates(
            unified_candidates,
            metrics=evidence_metrics_for_candidates(unified_candidates, [item.evidence_id]),
            branch_candidates=branch_candidates,
            branch_source_blocks=branch_blocks,
            branch_status=branch_status,
            branch_latency_ms=branch_latency,
            latency_breakdown_ms={
                **branch_latency,
                "normalization": unified_result.normalization_latency_ms,
                "final_reranker": unified_result.rerank_latency_ms,
                "total": unified_result.total_latency_ms,
            },
            total_latency_ms=unified_result.total_latency_ms,
            branch_audit=branch_audit,
        )
        citation_status: dict[str, Literal["valid", "not_applicable", "invalid"]] = {
            "dense_bge": "not_applicable",
            "multimodal_representation": "not_applicable",
            "unified": "not_applicable",
        }
        for candidate in mm_candidates:
            if candidate.evidence_id == item.evidence_id:
                citation_status["multimodal_representation"] = (
                    "valid" if (_multimodal_authoritative_match(candidate, item)) else "invalid"
                )
                break
        for candidate in unified_result.items:
            if candidate.evidence_id == item.evidence_id:
                citation_status["unified"] = (
                    "valid" if _citation_is_valid(candidate, item) else "invalid"
                )
                break
        records.append(
            A45MultimodalQueryRecord(
                eval_id=item.eval_id,
                split=item.split,
                query=item.query,
                subject=item.subject,
                query_type=item.query_type,
                modality=item.modality,
                gold_evidence_id=item.evidence_id,
                knowledge_base_id=item.knowledge_base_id,
                document_id=item.document_id,
                systems={
                    "dense_bge": dense_observation,
                    "multimodal_representation": mm_observation,
                    "unified": unified_observation,
                },
                citation_validation=citation_status,
            )
        )
    return records


def _summary_for_observations(
    observations: Sequence[A45SystemObservation],
) -> dict[str, object]:
    rows = [observation.metrics for observation in observations if observation.status == "success"]
    latencies = [observation.total_latency_ms for observation in observations]
    return {
        "query_count": len(observations),
        "metrics": aggregate_metric_rows(rows),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "branch_latency_ms": _aggregate_latency_maps(observations, "branch_latency_ms"),
        "latency_breakdown_ms": _aggregate_latency_maps(observations, "latency_breakdown_ms"),
    }


def _aggregate_latency_maps(
    observations: Sequence[A45SystemObservation],
    attribute: Literal["branch_latency_ms", "latency_breakdown_ms"],
) -> dict[str, float | None]:
    keys = sorted({key for observation in observations for key in getattr(observation, attribute)})
    return {
        key: statistics.fmean(
            getattr(observation, attribute)[key]
            for observation in observations
            if key in getattr(observation, attribute)
        )
        if any(key in getattr(observation, attribute) for observation in observations)
        else None
        for key in keys
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * percentile))]


def _delta_metrics(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float | None]]:
    absolute: dict[str, float] = {}
    relative: dict[str, float | None] = {}
    for name, value in metrics.items():
        if name not in baseline:
            continue
        delta = value - baseline[name]
        absolute[name] = delta
        relative[name] = delta / baseline[name] if baseline[name] else None
    return absolute, relative


def _first_relevant_rank(
    observation: A45SystemObservation,
    case: FrozenEvalCase,
) -> int | None:
    relevant = set(case.relevant_chunk_ids)
    for rank, candidate in enumerate(observation.ranked_candidates, start=1):
        if candidate.chunk_id in relevant:
            return rank
    return None


def _compare_rank_values(before: int | None, after: int | None) -> str:
    """Classify a first-relevant-rank change; lower rank is better."""

    if before == after:
        return "unchanged"
    if after is not None and (before is None or after < before):
        return "improved"
    return "regressed"


def _contribution_summary(
    baseline: Sequence[A45SystemObservation],
    candidate: Sequence[A45SystemObservation],
    cases: Sequence[FrozenEvalCase],
) -> dict[str, object]:
    if len(baseline) != len(candidate) or len(baseline) != len(cases):
        raise ValueError("contribution rows must have equal lengths")
    summary: dict[str, object] = {}
    for k in METRIC_KS:
        key = f"hit@{k}"
        counts = {"improved": 0, "unchanged": 0, "regressed": 0}
        for before, after in zip(baseline, candidate, strict=True):
            before_hit = bool(before.metrics.get(key, 0.0))
            after_hit = bool(after.metrics.get(key, 0.0))
            if after_hit and not before_hit:
                counts["improved"] += 1
            elif before_hit and not after_hit:
                counts["regressed"] += 1
            else:
                counts["unchanged"] += 1
        summary[str(k)] = counts
    mrr_counts = {"improved": 0, "unchanged": 0, "regressed": 0}
    first_rank_counts = {"improved": 0, "unchanged": 0, "regressed": 0}
    for before, after, case in zip(baseline, candidate, cases, strict=True):
        before_mrr = before.metrics.get("mrr", 0.0)
        after_mrr = after.metrics.get("mrr", 0.0)
        if after_mrr > before_mrr:
            mrr_counts["improved"] += 1
        elif after_mrr < before_mrr:
            mrr_counts["regressed"] += 1
        else:
            mrr_counts["unchanged"] += 1
        rank_change = _compare_rank_values(
            _first_relevant_rank(before, case),
            _first_relevant_rank(after, case),
        )
        first_rank_counts[rank_change] += 1
    summary["mrr"] = mrr_counts
    summary["first_relevant_rank"] = first_rank_counts
    summary["mean_mrr_delta"] = (
        statistics.fmean(
            after.metrics.get("mrr", 0.0) - before.metrics.get("mrr", 0.0)
            for before, after in zip(baseline, candidate, strict=True)
        )
        if baseline
        else 0.0
    )
    return summary


def _branch_recovery_summary(
    records: Sequence[A45GeneralQueryRecord],
) -> dict[str, object]:
    """Summarize branch candidate recovery without treating it as final ranking."""

    branch_names = ("dense", "sparse", "graph", "multimodal")

    def summarize(field: str) -> dict[str, object]:
        recovered_sets: list[set[str]] = []
        for record in records:
            recovered_sets.append(
                {
                    branch
                    for branch in branch_names
                    if record.branch_gold_recovery.get(branch, {}).get(field, [])
                }
            )
        branch_counts = {
            branch: sum(branch in recovered for recovered in recovered_sets)
            for branch in branch_names
        }
        exclusive = {
            f"{branch}_only": sum(recovered == {branch} for recovered in recovered_sets)
            for branch in branch_names
        }
        exclusive["multiple"] = sum(len(recovered) > 1 for recovered in recovered_sets)
        exclusive["none"] = sum(not recovered for recovered in recovered_sets)
        return {
            "branch_query_counts": branch_counts,
            "exclusive_query_counts": exclusive,
            "multi_branch_query_count": exclusive["multiple"],
        }

    return {
        "query_count": len(records),
        "gold_chunk_recovery": summarize("gold_chunk_ids"),
        "gold_source_block_recovery": summarize("gold_source_blocks"),
        "interpretation": (
            "candidate recovery only; a branch hit is not a final retrieval improvement "
            "unless it survives the evaluated final ranking"
        ),
    }


def _general_query_records(
    cases: Sequence[FrozenEvalCase],
    observations_by_system: Mapping[str, Sequence[A45SystemObservation]],
    split: Literal["dev", "test"],
) -> list[A45GeneralQueryRecord]:
    records: list[A45GeneralQueryRecord] = []
    for index, case in enumerate(cases):
        systems = {
            system: observations[index] for system, observations in observations_by_system.items()
        }
        records.append(
            A45GeneralQueryRecord(
                eval_id=case.item.item_id,
                split=split,
                query=case.item.query,
                subject=case.item.subject,
                query_type=case.item.query_type,
                gold_relevant_chunk_ids=sorted(case.relevant_chunk_ids),
                gold_source_blocks=_source_block_strings(sorted(case.gold_source_blocks)),
                branch_gold_recovery=_branch_gold_recovery(systems["unified"], case),
                systems=systems,
            )
        )
    return records


def _validate_a25_baseline(
    observations: Sequence[A45SystemObservation],
    cases: Sequence[FrozenEvalCase],
    split: Literal["dev", "test"],
) -> None:
    expected = {
        "dev": {
            "hit@1": 0.7916666666666666,
            "hit@3": 0.875,
            "hit@5": 0.9166666666666666,
            "hit@10": 0.9722222222222222,
            "mrr": 0.8493606735631895,
        },
        "test": {
            "hit@1": 0.7777777777777778,
            "hit@3": 0.8333333333333334,
            "hit@5": 0.8611111111111112,
            "hit@10": 0.8611111111111112,
            "mrr": 0.8078703703703703,
        },
    }[split]
    actual = aggregate_metric_rows([observation.metrics for observation in observations])
    for name, value in expected.items():
        if not math.isclose(actual.get(name, -1.0), value, rel_tol=0.0, abs_tol=1e-5):
            raise A45EvaluationError(
                f"frozen A2.5 Dense+BGE {split} baseline was not reproduced for {name}: "
                f"{actual.get(name)!r} != {value!r}"
            )
    if len(observations) != (72 if split == "dev" else 36) or len(cases) != len(observations):
        raise A45EvaluationError("A2.1 split count is inconsistent during baseline reproduction")


def _environment(runtime: _A45Runtime, preflight: A45Preflight) -> dict[str, object]:
    cuda_available = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_mb: float | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except ImportError:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": _package_version("torch"),
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_total_vram_mb": gpu_memory_mb,
        "preflight": preflight.model_dump(mode="json"),
    }


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_retrieval_evaluation(
    *,
    knowledge_base_id: UUID,
    split: RunSplit = "both",
    protocol: A45Protocol | None = None,
    chunk_index_path: Path = DEFAULT_CHUNK_INDEX,
    dataset_path: Path = DEFAULT_DATASET,
    materialized_path: Path = DEFAULT_MATERIALIZED,
    multimodal_dataset_path: Path = DEFAULT_MULTIMODAL_DATASET,
    multimodal_manifest_path: Path = DEFAULT_MULTIMODAL_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run the five general systems and the three Evidence systems read-only."""

    if split not in SUPPORTED_SPLITS:
        raise A45EvaluationError(f"unsupported evaluation split: {split}")
    protocol = protocol or A45Protocol()
    selected_settings = settings or get_settings()
    frozen_store_manifest = load_frozen_store_manifest()
    if frozen_store_manifest.knowledge_base_id != knowledge_base_id:
        raise A45EvaluationError("A4.5 frozen-store manifest is scoped to another KB")
    chunks = load_frozen_chunk_index(chunk_index_path)
    all_cases_by_split = {
        name: load_frozen_eval_cases(
            name,
            dataset_path=dataset_path,
            materialized_path=materialized_path,
            chunk_index=chunks,
        )[0]
        for name in ("dev", "test")
    }
    requested = ("dev", "test") if split == "both" else (split,)
    cases_by_split = {name: all_cases_by_split[name] for name in requested}
    multimodal_items, multimodal_manifest = load_multimodal_dataset(
        multimodal_dataset_path,
        multimodal_manifest_path,
    )
    if multimodal_manifest.knowledge_base_ids != [knowledge_base_id]:
        raise A45EvaluationError("multimodal benchmark is not scoped to the requested KB")

    runtime = _build_runtime(selected_settings, protocol)
    try:
        preflight = _preflight(
            runtime,
            knowledge_base_id=knowledge_base_id,
            chunks=chunks,
            cases_by_split=all_cases_by_split,
            dataset_path=dataset_path,
            chunk_index_path=chunk_index_path,
            frozen_manifest=frozen_store_manifest,
            frozen_manifest_path=DEFAULT_FROZEN_STORE_MANIFEST,
        )
        _warmup_runtime(runtime, knowledge_base_id)
        general_records: list[A45GeneralQueryRecord] = []
        summaries: dict[str, dict[str, object]] = {}
        contributions: dict[str, object] = {}
        type_slices: dict[str, object] = {}
        for split_name, split_cases in cases_by_split.items():
            baseline = [
                _evaluate_dense_general_case(runtime, case, chunks, knowledge_base_id)
                for case in split_cases
            ]
            _validate_a25_baseline(baseline, split_cases, split_name)
            observations_by_system: dict[str, list[A45SystemObservation]] = {"dense_bge": baseline}
            enabled_by_system = {
                "dense_sparse_bge": {"dense", "sparse"},
                "dense_graph_bge": {"dense", "graph"},
                "dense_multimodal_bge": {"dense", "multimodal"},
                "unified": set(BRANCHES),
            }
            for system, enabled in enabled_by_system.items():
                observations_by_system[system] = asyncio.run(
                    _evaluate_unified_general_cases(
                        runtime,
                        split_cases,
                        chunks,
                        knowledge_base_id,
                        enabled,
                    )
                )
            split_records = _general_query_records(split_cases, observations_by_system, split_name)
            general_records.extend(split_records)
            baseline_summary = _summary_for_observations(baseline)
            summaries[split_name] = {}
            for system, observations in observations_by_system.items():
                summary = _summary_for_observations(observations)
                metrics = summary["metrics"]
                baseline_metrics = baseline_summary["metrics"]
                absolute, relative = _delta_metrics(metrics, baseline_metrics)  # type: ignore[arg-type]
                summaries[split_name][system] = {
                    **summary,
                    "absolute_delta_vs_dense": absolute,
                    "relative_delta_vs_dense": relative,
                }
                if system != "dense_bge":
                    contributions[f"{split_name}:{system}"] = _contribution_summary(
                        baseline,
                        observations,
                        split_cases,
                    )
            type_slices[split_name] = {
                system: {
                    query_type: _summary_for_observations(
                        [
                            observation
                            for case, observation in zip(split_cases, observations, strict=True)
                            if case.item.query_type == query_type
                        ]
                    )
                    for query_type in sorted({case.item.query_type for case in split_cases})
                }
                for system, observations in observations_by_system.items()
            }

        multimodal_requested = [
            item for item in multimodal_items if split == "both" or item.split == split
        ]
        multimodal_records = asyncio.run(
            _evaluate_multimodal_cases(
                runtime,
                multimodal_requested,
                chunks,
                knowledge_base_id,
            )
        )
        multimodal_summary: dict[str, object] = {}
        for name in MULTIMODAL_SYSTEMS:
            observations = [record.systems[name] for record in multimodal_records]
            by_modality = {
                modality: _summary_for_observations(
                    [
                        record.systems[name]
                        for record in multimodal_records
                        if record.modality == modality
                    ]
                )
                for modality in MM_MODALITIES
            }
            multimodal_summary[name] = {
                "all": _summary_for_observations(observations),
                "by_split": {
                    split_name: _summary_for_observations(
                        [
                            record.systems[name]
                            for record in multimodal_records
                            if record.split == split_name
                        ]
                    )
                    for split_name in ("dev", "test")
                },
                "by_modality": by_modality,
                "citation_validation": dict(
                    Counter(record.citation_validation[name] for record in multimodal_records)
                ),
            }
        after_store_manifest = _current_frozen_store_manifest(
            runtime,
            knowledge_base_id=knowledge_base_id,
            chunks=chunks,
            cases_by_split=all_cases_by_split,
            dataset_path=dataset_path,
            chunk_index_path=chunk_index_path,
        )
        store_state_audit = _compare_store_state(
            preflight.store_state_before,
            _store_state_snapshot(after_store_manifest),
        )
        _ensure_store_state_unchanged(store_state_audit)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "general-query-results.jsonl", general_records)
        _write_jsonl(output_dir / "multimodal-query-results.jsonl", multimodal_records)
        aggregate = {
            "schema_version": A45_SCHEMA_VERSION,
            "protocol": protocol.model_dump(mode="json"),
            "protocol_fingerprint": protocol.fingerprint,
            "frozen_inputs": {
                "a21_dataset_sha256": _sha256_file(dataset_path),
                "a21_chunk_index_sha256": _sha256_file(chunk_index_path),
                "a25_results_sha256": _sha256_file(DEFAULT_A25_RESULTS)
                if DEFAULT_A25_RESULTS.is_file()
                else None,
                "a25_manifest_sha256": _sha256_file(DEFAULT_A25_MANIFEST)
                if DEFAULT_A25_MANIFEST.is_file()
                else None,
                "a25_profile_version": A25_PROFILE_VERSION,
                "a35_rrf_k": protocol.a35_rrf_k,
                "a36_run_id": A36_RUN_ID,
                "a36_run_manifest_sha256": (
                    _sha256_file(A36_RUN_MANIFEST) if A36_RUN_MANIFEST.is_file() else None
                ),
                "a36_corpus_snapshot_sha256": (
                    _sha256_file(Path("docs/benchmarks/a3-6-corpus-input-snapshot.json"))
                    if Path("docs/benchmarks/a3-6-corpus-input-snapshot.json").is_file()
                    else None
                ),
                "a36_graph_eligible_documents": A36_GRAPH_ELIGIBLE_DOCUMENTS,
                "a36_graph_evidence_chunks": A36_GRAPH_EVIDENCE_CHUNKS,
                "multimodal_dataset_sha256": multimodal_manifest.file_sha256,
                "multimodal_dataset_fingerprint": multimodal_manifest.dataset_fingerprint,
                "frozen_store_manifest_sha256": _sha256_file(DEFAULT_FROZEN_STORE_MANIFEST),
            },
            "preflight": preflight.model_dump(mode="json"),
            "store_state_audit": store_state_audit.model_dump(mode="json"),
            "general": summaries,
            "general_all_108": {
                system: _summary_for_observations(
                    [record.systems[system] for record in general_records]
                )
                for system in GENERAL_SYSTEMS
            }
            if split == "both"
            else None,
            "general_contributions": contributions,
            "general_query_type_slices": type_slices,
            "general_branch_recovery": {
                split_name: _branch_recovery_summary(
                    [record for record in general_records if record.split == split_name]
                )
                for split_name in ("dev", "test")
            }
            | {"all": _branch_recovery_summary(general_records)},
            "multimodal": multimodal_summary,
            "multimodal_item_count": len(multimodal_requested),
            "environment": _environment(runtime, preflight),
            "limitations": [
                "A2.5 baseline is checked before ablation results are accepted.",
                (
                    "Qdrant is evaluated as the current vector-store path; "
                    "this is not an ANN benchmark."
                ),
                (
                    "Multimodal gold is authoritative Evidence identity; "
                    "representation text is not source truth."
                ),
                "No answer-generation LLM call is made by this evaluator.",
            ],
        }
        (output_dir / "aggregate.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": A45_SCHEMA_VERSION,
            "profile_version": A45_PROFILE_VERSION,
            "protocol_fingerprint": protocol.fingerprint,
            "general_query_results": str(output_dir / "general-query-results.jsonl"),
            "multimodal_query_results": str(output_dir / "multimodal-query-results.jsonl"),
            "aggregate": str(output_dir / "aggregate.json"),
            "frozen_multimodal_dataset": str(multimodal_dataset_path),
            "store_state_before_fingerprint": store_state_audit.before_fingerprint,
            "store_state_after_fingerprint": store_state_audit.after_fingerprint,
            "changed_stores": store_state_audit.changed_stores,
            "changed_components": store_state_audit.changed_components,
            "stores_mutated": store_state_audit.stores_mutated,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"manifest": manifest, "aggregate": aggregate}
    finally:
        runtime.close()


__all__ = [
    "DEFAULT_CORPUS_MANIFEST",
    "DEFAULT_DATASET",
    "DEFAULT_EVIDENCE_DIR",
    "DEFAULT_FROZEN_STORE_MANIFEST",
    "DEFAULT_MULTIMODAL_DATASET",
    "DEFAULT_MULTIMODAL_MANIFEST",
    "DEFAULT_OUTPUT",
    "A45EvaluationError",
    "A45FrozenStoreManifest",
    "A45MultimodalDatasetManifest",
    "A45MultimodalEvalItem",
    "A45Protocol",
    "A45RankedCandidate",
    "A45StoreStateAudit",
    "A45StoreStateSnapshot",
    "A45SystemObservation",
    "aggregate_metric_rows",
    "build_multimodal_dataset",
    "compare_query_hits",
    "evidence_hit_at_k",
    "evidence_metrics",
    "evidence_metrics_for_candidates",
    "evidence_mrr",
    "load_frozen_store_manifest",
    "load_multimodal_dataset",
    "multimodal_dataset_fingerprint",
    "multimodal_eval_id",
    "run_retrieval_evaluation",
]
