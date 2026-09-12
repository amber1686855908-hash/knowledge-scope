"""Multimodal representation materialization, indexing, and retrieval.

This module deliberately sits beside the A2.3 chunk index.  It derives
searchable representations from the A4.1 evidence contract, but never changes
the authoritative canonical document or evidence objects.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError
from qdrant_client import QdrantClient, models

from knowledge_scope.evidence.lifecycle import (
    EvidenceArtifactError,
    evidence_artifact_path,
    load_evidence_artifact,
    remove_evidence_artifact,
    write_evidence_artifact,
)
from knowledge_scope.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    INDEX_PAYLOAD_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION,
    EvidenceModality,
    EvidenceRepresentation,
    MultimodalEvidence,
    MultimodalEvidenceDocument,
    RepresentationIndexPayload,
    canonical_json_bytes,
)
from knowledge_scope.evidence.service import (
    EvidenceValidationError,
    build_evidence_document,
    representation_index_payloads,
    validate_evidence_document,
)
from knowledge_scope.parsing.models import (
    CanonicalBlock,
    CanonicalDocument,
    TextBlock,
    TitleBlock,
)
from knowledge_scope.retrieval.embedding import (
    EMBEDDING_POOLING_STRATEGY,
    embedding_config_fingerprint,
)
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_DEFAULT_COLLECTION_NAME,
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    ChunkVectorPayload,
    point_id_for_chunk,
)
from knowledge_scope.shared.config import Settings

REPRESENTATION_COLLECTION_SCHEMA_VERSION = "1.0"
REPRESENTATION_COLLECTION_DEFAULT_NAME = "knowledgescope_representations_v1"
REPRESENTATION_POINT_NAMESPACE = UUID("8b7f155c-6cb0-5f3d-89f1-1e96db7ed0de")
REPRESENTATION_MATERIALIZATION_VERSION = "a4-2-structural-context-v2"
REPRESENTATION_SCROLL_PAGE_SIZE = 256
REPRESENTATION_SEARCH_LIMIT_MAX = 100
REPRESENTATION_CANDIDATE_MULTIPLIER = 3
MAX_SECTION_ITEMS = 4
MAX_SECTION_ITEM_CHARS = 160
MAX_NEIGHBOR_CHARS = 480
MAX_CONTEXT_CHARS = 1_200

RepresentationModalityFilter = Literal["all", "text", "image", "table", "formula"]


class RepresentationIndexError(RuntimeError):
    """Raised when a representation index operation cannot be trusted."""


class RepresentationCollectionConfigurationError(RepresentationIndexError):
    """Raised when the independent representation collection is incompatible."""


class RepresentationEmbeddingEncoder(Protocol):
    """Small embedding contract shared by the indexer and tests."""

    model_id: str
    model_revision: str | None
    config_fingerprint: str

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode searchable representation text."""

    def encode_query(self, query: str) -> list[float]:
        """Encode one user query."""


class IndexedRepresentationPayload(RepresentationIndexPayload):
    """A4.2 payload with collection and embedding provenance."""

    model_config = ConfigDict(extra="forbid")

    collection_schema_version: Literal["1.0"] = REPRESENTATION_COLLECTION_SCHEMA_VERSION
    evidence_schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    representation_schema_version: Literal["1.0"] = REPRESENTATION_SCHEMA_VERSION
    collection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str = Field(min_length=1)
    embedding_model_revision: str | None = None
    embedding_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    quarantined: StrictBool = False


@dataclass(frozen=True, slots=True)
class _RepresentationIndexContract:
    """Active collection and embedding contract shared by all A4.2 writes."""

    collection_name: str
    collection_schema_version: str
    evidence_schema_version: str
    representation_schema_version: str
    index_payload_schema_version: str
    materialization_version: str
    vector_dimension: int
    distance: str
    embedding_model: str
    embedding_model_revision: str | None
    embedding_pooling_strategy: str
    embedding_config_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "collection_name": self.collection_name,
            "collection_schema_version": self.collection_schema_version,
            "evidence_schema_version": self.evidence_schema_version,
            "representation_schema_version": self.representation_schema_version,
            "index_payload_schema_version": self.index_payload_schema_version,
            "materialization_version": self.materialization_version,
            "vector_dimension": self.vector_dimension,
            "distance": self.distance,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_pooling_strategy": self.embedding_pooling_strategy,
            "embedding_config_fingerprint": self.embedding_config_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class RepresentationVectorPoint:
    """One vector and its fully validated representation payload."""

    point_id: UUID
    vector: tuple[float, ...]
    payload: IndexedRepresentationPayload


@dataclass(frozen=True, slots=True)
class RepresentationIndexResult:
    """Facts returned after replacing one document's representation points."""

    document_id: UUID
    indexed_count: int
    removed_stale_count: int


@dataclass(frozen=True, slots=True)
class RepresentationVisibilitySnapshot:
    """Document-scoped visibility state used to compensate a failed deletion."""

    document_id: UUID
    knowledge_base_id: UUID
    point_states: tuple[tuple[UUID, bool], ...]


def _active_representation_index_contract(settings: Settings) -> _RepresentationIndexContract:
    """Return the single source of truth for the active A4.2 index contract."""

    return _RepresentationIndexContract(
        collection_name=settings.qdrant_representation_collection_name,
        collection_schema_version=REPRESENTATION_COLLECTION_SCHEMA_VERSION,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        representation_schema_version=REPRESENTATION_SCHEMA_VERSION,
        index_payload_schema_version=INDEX_PAYLOAD_SCHEMA_VERSION,
        materialization_version=REPRESENTATION_MATERIALIZATION_VERSION,
        vector_dimension=QDRANT_VECTOR_DIMENSION,
        distance="Cosine",
        embedding_model=QWEN_EMBEDDING_MODEL_ID,
        embedding_model_revision=settings.embedding_model_revision,
        embedding_pooling_strategy=EMBEDDING_POOLING_STRATEGY,
        embedding_config_fingerprint=embedding_config_fingerprint(settings),
    )


class RepresentationIndexReadiness(BaseModel):
    """Non-sensitive collection readiness facts."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "available", "unavailable"]
    collection_name: str
    collection_exists: bool
    vector_dimension: int | None = None
    error: str | None = None


class RetrievedRepresentation(BaseModel):
    """One raw representation hit returned by the independent index."""

    model_config = ConfigDict(extra="forbid")

    point_id: UUID
    score: float
    rank: StrictInt = Field(ge=1)
    payload: IndexedRepresentationPayload


class RepresentationEvidenceHit(BaseModel):
    """One representation contribution retained after Evidence deduplication."""

    model_config = ConfigDict(extra="forbid")

    representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    representation_type: str = Field(min_length=1)
    score: float
    rank: StrictInt = Field(ge=1)
    text: str | None = None
    reference: str | None = None


class RetrievedEvidence(BaseModel):
    """An Evidence-level result with all contributing raw representations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = INDEX_PAYLOAD_SCHEMA_VERSION
    rank: StrictInt = Field(ge=1)
    score: float
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    asset_refs: list[str]
    modality: EvidenceModality
    representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    representation_type: str = Field(min_length=1)
    text: str | None = None
    reference: str | None = None
    representations: list[RepresentationEvidenceHit] = Field(min_length=1)


class RepresentationRetrievalResult(BaseModel):
    """A bounded Evidence-level response plus the raw representation hits."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = INDEX_PAYLOAD_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    items: list[RetrievedEvidence] = Field(default_factory=list)
    raw_hits: list[RetrievedRepresentation] = Field(default_factory=list)


class RepresentationCoverageBucket(BaseModel):
    """Coverage counts for one modality/representation type pair."""

    model_config = ConfigDict(extra="forbid")

    modality: EvidenceModality
    representation_type: str
    source_evidence_count: int = Field(ge=0)
    representation_count: int = Field(ge=0)
    searchable_representation_count: int = Field(ge=0)
    indexed_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    stale_count: int = Field(ge=0)


class RepresentationCoverageAudit(BaseModel):
    """Read-only audit of source evidence and the independent collection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = INDEX_PAYLOAD_SCHEMA_VERSION
    collection_name: str
    collection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_base_id: UUID
    documents_scanned: int = Field(ge=0)
    source_evidence_count: int = Field(ge=0)
    representation_count: int = Field(ge=0)
    searchable_representation_count: int = Field(ge=0)
    indexed_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    stale_count: int = Field(ge=0)
    lineage_failure_count: int = Field(ge=0)
    lineage_failures_by_modality: dict[str, int] = Field(default_factory=dict)
    buckets: list[RepresentationCoverageBucket] = Field(default_factory=list)


def representation_collection_fingerprint(settings: Settings) -> str:
    """Return the versioned fingerprint for this independent collection."""

    return _active_representation_index_contract(settings).fingerprint


def point_id_for_representation(representation_id: str) -> UUID:
    """Return a stable UUID point ID for one A4.1 representation."""

    if not representation_id.strip():
        raise ValueError("representation_id must not be blank")
    return uuid5(
        REPRESENTATION_POINT_NAMESPACE,
        f"{REPRESENTATION_COLLECTION_SCHEMA_VERSION}:{representation_id}",
    )


def _sha256_json(payload: Mapping[str, object]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_representation_collection_role(settings: Settings) -> None:
    """Fail closed when A4.2 is pointed at the frozen A2/A3 collection."""

    if settings.qdrant_representation_collection_name == QDRANT_DEFAULT_COLLECTION_NAME:
        raise RepresentationCollectionConfigurationError(
            "the A4.2 representation collection cannot target the protected frozen "
            f"chunk collection {QDRANT_DEFAULT_COLLECTION_NAME!r}"
        )
    if settings.qdrant_collection_name == settings.qdrant_representation_collection_name:
        raise RepresentationCollectionConfigurationError(
            "the A4.2 representation collection must differ from the frozen chunk collection"
        )


_AUTHORITATIVE_PAYLOAD_FIELDS = (
    "representation_id",
    "evidence_id",
    "knowledge_base_id",
    "document_id",
    "page_start",
    "page_end",
    "source_block_ids",
    "section_path",
    "asset_refs",
    "modality",
    "representation_type",
    "text",
    "reference",
    "searchable",
    "source_fingerprint",
    "canonical_document_fingerprint",
)


def _payload_matches_authoritative(
    indexed: IndexedRepresentationPayload | RepresentationIndexPayload,
    authoritative: RepresentationIndexPayload,
) -> bool:
    """Compare every source field while ignoring only index lifecycle metadata."""

    return all(
        getattr(indexed, field) == getattr(authoritative, field)
        for field in _AUTHORITATIVE_PAYLOAD_FIELDS
    )


def _vector_config(info: Any) -> tuple[int | None, str | None]:
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if isinstance(vectors, models.VectorParams):
        return vectors.size, str(vectors.distance).casefold()
    if isinstance(vectors, dict):
        if "" in vectors:
            vectors = vectors[""]
        if isinstance(vectors, dict):
            return vectors.get("size"), str(vectors.get("distance", "")).casefold()
    return None, None


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if len(vector) != QDRANT_VECTOR_DIMENSION:
        raise RepresentationIndexError(
            f"every representation vector must have {QDRANT_VECTOR_DIMENSION} dimensions"
        )
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise RepresentationIndexError("representation vectors must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise RepresentationIndexError(
            "representation vectors must use the active L2-normalized embedding contract"
        )
    return values


class QdrantRepresentationStore:
    """Synchronous Qdrant adapter for the independent A4.2 collection."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        validate_representation_collection_role(settings)

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_representation_collection_name

    @property
    def collection_fingerprint(self) -> str:
        return representation_collection_fingerprint(self.settings)

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = (
                self.settings.qdrant_api_key.get_secret_value()
                if self.settings.qdrant_api_key is not None
                else None
            )
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key or None,
                timeout=self.settings.qdrant_timeout_seconds,
            )
        return self._client

    def close(self) -> None:
        """Close an owned Qdrant client."""

        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()

    def _reject_known_chunk_collection(self) -> None:
        """Reject an existing collection whose payload identifies A2/A3 chunks.

        The literal frozen name is protected independently by
        ``validate_representation_collection_role``.  This bounded sentinel
        check also protects a renamed existing chunk collection when its
        payload and deterministic point identity still expose the A2.3 role.
        """

        client = self._get_client()
        try:
            if not client.collection_exists(self.collection_name):
                return
            page, _ = client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:
            raise RepresentationIndexError(
                "the target Qdrant collection role could not be verified"
            ) from error
        if not page or not isinstance(page[0].payload, dict):
            return
        try:
            chunk_payload = ChunkVectorPayload.model_validate(page[0].payload)
            is_chunk_point = self._point_id(page[0].id) == point_id_for_chunk(
                chunk_payload.chunk_id
            )
        except (ValidationError, RepresentationIndexError, ValueError):
            return
        if (
            chunk_payload.collection_schema_version == QDRANT_COLLECTION_SCHEMA_VERSION
            and is_chunk_point
        ):
            raise RepresentationCollectionConfigurationError(
                "the A4.2 target collection is identified as the protected A2/A3 chunk index"
            )

    def _validate_collection(self, info: Any) -> None:
        contract = _active_representation_index_contract(self.settings)
        size, distance = _vector_config(info)
        if size != contract.vector_dimension or distance != contract.distance.casefold():
            raise RepresentationCollectionConfigurationError(
                "Qdrant representation collection has an incompatible vector schema; "
                f"expected {contract.vector_dimension} dimensions with cosine similarity"
            )

    def _validate_existing_collection_contract(self) -> None:
        """Validate one existing point before treating a collection as A4.2 state.

        Qdrant has no portable user-defined collection-role metadata.  The
        collection name is therefore protected by Settings, while an existing
        point provides a bounded payload-contract sentinel.  Document-scoped
        operations still validate every point they are about to mutate.
        """

        page, _ = self._get_client().scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if page:
            self._payload_from_record(page[0])

    def ensure_collection(self) -> RepresentationIndexReadiness:
        """Create the independent collection if absent and validate its vector schema."""

        validate_representation_collection_role(self.settings)
        self._reject_known_chunk_collection()
        client = self._get_client()
        try:
            if not client.collection_exists(self.collection_name):
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=QDRANT_VECTOR_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                    on_disk_payload=True,
                )
            info = client.get_collection(self.collection_name)
            self._validate_collection(info)
            self._validate_existing_collection_contract()
        except RepresentationCollectionConfigurationError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation collection could not be created or validated"
            ) from error
        return RepresentationIndexReadiness(
            status="ready",
            collection_name=self.collection_name,
            collection_exists=True,
            vector_dimension=QDRANT_VECTOR_DIMENSION,
        )

    def readiness(self) -> RepresentationIndexReadiness:
        """Check reachability without creating the collection."""

        validate_representation_collection_role(self.settings)
        client = self._get_client()
        try:
            self._reject_known_chunk_collection()
            if not client.collection_exists(self.collection_name):
                return RepresentationIndexReadiness(
                    status="available",
                    collection_name=self.collection_name,
                    collection_exists=False,
                )
            info = client.get_collection(self.collection_name)
            self._validate_collection(info)
            self._validate_existing_collection_contract()
            return RepresentationIndexReadiness(
                status="ready",
                collection_name=self.collection_name,
                collection_exists=True,
                vector_dimension=QDRANT_VECTOR_DIMENSION,
            )
        except RepresentationCollectionConfigurationError as error:
            return RepresentationIndexReadiness(
                status="unavailable",
                collection_name=self.collection_name,
                collection_exists=True,
                error=str(error),
            )
        except RepresentationIndexError as error:
            return RepresentationIndexReadiness(
                status="unavailable",
                collection_name=self.collection_name,
                collection_exists=True,
                error=str(error),
            )
        except Exception:
            return RepresentationIndexReadiness(
                status="unavailable",
                collection_name=self.collection_name,
                collection_exists=False,
                error="Qdrant representation collection is not reachable",
            )

    def _filter(
        self,
        *,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
        modality: RepresentationModalityFilter | None = None,
        searchable_only: bool = False,
        exclude_quarantined: bool = False,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if knowledge_base_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="knowledge_base_id",
                    match=models.MatchValue(value=str(knowledge_base_id)),
                )
            )
        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            )
        if modality is not None and modality != "all":
            conditions.append(
                models.FieldCondition(
                    key="modality",
                    match=models.MatchValue(value=modality),
                )
            )
        if searchable_only:
            conditions.append(
                models.FieldCondition(
                    key="searchable",
                    match=models.MatchValue(value=True),
                )
            )
        excluded: list[models.FieldCondition] = []
        if exclude_quarantined:
            excluded.append(
                models.FieldCondition(
                    key="quarantined",
                    match=models.MatchValue(value=True),
                )
            )
        if not conditions and not excluded:
            return None
        return models.Filter(
            must=conditions or None,
            must_not=excluded or None,
        )

    @staticmethod
    def _point_id(value: Any) -> UUID:
        try:
            return value if isinstance(value, UUID) else UUID(str(value))
        except (TypeError, ValueError) as error:
            raise RepresentationIndexError(
                "Qdrant returned a non-UUID representation point"
            ) from error

    def _scroll_records(
        self,
        *,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
        with_vectors: bool = False,
        searchable_only: bool = False,
        exclude_quarantined: bool = False,
    ) -> list[Any]:
        records: list[Any] = []
        offset: Any | None = None
        while True:
            page, offset = self._get_client().scroll(
                collection_name=self.collection_name,
                scroll_filter=self._filter(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    searchable_only=searchable_only,
                    exclude_quarantined=exclude_quarantined,
                ),
                limit=REPRESENTATION_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            records.extend(page)
            if offset is None:
                return records

    def _payload_from_record(self, record: Any) -> IndexedRepresentationPayload:
        if not isinstance(record.payload, dict):
            raise RepresentationIndexError("Qdrant representation point is missing its payload")
        try:
            payload = IndexedRepresentationPayload.model_validate(record.payload)
        except ValidationError as error:
            raise RepresentationIndexError("Qdrant representation payload is invalid") from error
        self._validate_indexed_payload_contract(payload)
        if self._point_id(record.id) != point_id_for_representation(payload.representation_id):
            raise RepresentationIndexError("Qdrant representation point ID is inconsistent")
        return payload

    def _validate_indexed_payload_contract(
        self,
        payload: IndexedRepresentationPayload,
    ) -> None:
        """Validate payload metadata against the active collection contract."""

        contract = _active_representation_index_contract(self.settings)
        expected = {
            "schema_version": contract.index_payload_schema_version,
            "collection_schema_version": contract.collection_schema_version,
            "evidence_schema_version": contract.evidence_schema_version,
            "representation_schema_version": contract.representation_schema_version,
            "collection_fingerprint": contract.fingerprint,
            "embedding_model": contract.embedding_model,
            "embedding_model_revision": contract.embedding_model_revision,
            "embedding_config_fingerprint": contract.embedding_config_fingerprint,
        }
        for field_name, expected_value in expected.items():
            if getattr(payload, field_name) != expected_value:
                raise RepresentationCollectionConfigurationError(
                    f"representation point has an incompatible active index contract: {field_name}"
                )

    def _validated_document_records(self, document_id: UUID, knowledge_base_id: UUID) -> list[Any]:
        """Read a document scope and validate every record before mutation."""

        records = self._scroll_records(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            with_vectors=True,
        )
        for record in records:
            payload = self._payload_from_record(record)
            if payload.document_id != document_id or payload.knowledge_base_id != knowledge_base_id:
                raise RepresentationIndexError(
                    "Qdrant representation record escaped its requested document scope"
                )
        return records

    def document_point_count(self, document_id: UUID, *, knowledge_base_id: UUID) -> int:
        """Return the validated point count for one explicit document scope."""

        validate_representation_collection_role(self.settings)
        readiness = self.readiness()
        if readiness.status == "available":
            return 0
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            return len(self._validated_document_records(document_id, knowledge_base_id))
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation document scope could not be inspected"
            ) from error

    def list_payloads(
        self,
        *,
        knowledge_base_id: UUID | None = None,
    ) -> tuple[IndexedRepresentationPayload, ...]:
        """Read validated payloads for an audit without loading vectors."""

        validate_representation_collection_role(self.settings)
        readiness = self.readiness()
        if readiness.status == "available":
            return ()
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            return tuple(
                self._payload_from_record(record)
                for record in self._scroll_records(knowledge_base_id=knowledge_base_id)
            )
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation payloads could not be read"
            ) from error

    def _delete_ids(self, point_ids: Iterable[Any]) -> None:
        ids = list(point_ids)
        if not ids:
            return
        validate_representation_collection_role(self.settings)
        self._reject_known_chunk_collection()
        self._get_client().delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[str(point_id) for point_id in ids]),
            wait=True,
        )

    def _upsert(self, points: Sequence[models.PointStruct]) -> None:
        validate_representation_collection_role(self.settings)
        self._reject_known_chunk_collection()
        batch_size = max(1, self.settings.qdrant_upsert_batch_size)
        for start in range(0, len(points), batch_size):
            self._get_client().upsert(
                collection_name=self.collection_name,
                points=list(points[start : start + batch_size]),
                wait=True,
            )

    def _set_quarantined(self, point_ids: Sequence[UUID], value: bool) -> None:
        if not point_ids:
            return
        validate_representation_collection_role(self.settings)
        self._reject_known_chunk_collection()
        batch_size = max(1, self.settings.qdrant_upsert_batch_size)
        for start in range(0, len(point_ids), batch_size):
            self._get_client().set_payload(
                collection_name=self.collection_name,
                payload={"quarantined": value},
                points=models.PointIdsList(
                    points=[str(point_id) for point_id in point_ids[start : start + batch_size]]
                ),
                wait=True,
            )

    def quarantine_document(
        self,
        document_id: UUID,
        *,
        knowledge_base_id: UUID,
    ) -> RepresentationVisibilitySnapshot:
        """Hide a document's points while an authoritative delete is pending."""

        validate_representation_collection_role(self.settings)
        readiness = self.readiness()
        if readiness.status == "available":
            return RepresentationVisibilitySnapshot(document_id, knowledge_base_id, ())
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            records = self._validated_document_records(document_id, knowledge_base_id)
            states = tuple(
                (
                    self._point_id(record.id),
                    self._payload_from_record(record).quarantined,
                )
                for record in records
            )
            active_ids = [point_id for point_id, quarantined in states if not quarantined]
            try:
                self._set_quarantined(active_ids, True)
            except Exception as error:
                try:
                    self._restore_visibility_states(states)
                except Exception as restore_error:
                    raise RepresentationIndexError(
                        "representation quarantine failed and visibility rollback also failed"
                    ) from restore_error
                raise error
            return RepresentationVisibilitySnapshot(document_id, knowledge_base_id, states)
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation document could not be quarantined"
            ) from error

    def _restore_visibility_states(self, states: Sequence[tuple[UUID, bool]]) -> None:
        for quarantined in (False, True):
            point_ids = [point_id for point_id, state in states if state == quarantined]
            self._set_quarantined(point_ids, quarantined)

    def restore_document_visibility(self, snapshot: RepresentationVisibilitySnapshot) -> None:
        """Restore only the captured visibility state for one KB/document scope."""

        validate_representation_collection_role(self.settings)
        if not snapshot.point_states:
            return
        readiness = self.readiness()
        if readiness.status == "available":
            return
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            current_records = self._validated_document_records(
                snapshot.document_id,
                snapshot.knowledge_base_id,
            )
            current_ids = {self._point_id(record.id) for record in current_records}
            snapshot_ids = {point_id for point_id, _state in snapshot.point_states}
            missing_ids = snapshot_ids - current_ids
            if missing_ids:
                raise RepresentationIndexError(
                    "Qdrant representation visibility snapshot is missing points"
                )
            self._restore_visibility_states(
                [
                    (point_id, state)
                    for point_id, state in snapshot.point_states
                    if point_id in current_ids
                ]
            )
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation visibility could not be restored"
            ) from error

    def _record_to_point(self, record: Any) -> models.PointStruct:
        if not isinstance(record.vector, list):
            raise RepresentationIndexError("Qdrant representation snapshot is missing a vector")
        vector = _validate_vector(record.vector)
        payload = self._payload_from_record(record)
        return models.PointStruct(
            id=str(record.id),
            vector=list(vector),
            payload=payload.model_dump(mode="json"),
        )

    def _record_to_vector_point(self, record: Any) -> RepresentationVectorPoint:
        if not isinstance(record.vector, list):
            raise RepresentationIndexError("Qdrant representation snapshot is missing a vector")
        return RepresentationVectorPoint(
            point_id=self._point_id(record.id),
            vector=_validate_vector(record.vector),
            payload=self._payload_from_record(record),
        )

    def snapshot_document(
        self,
        document_id: UUID,
        *,
        knowledge_base_id: UUID,
    ) -> tuple[RepresentationVectorPoint, ...]:
        """Capture one active document generation for parser compensation."""

        validate_representation_collection_role(self.settings)
        readiness = self.readiness()
        if readiness.status == "available":
            return ()
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            records = self._validated_document_records(document_id, knowledge_base_id)
            points = tuple(self._record_to_vector_point(record) for record in records)
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation document snapshot could not be captured"
            ) from error
        if any(not point.payload.searchable or point.payload.quarantined for point in points):
            raise RepresentationIndexError(
                "a quarantined or non-searchable representation generation cannot be reparsed"
            )
        return points

    def _rollback(self, new_ids: Sequence[UUID], snapshot: Sequence[Any]) -> None:
        try:
            snapshot_ids = {str(record.id) for record in snapshot}
            self._delete_ids(
                [point_id for point_id in new_ids if str(point_id) not in snapshot_ids]
            )
            self._upsert([self._record_to_point(record) for record in snapshot])
        except Exception as error:
            raise RepresentationIndexError(
                "representation replacement failed and compensating rollback also failed"
            ) from error

    def replace_document(
        self,
        document_id: UUID,
        knowledge_base_id: UUID,
        points: Sequence[RepresentationVectorPoint],
    ) -> RepresentationIndexResult:
        """Replace one KB/document representation set with compensation on failure."""

        validate_representation_collection_role(self.settings)
        if any(
            point.payload.document_id != document_id
            or point.payload.knowledge_base_id != knowledge_base_id
            for point in points
        ):
            raise ValueError("all representation points must belong to the requested KB/document")
        point_ids = [point.point_id for point in points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("representation point IDs must be unique")
        for point in points:
            if not point.payload.searchable or point.payload.quarantined:
                raise ValueError("only active searchable representations may be indexed")
            if point.point_id != point_id_for_representation(point.payload.representation_id):
                raise ValueError("representation point ID does not match its representation")
            self._validate_indexed_payload_contract(point.payload)
            _validate_vector(point.vector)

        self._reject_known_chunk_collection()
        self.ensure_collection()
        existing_records: list[Any] = []
        upsert_started = False
        stale_ids: list[Any] = []
        try:
            existing_records = self._validated_document_records(document_id, knowledge_base_id)
            qdrant_points = [
                models.PointStruct(
                    id=str(point.point_id),
                    vector=list(_validate_vector(point.vector)),
                    payload=point.payload.model_dump(mode="json"),
                )
                for point in points
            ]
            upsert_started = True
            self._upsert(qdrant_points)
            new_ids = {str(point.point_id) for point in points}
            stale_ids = [record.id for record in existing_records if str(record.id) not in new_ids]
            self._delete_ids(stale_ids)
        except Exception as error:
            if upsert_started:
                self._rollback(point_ids, existing_records)
            raise RepresentationIndexError(
                "Qdrant representation replacement failed; the previous set was restored"
            ) from error
        return RepresentationIndexResult(
            document_id=document_id,
            indexed_count=len(points),
            removed_stale_count=len(stale_ids),
        )

    def delete_document(self, document_id: UUID, *, knowledge_base_id: UUID) -> int:
        """Delete all representations for one explicit KB/document scope."""

        validate_representation_collection_role(self.settings)
        readiness = self.readiness()
        if readiness.status == "available":
            return 0
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            records = self._validated_document_records(document_id, knowledge_base_id)
            self._delete_ids([record.id for record in records])
            return len(records)
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError(
                "Qdrant representation document data could not be deleted"
            ) from error

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        knowledge_base_id: UUID,
        document_id: UUID | None = None,
        modality: RepresentationModalityFilter | None = None,
    ) -> list[RetrievedRepresentation]:
        """Search the independent collection with mandatory KB isolation."""

        validate_representation_collection_role(self.settings)
        if not 1 <= limit <= REPRESENTATION_SEARCH_LIMIT_MAX:
            raise ValueError(f"limit must be between 1 and {REPRESENTATION_SEARCH_LIMIT_MAX}")
        vector = _validate_vector(query_vector)
        readiness = self.readiness()
        if readiness.status == "available":
            return []
        if readiness.status == "unavailable":
            raise RepresentationIndexError(readiness.error or "Qdrant is unavailable")
        try:
            response = self._get_client().query_points(
                collection_name=self.collection_name,
                query=list(vector),
                query_filter=self._filter(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    modality=modality,
                    searchable_only=True,
                    exclude_quarantined=True,
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            hit_values: list[tuple[UUID, float, IndexedRepresentationPayload]] = []
            for point in response.points:
                payload = self._payload_from_record(point)
                if payload.knowledge_base_id != knowledge_base_id:
                    raise RepresentationIndexError("Qdrant returned a point from another KB")
                if not payload.searchable or payload.quarantined:
                    continue
                hit_values.append((self._point_id(point.id), float(point.score), payload))
            hit_values.sort(key=lambda item: (-item[1], str(item[0])))
            return [
                RetrievedRepresentation(
                    point_id=point_id,
                    score=score,
                    rank=rank,
                    payload=payload,
                )
                for rank, (point_id, score, payload) in enumerate(hit_values, 1)
            ]
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError("Qdrant representation search failed") from error


def _block_text(block: CanonicalBlock) -> str | None:
    if isinstance(block, (TextBlock, TitleBlock)):
        return block.text
    return None


def _bounded(value: str, limit: int) -> str:
    return value.strip()[:limit]


def _materialization_context(
    document: CanonicalDocument,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build bounded structural context without generating semantic content."""

    contexts: dict[str, str] = {}
    section_paths: dict[str, list[str]] = {}
    for page in document.pages:
        titles: list[str] = []
        text_blocks = [_block_text(block) for block in page.blocks]
        for index, block in enumerate(page.blocks):
            if isinstance(block, TitleBlock):
                titles.append(_bounded(block.text, MAX_SECTION_ITEM_CHARS))
                titles = titles[-MAX_SECTION_ITEMS:]
            section_paths[block.block_id] = list(titles)
            if isinstance(block, TitleBlock):
                continue

            neighbors: list[str] = []
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(text_blocks):
                    neighbor = text_blocks[neighbor_index]
                    if neighbor and neighbor.strip():
                        neighbors.append(_bounded(neighbor, MAX_NEIGHBOR_CHARS))
            parts: list[str] = []
            if titles:
                parts.append("章节: " + " / ".join(titles))
            if neighbors:
                parts.append("邻近文本: " + "; ".join(neighbors))
            context = "; ".join(parts)
            if context:
                contexts[block.block_id] = _bounded(context, MAX_CONTEXT_CHARS)
    return contexts, section_paths


def build_indexable_evidence(
    document: CanonicalDocument,
    knowledge_base_id: UUID,
    *,
    data_dir: Path | None = None,
) -> MultimodalEvidenceDocument:
    """Derive current A4.1 evidence plus structural text context.

    If ``data_dir`` is supplied, the derived A4.1 artifact is persisted using
    its existing safe atomic lifecycle.  No PDF, canonical source, or raw
    provider response is copied by this function.
    """

    contexts, section_paths = _materialization_context(document)
    artifact = build_evidence_document(
        document,
        knowledge_base_id,
        context_by_block_id=contexts,
        section_path_by_block_id=section_paths,
    )
    try:
        validate_evidence_document(document, artifact, knowledge_base_id)
    except EvidenceValidationError as error:
        raise RepresentationIndexError("derived evidence failed source validation") from error
    if data_dir is not None:
        try:
            write_evidence_artifact(data_dir, artifact)
        except Exception as error:
            raise RepresentationIndexError(
                "derived evidence artifact could not be persisted"
            ) from error
    return artifact


def load_canonical_document(path: Path) -> CanonicalDocument:
    """Load one parser-independent canonical artifact with a bounded error."""

    try:
        return CanonicalDocument.model_validate_json(path.read_bytes())
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValidationError) as error:
        raise RepresentationIndexError(f"invalid or missing canonical artifact: {path}") from error


def build_representation_points(
    artifact: MultimodalEvidenceDocument,
    *,
    settings: Settings,
    embedder: RepresentationEmbeddingEncoder,
) -> list[RepresentationVectorPoint]:
    """Encode searchable A4.1 representations into deterministic points."""

    contract = _active_representation_index_contract(settings)
    expected_config_fingerprint = contract.embedding_config_fingerprint
    if (
        embedder.model_id != contract.embedding_model
        or embedder.model_revision != contract.embedding_model_revision
        or embedder.config_fingerprint != expected_config_fingerprint
    ):
        raise RepresentationIndexError(
            "the representation index requires the configured Qwen embedding profile"
        )
    try:
        payloads = representation_index_payloads(artifact, searchable_only=True)
        vectors = embedder.encode_documents([payload.text or "" for payload in payloads])
    except Exception as error:
        raise RepresentationIndexError("representations could not be embedded") from error
    if len(vectors) != len(payloads):
        raise RepresentationIndexError("embedding result count does not match representation count")

    collection_fingerprint = contract.fingerprint
    points: list[RepresentationVectorPoint] = []
    for payload, vector in zip(payloads, vectors, strict=True):
        indexed_payload = IndexedRepresentationPayload(
            **payload.model_dump(mode="python"),
            collection_fingerprint=collection_fingerprint,
            embedding_model=contract.embedding_model,
            embedding_model_revision=contract.embedding_model_revision,
            embedding_config_fingerprint=contract.embedding_config_fingerprint,
        )
        points.append(
            RepresentationVectorPoint(
                point_id=point_id_for_representation(indexed_payload.representation_id),
                vector=_validate_vector(vector),
                payload=indexed_payload,
            )
        )
    return points


def index_canonical_document(
    canonical_path: Path,
    *,
    knowledge_base_id: UUID,
    settings: Settings,
    store: QdrantRepresentationStore,
    embedder: RepresentationEmbeddingEncoder,
    persist_evidence: bool = True,
) -> RepresentationIndexResult:
    """Build and safely replace one document in the A4.2 collection.

    The evidence artifact is written before Qdrant replacement so retrieval can
    resolve the new points authoritatively.  If replacement fails, restore the
    previous artifact after the store's own compensating rollback.
    """

    document = load_canonical_document(canonical_path)
    artifact = build_indexable_evidence(document, knowledge_base_id)
    points = build_representation_points(artifact, settings=settings, embedder=embedder)
    previous_artifact: MultimodalEvidenceDocument | None = None
    if persist_evidence:
        try:
            previous_path = evidence_artifact_path(settings.data_dir, document.document_id)
        except EvidenceArtifactError as error:
            raise RepresentationIndexError(
                "existing evidence artifact path is invalid; replacement stopped"
            ) from error
        if previous_path.exists() or previous_path.is_symlink():
            try:
                previous_artifact = load_evidence_artifact(
                    settings.data_dir,
                    document.document_id,
                )
            except EvidenceArtifactError as error:
                raise RepresentationIndexError(
                    "existing evidence artifact is invalid; replacement stopped"
                ) from error
        elif store.document_point_count(
            document.document_id,
            knowledge_base_id=knowledge_base_id,
        ):
            raise RepresentationIndexError(
                "existing representation points have no matching evidence artifact; "
                "replacement stopped"
            )

    evidence_written = False
    try:
        if persist_evidence:
            write_evidence_artifact(settings.data_dir, artifact)
            evidence_written = True
        return store.replace_document(document.document_id, knowledge_base_id, points)
    except Exception:
        if evidence_written:
            try:
                if previous_artifact is None:
                    remove_evidence_artifact(settings.data_dir, document.document_id)
                else:
                    write_evidence_artifact(settings.data_dir, previous_artifact)
            except Exception as restore_error:
                raise RepresentationIndexError(
                    "representation replacement failed and evidence rollback also failed"
                ) from restore_error
        raise


def index_canonical_corpus(
    canonical_root: Path,
    *,
    knowledge_base_id: UUID,
    settings: Settings,
    limit: int | None = None,
    store: QdrantRepresentationStore,
    embedder: RepresentationEmbeddingEncoder,
) -> list[RepresentationIndexResult]:
    """Index a bounded canonical corpus without rerunning MinerU."""

    if not canonical_root.is_dir():
        raise RepresentationIndexError(f"canonical root does not exist: {canonical_root}")
    paths = sorted(canonical_root.glob("*.json"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least one")
        paths = paths[:limit]
    if not paths:
        raise RepresentationIndexError("canonical root contains no JSON artifacts")
    return [
        index_canonical_document(
            path,
            knowledge_base_id=knowledge_base_id,
            settings=settings,
            store=store,
            embedder=embedder,
        )
        for path in paths
    ]


def _all_payloads(artifact: MultimodalEvidenceDocument) -> list[RepresentationIndexPayload]:
    return representation_index_payloads(artifact, searchable_only=False)


def audit_representation_index(
    canonical_root: Path,
    *,
    knowledge_base_id: UUID,
    settings: Settings,
    store: QdrantRepresentationStore,
) -> RepresentationCoverageAudit:
    """Compare current canonical-derived representations with indexed payloads."""

    contract = _active_representation_index_contract(settings)
    if not canonical_root.is_dir():
        raise RepresentationIndexError(f"canonical root does not exist: {canonical_root}")
    paths = sorted(canonical_root.glob("*.json"))
    expected: dict[str, IndexedRepresentationPayload] = {}
    source_evidence_count = 0
    representation_count = 0
    searchable_count = 0
    lineage_failures = 0
    lineage_failures_by_modality: dict[str, int] = defaultdict(int)
    bucket_source_evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    bucket_representation_counts: dict[tuple[str, str], int] = defaultdict(int)
    bucket_searchable_counts: dict[tuple[str, str], int] = defaultdict(int)
    bucket_counts: dict[tuple[str, str], list[int]] = {}

    for path in paths:
        document = load_canonical_document(path)
        try:
            artifact = build_indexable_evidence(document, knowledge_base_id)
            source_evidence_count += len(artifact.evidence)
            for item in artifact.evidence:
                for representation in item.representations:
                    key = (item.modality, representation.representation_type)
                    bucket_source_evidence[key].add(item.evidence_id)
                    bucket_representation_counts[key] += 1
                    if representation.searchable:
                        bucket_searchable_counts[key] += 1
            all_payloads = _all_payloads(artifact)
            representation_count += len(all_payloads)
            for payload in representation_index_payloads(artifact, searchable_only=True):
                indexed = IndexedRepresentationPayload(
                    **payload.model_dump(mode="python"),
                    collection_fingerprint=contract.fingerprint,
                    embedding_model=contract.embedding_model,
                    embedding_model_revision=contract.embedding_model_revision,
                    embedding_config_fingerprint=contract.embedding_config_fingerprint,
                )
                expected[indexed.representation_id] = indexed
                searchable_count += 1
        except (RepresentationIndexError, EvidenceValidationError, ValidationError) as error:
            lineage_failures += 1
            modality = "unknown"
            message = str(error)
            if "modality" in message:
                modality = message[:80]
            lineage_failures_by_modality[modality] += 1

    actual = store.list_payloads(knowledge_base_id=knowledge_base_id)
    actual_by_id: dict[str, list[IndexedRepresentationPayload]] = defaultdict(list)
    for payload in actual:
        actual_by_id[payload.representation_id].append(payload)

    current_representation_ids: set[str] = set()
    missing_count = 0
    stale_count = 0
    indexed_count = 0
    for representation_id, payload in expected.items():
        matches = actual_by_id.get(representation_id, [])
        if not matches:
            missing_count += 1
            continue
        valid_match = any(_payload_matches_authoritative(item, payload) for item in matches)
        if valid_match:
            indexed_count += 1
            current_representation_ids.add(representation_id)
        else:
            stale_count += 1
    for representation_id, matches in actual_by_id.items():
        if representation_id not in expected:
            stale_count += len(matches)

    for key, evidence_ids in bucket_source_evidence.items():
        modality, representation_type = key
        searchable_bucket = [
            payload
            for payload in expected.values()
            if payload.modality == modality and payload.representation_type == representation_type
        ]
        actual_bucket = [
            payload
            for payload in actual
            if payload.modality == modality and payload.representation_type == representation_type
        ]
        actual_ids = {payload.representation_id for payload in actual_bucket}
        current_ids = {
            representation_id
            for representation_id in actual_ids
            if representation_id in current_representation_ids
        }
        bucket_searchable_counts[key] = len(searchable_bucket)
        bucket_representation_counts.setdefault(key, bucket_representation_counts[key])
        bucket_indexed_count = len(current_ids)
        bucket_missing_count = max(0, len(searchable_bucket) - bucket_indexed_count)
        bucket_stale_count = sum(
            1
            for payload in actual_bucket
            if payload.representation_id not in current_representation_ids
        )
        bucket_source_evidence[key] = set(evidence_ids)
        # Store derived values under a private local mapping to keep the output
        # assembly below readable while retaining non-searchable source reps.
        bucket_counts[key] = [
            len(evidence_ids),
            bucket_representation_counts[key],
            bucket_searchable_counts[key],
            bucket_indexed_count,
            bucket_missing_count,
            bucket_stale_count,
        ]

    buckets = []
    for (
        modality,
        representation_type,
    ), (
        evidence_count,
        representation_total,
        searchable_total,
        indexed,
        bucket_missing,
        bucket_stale,
    ) in sorted(bucket_counts.items()):
        buckets.append(
            RepresentationCoverageBucket(
                modality=modality,  # type: ignore[arg-type]
                representation_type=representation_type,
                source_evidence_count=evidence_count,
                representation_count=representation_total,
                searchable_representation_count=searchable_total,
                indexed_count=indexed,
                missing_count=bucket_missing,
                stale_count=bucket_stale,
            )
        )
    return RepresentationCoverageAudit(
        collection_name=store.collection_name,
        collection_fingerprint=representation_collection_fingerprint(settings),
        knowledge_base_id=knowledge_base_id,
        documents_scanned=len(paths),
        source_evidence_count=source_evidence_count,
        representation_count=representation_count,
        searchable_representation_count=searchable_count,
        indexed_count=indexed_count,
        missing_count=missing_count,
        stale_count=stale_count,
        lineage_failure_count=lineage_failures,
        lineage_failures_by_modality=dict(lineage_failures_by_modality),
        buckets=buckets,
    )


class MultimodalRepresentationRetrievalService:
    """Embed a text query, search Qdrant, and deduplicate at Evidence level."""

    def __init__(
        self,
        store: QdrantRepresentationStore,
        embedder: RepresentationEmbeddingEncoder,
    ) -> None:
        self.store = store
        self.embedder = embedder

    def _authoritative_representation(
        self,
        payload: IndexedRepresentationPayload,
    ) -> tuple[MultimodalEvidence, EvidenceRepresentation]:
        """Resolve one hit from the current A4.1 artifact instead of trusting Qdrant."""

        try:
            artifact = load_evidence_artifact(self.store.settings.data_dir, payload.document_id)
        except EvidenceArtifactError as error:
            raise RepresentationIndexError(
                "authoritative evidence is unavailable for a representation hit"
            ) from error

        if (
            artifact.knowledge_base_id != payload.knowledge_base_id
            or artifact.document_id != payload.document_id
            or artifact.canonical_document_fingerprint != payload.canonical_document_fingerprint
        ):
            raise RepresentationIndexError(
                "representation hit does not match the current authoritative evidence"
            )

        current_canonical_path = (
            Path(self.store.settings.data_dir)
            / "parsing"
            / str(payload.document_id)
            / "canonical.json"
        )
        if current_canonical_path.is_file():
            document = load_canonical_document(current_canonical_path)
            try:
                validate_evidence_document(document, artifact, payload.knowledge_base_id)
            except EvidenceValidationError as error:
                raise RepresentationIndexError(
                    "authoritative evidence does not match the current canonical document"
                ) from error

        item_by_id = {item.evidence_id: item for item in artifact.evidence}
        item = item_by_id.get(payload.evidence_id)
        if item is None:
            raise RepresentationIndexError("representation hit references unknown evidence")
        representation_by_id = {
            representation.representation_id: representation
            for representation in item.representations
        }
        representation = representation_by_id.get(payload.representation_id)
        if representation is None:
            raise RepresentationIndexError(
                "representation hit references unknown authoritative representation"
            )

        authoritative_payloads = representation_index_payloads(
            artifact,
            searchable_only=False,
        )
        authoritative_payload = next(
            (
                candidate
                for candidate in authoritative_payloads
                if candidate.representation_id == payload.representation_id
            ),
            None,
        )
        if authoritative_payload is None or not _payload_matches_authoritative(
            payload,
            authoritative_payload,
        ):
            raise RepresentationIndexError(
                "representation hit provenance differs from authoritative evidence"
            )
        if not payload.searchable or payload.quarantined:
            raise RepresentationIndexError("a non-searchable representation was returned")
        return item, representation

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
        modality: RepresentationModalityFilter | None = None,
    ) -> RepresentationRetrievalResult:
        """Return up to ``top_k`` Evidence results with raw hits preserved."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if not 1 <= top_k <= REPRESENTATION_SEARCH_LIMIT_MAX:
            raise ValueError(f"top_k must be between 1 and {REPRESENTATION_SEARCH_LIMIT_MAX}")
        try:
            query_vector = self.embedder.encode_query(query)
            candidate_limit = min(
                REPRESENTATION_SEARCH_LIMIT_MAX,
                max(top_k, top_k * REPRESENTATION_CANDIDATE_MULTIPLIER),
            )
            raw_hits = self.store.search(
                query_vector,
                limit=candidate_limit,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                modality=modality,
            )
        except RepresentationIndexError:
            raise
        except Exception as error:
            raise RepresentationIndexError("multimodal representation retrieval failed") from error

        groups: dict[
            str,
            list[tuple[RetrievedRepresentation, MultimodalEvidence, EvidenceRepresentation]],
        ] = defaultdict(list)
        for hit in raw_hits:
            item, representation = self._authoritative_representation(hit.payload)
            groups[item.evidence_id].append((hit, item, representation))

        candidates: list[
            tuple[
                float,
                int,
                str,
                list[tuple[RetrievedRepresentation, MultimodalEvidence, EvidenceRepresentation]],
            ]
        ] = []
        for evidence_id, hits in groups.items():
            hits.sort(key=lambda hit: (-hit[0].score, hit[0].rank, str(hit[0].point_id)))
            best = hits[0][0]
            candidates.append((best.score, best.rank, evidence_id, hits))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        items: list[RetrievedEvidence] = []
        for result_rank, (score, _best_rank, _evidence_id, hits) in enumerate(
            candidates[:top_k], 1
        ):
            best, evidence, representation = hits[0]
            lineage = evidence.lineage
            items.append(
                RetrievedEvidence(
                    rank=result_rank,
                    score=score,
                    evidence_id=evidence.evidence_id,
                    knowledge_base_id=lineage.knowledge_base_id,
                    document_id=lineage.document_id,
                    page_start=lineage.page_start,
                    page_end=lineage.page_end,
                    source_block_ids=lineage.source_block_ids,
                    section_path=lineage.section_path,
                    asset_refs=lineage.asset_refs,
                    modality=evidence.modality,
                    representation_id=representation.representation_id,
                    representation_type=representation.representation_type,
                    text=representation.content,
                    reference=representation.reference,
                    representations=[
                        RepresentationEvidenceHit(
                            representation_id=hit.payload.representation_id,
                            representation_type=hit_representation.representation_type,
                            score=hit.score,
                            rank=hit.rank,
                            text=hit_representation.content,
                            reference=hit_representation.reference,
                        )
                        for hit, _hit_evidence, hit_representation in hits
                    ],
                )
            )
        return RepresentationRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            items=items,
            raw_hits=raw_hits,
        )


__all__ = [
    "INDEX_PAYLOAD_SCHEMA_VERSION",
    "MAX_CONTEXT_CHARS",
    "REPRESENTATION_CANDIDATE_MULTIPLIER",
    "REPRESENTATION_COLLECTION_DEFAULT_NAME",
    "REPRESENTATION_COLLECTION_SCHEMA_VERSION",
    "REPRESENTATION_MATERIALIZATION_VERSION",
    "REPRESENTATION_POINT_NAMESPACE",
    "IndexedRepresentationPayload",
    "MultimodalRepresentationRetrievalService",
    "QdrantRepresentationStore",
    "RepresentationCollectionConfigurationError",
    "RepresentationCoverageAudit",
    "RepresentationEmbeddingEncoder",
    "RepresentationIndexError",
    "RepresentationIndexReadiness",
    "RepresentationIndexResult",
    "RepresentationRetrievalResult",
    "RepresentationVectorPoint",
    "RepresentationVisibilitySnapshot",
    "RetrievedEvidence",
    "RetrievedRepresentation",
    "audit_representation_index",
    "build_indexable_evidence",
    "build_representation_points",
    "index_canonical_corpus",
    "index_canonical_document",
    "load_canonical_document",
    "point_id_for_representation",
    "representation_collection_fingerprint",
    "validate_representation_collection_role",
]
