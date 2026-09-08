"""Small, versioned Qdrant adapter for KnowledgeScope chunk vectors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from qdrant_client import QdrantClient, models

from knowledge_scope.shared.config import Settings

QDRANT_COLLECTION_SCHEMA_VERSION = "1.0"
QDRANT_VECTOR_DIMENSION = 1024
QDRANT_DEFAULT_COLLECTION_NAME = "knowledgescope_chunks_v1"
QWEN_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_MAX_SEQ_LENGTH = 512
QDRANT_POINT_NAMESPACE = UUID("3f4c8d2a-0a4f-5a5f-8cc6-e8d2fbf2ac35")
QDRANT_SCROLL_PAGE_SIZE = 256


class VectorStoreError(RuntimeError):
    """Raised when Qdrant cannot satisfy a vector-store operation safely."""


class CollectionConfigurationError(VectorStoreError):
    """Raised when an existing collection does not match the A2.3 contract."""


class ChunkVectorPayload(BaseModel):
    """Repository-controlled payload stored beside one indexed chunk vector."""

    model_config = ConfigDict(extra="forbid")

    collection_schema_version: Literal["1.0"]
    chunk_id: str = Field(min_length=1)
    document_id: UUID
    knowledge_base_id: UUID | None = None
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    content_types: list[str] = Field(min_length=1)
    asset_refs: list[str]
    text: str
    chunking_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str = Field(min_length=1)
    embedding_model_revision: str | None = None
    embedding_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_block_ids", "content_types", "asset_refs")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("payload lists must not contain blank values")
        return values

    @field_validator("page_end")
    @classmethod
    def validate_page_end(cls, value: int, info: Any) -> int:
        page_start = info.data.get("page_start")
        if page_start is not None and value < page_start:
            raise ValueError("page_end must not be less than page_start")
        return value


class RetrievedChunk(BaseModel):
    """A ranked chunk returned from dense retrieval."""

    model_config = ConfigDict(extra="forbid")

    point_id: UUID
    score: float
    payload: ChunkVectorPayload


class QdrantReadiness(BaseModel):
    """Non-sensitive Qdrant connectivity and collection readiness facts."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "available", "unavailable"]
    collection_name: str
    collection_exists: bool
    vector_dimension: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One validated vector and its repository-controlled payload."""

    point_id: UUID
    vector: tuple[float, ...]
    payload: ChunkVectorPayload


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Facts returned after a document replacement attempt."""

    document_id: UUID
    indexed_count: int
    removed_stale_count: int


def point_id_for_chunk(chunk_id: str) -> UUID:
    """Return a stable Qdrant UUID for a canonical chunk ID."""
    if not chunk_id.strip():
        raise ValueError("chunk_id must not be blank")
    return uuid5(QDRANT_POINT_NAMESPACE, f"{QDRANT_COLLECTION_SCHEMA_VERSION}:{chunk_id}")


def _as_point_id(value: Any) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as error:
        raise VectorStoreError("Qdrant returned a non-UUID point ID") from error


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


class QdrantVectorStore:
    """Synchronous Qdrant client wrapper used by CLI and request threadpools."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_collection_name

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = (
                self.settings.qdrant_api_key.get_secret_value()
                if self.settings.qdrant_api_key is not None
                else None
            )
            api_key = api_key or None
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key,
                timeout=self.settings.qdrant_timeout_seconds,
            )
        return self._client

    def close(self) -> None:
        """Close the owned HTTP client when the application shuts down."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()

    def _validate_collection(self, info: Any) -> None:
        size, distance = _vector_config(info)
        if size != QDRANT_VECTOR_DIMENSION or distance != str(models.Distance.COSINE).casefold():
            raise CollectionConfigurationError(
                "Qdrant collection has an incompatible vector schema; "
                f"expected {QDRANT_VECTOR_DIMENSION} dimensions with cosine similarity"
            )

    def ensure_collection(self) -> QdrantReadiness:
        """Create the versioned collection if absent and validate it if present."""
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
        except CollectionConfigurationError:
            raise
        except Exception as error:
            raise VectorStoreError("Qdrant collection could not be created or validated") from error
        return QdrantReadiness(
            status="ready",
            collection_name=self.collection_name,
            collection_exists=True,
            vector_dimension=QDRANT_VECTOR_DIMENSION,
        )

    def readiness(self) -> QdrantReadiness:
        """Check Qdrant reachability without creating a collection."""
        client = self._get_client()
        try:
            if not client.collection_exists(self.collection_name):
                return QdrantReadiness(
                    status="available",
                    collection_name=self.collection_name,
                    collection_exists=False,
                )
            info = client.get_collection(self.collection_name)
            self._validate_collection(info)
            return QdrantReadiness(
                status="ready",
                collection_name=self.collection_name,
                collection_exists=True,
                vector_dimension=QDRANT_VECTOR_DIMENSION,
            )
        except CollectionConfigurationError as error:
            return QdrantReadiness(
                status="unavailable",
                collection_name=self.collection_name,
                collection_exists=True,
                error=str(error),
            )
        except Exception:
            return QdrantReadiness(
                status="unavailable",
                collection_name=self.collection_name,
                collection_exists=False,
                error="Qdrant is not reachable",
            )

    def _scroll_ids(self, *, document_id: UUID) -> list[Any]:
        client = self._get_client()
        point_ids: list[Any] = []
        offset: Any | None = None
        document_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            ]
        )
        while True:
            records, offset = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=document_filter,
                limit=QDRANT_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(record.id for record in records)
            if offset is None:
                return point_ids

    def _retrieve_snapshot(self, point_ids: Sequence[Any]) -> list[Any]:
        if not point_ids:
            return []
        return self._get_client().retrieve(
            collection_name=self.collection_name,
            ids=[str(point_id) for point_id in point_ids],
            with_payload=True,
            with_vectors=True,
        )

    @staticmethod
    def _record_to_point(record: Any) -> models.PointStruct:
        if not isinstance(record.vector, list) or not all(
            isinstance(value, (float, int)) for value in record.vector
        ):
            raise VectorStoreError("Qdrant snapshot contains an unsupported vector format")
        if not isinstance(record.payload, dict):
            raise VectorStoreError("Qdrant snapshot is missing chunk payload")
        return models.PointStruct(
            id=str(record.id),
            vector=[float(value) for value in record.vector],
            payload=record.payload,
        )

    def _delete_point_ids(self, point_ids: Iterable[Any]) -> None:
        ids = list(point_ids)
        if not ids:
            return
        self._get_client().delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[str(point_id) for point_id in ids]),
            wait=True,
        )

    def _upsert_points(self, points: Sequence[models.PointStruct]) -> None:
        client = self._get_client()
        batch_size = max(1, self.settings.qdrant_upsert_batch_size)
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                points=list(points[start : start + batch_size]),
                wait=True,
            )

    def _rollback(
        self,
        new_point_ids: Sequence[Any],
        snapshot: Sequence[Any],
    ) -> None:
        try:
            self._delete_point_ids(new_point_ids)
            self._upsert_points([self._record_to_point(record) for record in snapshot])
        except Exception as error:
            raise VectorStoreError(
                "Qdrant indexing failed and rollback also failed; manual consistency repair is "
                "required"
            ) from error

    def replace_document(self, points: Sequence[VectorPoint]) -> IndexResult:
        """Replace one document's vectors with best-effort compensating rollback."""
        if not points:
            raise ValueError("at least one vector point is required")
        document_ids = {point.payload.document_id for point in points}
        if len(document_ids) != 1:
            raise ValueError("all vector points must belong to one document")
        if any(len(point.vector) != QDRANT_VECTOR_DIMENSION for point in points):
            raise ValueError(f"every vector must have {QDRANT_VECTOR_DIMENSION} dimensions")
        point_ids = [point.point_id for point in points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("vector point IDs must be unique")

        self.ensure_collection()
        document_id = next(iter(document_ids))
        upsert_started = False
        try:
            existing_ids = self._scroll_ids(document_id=document_id)
            snapshot = self._retrieve_snapshot(existing_ids)
            qdrant_points = [
                models.PointStruct(
                    id=str(point.point_id),
                    vector=list(point.vector),
                    payload=point.payload.model_dump(mode="json"),
                )
                for point in points
            ]
            upsert_started = True
            self._upsert_points(qdrant_points)
            new_id_set = {str(point.point_id) for point in points}
            stale_ids = [point_id for point_id in existing_ids if str(point_id) not in new_id_set]
            self._delete_point_ids(stale_ids)
        except Exception as error:
            if upsert_started:
                try:
                    self._rollback(point_ids, snapshot)
                except VectorStoreError:
                    raise
            raise VectorStoreError(
                "Qdrant document replacement failed; the previous set was restored"
            ) from error
        return IndexResult(
            document_id=document_id,
            indexed_count=len(points),
            removed_stale_count=len(stale_ids),
        )

    def delete_document(self, document_id: UUID) -> int:
        """Delete all vectors for a document; the operation is idempotent."""
        readiness = self.readiness()
        if readiness.status == "available":
            return 0
        if readiness.status == "unavailable":
            raise VectorStoreError(readiness.error or "Qdrant is unavailable")
        try:
            point_ids = self._scroll_ids(document_id=document_id)
            self._delete_point_ids(point_ids)
            return len(point_ids)
        except Exception as error:
            raise VectorStoreError("Qdrant document vectors could not be deleted") from error

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Run Qdrant dense top-k search with optional payload filters."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if len(query_vector) != QDRANT_VECTOR_DIMENSION:
            raise ValueError(f"query vector must have {QDRANT_VECTOR_DIMENSION} dimensions")
        readiness = self.readiness()
        if readiness.status == "available":
            return []
        if readiness.status == "unavailable":
            raise VectorStoreError(readiness.error or "Qdrant is unavailable")

        conditions = []
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
        query_filter = models.Filter(must=conditions) if conditions else None
        try:
            response = self._get_client().query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [
                RetrievedChunk(
                    point_id=_as_point_id(point.id),
                    score=float(point.score),
                    payload=ChunkVectorPayload.model_validate(point.payload or {}),
                )
                for point in response.points
            ]
        except Exception as error:
            raise VectorStoreError("Qdrant dense search failed") from error


__all__ = [
    "QDRANT_COLLECTION_SCHEMA_VERSION",
    "QDRANT_DEFAULT_COLLECTION_NAME",
    "QDRANT_VECTOR_DIMENSION",
    "QWEN_EMBEDDING_MAX_SEQ_LENGTH",
    "QWEN_EMBEDDING_MODEL_ID",
    "ChunkVectorPayload",
    "CollectionConfigurationError",
    "IndexResult",
    "QdrantReadiness",
    "QdrantVectorStore",
    "RetrievedChunk",
    "VectorPoint",
    "VectorStoreError",
    "point_id_for_chunk",
]
