"""Dense retrieval service: query embedding followed by Qdrant top-k search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from .qdrant import QdrantVectorStore, RetrievedChunk


class RetrievalError(RuntimeError):
    """Raised when a retrieval request cannot be completed safely."""


class QueryEncoderProtocol(Protocol):
    """Small structural query-encoder contract for service tests."""

    model_id: str

    def encode_query(self, query: str) -> list[float]:
        """Encode a single search query."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Ranked dense-retrieval response with model and collection context."""

    query: str
    limit: int
    model_id: str
    collection_name: str
    items: tuple[RetrievedChunk, ...]


class DenseRetrievalService:
    """Compose one query encoder with the persistent Qdrant vector store."""

    def __init__(self, store: QdrantVectorStore, encoder: QueryEncoderProtocol) -> None:
        self.store = store
        self.encoder = encoder

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> RetrievalResult:
        """Return ranked chunks, optionally restricted by KB or document."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            query_vector = self.encoder.encode_query(query)
            items = self.store.search(
                query_vector,
                limit=limit,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
            )
        except Exception as error:
            if isinstance(error, ValueError):
                raise
            raise RetrievalError("dense retrieval failed") from error
        return RetrievalResult(
            query=query,
            limit=limit,
            model_id=self.encoder.model_id,
            collection_name=self.store.collection_name,
            items=tuple(items),
        )


__all__ = ["DenseRetrievalService", "QueryEncoderProtocol", "RetrievalError", "RetrievalResult"]
