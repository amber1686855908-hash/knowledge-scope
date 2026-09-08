"""Chunk-to-vector indexing workflows for the A2.3 Qdrant integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select

from knowledge_scope.chunking.models import Chunk, ChunkedDocument, ChunkingConfig
from knowledge_scope.chunking.service import chunk_document
from knowledge_scope.documents.models import Document
from knowledge_scope.parsing.models import CanonicalDocument
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    IndexResult,
    QdrantVectorStore,
    VectorPoint,
    point_id_for_chunk,
)


class IndexingError(RuntimeError):
    """Raised when chunk artifacts cannot be converted into a safe index."""


class EmbeddingEncoder(Protocol):
    """Minimal embedding contract used by the indexer and its unit tests."""

    model_id: str
    model_revision: str | None
    config_fingerprint: str

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document texts into normalized vectors."""


def _embedding_text(chunk: Chunk) -> str:
    """Give asset-only chunks a stable textual carrier without inventing content."""
    if chunk.text.strip():
        return chunk.text
    context = " / ".join(chunk.section_path)
    content = ", ".join(chunk.content_types)
    return " ".join(part for part in (context, f"[{content}]") if part)


def build_vector_points(
    chunked: ChunkedDocument,
    *,
    knowledge_base_id: UUID | None,
    embedder: EmbeddingEncoder,
) -> list[VectorPoint]:
    """Build payload-complete deterministic points from one chunk artifact."""
    texts = [_embedding_text(chunk) for chunk in chunked.chunks]
    try:
        vectors = embedder.encode_documents(texts)
    except Exception as error:
        raise IndexingError("document chunks could not be embedded") from error
    if len(vectors) != len(chunked.chunks):
        raise IndexingError("embedding result count does not match chunk count")

    points: list[VectorPoint] = []
    for chunk, vector in zip(chunked.chunks, vectors, strict=True):
        payload = ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            knowledge_base_id=knowledge_base_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            section_path=chunk.section_path,
            content_types=chunk.content_types,
            asset_refs=chunk.asset_refs,
            text=chunk.text,
            chunking_config_fingerprint=chunked.config_fingerprint,
            embedding_model=embedder.model_id,
            embedding_model_revision=embedder.model_revision,
            embedding_config_fingerprint=embedder.config_fingerprint,
        )
        points.append(
            VectorPoint(
                point_id=point_id_for_chunk(chunk.chunk_id),
                vector=tuple(float(value) for value in vector),
                payload=payload,
            )
        )
    return points


def index_chunked_document(
    chunked: ChunkedDocument,
    *,
    knowledge_base_id: UUID | None,
    store: QdrantVectorStore,
    embedder: EmbeddingEncoder,
) -> IndexResult:
    """Embed and safely replace one document's complete vector set."""
    if not chunked.chunks:
        raise IndexingError("cannot index a document without chunks")
    points = build_vector_points(
        chunked,
        knowledge_base_id=knowledge_base_id,
        embedder=embedder,
    )
    try:
        return store.replace_document(points)
    except Exception as error:
        if isinstance(error, IndexingError):
            raise
        raise IndexingError("Qdrant document replacement failed") from error


def load_chunked_artifact(path: Path, document_id: UUID | None = None) -> ChunkedDocument:
    """Load and validate one existing A1.6 chunk artifact."""
    try:
        chunked = ChunkedDocument.model_validate_json(path.read_bytes())
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValidationError) as error:
        raise IndexingError(f"invalid or missing chunk artifact: {path}") from error
    if document_id is not None and chunked.document_id != document_id:
        raise IndexingError("chunk artifact document ID does not match the requested document")
    return chunked


def index_chunk_artifact(
    path: Path,
    *,
    document_id: UUID | None = None,
    knowledge_base_id: UUID | None,
    store: QdrantVectorStore,
    embedder: EmbeddingEncoder,
) -> IndexResult:
    """Index one already-produced A1.6 artifact without rerunning MinerU."""
    return index_chunked_document(
        load_chunked_artifact(path, document_id),
        knowledge_base_id=knowledge_base_id,
        store=store,
        embedder=embedder,
    )


async def _document_knowledge_base_id(document_id: UUID, settings: Settings) -> UUID:
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            document = await session.scalar(select(Document).where(Document.id == document_id))
            if document is None:
                raise IndexingError("document was not found")
            return document.knowledge_base_id
    except IndexingError:
        raise
    except Exception as error:
        raise IndexingError("document metadata could not be loaded") from error
    finally:
        await engine.dispose()


async def index_document_by_id(
    document_id: UUID,
    settings: Settings,
    *,
    store: QdrantVectorStore | None = None,
    embedder: EmbeddingEncoder | None = None,
) -> IndexResult:
    """Index a persisted document's existing chunk artifact."""
    from .embedding import QwenEmbeddingModel

    chunk_path = Path(settings.data_dir) / "chunking" / str(document_id) / "chunks.json"
    knowledge_base_id = await _document_knowledge_base_id(document_id, settings)
    vector_store = store or QdrantVectorStore(settings)
    embedding_model = embedder or QwenEmbeddingModel(settings)
    return index_chunk_artifact(
        chunk_path,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        store=vector_store,
        embedder=embedding_model,
    )


def index_canonical_corpus(
    canonical_root: Path,
    settings: Settings,
    *,
    limit: int | None = None,
    store: QdrantVectorStore | None = None,
    embedder: EmbeddingEncoder | None = None,
) -> list[IndexResult]:
    """Index existing A1.5 canonical artifacts using the frozen A1.6 policy."""
    from .embedding import QwenEmbeddingModel

    if not canonical_root.is_dir():
        raise IndexingError(f"canonical root does not exist: {canonical_root}")
    paths = sorted(canonical_root.glob("*.json"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least one")
        paths = paths[:limit]
    if not paths:
        raise IndexingError("canonical root contains no JSON artifacts")

    vector_store = store or QdrantVectorStore(settings)
    embedding_model = embedder or QwenEmbeddingModel(settings)
    config = ChunkingConfig()
    results: list[IndexResult] = []
    for path in paths:
        try:
            document = CanonicalDocument.model_validate_json(path.read_bytes())
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise IndexingError(f"invalid canonical artifact: {path}") from error
        chunked = chunk_document(document, config)
        results.append(
            index_chunked_document(
                chunked,
                knowledge_base_id=None,
                store=vector_store,
                embedder=embedding_model,
            )
        )
    return results


__all__ = [
    "EmbeddingEncoder",
    "IndexingError",
    "build_vector_points",
    "index_canonical_corpus",
    "index_chunk_artifact",
    "index_chunked_document",
    "index_document_by_id",
    "load_chunked_artifact",
]
