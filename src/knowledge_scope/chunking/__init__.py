"""Structure-aware semantic chunking for canonical documents."""

from .models import (
    CHUNK_SCHEMA_VERSION,
    CHUNKER_VERSION,
    SPLITTING_POLICY_VERSION,
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    ContentType,
    chunking_config_fingerprint,
)
from .service import (
    CHUNKING_DIRECTORY_NAME,
    ChunkingError,
    ChunkingResult,
    chunk_document,
    chunk_document_by_id,
    summarize_chunked_document,
)

__all__ = [
    "CHUNKER_VERSION",
    "CHUNKING_DIRECTORY_NAME",
    "CHUNK_SCHEMA_VERSION",
    "SPLITTING_POLICY_VERSION",
    "Chunk",
    "ChunkedDocument",
    "ChunkingConfig",
    "ChunkingError",
    "ChunkingResult",
    "ContentType",
    "chunk_document",
    "chunk_document_by_id",
    "chunking_config_fingerprint",
    "summarize_chunked_document",
]
