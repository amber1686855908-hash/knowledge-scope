from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.chunking.models import (
    CHUNK_SCHEMA_VERSION,
    CHUNKER_VERSION,
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    chunking_config_fingerprint,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _chunk(
    *,
    chunk_id: str = "chunk-1",
    document_id: UUID = DOCUMENT_ID,
    ordinal: int = 0,
    page_start: int = 1,
    page_end: int = 1,
    source_block_ids: list[str] | None = None,
    content_types: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        ordinal=ordinal,
        text="正文",
        page_start=page_start,
        page_end=page_end,
        source_block_ids=source_block_ids or ["p1-b1"],
        section_path=["章节"],
        content_types=content_types or ["text"],
    )


def test_chunking_config_has_profiled_character_baseline() -> None:
    config = ChunkingConfig()

    assert config.target_chars == 1200
    assert config.max_chars == 1600
    assert config.min_chars == 240


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_chars": 301, "target_chars": 300, "max_chars": 400},
        {"min_chars": 100, "target_chars": 401, "max_chars": 400},
        {"target_chars": 0},
    ],
)
def test_chunking_config_rejects_invalid_budget_order(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(**kwargs)


def test_chunking_config_rejects_extra_fields_and_fingerprint_changes() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(unknown_policy="not-supported")

    first = chunking_config_fingerprint(
        ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    )
    second = chunking_config_fingerprint(
        ChunkingConfig(target_chars=101, max_chars=200, min_chars=0)
    )
    assert first != second
    assert first == chunking_config_fingerprint(
        ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    )


def test_chunk_validates_lineage_shape_and_extra_fields() -> None:
    valid = _chunk()
    assert valid.schema_version == CHUNK_SCHEMA_VERSION
    assert valid.source_block_ids == ["p1-b1"]

    with pytest.raises(ValidationError):
        _chunk(page_start=2, page_end=1)
    with pytest.raises(ValidationError):
        _chunk(source_block_ids=["p1-b1", "p1-b1"])
    with pytest.raises(ValidationError):
        _chunk(source_block_ids=["   "])
    with pytest.raises(ValidationError):
        Chunk.model_validate({**valid.model_dump(), "unsupported": True})


def test_chunk_allows_asset_only_content_but_rejects_contentless_chunks() -> None:
    asset_only = Chunk(
        chunk_id="chunk-image",
        document_id=DOCUMENT_ID,
        ordinal=0,
        text="",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        content_types=["image"],
        asset_refs=["assets/image.png"],
    )

    assert asset_only.text == ""
    assert asset_only.asset_refs == ["assets/image.png"]

    with pytest.raises(ValidationError, match="text or at least one asset_ref"):
        Chunk(
            chunk_id="chunk-empty",
            document_id=DOCUMENT_ID,
            ordinal=0,
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1"],
            content_types=["image"],
        )


def test_chunked_document_requires_contiguous_ordinals_and_matching_document() -> None:
    first = _chunk()
    second = _chunk(chunk_id="chunk-2", ordinal=1, source_block_ids=["p1-b2"])
    artifact = ChunkedDocument(
        document_id=DOCUMENT_ID,
        page_count=1,
        chunker_version=CHUNKER_VERSION,
        config_fingerprint="a" * 64,
        chunks=[first, second],
    )
    assert [chunk.ordinal for chunk in artifact.chunks] == [0, 1]

    with pytest.raises(ValidationError):
        ChunkedDocument(
            document_id=DOCUMENT_ID,
            page_count=1,
            config_fingerprint="a" * 64,
            chunks=[_chunk(ordinal=1)],
        )
    with pytest.raises(ValidationError):
        ChunkedDocument(
            document_id=DOCUMENT_ID,
            page_count=1,
            config_fingerprint="a" * 64,
            chunks=[_chunk(document_id=UUID("22222222-2222-2222-2222-222222222222"))],
        )
    with pytest.raises(ValidationError):
        ChunkedDocument(
            document_id=DOCUMENT_ID,
            page_count=1,
            config_fingerprint="not-a-fingerprint",
            chunks=[],
        )
