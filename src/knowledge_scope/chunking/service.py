"""Deterministic, structure-aware chunking for canonical documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from knowledge_scope.parsing.models import (
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
    TitleBlock,
)
from knowledge_scope.shared.config import Settings

from .models import (
    CHUNK_SCHEMA_VERSION,
    CHUNKER_VERSION,
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    ContentType,
    chunking_config_fingerprint,
)

CHUNKING_DIRECTORY_NAME = "chunking"
PARSING_DIRECTORY_NAME = "parsing"
_JOIN_SEPARATOR = "\n\n"
_SENTENCE_BOUNDARIES = frozenset(
    {"\u3002", "\uff01", "\uff1f", "!", "?", "\uff1b", ";", "\uff1a", ":", "\n"}
)


class ChunkingError(RuntimeError):
    """Raised when canonical chunking or artifact persistence cannot complete."""


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Facts returned by the developer-only chunk artifact workflow."""

    document_id: UUID
    chunk_count: int
    config_fingerprint: str
    chunks_ref: str
    manifest_ref: str
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class _BlockRef:
    page_number: int
    block: CanonicalBlock


@dataclass(frozen=True, slots=True)
class _RenderedBlock:
    text: str
    content_type: ContentType
    asset_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Section:
    index: int
    section_path: tuple[str, ...]
    blocks: tuple[_BlockRef, ...]


@dataclass(frozen=True, slots=True)
class _BuiltChunk:
    section_index: int
    section_path: tuple[str, ...]
    blocks: tuple[_BlockRef, ...]
    text: str
    content_types: tuple[ContentType, ...]
    asset_refs: tuple[str, ...]


class _TableHTMLRenderer(HTMLParser):
    """Convert simple canonical table HTML into stable readable rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[list[str]]] = []
        self._current_row: list[list[str]] | None = None
        self._current_cell: list[str] | None = None
        self._fallback_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered == "tr":
            self._finish_row()
            self._current_row = []
        elif lowered in {"td", "th"}:
            if self._current_row is None:
                self._current_row = []
            self._finish_cell()
            self._current_cell = []
        elif lowered == "br" and self._current_cell is not None:
            self._current_cell.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"td", "th"}:
            self._finish_cell()
        elif lowered == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)
        else:
            self._fallback_text.append(data)

    def _finish_cell(self) -> None:
        if self._current_cell is None:
            return
        if self._current_row is None:
            self._current_row = []
        self._current_row.append(self._current_cell)
        self._current_cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def render(self) -> str:
        self._finish_row()
        rendered_rows = [
            " | ".join(_normalize_html_cell(cell) for cell in row) for row in self.rows if row
        ]
        if rendered_rows:
            return "\n".join(rendered_rows)
        return _normalize_html_cell(self._fallback_text)


def _normalize_html_cell(parts: Iterable[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _render_html_table(value: str) -> str:
    parser = _TableHTMLRenderer()
    parser.feed(value)
    parser.close()
    return parser.render() or value


def _render_block(block: CanonicalBlock) -> _RenderedBlock:
    if isinstance(block, TitleBlock):
        return _RenderedBlock(f"## {block.text}", "title", ())
    if isinstance(block, TextBlock):
        return _RenderedBlock(block.text, "text", ())
    if isinstance(block, FormulaBlock):
        return _RenderedBlock(f"$$\n{block.latex}\n$$", "formula", ())
    if isinstance(block, TableBlock):
        representation = block.markdown or (_render_html_table(block.html) if block.html else None)
        text = _join_texts(part for part in (block.caption, representation) if part is not None)
        asset_refs = (block.asset_ref,) if block.asset_ref is not None else ()
        return _RenderedBlock(text, "table", asset_refs)
    if isinstance(block, ImageBlock):
        return _RenderedBlock(block.caption or "", "image", (block.asset_ref,))
    raise TypeError(f"unsupported canonical block: {type(block).__name__}")


def _join_texts(texts: Iterable[str]) -> str:
    """Join rendered text while keeping asset-only blocks text-free."""
    return _JOIN_SEPARATOR.join(text for text in texts if text)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _built_chunk(
    section_index: int,
    section_path: tuple[str, ...],
    blocks: tuple[_BlockRef, ...],
    *,
    text_override: str | None = None,
) -> _BuiltChunk:
    rendered = [_render_block(ref.block) for ref in blocks]
    text = (
        text_override if text_override is not None else _join_texts(item.text for item in rendered)
    )
    return _BuiltChunk(
        section_index=section_index,
        section_path=section_path,
        blocks=blocks,
        text=text,
        content_types=_unique_in_order(item.content_type for item in rendered),
        asset_refs=_unique_in_order(
            asset_ref for item in rendered for asset_ref in item.asset_refs
        ),
    )


def _split_text_at_boundaries(text: str, max_chars: int) -> tuple[str, ...]:
    """Split one oversized text block without dropping or overlapping characters."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        if hard_end == len(text):
            pieces.append(text[start:hard_end])
            break

        punctuation_end = max(
            (
                index
                for index in range(start + 1, hard_end + 1)
                if text[index - 1] in _SENTENCE_BOUNDARIES
            ),
            default=0,
        )
        if punctuation_end > start:
            end = punctuation_end
        else:
            whitespace_end = max(
                (index for index in range(start + 1, hard_end + 1) if text[index - 1].isspace()),
                default=0,
            )
            end = whitespace_end if whitespace_end > start else hard_end
        pieces.append(text[start:end])
        start = end
    return tuple(pieces)


def _sectionize(document: CanonicalDocument) -> tuple[_Section, ...]:
    sections: list[_Section] = []
    current_blocks: list[_BlockRef] = []
    current_path: tuple[str, ...] = ()
    pending_titles: list[_BlockRef] = []

    for page in document.pages:
        for block in page.blocks:
            ref = _BlockRef(page.page_number, block)
            if isinstance(block, TitleBlock):
                if current_blocks:
                    sections.append(_Section(len(sections), current_path, tuple(current_blocks)))
                    current_blocks = []
                pending_titles.append(ref)
                continue

            if pending_titles:
                current_path = tuple(ref.block.text for ref in pending_titles)
                current_blocks.extend(pending_titles)
                pending_titles = []
            current_blocks.append(ref)

    if current_blocks:
        sections.append(_Section(len(sections), current_path, tuple(current_blocks)))
    if pending_titles:
        sections.append(
            _Section(
                len(sections),
                tuple(ref.block.text for ref in pending_titles),
                tuple(pending_titles),
            )
        )
    return tuple(sections)


def _append_or_flush(
    chunks: list[_BuiltChunk],
    section: _Section,
    pending: list[_BlockRef],
) -> None:
    if pending:
        chunks.append(_built_chunk(section.index, section.section_path, tuple(pending)))
        pending.clear()


def _merge_built_chunks(previous: _BuiltChunk, last: _BuiltChunk) -> _BuiltChunk:
    """Merge already rendered chunks without reconstituting split source text."""
    return _BuiltChunk(
        section_index=previous.section_index,
        section_path=previous.section_path,
        blocks=previous.blocks + last.blocks,
        text=_join_texts((previous.text, last.text)),
        content_types=_unique_in_order(previous.content_types + last.content_types),
        asset_refs=_unique_in_order(previous.asset_refs + last.asset_refs),
    )


def _can_rebuild_built_chunk(built_chunk: _BuiltChunk) -> bool:
    """Return whether a chunk retains whole-block text rather than a split override."""
    return built_chunk.text == _join_texts(
        _render_block(ref.block).text for ref in built_chunk.blocks
    )


def _attach_formula_context(
    chunks: list[_BuiltChunk],
    config: ChunkingConfig,
) -> None:
    """Reclaim nearby whole blocks for normal-sized formula-only chunks.

    The regular target budget can leave a formula alone even when its immediate
    preceding block would fit. Rebalancing only one whole block keeps the
    operation deterministic and never reconstructs a split text block.
    """
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if current.content_types != ("formula",) or len(current.text) > config.max_chars:
            index += 1
            continue

        if index > 0:
            previous = chunks[index - 1]
            merged = _merge_built_chunks(previous, current)
            if len(merged.text) <= config.max_chars:
                chunks[index - 1 : index + 1] = [merged]
                index = max(index - 1, 0)
                continue

            if previous.blocks and _can_rebuild_built_chunk(previous):
                moved = previous.blocks[-1]
                candidate = _built_chunk(
                    current.section_index,
                    current.section_path,
                    tuple([moved, *current.blocks]),
                )
                if len(candidate.text) <= config.max_chars:
                    remaining = previous.blocks[:-1]
                    chunks[index - 1 : index + 1] = (
                        [
                            _built_chunk(
                                previous.section_index,
                                previous.section_path,
                                remaining,
                            )
                        ]
                        if remaining
                        else []
                    ) + [candidate]
                    index = max(index - 1, 0)
                    continue

        index += 1


def _append_text_with_title_context(
    chunks: list[_BuiltChunk],
    section: _Section,
    pending_titles: list[_BlockRef],
    text_ref: _BlockRef,
    config: ChunkingConfig,
) -> None:
    """Keep pending titles with the first deterministic text fragment."""
    title_text = _join_texts(_render_block(item.block).text for item in pending_titles)
    first_piece_limit = max(
        1,
        config.max_chars - len(title_text) - len(_JOIN_SEPARATOR),
    )
    pieces = _split_text_at_boundaries(text_ref.block.text, first_piece_limit)
    chunks.append(
        _built_chunk(
            section.index,
            section.section_path,
            tuple([*pending_titles, text_ref]),
            text_override=_join_texts((title_text, pieces[0])),
        )
    )
    chunks.extend(
        _built_chunk(
            section.index,
            section.section_path,
            (text_ref,),
            text_override=piece,
        )
        for piece in pieces[1:]
    )


def _chunk_section(section: _Section, config: ChunkingConfig) -> tuple[_BuiltChunk, ...]:
    chunks: list[_BuiltChunk] = []
    pending: list[_BlockRef] = []
    pending_chars = 0

    for ref in section.blocks:
        rendered = _render_block(ref.block)
        block_chars = len(rendered.text)
        if block_chars > config.max_chars:
            title_context_only = bool(pending) and all(
                isinstance(item.block, TitleBlock) for item in pending
            )
            if isinstance(ref.block, TextBlock) and title_context_only:
                _append_text_with_title_context(chunks, section, pending, ref, config)
                pending.clear()
                pending_chars = 0
            else:
                if title_context_only:
                    chunks.append(
                        _built_chunk(
                            section.index,
                            section.section_path,
                            tuple([*pending, ref]),
                        )
                    )
                    pending.clear()
                else:
                    _append_or_flush(chunks, section, pending)
                    if isinstance(ref.block, TextBlock):
                        chunks.extend(
                            _built_chunk(
                                section.index,
                                section.section_path,
                                (ref,),
                                text_override=piece,
                            )
                            for piece in _split_text_at_boundaries(ref.block.text, config.max_chars)
                        )
                    else:
                        chunks.append(_built_chunk(section.index, section.section_path, (ref,)))
                pending_chars = 0
            continue

        candidate_chars = (
            block_chars if not pending else pending_chars + len(_JOIN_SEPARATOR) + block_chars
        )
        if pending and (pending_chars >= config.target_chars or candidate_chars > config.max_chars):
            title_context_only = all(isinstance(item.block, TitleBlock) for item in pending)
            if candidate_chars > config.max_chars and title_context_only:
                if isinstance(ref.block, TextBlock):
                    _append_text_with_title_context(chunks, section, pending, ref, config)
                else:
                    chunks.append(
                        _built_chunk(
                            section.index,
                            section.section_path,
                            tuple([*pending, ref]),
                        )
                    )
                pending.clear()
                pending_chars = 0
                continue
            _append_or_flush(chunks, section, pending)
            pending_chars = 0

        pending.append(ref)
        pending_chars = (
            block_chars if len(pending) == 1 else pending_chars + len(_JOIN_SEPARATOR) + block_chars
        )

    _append_or_flush(chunks, section, pending)

    _attach_formula_context(chunks, config)

    if len(chunks) > 1 and len(chunks[-1].text) < config.min_chars:
        previous, last = chunks[-2:]
        disjoint = {ref.block.block_id for ref in previous.blocks}.isdisjoint(
            ref.block.block_id for ref in last.blocks
        )
        merged_chars = len(_join_texts((previous.text, last.text)))
        if disjoint and merged_chars <= config.max_chars:
            chunks[-2:] = [_merge_built_chunks(previous, last)]
    return tuple(chunks)


def _chunk_id(
    document_id: UUID,
    config_fingerprint: str,
    ordinal: int,
    source_block_ids: tuple[str, ...],
) -> str:
    identity = json.dumps(
        {
            "document_id": str(document_id),
            "config_fingerprint": config_fingerprint,
            "ordinal": ordinal,
            "source_block_ids": source_block_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"chunk-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _document_block_index(
    document: CanonicalDocument,
) -> tuple[dict[str, _BlockRef], dict[str, tuple[int, tuple[str, ...]]]]:
    blocks: dict[str, _BlockRef] = {}
    sections: dict[str, tuple[int, tuple[str, ...]]] = {}
    for section in _sectionize(document):
        for ref in section.blocks:
            block_id = ref.block.block_id
            blocks[block_id] = ref
            sections[block_id] = (section.index, section.section_path)
    return blocks, sections


def _validate_lineage(
    document: CanonicalDocument,
    chunked: ChunkedDocument,
) -> None:
    block_index, section_index = _document_block_index(document)
    reference_counts: Counter[str] = Counter()
    for chunk in chunked.chunks:
        refs = [block_index.get(block_id) for block_id in chunk.source_block_ids]
        if any(ref is None for ref in refs):
            raise ChunkingError(
                "chunk references a block that is absent from the canonical document"
            )
        resolved_refs = [ref for ref in refs if ref is not None]
        positions = [(ref.page_number, ref.block.reading_order) for ref in resolved_refs]
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ChunkingError("chunk source blocks are not in canonical reading order")
        expected_page_start = min(ref.page_number for ref in resolved_refs)
        expected_page_end = max(ref.page_number for ref in resolved_refs)
        if (chunk.page_start, chunk.page_end) != (expected_page_start, expected_page_end):
            raise ChunkingError("chunk page range does not match its source blocks")

        source_sections = {section_index[ref.block.block_id] for ref in resolved_refs}
        if len(source_sections) != 1:
            raise ChunkingError("chunk crosses a canonical title boundary")
        ((_, expected_path),) = source_sections
        if tuple(chunk.section_path) != expected_path:
            raise ChunkingError("chunk section_path does not match its source section")

        expected_types = _unique_in_order(ref.block.type for ref in resolved_refs)
        if tuple(chunk.content_types) != expected_types:
            raise ChunkingError("chunk content_types do not match its source blocks")
        expected_assets = _unique_in_order(
            asset_ref
            for ref in resolved_refs
            for asset_ref in (
                (ref.block.asset_ref,)
                if isinstance(ref.block, (TableBlock, ImageBlock))
                and ref.block.asset_ref is not None
                else ()
            )
        )
        if tuple(chunk.asset_refs) != expected_assets:
            raise ChunkingError("chunk asset_refs do not match its source blocks")
        reference_counts.update(chunk.source_block_ids)

    expected_block_ids = set(block_index)
    if set(reference_counts) != expected_block_ids:
        raise ChunkingError("chunking did not preserve all canonical source blocks")


def chunk_document(document: CanonicalDocument, config: ChunkingConfig) -> ChunkedDocument:
    """Build deterministic chunks from canonical structure without I/O or model calls."""
    config_fingerprint = chunking_config_fingerprint(config)
    built_chunks = [
        built_chunk
        for section in _sectionize(document)
        for built_chunk in _chunk_section(section, config)
    ]
    chunks = [
        Chunk(
            chunk_id=_chunk_id(
                document.document_id,
                config_fingerprint,
                ordinal,
                tuple(ref.block.block_id for ref in built_chunk.blocks),
            ),
            document_id=document.document_id,
            ordinal=ordinal,
            text=built_chunk.text,
            page_start=min(ref.page_number for ref in built_chunk.blocks),
            page_end=max(ref.page_number for ref in built_chunk.blocks),
            source_block_ids=[ref.block.block_id for ref in built_chunk.blocks],
            section_path=list(built_chunk.section_path),
            content_types=list(built_chunk.content_types),
            asset_refs=list(built_chunk.asset_refs),
        )
        for ordinal, built_chunk in enumerate(built_chunks)
    ]
    chunked = ChunkedDocument(
        document_id=document.document_id,
        page_count=len(document.pages),
        chunker_version=CHUNKER_VERSION,
        config_fingerprint=config_fingerprint,
        chunks=chunks,
    )
    _validate_lineage(document, chunked)
    return chunked


def _distribution(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)

    median_value = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": round(median_value, 3),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "max": ordered[-1],
    }


def summarize_chunked_document(
    document: CanonicalDocument,
    chunked: ChunkedDocument,
    config: ChunkingConfig,
) -> dict[str, object]:
    """Summarize structural coverage and size facts for one chunked document."""
    block_index, section_index = _document_block_index(document)
    reference_counts: Counter[str] = Counter(
        block_id for chunk in chunked.chunks for block_id in chunk.source_block_ids
    )
    page_counts = [
        len({block_index[block_id].page_number for block_id in chunk.source_block_ids})
        for chunk in chunked.chunks
    ]
    source_block_counts = [len(chunk.source_block_ids) for chunk in chunked.chunks]
    chunk_sections = [
        section_index[chunk.source_block_ids[0]][0]
        for chunk in chunked.chunks
        if chunk.source_block_ids
    ]
    section_boundary_splits = sum(left != right for left, right in pairwise(chunk_sections))
    section_chunk_counts = Counter(chunk_sections)
    source_blocks = list(block_index.values())
    source_asset_blocks = [
        ref
        for ref in source_blocks
        if isinstance(ref.block, (TableBlock, ImageBlock)) and ref.block.asset_ref is not None
    ]
    referenced_asset_blocks = [
        ref for ref in source_asset_blocks if reference_counts[ref.block.block_id]
    ]
    chunks_with_type = {
        block_type: sum(block_type in chunk.content_types for chunk in chunked.chunks)
        for block_type in ("table", "formula", "image")
    }
    oversized_text_blocks = sum(
        isinstance(ref.block, TextBlock) and len(_render_block(ref.block).text) > config.max_chars
        for ref in source_blocks
    )
    oversized_non_text_blocks = sum(
        not isinstance(ref.block, TextBlock)
        and len(_render_block(ref.block).text) > config.max_chars
        for ref in source_blocks
    )
    single_source_non_text_chunks = []
    atomic_non_text_chunks = []
    for chunk in chunked.chunks:
        source_refs = [block_index[block_id] for block_id in chunk.source_block_ids]
        if len(source_refs) == 1 and isinstance(
            source_refs[0].block, (TableBlock, FormulaBlock, ImageBlock)
        ):
            single_source_non_text_chunks.append((chunk, source_refs[0]))
        atomic_refs = [
            ref
            for ref in source_refs
            if isinstance(ref.block, (TableBlock, FormulaBlock, ImageBlock))
        ]
        if len(atomic_refs) == 1 and not any(
            isinstance(ref.block, TextBlock) for ref in source_refs
        ):
            atomic_non_text_chunks.append((chunk, atomic_refs[0]))
    oversized_atomic_chunks = [
        (chunk, ref)
        for chunk, ref in atomic_non_text_chunks
        if len(_render_block(ref.block).text) > config.max_chars
    ]
    oversized_atomic_by_type = {
        block_type: sum(ref.block.type == block_type for chunk, ref in oversized_atomic_chunks)
        for block_type in ("table", "formula", "image")
    }
    return {
        "documents": 1,
        "pages": len(document.pages),
        "source_blocks": {
            "eligible": len(source_blocks),
            "referenced": len(reference_counts),
            "unreferenced_meaningful": len(set(block_index) - set(reference_counts)),
            "multiply_referenced": sum(count > 1 for count in reference_counts.values()),
            "coverage_rate": round(len(reference_counts) / len(source_blocks), 6)
            if source_blocks
            else None,
        },
        "chunks": len(chunked.chunks),
        "chunk_chars": _distribution(len(chunk.text) for chunk in chunked.chunks),
        "pages_per_chunk": _distribution(page_counts),
        "source_blocks_per_chunk": _distribution(source_block_counts),
        "section_count": len(_sectionize(document)),
        "section_boundary_split_count": section_boundary_splits,
        "sections_split_by_budget": sum(count > 1 for count in section_chunk_counts.values()),
        "section_boundary_violations": 0,
        "oversized_single_block_fallback_count": oversized_text_blocks,
        "oversized_non_text_block_count": oversized_non_text_blocks,
        "chunks_over_max_chars": sum(
            len(chunk.text) > config.max_chars for chunk in chunked.chunks
        ),
        "single_source_non_text_chunks": len(single_source_non_text_chunks),
        "single_source_table_chunks": sum(
            ref.block.type == "table" for chunk, ref in single_source_non_text_chunks
        ),
        "single_source_formula_chunks": sum(
            ref.block.type == "formula" for chunk, ref in single_source_non_text_chunks
        ),
        "single_source_image_chunks": sum(
            ref.block.type == "image" for chunk, ref in single_source_non_text_chunks
        ),
        "oversized_atomic_chunks": len(oversized_atomic_chunks),
        "oversized_atomic_table_chunks": oversized_atomic_by_type["table"],
        "oversized_atomic_formula_chunks": oversized_atomic_by_type["formula"],
        "oversized_atomic_image_chunks": oversized_atomic_by_type["image"],
        "chunks_containing_tables": chunks_with_type["table"],
        "chunks_containing_formulas": chunks_with_type["formula"],
        "chunks_containing_images": chunks_with_type["image"],
        "asset_ref_coverage": {
            "source_asset_blocks": len(source_asset_blocks),
            "referenced_asset_blocks": len(referenced_asset_blocks),
            "coverage_rate": round(len(referenced_asset_blocks) / len(source_asset_blocks), 6)
            if source_asset_blocks
            else None,
        },
    }


def _artifact_root(settings: Settings) -> Path:
    root = (Path(settings.data_dir) / CHUNKING_DIRECTORY_NAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _promote_staging(staging_dir: Path, final_dir: Path) -> None:
    backup_dir: Path | None = None
    try:
        if final_dir.exists():
            backup_dir = final_dir.parent / f".{final_dir.name}.previous-{uuid4().hex}"
            os.replace(final_dir, backup_dir)
        os.replace(staging_dir, final_dir)
    except OSError as error:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise ChunkingError("chunk artifacts could not be promoted") from error
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _load_canonical_document(document_id: UUID, settings: Settings) -> CanonicalDocument:
    path = Path(settings.data_dir) / PARSING_DIRECTORY_NAME / str(document_id) / "canonical.json"
    try:
        return CanonicalDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ChunkingError("the canonical document artifact was not found") from error
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise ChunkingError("the canonical document artifact is invalid") from error


def chunk_document_by_id(
    document_id: UUID,
    settings: Settings,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Chunk an existing canonical artifact and atomically persist the result."""
    document = _load_canonical_document(document_id, settings)
    if document.document_id != document_id:
        raise ChunkingError("canonical document ID does not match the requested document")
    chunking_config = config or ChunkingConfig()
    chunked = chunk_document(document, chunking_config)
    summary = summarize_chunked_document(document, chunked, chunking_config)
    config_fingerprint = chunking_config_fingerprint(chunking_config)

    try:
        root = _artifact_root(settings)
        final_dir = root / str(document_id)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{document_id}-", dir=root))
        (staging_dir / "chunks.json").write_text(
            chunked.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "document_id": str(document_id),
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "config": chunking_config.model_dump(mode="json"),
            "config_fingerprint": config_fingerprint,
            "source_canonical_ref": f"parsing/{document_id}/canonical.json",
            "chunks_ref": f"chunking/{document_id}/chunks.json",
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "summary": summary,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _promote_staging(staging_dir, final_dir)
    except ChunkingError:
        raise
    except OSError as error:
        raise ChunkingError("chunk artifacts could not be saved") from error
    finally:
        if "staging_dir" in locals() and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return ChunkingResult(
        document_id=document_id,
        chunk_count=len(chunked.chunks),
        config_fingerprint=config_fingerprint,
        chunks_ref=f"chunking/{document_id}/chunks.json",
        manifest_ref=f"chunking/{document_id}/manifest.json",
        summary=summary,
    )


__all__ = [
    "CHUNKING_DIRECTORY_NAME",
    "ChunkingError",
    "ChunkingResult",
    "chunk_document",
    "chunk_document_by_id",
    "summarize_chunked_document",
]
