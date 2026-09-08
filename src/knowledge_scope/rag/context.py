"""Deterministic, lineage-preserving context assembly for RAG QA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_scope.retrieval.reranking import RerankedChunk

from .schemas import RAGCitation


@dataclass(frozen=True, slots=True)
class SelectedContextItem:
    """One text fragment and its application-generated citation."""

    citation: RAGCitation
    text: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """The ordered context passed to the LLM and its source metadata."""

    items: tuple[SelectedContextItem, ...]
    character_count: int

    @property
    def citations(self) -> tuple[RAGCitation, ...]:
        """Return citations in the exact order used in the prompt."""
        return tuple(item.citation for item in self.items)

    def render(self) -> str:
        """Render bounded context with only application-created markers."""
        rendered: list[str] = []
        for item in self.items:
            citation = item.citation
            pages = str(citation.page_start)
            if citation.page_end != citation.page_start:
                pages = f"{pages}-{citation.page_end}"
            section = " / ".join(citation.section_path) or "未命名章节"
            rendered.append(
                f"[{citation.marker}] document_id={citation.document_id} "
                f"pages={pages} section={section}\n{item.text}"
            )
        return "\n\n".join(rendered)


def _citation_for(item: RerankedChunk, marker: str) -> RAGCitation:
    payload = item.chunk.payload
    return RAGCitation(
        marker=marker,
        document_id=payload.document_id,
        knowledge_base_id=payload.knowledge_base_id,
        chunk_id=payload.chunk_id,
        page_start=payload.page_start,
        page_end=payload.page_end,
        source_block_ids=list(payload.source_block_ids),
        section_path=list(payload.section_path),
        section_title=payload.section_path[-1] if payload.section_path else None,
    )


def _normalized_text(text: str) -> str:
    """Normalize whitespace for conservative duplicate-text detection."""
    return " ".join(text.split()).casefold()


def assemble_context(
    ranked_chunks: Sequence[RerankedChunk],
    *,
    budget_chars: int,
) -> ContextSelection:
    """Select text chunks in reranker order within a deterministic char budget.

    A chunk is included only when its complete text fits in the remaining
    character budget.  This keeps the citation lineage aligned with all text
    sent to the LLM; an oversized chunk is skipped and a later fitting chunk
    may still be selected.  Source-block overlap alone is not treated as a
    duplicate because A1.6 may split one source block across several
    non-overlapping chunks.  Exact normalized duplicate text with overlapping
    lineage is suppressed.
    """
    if budget_chars < 1:
        raise ValueError("budget_chars must be at least one")

    selected: list[SelectedContextItem] = []
    seen_chunk_ids: set[str] = set()
    selected_lineage: list[tuple[frozenset[str], str]] = []
    character_count = 0

    for ranked in ranked_chunks:
        payload = ranked.chunk.payload
        chunk_id = payload.chunk_id
        text = payload.text.strip()
        if not text or chunk_id in seen_chunk_ids:
            continue

        source_blocks = frozenset(payload.source_block_ids)
        normalized_text = _normalized_text(text)
        if not normalized_text:
            continue

        if any(
            source_blocks.intersection(previous_blocks) and normalized_text == previous_text
            for previous_blocks, previous_text in selected_lineage
        ):
            continue

        remaining = budget_chars - character_count
        if remaining <= 0:
            break
        if len(text) > remaining:
            continue

        selected.append(
            SelectedContextItem(
                citation=_citation_for(ranked, f"C{len(selected) + 1}"),
                text=text,
                truncated=False,
            )
        )
        character_count += len(text)
        selected_lineage.append((source_blocks, normalized_text))
        seen_chunk_ids.add(chunk_id)

    return ContextSelection(items=tuple(selected), character_count=character_count)


__all__ = ["ContextSelection", "SelectedContextItem", "assemble_context"]
