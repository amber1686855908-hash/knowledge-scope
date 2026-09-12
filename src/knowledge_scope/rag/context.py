"""Deterministic, lineage-preserving context assembly for RAG QA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_scope.retrieval.reranking import RerankedChunk
from knowledge_scope.retrieval.unified import UnifiedCandidate

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
            prefix = (
                f"[{citation.marker}] document_id={citation.document_id} "
                f"pages={pages} section={section}"
            )
            if citation.candidate_kind == "evidence":
                prefix += (
                    f" evidence_id={citation.evidence_id} modality={citation.modality}"
                    " context_kind=representation"
                )
            rendered.append(f"{prefix}\n{item.text}")
        return "\n\n".join(rendered)


def _citation_for_dense(item: RerankedChunk, marker: str) -> RAGCitation:
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
        asset_refs=list(payload.asset_refs),
    )


def _citation_for_unified(item: UnifiedCandidate, marker: str) -> RAGCitation:
    common = {
        "marker": marker,
        "document_id": item.document_id,
        "knowledge_base_id": item.knowledge_base_id,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "source_block_ids": list(item.source_block_ids),
        "section_path": list(item.section_path),
        "section_title": item.section_path[-1] if item.section_path else None,
        "asset_refs": list(item.asset_refs),
        "final_rank": item.final_rank,
        "final_reranker_score": item.final_reranker_score,
        "branch_provenance": [
            {
                "branch": contribution.branch,
                "rank": contribution.rank,
                "score": contribution.score,
                "graph_seed_entity_id": contribution.graph_seed_entity_id,
                "graph_retrieval_reason": contribution.graph_retrieval_reason,
                "graph_paths": list(contribution.graph_paths),
            }
            for contribution in item.branches
        ],
    }
    if item.candidate_kind == "chunk":
        return RAGCitation(
            **common,
            candidate_kind="chunk",
            chunk_id=item.chunk_id,
        )
    return RAGCitation(
        **common,
        candidate_kind="evidence",
        evidence_id=item.evidence_id,
        modality=item.modality,
        representation_ids=list(item.representation_ids),
    )


def _normalized_text(text: str) -> str:
    """Normalize whitespace for conservative duplicate-text detection."""
    return " ".join(text.split()).casefold()


def _candidate_identity(citation: RAGCitation) -> tuple[str, str]:
    local_id = citation.chunk_id if citation.candidate_kind == "chunk" else citation.evidence_id
    if local_id is None:  # pragma: no cover - RAGCitation validates this invariant.
        raise ValueError("context candidate has no local identity")
    return citation.candidate_kind, local_id


def assemble_context(
    ranked_chunks: Sequence[RerankedChunk | UnifiedCandidate],
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
    seen_candidate_ids: set[tuple[str, str]] = set()
    selected_lineage: list[tuple[frozenset[str], str]] = []
    character_count = 0

    for ranked in ranked_chunks:
        if isinstance(ranked, RerankedChunk):
            citation = _citation_for_dense(ranked, f"C{len(selected) + 1}")
            text = ranked.chunk.payload.text.strip()
        elif isinstance(ranked, UnifiedCandidate):
            citation = _citation_for_unified(ranked, f"C{len(selected) + 1}")
            if ranked.candidate_kind == "chunk":
                # A chunk citation may only carry authoritative chunk text.  Its
                # rerank carrier is not a substitute for missing source text.
                text = (ranked.source_text or "").strip()
            else:
                # Evidence-level context is a searchable representation, while
                # the citation remains anchored to the authoritative Evidence.
                text = ranked.rerank_text.strip()
        else:  # pragma: no cover - the public type restricts this input.
            raise TypeError("unsupported RAG context candidate")

        candidate_id = _candidate_identity(citation)
        if not text or candidate_id in seen_candidate_ids:
            continue

        source_blocks = frozenset(citation.source_block_ids)
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
                citation=citation,
                text=text,
                truncated=False,
            )
        )
        character_count += len(text)
        selected_lineage.append((source_blocks, normalized_text))
        seen_candidate_ids.add(candidate_id)

    return ContextSelection(items=tuple(selected), character_count=character_count)


__all__ = ["ContextSelection", "SelectedContextItem", "assemble_context"]
