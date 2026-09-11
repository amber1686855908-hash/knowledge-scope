"""Build and validate multimodal evidence from canonical documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope.parsing.models import CanonicalBlock, CanonicalDocument

from .models import (
    EvidenceModality,
    MultimodalEvidence,
    MultimodalEvidenceDocument,
    RepresentationIndexPayload,
    canonical_document_fingerprint,
    evidence_for_block,
    evidence_id_for,
    representation_id_for,
    source_block_fingerprint,
    validate_opaque_reference,
)


class EvidenceValidationError(ValueError):
    """Raised when evidence is invalid or no longer matches its source document."""


class StaleEvidenceError(EvidenceValidationError):
    """Raised when a persisted representation does not match current canonical source."""


def _block_modality(block: CanonicalBlock) -> EvidenceModality:
    if block.type in {"title", "text"}:
        return "text"
    return block.type


def build_evidence_document(
    document: CanonicalDocument,
    knowledge_base_id: UUID,
    *,
    context_by_block_id: Mapping[str, str] | None = None,
    section_path_by_block_id: Mapping[str, Sequence[str]] | None = None,
) -> MultimodalEvidenceDocument:
    """Create one evidence item per canonical block without external model calls.

    ``context_by_block_id`` is optional text already derived by an upstream
    caller. A4.1 does not generate OCR, captions, visual embeddings, or other
    content. The authoritative source remains the canonical block.
    """

    contexts = context_by_block_id or {}
    section_paths = section_path_by_block_id or {}
    block_ids = {block.block_id for page in document.pages for block in page.blocks}
    unknown_context_ids = (set(contexts) | set(section_paths)) - block_ids
    if unknown_context_ids:
        raise EvidenceValidationError("context or section path references an unknown block")

    evidence: list[MultimodalEvidence] = []
    for page in document.pages:
        for block in page.blocks:
            evidence.append(
                evidence_for_block(
                    knowledge_base_id,
                    document.document_id,
                    page.page_number,
                    block,
                    context=contexts.get(block.block_id),
                    section_path=list(section_paths.get(block.block_id, ())),
                )
            )

    return MultimodalEvidenceDocument(
        knowledge_base_id=knowledge_base_id,
        document_id=document.document_id,
        canonical_document_fingerprint=canonical_document_fingerprint(document),
        evidence=evidence,
    )


def _current_block_index(document: CanonicalDocument) -> dict[str, tuple[int, CanonicalBlock]]:
    return {
        block.block_id: (page.page_number, block)
        for page in document.pages
        for block in page.blocks
    }


def _validate_evidence_lineage(
    item: MultimodalEvidence,
    *,
    document: CanonicalDocument,
    knowledge_base_id: UUID,
    block_index: Mapping[str, tuple[int, CanonicalBlock]],
) -> None:
    lineage = item.lineage
    if lineage.knowledge_base_id != knowledge_base_id:
        raise EvidenceValidationError("evidence knowledge_base_id does not match the request")
    if lineage.document_id != document.document_id:
        raise EvidenceValidationError("evidence document_id does not match the source document")
    if len(lineage.source_block_ids) != 1:
        raise EvidenceValidationError("A4.1 evidence must identify exactly one source block")

    resolved = []
    for block_id in lineage.source_block_ids:
        try:
            resolved.append(block_index[block_id])
        except KeyError as error:
            raise StaleEvidenceError("evidence references a missing canonical block") from error
    pages = [page_number for page_number, _ in resolved]
    if (lineage.page_start, lineage.page_end) != (min(pages), max(pages)):
        raise StaleEvidenceError("evidence page range no longer matches its source block")
    blocks = [block for _, block in resolved]
    if lineage.source_fingerprint != source_block_fingerprint(blocks):
        raise StaleEvidenceError("evidence source fingerprint does not match canonical content")

    expected_asset_refs = [
        validate_opaque_reference(ref)
        for block in blocks
        if hasattr(block, "asset_ref")
        for ref in [block.asset_ref]
        if ref is not None
    ]
    if lineage.asset_refs != expected_asset_refs:
        raise StaleEvidenceError("evidence asset lineage does not match its source block")
    if item.modality != _block_modality(blocks[0]):
        raise EvidenceValidationError("evidence modality does not match its source block")
    expected_evidence_id = evidence_id_for(lineage, item.modality)
    if item.evidence_id != expected_evidence_id:
        raise EvidenceValidationError("evidence_id does not match current lineage")


def validate_evidence_document(
    document: CanonicalDocument,
    artifact: MultimodalEvidenceDocument,
    knowledge_base_id: UUID,
) -> None:
    """Fail closed unless every artifact item still matches current canonical source."""

    try:
        artifact = MultimodalEvidenceDocument.model_validate(artifact.model_dump(mode="python"))
    except ValidationError as error:
        raise EvidenceValidationError("evidence artifact structure is invalid") from error
    if artifact.knowledge_base_id != knowledge_base_id:
        raise EvidenceValidationError("artifact knowledge_base_id does not match the request")
    if artifact.document_id != document.document_id:
        raise StaleEvidenceError("artifact document_id does not match the source document")
    current_document_fingerprint = canonical_document_fingerprint(document)
    if artifact.canonical_document_fingerprint != current_document_fingerprint:
        raise StaleEvidenceError("evidence artifact is stale for the current canonical document")

    block_index = _current_block_index(document)
    seen_block_ids: set[str] = set()
    for item in artifact.evidence:
        block_id = item.lineage.source_block_ids[0]
        if block_id in seen_block_ids:
            raise EvidenceValidationError("a canonical block cannot back duplicate evidence items")
        seen_block_ids.add(block_id)
        _validate_evidence_lineage(
            item,
            document=document,
            knowledge_base_id=knowledge_base_id,
            block_index=block_index,
        )
        for representation in item.representations:
            expected_id = representation_id_for(
                item.evidence_id,
                representation.modality,
                representation.representation_type,
                content=representation.content,
                reference=representation.reference,
            )
            if representation.representation_id != expected_id:
                raise EvidenceValidationError(
                    "representation_id does not match the current representation content"
                )
    if seen_block_ids != set(block_index):
        raise StaleEvidenceError("evidence artifact does not cover every canonical source block")


def representation_index_payloads(
    artifact: MultimodalEvidenceDocument,
    *,
    searchable_only: bool = True,
) -> list[RepresentationIndexPayload]:
    """Materialize validated index payload contracts without creating an index."""

    try:
        artifact = MultimodalEvidenceDocument.model_validate(artifact.model_dump(mode="python"))
    except ValidationError as error:
        raise EvidenceValidationError("evidence artifact structure is invalid") from error
    payloads: list[RepresentationIndexPayload] = []
    for item in artifact.evidence:
        for representation in item.representations:
            if searchable_only and not representation.searchable:
                continue
            lineage = item.lineage
            payloads.append(
                RepresentationIndexPayload(
                    representation_id=representation.representation_id,
                    evidence_id=item.evidence_id,
                    knowledge_base_id=lineage.knowledge_base_id,
                    document_id=lineage.document_id,
                    page_start=lineage.page_start,
                    page_end=lineage.page_end,
                    source_block_ids=lineage.source_block_ids,
                    section_path=lineage.section_path,
                    asset_refs=lineage.asset_refs,
                    modality=item.modality,
                    representation_type=representation.representation_type,
                    text=representation.content,
                    reference=representation.reference,
                    searchable=representation.searchable,
                    source_fingerprint=lineage.source_fingerprint,
                    canonical_document_fingerprint=artifact.canonical_document_fingerprint,
                )
            )
    return payloads


__all__ = [
    "EvidenceValidationError",
    "StaleEvidenceError",
    "build_evidence_document",
    "representation_index_payloads",
    "validate_evidence_document",
]
