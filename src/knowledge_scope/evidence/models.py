"""Provider-independent evidence and searchable representation models."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import Final, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from knowledge_scope.parsing.models import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
    TitleBlock,
)

EVIDENCE_SCHEMA_VERSION: Final = "1.0"
REPRESENTATION_SCHEMA_VERSION: Final = "1.0"
INDEX_PAYLOAD_SCHEMA_VERSION: Final = "1.0"
EVIDENCE_ID_VERSION: Final = "v1"

EvidenceModality = Literal["text", "image", "table", "formula"]
RepresentationType = Literal[
    "text",
    "caption",
    "context",
    "markdown",
    "html",
    "latex",
    "asset_ref",
    "visual_embedding_ref",
]

_TEXT_REPRESENTATION_TYPES = frozenset({"text", "caption", "context", "markdown", "html", "latex"})
_REFERENCE_REPRESENTATION_TYPES = frozenset({"asset_ref", "visual_embedding_ref"})
_REPRESENTATION_TYPES_BY_MODALITY: dict[EvidenceModality, frozenset[str]] = {
    "text": frozenset({"text", "context"}),
    "image": frozenset({"caption", "context", "asset_ref", "visual_embedding_ref"}),
    "table": frozenset({"caption", "context", "markdown", "html", "asset_ref"}),
    "formula": frozenset({"latex", "context"}),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _EvidenceBaseModel(BaseModel):
    """Strict JSON-compatible settings shared by A4.1 models."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


def _require_non_blank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one non-whitespace character")
    return normalized


def _validate_source_block_id(value: str) -> str:
    if value != value.strip():
        raise ValueError("source_block_id must not have surrounding whitespace")
    return _require_non_blank(value, "source_block_id")


def validate_opaque_reference(value: str) -> str:
    """Validate a portable reference without interpreting it as a filesystem path."""

    value = _require_non_blank(value, "reference")
    segments = value.replace("\\", "/").split("/")
    windows_path = PureWindowsPath(value)
    if (
        any(segment in {".", ".."} for segment in segments)
        or value.startswith(("/", "\\"))
        or windows_path.drive
        or windows_path.is_absolute()
        or "://" in value
    ):
        raise ValueError("reference must be an opaque non-absolute reference")
    return value


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize identity data deterministically and unambiguously as UTF-8 JSON."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_fingerprint(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 fingerprint for structured JSON data."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class EvidenceLineage(_EvidenceBaseModel):
    """Authoritative source lineage for one evidence item."""

    knowledge_base_id: UUID
    document_id: UUID
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_block_ids")
    @classmethod
    def validate_source_block_ids(cls, values: list[str]) -> list[str]:
        normalized = [_validate_source_block_id(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_block_ids must be unique")
        return normalized

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, values: list[str]) -> list[str]:
        return [_require_non_blank(value, "section_path item") for value in values]

    @field_validator("asset_refs")
    @classmethod
    def validate_asset_refs(cls, values: list[str]) -> list[str]:
        normalized = [validate_opaque_reference(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("asset_refs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        return self


def evidence_id_for(lineage: EvidenceLineage, modality: EvidenceModality) -> str:
    """Return a deterministic identity for source evidence, independent of representations."""

    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "identity_version": EVIDENCE_ID_VERSION,
        "modality": modality,
        "lineage": {
            "knowledge_base_id": str(lineage.knowledge_base_id),
            "document_id": str(lineage.document_id),
            "page_start": lineage.page_start,
            "page_end": lineage.page_end,
            "source_block_ids": list(lineage.source_block_ids),
            "asset_refs": list(lineage.asset_refs),
            "source_fingerprint": lineage.source_fingerprint,
        },
    }
    return f"evidence_{EVIDENCE_ID_VERSION}_{sha256_fingerprint(payload)}"


def representation_id_for(
    evidence_id: str,
    modality: EvidenceModality,
    representation_type: RepresentationType,
    *,
    content: str | None = None,
    reference: str | None = None,
) -> str:
    """Return a deterministic identity for one representation payload."""

    payload = {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "identity_version": EVIDENCE_ID_VERSION,
        "evidence_id": evidence_id,
        "modality": modality,
        "representation_type": representation_type,
        "content": content,
        "reference": reference,
    }
    return f"representation_{EVIDENCE_ID_VERSION}_{sha256_fingerprint(payload)}"


def source_block_fingerprint(blocks: list[CanonicalBlock]) -> str:
    """Fingerprint the ordered canonical blocks that support one evidence item."""

    return sha256_fingerprint(
        {
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "blocks": [block.model_dump(mode="json") for block in blocks],
        }
    )


def canonical_document_fingerprint(document: CanonicalDocument) -> str:
    """Fingerprint a canonical document so reparses can invalidate stale artifacts."""

    return sha256_fingerprint(
        {
            "canonical_schema_version": document.schema_version,
            "document": document.model_dump(mode="json"),
        }
    )


class EvidenceRepresentation(_EvidenceBaseModel):
    """One searchable or opaque representation of an evidence item."""

    schema_version: Literal["1.0"] = REPRESENTATION_SCHEMA_VERSION
    representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    modality: EvidenceModality
    representation_type: RepresentationType
    content: str | None = None
    reference: str | None = None
    searchable: StrictBool

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_blank(value, "content")

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_opaque_reference(value)

    @model_validator(mode="after")
    def validate_representation_contract(self) -> Self:
        allowed_types = _REPRESENTATION_TYPES_BY_MODALITY[self.modality]
        if self.representation_type not in allowed_types:
            raise ValueError(
                f"representation_type {self.representation_type!r} is not valid for "
                f"modality {self.modality!r}"
            )
        if self.representation_type in _TEXT_REPRESENTATION_TYPES:
            if self.content is None:
                raise ValueError("textual representations require content")
            if self.reference is not None:
                raise ValueError("textual representations cannot contain reference")
            if not self.searchable:
                raise ValueError("textual representations must be searchable")
        elif self.representation_type in _REFERENCE_REPRESENTATION_TYPES:
            if self.reference is None:
                raise ValueError("reference representations require reference")
            if self.content is not None:
                raise ValueError("reference representations cannot contain content")
            if self.searchable:
                raise ValueError("opaque reference representations are not searchable")
        return self


class MultimodalEvidence(_EvidenceBaseModel):
    """One authoritative source evidence item with one or more representations."""

    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    modality: EvidenceModality
    lineage: EvidenceLineage
    representations: list[EvidenceRepresentation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> Self:
        if self.evidence_id != evidence_id_for(self.lineage, self.modality):
            raise ValueError("evidence_id does not match the evidence lineage")
        representation_ids = [
            representation.representation_id for representation in self.representations
        ]
        if len(representation_ids) != len(set(representation_ids)):
            raise ValueError("representations must have unique IDs")
        for representation in self.representations:
            if representation.evidence_id != self.evidence_id:
                raise ValueError("representation evidence_id must match its evidence")
            if representation.modality != self.modality:
                raise ValueError("representation modality must match its evidence")
            expected_id = representation_id_for(
                self.evidence_id,
                representation.modality,
                representation.representation_type,
                content=representation.content,
                reference=representation.reference,
            )
            if representation.representation_id != expected_id:
                raise ValueError("representation_id does not match the representation content")
            if (
                representation.representation_type == "asset_ref"
                and representation.reference not in self.lineage.asset_refs
            ):
                raise ValueError("asset_ref representation must refer to source asset lineage")
        return self


class MultimodalEvidenceDocument(_EvidenceBaseModel):
    """Persistable evidence artifact for one canonical document."""

    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    knowledge_base_id: UUID
    document_id: UUID
    canonical_document_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: list[MultimodalEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_document_scope(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a document")
        for item in self.evidence:
            if item.lineage.knowledge_base_id != self.knowledge_base_id:
                raise ValueError("evidence knowledge_base_id must match the artifact")
            if item.lineage.document_id != self.document_id:
                raise ValueError("evidence document_id must match the artifact")
        return self


class RepresentationIndexPayload(_EvidenceBaseModel):
    """Future index payload that resolves a representation back to source evidence."""

    schema_version: Literal["1.0"] = INDEX_PAYLOAD_SCHEMA_VERSION
    representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    knowledge_base_id: UUID
    document_id: UUID
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    asset_refs: list[str]
    modality: EvidenceModality
    representation_type: RepresentationType
    text: str | None = None
    reference: str | None = None
    searchable: StrictBool
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_document_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_block_ids")
    @classmethod
    def validate_payload_block_ids(cls, values: list[str]) -> list[str]:
        normalized = [_validate_source_block_id(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_block_ids must be unique")
        return normalized

    @field_validator("section_path")
    @classmethod
    def validate_payload_section_path(cls, values: list[str]) -> list[str]:
        return [_require_non_blank(value, "section_path item") for value in values]

    @field_validator("asset_refs")
    @classmethod
    def validate_payload_asset_refs(cls, values: list[str]) -> list[str]:
        normalized = [validate_opaque_reference(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("asset_refs must be unique")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_payload_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_blank(value, "text")

    @field_validator("reference")
    @classmethod
    def validate_payload_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_opaque_reference(value)

    @model_validator(mode="after")
    def validate_payload_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        expected_searchable = self.representation_type in _TEXT_REPRESENTATION_TYPES
        if self.searchable != expected_searchable:
            raise ValueError("searchable does not match the representation type")
        if expected_searchable and self.text is None:
            raise ValueError("textual index payloads require text")
        if expected_searchable and self.reference is not None:
            raise ValueError("textual index payloads cannot contain reference")
        if not expected_searchable and self.reference is None:
            raise ValueError("reference index payloads require reference")
        if not expected_searchable and self.text is not None:
            raise ValueError("reference index payloads cannot contain text")
        if self.representation_type == "asset_ref" and self.reference not in self.asset_refs:
            raise ValueError("asset_ref reference must be included in asset_refs")
        if self.representation_type not in _REPRESENTATION_TYPES_BY_MODALITY[self.modality]:
            raise ValueError(
                f"representation_type {self.representation_type!r} is not valid for "
                f"modality {self.modality!r}"
            )
        expected_evidence_id = evidence_id_for(
            EvidenceLineage(
                knowledge_base_id=self.knowledge_base_id,
                document_id=self.document_id,
                page_start=self.page_start,
                page_end=self.page_end,
                source_block_ids=self.source_block_ids,
                section_path=self.section_path,
                asset_refs=self.asset_refs,
                source_fingerprint=self.source_fingerprint,
            ),
            self.modality,
        )
        if self.evidence_id != expected_evidence_id:
            raise ValueError("evidence_id does not match the index payload lineage")
        expected_id = representation_id_for(
            self.evidence_id,
            self.modality,
            self.representation_type,
            content=self.text,
            reference=self.reference,
        )
        if self.representation_id != expected_id:
            raise ValueError("representation_id does not match the index payload")
        return self


def _representation(
    evidence_id: str,
    modality: EvidenceModality,
    representation_type: RepresentationType,
    *,
    content: str | None = None,
    reference: str | None = None,
) -> EvidenceRepresentation:
    """Construct one representation with its application-controlled identity."""

    normalized_content = None if content is None else _require_non_blank(content, "content")
    normalized_reference = None if reference is None else validate_opaque_reference(reference)
    return EvidenceRepresentation(
        representation_id=representation_id_for(
            evidence_id,
            modality,
            representation_type,
            content=normalized_content,
            reference=normalized_reference,
        ),
        evidence_id=evidence_id,
        modality=modality,
        representation_type=representation_type,
        content=normalized_content,
        reference=normalized_reference,
        searchable=representation_type in _TEXT_REPRESENTATION_TYPES,
    )


def _block_asset_refs(block: CanonicalBlock) -> list[str]:
    if isinstance(block, (TableBlock, ImageBlock)) and block.asset_ref is not None:
        return [block.asset_ref]
    return []


def evidence_for_block(
    knowledge_base_id: UUID,
    document_id: UUID,
    page_number: int,
    block: CanonicalBlock,
    *,
    context: str | None = None,
    section_path: list[str] | None = None,
) -> MultimodalEvidence:
    """Build one evidence item from a real canonical block.

    The optional context is caller-provided text derived from the same source
    document. This phase does not generate captions, OCR, or visual embeddings.
    """

    if isinstance(block, (TitleBlock, TextBlock)):
        modality: EvidenceModality = "text"
        values: list[tuple[RepresentationType, str | None, str | None]] = [
            ("text", block.text, None)
        ]
    elif isinstance(block, TableBlock):
        modality = "table"
        values = []
        if block.markdown is not None:
            values.append(("markdown", block.markdown, None))
        elif block.html is not None:
            values.append(("html", block.html, None))
        if block.caption is not None and block.caption.strip():
            values.append(("caption", block.caption, None))
        if block.asset_ref is not None:
            values.append(("asset_ref", None, block.asset_ref))
    elif isinstance(block, FormulaBlock):
        modality = "formula"
        values = [("latex", block.latex, None)]
    elif isinstance(block, ImageBlock):
        modality = "image"
        values = []
        if block.caption is not None and block.caption.strip():
            values.append(("caption", block.caption, None))
        values.append(("asset_ref", None, block.asset_ref))
    else:
        raise TypeError(f"unsupported canonical block: {type(block).__name__}")

    if context is not None and context.strip():
        values.append(("context", context, None))
    if not values:
        raise ValueError("canonical block does not contain a supported representation")

    lineage = EvidenceLineage(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        page_start=page_number,
        page_end=page_number,
        source_block_ids=[block.block_id],
        section_path=section_path or [],
        asset_refs=_block_asset_refs(block),
        source_fingerprint=source_block_fingerprint([block]),
    )
    evidence_id = evidence_id_for(lineage, modality)
    representations = [
        _representation(
            evidence_id,
            modality,
            representation_type,
            content=content,
            reference=reference,
        )
        for representation_type, content, reference in values
    ]
    return MultimodalEvidence(
        evidence_id=evidence_id,
        modality=modality,
        lineage=lineage,
        representations=representations,
    )


__all__ = [
    "EVIDENCE_ID_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "INDEX_PAYLOAD_SCHEMA_VERSION",
    "REPRESENTATION_SCHEMA_VERSION",
    "EvidenceLineage",
    "EvidenceModality",
    "EvidenceRepresentation",
    "MultimodalEvidence",
    "MultimodalEvidenceDocument",
    "RepresentationIndexPayload",
    "RepresentationType",
    "canonical_document_fingerprint",
    "canonical_json_bytes",
    "evidence_for_block",
    "evidence_id_for",
    "representation_id_for",
    "sha256_fingerprint",
    "source_block_fingerprint",
    "validate_opaque_reference",
]
