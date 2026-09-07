"""Auditable text retrieval evaluation materialization for A2.1.

The evaluation set is anchored to canonical source blocks.  Chunk IDs are a
derived runtime view, so a later chunking policy can be rebuilt without
changing the human-facing annotations.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from knowledge_scope.chunking.models import Chunk, ChunkingConfig
from knowledge_scope.chunking.service import chunk_document
from knowledge_scope.parsing.models import (
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)

RETRIEVAL_EVAL_SCHEMA_VERSION = "1.0"
RUNTIME_EVALUATION_DIRECTORY = Path("data/evaluation/a2-1")
QUERY_TYPES = (
    "factual",
    "definition",
    "explanation",
    "comparison",
    "formula_or_table",
    "cross_block",
)
_QUESTION_MARK = "\uff1f"
VerificationStatus = Literal["candidate", "verified", "rejected"]
ReviewAction = Literal["accept", "edit", "reject"]
FinalSplit = Literal["dev", "test"]
QueryType = Literal[
    "factual",
    "definition",
    "explanation",
    "comparison",
    "formula_or_table",
    "cross_block",
]


class RetrievalEvalError(RuntimeError):
    """Raised when an evaluation artifact cannot be safely built or edited."""


class _EvaluationBaseModel(BaseModel):
    """Strict JSON contracts used by the reviewable evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")


class RetrievalEvidence(_EvaluationBaseModel):
    """One canonical evidence location for a retrieval evaluation item."""

    document_id: UUID
    page_number: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)

    @field_validator("source_block_ids")
    @classmethod
    def validate_source_block_ids(cls, value: list[str]) -> list[str]:
        if any(not block_id.strip() for block_id in value):
            raise ValueError("source_block_ids must contain non-blank IDs")
        if len(value) != len(set(value)):
            raise ValueError("source_block_ids must be unique within an evidence location")
        return value


class RetrievalEvalItem(_EvaluationBaseModel):
    """A stable query annotation with audit-only source and answer gold evidence."""

    schema_version: Literal["1.0"] = RETRIEVAL_EVAL_SCHEMA_VERSION
    item_id: str = Field(pattern=r"^a2-1-[0-9a-f]{64}$")
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    query_type: QueryType
    verification_status: VerificationStatus
    evidence: list[RetrievalEvidence] = Field(min_length=1)
    query_source: list[RetrievalEvidence] = Field(default_factory=list)
    evidence_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("query", "subject")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query and subject must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        source_keys = [
            (str(location.document_id), block_id)
            for location in self.evidence
            for block_id in location.source_block_ids
        ]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("a source block may occur only once in an evaluation item")
        query_source_keys = [
            (str(location.document_id), block_id)
            for location in self.query_source
            for block_id in location.source_block_ids
        ]
        if len(query_source_keys) != len(set(query_source_keys)):
            raise ValueError("a query-source block may occur only once in an evaluation item")
        if set(source_keys) & set(query_source_keys):
            raise ValueError("query-source blocks must not be included in gold evidence")
        expected_id = deterministic_item_id(
            self.query,
            self.subject,
            self.query_type,
            self.evidence,
            self.evidence_fingerprint,
        )
        if self.item_id != expected_id:
            raise ValueError("item_id does not match the deterministic item identity")
        return self


class SourceBlockReference(_EvaluationBaseModel):
    """A JSON-friendly canonical source-block key used in runtime reports."""

    document_id: UUID
    page_number: StrictInt = Field(ge=1)
    source_block_id: str = Field(min_length=1)


class IndexedChunk(_EvaluationBaseModel):
    """The ignored runtime chunk index used to derive relevant chunks."""

    schema_version: Literal["1.0"]
    chunk_id: str = Field(min_length=1)
    document_id: UUID
    ordinal: StrictInt = Field(ge=0)
    text: str
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    content_types: list[str] = Field(min_length=1)
    asset_refs: list[str]
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_start > self.page_end:
            raise ValueError("page_start must not exceed page_end")
        return self


class MaterializedEvalItem(_EvaluationBaseModel):
    """Derived chunk relevance and evidence coverage for one evaluation item."""

    item_id: str = Field(pattern=r"^a2-1-[0-9a-f]{64}$")
    verification_status: VerificationStatus
    relevant_chunk_ids: list[str]
    gold_source_blocks: list[SourceBlockReference] = Field(min_length=1)
    covered_source_blocks: list[SourceBlockReference]
    uncovered_source_blocks: list[SourceBlockReference]
    all_gold_blocks_covered: bool
    leakage_group_id: str = Field(default="", min_length=0)


class ExclusionRecord(_EvaluationBaseModel):
    """A canonical block intentionally excluded from text evaluation."""

    document_id: UUID
    page_number: StrictInt | None = Field(default=None, ge=1)
    source_block_id: str | None = Field(default=None, min_length=1)
    benchmark_item_id: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    detail: str | None = None

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        has_canonical_location = self.page_number is not None and self.source_block_id is not None
        has_benchmark_location = self.benchmark_item_id is not None
        if not has_canonical_location and not has_benchmark_location:
            raise ValueError("an exclusion needs a canonical or benchmark location")
        return self


class ReviewDerivedChunk(_EvaluationBaseModel):
    """Bounded chunk context shown to a human reviewer."""

    chunk_id: str = Field(min_length=1)
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_context: list[str]
    text_excerpt: str
    content_types: list[str] = Field(min_length=1)
    asset_refs: list[str]


class ReviewEvidence(_EvaluationBaseModel):
    """Bounded canonical evidence and its derived chunk context."""

    document_id: UUID
    page_number: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    evidence_excerpt: str
    section_context: list[str]
    derived_chunks: list[ReviewDerivedChunk]


class ReviewQuerySource(_EvaluationBaseModel):
    """Bounded source-question context shown separately from answer evidence."""

    document_id: UUID
    page_number: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    question_excerpt: str
    section_context: list[str]


class ReviewPackItem(_EvaluationBaseModel):
    """One JSONL record intended for local semantic review."""

    item_id: str = Field(pattern=r"^a2-1-[0-9a-f]{64}$")
    query: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    query_type: QueryType
    verification_status: VerificationStatus
    evidence: list[ReviewEvidence] = Field(min_length=1)
    query_source: list[ReviewQuerySource] = Field(default_factory=list)
    leakage_group_id: str = Field(default="", min_length=0)


class ReviewRecommendation(_EvaluationBaseModel):
    """One externally supplied human-review decision for a candidate."""

    item_id: str = Field(pattern=r"^a2-1-[0-9a-f]{64}$")
    subject: str = Field(min_length=1)
    candidate_index: StrictInt = Field(ge=1)
    original_query: str = Field(min_length=1)
    original_query_type: QueryType
    action: ReviewAction
    proposed_query: str | None = None
    proposed_query_type: QueryType | None = None
    recommended_verification_status: Literal["verified", "rejected"]
    leakage_group_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class FinalDatasetItem(_EvaluationBaseModel):
    """Repository-safe final item with an explicit dev/test assignment."""

    dataset_version: Literal["retrieval-eval-v1"] = "retrieval-eval-v1"
    split: FinalSplit
    leakage_group_id: str = Field(min_length=1)
    item: RetrievalEvalItem

    @model_validator(mode="after")
    def validate_verified_item(self) -> Self:
        if self.item.verification_status != "verified":
            raise ValueError("retrieval-eval-v1 contains verified items only")
        return self


class ValidationIssue(_EvaluationBaseModel):
    """One actionable evaluation-integrity problem."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    item_id: str | None = None


class LeakageGroup(_EvaluationBaseModel):
    """Repeated evidence or derived-chunk identity requiring split review."""

    kind: Literal["evidence", "relevant_chunks"]
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_ids: list[str] = Field(min_length=2)


class ValidationReport(_EvaluationBaseModel):
    """Non-sensitive validation summary for the runtime evaluation package."""

    valid: bool
    candidate_count: StrictInt = Field(ge=0)
    issue_count: StrictInt = Field(ge=0)
    issues: list[ValidationIssue]
    query_in_gold_evidence_count: StrictInt = Field(default=0, ge=0)
    query_in_gold_chunk_count: StrictInt = Field(default=0, ge=0)
    repeated_evidence_groups: list[LeakageGroup] = Field(default_factory=list)
    repeated_relevant_chunk_groups: list[LeakageGroup] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """A canonical document plus safe A1.5 manifest metadata."""

    document: CanonicalDocument
    subject: str
    benchmark_item_id: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class _BlockReference:
    document: CorpusDocument
    page: Page
    block: CanonicalBlock
    text: str

    @property
    def key(self) -> tuple[str, str]:
        return str(self.document.document.document_id), self.block.block_id


@dataclass(frozen=True, slots=True)
class _CandidateDraft:
    query: str
    query_type: QueryType
    evidence: tuple[_BlockReference, ...]
    origin: Literal["source_question", "definition", "derived"]
    quality_score: int
    query_source: tuple[_BlockReference, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateGenerationStats:
    """Quality accounting for deterministic candidate-pool generation."""

    source_question_count: int
    accepted_draft_count: int
    rejected_by_quality_gate: dict[str, int]

    @property
    def no_answer_evidence_count(self) -> int:
        """Return source questions rejected without plausible answer text."""
        return self.rejected_by_quality_gate.get("no_answer_evidence", 0)


def normalize_query(value: str) -> str:
    """Normalize query text for duplicate detection and stable identity."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _block_record(reference: _BlockReference) -> dict[str, object]:
    return {
        "document_id": str(reference.document.document.document_id),
        "page_number": reference.page.page_number,
        "block": reference.block.model_dump(mode="json"),
    }


def _document_block_index(
    document: CanonicalDocument,
) -> dict[tuple[int, str], _BlockReference]:
    placeholder = CorpusDocument(
        document=document,
        subject="",
        benchmark_item_id="",
        relative_path="",
    )
    return {
        (page.page_number, block.block_id): _BlockReference(
            document=placeholder,
            page=page,
            block=block,
            text=_block_text(block),
        )
        for page in document.pages
        for block in page.blocks
    }


def _reference_for_evidence(
    document: CanonicalDocument,
    location: RetrievalEvidence,
    block_id: str,
) -> _BlockReference:
    block = _document_block_index(document).get((location.page_number, block_id))
    if block is None:
        raise RetrievalEvalError(
            f"source block {block_id!r} is missing from page {location.page_number}"
        )
    return block


def evidence_fingerprint(
    document_lookup: Mapping[UUID, CanonicalDocument],
    evidence: Sequence[RetrievalEvidence],
) -> str:
    """Fingerprint the exact canonical blocks selected by an annotation."""
    records: list[dict[str, object]] = []
    for location in evidence:
        document = document_lookup.get(location.document_id)
        if document is None:
            raise RetrievalEvalError(f"document {location.document_id} is not available")
        for block_id in location.source_block_ids:
            reference = _reference_for_evidence(document, location, block_id)
            records.append(_block_record(reference))
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def deterministic_item_id(
    query: str,
    subject: str,
    query_type: QueryType,
    evidence: Sequence[RetrievalEvidence],
    evidence_hash: str,
) -> str:
    """Build an ID from annotation content, never from derived chunk IDs."""
    payload = {
        "schema_version": RETRIEVAL_EVAL_SCHEMA_VERSION,
        "query": normalize_query(query),
        "subject": subject.strip(),
        "query_type": query_type,
        "evidence": [location.model_dump(mode="json") for location in evidence],
        "evidence_fingerprint": evidence_hash,
    }
    return f"a2-1-{hashlib.sha256(_json_bytes(payload)).hexdigest()}"


def make_eval_item(
    *,
    query: str,
    subject: str,
    query_type: QueryType,
    evidence: Sequence[RetrievalEvidence],
    document_lookup: Mapping[UUID, CanonicalDocument],
    query_source: Sequence[RetrievalEvidence] = (),
    verification_status: VerificationStatus = "candidate",
) -> RetrievalEvalItem:
    """Create an item with both integrity fields computed from canonical evidence."""
    evidence_list = list(evidence)
    fingerprint = evidence_fingerprint(document_lookup, evidence_list)
    item_id = deterministic_item_id(
        query,
        subject,
        query_type,
        evidence_list,
        fingerprint,
    )
    return RetrievalEvalItem(
        item_id=item_id,
        query=query,
        subject=subject,
        query_type=query_type,
        verification_status=verification_status,
        evidence=evidence_list,
        query_source=list(query_source),
        evidence_fingerprint=fingerprint,
    )


def _rebuild_reviewed_item(
    current: RetrievalEvalItem,
    *,
    query: str,
    query_type: QueryType,
    verification_status: VerificationStatus,
) -> RetrievalEvalItem:
    """Rebuild identity-bearing fields after a review query or type changes."""
    if not query.strip():
        raise RetrievalEvalError("reviewed query must contain non-whitespace characters")
    return RetrievalEvalItem(
        item_id=deterministic_item_id(
            query,
            current.subject,
            query_type,
            current.evidence,
            current.evidence_fingerprint,
        ),
        query=query,
        subject=current.subject,
        query_type=query_type,
        verification_status=verification_status,
        evidence=current.evidence,
        query_source=current.query_source,
        evidence_fingerprint=current.evidence_fingerprint,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetrievalEvalError(f"evaluation artifact is not readable: {path.name}") from error
    records: list[dict[str, object]] = []
    try:
        for line in lines:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError
                records.append(record)
    except (TypeError, json.JSONDecodeError) as error:
        raise RetrievalEvalError(f"evaluation JSONL is invalid: {path.name}") from error
    return records


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalEvalError(f"evaluation metadata is not readable: {path.name}") from error
    if not isinstance(payload, dict):
        raise RetrievalEvalError(f"evaluation metadata is not an object: {path.name}")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    _atomic_write(
        path,
        "".join(
            f"{json.dumps(record.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)}\n"
            for record in records
        ),
    )


def _write_json(path: Path, payload: object) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def load_canonical_corpus(
    canonical_root: Path,
    corpus_manifest_path: Path,
) -> dict[UUID, CorpusDocument]:
    """Load only ready A1.5 canonical artifacts referenced by the safe manifest."""
    try:
        manifest_lines = corpus_manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetrievalEvalError("A1.5 corpus manifest could not be read") from error
    resolved_root = canonical_root.resolve()
    loaded: dict[UUID, CorpusDocument] = {}
    try:
        records = [json.loads(line) for line in manifest_lines if line.strip()]
    except json.JSONDecodeError as error:
        raise RetrievalEvalError("A1.5 corpus manifest contains invalid JSON") from error
    for record in records:
        if not isinstance(record, dict):
            raise RetrievalEvalError("A1.5 corpus manifest contains a non-object record")
        if record.get("inventory_status") != "ready":
            continue
        item_id = record.get("benchmark_item_id")
        subject = record.get("subject")
        document_id_value = record.get("benchmark_document_uuid")
        relative_path = record.get("relative_path")
        if not all(isinstance(value, str) and value for value in (item_id, subject, relative_path)):
            raise RetrievalEvalError("A1.5 corpus manifest has incomplete ready metadata")
        if not isinstance(document_id_value, str):
            raise RetrievalEvalError("A1.5 corpus manifest is missing benchmark_document_uuid")
        try:
            expected_document_id = UUID(document_id_value)
        except ValueError as error:
            raise RetrievalEvalError("A1.5 corpus manifest has an invalid document UUID") from error
        canonical_path = (canonical_root / f"{item_id}.json").resolve()
        if not canonical_path.is_file():
            canonical_path = _find_canonical_path(
                canonical_root,
                expected_document_id,
            )
        if resolved_root not in canonical_path.parents:
            raise RetrievalEvalError("canonical artifact path escapes the configured root")
        try:
            document = CanonicalDocument.model_validate_json(canonical_path.read_bytes())
        except (OSError, ValueError) as error:
            raise RetrievalEvalError(f"canonical artifact is invalid: {item_id}.json") from error
        if document.document_id != expected_document_id:
            raise RetrievalEvalError(f"canonical document ID does not match manifest: {item_id}")
        if document.document_id in loaded:
            # A1.5 keeps duplicate physical PDFs in the manifest but stores one
            # canonical artifact per content/document UUID.
            continue
        loaded[document.document_id] = CorpusDocument(
            document=document,
            subject=subject,
            benchmark_item_id=item_id,
            relative_path=relative_path,
        )
    if not loaded:
        raise RetrievalEvalError(
            "the A1.5 manifest does not reference any ready canonical documents"
        )
    return dict(
        sorted(
            loaded.items(),
            key=lambda item: (item[1].subject, item[1].benchmark_item_id),
        )
    )


def _find_canonical_path(canonical_root: Path, document_id: UUID) -> Path:
    """Resolve a deduplicated A1.5 manifest entry without exposing corpus paths."""
    for candidate in sorted(canonical_root.glob("*.json")):
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("document_id") == str(document_id):
            return candidate.resolve()
    raise RetrievalEvalError(f"canonical artifact is missing for document {document_id}")


def _strip_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("|", " ").replace("#", " ")
    value = re.sub(r"&\s*(?:#?x?27|apos);?", "'", value, flags=re.IGNORECASE)
    value = re.sub(r"&\s*(?:#?x?26|amp);?", "&", value, flags=re.IGNORECASE)
    return " ".join(html.unescape(value).split())


def _block_text(block: CanonicalBlock) -> str:
    if isinstance(block, (TitleBlock, TextBlock)):
        return _strip_markup(block.text)
    if isinstance(block, FormulaBlock):
        return " ".join(block.latex.split())
    if isinstance(block, TableBlock):
        parts = [block.caption or ""]
        if block.markdown is not None:
            parts.append(block.markdown)
        elif block.html is not None:
            parts.append(block.html)
        return _strip_markup(" ".join(part for part in parts if part))
    return _strip_markup(block.caption or "") if isinstance(block, ImageBlock) else ""


def _bounded_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


_GENERIC_LABELS = frozenset(
    {
        "学习提示",
        "综合运用",
        "资料分析",
        "本章复习题",
        "复习与提高",
        "练习与应用",
        "学思之窗",
        "探究与分享",
        "思考与讨论",
        "讨论交流",
        "整理与提升",
        "探究与拓展",
        "问题探究",
        "Key expressions",
        "Did You Know",
    }
)
_NON_STANDALONE_MARKERS = (
    "上述",
    "下列",
    "以上",
    "以下",
    "本题",
    "小题",
    "下图",
    "下表",
    "本章",
    "本单元",
    "本节",
    "本课",
    "上面",
    "下面",
    "前面",
    "后面",
    "这样的",
    "这一现象",
    "这一系列",
    "由此",
    "这段",
    "这种",
    "这些",
    "这里",
    "这",
    "它",
    "其",
    "该",
    "这个",
    "该材料",
    "材料中",
    "图中",
    "表中",
    "根据图",
    "根据表",
    "根据材料",
    "本段",
    "前文",
    "后文",
    "the passage",
    "the text",
    "the figure",
    "the table",
    "above",
    "below",
    "following",
    "互联网",
    "参观",
    "生涯规划",
    "辩论题目",
    "同学们",
    "其他同学",
    "思考\uff1a",
    "思考:",
    "this",
    "that",
    "they",
    "them",
    "their",
    "in class",
    "notebooks",
    "your home town",
    "there's a sale",
    "how about",
    "hink share",
    "同一目的",
    "穿越磁场的时间",
)
_METADATA_MARKERS = (
    "出版社",
    "出版发行",
    "课程标准",
    "教材意见反馈",
    "ISBN",
    "定价",
    "版权",
    "本书目录",
    "人民教育出版社",
)
_QUESTION_PREFIX = re.compile(
    r"^\s*(?:Q\s*[:\uFF1A:]\s*|问\s*[:\uFF1A]\s*|问题\s*|思考题\s*|练习题\s*)?"
    r"(?:[\uFF08(]?\d+[\uFF09).\u3001\uFF0E.]|\d+\s+|[①②③④⑤⑥⑦⑧⑨⑩])?\s*"
)
_QUESTION_SPLIT = re.compile(r"(?<=[\u3002\uFF01\uFF1F!?\uFF1B;])")
_ELLIPSIS = ("…", "...")
_GENERIC_QUERY_FRAGMENTS = (
    "主要事实是什么",
    "形成原因或作用是什么",
    "如何定义或理解",
    "区别或联系",
    "公式或表格信息是什么",
    "如何得到说明",
)
_QUESTION_LEAD = re.compile(
    r"(?:什么|为何|为什么|如何|怎样|哪|哪些|谁|是否|能否|可否|何时|多少|哪里|what|where|who|why|how|is|are|can)",
    flags=re.IGNORECASE,
)
_QUESTION_WORDS = (
    "什么",
    "为何",
    "为什么",
    "如何",
    "怎样",
    "哪些",
    "哪",
    "谁",
    "是否",
    "能否",
    "可否",
    "何时",
    "多少",
    "哪里",
    "何以",
    "何",
    "奚",
    "焉",
    "吗",
    "呢",
    "what",
    "where",
    "who",
    "why",
    "how",
    "which",
    "is",
    "are",
    "can",
    "do",
    "does",
    "did",
)


def _is_heading_or_metadata(value: str) -> bool:
    normalized = " ".join(value.split()).strip(" \t\r\n\u3002\uff1f?!")
    if not normalized:
        return True
    if normalized in _GENERIC_LABELS:
        return True
    if any(marker in normalized for marker in _METADATA_MARKERS):
        return True
    compact = re.sub(
        r"[\s\d\u3001,\uFF0C\u3002\uFF1B;\uFF1A:./\\()\uFF08\uFF09\[\]\u3010\u3011-]",
        "",
        normalized,
    )
    return len(compact) < 3


def _is_substantive_reference(reference: _BlockReference) -> bool:
    text = reference.text
    if _is_heading_or_metadata(text):
        return False
    if len(text) < 12:
        return False
    return any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in text)


def _question_rejection_reason(value: str) -> str | None:
    if any(marker in value for marker in _ELLIPSIS):
        return "ellipsis_or_truncation"
    if len(value) < 8:
        return "short_fragment"
    if len(value) > 180:
        return "overlong_question"
    if any(marker.casefold() in value.casefold() for marker in _NON_STANDALONE_MARKERS):
        return "non_standalone_reference"
    if any(marker in value for marker in _METADATA_MARKERS):
        return "metadata_question"
    if value.casefold().startswith(("did you know", "what do you know")):
        return "generic_heading"
    if value.startswith(("你认为", "你愿意", "你将来愿意")):
        return "subjective_prompt"
    if value.startswith(("●", "·", "•", "▪", "◦", "‣", "\u2014", "\u2013", "-")):
        return "formatting_or_ocr_prefix"
    if re.match(r"^[A-Za-z]\s*[\uFE5D\uFE5E\u3010\u3011\uFF08\uFF09\[\]() ]", value):
        return "formatting_or_ocr_prefix"
    if re.search(r"\s+[A-Za-z]{1,2}\s*[\uFF1F?]$", value):
        return "formatting_or_ocr_suffix"
    if re.match(r"^(?:yes|no)\s*,", value, flags=re.IGNORECASE) or " / " in value:
        return "answer_option_fragment"
    if value.startswith(("\uff0c", ",", "\uff1a", ":")):
        return "formatting_or_ocr_prefix"
    if any(
        value.count(opening) != value.count(closing)
        for opening, closing in (
            ("\uff08", "\uff09"),
            ("(", ")"),
            ("【", "】"),
            ("[", "]"),
            ("\ufe5d", "\u3015"),
        )
    ):
        return "unbalanced_source_fragment"
    if any(
        value.startswith(marker)
        for marker in (
            "若是",
            "如果是",
            "那么",
            "这个",
            "这种",
            "这些",
            "上述",
            "其",
            "它",
            "他们",
            "她们",
            "该",
            "意为",
            "他说",
            "她说",
            "老人说",
            "或者是",
            "分别",
            "结合家人",
            "结合所学",
            "梳理本",
            "在细胞中",
            "为什么也",
            "现在\uff0c试",
            "问者曰",
            "数值上等于",
            "这",
            "哪些对",
            "哪些属于",
            "你现在",
            "为什么说这",
            "既是这个",
            "查阅相关",
            "结合课文内容",
            "如果不唯一",
        )
    ):
        return "context_dependent_question"
    if any(
        marker in value
        for marker in (
            "你还能说出",
            "你注意过",
            "你能根据所给",
            "你能画",
            "如果请你",
            "结合自身",
            "自己构造",
            "有条件的学校",
            "你能解释",
            "你能给出",
            "你能得出",
            "你能概括",
            "你能说说",
            "你能借助",
            "自己归纳",
            "结合家人的",
            "你觉得",
            "你关注过",
            "你杀死过",
            "你知道",
            "你是否已经",
            "你能",
            "你可以",
            "你现在",
            "你能举例",
        )
    ):
        return "instructional_or_personal_prompt"
    lowered = value.casefold()
    if any(
        marker in lowered
        for marker in (
            "what is your opinion",
            "how would you",
            "what do you think",
            "what can we do",
            "what difficulties do you think",
            "give you what inspiration",
            "给你什么启示",
            "why don't we",
            "why don",
            "why do you think",
            "how often do you",
            "how did you",
            "what have you",
            "could you",
            "talk about",
            "group discussion",
            "were you able",
            "agree with",
            "in the video",
            "these days",
            "these students",
            "doing sport",
            "doing sports",
            "what benefits can you",
            " can you ",
            " do you ",
            "your ",
            "yours",
        )
    ):
        return "subjective_prompt"
    if re.match(
        r"^(?:什么是|what is)\s*(?:它|其|该|这个|这种|还要|又要|如何|为什么)",
        value,
        flags=re.IGNORECASE,
    ):
        return "context_dependent_question"
    if any(marker in value.casefold() for marker in ("查阅相关信息", "& x27", "&amp;")):
        return "instructional_or_ocr_fragment"
    if re.search(r"\s+[A-Za-z]{1,2}\s+", value) and any(
        "\u4e00" <= char <= "\u9fff" for char in value
    ):
        return "formatting_or_ocr_fragment"
    if re.search(r"\b[a-z]{12,}\b", value) and any("\u4e00" <= char <= "\u9fff" for char in value):
        return "formatting_or_ocr_fragment"
    if any(
        marker in value.casefold()
        for marker in (
            "watch the video",
            "look at the pictures",
            "answer the questions",
            "说\uff1a\u201c",
            '说:"',
            "问他",
        )
    ):
        return "instructional_or_narrative_fragment"
    if value.startswith(("1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%")):
        return "formatting_or_ocr_prefix"
    if value.endswith(("质量各\uff1f", "质量各?")):
        return "incomplete_question"
    if value.count("\uff1f") + value.count("?") == 0:
        return "not_a_question"
    if not any(word in value for word in _QUESTION_WORDS):
        return "not_a_natural_question"
    meaningful = sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in value)
    if meaningful < 6:
        return "short_fragment"
    if _is_heading_or_metadata(value):
        return "heading_or_metadata"
    return None


def _clean_source_question(value: str) -> str:
    question = _QUESTION_PREFIX.sub("", value)
    question = re.sub(
        r"^(?:例如|想一想|思考|意为)\s*[:\uFF1A]?\s*",
        "",
        question,
    )
    question = re.sub(r"^(?:例题|例\s*\d+)\s*", "", question)
    question = re.sub(r"^【(?:例题|思考题|问题)】\s*", "", question)
    question = question.strip(" \t\r\n\"\u201c\u201d\u2018\u2019'\u3001\uff0c,\uff1a:")
    question = re.sub(
        r"^结合所学\s*[\uFF0C,]?\s*谈一谈\s*[\uFF1A:]?\s*",
        "",
        question,
    )
    question = question.strip(" \t\r\n\"\u201c\u201d\u2018\u2019'\u3001\uff0c,\uff1a:")
    parts = re.split(r"[\uFF0C,\uFF1A:]", question, maxsplit=1)
    if len(parts) == 2 and len(parts[0]) <= 30:
        suffix = parts[1].lstrip(' \t\r\n"\u201c\u201d')
        if _QUESTION_LEAD.match(suffix):
            question = suffix
    return " ".join(question.split())


def _extract_source_questions(value: str) -> list[str]:
    questions: list[str] = []
    for sentence in _QUESTION_SPLIT.split(value):
        if "\uff1f" not in sentence and "?" not in sentence:
            continue
        question = _clean_source_question(sentence)
        if question:
            questions.append(question)
    return questions


def _valid_entity(value: str, *, minimum: int = 2, maximum: int = 28) -> str | None:
    entity = value.strip(" \t\r\n\uff0c,\u3002\uff1b;\uff1a:")
    entity = re.split(r"(?:其中|而且|并且|因此|所以)$", entity)[-1].strip()
    if not minimum <= len(entity) <= maximum:
        return None
    if any(char in entity for char in ("\ufe5d", "\u3015", "【", "】", "(", ")", "[", "]")):
        return None
    if re.search(r"[A-Za-z]{1,2}", entity) and any("\u4e00" <= char <= "\u9fff" for char in entity):
        return None
    if entity.endswith(("各", "的", "吗", "呢")):
        return None
    if entity.endswith(("只", "却", "仅仅", "因为")):
        return None
    if _is_heading_or_metadata(entity):
        return None
    if any(
        marker in entity
        for marker in (
            "还要",
            "又要",
            "因为",
            "并不仅仅",
            "主要原理",
            "法定服务期限",
            "最大电容",
            "最小电容",
            "人均财富排名",
            "如何",
            "为什么",
            "理解作者",
            "大多数",
            "许多",
            "一些",
            "某些",
        )
    ):
        return None
    if re.search(r"在[^\uFF0C,\u3002\uFF1B;\uFF1A:]{1,16}中$", entity):
        return None
    if any(marker in entity for marker in _NON_STANDALONE_MARKERS):
        return None
    if not any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in entity):
        return None
    return entity


def _valid_definition_concept(value: str) -> str | None:
    """Reject fragments that are not standalone concepts despite a definition cue."""
    if any(
        marker in value
        for marker in (
            "因为",
            "所以",
            "因此",
            "因而",
            "后来",
            "有时",
            "也被",
            "他们",
            "有人认为",
            "研究对象",
            "式中的",
            "通常",
            "分别",
            "把",
            "将",
            "曾将",
            "种由",
            "人们",
            "类学家",
            "展绿色经济",
        )
    ):
        return None
    if any(
        char.isdigit() or char in "$\\\uff08\uff09()\u201c\u201d\u2018\u2019\u3010\u3011"
        for char in value
    ):
        return None
    if value.startswith(("第", "1.", "2.", "3.", "4.", "5.")):
        return None
    if value.endswith(("被", "把", "统", "主要", "通常", "并")):
        return None
    return value


def _query_occurs_in_text(query: str, text: str) -> bool:
    """Compare both normalized and punctuation-free query text."""
    normalized_query = normalize_query(query)
    normalized_text = normalize_query(text)
    if normalized_query in normalized_text:
        return True
    compact_query = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized_query)
    compact_text = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized_text)
    return bool(compact_query) and compact_query in compact_text


def _same_section_references(
    reference: _BlockReference,
    references: Sequence[_BlockReference],
) -> tuple[_BlockReference, ...]:
    same_document = [
        candidate
        for candidate in references
        if candidate.document.document.document_id == reference.document.document.document_id
    ]
    try:
        position = next(
            index for index, candidate in enumerate(same_document) if candidate.key == reference.key
        )
    except StopIteration:
        return (reference,)
    previous_title = max(
        (index for index in range(position) if isinstance(same_document[index].block, TitleBlock)),
        default=None,
    )
    next_title = next(
        (
            index
            for index in range(position + 1, len(same_document))
            if isinstance(same_document[index].block, TitleBlock)
        ),
        None,
    )
    start = previous_title + 1 if previous_title is not None else 0
    end = next_title if next_title is not None else len(same_document)
    section = same_document[start:end]
    if section:
        return tuple(section)
    return tuple(
        candidate
        for candidate in same_document
        if candidate.page.page_number == reference.page.page_number
    )


def _is_answer_bearing_reference(reference: _BlockReference, query: str) -> bool:
    """Allow only substantive text-like blocks that can answer a query."""
    if _query_occurs_in_text(query, reference.text):
        return False
    if isinstance(reference.block, TextBlock):
        if not _is_substantive_reference(reference):
            return False
        sentences = _QUESTION_SPLIT.split(reference.text)
        return not sentences or any(
            sentence.strip() and _QUESTION_MARK not in sentence and "?" not in sentence
            for sentence in sentences
        )
    if isinstance(reference.block, FormulaBlock):
        return len(reference.text) >= 5 and any(char.isalnum() for char in reference.text)
    if isinstance(reference.block, TableBlock):
        return len(_table_cells(reference.block)) >= 2
    return False


def _question_answer_evidence(
    reference: _BlockReference,
    query: str,
    references: Sequence[_BlockReference],
) -> tuple[_BlockReference, ...]:
    """Find nearby same-section answer blocks without reusing the question block."""
    section = _same_section_references(reference, references)
    try:
        position = next(index for index, item in enumerate(section) if item.key == reference.key)
    except StopIteration:
        return ()
    candidates = [
        (0 if index > position else 1, abs(index - position), index, candidate)
        for index, candidate in enumerate(section)
        if candidate.key != reference.key and _is_answer_bearing_reference(candidate, query)
    ]
    candidates.sort(key=lambda item: item[:3])
    selected = [candidate for *_order, candidate in candidates[:3]]
    return tuple(sorted(selected, key=lambda item: item.block.reading_order))


def _query_type_for_source_question(
    query: str,
    evidence: Sequence[_BlockReference],
) -> QueryType:
    lowered = query.casefold()
    if any(
        isinstance(reference.block, (FormulaBlock, TableBlock)) for reference in evidence
    ) or any(token in query for token in ("公式", "表格", "多少", "数值", "哪几类", "哪些类别")):
        return "formula_or_table"
    if any(token in query for token in ("区别", "不同", "异同", "相比", "比较", "联系")) or (
        "difference" in lowered
    ):
        return "comparison"
    if any(token in query for token in ("什么是", "指什么", "含义", "定义")) or (
        "what is" in lowered
    ):
        return "definition"
    if any(
        token in query.casefold()
        for token in (
            "为什么",
            "为何",
            "原因",
            "如何",
            "怎样",
            "作用",
            "影响",
            "机制",
            "why",
            "how",
        )
    ):
        if len(evidence) > 1 and any(token in query for token in ("关系", "联系", "分别", "共同")):
            return "cross_block"
        return "explanation"
    if len(evidence) > 1 and any(
        token in query for token in ("关系", "联系", "结合", "分别", "共同")
    ):
        return "cross_block"
    return "factual"


def _source_question_drafts(
    references: Sequence[_BlockReference],
    rejections: Counter[str],
    chunk_index: Mapping[str, IndexedChunk] | None = None,
) -> list[_CandidateDraft]:
    drafts: list[_CandidateDraft] = []
    chunks_by_block = _chunks_by_block(chunk_index) if chunk_index is not None else None
    for reference in references:
        if not isinstance(reference.block, (TextBlock, TitleBlock)):
            continue
        for raw_question in _extract_source_questions(reference.text):
            reason = _question_rejection_reason(raw_question)
            if reason is not None:
                rejections[reason] += 1
                continue
            evidence = _question_answer_evidence(reference, raw_question, references)
            if chunk_index is not None:
                source_keys = {reference.key}
                evidence = tuple(
                    answer_reference
                    for answer_reference in evidence
                    if chunks_by_block is not None
                    and _reference_has_clean_chunk(answer_reference, chunks_by_block, source_keys)
                )
            if not evidence:
                rejections["no_answer_evidence"] += 1
                continue
            drafts.append(
                _CandidateDraft(
                    query=raw_question,
                    query_type=_query_type_for_source_question(raw_question, evidence),
                    evidence=evidence,
                    origin="source_question",
                    quality_score=100 + min(len(evidence), 3),
                    query_source=(reference,),
                )
            )
    return drafts


def _definition_drafts(
    references: Sequence[_BlockReference],
    rejections: Counter[str],
) -> list[_CandidateDraft]:
    drafts: list[_CandidateDraft] = []
    pattern = re.compile(
        r"(?P<concept>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})(?:是指|指的是|可以理解为|定义为|称为)"
        r"(?P<definition>[^\u3002\uFF01\uFF1F?!]{8,90})"
    )
    for reference in references:
        if (
            not isinstance(reference.block, TextBlock)
            or not _is_substantive_reference(reference)
            or _extract_source_questions(reference.text)
        ):
            continue
        for match in pattern.finditer(reference.text):
            concept = _valid_entity(match.group("concept"))
            concept = _valid_definition_concept(concept) if concept is not None else None
            if concept is None:
                rejections["invalid_definition_concept"] += 1
                continue
            query = f"什么是{concept}{_QUESTION_MARK}"
            drafts.append(
                _CandidateDraft(
                    query=query,
                    query_type="definition",
                    evidence=(reference,),
                    origin="definition",
                    quality_score=92,
                )
            )
    return drafts


def _derived_fact_drafts(
    references: Sequence[_BlockReference],
    rejections: Counter[str],
) -> list[_CandidateDraft]:
    drafts: list[_CandidateDraft] = []
    patterns: tuple[tuple[re.Pattern[str], str, QueryType], ...] = (
        (
            re.compile(
                r"(?P<entity>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})包括"
                r"(?P<details>[^\u3002\uFF01\uFF1F?!]{2,48})"
            ),
            "{entity}包括哪些内容{question}",
            "factual",
        ),
        (
            re.compile(
                r"(?P<entity>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})分为"
                r"(?P<details>[^\u3002\uFF01\uFF1F?!]{2,48})"
            ),
            "{entity}分为哪几类{question}",
            "factual",
        ),
        (
            re.compile(
                r"(?P<entity>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})由"
                r"(?P<details>[^\u3002\uFF01\uFF1F?!]{2,48})(?:组成|构成)"
            ),
            "{entity}由什么组成{question}",
            "factual",
        ),
        (
            re.compile(
                r"(?P<entity>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})"
                r"(?:的原因是|原因在于)(?P<details>[^\u3002\uFF01\uFF1F?!]{4,60})"
            ),
            "为什么{entity}{question}",
            "explanation",
        ),
        (
            re.compile(
                r"(?P<entity>[^\uFF0C,\u3002\uFF1B;\uFF1A:]{2,24})"
                r"(?:导致|使得|促使)(?P<details>[^\u3002\uFF01\uFF1F?!]{4,60})"
            ),
            "{entity}会带来什么影响{question}",
            "explanation",
        ),
    )
    for reference in references:
        if not isinstance(reference.block, TextBlock) or not _is_substantive_reference(reference):
            continue
        for sentence in re.split(r"(?<=[\u3002\uFF01\uFF1F?!\uFF1B;])", reference.text):
            if "\uff1f" in sentence or "?" in sentence:
                continue
            for pattern, template, query_type in patterns:
                match = pattern.search(sentence)
                if match is None:
                    continue
                entity = _valid_entity(match.group("entity"))
                if entity is None:
                    rejections["invalid_derived_entity"] += 1
                    continue
                query = template.format(entity=entity, question=_QUESTION_MARK)
                drafts.append(
                    _CandidateDraft(
                        query=query,
                        query_type=query_type,
                        evidence=(reference,),
                        origin="derived",
                        quality_score=76,
                    )
                )
                break
    return drafts


def _table_cells(block: TableBlock) -> list[list[str]]:
    if block.markdown is not None:
        rows = []
        for line in block.markdown.splitlines():
            if "|" not in line or set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}:
                continue
            cells = [" ".join(cell.split()) for cell in line.strip().strip("|").split("|")]
            if any(cells):
                rows.append(cells)
        return rows
    if block.html is not None:
        rows = []
        for raw_row in re.findall(
            r"<tr[^>]*>(.*?)</tr>",
            block.html,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            cells = [
                _strip_markup(cell)
                for cell in re.findall(
                    r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>",
                    raw_row,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            ]
            if any(cells):
                rows.append(cells)
        return rows
    return []


def _formula_table_drafts(
    references: Sequence[_BlockReference],
    rejections: Counter[str],
) -> list[_CandidateDraft]:
    drafts: list[_CandidateDraft] = []
    for reference in references:
        if isinstance(reference.block, FormulaBlock):
            formula = " ".join(reference.block.latex.split())
            if len(formula) > 80 or any(marker in formula for marker in _ELLIPSIS):
                rejections["unbounded_formula"] += 1
                continue
            if "=" not in formula:
                rejections["non_relational_formula"] += 1
                continue
            drafts.append(
                _CandidateDraft(
                    query=f"公式 {formula} 表示哪些量之间的关系{_QUESTION_MARK}",
                    query_type="formula_or_table",
                    evidence=(reference,),
                    origin="derived",
                    quality_score=88,
                )
            )
        elif isinstance(reference.block, TableBlock):
            rows = _table_cells(reference.block)
            if len(rows) < 2 or len(rows[0]) < 2:
                rejections["unstructured_table"] += 1
                continue
            header = rows[0][1]
            row = next((candidate for candidate in rows[1:] if len(candidate) >= 2), None)
            if row is None or not _valid_entity(row[0], minimum=1, maximum=24):
                rejections["unanswerable_table_row"] += 1
                continue
            label = _valid_entity(row[0], minimum=1, maximum=24)
            assert label is not None
            if not _valid_entity(header, minimum=1, maximum=24):
                rejections["unanswerable_table_header"] += 1
                continue
            drafts.append(
                _CandidateDraft(
                    query=f"表格中\uff0c{label}对应的{header}是什么{_QUESTION_MARK}",
                    query_type="formula_or_table",
                    evidence=(reference,),
                    origin="derived",
                    quality_score=86,
                )
            )
    return drafts


def _eligible_references(
    corpus: Iterable[CorpusDocument],
) -> tuple[dict[str, list[_BlockReference]], list[ExclusionRecord]]:
    by_subject: dict[str, list[_BlockReference]] = defaultdict(list)
    exclusions: list[ExclusionRecord] = []
    for corpus_document in corpus:
        for page in corpus_document.document.pages:
            for block in page.blocks:
                if isinstance(block, (TitleBlock, TextBlock, FormulaBlock)):
                    reference = _BlockReference(
                        document=corpus_document,
                        page=page,
                        block=block,
                        text=_block_text(block),
                    )
                    if reference.text:
                        by_subject[corpus_document.subject].append(reference)
                    continue
                if isinstance(block, TableBlock):
                    if block.markdown is None and block.html is None:
                        exclusions.append(
                            ExclusionRecord(
                                document_id=corpus_document.document.document_id,
                                page_number=page.page_number,
                                source_block_id=block.block_id,
                                reason="table_missing_content",
                            )
                        )
                        continue
                    reference = _BlockReference(
                        document=corpus_document,
                        page=page,
                        block=block,
                        text=_block_text(block),
                    )
                    if reference.text:
                        by_subject[corpus_document.subject].append(reference)
                    continue
                if isinstance(block, ImageBlock):
                    reason = (
                        "captionless_image_only"
                        if block.caption is None or not block.caption.strip()
                        else "image_evidence_out_of_scope"
                    )
                    exclusions.append(
                        ExclusionRecord(
                            document_id=corpus_document.document.document_id,
                            page_number=page.page_number,
                            source_block_id=block.block_id,
                            reason=reason,
                        )
                    )
    for references in by_subject.values():
        references.sort(
            key=lambda reference: (
                reference.document.benchmark_item_id,
                reference.page.page_number,
                reference.block.reading_order,
                reference.block.block_id,
            )
        )
    exclusions.sort(
        key=lambda record: (str(record.document_id), record.page_number, record.source_block_id)
    )
    return dict(by_subject), exclusions


def load_a15_missing_table_exclusions(
    corpus: Mapping[UUID, CorpusDocument],
    results_path: Path,
) -> list[ExclusionRecord]:
    """Record A1.5 table warnings that have no canonical source block to search."""
    if not results_path.is_file():
        return []
    exclusions: list[ExclusionRecord] = []
    for record in _read_jsonl(results_path):
        if not isinstance(record.get("table_missing_content"), int):
            continue
        if record["table_missing_content"] < 1:
            continue
        document_id_value = record.get("benchmark_document_uuid")
        item_id = record.get("benchmark_item_id")
        if not isinstance(document_id_value, str) or not isinstance(item_id, str):
            continue
        try:
            document_id = UUID(document_id_value)
        except ValueError:
            continue
        if document_id not in corpus:
            continue
        warnings = record.get("warnings")
        details = (
            [
                warning
                for warning in warnings
                if isinstance(warning, str) and warning.startswith("table_missing_content:")
            ]
            if isinstance(warnings, list)
            else []
        )
        if not details:
            details = [f"count={record['table_missing_content']}"]
        exclusions.extend(
            ExclusionRecord(
                document_id=document_id,
                benchmark_item_id=item_id,
                reason="table_missing_content",
                detail=detail,
            )
            for detail in details
        )
    return exclusions


def _draft_key(draft: _CandidateDraft) -> tuple[object, ...]:
    first_reference = min(
        draft.evidence,
        key=lambda reference: (
            reference.document.document.document_id,
            reference.page.page_number,
            reference.block.reading_order,
            reference.block.block_id,
        ),
    )
    origin_rank = {"source_question": 3, "definition": 2, "derived": 1}[draft.origin]
    return (
        -draft.quality_score,
        -origin_rank,
        str(first_reference.document.document.document_id),
        first_reference.page.page_number,
        first_reference.block.reading_order,
        normalize_query(draft.query),
        tuple(reference.key for reference in draft.evidence),
    )


def _comparison_is_supported(draft: _CandidateDraft) -> bool:
    if draft.query_type != "comparison":
        return True
    query = draft.query.casefold()
    has_relation_word = any(
        marker in query
        for marker in ("区别", "不同", "异同", "相比", "比较", "联系", "difference", "compare")
    )
    has_pair_marker = any(
        marker in query for marker in ("和", "与", "之间", "及", "between", " and ")
    )
    return has_relation_word and has_pair_marker


def _candidate_quality_gate(draft: _CandidateDraft) -> str | None:
    query = draft.query.strip()
    if any(fragment in query for fragment in _GENERIC_QUERY_FRAGMENTS):
        return "generic_template"
    if any(marker in query for marker in _ELLIPSIS):
        return "ellipsis_or_truncation"
    if len(query) < 8:
        return "short_fragment"
    if len(query) > 220:
        return "overlong_question"
    if query.count("\uff1f") + query.count("?") == 0:
        return "not_a_question"
    if _is_heading_or_metadata(query):
        return "heading_or_metadata"
    if any(_query_occurs_in_text(query, reference.text) for reference in draft.evidence):
        return "query_in_gold_evidence"
    if not _comparison_is_supported(draft):
        return "comparison_without_comparable_pair"
    if draft.origin != "source_question" and not any(
        _is_substantive_reference(reference)
        or (isinstance(reference.block, (FormulaBlock, TableBlock)) and len(reference.text) >= 5)
        for reference in draft.evidence
    ):
        return "insubstantial_evidence"
    return None


def _candidate_drafts_for_subject(
    references: Sequence[_BlockReference],
    rejections: Counter[str],
    chunk_index: Mapping[str, IndexedChunk] | None = None,
) -> list[_CandidateDraft]:
    drafts = [
        *_source_question_drafts(references, rejections, chunk_index),
        *_definition_drafts(references, rejections),
        *_derived_fact_drafts(references, rejections),
        *_formula_table_drafts(references, rejections),
    ]
    unique: dict[tuple[str, tuple[tuple[str, int, int, str], ...]], _CandidateDraft] = {}
    for draft in drafts:
        reason = _candidate_quality_gate(draft)
        if reason is not None:
            rejections[reason] += 1
            continue
        evidence_key = tuple(
            sorted(
                (
                    str(reference.document.document.document_id),
                    reference.page.page_number,
                    reference.block.reading_order,
                    reference.block.block_id,
                )
                for reference in draft.evidence
            )
        )
        key = (normalize_query(draft.query), evidence_key)
        if key in unique:
            rejections["duplicate_draft"] += 1
            continue
        unique[key] = draft
    return sorted(unique.values(), key=_draft_key)


def _select_candidate_drafts(
    drafts: Sequence[_CandidateDraft],
    *,
    per_subject: int,
    globally_seen_queries: set[str],
) -> list[_CandidateDraft]:
    """Select a quality-first, type-diverse slice without enforcing fake quotas."""
    available = [
        draft for draft in drafts if normalize_query(draft.query) not in globally_seen_queries
    ]
    selected: list[_CandidateDraft] = []
    selected_queries: set[str] = set()
    for query_type in QUERY_TYPES:
        candidate = next(
            (
                draft
                for draft in available
                if draft.query_type == query_type
                and normalize_query(draft.query) not in selected_queries
            ),
            None,
        )
        if candidate is None:
            continue
        selected.append(candidate)
        selected_queries.add(normalize_query(candidate.query))
        if len(selected) == per_subject:
            break
    for draft in available:
        normalized = normalize_query(draft.query)
        if normalized in selected_queries:
            continue
        selected.append(draft)
        selected_queries.add(normalized)
        if len(selected) == per_subject:
            break
    globally_seen_queries.update(selected_queries)
    return selected


def _generate_candidate_items_with_stats(
    corpus: Mapping[UUID, CorpusDocument],
    *,
    per_subject: int = 18,
    chunk_index: Mapping[str, IndexedChunk] | None = None,
) -> tuple[list[RetrievalEvalItem], list[ExclusionRecord], CandidateGenerationStats]:
    if per_subject < 1:
        raise ValueError("per_subject must be at least one")
    by_subject, exclusions = _eligible_references(corpus.values())
    document_lookup = {document_id: entry.document for document_id, entry in corpus.items()}
    selected: list[RetrievalEvalItem] = []
    globally_seen_queries: set[str] = set()
    rejections: Counter[str] = Counter()
    accepted_draft_count = 0
    source_question_count = 0
    for subject in sorted(by_subject):
        drafts = _candidate_drafts_for_subject(by_subject[subject], rejections, chunk_index)
        accepted_draft_count += len(drafts)
        selected_drafts = _select_candidate_drafts(
            drafts,
            per_subject=per_subject,
            globally_seen_queries=globally_seen_queries,
        )
        if len(selected_drafts) < per_subject:
            raise RetrievalEvalError(
                f"subject {subject!r} has only {len(selected_drafts)} usable candidates"
            )
        source_question_count += sum(draft.origin == "source_question" for draft in selected_drafts)
        selected.extend(
            _draft_to_item(draft, document_lookup, subject) for draft in selected_drafts
        )
    selected.sort(key=lambda item: item.item_id)
    return (
        selected,
        exclusions,
        CandidateGenerationStats(
            source_question_count=source_question_count,
            accepted_draft_count=accepted_draft_count,
            rejected_by_quality_gate=dict(sorted(rejections.items())),
        ),
    )


def _draft_to_item(
    draft: _CandidateDraft,
    document_lookup: Mapping[UUID, CanonicalDocument],
    subject: str,
) -> RetrievalEvalItem:
    def to_locations(references: Sequence[_BlockReference]) -> list[RetrievalEvidence]:
        by_page: dict[tuple[UUID, int], list[_BlockReference]] = defaultdict(list)
        for reference in references:
            by_page[(reference.document.document.document_id, reference.page.page_number)].append(
                reference
            )
        return [
            RetrievalEvidence(
                document_id=document_id,
                page_number=page_number,
                source_block_ids=[
                    reference.block.block_id
                    for reference in sorted(items, key=lambda item: item.block.reading_order)
                ],
            )
            for (document_id, page_number), items in sorted(
                by_page.items(), key=lambda item: (str(item[0][0]), item[0][1])
            )
        ]

    locations = to_locations(draft.evidence)
    query_source = to_locations(draft.query_source)
    return make_eval_item(
        query=draft.query,
        subject=subject,
        query_type=draft.query_type,
        evidence=locations,
        document_lookup=document_lookup,
        query_source=query_source,
    )


def generate_candidate_items(
    corpus: Mapping[UUID, CorpusDocument],
    *,
    per_subject: int = 18,
) -> tuple[list[RetrievalEvalItem], list[ExclusionRecord]]:
    """Draft deterministic candidates from inspected canonical evidence."""
    items, exclusions, _stats = _generate_candidate_items_with_stats(
        corpus,
        per_subject=per_subject,
    )
    return items, exclusions


def _indexed_chunk(chunk: Chunk, config_fingerprint: str) -> IndexedChunk:
    return IndexedChunk(
        schema_version=chunk.schema_version,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        ordinal=chunk.ordinal,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        source_block_ids=list(chunk.source_block_ids),
        section_path=list(chunk.section_path),
        content_types=list(chunk.content_types),
        asset_refs=list(chunk.asset_refs),
        config_fingerprint=config_fingerprint,
    )


def _gold_references(
    item: RetrievalEvalItem,
    corpus: Mapping[UUID, CorpusDocument],
) -> list[SourceBlockReference]:
    return [
        SourceBlockReference(
            document_id=location.document_id,
            page_number=location.page_number,
            source_block_id=block_id,
        )
        for location in item.evidence
        for block_id in location.source_block_ids
    ]


def _query_source_keys(item: RetrievalEvalItem) -> set[tuple[str, str]]:
    return {
        (str(location.document_id), block_id)
        for location in item.query_source
        for block_id in location.source_block_ids
    }


def _chunk_contains_query_source(
    chunk: IndexedChunk,
    query_source_keys: set[tuple[str, str]],
) -> bool:
    return any(
        (str(chunk.document_id), block_id) in query_source_keys
        for block_id in chunk.source_block_ids
    )


def _reference_has_clean_chunk(
    reference: _BlockReference,
    chunks_by_block: Mapping[tuple[str, str], Sequence[IndexedChunk]],
    query_source_keys: set[tuple[str, str]],
) -> bool:
    return any(
        not _chunk_contains_query_source(chunk, query_source_keys)
        for chunk in chunks_by_block.get(reference.key, ())
    )


def _chunks_by_block(
    chunk_index: Mapping[str, IndexedChunk],
) -> dict[tuple[str, str], list[IndexedChunk]]:
    chunks_by_block: dict[tuple[str, str], list[IndexedChunk]] = defaultdict(list)
    for chunk in chunk_index.values():
        for block_id in chunk.source_block_ids:
            chunks_by_block[(str(chunk.document_id), block_id)].append(chunk)
    for chunks in chunks_by_block.values():
        chunks.sort(key=lambda value: value.ordinal)
    return chunks_by_block


def _clean_relevant_chunks(
    item: RetrievalEvalItem,
    chunk_index: Mapping[str, IndexedChunk],
    corpus: Mapping[UUID, CorpusDocument],
) -> tuple[list[SourceBlockReference], list[str], set[tuple[str, str]]]:
    gold = _gold_references(item, corpus)
    query_source_keys = _query_source_keys(item)
    chunks_by_block = _chunks_by_block(chunk_index)
    relevant: list[str] = []
    covered_keys: set[tuple[str, str]] = set()
    for reference in gold:
        key = (str(reference.document_id), reference.source_block_id)
        for chunk in chunks_by_block.get(key, ()):
            if _chunk_contains_query_source(chunk, query_source_keys):
                continue
            relevant.append(chunk.chunk_id)
            covered_keys.add(key)
    return gold, list(dict.fromkeys(relevant)), covered_keys


def _single_item_group_id(item_id: str) -> str:
    return f"a2-1-group-{hashlib.sha256(item_id.encode('utf-8')).hexdigest()}"


def derive_materialized_item(
    item: RetrievalEvalItem,
    chunk_index: Mapping[str, IndexedChunk],
    corpus: Mapping[UUID, CorpusDocument],
    *,
    leakage_group_id: str | None = None,
) -> MaterializedEvalItem:
    """Map answer evidence to chunks while excluding query-source chunks."""
    gold, relevant_ids, covered_keys = _clean_relevant_chunks(item, chunk_index, corpus)
    covered = [
        reference
        for reference in gold
        if (str(reference.document_id), reference.source_block_id) in covered_keys
    ]
    uncovered = [
        reference
        for reference in gold
        if (str(reference.document_id), reference.source_block_id) not in covered_keys
    ]
    return MaterializedEvalItem(
        item_id=item.item_id,
        verification_status=item.verification_status,
        relevant_chunk_ids=relevant_ids,
        gold_source_blocks=gold,
        covered_source_blocks=covered,
        uncovered_source_blocks=uncovered,
        all_gold_blocks_covered=not uncovered,
        leakage_group_id=leakage_group_id or _single_item_group_id(item.item_id),
    )


def build_review_pack_item(
    item: RetrievalEvalItem,
    chunk_index: Mapping[str, IndexedChunk],
    corpus: Mapping[UUID, CorpusDocument],
    *,
    leakage_group_id: str | None = None,
) -> ReviewPackItem:
    """Build separate bounded query-source and answer-evidence review context."""
    by_block = _chunks_by_block(chunk_index)
    query_source_keys = _query_source_keys(item)
    review_evidence: list[ReviewEvidence] = []
    for location in item.evidence:
        document = corpus.get(location.document_id)
        if document is None:
            raise RetrievalEvalError(f"review document is missing: {location.document_id}")
        references = [
            _reference_for_evidence(document.document, location, block_id)
            for block_id in location.source_block_ids
        ]
        derived: dict[str, IndexedChunk] = {}
        for reference in references:
            for chunk in by_block.get(reference.key, ()):
                if _chunk_contains_query_source(chunk, query_source_keys):
                    continue
                derived[chunk.chunk_id] = chunk
        derived_chunks = [
            ReviewDerivedChunk(
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source_block_ids=chunk.source_block_ids,
                section_context=chunk.section_path,
                text_excerpt=_bounded_text(chunk.text, 480),
                content_types=chunk.content_types,
                asset_refs=chunk.asset_refs,
            )
            for chunk in sorted(derived.values(), key=lambda value: value.ordinal)
        ]
        section_context = list(
            dict.fromkeys(title for chunk in derived.values() for title in chunk.section_path)
        )
        review_evidence.append(
            ReviewEvidence(
                document_id=location.document_id,
                page_number=location.page_number,
                source_block_ids=location.source_block_ids,
                evidence_excerpt=_bounded_text(
                    " ".join(reference.text for reference in references),
                    360,
                ),
                section_context=section_context,
                derived_chunks=derived_chunks,
            )
        )
    review_query_source: list[ReviewQuerySource] = []
    for location in item.query_source:
        document = corpus.get(location.document_id)
        if document is None:
            raise RetrievalEvalError(f"query-source document is missing: {location.document_id}")
        references = [
            _reference_for_evidence(document.document, location, block_id)
            for block_id in location.source_block_ids
        ]
        section_context = list(
            dict.fromkeys(
                title
                for reference in references
                for chunk in by_block.get(reference.key, ())
                for title in chunk.section_path
            )
        )
        review_query_source.append(
            ReviewQuerySource(
                document_id=location.document_id,
                page_number=location.page_number,
                source_block_ids=location.source_block_ids,
                question_excerpt=_bounded_text(
                    " ".join(reference.text for reference in references),
                    360,
                ),
                section_context=section_context,
            )
        )
    return ReviewPackItem(
        item_id=item.item_id,
        query=item.query,
        subject=item.subject,
        query_type=item.query_type,
        verification_status=item.verification_status,
        evidence=review_evidence,
        query_source=review_query_source,
        leakage_group_id=leakage_group_id or _single_item_group_id(item.item_id),
    )


def _validate_source_order(
    item: RetrievalEvalItem,
    corpus: Mapping[UUID, CorpusDocument],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for location_kind, locations in (
        ("evidence", item.evidence),
        ("query_source", item.query_source),
    ):
        previous_key: tuple[str, int, int] | None = None
        for location in locations:
            corpus_document = corpus.get(location.document_id)
            if corpus_document is None:
                issues.append(
                    ValidationIssue(
                        code="missing_document",
                        message=(
                            f"{location_kind} document {location.document_id} is not in "
                            "the current corpus"
                        ),
                        item_id=item.item_id,
                    )
                )
                continue
            page = next(
                (
                    page
                    for page in corpus_document.document.pages
                    if page.page_number == location.page_number
                ),
                None,
            )
            if page is None:
                issues.append(
                    ValidationIssue(
                        code="missing_page",
                        message=(
                            f"{location_kind} page {location.page_number} is missing from "
                            f"document {location.document_id}"
                        ),
                        item_id=item.item_id,
                    )
                )
                continue
            blocks = {block.block_id: block for block in page.blocks}
            for block_id in location.source_block_ids:
                block = blocks.get(block_id)
                if block is None:
                    issues.append(
                        ValidationIssue(
                            code="missing_block",
                            message=(
                                f"{location_kind} block {block_id!r} is missing from page "
                                f"{location.page_number}"
                            ),
                            item_id=item.item_id,
                        )
                    )
                    continue
                if isinstance(block, ImageBlock) or (
                    isinstance(block, TableBlock) and block.markdown is None and block.html is None
                ):
                    issues.append(
                        ValidationIssue(
                            code="evidence_out_of_scope",
                            message=(
                                f"{location_kind} block {block_id!r} is not searchable text "
                                "evidence"
                            ),
                            item_id=item.item_id,
                        )
                    )
                key = (str(location.document_id), location.page_number, block.reading_order)
                if previous_key is not None and key < previous_key:
                    issues.append(
                        ValidationIssue(
                            code="invalid_source_order",
                            message=f"{location_kind} blocks are not in canonical document order",
                            item_id=item.item_id,
                        )
                    )
                previous_key = key
    return issues


def _evidence_signature(evidence: Sequence[RetrievalEvidence]) -> str:
    payload = sorted(
        (str(location.document_id), location.page_number, block_id)
        for location in evidence
        for block_id in location.source_block_ids
    )
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _relevant_chunk_signature(chunk_ids: Sequence[str]) -> str:
    return hashlib.sha256(_json_bytes(sorted(set(chunk_ids)))).hexdigest()


def _repeated_leakage_groups(
    kind: Literal["evidence", "relevant_chunks"],
    signatures: Mapping[str, list[str]],
) -> list[LeakageGroup]:
    return [
        LeakageGroup(kind=kind, signature=signature, item_ids=sorted(item_ids))
        for signature, item_ids in sorted(signatures.items())
        if len(item_ids) > 1
    ]


def _leakage_group_ids(
    items: Sequence[RetrievalEvalItem],
    chunk_index: Mapping[str, IndexedChunk],
    corpus: Mapping[UUID, CorpusDocument],
) -> dict[str, str]:
    """Union items sharing answer evidence or a relevant-chunk set."""
    item_tokens: dict[str, set[str]] = {}
    token_items: dict[str, list[str]] = defaultdict(list)
    for item in items:
        _gold, relevant_ids, _covered = _clean_relevant_chunks(item, chunk_index, corpus)
        tokens = {
            f"evidence:{_evidence_signature(item.evidence)}",
            f"chunks:{_relevant_chunk_signature(relevant_ids)}",
        }
        item_tokens[item.item_id] = tokens
        for token in tokens:
            token_items[token].append(item.item_id)

    neighbors: dict[str, set[str]] = defaultdict(set)
    for item_ids in token_items.values():
        for item_id in item_ids:
            neighbors[item_id].update(other for other in item_ids if other != item_id)
    group_ids: dict[str, str] = {}
    for item_id in sorted(item_tokens):
        if item_id in group_ids:
            continue
        component: set[str] = set()
        pending = [item_id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        group_id = (
            "a2-1-group-" + hashlib.sha256("|".join(sorted(component)).encode("utf-8")).hexdigest()
        )
        group_ids.update({component_item: group_id for component_item in component})
    return group_ids


def validate_items(
    items: Sequence[RetrievalEvalItem],
    chunk_index: Mapping[str, IndexedChunk],
    corpus: Mapping[UUID, CorpusDocument],
) -> ValidationReport:
    """Validate current corpus references, fingerprints, order, and chunk coverage."""
    issues: list[ValidationIssue] = []
    normalized_queries: dict[str, str] = {}
    seen_item_ids: set[str] = set()
    document_lookup = {document_id: entry.document for document_id, entry in corpus.items()}
    chunks_by_block: dict[tuple[str, str], list[IndexedChunk]] = defaultdict(list)
    evidence_signatures: dict[str, list[str]] = defaultdict(list)
    relevant_chunk_signatures: dict[str, list[str]] = defaultdict(list)
    query_in_gold_evidence_count = 0
    query_in_gold_chunk_count = 0
    for chunk in chunk_index.values():
        document = corpus.get(chunk.document_id)
        if document is None:
            issues.append(
                ValidationIssue(
                    code="stale_chunk_document",
                    message=f"chunk {chunk.chunk_id} references an unknown document",
                )
            )
        else:
            document_block_ids = {
                block.block_id for page in document.document.pages for block in page.blocks
            }
            if chunk.page_end > len(document.document.pages):
                issues.append(
                    ValidationIssue(
                        code="stale_chunk_page_range",
                        message=f"chunk {chunk.chunk_id} exceeds the current page count",
                    )
                )
            for block_id in chunk.source_block_ids:
                if block_id not in document_block_ids:
                    issues.append(
                        ValidationIssue(
                            code="stale_chunk_block",
                            message=f"chunk {chunk.chunk_id} references an unknown source block",
                        )
                    )
        for block_id in chunk.source_block_ids:
            chunks_by_block[(str(chunk.document_id), block_id)].append(chunk)
    for item in items:
        if item.item_id in seen_item_ids:
            issues.append(
                ValidationIssue(
                    code="duplicate_item_id",
                    message="item_id occurs more than once",
                    item_id=item.item_id,
                )
            )
        seen_item_ids.add(item.item_id)
        normalized = normalize_query(item.query)
        previous_item = normalized_queries.get(normalized)
        if previous_item is not None:
            issues.append(
                ValidationIssue(
                    code="duplicate_normalized_query",
                    message=f"query is normalized-duplicate of {previous_item}",
                    item_id=item.item_id,
                )
            )
        normalized_queries[normalized] = item.item_id
        issues.extend(_validate_source_order(item, corpus))
        query_source_keys = _query_source_keys(item)
        evidence_signatures[_evidence_signature(item.evidence)].append(item.item_id)
        try:
            expected_fingerprint = evidence_fingerprint(document_lookup, item.evidence)
        except RetrievalEvalError as error:
            issues.append(
                ValidationIssue(
                    code="stale_evidence",
                    message=str(error),
                    item_id=item.item_id,
                )
            )
        else:
            if expected_fingerprint != item.evidence_fingerprint:
                issues.append(
                    ValidationIssue(
                        code="evidence_fingerprint_drift",
                        message="canonical evidence no longer matches evidence_fingerprint",
                        item_id=item.item_id,
                    )
                )
        if query_source_keys & {
            (str(location.document_id), block_id)
            for location in item.evidence
            for block_id in location.source_block_ids
        }:
            issues.append(
                ValidationIssue(
                    code="query_source_in_gold_evidence",
                    message="query-source blocks must not be used as gold evidence",
                    item_id=item.item_id,
                )
            )
        evidence_references: list[_BlockReference] = []
        for location in item.evidence:
            document = corpus.get(location.document_id)
            if document is None:
                continue
            for block_id in location.source_block_ids:
                try:
                    evidence_references.append(
                        _reference_for_evidence(document.document, location, block_id)
                    )
                except RetrievalEvalError:
                    continue
        if any(
            _query_occurs_in_text(item.query, reference.text) for reference in evidence_references
        ):
            query_in_gold_evidence_count += 1
            issues.append(
                ValidationIssue(
                    code="query_in_gold_evidence",
                    message="normalized query text occurs in gold answer evidence",
                    item_id=item.item_id,
                )
            )
        gold, relevant_ids, covered_keys = _clean_relevant_chunks(item, chunk_index, corpus)
        relevant_chunk_signatures[_relevant_chunk_signature(relevant_ids)].append(item.item_id)
        for reference in gold:
            key = (str(reference.document_id), reference.source_block_id)
            if key not in covered_keys:
                issues.append(
                    ValidationIssue(
                        code="evidence_not_covered",
                        message=(
                            f"no current chunk contains answer source block "
                            f"{reference.source_block_id!r} without query-source leakage"
                        ),
                        item_id=item.item_id,
                    )
                )
        relevant_chunks = [chunk_index[chunk_id] for chunk_id in relevant_ids]
        if any(_query_occurs_in_text(item.query, chunk.text) for chunk in relevant_chunks):
            query_in_gold_chunk_count += 1
            issues.append(
                ValidationIssue(
                    code="query_in_gold_chunk",
                    message="normalized query text occurs in a derived gold chunk",
                    item_id=item.item_id,
                )
            )
    repeated_evidence_groups = _repeated_leakage_groups("evidence", evidence_signatures)
    repeated_relevant_chunk_groups = _repeated_leakage_groups(
        "relevant_chunks", relevant_chunk_signatures
    )
    return ValidationReport(
        valid=not issues,
        candidate_count=len(items),
        issue_count=len(issues),
        issues=issues,
        query_in_gold_evidence_count=query_in_gold_evidence_count,
        query_in_gold_chunk_count=query_in_gold_chunk_count,
        repeated_evidence_groups=repeated_evidence_groups,
        repeated_relevant_chunk_groups=repeated_relevant_chunk_groups,
    )


def _materialize_chunks(
    corpus: Mapping[UUID, CorpusDocument],
    config: ChunkingConfig,
) -> dict[str, IndexedChunk]:
    chunks: dict[str, IndexedChunk] = {}
    for corpus_document in corpus.values():
        chunked = chunk_document(corpus_document.document, config)
        for chunk in chunked.chunks:
            indexed = _indexed_chunk(chunk, chunked.config_fingerprint)
            if indexed.chunk_id in chunks:
                raise RetrievalEvalError(f"duplicate derived chunk ID: {indexed.chunk_id}")
            chunks[indexed.chunk_id] = indexed
    return dict(
        sorted(
            chunks.items(),
            key=lambda item: (str(item[1].document_id), item[1].ordinal),
        )
    )


def _status_counts(items: Sequence[RetrievalEvalItem]) -> dict[str, int]:
    counts = Counter(item.verification_status for item in items)
    return {status: counts.get(status, 0) for status in ("candidate", "verified", "rejected")}


def _manifest(
    *,
    items: Sequence[RetrievalEvalItem],
    corpus: Mapping[UUID, CorpusDocument],
    chunk_index: Mapping[str, IndexedChunk],
    exclusions: Sequence[ExclusionRecord],
    materialized: Sequence[MaterializedEvalItem],
    validation: ValidationReport,
    generation_stats: CandidateGenerationStats,
) -> dict[str, object]:
    return {
        "schema_version": RETRIEVAL_EVAL_SCHEMA_VERSION,
        "candidate_count": len(items),
        "status_counts": _status_counts(items),
        "query_source_count": sum(bool(item.query_source) for item in items),
        "subject_counts": dict(sorted(Counter(item.subject for item in items).items())),
        "query_type_counts": dict(sorted(Counter(item.query_type for item in items).items())),
        "candidate_generation": {
            "source_question_count": generation_stats.source_question_count,
            "accepted_draft_count": generation_stats.accepted_draft_count,
            "rejected_by_quality_gate": generation_stats.rejected_by_quality_gate,
            "rejected_count": sum(generation_stats.rejected_by_quality_gate.values()),
            "no_answer_evidence_count": generation_stats.no_answer_evidence_count,
        },
        "corpus_document_count": len(corpus),
        "chunk_count": len(chunk_index),
        "chunk_config_fingerprints": sorted(
            {chunk.config_fingerprint for chunk in chunk_index.values()}
        ),
        "exclusion_counts": dict(sorted(Counter(record.reason for record in exclusions).items())),
        "all_gold_blocks_covered": all(record.all_gold_blocks_covered for record in materialized),
        "leakage": {
            "query_in_gold_evidence_count": validation.query_in_gold_evidence_count,
            "query_in_gold_chunk_count": validation.query_in_gold_chunk_count,
            "repeated_evidence_groups": [
                group.model_dump(mode="json") for group in validation.repeated_evidence_groups
            ],
            "repeated_relevant_chunk_groups": [
                group.model_dump(mode="json") for group in validation.repeated_relevant_chunk_groups
            ],
            "leakage_group_count": len(
                {record.leakage_group_id for record in materialized if record.leakage_group_id}
            ),
        },
        "validation": validation.model_dump(mode="json"),
        "files": [
            "candidates.jsonl",
            "chunk_index.jsonl",
            "materialized.jsonl",
            "review-pack.jsonl",
            "exclusions.jsonl",
        ],
        "annotation_identity": (
            "query -> evidence(document_id, page_number, source_block_ids); "
            "query_source is audit-only"
        ),
    }


def materialize_retrieval_eval_set(
    canonical_root: Path,
    corpus_manifest_path: Path,
    output_dir: Path = RUNTIME_EVALUATION_DIRECTORY,
    *,
    config: ChunkingConfig | None = None,
) -> dict[str, object]:
    """Build ignored runtime chunks, candidate annotations, and review JSONL."""
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    chunked_config = config or ChunkingConfig()
    chunk_index = _materialize_chunks(corpus, chunked_config)
    items, exclusions, generation_stats = _generate_candidate_items_with_stats(
        corpus,
        chunk_index=chunk_index,
    )
    exclusions.extend(
        load_a15_missing_table_exclusions(
            corpus,
            corpus_manifest_path.with_name("results.jsonl"),
        )
    )
    validation = validate_items(items, chunk_index, corpus)
    if not validation.valid:
        raise RetrievalEvalError(
            "generated retrieval evaluation set failed validation: "
            + "; ".join(issue.code for issue in validation.issues[:5])
        )
    group_ids = _leakage_group_ids(items, chunk_index, corpus)
    materialized = [
        derive_materialized_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in items
    ]
    review_pack = [
        build_review_pack_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in items
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "candidates.jsonl", items)
    _write_jsonl(output_dir / "chunk_index.jsonl", chunk_index.values())
    _write_jsonl(output_dir / "materialized.jsonl", materialized)
    _write_jsonl(output_dir / "review-pack.jsonl", review_pack)
    _write_jsonl(output_dir / "exclusions.jsonl", exclusions)
    manifest = _manifest(
        items=items,
        corpus=corpus,
        chunk_index=chunk_index,
        exclusions=exclusions,
        materialized=materialized,
        validation=validation,
        generation_stats=generation_stats,
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_runtime_chunk_index(output_dir: Path) -> dict[str, IndexedChunk]:
    """Load the existing chunk index without invoking the chunking service."""
    path = output_dir / "chunk_index.jsonl"
    try:
        indexed = [IndexedChunk.model_validate(record) for record in _read_jsonl(path)]
    except ValueError as error:
        raise RetrievalEvalError("runtime chunk index does not match the A2.1 schema") from error
    if not indexed:
        raise RetrievalEvalError(
            "runtime chunk index is empty; run retrieval-eval build before refreshing candidates"
        )
    chunk_index = {chunk.chunk_id: chunk for chunk in indexed}
    if len(chunk_index) != len(indexed):
        raise RetrievalEvalError("runtime chunk index contains duplicate chunk IDs")
    return chunk_index


def _load_runtime_exclusions(output_dir: Path) -> list[ExclusionRecord]:
    path = output_dir / "exclusions.jsonl"
    try:
        return [ExclusionRecord.model_validate(record) for record in _read_jsonl(path)]
    except ValueError as error:
        raise RetrievalEvalError("runtime exclusions do not match the A2.1 schema") from error


def regenerate_candidate_review_pack(
    canonical_root: Path,
    corpus_manifest_path: Path,
    output_dir: Path = RUNTIME_EVALUATION_DIRECTORY,
) -> dict[str, object]:
    """Refresh candidates and review artifacts while preserving the chunk index."""
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    chunk_index = _load_runtime_chunk_index(output_dir)
    exclusions = _load_runtime_exclusions(output_dir)
    items, _generated_exclusions, generation_stats = _generate_candidate_items_with_stats(
        corpus,
        chunk_index=chunk_index,
    )
    validation = validate_items(items, chunk_index, corpus)
    if not validation.valid:
        raise RetrievalEvalError(
            "refreshed retrieval evaluation set failed validation: "
            + "; ".join(issue.code for issue in validation.issues[:5])
        )
    group_ids = _leakage_group_ids(items, chunk_index, corpus)
    materialized = [
        derive_materialized_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in items
    ]
    review_pack = [
        build_review_pack_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in items
    ]
    _write_jsonl(output_dir / "candidates.jsonl", items)
    _write_jsonl(output_dir / "materialized.jsonl", materialized)
    _write_jsonl(output_dir / "review-pack.jsonl", review_pack)
    manifest = _manifest(
        items=items,
        corpus=corpus,
        chunk_index=chunk_index,
        exclusions=exclusions,
        materialized=materialized,
        validation=validation,
        generation_stats=generation_stats,
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_runtime_evaluation(
    output_dir: Path,
    canonical_root: Path,
    corpus_manifest_path: Path,
) -> ValidationReport:
    """Reload and validate an ignored runtime evaluation package."""
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    try:
        items = [
            RetrievalEvalItem.model_validate(record)
            for record in _read_jsonl(output_dir / "candidates.jsonl")
        ]
        indexed = [
            IndexedChunk.model_validate(record)
            for record in _read_jsonl(output_dir / "chunk_index.jsonl")
        ]
    except ValueError as error:
        raise RetrievalEvalError(
            "runtime evaluation JSONL does not match the A2.1 schema"
        ) from error
    chunk_index = {chunk.chunk_id: chunk for chunk in indexed}
    return validate_items(items, chunk_index, corpus)


def _load_review_recommendations(path: Path) -> list[ReviewRecommendation]:
    """Load strict human-review decisions from a JSONL file."""
    try:
        return [ReviewRecommendation.model_validate(record) for record in _read_jsonl(path)]
    except ValueError as error:
        raise RetrievalEvalError(
            "human review recommendations do not match the A2.1 review schema"
        ) from error


def _assign_final_splits(
    items: Sequence[RetrievalEvalItem],
    group_ids: Mapping[str, str],
    *,
    dev_per_subject: int,
    test_per_subject: int,
) -> dict[str, FinalSplit]:
    """Assign whole leakage groups while satisfying exact subject quotas."""
    verified = [item for item in items if item.verification_status == "verified"]
    subject_totals = Counter(item.subject for item in verified)
    expected_total = dev_per_subject + test_per_subject
    if any(total != expected_total for total in subject_totals.values()):
        raise RetrievalEvalError(
            "verified items must have the configured total per subject before splitting"
        )

    grouped: dict[str, list[RetrievalEvalItem]] = defaultdict(list)
    for item in verified:
        group_id = group_ids.get(item.item_id)
        if not group_id:
            raise RetrievalEvalError(f"verified item {item.item_id} has no leakage group")
        grouped[group_id].append(item)

    group_counts: dict[str, Counter[str]] = {
        group_id: Counter(item.subject for item in group_items)
        for group_id, group_items in grouped.items()
    }
    multi_groups = sorted(
        (
            group_id,
            counts,
        )
        for group_id, counts in group_counts.items()
        if len(grouped[group_id]) > 1
    )
    single_groups_by_subject: dict[str, list[str]] = defaultdict(list)
    for group_id, group_items in grouped.items():
        if len(group_items) == 1:
            single_groups_by_subject[group_items[0].subject].append(group_id)
    for group_ids_for_subject in single_groups_by_subject.values():
        group_ids_for_subject.sort()

    subjects = sorted(subject_totals)
    remaining_multi: list[Counter[str]] = [Counter() for _ in range(len(multi_groups) + 1)]
    for index in range(len(multi_groups) - 1, -1, -1):
        remaining_multi[index] = remaining_multi[index + 1].copy()
        remaining_multi[index].update(multi_groups[index][1])

    def search(
        index: int,
        dev_counts: Counter[str],
        assignments: dict[str, FinalSplit],
    ) -> dict[str, FinalSplit] | None:
        for subject in subjects:
            singleton_count = len(single_groups_by_subject.get(subject, ()))
            if dev_counts[subject] > dev_per_subject:
                return None
            if (
                dev_counts[subject] + remaining_multi[index][subject] + singleton_count
                < dev_per_subject
            ):
                return None

        if index == len(multi_groups):
            resolved = dict(assignments)
            for subject in subjects:
                needed = dev_per_subject - dev_counts[subject]
                singleton_groups = single_groups_by_subject.get(subject, [])
                if needed < 0 or needed > len(singleton_groups):
                    return None
                for singleton_index, group_id in enumerate(singleton_groups):
                    resolved[group_id] = "dev" if singleton_index < needed else "test"
            return resolved

        group_id, counts = multi_groups[index]
        for split in ("dev", "test"):
            assignments[group_id] = split
            next_counts = dev_counts.copy()
            if split == "dev":
                next_counts.update(counts)
            result = search(index + 1, next_counts, assignments)
            if result is not None:
                return result
            del assignments[group_id]
        return None

    group_assignments = search(0, Counter(), {})
    if group_assignments is None:
        raise RetrievalEvalError(
            "no dev/test split satisfies per-subject quotas without splitting leakage groups"
        )

    item_splits: dict[str, FinalSplit] = {}
    for group_id, group_items in grouped.items():
        split = group_assignments[group_id]
        for item in group_items:
            item_splits[item.item_id] = split
    return item_splits


def finalize_retrieval_eval_dataset(
    output_dir: Path,
    recommendations_path: Path,
    canonical_root: Path,
    corpus_manifest_path: Path,
    *,
    final_output_dir: Path | None = None,
    repository_safe_path: Path = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl"),
    repository_manifest_path: Path = Path("docs/benchmarks/a2-1-retrieval-eval-manifest.json"),
) -> dict[str, object]:
    """Apply all review decisions and materialize the verified v1 dataset."""
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    chunk_index = _load_runtime_chunk_index(output_dir)
    recommendations = _load_review_recommendations(recommendations_path)
    try:
        current_items = [
            RetrievalEvalItem.model_validate(record)
            for record in _read_jsonl(output_dir / "candidates.jsonl")
        ]
        current_materialized = [
            MaterializedEvalItem.model_validate(record)
            for record in _read_jsonl(output_dir / "materialized.jsonl")
        ]
    except ValueError as error:
        raise RetrievalEvalError("current A2.1 runtime artifacts are invalid") from error

    baseline = validate_items(current_items, chunk_index, corpus)
    if not baseline.valid:
        raise RetrievalEvalError(
            "current A2.1 candidate pool failed validation: "
            + "; ".join(issue.code for issue in baseline.issues[:5])
        )

    expected_actions = Counter({"accept": 12, "edit": 96, "reject": 54})
    action_counts = Counter(recommendation.action for recommendation in recommendations)
    if action_counts != expected_actions:
        raise RetrievalEvalError(
            "review action counts do not match the expected 12 accept, 96 edit, 54 reject"
        )
    if len(recommendations) != len(current_items):
        raise RetrievalEvalError("human review recommendations must cover every candidate")

    current_by_id = {item.item_id: item for item in current_items}
    if len(current_by_id) != len(current_items):
        raise RetrievalEvalError("current candidate pool contains duplicate item IDs")
    recommendation_by_id = {
        recommendation.item_id: recommendation for recommendation in recommendations
    }
    if len(recommendation_by_id) != len(recommendations):
        raise RetrievalEvalError("human review recommendations contain duplicate item IDs")
    if set(recommendation_by_id) != set(current_by_id):
        raise RetrievalEvalError("human review recommendations do not match current candidates")

    materialized_by_id = {item.item_id: item for item in current_materialized}
    if set(materialized_by_id) != set(current_by_id):
        raise RetrievalEvalError("materialized A2.1 items do not match current candidates")

    subject_positions: dict[str, dict[str, int]] = defaultdict(dict)
    for item in current_items:
        positions = subject_positions[item.subject]
        positions[item.item_id] = len(positions) + 1

    updated_by_old_id: dict[str, RetrievalEvalItem] = {}
    for current in current_items:
        recommendation = recommendation_by_id[current.item_id]
        if recommendation.subject != current.subject:
            raise RetrievalEvalError(f"review subject mismatch for {current.item_id}")
        if recommendation.candidate_index != subject_positions[current.subject][current.item_id]:
            raise RetrievalEvalError(f"review candidate index mismatch for {current.item_id}")
        if recommendation.original_query != current.query:
            raise RetrievalEvalError(f"review original query mismatch for {current.item_id}")
        if recommendation.original_query_type != current.query_type:
            raise RetrievalEvalError(f"review original query type mismatch for {current.item_id}")
        if recommendation.leakage_group_id != materialized_by_id[current.item_id].leakage_group_id:
            raise RetrievalEvalError(f"review leakage group mismatch for {current.item_id}")

        if recommendation.action == "reject":
            if recommendation.recommended_verification_status != "rejected":
                raise RetrievalEvalError(
                    f"reject decision has the wrong status for {current.item_id}"
                )
            if recommendation.proposed_query is not None:
                raise RetrievalEvalError(
                    f"reject decision unexpectedly contains a query for {current.item_id}"
                )
            if recommendation.proposed_query_type is not None:
                raise RetrievalEvalError(
                    f"reject decision unexpectedly contains a query type for {current.item_id}"
                )
            query = current.query
            query_type = current.query_type
            status: VerificationStatus = "rejected"
        elif recommendation.action == "accept":
            if recommendation.recommended_verification_status != "verified":
                raise RetrievalEvalError(
                    f"accept decision has the wrong status for {current.item_id}"
                )
            if (
                recommendation.proposed_query != current.query
                or recommendation.proposed_query_type != current.query_type
            ):
                raise RetrievalEvalError(f"accept decision changes content for {current.item_id}")
            query = current.query
            query_type = current.query_type
            status = "verified"
        else:
            if recommendation.recommended_verification_status != "verified":
                raise RetrievalEvalError(
                    f"edit decision has the wrong status for {current.item_id}"
                )
            if recommendation.proposed_query is None or recommendation.proposed_query_type is None:
                raise RetrievalEvalError(
                    f"edit decision is missing its proposed query for {current.item_id}"
                )
            query = recommendation.proposed_query
            query_type = recommendation.proposed_query_type
            status = "verified"

        updated_by_old_id[current.item_id] = _rebuild_reviewed_item(
            current,
            query=query,
            query_type=query_type,
            verification_status=status,
        )

    updated_items = [updated_by_old_id[item.item_id] for item in current_items]
    if len({item.item_id for item in updated_items}) != len(updated_items):
        raise RetrievalEvalError("reviewed decisions produced duplicate deterministic item IDs")
    validation = validate_items(updated_items, chunk_index, corpus)
    if not validation.valid:
        raise RetrievalEvalError(
            "reviewed A2.1 items failed validation: "
            + "; ".join(issue.code for issue in validation.issues[:8])
        )

    group_ids = _leakage_group_ids(updated_items, chunk_index, corpus)
    materialized = [
        derive_materialized_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in updated_items
    ]
    review_pack = [
        build_review_pack_item(
            item,
            chunk_index,
            corpus,
            leakage_group_id=group_ids[item.item_id],
        )
        for item in updated_items
    ]

    verified_items = [item for item in updated_items if item.verification_status == "verified"]
    rejected_items = [item for item in updated_items if item.verification_status == "rejected"]
    if len(verified_items) != 108 or len(rejected_items) != 54:
        raise RetrievalEvalError(
            "reviewed status counts do not produce 108 verified and 54 rejected items"
        )
    verified_subject_counts = Counter(item.subject for item in verified_items)
    if len(verified_subject_counts) != 9 or any(
        count != 12 for count in verified_subject_counts.values()
    ):
        raise RetrievalEvalError(
            "verified set must contain exactly 12 items for each of 9 subjects"
        )

    split_by_item = _assign_final_splits(
        verified_items,
        group_ids,
        dev_per_subject=8,
        test_per_subject=4,
    )
    group_splits: dict[str, set[FinalSplit]] = defaultdict(set)
    for item in verified_items:
        group_splits[group_ids[item.item_id]].add(split_by_item[item.item_id])
    split_leakage_groups = [
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    ]
    if split_leakage_groups:
        raise RetrievalEvalError("a leakage group was assigned to both dev and test")

    final_records = [
        FinalDatasetItem(
            split=split_by_item[item.item_id],
            leakage_group_id=group_ids[item.item_id],
            item=item,
        )
        for item in verified_items
    ]
    final_records.sort(key=lambda record: (record.item.subject, record.split, record.item.item_id))
    records_by_split = {
        split: [record for record in final_records if record.split == split]
        for split in ("dev", "test")
    }
    split_counts = {split: len(records) for split, records in records_by_split.items()}
    subject_split_counts = {
        subject: {
            "dev": sum(
                record.item.subject == subject and record.split == "dev" for record in final_records
            ),
            "test": sum(
                record.item.subject == subject and record.split == "test"
                for record in final_records
            ),
        }
        for subject in sorted(verified_subject_counts)
    }
    if split_counts != {"dev": 72, "test": 36} or any(
        counts != {"dev": 8, "test": 4} for counts in subject_split_counts.values()
    ):
        raise RetrievalEvalError("final dev/test counts do not match the required quotas")

    final_validation = validate_items(verified_items, chunk_index, corpus)
    if not final_validation.valid:
        raise RetrievalEvalError(
            "verified A2.1 items failed final validation: "
            + "; ".join(issue.code for issue in final_validation.issues[:8])
        )

    final_output_dir = final_output_dir or output_dir / "retrieval-eval-v1"
    materialized_by_new_id = {item.item_id: item for item in materialized}
    final_materialized = [materialized_by_new_id[record.item.item_id] for record in final_records]
    final_manifest: dict[str, object] = {
        "dataset_version": "retrieval-eval-v1",
        "schema_version": RETRIEVAL_EVAL_SCHEMA_VERSION,
        "source_recommendations": recommendations_path.name,
        "candidate_count": len(updated_items),
        "verified_count": len(verified_items),
        "rejected_count": len(rejected_items),
        "action_counts": dict(sorted(action_counts.items())),
        "split_counts": split_counts,
        "subject_counts": dict(sorted(verified_subject_counts.items())),
        "subject_split_counts": subject_split_counts,
        "query_type_counts": dict(
            sorted(Counter(item.query_type for item in verified_items).items())
        ),
        "query_source_count": sum(bool(item.query_source) for item in verified_items),
        "leakage": {
            "query_in_gold_evidence_count": final_validation.query_in_gold_evidence_count,
            "query_in_gold_chunk_count": final_validation.query_in_gold_chunk_count,
            "evidence_fingerprint_drift_count": sum(
                issue.code == "evidence_fingerprint_drift" for issue in final_validation.issues
            ),
            "evidence_not_covered_count": sum(
                issue.code == "evidence_not_covered" for issue in final_validation.issues
            ),
            "duplicate_normalized_query_count": sum(
                issue.code == "duplicate_normalized_query" for issue in final_validation.issues
            ),
            "repeated_evidence_group_count": len(final_validation.repeated_evidence_groups),
            "repeated_relevant_chunk_group_count": len(
                final_validation.repeated_relevant_chunk_groups
            ),
            "split_leakage_group_count": len(split_leakage_groups),
        },
        "all_gold_blocks_covered": all(item.all_gold_blocks_covered for item in final_materialized),
        "runtime_files": [
            "dev.jsonl",
            "test.jsonl",
            "materialized.jsonl",
            "manifest.json",
        ],
        "repository_safe_file": str(repository_safe_path),
    }

    _write_jsonl(output_dir / "candidates.jsonl", updated_items)
    _write_jsonl(output_dir / "materialized.jsonl", materialized)
    _write_jsonl(output_dir / "review-pack.jsonl", review_pack)
    _write_jsonl(final_output_dir / "dev.jsonl", records_by_split["dev"])
    _write_jsonl(final_output_dir / "test.jsonl", records_by_split["test"])
    _write_jsonl(final_output_dir / "materialized.jsonl", final_materialized)
    _write_json(final_output_dir / "manifest.json", final_manifest)
    _write_jsonl(repository_safe_path, final_records)

    runtime_manifest = _read_json(output_dir / "manifest.json")
    runtime_manifest.update(
        {
            "candidate_count": len(updated_items),
            "status_counts": _status_counts(updated_items),
            "query_source_count": sum(bool(item.query_source) for item in updated_items),
            "subject_counts": dict(sorted(Counter(item.subject for item in updated_items).items())),
            "query_type_counts": dict(
                sorted(Counter(item.query_type for item in updated_items).items())
            ),
            "all_gold_blocks_covered": all(item.all_gold_blocks_covered for item in materialized),
            "validation": validation.model_dump(mode="json"),
            "leakage": {
                "query_in_gold_evidence_count": validation.query_in_gold_evidence_count,
                "query_in_gold_chunk_count": validation.query_in_gold_chunk_count,
                "repeated_evidence_groups": [
                    group.model_dump(mode="json") for group in validation.repeated_evidence_groups
                ],
                "repeated_relevant_chunk_groups": [
                    group.model_dump(mode="json")
                    for group in validation.repeated_relevant_chunk_groups
                ],
                "leakage_group_count": len(set(group_ids.values())),
            },
            "review": {
                "recommendation_file": recommendations_path.name,
                "action_counts": dict(sorted(action_counts.items())),
            },
            "final_dataset": final_manifest,
        }
    )
    runtime_manifest["files"] = sorted(
        set(runtime_manifest.get("files", []))
        | {
            "retrieval-eval-v1/dev.jsonl",
            "retrieval-eval-v1/test.jsonl",
            "retrieval-eval-v1/materialized.jsonl",
            "retrieval-eval-v1/manifest.json",
        }
    )
    _write_json(output_dir / "manifest.json", runtime_manifest)
    _write_json(repository_manifest_path, runtime_manifest)
    return final_manifest


def _update_jsonl_record(
    path: Path,
    item_id: str,
    updated: BaseModel,
    *,
    old_item_id: str | None = None,
) -> None:
    records = _read_jsonl(path)
    target_id = old_item_id or item_id
    replaced = False
    serialized = updated.model_dump(mode="json")
    for index, record in enumerate(records):
        if record.get("item_id") == target_id:
            records[index] = serialized
            replaced = True
            break
    if not replaced:
        raise RetrievalEvalError(f"item {target_id} is missing from {path.name}")
    _atomic_write(
        path,
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records
        ),
    )


def apply_review_action(
    output_dir: Path,
    item_id: str,
    action: ReviewAction,
    *,
    query: str | None = None,
    query_type: QueryType | None = None,
) -> RetrievalEvalItem:
    """Apply one local review decision without touching canonical or chunk data."""
    records = _read_jsonl(output_dir / "candidates.jsonl")
    try:
        current = next(
            RetrievalEvalItem.model_validate(record)
            for record in records
            if record.get("item_id") == item_id
        )
    except StopIteration as error:
        raise RetrievalEvalError(f"evaluation item does not exist: {item_id}") from error
    if action == "edit":
        if query is None or not query.strip():
            raise RetrievalEvalError("--query is required for the edit action")
        updated = _rebuild_reviewed_item(
            current,
            query=query,
            query_type=query_type or current.query_type,
            verification_status="candidate",
        )
    else:
        payload = current.model_dump()
        payload["verification_status"] = "verified" if action == "accept" else "rejected"
        updated = RetrievalEvalItem(**payload)
    _update_jsonl_record(
        output_dir / "candidates.jsonl",
        updated.item_id,
        updated,
        old_item_id=current.item_id,
    )
    materialized_records = _read_jsonl(output_dir / "materialized.jsonl")
    for record in materialized_records:
        if record.get("item_id") == current.item_id:
            record["item_id"] = updated.item_id
            record["verification_status"] = updated.verification_status
    _atomic_write(
        output_dir / "materialized.jsonl",
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
            for record in materialized_records
        ),
    )
    review_records = _read_jsonl(output_dir / "review-pack.jsonl")
    for record in review_records:
        if record.get("item_id") == current.item_id:
            record["item_id"] = updated.item_id
            record["query"] = updated.query
            record["query_type"] = updated.query_type
            record["verification_status"] = updated.verification_status
    _atomic_write(
        output_dir / "review-pack.jsonl",
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
            for record in review_records
        ),
    )
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    candidate_records = [
        RetrievalEvalItem.model_validate(record)
        for record in _read_jsonl(output_dir / "candidates.jsonl")
    ]
    manifest["status_counts"] = _status_counts(candidate_records)
    _write_json(manifest_path, manifest)
    return updated


__all__ = [
    "QUERY_TYPES",
    "RETRIEVAL_EVAL_SCHEMA_VERSION",
    "CandidateGenerationStats",
    "CorpusDocument",
    "ExclusionRecord",
    "FinalDatasetItem",
    "IndexedChunk",
    "LeakageGroup",
    "MaterializedEvalItem",
    "RetrievalEvalError",
    "RetrievalEvalItem",
    "RetrievalEvidence",
    "ReviewPackItem",
    "ReviewQuerySource",
    "ReviewRecommendation",
    "SourceBlockReference",
    "ValidationReport",
    "apply_review_action",
    "build_review_pack_item",
    "derive_materialized_item",
    "deterministic_item_id",
    "evidence_fingerprint",
    "finalize_retrieval_eval_dataset",
    "generate_candidate_items",
    "load_a15_missing_table_exclusions",
    "load_canonical_corpus",
    "make_eval_item",
    "materialize_retrieval_eval_set",
    "normalize_query",
    "regenerate_candidate_review_pack",
    "validate_items",
    "validate_runtime_evaluation",
]
