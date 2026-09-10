"""Explicit registration of an existing parsed corpus as external documents."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.documents.models import (
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_SOURCE_REF_MAX_LENGTH,
    DOCUMENT_STATUS_REGISTERED,
    DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
    Document,
)
from knowledge_scope.documents.storage import normalize_original_filename
from knowledge_scope.evaluation.embedding_benchmark import (
    EmbeddingBenchmarkError,
    load_frozen_chunk_index,
)
from knowledge_scope.evaluation.retrieval_eval import FinalDatasetItem
from knowledge_scope.knowledge_bases.models import KnowledgeBase
from knowledge_scope.parsing.models import CanonicalDocument

REGISTRATION_SCHEMA_VERSION = "1.0"
EXPECTED_BENCHMARK_DOCUMENT_COUNT = 255
EXPECTED_BENCHMARK_CHUNK_COUNT = 7_524
DEFAULT_CORPUS_MANIFEST = Path("data/benchmarks/a1-5/corpus-manifest.jsonl")
DEFAULT_CANONICAL_ROOT = Path("data/benchmarks/a1-5/canonical")
DEFAULT_CHUNK_INDEX = Path("data/evaluation/a2-1/chunk_index.jsonl")
DEFAULT_EVAL_DATASET = Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl")
DEFAULT_EVAL_KB_MAPPING = Path("data/evaluation/a3-5/a2-1-kb-mapping.jsonl")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CorpusRegistrationError(RuntimeError):
    """Raised when an external corpus cannot be registered safely."""


@dataclass(frozen=True, slots=True)
class CorpusRegistrationDocument:
    """Safe metadata required for one external reference-backed document."""

    document_id: UUID
    original_filename: str
    size_bytes: int
    sha256: str
    source_ref: str


@dataclass(frozen=True, slots=True)
class CorpusRegistrationSpec:
    """Validated, side-effect-free corpus registration input."""

    target_knowledge_base_id: UUID
    documents: tuple[CorpusRegistrationDocument, ...]
    manifest_rows: int
    duplicate_manifest_rows: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class CorpusRegistrationPlan:
    """Database preflight result produced before any insert is attempted."""

    documents_to_insert: tuple[CorpusRegistrationDocument, ...]
    already_valid_count: int
    conflicts: tuple[str, ...]

    @property
    def can_apply(self) -> bool:
        return not self.conflicts


@dataclass(frozen=True, slots=True)
class CorpusRegistrationResult:
    """Counts from one committed or idempotent registration run."""

    knowledge_base_id: UUID
    documents_attempted: int
    documents_inserted: int
    documents_already_valid: int
    conflicts: int
    failures: int


@dataclass(frozen=True, slots=True)
class EvaluationKnowledgeBaseAssignment:
    """A runtime-only attribution for one frozen A2.1 evaluation item."""

    item_id: str
    split: str
    subject: str
    knowledge_base_id: UUID
    evidence_document_count: int


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusRegistrationError("JSONL input is not readable") from error

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusRegistrationError(
                f"corpus manifest contains invalid JSON on line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise CorpusRegistrationError(f"corpus manifest line {line_number} is not an object")
        records.append(value)
    return records


def build_frozen_eval_kb_mapping(
    dataset_path: Path,
    *,
    registered_document_ids: Collection[UUID],
    knowledge_base_id: UUID,
) -> tuple[EvaluationKnowledgeBaseAssignment, ...]:
    """Validate and derive A2.1 item attribution without changing its labels."""
    records = [FinalDatasetItem.model_validate(record) for record in _jsonl_records(dataset_path)]
    if len(records) != 108:
        raise CorpusRegistrationError(
            f"frozen A2.1 dataset must contain 108 items, found {len(records)}"
        )
    item_ids = [record.item.item_id for record in records]
    if len(set(item_ids)) != len(item_ids):
        raise CorpusRegistrationError("frozen A2.1 dataset contains duplicate item IDs")
    if Counter(record.split for record in records) != {"dev": 72, "test": 36}:
        raise CorpusRegistrationError("frozen A2.1 dataset does not have a 72/36 split")
    subject_splits = Counter((record.item.subject, record.split) for record in records)
    subjects = {record.item.subject for record in records}
    if len(subjects) != 9 or any(
        subject_splits[(subject, split)] != expected
        for subject in subjects
        for split, expected in (("dev", 8), ("test", 4))
    ):
        raise CorpusRegistrationError("frozen A2.1 dataset subject quotas are invalid")
    registered_ids = set(registered_document_ids)
    assignments: list[EvaluationKnowledgeBaseAssignment] = []
    for record in records:
        evidence_document_ids = {location.document_id for location in record.item.evidence}
        missing_ids = evidence_document_ids - registered_ids
        if missing_ids:
            raise CorpusRegistrationError(
                f"A2.1 item {record.item.item_id} references an unregistered document"
            )
        assignments.append(
            EvaluationKnowledgeBaseAssignment(
                item_id=record.item.item_id,
                split=record.split,
                subject=record.item.subject,
                knowledge_base_id=knowledge_base_id,
                evidence_document_count=len(evidence_document_ids),
            )
        )
    return tuple(assignments)


def write_frozen_eval_kb_mapping(
    path: Path,
    assignments: Sequence[EvaluationKnowledgeBaseAssignment],
) -> None:
    """Write only bounded A2.1 item-to-KB metadata to an ignored runtime path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(
            {
                "schema_version": REGISTRATION_SCHEMA_VERSION,
                "item_id": assignment.item_id,
                "split": assignment.split,
                "subject": assignment.subject,
                "knowledge_base_id": str(assignment.knowledge_base_id),
                "evidence_document_count": assignment.evidence_document_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for assignment in assignments
    )
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise CorpusRegistrationError("A2.1 Knowledge Base mapping could not be written") from error


def _safe_source_ref(relative_path: str) -> str:
    """Turn a manifest-relative source path into a non-filesystem reference."""
    normalized = unicodedata.normalize("NFC", relative_path).replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(relative_path)
    if (
        not normalized
        or path.is_absolute()
        or windows_path.drive
        or windows_path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CorpusRegistrationError("corpus source reference must be a safe relative path")
    source_ref = f"benchmark/a1-5/{path.as_posix()}"
    if len(source_ref) > DOCUMENT_SOURCE_REF_MAX_LENGTH:
        raise CorpusRegistrationError("corpus source reference is too long")
    return source_ref


def _manifest_document(record: dict[str, Any]) -> CorpusRegistrationDocument:
    try:
        document_id = UUID(str(record["benchmark_document_uuid"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusRegistrationError("corpus manifest has an invalid document UUID") from error

    basename = record.get("basename")
    benchmark_item_id = record.get("benchmark_item_id")
    relative_path = record.get("relative_path")
    sha256 = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if not all(
        isinstance(value, str) and value
        for value in (basename, benchmark_item_id, relative_path, sha256)
    ):
        raise CorpusRegistrationError("corpus manifest has incomplete document metadata")
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise CorpusRegistrationError("corpus manifest contains an invalid SHA-256")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise CorpusRegistrationError("corpus manifest contains an invalid document size")

    try:
        filename = normalize_original_filename(basename)
    except ValueError as error:
        raise CorpusRegistrationError("corpus manifest contains an invalid PDF filename") from error
    return CorpusRegistrationDocument(
        document_id=document_id,
        original_filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        source_ref=_safe_source_ref(relative_path),
    )


def _canonical_path(canonical_root: Path, item_id: str, document_id: UUID) -> Path:
    root = canonical_root.resolve()
    direct = (root / f"{item_id}.json").resolve()
    if root in direct.parents and direct.is_file():
        return direct

    for candidate in sorted(root.glob("*.json")):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("document_id") == str(document_id):
            return candidate.resolve()
    raise CorpusRegistrationError("a canonical artifact is missing for a manifest document")


def _canonical_block_pages(document: CanonicalDocument) -> dict[str, int]:
    return {block.block_id: page.page_number for page in document.pages for block in page.blocks}


def _validate_chunk_index(
    path: Path,
    documents: Sequence[tuple[CorpusRegistrationDocument, CanonicalDocument]],
    *,
    expected_chunk_count: int | None,
) -> int:
    try:
        chunks = load_frozen_chunk_index(path)
    except (EmbeddingBenchmarkError, ValidationError) as error:
        raise CorpusRegistrationError("A1.6 chunk index is missing or invalid") from error

    expected_ids = {document.document_id for document, _ in documents}
    chunk_document_ids = {chunk.document_id for chunk in chunks.values()}
    if chunk_document_ids != expected_ids:
        raise CorpusRegistrationError("A1.6 chunk index document IDs do not match the corpus")
    if expected_chunk_count is not None and len(chunks) != expected_chunk_count:
        raise CorpusRegistrationError(
            f"A1.6 chunk index must contain {expected_chunk_count} chunks"
        )

    block_pages = {
        document.document_id: _canonical_block_pages(canonical) for document, canonical in documents
    }
    for chunk in chunks.values():
        pages = block_pages[chunk.document_id]
        for block_id in chunk.source_block_ids:
            page_number = pages.get(block_id)
            if page_number is None or not chunk.page_start <= page_number <= chunk.page_end:
                raise CorpusRegistrationError(
                    "A1.6 chunk lineage does not match canonical evidence"
                )
    return len(chunks)


def build_registration_spec(
    target_knowledge_base_id: UUID,
    *,
    corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
    chunk_index: Path = DEFAULT_CHUNK_INDEX,
    expected_document_count: int | None = EXPECTED_BENCHMARK_DOCUMENT_COUNT,
    expected_chunk_count: int | None = EXPECTED_BENCHMARK_CHUNK_COUNT,
) -> CorpusRegistrationSpec:
    """Validate all corpus artifacts without touching PostgreSQL."""
    if not canonical_root.is_dir():
        raise CorpusRegistrationError("canonical artifact root does not exist")
    records = [
        record
        for record in _jsonl_records(corpus_manifest)
        if record.get("inventory_status") == "ready"
    ]
    if not records:
        raise CorpusRegistrationError("corpus manifest contains no ready documents")

    by_id: dict[UUID, CorpusRegistrationDocument] = {}
    duplicate_rows = 0
    selected_manifest_items: dict[UUID, list[tuple[str, CorpusRegistrationDocument]]] = {}
    canonical_documents: list[tuple[CorpusRegistrationDocument, CanonicalDocument]] = []
    for record in records:
        document = _manifest_document(record)
        existing = by_id.get(document.document_id)
        if existing is not None:
            if (existing.size_bytes, existing.sha256) != (document.size_bytes, document.sha256):
                raise CorpusRegistrationError("duplicate document IDs have conflicting metadata")
            duplicate_rows += 1
        else:
            by_id[document.document_id] = document
        selected_manifest_items.setdefault(document.document_id, []).append(
            (str(record["benchmark_item_id"]), document)
        )

    documents = tuple(
        sorted(
            (
                min(items, key=lambda item: (item[1].source_ref, item[0]))[1]
                for items in selected_manifest_items.values()
            ),
            key=lambda value: str(value.document_id),
        )
    )
    if expected_document_count is not None and len(documents) != expected_document_count:
        raise CorpusRegistrationError(
            f"corpus must contain {expected_document_count} unique documents"
        )
    if len({document.sha256 for document in documents}) != len(documents):
        raise CorpusRegistrationError(
            "distinct corpus documents must not share a SHA-256 in one Knowledge Base"
        )

    for document in documents:
        item_ids = [item_id for item_id, candidate in selected_manifest_items[document.document_id]]
        path = _canonical_path(canonical_root, item_ids[0], document.document_id)
        try:
            canonical = CanonicalDocument.model_validate_json(path.read_bytes())
        except (OSError, UnicodeDecodeError, ValidationError) as error:
            raise CorpusRegistrationError("a canonical artifact is invalid") from error
        if canonical.document_id != document.document_id:
            raise CorpusRegistrationError("canonical document ID does not match the manifest")
        canonical_documents.append((document, canonical))

    chunk_count = _validate_chunk_index(
        chunk_index,
        canonical_documents,
        expected_chunk_count=expected_chunk_count,
    )
    return CorpusRegistrationSpec(
        target_knowledge_base_id=target_knowledge_base_id,
        documents=documents,
        manifest_rows=len(records),
        duplicate_manifest_rows=duplicate_rows,
        chunk_count=chunk_count,
    )


def _document_matches(
    existing: Document,
    expected: CorpusRegistrationDocument,
    target_knowledge_base_id: UUID,
) -> bool:
    return (
        existing.knowledge_base_id == target_knowledge_base_id
        and existing.original_filename == expected.original_filename
        and existing.media_type == DOCUMENT_MEDIA_TYPE_PDF
        and existing.size_bytes == expected.size_bytes
        and existing.sha256 == expected.sha256
        and existing.status == DOCUMENT_STATUS_REGISTERED
        and existing.storage_kind == DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE
        and existing.storage_key is None
        and existing.source_ref == expected.source_ref
    )


async def plan_database_registration(
    session: AsyncSession,
    spec: CorpusRegistrationSpec,
) -> CorpusRegistrationPlan:
    """Check ownership and idempotency before any document insert."""
    target = await session.scalar(
        select(KnowledgeBase).where(KnowledgeBase.id == spec.target_knowledge_base_id)
    )
    if target is None:
        raise CorpusRegistrationError("target Knowledge Base does not exist")

    document_ids = tuple(document.document_id for document in spec.documents)
    hashes = tuple(document.sha256 for document in spec.documents)
    existing_documents = list(
        (
            await session.scalars(
                select(Document).where(
                    (Document.id.in_(document_ids)) | (Document.sha256.in_(hashes))
                )
            )
        ).all()
    )
    by_id = {document.id: document for document in existing_documents}
    by_hash: dict[str, list[Document]] = {}
    for document in existing_documents:
        by_hash.setdefault(document.sha256, []).append(document)

    to_insert: list[CorpusRegistrationDocument] = []
    already_valid = 0
    conflicts: list[str] = []
    for expected in spec.documents:
        existing = by_id.get(expected.document_id)
        if existing is not None:
            if _document_matches(existing, expected, spec.target_knowledge_base_id):
                already_valid += 1
            else:
                conflicts.append(
                    f"document {expected.document_id} has conflicting ownership/metadata"
                )
            continue

        same_hash_in_target = [
            document
            for document in by_hash.get(expected.sha256, [])
            if document.knowledge_base_id == spec.target_knowledge_base_id
        ]
        if same_hash_in_target:
            conflicts.append(
                f"SHA-256 already exists in target Knowledge Base for {expected.document_id}"
            )
            continue
        to_insert.append(expected)

    return CorpusRegistrationPlan(
        documents_to_insert=tuple(to_insert),
        already_valid_count=already_valid,
        conflicts=tuple(conflicts),
    )


async def register_corpus(
    session: AsyncSession,
    spec: CorpusRegistrationSpec,
) -> CorpusRegistrationResult:
    """Register the complete spec atomically and preserve external-source semantics."""
    try:
        async with session.begin():
            plan = await plan_database_registration(session, spec)
            if not plan.can_apply:
                raise CorpusRegistrationError("corpus registration preflight found conflicts")

            for expected in plan.documents_to_insert:
                session.add(
                    Document(
                        id=expected.document_id,
                        knowledge_base_id=spec.target_knowledge_base_id,
                        original_filename=expected.original_filename,
                        storage_key=None,
                        storage_kind=DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
                        source_ref=expected.source_ref,
                        media_type=DOCUMENT_MEDIA_TYPE_PDF,
                        size_bytes=expected.size_bytes,
                        sha256=expected.sha256,
                        status=DOCUMENT_STATUS_REGISTERED,
                    )
                )
            await session.flush()
    except CorpusRegistrationError:
        raise
    except (IntegrityError, SQLAlchemyError) as error:
        raise CorpusRegistrationError(
            "corpus registration failed; the database transaction was rolled back"
        ) from error

    return CorpusRegistrationResult(
        knowledge_base_id=spec.target_knowledge_base_id,
        documents_attempted=len(spec.documents),
        documents_inserted=len(plan.documents_to_insert),
        documents_already_valid=plan.already_valid_count,
        conflicts=0,
        failures=0,
    )


__all__ = [
    "DEFAULT_CANONICAL_ROOT",
    "DEFAULT_CHUNK_INDEX",
    "DEFAULT_CORPUS_MANIFEST",
    "DEFAULT_EVAL_DATASET",
    "DEFAULT_EVAL_KB_MAPPING",
    "EXPECTED_BENCHMARK_CHUNK_COUNT",
    "EXPECTED_BENCHMARK_DOCUMENT_COUNT",
    "CorpusRegistrationDocument",
    "CorpusRegistrationError",
    "CorpusRegistrationPlan",
    "CorpusRegistrationResult",
    "CorpusRegistrationSpec",
    "EvaluationKnowledgeBaseAssignment",
    "build_frozen_eval_kb_mapping",
    "build_registration_spec",
    "plan_database_registration",
    "register_corpus",
    "write_frozen_eval_kb_mapping",
]
