"""Developer-only document parsing orchestration and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from knowledge_scope.documents.models import DOCUMENT_STORAGE_KIND_MANAGED, Document
from knowledge_scope.documents.storage import (
    StorageError,
    TrashedResource,
    filesystem_path_for_storage_key,
    move_to_trash,
    permanently_remove_trash,
    restore_from_trash,
)
from knowledge_scope.evidence.lifecycle import (
    EvidenceArtifactError,
    move_evidence_artifact_to_trash,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .mineru_adapter import (
    AdaptedCanonicalDocument,
    AdapterStats,
    MineruAdapterError,
    adapt_content_list,
    infer_page_count,
)
from .mineru_runner import (
    MineruRunnerError,
    MineruRunResult,
    find_content_list,
    run_mineru,
)
from .models import CANONICAL_SCHEMA_VERSION

if TYPE_CHECKING:
    from knowledge_scope.retrieval.representation_index import (
        QdrantRepresentationStore,
        RepresentationEmbeddingEncoder,
    )

PARSING_DIRECTORY_NAME = "parsing"
CHUNKING_DIRECTORY_NAME = "chunking"
MAX_MANIFEST_WARNING_COUNT = 100
MAX_MANIFEST_WARNING_LENGTH = 512


class DocumentParseError(RuntimeError):
    """Raised when a document cannot be parsed or its artifacts cannot be saved."""


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Public CLI facts from one successful document parse."""

    document_id: UUID
    source_sha256: str
    parser_version: str
    backend: str
    elapsed_seconds: float
    canonical_ref: str
    raw_ref: str
    stats: AdapterStats


def _sha256_file(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomically(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _promote_staging(staging_dir: Path, final_dir: Path) -> Path | None:
    """Atomically promote a complete staging tree while retaining old output on failure."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir: Path | None = None
    try:
        if final_dir.exists():
            backup_dir = final_dir.parent / f".{final_dir.name}.previous-{uuid4().hex}"
            os.replace(final_dir, backup_dir)
        os.replace(staging_dir, final_dir)
    except OSError as error:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            try:
                os.replace(backup_dir, final_dir)
            except OSError as restore_error:
                raise DocumentParseError(
                    "parsed artifacts could not be promoted or previous output restored"
                ) from restore_error
        raise DocumentParseError("parsed artifacts could not be promoted") from error
    else:
        return backup_dir


def _restore_promoted_artifacts(final_dir: Path, backup_dir: Path | None) -> None:
    """Remove a new parse result and restore the previous one after invalidation failure."""
    try:
        if final_dir.exists() or final_dir.is_symlink():
            shutil.rmtree(final_dir)
        if backup_dir is not None:
            os.replace(backup_dir, final_dir)
    except OSError as error:
        raise DocumentParseError(
            "parsed artifacts could not be invalidated or previous output restored"
        ) from error


def _invalidate_chunking_artifacts(
    document_id: UUID,
    settings: Settings,
) -> TrashedResource | None:
    """Quarantine derived chunks until the new canonical state is committed."""
    chunking_path = Path(settings.data_dir).resolve() / CHUNKING_DIRECTORY_NAME / str(document_id)
    if not chunking_path.exists() and not chunking_path.is_symlink():
        return None
    if chunking_path.is_symlink() or not chunking_path.is_dir():
        raise DocumentParseError("chunking artifact directory is invalid")
    try:
        return move_to_trash(chunking_path, settings.data_dir)
    except (OSError, StorageError) as error:
        raise DocumentParseError("chunk artifacts could not be invalidated") from error


def _invalidate_evidence_artifacts(
    document_id: UUID,
    settings: Settings,
) -> TrashedResource | None:
    """Quarantine derived representations until the new canonical state is committed."""
    try:
        return move_evidence_artifact_to_trash(settings.data_dir, document_id)
    except EvidenceArtifactError as error:
        raise DocumentParseError("evidence artifacts could not be invalidated") from error


def _restore_reparse_derived_artifacts(
    resources: tuple[TrashedResource | None, ...],
) -> None:
    """Restore every derived artifact quarantined during a failed reparse."""
    restore_error: BaseException | None = None
    for resource in reversed(resources):
        if resource is None:
            continue
        try:
            restore_from_trash(resource)
        except (OSError, StorageError) as error:
            restore_error = restore_error or error
    if restore_error is not None:
        raise DocumentParseError(
            "derived artifacts could not be restored after reparse failure"
        ) from restore_error


def _discard_reparse_derived_artifacts(
    resources: tuple[TrashedResource | None, ...],
) -> None:
    """Best-effort cleanup of old derived artifacts after a successful reparse."""
    for resource in resources:
        if resource is None:
            continue
        try:
            permanently_remove_trash(resource)
        except (OSError, StorageError):
            continue


def _commit_representation_generation(
    *,
    staging_dir: Path,
    final_dir: Path,
    document_id: UUID,
    adapted: AdaptedCanonicalDocument,
    settings: Settings,
    knowledge_base_id: UUID,
    representation_store: QdrantRepresentationStore,
    representation_embedder: RepresentationEmbeddingEncoder,
) -> None:
    """Commit a parsed generation with A4.2 compensation across all artifacts.

    Qdrant cannot participate in a filesystem transaction.  The old canonical,
    derived artifacts, and active points therefore remain the recovery target
    until the new representation replacement succeeds.  Any later failure
    compensates every state that was changed; unresolved compensation fails
    closed instead of claiming an atomic distributed commit.
    """

    from knowledge_scope.evidence.lifecycle import (
        EvidenceArtifactError,
        evidence_artifact_path,
        load_evidence_artifact,
        remove_evidence_artifact,
        write_evidence_artifact,
    )
    from knowledge_scope.evidence.service import (
        EvidenceValidationError,
        validate_evidence_document,
    )
    from knowledge_scope.retrieval.representation_index import (
        RepresentationIndexError,
        build_indexable_evidence,
        build_representation_points,
        load_canonical_document,
    )

    try:
        previous_points = representation_store.snapshot_document(
            document_id,
            knowledge_base_id=knowledge_base_id,
        )
        previous_path = evidence_artifact_path(settings.data_dir, document_id)
        previous_artifact = None
        if previous_path.exists() or previous_path.is_symlink():
            previous_artifact = load_evidence_artifact(settings.data_dir, document_id)
            previous_canonical_path = final_dir / "canonical.json"
            if not previous_canonical_path.is_file():
                raise DocumentParseError(
                    "existing representation evidence has no current canonical artifact"
                )
            previous_canonical = load_canonical_document(previous_canonical_path)
            validate_evidence_document(
                previous_canonical,
                previous_artifact,
                knowledge_base_id,
            )
        elif previous_points:
            raise DocumentParseError(
                "existing representation points have no matching evidence artifact"
            )
    except (EvidenceArtifactError, EvidenceValidationError, RepresentationIndexError) as error:
        raise DocumentParseError(
            "the previous representation generation is missing or inconsistent"
        ) from error

    try:
        artifact = build_indexable_evidence(adapted.document, knowledge_base_id)
        points = build_representation_points(
            artifact,
            settings=settings,
            embedder=representation_embedder,
        )
    except RepresentationIndexError as error:
        raise DocumentParseError(str(error)) from error

    previous_dir: Path | None = None
    trashed_chunking: TrashedResource | None = None
    qdrant_replaced = False
    evidence_write_attempted = False
    promoted = False
    try:
        representation_store.replace_document(document_id, knowledge_base_id, points)
        qdrant_replaced = True
        evidence_write_attempted = True
        write_evidence_artifact(settings.data_dir, artifact)
        previous_dir = _promote_staging(staging_dir, final_dir)
        promoted = True
        trashed_chunking = _invalidate_chunking_artifacts(document_id, settings)
    except Exception as error:
        rollback_errors: list[BaseException] = []
        if qdrant_replaced:
            try:
                representation_store.replace_document(
                    document_id,
                    knowledge_base_id,
                    previous_points,
                )
            except Exception as restore_error:
                rollback_errors.append(restore_error)
        if evidence_write_attempted:
            try:
                if previous_artifact is None:
                    remove_evidence_artifact(settings.data_dir, document_id)
                else:
                    write_evidence_artifact(settings.data_dir, previous_artifact)
            except Exception as restore_error:
                rollback_errors.append(restore_error)
        if promoted:
            try:
                _restore_promoted_artifacts(final_dir, previous_dir)
            except DocumentParseError as restore_error:
                rollback_errors.append(restore_error)
        try:
            _restore_reparse_derived_artifacts((trashed_chunking,))
        except DocumentParseError as restore_error:
            rollback_errors.append(restore_error)
        if rollback_errors:
            raise DocumentParseError(
                "representation generation failed and previous generation rollback also failed"
            ) from rollback_errors[0]
        if isinstance(error, DocumentParseError):
            raise
        if isinstance(
            error,
            (EvidenceArtifactError, EvidenceValidationError, RepresentationIndexError),
        ):
            raise DocumentParseError(str(error)) from error
        raise DocumentParseError("representation generation could not be committed") from error
    else:
        _discard_reparse_derived_artifacts((trashed_chunking,))
        if previous_dir is not None:
            shutil.rmtree(previous_dir, ignore_errors=True)


def _artifact_root(settings: Settings) -> Path:
    root = (Path(settings.data_dir) / PARSING_DIRECTORY_NAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest(
    *,
    document_id: UUID,
    source_sha256: str,
    run_result: MineruRunResult,
    stats: AdapterStats,
) -> dict[str, object]:
    document_ref = str(document_id)
    parse_stats = {
        "elapsed_seconds": run_result.elapsed_seconds,
        "pages": stats.pages,
        "mineru_input_items": stats.input_items,
        "canonical_blocks": stats.canonical_blocks,
        "title_blocks": stats.title_blocks,
        "text_blocks": stats.text_blocks,
        "tables": stats.tables,
        "formulas": stats.formulas,
        "images": stats.images,
        "skipped_auxiliary": stats.skipped_auxiliary,
        "unsupported_items": stats.unsupported_items,
        "bbox_clamped": stats.bbox_clamped,
        "table_asset_only": stats.table_asset_only,
        "table_missing_content": stats.table_missing_content,
        "warning_count": len(stats.warnings),
        "warnings": [
            warning[:MAX_MANIFEST_WARNING_LENGTH]
            for warning in stats.warnings[:MAX_MANIFEST_WARNING_COUNT]
        ],
    }
    return {
        "document_id": document_ref,
        "source_sha256": source_sha256,
        "parser": "mineru",
        "parser_version": run_result.version,
        "backend": run_result.backend,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "parsed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "canonical_ref": f"parsing/{document_ref}/canonical.json",
        "raw_ref": f"parsing/{document_ref}/mineru",
        "parse_stats": parse_stats,
    }


def parse_document_file(
    document_id: UUID,
    source_path: Path,
    expected_sha256: str,
    settings: Settings,
    *,
    representation_store: QdrantRepresentationStore | None = None,
    representation_embedder: RepresentationEmbeddingEncoder | None = None,
    representation_knowledge_base_id: UUID | None = None,
) -> ParseResult:
    """Parse one verified application-owned PDF and atomically persist its artifacts."""
    representation_args = (
        representation_store,
        representation_embedder,
        representation_knowledge_base_id,
    )
    if any(argument is not None for argument in representation_args) and not all(
        argument is not None for argument in representation_args
    ):
        raise DocumentParseError(
            "representation generation requires a store, embedder, and knowledge base"
        )
    source = Path(source_path)
    if not source.is_file():
        raise DocumentParseError("the stored source PDF does not exist")
    try:
        source_sha256 = _sha256_file(source)
    except OSError as error:
        raise DocumentParseError("the stored source PDF could not be read") from error
    if source_sha256 != expected_sha256:
        raise DocumentParseError("the stored source PDF SHA-256 does not match its metadata")
    try:
        root = _artifact_root(settings)
        final_dir = root / str(document_id)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{document_id}-", dir=root))
    except OSError as error:
        raise DocumentParseError("the parsing artifact directory could not be created") from error
    mineru_output_dir = staging_dir / "mineru"

    try:
        mineru_output_dir.mkdir()
        run_result = run_mineru(
            source,
            mineru_output_dir,
            settings.mineru_command,
            timeout_seconds=settings.mineru_timeout_seconds,
        )
        (mineru_output_dir / "stdout.log").write_text(run_result.stdout, encoding="utf-8")
        (mineru_output_dir / "stderr.log").write_text(run_result.stderr, encoding="utf-8")

        content_list_path = find_content_list(mineru_output_dir)
        adapted: AdaptedCanonicalDocument = adapt_content_list(
            content_list_path,
            document_id,
            mineru_output_dir,
            page_count=infer_page_count(mineru_output_dir),
        )
        _write_atomically(
            staging_dir / "canonical.json",
            adapted.document.model_dump_json(indent=2) + "\n",
        )
        _write_atomically(
            staging_dir / "manifest.json",
            json.dumps(
                _manifest(
                    document_id=document_id,
                    source_sha256=source_sha256,
                    run_result=run_result,
                    stats=adapted.stats,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        if representation_store is None:
            previous_dir = _promote_staging(staging_dir, final_dir)
            trashed_chunking: TrashedResource | None = None
            trashed_evidence: TrashedResource | None = None
            try:
                trashed_chunking = _invalidate_chunking_artifacts(document_id, settings)
                trashed_evidence = _invalidate_evidence_artifacts(document_id, settings)
            except DocumentParseError as error:
                rollback_error: DocumentParseError | None = None
                try:
                    _restore_promoted_artifacts(final_dir, previous_dir)
                except DocumentParseError as restore_error:
                    rollback_error = restore_error
                try:
                    _restore_reparse_derived_artifacts((trashed_chunking, trashed_evidence))
                except DocumentParseError as restore_error:
                    rollback_error = rollback_error or restore_error
                if rollback_error is not None:
                    raise rollback_error from error
                raise
            _discard_reparse_derived_artifacts((trashed_evidence, trashed_chunking))
            if previous_dir is not None:
                shutil.rmtree(previous_dir, ignore_errors=True)
        else:
            # All three optional arguments were checked together above.
            assert representation_embedder is not None
            assert representation_knowledge_base_id is not None
            _commit_representation_generation(
                staging_dir=staging_dir,
                final_dir=final_dir,
                document_id=document_id,
                adapted=adapted,
                settings=settings,
                knowledge_base_id=representation_knowledge_base_id,
                representation_store=representation_store,
                representation_embedder=representation_embedder,
            )
    except DocumentParseError:
        raise
    except (MineruAdapterError, MineruRunnerError) as error:
        raise DocumentParseError(str(error)) from error
    except OSError as error:
        raise DocumentParseError("parsed artifacts could not be saved") from error
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return ParseResult(
        document_id=document_id,
        source_sha256=source_sha256,
        parser_version=run_result.version,
        backend=run_result.backend,
        elapsed_seconds=run_result.elapsed_seconds,
        canonical_ref=f"parsing/{document_id}/canonical.json",
        raw_ref=f"parsing/{document_id}/mineru",
        stats=adapted.stats,
    )


async def parse_document_by_id(
    document_id: UUID,
    settings: Settings,
    *,
    representation_store: QdrantRepresentationStore | None = None,
    representation_embedder: RepresentationEmbeddingEncoder | None = None,
    representation_knowledge_base_id: UUID | None = None,
) -> ParseResult:
    """Resolve an uploaded document from PostgreSQL before parsing it off-request."""
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                document = await session.scalar(select(Document).where(Document.id == document_id))
                if document is None:
                    raise DocumentParseError("document was not found")
                if document.storage_kind != DOCUMENT_STORAGE_KIND_MANAGED:
                    raise DocumentParseError("document does not have a managed source file")
                storage_key = document.storage_key
                if storage_key is None:
                    raise DocumentParseError("document does not have a managed source file")
                expected_sha256 = document.sha256
        except DocumentParseError:
            raise
        except SQLAlchemyError as error:
            raise DocumentParseError("document metadata could not be loaded") from error
    finally:
        await engine.dispose()

    try:
        source_path = filesystem_path_for_storage_key(settings.data_dir, storage_key)
    except (OSError, StorageError, ValueError) as error:
        raise DocumentParseError("document storage reference is invalid") from error
    return parse_document_file(
        document_id,
        source_path,
        expected_sha256,
        settings,
        representation_store=representation_store,
        representation_embedder=representation_embedder,
        representation_knowledge_base_id=representation_knowledge_base_id,
    )


__all__ = [
    "PARSING_DIRECTORY_NAME",
    "DocumentParseError",
    "ParseResult",
    "parse_document_by_id",
    "parse_document_file",
]
