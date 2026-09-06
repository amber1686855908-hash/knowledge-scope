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
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from knowledge_scope.documents.models import Document
from knowledge_scope.documents.storage import StorageError, filesystem_path_for_storage_key
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

PARSING_DIRECTORY_NAME = "parsing"
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


def _promote_staging(staging_dir: Path, final_dir: Path) -> None:
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
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


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
) -> ParseResult:
    """Parse one verified application-owned PDF and atomically persist its artifacts."""
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
        _promote_staging(staging_dir, final_dir)
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


async def parse_document_by_id(document_id: UUID, settings: Settings) -> ParseResult:
    """Resolve an uploaded document from PostgreSQL before parsing it off-request."""
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                document = await session.scalar(select(Document).where(Document.id == document_id))
                if document is None:
                    raise DocumentParseError("document was not found")
                storage_key = document.storage_key
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
    return parse_document_file(document_id, source_path, expected_sha256, settings)


__all__ = [
    "PARSING_DIRECTORY_NAME",
    "DocumentParseError",
    "ParseResult",
    "parse_document_by_id",
    "parse_document_file",
]
