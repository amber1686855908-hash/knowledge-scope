"""Safe local filesystem handling for uploaded PDF files."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from fastapi import UploadFile

from .models import DOCUMENT_FILENAME_MAX_LENGTH

UPLOAD_CHUNK_SIZE = 1024 * 1024
PDF_HEADER = b"%PDF-"
PDF_HEADER_SEARCH_BYTES = 1024


class StorageError(RuntimeError):
    """Raised when the controlled local storage layout cannot be used."""


class UploadValidationError(ValueError):
    """Raised when an uploaded file does not meet the PDF upload contract."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """Validated upload data stored in a temporary application-owned path."""

    path: Path
    directory: Path
    original_filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TrashedResource:
    """An application-owned file or directory moved aside during deletion."""

    original_path: Path
    trash_path: Path
    directory: Path
    is_directory: bool


def documents_root(data_dir: Path) -> Path:
    """Return the application-owned root for document files."""
    return Path(data_dir) / "documents"


def storage_key_for_document(knowledge_base_id: UUID, document_id: UUID) -> str:
    """Build the fixed relative storage key used for every uploaded document."""
    return PurePosixPath(
        "documents",
        str(knowledge_base_id),
        str(document_id),
        "original.pdf",
    ).as_posix()


def filesystem_path_for_storage_key(data_dir: Path, storage_key: str) -> Path:
    """Resolve a stored key only if it remains inside the documents root."""
    relative_key = PurePosixPath(storage_key)
    root = documents_root(data_dir).resolve()
    candidate = (Path(data_dir).resolve() / Path(relative_key)).resolve()
    if relative_key.is_absolute() or not candidate.is_relative_to(root):
        raise StorageError("stored document path is outside the documents directory")
    return candidate


def normalize_original_filename(filename: str | None) -> str:
    """Normalize a client filename for metadata without using it as a path."""
    if filename is None:
        raise UploadValidationError("文件名不能为空")

    normalized = unicodedata.normalize("NFC", filename).replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1].strip()
    if not basename or basename in {".", ".."}:
        raise UploadValidationError("文件名不能为空")
    if any(unicodedata.category(character).startswith("C") for character in basename):
        raise UploadValidationError("文件名包含不可用字符")
    if len(basename) > DOCUMENT_FILENAME_MAX_LENGTH:
        raise UploadValidationError("文件名过长")
    if not basename.lower().endswith(".pdf"):
        raise UploadValidationError("仅支持 PDF 文件", status_code=415)
    return basename


async def stage_pdf(upload: UploadFile, data_dir: Path, max_size_bytes: int) -> StagedUpload:
    """Stream, validate, hash, and stage one PDF without loading it wholly in memory."""
    original_filename = normalize_original_filename(upload.filename)
    root = documents_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    staging_path = staging_directory / "payload"
    size_bytes = 0
    digest = hashlib.sha256()
    header_sample = bytearray()

    try:
        with staging_path.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_size_bytes:
                    raise UploadValidationError("文件超过大小限制", status_code=413)
                output.write(chunk)
                digest.update(chunk)
                if len(header_sample) < PDF_HEADER_SEARCH_BYTES:
                    header_sample.extend(chunk[: PDF_HEADER_SEARCH_BYTES - len(header_sample)])
            output.flush()
            os.fsync(output.fileno())

        if size_bytes == 0:
            raise UploadValidationError("文件不能为空")
        if PDF_HEADER not in header_sample:
            raise UploadValidationError("文件不是有效的 PDF", status_code=415)

        return StagedUpload(
            path=staging_path,
            directory=staging_directory,
            original_filename=original_filename,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )
    except BaseException:
        remove_staged_upload(staging_directory, staging_path)
        raise


def promote_staged_upload(staged: StagedUpload, final_path: Path) -> None:
    """Atomically move a staged file into its fixed document location."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged.path, final_path)
    staged.directory.rmdir()


def remove_staged_upload(directory: Path, path: Path) -> None:
    """Remove a temporary upload file and its private directory."""
    with suppress(FileNotFoundError):
        path.unlink()
    with suppress(FileNotFoundError):
        directory.rmdir()


def remove_file(path: Path) -> None:
    """Remove one application-owned file if it still exists."""
    with suppress(FileNotFoundError):
        path.unlink()


def move_to_trash(final_path: Path, data_dir: Path) -> TrashedResource:
    """Move an application-owned file or directory aside before DB deletion."""
    if final_path.is_symlink() or not final_path.exists():
        raise StorageError("document resource is missing or invalid")
    is_directory = final_path.is_dir()
    if not is_directory and not final_path.is_file():
        raise StorageError("document resource is invalid")
    trash_directory = Path(tempfile.mkdtemp(prefix=".delete-", dir=documents_root(data_dir)))
    trash_path = trash_directory / "payload"
    try:
        os.replace(final_path, trash_path)
    except BaseException:
        shutil.rmtree(trash_directory)
        raise
    return TrashedResource(
        original_path=final_path,
        trash_path=trash_path,
        directory=trash_directory,
        is_directory=is_directory,
    )


def restore_from_trash(trashed: TrashedResource) -> None:
    """Restore a moved file or directory after a failed database deletion."""
    trashed.original_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(trashed.trash_path, trashed.original_path)
    trashed.directory.rmdir()


def permanently_remove_trash(trashed: TrashedResource) -> None:
    """Delete a committed file or directory and clean its empty staging parents."""
    if trashed.is_directory:
        shutil.rmtree(trashed.trash_path)
    else:
        trashed.trash_path.unlink(missing_ok=True)
    trashed.directory.rmdir()
    for directory in (trashed.original_path.parent, trashed.original_path.parent.parent):
        try:
            directory.rmdir()
        except OSError:
            break
