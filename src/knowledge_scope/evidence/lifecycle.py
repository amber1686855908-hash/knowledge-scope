"""Filesystem lifecycle for derived multimodal evidence artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from knowledge_scope.documents.storage import (
    StorageError,
    TrashedResource,
    move_to_trash,
)
from knowledge_scope.parsing.models import CanonicalDocument

from .models import MultimodalEvidenceDocument
from .service import (
    EvidenceValidationError,
    build_evidence_document,
    validate_evidence_document,
)

EVIDENCE_DIRECTORY_NAME = "evidence"
EVIDENCE_ARTIFACT_FILENAME = "evidence.json"


class EvidenceArtifactError(RuntimeError):
    """Raised when a derived evidence artifact cannot be safely managed."""


def _data_root(data_dir: Path) -> Path:
    configured_root = Path(data_dir)
    if configured_root.exists() and not configured_root.is_dir():
        raise EvidenceArtifactError("configured data directory is invalid")
    root = configured_root.resolve()
    if root.exists() and not root.is_dir():
        raise EvidenceArtifactError("configured data directory is invalid")
    return root


def _validate_existing_directory(path: Path, data_root: Path, label: str) -> None:
    """Reject symlinked or out-of-root evidence directory components."""
    if path.is_symlink():
        raise EvidenceArtifactError(f"{label} is a symlink")
    if path.exists() and not path.is_dir():
        raise EvidenceArtifactError(f"{label} is not a directory")
    try:
        resolved = path.resolve()
    except OSError as error:
        raise EvidenceArtifactError(f"{label} cannot be resolved") from error
    if not resolved.is_relative_to(data_root):
        raise EvidenceArtifactError(f"{label} is outside the data directory")


def _checked_storage_paths(
    data_dir: Path,
    document_id: UUID,
) -> tuple[Path, Path, Path, Path]:
    """Resolve evidence paths only after checking every existing ancestor."""
    data_root = _data_root(data_dir)
    evidence_root = data_root / EVIDENCE_DIRECTORY_NAME
    _validate_existing_directory(evidence_root, data_root, "evidence root")
    document_directory = evidence_root / str(document_id)
    _validate_existing_directory(document_directory, data_root, "evidence document directory")
    artifact_path = document_directory / EVIDENCE_ARTIFACT_FILENAME
    if artifact_path.is_symlink():
        raise EvidenceArtifactError("evidence artifact path is a symlink")
    if artifact_path.exists() and not artifact_path.is_file():
        raise EvidenceArtifactError("evidence artifact path is not a file")
    try:
        if not artifact_path.resolve().is_relative_to(data_root):
            raise EvidenceArtifactError("evidence artifact path is outside the data directory")
    except OSError as error:
        raise EvidenceArtifactError("evidence artifact path cannot be resolved") from error
    return data_root, evidence_root, document_directory, artifact_path


def evidence_artifact_directory(data_dir: Path, document_id: UUID) -> Path:
    """Return the controlled directory for one document's derived evidence."""

    return _checked_storage_paths(data_dir, document_id)[2]


def evidence_artifact_path(data_dir: Path, document_id: UUID) -> Path:
    """Return the controlled JSON artifact path for one document."""

    return _checked_storage_paths(data_dir, document_id)[3]


def _write_atomically(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    file_descriptor = -1
    try:
        file_descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            file_descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise EvidenceArtifactError("evidence artifact could not be saved") from error
    finally:
        if file_descriptor != -1:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)


def write_evidence_artifact(
    data_dir: Path,
    artifact: MultimodalEvidenceDocument,
) -> Path:
    """Persist one validated artifact with an atomic file replacement."""

    try:
        artifact = MultimodalEvidenceDocument.model_validate(artifact.model_dump(mode="python"))
    except ValidationError as error:
        raise EvidenceArtifactError("evidence artifact is invalid") from error
    _, _, document_directory, path = _checked_storage_paths(data_dir, artifact.document_id)
    try:
        document_directory.mkdir(parents=True, exist_ok=True)
        _checked_storage_paths(data_dir, artifact.document_id)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise EvidenceArtifactError("evidence artifact path is invalid")
        _write_atomically(path, artifact.model_dump_json(indent=2) + "\n")
    except EvidenceArtifactError:
        raise
    except OSError as error:
        raise EvidenceArtifactError("evidence artifact directory could not be created") from error
    return path


def load_evidence_artifact(data_dir: Path, document_id: UUID) -> MultimodalEvidenceDocument:
    """Load and validate one derived artifact without trusting its contents."""

    path = _checked_storage_paths(data_dir, document_id)[3]
    try:
        artifact = MultimodalEvidenceDocument.model_validate_json(path.read_bytes())
    except EvidenceArtifactError:
        raise
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValidationError) as error:
        raise EvidenceArtifactError("evidence artifact is missing or invalid") from error
    if artifact.document_id != document_id:
        raise EvidenceArtifactError("evidence artifact document ID does not match its path")
    return artifact


def rebuild_evidence_artifact(
    data_dir: Path,
    document: CanonicalDocument,
    knowledge_base_id: UUID,
) -> Path:
    """Build current representations and atomically replace a document artifact."""

    artifact = build_evidence_document(document, knowledge_base_id)
    return write_evidence_artifact(data_dir, artifact)


def assert_evidence_artifact_current(
    data_dir: Path,
    document: CanonicalDocument,
    knowledge_base_id: UUID,
) -> MultimodalEvidenceDocument:
    """Load an artifact and fail closed if its canonical source is stale or invalid."""

    artifact = load_evidence_artifact(data_dir, document.document_id)
    try:
        validate_evidence_document(document, artifact, knowledge_base_id)
    except EvidenceValidationError as error:
        raise EvidenceArtifactError("evidence artifact is stale or inconsistent") from error
    return artifact


def remove_evidence_artifact(data_dir: Path, document_id: UUID) -> bool:
    """Remove one document's derived artifact without following symlinks."""

    _, _, directory, path = _checked_storage_paths(data_dir, document_id)
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise EvidenceArtifactError("evidence artifact directory is invalid")
    try:
        entries = list(directory.iterdir())
        unexpected_entries = [
            entry for entry in entries if entry.name != EVIDENCE_ARTIFACT_FILENAME
        ]
        if unexpected_entries:
            raise EvidenceArtifactError("evidence artifact directory contains unexpected files")
        if path.exists():
            path.unlink()
        directory.rmdir()
    except EvidenceArtifactError:
        raise
    except OSError as error:
        raise EvidenceArtifactError("evidence artifact could not be removed") from error
    return True


def move_evidence_artifact_to_trash(
    data_dir: Path,
    document_id: UUID,
) -> TrashedResource | None:
    """Move one evidence artifact under its own safe, recoverable trash root."""
    _, evidence_root, directory, path = _checked_storage_paths(data_dir, document_id)
    if not directory.exists():
        return None
    if not path.exists() or not path.is_file():
        raise EvidenceArtifactError("evidence document directory contains no valid artifact")
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise EvidenceArtifactError("evidence document directory cannot be inspected") from error
    if any(entry.name != EVIDENCE_ARTIFACT_FILENAME for entry in entries):
        raise EvidenceArtifactError("evidence document directory contains unexpected files")
    try:
        return move_to_trash(directory, data_dir, trash_root=evidence_root)
    except (OSError, StorageError) as error:
        raise EvidenceArtifactError("evidence artifact could not be moved to trash") from error


__all__ = [
    "EVIDENCE_ARTIFACT_FILENAME",
    "EVIDENCE_DIRECTORY_NAME",
    "EvidenceArtifactError",
    "assert_evidence_artifact_current",
    "evidence_artifact_directory",
    "evidence_artifact_path",
    "load_evidence_artifact",
    "move_evidence_artifact_to_trash",
    "rebuild_evidence_artifact",
    "remove_evidence_artifact",
    "write_evidence_artifact",
]
