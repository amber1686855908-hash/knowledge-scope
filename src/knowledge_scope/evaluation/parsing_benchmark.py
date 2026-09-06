"""Resumable, sequential benchmark for the existing MinerU parsing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4, uuid5

from knowledge_scope.parsing.mineru_adapter import (
    AdapterStats,
    MineruAdapterError,
    adapt_content_list,
    infer_page_count,
)
from knowledge_scope.parsing.mineru_runner import (
    MINERU_BACKEND,
    MineruRunnerError,
    MineruRunResult,
    find_content_list,
    get_mineru_version,
    run_mineru,
)
from knowledge_scope.parsing.models import CANONICAL_SCHEMA_VERSION
from knowledge_scope.parsing.service import (
    MAX_MANIFEST_WARNING_COUNT,
    MAX_MANIFEST_WARNING_LENGTH,
)
from knowledge_scope.shared.config import Settings

BENCHMARK_SCHEMA_VERSION = "1.0"
BENCHMARK_CONCURRENCY = 1
BENCHMARK_NAMESPACE = UUID("5f82e8cb-3c3b-4cd1-8bd4-6a4930e7a2c2")
DEFAULT_BENCHMARK_WORKSPACE = Path("data/benchmarks/a1-5")
UNKNOWN_SUBJECT = "其他/未知"
RAW_RETENTION_VALUES = ("failures", "all", "none")
TERMINAL_STATUSES = frozenset({"success", "failed", "timeout", "skipped_duplicate"})
FAILURE_STATUSES = frozenset({"failed", "timeout"})

RawRetention = Literal["failures", "all", "none"]

SUBJECT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("思想政治", ("思想政治", "政治与法治", "思想品德")),
    ("语文", ("语文",)),
    ("数学", ("数学",)),
    ("英语", ("英语",)),
    ("物理", ("物理",)),
    ("化学", ("化学",)),
    ("生物", ("生物",)),
    ("历史", ("历史",)),
    ("地理", ("地理",)),
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot safely start or resume."""


class _SourceReadError(RuntimeError):
    """Internal marker for a source that cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """One regular PDF discovered in the external read-only corpus."""

    benchmark_item_id: str
    basename: str
    relative_path: str
    size_bytes: int | None
    sha256: str | None
    document_uuid: UUID | None
    physical_pages: int | None
    subject: str
    subject_rule: str | None
    subject_status: str
    inventory_status: str
    page_count_status: str

    def as_record(self) -> dict[str, object]:
        """Return a repository-safe JSON record without the corpus root."""
        return {
            "benchmark_item_id": self.benchmark_item_id,
            "basename": self.basename,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "benchmark_document_uuid": (
                str(self.document_uuid) if self.document_uuid is not None else None
            ),
            "physical_pages": self.physical_pages,
            "subject": self.subject,
            "subject_rule": self.subject_rule,
            "subject_status": self.subject_status,
            "inventory_status": self.inventory_status,
            "page_count_status": self.page_count_status,
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> InventoryItem:
        """Validate and load one previously written inventory record."""
        required = {
            "benchmark_item_id",
            "basename",
            "relative_path",
            "size_bytes",
            "sha256",
            "benchmark_document_uuid",
            "physical_pages",
            "subject",
            "subject_rule",
            "subject_status",
            "inventory_status",
            "page_count_status",
        }
        if not required <= record.keys():
            raise BenchmarkError("corpus inventory record is missing required fields")
        try:
            document_uuid = record["benchmark_document_uuid"]
            return cls(
                benchmark_item_id=_require_string(record, "benchmark_item_id"),
                basename=_require_string(record, "basename"),
                relative_path=_require_string(record, "relative_path"),
                size_bytes=_optional_int(record, "size_bytes"),
                sha256=_optional_string(record, "sha256"),
                document_uuid=UUID(document_uuid) if isinstance(document_uuid, str) else None,
                physical_pages=_optional_int(record, "physical_pages"),
                subject=_require_string(record, "subject"),
                subject_rule=_optional_string(record, "subject_rule"),
                subject_status=_require_string(record, "subject_status"),
                inventory_status=_require_string(record, "inventory_status"),
                page_count_status=_require_string(record, "page_count_status"),
            )
        except (TypeError, ValueError) as error:
            raise BenchmarkError("corpus inventory record is invalid") from error


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """Inventory entries, deterministic fingerprint, and human-readable totals."""

    items: tuple[InventoryItem, ...]
    fingerprint: str
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration that is persisted and checked on benchmark resume."""

    corpus_root: Path
    workspace: Path = DEFAULT_BENCHMARK_WORKSPACE
    raw_retention: RawRetention = "failures"
    resume: bool = False
    retry_failed: bool = False
    limit: int | None = None
    subject: str | None = None

    def __post_init__(self) -> None:
        if self.raw_retention not in RAW_RETENTION_VALUES:
            raise BenchmarkError("raw_retention must be failures, all, or none")
        if self.limit is not None and self.limit < 1:
            raise BenchmarkError("limit must be at least one")
        if self.retry_failed and not self.resume:
            raise BenchmarkError("retry_failed requires resume")


@dataclass(frozen=True, slots=True)
class BenchmarkOutcome:
    """Paths and aggregate data returned after one benchmark invocation."""

    run_id: str
    workspace: Path
    inventory: InventoryResult
    selected_items: tuple[InventoryItem, ...]
    aggregate: dict[str, object]


def _require_string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(key)
    return value


def _optional_string(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(key)
    return value


def _optional_int(record: dict[str, object], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(key)
    return value


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, content: str) -> None:
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


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    _atomic_write_text(path, "".join(f"{_json_line(record)}\n" for record in records))


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(_json_line(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"benchmark metadata could not be read: {path.name}") from error
    if not isinstance(payload, dict):
        raise BenchmarkError(f"benchmark metadata is not an object: {path.name}")
    return payload


def _load_inventory(path: Path) -> tuple[InventoryItem, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkError("corpus inventory could not be read") from error
    items: list[InventoryItem] = []
    try:
        for line in lines:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError
                items.append(InventoryItem.from_record(record))
    except (TypeError, json.JSONDecodeError, BenchmarkError) as error:
        raise BenchmarkError("corpus inventory contains an invalid record") from error
    return tuple(items)


def _load_results(path: Path) -> dict[str, dict[str, object]]:
    """Load the latest result per item and repair a truncated/corrupt tail."""
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise BenchmarkError("benchmark results could not be read") from error

    records: list[dict[str, object]] = []
    record_positions: dict[str, int] = {}
    corrupted = False
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError
            item_id = record.get("benchmark_item_id")
            status = record.get("status")
            if not isinstance(item_id, str) or status not in TERMINAL_STATUSES:
                raise TypeError
        except (TypeError, json.JSONDecodeError):
            corrupted = True
            break
        if item_id in record_positions:
            records[record_positions[item_id]] = record
        else:
            record_positions[item_id] = len(records)
            records.append(record)

    if corrupted:
        _write_jsonl(path, records)
    return {record["benchmark_item_id"]: record for record in records}


def classify_subject(relative_path: str) -> tuple[str, str | None, str]:
    """Classify a relative path only when exactly one explicit rule matches."""
    normalized = relative_path.replace("\\", "/").casefold()
    matches: list[tuple[str, str]] = []
    for subject, tokens in SUBJECT_RULES:
        for token in tokens:
            if token.casefold() in normalized:
                matches.append((subject, token))
                break
    if len(matches) == 1:
        subject, token = matches[0]
        return subject, token, "matched"
    if not matches:
        return UNKNOWN_SUBJECT, None, "unclassified"
    return UNKNOWN_SUBJECT, "+".join(subject for subject, _ in matches), "ambiguous"


def benchmark_item_id(relative_path: str, sha256: str | None) -> str:
    """Create a deterministic identity for one physical corpus entry."""
    identity = f"{relative_path}\0{sha256 or ''}"
    return f"item-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def benchmark_document_uuid(sha256: str | None, relative_path: str) -> UUID:
    """Create a benchmark-only UUIDv5, never a production Document identifier."""
    identity = f"content:{sha256}" if sha256 else f"path:{relative_path}"
    return uuid5(BENCHMARK_NAMESPACE, identity)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_page_count(path: Path) -> tuple[int | None, str]:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        return None, "unavailable"
    try:
        completed = subprocess.run(
            [pdfinfo, str(path)],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "unavailable"
    if completed.returncode != 0:
        return None, "unavailable"
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
    if match is None:
        return None, "unavailable"
    return int(match.group(1)), "available"


def _resolved_corpus_root(corpus_root: Path) -> Path:
    try:
        resolved = Path(corpus_root).resolve(strict=True)
    except OSError as error:
        raise BenchmarkError("corpus root does not exist or cannot be read") from error
    if not resolved.is_dir():
        raise BenchmarkError("corpus root must be a directory")
    return resolved


def _resolved_workspace(workspace: Path, corpus_root: Path) -> Path:
    resolved = Path(workspace).resolve()
    if resolved == corpus_root or resolved.is_relative_to(corpus_root):
        raise BenchmarkError("benchmark workspace must be outside the read-only corpus")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _inventory_summary(items: tuple[InventoryItem, ...]) -> dict[str, object]:
    by_subject: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pdfs": 0, "bytes": 0, "pages": 0, "pages_available": 0}
    )
    sha_groups: dict[str, list[InventoryItem]] = defaultdict(list)
    total_bytes = 0
    total_pages = 0
    pages_available = 0
    for item in items:
        subject_stats = by_subject[item.subject]
        subject_stats["pdfs"] += 1
        if item.size_bytes is not None:
            total_bytes += item.size_bytes
            subject_stats["bytes"] += item.size_bytes
        if item.physical_pages is not None:
            total_pages += item.physical_pages
            pages_available += 1
            subject_stats["pages"] += item.physical_pages
            subject_stats["pages_available"] += 1
        if item.sha256 is not None:
            sha_groups[item.sha256].append(item)

    duplicate_groups = {
        sha: {
            "entries": len(group),
            "item_ids": [item.benchmark_item_id for item in group],
        }
        for sha, group in sorted(sha_groups.items())
        if len(group) > 1
    }
    return {
        "pdfs": len(items),
        "total_bytes": total_bytes,
        "total_pages": total_pages,
        "pages_available_for": pages_available,
        "unknown_or_ambiguous_subjects": sum(
            1 for item in items if item.subject == UNKNOWN_SUBJECT
        ),
        "by_subject": {subject: by_subject[subject] for subject in sorted(by_subject)},
        "duplicate_sha256_groups": duplicate_groups,
    }


def inventory_corpus(corpus_root: Path, workspace: Path) -> InventoryResult:
    """Scan, hash, classify, and persist a deterministic corpus inventory."""
    resolved_corpus = _resolved_corpus_root(corpus_root)
    resolved_workspace = _resolved_workspace(workspace, resolved_corpus)
    items: list[InventoryItem] = []
    try:
        candidates = sorted(
            (
                path
                for path in resolved_corpus.rglob("*")
                if path.suffix.casefold() == ".pdf" and path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.relative_to(resolved_corpus).as_posix(),
        )
    except OSError as error:
        raise BenchmarkError("corpus could not be scanned") from error

    for path in candidates:
        relative_path = path.relative_to(resolved_corpus).as_posix()
        subject, subject_rule, subject_status = classify_subject(relative_path)
        size_bytes: int | None = None
        sha256: str | None = None
        physical_pages: int | None = None
        page_count_status = "unavailable"
        inventory_status = "ready"
        try:
            size_bytes = path.stat().st_size
            sha256 = _sha256_file(path)
            physical_pages, page_count_status = _physical_page_count(path)
        except OSError:
            inventory_status = "source_read_error"
        document_uuid = benchmark_document_uuid(sha256, relative_path)
        items.append(
            InventoryItem(
                benchmark_item_id=benchmark_item_id(relative_path, sha256),
                basename=path.name,
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=sha256,
                document_uuid=document_uuid,
                physical_pages=physical_pages,
                subject=subject,
                subject_rule=subject_rule,
                subject_status=subject_status,
                inventory_status=inventory_status,
                page_count_status=page_count_status,
            )
        )

    result_items = tuple(items)
    fingerprint_digest = hashlib.sha256()
    for item in result_items:
        fingerprint_digest.update(_json_line(item.as_record()).encode("utf-8"))
        fingerprint_digest.update(b"\n")
    fingerprint = fingerprint_digest.hexdigest()
    _write_jsonl(
        resolved_workspace / "corpus-manifest.jsonl",
        (item.as_record() for item in result_items),
    )
    return InventoryResult(
        items=result_items,
        fingerprint=fingerprint,
        summary=_inventory_summary(result_items),
    )


def _select_items(
    items: tuple[InventoryItem, ...], config: BenchmarkConfig
) -> tuple[InventoryItem, ...]:
    selected = [item for item in items if config.subject is None or item.subject == config.subject]
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected:
        raise BenchmarkError("benchmark selection contains no PDFs")
    return tuple(selected)


def _groups(items: Iterable[InventoryItem]) -> dict[str, list[InventoryItem]]:
    groups: dict[str, list[InventoryItem]] = defaultdict(list)
    for item in items:
        key = f"sha256:{item.sha256}" if item.sha256 else f"item:{item.benchmark_item_id}"
        groups[key].append(item)
    return {
        key: sorted(group, key=lambda item: item.relative_path) for key, group in groups.items()
    }


def _representatives(items: Iterable[InventoryItem]) -> dict[str, InventoryItem]:
    return {key: group[0] for key, group in _groups(items).items()}


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _gpu_metadata() -> dict[str, str | None]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"name": None, "driver_version": None}
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"name": None, "driver_version": None}
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if completed.returncode != 0 or not first_line:
        return {"name": None, "driver_version": None}
    name, separator, driver = first_line.partition(",")
    return {
        "name": name.strip() or None,
        "driver_version": driver.strip() if separator else None,
    }


def _disk_preflight(
    workspace: Path, total_bytes: int, raw_retention: RawRetention
) -> dict[str, object]:
    try:
        free_bytes = shutil.disk_usage(workspace).free
    except OSError as error:
        raise BenchmarkError("available disk space could not be inspected") from error
    # This is deliberately a coarse floor, not an expansion-ratio prediction.
    safety_margin = 1024**3
    minimum_free_bytes = (
        max(safety_margin, total_bytes + safety_margin) if raw_retention == "all" else safety_margin
    )
    if free_bytes < minimum_free_bytes:
        raise BenchmarkError(
            "insufficient free disk space for the selected retention policy; "
            "benchmark stopped before parsing"
        )
    return {
        "free_bytes_before_run": free_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "estimate_basis": "coarse safety floor; not a MinerU expansion-ratio estimate",
    }


def _run_config(
    config: BenchmarkConfig,
    inventory: InventoryResult,
    mineru_version: str,
    mineru_command: str,
    mineru_timeout_seconds: int,
) -> dict[str, object]:
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "mineru_version": mineru_version,
        "mineru_backend": MINERU_BACKEND,
        "mineru_executable": Path(mineru_command).name or "configured-mineru",
        "mineru_timeout_seconds": mineru_timeout_seconds,
        "model_source": "local",
        "concurrency": BENCHMARK_CONCURRENCY,
        "raw_retention": config.raw_retention,
        "subject": config.subject,
        "limit": config.limit,
        "inventory_fingerprint": inventory.fingerprint,
    }


def _new_run_metadata(
    config: BenchmarkConfig,
    inventory: InventoryResult,
    selected_items: tuple[InventoryItem, ...],
    mineru_version: str,
    mineru_command: str,
    mineru_timeout_seconds: int,
    disk_preflight: dict[str, object],
) -> dict[str, object]:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "git_commit": _git_commit(),
        "host": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "gpu": _gpu_metadata(),
        },
        "mineru": {
            "version": mineru_version,
            "backend": MINERU_BACKEND,
            "model_source": "local",
            "executable": Path(mineru_command).name or "configured-mineru",
            "timeout_seconds": mineru_timeout_seconds,
            "concurrency": BENCHMARK_CONCURRENCY,
        },
        "config": _run_config(
            config,
            inventory,
            mineru_version,
            mineru_command,
            mineru_timeout_seconds,
        ),
        "inventory": {
            "pdfs": len(inventory.items),
            "fingerprint": inventory.fingerprint,
            "selected_entries": len(selected_items),
            "summary": inventory.summary,
        },
        "disk_preflight": disk_preflight,
        "terminal_result_count": 0,
    }


def _update_run_metadata(path: Path, metadata: dict[str, object], **updates: object) -> None:
    metadata.update(updates)
    metadata["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _write_json(path, metadata)


def _clear_staging(workspace: Path) -> None:
    staging_root = workspace / ".staging"
    if not staging_root.exists():
        return
    for child in staging_root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _remove_path(destination)
    os.replace(source, destination)


def _warning_category(warning: str) -> str:
    lowered = warning.casefold()
    if "table_asset_only" in lowered:
        return "table_asset_only"
    if "table_missing_content" in lowered:
        return "table_missing_content"
    if "bbox_clamped" in lowered:
        return "bbox_clamped"
    if "non-latex" in lowered or ("text_format" in lowered and "latex" in lowered):
        return "non_latex_equation"
    if "non-integer text_level" in lowered:
        return "invalid_text_level"
    if "unsupported or empty" in lowered:
        match = re.search(r"type ['\"]([^'\"]+)['\"]", warning)
        known_types = {"text", "equation", "image", "chart", "list"}
        if match and match.group(1) in known_types:
            return "empty_content"
        return "unsupported_type"
    return "other"


def warning_categories(warnings: Iterable[str]) -> dict[str, int]:
    """Group adapter warnings without changing their original strings."""
    counts = Counter(_warning_category(warning) for warning in warnings)
    return {key: counts[key] for key in sorted(counts)}


def _bounded_warnings(warnings: Iterable[str]) -> list[str]:
    return [
        warning[:MAX_MANIFEST_WARNING_LENGTH]
        for warning in list(warnings)[:MAX_MANIFEST_WARNING_COUNT]
    ]


def _is_bad_case(stats: AdapterStats, elapsed_seconds: float) -> bool:
    return (
        stats.bbox_clamped > 0
        or stats.table_asset_only > 0
        or stats.table_missing_content > 0
        or len(stats.warnings) >= 10
        or stats.unsupported_items >= 10
        or elapsed_seconds >= 300
    )


def _base_result(item: InventoryItem, status: str) -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_item_id": item.benchmark_item_id,
        "benchmark_document_uuid": (
            str(item.document_uuid) if item.document_uuid is not None else None
        ),
        "basename": item.basename,
        "relative_path": item.relative_path,
        "subject": item.subject,
        "sha256": item.sha256,
        "status": status,
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def _failure_category(error: BaseException) -> str:
    message = str(error).casefold()
    if isinstance(error, _SourceReadError):
        return "source_read_error"
    if isinstance(error, MineruRunnerError):
        if "timed out" in message:
            return "mineru_timeout"
        if "content_list" in message:
            return "content_list_missing"
        return "mineru_process_error"
    if isinstance(error, MineruAdapterError):
        if "unsafe image asset" in message:
            return "unsafe_asset"
        if "bbox" in message:
            return "invalid_bbox"
        if "canonical validation" in message:
            return "canonical_validation_error"
        return "adapter_validation_error"
    return "unknown_internal_error"


def classify_failure(error: BaseException) -> str:
    """Return the small stable failure taxonomy used by benchmark results."""
    return _failure_category(error)


def _sanitize_error_message(
    error: BaseException,
    corpus_root: Path,
    workspace: Path,
    source_path: Path | None,
) -> str:
    message = str(error).strip() or error.__class__.__name__
    for path in (corpus_root, workspace, source_path):
        if path is not None:
            message = message.replace(str(path), "<path>")
    return message[:512]


def _failure_layer(category: str) -> str:
    if category.startswith("mineru_") or category == "content_list_missing":
        return "MinerU"
    if category in {"source_read_error"}:
        return "source PDF"
    if category in {
        "adapter_validation_error",
        "unsafe_asset",
        "invalid_bbox",
        "canonical_validation_error",
    }:
        return "adapter"
    return "unknown"


def _source_path(corpus_root: Path, item: InventoryItem) -> Path:
    candidate = (corpus_root / Path(item.relative_path)).resolve()
    if not candidate.is_relative_to(corpus_root) or not candidate.is_file():
        raise _SourceReadError("corpus PDF is missing or is not a regular file")
    return candidate


def _failure_result(
    item: InventoryItem,
    error: BaseException,
    *,
    elapsed_seconds: float,
    corpus_root: Path,
    workspace: Path,
    source_path: Path | None,
    raw_ref: str | None,
) -> dict[str, object]:
    category = _failure_category(error)
    result = _base_result(item, "timeout" if category == "mineru_timeout" else "failed")
    result.update(
        {
            "error_category": category,
            "error_message": _sanitize_error_message(error, corpus_root, workspace, source_path),
            "total_pipeline_elapsed_seconds": elapsed_seconds,
            "canonical_ref": None,
            "raw_ref": raw_ref,
        }
    )
    return result


def _success_result(
    item: InventoryItem,
    run_result: MineruRunResult,
    stats: AdapterStats,
    *,
    elapsed_seconds: float,
    raw_ref: str | None,
) -> dict[str, object]:
    result = _base_result(item, "success")
    result.update(
        {
            "mineru_elapsed_seconds": run_result.elapsed_seconds,
            "total_pipeline_elapsed_seconds": elapsed_seconds,
            "pages": stats.pages,
            "pages_per_second": (
                stats.pages / run_result.elapsed_seconds if run_result.elapsed_seconds > 0 else None
            ),
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
            "warnings": _bounded_warnings(stats.warnings),
            "warning_categories": warning_categories(stats.warnings),
            "canonical_ref": f"canonical/{item.benchmark_item_id}.json",
            "raw_ref": raw_ref,
        }
    )
    return result


def _process_item(
    item: InventoryItem,
    *,
    corpus_root: Path,
    workspace: Path,
    settings: Settings,
    raw_retention: RawRetention,
) -> dict[str, object]:
    """Process exactly one representative using the existing runner and adapter."""
    started_at = perf_counter()
    source_path: Path | None = None
    staging_dir: Path | None = None
    raw_ref: str | None = None
    canonical_path = workspace / "canonical" / f"{item.benchmark_item_id}.json"
    try:
        if item.sha256 is None or item.document_uuid is None:
            raise _SourceReadError("corpus inventory has no usable SHA-256")
        source_path = _source_path(corpus_root, item)
        staging_root = workspace / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{item.benchmark_item_id}-", dir=staging_root))
        mineru_output_dir = staging_dir / "mineru"
        mineru_output_dir.mkdir()
        run_result = run_mineru(
            source_path,
            mineru_output_dir,
            settings.mineru_command,
            timeout_seconds=settings.mineru_timeout_seconds,
        )
        (mineru_output_dir / "stdout.log").write_text(run_result.stdout, encoding="utf-8")
        (mineru_output_dir / "stderr.log").write_text(run_result.stderr, encoding="utf-8")
        content_list_path = find_content_list(mineru_output_dir)
        adapted = adapt_content_list(
            content_list_path,
            item.document_uuid,
            mineru_output_dir,
            page_count=infer_page_count(mineru_output_dir),
        )
        _atomic_write_text(
            canonical_path,
            adapted.document.model_dump_json(indent=2) + "\n",
        )
        total_elapsed = perf_counter() - started_at
        keep_raw = raw_retention == "all" or (
            raw_retention == "failures" and _is_bad_case(adapted.stats, total_elapsed)
        )
        if keep_raw:
            raw_path = workspace / "raw" / item.benchmark_item_id
            _move_path(staging_dir, raw_path)
            staging_dir = None
            raw_ref = f"raw/{item.benchmark_item_id}"
        else:
            _remove_path(staging_dir)
            staging_dir = None
        _remove_path(workspace / "failures" / item.benchmark_item_id)
        return _success_result(
            item,
            run_result,
            adapted.stats,
            elapsed_seconds=total_elapsed,
            raw_ref=raw_ref,
        )
    except Exception as error:
        if canonical_path.exists() or canonical_path.is_symlink():
            _remove_path(canonical_path)
        if staging_dir is not None and staging_dir.exists():
            if raw_retention != "none":
                failure_path = workspace / "failures" / item.benchmark_item_id
                _move_path(staging_dir, failure_path)
                staging_dir = None
                raw_ref = f"failures/{item.benchmark_item_id}"
            else:
                _remove_path(staging_dir)
                staging_dir = None
        return _failure_result(
            item,
            error,
            elapsed_seconds=perf_counter() - started_at,
            corpus_root=corpus_root,
            workspace=workspace,
            source_path=source_path,
            raw_ref=raw_ref,
        )


def _duplicate_result(
    item: InventoryItem, representative: InventoryItem, representative_result: dict[str, object]
) -> dict[str, object]:
    result = _base_result(item, "skipped_duplicate")
    result.update(
        {
            "duplicate_of": representative.benchmark_item_id,
            "representative_status": representative_result.get("status"),
            "accounting": "reuse_representative_content_result",
        }
    )
    return result


def _numeric(record: dict[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def _sum_field(records: Iterable[dict[str, object]], key: str) -> int:
    return sum(int(value) for record in records if (value := record.get(key)) is not None)


def _subject_aggregate(
    representatives: tuple[InventoryItem, ...],
    results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[InventoryItem]] = defaultdict(list)
    for item in representatives:
        grouped[item.subject].append(item)
    summary: dict[str, dict[str, object]] = {}
    block_fields = ("title_blocks", "text_blocks", "tables", "formulas", "images")
    for subject in sorted(grouped):
        items = grouped[subject]
        item_results = [
            results[item.benchmark_item_id] for item in items if item.benchmark_item_id in results
        ]
        successful = [record for record in item_results if record.get("status") == "success"]
        failure_count = sum(record.get("status") in FAILURE_STATUSES for record in item_results)
        mineru_seconds = [_numeric(record, "mineru_elapsed_seconds") for record in successful]
        mineru_seconds = [value for value in mineru_seconds if value is not None]
        successful_pages = _sum_field(successful, "pages")
        subject_metrics: dict[str, object] = {
            "pdfs": len(items),
            "inventory_pages": sum(item.physical_pages or 0 for item in items),
            "successful_pages": successful_pages,
            "success": len(successful),
            "failure": failure_count,
            "median_mineru_seconds": _percentile(mineru_seconds, 50),
            "pages_per_second": (
                round(successful_pages / sum(mineru_seconds), 6)
                if sum(mineru_seconds) > 0
                else None
            ),
            "blocks": {field: _sum_field(successful, field) for field in block_fields},
            "unsupported_items": _sum_field(successful, "unsupported_items"),
            "table_asset_only": _sum_field(successful, "table_asset_only"),
            "table_missing_content": _sum_field(successful, "table_missing_content"),
            "warning_count": _sum_field(successful, "warning_count"),
            "bbox_clamped": _sum_field(successful, "bbox_clamped"),
        }
        summary[subject] = subject_metrics
    return summary


def _bad_case_candidates(
    representatives: tuple[InventoryItem, ...],
    results: dict[str, dict[str, object]],
    workspace: Path,
) -> list[dict[str, object]]:
    candidates: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for item in representatives:
        record = results.get(item.benchmark_item_id)
        if record is None:
            continue
        status = record.get("status")
        warning_count = _numeric(record, "warning_count") or 0
        unsupported = _numeric(record, "unsupported_items") or 0
        pages = max(_numeric(record, "pages") or item.physical_pages or 1, 1)
        reasons: list[str] = []
        if status in FAILURE_STATUSES:
            reasons.append(str(record.get("error_category", "failure")))
        if (_numeric(record, "bbox_clamped") or 0) > 0:
            reasons.append("bbox_clamped")
        if warning_count > 0:
            reasons.append("warning_density")
        if unsupported > 0:
            reasons.append("unsupported_density")
        if (_numeric(record, "table_asset_only") or 0) > 0:
            reasons.append("table_asset_only")
        if (_numeric(record, "table_missing_content") or 0) > 0:
            reasons.append("table_missing_content")
        elapsed = _numeric(record, "mineru_elapsed_seconds") or 0
        if elapsed >= 300:
            reasons.append("slow_document")
        if not reasons:
            continue
        raw_ref = record.get("raw_ref")
        raw_available = isinstance(raw_ref, str) and (workspace / raw_ref).exists()
        candidate = {
            "benchmark_item_id": item.benchmark_item_id,
            "basename": item.basename,
            "relative_path": item.relative_path,
            "subject": item.subject,
            "reason": reasons,
            "observed_layer": (
                _failure_layer(str(record.get("error_category")))
                if status in FAILURE_STATUSES
                else "adapter"
            ),
            "requires_code_change_now": False,
            "raw_available": raw_available,
        }
        priority = (
            1.0 if status in FAILURE_STATUSES else 0.0,
            1.0 if (_numeric(record, "bbox_clamped") or 0) > 0 else 0.0,
            warning_count / pages,
            unsupported / pages,
            elapsed,
        )
        candidates.append((priority, candidate))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return [candidate for _, candidate in candidates[:10]]


def aggregate_results(
    all_items: tuple[InventoryItem, ...],
    selected_items: tuple[InventoryItem, ...],
    results: dict[str, dict[str, object]],
    workspace: Path,
) -> dict[str, object]:
    """Aggregate only persisted results, deduplicating by content SHA-256."""
    all_groups = _groups(all_items)
    selected_groups = _groups(selected_items)
    representatives = tuple(group[0] for group in selected_groups.values())
    representative_results = [
        results[item.benchmark_item_id]
        for item in representatives
        if item.benchmark_item_id in results
    ]
    successful = [record for record in representative_results if record.get("status") == "success"]
    failures = [
        record for record in representative_results if record.get("status") in FAILURE_STATUSES
    ]
    terminal = [
        record for record in representative_results if record.get("status") in TERMINAL_STATUSES
    ]
    mineru_seconds = [_numeric(record, "mineru_elapsed_seconds") for record in successful]
    pipeline_seconds = [_numeric(record, "total_pipeline_elapsed_seconds") for record in successful]
    mineru_seconds = [value for value in mineru_seconds if value is not None]
    pipeline_seconds = [value for value in pipeline_seconds if value is not None]
    successful_pages = _sum_field(successful, "pages")
    total_mineru = round(sum(mineru_seconds), 6)
    total_pipeline = round(sum(pipeline_seconds), 6)
    warning_counter: Counter[str] = Counter()
    for record in successful:
        categories = record.get("warning_categories")
        if isinstance(categories, dict):
            warning_counter.update(
                {key: int(value) for key, value in categories.items() if isinstance(key, str)}
            )
    canonical_counts = {
        field.removesuffix("_blocks"): _sum_field(successful, field)
        for field in ("title_blocks", "text_blocks", "tables", "formulas", "images")
    }
    total_warnings = _sum_field(successful, "warning_count")
    total_unsupported = _sum_field(successful, "unsupported_items")
    total_table_asset_only = _sum_field(successful, "table_asset_only")
    total_table_missing_content = _sum_field(successful, "table_missing_content")
    total_bbox_clamped = _sum_field(successful, "bbox_clamped")
    inventory_summary = _inventory_summary(all_items)
    unique_bytes = sum(item.size_bytes or 0 for group in all_groups.values() for item in group[:1])
    unique_pages = sum(
        item.physical_pages or 0 for group in all_groups.values() for item in group[:1]
    )
    completed_count = len(terminal)
    unique_count = len(representatives)
    failure_categories = Counter(
        str(record.get("error_category")) for record in failures if record.get("error_category")
    )
    success_rate = round(len(successful) / completed_count, 6) if completed_count else None
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "corpus": {
            "inventory_pdf_entries": len(all_items),
            "inventory_unique_pdf_contents": len(all_groups),
            "selected_inventory_entries": len(selected_items),
            "unique_pdf_contents": unique_count,
            "duplicate_content_entries": len(selected_items) - unique_count,
            "total_bytes": inventory_summary["total_bytes"],
            "total_pages": inventory_summary["total_pages"],
            "unique_content_bytes": unique_bytes,
            "unique_content_pages": unique_pages,
            "pages_available_for": inventory_summary["pages_available_for"],
            "subject_distribution": inventory_summary["by_subject"],
            "unknown_or_ambiguous_subjects": inventory_summary["unknown_or_ambiguous_subjects"],
        },
        "benchmark": {
            "concurrency": BENCHMARK_CONCURRENCY,
            "completed_unique_pdfs": completed_count,
            "successful_unique_pdfs": len(successful),
            "failed_unique_pdfs": len(failures),
            "success_rate": success_rate,
            "all_unique_pdfs_terminal": completed_count == unique_count,
            "timeout_count": sum(record.get("status") == "timeout" for record in failures),
            "failure_categories": dict(sorted(failure_categories.items())),
        },
        "performance": {
            "total_mineru_elapsed_seconds": total_mineru,
            "total_pipeline_elapsed_seconds": total_pipeline,
            "p50_mineru_latency_seconds": _percentile(mineru_seconds, 50),
            "p95_mineru_latency_seconds": _percentile(mineru_seconds, 95),
            "min_mineru_latency_seconds": min(mineru_seconds) if mineru_seconds else None,
            "max_mineru_latency_seconds": max(mineru_seconds) if mineru_seconds else None,
            "successful_pages": successful_pages,
            "aggregate_pages_per_second": (
                round(successful_pages / total_mineru, 6) if total_mineru > 0 else None
            ),
        },
        "canonical_blocks": {
            "total": sum(canonical_counts.values()),
            **canonical_counts,
        },
        "signals": {
            "skipped_auxiliary": _sum_field(successful, "skipped_auxiliary"),
            "unsupported_items": total_unsupported,
            "table_asset_only": total_table_asset_only,
            "table_missing_content": total_table_missing_content,
            "bbox_clamped": total_bbox_clamped,
            "warning_count": total_warnings,
            "warning_categories": dict(sorted(warning_counter.items())),
        },
        "normalized_per_100_successful_pages": {
            "warning_count": round(total_warnings * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "unsupported_items": round(total_unsupported * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "table_asset_only": round(total_table_asset_only * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "table_missing_content": round(total_table_missing_content * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "tables": round(canonical_counts["tables"] * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "formulas": round(canonical_counts["formulas"] * 100 / successful_pages, 6)
            if successful_pages
            else None,
            "images": round(canonical_counts["images"] * 100 / successful_pages, 6)
            if successful_pages
            else None,
        },
        "methodology": {
            "mineru_elapsed_seconds": "the existing runner's timed MinerU subprocess only",
            "total_pipeline_elapsed_seconds": (
                "from benchmark item start through adaptation and canonical/raw persistence"
            ),
            "percentiles": "linear interpolation over successful unique representative documents",
            "aggregate_pages_per_second": (
                "successful canonical pages divided by total MinerU elapsed time"
            ),
            "duplicate_accounting": (
                "one representative is parsed per SHA-256; duplicate entries reuse "
                "that content result"
            ),
        },
        "subject_summary": _subject_aggregate(representatives, results),
        "bad_case_candidates": _bad_case_candidates(representatives, results, workspace),
    }


def run_benchmark(config: BenchmarkConfig, settings: Settings) -> BenchmarkOutcome:
    """Run or resume a sequential benchmark with an fsynced result checkpoint."""
    inventory = inventory_corpus(config.corpus_root, config.workspace)
    selected_items = _select_items(inventory.items, config)
    workspace = Path(config.workspace).resolve()
    run_path = workspace / "run.json"
    results_path = workspace / "results.jsonl"
    mineru_version = get_mineru_version(settings.mineru_command)
    disk_preflight = _disk_preflight(
        workspace,
        int(inventory.summary["total_bytes"]),
        config.raw_retention,
    )
    expected_config = _run_config(
        config,
        inventory,
        mineru_version,
        settings.mineru_command,
        settings.mineru_timeout_seconds,
    )

    if run_path.exists():
        if not config.resume:
            raise BenchmarkError("benchmark workspace already exists; use --resume")
        metadata = _load_json(run_path)
        if metadata.get("config") != expected_config:
            raise BenchmarkError("benchmark resume configuration is incompatible with run.json")
        run_id = str(metadata.get("run_id", ""))
        if not run_id:
            raise BenchmarkError("run.json has no run_id")
    else:
        if config.resume:
            raise BenchmarkError("--resume requires an existing run.json")
        if results_path.exists():
            raise BenchmarkError("benchmark results exist without compatible run metadata")
        metadata = _new_run_metadata(
            config,
            inventory,
            selected_items,
            mineru_version,
            settings.mineru_command,
            settings.mineru_timeout_seconds,
            disk_preflight,
        )
        run_id = str(metadata["run_id"])
        _write_json(run_path, metadata)

    _clear_staging(workspace)
    results = _load_results(results_path)
    representatives = _representatives(selected_items)
    for item in selected_items:
        existing = results.get(item.benchmark_item_id)
        if existing is not None:
            status = existing.get("status")
            if status in {"success", "skipped_duplicate"} or (
                status in FAILURE_STATUSES and not config.retry_failed
            ):
                continue

        group_key = f"sha256:{item.sha256}" if item.sha256 else f"item:{item.benchmark_item_id}"
        representative = representatives[group_key]
        if representative.benchmark_item_id != item.benchmark_item_id:
            representative_result = results.get(representative.benchmark_item_id)
            if representative_result is None:
                raise BenchmarkError(
                    "duplicate representative was not checkpointed before duplicate"
                )
            result = _duplicate_result(item, representative, representative_result)
        else:
            result = _process_item(
                item,
                corpus_root=_resolved_corpus_root(config.corpus_root),
                workspace=workspace,
                settings=settings,
                raw_retention=config.raw_retention,
            )
        _append_jsonl(results_path, result)
        results[item.benchmark_item_id] = result
        current_terminal = sum(
            value.get("status") in TERMINAL_STATUSES for value in results.values()
        )
        _update_run_metadata(
            run_path,
            metadata,
            terminal_result_count=current_terminal,
        )

    aggregate = aggregate_results(inventory.items, selected_items, results, workspace)
    _write_json(workspace / "aggregate.json", aggregate)
    selected_unique_count = len(representatives)
    selected_terminal_count = sum(
        results.get(item.benchmark_item_id, {}).get("status") in TERMINAL_STATUSES
        for item in representatives.values()
    )
    _update_run_metadata(
        run_path,
        metadata,
        status=("complete" if selected_terminal_count == selected_unique_count else "partial"),
        terminal_result_count=sum(
            value.get("status") in TERMINAL_STATUSES for value in results.values()
        ),
        aggregate_ref="aggregate.json",
    )
    return BenchmarkOutcome(
        run_id=run_id,
        workspace=workspace,
        inventory=inventory,
        selected_items=selected_items,
        aggregate=aggregate,
    )


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkOutcome",
    "InventoryItem",
    "InventoryResult",
    "aggregate_results",
    "benchmark_document_uuid",
    "benchmark_item_id",
    "classify_failure",
    "classify_subject",
    "inventory_corpus",
    "run_benchmark",
    "warning_categories",
]
