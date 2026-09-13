"""Persistent, dependency-free BM25 retrieval for the A4.3 text branch.

The sparse index is deliberately separate from Qdrant, Neo4j, and the
multimodal representation index.  It stores derived chunk text and lineage in
an SQLite file and never treats the derived index as authoritative source
data.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported development platform is POSIX.
    fcntl = None  # type: ignore[assignment]

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from knowledge_scope.chunking.models import ChunkedDocument, ChunkingConfig
from knowledge_scope.chunking.service import chunk_document
from knowledge_scope.documents.registration import DEFAULT_CORPUS_MANIFEST
from knowledge_scope.evaluation.retrieval_eval import (
    CorpusDocument,
    load_canonical_corpus,
)

SPARSE_INDEX_SCHEMA_VERSION = "1.0"
SPARSE_ALGORITHM = "bm25"
SPARSE_ALGORITHM_VERSION = "bm25-v1"
SPARSE_TOKENIZER_VERSION = "mixed-script-v2"
SPARSE_NORMALIZATION = (
    "NFKC + casefold; CJK unigrams and bigrams; non-CJK alphanumeric runs split at CJK boundaries"
)
SPARSE_DEFAULT_K1 = 1.2
SPARSE_DEFAULT_B = 0.75
SPARSE_DEFAULT_INDEX_PATH = Path("data/evaluation/a4-3/sparse.sqlite3")
SPARSE_PROTECTED_NAMES = frozenset(
    {"knowledgescope_chunks_v1", "knowledgescope_representations_v1"}
)

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)


class SparseIndexError(RuntimeError):
    """Raised when the independent sparse index cannot operate safely."""


class SparseIndexConfigurationError(SparseIndexError):
    """Raised when an existing sparse index has an incompatible contract."""


class SparseIndexBusyError(SparseIndexError):
    """Raised when another process is mutating the same sparse index."""


class SparseIndexCorruptionError(SparseIndexError):
    """Raised when persisted sparse metadata and rows disagree."""


class SparseIndexConfig(BaseModel):
    """Material BM25/tokenizer settings captured by the index contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    algorithm: Literal["bm25"] = SPARSE_ALGORITHM
    algorithm_version: str = Field(default=SPARSE_ALGORITHM_VERSION, min_length=1)
    tokenizer_version: str = Field(default=SPARSE_TOKENIZER_VERSION, min_length=1)
    normalization: str = Field(default=SPARSE_NORMALIZATION, min_length=1)
    bm25_k1: float = Field(default=SPARSE_DEFAULT_K1, gt=0, le=10)
    bm25_b: float = Field(default=SPARSE_DEFAULT_B, ge=0, le=1)

    def static_payload(self) -> dict[str, object]:
        """Return the configuration fields excluding the changing corpus."""

        return self.model_dump(mode="json")

    @property
    def static_fingerprint(self) -> str:
        """Fingerprint algorithm, tokenizer, normalization, and BM25 values."""

        return _sha256_json(self.static_payload())


class SparseIndexContract(BaseModel):
    """Complete contract for one active sparse generation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    algorithm: Literal["bm25"] = SPARSE_ALGORITHM
    algorithm_version: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    normalization: str = Field(min_length=1)
    bm25_k1: float = Field(gt=0, le=10)
    bm25_b: float = Field(ge=0, le=1)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: StrictInt = Field(ge=0)
    document_count: StrictInt = Field(ge=0)

    @classmethod
    def from_config(
        cls,
        config: SparseIndexConfig,
        *,
        corpus_fingerprint: str,
        chunk_count: int,
        document_count: int,
    ) -> SparseIndexContract:
        return cls(
            **config.model_dump(mode="json"),
            corpus_fingerprint=corpus_fingerprint,
            chunk_count=chunk_count,
            document_count=document_count,
        )

    @property
    def static_fingerprint(self) -> str:
        """Fingerprint the non-corpus portion of this contract."""

        payload = self.model_dump(
            mode="json",
            exclude={"corpus_fingerprint", "chunk_count", "document_count"},
        )
        return _sha256_json(payload)

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete active generation contract."""

        return _sha256_json(self.model_dump(mode="json"))


class SparseChunkRecord(BaseModel):
    """One indexed chunk with the authoritative A1.6 lineage copied in."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: UUID
    schema_version: Literal["1.0"] = "1.0"
    chunking_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_id: str = Field(min_length=1)
    document_id: UUID
    ordinal: StrictInt = Field(ge=0)
    text: str = ""
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    content_types: list[str] = Field(min_length=1)
    asset_refs: list[str]

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must contain at least one non-whitespace character")
        return value

    @model_validator(mode="after")
    def validate_lineage(self) -> SparseChunkRecord:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        if any(not value.strip() for value in self.source_block_ids):
            raise ValueError("source_block_ids must not contain blank values")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("source_block_ids must be unique")
        if any(not value.strip() for value in self.content_types):
            raise ValueError("content_types must not contain blank values")
        if len(self.content_types) != len(set(self.content_types)):
            raise ValueError("content_types must be unique")
        if any(not value.strip() for value in self.asset_refs):
            raise ValueError("asset_refs must not contain blank values")
        if not self.text.strip() and not self.asset_refs:
            raise ValueError("chunk must contain text or an asset reference")
        return self

    @classmethod
    def from_chunked_document(
        cls,
        chunked: ChunkedDocument,
        knowledge_base_id: UUID,
    ) -> list[SparseChunkRecord]:
        """Copy one A1.6 artifact into repository-controlled sparse records."""

        return [
            cls(
                knowledge_base_id=knowledge_base_id,
                chunking_config_fingerprint=chunked.config_fingerprint,
                **chunk.model_dump(mode="json"),
            )
            for chunk in chunked.chunks
        ]


class SparseSearchHit(BaseModel):
    """One deterministic BM25 hit with complete chunk lineage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    rank: StrictInt = Field(ge=1)
    bm25_score: float = Field(ge=0)
    knowledge_base_id: UUID
    document_id: UUID
    chunk_id: str = Field(min_length=1)
    ordinal: StrictInt = Field(ge=0)
    page_start: StrictInt = Field(ge=1)
    page_end: StrictInt = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    section_path: list[str]
    content_types: list[str] = Field(min_length=1)
    asset_refs: list[str]
    text: str
    matched_terms: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> SparseSearchHit:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        return self


class SparseSearchResponse(BaseModel):
    """Bounded sparse search response and active index identity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    document_id: UUID | None = None
    top_k: StrictInt = Field(ge=1, le=100)
    index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[SparseSearchHit]


class SparseBuildResult(BaseModel):
    """Safe facts returned by a full or document-scoped rebuild."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    generation_id: str = Field(min_length=1)
    index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_base_count: StrictInt = Field(ge=0)
    document_count: StrictInt = Field(ge=0)
    chunk_count: StrictInt = Field(ge=0)
    searchable_chunk_count: StrictInt = Field(ge=0)
    stale_generations_removed: StrictInt = Field(ge=0)
    cleanup_failed: bool = False


class SparseAuditResult(BaseModel):
    """Read-only comparison between expected A1.6 records and active rows."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SPARSE_INDEX_SCHEMA_VERSION
    status: Literal["ready", "missing", "stale", "incompatible"]
    expected_chunk_count: StrictInt = Field(ge=0)
    indexed_chunk_count: StrictInt = Field(ge=0)
    expected_document_count: StrictInt = Field(ge=0)
    indexed_document_count: StrictInt = Field(ge=0)
    missing_chunk_count: StrictInt = Field(ge=0)
    stale_chunk_count: StrictInt = Field(ge=0)
    corpus_fingerprint: str | None = None
    indexed_corpus_fingerprint: str | None = None
    index_fingerprint: str | None = None
    fingerprint_matches: bool = False


class SparseReadiness(BaseModel):
    """Non-sensitive local sparse-index readiness facts."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "available", "unavailable"]
    index_path: str
    indexed_chunk_count: StrictInt = Field(ge=0)
    index_fingerprint: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ActiveGeneration:
    generation_id: str
    contract: SparseIndexContract


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Chinese, mixed text, numbers, abbreviations, and formulas.

    CJK runs produce unigrams and adjacent bigrams.  Non-CJK alphanumeric
    runs stop at CJK boundaries and retain internal ``.``, ``-`` and ``_``
    when they are unambiguous, which keeps values such as ``2024``, ``bge-m3``
    and ``x_i`` searchable.
    Other punctuation is a boundary; meaningful numeric/symbol operands are
    therefore retained without indexing punctuation as a term.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if _is_cjk(character):
            end = index + 1
            while end < len(normalized) and _is_cjk(normalized[end]):
                end += 1
            run = normalized[index:end]
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run)
                tokens.extend(run[offset : offset + 2] for offset in range(len(run) - 1))
            index = end
            continue

        if character.isalnum() and not _is_cjk(character):
            parts = [character]
            index += 1
            while index < len(normalized):
                current = normalized[index]
                if current.isalnum() and not _is_cjk(current):
                    parts.append(current)
                    index += 1
                    continue
                if (
                    current in ".-_"
                    and index + 1 < len(normalized)
                    and normalized[index - 1].isalnum()
                    and normalized[index + 1].isalnum()
                ):
                    parts.append(current)
                    index += 1
                    continue
                break
            tokens.append("".join(parts))
            continue

        index += 1
    return tuple(tokens)


def term_frequencies(text: str) -> dict[str, int]:
    """Return deterministic term frequencies for one chunk."""

    return dict(sorted(Counter(tokenize(text)).items()))


def corpus_fingerprint(records: Sequence[SparseChunkRecord]) -> str:
    """Fingerprint all indexed text and lineage in stable key order."""

    payload = [
        record.model_dump(mode="json")
        for record in sorted(
            records,
            key=lambda item: (
                str(item.knowledge_base_id),
                str(item.document_id),
                item.ordinal,
                item.chunk_id,
            ),
        )
    ]
    return _sha256_json(payload)


def validate_sparse_index_path(path: Path) -> None:
    """Reject paths that could be confused with protected vector collections."""

    raw = str(path).replace("\\", "/").rstrip("/")
    if raw == ":memory:":
        return
    basename = raw.rsplit("/", maxsplit=1)[-1]
    stem = Path(basename).stem
    if basename in SPARSE_PROTECTED_NAMES or stem in SPARSE_PROTECTED_NAMES:
        raise SparseIndexConfigurationError(
            "sparse index path must not be a protected Qdrant collection name"
        )
    if path.exists() and path.is_dir():
        raise SparseIndexConfigurationError("sparse index path must be a file")


def records_from_chunked_documents(
    chunked_documents: Iterable[ChunkedDocument],
    knowledge_base_id: UUID,
) -> list[SparseChunkRecord]:
    """Convert existing A1.6 artifacts without invoking MinerU."""

    records: list[SparseChunkRecord] = []
    for chunked in chunked_documents:
        records.extend(SparseChunkRecord.from_chunked_document(chunked, knowledge_base_id))
    return records


def load_canonical_chunk_records(
    canonical_root: Path,
    *,
    knowledge_base_id: UUID,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    limit: int | None = None,
    config: ChunkingConfig | None = None,
) -> list[SparseChunkRecord]:
    """Load the same canonical corpus and A1.6 chunking policy used elsewhere."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one")
    corpus = load_canonical_corpus(canonical_root, corpus_manifest_path)
    documents: Iterable[CorpusDocument] = corpus.values()
    if limit is not None:
        documents = list(documents)[:limit]
    chunking_config = config or ChunkingConfig()
    chunked_documents = (
        chunk_document(corpus_document.document, chunking_config) for corpus_document in documents
    )
    records = records_from_chunked_documents(chunked_documents, knowledge_base_id)
    if not records:
        raise SparseIndexError("the selected canonical corpus produced no chunks")
    return records


class SparseIndexStore:
    """SQLite-backed BM25 index with atomic generation replacement."""

    def __init__(
        self,
        path: Path = SPARSE_DEFAULT_INDEX_PATH,
        *,
        config: SparseIndexConfig | None = None,
        read_only: bool = False,
    ) -> None:
        validate_sparse_index_path(path)
        self.path = path
        self.read_only = read_only
        resolved_path = path.resolve()
        self._lock_path = resolved_path.with_name(f"{resolved_path.name}.lock")
        self.config = config or SparseIndexConfig()
        if str(path) == ":memory:":
            if read_only:
                raise SparseIndexError("a read-only sparse index must be an existing file")
            database = ":memory:"
        else:
            if read_only:
                if not path.is_file():
                    raise SparseIndexError(f"read-only sparse index does not exist: {path}")
                database = f"file:{resolved_path.as_posix()}?mode=ro"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                database = str(path)
        try:
            self._connection = sqlite3.connect(
                database,
                timeout=30,
                check_same_thread=False,
                uri=read_only,
            )
            self._connection.execute("PRAGMA foreign_keys = ON")
            if not read_only and database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            if not read_only:
                self._create_schema()
        except sqlite3.Error as error:
            raise SparseIndexError("sparse index storage could not be opened") from error

    def close(self) -> None:
        """Close the local SQLite connection."""

        self._connection.close()

    def __enter__(self) -> SparseIndexStore:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        """Serialize one index's mutating lifecycle across processes."""

        if self.read_only:
            raise SparseIndexError("read-only sparse index cannot be mutated")
        if str(self.path) == ":memory:":
            yield
            return
        if fcntl is None:
            raise SparseIndexError("sparse writer locking is unavailable on this platform")
        try:
            lock_file = self._lock_path.open("a+", encoding="utf-8")
        except OSError as error:
            raise SparseIndexError("sparse index writer lock could not be opened") from error
        try:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise SparseIndexError(
                        "sparse index writer lock could not be acquired"
                    ) from error
                raise SparseIndexBusyError(
                    "sparse index is busy; another process is already writing it"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    @contextmanager
    def _reader_snapshot(self) -> Iterator[None]:
        """Keep one read operation on a consistent WAL snapshot."""

        try:
            self._connection.execute("BEGIN DEFERRED")
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.rollback()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sparse_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sparse_generations (
                generation_id TEXT PRIMARY KEY,
                corpus_fingerprint TEXT NOT NULL,
                index_fingerprint TEXT NOT NULL,
                contract_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                document_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sparse_chunks (
                generation_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                source_block_ids_json TEXT NOT NULL,
                section_path_json TEXT NOT NULL,
                content_types_json TEXT NOT NULL,
                asset_refs_json TEXT NOT NULL,
                text TEXT NOT NULL,
                chunking_config_fingerprint TEXT NOT NULL,
                PRIMARY KEY (generation_id, knowledge_base_id, document_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS sparse_doc_stats (
                generation_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_length INTEGER NOT NULL,
                PRIMARY KEY (generation_id, knowledge_base_id, document_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS sparse_postings (
                generation_id TEXT NOT NULL,
                knowledge_base_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                term TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                PRIMARY KEY (
                    generation_id, knowledge_base_id, document_id, chunk_id, term
                )
            );
            CREATE INDEX IF NOT EXISTS sparse_postings_term_idx
                ON sparse_postings (generation_id, knowledge_base_id, term);
            """
        )
        self._connection.commit()

    def _ensure_static_contract(self) -> None:
        expected_fingerprint = self.config.static_fingerprint
        row = self._connection.execute(
            "SELECT value FROM sparse_meta WHERE key = 'static_contract_fingerprint'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO sparse_meta(key, value) VALUES (?, ?)",
                ("static_contract_fingerprint", expected_fingerprint),
            )
            self._connection.execute(
                "INSERT INTO sparse_meta(key, value) VALUES (?, ?)",
                ("static_contract_json", json.dumps(self.config.static_payload(), sort_keys=True)),
            )
            return
        if row[0] != expected_fingerprint:
            raise SparseIndexConfigurationError(
                "sparse index algorithm/tokenizer configuration is incompatible"
            )

    def _active_generation(self) -> _ActiveGeneration | None:
        row = self._connection.execute(
            "SELECT value FROM sparse_meta WHERE key = 'active_generation_id'"
        ).fetchone()
        if row is None:
            return None
        generation_row = self._connection.execute(
            """
            SELECT generation_id, contract_json
            FROM sparse_generations
            WHERE generation_id = ?
            """,
            (row[0],),
        ).fetchone()
        if generation_row is None:
            raise SparseIndexCorruptionError("active sparse generation is missing")
        try:
            contract = SparseIndexContract.model_validate_json(generation_row[1])
        except ValueError as error:
            raise SparseIndexCorruptionError(
                "active sparse generation contract is invalid"
            ) from error
        if contract.static_fingerprint != self.config.static_fingerprint:
            raise SparseIndexConfigurationError(
                "active sparse generation has an incompatible configuration"
            )
        if contract.fingerprint != generation_row[0].removeprefix("generation-"):
            raise SparseIndexCorruptionError("active sparse generation fingerprint is invalid")
        return _ActiveGeneration(generation_id=generation_row[0], contract=contract)

    @staticmethod
    def _record_key(record: SparseChunkRecord) -> tuple[str, str, str]:
        return str(record.knowledge_base_id), str(record.document_id), record.chunk_id

    def _validate_records(
        self,
        records: Sequence[SparseChunkRecord],
        *,
        allow_empty: bool,
    ) -> list[SparseChunkRecord]:
        if not records and not allow_empty:
            raise ValueError("at least one sparse chunk record is required")
        validated = list(records)
        keys = [self._record_key(record) for record in validated]
        if len(keys) != len(set(keys)):
            raise SparseIndexError("sparse chunk identity is duplicated")
        for record in validated:
            if record.schema_version != SPARSE_INDEX_SCHEMA_VERSION:
                raise SparseIndexConfigurationError(
                    "chunk schema is incompatible with sparse index"
                )
        ordinals_by_document: dict[tuple[str, str], list[int]] = {}
        for record in validated:
            document_key = (str(record.knowledge_base_id), str(record.document_id))
            ordinals_by_document.setdefault(document_key, []).append(record.ordinal)
        for ordinals in ordinals_by_document.values():
            if sorted(ordinals) != list(range(len(ordinals))):
                raise SparseIndexError(
                    "sparse chunk ordinals must be contiguous from zero for each KB/document"
                )
        return sorted(
            validated,
            key=lambda record: (
                str(record.knowledge_base_id),
                str(record.document_id),
                record.ordinal,
                record.chunk_id,
            ),
        )

    def _insert_generation_records(
        self,
        generation_id: str,
        records: Sequence[SparseChunkRecord],
    ) -> None:
        for record in records:
            key = self._record_key(record)
            frequencies = term_frequencies(record.text)
            self._connection.execute(
                """
                INSERT INTO sparse_chunks(
                    generation_id, knowledge_base_id, document_id, chunk_id, ordinal,
                    page_start, page_end, source_block_ids_json, section_path_json,
                    content_types_json, asset_refs_json, text, chunking_config_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    *key,
                    record.ordinal,
                    record.page_start,
                    record.page_end,
                    json.dumps(record.source_block_ids, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(record.section_path, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(record.content_types, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(record.asset_refs, ensure_ascii=False, separators=(",", ":")),
                    record.text,
                    record.chunking_config_fingerprint,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO sparse_doc_stats(
                    generation_id, knowledge_base_id, document_id, chunk_id, document_length
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (generation_id, *key, len(tokenize(record.text))),
            )
            if frequencies:
                self._connection.executemany(
                    """
                    INSERT INTO sparse_postings(
                        generation_id, knowledge_base_id, document_id, chunk_id,
                        term, term_frequency
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (generation_id, *key, term, frequency)
                        for term, frequency in frequencies.items()
                    ),
                )

    def _delete_generation_rows(self, generation_id: str) -> None:
        for table in ("sparse_postings", "sparse_doc_stats", "sparse_chunks"):
            self._connection.execute(
                f"DELETE FROM {table} WHERE generation_id = ?", (generation_id,)
            )

    def _superseded_generation_ids(self, active_generation_id: str) -> set[str]:
        """Return only complete, committed generations safe to remove.

        A candidate is inserted and activated in one transaction, so this
        table contains no visible staging generation.  Rows are therefore
        eligible for cleanup only when their committed generation record is
        not the active candidate.
        """

        return {
            row[0]
            for row in self._connection.execute(
                "SELECT generation_id FROM sparse_generations WHERE generation_id != ?",
                (active_generation_id,),
            )
        }

    def _commit_generation_locked(
        self,
        records: Sequence[SparseChunkRecord],
        *,
        allow_empty: bool,
    ) -> SparseBuildResult:
        validated = self._validate_records(records, allow_empty=allow_empty)
        corpus_hash = corpus_fingerprint(validated)
        document_keys = {
            (str(record.knowledge_base_id), str(record.document_id)) for record in validated
        }
        contract = SparseIndexContract.from_config(
            self.config,
            corpus_fingerprint=corpus_hash,
            chunk_count=len(validated),
            document_count=len(document_keys),
        )
        generation_id = f"generation-{contract.fingerprint}"
        previous_active = self._active_generation()
        try:
            with self._connection:
                self._ensure_static_contract()
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO sparse_generations(
                        generation_id, corpus_fingerprint, index_fingerprint, contract_json,
                        created_at, chunk_count, document_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        corpus_hash,
                        contract.fingerprint,
                        contract.model_dump_json(),
                        datetime.now(UTC).isoformat(),
                        len(validated),
                        len(document_keys),
                    ),
                )
                self._delete_generation_rows(generation_id)
                self._insert_generation_records(generation_id, validated)
                for table in ("sparse_chunks", "sparse_doc_stats"):
                    row = self._connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE generation_id = ?",
                        (generation_id,),
                    ).fetchone()
                    if row[0] != len(validated):
                        raise SparseIndexError(
                            f"candidate sparse generation is incomplete in {table}"
                        )
                self._connection.execute(
                    "INSERT OR REPLACE INTO sparse_meta(key, value) VALUES (?, ?)",
                    ("active_generation_id", generation_id),
                )
        except SparseIndexError:
            raise
        except Exception as error:
            raise SparseIndexError(
                "sparse generation replacement failed; the previous generation was preserved"
            ) from error

        stale_removed = 0
        cleanup_failed = False
        try:
            with self._connection:
                stale_ids = (
                    self._superseded_generation_ids(generation_id)
                    if previous_active is not None
                    else set()
                )
                for stale_id in stale_ids:
                    self._delete_generation_rows(stale_id)
                    self._connection.execute(
                        "DELETE FROM sparse_generations WHERE generation_id = ?",
                        (stale_id,),
                    )
                stale_removed = len(stale_ids)
        except Exception:
            cleanup_failed = True
        return SparseBuildResult(
            generation_id=generation_id,
            index_fingerprint=contract.fingerprint,
            corpus_fingerprint=corpus_hash,
            knowledge_base_count=len({str(record.knowledge_base_id) for record in validated}),
            document_count=len(document_keys),
            chunk_count=len(validated),
            searchable_chunk_count=sum(bool(term_frequencies(record.text)) for record in validated),
            stale_generations_removed=stale_removed,
            cleanup_failed=cleanup_failed,
        )

    def _commit_generation(
        self,
        records: Sequence[SparseChunkRecord],
        *,
        allow_empty: bool,
    ) -> SparseBuildResult:
        with self._writer_lock():
            return self._commit_generation_locked(records, allow_empty=allow_empty)

    def build(self, records: Sequence[SparseChunkRecord]) -> SparseBuildResult:
        """Atomically build a new full-corpus generation."""

        return self._commit_generation(records, allow_empty=False)

    def _active_records(self) -> list[SparseChunkRecord]:
        active = self._active_generation()
        if active is None:
            return []
        rows = self._connection.execute(
            """
            SELECT knowledge_base_id, document_id, chunk_id, ordinal, page_start, page_end,
                   source_block_ids_json, section_path_json, content_types_json,
                   asset_refs_json, text, chunking_config_fingerprint
            FROM sparse_chunks
            WHERE generation_id = ?
            ORDER BY knowledge_base_id, document_id, ordinal, chunk_id
            """,
            (active.generation_id,),
        ).fetchall()
        records: list[SparseChunkRecord] = []
        try:
            for row in rows:
                records.append(
                    SparseChunkRecord(
                        knowledge_base_id=UUID(row[0]),
                        document_id=UUID(row[1]),
                        chunk_id=row[2],
                        ordinal=row[3],
                        page_start=row[4],
                        page_end=row[5],
                        source_block_ids=json.loads(row[6]),
                        section_path=json.loads(row[7]),
                        content_types=json.loads(row[8]),
                        asset_refs=json.loads(row[9]),
                        text=row[10],
                        chunking_config_fingerprint=row[11],
                    )
                )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise SparseIndexCorruptionError("sparse chunk lineage is invalid") from error
        return records

    def replace_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
        records: Sequence[SparseChunkRecord],
    ) -> SparseBuildResult:
        """Replace one KB/document generation while retaining every other document."""

        with self._writer_lock():
            replacement = self._validate_records(records, allow_empty=False)
            if any(
                record.knowledge_base_id != knowledge_base_id or record.document_id != document_id
                for record in replacement
            ):
                raise SparseIndexError("replacement records must share the requested KB/document")
            current = [
                record
                for record in self._active_records()
                if record.knowledge_base_id != knowledge_base_id
                or record.document_id != document_id
            ]
            return self._commit_generation_locked([*current, *replacement], allow_empty=False)

    def delete_document(self, knowledge_base_id: UUID, document_id: UUID) -> int:
        """Atomically remove one KB/document and return its former chunk count."""

        with self._writer_lock():
            current = self._active_records()
            retained = [
                record
                for record in current
                if record.knowledge_base_id != knowledge_base_id
                or record.document_id != document_id
            ]
            removed = len(current) - len(retained)
            if removed == 0:
                return 0
            self._commit_generation_locked(retained, allow_empty=True)
            return removed

    def readiness(self) -> SparseReadiness:
        """Inspect the active local index without creating or rebuilding it."""

        try:
            active = self._active_generation()
            if active is None:
                return SparseReadiness(
                    status="available",
                    index_path=str(self.path),
                    indexed_chunk_count=0,
                )
            return SparseReadiness(
                status="ready",
                index_path=str(self.path),
                indexed_chunk_count=active.contract.chunk_count,
                index_fingerprint=active.contract.fingerprint,
            )
        except SparseIndexError as error:
            return SparseReadiness(
                status="unavailable",
                index_path=str(self.path),
                indexed_chunk_count=0,
                error=str(error),
            )

    def audit(self, expected_records: Sequence[SparseChunkRecord]) -> SparseAuditResult:
        """Compare the active generation with the current authoritative chunk snapshot."""

        expected = self._validate_records(expected_records, allow_empty=True)
        expected_keys = {self._record_key(record) for record in expected}
        expected_hash = corpus_fingerprint(expected)
        active = self._active_generation()
        if active is None:
            return SparseAuditResult(
                status="missing",
                expected_chunk_count=len(expected),
                indexed_chunk_count=0,
                expected_document_count=len(
                    {
                        (str(record.knowledge_base_id), str(record.document_id))
                        for record in expected
                    }
                ),
                indexed_document_count=0,
                missing_chunk_count=len(expected),
                stale_chunk_count=0,
                corpus_fingerprint=expected_hash,
            )
        actual_records = self._active_records()
        actual_keys = {self._record_key(record) for record in actual_records}
        missing = expected_keys - actual_keys
        stale = actual_keys - expected_keys
        indexed_documents = {
            (str(record.knowledge_base_id), str(record.document_id)) for record in actual_records
        }
        fingerprint_matches = active.contract.corpus_fingerprint == expected_hash
        status: Literal["ready", "missing", "stale", "incompatible"]
        if active.contract.static_fingerprint != self.config.static_fingerprint:
            status = "incompatible"
        elif not fingerprint_matches or missing or stale:
            status = "stale"
        else:
            status = "ready"
        return SparseAuditResult(
            status=status,
            expected_chunk_count=len(expected),
            indexed_chunk_count=len(actual_records),
            expected_document_count=len(
                {(str(record.knowledge_base_id), str(record.document_id)) for record in expected}
            ),
            indexed_document_count=len(indexed_documents),
            missing_chunk_count=len(missing),
            stale_chunk_count=len(stale),
            corpus_fingerprint=expected_hash,
            indexed_corpus_fingerprint=active.contract.corpus_fingerprint,
            index_fingerprint=active.contract.fingerprint,
            fingerprint_matches=fingerprint_matches,
        )

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> SparseSearchResponse:
        """Run exact BM25 search against one consistent active generation."""

        with self._reader_snapshot():
            return self._search_in_snapshot(
                query,
                knowledge_base_id=knowledge_base_id,
                top_k=top_k,
                document_id=document_id,
            )

    def _search_in_snapshot(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int = 10,
        document_id: UUID | None = None,
    ) -> SparseSearchResponse:
        """Run exact in-memory BM25 scoring over bounded SQLite postings."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        active = self._active_generation()
        if active is None:
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint="0" * 64,
                items=[],
            )
        terms = tuple(sorted(set(tokenize(query))))
        if not terms:
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint=active.contract.fingerprint,
                items=[],
            )

        scope = [active.generation_id, str(knowledge_base_id)]
        scope_where = "p.generation_id = ? AND p.knowledge_base_id = ?"
        document_clause = ""
        if document_id is not None:
            document_clause = " AND p.document_id = ?"
            scope.append(str(document_id))
        placeholders = ",".join("?" for _ in terms)
        posting_rows = self._connection.execute(
            f"""
            SELECT p.term, p.document_id, p.chunk_id, p.term_frequency, s.document_length
            FROM sparse_postings AS p
            JOIN sparse_doc_stats AS s
              ON s.generation_id = p.generation_id
             AND s.knowledge_base_id = p.knowledge_base_id
             AND s.document_id = p.document_id
             AND s.chunk_id = p.chunk_id
            WHERE {scope_where}{document_clause} AND p.term IN ({placeholders})
            """,
            [*scope, *terms],
        ).fetchall()
        if not posting_rows:
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint=active.contract.fingerprint,
                items=[],
            )

        stats_scope = [active.generation_id, str(knowledge_base_id)]
        stats_clause = ""
        if document_id is not None:
            stats_clause = " AND document_id = ?"
            stats_scope.append(str(document_id))
        count_row = self._connection.execute(
            f"SELECT COUNT(*), COALESCE(AVG(document_length), 0) FROM sparse_doc_stats "
            f"WHERE generation_id = ? AND knowledge_base_id = ?{stats_clause}",
            stats_scope,
        ).fetchone()
        document_count = int(count_row[0])
        average_length = float(count_row[1] or 0)
        if document_count == 0:
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint=active.contract.fingerprint,
                items=[],
            )

        document_frequency = Counter(row[0] for row in posting_rows)
        score_by_key: dict[tuple[str, str], float] = Counter()
        matched_by_key: dict[tuple[str, str], set[str]] = {}
        for term, row_document_id, row_chunk_id, frequency, document_length in posting_rows:
            key = (row_document_id, row_chunk_id)
            idf = math.log1p(
                (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            normalization = (
                1
                - self.config.bm25_b
                + self.config.bm25_b * (document_length / average_length if average_length else 0)
            )
            score_by_key[key] += idf * (
                frequency
                * (self.config.bm25_k1 + 1)
                / (frequency + self.config.bm25_k1 * normalization)
            )
            matched_by_key.setdefault(key, set()).add(term)

        chunk_scope = [active.generation_id, str(knowledge_base_id)]
        chunk_clause = ""
        if document_id is not None:
            chunk_clause = " AND document_id = ?"
            chunk_scope.append(str(document_id))
        metadata_rows = self._connection.execute(
            f"""
            SELECT document_id, chunk_id, ordinal, page_start, page_end,
                   source_block_ids_json, section_path_json, content_types_json,
                   asset_refs_json, text
            FROM sparse_chunks
            WHERE generation_id = ? AND knowledge_base_id = ?{chunk_clause}
            """,
            chunk_scope,
        ).fetchall()
        metadata = {(row[0], row[1]): row for row in metadata_rows}
        scored: list[tuple[float, str, str, tuple[object, ...]]] = []
        for key, score in score_by_key.items():
            row = metadata.get(key)
            if row is None:
                raise SparseIndexCorruptionError("sparse posting has no chunk lineage")
            scored.append((score, key[0], key[1], row))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        hits: list[SparseSearchHit] = []
        for rank, (score, row_document_id, row_chunk_id, row) in enumerate(scored[:top_k], start=1):
            try:
                hits.append(
                    SparseSearchHit(
                        rank=rank,
                        bm25_score=score,
                        knowledge_base_id=knowledge_base_id,
                        document_id=UUID(row_document_id),
                        chunk_id=row_chunk_id,
                        ordinal=row[2],
                        page_start=row[3],
                        page_end=row[4],
                        source_block_ids=json.loads(row[5]),
                        section_path=json.loads(row[6]),
                        content_types=json.loads(row[7]),
                        asset_refs=json.loads(row[8]),
                        text=row[9],
                        matched_terms=sorted(matched_by_key[(row_document_id, row_chunk_id)]),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SparseIndexCorruptionError("sparse result lineage is invalid") from error
        return SparseSearchResponse(
            query=query,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            top_k=top_k,
            index_fingerprint=active.contract.fingerprint,
            items=hits,
        )


__all__ = [
    "SPARSE_ALGORITHM",
    "SPARSE_ALGORITHM_VERSION",
    "SPARSE_DEFAULT_B",
    "SPARSE_DEFAULT_INDEX_PATH",
    "SPARSE_DEFAULT_K1",
    "SPARSE_INDEX_SCHEMA_VERSION",
    "SPARSE_NORMALIZATION",
    "SPARSE_PROTECTED_NAMES",
    "SPARSE_TOKENIZER_VERSION",
    "SparseAuditResult",
    "SparseBuildResult",
    "SparseChunkRecord",
    "SparseIndexBusyError",
    "SparseIndexConfig",
    "SparseIndexConfigurationError",
    "SparseIndexCorruptionError",
    "SparseIndexError",
    "SparseIndexStore",
    "SparseReadiness",
    "SparseSearchHit",
    "SparseSearchResponse",
    "corpus_fingerprint",
    "load_canonical_chunk_records",
    "records_from_chunked_documents",
    "term_frequencies",
    "tokenize",
    "validate_sparse_index_path",
]
