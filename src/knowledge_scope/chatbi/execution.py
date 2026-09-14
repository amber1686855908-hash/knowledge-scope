"""Datasource-bound, read-only PostgreSQL execution for validated SQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Protocol
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from sqlglot import exp, parse

from .credentials import CredentialResolver, ResolvedDatabaseCredentials
from .errors import ChatBIError, ChatBIErrorCategory
from .nl2sql import NL2SQLService, _RegisteredValidation
from .nl2sql_models import SQL_VALIDATION_VERSION, SQLCandidate, ValidatedSQL
from .policy import QueryPolicy, SQLDialect
from .postgres_schema import is_system_schema
from .schemas import (
    SQL_TEXT_MAX_LENGTH,
    ColumnMetadata,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryTruncationReason,
    ScalarValue,
)
from .sql_validation import policy_fingerprint


class AsyncPreparedStatement(Protocol):
    """Small prepared-statement surface used by the execution adapter."""

    def get_attributes(self) -> Sequence[object]:
        """Return result-column descriptions."""

    def cursor(self, *, prefetch: int = 50) -> AsyncIterator[Mapping[str, object]]:
        """Return a bounded async result cursor."""


class AsyncExecutionConnection(Protocol):
    """Connection operations needed for one read-only query."""

    async def prepare(self, query: str) -> AsyncPreparedStatement:
        """Prepare the already validated SQL."""

    def transaction(self, *, readonly: bool = False) -> AbstractAsyncContextManager[object]:
        """Open an explicit transaction."""

    async def execute(self, query: str) -> str:
        """Execute a fixed adapter command, never caller SQL."""

    async def close(self) -> None:
        """Close the connection."""


ConnectionFactory = Callable[
    [ResolvedDatabaseCredentials, float, int], Awaitable[AsyncExecutionConnection]
]


class ExecutionAdapter(Protocol):
    """Internal adapter contract for datasource-specific execution."""

    async def _execute_validated(
        self,
        credentials: ResolvedDatabaseCredentials,
        *,
        datasource_id: UUID,
        query_id: UUID,
        validated: ValidatedSQL,
        policy: QueryPolicy,
    ) -> QueryExecutionResult:
        """Execute an internal validation result for one datasource."""


async def _connect_postgres(
    credentials: ResolvedDatabaseCredentials,
    connection_timeout_seconds: float,
    statement_timeout_ms: int,
) -> AsyncExecutionConnection:
    """Connect with server-side read-only and timeout defaults."""
    return await asyncpg.connect(
        dsn=credentials.connection_url,
        timeout=connection_timeout_seconds,
        server_settings={
            "default_transaction_read_only": "on",
            "statement_timeout": str(statement_timeout_ms),
            "search_path": "pg_catalog, pg_temp",
        },
    )


class SQLExecutionAuditRecord(BaseModel):
    """Request-scoped execution audit record with no credentials, DSN, or raw SQL."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    audit_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    datasource_id: UUID
    sql_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    state: QueryLifecycleState
    schema_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    validation_version: str | None = Field(default=None, min_length=1, max_length=64)
    # These field names are retained for compatibility, but their values are
    # always parser-redacted audit/display SQL rather than raw execution input.
    original_sql: str | None = Field(default=None, max_length=SQL_TEXT_MAX_LENGTH)
    normalized_sql: str | None = Field(default=None, max_length=SQL_TEXT_MAX_LENGTH)
    row_count: int | None = Field(default=None, ge=0)
    max_rows: int = Field(ge=1)
    truncated: StrictBool | None = None
    truncation_reason: QueryTruncationReason | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_category: ChatBIErrorCategory | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("original_sql", "normalized_sql", mode="before")
    @classmethod
    def normalize_sql(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        redacted = redact_sql_literals(value).strip()
        return redacted or None

    @model_validator(mode="after")
    def validate_lifecycle_error(self) -> SQLExecutionAuditRecord:
        if (
            self.state
            in {
                QueryLifecycleState.CREATED,
                QueryLifecycleState.VALIDATED,
                QueryLifecycleState.EXECUTING,
            }
            and self.error_category is not None
        ):
            raise ValueError("non-terminal execution audit states cannot contain an error")
        if self.state is QueryLifecycleState.SUCCEEDED and self.error_category is not None:
            raise ValueError("successful execution audit records cannot contain an error")
        if (
            self.state
            in {
                QueryLifecycleState.REJECTED,
                QueryLifecycleState.FAILED,
                QueryLifecycleState.CANCELLED,
            }
            and self.error_category is None
        ):
            raise ValueError("terminal execution errors require an error category")
        if self.state is QueryLifecycleState.SUCCEEDED:
            if self.truncated and self.truncation_reason is None:
                raise ValueError("truncated execution audit records require a reason")
            if not self.truncated and self.truncation_reason is not None:
                raise ValueError("complete execution audit records cannot have a reason")
        elif self.truncated or self.truncation_reason is not None:
            raise ValueError(
                "non-successful execution audit records cannot contain truncation metadata"
            )
        return self


class ExecutionAuditRecorder(Protocol):
    """Optional sink for request-scoped audit records."""

    def record(self, record: SQLExecutionAuditRecord) -> None:
        """Record a non-secret lifecycle observation."""


@dataclass(frozen=True, slots=True)
class SQLExecutionOutcome:
    """Normalized result together with its lifecycle audit records."""

    result: QueryExecutionResult
    audit: tuple[SQLExecutionAuditRecord, ...]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "result": self.result.model_dump(mode=mode),
            "audit": [record.model_dump(mode=mode) for record in self.audit],
        }


def _quote_identifier(identifier: str) -> str:
    """Quote one policy-owned PostgreSQL identifier."""
    return '"' + identifier.replace('"', '""') + '"'


_REDACTION_FALLBACK = "<sql-redaction-unavailable>"

# SQLGlot uses dedicated nodes for PostgreSQL string syntaxes that are not
# represented by ``exp.Literal``.  Keep this list explicit so a newly
# encountered literal node can never silently pass through the audit redactor.
_REDACTABLE_LITERAL_TYPES = (
    exp.RawString,
    exp.ByteString,
    exp.BitString,
    exp.HexString,
    exp.National,
    exp.UnicodeString,
)


def redact_sql_literals(sql: str) -> str:
    """Return a safe audit/display form without retaining literal values.

    The PostgreSQL parser preserves the statement structure and identifiers while
    replacing values represented by the AST.  Any parse or rendering failure
    returns a fixed marker rather than falling back to the caller's raw SQL.
    """
    if not isinstance(sql, str) or not sql or len(sql) > SQL_TEXT_MAX_LENGTH:
        return _REDACTION_FALLBACK
    try:
        statements = parse(sql, read="postgres")
        if len(statements) != 1 or statements[0] is None:
            return _REDACTION_FALLBACK
        statement = statements[0]
        for node in tuple(statement.walk()):
            if isinstance(node, exp.Boolean):
                node.replace(exp.Boolean(this=False))
            elif isinstance(node, exp.Literal):
                replacement = (
                    exp.Literal.string("<redacted>") if node.is_string else exp.Literal.number("0")
                )
                node.replace(replacement)
            elif isinstance(node, _REDACTABLE_LITERAL_TYPES):
                node.replace(exp.Literal.string("<redacted>"))
        rendered = statement.sql(dialect="postgres", comments=False).strip()
        return rendered if rendered else _REDACTION_FALLBACK
    except Exception:
        return _REDACTION_FALLBACK


def build_safe_search_path_statement(policy: QueryPolicy) -> str:
    """Build a fixed search_path from the approved policy schemas.

    Physical tables are qualified by A5.3 before execution.  The explicit
    path is still set to prevent an ambient session path from affecting any
    remaining name resolution.  ``pg_catalog`` is first and ``pg_temp`` is
    last; caller-provided path fragments are never accepted.
    """
    if not isinstance(policy, QueryPolicy) or not policy.read_only:
        raise ChatBIError(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "SQL execution requires a read-only query policy",
        )
    schemas = tuple(policy.allowed_schemas)
    if not schemas or any(is_system_schema(schema) for schema in schemas):
        raise ChatBIError(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "SQL execution requires non-system approved schemas",
        )
    quoted = [_quote_identifier("pg_catalog")]
    quoted.extend(_quote_identifier(schema) for schema in schemas)
    quoted.append(_quote_identifier("pg_temp"))
    return "SET LOCAL search_path TO " + ", ".join(quoted)


def _attribute_name(attribute: object) -> str:
    value = getattr(attribute, "name", None)
    if not isinstance(value, str) or not value:
        raise ChatBIError(
            ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
            "PostgreSQL returned an invalid result column name",
        )
    return value


def _attribute_type(attribute: object) -> str:
    type_info = getattr(attribute, "type", None)
    value = getattr(type_info, "name", None)
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _record_values(record: object) -> tuple[object, ...]:
    values = getattr(record, "values", None)
    if callable(values):
        try:
            return tuple(values())
        except Exception:
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
                "PostgreSQL returned an invalid result row",
            ) from None
    if isinstance(record, Mapping):
        return tuple(record.values())
    raise ChatBIError(
        ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
        "PostgreSQL returned an invalid result row",
    )


def _record_names(record: object) -> tuple[str, ...]:
    keys = getattr(record, "keys", None)
    if callable(keys):
        names = tuple(keys())
    elif isinstance(record, Mapping):
        names = tuple(record)
    else:
        return ()
    if any(not isinstance(name, str) or not name for name in names):
        raise ChatBIError(
            ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
            "PostgreSQL returned invalid result column names",
        )
    return names


def _json_safe(
    value: object,
    *,
    max_depth: int = 32,
    max_collection_items: int = 10_000,
    _depth: int = 0,
) -> ScalarValue:
    """Convert a PostgreSQL value while bounding nested containers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
                "PostgreSQL returned a non-finite numeric value",
            )
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Mapping):
        if _depth >= max_depth:
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_SIZE_EXCEEDED,
                "a PostgreSQL result value exceeds the configured nesting bound",
            )
        if len(value) > max_collection_items:
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_SIZE_EXCEEDED,
                "a PostgreSQL result collection exceeds the configured item bound",
            )
        result: dict[str, ScalarValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ChatBIError(
                    ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
                    "PostgreSQL returned a JSON object with a non-text key",
                )
            result[key] = _json_safe(
                item,
                max_depth=max_depth,
                max_collection_items=max_collection_items,
                _depth=_depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        if _depth >= max_depth:
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_SIZE_EXCEEDED,
                "a PostgreSQL result value exceeds the configured nesting bound",
            )
        if len(value) > max_collection_items:
            raise ChatBIError(
                ChatBIErrorCategory.RESULT_SIZE_EXCEEDED,
                "a PostgreSQL result collection exceeds the configured item bound",
            )
        return [
            _json_safe(
                item,
                max_depth=max_depth,
                max_collection_items=max_collection_items,
                _depth=_depth + 1,
            )
            for item in value
        ]
    raise ChatBIError(
        ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
        "PostgreSQL returned a value outside the JSON-safe result policy",
    )


def _json_bytes(value: ScalarValue) -> bytes:
    """Serialize normalized data with one stable UTF-8 size definition."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise ChatBIError(
            ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
            "PostgreSQL returned a value that cannot be serialized safely",
        ) from None


def _normalize_columns(
    attributes: Sequence[object],
) -> tuple[tuple[str, ...], list[ColumnMetadata]]:
    names = tuple(_attribute_name(attribute) for attribute in attributes)
    columns = [
        ColumnMetadata(
            name=name,
            data_type=_attribute_type(attribute),
            nullable=True,
            ordinal=index,
        )
        for index, (attribute, name) in enumerate(zip(attributes, names, strict=True))
    ]
    return names, columns


class PostgresExecutionAdapter:
    """Execute one internally validated query in a bounded read-only transaction."""

    def __init__(
        self,
        *,
        connection_timeout_seconds: float = 10.0,
        statement_timeout_ms: int = 30_000,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if connection_timeout_seconds <= 0:
            raise ValueError("connection_timeout_seconds must be positive")
        if statement_timeout_ms < 100:
            raise ValueError("statement_timeout_ms must be at least 100")
        self._connection_timeout_seconds = connection_timeout_seconds
        self._statement_timeout_ms = statement_timeout_ms
        self._connection_factory = connection_factory or _connect_postgres

    @staticmethod
    def _validate_execution_context(
        datasource_id: UUID,
        validated: ValidatedSQL,
        policy: QueryPolicy,
    ) -> None:
        """Reject stale internal results even at the adapter boundary."""
        if not isinstance(validated, ValidatedSQL) or validated.datasource_id != datasource_id:
            raise ChatBIError(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "validated SQL is not bound to the requested datasource",
            )
        if validated.validation_version != SQL_VALIDATION_VERSION:
            raise ChatBIError(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "validated SQL uses an unsupported validator version",
            )
        if validated.policy_fingerprint != policy_fingerprint(policy):
            raise ChatBIError(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "validated SQL uses a stale query policy",
            )
        if (
            validated.dialect is not SQLDialect.POSTGRESQL
            or policy.dialect is not SQLDialect.POSTGRESQL
        ):
            raise ChatBIError(
                ChatBIErrorCategory.UNSUPPORTED_DIALECT,
                "SQL execution currently supports PostgreSQL only",
            )
        if not policy.read_only or not validated.limit_bounded:
            raise ChatBIError(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "SQL execution requires bounded read-only validation",
            )
        if validated.effective_limit > policy.max_rows:
            raise ChatBIError(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "validated SQL exceeds the configured result bound",
            )

    async def _execute_validated(
        self,
        credentials: ResolvedDatabaseCredentials,
        *,
        datasource_id: UUID,
        query_id: UUID,
        validated: ValidatedSQL,
        policy: QueryPolicy,
    ) -> QueryExecutionResult:
        """Execute only a result issued by the trusted service path."""
        self._validate_execution_context(datasource_id, validated, policy)
        connection: AsyncExecutionConnection | None = None
        started = asyncio.get_running_loop().time()
        try:
            try:
                connection = await self._connection_factory(
                    credentials,
                    self._connection_timeout_seconds,
                    self._statement_timeout_ms,
                )
            except TimeoutError:
                raise ChatBIError(
                    ChatBIErrorCategory.EXECUTION_TIMEOUT,
                    "PostgreSQL connection timed out",
                ) from None
            except (asyncpg.exceptions.PostgresConnectionError, ConnectionError, OSError):
                raise ChatBIError(
                    ChatBIErrorCategory.DATASOURCE_UNAVAILABLE,
                    "PostgreSQL datasource is unavailable",
                ) from None

            async with connection.transaction(readonly=True):
                await connection.execute(build_safe_search_path_statement(policy))
                await connection.execute(
                    f"SET LOCAL statement_timeout = {policy.statement_timeout_ms}"
                )
                statement = await connection.prepare(validated.normalized_sql)
                attributes = tuple(statement.get_attributes())
                column_names, columns = _normalize_columns(attributes)
                rows: list[list[ScalarValue]] = []
                truncated = False
                truncation_reason: QueryTruncationReason | None = None
                payload_bytes = len(b"[]")
                cursor = statement.cursor(prefetch=1)
                async for record in cursor:
                    if not column_names:
                        column_names = _record_names(record)
                        columns = [
                            ColumnMetadata(
                                name=name,
                                data_type="unknown",
                                nullable=True,
                                ordinal=index,
                            )
                            for index, name in enumerate(column_names)
                        ]
                    if len(rows) >= policy.max_rows:
                        truncated = True
                        truncation_reason = QueryTruncationReason.ROW_LIMIT
                        break
                    values = _record_values(record)
                    if len(values) != len(column_names):
                        raise ChatBIError(
                            ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED,
                            "PostgreSQL returned a row with inconsistent column count",
                        )
                    normalized_row: list[ScalarValue] = []
                    for value in values:
                        normalized_value = _json_safe(
                            value,
                            max_depth=policy.max_nested_value_depth,
                            max_collection_items=policy.max_collection_items,
                        )
                        if len(_json_bytes(normalized_value)) > policy.max_cell_bytes:
                            raise ChatBIError(
                                ChatBIErrorCategory.RESULT_SIZE_EXCEEDED,
                                "a PostgreSQL result cell exceeds the configured size bound",
                            )
                        normalized_row.append(normalized_value)
                    row_bytes = len(_json_bytes(normalized_row))
                    separator_bytes = 1 if rows else 0
                    if payload_bytes + separator_bytes + row_bytes > policy.max_result_bytes:
                        truncated = True
                        truncation_reason = QueryTruncationReason.PAYLOAD_BYTES
                        break
                    rows.append(normalized_row)
                    payload_bytes += separator_bytes + row_bytes

            return QueryExecutionResult(
                query_id=query_id,
                datasource_id=datasource_id,
                state=QueryLifecycleState.SUCCEEDED,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                truncation_reason=truncation_reason,
                max_rows=policy.max_rows,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except ChatBIError:
            raise
        except TimeoutError:
            raise ChatBIError(
                ChatBIErrorCategory.EXECUTION_TIMEOUT,
                "PostgreSQL query timed out",
            ) from None
        except asyncpg.exceptions.QueryCanceledError:
            raise ChatBIError(
                ChatBIErrorCategory.EXECUTION_TIMEOUT,
                "PostgreSQL query timed out",
            ) from None
        except asyncpg.exceptions.PostgresConnectionError:
            raise ChatBIError(
                ChatBIErrorCategory.DATASOURCE_UNAVAILABLE,
                "PostgreSQL datasource connection failed",
            ) from None
        except asyncpg.exceptions.PostgresError:
            raise ChatBIError(
                ChatBIErrorCategory.EXECUTION_FAILED,
                "PostgreSQL query execution failed",
            ) from None
        except Exception:
            raise ChatBIError(
                ChatBIErrorCategory.EXECUTION_FAILED,
                "PostgreSQL query execution failed",
            ) from None
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass


_REJECTION_CATEGORIES = frozenset(
    {
        ChatBIErrorCategory.INVALID_QUERY,
        ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT,
        ChatBIErrorCategory.SQL_PARSE_ERROR,
        ChatBIErrorCategory.UNSAFE_QUERY,
        ChatBIErrorCategory.POLICY_VIOLATION,
        ChatBIErrorCategory.UNKNOWN_TABLE,
        ChatBIErrorCategory.UNKNOWN_COLUMN,
        ChatBIErrorCategory.DATASOURCE_DISABLED,
        ChatBIErrorCategory.UNSUPPORTED_DIALECT,
    }
)


class SQLExecutionService:
    """Re-enter trusted discovery/validation before every execution."""

    def __init__(
        self,
        validation_service: NL2SQLService,
        credential_resolver: CredentialResolver,
        execution_adapter: ExecutionAdapter,
        *,
        audit_recorder: ExecutionAuditRecorder | None = None,
    ) -> None:
        self._validation_service = validation_service
        self._credential_resolver = credential_resolver
        self._execution_adapter = execution_adapter
        self._audit_recorder = audit_recorder

    @staticmethod
    def _fingerprint(sql: str) -> str:
        return hashlib.sha256(sql.encode("utf-8")).hexdigest()

    def _emit(
        self,
        records: list[SQLExecutionAuditRecord],
        record: SQLExecutionAuditRecord,
    ) -> None:
        records.append(record)
        if self._audit_recorder is not None:
            with suppress(Exception):
                self._audit_recorder.record(record)

    def _audit(
        self,
        *,
        query_id: UUID,
        datasource_id: UUID,
        sql: str,
        policy: QueryPolicy,
        state: QueryLifecycleState,
        registered: _RegisteredValidation | None = None,
        result: QueryExecutionResult | None = None,
        error: ChatBIError | None = None,
        duration_ms: float | None = None,
    ) -> SQLExecutionAuditRecord:
        validated = registered.validated if registered is not None else None
        return SQLExecutionAuditRecord(
            audit_id=uuid4(),
            query_id=query_id,
            datasource_id=datasource_id,
            sql_fingerprint=self._fingerprint(sql),
            state=state,
            schema_fingerprint=validated.schema_fingerprint if validated else None,
            policy_fingerprint=policy_fingerprint(policy),
            validation_version=validated.validation_version if validated else None,
            original_sql=redact_sql_literals(sql) if len(sql) <= SQL_TEXT_MAX_LENGTH else None,
            normalized_sql=(redact_sql_literals(validated.normalized_sql) if validated else None),
            row_count=result.row_count if result else None,
            max_rows=policy.max_rows,
            truncated=result.truncated if result else None,
            truncation_reason=result.truncation_reason if result else None,
            duration_ms=duration_ms,
            error_category=error.category if error else None,
        )

    @staticmethod
    def _error_result(
        *,
        query_id: UUID,
        datasource_id: UUID,
        policy: QueryPolicy,
        state: QueryLifecycleState,
        error: ChatBIError,
        duration_ms: float,
    ) -> QueryExecutionResult:
        return QueryExecutionResult(
            query_id=query_id,
            datasource_id=datasource_id,
            state=state,
            columns=[],
            rows=[],
            row_count=0,
            truncated=False,
            max_rows=policy.max_rows,
            duration_ms=max(duration_ms, 0.0),
            error_category=error.category,
            error_message=error.safe_message,
        )

    async def _run(
        self,
        *,
        datasource_id: UUID,
        sql: str,
        policy: QueryPolicy,
        query_id: UUID,
        prepare: Callable[[], Awaitable[_RegisteredValidation]],
    ) -> SQLExecutionOutcome:
        records: list[SQLExecutionAuditRecord] = []
        started = asyncio.get_running_loop().time()
        self._emit(
            records,
            self._audit(
                query_id=query_id,
                datasource_id=datasource_id,
                sql=sql,
                policy=policy,
                state=QueryLifecycleState.CREATED,
            ),
        )
        try:
            registered = await prepare()
        except asyncio.CancelledError:
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.CANCELLED,
                    error=ChatBIError(
                        ChatBIErrorCategory.EXECUTION_CANCELLED,
                        "SQL execution was cancelled",
                    ),
                    duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
                ),
            )
            raise
        except ChatBIError as error:
            state = (
                QueryLifecycleState.REJECTED
                if error.category in _REJECTION_CATEGORIES
                else QueryLifecycleState.FAILED
            )
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=state,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=state,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))
        except Exception:
            error = ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "trusted schema validation failed",
            )
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=QueryLifecycleState.FAILED,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.FAILED,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))

        self._emit(
            records,
            self._audit(
                query_id=query_id,
                datasource_id=datasource_id,
                sql=sql,
                policy=policy,
                state=QueryLifecycleState.VALIDATED,
                registered=registered,
            ),
        )
        try:
            credentials = self._credential_resolver.resolve(registered.data_source.connection_ref)
        except ChatBIError as error:
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=QueryLifecycleState.FAILED,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.FAILED,
                    registered=registered,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))
        except Exception:
            error = ChatBIError(
                ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
                "the datasource credential could not be resolved",
            )
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=QueryLifecycleState.FAILED,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.FAILED,
                    registered=registered,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))

        self._emit(
            records,
            self._audit(
                query_id=query_id,
                datasource_id=datasource_id,
                sql=sql,
                policy=policy,
                state=QueryLifecycleState.EXECUTING,
                registered=registered,
            ),
        )
        try:
            result = await self._execution_adapter._execute_validated(
                credentials,
                datasource_id=datasource_id,
                query_id=query_id,
                validated=registered.validated,
                policy=policy,
            )
            if result.state is not QueryLifecycleState.SUCCEEDED:
                raise ChatBIError(
                    ChatBIErrorCategory.EXECUTION_FAILED,
                    "PostgreSQL execution returned an invalid terminal state",
                )
            result = result.model_copy(update={"query_id": query_id})
        except asyncio.CancelledError:
            error = ChatBIError(
                ChatBIErrorCategory.EXECUTION_CANCELLED,
                "SQL execution was cancelled",
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.CANCELLED,
                    registered=registered,
                    error=error,
                    duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
                ),
            )
            raise
        except ChatBIError as error:
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=QueryLifecycleState.FAILED,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.FAILED,
                    registered=registered,
                    result=result,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))
        except Exception:
            error = ChatBIError(
                ChatBIErrorCategory.EXECUTION_FAILED,
                "PostgreSQL query execution failed",
            )
            result = self._error_result(
                query_id=query_id,
                datasource_id=datasource_id,
                policy=policy,
                state=QueryLifecycleState.FAILED,
                error=error,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            self._emit(
                records,
                self._audit(
                    query_id=query_id,
                    datasource_id=datasource_id,
                    sql=sql,
                    policy=policy,
                    state=QueryLifecycleState.FAILED,
                    registered=registered,
                    result=result,
                    error=error,
                    duration_ms=result.duration_ms,
                ),
            )
            return SQLExecutionOutcome(result=result, audit=tuple(records))

        self._emit(
            records,
            self._audit(
                query_id=query_id,
                datasource_id=datasource_id,
                sql=sql,
                policy=policy,
                state=QueryLifecycleState.SUCCEEDED,
                registered=registered,
                result=result,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            ),
        )
        return SQLExecutionOutcome(result=result, audit=tuple(records))

    async def execute_for_registered_data_source(
        self,
        datasource_id: UUID,
        candidate: SQLCandidate,
        *,
        policy: QueryPolicy,
        max_chars: int,
        query_id: UUID | None = None,
    ) -> SQLExecutionOutcome:
        """Execute an untrusted candidate after fresh trusted validation."""
        if not isinstance(datasource_id, UUID) or not isinstance(candidate, SQLCandidate):
            raise TypeError("datasource_id and candidate must use the registered input contracts")
        if not isinstance(policy, QueryPolicy):
            raise TypeError("policy must be a QueryPolicy")
        return await self._run(
            datasource_id=datasource_id,
            sql=candidate.sql,
            policy=policy,
            query_id=query_id or uuid4(),
            prepare=lambda: self._validation_service._validate_registered_candidate(
                datasource_id,
                candidate,
                policy=policy,
                max_chars=max_chars,
            ),
        )

    async def execute_sql_for_registered_data_source(
        self,
        datasource_id: UUID,
        sql: str,
        *,
        policy: QueryPolicy,
        max_chars: int,
        query_id: UUID | None = None,
    ) -> SQLExecutionOutcome:
        """Execute developer/user SQL as raw untrusted input.

        The service creates the temporary candidate only after it has obtained
        the registered datasource and a fresh schema snapshot.  There is no
        public method accepting ``ValidatedSQL`` or a caller-owned snapshot.
        """
        if not isinstance(datasource_id, UUID) or not isinstance(sql, str):
            raise TypeError("datasource_id and sql must use the registered input contracts")
        if not isinstance(policy, QueryPolicy):
            raise TypeError("policy must be a QueryPolicy")
        return await self._run(
            datasource_id=datasource_id,
            sql=sql,
            policy=policy,
            query_id=query_id or uuid4(),
            prepare=lambda: self._validation_service._validate_raw_registered_sql(
                datasource_id,
                sql,
                policy=policy,
                max_chars=max_chars,
            ),
        )


__all__ = [
    "AsyncExecutionConnection",
    "AsyncPreparedStatement",
    "ExecutionAdapter",
    "ExecutionAuditRecorder",
    "PostgresExecutionAdapter",
    "SQLExecutionAuditRecord",
    "SQLExecutionOutcome",
    "SQLExecutionService",
    "build_safe_search_path_statement",
    "redact_sql_literals",
]
