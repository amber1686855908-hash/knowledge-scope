"""Provider-independent ChatBI domain and execution contracts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .errors import ChatBIErrorCategory
from .policy import QueryPolicy, SQLDialect

CONNECTION_REF_MAX_LENGTH: Final = 255
DATASOURCE_DISPLAY_NAME_MAX_LENGTH: Final = 200
POSTGRES_IDENTIFIER_MAX_LENGTH: Final = 63
SQL_TEXT_MAX_LENGTH: Final = 100_000

# A connection_ref identifies a separately managed environment/secret entry; it
# is deliberately not a database URL and therefore cannot contain a password.
_CONNECTION_REF_PATTERN = re.compile(r"^(?:env|secret):[A-Za-z][A-Za-z0-9_.:/-]{0,247}$")

type ScalarValue = str | int | float | bool | None


def _trimmed_required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    return normalized


def _trimmed_optional(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def validate_connection_ref(value: str) -> str:
    normalized = _trimmed_required(value, "connection_ref")
    if not _CONNECTION_REF_PATTERN.fullmatch(normalized):
        raise ValueError("connection_ref must use the opaque env:NAME or secret:NAME format")
    return normalized


def _scrub_invalid_connection_ref(value: object) -> object:
    """Avoid echoing a malformed, potentially credential-bearing URL in errors."""
    if not isinstance(value, str) or not _CONNECTION_REF_PATTERN.fullmatch(value.strip()):
        return "__invalid_chatbi_connection_ref__"
    return value


class _ChatBIModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class QueryLifecycleState(StrEnum):
    """Lifecycle states for a future query audit record."""

    CREATED = "created"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSourceCreate(_ChatBIModel):
    """Input for registering a relational datasource."""

    display_name: str = Field(min_length=1, max_length=DATASOURCE_DISPLAY_NAME_MAX_LENGTH)
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    enabled: StrictBool = True
    connection_ref: str = Field(min_length=1, max_length=CONNECTION_REF_MAX_LENGTH)
    default_database: str | None = Field(default=None, max_length=POSTGRES_IDENTIFIER_MAX_LENGTH)
    default_schema: str | None = Field(default=None, max_length=POSTGRES_IDENTIFIER_MAX_LENGTH)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return _trimmed_required(value, "display_name")

    @field_validator("connection_ref", mode="before")
    @classmethod
    def scrub_invalid_connection_ref(cls, value: object) -> object:
        return _scrub_invalid_connection_ref(value)

    @field_validator("connection_ref")
    @classmethod
    def validate_connection_ref(cls, value: str) -> str:
        return validate_connection_ref(value)

    @field_validator("default_database", "default_schema")
    @classmethod
    def normalize_defaults(cls, value: str | None) -> str | None:
        return _trimmed_optional(value, "default value")


class DataSourceUpdate(_ChatBIModel):
    """Safe metadata updates; credentials and dialect are immutable here."""

    display_name: str | None = Field(default=None, max_length=DATASOURCE_DISPLAY_NAME_MAX_LENGTH)
    enabled: StrictBool | None = None
    default_database: str | None = Field(default=None, max_length=POSTGRES_IDENTIFIER_MAX_LENGTH)
    default_schema: str | None = Field(default=None, max_length=POSTGRES_IDENTIFIER_MAX_LENGTH)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        return _trimmed_required(value, "display_name") if value is not None else None

    @field_validator("default_database", "default_schema")
    @classmethod
    def normalize_defaults(cls, value: str | None) -> str | None:
        return _trimmed_optional(value, "default value")

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        if "display_name" in self.model_fields_set and self.display_name is None:
            raise ValueError("display_name cannot be null")
        if "enabled" in self.model_fields_set and self.enabled is None:
            raise ValueError("enabled cannot be null")
        return self


class DataSource(DataSourceCreate):
    """Internal datasource representation; ``connection_ref`` is never public."""

    id: UUID
    created_at: datetime
    updated_at: datetime


class DataSourcePublic(_ChatBIModel):
    """Safe API representation with no credential or connection reference."""

    id: UUID
    display_name: str
    dialect: SQLDialect
    enabled: StrictBool
    default_database: str | None
    default_schema: str | None
    connection_configured: StrictBool
    created_at: datetime
    updated_at: datetime


class DataSourceListResponse(_ChatBIModel):
    """Bounded datasource metadata listing."""

    items: list[DataSourcePublic]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ColumnMetadata(_ChatBIModel):
    """Driver-independent description of one result column."""

    name: str = Field(min_length=1, max_length=255)
    data_type: str = Field(min_length=1, max_length=255)
    nullable: StrictBool = True
    ordinal: int = Field(ge=0)

    @field_validator("name", "data_type")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _trimmed_required(value, "column metadata")


class QueryExecutionRequest(_ChatBIModel):
    """Future execution input, without implementing SQL execution in A5.1."""

    query_id: UUID = Field(default_factory=uuid4)
    datasource_id: UUID
    sql: str = Field(min_length=1, max_length=SQL_TEXT_MAX_LENGTH)
    parameters: dict[str, ScalarValue] = Field(default_factory=dict)
    policy: QueryPolicy = Field(default_factory=QueryPolicy)
    context_metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("sql")
    @classmethod
    def normalize_sql(cls, value: str) -> str:
        return _trimmed_required(value, "sql")

    @property
    def sql_fingerprint(self) -> str:
        """Return a safe identifier for audit correlation without exposing SQL."""
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


class QueryAuditRecord(_ChatBIModel):
    """Safe lifecycle/audit projection for one future query execution."""

    audit_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    datasource_id: UUID
    sql_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    state: QueryLifecycleState
    truncated: StrictBool | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_category: ChatBIErrorCategory | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_lifecycle_error(self) -> Self:
        """Keep audit lifecycle state and error classification consistent."""
        if (
            self.state
            in {
                QueryLifecycleState.CREATED,
                QueryLifecycleState.VALIDATED,
                QueryLifecycleState.EXECUTING,
            }
            and self.error_category is not None
        ):
            raise ValueError("non-terminal audit states cannot contain an error category")
        if self.state is QueryLifecycleState.SUCCEEDED and self.error_category is not None:
            raise ValueError("successful audit records cannot contain an error category")
        if (
            self.state
            in {
                QueryLifecycleState.REJECTED,
                QueryLifecycleState.FAILED,
                QueryLifecycleState.CANCELLED,
            }
            and self.error_category is None
        ):
            raise ValueError(
                "rejected, failed, and cancelled audit records require an error category"
            )
        return self


class QueryExecutionResult(_ChatBIModel):
    """Normalized tabular result contract for a future execution adapter."""

    query_id: UUID
    datasource_id: UUID
    state: QueryLifecycleState
    columns: list[ColumnMetadata]
    rows: list[list[ScalarValue]]
    row_count: int = Field(ge=0)
    truncated: StrictBool = False
    duration_ms: float = Field(ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_category: ChatBIErrorCategory | None = None
    error_message: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.state not in {
            QueryLifecycleState.SUCCEEDED,
            QueryLifecycleState.FAILED,
            QueryLifecycleState.CANCELLED,
        }:
            raise ValueError("execution results must use a terminal lifecycle state")
        if self.row_count != len(self.rows):
            raise ValueError("row_count must equal the number of materialized rows")
        expected_width = len(self.columns)
        if any(len(row) != expected_width for row in self.rows):
            raise ValueError("every row must match the column count")
        if self.state is QueryLifecycleState.SUCCEEDED and (
            self.error_category is not None or self.error_message is not None
        ):
            raise ValueError("successful results cannot contain an error")
        if self.state is QueryLifecycleState.FAILED and self.error_category is None:
            raise ValueError("failed results must include an error category")
        if self.error_message is not None and not self.error_message.strip():
            raise ValueError("error_message must not be blank")
        return self

    def to_deterministic_json(self) -> str:
        """Serialize the result with stable keys and separators for audit/tests."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
