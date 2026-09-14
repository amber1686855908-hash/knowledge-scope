"""Conservative policy contracts for future read-only SQL execution."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from knowledge_scope.shared.config import Settings


class SQLDialect(StrEnum):
    """SQL dialects understood by the current foundation."""

    POSTGRESQL = "postgresql"


SUPPORTED_SQL_DIALECTS: Final = frozenset({SQLDialect.POSTGRESQL})
DEFAULT_CHATBI_ALLOWED_SCHEMAS: Final = ("public",)

# These are documented invariants for A5.3's real validator, not a parser.
FORBIDDEN_SQL_COMMANDS: Final = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COPY",
)
FORBIDDEN_SQL_TRANSACTION_CONTROL: Final = (
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE",
)


class QueryPolicy(BaseModel):
    """The bounded, read-only policy future SQL execution must honor."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dialect: SQLDialect = SQLDialect.POSTGRESQL
    read_only: StrictBool = True
    max_rows: StrictInt = Field(default=1_000, ge=1, le=100_000)
    max_result_bytes: StrictInt = Field(default=4_000_000, ge=2, le=100_000_000)
    max_cell_bytes: StrictInt = Field(default=1_000_000, ge=1, le=100_000_000)
    max_nested_value_depth: StrictInt = Field(default=32, ge=1, le=256)
    max_collection_items: StrictInt = Field(default=10_000, ge=1, le=1_000_000)
    statement_timeout_ms: StrictInt = Field(default=30_000, ge=100, le=600_000)
    allowed_schemas: tuple[str, ...] = DEFAULT_CHATBI_ALLOWED_SCHEMAS
    denied_schemas: tuple[str, ...] = ()
    allow_views: StrictBool = False
    max_statement_count: StrictInt = Field(default=1, ge=1, le=1)

    @field_validator("allowed_schemas", "denied_schemas", mode="before")
    @classmethod
    def normalize_schemas(cls, value: object) -> tuple[str, ...]:
        """Require explicit, unique, non-empty schema names."""
        if isinstance(value, str) or value is None:
            raise ValueError("schemas must be provided as a sequence of names")
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("schemas must be provided as a sequence of names") from error
        normalized = tuple(
            item.strip() if isinstance(item, str) else item  # type: ignore[union-attr]
            for item in values
        )
        if any(not isinstance(item, str) or not item for item in normalized):
            raise ValueError("schema names must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError("schema names must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if not self.read_only:
            raise ValueError("ChatBI query policy must be read-only")
        if not self.allowed_schemas:
            raise ValueError("at least one allowed schema is required")
        if set(self.allowed_schemas) & set(self.denied_schemas):
            raise ValueError("allowed and denied schemas cannot overlap")
        if self.dialect not in SUPPORTED_SQL_DIALECTS:
            raise ValueError(f"unsupported SQL dialect: {self.dialect}")
        return self


def default_query_policy(settings: Settings) -> QueryPolicy:
    """Build the policy from validated application settings."""
    return QueryPolicy(
        max_rows=settings.chatbi_max_rows,
        max_result_bytes=settings.chatbi_max_result_bytes,
        max_cell_bytes=settings.chatbi_max_cell_bytes,
        max_nested_value_depth=settings.chatbi_max_nested_value_depth,
        max_collection_items=settings.chatbi_max_collection_items,
        statement_timeout_ms=settings.chatbi_statement_timeout_ms,
        allowed_schemas=tuple(settings.chatbi_allowed_schemas),
        allow_views=settings.chatbi_allow_views,
    )
