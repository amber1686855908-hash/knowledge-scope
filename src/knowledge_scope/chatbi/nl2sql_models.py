"""Provider-independent contracts for safe, non-executing NL2SQL generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from .policy import QueryPolicy, SQLDialect
from .schema_models import SchemaSnapshot, SemanticSchemaContext

NL2SQL_SCHEMA_VERSION: Final = "1.0"
NL2SQL_PROMPT_VERSION: Final = "a5.3-v2"
SQL_VALIDATION_VERSION: Final = "1.0"
NL2SQL_MAX_TOKENS: Final = 16_384
NL2SQL_FINGERPRINT_PATTERN: Final = r"^[0-9a-f]{64}$"


def _trimmed_required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    return normalized


class _NL2SQLModel(BaseModel):
    """Strict JSON-safe model base for the NL2SQL boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class _NL2SQLRequest(_NL2SQLModel):
    """Internal request after trusted schema discovery has completed.

    Production callers should use ``NL2SQLInput`` and let the service obtain a
    fresh snapshot through the registered datasource discovery path.
    """

    datasource_id: UUID
    question: StrictStr = Field(min_length=1, max_length=10_000)
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    schema_snapshot: SchemaSnapshot
    semantic_context: SemanticSchemaContext
    policy: QueryPolicy = Field(default_factory=QueryPolicy)
    model: StrictStr | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return _trimmed_required(value, "question")

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        return _trimmed_required(value, "model") if value is not None else None

    @model_validator(mode="after")
    def validate_context_identity(self) -> _NL2SQLRequest:
        if self.schema_snapshot.datasource_id != self.datasource_id:
            raise ValueError("schema snapshot datasource does not match the request")
        if self.schema_snapshot.dialect is not self.dialect:
            raise ValueError("schema snapshot dialect does not match the request")
        if self.semantic_context.snapshot_fingerprint != self.schema_snapshot.fingerprint:
            raise ValueError("semantic context fingerprint does not match the schema snapshot")
        if self.policy.dialect is not self.dialect:
            raise ValueError("query policy dialect does not match the request")
        return self


class SQLGenerationPayload(_NL2SQLModel):
    """The only model-produced value accepted from a structured response."""

    sql: StrictStr = Field(min_length=1, max_length=100_000)

    @field_validator("sql")
    @classmethod
    def normalize_sql(cls, value: str) -> str:
        return _trimmed_required(value, "sql")


class SQLCandidate(_NL2SQLModel):
    """Application-enriched generated SQL before AST validation."""

    schema_version: Literal["1.0"] = NL2SQL_SCHEMA_VERSION
    datasource_id: UUID
    dialect: SQLDialect
    question: StrictStr = Field(min_length=1, max_length=10_000)
    sql: StrictStr = Field(min_length=1, max_length=100_000)
    context_fingerprint: StrictStr = Field(pattern=NL2SQL_FINGERPRINT_PATTERN)
    provider: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(min_length=1, max_length=255)
    prompt_version: Literal["a5.3-v2"] = NL2SQL_PROMPT_VERSION

    @field_validator("question", "sql")
    @classmethod
    def normalize_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _trimmed_required(value, str(field_name))


class NL2SQLInput(_NL2SQLModel):
    """Public generation input; schema authorization is service-owned."""

    datasource_id: UUID
    question: StrictStr = Field(min_length=1, max_length=10_000)
    model: StrictStr | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return _trimmed_required(value, "question")

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        return _trimmed_required(value, "model") if value is not None else None


# This construction guard is an implementation convenience, not an
# authorization boundary. ``ValidatedSQL`` is not a Pydantic input model and
# therefore has no JSON deserialization path.
_VALIDATION_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedSQL:
    """Internal validation result and audit projection.

    This Python value is not an authorization capability: its private factory
    and fields are not a cryptographic trust boundary. No executor exists in
    this phase; future execution code must re-enter trusted validation from a
    datasource ID and untrusted SQL instead of accepting this value.
    """

    validation_version: str
    datasource_id: UUID
    dialect: SQLDialect
    provider: str
    model: str
    prompt_version: str
    original_sql: str
    normalized_sql: str
    referenced_schemas: tuple[str, ...]
    referenced_relations: tuple[str, ...]
    schema_fingerprint: str
    policy_fingerprint: str
    limit_bounded: bool
    effective_limit: int
    limit_source: str

    def __init__(
        self,
        *,
        validation_version: str,
        datasource_id: UUID,
        dialect: SQLDialect,
        provider: str,
        model: str,
        prompt_version: str,
        original_sql: str,
        normalized_sql: str,
        referenced_schemas: tuple[str, ...],
        referenced_relations: tuple[str, ...],
        schema_fingerprint: str,
        policy_fingerprint: str,
        limit_bounded: bool,
        effective_limit: int,
        limit_source: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _VALIDATION_TOKEN:
            raise TypeError("ValidatedSQL must be created by the internal validation factory")
        object.__setattr__(self, "validation_version", validation_version)
        object.__setattr__(self, "datasource_id", datasource_id)
        object.__setattr__(self, "dialect", dialect)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(self, "original_sql", original_sql)
        object.__setattr__(self, "normalized_sql", normalized_sql)
        object.__setattr__(self, "referenced_schemas", referenced_schemas)
        object.__setattr__(self, "referenced_relations", referenced_relations)
        object.__setattr__(self, "schema_fingerprint", schema_fingerprint)
        object.__setattr__(self, "policy_fingerprint", policy_fingerprint)
        object.__setattr__(self, "limit_bounded", limit_bounded)
        object.__setattr__(self, "effective_limit", effective_limit)
        object.__setattr__(self, "limit_source", limit_source)

    @classmethod
    def _from_validator(cls, **values: object) -> ValidatedSQL:
        return cls(_token=_VALIDATION_TOKEN, **values)  # type: ignore[arg-type]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        """Return an audit projection without exposing a construction path."""
        result: dict[str, object] = {
            "validation_version": self.validation_version,
            "datasource_id": self.datasource_id,
            "dialect": self.dialect,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "original_sql": self.original_sql,
            "normalized_sql": self.normalized_sql,
            "referenced_schemas": self.referenced_schemas,
            "referenced_relations": self.referenced_relations,
            "schema_fingerprint": self.schema_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "limit_bounded": self.limit_bounded,
            "effective_limit": self.effective_limit,
            "limit_source": self.limit_source,
        }
        if mode == "json":
            result["datasource_id"] = str(self.datasource_id)
            result["dialect"] = self.dialect.value
            result["referenced_schemas"] = list(self.referenced_schemas)
            result["referenced_relations"] = list(self.referenced_relations)
        return result


@dataclass(frozen=True, slots=True)
class NL2SQLResult:
    """Output containing an auditable candidate and validation result."""

    candidate: SQLCandidate
    validated_sql: ValidatedSQL

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SQLCandidate) or not isinstance(
            self.validated_sql, ValidatedSQL
        ):
            raise TypeError("NL2SQLResult requires a SQLCandidate and validation result")

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "candidate": self.candidate.model_dump(mode=mode),
            "validated_sql": self.validated_sql.model_dump(mode=mode),
        }


__all__ = [
    "NL2SQL_FINGERPRINT_PATTERN",
    "NL2SQL_MAX_TOKENS",
    "NL2SQL_PROMPT_VERSION",
    "NL2SQL_SCHEMA_VERSION",
    "SQL_VALIDATION_VERSION",
    "NL2SQLInput",
    "NL2SQLResult",
    "SQLCandidate",
    "SQLGenerationPayload",
]
