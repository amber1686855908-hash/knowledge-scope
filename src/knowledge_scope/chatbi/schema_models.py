"""Provider-independent schema snapshot and semantic context contracts."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import StrEnum
from typing import Final, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from .policy import SQLDialect

SCHEMA_SNAPSHOT_VERSION: Final = "1.0"
SEMANTIC_CONTEXT_VERSION: Final = "1.0"
SCHEMA_FINGERPRINT_PATTERN: Final = r"^[0-9a-f]{64}$"
SCHEMA_COMMENT_MAX_LENGTH: Final = 1_000
SCHEMA_TYPE_MAX_LENGTH: Final = 255


class SchemaContextBudgetError(ValueError):
    """Raised when the required semantic-context envelope cannot fit."""


def _normalize_identifier(value: str, field_name: str = "identifier") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL characters")
    return normalized


def normalize_postgres_type(value: object) -> str:
    """Normalize an adapter-provided PostgreSQL type name without reading values."""
    if not isinstance(value, str):
        raise ValueError("normalized_type must be a string")
    normalized = unicodedata.normalize("NFC", value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    replacements = (
        ("character varying", "varchar"),
        ("timestamp without time zone", "timestamp"),
        ("timestamp with time zone", "timestamptz"),
        ("time without time zone", "time"),
        ("time with time zone", "timetz"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    if not normalized or any(
        unicodedata.category(character).startswith("C") for character in normalized
    ):
        raise ValueError("normalized_type must contain safe non-empty text")
    return normalized


def _normalize_optional_comment(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", value)
    safe_characters = (
        " " if character.isspace() or unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    normalized = " ".join("".join(safe_characters).split())
    return normalized[:SCHEMA_COMMENT_MAX_LENGTH] or None


def _normalize_name_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence of names")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of names") from error
    normalized = tuple(_normalize_identifier(item, field_name) for item in values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SchemaObjectKind(StrEnum):
    """Schema objects exposed to the semantic context."""

    TABLE = "table"
    VIEW = "view"


class SchemaColumn(_SchemaModel):
    """One column with metadata only; defaults and row values are excluded."""

    name: str = Field(min_length=1, max_length=63)
    normalized_type: str = Field(min_length=1, max_length=SCHEMA_TYPE_MAX_LENGTH)
    nullable: StrictBool
    ordinal: StrictInt = Field(ge=1)
    comment: str | None = Field(default=None, max_length=SCHEMA_COMMENT_MAX_LENGTH)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_identifier(value, "column name")

    @field_validator("normalized_type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return normalize_postgres_type(value)

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        return _normalize_optional_comment(value)


class SchemaForeignKey(_SchemaModel):
    """A directed source-column to target-column relationship."""

    constraint_name: str = Field(min_length=1, max_length=63)
    source_columns: tuple[str, ...]
    target_schema: str = Field(min_length=1, max_length=63)
    target_relation: str = Field(min_length=1, max_length=63)
    target_columns: tuple[str, ...]

    @field_validator("constraint_name", "target_schema", "target_relation")
    @classmethod
    def normalize_names(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _normalize_identifier(value, str(field_name))

    @field_validator("source_columns", "target_columns", mode="before")
    @classmethod
    def normalize_columns(cls, value: object, info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "columns")
        return _normalize_name_tuple(value, str(field_name))

    @model_validator(mode="after")
    def validate_column_pairs(self) -> Self:
        if len(self.source_columns) != len(self.target_columns):
            raise ValueError("foreign-key source and target columns must have equal length")
        return self


class SchemaUniqueConstraint(_SchemaModel):
    """A named unique constraint and its ordered columns."""

    constraint_name: str = Field(min_length=1, max_length=63)
    columns: tuple[str, ...]

    @field_validator("constraint_name")
    @classmethod
    def normalize_constraint_name(cls, value: str) -> str:
        return _normalize_identifier(value, "constraint name")

    @field_validator("columns", mode="before")
    @classmethod
    def normalize_constraint_columns(cls, value: object) -> tuple[str, ...]:
        return _normalize_name_tuple(value, "constraint columns")


class SchemaRelation(_SchemaModel):
    """One table or view and its bounded structural metadata."""

    schema_name: str = Field(min_length=1, max_length=63)
    name: str = Field(min_length=1, max_length=63)
    kind: SchemaObjectKind
    comment: str | None = Field(default=None, max_length=SCHEMA_COMMENT_MAX_LENGTH)
    columns: tuple[SchemaColumn, ...] = ()
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[SchemaForeignKey, ...] = ()
    unique_constraints: tuple[SchemaUniqueConstraint, ...] = ()

    @field_validator("schema_name", "name")
    @classmethod
    def normalize_relation_names(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "relation name")
        return _normalize_identifier(value, str(field_name))

    @field_validator("comment")
    @classmethod
    def normalize_relation_comment(cls, value: str | None) -> str | None:
        return _normalize_optional_comment(value)

    @field_validator("primary_key", mode="before")
    @classmethod
    def normalize_primary_key(cls, value: object) -> tuple[str, ...]:
        if value in (None, (), []):
            return ()
        return _normalize_name_tuple(value, "primary_key")

    @model_validator(mode="after")
    def validate_and_order(self) -> Self:
        column_names = tuple(column.name for column in self.columns)
        if len(set(column_names)) != len(column_names):
            raise ValueError("relation columns must have unique names")
        ordinals = tuple(column.ordinal for column in self.columns)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("relation columns must have unique ordinals")
        if any(name not in column_names for name in self.primary_key):
            raise ValueError("primary-key columns must belong to the relation")
        unique_names = tuple(item.constraint_name for item in self.unique_constraints)
        if len(set(unique_names)) != len(unique_names):
            raise ValueError("unique constraint names must be unique per relation")
        for constraint in self.unique_constraints:
            if any(name not in column_names for name in constraint.columns):
                raise ValueError("unique-constraint columns must belong to the relation")
        for foreign_key in self.foreign_keys:
            if any(name not in column_names for name in foreign_key.source_columns):
                raise ValueError("foreign-key source columns must belong to the relation")
        object.__setattr__(
            self,
            "columns",
            tuple(sorted(self.columns, key=lambda item: item.ordinal)),
        )
        object.__setattr__(
            self,
            "foreign_keys",
            tuple(
                sorted(
                    self.foreign_keys,
                    key=lambda item: (
                        item.constraint_name,
                        item.target_schema,
                        item.target_relation,
                        item.source_columns,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "unique_constraints",
            tuple(sorted(self.unique_constraints, key=lambda item: item.constraint_name)),
        )
        return self


class SchemaSnapshot(_SchemaModel):
    """Immutable, deterministic schema metadata for one registered datasource."""

    schema_version: Literal["1.0"] = SCHEMA_SNAPSHOT_VERSION
    datasource_id: UUID
    dialect: SQLDialect
    database_name: str = Field(min_length=1, max_length=63)
    schemas: tuple[str, ...]
    relations: tuple[SchemaRelation, ...]

    @field_validator("database_name")
    @classmethod
    def normalize_database_name(cls, value: str) -> str:
        return _normalize_identifier(value, "database name")

    @field_validator("schemas", mode="before")
    @classmethod
    def normalize_schema_names(cls, value: object) -> tuple[str, ...]:
        normalized = _normalize_name_tuple(value, "schemas")
        if len(set(normalized)) != len(normalized):
            raise ValueError("schemas must be unique")
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def validate_and_order(self) -> Self:
        relation_keys = tuple(
            (relation.schema_name, relation.name, relation.kind.value)
            for relation in self.relations
        )
        if len(set(relation_keys)) != len(relation_keys):
            raise ValueError("schema relations must be unique")
        allowed_schemas = set(self.schemas)
        if any(relation.schema_name not in allowed_schemas for relation in self.relations):
            raise ValueError("every relation must belong to a discovered schema")
        object.__setattr__(
            self,
            "relations",
            tuple(
                sorted(
                    self.relations,
                    key=lambda item: (item.schema_name, item.name, item.kind.value),
                )
            ),
        )
        return self

    def to_deterministic_json(self) -> str:
        """Serialize only stable metadata with fixed JSON settings."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def fingerprint(self) -> str:
        """Return the content fingerprint used for schema drift detection."""
        return hashlib.sha256(self.to_deterministic_json().encode("utf-8")).hexdigest()


class SemanticSchemaContext(_SchemaModel):
    """Bounded deterministic text plus explicit whole-relation omissions."""

    schema_version: Literal["1.0"] = SEMANTIC_CONTEXT_VERSION
    snapshot_fingerprint: str = Field(pattern=SCHEMA_FINGERPRINT_PATTERN)
    text: str = Field(min_length=1)
    max_chars: StrictInt = Field(ge=1)
    included_relations: tuple[str, ...] = ()
    omitted_relations: tuple[str, ...] = ()
    omitted_relationships: tuple[str, ...] = ()
    truncated: StrictBool = False

    @model_validator(mode="after")
    def validate_budget_metadata(self) -> Self:
        if len(self.text) > self.max_chars:
            raise ValueError("semantic schema context exceeds its character budget")
        if len(set(self.included_relations)) != len(self.included_relations):
            raise ValueError("included relations must be unique")
        if len(set(self.omitted_relations)) != len(self.omitted_relations):
            raise ValueError("omitted relations must be unique")
        if len(set(self.omitted_relationships)) != len(self.omitted_relationships):
            raise ValueError("omitted relationships must be unique")
        has_omissions = bool(self.omitted_relations or self.omitted_relationships)
        if self.truncated is not has_omissions:
            raise ValueError("truncated must match the omission metadata")
        if set(self.included_relations) & set(self.omitted_relations):
            raise ValueError("a relation cannot be both included and omitted")
        return self


class SchemaDiscoveryResult(_SchemaModel):
    """Safe discovery response; credentials are intentionally absent."""

    schema_version: Literal["1.0"] = SEMANTIC_CONTEXT_VERSION
    snapshot: SchemaSnapshot
    fingerprint: str = Field(pattern=SCHEMA_FINGERPRINT_PATTERN)
    context: SemanticSchemaContext

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        if self.fingerprint != self.snapshot.fingerprint:
            raise ValueError("discovery fingerprint must match the snapshot")
        if self.context.snapshot_fingerprint != self.fingerprint:
            raise ValueError("context fingerprint must match the snapshot")
        return self


def _relation_label(relation: SchemaRelation) -> str:
    return f"{relation.schema_name}.{relation.name}"


def _foreign_key_label(relation: SchemaRelation, foreign_key: SchemaForeignKey) -> str:
    return (
        f"{_relation_label(relation)}.{foreign_key.constraint_name}->"
        f"{foreign_key.target_schema}.{foreign_key.target_relation}"
    )


def _render_foreign_key(relation: SchemaRelation, foreign_key: SchemaForeignKey) -> str:
    source = ", ".join(foreign_key.source_columns)
    target = ", ".join(foreign_key.target_columns)
    return (
        f"Foreign key {foreign_key.constraint_name}: {source} -> "
        f"{foreign_key.target_schema}.{foreign_key.target_relation}({target})"
    )


def _render_relation(
    relation: SchemaRelation,
    *,
    foreign_keys: tuple[SchemaForeignKey, ...] | None = None,
    include_comments: bool = True,
) -> str:
    lines = [f"{relation.kind.value.title()}: {_relation_label(relation)}"]
    if include_comments and relation.comment:
        lines.append(f"Comment: {relation.comment}")
    for column in relation.columns:
        nullability = "NULL" if column.nullable else "NOT NULL"
        line = f"- {column.name}: {column.normalized_type} {nullability}"
        if include_comments and column.comment:
            line += f" — {column.comment}"
        lines.append(line)
    if relation.primary_key:
        lines.append(f"Primary key: {', '.join(relation.primary_key)}")
    for constraint in relation.unique_constraints:
        lines.append(f"Unique {constraint.constraint_name}: {', '.join(constraint.columns)}")
    for foreign_key in relation.foreign_keys if foreign_keys is None else foreign_keys:
        lines.append(_render_foreign_key(relation, foreign_key))
    return "\n".join(lines)


def build_semantic_schema_context(
    snapshot: SchemaSnapshot,
    *,
    max_chars: int,
) -> SemanticSchemaContext:
    """Render whole relation blocks and endpoint-safe relationships within a character budget."""
    if max_chars < 1:
        raise SchemaContextBudgetError("schema context max_chars must be positive")

    header = (
        f"Database: {snapshot.database_name}\n"
        f"Dialect: {snapshot.dialect.value}\n"
        f"Schemas: {', '.join(snapshot.schemas)}"
    )
    if len(header) > max_chars:
        raise SchemaContextBudgetError(
            "schema context budget is smaller than the required context envelope"
        )

    text = header
    included: list[str] = []
    omitted: list[str] = []
    for relation in snapshot.relations:
        label = _relation_label(relation)
        # Select complete table definitions first.  Foreign keys are added only
        # after both endpoint tables have been selected.
        block = _render_relation(relation, foreign_keys=())
        candidate = f"{text}\n\n{block}"
        if len(candidate) <= max_chars:
            text = candidate
            included.append(label)
        else:
            omitted.append(label)

    included_set = set(included)
    omitted_relationships: list[str] = []
    relationship_started = False
    for relation in snapshot.relations:
        source_label = _relation_label(relation)
        for foreign_key in relation.foreign_keys:
            relationship_label = _foreign_key_label(relation, foreign_key)
            target_label = f"{foreign_key.target_schema}.{foreign_key.target_relation}"
            if source_label not in included_set or target_label not in included_set:
                omitted_relationships.append(relationship_label)
                continue
            relationship_line = _render_foreign_key(relation, foreign_key)
            separator = "\n" if relationship_started else "\n\nRelationships:\n"
            candidate = f"{text}{separator}{relationship_line}"
            if len(candidate) <= max_chars:
                text = candidate
                relationship_started = True
            else:
                omitted_relationships.append(relationship_label)

    return SemanticSchemaContext(
        snapshot_fingerprint=snapshot.fingerprint,
        text=text,
        max_chars=max_chars,
        included_relations=tuple(included),
        omitted_relations=tuple(omitted),
        omitted_relationships=tuple(omitted_relationships),
        truncated=bool(omitted or omitted_relationships),
    )


def render_structural_schema_context(
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    *,
    allowed_schemas: tuple[str, ...] = (),
) -> str:
    """Render the approved context as deterministic, escaped JSON data.

    Comments remain part of the discovery snapshot and its fingerprint, but
    are not sent to the NL2SQL provider.  The selected relations,
    relationships, and policy allow-list are taken from the already budgeted
    context so the prompt cannot expand the approved object set.  Delimiter-like
    characters are escaped after JSON encoding so identifiers remain data inside
    the prompt's schema envelope.
    """
    included = set(context.included_relations)
    known = {_relation_label(relation) for relation in snapshot.relations}
    if not included <= known:
        raise SchemaContextBudgetError("semantic context contains an unknown relation")

    omitted_relationships = set(context.omitted_relationships)
    relationships: list[dict[str, object]] = []
    relation_payloads: list[dict[str, object]] = []
    for relation in snapshot.relations:
        source_label = _relation_label(relation)
        if source_label not in included:
            continue
        relation_payloads.append(
            {
                "columns": [
                    {
                        "name": column.name,
                        "nullable": column.nullable,
                        "type": column.normalized_type,
                    }
                    for column in relation.columns
                ],
                "kind": relation.kind.value,
                "name": relation.name,
                "primary_key": list(relation.primary_key),
                "schema": relation.schema_name,
                "unique_constraints": [
                    {
                        "columns": list(constraint.columns),
                        "name": constraint.constraint_name,
                    }
                    for constraint in relation.unique_constraints
                ],
            }
        )
        for foreign_key in relation.foreign_keys:
            target_label = f"{foreign_key.target_schema}.{foreign_key.target_relation}"
            relationship_label = _foreign_key_label(relation, foreign_key)
            if target_label in included and relationship_label not in omitted_relationships:
                relationships.append(
                    {
                        "constraint": foreign_key.constraint_name,
                        "source_columns": list(foreign_key.source_columns),
                        "source_relation": source_label,
                        "target_columns": list(foreign_key.target_columns),
                        "target_relation": target_label,
                    }
                )

    payload = {
        "allowed_schemas": list(allowed_schemas),
        "database": snapshot.database_name,
        "dialect": snapshot.dialect.value,
        "relations": relation_payloads,
        "relationships": relationships,
        "schemas": list(snapshot.schemas),
    }
    text = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).translate(
        str.maketrans(
            {
                "<": r"\u003c",
                ">": r"\u003e",
                "&": r"\u0026",
                "`": r"\u0060",
            }
        )
    )
    if len(text) > context.max_chars:
        raise SchemaContextBudgetError("structural schema context exceeds its character budget")
    return text


__all__ = [
    "SCHEMA_COMMENT_MAX_LENGTH",
    "SCHEMA_SNAPSHOT_VERSION",
    "SCHEMA_TYPE_MAX_LENGTH",
    "SEMANTIC_CONTEXT_VERSION",
    "SchemaColumn",
    "SchemaContextBudgetError",
    "SchemaDiscoveryResult",
    "SchemaForeignKey",
    "SchemaObjectKind",
    "SchemaRelation",
    "SchemaSnapshot",
    "SchemaUniqueConstraint",
    "SemanticSchemaContext",
    "build_semantic_schema_context",
    "normalize_postgres_type",
    "render_structural_schema_context",
]
