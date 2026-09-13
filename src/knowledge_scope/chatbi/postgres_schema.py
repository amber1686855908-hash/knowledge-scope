"""Read-only PostgreSQL schema discovery adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol, cast
from uuid import UUID

import asyncpg

from .credentials import ResolvedDatabaseCredentials
from .errors import ChatBIError, ChatBIErrorCategory
from .policy import SQLDialect
from .schema_models import (
    SchemaColumn,
    SchemaForeignKey,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SchemaUniqueConstraint,
)

SYSTEM_SCHEMAS = frozenset({"information_schema", "pg_catalog", "pg_temp", "pg_toast"})
SYSTEM_SCHEMA_PREFIXES = ("pg_toast_", "pg_temp_")


def is_system_schema(schema_name: str) -> bool:
    """Return whether a PostgreSQL catalog or temporary schema is internal."""
    normalized = schema_name.strip().casefold()
    return normalized in SYSTEM_SCHEMAS or normalized.startswith(SYSTEM_SCHEMA_PREFIXES)


_DATABASE_QUERY = "SELECT current_database() AS database_name"
_RELATIONS_QUERY = """
SELECT
    n.nspname AS schema_name,
    c.relname AS relation_name,
    CASE WHEN c.relkind::text IN ('v', 'm') THEN 'view' ELSE 'table' END AS relation_kind,
    pg_catalog.obj_description(c.oid, 'pg_class') AS comment
FROM pg_catalog.pg_class AS c
JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = ANY($1::text[])
  AND c.relkind::text = ANY($2::text[])
ORDER BY n.nspname, c.relname, c.relkind::text
"""
_COLUMNS_QUERY = """
SELECT
    n.nspname AS schema_name,
    c.relname AS relation_name,
    a.attnum AS ordinal,
    a.attname AS column_name,
    pg_catalog.format_type(a.atttypid, a.atttypmod) AS normalized_type,
    NOT a.attnotnull AS nullable,
    pg_catalog.col_description(c.oid, a.attnum) AS comment
FROM pg_catalog.pg_attribute AS a
JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = ANY($1::text[])
  AND c.relkind::text = ANY($2::text[])
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY n.nspname, c.relname, a.attnum
"""
_CONSTRAINTS_QUERY = """
SELECT
    src_n.nspname AS schema_name,
    src_c.relname AS relation_name,
    con.conname AS constraint_name,
    con.contype::text AS constraint_type,
    con.conkey AS source_attnums,
    target_n.nspname AS target_schema,
    target_c.relname AS target_relation,
    con.confkey AS target_attnums
FROM pg_catalog.pg_constraint AS con
JOIN pg_catalog.pg_class AS src_c ON src_c.oid = con.conrelid
JOIN pg_catalog.pg_namespace AS src_n ON src_n.oid = src_c.relnamespace
LEFT JOIN pg_catalog.pg_class AS target_c ON target_c.oid = con.confrelid
LEFT JOIN pg_catalog.pg_namespace AS target_n ON target_n.oid = target_c.relnamespace
WHERE src_n.nspname = ANY($1::text[])
  AND con.contype IN ('p', 'u', 'f')
ORDER BY src_n.nspname, src_c.relname, con.conname
"""


class AsyncMetadataConnection(Protocol):
    """Small connection surface required by the adapter and its test doubles."""

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        """Fetch metadata rows."""

    def transaction(self, *, readonly: bool = False) -> AbstractAsyncContextManager[object]:
        """Open a transaction with an explicit read-only flag."""

    async def close(self) -> None:
        """Close the connection."""


ConnectionFactory = Callable[
    [ResolvedDatabaseCredentials, float, int], Awaitable[AsyncMetadataConnection]
]


async def _connect_postgres(
    credentials: ResolvedDatabaseCredentials,
    connection_timeout_seconds: float,
    statement_timeout_ms: int,
) -> AsyncMetadataConnection:
    return await asyncpg.connect(
        dsn=credentials.connection_url,
        timeout=connection_timeout_seconds,
        server_settings={
            "default_transaction_read_only": "on",
            "statement_timeout": str(statement_timeout_ms),
        },
    )


def _safe_error(message: str) -> ChatBIError:
    return ChatBIError(ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED, message)


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _safe_error("PostgreSQL returned incomplete schema metadata")
    return value


def _attnums(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise _safe_error("PostgreSQL returned incomplete constraint metadata")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise _safe_error("PostgreSQL returned incomplete constraint metadata") from None
    if not values or any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise _safe_error("PostgreSQL returned incomplete constraint metadata")
    return values


def _normalize_allowed_schemas(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise _safe_error("schema allow-list must be a sequence of names")
    try:
        requested = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise _safe_error("schema allow-list must be a sequence of names") from None
    if any(not isinstance(schema, str) for schema in requested):
        raise _safe_error("schema allow-list must contain only string names")
    schemas = tuple(sorted({schema.strip() for schema in requested if schema.strip()}))
    return tuple(schema for schema in schemas if not is_system_schema(schema))


class PostgresSchemaInspector:
    """Discover only allow-listed PostgreSQL catalog metadata in a read-only transaction."""

    def __init__(
        self,
        *,
        connection_timeout_seconds: float = 10.0,
        statement_timeout_ms: int = 30_000,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self._connection_timeout_seconds = connection_timeout_seconds
        self._statement_timeout_ms = statement_timeout_ms
        self._connection_factory = connection_factory or _connect_postgres

    async def discover(
        self,
        credentials: ResolvedDatabaseCredentials,
        *,
        datasource_id: UUID,
        dialect: SQLDialect,
        allowed_schemas: tuple[str, ...],
        allow_views: bool,
    ) -> SchemaSnapshot:
        """Return a normalized snapshot without reading business rows or defaults."""
        if dialect is not SQLDialect.POSTGRESQL:
            raise ChatBIError(
                ChatBIErrorCategory.UNSUPPORTED_DIALECT,
                "schema discovery currently supports PostgreSQL only",
            )
        schemas = _normalize_allowed_schemas(allowed_schemas)
        if not schemas:
            raise _safe_error("schema discovery requires at least one non-system schema")
        relation_kinds = ("r", "p", "v", "m") if allow_views else ("r", "p")

        connection: AsyncMetadataConnection | None = None
        try:
            connection = await self._connection_factory(
                credentials,
                self._connection_timeout_seconds,
                self._statement_timeout_ms,
            )
            async with connection.transaction(readonly=True):
                database_rows = await connection.fetch(_DATABASE_QUERY)
                relation_rows = await connection.fetch(
                    _RELATIONS_QUERY,
                    list(schemas),
                    list(relation_kinds),
                )
                column_rows = await connection.fetch(
                    _COLUMNS_QUERY,
                    list(schemas),
                    list(relation_kinds),
                )
                constraint_rows = await connection.fetch(_CONSTRAINTS_QUERY, list(schemas))

            return self._build_snapshot(
                datasource_id=datasource_id,
                dialect=dialect,
                schemas=schemas,
                allow_views=allow_views,
                database_rows=database_rows,
                relation_rows=relation_rows,
                column_rows=column_rows,
                constraint_rows=constraint_rows,
            )
        except asyncio.CancelledError:
            raise
        except ChatBIError:
            raise
        except Exception:
            raise _safe_error("PostgreSQL schema discovery failed") from None
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

    @staticmethod
    def _build_snapshot(
        *,
        datasource_id: UUID,
        dialect: SQLDialect,
        schemas: tuple[str, ...],
        allow_views: bool,
        database_rows: list[Mapping[str, object]],
        relation_rows: list[Mapping[str, object]],
        column_rows: list[Mapping[str, object]],
        constraint_rows: list[Mapping[str, object]],
    ) -> SchemaSnapshot:
        if len(database_rows) != 1:
            raise _safe_error("PostgreSQL returned incomplete database metadata")
        database_name = _required_text(database_rows[0], "database_name")
        allowed = set(schemas)

        relation_kinds: dict[tuple[str, str], SchemaObjectKind] = {}
        relation_comments: dict[tuple[str, str], str | None] = {}
        relation_columns: dict[tuple[str, str], list[SchemaColumn]] = {}
        for row in relation_rows:
            schema_name = _required_text(row, "schema_name")
            relation_name = _required_text(row, "relation_name")
            relation_kind = _required_text(row, "relation_kind")
            if schema_name not in allowed or is_system_schema(schema_name):
                continue
            if relation_kind == SchemaObjectKind.VIEW.value and not allow_views:
                continue
            try:
                kind = SchemaObjectKind(relation_kind)
            except ValueError:
                raise _safe_error("PostgreSQL returned an unknown relation kind") from None
            key = (schema_name, relation_name)
            if key in relation_kinds:
                raise _safe_error("PostgreSQL returned duplicate relation metadata")
            comment = row.get("comment")
            if comment is not None and not isinstance(comment, str):
                raise _safe_error("PostgreSQL returned invalid relation comment metadata")
            relation_kinds[key] = kind
            relation_comments[key] = cast(str | None, comment)
            relation_columns[key] = []

        for row in column_rows:
            key = (_required_text(row, "schema_name"), _required_text(row, "relation_name"))
            if key not in relation_kinds:
                continue
            ordinal = row.get("ordinal")
            nullable = row.get("nullable")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or not isinstance(nullable, bool)
            ):
                raise _safe_error("PostgreSQL returned invalid column metadata")
            relation_columns[key].append(
                SchemaColumn(
                    name=_required_text(row, "column_name"),
                    normalized_type=_required_text(row, "normalized_type"),
                    nullable=nullable,
                    ordinal=ordinal,
                    comment=cast(str | None, row.get("comment")),
                )
            )

        column_names = {
            (schema_name, relation_name, column.ordinal): column.name
            for (schema_name, relation_name), columns in relation_columns.items()
            for column in columns
        }
        primary_keys: dict[tuple[str, str], tuple[str, ...]] = {}
        foreign_keys: dict[tuple[str, str], list[SchemaForeignKey]] = {
            key: [] for key in relation_kinds
        }
        unique_constraints: dict[tuple[str, str], list[SchemaUniqueConstraint]] = {
            key: [] for key in relation_kinds
        }
        for row in constraint_rows:
            source_key = (_required_text(row, "schema_name"), _required_text(row, "relation_name"))
            if source_key not in relation_kinds:
                continue
            constraint_type = _required_text(row, "constraint_type")
            source_column_names = tuple(
                column_names[(source_key[0], source_key[1], ordinal)]
                for ordinal in _attnums(row.get("source_attnums"))
            )
            constraint_name = _required_text(row, "constraint_name")
            if constraint_type == "p":
                if source_key in primary_keys:
                    raise _safe_error("PostgreSQL returned duplicate primary-key metadata")
                primary_keys[source_key] = source_column_names
            elif constraint_type == "u":
                unique_constraints[source_key].append(
                    SchemaUniqueConstraint(
                        constraint_name=constraint_name,
                        columns=source_column_names,
                    )
                )
            elif constraint_type == "f":
                target_schema = _required_text(row, "target_schema")
                target_relation = _required_text(row, "target_relation")
                target_key = (target_schema, target_relation)
                # A foreign key whose target is outside the allow-list is not
                # inspected or emitted as partial out-of-scope metadata.
                if target_schema not in allowed or target_key not in relation_kinds:
                    continue
                target_column_names = tuple(
                    column_names[(target_schema, target_relation, ordinal)]
                    for ordinal in _attnums(row.get("target_attnums"))
                )
                foreign_keys[source_key].append(
                    SchemaForeignKey(
                        constraint_name=constraint_name,
                        source_columns=source_column_names,
                        target_schema=target_schema,
                        target_relation=target_relation,
                        target_columns=target_column_names,
                    )
                )
            else:
                raise _safe_error("PostgreSQL returned an unsupported constraint type")

        relations = tuple(
            SchemaRelation(
                schema_name=schema_name,
                name=relation_name,
                kind=kind,
                comment=relation_comments[(schema_name, relation_name)],
                columns=tuple(relation_columns[(schema_name, relation_name)]),
                primary_key=primary_keys.get((schema_name, relation_name), ()),
                foreign_keys=tuple(foreign_keys[(schema_name, relation_name)]),
                unique_constraints=tuple(unique_constraints[(schema_name, relation_name)]),
            )
            for (schema_name, relation_name), kind in relation_kinds.items()
        )
        return SchemaSnapshot(
            datasource_id=datasource_id,
            dialect=dialect,
            database_name=database_name,
            schemas=schemas,
            relations=relations,
        )


__all__ = [
    "SYSTEM_SCHEMAS",
    "SYSTEM_SCHEMA_PREFIXES",
    "AsyncMetadataConnection",
    "ConnectionFactory",
    "PostgresSchemaInspector",
    "is_system_schema",
]
