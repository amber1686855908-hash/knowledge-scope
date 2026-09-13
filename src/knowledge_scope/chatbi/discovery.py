"""Schema discovery orchestration for registered ChatBI datasources."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .credentials import (
    CredentialResolver,
    EnvironmentCredentialResolver,
    ResolvedDatabaseCredentials,
)
from .errors import ChatBIError, ChatBIErrorCategory
from .policy import QueryPolicy, SQLDialect
from .postgres_schema import PostgresSchemaInspector
from .schema_models import (
    SchemaContextBudgetError,
    SchemaDiscoveryResult,
    SchemaSnapshot,
    build_semantic_schema_context,
)
from .schemas import DataSource


class SchemaDiscoveryAdapter(Protocol):
    """Provider-independent adapter contract for metadata discovery."""

    async def discover(
        self,
        credentials: ResolvedDatabaseCredentials,
        *,
        datasource_id: UUID,
        dialect: SQLDialect,
        allowed_schemas: tuple[str, ...],
        allow_views: bool,
    ) -> SchemaSnapshot:
        """Discover a normalized snapshot from one external datasource."""


class SchemaDiscoveryService:
    """Resolve a registered datasource and build its safe semantic schema context."""

    def __init__(
        self,
        credential_resolver: CredentialResolver,
        adapter: SchemaDiscoveryAdapter,
    ) -> None:
        self._credential_resolver = credential_resolver
        self._adapter = adapter

    async def discover(
        self,
        data_source: DataSource,
        policy: QueryPolicy,
        *,
        max_chars: int,
    ) -> SchemaDiscoveryResult:
        """Perform one bounded, read-only discovery operation."""
        if not data_source.enabled:
            raise ChatBIError(
                ChatBIErrorCategory.DATASOURCE_DISABLED,
                "the data source is disabled",
            )
        if data_source.dialect is not SQLDialect.POSTGRESQL:
            raise ChatBIError(
                ChatBIErrorCategory.UNSUPPORTED_DIALECT,
                "schema discovery currently supports PostgreSQL only",
            )
        if max_chars < 1:
            raise ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "schema context budget must be positive",
            )

        credentials = self._credential_resolver.resolve(data_source.connection_ref)
        allowed_schemas = tuple(
            schema for schema in policy.allowed_schemas if schema not in policy.denied_schemas
        )
        snapshot = await self._adapter.discover(
            credentials,
            datasource_id=data_source.id,
            dialect=data_source.dialect,
            allowed_schemas=allowed_schemas,
            allow_views=policy.allow_views,
        )
        allowed_schema_set = set(allowed_schemas)
        if any(schema not in allowed_schema_set for schema in snapshot.schemas) or any(
            relation.schema_name not in allowed_schema_set for relation in snapshot.relations
        ):
            raise ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "schema discovery returned data outside the configured schema allow-list",
            )
        if snapshot.datasource_id != data_source.id or snapshot.dialect is not data_source.dialect:
            raise ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "schema discovery returned an inconsistent datasource identity",
            )
        if data_source.default_database and snapshot.database_name != data_source.default_database:
            raise ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "connected database does not match the datasource default database",
            )
        fingerprint = snapshot.fingerprint
        try:
            context = build_semantic_schema_context(snapshot, max_chars=max_chars)
        except SchemaContextBudgetError:
            raise ChatBIError(
                ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED,
                "schema context budget is too small for the required context envelope",
            ) from None
        return SchemaDiscoveryResult(
            snapshot=snapshot,
            fingerprint=fingerprint,
            context=context,
        )


def create_postgres_schema_discovery_service(
    *,
    connection_timeout_seconds: float,
    statement_timeout_ms: int,
) -> SchemaDiscoveryService:
    """Build the production PostgreSQL discovery service with bounded timeouts."""
    return SchemaDiscoveryService(
        credential_resolver=EnvironmentCredentialResolver(),
        adapter=PostgresSchemaInspector(
            connection_timeout_seconds=connection_timeout_seconds,
            statement_timeout_ms=statement_timeout_ms,
        ),
    )


__all__ = [
    "SchemaDiscoveryAdapter",
    "SchemaDiscoveryService",
    "create_postgres_schema_discovery_service",
]
