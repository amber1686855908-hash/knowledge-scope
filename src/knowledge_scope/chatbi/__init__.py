"""ChatBI / NL2SQL domain contracts and datasource metadata foundation."""

from .credentials import (
    CredentialResolver,
    EnvironmentCredentialResolver,
    ResolvedDatabaseCredentials,
    SecretReferenceResolver,
)
from .discovery import (
    SchemaDiscoveryAdapter,
    SchemaDiscoveryService,
    create_postgres_schema_discovery_service,
)
from .errors import ChatBIError, ChatBIErrorCategory
from .policy import QueryPolicy, SQLDialect, default_query_policy
from .schema_models import (
    SchemaColumn,
    SchemaContextBudgetError,
    SchemaDiscoveryResult,
    SchemaForeignKey,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SchemaUniqueConstraint,
    SemanticSchemaContext,
    build_semantic_schema_context,
    normalize_postgres_type,
)
from .schemas import (
    ColumnMetadata,
    DataSource,
    DataSourceCreate,
    DataSourcePublic,
    DataSourceUpdate,
    QueryAuditRecord,
    QueryExecutionRequest,
    QueryExecutionResult,
    QueryLifecycleState,
    validate_connection_ref,
)

__all__ = [
    "ChatBIError",
    "ChatBIErrorCategory",
    "ColumnMetadata",
    "CredentialResolver",
    "DataSource",
    "DataSourceCreate",
    "DataSourcePublic",
    "DataSourceUpdate",
    "EnvironmentCredentialResolver",
    "QueryAuditRecord",
    "QueryExecutionRequest",
    "QueryExecutionResult",
    "QueryLifecycleState",
    "QueryPolicy",
    "ResolvedDatabaseCredentials",
    "SQLDialect",
    "SchemaColumn",
    "SchemaContextBudgetError",
    "SchemaDiscoveryAdapter",
    "SchemaDiscoveryResult",
    "SchemaDiscoveryService",
    "SchemaForeignKey",
    "SchemaObjectKind",
    "SchemaRelation",
    "SchemaSnapshot",
    "SchemaUniqueConstraint",
    "SecretReferenceResolver",
    "SemanticSchemaContext",
    "build_semantic_schema_context",
    "create_postgres_schema_discovery_service",
    "default_query_policy",
    "normalize_postgres_type",
    "validate_connection_ref",
]
