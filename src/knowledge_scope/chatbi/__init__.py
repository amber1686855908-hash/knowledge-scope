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
from .nl2sql import (
    NL2SQLService,
    RegisteredDataSourceProvider,
    SchemaDiscoveryProvider,
    build_nl2sql_messages,
)
from .nl2sql_models import (
    NL2SQL_MAX_TOKENS,
    NL2SQLInput,
    NL2SQLResult,
    SQLCandidate,
    SQLGenerationPayload,
)
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
    render_structural_schema_context,
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
from .sql_validation import policy_fingerprint

__all__ = [
    "NL2SQL_MAX_TOKENS",
    "ChatBIError",
    "ChatBIErrorCategory",
    "ColumnMetadata",
    "CredentialResolver",
    "DataSource",
    "DataSourceCreate",
    "DataSourcePublic",
    "DataSourceUpdate",
    "EnvironmentCredentialResolver",
    "NL2SQLInput",
    "NL2SQLResult",
    "NL2SQLService",
    "QueryAuditRecord",
    "QueryExecutionRequest",
    "QueryExecutionResult",
    "QueryLifecycleState",
    "QueryPolicy",
    "RegisteredDataSourceProvider",
    "ResolvedDatabaseCredentials",
    "SQLCandidate",
    "SQLDialect",
    "SQLGenerationPayload",
    "SchemaColumn",
    "SchemaContextBudgetError",
    "SchemaDiscoveryAdapter",
    "SchemaDiscoveryProvider",
    "SchemaDiscoveryResult",
    "SchemaDiscoveryService",
    "SchemaForeignKey",
    "SchemaObjectKind",
    "SchemaRelation",
    "SchemaSnapshot",
    "SchemaUniqueConstraint",
    "SecretReferenceResolver",
    "SemanticSchemaContext",
    "build_nl2sql_messages",
    "build_semantic_schema_context",
    "create_postgres_schema_discovery_service",
    "default_query_policy",
    "normalize_postgres_type",
    "policy_fingerprint",
    "render_structural_schema_context",
    "validate_connection_ref",
]
