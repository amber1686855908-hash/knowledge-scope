"""ChatBI / NL2SQL domain contracts and datasource metadata foundation."""

from .errors import ChatBIError, ChatBIErrorCategory
from .policy import QueryPolicy, SQLDialect, default_query_policy
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
)

__all__ = [
    "ChatBIError",
    "ChatBIErrorCategory",
    "ColumnMetadata",
    "DataSource",
    "DataSourceCreate",
    "DataSourcePublic",
    "DataSourceUpdate",
    "QueryAuditRecord",
    "QueryExecutionRequest",
    "QueryExecutionResult",
    "QueryLifecycleState",
    "QueryPolicy",
    "SQLDialect",
    "default_query_policy",
]
