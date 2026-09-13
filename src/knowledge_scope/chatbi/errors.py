"""Explicit, safe error categories for future ChatBI execution paths."""

from __future__ import annotations

from enum import StrEnum


class ChatBIErrorCategory(StrEnum):
    """Stable categories exposed to application-level error handling."""

    DATASOURCE_NOT_FOUND = "datasource_not_found"
    DATASOURCE_DISABLED = "datasource_disabled"
    UNSUPPORTED_DIALECT = "unsupported_dialect"
    CREDENTIAL_RESOLUTION_FAILED = "credential_resolution_failed"
    SCHEMA_DISCOVERY_FAILED = "schema_discovery_failed"
    INVALID_QUERY = "invalid_query"
    GENERATION_FAILED = "generation_failed"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    SQL_PARSE_ERROR = "sql_parse_error"
    UNSAFE_QUERY = "unsafe_query"
    POLICY_VIOLATION = "policy_violation"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_FAILED = "execution_failed"


class ChatBIError(RuntimeError):
    """An application error whose message is safe to expose to a caller."""

    def __init__(self, category: ChatBIErrorCategory, message: str) -> None:
        self.category = category
        self.safe_message = message
        super().__init__(message)
