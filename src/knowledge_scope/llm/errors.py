"""Small, stable error types for LLM provider and gateway failures."""

from __future__ import annotations

from typing import Literal

LLMErrorCategory = Literal[
    "configuration",
    "timeout",
    "connection",
    "api",
    "malformed_response",
    "cancelled",
    "provider",
    "usage_persistence",
]


class LLMError(Exception):
    """Base error with a safe category for usage observability."""

    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


class LLMConfigurationError(LLMError):
    """Raised when a provider cannot be used with the current settings."""

    def __init__(self, message: str) -> None:
        super().__init__("configuration", message)


class LLMProviderError(LLMError):
    """Raised for provider transport, API, or response failures."""

    def __init__(
        self,
        category: Literal["timeout", "connection", "api", "malformed_response", "provider"],
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(category, message, retryable=retryable, status_code=status_code)


class LLMUsagePersistenceError(LLMError):
    """Raised when a completed call cannot be durably recorded."""

    def __init__(self, message: str = "failed to persist LLM usage record") -> None:
        super().__init__("usage_persistence", message)
