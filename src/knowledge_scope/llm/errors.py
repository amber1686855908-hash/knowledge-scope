"""Small, stable error types for LLM provider and gateway failures."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

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
        provider_attempts: int = 0,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        invocation_id: UUID | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.provider_attempts = max(0, provider_attempts)
        self.provider = provider
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.invocation_id = invocation_id


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
        provider_attempts: int = 1,
        provider: str | None = None,
        model: str | None = None,
        invocation_id: UUID | None = None,
    ) -> None:
        super().__init__(
            category,
            message,
            retryable=retryable,
            status_code=status_code,
            provider_attempts=provider_attempts,
            provider=provider,
            model=model,
            invocation_id=invocation_id,
        )


class LLMUsagePersistenceError(LLMError):
    """Raised when a completed call cannot be durably recorded."""

    def __init__(self, message: str = "failed to persist LLM usage record") -> None:
        super().__init__("usage_persistence", message)
