"""Strict provider-independent request, result, and usage schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

LLMMessageRole = Literal["system", "user"]
LLMReasoningMode = Literal["enabled", "disabled"]
LLMTaskType = Literal[
    "rag_answer",
    "graph_extraction",
    "report_generation",
    "agent",
    "evaluation",
    "entity_linking",
    "nl2sql",
    "chatbi_analysis",
]

LLMLogicalStage = Literal["generation", "repair_generation", "analysis", "other"]
LLMInvocationOutcome = Literal["success", "failure", "cancelled"]
LLMInvocationStatusClass = Literal["1xx", "2xx", "3xx", "4xx", "5xx", "unknown"]
LLMResponseParseOutcome = Literal[
    "provider_success",
    "parsed",
    "structured_output_parse_error",
    "structured_output_schema_error",
    "malformed_provider_response",
]
LLMTokenLimitStatus = Literal["confirmed", "suspected", "not_reached", "unknown"]

LLM_TASK_TYPES: tuple[LLMTaskType, ...] = (
    "rag_answer",
    "graph_extraction",
    "report_generation",
    "agent",
    "evaluation",
    "entity_linking",
    "nl2sql",
    "chatbi_analysis",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must contain at least one non-whitespace character")
    return normalized


class LLMMessage(_StrictModel):
    """A system or user message accepted by the gateway."""

    role: LLMMessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return _non_empty(value)


class LLMResponseFormat(_StrictModel):
    """A provider-independent request for an OpenAI-compatible JSON object."""

    type: Literal["json_object"]


class LLMRequest(_StrictModel):
    """Normalized input for one logical LLM request."""

    messages: list[LLMMessage] = Field(min_length=1)
    task_type: LLMTaskType
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    model: str | None = Field(default=None, min_length=1)
    response_format: LLMResponseFormat | None = None
    reasoning: LLMReasoningMode | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        return _non_empty(value) if value is not None else None


class LLMResult(_StrictModel):
    """Provider-independent result for a completed call."""

    text: str
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    finish_reason: str | None = None
    provider_attempts: int = Field(default=1, ge=1)
    provider_invocation_id: UUID | None = Field(default=None, exclude=True)


class LLMStreamEvent(_StrictModel):
    """A normalized incremental event from a streaming completion."""

    delta: str = ""
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None


class LLMUsageRecordInput(_StrictModel):
    """Durable, non-secret usage facts recorded by the gateway."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    task_type: LLMTaskType
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    success: bool
    error_category: str | None = Field(default=None, min_length=1)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LLMProviderInvocation(_StrictModel):
    """Safe metadata for one actual provider invocation attempt.

    This record is intentionally separate from :class:`LLMUsageRecordInput`:
    one logical gateway call may contain multiple provider attempts, and a
    provider attempt may fail before an ``LLMResult`` exists.
    """

    id: UUID = Field(default_factory=uuid4)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    logical_stage: LLMLogicalStage = "other"
    attempt_index: int = Field(ge=1)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=255)
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    outcome: LLMInvocationOutcome
    finish_reason: str | None = Field(default=None, max_length=64)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status_class: LLMInvocationStatusClass | None = None
    error_category: str | None = Field(default=None, min_length=1, max_length=64)
    retryable: bool | None = None
    llm_result_returned: bool = False
    response_parse_outcome: LLMResponseParseOutcome | None = None
    output_token_budget: int | None = Field(default=None, ge=1)
    token_limit_status: LLMTokenLimitStatus = "unknown"
