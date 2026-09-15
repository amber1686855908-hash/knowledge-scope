"""Persistent SQLAlchemy model for LLM usage observations."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

LLM_PROVIDER_MAX_LENGTH: Final = 64
LLM_MODEL_MAX_LENGTH: Final = 255
LLM_TASK_TYPE_MAX_LENGTH: Final = 64
LLM_ERROR_CATEGORY_MAX_LENGTH: Final = 32
LLM_INVOCATION_STAGE_MAX_LENGTH: Final = 32
LLM_INVOCATION_OUTCOME_MAX_LENGTH: Final = 16
LLM_INVOCATION_FINISH_REASON_MAX_LENGTH: Final = 64
LLM_INVOCATION_STATUS_CLASS_MAX_LENGTH: Final = 8
LLM_INVOCATION_ERROR_CATEGORY_MAX_LENGTH: Final = 64
LLM_INVOCATION_PARSE_OUTCOME_MAX_LENGTH: Final = 48
LLM_INVOCATION_TOKEN_STATUS_MAX_LENGTH: Final = 16
LLM_INVOCATION_CASE_ID_MAX_LENGTH: Final = 128


class LLMUsageRecord(Base):
    """One logical gateway call, including failed calls and known usage."""

    __tablename__ = "llm_usage_records"
    __table_args__ = (
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_llm_usage_records_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_llm_usage_records_output_tokens_non_negative",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_llm_usage_records_latency_non_negative",
        ),
        CheckConstraint(
            "estimated_cost IS NULL OR estimated_cost >= 0",
            name="ck_llm_usage_records_cost_non_negative",
        ),
        Index("ix_llm_usage_records_created_at", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    provider: Mapped[str] = mapped_column(String(length=LLM_PROVIDER_MAX_LENGTH), nullable=False)
    model: Mapped[str] = mapped_column(String(length=LLM_MODEL_MAX_LENGTH), nullable=False)
    task_type: Mapped[str] = mapped_column(String(length=LLM_TASK_TYPE_MAX_LENGTH), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_category: Mapped[str | None] = mapped_column(
        String(length=LLM_ERROR_CATEGORY_MAX_LENGTH), nullable=True
    )
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LLMProviderInvocationRecord(Base):
    """One actual provider attempt, including attempts with no ``LLMResult``."""

    __tablename__ = "llm_provider_invocations"
    __table_args__ = (
        CheckConstraint(
            "attempt_index >= 1",
            name="ck_llm_provider_invocations_attempt_index_positive",
        ),
        CheckConstraint(
            "duration_ms >= 0",
            name="ck_llm_provider_invocations_duration_non_negative",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_llm_provider_invocations_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_llm_provider_invocations_output_tokens_non_negative",
        ),
        CheckConstraint(
            "output_token_budget IS NULL OR output_token_budget >= 1",
            name="ck_llm_provider_invocations_output_budget_positive",
        ),
        Index("ix_llm_provider_invocations_started_at", "started_at"),
        Index("ix_llm_provider_invocations_case_id", "case_id"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[str | None] = mapped_column(
        String(length=LLM_INVOCATION_CASE_ID_MAX_LENGTH), nullable=True
    )
    logical_stage: Mapped[str] = mapped_column(
        String(length=LLM_INVOCATION_STAGE_MAX_LENGTH), nullable=False
    )
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(length=LLM_PROVIDER_MAX_LENGTH), nullable=False)
    model: Mapped[str] = mapped_column(String(length=LLM_MODEL_MAX_LENGTH), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    outcome: Mapped[str] = mapped_column(
        String(length=LLM_INVOCATION_OUTCOME_MAX_LENGTH), nullable=False
    )
    finish_reason: Mapped[str | None] = mapped_column(
        String(length=LLM_INVOCATION_FINISH_REASON_MAX_LENGTH), nullable=True
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status_class: Mapped[str | None] = mapped_column(
        String(length=LLM_INVOCATION_STATUS_CLASS_MAX_LENGTH), nullable=True
    )
    error_category: Mapped[str | None] = mapped_column(
        String(length=LLM_INVOCATION_ERROR_CATEGORY_MAX_LENGTH), nullable=True
    )
    retryable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    llm_result_returned: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response_parse_outcome: Mapped[str | None] = mapped_column(
        String(length=LLM_INVOCATION_PARSE_OUTCOME_MAX_LENGTH), nullable=True
    )
    output_token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_limit_status: Mapped[str] = mapped_column(
        String(length=LLM_INVOCATION_TOKEN_STATUS_MAX_LENGTH), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
