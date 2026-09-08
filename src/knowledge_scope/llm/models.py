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
