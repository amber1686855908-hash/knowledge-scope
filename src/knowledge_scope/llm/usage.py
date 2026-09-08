"""Usage recording and configuration-driven cost estimation."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_scope.shared.config import Settings

from .errors import LLMUsagePersistenceError
from .models import LLMUsageRecord
from .schemas import LLMUsageRecordInput


class UsageRecorder(Protocol):
    """Async sink for one logical LLM usage record."""

    async def record(self, usage: LLMUsageRecordInput) -> None:
        """Persist or otherwise durably store the usage record."""


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    settings: Settings,
) -> Decimal | None:
    """Estimate cost only when both token counts and both configured rates exist."""
    if input_tokens is None or output_tokens is None:
        return None
    input_rate = settings.llm_input_cost_per_1k_tokens
    output_rate = settings.llm_output_cost_per_1k_tokens
    if input_rate is None or output_rate is None:
        return None
    cost = Decimal(input_tokens) * input_rate / Decimal(1000) + Decimal(
        output_tokens
    ) * output_rate / Decimal(1000)
    return cost.quantize(Decimal("0.00000001"))


class DatabaseUsageRecorder:
    """Persist each usage event in its own short-lived async session."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(self, usage: LLMUsageRecordInput) -> None:
        """Insert and commit one usage record, surfacing persistence failures."""
        async with self._session_factory() as session:
            try:
                session.add(
                    LLMUsageRecord(
                        provider=usage.provider,
                        model=usage.model,
                        task_type=usage.task_type,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        latency_ms=Decimal(str(usage.latency_ms)),
                        success=usage.success,
                        error_category=usage.error_category,
                        estimated_cost=usage.estimated_cost,
                        created_at=usage.created_at,
                    )
                )
                await session.commit()
            except SQLAlchemyError as error:
                await session.rollback()
                raise LLMUsagePersistenceError() from error
