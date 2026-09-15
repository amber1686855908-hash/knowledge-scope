"""Usage recording and configuration-driven cost estimation."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_scope.shared.config import Settings

from .errors import LLMUsagePersistenceError
from .models import LLMProviderInvocationRecord, LLMUsageRecord
from .schemas import LLMProviderInvocation, LLMUsageRecordInput


class UsageRecorder(Protocol):
    """Async sink for one logical LLM usage record."""

    async def record(self, usage: LLMUsageRecordInput) -> None:
        """Persist or otherwise durably store the usage record."""


class ProviderInvocationRecorder(Protocol):
    """Durable sink for one real provider invocation attempt."""

    async def record_invocation(self, invocation: LLMProviderInvocation) -> None:
        """Persist one safe attempt observation."""

    async def update_invocation(
        self,
        invocation_id: UUID,
        **changes: object,
    ) -> None:
        """Update only controlled post-processing metadata for one observation."""


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
    """Persist usage and provider observations in short-lived async sessions."""

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

    async def record_invocation(self, invocation: LLMProviderInvocation) -> None:
        """Insert one provider attempt without retaining prompts or responses."""
        async with self._session_factory() as session:
            try:
                session.add(
                    LLMProviderInvocationRecord(
                        id=invocation.id,
                        case_id=invocation.case_id,
                        logical_stage=invocation.logical_stage,
                        attempt_index=invocation.attempt_index,
                        provider=invocation.provider,
                        model=invocation.model,
                        started_at=invocation.started_at,
                        completed_at=invocation.completed_at,
                        duration_ms=Decimal(str(invocation.duration_ms)),
                        outcome=invocation.outcome,
                        finish_reason=invocation.finish_reason,
                        input_tokens=invocation.input_tokens,
                        output_tokens=invocation.output_tokens,
                        status_class=invocation.status_class,
                        error_category=invocation.error_category,
                        retryable=invocation.retryable,
                        llm_result_returned=invocation.llm_result_returned,
                        response_parse_outcome=invocation.response_parse_outcome,
                        output_token_budget=invocation.output_token_budget,
                        token_limit_status=invocation.token_limit_status,
                    )
                )
                await session.commit()
            except SQLAlchemyError as error:
                await session.rollback()
                raise LLMUsagePersistenceError(
                    "failed to persist LLM provider invocation"
                ) from error

    async def update_invocation(
        self,
        invocation_id: UUID,
        **changes: object,
    ) -> None:
        """Update bounded parse metadata after downstream processing completes."""
        allowed = {"response_parse_outcome", "error_category"}
        if set(changes) - allowed:
            raise ValueError("unsupported provider invocation update")
        async with self._session_factory() as session:
            try:
                record = await session.get(LLMProviderInvocationRecord, invocation_id)
                if record is None:
                    raise LLMUsagePersistenceError("provider invocation record was not found")
                for key, value in changes.items():
                    setattr(record, key, value)
                await session.commit()
            except LLMUsagePersistenceError:
                await session.rollback()
                raise
            except SQLAlchemyError as error:
                await session.rollback()
                raise LLMUsagePersistenceError(
                    "failed to update LLM provider invocation"
                ) from error


class NullProviderInvocationRecorder:
    """No-op recorder retained for provider-free/unit gateway construction."""

    async def record_invocation(self, _invocation: LLMProviderInvocation) -> None:
        return None

    async def update_invocation(self, _invocation_id: UUID, **_changes: object) -> None:
        return None


class InMemoryProviderInvocationRecorder:
    """Deterministic per-run recorder used by provider evaluation and unit tests."""

    def __init__(self) -> None:
        self.records: list[LLMProviderInvocation] = []

    async def record_invocation(self, invocation: LLMProviderInvocation) -> None:
        self.records.append(invocation)

    async def update_invocation(
        self,
        invocation_id: UUID,
        **changes: object,
    ) -> None:
        for index, record in enumerate(self.records):
            if record.id == invocation_id:
                self.records[index] = LLMProviderInvocation.model_validate(
                    record.model_copy(update=changes).model_dump()
                )
                return
        raise LLMUsagePersistenceError("provider invocation record was not found")


class CompositeProviderInvocationRecorder:
    """Forward safe observations to durable and run-local sinks."""

    def __init__(self, *recorders: ProviderInvocationRecorder) -> None:
        self._recorders = recorders

    async def record_invocation(self, invocation: LLMProviderInvocation) -> None:
        for recorder in self._recorders:
            await recorder.record_invocation(invocation)

    async def update_invocation(
        self,
        invocation_id: UUID,
        **changes: object,
    ) -> None:
        for recorder in self._recorders:
            await recorder.update_invocation(invocation_id, **changes)
