from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from knowledge_scope.llm.models import LLMUsageRecord
from knowledge_scope.llm.schemas import LLMUsageRecordInput
from knowledge_scope.llm.usage import DatabaseUsageRecorder, estimate_cost
from knowledge_scope.shared.config import Settings


def test_estimate_cost_requires_complete_pricing() -> None:
    settings = Settings(
        _env_file=None,
        llm_input_cost_per_1k_tokens=Decimal("0.10"),
    )

    assert estimate_cost(100, 20, settings) is None


@pytest.mark.anyio
async def test_database_usage_recorder_persists_one_call(
    postgres_test_engine: AsyncEngine,
) -> None:
    session_factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    recorder = DatabaseUsageRecorder(session_factory)
    await recorder.record(
        LLMUsageRecordInput(
            provider="deepseek",
            model="configured-model",
            task_type="evaluation",
            input_tokens=10,
            output_tokens=4,
            latency_ms=12.3456,
            success=True,
            estimated_cost=Decimal("0.0014"),
        )
    )

    async with session_factory() as session:
        record = await session.scalar(
            select(LLMUsageRecord)
            .where(LLMUsageRecord.model == "configured-model")
            .order_by(LLMUsageRecord.created_at.desc())
        )

    assert record is not None
    assert record.provider == "deepseek"
    assert record.task_type == "evaluation"
    assert record.input_tokens == 10
    assert record.output_tokens == 4
    assert record.success is True
    assert record.estimated_cost == Decimal("0.00140000")
