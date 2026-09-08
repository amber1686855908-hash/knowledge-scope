from decimal import Decimal

import pytest
from pydantic import ValidationError

from knowledge_scope.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMUsageRecordInput,
)


def test_llm_request_normalizes_messages_and_model() -> None:
    request = LLMRequest(
        messages=[LLMMessage(role="user", content="  问题  ")],
        task_type="rag_answer",
        model="  local-model  ",
        temperature=0.2,
        max_tokens=128,
    )

    assert request.messages[0].content == "问题"
    assert request.model == "local-model"
    assert request.task_type == "rag_answer"


def test_llm_schemas_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="user", content="问题", secret="should not pass")

    with pytest.raises(ValidationError):
        LLMRequest(
            messages=[LLMMessage(role="user", content="问题")],
            task_type="evaluation",
            unexpected="should not pass",
        )


def test_usage_record_accepts_optional_cost_and_known_task_type() -> None:
    usage = LLMUsageRecordInput(
        provider="deepseek",
        model="deepseek-chat",
        task_type="evaluation",
        input_tokens=100,
        output_tokens=20,
        latency_ms=12.5,
        success=True,
        estimated_cost=Decimal("0.0123"),
    )

    assert usage.estimated_cost == Decimal("0.0123")
    assert usage.error_category is None
