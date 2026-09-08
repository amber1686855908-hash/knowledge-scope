from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

from knowledge_scope.llm.errors import LLMProviderError, LLMUsagePersistenceError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsageRecordInput,
)
from knowledge_scope.shared.config import Settings


class _Recorder:
    def __init__(self) -> None:
        self.records: list[LLMUsageRecordInput] = []

    async def record(self, usage: LLMUsageRecordInput) -> None:
        self.records.append(usage)


class _FailingRecorder:
    async def record(self, _usage: LLMUsageRecordInput) -> None:
        raise LLMUsagePersistenceError()


class _Provider:
    provider_name = "fake"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def complete(self, _request: LLMRequest, *, model: str) -> LLMResult:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResult(
            text="回答",
            provider="provider-value-is-normalized",
            model=model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=1,
            finish_reason="stop",
        )

    async def _stream(self, _request: LLMRequest, *, model: str) -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(
            delta="回答",
            provider="fake",
            model=model,
            finish_reason="stop",
        )
        yield LLMStreamEvent(
            delta="",
            provider="fake",
            model=model,
            input_tokens=100,
            output_tokens=20,
        )

    def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[LLMStreamEvent]:
        return self._stream(request, model=model)


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="问题")],
        task_type="evaluation",
    )


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        llm_model="configured-model",
        llm_input_cost_per_1k_tokens=Decimal("0.10"),
        llm_output_cost_per_1k_tokens=Decimal("0.20"),
        **overrides,
    )


@pytest.mark.anyio
async def test_gateway_normalizes_result_and_records_cost() -> None:
    provider = _Provider([object()])
    recorder = _Recorder()
    result = await LLMGateway(provider, recorder, _settings()).complete(_request())

    assert result.provider == "fake"
    assert result.model == "configured-model"
    assert result.latency_ms >= 0
    assert len(recorder.records) == 1
    usage = recorder.records[0]
    assert usage.success is True
    assert usage.task_type == "evaluation"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.estimated_cost == Decimal("0.01400000")


@pytest.mark.anyio
async def test_gateway_retries_only_retryable_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(
        [
            LLMProviderError("api", "temporary failure", retryable=True, status_code=503),
            object(),
        ]
    )
    recorder = _Recorder()

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("knowledge_scope.llm.gateway.asyncio.sleep", no_sleep)
    await LLMGateway(provider, recorder, _settings(llm_max_retries=1)).complete(_request())

    assert provider.calls == 2
    assert len(recorder.records) == 1
    assert recorder.records[0].success is True


@pytest.mark.anyio
async def test_gateway_records_non_retryable_failure() -> None:
    provider = _Provider([LLMProviderError("api", "bad request", status_code=400)])
    recorder = _Recorder()

    with pytest.raises(LLMProviderError):
        await LLMGateway(provider, recorder, _settings(llm_max_retries=2)).complete(_request())

    assert provider.calls == 1
    assert len(recorder.records) == 1
    assert recorder.records[0].success is False
    assert recorder.records[0].error_category == "api"


@pytest.mark.anyio
async def test_gateway_records_stream_usage_after_consumption() -> None:
    provider = _Provider([])
    recorder = _Recorder()
    events = [
        event async for event in LLMGateway(provider, recorder, _settings()).stream(_request())
    ]

    assert [event.delta for event in events] == ["回答", ""]
    assert len(recorder.records) == 1
    assert recorder.records[0].success is True
    assert recorder.records[0].input_tokens == 100
    assert recorder.records[0].output_tokens == 20


@pytest.mark.anyio
async def test_gateway_records_cancellation_and_reraises() -> None:
    provider = _Provider([asyncio.CancelledError()])
    recorder = _Recorder()

    with pytest.raises(asyncio.CancelledError):
        await LLMGateway(provider, recorder, _settings()).complete(_request())

    assert len(recorder.records) == 1
    assert recorder.records[0].success is False
    assert recorder.records[0].error_category == "cancelled"


@pytest.mark.anyio
async def test_gateway_preserves_completion_cancellation_when_recording_fails() -> None:
    provider = _Provider([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError) as error:
        await LLMGateway(provider, _FailingRecorder(), _settings()).complete(_request())

    assert isinstance(error.value.__cause__, LLMUsagePersistenceError)


class _CancelledStreamProvider(_Provider):
    async def _stream(self, _request: LLMRequest, *, model: str) -> AsyncIterator[LLMStreamEvent]:
        raise asyncio.CancelledError()
        yield LLMStreamEvent(delta="unreachable", provider="fake", model=model)


@pytest.mark.anyio
async def test_gateway_preserves_stream_cancellation_when_recording_fails() -> None:
    provider = _CancelledStreamProvider([])

    with pytest.raises(asyncio.CancelledError) as error:
        async for _event in LLMGateway(provider, _FailingRecorder(), _settings()).stream(
            _request()
        ):
            pass

    assert isinstance(error.value.__cause__, LLMUsagePersistenceError)
