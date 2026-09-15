from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

from knowledge_scope.llm.errors import LLMProviderError, LLMUsagePersistenceError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.observability import provider_observation_context
from knowledge_scope.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsageRecordInput,
)
from knowledge_scope.llm.usage import InMemoryProviderInvocationRecorder
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

    def __init__(self, outcomes: list[object], *, finish_reason: str = "stop") -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.finish_reason = finish_reason

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
            finish_reason=self.finish_reason,
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


class _HangingProvider(_Provider):
    def __init__(self) -> None:
        super().__init__([])
        self.cancelled = False

    async def complete(self, _request: LLMRequest, *, model: str) -> LLMResult:
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("the hanging provider should not complete")


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
async def test_gateway_records_one_invocation_separately_from_logical_usage() -> None:
    provider = _Provider([object()])
    recorder = _Recorder()
    invocations = InMemoryProviderInvocationRecorder()

    result = await LLMGateway(
        provider,
        recorder,
        _settings(),
        invocation_recorder=invocations,
    ).complete(_request())

    assert result.provider_invocation_id == invocations.records[0].id
    assert len(recorder.records) == 1
    assert len(invocations.records) == 1
    invocation = invocations.records[0]
    assert invocation.attempt_index == 1
    assert invocation.outcome == "success"
    assert invocation.llm_result_returned is True
    assert invocation.response_parse_outcome == "provider_success"
    assert invocation.input_tokens == 100
    assert invocation.output_tokens == 20


@pytest.mark.anyio
async def test_gateway_records_malformed_provider_attempt_without_result() -> None:
    provider = _Provider([LLMProviderError("malformed_response", "invalid response")])
    invocations = InMemoryProviderInvocationRecorder()

    with pytest.raises(LLMProviderError):
        await LLMGateway(
            provider,
            _Recorder(),
            _settings(),
            invocation_recorder=invocations,
        ).complete(_request())

    assert len(invocations.records) == 1
    invocation = invocations.records[0]
    assert invocation.outcome == "failure"
    assert invocation.error_category == "malformed_provider_response"
    assert invocation.llm_result_returned is False
    assert invocation.response_parse_outcome is None


@pytest.mark.anyio
async def test_gateway_marks_token_limit_only_from_finish_reason() -> None:
    request = _request().model_copy(update={"max_tokens": 20})
    suspected = InMemoryProviderInvocationRecorder()
    await LLMGateway(
        _Provider([object()]),
        _Recorder(),
        _settings(),
        invocation_recorder=suspected,
    ).complete(request)
    assert suspected.records[0].token_limit_status == "suspected"

    confirmed = InMemoryProviderInvocationRecorder()
    await LLMGateway(
        _Provider([object()], finish_reason="length"),
        _Recorder(),
        _settings(),
        invocation_recorder=confirmed,
    ).complete(request)
    assert confirmed.records[0].token_limit_status == "confirmed"


@pytest.mark.anyio
async def test_gateway_records_the_actual_1024_output_budget() -> None:
    request = _request().model_copy(update={"max_tokens": 1024})
    invocations = InMemoryProviderInvocationRecorder()

    await LLMGateway(
        _Provider([object()]),
        _Recorder(),
        _settings(),
        invocation_recorder=invocations,
    ).complete(request)

    assert invocations.records[0].output_token_budget == 1024
    assert invocations.records[0].token_limit_status == "not_reached"


@pytest.mark.anyio
async def test_gateway_records_each_retry_attempt_and_one_logical_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(
        [
            LLMProviderError("api", "temporary failure", retryable=True, status_code=503),
            object(),
        ]
    )
    invocations = InMemoryProviderInvocationRecorder()

    async def no_sleep(_delay: float) -> None:
        return None

    # Keep this test independent of wall-clock backoff.
    monkeypatch.setattr("knowledge_scope.llm.gateway.asyncio.sleep", no_sleep)
    result = await LLMGateway(
        provider,
        _Recorder(),
        _settings(llm_max_retries=1),
        invocation_recorder=invocations,
    ).complete(_request())

    assert result.provider_attempts == 2
    assert [item.attempt_index for item in invocations.records] == [1, 2]
    assert [item.outcome for item in invocations.records] == ["failure", "success"]
    assert invocations.records[0].status_class == "5xx"
    assert invocations.records[1].llm_result_returned is True


@pytest.mark.anyio
async def test_gateway_absolute_timeout_cancels_and_records_provider_attempt() -> None:
    provider = _HangingProvider()
    invocations = InMemoryProviderInvocationRecorder()

    with pytest.raises(LLMProviderError) as error:
        await LLMGateway(
            provider,
            _Recorder(),
            _settings(llm_timeout_seconds=0.01),
            invocation_recorder=invocations,
        ).complete(_request())

    assert error.value.category == "timeout"
    assert provider.calls == 1
    assert provider.cancelled is True
    assert len(invocations.records) == 1
    assert invocations.records[0].error_category == "provider_timeout"
    assert invocations.records[0].outcome == "failure"


@pytest.mark.anyio
async def test_gateway_invocation_context_is_task_local_and_safe() -> None:
    provider = _Provider([object()])
    invocations = InMemoryProviderInvocationRecorder()

    with provider_observation_context(case_id="case-001", logical_stage="analysis"):
        await LLMGateway(
            provider,
            _Recorder(),
            _settings(),
            invocation_recorder=invocations,
        ).complete(_request())

    assert invocations.records[0].case_id == "case-001"
    assert invocations.records[0].logical_stage == "analysis"


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
    invocations = InMemoryProviderInvocationRecorder()
    events = [
        event
        async for event in LLMGateway(
            provider,
            recorder,
            _settings(),
            invocation_recorder=invocations,
        ).stream(_request())
    ]

    assert [event.delta for event in events] == ["回答", ""]
    assert len(recorder.records) == 1
    assert recorder.records[0].success is True
    assert recorder.records[0].input_tokens == 100
    assert recorder.records[0].output_tokens == 20
    assert len(invocations.records) == 1
    assert invocations.records[0].outcome == "success"
    assert invocations.records[0].finish_reason == "stop"
    assert invocations.records[0].llm_result_returned is True


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
