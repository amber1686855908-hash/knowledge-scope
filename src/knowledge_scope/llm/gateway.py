"""Small provider-independent async gateway with usage recording."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from knowledge_scope.shared.config import Settings

from .errors import LLMError, LLMProviderError
from .observability import current_observation_context
from .providers import LLMProvider
from .schemas import (
    LLMProviderInvocation,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsageRecordInput,
)
from .usage import (
    NullProviderInvocationRecorder,
    ProviderInvocationRecorder,
    UsageRecorder,
    estimate_cost,
)


def _status_class(status_code: int | None) -> str:
    if status_code is None or status_code < 100 or status_code > 599:
        return "unknown"
    return f"{status_code // 100}xx"


def _invocation_error_category(error: LLMError) -> str:
    return {
        "timeout": "provider_timeout",
        "connection": "transport_error",
        "api": "provider_api_error",
        "malformed_response": "malformed_provider_response",
        "cancelled": "cancelled",
    }.get(error.category, error.category)


def _token_limit_status(result: LLMResult, request: LLMRequest) -> str:
    if result.finish_reason == "length":
        return "confirmed"
    if request.max_tokens is None or result.output_tokens is None:
        return "unknown"
    if result.output_tokens == request.max_tokens:
        return "suspected"
    return "not_reached"


class LLMGateway:
    """Execute completions through an injected provider and usage recorder."""

    def __init__(
        self,
        provider: LLMProvider,
        usage_recorder: UsageRecorder,
        settings: Settings,
        invocation_recorder: ProviderInvocationRecorder | None = None,
    ) -> None:
        self._provider = provider
        self._usage_recorder = usage_recorder
        self._settings = settings
        if invocation_recorder is not None:
            self._invocation_recorder = invocation_recorder
        else:
            recorder = getattr(usage_recorder, "record_invocation", None)
            self._invocation_recorder = (
                usage_recorder if callable(recorder) else NullProviderInvocationRecorder()
            )

    async def _record_invocation(
        self,
        request: LLMRequest,
        *,
        invocation_id: UUID,
        attempt_index: int,
        started_at: datetime,
        started: float,
        outcome: str,
        finish_reason: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        status_code: int | None = None,
        error_category: str | None = None,
        retryable: bool | None = None,
        llm_result_returned: bool = False,
        response_parse_outcome: str | None = None,
        token_limit_status: str = "unknown",
    ) -> None:
        context = current_observation_context(request.task_type)
        invocation = LLMProviderInvocation(
            id=invocation_id,
            case_id=context.case_id,
            logical_stage=context.logical_stage or "other",
            attempt_index=attempt_index,
            provider=self._provider.provider_name,
            model=request.model or self._settings.llm_model,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            duration_ms=max(0.0, (perf_counter() - started) * 1_000),
            outcome=outcome,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status_class=_status_class(status_code),
            error_category=error_category,
            retryable=retryable,
            llm_result_returned=llm_result_returned,
            response_parse_outcome=response_parse_outcome,
            output_token_budget=request.max_tokens,
            token_limit_status=token_limit_status,
        )
        await self._invocation_recorder.record_invocation(invocation)

    async def _invoke_once(
        self,
        request: LLMRequest,
        *,
        model: str,
        attempt_index: int,
        operation: Callable[[], Awaitable[LLMResult]],
    ) -> LLMResult:
        """Run and observe one real provider attempt under an absolute deadline."""
        invocation_id = uuid4()
        started_at = datetime.now(UTC)
        started = perf_counter()
        try:
            try:
                async with asyncio.timeout(self._settings.llm_timeout_seconds):
                    result = await operation()
            except TimeoutError as error:
                raise LLMProviderError(
                    "timeout",
                    "LLM provider request exceeded the configured deadline",
                    retryable=True,
                    provider=self._provider.provider_name,
                    model=model,
                    invocation_id=invocation_id,
                ) from error
            if not isinstance(result, LLMResult):
                raise LLMProviderError(
                    "malformed_response",
                    "LLM provider returned an invalid normalized result",
                    provider=self._provider.provider_name,
                    model=model,
                    invocation_id=invocation_id,
                )
        except asyncio.CancelledError as cancellation:
            try:
                await self._record_invocation(
                    request,
                    invocation_id=invocation_id,
                    attempt_index=attempt_index,
                    started_at=started_at,
                    started=started,
                    outcome="cancelled",
                    error_category="cancelled",
                    retryable=False,
                )
            except Exception as persistence_error:
                raise cancellation from persistence_error
            raise
        except LLMProviderError as error:
            error.provider = self._provider.provider_name
            error.model = model
            error.invocation_id = invocation_id
            await self._record_invocation(
                request,
                invocation_id=invocation_id,
                attempt_index=attempt_index,
                started_at=started_at,
                started=started,
                outcome="failure",
                status_code=error.status_code,
                error_category=_invocation_error_category(error),
                retryable=error.retryable,
            )
            raise
        except LLMError:
            # Configuration and usage-persistence failures happen outside a
            # provider request and therefore are not outbound attempts.
            raise
        except Exception as error:
            wrapped = LLMProviderError(
                "provider",
                "LLM provider call failed",
                provider=self._provider.provider_name,
                model=model,
                invocation_id=invocation_id,
            )
            await self._record_invocation(
                request,
                invocation_id=invocation_id,
                attempt_index=attempt_index,
                started_at=started_at,
                started=started,
                outcome="failure",
                error_category=_invocation_error_category(wrapped),
                retryable=wrapped.retryable,
            )
            raise wrapped from error

        normalized = result.model_copy(
            update={"provider": self._provider.provider_name, "model": model}
        )
        await self._record_invocation(
            request,
            invocation_id=invocation_id,
            attempt_index=attempt_index,
            started_at=started_at,
            started=started,
            outcome="success",
            finish_reason=normalized.finish_reason,
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
            llm_result_returned=True,
            response_parse_outcome="provider_success",
            token_limit_status=_token_limit_status(normalized, request),
        )
        return normalized.model_copy(update={"provider_invocation_id": invocation_id})

    async def _call_with_retries(
        self,
        request: LLMRequest,
        *,
        model: str,
        operation: Callable[[], Awaitable[LLMResult]],
    ) -> LLMResult:
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                result = await self._invoke_once(
                    request,
                    model=model,
                    attempt_index=attempt + 1,
                    operation=operation,
                )
                return result.model_copy(update={"provider_attempts": attempt + 1})
            except LLMProviderError as error:
                error.provider_attempts = attempt + 1
                if not error.retryable or attempt >= self._settings.llm_max_retries:
                    raise
                await asyncio.sleep(0.25 * (2**attempt))
        raise AssertionError("retry loop must return or raise")

    async def _record(
        self,
        request: LLMRequest,
        *,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: float,
        success: bool,
        error_category: str | None,
    ) -> None:
        await self._usage_recorder.record(
            LLMUsageRecordInput(
                provider=provider,
                model=model,
                task_type=request.task_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=max(0.0, latency_ms),
                success=success,
                error_category=error_category,
                estimated_cost=estimate_cost(input_tokens, output_tokens, self._settings),
            )
        )

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Run one completion, record it, and retry only explicit transient errors."""
        model = request.model or self._settings.llm_model
        provider = self._provider.provider_name
        started = perf_counter()
        try:
            result = await self._call_with_retries(
                request,
                model=model,
                operation=lambda: self._provider.complete(request, model=model),
            )
        except asyncio.CancelledError as cancellation:
            try:
                await self._record(
                    request,
                    provider=provider,
                    model=model,
                    input_tokens=None,
                    output_tokens=None,
                    latency_ms=(perf_counter() - started) * 1000,
                    success=False,
                    error_category="cancelled",
                )
            except Exception as persistence_error:
                raise cancellation from persistence_error
            raise
        except LLMError as error:
            if error.category != "configuration":
                await self._record(
                    request,
                    provider=error.provider or provider,
                    model=error.model or model,
                    input_tokens=error.input_tokens,
                    output_tokens=error.output_tokens,
                    latency_ms=(perf_counter() - started) * 1000,
                    success=False,
                    error_category=error.category,
                )
            raise
        except Exception as error:
            wrapped = LLMError("provider", "LLM provider call failed")
            await self._record(
                request,
                provider=provider,
                model=model,
                input_tokens=None,
                output_tokens=None,
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
                error_category=wrapped.category,
            )
            raise wrapped from error

        latency_ms = (perf_counter() - started) * 1000
        normalized = result.model_copy(
            update={
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
            }
        )
        await self._record(
            request,
            provider=provider,
            model=model,
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
            latency_ms=latency_ms,
            success=True,
            error_category=None,
        )
        return normalized

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Yield a normalized stream with an absolute deadline and invocation record."""
        model = request.model or self._settings.llm_model
        provider = self._provider.provider_name
        invocation_id = uuid4()
        started_at = datetime.now(UTC)
        started = perf_counter()
        input_tokens: int | None = None
        output_tokens: int | None = None
        finish_reason: str | None = None
        error_category: str | None = None
        logical_error_category: str | None = None
        success = False
        cancelled = False
        skip_invocation = False
        try:
            try:
                async with asyncio.timeout(self._settings.llm_timeout_seconds):
                    async for event in self._provider.stream(request, model=model):
                        if event.input_tokens is not None:
                            input_tokens = event.input_tokens
                        if event.output_tokens is not None:
                            output_tokens = event.output_tokens
                        if event.finish_reason is not None:
                            finish_reason = event.finish_reason
                        yield event.model_copy(update={"provider": provider, "model": model})
                success = True
            except TimeoutError as error:
                raise LLMProviderError(
                    "timeout",
                    "LLM provider stream exceeded the configured deadline",
                    retryable=False,
                    provider=provider,
                    model=model,
                    invocation_id=invocation_id,
                ) from error
        except asyncio.CancelledError:
            cancelled = True
            error_category = "cancelled"
            logical_error_category = "cancelled"
            raise
        except LLMProviderError as error:
            error.provider = provider
            error.model = model
            error.invocation_id = invocation_id
            error_category = _invocation_error_category(error)
            logical_error_category = error.category
            raise
        except LLMError as error:
            error_category = error.category
            logical_error_category = error.category
            skip_invocation = error.category == "configuration"
            raise
        except Exception as error:
            wrapped = LLMProviderError(
                "provider",
                "LLM provider stream failed",
                provider=provider,
                model=model,
                invocation_id=invocation_id,
            )
            error_category = _invocation_error_category(wrapped)
            logical_error_category = wrapped.category
            raise wrapped from error
        finally:
            if not skip_invocation:
                try:
                    final_outcome = (
                        "success" if success else ("cancelled" if cancelled else "failure")
                    )
                    await self._record_invocation(
                        request,
                        invocation_id=invocation_id,
                        attempt_index=1,
                        started_at=started_at,
                        started=started,
                        outcome=final_outcome,
                        finish_reason=finish_reason,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error_category=None if success else error_category or "cancelled",
                        retryable=False if success else None,
                        llm_result_returned=success,
                        response_parse_outcome="provider_success" if success else None,
                        token_limit_status=(
                            "confirmed"
                            if finish_reason == "length"
                            else (
                                "suspected"
                                if request.max_tokens is not None
                                and output_tokens == request.max_tokens
                                else "unknown"
                            )
                        ),
                    )
                    await self._record(
                        request,
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=(perf_counter() - started) * 1_000,
                        success=success,
                        error_category=(None if success else logical_error_category or "cancelled"),
                    )
                except Exception as persistence_error:
                    if cancelled:
                        raise asyncio.CancelledError() from persistence_error
                    raise
