"""Small provider-independent async gateway with usage recording."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from time import perf_counter

from knowledge_scope.shared.config import Settings

from .errors import LLMError, LLMProviderError
from .providers import LLMProvider
from .schemas import LLMRequest, LLMResult, LLMStreamEvent, LLMUsageRecordInput
from .usage import UsageRecorder, estimate_cost


class LLMGateway:
    """Execute completions through an injected provider and usage recorder."""

    def __init__(
        self,
        provider: LLMProvider,
        usage_recorder: UsageRecorder,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._usage_recorder = usage_recorder
        self._settings = settings

    async def _call_with_retries(
        self,
        operation: Callable[[], Awaitable[LLMResult]],
    ) -> LLMResult:
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                result = await operation()
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
                lambda: self._provider.complete(request, model=model)
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
                    provider=provider,
                    model=model,
                    input_tokens=None,
                    output_tokens=None,
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
        """Yield a normalized stream and record it when fully consumed or cancelled."""
        model = request.model or self._settings.llm_model
        provider = self._provider.provider_name
        started = perf_counter()
        input_tokens: int | None = None
        output_tokens: int | None = None
        error_category: str | None = None
        success = False
        cancellation: asyncio.CancelledError | None = None
        try:
            async for event in self._provider.stream(request, model=model):
                if event.input_tokens is not None:
                    input_tokens = event.input_tokens
                if event.output_tokens is not None:
                    output_tokens = event.output_tokens
                yield event.model_copy(update={"provider": provider, "model": model})
            success = True
        except asyncio.CancelledError as error:
            cancellation = error
            error_category = "cancelled"
            raise
        except LLMError as error:
            error_category = error.category
            raise
        except Exception as error:
            error_category = "provider"
            raise LLMError("provider", "LLM provider stream failed") from error
        finally:
            if error_category != "configuration":
                try:
                    await self._record(
                        request,
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=(perf_counter() - started) * 1000,
                        success=success,
                        error_category=None if success else error_category or "cancelled",
                    )
                except Exception as persistence_error:
                    if cancellation is not None:
                        raise cancellation from persistence_error
                    raise
