"""OpenAI-compatible provider adapter used by the LLM gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any, Protocol

import httpx

from knowledge_scope.shared.config import Settings

from .errors import LLMConfigurationError, LLMProviderError
from .schemas import LLMRequest, LLMResult, LLMStreamEvent


class LLMProvider(Protocol):
    """Minimal async provider contract implemented by gateway adapters."""

    provider_name: str

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        """Return one normalized completion."""

    def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[LLMStreamEvent]:
        """Yield normalized completion events."""


def _parse_usage(payload: object) -> tuple[int | None, int | None]:
    if payload is None:
        return None, None
    if not isinstance(payload, dict):
        raise LLMProviderError("malformed_response", "provider usage has an invalid shape")

    def token_value(key: str) -> int | None:
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LLMProviderError("malformed_response", "provider usage has invalid token data")
        return value

    return token_value("prompt_tokens"), token_value("completion_tokens")


def _parse_completion(
    payload: object,
    *,
    provider: str,
    model: str,
    latency_ms: float,
) -> LLMResult:
    if not isinstance(payload, dict):
        raise LLMProviderError("malformed_response", "provider response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderError("malformed_response", "provider response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMProviderError("malformed_response", "provider choice has an invalid shape")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LLMProviderError("malformed_response", "provider message has no text content")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise LLMProviderError("malformed_response", "provider finish reason is invalid")
    input_tokens, output_tokens = _parse_usage(payload.get("usage"))
    return LLMResult(
        text=message["content"],
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


def _parse_stream_event(
    payload: object,
    *,
    provider: str,
    model: str,
) -> LLMStreamEvent:
    if not isinstance(payload, dict):
        raise LLMProviderError("malformed_response", "provider stream event must be an object")
    input_tokens, output_tokens = _parse_usage(payload.get("usage"))
    choices = payload.get("choices")
    if choices == []:
        if payload.get("usage") is None:
            raise LLMProviderError("malformed_response", "provider stream event has no choices")
        return LLMStreamEvent(
            delta="",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMProviderError("malformed_response", "provider stream choices are invalid")
    choice = choices[0]
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        raise LLMProviderError("malformed_response", "provider stream delta is invalid")
    content = delta.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise LLMProviderError("malformed_response", "provider stream content is invalid")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise LLMProviderError("malformed_response", "provider stream finish reason is invalid")
    return LLMStreamEvent(
        delta=content,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
    )


class DeepSeekProvider:
    """Async adapter for DeepSeek's OpenAI-compatible chat completions API."""

    provider_name = "deepseek"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = settings.llm_api_key
        self._client = client or httpx.AsyncClient(
            base_url=f"{settings.llm_base_url.rstrip('/')}/",
            timeout=settings.llm_timeout_seconds,
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        if self._api_key is None or not self._api_key.get_secret_value().strip():
            raise LLMConfigurationError("KNOWLEDGE_SCOPE_LLM_API_KEY is not configured")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _payload(request: LLMRequest, model: str, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format.model_dump(mode="json")
        if request.reasoning is not None:
            payload["thinking"] = {"type": request.reasoning}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        retryable = response.status_code == 429 or response.status_code >= 500
        raise LLMProviderError(
            "api",
            f"LLM provider returned HTTP {response.status_code}",
            retryable=retryable,
            status_code=response.status_code,
        )

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        """Call `/chat/completions` and validate the normalized response shape."""
        started = perf_counter()
        try:
            response = await self._client.post(
                "chat/completions",
                headers=self._headers(),
                json=self._payload(request, model, stream=False),
            )
            self._raise_for_status(response)
            payload = response.json()
        except LLMProviderError:
            raise
        except LLMConfigurationError:
            raise
        except httpx.TimeoutException as error:
            raise LLMProviderError(
                "timeout", "LLM provider request timed out", retryable=True
            ) from error
        except httpx.ConnectError as error:
            raise LLMProviderError(
                "connection", "LLM provider connection failed", retryable=True
            ) from error
        except httpx.RequestError as error:
            raise LLMProviderError(
                "connection", "LLM provider request failed", retryable=True
            ) from error
        except ValueError as error:
            raise LLMProviderError(
                "malformed_response", "LLM provider returned invalid JSON"
            ) from error

        return _parse_completion(
            payload,
            provider=self.provider_name,
            model=model,
            latency_ms=(perf_counter() - started) * 1000,
        )

    async def _stream_events(
        self, request: LLMRequest, *, model: str
    ) -> AsyncIterator[LLMStreamEvent]:
        try:
            async with self._client.stream(
                "POST",
                "chat/completions",
                headers=self._headers(),
                json=self._payload(request, model, stream=True),
            ) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise LLMProviderError(
                            "malformed_response", "LLM provider stream line is invalid"
                        )
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        payload = json.loads(data)
                    except (TypeError, ValueError) as error:
                        raise LLMProviderError(
                            "malformed_response", "LLM provider stream event is invalid JSON"
                        ) from error
                    yield _parse_stream_event(
                        payload,
                        provider=self.provider_name,
                        model=model,
                    )
        except (LLMProviderError, LLMConfigurationError):
            raise
        except httpx.TimeoutException as error:
            raise LLMProviderError(
                "timeout", "LLM provider stream timed out", retryable=False
            ) from error
        except httpx.ConnectError as error:
            raise LLMProviderError(
                "connection", "LLM provider stream connection failed", retryable=False
            ) from error
        except httpx.RequestError as error:
            raise LLMProviderError(
                "connection", "LLM provider stream request failed", retryable=False
            ) from error

    def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[LLMStreamEvent]:
        """Return an SSE-backed stream; retries are intentionally not automatic."""
        return self._stream_events(request, model=model)


def create_llm_provider(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
) -> DeepSeekProvider:
    """Create the configured first provider without exposing its API key."""
    if settings.llm_provider != "deepseek":
        raise LLMConfigurationError(f"unsupported LLM provider: {settings.llm_provider}")
    return DeepSeekProvider(settings, client=client)
