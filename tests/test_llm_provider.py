from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from knowledge_scope.llm.errors import LLMConfigurationError, LLMProviderError
from knowledge_scope.llm.providers import DeepSeekProvider
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest
from knowledge_scope.shared.config import Settings


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(role="system", content="保持简洁"),
            LLMMessage(role="user", content="回答问题"),
        ],
        task_type="evaluation",
        temperature=0.1,
        max_tokens=32,
    )


def _settings() -> Settings:
    return Settings(_env_file=None, llm_api_key=SecretStr("test-secret"))


@pytest.mark.anyio
async def test_deepseek_provider_sends_openai_compatible_request() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads((await request.aread()).decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "已完成"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com/v1/",
    )
    provider = DeepSeekProvider(_settings(), client=client)
    try:
        result = await provider.complete(_request(), model="configured-model")
    finally:
        await client.aclose()

    assert seen["path"] == "/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-secret"
    assert seen["body"] == {
        "model": "configured-model",
        "messages": [
            {"role": "system", "content": "保持简洁"},
            {"role": "user", "content": "回答问题"},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 32,
    }
    assert result.text == "已完成"
    assert result.provider == "deepseek"
    assert result.model == "configured-model"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0


@pytest.mark.anyio
async def test_deepseek_provider_parses_stream_and_usage_event() -> None:
    stream_body = (
        'data: {"choices":[{"delta":{"content":"第一部分"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"第二部分"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body.encode(),
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com",
    )
    provider = DeepSeekProvider(_settings(), client=client)
    try:
        events = [event async for event in provider.stream(_request(), model="configured-model")]
    finally:
        await client.aclose()

    assert [event.delta for event in events] == ["第一部分", "第二部分", ""]
    assert events[-1].input_tokens == 9
    assert events[-1].output_tokens == 4
    assert events[1].finish_reason == "stop"
    assert all(event.provider == "deepseek" for event in events)


@pytest.mark.anyio
async def test_provider_does_not_include_api_key_in_configuration_error() -> None:
    provider = DeepSeekProvider(Settings(_env_file=None))

    with pytest.raises(LLMConfigurationError, match="KNOWLEDGE_SCOPE_LLM_API_KEY") as error:
        await provider.complete(_request(), model="configured-model")

    assert "test-secret" not in str(error.value)
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_marks_server_errors_retryable_without_response_body() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="secret provider diagnostics")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com",
    )
    provider = DeepSeekProvider(_settings(), client=client)
    try:
        with pytest.raises(LLMProviderError) as error:
            await provider.complete(_request(), model="configured-model")
    finally:
        await client.aclose()

    assert error.value.category == "api"
    assert error.value.retryable is True
    assert error.value.status_code == 503
    assert "secret provider diagnostics" not in str(error.value)


@pytest.mark.anyio
async def test_provider_rejects_malformed_completion_shape() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com",
    )
    provider = DeepSeekProvider(_settings(), client=client)
    try:
        with pytest.raises(LLMProviderError) as error:
            await provider.complete(_request(), model="configured-model")
    finally:
        await client.aclose()

    assert error.value.category == "malformed_response"


@pytest.mark.anyio
async def test_provider_maps_timeout_without_exposing_transport_details() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private transport detail")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com",
    )
    provider = DeepSeekProvider(_settings(), client=client)
    try:
        with pytest.raises(LLMProviderError) as error:
            await provider.complete(_request(), model="configured-model")
    finally:
        await client.aclose()

    assert error.value.category == "timeout"
    assert error.value.retryable is True
    assert "private transport detail" not in str(error.value)
