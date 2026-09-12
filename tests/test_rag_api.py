from __future__ import annotations

import json

import anyio
import httpx
import pytest
from pydantic import ValidationError

from knowledge_scope.api.app import create_app
from knowledge_scope.rag.schemas import RAGQueryRequest, RAGStreamEvent
from knowledge_scope.shared.config import Settings


class _FakeRAGService:
    def __init__(self) -> None:
        self.closed = False
        self.payloads: list[object] = []

    async def stream(self, payload: object):
        self.payloads.append(payload)
        try:
            yield RAGStreamEvent(event="answer_delta", data={"text": "答案"})
            yield RAGStreamEvent(
                event="citations",
                data={"prompt_version": "rag-qa-v1", "items": []},
            )
            yield RAGStreamEvent(
                event="complete",
                data={
                    "status": "completed",
                    "prompt_version": "rag-qa-v1",
                    "latency_ms": 0,
                },
            )
        finally:
            self.closed = True


def _settings() -> Settings:
    return Settings(_env_file=None, environment="test")


def test_rag_endpoint_returns_ordered_sse_events() -> None:
    service = _FakeRAGService()
    application = create_app(_settings(), vector_store=object(), rag_service=service)  # type: ignore[arg-type]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/rag/query", json={"query": "问题"})

    response = anyio.run(request)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [frame for frame in response.text.split("\n\n") if frame]
    assert [frame.splitlines()[0] for frame in frames] == [
        "event: answer_delta",
        "event: citations",
        "event: complete",
    ]
    assert json.loads(frames[0].split("data: ", 1)[1]) == {"text": "答案"}
    assert service.closed is True


def test_rag_endpoint_reports_missing_runtime_service() -> None:
    application = create_app(_settings(), vector_store=object())  # type: ignore[arg-type]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/rag/query", json={"query": "问题"})

    response = anyio.run(request)

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service is not initialized"


def test_rag_stream_event_rejects_duplicate_citation_markers() -> None:
    citation = {
        "marker": "C1",
        "document_id": "11111111-1111-4111-8111-111111111111",
        "chunk_id": "chunk-1",
        "page_start": 1,
        "page_end": 1,
        "source_block_ids": ["block-1"],
        "section_path": ["章节"],
    }

    with pytest.raises(ValidationError, match="citation markers must be unique"):
        RAGStreamEvent(
            event="citations",
            data={
                "prompt_version": "rag-qa-v1",
                "items": [citation, citation],
            },
        )


def test_rag_stream_event_rejects_malformed_citation_metadata() -> None:
    with pytest.raises(ValidationError):
        RAGStreamEvent(
            event="citations",
            data={"prompt_version": "rag-qa-v1", "items": [{"marker": "C0"}]},
        )


def test_rag_query_rejects_unbounded_input() -> None:
    with pytest.raises(ValidationError):
        RAGQueryRequest(query="x" * 4_001)


def test_rag_endpoint_rejects_unified_mode_without_knowledge_base() -> None:
    service = _FakeRAGService()
    application = create_app(_settings(), vector_store=object(), rag_service=service)  # type: ignore[arg-type]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/v1/rag/query",
                json={"query": "问题", "retrieval_mode": "unified"},
            )

    response = anyio.run(request)

    assert response.status_code == 422


def test_rag_endpoint_rejects_unknown_retrieval_mode() -> None:
    service = _FakeRAGService()
    application = create_app(_settings(), vector_store=object(), rag_service=service)  # type: ignore[arg-type]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/v1/rag/query",
                json={"query": "问题", "retrieval_mode": "other"},
            )

    response = anyio.run(request)

    assert response.status_code == 422


def test_rag_endpoint_passes_unified_mode_to_service() -> None:
    service = _FakeRAGService()
    application = create_app(_settings(), vector_store=object(), rag_service=service)  # type: ignore[arg-type]
    knowledge_base_id = "22222222-2222-4222-8222-222222222222"

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/v1/rag/query",
                json={
                    "query": "问题",
                    "knowledge_base_id": knowledge_base_id,
                    "retrieval_mode": "unified",
                },
            )

    response = anyio.run(request)

    assert response.status_code == 200
    assert isinstance(service.payloads[0], RAGQueryRequest)
    assert service.payloads[0].retrieval_mode == "unified"
    assert str(service.payloads[0].knowledge_base_id) == knowledge_base_id
