from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from uuid import UUID, uuid4

import pytest

from knowledge_scope.llm.errors import LLMProviderError
from knowledge_scope.llm.schemas import LLMRequest, LLMStreamEvent
from knowledge_scope.rag.schemas import RAGQueryRequest
from knowledge_scope.rag.service import RAG_INSUFFICIENT_EVIDENCE, RAGService
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk
from knowledge_scope.retrieval.service import RetrievalResult
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("33333333-3333-4333-8333-333333333333")


def _chunk(chunk_id: str, text: str, block_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=uuid4(),
        score=1.0 / rank,
        payload=ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=chunk_id,
            document_id=DOCUMENT_ID,
            page_start=rank,
            page_end=rank,
            source_block_ids=[block_id],
            section_path=["章节"],
            content_types=["text"],
            asset_refs=[],
            text=text,
            chunking_config_fingerprint="a" * 64,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_model_revision="revision",
            embedding_config_fingerprint="b" * 64,
        ),
    )


class _Retrieval:
    def __init__(
        self,
        items: Sequence[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.items = tuple(items or ())
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, **filters: object) -> RetrievalResult:
        self.calls.append({"query": query, **filters})
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            query=query,
            limit=10,
            model_id="Qwen/Qwen3-Embedding-0.6B",
            collection_name="test_chunks",
            items=self.items,
        )


class _Reranking:
    def __init__(self, items: Sequence[RetrievedChunk] | None = None) -> None:
        self.items = tuple(items) if items is not None else None
        self.calls: list[tuple[str, tuple[RetrievedChunk, ...], int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> tuple[RerankedChunk, ...]:
        candidate_tuple = tuple(candidates)
        self.calls.append((query, candidate_tuple, limit))
        ranked_items = self.items if self.items is not None else candidate_tuple
        return tuple(
            RerankedChunk(chunk=item, dense_rank=index, reranker_score=float(-index))
            for index, item in enumerate(ranked_items, start=1)
        )[:limit]


class _Gateway:
    def __init__(
        self,
        events: Sequence[LLMStreamEvent] = (),
        error: BaseException | None = None,
    ) -> None:
        self.events = tuple(events)
        self.error = error
        self.requests: list[LLMRequest] = []
        self.closed = False

    async def _stream(self) -> AsyncIterator[LLMStreamEvent]:
        try:
            if self.error is not None:
                raise self.error
            for event in self.events:
                yield event
        finally:
            self.closed = True

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        return self._stream()


def _service(
    retrieval: _Retrieval,
    reranking: _Reranking,
    gateway: _Gateway,
) -> RAGService:
    settings = Settings(_env_file=None, environment="test", rag_context_budget_chars=100)
    return RAGService(retrieval, reranking, gateway, settings)  # type: ignore[arg-type]


async def _events(service: RAGService) -> list[object]:
    return [event async for event in service.stream(RAGQueryRequest(query="问题"))]


@pytest.mark.anyio
async def test_rag_stream_orders_answer_citations_and_completion() -> None:
    retrieval = _Retrieval(
        [
            _chunk("chunk-1", "第一条答案", "block-1", 1),
            _chunk("chunk-2", "第二条答案", "block-2", 2),
        ]
    )
    reranking = _Reranking()
    gateway = _Gateway(
        [
            LLMStreamEvent(delta="答案", provider="fake", model="fake-model"),
            LLMStreamEvent(
                delta=" [C1]",
                provider="fake",
                model="fake-model",
                input_tokens=18,
                output_tokens=3,
                finish_reason="stop",
            ),
        ]
    )

    events = await _events(_service(retrieval, reranking, gateway))

    assert [event.event for event in events] == [
        "answer_delta",
        "answer_delta",
        "citations",
        "complete",
    ]
    assert events[2].data["items"][0]["marker"] == "C1"
    assert events[2].data["items"][0]["document_id"] == str(DOCUMENT_ID)
    assert events[3].data["status"] == "completed"
    assert events[3].data["input_tokens"] == 18
    assert events[3].data["retrieval_latency_ms"] >= 0
    assert events[3].data["llm_latency_ms"] >= 0
    assert events[3].data["latency_ms"] >= events[3].data["llm_latency_ms"]
    assert gateway.requests[0].task_type == "rag_answer"
    assert "[C1]" in gateway.requests[0].messages[1].content
    assert retrieval.calls[0]["knowledge_base_id"] is None


@pytest.mark.anyio
async def test_rag_returns_controlled_insufficient_evidence_without_llm_call() -> None:
    retrieval = _Retrieval([_chunk("image", "", "image-block", 1)])
    gateway = _Gateway()

    events = await _events(_service(retrieval, _Reranking(), gateway))

    assert [event.event for event in events] == ["answer_delta", "citations", "complete"]
    assert events[0].data["text"] == RAG_INSUFFICIENT_EVIDENCE
    assert events[1].data["items"] == []
    assert events[2].data["status"] == "insufficient_evidence"
    assert gateway.requests == []


@pytest.mark.anyio
async def test_rag_converts_retrieval_failure_to_sse_error_events() -> None:
    service = _service(_Retrieval(error=RuntimeError("private failure")), _Reranking(), _Gateway())

    events = await _events(service)

    assert [event.event for event in events] == ["error", "complete"]
    assert events[0].data == {
        "category": "retrieval",
        "message": "retrieval pipeline failed",
    }
    assert events[1].data["status"] == "error"


@pytest.mark.anyio
async def test_rag_converts_provider_failure_without_exposing_provider_details() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway(error=LLMProviderError("api", "private provider detail"))

    events = await _events(_service(retrieval, _Reranking(), gateway))

    assert [event.event for event in events] == ["error", "complete"]
    assert events[0].data == {
        "category": "api",
        "message": "LLM provider rejected the request",
    }
    assert "private provider detail" not in str(events)


@pytest.mark.anyio
async def test_model_markers_never_change_application_citation_metadata() -> None:
    retrieval = _Retrieval(
        [
            _chunk("chunk-1", "答案一", "block-1", 1),
            _chunk("chunk-2", "答案二", "block-2", 2),
        ]
    )
    gateway = _Gateway(
        [LLMStreamEvent(delta="模型输出 [C999] [C1] [C1]", provider="fake", model="fake")]
    )

    events = await _events(_service(retrieval, _Reranking(), gateway))

    citation_items = events[-2].data["items"]
    assert [item["marker"] for item in citation_items] == ["C1", "C2"]
    assert "C999" not in {item["marker"] for item in citation_items}


@pytest.mark.anyio
async def test_rag_closes_gateway_stream_when_consumer_stops() -> None:
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake")])
    stream = _service(
        _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)]),
        _Reranking(),
        gateway,
    ).stream(RAGQueryRequest(query="问题"))

    first = await anext(stream)
    assert first.event == "answer_delta"
    assert gateway.closed is False

    await stream.aclose()

    assert gateway.closed is True


@pytest.mark.anyio
async def test_rag_propagates_cancellation() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    service = _service(
        retrieval,
        _Reranking(),
        _Gateway(error=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await _events(service)
