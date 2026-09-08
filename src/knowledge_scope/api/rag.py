"""Streaming HTTP endpoint for retrieval-augmented question answering."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from knowledge_scope.rag.schemas import RAGQueryRequest, RAGStreamEvent
from knowledge_scope.rag.service import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])


def encode_sse(event: RAGStreamEvent) -> str:
    """Encode one validated RAG event using the SSE wire format."""
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event}\ndata: {data}\n\n"


async def _event_stream(
    service: RAGService,
    payload: RAGQueryRequest,
) -> AsyncIterator[str]:
    async with aclosing(service.stream(payload)) as event_stream:
        async for event in event_stream:
            yield encode_sse(event)


@router.post("/query", response_class=StreamingResponse)
async def query(payload: RAGQueryRequest, request: Request) -> StreamingResponse:
    """Stream a grounded answer and application-generated citation metadata."""
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized",
        )
    return StreamingResponse(
        _event_stream(service, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["encode_sse", "router"]
