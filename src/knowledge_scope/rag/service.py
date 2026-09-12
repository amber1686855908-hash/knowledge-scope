"""Small async RAG orchestration over the existing retrieval and LLM services."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from time import perf_counter

from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMRequest
from knowledge_scope.retrieval.embedding import EmbeddingModelError
from knowledge_scope.retrieval.qdrant import VectorStoreError
from knowledge_scope.retrieval.reranking import RerankedChunk, RerankerError, RerankingService
from knowledge_scope.retrieval.service import DenseRetrievalService, RetrievalError, RetrievalResult
from knowledge_scope.retrieval.unified import (
    UnifiedRetrievalError,
    UnifiedRetrievalResult,
    UnifiedRetrievalService,
)
from knowledge_scope.shared.config import Settings

from .context import ContextSelection, assemble_context
from .prompt import RAG_PROMPT_VERSION, build_rag_messages
from .schemas import RAGQueryRequest, RAGRetrievalMode, RAGStreamEvent

RAG_INSUFFICIENT_EVIDENCE = "当前检索到的资料不足以回答该问题。"


@dataclass(frozen=True, slots=True)
class RAGSelection:
    """Selected context plus the retrieval result that produced it."""

    retrieval_mode: RAGRetrievalMode
    context: ContextSelection
    dense_result: RetrievalResult | None = None
    reranked: tuple[RerankedChunk, ...] = ()
    unified_result: UnifiedRetrievalResult | None = None


class RAGService:
    """Coordinate synchronous local retrieval with the async LLM gateway."""

    def __init__(
        self,
        retrieval: DenseRetrievalService,
        reranking: RerankingService,
        gateway: LLMGateway,
        settings: Settings,
        *,
        unified_retrieval: UnifiedRetrievalService | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._reranking = reranking
        self._gateway = gateway
        self._settings = settings
        self._unified_retrieval = unified_retrieval
        # Avoid queueing cancelled requests into the thread pool while keeping
        # synchronous local model work off the event loop.
        self._selection_gate = asyncio.Semaphore(1)

    def _select_dense_context(self, request: RAGQueryRequest) -> RAGSelection:
        dense_result = self._retrieval.search(
            request.query,
            limit=self._settings.rag_candidate_limit,
            knowledge_base_id=request.knowledge_base_id,
            document_id=request.document_id,
        )
        text_candidates = tuple(item for item in dense_result.items if item.payload.text.strip())
        reranked = self._reranking.rerank(
            request.query,
            text_candidates,
            limit=self._settings.rag_rerank_limit,
        )
        context = assemble_context(
            reranked,
            budget_chars=self._settings.rag_context_budget_chars,
        )
        return RAGSelection(
            retrieval_mode="dense",
            context=context,
            dense_result=dense_result,
            reranked=reranked,
        )

    async def _select_unified_context(self, request: RAGQueryRequest) -> RAGSelection:
        if self._unified_retrieval is None:
            raise UnifiedRetrievalError("unified retrieval is not configured")
        if request.knowledge_base_id is None:
            raise ValueError("knowledge_base_id is required for unified retrieval")
        unified_result = await self._unified_retrieval.search(
            request.query,
            request.knowledge_base_id,
            document_id=request.document_id,
        )
        context = assemble_context(
            unified_result.items,
            budget_chars=self._settings.rag_context_budget_chars,
        )
        return RAGSelection(
            retrieval_mode="unified",
            context=context,
            unified_result=unified_result,
        )

    @staticmethod
    def _selection_metadata(selection: RAGSelection) -> dict[str, object]:
        if selection.unified_result is None:
            return {"retrieval_mode": "dense"}
        return {
            "retrieval_mode": "unified",
            "retrieval_degraded": selection.unified_result.degraded,
            "retrieval_branch_statuses": {
                branch.branch: branch.status for branch in selection.unified_result.branches
            },
        }

    @staticmethod
    def _retrieval_error_category(error: BaseException) -> str:
        if isinstance(error, ValueError):
            return "request"
        if isinstance(
            error,
            (
                EmbeddingModelError,
                RetrievalError,
                VectorStoreError,
                RerankerError,
                UnifiedRetrievalError,
            ),
        ):
            return "retrieval"
        return "retrieval"

    @staticmethod
    def _llm_error_message(error: LLMError) -> str:
        messages = {
            "configuration": "LLM provider is not configured",
            "timeout": "LLM provider timed out",
            "connection": "LLM provider is unavailable",
            "api": "LLM provider rejected the request",
            "malformed_response": "LLM provider returned an invalid response",
            "provider": "LLM generation failed",
            "usage_persistence": "LLM usage could not be recorded",
            "cancelled": "LLM generation was cancelled",
        }
        return messages.get(error.category, "LLM generation failed")

    @staticmethod
    def _error_events(
        *,
        category: str,
        message: str,
        started: float,
        retrieval_latency_ms: float | None = None,
        llm_latency_ms: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        retrieval_mode: RAGRetrievalMode = "dense",
        retrieval_degraded: bool | None = None,
        retrieval_branch_statuses: dict[str, str] | None = None,
    ) -> tuple[RAGStreamEvent, RAGStreamEvent]:
        return (
            RAGStreamEvent(
                event="error",
                data={"category": category, "message": message},
            ),
            RAGStreamEvent(
                event="complete",
                data={
                    "status": "error",
                    "prompt_version": RAG_PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_degraded": retrieval_degraded,
                    "retrieval_branch_statuses": retrieval_branch_statuses,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": finish_reason,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            ),
        )

    async def stream(self, request: RAGQueryRequest) -> AsyncIterator[RAGStreamEvent]:
        """Yield answer, citation, and terminal status events for one question."""
        started = perf_counter()
        selection_started = perf_counter()
        try:
            async with self._selection_gate:
                if request.retrieval_mode == "unified":
                    selection = await self._select_unified_context(request)
                else:
                    selection = await asyncio.to_thread(self._select_dense_context, request)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            retrieval_latency_ms = (perf_counter() - selection_started) * 1000
            events = self._error_events(
                category=self._retrieval_error_category(error),
                message=(
                    "query is invalid"
                    if isinstance(error, ValueError)
                    else "retrieval pipeline failed"
                ),
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_mode=request.retrieval_mode,
            )
            for event in events:
                yield event
            return

        retrieval_latency_ms = (perf_counter() - selection_started) * 1000
        selection_metadata = self._selection_metadata(selection)
        if not selection.context.items:
            yield RAGStreamEvent(
                event="answer_delta",
                data={"text": RAG_INSUFFICIENT_EVIDENCE},
            )
            yield RAGStreamEvent(
                event="citations",
                data={"prompt_version": RAG_PROMPT_VERSION, "items": []},
            )
            yield RAGStreamEvent(
                event="complete",
                data={
                    "status": "insufficient_evidence",
                    "prompt_version": RAG_PROMPT_VERSION,
                    "provider": None,
                    "model": None,
                    **selection_metadata,
                    "input_tokens": None,
                    "output_tokens": None,
                    "finish_reason": None,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": None,
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
            return

        llm_request = LLMRequest(
            messages=build_rag_messages(request.query, selection.context),
            task_type="rag_answer",
            temperature=0.0,
            max_tokens=self._settings.rag_max_tokens,
        )
        provider: str | None = None
        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        finish_reason: str | None = None
        llm_started = perf_counter()
        try:
            async with aclosing(self._gateway.stream(llm_request)) as gateway_stream:
                async for event in gateway_stream:
                    provider = event.provider
                    model = event.model
                    if event.input_tokens is not None:
                        input_tokens = event.input_tokens
                    if event.output_tokens is not None:
                        output_tokens = event.output_tokens
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason
                    if event.delta:
                        yield RAGStreamEvent(
                            event="answer_delta",
                            data={"text": event.delta},
                        )
        except asyncio.CancelledError:
            raise
        except LLMError as error:
            llm_latency_ms = (perf_counter() - llm_started) * 1000
            events = self._error_events(
                category=error.category,
                message=self._llm_error_message(error),
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_mode=selection.retrieval_mode,
                retrieval_degraded=selection_metadata.get("retrieval_degraded"),
                retrieval_branch_statuses=selection_metadata.get("retrieval_branch_statuses"),
            )
            for event in events:
                yield event
            return
        except Exception:
            llm_latency_ms = (perf_counter() - llm_started) * 1000
            events = self._error_events(
                category="provider",
                message="LLM generation failed",
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_mode=selection.retrieval_mode,
                retrieval_degraded=selection_metadata.get("retrieval_degraded"),
                retrieval_branch_statuses=selection_metadata.get("retrieval_branch_statuses"),
            )
            for event in events:
                yield event
            return

        llm_latency_ms = (perf_counter() - llm_started) * 1000
        yield RAGStreamEvent(
            event="citations",
            data={
                "prompt_version": RAG_PROMPT_VERSION,
                "items": [
                    citation.model_dump(mode="json") for citation in selection.context.citations
                ],
            },
        )
        yield RAGStreamEvent(
            event="complete",
            data={
                "status": "completed",
                "prompt_version": RAG_PROMPT_VERSION,
                "provider": provider or self._settings.llm_provider,
                "model": model or self._settings.llm_model,
                **selection_metadata,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "retrieval_latency_ms": retrieval_latency_ms,
                "llm_latency_ms": llm_latency_ms,
                "latency_ms": (perf_counter() - started) * 1000,
            },
        )


__all__ = ["RAG_INSUFFICIENT_EVIDENCE", "RAGSelection", "RAGService"]
