from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.graph.retrieval import GraphPath
from knowledge_scope.llm.schemas import LLMRequest, LLMStreamEvent
from knowledge_scope.rag.context import assemble_context
from knowledge_scope.rag.schemas import RAGCitation, RAGQueryRequest
from knowledge_scope.rag.service import RAGService
from knowledge_scope.retrieval.unified import (
    UnifiedBranchContribution,
    UnifiedBranchExecution,
    UnifiedCandidate,
    UnifiedRetrievalError,
    UnifiedRetrievalResult,
    chunk_candidate_id,
    evidence_candidate_id,
)
from knowledge_scope.shared.config import Settings

KNOWLEDGE_BASE_ID = UUID("44444444-4444-4444-8444-444444444444")
DOCUMENT_ID = UUID("55555555-5555-4555-8555-555555555555")


def _chunk_candidate(
    *,
    branches: Sequence[UnifiedBranchContribution] = (),
    text: str = "Unified chunk answer",
) -> UnifiedCandidate:
    contributions = list(branches) or [
        UnifiedBranchContribution(branch="dense", rank=1, score=0.9),
    ]
    source = contributions[0].branch if len(contributions) == 1 else "multiple"
    return UnifiedCandidate(
        candidate_id=chunk_candidate_id(KNOWLEDGE_BASE_ID, DOCUMENT_ID, "chunk-1"),
        candidate_kind="chunk",
        source=source,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        chunk_id="chunk-1",
        page_start=1,
        page_end=1,
        source_block_ids=["block-1"],
        section_path=["章节"],
        content_types=["text"],
        source_text=text,
        rerank_text=text,
        branches=contributions,
        final_rank=1,
        final_reranker_score=0.8,
    )


def _evidence_candidate(modality: str) -> UnifiedCandidate:
    evidence_id = f"evidence-{modality}"
    return UnifiedCandidate(
        candidate_id=evidence_candidate_id(KNOWLEDGE_BASE_ID, DOCUMENT_ID, evidence_id),
        candidate_kind="evidence",
        source="multimodal",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        evidence_id=evidence_id,
        page_start=2,
        page_end=2,
        source_block_ids=[f"{modality}-block"],
        section_path=["章节", modality],
        asset_refs=[f"assets/{modality}-1"],
        modality=modality,  # type: ignore[arg-type]
        representation_ids=[f"representation-{modality}"],
        rerank_text=f"{modality} 的可检索表示",
        branches=[
            UnifiedBranchContribution(
                branch="multimodal",
                rank=1,
                score=0.7,
                representation_ids=[f"representation-{modality}"],
            )
        ],
        final_rank=1,
        final_reranker_score=0.7,
    )


def _result(
    items: Sequence[UnifiedCandidate],
    *,
    graph_status: str = "empty",
    graph_error: str | None = None,
    degraded: bool = False,
) -> UnifiedRetrievalResult:
    branches = [
        UnifiedBranchExecution(
            branch="dense",
            status="success" if items else "empty",
            candidate_count=len(items),
            latency_ms=1,
        ),
        UnifiedBranchExecution(
            branch="graph",
            status=graph_status,  # type: ignore[arg-type]
            candidate_count=0,
            latency_ms=1,
            error=graph_error,
        ),
    ]
    return UnifiedRetrievalResult(
        query="问题",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        config_fingerprint="a" * 64,
        reranker_model_id="BAAI/bge-reranker-v2-m3",
        branches=branches,
        candidate_pool_size=len(items),
        normalization_latency_ms=0,
        rerank_latency_ms=0,
        total_latency_ms=2,
        degraded=degraded,
        items=list(items),
    )


class _UnifiedRetrieval:
    def __init__(
        self,
        result: UnifiedRetrievalResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, UUID, UUID | None]] = []

    async def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> UnifiedRetrievalResult:
        self.calls.append((query, knowledge_base_id, document_id))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class _Gateway:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def _events(self) -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(delta="回答", provider="fake", model="fake")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        return self._events()


def _service(unified: _UnifiedRetrieval, gateway: _Gateway) -> RAGService:
    settings = Settings(_env_file=None, environment="test", rag_context_budget_chars=100)
    return RAGService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        gateway,  # type: ignore[arg-type]
        settings,
        unified_retrieval=unified,  # type: ignore[arg-type]
    )


async def _events(service: RAGService) -> list[object]:
    request = RAGQueryRequest(
        query="问题",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_mode="unified",
    )
    return [event async for event in service.stream(request)]


def test_unified_query_requires_kb_and_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError, match="knowledge_base_id is required"):
        RAGQueryRequest(query="问题", retrieval_mode="unified")
    with pytest.raises(ValidationError):
        RAGQueryRequest(query="问题", retrieval_mode="unsupported")


@pytest.mark.anyio
async def test_unified_rag_uses_final_candidate_without_legacy_reranking() -> None:
    unified = _UnifiedRetrieval(_result([_chunk_candidate()]))
    gateway = _Gateway()

    events = await _events(_service(unified, gateway))

    assert [event.event for event in events] == [
        "answer_delta",
        "citations",
        "complete",
    ]
    assert unified.calls == [("问题", KNOWLEDGE_BASE_ID, None)]
    citation = events[1].data["items"][0]
    assert citation["candidate_kind"] == "chunk"
    assert citation["chunk_id"] == "chunk-1"
    assert citation["evidence_id"] is None
    assert citation["final_rank"] == 1
    assert citation["branch_provenance"][0]["branch"] == "dense"
    assert events[2].data["retrieval_mode"] == "unified"
    assert events[2].data["retrieval_degraded"] is False
    assert gateway.requests


@pytest.mark.anyio
@pytest.mark.parametrize("modality", ["image", "table", "formula"])
async def test_unified_rag_streams_evidence_citation(
    modality: str,
) -> None:
    unified = _UnifiedRetrieval(_result([_evidence_candidate(modality)]))

    events = await _events(_service(unified, _Gateway()))

    citation = events[1].data["items"][0]
    assert citation["candidate_kind"] == "evidence"
    assert citation["evidence_id"] == f"evidence-{modality}"
    assert citation["chunk_id"] is None
    assert citation["modality"] == modality
    assert citation["representation_ids"] == [f"representation-{modality}"]


@pytest.mark.parametrize("modality", ["image", "table", "formula"])
def test_multimodal_context_uses_evidence_citation_without_fake_chunk(
    modality: str,
) -> None:
    selection = assemble_context([_evidence_candidate(modality)], budget_chars=100)

    assert len(selection.items) == 1
    citation = selection.items[0].citation
    assert citation.candidate_kind == "evidence"
    assert citation.chunk_id is None
    assert citation.evidence_id == f"evidence-{modality}"
    assert citation.modality == modality
    assert citation.representation_ids == [f"representation-{modality}"]
    assert citation.asset_refs == [f"assets/{modality}-1"]
    assert "context_kind=representation" in selection.render()


def test_unified_context_preserves_one_candidate_with_multiple_branches() -> None:
    candidate = _chunk_candidate(
        branches=[
            UnifiedBranchContribution(branch="dense", rank=2, score=0.8),
            UnifiedBranchContribution(branch="sparse", rank=1, score=3.0),
        ]
    )

    selection = assemble_context([candidate, candidate], budget_chars=100)

    assert len(selection.items) == 1
    assert {item.branch for item in selection.items[0].citation.branch_provenance} == {
        "dense",
        "sparse",
    }


def test_unified_context_preserves_graph_path_provenance() -> None:
    seed_entity_id = "entity_v2_" + "a" * 64
    path = GraphPath(
        seed_entity_id=seed_entity_id,
        entity_ids=[seed_entity_id],
        hop_distance=0,
        kind="seed",
    )
    candidate = _chunk_candidate(
        branches=[
            UnifiedBranchContribution(
                branch="graph",
                rank=1,
                score=0.9,
                graph_seed_entity_id=seed_entity_id,
                graph_retrieval_reason="seed_entity_evidence",
                graph_paths=[path],
            )
        ]
    )

    selection = assemble_context([candidate], budget_chars=100)

    graph_provenance = selection.items[0].citation.branch_provenance[0]
    assert graph_provenance.branch == "graph"
    assert graph_provenance.graph_paths == [path]


@pytest.mark.anyio
async def test_unified_degraded_result_is_explicit_in_completion() -> None:
    unified = _UnifiedRetrieval(
        _result(
            [_chunk_candidate()],
            graph_status="failed",
            graph_error="graph unavailable",
            degraded=True,
        )
    )

    events = await _events(_service(unified, _Gateway()))

    assert events[-1].data["status"] == "completed"
    assert events[-1].data["retrieval_degraded"] is True
    assert events[-1].data["retrieval_branch_statuses"]["graph"] == "failed"


@pytest.mark.anyio
async def test_unified_strict_failure_does_not_fallback_to_dense() -> None:
    gateway = _Gateway()
    unified = _UnifiedRetrieval(error=UnifiedRetrievalError("graph branch failed"))

    events = await _events(_service(unified, gateway))

    assert [event.event for event in events] == ["error", "complete"]
    assert events[-1].data["status"] == "error"
    assert events[-1].data["retrieval_mode"] == "unified"
    assert gateway.requests == []


@pytest.mark.anyio
async def test_unified_empty_context_stays_insufficient_without_llm() -> None:
    gateway = _Gateway()
    unified = _UnifiedRetrieval(_result([]))

    events = await _events(_service(unified, gateway))

    assert events[0].data["text"] == "当前检索到的资料不足以回答该问题。"
    assert events[-1].data["status"] == "insufficient_evidence"
    assert events[-1].data["retrieval_mode"] == "unified"
    assert gateway.requests == []


def test_evidence_citation_contract_rejects_chunk_identity() -> None:
    with pytest.raises(ValidationError):
        RAGCitation(
            marker="C1",
            document_id=DOCUMENT_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            candidate_kind="evidence",
            chunk_id="chunk-1",
            evidence_id="evidence-1",
            modality="image",
            representation_ids=["representation-1"],
            page_start=1,
            page_end=1,
            source_block_ids=["block-1"],
            section_path=["章节"],
        )
