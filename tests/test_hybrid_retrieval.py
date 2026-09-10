from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Barrier
from uuid import UUID, uuid4

import pytest

from knowledge_scope.graph.models import GraphProvenance, entity_id_for, evidence_id_for
from knowledge_scope.graph.retrieval import (
    GraphEvidence,
    GraphEvidenceResult,
    GraphPath,
    GraphRetrievalResult,
)
from knowledge_scope.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridRetrievalService,
)
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk, RerankingService
from knowledge_scope.retrieval.service import RetrievalResult

KB = UUID("11111111-1111-4111-8111-111111111111")
OTHER_KB = UUID("22222222-2222-4222-8222-222222222222")
DOC = UUID("33333333-3333-4333-8333-333333333333")


def _chunk(
    chunk_id: str,
    *,
    document_id: UUID = DOC,
    knowledge_base_id: UUID | None = KB,
    source_block_ids: list[str] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=uuid4(),
        score=0.5,
        payload=ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=chunk_id,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            page_start=1,
            page_end=1,
            source_block_ids=source_block_ids or [f"{chunk_id}-block"],
            section_path=["第一章"],
            content_types=["text"],
            asset_refs=[],
            text=f"{chunk_id} 的答案正文",
            chunking_config_fingerprint="a" * 64,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_model_revision="revision",
            embedding_config_fingerprint="b" * 64,
        ),
    )


def _evidence(
    chunk_id: str,
    block_id: str,
    *,
    document_id: UUID = DOC,
    knowledge_base_id: UUID = KB,
) -> GraphEvidence:
    provenance = GraphProvenance(
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["第一章"],
        extraction_provenance=None,
    )
    return GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["第一章"],
        extraction_provenance=None,
    )


def _graph_item(
    evidence: GraphEvidence,
    *,
    score: float = 0.8,
    seed_entity_id: str | None = None,
) -> GraphEvidenceResult:
    seed = seed_entity_id or entity_id_for(
        "种子",
        "概念",
        knowledge_base_id=evidence.knowledge_base_id,
        document_id=evidence.document_id,
    )
    path = GraphPath(
        seed_entity_id=seed,
        entity_ids=[seed],
        hop_distance=0,
        kind="seed",
    )
    return GraphEvidenceResult(
        evidence=evidence,
        score=score,
        seed_entity_id=seed,
        retrieval_reason="seed_entity_evidence",
        paths=[path],
    )


def _graph_result(
    items: list[GraphEvidenceResult],
    *,
    knowledge_base_id: UUID = KB,
) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        query="查询",
        knowledge_base_id=knowledge_base_id,
        items=items,
    )


@dataclass
class _FakeVectorRetrieval:
    items: tuple[RetrievedChunk, ...]
    calls: list[tuple[str, int, UUID | None, UUID | None]] = field(default_factory=list)
    error: Exception | None = None
    barrier: Barrier | None = None

    def search(
        self,
        query: str,
        *,
        limit: int,
        knowledge_base_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> RetrievalResult:
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.error is not None:
            raise self.error
        self.calls.append((query, limit, knowledge_base_id, document_id))
        return RetrievalResult(
            query=query,
            limit=limit,
            model_id="fake-embedding",
            collection_name="fake",
            items=self.items,
        )


@dataclass
class _FakeGraphRetrieval:
    result: GraphRetrievalResult
    error: Exception | None = None
    barrier: Barrier | None = None
    calls: list[UUID | None] = field(default_factory=list)

    def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> GraphRetrievalResult:
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.error is not None:
            raise self.error
        assert query == self.result.query
        assert knowledge_base_id == self.result.knowledge_base_id
        self.calls.append(document_id)
        return self.result


class _FakeReranker:
    model_id = "fake-reranker"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score_pairs(self, _query: str, _passages: list[str]) -> list[float]:
        return self.scores


def _service(
    vector: _FakeVectorRetrieval,
    graph: _FakeGraphRetrieval,
    *,
    config: HybridRetrievalConfig | None = None,
    reranker_scores: list[float] | None = None,
) -> HybridRetrievalService:
    return HybridRetrievalService(
        vector,
        RerankingService(_FakeReranker(reranker_scores or [0.5] * len(vector.items))),
        graph,
        config=config,
    )


def test_rrf_deduplicates_chunk_and_keeps_both_branch_lineage() -> None:
    vector_items = (
        _chunk("chunk-a", source_block_ids=["block-a"]),
        _chunk("chunk-b", source_block_ids=["chunk-b-block", "graph-b", "graph-b-2"]),
    )
    graph_items = [
        _graph_item(_evidence("chunk-b", "graph-b"), score=0.9),
        _graph_item(_evidence("chunk-b", "graph-b-2"), score=0.8),
        _graph_item(_evidence("chunk-c", "graph-c"), score=0.7),
    ]
    service = _service(
        _FakeVectorRetrieval(vector_items),
        _FakeGraphRetrieval(_graph_result(graph_items)),
        reranker_scores=[0.1, 0.9],
    )

    result = asyncio.run(service.search("查询", KB))

    assert [item.chunk_id for item in result.items] == ["chunk-b", "chunk-a", "chunk-c"]
    both = result.items[0]
    assert both.source == "both"
    assert both.vector_rank == 1
    assert both.dense_rank == 2
    assert both.graph_rank == 1
    assert both.graph_contribution is not None
    assert len(both.graph_contribution.evidence_ids) == 2
    assert both.source_block_ids == ["chunk-b-block", "graph-b", "graph-b-2"]
    assert both.vector_rrf_contribution == round(1 / 61, 12)
    assert both.graph_rrf_contribution == round(1 / 61, 12)
    assert result.source_distribution == {"vector": 1, "graph": 1, "both": 1}


def test_rrf_is_rank_based_and_does_not_add_duplicate_graph_rank_contributions() -> None:
    service = _service(
        _FakeVectorRetrieval(
            (_chunk("chunk-a", source_block_ids=["chunk-a-block", "block-a", "block-b"]),)
        ),
        _FakeGraphRetrieval(
            _graph_result(
                [
                    _graph_item(_evidence("chunk-a", "block-a")),
                    _graph_item(_evidence("chunk-a", "block-b")),
                ]
            )
        ),
    )

    result = asyncio.run(service.search("查询", KB))

    assert len(result.items) == 1
    item = result.items[0]
    assert item.graph_rank == 1
    assert item.graph_rrf_contribution == round(1 / 61, 12)
    assert item.fusion_score == round(2 / 61, 12)


def test_conflicting_cross_branch_lineage_is_rejected_before_fusion() -> None:
    evidence_provenance = GraphProvenance(
        document_id=DOC,
        knowledge_base_id=KB,
        chunk_id="chunk-a",
        page_start=2,
        page_end=2,
        source_block_ids=["foreign-block"],
        section_path=["错误章节"],
    )
    evidence = GraphEvidence(
        evidence_id=evidence_id_for(evidence_provenance),
        knowledge_base_id=KB,
        document_id=DOC,
        chunk_id="chunk-a",
        page_start=2,
        page_end=2,
        source_block_ids=["foreign-block"],
        section_path=["错误章节"],
    )
    service = _service(
        _FakeVectorRetrieval((_chunk("chunk-a", source_block_ids=["chunk-a-block"]),)),
        _FakeGraphRetrieval(_graph_result([_graph_item(evidence)])),
    )

    with pytest.raises(HybridRetrievalError, match="conflicting"):
        asyncio.run(service.search("查询", KB))


def test_duplicate_vector_rows_do_not_consume_vector_rank() -> None:
    duplicate = _chunk("duplicate")
    service = _service(
        _FakeVectorRetrieval((duplicate, duplicate, _chunk("unique"))),
        _FakeGraphRetrieval(_graph_result([])),
        reranker_scores=[0.9, 0.9, 0.8],
    )

    result = asyncio.run(service.search("查询", KB))

    assert [(item.chunk_id, item.vector_rank) for item in result.items] == [
        ("duplicate", 1),
        ("unique", 2),
    ]


def test_duplicate_graph_rows_do_not_consume_graph_rank() -> None:
    service = _service(
        _FakeVectorRetrieval(()),
        _FakeGraphRetrieval(
            _graph_result(
                [
                    _graph_item(_evidence("duplicate", "duplicate-block-1")),
                    _graph_item(_evidence("duplicate", "duplicate-block-2")),
                    _graph_item(_evidence("unique", "unique-block")),
                ]
            )
        ),
    )

    result = asyncio.run(service.search("查询", KB))

    assert [(item.chunk_id, item.graph_rank) for item in result.items] == [
        ("duplicate", 1),
        ("unique", 2),
    ]


def test_timeout_is_visible_as_a_timed_out_branch() -> None:
    result = asyncio.run(
        _service(
            _FakeVectorRetrieval((), error=TimeoutError()),
            _FakeGraphRetrieval(_graph_result([])),
        ).search("查询", KB)
    )

    assert result.vector_status == "timed_out"
    assert result.vector_error == "vector retrieval timed out"
    assert result.degraded is True


def test_wrapped_timeout_is_visible_as_a_timed_out_branch() -> None:
    wrapped_error = RuntimeError("transport failure")
    wrapped_error.__cause__ = TimeoutError()
    result = asyncio.run(
        _service(
            _FakeVectorRetrieval((), error=wrapped_error),
            _FakeGraphRetrieval(_graph_result([])),
        ).search("查询", KB)
    )

    assert result.vector_status == "timed_out"
    assert result.vector_error == "vector retrieval timed out"


def test_empty_branch_is_visible_and_other_branch_is_not_marked_degraded() -> None:
    result = asyncio.run(
        _service(
            _FakeVectorRetrieval(()),
            _FakeGraphRetrieval(_graph_result([])),
        ).search("查询", KB)
    )

    assert result.items == []
    assert result.vector_status == "empty"
    assert result.graph_status == "empty"
    assert result.degraded is False
    assert result.source_distribution == {"vector": 0, "graph": 0, "both": 0}


def test_degraded_mode_returns_surviving_branch_and_exposes_failure() -> None:
    graph_item = _graph_item(_evidence("chunk-g", "block-g"))
    result = asyncio.run(
        _service(
            _FakeVectorRetrieval((), error=RuntimeError("secret qdrant detail")),
            _FakeGraphRetrieval(_graph_result([graph_item])),
        ).search("查询", KB)
    )

    assert result.degraded is True
    assert result.vector_status == "failed"
    assert result.vector_error == "vector retrieval failed"
    assert result.graph_status == "success"
    assert [item.source for item in result.items] == ["graph"]


def test_strict_mode_rejects_a_failed_branch() -> None:
    service = _service(
        _FakeVectorRetrieval((), error=RuntimeError("qdrant unavailable")),
        _FakeGraphRetrieval(_graph_result([])),
        config=HybridRetrievalConfig(failure_mode="strict"),
    )

    with pytest.raises(HybridRetrievalError, match="vector branch failed"):
        asyncio.run(service.search("查询", KB))


def test_both_failed_branches_do_not_look_like_an_empty_success() -> None:
    service = _service(
        _FakeVectorRetrieval((), error=RuntimeError("qdrant unavailable")),
        _FakeGraphRetrieval(_graph_result([]), error=RuntimeError("neo4j unavailable")),
    )

    with pytest.raises(HybridRetrievalError, match="both hybrid retrieval branches failed"):
        asyncio.run(service.search("查询", KB))


def test_wrong_vector_kb_is_rejected_before_fusion() -> None:
    service = _service(
        _FakeVectorRetrieval((_chunk("wrong", knowledge_base_id=OTHER_KB),)),
        _FakeGraphRetrieval(_graph_result([])),
    )

    result = asyncio.run(service.search("查询", KB))

    assert result.vector_status == "failed"
    assert result.vector_error == "vector result has invalid knowledge-base lineage"
    assert result.items == []


def test_document_filter_applies_to_both_retrieval_branches() -> None:
    requested_document = uuid4()
    vector = _FakeVectorRetrieval(
        (_chunk("requested", document_id=requested_document),),
    )
    graph = _FakeGraphRetrieval(
        _graph_result(
            [
                _graph_item(_evidence("other", "other-block")),
                _graph_item(
                    _evidence("requested", "requested-block", document_id=requested_document)
                ),
            ]
        )
    )

    result = asyncio.run(_service(vector, graph).search("查询", KB, document_id=requested_document))

    assert [item.chunk_id for item in result.items] == ["requested"]
    assert result.vector_status == "success"
    assert result.graph_status == "success"
    assert graph.calls == [requested_document]


def test_mutated_graph_lineage_is_revalidated_before_fusion() -> None:
    graph_item = _graph_item(_evidence("chunk-g", "block-g"))
    graph_result = _graph_result([graph_item])
    graph_result.items[0].evidence.knowledge_base_id = OTHER_KB
    service = _service(
        _FakeVectorRetrieval(()),
        _FakeGraphRetrieval(graph_result),
    )

    result = asyncio.run(service.search("查询", KB))

    assert result.graph_status == "failed"
    assert result.graph_error == "graph result has invalid evidence lineage"
    assert result.items == []


def test_result_limit_bounds_fused_results() -> None:
    service = _service(
        _FakeVectorRetrieval((_chunk("chunk-a"), _chunk("chunk-b"))),
        _FakeGraphRetrieval(_graph_result([_graph_item(_evidence("chunk-c", "block-c"))])),
        config=HybridRetrievalConfig(result_limit=2),
    )

    result = asyncio.run(service.search("查询", KB))

    assert len(result.items) == 2


def test_vector_and_graph_branches_run_concurrently_in_worker_threads() -> None:
    barrier = Barrier(2)
    vector = _FakeVectorRetrieval((_chunk("chunk-a"),), barrier=barrier)
    graph = _FakeGraphRetrieval(_graph_result([]), barrier=barrier)

    result = asyncio.run(_service(vector, graph).search("查询", KB))

    assert result.vector_status == "success"
    assert result.graph_status == "empty"


def test_hybrid_config_rejects_rerank_limit_above_candidate_limit() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        HybridRetrievalConfig(vector_candidate_limit=5, vector_rerank_limit=6)


def test_reranked_chunk_can_be_used_as_a_typed_vector_contribution() -> None:
    chunk = _chunk("chunk-a")
    assert isinstance(
        RerankedChunk(chunk=chunk, dense_rank=1, reranker_score=0.5),
        RerankedChunk,
    )
