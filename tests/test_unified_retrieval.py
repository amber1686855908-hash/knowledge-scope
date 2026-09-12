from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from knowledge_scope.graph.models import GraphProvenance, entity_id_for, evidence_id_for
from knowledge_scope.graph.retrieval import GraphEvidence, GraphEvidenceResult, GraphPath
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.representation_index import (
    RepresentationEvidenceHit,
    RepresentationRetrievalResult,
    RetrievedEvidence,
)
from knowledge_scope.retrieval.service import RetrievalResult
from knowledge_scope.retrieval.sparse import SparseSearchHit, SparseSearchResponse
from knowledge_scope.retrieval.unified import (
    UnifiedBranchContribution,
    UnifiedCandidate,
    UnifiedRetrievalConfig,
    UnifiedRetrievalError,
    UnifiedRetrievalService,
    chunk_candidate_id,
)

KB = UUID("11111111-1111-4111-8111-111111111111")
OTHER_KB = UUID("22222222-2222-4222-8222-222222222222")
DOC = UUID("33333333-3333-4333-8333-333333333333")
OTHER_DOC = UUID("44444444-4444-4444-8444-444444444444")


def _payload(
    chunk_id: str,
    *,
    knowledge_base_id: UUID = KB,
    document_id: UUID = DOC,
    text: str | None = None,
) -> ChunkVectorPayload:
    return ChunkVectorPayload(
        collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
        chunk_id=chunk_id,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["第一章"],
        content_types=["text"],
        asset_refs=[],
        text=text or f"{chunk_id} answer",
        chunking_config_fingerprint="a" * 64,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_model_revision="revision",
        embedding_config_fingerprint="b" * 64,
    )


def _chunk(chunk_id: str, **kwargs: object) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=uuid4(),
        score=0.5,
        payload=_payload(chunk_id, **kwargs),
    )


def _sparse_hit(
    chunk_id: str,
    *,
    score: float = 1.0,
    text: str | None = None,
) -> SparseSearchHit:
    payload = _payload(chunk_id, text=text)
    return SparseSearchHit(
        rank=1,
        bm25_score=score,
        knowledge_base_id=payload.knowledge_base_id or KB,
        document_id=payload.document_id,
        chunk_id=payload.chunk_id,
        ordinal=0,
        page_start=payload.page_start,
        page_end=payload.page_end,
        source_block_ids=payload.source_block_ids,
        section_path=payload.section_path,
        content_types=payload.content_types,
        asset_refs=payload.asset_refs,
        text=payload.text,
        matched_terms=["answer"],
    )


def _graph_item(chunk_id: str, *, score: float = 0.8) -> GraphEvidenceResult:
    entity_id = entity_id_for(
        "种子",
        "概念",
        knowledge_base_id=KB,
        document_id=DOC,
    )
    provenance = GraphProvenance(
        document_id=DOC,
        knowledge_base_id=KB,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["第一章"],
        extraction_provenance=None,
    )
    evidence = GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=KB,
        document_id=DOC,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["第一章"],
        extraction_provenance=None,
    )
    return GraphEvidenceResult(
        evidence=evidence,
        score=score,
        seed_entity_id=entity_id,
        retrieval_reason="seed_entity_evidence",
        paths=[
            GraphPath(
                seed_entity_id=entity_id,
                entity_ids=[entity_id],
                hop_distance=0,
                kind="seed",
            )
        ],
    )


def _multimodal_item(
    *,
    evidence_id: str = "evidence_v1_" + "c" * 64,
    document_id: UUID = DOC,
) -> RetrievedEvidence:
    representation_id = "representation_v1_" + "d" * 64
    return RetrievedEvidence(
        rank=1,
        score=0.9,
        evidence_id=evidence_id,
        knowledge_base_id=KB,
        document_id=document_id,
        page_start=2,
        page_end=2,
        source_block_ids=["image-block"],
        section_path=["第二章"],
        asset_refs=["images/figure.png"],
        modality="image",
        representation_id=representation_id,
        representation_type="caption",
        text="设备结构示意图",
        reference=None,
        representations=[
            RepresentationEvidenceHit(
                representation_id=representation_id,
                representation_type="caption",
                score=0.9,
                rank=1,
                text="设备结构示意图",
                reference=None,
            )
        ],
    )


@dataclass
class _Dense:
    items: tuple[RetrievedChunk, ...]

    def search(self, query: str, *, limit: int, knowledge_base_id: UUID, document_id=None):
        return RetrievalResult(query, limit, "fake-dense", "chunks", self.items[:limit])


@dataclass
class _Sparse:
    items: tuple[SparseSearchHit, ...]
    error: Exception | None = None

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int,
        document_id=None,
    ):
        if self.error is not None:
            raise self.error
        return SparseSearchResponse(
            query=query,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            top_k=top_k,
            index_fingerprint="e" * 64,
            items=list(self.items[:top_k]),
        )


@dataclass
class _RawItems:
    items: tuple[object, ...]


@dataclass
class _PartiallyInvalidSparse:
    valid_item: SparseSearchHit

    def search(self, query: str, *, knowledge_base_id: UUID, top_k: int, document_id=None):
        return _RawItems((self.valid_item, object()))


@dataclass
class _Graph:
    items: tuple[GraphEvidenceResult, ...]
    error: Exception | None = None

    def search(self, query: str, knowledge_base_id: UUID, *, document_id=None):
        if self.error is not None:
            raise self.error
        from knowledge_scope.graph.retrieval import GraphRetrievalResult

        return GraphRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            items=list(self.items),
        )


@dataclass
class _RawGraph:
    """Expose duplicate raw graph items to exercise unified defensive deduping."""

    items: tuple[object, ...]

    def search(self, query: str, knowledge_base_id: UUID, *, document_id=None):
        return _RawItems(self.items)


@dataclass
class _Multimodal:
    items: tuple[RetrievedEvidence, ...]

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: UUID,
        top_k: int,
        document_id: UUID | None = None,
    ):
        items = self.items
        if document_id is not None:
            items = tuple(item for item in items if item.document_id == document_id)
        return RepresentationRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            items=list(items[:top_k]),
            raw_hits=[],
        )


@dataclass
class _Lookup:
    payloads: dict[str, ChunkVectorPayload]

    def get_chunk_payload(
        self,
        *,
        knowledge_base_id: UUID,
        document_id: UUID,
        chunk_id: str,
    ) -> ChunkVectorPayload | None:
        payload = self.payloads.get(chunk_id)
        if payload is None:
            return None
        if payload.knowledge_base_id != knowledge_base_id or payload.document_id != document_id:
            raise RuntimeError("lookup scope mismatch")
        return payload


class _Reranker:
    model_id = "BAAI/bge-reranker-v2-m3"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[list[str]] = []

    def score_pairs(self, _query: str, passages: list[str]) -> list[float]:
        self.calls.append(list(passages))
        return [self.scores.get(passage, 0.0) for passage in passages]


def _service(
    dense: _Dense,
    sparse: _Sparse,
    graph: _Graph,
    multimodal: _Multimodal,
    reranker: _Reranker,
    *,
    config: UnifiedRetrievalConfig | None = None,
    payloads: dict[str, ChunkVectorPayload] | None = None,
) -> UnifiedRetrievalService:
    return UnifiedRetrievalService(
        dense,
        sparse,
        graph,
        multimodal,
        reranker,
        chunk_lookup=_Lookup(payloads or {}),
        config=config,
    )


def test_unified_deduplicates_chunks_and_preserves_evidence_provenance() -> None:
    chunk_a = _chunk("chunk-a", text="低分 chunk")
    chunk_b = _chunk("chunk-b", text="高分 chunk")
    graph_item = _graph_item("chunk-a")
    reranker = _Reranker({"低分 chunk": 0.1, "高分 chunk": 0.9, "设备结构示意图": 0.8})
    result = asyncio.run(
        _service(
            _Dense((chunk_a, chunk_b)),
            _Sparse((_sparse_hit("chunk-a", text="低分 chunk"),)),
            _Graph((graph_item,)),
            _Multimodal((_multimodal_item(),)),
            reranker,
            payloads={"chunk-a": chunk_a.payload, "chunk-b": chunk_b.payload},
        ).search("查询", KB)
    )

    assert result.degraded is False
    assert result.candidate_pool_size == 3
    assert result.source_distribution == {
        "dense": 1,
        "sparse": 0,
        "graph": 0,
        "multimodal": 1,
        "multiple": 1,
    }
    assert result.items[0].chunk_id == "chunk-b"
    chunk_a_result = next(item for item in result.items if item.chunk_id == "chunk-a")
    assert chunk_a_result.source == "multiple"
    assert {branch.branch for branch in chunk_a_result.branches} == {"dense", "sparse", "graph"}
    assert chunk_a_result.evidence_ids == [graph_item.evidence.evidence_id]
    evidence_result = next(item for item in result.items if item.candidate_kind == "evidence")
    assert evidence_result.evidence_id == "evidence_v1_" + "c" * 64
    assert evidence_result.representation_ids == ["representation_v1_" + "d" * 64]
    assert reranker.calls == [["低分 chunk", "设备结构示意图", "高分 chunk"]]


def test_unified_pool_uses_deterministic_round_robin_bounds() -> None:
    chunks = tuple(_chunk(f"dense-{index}") for index in range(3))
    sparse = tuple(_sparse_hit(f"sparse-{index}") for index in range(3))
    graph = tuple(_graph_item(f"graph-{index}") for index in range(3))
    payloads = {
        **{item.payload.chunk_id: item.payload for item in chunks},
        **{item.chunk_id: _payload(item.chunk_id) for item in sparse},
        **{item.evidence.chunk_id: _payload(item.evidence.chunk_id) for item in graph},
    }
    result = asyncio.run(
        _service(
            _Dense(chunks),
            _Sparse(sparse),
            _Graph(graph),
            _Multimodal(()),
            _Reranker(),
            config=UnifiedRetrievalConfig(candidate_pool_limit=4, result_limit=4),
            payloads=payloads,
        ).search("查询", KB)
    )

    assert result.candidate_pool_size == 4
    assert {item.branches[0].branch for item in result.items} == {
        "dense",
        "sparse",
        "graph",
    }


def test_chunk_deduplication_keeps_same_local_id_in_distinct_documents() -> None:
    result = asyncio.run(
        _service(
            _Dense(
                (
                    _chunk("shared", document_id=DOC),
                    _chunk("shared", document_id=OTHER_DOC),
                )
            ),
            _Sparse(()),
            _Graph(()),
            _Multimodal(()),
            _Reranker(),
            config=UnifiedRetrievalConfig(candidate_pool_limit=2, result_limit=2),
        ).search("查询", KB)
    )

    assert result.candidate_pool_size == 2
    assert {item.document_id for item in result.items} == {DOC, OTHER_DOC}


def test_document_filter_reaches_multimodal_branch_before_top_k() -> None:
    result = asyncio.run(
        _service(
            _Dense(()),
            _Sparse(()),
            _Graph(()),
            _Multimodal(
                (
                    _multimodal_item(
                        evidence_id="evidence_v1_" + "e" * 64,
                        document_id=OTHER_DOC,
                    ),
                    _multimodal_item(
                        evidence_id="evidence_v1_" + "f" * 64,
                        document_id=DOC,
                    ),
                )
            ),
            _Reranker(),
            config=UnifiedRetrievalConfig(
                multimodal_candidate_limit=1,
                candidate_pool_limit=1,
                result_limit=1,
            ),
        ).search("查询", KB, document_id=DOC)
    )

    assert result.candidate_pool_size == 1
    assert result.items[0].document_id == DOC


def test_unified_applies_graph_candidate_bound() -> None:
    graph = _Graph(tuple(_graph_item(f"graph-{index}") for index in range(3)))
    result = asyncio.run(
        _service(
            _Dense(()),
            _Sparse(()),
            graph,
            _Multimodal(()),
            _Reranker(),
            config=UnifiedRetrievalConfig(
                graph_candidate_limit=1,
                candidate_pool_limit=1,
                result_limit=1,
            ),
            payloads={f"graph-{index}": _payload(f"graph-{index}") for index in range(3)},
        ).search("查询", KB)
    )

    graph_branch = next(branch for branch in result.branches if branch.branch == "graph")
    assert graph_branch.status == "success"
    assert graph_branch.candidate_count == 1
    assert result.candidate_pool_size == 1


def test_graph_evidence_must_match_current_chunk_lineage() -> None:
    valid = _graph_item("graph")
    invalid_evidence = valid.evidence.model_copy(update={"chunk_id": "other"})
    invalid = valid.model_copy(update={"evidence": invalid_evidence})
    result = asyncio.run(
        _service(
            _Dense(()),
            _Sparse(()),
            _Graph((invalid,)),
            _Multimodal(()),
            _Reranker(),
            payloads={"graph": _payload("graph")},
        ).search("查询", KB)
    )

    graph_branch = next(branch for branch in result.branches if branch.branch == "graph")
    assert graph_branch.status == "failed"
    assert result.items == []


def test_cross_knowledge_base_candidates_are_rejected() -> None:
    with pytest.raises(UnifiedRetrievalError, match="dense branch failed"):
        asyncio.run(
            _service(
                _Dense((_chunk("foreign", knowledge_base_id=OTHER_KB),)),
                _Sparse(()),
                _Graph(()),
                _Multimodal(()),
                _Reranker(),
                config=UnifiedRetrievalConfig(failure_mode="strict"),
            ).search("查询", KB)
        )


def test_duplicate_graph_hits_do_not_create_duplicate_contributions() -> None:
    graph_item = _graph_item("graph")
    result = asyncio.run(
        _service(
            _Dense(()),
            _Sparse(()),
            _RawGraph((graph_item, graph_item)),
            _Multimodal(()),
            _Reranker(),
            config=UnifiedRetrievalConfig(candidate_pool_limit=1, result_limit=1),
            payloads={"graph": _payload("graph")},
        ).search("查询", KB)
    )

    assert len(result.items) == 1
    graph_contribution = result.items[0].branches[0]
    assert graph_contribution.branch == "graph"
    assert graph_contribution.rank == 1
    assert len(graph_contribution.graph_paths) == 1
    assert result.items[0].evidence_ids == [graph_item.evidence.evidence_id]


def test_unified_degraded_mode_exposes_failed_branch_and_strict_mode_raises() -> None:
    dense = _Dense((_chunk("dense"),))
    sparse = _Sparse((), error=RuntimeError("sparse unavailable"))
    graph = _Graph(())
    multimodal = _Multimodal(())
    degraded = asyncio.run(
        _service(
            dense,
            sparse,
            graph,
            multimodal,
            _Reranker(),
            payloads={"dense": dense.items[0].payload},
        ).search("查询", KB)
    )
    assert degraded.degraded is True
    assert (
        next(branch for branch in degraded.branches if branch.branch == "sparse").status == "failed"
    )

    with pytest.raises(UnifiedRetrievalError, match="sparse branch failed"):
        asyncio.run(
            _service(
                dense,
                sparse,
                graph,
                multimodal,
                _Reranker(),
                config=UnifiedRetrievalConfig(failure_mode="strict"),
                payloads={"dense": dense.items[0].payload},
            ).search("查询", KB)
        )


def test_failed_normalization_discards_partial_branch_candidates() -> None:
    dense = _Dense((_chunk("dense"),))
    sparse_item = _sparse_hit("sparse")
    result = asyncio.run(
        _service(
            dense,
            _PartiallyInvalidSparse(sparse_item),
            _Graph(()),
            _Multimodal(()),
            _Reranker(),
            payloads={"dense": dense.items[0].payload},
        ).search("查询", KB)
    )

    sparse_branch = next(branch for branch in result.branches if branch.branch == "sparse")
    assert sparse_branch.status == "failed"
    assert sparse_branch.candidate_count == 2
    assert result.candidate_pool_size == 1
    assert [item.chunk_id for item in result.items] == ["dense"]


def test_non_finite_final_reranker_score_is_rejected() -> None:
    dense = _Dense((_chunk("dense"),))
    with pytest.raises(UnifiedRetrievalError, match="non-finite"):
        asyncio.run(
            _service(
                dense,
                _Sparse(()),
                _Graph(()),
                _Multimodal(()),
                _Reranker({"dense answer": math.nan}),
                payloads={"dense": dense.items[0].payload},
            ).search("查询", KB)
        )


def test_unified_cancellation_propagates() -> None:
    class _CancelledDense(_Dense):
        def search(self, query: str, *, limit: int, knowledge_base_id: UUID, document_id=None):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _service(
                _CancelledDense(()),
                _Sparse(()),
                _Graph(()),
                _Multimodal(()),
                _Reranker(),
            ).search("查询", KB)
        )


def test_unified_candidate_identity_rejects_mismatch() -> None:
    payload = _payload("chunk-a")
    with pytest.raises(ValueError, match="candidate ID"):
        UnifiedCandidate(
            candidate_id=chunk_candidate_id(KB, DOC, "other"),
            candidate_kind="chunk",
            source="dense",
            knowledge_base_id=KB,
            document_id=DOC,
            chunk_id=payload.chunk_id,
            page_start=payload.page_start,
            page_end=payload.page_end,
            source_block_ids=payload.source_block_ids,
            content_types=payload.content_types,
            rerank_text=payload.text,
            branches=[UnifiedBranchContribution(branch="dense", rank=1, score=0.5)],
        )
