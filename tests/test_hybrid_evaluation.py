from __future__ import annotations

from uuid import UUID

import pytest

from knowledge_scope.evaluation.embedding_benchmark import FrozenEvalCase
from knowledge_scope.evaluation.hybrid_evaluation import (
    HybridEvaluationError,
    HybridEvaluationProtocol,
    _frozen_a25_protocol,
    _validated_graph_prompt_layout,
    evaluate_query_record,
    summarize_branch_observations,
    summarize_contribution,
)
from knowledge_scope.evaluation.retrieval_eval import (
    IndexedChunk,
    RetrievalEvalItem,
    RetrievalEvidence,
    deterministic_item_id,
)
from knowledge_scope.graph.models import GraphProvenance, entity_id_for, evidence_id_for
from knowledge_scope.graph.retrieval import (
    GraphEvidence,
    GraphEvidenceResult,
    GraphPath,
    GraphRetrievalResult,
)
from knowledge_scope.retrieval.hybrid import HybridGraphContribution, HybridResult
from knowledge_scope.shared.config import Settings

KB = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT = UUID("22222222-2222-4222-8222-222222222222")


def _case() -> FrozenEvalCase:
    evidence = RetrievalEvidence(
        document_id=DOCUMENT,
        page_number=1,
        source_block_ids=["gold-block"],
    )
    query = "什么是金标准?"
    item = RetrievalEvalItem(
        item_id=deterministic_item_id(query, "化学", "definition", [evidence], "a" * 64),
        query=query,
        subject="化学",
        query_type="definition",
        verification_status="verified",
        evidence=[evidence],
        evidence_fingerprint="a" * 64,
    )
    return FrozenEvalCase(
        item=item,
        relevant_chunk_ids=frozenset({"gold-chunk"}),
        gold_source_blocks=frozenset({(str(DOCUMENT), "gold-block")}),
    )


def _chunk(chunk_id: str, block_id: str) -> IndexedChunk:
    return IndexedChunk(
        schema_version="1.0",
        chunk_id=chunk_id,
        document_id=DOCUMENT,
        ordinal=0,
        text=f"正文 {chunk_id}",
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["第一章"],
        content_types=["text"],
        asset_refs=[],
        config_fingerprint="b" * 64,
    )


def _graph_result() -> GraphRetrievalResult:
    provenance = GraphProvenance(
        knowledge_base_id=KB,
        document_id=DOCUMENT,
        chunk_id="gold-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["gold-block"],
        section_path=["第一章"],
    )
    evidence = GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=KB,
        document_id=DOCUMENT,
        chunk_id="gold-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["gold-block"],
        section_path=["第一章"],
    )
    seed = entity_id_for("金标准", "概念", knowledge_base_id=KB, document_id=DOCUMENT)
    return GraphRetrievalResult(
        query="什么是金标准?",
        knowledge_base_id=KB,
        seeds=[],
        items=[
            GraphEvidenceResult(
                evidence=evidence,
                score=0.9,
                seed_entity_id=seed,
                retrieval_reason="seed_entity_evidence",
                paths=[
                    GraphPath(
                        seed_entity_id=seed,
                        entity_ids=[seed],
                        hop_distance=0,
                        kind="seed",
                    )
                ],
            )
        ],
    )


def _hybrid_graph_item() -> HybridResult:
    graph = _graph_result().items[0]
    return HybridResult(
        knowledge_base_id=KB,
        document_id=DOCUMENT,
        chunk_id="gold-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["gold-block"],
        section_path=["第一章"],
        evidence_ids=[graph.evidence.evidence_id],
        source="graph",
        graph_rank=1,
        graph_score=graph.score,
        vector_rrf_contribution=0,
        graph_rrf_contribution=round(1 / 61, 12),
        fusion_score=round(1 / 61, 12),
        graph_contribution=HybridGraphContribution(
            rank=1,
            score=graph.score,
            seed_entity_id=graph.seed_entity_id,
            retrieval_reason=graph.retrieval_reason,
            evidence_ids=[graph.evidence.evidence_id],
            paths=graph.paths,
        ),
    )


def test_query_record_tracks_graph_recovery_and_a2_metrics() -> None:
    case = _case()
    chunks = {
        "other-chunk": _chunk("other-chunk", "other-block"),
        "gold-chunk": _chunk("gold-chunk", "gold-block"),
    }
    record = evaluate_query_record(
        case,
        split="test",
        coverage="complete",
        chunks_by_id=chunks,
        vector_ranked_chunk_ids=["other-chunk"],
        graph_result=_graph_result(),
        hybrid_items=[_hybrid_graph_item()],
    )

    assert record.split == "test"
    assert record.vector_metrics["hit@10"] == 0.0
    assert record.hybrid_metrics["hit@10"] == 1.0
    assert record.vector_gold_chunk_ids_at_10 == []
    assert record.graph_gold_chunk_ids_at_10 == ["gold-chunk"]
    assert record.vector_missed_graph_recovered_source_blocks_at_10 == [f"{DOCUMENT}:gold-block"]
    assert record.graph_only_final_recovered_gold_chunk_ids_at_10 == ["gold-chunk"]
    assert record.graph_only_final_recovered_source_blocks_at_10 == [f"{DOCUMENT}:gold-block"]
    assert record.vector_first_relevant_rank is None
    assert record.hybrid_first_relevant_rank == 1
    assert record.mrr_delta == 1.0

    summary = summarize_contribution([record])
    assert summary.improved_query_count == 1
    assert summary.regressed_query_count == 0
    assert summary.graph_only_recovered_gold_chunk_count == 1
    assert summary.graph_only_recovered_gold_source_block_count == 1
    assert summary.by_k["10"].improved_query_count == 1


def test_graph_coverage_class_is_retained_in_machine_record() -> None:
    case = _case()
    record = evaluate_query_record(
        case,
        coverage="none",
        chunks_by_id={"gold-chunk": _chunk("gold-chunk", "gold-block")},
        vector_ranked_chunk_ids=["gold-chunk"],
        graph_result=GraphRetrievalResult(
            query=case.item.query,
            knowledge_base_id=KB,
            items=[],
        ),
        hybrid_items=[],
    )

    assert record.coverage == "none"
    assert record.graph_status == "success"
    summary = summarize_contribution([record])
    assert summary.regressed_query_count == 1
    assert summary.by_k["1"].regressed_query_count == 1


def test_branch_observations_count_results_and_overlap() -> None:
    case = _case()
    chunks = {
        "other-chunk": _chunk("other-chunk", "other-block"),
        "gold-chunk": _chunk("gold-chunk", "gold-block"),
    }
    graph = _graph_result()
    both_item = _hybrid_graph_item().model_copy(
        update={
            "source": "both",
            "vector_rank": 1,
            "vector_rrf_contribution": round(1 / 61, 12),
            "fusion_score": round(2 / 61, 12),
        }
    )
    record = evaluate_query_record(
        case,
        coverage="complete",
        chunks_by_id=chunks,
        vector_ranked_chunk_ids=["gold-chunk"],
        graph_result=graph,
        hybrid_items=[both_item],
    )

    observations = summarize_branch_observations([record])

    assert observations.vector_result_count == 1
    assert observations.graph_result_count == 1
    assert observations.hybrid_result_count == 1
    assert observations.both_branch_overlap_chunk_count == 1
    assert observations.both_branch_overlap_query_count == 1
    assert observations.source_distribution == {"vector": 0, "graph": 0, "both": 1}


def test_a37_profile_is_frozen_and_rejects_candidate_drift() -> None:
    settings, protocol = _frozen_a25_protocol(Settings(_env_file=None))

    assert settings.embedding_batch_size == 8
    assert settings.reranker_batch_size == 8
    assert protocol.vector_candidate_limit == 10
    assert protocol.vector_rerank_limit == 10

    incompatible = protocol.model_dump(mode="python")
    incompatible["vector_candidate_limit"] = 20
    with pytest.raises(ValueError, match=r"frozen A2\.5 profile"):
        HybridEvaluationProtocol.model_validate(incompatible)


def test_a37_profile_rejects_incompatible_runtime_revision() -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"embedding_model_revision": "not-the-frozen-revision"}
    )

    with pytest.raises(HybridEvaluationError, match=r"frozen A2\.5 vector profile"):
        _frozen_a25_protocol(settings)


def test_a36_missing_prompt_layout_uses_documented_legacy_compatibility() -> None:
    assert _validated_graph_prompt_layout(None) == "legacy-v1"
    assert _validated_graph_prompt_layout("legacy-v1") == "legacy-v1"
    with pytest.raises(HybridEvaluationError, match="incompatible prompt layout"):
        _validated_graph_prompt_layout("cache-v2")
