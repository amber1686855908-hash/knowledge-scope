from __future__ import annotations

from uuid import UUID

from knowledge_scope.evaluation.graph_retrieval_sample import (
    _result_record,
    build_review_queries,
)
from knowledge_scope.graph.models import (
    GraphEntity,
    GraphProvenance,
    entity_id_for,
    evidence_id_for,
)
from knowledge_scope.graph.retrieval import (
    GraphEntityReference,
    GraphEvidence,
    GraphEvidenceResult,
    GraphPath,
    GraphRetrievalResult,
)
from knowledge_scope.linking.service_types import LocalEntityContext

KB = UUID("66666666-6666-4666-8666-666666666666")
DOC = UUID("77777777-7777-4777-8777-777777777777")


def _entity(name: str = "样本实体") -> GraphEntity:
    provenance = GraphProvenance(
        knowledge_base_id=KB,
        document_id=DOC,
        chunk_id="sample-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["sample-block"],
        section_path=["样本"],
    )
    return GraphEntity(
        entity_id=entity_id_for(name, "概念", knowledge_base_id=KB, document_id=DOC),
        knowledge_base_id=KB,
        document_id=DOC,
        canonical_name=name,
        entity_type="概念",
        provenance=[provenance],
    )


def _evidence() -> GraphEvidence:
    provenance = GraphProvenance(
        knowledge_base_id=KB,
        document_id=DOC,
        chunk_id="sample-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["sample-block"],
        section_path=["样本"],
    )
    return GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=KB,
        document_id=DOC,
        chunk_id="sample-chunk",
        page_start=1,
        page_end=1,
        source_block_ids=["sample-block"],
        section_path=["样本"],
    )


def test_no_seed_control_is_outside_local_graph_vocabulary() -> None:
    context = LocalEntityContext(_entity("样本实体"), subject="测试")
    second_context = LocalEntityContext(_entity("no"), subject="测试")
    queries = build_review_queries((context, second_context), knowledge_base_id=KB)
    control = next(item for item in queries if item["kind"] == "no_graph_seed")

    query = str(control["query"])
    assert "样本实体" not in query
    assert "no" not in query
    assert query not in {"样本实体"}
    assert control["expected_entity_id"] is None


def test_result_record_counts_lineage_per_evidence_result() -> None:
    entity = _entity()
    reference = GraphEntityReference(
        entity_id=entity.entity_id,
        knowledge_base_id=KB,
        document_id=DOC,
        canonical_name=entity.canonical_name,
        entity_type=entity.entity_type,
    )
    evidence = _evidence()
    path = GraphPath(
        seed_entity_id=reference.entity_id,
        entity_ids=[reference.entity_id],
        hop_distance=0,
        kind="seed",
    )
    result = GraphRetrievalResult(
        query="样本实体",
        knowledge_base_id=KB,
        items=[
            GraphEvidenceResult(
                evidence=evidence,
                score=0.5,
                seed_entity_id=reference.entity_id,
                retrieval_reason="seed_entity_evidence",
                paths=[path],
            ),
            GraphEvidenceResult(
                evidence=GraphEvidence(
                    evidence_id=evidence_id_for(
                        GraphProvenance(
                            knowledge_base_id=KB,
                            document_id=DOC,
                            chunk_id="sample-chunk-2",
                            page_start=2,
                            page_end=2,
                            source_block_ids=["sample-block-2"],
                            section_path=["样本"],
                        )
                    ),
                    knowledge_base_id=KB,
                    document_id=DOC,
                    chunk_id="sample-chunk-2",
                    page_start=2,
                    page_end=2,
                    source_block_ids=["sample-block-2"],
                    section_path=["样本"],
                ),
                score=0.4,
                seed_entity_id=reference.entity_id,
                retrieval_reason="seed_entity_evidence",
                paths=[path],
            ),
        ],
    )

    record = _result_record(
        {
            "query_id": "sample",
            "kind": "direct_entity_fact",
            "subject": "测试",
            "query": "样本实体",
            "expected_entity_id": entity.entity_id,
            "source_excerpt": "样本",
        },
        result,
        elapsed_ms=1.0,
    )

    assert len(record["items"]) == 2
    assert len(record["valid_lineage_evidence_ids"]) == 2
    assert record["invalid_lineage_evidence_ids"] == []
