from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import pytest

from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphProvenance,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)
from knowledge_scope.graph.retrieval import (
    GraphCanonicalReference,
    GraphEntityReference,
    GraphEntitySnapshot,
    GraphEvidence,
    GraphEvidenceResult,
    GraphNeighbor,
    GraphPath,
    GraphRelationReference,
    GraphRetrievalConfig,
)
from knowledge_scope.graph.retrieval_service import GraphRetrievalError, GraphRetrievalService

KB = UUID("11111111-1111-4111-8111-111111111111")
OTHER_KB = UUID("22222222-2222-4222-8222-222222222222")
DOC_A = UUID("33333333-3333-4333-8333-333333333333")
DOC_B = UUID("44444444-4444-4444-8444-444444444444")


def _evidence(
    document_id: UUID,
    chunk_id: str,
    block_id: str,
    *,
    knowledge_base_id: UUID = KB,
    extraction: ExtractionProvenance | None = None,
) -> GraphEvidence:
    provenance = GraphProvenance(
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["测试"],
        extraction_provenance=extraction,
    )
    return GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[block_id],
        section_path=["测试"],
        extraction_provenance=extraction,
    )


def _entity(
    name: str,
    document_id: UUID = DOC_A,
    *,
    aliases: list[str] | None = None,
    knowledge_base_id: UUID = KB,
) -> GraphEntityReference:
    return GraphEntityReference(
        entity_id=entity_id_for(
            name,
            "概念",
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        ),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        canonical_name=name,
        entity_type="概念",
        aliases=aliases or [],
    )


def _snapshot(
    entity: GraphEntityReference,
    *,
    evidence: list[GraphEvidence] | None = None,
    canonical: list[GraphCanonicalReference] | None = None,
) -> GraphEntitySnapshot:
    return GraphEntitySnapshot(
        entity=entity,
        evidence=evidence or [],
        canonical_entities=canonical or [],
    )


def _relation(
    source: GraphEntityReference,
    target: GraphEntityReference,
    *,
    relation_type: str = "组成",
) -> GraphRelationReference:
    return GraphRelationReference(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            relation_type,
            knowledge_base_id=KB,
            document_id=source.document_id,
        ),
        knowledge_base_id=KB,
        document_id=source.document_id,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type=relation_type,
    )


@dataclass
class _FakeStore:
    snapshots: tuple[GraphEntitySnapshot, ...]
    neighbors: dict[str, tuple[GraphNeighbor, ...]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def list_retrieval_entities(self, knowledge_base_id: UUID, *, max_entities: int):
        del knowledge_base_id, max_entities
        return self.snapshots

    def get_retrieval_neighbors(
        self,
        knowledge_base_id: UUID,
        entity_id: str,
        *,
        max_neighbors: int,
        max_relations: int,
    ):
        del knowledge_base_id, max_neighbors, max_relations
        self.calls.append(entity_id)
        return self.neighbors.get(entity_id, ())


def test_seed_resolution_prefers_exact_and_supports_aliases_without_short_explosion() -> None:
    entity = _entity("光合作用", aliases=["光合"])
    store = _FakeStore(
        (_snapshot(entity, evidence=[_evidence(DOC_A, "chunk-seed", "block-seed")]),)
    )
    service = GraphRetrievalService(store)

    exact = service.resolve_seed_candidates("光合作用", KB)
    alias = service.resolve_seed_candidates("光合过程", KB)
    generic = service.resolve_seed_candidates("的", KB)

    assert exact[0].method == "exact_local_name"
    assert alias[0].method == "contains_local_alias"
    assert generic == ()


def test_ascii_contains_match_checks_later_standalone_occurrence() -> None:
    entity = _entity("AB")
    store = _FakeStore((_snapshot(entity, evidence=[_evidence(DOC_A, "chunk-ab", "block-ab")]),))

    seeds = GraphRetrievalService(store).resolve_seed_candidates("XAB? AB", KB)

    assert len(seeds) == 1
    assert seeds[0].method == "contains_local_name"


def test_canonical_name_match_returns_local_member_seed() -> None:
    entity = _entity("甲实体")
    entity_evidence = _evidence(DOC_A, "chunk-canonical", "block-canonical")
    canonical = GraphCanonicalReference(
        canonical_entity_id="canonical_entity_v1_" + "a" * 32,
        knowledge_base_id=KB,
        canonical_name="跨文档实体",
        entity_type="概念",
    )
    store = _FakeStore((_snapshot(entity, evidence=[entity_evidence], canonical=[canonical]),))

    seeds = GraphRetrievalService(store).resolve_seed_candidates("跨文档实体", KB)

    assert len(seeds) == 1
    assert seeds[0].seed_entity_id == entity.entity_id
    assert seeds[0].method == "exact_canonical_name"
    assert seeds[0].matched_canonical_entity_id == canonical.canonical_entity_id


def test_unsupported_local_entity_and_canonical_member_are_not_seeds() -> None:
    entity = _entity("无来源实体")
    canonical = GraphCanonicalReference(
        canonical_entity_id="canonical_entity_v1_" + "e" * 32,
        knowledge_base_id=KB,
        canonical_name="无来源规范实体",
        entity_type="概念",
    )
    store = _FakeStore((_snapshot(entity, canonical=[canonical]),))

    assert GraphRetrievalService(store).resolve_seed_candidates("无来源实体", KB) == ()
    assert GraphRetrievalService(store).resolve_seed_candidates("无来源规范实体", KB) == ()


def test_evidence_identity_is_bound_to_a3_1_lineage() -> None:
    extraction = ExtractionProvenance(method="a3.2", model="deepseek", version="prompt-1")
    evidence = _evidence(
        DOC_A,
        "chunk-grounded",
        "block-grounded",
        extraction=extraction,
    )
    assert evidence.evidence_id == evidence_id_for(
        GraphProvenance(
            knowledge_base_id=KB,
            document_id=DOC_A,
            chunk_id="chunk-grounded",
            page_start=1,
            page_end=1,
            source_block_ids=["block-grounded"],
            section_path=["测试"],
            extraction_provenance=extraction,
        )
    )

    base = evidence.model_dump(mode="json")
    mutations = {
        "document_id": str(DOC_B),
        "chunk_id": "chunk-other",
        "source_block_ids": ["block-other"],
        "extraction_provenance": {
            "method": "other-method",
            "model": "other-model",
            "version": "other-version",
        },
    }
    for mutated_field, value in mutations.items():
        with pytest.raises(ValueError, match="evidence_id"):
            GraphEvidence.model_validate({**base, mutated_field: value})


def test_cross_kb_snapshot_is_rejected_by_service() -> None:
    other = _entity("安全实体", knowledge_base_id=OTHER_KB)
    store = _FakeStore((_snapshot(other),))

    with pytest.raises(GraphRetrievalError, match="cross-KB"):
        GraphRetrievalService(store).resolve_seed_candidates("安全实体", KB)


def test_conflicting_duplicate_snapshots_are_rejected() -> None:
    first = _entity("重复实体")
    conflicting = GraphEntityReference.model_construct(
        entity_id=first.entity_id,
        knowledge_base_id=KB,
        document_id=DOC_A,
        canonical_name="另一名称",
        entity_type="概念",
        aliases=[],
    )
    conflicting_snapshot = GraphEntitySnapshot.model_construct(
        entity=conflicting,
        evidence=[],
        canonical_entities=[],
    )
    store = _FakeStore((_snapshot(first), conflicting_snapshot))

    with pytest.raises(GraphRetrievalError, match="conflicting entity snapshots"):
        GraphRetrievalService(store).resolve_seed_candidates("重复实体", KB)


def test_retrieval_references_reject_stale_identity_fields() -> None:
    entity = _entity("实体")

    with pytest.raises(ValueError, match="entity_id"):
        GraphEntityReference(
            entity_id=entity.entity_id,
            knowledge_base_id=KB,
            document_id=DOC_A,
            canonical_name="被篡改的名称",
            entity_type="概念",
        )

    with pytest.raises(ValueError, match="relation_id"):
        GraphRelationReference(
            relation_id=_relation(entity, _entity("邻居")).relation_id,
            knowledge_base_id=KB,
            document_id=DOC_A,
            source_entity_id=entity.entity_id,
            target_entity_id=_entity("另一个邻居").entity_id,
            relation_type="组成",
        )


def test_one_and_two_hop_results_preserve_relation_and_canonical_paths() -> None:
    seed = _entity("起点")
    direct = _entity("直接邻居")
    second = _entity("第二跳")
    seed_evidence = _evidence(DOC_A, "chunk-seed", "block-seed")
    direct_evidence = _evidence(DOC_A, "chunk-direct", "block-direct")
    second_evidence = _evidence(DOC_A, "chunk-second", "block-second")
    canonical = GraphCanonicalReference(
        canonical_entity_id="canonical_entity_v1_" + "b" * 32,
        knowledge_base_id=KB,
        canonical_name="共享起点",
        entity_type="概念",
    )
    relation = _relation(seed, direct)
    second_relation = _relation(direct, second)
    direct_edge = GraphNeighbor(
        seed_entity_id=seed.entity_id,
        neighbor=direct,
        relation=relation,
        direction="forward",
        evidence=[seed_evidence, direct_evidence],
        neighbor_evidence=[direct_evidence],
        relation_evidence=[direct_evidence],
    )
    bridge_edge = GraphNeighbor(
        seed_entity_id=direct.entity_id,
        neighbor=second,
        relation=second_relation,
        direction="forward",
        evidence=[direct_evidence, second_evidence],
        neighbor_evidence=[second_evidence],
        relation_evidence=[direct_evidence],
    )
    store = _FakeStore(
        (
            _snapshot(seed, evidence=[seed_evidence]),
            _snapshot(direct, evidence=[direct_evidence]),
            _snapshot(second, evidence=[second_evidence], canonical=[canonical]),
        ),
        neighbors={seed.entity_id: (direct_edge,), direct.entity_id: (bridge_edge,)},
    )

    result = GraphRetrievalService(store).search("起点", KB)

    assert [item.evidence.chunk_id for item in result.items] == [
        "chunk-seed",
        "chunk-direct",
        "chunk-second",
    ]
    second_result = next(item for item in result.items if item.evidence.chunk_id == "chunk-second")
    assert second_result.paths[0].hop_distance == 2
    assert second_result.paths[0].relation_ids == [
        relation.relation_id,
        second_relation.relation_id,
    ]
    assert len(store.calls) == 2


def test_canonical_bridge_consolidates_duplicate_evidence_and_is_deterministic() -> None:
    first = _entity("第一实体")
    second = _entity("第二实体")
    evidence = _evidence(DOC_A, "chunk-shared", "block-shared")
    canonical = GraphCanonicalReference(
        canonical_entity_id="canonical_entity_v1_" + "c" * 32,
        knowledge_base_id=KB,
        canonical_name="共同实体",
        entity_type="概念",
    )
    bridge = GraphNeighbor(
        seed_entity_id=first.entity_id,
        neighbor=second,
        canonical=canonical,
        direction="canonical_bridge",
        evidence=[evidence],
        neighbor_evidence=[evidence],
    )
    store = _FakeStore(
        (
            _snapshot(first, evidence=[evidence], canonical=[canonical]),
            _snapshot(second, evidence=[evidence], canonical=[canonical]),
        ),
        neighbors={first.entity_id: (bridge,)},
    )

    service = GraphRetrievalService(store, config=GraphRetrievalConfig(max_hops=1))
    first_result = service.search("共同实体", KB)
    second_result = service.search("共同实体", KB)

    assert len(first_result.items) == 1
    assert first_result.model_dump() == second_result.model_dump()
    assert any(path.kind == "canonical_bridge" for path in first_result.items[0].paths)


def test_repeated_canonical_bridge_does_not_create_a_cyclic_path() -> None:
    first = _entity("第一个")
    second = _entity("第二个", document_id=DOC_B)
    third = _entity("第三个", document_id=UUID("55555555-5555-4555-8555-555555555555"))
    first_evidence = _evidence(DOC_A, "chunk-first", "block-first")
    second_evidence = _evidence(DOC_B, "chunk-second", "block-second")
    third_evidence = _evidence(
        UUID("55555555-5555-4555-8555-555555555555"), "chunk-third", "block-third"
    )
    canonical = GraphCanonicalReference(
        canonical_entity_id="canonical_entity_v1_" + "d" * 32,
        knowledge_base_id=KB,
        canonical_name="共同 canonical",
        entity_type="概念",
    )
    first_to_second = GraphNeighbor(
        seed_entity_id=first.entity_id,
        neighbor=second,
        canonical=canonical,
        direction="canonical_bridge",
        evidence=[first_evidence, second_evidence],
        neighbor_evidence=[second_evidence],
    )
    second_to_third = GraphNeighbor(
        seed_entity_id=second.entity_id,
        neighbor=third,
        canonical=canonical,
        direction="canonical_bridge",
        evidence=[second_evidence, third_evidence],
        neighbor_evidence=[third_evidence],
    )
    store = _FakeStore(
        (
            _snapshot(first, evidence=[first_evidence], canonical=[canonical]),
            _snapshot(second, evidence=[second_evidence], canonical=[canonical]),
            _snapshot(third, evidence=[third_evidence], canonical=[canonical]),
        ),
        neighbors={
            first.entity_id: (first_to_second,),
            second.entity_id: (second_to_third,),
        },
    )

    result = GraphRetrievalService(store).search("共同 canonical", KB)

    assert all(
        path.hop_distance == 0
        or path.canonical_entity_ids.count(canonical.canonical_entity_id) == 1
        for item in result.items
        for path in item.paths
    )


def test_cross_document_relation_is_rejected_before_traversal() -> None:
    seed = _entity("文档 A")
    neighbor = _entity("文档 B", document_id=DOC_B)
    seed_evidence = _evidence(DOC_A, "chunk-doc-a", "block-doc-a")
    neighbor_evidence = _evidence(DOC_B, "chunk-doc-b", "block-doc-b")
    relation_evidence = _evidence(DOC_B, "chunk-doc-rel", "block-doc-rel")
    relation = GraphRelationReference(
        relation_id=relation_id_for(
            seed.entity_id,
            neighbor.entity_id,
            "跨文档",
            knowledge_base_id=KB,
            document_id=DOC_B,
        ),
        knowledge_base_id=KB,
        document_id=DOC_B,
        source_entity_id=seed.entity_id,
        target_entity_id=neighbor.entity_id,
        relation_type="跨文档",
    )
    edge = GraphNeighbor(
        seed_entity_id=seed.entity_id,
        neighbor=neighbor,
        relation=relation,
        direction="forward",
        evidence=[neighbor_evidence, relation_evidence],
        neighbor_evidence=[neighbor_evidence],
        relation_evidence=[relation_evidence],
    )
    store = _FakeStore(
        (
            _snapshot(seed, evidence=[seed_evidence]),
            _snapshot(neighbor, evidence=[neighbor_evidence]),
        ),
        neighbors={seed.entity_id: (edge,)},
    )

    with pytest.raises(GraphRetrievalError, match="cross-document"):
        GraphRetrievalService(store).search("文档 A", KB)


def test_invalid_or_unsupported_graph_edge_is_rejected() -> None:
    entity = _entity("实体")
    other = _entity("邻居")
    with pytest.raises(ValueError, match="relation direction"):
        GraphNeighbor(
            seed_entity_id=entity.entity_id,
            neighbor=other,
            relation=_relation(other, entity),
            direction="forward",
        )

    neighbor_evidence = _evidence(DOC_A, "chunk-neighbor", "block-neighbor")
    with pytest.raises(ValueError, match="relation must have current support evidence"):
        GraphNeighbor(
            seed_entity_id=entity.entity_id,
            neighbor=other,
            relation=_relation(entity, other),
            direction="forward",
            evidence=[neighbor_evidence],
            neighbor_evidence=[neighbor_evidence],
        )


def test_high_degree_neighbor_does_not_starve_later_neighbors() -> None:
    seed = _entity("公平起点")
    hub = _entity("高阶邻居")
    later_one = _entity("后续邻居一")
    later_two = _entity("后续邻居二")
    seed_evidence = _evidence(DOC_A, "chunk-fair-seed", "block-fair-seed")
    hub_evidence = _evidence(DOC_A, "chunk-fair-hub", "block-fair-hub")
    later_one_evidence = _evidence(DOC_A, "chunk-fair-one", "block-fair-one")
    later_two_evidence = _evidence(DOC_A, "chunk-fair-two", "block-fair-two")

    hub_edges = []
    for index in range(5):
        relation_evidence = _evidence(
            DOC_A,
            f"chunk-fair-relation-{index}",
            f"block-fair-relation-{index}",
        )
        hub_edges.append(
            GraphNeighbor(
                seed_entity_id=seed.entity_id,
                neighbor=hub,
                relation=_relation(seed, hub, relation_type=f"关系-{index}"),
                direction="forward",
                evidence=[seed_evidence, hub_evidence, relation_evidence],
                neighbor_evidence=[hub_evidence],
                relation_evidence=[relation_evidence],
            )
        )

    later_one_relation = _relation(seed, later_one, relation_type="关系-一")
    later_two_relation = _relation(seed, later_two, relation_type="关系-二")
    later_one_relation_evidence = _evidence(DOC_A, "chunk-fair-rel-one", "block-fair-rel-one")
    later_two_relation_evidence = _evidence(DOC_A, "chunk-fair-rel-two", "block-fair-rel-two")
    edges = [
        *hub_edges,
        GraphNeighbor(
            seed_entity_id=seed.entity_id,
            neighbor=later_one,
            relation=later_one_relation,
            direction="forward",
            evidence=[seed_evidence, later_one_evidence, later_one_relation_evidence],
            neighbor_evidence=[later_one_evidence],
            relation_evidence=[later_one_relation_evidence],
        ),
        GraphNeighbor(
            seed_entity_id=seed.entity_id,
            neighbor=later_two,
            relation=later_two_relation,
            direction="forward",
            evidence=[seed_evidence, later_two_evidence, later_two_relation_evidence],
            neighbor_evidence=[later_two_evidence],
            relation_evidence=[later_two_relation_evidence],
        ),
    ]
    store = _FakeStore(
        (_snapshot(seed, evidence=[seed_evidence]),),
        neighbors={seed.entity_id: tuple(edges)},
    )

    result = GraphRetrievalService(
        store,
        config=GraphRetrievalConfig(max_hops=1, max_neighbors=3, max_relations=3),
    ).search("公平起点", KB)

    result_entity_ids = {
        path.entity_ids[-1]
        for item in result.items
        for path in item.paths
        if path.hop_distance == 1
    }
    assert {hub.entity_id, later_one.entity_id, later_two.entity_id} <= result_entity_ids


def test_unlisted_neighbor_aggregate_evidence_is_not_retrieved() -> None:
    seed = _entity("聚合起点")
    neighbor = _entity("聚合邻居")
    seed_evidence = _evidence(DOC_A, "chunk-aggregate-seed", "block-aggregate-seed")
    neighbor_evidence = _evidence(DOC_A, "chunk-aggregate-neighbor", "block-aggregate-neighbor")
    relation_evidence = _evidence(DOC_A, "chunk-aggregate-relation", "block-aggregate-relation")
    unlisted_evidence = _evidence(DOC_A, "chunk-aggregate-extra", "block-aggregate-extra")
    relation = _relation(seed, neighbor, relation_type="聚合关系")
    edge = GraphNeighbor(
        seed_entity_id=seed.entity_id,
        neighbor=neighbor,
        relation=relation,
        direction="forward",
        evidence=[neighbor_evidence, relation_evidence, unlisted_evidence],
        neighbor_evidence=[neighbor_evidence],
        relation_evidence=[relation_evidence],
    )
    store = _FakeStore(
        (_snapshot(seed, evidence=[seed_evidence]),),
        neighbors={seed.entity_id: (edge,)},
    )

    result = GraphRetrievalService(store, config=GraphRetrievalConfig(max_hops=1)).search(
        "聚合起点", KB
    )

    assert "chunk-aggregate-extra" not in {item.evidence.chunk_id for item in result.items}


def test_path_requires_entity_count_to_match_hops() -> None:
    entity = _entity("实体")
    with pytest.raises(ValueError, match="entity IDs"):
        GraphPath(
            seed_entity_id=entity.entity_id,
            entity_ids=[entity.entity_id],
            directions=["forward"],
            hop_distance=1,
            kind="relation",
        )


def test_path_requires_edge_count_to_match_hops() -> None:
    entity = _entity("实体")
    other = _entity("邻居")
    with pytest.raises(ValueError, match="edge IDs"):
        GraphPath(
            seed_entity_id=entity.entity_id,
            entity_ids=[entity.entity_id, other.entity_id],
            directions=["forward"],
            hop_distance=1,
            kind="relation",
        )


def test_evidence_result_allows_alternate_seed_paths() -> None:
    evidence = _evidence(DOC_A, "chunk", "block")
    first = _entity("第一")
    second = _entity("第二", document_id=DOC_B)
    paths = [
        GraphPath(
            seed_entity_id=first.entity_id,
            entity_ids=[first.entity_id],
            kind="seed",
            hop_distance=0,
        ),
        GraphPath(
            seed_entity_id=second.entity_id,
            entity_ids=[second.entity_id],
            kind="seed",
            hop_distance=0,
        ),
    ]
    result = GraphEvidenceResult(
        evidence=evidence,
        score=0.5,
        seed_entity_id=first.entity_id,
        retrieval_reason="seed_entity_evidence",
        paths=paths,
    )
    assert len(result.paths) == 2
