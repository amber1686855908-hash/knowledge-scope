from __future__ import annotations

import asyncio
import os
from uuid import UUID, uuid4

import pytest

from knowledge_scope.graph.models import (
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    entity_id_for,
    relation_id_for,
)
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.graph.retrieval import GraphRetrievalConfig, graph_score
from knowledge_scope.graph.retrieval_service import GraphRetrievalService
from knowledge_scope.linking.service import link_entities
from knowledge_scope.linking.service_types import LocalEntityContext
from knowledge_scope.llm.schemas import LLMResult
from knowledge_scope.shared.config import get_settings


class _LinkGateway:
    async def complete(self, _request: object) -> LLMResult:
        return LLMResult(
            text='{"decision":"LINK","confidence":0.9,"reason":"同名且类型一致"}',
            provider="fake",
            model="fake-model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


def _entity(
    name: str,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    chunk_id: str,
) -> GraphEntity:
    provenance = GraphProvenance(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["A3.4 集成测试"],
    )
    return GraphEntity(
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
        provenance=[provenance],
    )


def _relation(
    source: GraphEntity,
    target: GraphEntity,
    *,
    relation_type: str = "组成",
) -> GraphRelation:
    return GraphRelation(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            relation_type,
            knowledge_base_id=source.knowledge_base_id,
            document_id=source.document_id,
        ),
        knowledge_base_id=source.knowledge_base_id,
        document_id=source.document_id,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type=relation_type,
        provenance=source.provenance,
    )


@pytest.mark.integration
def test_real_neo4j_graph_retrieval_preserves_scope_and_canonical_bridge() -> None:
    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    knowledge_base_id = uuid4()
    other_knowledge_base_id = uuid4()
    document_a = uuid4()
    document_b = uuid4()
    document_c = uuid4()
    foreign_document = uuid4()
    start = _entity(
        "起点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_a,
        chunk_id="a-start",
    )
    target = _entity(
        "直接邻居",
        knowledge_base_id=knowledge_base_id,
        document_id=document_a,
        chunk_id="a-target",
    )
    linked_a = _entity(
        "跨文档实体",
        knowledge_base_id=knowledge_base_id,
        document_id=document_b,
        chunk_id="b-linked",
    )
    linked_b = _entity(
        "跨文档实体",
        knowledge_base_id=knowledge_base_id,
        document_id=document_c,
        chunk_id="c-linked",
    )
    foreign = _entity(
        "起点",
        knowledge_base_id=other_knowledge_base_id,
        document_id=foreign_document,
        chunk_id="foreign-start",
    )
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        store.upsert_extraction(
            [start, target, linked_a, linked_b, foreign],
            [_relation(start, target)],
        )
        linking_run = asyncio.run(
            link_entities(
                [linked_a, linked_b],
                gateway=_LinkGateway(),
                settings=settings,
                contexts=[
                    LocalEntityContext(linked_a, "跨文档来源 A"),
                    LocalEntityContext(linked_b, "跨文档来源 B"),
                ],
            )
        )
        store.upsert_linking_result(
            linking_run.canonical_entities,
            linking_run.decisions,
            linking_run.mappings,
        )

        service = GraphRetrievalService(store)
        forward = service.search("起点", knowledge_base_id)
        reverse = service.search("直接邻居", knowledge_base_id)
        canonical = service.search("跨文档实体", knowledge_base_id)

        assert start.entity_id in {seed.seed_entity_id for seed in forward.seeds}
        assert foreign.entity_id not in {seed.seed_entity_id for seed in forward.seeds}
        assert {item.evidence.chunk_id for item in forward.items} >= {"a-start", "a-target"}
        assert any(
            any(
                path.hop_distance == 1
                and path.directions == ["reverse"]
                and path.entity_ids == [target.entity_id, start.entity_id]
                and path.relation_ids
                == [
                    _relation(start, target).relation_id,
                ]
                for path in item.paths
            )
            for item in reverse.items
        )
        assert any(
            any(path.kind == "canonical_bridge" for path in item.paths) for item in canonical.items
        )
        assert all(
            path.canonical_entity_ids.count(canonical_entity_id) <= 1
            for item in canonical.items
            for path in item.paths
            for canonical_entity_id in set(path.canonical_entity_ids)
        )
        assert all(
            path.hop_distance <= 1
            for item in canonical.items
            for path in item.paths
            if path.kind == "canonical_bridge"
        )
        assert all(
            item.evidence.knowledge_base_id == knowledge_base_id
            for item in (*forward.items, *reverse.items, *canonical.items)
        )
    finally:
        store._write(
            lambda tx: tx.run(
                "MATCH (node) WHERE node.knowledge_base_id IN $knowledge_base_ids "
                "DETACH DELETE node",
                knowledge_base_ids=[str(knowledge_base_id), str(other_knowledge_base_id)],
            ).consume()
        )
        store.close()


@pytest.mark.integration
def test_real_neo4j_graph_retrieval_hops_support_and_lifecycle() -> None:
    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    knowledge_base_id = uuid4()
    document_id = uuid4()
    first = _entity(
        "一跳起点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="hop-first",
    )
    second = _entity(
        "二跳中间",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="hop-second",
    )
    third = _entity(
        "三跳终点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="hop-third",
    )
    first_relation = _relation(first, second, relation_type="一跳关系")
    second_relation = _relation(second, third, relation_type="二跳关系")
    shared_first = _entity(
        "重复种子甲",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="shared-evidence",
    )
    shared_second = _entity(
        "重复种子乙",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="shared-evidence",
    )
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        store.upsert_extraction(
            [first, second, third, shared_first, shared_second],
            [first_relation, second_relation],
        )
        service = GraphRetrievalService(store)

        full = service.search("一跳起点", knowledge_base_id)
        second_result = next(item for item in full.items if item.evidence.chunk_id == "hop-second")
        second_path = next(path for path in second_result.paths if path.hop_distance == 1)
        assert second_path.directions == ["forward"]
        assert second_path.entity_ids == [first.entity_id, second.entity_id]
        assert second_path.relation_ids == [first_relation.relation_id]
        third_result = next(item for item in full.items if item.evidence.chunk_id == "hop-third")
        third_path = next(path for path in third_result.paths if path.hop_distance == 2)
        assert third_path.directions == ["forward", "forward"]
        assert third_path.entity_ids == [first.entity_id, second.entity_id, third.entity_id]
        assert third_path.relation_ids == [
            first_relation.relation_id,
            second_relation.relation_id,
        ]
        assert third_result.evidence.source_block_ids == ["hop-third-block"]

        one_hop = GraphRetrievalService(
            store,
            config=GraphRetrievalConfig(max_hops=1),
        ).search("一跳起点", knowledge_base_id)
        assert "hop-third" not in {item.evidence.chunk_id for item in one_hop.items}
        assert any(item.evidence.chunk_id == "hop-second" for item in one_hop.items)

        store._write(
            lambda tx: tx.run(
                "MATCH (relation:KnowledgeRelation {relation_id: $relation_id})"
                "-[support:SUPPORTED_BY]->() DELETE support",
                relation_id=second_relation.relation_id,
            ).consume()
        )
        without_second_relation_support = service.search("一跳起点", knowledge_base_id)
        assert "hop-second" in {
            item.evidence.chunk_id for item in without_second_relation_support.items
        }
        assert "hop-third" not in {
            item.evidence.chunk_id for item in without_second_relation_support.items
        }

        store.upsert_extraction([], [second_relation])
        store._write(
            lambda tx: tx.run(
                "MATCH (relation:KnowledgeRelation {relation_id: $relation_id})"
                "-[support:SUPPORTED_BY]->() DELETE support",
                relation_id=first_relation.relation_id,
            ).consume()
        )
        without_first_relation_support = service.search("一跳起点", knowledge_base_id)
        assert {item.evidence.chunk_id for item in without_first_relation_support.items} == {
            "hop-first"
        }
        store.upsert_extraction([], [first_relation])

        duplicate = GraphRetrievalService(store).search(
            "重复种子甲 重复种子乙",
            knowledge_base_id,
        )
        duplicate_again = GraphRetrievalService(store).search(
            "重复种子甲 重复种子乙",
            knowledge_base_id,
        )
        assert len(duplicate.items) == 1
        assert len(duplicate.items[0].paths) == 2
        assert duplicate.items[0].score == graph_score(0.93, 0, "seed")
        assert duplicate.model_dump() == duplicate_again.model_dump()

        deleted = store.delete_document(document_id, knowledge_base_id=knowledge_base_id)
        assert deleted.evidence_count > 0
        assert store.list_retrieval_entities(knowledge_base_id) == ()
        assert store.get_retrieval_neighbors(knowledge_base_id, first.entity_id) == ()
        deleted_again = store.delete_document(document_id, knowledge_base_id=knowledge_base_id)
        assert deleted_again.evidence_count == 0
        assert deleted_again.relation_count == 0
        assert deleted_again.entity_count == 0
    finally:
        store._write(
            lambda tx: tx.run(
                "MATCH (node) WHERE node.knowledge_base_id = $knowledge_base_id DETACH DELETE node",
                knowledge_base_id=str(knowledge_base_id),
            ).consume()
        )
        store.close()


@pytest.mark.integration
def test_real_neo4j_graph_retrieval_fair_bounds_and_supported_relations() -> None:
    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    knowledge_base_id = uuid4()
    document_id = uuid4()
    seed = _entity(
        "公平起点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="fair-seed",
    )
    hub = _entity(
        "高阶邻居",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="fair-hub",
    )
    later_one = _entity(
        "后续邻居一",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="fair-one",
    )
    later_two = _entity(
        "后续邻居二",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="fair-two",
    )
    orphan_start = _entity(
        "孤立关系起点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="orphan-start",
    )
    orphan_target = _entity(
        "孤立关系终点",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id="orphan-target",
    )
    relations = [_relation(seed, hub, relation_type=f"枢纽关系-{index}") for index in range(5)]
    relations.extend(
        [
            _relation(seed, later_one, relation_type="后续关系-一"),
            _relation(seed, later_two, relation_type="后续关系-二"),
        ]
    )
    orphan_relation = _relation(orphan_start, orphan_target, relation_type="无支撑关系")
    relations.append(orphan_relation)
    entities = [seed, hub, later_one, later_two, orphan_start, orphan_target]
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        store.upsert_extraction(entities, relations)

        bounded_neighbors = store.get_retrieval_neighbors(
            knowledge_base_id,
            seed.entity_id,
            max_neighbors=3,
            max_relations=2,
        )
        bounded_neighbor_ids = {item.neighbor.entity_id for item in bounded_neighbors}
        assert {hub.entity_id, later_one.entity_id, later_two.entity_id} <= bounded_neighbor_ids

        store._write(
            lambda tx: tx.run(
                "MATCH (entity:KnowledgeEntity {entity_id: $entity_id})"
                "-[support:SUPPORTED_BY]->() DELETE support",
                entity_id=hub.entity_id,
            ).consume()
        )
        assert hub.entity_id not in {
            snapshot.entity.entity_id
            for snapshot in store.list_retrieval_entities(knowledge_base_id)
        }
        assert store.get_retrieval_neighbors(knowledge_base_id, hub.entity_id) == ()

        store._write(
            lambda tx: tx.run(
                "MATCH (relation:KnowledgeRelation {relation_id: $relation_id})"
                "-[support:SUPPORTED_BY]->() DELETE support",
                relation_id=orphan_relation.relation_id,
            ).consume()
        )
        assert store.get_retrieval_neighbors(knowledge_base_id, orphan_start.entity_id) == ()
    finally:
        store._write(
            lambda tx: tx.run(
                "MATCH (node) WHERE node.knowledge_base_id = $knowledge_base_id DETACH DELETE node",
                knowledge_base_id=str(knowledge_base_id),
            ).consume()
        )
        store.close()
