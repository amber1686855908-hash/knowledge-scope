from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from uuid import UUID

import pytest

from knowledge_scope.graph.models import GraphEntity, GraphProvenance, entity_id_for
from knowledge_scope.graph.neo4j import (
    GraphLinkUpsertResult,
    GraphStoreError,
    Neo4jGraphStore,
)
from knowledge_scope.linking.models import (
    EntityCanonicalLink,
    canonical_link_id_for,
    new_canonical_entity_id,
    normalize_linking_label,
)
from knowledge_scope.linking.service import (
    LocalEntityContext,
    build_canonical_plan,
    link_entities,
)
from knowledge_scope.llm.schemas import LLMResult
from knowledge_scope.shared.config import Settings

KB = UUID("11111111-1111-4111-8111-111111111111")
DOC_A = UUID("33333333-3333-4333-8333-333333333333")
DOC_B = UUID("44444444-4444-4444-8444-444444444444")
DOC_C = UUID("55555555-5555-4555-8555-555555555555")


def _entity(
    name: str,
    document_id: UUID = DOC_A,
    *,
    entity_type: str = "概念",
) -> GraphEntity:
    return GraphEntity(
        entity_id=entity_id_for(
            name,
            entity_type,
            knowledge_base_id=KB,
            document_id=document_id,
        ),
        knowledge_base_id=KB,
        document_id=document_id,
        canonical_name=name,
        entity_type=entity_type,
        aliases=["共同别名一", "共同别名二"],
        provenance=[
            GraphProvenance(
                document_id=document_id,
                knowledge_base_id=KB,
                chunk_id=f"chunk-{document_id}",
                page_start=1,
                page_end=1,
                source_block_ids=[f"block-{document_id}"],
                section_path=["测试"],
            )
        ],
    )


@dataclass
class _Result:
    records: list[dict[str, object]]

    def single(self) -> dict[str, object] | None:
        return self.records[0] if self.records else None

    def consume(self) -> None:
        return None


class _LinkTransaction:
    def __init__(self, driver: _LinkDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: object) -> _Result:
        self.driver.queries.append((query, params))
        if "RETURN canonical.canonical_entity_id AS canonical_entity_id" in query:
            self.driver.canonical_ids.add(str(params["canonical_entity_id"]))
            return _Result([{"canonical_entity_id": params["canonical_entity_id"]}])
        if "RETURN decision.link_decision_id AS link_decision_id" in query:
            endpoints = {
                str(params["local_entity_a_id"]),
                str(params["local_entity_b_id"]),
            }
            if not endpoints.issubset(self.driver.local_ids):
                return _Result([])
            self.driver.decision_ids.add(str(params["link_decision_id"]))
            return _Result([{"link_decision_id": params["link_decision_id"]}])
        if "RETURN membership.link_id AS link_id" in query:
            if (
                str(params["local_entity_id"]) not in self.driver.local_ids
                or str(params["canonical_entity_id"]) not in self.driver.canonical_ids
                or str(params["primary_decision_id"]) not in self.driver.decision_ids
            ):
                return _Result([])
            return _Result([{"link_id": params["link_id"]}])
        if "RETURN collect(entity.entity_id) AS entity_ids" in query:
            document_id = str(params["document_id"])
            ids = [
                entity_id
                for entity_id, entity_document_id in self.driver.local_documents.items()
                if entity_document_id == document_id
            ]
            return _Result([{"entity_ids": ids}])
        if "RETURN count(" in query:
            return _Result([{"count": 0}])
        return _Result([])


class _LinkSession:
    def __init__(self, driver: _LinkDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _LinkSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute_write(self, work):
        self.driver.write_calls += 1
        return work(_LinkTransaction(self.driver))

    def run(self, query: str, **params: object) -> _Result:
        self.driver.queries.append((query, params))
        return _Result([])


class _LinkDriver:
    def __init__(self, entities: list[GraphEntity]) -> None:
        self.local_ids = {entity.entity_id for entity in entities}
        self.local_documents = {entity.entity_id: str(entity.document_id) for entity in entities}
        self.canonical_ids: set[str] = set()
        self.decision_ids: set[str] = set()
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.write_calls = 0

    def session(self, *, database: str) -> _LinkSession:
        assert database == "neo4j"
        return _LinkSession(self)

    def close(self) -> None:
        return None


class _LinkGateway:
    async def complete(self, _request):
        return LLMResult(
            text='{"decision":"LINK","confidence":0.9,"reason":"测试确认"}',
            provider="fake",
            model="fake-model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


def _linked_run(*entities: GraphEntity):
    return asyncio.run(
        link_entities(
            list(entities),
            gateway=_LinkGateway(),
            settings=Settings(_env_file=None, environment="test"),
            contexts=[
                LocalEntityContext(entity, f"{entity.canonical_name} 的来源") for entity in entities
            ],
        )
    )


def test_linking_upsert_is_one_transaction_and_idempotent() -> None:
    first = _entity("同一概念", DOC_A)
    second = _entity("同一概念", DOC_B)
    run = _linked_run(first, second)
    driver = _LinkDriver([first, second])
    store = Neo4jGraphStore(Settings(_env_file=None, environment="test"), driver=driver)

    expected = GraphLinkUpsertResult(
        canonical_entity_count=1,
        decision_count=1,
        mapping_count=2,
    )
    assert (
        store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            run.mappings,
        )
        == expected
    )
    assert (
        store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            run.mappings,
        )
        == expected
    )
    assert driver.write_calls == 2
    assert sum("MERGE (canonical:CanonicalEntity" in query for query, _ in driver.queries) == 2
    assert sum("MERGE (decision:KnowledgeLinkDecision" in query for query, _ in driver.queries) == 2
    assert (
        sum("MERGE (local)-[membership:CANONICAL_MEMBER_OF" in query for query, _ in driver.queries)
        == 4
    )


def test_canonical_metadata_mutation_keeps_persistent_identity() -> None:
    first = _entity("同一概念", DOC_A)
    second = _entity("同一概念", DOC_B)
    run = _linked_run(first, second)
    driver = _LinkDriver([first, second])
    store = Neo4jGraphStore(Settings(_env_file=None, environment="test"), driver=driver)

    canonical = run.canonical_entities[0]
    canonical.canonical_name = "被篡改的名称"
    result = store.upsert_linking_result([canonical], run.decisions, run.mappings)
    assert result.canonical_entity_count == 1
    assert canonical.canonical_entity_id == run.canonical_entities[0].canonical_entity_id
    assert driver.write_calls == 1


def test_linking_persistence_rejects_identity_and_audit_drift() -> None:
    first = _entity("同一概念", DOC_A)
    second = _entity("同一概念", DOC_B)
    run = _linked_run(first, second)
    driver = _LinkDriver([first, second])
    store = Neo4jGraphStore(Settings(_env_file=None, environment="test"), driver=driver)

    original_mapping = next(
        mapping for mapping in run.mappings if mapping.local_entity_id == first.entity_id
    )
    mismatched_canonical = run.canonical_entities[0].model_copy(
        update={
            "canonical_entity_id": new_canonical_entity_id(),
        }
    )
    mismatched_mapping = EntityCanonicalLink(
        link_id=canonical_link_id_for(
            KB,
            first.entity_id,
            mismatched_canonical.canonical_entity_id,
        ),
        knowledge_base_id=KB,
        local_entity_id=first.entity_id,
        canonical_entity_id=mismatched_canonical.canonical_entity_id,
        decision_id=original_mapping.decision_id,
        method=original_mapping.method,
        confidence=original_mapping.confidence,
        reason=original_mapping.reason,
        candidate_signals=original_mapping.candidate_signals,
        provider=original_mapping.provider,
        model=original_mapping.model,
        prompt_version=original_mapping.prompt_version,
    )
    mismatched_mappings = [
        mismatched_mapping,
        run.mappings[1].model_copy(
            update={
                "canonical_entity_id": mismatched_canonical.canonical_entity_id,
                "link_id": canonical_link_id_for(
                    KB,
                    run.mappings[1].local_entity_id,
                    mismatched_canonical.canonical_entity_id,
                ),
            }
        ),
    ]
    with pytest.raises(GraphStoreError, match="canonical_entity_id is immutable"):
        store.upsert_linking_result(
            [mismatched_canonical],
            run.decisions,
            mismatched_mappings,
        )
    assert driver.write_calls == 0

    tampered_mapping = run.mappings[0].model_copy(update={"reason": "审计内容被篡改"})
    with pytest.raises(GraphStoreError, match="audit fields"):
        store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            [tampered_mapping, run.mappings[1]],
        )
    assert driver.write_calls == 0


def test_linking_persistence_requires_existing_local_endpoints() -> None:
    first = _entity("同一概念", DOC_A)
    second = _entity("同一概念", DOC_B)
    run = _linked_run(first, second)
    driver = _LinkDriver([])
    store = Neo4jGraphStore(Settings(_env_file=None, environment="test"), driver=driver)

    with pytest.raises(GraphStoreError, match="local entities"):
        store.upsert_linking_result(run.canonical_entities, run.decisions, run.mappings)
    assert driver.write_calls == 1


@pytest.mark.integration
def test_real_neo4j_reconciliation_preserves_independent_support() -> None:
    """A new NO_LINK or source deletion must not remove an independent LINK support."""

    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")
    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    first = _entity("共享支持", DOC_A)
    second = _entity("共享支持", DOC_B)
    third = _entity("共享支持", DOC_C)
    run_ab = _linked_run(first, second)
    run_bc = _linked_run(second, third)
    canonical_id = run_ab.canonical_entities[0].canonical_entity_id
    canonical_bc, mappings_bc = build_canonical_plan(
        [second, third],
        run_bc.candidates,
        run_bc.decisions,
        canonical_ids_by_members={
            tuple(sorted((second.entity_id, third.entity_id))): canonical_id,
        },
    )
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        for document_id in (DOC_A, DOC_B, DOC_C):
            store.delete_document(document_id, knowledge_base_id=KB)
        for entity in (first, second, third):
            store.upsert_entity(entity)
        store.upsert_linking_result(
            run_ab.canonical_entities,
            run_ab.decisions,
            run_ab.mappings,
        )
        store.upsert_linking_result(canonical_bc, run_bc.decisions, mappings_bc)

        no_link = asyncio.run(link_entities([first, second], run_id=UUID(int=99)))
        assert no_link.stats.uncertain_count == 1
        store.upsert_linking_result([], no_link.decisions, [])
        assert store.get_canonical_entity(canonical_id) is not None
        support_ids = store._read(
            lambda session: session.run(
                """
                MATCH (local:KnowledgeEntity {entity_id: $entity_id})
                    -[membership:CANONICAL_MEMBER_OF]->()
                RETURN membership.decision_ids AS decision_ids
                """,
                entity_id=second.entity_id,
            ).single()["decision_ids"]
        )
        assert run_ab.decisions[0].decision_id not in support_ids
        assert run_bc.decisions[0].decision_id in support_ids

        store.delete_document(DOC_A, knowledge_base_id=KB)
        assert store.get_canonical_entity(canonical_id) is not None
        store.delete_document(DOC_B, knowledge_base_id=KB)
        assert store.get_canonical_entity(canonical_id) is None
    finally:
        for document_id in (DOC_A, DOC_B, DOC_C):
            store.delete_document(document_id, knowledge_base_id=KB)
        store.close()


@pytest.mark.integration
def test_real_neo4j_delete_handles_shared_membership_within_document() -> None:
    """Deleting a document tolerates several decisions sharing one membership."""

    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")
    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    first = _entity("同文档实体甲", DOC_A)
    second = _entity("同文档实体乙", DOC_A)
    third = _entity("同文档实体丙", DOC_A)
    run = _linked_run(first, second, third)
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        store.delete_document(DOC_A, knowledge_base_id=KB)
        for entity in (first, second, third):
            store.upsert_entity(entity)
        assert run.stats.link_count == 3
        store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            run.mappings,
        )

        deleted = store.delete_document_links(KB, DOC_A)
        assert deleted.decision_count == 3
        assert deleted.mapping_count == 3
        assert deleted.canonical_entity_count == 1
        assert store.get_canonical_entity(run.canonical_entities[0].canonical_entity_id) is None
    finally:
        store.delete_document(DOC_A, knowledge_base_id=KB)
        store.close()


@pytest.mark.integration
def test_real_neo4j_backfills_legacy_entity_type_for_a3_3_linking() -> None:
    """Backfill A3.1 entities before strict A3.3 type matching is used."""

    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")
    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    first = _entity("旧版链接实体", DOC_A)
    second = _entity("旧版链接实体", DOC_B)
    run = _linked_run(first, second)
    store = Neo4jGraphStore(settings)

    def normalized_type(entity_id: str) -> str | None:
        record = store._read(
            lambda session: session.run(
                """
                MATCH (entity:KnowledgeEntity {entity_id: $entity_id})
                RETURN entity.entity_type_normalized AS normalized
                """,
                entity_id=entity_id,
            ).single()
        )
        return record["normalized"] if record is not None else None

    try:
        store.ensure_schema()
        store.delete_document(DOC_A, knowledge_base_id=KB)
        store.delete_document(DOC_B, knowledge_base_id=KB)
        store.upsert_entity(first)
        store.upsert_entity(second)
        store._write(
            lambda tx: tx.run(
                """
                MATCH (entity:KnowledgeEntity)
                WHERE entity.entity_id IN $entity_ids
                REMOVE entity.entity_type_normalized
                """,
                entity_ids=[first.entity_id, second.entity_id],
            ).consume()
        )
        assert normalized_type(first.entity_id) is None
        assert normalized_type(second.entity_id) is None

        store.ensure_schema()
        expected_type = normalize_linking_label(first.entity_type)
        assert normalized_type(first.entity_id) == expected_type
        assert normalized_type(second.entity_id) == expected_type
        assert store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            run.mappings,
        ) == GraphLinkUpsertResult(1, 1, 2)

        store._write(
            lambda tx: tx.run(
                """
                MATCH (entity:KnowledgeEntity {entity_id: $entity_id})
                SET entity.entity_type_normalized = $normalized
                """,
                entity_id=first.entity_id,
                normalized="preexisting-value",
            ).consume()
        )
        store.ensure_schema()
        assert normalized_type(first.entity_id) == "preexisting-value"
        assert normalized_type(second.entity_id) == expected_type
    finally:
        store.delete_document(DOC_A, knowledge_base_id=KB)
        store.delete_document(DOC_B, knowledge_base_id=KB)
        store.close()


@pytest.mark.integration
def test_real_neo4j_linking_lifecycle() -> None:
    """Run only with an explicitly enabled local Neo4j service."""

    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")
    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    first = _entity("集成概念", DOC_A)
    second = _entity("集成概念", DOC_B)
    incompatible = _entity("集成概念", DOC_A, entity_type="人物")
    run = _linked_run(first, second, incompatible)
    store = Neo4jGraphStore(settings)
    try:
        store.ensure_schema()
        store.delete_document_links(KB, DOC_A)
        store.delete_document_links(KB, DOC_B)
        store.delete_document(DOC_A, knowledge_base_id=KB)
        store.delete_document(DOC_B, knowledge_base_id=KB)
        store.upsert_entity(first)
        store.upsert_entity(second)
        store.upsert_entity(incompatible)
        assert store.upsert_linking_result(
            run.canonical_entities,
            run.decisions,
            run.mappings,
        ) == GraphLinkUpsertResult(1, 3, 2)
        updated_canonical = run.canonical_entities[0].model_copy(
            update={"aliases": ["追加来源别名"]}
        )
        assert store.upsert_linking_result(
            [updated_canonical],
            run.decisions,
            run.mappings,
        ) == GraphLinkUpsertResult(1, 3, 2)
        canonical_id = run.canonical_entities[0].canonical_entity_id
        stored_canonical = store.get_canonical_entity(canonical_id)
        assert stored_canonical is not None
        assert "追加来源别名" in stored_canonical.aliases
        assert (
            store._read(
                lambda session: int(
                    session.run(
                        "MATCH (decision:KnowledgeLinkDecision "
                        "{knowledge_base_id: $knowledge_base_id}) "
                        "RETURN count(decision) AS count",
                        knowledge_base_id=str(KB),
                    ).single()["count"]
                )
            )
            == 3
        )
        deleted_a = store.delete_document_links(KB, DOC_A)
        assert deleted_a.decision_count == 3
        assert deleted_a.canonical_entity_count == 1
        assert store.get_canonical_entity(canonical_id) is None
        assert (
            store._read(
                lambda session: int(
                    session.run(
                        "MATCH (decision:KnowledgeLinkDecision "
                        "{knowledge_base_id: $knowledge_base_id}) "
                        "RETURN count(decision) AS count",
                        knowledge_base_id=str(KB),
                    ).single()["count"]
                )
            )
            == 0
        )
        deleted_b = store.delete_document_links(KB, DOC_B)
        assert deleted_b.decision_count == 0
        assert deleted_b.canonical_entity_count == 0
        assert (
            store._read(
                lambda session: int(
                    session.run(
                        "MATCH (decision:KnowledgeLinkDecision "
                        "{knowledge_base_id: $knowledge_base_id}) "
                        "RETURN count(decision) AS count",
                        knowledge_base_id=str(KB),
                    ).single()["count"]
                )
            )
            == 0
        )
        assert store.get_canonical_entity(canonical_id) is None
    finally:
        store.delete_document_links(KB, DOC_A)
        store.delete_document_links(KB, DOC_B)
        store.delete_document(DOC_A, knowledge_base_id=KB)
        store.delete_document(DOC_B, knowledge_base_id=KB)
        store.close()
