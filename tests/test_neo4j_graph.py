from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import UUID

import pytest

from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    entity_id_for,
    relation_id_for,
)
from knowledge_scope.graph.neo4j import (
    SCHEMA_STATEMENTS,
    GraphStoreError,
    GraphUpsertResult,
    Neo4jGraphStore,
)
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
BATCH_DOCUMENT_ID = UUID("55555555-5555-4555-8555-555555555555")
KNOWLEDGE_BASE_ID = UUID("33333333-3333-4333-8333-333333333333")
OTHER_KNOWLEDGE_BASE_ID = UUID("44444444-4444-4444-8444-444444444444")


def _provenance(
    document_id: UUID = DOCUMENT_ID,
    *,
    knowledge_base_id: UUID = KNOWLEDGE_BASE_ID,
    chunk_id: str = "chunk-1",
    blocks: list[str] | None = None,
    extraction_provenance: ExtractionProvenance | None = None,
) -> GraphProvenance:
    return GraphProvenance(
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        page_start=2,
        page_end=2,
        source_block_ids=blocks or ["block-1"],
        section_path=["章节"],
        extraction_provenance=extraction_provenance,
    )


def _entity(
    name: str,
    *,
    knowledge_base_id: UUID = KNOWLEDGE_BASE_ID,
    document_id: UUID = DOCUMENT_ID,
    aliases: list[str] | None = None,
    provenance: list[GraphProvenance] | None = None,
) -> GraphEntity:
    entity_type = "概念"
    return GraphEntity(
        entity_id=entity_id_for(
            name,
            entity_type,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        ),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        canonical_name=name,
        entity_type=entity_type,
        aliases=aliases or [],
        provenance=provenance or [_provenance(document_id, knowledge_base_id=knowledge_base_id)],
    )


@dataclass
class _Result:
    records: list[dict[str, object]]

    def __iter__(self):
        return iter(self.records)

    def single(self) -> dict[str, object] | None:
        return self.records[0] if self.records else None

    def consume(self) -> None:
        return None


class _SpyTransaction:
    def __init__(self, driver: _SpyDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: object) -> _Result:
        self.driver.queries.append((query, params))
        if "RETURN entity.entity_id AS entity_id" in query:
            self.driver.entity_ids.add(str(params["entity_id"]))
            return _Result([{"entity_id": params["entity_id"]}])
        if "RETURN relation.relation_id AS relation_id" in query:
            if not {
                str(params["source_entity_id"]),
                str(params["target_entity_id"]),
            }.issubset(self.driver.entity_ids):
                return _Result([])
            return _Result([{"relation_id": params["relation_id"]}])
        return _Result([])


class _SpySession:
    def __init__(self, driver: _SpyDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _SpySession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute_write(self, work):
        self.driver.write_calls += 1
        return work(_SpyTransaction(self.driver))

    def run(self, query: str, **params: object) -> _Result:
        self.driver.queries.append((query, params))
        if query == "RETURN 1 AS ok":
            return _Result([{"ok": 1}])
        if "entity.entity_type_normalized IS NULL" in query:
            return _Result(self.driver.legacy_entity_records)
        return _Result([])


class _SpyDriver:
    def __init__(self, legacy_entity_records: list[dict[str, object]] | None = None) -> None:
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.entity_ids: set[str] = set()
        self.legacy_entity_records = legacy_entity_records or []
        self.write_calls = 0
        self.closed = False

    def session(self, *, database: str) -> _SpySession:
        assert database == "neo4j"
        return _SpySession(self)

    def close(self) -> None:
        self.closed = True


class _FailingDriver:
    def session(self, *, database: str) -> _SpySession:
        del database
        raise RuntimeError("connection failed with password=should-not-leak")


def _store(driver: _SpyDriver) -> Neo4jGraphStore:
    return Neo4jGraphStore(
        Settings(_env_file=None, environment="test"),
        driver=driver,
    )


def test_readiness_and_schema_are_explicit_and_non_sensitive() -> None:
    driver = _SpyDriver()
    store = _store(driver)

    assert store.readiness().status == "ready"
    readiness = store.ensure_schema()
    assert readiness.status == "ready"
    assert len([query for query, _ in driver.queries if query in SCHEMA_STATEMENTS]) == len(
        SCHEMA_STATEMENTS
    )


@pytest.mark.parametrize("legacy_entity_type", [None, "", "   ", 123])
def test_schema_rejects_invalid_legacy_entity_types_before_backfill(
    legacy_entity_type: object,
) -> None:
    driver = _SpyDriver(
        legacy_entity_records=[
            {
                "entity_id": "entity_v2_" + "0" * 64,
                "entity_type": legacy_entity_type,
            }
        ]
    )

    with pytest.raises(GraphStoreError, match="entity_type"):
        _store(driver).ensure_schema()

    assert not any("SET entity.entity_type_normalized" in query for query, _ in driver.queries)


def test_entity_and_relation_upserts_use_scoped_merges_and_evidence() -> None:
    driver = _SpyDriver()
    store = _store(driver)
    source = _entity("水")
    target = _entity("氢")
    relation = GraphRelation(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type="组成",
        provenance=[_provenance()],
    )

    assert store.upsert_entity(source) == source
    assert store.upsert_entity(source) == source
    with pytest.raises(GraphStoreError, match="endpoints"):
        store.upsert_relation(relation)
    store.upsert_entity(target)
    assert store.upsert_relation(relation) == relation

    query_text = "\n".join(query for query, _ in driver.queries)
    assert query_text.count("MERGE (entity:KnowledgeEntity") == 3
    assert "MERGE (evidence:KnowledgeEvidence" in query_text
    assert "MERGE (source)-[:SOURCE_OF]->(relation)" in query_text
    assert "collect(alias) AS merged_aliases" in query_text
    assert "entity.aliases = $aliases" not in query_text
    assert all("password" not in params for _, params in driver.queries)


def test_upserts_revalidate_mutable_models_at_store_boundary() -> None:
    driver = _SpyDriver()
    store = _store(driver)

    entity = _entity("水")
    entity.provenance[0] = _provenance(OTHER_DOCUMENT_ID, chunk_id="foreign-document")
    with pytest.raises(GraphStoreError, match="canonical validation"):
        store.upsert_entity(entity)

    relation_source = _entity("源")
    relation_target = _entity("目标")
    relation = GraphRelation(
        relation_id=relation_id_for(
            relation_source.entity_id,
            relation_target.entity_id,
            "关联",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=relation_source.entity_id,
        target_entity_id=relation_target.entity_id,
        relation_type="关联",
        provenance=[_provenance()],
    )
    relation.provenance.clear()
    with pytest.raises(GraphStoreError, match="canonical validation"):
        store.upsert_relation(relation)
    relation.provenance.append(_provenance(OTHER_DOCUMENT_ID, chunk_id="foreign-relation"))
    with pytest.raises(GraphStoreError, match="canonical validation"):
        store.upsert_relation(relation)

    assert driver.queries == []


def test_extraction_batch_uses_one_transaction_for_entities_and_relations() -> None:
    driver = _SpyDriver()
    store = _store(driver)
    source = _entity("水")
    target = _entity("氢")
    relation = GraphRelation(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type="组成",
        provenance=[_provenance()],
    )

    result = store.upsert_extraction([source, target], [relation])

    assert result == GraphUpsertResult(entity_count=2, relation_count=1)
    assert driver.write_calls == 1
    assert sum("MERGE (entity:KnowledgeEntity" in query for query, _ in driver.queries) == 2
    assert sum("MERGE (relation:KnowledgeRelation" in query for query, _ in driver.queries) == 1

    driver.write_calls = 0
    driver.queries.clear()
    relation.provenance.clear()
    with pytest.raises(GraphStoreError, match="canonical validation"):
        store.upsert_extraction([source, target], [relation])
    assert driver.write_calls == 0
    assert driver.queries == []


def test_missing_password_is_a_controlled_non_sensitive_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KNOWLEDGE_SCOPE_NEO4J_PASSWORD", raising=False)
    store = Neo4jGraphStore(Settings(_env_file=None, environment="test"))

    readiness = store.readiness()
    assert readiness.status == "unavailable"
    assert readiness.error == "Neo4j is not reachable"


@pytest.mark.integration
def test_real_neo4j_extraction_batch_is_idempotent_and_atomic() -> None:
    """Verify the A3.2 batch boundary when an opt-in local Neo4j is available."""

    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")

    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    store = Neo4jGraphStore(settings)
    source = _entity("批量源", document_id=BATCH_DOCUMENT_ID)
    target = _entity("批量目标", document_id=BATCH_DOCUMENT_ID)
    provenance = _provenance(BATCH_DOCUMENT_ID, chunk_id="batch-chunk")
    relation = GraphRelation(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=BATCH_DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=BATCH_DOCUMENT_ID,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type="组成",
        provenance=[provenance],
    )
    failing_source = _entity("失败源", document_id=BATCH_DOCUMENT_ID)
    missing_target = _entity("缺失目标", document_id=BATCH_DOCUMENT_ID)
    failing_relation = GraphRelation(
        relation_id=relation_id_for(
            failing_source.entity_id,
            missing_target.entity_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=BATCH_DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=BATCH_DOCUMENT_ID,
        source_entity_id=failing_source.entity_id,
        target_entity_id=missing_target.entity_id,
        relation_type="组成",
        provenance=[provenance],
    )
    try:
        store.ensure_schema()
        store.delete_document(BATCH_DOCUMENT_ID)
        assert store.upsert_extraction([source, target], [relation]) == GraphUpsertResult(
            entity_count=2,
            relation_count=1,
        )
        store.upsert_extraction([source, target], [relation])
        assert store.get_entity(source.entity_id) is not None
        assert store.get_relation(relation.relation_id) is not None

        with pytest.raises(GraphStoreError, match="endpoints"):
            store.upsert_extraction([failing_source], [failing_relation])
        assert store.get_entity(failing_source.entity_id) is None
    finally:
        store.delete_document(BATCH_DOCUMENT_ID)
        store.close()


def test_driver_failures_are_normalized_without_provider_details() -> None:
    store = _store(_FailingDriver())  # type: ignore[arg-type]

    with pytest.raises(GraphStoreError, match="read operation failed") as error:
        store.get_entity(
            entity_id_for(
                "水",
                "概念",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                document_id=DOCUMENT_ID,
            )
        )
    assert "should-not-leak" not in str(error.value)


@pytest.mark.integration
def test_real_neo4j_graph_lifecycle() -> None:
    """Run only when a local Neo4j service and explicit opt-in are available."""
    pytest.importorskip("neo4j")
    if os.environ.get("KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION") != "1":
        pytest.skip("set KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1 to run Neo4j integration")

    from knowledge_scope.shared.config import get_settings

    settings = get_settings()
    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        pytest.skip("configure KNOWLEDGE_SCOPE_NEO4J_PASSWORD for Neo4j integration")

    store = Neo4jGraphStore(settings)
    extraction_a = ExtractionProvenance(method="mineru", model="model-a", version="1")
    extraction_a_second = ExtractionProvenance(method="mineru", model="model-a", version="2")
    provenance_a = _provenance(
        extraction_provenance=extraction_a,
        chunk_id="chunk-a",
        blocks=["block-a"],
    )
    provenance_a_second = _provenance(
        extraction_provenance=extraction_a_second,
        chunk_id="chunk-a-2",
        blocks=["block-a-2"],
    )
    extraction_b = ExtractionProvenance(method="manual", version="2")
    provenance_b = _provenance(OTHER_DOCUMENT_ID, extraction_provenance=extraction_b)
    provenance_other_kb = _provenance(
        knowledge_base_id=OTHER_KNOWLEDGE_BASE_ID,
        extraction_provenance=extraction_a,
    )
    source_a = _entity("水", aliases=["H2O"], provenance=[provenance_a])
    target_a = _entity("氢", aliases=["氢元素"], provenance=[provenance_a])
    other_kb_source = _entity(
        "水",
        knowledge_base_id=OTHER_KNOWLEDGE_BASE_ID,
        aliases=["另一个知识库"],
        provenance=[provenance_other_kb],
    )
    relation_a = GraphRelation(
        relation_id=relation_id_for(
            source_a.entity_id,
            target_a.entity_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source_a.entity_id,
        target_entity_id=target_a.entity_id,
        relation_type="组成",
        provenance=[provenance_a],
    )
    source_b = _entity("水", document_id=OTHER_DOCUMENT_ID, provenance=[provenance_b])
    target_b = _entity("氢", document_id=OTHER_DOCUMENT_ID, provenance=[provenance_b])
    source_a_second_input = GraphEntity(
        entity_id=source_a.entity_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        canonical_name="水",
        entity_type="概念",
        aliases=["第二次写入别名"],
        provenance=[provenance_a_second],
    )
    target_a_second_input = GraphEntity(
        entity_id=target_a.entity_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        canonical_name="氢",
        entity_type="概念",
        aliases=["第二次写入别名"],
        provenance=[provenance_a_second],
    )
    source_a_expected = GraphEntity(
        entity_id=source_a.entity_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        canonical_name="水",
        entity_type="概念",
        aliases=["H2O", "第二次写入别名"],
        provenance=[provenance_a, provenance_a_second],
    )
    target_a_expected = GraphEntity(
        entity_id=target_a.entity_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        canonical_name="氢",
        entity_type="概念",
        aliases=["氢元素", "第二次写入别名"],
        provenance=[provenance_a, provenance_a_second],
    )
    relation_a_second_input = GraphRelation(
        relation_id=relation_a.relation_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source_a.entity_id,
        target_entity_id=target_a.entity_id,
        relation_type="组成",
        provenance=[provenance_a_second],
    )
    relation_a_expected = GraphRelation(
        relation_id=relation_a.relation_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source_a.entity_id,
        target_entity_id=target_a.entity_id,
        relation_type="组成",
        provenance=[provenance_a, provenance_a_second],
    )
    try:
        store.ensure_schema()
        store.ensure_schema()
        store.delete_document(DOCUMENT_ID)
        store.delete_document(OTHER_DOCUMENT_ID)
        with pytest.raises(GraphStoreError, match="endpoints"):
            store.upsert_relation(relation_a)

        store.upsert_entity(source_a)
        store.upsert_entity(target_a)
        store.upsert_relation(relation_a)
        store.upsert_entity(source_a_second_input)
        store.upsert_entity(target_a_second_input)
        store.upsert_relation(relation_a_second_input)
        store.upsert_entity(source_b)
        store.upsert_entity(target_b)
        store.upsert_entity(other_kb_source)

        assert source_a.entity_id != source_b.entity_id
        assert source_a.entity_id != other_kb_source.entity_id
        assert store.get_entity(source_a.entity_id) == source_a_expected
        assert store.get_entity(target_a.entity_id) == target_a_expected
        assert store.get_entity(source_b.entity_id) == source_b
        assert store.get_entity(other_kb_source.entity_id) == other_kb_source
        assert store.get_relation(relation_a.relation_id) == relation_a_expected

        def upsert_concurrent_alias(index: int) -> GraphEntity:
            return store.upsert_entity(
                _entity(
                    "水",
                    aliases=[f"并发别名-{index}"],
                    provenance=[provenance_a_second],
                )
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(upsert_concurrent_alias, range(100)))
        stored_source = store.get_entity(source_a.entity_id)
        assert stored_source is not None
        assert stored_source.provenance == source_a_expected.provenance
        assert set(source_a_expected.aliases).issubset(stored_source.aliases)
        assert {f"并发别名-{index}" for index in range(100)}.issubset(stored_source.aliases)

        deleted_first = store.delete_document(DOCUMENT_ID)
        assert deleted_first.evidence_count == 3
        assert deleted_first.relation_count == 1
        assert deleted_first.entity_count == 3
        assert store.get_entity(source_a.entity_id) is None
        assert store.get_entity(target_a.entity_id) is None
        assert store.get_entity(source_b.entity_id) == source_b
        assert store.get_entity(other_kb_source.entity_id) is None
        assert store.get_relation(relation_a.relation_id) is None

        repeated_first = store.delete_document(DOCUMENT_ID)
        assert repeated_first.evidence_count == 0
        assert repeated_first.relation_count == 0
        assert repeated_first.entity_count == 0

        deleted = store.delete_document(OTHER_DOCUMENT_ID)
        assert deleted.evidence_count == 1
        assert deleted.relation_count == 0
        assert deleted.entity_count == 2
        assert store.get_entity(source_b.entity_id) is None

        repeated = store.delete_document(OTHER_DOCUMENT_ID)
        assert repeated.evidence_count == 0
        assert repeated.relation_count == 0
        assert repeated.entity_count == 0
    finally:
        store.delete_document(DOCUMENT_ID)
        store.delete_document(OTHER_DOCUMENT_ID)
        store.close()
