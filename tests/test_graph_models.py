from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.graph.models import (
    GRAPH_ID_VERSION,
    GRAPH_SCHEMA_VERSION,
    ExtractionProvenance,
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    canonical_identity_json,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
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
        page_end=3,
        source_block_ids=blocks or ["page-2-block-1"],
        section_path=["第一章", "概念"],
        extraction_provenance=extraction_provenance,
    )


def _entity(
    name: str = "水",
    entity_type: str = "物质",
    *,
    knowledge_base_id: UUID = KNOWLEDGE_BASE_ID,
    document_id: UUID = DOCUMENT_ID,
    provenance: list[GraphProvenance] | None = None,
    aliases: list[str] | None = None,
) -> GraphEntity:
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
        aliases=aliases if aliases is not None else ["水分子"],
        provenance=provenance or [_provenance(document_id, knowledge_base_id=knowledge_base_id)],
    )


def test_ids_are_deterministic_and_direction_sensitive() -> None:
    assert entity_id_for(
        " 水  ",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    ) == entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    assert entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    ) != entity_id_for(
        "水",
        "元素",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    assert entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    ) != entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_DOCUMENT_ID,
    )
    assert entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    ) != entity_id_for(
        "水",
        "物质",
        knowledge_base_id=OTHER_KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )

    source_id = entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    target_id = entity_id_for(
        "氢",
        "元素",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    forward = relation_id_for(
        source_id,
        target_id,
        "组成",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    reverse = relation_id_for(
        target_id,
        source_id,
        "组成",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    assert forward != reverse
    assert forward != relation_id_for(
        source_id,
        target_id,
        "组成",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_DOCUMENT_ID,
    )
    assert forward != relation_id_for(
        source_id,
        target_id,
        "组成",
        knowledge_base_id=OTHER_KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    assert forward.startswith(f"relation_{GRAPH_ID_VERSION}_")


def test_identity_serialization_is_structured_and_collision_safe() -> None:
    assert canonical_identity_json("entity", {"b": "二", "a": "一"}) == (
        '{"identity_schema_version":"v2","kind":"entity","payload":{"a":"一","b":"二"}}'
    )

    first = entity_id_for(
        "b\x00c",
        "a",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    second = entity_id_for(
        "c",
        "a\x00b",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    assert first != second
    assert entity_id_for(
        "e\u0301",
        "概念\u2028",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    ) == entity_id_for(
        "é",
        "概念\u2028",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )


def test_provenance_and_entity_round_trip_are_strict() -> None:
    extraction = ExtractionProvenance(method="mineru", model="model-a", version="1")
    entity = _entity(
        aliases=["别名 B", "别名 A"],
        provenance=[
            _provenance(extraction_provenance=extraction),
            _provenance(
                DOCUMENT_ID,
                chunk_id="chunk-2",
                blocks=["page-3-block-1"],
                extraction_provenance=ExtractionProvenance(method="manual"),
            ),
        ],
    )

    assert entity.schema_version == GRAPH_SCHEMA_VERSION
    assert entity.aliases == ["别名 A", "别名 B"]
    assert entity.provenance[0].extraction_provenance is not None
    assert GraphEntity.model_validate_json(entity.model_dump_json()) == entity
    assert evidence_id_for(entity.provenance[0]).startswith(f"evidence_{GRAPH_ID_VERSION}_")

    with pytest.raises(ValidationError):
        GraphEntity(
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            provenance=[],
        )
    with pytest.raises(ValidationError):
        GraphProvenance(
            document_id=DOCUMENT_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            chunk_id="chunk-1",
            page_start=3,
            page_end=2,
            source_block_ids=["block-1"],
        )
    with pytest.raises(ValidationError):
        GraphEntity.model_validate({**entity.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        GraphProvenance.model_validate({**_provenance().model_dump(), "storage_key": "secret"})
    with pytest.raises(ValidationError):
        GraphEntity.model_validate(
            {**entity.model_dump(), "extraction_provenance": {"method": "manual"}}
        )


def test_entity_provenance_cannot_cross_document_or_knowledge_base() -> None:
    with pytest.raises(ValidationError, match="document_id"):
        _entity(
            provenance=[
                _provenance(
                    document_id=OTHER_DOCUMENT_ID,
                    chunk_id="other-document-chunk",
                )
            ]
        )
    with pytest.raises(ValidationError, match="knowledge_base_id"):
        _entity(
            provenance=[
                _provenance(
                    knowledge_base_id=OTHER_KNOWLEDGE_BASE_ID,
                )
            ]
        )


def test_entity_id_must_match_canonical_identity() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        GraphEntity(
            entity_id=entity_id_for(
                "水",
                "物质",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                document_id=DOCUMENT_ID,
            ),
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
            canonical_name="氢",
            entity_type="元素",
            provenance=[_provenance()],
        )


def test_relation_requires_deterministic_identity_and_provenance() -> None:
    source_id = entity_id_for(
        "水",
        "物质",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    target_id = entity_id_for(
        "氢",
        "元素",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
    )
    relation = GraphRelation(
        relation_id=relation_id_for(
            source_id,
            target_id,
            "组成",
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relation_type="组成",
        provenance=[_provenance()],
    )

    assert GraphRelation.model_validate_json(relation.model_dump_json()) == relation

    with pytest.raises(ValidationError):
        GraphRelation(
            relation_id=relation.relation_id,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="组成",
            provenance=[_provenance(), _provenance()],
        )
    with pytest.raises(ValidationError, match="document_id"):
        GraphRelation(
            relation_id=relation.relation_id,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="组成",
            provenance=[_provenance(OTHER_DOCUMENT_ID, chunk_id="other-document-chunk")],
        )
