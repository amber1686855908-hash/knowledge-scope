from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from knowledge_scope.retrieval.qdrant import QdrantPointMetadata
from knowledge_scope.retrieval.qdrant_attribution import (
    DocumentKnowledgeBaseMapping,
    QdrantAttributionError,
    apply_attribution_repair,
    build_attribution_audit,
    point_identity_signature,
)

KB_A = UUID("11111111-1111-4111-8111-111111111111")
KB_B = UUID("22222222-2222-4222-8222-222222222222")
DOC_A = UUID("33333333-3333-4333-8333-333333333333")
DOC_B = UUID("44444444-4444-4444-8444-444444444444")


def _point(
    document_id: UUID,
    chunk_id: str,
    *,
    knowledge_base_id: UUID | None = None,
) -> QdrantPointMetadata:
    return QdrantPointMetadata(
        point_id=uuid4(),
        document_id=document_id,
        chunk_id=chunk_id,
        knowledge_base_id=knowledge_base_id,
    )


@dataclass
class _FakeAttributionStore:
    updates: list[tuple[tuple[UUID, ...], UUID]] = field(default_factory=list)

    def set_point_knowledge_base_ids(
        self,
        point_ids: Sequence[UUID],
        knowledge_base_id: UUID,
    ) -> None:
        self.updates.append((tuple(point_ids), knowledge_base_id))


def test_complete_mapping_plans_only_missing_payloads() -> None:
    first = _point(DOC_A, "chunk-a")
    second = _point(DOC_A, "chunk-b", knowledge_base_id=KB_A)
    audit = build_attribution_audit(
        [first, second],
        [DocumentKnowledgeBaseMapping(DOC_A, KB_A)],
    )

    assert audit.can_apply is True
    assert audit.mapped_document_ids == 1
    assert audit.planned_update_points == 1
    assert audit.already_correct_points == 1
    assert audit.points_with_null_kb == 1
    assert audit.status == "ready"


def test_incomplete_mapping_is_blocked_without_a_repair_plan() -> None:
    point = _point(DOC_A, "chunk-a")
    audit = build_attribution_audit([point], [])
    store = _FakeAttributionStore()

    assert audit.can_apply is False
    assert audit.unmapped_document_ids == 1
    assert audit.point_updates == ()
    with pytest.raises(QdrantAttributionError, match="mapping is incomplete"):
        apply_attribution_repair(store, audit)
    assert store.updates == []


def test_ambiguous_and_null_mappings_are_blocked() -> None:
    points = [_point(DOC_A, "chunk-a"), _point(DOC_B, "chunk-b")]
    audit = build_attribution_audit(
        points,
        [
            DocumentKnowledgeBaseMapping(DOC_A, KB_A),
            DocumentKnowledgeBaseMapping(DOC_A, KB_B),
            DocumentKnowledgeBaseMapping(DOC_B, None),
        ],
    )

    assert audit.can_apply is False
    assert audit.ambiguous_document_ids == 1
    assert audit.null_kb_mapping_document_ids == 1
    assert audit.planned_update_points == 0


def test_existing_wrong_payload_is_a_conflict_not_a_silent_overwrite() -> None:
    audit = build_attribution_audit(
        [_point(DOC_A, "chunk-a", knowledge_base_id=KB_B)],
        [DocumentKnowledgeBaseMapping(DOC_A, KB_A)],
    )

    assert audit.can_apply is False
    assert audit.conflicting_payload_points == 1
    assert audit.point_updates == ()


def test_apply_groups_updates_by_authoritative_kb_and_is_repeatable() -> None:
    first = _point(DOC_A, "chunk-a")
    second = _point(DOC_B, "chunk-b")
    audit = build_attribution_audit(
        [first, second],
        [
            DocumentKnowledgeBaseMapping(DOC_A, KB_A),
            DocumentKnowledgeBaseMapping(DOC_B, KB_B),
        ],
    )
    store = _FakeAttributionStore()

    assert apply_attribution_repair(store, audit) == 2
    assert store.updates == [
        ((first.point_id,), KB_A),
        ((second.point_id,), KB_B),
    ]

    repaired_points = [
        _point(DOC_A, "chunk-a", knowledge_base_id=KB_A),
        _point(DOC_B, "chunk-b", knowledge_base_id=KB_B),
    ]
    repaired_points[0] = QdrantPointMetadata(
        point_id=first.point_id,
        document_id=DOC_A,
        chunk_id="chunk-a",
        knowledge_base_id=KB_A,
    )
    repaired_points[1] = QdrantPointMetadata(
        point_id=second.point_id,
        document_id=DOC_B,
        chunk_id="chunk-b",
        knowledge_base_id=KB_B,
    )
    rerun = build_attribution_audit(
        repaired_points,
        [
            DocumentKnowledgeBaseMapping(DOC_A, KB_A),
            DocumentKnowledgeBaseMapping(DOC_B, KB_B),
        ],
    )
    assert rerun.can_apply is True
    assert rerun.planned_update_points == 0
    assert rerun.already_correct_points == 2


def test_point_identity_signature_ignores_only_attribution_changes() -> None:
    point = _point(DOC_A, "chunk-a")
    attributed = QdrantPointMetadata(
        point_id=point.point_id,
        document_id=point.document_id,
        chunk_id=point.chunk_id,
        knowledge_base_id=KB_A,
    )

    assert point_identity_signature([point]) == point_identity_signature([attributed])
