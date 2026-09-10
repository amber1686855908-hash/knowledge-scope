"""Safe Knowledge Base attribution maintenance for Qdrant payloads."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import select

from knowledge_scope.documents.models import Document
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .qdrant import QdrantPointMetadata


class QdrantAttributionError(RuntimeError):
    """Raised when Qdrant KB attribution cannot be verified or repaired safely."""


@dataclass(frozen=True, slots=True)
class DocumentKnowledgeBaseMapping:
    """One authoritative PostgreSQL document-to-KB mapping row."""

    document_id: UUID
    knowledge_base_id: UUID | None


@dataclass(frozen=True, slots=True)
class QdrantAttributionAudit:
    """Preflight facts and the payload-only updates that are safe to apply."""

    point_count: int
    distinct_document_count: int
    mapped_document_ids: int
    unmapped_document_ids: int
    ambiguous_document_ids: int
    null_kb_mapping_document_ids: int
    points_with_null_kb: int
    planned_update_points: int
    already_correct_points: int
    conflicting_payload_points: int
    point_updates: tuple[tuple[UUID, UUID], ...]

    @property
    def can_apply(self) -> bool:
        """Return whether every point has one non-null, non-conflicting mapping."""
        return not (
            self.unmapped_document_ids
            or self.ambiguous_document_ids
            or self.null_kb_mapping_document_ids
            or self.conflicting_payload_points
        )

    @property
    def status(self) -> str:
        """Return a stable human- and automation-readable preflight status."""
        return "ready" if self.can_apply else "blocked"

    def as_dict(self) -> dict[str, int | str | bool]:
        """Serialize counts without exposing document-level runtime data."""
        return {
            "status": self.status,
            "can_apply": self.can_apply,
            "point_count": self.point_count,
            "distinct_document_count": self.distinct_document_count,
            "mapped_document_ids": self.mapped_document_ids,
            "unmapped_document_ids": self.unmapped_document_ids,
            "ambiguous_document_ids": self.ambiguous_document_ids,
            "null_kb_mapping_document_ids": self.null_kb_mapping_document_ids,
            "points_with_null_kb": self.points_with_null_kb,
            "planned_update_points": self.planned_update_points,
            "already_correct_points": self.already_correct_points,
            "conflicting_payload_points": self.conflicting_payload_points,
        }


class QdrantAttributionStore(Protocol):
    """Minimal Qdrant mutation surface used by the guarded repair function."""

    def set_point_knowledge_base_ids(
        self,
        point_ids: Sequence[UUID],
        knowledge_base_id: UUID,
    ) -> None:
        """Set only the Knowledge Base payload for the supplied point IDs."""


def build_attribution_audit(
    points: Sequence[QdrantPointMetadata],
    mappings: Sequence[DocumentKnowledgeBaseMapping],
) -> QdrantAttributionAudit:
    """Build a write plan without contacting either PostgreSQL or Qdrant."""
    mapping_values: dict[UUID, set[UUID | None]] = defaultdict(set)
    for mapping in mappings:
        mapping_values[mapping.document_id].add(mapping.knowledge_base_id)

    point_by_document: dict[UUID, list[QdrantPointMetadata]] = defaultdict(list)
    for point in points:
        point_by_document[point.document_id].append(point)

    mapped_documents = 0
    unmapped_documents = 0
    ambiguous_documents = 0
    null_mapping_documents = 0
    updates: list[tuple[UUID, UUID]] = []
    already_correct = 0
    conflicting_payload_points = 0

    for document_id, document_points in point_by_document.items():
        values = mapping_values.get(document_id, set())
        if not values:
            unmapped_documents += 1
            continue
        if len(values) > 1:
            ambiguous_documents += 1
            continue
        knowledge_base_id = next(iter(values))
        if knowledge_base_id is None:
            null_mapping_documents += 1
            continue
        mapped_documents += 1
        for point in document_points:
            if point.knowledge_base_id is None:
                updates.append((point.point_id, knowledge_base_id))
            elif point.knowledge_base_id == knowledge_base_id:
                already_correct += 1
            else:
                conflicting_payload_points += 1

    points_with_null_kb = sum(point.knowledge_base_id is None for point in points)
    return QdrantAttributionAudit(
        point_count=len(points),
        distinct_document_count=len(point_by_document),
        mapped_document_ids=mapped_documents,
        unmapped_document_ids=unmapped_documents,
        ambiguous_document_ids=ambiguous_documents,
        null_kb_mapping_document_ids=null_mapping_documents,
        points_with_null_kb=points_with_null_kb,
        planned_update_points=len(updates),
        already_correct_points=already_correct,
        conflicting_payload_points=conflicting_payload_points,
        point_updates=tuple(updates),
    )


def point_identity_signature(
    points: Sequence[QdrantPointMetadata],
) -> tuple[tuple[str, str, str], ...]:
    """Return stable point/document/chunk identities for post-write verification."""
    return tuple(
        sorted((str(point.point_id), str(point.document_id), point.chunk_id) for point in points)
    )


def apply_attribution_repair(
    store: QdrantAttributionStore,
    audit: QdrantAttributionAudit,
) -> int:
    """Apply only a complete, preflight-approved payload update plan."""
    if not audit.can_apply:
        raise QdrantAttributionError(
            "refusing Qdrant KB payload repair because authoritative mapping is incomplete"
        )
    updates_by_kb: dict[UUID, list[UUID]] = defaultdict(list)
    for point_id, knowledge_base_id in audit.point_updates:
        updates_by_kb[knowledge_base_id].append(point_id)
    for knowledge_base_id in sorted(updates_by_kb, key=str):
        store.set_point_knowledge_base_ids(updates_by_kb[knowledge_base_id], knowledge_base_id)
    return len(audit.point_updates)


async def load_authoritative_document_mappings(
    document_ids: Sequence[UUID],
    settings: Settings,
) -> tuple[DocumentKnowledgeBaseMapping, ...]:
    """Load document-to-KB mappings from PostgreSQL without inferring missing rows."""
    unique_ids = tuple(dict.fromkeys(document_ids))
    if not unique_ids:
        return ()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(Document.id, Document.knowledge_base_id).where(
                        Document.id.in_(unique_ids)
                    )
                )
            ).all()
            return tuple(
                DocumentKnowledgeBaseMapping(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                )
                for document_id, knowledge_base_id in rows
            )
    except QdrantAttributionError:
        raise
    except Exception as error:
        raise QdrantAttributionError(
            "authoritative PostgreSQL document mapping could not be read"
        ) from error
    finally:
        await engine.dispose()


__all__ = [
    "DocumentKnowledgeBaseMapping",
    "QdrantAttributionAudit",
    "QdrantAttributionError",
    "apply_attribution_repair",
    "build_attribution_audit",
    "load_authoritative_document_mappings",
    "point_identity_signature",
]
