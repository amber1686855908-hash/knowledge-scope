"""Persistent model for a KnowledgeScope knowledge base."""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

KNOWLEDGE_BASE_NAME_MAX_LENGTH: Final = 200
KNOWLEDGE_BASE_DESCRIPTION_MAX_LENGTH: Final = 2_000


class KnowledgeBase(Base):
    """A named, persistent container for future document workflows."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        CheckConstraint(
            "name = btrim(name) AND btrim(name) <> ''",
            name="ck_knowledge_bases_name_trimmed_non_empty",
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(length=KNOWLEDGE_BASE_NAME_MAX_LENGTH), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
