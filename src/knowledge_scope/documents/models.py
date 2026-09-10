"""Persistent document model for uploaded source files."""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

DOCUMENT_FILENAME_MAX_LENGTH: Final = 255
DOCUMENT_MEDIA_TYPE_PDF: Final = "application/pdf"
DOCUMENT_STATUS_UPLOADED: Final = "uploaded"
DOCUMENT_STATUS_REGISTERED: Final = "registered"
DOCUMENT_STORAGE_KIND_MANAGED: Final = "managed"
DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE: Final = "external_reference"
DOCUMENT_SOURCE_REF_MAX_LENGTH: Final = 2_048


class Document(Base):
    """A PDF registered in a knowledge base, either managed or external."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "sha256",
            name="uq_documents_knowledge_base_sha256",
        ),
        CheckConstraint("size_bytes > 0", name="ck_documents_size_bytes_positive"),
        CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_documents_sha256_lowercase_hex_length",
        ),
        CheckConstraint(
            "media_type = 'application/pdf'",
            name="ck_documents_media_type_pdf",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'registered')",
            name="ck_documents_status_known",
        ),
        CheckConstraint(
            """
            storage_kind IN ('managed', 'external_reference')
            AND (
                (
                    storage_kind = 'managed'
                    AND storage_key IS NOT NULL
                    AND storage_key <> ''
                    AND storage_key NOT LIKE '/%'
                    AND storage_key NOT LIKE '%..%'
                    AND source_ref IS NULL
                )
                OR (
                    storage_kind = 'external_reference'
                    AND storage_key IS NULL
                    AND source_ref IS NOT NULL
                    AND btrim(source_ref) = source_ref
                    AND source_ref <> ''
                    AND source_ref NOT LIKE '/%'
                    AND source_ref NOT LIKE '%..%'
                    AND source_ref NOT LIKE '%:%'
                )
            )
            """,
            name="ck_documents_storage_reference_consistent",
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(
        String(length=DOCUMENT_FILENAME_MAX_LENGTH), nullable=False
    )
    storage_key: Mapped[str | None] = mapped_column(String(length=512), nullable=True)
    storage_kind: Mapped[str] = mapped_column(
        String(length=32),
        nullable=False,
        default=DOCUMENT_STORAGE_KIND_MANAGED,
        server_default=DOCUMENT_STORAGE_KIND_MANAGED,
    )
    source_ref: Mapped[str | None] = mapped_column(
        String(length=DOCUMENT_SOURCE_REF_MAX_LENGTH), nullable=True
    )
    media_type: Mapped[str] = mapped_column(String(length=100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(length=64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(length=32),
        nullable=False,
        default=DOCUMENT_STATUS_UPLOADED,
        server_default=DOCUMENT_STATUS_UPLOADED,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
