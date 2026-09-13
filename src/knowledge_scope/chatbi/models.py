"""Persistent ChatBI datasource metadata model."""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

CHATBI_DATASOURCE_DISPLAY_NAME_MAX_LENGTH: Final = 200
CHATBI_DATASOURCE_DIALECT_MAX_LENGTH: Final = 32
CHATBI_DATASOURCE_CONNECTION_REF_MAX_LENGTH: Final = 255
CHATBI_DATASOURCE_IDENTIFIER_MAX_LENGTH: Final = 63


class ChatBIDataSourceRecord(Base):
    """A safe reference to a business database, not the business database itself."""

    __tablename__ = "chatbi_data_sources"
    __table_args__ = (
        CheckConstraint(
            "display_name = btrim(display_name) AND btrim(display_name) <> ''",
            name="ck_chatbi_data_sources_display_name_non_empty",
        ),
        CheckConstraint(
            "dialect = 'postgresql'",
            name="ck_chatbi_data_sources_supported_dialect",
        ),
        CheckConstraint(
            "connection_ref ~ '^(env|secret):[A-Za-z][A-Za-z0-9_.:/-]{0,247}$'",
            name="ck_chatbi_data_sources_connection_ref_format",
        ),
        Index("ix_chatbi_data_sources_enabled", "enabled"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    display_name: Mapped[str] = mapped_column(
        String(length=CHATBI_DATASOURCE_DISPLAY_NAME_MAX_LENGTH), nullable=False
    )
    dialect: Mapped[str] = mapped_column(
        String(length=CHATBI_DATASOURCE_DIALECT_MAX_LENGTH), nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    connection_ref: Mapped[str] = mapped_column(
        String(length=CHATBI_DATASOURCE_CONNECTION_REF_MAX_LENGTH), nullable=False
    )
    default_database: Mapped[str | None] = mapped_column(
        String(length=CHATBI_DATASOURCE_IDENTIFIER_MAX_LENGTH), nullable=True
    )
    default_schema: Mapped[str | None] = mapped_column(
        String(length=CHATBI_DATASOURCE_IDENTIFIER_MAX_LENGTH), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
