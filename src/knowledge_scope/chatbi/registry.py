"""Application-owned lookup for registered ChatBI datasources."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import ChatBIDataSourceRecord
from .schemas import DataSource


def data_source_from_record(record: ChatBIDataSourceRecord) -> DataSource:
    """Project one database record into the internal datasource contract."""
    return DataSource.model_validate(
        {
            "id": record.id,
            "display_name": record.display_name,
            "dialect": record.dialect,
            "enabled": record.enabled,
            "connection_ref": record.connection_ref,
            "default_database": record.default_database,
            "default_schema": record.default_schema,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
    )


class DatabaseDataSourceProvider:
    """Load datasource metadata from the trusted KnowledgeScope registry."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, datasource_id: UUID) -> DataSource | None:
        """Return only the registered datasource for the requested identity."""
        async with self._session_factory() as session:
            record = await session.scalar(
                select(ChatBIDataSourceRecord).where(ChatBIDataSourceRecord.id == datasource_id)
            )
        return data_source_from_record(record) if record is not None else None


__all__ = ["DatabaseDataSourceProvider", "data_source_from_record"]
