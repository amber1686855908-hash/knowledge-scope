"""Minimal ChatBI datasource metadata API; no SQL execution endpoint is exposed."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.shared.database import get_session

from .models import ChatBIDataSourceRecord
from .schemas import (
    DataSourceCreate,
    DataSourceListResponse,
    DataSourcePublic,
    DataSourceUpdate,
)

router = APIRouter(prefix="/chatbi/data-sources", tags=["chatbi"])


def _to_public(record: ChatBIDataSourceRecord) -> DataSourcePublic:
    """Project a record without exposing its opaque connection reference."""
    return DataSourcePublic(
        id=record.id,
        display_name=record.display_name,
        dialect=record.dialect,
        enabled=record.enabled,
        default_database=record.default_database,
        default_schema=record.default_schema,
        connection_configured=bool(record.connection_ref),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def _get_data_source(session: AsyncSession, datasource_id: UUID) -> ChatBIDataSourceRecord:
    record = await session.scalar(
        select(ChatBIDataSourceRecord).where(ChatBIDataSourceRecord.id == datasource_id)
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )
    return record


@router.post("", response_model=DataSourcePublic, status_code=status.HTTP_201_CREATED)
async def create_data_source(
    payload: DataSourceCreate,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    """Register safe metadata and an opaque external connection reference."""
    record = ChatBIDataSourceRecord(
        display_name=payload.display_name,
        dialect=payload.dialect.value,
        enabled=payload.enabled,
        connection_ref=payload.connection_ref,
        default_database=payload.default_database,
        default_schema=payload.default_schema,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return _to_public(record)


@router.get("", response_model=DataSourceListResponse)
async def list_data_sources(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> DataSourceListResponse:
    total = int(await session.scalar(select(func.count()).select_from(ChatBIDataSourceRecord)) or 0)
    result = await session.scalars(
        select(ChatBIDataSourceRecord)
        .order_by(ChatBIDataSourceRecord.created_at.desc(), ChatBIDataSourceRecord.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return DataSourceListResponse(
        items=[_to_public(record) for record in result.all()],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{datasource_id}", response_model=DataSourcePublic)
async def get_data_source(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    return _to_public(await _get_data_source(session, datasource_id))


@router.patch("/{datasource_id}", response_model=DataSourcePublic)
async def update_data_source(
    datasource_id: UUID,
    payload: DataSourceUpdate,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    record = await _get_data_source(session, datasource_id)
    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(record, field_name, value)
    record.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(record)
    return _to_public(record)


@router.delete("/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    record = await _get_data_source(session, datasource_id)
    await session.delete(record)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
