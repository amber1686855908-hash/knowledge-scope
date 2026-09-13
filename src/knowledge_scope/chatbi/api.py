"""Minimal ChatBI datasource metadata API; no SQL execution endpoint is exposed."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

from .discovery import SchemaDiscoveryService, create_postgres_schema_discovery_service
from .errors import ChatBIError, ChatBIErrorCategory
from .models import ChatBIDataSourceRecord
from .policy import default_query_policy
from .schema_models import SchemaDiscoveryResult
from .schemas import (
    DataSource,
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


def _to_internal(record: ChatBIDataSourceRecord) -> DataSource:
    """Build the internal datasource contract without exposing it in a response."""
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


def _build_schema_discovery_service(settings: Settings) -> SchemaDiscoveryService:
    """Create a stateless discovery service for one API request."""
    return create_postgres_schema_discovery_service(
        connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
        statement_timeout_ms=settings.chatbi_statement_timeout_ms,
    )


def _schema_discovery_http_error(error: ChatBIError) -> HTTPException:
    if error.category is ChatBIErrorCategory.DATASOURCE_DISABLED:
        error_status = status.HTTP_409_CONFLICT
    elif error.category is ChatBIErrorCategory.UNSUPPORTED_DIALECT:
        error_status = status.HTTP_422_UNPROCESSABLE_CONTENT
    else:
        error_status = status.HTTP_502_BAD_GATEWAY
    return HTTPException(
        status_code=error_status,
        detail={"category": error.category.value, "message": error.safe_message},
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


@router.get("/{datasource_id}/schema", response_model=SchemaDiscoveryResult)
async def discover_data_source_schema(
    datasource_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> SchemaDiscoveryResult:
    """Discover allow-listed external schema metadata without executing SQL."""
    record = await _get_data_source(session, datasource_id)
    settings = request.app.state.settings
    service = _build_schema_discovery_service(settings)
    try:
        return await service.discover(
            _to_internal(record),
            default_query_policy(settings),
            max_chars=settings.chatbi_schema_context_max_chars,
        )
    except ChatBIError as error:
        raise _schema_discovery_http_error(error) from None


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
