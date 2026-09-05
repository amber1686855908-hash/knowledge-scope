"""Knowledge base CRUD endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.knowledge_bases.models import KnowledgeBase
from knowledge_scope.shared.database import get_session

from .schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


async def _get_knowledge_base(session: AsyncSession, knowledge_base_id: UUID) -> KnowledgeBase:
    knowledge_base = await session.scalar(
        select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
    )
    if knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )
    return knowledge_base


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(name=payload.name, description=payload.description)
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeBaseListResponse:
    total = int(await session.scalar(select(func.count()).select_from(KnowledgeBase)) or 0)
    result = await session.scalars(
        select(KnowledgeBase)
        .order_by(KnowledgeBase.created_at.desc(), KnowledgeBase.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return KnowledgeBaseListResponse(
        items=list(result.all()),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeBase:
    return await _get_knowledge_base(session, knowledge_base_id)


@router.patch("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    knowledge_base_id: UUID,
    payload: KnowledgeBaseUpdate,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeBase:
    knowledge_base = await _get_knowledge_base(session, knowledge_base_id)
    changes = payload.model_dump(exclude_unset=True)
    for field_name, value in changes.items():
        setattr(knowledge_base, field_name, value)
    knowledge_base.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


@router.delete("/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    knowledge_base = await _get_knowledge_base(session, knowledge_base_id)
    await session.delete(knowledge_base)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
