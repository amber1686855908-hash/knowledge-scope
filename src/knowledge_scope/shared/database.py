"""Async SQLAlchemy engine, session, and declarative base setup."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from knowledge_scope.shared.config import Settings


class Base(DeclarativeBase):
    """Base class for persistent KnowledgeScope models."""


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Create an async engine without opening a connection eagerly."""
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the request-session factory used by the API."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield one request-scoped session and always close it afterwards."""
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.db_session_factory
    async with session_factory() as session:
        yield session
