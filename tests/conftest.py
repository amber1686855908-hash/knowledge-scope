from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from knowledge_scope.api.app import create_app
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATABASE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _create_database(base_url: URL, database_name: str) -> None:
    engine = create_async_engine(
        base_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    finally:
        await engine.dispose()


async def _drop_database(base_url: URL, database_name: str) -> None:
    engine = create_async_engine(
        base_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
            )
    finally:
        await engine.dispose()


def _upgrade_database(database_url: str) -> None:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(alembic_config, "head")


def _resolve_test_base_url() -> URL:
    """Resolve the server URL without allowing an accidental remote fallback."""
    configured_url = os.environ.get("KNOWLEDGE_SCOPE_TEST_DATABASE_URL")
    if configured_url is not None:
        if not configured_url.strip():
            raise pytest.UsageError(
                "KNOWLEDGE_SCOPE_TEST_DATABASE_URL must be a non-empty PostgreSQL URL."
            )
        return make_url(configured_url)

    base_url = make_url(Settings(_env_file=None).database_url)
    if base_url.host not in LOCAL_DATABASE_HOSTS:
        host = base_url.host or "<missing>"
        raise pytest.UsageError(
            "KNOWLEDGE_SCOPE_DATABASE_URL points to non-local host "
            f"{host!r}; set KNOWLEDGE_SCOPE_TEST_DATABASE_URL explicitly before "
            "running PostgreSQL integration tests."
        )
    return base_url


@pytest.fixture(scope="session")
def postgres_test_database() -> Iterator[str]:
    base_url = _resolve_test_base_url()
    database_name = f"knowledgescope_test_{uuid4().hex}"

    asyncio.run(_create_database(base_url, database_name))

    test_url = base_url.set(database=database_name).render_as_string(hide_password=False)
    try:
        _upgrade_database(test_url)
        yield test_url
    finally:
        asyncio.run(_drop_database(base_url, database_name))


@pytest.fixture
async def postgres_test_engine(postgres_test_database: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(postgres_test_database, pool_pre_ping=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def client(
    postgres_test_database: str,
    postgres_test_engine: AsyncEngine,
) -> AsyncIterator[AsyncClient]:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=postgres_test_database,
    )
    application = create_app(settings, database_engine=postgres_test_engine)

    async with postgres_test_engine.connect() as connection:
        transaction = await connection.begin()
        session_factory = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

        async def override_session() -> AsyncIterator[AsyncSession]:
            async with session_factory() as session:
                yield session

        application.dependency_overrides[get_session] = override_session
        transport = ASGITransport(app=application)
        try:
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as http_client:
                yield http_client
        finally:
            application.dependency_overrides.clear()
            await transaction.rollback()
