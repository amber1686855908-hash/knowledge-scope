"""Alembic environment configured for async PostgreSQL migrations."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from knowledge_scope.documents import models as document_models
from knowledge_scope.knowledge_bases import models as _knowledge_base_models  # noqa: F401
from knowledge_scope.shared.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = document_models.Base.metadata


def database_url() -> str:
    """Return an explicitly configured URL or the application setting."""
    configured_url = config.get_main_option("sqlalchemy.url")
    return configured_url or get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure Alembic with a live SQLAlchemy connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and execute the migration transaction."""
    connectable = async_engine_from_config(
        {"sqlalchemy.url": database_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations against PostgreSQL."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
