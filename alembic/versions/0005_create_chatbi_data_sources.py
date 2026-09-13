"""Create safe ChatBI datasource metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chatbi_data_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("dialect", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("connection_ref", sa.String(length=255), nullable=False),
        sa.Column("default_database", sa.String(length=63), nullable=True),
        sa.Column("default_schema", sa.String(length=63), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "display_name = btrim(display_name) AND btrim(display_name) <> ''",
            name="ck_chatbi_data_sources_display_name_non_empty",
        ),
        sa.CheckConstraint(
            "dialect = 'postgresql'",
            name="ck_chatbi_data_sources_supported_dialect",
        ),
        sa.CheckConstraint(
            "connection_ref ~ '^(env|secret):[A-Za-z][A-Za-z0-9_.:/-]{0,247}$'",
            name="ck_chatbi_data_sources_connection_ref_format",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chatbi_data_sources_enabled",
        "chatbi_data_sources",
        ["enabled"],
    )


def downgrade() -> None:
    op.drop_index("ix_chatbi_data_sources_enabled", table_name="chatbi_data_sources")
    op.drop_table("chatbi_data_sources")
