"""Create the LLM usage observability table."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Numeric(precision=12, scale=3), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_category", sa.String(length=32), nullable=True),
        sa.Column("estimated_cost", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_llm_usage_records_input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_llm_usage_records_output_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name="ck_llm_usage_records_latency_non_negative",
        ),
        sa.CheckConstraint(
            "estimated_cost IS NULL OR estimated_cost >= 0",
            name="ck_llm_usage_records_cost_non_negative",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_usage_records_created_at",
        "llm_usage_records",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_usage_records_created_at", table_name="llm_usage_records")
    op.drop_table("llm_usage_records")
