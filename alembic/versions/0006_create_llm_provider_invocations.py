"""Create per-provider-attempt observability records."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_invocations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", sa.String(length=128), nullable=True),
        sa.Column("logical_stage", sa.String(length=32), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Numeric(precision=12, scale=3), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("status_class", sa.String(length=8), nullable=True),
        sa.Column("error_category", sa.String(length=64), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=True),
        sa.Column("llm_result_returned", sa.Boolean(), nullable=False),
        sa.Column("response_parse_outcome", sa.String(length=48), nullable=True),
        sa.Column("output_token_budget", sa.Integer(), nullable=True),
        sa.Column("token_limit_status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempt_index >= 1",
            name="ck_llm_provider_invocations_attempt_index_positive",
        ),
        sa.CheckConstraint(
            "duration_ms >= 0",
            name="ck_llm_provider_invocations_duration_non_negative",
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_llm_provider_invocations_input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_llm_provider_invocations_output_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_token_budget IS NULL OR output_token_budget >= 1",
            name="ck_llm_provider_invocations_output_budget_positive",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_provider_invocations_started_at",
        "llm_provider_invocations",
        ["started_at"],
    )
    op.create_index(
        "ix_llm_provider_invocations_case_id",
        "llm_provider_invocations",
        ["case_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_provider_invocations_case_id", table_name="llm_provider_invocations")
    op.drop_index(
        "ix_llm_provider_invocations_started_at",
        table_name="llm_provider_invocations",
    )
    op.drop_table("llm_provider_invocations")
