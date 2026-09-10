"""Allow explicit external reference document registrations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("storage_kind", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("source_ref", sa.String(length=2048), nullable=True),
    )
    op.alter_column(
        "documents",
        "storage_key",
        existing_type=sa.String(length=512),
        nullable=True,
    )
    op.execute("UPDATE documents SET storage_kind = 'managed' WHERE storage_kind IS NULL")
    op.alter_column(
        "documents",
        "storage_kind",
        existing_type=sa.String(length=32),
        nullable=False,
        server_default="managed",
    )
    op.drop_constraint("ck_documents_status_uploaded", "documents", type_="check")
    op.drop_constraint("ck_documents_storage_key_relative_safe", "documents", type_="check")
    op.create_check_constraint(
        "ck_documents_status_known",
        "documents",
        "status IN ('uploaded', 'registered')",
    )
    op.create_check_constraint(
        "ck_documents_storage_reference_consistent",
        "documents",
        """
        storage_kind IN ('managed', 'external_reference')
        AND (
            (
                storage_kind = 'managed'
                AND storage_key IS NOT NULL
                AND storage_key <> ''
                AND storage_key NOT LIKE '/%'
                AND storage_key NOT LIKE '%..%'
                AND source_ref IS NULL
            )
            OR (
                storage_kind = 'external_reference'
                AND storage_key IS NULL
                AND source_ref IS NOT NULL
                AND btrim(source_ref) = source_ref
                AND source_ref <> ''
                AND source_ref NOT LIKE '/%'
                AND source_ref NOT LIKE '%..%'
                AND source_ref NOT LIKE '%:%'
            )
        )
        """,
    )


def downgrade() -> None:
    connection = op.get_bind()
    unsupported_count = connection.execute(
        sa.text(
            """
            SELECT count(*)
            FROM documents
            WHERE storage_kind <> 'managed' OR storage_key IS NULL
            """
        )
    ).scalar_one()
    if unsupported_count:
        raise RuntimeError(
            "cannot downgrade 0004 while external-reference documents exist; "
            "remove or migrate them explicitly first"
        )
    op.drop_constraint("ck_documents_storage_reference_consistent", "documents", type_="check")
    op.drop_constraint("ck_documents_status_known", "documents", type_="check")
    op.create_check_constraint(
        "ck_documents_storage_key_relative_safe",
        "documents",
        "storage_key <> '' AND storage_key NOT LIKE '/%' AND storage_key NOT LIKE '%..%'",
    )
    op.create_check_constraint(
        "ck_documents_status_uploaded",
        "documents",
        "status = 'uploaded'",
    )
    op.alter_column(
        "documents",
        "storage_key",
        existing_type=sa.String(length=512),
        nullable=False,
    )
    op.drop_column("documents", "source_ref")
    op.drop_column("documents", "storage_kind")
