import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from knowledge_scope.chatbi import (
    DataSourceCreate,
    DataSourcePublic,
    DataSourceUpdate,
    QueryAuditRecord,
    QueryExecutionRequest,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryPolicy,
    SQLDialect,
    default_query_policy,
)
from knowledge_scope.chatbi.errors import ChatBIErrorCategory
from knowledge_scope.shared.config import Settings


def test_datasource_accepts_only_opaque_connection_references() -> None:
    data_source = DataSourceCreate(
        display_name="  销售数据库  ",
        connection_ref="env:CHATBI_DEMO_DATABASE_URL",
        default_schema=" public ",
    )

    assert data_source.display_name == "销售数据库"
    assert data_source.dialect is SQLDialect.POSTGRESQL
    assert data_source.default_schema == "public"

    for connection_ref in (
        "postgresql://user:password@example.test/db",
        "https://secret.example.test",
        "password",
        "env:",
    ):
        with pytest.raises(ValidationError):
            DataSourceCreate(display_name="销售数据库", connection_ref=connection_ref)

    with pytest.raises(ValidationError) as error:
        DataSourceCreate(
            display_name="销售数据库",
            connection_ref="postgresql://user:password@example.test/db",
        )
    assert "password" not in str(error.value)


def test_datasource_contract_rejects_unsupported_dialect_and_public_redacts_reference() -> None:
    with pytest.raises(ValidationError):
        DataSourceCreate(
            display_name="销售数据库",
            dialect="mysql",
            connection_ref="env:CHATBI_DEMO_DATABASE_URL",
        )

    public = DataSourcePublic(
        id=uuid4(),
        display_name="销售数据库",
        dialect=SQLDialect.POSTGRESQL,
        enabled=True,
        default_database=None,
        default_schema="public",
        connection_configured=True,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert "connection_ref" not in public.model_dump()
    assert "password" not in public.model_dump_json().lower()


def test_datasource_update_is_safe_metadata_only() -> None:
    update = DataSourceUpdate(display_name="新名称", enabled=False)

    assert update.model_dump(exclude_unset=True) == {"display_name": "新名称", "enabled": False}

    with pytest.raises(ValidationError):
        DataSourceUpdate()
    with pytest.raises(ValidationError):
        DataSourceUpdate.model_validate({"display_name": None})
    with pytest.raises(ValidationError):
        DataSourceUpdate.model_validate({"enabled": None})
    with pytest.raises(ValidationError):
        DataSourceUpdate.model_validate({"connection_ref": "env:NEW_REF"})


def test_query_policy_has_conservative_defaults_and_bounds() -> None:
    policy = QueryPolicy()

    assert policy.read_only is True
    assert policy.max_rows == 1_000
    assert policy.max_result_bytes == 4_000_000
    assert policy.max_cell_bytes == 1_000_000
    assert policy.max_nested_value_depth == 32
    assert policy.max_collection_items == 10_000
    assert policy.statement_timeout_ms == 30_000
    assert policy.allowed_schemas == ("public",)
    assert policy.allow_views is False
    assert policy.max_statement_count == 1

    for payload in (
        {"max_rows": 0},
        {"max_result_bytes": 1},
        {"max_cell_bytes": 0},
        {"max_nested_value_depth": 0},
        {"max_collection_items": 0},
        {"statement_timeout_ms": 99},
        {"read_only": False},
        {"allowed_schemas": []},
        {"allowed_schemas": ["public", "public"]},
        {"allowed_schemas": ["public"], "denied_schemas": ["public"]},
        {"max_statement_count": 2},
    ):
        with pytest.raises(ValidationError):
            QueryPolicy.model_validate(payload)


def test_query_policy_reads_only_the_chatbi_settings() -> None:
    settings = Settings(
        _env_file=None,
        chatbi_max_rows=250,
        chatbi_max_result_bytes=2_000_000,
        chatbi_max_cell_bytes=500_000,
        chatbi_max_nested_value_depth=16,
        chatbi_max_collection_items=5_000,
        chatbi_statement_timeout_ms=5_000,
        chatbi_allowed_schemas=["analytics", "public"],
        chatbi_allow_views=True,
    )

    policy = default_query_policy(settings)

    assert policy.max_rows == 250
    assert policy.max_result_bytes == 2_000_000
    assert policy.max_cell_bytes == 500_000
    assert policy.max_nested_value_depth == 16
    assert policy.max_collection_items == 5_000
    assert policy.statement_timeout_ms == 5_000
    assert policy.allowed_schemas == ("analytics", "public")
    assert policy.allow_views is True


def test_execution_request_is_intent_only_and_cannot_carry_raw_sql() -> None:
    datasource_id = uuid4()
    request = QueryExecutionRequest(
        datasource_id=datasource_id,
        question="  按客户统计销售额  ",
        context_metadata={"task": "evaluation"},
    )
    same_request = QueryExecutionRequest(
        query_id=request.query_id,
        datasource_id=datasource_id,
        question=request.question,
        context_metadata=request.context_metadata,
        created_at=request.created_at,
    )

    assert request.question == "按客户统计销售额"
    assert request == same_request
    with pytest.raises(ValidationError):
        QueryExecutionRequest.model_validate({"datasource_id": datasource_id, "sql": "SELECT 1"})
    with pytest.raises(ValidationError):
        QueryExecutionRequest.model_validate(
            {
                "datasource_id": datasource_id,
                "question": "查询销售额",
                "validated_sql": {"normalized_sql": "DROP TABLE public.sales"},
            }
        )


def test_query_audit_record_tracks_sql_identity_and_lifecycle() -> None:
    request = QueryExecutionRequest(datasource_id=uuid4(), question="查询销售额")
    sql_fingerprint = hashlib.sha256(b"SELECT 1").hexdigest()
    audit = QueryAuditRecord(
        query_id=request.query_id,
        datasource_id=request.datasource_id,
        sql_fingerprint=sql_fingerprint,
        state=QueryLifecycleState.VALIDATED,
    )

    assert audit.sql_fingerprint == sql_fingerprint
    assert "sql" not in audit.model_dump()
    assert "SELECT 1" not in audit.model_dump_json()

    with pytest.raises(ValidationError):
        QueryAuditRecord(
            query_id=request.query_id,
            datasource_id=request.datasource_id,
            sql_fingerprint=sql_fingerprint,
            state=QueryLifecycleState.SUCCEEDED,
            error_category=ChatBIErrorCategory.EXECUTION_FAILED,
        )
    with pytest.raises(ValidationError):
        QueryAuditRecord(
            query_id=request.query_id,
            datasource_id=request.datasource_id,
            sql_fingerprint=sql_fingerprint,
            state=QueryLifecycleState.FAILED,
        )


def test_execution_result_round_trips_deterministically() -> None:
    result = QueryExecutionResult(
        query_id=uuid4(),
        datasource_id=uuid4(),
        state=QueryLifecycleState.SUCCEEDED,
        columns=[
            {"name": "region", "data_type": "text", "ordinal": 0},
            {"name": "amount", "data_type": "numeric", "ordinal": 1},
        ],
        rows=[["华东", 1200.0]],
        row_count=1,
        truncated=False,
        duration_ms=12.5,
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    encoded = result.to_deterministic_json()
    decoded = QueryExecutionResult.model_validate_json(encoded)

    assert decoded == result
    assert json.loads(encoded) == result.model_dump(mode="json")
    assert list(json.loads(encoded)) == sorted(json.loads(encoded))


def test_execution_result_requires_consistent_terminal_data() -> None:
    base = {
        "query_id": uuid4(),
        "datasource_id": uuid4(),
        "columns": [{"name": "value", "data_type": "integer", "ordinal": 0}],
        "rows": [[1]],
        "row_count": 1,
        "duration_ms": 1.0,
    }

    with pytest.raises(ValidationError):
        QueryExecutionResult.model_validate({**base, "state": "created"})
    with pytest.raises(ValidationError):
        QueryExecutionResult.model_validate({**base, "state": "succeeded", "row_count": 2})
    with pytest.raises(ValidationError):
        QueryExecutionResult.model_validate({**base, "state": "failed"})
    with pytest.raises(ValidationError):
        QueryExecutionResult.model_validate(
            {
                **base,
                "state": "succeeded",
                "error_category": ChatBIErrorCategory.EXECUTION_FAILED,
            }
        )
