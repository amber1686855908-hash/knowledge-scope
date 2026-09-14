from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from knowledge_scope.chatbi import (
    ChatBIAgentService,
    DataSource,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryLifecycleState,
    QueryPolicy,
    SchemaDiscoveryService,
    SQLDialect,
    SQLExecutionService,
)
from knowledge_scope.chatbi.postgres_schema import PostgresSchemaInspector
from knowledge_scope.llm.schemas import LLMRequest, LLMResult

DATASOURCE_ID = UUID("22222222-2222-4222-8222-222222222222")


class _DeterministicAgentGateway:
    """A provider-free gateway stub that still exercises both agent calls."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task_type == "nl2sql":
            text = (
                '{"sql":"SELECT c.customer_name, SUM(s.amount) AS total_amount '
                "FROM chatbi_demo.sales AS s "
                "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
                'GROUP BY c.customer_name ORDER BY c.customer_name"}'
            )
            input_tokens, output_tokens = 10, 5
        elif request.task_type == "chatbi_analysis":
            text = '{"answer":"结果包含 2 位客户。","warning":null}'
            input_tokens, output_tokens = 11, 7
        else:
            raise AssertionError(f"unexpected task type: {request.task_type}")
        return LLMResult(
            text=text,
            provider="fake-agent",
            model="fake-agent-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1,
            provider_attempts=1,
        )


class _RegisteredDataSource:
    def __init__(self, source: DataSource) -> None:
        self.source = source
        self.calls = 0

    async def get(self, datasource_id: UUID) -> DataSource | None:
        assert datasource_id == self.source.id
        self.calls += 1
        return self.source


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_chatbi_agent_end_to_end(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """Run the agent through real discovery, validation, execution and normalization."""
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    os.environ["CHATBI_AGENT_TEST_DATABASE_URL"] = postgres_test_database
    try:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        source = DataSource(
            id=DATASOURCE_ID,
            display_name="ChatBI 演示库",
            dialect=SQLDialect.POSTGRESQL,
            enabled=True,
            connection_ref="env:CHATBI_AGENT_TEST_DATABASE_URL",
            default_database=None,
            default_schema="chatbi_demo",
            created_at=timestamp,
            updated_at=timestamp,
        )
        registry = _RegisteredDataSource(source)
        discovery = SchemaDiscoveryService(
            EnvironmentCredentialResolver(),
            PostgresSchemaInspector(),
        )
        gateway = _DeterministicAgentGateway()
        generation = NL2SQLService(
            gateway,
            schema_discovery=discovery,
            data_source_provider=registry,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(),
        )

        result = await ChatBIAgentService(generation, execution, gateway).ask(
            DATASOURCE_ID,
            "按客户统计销售额",
            policy=QueryPolicy(allowed_schemas=("chatbi_demo",), max_rows=10),
            max_chars=20_000,
        )

        assert result.execution_status is QueryLifecycleState.SUCCEEDED
        assert result.datasource_id == DATASOURCE_ID
        assert result.answer == "结果包含 2 位客户。"
        assert result.row_count == 2
        assert result.truncated is False
        assert result.truncation_reason is None
        assert result.error_category is None
        assert result.redacted_sql is not None
        assert "chatbi_demo" in result.redacted_sql
        assert result.usage.llm_calls == 2
        assert result.usage.provider_attempts == 2
        assert result.usage.provider == "fake-agent"
        assert result.usage.model == "fake-agent-model"
        assert result.usage.input_tokens == 21
        assert result.usage.output_tokens == 12
        assert [request.task_type for request in gateway.requests] == [
            "nl2sql",
            "chatbi_analysis",
        ]
        assert registry.calls >= 2
        assert [event.event for event in result.trace] == [
            "schema_prepared",
            "sql_generated",
            "validation_accepted",
            "sql_executed",
            "result_normalized",
            "analysis_requested",
            "analysis_completed",
        ]

        safe_result = result.to_deterministic_json()
        assert "CHATBI_AGENT_TEST_DATABASE_URL" not in safe_result
        assert "knowledgescope:knowledgescope" not in safe_result
        assert "password" not in safe_result.lower()
    finally:
        os.environ.pop("CHATBI_AGENT_TEST_DATABASE_URL", None)
