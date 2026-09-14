from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_scope.chatbi.models import ChatBIDataSourceRecord
from knowledge_scope.llm.schemas import LLMRequest, LLMResult
from knowledge_scope.mcp.server import build_mcp_application, create_mcp_server
from knowledge_scope.shared.config import Settings

DATASOURCE_ID = UUID("55555555-5555-4555-8555-555555555555")


class _DeterministicMCPGateway:
    """Fake gateway; discovery, validation, execution and normalization stay real."""

    async def complete(self, request: LLMRequest) -> LLMResult:
        if request.task_type == "nl2sql":
            text = json.dumps(
                {
                    "sql": (
                        "SELECT c.customer_name, SUM(s.amount) AS total_amount "
                        "FROM chatbi_demo.sales AS s "
                        "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
                        "GROUP BY c.customer_name ORDER BY c.customer_name"
                    )
                },
                ensure_ascii=False,
            )
            input_tokens, output_tokens = 10, 5
        elif request.task_type == "chatbi_analysis":
            text = '{"answer":"MCP 返回 2 位客户。","warning":null}'
            input_tokens, output_tokens = 11, 7
        else:  # pragma: no cover - the real Agent should use only these two tasks
            raise AssertionError(f"unexpected task type: {request.task_type}")
        return LLMResult(
            text=text,
            provider="fake-mcp",
            model="fake-mcp-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1,
            provider_attempts=1,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_MCP_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_MCP_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_mcp_chatbi_path(
    postgres_test_engine: AsyncEngine,
    postgres_test_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the MCP protocol through real discovery, validation and execution."""
    engine = postgres_test_engine
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)
        await connection.execute(
            insert(ChatBIDataSourceRecord).values(
                id=DATASOURCE_ID,
                display_name="MCP 演示库",
                dialect="postgresql",
                enabled=True,
                connection_ref="env:KNOWLEDGE_SCOPE_MCP_TEST_DATABASE_URL",
                default_schema="chatbi_demo",
            )
        )

    monkeypatch.setenv("KNOWLEDGE_SCOPE_MCP_TEST_DATABASE_URL", postgres_test_database)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=postgres_test_database,
        chatbi_allowed_schemas=["chatbi_demo"],
        chatbi_max_rows=10,
        chatbi_schema_context_max_chars=20_000,
    )
    application = build_mcp_application(settings, gateway=_DeterministicMCPGateway())
    server = create_mcp_server(application)
    try:
        async with create_connected_server_and_client_session(server) as client:
            schema_response = await client.call_tool(
                "chatbi_schema",
                {"datasource_id": str(DATASOURCE_ID)},
            )
            assert schema_response.isError is False
            schema_payload = schema_response.structuredContent
            assert schema_payload is not None
            assert schema_payload["status"] == "ok"
            schema_result = schema_payload["result"]
            assert schema_result["context"]["included_relations"] == [  # type: ignore[index]
                "chatbi_demo.customers",
                "chatbi_demo.sales",
            ]

            response = await client.call_tool(
                "chatbi_ask",
                {"datasource_id": str(DATASOURCE_ID), "question": "按客户统计销售额"},
            )
            assert response.isError is False
            payload = response.structuredContent
            assert payload is not None
            result = payload["result"]
            assert result["answer"] == "MCP 返回 2 位客户。"  # type: ignore[index]
            assert result["datasource_id"] == str(DATASOURCE_ID)  # type: ignore[index]
            assert result["row_count"] == 2  # type: ignore[index]
            assert result["truncated"] is False  # type: ignore[index]
            assert "chatbi_demo" in result["redacted_sql"]  # type: ignore[index]
            assert result["usage"]["llm_calls"] == 2  # type: ignore[index]
            assert result["usage"]["provider"] == "fake-mcp"  # type: ignore[index]
            assert result["usage"]["model"] == "fake-mcp-model"  # type: ignore[index]
            assert result["usage"]["input_tokens"] == 21  # type: ignore[index]
            assert result["usage"]["output_tokens"] == 12  # type: ignore[index]
            assert [event["event"] for event in result["trace"]] == [  # type: ignore[index]
                "schema_prepared",
                "sql_generated",
                "validation_accepted",
                "sql_executed",
                "result_normalized",
                "analysis_requested",
                "analysis_completed",
            ]
            serialized = json.dumps(payload, ensure_ascii=False)
            assert "KNOWLEDGE_SCOPE_MCP_TEST_DATABASE_URL" not in serialized
            assert "password" not in serialized.lower()

            unknown_response = await client.call_tool(
                "chatbi_schema",
                {"datasource_id": str(UUID("66666666-6666-4666-8666-666666666666"))},
            )
            assert unknown_response.isError is True
            assert unknown_response.structuredContent == {
                "status": "error",
                "result": None,
                "error": {
                    "category": "datasource_not_found",
                    "message": "data source not found",
                },
            }
    finally:
        await application.aclose()
