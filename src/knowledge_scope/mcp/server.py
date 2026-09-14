"""Small, provider-independent MCP adapter for the trusted ChatBI services."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, Protocol
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_scope import __version__
from knowledge_scope.chatbi import (
    ChatBIAgentLimits,
    ChatBIAgentService,
    ChatBIError,
    ChatBIErrorCategory,
    ChatBIResult,
    QueryLifecycleState,
    SchemaContextBudgetError,
    SchemaDiscoveryResult,
    SchemaRelation,
    SQLExecutionService,
    build_semantic_schema_context,
    default_query_policy,
)
from knowledge_scope.chatbi.credentials import EnvironmentCredentialResolver
from knowledge_scope.chatbi.discovery import create_postgres_schema_discovery_service
from knowledge_scope.chatbi.execution import PostgresExecutionAdapter
from knowledge_scope.chatbi.nl2sql import NL2SQLService
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.providers import create_llm_provider
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

MCP_DEFAULT_MAX_IN_FLIGHT = 4
MCP_MAX_IN_FLIGHT_LIMIT = 32
MCP_SCHEMA_MAX_TEXT_CHARS = 8_000
MCP_SCHEMA_MAX_RESPONSE_BYTES = 64_000
MCP_SCHEMA_MAX_RELATIONS = 128
MCP_SCHEMA_MAX_COLUMNS = 2_048
MCP_SCHEMA_MAX_COLUMNS_PER_RELATION = 256
MCP_SCHEMA_MAX_STRUCTURAL_ITEMS_PER_RELATION = 512
MCP_SCHEMA_OMISSION_SAMPLE_SIZE = 16


class _MCPModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class MCPAskArguments(_MCPModel):
    """Only the datasource identity and user question cross the MCP boundary."""

    datasource_id: UUID
    question: StrictStr = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_question(self) -> MCPAskArguments:
        if not self.question.strip():
            raise ValueError("question must contain non-whitespace characters")
        return self


class MCPSchemaArguments(_MCPModel):
    """Arguments for bounded schema discovery."""

    datasource_id: UUID


class MCPToolError(_MCPModel):
    """Stable, non-sensitive tool error data."""

    category: StrictStr = Field(min_length=1, max_length=64)
    message: StrictStr = Field(min_length=1, max_length=500)


class MCPSchemaContext(_MCPModel):
    """Bounded public schema projection with count-based omission metadata."""

    schema_version: Literal["1.0"]
    snapshot_fingerprint: StrictStr = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    text: StrictStr = Field(min_length=1, max_length=MCP_SCHEMA_MAX_TEXT_CHARS)
    max_chars: int = Field(ge=1, le=MCP_SCHEMA_MAX_TEXT_CHARS)
    included_relations: tuple[str, ...] = Field(max_length=MCP_SCHEMA_MAX_RELATIONS)
    relation_count: int = Field(ge=0)
    returned_relation_count: int = Field(ge=0)
    omitted_relation_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    returned_column_count: int = Field(ge=0)
    omitted_column_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    returned_relationship_count: int = Field(ge=0)
    omitted_relationship_count: int = Field(ge=0)
    omitted_relation_sample: tuple[str, ...] = Field(
        default=(),
        max_length=MCP_SCHEMA_OMISSION_SAMPLE_SIZE,
    )
    omitted_relationship_sample: tuple[str, ...] = Field(
        default=(),
        max_length=MCP_SCHEMA_OMISSION_SAMPLE_SIZE,
    )
    truncated: bool = False

    @model_validator(mode="after")
    def validate_projection(self) -> MCPSchemaContext:
        if len(self.text) > self.max_chars:
            raise ValueError("MCP schema context exceeds its character budget")
        if self.returned_relation_count != len(self.included_relations):
            raise ValueError("returned relation count must match included relations")
        if self.relation_count != self.returned_relation_count + self.omitted_relation_count:
            raise ValueError("relation omission counts are inconsistent")
        if self.column_count != self.returned_column_count + self.omitted_column_count:
            raise ValueError("column omission counts are inconsistent")
        if self.relationship_count != (
            self.returned_relationship_count + self.omitted_relationship_count
        ):
            raise ValueError("relationship omission counts are inconsistent")
        if len(set(self.included_relations)) != len(self.included_relations):
            raise ValueError("included relations must be unique")
        if len(set(self.omitted_relation_sample)) != len(self.omitted_relation_sample):
            raise ValueError("omitted relation samples must be unique")
        if len(set(self.omitted_relationship_sample)) != len(self.omitted_relationship_sample):
            raise ValueError("omitted relationship samples must be unique")
        has_omissions = any(
            (
                self.omitted_relation_count,
                self.omitted_column_count,
                self.omitted_relationship_count,
            )
        )
        if self.truncated is not has_omissions:
            raise ValueError("truncated must match the omission counts")
        if len(self.omitted_relation_sample) > self.omitted_relation_count:
            raise ValueError("omitted relation sample exceeds omitted count")
        if len(self.omitted_relationship_sample) > self.omitted_relationship_count:
            raise ValueError("omitted relationship sample exceeds omitted count")
        return self


class MCPAskResponse(_MCPModel):
    """Structured response envelope for ``chatbi_ask``."""

    status: Literal["ok", "error"]
    result: ChatBIResult | None = None
    error: MCPToolError | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> MCPAskResponse:
        if self.status == "ok" and (self.result is None or self.error is not None):
            raise ValueError("successful MCP responses require a result only")
        if self.status == "error" and (self.result is not None or self.error is None):
            raise ValueError("failed MCP responses require an error only")
        return self


class MCPSchemaPayload(_MCPModel):
    """Bounded schema projection; raw comments and connection data are omitted."""

    schema_version: Literal["1.0"]
    fingerprint: StrictStr = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    context: MCPSchemaContext

    @model_validator(mode="after")
    def validate_response_size(self) -> MCPSchemaPayload:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > MCP_SCHEMA_MAX_RESPONSE_BYTES:
            raise ValueError("MCP schema response exceeds its byte budget")
        return self


class MCPSchemaResponse(_MCPModel):
    """Structured response envelope for ``chatbi_schema``."""

    status: Literal["ok", "error"]
    result: MCPSchemaPayload | None = None
    error: MCPToolError | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> MCPSchemaResponse:
        if self.status == "ok" and (self.result is None or self.error is not None):
            raise ValueError("successful MCP responses require a result only")
        if self.status == "error" and (self.result is not None or self.error is None):
            raise ValueError("failed MCP responses require an error only")
        return self


class MCPErrorResponse(_MCPModel):
    """Generic safe envelope used for an unknown tool name."""

    status: Literal["error"] = "error"
    error: MCPToolError


class MCPApplication(Protocol):
    """Application surface deliberately narrower than the ChatBI internals."""

    async def chatbi_ask(self, datasource_id: UUID, question: str) -> ChatBIResult:
        """Run the existing bounded ChatBI agent."""

    async def chatbi_schema(self, datasource_id: UUID) -> SchemaDiscoveryResult:
        """Run trusted registered-datasource schema discovery."""


class _ClosableProvider(Protocol):
    provider_name: str

    async def aclose(self) -> None:
        """Close provider-owned resources."""


class ProductionMCPApplication:
    """Per-process MCP application composition using existing ChatBI services."""

    def __init__(self, settings: Settings, *, gateway: object | None = None) -> None:
        self._engine: AsyncEngine = create_database_engine(settings)
        session_factory = create_session_factory(self._engine)
        self._provider: _ClosableProvider | None = None
        if gateway is None:
            provider = create_llm_provider(settings)
            self._provider = provider
            gateway = LLMGateway(provider, DatabaseUsageRecorder(session_factory), settings)
        if not callable(getattr(gateway, "complete", None)):
            raise TypeError("gateway must provide an async complete method")

        self._data_source_provider = DatabaseDataSourceProvider(session_factory)
        self._discovery = create_postgres_schema_discovery_service(
            connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
            statement_timeout_ms=settings.chatbi_statement_timeout_ms,
        )
        generation = NL2SQLService(
            gateway,  # type: ignore[arg-type]
            schema_discovery=self._discovery,
            data_source_provider=self._data_source_provider,
            max_tokens=settings.chatbi_nl2sql_max_tokens,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(
                connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                statement_timeout_ms=settings.chatbi_statement_timeout_ms,
            ),
        )
        self._agent = ChatBIAgentService(
            generation,
            execution,
            gateway,  # type: ignore[arg-type]
            limits=ChatBIAgentLimits.from_settings(settings),
        )
        self._policy = default_query_policy(settings)
        self._max_chars = settings.chatbi_schema_context_max_chars
        self._closed = False

    async def chatbi_ask(self, datasource_id: UUID, question: str) -> ChatBIResult:
        """Delegate to the existing request-local Agent with trusted settings."""
        return await self._agent.ask(
            datasource_id,
            question,
            policy=self._policy,
            max_chars=self._max_chars,
        )

    async def chatbi_schema(self, datasource_id: UUID) -> SchemaDiscoveryResult:
        """Resolve the datasource from PostgreSQL-backed registry before discovery."""
        data_source = await self._data_source_provider.get(datasource_id)
        if data_source is None:
            raise ChatBIError(ChatBIErrorCategory.DATASOURCE_NOT_FOUND, "data source not found")
        return await self._discovery.discover(data_source, self._policy, max_chars=self._max_chars)

    async def aclose(self) -> None:
        """Close provider and database resources; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._provider is not None:
                await self._provider.aclose()
        finally:
            await self._engine.dispose()


def build_mcp_application(
    settings: Settings,
    *,
    gateway: object | None = None,
) -> ProductionMCPApplication:
    """Build one process-scoped application without making a provider call."""
    return ProductionMCPApplication(settings, gateway=gateway)


_TOOL_NAMES = frozenset({"chatbi_ask", "chatbi_schema"})


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name="chatbi_ask",
            description="Ask one registered ChatBI datasource through the bounded safe pipeline.",
            inputSchema=MCPAskArguments.model_json_schema(),
            outputSchema=MCPAskResponse.model_json_schema(),
        ),
        Tool(
            name="chatbi_schema",
            description="Discover a bounded schema context for one registered datasource.",
            inputSchema=MCPSchemaArguments.model_json_schema(),
            outputSchema=MCPSchemaResponse.model_json_schema(),
        ),
    ]


def _json_text(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _call_result(value: BaseModel) -> CallToolResult:
    payload = value.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=_json_text(value))],
        structuredContent=payload,
        isError=payload.get("status") == "error",
    )


_SAFE_ERROR_MESSAGES: dict[ChatBIErrorCategory, str] = {
    ChatBIErrorCategory.DATASOURCE_NOT_FOUND: "data source not found",
    ChatBIErrorCategory.DATASOURCE_DISABLED: "data source is disabled",
    ChatBIErrorCategory.UNSUPPORTED_DIALECT: "data source dialect is not supported",
    ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED: "data source credentials are unavailable",
    ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED: "schema discovery failed",
    ChatBIErrorCategory.INVALID_QUERY: "request is invalid",
    ChatBIErrorCategory.GENERATION_FAILED: "SQL generation failed",
    ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT: "model output was malformed",
    ChatBIErrorCategory.SQL_PARSE_ERROR: "generated SQL could not be parsed",
    ChatBIErrorCategory.UNSAFE_QUERY: "generated SQL was rejected by policy",
    ChatBIErrorCategory.POLICY_VIOLATION: "request was rejected by policy",
    ChatBIErrorCategory.UNKNOWN_TABLE: "generated SQL references an unknown table",
    ChatBIErrorCategory.UNKNOWN_COLUMN: "generated SQL references an unknown column",
    ChatBIErrorCategory.DATASOURCE_UNAVAILABLE: "data source is unavailable",
    ChatBIErrorCategory.EXECUTION_TIMEOUT: "query execution timed out",
    ChatBIErrorCategory.EXECUTION_FAILED: "query execution failed",
    ChatBIErrorCategory.RESULT_NORMALIZATION_FAILED: "query result normalization failed",
    ChatBIErrorCategory.RESULT_SIZE_EXCEEDED: "query result exceeded its configured bound",
    ChatBIErrorCategory.EXECUTION_CANCELLED: "query execution was cancelled",
    ChatBIErrorCategory.ANALYSIS_FAILED: "result analysis failed",
    ChatBIErrorCategory.AGENT_LIMIT_EXCEEDED: "ChatBI request reached its configured bound",
}


def _chatbi_error(error: ChatBIError) -> MCPToolError:
    return MCPToolError(
        category=error.category.value,
        message=_SAFE_ERROR_MESSAGES.get(error.category, "ChatBI request failed"),
    )


def _invalid_arguments(tool_name: str) -> MCPToolError:
    return MCPToolError(category="invalid_arguments", message=f"{tool_name} arguments are invalid")


def _internal_error() -> MCPToolError:
    return MCPToolError(category="internal_error", message="MCP tool failed")


def _parse_arguments(
    model: type[BaseModel],
    tool_name: str,
    arguments: object,
) -> BaseModel | MCPToolError:
    if not isinstance(arguments, Mapping):
        return _invalid_arguments(tool_name)
    try:
        return model.model_validate(dict(arguments))
    except (TypeError, ValueError, ValidationError):
        return _invalid_arguments(tool_name)


def _mcp_relation_label(relation: SchemaRelation) -> str:
    return f"{relation.schema_name}.{relation.name}"


def _mcp_relationship_label(relation: SchemaRelation, foreign_key: object) -> str:
    constraint_name = getattr(foreign_key, "constraint_name", "")
    target_schema = getattr(foreign_key, "target_schema", "")
    target_relation = getattr(foreign_key, "target_relation", "")
    return f"{_mcp_relation_label(relation)}.{constraint_name}->{target_schema}.{target_relation}"


def _mcp_relation_without_comments(relation: SchemaRelation) -> SchemaRelation:
    return relation.model_copy(
        update={
            "comment": None,
            "columns": tuple(
                column.model_copy(update={"comment": None}) for column in relation.columns
            ),
        }
    )


def _mcp_structural_item_count(relation: SchemaRelation) -> int:
    return (
        len(relation.columns)
        + len(relation.primary_key)
        + len(relation.unique_constraints)
        + len(relation.foreign_keys)
    )


def _schema_payload(result: SchemaDiscoveryResult) -> MCPSchemaPayload:
    if not isinstance(result, SchemaDiscoveryResult):
        raise TypeError("schema discovery returned an invalid result")

    relation_count = len(result.snapshot.relations)
    column_count = sum(len(relation.columns) for relation in result.snapshot.relations)
    selected_relations: list[SchemaRelation] = []
    selected_columns = 0
    for relation in result.snapshot.relations:
        if len(selected_relations) >= MCP_SCHEMA_MAX_RELATIONS:
            continue
        if len(relation.columns) > MCP_SCHEMA_MAX_COLUMNS_PER_RELATION:
            continue
        if selected_columns + len(relation.columns) > MCP_SCHEMA_MAX_COLUMNS:
            continue
        if _mcp_structural_item_count(relation) > MCP_SCHEMA_MAX_STRUCTURAL_ITEMS_PER_RELATION:
            continue
        selected_relations.append(_mcp_relation_without_comments(relation))
        selected_columns += len(relation.columns)

    public_snapshot = result.snapshot.model_copy(update={"relations": tuple(selected_relations)})
    public_max_chars = min(result.context.max_chars, MCP_SCHEMA_MAX_TEXT_CHARS)
    try:
        context = build_semantic_schema_context(public_snapshot, max_chars=public_max_chars)
    except SchemaContextBudgetError as error:
        raise ValueError("MCP schema projection cannot fit its bounded envelope") from error

    all_relation_labels = tuple(
        _mcp_relation_label(relation) for relation in result.snapshot.relations
    )
    included_relation_set = set(context.included_relations)
    all_relationship_labels: list[str] = []
    returned_relationship_labels: list[str] = []
    omitted_relationships = set(context.omitted_relationships)
    for relation in result.snapshot.relations:
        source_label = _mcp_relation_label(relation)
        for foreign_key in relation.foreign_keys:
            relationship_label = _mcp_relationship_label(relation, foreign_key)
            all_relationship_labels.append(relationship_label)
            target_label = f"{foreign_key.target_schema}.{foreign_key.target_relation}"
            if (
                source_label in included_relation_set
                and target_label in included_relation_set
                and relationship_label not in omitted_relationships
            ):
                returned_relationship_labels.append(relationship_label)

    returned_column_count = sum(
        len(relation.columns)
        for relation in result.snapshot.relations
        if _mcp_relation_label(relation) in included_relation_set
    )
    omitted_relation_labels = tuple(
        label for label in all_relation_labels if label not in included_relation_set
    )
    omitted_relationship_labels = tuple(
        label for label in all_relationship_labels if label not in returned_relationship_labels
    )
    public_context = MCPSchemaContext(
        schema_version=context.schema_version,
        snapshot_fingerprint=result.fingerprint,
        text=context.text,
        max_chars=public_max_chars,
        included_relations=context.included_relations,
        relation_count=relation_count,
        returned_relation_count=len(included_relation_set),
        omitted_relation_count=len(omitted_relation_labels),
        column_count=column_count,
        returned_column_count=returned_column_count,
        omitted_column_count=column_count - returned_column_count,
        relationship_count=len(all_relationship_labels),
        returned_relationship_count=len(returned_relationship_labels),
        omitted_relationship_count=len(omitted_relationship_labels),
        omitted_relation_sample=omitted_relation_labels[:MCP_SCHEMA_OMISSION_SAMPLE_SIZE],
        omitted_relationship_sample=omitted_relationship_labels[:MCP_SCHEMA_OMISSION_SAMPLE_SIZE],
        truncated=bool(omitted_relation_labels or omitted_relationship_labels),
    )
    return MCPSchemaPayload(
        schema_version=result.schema_version,
        fingerprint=result.fingerprint,
        context=public_context,
    )


def _chatbi_result_error(result: ChatBIResult) -> MCPToolError | None:
    if result.execution_status is QueryLifecycleState.SUCCEEDED:
        return None
    category = result.error_category or ChatBIErrorCategory.EXECUTION_FAILED
    return _chatbi_error(ChatBIError(category, result.error_message or "ChatBI request failed"))


def create_mcp_server(
    application: MCPApplication,
    *,
    max_in_flight: int = MCP_DEFAULT_MAX_IN_FLIGHT,
) -> Server:
    """Create an official low-level MCP server around one application instance."""
    if not callable(getattr(application, "chatbi_ask", None)) or not callable(
        getattr(application, "chatbi_schema", None)
    ):
        raise TypeError("application must provide the ChatBI MCP methods")
    if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
        raise TypeError("max_in_flight must be an integer")
    if not 1 <= max_in_flight <= MCP_MAX_IN_FLIGHT_LIMIT:
        raise ValueError(f"max_in_flight must be between 1 and {MCP_MAX_IN_FLIGHT_LIMIT}")

    in_flight = asyncio.Semaphore(max_in_flight)

    async def run_bounded(operation: Callable[[], Awaitable[object]]) -> object:
        async with in_flight:
            return await operation()

    server = Server(
        "KnowledgeScope MCP",
        version=__version__,
        instructions=(
            "Only high-level ChatBI tools are exposed; SQL remains behind the application policy."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, object] | None) -> CallToolResult:
        supplied = arguments if arguments is not None else {}
        if name == "chatbi_ask":
            parsed = _parse_arguments(MCPAskArguments, name, supplied)
            if isinstance(parsed, MCPToolError):
                return _call_result(MCPAskResponse(status="error", error=parsed))
            try:
                result = await run_bounded(
                    lambda: application.chatbi_ask(parsed.datasource_id, parsed.question)  # type: ignore[attr-defined]
                )
                if not isinstance(result, ChatBIResult):
                    raise TypeError("ChatBI application returned an invalid result")
                result_error = _chatbi_result_error(result)
                response = (
                    MCPAskResponse(status="ok", result=result)
                    if result_error is None
                    else MCPAskResponse(status="error", error=result_error)
                )
            except asyncio.CancelledError:
                raise
            except ChatBIError as error:
                response = MCPAskResponse(status="error", error=_chatbi_error(error))
            except (TypeError, ValueError, ValidationError):
                response = MCPAskResponse(status="error", error=_internal_error())
            except Exception:
                response = MCPAskResponse(status="error", error=_internal_error())
            return _call_result(response)

        if name == "chatbi_schema":
            parsed = _parse_arguments(MCPSchemaArguments, name, supplied)
            if isinstance(parsed, MCPToolError):
                return _call_result(MCPSchemaResponse(status="error", error=parsed))
            try:
                discovery = await run_bounded(
                    lambda: application.chatbi_schema(parsed.datasource_id)  # type: ignore[attr-defined]
                )
                if not isinstance(discovery, SchemaDiscoveryResult):
                    raise TypeError("schema application returned an invalid result")
                response = MCPSchemaResponse(status="ok", result=_schema_payload(discovery))
            except asyncio.CancelledError:
                raise
            except ChatBIError as error:
                response = MCPSchemaResponse(status="error", error=_chatbi_error(error))
            except (TypeError, ValueError, ValidationError):
                response = MCPSchemaResponse(status="error", error=_internal_error())
            except Exception:
                response = MCPSchemaResponse(status="error", error=_internal_error())
            return _call_result(response)

        if name not in _TOOL_NAMES:
            return _call_result(
                MCPErrorResponse(
                    error=MCPToolError(category="unknown_tool", message="tool is not available")
                )
            )
        return _call_result(MCPErrorResponse(error=_internal_error()))

    return server


async def run_mcp_stdio(settings: Settings) -> None:
    """Run the local stdio transport and always close process-scoped services."""
    application = build_mcp_application(settings)
    try:
        server = create_mcp_server(application, max_in_flight=settings.mcp_max_in_flight)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await application.aclose()


__all__ = [
    "MCPApplication",
    "MCPAskArguments",
    "MCPSchemaArguments",
    "ProductionMCPApplication",
    "build_mcp_application",
    "create_mcp_server",
    "run_mcp_stdio",
]
