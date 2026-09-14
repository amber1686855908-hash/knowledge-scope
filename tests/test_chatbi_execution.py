from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from knowledge_scope.chatbi import (
    DataSource,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryPolicy,
    SchemaColumn,
    SchemaDiscoveryResult,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SQLCandidate,
    SQLDialect,
    SQLExecutionService,
)
from knowledge_scope.chatbi.credentials import ResolvedDatabaseCredentials
from knowledge_scope.chatbi.discovery import SchemaDiscoveryService
from knowledge_scope.chatbi.errors import ChatBIError, ChatBIErrorCategory
from knowledge_scope.chatbi.execution import redact_sql_literals
from knowledge_scope.chatbi.nl2sql_models import ValidatedSQL
from knowledge_scope.chatbi.postgres_schema import PostgresSchemaInspector
from knowledge_scope.chatbi.schema_models import build_semantic_schema_context
from knowledge_scope.chatbi.schemas import (
    QueryExecutionResult,
    QueryLifecycleState,
    QueryTruncationReason,
)
from knowledge_scope.chatbi.sql_validation import _validate_sql_candidate

DATASOURCE_ID = UUID("11111111-1111-4111-8111-111111111111")


def _data_source() -> DataSource:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return DataSource(
        id=DATASOURCE_ID,
        display_name="业务库",
        dialect=SQLDialect.POSTGRESQL,
        enabled=True,
        connection_ref="env:CHATBI_DEMO_DATABASE_URL",
        default_database=None,
        default_schema="public",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _snapshot(*relation_names: str) -> SchemaSnapshot:
    relations = tuple(
        SchemaRelation(
            schema_name="public",
            name=name,
            kind=SchemaObjectKind.TABLE,
            columns=(
                SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),
                SchemaColumn(name="amount", normalized_type="numeric", nullable=True, ordinal=2),
            ),
        )
        for name in relation_names
    )
    return SchemaSnapshot(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        database_name="business",
        schemas=("public",),
        relations=relations,
    )


def _candidate(snapshot: SchemaSnapshot, sql: str) -> SQLCandidate:
    return SQLCandidate(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        question="查询数据",
        sql=sql,
        context_fingerprint=snapshot.fingerprint,
        provider="test-provider",
        model="test-model",
    )


def _validated(
    sql: str = "SELECT * FROM public.sales",
    *,
    policy: QueryPolicy | None = None,
) -> tuple[ValidatedSQL, QueryPolicy]:
    selected_policy = policy or QueryPolicy()
    snapshot = _snapshot("sales")
    semantic_context = build_semantic_schema_context(snapshot, max_chars=10_000)
    candidate = _candidate(snapshot, sql)
    return (
        _validate_sql_candidate(
            candidate,
            schema_snapshot=snapshot,
            semantic_context=semantic_context,
            policy=selected_policy,
        ),
        selected_policy,
    )


class _FakeDiscovery:
    def __init__(self, snapshot: SchemaSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[DataSource] = []

    async def discover(
        self,
        data_source: DataSource,
        _policy: QueryPolicy,
        *,
        max_chars: int,
    ) -> SchemaDiscoveryResult:
        self.calls.append(data_source)
        context = build_semantic_schema_context(self.snapshot, max_chars=max_chars)
        return SchemaDiscoveryResult(
            snapshot=self.snapshot,
            fingerprint=self.snapshot.fingerprint,
            context=context,
        )


class _FakeDataSourceProvider:
    async def get(self, datasource_id: UUID) -> DataSource | None:
        return _data_source() if datasource_id == DATASOURCE_ID else None


class _FakeCredentialResolver:
    def __init__(self) -> None:
        self.refs: list[str] = []

    def resolve(self, connection_ref: str) -> ResolvedDatabaseCredentials:
        self.refs.append(connection_ref)
        return ResolvedDatabaseCredentials("postgresql://user:password@127.0.0.1/business")


class _FakeExecutionAdapter:
    def __init__(self) -> None:
        self.calls: list[ValidatedSQL] = []

    async def _execute_validated(
        self,
        _credentials: ResolvedDatabaseCredentials,
        *,
        datasource_id: UUID,
        query_id: UUID,
        validated: ValidatedSQL,
        policy: QueryPolicy,
    ) -> QueryExecutionResult:
        self.calls.append(validated)
        return QueryExecutionResult(
            query_id=query_id,
            datasource_id=datasource_id,
            state=QueryLifecycleState.SUCCEEDED,
            columns=[{"name": "value", "data_type": "integer", "ordinal": 0}],
            rows=[[1]],
            row_count=1,
            max_rows=policy.max_rows,
            truncated=False,
            duration_ms=0.1,
        )


def _service(
    *,
    snapshot: SchemaSnapshot | None = None,
    adapter: object | None = None,
    recorder: object | None = None,
) -> tuple[SQLExecutionService, _FakeDiscovery, _FakeExecutionAdapter, _FakeCredentialResolver]:
    discovery = _FakeDiscovery(snapshot or _snapshot("sales"))
    execution_adapter = adapter or _FakeExecutionAdapter()
    resolver = _FakeCredentialResolver()
    service = SQLExecutionService(
        NL2SQLService(
            None,
            schema_discovery=discovery,
            data_source_provider=_FakeDataSourceProvider(),
        ),
        resolver,
        execution_adapter,  # type: ignore[arg-type]
        audit_recorder=recorder,  # type: ignore[arg-type]
    )
    return service, discovery, execution_adapter, resolver


@pytest.mark.anyio
async def test_execution_revalidates_against_registered_snapshot_and_has_no_forged_paths() -> None:
    service, _discovery, adapter, _resolver = _service()
    forged_snapshot = _snapshot("invented")
    forged_candidate = _candidate(forged_snapshot, "SELECT * FROM public.invented")

    outcome = await service.execute_for_registered_data_source(
        DATASOURCE_ID,
        forged_candidate,
        policy=QueryPolicy(),
        max_chars=10_000,
    )

    assert outcome.result.state is QueryLifecycleState.REJECTED
    assert outcome.result.error_category is ChatBIErrorCategory.POLICY_VIOLATION
    assert adapter.calls == []
    with pytest.raises(TypeError):
        await service.execute_for_registered_data_source(  # type: ignore[arg-type]
            DATASOURCE_ID,
            ValidatedSQL,  # type: ignore[arg-type]
            policy=QueryPolicy(),
            max_chars=10_000,
        )


@pytest.mark.anyio
async def test_execution_success_uses_registered_validation_and_audit_lifecycle() -> None:
    service, discovery, adapter, resolver = _service()
    candidate = _candidate(_snapshot("sales"), "SELECT * FROM public.sales")

    outcome = await service.execute_for_registered_data_source(
        DATASOURCE_ID,
        candidate,
        policy=QueryPolicy(),
        max_chars=10_000,
    )

    assert outcome.result.state is QueryLifecycleState.SUCCEEDED
    assert outcome.result.query_id.int != 0
    assert [record.state for record in outcome.audit] == [
        QueryLifecycleState.CREATED,
        QueryLifecycleState.VALIDATED,
        QueryLifecycleState.EXECUTING,
        QueryLifecycleState.SUCCEEDED,
    ]
    assert len({record.audit_id for record in outcome.audit}) == 4
    assert discovery.calls == [_data_source()]
    assert resolver.refs == [_data_source().connection_ref]
    assert len(adapter.calls) == 1


@pytest.mark.anyio
async def test_execution_audit_uses_redacted_sql_literals() -> None:
    service, _discovery, _adapter, _resolver = _service()
    outcome = await service.execute_sql_for_registered_data_source(
        DATASOURCE_ID,
        "SELECT * FROM public.sales WHERE amount = 987654 AND id = 'secret-region'",
        policy=QueryPolicy(),
        max_chars=10_000,
    )

    assert outcome.result.state is QueryLifecycleState.SUCCEEDED
    for record in outcome.audit:
        encoded = record.model_dump_json()
        assert "987654" not in encoded
        assert "secret-region" not in encoded
    assert '"public"."sales"' in (outcome.audit[1].normalized_sql or "")


def test_redact_sql_literals_preserves_structure_and_fails_safe() -> None:
    sql = (
        "WITH filtered AS (SELECT * FROM public.sales WHERE note = 'secret@example.com') "
        "SELECT s.region, SUM(s.amount) FROM filtered AS s "
        "WHERE s.amount > 42 AND s.active = TRUE AND s.closed_at IS NOT NULL "
        "AND s.sold_on = DATE '2026-01-01' GROUP BY s.region "
        "UNION SELECT 'other-secret', 99"
    )

    redacted = redact_sql_literals(sql)

    assert "secret@example.com" not in redacted
    assert "other-secret" not in redacted
    assert "42" not in redacted
    assert "99" not in redacted
    assert "TRUE" not in redacted
    assert "2026-01-01" not in redacted
    assert "closed_at" in redacted
    assert "NULL" in redacted
    assert "public.sales" in redacted
    assert "GROUP BY" in redacted
    assert "UNION" in redacted
    assert redact_sql_literals("SELECT 'unterminated") == "<sql-redaction-unavailable>"


@pytest.mark.anyio
async def test_raw_sql_path_builds_candidate_after_discovery_and_rejects_write() -> None:
    service, _discovery, adapter, _resolver = _service()

    success = await service.execute_sql_for_registered_data_source(
        DATASOURCE_ID,
        "SELECT * FROM public.sales",
        policy=QueryPolicy(),
        max_chars=10_000,
    )
    rejected = await service.execute_sql_for_registered_data_source(
        DATASOURCE_ID,
        "DELETE FROM public.sales",
        policy=QueryPolicy(),
        max_chars=10_000,
    )

    assert success.result.state is QueryLifecycleState.SUCCEEDED
    assert rejected.result.state is QueryLifecycleState.REJECTED
    assert rejected.result.error_category in {
        ChatBIErrorCategory.UNSAFE_QUERY,
        ChatBIErrorCategory.POLICY_VIOLATION,
    }
    assert len(adapter.calls) == 1


class _FakeType:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAttribute:
    def __init__(self, name: str, data_type: str) -> None:
        self.name = name
        self.type = _FakeType(data_type)


class _FakeCursor:
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self._records = iter(records)

    def __aiter__(self) -> _FakeCursor:
        return self

    async def __anext__(self) -> Mapping[str, object]:
        try:
            return next(self._records)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeStatement:
    def __init__(
        self,
        attributes: Sequence[object],
        records: Sequence[Mapping[str, object]],
    ) -> None:
        self._attributes = tuple(attributes)
        self._records = records
        self.prefetch: int | None = None

    def get_attributes(self) -> Sequence[object]:
        return self._attributes

    def cursor(self, *, prefetch: int = 50) -> AsyncIterator[Mapping[str, object]]:
        self.prefetch = prefetch
        return _FakeCursor(self._records)


class _FakeTransaction(AbstractAsyncContextManager[object]):
    def __init__(self, connection: _FakeConnection, readonly: bool) -> None:
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self) -> _FakeTransaction:
        self.connection.transaction_readonly = self.readonly
        return self

    async def __aexit__(self, exc_type: object, *_args: object) -> None:
        self.connection.transaction_error = exc_type


class _FakeConnection:
    def __init__(
        self,
        attributes: Sequence[object],
        records: Sequence[Mapping[str, object]],
        *,
        prepare_error: BaseException | None = None,
    ) -> None:
        self.attributes = attributes
        self.records = records
        self.prepare_error = prepare_error
        self.queries: list[str] = []
        self.transaction_readonly = False
        self.transaction_error: object | None = None
        self.closed = False
        self.prepared_query: str | None = None

    def transaction(self, *, readonly: bool = False) -> _FakeTransaction:
        return _FakeTransaction(self, readonly)

    async def execute(self, query: str) -> str:
        self.queries.append(query)
        return "OK"

    async def prepare(self, query: str) -> _FakeStatement:
        self.prepared_query = query
        if self.prepare_error is not None:
            raise self.prepare_error
        return _FakeStatement(self.attributes, self.records)

    async def close(self) -> None:
        self.closed = True


def _adapter_for(connection: _FakeConnection) -> PostgresExecutionAdapter:
    async def factory(
        _credentials: ResolvedDatabaseCredentials,
        _connection_timeout: float,
        _statement_timeout: int,
    ) -> _FakeConnection:
        return connection

    return PostgresExecutionAdapter(connection_factory=factory)  # type: ignore[arg-type]


def test_safe_search_path_quotes_policy_owned_schema_names() -> None:
    from knowledge_scope.chatbi import build_safe_search_path_statement

    policy = QueryPolicy(allowed_schemas=('tenant"x',))

    assert build_safe_search_path_statement(policy) == (
        'SET LOCAL search_path TO "pg_catalog", "tenant""x", "pg_temp"'
    )


@pytest.mark.anyio
async def test_non_success_adapter_result_is_not_a_successful_lifecycle() -> None:
    class _BadAdapter(_FakeExecutionAdapter):
        async def _execute_validated(
            self,
            _credentials: ResolvedDatabaseCredentials,
            *,
            datasource_id: UUID,
            query_id: UUID,
            validated: ValidatedSQL,
            policy: QueryPolicy,
        ) -> QueryExecutionResult:
            return QueryExecutionResult(
                query_id=query_id,
                datasource_id=datasource_id,
                state=QueryLifecycleState.FAILED,
                columns=[],
                rows=[],
                row_count=0,
                max_rows=policy.max_rows,
                truncated=False,
                duration_ms=0.1,
                error_category=ChatBIErrorCategory.EXECUTION_FAILED,
                error_message="adapter failure",
            )

    service, _discovery, _adapter, _resolver = _service(adapter=_BadAdapter())
    outcome = await service.execute_for_registered_data_source(
        DATASOURCE_ID,
        _candidate(_snapshot("sales"), "SELECT * FROM public.sales"),
        policy=QueryPolicy(),
        max_chars=10_000,
    )

    assert outcome.result.state is QueryLifecycleState.FAILED
    assert outcome.result.error_category is ChatBIErrorCategory.EXECUTION_FAILED
    assert outcome.audit[-1].state is QueryLifecycleState.FAILED


@pytest.mark.anyio
async def test_postgres_adapter_enforces_readonly_search_path_and_normalizes_values() -> None:
    values: Mapping[str, object] = {
        "text_value": "华东",
        "integer_value": 3,
        "decimal_value": Decimal("12.30"),
        "uuid_value": UUID("11111111-1111-4111-8111-111111111111"),
        "date_value": date(2026, 1, 1),
        "time_value": time(8, 30),
        "timestamp_value": datetime(2026, 1, 1, 8, 30, tzinfo=UTC),
        "json_value": {"ok": True, "items": [1, None]},
        "array_value": ["a", 2],
        "null_value": None,
    }
    attributes = tuple(
        _FakeAttribute(name, "jsonb" if name == "json_value" else "text") for name in values
    )
    connection = _FakeConnection(attributes, [values])
    validated, policy = _validated()
    result = await _adapter_for(connection)._execute_validated(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        query_id=uuid4(),
        validated=validated,
        policy=policy,
    )

    assert result.state is QueryLifecycleState.SUCCEEDED
    assert result.row_count == 1
    assert result.rows[0][2] == "12.30"
    assert result.rows[0][3] == "11111111-1111-4111-8111-111111111111"
    assert result.rows[0][4:7] == ["2026-01-01", "08:30:00", "2026-01-01T08:30:00+00:00"]
    assert result.rows[0][7] == {"ok": True, "items": [1, None]}
    assert connection.transaction_readonly is True
    assert connection.queries[0] == ('SET LOCAL search_path TO "pg_catalog", "public", "pg_temp"')
    assert connection.queries[1] == "SET LOCAL statement_timeout = 30000"
    assert connection.prepared_query == validated.normalized_sql
    assert connection.closed is True


@pytest.mark.anyio
async def test_postgres_adapter_applies_row_bound_and_reports_truncation() -> None:
    policy = QueryPolicy(max_rows=2)
    validated, _ = _validated(policy=policy)
    records = [{"id": 1, "amount": 1}, {"id": 2, "amount": 2}, {"id": 3, "amount": 3}]
    connection = _FakeConnection(
        [_FakeAttribute("id", "int4"), _FakeAttribute("amount", "numeric")],
        records,
    )

    result = await _adapter_for(connection)._execute_validated(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        query_id=uuid4(),
        validated=validated,
        policy=policy,
    )

    assert result.row_count == 2
    assert result.truncated is True
    assert result.truncation_reason is QueryTruncationReason.ROW_LIMIT
    assert result.max_rows == 2
    assert result.rows == [[1, 1], [2, 2]]


@pytest.mark.anyio
async def test_postgres_adapter_accepts_exact_payload_boundary_and_reports_byte_truncation() -> (
    None
):
    policy = QueryPolicy(max_rows=10, max_result_bytes=7, max_cell_bytes=10)
    validated, _ = _validated(policy=policy)
    connection = _FakeConnection(
        [_FakeAttribute("value", "text")],
        [{"value": "a"}, {"value": "b"}],
    )

    result = await _adapter_for(connection)._execute_validated(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        query_id=uuid4(),
        validated=validated,
        policy=policy,
    )

    assert result.rows == [["a"]]
    assert result.row_count == 1
    assert result.truncated is True
    assert result.truncation_reason is QueryTruncationReason.PAYLOAD_BYTES

    exact_connection = _FakeConnection(
        [_FakeAttribute("value", "text")],
        [{"value": "a"}],
    )
    exact = await _adapter_for(exact_connection)._execute_validated(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        query_id=uuid4(),
        validated=validated,
        policy=policy,
    )
    assert exact.rows == [["a"]]
    assert exact.truncated is False
    assert exact.truncation_reason is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("value", "policy"),
    [
        ("abcd", QueryPolicy(max_cell_bytes=5)),
        ({"text": "too-large"}, QueryPolicy(max_cell_bytes=10)),
        ({"nested": {"value": 1}}, QueryPolicy(max_nested_value_depth=1)),
        ([[1]], QueryPolicy(max_nested_value_depth=1)),
        ([1, 2, 3], QueryPolicy(max_collection_items=2)),
    ],
)
async def test_postgres_adapter_rejects_oversized_or_deep_values(
    value: object,
    policy: QueryPolicy,
) -> None:
    validated, _ = _validated(policy=policy)
    connection = _FakeConnection(
        [_FakeAttribute("value", "jsonb")],
        [{"value": value}],
    )

    with pytest.raises(ChatBIError) as error:
        await _adapter_for(connection)._execute_validated(
            ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
            datasource_id=DATASOURCE_ID,
            query_id=uuid4(),
            validated=validated,
            policy=policy,
        )

    assert error.value.category is ChatBIErrorCategory.RESULT_SIZE_EXCEEDED
    assert connection.closed is True
    assert connection.transaction_error is not None


@pytest.mark.anyio
async def test_postgres_adapter_returns_empty_result_without_truncation() -> None:
    policy = QueryPolicy(max_result_bytes=2)
    validated, _ = _validated(policy=policy)
    connection = _FakeConnection(
        [_FakeAttribute("value", "text")],
        [],
    )

    result = await _adapter_for(connection)._execute_validated(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        query_id=uuid4(),
        validated=validated,
        policy=policy,
    )

    assert result.rows == []
    assert result.truncated is False
    assert result.truncation_reason is None


@pytest.mark.anyio
async def test_postgres_adapter_classifies_timeout_and_closes_connection() -> None:
    connection = _FakeConnection([], [], prepare_error=TimeoutError())
    validated, policy = _validated()

    with pytest.raises(ChatBIError) as error:
        await _adapter_for(connection)._execute_validated(
            ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
            datasource_id=DATASOURCE_ID,
            query_id=uuid4(),
            validated=validated,
            policy=policy,
        )

    assert error.value.category is ChatBIErrorCategory.EXECUTION_TIMEOUT
    assert connection.closed is True
    assert connection.transaction_error is not None


@pytest.mark.anyio
async def test_cancellation_is_audited_and_propagates() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.records: list[object] = []

        def record(self, record: object) -> None:
            self.records.append(record)

    class _CancellableAdapter(_FakeExecutionAdapter):
        async def _execute_validated(
            self, *_args: object, **_kwargs: object
        ) -> QueryExecutionResult:
            raise asyncio.CancelledError

    recorder = _Recorder()
    service, _discovery, _adapter, _resolver = _service(
        adapter=_CancellableAdapter(),
        recorder=recorder,
    )
    candidate = _candidate(_snapshot("sales"), "SELECT * FROM public.sales")

    with pytest.raises(asyncio.CancelledError):
        await service.execute_for_registered_data_source(
            DATASOURCE_ID,
            candidate,
            policy=QueryPolicy(),
            max_chars=10_000,
        )

    assert recorder.records[-1].state is QueryLifecycleState.CANCELLED  # type: ignore[union-attr]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EXECUTION_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EXECUTION_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_execution_is_read_only_and_schema_bound(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    os.environ["CHATBI_EXECUTION_TEST_DATABASE_URL"] = postgres_test_database
    try:
        source_data = _data_source().model_dump()
        source_data["connection_ref"] = "env:CHATBI_EXECUTION_TEST_DATABASE_URL"
        source_data["default_database"] = None
        source_data["default_schema"] = "chatbi_demo"
        source = DataSource.model_validate(source_data)
        snapshot_policy = QueryPolicy(allowed_schemas=("chatbi_demo",), max_rows=10)
        discovery_service = SchemaDiscoveryService(
            EnvironmentCredentialResolver(),
            PostgresSchemaInspector(),
        )

        class _RealProvider:
            async def get(self, _datasource_id: UUID) -> DataSource:
                return source

        service = SQLExecutionService(
            NL2SQLService(
                None,
                schema_discovery=discovery_service,
                data_source_provider=_RealProvider(),
            ),
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(),
        )
        aggregate = await service.execute_sql_for_registered_data_source(
            DATASOURCE_ID,
            "SELECT c.customer_name, SUM(s.amount) AS total_amount "
            "FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
            "GROUP BY c.customer_name ORDER BY c.customer_name",
            policy=snapshot_policy,
            max_chars=20_000,
        )
        set_operation = await service.execute_sql_for_registered_data_source(
            DATASOURCE_ID,
            "WITH regions AS (SELECT region FROM chatbi_demo.sales) "
            "SELECT region FROM regions UNION SELECT region FROM regions",
            policy=snapshot_policy,
            max_chars=20_000,
        )
        rejected = await service.execute_sql_for_registered_data_source(
            DATASOURCE_ID,
            "DELETE FROM chatbi_demo.sales",
            policy=snapshot_policy,
            max_chars=20_000,
        )

        assert aggregate.result.state is QueryLifecycleState.SUCCEEDED
        assert aggregate.result.row_count == 2
        assert set_operation.result.state is QueryLifecycleState.SUCCEEDED
        assert set_operation.result.row_count == 2
        assert rejected.result.state is QueryLifecycleState.REJECTED
        assert rejected.result.error_category is not None
        assert aggregate.audit[1].normalized_sql is not None
        assert "chatbi_demo" in aggregate.audit[1].normalized_sql
    finally:
        os.environ.pop("CHATBI_EXECUTION_TEST_DATABASE_URL", None)
