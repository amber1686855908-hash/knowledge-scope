from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from knowledge_scope.chatbi import (
    DataSource,
    EnvironmentCredentialResolver,
    QueryPolicy,
    SchemaColumn,
    SchemaContextBudgetError,
    SchemaForeignKey,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SchemaUniqueConstraint,
    SemanticSchemaContext,
    SQLDialect,
    build_semantic_schema_context,
)
from knowledge_scope.chatbi.credentials import ResolvedDatabaseCredentials
from knowledge_scope.chatbi.discovery import SchemaDiscoveryService
from knowledge_scope.chatbi.errors import ChatBIError, ChatBIErrorCategory
from knowledge_scope.chatbi.postgres_schema import (
    PostgresSchemaInspector,
    is_system_schema,
)

DATASOURCE_ID = UUID("11111111-1111-4111-8111-111111111111")


def _data_source(
    *,
    enabled: bool = True,
    default_database: str | None = None,
) -> DataSource:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return DataSource(
        id=DATASOURCE_ID,
        display_name="销售数据库",
        dialect=SQLDialect.POSTGRESQL,
        enabled=enabled,
        connection_ref="env:CHATBI_DEMO_DATABASE_URL",
        default_database=default_database,
        default_schema="public",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _relation(
    name: str,
    *,
    ordinal: int = 1,
    comment: str | None = None,
) -> SchemaRelation:
    return SchemaRelation(
        schema_name="public",
        name=name,
        kind=SchemaObjectKind.TABLE,
        comment=comment,
        columns=(
            SchemaColumn(
                name="id",
                normalized_type="integer",
                nullable=False,
                ordinal=ordinal,
            ),
        ),
        primary_key=("id",),
    )


def _snapshot(*relations: SchemaRelation) -> SchemaSnapshot:
    return SchemaSnapshot(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        database_name="business",
        schemas=("public",),
        relations=relations,
    )


def test_schema_snapshot_is_deterministic_and_fingerprint_tracks_material_changes() -> None:
    first = SchemaRelation(
        schema_name="public",
        name="sales",
        kind=SchemaObjectKind.TABLE,
        columns=(
            SchemaColumn(
                name="amount", normalized_type="numeric(12, 2)", nullable=False, ordinal=2
            ),
            SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),
        ),
        primary_key=("id",),
    )
    second = SchemaRelation(
        schema_name="public",
        name="customers",
        kind=SchemaObjectKind.TABLE,
        columns=(SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),),
        primary_key=("id",),
    )
    unordered = _snapshot(first, second)
    ordered = _snapshot(second, first)

    assert unordered.to_deterministic_json() == ordered.to_deterministic_json()
    assert unordered.fingerprint == ordered.fingerprint
    assert "created_at" not in unordered.to_deterministic_json()

    changed = _snapshot(
        SchemaRelation(
            schema_name="public",
            name="sales",
            kind=SchemaObjectKind.TABLE,
            comment="业务销售表",
            columns=first.columns,
            primary_key=("id",),
        ),
        second,
    )
    assert changed.fingerprint != unordered.fingerprint

    with pytest.raises(ValidationError):
        SchemaSnapshot.model_validate({**unordered.model_dump(mode="json"), "unexpected": True})


def test_schema_models_validate_lineage_and_constraint_shapes() -> None:
    with pytest.raises(ValidationError):
        SchemaForeignKey(
            constraint_name="sales_customer_fkey",
            source_columns=("customer_id",),
            target_schema="public",
            target_relation="customers",
            target_columns=("id", "extra"),
        )

    with pytest.raises(ValidationError):
        SchemaRelation(
            schema_name="public",
            name="sales",
            kind=SchemaObjectKind.TABLE,
            columns=(
                SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),
            ),
            primary_key=("missing",),
        )

    relation = SchemaRelation(
        schema_name="public",
        name="sales",
        kind=SchemaObjectKind.TABLE,
        columns=(
            SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),
            SchemaColumn(name="customer_id", normalized_type="integer", nullable=False, ordinal=2),
        ),
        primary_key=("id",),
        foreign_keys=(
            SchemaForeignKey(
                constraint_name="sales_customer_fkey",
                source_columns=("customer_id",),
                target_schema="public",
                target_relation="customers",
                target_columns=("id",),
            ),
        ),
        unique_constraints=(
            SchemaUniqueConstraint(constraint_name="sales_id_key", columns=("id",)),
        ),
    )
    assert relation.foreign_keys[0].target_relation == "customers"
    assert relation.primary_key == ("id",)


def test_schema_context_omits_whole_relations_and_exposes_metadata() -> None:
    snapshot = _snapshot(
        _relation("large_table", comment="x" * 1_000),
        _relation("small_table"),
    )

    context = build_semantic_schema_context(snapshot, max_chars=180)

    assert len(context.text) <= 180
    assert context.truncated is True
    assert set(context.included_relations).isdisjoint(context.omitted_relations)
    assert "public.large_table" in context.omitted_relations
    assert context.snapshot_fingerprint == snapshot.fingerprint
    assert (
        SemanticSchemaContext.model_validate(context.model_dump()).model_dump()
        == context.model_dump()
    )


def _foreign_key_snapshot() -> SchemaSnapshot:
    source = SchemaRelation(
        schema_name="public",
        name="a_orders",
        kind=SchemaObjectKind.TABLE,
        columns=(
            SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),
            SchemaColumn(
                name="customer_id",
                normalized_type="integer",
                nullable=False,
                ordinal=2,
            ),
        ),
        foreign_keys=(
            SchemaForeignKey(
                constraint_name="orders_customer_fkey",
                source_columns=("customer_id",),
                target_schema="public",
                target_relation="z_customers",
                target_columns=("id",),
            ),
        ),
    )
    return _snapshot(source, _relation("z_customers"))


def _context_with_relations(
    snapshot: SchemaSnapshot,
    included_relations: tuple[str, ...],
) -> SemanticSchemaContext:
    for max_chars in range(1, 2_000):
        try:
            context = build_semantic_schema_context(snapshot, max_chars=max_chars)
        except SchemaContextBudgetError:
            continue
        if context.included_relations == included_relations:
            return context
    raise AssertionError(f"could not find a budget for {included_relations}")


def test_schema_context_omits_fk_when_target_table_is_omitted() -> None:
    snapshot = _foreign_key_snapshot()

    context = _context_with_relations(snapshot, ("public.a_orders",))

    assert context.omitted_relations == ("public.z_customers",)
    assert context.omitted_relationships == (
        "public.a_orders.orders_customer_fkey->public.z_customers",
    )
    assert "Relationships:" not in context.text
    assert context.truncated is True


def test_schema_context_omits_fk_when_referencing_table_is_omitted() -> None:
    snapshot = _foreign_key_snapshot()

    context = _context_with_relations(snapshot, ("public.z_customers",))

    assert context.omitted_relations == ("public.a_orders",)
    assert context.omitted_relationships == (
        "public.a_orders.orders_customer_fkey->public.z_customers",
    )
    assert "Relationships:" not in context.text


def test_schema_context_includes_fk_only_for_retained_endpoints_and_is_deterministic() -> None:
    snapshot = _foreign_key_snapshot()

    first = build_semantic_schema_context(snapshot, max_chars=2_000)
    second = build_semantic_schema_context(snapshot, max_chars=2_000)

    assert first.included_relations == ("public.a_orders", "public.z_customers")
    assert first.omitted_relationships == ()
    assert "Relationships:" in first.text
    assert "Foreign key orders_customer_fkey" in first.text
    assert first.model_dump() == second.model_dump()


def test_schema_context_budget_boundaries_are_explicit_and_whole_block() -> None:
    snapshot = _snapshot(_relation("sales"))
    header = "Database: business\nDialect: postgresql\nSchemas: public"

    with pytest.raises(SchemaContextBudgetError):
        build_semantic_schema_context(snapshot, max_chars=len(header) - 1)

    minimum = build_semantic_schema_context(_snapshot(), max_chars=len(header))
    assert minimum.text == header
    assert minimum.truncated is False
    assert minimum.omitted_relations == ()

    first_table = build_semantic_schema_context(snapshot, max_chars=2_000)
    assert first_table.included_relations == ("public.sales",)
    assert len(first_table.text) <= first_table.max_chars

    no_table = build_semantic_schema_context(snapshot, max_chars=len(header))
    assert no_table.text == header
    assert no_table.included_relations == ()
    assert no_table.omitted_relations == ("public.sales",)
    assert no_table.truncated is True

    multiple = build_semantic_schema_context(
        _snapshot(_relation("alpha"), _relation("beta")),
        max_chars=len(header),
    )
    assert multiple.omitted_relations == ("public.alpha", "public.beta")
    assert multiple.truncated is True
    assert len(multiple.text) <= multiple.max_chars


@pytest.mark.anyio
async def test_discovery_service_reports_too_small_context_budget_safely() -> None:
    service = SchemaDiscoveryService(_FakeResolver(), _FakeAdapter(_snapshot(_relation("sales"))))
    header = "Database: business\nDialect: postgresql\nSchemas: public"

    with pytest.raises(ChatBIError) as error:
        await service.discover(_data_source(), QueryPolicy(), max_chars=len(header) - 1)

    assert error.value.category is ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED
    assert "too small" in error.value.safe_message


def test_credential_resolution_supports_env_and_redacts_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "do-not-print-this-password"
    monkeypatch.setenv(
        "CHATBI_DEMO_DATABASE_URL",
        f"postgresql+asyncpg://user:{secret}@127.0.0.1:5433/business",
    )

    credentials = EnvironmentCredentialResolver().resolve("env:CHATBI_DEMO_DATABASE_URL")

    assert secret not in repr(credentials)
    assert secret not in str(credentials)
    assert credentials.connection_url.startswith("postgresql://")

    monkeypatch.delenv("CHATBI_DEMO_DATABASE_URL")
    with pytest.raises(ChatBIError, match="missing") as error:
        EnvironmentCredentialResolver().resolve("env:CHATBI_DEMO_DATABASE_URL")
    assert error.value.category is ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED


def test_credential_resolution_requires_explicit_secret_backend() -> None:
    with pytest.raises(ChatBIError, match="configured secret backend"):
        EnvironmentCredentialResolver().resolve("secret:CHATBI_DEMO_DATABASE_URL")

    class _SecretBackend:
        def resolve(self, _name: str) -> str:
            return "postgresql://user:password@127.0.0.1:5432/business"

    credentials = EnvironmentCredentialResolver(secret_resolver=_SecretBackend()).resolve(
        "secret:CHATBI_DEMO_DATABASE_URL"
    )
    assert credentials.connection_url.startswith("postgresql://")

    class _InvalidSecretBackend:
        def resolve(self, _name: str) -> str:
            return "postgresqlfoo://user:password@localhost/business"

    with pytest.raises(ChatBIError, match="must use PostgreSQL"):
        EnvironmentCredentialResolver(secret_resolver=_InvalidSecretBackend()).resolve(
            "secret:CHATBI_DEMO_DATABASE_URL"
        )


def test_schema_snapshot_rejects_duplicate_schema_names() -> None:
    with pytest.raises(ValidationError, match="schemas must be unique"):
        SchemaSnapshot(
            datasource_id=DATASOURCE_ID,
            dialect=SQLDialect.POSTGRESQL,
            database_name="business",
            schemas=("public", "public"),
            relations=(),
        )


class _FakeTransaction:
    def __init__(self, connection: _FakeConnection, readonly: bool) -> None:
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self) -> object:
        self.connection.transaction_readonly = self.readonly
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.transaction_readonly = False
        self.closed = False

    def transaction(self, *, readonly: bool = False) -> _FakeTransaction:
        return _FakeTransaction(self, readonly)

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        self.queries.append((query, args))
        if "current_database" in query:
            return [{"database_name": "business"}]
        if "obj_description" in query:
            return [
                {
                    "schema_name": "public",
                    "relation_name": "customers",
                    "relation_kind": "table",
                    "comment": "客户表",
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales",
                    "relation_kind": "table",
                    "comment": "销售表",
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales_view",
                    "relation_kind": "view",
                    "comment": "销售视图",
                },
                {
                    "schema_name": "pg_catalog",
                    "relation_name": "pg_class",
                    "relation_kind": "table",
                    "comment": None,
                },
                {
                    "schema_name": "pg_toast_123",
                    "relation_name": "pg_toast_12345",
                    "relation_kind": "table",
                    "comment": None,
                },
                {
                    "schema_name": "pg_temp_7",
                    "relation_name": "temporary_relation",
                    "relation_kind": "table",
                    "comment": None,
                },
            ]
        if "pg_attribute" in query:
            return [
                {
                    "schema_name": "public",
                    "relation_name": "customers",
                    "ordinal": 1,
                    "column_name": "id",
                    "normalized_type": "integer",
                    "nullable": False,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales",
                    "ordinal": 1,
                    "column_name": "id",
                    "normalized_type": "integer",
                    "nullable": False,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales",
                    "ordinal": 2,
                    "column_name": "customer_id",
                    "normalized_type": "integer",
                    "nullable": False,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales_view",
                    "ordinal": 1,
                    "column_name": "id",
                    "normalized_type": "integer",
                    "nullable": True,
                    "comment": None,
                },
            ]
        if "pg_constraint" in query:
            return [
                {
                    "schema_name": "public",
                    "relation_name": "customers",
                    "constraint_name": "customers_pkey",
                    "constraint_type": "p",
                    "source_attnums": [1],
                    "target_schema": None,
                    "target_relation": None,
                    "target_attnums": None,
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales",
                    "constraint_name": "sales_pkey",
                    "constraint_type": "p",
                    "source_attnums": [1],
                    "target_schema": None,
                    "target_relation": None,
                    "target_attnums": None,
                },
                {
                    "schema_name": "public",
                    "relation_name": "sales",
                    "constraint_name": "sales_customer_fkey",
                    "constraint_type": "f",
                    "source_attnums": [2],
                    "target_schema": "public",
                    "target_relation": "customers",
                    "target_attnums": [1],
                },
            ]
        raise AssertionError(f"unexpected metadata query: {query}")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_postgres_adapter_is_read_only_allowlisted_and_metadata_only() -> None:
    connection = _FakeConnection()

    async def factory(
        _credentials: ResolvedDatabaseCredentials,
        _connection_timeout: float,
        _statement_timeout: int,
    ) -> _FakeConnection:
        return connection

    snapshot = await PostgresSchemaInspector(connection_factory=factory).discover(
        ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        allowed_schemas=(
            "information_schema",
            "pg_catalog",
            "pg_toast",
            "pg_toast_123",
            "pg_temp",
            "pg_temp_7",
            "public",
        ),
        allow_views=True,
    )

    assert connection.transaction_readonly is True
    assert connection.closed is True
    assert {relation.name for relation in snapshot.relations} == {
        "customers",
        "sales",
        "sales_view",
    }
    sales = next(relation for relation in snapshot.relations if relation.name == "sales")
    assert sales.foreign_keys[0].source_columns == ("customer_id",)
    assert sales.foreign_keys[0].target_columns == ("id",)
    assert all("SELECT *" not in query.upper() for query, _ in connection.queries)
    assert all("INSERT" not in query.upper() for query, _ in connection.queries)
    assert all("chatbi_demo.sales" not in query for query, _ in connection.queries)
    schema_arguments = [args[0] for query, args in connection.queries if "ANY($1::text[])" in query]
    assert schema_arguments == [["public"], ["public"], ["public"]]


@pytest.mark.parametrize(
    ("schema_name", "expected"),
    (
        ("pg_catalog", True),
        ("PG_CATALOG", True),
        ("information_schema", True),
        ("pg_toast", True),
        ("pg_toast_123", True),
        ("pg_temp_7", True),
        ("public", False),
        ("pg_user", False),
    ),
)
def test_postgres_system_schema_detection_is_explicit(
    schema_name: str,
    expected: bool,
) -> None:
    assert is_system_schema(schema_name) is expected


@pytest.mark.anyio
async def test_postgres_schema_allowlist_rejects_non_string_values_before_connecting() -> None:
    connection = _FakeConnection()

    async def factory(
        _credentials: ResolvedDatabaseCredentials,
        _connection_timeout: float,
        _statement_timeout: int,
    ) -> _FakeConnection:
        return connection

    with pytest.raises(ChatBIError, match="only string names"):
        await PostgresSchemaInspector(connection_factory=factory).discover(
            ResolvedDatabaseCredentials("postgresql://user:password@localhost/business"),
            datasource_id=DATASOURCE_ID,
            dialect=SQLDialect.POSTGRESQL,
            allowed_schemas=("public", 1),  # type: ignore[arg-type]
            allow_views=True,
        )

    assert connection.queries == []


class _FakeResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, _connection_ref: str) -> ResolvedDatabaseCredentials:
        self.calls += 1
        return ResolvedDatabaseCredentials("postgresql://localhost/business")


class _FakeAdapter:
    def __init__(self, snapshot: SchemaSnapshot) -> None:
        self.snapshot = snapshot

    async def discover(self, *_args: Any, **_kwargs: Any) -> SchemaSnapshot:
        return self.snapshot


@pytest.mark.anyio
async def test_discovery_service_checks_enabled_state_and_returns_safe_result() -> None:
    resolver = _FakeResolver()
    service = SchemaDiscoveryService(resolver, _FakeAdapter(_snapshot(_relation("sales"))))

    result = await service.discover(_data_source(), QueryPolicy(), max_chars=2_000)

    assert result.fingerprint == result.snapshot.fingerprint
    assert result.context.snapshot_fingerprint == result.fingerprint
    assert "connection_ref" not in result.model_dump()
    assert resolver.calls == 1

    disabled_resolver = _FakeResolver()
    disabled_service = SchemaDiscoveryService(
        disabled_resolver,
        _FakeAdapter(_snapshot(_relation("sales"))),
    )
    with pytest.raises(ChatBIError) as error:
        await disabled_service.discover(
            _data_source(enabled=False),
            QueryPolicy(),
            max_chars=2_000,
        )
    assert error.value.category is ChatBIErrorCategory.DATASOURCE_DISABLED
    assert disabled_resolver.calls == 0


@pytest.mark.anyio
async def test_discovery_service_rejects_inconsistent_adapter_result() -> None:
    other_snapshot = SchemaSnapshot(
        datasource_id=uuid4(),
        dialect=SQLDialect.POSTGRESQL,
        database_name="business",
        schemas=("public",),
        relations=(_relation("sales"),),
    )
    service = SchemaDiscoveryService(_FakeResolver(), _FakeAdapter(other_snapshot))

    with pytest.raises(ChatBIError, match="inconsistent datasource identity"):
        await service.discover(_data_source(), QueryPolicy(), max_chars=2_000)


@pytest.mark.anyio
async def test_discovery_service_rejects_out_of_scope_adapter_result() -> None:
    out_of_scope_relation = SchemaRelation(
        schema_name="private",
        name="sales",
        kind=SchemaObjectKind.TABLE,
        columns=(SchemaColumn(name="id", normalized_type="integer", nullable=False, ordinal=1),),
    )
    out_of_scope_snapshot = SchemaSnapshot(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        database_name="business",
        schemas=("private",),
        relations=(out_of_scope_relation,),
    )
    service = SchemaDiscoveryService(_FakeResolver(), _FakeAdapter(out_of_scope_snapshot))

    with pytest.raises(ChatBIError, match="outside the configured schema allow-list"):
        await service.discover(_data_source(), QueryPolicy(), max_chars=2_000)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_SCHEMA_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_SCHEMA_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_schema_discovery_uses_demo_schema(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        # asyncpg prepares one statement at a time and rejects a script with
        # multiple commands.  The fixture is deliberately simple and contains
        # no semicolons inside literals, so execute its statements explicitly.
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    os.environ["CHATBI_SCHEMA_TEST_DATABASE_URL"] = postgres_test_database
    try:
        source_data = _data_source().model_dump()
        source_data["connection_ref"] = "env:CHATBI_SCHEMA_TEST_DATABASE_URL"
        source = DataSource.model_validate(source_data)
        result = await SchemaDiscoveryService(
            EnvironmentCredentialResolver(),
            PostgresSchemaInspector(),
        ).discover(
            source,
            QueryPolicy(allowed_schemas=("chatbi_demo",), allow_views=True),
            max_chars=10_000,
        )
    finally:
        os.environ.pop("CHATBI_SCHEMA_TEST_DATABASE_URL", None)

    names = {
        (relation.schema_name, relation.name, relation.kind.value)
        for relation in result.snapshot.relations
    }
    assert ("chatbi_demo", "customers", "table") in names
    assert ("chatbi_demo", "sales", "table") in names
    assert ("chatbi_demo", "region_sales", "view") in names
    sales = next(relation for relation in result.snapshot.relations if relation.name == "sales")
    assert sales.primary_key == ("sale_id",)
    assert sales.foreign_keys[0].target_relation == "customers"
    assert "amount" in result.context.text
