from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from knowledge_scope.chatbi import (
    ChatBIAgentService,
    ChatBIResult,
    ChatBIUsageSummary,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryLifecycleState,
    QueryPolicy,
    SQLExecutionService,
)
from knowledge_scope.chatbi.discovery import create_postgres_schema_discovery_service
from knowledge_scope.chatbi.models import ChatBIDataSourceRecord
from knowledge_scope.chatbi.policy import SQLDialect
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider
from knowledge_scope.evaluation.chatbi_evaluation import (
    ChatBIEvaluationObservation,
    ChatBIStageTimings,
    EvaluationStageState,
    RepairOutcome,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    ChatBIEvaluationV2Category,
    ChatBIEvaluationV2Difficulty,
    load_chatbi_evaluation_dataset_v2,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_provider import (
    CHATBI_EVALUATION_V2_CONNECTION_REF,
    CHATBI_EVALUATION_V2_DATABASE_NAME,
    CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
    CHATBI_EVALUATION_V2_DATASOURCE_ID,
    CHATBI_EVALUATION_V2_SCHEMA,
    DEFAULT_DATASET_V2,
    DEFAULT_FIXTURE_PATH_V2,
    EXPECTED_DATASET_FINGERPRINT_V2,
    EXPECTED_FIXTURE_DATA_FINGERPRINT_V2,
    EXPECTED_FIXTURE_FINGERPRINT_V2,
    EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
    V2ProviderBenchmarkError,
    V2ProviderCaseRecord,
    V2ProviderPreflightReport,
    V2ProviderSplit,
    _authoritative_record_matches,
    _build_v2_aggregate,
    _fixture_data_fingerprint,
    _fixture_schema_fingerprint,
    _fixture_schema_payload,
    _local_postgres_urls,
    _prepare_v2_preflight,
    _TimingState,
    _v2_query_policy,
    _v2_record,
    _v2_usage_aggregate,
    _V2AgentRunner,
    evaluate_v2_provider_cases,
    select_v2_provider_cases,
)
from knowledge_scope.llm import LLMProviderInvocation, LLMRequest, LLMResult
from knowledge_scope.llm.usage import (
    CompositeProviderInvocationRecorder,
    InMemoryProviderInvocationRecorder,
)
from knowledge_scope.shared.config import Settings


def test_v2_provider_selects_only_the_frozen_dev_split() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)

    selected = select_v2_provider_cases(dataset, V2ProviderSplit.DEV.value)

    assert len(selected) == 50
    assert {case.split for case in selected} == {"dev"}
    with pytest.raises(V2ProviderBenchmarkError, match="permits only"):
        select_v2_provider_cases(dataset, V2ProviderSplit.TEST.value)


def test_v2_provider_cli_rejects_test_before_provider_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from knowledge_scope import cli

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))

    assert cli.main(["chatbi", "eval-v2-provider", "--split", "test", "--preflight"]) == 1
    assert "permits only" in capsys.readouterr().err


class _CountingV2Runner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, _case: object) -> ChatBIEvaluationObservation:
        self.calls += 1
        return ChatBIEvaluationObservation(
            result=ChatBIResult(
                query_id=uuid4(),
                datasource_id=uuid4(),
                execution_status=QueryLifecycleState.SUCCEEDED,
                row_count=0,
                sql_attempts=1,
                repair_attempts=0,
                usage=ChatBIUsageSummary(
                    llm_calls=0,
                    provider_attempts=0,
                ),
            )
        )


def _provider_case_record(
    *,
    case_id: str,
    positive: bool = True,
    initial_generation_parseable: bool = True,
    repair_attempted: bool = False,
    repair_generation_parseable: bool | None = None,
    repair_recovery_success: bool = False,
    provider_invocations: list[LLMProviderInvocation] | None = None,
) -> V2ProviderCaseRecord:
    if not repair_attempted:
        repair_outcome = RepairOutcome.NOT_ATTEMPTED
        repair_generation_state = "not_attempted"
    elif repair_recovery_success:
        repair_outcome = RepairOutcome.REPAIRED
        repair_generation_state = "passed"
    else:
        repair_outcome = RepairOutcome.EXHAUSTED
        repair_generation_state = "passed" if repair_generation_parseable else "failed"
    return V2ProviderCaseRecord(
        case_id=case_id,
        positive=positive,
        category=ChatBIEvaluationV2Category.SIMPLE_FILTER_PROJECTION,
        difficulty=ChatBIEvaluationV2Difficulty.EASY,
        stages=EvaluationStageState(
            generation="passed" if initial_generation_parseable else "failed",
            validation="passed" if positive else "failed",
            execution="passed" if positive else "not_attempted",
            analysis="passed" if positive else "not_attempted",
            repair_generation=repair_generation_state,
        ),
        result_status=QueryLifecycleState.SUCCEEDED if positive else QueryLifecycleState.REJECTED,
        generation_parseable=initial_generation_parseable,
        initial_generation_parseable=initial_generation_parseable,
        eventual_sql_candidate_available=positive or repair_generation_parseable is True,
        repair_generation_parseable=repair_generation_parseable if repair_attempted else None,
        repair_recovery_success=repair_recovery_success,
        validation_accepted=positive,
        execution_equivalent=True if positive else None,
        structured_result_fact_coverage=None,
        expected_failure_match=True if not positive else None,
        repair_attempted=repair_attempted,
        repair_outcome=repair_outcome,
        row_count=1 if positive else 0,
        truncated=False,
        latency=ChatBIStageTimings(
            schema_prep_ms=1,
            generation_ms=1,
            validation_ms=1,
            execution_ms=1,
            repair_ms=1 if repair_attempted else None,
            analysis_ms=1 if positive else None,
            total_ms=1,
        ),
        usage=ChatBIUsageSummary(
            llm_calls=1,
            provider_attempts=1,
            input_tokens=10,
            output_tokens=5,
        ),
        provider_invocations=provider_invocations or [],
        trace_events=["sql_executed"] if positive else [],
    )


def test_v2_usage_aggregate_separates_attempts_from_completed_results() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    invocations = [
        LLMProviderInvocation(
            case_id="case-001",
            logical_stage="generation",
            attempt_index=1,
            provider="fake",
            model="fake-model",
            started_at=now,
            completed_at=now,
            duration_ms=1,
            outcome="success",
            input_tokens=10,
            output_tokens=5,
            llm_result_returned=True,
        ),
        LLMProviderInvocation(
            case_id="case-001",
            logical_stage="generation",
            attempt_index=2,
            provider="fake",
            model="fake-model",
            started_at=now,
            completed_at=now,
            duration_ms=1,
            outcome="failure",
            error_category="provider_api_error",
            llm_result_returned=False,
        ),
    ]

    aggregate = _v2_usage_aggregate(
        [
            _provider_case_record(
                case_id="case-001",
                provider_invocations=invocations,
            )
        ]
    )

    assert aggregate.provider_attempt_count == 2
    assert aggregate.provider_attempts_total == 2
    assert aggregate.provider_success_count == 1
    assert aggregate.provider_failure_count == 1
    assert aggregate.llm_result_count == 1
    assert aggregate.input_tokens.total == 10
    assert aggregate.output_tokens.total == 5


def test_v2_repair_aggregate_uses_runtime_recovery_not_review_metadata() -> None:
    recovered = _provider_case_record(
        case_id="repair-recovered",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=True,
        repair_recovery_success=True,
    )
    parseable_but_not_recovered = _provider_case_record(
        case_id="repair-exhausted",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=True,
    )
    failed_repair = _provider_case_record(
        case_id="repair-malformed",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=False,
    )

    aggregate = _build_v2_aggregate([recovered, parseable_but_not_recovered, failed_repair])

    assert aggregate.generation_parseable_count == 0
    assert aggregate.repair_attempted_count == 3
    assert aggregate.repair_generation_success_count == 2
    assert aggregate.repair_recovery_success_count == 1
    assert aggregate.repair_success_count == 1
    assert aggregate.repair_failure_count == 2
    assert aggregate.incremental_recovered_count == 1


def test_v2_record_keeps_initial_and_eventual_generation_metrics_distinct() -> None:
    case = next(
        case
        for case in load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2).cases
        if case.split == V2ProviderSplit.DEV.value and case.positive
    )
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        execution_status=QueryLifecycleState.SUCCEEDED,
        redacted_sql="SELECT 1",
        row_count=1,
        sql_attempts=2,
        repair_attempts=1,
        usage=ChatBIUsageSummary(llm_calls=2, provider_attempts=2),
    )
    record = _v2_record(
        case,
        ChatBIEvaluationObservation(
            result=result,
            stages=EvaluationStageState(
                generation="failed",
                repair_generation="passed",
                validation="passed",
                execution="passed",
            ),
        ),
    )

    assert record.initial_generation_parseable is False
    assert record.generation_parseable is False
    assert record.repair_generation_parseable is True
    assert record.eventual_sql_candidate_available is True


@pytest.mark.anyio
async def test_v2_runner_updates_durable_and_local_invocation_metadata() -> None:
    """Post-processing parse outcomes reach both evaluation and durable sinks."""
    local_recorder = InMemoryProviderInvocationRecorder()
    durable_recorder = InMemoryProviderInvocationRecorder()
    sink = CompositeProviderInvocationRecorder(durable_recorder, local_recorder)
    case = select_v2_provider_cases(
        load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2), V2ProviderSplit.DEV.value
    )[0]

    class _Agent:
        async def ask(self, *_args: object, **_kwargs: object) -> ChatBIResult:
            now = datetime.now(UTC)
            await sink.record_invocation(
                LLMProviderInvocation(
                    case_id=case.case_id,
                    logical_stage="analysis",
                    attempt_index=1,
                    provider="fake",
                    model="fake-model",
                    started_at=now,
                    completed_at=now,
                    duration_ms=1,
                    outcome="success",
                    input_tokens=10,
                    output_tokens=4,
                    llm_result_returned=True,
                    response_parse_outcome="provider_success",
                )
            )
            return ChatBIResult(
                query_id=uuid4(),
                datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
                execution_status=QueryLifecycleState.SUCCEEDED,
                row_count=0,
                sql_attempts=1,
                repair_attempts=0,
                usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
                analysis_parse_outcome="structured_output_parse_error",
            )

    runner = _V2AgentRunner(
        _Agent(),
        _TimingState(),
        CHATBI_EVALUATION_V2_DATASOURCE_ID,
        QueryPolicy(),
        10_000,
        local_recorder,
        sink,
    )

    observation = await runner.run(case)

    assert observation.provider_invocations[0].response_parse_outcome == (
        "structured_output_parse_error"
    )
    assert durable_recorder.records[0].response_parse_outcome == ("structured_output_parse_error")
    assert durable_recorder.records[0].error_category == "structured_output_parse_error"


@pytest.mark.anyio
async def test_v2_provider_api_rejects_non_dev_cases_before_runner() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    test_case = next(case for case in dataset.cases if case.split == "test")
    dev_case = next(case for case in dataset.cases if case.split == "dev")
    mutated_dev_case = dev_case.model_copy(update={"question": "changed"})

    for cases in (
        [test_case],
        [dev_case, test_case],
        [dev_case, dev_case],
        [mutated_dev_case],
    ):
        runner = _CountingV2Runner()
        with pytest.raises(V2ProviderBenchmarkError):
            await evaluate_v2_provider_cases(cases, runner)
        assert runner.calls == 0


@pytest.mark.anyio
async def test_v2_provider_api_accepts_the_complete_frozen_dev_split() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    cases = select_v2_provider_cases(dataset, V2ProviderSplit.DEV.value)
    runner = _CountingV2Runner()

    records = await evaluate_v2_provider_cases(cases, runner)

    assert len(records) == 50
    assert runner.calls == 50
    assert {record.split for record in records} == {"dev"}


class _CatalogRow:
    def __init__(self, **values: object) -> None:
        self._mapping = values


def test_v2_schema_payload_is_independent_of_catalog_row_order() -> None:
    relations = [
        _CatalogRow(
            relation_name="customers",
            relation_kind="table",
            column_position=1,
            column_name="customer_id",
            data_type="bigint",
            not_null=True,
        ),
        _CatalogRow(
            relation_name="customers",
            relation_kind="table",
            column_position=2,
            column_name="customer_name",
            data_type="text",
            not_null=True,
        ),
        _CatalogRow(
            relation_name="region_sales",
            relation_kind="view",
            column_position=1,
            column_name="region_code",
            data_type="text",
            not_null=False,
        ),
    ]
    constraints = [
        _CatalogRow(
            relation_name="customers",
            constraint_type="p",
            columns=["customer_id"],
            referenced_schema=None,
            referenced_relation=None,
            referenced_columns=[],
            check_definition=None,
        )
    ]

    assert _fixture_schema_payload(relations, constraints) == _fixture_schema_payload(
        list(reversed(relations)), list(reversed(constraints))
    )


def test_v2_provider_policy_is_fixed_to_the_isolated_business_schema() -> None:
    settings = Settings(_env_file=None, chatbi_allowed_schemas=["public"])

    policy = _v2_query_policy(settings)

    assert policy.allowed_schemas == (CHATBI_EVALUATION_V2_SCHEMA,)
    assert policy.denied_schemas == ()
    assert policy.allow_views is False
    assert policy.max_rows == settings.chatbi_max_rows
    assert policy.statement_timeout_ms == settings.chatbi_statement_timeout_ms


def test_v2_database_identity_rejects_application_database_collision() -> None:
    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql+asyncpg://benchmark:super-secret@localhost:5433/"
            "knowledgescope_chatbi_eval_v2"
        ),
    )

    with pytest.raises(V2ProviderBenchmarkError) as error:
        _local_postgres_urls(settings)

    assert "super-secret" not in str(error.value)
    assert "postgresql" not in str(error.value).lower()


def test_v2_database_identity_allows_distinct_local_application_database() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://benchmark:secret@127.0.0.1:5433/knowledgescope",
    )

    admin_url, evaluation_url = _local_postgres_urls(settings)

    assert admin_url.database == "postgres"
    assert evaluation_url.database == CHATBI_EVALUATION_V2_DATABASE_NAME


@pytest.mark.anyio
async def test_v2_database_collision_fails_before_fixture_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_scope.evaluation import chatbi_evaluation_v2_provider as provider_module

    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://benchmark:secret@127.0.0.1:5433/knowledgescope_chatbi_eval_v2",
    )

    async def unexpected_database_probe(_url: object) -> bool:
        raise AssertionError("database probe must not run on an isolation collision")

    monkeypatch.setattr(provider_module, "_database_exists", unexpected_database_probe)
    with pytest.raises(V2ProviderBenchmarkError):
        await provider_module._ensure_benchmark_database(settings, DEFAULT_FIXTURE_PATH_V2)


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.anyio
async def test_v2_start_git_state_is_captured_before_output_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    dirty: bool,
) -> None:
    from knowledge_scope.evaluation import chatbi_evaluation_v2_provider as provider_module

    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    output_path = tmp_path / "unignored" / "provider.json"
    bootstrap = SimpleNamespace(
        data_source=SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID),
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        environment_was_present=True,
    )

    class _Engine:
        async def dispose(self) -> None:
            return None

    class _Registry:
        def __init__(self, _session_factory: object) -> None:
            pass

        async def get(self, _datasource_id: object) -> object:
            return SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID)

    def fake_current_git_dirty() -> bool:
        assert not output_path.parent.exists()
        return dirty

    async def fake_ensure(_settings: Settings, *, fixture_path: Any) -> object:
        return bootstrap

    async def fake_discover(
        _settings: Settings,
        _data_source: object,
        *,
        session_factory: object,
        max_chars: int,
    ) -> str:
        return "fixture-data-fingerprint"

    monkeypatch.setattr(provider_module, "current_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(provider_module, "current_git_dirty", fake_current_git_dirty)
    monkeypatch.setattr(
        provider_module,
        "_verify_v2_dataset_and_fixture",
        lambda _dataset_path, _fixture_path: (dataset, EXPECTED_FIXTURE_FINGERPRINT_V2),
    )
    monkeypatch.setattr(provider_module, "_check_provider_configuration", lambda _settings: None)
    monkeypatch.setattr(provider_module, "ensure_authoritative_v2_datasource", fake_ensure)
    monkeypatch.setattr(provider_module, "create_database_engine", lambda _settings: _Engine())
    monkeypatch.setattr(provider_module, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(provider_module, "DatabaseDataSourceProvider", _Registry)
    monkeypatch.setattr(provider_module, "_discover_and_verify_v2_schema", fake_discover)

    context = await _prepare_v2_preflight(
        Settings(_env_file=None),
        split=V2ProviderSplit.DEV.value,
        dataset_path=DEFAULT_DATASET_V2,
        fixture_path=DEFAULT_FIXTURE_PATH_V2,
        output_path=output_path,
    )

    assert context.git_revision == "a" * 40
    assert context.git_dirty is dirty
    assert output_path.parent.is_dir()


def test_authoritative_datasource_identity_is_exact_and_opaque() -> None:
    record = ChatBIDataSourceRecord(
        id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        display_name=CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
        dialect=SQLDialect.POSTGRESQL.value,
        enabled=True,
        connection_ref=CHATBI_EVALUATION_V2_CONNECTION_REF,
        default_database=CHATBI_EVALUATION_V2_DATABASE_NAME,
        default_schema=CHATBI_EVALUATION_V2_SCHEMA,
    )

    assert _authoritative_record_matches(record)
    assert "postgresql" not in record.connection_ref
    record.display_name = "different"  # type: ignore[assignment]
    assert not _authoritative_record_matches(record)


def test_preflight_report_is_provider_free_and_safe() -> None:
    report = V2ProviderPreflightReport(
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        fixture_data_fingerprint="0" * 64,
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        split="dev",
        selected_case_count=50,
        selected_test_case_count=0,
        provider="deepseek",
        model="configured-model",
        provider_configured=True,
        database_created=False,
        datasource_created=False,
        git_revision="a" * 40,
        git_dirty=False,
    )

    output = report.model_dump(mode="json")

    assert output["provider_calls"] == 0
    assert "connection_url" not in output
    assert "api_key" not in json.dumps(output).lower()


class _DeterministicV2Gateway:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task_type == "nl2sql":
            response = (
                '{"sql":"SELECT customer_name FROM chatbi_demo.customers ORDER BY customer_id"}'
            )
            input_tokens, output_tokens = 13, 8
        elif request.task_type == "chatbi_analysis":
            response = '{"answer":"已返回客户名称。","warning":null}'
            input_tokens, output_tokens = 17, 6
        else:
            raise AssertionError(f"unexpected task type: {request.task_type}")
        return LLMResult(
            text=response,
            provider="fake-v2",
            model="fake-v2-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1,
            provider_attempts=1,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_schema_fingerprint_rejects_structural_mutations(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """The frozen schema contract rejects every relevant structural mutation."""
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    assert await _fixture_schema_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )
    mutations = (
        (
            "added column",
            ("ALTER TABLE chatbi_demo.customers ADD COLUMN temporary_extra TEXT",),
            ("ALTER TABLE chatbi_demo.customers DROP COLUMN temporary_extra",),
        ),
        (
            "changed column type",
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name TYPE VARCHAR",),
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name TYPE TEXT",),
        ),
        (
            "changed nullability",
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name DROP NOT NULL",),
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name SET NOT NULL",),
        ),
        (
            "removed foreign key",
            ("ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_customer_id_fkey",),
            (
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_customer_id_fkey "
                "FOREIGN KEY (customer_id) REFERENCES chatbi_demo.customers(customer_id)",
            ),
        ),
        (
            "changed foreign key target",
            (
                "ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_region_code_fkey",
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_region_code_fkey "
                "FOREIGN KEY (region_code) REFERENCES chatbi_demo.regions(region_name) NOT VALID",
            ),
            (
                "ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_region_code_fkey",
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_region_code_fkey "
                "FOREIGN KEY (region_code) REFERENCES chatbi_demo.regions(region_code)",
            ),
        ),
        (
            "extra table",
            ("CREATE TABLE chatbi_demo.temporary_extra (value TEXT)",),
            ("DROP TABLE chatbi_demo.temporary_extra",),
        ),
        (
            "extra view",
            ("CREATE VIEW chatbi_demo.temporary_extra AS SELECT 1 AS value",),
            ("DROP VIEW chatbi_demo.temporary_extra",),
        ),
        (
            "missing table",
            ("ALTER TABLE chatbi_demo.customers RENAME TO customers_missing",),
            ("ALTER TABLE chatbi_demo.customers_missing RENAME TO customers",),
        ),
        (
            "missing view",
            ("ALTER VIEW chatbi_demo.region_sales RENAME TO region_sales_missing",),
            ("ALTER VIEW chatbi_demo.region_sales_missing RENAME TO region_sales",),
        ),
    )
    for label, apply_sql, restore_sql in mutations:
        async with postgres_test_engine.begin() as connection:
            for statement in apply_sql:
                await connection.exec_driver_sql(statement)
        try:
            assert await _fixture_schema_fingerprint(postgres_test_database) != (
                EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
            ), label
        finally:
            async with postgres_test_engine.begin() as connection:
                for statement in restore_sql:
                    await connection.exec_driver_sql(statement)
        assert await _fixture_schema_fingerprint(postgres_test_database) == (
            EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
        ), label


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_provider_path_uses_real_discovery_validation_and_execution(
    postgres_test_engine: Any,
    postgres_test_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the registered-datasource production chain without a provider."""
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)
    assert await _fixture_data_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_DATA_FINGERPRINT_V2
    )
    assert await _fixture_schema_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )

    credential_name = "KNOWLEDGE_SCOPE_CHATBI_EVAL_V2_TEST_DATABASE_URL"
    monkeypatch.setenv(credential_name, postgres_test_database)
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = ChatBIDataSourceRecord(
        id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        display_name=CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
        dialect=SQLDialect.POSTGRESQL.value,
        enabled=True,
        connection_ref=f"env:{credential_name}",
        default_database=None,
        default_schema=CHATBI_EVALUATION_V2_SCHEMA,
    )
    record.created_at = timestamp
    record.updated_at = timestamp

    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add(record)
        await session.commit()

    try:
        registry = DatabaseDataSourceProvider(session_factory)
        discovery = create_postgres_schema_discovery_service(
            connection_timeout_seconds=5,
            statement_timeout_ms=30_000,
        )
        gateway = _DeterministicV2Gateway()
        generation = NL2SQLService(
            gateway,
            schema_discovery=discovery,
            data_source_provider=registry,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(
                connection_timeout_seconds=5,
                statement_timeout_ms=30_000,
            ),
        )
        result = await ChatBIAgentService(generation, execution, gateway).ask(
            CHATBI_EVALUATION_V2_DATASOURCE_ID,
            "列出客户名称。",
            policy=QueryPolicy(allowed_schemas=(CHATBI_EVALUATION_V2_SCHEMA,), max_rows=10),
            max_chars=20_000,
        )

        assert result.execution_status is QueryLifecycleState.SUCCEEDED
        assert result.datasource_id == CHATBI_EVALUATION_V2_DATASOURCE_ID
        assert result.answer == "已返回客户名称。"
        assert result.row_count == 10
        assert result.truncated is False
        assert result.truncation_reason is None
        assert result.redacted_sql is not None
        assert "chatbi_demo.customers" in result.redacted_sql
        assert result.usage.llm_calls == 2
        assert result.usage.provider_attempts == 2
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 14
        assert [request.task_type for request in gateway.requests] == [
            "nl2sql",
            "chatbi_analysis",
        ]
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
        assert credential_name not in safe_result
        assert postgres_test_database not in safe_result
        assert "password" not in safe_result.lower()
    finally:
        async with session_factory() as session:
            existing = await session.get(ChatBIDataSourceRecord, CHATBI_EVALUATION_V2_DATASOURCE_ID)
            if existing is not None:
                await session.delete(existing)
                await session.commit()
