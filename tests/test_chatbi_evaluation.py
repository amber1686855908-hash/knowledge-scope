from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from knowledge_scope.chatbi import (
    ChatBIErrorCategory,
    ChatBIResult,
    ChatBIUsageSummary,
    ColumnMetadata,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryTruncationReason,
)
from knowledge_scope.evaluation.chatbi_evaluation import (
    DEFAULT_DATASET,
    DEFAULT_FIXTURE_PATH,
    DEFAULT_OFFLINE_SCENARIOS,
    ChatBIEvaluationCategory,
    ChatBIEvaluationError,
    ChatBIStageTimings,
    EvaluationCaseRecord,
    EvaluationStageState,
    ExpectedQueryResult,
    OfflineScenario,
    RepairOutcome,
    _aggregate_records,
    _TimingState,
    assert_no_oracle_leakage,
    compare_normalized_result,
    load_chatbi_evaluation_dataset,
    load_offline_scenarios,
    run_offline_chatbi_evaluation,
    run_provider_chatbi_evaluation,
    verify_fixture_fingerprint,
)
from knowledge_scope.llm.schemas import LLMMessage
from knowledge_scope.shared.config import Settings


def _successful_result(
    rows: list[list[object]],
    *,
    columns: list[str] | None = None,
    truncated: bool = False,
    truncation_reason: QueryTruncationReason | None = None,
) -> QueryExecutionResult:
    names = columns or [f"column_{index}" for index in range(len(rows[0]) if rows else 1)]
    return QueryExecutionResult(
        query_id=uuid4(),
        datasource_id=UUID("00000000-0000-4000-8000-000000000057"),
        state=QueryLifecycleState.SUCCEEDED,
        columns=[
            ColumnMetadata(name=name, data_type="text", nullable=True, ordinal=index)
            for index, name in enumerate(names)
        ],
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        truncation_reason=truncation_reason,
        max_rows=max(1, len(rows)),
        duration_ms=0,
    )


def test_frozen_dataset_and_fixture_fingerprint_are_valid() -> None:
    dataset = load_chatbi_evaluation_dataset(DEFAULT_DATASET)

    verify_fixture_fingerprint(dataset, DEFAULT_FIXTURE_PATH)
    assert dataset.dataset_fingerprint == dataset.fingerprint
    assert len(dataset.cases) == 20
    assert sum(case.split == "dev" for case in dataset.cases) == 14
    assert sum(case.split == "test" for case in dataset.cases) == 6
    assert sum(case.expected_result is not None for case in dataset.cases) == 16


def test_dataset_fingerprint_rejects_material_changes() -> None:
    dataset = load_chatbi_evaluation_dataset()
    payload = dataset.model_dump(mode="json")
    payload["cases"][0]["question"] = "改动后的评测问题"
    payload["dataset_fingerprint"] = dataset.fingerprint

    with pytest.raises(ValidationError):
        type(dataset).model_validate(payload)


def test_oracle_data_is_rejected_if_it_enters_model_messages() -> None:
    dataset = load_chatbi_evaluation_dataset()
    case = dataset.cases[0]

    assert_no_oracle_leakage(case, [LLMMessage(role="user", content=case.question)])
    with pytest.raises(ChatBIEvaluationError, match="reference SQL leaked"):
        assert_no_oracle_leakage(
            case,
            [{"role": "user", "content": case.reference_sql or ""}],
        )
    with pytest.raises(ChatBIEvaluationError, match="expected answer fact leaked"):
        assert_no_oracle_leakage(
            case,
            [{"role": "user", "content": case.expected_answer_facts[1]}],
        )


def test_result_comparison_handles_empty_rows_numeric_tolerance_and_aliases() -> None:
    expected = ExpectedQueryResult(
        columns=["expected_amount", "note"],
        rows=[["1.0000000", None]],
        numeric_columns=[0],
    )
    actual = _successful_result(
        [["1.0000005", None]],
        columns=["renamed_amount", "renamed_note"],
    )

    comparison = compare_normalized_result(actual, expected)
    assert comparison.equivalent is True

    empty_expected = ExpectedQueryResult(columns=["id"], rows=[])
    assert compare_normalized_result(
        _successful_result([], columns=["id"]), empty_expected
    ).equivalent


def test_result_comparison_preserves_duplicate_rows_when_order_is_irrelevant() -> None:
    expected = ExpectedQueryResult(
        columns=["value"],
        rows=[[1], [1], [2]],
        row_order_sensitive=False,
    )
    actual = _successful_result([[2], [1], [1]], columns=["different_name"])

    assert compare_normalized_result(actual, expected).equivalent is True


def test_result_comparison_requires_matching_truncation_metadata() -> None:
    expected = ExpectedQueryResult(
        columns=["value"],
        rows=[[1]],
        expected_truncated=True,
        truncation_reason="row_limit",
    )
    actual = _successful_result(
        [[1]],
        columns=["value"],
        truncated=True,
        truncation_reason=QueryTruncationReason.ROW_LIMIT,
    )

    assert compare_normalized_result(actual, expected).equivalent is True
    assert (
        compare_normalized_result(_successful_result([[1]], columns=["value"]), expected).equivalent
        is False
    )


def test_numeric_comparison_handles_decimal_and_rejects_non_finite_values() -> None:
    from knowledge_scope.evaluation.chatbi_evaluation import _values_equal

    assert _values_equal(Decimal("1.00"), "1.0000001", numeric=True)
    assert not _values_equal(float("nan"), "1", numeric=True)
    assert not _values_equal(float("inf"), "1", numeric=True)


def test_offline_scenario_contract_rejects_inconsistent_rows() -> None:
    with pytest.raises(ValidationError):
        OfflineScenario(
            case_id="bad-case",
            execution_status=QueryLifecycleState.SUCCEEDED,
            answer="answer",
            columns=["value"],
            rows=[[1]],
            row_count=0,
            sql_attempts=1,
            repair_attempts=0,
            llm_calls=1,
            provider_attempts=1,
        )


@pytest.mark.anyio
async def test_offline_evaluation_is_provider_free_and_compares_materialized_rows(
    tmp_path: Path,
) -> None:
    dataset = load_chatbi_evaluation_dataset()
    scenarios = load_offline_scenarios(DEFAULT_OFFLINE_SCENARIOS)
    output = tmp_path / "run.json"

    run = await run_offline_chatbi_evaluation(dataset, scenarios, output_path=output)

    assert run.mode == "offline"
    assert run.quality_claim == "offline_infrastructure_only"
    assert run.case_count == 20
    assert run.aggregates["all"].positive_case_count == 16
    assert run.aggregates["all"].negative_case_count == 4
    aggregate = run.aggregates["all"]
    assert aggregate.offline_verification is not None
    assert aggregate.offline_verification.scenario_match_rate == 1
    assert aggregate.offline_verification.comparator_checks == 16
    assert aggregate.offline_verification.comparator_pass_count == 16
    assert aggregate.offline_verification.taxonomy_checks == 4
    assert aggregate.offline_verification.taxonomy_pass_count == 4
    assert aggregate.offline_verification.repair_scenario_checks == 8
    assert aggregate.offline_verification.repair_scenario_pass_count == 8
    assert aggregate.provider_quality is None
    assert run.aggregates["all"].negative_failure_match_count == 4
    assert run.aggregates["all"].usage.llm_calls_total == 44
    assert run.git_revision == run.provenance.git_revision
    assert all(record.answer_fact_coverage is None for record in run.records)
    assert all(record.end_to_end_success is None for record in run.records)
    assert output.exists()
    serialized = output.read_text(encoding="utf-8")
    assert "reference_sql" not in serialized
    assert "销售总额为 3600.75" not in serialized
    assert "execution_accuracy" not in serialized
    assert "answer_factual_correctness" not in serialized


def test_offline_scenarios_have_safe_failure_categories_and_usage() -> None:
    scenarios = load_offline_scenarios()
    assert scenarios["unsupported-table-01"].error_category is ChatBIErrorCategory.UNKNOWN_TABLE
    assert scenarios["unsupported-view-01"].error_category is ChatBIErrorCategory.POLICY_VIOLATION
    assert all(scenario.provider_attempts >= scenario.llm_calls for scenario in scenarios.values())


def test_offline_scenario_format_is_versioned(tmp_path: Path) -> None:
    payload = {"format": "unknown", "scenarios": []}
    path = tmp_path / "invalid-scenarios.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChatBIEvaluationError, match="invalid offline scenario"):
        load_offline_scenarios(path)


@pytest.mark.anyio
async def test_offline_evaluation_rejects_scenario_coverage_or_datasource_drift() -> None:
    dataset = load_chatbi_evaluation_dataset()
    scenarios = load_offline_scenarios()
    removed = dict(scenarios)
    removed.pop("filtering-01")

    with pytest.raises(ChatBIEvaluationError, match="coverage"):
        await run_offline_chatbi_evaluation(dataset, removed)

    split_datasources = dict(scenarios)
    split_datasources["filtering-01"] = split_datasources["filtering-01"].model_copy(
        update={"datasource_id": UUID("00000000-0000-4000-8000-000000000058")}
    )
    with pytest.raises(ChatBIEvaluationError, match="one isolated datasource"):
        await run_offline_chatbi_evaluation(dataset, split_datasources)


def test_chatbi_result_without_rows_never_looks_like_an_empty_success() -> None:
    dataset = load_chatbi_evaluation_dataset()
    expected = dataset.cases[0].expected_result
    assert expected is not None
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=UUID("00000000-0000-4000-8000-000000000057"),
        execution_status=QueryLifecycleState.SUCCEEDED,
        columns=[],
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=0, provider_attempts=0),
    )

    comparison = compare_normalized_result(result, expected)
    assert comparison.equivalent is False
    assert "unavailable" in comparison.reason


def _evaluation_record(
    case_id: str,
    *,
    stages: EvaluationStageState,
    positive_case: bool = True,
    execution_equivalent: bool | None = None,
    answer_fact_coverage: bool | None = None,
    expected_failure_match: bool | None = None,
    end_to_end_success: bool = False,
    repair_outcome: RepairOutcome = RepairOutcome.NOT_APPLICABLE,
) -> EvaluationCaseRecord:
    return EvaluationCaseRecord(
        case_id=case_id,
        split="dev",
        category=ChatBIEvaluationCategory.FILTERING,
        positive_case=positive_case,
        stages=stages,
        execution_status=QueryLifecycleState.SUCCEEDED
        if end_to_end_success
        else QueryLifecycleState.FAILED,
        execution_equivalent=execution_equivalent,
        answer_fact_coverage=answer_fact_coverage,
        expected_failure_match=expected_failure_match,
        end_to_end_success=end_to_end_success,
        repair_outcome=repair_outcome,
        failure_category=None,
        row_count=0,
        truncated=False,
        truncation_reason=None,
        metrics={},
        latency=ChatBIStageTimings(),
        usage=ChatBIUsageSummary(llm_calls=0, provider_attempts=0),
    )


def test_provider_quality_uses_explicit_stage_attempt_denominators() -> None:
    records = [
        _evaluation_record(
            "passed-case",
            stages=EvaluationStageState(
                schema_preparation="passed",
                generation="passed",
                validation="passed",
                execution="passed",
                analysis="passed",
            ),
            execution_equivalent=True,
            answer_fact_coverage=True,
            end_to_end_success=True,
        ),
        _evaluation_record(
            "no-fact-case",
            stages=EvaluationStageState(
                schema_preparation="passed",
                generation="passed",
                validation="passed",
                execution="passed",
                analysis="passed",
            ),
            execution_equivalent=True,
            end_to_end_success=True,
        ),
        _evaluation_record(
            "malformed-case",
            stages=EvaluationStageState(
                schema_preparation="passed",
                generation="failed",
            ),
        ),
        _evaluation_record(
            "discovery-failure",
            stages=EvaluationStageState(schema_preparation="failed"),
        ),
    ]

    quality = _aggregate_records(records, mode="provider").provider_quality

    assert quality is not None
    assert quality.generation_attempted_count == 3
    assert quality.generation_parseable_count == 2
    assert quality.generation_parseable_rate == 2 / 3
    assert quality.generation_not_attempted_count == 1
    assert quality.validation_attempted_count == 2
    assert quality.validator_accepted_count == 2
    assert quality.validator_acceptance_rate == 1
    assert quality.answer_fact_coverage_count == 1
    assert quality.answer_fact_coverage_rate == 1


def test_stage_timer_excludes_nested_discovery_time() -> None:
    assert _TimingState._exclusive_elapsed(100.0, 35.0) == 65.0
    assert _TimingState._exclusive_elapsed(10.0, 20.0) == 0.0


def test_unattempted_stage_timings_are_not_aggregated_as_zero() -> None:
    timing = _TimingState()

    initial = timing.snapshot()
    assert initial.schema_prep_ms is None
    assert initial.generation_ms is None
    assert initial.validation_ms is None
    assert initial.execution_ms is None
    assert initial.repair_ms is None
    assert initial.analysis_ms is None

    timing.schema_prep_ms = 1.5
    timing.generation_ms = 2.5
    timing.stages = EvaluationStageState(
        schema_preparation="failed",
        generation="passed",
    )
    measured = timing.snapshot()
    assert measured.schema_prep_ms == 1.5
    assert measured.generation_ms == 2.5
    assert measured.validation_ms is None


@pytest.mark.anyio
async def test_provider_evaluation_rejects_non_authoritative_datasource_before_provider_setup() -> (
    None
):
    dataset = load_chatbi_evaluation_dataset()
    settings = Settings(
        _env_file=None,
        chatbi_evaluation_datasource_id=UUID("00000000-0000-4000-8000-000000000058"),
    )

    with pytest.raises(ChatBIEvaluationError, match="authoritative demo datasource"):
        await run_provider_chatbi_evaluation(
            dataset,
            settings=settings,
            datasource_id=UUID("00000000-0000-4000-8000-000000000057"),
        )


@pytest.mark.anyio
async def test_provider_evaluation_rejects_invalid_max_chars_before_database_setup() -> None:
    dataset = load_chatbi_evaluation_dataset()
    settings = Settings(
        _env_file=None,
        chatbi_evaluation_datasource_id=UUID("00000000-0000-4000-8000-000000000057"),
    )

    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        await run_provider_chatbi_evaluation(
            dataset,
            settings=settings,
            datasource_id=UUID("00000000-0000-4000-8000-000000000057"),
            max_chars=0,
        )
