from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.chatbi import (
    DataSource,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryPolicy,
    SchemaDiscoveryService,
    SQLDialect,
    SQLExecutionService,
)
from knowledge_scope.chatbi.postgres_schema import PostgresSchemaInspector
from knowledge_scope.evaluation.chatbi_evaluation import (
    CHATBI_DEMO_FIXTURE_VERSION,
    DEFAULT_FIXTURE_PATH,
    ExpectedQueryResult,
    compare_normalized_result,
    verify_fixture_fingerprint,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    CHATBI_EVALUATION_V2_STATUS,
    DEFAULT_DATASET_V2,
    DEFAULT_FIXTURE_PATH_V2,
    DEFAULT_REVIEWER_ARTIFACT_V2,
    V2_CATEGORY_TARGETS,
    ChatBIEvaluationDatasetV2,
    ChatBIEvaluationV2AnswerFact,
    ChatBIEvaluationV2Category,
    audit_chatbi_evaluation_dataset_v2,
    load_chatbi_evaluation_dataset_v2,
    structured_result_facts_match,
    validate_v2_reference_sql,
    write_v2_reviewer_artifact,
)

V2_FINGERPRINT = "60c75c597da8fc71a0fa5b25d335b63410b44a4ab3a403da40ca72c5ae375ab3"
V2_SEMANTIC_PAYLOAD_SHA256 = "6c686dc125fefe7c7bfe50f5ff3d8a9cd3cb24ce8f754f9b34682e09eaff6bca"
V2_FIXTURE_FINGERPRINT = "fd972106c39c7a8b31b57975118708e213a32e4e008ee13fafc15b1ea1b5182d"
V1_FIXTURE_FINGERPRINT = "cd5334e2aa7cb5d3a0e8a4ac95c876fd396f7ae13c5b7a7620497bd3de753c49"
TEST_DATASOURCE_ID = UUID("00000000-0000-4000-8000-000000000257")


def _fake_result(expected: ExpectedQueryResult) -> QueryExecutionResult:
    from knowledge_scope.chatbi.schemas import ColumnMetadata

    return QueryExecutionResult(
        query_id=UUID("00000000-0000-4000-8000-000000000001"),
        datasource_id=TEST_DATASOURCE_ID,
        state=QueryLifecycleState.SUCCEEDED,
        columns=[
            ColumnMetadata(
                name=column,
                data_type="text",
                nullable=True,
                ordinal=index,
            )
            for index, column in enumerate(expected.columns)
        ],
        rows=expected.rows,
        row_count=len(expected.rows),
        truncated=expected.expected_truncated,
        truncation_reason=None,
        max_rows=100,
        duration_ms=0,
    )


def test_v2_dataset_has_exact_frozen_contract_and_fingerprints() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)

    verify_fixture_fingerprint(dataset, DEFAULT_FIXTURE_PATH_V2)
    assert dataset.dataset_status == CHATBI_EVALUATION_V2_STATUS == "human_reviewed_frozen"
    assert dataset.fingerprint == V2_FINGERPRINT
    assert dataset.datasource_fixture == DEFAULT_FIXTURE_PATH_V2.as_posix()
    assert dataset.fixture_version == "chatbi-demo-v2"
    assert dataset.datasource_fixture_sha256 == V2_FIXTURE_FINGERPRINT
    assert dataset.coverage.case_count == 80
    assert dataset.coverage.dev_count == 50
    assert dataset.coverage.test_count == 30
    assert dataset.coverage.positive_count == 74
    assert dataset.coverage.negative_count == 6
    assert dataset.coverage.category_counts == V2_CATEGORY_TARGETS
    assert dataset.coverage.difficulty_counts == {"easy": 18, "hard": 21, "medium": 41}
    assert all(case.dataset_version == "a5.7-v2" for case in dataset.cases)


def test_v2_freeze_preserves_semantic_payload_and_rejects_legacy_status() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    semantic_payload = dataset.model_dump(
        mode="json", exclude={"dataset_status", "dataset_fingerprint"}
    )
    semantic_bytes = json.dumps(
        semantic_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(semantic_bytes).hexdigest() == V2_SEMANTIC_PAYLOAD_SHA256

    legacy_status_payload = dataset.model_dump(mode="json")
    legacy_status_payload["dataset_status"] = "candidate_" + "pending_final_human_review"
    legacy_status_payload["dataset_fingerprint"] = None
    with pytest.raises(ValidationError):
        ChatBIEvaluationDatasetV2.model_validate(legacy_status_payload)


def test_v1_fixture_remains_separate_and_unchanged() -> None:
    assert CHATBI_DEMO_FIXTURE_VERSION == "chatbi-demo-v1"
    assert hashlib.sha256(DEFAULT_FIXTURE_PATH.read_bytes()).hexdigest() == V1_FIXTURE_FINGERPRINT
    assert DEFAULT_FIXTURE_PATH != DEFAULT_FIXTURE_PATH_V2


def test_v2_fingerprints_match_json_reviewer_and_methodology() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    reviewer = DEFAULT_REVIEWER_ARTIFACT_V2.read_text(encoding="utf-8")
    methodology = Path("docs/benchmarks/a5-7-chatbi-evaluation-v2.md").read_text(encoding="utf-8")

    assert f"`{dataset.fingerprint}`" in reviewer
    assert f"`{dataset.fingerprint}`" in methodology
    assert f"`{dataset.datasource_fixture_sha256}`" in reviewer
    assert f"`{dataset.datasource_fixture_sha256}`" in methodology
    assert "candidate_pending_final_human_review" not in reviewer
    assert "candidate_pending_final_human_review" not in methodology
    assert "human_reviewed_frozen" in reviewer
    assert "human_reviewed_frozen" in methodology


def test_v2_fixture_has_identity_safe_customers_null_boundary_and_three_tables() -> None:
    fixture = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    dataset = load_chatbi_evaluation_dataset_v2()
    multi_join = [case for case in dataset.cases if case.category.value == "multi_table_join"]

    assert fixture.count("CREATE TABLE chatbi_demo.") == 3
    assert "(10, '未成交客户')" in fixture
    assert "customer_name TEXT NOT NULL" in fixture
    assert "customer_name TEXT NOT NULL UNIQUE" not in fixture
    assert len(multi_join) == 6
    assert all(
        all(table in case.reference_sql for table in ("customers", "sales", "regions"))
        for case in multi_join
    )


def test_v2_order_sensitivity_and_ambiguous_behavior_are_explicit() -> None:
    cases = {case.case_id: case for case in load_chatbi_evaluation_dataset_v2().cases}

    assert cases["top-k-01"].order_sensitive
    assert cases["top-k-06"].order_sensitive
    assert not cases["aggregation-01"].order_sensitive
    assert not cases["group-by-01"].order_sensitive
    for case_id in (
        "simple-05",
        "simple-06",
        "predicate-01",
        "predicate-02",
        "predicate-03",
        "predicate-04",
        "predicate-05",
        "predicate-06",
        "predicate-07",
        "join-01",
        "join-02",
        "join-03",
        "join-04",
        "join-06",
        "join-08",
        "join-09",
        "multi-join-01",
        "multi-join-02",
        "multi-join-03",
        "multi-join-05",
        "derived-01",
        "cte-set-02",
    ):
        assert not cases[case_id].order_sensitive
    assert not cases["group-by-08"].order_sensitive
    assert cases["ambiguous-01"].expected_behavior.value == "clarify_or_refuse_safely"
    assert cases["ambiguous-02"].expected_behavior.value == "clarify_or_refuse_safely"
    assert cases["ambiguous-03"].expected_behavior.value == "clarify_or_refuse_safely"


def test_v2_repair_applicable_is_non_authoritative_metadata_only() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    assert not any(case.repair_applicable for case in dataset.cases)


def test_v2_dataset_audit_is_deterministic_and_surfaces_repeated_answers() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    first = audit_chatbi_evaluation_dataset_v2(dataset)
    second = audit_chatbi_evaluation_dataset_v2(dataset)

    assert first == second
    assert first.duplicate_normalized_question_groups == []
    assert first.near_duplicate_question_pairs == []
    assert first.same_reference_sql_groups == []
    assert first.dev_test_semantic_concern_pairs == []
    assert first.leakage_issues == []
    assert first.repeated_expected_result_groups


def test_v2_audit_flags_strong_cross_category_template_overlap() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    cases = list(dataset.cases)
    index = next(index for index, case in enumerate(cases) if case.case_id == "simple-07")
    cases[index] = cases[index].model_copy(
        update={
            "category": ChatBIEvaluationV2Category.JOIN,
            "question": "客户编号为 9 的客户名称是什么？",  # noqa: RUF001
        }
    )
    audit = audit_chatbi_evaluation_dataset_v2(dataset.model_copy(update={"cases": cases}))
    assert ["simple-04", "simple-07"] in audit.dev_test_semantic_concern_pairs

    index = next(index for index, case in enumerate(cases) if case.case_id == "join-10")
    cases[index] = cases[index].model_copy(
        update={
            "category": ChatBIEvaluationV2Category.JOIN,
            "question": "每位客户的销售总额是多少？包含没有销售记录的客户。",  # noqa: RUF001
        }
    )
    audit = audit_chatbi_evaluation_dataset_v2(dataset.model_copy(update={"cases": cases}))
    assert ["join-10", "null-02"] in audit.dev_test_semantic_concern_pairs


def test_v2_case_semantics_reject_inconsistent_positive_and_negative_records() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    payload = dataset.model_dump(mode="json")

    negative_with_oracle = payload.copy()
    negative_with_oracle["cases"] = [dict(case) for case in payload["cases"]]
    negative_with_oracle["cases"][0]["positive"] = False
    with pytest.raises(ValidationError):
        ChatBIEvaluationDatasetV2.model_validate(negative_with_oracle)

    positive_without_oracle = payload.copy()
    positive_without_oracle["cases"] = [dict(case) for case in payload["cases"]]
    positive_without_oracle["cases"][0]["reference_sql"] = None
    positive_without_oracle["cases"][0]["expected_result"] = None
    with pytest.raises(ValidationError):
        ChatBIEvaluationDatasetV2.model_validate(positive_without_oracle)

    extra_field = payload.copy()
    extra_field["cases"] = [dict(case) for case in payload["cases"]]
    extra_field["cases"][0]["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        ChatBIEvaluationDatasetV2.model_validate(extra_field)


@pytest.mark.anyio
async def test_v2_reference_validator_compares_all_positive_oracles_without_provider() -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    calls: list[str] = []
    expected_by_sql = {
        case.reference_sql: case.expected_result
        for case in dataset.cases
        if case.positive and case.reference_sql and case.expected_result
    }

    async def execute(sql: str) -> QueryExecutionResult:
        calls.append(sql)
        expected = expected_by_sql[sql]
        return _fake_result(expected)

    summary = await validate_v2_reference_sql(dataset, execute)

    assert summary.valid
    assert summary.positive_case_count == 74
    assert summary.executed_count == 74
    assert summary.passed_count == 74
    assert summary.failed_case_ids == []
    assert len(calls) == 74


def test_structured_answer_facts_preserve_row_pairing_and_nulls() -> None:
    expected = load_chatbi_evaluation_dataset_v2().cases
    by_id = {case.case_id: case for case in expected}
    case = by_id["join-05"]
    assert case.expected_result is not None
    assert any(fact.kind == "null" for fact in case.structured_answer_facts)
    assert (
        structured_result_facts_match(
            _fake_result(case.expected_result), case.structured_answer_facts
        )
        is True
    )

    empty_case = by_id["empty-01"]
    assert empty_case.expected_result is not None
    assert empty_case.structured_answer_facts == [ChatBIEvaluationV2AnswerFact(kind="empty")]
    assert (
        structured_result_facts_match(
            _fake_result(empty_case.expected_result), empty_case.structured_answer_facts
        )
        is True
    )


def test_v2_result_comparator_ignores_harmless_output_aliases() -> None:
    expected = ExpectedQueryResult(
        columns=["total_amount"],
        rows=[[1234.50]],
        numeric_columns=[0],
        row_order_sensitive=False,
    )
    actual = _fake_result(expected)
    actual = actual.model_copy(
        update={
            "columns": [actual.columns[0].model_copy(update={"name": "sum"})],
        }
    )
    comparison = compare_normalized_result(actual, expected)
    assert comparison.equivalent


def test_structured_answer_fact_model_rejects_invalid_shapes() -> None:
    with pytest.raises(ValidationError):
        ChatBIEvaluationV2AnswerFact(kind="scalar", values={"a": 1, "b": 2})
    with pytest.raises(ValidationError):
        ChatBIEvaluationV2AnswerFact(kind="empty", values={"a": 1})
    with pytest.raises(ValidationError):
        ChatBIEvaluationV2AnswerFact(kind="null", values={"a": "not-null"})


def test_v2_reviewer_artifact_is_bounded_and_explicit(tmp_path: Path) -> None:
    dataset = load_chatbi_evaluation_dataset_v2()
    output = tmp_path / "review.md"

    write_v2_reviewer_artifact(dataset, output)
    text = output.read_text(encoding="utf-8")

    assert text.startswith("# ChatBI 评测数据集 v2 审阅清单")
    assert text.count("- review status: `applied`") == 80
    assert "已完成最终人工审核与技术冻结审计" in text
    assert "SELECT" in text
    assert "structured answer facts" in text
    assert "repair-applicable` 仅是审核元数据" in text


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_all_positive_reference_sql_executes_on_isolated_demo_fixture(
    postgres_test_engine: object,
    postgres_test_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute each repository oracle through the normal trusted SQL service."""
    engine = postgres_test_engine
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with engine.begin() as connection:  # type: ignore[union-attr]
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    credential_name = "CHATBI_EVAL_V2_TEST_DATABASE_URL"
    monkeypatch.setenv(credential_name, postgres_test_database)
    source = DataSource(
        id=TEST_DATASOURCE_ID,
        display_name="ChatBI v2 isolated demo",
        dialect=SQLDialect.POSTGRESQL,
        enabled=True,
        connection_ref=f"env:{credential_name}",
        default_database=None,
        default_schema="chatbi_demo",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class Registry:
        async def get(self, datasource_id: UUID) -> DataSource | None:
            return source if datasource_id == source.id else None

    discovery = SchemaDiscoveryService(
        EnvironmentCredentialResolver(),
        PostgresSchemaInspector(),
    )
    validation = NL2SQLService(
        None,
        schema_discovery=discovery,
        data_source_provider=Registry(),
    )
    execution = SQLExecutionService(
        validation,
        EnvironmentCredentialResolver(),
        PostgresExecutionAdapter(),
    )
    policy = QueryPolicy(allowed_schemas=("chatbi_demo",), max_rows=100)
    dataset = load_chatbi_evaluation_dataset_v2()

    async def execute(sql: str) -> QueryExecutionResult:
        outcome = await execution.execute_sql_for_registered_data_source(
            TEST_DATASOURCE_ID,
            sql,
            policy=policy,
            max_chars=20_000,
        )
        return outcome.result

    summary = await validate_v2_reference_sql(dataset, execute)
    assert summary.valid, summary.failed_case_ids
