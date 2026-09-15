"""Repository-safe ChatBI evaluation dataset v2 contracts and audits.

Version 2 is a human-reviewed frozen dataset.  It reuses the v1 normalized
result oracle and comparison semantics, but adds explicit case metadata and
coverage accounting.  Reference SQL is an evaluator-only oracle and is never
part of a provider request.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from knowledge_scope.chatbi.schemas import QueryExecutionResult

from .chatbi_evaluation import (
    CHATBI_EVALUATION_PROMPT_POLICY_VERSION,
    ChatBIEvaluationError,
    ChatBIEvaluationFailureCategory,
    ExpectedQueryResult,
    _canonical_json,
    _EvaluationModel,
    _sha256_json,
    _validate_json_value,
    _values_equal,
    compare_normalized_result,
    normalize_evaluation_text,
)

CHATBI_EVALUATION_V2_SCHEMA_VERSION = "a5.7-v2"
CHATBI_EVALUATION_V2_STATUS = "human_reviewed_frozen"
CHATBI_EVALUATION_V2_DATASET_VERSION = "a5.7-v2"
CHATBI_DEMO_FIXTURE_VERSION_V2 = "chatbi-demo-v2"
DEFAULT_DATASET_V2 = Path("docs/benchmarks/a5-7-chatbi-eval-v2.json")
DEFAULT_REVIEWER_ARTIFACT_V2 = Path("docs/benchmarks/a5-7-chatbi-eval-v2-review.md")
DEFAULT_FIXTURE_PATH_V2 = Path("tests/fixtures/chatbi_demo_v2.sql")


class ChatBIEvaluationV2Category(StrEnum):
    """Balanced, fixture-oriented v2 query taxonomy."""

    SIMPLE_FILTER_PROJECTION = "simple_filter_projection"
    AGGREGATION = "aggregation"
    GROUP_BY = "group_by"
    ORDERING_TOP_K = "ordering_top_k"
    MULTI_PREDICATE = "multi_predicate"
    JOIN = "join"
    MULTI_TABLE_JOIN = "multi_table_join"
    DATE_FILTER = "date_filter"
    NULL_BOUNDARY = "null_boundary"
    EMPTY = "empty"
    ALIAS_DERIVED = "alias_derived"
    CTE_SET_OPERATION = "cte_set_operation"
    AMBIGUOUS = "ambiguous_question"
    UNSUPPORTED_REQUEST = "unsupported_request"


class ChatBIEvaluationV2Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ChatBIEvaluationV2Behavior(StrEnum):
    ANSWERABLE = "answerable"
    REFUSE = "refuse"
    CLARIFY_OR_REFUSE_SAFELY = "clarify_or_refuse_safely"


class ChatBIEvaluationV2AnswerFact(_EvaluationModel):
    """Structured oracle fact that keeps entity/value relationships intact."""

    kind: Literal["scalar", "row", "empty", "null"]
    values: dict[StrictStr, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fact_shape(self) -> ChatBIEvaluationV2AnswerFact:
        for key, value in self.values.items():
            if not key.strip():
                raise ValueError("structured answer fact keys must be non-blank")
            _validate_json_value(value)
        if self.kind == "empty" and self.values:
            raise ValueError("empty answer facts cannot contain values")
        if self.kind == "scalar" and len(self.values) != 1:
            raise ValueError("scalar answer facts require exactly one value")
        if self.kind in {"row", "null"} and not self.values:
            raise ValueError("row/null answer facts require values")
        if self.kind == "null" and None not in self.values.values():
            raise ValueError("null answer facts must contain a null value")
        return self


V2_CATEGORY_TARGETS: dict[str, int] = {
    ChatBIEvaluationV2Category.SIMPLE_FILTER_PROJECTION.value: 7,
    ChatBIEvaluationV2Category.AGGREGATION.value: 9,
    ChatBIEvaluationV2Category.GROUP_BY.value: 8,
    ChatBIEvaluationV2Category.ORDERING_TOP_K.value: 7,
    ChatBIEvaluationV2Category.MULTI_PREDICATE.value: 7,
    ChatBIEvaluationV2Category.JOIN.value: 10,
    ChatBIEvaluationV2Category.MULTI_TABLE_JOIN.value: 6,
    ChatBIEvaluationV2Category.DATE_FILTER.value: 5,
    ChatBIEvaluationV2Category.NULL_BOUNDARY.value: 4,
    ChatBIEvaluationV2Category.EMPTY.value: 3,
    ChatBIEvaluationV2Category.ALIAS_DERIVED.value: 4,
    ChatBIEvaluationV2Category.CTE_SET_OPERATION.value: 4,
    ChatBIEvaluationV2Category.AMBIGUOUS.value: 3,
    ChatBIEvaluationV2Category.UNSUPPORTED_REQUEST.value: 3,
}


class ChatBIEvaluationCaseV2(_EvaluationModel):
    """One frozen v2 case with explicit positive/negative semantics."""

    dataset_version: Literal["a5.7-v2"] = CHATBI_EVALUATION_V2_DATASET_VERSION
    case_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: Literal["dev", "test"]
    question: StrictStr = Field(min_length=1, max_length=10_000)
    category: ChatBIEvaluationV2Category
    difficulty: ChatBIEvaluationV2Difficulty
    positive: StrictBool
    order_sensitive: StrictBool
    expected_behavior: ChatBIEvaluationV2Behavior
    reference_sql: StrictStr | None = Field(default=None, max_length=100_000)
    expected_result: ExpectedQueryResult | None = None
    expected_answer_facts: list[StrictStr] = Field(default_factory=list)
    structured_answer_facts: list[ChatBIEvaluationV2AnswerFact] = Field(default_factory=list)
    # Retained as review metadata only.  v2 has no repair-rate denominator;
    # runtime repair attempts are the authoritative source for such metrics.
    repair_applicable: StrictBool = False
    expected_failure_category: ChatBIEvaluationFailureCategory | None = None
    evaluation_notes: StrictStr | None = Field(default=None, max_length=500)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evaluation question must be non-blank")
        return normalized

    @field_validator("reference_sql")
    @classmethod
    def normalize_reference_sql(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("expected_answer_facts")
    @classmethod
    def normalize_answer_facts(cls, value: list[str]) -> list[str]:
        normalized = [fact.strip() for fact in value]
        if any(not fact for fact in normalized):
            raise ValueError("expected answer facts must be non-blank")
        normalized_keys = [normalize_evaluation_text(fact) for fact in normalized]
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError("expected answer facts must be unique after normalization")
        return normalized

    @field_validator("structured_answer_facts")
    @classmethod
    def validate_structured_answer_facts(
        cls, value: list[ChatBIEvaluationV2AnswerFact]
    ) -> list[ChatBIEvaluationV2AnswerFact]:
        # Duplicate rows are meaningful result data and must remain representable.
        return value

    @model_validator(mode="after")
    def validate_case_semantics(self) -> ChatBIEvaluationCaseV2:
        is_negative = not self.positive
        if self.positive:
            if self.expected_behavior is not ChatBIEvaluationV2Behavior.ANSWERABLE:
                raise ValueError("positive cases must be answerable")
            if self.reference_sql is None or self.expected_result is None:
                raise ValueError("positive cases require reference_sql and expected_result")
            if self.expected_failure_category is not None:
                raise ValueError("positive cases cannot carry a failure category")
            if self.order_sensitive != self.expected_result.row_order_sensitive:
                raise ValueError("order_sensitive must match the result oracle")
            if not self.structured_answer_facts:
                raise ValueError("positive cases require structured answer facts")
        if is_negative:
            if self.expected_behavior not in {
                ChatBIEvaluationV2Behavior.REFUSE,
                ChatBIEvaluationV2Behavior.CLARIFY_OR_REFUSE_SAFELY,
            }:
                raise ValueError("negative cases must refuse or clarify safely")
            if self.reference_sql is not None or self.expected_result is not None:
                raise ValueError("negative cases cannot carry an executable SQL oracle")
            if self.expected_failure_category is None:
                raise ValueError("negative cases require a failure category")
            if self.order_sensitive:
                raise ValueError("negative cases cannot be order-sensitive")
            if self.structured_answer_facts:
                raise ValueError("negative cases cannot carry structured answer facts")
        return self


class ChatBIEvaluationV2Coverage(_EvaluationModel):
    """Stored counts that are checked against the cases at load time."""

    case_count: StrictInt = Field(ge=0)
    dev_count: StrictInt = Field(ge=0)
    test_count: StrictInt = Field(ge=0)
    positive_count: StrictInt = Field(ge=0)
    negative_count: StrictInt = Field(ge=0)
    category_counts: dict[str, StrictInt]
    difficulty_counts: dict[str, StrictInt]


class ChatBIEvaluationDatasetV2(_EvaluationModel):
    """Exactly 80 frozen cases bound to the small demo fixture."""

    schema_version: Literal["a5.7-v2"] = CHATBI_EVALUATION_V2_SCHEMA_VERSION
    dataset_status: Literal["human_reviewed_frozen"] = CHATBI_EVALUATION_V2_STATUS
    datasource_fixture: StrictStr
    datasource_fixture_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = CHATBI_DEMO_FIXTURE_VERSION_V2
    prompt_policy_version: Literal["a5.7-oracle-separation-v1"] = (
        CHATBI_EVALUATION_PROMPT_POLICY_VERSION
    )
    cases: list[ChatBIEvaluationCaseV2] = Field(min_length=80, max_length=80)
    coverage: ChatBIEvaluationV2Coverage
    dataset_fingerprint: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("datasource_fixture")
    @classmethod
    def require_relative_fixture_path(cls, value: str) -> str:
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("evaluation fixture path must be repository-relative")
        return value

    @model_validator(mode="after")
    def validate_dataset(self) -> ChatBIEvaluationDatasetV2:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("v2 evaluation case IDs must be unique")
        normalized_questions = [normalize_evaluation_text(case.question) for case in self.cases]
        if len(normalized_questions) != len(set(normalized_questions)):
            raise ValueError("v2 questions must be unique after normalization")

        category_counts = {category: 0 for category in V2_CATEGORY_TARGETS}
        difficulty_counts: dict[str, int] = defaultdict(int)
        for case in self.cases:
            category_counts[case.category.value] += 1
            difficulty_counts[case.difficulty.value] += 1
        if category_counts != V2_CATEGORY_TARGETS:
            raise ValueError("v2 category counts do not match the frozen target composition")

        computed_coverage = ChatBIEvaluationV2Coverage(
            case_count=len(self.cases),
            dev_count=sum(case.split == "dev" for case in self.cases),
            test_count=sum(case.split == "test" for case in self.cases),
            positive_count=sum(case.positive for case in self.cases),
            negative_count=sum(not case.positive for case in self.cases),
            category_counts=category_counts,
            difficulty_counts=dict(sorted(difficulty_counts.items())),
        )
        if computed_coverage.case_count != 80:
            raise ValueError("v2 dataset must contain exactly 80 cases")
        if (computed_coverage.dev_count, computed_coverage.test_count) != (50, 30):
            raise ValueError("v2 dataset must contain exactly 50 dev and 30 test cases")
        if (computed_coverage.positive_count, computed_coverage.negative_count) != (74, 6):
            raise ValueError("v2 dataset must contain exactly 74 positive and 6 negative cases")
        if computed_coverage != self.coverage:
            raise ValueError("stored v2 coverage does not match the cases")
        computed_fingerprint = self.fingerprint
        if (
            self.dataset_fingerprint is not None
            and self.dataset_fingerprint != computed_fingerprint
        ):
            raise ValueError("v2 dataset_fingerprint does not match dataset contents")
        return self

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"dataset_fingerprint"})

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.fingerprint_payload())


def load_chatbi_evaluation_dataset_v2(
    path: Path = DEFAULT_DATASET_V2,
) -> ChatBIEvaluationDatasetV2:
    """Load v2 without loading runtime corpus or contacting a provider."""
    try:
        return ChatBIEvaluationDatasetV2.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise ChatBIEvaluationError(f"invalid ChatBI v2 evaluation dataset: {path}") from error


class ChatBIEvaluationV2Audit(_EvaluationModel):
    """Deterministic duplicate, similarity and prompt-leakage audit."""

    duplicate_normalized_question_groups: list[list[StrictStr]]
    near_duplicate_question_pairs: list[list[StrictStr]]
    same_reference_sql_groups: list[list[StrictStr]]
    repeated_expected_result_groups: list[list[StrictStr]]
    dev_test_semantic_concern_pairs: list[list[StrictStr]]
    leakage_issues: list[StrictStr]


def _group_by_value(values: Mapping[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, value in values.items():
        groups[value].append(key)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def audit_chatbi_evaluation_dataset_v2(
    dataset: ChatBIEvaluationDatasetV2,
) -> ChatBIEvaluationV2Audit:
    """Audit likely memorization/leakage issues without judging answer quality."""
    question_keys = {
        case.case_id: normalize_evaluation_text(case.question) for case in dataset.cases
    }
    duplicate_groups = _group_by_value(question_keys)
    near_pairs: list[list[str]] = []
    ordered_cases = sorted(dataset.cases, key=lambda case: case.case_id)
    for index, left in enumerate(ordered_cases):
        left_text = normalize_evaluation_text(left.question)
        for right in ordered_cases[index + 1 :]:
            right_text = normalize_evaluation_text(right.question)
            ratio = difflib.SequenceMatcher(a=left_text, b=right_text).ratio()
            if ratio >= 0.92 and left_text != right_text:
                near_pairs.append([left.case_id, right.case_id])

    sql_values = {
        case.case_id: normalize_evaluation_text(case.reference_sql)
        for case in dataset.cases
        if case.reference_sql is not None
    }
    result_values = {
        case.case_id: _canonical_json(case.expected_result.model_dump(mode="json"))
        for case in dataset.cases
        if case.expected_result is not None
    }
    leakage_issues: list[str] = []
    forbidden_markers = (
        "select ",
        " from ",
        " where ",
        " group by ",
        " order by ",
        " join ",
        "reference_sql",
        "expected_result",
        "a5.7-v2",
        "human_reviewed_frozen",
    )
    for case in dataset.cases:
        question = normalize_evaluation_text(case.question)
        if case.reference_sql and normalize_evaluation_text(case.reference_sql) in question:
            leakage_issues.append(f"{case.case_id}:reference_sql")
        for marker in forbidden_markers:
            if marker in question:
                leakage_issues.append(f"{case.case_id}:metadata:{marker.strip()}")

    semantic_shapes: dict[str, list[ChatBIEvaluationCaseV2]] = defaultdict(list)
    for case in dataset.cases:
        semantic_shapes[_semantic_question_shape(case.question)].append(case)
    semantic_pair_set = {
        tuple(sorted((left.case_id, right.case_id)))
        for cases in semantic_shapes.values()
        for index, left in enumerate(sorted(cases, key=lambda item: item.case_id))
        for right in sorted(cases, key=lambda item: item.case_id)[index + 1 :]
        if left.split != right.split
    }
    semantic_pair_set.update(
        tuple(sorted((left.case_id, right.case_id)))
        for index, left in enumerate(ordered_cases)
        for right in ordered_cases[index + 1 :]
        if left.split != right.split
        and left.category != right.category
        and _strong_cross_category_overlap(left.question, right.question)
    )
    return ChatBIEvaluationV2Audit(
        duplicate_normalized_question_groups=duplicate_groups,
        near_duplicate_question_pairs=sorted(near_pairs),
        same_reference_sql_groups=_group_by_value(sql_values),
        repeated_expected_result_groups=_group_by_value(result_values),
        dev_test_semantic_concern_pairs=[list(pair) for pair in sorted(semantic_pair_set)],
        leakage_issues=sorted(set(leakage_issues)),
    )


_SEMANTIC_VALUE_PATTERNS = (
    re.compile(r"20\d{2}年\s*\d{1,2}月\s*\d{1,2}日"),
    re.compile(r"20\d{2}年\s*\d{1,2}月"),
    re.compile(r"\d+(?:\.\d+)?"),
)
_REGION_NAMES = ("华东", "华南", "华北", "西南", "西北")
_SEMANTIC_METRIC_PATTERNS = {
    "sales_count": re.compile(r"销售(?:笔数|数量|记录|几笔|多少笔)"),
    "sales_total": re.compile(r"(?:销售(?:总额|总金额|合计)|总额)"),
    "distinct_region": re.compile(r"(?:不同|涉及过多少个)销售地区"),
    "sale_date": re.compile(r"(?:销售|成交).{0,8}(?:日期|时间|最早|最近)"),
}


def _semantic_question_shape(question: str) -> str:
    """Build a conservative shape for cross-split memorization review."""
    shape = normalize_evaluation_text(question)
    for region in _REGION_NAMES:
        shape = shape.replace(region, "<region>")
    for pattern in _SEMANTIC_VALUE_PATTERNS:
        shape = pattern.sub("<value>", shape)
    return shape


def _semantic_question_tags(question: str) -> frozenset[str]:
    normalized = normalize_evaluation_text(question)
    tags: set[str] = set()
    if "每位客户" in normalized:
        tags.add("per_customer")
    if any(marker in normalized for marker in ("没有销售", "无销售", "未销售")):
        tags.add("includes_no_sale")
    if "客户编号" in normalized:
        tags.add("customer_id")
    if "客户名称" in normalized or "名称字段" in normalized:
        tags.add("customer_name")
    for tag, pattern in _SEMANTIC_METRIC_PATTERNS.items():
        if pattern.search(normalized):
            tags.add(tag)
    return frozenset(tags)


def _strong_cross_category_overlap(left_question: str, right_question: str) -> bool:
    """Flag only high-signal cross-category template reuse for review.

    Exact normalized shapes are handled by the main audit.  This supplement
    covers common outer-join templates where the metric wording changes but
    the customer/no-sale task shape and metric remain the same.  It is an
    audit signal, not an automatic split or rejection rule.
    """
    left = _semantic_question_tags(left_question)
    right = _semantic_question_tags(right_question)
    shared_outer_join = {"per_customer", "includes_no_sale"} <= left & right
    metric_tags = set(_SEMANTIC_METRIC_PATTERNS)
    # Require the higher-risk total-sales metric to be shared.  Count-only
    # questions that both mention no-sale customers are common but do not by
    # themselves prove that two splits reuse the same aggregation template.
    return (
        shared_outer_join and "sales_total" in (left & right) and bool((left & right) & metric_tags)
    )


class V2ReferenceSQLValidationSummary(_EvaluationModel):
    """Safe aggregate for fixture execution of every positive oracle."""

    positive_case_count: StrictInt = Field(ge=0)
    executed_count: StrictInt = Field(ge=0)
    passed_count: StrictInt = Field(ge=0)
    failed_case_ids: list[StrictStr] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.failed_case_ids and self.executed_count == self.positive_case_count


V2ReferenceExecutor = Callable[[str], Awaitable[QueryExecutionResult]]


async def validate_v2_reference_sql(
    dataset: ChatBIEvaluationDatasetV2,
    execute: V2ReferenceExecutor,
) -> V2ReferenceSQLValidationSummary:
    """Execute and compare every positive oracle through an injected executor."""
    positive_cases = [case for case in dataset.cases if case.positive]
    failed: list[str] = []
    passed = 0
    for case in positive_cases:
        if case.reference_sql is None or case.expected_result is None:
            raise ChatBIEvaluationError(
                f"positive v2 case is missing its SQL oracle: {case.case_id}"
            )
        try:
            actual = await execute(case.reference_sql)
            comparison = compare_normalized_result(actual, case.expected_result)
        except Exception:
            comparison = None
        if comparison is None or not comparison.equivalent:
            failed.append(case.case_id)
        else:
            passed += 1
    return V2ReferenceSQLValidationSummary(
        positive_case_count=len(positive_cases),
        executed_count=len(positive_cases),
        passed_count=passed,
        failed_case_ids=sorted(failed),
    )


def _bounded_result_summary(result: ExpectedQueryResult) -> str:
    bounded_rows = result.rows[:8]
    payload = {
        "columns": result.columns,
        "rows": bounded_rows,
        "omitted_rows": max(0, len(result.rows) - len(bounded_rows)),
        "row_order_sensitive": result.row_order_sensitive,
        "truncated": result.expected_truncated,
        "truncation_reason": result.truncation_reason,
    }
    return _canonical_json(payload)


def structured_result_facts_match(
    actual: QueryExecutionResult,
    expected_facts: Sequence[ChatBIEvaluationV2AnswerFact],
) -> bool | None:
    """Compare labeled result-row facts without an LLM judge.

    This is an auxiliary result-row fact check, not a natural-language answer
    correctness metric.  Execution Accuracy remains the primary v2 metric
    because it compares the complete normalized result.
    """
    if not expected_facts:
        return None
    if actual.state.value != "succeeded":
        return False
    column_names = [column.name for column in actual.columns]
    if len(set(column_names)) != len(column_names):
        return False
    if len(expected_facts) == 1 and expected_facts[0].kind == "empty":
        return not actual.rows
    remaining = list(actual.rows)
    for fact in expected_facts:
        if fact.kind == "empty":
            if actual.rows:
                return False
            continue
        matching_index = next(
            (
                index
                for index, row in enumerate(remaining)
                if _structured_fact_matches_row(column_names, row, fact)
            ),
            None,
        )
        if matching_index is None:
            return False
        remaining.pop(matching_index)
    return True


def _structured_fact_matches_row(
    columns: Sequence[str],
    row: Sequence[object],
    fact: ChatBIEvaluationV2AnswerFact,
) -> bool:
    for column, expected in fact.values.items():
        try:
            actual = row[columns.index(column)]
        except (ValueError, IndexError):
            return False
        numeric = isinstance(expected, (int, float)) and not isinstance(expected, bool)
        if not _values_equal(actual, expected, numeric=numeric):
            return False
    return True


def write_v2_reviewer_artifact(
    dataset: ChatBIEvaluationDatasetV2,
    path: Path = DEFAULT_REVIEWER_ARTIFACT_V2,
) -> None:
    """Write a bounded, human-reviewable artifact separate from production input."""
    lines = [
        "# ChatBI 评测数据集 v2 审阅清单",
        "",
        f"状态: `{dataset.dataset_status}` (已完成最终人工审核与技术冻结审计)",
        f"数据集指纹: `{dataset.fingerprint}`",
        f"fixture 指纹: `{dataset.datasource_fixture_sha256}`",
        "",
        "每个条目用于记录问题自然性、SQL oracle、结果语义和负例行为的冻结审计。",
        "reference SQL 与期望结果不会进入模型输入；条目级人工审核结论已经应用，",  # noqa: RUF001
        "本文件不是待填写的审核表单。",
        "`repair-applicable` 仅是审核元数据，不作为 v2 修复率的分母；修复尝试以运行时记录为准。",  # noqa: RUF001
        "",
    ]
    for case in dataset.cases:
        expected_result = (
            _bounded_result_summary(case.expected_result) if case.expected_result else "(无)"
        )
        expected_failure = (
            case.expected_failure_category.value if case.expected_failure_category else "(无)"
        )
        structured_facts = _canonical_json(
            [fact.model_dump(mode="json") for fact in case.structured_answer_facts]
        )
        if len(structured_facts) > 1_000:
            structured_facts = structured_facts[:997] + "..."
        lines.extend(
            [
                f"## {case.case_id}",
                "",
                f"- split: `{case.split}`",
                f"- category: `{case.category.value}`",
                f"- difficulty: `{case.difficulty.value}`",
                f"- question: {case.question}",
                f"- semantic: `{'positive' if case.positive else 'negative'}` / "
                f"`{case.expected_behavior.value}`",
                f"- reference SQL: `{case.reference_sql or '(无: 应拒绝)'}`",
                f"- expected result: `{expected_result}`",
                f"- order-sensitive: `{case.order_sensitive}`",
                f"- structured answer facts: `{structured_facts}`",
                f"- repair-applicable: `{case.repair_applicable}`",
                f"- expected failure: `{expected_failure}`",
                "- review status: `applied`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "CHATBI_DEMO_FIXTURE_VERSION_V2",
    "CHATBI_EVALUATION_V2_DATASET_VERSION",
    "CHATBI_EVALUATION_V2_SCHEMA_VERSION",
    "CHATBI_EVALUATION_V2_STATUS",
    "DEFAULT_DATASET_V2",
    "DEFAULT_FIXTURE_PATH_V2",
    "DEFAULT_REVIEWER_ARTIFACT_V2",
    "V2_CATEGORY_TARGETS",
    "ChatBIEvaluationCaseV2",
    "ChatBIEvaluationDatasetV2",
    "ChatBIEvaluationV2AnswerFact",
    "ChatBIEvaluationV2Audit",
    "ChatBIEvaluationV2Behavior",
    "ChatBIEvaluationV2Category",
    "ChatBIEvaluationV2Coverage",
    "ChatBIEvaluationV2Difficulty",
    "V2ReferenceSQLValidationSummary",
    "audit_chatbi_evaluation_dataset_v2",
    "load_chatbi_evaluation_dataset_v2",
    "structured_result_facts_match",
    "validate_v2_reference_sql",
    "write_v2_reviewer_artifact",
]
