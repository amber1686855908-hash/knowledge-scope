"""Reproducible offline evaluation for the existing ChatBI pipeline.

The repository-safe dataset stores an oracle SQL statement and expected answer
facts for comparison only.  The provider-backed runner passes only the case
question to :class:`ChatBIAgentService`; the oracle is never part of a model
prompt.  Offline fixtures are explicitly scripted infrastructure scenarios and
must not be described as model-quality measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from knowledge_scope.chatbi.agent import (
    CHATBI_ANALYSIS_PROMPT_VERSION,
    ChatBIAgentLimits,
    ChatBIAgentService,
    ChatBIResult,
    ChatBITraceEvent,
    ChatBITraceName,
    ChatBIUsageSummary,
)
from knowledge_scope.chatbi.credentials import EnvironmentCredentialResolver
from knowledge_scope.chatbi.discovery import (
    SchemaDiscoveryService,
    create_postgres_schema_discovery_service,
)
from knowledge_scope.chatbi.errors import ChatBIErrorCategory
from knowledge_scope.chatbi.execution import (
    ExecutionAdapter,
    PostgresExecutionAdapter,
    SQLExecutionOutcome,
    SQLExecutionService,
)
from knowledge_scope.chatbi.nl2sql import NL2SQLService
from knowledge_scope.chatbi.nl2sql_models import NL2SQL_PROMPT_VERSION, SQLCandidate
from knowledge_scope.chatbi.policy import QueryPolicy, SQLDialect, default_query_policy
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider
from knowledge_scope.chatbi.schema_models import SchemaDiscoveryResult, SchemaObjectKind
from knowledge_scope.chatbi.schemas import (
    ColumnMetadata,
    DataSource,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryTruncationReason,
)
from knowledge_scope.llm import LLMGateway, LLMResult, create_llm_provider
from knowledge_scope.llm.schemas import LLMRequest
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

CHATBI_EVALUATION_SCHEMA_VERSION = "a5.7-v1"
CHATBI_EVALUATION_RUN_SCHEMA_VERSION = "a5.7-run-v2"
CHATBI_EVALUATION_PROMPT_POLICY_VERSION = "a5.7-oracle-separation-v1"
CHATBI_DEMO_FIXTURE_VERSION = "chatbi-demo-v1"
CHATBI_DEMO_SCHEMA = "chatbi_demo"
DEFAULT_DATASET = Path("docs/benchmarks/a5-7-chatbi-eval-v1.json")
DEFAULT_OFFLINE_SCENARIOS = Path("docs/benchmarks/a5-7-chatbi-offline-scenarios.json")
DEFAULT_OUTPUT = Path("data/evaluation/a5-7/run.json")
DEFAULT_FIXTURE_PATH = Path("tests/fixtures/chatbi_demo.sql")
DEFAULT_OFFLINE_DATASOURCE_ID = UUID("00000000-0000-4000-8000-000000000057")
OFFLINE_SCENARIO_FORMAT = "a5.7-offline-scenarios-v1"
_DEMO_CUSTOMERS_PROBE_SQL = (
    "SELECT customer_id::text AS customer_id, customer_name "
    "FROM chatbi_demo.customers ORDER BY customer_id"
)
_DEMO_SALES_PROBE_SQL = (
    "SELECT sale_id::text AS sale_id, customer_id::text AS customer_id, "
    "sold_on::text AS sold_on, region, amount::text AS amount "
    "FROM chatbi_demo.sales ORDER BY sale_id"
)
_DEMO_DATA_PROBE_EXPECTATIONS: tuple[
    tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]], ...
] = (
    (
        _DEMO_CUSTOMERS_PROBE_SQL,
        ("customer_id", "customer_name"),
        (("1", "示例客户 A"), ("2", "示例客户 B")),
    ),
    (
        _DEMO_SALES_PROBE_SQL,
        ("sale_id", "customer_id", "sold_on", "region", "amount"),
        (
            ("1", "1", "2026-01-05", "华东", "1200.00"),
            ("2", "2", "2026-01-12", "华南", "860.50"),
            ("3", "1", "2026-02-03", "华东", "1540.25"),
        ),
    ),
)


class ChatBIEvaluationError(RuntimeError):
    """Raised when a frozen evaluation input or runtime artifact is invalid."""


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class ChatBIEvaluationCategory(StrEnum):
    """Small fixture-oriented query taxonomy."""

    FILTERING = "filtering"
    PROJECTION = "projection"
    AGGREGATION = "aggregation"
    GROUP_BY = "group_by"
    ORDERING_TOP_K = "ordering_top_k"
    DATE_FILTER = "date_filter"
    JOIN = "join"
    PREDICATES = "predicates"
    NULL_SEMANTICS = "null_semantics"
    CTE = "cte"
    SET_OPERATION = "set_operation"
    EMPTY = "empty"
    ALIAS_DERIVED = "alias_derived"
    AMBIGUOUS = "ambiguous_question"
    DIFFICULT_ANSWERABLE = "difficult_answerable"
    NEGATIVE = "unsupported_request"


class ChatBIEvaluationFailureCategory(StrEnum):
    """Controlled categories used in offline evaluation diagnostics."""

    SCHEMA_CONTEXT = "schema_context"
    GENERATION_PARSE = "generation_parse"
    GENERATION_SEMANTIC = "generation_semantic"
    VALIDATION = "validation"
    UNSUPPORTED_SQL = "unsupported_sql"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RESULT_MISMATCH = "result_mismatch"
    REPAIR_EXHAUSTED = "repair_exhausted"
    ANALYSIS = "analysis"
    ANSWER_MISMATCH = "answer_mismatch"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    UNSUPPORTED_REQUEST = "unsupported_request"


EvaluationSplit = Literal["dev", "test"]
_TRUNCATION_REASONS = ("row_limit", "payload_bytes")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ChatBIEvaluationError(
            "evaluation data cannot be serialized deterministically"
        ) from error


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_evaluation_text(value: str) -> str:
    """Normalize text for duplicate-question and answer-fact comparisons."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evaluation result values must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("evaluation JSON object keys must be strings")
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("evaluation result values must be JSON-safe")


class ExpectedQueryResult(_EvaluationModel):
    """Trusted normalized-result oracle, never sent to generation or analysis."""

    columns: list[StrictStr] = Field(min_length=1)
    rows: list[list[object]] = Field(default_factory=list)
    numeric_columns: list[StrictInt] = Field(default_factory=list)
    row_order_sensitive: StrictBool = True
    expected_truncated: StrictBool = False
    truncation_reason: Literal["row_limit", "payload_bytes"] | None = None

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: list[str]) -> list[str]:
        if any(not column.strip() for column in value):
            raise ValueError("expected result columns must be non-blank")
        return [column.strip() for column in value]

    @model_validator(mode="after")
    def validate_result(self) -> ExpectedQueryResult:
        if any(index < 0 or index >= len(self.columns) for index in self.numeric_columns):
            raise ValueError("numeric_columns contains an invalid column index")
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("numeric_columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("expected result rows must match the column count")
        for row in self.rows:
            for value in row:
                _validate_json_value(value)
        if self.expected_truncated != (self.truncation_reason is not None):
            raise ValueError("truncation_reason must match expected_truncated")
        return self


class ChatBIEvaluationCase(_EvaluationModel):
    """One frozen question with a comparison-only SQL oracle."""

    case_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    split: EvaluationSplit
    question: StrictStr = Field(min_length=1, max_length=10_000)
    category: ChatBIEvaluationCategory
    reference_sql: StrictStr | None = Field(default=None, max_length=100_000)
    expected_result: ExpectedQueryResult | None = None
    expected_answer_facts: list[StrictStr] = Field(default_factory=list)
    repair_applicable: StrictBool = False
    expected_failure_category: ChatBIEvaluationFailureCategory | None = None

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
        if len(normalized) != len({normalize_evaluation_text(fact) for fact in normalized}):
            raise ValueError("expected answer facts must be unique after normalization")
        return normalized

    @model_validator(mode="after")
    def validate_oracle_shape(self) -> ChatBIEvaluationCase:
        is_negative = self.expected_failure_category is not None
        if is_negative and (self.reference_sql is not None or self.expected_result is not None):
            raise ValueError("negative cases cannot carry an executable SQL oracle")
        if not is_negative and (self.reference_sql is None or self.expected_result is None):
            raise ValueError("answerable cases require both reference_sql and expected_result")
        return self


class ChatBIEvaluationDataset(_EvaluationModel):
    """Frozen repository-safe dataset with a deterministic content fingerprint."""

    schema_version: Literal["a5.7-v1"] = CHATBI_EVALUATION_SCHEMA_VERSION
    datasource_fixture: StrictStr
    datasource_fixture_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_policy_version: Literal["a5.7-oracle-separation-v1"] = (
        CHATBI_EVALUATION_PROMPT_POLICY_VERSION
    )
    cases: list[ChatBIEvaluationCase] = Field(min_length=1)
    dataset_fingerprint: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_dataset(self) -> ChatBIEvaluationDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        normalized_questions = [normalize_evaluation_text(case.question) for case in self.cases]
        if len(normalized_questions) != len(set(normalized_questions)):
            raise ValueError("evaluation questions must be unique after normalization")
        computed = self.fingerprint
        if self.dataset_fingerprint is not None and self.dataset_fingerprint != computed:
            raise ValueError("dataset_fingerprint does not match the frozen dataset contents")
        return self

    def fingerprint_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"dataset_fingerprint"})

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.fingerprint_payload())


def load_chatbi_evaluation_dataset(path: Path = DEFAULT_DATASET) -> ChatBIEvaluationDataset:
    """Load and validate one frozen dataset without loading any runtime corpus."""
    try:
        return ChatBIEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise ChatBIEvaluationError(f"invalid ChatBI evaluation dataset: {path}") from error


def verify_fixture_fingerprint(dataset: ChatBIEvaluationDataset, fixture_path: Path) -> None:
    """Verify the small demo fixture identity used by the repository-safe oracle."""
    try:
        actual = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ChatBIEvaluationError(f"evaluation fixture is unavailable: {fixture_path}") from error
    if actual != dataset.datasource_fixture_sha256:
        raise ChatBIEvaluationError(
            "evaluation fixture fingerprint does not match the frozen dataset"
        )


def assert_no_oracle_leakage(
    case: ChatBIEvaluationCase,
    messages: Sequence[Mapping[str, str] | object],
) -> None:
    """Fail a test/adapter audit if comparison-only data enters model messages."""

    def field(message: Mapping[str, str] | object, name: str) -> str:
        if isinstance(message, Mapping):
            value = message.get(name, "")
        else:
            value = getattr(message, name, "")
        return value if isinstance(value, str) else ""

    serialized = "\n".join(
        f"{field(message, 'role')}:{field(message, 'content')}" for message in messages
    )
    normalized_prompt = normalize_evaluation_text(serialized)
    if case.reference_sql and normalize_evaluation_text(case.reference_sql) in normalized_prompt:
        raise ChatBIEvaluationError(f"reference SQL leaked into model input for {case.case_id}")
    normalized_question = normalize_evaluation_text(case.question)
    for fact in case.expected_answer_facts:
        normalized_fact = normalize_evaluation_text(fact)
        # A fact can be part of the user's question (for example, a requested
        # region). It is only an oracle leak when the evaluator-only fact is
        # introduced beyond the question itself.
        if normalized_fact not in normalized_question and normalized_fact in normalized_prompt:
            raise ChatBIEvaluationError(
                f"expected answer fact leaked into model input for {case.case_id}"
            )


class ResultComparison(_EvaluationModel):
    """Explainable semantic comparison of an actual bounded result and the oracle."""

    equivalent: StrictBool
    reason: StrictStr
    actual_row_count: StrictInt = Field(ge=0)
    expected_row_count: StrictInt = Field(ge=0)


def _decimal_value(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            return None
        return parsed if parsed.is_finite() else None
    return None


def _values_equal(actual: object, expected: object, *, numeric: bool) -> bool:
    if numeric:
        actual_decimal = _decimal_value(actual)
        expected_decimal = _decimal_value(expected)
        if actual_decimal is None or expected_decimal is None:
            return False
        difference = abs(actual_decimal - expected_decimal)
        scale = max(abs(actual_decimal), abs(expected_decimal), Decimal(1))
        return difference <= max(Decimal("0.000001"), scale * Decimal("1e-9"))
    if isinstance(actual, float) and not math.isfinite(actual):
        return False
    if isinstance(expected, float) and not math.isfinite(expected):
        return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_equal(left, right, numeric=False)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_equal(actual[key], expected[key], numeric=False) for key in actual
        )
    return type(actual) is type(expected) and actual == expected


def compare_normalized_result(
    actual: QueryExecutionResult | ChatBIResult,
    expected: ExpectedQueryResult,
) -> ResultComparison:
    """Compare rows without requiring exact SQL or alias names.

    Column order and row values remain positional.  Row order is ignored only
    for cases whose oracle explicitly marks it irrelevant; arrays remain
    order-sensitive.  Numeric columns accept PostgreSQL Decimal strings,
    integers and finite floats with a small documented tolerance.
    """
    actual_state = (
        actual.state if isinstance(actual, QueryExecutionResult) else actual.execution_status
    )
    actual_columns = actual.columns
    if isinstance(actual, QueryExecutionResult):
        actual_rows = actual.rows
        actual_truncated = actual.truncated
        actual_reason = actual.truncation_reason.value if actual.truncation_reason else None
    else:
        # ChatBIResult deliberately exposes only bounded metadata. An evaluator
        # must provide the runner-side QueryExecutionResult to compare rows.
        return ResultComparison(
            equivalent=False,
            reason="materialized rows are unavailable in ChatBIResult",
            actual_row_count=actual.row_count,
            expected_row_count=len(expected.rows),
        )
    if actual_state is not QueryLifecycleState.SUCCEEDED:
        return ResultComparison(
            equivalent=False,
            reason="execution did not succeed",
            actual_row_count=0,
            expected_row_count=len(expected.rows),
        )
    if len(actual_columns) != len(expected.columns):
        return ResultComparison(
            equivalent=False,
            reason="column count differs",
            actual_row_count=len(actual_rows),
            expected_row_count=len(expected.rows),
        )
    if (
        actual_truncated != expected.expected_truncated
        or actual_reason != expected.truncation_reason
    ):
        return ResultComparison(
            equivalent=False,
            reason="truncation metadata differs",
            actual_row_count=len(actual_rows),
            expected_row_count=len(expected.rows),
        )
    if len(actual_rows) != len(expected.rows):
        return ResultComparison(
            equivalent=False,
            reason="row count differs",
            actual_row_count=len(actual_rows),
            expected_row_count=len(expected.rows),
        )

    def row_matches(actual_row: Sequence[object], expected_row: Sequence[object]) -> bool:
        return all(
            _values_equal(
                actual_value,
                expected_value,
                numeric=index in expected.numeric_columns,
            )
            for index, (actual_value, expected_value) in enumerate(
                zip(actual_row, expected_row, strict=True)
            )
        )

    if expected.row_order_sensitive:
        equivalent = all(
            row_matches(actual_row, expected_row)
            for actual_row, expected_row in zip(actual_rows, expected.rows, strict=True)
        )
    else:
        unmatched = list(expected.rows)
        equivalent = True
        for actual_row in actual_rows:
            match_index = next(
                (
                    index
                    for index, expected_row in enumerate(unmatched)
                    if row_matches(actual_row, expected_row)
                ),
                None,
            )
            if match_index is None:
                equivalent = False
                break
            unmatched.pop(match_index)
        equivalent = equivalent and not unmatched
    return ResultComparison(
        equivalent=equivalent,
        reason="equivalent" if equivalent else "row values differ",
        actual_row_count=len(actual_rows),
        expected_row_count=len(expected.rows),
    )


def answer_facts_match(answer: str | None, expected_facts: Sequence[str]) -> bool | None:
    """Compare only explicitly labelled deterministic answer facts."""
    if not expected_facts:
        return None
    if answer is None:
        return False
    normalized_answer = normalize_evaluation_text(answer)
    return all(normalize_evaluation_text(fact) in normalized_answer for fact in expected_facts)


class RepairOutcome(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    NOT_ATTEMPTED = "not_attempted"
    REPAIRED = "repaired"
    EXHAUSTED = "exhausted"


class ChatBIStageTimings(_EvaluationModel):
    """Wall-clock stage timings; ``None`` means the runner did not measure it."""

    schema_prep_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    validation_ms: float | None = Field(default=None, ge=0)
    execution_ms: float | None = Field(default=None, ge=0)
    repair_ms: float | None = Field(default=None, ge=0)
    analysis_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


EvaluationStageStatus = Literal["not_attempted", "passed", "failed"]


class EvaluationStageState(_EvaluationModel):
    """Explicit terminal status for each stage of one evaluated case."""

    schema_preparation: EvaluationStageStatus = "not_attempted"
    generation: EvaluationStageStatus = "not_attempted"
    validation: EvaluationStageStatus = "not_attempted"
    execution: EvaluationStageStatus = "not_attempted"
    analysis: EvaluationStageStatus = "not_attempted"
    repair_generation: EvaluationStageStatus = "not_attempted"


class EvaluationQueryPolicy(_EvaluationModel):
    """Safe, non-secret copy of the policy values that affect evaluation."""

    dialect: StrictStr
    read_only: StrictBool
    max_rows: StrictInt = Field(ge=1)
    max_result_bytes: StrictInt = Field(ge=2)
    max_cell_bytes: StrictInt = Field(ge=1)
    max_nested_value_depth: StrictInt = Field(ge=1)
    max_collection_items: StrictInt = Field(ge=1)
    statement_timeout_ms: StrictInt = Field(ge=100)
    allowed_schemas: tuple[StrictStr, ...]
    denied_schemas: tuple[StrictStr, ...]
    allow_views: StrictBool
    max_statement_count: StrictInt = Field(ge=1, le=1)


class EvaluationAgentLimits(_EvaluationModel):
    """Non-secret copy of the finite Agent loop limits."""

    max_sql_attempts: StrictInt = Field(ge=1)
    max_repair_attempts: StrictInt = Field(ge=0)
    max_steps: StrictInt = Field(ge=1)
    max_llm_calls: StrictInt = Field(ge=1)
    analysis_max_tokens: StrictInt = Field(ge=1)


class EvaluationRuntimeConfiguration(_EvaluationModel):
    """Configuration provenance for provider-backed evaluation runs."""

    max_chars: StrictInt = Field(ge=1)
    nl2sql_max_tokens: StrictInt = Field(ge=1)
    provider_timeout_seconds: StrictFloat = Field(gt=0)
    provider_max_retries: StrictInt = Field(ge=0)
    nl2sql_prompt_version: StrictStr = Field(min_length=1, max_length=64)
    analysis_prompt_version: StrictStr = Field(min_length=1, max_length=64)
    agent_limits: EvaluationAgentLimits
    query_policy: EvaluationQueryPolicy


class EvaluationRunProvenance(_EvaluationModel):
    """Reproducibility metadata with no DSN, password, key, or raw SQL."""

    run_schema_version: Literal["a5.7-run-v2"] = CHATBI_EVALUATION_RUN_SCHEMA_VERSION
    benchmark_mode: Literal["offline", "provider"]
    dataset_version: Literal["a5.7-v1"]
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v1"] = CHATBI_DEMO_FIXTURE_VERSION
    fixture_path: StrictStr = Field(min_length=1, max_length=500)
    fixture_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_data_fingerprint: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID | None
    git_revision: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    git_dirty: StrictBool
    provider: StrictStr | None = Field(default=None, min_length=1, max_length=64)
    model: StrictStr | None = Field(default=None, min_length=1, max_length=255)
    configuration: EvaluationRuntimeConfiguration | None = None
    case_count: StrictInt = Field(ge=0)

    @field_validator("fixture_path")
    @classmethod
    def require_relative_fixture_path(cls, value: str) -> str:
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("evaluation fixture path must be repository-relative")
        return value

    @model_validator(mode="after")
    def validate_provider_provenance(self) -> EvaluationRunProvenance:
        if self.benchmark_mode == "provider":
            if self.datasource_id is None:
                raise ValueError("provider provenance requires a datasource identity")
            if self.fixture_data_fingerprint is None:
                raise ValueError("provider provenance requires a fixture data fingerprint")
            if self.provider is None or self.model is None or self.configuration is None:
                raise ValueError("provider provenance requires provider configuration")
            if self.git_revision is None:
                raise ValueError("provider provenance requires a Git revision")
        return self


class NumericSummary(_EvaluationModel):
    sample_count: StrictInt = Field(ge=0)
    known_count: StrictInt = Field(ge=0)
    total: float | None = Field(default=None, ge=0)
    mean: float | None = Field(default=None, ge=0)
    p50: float | None = Field(default=None, ge=0)
    p95: float | None = Field(default=None, ge=0)


class UsageAggregate(_EvaluationModel):
    """Usage totals and distribution summaries copied from ChatBIResult only."""

    case_count: StrictInt = Field(ge=0)
    llm_calls_total: StrictInt = Field(ge=0)
    provider_attempts_total: StrictInt = Field(ge=0)
    input_tokens: NumericSummary
    output_tokens: NumericSummary
    estimated_cost: None = None


class EvaluationCaseRecord(_EvaluationModel):
    """Safe per-case diagnostics; no oracle SQL, prompt, or provider response."""

    case_id: StrictStr
    split: EvaluationSplit
    category: ChatBIEvaluationCategory
    positive_case: StrictBool
    stages: EvaluationStageState
    execution_status: QueryLifecycleState
    execution_equivalent: StrictBool | None
    answer_fact_coverage: StrictBool | None
    expected_failure_match: StrictBool | None
    end_to_end_success: StrictBool | None
    repair_outcome: RepairOutcome
    failure_category: ChatBIEvaluationFailureCategory | None
    row_count: StrictInt = Field(ge=0)
    truncated: StrictBool
    truncation_reason: QueryTruncationReason | None
    metrics: dict[str, float]
    latency: ChatBIStageTimings
    usage: ChatBIUsageSummary


class LatencyAggregate(_EvaluationModel):
    schema_prep_ms: NumericSummary
    generation_ms: NumericSummary
    validation_ms: NumericSummary
    execution_ms: NumericSummary
    repair_ms: NumericSummary
    analysis_ms: NumericSummary
    total_ms: NumericSummary


class OfflineVerificationAggregate(_EvaluationModel):
    """Deterministic scenario checks, deliberately separate from model quality."""

    scenario_match_count: StrictInt = Field(ge=0)
    scenario_match_rate: float | None = Field(default=None, ge=0, le=1)
    comparator_checks: StrictInt = Field(ge=0)
    comparator_pass_count: StrictInt = Field(ge=0)
    taxonomy_checks: StrictInt = Field(ge=0)
    taxonomy_pass_count: StrictInt = Field(ge=0)
    repair_scenario_checks: StrictInt = Field(ge=0)
    repair_scenario_pass_count: StrictInt = Field(ge=0)


class ProviderQualityAggregate(_EvaluationModel):
    """Provider-observed quality and stage metrics; never populated offline."""

    generation_attempted_count: StrictInt = Field(ge=0)
    generation_parseable_count: StrictInt = Field(ge=0)
    generation_parseable_rate: float | None = Field(default=None, ge=0, le=1)
    generation_not_attempted_count: StrictInt = Field(ge=0)
    validation_attempted_count: StrictInt = Field(ge=0)
    validator_accepted_count: StrictInt = Field(ge=0)
    validator_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    execution_equivalent_count: StrictInt = Field(ge=0)
    execution_accuracy: float | None = Field(default=None, ge=0, le=1)
    answer_fact_coverage_count: StrictInt = Field(ge=0)
    answer_fact_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    end_to_end_success_count: StrictInt = Field(ge=0)
    end_to_end_success_rate: float | None = Field(default=None, ge=0, le=1)
    failure_categories: dict[str, StrictInt]


class EvaluationAggregate(_EvaluationModel):
    case_count: StrictInt = Field(ge=0)
    positive_case_count: StrictInt = Field(ge=0)
    negative_case_count: StrictInt = Field(ge=0)
    repair_applicable_count: StrictInt = Field(ge=0)
    repair_attempted_count: StrictInt = Field(ge=0)
    repair_success_count: StrictInt = Field(ge=0)
    repair_exhausted_count: StrictInt = Field(ge=0)
    negative_failure_match_count: StrictInt = Field(ge=0)
    offline_verification: OfflineVerificationAggregate | None = None
    provider_quality: ProviderQualityAggregate | None = None
    latency: LatencyAggregate
    usage: UsageAggregate


class ChatBIEvaluationRun(_EvaluationModel):
    """One reproducibility record for an offline or explicit provider run."""

    run_id: UUID
    artifact_schema_version: Literal["a5.7-run-v2"] = CHATBI_EVALUATION_RUN_SCHEMA_VERSION
    started_at: datetime
    completed_at: datetime
    mode: Literal["offline", "provider"]
    dataset_version: Literal["a5.7-v1"]
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID | None
    git_revision: StrictStr | None = None
    case_count: StrictInt = Field(ge=0)
    records: list[EvaluationCaseRecord]
    aggregates: dict[str, EvaluationAggregate]
    provenance: EvaluationRunProvenance
    quality_claim: Literal[
        "offline_infrastructure_only", "provider_quality_pending", "provider_observed"
    ]

    @model_validator(mode="after")
    def validate_run(self) -> ChatBIEvaluationRun:
        if self.case_count != len(self.records):
            raise ValueError("evaluation run case_count must match its records")
        case_ids = [record.case_id for record in self.records]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation run records must have unique case IDs")
        if self.provenance.benchmark_mode != self.mode:
            raise ValueError("run provenance mode must match evaluation mode")
        if self.provenance.dataset_version != self.dataset_version:
            raise ValueError("run provenance dataset version must match evaluation run")
        if self.provenance.dataset_fingerprint != self.dataset_fingerprint:
            raise ValueError("run provenance dataset fingerprint must match evaluation run")
        if self.provenance.datasource_id != self.datasource_id:
            raise ValueError("run provenance datasource identity must match evaluation run")
        if self.provenance.git_revision != self.git_revision:
            raise ValueError("run git revision must match its provenance")
        if self.provenance.case_count != self.case_count:
            raise ValueError("run provenance case count must match evaluation run")
        expected_claim = (
            "offline_infrastructure_only" if self.mode == "offline" else "provider_observed"
        )
        if self.quality_claim != expected_claim:
            raise ValueError("evaluation quality claim does not match its mode")
        return self


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


def _numeric_summary(values: Sequence[float | None]) -> NumericSummary:
    known = [value for value in values if value is not None]
    return NumericSummary(
        sample_count=len(values),
        known_count=len(known),
        total=sum(known) if known else None,
        mean=sum(known) / len(known) if known else None,
        p50=_percentile(known, 0.50),
        p95=_percentile(known, 0.95),
    )


def _usage_aggregate(records: Sequence[EvaluationCaseRecord]) -> UsageAggregate:
    return UsageAggregate(
        case_count=len(records),
        llm_calls_total=sum(record.usage.llm_calls for record in records),
        provider_attempts_total=sum(record.usage.provider_attempts for record in records),
        input_tokens=_numeric_summary([record.usage.input_tokens for record in records]),
        output_tokens=_numeric_summary([record.usage.output_tokens for record in records]),
    )


def _latency_aggregate(records: Sequence[EvaluationCaseRecord]) -> LatencyAggregate:
    fields = (
        "schema_prep_ms",
        "generation_ms",
        "validation_ms",
        "execution_ms",
        "repair_ms",
        "analysis_ms",
        "total_ms",
    )
    summaries = {
        field: _numeric_summary([getattr(record.latency, field) for record in records])
        for field in fields
    }
    return LatencyAggregate(**summaries)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _evaluation_policy(policy: QueryPolicy) -> EvaluationQueryPolicy:
    return EvaluationQueryPolicy.model_validate(policy.model_dump(mode="json"))


def _evaluation_configuration(
    settings: Settings,
    policy: QueryPolicy,
    *,
    max_chars: int,
) -> EvaluationRuntimeConfiguration:
    limits = ChatBIAgentLimits.from_settings(settings)
    return EvaluationRuntimeConfiguration(
        max_chars=max_chars,
        nl2sql_max_tokens=settings.chatbi_nl2sql_max_tokens,
        provider_timeout_seconds=settings.llm_timeout_seconds,
        provider_max_retries=settings.llm_max_retries,
        nl2sql_prompt_version=NL2SQL_PROMPT_VERSION,
        analysis_prompt_version=CHATBI_ANALYSIS_PROMPT_VERSION,
        agent_limits=EvaluationAgentLimits.model_validate(limits.model_dump(mode="json")),
        query_policy=_evaluation_policy(policy),
    )


def _build_minimal_provenance(
    dataset: ChatBIEvaluationDataset,
    *,
    mode: Literal["offline", "provider"],
    datasource_id: UUID | None,
    git_revision: str | None,
    case_count: int,
    git_dirty: bool | None = None,
) -> EvaluationRunProvenance:
    if mode == "provider":
        raise ChatBIEvaluationError("provider evaluation requires complete provider provenance")
    revision = git_revision if git_revision is not None else current_git_revision()
    return EvaluationRunProvenance(
        benchmark_mode=mode,
        dataset_version=dataset.schema_version,
        dataset_fingerprint=dataset.fingerprint,
        fixture_path=dataset.datasource_fixture,
        fixture_sha256=dataset.datasource_fixture_sha256,
        datasource_id=datasource_id,
        git_revision=revision,
        git_dirty=current_git_dirty() if git_dirty is None else git_dirty,
        provider="offline-fixture",
        model="offline-fixture-model",
        case_count=case_count,
    )


def _build_provider_provenance(
    dataset: ChatBIEvaluationDataset,
    *,
    settings: Settings,
    policy: QueryPolicy,
    max_chars: int,
    datasource_id: UUID,
    fixture_data_fingerprint: str,
    git_revision: str | None,
    git_dirty: bool | None,
    case_count: int,
) -> EvaluationRunProvenance:
    revision = git_revision if git_revision is not None else current_git_revision()
    return EvaluationRunProvenance(
        benchmark_mode="provider",
        dataset_version=dataset.schema_version,
        dataset_fingerprint=dataset.fingerprint,
        fixture_path=dataset.datasource_fixture,
        fixture_sha256=dataset.datasource_fixture_sha256,
        fixture_data_fingerprint=fixture_data_fingerprint,
        datasource_id=datasource_id,
        git_revision=revision,
        git_dirty=current_git_dirty() if git_dirty is None else git_dirty,
        provider=settings.llm_provider,
        model=settings.llm_model,
        configuration=_evaluation_configuration(settings, policy, max_chars=max_chars),
        case_count=case_count,
    )


def _provider_quality_aggregate(
    records: Sequence[EvaluationCaseRecord],
) -> ProviderQualityAggregate:
    positive = [record for record in records if record.positive_case]
    answer_fact_records = [record for record in positive if record.answer_fact_coverage is not None]
    failure_categories = Counter(
        record.failure_category.value for record in records if record.failure_category is not None
    )
    generation_attempted = [
        record for record in records if record.stages.generation != "not_attempted"
    ]
    validation_attempted = [
        record for record in records if record.stages.validation != "not_attempted"
    ]
    generation_parseable_count = sum(
        record.stages.generation == "passed" for record in generation_attempted
    )
    validator_accepted_count = sum(
        record.stages.validation == "passed" for record in validation_attempted
    )
    return ProviderQualityAggregate(
        generation_attempted_count=len(generation_attempted),
        generation_parseable_count=generation_parseable_count,
        generation_parseable_rate=_rate(generation_parseable_count, len(generation_attempted)),
        generation_not_attempted_count=len(records) - len(generation_attempted),
        validation_attempted_count=len(validation_attempted),
        validator_accepted_count=validator_accepted_count,
        validator_acceptance_rate=_rate(validator_accepted_count, len(validation_attempted)),
        execution_equivalent_count=sum(record.execution_equivalent is True for record in positive),
        execution_accuracy=_rate(
            sum(record.execution_equivalent is True for record in positive), len(positive)
        ),
        answer_fact_coverage_count=sum(
            record.answer_fact_coverage is True for record in answer_fact_records
        ),
        answer_fact_coverage_rate=_rate(
            sum(record.answer_fact_coverage is True for record in answer_fact_records),
            len(answer_fact_records),
        ),
        end_to_end_success_count=sum(record.end_to_end_success is True for record in positive),
        end_to_end_success_rate=_rate(
            sum(record.end_to_end_success is True for record in positive), len(positive)
        ),
        failure_categories=dict(sorted(failure_categories.items())),
    )


def _offline_verification_aggregate(
    records: Sequence[EvaluationCaseRecord],
) -> OfflineVerificationAggregate:
    positive = [record for record in records if record.positive_case]
    negative = [record for record in records if not record.positive_case]
    comparator_pass_count = sum(record.execution_equivalent is True for record in positive)
    taxonomy_pass_count = sum(record.expected_failure_match is True for record in negative)
    repair_records = [
        record for record in records if record.repair_outcome is not RepairOutcome.NOT_APPLICABLE
    ]
    repair_pass_count = sum(
        record.repair_outcome is RepairOutcome.REPAIRED for record in repair_records
    )
    scenario_match_count = comparator_pass_count + taxonomy_pass_count
    return OfflineVerificationAggregate(
        scenario_match_count=scenario_match_count,
        scenario_match_rate=_rate(scenario_match_count, len(records)),
        comparator_checks=len(positive),
        comparator_pass_count=comparator_pass_count,
        taxonomy_checks=len(negative),
        taxonomy_pass_count=taxonomy_pass_count,
        repair_scenario_checks=len(repair_records),
        repair_scenario_pass_count=repair_pass_count,
    )


def _aggregate_records(
    records: Sequence[EvaluationCaseRecord],
    *,
    mode: Literal["offline", "provider"],
) -> EvaluationAggregate:
    positive = [record for record in records if record.positive_case]
    repaired = [record for record in records if record.repair_outcome is RepairOutcome.REPAIRED]
    exhausted = [record for record in records if record.repair_outcome is RepairOutcome.EXHAUSTED]
    return EvaluationAggregate(
        case_count=len(records),
        positive_case_count=len(positive),
        negative_case_count=len(records) - len(positive),
        repair_applicable_count=sum(
            record.repair_outcome is not RepairOutcome.NOT_APPLICABLE for record in records
        ),
        repair_attempted_count=sum(
            record.repair_outcome in {RepairOutcome.REPAIRED, RepairOutcome.EXHAUSTED}
            for record in records
        ),
        repair_success_count=len(repaired),
        repair_exhausted_count=len(exhausted),
        negative_failure_match_count=sum(
            record.expected_failure_match is True for record in records
        ),
        offline_verification=(
            _offline_verification_aggregate(records) if mode == "offline" else None
        ),
        provider_quality=(_provider_quality_aggregate(records) if mode == "provider" else None),
        latency=_latency_aggregate(records),
        usage=_usage_aggregate(records),
    )


def _map_failure_category(result: ChatBIResult) -> ChatBIEvaluationFailureCategory | None:
    if (
        result.execution_status is QueryLifecycleState.SUCCEEDED
        and result.error_category is not ChatBIErrorCategory.ANALYSIS_FAILED
    ):
        return None
    category = result.error_category
    if category is None:
        return ChatBIEvaluationFailureCategory.EXECUTION
    if category is ChatBIErrorCategory.SCHEMA_DISCOVERY_FAILED:
        return ChatBIEvaluationFailureCategory.SCHEMA_CONTEXT
    if category is ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT:
        return ChatBIEvaluationFailureCategory.GENERATION_PARSE
    if category in {
        ChatBIErrorCategory.UNKNOWN_TABLE,
        ChatBIErrorCategory.UNKNOWN_COLUMN,
    }:
        return ChatBIEvaluationFailureCategory.GENERATION_SEMANTIC
    if category in {
        ChatBIErrorCategory.INVALID_QUERY,
        ChatBIErrorCategory.SQL_PARSE_ERROR,
        ChatBIErrorCategory.UNSAFE_QUERY,
        ChatBIErrorCategory.POLICY_VIOLATION,
        ChatBIErrorCategory.UNSUPPORTED_DIALECT,
    }:
        return ChatBIEvaluationFailureCategory.VALIDATION
    if category is ChatBIErrorCategory.EXECUTION_TIMEOUT:
        return ChatBIEvaluationFailureCategory.TIMEOUT
    if category is ChatBIErrorCategory.AGENT_LIMIT_EXCEEDED:
        return ChatBIEvaluationFailureCategory.REPAIR_EXHAUSTED
    if category is ChatBIErrorCategory.ANALYSIS_FAILED:
        return ChatBIEvaluationFailureCategory.ANALYSIS
    return ChatBIEvaluationFailureCategory.EXECUTION


def _expected_failure_match(
    case: ChatBIEvaluationCase,
    result: ChatBIResult,
) -> bool | None:
    if case.expected_failure_category is None:
        return None
    actual = result.error_category
    if actual is None or result.execution_status is QueryLifecycleState.SUCCEEDED:
        return False
    allowed: dict[ChatBIEvaluationFailureCategory, frozenset[ChatBIErrorCategory]] = {
        ChatBIEvaluationFailureCategory.AMBIGUOUS_QUESTION: frozenset(
            {ChatBIErrorCategory.INVALID_QUERY, ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT}
        ),
        ChatBIEvaluationFailureCategory.UNSUPPORTED_REQUEST: frozenset(
            {
                ChatBIErrorCategory.INVALID_QUERY,
                ChatBIErrorCategory.UNSAFE_QUERY,
                ChatBIErrorCategory.POLICY_VIOLATION,
            }
        ),
        ChatBIEvaluationFailureCategory.VALIDATION: frozenset(
            {
                ChatBIErrorCategory.UNSAFE_QUERY,
                ChatBIErrorCategory.POLICY_VIOLATION,
                ChatBIErrorCategory.UNKNOWN_TABLE,
                ChatBIErrorCategory.UNKNOWN_COLUMN,
            }
        ),
        ChatBIEvaluationFailureCategory.UNSUPPORTED_SQL: frozenset(
            {
                ChatBIErrorCategory.UNSAFE_QUERY,
                ChatBIErrorCategory.POLICY_VIOLATION,
                ChatBIErrorCategory.UNKNOWN_TABLE,
                ChatBIErrorCategory.UNKNOWN_COLUMN,
            }
        ),
    }
    return actual in allowed.get(case.expected_failure_category, frozenset())


def _repair_outcome(case: ChatBIEvaluationCase, result: ChatBIResult) -> RepairOutcome:
    if not case.repair_applicable:
        return RepairOutcome.NOT_APPLICABLE
    if result.repair_attempts == 0:
        return RepairOutcome.NOT_ATTEMPTED
    if result.execution_status is QueryLifecycleState.SUCCEEDED:
        return RepairOutcome.REPAIRED
    return RepairOutcome.EXHAUSTED


def _case_record(
    case: ChatBIEvaluationCase,
    observation: ChatBIEvaluationObservation,
    *,
    mode: Literal["offline", "provider"],
) -> EvaluationCaseRecord:
    result = observation.result
    stages = observation.stages
    if result.error_category is ChatBIErrorCategory.ANALYSIS_FAILED:
        stages = stages.model_copy(update={"analysis": "failed"})
    actual_result = observation.execution_result or result
    comparison = (
        compare_normalized_result(actual_result, case.expected_result)
        if case.expected_result is not None
        else None
    )
    execution_equivalent = comparison.equivalent if comparison is not None else None
    answer_fact_coverage = (
        answer_facts_match(result.answer, case.expected_answer_facts)
        if mode == "provider"
        else None
    )
    positive = case.expected_result is not None
    if result.error_category is ChatBIErrorCategory.ANALYSIS_FAILED:
        failure_category = ChatBIEvaluationFailureCategory.ANALYSIS
    elif positive and result.execution_status is QueryLifecycleState.SUCCEEDED:
        failure_category = (
            ChatBIEvaluationFailureCategory.RESULT_MISMATCH
            if execution_equivalent is False
            else (
                ChatBIEvaluationFailureCategory.ANSWER_MISMATCH
                if answer_fact_coverage is False
                else None
            )
        )
    else:
        failure_category = _map_failure_category(result)
    return EvaluationCaseRecord(
        case_id=case.case_id,
        split=case.split,
        category=case.category,
        positive_case=positive,
        stages=stages,
        execution_status=result.execution_status,
        execution_equivalent=execution_equivalent,
        answer_fact_coverage=answer_fact_coverage,
        expected_failure_match=_expected_failure_match(case, result),
        end_to_end_success=(
            positive
            and result.execution_status is QueryLifecycleState.SUCCEEDED
            and result.error_category is None
            and execution_equivalent is True
            and answer_fact_coverage is not False
            if mode == "provider"
            else None
        ),
        repair_outcome=_repair_outcome(case, result),
        failure_category=failure_category,
        row_count=result.row_count,
        truncated=result.truncated,
        truncation_reason=result.truncation_reason,
        metrics={
            "execution_equivalent": float(execution_equivalent is True),
            **(
                {"answer_fact_coverage": float(answer_fact_coverage is True)}
                if mode == "provider"
                else {}
            ),
        },
        latency=observation.timings,
        usage=result.usage,
    )


class ChatBIEvaluationObservation(_EvaluationModel):
    """Runner output with optional timings and evaluator-side materialized rows.

    ``ChatBIResult`` deliberately exposes only bounded result metadata to product
    callers. The optional execution result stays inside the evaluator so a
    successful empty result is not confused with unavailable result rows.
    """

    result: ChatBIResult
    execution_result: QueryExecutionResult | None = None
    timings: ChatBIStageTimings = Field(default_factory=ChatBIStageTimings)
    stages: EvaluationStageState = Field(default_factory=EvaluationStageState)


class ChatBICaseRunner(Protocol):
    async def run(self, case: ChatBIEvaluationCase) -> ChatBIEvaluationObservation:
        """Run one case without exposing its oracle to a model prompt."""


async def evaluate_chatbi_dataset(
    dataset: ChatBIEvaluationDataset,
    runner: ChatBICaseRunner,
    *,
    mode: Literal["offline", "provider"],
    datasource_id: UUID | None = None,
    git_revision: str | None = None,
    provenance: EvaluationRunProvenance | None = None,
    max_cases: int | None = None,
) -> ChatBIEvaluationRun:
    """Evaluate cases and write no artifacts unless the caller requests it."""
    if max_cases is not None and (max_cases < 1 or max_cases > len(dataset.cases)):
        raise ValueError("max_cases must be within the dataset case count")
    selected = dataset.cases if max_cases is None else dataset.cases[:max_cases]
    started = datetime.now(UTC)
    records: list[EvaluationCaseRecord] = []
    for case in selected:
        observation = await runner.run(case)
        if not isinstance(observation, ChatBIEvaluationObservation):
            raise ChatBIEvaluationError("case runner returned an invalid observation")
        records.append(_case_record(case, observation, mode=mode))
    completed = datetime.now(UTC)
    aggregate_mode = mode
    aggregates = {"all": _aggregate_records(records, mode=aggregate_mode)}
    for split in ("dev", "test"):
        split_records = [record for record in records if record.split == split]
        if split_records:
            aggregates[split] = _aggregate_records(split_records, mode=aggregate_mode)
    quality_claim = "offline_infrastructure_only" if mode == "offline" else "provider_observed"
    if provenance is None:
        provenance = _build_minimal_provenance(
            dataset,
            mode=mode,
            datasource_id=datasource_id,
            git_revision=git_revision,
            case_count=len(records),
        )
    run_git_revision = provenance.git_revision
    return ChatBIEvaluationRun(
        run_id=uuid4(),
        artifact_schema_version=CHATBI_EVALUATION_RUN_SCHEMA_VERSION,
        started_at=started,
        completed_at=completed,
        mode=mode,
        dataset_version=dataset.schema_version,
        dataset_fingerprint=dataset.fingerprint,
        datasource_id=datasource_id,
        git_revision=run_git_revision,
        case_count=len(records),
        records=records,
        aggregates=aggregates,
        provenance=provenance,
        quality_claim=quality_claim,
    )


def write_evaluation_run(path: Path, run: ChatBIEvaluationRun) -> None:
    """Write a safe runtime JSON artifact; oracle SQL is intentionally absent."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _canonical_json(run.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ChatBIEvaluationError(f"could not write evaluation output: {path}") from error


class OfflineScenario(_EvaluationModel):
    """A provider-free scripted outcome, separate from the frozen oracle dataset."""

    case_id: StrictStr
    datasource_id: UUID = DEFAULT_OFFLINE_DATASOURCE_ID
    execution_status: QueryLifecycleState
    answer: StrictStr | None = None
    columns: list[StrictStr] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    row_count: StrictInt = Field(ge=0)
    truncated: StrictBool = False
    truncation_reason: QueryTruncationReason | None = None
    sql_attempts: StrictInt = Field(ge=0)
    repair_attempts: StrictInt = Field(ge=0)
    llm_calls: StrictInt = Field(ge=0)
    provider_attempts: StrictInt = Field(ge=0)
    provider: StrictStr | None = None
    model: StrictStr | None = None
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    error_category: ChatBIErrorCategory | None = None
    failure_stage: Literal["generation", "validation", "execution", "analysis"] | None = None

    @model_validator(mode="after")
    def validate_scenario(self) -> OfflineScenario:
        if self.row_count != len(self.rows):
            raise ValueError("offline scenario row_count must match materialized rows")
        if self.execution_status is QueryLifecycleState.SUCCEEDED:
            if self.failure_stage is not None and self.failure_stage != "analysis":
                raise ValueError("successful offline scenarios only support analysis failure")
            if self.failure_stage != "analysis" and self.answer is None:
                raise ValueError("successful offline scenarios require an answer")
            if any(len(row) != len(self.columns) for row in self.rows):
                raise ValueError("successful offline scenarios require aligned rows")
            if self.failure_stage != "analysis" and self.error_category is not None:
                raise ValueError("successful offline scenarios cannot have an error")
            if (
                self.failure_stage == "analysis"
                and self.error_category is not ChatBIErrorCategory.ANALYSIS_FAILED
            ):
                raise ValueError("analysis failures must use the analysis_failed error category")
        else:
            if self.error_category is None:
                raise ValueError("failed offline scenarios require error_category")
            if self.failure_stage is None:
                raise ValueError("failed offline scenarios require failure_stage")
            if self.answer is not None or self.rows or self.columns:
                raise ValueError("failed offline scenarios cannot carry result data")
            if self.truncated or self.truncation_reason is not None:
                raise ValueError("failed offline scenarios cannot carry truncation metadata")
        if self.repair_attempts > self.sql_attempts:
            raise ValueError("repair_attempts cannot exceed sql_attempts")
        if self.sql_attempts < 1 or self.llm_calls < 1:
            raise ValueError("offline scenarios require at least one model call")
        if self.provider_attempts < self.llm_calls:
            raise ValueError("provider_attempts cannot be lower than llm_calls")
        if self.truncated != (self.truncation_reason is not None):
            raise ValueError("offline truncation metadata is inconsistent")
        return self


def load_offline_scenarios(path: Path = DEFAULT_OFFLINE_SCENARIOS) -> dict[str, OfflineScenario]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if raw.get("format") != OFFLINE_SCENARIO_FORMAT:
                raise ValueError("unsupported offline scenario format")
            entries = raw["scenarios"]
        else:
            entries = raw
        scenarios = [OfflineScenario.model_validate(entry) for entry in entries]
    except (OSError, TypeError, ValueError, ValidationError, KeyError) as error:
        raise ChatBIEvaluationError(f"invalid offline scenario file: {path}") from error
    result = {scenario.case_id: scenario for scenario in scenarios}
    if len(result) != len(scenarios):
        raise ChatBIEvaluationError("offline scenario IDs must be unique")
    return result


def _offline_trace(scenario: OfflineScenario) -> list[ChatBITraceEvent]:
    events: list[tuple[ChatBITraceName, str | None]] = []
    if scenario.failure_stage == "generation":
        events.append(("generation_failed", scenario.error_category.value))
    else:
        events.extend(
            [
                ("schema_prepared", None),
                ("sql_generated", None),
            ]
        )
        if scenario.repair_attempts:
            events.extend(
                [
                    ("generation_failed", "sql_parse_error"),
                    ("repair_requested", "sql_parse_error"),
                    ("schema_prepared", None),
                    ("sql_generated", None),
                ]
            )
        if scenario.execution_status is QueryLifecycleState.SUCCEEDED:
            events.extend(
                [
                    ("validation_accepted", None),
                    ("sql_executed", None),
                    ("result_normalized", None),
                    ("analysis_requested", None),
                ]
            )
            if scenario.failure_stage == "analysis":
                events.append(("analysis_failed", "analysis_failed"))
            else:
                events.append(("analysis_completed", None))
        elif scenario.failure_stage == "validation":
            events.append(("validation_rejected", scenario.error_category.value))
        else:
            events.append(("execution_failed", scenario.error_category.value))
    return [
        ChatBITraceEvent(event=event, step=index, error_category=error)
        for index, (event, error) in enumerate(events, start=1)
    ]


def _offline_stage_state(scenario: OfflineScenario) -> EvaluationStageState:
    """Translate one scripted scenario into explicit stage outcomes."""
    if scenario.failure_stage == "generation":
        return EvaluationStageState(generation="failed")
    if scenario.failure_stage == "validation":
        return EvaluationStageState(
            schema_preparation="passed",
            generation="passed",
            validation="failed",
            repair_generation=("passed" if scenario.repair_attempts else "not_attempted"),
        )
    if scenario.execution_status is QueryLifecycleState.SUCCEEDED:
        return EvaluationStageState(
            schema_preparation="passed",
            generation="passed",
            validation="passed",
            execution="passed",
            analysis="failed" if scenario.failure_stage == "analysis" else "passed",
            repair_generation=("passed" if scenario.repair_attempts else "not_attempted"),
        )
    return EvaluationStageState(
        schema_preparation="passed",
        generation="passed",
        validation="passed",
        execution="failed",
        repair_generation=("passed" if scenario.repair_attempts else "not_attempted"),
    )


class StoredOfflineScenarioRunner:
    """Return explicitly scripted outcomes without model, database, or network work."""

    def __init__(self, scenarios: Mapping[str, OfflineScenario]) -> None:
        self._scenarios = dict(scenarios)

    async def run(self, case: ChatBIEvaluationCase) -> ChatBIEvaluationObservation:
        try:
            scenario = self._scenarios[case.case_id]
        except KeyError as error:
            raise ChatBIEvaluationError(
                f"offline scenario is missing case {case.case_id}"
            ) from error
        columns = [
            ColumnMetadata(
                name=name,
                data_type="offline_fixture",
                nullable=True,
                ordinal=index,
            )
            for index, name in enumerate(scenario.columns)
        ]
        usage = ChatBIUsageSummary(
            llm_calls=scenario.llm_calls,
            provider_attempts=scenario.provider_attempts,
            provider=scenario.provider,
            model=scenario.model,
            input_tokens=scenario.input_tokens,
            output_tokens=scenario.output_tokens,
        )
        result = ChatBIResult(
            query_id=uuid4(),
            datasource_id=scenario.datasource_id,
            answer=scenario.answer
            if scenario.execution_status is QueryLifecycleState.SUCCEEDED
            else None,
            execution_status=scenario.execution_status,
            redacted_sql=None,
            columns=columns,
            row_count=scenario.row_count,
            truncated=scenario.truncated,
            truncation_reason=scenario.truncation_reason,
            sql_attempts=scenario.sql_attempts,
            repair_attempts=scenario.repair_attempts,
            usage=usage,
            error_category=scenario.error_category,
            error_message=(
                f"offline fixture: {scenario.error_category}"
                if scenario.error_category is not None
                else None
            ),
            trace=_offline_trace(scenario),
        )
        execution_result = QueryExecutionResult(
            query_id=result.query_id,
            datasource_id=result.datasource_id,
            state=scenario.execution_status,
            columns=columns,
            rows=(
                scenario.rows if scenario.execution_status is QueryLifecycleState.SUCCEEDED else []
            ),
            row_count=(
                scenario.row_count
                if scenario.execution_status is QueryLifecycleState.SUCCEEDED
                else 0
            ),
            truncated=scenario.truncated,
            truncation_reason=scenario.truncation_reason,
            max_rows=max(1, scenario.row_count),
            duration_ms=0,
            error_category=(
                scenario.error_category
                if scenario.execution_status is not QueryLifecycleState.SUCCEEDED
                else None
            ),
            error_message=(
                f"offline fixture: {scenario.error_category}"
                if scenario.execution_status is not QueryLifecycleState.SUCCEEDED
                and scenario.error_category is not None
                else None
            ),
        )
        return ChatBIEvaluationObservation(
            result=result,
            execution_result=execution_result,
            stages=_offline_stage_state(scenario),
        )


class _TimingState:
    def __init__(self) -> None:
        self.schema_prep_ms = 0.0
        self.generation_ms = 0.0
        self.validation_ms = 0.0
        self.execution_ms = 0.0
        self.repair_ms = 0.0
        self.analysis_ms = 0.0
        self.total_ms = 0.0
        self.execution_result: QueryExecutionResult | None = None
        self.stages = EvaluationStageState()
        self.generation_provider_called = False

    def reset(self) -> None:
        self.schema_prep_ms = 0.0
        self.generation_ms = 0.0
        self.validation_ms = 0.0
        self.execution_ms = 0.0
        self.repair_ms = 0.0
        self.analysis_ms = 0.0
        self.total_ms = 0.0
        self.execution_result = None
        self.stages = EvaluationStageState()
        self.generation_provider_called = False

    def snapshot(self) -> ChatBIStageTimings:
        def measured(value: float, status: EvaluationStageStatus) -> float | None:
            return value if status != "not_attempted" else None

        return ChatBIStageTimings(
            schema_prep_ms=measured(self.schema_prep_ms, self.stages.schema_preparation),
            generation_ms=measured(self.generation_ms, self.stages.generation),
            validation_ms=measured(self.validation_ms, self.stages.validation),
            execution_ms=measured(self.execution_ms, self.stages.execution),
            repair_ms=measured(self.repair_ms, self.stages.repair_generation),
            analysis_ms=measured(self.analysis_ms, self.stages.analysis),
            total_ms=self.total_ms,
        )

    def stage_snapshot(self) -> EvaluationStageState:
        return self.stages

    @staticmethod
    def _exclusive_elapsed(elapsed_ms: float, nested_ms: float) -> float:
        """Remove measured nested work from an outer stage timer."""
        return max(0.0, elapsed_ms - nested_ms)


class _TimedDiscovery:
    def __init__(self, inner: SchemaDiscoveryService, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def discover(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            result = await self._inner.discover(*args, **kwargs)
            self._timing.stages = self._timing.stages.model_copy(
                update={"schema_preparation": "passed"}
            )
            return result
        except BaseException:
            self._timing.stages = self._timing.stages.model_copy(
                update={"schema_preparation": "failed"}
            )
            raise
        finally:
            self._timing.schema_prep_ms += (perf_counter() - started) * 1_000


class _TimedGenerationGateway:
    """Mark the boundary where NL2SQL actually reaches the provider."""

    def __init__(self, inner: LLMGateway, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def complete(self, request: LLMRequest) -> LLMResult:
        if request.task_type == "nl2sql":
            self._timing.generation_provider_called = True
        return await self._inner.complete(request)


class _TimedGeneration:
    def __init__(self, inner: NL2SQLService, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def generate_candidate_with_usage_for_registered_data_source(
        self,
        datasource_id: UUID,
        query: Any,
        *,
        policy: QueryPolicy,
        max_chars: int,
        max_tokens: int | None = None,
        previous_sql: str | None = None,
        validation_error: str | None = None,
    ) -> tuple[SQLCandidate, LLMResult]:
        is_repair = previous_sql is not None or validation_error is not None
        schema_before = self._timing.schema_prep_ms
        self._timing.generation_provider_called = False
        started = perf_counter()
        try:
            result = await self._inner.generate_candidate_with_usage_for_registered_data_source(
                datasource_id,
                query,
                policy=policy,
                max_chars=max_chars,
                max_tokens=max_tokens,
                previous_sql=previous_sql,
                validation_error=validation_error,
            )
            stage = "repair_generation" if is_repair else "generation"
            self._timing.stages = self._timing.stages.model_copy(update={stage: "passed"})
            return result
        except BaseException:
            stage = "repair_generation" if is_repair else "generation"
            status = "failed" if self._timing.generation_provider_called else "not_attempted"
            self._timing.stages = self._timing.stages.model_copy(update={stage: status})
            raise
        finally:
            elapsed = (perf_counter() - started) * 1_000
            exclusive = self._timing._exclusive_elapsed(
                elapsed,
                self._timing.schema_prep_ms - schema_before,
            )
            self._timing.repair_ms += exclusive if is_repair else 0.0
            self._timing.generation_ms += exclusive if not is_repair else 0.0


class _TimedValidation:
    def __init__(self, inner: NL2SQLService, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def _validate_registered_candidate(self, *args: Any, **kwargs: Any) -> Any:
        schema_before = self._timing.schema_prep_ms
        started = perf_counter()
        try:
            result = await self._inner._validate_registered_candidate(*args, **kwargs)
            self._timing.stages = self._timing.stages.model_copy(update={"validation": "passed"})
            return result
        except BaseException:
            self._timing.stages = self._timing.stages.model_copy(update={"validation": "failed"})
            raise
        finally:
            elapsed = (perf_counter() - started) * 1_000
            self._timing.validation_ms += self._timing._exclusive_elapsed(
                elapsed,
                self._timing.schema_prep_ms - schema_before,
            )


class _TimedExecutionAdapter:
    def __init__(self, inner: ExecutionAdapter, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def _execute_validated(self, *args: Any, **kwargs: Any) -> QueryExecutionResult:
        started = perf_counter()
        try:
            result = await self._inner._execute_validated(*args, **kwargs)
            self._timing.stages = self._timing.stages.model_copy(
                update={
                    "execution": (
                        "passed" if result.state is QueryLifecycleState.SUCCEEDED else "failed"
                    )
                }
            )
            return result
        except BaseException:
            self._timing.stages = self._timing.stages.model_copy(update={"execution": "failed"})
            raise
        finally:
            self._timing.execution_ms += (perf_counter() - started) * 1_000


class _TimedExecutionService:
    """Capture the evaluator-only result while preserving the Agent contract."""

    def __init__(self, inner: SQLExecutionService, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def execute_for_registered_data_source(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> SQLExecutionOutcome:
        outcome = await self._inner.execute_for_registered_data_source(*args, **kwargs)
        if not isinstance(outcome, SQLExecutionOutcome):
            raise ChatBIEvaluationError("execution service returned an invalid outcome")
        self._timing.execution_result = outcome.result
        if outcome.result.state is QueryLifecycleState.SUCCEEDED:
            self._timing.stages = self._timing.stages.model_copy(update={"execution": "passed"})
        elif outcome.result.state is not QueryLifecycleState.REJECTED:
            self._timing.stages = self._timing.stages.model_copy(update={"execution": "failed"})
        return outcome


class _TimedAnalysisGateway:
    def __init__(self, inner: LLMGateway, timing: _TimingState) -> None:
        self._inner = inner
        self._timing = timing

    async def complete(self, request: LLMRequest) -> LLMResult:
        started = perf_counter()
        try:
            result = await self._inner.complete(request)
            if request.task_type == "chatbi_analysis":
                self._timing.stages = self._timing.stages.model_copy(update={"analysis": "passed"})
            return result
        except BaseException:
            if request.task_type == "chatbi_analysis":
                self._timing.stages = self._timing.stages.model_copy(update={"analysis": "failed"})
            raise
        finally:
            if request.task_type == "chatbi_analysis":
                self._timing.analysis_ms += (perf_counter() - started) * 1_000


class ProviderChatBICaseRunner:
    """Run real provider-backed cases through the accepted Agent chain."""

    def __init__(
        self,
        agent: ChatBIAgentService,
        timing: _TimingState,
        datasource_id: UUID,
        policy: QueryPolicy,
        max_chars: int,
    ) -> None:
        self._agent = agent
        self._timing = timing
        self._datasource_id = datasource_id
        self._policy = policy
        self._max_chars = max_chars

    async def run(self, case: ChatBIEvaluationCase) -> ChatBIEvaluationObservation:
        self._timing.reset()
        started = perf_counter()
        try:
            # Only the user question crosses this boundary.  The reference SQL,
            # expected rows, and answer facts remain evaluator-side data.
            result = await self._agent.ask(
                self._datasource_id,
                case.question,
                policy=self._policy,
                max_chars=self._max_chars,
            )
        finally:
            self._timing.total_ms = (perf_counter() - started) * 1_000
        return ChatBIEvaluationObservation(
            result=result,
            execution_result=self._timing.execution_result,
            timings=self._timing.snapshot(),
            stages=self._timing.stage_snapshot(),
        )


def _verify_frozen_provider_inputs(
    dataset: ChatBIEvaluationDataset,
    fixture_path: Path,
) -> None:
    """Reject non-frozen dataset/fixture inputs before creating a provider."""
    if dataset.datasource_fixture != DEFAULT_FIXTURE_PATH.as_posix():
        raise ChatBIEvaluationError(
            "provider evaluation requires the frozen ChatBI demo fixture identity"
        )
    frozen = load_chatbi_evaluation_dataset(DEFAULT_DATASET)
    if dataset.fingerprint != frozen.fingerprint:
        raise ChatBIEvaluationError(
            "provider evaluation requires the frozen ChatBI evaluation dataset"
        )
    if fixture_path.as_posix() != DEFAULT_FIXTURE_PATH.as_posix():
        raise ChatBIEvaluationError(
            "provider evaluation requires the repository ChatBI demo fixture"
        )
    verify_fixture_fingerprint(dataset, fixture_path)


def _verify_demo_schema(
    data_source: DataSource,
    discovered: SchemaDiscoveryResult,
    *,
    datasource_id: UUID,
    policy: QueryPolicy,
) -> None:
    """Verify the discovered schema is the authoritative demo fixture shape."""
    if data_source.id != datasource_id:
        raise ChatBIEvaluationError("registered benchmark datasource identity mismatch")
    if data_source.dialect is not SQLDialect.POSTGRESQL:
        raise ChatBIEvaluationError("benchmark datasource must use PostgreSQL")
    if data_source.default_schema != CHATBI_DEMO_SCHEMA:
        raise ChatBIEvaluationError("benchmark datasource is not bound to chatbi_demo")
    if policy.max_rows < 3 or CHATBI_DEMO_SCHEMA not in policy.allowed_schemas:
        raise ChatBIEvaluationError("benchmark policy must allow the complete chatbi_demo fixture")
    snapshot = discovered.snapshot
    if snapshot.datasource_id != datasource_id or snapshot.dialect is not SQLDialect.POSTGRESQL:
        raise ChatBIEvaluationError("discovered benchmark schema identity mismatch")
    if snapshot.schemas != (CHATBI_DEMO_SCHEMA,):
        raise ChatBIEvaluationError("discovered benchmark schema set does not match the fixture")
    required_columns = {
        "customers": {"customer_id", "customer_name"},
        "sales": {"sale_id", "customer_id", "sold_on", "region", "amount"},
    }
    tables = {
        relation.name: relation
        for relation in snapshot.relations
        if relation.schema_name == CHATBI_DEMO_SCHEMA and relation.kind is SchemaObjectKind.TABLE
    }
    if set(tables) != set(required_columns):
        raise ChatBIEvaluationError("discovered benchmark tables do not match the fixture")
    if any(
        {column.name for column in relation.columns} != required_columns[name]
        for name, relation in tables.items()
    ):
        raise ChatBIEvaluationError("discovered benchmark columns do not match the fixture")


async def _probe_demo_data(
    execution: SQLExecutionService,
    *,
    datasource_id: UUID,
    policy: QueryPolicy,
    max_chars: int,
) -> str:
    """Read a fixed, bounded fixture probe and return its safe content fingerprint."""
    observed: list[dict[str, object]] = []
    for sql, expected_columns, expected_rows in _DEMO_DATA_PROBE_EXPECTATIONS:
        try:
            outcome = await execution.execute_sql_for_registered_data_source(
                datasource_id,
                sql,
                policy=policy,
                max_chars=max_chars,
            )
        except Exception as error:
            raise ChatBIEvaluationError("benchmark fixture data probe failed") from error
        result = outcome.result
        actual_columns = tuple(column.name for column in result.columns)
        actual_rows = tuple(tuple(row) for row in result.rows)
        if (
            result.datasource_id != datasource_id
            or result.state is not QueryLifecycleState.SUCCEEDED
            or result.truncated
            or actual_columns != expected_columns
            or actual_rows != expected_rows
        ):
            raise ChatBIEvaluationError(
                "registered benchmark datasource data does not match fixture"
            )
        observed.append(
            {
                "columns": list(actual_columns),
                "rows": [list(row) for row in actual_rows],
            }
        )
    return _sha256_json(observed)


async def run_provider_chatbi_evaluation(
    dataset: ChatBIEvaluationDataset,
    *,
    settings: Settings,
    datasource_id: UUID,
    output_path: Path | None = None,
    max_chars: int | None = None,
    max_cases: int | None = None,
    git_revision: str | None = None,
    git_dirty: bool | None = None,
    fixture_path: Path | None = None,
) -> ChatBIEvaluationRun:
    """Run the explicit provider benchmark through the existing production services."""
    selected_fixture_path = fixture_path or DEFAULT_FIXTURE_PATH
    _verify_frozen_provider_inputs(dataset, selected_fixture_path)
    if settings.chatbi_evaluation_datasource_id != datasource_id:
        raise ChatBIEvaluationError(
            "provider evaluation requires the configured authoritative demo datasource"
        )
    if max_cases is not None and (max_cases < 1 or max_cases > len(dataset.cases)):
        raise ValueError("max_cases must be within the dataset case count")
    case_count = len(dataset.cases) if max_cases is None else max_cases
    effective_max_chars = (
        settings.chatbi_schema_context_max_chars if max_chars is None else max_chars
    )
    if isinstance(effective_max_chars, bool) or not isinstance(effective_max_chars, int):
        raise TypeError("max_chars must be a positive integer")
    if effective_max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    policy = default_query_policy(settings)
    engine = create_database_engine(settings)
    provider = None
    try:
        session_factory = create_session_factory(engine)
        data_source_provider = DatabaseDataSourceProvider(session_factory)
        discovery = create_postgres_schema_discovery_service(
            connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
            statement_timeout_ms=settings.chatbi_statement_timeout_ms,
        )
        data_source = await data_source_provider.get(datasource_id)
        if data_source is None:
            raise ChatBIEvaluationError("configured benchmark datasource is not registered")
        discovered = await discovery.discover(
            data_source,
            policy,
            max_chars=effective_max_chars,
        )
        if not isinstance(discovered, SchemaDiscoveryResult):
            raise ChatBIEvaluationError("benchmark schema discovery returned an invalid result")
        _verify_demo_schema(
            data_source,
            discovered,
            datasource_id=datasource_id,
            policy=policy,
        )
        preflight_validation = NL2SQLService(
            None,
            schema_discovery=discovery,
            data_source_provider=data_source_provider,
            max_tokens=settings.chatbi_nl2sql_max_tokens,
        )
        preflight_execution = SQLExecutionService(
            preflight_validation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(
                connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                statement_timeout_ms=settings.chatbi_statement_timeout_ms,
            ),
        )
        fixture_data_fingerprint = await _probe_demo_data(
            preflight_execution,
            datasource_id=datasource_id,
            policy=policy,
            max_chars=effective_max_chars,
        )

        provider = create_llm_provider(settings)
        gateway = LLMGateway(provider, DatabaseUsageRecorder(session_factory), settings)
        timing = _TimingState()
        timed_discovery = _TimedDiscovery(discovery, timing)
        generation = NL2SQLService(
            _TimedGenerationGateway(gateway, timing),
            schema_discovery=timed_discovery,
            data_source_provider=data_source_provider,
            max_tokens=settings.chatbi_nl2sql_max_tokens,
        )
        timed_generation = _TimedGeneration(generation, timing)
        execution = SQLExecutionService(
            _TimedValidation(generation, timing),
            EnvironmentCredentialResolver(),
            _TimedExecutionAdapter(
                PostgresExecutionAdapter(
                    connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                    statement_timeout_ms=settings.chatbi_statement_timeout_ms,
                ),
                timing,
            ),
        )
        timed_execution = _TimedExecutionService(execution, timing)
        agent = ChatBIAgentService(
            timed_generation,
            timed_execution,
            _TimedAnalysisGateway(gateway, timing),
            limits=ChatBIAgentLimits.from_settings(settings),
        )
        run = await evaluate_chatbi_dataset(
            dataset,
            ProviderChatBICaseRunner(
                agent,
                timing,
                datasource_id,
                policy,
                effective_max_chars,
            ),
            mode="provider",
            datasource_id=datasource_id,
            git_revision=git_revision,
            provenance=_build_provider_provenance(
                dataset,
                settings=settings,
                policy=policy,
                max_chars=effective_max_chars,
                datasource_id=datasource_id,
                fixture_data_fingerprint=fixture_data_fingerprint,
                git_revision=git_revision,
                git_dirty=git_dirty,
                case_count=case_count,
            ),
            max_cases=max_cases,
        )
        if output_path is not None:
            write_evaluation_run(output_path, run)
        return run
    finally:
        if provider is not None:
            await provider.aclose()
        await engine.dispose()


async def run_offline_chatbi_evaluation(
    dataset: ChatBIEvaluationDataset,
    scenarios: Mapping[str, OfflineScenario],
    *,
    output_path: Path | None = None,
    max_cases: int | None = None,
    git_revision: str | None = None,
) -> ChatBIEvaluationRun:
    """Run the provider-free infrastructure path using stored outcomes."""
    dataset_case_ids = {case.case_id for case in dataset.cases}
    scenario_case_ids = set(scenarios)
    if scenario_case_ids != dataset_case_ids:
        missing = sorted(dataset_case_ids - scenario_case_ids)
        extra = sorted(scenario_case_ids - dataset_case_ids)
        raise ChatBIEvaluationError(
            f"offline scenario coverage does not match dataset (missing={missing}, extra={extra})"
        )
    scenario_datasource_ids = {scenario.datasource_id for scenario in scenarios.values()}
    if len(scenario_datasource_ids) != 1:
        raise ChatBIEvaluationError("offline scenarios must use one isolated datasource identity")
    run = await evaluate_chatbi_dataset(
        dataset,
        StoredOfflineScenarioRunner(scenarios),
        mode="offline",
        datasource_id=next(iter(scenarios.values())).datasource_id if scenarios else None,
        git_revision=git_revision,
        max_cases=max_cases,
    )
    if output_path is not None:
        write_evaluation_run(output_path, run)
    return run


def current_git_revision(repo_root: Path | None = None) -> str | None:
    """Return a safe revision identifier for a runtime reproducibility record."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root or Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and len(revision) == 40 else None


def current_git_dirty(repo_root: Path | None = None) -> bool:
    """Return the working-tree dirty state for reproducibility metadata."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root or Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ChatBIEvaluationError("could not determine Git working-tree state") from error
    if completed.returncode != 0:
        raise ChatBIEvaluationError("could not determine Git working-tree state")
    return bool(completed.stdout.strip())


__all__ = [
    "CHATBI_EVALUATION_PROMPT_POLICY_VERSION",
    "CHATBI_EVALUATION_SCHEMA_VERSION",
    "DEFAULT_DATASET",
    "DEFAULT_FIXTURE_PATH",
    "DEFAULT_OFFLINE_SCENARIOS",
    "DEFAULT_OUTPUT",
    "OFFLINE_SCENARIO_FORMAT",
    "ChatBICaseRunner",
    "ChatBIEvaluationCase",
    "ChatBIEvaluationCategory",
    "ChatBIEvaluationDataset",
    "ChatBIEvaluationError",
    "ChatBIEvaluationFailureCategory",
    "ChatBIEvaluationObservation",
    "ChatBIEvaluationRun",
    "ChatBIStageTimings",
    "EvaluationAgentLimits",
    "EvaluationAggregate",
    "EvaluationCaseRecord",
    "EvaluationQueryPolicy",
    "EvaluationRunProvenance",
    "EvaluationRuntimeConfiguration",
    "EvaluationStageState",
    "ExpectedQueryResult",
    "OfflineScenario",
    "ProviderChatBICaseRunner",
    "RepairOutcome",
    "ResultComparison",
    "StoredOfflineScenarioRunner",
    "answer_facts_match",
    "assert_no_oracle_leakage",
    "compare_normalized_result",
    "current_git_dirty",
    "current_git_revision",
    "evaluate_chatbi_dataset",
    "load_chatbi_evaluation_dataset",
    "load_offline_scenarios",
    "normalize_evaluation_text",
    "run_offline_chatbi_evaluation",
    "run_provider_chatbi_evaluation",
    "verify_fixture_fingerprint",
    "write_evaluation_run",
]
