"""Provider-backed execution for the frozen ChatBI evaluation v2 dataset.

The v2 provider benchmark is intentionally separate from the v1 evaluator.  It
bootstraps one deterministic, isolated demo datasource and then reuses the
normal registered-datasource ChatBI services.  Frozen oracle fields stay on the
evaluator side and are never passed to the Agent or an LLM gateway.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator
from sqlalchemy import or_, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from knowledge_scope.chatbi import (
    ChatBIAgentLimits,
    ChatBIAgentService,
    ChatBIResult,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryLifecycleState,
    QueryTruncationReason,
    SQLExecutionService,
    default_query_policy,
)
from knowledge_scope.chatbi.discovery import create_postgres_schema_discovery_service
from knowledge_scope.chatbi.errors import ChatBIErrorCategory
from knowledge_scope.chatbi.models import ChatBIDataSourceRecord
from knowledge_scope.chatbi.nl2sql_models import NL2SQL_PROMPT_VERSION
from knowledge_scope.chatbi.policy import QueryPolicy, SQLDialect
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider, data_source_from_record
from knowledge_scope.chatbi.schema_models import SchemaDiscoveryResult, SchemaObjectKind
from knowledge_scope.chatbi.schemas import DataSource
from knowledge_scope.evaluation.chatbi_evaluation import (
    CHATBI_ANALYSIS_PROMPT_VERSION,
    ChatBIEvaluationError,
    ChatBIEvaluationFailureCategory,
    ChatBIEvaluationObservation,
    ChatBIStageTimings,
    ChatBIUsageSummary,
    EvaluationAgentLimits,
    EvaluationQueryPolicy,
    EvaluationRuntimeConfiguration,
    EvaluationStageState,
    LatencyAggregate,
    RepairOutcome,
    UsageAggregate,
    _EvaluationModel,
    _map_failure_category,
    _numeric_summary,
    _sha256_json,
    _TimedAnalysisGateway,
    _TimedDiscovery,
    _TimedExecutionAdapter,
    _TimedExecutionService,
    _TimedGeneration,
    _TimedGenerationGateway,
    _TimedValidation,
    _TimingState,
    compare_normalized_result,
    current_git_dirty,
    current_git_revision,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    CHATBI_DEMO_FIXTURE_VERSION_V2,
    CHATBI_EVALUATION_V2_DATASET_VERSION,
    CHATBI_EVALUATION_V2_STATUS,
    DEFAULT_DATASET_V2,
    DEFAULT_FIXTURE_PATH_V2,
    ChatBIEvaluationCaseV2,
    ChatBIEvaluationDatasetV2,
    ChatBIEvaluationV2Category,
    ChatBIEvaluationV2Difficulty,
    load_chatbi_evaluation_dataset_v2,
    structured_result_facts_match,
)
from knowledge_scope.llm import LLMGateway, create_llm_provider
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

CHATBI_EVALUATION_V2_PROVIDER_VERSION = "a5.7b"
CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION = "a5.7b-provider-run-v1"
CHATBI_EVALUATION_V2_DATASOURCE_ID = uuid5(
    NAMESPACE_URL, "https://knowledgescope.local/evaluation/chatbi-demo-v2"
)
CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME = "KnowledgeScope ChatBI evaluation v2"
CHATBI_EVALUATION_V2_DATASOURCE_IDENTITY = "chatbi-demo-v2-authoritative"
CHATBI_EVALUATION_V2_DATABASE_NAME = "knowledgescope_chatbi_eval_v2"
CHATBI_EVALUATION_V2_CREDENTIAL_ENV = "KNOWLEDGE_SCOPE_CHATBI_EVALUATION_V2_DATABASE_URL"
CHATBI_EVALUATION_V2_CONNECTION_REF = f"env:{CHATBI_EVALUATION_V2_CREDENTIAL_ENV}"
CHATBI_EVALUATION_V2_SCHEMA = "chatbi_demo"
DEFAULT_PROVIDER_OUTPUT_V2 = Path("data/evaluation/a5-7/provider/chatbi-eval-v2-dev.json")
EXPECTED_DATASET_FINGERPRINT_V2 = "60c75c597da8fc71a0fa5b25d335b63410b44a4ab3a403da40ca72c5ae375ab3"
EXPECTED_FIXTURE_FINGERPRINT_V2 = "fd972106c39c7a8b31b57975118708e213a32e4e008ee13fafc15b1ea1b5182d"
EXPECTED_FIXTURE_DATA_FINGERPRINT_V2 = (
    "58dce6bae9c916a218b7ca57861a337f050658caf3f73163ac2df17e51c5e55d"
)
EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2 = (
    "592681fd7c63ee654f87ecfac62455536f90c994fc74fc4e00973743be607071"
)
_LOCAL_POSTGRES_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class V2ProviderBenchmarkError(ChatBIEvaluationError):
    """Raised when v2 provider preflight or benchmark execution is invalid."""


class V2ProviderSplit(StrEnum):
    DEV = "dev"
    TEST = "test"


class V2ProviderPreflightReport(_EvaluationModel):
    """Provider-free readiness result with no URL, key, SQL, or raw response."""

    status: Literal["ready_for_real_dev_provider_run"] = "ready_for_real_dev_provider_run"
    dataset_status: Literal["human_reviewed_frozen"] = CHATBI_EVALUATION_V2_STATUS
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = CHATBI_DEMO_FIXTURE_VERSION_V2
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_data_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID
    datasource_identity: Literal["chatbi-demo-v2-authoritative"] = (
        CHATBI_EVALUATION_V2_DATASOURCE_IDENTITY
    )
    split: Literal["dev"]
    selected_case_count: StrictInt = Field(ge=0)
    selected_test_case_count: StrictInt = Field(ge=0)
    provider: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(min_length=1, max_length=255)
    provider_configured: StrictBool
    database_created: StrictBool
    datasource_created: StrictBool
    git_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: StrictBool
    provider_calls: Literal[0] = 0


class V2ProviderCaseRecord(_EvaluationModel):
    """Safe per-case provider observation; frozen oracle data is absent."""

    case_id: StrictStr
    split: Literal["dev"] = "dev"
    positive: StrictBool
    category: ChatBIEvaluationV2Category
    difficulty: ChatBIEvaluationV2Difficulty
    stages: EvaluationStageState
    result_status: QueryLifecycleState
    generation_parseable: StrictBool
    validation_accepted: StrictBool
    execution_equivalent: StrictBool | None = None
    structured_result_fact_coverage: StrictBool | None = None
    expected_failure_match: StrictBool | None = None
    repair_attempted: StrictBool
    repair_outcome: RepairOutcome
    failure_category: ChatBIEvaluationFailureCategory | None = None
    row_count: StrictInt = Field(ge=0)
    truncated: StrictBool
    truncation_reason: QueryTruncationReason | None = None
    latency: ChatBIStageTimings
    usage: ChatBIUsageSummary
    redacted_sql: StrictStr | None = Field(default=None, max_length=100_000)
    trace_events: list[StrictStr] = Field(default_factory=list)


class V2ProviderAggregate(_EvaluationModel):
    """Primary and auxiliary aggregate metrics for the provider split."""

    case_count: StrictInt = Field(ge=0)
    positive_case_count: StrictInt = Field(ge=0)
    negative_case_count: StrictInt = Field(ge=0)
    generation_attempted_count: StrictInt = Field(ge=0)
    generation_parseable_count: StrictInt = Field(ge=0)
    generation_parseable_rate: float | None = Field(default=None, ge=0, le=1)
    validation_attempted_count: StrictInt = Field(ge=0)
    validator_accepted_count: StrictInt = Field(ge=0)
    validator_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    execution_equivalent_count: StrictInt = Field(ge=0)
    execution_accuracy: float | None = Field(default=None, ge=0, le=1)
    structured_result_fact_coverage_count: StrictInt = Field(ge=0)
    structured_result_fact_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    negative_safe_success_count: StrictInt = Field(ge=0)
    negative_safe_success_rate: float | None = Field(default=None, ge=0, le=1)
    repair_attempted_count: StrictInt = Field(ge=0)
    repair_success_count: StrictInt = Field(ge=0)
    repair_failure_count: StrictInt = Field(ge=0)
    repair_success_rate: float | None = Field(default=None, ge=0, le=1)
    incremental_recovered_count: StrictInt = Field(ge=0)
    failure_categories: dict[str, StrictInt] = Field(default_factory=dict)
    latency: LatencyAggregate
    usage: UsageAggregate


class V2ProviderRunProvenance(_EvaluationModel):
    """Reproducibility metadata without credentials, DSNs, SQL, or responses."""

    benchmark_version: Literal["a5.7b"] = CHATBI_EVALUATION_V2_PROVIDER_VERSION
    benchmark_mode: Literal["provider"] = "provider"
    dataset_version: Literal["a5.7-v2"] = CHATBI_EVALUATION_V2_DATASET_VERSION
    dataset_status: Literal["human_reviewed_frozen"] = CHATBI_EVALUATION_V2_STATUS
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = CHATBI_DEMO_FIXTURE_VERSION_V2
    fixture_path: StrictStr = Field(min_length=1, max_length=500)
    fixture_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_data_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID
    datasource_identity: Literal["chatbi-demo-v2-authoritative"] = (
        CHATBI_EVALUATION_V2_DATASOURCE_IDENTITY
    )
    split: Literal["dev"] = "dev"
    case_count: StrictInt = Field(ge=0)
    git_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: StrictBool
    provider: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(min_length=1, max_length=255)
    temperature: float = Field(ge=0, le=2)
    response_format: Literal["json_object"]
    configuration: EvaluationRuntimeConfiguration
    started_at: datetime
    completed_at: datetime

    @field_validator("fixture_path")
    @classmethod
    def require_relative_fixture_path(cls, value: str) -> str:
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("provider fixture path must be repository-relative")
        return value


class V2ProviderRun(_EvaluationModel):
    """Repository-safe artifact contract for exactly the frozen DEV split."""

    run_id: UUID
    artifact_schema_version: Literal["a5.7b-provider-run-v1"] = (
        CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION
    )
    started_at: datetime
    completed_at: datetime
    benchmark_version: Literal["a5.7b"] = CHATBI_EVALUATION_V2_PROVIDER_VERSION
    mode: Literal["provider"] = "provider"
    split: Literal["dev"] = "dev"
    dataset_version: Literal["a5.7-v2"] = CHATBI_EVALUATION_V2_DATASET_VERSION
    dataset_status: Literal["human_reviewed_frozen"] = CHATBI_EVALUATION_V2_STATUS
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID
    case_count: StrictInt = Field(ge=0)
    records: list[V2ProviderCaseRecord] = Field(min_length=50, max_length=50)
    aggregates: V2ProviderAggregate
    provenance: V2ProviderRunProvenance
    quality_claim: Literal["provider_observed"] = "provider_observed"

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> V2ProviderRun:
        if self.case_count != len(self.records) or self.case_count != 50:
            raise ValueError("v2 provider artifacts must contain exactly 50 cases")
        if len({record.case_id for record in self.records}) != self.case_count:
            raise ValueError("v2 provider records must have unique case IDs")
        if any(record.split != self.split for record in self.records):
            raise ValueError("v2 provider records must remain on the DEV split")
        if self.datasource_id != self.provenance.datasource_id:
            raise ValueError("run and provenance datasource identities must match")
        if self.dataset_fingerprint != self.provenance.dataset_fingerprint:
            raise ValueError("run and provenance dataset fingerprints must match")
        if self.fixture_fingerprint != self.provenance.fixture_sha256:
            raise ValueError("run and provenance fixture fingerprints must match")
        if self.fixture_schema_fingerprint != self.provenance.fixture_schema_fingerprint:
            raise ValueError("run and provenance schema fingerprints must match")
        if self.provenance.case_count != self.case_count:
            raise ValueError("run and provenance case counts must match")
        if self.aggregates.case_count != self.case_count:
            raise ValueError("run and aggregate case counts must match")
        return self


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _negative_safe_success(record: V2ProviderCaseRecord) -> bool:
    return (
        not record.positive
        and record.result_status is not QueryLifecycleState.SUCCEEDED
        and "sql_executed" not in record.trace_events
    )


def _v2_latency_aggregate(records: Sequence[V2ProviderCaseRecord]) -> LatencyAggregate:
    fields = (
        "schema_prep_ms",
        "generation_ms",
        "validation_ms",
        "execution_ms",
        "repair_ms",
        "analysis_ms",
        "total_ms",
    )
    return LatencyAggregate(
        **{
            name: _numeric_summary([getattr(record.latency, name) for record in records])
            for name in fields
        }
    )


def _v2_usage_aggregate(records: Sequence[V2ProviderCaseRecord]) -> UsageAggregate:
    return UsageAggregate(
        case_count=len(records),
        llm_calls_total=sum(record.usage.llm_calls for record in records),
        provider_attempts_total=sum(record.usage.provider_attempts for record in records),
        input_tokens=_numeric_summary([record.usage.input_tokens for record in records]),
        output_tokens=_numeric_summary([record.usage.output_tokens for record in records]),
    )


def _build_v2_aggregate(records: Sequence[V2ProviderCaseRecord]) -> V2ProviderAggregate:
    positives = [record for record in records if record.positive]
    negatives = [record for record in records if not record.positive]
    generation_attempted = [
        record for record in records if record.stages.generation != "not_attempted"
    ]
    validation_attempted = [
        record for record in records if record.stages.validation != "not_attempted"
    ]
    repair_attempted = [record for record in records if record.repair_attempted]
    repair_success = [
        record for record in repair_attempted if record.repair_outcome is RepairOutcome.REPAIRED
    ]
    fact_records = [
        record for record in positives if record.structured_result_fact_coverage is not None
    ]
    generation_parseable_count = sum(record.generation_parseable for record in generation_attempted)
    validator_accepted_count = sum(record.validation_accepted for record in validation_attempted)
    execution_equivalent_count = sum(record.execution_equivalent is True for record in positives)
    fact_coverage_count = sum(
        record.structured_result_fact_coverage is True for record in fact_records
    )
    return V2ProviderAggregate(
        case_count=len(records),
        positive_case_count=len(positives),
        negative_case_count=len(negatives),
        generation_attempted_count=len(generation_attempted),
        generation_parseable_count=generation_parseable_count,
        generation_parseable_rate=_safe_rate(
            generation_parseable_count,
            len(generation_attempted),
        ),
        validation_attempted_count=len(validation_attempted),
        validator_accepted_count=validator_accepted_count,
        validator_acceptance_rate=_safe_rate(
            validator_accepted_count,
            len(validation_attempted),
        ),
        execution_equivalent_count=execution_equivalent_count,
        execution_accuracy=_safe_rate(execution_equivalent_count, len(positives)),
        structured_result_fact_coverage_count=fact_coverage_count,
        structured_result_fact_coverage_rate=_safe_rate(fact_coverage_count, len(fact_records)),
        negative_safe_success_count=sum(_negative_safe_success(record) for record in negatives),
        negative_safe_success_rate=_safe_rate(
            sum(_negative_safe_success(record) for record in negatives),
            len(negatives),
        ),
        repair_attempted_count=len(repair_attempted),
        repair_success_count=len(repair_success),
        repair_failure_count=len(repair_attempted) - len(repair_success),
        repair_success_rate=_safe_rate(len(repair_success), len(repair_attempted)),
        incremental_recovered_count=sum(
            record.positive and record.repair_attempted and record.execution_equivalent is True
            for record in records
        ),
        failure_categories=dict(
            sorted(
                Counter(
                    record.failure_category.value
                    for record in records
                    if record.failure_category is not None
                ).items()
            )
        ),
        latency=_v2_latency_aggregate(records),
        usage=_v2_usage_aggregate(records),
    )


@dataclass(frozen=True, slots=True)
class _V2DatasourceBootstrap:
    data_source: DataSource
    database_url: str = field(repr=False)
    fixture_schema_fingerprint: str
    fixture_data_fingerprint: str
    database_created: bool
    datasource_created: bool
    environment_was_present: bool


class V2ProviderCaseRunner(Protocol):
    async def run(self, case: ChatBIEvaluationCaseV2) -> ChatBIEvaluationObservation:
        """Run one case through the production ChatBI Agent path."""


def select_v2_provider_cases(
    dataset: ChatBIEvaluationDatasetV2,
    split: str,
) -> list[ChatBIEvaluationCaseV2]:
    """Select only the currently enabled frozen provider split."""
    if split != V2ProviderSplit.DEV.value:
        raise V2ProviderBenchmarkError(
            "provider-backed v2 evaluation currently permits only --split dev; test is frozen"
        )
    selected = [case for case in dataset.cases if case.split == V2ProviderSplit.DEV.value]
    selected_tests = [case for case in dataset.cases if case.split == V2ProviderSplit.TEST.value]
    if len(selected) != 50 or len(selected_tests) != 30:
        raise V2ProviderBenchmarkError(
            "v2 provider selection requires the frozen 50 DEV / 30 TEST dataset"
        )
    return selected


def _validate_frozen_dev_cases(
    cases: Sequence[ChatBIEvaluationCaseV2],
) -> tuple[ChatBIEvaluationCaseV2, ...]:
    """Validate the complete frozen DEV set before any provider call."""
    try:
        dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    except ChatBIEvaluationError as error:
        raise V2ProviderBenchmarkError("the frozen v2 provider dataset is unavailable") from error
    if dataset.fingerprint != EXPECTED_DATASET_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError("the frozen v2 provider dataset identity does not match")
    expected_cases = {
        case.case_id: case for case in select_v2_provider_cases(dataset, V2ProviderSplit.DEV.value)
    }
    supplied_ids = [case.case_id for case in cases]
    if any(case.split != V2ProviderSplit.DEV.value for case in cases):
        raise V2ProviderBenchmarkError("v2 provider API accepts DEV cases only")
    if len(supplied_ids) != len(set(supplied_ids)):
        raise V2ProviderBenchmarkError("v2 provider cases must not contain duplicates")
    if set(supplied_ids) != set(expected_cases):
        raise V2ProviderBenchmarkError("v2 provider cases must be the complete frozen DEV split")
    if any(case != expected_cases[case.case_id] for case in cases):
        raise V2ProviderBenchmarkError("v2 provider cases do not match the frozen DEV dataset")
    return tuple(cases)


def _verify_v2_dataset_and_fixture(
    dataset_path: Path,
    fixture_path: Path,
) -> tuple[ChatBIEvaluationDatasetV2, str]:
    """Verify the repository-owned dataset and fixture identities."""
    if dataset_path.as_posix() != DEFAULT_DATASET_V2.as_posix():
        raise V2ProviderBenchmarkError("v2 provider requires the repository frozen v2 dataset")
    if fixture_path.as_posix() != DEFAULT_FIXTURE_PATH_V2.as_posix():
        raise V2ProviderBenchmarkError("v2 provider requires the repository frozen v2 fixture")
    try:
        dataset = load_chatbi_evaluation_dataset_v2(dataset_path)
        fixture_fingerprint = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise V2ProviderBenchmarkError("v2 frozen dataset or fixture is unavailable") from error
    if dataset.dataset_status != CHATBI_EVALUATION_V2_STATUS:
        raise V2ProviderBenchmarkError("v2 dataset is not human_reviewed_frozen")
    if dataset.fingerprint != EXPECTED_DATASET_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError("v2 dataset fingerprint does not match the frozen identity")
    if fixture_fingerprint != EXPECTED_FIXTURE_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError("v2 fixture fingerprint does not match the frozen identity")
    if dataset.datasource_fixture_sha256 != fixture_fingerprint:
        raise V2ProviderBenchmarkError("v2 dataset and fixture fingerprints disagree")
    return dataset, fixture_fingerprint


def _local_postgres_urls(settings: Settings) -> tuple[URL, URL]:
    """Return admin and deterministic benchmark URLs, only for local PostgreSQL."""
    try:
        base_url = make_url(settings.database_url)
    except Exception as error:
        raise V2ProviderBenchmarkError("benchmark database URL configuration is invalid") from error
    if base_url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise V2ProviderBenchmarkError("v2 benchmark bootstrap requires PostgreSQL")
    host = (base_url.host or "").strip().casefold().rstrip(".")
    if host not in _LOCAL_POSTGRES_HOSTS:
        raise V2ProviderBenchmarkError("v2 benchmark bootstrap requires a local PostgreSQL host")
    application_identity = _database_identity(base_url)
    benchmark_url = base_url.set(database=CHATBI_EVALUATION_V2_DATABASE_NAME)
    if application_identity == _database_identity(benchmark_url):
        raise V2ProviderBenchmarkError(
            "v2 benchmark database must be isolated from the application database"
        )
    return (
        base_url.set(database="postgres"),
        benchmark_url,
    )


def _database_identity(url: URL) -> tuple[str, int, str]:
    """Return a credential-free identity for one local PostgreSQL database."""
    host = (url.host or "").strip().casefold().rstrip(".")
    server = "<loopback>" if host in _LOCAL_POSTGRES_HOSTS else host
    database = (url.database or "").strip()
    if not server or not database:
        raise V2ProviderBenchmarkError(
            "v2 benchmark requires an explicit application database identity"
        )
    return server, url.port or 5432, database


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


_FIXTURE_SCHEMA_RELATIONS_QUERY = text(
    """
    SELECT
        c.relname AS relation_name,
        CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view' END AS relation_kind,
        a.attnum AS column_position,
        a.attname AS column_name,
        format_type(a.atttypid, a.atttypmod) AS data_type,
        a.attnotnull AS not_null
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid
    WHERE n.nspname = :schema
      AND c.relkind IN ('r', 'v')
      AND a.attnum > 0
      AND NOT a.attisdropped
    ORDER BY c.relkind, c.relname, a.attnum
    """
)
_FIXTURE_SCHEMA_CONSTRAINTS_QUERY = text(
    """
    SELECT
        c.relname AS relation_name,
        con.contype AS constraint_type,
        ARRAY(
            SELECT a.attname
            FROM unnest(con.conkey) WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = con.conrelid AND a.attnum = key.attnum
            ORDER BY key.position
        ) AS columns,
        rn.nspname AS referenced_schema,
        rc.relname AS referenced_relation,
        ARRAY(
            SELECT a.attname
            FROM unnest(con.confkey) WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = con.confrelid AND a.attnum = key.attnum
            ORDER BY key.position
        ) AS referenced_columns,
        CASE
            WHEN con.contype = 'c' THEN pg_catalog.pg_get_constraintdef(con.oid, true)
            ELSE NULL
        END AS check_definition
    FROM pg_catalog.pg_constraint AS con
    JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    LEFT JOIN pg_catalog.pg_class AS rc ON rc.oid = con.confrelid
    LEFT JOIN pg_catalog.pg_namespace AS rn ON rn.oid = rc.relnamespace
    WHERE n.nspname = :schema
      AND c.relkind = 'r'
      AND con.contype IN ('p', 'u', 'f', 'c')
    ORDER BY
        c.relname,
        con.contype,
        columns,
        referenced_schema,
        referenced_relation,
        referenced_columns,
        check_definition
    """
)


def _metadata_array(value: object) -> list[str]:
    """Convert a PostgreSQL text[] catalog value into a stable JSON list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise V2ProviderBenchmarkError("v2 fixture schema metadata has an invalid array value")


def _metadata_text(value: object) -> str:
    """Normalize text-like PostgreSQL catalog scalars without driver artifacts."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise V2ProviderBenchmarkError("v2 fixture schema metadata has an invalid text value")


def _fixture_schema_payload(
    relation_rows: Sequence[object],
    constraint_rows: Sequence[object],
) -> dict[str, object]:
    """Build a catalog-order-independent, credential-free schema contract."""
    relation_map: dict[tuple[str, str], dict[str, object]] = {}
    for row in relation_rows:
        mapping = row._mapping  # type: ignore[attr-defined]
        relation_name = _metadata_text(mapping["relation_name"])
        relation_kind = _metadata_text(mapping["relation_kind"])
        key = (relation_kind, relation_name)
        relation = relation_map.setdefault(
            key,
            {
                "name": relation_name,
                "kind": relation_kind,
                "columns": [],
                "constraints": [],
            },
        )
        columns = relation["columns"]
        if not isinstance(columns, list):
            raise V2ProviderBenchmarkError("v2 fixture schema metadata has an invalid column list")
        columns.append(
            {
                "position": int(mapping["column_position"]),
                "name": str(mapping["column_name"]),
                "type": str(mapping["data_type"]),
                "nullable": not bool(mapping["not_null"]),
            }
        )

    constraint_types = {
        "p": "primary_key",
        "u": "unique",
        "f": "foreign_key",
        "c": "check",
    }
    for row in constraint_rows:
        mapping = row._mapping  # type: ignore[attr-defined]
        relation_name = _metadata_text(mapping["relation_name"])
        constraint_type = _metadata_text(mapping["constraint_type"])
        if constraint_type not in constraint_types:
            raise V2ProviderBenchmarkError("v2 fixture schema metadata has an invalid constraint")
        key = ("table", relation_name)
        relation = relation_map.get(key)
        if relation is None:
            raise V2ProviderBenchmarkError(
                "v2 fixture constraint references an unexpected relation"
            )
        constraints = relation["constraints"]
        if not isinstance(constraints, list):
            raise V2ProviderBenchmarkError(
                "v2 fixture schema metadata has an invalid constraint list"
            )
        constraints.append(
            {
                "type": constraint_types[constraint_type],
                "columns": _metadata_array(mapping["columns"]),
                "referenced_schema": (
                    _metadata_text(mapping["referenced_schema"])
                    if mapping["referenced_schema"] is not None
                    else None
                ),
                "referenced_relation": (
                    _metadata_text(mapping["referenced_relation"])
                    if mapping["referenced_relation"] is not None
                    else None
                ),
                "referenced_columns": _metadata_array(mapping["referenced_columns"]),
                "check_definition": (
                    _metadata_text(mapping["check_definition"])
                    if mapping["check_definition"] is not None
                    else None
                ),
            }
        )

    relations = []
    for key in sorted(relation_map):
        relation = relation_map[key]
        columns = relation["columns"]
        constraints = relation["constraints"]
        assert isinstance(columns, list)
        assert isinstance(constraints, list)
        relation["columns"] = sorted(columns, key=lambda column: column["position"])
        relation["constraints"] = sorted(
            constraints,
            key=lambda constraint: _sha256_json(constraint),
        )
        relations.append(relation)
    return {"schema": CHATBI_EVALUATION_V2_SCHEMA, "relations": relations}


async def _fixture_schema_fingerprint(database_url: str) -> str:
    """Hash the authoritative benchmark schema from stable PostgreSQL metadata."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            relations = await connection.execute(
                _FIXTURE_SCHEMA_RELATIONS_QUERY,
                {"schema": CHATBI_EVALUATION_V2_SCHEMA},
            )
            constraints = await connection.execute(
                _FIXTURE_SCHEMA_CONSTRAINTS_QUERY,
                {"schema": CHATBI_EVALUATION_V2_SCHEMA},
            )
            payload = _fixture_schema_payload(relations.fetchall(), constraints.fetchall())
            return _sha256_json(payload)
    except V2ProviderBenchmarkError:
        raise
    except SQLAlchemyError as error:
        raise V2ProviderBenchmarkError("could not verify the v2 benchmark schema") from error
    finally:
        await engine.dispose()


async def _database_exists(admin_url: URL) -> bool:
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            result = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :database_name"),
                {"database_name": CHATBI_EVALUATION_V2_DATABASE_NAME},
            )
            return result == 1
    except SQLAlchemyError as error:
        raise V2ProviderBenchmarkError("could not inspect local PostgreSQL databases") from error
    finally:
        await engine.dispose()


async def _create_database(admin_url: URL) -> None:
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.exec_driver_sql(
                f"CREATE DATABASE {_quote_identifier(CHATBI_EVALUATION_V2_DATABASE_NAME)}"
            )
    except SQLAlchemyError as error:
        raise V2ProviderBenchmarkError(
            "could not create the isolated v2 benchmark database"
        ) from error
    finally:
        await engine.dispose()


async def _load_fixture(database_url: str, fixture_path: Path) -> None:
    try:
        fixture_sql = fixture_path.read_text(encoding="utf-8")
    except OSError as error:
        raise V2ProviderBenchmarkError(
            "could not read the isolated v2 benchmark fixture"
        ) from error
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            for statement in fixture_sql.split(";"):
                if statement.strip():
                    await connection.exec_driver_sql(statement)
    except SQLAlchemyError as error:
        raise V2ProviderBenchmarkError(
            "could not load the isolated v2 benchmark fixture"
        ) from error
    finally:
        await engine.dispose()


async def _fixture_data_fingerprint(database_url: str) -> str:
    """Verify the exact v2 table/view shape and return a safe data fingerprint."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            tables = await connection.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema AND table_type = 'BASE TABLE' "
                    "ORDER BY table_name"
                ),
                {"schema": CHATBI_EVALUATION_V2_SCHEMA},
            )
            table_names = [row[0] for row in tables]
            if table_names != ["customers", "regions", "sales"]:
                raise V2ProviderBenchmarkError(
                    "existing v2 benchmark database has the wrong tables"
                )
            views = await connection.execute(
                text(
                    "SELECT table_name FROM information_schema.views "
                    "WHERE table_schema = :schema ORDER BY table_name"
                ),
                {"schema": CHATBI_EVALUATION_V2_SCHEMA},
            )
            if [row[0] for row in views] != ["region_sales"]:
                raise V2ProviderBenchmarkError("existing v2 benchmark database has the wrong views")

            payload: dict[str, list[list[str]]] = {}
            queries = (
                (
                    "customers",
                    "SELECT customer_id::text, customer_name "
                    "FROM chatbi_demo.customers ORDER BY customer_id::bigint",
                ),
                (
                    "regions",
                    "SELECT region_code, region_name, market "
                    "FROM chatbi_demo.regions ORDER BY region_code",
                ),
                (
                    "sales",
                    "SELECT sale_id::text, customer_id::text, region_code, sold_on::text, "
                    "amount::text FROM chatbi_demo.sales ORDER BY sale_id::bigint",
                ),
            )
            for key, query in queries:
                rows = await connection.execute(text(query))
                payload[key] = [[str(value) for value in row] for row in rows]
            fingerprint = _sha256_json(payload)
            if fingerprint != EXPECTED_FIXTURE_DATA_FINGERPRINT_V2:
                raise V2ProviderBenchmarkError(
                    "existing v2 benchmark database does not match the frozen fixture data"
                )
            return fingerprint
    except V2ProviderBenchmarkError:
        raise
    except SQLAlchemyError as error:
        raise V2ProviderBenchmarkError("could not verify the v2 benchmark fixture data") from error
    finally:
        await engine.dispose()


async def _ensure_benchmark_database(
    settings: Settings, fixture_path: Path
) -> tuple[str, bool, str, str]:
    admin_url, benchmark_url = _local_postgres_urls(settings)
    database_url = benchmark_url.render_as_string(hide_password=False)
    exists = await _database_exists(admin_url)
    if not exists:
        await _create_database(admin_url)
        await _load_fixture(database_url, fixture_path)
        created = True
    else:
        created = False
    schema_fingerprint = await _fixture_schema_fingerprint(database_url)
    if schema_fingerprint != EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError(
            "existing v2 benchmark database does not match the frozen fixture schema"
        )
    return (
        database_url,
        created,
        schema_fingerprint,
        await _fixture_data_fingerprint(database_url),
    )


def _authoritative_record_matches(record: ChatBIDataSourceRecord) -> bool:
    """Check every persisted field of the deterministic evaluation identity."""
    return (
        record.id == CHATBI_EVALUATION_V2_DATASOURCE_ID
        and record.display_name == CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME
        and record.dialect == SQLDialect.POSTGRESQL.value
        and record.enabled is True
        and record.connection_ref == CHATBI_EVALUATION_V2_CONNECTION_REF
        and record.default_database == CHATBI_EVALUATION_V2_DATABASE_NAME
        and record.default_schema == CHATBI_EVALUATION_V2_SCHEMA
    )


async def _register_authoritative_datasource(
    settings: Settings,
    app_engine: AsyncEngine,
) -> tuple[DataSource, bool]:
    configured_id = settings.chatbi_evaluation_datasource_id
    if configured_id is not None and configured_id != CHATBI_EVALUATION_V2_DATASOURCE_ID:
        raise V2ProviderBenchmarkError(
            "configured ChatBI evaluation datasource is not the frozen v2 identity"
        )

    session_factory = create_session_factory(app_engine)
    async with session_factory() as session:
        matches = (
            await session.scalars(
                select(ChatBIDataSourceRecord).where(
                    or_(
                        ChatBIDataSourceRecord.id == CHATBI_EVALUATION_V2_DATASOURCE_ID,
                        ChatBIDataSourceRecord.display_name
                        == CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
                        ChatBIDataSourceRecord.connection_ref
                        == CHATBI_EVALUATION_V2_CONNECTION_REF,
                    )
                )
            )
        ).all()
        by_id = next(
            (record for record in matches if record.id == CHATBI_EVALUATION_V2_DATASOURCE_ID),
            None,
        )
        if len(matches) > 1 or (matches and by_id is None):
            raise V2ProviderBenchmarkError(
                "multiple or conflicting authoritative v2 datasource records exist"
            )
        created = by_id is None
        if by_id is None:
            by_id = ChatBIDataSourceRecord(
                id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
                display_name=CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
                dialect=SQLDialect.POSTGRESQL.value,
                enabled=True,
                connection_ref=CHATBI_EVALUATION_V2_CONNECTION_REF,
                default_database=CHATBI_EVALUATION_V2_DATABASE_NAME,
                default_schema=CHATBI_EVALUATION_V2_SCHEMA,
            )
            session.add(by_id)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                by_id = await session.get(
                    ChatBIDataSourceRecord,
                    CHATBI_EVALUATION_V2_DATASOURCE_ID,
                )
                if by_id is None:
                    raise V2ProviderBenchmarkError(
                        "authoritative v2 datasource registration raced and could not be verified"
                    ) from None
                created = False
            except SQLAlchemyError as error:
                await session.rollback()
                raise V2ProviderBenchmarkError(
                    "could not register the authoritative v2 datasource"
                ) from error
        if not _authoritative_record_matches(by_id):
            raise V2ProviderBenchmarkError(
                "registered authoritative v2 datasource metadata does not match the frozen identity"
            )
        return data_source_from_record(by_id), created


async def ensure_authoritative_v2_datasource(
    settings: Settings,
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH_V2,
) -> _V2DatasourceBootstrap:
    """Create or verify the isolated fixture and deterministic registry record.

    The process environment reference is installed for the lifetime of the
    caller's operation.  Benchmark entry points restore it in ``finally``;
    direct callers should do the same when the variable was previously absent.
    """
    if fixture_path.as_posix() != DEFAULT_FIXTURE_PATH_V2.as_posix():
        raise V2ProviderBenchmarkError("v2 bootstrap requires the repository frozen fixture path")
    try:
        fixture_fingerprint = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except OSError as error:
        raise V2ProviderBenchmarkError("v2 fixture is unavailable") from error
    if fixture_fingerprint != EXPECTED_FIXTURE_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError("v2 fixture fingerprint does not match the frozen identity")

    (
        database_url,
        database_created,
        schema_fingerprint,
        data_fingerprint,
    ) = await _ensure_benchmark_database(
        settings,
        fixture_path,
    )
    previous_environment_value = os.environ.get(CHATBI_EVALUATION_V2_CREDENTIAL_ENV)
    if previous_environment_value is not None and previous_environment_value != database_url:
        raise V2ProviderBenchmarkError(
            "the authoritative v2 credential environment reference is already bound elsewhere"
        )
    installed_environment = previous_environment_value is None
    if installed_environment:
        os.environ[CHATBI_EVALUATION_V2_CREDENTIAL_ENV] = database_url

    app_engine = create_database_engine(settings)
    try:
        data_source, datasource_created = await _register_authoritative_datasource(
            settings,
            app_engine,
        )
    except BaseException:
        if installed_environment:
            os.environ.pop(CHATBI_EVALUATION_V2_CREDENTIAL_ENV, None)
        raise
    finally:
        await app_engine.dispose()
    return _V2DatasourceBootstrap(
        data_source=data_source,
        database_url=database_url,
        fixture_schema_fingerprint=schema_fingerprint,
        fixture_data_fingerprint=data_fingerprint,
        database_created=database_created,
        datasource_created=datasource_created,
        environment_was_present=previous_environment_value is not None,
    )


def _v2_query_policy(settings: Settings) -> QueryPolicy:
    """Use production resource bounds with only the fixture business schema."""
    configured = default_query_policy(settings)
    return configured.model_copy(
        update={
            "allowed_schemas": (CHATBI_EVALUATION_V2_SCHEMA,),
            "denied_schemas": (),
            "allow_views": False,
        }
    )


async def _probe_v2_data(
    settings: Settings,
    *,
    datasource_id: UUID,
    session_factory: async_sessionmaker[AsyncSession],
    max_chars: int,
) -> str:
    """Verify fixture rows through the normal trusted read-only execution path."""
    discovery = create_postgres_schema_discovery_service(
        connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
        statement_timeout_ms=settings.chatbi_statement_timeout_ms,
    )
    registry = DatabaseDataSourceProvider(session_factory)
    validation = NL2SQLService(
        None,
        schema_discovery=discovery,
        data_source_provider=registry,
    )
    execution = SQLExecutionService(
        validation,
        EnvironmentCredentialResolver(),
        PostgresExecutionAdapter(
            connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
            statement_timeout_ms=settings.chatbi_statement_timeout_ms,
        ),
    )
    policy = _v2_query_policy(settings)
    probe_queries = (
        (
            "SELECT customer_id::text, customer_name "
            "FROM chatbi_demo.customers ORDER BY customer_id::bigint",
            "customers",
        ),
        (
            "SELECT region_code, region_name, market FROM chatbi_demo.regions ORDER BY region_code",
            "regions",
        ),
        (
            "SELECT sale_id::text, customer_id::text, region_code, sold_on::text, "
            "amount::text FROM chatbi_demo.sales ORDER BY sale_id::bigint",
            "sales",
        ),
    )
    payload: dict[str, list[list[str]]] = {}
    for sql, name in probe_queries:
        outcome = await execution.execute_sql_for_registered_data_source(
            datasource_id,
            sql,
            policy=policy,
            max_chars=max_chars,
        )
        result = outcome.result
        if result.state is not QueryLifecycleState.SUCCEEDED or result.truncated:
            raise V2ProviderBenchmarkError("v2 fixture data probe did not complete safely")
        payload[name] = [[str(value) for value in row] for row in result.rows]
    fingerprint = _sha256_json(payload)
    if fingerprint != EXPECTED_FIXTURE_DATA_FINGERPRINT_V2:
        raise V2ProviderBenchmarkError("v2 fixture data probe does not match the frozen fixture")
    return fingerprint


async def _discover_and_verify_v2_schema(
    settings: Settings,
    data_source: DataSource,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    max_chars: int,
) -> str:
    """Resolve schema metadata through the normal discovery implementation."""
    policy = _v2_query_policy(settings)
    discovery = create_postgres_schema_discovery_service(
        connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
        statement_timeout_ms=settings.chatbi_statement_timeout_ms,
    )
    discovered = await discovery.discover(data_source, policy, max_chars=max_chars)
    if not isinstance(discovered, SchemaDiscoveryResult):
        raise V2ProviderBenchmarkError("v2 schema discovery returned an invalid result")
    snapshot = discovered.snapshot
    if snapshot.datasource_id != CHATBI_EVALUATION_V2_DATASOURCE_ID:
        raise V2ProviderBenchmarkError("v2 schema discovery datasource identity mismatch")
    if snapshot.schemas != (CHATBI_EVALUATION_V2_SCHEMA,):
        raise V2ProviderBenchmarkError("v2 schema discovery returned an unexpected schema set")
    tables = {
        relation.name
        for relation in snapshot.relations
        if relation.schema_name == CHATBI_EVALUATION_V2_SCHEMA
        and relation.kind is SchemaObjectKind.TABLE
    }
    if tables != {"customers", "regions", "sales"}:
        raise V2ProviderBenchmarkError("v2 schema discovery returned an unexpected table set")
    return await _probe_v2_data(
        settings,
        datasource_id=data_source.id,
        session_factory=session_factory,
        max_chars=max_chars,
    )


def _check_provider_configuration(settings: Settings) -> None:
    """Check configuration without constructing a provider or making a call."""
    if settings.llm_provider != "deepseek":
        raise V2ProviderBenchmarkError("v2 provider evaluation requires the DeepSeek provider")
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise V2ProviderBenchmarkError("DeepSeek provider configuration is incomplete")
    if not settings.llm_model.strip() or not settings.llm_base_url.strip():
        raise V2ProviderBenchmarkError("LLM provider model/base URL configuration is incomplete")


def _check_output_path(path: Path | None) -> None:
    """Validate the ignored destination without writing benchmark results."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise V2ProviderBenchmarkError("v2 provider output path is not a file")
    except OSError as error:
        raise V2ProviderBenchmarkError("v2 provider output path is not writable") from error


@dataclass(frozen=True, slots=True)
class _V2PreflightContext:
    dataset: ChatBIEvaluationDatasetV2
    selected_cases: tuple[ChatBIEvaluationCaseV2, ...]
    fixture_fingerprint: str
    fixture_schema_fingerprint: str
    bootstrap: _V2DatasourceBootstrap
    fixture_data_fingerprint: str
    git_revision: str
    git_dirty: bool


async def _prepare_v2_preflight(
    settings: Settings,
    *,
    split: str,
    dataset_path: Path,
    fixture_path: Path,
    output_path: Path | None,
) -> _V2PreflightContext:
    # Capture the repository state before any validation/bootstrap step can
    # create directories or other local artifacts.
    revision = current_git_revision()
    if revision is None:
        raise V2ProviderBenchmarkError("could not determine the current Git revision")
    dirty = current_git_dirty()
    dataset, fixture_fingerprint = _verify_v2_dataset_and_fixture(dataset_path, fixture_path)
    selected = tuple(select_v2_provider_cases(dataset, split))
    _check_provider_configuration(settings)
    bootstrap = await ensure_authoritative_v2_datasource(settings, fixture_path=fixture_path)
    app_engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(app_engine)
        registered = await DatabaseDataSourceProvider(session_factory).get(
            CHATBI_EVALUATION_V2_DATASOURCE_ID
        )
        if registered is None or registered.id != bootstrap.data_source.id:
            raise V2ProviderBenchmarkError("authoritative v2 datasource could not be resolved")
        data_fingerprint = await _discover_and_verify_v2_schema(
            settings,
            registered,
            session_factory=session_factory,
            max_chars=settings.chatbi_schema_context_max_chars,
        )
    except BaseException:
        if not bootstrap.environment_was_present:
            os.environ.pop(CHATBI_EVALUATION_V2_CREDENTIAL_ENV, None)
        raise
    finally:
        await app_engine.dispose()
    _check_output_path(output_path)
    return _V2PreflightContext(
        dataset=dataset,
        selected_cases=selected,
        fixture_fingerprint=fixture_fingerprint,
        fixture_schema_fingerprint=bootstrap.fixture_schema_fingerprint,
        bootstrap=bootstrap,
        fixture_data_fingerprint=data_fingerprint,
        git_revision=revision,
        git_dirty=dirty,
    )


async def preflight_v2_provider_benchmark(
    settings: Settings,
    *,
    split: str,
    dataset_path: Path = DEFAULT_DATASET_V2,
    fixture_path: Path = DEFAULT_FIXTURE_PATH_V2,
    output_path: Path | None = None,
) -> V2ProviderPreflightReport:
    """Run all provider-free gates and make zero provider calls."""
    environment_was_present = CHATBI_EVALUATION_V2_CREDENTIAL_ENV in os.environ
    try:
        context = await _prepare_v2_preflight(
            settings,
            split=split,
            dataset_path=dataset_path,
            fixture_path=fixture_path,
            output_path=output_path,
        )
        return V2ProviderPreflightReport(
            dataset_fingerprint=context.dataset.fingerprint,
            fixture_fingerprint=context.fixture_fingerprint,
            fixture_schema_fingerprint=context.fixture_schema_fingerprint,
            fixture_data_fingerprint=context.fixture_data_fingerprint,
            datasource_id=context.bootstrap.data_source.id,
            split="dev",
            selected_case_count=len(context.selected_cases),
            selected_test_case_count=0,
            provider=settings.llm_provider,
            model=settings.llm_model,
            provider_configured=True,
            database_created=context.bootstrap.database_created,
            datasource_created=context.bootstrap.datasource_created,
            git_revision=context.git_revision,
            git_dirty=context.git_dirty,
        )
    finally:
        if not environment_was_present:
            os.environ.pop(CHATBI_EVALUATION_V2_CREDENTIAL_ENV, None)


def _v2_expected_failure_match(
    case: ChatBIEvaluationCaseV2,
    result: ChatBIResult,
) -> bool | None:
    if case.expected_failure_category is None:
        return None
    if result.error_category is None or result.execution_status is QueryLifecycleState.SUCCEEDED:
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
    }
    return result.error_category in allowed.get(case.expected_failure_category, frozenset())


def _v2_record(
    case: ChatBIEvaluationCaseV2,
    observation: ChatBIEvaluationObservation,
) -> V2ProviderCaseRecord:
    if case.split != V2ProviderSplit.DEV.value:
        raise V2ProviderBenchmarkError("v2 provider records accept DEV cases only")
    result = observation.result
    execution_result = observation.execution_result
    stages = observation.stages
    if result.error_category is ChatBIErrorCategory.ANALYSIS_FAILED:
        stages = stages.model_copy(update={"analysis": "failed"})

    comparison = (
        compare_normalized_result(execution_result, case.expected_result)
        if case.expected_result is not None and execution_result is not None
        else None
    )
    execution_equivalent = comparison.equivalent if comparison is not None else None
    structured_coverage = (
        structured_result_facts_match(execution_result, case.structured_answer_facts)
        if case.structured_answer_facts and execution_result is not None
        else (False if case.structured_answer_facts else None)
    )
    repair_attempted = result.repair_attempts > 0
    if not case.repair_applicable:
        repair_outcome = RepairOutcome.NOT_APPLICABLE
    elif not repair_attempted:
        repair_outcome = RepairOutcome.NOT_ATTEMPTED
    elif result.execution_status is QueryLifecycleState.SUCCEEDED:
        repair_outcome = RepairOutcome.REPAIRED
    else:
        repair_outcome = RepairOutcome.EXHAUSTED

    if result.error_category is ChatBIErrorCategory.ANALYSIS_FAILED:
        failure_category = ChatBIEvaluationFailureCategory.ANALYSIS
    elif case.positive and result.execution_status is QueryLifecycleState.SUCCEEDED:
        failure_category = (
            ChatBIEvaluationFailureCategory.RESULT_MISMATCH
            if execution_equivalent is False
            else None
        )
    else:
        failure_category = _map_failure_category(result)

    return V2ProviderCaseRecord(
        case_id=case.case_id,
        split=V2ProviderSplit.DEV.value,
        positive=case.positive,
        category=case.category,
        difficulty=case.difficulty,
        stages=stages,
        result_status=result.execution_status,
        generation_parseable=stages.generation == "passed",
        validation_accepted=stages.validation == "passed",
        execution_equivalent=execution_equivalent,
        structured_result_fact_coverage=structured_coverage,
        expected_failure_match=_v2_expected_failure_match(case, result),
        repair_attempted=repair_attempted,
        repair_outcome=repair_outcome,
        failure_category=failure_category,
        row_count=result.row_count,
        truncated=result.truncated,
        truncation_reason=result.truncation_reason,
        latency=observation.timings,
        usage=result.usage,
        redacted_sql=result.redacted_sql,
        trace_events=[event.event for event in result.trace],
    )


async def evaluate_v2_provider_cases(
    cases: Sequence[ChatBIEvaluationCaseV2],
    runner: V2ProviderCaseRunner,
) -> list[V2ProviderCaseRecord]:
    """Evaluate injected cases through the same Agent runner used in production."""
    validated_cases = _validate_frozen_dev_cases(cases)
    records: list[V2ProviderCaseRecord] = []
    for case in validated_cases:
        observation = await runner.run(case)
        if not isinstance(observation, ChatBIEvaluationObservation):
            raise V2ProviderBenchmarkError(
                "v2 provider case runner returned an invalid observation"
            )
        records.append(_v2_record(case, observation))
    return records


def _v2_configuration(
    settings: Settings,
    policy: QueryPolicy,
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
        query_policy=EvaluationQueryPolicy.model_validate(policy.model_dump(mode="json")),
    )


class _V2AgentRunner:
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

    async def run(self, case: ChatBIEvaluationCaseV2) -> ChatBIEvaluationObservation:
        self._timing.reset()
        started = perf_counter()
        try:
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


async def run_v2_provider_benchmark(
    settings: Settings,
    *,
    split: str = "dev",
    dataset_path: Path = DEFAULT_DATASET_V2,
    fixture_path: Path = DEFAULT_FIXTURE_PATH_V2,
    output_path: Path | None = DEFAULT_PROVIDER_OUTPUT_V2,
) -> V2ProviderRun:
    """Run exactly the frozen v2 DEV cases through the real Agent path.

    Preflight performs all local fixture, registry, discovery, and execution
    checks before this function constructs the provider.  ``split=test`` is
    deliberately rejected by :func:`select_v2_provider_cases`.
    """
    environment_was_present = CHATBI_EVALUATION_V2_CREDENTIAL_ENV in os.environ
    started_at = datetime.now(UTC)
    try:
        context = await _prepare_v2_preflight(
            settings,
            split=split,
            dataset_path=dataset_path,
            fixture_path=fixture_path,
            output_path=output_path,
        )
        app_engine = create_database_engine(settings)
        provider = None
        try:
            session_factory = create_session_factory(app_engine)
            registry = DatabaseDataSourceProvider(session_factory)
            registered = await registry.get(CHATBI_EVALUATION_V2_DATASOURCE_ID)
            if registered is None:
                raise V2ProviderBenchmarkError(
                    "authoritative v2 datasource disappeared after preflight"
                )

            # Provider construction is intentionally after the provider-free gate.
            provider = create_llm_provider(settings)
            timing = _TimingState()
            discovery = create_postgres_schema_discovery_service(
                connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                statement_timeout_ms=settings.chatbi_statement_timeout_ms,
            )
            gateway = LLMGateway(
                provider,
                DatabaseUsageRecorder(session_factory),
                settings,
            )
            timed_discovery = _TimedDiscovery(discovery, timing)
            generation = NL2SQLService(
                _TimedGenerationGateway(gateway, timing),
                schema_discovery=timed_discovery,
                data_source_provider=registry,
                max_tokens=settings.chatbi_nl2sql_max_tokens,
            )
            timed_generation = _TimedGeneration(generation, timing)
            policy = _v2_query_policy(settings)
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
            records = await evaluate_v2_provider_cases(
                context.selected_cases,
                _V2AgentRunner(
                    agent,
                    timing,
                    CHATBI_EVALUATION_V2_DATASOURCE_ID,
                    policy,
                    settings.chatbi_schema_context_max_chars,
                ),
            )
            completed_at = datetime.now(UTC)
            provenance = V2ProviderRunProvenance(
                started_at=started_at,
                completed_at=completed_at,
                dataset_fingerprint=context.dataset.fingerprint,
                fixture_path=DEFAULT_FIXTURE_PATH_V2.as_posix(),
                fixture_sha256=context.fixture_fingerprint,
                fixture_schema_fingerprint=context.fixture_schema_fingerprint,
                fixture_data_fingerprint=context.fixture_data_fingerprint,
                datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
                case_count=len(records),
                git_revision=context.git_revision,
                git_dirty=context.git_dirty,
                provider=settings.llm_provider,
                model=settings.llm_model,
                temperature=0.0,
                response_format="json_object",
                configuration=_v2_configuration(
                    settings,
                    policy,
                    settings.chatbi_schema_context_max_chars,
                ),
            )
            run = V2ProviderRun(
                run_id=uuid4(),
                started_at=started_at,
                completed_at=completed_at,
                dataset_fingerprint=context.dataset.fingerprint,
                fixture_fingerprint=context.fixture_fingerprint,
                fixture_schema_fingerprint=context.fixture_schema_fingerprint,
                datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
                case_count=len(records),
                records=records,
                aggregates=_build_v2_aggregate(records),
                provenance=provenance,
            )
            if output_path is not None:
                write_v2_provider_run(output_path, run)
            return run
        finally:
            if provider is not None:
                await provider.aclose()
            await app_engine.dispose()
    finally:
        if not environment_was_present:
            os.environ.pop(CHATBI_EVALUATION_V2_CREDENTIAL_ENV, None)


def write_v2_provider_run(path: Path, run: V2ProviderRun) -> None:
    """Write a safe runtime artifact; the default path is gitignored."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                run.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise V2ProviderBenchmarkError("could not write v2 provider benchmark output") from error


__all__ = [
    "CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME",
    "CHATBI_EVALUATION_V2_DATASOURCE_ID",
    "CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION",
    "CHATBI_EVALUATION_V2_PROVIDER_VERSION",
    "DEFAULT_PROVIDER_OUTPUT_V2",
    "EXPECTED_DATASET_FINGERPRINT_V2",
    "EXPECTED_FIXTURE_DATA_FINGERPRINT_V2",
    "EXPECTED_FIXTURE_FINGERPRINT_V2",
    "EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2",
    "V2ProviderAggregate",
    "V2ProviderBenchmarkError",
    "V2ProviderCaseRecord",
    "V2ProviderPreflightReport",
    "V2ProviderRun",
    "V2ProviderRunProvenance",
    "V2ProviderSplit",
    "ensure_authoritative_v2_datasource",
    "evaluate_v2_provider_cases",
    "preflight_v2_provider_benchmark",
    "run_v2_provider_benchmark",
    "select_v2_provider_cases",
    "write_v2_provider_run",
]
