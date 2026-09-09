"""Grounded, provider-independent graph extraction for one canonical chunk."""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope.chunking.models import Chunk
from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphEntity,
    GraphProvenance,
    GraphRelation,
    entity_id_for,
    relation_id_for,
)
from knowledge_scope.graph.neo4j import GraphStoreError, Neo4jGraphStore
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.schemas import LLMRequest, LLMResponseFormat, LLMResult
from knowledge_scope.llm.usage import estimate_cost
from knowledge_scope.shared.config import Settings

from .models import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATION_TYPES,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionOutput,
    RelationType,
    normalize_label,
)
from .prompt import EXTRACTION_PROMPT_VERSION, build_extraction_messages

ExtractionStatus = Literal["accepted", "empty", "rejected", "schema_rejected"]
GroundingRejectionReason = Literal[
    "entity_mention_not_grounded",
    "relation_endpoint_not_in_output",
    "relation_source_not_grounded",
    "relation_target_not_grounded",
    "relation_evidence_not_in_chunk",
]
ExtractionAttemptCategory = Literal[
    "valid_extraction",
    "empty_valid_extraction",
    "grounding_rejection",
    "invalid_json",
    "markdown_code_fence",
    "extra_prose",
    "schema_validation",
    "unknown_entity_type",
    "unknown_relation_type",
    "relation_unknown_entity",
    "truncated_response",
    "empty_response",
    "provider_error",
    "timeout",
    "connection",
    "api",
    "malformed_response",
    "cancelled",
    "other",
]
ExtractionTransportFormat = Literal["plain_json", "json_code_fence", "extra_prose"]


class ExtractionError(RuntimeError):
    """A safe, non-sensitive extraction failure."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "extraction",
        attempts: Sequence[ExtractionAttempt] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = tuple(attempts)


class ExtractionPersistenceError(ExtractionError):
    """Raised when a validated chunk extraction cannot be persisted atomically."""

    def __init__(self, *, result: ChunkExtractionResult | None = None) -> None:
        self.result = result
        super().__init__(
            "validated extraction could not be persisted",
            category="persistence",
            attempts=result.attempts if result is not None else (),
        )


class ExtractionParseError(ValueError):
    """Raised when the provider response is not a JSON object."""

    def __init__(self, message: str = "LLM extraction output was not valid JSON") -> None:
        super().__init__(message)
        self.issue_category: ExtractionAttemptCategory = "invalid_json"


class ExtractionSchemaError(ValueError):
    """Raised when a JSON object does not match the extraction contract."""

    def __init__(
        self,
        message: str = "LLM extraction output failed the extraction schema",
        *,
        issue_categories: Sequence[ExtractionAttemptCategory] = (),
        issue_details: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.issue_categories = tuple(issue_categories)
        self.issue_details = tuple(issue_details)


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    """Safe per-provider-attempt diagnostics; never contains model output."""

    number: int
    category: ExtractionAttemptCategory
    transport_format: ExtractionTransportFormat | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroundedRelationEvidence:
    """A verified bounded quote retained for review, not graph identity."""

    relation_id: str
    source: str
    relation_type: RelationType
    target: str
    evidence: str


class ExtractionGateway(Protocol):
    """The small gateway surface required by the extraction service."""

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Complete one structured extraction request."""


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    """Deterministic counters for one chunk and all parse retries."""

    attempts: int
    parse_failures: int
    schema_failures: int
    grounding_rejections: int
    duplicate_entities: int
    duplicate_relations: int
    skipped_no_text: bool = False
    grounding_rejection_reasons: tuple[GroundingRejectionReason, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkExtractionResult:
    """Validated graph facts and safe observations for one source chunk."""

    document_id: UUID
    chunk_id: str
    status: ExtractionStatus
    entities: tuple[GraphEntity, ...]
    relations: tuple[GraphRelation, ...]
    stats: ExtractionStats
    grounded_relation_evidence: tuple[GroundedRelationEvidence, ...] = ()
    llm_results: tuple[LLMResult, ...] = ()
    attempts: tuple[ExtractionAttempt, ...] = ()
    error: str | None = None

    @property
    def input_tokens(self) -> int | None:
        return _sum_known_tokens(self.llm_results, "input_tokens")

    @property
    def output_tokens(self) -> int | None:
        return _sum_known_tokens(self.llm_results, "output_tokens")

    @property
    def latency_ms(self) -> float:
        return sum(result.latency_ms for result in self.llm_results)

    def estimated_cost(self, settings: Settings) -> Decimal | None:
        if not self.llm_results:
            return None
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return estimate_cost(self.input_tokens, self.output_tokens, settings)


def _sum_known_tokens(results: Sequence[LLMResult], field_name: str) -> int | None:
    values = [getattr(result, field_name) for result in results]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _transport_format(value: str) -> ExtractionTransportFormat | None:
    """Classify harmless transport wrapping without accepting it as semantics."""

    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
            return "json_code_fence"
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start > 0 or (end >= 0 and end < len(stripped) - 1):
            try:
                json.loads(stripped[start : end + 1])
            except (TypeError, ValueError):
                pass
            else:
                return "extra_prose"
    return "plain_json"


def _schema_issue_info(
    error: ValidationError,
) -> tuple[tuple[ExtractionAttemptCategory, ...], tuple[str, ...]]:
    """Map Pydantic errors to safe categories and field/type diagnostics.

    Only the field path and Pydantic error type are retained.  In particular, do
    not include ``msg`` or ``input`` because either may echo unrestricted model
    output into runtime artifacts or a retry prompt.
    """

    safe_path_parts = {
        "entities",
        "relations",
        "name",
        "entity_type",
        "aliases",
        "source",
        "target",
        "relation_type",
        "evidence",
    }
    categories: list[ExtractionAttemptCategory] = []
    details: list[str] = []
    for issue in error.errors():
        location = issue.get("loc", ())
        issue_type = issue.get("type")
        if issue_type == "literal_error" and "entity_type" in location:
            categories.append("unknown_entity_type")
        elif issue_type == "literal_error" and "relation_type" in location:
            categories.append("unknown_relation_type")
        else:
            categories.append("schema_validation")
        path_parts: list[str] = []
        for part in location:
            if isinstance(part, int):
                if path_parts:
                    path_parts[-1] = f"{path_parts[-1]}[{part}]"
                else:
                    path_parts.append(f"[{part}]")
            elif isinstance(part, str) and part in safe_path_parts:
                path_parts.append(str(part))
            else:
                path_parts.append("unknown_field")
        path = ".".join(path_parts) or "root"
        safe_type = str(issue_type or "validation_error")
        details.append(f"{path}: {safe_type}")
    return (
        tuple(dict.fromkeys(categories)) or ("schema_validation",),
        tuple(dict.fromkeys(details)) or ("root: schema_validation",),
    )


def parse_extraction_output(text: str) -> ExtractionOutput:
    """Parse only the application-owned extraction JSON contract."""

    try:
        payload = json.loads(_strip_json_fence(text))
    except (TypeError, json.JSONDecodeError) as error:
        raise ExtractionParseError("LLM extraction output was not valid JSON") from error
    if not isinstance(payload, dict):
        raise ExtractionSchemaError(
            "LLM extraction output must be a JSON object",
            issue_categories=("schema_validation",),
            issue_details=("root: model_type",),
        )
    try:
        return ExtractionOutput.model_validate(payload)
    except ValidationError as error:
        issue_categories, issue_details = _schema_issue_info(error)
        raise ExtractionSchemaError(
            "LLM extraction output failed schema validation",
            issue_categories=issue_categories,
            issue_details=issue_details,
        ) from error


def _grounded(value: str, source_text: str) -> bool:
    return normalize_label(value) in normalize_label(source_text)


def _entity_mention_grounded(
    entity_id: str,
    evidence: str,
    mentions_by_entity_id: dict[str, tuple[str, ...]],
) -> bool:
    """Check that evidence supports an accepted entity name or alias."""

    return any(_grounded(mention, evidence) for mention in mentions_by_entity_id.get(entity_id, ()))


def _entity_key(entity: ExtractedEntity) -> tuple[str, str]:
    return normalize_label(entity.name), entity.entity_type


def _relation_key(relation: ExtractedRelation) -> tuple[str, str, str]:
    return (
        normalize_label(relation.source),
        normalize_label(relation.target),
        relation.relation_type,
    )


def _has_unknown_relation_entity(output: ExtractionOutput) -> bool:
    entity_mentions = {
        normalize_label(mention)
        for entity in output.entities
        for mention in (entity.name, *entity.aliases)
    }
    return any(
        normalize_label(relation.source) not in entity_mentions
        or normalize_label(relation.target) not in entity_mentions
        for relation in output.relations
    )


def _structured_success(category: ExtractionAttemptCategory) -> bool:
    return category in {
        "valid_extraction",
        "empty_valid_extraction",
        "grounding_rejection",
    }


def _corrective_instruction(
    category: ExtractionAttemptCategory,
    details: Sequence[str] = (),
) -> str:
    """Return safe feedback for one complete corrective regeneration.

    ``details`` contains only application-generated categories and field/type
    diagnostics; it must never contain model output values.
    """

    if category == "truncated_response":
        return (
            "上一轮响应被截断（finish_reason=length）；请重新生成完整 extraction payload，"
            "减少抽取项，只输出完整 JSON object。"
        )
    if category in {"empty_response", "invalid_json"}:
        return (
            "上一轮不是完整 JSON object；请重新生成完整 extraction payload，"
            "只输出一个完整 JSON object，不要解释或 Markdown。"
        )
    if category == "markdown_code_fence":
        return (
            "请重新生成完整 extraction payload；不要使用 Markdown code fence，只输出 JSON object。"
        )
    if category == "extra_prose":
        return "请重新生成完整 extraction payload；不要在 JSON object 前后添加解释，只输出 JSON。"
    safe_details = tuple(
        detail
        for detail in details
        if detail
        not in {
            "unknown_entity_type",
            "unknown_relation_type",
            "schema_validation",
        }
    )
    if category in {"unknown_entity_type", "unknown_relation_type", "schema_validation"}:
        entity_types = "、".join(ALLOWED_ENTITY_TYPES)
        relation_types = "、".join(ALLOWED_RELATION_TYPES)
        issues = "；".join(safe_details) or "字段未通过 extraction schema"
        return (
            "上一轮完整 JSON 已被应用 schema 拒绝；问题位置/类型："
            f"{issues}。entity_type 只能逐字使用：{entity_types}；"
            f"relation_type 只能逐字使用：{relation_types}。"
            "请重新生成完整 extraction payload，不要修补或复述上一轮输出。"
        )
    if category == "relation_unknown_entity":
        return (
            "请重新生成完整 extraction payload；每条 relation 的 source 和 target 必须引用"
            "同一输出中的 entity name 或 alias，否则删除该 relation。"
        )
    if details:
        return "上一轮未通过应用校验；请重新生成完整 extraction payload，只输出 JSON object。"
    return (
        "上一轮输出未通过应用校验；请重新生成完整 extraction payload，"
        "只输出符合 contract 的 JSON object。"
    )


def _sorted_aliases(aliases: Iterable[str], *, canonical_name: str) -> list[str]:
    canonical_key = normalize_label(canonical_name)
    by_key: dict[str, str] = {}
    for alias in aliases:
        key = normalize_label(alias)
        if key and key != canonical_key:
            by_key.setdefault(key, alias)
    return [by_key[key] for key in sorted(by_key)]


def _build_provenance(
    chunk: Chunk,
    knowledge_base_id: UUID,
    llm_result: LLMResult,
) -> GraphProvenance:
    return GraphProvenance(
        document_id=chunk.document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        source_block_ids=list(chunk.source_block_ids),
        section_path=list(chunk.section_path),
        extraction_provenance=ExtractionProvenance(
            method=llm_result.provider,
            model=llm_result.model,
            version=EXTRACTION_PROMPT_VERSION,
        ),
    )


def _materialize_output(
    output: ExtractionOutput,
    chunk: Chunk,
    knowledge_base_id: UUID,
    llm_result: LLMResult,
) -> tuple[
    tuple[GraphEntity, ...],
    tuple[GraphRelation, ...],
    int,
    int,
    int,
    tuple[GroundingRejectionReason, ...],
    tuple[GroundedRelationEvidence, ...],
]:
    """Ground, deduplicate, and convert model output without trusting its metadata."""

    source_text = chunk.text
    grouped_entities: dict[tuple[str, str], list[ExtractedEntity]] = {}
    duplicate_entities = 0
    for item in sorted(output.entities, key=lambda value: (_entity_key(value), value.name)):
        key = _entity_key(item)
        if key in grouped_entities:
            duplicate_entities += 1
        grouped_entities.setdefault(key, []).append(item)

    accepted_entities: list[GraphEntity] = []
    entity_ids_by_mention: dict[str, set[str]] = {}
    mentions_by_entity_id: dict[str, tuple[str, ...]] = {}
    grounding_rejections = 0
    grounding_rejection_reasons: list[GroundingRejectionReason] = []
    provenance = _build_provenance(chunk, knowledge_base_id, llm_result)
    for _key, candidates in sorted(grouped_entities.items()):
        first = candidates[0]
        aliases = _sorted_aliases(
            [alias for candidate in candidates for alias in candidate.aliases],
            canonical_name=first.name,
        )
        grounded_name = _grounded(first.name, source_text)
        grounded_aliases = [_grounded(alias, source_text) for alias in aliases]
        if not grounded_name and not any(grounded_aliases):
            grounding_rejections += 1
            grounding_rejection_reasons.append("entity_mention_not_grounded")
            continue
        if any(not grounded for grounded in grounded_aliases):
            grounding_rejections += 1
            grounding_rejection_reasons.append("entity_mention_not_grounded")
            continue
        entity_id = entity_id_for(
            first.name,
            first.entity_type,
            knowledge_base_id=knowledge_base_id,
            document_id=chunk.document_id,
        )
        entity = GraphEntity(
            entity_id=entity_id,
            knowledge_base_id=knowledge_base_id,
            document_id=chunk.document_id,
            canonical_name=first.name,
            entity_type=first.entity_type,
            aliases=aliases,
            provenance=[provenance],
        )
        accepted_entities.append(entity)
        mentions = tuple(dict.fromkeys((first.name, *aliases)))
        mentions_by_entity_id[entity_id] = mentions
        for mention in mentions:
            entity_ids_by_mention.setdefault(normalize_label(mention), set()).add(entity_id)

    entity_lookup = {
        mention: next(iter(entity_ids))
        for mention, entity_ids in entity_ids_by_mention.items()
        if len(entity_ids) == 1
    }
    grouped_relations: dict[tuple[str, str, str], list[ExtractedRelation]] = {}
    duplicate_relations = 0
    for item in sorted(output.relations, key=lambda value: (_relation_key(value), value.source)):
        key = _relation_key(item)
        if key in grouped_relations:
            duplicate_relations += 1
        grouped_relations.setdefault(key, []).append(item)

    accepted_relations: list[GraphRelation] = []
    grounded_relation_evidence: list[GroundedRelationEvidence] = []
    accepted_relation_ids: set[str] = set()
    for key, candidates in sorted(grouped_relations.items()):
        item = candidates[0]
        source_id = entity_lookup.get(key[0])
        target_id = entity_lookup.get(key[1])
        if source_id is None or target_id is None:
            grounding_rejections += 1
            grounding_rejection_reasons.append("relation_endpoint_not_in_output")
            continue
        if not _grounded(item.evidence, source_text):
            grounding_rejections += 1
            grounding_rejection_reasons.append("relation_evidence_not_in_chunk")
            continue
        if not _entity_mention_grounded(source_id, item.evidence, mentions_by_entity_id):
            grounding_rejections += 1
            grounding_rejection_reasons.append("relation_source_not_grounded")
            continue
        if not _entity_mention_grounded(target_id, item.evidence, mentions_by_entity_id):
            grounding_rejections += 1
            grounding_rejection_reasons.append("relation_target_not_grounded")
            continue
        relation_id = relation_id_for(
            source_id,
            target_id,
            item.relation_type,
            knowledge_base_id=knowledge_base_id,
            document_id=chunk.document_id,
        )
        if relation_id in accepted_relation_ids:
            duplicate_relations += 1
            continue
        accepted_relation_ids.add(relation_id)
        accepted_relations.append(
            GraphRelation(
                relation_id=relation_id,
                knowledge_base_id=knowledge_base_id,
                document_id=chunk.document_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=item.relation_type,
                provenance=[provenance],
            )
        )
        grounded_relation_evidence.append(
            GroundedRelationEvidence(
                relation_id=relation_id,
                source=item.source,
                relation_type=item.relation_type,
                target=item.target,
                evidence=item.evidence,
            )
        )
    return (
        tuple(accepted_entities),
        tuple(accepted_relations),
        grounding_rejections,
        duplicate_entities,
        duplicate_relations,
        tuple(grounding_rejection_reasons),
        tuple(grounded_relation_evidence),
    )


class ExtractionService:
    """Run one grounded extraction and optionally persist it as one graph transaction."""

    def __init__(
        self,
        gateway: ExtractionGateway,
        settings: Settings,
        *,
        max_parse_retries: int | None = None,
    ) -> None:
        self._gateway = gateway
        self._settings = settings
        self._max_parse_retries = (
            settings.graph_extraction_max_parse_retries
            if max_parse_retries is None
            else max_parse_retries
        )
        if self._max_parse_retries < 0 or self._max_parse_retries > 1:
            raise ValueError("max_parse_retries must be between zero and one")

    async def extract_chunk(
        self,
        chunk: Chunk,
        *,
        knowledge_base_id: UUID,
    ) -> ChunkExtractionResult:
        """Extract one chunk; no graph write occurs in this method."""

        if not chunk.text.strip():
            return ChunkExtractionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                status="empty",
                entities=(),
                relations=(),
                stats=ExtractionStats(
                    attempts=0,
                    parse_failures=0,
                    schema_failures=0,
                    grounding_rejections=0,
                    duplicate_entities=0,
                    duplicate_relations=0,
                    skipped_no_text=True,
                ),
            )

        request = LLMRequest(
            messages=build_extraction_messages(chunk),
            task_type="graph_extraction",
            temperature=0.0,
            max_tokens=self._settings.graph_extraction_max_tokens,
            model=self._settings.llm_model,
            response_format=LLMResponseFormat(type="json_object"),
            reasoning="disabled",
        )
        llm_results: list[LLMResult] = []
        attempts: list[ExtractionAttempt] = []
        parse_failures = 0
        schema_failures = 0
        for attempt in range(self._max_parse_retries + 1):
            try:
                result = await self._gateway.complete(request)
            except asyncio.CancelledError:
                attempts.append(
                    ExtractionAttempt(
                        number=len(attempts) + 1,
                        category="cancelled",
                    )
                )
                raise
            except LLMError as error:
                category: ExtractionAttemptCategory = (
                    error.category
                    if error.category in {"timeout", "connection", "api", "malformed_response"}
                    else "provider_error"
                )
                attempts.append(
                    ExtractionAttempt(
                        number=len(attempts) + 1,
                        category=category,
                        details=(error.category,),
                    )
                )
                raise ExtractionError(
                    "LLM extraction request failed",
                    category=error.category,
                    attempts=attempts,
                ) from error
            llm_results.append(result)
            transport_format = _transport_format(result.text)
            if result.finish_reason == "length":
                category = "truncated_response"
                details = ("finish_reason=length",)
                parse_failures += 1
                attempts.append(
                    ExtractionAttempt(
                        number=len(attempts) + 1,
                        category=category,
                        transport_format=transport_format,
                        finish_reason=result.finish_reason,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        details=details,
                    )
                )
                if attempt < self._max_parse_retries:
                    request = request.model_copy(
                        update={
                            "messages": build_extraction_messages(
                                chunk,
                                correction=_corrective_instruction(category, details),
                            )
                        }
                    )
                    continue
                return ChunkExtractionResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    status="schema_rejected",
                    entities=(),
                    relations=(),
                    stats=ExtractionStats(
                        attempts=len(attempts),
                        parse_failures=parse_failures,
                        schema_failures=schema_failures,
                        grounding_rejections=0,
                        duplicate_entities=0,
                        duplicate_relations=0,
                    ),
                    llm_results=tuple(llm_results),
                    attempts=tuple(attempts),
                    error="structured output was truncated",
                )
            try:
                output = parse_extraction_output(result.text)
            except ExtractionParseError:
                parse_failures += 1
                category = (
                    "markdown_code_fence"
                    if transport_format == "json_code_fence"
                    else "extra_prose"
                    if transport_format == "extra_prose"
                    else "empty_response"
                    if not result.text.strip()
                    else "invalid_json"
                )
                details = (category,)
                attempts.append(
                    ExtractionAttempt(
                        number=len(attempts) + 1,
                        category=category,
                        transport_format=transport_format,
                        finish_reason=result.finish_reason,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        details=details,
                    )
                )
                if attempt < self._max_parse_retries:
                    request = request.model_copy(
                        update={
                            "messages": build_extraction_messages(
                                chunk,
                                correction=_corrective_instruction(category, details),
                            )
                        }
                    )
                    continue
                return ChunkExtractionResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    status="schema_rejected",
                    entities=(),
                    relations=(),
                    stats=ExtractionStats(
                        attempts=len(llm_results),
                        parse_failures=parse_failures,
                        schema_failures=schema_failures,
                        grounding_rejections=0,
                        duplicate_entities=0,
                        duplicate_relations=0,
                    ),
                    llm_results=tuple(llm_results),
                    attempts=tuple(attempts),
                    error="structured output was not valid JSON",
                )
            except ExtractionSchemaError as error:
                schema_failures += 1
                category = (
                    error.issue_categories[0]
                    if len(error.issue_categories) == 1
                    else "schema_validation"
                )
                details = tuple(dict.fromkeys((*error.issue_categories, *error.issue_details)))
                attempts.append(
                    ExtractionAttempt(
                        number=len(attempts) + 1,
                        category=category,
                        transport_format=transport_format,
                        finish_reason=result.finish_reason,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        details=details,
                    )
                )
                if attempt < self._max_parse_retries:
                    request = request.model_copy(
                        update={
                            "messages": build_extraction_messages(
                                chunk,
                                correction=_corrective_instruction(category, details),
                            )
                        }
                    )
                    continue
                return ChunkExtractionResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    status="schema_rejected",
                    entities=(),
                    relations=(),
                    stats=ExtractionStats(
                        attempts=len(llm_results),
                        parse_failures=parse_failures,
                        schema_failures=schema_failures,
                        grounding_rejections=0,
                        duplicate_entities=0,
                        duplicate_relations=0,
                    ),
                    llm_results=tuple(llm_results),
                    attempts=tuple(attempts),
                    error="structured output failed the extraction schema",
                )

            (
                entities,
                relations,
                grounding_rejections,
                duplicate_entities,
                duplicate_relations,
                grounding_rejection_reasons,
                grounded_relation_evidence,
            ) = _materialize_output(output, chunk, knowledge_base_id, result)
            rejection_details: list[str] = []
            if _has_unknown_relation_entity(output):
                rejection_details.append("relation_unknown_entity")
            if grounding_rejections:
                rejection_details.append("grounding_rejection")
            status: ExtractionStatus
            if not output.entities and not output.relations:
                status = "empty"
                category = "empty_valid_extraction"
            elif not entities and not relations:
                status = "rejected"
                category = (
                    "relation_unknown_entity"
                    if "relation_unknown_entity" in rejection_details
                    else "grounding_rejection"
                )
            else:
                status = "accepted"
                category = "valid_extraction"
            attempts.append(
                ExtractionAttempt(
                    number=len(attempts) + 1,
                    category=category,
                    transport_format=transport_format,
                    finish_reason=result.finish_reason,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    details=tuple(rejection_details),
                )
            )
            return ChunkExtractionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                status=status,
                entities=entities,
                relations=relations,
                stats=ExtractionStats(
                    attempts=len(llm_results),
                    parse_failures=parse_failures,
                    schema_failures=schema_failures,
                    grounding_rejections=grounding_rejections,
                    duplicate_entities=duplicate_entities,
                    duplicate_relations=duplicate_relations,
                    grounding_rejection_reasons=grounding_rejection_reasons,
                ),
                grounded_relation_evidence=grounded_relation_evidence,
                llm_results=tuple(llm_results),
                attempts=tuple(attempts),
            )
        raise AssertionError("extraction retry loop must return")

    async def extract_and_persist(
        self,
        chunk: Chunk,
        *,
        knowledge_base_id: UUID,
        store: Neo4jGraphStore,
    ) -> ChunkExtractionResult:
        """Persist one validated result through a single Neo4j transaction."""

        result = await self.extract_chunk(chunk, knowledge_base_id=knowledge_base_id)
        if result.status != "accepted":
            return result
        try:
            await asyncio.to_thread(
                store.upsert_extraction,
                result.entities,
                result.relations,
            )
        except GraphStoreError as error:
            raise ExtractionPersistenceError(result=result) from error
        return result


__all__ = [
    "ChunkExtractionResult",
    "ExtractionAttempt",
    "ExtractionAttemptCategory",
    "ExtractionError",
    "ExtractionGateway",
    "ExtractionParseError",
    "ExtractionPersistenceError",
    "ExtractionSchemaError",
    "ExtractionService",
    "ExtractionStats",
    "ExtractionStatus",
    "GroundedRelationEvidence",
    "parse_extraction_output",
]
