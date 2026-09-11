"""Conservative, explicit local-entity linking for one knowledge base."""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from math import comb
from typing import Protocol
from uuid import UUID, uuid4

from knowledge_scope.graph.models import GraphEntity
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.schemas import LLMRequest, LLMResponseFormat, LLMResult
from knowledge_scope.llm.usage import estimate_cost
from knowledge_scope.shared.config import Settings

from .models import (
    DEFAULT_DECISION_RUN_ID,
    CandidateSignals,
    CanonicalEntity,
    EntityCanonicalLink,
    EntityLinkCandidate,
    EntityLinkDecision,
    LinkDecision,
    LinkingPayloadError,
    LinkMethod,
    canonical_link_id_for,
    link_decision_id_for,
    link_pair_id_for,
    new_canonical_entity_id,
    normalize_linking_label,
    parse_link_adjudication_output,
)
from .prompt import LINKING_PROMPT_VERSION, build_link_adjudication_messages
from .service_types import LocalEntityContext

DEFAULT_MAX_CANDIDATES = 200
DEFAULT_MAX_BLOCK_SIZE = 64
DEFAULT_PREFIX_WINDOW = 16


class LinkingValidationError(ValueError):
    """Raised when local entities do not form one valid KB-scoped input set."""


class LinkingGateway(Protocol):
    """The minimal gateway surface required for ambiguous candidates."""

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Adjudicate one ambiguous candidate."""


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    """Candidates plus explicit bounds telemetry for one generation pass."""

    candidates: tuple[EntityLinkCandidate, ...]
    skipped_candidate_blocks: int
    skipped_candidate_pairs: int
    candidate_budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class LinkingStats:
    """Safe counters for one linking run."""

    candidate_count: int
    skipped_candidate_blocks: int
    skipped_candidate_pairs: int
    candidate_budget_exhausted: bool
    deterministic_decisions: int
    manual_review_fallbacks: int
    provider_failure_fallbacks: int
    llm_adjudications: int
    llm_failures: int
    link_count: int
    no_link_count: int
    uncertain_count: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    estimated_cost: Decimal | None
    provider_attempts: int = 0
    provider_retry_calls: int = 0


@dataclass(frozen=True, slots=True)
class LinkingRun:
    """Decisions, canonical cluster plan and safe observations for a run."""

    run_id: UUID
    knowledge_base_id: UUID
    candidates: tuple[EntityLinkCandidate, ...]
    decisions: tuple[EntityLinkDecision, ...]
    canonical_entities: tuple[CanonicalEntity, ...]
    mappings: tuple[EntityCanonicalLink, ...]
    stats: LinkingStats


def _entity_terms(entity: GraphEntity) -> set[str]:
    return {
        normalize_linking_label(value)
        for value in (entity.canonical_name, *entity.aliases)
        if normalize_linking_label(value)
    }


def _alias_terms(entity: GraphEntity) -> set[str]:
    return {
        normalize_linking_label(value) for value in entity.aliases if normalize_linking_label(value)
    }


def _compatible_type(first: GraphEntity, second: GraphEntity) -> bool:
    return normalize_linking_label(first.entity_type) == normalize_linking_label(second.entity_type)


def _lexical_similarity(first: GraphEntity, second: GraphEntity) -> float:
    score = SequenceMatcher(
        None,
        normalize_linking_label(first.canonical_name),
        normalize_linking_label(second.canonical_name),
    ).ratio()
    return round(score, 6)


def _lexical_block_key(entity: GraphEntity) -> str | None:
    name = normalize_linking_label(entity.canonical_name)
    return name[:2] if len(name) >= 2 else None


def _validated_entities(
    entities: Sequence[GraphEntity],
    *,
    knowledge_base_id: UUID | None = None,
) -> tuple[GraphEntity, ...]:
    by_id: dict[str, GraphEntity] = {}
    for entity in entities:
        if knowledge_base_id is not None and entity.knowledge_base_id != knowledge_base_id:
            raise LinkingValidationError("all local entities must belong to the requested KB")
        previous = by_id.get(entity.entity_id)
        if previous is not None and previous != entity:
            raise LinkingValidationError("duplicate local entity ID has conflicting payloads")
        by_id[entity.entity_id] = entity
    if not by_id:
        raise LinkingValidationError("at least one local entity is required")
    observed = {entity.knowledge_base_id for entity in by_id.values()}
    if len(observed) != 1:
        raise LinkingValidationError("cross-KB linking candidates are not allowed")
    return tuple(by_id[key] for key in sorted(by_id))


def _candidate_for(first: GraphEntity, second: GraphEntity) -> EntityLinkCandidate:
    if first.knowledge_base_id != second.knowledge_base_id:
        raise LinkingValidationError("cross-KB linking candidates are not allowed")
    first, second = sorted((first, second), key=lambda entity: entity.entity_id)
    first_terms = _entity_terms(first)
    second_terms = _entity_terms(second)
    first_aliases = _alias_terms(first)
    second_aliases = _alias_terms(second)
    return EntityLinkCandidate(
        link_pair_id=link_pair_id_for(
            first.knowledge_base_id,
            first.entity_id,
            second.entity_id,
        ),
        knowledge_base_id=first.knowledge_base_id,
        local_entity_a_id=first.entity_id,
        local_entity_b_id=second.entity_id,
        signals=CandidateSignals(
            normalized_name_match=normalize_linking_label(first.canonical_name)
            == normalize_linking_label(second.canonical_name),
            mention_overlap=sorted(first_terms & second_terms),
            alias_overlap=sorted(first_aliases & second_aliases),
            compatible_entity_type=_compatible_type(first, second),
            lexical_similarity=_lexical_similarity(first, second),
            same_document=first.document_id == second.document_id,
        ),
    )


def generate_candidates_with_stats(
    entities: Sequence[GraphEntity],
    *,
    knowledge_base_id: UUID | None = None,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    prefix_window: int = DEFAULT_PREFIX_WINDOW,
) -> CandidateGenerationResult:
    """Generate bounded name/alias-blocked candidates without all-pairs scans.

    Exact blocks larger than ``max_block_size`` are skipped as generic/high
    frequency blocks.  Prefix blocks use the same bound and a deterministic
    comparison window.  A global pair budget is checked before a pair is
    materialized, so a large block cannot first create an unbounded set.
    """

    if max_candidates is not None and max_candidates < 1:
        raise LinkingValidationError("max_candidates must be at least one")
    if max_block_size < 2:
        raise LinkingValidationError("max_block_size must be at least two")
    if prefix_window < 1:
        raise LinkingValidationError("prefix_window must be at least one")

    validated = _validated_entities(entities, knowledge_base_id=knowledge_base_id)
    by_id = {entity.entity_id: entity for entity in validated}
    blocked: dict[tuple[str, str], set[str]] = {}
    for entity in validated:
        for term in _entity_terms(entity):
            blocked.setdefault(("exact", term), set()).add(entity.entity_id)
        lexical_key = _lexical_block_key(entity)
        if lexical_key is not None:
            blocked.setdefault(("prefix", lexical_key), set()).add(entity.entity_id)

    pairs: set[tuple[str, str]] = set()
    skipped_blocks = 0
    skipped_pairs = 0
    budget_exhausted = False
    for (block_type, _block_value), ids in sorted(blocked.items()):
        ordered = sorted(ids)
        if len(ordered) > max_block_size:
            skipped_blocks += 1
            skipped_pairs += comb(len(ordered), 2)
            continue
        for index, first_id in enumerate(ordered):
            if block_type == "prefix":
                comparison_ids = ordered[index + 1 : index + 1 + prefix_window]
            else:
                comparison_ids = ordered[index + 1 :]
            for second_id in comparison_ids:
                first_entity = by_id[first_id]
                second_entity = by_id[second_id]
                if (
                    block_type == "prefix"
                    and _lexical_similarity(first_entity, second_entity) < 0.72
                ):
                    continue
                pair = (first_id, second_id)
                if pair in pairs:
                    continue
                if max_candidates is not None and len(pairs) >= max_candidates:
                    budget_exhausted = True
                    skipped_pairs += 1
                    break
                pairs.add(pair)
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    candidates = tuple(
        _candidate_for(by_id[first], by_id[second]) for first, second in sorted(pairs)
    )
    return CandidateGenerationResult(
        candidates=candidates,
        skipped_candidate_blocks=skipped_blocks,
        skipped_candidate_pairs=skipped_pairs,
        candidate_budget_exhausted=budget_exhausted,
    )


def generate_candidates(
    entities: Sequence[GraphEntity],
    *,
    knowledge_base_id: UUID | None = None,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    prefix_window: int = DEFAULT_PREFIX_WINDOW,
) -> tuple[EntityLinkCandidate, ...]:
    """Generate bounded candidates and return only the candidate records."""

    return generate_candidates_with_stats(
        entities,
        knowledge_base_id=knowledge_base_id,
        max_candidates=max_candidates,
        max_block_size=max_block_size,
        prefix_window=prefix_window,
    ).candidates


def _decision(
    candidate: EntityLinkCandidate,
    *,
    run_id: UUID,
    decision: LinkDecision,
    method: LinkMethod,
    confidence: float | None,
    reason: str,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> EntityLinkDecision:
    return EntityLinkDecision(
        decision_id=link_decision_id_for(
            candidate.knowledge_base_id,
            candidate.local_entity_a_id,
            candidate.local_entity_b_id,
            run_id,
        ),
        link_pair_id=candidate.link_pair_id,
        run_id=run_id,
        knowledge_base_id=candidate.knowledge_base_id,
        local_entity_a_id=candidate.local_entity_a_id,
        local_entity_b_id=candidate.local_entity_b_id,
        decision=decision,
        method=method,
        confidence=confidence,
        reason=reason,
        candidate_signals=candidate.signals,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
    )


def deterministic_decision(
    candidate: EntityLinkCandidate,
    *,
    run_id: UUID = DEFAULT_DECISION_RUN_ID,
) -> EntityLinkDecision:
    """Apply only safe deterministic rejection rules before any LLM call.

    Compatible name, alias, and lexical signals are never sufficient for a
    destructive automatic LINK.  Ambiguous candidates remain UNCERTAIN.
    """

    signals = candidate.signals
    if not signals.compatible_entity_type:
        return _decision(
            candidate,
            run_id=run_id,
            decision="NO_LINK",
            method="incompatible_type",
            confidence=1.0,
            reason="实体类型不兼容，禁止自动链接",
        )
    if not signals.mention_overlap and signals.lexical_similarity < 0.72:
        return _decision(
            candidate,
            run_id=run_id,
            decision="NO_LINK",
            method="no_shared_signal",
            confidence=1.0,
            reason="没有共享的规范化名称或别名信号",
        )
    return _decision(
        candidate,
        run_id=run_id,
        decision="UNCERTAIN",
        method="similarity_rule",
        confidence=signals.lexical_similarity if signals.lexical_similarity >= 0.5 else None,
        reason="存在名称或别名信号，但不足以安全自动链接",
    )


def _fallback_uncertain(
    candidate: EntityLinkCandidate,
    *,
    run_id: UUID,
    method: LinkMethod,
    reason: str,
    provider: str | None = None,
    model: str | None = None,
) -> EntityLinkDecision:
    return _decision(
        candidate,
        run_id=run_id,
        decision="UNCERTAIN",
        method=method,
        confidence=None,
        reason=reason,
        provider=provider,
        model=model,
        prompt_version=LINKING_PROMPT_VERSION if provider else None,
    )


async def _adjudicate_candidate(
    candidate: EntityLinkCandidate,
    *,
    run_id: UUID,
    context_by_id: dict[str, LocalEntityContext],
    gateway: LinkingGateway,
    settings: Settings,
) -> tuple[EntityLinkDecision, LLMResult | None, bool, bool, int]:
    messages = build_link_adjudication_messages(
        candidate,
        context_by_id[candidate.local_entity_a_id],
        context_by_id[candidate.local_entity_b_id],
    )
    request = LLMRequest(
        messages=messages,
        task_type="entity_linking",
        temperature=0.0,
        max_tokens=256,
        model=settings.llm_model,
        response_format=LLMResponseFormat(type="json_object"),
        reasoning="disabled",
    )
    try:
        result = await gateway.complete(request)
    except asyncio.CancelledError:
        raise
    except LLMError as error:
        if error.category == "usage_persistence":
            raise
        return (
            _fallback_uncertain(
                candidate,
                run_id=run_id,
                method="llm_unavailable",
                reason="LLM 审核不可用，保留为待人工审核",
                provider=settings.llm_provider,
                model=settings.llm_model,
            ),
            None,
            True,
            True,
            max(0, error.provider_attempts),
        )
    try:
        output = parse_link_adjudication_output(result.text)
    except LinkingPayloadError:
        return (
            _fallback_uncertain(
                candidate,
                run_id=run_id,
                method="llm_invalid_output",
                reason="LLM 返回不符合链接决策 schema，保留为待人工审核",
                provider=result.provider,
                model=result.model,
            ),
            result,
            True,
            False,
            result.provider_attempts,
        )

    decision: LinkDecision = output.decision
    method: LinkMethod = "llm_adjudication"
    reason = output.reason
    if not candidate.signals.compatible_entity_type and decision == "LINK":
        decision = "NO_LINK"
        method = "incompatible_type"
        reason = "应用侧类型保护覆盖了不安全的 LLM LINK 决策"
    return (
        _decision(
            candidate,
            run_id=run_id,
            decision=decision,
            method=method,
            confidence=output.confidence,
            reason=reason,
            provider=result.provider,
            model=result.model,
            prompt_version=LINKING_PROMPT_VERSION,
        ),
        result,
        False,
        False,
        result.provider_attempts,
    )


def _union_find_components(
    entities: dict[str, GraphEntity],
    decisions: Sequence[EntityLinkDecision],
    candidates: dict[str, EntityLinkCandidate],
) -> tuple[tuple[str, ...], ...]:
    parent = {entity_id: entity_id for entity_id in entities}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for decision in sorted(decisions, key=lambda item: item.decision_id):
        if decision.decision != "LINK":
            continue
        candidate = candidates.get(decision.link_pair_id)
        if candidate is None or not candidate.signals.compatible_entity_type:
            continue
        union(decision.local_entity_a_id, decision.local_entity_b_id)

    grouped: dict[str, list[str]] = {}
    for entity_id in sorted(entities):
        grouped.setdefault(find(entity_id), []).append(entity_id)
    return tuple(sorted(tuple(values) for values in grouped.values() if len(values) >= 2))


def build_canonical_plan(
    entities: Sequence[GraphEntity],
    candidates: Sequence[EntityLinkCandidate],
    decisions: Sequence[EntityLinkDecision],
    *,
    canonical_ids_by_members: Mapping[tuple[str, ...], str] | None = None,
) -> tuple[tuple[CanonicalEntity, ...], tuple[EntityCanonicalLink, ...]]:
    """Turn accepted decisions into a canonical plan.

    New canonical IDs are generated once for the plan.  A caller rebuilding an
    existing persisted cluster must explicitly supply its existing opaque ID
    keyed by the member tuple; content or processing order never derives it.
    """

    validated = _validated_entities(entities)
    by_id = {entity.entity_id: entity for entity in validated}
    candidate_by_id = {candidate.link_pair_id: candidate for candidate in candidates}
    decision_by_id = {decision.decision_id: decision for decision in decisions}
    if len(candidate_by_id) != len(candidates) or len(decision_by_id) != len(decisions):
        raise LinkingValidationError("candidates and decisions must have unique identities")
    kb = validated[0].knowledge_base_id
    for candidate in candidates:
        if candidate.knowledge_base_id != kb:
            raise LinkingValidationError("candidates must belong to the local entity KB")
        if candidate.local_entity_a_id not in by_id or candidate.local_entity_b_id not in by_id:
            raise LinkingValidationError("candidate endpoints must be in the local entity set")
    if {decision.link_pair_id for decision in decisions} != set(candidate_by_id):
        raise LinkingValidationError("decisions must cover the generated candidate set")
    for decision in decisions:
        if decision.knowledge_base_id != kb:
            raise LinkingValidationError("decisions must belong to the local entity KB")
        candidate = candidate_by_id[decision.link_pair_id]
        if (
            decision.local_entity_a_id != candidate.local_entity_a_id
            or decision.local_entity_b_id != candidate.local_entity_b_id
        ):
            raise LinkingValidationError("decision endpoints do not match candidate endpoints")

    canonical_entities: list[CanonicalEntity] = []
    mappings: list[EntityCanonicalLink] = []
    canonical_ids_by_members = canonical_ids_by_members or {}
    for component in _union_find_components(by_id, decisions, candidate_by_id):
        members = [by_id[entity_id] for entity_id in component]
        representative = min(
            members,
            key=lambda entity: (
                normalize_linking_label(entity.canonical_name),
                entity.entity_id,
            ),
        )
        aliases = [
            value for entity in members for value in (entity.canonical_name, *entity.aliases)
        ]
        canonical = CanonicalEntity(
            canonical_entity_id=canonical_ids_by_members.get(component, new_canonical_entity_id()),
            knowledge_base_id=representative.knowledge_base_id,
            canonical_name=representative.canonical_name,
            entity_type=representative.entity_type,
            aliases=aliases,
        )
        canonical_entities.append(canonical)
        accepted_for_member: dict[str, list[EntityLinkDecision]] = {
            entity_id: [] for entity_id in component
        }
        for decision in decisions:
            if decision.decision != "LINK":
                continue
            if decision.local_entity_a_id in accepted_for_member:
                accepted_for_member[decision.local_entity_a_id].append(decision)
            if decision.local_entity_b_id in accepted_for_member:
                accepted_for_member[decision.local_entity_b_id].append(decision)
        for entity_id in component:
            supporting = tuple(
                sorted(
                    (item.decision_id for item in accepted_for_member[entity_id]),
                )
            )
            if not supporting:
                raise LinkingValidationError("a canonical member needs LINK decision support")
            primary = decision_by_id[supporting[0]]
            mappings.append(
                EntityCanonicalLink(
                    link_id=canonical_link_id_for(
                        canonical.knowledge_base_id,
                        entity_id,
                        canonical.canonical_entity_id,
                    ),
                    knowledge_base_id=canonical.knowledge_base_id,
                    local_entity_id=entity_id,
                    canonical_entity_id=canonical.canonical_entity_id,
                    decision_id=primary.decision_id,
                    supporting_decision_ids=list(supporting),
                    method=primary.method,
                    confidence=primary.confidence,
                    reason=primary.reason,
                    candidate_signals=primary.candidate_signals,
                    provider=primary.provider,
                    model=primary.model,
                    prompt_version=primary.prompt_version,
                )
            )
    return (
        tuple(sorted(canonical_entities, key=lambda item: item.canonical_entity_id)),
        tuple(sorted(mappings, key=lambda item: item.link_id)),
    )


def _sum_tokens(results: Sequence[LLMResult], field: str) -> int | None:
    values = [getattr(result, field) for result in results]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


async def link_entities(
    entities: Sequence[GraphEntity],
    *,
    gateway: LinkingGateway | None = None,
    settings: Settings | None = None,
    contexts: Sequence[LocalEntityContext] | None = None,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    prefix_window: int = DEFAULT_PREFIX_WINDOW,
    run_id: UUID | None = None,
) -> LinkingRun:
    """Generate candidates, adjudicate ambiguity, and build a reversible plan."""

    validated = _validated_entities(entities)
    kb = validated[0].knowledge_base_id
    effective_run_id = run_id or uuid4()
    generated = generate_candidates_with_stats(
        validated,
        knowledge_base_id=kb,
        max_candidates=max_candidates,
        max_block_size=max_block_size,
        prefix_window=prefix_window,
    )
    candidates = generated.candidates
    expected_entities = {entity.entity_id: entity for entity in validated}
    context_by_id: dict[str, LocalEntityContext] = {}
    for context in contexts or tuple(LocalEntityContext(entity) for entity in validated):
        entity_id = context.entity.entity_id
        if entity_id in context_by_id:
            raise LinkingValidationError(
                "linking contexts must contain each local entity exactly once"
            )
        if expected_entities.get(entity_id) != context.entity:
            raise LinkingValidationError("linking context entity does not match the local entity")
        context_by_id[entity_id] = context
    if set(context_by_id) != set(expected_entities):
        raise LinkingValidationError("linking contexts must cover every local entity exactly once")
    if gateway is not None and any(
        not context.source_excerpt.strip() for context in context_by_id.values()
    ):
        raise LinkingValidationError("LLM adjudication contexts must include source excerpts")

    llm_results: list[LLMResult] = []
    decisions: list[EntityLinkDecision] = []
    deterministic_count = 0
    manual_review_fallbacks = 0
    provider_failure_fallbacks = 0
    llm_calls = 0
    llm_failures = 0
    provider_attempts = 0
    for candidate in candidates:
        initial = deterministic_decision(candidate, run_id=effective_run_id)
        if initial.decision != "UNCERTAIN":
            decisions.append(initial)
            deterministic_count += 1
            continue
        if gateway is None:
            decisions.append(
                _fallback_uncertain(
                    candidate,
                    run_id=effective_run_id,
                    method="manual_review",
                    reason="候选需要人工审核或显式启用 LLM 仲裁",
                )
            )
            manual_review_fallbacks += 1
            continue
        if settings is None:
            raise LinkingValidationError("settings are required when LLM adjudication is enabled")
        llm_calls += 1
        decision, result, failed, provider_failed, attempts = await _adjudicate_candidate(
            candidate,
            run_id=effective_run_id,
            context_by_id=context_by_id,
            gateway=gateway,
            settings=settings,
        )
        decisions.append(decision)
        if result is not None:
            llm_results.append(result)
        provider_attempts += attempts
        if failed:
            llm_failures += 1
        if provider_failed:
            provider_failure_fallbacks += 1

    decisions_tuple = tuple(sorted(decisions, key=lambda item: item.decision_id))
    canonical_entities, mappings = build_canonical_plan(validated, candidates, decisions_tuple)
    decision_counts = {
        value: sum(item.decision == value for item in decisions_tuple)
        for value in ("LINK", "NO_LINK", "UNCERTAIN")
    }
    input_tokens = _sum_tokens(llm_results, "input_tokens")
    output_tokens = _sum_tokens(llm_results, "output_tokens")
    latency_ms = sum(result.latency_ms for result in llm_results)
    estimated = (
        estimate_cost(input_tokens, output_tokens, settings) if settings is not None else None
    )
    return LinkingRun(
        run_id=effective_run_id,
        knowledge_base_id=kb,
        candidates=candidates,
        decisions=decisions_tuple,
        canonical_entities=canonical_entities,
        mappings=mappings,
        stats=LinkingStats(
            candidate_count=len(candidates),
            skipped_candidate_blocks=generated.skipped_candidate_blocks,
            skipped_candidate_pairs=generated.skipped_candidate_pairs,
            candidate_budget_exhausted=generated.candidate_budget_exhausted,
            deterministic_decisions=deterministic_count,
            manual_review_fallbacks=manual_review_fallbacks,
            provider_failure_fallbacks=provider_failure_fallbacks,
            llm_adjudications=llm_calls,
            llm_failures=llm_failures,
            link_count=decision_counts["LINK"],
            no_link_count=decision_counts["NO_LINK"],
            uncertain_count=decision_counts["UNCERTAIN"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost=estimated,
            provider_attempts=provider_attempts,
            provider_retry_calls=max(0, provider_attempts - llm_calls),
        ),
    )


__all__ = [
    "DEFAULT_MAX_BLOCK_SIZE",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_PREFIX_WINDOW",
    "CandidateGenerationResult",
    "LinkingGateway",
    "LinkingRun",
    "LinkingStats",
    "LinkingValidationError",
    "LocalEntityContext",
    "build_canonical_plan",
    "deterministic_decision",
    "generate_candidates",
    "generate_candidates_with_stats",
    "link_entities",
]
