from __future__ import annotations

import asyncio
import json
import math
from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphEntity,
    GraphProvenance,
    entity_id_for,
)
from knowledge_scope.graph.neo4j import _decision_fingerprint
from knowledge_scope.linking.models import (
    CanonicalEntity,
    EntityCanonicalLink,
    LinkingPayloadError,
    canonical_link_id_for,
    linking_identity_json,
    new_canonical_entity_id,
    parse_link_adjudication_output,
)
from knowledge_scope.linking.service import (
    LocalEntityContext,
    build_canonical_plan,
    deterministic_decision,
    generate_candidates,
    generate_candidates_with_stats,
    link_entities,
)
from knowledge_scope.shared.config import Settings

KB = UUID("11111111-1111-4111-8111-111111111111")
OTHER_KB = UUID("22222222-2222-4222-8222-222222222222")
DOC_A = UUID("33333333-3333-4333-8333-333333333333")
DOC_B = UUID("44444444-4444-4444-8444-444444444444")


def _entity(
    name: str,
    *,
    document_id: UUID = DOC_A,
    knowledge_base_id: UUID = KB,
    entity_type: str = "概念",
    aliases: list[str] | None = None,
) -> GraphEntity:
    provenance = GraphProvenance(
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=f"chunk-{document_id}",
        page_start=1,
        page_end=1,
        source_block_ids=[f"block-{document_id}"],
        section_path=["测试"],
        extraction_provenance=ExtractionProvenance(method="test", version="1"),
    )
    return GraphEntity(
        entity_id=entity_id_for(
            name,
            entity_type,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        ),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        canonical_name=name,
        entity_type=entity_type,
        aliases=aliases or [],
        provenance=[provenance],
    )


def test_linking_identity_is_versioned_structured_and_alias_independent() -> None:
    first = linking_identity_json("test", {"b": "二", "a": "一\x00"})
    second = linking_identity_json("test", {"a": "一\x00", "b": "二"})
    assert first == second
    assert '"identity_schema_version":"v2"' in first

    delimiter_like_first = linking_identity_json("test", {"left": "a\x00b", "right": "c"})
    delimiter_like_second = linking_identity_json("test", {"left": "a", "right": "b\x00c"})
    assert delimiter_like_first != delimiter_like_second
    assert linking_identity_json("test", {"value": "é"}) != linking_identity_json(
        "test", {"value": "e\u0301"}
    )

    canonical_a = CanonicalEntity(
        canonical_entity_id=new_canonical_entity_id(),
        knowledge_base_id=KB,
        canonical_name="量子",
        entity_type="概念",
    )
    canonical_b = CanonicalEntity(
        canonical_entity_id=new_canonical_entity_id(),
        knowledge_base_id=OTHER_KB,
        canonical_name="量子",
        entity_type="概念",
    )
    assert canonical_a.canonical_entity_id != canonical_b.canonical_entity_id


def test_candidate_generation_is_blocked_and_keeps_incompatible_types_visible() -> None:
    same_name_a = _entity("苹果", document_id=DOC_A, entity_type="物质")
    same_name_b = _entity("苹果", document_id=DOC_B, entity_type="物质")
    incompatible = _entity("苹果", document_id=DOC_B, entity_type="组织")
    unrelated = _entity("河流", document_id=DOC_B)

    candidates = generate_candidates([same_name_a, same_name_b, incompatible, unrelated])
    pairs = {(candidate.local_entity_a_id, candidate.local_entity_b_id) for candidate in candidates}
    assert len(candidates) == 3
    assert all(unrelated.entity_id not in pair for pair in pairs)
    incompatible_candidate = next(
        candidate
        for candidate in candidates
        if incompatible.entity_id in {candidate.local_entity_a_id, candidate.local_entity_b_id}
    )
    assert incompatible_candidate.signals.compatible_entity_type is False
    assert deterministic_decision(incompatible_candidate).decision == "NO_LINK"


def test_lexical_blocking_adds_only_plausible_non_exact_candidates() -> None:
    first = _entity("社会主义制度", document_id=DOC_A)
    second = _entity("社会主义", document_id=DOC_B)
    unrelated = _entity("地理坐标", document_id=DOC_B)

    candidates = generate_candidates([first, second, unrelated])
    assert len(candidates) == 1
    assert deterministic_decision(candidates[0]).decision == "UNCERTAIN"
    assert candidates[0].signals.mention_overlap == []


def test_same_name_homonyms_are_not_automatically_linked() -> None:
    first = _entity("银行", document_id=DOC_A)
    second = _entity("银行", document_id=DOC_B)
    run = asyncio.run(link_entities([first, second]))
    assert run.stats.uncertain_count == 1
    assert run.stats.link_count == 0
    assert not run.canonical_entities
    assert not run.mappings


def test_explicit_alias_signal_stays_uncertain_without_adjudication() -> None:
    first = _entity("长江", document_id=DOC_A, aliases=["长江河", "Yangtze"])
    second = _entity("长江", document_id=DOC_B, aliases=["长江河", "Yangtze"])
    run = asyncio.run(link_entities([first, second]))

    assert run.stats.link_count == 0
    assert run.stats.uncertain_count == 1
    assert not run.canonical_entities
    assert not run.mappings


class _FakeGateway:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        from knowledge_scope.llm.schemas import LLMResult

        return LLMResult(
            text=self.response,
            provider="fake",
            model="fake-model",
            input_tokens=12,
            output_tokens=4,
            latency_ms=2.5,
        )


def test_llm_adjudication_cannot_supply_internal_ids() -> None:
    first = _entity("太阳", document_id=DOC_A)
    second = _entity("太阳", document_id=DOC_B)
    gateway = _FakeGateway(
        json.dumps(
            {
                "decision": "LINK",
                "confidence": 0.91,
                "reason": "来源片段支持同一概念",
                "canonical_entity_id": "attacker-provided-id",
            },
            ensure_ascii=False,
        )
    )
    run = asyncio.run(
        link_entities(
            [first, second],
            gateway=gateway,
            settings=Settings(_env_file=None),
            contexts=[
                LocalEntityContext(first, "太阳是恒星"),
                LocalEntityContext(second, "太阳光"),
            ],
        )
    )
    assert run.stats.llm_failures == 1
    assert run.stats.uncertain_count == 1
    assert not run.mappings
    assert "entity_id" not in gateway.requests[0].messages[1].content


def test_llm_link_decision_creates_only_application_owned_mapping() -> None:
    first = _entity("星系", document_id=DOC_A)
    second = _entity("星系", document_id=DOC_B)
    gateway = _FakeGateway('{"decision":"LINK","confidence":0.88,"reason":"两段来源使用同一术语"}')
    run = asyncio.run(
        link_entities(
            [first, second],
            gateway=gateway,
            settings=Settings(_env_file=None),
            contexts=[
                LocalEntityContext(first, "星系由恒星组成"),
                LocalEntityContext(second, "星系是天体系统"),
            ],
        )
    )
    assert run.stats.llm_adjudications == 1
    assert run.stats.llm_failures == 0
    assert run.stats.link_count == 1
    assert len(run.mappings) == 2
    assert run.mappings[0].canonical_entity_id == run.mappings[1].canonical_entity_id
    assert all(
        mapping.canonical_entity_id not in gateway.requests[0].messages[1].content
        for mapping in run.mappings
    )


def test_llm_contexts_must_match_each_local_entity_once_and_have_evidence() -> None:
    first = _entity("恒星", document_id=DOC_A)
    second = _entity("恒星", document_id=DOC_B)
    gateway = _FakeGateway('{"decision":"UNCERTAIN","reason":"证据不足"}')
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="source excerpts"):
        asyncio.run(
            link_entities(
                [first, second],
                gateway=gateway,
                settings=settings,
                contexts=[LocalEntityContext(first, ""), LocalEntityContext(second, "恒星")],
            )
        )
    with pytest.raises(ValueError, match="exactly once"):
        asyncio.run(
            link_entities(
                [first, second],
                gateway=gateway,
                settings=settings,
                contexts=[
                    LocalEntityContext(first, "恒星是发光的天体"),
                    LocalEntityContext(first, "重复上下文"),
                    LocalEntityContext(second, "恒星由等离子体组成"),
                ],
            )
        )


def test_cross_knowledge_base_entities_are_rejected_before_candidate_generation() -> None:
    with pytest.raises(ValueError, match="cross-KB"):
        generate_candidates(
            [_entity("概念", knowledge_base_id=KB), _entity("概念", knowledge_base_id=OTHER_KB)]
        )


def test_adjudication_parser_rejects_malformed_or_extra_fields() -> None:
    parsed = parse_link_adjudication_output(
        '{"decision":"UNCERTAIN","confidence":0.4,"reason":"证据不足"}'
    )
    assert parsed.decision == "UNCERTAIN"
    with pytest.raises(LinkingPayloadError):
        parse_link_adjudication_output(
            '{"decision":"LINK","confidence":1,"reason":"x","local_entity_id":"bad"}'
        )
    with pytest.raises(LinkingPayloadError):
        parse_link_adjudication_output("not-json")
    with pytest.raises(LinkingPayloadError, match="duplicate key"):
        parse_link_adjudication_output('{"decision":"UNCERTAIN","decision":"LINK","reason":"x"}')


@pytest.mark.parametrize(
    "confidence",
    [True, False, "0.9", "1", None, float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_adjudication_parser_rejects_non_strict_confidence(confidence: object) -> None:
    if isinstance(confidence, float) and not math.isfinite(confidence):
        payload = json.dumps(
            {"decision": "LINK", "confidence": confidence, "reason": "x"},
            allow_nan=True,
        )
    else:
        payload = json.dumps(
            {"decision": "LINK", "confidence": confidence, "reason": "x"},
            ensure_ascii=False,
        )
    with pytest.raises(LinkingPayloadError):
        parse_link_adjudication_output(payload)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, 0.5])
def test_adjudication_parser_accepts_json_numbers_in_range(confidence: int | float) -> None:
    parsed = parse_link_adjudication_output(
        json.dumps({"decision": "LINK", "confidence": confidence, "reason": "x"})
    )
    assert parsed.confidence == float(confidence)


def test_adjudication_parser_allows_omitted_confidence_for_uncertain_result() -> None:
    parsed = parse_link_adjudication_output('{"decision":"UNCERTAIN","reason":"证据不足"}')
    assert parsed.confidence is None


def test_canonical_models_keep_opaque_identity_stable_and_require_link_decision() -> None:
    entity = _entity("地球")
    canonical_id = new_canonical_entity_id()
    canonical = CanonicalEntity(
        canonical_entity_id=canonical_id,
        knowledge_base_id=KB,
        canonical_name="地球",
        entity_type="概念",
        aliases=["地球", "蓝色星球", "蓝色星球"],
    )
    assert canonical.aliases == ["蓝色星球"]
    changed = canonical.model_copy(update={"canonical_name": "月球", "aliases": ["月亮"]})
    assert changed.canonical_entity_id == canonical_id

    link_id = canonical_link_id_for(KB, entity.entity_id, canonical.canonical_entity_id)
    with pytest.raises(ValidationError):
        EntityCanonicalLink(
            link_id=link_id,
            knowledge_base_id=KB,
            local_entity_id=entity.entity_id,
            canonical_entity_id=canonical.canonical_entity_id,
            decision_id="not-a-link-decision-id",
            method="manual_review",
            reason="没有决策",
            candidate_signals={
                "normalized_name_match": False,
                "compatible_entity_type": True,
                "lexical_similarity": 0,
                "same_document": False,
            },
        )


def test_decision_timestamp_is_audit_metadata_not_identity() -> None:
    first = _entity("时间审计")
    second = _entity("时间审计", document_id=DOC_B)
    candidate = generate_candidates([first, second])[0]
    decision = deterministic_decision(candidate)
    rebuilt = decision.model_copy(update={"created_at": decision.created_at + timedelta(seconds=1)})

    assert decision.created_at.tzinfo is not None
    assert rebuilt.decision_id == decision.decision_id
    assert _decision_fingerprint(rebuilt) == _decision_fingerprint(decision)


def test_plan_is_deterministic_for_transitive_links() -> None:
    first = _entity("水", document_id=DOC_A, aliases=["H2O", "氧化氢"])
    second = _entity("水", document_id=DOC_B, aliases=["H2O", "氧化氢"])
    third = _entity(
        "水", document_id=UUID("55555555-5555-4555-8555-555555555555"), aliases=["H2O", "氧化氢"]
    )
    candidates = generate_candidates([first, second, third])
    decisions = [
        deterministic_decision(candidate).model_copy(
            update={
                "decision": "LINK",
                "method": "manual_review",
                "reason": "人工确认同一实体",
            }
        )
        for candidate in candidates
    ]
    canonical, mappings = build_canonical_plan([first, second, third], candidates, decisions)
    assert len(canonical) == 1
    assert len(mappings) == 3


def test_candidate_generation_bounds_high_frequency_blocks_before_materialization() -> None:
    entities = [_entity("通用", document_id=UUID(int=100 + index)) for index in range(20)]

    result = generate_candidates_with_stats(
        entities,
        max_candidates=3,
        max_block_size=4,
    )

    assert len(result.candidates) <= 3
    assert result.skipped_candidate_blocks >= 1
    assert result.skipped_candidate_pairs >= 1


def test_prompt_metadata_is_bounded() -> None:
    first = _entity("实体一", aliases=[f"别名-{index}-" + "x" * 120 for index in range(20)])
    second = _entity("实体一", document_id=DOC_B)
    candidate = generate_candidates([first, second])[0]
    from knowledge_scope.linking.prompt import build_link_adjudication_messages

    messages = build_link_adjudication_messages(
        candidate,
        LocalEntityContext(first, "证据" * 500),
        LocalEntityContext(second, "另一证据"),
    )
    assert len(messages[1].content) < 2_000
