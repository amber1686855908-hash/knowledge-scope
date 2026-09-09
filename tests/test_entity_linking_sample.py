from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from knowledge_scope.evaluation.entity_linking_sample import run_linking_review_sample
from knowledge_scope.graph.models import GraphEntity, GraphProvenance, entity_id_for
from knowledge_scope.shared.config import Settings

KB = UUID("11111111-1111-4111-8111-111111111111")
DOC_A = UUID("33333333-3333-4333-8333-333333333333")
DOC_B = UUID("44444444-4444-4444-8444-444444444444")


def _entity(name: str, document_id: UUID, *, alias: str) -> GraphEntity:
    return GraphEntity(
        entity_id=entity_id_for(
            name,
            "概念",
            knowledge_base_id=KB,
            document_id=document_id,
        ),
        knowledge_base_id=KB,
        document_id=document_id,
        canonical_name=name,
        entity_type="概念",
        aliases=[alias],
        provenance=[
            GraphProvenance(
                document_id=document_id,
                knowledge_base_id=KB,
                chunk_id=f"chunk-{document_id}",
                page_start=1,
                page_end=1,
                source_block_ids=[f"block-{document_id}"],
                section_path=["测试"],
            )
        ],
    )


def test_review_sample_is_bounded_and_merges_repeated_local_context(tmp_path: Path) -> None:
    first = _entity("同一对象", DOC_A, alias="别名一")
    second = _entity("同一对象", DOC_B, alias="别名二")
    source = tmp_path / "sample.jsonl"
    records = [
        {
            "status": "accepted",
            "subject": "历史",
            "text_excerpt": "第一段来源",
            "entities": [first.model_dump(mode="json")],
        },
        {
            "status": "accepted",
            "subject": "历史",
            "text_excerpt": "第二段来源",
            "entities": [second.model_dump(mode="json")],
        },
        {"status": "empty", "subject": "历史", "text_excerpt": "不应参与", "entities": []},
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = asyncio.run(
        run_linking_review_sample(
            [source],
            settings=Settings(_env_file=None),
            output_dir=tmp_path / "output",
            max_candidates=10,
        )
    )

    assert summary["sample_entity_count"] == 2
    assert summary["candidate_count"] == 1
    review = (tmp_path / "output" / "review.jsonl").read_text(encoding="utf-8")
    assert "第一段来源" in review
    assert "第二段来源" in review
    assert "decision_id" in review
    assert "link_pair_id" in review
    assert str(tmp_path) not in review
