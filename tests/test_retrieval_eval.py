from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.chunking.models import ChunkingConfig
from knowledge_scope.chunking.service import chunk_document
from knowledge_scope.evaluation.retrieval_eval import (
    CorpusDocument,
    FinalDatasetItem,
    IndexedChunk,
    RetrievalEvalError,
    RetrievalEvalItem,
    RetrievalEvidence,
    _assign_final_splits,
    apply_review_action,
    build_review_pack_item,
    derive_materialized_item,
    evidence_fingerprint,
    generate_candidate_items,
    load_a15_missing_table_exclusions,
    make_eval_item,
    normalize_query,
    validate_items,
)
from knowledge_scope.parsing.models import (
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _document(*, document_id: UUID = DOCUMENT_ID, extra_blocks: bool = False) -> CanonicalDocument:
    blocks = [
        TitleBlock(block_id="p1-b1", reading_order=0, text="细胞结构"),
        TextBlock(
            block_id="p1-b2",
            reading_order=1,
            text="细胞膜具有选择透过性,能够控制物质进出细胞。",
        ),
        FormulaBlock(block_id="p1-b3", reading_order=2, latex="E=mc^2"),
        TableBlock(
            block_id="p1-b4",
            reading_order=3,
            markdown="| 项目 | 数值 |\n| --- | --- |\n| 样本 | 1 |",
        ),
        ImageBlock(block_id="p1-b5", reading_order=4, asset_ref="image-1"),
        TableBlock(block_id="p1-b6", reading_order=5, asset_ref="table-1"),
    ]
    if extra_blocks:
        blocks.extend(
            TextBlock(
                block_id=f"p1-b{index}",
                reading_order=index - 1,
                text=f"补充说明 {index} 与前文的关系。",
            )
            for index in range(7, 22)
        )
    return CanonicalDocument(
        document_id=document_id,
        pages=[
            Page(page_number=1, blocks=blocks),
            Page(
                page_number=2,
                blocks=[
                    TitleBlock(block_id="p2-b1", reading_order=0, text="实验结论"),
                    TextBlock(
                        block_id="p2-b2",
                        reading_order=1,
                        text="实验结果支持上述细胞结构和功能之间的联系。",
                    ),
                ],
            ),
        ],
    )


def _corpus_document(subject: str = "生物", *, extra_blocks: bool = False) -> CorpusDocument:
    return CorpusDocument(
        document=_document(extra_blocks=extra_blocks),
        subject=subject,
        benchmark_item_id="item-test",
        relative_path="subject/test.pdf",
    )


def test_eval_schema_is_strict_and_identity_is_deterministic() -> None:
    document = _document()
    lookup = {document.document_id: document}
    evidence = [
        RetrievalEvidence(
            document_id=document.document_id,
            page_number=1,
            source_block_ids=["p1-b1", "p1-b2"],
        )
    ]
    item = make_eval_item(
        query="细胞膜的作用是什么?",
        subject="生物",
        query_type="explanation",
        evidence=evidence,
        document_lookup=lookup,
    )
    assert (
        item.item_id
        == make_eval_item(
            query=" 细胞膜的作用是什么? ",
            subject="生物",
            query_type="explanation",
            evidence=evidence,
            document_lookup=lookup,
        ).item_id
    )
    assert normalize_query(" 细胞膜  的作用是什么? ") == "细胞膜 的作用是什么?"
    source_annotated = make_eval_item(
        query="细胞膜的作用是什么?",
        subject="生物",
        query_type="explanation",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b2"],
            )
        ],
        query_source=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b1"],
            )
        ],
        document_lookup=lookup,
    )
    source_free = make_eval_item(
        query="细胞膜的作用是什么?",
        subject="生物",
        query_type="explanation",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b2"],
            )
        ],
        document_lookup=lookup,
    )
    assert source_annotated.item_id == source_free.item_id
    assert source_annotated.query_source[0].source_block_ids == ["p1-b1"]
    with pytest.raises(ValidationError, match="extra"):
        RetrievalEvidence(
            document_id=document.document_id,
            page_number=1,
            source_block_ids=["p1-b1"],
            unexpected="not allowed",
        )

    changed = _document()
    changed.pages[0].blocks[1].text = "细胞膜可以选择性地控制物质进出。"
    assert evidence_fingerprint(lookup, evidence) != evidence_fingerprint(
        {changed.document_id: changed},
        evidence,
    )


def test_candidate_generation_records_text_scope_exclusions() -> None:
    corpus = {DOCUMENT_ID: _corpus_document()}
    candidates, exclusions = generate_candidate_items(corpus, per_subject=1)

    assert len(candidates) == 1
    assert candidates[0].verification_status == "candidate"
    assert {record.reason for record in exclusions} == {
        "captionless_image_only",
        "table_missing_content",
    }
    assert "p1-b5" not in candidates[0].evidence[0].source_block_ids
    assert "p1-b6" not in candidates[0].evidence[0].source_block_ids


def test_candidate_generation_prefers_source_questions_over_heading_templates() -> None:
    document = CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="学习提示"),
                    TextBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        text="为什么金属会生锈\uff1f",
                    ),
                    TextBlock(
                        block_id="p1-b3",
                        reading_order=2,
                        text="金属锈蚀会影响材料的使用寿命,潮湿且有氧气时更容易发生。",
                    ),
                ],
            )
        ],
    )
    candidates, _ = generate_candidate_items(
        {
            DOCUMENT_ID: CorpusDocument(
                document=document,
                subject="化学",
                benchmark_item_id="item-test",
                relative_path="subject/test.pdf",
            )
        },
        per_subject=1,
    )

    assert candidates[0].query == "为什么金属会生锈\uff1f"
    assert candidates[0].verification_status == "candidate"
    assert "主要事实是什么" not in candidates[0].query
    assert "什么是学习提示" not in candidates[0].query
    assert "…" not in candidates[0].query
    assert "..." not in candidates[0].query
    assert candidates[0].query_source[0].source_block_ids == ["p1-b2"]
    assert candidates[0].evidence[0].source_block_ids == ["p1-b3"]


def test_candidate_generation_rejects_heading_metadata_and_truncated_questions() -> None:
    document = CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="学习提示"),
                    TextBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        text=(
                            "本章复习题\uff1f根据上述材料说明……\uff1f出版社信息是什么\uff1f"
                            "为什么铁制品在潮湿环境中更容易生锈\uff1f"
                        ),
                    ),
                    TextBlock(
                        block_id="p1-b3",
                        reading_order=2,
                        text="铁制品在潮湿并且有氧气的环境中更容易生锈。",
                    ),
                ],
            )
        ],
    )
    candidates, _ = generate_candidate_items(
        {
            DOCUMENT_ID: CorpusDocument(
                document=document,
                subject="化学",
                benchmark_item_id="item-test",
                relative_path="subject/test.pdf",
            )
        },
        per_subject=1,
    )

    assert candidates[0].query == "为什么铁制品在潮湿环境中更容易生锈\uff1f"
    assert "学习提示" not in candidates[0].query
    assert "出版社" not in candidates[0].query
    assert "上述" not in candidates[0].query
    assert "…" not in candidates[0].query
    assert candidates[0].query_source[0].source_block_ids == ["p1-b2"]
    assert candidates[0].evidence[0].source_block_ids == ["p1-b3"]


def test_source_question_without_nearby_answer_text_is_rejected() -> None:
    document = CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="学习提示"),
                    TextBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        text="为什么金属会生锈\uff1f",
                    ),
                ],
            )
        ],
    )

    with pytest.raises(RetrievalEvalError, match="usable candidates"):
        generate_candidate_items(
            {
                DOCUMENT_ID: CorpusDocument(
                    document=document,
                    subject="化学",
                    benchmark_item_id="item-test",
                    relative_path="subject/test.pdf",
                )
            },
            per_subject=1,
        )


def test_definition_candidates_require_substantive_definition_evidence() -> None:
    document = CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="学习提示"),
                    TextBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        text="金属材料是指由纯金属或合金制成的材料。",
                    ),
                ],
            )
        ],
    )
    candidates, _ = generate_candidate_items(
        {
            DOCUMENT_ID: CorpusDocument(
                document=document,
                subject="化学",
                benchmark_item_id="item-test",
                relative_path="subject/test.pdf",
            )
        },
        per_subject=1,
    )

    assert candidates[0].query == "什么是金属材料\uff1f"
    assert candidates[0].query_type == "definition"
    assert candidates[0].evidence[0].source_block_ids == ["p1-b2"]


def test_validation_reports_query_leakage_in_evidence_and_chunks() -> None:
    document = _document()
    corpus = {document.document_id: _corpus_document()}
    item = make_eval_item(
        query="细胞膜具有选择透过性\uff1f",
        subject="生物",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b2"],
            )
        ],
        document_lookup={document.document_id: document},
    )
    chunked = chunk_document(document, ChunkingConfig())
    indexed = {
        chunk.chunk_id: IndexedChunk(
            schema_version=chunk.schema_version,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            section_path=chunk.section_path,
            content_types=chunk.content_types,
            asset_refs=chunk.asset_refs,
            config_fingerprint=chunked.config_fingerprint,
        )
        for chunk in chunked.chunks
    }

    report = validate_items([item], indexed, corpus)

    assert not report.valid
    assert report.query_in_gold_evidence_count == 1
    assert report.query_in_gold_chunk_count == 1
    assert {"query_in_gold_evidence", "query_in_gold_chunk"} <= {
        issue.code for issue in report.issues
    }


def test_validation_surfaces_repeated_evidence_and_chunk_groups() -> None:
    document = _document()
    corpus = {document.document_id: _corpus_document()}
    lookup = {document.document_id: document}
    evidence = [
        RetrievalEvidence(
            document_id=document.document_id,
            page_number=1,
            source_block_ids=["p1-b2"],
        )
    ]
    first = make_eval_item(
        query="细胞膜的选择透过性说明什么\uff1f",
        subject="生物",
        query_type="explanation",
        evidence=evidence,
        document_lookup=lookup,
    )
    second = make_eval_item(
        query="细胞膜如何控制物质进出\uff1f",
        subject="生物",
        query_type="explanation",
        evidence=evidence,
        document_lookup=lookup,
    )
    chunked = chunk_document(document, ChunkingConfig())
    indexed = {
        chunk.chunk_id: IndexedChunk(
            schema_version=chunk.schema_version,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            section_path=chunk.section_path,
            content_types=chunk.content_types,
            asset_refs=chunk.asset_refs,
            config_fingerprint=chunked.config_fingerprint,
        )
        for chunk in chunked.chunks
    }

    report = validate_items([first, second], indexed, corpus)

    assert report.valid
    assert report.repeated_evidence_groups
    assert report.repeated_relevant_chunk_groups
    assert {first.item_id, second.item_id} <= set(report.repeated_evidence_groups[0].item_ids)


def test_review_pack_keeps_query_source_out_of_answer_chunks() -> None:
    document = CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="金属锈蚀"),
                    TextBlock(block_id="p1-b2", reading_order=1, text="为什么金属会生锈\uff1f"),
                    TextBlock(
                        block_id="p1-b3",
                        reading_order=2,
                        text="潮湿且有氧气时,铁制品更容易发生锈蚀。",
                    ),
                ],
            )
        ],
    )
    corpus_document = CorpusDocument(
        document=document,
        subject="化学",
        benchmark_item_id="item-test",
        relative_path="subject/test.pdf",
    )
    item, _ = generate_candidate_items({DOCUMENT_ID: corpus_document}, per_subject=1)
    answer = item[0]
    chunk = IndexedChunk(
        schema_version="1.0",
        chunk_id="chunk-answer",
        document_id=DOCUMENT_ID,
        ordinal=0,
        text="潮湿且有氧气时,铁制品更容易发生锈蚀。",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b3"],
        section_path=["金属锈蚀"],
        content_types=["text"],
        asset_refs=[],
        config_fingerprint="1" * 64,
    )

    review = build_review_pack_item(
        answer,
        {chunk.chunk_id: chunk},
        {DOCUMENT_ID: corpus_document},
    )

    assert review.query_source[0].source_block_ids == ["p1-b2"]
    assert all(
        "p1-b2" not in derived.source_block_ids
        for evidence in review.evidence
        for derived in evidence.derived_chunks
    )


def test_a15_missing_table_warnings_are_recorded_without_fake_source_blocks(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "benchmark_document_uuid": str(DOCUMENT_ID),
                "benchmark_item_id": "item-test",
                "table_missing_content": 1,
                "warnings": ["table_missing_content:item=42"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exclusions = load_a15_missing_table_exclusions(
        {DOCUMENT_ID: _corpus_document()},
        results_path,
    )

    assert len(exclusions) == 1
    assert exclusions[0].reason == "table_missing_content"
    assert exclusions[0].source_block_id is None
    assert exclusions[0].detail == "table_missing_content:item=42"


def test_lineage_derivation_supports_one_source_block_in_multiple_chunks() -> None:
    document = _document()
    lookup = {document.document_id: document}
    item = make_eval_item(
        query="细胞膜的作用是什么?",
        subject="生物",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b2"],
            )
        ],
        document_lookup=lookup,
    )
    chunked = chunk_document(document, ChunkingConfig(target_chars=10, max_chars=10, min_chars=0))
    indexed = {
        chunk.chunk_id: IndexedChunk(
            schema_version=chunk.schema_version,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            section_path=chunk.section_path,
            content_types=chunk.content_types,
            asset_refs=chunk.asset_refs,
            config_fingerprint=chunked.config_fingerprint,
        )
        for chunk in chunked.chunks
    }
    materialized = derive_materialized_item(
        item,
        indexed,
        {document.document_id: _corpus_document()},
    )

    assert len(materialized.relevant_chunk_ids) > 1
    assert materialized.all_gold_blocks_covered
    review = build_review_pack_item(item, indexed, {document.document_id: _corpus_document()})
    assert review.evidence[0].evidence_excerpt
    assert all(len(chunk.text_excerpt) <= 480 for chunk in review.evidence[0].derived_chunks)


def test_validation_detects_order_drift_fingerprint_duplicates_and_uncovered_evidence() -> None:
    document = _document()
    corpus = {document.document_id: _corpus_document()}
    lookup = {document.document_id: document}
    ordered = make_eval_item(
        query="顺序有效的问题",
        subject="生物",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b1", "p1-b2"],
            )
        ],
        document_lookup=lookup,
    )
    reversed_item = make_eval_item(
        query="顺序错误的问题",
        subject="生物",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b2", "p1-b1"],
            )
        ],
        document_lookup=lookup,
    )
    duplicate_query = make_eval_item(
        query="顺序有效的问题",
        subject="生物",
        query_type="definition",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b3"],
            )
        ],
        document_lookup=lookup,
    )
    stale = ordered.model_copy(update={"evidence_fingerprint": "0" * 64})
    report = validate_items(
        [ordered, reversed_item, duplicate_query, stale],
        {},
        corpus,
    )
    codes = {issue.code for issue in report.issues}
    assert not report.valid
    assert {
        "invalid_source_order",
        "duplicate_normalized_query",
        "evidence_fingerprint_drift",
        "evidence_not_covered",
    } <= codes

    missing_payload = ordered.model_dump()
    missing_payload["evidence"] = [
        RetrievalEvidence(
            document_id=UUID("22222222-2222-2222-2222-222222222222"),
            page_number=9,
            source_block_ids=["missing"],
        )
    ]
    missing = RetrievalEvalItem.model_construct(**missing_payload)
    missing_report = validate_items([missing], {}, corpus)
    assert "missing_document" in {issue.code for issue in missing_report.issues}

    missing_page_payload = ordered.model_dump()
    missing_page_payload["evidence"] = [
        RetrievalEvidence(
            document_id=document.document_id,
            page_number=9,
            source_block_ids=["p9-b1"],
        )
    ]
    missing_page = RetrievalEvalItem.model_construct(**missing_page_payload)
    missing_block_payload = ordered.model_dump()
    missing_block_payload["evidence"] = [
        RetrievalEvidence(
            document_id=document.document_id,
            page_number=1,
            source_block_ids=["p1-missing"],
        )
    ]
    missing_block = RetrievalEvalItem.model_construct(**missing_block_payload)
    missing_parts = validate_items([missing_page, missing_block], {}, corpus)
    missing_codes = {issue.code for issue in missing_parts.issues}
    assert {"missing_page", "missing_block"} <= missing_codes


def test_review_actions_update_status_and_recompute_id_for_query_edit(tmp_path: Path) -> None:
    document = _document()
    corpus = {document.document_id: _corpus_document()}
    item = make_eval_item(
        query="原始问题",
        subject="生物",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=document.document_id,
                page_number=1,
                source_block_ids=["p1-b1"],
            )
        ],
        document_lookup={document.document_id: document},
    )
    chunked = chunk_document(document, ChunkingConfig())
    chunks = {
        chunk.chunk_id: IndexedChunk(
            schema_version=chunk.schema_version,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_block_ids=chunk.source_block_ids,
            section_path=chunk.section_path,
            content_types=chunk.content_types,
            asset_refs=chunk.asset_refs,
            config_fingerprint=chunked.config_fingerprint,
        )
        for chunk in chunked.chunks
    }
    from knowledge_scope.evaluation.retrieval_eval import (
        _status_counts,
        _write_json,
        _write_jsonl,
    )

    output_dir = tmp_path / "evaluation"
    materialized = derive_materialized_item(item, chunks, corpus)
    review = build_review_pack_item(item, chunks, corpus)
    _write_jsonl(output_dir / "candidates.jsonl", [item])
    _write_jsonl(output_dir / "materialized.jsonl", [materialized])
    _write_jsonl(output_dir / "review-pack.jsonl", [review])
    _write_json(output_dir / "manifest.json", {"status_counts": _status_counts([item])})

    edited = apply_review_action(output_dir, item.item_id, "edit", query="编辑后的问题")
    assert edited.verification_status == "candidate"
    assert edited.item_id != item.item_id
    typed_edit = apply_review_action(
        output_dir,
        edited.item_id,
        "edit",
        query="编辑后的定义问题",
        query_type="definition",
    )
    assert typed_edit.query_type == "definition"
    assert typed_edit.item_id != edited.item_id
    accepted = apply_review_action(output_dir, typed_edit.item_id, "accept")
    assert accepted.verification_status == "verified"
    stored = [
        json.loads(line) for line in (output_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert stored[0]["verification_status"] == "verified"


def test_final_split_assigns_whole_groups_and_exact_subject_quotas() -> None:
    items = []
    group_ids: dict[str, str] = {}
    for subject_index, subject in enumerate(("甲", "乙")):
        for item_index in range(3):
            item_id = f"a2-1-{subject_index * 3 + item_index + 1:064x}"
            item = RetrievalEvalItem.model_construct(
                schema_version="1.0",
                item_id=item_id,
                query=f"问题 {subject} {item_index}",
                subject=subject,
                query_type="factual",
                verification_status="verified",
                evidence=[],
                query_source=[],
                evidence_fingerprint="0" * 64,
            )
            items.append(item)
            group_ids[item_id] = f"group-{subject}-{item_index}"
    group_ids[items[0].item_id] = "shared-group"
    group_ids[items[1].item_id] = "shared-group"

    splits = _assign_final_splits(
        items,
        group_ids,
        dev_per_subject=2,
        test_per_subject=1,
    )

    assert splits[items[0].item_id] == splits[items[1].item_id]
    for subject in ("甲", "乙"):
        assert sum(splits[item.item_id] == "dev" for item in items if item.subject == subject) == 2
        assert sum(splits[item.item_id] == "test" for item in items if item.subject == subject) == 1


def test_final_dataset_item_rejects_non_verified_items() -> None:
    item = make_eval_item(
        query="问题",
        subject="甲",
        query_type="factual",
        evidence=[
            RetrievalEvidence(
                document_id=DOCUMENT_ID,
                page_number=1,
                source_block_ids=["p1-b1"],
            )
        ],
        document_lookup={DOCUMENT_ID: _document()},
    ).model_copy(update={"verification_status": "rejected"})

    with pytest.raises(ValidationError, match="verified items only"):
        FinalDatasetItem(split="dev", leakage_group_id="group", item=item)
