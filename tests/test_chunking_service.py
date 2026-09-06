from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from knowledge_scope.chunking.models import ChunkedDocument, ChunkingConfig
from knowledge_scope.chunking.service import (
    chunk_document,
    chunk_document_by_id,
    summarize_chunked_document,
)
from knowledge_scope.parsing.models import (
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _document(pages: list[list[CanonicalBlock]]) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(page_number=page_number, blocks=blocks)
            for page_number, blocks in enumerate(pages, start=1)
        ],
    )


def _text(block_id: str, order: int, text: str) -> TextBlock:
    return TextBlock(block_id=block_id, reading_order=order, text=text)


def _title(block_id: str, order: int, text: str) -> TitleBlock:
    return TitleBlock(block_id=block_id, reading_order=order, text=text)


def test_under_budget_section_groups_blocks_and_preserves_context() -> None:
    document = _document(
        [
            [
                _title("p1-b1", 0, "章节一"),
                _text("p1-b2", 1, "第一段正文。"),
                _text("p1-b3", 2, "第二段正文。"),
            ]
        ]
    )

    chunked = chunk_document(document, ChunkingConfig(target_chars=100, max_chars=200, min_chars=0))

    assert len(chunked.chunks) == 1
    chunk = chunked.chunks[0]
    assert chunk.ordinal == 0
    assert chunk.document_id == DOCUMENT_ID
    assert chunk.page_start == chunk.page_end == 1
    assert chunk.source_block_ids == ["p1-b1", "p1-b2", "p1-b3"]
    assert chunk.section_path == ["章节一"]
    assert chunk.content_types == ["title", "text"]
    assert "## 章节一" in chunk.text
    assert "第一段正文。" in chunk.text


def test_titles_create_independent_sections_without_crossing_boundaries() -> None:
    document = _document(
        [
            [
                _title("p1-b1", 0, "章节一"),
                _text("p1-b2", 1, "第一章正文。"),
                _title("p1-b3", 2, "章节二"),
                _text("p1-b4", 3, "第二章正文。"),
            ]
        ]
    )

    chunked = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800))

    assert len(chunked.chunks) == 2
    assert [chunk.section_path for chunk in chunked.chunks] == [["章节一"], ["章节二"]]
    assert chunked.chunks[0].source_block_ids == ["p1-b1", "p1-b2"]
    assert chunked.chunks[1].source_block_ids == ["p1-b3", "p1-b4"]


def test_consecutive_titles_join_context_when_substantive_content_arrives() -> None:
    document = _document(
        [
            [
                _title("p1-b1", 0, "一级标题"),
                _title("p1-b2", 1, "二级标题"),
                _text("p1-b3", 2, "正文内容。"),
                FormulaBlock(block_id="p1-b4", reading_order=3, latex=r"x = 1"),
            ]
        ]
    )

    chunk = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800)).chunks[0]

    assert chunk.section_path == ["一级标题", "二级标题"]
    assert chunk.source_block_ids == ["p1-b1", "p1-b2", "p1-b3", "p1-b4"]
    assert chunk.content_types == ["title", "text", "formula"]
    assert "## 一级标题" in chunk.text
    assert "## 二级标题" in chunk.text


def test_new_consecutive_title_sequence_starts_a_new_section() -> None:
    document = _document(
        [
            [
                _title("p1-b1", 0, "第一组标题"),
                _title("p1-b2", 1, "第一组副标题"),
                _text("p1-b3", 2, "第一组正文。"),
                _title("p1-b4", 3, "第二组标题"),
                _title("p1-b5", 4, "第二组副标题"),
                _text("p1-b6", 5, "第二组正文。"),
            ]
        ]
    )

    chunked = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800))

    assert [chunk.section_path for chunk in chunked.chunks] == [
        ["第一组标题", "第一组副标题"],
        ["第二组标题", "第二组副标题"],
    ]
    assert [chunk.source_block_ids for chunk in chunked.chunks] == [
        ["p1-b1", "p1-b2", "p1-b3"],
        ["p1-b4", "p1-b5", "p1-b6"],
    ]


def test_trailing_title_sequence_remains_a_title_only_section() -> None:
    document = _document(
        [
            [
                _text("p1-b1", 0, "正文。"),
                _title("p1-b2", 1, "尾部标题"),
                _title("p1-b3", 2, "尾部副标题"),
            ]
        ]
    )

    chunked = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800))

    assert len(chunked.chunks) == 2
    assert chunked.chunks[-1].section_path == ["尾部标题", "尾部副标题"]
    assert chunked.chunks[-1].source_block_ids == ["p1-b2", "p1-b3"]
    assert chunked.chunks[-1].content_types == ["title"]
    assert {block_id for chunk in chunked.chunks for block_id in chunk.source_block_ids} == {
        "p1-b1",
        "p1-b2",
        "p1-b3",
    }


def test_long_section_splits_at_block_boundaries() -> None:
    document = _document(
        [
            [
                _text("p1-b1", 0, "a" * 40),
                _text("p1-b2", 1, "b" * 40),
                _text("p1-b3", 2, "c" * 40),
            ]
        ]
    )

    chunked = chunk_document(document, ChunkingConfig(target_chars=50, max_chars=90, min_chars=10))

    assert [chunk.source_block_ids for chunk in chunked.chunks] == [
        ["p1-b1", "p1-b2"],
        ["p1-b3"],
    ]
    assert all(len(chunk.text) <= 90 for chunk in chunked.chunks)


def test_oversized_single_text_block_uses_deterministic_fallback() -> None:
    source_text = "第一句。" + "连续文本" * 20 + "最后一句\uff01"
    document = _document([[_text("p1-b1", 0, source_text)]])

    chunked = chunk_document(document, ChunkingConfig(target_chars=20, max_chars=30, min_chars=0))

    assert len(chunked.chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in chunked.chunks)
    assert all(chunk.source_block_ids == ["p1-b1"] for chunk in chunked.chunks)
    assert "".join(chunk.text for chunk in chunked.chunks) == source_text
    assert len({chunk.chunk_id for chunk in chunked.chunks}) == len(chunked.chunks)


def test_pending_titles_join_the_first_piece_of_oversized_text() -> None:
    document = _document([[_title("p1-b1", 0, "章节"), _text("p1-b2", 1, "正文" * 40)]])
    config = ChunkingConfig(target_chars=20, max_chars=30, min_chars=0)

    chunked = chunk_document(document, config)

    assert chunked.chunks[0].source_block_ids == ["p1-b1", "p1-b2"]
    assert chunked.chunks[0].content_types == ["title", "text"]
    assert all(chunk.content_types != ["title"] for chunk in chunked.chunks)
    assert all(len(chunk.text) <= config.max_chars for chunk in chunked.chunks)


def test_title_context_is_not_emitted_alone_when_text_needs_budget() -> None:
    document = _document([[_title("p1-b1", 0, "标题" * 10), _text("p1-b2", 1, "正文" * 10)]])
    config = ChunkingConfig(target_chars=40, max_chars=40, min_chars=0)

    chunked = chunk_document(document, config)

    assert chunked.chunks[0].source_block_ids == ["p1-b1", "p1-b2"]
    assert chunked.chunks[0].content_types == ["title", "text"]
    assert all(chunk.content_types != ["title"] for chunk in chunked.chunks)
    assert all(len(chunk.text) <= config.max_chars for chunk in chunked.chunks)


def test_oversized_text_fallback_is_not_reconstituted_when_followed_by_image() -> None:
    source_text = "长文本。" + "连续内容" * 60
    document = _document(
        [
            [
                _text("p1-b1", 0, source_text),
                ImageBlock(block_id="p1-b2", reading_order=1, asset_ref="assets/image.png"),
            ]
        ]
    )

    config = ChunkingConfig(target_chars=100, max_chars=200, min_chars=20)
    chunked = chunk_document(document, config)

    assert all(len(chunk.text) <= config.max_chars for chunk in chunked.chunks)
    assert chunked.chunks[-1].source_block_ids == ["p1-b1", "p1-b2"]
    assert "[image]" not in chunked.chunks[-1].text
    assert chunked.chunks[-1].asset_refs == ["assets/image.png"]


def test_rendering_preserves_formula_tables_captions_and_assets() -> None:
    document = _document(
        [
            [
                TableBlock(
                    block_id="p1-b1",
                    reading_order=0,
                    markdown="| 项目 | 状态 |\n| --- | --- |\n| 温度 | 正常 |",
                    asset_ref="assets/table.png",
                ),
                FormulaBlock(block_id="p1-b2", reading_order=1, latex=r"E = mc^2"),
                ImageBlock(
                    block_id="p1-b3",
                    reading_order=2,
                    asset_ref="assets/diagram.png",
                    caption="结构示意图",
                ),
                ImageBlock(block_id="p1-b4", reading_order=3, asset_ref="assets/photo.png"),
            ]
        ]
    )

    chunk = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800)).chunks[0]

    assert "| 项目 | 状态 |" in chunk.text
    assert "$$\nE = mc^2\n$$" in chunk.text
    assert "结构示意图" in chunk.text
    assert "[image]" not in chunk.text
    assert chunk.content_types == ["table", "formula", "image"]
    assert chunk.asset_refs == ["assets/table.png", "assets/diagram.png", "assets/photo.png"]


def test_html_table_is_rendered_as_stable_readable_rows() -> None:
    document = _document(
        [
            [
                TableBlock(
                    block_id="p1-b1",
                    reading_order=0,
                    html="<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                )
            ]
        ]
    )

    chunk = chunk_document(
        document, ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    ).chunks[0]

    assert chunk.text == "A | B\n1 | 2"
    assert "<table>" not in chunk.text


def test_asset_only_table_has_no_invented_table_content() -> None:
    document = _document(
        [
            [
                TableBlock(
                    block_id="p1-b1",
                    reading_order=0,
                    asset_ref="assets/table.png",
                )
            ]
        ]
    )

    chunk = chunk_document(
        document, ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    ).chunks[0]

    assert chunk.text == ""
    assert chunk.asset_refs == ["assets/table.png"]


def test_captionless_image_is_represented_by_asset_ref_without_fake_text() -> None:
    document = _document(
        [[ImageBlock(block_id="p1-b1", reading_order=0, asset_ref="assets/image.png")]]
    )

    chunk = chunk_document(
        document, ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    ).chunks[0]

    assert chunk.text == ""
    assert chunk.content_types == ["image"]
    assert chunk.asset_refs == ["assets/image.png"]


def test_formula_reclaims_nearby_whole_block_context_without_crossing_budget() -> None:
    document = _document(
        [
            [
                _title("p1-b1", 0, "章节"),
                _text("p1-b2", 1, "a" * 700),
                _text("p1-b3", 2, "b" * 40),
                FormulaBlock(block_id="p1-b4", reading_order=3, latex="x" * 1500),
            ]
        ]
    )
    config = ChunkingConfig(target_chars=1200, max_chars=1600, min_chars=0)

    chunked = chunk_document(document, config)

    assert [chunk.source_block_ids for chunk in chunked.chunks] == [
        ["p1-b1", "p1-b2"],
        ["p1-b3", "p1-b4"],
    ]
    assert chunked.chunks[-1].content_types == ["text", "formula"]
    assert len(chunked.chunks[-1].text) <= config.max_chars


def test_oversized_formula_remains_atomic_and_is_reported() -> None:
    document = _document([[FormulaBlock(block_id="p1-b1", reading_order=0, latex="x" * 100)]])
    config = ChunkingConfig(target_chars=20, max_chars=30, min_chars=0)

    chunked = chunk_document(document, config)
    summary = summarize_chunked_document(document, chunked, config)

    assert len(chunked.chunks) == 1
    assert chunked.chunks[0].source_block_ids == ["p1-b1"]
    assert len(chunked.chunks[0].text) > config.max_chars
    assert summary["oversized_atomic_chunks"] == 1
    assert summary["oversized_atomic_formula_chunks"] == 1


def test_chunks_can_span_pages_while_preserving_source_order_and_range() -> None:
    document = _document(
        [
            [_title("p1-b1", 0, "跨页章节"), _text("p1-b2", 1, "第一页内容。")],
            [_text("p2-b1", 0, "第二页内容。")],
        ]
    )

    chunk = chunk_document(document, ChunkingConfig(target_chars=500, max_chars=800)).chunks[0]

    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert chunk.source_block_ids == ["p1-b1", "p1-b2", "p2-b1"]
    assert chunk.section_path == ["跨页章节"]


def test_empty_canonical_page_produces_a_valid_empty_chunk_artifact() -> None:
    document = _document([[]])

    chunked = chunk_document(document, ChunkingConfig())

    assert chunked.page_count == 1
    assert chunked.chunks == []


def test_chunking_is_repeatable_and_ids_include_config_fingerprint() -> None:
    document = _document([[_text("p1-b1", 0, "稳定内容。")]])
    first_config = ChunkingConfig(target_chars=100, max_chars=200, min_chars=0)
    second_config = ChunkingConfig(target_chars=101, max_chars=200, min_chars=0)

    first = chunk_document(document, first_config)
    repeated = chunk_document(document, first_config)
    changed = chunk_document(document, second_config)

    assert first.model_dump_json() == repeated.model_dump_json()
    assert first.config_fingerprint != changed.config_fingerprint
    assert first.chunks[0].chunk_id != changed.chunks[0].chunk_id


def test_summary_reports_full_coverage_and_fallback_references() -> None:
    document = _document([[_text("p1-b1", 0, "很长的文本。" * 20)]])
    config = ChunkingConfig(target_chars=20, max_chars=30, min_chars=0)
    chunked = chunk_document(document, config)

    summary = summarize_chunked_document(document, chunked, config)

    assert summary["source_blocks"] == {
        "eligible": 1,
        "referenced": 1,
        "unreferenced_meaningful": 0,
        "multiply_referenced": 1,
        "coverage_rate": 1.0,
    }
    assert summary["oversized_single_block_fallback_count"] == 1
    assert summary["chunks_over_max_chars"] == 0
    assert summary["oversized_atomic_chunks"] == 0
    assert summary["sections_split_by_budget"] == 1
    assert summary["section_boundary_violations"] == 0


def test_chunk_document_by_id_persists_validated_artifacts_atomically(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    canonical_path = settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json"
    canonical_path.parent.mkdir(parents=True)
    document = _document([[_title("p1-b1", 0, "章节"), _text("p1-b2", 1, "内容。")]])
    canonical_path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    result = chunk_document_by_id(DOCUMENT_ID, settings)
    artifact_root = settings.data_dir / "chunking" / str(DOCUMENT_ID)
    chunked_path = artifact_root / "chunks.json"
    manifest_path = artifact_root / "manifest.json"
    stored = ChunkedDocument.model_validate_json(chunked_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.chunk_count == len(stored.chunks) == 1
    assert result.chunks_ref == f"chunking/{DOCUMENT_ID}/chunks.json"
    assert manifest["source_canonical_ref"] == f"parsing/{DOCUMENT_ID}/canonical.json"
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")

    first_chunks = chunked_path.read_bytes()
    chunk_document_by_id(DOCUMENT_ID, settings)
    assert chunked_path.read_bytes() == first_chunks
