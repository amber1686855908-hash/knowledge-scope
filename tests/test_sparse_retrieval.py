from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from knowledge_scope.chunking.models import Chunk
from knowledge_scope.evaluation.sparse_diagnostic import (
    DiagnosticQuery,
    compare_rankings,
    run_diagnostic,
)
from knowledge_scope.retrieval.sparse import (
    SparseChunkRecord,
    SparseIndexBusyError,
    SparseIndexConfig,
    SparseIndexConfigurationError,
    SparseIndexError,
    SparseIndexStore,
    corpus_fingerprint,
    term_frequencies,
    tokenize,
)

KB = UUID("11111111-1111-4111-8111-111111111111")
OTHER_KB = UUID("22222222-2222-4222-8222-222222222222")


def _record(
    knowledge_base_id: UUID,
    document_id: UUID,
    chunk_id: str,
    text: str,
    *,
    ordinal: int = 0,
    content_types: list[str] | None = None,
    asset_refs: list[str] | None = None,
) -> SparseChunkRecord:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        ordinal=ordinal,
        text=text,
        page_start=1,
        page_end=1,
        source_block_ids=[f"{chunk_id}-block"],
        section_path=["测试章节"],
        content_types=content_types or ["text"],
        asset_refs=asset_refs or [],
    )
    return SparseChunkRecord(
        knowledge_base_id=knowledge_base_id,
        chunking_config_fingerprint="a" * 64,
        **chunk.model_dump(mode="json"),
    )


def test_tokenizer_supports_chinese_mixed_terms_numbers_and_formula_operands() -> None:
    tokens = tokenize("温度表 Qwen3-Embedding-0.6B 2024 H₂O E=mc^2 x_i\uff0c温度")

    assert "温度" in tokens
    assert "度表" in tokens
    assert "qwen3-embedding-0.6b" in tokens
    assert "2024" in tokens
    assert "e" in tokens
    assert "mc" in tokens
    assert "2" in tokens
    assert "x_i" in tokens
    assert "\uff0c" not in tokens


def test_tokenizer_splits_mixed_script_boundaries_without_losing_lexical_units() -> None:
    assert tokenize("CO2浓度") == ("co2", "浓", "度", "浓度")
    assert tokenize("2024年") == ("2024", "年")
    assert tokenize("NaCl浓度") == ("nacl", "浓", "度", "浓度")


def test_term_frequencies_are_deterministic_and_keep_repeated_terms() -> None:
    assert term_frequencies("温度 温度 2024") == {
        "2024": 1,
        "度": 2,
        "温": 2,
        "温度": 2,
    }
    assert term_frequencies("\uff21\uff22\uff23") == term_frequencies("abc")


def test_sparse_bm25_returns_lineage_and_deterministic_ranking(tmp_path: Path) -> None:
    document = uuid4()
    records = [
        _record(KB, document, "chunk-a", "温度表显示 2024 年的温度数据", ordinal=0),
        _record(KB, uuid4(), "chunk-b", "压力表显示 2023 年的数据", ordinal=0),
    ]

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        built = store.build(records)
        first = store.search("温度 2024", knowledge_base_id=KB, top_k=2)
        second = store.search("温度 2024", knowledge_base_id=KB, top_k=2)

    assert built.chunk_count == 2
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [item.chunk_id for item in first.items] == ["chunk-a"]
    assert first.items[0].document_id == document
    assert first.items[0].source_block_ids == ["chunk-a-block"]
    assert first.items[0].page_start == 1
    assert first.items[0].matched_terms


def test_sparse_search_preserves_kb_isolation_and_document_filter(tmp_path: Path) -> None:
    document = uuid4()
    records = [
        _record(KB, document, "same-chunk", "只属于第一个知识库"),
        _record(OTHER_KB, document, "same-chunk", "只属于第二个知识库"),
    ]

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        store.build(records)
        first = store.search("第一个", knowledge_base_id=KB)
        second = store.search("第二个", knowledge_base_id=OTHER_KB)
        filtered = store.search("知识库", knowledge_base_id=KB, document_id=document)

    assert [item.knowledge_base_id for item in first.items] == [KB]
    assert [item.knowledge_base_id for item in second.items] == [OTHER_KB]
    assert [item.chunk_id for item in filtered.items] == ["same-chunk"]


def test_mixed_script_terms_are_retrievable_as_individual_components(tmp_path: Path) -> None:
    record = _record(KB, uuid4(), "mixed", "CO2浓度在2024年有记录")

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        store.build([record])
        cjk_result = store.search("浓度", knowledge_base_id=KB)
        year_result = store.search("2024", knowledge_base_id=KB)

    assert [item.chunk_id for item in cjk_result.items] == ["mixed"]
    assert [item.chunk_id for item in year_result.items] == ["mixed"]


def test_formula_and_asset_only_chunks_keep_lineage_without_fake_terms(tmp_path: Path) -> None:
    document = uuid4()
    formula = _record(KB, document, "formula", "$$ E=mc^2 $$", content_types=["formula"])
    asset_only = _record(
        KB,
        document,
        "image",
        "",
        ordinal=1,
        content_types=["image"],
        asset_refs=["image-1"],
    )

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        built = store.build([formula, asset_only])
        result = store.search("mc 2", knowledge_base_id=KB)

    assert built.chunk_count == 2
    assert built.searchable_chunk_count == 1
    assert [item.chunk_id for item in result.items] == ["formula"]


def test_rebuild_removes_stale_chunks_and_delete_is_kb_scoped(tmp_path: Path) -> None:
    document = uuid4()
    other_document = uuid4()
    old = _record(KB, document, "old", "苹果")
    new = _record(KB, document, "new", "橘子")
    other = _record(OTHER_KB, other_document, "other", "保留词")

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        store.build([old, other])
        store.replace_document(KB, document, [new])
        assert store.search("苹果", knowledge_base_id=KB).items == []
        assert [item.chunk_id for item in store.search("橘子", knowledge_base_id=KB).items] == [
            "new"
        ]
        assert [
            item.chunk_id for item in store.search("保留词", knowledge_base_id=OTHER_KB).items
        ] == ["other"]
        assert store.delete_document(KB, document) == 1
        assert store.delete_document(KB, document) == 0
        assert [
            item.chunk_id for item in store.search("保留词", knowledge_base_id=OTHER_KB).items
        ] == ["other"]


def test_failed_generation_preserves_last_known_good_searchable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = uuid4()
    old = _record(KB, document, "old", "甲乙")
    new = _record(KB, document, "new", "丙丁")

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        store.build([old])

        def fail_insert(_generation_id: str, _records: object) -> None:
            raise sqlite3.OperationalError("injected build failure")

        monkeypatch.setattr(store, "_insert_generation_records", fail_insert)
        with pytest.raises(SparseIndexError, match="previous generation was preserved"):
            store.build([new])
        assert [item.chunk_id for item in store.search("甲乙", knowledge_base_id=KB).items] == [
            "old"
        ]
        assert store.search("丙丁", knowledge_base_id=KB).items == []


def test_build_is_idempotent_and_audit_detects_content_drift(tmp_path: Path) -> None:
    record = _record(KB, uuid4(), "chunk", "稳定内容")
    drifted = record.model_copy(update={"text": "已变化内容"})

    with SparseIndexStore(tmp_path / "sparse.sqlite3") as store:
        first = store.build([record])
        second = store.build([record])
        ready = store.audit([record])
        stale = store.audit([drifted])

    assert first.generation_id == second.generation_id
    assert ready.status == "ready"
    assert ready.fingerprint_matches is True
    assert stale.status == "stale"
    assert stale.fingerprint_matches is False
    assert corpus_fingerprint([record]) == corpus_fingerprint([record])


def test_rebuild_removes_stale_generation_rows(tmp_path: Path) -> None:
    document = uuid4()
    old = _record(KB, document, "old", "旧内容")
    new = _record(KB, document, "new", "新内容")
    path = tmp_path / "sparse.sqlite3"

    with SparseIndexStore(path) as store:
        first = store.build([old])
        second = store.build([new])

    assert second.stale_generations_removed == 1
    with sqlite3.connect(path) as connection:
        active_generation = connection.execute(
            "SELECT value FROM sparse_meta WHERE key = 'active_generation_id'"
        ).fetchone()[0]
        for table in ("sparse_generations", "sparse_chunks", "sparse_doc_stats", "sparse_postings"):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE generation_id != ?",
                    (active_generation,),
                ).fetchone()[0]
                == 0
            )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sparse_generations WHERE generation_id = ?",
                (first.generation_id,),
            ).fetchone()[0]
            == 0
        )


def test_concurrent_writers_fail_without_invalidating_the_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sparse.sqlite3"
    base = _record(KB, uuid4(), "base", "base-generation")
    first_replacement = _record(KB, uuid4(), "first", "alpha-first-generation")
    second_replacement = _record(KB, uuid4(), "second", "beta-second-generation")

    with SparseIndexStore(path) as first, SparseIndexStore(path) as second:
        first.build([base])
        entered_cleanup = threading.Event()
        release_cleanup = threading.Event()
        original_superseded_ids = first._superseded_generation_ids

        def paused_superseded_ids(active_generation_id: str) -> set[str]:
            entered_cleanup.set()
            if not release_cleanup.wait(timeout=5):
                raise RuntimeError("test synchronization timed out")
            return original_superseded_ids(active_generation_id)

        monkeypatch.setattr(first, "_superseded_generation_ids", paused_superseded_ids)
        errors: list[BaseException] = []

        def replace_first() -> None:
            try:
                first.build([first_replacement])
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=replace_first)
        worker.start()
        assert entered_cleanup.wait(timeout=5)
        with pytest.raises(SparseIndexBusyError, match="already writing"):
            second.build([second_replacement])
        release_cleanup.set()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert errors == []
        active = second._active_generation()
        assert active is not None
        assert [
            item.chunk_id
            for item in second.search("alpha-first-generation", knowledge_base_id=KB).items
        ] == ["first"]
        assert second.search("beta-second-generation", knowledge_base_id=KB).items == []


def test_inflight_reader_keeps_one_generation_snapshot_during_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sparse.sqlite3"
    old = _record(KB, uuid4(), "old", "reader-old-term")
    new = _record(KB, uuid4(), "new", "writer-new-term")

    with SparseIndexStore(path) as writer, SparseIndexStore(path) as reader:
        writer.build([old])
        active_captured = threading.Event()
        release_reader = threading.Event()
        original_active = reader._active_generation

        def paused_active() -> object:
            active = original_active()
            active_captured.set()
            if not release_reader.wait(timeout=5):
                raise RuntimeError("test synchronization timed out")
            return active

        monkeypatch.setattr(reader, "_active_generation", paused_active)
        responses = []
        errors: list[BaseException] = []

        def read_old_generation() -> None:
            try:
                responses.append(reader.search("reader-old-term", knowledge_base_id=KB))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=read_old_generation)
        thread.start()
        assert active_captured.wait(timeout=5)
        writer.build([new])
        release_reader.set()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []
        assert len(responses) == 1
        assert [item.chunk_id for item in responses[0].items] == ["old"]


def test_incompatible_bm25_configuration_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "sparse.sqlite3"
    record = _record(KB, uuid4(), "chunk", "内容")
    with SparseIndexStore(path) as store:
        store.build([record])

    with SparseIndexStore(path, config=SparseIndexConfig(bm25_b=0.5)) as incompatible:
        readiness = incompatible.readiness()
        assert readiness.status == "unavailable"
        with pytest.raises(SparseIndexConfigurationError):
            incompatible.build([record])


def test_mixed_script_v1_index_is_incompatible_with_v2(tmp_path: Path) -> None:
    path = tmp_path / "sparse.sqlite3"
    record = _record(KB, uuid4(), "chunk", "2024年 CO2浓度")
    legacy_config = SparseIndexConfig(
        tokenizer_version="mixed-script-v1",
        normalization="NFKC + casefold; CJK unigrams and bigrams; alphanumeric runs",
    )
    current_config = SparseIndexConfig()
    assert legacy_config.static_fingerprint != current_config.static_fingerprint

    with SparseIndexStore(path, config=legacy_config) as legacy:
        legacy.build([record])

    with SparseIndexStore(path, config=current_config) as current:
        assert current.readiness().status == "unavailable"
        with pytest.raises(SparseIndexConfigurationError):
            current.build([record])


def test_protected_vector_collection_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SparseIndexConfigurationError):
        SparseIndexStore(tmp_path / "knowledgescope_chunks_v1.sqlite3")
    with pytest.raises(SparseIndexConfigurationError):
        SparseIndexStore(tmp_path / "knowledgescope_representations_v1")


def test_sparse_record_rejects_invalid_lineage() -> None:
    payload = _record(KB, uuid4(), "invalid", "text").model_dump(mode="json")
    payload["page_end"] = 0
    with pytest.raises(ValidationError):
        SparseChunkRecord.model_validate(payload)


def test_sparse_record_rejects_blank_chunk_id() -> None:
    payload = _record(KB, uuid4(), "valid", "text").model_dump(mode="json")
    payload["chunk_id"] = "   "

    with pytest.raises(ValidationError, match="chunk_id"):
        SparseChunkRecord.model_validate(payload)


def test_sparse_build_rejects_non_contiguous_document_ordinals(tmp_path: Path) -> None:
    document = uuid4()
    records = [
        _record(KB, document, "first", "第一段", ordinal=0),
        _record(KB, document, "third", "第三段", ordinal=2),
    ]

    with (
        SparseIndexStore(tmp_path / "sparse.sqlite3") as store,
        pytest.raises(SparseIndexError, match="ordinals must be contiguous"),
    ):
        store.build(records)


def test_sparse_diagnostic_keeps_branch_rankings_and_reports_overlap() -> None:
    query = DiagnosticQuery(
        item_id="a2-1-item",
        query="温度",
        knowledge_base_id=KB,
        relevant_chunk_ids=frozenset({"sparse-gold"}),
    )

    records, summary = run_diagnostic(
        [query],
        sparse_search=lambda _query, _kb: ["sparse-gold", "shared"],
        dense_search=lambda _query, _kb: ["shared", "dense-only"],
        top_k=2,
    )

    assert records[0] == compare_rankings(
        "a2-1-item",
        ["sparse-gold", "shared"],
        ["shared", "dense-only"],
        frozenset({"sparse-gold"}),
    )
    assert summary.sparse_only_hit_count == 1
    assert summary.dense_only_hit_count == 0
