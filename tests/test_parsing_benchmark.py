from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_scope.evaluation import parsing_benchmark as benchmark
from knowledge_scope.parsing.mineru_adapter import (
    AdaptedCanonicalDocument,
    AdapterStats,
    MineruAdapterError,
)
from knowledge_scope.parsing.mineru_runner import MineruRunnerError, MineruRunResult
from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock
from knowledge_scope.shared.config import Settings


def _write_pdf(path: Path, content: bytes = b"%PDF-test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _settings(data_dir: Path) -> Settings:
    return Settings(_env_file=None, data_dir=data_dir)


def _manual_item(
    item_id: str,
    relative_path: str,
    sha256: str,
    *,
    subject: str = "数学",
    physical_pages: int = 1,
) -> benchmark.InventoryItem:
    return benchmark.InventoryItem(
        benchmark_item_id=item_id,
        basename=Path(relative_path).name,
        relative_path=relative_path,
        size_bytes=10,
        sha256=sha256,
        document_uuid=benchmark.benchmark_document_uuid(sha256, relative_path),
        physical_pages=physical_pages,
        subject=subject,
        subject_rule=subject,
        subject_status="matched",
        inventory_status="ready",
        page_count_status="available",
    )


def _success_result(item: benchmark.InventoryItem, elapsed: float = 1.0) -> dict[str, object]:
    return benchmark._success_result(
        item,
        MineruRunResult(
            output_dir=Path("raw"),
            version="3.4.5",
            backend="pipeline",
            elapsed_seconds=elapsed,
            stdout="",
            stderr="",
        ),
        AdapterStats(
            pages=1,
            input_items=1,
            canonical_blocks=1,
            title_blocks=0,
            text_blocks=1,
            tables=0,
            formulas=0,
            images=0,
            skipped_auxiliary=0,
            unsupported_items=0,
        ),
        elapsed_seconds=elapsed + 0.1,
        raw_ref=None,
    )


def test_inventory_is_deterministic_and_groups_duplicates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    shared = b"%PDF-shared"
    _write_pdf(corpus / "a" / "数学教材.pdf", shared)
    _write_pdf(corpus / "b" / "化学教材.pdf", shared)
    _write_pdf(corpus / "unknown.pdf", b"%PDF-unknown")

    inventory = benchmark.inventory_corpus(corpus, workspace)
    repeated = benchmark.inventory_corpus(corpus, tmp_path / "workspace-again")

    assert [item.relative_path for item in inventory.items] == [
        "a/数学教材.pdf",
        "b/化学教材.pdf",
        "unknown.pdf",
    ]
    assert [item.benchmark_item_id for item in inventory.items] == [
        item.benchmark_item_id for item in repeated.items
    ]
    assert inventory.items[0].subject == "数学"
    assert inventory.items[1].subject == "化学"
    assert inventory.items[2].subject == benchmark.UNKNOWN_SUBJECT
    assert inventory.items[0].document_uuid == inventory.items[1].document_uuid
    assert inventory.items[0].benchmark_item_id != inventory.items[1].benchmark_item_id
    assert inventory.summary["pdfs"] == 3
    assert len(inventory.summary["duplicate_sha256_groups"]) == 1
    assert str(corpus) not in (workspace / "corpus-manifest.jsonl").read_text(encoding="utf-8")


def test_subject_rules_and_benchmark_identity_are_explicit() -> None:
    assert benchmark.classify_subject("数学/教材.pdf") == ("数学", "数学", "matched")
    assert benchmark.classify_subject("数学/物理教材.pdf")[0] == benchmark.UNKNOWN_SUBJECT
    assert benchmark.classify_subject("unknown/教材.pdf") == (
        benchmark.UNKNOWN_SUBJECT,
        None,
        "unclassified",
    )
    assert benchmark.benchmark_item_id("a.pdf", "a" * 64) == benchmark.benchmark_item_id(
        "a.pdf", "a" * 64
    )
    assert benchmark.benchmark_item_id("a.pdf", "a" * 64) != benchmark.benchmark_item_id(
        "b.pdf", "a" * 64
    )


def test_checkpoint_resume_skips_successful_items_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf", b"%PDF-a")
    _write_pdf(corpus / "b.pdf", b"%PDF-b")
    workspace = tmp_path / "workspace"
    settings = _settings(tmp_path / "data")
    monkeypatch.setattr(benchmark, "get_mineru_version", lambda _: "3.4.5")
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark, "_gpu_metadata", lambda: {"name": None, "driver_version": None})
    calls: list[str] = []

    def interrupt_on_second(item: benchmark.InventoryItem, **_: object) -> dict[str, object]:
        calls.append(item.basename)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _success_result(item)

    monkeypatch.setattr(benchmark, "_process_item", interrupt_on_second)
    config = benchmark.BenchmarkConfig(corpus_root=corpus, workspace=workspace)

    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(config, settings)

    checkpoint_lines = (workspace / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 1

    calls.clear()
    outcome = benchmark.run_benchmark(
        benchmark.BenchmarkConfig(corpus_root=corpus, workspace=workspace, resume=True),
        settings,
    )

    assert calls == ["b.pdf"]
    assert outcome.aggregate["benchmark"]["all_unique_pdfs_terminal"] is True
    assert json.loads((workspace / "run.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_resume_retries_failed_items_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "a.pdf")
    workspace = tmp_path / "workspace"
    settings = _settings(tmp_path / "data")
    monkeypatch.setattr(benchmark, "get_mineru_version", lambda _: "3.4.5")
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark, "_gpu_metadata", lambda: {"name": None, "driver_version": None})
    calls: list[str] = []
    fail_once = True

    def failed_then_success(item: benchmark.InventoryItem, **kwargs: object) -> dict[str, object]:
        nonlocal fail_once
        calls.append(item.basename)
        if fail_once:
            fail_once = False
            return benchmark._failure_result(
                item,
                MineruRunnerError("MinerU exited with code 9"),
                elapsed_seconds=0.2,
                corpus_root=kwargs["corpus_root"],
                workspace=kwargs["workspace"],
                source_path=None,
                raw_ref=None,
            )
        return _success_result(item)

    monkeypatch.setattr(benchmark, "_process_item", failed_then_success)
    benchmark.run_benchmark(
        benchmark.BenchmarkConfig(corpus_root=corpus, workspace=workspace),
        settings,
    )
    calls.clear()
    benchmark.run_benchmark(
        benchmark.BenchmarkConfig(corpus_root=corpus, workspace=workspace, resume=True),
        settings,
    )
    assert calls == []

    outcome = benchmark.run_benchmark(
        benchmark.BenchmarkConfig(
            corpus_root=corpus,
            workspace=workspace,
            resume=True,
            retry_failed=True,
        ),
        settings,
    )
    assert calls == ["a.pdf"]
    assert outcome.aggregate["benchmark"]["successful_unique_pdfs"] == 1


def test_corrupt_result_tail_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    valid = {"benchmark_item_id": "item-a", "status": "success"}
    path.write_text(json.dumps(valid) + "\n{", encoding="utf-8")

    loaded = benchmark._load_results(path)

    assert loaded == {"item-a": valid}
    assert path.read_text(encoding="utf-8").strip() == json.dumps(
        valid, separators=(",", ":"), sort_keys=True
    )


def test_failure_and_warning_taxonomies_are_stable() -> None:
    assert benchmark.classify_failure(MineruRunnerError("MinerU timed out after 1 seconds")) == (
        "mineru_timeout"
    )
    assert benchmark.classify_failure(MineruRunnerError("content_list artifact missing")) == (
        "content_list_missing"
    )
    assert benchmark.classify_failure(MineruAdapterError("unsafe image asset path")) == (
        "unsafe_asset"
    )
    assert benchmark.classify_failure(MineruAdapterError("invalid bbox")) == "invalid_bbox"
    assert benchmark.classify_failure(RuntimeError("unexpected")) == "unknown_internal_error"
    assert benchmark.warning_categories(
        (
            "bbox_clamped:item=1",
            "item 2: unsupported or empty MinerU block type 'code'",
            "item 3: unsupported or empty MinerU block type 'text'",
            "item 4: equation text_format was not latex",
            "item 5: non-integer text_level mapped as ordinary text",
            "table_asset_only:item=6",
            "table_missing_content:item=7",
        )
    ) == {
        "bbox_clamped": 1,
        "empty_content": 1,
        "invalid_text_level": 1,
        "non_latex_equation": 1,
        "table_asset_only": 1,
        "table_missing_content": 1,
        "unsupported_type": 1,
    }


def test_table_degradation_stats_are_persisted_and_aggregated(tmp_path: Path) -> None:
    item = _manual_item("item-table", "数学/table.pdf", "a" * 64)
    record = benchmark._success_result(
        item,
        MineruRunResult(
            output_dir=Path("raw"),
            version="3.4.5",
            backend="pipeline",
            elapsed_seconds=1.0,
            stdout="",
            stderr="",
        ),
        AdapterStats(
            pages=1,
            input_items=3,
            canonical_blocks=1,
            title_blocks=0,
            text_blocks=1,
            tables=0,
            formulas=0,
            images=0,
            skipped_auxiliary=0,
            unsupported_items=2,
            table_asset_only=1,
            table_missing_content=1,
            warnings=("table_asset_only:item=0", "table_missing_content:item=1"),
        ),
        elapsed_seconds=1.1,
        raw_ref="raw/item-table",
    )

    assert record["table_asset_only"] == 1
    assert record["table_missing_content"] == 1
    assert record["warning_categories"] == {
        "table_asset_only": 1,
        "table_missing_content": 1,
    }

    aggregate = benchmark.aggregate_results(
        (item,), (item,), {item.benchmark_item_id: record}, tmp_path
    )

    assert aggregate["signals"]["table_asset_only"] == 1
    assert aggregate["signals"]["table_missing_content"] == 1
    assert aggregate["signals"]["warning_categories"] == {
        "table_asset_only": 1,
        "table_missing_content": 1,
    }


def test_aggregate_uses_unique_contents_and_linear_percentiles(tmp_path: Path) -> None:
    sha_values = [f"{index:064x}" for index in range(4)]
    items = (
        _manual_item("item-a", "a.pdf", sha_values[0]),
        _manual_item("item-b", "b.pdf", sha_values[1]),
        _manual_item("item-b-duplicate", "z-b-copy.pdf", sha_values[1], subject="化学"),
        _manual_item("item-c", "c.pdf", sha_values[2]),
        _manual_item("item-d", "d.pdf", sha_values[3]),
    )
    results = {
        item.benchmark_item_id: _success_result(item, elapsed=index + 1)
        for index, item in enumerate((items[0], items[1], items[3], items[4]))
    }
    results[items[2].benchmark_item_id] = {
        **benchmark._base_result(items[2], "skipped_duplicate"),
        "duplicate_of": items[1].benchmark_item_id,
    }

    aggregate = benchmark.aggregate_results(items, items, results, tmp_path)

    assert aggregate["corpus"]["unique_pdf_contents"] == 4
    assert aggregate["corpus"]["duplicate_content_entries"] == 1
    assert aggregate["benchmark"]["successful_unique_pdfs"] == 4
    assert aggregate["performance"]["p50_mineru_latency_seconds"] == 2.5
    assert aggregate["performance"]["p95_mineru_latency_seconds"] == 3.85
    assert aggregate["performance"]["aggregate_pages_per_second"] == 0.4


def test_process_item_retains_canonical_and_applies_raw_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    source = corpus / "数学教材.pdf"
    _write_pdf(source)
    workspace = tmp_path / "workspace"
    item = benchmark.inventory_corpus(corpus, tmp_path / "inventory").items[0]
    settings = _settings(tmp_path / "data")
    canonical_document = CanonicalDocument(
        document_id=item.document_uuid,
        pages=[
            Page(
                page_number=1,
                blocks=[TextBlock(block_id="p1-b1", reading_order=0, text="text")],
            )
        ],
    )
    adapted = AdaptedCanonicalDocument(
        document=canonical_document,
        stats=AdapterStats(
            pages=1,
            input_items=1,
            canonical_blocks=1,
            title_blocks=0,
            text_blocks=1,
            tables=0,
            formulas=0,
            images=0,
            skipped_auxiliary=0,
            unsupported_items=0,
        ),
    )

    def fake_run(_source: Path, output_dir: Path, *_: object, **__: object) -> MineruRunResult:
        return MineruRunResult(output_dir, "3.4.5", "pipeline", 1.0, "", "")

    monkeypatch.setattr(benchmark, "run_mineru", fake_run)
    monkeypatch.setattr(
        benchmark, "find_content_list", lambda output_dir: output_dir / "content.json"
    )
    monkeypatch.setattr(benchmark, "infer_page_count", lambda _: 1)
    monkeypatch.setattr(benchmark, "adapt_content_list", lambda *_args, **_kwargs: adapted)

    result = benchmark._process_item(
        item,
        corpus_root=corpus.resolve(),
        workspace=workspace,
        settings=settings,
        raw_retention="none",
    )

    assert result["status"] == "success"
    assert (workspace / "canonical" / f"{item.benchmark_item_id}.json").is_file()
    assert not (workspace / "raw").exists()
    assert not list((workspace / ".staging").glob("*"))

    second_workspace = tmp_path / "workspace-all"
    result = benchmark._process_item(
        item,
        corpus_root=corpus.resolve(),
        workspace=second_workspace,
        settings=settings,
        raw_retention="all",
    )
    assert result["raw_ref"] == f"raw/{item.benchmark_item_id}"
    assert (second_workspace / "raw" / item.benchmark_item_id).is_dir()


def test_process_item_preserves_failed_raw_output_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    _write_pdf(corpus / "数学教材.pdf")
    item = benchmark.inventory_corpus(corpus, tmp_path / "inventory").items[0]
    workspace = tmp_path / "workspace"

    def fail_run(*_: object, **__: object) -> MineruRunResult:
        raise MineruRunnerError("MinerU exited with code 3")

    monkeypatch.setattr(benchmark, "run_mineru", fail_run)
    result = benchmark._process_item(
        item,
        corpus_root=corpus.resolve(),
        workspace=workspace,
        settings=_settings(tmp_path / "data"),
        raw_retention="failures",
    )

    assert result["status"] == "failed"
    assert result["raw_ref"] == f"failures/{item.benchmark_item_id}"
    assert (workspace / "failures" / item.benchmark_item_id).is_dir()
