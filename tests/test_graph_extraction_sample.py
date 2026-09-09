from __future__ import annotations

# ruff: noqa: RUF001
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid5

import pytest

from knowledge_scope.chunking.models import Chunk
from knowledge_scope.evaluation.graph_extraction_sample import (
    SampleChunk,
    run_sample_evaluation,
    select_sample_chunks,
)
from knowledge_scope.graph.neo4j import GraphStoreError
from knowledge_scope.llm.schemas import LLMRequest, LLMResult
from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock
from knowledge_scope.shared.config import Settings

SUBJECTS = ("历史", "地理", "政治", "语文", "数学", "物理", "化学", "生物", "技术")


class _Gateway:
    async def complete(self, _request: LLMRequest) -> LLMResult:
        return LLMResult(
            text=json.dumps(
                {
                    "entities": [{"name": "水", "entity_type": "物质", "aliases": []}],
                    "relations": [],
                },
                ensure_ascii=False,
            ),
            provider="fake",
            model="fake",
            input_tokens=10,
            output_tokens=2,
            latency_ms=1,
        )


class _ResponseGateway:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(self, _request: LLMRequest) -> LLMResult:
        return LLMResult(
            text=self.response,
            provider="fake",
            model="fake",
            input_tokens=10,
            output_tokens=2,
            latency_ms=1,
        )


class _FailingStore:
    def upsert_extraction(self, _entities: object, _relations: object) -> None:
        raise GraphStoreError("neo4j unavailable")


def _write_corpus(root: Path, manifest_path: Path) -> None:
    lines: list[str] = []
    for subject_index, subject in enumerate(SUBJECTS):
        for document_index in range(2):
            item_id = f"item-{subject_index}-{document_index}"
            document_id = uuid5(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"), item_id)
            document = CanonicalDocument(
                document_id=document_id,
                pages=[
                    Page(
                        page_number=1,
                        blocks=[
                            TextBlock(
                                block_id="p1-b1",
                                reading_order=0,
                                text=(
                                    "水由氢和氧组成，是教材样本中的一段足够长的说明文字，"
                                    "用于验证分主题抽样和运行时报告写入。"
                                ),
                            )
                        ],
                    )
                ],
            )
            (root / f"{item_id}.json").write_text(document.model_dump_json(), encoding="utf-8")
            lines.append(
                json.dumps(
                    {
                        "inventory_status": "ready",
                        "benchmark_item_id": item_id,
                        "benchmark_document_uuid": str(document_id),
                        "subject": subject,
                        "relative_path": f"{subject}/{item_id}.pdf",
                    },
                    ensure_ascii=False,
                )
            )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sample_selection_is_stable_and_stratified(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"
    _write_corpus(canonical_root, manifest_path)

    samples = select_sample_chunks(canonical_root, manifest_path, sample_per_subject=2)

    assert len(samples) == 18
    assert [sample.subject for sample in samples] == sorted(sample.subject for sample in samples)
    assert {sample.subject for sample in samples} == set(SUBJECTS)
    assert all(sample.chunk.text for sample in samples)

    development = select_sample_chunks(canonical_root, manifest_path, sample_per_subject=1)
    holdout = select_sample_chunks(
        canonical_root,
        manifest_path,
        sample_per_subject=1,
        sample_offset=1,
    )
    assert len(holdout) == len(development) == len(SUBJECTS)
    assert not {sample.chunk.chunk_id for sample in development} & {
        sample.chunk.chunk_id for sample in holdout
    }


@pytest.mark.anyio
async def test_sample_runner_writes_bounded_runtime_summary(tmp_path: Path) -> None:
    text = "水由氢和氧组成。" + "这是运行时样本的说明文字。" * 60
    chunk = SampleChunk(
        subject="化学",
        benchmark_item_id="item-chemical",
        config_fingerprint="a" * 64,
        chunk=Chunk(
            chunk_id="chunk-chemical",
            document_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            ordinal=0,
            text=text,
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1"],
            section_path=["样本"],
            content_types=["text"],
        ),
    )
    output_dir = tmp_path / "runtime"
    summary = await run_sample_evaluation(
        [chunk],
        gateway=_Gateway(),
        settings=Settings(
            _env_file=None,
            environment="test",
            llm_input_cost_per_1k_tokens=Decimal("0.12"),
            llm_output_cost_per_1k_tokens=Decimal("0.34"),
        ),
        output_dir=output_dir,
    )

    assert summary["attempted_chunks"] == 1
    assert summary["status_counts"] == {"accepted": 1}
    record = json.loads((output_dir / "sample.jsonl").read_text(encoding="utf-8"))
    assert len(record["text_excerpt"]) <= 281
    assert text not in record["text_excerpt"]
    assert record["grounding"]["decision"] == "accepted"
    assert record["attempts"][0]["category"] == "valid_extraction"
    assert record["estimated_cost"] == "0.00188000"
    summary_record = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_record["entities_produced"] == 1
    assert summary_record["first_attempt_structured_success"] == 1
    assert summary_record["provider_calls"] == 1
    sample_content = (output_dir / "sample.jsonl").read_bytes()
    assert summary_record["record_count"] == 1
    assert summary_record["sample_sha256"] == hashlib.sha256(sample_content).hexdigest()


@pytest.mark.anyio
async def test_sample_runner_preserves_extraction_observations_on_persistence_failure(
    tmp_path: Path,
) -> None:
    chunk = SampleChunk(
        subject="化学",
        benchmark_item_id="item-chemical",
        config_fingerprint="a" * 64,
        chunk=Chunk(
            chunk_id="chunk-chemical",
            document_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            ordinal=0,
            text="水由氢和氧组成。",
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1"],
            section_path=["样本"],
            content_types=["text"],
        ),
    )

    output_dir = tmp_path / "runtime"
    summary = await run_sample_evaluation(
        [chunk],
        gateway=_Gateway(),
        settings=Settings(_env_file=None, environment="test"),
        output_dir=output_dir,
        store=_FailingStore(),  # type: ignore[arg-type]
    )

    assert summary["status_counts"] == {"failed": 1}
    assert summary["provider_calls"] == 1
    assert summary["structured_parse_success"] == 1
    assert summary["input_tokens"] == 10
    assert summary["latency_ms"] == 1
    record = json.loads((output_dir / "sample.jsonl").read_text(encoding="utf-8"))
    assert record["error_category"] == "persistence"
    assert record["extraction_status"] == "accepted"
    assert record["attempts"][0]["category"] == "valid_extraction"
    assert record["input_tokens"] == 10
    assert len(record["entities"]) == 1


@pytest.mark.anyio
async def test_sample_summary_counts_schema_failures_and_provider_calls(tmp_path: Path) -> None:
    sample = SampleChunk(
        subject="化学",
        benchmark_item_id="item-chemical",
        config_fingerprint="a" * 64,
        chunk=Chunk(
            chunk_id="chunk-chemical",
            document_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            ordinal=0,
            text="水由氢和氧组成。",
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1"],
            section_path=["样本"],
            content_types=["text"],
        ),
    )

    summary = await run_sample_evaluation(
        [sample],
        gateway=_ResponseGateway('{"entities": [{"name": "水"}], "relations": []}'),
        settings=Settings(
            _env_file=None,
            environment="test",
            graph_extraction_max_parse_retries=0,
        ),
        output_dir=tmp_path / "runtime",
    )

    assert summary["structured_parse_success"] == 0
    assert summary["structured_parse_failure"] == 1
    assert summary["parse_failure_count"] == 0
    assert summary["schema_rejection_count"] == 1
    assert summary["provider_calls"] == 1
