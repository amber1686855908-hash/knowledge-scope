# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.evaluation.a4_5_retrieval_evaluation import (
    A45EvaluationError,
    A45GeneralQueryRecord,
    A45MultimodalDatasetManifest,
    A45MultimodalEvalItem,
    A45Protocol,
    A45RankedCandidate,
    A45StoreStateSnapshot,
    _branch_recovery_summary,
    _compare_rank_values,
    _compare_store_state,
    _ensure_store_state_unchanged,
    _is_substantive_fragment,
    _query_fragment,
    _store_state_snapshot,
    aggregate_metric_rows,
    compare_query_hits,
    evidence_metrics,
    evidence_metrics_for_candidates,
    load_frozen_store_manifest,
    load_multimodal_dataset,
    multimodal_dataset_fingerprint,
    multimodal_eval_id,
)
from knowledge_scope.retrieval.sparse import SparseIndexError, SparseIndexStore

KB_ID = UUID("3593a2ee-a326-5a0a-89a2-9a3012666c83")
DOCUMENT_ID = UUID("98cc9850-7b07-5917-b8ab-3278369198c4")
EVIDENCE_ID = "evidence_v1_" + "1" * 64
REPRESENTATION_ID = "representation_v1_" + "2" * 64
SOURCE_FINGERPRINT = "3" * 64
CANONICAL_FINGERPRINT = "4" * 64


def _item_data(*, query: str = "图中说明了什么现象？") -> dict[str, object]:
    source_block_ids = ["p1-b1"]
    asset_refs = ["image-1.png"]
    return {
        "eval_id": multimodal_eval_id(
            query,
            "image",
            EVIDENCE_ID,
            KB_ID,
            DOCUMENT_ID,
            1,
            1,
            source_block_ids,
            asset_refs,
            source_fingerprint=SOURCE_FINGERPRINT,
            canonical_document_fingerprint=CANONICAL_FINGERPRINT,
        ),
        "query": query,
        "subject": "物理",
        "query_type": "explanation",
        "modality": "image",
        "evidence_id": EVIDENCE_ID,
        "knowledge_base_id": KB_ID,
        "document_id": DOCUMENT_ID,
        "page_start": 1,
        "page_end": 1,
        "source_block_ids": source_block_ids,
        "asset_refs": asset_refs,
        "split": "dev",
        "rationale": "使用同一 Evidence 的受限文本表示生成，待人工核验。",
        "source_representation_type": "context",
        "source_representation_id": REPRESENTATION_ID,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "canonical_document_fingerprint": CANONICAL_FINGERPRINT,
    }


def test_a45_protocol_is_frozen_and_records_a35_rrf() -> None:
    protocol = A45Protocol()

    assert protocol.embedding_batch_size == 4
    assert protocol.reranker_batch_size == 8
    assert protocol.a35_rrf_k == 60
    assert protocol.fingerprint == A45Protocol().fingerprint

    with pytest.raises(ValidationError):
        A45Protocol(rrf_k=60)


def test_a45_protocol_rejects_material_profile_overrides() -> None:
    with pytest.raises(ValidationError, match="frozen profile mismatch"):
        A45Protocol(embedding_max_seq_length=1024)
    with pytest.raises(ValidationError, match="frozen profile mismatch"):
        A45Protocol(reranker_revision="different-revision")


def test_multimodal_item_identity_is_deterministic_and_extra_fields_are_forbidden() -> None:
    first = A45MultimodalEvalItem.model_validate(_item_data())
    second = A45MultimodalEvalItem.model_validate(_item_data())

    assert first.eval_id == second.eval_id
    with pytest.raises(ValidationError):
        A45MultimodalEvalItem.model_validate({**_item_data(), "unexpected": True})
    with pytest.raises(ValidationError):
        A45MultimodalEvalItem.model_validate(
            {
                **_item_data(),
                "source_block_ids": ["p1-b1", "p1-b1"],
            }
        )


def test_multimodal_eval_id_changes_when_query_changes() -> None:
    first = multimodal_eval_id(
        "问题一？", "image", EVIDENCE_ID, KB_ID, DOCUMENT_ID, 1, 1, ["p1-b1"], ["image-1"]
    )
    second = multimodal_eval_id(
        "问题二？", "image", EVIDENCE_ID, KB_ID, DOCUMENT_ID, 1, 1, ["p1-b1"], ["image-1"]
    )

    assert first != second


def test_multimodal_dataset_loader_rejects_non_frozen_size(tmp_path) -> None:
    item = A45MultimodalEvalItem.model_validate(_item_data())
    dataset_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    dataset_path.write_text(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = A45MultimodalDatasetManifest(
        item_count=1,
        dev_count=1,
        test_count=0,
        modality_counts={"image": 1},
        subject_counts={"物理": 1},
        dataset_fingerprint=multimodal_dataset_fingerprint([item]),
        file_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        knowledge_base_ids=[KB_ID],
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(A45EvaluationError, match="exactly 72"):
        load_multimodal_dataset(dataset_path, manifest_path)


def test_multimodal_loader_rejects_self_consistent_replacement(tmp_path) -> None:
    dataset_source = Path("docs/benchmarks/a4-5-multimodal-eval-v2.jsonl")
    manifest_source = Path("docs/benchmarks/a4-5-multimodal-eval-v2-manifest.json")
    rows = [
        json.loads(line)
        for line in dataset_source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["query"] = "一个独立替换问题？"
    rows[0]["eval_id"] = multimodal_eval_id(
        rows[0]["query"],
        rows[0]["modality"],
        rows[0]["evidence_id"],
        UUID(rows[0]["knowledge_base_id"]),
        UUID(rows[0]["document_id"]),
        rows[0]["page_start"],
        rows[0]["page_end"],
        rows[0]["source_block_ids"],
        rows[0]["asset_refs"],
        source_fingerprint=rows[0]["source_fingerprint"],
        canonical_document_fingerprint=rows[0]["canonical_document_fingerprint"],
    )
    items = [A45MultimodalEvalItem.model_validate(row) for row in rows]
    dataset_bytes = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in items
    ).encode("utf-8")
    dataset_path = tmp_path / "replacement.jsonl"
    dataset_path.write_bytes(dataset_bytes)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest["dataset_fingerprint"] = multimodal_dataset_fingerprint(items)
    manifest["file_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
    manifest_path = tmp_path / "replacement-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(A45EvaluationError, match="frozen repository artifact identity"):
        load_multimodal_dataset(dataset_path, manifest_path)


def test_frozen_store_manifest_rejects_tampering(tmp_path) -> None:
    manifest = json.loads(
        Path("docs/benchmarks/a4-5-frozen-store-manifest.json").read_text(encoding="utf-8")
    )
    manifest["a21"]["item_count"] = 107
    path = tmp_path / "frozen-store-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(A45EvaluationError, match="manifest identity"):
        load_frozen_store_manifest(path)


def _frozen_store_state_snapshot() -> A45StoreStateSnapshot:
    return _store_state_snapshot(load_frozen_store_manifest())


def test_store_state_audit_reports_unchanged_stores() -> None:
    snapshot = _frozen_store_state_snapshot()

    audit = _compare_store_state(snapshot, snapshot)

    assert audit.stores_mutated is False
    assert audit.changed_stores == []
    assert audit.changed_components == []
    _ensure_store_state_unchanged(audit)


def test_store_state_audit_rejects_changed_dense_store() -> None:
    before = _frozen_store_state_snapshot()
    after = before.model_copy(
        update={
            "dense": before.dense.model_copy(update={"point_count": before.dense.point_count + 1})
        }
    )

    audit = _compare_store_state(before, after)

    assert audit.stores_mutated is True
    assert audit.changed_stores == ["dense"]
    with pytest.raises(A45EvaluationError, match="dense"):
        _ensure_store_state_unchanged(audit)


@pytest.mark.parametrize("store_name", ["sparse", "representation", "graph"])
def test_store_state_audit_identifies_changed_store(store_name: str) -> None:
    before = _frozen_store_state_snapshot()
    store = getattr(before, store_name)
    field_name = {
        "sparse": "active_generation_id",
        "representation": "point_count",
        "graph": "snapshot_fingerprint",
    }[store_name]
    current_value = getattr(store, field_name)
    if field_name.endswith("fingerprint"):
        changed_value = ("0" if current_value[0] != "0" else "1") + current_value[1:]
    elif isinstance(current_value, str):
        changed_value = f"{current_value}-changed"
    else:
        changed_value = current_value + 1
    after = before.model_copy(
        update={store_name: store.model_copy(update={field_name: changed_value})}
    )

    audit = _compare_store_state(before, after)

    assert audit.stores_mutated is True
    assert audit.changed_stores == [store_name]
    with pytest.raises(A45EvaluationError, match=store_name):
        _ensure_store_state_unchanged(audit)


def test_store_state_snapshot_requires_every_store() -> None:
    data = _frozen_store_state_snapshot().model_dump(mode="json")
    del data["sparse"]

    with pytest.raises(ValidationError, match="sparse"):
        A45StoreStateSnapshot.model_validate(data)


def test_multimodal_quality_gate_ignores_labels_and_picks_context_sentence() -> None:
    assert not _is_substantive_fragment("学习提示")
    assert not _is_substantive_fragment("人民教育出版社")
    assert _query_fragment("章节: 学习提示; 邻近文本: 二氧化碳在水中的溶解度随温度变化") == (
        "二氧化碳在水中的溶解度随温度变化"
    )
    assert _query_fragment("章节: 资料分析; 邻近文本: 2024") is None
    assert "…" not in (_query_fragment("二氧化碳的性质与用途" * 20) or "")


def test_evidence_metrics_and_query_hit_comparison_are_query_level() -> None:
    metrics = evidence_metrics(["other", "gold", "gold"], ["gold"])
    assert metrics["evidence_hit@1"] == 0.0
    assert metrics["evidence_hit@3"] == 1.0
    assert metrics["evidence_mrr"] == 0.5
    assert metrics["evidence_recall@3"] == 1.0

    baseline = [{"hit@1": 0.0}, {"hit@1": 1.0}, {"hit@1": 1.0}]
    candidate = [{"hit@1": 1.0}, {"hit@1": 1.0}, {"hit@1": 0.0}]
    assert compare_query_hits(baseline, candidate, k=1) == {
        "improved": 1,
        "unchanged": 1,
        "regressed": 1,
    }


def test_unified_evidence_metrics_keep_actual_mixed_candidate_positions() -> None:
    chunk = A45RankedCandidate(
        candidate_id="chunk-1",
        candidate_kind="chunk",
        rank=1,
        score=0.9,
        source="dense",
        knowledge_base_id=KB_ID,
        document_id=DOCUMENT_ID,
        chunk_id="chunk-1",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        section_path=[],
        asset_refs=[],
        representation_ids=[],
        branch_ranks={"dense": 1},
    )
    evidence = A45RankedCandidate(
        candidate_id="evidence-1",
        candidate_kind="evidence",
        rank=2,
        score=0.8,
        source="multimodal",
        knowledge_base_id=KB_ID,
        document_id=DOCUMENT_ID,
        evidence_id=EVIDENCE_ID,
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        section_path=[],
        asset_refs=[],
        modality="image",
        representation_ids=[REPRESENTATION_ID],
        branch_ranks={"multimodal": 1},
    )

    metrics = evidence_metrics_for_candidates([chunk, evidence], [EVIDENCE_ID])

    assert metrics["evidence_hit@1"] == 0.0
    assert metrics["evidence_hit@3"] == 1.0
    assert metrics["evidence_mrr"] == 0.5


def test_read_only_sparse_access_does_not_create_missing_index(tmp_path) -> None:
    missing = tmp_path / "missing" / "sparse.sqlite3"

    with pytest.raises(SparseIndexError, match="does not exist"):
        SparseIndexStore(missing, read_only=True)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_metric_aggregation_does_not_synthesize_missing_fields() -> None:
    assert aggregate_metric_rows([{"hit@1": 1.0}, {"hit@1": 0.0}]) == {"hit@1": 0.5}
    assert aggregate_metric_rows([]) == {}


def test_branch_recovery_summary_separates_candidate_recovery_from_final_ranking() -> None:
    record = A45GeneralQueryRecord(
        eval_id="a2-1-test",
        split="dev",
        query="问题？",
        subject="物理",
        query_type="factual",
        gold_relevant_chunk_ids=["chunk-gold"],
        gold_source_blocks=["document:p1-b1"],
        branch_gold_recovery={
            "dense": {
                "gold_chunk_ids": ["chunk-gold"],
                "gold_source_blocks": ["document:p1-b1"],
            },
            "sparse": {"gold_chunk_ids": [], "gold_source_blocks": ["document:p1-b1"]},
            "graph": {"gold_chunk_ids": [], "gold_source_blocks": []},
            "multimodal": {"gold_chunk_ids": [], "gold_source_blocks": []},
        },
        systems={},
    )

    summary = _branch_recovery_summary([record])

    assert summary["query_count"] == 1
    assert summary["gold_chunk_recovery"]["branch_query_counts"] == {
        "dense": 1,
        "sparse": 0,
        "graph": 0,
        "multimodal": 0,
    }
    assert summary["gold_source_block_recovery"]["exclusive_query_counts"]["multiple"] == 1


def test_first_relevant_rank_comparison_is_lower_is_better() -> None:
    assert _compare_rank_values(None, 3) == "improved"
    assert _compare_rank_values(3, 1) == "improved"
    assert _compare_rank_values(1, 3) == "regressed"
    assert _compare_rank_values(2, None) == "regressed"
    assert _compare_rank_values(2, 2) == "unchanged"
