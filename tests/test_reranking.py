from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from uuid import uuid4

import pytest

from knowledge_scope.evaluation.reranker_benchmark import RerankerBenchmarkProtocol
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import (
    RERANKER_MODEL_SPECS,
    LocalCrossEncoderReranker,
    RerankerError,
    RerankingService,
)


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.inputs: list[tuple[str, str]] | None = None

    def predict(self, inputs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
        self.inputs = inputs
        return self.scores


class _FakeReranker:
    model_id = "fake-reranker"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score_pairs(self, _query: str, _passages: list[str]) -> list[float]:
        return self.scores


class _ConcurrentCrossEncoder:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.max_active = 0

    def predict(self, inputs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep(0.01)
            return [1.0] * len(inputs)
        finally:
            with self._lock:
                self.active -= 1


def _chunk(chunk_id: str, text: str = "答案正文") -> RetrievedChunk:
    document_id = uuid4()
    return RetrievedChunk(
        point_id=uuid4(),
        score=0.5,
        payload=ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=chunk_id,
            document_id=document_id,
            page_start=1,
            page_end=1,
            source_block_ids=["p1-b1"],
            section_path=["章节"],
            content_types=["image"],
            asset_refs=["image-1"],
            text=text,
            chunking_config_fingerprint="a" * 64,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_model_revision="revision",
            embedding_config_fingerprint="b" * 64,
        ),
    )


def test_model_specs_keep_three_local_official_conventions() -> None:
    assert set(RERANKER_MODEL_SPECS) == {
        "qwen3-reranker-0.6b",
        "bge-reranker-v2-m3",
        "gte-multilingual-reranker-base",
    }
    assert RERANKER_MODEL_SPECS["qwen3-reranker-0.6b"].prompt_mode == "qwen_default_query"
    assert all(
        spec.official_reference.startswith("https://huggingface.co/")
        for spec in RERANKER_MODEL_SPECS.values()
    )


def test_local_cross_encoder_adapter_preserves_pair_order() -> None:
    model = _FakeCrossEncoder([1.0, -2.0])
    reranker = LocalCrossEncoderReranker(
        RERANKER_MODEL_SPECS["bge-reranker-v2-m3"],
        model=model,
    )

    assert reranker.score_pairs("查询", ["正文一", "正文二"]) == [1.0, -2.0]
    assert model.inputs == [("查询", "正文一"), ("查询", "正文二")]


def test_local_cross_encoder_rejects_invalid_scores() -> None:
    mismatch = LocalCrossEncoderReranker(
        RERANKER_MODEL_SPECS["bge-reranker-v2-m3"],
        model=_FakeCrossEncoder([1.0]),
    )
    with pytest.raises(RerankerError, match="score count"):
        mismatch.score_pairs("查询", ["正文一", "正文二"])

    non_finite = LocalCrossEncoderReranker(
        RERANKER_MODEL_SPECS["bge-reranker-v2-m3"],
        model=_FakeCrossEncoder([math.nan]),
    )
    with pytest.raises(RerankerError, match="non-finite"):
        non_finite.score_pairs("查询", ["正文一"])


def test_local_cross_encoder_serializes_shared_model_inference() -> None:
    model = _ConcurrentCrossEncoder()
    reranker = LocalCrossEncoderReranker(
        RERANKER_MODEL_SPECS["bge-reranker-v2-m3"],
        model=model,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: reranker.score_pairs("查询", ["正文"]), range(4)))

    assert results == [[1.0], [1.0], [1.0], [1.0]]
    assert model.max_active == 1


def test_reranking_service_sorts_by_score_and_uses_dense_rank_as_tiebreaker() -> None:
    service = RerankingService(_FakeReranker([0.4, 0.9, 0.9]))
    candidates = [_chunk("chunk-1"), _chunk("chunk-2"), _chunk("chunk-3")]

    ranked = service.rerank("查询", candidates, limit=2)

    assert [item.chunk.payload.chunk_id for item in ranked] == ["chunk-2", "chunk-3"]
    assert [item.dense_rank for item in ranked] == [2, 3]
    assert [item.reranker_score for item in ranked] == [0.9, 0.9]


def test_reranking_service_keeps_asset_only_carrier_text() -> None:
    model = _FakeCrossEncoder([1.0])
    service = RerankingService(
        LocalCrossEncoderReranker(RERANKER_MODEL_SPECS["bge-reranker-v2-m3"], model=model)
    )

    service.rerank("查询", [_chunk("asset-only", text="")])

    assert model.inputs == [("查询", "章节 [image]")]


def test_reranking_protocol_requires_sorted_unique_candidate_sizes() -> None:
    assert RerankerBenchmarkProtocol(candidate_sizes=(10, 20, 50)).candidate_sizes == (
        10,
        20,
        50,
    )
    with pytest.raises(ValueError, match="sorted unique"):
        RerankerBenchmarkProtocol(candidate_sizes=(20, 10))
    with pytest.raises(ValueError, match="sorted unique"):
        RerankerBenchmarkProtocol(candidate_sizes=(10, 10))
