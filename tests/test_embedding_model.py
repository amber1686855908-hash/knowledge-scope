from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import QDRANT_VECTOR_DIMENSION
from knowledge_scope.shared.config import Settings


class _Encoded:
    def tolist(self) -> list[list[float]]:
        return [[0.0] * QDRANT_VECTOR_DIMENSION]


class _ConcurrentSentenceTransformer:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.max_active = 0

    def encode(self, _texts: list[str], **_kwargs: object) -> _Encoded:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep(0.01)
            return _Encoded()
        finally:
            with self._lock:
                self.active -= 1


def test_embedding_adapter_serializes_shared_model_inference() -> None:
    model = _ConcurrentSentenceTransformer()
    adapter = QwenEmbeddingModel(
        Settings(_env_file=None, embedding_device="cpu"),
        model=model,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: adapter.encode_query("问题"), range(4)))

    assert len(results) == 4
    assert all(len(vector) == QDRANT_VECTOR_DIMENSION for vector in results)
    assert model.max_active == 1
