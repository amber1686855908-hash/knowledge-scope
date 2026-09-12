"""Lazy local Qwen embedding adapter used by A2.3 indexing and search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from threading import Lock
from typing import Any

from knowledge_scope.shared.config import Settings

from .qdrant import QDRANT_VECTOR_DIMENSION, QWEN_EMBEDDING_MODEL_ID

EMBEDDING_POOLING_STRATEGY = "SentenceTransformer model-native pooling"


class EmbeddingModelError(RuntimeError):
    """Raised when the configured local embedding model cannot run."""


def embedding_config_fingerprint(settings: Settings) -> str:
    """Return the stable vector-generation configuration fingerprint."""
    payload = {
        "embedding_model": QWEN_EMBEDDING_MODEL_ID,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_dtype": settings.embedding_dtype,
        "embedding_max_seq_length": settings.embedding_max_seq_length,
        "query_prompt": "SentenceTransformers prompt_name=query",
        "normalization": "l2",
        "pooling": EMBEDDING_POOLING_STRATEGY,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class QwenEmbeddingModel:
    """Load Qwen3-Embedding-0.6B only when a real vector operation needs it."""

    def __init__(self, settings: Settings, *, model: Any | None = None) -> None:
        self.settings = settings
        self._model = model
        self._torch: Any | None = None
        self._inference_lock = Lock()

    @property
    def model_id(self) -> str:
        return QWEN_EMBEDDING_MODEL_ID

    @property
    def model_revision(self) -> str:
        return self.settings.embedding_model_revision

    @property
    def config_fingerprint(self) -> str:
        return embedding_config_fingerprint(self.settings)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingModelError(
                "embedding dependencies are missing; run uv sync --group embedding-benchmark"
            ) from error

        device = self.settings.embedding_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise EmbeddingModelError("CUDA embedding device was requested but is unavailable")
        dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.settings.embedding_dtype]
        model_kwargs: dict[str, Any] = {}
        if device.startswith("cuda"):
            model_kwargs["dtype"] = dtype
        try:
            self._model = SentenceTransformer(
                self.model_id,
                device=device,
                revision=self.model_revision,
                model_kwargs=model_kwargs,
                processor_kwargs={"padding_side": "left"},
            )
            self._model.max_seq_length = self.settings.embedding_max_seq_length
        except Exception as error:
            raise EmbeddingModelError(
                f"Qwen embedding model could not be loaded: {self.model_id}"
            ) from error
        self._torch = torch
        return self._model

    def _encode(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        with self._inference_lock:
            model = self._load()
            kwargs: dict[str, Any] = {
                "batch_size": self.settings.embedding_batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            }
            if query:
                kwargs["prompt_name"] = "query"
            try:
                encoded = model.encode(list(texts), **kwargs)
            except Exception as error:
                operation = "query" if query else "document"
                raise EmbeddingModelError(f"Qwen {operation} encoding failed") from error
            vectors = encoded.tolist()
            if len(vectors) != len(texts) or any(
                len(vector) != QDRANT_VECTOR_DIMENSION for vector in vectors
            ):
                raise EmbeddingModelError(
                    f"Qwen embedding dimension must be {QDRANT_VECTOR_DIMENSION}"
                )
            return [[float(value) for value in vector] for vector in vectors]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document chunks without the query-only prompt."""
        return self._encode(texts, query=False)

    def encode_query(self, query: str) -> list[float]:
        """Encode one query with the model's official ``prompt_name=query``."""
        if not query.strip():
            raise ValueError("query must not be blank")
        return self._encode([query], query=True)[0]


__all__ = [
    "EMBEDDING_POOLING_STRATEGY",
    "EmbeddingModelError",
    "QwenEmbeddingModel",
    "embedding_config_fingerprint",
]
