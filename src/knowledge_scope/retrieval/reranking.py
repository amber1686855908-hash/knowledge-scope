"""Local cross-encoder reranking for dense KnowledgeScope results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Protocol

from knowledge_scope.shared.config import Settings

from .qdrant import RetrievedChunk

RerankerDType = Literal["float16", "float32", "bfloat16"]
RerankerPromptMode = Literal["qwen_default_query", "raw_query_passage"]

RERANKER_MODEL_KEYS = (
    "qwen3-reranker-0.6b",
    "bge-reranker-v2-m3",
    "gte-multilingual-reranker-base",
)
DEFAULT_RERANKER_MODEL_KEY = "qwen3-reranker-0.6b"
QWEN_RERANKER_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class RerankerError(RuntimeError):
    """Raised when a local reranker cannot score a request safely."""


@dataclass(frozen=True, slots=True)
class RerankerModelSpec:
    """Model identity and the input convention documented by its model card."""

    key: str
    model_id: str
    prompt_mode: RerankerPromptMode
    prompt_detail: str
    official_reference: str
    trust_remote_code: bool = False


RERANKER_MODEL_SPECS: dict[str, RerankerModelSpec] = {
    "qwen3-reranker-0.6b": RerankerModelSpec(
        key="qwen3-reranker-0.6b",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        prompt_mode="qwen_default_query",
        prompt_detail=(
            "SentenceTransformers CrossEncoder model-card default prompt=query; "
            f"instruction={QWEN_RERANKER_DEFAULT_INSTRUCTION!r}"
        ),
        official_reference="https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
    ),
    "bge-reranker-v2-m3": RerankerModelSpec(
        key="bge-reranker-v2-m3",
        model_id="BAAI/bge-reranker-v2-m3",
        prompt_mode="raw_query_passage",
        prompt_detail="raw query/passage pair; model-card Transformers convention",
        official_reference="https://huggingface.co/BAAI/bge-reranker-v2-m3",
    ),
    "gte-multilingual-reranker-base": RerankerModelSpec(
        key="gte-multilingual-reranker-base",
        model_id="Alibaba-NLP/gte-multilingual-reranker-base",
        prompt_mode="raw_query_passage",
        prompt_detail="raw query/passage pair; model-card Transformers convention",
        official_reference="https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base",
        trust_remote_code=True,
    ),
}


class RerankerProtocol(Protocol):
    """Minimal scoring contract shared by local adapters and test doubles."""

    model_id: str

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score for each query/passage pair."""


def _torch_dtype(torch: Any, dtype: RerankerDType) -> Any:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _model_revision(model: Any) -> str | None:
    loaded_model = getattr(model, "model", model)
    config = getattr(loaded_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision) if revision else None


def _model_dtype(model: Any) -> str | None:
    parameters = getattr(getattr(model, "model", model), "parameters", None)
    if parameters is None:
        return None
    try:
        return str(next(parameters()).dtype).removeprefix("torch.")
    except (StopIteration, TypeError):
        return None


class LocalCrossEncoderReranker:
    """Lazy Sentence-Transformers adapter for the selected local reranker."""

    def __init__(
        self,
        spec: RerankerModelSpec,
        *,
        device: str = "cuda",
        dtype: RerankerDType = "float16",
        batch_size: int = 4,
        max_seq_length: int = 512,
        revision: str | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be at least one")
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.revision = revision
        self._model = model
        self._torch: Any | None = None
        self._inference_lock = Lock()

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def model_revision(self) -> str | None:
        return _model_revision(self._model) if self._model is not None else self.revision

    @property
    def actual_dtype(self) -> str | None:
        return _model_dtype(self._model) if self._model is not None else None

    @property
    def prompt_detail(self) -> str:
        return self.spec.prompt_detail

    def load(self) -> Any:
        """Load and return the local model, importing GPU dependencies lazily."""
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerError(
                "reranker dependencies are missing; run uv sync --group reranker-benchmark"
            ) from error

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RerankerError("CUDA reranker device was requested but is unavailable")

        model_kwargs: dict[str, Any] = {}
        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = _torch_dtype(torch, self.dtype)
        processor_kwargs: dict[str, Any] = {}
        if self.spec.prompt_mode == "qwen_default_query":
            processor_kwargs["padding_side"] = "left"
        constructor_kwargs: dict[str, Any] = {
            "model_name_or_path": self.spec.model_id,
            "device": self.device,
            "revision": self.revision,
            "trust_remote_code": self.spec.trust_remote_code,
            "model_kwargs": model_kwargs,
            "processor_kwargs": processor_kwargs,
            "max_length": self.max_seq_length,
        }
        if self.spec.prompt_mode == "qwen_default_query":
            constructor_kwargs["default_prompt_name"] = "query"
        try:
            self._model = CrossEncoder(**constructor_kwargs)
        except Exception as error:
            raise RerankerError(f"local reranker could not be loaded: {self.model_id}") from error
        self._torch = torch
        return self._model

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score query/passage pairs in input order with higher-is-more-relevant scores."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if not passages:
            return []
        with self._inference_lock:
            model = self.load()
            pairs = [(query, passage) for passage in passages]
            try:
                scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as error:
                raise RerankerError(f"local reranker scoring failed: {self.model_id}") from error
            if hasattr(scores, "detach"):
                values: Any = scores.detach().cpu().reshape(-1).tolist()
            elif hasattr(scores, "reshape"):
                values = scores.reshape(-1).tolist()
            else:
                values = scores if isinstance(scores, (list, tuple)) else [scores]
            try:
                result = [float(value) for value in values]
            except (TypeError, ValueError) as error:
                raise RerankerError("local reranker returned non-numeric scores") from error
            if len(result) != len(passages):
                raise RerankerError("local reranker score count does not match passage count")
            if any(not math.isfinite(value) for value in result):
                raise RerankerError("local reranker returned a non-finite score")
            return result


def _passage_text(chunk: RetrievedChunk) -> str:
    """Use stored text, retaining a small deterministic carrier for asset-only chunks."""
    if chunk.payload.text.strip():
        return chunk.payload.text
    section = " / ".join(chunk.payload.section_path)
    content_types = ", ".join(chunk.payload.content_types)
    return " ".join(part for part in (section, f"[{content_types}]") if part)


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """One dense result annotated with its reranker score and original rank."""

    chunk: RetrievedChunk
    dense_rank: int
    reranker_score: float


class RerankingService:
    """Rerank an existing dense result list without changing the candidate pool."""

    def __init__(self, reranker: RerankerProtocol) -> None:
        self.reranker = reranker

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int = 10,
    ) -> tuple[RerankedChunk, ...]:
        """Return the highest-scoring candidates, using dense rank as tie-breaker."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least one")
        if not candidates:
            return ()
        scores = self.reranker.score_pairs(query, [_passage_text(item) for item in candidates])
        ranked = sorted(
            (
                RerankedChunk(
                    chunk=item,
                    dense_rank=rank,
                    reranker_score=score,
                )
                for rank, (item, score) in enumerate(zip(candidates, scores, strict=True), start=1)
            ),
            key=lambda item: (-item.reranker_score, item.dense_rank),
        )
        return tuple(ranked[:limit])


def create_local_reranker(
    settings: Settings,
    *,
    model_key: str | None = None,
) -> LocalCrossEncoderReranker:
    """Create the configured local reranker without loading its model yet."""
    selected_key = model_key or settings.reranker_model_key
    try:
        spec = RERANKER_MODEL_SPECS[selected_key]
    except KeyError as error:
        raise ValueError(f"unknown reranker model key: {selected_key}") from error
    return LocalCrossEncoderReranker(
        spec,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        batch_size=settings.reranker_batch_size,
        max_seq_length=settings.reranker_max_seq_length,
        revision=settings.reranker_model_revision,
    )


__all__ = [
    "DEFAULT_RERANKER_MODEL_KEY",
    "RERANKER_MODEL_KEYS",
    "RERANKER_MODEL_SPECS",
    "LocalCrossEncoderReranker",
    "RerankedChunk",
    "RerankerDType",
    "RerankerError",
    "RerankerModelSpec",
    "RerankerProtocol",
    "RerankingService",
    "create_local_reranker",
]
