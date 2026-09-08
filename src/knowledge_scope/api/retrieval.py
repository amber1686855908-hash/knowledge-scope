"""Minimal dense retrieval HTTP endpoint for developer and API consumers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from knowledge_scope.retrieval.embedding import EmbeddingModelError
from knowledge_scope.retrieval.qdrant import VectorStoreError
from knowledge_scope.retrieval.service import DenseRetrievalService, RetrievalError

from .schemas import RetrievalSearchItem, RetrievalSearchRequest, RetrievalSearchResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
def search(
    payload: RetrievalSearchRequest,
    request: Request,
) -> RetrievalSearchResponse:
    """Embed one query and return filtered ranked chunks from Qdrant."""
    service = DenseRetrievalService(
        request.app.state.vector_store,
        request.app.state.embedding_model,
    )
    try:
        result = service.search(
            payload.query,
            limit=payload.limit,
            knowledge_base_id=payload.knowledge_base_id,
            document_id=payload.document_id,
        )
    except (EmbeddingModelError, RetrievalError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from None
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from None

    return RetrievalSearchResponse(
        query=result.query,
        model=result.model_id,
        collection=result.collection_name,
        items=[
            RetrievalSearchItem(
                point_id=item.point_id,
                score=item.score,
                chunk_id=item.payload.chunk_id,
                document_id=item.payload.document_id,
                knowledge_base_id=item.payload.knowledge_base_id,
                page_start=item.payload.page_start,
                page_end=item.payload.page_end,
                source_block_ids=item.payload.source_block_ids,
                section_path=item.payload.section_path,
                content_types=item.payload.content_types,
                asset_refs=item.payload.asset_refs,
                text=item.payload.text,
            )
            for item in result.items
        ],
    )


__all__ = ["router"]
