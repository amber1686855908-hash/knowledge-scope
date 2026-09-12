"""FastAPI application for the current KnowledgeScope product surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_scope import __version__
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.providers import create_llm_provider
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.rag.service import RAGService
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import QdrantVectorStore
from knowledge_scope.retrieval.representation_index import (
    QdrantRepresentationStore,
    validate_representation_collection_role,
)
from knowledge_scope.retrieval.reranking import RerankingService, create_local_reranker
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .documents import router as documents_router
from .knowledge_bases import router as knowledge_bases_router
from .rag import router as rag_router
from .retrieval import router as retrieval_router
from .schemas import HealthResponse, MetaResponse, Neo4jHealthResponse, QdrantHealthResponse

API_PREFIX: Final = "/api/v1"
CURRENT_PHASE: Final = "A3.1"
PROJECT_STATUS: Final = "foundation"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize the default RAG graph and dispose owned clients on shutdown."""
    provider = None
    try:
        if application.state.representation_store is None:
            application.state.representation_store = QdrantRepresentationStore(
                application.state.settings
            )
        if application.state.rag_service is None:
            settings: Settings = application.state.settings
            provider = create_llm_provider(settings)
            application.state.llm_provider = provider
            gateway = LLMGateway(
                provider,
                DatabaseUsageRecorder(application.state.db_session_factory),
                settings,
            )
            retrieval = DenseRetrievalService(
                application.state.vector_store,
                application.state.embedding_model,
            )
            reranking = RerankingService(
                create_local_reranker(settings, model_key="bge-reranker-v2-m3")
            )
            application.state.rag_service = RAGService(
                retrieval,
                reranking,
                gateway,
                settings,
            )
        yield
    finally:
        if provider is not None:
            await provider.aclose()
        await application.state.db_engine.dispose()
        application.state.vector_store.close()
        if application.state.representation_store is not None:
            application.state.representation_store.close()
        application.state.graph_store.close()


def create_app(
    settings: Settings | None = None,
    *,
    database_engine: AsyncEngine | None = None,
    vector_store: QdrantVectorStore | None = None,
    embedding_model: QwenEmbeddingModel | None = None,
    rag_service: RAGService | None = None,
    graph_store: Neo4jGraphStore | None = None,
    representation_store: QdrantRepresentationStore | None = None,
) -> FastAPI:
    """Create the API application with validated runtime settings."""
    runtime_settings = settings if settings is not None else get_settings()
    validate_representation_collection_role(runtime_settings)
    engine = (
        database_engine if database_engine is not None else create_database_engine(runtime_settings)
    )
    application = FastAPI(
        title="KnowledgeScope API",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.db_engine = engine
    application.state.db_session_factory = create_session_factory(engine)
    application.state.settings = runtime_settings
    application.state.vector_store = vector_store or QdrantVectorStore(runtime_settings)
    application.state.embedding_model = embedding_model or QwenEmbeddingModel(runtime_settings)
    application.state.rag_service = rag_service
    application.state.graph_store = graph_store or Neo4jGraphStore(runtime_settings)
    application.state.representation_store = representation_store
    application.state.llm_provider = None
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Accept", "Content-Type"],
    )

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        report = build_health_report(runtime_settings)
        return HealthResponse.model_validate({**report.as_dict(), "status": "ok"})

    @router.get("/meta", response_model=MetaResponse, tags=["system"])
    def meta() -> MetaResponse:
        report = build_health_report(runtime_settings)
        return MetaResponse(
            project_name=report.project_name,
            version=report.version,
            phase=CURRENT_PHASE,
            status=PROJECT_STATUS,
            config_status=report.config_status,
        )

    @router.get("/health/qdrant", response_model=QdrantHealthResponse, tags=["system"])
    def qdrant_health(request: Request) -> QdrantHealthResponse:
        readiness = request.app.state.vector_store.readiness()
        if readiness.status == "unavailable":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.error or "Qdrant is unavailable",
            )
        return QdrantHealthResponse.model_validate(readiness.model_dump())

    @router.get("/health/neo4j", response_model=Neo4jHealthResponse, tags=["system"])
    def neo4j_health(request: Request) -> Neo4jHealthResponse:
        readiness = request.app.state.graph_store.readiness()
        if readiness.status == "unavailable":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.error or "Neo4j is unavailable",
            )
        return Neo4jHealthResponse.model_validate(readiness.model_dump())

    application.include_router(router)
    application.include_router(knowledge_bases_router, prefix=API_PREFIX)
    application.include_router(documents_router, prefix=API_PREFIX)
    application.include_router(retrieval_router, prefix=API_PREFIX)
    application.include_router(rag_router, prefix=API_PREFIX)
    return application


app = create_app()
