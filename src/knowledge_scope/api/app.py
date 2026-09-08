"""FastAPI application for the current KnowledgeScope product surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_scope import __version__
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import QdrantVectorStore
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .documents import router as documents_router
from .knowledge_bases import router as knowledge_bases_router
from .retrieval import router as retrieval_router
from .schemas import HealthResponse, MetaResponse, QdrantHealthResponse

API_PREFIX: Final = "/api/v1"
CURRENT_PHASE: Final = "A2.3"
PROJECT_STATUS: Final = "foundation"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Dispose application-owned database and vector-store clients on shutdown."""
    yield
    await application.state.db_engine.dispose()
    application.state.vector_store.close()


def create_app(
    settings: Settings | None = None,
    *,
    database_engine: AsyncEngine | None = None,
    vector_store: QdrantVectorStore | None = None,
    embedding_model: QwenEmbeddingModel | None = None,
) -> FastAPI:
    """Create the API application with validated runtime settings."""
    runtime_settings = settings if settings is not None else get_settings()
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

    application.include_router(router)
    application.include_router(knowledge_bases_router, prefix=API_PREFIX)
    application.include_router(documents_router, prefix=API_PREFIX)
    application.include_router(retrieval_router, prefix=API_PREFIX)
    return application


app = create_app()
