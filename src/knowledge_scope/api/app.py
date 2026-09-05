"""FastAPI application for the currently implemented project foundation."""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge_scope import __version__
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.config import Settings

from .schemas import HealthResponse, MetaResponse

API_PREFIX: Final = "/api/v1"
CURRENT_PHASE: Final = "A0.5"
PROJECT_STATUS: Final = "foundation"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application with validated runtime settings."""
    runtime_settings = settings or get_settings()
    application = FastAPI(
        title="KnowledgeScope API",
        version=__version__,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept"],
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

    application.include_router(router)
    return application


app = create_app()
