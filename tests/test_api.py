import anyio
import httpx
from fastapi import FastAPI

from knowledge_scope.api.app import create_app
from knowledge_scope.shared.config import Settings


def make_app() -> FastAPI:
    settings = Settings(
        _env_file=None,
        environment="test",
        cors_origins=["http://localhost:5173"],
    )
    return create_app(settings)


def get_response(
    path: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> httpx.Response:
    app = make_app()

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, headers=headers)

    return anyio.run(request)


def test_health_endpoint_reuses_core_health_report() -> None:
    response = get_response("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project_name"] == "KnowledgeScope"
    assert payload["version"] == "0.1.0"
    assert payload["config_status"] == "ok"
    assert payload["environment"] == "test"


def test_meta_endpoint_reports_current_foundation_status() -> None:
    response = get_response("/api/v1/meta")

    assert response.status_code == 200
    assert response.json() == {
        "project_name": "KnowledgeScope",
        "version": "0.1.0",
        "phase": "A1.4",
        "status": "foundation",
        "config_status": "ok",
    }


def test_cors_uses_configured_origins() -> None:
    response = get_response(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_allow_unconfigured_origins() -> None:
    response = get_response(
        "/api/v1/health",
        headers={"Origin": "https://example.com"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_json_crud_preflight() -> None:
    response = get_response(
        "/api/v1/knowledge-bases",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        method="OPTIONS",
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
