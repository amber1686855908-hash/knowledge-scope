import pytest
from httpx import AsyncClient


async def create_data_source(client: AsyncClient) -> dict[str, object]:
    response = await client.post(
        "/api/v1/chatbi/data-sources",
        json={
            "display_name": "  销售数据库  ",
            "connection_ref": "env:CHATBI_DEMO_DATABASE_URL",
            "default_schema": " public ",
        },
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.anyio
async def test_datasource_api_never_returns_connection_reference(client: AsyncClient) -> None:
    created = await create_data_source(client)

    assert created["display_name"] == "销售数据库"
    assert created["dialect"] == "postgresql"
    assert created["default_schema"] == "public"
    assert created["connection_configured"] is True
    assert "connection_ref" not in created

    detail = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}")
    listing = await client.get("/api/v1/chatbi/data-sources?limit=10")

    assert detail.status_code == 200
    assert detail.json() == created
    assert listing.status_code == 200
    assert listing.json()["items"] == [created]
    assert listing.json()["total"] == 1


@pytest.mark.anyio
async def test_datasource_api_updates_safe_metadata_and_supports_disable_delete(
    client: AsyncClient,
) -> None:
    created = await create_data_source(client)

    updated_response = await client.patch(
        f"/api/v1/chatbi/data-sources/{created['id']}",
        json={"display_name": "销售数据库 (停用)", "enabled": False},
    )

    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["display_name"] == "销售数据库 (停用)"
    assert updated["enabled"] is False
    assert updated["connection_configured"] is True
    assert "connection_ref" not in updated

    delete_response = await client.delete(f"/api/v1/chatbi/data-sources/{created['id']}")
    missing_response = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}")

    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert missing_response.status_code == 404


@pytest.mark.anyio
async def test_datasource_api_rejects_raw_urls_and_invalid_patches(client: AsyncClient) -> None:
    raw_url_response = await client.post(
        "/api/v1/chatbi/data-sources",
        json={
            "display_name": "不安全数据源",
            "connection_ref": "postgresql://user:password@db.example.test/business",
        },
    )
    empty_patch_response = await client.patch(
        "/api/v1/chatbi/data-sources/00000000-0000-0000-0000-000000000000",
        json={},
    )

    assert raw_url_response.status_code == 422
    assert "password" not in raw_url_response.text
    assert empty_patch_response.status_code == 422
