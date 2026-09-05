from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient


async def create_knowledge_base(
    client: AsyncClient,
    *,
    name: str = "质量管理规范",
    description: str | None = "质量管理相关资料",
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": name, "description": description},
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.anyio
async def test_create_and_get_knowledge_base(client: AsyncClient) -> None:
    created = await create_knowledge_base(client, name="  质量管理规范  ")

    assert UUID(created["id"])
    assert created["name"] == "质量管理规范"
    assert created["description"] == "质量管理相关资料"
    assert datetime.fromisoformat(created["created_at"]).tzinfo is not None
    assert datetime.fromisoformat(created["updated_at"]).tzinfo is not None

    response = await client.get(f"/api/v1/knowledge-bases/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


@pytest.mark.anyio
async def test_list_knowledge_bases_supports_deterministic_pagination(client: AsyncClient) -> None:
    created_items = [
        await create_knowledge_base(client, name=name) for name in ("第一条", "第二条", "第三条")
    ]

    all_response = await client.get("/api/v1/knowledge-bases?limit=100&offset=0")
    page_response = await client.get("/api/v1/knowledge-bases?limit=2&offset=1")

    assert all_response.status_code == 200
    assert page_response.status_code == 200
    all_items = all_response.json()["items"]
    expected_items = sorted(
        created_items,
        key=lambda item: (item["created_at"], item["id"]),
        reverse=True,
    )
    assert all_response.json()["total"] == 3
    assert all_items == expected_items
    assert page_response.json() == {
        "items": expected_items[1:3],
        "total": 3,
        "limit": 2,
        "offset": 1,
    }


@pytest.mark.anyio
async def test_update_knowledge_base_persists_changes(client: AsyncClient) -> None:
    created = await create_knowledge_base(client)

    partial_response = await client.patch(
        f"/api/v1/knowledge-bases/{created['id']}",
        json={"description": "更新后的说明"},
    )

    assert partial_response.status_code == 200
    partially_updated = partial_response.json()
    assert partially_updated["id"] == created["id"]
    assert partially_updated["name"] == created["name"]
    assert partially_updated["description"] == "更新后的说明"
    assert partially_updated["updated_at"] > created["updated_at"]

    response = await client.patch(
        f"/api/v1/knowledge-bases/{created['id']}",
        json={"name": "  更新后的规范  ", "description": None},
    )

    assert response.status_code == 200
    updated = response.json()
    assert updated["id"] == created["id"]
    assert updated["name"] == "更新后的规范"
    assert updated["description"] is None
    assert updated["updated_at"] > partially_updated["updated_at"]

    persisted = await client.get(f"/api/v1/knowledge-bases/{created['id']}")
    assert persisted.status_code == 200
    assert persisted.json() == updated


@pytest.mark.anyio
async def test_delete_knowledge_base_removes_resource(client: AsyncClient) -> None:
    created = await create_knowledge_base(client)

    response = await client.delete(f"/api/v1/knowledge-bases/{created['id']}")

    assert response.status_code == 204
    assert response.content == b""
    missing_response = await client.get(f"/api/v1/knowledge-bases/{created['id']}")
    empty_list_response = await client.get("/api/v1/knowledge-bases")
    assert missing_response.status_code == 404
    assert empty_list_response.status_code == 200
    assert empty_list_response.json()["items"] == []
    assert empty_list_response.json()["total"] == 0


@pytest.mark.anyio
async def test_missing_knowledge_base_returns_not_found(client: AsyncClient) -> None:
    missing_id = uuid4()

    get_response = await client.get(f"/api/v1/knowledge-bases/{missing_id}")
    patch_response = await client.patch(
        f"/api/v1/knowledge-bases/{missing_id}",
        json={"name": "不会创建"},
    )
    delete_response = await client.delete(f"/api/v1/knowledge-bases/{missing_id}")

    assert get_response.status_code == 404
    assert patch_response.status_code == 404
    assert delete_response.status_code == 404


@pytest.mark.anyio
async def test_invalid_name_and_pagination_input_returns_validation_error(
    client: AsyncClient,
) -> None:
    invalid_create_payloads = [
        {"name": "   "},
        {"name": "x" * 201},
        {},
    ]
    for payload in invalid_create_payloads:
        response = await client.post("/api/v1/knowledge-bases", json=payload)
        assert response.status_code == 422

    created = await create_knowledge_base(client)
    for payload in ({}, {"name": None}, {"name": "   "}):
        response = await client.patch(
            f"/api/v1/knowledge-bases/{created['id']}",
            json=payload,
        )
        assert response.status_code == 422

    assert (await client.get("/api/v1/knowledge-bases?limit=0")).status_code == 422
    assert (await client.get("/api/v1/knowledge-bases?limit=101")).status_code == 422
    assert (await client.get("/api/v1/knowledge-bases?offset=-1")).status_code == 422
    assert (await client.get("/api/v1/knowledge-bases/not-a-uuid")).status_code == 422
