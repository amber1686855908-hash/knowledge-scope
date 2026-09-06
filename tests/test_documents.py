from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient, Response
from sqlalchemy.exc import SQLAlchemyError

from knowledge_scope.documents.storage import storage_key_for_document

VALID_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF\n"


async def create_knowledge_base(client: AsyncClient, name: str = "测试知识库") -> dict[str, object]:
    response = await client.post("/api/v1/knowledge-bases", json={"name": name})
    assert response.status_code == 201
    return response.json()


async def upload_pdf(
    client: AsyncClient,
    knowledge_base_id: str,
    *,
    filename: str = "manual.pdf",
    content: bytes = VALID_PDF,
    content_type: str = "application/pdf",
) -> Response:
    response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": (filename, content, content_type)},
    )
    return response


@pytest.mark.anyio
async def test_upload_persists_pdf_metadata_and_uses_fixed_storage_path(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    response = await upload_pdf(
        client,
        str(knowledge_base["id"]),
        filename="../原始文件.PDF",
        content_type="text/plain",
    )

    assert response.status_code == 201
    document = response.json()
    document_id = UUID(document["id"])
    knowledge_base_id = UUID(knowledge_base["id"])
    assert set(document) == {
        "id",
        "knowledge_base_id",
        "original_filename",
        "media_type",
        "size_bytes",
        "sha256",
        "status",
        "created_at",
        "updated_at",
    }
    expected_storage_key = storage_key_for_document(knowledge_base_id, document_id)
    expected_path = test_data_dir / expected_storage_key

    assert document["knowledge_base_id"] == str(knowledge_base_id)
    assert document["original_filename"] == "原始文件.PDF"
    assert document["media_type"] == "application/pdf"
    assert document["size_bytes"] == len(VALID_PDF)
    assert document["sha256"] == hashlib.sha256(VALID_PDF).hexdigest()
    assert document["status"] == "uploaded"
    assert expected_path.read_bytes() == VALID_PDF
    assert expected_path.name == "original.pdf"
    assert "原始文件" not in str(expected_path)

    detail_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )
    list_response = await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert detail_response.status_code == 200
    assert detail_response.json() == document
    assert list_response.status_code == 200
    assert list_response.json() == {
        "items": [document],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }


@pytest.mark.anyio
async def test_duplicate_pdf_is_rejected_only_within_the_same_knowledge_base(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])

    first_response = await upload_pdf(client, knowledge_base_id, filename="first.pdf")
    files_after_first_upload = sorted(
        path.relative_to(test_data_dir) for path in test_data_dir.rglob("*") if path.is_file()
    )
    duplicate_response = await upload_pdf(client, knowledge_base_id, filename="renamed.pdf")

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert "相同文件" in duplicate_response.json()["detail"]
    assert files_after_first_upload == sorted(
        path.relative_to(test_data_dir) for path in test_data_dir.rglob("*") if path.is_file()
    )

    list_response = await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert list_response.json()["total"] == 1


@pytest.mark.anyio
async def test_same_pdf_can_be_uploaded_to_different_knowledge_bases(
    client: AsyncClient,
) -> None:
    first_knowledge_base = await create_knowledge_base(client, "第一个知识库")
    second_knowledge_base = await create_knowledge_base(client, "第二个知识库")

    first_response = await upload_pdf(client, str(first_knowledge_base["id"]))
    second_response = await upload_pdf(client, str(second_knowledge_base["id"]))

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["sha256"] == second_response.json()["sha256"]
    first_storage_key = storage_key_for_document(
        UUID(first_knowledge_base["id"]), UUID(first_response.json()["id"])
    )
    second_storage_key = storage_key_for_document(
        UUID(second_knowledge_base["id"]), UUID(second_response.json()["id"])
    )
    assert first_storage_key != second_storage_key


@pytest.mark.anyio
async def test_document_list_is_newest_first_and_supports_bounded_pagination(
    client: AsyncClient,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    created_documents = [
        (
            await upload_pdf(
                client,
                knowledge_base_id,
                filename=f"manual-{index}.pdf",
                content=VALID_PDF + str(index).encode(),
            )
        ).json()
        for index in range(3)
    ]

    all_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=100&offset=0"
    )
    page_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=2&offset=1"
    )

    assert all_response.status_code == 200
    assert page_response.status_code == 200
    all_items = all_response.json()["items"]
    expected_items = sorted(
        created_documents,
        key=lambda item: (item["created_at"], item["id"]),
        reverse=True,
    )
    assert all_items == expected_items
    assert page_response.json() == {
        "items": expected_items[1:3],
        "total": 3,
        "limit": 2,
        "offset": 1,
    }
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=101")
    ).status_code == 422


@pytest.mark.anyio
async def test_invalid_uploads_return_controlled_errors_without_orphan_files(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    invalid_uploads = (
        ("manual.txt", VALID_PDF, 415),
        ("fake.pdf", b"not a PDF", 415),
        ("empty.pdf", b"", 400),
    )

    for filename, content, expected_status in invalid_uploads:
        response = await upload_pdf(
            client,
            knowledge_base_id,
            filename=filename,
            content=content,
        )
        assert response.status_code == expected_status

    missing_filename_response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": (None, VALID_PDF, "application/pdf")},
    )
    assert missing_filename_response.status_code == 422

    assert not [path for path in test_data_dir.rglob("*") if path.is_file()]


@pytest.mark.anyio
async def test_upload_size_limit_is_configurable_and_staged_file_is_removed(
    limited_client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(limited_client)
    response = await upload_pdf(
        limited_client,
        str(knowledge_base["id"]),
        content=VALID_PDF,
    )

    assert response.status_code == 413
    assert not [path for path in test_data_dir.rglob("*") if path.is_file()]


@pytest.mark.anyio
async def test_missing_knowledge_base_does_not_create_storage(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    response = await upload_pdf(client, str(uuid4()))

    assert response.status_code == 404
    assert not test_data_dir.exists()


@pytest.mark.anyio
async def test_document_is_owned_by_its_knowledge_base(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    first_knowledge_base = await create_knowledge_base(client, "文档所属知识库")
    second_knowledge_base = await create_knowledge_base(client, "另一个知识库")
    document_response = await upload_pdf(client, str(first_knowledge_base["id"]))
    document = document_response.json()
    first_knowledge_base_id = UUID(first_knowledge_base["id"])
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(first_knowledge_base_id, document_id)

    get_response = await client.get(
        f"/api/v1/knowledge-bases/{second_knowledge_base['id']}/documents/{document_id}"
    )
    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{second_knowledge_base['id']}/documents/{document_id}"
    )

    assert get_response.status_code == 404
    assert delete_response.status_code == 404
    assert stored_path.exists()


@pytest.mark.anyio
async def test_delete_removes_document_metadata_and_file_and_allows_empty_kb_delete(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    document = (await upload_pdf(client, knowledge_base_id)).json()
    stored_path = test_data_dir / storage_key_for_document(
        UUID(knowledge_base_id), UUID(document["id"])
    )

    blocked_delete = await client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}")
    assert blocked_delete.status_code == 409
    assert stored_path.exists()

    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}"
    )
    knowledge_base_delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}"
    )

    assert delete_response.status_code == 204
    assert knowledge_base_delete_response.status_code == 204
    assert not stored_path.exists()
    assert not stored_path.parent.exists()
    assert not (test_data_dir / "parsing" / document["id"]).exists()
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}")
    ).status_code == 404


@pytest.mark.anyio
async def test_delete_removes_parsing_artifacts_with_document(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(knowledge_base_id, document_id)
    parsing_dir = test_data_dir / "parsing" / str(document_id)
    (parsing_dir / "mineru" / "images").mkdir(parents=True)
    (parsing_dir / "canonical.json").write_text("{}", encoding="utf-8")
    (parsing_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (parsing_dir / "mineru" / "images" / "image.png").write_bytes(b"asset")

    response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )

    assert response.status_code == 204
    assert not stored_path.exists()
    assert not parsing_dir.exists()
    assert not list((test_data_dir / "documents").glob(".delete-*"))
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    ).status_code == 404


@pytest.mark.anyio
async def test_delete_restores_source_and_parsing_artifacts_when_database_delete_fails(
    client: AsyncClient,
    test_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(knowledge_base_id, document_id)
    parsing_dir = test_data_dir / "parsing" / str(document_id)
    (parsing_dir / "mineru").mkdir(parents=True)
    (parsing_dir / "canonical.json").write_text("canonical", encoding="utf-8")
    (parsing_dir / "mineru" / "raw.json").write_text("raw", encoding="utf-8")

    async def fail_commit(_session: object) -> None:
        raise SQLAlchemyError("simulated database failure")

    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.commit", fail_commit)

    response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )

    assert response.status_code == 500
    assert "文档删除失败" in response.json()["detail"]
    assert stored_path.read_bytes() == VALID_PDF
    assert (parsing_dir / "canonical.json").read_text(encoding="utf-8") == "canonical"
    assert (parsing_dir / "mineru" / "raw.json").read_text(encoding="utf-8") == "raw"
    assert not list((test_data_dir / "documents").glob(".delete-*"))
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    ).status_code == 200
