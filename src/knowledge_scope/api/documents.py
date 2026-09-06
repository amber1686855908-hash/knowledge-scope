"""PDF document upload and metadata endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.documents.models import (
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_STATUS_UPLOADED,
    Document,
)
from knowledge_scope.documents.storage import (
    StagedUpload,
    StorageError,
    UploadValidationError,
    filesystem_path_for_storage_key,
    move_to_trash,
    permanently_remove_trash,
    promote_staged_upload,
    remove_file,
    remove_staged_upload,
    restore_from_trash,
    stage_pdf,
    storage_key_for_document,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

from .knowledge_bases import _get_knowledge_base
from .schemas import DocumentListResponse, DocumentResponse

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/documents",
    tags=["documents"],
)


def _runtime_settings(request: Request) -> Settings:
    return request.app.state.settings


def _is_duplicate_error(error: IntegrityError) -> bool:
    return "uq_documents_knowledge_base_sha256" in str(error)


def _cleanup_upload(staged: StagedUpload | None, final_path: Path | None) -> None:
    if final_path is not None:
        remove_file(final_path)
    if staged is not None:
        remove_staged_upload(staged.directory, staged.path)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    knowledge_base_id: UUID,
    request: Request,
    file: Annotated[UploadFile, File(...)],
    session: AsyncSession = Depends(get_session),
) -> Document:
    await _get_knowledge_base(session, knowledge_base_id)
    settings = _runtime_settings(request)
    staged: StagedUpload | None = None
    final_path: Path | None = None
    metadata_committed = False

    try:
        staged = await stage_pdf(file, settings.data_dir, settings.max_upload_size_bytes)
        existing_id = await session.scalar(
            select(Document.id).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.sha256 == staged.sha256,
            )
        )
        if existing_id is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="同一知识库中已存在相同文件",
            )

        document_id = uuid4()
        storage_key = storage_key_for_document(knowledge_base_id, document_id)
        final_path = filesystem_path_for_storage_key(settings.data_dir, storage_key)
        promote_staged_upload(staged, final_path)

        document = Document(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            original_filename=staged.original_filename,
            storage_key=storage_key,
            media_type=DOCUMENT_MEDIA_TYPE_PDF,
            size_bytes=staged.size_bytes,
            sha256=staged.sha256,
            status=DOCUMENT_STATUS_UPLOADED,
        )
        session.add(document)
        try:
            await session.commit()
        except IntegrityError as error:
            await session.rollback()
            _cleanup_upload(staged, final_path)
            if _is_duplicate_error(error):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="同一知识库中已存在相同文件",
                ) from None
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档元数据保存失败",
            ) from None
        metadata_committed = True
        await session.refresh(document)
        return document
    except UploadValidationError as error:
        await session.rollback()
        _cleanup_upload(staged, final_path)
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
    except HTTPException:
        await session.rollback()
        if not metadata_committed:
            _cleanup_upload(staged, final_path)
        raise
    except (OSError, StorageError, SQLAlchemyError):
        await session.rollback()
        if not metadata_committed:
            _cleanup_upload(staged, final_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档保存失败",
        ) from None
    finally:
        if staged is not None:
            remove_staged_upload(staged.directory, staged.path)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    await _get_knowledge_base(session, knowledge_base_id)

    total = int(
        await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )
        or 0
    )
    result = await session.scalars(
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return DocumentListResponse(
        items=list(result.all()),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Document:
    await _get_knowledge_base(session, knowledge_base_id)
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.knowledge_base_id == knowledge_base_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await _get_knowledge_base(session, knowledge_base_id)
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.knowledge_base_id == knowledge_base_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")

    settings = _runtime_settings(request)
    try:
        final_path = filesystem_path_for_storage_key(settings.data_dir, document.storage_key)
        trashed = move_to_trash(final_path, settings.data_dir)
    except (OSError, StorageError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档文件不可用, 删除失败",
        ) from None

    try:
        await session.delete(document)
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        try:
            restore_from_trash(trashed)
        except OSError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档删除失败, 且文件恢复失败",
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档删除失败",
        ) from None

    try:
        permanently_remove_trash(trashed)
    except OSError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但文件清理失败",
        ) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
