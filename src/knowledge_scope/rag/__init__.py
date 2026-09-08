"""Retrieval-augmented question answering orchestration."""

from knowledge_scope.rag.context import ContextSelection, SelectedContextItem, assemble_context
from knowledge_scope.rag.prompt import RAG_PROMPT_VERSION, build_rag_messages
from knowledge_scope.rag.schemas import (
    RAGAnswerDeltaData,
    RAGCitation,
    RAGCitationsData,
    RAGCompleteData,
    RAGErrorData,
    RAGQueryRequest,
    RAGStreamEvent,
)
from knowledge_scope.rag.service import RAGService

__all__ = [
    "RAG_PROMPT_VERSION",
    "ContextSelection",
    "RAGAnswerDeltaData",
    "RAGCitation",
    "RAGCitationsData",
    "RAGCompleteData",
    "RAGErrorData",
    "RAGQueryRequest",
    "RAGService",
    "RAGStreamEvent",
    "SelectedContextItem",
    "assemble_context",
    "build_rag_messages",
]
