# KnowledgeScope

KnowledgeScope is a Python 3.12 project foundation for a future multimodal RAG platform focused on industry documents.

## Current status

Phase A0 currently provides:

- a `src/knowledge_scope` package managed with `uv`;
- validated application settings using Pydantic v2 and `pydantic-settings`;
- a small health CLI that reports the project, version, runtime, and configuration status;
- pytest coverage plus Ruff linting and formatting configuration; and
- package boundaries for future ingestion, parsing, chunking, retrieval, evaluation, and shared infrastructure work.

The document-processing, retrieval, GraphRAG, multimodal, evaluation, and ChatBI/NL2SQL capabilities are not implemented yet.

## Getting started

Install the project and development dependencies with:

```bash
uv sync
```

Run the health check:

```bash
uv run knowledgescope health
```

Local settings can be supplied through a `.env` file based on [.env.example](.env.example). `.env` is ignored by Git and must not contain committed secrets.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Direction

Later phases may add document ingestion and parsing, chunking, vector and graph retrieval, multimodal retrieval, evaluation, and ChatBI/NL2SQL. Each capability will be introduced with its own implementation and tests; this foundation intentionally does not pretend those capabilities exist.
