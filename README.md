# KnowledgeScope

KnowledgeScope 是一个面向行业文档、用于未来 multimodal RAG platform 的 Python 3.12 项目。

## 当前状态

Phase A0.5 已完成，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package，以及现有的 settings 和 health CLI；
- 基于 FastAPI 的 `GET /api/v1/health` 和 `GET /api/v1/meta`；
- 使用 Pydantic response models 的非敏感 health/meta 信息；
- 通过 `KNOWLEDGE_SCOPE_CORS_ORIGINS` 配置 CORS origins；
- 使用 Vue 3、TypeScript、Vite、Vue Router、Element Plus、`@tanstack/vue-query` 和 Pinia 的 `frontend/` 应用；
- 只包含项目概览和 404 页面的 Web application shell，概览数据来自真实后端 API，并包含 loading、error 和 success 状态。

MinerU、document ingestion、parsing、chunking、RAG、GraphRAG、multimodal retrieval、evaluation、ChatBI 和 NL2SQL 均尚未实现。

## 开始使用

后端依赖和环境：

```bash
uv sync
```

启动后端 API：

```bash
uv run uvicorn knowledge_scope.api.app:app --reload
```

启动前端：

```bash
cd frontend
npm install
npm run dev
```

本地完整开发时，保持后端和前端分别运行。Vite 会将 `/api` 请求代理到 `http://127.0.0.1:8000`，前端默认通过 `VITE_API_BASE_URL=/api` 访问 API。

Vite 默认使用 native file watcher。若当前开发机受到 inotify 或 file watcher limit 限制，可在 `frontend/.env` 中将 `VITE_USE_POLLING=true`，再运行 `npm run dev`；后端 `uvicorn --reload` 也可在命令前加 `WATCHFILES_FORCE_POLLING=true`。这些 polling 选项仅用于开发环境。

本地 settings 可通过基于 [.env.example](.env.example) 创建的 `.env` 文件提供；前端 API base 可参考 [frontend/.env.example](frontend/.env.example)。`.env` 已被 Git 忽略，不得提交 secrets。

## 验证

后端：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

前端：

```bash
cd frontend
npm run lint
npm run type-check
npm run build
```

## 后续方向

后续阶段可能加入 document ingestion and parsing、chunking、vector and graph retrieval、multimodal retrieval、evaluation 和 ChatBI/NL2SQL。每项能力都会配套独立实现和测试；当前基础不会将尚未实现的能力描述为已具备。
