# KnowledgeScope

KnowledgeScope 是一个面向行业文档的 Python 3.12 项目，目前提供知识库管理的 Web 应用基础。

## 当前状态

Phase A1.1 已完成，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package，以及现有的 settings 和 health CLI；
- 基于 FastAPI 的 `GET /api/v1/health` 和 `GET /api/v1/meta`；
- 基于 PostgreSQL、SQLAlchemy 2.x async 和 `asyncpg` 的知识库持久化；
- 使用 Alembic 管理 `knowledge_bases` 表结构；
- `POST`、`GET`、`PATCH`、`DELETE /api/v1/knowledge-bases` CRUD API，支持分页、校验和 404 响应；
- 使用 Vue 3、TypeScript、Vite、Vue Router、Element Plus、`@tanstack/vue-query` 和 Pinia 的 `frontend/` 应用；
- 知识库列表和详情页面，支持真实数据的加载、空状态、错误重试、新建、编辑、删除确认和分页。

文档 ingestion、MinerU、parsing、chunking、RAG、GraphRAG、multimodal retrieval、evaluation、LLM、Agent、ChatBI 和 NL2SQL 均尚未实现。

## 本地开发

### 安装依赖

```bash
uv sync
```

前端依赖安装：

```bash
cd frontend
npm install
```

### 启动 PostgreSQL

```bash
docker compose up -d postgres
```

Compose 默认将 PostgreSQL 映射到 `127.0.0.1:5433`，本地开发凭据和数据库名定义在 [compose.yaml](compose.yaml) 中。复制 [.env.example](.env.example) 为 `.env` 后，可通过 `KNOWLEDGE_SCOPE_POSTGRES_PORT` 修改宿主机端口，并同步更新 `KNOWLEDGE_SCOPE_DATABASE_URL`；不要在 `.env` 中提交 secrets。

### 执行数据库迁移

```bash
uv run alembic upgrade head
```

Alembic 是数据库 schema 的唯一来源；应用启动不会调用 `create_all()`。回退最近一次迁移：

```bash
uv run alembic downgrade -1
```

### 启动后端和前端

后端：

```bash
uv run uvicorn knowledge_scope.api.app:app --reload
```

前端（在另一个终端执行）：

```bash
cd frontend
npm run dev
```

Vite 会将 `/api` 请求代理到 `http://127.0.0.1:8000`，前端默认通过 `VITE_API_BASE_URL=/api` 访问 API。

Vite 默认使用 native file watcher。若开发机受到 inotify 或 file watcher limit 限制，可在 `frontend/.env` 中将 `VITE_USE_POLLING=true`；后端 `uvicorn --reload` 可在命令前加 `WATCHFILES_FORCE_POLLING=true`。这些 polling 选项仅用于开发环境。

## 验证

后端测试会使用 PostgreSQL 创建独立的临时测试数据库，执行迁移，并在测试会话结束后清理；Compose 默认用户具备所需权限。

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

文档 ingestion、parsing、chunking、vector retrieval、GraphRAG、multimodal retrieval、evaluation、ChatBI 和 NL2SQL 将在后续阶段单独设计和实现；当前版本不包含这些能力。
