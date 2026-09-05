# KnowledgeScope

KnowledgeScope 是一个面向行业文档的 Python 3.12 项目，当前提供知识库管理和 PDF 文档上传的 Web 应用基础。

## 当前状态

Phase A1.2 已完成，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package，以及现有的 settings 和 health CLI；
- 基于 FastAPI 的 `GET /api/v1/health` 和 `GET /api/v1/meta`；
- 基于 PostgreSQL、SQLAlchemy 2.x async 和 `asyncpg` 的知识库与文档元数据持久化；
- 使用 Alembic 管理 `knowledge_bases` 和 `documents` 表结构；
- 知识库的 `POST`、`GET`、`PATCH`、`DELETE /api/v1/knowledge-bases` CRUD API，支持分页、校验和 404 响应；
- 知识库文档的 PDF-only `POST`、`GET`、`DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents` API，支持分页、SHA-256 去重和受控错误响应；
- 将上传的原始 PDF 保存在 Settings 的 `data_dir` 下，使用固定的相对路径 `documents/<knowledge_base_id>/<document_id>/original.pdf`（默认文件系统路径为 `data/documents/<knowledge_base_id>/<document_id>/original.pdf`）；文件名只作为规范化元数据保存；
- 默认通过 `KNOWLEDGE_SCOPE_MAX_UPLOAD_SIZE_BYTES=52428800` 将单文件大小限制为 50 MiB，并统一保存为 `application/pdf`；同一知识库中的相同 SHA-256 文件会被拒绝，不同知识库可以分别上传；
- 使用 Vue 3、TypeScript、Vite、Vue Router、Element Plus、`@tanstack/vue-query` 和 Pinia 的 `frontend/` 应用；
- 知识库列表和详情页面，支持真实数据的加载、空状态、错误重试、新建、编辑、删除确认、分页，以及 PDF 文档上传、列表、删除和删除确认。

当前本地文件布局用于开发和参考环境，不等同于生产对象存储方案。MinerU、文档解析、chunking、向量检索、RAG、GraphRAG、multimodal retrieval、evaluation、LLM、Agent、ChatBI、NL2SQL、MinIO 和 S3 均尚未实现。

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

### 上传文件说明

当前上传接口只接受真实 PDF：服务端会检查文件名、`.pdf` 扩展名、文件头和大小，并以流式方式计算 SHA-256 后写入本地文件。上传文件不会被加入 Git；`.env`、`data/`、`node_modules/` 和前端构建产物也不会被提交。

## 验证

后端测试会使用 PostgreSQL 创建独立的 UUID 命名临时数据库，执行迁移，并在测试会话结束后只清理该临时库。未设置 `KNOWLEDGE_SCOPE_TEST_DATABASE_URL` 时，测试仅允许从 `KNOWLEDGE_SCOPE_DATABASE_URL` 的本机地址（`localhost`、`127.0.0.1` 或 `::1`）创建临时库；若应用地址是远程主机，必须显式配置 `KNOWLEDGE_SCOPE_TEST_DATABASE_URL`。Compose 默认用户具备所需权限。文件测试使用独立的临时 `data_dir`。

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

后续阶段将单独设计文档 ingestion、parsing、chunking、vector retrieval、GraphRAG、multimodal retrieval、evaluation、ChatBI 和 NL2SQL；当前版本不包含这些能力。
