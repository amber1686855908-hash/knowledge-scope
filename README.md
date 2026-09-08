# KnowledgeScope

KnowledgeScope 是一个面向行业文档的 Python 3.12 项目，当前提供知识库管理和 PDF 文档上传的 Web 应用基础。

## 当前状态

Phase A2.3 已完成，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package，以及通过 Settings 驱动的 health、parse-document 和 chunk-document CLI；
- 基于 FastAPI 的 `GET /api/v1/health` 和 `GET /api/v1/meta`；
- 基于 PostgreSQL、SQLAlchemy 2.x async 和 `asyncpg` 的知识库与文档元数据持久化；
- 使用 Alembic 管理 `knowledge_bases` 和 `documents` 表结构；
- 知识库的 `POST`、`GET`、`PATCH`、`DELETE /api/v1/knowledge-bases` CRUD API，支持分页、校验和 404 响应；
- 知识库文档的 PDF-only `POST`、`GET`、`DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents` API，支持分页、SHA-256 去重和受控错误响应；
- 将上传的原始 PDF 保存在 Settings 的 `data_dir` 下，使用固定的相对路径 `documents/<knowledge_base_id>/<document_id>/original.pdf`（默认文件系统路径为 `data/documents/<knowledge_base_id>/<document_id>/original.pdf`）；文件名只作为规范化元数据保存；
- 默认通过 `KNOWLEDGE_SCOPE_MAX_UPLOAD_SIZE_BYTES=52428800` 将单文件大小限制为 50 MiB，并统一保存为 `application/pdf`；同一知识库中的相同 SHA-256 文件会被拒绝，不同知识库可以分别上传；
- 使用 Vue 3、TypeScript、Vite、Vue Router、Element Plus、`@tanstack/vue-query` 和 Pinia 的 `frontend/` 应用；
- 知识库列表和详情页面，支持真实数据的加载、空状态、错误重试、新建、编辑、删除确认、分页，以及 PDF 文档上传、列表、删除和删除确认。
- 位于 `knowledge_scope.parsing.models` 的解析器无关 `CanonicalDocument` Pydantic v2 数据模型，包含页面、标题、正文、表格、公式和图片引用的规范化表示、结构校验与 JSON 往返序列化；`TableBlock` 支持非空 Markdown、HTML 或 opaque `asset_ref`，并可同时保留结构化表示与资产引用。
- 通过独立虚拟环境中的 MinerU `3.4.5` `pipeline` backend 解析单个已上传 PDF 的开发者 CLI：`uv run knowledgescope parse-document <document-id>`；MinerU 不作为 KnowledgeScope 的 Python 依赖，也不被 FastAPI 请求直接调用。
- 解析器会读取 `*_content_list.json`，将页面、文本、标题、表格、公式、图片和可读列表转换为 `CanonicalDocument`，对页码、reading order、bbox、block ID 和图片/表格 artifact 引用执行校验，并报告跳过项、unsupported 项、`bbox_clamped`、表格降级统计和 warning 数量。
- 规范化结果和 MinerU 原始输出分别保存在 `data/parsing/<document_id>/canonical.json`、`manifest.json` 和 `mineru/` 下；文件使用 staging 与原子提升，原始上传仍保存在 `data/documents/`。
- 位于 `knowledge_scope.chunking` 的解析器无关 `Chunk`/`ChunkedDocument` Pydantic 模型，保存确定性文本、页面范围、来源 `source_block_ids`、`section_path`、`content_types` 和 opaque `asset_refs`；分块不增加数据库表，asset-only chunk 可以没有文本但必须保留资产引用。
- 纯核心 API `chunk_document(document, config)` 按连续标题上下文、canonical block 和字符预算执行可重复的结构感知分块；默认 `target_chars=1200`、`max_chars=1600`、`min_chars=240`，字符预算不是 token 或检索质量承诺。
- 开发者 CLI `uv run knowledgescope chunk-document <document-id>` 读取已有 `data/parsing/<document_id>/canonical.json`，将 chunk artifact 写入 `data/chunking/<document_id>/chunks.json` 和 `manifest.json`；当前不会在上传请求中自动分块，也不需要 MinerU、GPU 或 embedding 模型。

Phase A1.5 已完成只读的全量解析基准：清单包含 257 个 PDF 条目、255 个唯一内容、1,580,533,246 字节和 3,833 个物理页，覆盖 9 个学科且没有未分类条目；2 个重复内容条目只解析代表文件，重复条目记录为 `skipped_duplicate`。首次运行中 244 个唯一内容成功、11 个因 MinerU 表格缺少 `table_body` 被严格 adapter 拒绝，成功率为 95.6863%；随后只重试这 11 个失败代表，最新结果为 255/255 个唯一内容均完成 `CanonicalDocument` 适配。100% 表示流水线完成，不是标注数据支持的解析准确率；其中 13 个表格条目因同时缺少结构化内容和可用资产而以 `table_missing_content` warning 降级。基准支持逐条 checkpoint、`--resume`、显式重试失败项和 `failures|all|none` 原始结果保留策略；运行产物位于被忽略的 `data/benchmarks/a1-5/`，汇总见 [A1.5 基准报告](docs/benchmarks/a1-5-corpus-parsing.md)。

Phase A1.6 使用这批已有 canonical 结果做了 CPU-only 结构分块校验：255/255 个文档成功生成 7,524 个默认 chunk，46,504 个 source block 和 6,176 个 asset block 的覆盖率均为 100%，未重新运行 MinerU；画像、策略比较、硬化前后对比和学科统计见 [A1.6 分块基准报告](docs/benchmarks/a1-6-semantic-chunking.md)。

Phase A2.1 已完成基于 canonical `document_id`、页码和 `source_block_ids` 的文本检索评估集：人工审核后保留 108 条 `verified`、54 条 `rejected`，覆盖 9 个学科且每科 12 条；最终 `retrieval-eval-v1` 划分为 72 条 dev 和 36 条 test，每科分别为 8/4。教材问题只保存在 `query_source` 审计字段，gold `evidence` 仅使用答案性正文、公式或文本表格，并通过 query 泄漏、evidence fingerprint、chunk lineage、重复 evidence/chunk 分组和 dev/test 分组校验；编辑 query 后会重新生成 deterministic item ID。运行时产物位于被忽略的 `data/evaluation/a2-1/`，仓库安全的最终标注见 [A2.1 retrieval-eval-v1](docs/benchmarks/a2-1-retrieval-eval-v1.jsonl)；详见 [A2.1 检索评估集报告](docs/benchmarks/a2-1-retrieval-eval-set.md)。A2.1 评测集本身不包含 embedding、真实检索分数或任何下游 RAG 生成能力。

Phase A2.2 已完成本地 dense embedding 模型选型基准：在冻结的 A2.1 dev/test 集上比较 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Embedding-4B`、`BAAI/bge-m3` 和 `intfloat/multilingual-e5-large-instruct`，记录 Hit@K、MRR、EvidenceRecall、编码吞吐、查询延迟和 CUDA 显存。当前建议将 `Qwen/Qwen3-Embedding-0.6B` 作为默认工程基线，将 4B 保留为质量优先候选；该结论受 `512-token common profile` 截断限制约束，详见 [A2.2 模型选型报告](docs/benchmarks/a2-2-embedding-model-selection.md)。

Phase A2.3 已完成 Qdrant 持久化 dense retrieval 基础：Qdrant 使用本地 Docker 服务和 `knowledgescope_chunks_v1` collection，向量维度为 1024、距离为 cosine；`Qwen/Qwen3-Embedding-0.6B` 使用 A2.2 固定 revision、`prompt_name=query`、L2 归一化和 512-token 配置。索引 payload 保留 `chunk_id`、`document_id`、可选的 `knowledge_base_id`、页码、`source_block_ids`、`section_path`、`content_types`、`asset_refs`、文本和 chunk/embedding 指纹，支持确定性 point ID、文档级安全重建、旧点清理、删除清理和失败时的补偿式回滚。PostgreSQL、文件系统与 Qdrant 之间不是原子事务；向量清理失败会明确报错，并可能需要后续人工修复。详见 [A2.3 Qdrant 集成报告](docs/benchmarks/a2-3-qdrant-integration.md)。

Phase A2.4 已完成本地 reranker 基础：在 Qwen dense Top-K 之后，可通过 CLI 调用本地 Cross-Encoder 对候选 chunk 重新排序；当前基准比较 `Qwen/Qwen3-Reranker-0.6B`、`BAAI/bge-reranker-v2-m3` 和 `Alibaba-NLP/gte-multilingual-reranker-base`，并记录不同候选池大小下的质量、延迟、吞吐和 CUDA 显存。当前建议将 `BAAI/bge-reranker-v2-m3` 作为质量优先的 reranker 候选，将 GTE 保留为低延迟候选；结论只基于 108 条冻结评测集和单机离线运行，不代表生产模型定论。详见 [A2.4 本地 reranker 基准报告](docs/benchmarks/a2-4-local-reranker.md)。

当前 PDF 不会在上传请求中自动解析；需要使用开发者 CLI 显式触发。当前本地文件布局用于开发和参考环境，不等同于生产对象存储方案。解析集成说明详见 [MinerU 本地集成](docs/integrations/mineru.md)，模型约定详见 [CanonicalDocument 规范](docs/architecture/canonical-document-model.md)，分块约定详见 [CanonicalDocument → Chunk 规范](docs/architecture/canonical-document-chunking.md)。

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
docker compose up -d postgres qdrant
```

Compose 默认将 PostgreSQL 映射到 `127.0.0.1:5433`、Qdrant 映射到 `127.0.0.1:6333`，本地开发凭据、数据库名和 Qdrant collection 配置定义在 [compose.yaml](compose.yaml) 与 [.env.example](.env.example) 中。复制 [.env.example](.env.example) 为 `.env` 后，可通过 `KNOWLEDGE_SCOPE_POSTGRES_PORT` 和 `KNOWLEDGE_SCOPE_QDRANT_URL` 修改连接配置；不要在 `.env` 中提交 secrets。Qdrant 数据保存在 Docker named volume，不会写入 Git。

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

MinerU 运行时位于仓库之外。解析 CLI 使用 `KNOWLEDGE_SCOPE_MINERU_COMMAND` 指定外部 `mineru` 可执行文件，并使用 `KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS` 设置超时；安装和模型配置见 [MinerU 本地集成](docs/integrations/mineru.md)。

对已经存在的 `CanonicalDocument` artifact，可显式运行结构分块：

```bash
uv run knowledgescope chunk-document <document-id>
```

该命令不会重新运行 MinerU，也不会在上传时自动执行。

A1.5 基准命令必须显式接收只读语料根目录，不会写入语料目录：

```bash
uv run knowledgescope benchmark-parsing --corpus /path/to/read-only-corpus
```

可使用 `--resume`、`--retry-failed`、`--raw-retention failures|all|none`、`--subject` 和 `--limit` 控制可恢复运行与 smoke test；默认运行工作区为被忽略的 `data/benchmarks/a1-5/`。

Qwen embedding 运行需要本地安装 embedding benchmark 依赖：

```bash
uv sync --group embedding-benchmark
```

默认 embedding device 为 CUDA；没有可用 GPU 时，可在 `.env` 中设置
`KNOWLEDGE_SCOPE_EMBEDDING_DEVICE=cpu` 和 `KNOWLEDGE_SCOPE_EMBEDDING_DTYPE=float32`。

检查或创建 Qdrant collection：

```bash
uv run knowledgescope qdrant check
uv run knowledgescope qdrant create
```

对已经存在的 `data/chunking/<document_id>/chunks.json` 执行单文档索引：

```bash
uv run knowledgescope qdrant index-document <document-id>
```

也可以直接使用 A1.5 的 canonical artifacts，按 A1.6 默认策略生成内存中的 chunk 并索引若干文档；该命令不会重新运行 MinerU：

```bash
uv run knowledgescope qdrant index-corpus --canonical-root data/benchmarks/a1-5/canonical --limit 3
```

执行 dense Top-K 查询：

```bash
uv run knowledgescope qdrant search "说明事理时应重点说明哪些内容?" --limit 5
```

安装本地 reranker 基准所需依赖并运行完整比较：

```bash
uv sync --group reranker-benchmark
uv run knowledgescope reranker-benchmark \
  --split both \
  --models qwen3-reranker-0.6b bge-reranker-v2-m3 gte-multilingual-reranker-base \
  --candidate-sizes 10 20 50
```

对已经建立的 Qdrant dense 索引执行 dense + reranker 查询：

```bash
uv run knowledgescope rerank-search "说明事理时应重点说明哪些内容?" --candidate-limit 20 --limit 5
```

后端提供 `GET /api/v1/health/qdrant` readiness 检查和 `POST /api/v1/retrieval/search` 检索接口；当前没有前端检索页面。Qdrant、模型、向量和 benchmark 运行产物均不提交到仓库。

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

后续阶段将继续扩展文档 ingestion 和 parsing 覆盖范围，并在当前 dense retrieval 与 reranker 基础上评估 sparse/hybrid retrieval、GraphRAG、multimodal retrieval、ChatBI 和 NL2SQL；当前版本不包含这些后续能力。
