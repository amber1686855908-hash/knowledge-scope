# KnowledgeScope

KnowledgeScope 是一个面向行业文档、用于未来 multimodal RAG platform 的 Python 3.12 项目基础。

## 当前状态

Phase A0 已完成，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package；
- 使用 Pydantic v2 和 `pydantic-settings` 实现的配置校验；
- 可报告 project、version、runtime 和 configuration status 的 health CLI；
- pytest 测试覆盖，以及 Ruff linting 和 formatting 配置；
- 面向后续 ingestion、parsing、chunking、retrieval、evaluation 和 shared infrastructure 的 package boundaries。

MinerU、document-processing、retrieval、GraphRAG、multimodal retrieval、evaluation、ChatBI 和 NL2SQL 均尚未实现。

## 开始使用

使用以下命令安装项目及开发依赖：

```bash
uv sync
```

运行 health check：

```bash
uv run knowledgescope health
```

本地 settings 可通过基于 [.env.example](.env.example) 创建的 `.env` 文件提供。`.env` 已被 Git 忽略，不得提交 secrets。

## 验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## 后续方向

后续阶段可能加入 document ingestion and parsing、chunking、vector and graph retrieval、multimodal retrieval、evaluation 和 ChatBI/NL2SQL。每项能力都会配套独立实现和测试；当前基础不会将尚未实现的能力描述为已具备。
