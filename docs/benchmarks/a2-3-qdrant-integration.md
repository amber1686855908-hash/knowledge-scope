# A2.3 Qdrant 集成报告

## 范围

A2.3 将 A1.6 chunk artifact 接入本地 Qdrant，使用 A2.2 固定的
`Qwen/Qwen3-Embedding-0.6B`。本阶段只实现 dense vector indexing 和 Top-K
retrieval，不包含 reranker、sparse/hybrid retrieval、GraphRAG、生成式 RAG 或前端检索页面。

## Collection 设计

- Docker image：`qdrant/qdrant:v1.15.5`；Python client：`qdrant-client==1.15.1`。
- collection：`knowledgescope_chunks_v1`，collection schema version `1.0`。
- vector size：`1024`；distance：`Cosine`。
- point ID：对 `chunk_id` 和 collection schema version 做 UUID5，重复索引会覆盖同一点。
- Qdrant 端口只绑定到 `127.0.0.1:6333`，数据使用 Docker named volume。

每个 point 的 payload 保留：

`chunk_id`、`document_id`、可选的 `knowledge_base_id`、`page_start`、`page_end`、
`source_block_ids`、`section_path`、`content_types`、`asset_refs`、chunk text、
`chunking_config_fingerprint`、embedding model/revision/config fingerprint 和
collection schema version。因此可以沿着 `Chunk → Block → Page → Document` 做后续引用和 reranking。

## 生命周期

索引一个文档时先读取并校验已有 chunk artifact，再生成全部向量；新点全部写入成功后，
才删除该文档旧的 stale point。upsert 或 stale 删除失败时，会删除本次新点并恢复旧 point
快照；如果回滚也失败，则显式报告需要人工一致性修复。这是 Qdrant 内部的补偿式恢复，
不是 PostgreSQL、文件系统与 Qdrant 之间的原子事务。重新生成 chunk 后再次执行
`index-document` 会根据新的确定性 ID 清理旧 chunk。删除文档时 API 会先完成文件/元数据删除，
再在线程池中执行 Qdrant vector cleanup；失败会返回明确的清理错误，但跨系统删除仍可能留下
需要后续人工清理的 orphan vector。

## 真实本地 smoke test

未重新运行 MinerU。使用已有 `data/benchmarks/a1-5/canonical/`，按 A1.6 默认 chunk 策略
索引 3 个 canonical 文档：

| document_id | indexed chunks | removed stale chunks |
| --- | ---: | ---: |
| `239e8ee8-4b77-5201-9d3d-625caba1c96c` | 17 | 0 |
| `8af9b576-e0a3-55c4-ba7e-861eba558337` | 19 | 0 |
| `8482bf4e-2862-56e1-bb2e-bb64a01b8d2d` | 19 | 0 |

总计写入 55 个 Qdrant points。使用 Qwen3-Embedding-0.6B 查询：

```text
说明事理时应重点说明哪些内容?
```

在 `document_id=8af9b576-e0a3-55c4-ba7e-861eba558337` 过滤下，Top-1 为：

- score：`0.7120327`
- chunk：`chunk-17f3732bb9fb5317ab6c48b8916dbe24566a6fb82fcbb31b59bf18d6dd3a479b`
- page：18
- source blocks：`p18-b3` 至 `p18-b7`
- section：`如何清晰地说明事理`

这是实际本地 Qdrant 和实际 Qwen 模型返回的结果，不是 mock 分数。

## 验证

- Qdrant container health：healthy。
- `uv run knowledgescope qdrant check`：collection ready，1024 dimensions。
- Qdrant integration round-trip：passed。
- A2.3 默认单元/API 路径：9 passed；真实 Qdrant integration round-trip：1 passed。
- 完整后端套件：200 passed；未设置 integration 开关时真实服务测试会跳过。
- 完整 Ruff、pytest、frontend 回归和 `git diff --check` 在交付前执行。

## 已知限制

- 上传、解析或 chunk artifact 更新不会自动加载模型并索引；需要显式执行 Qdrant CLI。
- `index-document` 依赖已有 `data/chunking/<document_id>/chunks.json`；`index-corpus` 可从现有
  canonical artifact 按默认 A1.6 策略生成内存 chunk。
- 当前只支持本地 Qdrant、单 collection、Qwen3-Embedding-0.6B 和 dense Top-K；生产对象存储、
  多模型、多租户权限、reranking、hybrid retrieval 和前端检索体验留待后续阶段。
