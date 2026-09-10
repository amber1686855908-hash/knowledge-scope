# A3.5 Vector + Graph Hybrid Retrieval

Phase A3.5 提供一个独立的 hybrid retrieval 基础，不修改 A2.1–A2.5
评测结果，也不接入 A2.7 的 RAG answer generation。

## 检索链路

一次查询同时执行两条已有链路：

```text
向量分支：Qwen3-Embedding-0.6B → Qdrant dense candidates → BGE reranker
图分支：   A3.4 seed resolution → bounded graph traversal → KnowledgeEvidence
                         ↓
              以 (knowledge_base_id, document_id, chunk_id) 去重
                         ↓
                         RRF
```

hybrid service 只负责编排和融合，不复制 Qdrant、embedding、reranker 或
A3.4 的图检索逻辑。向量分支使用 `BAAI/bge-reranker-v2-m3`，这是当前
hybrid 默认 profile；图分支仍由 A3.4 的配置控制。

## 结果和 lineage

每个 `HybridResult` 以同一知识库内的
`knowledge_base_id`、`document_id`、`chunk_id` 作为去重键，并保留页码、
`source_block_ids` 和 `section_path`。`source` 明确表示 `vector`、
`graph` 或 `both`。

当同一 chunk 同时来自两个分支时，图 evidence 的页范围必须落在向量
chunk 范围内，source blocks 必须是该 chunk lineage 的子集；非空
`section_path` 也必须一致。发现冲突时拒绝融合并报告失败，不会静默选择
一侧或拼接不可信 lineage。图 evidence 可以只覆盖 chunk 的部分 source
blocks；图侧缺失 section path 只表示该字段未知。

向量侧保留 dense rank/score 和 reranker rank/score。图侧保留一个合并后的
graph rank/score、最佳 seed/retrieval reason、所有去重后的 evidence IDs
和 paths。一个 chunk 通过多个图 evidence 或 path 命中时只产生一次 graph
RRF rank contribution，避免重复路径人为抬高排序。

## 融合规则

默认使用 Reciprocal Rank Fusion：

```text
RRF(branch, item) = 1 / (rrf_k + rank)
hybrid_score      = vector_RRF + graph_RRF
```

`rrf_k` 默认由 `KNOWLEDGE_SCOPE_HYBRID_RRF_K=60` 配置。dense、reranker
和 graph 原始分数只用于解释和确定分支内的选择，不会被当作同一概率空间
直接相加。相同最终分数使用 KB、文档、页码、chunk 和 lineage 的稳定键
排序。

## 失败和并发语义

每个响应都包含 `vector_status` 和 `graph_status`，状态可以是 `success`、
`empty`、`failed` 或 `timed_out`。空结果表示分支正常完成但没有命中；失败
和超时会保留不含底层敏感细节的错误摘要。调用方取消整个请求时，
`CancelledError` 会向上游传播，不会伪造一个成功或空的 hybrid 结果；这与
分支失败是有意区分的。

- `degraded`（默认）：一条分支失败时返回另一条成功分支的结果，并明确
  标记 `degraded=true`；两条都失败时抛出错误。
- `strict`：任一分支失败即失败，不把失败伪装成空结果。

两条同步的现有检索链路通过 `asyncio.to_thread` 并发执行，避免阻塞
FastAPI/CLI 的 async 调用方。embedding adapter 和 reranker adapter 仍
复用进程级模型，并由各自已有的 inference lock 保护；当前没有新增 worker
池，也不承诺 GPU 多请求并行吞吐。线程中的本地推理在调用取消后不能被
强制中断，但 async 等待会停止，并会取消尚未完成的另一条任务。指定
`document_id` 时，图分支会把过滤条件传给 A3.4 检索服务，在其结果上限
生效前完成文档筛选，不会先截断整个知识库结果再在 hybrid 层补救。

默认 BGE profile 使用 A2.5 已验证的 revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`；如需切换，必须通过
`KNOWLEDGE_SCOPE_RERANKER_MODEL_REVISION` 显式固定一个 revision。

向量候选、rerank 结果、图结果和最终结果均有上限，配置键分别为：

```text
KNOWLEDGE_SCOPE_HYBRID_VECTOR_CANDIDATE_LIMIT
KNOWLEDGE_SCOPE_HYBRID_VECTOR_RERANK_LIMIT
KNOWLEDGE_SCOPE_HYBRID_GRAPH_RESULT_LIMIT
KNOWLEDGE_SCOPE_HYBRID_RESULT_LIMIT
KNOWLEDGE_SCOPE_HYBRID_FAILURE_MODE
```

## 开发者命令

需要本地 Qdrant、Neo4j、embedding benchmark 依赖和 BGE reranker 依赖：

```bash
uv run knowledgescope hybrid-search "查询内容" --knowledge-base-id <knowledge-base-uuid>
```

命令只输出 bounded 的 typed result 和分支状态，不创建 RAG 回答，不执行
全语料图抽取。Qdrant/Neo4j runtime data、模型和查询输出仍属于本地运行
产物，不提交到 Git。

现有 A2.3 reference collection 的 7,524 个 points 最初由
`index_canonical_corpus` 建立，payload 的 `knowledge_base_id` 为
`null`；canonical 参考文件本身没有可推断的知识库映射。对于明确属于同一
benchmark KnowledgeBase 的现有语料，必须先运行显式的
`knowledgescope corpus register --knowledge-base-id <uuid>`，由 PostgreSQL
`Document` 记录成为唯一归属来源，再执行下方的 payload-only repair。不能
从文件名、路径或 benchmark 结构猜测 KB，也不能为了补归属而重新 embedding。

维护已有 collection 时先运行只读审计：

```bash
uv run knowledgescope qdrant audit-kb
```

该命令只用 PostgreSQL `Document.id → knowledge_base_id` 做权威映射，不从
文件名、路径或 benchmark 结构推断。只有映射完整、唯一且非空时，才显式
追加 `--apply`；apply 只调用 Qdrant payload update，不重算向量、不改变
point/chunk ID。批量 payload 更新与 PostgreSQL 不是分布式事务；中途失败
可能已经完成部分批次，命令会报错，按同一映射重复执行即可收敛。审计被
阻塞时应停止写入并先修正明确的 PostgreSQL 登记/所有权冲突；不能填入猜测的
KB。登记使用 `external_reference` 文档语义，不复制源 PDF，也不会改变普通
上传文档的 managed storage 约束。
