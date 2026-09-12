# A4.4 统一多检索器候选池与最终重排

Phase A4.4 将现有的四条检索分支编排到一个独立服务中：

```text
Dense chunk ─┐
Sparse chunk ─┤
Graph Evidence ─┼─→ 有界候选池 → 一个本地 BGE final reranker → 最终结果
Multimodal Evidence ─┘
```

本阶段不复制分支内部算法。Dense 复用 Qwen embedding、Qdrant 和已有
reranker，Sparse 复用 A4.3 SQLite BM25，Graph 复用 A3.4 bounded graph
retrieval，多模态分支复用 A4.2 representation index。A4.4 只负责并发编排、
候选统一、最终重排和结果解释；不修改 A2/A3 冻结评测、原有 RRF 或 Dense
默认行为。现有 RAG endpoint 通过显式的 `retrieval_mode=unified` 选择本服务，
未指定时仍使用原有 `dense` 路径。

## 统一候选和身份

chunk 候选的语义身份是：

```text
(knowledge_base_id, document_id, chunk_id)
```

Evidence 候选的语义身份是：

```text
(knowledge_base_id, document_id, evidence_id)
```

应用使用带 schema version 的稳定 JSON（固定 key 顺序和 separators、UTF-8）
计算 SHA-256 候选 ID。Dense、Sparse、Graph 的结果按 chunk 身份合并；
Multimodal 结果保留为 Evidence 级候选，不把多个 representation 误展平成
多个 source evidence。每个候选保留 document/page/source block/section/asset
lineage，以及命中的分支、分支 rank/score、Graph seed/path、Evidence ID 和
representation ID。冲突的 lineage 或文本不会静默选边，而会使该分支进入失败
状态。

Graph 结果没有重复复制 chunk 文本；Graph-only 候选会通过现有
`knowledgescope_chunks_v1` 的只读、KB/document/chunk scoped lookup 获取当前
chunk 文本。该查询不会修改冻结 chunk collection。

## 候选池和最终排序

四条分支在独立线程中并发调用，以避免同步的本地模型、Qdrant、SQLite 和
Neo4j 操作阻塞 async 调用方。每条分支、候选池和最终结果都有显式上限；候选
池使用固定分支顺序的 round-robin，避免某条高产分支一次性耗尽池容量。随后
只把 bounded 的 `rerank_text` 发送给最终 `BAAI/bge-reranker-v2-m3`，最终顺序
按 reranker score 降序、候选 ID 升序稳定排序。

A4.4 不把 Dense/Sparse/Graph/Multimodal 的原始 score 放在同一数值空间相加，
也不在本阶段再次使用 RRF。A3.5 的 RRF 仍由原服务独立负责；A4.4 的分支
rank/score 仅用于审计和可解释性，最终 reranker 才是最终排序依据。RRF 公式
仍为 A3.5 的既有约定：`1 / (rrf_k + rank)`，但 A4.4 不改变或重新调节
`KNOWLEDGE_SCOPE_HYBRID_RRF_K`。

## 失败、降级和取消

每个分支区分 `success`、`empty`、`failed`、`timed_out` 和 `cancelled`。空结果
表示分支正常完成但没有候选；失败不会被伪装成空结果。

- `degraded`（默认）：保留成功分支，结果中的分支状态标出失败或超时；全部
  分支失败时返回错误。
- `strict`：任一分支失败或规范化失败，整体返回错误。

调用方取消时，取消会向上游传播并取消尚未完成的 async 任务；已经在线程中
运行的本地推理不能被强制终止。Qwen embedding 和 BGE reranker 由进程级实例
复用，并使用现有 adapter 的 inference lock；这保证模型使用安全，但不承诺
多请求 GPU 并行吞吐。失败分支的部分规范化候选不会进入共享候选池。

## RAG 接入边界

`POST /api/v1/rag/query` 保留原有 `dense` 默认路径；调用方只有在请求中提供
`retrieval_mode=unified` 且指定 `knowledge_base_id` 时才会进入 A4.4。Unified
返回的最终候选直接进入现有 context selection，不会再次执行 RRF 或额外的
reranker。chunk 候选只使用权威 chunk 文本；Evidence 候选使用已建立的可搜索
representation 作为上下文载体，同时在 citation 中保留 `evidence_id`、modality、
representation IDs 和原始 document/page/source block lineage，不生成虚假的
`chunk_id`。分支失败状态会进入 `complete` 的检索元数据；`strict` 或全部分支
失败时请求结束为受控错误，`degraded` 仅在仍有可用上下文时继续。没有可用上下文
时不会调用 LLM。

## 配置和命令

候选上限、字符级 rerank 文本上限和模式通过以下 Settings/env 配置：

```text
KNOWLEDGE_SCOPE_UNIFIED_DENSE_CANDIDATE_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_SPARSE_CANDIDATE_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_GRAPH_CANDIDATE_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_MULTIMODAL_CANDIDATE_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_CANDIDATE_POOL_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_RESULT_LIMIT
KNOWLEDGE_SCOPE_UNIFIED_RERANK_TEXT_MAX_CHARS
KNOWLEDGE_SCOPE_UNIFIED_FAILURE_MODE
```

开发者查询命令：

```bash
uv run knowledgescope unified-search "查询内容" --knowledge-base-id <knowledge-base-uuid>
```

运行它需要本地 Qdrant、Sparse index、Neo4j 以及已有 embedding/reranker 依赖。
默认 `degraded` 适合本地开发时部分基础设施尚不可用的情况；需要所有分支
都成功时使用 `--failure-mode strict`。命令返回 bounded typed result，不生成
回答，也不调用外部 LLM Provider。

## 边界

本阶段没有新增 retriever、BM25 实现、视觉模型、OCR、caption、Graph fusion
算法或最终质量 benchmark。A4.2 的 representation index 和本地模型文件仍
属于运行时 derived state，不能提交到 Git。当前结果只能作为统一候选与最终
重排的功能基础，不能据此宣称多路检索质量提升。
