# A2.7 RAG QA

KnowledgeScope 当前提供一个最小的 RAG QA 编排：

默认路径为 `query → Qwen/Qwen3-Embedding-0.6B → Qdrant dense Top-10 → BGE reranker → bounded context → LLM Gateway`。
请求显式指定 `retrieval_mode=unified` 时，改为复用 A4.4 的统一候选池和最终重排，
再进入同一套 context、citation 和 LLM Gateway 流程。

## 当前实现

- `RAGService` 默认复用现有 dense retrieval、`RerankingService` 和 async `LLMGateway`；`unified` 模式复用现有 `UnifiedRetrievalService` 的四条分支和最终候选，不在 RAG 层重复检索或重排。不重新运行 MinerU，也不引入 LangChain。
- 默认检索 10 个 dense candidates，再用显式的 `bge-reranker-v2-m3` 重排；本地 embedding、Qdrant I/O 和 reranker 在 `asyncio.to_thread` 中执行，不占用 FastAPI event loop。每个进程内的模型 adapter 通过小粒度线程锁串行化同一模型的本地推理；RAG service 也会串行化一次 context selection。当前不承诺 GPU 并发吞吐。
- Dense 默认路径只把有文本的 chunk 放入回答上下文；Unified 路径还可以放入有界的 Evidence representation 文本。上下文按对应检索结果顺序选择，并受 `KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS` 字符预算限制；完整文本放不进剩余预算的候选会被跳过，因此不会为未发送的内容生成 citation。source block 重叠本身不等于重复，因为 A1.6 可能把同一 source block 拆成多个不重叠 chunk；只有同一 lineage 下的精确规范化重复文本会被抑制。没有可用文本时直接返回受控的“当前检索到的资料不足以回答该问题”，不会调用 LLM。
- 字符预算只约束选中 chunk 文本的字符数，不包含 system/user prompt、marker 和协议包装，也不使用 tokenizer；它是近似的工程上限，不是精确的 LLM token context budget。当前没有 source block 的字符/token offset，因此无法可靠地自动消除所有近似重复内容。
- 每个选中的 context item 都生成确定性的请求内 marker（`C1`、`C2`……）。marker 和 document、page、chunk 或 Evidence、source block、section 元数据由应用生成并在最终 SSE citation event 中返回；模型输出中的未知、重复或格式错误 marker 不会被解析为 citation metadata，应用只信任自己的 citation event。
- `unified` 模式下，chunk 候选仍引用真实 `chunk_id`；image、table、formula 等 Evidence 候选引用真实 `evidence_id` 和 `representation_ids`，不伪造 `chunk_id`。representation 文本只是送入模型的可检索上下文载体，权威 citation lineage 仍来自当前 Evidence/Representation 状态；Unified branch 的 rank、score、Graph seed/path 会在 citation provenance 中保留。
- prompt 版本为 `rag-qa-v1`，要求回答只依据给定资料、证据不足时明确说明、不得编造，并只能引用上下文中的应用 marker。

## SSE API

```http
POST /api/v1/rag/query
Content-Type: application/json

{"query":"你的问题","knowledge_base_id":null,"document_id":null,"retrieval_mode":"dense"}
```

要选择 A4.4 统一路径，必须显式指定知识库：

```json
{"query":"你的问题","knowledge_base_id":"<knowledge-base-uuid>","retrieval_mode":"unified"}
```

未知 `retrieval_mode` 会被拒绝；`unified` 缺少 `knowledge_base_id` 也会被拒绝。
未指定该字段的既有客户端继续使用 `dense`。

响应为 `text/event-stream`，事件顺序如下：

1. `answer_delta`：增量回答文本；
2. `citations`：应用生成的 citation metadata；
3. `complete`：`completed`、`insufficient_evidence` 或 `error` 状态，以及 provider、model、token 和端到端延迟（可用时）。

成功请求的顺序是 `answer_delta* → citations → complete`，且只发送一次 `complete`。检索、模型、provider 或 usage 持久化失败会发送受控的 `error`，再发送 `status=error` 的终止 `complete`，不会伪装成成功；客户端取消会传播取消信号并关闭下游 async generator。`complete` 中的 `retrieval_latency_ms`、`llm_latency_ms` 和 `latency_ms` 分别表示 context selection、LLM gateway stream 消费和端到端 wall-clock 范围（后者包含前两阶段及调度开销）；LLM token 字段按 provider 可用性填写。LLM usage 仍由 A2.6 gateway 尝试写入 PostgreSQL，provider 调用、检索和 usage 写入不是原子事务。

`query` 最多接受 `4,000` 个字符；SSE event payload 使用严格的 Pydantic 模型校验。当前 endpoint 没有认证或限流，适用于受信任的本地/内网开发环境，不应直接暴露到公网。客户端取消可以阻止后续 LLM 调用并关闭 provider stream，但已经提交给 `asyncio.to_thread` 的同步 embedding、Qdrant 或 reranker 工作无法被 Python 线程强制中断，可能运行到当前调用结束。

## 配置

上下文和生成预算通过 `Settings` / `.env` 配置：

```dotenv
KNOWLEDGE_SCOPE_RAG_CANDIDATE_LIMIT=10
KNOWLEDGE_SCOPE_RAG_RERANK_LIMIT=5
KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS=6000
KNOWLEDGE_SCOPE_RAG_MAX_TOKENS=512
```

默认 provider 仍需要按 [A2.6 LLM gateway 说明](llm-gateway.md) 配置 `KNOWLEDGE_SCOPE_LLM_API_KEY`。正常测试使用 fake retrieval/reranker/gateway，不需要 GPU、网络或 API key；当前没有前端聊天页面、答案质量 benchmark 或 GraphRAG。

真实运行还需要安装已有的本地模型依赖：

```bash
uv sync --group embedding-benchmark --group reranker-benchmark
```
