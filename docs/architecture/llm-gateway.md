# A2.6 LLM Gateway

KnowledgeScope 当前提供一个小型、provider-independent 的 async LLM gateway，为后续 RAG、GraphRAG、Agent 和报告生成保留统一调用边界；这些业务能力尚未实现。

## 当前实现

- `knowledge_scope.llm.schemas` 定义严格的 `LLMRequest`、`LLMMessage`、`LLMResult`、`LLMStreamEvent` 和 `LLMUsageRecordInput`。消息角色目前为 `system` 或 `user`，任务标签为 `rag_answer`、`graph_extraction`、`report_generation`、`agent` 和 `evaluation`。
- `LLMGateway` 接收注入的 provider 和 usage recorder，支持普通 completion 与 streaming。completion 结果统一包含 `text`、`provider`、`model`、token 用量、毫秒延迟和可用的 `finish_reason`。
- `DeepSeekProvider` 通过 OpenAI-compatible `/chat/completions` 接口工作。`base URL`、API key、model、timeout 和 retry 上限全部来自 `Settings`；API key 使用 `SecretStr`，不会写入异常消息、usage 记录或 CLI 输出。
- 网络 timeout、连接失败、HTTP API 错误、响应结构错误和取消会转换为稳定的错误类别。默认不重试；仅显式配置的 `KNOWLEDGE_SCOPE_LLM_MAX_RETRIES` 会对 timeout、连接错误、429 和 5xx 做有限重试。streaming 不自动重试，以避免部分输出后重复内容。重试不是 exactly-once：timeout 等情况下 provider 可能已经完成工作，重试可能重复 provider-side work/cost；usage 记录按一次逻辑调用保存，不能替代 provider 端的 attempt 或账单明细。
- 每次已执行的逻辑调用都会尝试写入 PostgreSQL 的 `llm_usage_records`，失败调用也会记录；未发生 provider 请求的 configuration error 不产生调用记录。成本只有在输入/输出 token 和两种单价都由 Settings 配置时才估算，代码不内置供应商价格。provider 调用与 usage 写入不在同一事务，属于非原子、fail-closed 的一致性边界：usage 记录失败会向调用方报告，不会静默返回成功；取消语义优先保留为 `CancelledError`，记录失败会作为其异常原因保留。

## 配置

复制 `.env.example` 为 `.env`，按需设置：

```dotenv
KNOWLEDGE_SCOPE_LLM_PROVIDER=deepseek
KNOWLEDGE_SCOPE_LLM_BASE_URL=https://api.deepseek.com
KNOWLEDGE_SCOPE_LLM_API_KEY=你的本地密钥
KNOWLEDGE_SCOPE_LLM_MODEL=deepseek-chat
KNOWLEDGE_SCOPE_LLM_TIMEOUT_SECONDS=60
KNOWLEDGE_SCOPE_LLM_MAX_RETRIES=0
```

可选的 `KNOWLEDGE_SCOPE_LLM_INPUT_COST_PER_1K_TOKENS` 和 `KNOWLEDGE_SCOPE_LLM_OUTPUT_COST_PER_1K_TOKENS` 用于成本估算；留空时 `estimated_cost` 为 `null`。不要提交 `.env` 或真实密钥。

执行迁移并运行一次真实 provider smoke test：

```bash
uv run alembic upgrade head
uv run knowledgescope llm-smoke-test --task-type evaluation
```

正常单元测试通过 mock HTTP transport、fake provider 和测试数据库验证，不会发起网络请求。当前没有 LLM API endpoint、RAG prompt 编排、对话历史、流式 WebSocket 或前端聊天 UI。
