# MCP 服务与安全工具暴露

KnowledgeScope 通过官方 Python MCP SDK 提供本地 stdio 服务。MCP 在这里是协议适配层，
不会复制 ChatBI 的 SQL 生成、校验或执行逻辑：

```text
MCP client
  → MCP tools/call
  → registered datasource
  → schema discovery → A5.3 validation → A5.4 execution → A5.5 Agent
```

## 工具边界

服务只注册两个高层工具：

- `chatbi_ask(datasource_id, question)`：调用现有 `ChatBIAgentService`，返回安全的
  成功 `ChatBIResult`，包括答案、执行状态、有界结果元数据、脱敏 SQL、usage 和安全 trace；
  预期的 ChatBI 失败会使用 MCP 的 `isError` 工具结果和固定错误消息返回。
- `chatbi_schema(datasource_id)`：通过注册表和只读 Schema Discovery 获取有界的
  Schema Context 及其 fingerprint。MCP 投影不返回关系/字段注释，也不返回连接配置；
  关系、字段和外键会按固定上限选择，响应包含返回数、遗漏数、截断标志和有界的遗漏样本。

两个工具的输入都是严格的 JSON 对象，拒绝未知字段。MCP 调用不能提交
`SchemaSnapshot`、`ValidatedSQL`、规范化 SQL、连接串、凭据、策略或 Agent 预算覆盖。
服务不会暴露 `execute_sql`、`generate_sql`、`validate_sql`、`get_connection` 或其他
低层数据库工具。

## 可信边界与结果

`datasource_id` 只用于查询 KnowledgeScope 数据库中的已注册数据源；权威 schema 由当前
目标数据源的 Discovery 重新读取。`chatbi_ask` 继续经过 A5.3 的数据源绑定校验和 A5.4
只读执行，MCP 层没有第二条 SQL 路径。

工具返回结构化 JSON，并同时提供确定性序列化的文本内容。内部异常映射为稳定的错误类别
和固定的安全消息，不返回 DSN、凭据、provider 原始响应、堆栈或思维链。成功结果中的
SQL 仍沿用 ChatBI 的脱敏审计表示。

MCP schema 响应使用固定的文本和字节上限；遗漏元数据只保留固定数量的样本，完整数量通过
计数字段表达。数据库注释仍可保留在权威 `SchemaSnapshot` 中用于指纹和内部校验，但不会
进入 MCP schema 响应，也不会默认进入 `chatbi_ask` 的模型 prompt。

## 传输与生命周期

`knowledgescope mcp serve` 默认启动官方 SDK 的 stdio transport。它适合本地 MCP 客户端，
不会在 API 启动时自动开启网络监听。进程级 provider HTTP client、数据库 engine 在服务
结束时统一关闭；启动阶段只构造服务，不发起 provider 请求。每个进程默认最多并发处理
4 个 MCP 工具调用，可通过 `KNOWLEDGE_SCOPE_MCP_MAX_IN_FLIGHT` 调整到 1–32；等待、取消和
异常路径都会释放容量。问题长度、Agent 步数/调用数、SQL 与结果大小继续使用 ChatBI
已有上限。

当前版本使用官方 SDK 的 stdio reader；该 SDK 没有提供应用层可配置的预解析 frame-size
钩子，因此 oversized frame 可能先由 SDK 解析，再由工具参数校验拒绝。应用仍保持严格
输入模型和有界输出，不会因此开启自定义传输或网络监听。

跨 PostgreSQL、provider 和 MCP 客户端的操作不构成分布式原子事务；各层继续遵循既有
超时、错误、只读和结果大小限制。MCP 协议测试使用官方 SDK 的 client/session，另外有
隔离 PostgreSQL demo 数据库上的完整工具调用测试，验证 discovery、校验、执行和结果
规范化链路。

后续工具若要扩展，必须重新经过数据源注册、可信 schema discovery 和对应领域服务的
边界评审，不能把内部对象直接变成 MCP 输入。
