# A5.1 ChatBI / NL2SQL 领域基础

本文件记录 ChatBI 的 provider-independent 领域契约、数据源元数据和安全策略基础。当前可以
发现外部 PostgreSQL schema、生成并静态验证只读 SQL，也可以通过独立的 A5.4 执行服务运行
经过同一安全校验边界的只读查询。

## 当前边界

`knowledge_scope.chatbi` 提供以下可复用契约：

- `DataSource` 描述稳定 ID、显示名称、`postgresql` dialect、启用状态、可选默认 database/schema 和时间元数据；
- `connection_ref` 只能使用 `env:NAME` 或 `secret:NAME` 形式的 opaque reference，指向外部管理的连接/凭据配置，不是数据库 URL，也不保存明文密码；
- `QueryPolicy` 默认只读、最多 1,000 行、30,000 ms statement timeout、仅允许 `public` schema、默认不允许 views，并固定 `max_statement_count=1`；
- `QueryExecutionRequest` 只表示执行意图（datasource、question 和审计 metadata），不接受 raw SQL；`QueryExecutionResult` 定义执行 adapter 返回的规范化列/行结果、截断状态、时长和安全错误类别；结果不会携带 SQLAlchemy row 对象；
- `QueryLifecycleState` 固定了 `created`、`validated`、`rejected`、`executing`、`succeeded`、`failed`、`cancelled` 等最小审计状态；`QueryAuditRecord` 要求拒绝、失败和取消状态带有错误类别，成功和进行中状态不得带错误类别；
- `/api/v1/chatbi/data-sources` 仅管理数据源安全元数据，响应不会返回 `connection_ref`；SQL 执行使用受控的开发者 CLI，不提供通用的未认证 `/execute-sql` endpoint。

数据源元数据保存在 KnowledgeScope 应用数据库的 `chatbi_data_sources` 表中；被查询的业务数据库仍通过
外部 connection reference 管理。schema discovery 和 SQL execution 都只在需要时解析该引用，不把
外部数据库连接配置复制进应用模型。

数据源接口对无效请求返回 422，但会去除请求体和验证上下文中的原始输入，避免把误提交的 URL、密码或其他
凭据片段回显到 API 响应；这不替代调用方自身的安全日志策略。

## SQL 安全边界

`chatbi-nl2sql-safety.md` 中的 validator 使用 PostgreSQL AST 拒绝 `INSERT`、`UPDATE`、`DELETE`、
`MERGE`、`DROP`、`ALTER`、`CREATE`、`TRUNCATE`、`GRANT`、`REVOKE`、`COPY`、多语句和事务控制语句，
只接受经过完整策略检查的读查询（包括安全的 `WITH`）。这里的规则由 `sqlglot` AST、权威
`SchemaSnapshot` 和 `QueryPolicy` 共同执行；正则表达式不作为 SQL 安全校验。

## 本地数据源夹具

[`tests/fixtures/chatbi_demo.sql`](../../tests/fixtures/chatbi_demo.sql) 是可重复的业务示例数据。它应加载到单独的
PostgreSQL demo/test database（或明确隔离的 demo schema），而不是 KnowledgeScope 应用数据库，因此未来
ChatBI 查询不会意外读取 `knowledge_bases`、`documents` 等内部表。应用不会自动加载或执行该夹具。

## Schema discovery

当前的只读 PostgreSQL schema discovery、快照 fingerprint 和 `SemanticSchemaContext` 见
[`chatbi-schema-discovery.md`](chatbi-schema-discovery.md)。该能力仍不执行 SQL，也不读取业务行；
本文件中的 A5.1 数据源和 `QueryPolicy` 契约继续作为它的输入边界。

## 尚未覆盖的边界

MCP 和面向更复杂任务的工具编排仍未实现。有界 ChatBI Agent 的编排与结果分析见
[`chatbi-agent.md`](chatbi-agent.md)；SQL 执行的边界见
[`chatbi-sql-execution.md`](chatbi-sql-execution.md)。后续实现必须继续通过 `DataSource`、
`QueryPolicy`、生命周期和安全错误契约。
