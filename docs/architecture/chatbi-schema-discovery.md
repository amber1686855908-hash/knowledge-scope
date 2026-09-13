# ChatBI Schema Discovery

当前 ChatBI 只负责发现外部 PostgreSQL 的结构元数据，结果用于后续 NL2SQL 的语义上下文。
它不执行 SQL，不读取业务行，也不把外部数据库的凭据或数据值写入快照。

## 数据流

```text
DataSource
  -> CredentialResolver
  -> PostgresSchemaInspector（只读事务）
  -> SchemaSnapshot
  -> SemanticSchemaContext
```

`DataSource` 继续使用 A5.1 的 `connection_ref`。`env:NAME` 从当前进程环境解析；
`secret:NAME` 只有在调用方显式提供 `SecretReferenceResolver` 时才可解析，默认会以安全错误失败。
解析后的 URL 只在内存中传给 adapter，不进入 API、CLI、日志、异常、fingerprint 或 schema artifact。

## Snapshot 契约

`SchemaSnapshot` 是 provider-independent 的 Pydantic v2 模型，版本为 `1.0`，包含：

- `datasource_id`、PostgreSQL `dialect`、实际 `database_name` 和 allow-list 中的 schemas；
- table/view、关系级 comment、column 名称、规范化 PostgreSQL type、nullable 和有限长度的 comment；
- ordered primary key、unique constraint 和有方向的 foreign key（source columns 指向 target columns）。

adapter 只查询 `pg_catalog` 的 relation、column、constraint、comment 元数据，并且以参数绑定的
schema allow-list 限制范围。`information_schema`、`pg_catalog`、`pg_toast`、`pg_toast_*`、`pg_temp`
和 `pg_temp_*` 等系统 schema 会在 catalog 查询边界被排除，永不输出；关闭 `allow_views` 时 view
也不输出。不会读取 column default、physical storage metadata 或业务行。

模型会按 schema、relation、column ordinal 和约束名称规范化排序，因此 discovery 顺序不会改变结果。
`SchemaSnapshot.to_deterministic_json()` 使用 UTF-8、固定 separators 和 sorted keys；其 SHA-256
`fingerprint` 会随结构或语义 comment 改变，但不包含时间戳或凭据。

## SemanticSchemaContext

context 使用确定性的纯文本格式表达完整 relation block。`max_chars` 是字符预算，不会在 table、
column 或 foreign-key block 中间截断。系统先选择完整的 relation block，再只保留两个端点都已选中的
foreign key；放不下的 relation 进入 `omitted_relations`，放不下或端点缺失的关系进入
`omitted_relationships`，并设置 `truncated=true`。`included_relations`、`omitted_relations`、
`omitted_relationships`、`max_chars` 和 snapshot fingerprint 会随结果返回，因此调用方不会把不完整
上下文误认为完整 schema。若预算连固定 header 都无法容纳，builder 会抛出
`SchemaContextBudgetError`，service 会转换为安全的 `schema_discovery_failed`，而不是返回空上下文。

## 连接与生命周期

每次 discovery 使用一个有连接超时和 statement timeout 的 PostgreSQL 连接，并在 `readonly=True`
事务中执行固定的 catalog 查询，完成或异常后关闭连接。这里的只读保证由 PostgreSQL connection
settings 与事务模式共同提供；数据库角色、网络代理等外部策略不在应用控制范围内。

schema discovery 结果是一次实时快照，不在 KnowledgeScope 数据库中持久化。数据库中的
`chatbi_data_sources` 只保存外部连接引用和安全元数据。没有分布式事务：外部数据库读取与应用元数据
之间不存在需要回滚的联合写入。

## 接口与边界

- `GET /api/v1/chatbi/data-sources/{datasource_id}/schema` 返回不含凭据的 `SchemaDiscoveryResult`；
- `knowledgescope chatbi schema <datasource_id>` 提供同一只读 discovery 的开发命令；
- 失败会归类为 `credential_resolution_failed`、`unsupported_dialect` 或 `schema_discovery_failed`，
  对外只返回固定的安全消息；
- `tests/fixtures/chatbi_demo.sql` 是隔离的 demo schema，集成测试只允许读取其 catalog 元数据。

本阶段没有 schema snapshot 数据库表，也没有 NL2SQL、SQL AST 校验、SQL 执行、MCP 或 Agent loop。
