# ChatBI 只读 SQL 执行

本文件记录 PostgreSQL 的只读执行边界。它只负责执行已经通过 A5.3
AST/policy 校验的查询并返回规范化表格结果。ChatBI Agent 会复用这条边界；本文件不负责
SQL 生成、MCP、结果分析或前端 ChatBI 页面。

## 信任边界

生产执行入口接收注册数据源 ID 和未受信任 SQL：

```text
datasource_id + raw SQL / SQLCandidate
        ↓
KnowledgeScope registry lookup
        ↓
trusted PostgreSQL schema discovery
        ↓
SemanticSchemaContext + 当前 SchemaSnapshot
        ↓
A5.3 AST / QueryPolicy 校验
        ↓
内部 ValidatedSQL（仅作校验结果和审计投影）
        ↓
PostgresExecutionAdapter
        ↓
bounded JSON-safe QueryExecutionResult
```

执行服务没有接收调用方 `ValidatedSQL` 的公开入口，也没有接收调用方
`SchemaSnapshot` 的入口。`ValidatedSQL` 是 Python 内部模型，不是授权 capability；
执行服务在每次执行前根据 `datasource_id` 重新查找数据源、重新 discovery，并重新运行
validator。适配器还检查 datasource、validator version、policy fingerprint、PostgreSQL
dialect 和有界结果标记。未来的执行扩展也必须回到这条路径，不能执行调用方提交或反序列化的
`ValidatedSQL`。

## PostgreSQL 只读保证

适配器为每次查询创建独立连接，并同时使用：

- asyncpg `default_transaction_read_only=on` 连接设置；
- `transaction(readonly=True)` 产生的数据库只读事务；
- `SET LOCAL statement_timeout` 设置的有界语句超时；
- `SET LOCAL search_path TO "pg_catalog", <approved schemas>, "pg_temp"`。

`search_path` 只由已经验证的 `QueryPolicy.allowed_schemas` 生成，标识符会按 PostgreSQL
规则转义，调用方不能传入路径片段。A5.3 会把物理表解析为已授权的 schema-qualified
引用；`pg_catalog` 放在前面，临时 schema 放在最后。数据库角色仍应在部署侧没有写权限，
因为应用层的只读设置不能替代 PostgreSQL 权限。

连接建立、查询、结果读取、取消和异常路径都会关闭连接；事务上下文负责回滚未完成的
查询。超时、连接不可用、执行失败和结果规范化失败分别映射到安全的 `ChatBIErrorCategory`，
不会把 DSN、密码、堆栈或驱动原始错误返回给 CLI。

## 结果边界

A5.3 validator 会在规范化 SQL 中注入或裁剪顶层 `LIMIT`。适配器仍使用游标读取
`max_rows + 1` 的探测上限，在应用边界保留最多 `QueryPolicy.max_rows` 行，并返回：

- `columns`：列名、驱动类型（不可得时为 `unknown`）、是否可空和序号；
- `rows`：按列顺序排列的 JSON-safe 值；
- `row_count`、`max_rows`、`truncated`、`truncation_reason` 和 `duration_ms`；
- `query_id`、`datasource_id`、终态和安全错误信息。

支持 null、布尔、整数、有限浮点数、Decimal（字符串表示）、文本、UUID、日期/时间/时间戳、
JSON/JSONB 和数组。默认策略还限制：

- `max_result_bytes=4_000_000`：规范化 `rows` 使用紧凑 JSON、`ensure_ascii=False`、稳定对象键顺序
  编码为 UTF-8 后的总字节数；
- `max_cell_bytes=1_000_000`：单个规范化单元格采用同一 JSON 编码后的字节数；
- `max_nested_value_depth=32`：JSON 对象/数组容器层级上限；
- `max_collection_items=10_000`：单个 JSON 对象或数组的项目数上限。

这些值可通过 `KNOWLEDGE_SCOPE_CHATBI_MAX_RESULT_BYTES`、
`KNOWLEDGE_SCOPE_CHATBI_MAX_CELL_BYTES`、`KNOWLEDGE_SCOPE_CHATBI_MAX_NESTED_VALUE_DEPTH` 和
`KNOWLEDGE_SCOPE_CHATBI_MAX_COLLECTION_ITEMS` 配置。结果按游标逐行规范化，不会先把完整结果
加载到内存。行数或总字节数超限时只在完整行边界停止，并分别返回
`truncation_reason=row_limit` 或 `truncation_reason=payload_bytes`；没有行内容被静默切片。
单元格过大、嵌套过深或集合过大时查询失败，而不是返回部分单元格。单个超大 PostgreSQL 值
可能在驱动交给应用前已经到达客户端，这是应用边界无法消除的残余限制。

空结果仍是完整结果；如果结果恰好落在字节上限内，也不会标记为截断。结果不承诺无界读取，
也不执行第二条调用方语句。

## 生命周期与一致性

一次请求的审计状态按 `created → validated → executing → succeeded` 推进；校验拒绝进入
`rejected`，超时/执行/规范化错误进入 `failed`，取消进入 `cancelled`。执行专用的
`SQLExecutionAuditRecord` 包含 SQL fingerprint、当前 schema/policy/validator 版本、经过 AST
脱敏的原始和规范化 SQL、行数、截断状态/原因、时长和安全错误类别。审计字段保留原有字段名
以兼容已有调用方，但绝不保存 SQL 原始字面量；字符串、数字、布尔和日期类字面量会被替换，
解析或渲染失败则返回固定的脱敏不可用标记。fingerprint 只用于请求关联，不用于还原 SQL。
`SQLExecutionOutcome` 返回完整的 request-scoped audit；也可通过注入的
`ExecutionAuditRecorder` 转发到应用自己的审计系统。审计记录不是分布式事务的一部分，
审计 sink 失败不会改变查询的执行语义。

注册数据源、外部 schema discovery 和外部 PostgreSQL 查询跨越不同存储边界，不构成分布式
原子事务。执行本身不写业务库；后续若增加持久化审计或结果脱敏，需要单独定义失败恢复策略。

## 开发命令

```bash
knowledgescope chatbi execute <datasource_id> "SELECT ..."
```

命令只通过 KnowledgeScope 应用数据库中已注册且启用的数据源执行，SQL 作为未受信任输入，
不会调用 LLM；命令输出中的 SQL 只显示经过字面量脱敏的结构。`chatbi nl2sql` 仍只生成并校验
SQL，不会隐式执行模型输出，且其 CLI 展示同样不保留字面量。

`knowledgescope chatbi ask <datasource_id> <question>` 会在同一个注册数据源、discovery、
校验和只读执行边界上运行有界 Agent，并在执行成功后调用结果分析；它不会把调用方提交的
`ValidatedSQL` 当作授权凭证。
