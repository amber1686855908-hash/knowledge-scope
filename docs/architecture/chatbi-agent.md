# ChatBI Agent 与结果分析

本文件记录 ChatBI 从自然语言问题到结果说明的有界编排。它复用已有的数据源注册、
PostgreSQL schema discovery、NL2SQL 校验和只读执行服务，不建立第二套数据源、SQL 校验或
执行路径。

## 数据流

```text
question
  -> registered datasource lookup
  -> trusted schema discovery
  -> NL2SQL LLM call
  -> SQLCandidate（不可信）
  -> fresh datasource-bound AST / policy validation
  -> read-only PostgreSQL execution
  -> bounded JSON-safe result
  -> ChatBI result-analysis LLM call
  -> answer + safe execution metadata
```

每次生成的 `SQLCandidate` 都会通过 `SQLExecutionService` 重新发现 schema 并校验。内部
`ValidatedSQL` 只是校验结果和审计投影，不是授权 capability；调用方不能用它、序列化后的
对象或 caller-owned `SchemaSnapshot` 跳过这条路径。A5.4 的执行服务是唯一的实际 SQL 执行
入口。

## 有界循环

Agent 不是无界 ReAct 循环。默认设置如下，均可通过 `Settings` 调整且有上限：

- `chatbi_agent_max_sql_attempts=2`：所有 SQL 生成尝试（初次和修复）总数；
- `chatbi_agent_max_repair_attempts=1`：允许的修复次数；
- `chatbi_agent_max_steps=6`：生成、执行和分析动作总数；
- `chatbi_agent_max_llm_calls=3`：SQL 生成和结果分析的网关调用总数；
- `chatbi_analysis_max_tokens=512`：结果分析的输出预算。

只有 malformed model output、SQL parse error、unknown table 和 unknown column 会触发一次
有界修复。修复提示中的上一候选和受控错误信息以 JSON 数据传入。安全策略拒绝、执行错误、
超时和 provider 错误不会被伪装成可绕过的修复建议；provider 级重试仍由 LLM Gateway 负责，
重试可能产生重复 provider 工作或费用，不承诺 exactly-once。网关已完成但结构化输出解析
失败时，Agent 仍会合并该次调用的 provider、model 和 token usage；provider 调用本身失败则
遵循网关既有的失败记录语义。

## 结果分析

结果分析使用现有 LLM Gateway 的 `task_type=chatbi_analysis` 和版本化提示
`a5.5-v1`。传给模型的内容只有用户问题、经过字面量脱敏的 SQL、列元数据、已经规范化且有界
的行数据，以及 `row_count`、`truncated` 和截断原因。问题和结果都作为确定性 JSON 数据编码；
单元格内容不能改变提示结构，也不能成为指令。

模型只能返回严格 JSON：

```json
{"answer":"...","warning":null}
```

空结果会被提示为“没有匹配数据”，截断结果必须说明结果不完整。解析失败不会产生答案，
而是返回 `analysis_failed`。Agent 不保留 chain-of-thought。

## 返回与审计

`ChatBIResult` 返回：

- `answer`、`datasource_id`、`query_id` 和 `execution_status`；
- AST 脱敏的 `redacted_sql`、列元数据、行数和截断信息；
- SQL/修复尝试次数、LLM 调用及 token 汇总；
- `warnings`、受控错误类别/消息和不含 prompt、原始 provider 响应或 SQL 字面量的 trace。

执行成功但结果分析失败时，`execution_status` 仍为 `succeeded`，同时返回
`error_category=analysis_failed`，以区分 SQL 已成功执行和答案分析未完成。分析调用失败不
改变已经发生的数据库读取，也不把失败伪装成成功答案。审计与 usage 记录由各自已有的
服务负责；它们与外部 PostgreSQL 读取不构成分布式原子事务。

## CLI 与边界

```bash
knowledgescope chatbi ask <datasource_id> <question>
```

CLI 只接受注册数据源 ID 和问题，输出 `ChatBIResult` 的安全投影；无通用 raw SQL 或
`ValidatedSQL` 输入参数。当前实现停在 SQL 生成、只读执行和结果分析，不包含 MCP、额外的
外部工具循环、前端 ChatBI 页面或外部数据写入。
