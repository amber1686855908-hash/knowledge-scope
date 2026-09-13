# ChatBI NL2SQL 与 SQL AST 安全边界

本模块只负责把自然语言问题生成并验证为 `ValidatedSQL`。它不连接业务
数据库执行 SQL，也不提供结果分析、MCP 或 Agent loop。

## 流程与信任边界

生产生成入口只接受注册数据源 ID 和 `NL2SQLInput`；面向未来执行边界的校验入口
`validate_for_registered_data_source()` 只接受注册数据源 ID 和不受信任的
`SQLCandidate`。两条路径都不接受调用方提供的 snapshot：

```text
datasource_id + NL2SQLInput（question + 可选 model）
        ↓
KnowledgeScope registry lookup（注册的 DataSource）
        ↓
A5.2 只读 schema discovery
        ↓
SchemaDiscoveryResult（当前 snapshot + 已预算 context）
        ↓
LLMGateway（结构化 JSON：只接受 {"sql":"..."}）
        ↓
SQLCandidate（应用补充 provider/model/context fingerprint）
        ↓
sqlglot PostgreSQL AST + SchemaSnapshot/QueryPolicy 校验
        ↓
ValidatedSQL（内部校验结果与审计投影）
```

`SchemaSnapshot` 是可序列化的 metadata 模型，可用于测试和审计文件；它本身
不是数据库访问授权。`NL2SQLService` 的两个注册数据源入口都从 KnowledgeScope
registry 取得 `DataSource`，再重新执行受策略约束的 discovery；调用方不能传入
authoritative snapshot。低层 `_SQLSafetyValidator` 和
`_validate_sql_candidate()` 只作为模块内部的纯校验 helper，测试可以直接使用，
但它们不是生产授权入口。

`SQLCandidate` 是生成的 raw SQL 候选，任何 raw SQL 都必须经过
内部 AST validator。`ValidatedSQL` 是校验后的内部结果和审计投影，不是
不可伪造的授权 capability；Python 类型、冻结字段和私有工厂都不能承担密码学
信任。它不是 Pydantic 输入模型，没有 `model_validate` 或 JSON 反序列化入口。
当前没有 SQL executor；未来执行代码必须重新从 `datasource_id` 和不受信任的
SQL 候选进入上述 trusted validation path，不能接收调用方传入或反序列化的
`ValidatedSQL`。

## AST 校验范围

当前只支持 PostgreSQL，解析器为直接依赖的 `sqlglot`。根语句必须是单条
`SELECT`，或由只读 `SELECT` 组成的有界 `UNION`、`INTERSECT`、`EXCEPT`。
校验器拒绝：

- 多语句、DML、DDL、权限/事务/会话控制、`COPY`、`CALL` 等命令；
- 可写或递归 CTE、`SELECT INTO` 和行锁；
- `PIVOT`、`QUALIFY`、`TABLESAMPLE`、表 hint、其他未支持的动态表来源；
- `OFFSET`、`FETCH`、嵌套 `LIMIT`，以及 `LIMIT ALL`、参数、表达式或负数；
- 显式 `OPERATOR(...)`、未知/自定义二元或一元操作符和非 allow-list 函数；
- 除明确安全的内建 scalar 类型以外的 cast，包括限定类型和 user-defined type。

非递归 CTE 按 PostgreSQL 词法作用域校验：CTE 只能引用前面已经声明的兄弟
CTE；嵌套作用域不会泄漏到外层；不支持递归 CTE。所有物理表引用都必须在
当前 `SchemaSnapshot`、`SemanticSchemaContext` 和 `QueryPolicy` 中存在。未限定
表名只有在候选唯一时通过，并在规范化 SQL 中改写为显式、带引号的
`schema.table`；CTE 名称和别名保持逻辑引用，不会被改写。当前 view 即使被
discovery 发现也会被 validator 拒绝，直到完成 view dependency analysis。

缺少 `LIMIT` 时，校验器在 AST 上注入 `QueryPolicy.max_rows`；显式超出上限时
在 AST 上裁剪。规范化 SQL 会重新解析并完整复验，避免 parser 序列化丢失的
节点绕过策略。SQL 字符数和 AST 节点数都有上限；parser/遍历递归异常会转为
受控的安全错误，不向调用方返回 traceback。

## Schema 与 prompt

对象引用始终与 discovery 得到的权威 `SchemaSnapshot` 比对。schema context
预算先选择完整 relation，再只保留两个端点都在 context 中的外键关系；遗漏项
写入 omission metadata。NL2SQL prompt 只发送结构化 schema 名称、列、类型、
约束、允许的 schema 列表和允许的关系，并使用固定键序、紧凑 JSON 与
delimiter-safe escaping；标识符
中的引号、换行、反引号和类似指令的文本仍是 JSON 字符串数据。不发送 snapshot
中的 relation/column comments。comments 仍保留在 snapshot 和 fingerprint 中，
但不作为模型指令。

`QueryPolicy` 默认只读、最多 1,000 行、30,000 ms statement timeout、只允许
`public` schema，并固定单语句。规范化物理表名减少对 session `search_path`
的依赖；A5.4 的实际执行 adapter 仍必须设置固定且安全的 `search_path`，并在
执行前重新进入 trusted validation path，不能只检查 `ValidatedSQL` 的 Python 类型。

## 错误与后续边界

生成失败、模型输出格式错误、SQL parse error、policy violation 和对象/列
错误使用独立的 `ChatBIErrorCategory`。错误文本不包含 raw provider payload、
`connection_ref`、数据库 URL 或凭据。

`chatbi nl2sql` 是开发者 smoke-test：它先从注册数据源执行只读 schema
discovery，再生成和验证 SQL，输出安全的结构化结果；没有 SQL 执行 endpoint。
当前没有参数绑定、SQL execution adapter、结果行脱敏、MCP、Agent loop、
NL2SQL 质量评测或前端 ChatBI 页面。AST 通过只表示结构和策略检查通过，不能
证明业务问题一定得到正确回答。
