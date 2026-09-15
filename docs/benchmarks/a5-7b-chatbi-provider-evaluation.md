# ChatBI 评测 v2 Provider 基础设施

本文记录冻结的 ChatBI 评测 v2 在真实 provider 评测前需要的本地基础设施。本文不
记录 provider 质量结果；真实评测必须在本地 preflight 通过后，显式运行 DEV 命令。

## 冻结输入

评测输入来自 `docs/benchmarks/a5-7-chatbi-eval-v2.json`，状态为
`human_reviewed_frozen`。数据集包含 80 条问题，其中 DEV 50 条、TEST 30 条，数据集
指纹为：

`60c75c597da8fc71a0fa5b25d335b63410b44a4ab3a403da40ca72c5ae375ab3`

当前 provider runner 只允许 `--split dev`，因此一次运行固定选择 50 条 DEV 问题。
`--split test` 会明确失败，不会把 TEST 当作调参数据，也不会产生 TEST provider 结果。

fixture 为 `tests/fixtures/chatbi_demo_v2.sql`，其 SHA-256 为：

`fd972106c39c7a8b31b57975118708e213a32e4e008ee13fafc15b1ea1b5182d`

fixture 只加载到独立数据库 `knowledgescope_chatbi_eval_v2` 的
`chatbi_demo` schema。数据库中有 `customers`、`regions`、`sales` 三张表和
`region_sales` 视图；preflight 还会通过受信任的只读执行链校验表数据指纹：

`58dce6bae9c916a218b7ca57861a337f050658caf3f73163ac2df17e51c5e55d`

旧的 v1 fixture 和 v1 评测命令保持不变。

preflight 还会从 PostgreSQL catalog 计算冻结的 schema 指纹：

`592681fd7c63ee654f87ecfac62455536f90c994fc74fc4e00973743be607071`

该指纹覆盖 `chatbi_demo` 中的表和视图、列顺序与类型、可空性、主键、唯一约束、外键及
CHECK 约束。表/视图集合、结构指纹和数据指纹必须同时匹配；catalog OID、约束名称等不稳定
元数据不参与计算。应用数据库与固定评测数据库的标准化 `(server, port, database)` 身份也
必须不同，冲突会在任何 fixture DDL 之前 fail closed。

## 权威数据源注册

preflight 会在本地 PostgreSQL 中创建或校验上述隔离数据库，并在 KnowledgeScope
应用数据库的 `chatbi_data_sources` 表中注册一个固定数据源：

- display name：`KnowledgeScope ChatBI evaluation v2`
- datasource ID：`a0e1eff3-45ad-512a-8dc4-71ab7cbe2125`
- connection reference：`env:KNOWLEDGE_SCOPE_CHATBI_EVALUATION_V2_DATABASE_URL`
- default database：`knowledgescope_chatbi_eval_v2`
- default schema：`chatbi_demo`

connection reference 是不透明的 registry 元数据，实际数据库 URL 只在当前进程环境
中供凭据解析器使用，不会写入评测产物、日志或 Git。注册过程按固定 ID、名称和
reference 查找；已存在的记录必须完全匹配，否则 fail closed。重复执行只校验已有
fixture 和记录，不重载或改写业务数据。

fixture 数据库名是固定的本地开发标识。如果已有同名数据库内容不是冻结 fixture，
preflight 会失败；不会猜测、覆盖或把错误数据库当作评测数据源。

## Preflight 与运行命令

先完成应用数据库迁移，然后运行不构造 provider、也不发起 provider 请求的 preflight：

```bash
uv run alembic upgrade head
uv run knowledgescope chatbi eval-v2-provider \
  --split dev \
  --preflight
```

preflight 顺序为：捕获 Git 开始状态 → 冻结数据与 fixture 校验 → DEV split/TEST 拒绝 →
provider 配置检查 → 应用库/评测库身份隔离检查 → 隔离数据库校验 → registry 查找 → 真实
PostgreSQL schema discovery → catalog schema 指纹校验 → `SemanticSchemaContext` 构建 →
受信任的 v2 fixture 数据探针 → ignored 产物路径检查。输出中的 `provider_calls` 固定为 `0`。

provider 访问恢复且 preflight 通过后，才可显式运行：

```bash
KNOWLEDGE_SCOPE_LLM_MODEL=deepseek-v4-flash-vision-exp \
uv run knowledgescope chatbi eval-v2-provider \
  --split dev
```

该命令复用正式链路：

问题 → registry 数据源 → trusted schema discovery → `SemanticSchemaContext` →
NL2SQL gateway → `SQLCandidate` → A5.3 AST/policy validation → A5.4 只读执行 →
结果归一化 → A5.5 分析 gateway → `ChatBIResult`。

reference SQL、期望结果和结构化答案事实只在评测器中用于运行后比较，不会进入
问题 prompt 或分析 prompt。负例也只能在 provider 调用完成后按结果判断是否安全拒答。

## 产物和口径

默认输出为 `data/evaluation/a5-7/provider/chatbi-eval-v2-dev.json`，属于运行时目录，
由 Git 忽略。产物只包含条目状态、受限的已脱敏 SQL、阶段/时延/usage 摘要、冻结输入
指纹和 Git/provider 配置元数据（Git revision/dirty 状态取自开始阶段），不包含 DSN、密码、API key、provider 原始响应、完整
oracle SQL 或完整结果集。provider 运行被中断时不写入可被误用的部分结果。

主要指标是正例的 Execution Accuracy（运行结果与冻结归一化结果的比较）。结构化结果
fact coverage 是辅助指标；修复次数、token 和时延只报告实际运行观察。没有人工答案
标注时，不把这些数据解释成自然语言答案准确率、精确率或召回率。

v2 的 schema/context policy 固定只允许 `chatbi_demo`，同时沿用 Settings 中的只读、
行数、结果大小和超时边界。provider key 仍由本地 `.env` 或环境变量提供；`.env`、模型
文件和运行产物不得提交。
