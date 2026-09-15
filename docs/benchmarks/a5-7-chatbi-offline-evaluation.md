# A5.7 ChatBI 离线评测

## 目的与边界

本评测记录 ChatBI 现有链路的可重复检查：

`问题 → Schema Discovery → NL2SQL → 校验 → 只读执行 → 修复（如适用） → 分析 → 结果`

评测停留在 A5.5 已实现的 Agent 链路，不增加新的 SQL 能力、执行策略或前端功能。
离线模式使用仓库内的显式脚本结果，只检查评测器、结果比较和失败分类的基础设施，不能
作为模型质量、SQL 正确率或回答质量的实测结论。provider 模式必须由操作者显式选择，
不会自动调用外部 provider。

## 冻结数据集

数据集文件为 `docs/benchmarks/a5-7-chatbi-eval-v1.json`，包含 20 个问题：16 个可回答
正例和 4 个拒绝类负例。正例覆盖过滤、投影、聚合、分组、排序/Top-K、日期条件、连接、
谓词、NULL、CTE、集合操作、空结果和派生列；负例覆盖问题不明确、写操作、未知表和不受
支持的视图。

数据集保存 reference SQL、归一化结果和明确标注的回答事实。它们只用于评测端比较，
绝不会进入 NL2SQL 或结果分析消息。每个数据集包含版本和内容指纹；评测启动时还会校验
`tests/fixtures/chatbi_demo.sql` 的 SHA-256：

`cd5334e2aa7cb5d3a0e8a4ac95c876fd396f7ae13c5b7a7620497bd3de753c49`

修改问题、oracle、字段或数据集元数据后，旧指纹不能继续通过加载校验。

## 结果比较

结果比较使用 A5.4 的已归一化完整行数据：列数和列顺序必须一致，列名可以不同；行顺序
是否重要由每个 case 明确指定，重复行不会被集合比较错误去重。数值列使用有限的
Decimal 语义和小容差比较，`NULL`、字符串、布尔值、数组和对象按 JSON 安全值比较。
`row_limit` 与 `payload_bytes` 截断原因也必须一致，空结果是有效结果。

产品返回的 `ChatBIResult` 只暴露列和行数等安全元数据，不包含物化行。provider 评测器
在执行服务边界捕获同一次 `SQLExecutionOutcome.result`，仅把它放在评测器内部用于语义
比较，不改变产品响应契约，也不会把 oracle SQL 写入运行结果。

## 离线场景与指标边界

`docs/benchmarks/a5-7-chatbi-offline-scenarios.json` 是显式的 provider-free 场景文件。
它为每个 case 提供状态、行、截断元数据、修复计数和安全 usage 元数据。运行器会校验
行宽、行数、失败类别、修复次数和 provider attempt 计数；失败场景不能携带结果数据。

离线输出的质量指标部分只包含 `offline_verification`，其中：

- `scenario_match_rate` 是脚本场景与冻结 oracle 的匹配比例；
- `comparator_checks` 检查归一化结果，`taxonomy_checks` 检查负例类别；
- `repair_scenario_checks` 检查显式修复场景。

这些是评测器基础设施的确定性检查，不是模型质量指标。离线运行不会填充
`provider_quality`，也不会报告 `execution_accuracy`、`answer_fact_coverage` 或
`end_to_end_success_rate`。因此脚本场景的 100% 匹配不能被写成 SQL 正确率或回答正确率。

provider 模式才会按 `all`、`dev`、`test` 分组记录：

- `generation_parseable`：实际调用并成功解析 NL2SQL 输出的 case；
- `validator_accepted`：实际进入校验阶段并通过 A5.3 校验的 case；
- `execution_accuracy`：正例中结果与冻结 oracle 等价的比例；
- `answer_fact_coverage`：回答是否包含 case 明确列出的必需事实。它只做规范化后的事实
  覆盖检查，不判断完整事实性、幻觉、数值推理或自然语言质量，也不使用 LLM judge；
- `end_to_end_success_rate` 和受控失败类别。

`generation_parseable` 是三态阶段结果：`not_attempted`、`passed`、`failed`。例如 Schema
Discovery 失败时生成阶段是 `not_attempted`，不能计入解析成功。`validator_accepted` 来自
实际 A5.3 validation stage，而不是从执行成功后的 trace 反推；“校验通过但执行失败”和
“校验拒绝、未执行”保持可区分。

每个 provider case 的耗时字段采用互斥口径：

- `schema_prep_ms`：Schema Discovery；
- `generation_ms` / `repair_ms`：对应 NL2SQL 首次生成 / 修复 provider 调用，不包含其中
  嵌套的 Schema Discovery；
- `validation_ms`：A5.3 AST/policy 校验，不包含嵌套的 Schema Discovery；
- `execution_ms`：A5.4 执行与结果归一化；
- `analysis_ms`：结果分析调用；
- `total_ms`：单 case 的端到端 wall-clock，包含调度和未单独计时的少量开销，因此不要求
  等于各阶段简单相加，但不应因 Discovery 嵌套而重复计时。

未执行的阶段耗时写为 `null`，不会在阶段耗时聚合中当作 0ms 计入；`total_ms` 仍记录该
case 的端到端 wall-clock。

离线运行仍会输出通用的修复计数、受控失败类别、阶段状态和安全 usage 汇总；这些字段
描述脚本场景，不改变离线模式的质量边界。

## Provider 运行门禁与 provenance

provider 运行不是离线结果的延伸，必须显式配置并验证唯一的演示业务数据源：

- `KNOWLEDGE_SCOPE_CHATBI_EVALUATION_DATASOURCE_ID` 必须与命令行的 `--datasource-id` 相同；
- 数据集必须是冻结的 `a5.7-v1`，并匹配仓库中的 `tests/fixtures/chatbi_demo.sql` 文件指纹；
- 运行器先从 KnowledgeScope 注册表读取数据源，经可信 Schema Discovery 验证
  `chatbi_demo.customers` 与 `chatbi_demo.sales` 的结构，再用固定的有限只读数据探针验证
  fixture 内容；全部门禁通过后才创建 LLM provider；
- 不支持把任意自定义 datasource 静默当作官方评测目标。自定义目标若要实验，应使用独立的
  非权威流程，不得与冻结结果直接比较。

运行产物使用 `a5.7-run-v2`，并记录数据集版本/指纹、fixture 版本与文件/数据指纹、注册
datasource ID、Git revision 与 dirty 状态、provider/model、NL2SQL/analysis prompt 版本、
Agent 限制、QueryPolicy、statement timeout 和结果边界。不会记录 DSN、密码、API key、完整
prompt、原始 provider 响应或 oracle SQL。

## Provider 统计边界

provider 运行的结果只表示本次真实调用；没有实际计价配置时成本保持未知，不写入虚构的零
成本。所有阶段计时和 token 统计均限于该次运行，不能从离线脚本推导质量结论。

## 命令

默认运行不需要数据库、GPU、网络或 API key：

```bash
uv run knowledgescope chatbi eval
```

输出默认写入被 Git 忽略的 `data/evaluation/a5-7/run.json`。可以用 `--max-cases` 做
小范围检查。provider 模式必须显式指定已注册 datasource：

```bash
uv run knowledgescope chatbi eval \
  --mode provider \
  --datasource-id <registered-datasource-uuid>
```

该模式会复用现有注册 datasource、可信 Schema Discovery、A5.3 校验、A5.4 执行服务和
A5.5 Agent；reference SQL、期望行和回答事实仍停留在评测端。provider 运行的耗时、token
和质量统计只表示本次实际运行，不是离线场景的预测。

运行结果、provider 相关输出和其他临时评测数据位于 `data/evaluation/a5-7/`，不得提交
模型文件、数据库内容、原始 provider 响应、密钥或 PDF。

## 当前状态

本阶段提供数据集、oracle 隔离、纯结果比较、失败/修复/usage/耗时统计和显式 provider
运行入口。正式的 provider 质量评测仍需在受控环境中运行；在没有人工或正式标注协议
之前，不报告准确率、回答质量或模型优劣结论。
