# ChatBI 评测数据集 v2

## 目的与边界

v2 已完成机器构造、PostgreSQL oracle 校验、初始人工语义审核、修订、最终人工冻结审核和
最终技术冻结审核，当前状态为 `human_reviewed_frozen`。冻结数据集质量已验证；模型和
provider 质量尚未测量。A5.7 v1 继续保持冻结，用于既有 harness、离线场景和回归；v2
使用独立的业务 fixture，避免修改 v1 的数据与语义。

本轮修订明确了无序结果、并列排序、派生指标、空结果和歧义问题的语义；问题文本只保留
业务意图，reference SQL 仅作为评测 oracle。

reference SQL、期望结果和结构化答案事实只属于评测端，不会进入 generation 或 analysis
prompt。正式 benchmark 仍应通过 A5.3 trusted validation 和 A5.4 只读执行获得结果；
Execution Accuracy 是 v2 的主要质量指标，`answer_fact_coverage` 只作为辅助信号。

## 独立 v2 fixture

文件：`tests/fixtures/chatbi_demo_v2.sql`

fixture 创建 `chatbi_demo` schema，包含三张业务表和一个仅用于覆盖清单的视图：

- `customers(customer_id, customer_name)`：10 位客户，`customer_id` 是身份键；其中两条
  记录共享显示名称，另有一位客户没有销售记录；
- `sales(sale_id, customer_id, region_code, sold_on, amount)`：30 笔销售，覆盖 5 个地区和
  2026 年 1 月至 6 月，包含同额记录、月初/月末日期和多次客户销售；
- `regions(region_code, region_name, market)`：5 个地区维度，用于真实三表业务问题；
- `region_sales`：派生视图，不作为 v2 正例执行目标。

外键连接为 `sales.customer_id → customers.customer_id` 和
`sales.region_code → regions.region_code`。v2 的多表问题均在 SQL oracle 中实际引用
`customers`、`sales`、`regions` 三张物理业务表，而不是用表数量装饰问题。

fixture SHA-256：

`fd972106c39c7a8b31b57975118708e213a32e4e008ee13fafc15b1ea1b5182d`

v1 fixture `tests/fixtures/chatbi_demo.sql`、v1 数据集、v1 指纹和 v1 harness 语义均未
改动。v2 数据源路径和版本独立记录为 `chatbi-demo-v2`。

## 数据集结构

文件：`docs/benchmarks/a5-7-chatbi-eval-v2.json`

- schema：`a5.7-v2`；状态：`human_reviewed_frozen`；
- 总计 80 条：dev 50、test 30；正例 74、负例 6；split 已冻结，不根据 provider 输出移动；
- 类别数量保持原覆盖配额；难度按问题的实际语义/组合复杂度标注，不追求旧的难度比例。

冻结后的 dev/test 使用边界如下：dev 可用于分析以及未来的 prompt/model 开发，test 只能
作为最终评测数据，不能用于 prompt 调优。正式 test 指标必须标注本版本的完整数据集指纹；
任何语义修改都必须创建新的数据集版本，不能在 v2 上原地修改。

| 类别 | 数量 |
| --- | ---: |
| simple_filter_projection | 7 |
| aggregation | 9 |
| group_by | 8 |
| ordering_top_k | 7 |
| multi_predicate | 7 |
| join | 10 |
| multi_table_join | 6 |
| date_filter | 5 |
| null_boundary | 4 |
| empty | 3 |
| alias_derived | 4 |
| cte_set_operation | 4 |
| ambiguous_question | 3 |
| unsupported_request | 3 |

难度分布为 easy 18、medium 41、hard 21。简单的单表筛选/标量聚合为 easy；需要连接、
外连接、派生指标、多阶段或集合运算的问题按实际结构提高难度；`GROUP BY` 本身不会
自动标为 hard。

## 人工反馈的应用

当前版本保留原始 ACCEPT/EDIT/REPLACE 审核决定，并按审核建议修订问题文本、oracle 语义
和结构化答案事实。修订重点包括无序结果的 `order_sensitive` 标记、top-k 并列规则、客户/
地区业务语义、空结果日期边界和安全歧义行为。初始人工语义审核、最终人工冻结审核和最终
技术冻结审核均已完成；审阅清单是冻结后的可追溯记录，不是待填写表单。

审阅清单：`docs/benchmarks/a5-7-chatbi-eval-v2-review.md`。每条记录展示 case ID、split、
类别、难度、问题、正负语义、reference SQL、截断的结果摘要、行序策略、结构化答案事实
摘要和已应用的 review status。

## Oracle 与结果语义

每条正例的 reference SQL 都在生成阶段通过独立的确定性 oracle 计算期望结果，并由
PostgreSQL 集成测试再次通过真实 v2 fixture、Schema Discovery、A5.3 validation 和
A5.4 read-only execution 逐条验证。不得手工改写期望结果来迎合模型输出。

结果比较继续复用 v1 的 `compare_normalized_result`：列顺序、行宽、行值、数值容差和
截断元数据必须一致；只有问题明确要求排序时才比较行序。oracle 可以带稳定的
`ORDER BY`，但无序问题的 comparator 不会因为返回行顺序不同而失败。

结果列按返回位置比较数量和行值；输出列名/别名本身不参与等价判断，但列顺序会改变
位置语义，因此必须保持一致。重复行、多重集合语义、数值容差和 NULL 语义保持不变。

客户实体的身份是 `customer_id`，涉及客户分组的 query 同时按
`customer_id, customer_name` 分组；不能把显示名称当作唯一键。所有客户语义明确标注
是否包含无销售客户。需要保留无销售客户的问题使用 `LEFT JOIN`、`COUNT(s.sale_id)`、
`COALESCE(SUM(...), 0)` 或明确的 NULL 结果。

结构化答案事实 `structured_answer_facts` 记录 `scalar`、`row`、`null` 和 `empty` 类型。
row fact 使用列名到值的映射，保留客户/地区/金额等实体和值的配对；对应代码函数为
`structured_result_facts_match`，只检查已规范化查询结果中的行事实，不判断最终自然语言
回答。它是辅助检查，不替代完整的 Execution Accuracy，也不使用 LLM judge。v1 的
`expected_answer_facts` 字段和覆盖语义保持不变。

`derived-01` 使用带符号的逐笔销售金额差值；`derived-02` 使用地区销售笔数占全部销售
笔数的比例，结果按数据库数值聚合生成，避免把二进制浮点展示误当作精度。两者都不要求
人为补充百分号或隐藏舍入规则。

歧义负例 `ambiguous-01`、`ambiguous-02` 接受
`clarify_or_refuse_safely`，`ambiguous-03` 保持安全拒绝；评测不会把自信的无依据分析
判为正确。负例不携带可执行 SQL oracle。

## 重复、相似度与泄漏审计

`audit_chatbi_evaluation_dataset_v2()` 使用 NFKC/casefold/空白归一化检查问题重复，并
审计近重复问题、相同 reference SQL、相同结果组和 oracle/元数据泄漏。语义审计先检查
跨 split 的归一化问题形状，再用保守的跨类别信号检查客户编号到客户名称、外连接和指标
重叠；这些只是人工复核信号，不会自动移动或淘汰 case。当前结果：

- 归一化问题重复：0；
- 高相似问题对：0；
- 相同 reference SQL：0；
- 当前 heuristic 标记的 dev/test 语义关注对：0；
- 问题中的 SQL、数据集状态、评测字段等泄漏：0；
- 相同结果组会被展示供审阅，不自动视为质量结论。

语义关注审计会将日期、数字和地区等可替换值归一化，帮助发现“只换客户/地区/阈值”的
dev/test 近邻；简单改写措辞不能证明独立。当前任务不自动移动 case，最终 split 变更
需要单独的人审决定。

## 指纹

数据集指纹只通过 `ChatBIEvaluationDatasetV2.fingerprint` 计算一次，使用仓库统一的
确定性 JSON（UTF-8、排序键、固定分隔符）并排除自身 `dataset_fingerprint` 字段：

`60c75c597da8fc71a0fa5b25d335b63410b44a4ab3a403da40ca72c5ae375ab3`

同一指纹写入 v2 JSON、审阅 artifact 和本文件；fixture 指纹也同时写入 JSON 与审阅
artifact。加载测试会重新计算指纹并核对 v1 fixture 未变，防止文档或数据漂移。冻结状态
参与数据集指纹，状态变化会生成新的数据集版本指纹。
`repair_applicable` 仅保留为审核元数据，不作为 v2 修复率分母；修复是否发生以运行时
尝试记录为准。

## 运行边界

默认加载、结构审计和本地 oracle 草稿生成不需要 provider、GPU 或 API key。最终 oracle
以隔离的 v2 PostgreSQL 业务数据库逐条复核；验证不查询 KnowledgeScope 应用表，不执行
负例，不调用外部 LLM。运行时数据库和 provider 输出不进入 Git。

本版本不修改 v2 fixture、v1 数据集或生产 Agent、NL2SQL 安全校验、SQL 执行、MCP、检索和
frontend。冻结数据集质量为已验证；provider benchmark 尚未运行，因此不得将 provider
准确率、Execution Accuracy、端到端准确率、延迟或 token 使用量写入冻结结论。
