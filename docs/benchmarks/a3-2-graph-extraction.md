# A3.2 grounding 修复与三次样本验证

本文档记录 A3.2 结构化抽取的 grounding contract、一次已确认的正确性修复，以及修复后
的三个不重叠小样本。它不把 schema 通过率、grounding 接受率或空抽取当作事实准确率，
也不包含全量抽取。PDF、完整 canonical/chunk corpus、LLM 原始响应和 runtime review
pack 均未提交。

## 统计口径

一次 provider 调用依次经过以下独立层次：

1. transport/provider success：调用没有被 gateway/provider error 终止；
2. JSON parse success：响应是 JSON，允许单个 JSON code fence 包装；
3. schema success：JSON object 通过应用侧 Pydantic v2 contract 和受控 taxonomy；
4. grounding acceptance：实体 mention、关系端点和 relation evidence 通过当前 chunk 的
   保守 lexical grounding；relation type 是规范化标签，不要求逐字出现在 chunk。

因此 JSON 语法有效不等于 schema 有效，schema 有效也不等于 chunk 支持该事实。
合法的空 `entities`/`relations` 是一次成功的结构化抽取，不是失败；chunk 没有关系
也不构成错误。

## taxonomy 与 corrective retry

旧 prompt 主要列出允许值，模型仍可能把自然语言关系词或自造类别写入 JSON object。
这不是 JSON transport 能解决的语义约束：当前 OpenAI-compatible `response_format` 只
要求 JSON object，Pydantic schema、taxonomy 和 grounding 仍由应用执行。现在的
`graph-extraction-v1.3` 为每个实体/关系类型补充简短、方向明确的中文定义，并明确
要求类型值逐字匹配；没有加入模糊匹配或未经审核的 synonym alias table。未知类型
仍会被拒绝，不会被静默映射到合法类型。每条关系还必须提供当前 chunk 中的短、连续
`evidence` 原文片段；应用会重新验证该片段和 source/target mention，而不会信任模型
提供的 ID 或 provenance。

schema failure 最多触发一次 corrective retry。重试会重新生成完整 payload，并携带
应用生成的安全诊断：字段路径和 Pydantic error type，以及完整的允许类型列表；不
携带原始响应值。provider/network/timeout/cancellation 仍由 A2.6 gateway 处理，不在
抽取层重试。重复 provider 调用可能重复费用，不承诺 exactly-once。

## 原始 9-chunk 诊断失败

改进前的 SAME-9 运行记录为 3 个最终 schema rejection：

| subject | 第一次/重试的安全分类 | 可确定的 schema 原因 |
| --- | --- | --- |
| 物理 | `unknown_relation_type` / `unknown_relation_type` | `relations[*].relation_type` 的 Pydantic `literal_error` |
| 生物 | `schema_validation` / `schema_validation` | 同时包含 `entity_type` 与 `relation_type` 的 Pydantic `literal_error` |
| 语文 | `schema_validation` / `schema_validation` | 同时包含 `entity_type` 与 `relation_type` 的 Pydantic `literal_error` |

旧 runtime 文件只保存了安全类别，没有保存字段位置或模型返回值；为避免把原始模型
输出写入 artifact，以上三项无法安全地恢复到具体数组下标或非法字面值，不作臆测。

本次将字段路径诊断加入应用后，对相同的稳定选择使用最终 `graph-extraction-v1.3`
配置重新运行一次。该结果取代此前的调试观察，不再继续针对 SAME-9 调参：

| subject | 最终状态 | attempts 中的安全 schema 诊断 | entities / relations | grounding rejects |
| --- | --- | --- | ---: | ---: |
| 化学 | `empty` | — | 0 / 0 | 0 |
| 历史 | `accepted` | — | 3 / 1 | 2 |
| 地理 | `accepted` | — | 8 / 3 | 1 |
| 思想政治 | `accepted` | — | 6 / 1 | 3 |
| 数学 | `empty` | — | 0 / 0 | 0 |
| 物理 | `schema_rejected` | 第 1 次 `relations[6..8].relation_type: literal_error`；第 2 次 `relations[5..7].relation_type: literal_error` | 0 / 0 | 0 |
| 生物 | `accepted` | 无 schema failure（第 1 次成功） | 5 / 0 | 9 |
| 英语 | `accepted` | 无 schema failure（第 1 次成功） | 7 / 2 | 2 |
| 语文 | `accepted` | 无 schema failure（第 1 次成功） | 10 / 3 | 2 |

该次运行的汇总为：9 chunks、10 provider calls、首次结构化成功 8/9、最终结构化
成功 8/9、retry recovery 0、合法空抽取 2、最终 schema rejection 1、实体 45、关系
10、输入/输出 token `10,000 / 2,566`、累计 provider latency `26,931.398 ms`、
采样请求 elapsed `27,122.306 ms`、
`finish_reason=stop` 10 次、配置成本为 `null`。这只是一次真实运行观察，不是准确率。

## grounding rejection breakdown

旧版 21 条 rejection 发生在保守 lexical guard 之后，但旧 artifact 只保留总数，未保留
未接受事实，因此不能事后可靠地拆分其原因。旧版 v1.2 固定 9-chunk 诊断另外记录了
41 条安全原因计数；它们只用于解释本次修复的动机：

| 原因 | 数量 | 解释 |
| --- | ---: | --- |
| `entity_name_not_in_chunk` | 5 | 旧字段名；对应当前 `entity_mention_not_grounded` 的一类实体支持失败 |
| `relation_endpoint_not_in_output` | 9 | relation 端点没有对应的同一输出 entity；属于输出结构/支持不足 |
| `relation_type_not_in_chunk` | 27 | 已确认是错误规则；规范化 relation type 不应要求在原文逐字出现 |

这些旧 reason 不等价于人工判定的 hallucination，也不代表语义错误率。修复后不再生成
`relation_type_not_in_chunk`；关系是否成立由受控 type、当前 chunk 中的 evidence 以及
source/target mention 的联合校验决定。

## 9-chunk 人工 review pack

被 `.gitignore` 忽略的
`data/evaluation/a3-2/debug-9/sample.jsonl` 是逐条 review pack，包含每个 subject、
chunk/page/section metadata、最长 280 字符的来源 excerpt、最终状态、应用接受的实体/关系、
grounding decision/reason、已验证的 relation evidence 以及每次尝试的安全诊断。它不保存
被拒绝的原始模型 JSON。

本次人工 review 的观察仅限于：化学和数学样本返回合法空结果；历史、地理、思想政治、
生物、英语、语文产生了部分应用接受的实体，部分关系或端点被保守拒绝；物理两次
taxonomy schema rejection。没有据此声称语义准确。

## 修复后 holdout-18（18 chunks）

holdout 使用同一 A1.5 canonical artifacts 和 A1.6 `chunk_document()`，每个 subject
跳过 debug-9 的首个 eligible chunk，再取 2 个。它是修复后的回归观察，不再称为独立
调参 holdout。命令为：

```bash
uv run knowledgescope graph-extraction-sample \
  --sample-per-subject 2 \
  --sample-offset 1 \
  --output data/evaluation/a3-2/holdout-18
```

与 9-chunk debug sample 的 chunk ID 交集为 0；18 个 chunk 覆盖 9 个 subject、每科 2 个。
使用最终 `graph-extraction-v1.3` prompt/config 只运行一次，之后不根据结果调参。
逐条 review pack 位于被忽略的 `data/evaluation/a3-2/holdout-18/sample.jsonl`，汇总在
`summary.json`。

| 项目 | holdout 实际观察 |
| --- | ---: |
| attempted chunks | 18 |
| provider calls | 20 |
| first-attempt structured success | 16 / 18 |
| retry-recovered chunks | 0 |
| final structured success / failure | 16 / 2 |
| valid empty extractions | 5 |
| final status: accepted / empty / schema rejected | 11 / 5 / 2 |
| grounding rejection facts | 46 |
| entities / relations | 84 / 23 |
| input / output tokens | `18,888 / 4,686` |
| finish reasons | `stop=20` |
| total recorded provider latency | `49,972.871 ms` |
| total recorded request elapsed | `50,295.438 ms` |
| estimated cost | `null`（未配置 token price） |

18 个 holdout chunk 的 grounding reason 总数为：`entity_mention_not_grounded=3`、
`relation_endpoint_not_in_output=12`、`relation_evidence_not_in_chunk=1`、
`relation_source_not_grounded=15`、`relation_target_not_grounded=15`。本次没有因
`finish_reason=length` 触发 retry recovery；4 个 schema failure 是两个最终失败 chunk
各自两次尝试的 attempt-level 计数。

## fresh-18（18 chunks）

这是修复后唯一新增的确定性验证样本，固定使用 `sample-offset=3`，在看到结果后不再
调参：

```bash
uv run knowledgescope graph-extraction-sample \
  --sample-per-subject 2 \
  --sample-offset 3 \
  --output data/evaluation/a3-2/fresh-18
```

它覆盖 9 个 subject、每科 2 个；与 debug-9 和 holdout-18 的 chunk ID 交集均为 0。

| 项目 | fresh-18 实际观察 |
| --- | ---: |
| attempted chunks | 18 |
| provider calls | 19 |
| first/final structured success | 17 / 18 |
| retry-recovered chunks | 1 |
| final status: accepted / empty / schema rejected | 15 / 3 / 0 |
| grounding rejection facts | 45 |
| entities / relations | 198 / 64 |
| input / output tokens | `18,954 / 6,942` |
| finish reasons | `stop=19` |
| total recorded provider latency | `68,079.978 ms` |
| total recorded request elapsed | `68,383.800 ms` |
| estimated cost | `null`（未配置 token price） |

grounding reasons：`entity_mention_not_grounded=3`、`relation_endpoint_not_in_output=11`、
`relation_evidence_not_in_chunk=3`、`relation_source_not_grounded=27`、
`relation_target_not_grounded=1`。1 个首轮 schema failure 在一次 corrective retry 后恢复；
`schema_rejection_count=1` 是 attempt-level 计数，最终 schema-rejected chunk 为 0。

fresh review pack 的 accepted relation 单独记录 source、规范化 relation type、target、
relation ID 和已验证 evidence。例如以下 bounded evidence 片段均能在对应 fresh chunk
中定位并同时包含两个端点：

| source | relation type | target | evidence（bounded） |
| --- | --- | --- | --- |
| 体液 | `属于` | 溶液 | 人的各种体液都是含有多种溶质的溶液 |
| 暴雨 | `导致` | 洪灾 | 暴雨引发的洪灾 |
| 正温度系数热敏电阻 | `属于` | 热敏电阻 | 另一类是阻值随温度升高而增大的,称为正温度系数热敏电阻 |
| 原核生物 | `属于` | 单细胞生物 | 原核生物是单细胞生物 |

这些记录只证明应用侧的证据片段和端点校验通过；关系类型是否语义正确仍需人工复核，
不能从结构验证推导 precision、recall 或 accuracy。

## 运行边界

普通测试使用 fake gateway，不需要网络、GPU、DeepSeek 或 Neo4j。真实运行使用本地
`.env` 中已配置的 provider，但输出和文档不包含 API key；`data/evaluation/a3-2/`、
模型原始响应和完整 corpus 保持 ignored。A3.2 仍不做跨文档实体链接、全语料抽取、
GraphRAG、图检索或前端图可视化。
