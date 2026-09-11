# A3.6 图语料构建与覆盖审计记录

## 范围与口径

A3.6 只负责把已经登记的 A1.6 chunks 送入既有 A3.2 grounded extraction、A3.1 Neo4j persistence 和 A3.3 linking runner。它没有重新运行 MinerU，也没有修改 A1/A2/A3.2 的冻结数据、抽取 taxonomy 或 linking policy。本文前半部分记录 9 个 subject 的生产形态 staged run；文末另行记录 2026-09-10 对完整 255 个文档的真实运行尝试。所有数字都是运行计数，不代表 precision、recall、accuracy 或 GraphRAG 质量。

输入由 `a3-6-corpus-input-snapshot.json` 权威快照和 PostgreSQL 登记状态共同约束。快照记录 255 个唯一文档、7,524 个 chunks、9 个 subject 以及完整输入 fingerprint；每个选中文档都使用完整 chunk 集。运行时 checkpoint、manifest、selection 和锁文件位于被 Git 忽略的 `data/evaluation/a3-6/`，不保存 PDF、课文原文或 provider raw response。

## 分阶段选择与配置

每个 subject 选择 chunk 数最接近该 subject 中位数的一个完整已登记文档，距离相同时按 `document_id` 排序。provider 调用前的选择结果如下：

| subject | document_id | chunks |
| --- | --- | ---: |
| 化学 | `181e7c3f-311e-5da6-89b4-3361880fa554` | 37 |
| 历史 | `19e802b0-8d5b-5516-91c7-bc4ae25a24e6` | 52 |
| 地理 | `2b50d13d-34e7-56c0-a1fc-d9f63371a156` | 29 |
| 思想政治 | `d75f8143-129c-55e9-88a6-aac987af682c` | 49 |
| 数学 | `916f3a4d-2535-5876-b5d8-4c70dff5d5c0` | 23 |
| 物理 | `25e6fcaf-f773-5a9f-b99d-27c940aa4403` | 18 |
| 生物 | `464dd6f8-7bd5-5161-9c8a-a1f61f616e3a` | 29 |
| 英语 | `1450274c-b378-59c2-937e-5d1472ce96f7` | 22 |
| 语文 | `886cefd0-7afd-57fb-802a-683eaaa5f42e` | 17 |
| **合计** | **9 个文档** | **276** |

运行使用 `max_concurrency=4`、`link_batch_documents=8`、当前 DeepSeek/OpenAI-compatible 配置、A3.2 当前 corrective parse retry、A3.3 linking adjudication、checkpoint/resume 和 POSIX run lock。没有增加额外 provider retry；在首轮终态失败后只执行了一次 runner 已支持的显式 `retry_failed` corpus retry。

## 抽取、图写入与完成状态

下表的 provider/token 统计来自所有实际 checkpoint attempt records；latency P50/P95 使用每个逻辑 chunk 的最终 checkpoint，实体、关系和 evidence 是清理前最终 checkpoint/Neo4j 中的结果。`schema_rejected` 同时属于 terminal `failed`，两列不能相加。`grounding reject` 是 A3.2 拒绝的候选事实数，不是 terminal extraction 状态。

表中的 `corpus` 是显式重试事件；它触发的 provider calls 已包含在 `initial`、`corrective` 和 `actual` 的 attempt records 中，不应再次相加。

| subject | chunks | completed (accepted / empty / rejected) | schema rejected / failed | provider initial / corrective / corpus / actual | input / output tokens | latency P50 / P95 (ms) | graph entity / relation / evidence | grounding reject | linking gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 化学 | 37 | 36 (34 / 2 / 0) | 1 / 1 | 39 / 4 / 2 / 43 | 38,856 / 21,092 | 1,527 / 5,941 | 259 / 154 / 34 | 95 | blocked |
| 历史 | 52 | 51 (48 / 3 / 0) | 1 / 1 | 53 / 5 / 1 / 58 | 49,856 / 30,818 | 1,706 / 5,597 | 444 / 262 / 48 | 164 | blocked |
| 地理 | 29 | 29 (29 / 0 / 0) | 0 / 0 | 29 / 2 / 0 / 31 | 27,048 / 13,766 | 1,993 / 4,441 | 230 / 157 / 29 | 54 | complete |
| 思想政治 | 49 | 49 (44 / 5 / 0) | 0 / 0 | 49 / 2 / 0 / 51 | 41,125 / 20,352 | 1,735 / 3,524 | 326 / 181 / 44 | 161 | complete |
| 数学 | 23 | 23 (23 / 0 / 0) | 0 / 0 | 23 / 0 / 0 / 23 | 25,085 / 7,570 | 1,645 / 2,826 | 114 / 81 / 23 | 45 | complete |
| 物理 | 18 | 18 (17 / 1 / 0) | 0 / 0 | 18 / 2 / 0 / 20 | 19,278 / 8,910 | 1,707 / 5,938 | 151 / 75 / 17 | 44 | complete |
| 生物 | 29 | 29 (26 / 3 / 0) | 0 / 0 | 28 / 1 / 0 / 29 | 24,451 / 9,884 | 1,421 / 3,158 | 153 / 86 / 26 | 71 | complete |
| 英语 | 22 | 22 (9 / 13 / 0) | 0 / 0 | 22 / 0 / 0 / 22 | 17,431 / 2,236 | 819 / 1,882 | 45 / 24 / 9 | 11 | complete |
| 语文 | 17 | 17 (17 / 0 / 0) | 0 / 0 | 17 / 2 / 0 / 19 | 22,123 / 11,015 | 2,451 / 6,815 | 193 / 40 / 17 | 127 | complete |
| **合计** | **276** | **274 (247 / 27 / 0)** | **2 / 2** | **278 / 18 / 3 / 296** | **265,253 / 125,643** | **1,659 / 4,441** | **1,915 / 1,060 / 247** | **772** | **7 complete / 2 blocked** |

2 个 `schema_rejected` chunk 在首轮 corrective retry 和一次显式 corpus retry 后仍未完成，因此对应化学、历史文档保持 `extraction_partial=true`、`linking_blocked=true`；没有取消状态。247 个 accepted extraction 的事实在 Neo4j 中去重后形成 1,915 个 local entities、1,060 个 relations 和 247 个 KnowledgeEvidence。checkpoint 中的事实计数为 2,509 entities、1,075 relations，因为同一确定性 ID 的重复 upsert 不会重复创建节点。

上表是截断策略升级前的历史 staged 运行记录；当时 `finish_reason=length` 重用了同一
1024 输出上限并消耗了 corrective retry。当前实现已将截断重试独立为有界的
`1024 → 2048 → 4096` 预算序列。修改该序列会改变 pipeline fingerprint，因此不能直接
恢复旧 checkpoint；后续受控重跑必须使用新的兼容 checkpoint 目录。

### 截断策略修复后的两文档回归

使用同一冻结模型配置和新的兼容 checkpoint 目录，对上述化学、历史两份完整文档做了
受控回归，没有启动全语料运行：共 89 个 chunk，化学 37/37、历史 52/52 均达到
`accepted` 或 `empty` 终态，没有 terminal `schema_rejected` 或 truncation failure；其中
84 个 accepted、5 个 empty。实际 extraction provider calls 为 164，截断预算 retry 为
8，corrective retry、gateway provider retry 和 corpus retry 均为 0。两个文档均通过
document-complete gate，1 个 linking batch 完成（112 个 candidate、10 个 LINK、92 个
NO_LINK、10 个 UNCERTAIN）。运行中产生的临时图数据随后按目标 KB+document 清理，重复
清理返回 0，目标 KB 的 evidence/entity/relation/canonical/membership 计数均为 0。

## Linking 计数

linking 按两个完成的 document window 统计，不能把跨文档 candidate pair 任意分摊到单个 subject：

- candidate pairs：203；其中 92 个由现有 deterministic safe rule 直接 `NO_LINK`，111 个进入 LLM adjudication；provider/network retry 为 0，adjudication failed attempts 为 7，但两个 batch 最终完成；
- decisions：`LINK=26`、`NO_LINK=154`、`UNCERTAIN=23`；没有自动改变 A3.3 的决策语义；
- canonical entities：23；memberships：48；linking batch failures：0；因抽取不完整而 blocked 的文档：2；
- linking input/output tokens：51,460 / 5,485；linking latency：106,700 ms；价格未配置，所以 cost 为 `null`。

只有抽取完整的 7 个文档进入 linking；失败文档不会用不完整的 local entity 集覆盖已有 linking state。

## 运行时观察

首轮 9 文档处理 wall-clock 为约 252.1 秒；对 3 个失败 chunk 的一次显式恢复约 12.1 秒，合计 active wall-clock 约 264.2 秒（不包含人工检查之间的等待时间）。按 276 个逻辑 chunks 计算，首轮约 65.5 chunks/min，含恢复约 62.6 chunks/min。

按所有 attempt records，平均每个逻辑 chunk 约 961 input tokens、455 output tokens、1.072 次实际 provider attempt；最终 extraction 结果平均约 9.09 entities 和 3.89 relations。最终 checkpoint latency 的整体 P50/P95 为 1,659 / 4,441 ms。subject 存在明显 workload skew：英语 P50 约 819 ms 且大量 empty，语文 P50 约 2,451 ms；历史拥有最大的目标 subject chunk 数，语文/化学的输出 token 和实体密度较高。上述 latency 是 runner checkpoint 的 extraction attempt latency，不是全量运行承诺。

## Resume 验证

使用完全相同的 9 文档 selection、input snapshot、pipeline fingerprint 和 checkpoint 目录再次调用 runner，且不启用 `retry_failed`。结果为：

- checkpoint records：279 → 279；
- 实际 extraction provider call 总数：296 → 296；
- 已完成 checkpoint 被复用，link batch 被跳过；没有新的 provider 调用或新的图 upsert；
- 仍准确报告 2 个失败 chunk，说明 resume 不会把 terminal failure 静默当成成功。

这是同一 staged workspace 的安全 resume 检查，不是人为损坏全量 checkpoint。

## 7,524 chunks 全量估算

估算按 9 个 subject 的实际 selected-chunk 比例分别外推到权威快照的 7,524 chunks，而不是把单个 37-chunk 样本直接放大：

- extraction provider attempts：约 **7,945**；
- input tokens：约 **7.208M**；output tokens：约 **3.029M**；
- corrective/corpus 等 retry events：约 **435**；network retry 尚无法从本次运行观察到；
- subject-weighted extraction service latency 总量约 13,884 秒，除以 `max_concurrency=4` 得到约 **58 分钟**的服务时间下界；将本次 active wall-clock 按 `7,524 / 276` 线性放大约为 **2.0 小时**。因此当前可用于排期的 extraction+现有编排区间约为 **58 分钟–2.0 小时**，不是吞吐保证；255 文档的 linking 扩张、rate limit、Neo4j contention 和失败聚集未被可靠建模；
- 未配置价格，full-run cost 保持 `null`。

本次估算来自升级前的历史样本，不能用来预测新增截断预算下的 provider calls；新的运行
应分别报告 `initial`、`provider/network`、`truncation`、`corrective` 和 `corpus` 计数。

与旧的单 subject 37-chunk 估算相比：旧记录为 39 provider calls、34,524 input tokens、16,079 output tokens、2 次 corrective retry，线性估算约 8,193 calls / 7.999M input / 2.373M output / 669 retries。新的 9-subject 测量得到更低的 calls/input 预测但更高的 output 预测，原因是 subject mix、文档内容密度和实际终态 retry 不同；两者不能视为同一分布的重复测量。

## 临时图清理与运行问题

运行结束后只对目标 KB `3593a2ee-a326-5a0a-89a2-9a3012666c83` 的上述 9 个 document UUID 调用 scoped `delete_document`。第一次清理暴露出一个真实的 Neo4j 删除边界问题：同一 local entity 参与多个 LINK decision 时，旧 Cypher 可能在同一事务中重复访问已删除的 `CANONICAL_MEMBER_OF` relationship。已用最小查询改动将待删除 decision ID 先聚合，再一次性更新/删除受影响 membership，并增加真实 Neo4j 回归测试；没有改变抽取或 linking 语义。

修复后重新执行精确清理：9 个文档每个返回 `evidence=0 / relations=0 / entities=0`，目标 KB 的 `KnowledgeEntity`、`KnowledgeRelation`、`KnowledgeEvidence`、`CanonicalEntity`、`KnowledgeLinkDecision` 和 `KnowledgeLinkPair` 均为 0；再次重复清理也全部返回 0。没有触碰其他 KB 或未选中文档的图状态。原始 checkpoint/manifest 仍保留在 ignored runtime 目录供审计。

## 255 文档全量运行记录与最终审计（2026-09-10 至 2026-09-11）

在上述 staged 记录之后，使用同一权威输入快照、目标 Knowledge Base `3593a2ee-a326-5a0a-89a2-9a3012666c83`、冻结模型配置 `deepseek-v4-flash-vision-exp`、`max_concurrency=4`、`1024 → 2048 → 4096` 截断预算和运行目录 `data/evaluation/a3-6/full-run-20260910/` 执行了完整 7,524 chunk 选择。首轮结束后只执行了一次显式 `--retry-failed` 恢复轮，之后只对终态失败的 17 个 chunk 做了一次有界 targeted recovery；没有使用 `--reprocess` 或隐藏重试。

- 权威输入与服务状态：255/255 文档已登记，7,524/7,524 chunks 校验通过，Qdrant 7,524 个 point 的 KB attribution 全部正确；目标 Neo4j 初始为空且运行锁未被占用。
- 首轮中间状态（非最终状态）：3,967 个 chunk、120 个文档完成；3,557 个 chunk、135 个文档因 provider/API 错误或其他终态失败而阻断 linking。首轮失败包括 3,543 个安全记录的 `api` provider 错误和 14 个 schema/截断类终态失败。
- 首次显式恢复轮（非最终状态）：没有修复当时的 3,557 个失败 chunk；这些中间状态不会替代最终 checkpoint，持久化 checkpoint 未保存 provider raw response 或 secrets。
- 首轮阶段图状态（非最终状态）：保留 3,334 个有证据支撑的 chunks/evidence、29,566 个 local entities、14,841 个 relations、500 个 canonical entities、1,051 个 memberships；17 个 linking batch 完成。135 个失败文档未进入 linking gate；其中 11 个文档含有已成功 chunk 的阶段性图事实，后续已按 KB+document scope 清理。
- 首轮阶段审计（非最终状态）未发现跨 KB 节点、cross-scope identity mismatch、orphan evidence/entity/relation 或被占用的 runner lock；A2.1 的 108 条评测映射仍指向目标 KB。上述数字只是故障前后的执行历史，不能解释为最终图覆盖、抽取质量或检索质量。
- provider/API 故障使 full-run cost 与完整 token 总量保持未知；运行时目录保持 ignored，未跟踪 PDF、原文、模型、Qdrant 数据或 raw provider response。

### 最终审计状态（当前 A3.6 语料边界）

最终运行仍保持 `partial_failure`，不是 100% 成功：

- 语料：255 个已登记文档、7,524 个唯一 chunks；其中 7,513 个达到 successful-terminal，11 个为 terminal failed，失败分布在 9 个文档。
- 资格：246 个文档 extraction/link-complete 并进入 graph corpus；9 个文档为 extraction-partial/linking-blocked，已移除其 document-scoped partial graph state，因此没有 retrieval-visible partial graph state。extraction completion 不等于 graph eligibility，只有 link-complete 文档进入最终图语料。
- 合格图谱：55,098 个 local entities、27,268 个 relations、6,087 个 `KnowledgeEvidence` records（也是当前 supported chunks 数）、1,241 个 canonical entities、2,609 个 canonical memberships。
- 覆盖口径：extraction success 为 `7,513 / 7,524` chunks；graph evidence coverage 为 `6,087` chunks；graph-eligible documents 为 `246 / 255`。这些是运行/覆盖计数，不是 extraction accuracy、precision、recall 或 retrieval improvement。
- A2.1：冻结评测集共 108 条，其中 105 条 complete graph coverage、0 条 partial、3 条 no graph coverage；三类合计为 108，标签未被修改。
- provenance：`run_id=e926e1d3-9050-4911-89ef-1632ed0894c2`，corpus snapshot 为 255 documents / 7,524 chunks，pipeline fingerprint 为 `a9407046dad32e8da8f2ef4256497b60e8ffdb21771c4075a7c87a013ff70cc7`，prompt contract 为 `legacy-v1`，truncation policy 为 `1024 → 2048 → 4096`。`cache-v2` 未用于本次运行。

这组最终数字来自当前 checkpoint、文档 checkpoint、排除审计记录和目标 KB 的只读图审计；provider outage/recovery 仅作为运行历史保留，不改变最终的 `partial_failure` 状态。

### Provider/API 故障的离线审计与熔断保护

对 `data/evaluation/a3-6/full-run-20260910/` 的审计只读取现有 manifest、checkpoint、
usage 记录和锁状态，没有发起 provider 请求、没有重试失败项、没有创建新 run，也没有
写入 Neo4j。该目录的 `run_id` 为
`e926e1d3-9050-4911-89ef-1632ed0894c2`，pipeline fingerprint 仍为
`a9407046dad32e8da8f2ef4256497b60e8ffdb21771c4075a7c87a013ff70cc7`。最终 checkpoint
包含 7,524 个唯一 chunk：7,513 completed、11 terminal failed，后者分布在 9 个被排除
文档；运行锁当前未被占用。

安全的 usage 表在该运行时间窗保存了 13,463 条记录，字段只有 provider/model/task、
token、延迟、成功状态和受控 `error_category`，不保存 HTTP status、provider error code、
请求体或 raw response。因此可确认的分类如下：

| 范围 | 结果 | 数量 | 可确认信息 |
| --- | --- | ---: | --- |
| `graph_extraction` | success | 4,252 | `error_category=null` |
| `graph_extraction` | failure | 7,100 | `error_category=api`，其中首轮 3,543、恢复轮 3,557 |
| `entity_linking` | success | 2,093 | `error_category=null` |
| `entity_linking` | failure | 18 | `error_category=api` |

首轮故障及首次恢复轮中的 3,557 个历史失败 checkpoint 全部是 `api`，每个
`attempts=1`、`provider_calls=1`、`extraction_corpus_retry_calls=1`；最终 11 个失败 chunk
则按其终态类别保留在当前 checkpoint。没有可从安全记录拆分出的 quota/
balance/access、auth、429、5xx、timeout/network 或 request-validation 子类，所以这些
细分类别均记录为 **unknown/not recorded**，不能把历史波次臆测成某一种账户原因。时间上，
首个 graph extraction success 为 `2026-09-10T13:17:36Z`，最后一个明确成功为
`14:34:28Z`；首个 failure 为 `14:34:27Z`，之后到 `14:51:30Z` 持续失败。UTC 5 分钟
失败计数为 `14:30=216`、`14:35=2,639`、`14:40=756`、`14:45=2,670`、
`14:50=819`，表现为跨文档/学科的突然 provider/API failure wave，而不是可由 checkpoint
证明的单一内容问题。

本次失败波次没有 gateway provider retry：full-run 使用的 `KNOWLEDGE_SCOPE_LLM_MAX_RETRIES`
为 0，gateway 也只对显式 retryable 的 429/5xx/transport error 重试；随后 3,557 次是
runner 的一次显式 corpus retry，不是 gateway retry。由于历史记录没有保存 status code，
不能仅凭 `api` 类别证明当时具体属于 non-retryable 4xx。

runner 现已加入保守的 run-level circuit breaker，但它不会回写或重分类本次旧运行：只有
运行时实际拿到相同、明确的非临时 4xx `LLMProviderError`，连续 3 次才持久化
`provider_blocked`；429、5xx、timeout、network 和无状态码错误不触发。触发后只允许已在
途的有限 batch 收尾，未调度 chunk 不写失败记录；completed/failed checkpoint 与图状态
保留，provider 恢复后可在同一目录显式 `--retry-failed`。该保护是 at-least-once、补偿式
流程，不宣称 exactly-once。

失败计数必须区分粒度：最终 checkpoint 视图中的 11 是唯一的 terminal failed chunk；manifest
的 `chunks_failed=7,142` 和 append-only checkpoint 行数都是累计观察值，不是 7,142 个不同
chunk。历史故障阶段按 append-only checkpoint attempt records 汇总，graph extraction 的
provider API 失败尝试为 7,100（首轮 3,543 + corpus retry 3,557），对应 UTC 五分钟桶
`216 + 2,639 + 756 + 2,670 + 819 = 7,100`。这 7,100 不包含首轮另外 14 个非 `api`
终态失败或后续 targeted recovery 记录，也不应被表述为唯一失败 chunk 数。

### Prompt cache 离线结构结论

没有调用 API 的前提下，按现有 7,524 个 chunk 对 prompt 结构做了离线比较。当前 full
run 必须使用 `legacy-v1`：页码/章节和 corrective feedback 仍位于 `BEGIN CHUNK` 之前，
旧布局 system+user 的可复用前缀约为 2,412 UTF-8 bytes。

把 chunk-specific 内容移到 marker 之后的布局只是缓存优化实验，未应用到当前 full run。
该实验测得 system 1,071、user 1,888、合计 2,959 bytes 的稳定前缀，变化后缀为
55–4,675 bytes（中位数 687 bytes），相对旧布局增加约 547 bytes。这里没有 provider
cache hit 数据，不能推导命中率或成本节省百分比；未来若启用，必须用新的布局版本和
checkpoint 目录，不能混入现有运行。当前请求未包含 run/document/chunk ID、时间戳、
计数器或随机 nonce；API 恢复后再测 provider 报告的 cache metrics。

## 限制与结论

- 最终 255 文档 / 7,524 chunk 运行仍为 `partial_failure`：7,513 个 chunk 成功、11 个终态失败；不能将其描述为 100% 成功。
- 只有 246 个 link-complete 文档进入最终图语料；9 个 extraction-partial/linking-blocked 文档已清理其 document-scoped partial graph state，不具备 retrieval-visible partial graph state。
- 应分别报告 extraction success（7,513/7,524）、graph evidence coverage（6,087 chunks）和 graph-eligible documents（246/255）；实体、关系和 evidence 计数是运行/覆盖事实，不是抽取准确率、precision、recall 或 retrieval improvement。
- 上文 staged 运行中的 linking 计数和 provider outage/recovery 计数均为历史执行记录，不能代表当前最终图谱或质量结论；A3.6 后续若需处理 11 个失败 chunk，必须是明确的、有界运维操作。
