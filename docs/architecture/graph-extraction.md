# A3.2 LLM 实体与关系抽取

Phase A3.2 将单个 A1.6 `Chunk` 转换为经过应用校验的 A3.1 图对象。流程是：

```text
Chunk → A2.6 LLM Gateway → JSON object → Pydantic schema → grounding/去重 → GraphEntity/GraphRelation
```

本阶段不做跨文档实体链接、图检索或全语料自动处理。

## 输出 contract 与 taxonomy

LLM 只能返回以下应用定义的内容字段：

- entity：`name`、`entity_type`、可选的 `aliases`；
- relation：`source`、`target`、`relation_type`、受限的 `evidence` 原文片段。

Pydantic v2 schema 禁止额外字段，因此 LLM 提供的 ID、页码、路径和
provenance 都会被拒绝。应用根据当前 chunk 和 A3.1 的 local identity 规则生成
ID 与 provenance。

初始实体类型保持保守：`人物`、`地点`、`组织`、`事件`、`概念`、`物质`、`制度`、
`技术`、`时间`、`数值`；prompt 同时给出每一类的简短定义。初始关系类型为：`属于`、
`位于`、`包括`、`组成`、`导致`、`影响`、`发生于`、`先于`、`用于`、`体现`、`具有`、
`相关`，并给出每一类的方向性定义。类型值必须逐字匹配，不做模糊匹配或任意同义词
映射；未列入 taxonomy 的类型不会进入图模型，后续扩展必须同步更新 schema、prompt
和测试。

## Gateway、prompt 与解析

稳定 prompt 版本为 `graph-extraction-v1.3`。抽取请求通过 A2.6 的
`LLMRequest.response_format` 请求 OpenAI-compatible JSON Output：
`{"type":"json_object"}`；同时对当前 DeepSeek v4 配置请求
`thinking: {"type":"disabled"}`，避免把推理/解释挤占结构化输出预算。
这使用的是 provider 的 JSON object 模式，不是 provider 强制执行的完整 Pydantic
JSON Schema；应用仍必须执行自己的字段、taxonomy、关系端点和 grounding 校验。也就是说，
JSON 语法有效不等于 Pydantic schema 有效，schema 有效也不等于事实已被 chunk 支持。
DeepSeek 的 JSON Output 要求 prompt 中明确出现 json 和目标格式，并且仍可能返回空
content 或因 `finish_reason=length` 截断；参见其
[JSON Output 说明](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)。

prompt 只提供必要的 taxonomy、输出示例和 chunk 内容，要求每条 relation 提供当前
chunk 中可逐字定位的短、连续 `evidence` 原文片段；证据不足时返回空数组，
不输出 Markdown、解释或应用元数据。解析器保持语义严格：允许单个 JSON code fence
作为传输包装，但不会从任意 prose 中正则提取事实；额外 prose、空 content、截断响应、
未知类型、额外字段和其他 schema 错误都不会变成成功抽取。

抽取层最多进行一次 corrective retry。重试会把安全的失败类别、字段路径和 Pydantic
错误类型（不包含原始模型输出值）放入新的 prompt，并再次生成完整 payload；未知 taxonomy
时还会重新列出允许值。不会盲目重复完全相同的请求。timeout、provider/API、网络错误和
取消由 A2.6 gateway 处理，抽取层不重试它们。`finish_reason=length` 属于独立的截断重试：
初始 `KNOWLEDGE_SCOPE_GRAPH_EXTRACTION_MAX_TOKENS=1024` 后，默认按
`KNOWLEDGE_SCOPE_GRAPH_EXTRACTION_TRUNCATION_BUDGETS=[1024,2048,4096]` 逐级增加输出预算，
不会因此消耗 corrective retry；到最后预算仍被截断时，终态安全错误类别为
`truncated_response`，不会解析不完整 JSON。每次实际 provider 调用仍可能重复 provider
侧工作和费用，不承诺 exactly-once。

## Grounding 与诊断

应用使用 NFKC、去首尾空白、折叠空白和大小写折叠后的字符串包含关系执行保守的
lexical grounding：实体的 canonical name 或至少一个显式 alias 必须在当前 chunk 中
出现；如果提供 alias，则每个 alias 也必须得到当前 chunk 支持。关系端点必须引用同一
输出中已接受的实体 name 或 alias，且其 `evidence` 必须是当前 chunk 的原文片段并同时
支持 source/target mention。`relation_type` 是受控的规范化语义标签，不要求其字面值
出现在原文或 evidence 中；`evidence` 才是关系的文本依据。应用不会仅凭 schema 通过
就宣称事实为真。这些检查仍不代表人工语义核验。schema/parse success 与 grounding
acceptance 分开统计；解析成功但事实无法被 chunk 支持时仍会被拒绝。runtime review 只
记录 bounded 的来源 excerpt、字段路径/错误类型、relation evidence 和 grounding reason，
不保存原始模型响应。

当前 grounding reason 只描述文本/结构支持问题：
`entity_mention_not_grounded`、`relation_source_not_grounded`、
`relation_target_not_grounded`、`relation_evidence_not_in_chunk` 和
`relation_endpoint_not_in_output`。未知或非法 `relation_type` 在更早的 Pydantic
taxonomy/schema 层以 `unknown_relation_type` 拒绝，不再伪装成文本 grounding 失败。

每个 runtime `sample.jsonl` 记录每一个 provider attempt 的安全类别、`finish_reason`、
token、延迟和有限的 transport format，不记录模型原文。类别包括 valid/empty、非法
JSON、code fence、extra prose、schema/未知 taxonomy、unknown relation entity、
grounding rejection、truncated response 和 provider/API failure。首次尝试、截断预算
retry 与 corrective retry 分开汇总。

## 来源与写入边界

每个接受的图对象都携带当前 chunk 的 `knowledge_base_id`、`document_id`、
`chunk_id`、页码、`source_block_ids`、`section_path`，并在
`GraphProvenance.extraction_provenance` 中记录 provider、model 与 prompt version。
A3.1 的 document-local provisional ID 保持不变；A3.2 不通过名称自动合并不同文档
或知识库的实体。

LLM 提供的 `relation.evidence` 只作为抽取阶段的 bounded 原文证明，并由应用重新验证；
图对象不信任或接收模型生成的 ID、路径等元数据。持久化后的权威来源仍是
`GraphProvenance` 中的 chunk/page/source block lineage。runtime review pack 另外保留
已验证的短 evidence，便于人工检查关系是否确实由原文支持。

`ExtractionService.extract_and_persist()` 将一个 chunk 的全部实体和关系交给
`Neo4jGraphStore.upsert_extraction()`。该方法先验证全部对象，再在一次 Neo4j
managed transaction 中写入实体、evidence 和关系；任意写入失败时该 Neo4j 事务回滚，
不会把该 chunk 的图写入伪装成成功。PostgreSQL、文件系统、LLM 和 Neo4j 仍不是
分布式原子事务，跨系统恢复需要上层补偿或 reconciliation。

抽取样本的逐条 `latency_ms` 和汇总 `latency_ms` 只统计 provider 返回的 LLM 调用延迟；
逐条 `elapsed_ms` 和汇总 `elapsed_ms` 是包含解析、grounding 以及（启用 `--persist` 时）
Neo4j 写入的请求耗时。样本中的 provider 调用、token、成本和抽取结果即使随后发生
Neo4j 持久化失败也会保留在 `status=failed` 的 bounded 记录中，并明确表示抽取结果未
持久化。`sample.jsonl` 与 `summary.json` 各自完成写入后才原子替换；summary 还包含
`record_count` 和 `sample_sha256`，用于发现两份运行产物不匹配。两份文件不是分布式
原子事务。

## 小样本开发入口

以下命令从已存在的 A1.5 canonical artifact 直接调用 A1.6 `chunk_document()`，不会
重新运行 MinerU，也不会自动扫描全部文档：

```bash
uv run knowledgescope graph-extraction-sample --sample-per-subject 1
```

默认每个已分类 subject 选择两个文档的文本 chunk；`--sample-per-subject 1` 是 9-chunk
开发/诊断样本的固定选择。`--sample-offset N` 可稳定跳过每个 subject 的前 N 个候选，
用于建立不重叠 holdout。输出写入被 `.gitignore` 忽略的
`data/evaluation/a3-2/summary.json` 和 `sample.jsonl`。使用 `--persist` 才会通过
Neo4j 写入这批结果。输出只保留 bounded excerpt 和 metadata，不保存 PDF、完整
canonical/chunk corpus 或 LLM 原始响应。样本报告记录 parse、schema、grounding、
empty、重复、token、finish reason、attempt、字段级 schema diagnostics、grounding reason、
latency 和配置成本信息，不报告
precision、recall 或 accuracy。
