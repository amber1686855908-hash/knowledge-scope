# A2.1 检索评估集

Phase A2.1 已完成一个面向文本检索、可审计的 `retrieval-eval-v1` 评估集。标注的稳定主线是：

```text
query → evidence(document_id, page_number, source_block_ids)
```

教材中的原始问题另存为 `query_source`，只用于候选生成和人工审计，不参与 gold relevance。`evidence` 只接受可回答问题的正文、公式或有文本内容的表格 block；来源问题所在 block 会从答案证据和 derived gold chunks 中排除。来源问题没有邻近答案性文本时直接拒绝，不把问题 block 被 chunk 覆盖误当作答案证据。

仓库中保存的最终标注见 [retrieval-eval-v1](a2-1-retrieval-eval-v1.jsonl)，其 wrapper 契约见 [retrieval-eval-v1 schema](a2-1-retrieval-eval-v1-schema.json)，状态快照见 [A2.1 manifest](a2-1-retrieval-eval-manifest.json)，item 字段契约见 [A2.1 schema](a2-1-retrieval-eval-schema.json)。

最终 JSONL 每行包含 `dataset_version`、`split`、`leakage_group_id` 和 `item`；`item` 遵循 A2.1 schema，相关 chunk ID 仍只保存在被忽略的运行时 materialization 中。

当前实现不依赖 embedding model，也不产生真实检索分数。chunk ID 只在运行时由现有 A1.6 `chunk_document()` 从 canonical evidence 推导，不能替代人工证据标注。

## 当前物化结果

- 语料：A1.5 的 255 个唯一 `CanonicalDocument`，覆盖化学、历史、地理、思想政治、数学、物理、生物、英语、语文 9 个学科；manifest 中的 2 个重复 PDF 别名复用其代表 canonical artifact。
- 候选总数：162，每个学科 18 条；人工审核决定为 `accept=12`、`edit=96`、`reject=54`。
- 最终 `retrieval-eval-v1`：`verified=108`、`rejected=54`，9 个学科各 12 条；query 类型为 `factual=31`、`definition=25`、`explanation=25`、`comparison=6`、`formula_or_table=13`、`cross_block=8`。
- split：`dev=72`、`test=36`，每个学科均为 `8 dev + 4 test`；共享 `leakage_group_id`、答案 evidence 或 relevant-chunk 集合的项目不会跨 split。
- 运行时 chunk：7,524 个，使用 A1.6 默认配置；其配置 fingerprint 为 `4471abd3ca3fae31890a04703d39cfd1af8f5ba33ab0fd9e46f6580f871bdb7b`。
- 每条审核后的项目都通过当前 canonical evidence fingerprint 与 chunk lineage 覆盖校验；最终 gold source blocks 全部被当前 chunk 覆盖。

候选 query 优先直接采用实际 canonical 正文中的教材问题、练习题、思考题或讨论题，只做题号、提示词和空白的最小清理；来源问题的答案证据从同一 section 的邻近正文中独立选择。只有在正文确实包含定义、事实/机制、可比较实体或具体公式/表格关系时，才派生 query。生成器会拒绝标题/元数据、上下文依赖片段、OCR 残片、截断内容、泛化教学指令、无可比实体的 comparison，以及没有答案性文本的来源问题，并记录拒绝原因。本次保留 52 条来源于真实教材问题的候选，质量门槛拒绝 7,284 个 draft，其中 2,107 条因没有答案性文本被拒绝。人工审核后的最终 verified 集合保留 30 条 `query_source` 记录；编辑 query 或 query type 后均重新生成 deterministic item ID。

最终泄漏校验结果为：query 出现在 gold answer evidence 中 `0` 条，出现在 derived gold chunks 中 `0` 条；evidence fingerprint drift 为 `0`，未覆盖 evidence 为 `0`，重复 normalized query 为 `0`。最终 verified 集合中发现 3 个重复答案 evidence 组、7 个重复 relevant-chunk 组，dev/test split 泄漏组为 `0`；所有答案 block 均有不含 query-source block 的当前 chunk 覆盖。

## 文本范围与排除

本阶段只评估文本检索：接受标题、正文、公式和带 Markdown/HTML 表示的表格。captionless image-only evidence 不作为可搜索文本，captioned image 也暂不进入本阶段的文本候选；缺少 Markdown/HTML 的表格记录为 `table_missing_content`，不会伪装成可检索文本。

本次运行记录了：`captionless_image_only=4,392`、`image_evidence_out_of_scope=1,760`、`table_missing_content=13`。其中 13 条来自 A1.5 已记录但没有稳定 canonical `source_block_id` 的 table warning，因此只记录 benchmark item 和 warning detail，不伪造 page/block 证据；canonical 中若存在 asset-only `TableBlock`，也会按相同 reason 记录。排除记录由生成器统一写入 ignored runtime workspace。

## 运行与 review workflow

使用已有 A1.5 canonical 结果生成被 Git 忽略的运行时包：

```bash
uv run knowledgescope retrieval-eval build \
  --canonical-root data/benchmarks/a1-5/canonical \
  --corpus-manifest data/benchmarks/a1-5/corpus-manifest.jsonl \
  --output data/evaluation/a2-1
```

候选质量策略调整后，只刷新候选、由候选派生的 materialized/review pack 和 manifest；不会重新运行 MinerU，也不会重建现有 `chunk_index.jsonl` 或排除记录：

```bash
uv run knowledgescope retrieval-eval refresh-candidates \
  --canonical-root data/benchmarks/a1-5/canonical \
  --corpus-manifest data/benchmarks/a1-5/corpus-manifest.jsonl \
  --output data/evaluation/a2-1
```

校验当前语料、evidence fingerprint、文档/页码/block 是否存在、source ordering、normalized query 重复、query 在 gold evidence/derived chunk 中的泄漏、重复 evidence/chunk 分组和 chunk lineage 覆盖：

```bash
uv run knowledgescope retrieval-eval validate \
  --canonical-root data/benchmarks/a1-5/canonical \
  --corpus-manifest data/benchmarks/a1-5/corpus-manifest.jsonl \
  --output data/evaluation/a2-1
```

`review-pack.jsonl` 为本地 reviewer 分开展示 query、`query_source` block/page、答案 evidence block/page、有限长度 answer-evidence excerpt、section context、derived chunks 和 `leakage_group_id`；query-source block 不会出现在 derived gold chunks。reviewer 可以接受、拒绝或编辑 query：

```bash
uv run knowledgescope retrieval-eval review \
  --output data/evaluation/a2-1 \
  --item-id a2-1-<sha256> \
  --action accept

uv run knowledgescope retrieval-eval review \
  --output data/evaluation/a2-1 \
  --item-id a2-1-<sha256> \
  --action edit \
  --query "修订后的问题"
```

人工审核完成后，使用完整 JSONL 决策一次性生成最终集合。此命令只重用已有 `chunk_index.jsonl`，不会重新运行 MinerU：

```bash
uv run knowledgescope retrieval-eval finalize \
  --canonical-root data/benchmarks/a1-5/canonical \
  --corpus-manifest data/benchmarks/a1-5/corpus-manifest.jsonl \
  --output data/evaluation/a2-1 \
  --recommendations data/evaluation/a2-1/A2.1_人工审核建议.jsonl \
  --final-output data/evaluation/a2-1/retrieval-eval-v1 \
  --repository-safe-output docs/benchmarks/a2-1-retrieval-eval-v1.jsonl
```

`data/evaluation/a2-1/` 中的 chunk index、review pack、candidate JSONL 和 excerpts 都是运行时产物，不应提交到 Git；本仓库只保存 schema、代码、测试和本报告，不保存 PDF、完整 canonical/chunk corpus 或长篇教材摘录。

## 指标与范围边界

代码已实现纯函数 `Hit@1/3/5/10`、`MRR` 和 `EvidenceRecall@1/3/5/10`。其中 `EvidenceRecall@K` 是 top-K chunks 覆盖的 unique gold `source_block_ids` 除以该 query 的 gold source block 总数。A2.1 不运行 embedding 或真实 retrieval，因此当前没有检索分数。

本阶段已完成 162 条候选的人工决策、最终 108 条 verified 集合和按 leakage group 约束的 dev/test 划分。A2.1 仍不运行 embedding 或真实 retrieval，因此当前没有检索分数；embedding model 选择、向量索引、reranker 和 RAG 属于后续阶段。
