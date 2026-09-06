# Phase A1.6 结构感知语义分块基准

状态：Phase A1.6 已完成实现，本报告包含本轮 acceptance hardening 的最终结果。本报告只记录基于已有 `CanonicalDocument` 的结构分块事实，不代表检索质量。

## 语料画像

画像来源是被忽略的 `data/benchmarks/a1-5/canonical/`，共 255 份已有 canonical JSON；每份先通过 `CanonicalDocument.model_validate_json` 校验。全量运行只调用纯 Python chunker，不重新运行 MinerU，不写入完整 chunk corpus。可复用的画像摘要见 [a1-6-corpus-profile.json](a1-6-corpus-profile.json)。

| 项目 | 结果 |
| --- | ---: |
| 文档 / 页面 / block | 255 / 3,813 / 46,504 |
| block 类型 | text 30,470；title 8,138；image 6,152；formula 1,335；table 409 |
| TextBlock 字符数 P50 / P90 / P95 / max | 55 / 187 / 249 / 2,833 |
| TitleBlock 字符数 P50 / P90 / P95 / max | 6 / 21 / 44 / 173 |
| 每页 TextBlock P50 / P90 / P95 / max | 7 / 15 / 18 / 82 |
| 相邻标题之间 block P50 / P90 / P95 / max | 3 / 10 / 15 / 134 |
| section 字符数 P50 / P90 / P95 / max | 201 / 908 / 1,452.4 / 13,138 |

画像中没有空的必填内容；4,392 个 image block 没有 caption。409 个表格均提供 HTML，179 个有 caption，230 个没有 caption。section、表格、公式和图片分布只用于选择可解释的字符预算，不是质量指标。

## 首轮基线（硬化前，保留证据）

首轮默认配置为 `target_chars=1200`、`max_chars=1600`、`min_chars=240`。以下结果完整保留，作为本轮 hardening 的 before evidence；首轮当时将每个标题直接视为 section 起点，并为无 caption 图片生成了占位文本。

| 指标 | 首轮结果 |
| --- | ---: |
| processed / failed | 255 / 0 |
| chunks / chunks per document | 8,787 / 34.459 |
| chunk 字符数 P50 / P90 / P95 / max | 229 / 983.4 / 1,252 / 2,994 |
| pages per chunk P50 / P90 / P95 / max | 1 / 2 / 2 / 5 |
| source blocks per chunk P50 / P90 / P95 / max | 4 / 11 / 15 / 62 |
| section 数 / section boundary split count | 8,293 / 8,038 |
| sections split by budget | 325 |
| section-boundary violations | 0 |
| oversized single-text fallback | 16 个 source block |
| oversized non-text block / chunks over max_chars | 10 / 10 |
| 含 table / formula / image 的 chunk | 342 / 541 / 3,644 |
| isolated title / formula / image chunks | 1,266 / 11 / 22 |
| source coverage | 46,504 / 46,504（100%） |

首轮中没有未引用的 meaningful block；16 个超长正文 block 因 deterministic fallback 被多个 chunk 引用。6,176 个带 `asset_ref` 的 table/image block 全部被引用，asset-ref coverage 为 100%。首轮这些结果不是当前输出格式的声明；最终渲染规则见下文。

## 本轮硬化规则与结构原因

### 标题上下文

首轮的 1,266 个 isolated title chunks 经过全量结构审计后得到：

- 连续 `TitleBlock` 序列共有 1,109 组（长度 2–5），包含 2,354 个 title block；另外有 5,784 组长度为 1 的标题。
- 首轮 isolated title 中 1,249 个来自连续标题序列，17 个来自单标题触发的原有边界切分。
- 现在连续标题先进入 pending context；第一个 substantive block 到达时，所有标题按原顺序同时进入 `source_block_ids` 与 `section_path`。substantive 内容之后的新标题会开启新 section。
- 只有文档末尾没有任何 substantive block 的 pending 标题才产生 title-only chunk。最终全量仅有 3 个 title-only chunk，均没有后续 substantive block；所有 8,138 个 title block 都保留了 lineage。

该策略不从标题文字推断层级，不跨 section 合并，也不丢弃没有正文承接的尾部标题。对于标题后面的超长正文，首个正文片段会预留标题空间；不可拆的超长表格/公式会保留标题上下文并保持完整 block。

### 公式审计与局部重平衡

首轮的 11 个 formula-only chunk 分类如下。A 是超过 `max_chars` 的不可拆 FormulaBlock；B 是首轮预算边界造成的普通公式孤立。最终结果列展示同一来源 block 在当前策略下的 chunk 组成。

| document_id | 首轮来源 block | 首轮 | 最终 chunk 结果 |
| --- | --- | --- | --- |
| `916f3a4d-2535-5876-b5d8-4c70dff5d5c0` | `p7-b11,p7-b12` | B | `text/formula`，1,378 字符 |
| `4cd21fc7-96e6-5771-a873-f53fc25503d5` | `p2-b11,p2-b12` | B | `text/formula`，1,147 字符 |
| `a2ee146b-7f39-5b04-aff0-9dca13ea5c22` | `p1-b1,p1-b2` | B | formula-only，824 字符；文档开头且下一 section 有边界 |
| `184b119b-70de-536e-9e23-928b66a156c2` | `p10-b5` | A | formula-only，2,049 字符；保留完整 atomic block |
| `4ccb7235-a68e-5159-8f8a-9b350428bde2` | `p11-b8` | A | formula-only，1,668 字符；保留完整 atomic block |
| `a008b93a-219f-5105-a326-d8b44df95df7` | `p6-b7` | B | `text/formula`，1,552 字符 |
| `a008b93a-219f-5105-a326-d8b44df95df7` | `p15-b3` | B | `text/formula`，1,530 字符 |
| `a008b93a-219f-5105-a326-d8b44df95df7` | `p15-b4` | B | formula-only，953 字符；相邻公式合并会超过上限 |
| `f0764a22-bb45-5877-8945-d9a9135cbe38` | `p7-b4` | A | formula-only，2,197 字符；保留完整 atomic block |
| `269da6d6-33d3-5db5-b7bc-ff996e6e8532` | `p4-b6` | B | `text/formula`，1,547 字符 |
| `269da6d6-33d3-5db5-b7bc-ff996e6e8532` | `p6-b11` | B | `title/text/image/formula`，1,567 字符 |

因此，8 个首轮 B case 中有 6 个获得了同 section 的邻近上下文；剩余 2 个无法在不跨 boundary 或超过 `max_chars` 的前提下安全合并。最终 formula-only chunk 为 5 个，其中 A 3 个、不可安全合并的 B 2 个。`max_chars` 是 packing ceiling，不截断不可拆的 FormulaBlock。

### 图片和 asset-only 表示

首轮画像中 4,392 个无 caption image block 没有自然文本，但旧渲染为 `[image]` 占位文本；本轮改为不生成任何虚构文本，只保留 `asset_ref`。最终有 20 个纯 captionless image asset-only chunk，`text` 为空且 `asset_refs` 非空；其余无 caption image 进入包含其他内容的 chunk，同样不产生 `[image]`。本轮不存在 `[table]`、`[table asset]` 或 `[formula]` 之类的合成文本。asset-only `Chunk` 允许空 `text`，但模型拒绝同时没有文本和 `asset_refs` 的完全空 chunk。

## 最终全量结果（硬化后）

使用相同的 1200 / 1600 / 240 字符预算，对 255 个已有 canonical JSON 重新执行 CPU-only chunking，并对每份结果执行 lineage 校验。

| 指标 | 最终结果 |
| --- | ---: |
| processed / failed | 255 / 0 |
| chunks / chunks per document | 7,524 / 29.506 |
| chunk 字符数 P50 / P90 / P95 / max | 289 / 1,097 / 1,270 / 3,017 |
| pages per chunk P50 / P90 / P95 / max | 1 / 2 / 2 / 5 |
| source blocks per chunk P50 / P90 / P95 / max | 5 / 12 / 16 / 62 |
| section 数 / section boundary split count | 7,048 / 6,793 |
| sections split by budget | 322 |
| section-boundary violations | 0 |
| oversized single-text fallback | 16 个 source block |
| oversized non-text / `oversized_atomic_chunks` | 10 / 10 |
| oversized atomic table / formula / image | 7 / 3 / 0 |
| single-source table / formula / image chunks | 9 / 4 / 16 |
| 含 table / formula / image 的 chunk | 342 / 541 / 3,643 |
| title-only / formula-only / captionless image-only chunks | 3 / 5 / 20 |

所有 46,504 个 canonical source block 都至少被一个 chunk 引用，source coverage 为 100%；没有 unreferenced meaningful block，16 个超长正文 block 被多个 chunk 引用。6,176 个带 `asset_ref` 的 source block 全部被引用，asset-ref coverage 为 100%。全量不存在完全无文本且无资产引用的 chunk；空文本 chunk 共 20 个，全部是 image asset-only chunk。

`oversized_atomic_chunks` 的定义是：chunk 中最多包含标题上下文和一个不可拆的 table/formula/image source block，该 source block 的完整渲染超过 `max_chars`。因此有些带标题上下文的 chunk 仍可能超过上限；内容没有被截断或重新拼接。

## 最终学科统计

下表的 `tables/formulas/images per chunk` 是该学科中包含对应类型的 chunk 数除以 chunk 数，仅用于描述语料构成。

| 学科 | 文档 | 页面 | chunks | chunks/doc | chunk P50 | chunk P95 | tables/chunk | formulas/chunk | images/chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 化学 | 6 | 127 | 210 | 35.000 | 201.0 | 1,227.4 | 0.2429 | 0.0762 | 0.6286 |
| 历史 | 32 | 663 | 1,682 | 52.562 | 179.0 | 654.95 | 0.0107 | 0.0000 | 0.5077 |
| 地理 | 30 | 602 | 897 | 29.900 | 245.0 | 806.8 | 0.0357 | 0.0000 | 0.6321 |
| 思想政治 | 11 | 252 | 591 | 53.727 | 222.0 | 630.5 | 0.0102 | 0.0000 | 0.5482 |
| 数学 | 53 | 706 | 1,313 | 24.774 | 564.0 | 1,414.4 | 0.1127 | 0.3526 | 0.4745 |
| 物理 | 29 | 291 | 566 | 19.517 | 281.0 | 1,204.75 | 0.0371 | 0.1042 | 0.6290 |
| 生物 | 14 | 146 | 403 | 28.786 | 179.0 | 856.8 | 0.0273 | 0.0074 | 0.4913 |
| 英语 | 60 | 619 | 1,491 | 24.850 | 354.0 | 1,337.0 | 0.0335 | 0.0000 | 0.3139 |
| 语文 | 20 | 407 | 371 | 18.550 | 763.0 | 1,399.5 | 0.0135 | 0.0000 | 0.3261 |

## 首轮策略比较（保留）

以下比较来自 hardening 前，预算选择仍固定为当前的 1200 / 1600 / 240；本轮没有为了迎合 prettier 分布调整预算。

| 策略 | `target/max/min` | chunks | chunk P50 / P90 / P95 / max | sections split | text fallback | non-text oversized | coverage |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| tight | 900 / 1400 / 200 | 9,084 | 240 / 924 / 1,017 / 2,994 | 483 | 20 | 13 | 100% |
| default | 1200 / 1600 / 240 | 8,787 | 229 / 983.4 / 1,252 / 2,994 | 325 | 16 | 10 | 100% |
| spacious | 1500 / 2000 / 300 | 8,593 | 223 / 961 / 1,480.4 / 2,994 | 216 | 6 | 9 | 100% |

这些策略比较只用于解释初始 budget 选择；最终 hardening 结果只使用默认预算。

## 九学科结构抽查

对九个学科各选一个 deterministic representative，并复核其 heading、heading + prose、图片资产 lineage 和 section transition；对包含公式的样本检查公式上下文，对全量三个 oversized FormulaBlock 检查完整保留。结果为：九个代表样本均存在 heading + prose；有 image 的样本均保留 `asset_ref`，captionless image 不产生合成文本；公式样本在可容纳时带有同 section 上下文；所有样本 `section_boundary_violations=0`。同时检查了跨页 chunk、表格渲染、captioned/captionless image 和 section 内 block 顺序，未发现跨 section 混合或 source lineage 缺失。

## Artifact、范围与后续阶段

`chunk_document(document, config)` 是纯函数式核心入口，不执行数据库、网络、GPU、MinerU 或 embedding 操作。开发者 CLI `uv run knowledgescope chunk-document <document-id>` 只读取已有 `data/parsing/<document_id>/canonical.json`，并将 validated chunk artifact 写入 `data/chunking/<document_id>/chunks.json` 与 `manifest.json`；本次全量基准没有持久化完整 chunk corpus。

A1.6 仍不包含 embedding、tokenizer、向量库、reranking、LLM、RAG/GraphRAG、多模态检索、chunk 数据库表、在线上传联动或前端页面。后续阶段可以使用当前稳定的 `document_id`、`page_number`、`source_block_ids` 和 `asset_refs` lineage，但本报告不对这些下游能力的效果作声明。
