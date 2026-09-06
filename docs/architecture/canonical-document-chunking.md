# CanonicalDocument → Chunk 规范

## 范围

Phase A1.6 在已有 `CanonicalDocument` 上提供结构感知的语义分块。这里的“语义”指 canonical schema 已经表达的标题、正文、表格、公式、图片、页面和 reading order；它不是 embedding-similarity splitting，也不依赖模型、MinerU 或网络。

分块是解析器无关的纯应用层处理：输入是 `CanonicalDocument`，输出是 `ChunkedDocument`。当前不创建数据库表，不在上传请求中自动运行，也不向普通用户暴露分块控制。

## 模型

`Chunk` 的 schema version 当前固定为 `"1.0"`，包含：

| 字段 | 约定 |
| --- | --- |
| `chunk_id` | 由文档 ID、配置 fingerprint、ordinal 和来源 block ID 稳定计算，不使用随机数或时间 |
| `document_id` | 对应源 `CanonicalDocument.document_id` |
| `ordinal` | 文档内从 0 开始的连续序号 |
| `text` | 按 canonical block 确定性渲染的 embedding/display 文本；asset-only chunk 可以为空 |
| `page_start` / `page_end` | 所有来源 block 覆盖的页码范围 |
| `source_block_ids` | 按页面和 reading order 排列的非空来源 ID；同一 chunk 内不重复 |
| `section_path` | substantive block 到达前累计的连续 `TitleBlock` 文本，按出现顺序保存；没有标题的前置内容为空路径 |
| `content_types` | 去重后的来源类型：`title`、`text`、`table`、`formula`、`image` |
| `asset_refs` | 来源表格/图片的 opaque 引用，按出现顺序去重 |

`ChunkedDocument` 额外保存 `chunker_version`、`config_fingerprint` 和 `page_count`，并校验所有 chunk 的文档 ID、ordinal 和页码范围。当前 hardening 后的 `SPLITTING_POLICY_VERSION` 为 `section-block-v2`，用于让策略变化生成新的 fingerprint。Pydantic 模型拒绝未定义的额外字段。

## 分块规则

处理优先级为“section boundary → canonical block boundary → character budget”：

1. 连续的 `TitleBlock` 先作为 pending heading context 保留。第一个 substantive block（正文、表格、公式或图片 asset）到达时，所有 pending 标题按原顺序加入同一 section 的 `source_block_ids` 和 `section_path`；已有 substantive 内容后出现的新标题会开启新 section。文档末尾没有 substantive 内容的 pending 标题才允许形成 title-only chunk，不臆造标题层级，也不跨 section。
2. 同一 section 内按 page/reading order 顺序累积 block。达到 `target_chars`，或加入下一个 block 会超过 `max_chars` 时，在 block 边界提交当前 chunk。不同页面可以在同一 section chunk 中连续出现。
3. 普通大小的公式如果因 target budget 即将单独成 chunk，会在同一 section 内尝试回收相邻的完整 block；只有合并后不超过 `max_chars` 才调整边界，不重建已切分的正文，也不跨标题边界。
4. section 末尾的小 chunk 仅在仍处于同一 section、合并后不超过 `max_chars` 且来源 block 不重复时合并。合并复用已经渲染的文本，不会把超长源文本重新拼回去。
5. 单个 `TextBlock` 超过 `max_chars` 时，优先按中文/英文标点、换行或空白做确定性切分，最后才使用不拆 Unicode code point 的硬切分。若前面有 pending 标题，首个正文片段会预留标题上下文空间；每个片段仍指向同一个 `source_block_id`，不添加任意大段 overlap。
6. 单个不可拆的表格、公式或图片渲染结果超过预算时保留完整内容，并在统计中作为 `oversized_atomic_chunks` 单独报告；不伪造或截断其内容。`max_chars` 是 packing ceiling，对不可拆 canonical block 不是绝对截断令。

默认配置是字符预算 `target_chars=1200`、`max_chars=1600`、`min_chars=240`。字符数不是 token 数；该配置只是 A1.6 的结构基线，后续 embedding/retrieval 评估可能调整它。

## 确定性渲染

- `TextBlock` 保留原文。
- `TitleBlock` 以 `## ` 标题形式提供上下文。
- `FormulaBlock` 保留为 `$$` 包围的 LaTeX 文本。
- `TableBlock` 优先保留 Markdown；只有 HTML 时，用 Python 标准库解析成稳定的行文本；只有 `asset_ref` 时不生成文本，只保留 opaque `asset_ref`。
- `ImageBlock` 使用 caption；没有 caption 时不生成文本，只保留 `asset_ref`。

asset-only chunk 的 `text` 可以为空，但必须有 `asset_refs`；不会添加 `[image]`、`[table]`、`[formula]` 或其他虚构内容。

## Lineage 与 artifact

每个 chunk 都能沿以下链路回到原始文档：

```text
Chunk.document_id
  -> Chunk.source_block_ids
    -> Page.page_number
      -> CanonicalDocument
```

纯核心 API 为：

```python
chunk_document(document: CanonicalDocument, config: ChunkingConfig) -> ChunkedDocument
```

它不执行数据库、网络、GPU、MinerU 或 embedding 操作。开发者 CLI `uv run knowledgescope chunk-document <document-id>` 读取已有的 `data/parsing/<document_id>/canonical.json`，使用 staging 和原子提升写入 `data/chunking/<document_id>/chunks.json` 与 `manifest.json`。artifact 中记录配置和 fingerprint，但不记录机器路径。

## 明确不包含的内容

A1.6 不包含 embedding、向量库、reranking、LLM、RAG/GraphRAG、多模态检索、chunk 数据库表、在线上传联动或前端页面。上述能力留到后续阶段，并应使用本规范提供的稳定 lineage 作为输入。
