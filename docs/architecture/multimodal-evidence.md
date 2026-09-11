# A4.1 多模态 Evidence 与多表示基础

## 范围

Phase A4.1 在已有 `CanonicalDocument → Page → Block` 规范之上，增加
`knowledge_scope.evidence` 的解析器无关数据模型。它把文本、图片、表格和公式
block 表达为可被未来索引的 Evidence，但当前不创建向量或稀疏索引，也不调用视觉
模型、LLM 或新的检索/融合服务。

当前实现的职责是：从一个已经校验过的 `CanonicalDocument` 构建确定性的证据
artifact，校验 artifact 是否仍与当前 canonical source 一致，并提供未来索引可以
使用的 payload contract。Searchable representation 只是 source block 的派生表示，
不是权威内容；权威内容和来源始终是 canonical block 及其 PDF/资产边界。

## 数据层级

```text
MultimodalEvidenceDocument
└── MultimodalEvidence                  一个 source block 一个 evidence
    ├── EvidenceLineage                 KB → document → page → block
    └── representations[]                同一 evidence 的多个表示
        └── RepresentationIndexPayload  未来索引的可解析 payload
```

一个 Evidence 可以同时拥有多个 representation，例如：

| source modality | 当前可用 representation | 是否可搜索 |
| --- | --- | --- |
| `text` / `title` | `text` | 是 |
| `image` | `caption`、可选 `context`、opaque `asset_ref` | 文本表示是；资产引用否 |
| `table` | `markdown` 或 `html`、`caption`、opaque `asset_ref` | 文本表示是；资产引用否 |
| `formula` | `latex`、可选 `context` | 是 |

`visual_embedding_ref` 已作为未来的 opaque reference 类型保留，但 A4.1 不生成或
写入视觉 embedding。当前没有 caption/OCR 生成逻辑；图片没有 caption 时仍保留
Evidence 和资产引用，但不会被错误地伪装成可搜索文本。调用方如果已经从同一
canonical source 得到上下文，可以通过 `context_by_block_id` 提供有限的 `context`
representation。

## Identity 与 lineage

所有 identity 输入先包装在显式版本字段中，再用固定 `json.dumps` 规则编码：

- `ensure_ascii=False`
- `sort_keys=True`
- `separators=(",", ":")`
- `allow_nan=False`
- UTF-8 编码后计算 SHA-256

Evidence ID 的输入是 `schema_version`、`identity_version`、source modality，以及
稳定的 KB、document、page range、按 canonical reading order 的 `source_block_ids`、
asset refs 和 source fingerprint。`section_path` 只属于可变的展示/上下文元数据，
不参与 Evidence 身份；改变它不会改变同一 source evidence 的 ID。格式为
`evidence_v1_<sha256>`。representation ID 另外包含 evidence ID、representation
类型和实际 content/reference，格式为 `representation_v1_<sha256>`。

`block_id` 是 canonical source 的机器标识符，首尾空白会被拒绝；Evidence lineage
和未来 index payload 使用同样的 fail-closed 规则，不在不同层级悄悄做不同的 trim。
`source_fingerprint` 仍单独用于检测 canonical 内容是否过期。`RepresentationIndexPayload`
在扁平化 lineage 时会重新计算并校验 `evidence_id`，并要求 `asset_ref` 属于同一
evidence 的 `asset_refs`，不能把一个 representation 与另一份来源元数据拼接。

因此，同一 source evidence 的多个表示不会产生多个 Evidence；只有表示自身有
独立的 representation ID。canonical block 内容变化会改变 source fingerprint，
canonical document fingerprint 也会变化。Evidence artifact 校验会检查 document/KB
范围、block 是否存在、页码、资产引用、source fingerprint 以及所有派生 ID，发现
不一致时 fail closed。

`asset_ref` 和其他 opaque reference 只允许非绝对、无 `.`/`..` path segment 的引用；
它们不是绝对文件路径，也不替代原始 PDF 或资产存储边界。`document_id` 是 PDF 的
来源标识，Evidence lineage 中的 `asset_refs` 保存图片/表格资产引用。

## Future index contract

`representation_index_payloads()` 可以从已验证的 artifact 生成
`RepresentationIndexPayload`。payload 同时包含：

- `representation_id`、`evidence_id`、modality 和 representation type；
- `knowledge_base_id`、`document_id`、页码范围、`source_block_ids`、`section_path`；
- `asset_refs`、source fingerprint、canonical document fingerprint；
- 文本 `text` 或 opaque `reference`，以及 `searchable` 标志。

未来 Dense、Sparse 或 multimodal index 可以存储这些 payload，但检索结果必须先
通过 `evidence_id` 回到 Evidence，再通过 lineage 回到 canonical block、Page、
Document 和原始 PDF/资产。A4.1 不规定具体 index 后端、向量维度或融合策略。

## Artifact 生命周期

当前派生 artifact 使用受控路径：

```text
data/evidence/<document_id>/evidence.json
```

`rebuild_evidence_artifact()` 从当前 canonical document 原子替换该文件；
`assert_evidence_artifact_current()` 在使用前进行 fingerprint 和 lineage 校验。
解析服务成功提升新的 canonical artifact 后会使旧 evidence artifact 失效，避免
重解析后继续使用旧表示；重解析期间旧 canonical、chunk 和 evidence 会先进入可
恢复的临时隔离区，任一步失效都会恢复完整旧状态，成功后再清理。文档删除会把
evidence 目录纳入独立且受路径约束的可恢复 trash 流程，即使
`external_reference` 文档没有 `data/documents/` 也可以清理；数据库删除失败时
一并恢复，提交成功后再永久清理。artifact 路径的 evidence root、document 目录和
文件均拒绝 symlink 或越出配置的 data root。

Evidence 仍是 derived data，不增加 PostgreSQL 表或 Alembic migration。当前
`data/evidence/` 只用于本地运行，已被 `.gitignore` 忽略；A4.1 不改变已有
Qdrant、Neo4j、A3.7 benchmark 或 RAG 行为。
