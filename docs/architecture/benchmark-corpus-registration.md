# Benchmark Corpus Registration

## 目的

现有 A1.5/A1.6 benchmark corpus 只有 canonical document 和 chunk artifact，
不能从文件名、路径、学科或正文推断 KnowledgeBase。A3.5 提供显式的
`corpus register` 开发者/管理员流程，将调用方指定的 255 个文档登记到一个
明确的 KnowledgeBase，供 KB-scoped Qdrant/hybrid 查询使用。

```bash
uv run knowledgescope corpus register --knowledge-base-id <knowledge-base-uuid>
```

目标 KnowledgeBase 必须事先存在，UUID 是唯一的归属输入。默认预检读取：

- `data/benchmarks/a1-5/corpus-manifest.jsonl`
- `data/benchmarks/a1-5/canonical/`
- `data/evaluation/a2-1/chunk_index.jsonl`

预检要求 manifest 的 257 行归并为 255 个唯一文档，canonical 的
`document_id` 与 manifest 一致，7,524 个 chunk 的文档和 source-block lineage
可以在 canonical 中找到，并且 PostgreSQL 中没有其他 KnowledgeBase 的所有权
冲突。整个文档登记在一个 PostgreSQL 事务中执行；预检或写入失败不会留下
部分登记。重复执行只报告已经正确登记的文档，冲突则整体拒绝。

## 存储语义

登记的 benchmark 文档使用：

- `status=registered`
- `storage_kind=external_reference`
- `storage_key=NULL`
- 受控的逻辑 `source_ref`

这表示源 PDF 位于仓库之外且按只读外部引用使用。流程不会创建假的
`original.pdf`，不会复制、移动或删除源语料；普通上传仍使用
`storage_kind=managed` 和 `storage_key`，上传、删除和 MinerU 解析的不变量不变。
`parse-document` 只接受 managed source file，因此登记本身不会误触发本地
文件解析。

迁移 `0004` 会为旧行回填 `storage_kind=managed`，并将 `storage_key` 改为可空，
同时增加 registered/external-reference 约束。迁移只在缺失时填入 managed
默认值，不会把现有 managed storage 改成外部引用。回退 `0004` 前必须先移除
或显式迁移无法由旧 schema 表示的 external-reference 文档；只要这些行存在，
downgrade 会明确失败，不会静默删除登记数据。

## Qdrant 与评测归属

登记完成后，`corpus register` 会在被忽略的
`data/evaluation/a3-5/a2-1-kb-mapping.jsonl` 生成 108 条 bounded 的 A2.1
item-to-KnowledgeBase 映射。它只引用 item ID、split、subject、KB UUID 和
证据文档数量，不改变仓库安全的 A2.1 JSONL。

随后可审计并修复已有 Qdrant collection：

```bash
uv run knowledgescope qdrant audit-kb
uv run knowledgescope qdrant audit-kb --apply
```

修复只更新 `knowledge_base_id` payload，不重算向量，不改变 point/chunk ID、
fingerprint 或其他 lineage。Qdrant 批量更新与 PostgreSQL 不是分布式原子
事务；中途失败可能留下部分 payload 更新，但会明确报错，使用同一权威映射
重复执行即可收敛。映射缺失、为空、冲突或不唯一时不会写 Qdrant。
