# A4.2 多模态 Representation Index 与检索

## 范围

Phase A4.2 将 A4.1 的 `MultimodalEvidence` 和
`RepresentationIndexPayload` 物化为一个独立的、可查询的 Qdrant collection。
它支持文本查询命中来自 text、image、table 和 formula block 的 Evidence，并在
返回结果中保留完整来源链路。本阶段不修改 A2.3 的
`knowledgescope_chunks_v1`，不改写 A3 图检索或 RRF，也不接入 RAG answer
generation。

当前使用 `Qwen/Qwen3-Embedding-0.6B` 的既有本地 embedding adapter；本阶段只
索引已有 canonical 内容和从同一文档推导的有限文本上下文，不重新运行 MinerU，
不调用 LLM 生成 caption/OCR，也不产生视觉 embedding。

## 数据层级与物化

```text
CanonicalDocument → A4.1 MultimodalEvidence → Representation → Qdrant point
       block → page → document → original PDF/asset
```

每个 canonical block 保持一个 authoritative Evidence。一个 Evidence 可以有多个
representation，索引只写入 `searchable=true` 的文本表示：

| block modality | 当前可索引文本表示 |
| --- | --- |
| `text` / `title` | 原始 `text`；普通 `text` block 另有同页 section/title 与有限邻近文本 context |
| `image` | 已有 `caption`；没有 caption 时使用同页附近的 canonical text/title 和 section context |
| `table` | `markdown` 或 `html`、`caption`、有限同页 context |
| `formula` | `latex`、有限同页 context |

opaque `asset_ref` 仍保存在 Evidence 和 payload 中，但不是可搜索文本。上下文由
canonical 的既有 title/text block 按固定顺序截断生成，不代表新的事实；section
path 是展示元数据。没有可搜索文本的 image/table Evidence 不会被伪装成可搜索
内容，审计会把它作为 source Evidence 记录。

## Collection 与 payload

独立 collection 名称由
`KNOWLEDGE_SCOPE_QDRANT_REPRESENTATION_COLLECTION_NAME` 配置，默认是
`knowledgescope_representations_v1`。向量 schema 固定为 1024 维 cosine，与 A2.2
选定的 Qwen baseline 一致，但 collection 与 `knowledgescope_chunks_v1` 完全分离。
启动/写入时会校验维度和距离；每个 point payload 还带有
`collection_schema_version`、`evidence_schema_version`、
`representation_schema_version`、collection fingerprint、embedding model/revision/config
fingerprint 和完整的 `RepresentationIndexPayload`。当前结构化上下文物化版本为
`a4-2-structural-context-v2`。

`knowledgescope_chunks_v1` 与 representation collection 是两个不同的 collection
role。A4.2 独立保护这个固定的 frozen chunk identity；即使运行时把 chunk collection
改成别的名称，representation target 仍不能使用 `knowledgescope_chunks_v1`。如果一个
已有的其他名称 collection 的有效 sentinel payload 和确定性 point ID 符合 A2.3
`ChunkVectorPayload`，它也会被识别为 chunk index 并 fail closed。配置、启动和每次
create/replace/stale-cleanup/delete/rebuild 边界都会执行该检查，并在写入前校验向量
schema、payload contract 和当前 collection fingerprint。embedding fingerprint 还显式
记录实际使用的 model-native pooling strategy，以及模型 revision、normalization、max
sequence length 和 query prompt 约定。

collection fingerprint 是由 collection 名称、schema/version、物化版本、向量配置
和 embedding 配置共同计算的 SHA-256。现有 point 的 fingerprint 不匹配时检索、
审计和重建会 fail closed，而不会把不同配置的向量混在一起。

`representation_id` 复用 A4.1 的版本化结构化 SHA-256 identity；Qdrant point ID 是
由 representation ID 派生的稳定 UUID。重复执行同一文档的 build 会 upsert 同一
组 point，随后按显式 `knowledge_base_id + document_id` 删除 stale point。删除和
重建始终带 KB/document scope，不会清理其他知识库的同名 document。

## 查询与去重

`MultimodalRepresentationRetrievalService` 对 query 使用同一 Qwen query encoding
约定，在 representation collection 中执行有界 Top-K 搜索，可按 payload 的
`modality` 过滤；`all` 或省略过滤表示所有 modality。内部会读取至多
`max(top_k, top_k * 3)` 个 raw representation
hits（上限 100），再按 `evidence_id` 去重。

一个 Evidence 的多个 representation 只生成一个 `RetrievedEvidence`；结果分数取
贡献表示中的最高 cosine score，平分时按 raw rank 和 point UUID 稳定排序。结果
同时保留 best representation 和所有 contributing representation，方便后续引用或
审计；任何同一 Evidence 的 authoritative document/page/block/asset/fingerprint
不一致都会拒绝返回。KB filter 是强制的，服务不会返回另一个 KB 的 point。

## 生命周期与边界

`build` 是可重复的文档替换操作：先在内存中完成当前 canonical 的 Evidence 和
vectors 物化，再以 KB/document scope upsert 新点，成功后才删除 stale points；写入或
stale 清理失败时会用已验证的旧 point snapshot 做补偿恢复。每个新 point 在任何
Qdrant mutation 前都必须匹配当前 collection/embedding contract，包括 model、revision、
config fingerprint 和所有 schema version。Evidence artifact 也会保留旧版本，直到
这一轮 point replacement 成功；若旧 artifact 存在但不可验证，操作会 fail closed，
避免用不完整来源覆盖可用索引。

重解析若需要同时更新 A4.2，应通过解析服务的 coordinated generation 路径传入
`representation_store`、`representation_embedder` 和显式 KB：新 Evidence/points 先在
内存准备并完成替换，之后才 promote 新 canonical，再隔离旧 chunks。任一步失败都会
补偿恢复旧 canonical、旧 chunks、旧 Evidence 和旧 points；成功后才清理旧 chunks。
这是 generation-level compensated consistency，跨 PostgreSQL、文件系统和 Qdrant 仍
不是分布式原子事务；进程崩溃或补偿本身失败时需通过 audit/rebuild 恢复，不能声称
exactly-once。仅使用 A4.1 的普通解析路径不会自动完成 A4.2 replacement，调用方必须
显式使用该协调参数，避免在同一轮中静默混用两个 generation。

文档删除 API 在配置了 `representation_store` 时，先按 KB/document scope 将当前点
标记为 `quarantined=true`；正常搜索同时强制 `searchable=true` 且排除 quarantine。
若 PostgreSQL 权威删除失败，恢复原先的 quarantine 状态；若提交成功但物理 Qdrant
删除失败，残留点继续保持不可检索并等待后续清理。external reference 文档也遵循同一
scope 和幂等语义。解析服务的 canonical/chunk/evidence 仍是 A4.1 的独立文件生命周期；
应在新的 canonical 完成并通过本索引 build 后再把 representation state 视为当前，跨
系统切换始终按上述非原子、可补偿模型处理。

索引 artifact 位于被忽略的 `data/evidence/`，审计和运行输出位于被忽略的
`data/evaluation/a4-2/`。提交内容只允许代码、schema/architecture 文档和短的
安全摘要，不提交 PDF、完整 canonical/evidence、embedding、Qdrant storage 或
secrets。

## 开发者命令

对现有 A1.5 canonical artifacts 审计来源与索引覆盖：

```bash
uv run knowledgescope multimodal-index audit \
  --knowledge-base-id <knowledge-base-uuid> \
  --canonical-root data/benchmarks/a1-5/canonical
```

建立或重建一个有界 canonical corpus：

```bash
uv run knowledgescope multimodal-index build \
  --knowledge-base-id <knowledge-base-uuid> \
  --canonical-root data/benchmarks/a1-5/canonical \
  --limit 3
```

也可用 `--canonical-path` 只重建一个已有 canonical JSON。执行文本查询：

```bash
uv run knowledgescope multimodal-index search "温度表中的检查项目是什么?" \
  --knowledge-base-id <knowledge-base-uuid> \
  --modality table \
  --limit 5
```

这些命令只作用于 A4.2 的 collection，不会重算 A1.6 chunk 或改动 A2/A3 的
benchmark/index/graph。当前结果是 representation 覆盖与 lineage 验证，不是检索
质量 benchmark，也不是最终统一的 multimodal RAG pipeline。
