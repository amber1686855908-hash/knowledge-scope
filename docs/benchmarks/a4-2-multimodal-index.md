# A4.2 多模态 Representation Index 验证记录

## 目的与边界

本记录只验证 A4.1 Evidence 到独立 representation index 的物化、覆盖、lineage 和
检索去重。它不修改冻结的 `knowledgescope_chunks_v1`、A2.1/A2.5/A3.7 结果，
不调用 provider，不生成 LLM caption/OCR，不报告 Hit@K、MRR 或检索质量提升。

默认 collection 为 `knowledgescope_representations_v1`，向量为 1024 维 cosine，
embedding 使用 `Qwen/Qwen3-Embedding-0.6B` 的既有本地配置。实际运行的 collection
fingerprint、canonical root、KB UUID 和 point 数量应以 CLI 的 JSON 输出为准；运行
产物写入被 `.gitignore` 忽略的 `data/evaluation/a4-2/`，不会把完整语料或向量提交
到仓库。该 collection 与受保护的 A2.3/A3 chunk collection role 隔离；保护固定的
`knowledgescope_chunks_v1`，并对已存在且 payload/point identity 可验证为 chunk index
的改名 collection fail closed。写入、重建和删除边界会校验 1024/cosine、payload
contract 和 fingerprint。fingerprint 包含 model、revision、query prompt、L2
normalization、max sequence length 以及实际的 model-native pooling strategy；point 写入
前还会逐点复核这些 active contract 字段。

## 可重复命令

```bash
uv run knowledgescope multimodal-index audit \
  --knowledge-base-id <knowledge-base-uuid> \
  --canonical-root data/benchmarks/a1-5/canonical
uv run knowledgescope multimodal-index build \
  --knowledge-base-id <knowledge-base-uuid> \
  --canonical-root data/benchmarks/a1-5/canonical \
  --limit 3
uv run knowledgescope multimodal-index search "表格中的温度状态是什么?" \
  --knowledge-base-id <knowledge-base-uuid> --modality all --limit 5
```

`build` 对每个 document 使用显式 `knowledge_base_id`，重新生成当前 A4.1 evidence
artifact，并在 A4.2 collection 中执行文档级、先 upsert 后清理 stale 的替换；重复执行
应保持 point 数量和 representation identity 稳定，失败时补偿恢复旧 Evidence/points。
需要同时重解析时，解析服务的 coordinated generation 路径会在 promote 新 canonical
之前完成新的 Evidence/points replacement，并在失败时一起恢复旧 canonical、chunks、
Evidence 和 points；这是一种 generation-level compensated consistency，不是跨文件、
PostgreSQL 与 Qdrant 的全局原子事务。
`search` 的输出同时包含 Evidence-level items 和
有界 raw representation hits，多个表示命中同一 Evidence 时只返回一个 Evidence
结果。

## 验证口径

审计区分：

- source Evidence 总数（包括只有 opaque `asset_ref` 的 Evidence）；
- representation 总数和可搜索 representation 数；
- 当前 collection 中的 indexed、missing、stale 数；
- canonical lineage 构建失败数及 modality 分类。

可搜索文本只来自 canonical text/title、已有 table markdown/html/caption、formula
LaTeX、已有 image caption 或同页已有 text/title context。图片没有 caption/context
时仍保留 source Evidence 和 asset lineage，但不会出现在文本向量检索结果中。

所有正常 Qdrant 查询都在 filter 中强制 `searchable=true`，并排除已 quarantine 的
points；所有返回结果必须满足 A4.1 的
`Representation → Evidence → source block → Page → Document → asset/PDF` 链路；
KB filter 是强制条件。A4.2 的 representation collection 与
`knowledgescope_chunks_v1` 独立，当前不代表视觉 embedding、BM25、最终融合或统一
RAG 检索已经实现。

## 本机验证记录

2026-09-12 使用本地 `data/benchmarks/a1-5/canonical` 做了不写入 Qdrant 的全量来源
物化审计，并使用一份真实 canonical 文档做受控的 Qdrant + Qwen smoke。全量审计的
结果是来源事实，不代表已经完成全量 embedding/index build：

| 项目 | 实际值 |
| --- | --- |
| canonical documents scanned | 255 |
| source Evidence | 46,504 |
| representations / searchable representations | 85,514 / 79,338 |
| full-corpus indexed / missing / stale | 未执行全量 build；不适用 |
| lineage failures in full-corpus audit | 0 |
| real smoke document | `181e7c3f-311e-5da6-89b4-3361880fa554` |
| real smoke source Evidence / searchable representations | 231 / 393 |
| real smoke indexed points | 393 |

受控 smoke 使用临时 KB，仅写入 A4.2 的独立 collection，结束后删除了该文档的 393 个
points。三个有 modality filter 的文本查询都命中了对应 Evidence：image 为 page 1、
source block `p1-b4`；table 为 page 1、source block `p1-b12`；formula 为 page 3、
source block `p3-b6`。返回结果保留了 Evidence ID、representation type、页码和
source block lineage。临时 points 已清理，因此当前 representation collection 的
本机 point count 为 0；这不表示来源 representation 不存在。

这些数字仅是当前 canonical/index 运行事实。A4.2 没有使用质量标注，因此不能据此
推断检索效果、生产性能或视觉检索能力。

生命周期口径：Qdrant 是派生的可检索状态，不与 PostgreSQL、文件系统组成分布式
事务。文档删除先隔离表示；权威删除失败时恢复，提交成功后的物理清理失败则保留
不可检索、可审计的 quarantine points。重解析/重建依赖 document-scoped 的
last-known-good compensation，不作 exactly-once 或跨系统原子切换承诺。
