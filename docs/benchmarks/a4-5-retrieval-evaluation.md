# A4.5 最终检索消融与多模态评测

Phase A4.5 建立只读、可复现的检索评测器。它复用已经冻结的 A2.1 标注、A2.5
Dense+BGE 基线、A3.4 Graph Retrieval、A4.2 representation index 和 A4.4
Unified Retrieval，不修改这些组件的算法、参数或持久化状态，也不调用答案生成
LLM。运行结果只说明当前冻结 profile 下的离线观察，不是生产 SLO 或质量承诺。

## 固定输入与只读边界

通用文本评测使用 A2.1 的 108 条项目（dev 72、test 36），指标沿用
`knowledge_scope.evaluation.retrieval_metrics.ranking_metrics` 的
`Hit@1/3/5/10`、`MRR` 和 `EvidenceRecall@1/3/5/10`。多模态评测使用 A4.1
authoritative Evidence 作为 gold；representation 是可检索的派生内容，不是
权威来源。

运行前会 fail closed 检查并比较已登记的
[`a4-5-frozen-store-manifest.json`](a4-5-frozen-store-manifest.json)：

- A2.1 dataset/chunk index hash、108/72/36 计数和知识库范围；
- `knowledgescope_chunks_v1` 的角色、1.0 schema、1024 维 cosine、Qwen
  revision/config fingerprint、7,524 个 point 及其身份/lineage fingerprint；
- Sparse 当前 active generation、tokenizer/static fingerprint、corpus fingerprint
  和可搜索 chunk 数；
- `knowledgescope_representations_v1` 的 collection fingerprint、79,338 个 point
  和 identity/payload fingerprint；
- A3.6 的知识库、run ID、graph-eligible 文档/Evidence chunk 计数和当前图快照
  fingerprint。

本次冻结身份为：A2.1 dataset SHA-256
`ca5f25fd43fa78973de76e0531a1df8e2c0046c28eb5b6b033438a7d0c34c9ad`、chunk index
SHA-256 `3b9bae28a7e864c858a957043ed9232ec11b1ad635dfd9a94edd5fa7192a8d40`；A2.5
results/manifest SHA-256 分别为
`d59c868b08d9adee8d6bcbb83ce2193a62a4fedad8ab5dff7a1434346587dcc7` 和
`f21a35722a53c71a330508676d265ba7089c8ab1daaa164b69b39a4aaf78fafd`；v2 多模态
dataset fingerprint 为
`467c7a09d5db41c8c02af3562a55cc87fae0ac9bf20be6515b63b10c010ba28c`，文件 SHA-256
为 `217d83f37d0e84033b3c784044ae2734f6136f6b3604a4406f9b911cff9f5359`。冻结存储
清单文件 SHA-256 为
`16df2c177aec5ff3a41b28043dbe50e8ad163ed235bd3844da1248d4924c19ac`，评测协议
fingerprint 为 `12bbd780f93cb76acaeef9eda8b8f768e164dc612782267bb5e6b68124c68e7d`。
这些值来自已接受的当前存储状态，不是运行时重新生成的替代清单。

评测访问 Sparse 使用 SQLite read-only URI；缺失路径不会创建目录、数据库、schema
或 WAL。Qdrant、Sparse、Neo4j、canonical/evidence 和 A2.1 文件均不会被修复、重建
或写入。评测器会在运行前后对 Dense Qdrant、Representation Qdrant、Sparse 和
Neo4j 的物质身份做确定性快照比较；只有比较结果为 unchanged 时，运行 manifest 才会
记录 `stores_mutated=false`，如发现存储或冻结输入发生变化则 fail closed，不发布该次
评测结论。这里表示“未检测到物质冻结存储变更”，不对操作系统层面的 atime 等元数据变化
作绝对保证。Qdrant 当前只作为既有 vector-store 路径验证，7,524 points 的运行不构成
HNSW/ANN 性能 benchmark。

命令：

```bash
uv run knowledgescope retrieval-evaluation build-multimodal-dataset
uv run knowledgescope retrieval-evaluation run \
  --knowledge-base-id <knowledge-base-uuid> \
  --split both
```

## 通用文本系统

固定比较以下已有路径：

1. `dense_bge`：Qwen dense Top-10，再使用 A2.5 BGE Top-10，作为基线；
2. `dense_sparse_bge`：Dense + Sparse，由 A4.4 Unified 完成最终 BGE；
3. `dense_graph_bge`：Dense + Graph，由 A4.4 Unified 完成最终 BGE；
4. `dense_multimodal_bge`：Dense + A4.2 representation，由 A4.4 Unified 完成最终 BGE；
5. `unified`：四条已有分支和一次最终 BGE。

没有在 A4.5 中增加第二次 RRF、学习式融合或新的 reranker。A3.5 的 `rrf_k=60`
只作为上游 provenance 记录。协议 fingerprint 为
`12bbd780f93cb76acaeef9eda8b8f768e164dc612782267bb5e6b68124c68e7d`。

| split / system | n | H@1 | H@3 | H@5 | H@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / `dense_bge` | 72 | 0.7917 | 0.8750 | 0.9167 | 0.9722 | 0.8494 |
| dev / `dense_sparse_bge` | 72 | 0.7917 | 0.8889 | 0.9306 | 0.9444 | 0.8513 |
| dev / `dense_graph_bge` | 72 | 0.7917 | 0.8750 | 0.9167 | 0.9583 | 0.8476 |
| dev / `dense_multimodal_bge` | 72 | 0.1667 | 0.6111 | 0.7917 | 0.8611 | 0.4101 |
| dev / `unified` | 72 | 0.1667 | 0.6250 | 0.8056 | 0.8750 | 0.4144 |
| test / `dense_bge` | 36 | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 |
| test / `dense_sparse_bge` | 36 | 0.8056 | 0.9722 | 1.0000 | 1.0000 | 0.8852 |
| test / `dense_graph_bge` | 36 | 0.7778 | 0.9167 | 0.9444 | 0.9444 | 0.8449 |
| test / `dense_multimodal_bge` | 36 | 0.1667 | 0.4722 | 0.7500 | 0.8611 | 0.3881 |
| test / `unified` | 36 | 0.1944 | 0.5278 | 0.8611 | 1.0000 | 0.4393 |

all-108 的描述性结果为：`dense_bge` H@1/3/5/10=`0.7870/0.8611/0.8981/0.9352`、
MRR=`0.8355`；`dense_sparse_bge`=`0.7963/0.9167/0.9537/0.9630`、`0.8626`；
`dense_graph_bge`=`0.7870/0.8889/0.9259/0.9537`、`0.8467`；
`dense_multimodal_bge`=`0.1667/0.5648/0.7778/0.8611`、`0.4028`；
`unified`=`0.1759/0.5926/0.8241/0.9167`、`0.4227`。完整
EvidenceRecall、逐 query 结果和 query-type slices 保存在运行产物中。

通用系统的 EvidenceRecall@1/3/5/10（按 query 取平均）为：

| split / system | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | ---: | ---: | ---: | ---: |
| dev / `dense_bge` | 0.7917 | 0.8750 | 0.9074 | 0.9722 |
| dev / `dense_sparse_bge` | 0.7917 | 0.8889 | 0.9213 | 0.9444 |
| dev / `dense_graph_bge` | 0.7917 | 0.8750 | 0.9074 | 0.9583 |
| dev / `dense_multimodal_bge` | 0.4606 | 0.8403 | 0.9236 | 0.9306 |
| dev / `unified` | 0.4606 | 0.8403 | 0.9236 | 0.9306 |
| test / `dense_bge` | 0.7778 | 0.8333 | 0.8611 | 0.8611 |
| test / `dense_sparse_bge` | 0.8056 | 0.9722 | 1.0000 | 1.0000 |
| test / `dense_graph_bge` | 0.7778 | 0.9167 | 0.9444 | 0.9444 |
| test / `dense_multimodal_bge` | 0.3981 | 0.7778 | 0.8889 | 0.9722 |
| test / `unified` | 0.3981 | 0.7778 | 0.8889 | 1.0000 |
| all / `dense_bge` | 0.7870 | 0.8611 | 0.8920 | 0.9352 |
| all / `dense_sparse_bge` | 0.7963 | 0.9167 | 0.9475 | 0.9630 |
| all / `dense_graph_bge` | 0.7870 | 0.8889 | 0.9198 | 0.9537 |
| all / `dense_multimodal_bge` | 0.4398 | 0.8194 | 0.9120 | 0.9444 |
| all / `unified` | 0.4398 | 0.8194 | 0.9120 | 0.9537 |

### 候选恢复与最终排序变化

下面的恢复是“某分支候选中是否出现 gold chunk”的 query-level 统计，只用于审计
候选覆盖，不能替代最终排序指标。`Dense/Sparse/Graph/Multimodal` 在 test 分别
覆盖 31/34/13/0 条 query；对应 gold source block 覆盖为 31/34/13/34。test 的
安全示例 eval ID 为：Sparse
`a2-1-143850af541ec6696efef05fb6c6a0ef4de76c8fd2ae45ac75a824e1bd2a6f23`、
`a2-1-30a2d7548675d89cad0f1d4446d2367973ad3a6d2f84f5d454508967e09e8790`，以及
Graph
`a2-1-675c5d9343d1f85dd03b0a8bfa69e8ecf6273fa573086cbb955f3aa874373bf0`；记录中
只保留 ID 和分支元数据，不保留教材正文。

| split | query 数 | Dense gold chunk | Sparse | Graph | Multimodal | 无分支命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 72 | 70 | 67 | 15 | 0 | 1 |
| test | 36 | 31 | 34 | 13 | 0 | 0 |
| all | 108 | 101 | 101 | 28 | 0 | 1 |

以下每格为“改善/不变/回退”（I/U/R），三数之和等于该 split 的 query 数；MRR
以分数上升为改善，first relevant rank 以名次变小为改善。这样既能核对 aggregate
中的 H@1/3/5/10，也能核对 MRR 和首个相关 chunk 名次，且不会把候选出现误报为最终
质量提升。

| split | system | H@1 I/U/R | H@3 I/U/R | H@5 I/U/R | H@10 I/U/R | MRR I/U/R | first rank I/U/R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `dense_sparse_bge` | 1/70/1 | 1/71/0 | 1/71/0 | 1/68/3 | 1/67/4 | 1/67/4 |
| dev | `dense_graph_bge` | 0/72/0 | 0/72/0 | 0/72/0 | 0/71/1 | 0/71/1 | 0/71/1 |
| dev | `dense_multimodal_bge` | 0/27/45 | 0/53/19 | 0/63/9 | 0/64/8 | 0/15/57 | 0/15/57 |
| dev | `unified` | 0/27/45 | 1/52/19 | 1/62/9 | 1/63/8 | 1/14/57 | 1/14/57 |
| test | `dense_sparse_bge` | 3/31/2 | 5/31/0 | 5/31/0 | 5/31/0 | 5/28/3 | 5/28/3 |
| test | `dense_graph_bge` | 1/34/1 | 3/33/0 | 3/33/0 | 3/33/0 | 3/32/1 | 3/32/1 |
| test | `dense_multimodal_bge` | 0/14/22 | 0/23/13 | 0/32/4 | 0/36/0 | 0/11/25 | 0/11/25 |
| test | `unified` | 1/13/22 | 2/21/13 | 4/28/4 | 5/31/0 | 5/6/25 | 5/6/25 |

逐 query 的 query-type 分片也写入 `general-query-results.jsonl` 和
`aggregate.json`；分片样本很小，不作单独质量结论。

## 多模态 v2 评测集

仓库安全的 v2 文件是
[`a4-5-multimodal-eval-v2.jsonl`](a4-5-multimodal-eval-v2.jsonl)，manifest 是
[`a4-5-multimodal-eval-v2-manifest.json`](a4-5-multimodal-eval-v2-manifest.json)。
它从当前 A4.1/A4.2 authoritative Evidence 派生，固定为 72 条：dev/test 各 36，
image/table/formula 为 38/24/10，9 个学科各 8 条；dev/test 文档分布为 31/27，
文档交集为 0，重复 query、近重复 query 和重复 gold Evidence 均为 0。查询来源为
`source_derived_template_assisted`，只使用章节/标题等独立上下文，不复制或截断
representation 正文；构建时的非搜索 Evidence 淘汰 219 条，最终泄漏审计为 0。
v2 manifest 同时记录 `leakage_checked_item_count=72`、`leakage_item_count=0` 和
`leakage_audit_passed=true`。
每次加载还会重新校验当前 canonical、Evidence、representation、page/block、
asset 和 source/canonical fingerprint，manifest 本身不能替代权威校验。

v2 是用于流程和检索路径核验的 source-derived 集合，需要后续人工语义核验，不是
人工质量 gold。Evidence-level 结果如下；`dense_bge` 普通 text chunk 命中相同
source block 也不算命中 authoritative multimodal Evidence。

| split / system | n | Evidence H@1 | H@3 | H@5 | H@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / `dense_bge` | 36 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev / `multimodal_representation` | 36 | 0.0000 | 0.2222 | 0.3611 | 0.5000 | 0.1538 |
| dev / `unified` | 36 | 0.0000 | 0.0556 | 0.1667 | 0.3889 | 0.0710 |
| test / `dense_bge` | 36 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test / `multimodal_representation` | 36 | 0.0833 | 0.1667 | 0.4167 | 0.5833 | 0.1929 |
| test / `unified` | 36 | 0.0000 | 0.3333 | 0.4444 | 0.6111 | 0.1909 |

这里每条 item 只有一个 gold Evidence，因此 EvidenceRecall 与 Evidence Hit 的数值
一致。成功返回的 Evidence 会保留 `evidence_id`、KB/document/page/source blocks、
modality、asset refs、representation IDs 和分支审计；应用不会把 representation
正文或内部 ID 当作权威来源。

v2 的 test 按模态结果如下；`EvidenceRecall@K` 与同一行的 H@K 完全相同。`formula`
只有 3 条，结论不稳定，应仅作诊断观察。

| system | modality | n | H@1 | H@3 | H@5 | H@10 | MRR | EvidenceRecall@1/3/5/10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_bge` | image | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/0/0 |
| `dense_bge` | table | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/0/0 |
| `dense_bge` | formula | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/0/0 |
| `multimodal_representation` | image | 19 | 0.0263 | 0.1579 | 0.3684 | 0.5789 | 0.1604 | 0.0263/0.1579/0.3684/0.5789 |
| `multimodal_representation` | table | 14 | 0.0833 | 0.2500 | 0.4167 | 0.5417 | 0.2125 | 0.0833/0.2500/0.4167/0.5417 |
| `multimodal_representation` | formula | 3 | 0.0000 | 0.2000 | 0.4000 | 0.4000 | 0.1283 | 0/0.2/0.4/0.4 |
| `unified` | image | 19 | 0.0000 | 0.1842 | 0.2632 | 0.5000 | 0.1256 | 0/0.1842/0.2632/0.5000 |
| `unified` | table | 14 | 0.0000 | 0.2500 | 0.3750 | 0.5417 | 0/0.25/0.375/0.5417 |
| `unified` | formula | 3 | 0.0000 | 0.1000 | 0.3000 | 0.4000 | 0.1093 | 0/0.1/0.3/0.4 |

v2 citation audit 没有无效 lineage：`multimodal_representation` 为
`valid=39/not_applicable=33`，`unified` 为 `valid=36/not_applicable=36`；未命中
不是 citation 验证成功。每个 valid hit 都重新解析当前 authoritative Evidence，核对
evidence/modality/document/page/source blocks/asset refs。

## 延迟与审计产物

以下是本机 warm execution 的 query-level wall-clock 均值/P50/P95，单位毫秒；不含
答案生成 LLM。Unified total 是实际墙钟时间，不把并发分支延迟相加。

| system | dev mean / P50 / P95 | test mean / P50 / P95 |
| --- | --- | --- |
| `dense_bge` | 172.2 / 179.2 / 210.4 | 166.1 / 179.1 / 186.7 |
| `dense_sparse_bge` | 455.6 / 433.5 / 614.8 | 450.9 / 461.1 / 606.7 |
| `dense_graph_bge` | 2404.7 / 2400.3 / 2653.5 | 2449.7 / 2445.5 / 2708.3 |
| `dense_multimodal_bge` | 1840.7 / 1793.1 / 2739.0 | 1763.7 / 1743.8 / 2297.3 |
| `unified` | 4185.7 / 4124.2 / 5310.1 | 4151.7 / 4184.5 / 4549.1 |

逐 query JSONL 会同时保存最终 `candidate_id/score/rank` 与各分支的安全审计：Dense
原生分数和 rank、Sparse BM25 与 matched terms、Graph 的 evidence/rank/score/seed/
path/hop，以及 Multimodal 的 Evidence、representation、rank/similarity/modality。
所有返回的 chunk/Evidence lineage 来自当前服务结果并接受 KB 范围检查。

运行产物写入被忽略的 `data/evaluation/a4-5/`：

- `general-query-results.jsonl`：108 条通用 query 的五系统结果；
- `multimodal-query-results.jsonl`：72 条多模态 query 的三系统结果；
- `aggregate.json`：split、all-108、指标、EvidenceRecall、分支贡献、延迟和冻结输入；
- `manifest.json`：运行协议 fingerprint 和只读声明。

## 旧 v1 与结论边界

[`a4-5-multimodal-eval-v1.jsonl`](a4-5-multimodal-eval-v1.jsonl) 及其 manifest 保留
为历史 exploratory evidence，并明确标记为
`historical_exploratory_invalid_for_final_claims`：v1 查询直接复用了 gold
representation text，不能用于最终多模态质量结论。v1 的任何旧指标都不出现在本
报告的最终结果中。

本次 v2 结果只证明评测器能够在当前真实存储状态上执行受控、可追溯的离线比较，不能
宣称“多路统一检索整体提升”、多模态质量提升或 representation 的泛化效果。后续如
需质量结论，应先完成人工语义核验，并继续使用相同的 A2.1、存储快照和权威 lineage
审计口径。
