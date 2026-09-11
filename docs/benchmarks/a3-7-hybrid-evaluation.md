# A3.7 Vector-only 与 Vector + Graph 混合检索评测

## 结论

A3.7 在冻结的 A2.1 评测集上，对照现有 Vector 分支和 A3.5 的 Vector + Graph 分支。两条
路径使用同一 Knowledge Base、同一 Qdrant collection、同一 Qwen embedding、同一 BGE
reranker 和同一查询；唯一增加的是 A3.4 Graph Retrieval 与 A3.5 RRF fusion。评测只测
检索，不调用答案生成 LLM，不修改 A2.1 标签、Qdrant point 或 Neo4j 图。

当前固定运行覆盖 108 条评测项：dev 72、test 36。test Top-10 的 Hit@10 和
EvidenceRecall@10 从 Vector 的 `0.8611` 到 Hybrid 的 `0.8889`（绝对 `+0.0278`），有
1 条 query 在最终 Top-10 得到改善；但 Hit@1/3/5 和 MRR 均下降。dev 的 Top-10 反而从
`0.9722` 降至 `0.9583`，all-108 的 Top-10 不变，早期 cutoff 和 MRR 也明显下降。因此，
“Vector + Graph improves retrieval recall”只能得到**部分支持**：它只描述本次固定 test
Top-10 的局部现象，不支持普遍改善、统计显著性或生产质量结论。

此前使用 Graph/Hybrid 20、Vector candidate 20、rerank 10、batch 4 的探索性结果与本次
固定协议不具有可比性，已明确标为 exploratory，不用于结论或推荐。

## 固定协议与 preflight

### A2.5 Vector 基线

A3.7 先用当前实现重现冻结的 A2.5 `qdrant_dense_bge_top10` 结果，未通过 test 调参；
若重现不匹配，评测会 fail-closed。固定配置为：

| 项目 | 固定值 |
| --- | --- |
| embedding | `Qwen/Qwen3-Embedding-0.6B` |
| embedding revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| embedding query convention | `SentenceTransformers prompt_name=query` |
| document convention | `no query prompt` |
| embedding device/dtype/batch/max length | `cuda` / `float16` / `8` / `512` |
| Qdrant | `knowledgescope_chunks_v1`，7,524 points，1,024 维，cosine |
| dense candidate / rerank | `10` / `10` |
| reranker | `BAAI/bge-reranker-v2-m3` |
| reranker revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| reranker convention/device/dtype/batch/max length | raw query/passage pair / `cuda` / `float16` / `8` / `512` |
| Graph / Hybrid result limit | `20` / `20` |
| RRF | `1 / (rrf_k + rank)`，`rrf_k=60` |

冻结 A2.5 的 reference metrics 与本次 Vector baseline 在 `1e-5` 内一致：

| split | H@1 | H@3 | H@5 | H@10 | MRR | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2.5 dev/reference | 0.7917 | 0.8750 | 0.9167 | 0.9722 | 0.8494 | 0.7917 | 0.8750 | 0.9074 | 0.9722 |
| A2.5 test/reference | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 | 0.7778 | 0.8333 | 0.8611 | 0.8611 |

`data/evaluation/a2-5/manifest.json` 历史上记录了 `dense_batch_size=4`，而已接受的 A2.5
报告和本 A3.7 固定 profile 使用 batch `8`。A3.7 不修改冻结 A2.5 artifact；本评测记录
了 manifest hash，并以已接受报告的 batch-8 profile 运行。query embedding 在评测路径中
逐条调用，因此当前排名重现通过，但该历史 metadata 差异仍是可追溯的限制。

### 只读 preflight

运行前 fail-closed 校验了以下事实：

| 输入 | 校验值 |
| --- | --- |
| A2.1 | 108 items，dev 72 / test 36；9 个 subject；标签和 evidence lineage 未改动 |
| A1.6 corpus | 255 documents / 7,524 unique chunks |
| Qdrant | 7,524 points，chunk universe 与 A1.6 完全一致；point ID 为 `point_id_for_chunk(chunk_id)` |
| Qdrant attribution | 目标 KnowledgeBase、model/revision/config fingerprint、collection schema 均一致 |
| A3.6 graph | 246 eligible documents、6,087 supported chunks；9 个 excluded documents 不参与 retrieval |
| graph counts | 55,098 entities、27,268 relations、1,241 canonical entities、2,609 memberships |
| A2.1 graph coverage | 105 complete / 0 partial / 3 none |

Graph manifest 使用 `run_id=e926e1d3-9050-4911-89ef-1632ed0894c2`、pipeline fingerprint
`a9407046dad32e8da8f2ef4256497b60e8ffdb21771c4075a7c87a013ff70cc7`。当前 A3.6 manifest
缺少 `prompt_layout_version` 时，依据 A3.6 已记录的兼容规则按 `legacy-v1` 解释；其他
未知 layout 会 fail-closed。graph provenance 为 `legacy-v1`、truncation budgets
`1024 → 2048 → 4096`，`cache-v2` 未用于本次运行。

Qdrant 当前 collection 的规模低于 `full_scan_threshold=10000`。所以这里是当前规模的
vector-store/ranking validation，不是 HNSW/ANN 性能 benchmark；Qdrant 与 Neo4j 均只读。

## dev/test 指标

指标沿用 A2.1 的 chunk-level Hit@K、MRR 和 source-block-level EvidenceRecall@K。所有
delta 均为 `Hybrid - Vector`，test 只有 36 条 query，不能据此宣称统计显著性。

| split | n | 系统 | H@1 | H@3 | H@5 | H@10 | MRR | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 72 | Vector | 0.7917 | 0.8750 | 0.9167 | 0.9722 | 0.8494 | 0.7917 | 0.8750 | 0.9074 | 0.9722 |
| dev | 72 | Hybrid | 0.4167 | 0.8056 | 0.9028 | 0.9583 | 0.6226 | 0.4167 | 0.8056 | 0.8981 | 0.9583 |
| dev | 72 | Δ | -0.3750 | -0.0694 | -0.0139 | -0.0139 | -0.2268 | -0.3750 | -0.0694 | -0.0093 | -0.0139 |
| test | 36 | Vector | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 | 0.7778 | 0.8333 | 0.8611 | 0.8611 |
| test | 36 | Hybrid | 0.4722 | 0.6944 | 0.8333 | 0.8889 | 0.6252 | 0.4722 | 0.6944 | 0.8333 | 0.8889 |
| test | 36 | Δ | -0.3056 | -0.1389 | -0.0278 | +0.0278 | -0.1827 | -0.3056 | -0.1389 | -0.0278 | +0.0278 |

### all-108 描述性汇总

| 系统 | H@1 | H@3 | H@5 | H@10 | MRR | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.7870 | 0.8611 | 0.8981 | 0.9352 | 0.8355 | 0.7870 | 0.8611 | 0.8920 | 0.9352 |
| Hybrid | 0.4352 | 0.7685 | 0.8796 | 0.9352 | 0.6234 | 0.4352 | 0.7685 | 0.8765 | 0.9352 |
| Δ | -0.3519 | -0.0926 | -0.0185 | 0 | -0.2121 | -0.3519 | -0.0926 | -0.0154 | 0 |

### Graph-covered / Graph-none 诊断

Graph-covered 的 105 条记录（dev 69、test 36）不是替代全量评测的子集：

| slice / 系统 | n | H@1 | H@3 | H@5 | H@10 | MRR | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graph complete / Vector | 105 | 0.7905 | 0.8571 | 0.8952 | 0.9333 | 0.8356 | 0.7905 | 0.8571 | 0.8889 | 0.9333 |
| graph complete / Hybrid | 105 | 0.4286 | 0.7619 | 0.8762 | 0.9333 | 0.6190 | 0.4286 | 0.7619 | 0.8730 | 0.9333 |
| dev graph complete / Vector | 69 | 0.7971 | 0.8696 | 0.9130 | 0.9710 | 0.8501 | 0.7971 | 0.8696 | 0.9034 | 0.9710 |
| dev graph complete / Hybrid | 69 | 0.4058 | 0.7971 | 0.8986 | 0.9565 | 0.6158 | 0.4058 | 0.7971 | 0.8937 | 0.9565 |
| test graph complete / Vector | 36 | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 | 0.7778 | 0.8333 | 0.8611 | 0.8611 |
| test graph complete / Hybrid | 36 | 0.4722 | 0.6944 | 0.8333 | 0.8889 | 0.6252 | 0.4722 | 0.6944 | 0.8333 | 0.8889 |
| graph none / Vector | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 0.8333 | — | — | — | — |
| graph none / Hybrid | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 0.7778 | — | — | — | — |

Graph-none 只有 dev 的 3 条 query；其 Hit@K 为 `0.6667/1/1/1`，MRR 为 Vector `0.8333`、
Hybrid `0.7778`。该诊断切片不应被解释为 Graph coverage 贡献，也不表示缺失或修改标签。

## Graph 贡献与 rank 变化

以下均为 query-level 统计；gold evidence/chunk 仍以 A2.1 authoritative evidence 为准。
`candidate recovered` 表示 Graph Top-10 找到而 Vector Top-10 未找到，`final recovered`
还要求该结果最终进入 Hybrid Top-10。重复 graph path 按 chunk 去重，不重复贡献 rank。

| split | Vector 找到 gold query | Graph 找到 gold query | 两者都找到 | candidate recovered chunk/block | final recovered chunk/block | 改善/不变/退化 | Graph 无 usable result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 70 | 14 | 14 | 0 / 0 | 0 / 0 | 0 / 71 / 1 | 17 |
| test | 31 | 12 | 10 | 2 / 2 | 1 / 1 | 1 / 35 / 0 | 7 |
| all | 101 | 26 | 24 | 2 / 2 | 1 / 1 | 1 / 106 / 1 | 24 |

Hit@K 的改善/不变/退化表示该 K 下 Hybrid 是否从 Vector 的 miss 变为 hit、保持原
状态、或从 hit 变为 miss：

| split | K | Vector hit query | Hybrid hit query | 改善 | 不变 | 退化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 1 | 57 | 30 | 0 | 45 | 27 |
| dev | 3 | 63 | 58 | 1 | 65 | 6 |
| dev | 5 | 66 | 65 | 1 | 69 | 2 |
| dev | 10 | 70 | 69 | 0 | 71 | 1 |
| test | 1 | 28 | 17 | 1 | 23 | 12 |
| test | 3 | 30 | 25 | 1 | 29 | 6 |
| test | 5 | 31 | 30 | 1 | 33 | 2 |
| test | 10 | 31 | 32 | 1 | 35 | 0 |
| all | 1 | 85 | 47 | 1 | 68 | 39 |
| all | 3 | 93 | 83 | 2 | 94 | 12 |
| all | 5 | 97 | 95 | 2 | 102 | 4 |
| all | 10 | 101 | 101 | 1 | 106 | 1 |

首个相关 chunk 的 rank movement：

| split | both-ranked | Vector miss → Hybrid hit | both missed | Vector hit → Hybrid miss | earlier / unchanged / later | mean rank delta | P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 70 | 0 | 2 | 0 | 1 / 33 / 36 | +0.8143 | 0 / 3 |
| test | 31 | 3 | 2 | 0 | 0 / 17 / 14 | +1.0645 | 0 / 4 |
| all | 101 | 3 | 4 | 0 | 1 / 50 / 50 | +0.8911 | 0 / 4 |

MRR delta 分布也显示早期排名退化：

| split | positive | zero | negative | mean delta | min / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 1 | 35 | 36 | -0.2268 | -0.8000 / +0.1905 |
| test | 3 | 19 | 14 | -0.1827 | -0.8750 / +1.0000 |
| all | 4 | 54 | 50 | -0.2121 | -0.8750 / +1.0000 |

test 的 2 个 Graph-only candidate chunks 中，只有 1 个最终进入 Hybrid Top-10；因此
Graph candidate recall 不等于最终 fused recall，Graph 也不能恢复未进入其 candidate set
的 gold evidence。

## 按 query type 的诊断

下表只列 Hit@10 / MRR；切片很小，特别是 test `comparison=1`、`formula_or_table=2`，
不作稳定性结论。完整 cutoff、EvidenceRecall 和逐 query lineage 位于被忽略的 runtime
artifact。

| split/type | n | Vector H@10 / MRR | Hybrid H@10 / MRR |
| --- | ---: | ---: | ---: |
| dev / comparison | 5 | 1.0000 / 1.0000 | 1.0000 / 0.6667 |
| dev / cross_block | 4 | 0.7500 / 0.7500 | 0.7500 / 0.5625 |
| dev / definition | 16 | 1.0000 / 0.9688 | 1.0000 / 0.7703 |
| dev / explanation | 19 | 1.0000 / 0.7730 | 1.0000 / 0.5379 |
| dev / factual | 17 | 0.9412 / 0.8309 | 0.8824 / 0.6317 |
| dev / formula_or_table | 11 | 1.0000 / 0.8039 | 1.0000 / 0.5417 |
| test / comparison | 1 | 1.0000 / 1.0000 | 1.0000 / 0.5000 |
| test / cross_block | 4 | 0.7500 / 0.7500 | 1.0000 / 0.7500 |
| test / definition | 9 | 0.7778 / 0.7778 | 0.7778 / 0.4526 |
| test / explanation | 6 | 1.0000 / 0.6806 | 1.0000 / 0.5250 |
| test / factual | 14 | 0.9286 / 0.9286 | 0.9286 / 0.7702 |
| test / formula_or_table | 2 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |

## 延迟口径

以下为同一进程、同一 GPU、同一 Qdrant/Neo4j 工作区的 warm-up 后逐 query 测量。模型
加载不计入结果；mean 是 query mean，P50/P95 是逐 query total 的顺序统计量，不是
生产 SLA，也不包含 LLM answer generation。

| 路径 | mean total (ms) | P50 (ms) | P95 (ms) | 分阶段 mean (ms) |
| --- | ---: | ---: | ---: | --- |
| Vector dev | 199.936 | 209.310 | 237.626 | embedding 16.181；Qdrant search 14.391；BGE rerank 168.971 |
| Vector test | 193.584 | 207.942 | 215.267 | embedding 16.238；Qdrant search 14.253；BGE rerank 162.720 |
| Hybrid dev | 2,251.656 | 2,223.981 | 2,442.341 | 并行 Vector branch 214.519；Graph branch 2,250.539；RRF fusion 0.895 |
| Hybrid test | 2,258.250 | 2,244.101 | 2,415.413 | 并行 Vector branch 207.703；Graph branch 2,257.056；RRF fusion 0.965 |

Vector total 包含 query embedding、Qdrant search、BGE reranking 和本地编排；Qdrant stage
只包 search，不包 readiness。Hybrid 的 Vector 与 Graph 从同一请求中并发启动，total 是
两分支开始到 fusion 完成的 wall-clock，因此不能将两个 branch mean 相加。Graph latency
包含现有 A3.4 seed resolution、Neo4j 读取和本地编排。

local model adapter 的同步推理和 I/O 在线程边界执行；返回 NumPy 前的 adapter 行为会
把所需设备工作纳入服务调用边界，但评测没有额外的显式 `torch.cuda.synchronize()` 或
内核级 profiler。因此上述是工程服务边界测量，而非 CUDA kernel latency。首次 warm-up
已在正式计时前同时覆盖 Vector、Graph 和 fusion。

## 安全边界与可复现产物

- strict failure mode 下，Vector/Graph 失败或超时会 fail-closed，不把失败分支改写成成功的空结果；正常 no-seed/no-evidence 是 Graph 的成功空分支。
- A3.5 fused result 按 `(knowledge_base_id, document_id, chunk_id)` 去重，并保留 branch provenance、rank、RRF contribution、Graph seed/path 和 evidence lineage。
- 评测不调用 DeepSeek、不调用答案生成 LLM，不写 Qdrant/Neo4j，不运行 MinerU，也不修改冻结 A2.1 数据。
- `data/evaluation/a3-7/per-query.jsonl` 与 `data/evaluation/a3-7/aggregate.json` 是被 `.gitignore` 排除的 runtime artifacts；不提交模型、embedding、Qdrant 数据、Neo4j 数据、语料、PDF 或 secrets。

复现命令：

```bash
KNOWLEDGE_SCOPE_NEO4J_PASSWORD=knowledgescope \
uv run knowledgescope hybrid-evaluation \
  --knowledge-base-id 3593a2ee-a326-5a0a-89a2-9a3012666c83 \
  --split both \
  --output data/evaluation/a3-7
```

runtime `aggregate.json` 保存 protocol、preflight、A2.1/A2.5/A3.5/A3.6 provenance、输入
fingerprint、分组 metrics、contribution、rank movement、MRR distribution 和 latency；
`per-query.jsonl` 保存 108 条 bounded query/rank/lineage/latency 记录。当前输入 hash
包括：A2.1 dataset `ca5f25fd43fa78973de76e0531a1df8e2c0046c28eb5b6b033438a7d0c34c9ad`、
A2.5 results `d59c868b08d9adee8d6bcbb83ce2193a62a4fedad8ab5dff7a1434346587dcc7`、A3.6
manifest `4e16818c3a42f315007fde154c24c5432c542fe41ed5c7a445ae3941addc56dd`，详见 ignored
aggregate。

验证命令：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest

cd frontend
npm run lint
npm run type-check
npm run build
cd ..

docker compose config --quiet
git diff --check
```

真实 A3.7 evaluator 的 Qdrant/Neo4j preflight 和只读评测已完成；可变更存储的集成测试不
在本阶段强制执行，以保持“评测期间不修改 Qdrant/Neo4j”的边界。
