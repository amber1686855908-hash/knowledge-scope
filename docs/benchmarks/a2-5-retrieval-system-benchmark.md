# A2.5 检索系统基准报告

## 结论

本阶段在冻结的 A2.1 `retrieval-eval-v1` 上完成检索阶段比较，没有修改标注、没有重新运行 MinerU，也没有选择新的 embedding 或 reranker 模型。比较对象是：

1. `Qwen/Qwen3-Embedding-0.6B` 的内存 exact dense Top-10；
2. 同一批 chunk 在 A2.3 中持久化的 Qwen 向量上的 Qdrant dense Top-10；
3. Qdrant dense Top-10 后接固定的 `BAAI/bge-reranker-v2-m3`。

这里的 exact dense 与 Qdrant 并没有共享同一份内存向量：exact dense 在本次 benchmark
中重新编码 7,524 个 chunk，Qdrant 使用 benchmark 之前已经写入 collection 的向量；两者
的模型、revision、配置 fingerprint 和维度相同，但向量数值并未逐点复用。两条路径的
query embedding 也分别重新调用同一个 Qwen adapter 生成。因而 dev 的顺序差异不能被
解读为纯粹的 ANN 差异；本次 Qdrant 在该规模实际为 full scan。

在本次 RTX 3060 本地运行中，Qdrant collection 有完整的 `7,524` 个点，且由于点数低于 `full_scan_threshold=10000`，实际使用 full scan。Qdrant test 排名与 exact dense 完全一致，质量指标没有变化；dev 只出现边界排序差异，也没有质量变化。BGE Top-10 在 test 上将 `Hit@1` 从 `0.6389` 提高到 `0.7778`、MRR 从 `0.7313` 提高到 `0.8079`，但 `EvidenceRecall@10` 仍为 `0.8611`。因此当前建议的检索阶段默认 profile 是：

`Qwen/Qwen3-Embedding-0.6B → Qdrant dense Top-10 → BAAI/bge-reranker-v2-m3 → Top-10`

`Alibaba-NLP/gte-multilingual-reranker-base` 只作为 A2.4 已完成实验中的低延迟参考，不是本次 Qdrant 系统的主比较对象。reranker 只能重排已经进入 dense Top-10 的候选，不能找回候选池之外的 gold evidence。

## 冻结输入与环境

- A2.1：`108` 条 verified item，`dev=72`、`test=36`；本次只读取，不写回。
- A1.6：`7,524` 个 chunk；通过既有 A1.5 canonical artifact 和 A1.6 默认策略写入本地 Qdrant，没有重新解析 PDF。
- chunk index SHA-256：`3b9bae28a7e864c858a957043ed9232ec11b1ad635dfd9a94edd5fa7192a8d40`。
- A2.1 dataset SHA-256：`ca5f25fd43fa78973de76e0531a1df8e2c0046c28eb5b6b033438a7d0c34c9ad`。
- Qdrant：`knowledgescope_chunks_v1`，`7,524` points，`1024` 维，`cosine`。
- dense model：`Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
- reranker：`BAAI/bge-reranker-v2-m3`，revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。
- 运行环境：Python `3.12.3`、PyTorch `2.14.0`、Transformers `4.57.6`、Sentence Transformers `5.7.0`、CUDA `13.0`、`NVIDIA GeForce RTX 3060`，总显存约 `11889 MiB`。

## 评测协议

### Dense 阶段

- 文档和 query 使用 A2.2 接受的 Qwen 约定：query 使用 SentenceTransformers `prompt_name="query"`，两侧执行 L2 归一化。
- exact dense 在内存中重新编码 chunk，再执行归一化向量的精确点积，并按分数排序；分数相同时以 `chunk_id` 做确定性 tie-breaker。
- Qdrant 使用 A2.3 已建立的同一 collection 和同一 7,524 点向量。主 benchmark 不使用知识库或文档过滤，以保持全语料比较；另对每个 eval item 做一次 `document_id` filter 检查。
- `dev=72` 与 `test=36` 使用同一份预先固定的模型、Top-10、batch、dtype 和长度配置；本阶段没有根据 test 结果调参，test 只作最终确认。

当前 collection 的 7,524 个 payload 都报告相同的 Qwen model/revision/config fingerprint；抽查一个
chunk 时，持久化向量与本次 fresh encode 均为 1,024 维，但并非 bitwise identical（最大绝对差约
`3.1e-4`）。这说明输入生成配置一致，却不能把两组向量称为同一份存储值。

### Reranker 阶段

- BGE 接收的候选严格是同一条 Qdrant dense Top-10 列表，不能扩展候选集合。
- 使用 BGE 模型卡约定的原始 `(query, passage)` pair，不套用 Qwen 或 E5 的 query instruction。
- 使用 CUDA、`float16`、`batch_size=8`、`max_length=512`；模型加载后先预热一个 pair。
- `Hit@1/3/5/10`、MRR 和 `EvidenceRecall@1/3/5/10` 复用 A2.1 的 `ranking_metrics`。本报告中 MRR 和各指标均基于传入的 Top-10 ranking list。

### 时间与资源口径

- exact dense latency：单条 query 的 Qwen query encoding + 内存精确排序，包含 CUDA 同步；不包含模型加载和 7,524 chunk 的离线编码。
- Qdrant dense latency：单条 query 的 Qwen query encoding + Qdrant readiness/query I/O，包含 CUDA 同步；不包含模型加载和离线索引。
- BGE rerank latency：对已返回的 10 个候选执行 Cross-Encoder scoring 并同步；不包含模型加载和 Qdrant I/O。
- total query latency：一次 Qdrant dense 调用接一次 BGE scoring 的实际连续 wall-clock；不是把不同实验的分数相加，也不包含 benchmark-only 的截断 tokenizer 审计和指标聚合。
- throughput 是各自 split 的 query 数除以对应阶段累计秒数；不是并发服务吞吐或生产 SLO。
- CUDA `allocated/reserved` 是 inference peak；dense 与组合 profile 分开测量。BGE load peak 发生在 Qwen 已驻留时，因此它表示加载 BGE 后的组合驻留峰值，不应被误读为 BGE 权重单独占用。

## 检索质量

下表为主比较。`EvidenceRecall@10` 反映 Top-10 候选覆盖的 gold source blocks；它不会因 reranker 重新排序而超过 dense candidate pool 的覆盖上限。

| 系统 | split | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | EvidenceRecall@1 | EvidenceRecall@3 | EvidenceRecall@5 | EvidenceRecall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact dense Top-10 | dev | 0.6389 | 0.8333 | 0.9028 | 0.9722 | 0.7427 | 0.6389 | 0.8241 | 0.8935 | 0.9722 |
| Qdrant dense Top-10 | dev | 0.6389 | 0.8333 | 0.9028 | 0.9722 | 0.7427 | 0.6389 | 0.8241 | 0.8935 | 0.9722 |
| Qdrant + BGE Top-10 | dev | 0.7917 | 0.8750 | 0.9167 | 0.9722 | 0.8494 | 0.7917 | 0.8750 | 0.9074 | 0.9722 |
| exact dense Top-10 | test | 0.6389 | 0.8056 | 0.8333 | 0.8611 | 0.7313 | 0.6389 | 0.8056 | 0.8333 | 0.8611 |
| Qdrant dense Top-10 | test | 0.6389 | 0.8056 | 0.8333 | 0.8611 | 0.7313 | 0.6389 | 0.8056 | 0.8333 | 0.8611 |
| Qdrant + BGE Top-10 | test | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 | 0.7778 | 0.8333 | 0.8611 | 0.8611 |

相对 Qdrant dense，BGE 的 test 提升为：`Hit@1 +13.89` 个百分点、`Hit@3 +2.78` 个百分点、`Hit@5 +2.78` 个百分点、MRR `+0.0766`；`Hit@10` 和 `EvidenceRecall@10` 不变。dev 的对应 Hit@1 和 MRR 提升分别为 `+15.28` 个百分点和 `+0.1067`。

### GTE 低延迟参考

以下数值来自 A2.4 已完成的本地 Cross-Encoder 基准，候选列表是同一 Qwen exact dense profile，但不是本次 Qdrant system run，故只作为低延迟参考：

| profile | split | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | EvidenceRecall@10 | rerank mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GTE Top-10 reference | dev | 0.7222 | 0.9306 | 0.9722 | 0.9722 | 0.8338 | 0.9722 | 80.2 ms |
| GTE Top-10 reference | test | 0.7500 | 0.8056 | 0.8056 | 0.8611 | 0.7813 | 0.8611 | 77.2 ms |

A2.4 中 GTE test reranker throughput 为 `12.96 query/s`、inference peak 为约 `741/814 MiB`（allocated/reserved）。这两个数值与本次 BGE 组合 profile 的 Qdrant total latency 不可直接相减；GTE 仍是低延迟/较低显存的备选 profile，而非本阶段重新选型。

## Qdrant 与 exact dense

| split | Top-10 顺序完全一致 | 顺序一致率 | 平均 Top-10 集合重叠 | 质量 delta | ANN 相关 miss | filter 检查 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| dev | 67 / 72 | 93.06% | 99.583% | 所有指标 `0.0000` | 0 | 72 次，空结果 0，越界 0 |
| test | 36 / 36 | 100% | 100% | 所有指标 `0.0000` | 0 | 36 次，空结果 0，越界 0 |

dev 的 5 条非完全一致列表中，2 条只是同一 Top-10 集合内的顺序变化，3 条在第 10 名边界各发生一个点的替换；这没有造成任何本次指标变化。由于 exact dense 重新编码了 corpus，而 Qdrant 使用预存向量，这些差异也包含两次独立编码的数值差异。因为本次 collection 在 `7,524 < full_scan_threshold` 下走 full scan，不能把这些差异称为 ANN 近似搜索误差，也不能由此推断更大 collection 的 HNSW 表现。

## 延迟、吞吐与显存

| 系统 | split | retrieval mean | rerank mean | total mean | retrieval q/s | rerank q/s | total q/s | inference peak allocated / reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact dense | dev | 28.5 ms | — | 28.5 ms | 35.11 | — | 35.11 | 1182 / 1588 MiB |
| exact dense | test | 29.3 ms | — | 29.3 ms | 34.16 | — | 34.16 | 1182 / 1588 MiB |
| Qdrant dense | dev | 38.3 ms | — | 38.3 ms | 26.09 | — | 26.09 | 1182 / 1588 MiB |
| Qdrant dense | test | 38.5 ms | — | 38.5 ms | 25.99 | — | 25.99 | 1182 / 1588 MiB |
| Qdrant + BGE Top-10 | dev | 35.9 ms | 187.1 ms | 223.1 ms | 27.84 | 5.35 | 4.48 | 2350 / 2494 MiB |
| Qdrant + BGE Top-10 | test | 35.5 ms | 180.2 ms | 215.8 ms | 28.13 | 5.55 | 4.63 | 2350 / 2494 MiB |

共享初始化成本：Qwen model load `9.45 s`，7,524 chunk corpus encoding `215.51 s`。BGE 在 Qwen 已驻留的进程中加载耗时 `5.01 s`；BGE load peak 为约 `2258/2338 MiB`，是组合驻留口径。模型文件已在本机缓存，load time 不包含网络下载时间。

## 截断与坏例分析

本次 BGE 使用 `max_length=512`，因此应将结果描述为 `512-token common profile`，而不是模型原生长上下文能力。实际 Top-10 candidate pair 中：

- dev：`147 / 720`（`20.42%`）pair 被判定为超过 512 token；`30 / 72` 个 item 含有至少一个可能受截断影响的 gold-relevant candidate；
- test：`61 / 360`（`16.94%`）pair 被判定为超过 512 token；`9 / 36` 个 item 含有至少一个可能受截断影响的 gold-relevant candidate。

A1.6 chunk 只有 `source_block_ids`，没有字符/token offset，因此这些计数只能说明 candidate 输入存在截断风险，不能证明答案文字一定落在窗口外，也不能把截断确定为某条 query 的唯一失败原因。

test 坏例按可观测事实分类如下；类别可以重叠：

| 类别 | 条数 | 含义 |
| --- | ---: | --- |
| evidence absent from dense Top-10 | 5 | gold relevant chunk 没进入 dense 候选，BGE 无法恢复；这正是 Hit@10/EvidenceRecall@10 的上限 |
| reranker ranked relevant lower | 2 | gold 已在候选中，但 BGE 将首个相关 chunk 排在 dense 顺序之后 |
| ambiguous or cross-block query | 1 | 与跨 block/语义歧义相关的 bad case；它同时属于候选池外 evidence 的一条 |
| long/truncated relevant candidate | 0 条 bad case | 失败样本中没有被本次可观测规则单独标出的该类别，但全体候选仍有上述截断风险 |

本阶段没有观察到需要归因于多 chunk 边界的 test bad case。坏例文件只保存 item ID、类别和长度元数据，完整 query/语料/向量均留在被忽略的运行目录。

## 运行与产物边界

安装 embedding/reranker 运行依赖并启动本地服务：

```bash
uv sync --group embedding-benchmark
docker compose up -d qdrant
```

先用已有 canonical artifacts 填充完整 collection（不会运行 MinerU）：

```bash
uv run knowledgescope qdrant index-corpus \
  --canonical-root data/benchmarks/a1-5/canonical
```

运行 A2.5：

```bash
uv run knowledgescope retrieval-system-benchmark --split both
```

评测代码位于 `knowledge_scope.evaluation.retrieval_system_benchmark`，CLI 只写入被 `.gitignore` 忽略的 `data/evaluation/a2-5/`：

- `manifest.json`：协议、输入指纹、环境、Qdrant schema 和结果摘要；
- `results.jsonl`：三个 system 的 dev/test 指标和阶段计时；
- `rankings/`：只含 item ID 与 chunk ID 的运行时排名，不含教材正文；
- `qdrant-comparison.json`：exact/Qdrant 一致性与 filter 检查；
- `bad-cases.jsonl`：test 坏例的 metadata-only 分类。

仓库不提交模型文件、embedding、Qdrant named volume、PDF、完整 canonical/chunk corpus 或 secrets。本报告引用的 GTE 数据来自 A2.4 忽略运行产物；本阶段没有添加 sparse/hybrid retrieval、GraphRAG、LLM/RAG generation 或 frontend。

## 限制与下一步

- test 只有 `36` 条 query，结果适合工程 profile 比较，不代表统计显著性；没有计算置信区间、重复种子或并发服务吞吐。
- Qdrant 本次走 full scan，尚未验证大规模 collection 的 HNSW/ANN 参数、索引构建成本和 recall 变化。
- `512-token common profile` 会限制长 chunk 的绝对效果，且当前 lineage 没有 token span，无法精确回答 gold answer 是否被截断。
- total latency 是当前单机、单 query、batch8 的本地 wall-clock，不是生产部署 SLO。
- 本阶段只完成 retrieval-stage benchmark；sparse/hybrid、GraphRAG、LLM 生成和前端检索仍未实现。
