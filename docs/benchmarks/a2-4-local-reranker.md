# A2.4 本地 Reranker 基准报告

## 结论

A2.4 在冻结的 A2.1 评测集上完成了本地 Cross-Encoder reranking 基准。三
个候选模型都在本机 RTX 3060 12 GB、CUDA、`float16` 下成功完成
`dev=72` 和 `test=36` 的 Top-10/20/50 比较，没有 OOM 或加载失败。

模型选择以 dev 为主：`BAAI/bge-reranker-v2-m3` 在 dev 上取得最高的
`Hit@1` 和 `MRR`，因此是默认的质量/成本平衡模型；
`Alibaba-NLP/gte-multilingual-reranker-base` 延迟和显存明显更低，适合作为
低延迟 profile。`Qwen/Qwen3-Reranker-0.6B` 可以运行，但在本协议下没有
BGE 的质量或 GTE 的延迟优势。test 只有 36 条 query，只用于最终确认，不能
视为统计显著性结论或独立调参依据。

## 候选模型与输入约定

候选均使用本地 `SentenceTransformers CrossEncoder` 适配器：

| 模型 key | 模型 | query/passage 约定 | 选择理由 |
| --- | --- | --- | --- |
| `qwen3-reranker-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | 使用模型卡默认 `prompt_name="query"`，其默认 instruction 为 `Given a web search query, retrieve relevant passages that answer the query` | 参数规模较小，与当前 Qwen dense pipeline 生态一致 |
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` | 原始 query/passage pair，不额外套用其他模型的 instruction | 中文和多语言检索场景、成熟的 Cross-Encoder 用法 |
| `gte-multilingual-reranker-base` | `Alibaba-NLP/gte-multilingual-reranker-base` | 原始 query/passage pair，不额外套用其他模型的 instruction | 多语言覆盖和较低的本地资源成本 |

官方用法参考：[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)、
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 和
[gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)。
GTE 需要 `trust_remote_code=True`；运行结果记录了实际 revision，生产环境
应通过 `KNOWLEDGE_SCOPE_RERANKER_MODEL_REVISION` 显式固定 revision。

## 冻结输入与基准协议

- 读取 A1.6 的 `7524` 个 chunk 和 A2.1 的 `108` 条 verified item，不修改
  A2.1 ground truth，也不重新运行 MinerU。
- A2.1 split 固定为 `dev=72`、`test=36`；先用同一套
  `Qwen/Qwen3-Embedding-0.6B` 生成一次 dense Top-50 候选，所有 reranker
  看到完全相同的候选池。
- dense query 使用既有 `prompt_name="query"` 约定，文档和 query embedding
  做 L2 归一化，使用内存中的精确点积排序；chunk ID 作为分数相同时的确定性
  tie-breaker。
- 每个 reranker 使用 `batch_size=8`、`max_seq_length=512`、CUDA、fp16；
  预热 1 个 pair 后，分别测量 dense Top-10、Top-20、Top-50 候选池。本报告
  将它明确称为 `512-token common profile`，而不是任何模型的原生最大上下文
  能力；实际 pair 预处理使用 `longest_first` 截断。
- 质量计算复用 A2.1 的 `ranking_metrics`：`Hit@1/3/5/10`、`MRR` 和
  `EvidenceRecall@1/3/5/10`。表格为 split 内逐条结果的平均值。
- 运行环境和输入指纹：Python `3.12.3`、PyTorch `2.14.0`、Transformers
  `4.57.6`、Sentence Transformers `5.7.0`、Accelerate `1.14.0`、
  `NVIDIA GeForce RTX 3060`，CUDA `13.0`，总显存约 `11889 MiB`。
- chunk index SHA-256：
  `3b9bae28a7e864c858a957043ed9232ec11b1ad635dfd9a94edd5fa7192a8d40`。
- A2.1 dataset SHA-256：
  `ca5f25fd43fa78973de76e0531a1df8e2c0046c28eb5b6b033438a7d0c34c9ad`。

### 候选池上限

reranker 只能重排已经进入 dense 候选池的 chunk，不能恢复初始 dense 检索
没有召回的 evidence。以完整 Top-50 dense 候选计算的 EvidenceRecall@50 为：

| split | EvidenceRecall@50 | 完整覆盖 gold evidence 的 item |
| --- | ---: | ---: |
| dev | 0.9861 | 71 / 72 |
| test | 0.9722 | 35 / 36 |

因此至少有 1 条 dev 和 1 条 test query 的完整 gold evidence 不在 Top-50
候选池内；reranker 对这类 item 的上限已经由 dense 阶段决定。

## Dense-only 基线

这是同一 Top-50 候选池的 dense-only 排名；reranker 的结果不应与全量
`7524` 个 chunk 直接混淆。

| split | Hit@1 | Hit@3 | MRR | EvidenceRecall@10 |
| --- | ---: | ---: | ---: | ---: |
| dev | 0.6389 | 0.8333 | 0.7430 | 0.9722 |
| test | 0.6389 | 0.8056 | 0.7384 | 0.8611 |

### 与 A2.2 dense 基线的口径对齐

- A2.2 的 `0.7313` 是 `Qwen/Qwen3-Embedding-0.6B` 精确 dense **Top-10
  评估视野**上的 test MRR；A2.4 的 `0.7384` 是同一 dense 排名在
  **Top-50 评估视野**上的 MRR。`MRR` 会在传入的完整排名列表中继续寻找
  第一个相关 chunk，因此 Top-50 可以发现位于第 11--50 位的相关 chunk。
- A2.4 的 benchmark dense 阶段没有使用 Qdrant，而是重新编码同一冻结 chunk
  index 后在内存中做精确点积排序；test 的 Top-10 排名和 A2.2 一致，Top-10
  MRR 仍为 `0.7313`。因此 `0.7384` 是评估 cutoff 不同造成的预期差异，
  不是 Qdrant 改变了 dense 排名或质量。
- dev 的 Top-10 有 `7/72` 条列表因 A2.4 对空文本 asset-only chunk 使用了
  section/content-type carrier 且发生确定性 tie-break 变化；dev MRR 由
  `0.7427` 变为 `0.7430`，Hit 和 EvidenceRecall 表现未改变。这是当前
  A2.4“实际 reranker passage carrier”协议与 A2.2 直接使用 `chunk.text`
  的小幅输入差异，已在本报告中单独标注，不能归因于 Qdrant。

## 检索质量

下表展示 `Hit@1`、`Hit@3`、`MRR` 和 `EvidenceRecall@10`；被忽略的
`data/evaluation/a2-4/results.jsonl` 同时保存所有 `@1/3/5/10` 指标。
其中 `MRR` 在该行完整的 Top-10/20/50 排名列表上计算，而 `Hit@1/3` 和
`EvidenceRecall@10` 使用各自标注的 cutoff；不同候选数的 MRR 不应与 A2.2
的 Top-10 MRR 直接横向比较。

### Dev（72 条）

| 模型 | rerank 候选数 | Hit@1 | Hit@3 | MRR | EvidenceRecall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Reranker-0.6B` | 10 | 0.7500 | 0.9167 | 0.8409 | 0.9722 |
| `Qwen/Qwen3-Reranker-0.6B` | 20 | 0.7500 | 0.9167 | 0.8361 | 0.9444 |
| `Qwen/Qwen3-Reranker-0.6B` | 50 | 0.7639 | 0.9167 | 0.8434 | 0.9444 |
| `BAAI/bge-reranker-v2-m3` | 10 | 0.7917 | 0.8750 | 0.8494 | 0.9722 |
| `BAAI/bge-reranker-v2-m3` | 20 | 0.7917 | 0.8750 | 0.8477 | 0.9444 |
| `BAAI/bge-reranker-v2-m3` | 50 | 0.7778 | 0.8750 | 0.8392 | 0.9306 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 10 | 0.7222 | 0.9306 | 0.8338 | 0.9722 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 20 | 0.7083 | 0.9306 | 0.8258 | 0.9722 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 50 | 0.7083 | 0.9444 | 0.8278 | 0.9861 |

### Test（36 条）

| 模型 | rerank 候选数 | Hit@1 | Hit@3 | MRR | EvidenceRecall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Reranker-0.6B` | 10 | 0.6944 | 0.8333 | 0.7708 | 0.8611 |
| `Qwen/Qwen3-Reranker-0.6B` | 20 | 0.7500 | 0.8889 | 0.8281 | 0.9444 |
| `Qwen/Qwen3-Reranker-0.6B` | 50 | 0.7778 | 0.9167 | 0.8531 | 0.9722 |
| `BAAI/bge-reranker-v2-m3` | 10 | 0.7778 | 0.8333 | 0.8079 | 0.8611 |
| `BAAI/bge-reranker-v2-m3` | 20 | 0.8333 | 0.9167 | 0.8759 | 0.9444 |
| `BAAI/bge-reranker-v2-m3` | 50 | 0.8611 | 0.9444 | 0.9037 | 0.9722 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 10 | 0.7500 | 0.8056 | 0.7812 | 0.8611 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 20 | 0.8056 | 0.8611 | 0.8405 | 0.8889 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 50 | 0.8333 | 0.8889 | 0.8645 | 0.9167 |

在 test 上，BGE 在三个候选池大小下的 `Hit@1` 和 `MRR` 都高于另外
两个候选；Top-50 的数值最高，但它也带来更高的 reranker 成本。Top-20
可以作为后续工程试验的折中点，但最终线上候选数不在 A2.4 内确定。

## Dev-first 选型结论

模型和候选池的预选只依据 dev；test 在协议固定后才运行，不能反过来调参。
上面的 dev 全表是本阶段的完整模型/候选数比较：BGE Top-10 的 `MRR=0.8494`、
`Hit@1=0.7917`，高于 Qwen 在 Top-50 的最佳 `MRR=0.8434`、`Hit@1=0.7639`，
也高于 GTE 在 Top-10 的 `MRR=0.8338`、`Hit@1=0.7222`。GTE 的 `Hit@3`
较高（Top-10/20 为 `0.9306`），这是低延迟 profile 的质量取舍，而不是
BGE 被所有指标绝对支配。

候选数方面，dev 并不支持把 Top-20 说成质量最优：BGE Top-10 与 Top-20 的
`Hit@1/3` 持平，但 Top-10 的 MRR 和 `EvidenceRecall@10` 更高，Top-50 更低。
因此严格的 dev-first 默认 profile 是 BGE + Top-10。BGE + Top-20 可以保留为
一个需要后续 SLO 验证的 provisional balanced profile；test 中它相对 Top-10
出现质量提升只能作为协议固定后的确认，不能作为 test 调参依据，也不足以证明
它是 A2.4 唯一或最终线上默认值。

## Reranker 输入与截断审计

三种模型均使用实际 Cross-Encoder 预处理：Qwen 只在 query 侧注入模型卡默认
`query` instruction，BGE/GTE 使用原始 query/passage pair；三者均配置
`max_seq_length=512`，tokenizer 使用 `longest_first`，最终输入上限为 512
tokens。为避免把 query 无关的数字误称为真实 pair 结果，下面分别给出
7,524 个 chunk 的空 query 配对代理，以及实际 benchmark candidate pairs。

| 模型 | 7,524 chunk 空 query 配对代理中被截断 | dev Top-10 / Top-20 / Top-50 candidate pairs | test Top-10 / Top-20 / Top-50 candidate pairs |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen3-Reranker-0.6B` | 1,143 / 7,524（15.191%） | 161/720（22.361%） / 312/1,440（21.667%） / 728/3,600（20.222%） | 64/360（17.778%） / 132/720（18.333%） / 315/1,800（17.500%） |
| `BAAI/bge-reranker-v2-m3` | 982 / 7,524（13.052%） | 147/720（20.417%） / 285/1,440（19.792%） / 643/3,600（17.861%） | 61/360（16.944%） / 117/720（16.250%） / 273/1,800（15.167%） |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 982 / 7,524（13.052%） | 147/720（20.417%） / 285/1,440（19.792%） / 643/3,600（17.861%） | 61/360（16.944%） / 117/720（16.250%） / 273/1,800（15.167%） |

真实 candidate pair 中含有至少一个 dense-retrieved gold-relevant chunk 的 item，
其对应 pair 发生截断的数量如下；分母是该 split 的全部 item，括号内是其中确实
含有 gold candidate 的 item 数：

| 模型 | dev Top-10 / Top-20 / Top-50 | test Top-10 / Top-20 / Top-50 |
| --- | ---: | ---: |
| `Qwen/Qwen3-Reranker-0.6B` | 31/72（31/70） / 31/72（31/70） / 32/72（32/71） | 9/36（9/31） / 11/36（11/34） / 11/36（11/35） |
| `BAAI/bge-reranker-v2-m3` | 30/72（30/70） / 30/72（30/70） / 31/72（31/71） | 9/36（9/31） / 10/36（10/34） / 10/36（10/35） |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 30/72（30/70） / 30/72（30/70） / 31/72（31/71） | 9/36（9/31） / 10/36（10/34） / 10/36（10/35） |

A1.6 chunk index 保存了 `source_block_ids`，但没有字符/ token span offset，
所以不能可靠断言某个 gold block 的答案文字已经落在截断窗口之外。上表应解读为
“gold-containing candidate 可能受截断影响”的 item 数，而不是已确认的 gold
evidence 丢失数。截断比例不低，会限制绝对质量分数及对原生长上下文能力的外推；
但三种模型使用同一个 512-token common profile，候选列表和截断统计可比，因此
当前 BGE 的相对选择仍应准确表述为“在 512-token common profile 下成立”，而不是
长上下文能力的结论。

## 延迟、吞吐与显存

Dense 阶段在本次完整运行中加载耗时 `6.638 s`，编码 7,524 个 chunk 耗时
`200.269 s`（约 `37.6 chunks/s`）；单 query dense 编码加精确搜索平均
`19.46 ms`，吞吐 `51.39 query/s`，峰值 CUDA allocated/reserved 为
`1710.1 / 1874.0 MiB`。

### 测量边界复核

- 模型加载前、每个 candidate size 开始前都重置 CUDA peak counter；加载和每次
  scoring 之后都执行 CUDA synchronize。模型加载峰值只覆盖构造/加载，inference
  峰值覆盖模型常驻后的 scoring 临时 buffer。
- 每个模型先预热 1 个 pair；表中 latency 从单条 query 的 scoring 调用开始，
  到 scoring 完成并同步结束，不包含模型加载、dense 检索或 Qdrant I/O。三个模型
  使用相同 `batch_size=8`、相同 query 数、相同 candidate size 和相同计时边界。
- `pairs/s` 是总 reranker pair 数除 scoring 总秒数，`query/s` 是 query 数除同一
  总秒数；这两个值不是端到端服务吞吐。每个 candidate size 的首条请求仍属于
  测量范围，因此结果是固定协议下的真实批处理成本，不宣称是并发稳态吞吐。

下表为 test split 的 reranker 阶段。延迟是单条 query 对该候选数执行
Cross-Encoder scoring 并完成 CUDA 同步的 `mean / p50 / p95`；不包含模型加载、
dense 检索和 Qdrant 网络 I/O。`pairs/s` 是 reranker pair 吞吐，`query/s`
是 reranker query 吞吐，不是端到端服务吞吐。

| 模型 | 候选数 | 延迟 mean / p50 / p95（ms） | pairs/s | query/s | inference peak allocated / reserved（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Reranker-0.6B` | 10 | 432.0 / 462.9 / 484.2 | 23.1 | 2.31 | 1698.8 / 1828.0 |
| `Qwen/Qwen3-Reranker-0.6B` | 20 | 738.8 / 740.9 / 918.7 | 27.1 | 1.35 | 1701.2 / 1828.0 |
| `Qwen/Qwen3-Reranker-0.6B` | 50 | 1647.1 / 1608.6 / 2030.8 | 30.4 | 0.61 | 1701.2 / 1828.0 |
| `BAAI/bge-reranker-v2-m3` | 10 | 165.7 / 181.6 / 187.4 | 60.3 | 6.03 | 1183.4 / 1252.0 |
| `BAAI/bge-reranker-v2-m3` | 20 | 271.6 / 274.8 / 356.9 | 73.6 | 3.68 | 1183.4 / 1252.0 |
| `BAAI/bge-reranker-v2-m3` | 50 | 584.3 / 569.2 / 789.9 | 85.6 | 1.71 | 1183.3 / 1252.0 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 10 | 77.2 / 83.5 / 89.0 | 129.6 | 12.96 | 741.0 / 814.0 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 20 | 125.1 / 123.9 / 165.8 | 159.9 | 8.00 | 741.1 / 814.0 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 50 | 270.9 / 263.0 / 359.9 | 184.6 | 3.69 | 741.1 / 814.0 |

模型加载阶段（每个模型只加载一次）为：

| 模型 | Hugging Face revision | load time（s） | load peak allocated / reserved（MiB） | 实际 dtype |
| --- | --- | ---: | ---: | --- |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 9.720 | 1144.5 / 1160.0 | `float16` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 7.887 | 1091.2 / 1110.0 | `float16` |
| `Alibaba-NLP/gte-multilingual-reranker-base` | `8215cf04918ba6f7b6a62bb44238ce2953d8831c` | 9.786 | 614.0 / 650.0 | `float16` |

`load peak` 在每个模型加载前重置 CUDA peak counter，只覆盖模型构造和加载；
`inference peak` 在模型常驻后、每个候选池的 scoring 阶段单独测量，反映模型
和推理临时 buffer 的峰值。两者边界不同，不能直接相减。模型权重已在本机
缓存，load time 不包含网络下载时间。

## 实现与运行产物

实现包含：

- `knowledge_scope.retrieval.reranking` 中的模型规格、惰性本地 Cross-Encoder
  adapter、统一 score contract 和 `RerankingService`；
- `knowledge_scope.evaluation.reranker_benchmark` 中的冻结输入校验、dense
  候选池生成、三模型/三候选数 benchmark、失败/OOM 记录和运行 manifest；
- CLI `reranker-benchmark` 和 `rerank-search`；
- `Settings` 中的 `KNOWLEDGE_SCOPE_RERANKER_*` 配置，以及独立的
  `reranker-benchmark` uv dependency group。

运行示例：

```bash
uv sync --group reranker-benchmark
uv run knowledgescope reranker-benchmark \
  --split both \
  --models qwen3-reranker-0.6b bge-reranker-v2-m3 gte-multilingual-reranker-base \
  --candidate-sizes 10 20 50 \
  --batch-size 8 --max-seq-length 512 --device cuda --dtype float16
```

模型文件、ranking、embedding、完整运行结果和 manifest 都位于被 `.gitignore`
忽略的 `data/evaluation/a2-4/`，没有把模型、PDF、语料或 secrets 写入仓库。
本阶段没有前端检索页面，也没有加入 sparse/hybrid retrieval、GraphRAG、LLM
生成或其他 A2.5 系统基准能力。

## 真实本地 smoke test

使用已经存在的 A1.5 canonical artifacts，按 A1.6 默认策略生成 chunk，并通过
本地 Qdrant 和实际 `Qwen/Qwen3-Embedding-0.6B` 完成 3 个文档的索引；随后用
默认 `Qwen/Qwen3-Reranker-0.6B` 执行一次 dense + reranker 查询。没有重新运行
MinerU，也没有使用 mock 分数。

| document_id | indexed chunks | removed stale chunks |
| --- | ---: | ---: |
| `239e8ee8-4b77-5201-9d3d-625caba1c96c` | 17 | 0 |
| `8af9b576-e0a3-55c4-ba7e-861eba558337` | 19 | 0 |
| `8482bf4e-2862-56e1-bb2e-bb64a01b8d2d` | 19 | 0 |

总计写入 55 个 Qdrant points。查询
`说明事理时应重点说明哪些内容?` 的 Top-1 为：

- document：`8af9b576-e0a3-55c4-ba7e-861eba558337`；page：18；
- chunk：`chunk-17f3732bb9fb5317ab6c48b8916dbe24566a6fb82fcbb31b59bf18d6dd3a479b`；
- dense rank：1；dense score：`0.7120327`；reranker score：`8.078125`；
- section：`如何清晰地说明事理`。

## 建议与限制

- 模型层面的默认推荐为 `BAAI/bge-reranker-v2-m3`：它在 dev 上的
  `Hit@1/MRR` 领先 Qwen 和 GTE，且显存、延迟明显低于 Qwen；Qwen reranker
  不再作为 preferred option，仅保留为 Qwen 生态下的可运行对照。
- 严格 dev-first 的默认候选数是 Top-10。若产品把更多 recall 换成更高延迟，
  BGE + Top-20 可作为 provisional balanced profile；test Top-20 为
  `Hit@1=0.8333`、`MRR=0.8759`，但这些 test 数值只是最终确认，不能把 Top-20
  说成由 test 调出来的最终默认值。
- 延迟/显存优先时，保留 GTE Top-10 作为 low-latency profile：dev 平均约
  `80.2 ms`、`741.0 MiB` inference peak；代价是 dev `Hit@1/MRR` 低于 BGE。
  如果后续固定使用 Top-20，test 中 GTE 平均 `125.1 ms`、`741.1 MiB`，但
  `EvidenceRecall@10=0.8889` 低于 BGE 的 `0.9444`。
- test 只有 36 条 query，未计算置信区间、重复种子或并发服务吞吐；结果来自
  单机、单 GPU、单 batch 配置。
- `max_seq_length=512` 是本报告的 `512-token common profile`，长 chunk 及
  query+chunk pair 可能被截断；上面的审计不能证明答案 span 一定落在模型输入
  窗口内，也不代表 Qwen/GTE 等模型的原生长上下文被充分利用。
- reranker 只负责重新排序，不改变 Qdrant、dense embedding 或跨系统事务
  一致性；线上端到端默认候选数、超时、批处理和回退策略留待后续阶段。
