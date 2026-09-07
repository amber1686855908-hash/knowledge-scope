# A2.2 本地 Embedding 模型选择基准

## 结论

本轮在 RTX 3060 12 GB 上完成了四个候选模型的同协议 dense retrieval
基准，没有发现 OOM 或未完成的模型。

建议将 `Qwen/Qwen3-Embedding-0.6B` 作为 KnowledgeScope 的默认工程基线，
将 `Qwen/Qwen3-Embedding-4B` 保留为质量优先的候选；两者不等价。4B 在本次
test 上的 `MRR`、`Hit@3/5/10` 和 `EvidenceRecall@3/5/10` 更好，但 corpus
编码吞吐约低 4.6 倍，峰值显存约高 5.4 倍，查询 p50 延迟约高 2.2 倍。0.6B
在 test 上 `Hit@1` 与 4B 相同，同时更适合本地开发和后续迭代。该建议是基于
36 条 test 的工程权衡，不代表最终生产模型定论。

## 冻结输入与运行环境

- A1.6 chunk 数：`7524`
- A2.1 最终数据集：`108` 条，其中 `dev=72`、`test=36`
- 本次只读取 A1.6/A2.1 产物，没有修改 A2.1 ground truth，也没有重新运行 MinerU
- chunk index SHA-256：`3b9bae28a7e864c858a957043ed9232ec11b1ad635dfd9a94edd5fa7192a8d40`
- dataset SHA-256：`ca5f25fd43fa78973de76e0531a1df8e2c0046c28eb5b6b033438a7d0c34c9ad`
- Python `3.12.3`
- PyTorch `2.14.0`，CUDA build `13.0`
- Transformers `4.57.6`，Sentence Transformers `5.7.0`，Accelerate `1.14.0`
- GPU：`NVIDIA GeForce RTX 3060`，总显存约 `11889 MiB`
- NVIDIA driver：`595.84`

模型 revision：

| 模型 | Hugging Face revision |
| --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `intfloat/multilingual-e5-large-instruct` | `274baa43b0e13e37fafa6428dbc7938e62e5c439` |

## 固定协议

四个模型均使用相同的 `batch_size=4`、`max_seq_length=512`、CUDA、fp16、L2
归一化、全量 7524 chunks 和内存中的精确点积检索。每个模型分别重新编码
chunk corpus；排名取前 10，再计算既有 A2.1 的 `Hit@1/3/5/10`、`MRR` 和
`EvidenceRecall@1/3/5/10`。dev 完成后未根据 test 调参。

query 侧遵循各模型的官方约定：

- Qwen 使用 Sentence Transformers 的 `prompt_name="query"`：
  [`Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、
  [`Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)。
- BGE-M3 使用原始 query，不额外添加 instruction，参考
  [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)。
- E5 使用官方的 `Instruct: ...\nQuery: ...` 结构，任务描述为
  `Given a textbook question, retrieve passages that contain its answer`，
  文档侧不添加 instruction，参考
  [`multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)。

### 512-token common profile 与截断审计

本次结果应准确描述为 **512-token common profile**：代码在模型加载后统一设置
`model.max_seq_length=512`，所有模型都使用相同的输入预算，但 tokenizer 的
分词结果不同。以下统计使用各模型实际 tokenizer、包含 special tokens、关闭
tokenizer 侧截断后计算长度；超过 512 的 chunk 在实际编码时会被截断。

| 模型 | tokenizer | 被截断 chunks | 占 7524 | 有截断 relevant chunk 的 eval items | dev / test |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-0.6B` | `Qwen2TokenizerFast` | 915 | 12.161% | 39 | 30 / 9 |
| `Qwen/Qwen3-Embedding-4B` | `Qwen2TokenizerFast` | 915 | 12.161% | 39 | 30 / 9 |
| `BAAI/bge-m3` | `XLMRobertaTokenizerFast` | 978 | 12.998% | 41 | 31 / 10 |
| `intfloat/multilingual-e5-large-instruct` | `XLMRobertaTokenizerFast` | 978 | 12.998% | 41 | 31 / 10 |

A2.1 的 108 条样本共有 101 个唯一 relevant chunk。其中 Qwen 有 35 个
（34.653%），BGE/E5 有 37 个（36.634%）超过 512。chunk index 保存了
source-block lineage，但没有保存 source block 在 chunk 文本中的 token offset；
因此对这些 gold-relevant chunks，可以确认答案证据存在被截断的可能，不能在
当前安全运行产物上断言每个答案 span 一定落在窗口外。test 中已有 9–10/36
条样本受到该风险影响。

这不会破坏同一 512-token 预算下的相对比较，但会实质限制绝对分数的解释，
并可能影响长 chunk 上的模型排序。因此当前推荐应视为 common-profile 下的
工程建议，而不是对无截断长上下文表现的结论；后续若要确定生产模型，应做
单独的长度敏感性复测。本阶段不改变既定 `max_length=512`，也不把截断结果
伪装成完整证据覆盖。

## 检索质量

数值为各 split 的平均值，保留四位小数；完整逐条排名和未四舍五入结果只
写入被忽略的运行目录 `data/evaluation/a2-2/`。

| 模型 | split | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | ER@1 | ER@3 | ER@5 | ER@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-0.6B` | dev | 0.6389 | 0.8333 | 0.9028 | 0.9722 | 0.7427 | 0.6389 | 0.8241 | 0.8935 | 0.9722 |
| `Qwen/Qwen3-Embedding-4B` | dev | 0.6528 | 0.8472 | 0.9167 | 0.9444 | 0.7630 | 0.6528 | 0.8472 | 0.9167 | 0.9444 |
| `BAAI/bge-m3` | dev | 0.6250 | 0.7778 | 0.8333 | 0.8611 | 0.7154 | 0.6250 | 0.7778 | 0.8287 | 0.8611 |
| `intfloat/multilingual-e5-large-instruct` | dev | 0.6111 | 0.7917 | 0.9167 | 0.9306 | 0.7223 | 0.6111 | 0.7917 | 0.9074 | 0.9306 |
| `Qwen/Qwen3-Embedding-0.6B` | test | 0.6389 | 0.8056 | 0.8333 | 0.8611 | 0.7313 | 0.6389 | 0.8056 | 0.8333 | 0.8611 |
| `Qwen/Qwen3-Embedding-4B` | test | 0.6389 | 0.8333 | 0.9167 | 0.9444 | 0.7586 | 0.6389 | 0.8333 | 0.9167 | 0.9444 |
| `BAAI/bge-m3` | test | 0.5278 | 0.7500 | 0.8056 | 0.8333 | 0.6502 | 0.5278 | 0.7500 | 0.8056 | 0.8333 |
| `intfloat/multilingual-e5-large-instruct` | test | 0.5556 | 0.7778 | 0.8333 | 0.8333 | 0.6745 | 0.5556 | 0.7778 | 0.8333 | 0.8333 |

### Test paired comparison

以 `Qwen/Qwen3-Embedding-0.6B` 为 baseline，按同一条 query 的逐条
`MRR`（首个 relevant chunk 的 reciprocal rank）比较；`win` 表示候选模型
MRR 更高，`tie` 表示相同，`loss` 表示更低。test 只有 36 条 query，以下
不是统计显著性检验，也不支持把模型宣称为等价。

| 模型（相对 0.6B） | win | tie | loss | 平均逐 query MRR 差 |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-4B` | 7 | 27 | 2 | +0.0274 |
| `BAAI/bge-m3` | 4 | 25 | 7 | -0.0810 |
| `intfloat/multilingual-e5-large-instruct` | 4 | 25 | 7 | -0.0567 |
| `Qwen/Qwen3-Embedding-0.6B`（自身） | 0 | 36 | 0 | 0.0000 |

## 资源与延迟

`corpus/s` 是 7524 个 chunk 的实际编码吞吐；查询吞吐包含 query 编码和
精确搜索；显存为 CUDA peak allocated/reserved。加载时间和 corpus 时间因
split 记录，模型权重已在本机缓存。

测量边界如下：`load_time` 从构造 `SentenceTransformer` 开始，到 CUDA
同步完成为止，不包含下载时间；`corpus_encoding_time` 不包含模型加载和
warmup；query latency 包含单条 query 编码、精确矩阵乘法和 top-k 搜索，
并在 CUDA 同步后计时；query throughput 使用同一范围；peak allocated/reserved
在加载前重置，覆盖模型、corpus embeddings 和 query 处理期间的 CUDA 峰值。

| 模型 | 维度 | split | load s | corpus s / chunks/s | query mean / p50 / p95 ms | query/s | peak alloc / reserved MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | dev | 9.982 | 211.292 / 35.610 | 25.299 / 23.568 / 31.204 | 39.530 | 1429.2 / 1606.0 |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | test | 10.023 | 215.517 / 34.911 | 25.351 / 22.508 / 27.266 | 39.449 | 1429.2 / 1606.0 |
| `Qwen/Qwen3-Embedding-4B` | 2560 | dev | 16.871 | 1020.449 / 7.373 | 54.546 / 53.514 / 56.988 | 18.334 | 8193.7 / 8588.0 |
| `Qwen/Qwen3-Embedding-4B` | 2560 | test | 14.979 | 987.900 / 7.616 | 49.143 / 48.516 / 53.016 | 20.349 | 8195.6 / 8598.0 |
| `BAAI/bge-m3` | 1024 | dev | 10.895 | 89.110 / 84.435 | 13.424 / 12.292 / 14.664 | 74.501 | 1165.9 / 1264.0 |
| `BAAI/bge-m3` | 1024 | test | 10.167 | 80.394 / 93.590 | 11.530 / 11.514 / 12.595 | 86.739 | 1165.9 / 1264.0 |
| `intfloat/multilingual-e5-large-instruct` | 1024 | dev | 10.267 | 90.248 / 83.370 | 13.474 / 12.923 / 17.028 | 74.224 | 1151.5 / 1244.0 |
| `intfloat/multilingual-e5-large-instruct` | 1024 | test | 12.040 | 80.673 / 93.265 | 11.392 / 11.309 / 12.441 | 87.787 | 1151.5 / 1244.0 |

所有模型实际使用 `float16`，本轮没有 OOM、加载失败或部分 corpus 完成。
Qwen 4B 是未量化的 fp16 推理，不是量化替代方案；在本机的 batch 和长度
配置下可以完整跑通，但其显存和耗时成本应在后续部署决策中显式保留。

## 运行与产物边界

基准入口为：

```bash
uv sync --group embedding-benchmark
uv run knowledgescope embedding-benchmark \
  --split dev \
  --models qwen3-embedding-0.6b qwen3-embedding-4b bge-m3 multilingual-e5-large-instruct \
  --batch-size 4 --max-seq-length 512 --device cuda --dtype float16
uv run knowledgescope embedding-benchmark \
  --split test \
  --models qwen3-embedding-0.6b qwen3-embedding-4b bge-m3 multilingual-e5-large-instruct \
  --batch-size 4 --max-seq-length 512 --device cuda --dtype float16
```

模型权重、向量、ranking、逐条运行结果和 manifest 均保存在被 `.gitignore`
忽略的 `data/evaluation/a2-2/`；仓库不包含 PDF、完整 chunk corpus 或模型文件。
报告中的结果来自本次真实运行，不应将忽略目录缺失解释为已提交的模型缓存。

## 限制与后续边界

- test 只有 36 条，尚未做置信区间、重复种子或更大人工评测。
- 这是单 GPU、单 batch 配置的离线 benchmark，不代表并发服务吞吐或 CPU 表现。
- 只评估 dense retrieval，没有加入 sparse、hybrid、reranker、Qdrant、LLM 或生成式 RAG。
- 没有针对 test 结果调参，也没有做量化；A2.2 不决定 A2.3 的索引或服务架构。
