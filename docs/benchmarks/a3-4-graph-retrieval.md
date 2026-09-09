# A3.4 图检索小样本记录

本页记录 A3.4 的可重复开发者 review sample 协议，不是检索质量基准。

## 范围

- 输入：已有 A3.2 accepted extraction runtime records；不重新运行 MinerU 或 LLM 抽取；
- 图：已经写入本地 Neo4j 的 A3.1/A3.3 sample graph；
- 解析：`GraphRetrievalService` 的 exact/alias/bounded lexical seed resolution；
- traversal：有方向 local relation、显式 canonical membership bridge，最多两跳；
- 输出：被忽略的 `data/evaluation/a3-4/review.jsonl` 与 `summary.json`；
- 统计：`query_count`、`resolved_query_count`、`no_seed_count`、
  `total_graph_results`、结果级的 `valid_lineage_results` /
  `invalid_lineage_results`、hop 分布和 wall-clock latency。

每条 review 记录的路径形状为：

```text
query → seed entity → relation/canonical path → KnowledgeEvidence lineage
```

每条返回的 evidence 都单独校验知识库、文档、chunk、source block lineage 以及
`GraphEvidence` 与 A3.1 `evidence_id_for()` 的一致性；因此 valid/invalid lineage 是
结果级计数，不是把有结果的 query 数当作 evidence 数。no-seed control 使用由样本
知识库和实体词表确定的 synthetic sentinel，并在运行时断言解析出的 seed 数为零。

Neo4j expansion 对每个 frontier entity 的邻居和 relation 都设置确定性上限：先保证
多个邻居各得到一次机会，再轮询额外 relation。只有 entity 和 relation 当前存在
同知识库、同文档的 `SUPPORTED_BY` evidence 时才会参与 seed 或 traversal；孤儿
relation 不会因为端点仍有 evidence 而被合成支持。

当前实现不产生 precision、recall 或 accuracy，也不把图结果接入 Qdrant、RAG 或
前端。运行环境没有可用的预先构建 sample graph 时，命令仍可生成控制类 no-seed
记录；需要实际 evidence 结果时应先运行 A3.2/A3.3 的显式本地 sample persistence。

## 安全边界

提交代码只包含 schema、查询逻辑、测试和本说明。实际 review pack、Neo4j 数据、
模型、PDF、完整教材摘录以及 runtime corpus 均保持在本地并被 `.gitignore` 忽略。
Neo4j 查询为参数化、有界读取；跨知识库请求不会读取或返回其他知识库的 seed 或
evidence。图读取是只读操作，不与 PostgreSQL、文件系统或 Qdrant 构成分布式原子
事务。

## 当前观察

真正的结果取决于本机 Neo4j 中是否存在 A3.2/A3.3 sample graph。运行命令后，以本地
`summary.json` 为准；该文件不应提交到仓库。A3.4 的确定性 ranking score 仅用于
排序解释，不代表概率或人工相关性判断。
