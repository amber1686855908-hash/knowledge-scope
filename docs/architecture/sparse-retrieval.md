# A4.3 Sparse / Lexical Retrieval

## 范围

A4.3 在现有 A1.6 `Chunk` 之上增加一条独立的文本词法检索分支。它只读
A1.5 的 `CanonicalDocument` artifact，并复用默认的
`chunk_document()`；不会重新运行 MinerU，也不会修改
`knowledgescope_chunks_v1`、A4.2 representation collection、Neo4j 或冻结的
A2/A3 结果。

本阶段采用标准 BM25 作为可解释的 lexical baseline，不把它描述成语义稀疏
模型，也不实现 Dense/Sparse 融合、最终 reranking 或新的检索质量结论。

## 索引结构

`knowledge_scope.retrieval.sparse.SparseIndexStore` 使用 Python 标准库
`sqlite3` 保存一个被忽略的本地文件，默认路径为
`data/evaluation/a4-3/sparse.sqlite3`。表中保存：

- `sparse_chunks`：完整的 KB/document/chunk、页码、source block、section、
  content type、asset ref、文本和 A1.6 chunk fingerprint；
- `sparse_doc_stats`：每个 chunk 的词项长度；
- `sparse_postings`：`term → chunk → term_frequency` 倒排表；
- `sparse_generations` 与 `sparse_meta`：活动 generation、算法 contract 和
  corpus fingerprint。

每次查询都必须提供 `knowledge_base_id`，可选 `document_id` 在 SQL 候选边界
过滤。结果只从活动 generation 读取，因此不同知识库即使复用 document/chunk
字符串也不会互相返回。输入记录还会按 KB/document 校验 A1.6 的零起始连续 `ordinal`，避免错误的来源顺序进入索引。

## Tokenizer 与 BM25

tokenizer contract 为 `mixed-script-v2`：先做 Unicode NFKC 和 casefold；中文
连续字符保留 unigram 与相邻 bigram；Latin/数字与 CJK 在连续输入中也会分界，
例如 `CO2浓度` 至少产生 `co2`、`浓`、`度`、`浓度`，`2024年` 至少产生 `2024`
和 `年`。英文、数字、缩写和公式操作数保留为 alphanumeric token，明确的内部
`.`, `-`, `_` 会保留；其他标点是边界，不被静默当作有意义词项。这样可以查询
中文术语、年份、`bge-m3`、`x_i`、`NaCl` 和 `E=mc^2` 中的操作数，但它不是中文
分词器，单字噪声和 bigram 词表膨胀是已知限制。

BM25 使用配置化的 `k1=1.2`、`b=0.75`，每个查询在限定 KB/document 范围内
计算：

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(d, q) = Σ idf(t) * tf(t,d) * (k1 + 1)
               / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

分数只用于同一索引、同一查询内排序，不能和 dense 或 graph 分数直接相加。
得分相同时按 `document_id`、`chunk_id` 稳定排序。

## Contract 与生命周期

索引 contract 包含 schema/version、BM25 算法版本、tokenizer/normalization
规则（包括 mixed-script 的脚本边界和 CJK unigram/bigram）以及 `k1/b`；活动 generation 另外包含有序 chunk/lineage/text 的
SHA-256 corpus fingerprint、chunk 数和 document 数。已有索引若静态 contract
不匹配会 fail closed；当前 corpus 变化必须通过 build/rebuild 产生新的
generation，不能把旧 postings 当成新语料。

同一 index 使用一个跨进程 sidecar file lock；一个 writer 独占 build/rebuild、
replace/delete、active-generation switch 和 stale cleanup，第二个 writer 会明确
以 busy 错误失败。SQLite WAL 允许正常读取；每次完整 search 都在一个只读事务快照
内完成，避免 generation 切换期间把旧 active ID 与已清理的旧行混用。代码不宣称
并发 writer 或高吞吐保证。
全量 build 先在 SQLite 事务内写入并核验完整的 candidate generation，再一次性切换
`active_generation_id`；candidate 不在提交前可见，因此 `sparse_generations` 中只有
已提交、可判定为 complete 的 generation，清理只删除非 active 的 superseded generation。
失败会回滚，旧 generation 仍可查询。切换成功后旧 generation 的 metadata、chunks、
document stats 和 postings 一起做清理，清理失败不会隐藏已经提交的新 generation，
结果会明确报告 `cleanup_failed`。单文档 replace/delete 通过保留其他 KB/document 并生成
新的完整 generation 实现 scoped、可重复的生命周期。SQLite 事务只覆盖本地
derived index；它与 PostgreSQL、Qdrant、Neo4j、文件系统之间不是分布式原子
事务。

## 角色隔离与开发命令

Sparse index 是自有 SQLite 文件，不接受 Qdrant collection name 作为路径，并
拒绝 `knowledgescope_chunks_v1` 与 `knowledgescope_representations_v1` 的文件名。
代码不导入 Qdrant/Neo4j client，因此 A4.3 build、replace、delete 不能触碰
这些冻结/独立存储。默认的 build/audit/query 都需要显式 KB UUID：

```bash
uv run knowledgescope sparse-index build \
  --knowledge-base-id <knowledge-base-uuid> \
  --canonical-root data/benchmarks/a1-5/canonical \
  --corpus-manifest data/benchmarks/a1-5/corpus-manifest.jsonl

uv run knowledgescope sparse-index audit \
  --knowledge-base-id <knowledge-base-uuid>

uv run knowledgescope sparse-index query "温度 2024 H2O" \
  --knowledge-base-id <knowledge-base-uuid> --top-k 5
```

`KNOWLEDGE_SCOPE_SPARSE_INDEX_PATH`、`KNOWLEDGE_SCOPE_SPARSE_BM25_K1` 和
`KNOWLEDGE_SCOPE_SPARSE_BM25_B` 可以通过 `.env` 配置；`.env` 不应提交。

## 当前边界

当前分支只提供文本 BM25、索引生命周期、lineage-preserving 结果和一个小型
Dense-vs-Sparse 描述性诊断 helper。没有 BM25/向量/图融合、视觉 embedding、
OCR、LLM caption、前端检索页面或全量质量 benchmark；运行时 SQLite、排名和
诊断文件均位于被忽略的 `data/evaluation/a4-3/`。
