# A4.3 Sparse / Lexical Retrieval 基准记录

本文件记录 A4.3 的工程验证口径，不把词法检索称为语义稀疏检索，也不宣称
它已经改善了检索质量。所有完整倒排表、查询结果和逐条诊断文件都位于被忽略
的 `data/evaluation/a4-3/`；仓库只保留本报告和实现代码。

## 固定配置

- 输入：A1.5 `data/benchmarks/a1-5/canonical/` 与
  `corpus-manifest.jsonl`；每份 canonical 复用 A1.6 默认
  `chunk_document()`；不重新运行 MinerU；
- 算法：BM25，`SPARSE_ALGORITHM_VERSION=bm25-v1`，`k1=1.2`，`b=0.75`；
- tokenizer：`mixed-script-v2`，NFKC + casefold、CJK unigram/bigram、Latin/数字
  与 CJK 脚本边界切分、混合 alphanumeric token；
- 索引：独立 SQLite generation，默认
  `data/evaluation/a4-3/sparse.sqlite3`；每个 build 记录静态 contract 和
  corpus SHA-256 fingerprint；
- 作用域：每次 build/search 都使用显式 Knowledge Base UUID；默认实际完整
  语料使用已登记的目标 KB（运行命令时由操作者显式传入）。

## 全量构建结果

已在真实 A1.5/A1.6 输入上完成一次完整构建，并随后执行 audit。当前运行时
结果如下；SQLite 文件和逐条输出均在被忽略目录中，不属于仓库提交内容：

| 项目 | 结果 |
| --- | --- |
| canonical documents | 255 |
| indexed chunks | 7,524 |
| searchable chunks | 7,504（其余为 asset-only 或没有词项的 chunk） |
| knowledge bases | 1（显式 KB `3593a2ee-a326-5a0a-89a2-9a3012666c83`） |
| corpus fingerprint | `ccf6cc98aba94a09ecbee212db3c7ee8362fdc07ca98e69d8fb2ad16b1e33371` |
| static contract fingerprint | `c304c70fc09057830ff1b3c513835b7cab577304074aa984c8ff2b1f4f263ae9` |
| active index/generation fingerprint | `6774b78530a3aa8f8953bf10a285d8eec77191e9394d71f8203e2fd505391400` |
| audit missing/stale chunks | 0 / 0 |
| old-generation cleanup | 成功（`cleanup_failed=false`） |
| build wall-clock | 76.63 秒（本机 v2 rebuild 观察值） |
| SQLite 文件大小 | 约 1.176 GB（本地运行时观察值，约 1.10 GiB） |

`sparse-index audit` 同时确认 expected/indexed chunk 都是 7,524、文档都是
255，且 corpus fingerprint 匹配。当前 v2 generation 没有遗留旧 generation
行；BM25 postings 为 1,764,005 条。构建耗时属于本机运行观察值，不作为跨环境
性能承诺；SQLite 文件较大是已知的本地 baseline 成本。

此前 `mixed-script-v1` 的运行时索引仅作为兼容性审计历史保留在被忽略的
`sparse-mixed-script-v1.sqlite3`，其静态 contract fingerprint 为
`5aa6687a3064712b0096a61a4aa059afda88325074038ce74a349a6fab759ef1`，不能与
当前 v2 索引混用；当前默认 `sparse.sqlite3` 只代表 v2 contract。

本报告不把 `searchable_chunk_count` 当作 extraction success，也不把 BM25
命中当作准确率。asset-only chunk 会保留完整 lineage，但没有凭空生成的文本
词项，因而不会被纯文本查询强行返回。

## 词法 smoke 与诊断

受控 smoke 覆盖了真实语料中的中文术语、年份、英文术语和化学式样符号，并确认
结果包含 `knowledge_base_id`、document/page/chunk/source block/section lineage。
本次代表性观察为：`温度表` 返回真实的 document/page/chunk 命中，`2024` 返回
数字词项命中，`English` 返回英文词项命中，`NaCl` 返回化学式样词项命中。输出
只用于功能核验，不据此宣称质量提升；可重复执行的命令见架构文档。

另外完成了 6 条有界 Dense-vs-Sparse 描述性诊断（dev 3、test 3）：Sparse
触及已有 A2.1 gold chunk 的查询为 6/6，现有 A2.5 dense Top-10 为 5/6，
两分支同时命中为 5/6，Sparse-only 命中为 1/6，平均 Top-10 ID 重叠数为
3.5。该样本不是 A4.4 质量基准，未进行调参，也没有将 sparse 分数与 dense
分数合并。

`knowledge_scope.evaluation.sparse_diagnostic` 只做 bounded 的 Dense-vs-Sparse
观察：分别记录 Top-K chunk IDs、重叠数、两分支是否触及已有 A2.1 gold chunk，
不合并 raw score，不调参，也不替代后续 A4 benchmark。若未加载本地 dense
模型或 Qdrant 不可用，应如实记录为未执行，而不是补写数字。

## 生命周期与保护

同一输入重复 build 的 generation ID 和结果 identity 稳定；`mixed-script-v1` 与
当前 `mixed-script-v2` 的静态 contract 不兼容，旧 index 必须先重建，不能静默
复用。单 index 采用跨进程 single-writer/multi-reader 约束：writer 通过 sidecar
lock 独占 generation replacement、active switch 和 stale cleanup；第二个 writer
明确失败，WAL reader 可继续读取，单次 search 使用一致的只读事务快照。单文档 replace 会
清除该 KB/document 的旧 postings，删除操作可重复且不会删除其他 KB 的同名
chunk。新 generation 在完整写入并切换成功前不会成为 active；失败时旧索引仍
可查询。A4.3 代码不调用 Qdrant、Neo4j，也不写入 A2.3 dense collection 或
A4.2 representation collection。

## 限制

当前 tokenizer 是无外部词典的确定性字符 n-gram 方案，不等同于高质量中文
分词；SQLite BM25 适合本地 baseline 和当前规模，不是分布式搜索服务或大规模
ANN/倒排性能结论。没有在本阶段加入 BM25 与 dense/graph 的统一融合、视觉模型、
OCR、LLM 表述生成或 A4.4 评测。
