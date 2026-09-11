# KnowledgeScope

KnowledgeScope 是一个面向行业文档的 Python 3.12 项目，当前提供知识库管理和 PDF 文档上传的 Web 应用基础。

## 当前状态

当前已完成 A3.7，提供全量图语料构建、覆盖审计和只读混合检索评测基础，目前提供：

- 使用 `uv` 管理的 `src/knowledge_scope` package，以及通过 Settings 驱动的 health、parse-document 和 chunk-document CLI；
- 基于 FastAPI 的 `GET /api/v1/health` 和 `GET /api/v1/meta`；
- 基于 PostgreSQL、SQLAlchemy 2.x async 和 `asyncpg` 的知识库与文档元数据持久化；
- 使用 Alembic 管理 `knowledge_bases` 和 `documents` 表结构；
- 知识库的 `POST`、`GET`、`PATCH`、`DELETE /api/v1/knowledge-bases` CRUD API，支持分页、校验和 404 响应；
- 知识库文档的 PDF-only `POST`、`GET`、`DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents` API，支持分页、SHA-256 去重和受控错误响应；
- 将上传的原始 PDF 保存在 Settings 的 `data_dir` 下，使用固定的相对路径 `documents/<knowledge_base_id>/<document_id>/original.pdf`（默认文件系统路径为 `data/documents/<knowledge_base_id>/<document_id>/original.pdf`）；文件名只作为规范化元数据保存；
- 默认通过 `KNOWLEDGE_SCOPE_MAX_UPLOAD_SIZE_BYTES=52428800` 将单文件大小限制为 50 MiB，并统一保存为 `application/pdf`；同一知识库中的相同 SHA-256 文件会被拒绝，不同知识库可以分别上传；
- 使用 Vue 3、TypeScript、Vite、Vue Router、Element Plus、`@tanstack/vue-query` 和 Pinia 的 `frontend/` 应用；
- 知识库列表和详情页面，支持真实数据的加载、空状态、错误重试、新建、编辑、删除确认、分页，以及 PDF 文档上传、列表、删除和删除确认。
- 位于 `knowledge_scope.parsing.models` 的解析器无关 `CanonicalDocument` Pydantic v2 数据模型，包含页面、标题、正文、表格、公式和图片引用的规范化表示、结构校验与 JSON 往返序列化；`TableBlock` 支持非空 Markdown、HTML 或 opaque `asset_ref`，并可同时保留结构化表示与资产引用。
- 通过独立虚拟环境中的 MinerU `3.4.5` `pipeline` backend 解析单个已上传 PDF 的开发者 CLI：`uv run knowledgescope parse-document <document-id>`；MinerU 不作为 KnowledgeScope 的 Python 依赖，也不被 FastAPI 请求直接调用。
- 解析器会读取 `*_content_list.json`，将页面、文本、标题、表格、公式、图片和可读列表转换为 `CanonicalDocument`，对页码、reading order、bbox、block ID 和图片/表格 artifact 引用执行校验，并报告跳过项、unsupported 项、`bbox_clamped`、表格降级统计和 warning 数量。
- 规范化结果和 MinerU 原始输出分别保存在 `data/parsing/<document_id>/canonical.json`、`manifest.json` 和 `mineru/` 下；文件使用 staging 与原子提升，原始上传仍保存在 `data/documents/`。
- 位于 `knowledge_scope.chunking` 的解析器无关 `Chunk`/`ChunkedDocument` Pydantic 模型，保存确定性文本、页面范围、来源 `source_block_ids`、`section_path`、`content_types` 和 opaque `asset_refs`；分块不增加数据库表，asset-only chunk 可以没有文本但必须保留资产引用。
- 纯核心 API `chunk_document(document, config)` 按连续标题上下文、canonical block 和字符预算执行可重复的结构感知分块；默认 `target_chars=1200`、`max_chars=1600`、`min_chars=240`，字符预算不是 token 或检索质量承诺。
- 开发者 CLI `uv run knowledgescope chunk-document <document-id>` 读取已有 `data/parsing/<document_id>/canonical.json`，将 chunk artifact 写入 `data/chunking/<document_id>/chunks.json` 和 `manifest.json`；当前不会在上传请求中自动分块，也不需要 MinerU、GPU 或 embedding 模型。

Phase A1.5 已完成只读的全量解析基准：清单包含 257 个 PDF 条目、255 个唯一内容、1,580,533,246 字节和 3,833 个物理页，覆盖 9 个学科且没有未分类条目；2 个重复内容条目只解析代表文件，重复条目记录为 `skipped_duplicate`。首次运行中 244 个唯一内容成功、11 个因 MinerU 表格缺少 `table_body` 被严格 adapter 拒绝，成功率为 95.6863%；随后只重试这 11 个失败代表，最新结果为 255/255 个唯一内容均完成 `CanonicalDocument` 适配。100% 表示流水线完成，不是标注数据支持的解析准确率；其中 13 个表格条目因同时缺少结构化内容和可用资产而以 `table_missing_content` warning 降级。基准支持逐条 checkpoint、`--resume`、显式重试失败项和 `failures|all|none` 原始结果保留策略；运行产物位于被忽略的 `data/benchmarks/a1-5/`，汇总见 [A1.5 基准报告](docs/benchmarks/a1-5-corpus-parsing.md)。

Phase A1.6 使用这批已有 canonical 结果做了 CPU-only 结构分块校验：255/255 个文档成功生成 7,524 个默认 chunk，46,504 个 source block 和 6,176 个 asset block 的覆盖率均为 100%，未重新运行 MinerU；画像、策略比较、硬化前后对比和学科统计见 [A1.6 分块基准报告](docs/benchmarks/a1-6-semantic-chunking.md)。

Phase A2.1 已完成基于 canonical `document_id`、页码和 `source_block_ids` 的文本检索评估集：人工审核后保留 108 条 `verified`、54 条 `rejected`，覆盖 9 个学科且每科 12 条；最终 `retrieval-eval-v1` 划分为 72 条 dev 和 36 条 test，每科分别为 8/4。教材问题只保存在 `query_source` 审计字段，gold `evidence` 仅使用答案性正文、公式或文本表格，并通过 query 泄漏、evidence fingerprint、chunk lineage、重复 evidence/chunk 分组和 dev/test 分组校验；编辑 query 后会重新生成 deterministic item ID。运行时产物位于被忽略的 `data/evaluation/a2-1/`，仓库安全的最终标注见 [A2.1 retrieval-eval-v1](docs/benchmarks/a2-1-retrieval-eval-v1.jsonl)；详见 [A2.1 检索评估集报告](docs/benchmarks/a2-1-retrieval-eval-set.md)。A2.1 评测集本身不包含 embedding、真实检索分数或任何下游 RAG 生成能力。

Phase A2.2 已完成本地 dense embedding 模型选型基准：在冻结的 A2.1 dev/test 集上比较 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Embedding-4B`、`BAAI/bge-m3` 和 `intfloat/multilingual-e5-large-instruct`，记录 Hit@K、MRR、EvidenceRecall、编码吞吐、查询延迟和 CUDA 显存。当前建议将 `Qwen/Qwen3-Embedding-0.6B` 作为默认工程基线，将 4B 保留为质量优先候选；该结论受 `512-token common profile` 截断限制约束，详见 [A2.2 模型选型报告](docs/benchmarks/a2-2-embedding-model-selection.md)。

Phase A2.3 已完成 Qdrant 持久化 dense retrieval 基础：Qdrant 使用本地 Docker 服务和 `knowledgescope_chunks_v1` collection，向量维度为 1024、距离为 cosine；`Qwen/Qwen3-Embedding-0.6B` 使用 A2.2 固定 revision、`prompt_name=query`、L2 归一化和 512-token 配置。索引 payload 保留 `chunk_id`、`document_id`、可选的 `knowledge_base_id`、页码、`source_block_ids`、`section_path`、`content_types`、`asset_refs`、文本和 chunk/embedding 指纹，支持确定性 point ID、文档级安全重建、旧点清理、删除清理和失败时的补偿式回滚。PostgreSQL、文件系统与 Qdrant 之间不是原子事务；向量清理失败会明确报错，并可能需要后续人工修复。详见 [A2.3 Qdrant 集成报告](docs/benchmarks/a2-3-qdrant-integration.md)。

Phase A2.4 已完成本地 reranker 基础：在 Qwen dense Top-K 之后，可通过 CLI 调用本地 Cross-Encoder 对候选 chunk 重新排序；当前基准比较 `Qwen/Qwen3-Reranker-0.6B`、`BAAI/bge-reranker-v2-m3` 和 `Alibaba-NLP/gte-multilingual-reranker-base`，并记录不同候选池大小下的质量、延迟、吞吐和 CUDA 显存。当前建议将 `BAAI/bge-reranker-v2-m3` 作为质量优先的 reranker 候选，将 GTE 保留为低延迟候选；结论只基于 108 条冻结评测集和单机离线运行，不代表生产模型定论。详见 [A2.4 本地 reranker 基准报告](docs/benchmarks/a2-4-local-reranker.md)。

Phase A2.5 已完成冻结 A2.1 评测集上的检索阶段基准，比较了同一 `Qwen/Qwen3-Embedding-0.6B` 的 exact dense、Qdrant dense，以及 Qdrant Top-10 + `BAAI/bge-reranker-v2-m3`；结果、Qdrant 排名一致性、延迟和 test 坏例分析见 [A2.5 检索系统基准报告](docs/benchmarks/a2-5-retrieval-system-benchmark.md)。该基准不增加 sparse/hybrid retrieval、GraphRAG、LLM 生成或前端检索功能；Qdrant 当前 collection 规模下走 full scan，不能代表大规模 ANN 性能。

Phase A2.6 已完成 provider-independent 的 async LLM gateway 基础：`knowledge_scope.llm` 提供统一的 system/user 消息、模型与生成参数、普通 completion、streaming event 和规范化结果；首个真实适配器使用 OpenAI-compatible 的 DeepSeek 配置。每次已执行的逻辑调用都会尝试记录到 `llm_usage_records`，保存 provider、model、task type、token、延迟、成功状态、错误类别和可选的配置驱动成本估算；API key 不写入日志或记录。使用 `uv run knowledgescope llm-smoke-test` 可对已配置的 provider 执行一次开发者 smoke test；当前不包含 GraphRAG、Agent 或聊天页面。详见 [A2.6 LLM gateway 说明](docs/architecture/llm-gateway.md)。

Phase A2.7 已完成首个文本 RAG QA 编排：`POST /api/v1/rag/query` 复用 `Qwen/Qwen3-Embedding-0.6B`、Qdrant dense Top-10、显式 `BAAI/bge-reranker-v2-m3` 和 A2.6 LLM Gateway，通过 SSE 返回增量回答、应用生成的 citation metadata 及最终状态/usage。上下文按 reranker 顺序选择，抑制同一 lineage 下的精确重复文本，并受 `KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS` 字符预算限制；该预算只约束 chunk 文本字符数，不是 tokenizer-aware 的精确 LLM token context budget。无可用文本证据时受控返回证据不足，不调用 LLM；本地模型推理在共享 adapter 上串行化，当前不承诺 GPU 并发吞吐。请求 query 限制为 4,000 字符，当前接口适用于受信任的本地/内网开发环境，不应直接暴露公网。当前没有前端聊天页面、答案质量 benchmark 或后续 GraphRAG/Agent 能力；详见 [A2.7 RAG QA 说明](docs/architecture/rag-qa.md)。

Phase A3.1 已完成 provider-independent 的知识图谱 schema 与 Neo4j 基础设施：`knowledge_scope.graph` 提供带来源校验的 `GraphEntity`、`GraphRelation` 和 `GraphProvenance`。实体/关系使用包含知识库、文档本地作用域的结构化 `entity_v2_*`、`relation_v2_*` ID；这只是抽取阶段的 provisional/local identity，不会按名称自动执行跨文档或跨知识库实体链接。`KnowledgeEvidence` 节点保存 `document_id`、页码、`chunk_id` 和 `source_block_ids` lineage，extraction provenance 保存在证据上下文中。`Neo4jGraphStore` 支持连接检查、显式 schema/index 初始化、实体/关系幂等 upsert、别名集合并集、按文档清理和基础查找；Neo4j 使用本地 Docker 服务，数据保存在 named volume。PostgreSQL、文件系统与 Neo4j 之间不是分布式原子事务，Neo4j 单库写入失败由事务回滚，跨系统恢复仍需上层补偿。A3.1 本身不负责 LLM 实体/关系抽取、实体链接、GraphRAG、混合检索或前端图可视化；A3.2 的抽取能力见下文，基础设施说明见 [A3.1 知识图谱基础设施说明](docs/architecture/knowledge-graph.md)。

Phase A3.2 已加入单 chunk 的 grounded LLM 实体/关系抽取基础：`knowledge_scope.extraction` 使用 A2.6 gateway、严格的 `graph-extraction-v1.3` 应用 JSON contract、保守 taxonomy、chunk 文本与 relation evidence grounding、应用侧 A3.1 ID/provenance 生成和单次 Neo4j managed transaction 持久化。`uv run knowledgescope graph-extraction-sample` 可从已有 A1.5 canonical artifacts 重新生成少量 A1.6 chunks，按 subject 做默认小样本运行；`--sample-offset` 可用于建立不重叠的开发/holdout 样本。输出写入被忽略的 `data/evaluation/a3-2/`，不保存完整语料或 LLM 原始响应，也不报告未完成人工核验的准确率。A3.2 不做跨文档实体链接、全语料抽取、GraphRAG 或前端图可视化；详见 [A3.2 LLM 实体与关系抽取说明](docs/architecture/graph-extraction.md) 和 [A3.2 小样本记录](docs/benchmarks/a3-2-graph-extraction.md)。

Phase A3.3 当前提供显式的 local entity linking 基础：`knowledge_scope.linking` 从已有 A3.2 accepted extraction 生成同一知识库内的保守候选；确定性规则只拒绝明显不兼容或无共享信号的候选，其余候选保持 `UNCERTAIN`，可显式交给现有 LLM Gateway，最终只生成 `LINK`、`NO_LINK` 或 `UNCERTAIN` 决策。`CanonicalEntity` 使用创建后保持不变的 opaque lifecycle ID，与 A3.1 的 document-scoped `GraphEntity` 分开保存；local→canonical membership 可撤销，current decision 与历史决策分开记录，别名使用集合并集；不执行跨知识库链接、破坏性合并、全量链接或图检索。`uv run knowledgescope entity-linking-sample` 可生成被忽略的 bounded review pack；默认不调用 LLM，显式追加 `--adjudicate` 才使用已配置的 provider，追加 `--persist` 才写入本地 Neo4j。文档删除会通过 scoped graph cleanup 清理 A3.3 linking state；该样本只支持人工审核，不报告 precision、recall 或 accuracy；详见 [A3.3 实体链接说明](docs/architecture/entity-linking.md) 和 [A3.3 小样本记录](docs/benchmarks/a3-3-entity-linking.md)。

Phase A3.4 已加入独立的 bounded Graph Retriever：`knowledge_scope.graph.retrieval_service` 在指定 `knowledge_base_id` 内，使用 A3.3 当前 canonical membership 做保守的 local entity、alias 和规范化 lexical seed resolution，再通过 Neo4j 的有方向 local relation 和显式 canonical bridge 做最多两跳遍历。结果只返回有 `KnowledgeEvidence` 支撑的 `document_id`、页码、`chunk_id`、`source_block_ids` 和 `section_path` lineage，并按可解释的确定性信号排序；没有可靠 seed 时返回空结果。`uv run knowledgescope graph-search` 可执行单次开发者查询，`uv run knowledgescope graph-retrieval-sample` 可对已有本地 A3.2/A3.3 sample graph 生成被忽略的 review pack。A3.4 不接入 Qdrant、RAG 或前端，不调用 LLM 做查询解析，也不执行全语料图检索；边界和小样本口径见 [A3.4 图检索说明](docs/architecture/graph-retrieval.md) 和 [A3.4 小样本记录](docs/benchmarks/a3-4-graph-retrieval.md)。

Phase A3.5 已加入独立的 vector + graph hybrid retrieval：向量分支复用 `Qwen/Qwen3-Embedding-0.6B`、Qdrant dense candidates 和 `BAAI/bge-reranker-v2-m3`，图分支复用 A3.4 bounded graph retrieval；两条分支并发执行后，以 `(knowledge_base_id, document_id, chunk_id)` 去重并使用可配置的 RRF 融合。结果保留 vector/graph/both 来源、分支排名、图 seed/path、evidence IDs 和完整 chunk lineage；分支状态明确区分成功、空结果、失败和超时，支持 `degraded` 与 `strict` 模式。`uv run knowledgescope hybrid-search` 可执行单次开发者查询；当前不修改 RAG answer generation，不增加 sparse/hybrid score calibration、学习式融合或前端检索 UI。详见 [A3.5 hybrid retrieval 说明](docs/architecture/hybrid-retrieval.md)。

Phase A3.6 当前提供面向已登记 255 个 benchmark 文档和 7,524 个 A1.6 chunk 的可恢复图语料构建 runner，以及只读 corpus/graph/A2.1 覆盖审计。runner 复用 A3.2 的单 chunk grounded extraction 和 A3.3 的 bounded linking，不重新运行 MinerU，不改变 A1/A2 标注或 chunking；输入按文档流式读取，使用输入/config/pipeline fingerprint、追加式 fsynced checkpoint 和显式批次边界支持恢复、失败重试与文档重处理。只有 `accepted`、空抽取和 grounding rejection 才是成功终态，schema rejection 或其他未解决失败会阻断该文档的 linking。`graph-corpus-build` 需要显式 `--persist` 和已配置的 LLM key，默认只选每个学科一个文档；全量运行前可用 `graph-corpus-estimate` 查看基于已有样本的成本/耗时外推，`graph-corpus-audit` 用于核对登记、chunk、A2.1 和当前 Neo4j 覆盖。

最终审计的运行状态为 `partial_failure`：`run_id=e926e1d3-9050-4911-89ef-1632ed0894c2`，7,513/7,524 个 chunk 达到成功终态，11 个 chunk 终态失败；246 个文档为 link-complete/graph-eligible，9 个文档被排除并保持 linking-blocked，排除文档没有 retrieval-visible partial graph state。合格图谱包含 55,098 个 local entities、27,268 个 relations、6,087 个 `KnowledgeEvidence` records/supported chunks、1,241 个 canonical entities 和 2,609 个 canonical memberships。A2.1 的冻结 108 条评测映射中，105 条为 complete graph coverage、0 条 partial、3 条 no graph coverage。这里明确区分 extraction success（7,513/7,524）、graph evidence coverage（6,087 chunks）和 graph-eligible documents（246/255）；这些运行计数不代表抽取准确率或检索效果，也不表示 100% 成功。该运行使用 `legacy-v1` prompt contract 和 `1024 → 2048 → 4096` truncation policy，运行时 checkpoint、manifest 和审计文件均位于被忽略的 `data/evaluation/a3-6/`；cache-v2 只作为未来优化，不属于本次运行。设计说明见 [A3.6 图语料构建](docs/architecture/graph-corpus-build.md)，审计口径见 [A3.6 覆盖审计报告](docs/benchmarks/a3-6-graph-corpus-build.md)。

Phase A3.7 提供只读的 Vector-only 与 Vector + Graph 对照评测：复用现有 Qwen embedding、Qdrant、BGE reranker、A3.4 Graph Retriever 和 A3.5 RRF，在冻结的 A2.1 108 条评测项上记录分支排名、来源、lineage、指标和延迟；不调用答案生成 LLM，也不修改冻结标签、Qdrant 或 Neo4j。固定 profile 使用 Vector candidate/rerank `10/10`、Graph/Hybrid result limit `20/20`、`rrf_k=60` 和现有 A2.5 模型 revision；本次 test 的 H@10/ER@10 为 `0.8611 → 0.8889`，但 H@1/3/5 与 MRR 下降，dev H@10 下降、all-108 H@10 不变，因此只部分支持当前 test Top-10 的局部召回改善，不构成普遍质量结论。完整口径与结果见 [A3.7 混合检索评测](docs/benchmarks/a3-7-hybrid-evaluation.md)。

当前 A1.5/A1.6 的 255 个 benchmark 文档可以通过显式的 corpus registration workflow 登记为一个指定 KnowledgeBase。该流程不从文件名、路径、学科或正文推断归属，不重新运行 MinerU、分块或 embedding，也不复制/移动源语料；登记后的 `Document` 使用 `status=registered` 和 `storage_kind=external_reference`，`storage_key` 保持为空，原有普通上传文档的本地托管语义不变。登记前会校验 255 个唯一文档、canonical artifact、7,524 个 chunk lineage、目标 KB 和所有权冲突，并在一个 PostgreSQL 事务中完成；重复执行是幂等的，冲突会整体拒绝。登记还会生成被忽略的 `data/evaluation/a3-5/a2-1-kb-mapping.jsonl`，将冻结 A2.1 的 108 个 item 映射到同一显式 KB，但不会修改冻结标注；完整的存储与升级说明见 [benchmark corpus registration](docs/architecture/benchmark-corpus-registration.md)。

登记已有 benchmark 语料时，必须显式提供目标 KB UUID：

```bash
uv run knowledgescope corpus register --knowledge-base-id <knowledge-base-uuid>
```

登记完成后，先审计再修复已有 Qdrant reference collection 的 KB payload：

```bash
uv run knowledgescope qdrant audit-kb
uv run knowledgescope qdrant audit-kb --apply
```

`--apply` 只更新 Qdrant point payload 中的 `knowledge_base_id`；不会重算或替换向量、point ID、chunk ID、fingerprint 或 lineage。Qdrant 批量 payload 更新与 PostgreSQL 登记不是分布式原子事务，但在完整映射预检后可安全重复执行；中途失败时命令会明确报错，应重新运行审计。完成显式登记和 payload 修复后，才可将该 collection 用于 KB-scoped hybrid retrieval。原先由 `index_canonical_corpus` 建立的无 KB reference collection 在修复前不能用于该查询，也不能猜测或伪造 KB。

当前 PDF 不会在上传请求中自动解析；需要使用开发者 CLI 显式触发。当前本地文件布局用于开发和参考环境，不等同于生产对象存储方案。解析集成说明详见 [MinerU 本地集成](docs/integrations/mineru.md)，模型约定详见 [CanonicalDocument 规范](docs/architecture/canonical-document-model.md)，分块约定详见 [CanonicalDocument → Chunk 规范](docs/architecture/canonical-document-chunking.md)。

## 本地开发

### 安装依赖

```bash
uv sync
```

前端依赖安装：

```bash
cd frontend
npm install
```

### 启动 PostgreSQL

```bash
docker compose up -d postgres qdrant neo4j
```

Compose 默认将 PostgreSQL 映射到 `127.0.0.1:5433`、Qdrant 映射到 `127.0.0.1:6333`、Neo4j Bolt 映射到 `127.0.0.1:7687`、HTTP 映射到 `127.0.0.1:7474`，本地开发凭据、数据库名、Qdrant collection 和 Neo4j 连接配置定义在 [compose.yaml](compose.yaml) 与 [.env.example](.env.example) 中。复制 [.env.example](.env.example) 为 `.env` 后，可通过 `KNOWLEDGE_SCOPE_POSTGRES_PORT`、`KNOWLEDGE_SCOPE_QDRANT_URL` 和 `KNOWLEDGE_SCOPE_NEO4J_URI` 修改连接配置；Compose 在未提供 Neo4j 密码时只使用本地开发默认值，任何共享环境都必须显式配置凭据。不要在 `.env` 中提交 secrets。Qdrant 和 Neo4j 数据保存在 Docker named volume，不会写入 Git。外部 benchmark registration 仅保存受控的逻辑 `source_ref`，不会把原始 PDF 伪装成本地 `original.pdf`；这类只读引用文档不会直接进入普通 MinerU 上传解析流程。

### 执行数据库迁移

```bash
uv run alembic upgrade head
```

Alembic 是数据库 schema 的唯一来源；应用启动不会调用 `create_all()`。回退最近一次迁移：

```bash
uv run alembic downgrade -1
```

### 启动后端和前端

后端：

```bash
uv run uvicorn knowledge_scope.api.app:app --reload
```

前端（在另一个终端执行）：

```bash
cd frontend
npm run dev
```

Vite 会将 `/api` 请求代理到 `http://127.0.0.1:8000`，前端默认通过 `VITE_API_BASE_URL=/api` 访问 API。

Vite 默认使用 native file watcher。若开发机受到 inotify 或 file watcher limit 限制，可在 `frontend/.env` 中将 `VITE_USE_POLLING=true`；后端 `uvicorn --reload` 可在命令前加 `WATCHFILES_FORCE_POLLING=true`。这些 polling 选项仅用于开发环境。

MinerU 运行时位于仓库之外。解析 CLI 使用 `KNOWLEDGE_SCOPE_MINERU_COMMAND` 指定外部 `mineru` 可执行文件，并使用 `KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS` 设置超时；安装和模型配置见 [MinerU 本地集成](docs/integrations/mineru.md)。

对已经存在的 `CanonicalDocument` artifact，可显式运行结构分块：

```bash
uv run knowledgescope chunk-document <document-id>
```

该命令不会重新运行 MinerU，也不会在上传时自动执行。

A1.5 基准命令必须显式接收只读语料根目录，不会写入语料目录：

```bash
uv run knowledgescope benchmark-parsing --corpus /path/to/read-only-corpus
```

可使用 `--resume`、`--retry-failed`、`--raw-retention failures|all|none`、`--subject` 和 `--limit` 控制可恢复运行与 smoke test；默认运行工作区为被忽略的 `data/benchmarks/a1-5/`。

Qwen embedding 运行需要本地安装 embedding benchmark 依赖：

```bash
uv sync --group embedding-benchmark
```

默认 embedding device 为 CUDA；没有可用 GPU 时，可在 `.env` 中设置
`KNOWLEDGE_SCOPE_EMBEDDING_DEVICE=cpu` 和 `KNOWLEDGE_SCOPE_EMBEDDING_DTYPE=float32`。

检查或创建 Qdrant collection：

```bash
uv run knowledgescope qdrant check
uv run knowledgescope qdrant create
```

对已经存在的 `data/chunking/<document_id>/chunks.json` 执行单文档索引：

```bash
uv run knowledgescope qdrant index-document <document-id>
```

也可以直接使用 A1.5 的 canonical artifacts，按 A1.6 默认策略生成内存中的 chunk 并索引若干文档；该命令不会重新运行 MinerU：

```bash
uv run knowledgescope qdrant index-corpus --canonical-root data/benchmarks/a1-5/canonical --limit 3
```

执行 dense Top-K 查询：

```bash
uv run knowledgescope qdrant search "说明事理时应重点说明哪些内容?" --limit 5
```

安装本地 reranker 基准所需依赖并运行完整比较：

```bash
uv sync --group reranker-benchmark
uv run knowledgescope reranker-benchmark \
  --split both \
  --models qwen3-reranker-0.6b bge-reranker-v2-m3 gte-multilingual-reranker-base \
  --candidate-sizes 10 20 50
```

对已经建立的 Qdrant dense 索引执行 dense + reranker 查询：

```bash
uv run knowledgescope rerank-search "说明事理时应重点说明哪些内容?" --candidate-limit 20 --limit 5
```

运行 A2.5 检索阶段基准（默认使用冻结的 dev/test 评测集）：

```bash
uv run knowledgescope retrieval-system-benchmark --split both
```

对已配置 API key 的 DeepSeek/OpenAI-compatible provider 执行一次 LLM smoke test：

```bash
uv run knowledgescope llm-smoke-test --task-type evaluation
```

该命令会将调用用量写入 PostgreSQL 的 `llm_usage_records`；未配置 `KNOWLEDGE_SCOPE_LLM_API_KEY` 时会受控失败。测试套件使用 mock/fake，不需要网络、API key 或付费模型。

运行 A3.2 的少量分主题图抽取样本（不会重新运行 MinerU，也不会自动处理全部文档）：

```bash
uv run knowledgescope graph-extraction-sample
```

若需要把已验证的本次样本写入本地 Neo4j，显式追加 `--persist`；默认只生成被忽略的运行时 review 文件。抽取 taxonomy、grounding 和一致性边界见 [A3.2 LLM 实体与关系抽取说明](docs/architecture/graph-extraction.md)。

运行 A3.3 的少量实体链接 review sample（复用已有 A3.2 runtime extraction，不重新调用 MinerU）：

```bash
uv run knowledgescope entity-linking-sample
```

默认使用 `data/evaluation/a3-2/debug-9/sample.jsonl`、`data/evaluation/a3-2/holdout-18/sample.jsonl` 和 `data/evaluation/a3-2/fresh-18/sample.jsonl`，只输出 bounded 的 `data/evaluation/a3-3/review.jsonl` 与 `summary.json`。需要 LLM 仲裁时显式使用 `--adjudicate`；需要把本次 linking plan 写入 Neo4j 时再显式使用 `--persist`。A3.3 linking 操作不会修改 A3.1 local entity、relation 或 evidence。

运行 A2.7 RAG endpoint 还需要安装已有的本地 embedding/reranker 依赖：

```bash
uv sync --group embedding-benchmark --group reranker-benchmark
```

后端提供 `GET /api/v1/health/qdrant` readiness 检查、`POST /api/v1/retrieval/search` dense 检索接口和 `POST /api/v1/rag/query` SSE RAG QA 接口；当前没有前端检索或聊天页面。Qdrant、模型、向量和 benchmark 运行产物均不提交到仓库。

执行独立的 vector + graph hybrid 查询（不会改变 RAG 回答接口）：

```bash
uv run knowledgescope hybrid-search "说明事理时应重点说明哪些内容?" --knowledge-base-id <knowledge-base-uuid>
```

默认使用 `KNOWLEDGE_SCOPE_HYBRID_FAILURE_MODE=degraded`：一条分支失败时
返回另一条分支并在结果中标记失败状态；需要两条分支都成功时设置为
`strict`。相关 candidate、rerank、graph 和最终结果上限以及
`KNOWLEDGE_SCOPE_HYBRID_RRF_K` 均可通过 `.env` 配置。该命令不接入
`/api/v1/rag/query`，也不执行全语料图抽取。

运行 A3.7 的冻结检索对照评测（只读，不调用答案生成 LLM）：

```bash
uv run knowledgescope hybrid-evaluation \
  --knowledge-base-id <knowledge-base-uuid> \
  --split both
```

该命令在同一知识库和同一 Qdrant 语料上比较现有 Vector 分支与 Vector + Graph 分支，
使用冻结的 108 条 A2.1 评测项（dev 72 / test 36），并先 fail-closed 重现 A2.5
`qdrant_dense_bge_top10` baseline。逐查询结果和汇总写入被忽略的
`data/evaluation/a3-7/`；不会修改 A2.1 标签、Qdrant point 或 Neo4j 图。当前 Qdrant
规模低于 `full_scan_threshold=10000`，结果是当前规模 vector-store validation，不是
ANN 性能 benchmark；评测不包含答案生成，Hybrid 仅在固定 test Top-10 上出现局部改善，
不应表述为普遍检索质量提升。此前不一致的 exploratory 运行不用于比较。

审计已登记语料和当前图覆盖：

```bash
uv run knowledgescope graph-corpus-audit \
  --knowledge-base-id <knowledge-base-uuid>
```

根据已有 A3.2 样本估算全量运行的 token、成本和耗时（未配置价格时成本为 `null`）：

```bash
uv run knowledgescope graph-corpus-estimate
```

先完成估算并确认 LLM key 后，再显式执行一个小的分主题批次；`--persist` 才会写入 Neo4j：

```bash
uv run knowledgescope graph-corpus-build \
  --knowledge-base-id <knowledge-base-uuid> \
  --sample-per-subject 1 \
  --persist
```

该 runner 以 `docs/benchmarks/a3-6-corpus-input-snapshot.json` 作为当前 corpus/chunk universe 的权威快照；它会校验 manifest、chunk index 的 hash、文档/chunk 身份和完整性，不把 255/7,524 当作通用代码常量。同一 checkpoint 目录由本地 file lock 独占；发现新目录已有目标 KB/document 图状态时，必须显式使用 `--reprocess`，不会自动删除。checkpoint 只记录标识、fingerprint、状态、token、延迟和错误类别，不保存原始 chunk、PDF 或 provider response。Neo4j 单库事务与 PostgreSQL、文件系统、Qdrant 之间不是分布式原子事务；恢复是 at-least-once、幂等/补偿式流程，不宣称 exactly-once。

修复已有 Qdrant points 的知识库归属前，先执行只读审计：

```bash
uv run knowledgescope qdrant audit-kb
```

只有所有 `document_id` 都能在 PostgreSQL `Document` 记录中唯一映射到非空
`knowledge_base_id` 时，才可以显式追加 `--apply`。该操作只更新 Qdrant
payload 的 `knowledge_base_id`，不重算向量、不改变 point/chunk ID；批量写入
不是事务，若中途失败可安全重复执行。映射不完整时命令拒绝写入，不能从
文件名、路径或 benchmark 结构猜测知识库。

检查 Neo4j 连接并显式初始化图 schema：

```bash
uv run knowledgescope neo4j check
uv run knowledgescope neo4j schema
```

上述命令读取 `KNOWLEDGE_SCOPE_NEO4J_URI`、`KNOWLEDGE_SCOPE_NEO4J_USERNAME`、`KNOWLEDGE_SCOPE_NEO4J_PASSWORD` 和 `KNOWLEDGE_SCOPE_NEO4J_DATABASE`；密码只从未跟踪的 `.env` 读取，不会出现在输出、图节点或关系属性中。Neo4j readiness 只代表连接可用，schema 初始化仍需显式执行。

### 上传文件说明

当前上传接口只接受真实 PDF：服务端会检查文件名、`.pdf` 扩展名、文件头和大小，并以流式方式计算 SHA-256 后写入本地文件。上传文件不会被加入 Git；`.env`、`data/`、`node_modules/` 和前端构建产物也不会被提交。

## 验证

后端测试会使用 PostgreSQL 创建独立的 UUID 命名临时数据库，执行迁移，并在测试会话结束后只清理该临时库。未设置 `KNOWLEDGE_SCOPE_TEST_DATABASE_URL` 时，测试仅允许从 `KNOWLEDGE_SCOPE_DATABASE_URL` 的本机地址（`localhost`、`127.0.0.1` 或 `::1`）创建临时库；若应用地址是远程主机，必须显式配置 `KNOWLEDGE_SCOPE_TEST_DATABASE_URL`。Compose 默认用户具备所需权限。文件测试使用独立的临时 `data_dir`。

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

前端：

```bash
cd frontend
npm run lint
npm run type-check
npm run build
```

## 后续方向

后续阶段将继续扩展文档 ingestion 和 parsing 覆盖范围，并在当前 dense retrieval 与 reranker 基础上评估 sparse/hybrid retrieval、GraphRAG、multimodal retrieval、ChatBI 和 NL2SQL；当前版本不包含这些后续能力。
