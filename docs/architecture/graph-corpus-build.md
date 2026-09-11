# A3.6 图语料构建与覆盖审计

Phase A3.6 只解决把已经登记的 benchmark 语料送入现有 A3.2/A3.3 图流水线时的运行边界、恢复能力和覆盖可见性。它不改变 MinerU、A1.6 chunking、A2.1 标注、A3.2 taxonomy 或 A3.3 linking policy，也不实现新的图检索能力。

## 输入与权威快照

runner 使用 A1.5 的安全 manifest 元数据和 A1.6 的 `IndexedChunk` JSONL。它按文档流式读取 chunk，不加载完整 canonical corpus、PDF 或全文到内存；每个 `IndexedChunk` 在交给 `ExtractionService` 前转换为既有 `Chunk` 类型。CLI 会先用权威快照验证输入，再把本次选定文档的 materialized snapshot 传给 runner；runner 不接受只有数量/ID、没有每文档指纹的权威摘要作为执行快照。运行前通过 PostgreSQL `Document` 的 `knowledge_base_id` 和 `status=registered` 核对选定文档，避免把同一文件误写入其他知识库。

`docs/benchmarks/a3-6-corpus-input-snapshot.json` 是当前输入的仓库安全权威快照：它记录当前 manifest/chunk index 的 SHA-256、文档 ID 集、文档数、chunk 数和覆盖文档/chunk 身份的 `input_fingerprint`，不记录原文。该快照描述 corpus 身份而不是所有权，因此不绑定某个目标 KB；目标 `knowledge_base_id` 仍会写入 run manifest 和 pipeline fingerprint。构建和审计都会重新计算完整输入快照，文件截断、替换、重排、缺失或新增文档/chunk 会 fail closed。255 和 7,524 只是该快照当前记录的值，不是应用常量。

runner 的实际阶段是：

1. 按 subject 或 `--full` 选择文档，并检查文档、chunk 分组、ordinal、完整快照和登记状态。
2. 在有限的 `max_concurrency`（1–8）内调用 A3.2 `ExtractionService.extract_chunk`；只有 `accepted` 结果才通过既有 Neo4j `upsert_extraction` 写图。
3. 按有限的 `link_batch_documents`（1–64）读取这些文档当前的 local entities，交给 A3.3 `link_entities`；只有显式配置 linking gateway 时才会对 `UNCERTAIN` 候选调用 LLM。
4. 写入文档、chunk 和 linking-batch checkpoint，并更新不包含原文的运行 manifest。

失败的 chunk 会保留受控 error category、attempt/retry/token/latency 计数；provider/network retry 仍由 A2.6 gateway 管理。`--retry-failed` 只重试失败 checkpoint，不重做已经成功的 chunk。输入、配置或 pipeline fingerprint 变化，以及显式 `--reprocess`，会先对该文档调用 A3.1 的 KB+document scoped graph cleanup，再强制重建全部 chunk，避免只重建一部分而丢失其余支持证据。空抽取和 grounding rejection 是成功完成但不产生图事实的终态；schema rejection 属于未完成的失败终态。任一 required chunk 失败时，文档为 `extraction_partial=true` 且 `linking_blocked=true`，不会进入 linking。

## Checkpoint、锁与一致性

运行时目录默认是 `data/evaluation/a3-6/`，包含 append-only JSONL checkpoint、原子替换的 `run-manifest.json` 和 `run.lock`。checkpoint 只保存 UUID、chunk/entity/batch ID、输入/config/pipeline fingerprint、状态、计数、token、成本、延迟和受控错误类别，不保存课文原文、PDF、完整抽取 JSON 或 provider raw response。文件写入使用 flush、`fsync` 和原子 rename；最后一条记录覆盖同一逻辑 key。

每个 checkpoint 目录由 POSIX `flock` 保护，同一目录的第二个 writer 会立即失败。锁文件中的 PID/host/time 仅用于诊断，不能据此盲目接管；进程死亡后由操作系统释放锁，管理员应先确认旧进程/目录状态再重新运行。该锁只保护本地 runner workspace，Neo4j 仍依靠事务和确定性 upsert。

成功 checkpoint 的可复用条件是：KB、chunk 输入 fingerprint、chunk config fingerprint 和 pipeline fingerprint 都一致。`--reprocess` 会在每个已有文档上先执行 scoped graph cleanup，并强制重新抽取和重新 linking；如果新 checkpoint 目录发现已有该 KB/document 图状态而未指定 `--reprocess`，runner 会在任何删除前 fail closed。输入变化、损坏的完成 checkpoint 或已取消且已有图状态时也会清理后重建。`--retry-failed` 只按失败 chunk/link batch 的 checkpoint 重试；不会无限恢复 linking。

linking 会按当前文档完整的 local entity 集重算，并在重算前清理该 KB/document 的旧 membership、CURRENT_DECISION 和无支持 canonical state；当前 A3.1 scoped cleanup 也会删除与这些本地实体相关的旧 linking records，因此不把历史 decision 的长期保留作为 A3.6 保证。当前文档窗口若因旧实体读取失败留下失败批次记录，新的批次键会取代该记录，避免恢复成功后 manifest 仍报告陈旧失败；当前批次失败仍需显式 `--retry-failed` 或重处理。文档状态明确区分 `extraction_complete`、`extraction_partial`、`linking_complete` 和 `linking_blocked`。

### Provider 阻断与安全恢复

runner 只把 A2.6 实际返回的、`category=api`、`retryable=false` 且状态码属于非临时
4xx 的 provider 错误视为 fatal signature；`408`、`409`、`425`、`429` 被排除，
`5xx`、timeout、connection 和没有明确状态码的错误不会触发该熔断。相同 signature
连续观察到 3 次后，runner 原子保存 `provider_blocked=true`、
`status=provider_blocked` 和不含密钥的 HTTP 状态说明。该阈值是固定的运行时保护，不
改变 A3.2 抽取策略，也不改变 pipeline fingerprint；gateway 自己的 retry 配置保持不变。

触发后，已经开始的有限并发 batch 可以完成，但 runner 不再调度新的 chunk；尚未调度
的 chunk 不会伪造失败 checkpoint，已完成和已经失败的记录保持不变。provider 恢复后，
在同一目录用相同输入和 fingerprint 显式执行 `--retry-failed`，只重试失败及尚未调度的
工作；成功 checkpoint 不会再次调用。该流程是 at-least-once 和补偿式的，不宣称
exactly-once，也不把熔断误当作 provider 健康检查。

失败统计也区分逻辑块和 provider 尝试：唯一失败 chunk 数从 chunk ID 去重得到，失败
provider 尝试数则累加 append-only checkpoint 中的 `provider_calls`；时间桶统计必须与
后者相加一致。`manifest.counters.chunks_failed` 是运行轮次的累计观察值，不能直接当作
唯一失败 chunk 数。

### Prompt cache 结构审计

当前 full run 使用 `legacy-v1` 提示布局，保持首批 3,967 个成功 chunk 使用的精确
消息顺序：页码/章节和 corrective feedback 位于 `BEGIN CHUNK` 之前。对现有 7,524 条
A1.6 chunk 的离线比较（只计算 UTF-8 message content，不含 provider 协议封装）显示，
该旧布局的 system+user 可复用前缀约为 2,412 bytes。

此前测量的另一种缓存布局只是离线实验，当前 full run 没有启用：它把 chunk-specific
内容移到 marker 之后，system 前缀 1,071 bytes、user 前缀 1,888 bytes，合计 2,959
bytes，代表性变化后缀为 55–4,675 bytes（中位数 687 bytes）。这相对旧布局增加约
547 bytes 的结构性前缀，但不是 provider 已报告的命中率或成本节省。该优化留到未来；
如要采用，必须升级 `prompt_layout_version` 并使用新的 checkpoint 目录，不能在现有
full run 中静默混用。旧 manifest 缺少该字段时固定按 `legacy-v1` 解释，布局闸门会对
其他值 fail closed。若改变抽取 contract 或语义，仍必须升级 prompt version 并让
pipeline fingerprint 失配。当前消息没有注入 run ID、document/chunk ID、时间戳、计数器
或 nonce；API 恢复后才能读取真实 provider cache metrics。

runner 是 at-least-once 执行：Neo4j 单次抽取或 linking 写入在 Neo4j 内部是事务性的，确定性 ID/upsert 使重试幂等，但不等于 exactly-once。PostgreSQL 登记、文件输入、LLM usage 记录、Neo4j 与未来 Qdrant 之间不是分布式原子事务；checkpoint/manifest 是 runner 观察值，跨系统中断时通过 retry、reprocess、确定性 upsert 和审计补偿。manifest 会分别记录抽取初始调用、gateway provider retry、截断预算 retry、corrective retry、corpus retry，以及 linking 初始调用、provider retry 和失败尝试；provider 调用还可能在 checkpoint 写入失败窗口重复并产生额外成本。若 gateway 每次调用最多 `1 + llm_max_retries` 次 provider attempt，抽取的截断 retry 上限为 `len(graph_extraction_truncation_budgets) - 1`，corrective 上限为 `graph_extraction_max_parse_retries`，一次 `--retry-failed` 至多再跑一轮，则单 chunk 的最坏 provider attempt 上限为 `(1 + llm_max_retries) × (1 + 截断 retry 上限 + graph_extraction_max_parse_retries) × 2`；不使用 corpus retry 时去掉最后的 `× 2`。默认预算 `[1024,2048,4096]` 因而最多产生 2 次截断 retry；截断重试与 corrective retry 分开计数。linking 的每个 LLM adjudication 最多 `1 + llm_max_retries` 次 provider attempt；候选数受 `DEFAULT_MAX_CANDIDATES`（当前 200）约束，因此单个 linking batch 最多 `200 × (1 + llm_max_retries)` 次 provider attempt，显式一次 `--retry-failed` 可能再产生同等一轮。不会自动无限重试；反复执行命令仍属于新的显式操作。`linking` 失败批次只有 `--retry-failed` 或显式重处理才会再次运行。

## 覆盖审计

`graph-corpus-audit --input-snapshot <path>` 先用权威快照验证完整 corpus，再将三类事实分开报告：

- 文件侧：manifest 行数、ready 唯一文档、subject、chunk 总数、没有 chunk 的文档和没有 manifest 的 chunk；
- 注册侧：目标 PostgreSQL Knowledge Base 中实际 `registered` 文档数；
- 图侧：当前 Neo4j evidence、entity、relation、canonical entity、membership 和被支持 chunk 集合。

A2.1 coverage 仍以 canonical evidence 的 `(document_id, source_block_id)` 到 A1.6 chunk 的反向映射为权威；审计只报告这些 relevant chunks 中哪些已有图 evidence 支撑，并分别给出 query-level 的 any/complete/partial/no coverage。它不把图覆盖率解释为抽取质量、precision、recall 或 answer quality。

## CLI

```bash
uv run knowledgescope graph-corpus-estimate
uv run knowledgescope graph-corpus-audit --knowledge-base-id <knowledge-base-uuid>
uv run knowledgescope graph-corpus-build --knowledge-base-id <knowledge-base-uuid> --sample-per-subject 1 --persist
```

`graph-corpus-estimate` 默认从权威快照读取 target chunk 数；需要覆盖时显式传入 `--target-chunks`。`graph-corpus-build` 当前是开发者/批处理命令，不由 FastAPI 请求自动触发；需要 `KNOWLEDGE_SCOPE_LLM_API_KEY`。未配置 key 时，命令会在创建任何 provider 调用前失败。A3.6 不宣称已经完成 255 个文档的真实全量抽取；一次 full-run 尝试的终态、失败类别和覆盖边界见 [A3.6 运行记录](../benchmarks/a3-6-graph-corpus-build.md)。
