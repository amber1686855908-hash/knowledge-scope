# A3.4 独立图检索

Phase A3.4 在 A3.1 的 `KnowledgeEntity`、`KnowledgeRelation`、
`KnowledgeEvidence` 和 A3.3 的当前 canonical membership 之上，提供一个独立的、
证据优先的 Graph Retriever：

```text
query → 保守实体解析 → local seed → 有界 Neo4j traversal → KnowledgeEvidence
      → 排序后的 chunk/page/block lineage
```

本阶段不调用 LLM 做 query entity resolution，不修改 A3.3 的链接决策，也不把
结果接入 Qdrant、RAG 或前端。Neo4j adapter 只向 service 返回经过 Pydantic 校验的
typed records，不向业务层暴露 driver、session 或 raw record。

## Query entity resolution

请求必须带一个 `knowledge_base_id`。adapter 只读取该知识库内有来源支撑的
local entity 快照，并同时读取该 local entity 当前有效的 A3.3 canonical membership
名称和别名。应用层按以下顺序选择每个 local entity 的最佳信号：

1. 规范化后的 local name/alias 或 canonical name/alias 的 exact match；
2. 较长 label 在 query 中的 bounded contains match；
3. 至少 3 个字符的 bounded `SequenceMatcher` lexical match，默认阈值为 `0.78`。

名称使用 A3.3 的 `normalize_linking_label()`。长度不足两个字符的 label 不作为
seed；同一 label 命中超过 `max_seed_entities * 4` 个 local entity 时跳过，避免
“的”等短词或高频通用词造成 seed explosion。结果最多保留
`KNOWLEDGE_SCOPE_GRAPH_RETRIEVAL_MAX_SEED_ENTITIES` 个 seed，并返回方法、匹配词和
可解释分数。没有可靠 seed 时返回空结果，不猜测实体。

## Traversal 与边界

Neo4j 使用两个明确的单跳模式，不使用无界 variable-length Cypher：

- `local SOURCE_OF relation TARGET_OF local`：保留 forward/reverse direction；
- `local CANONICAL_MEMBER_OF canonical CANONICAL_MEMBER_OF local`：作为显式的
  `canonical_bridge`，允许连接不同文档的 local evidence，但不扁平化节点。

service 以 local entity 做有界 BFS，默认最多两跳。每次 adapter expansion 和整个
请求都受以下配置限制：

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `graph_retrieval_max_seed_entities` | 5 | 最大 seed 数 |
| `graph_retrieval_max_hops` | 2 | 最大 local-entity hop 数，只允许 1 或 2 |
| `graph_retrieval_max_neighbors` | 20 | 每个 frontier entity 的最大邻居数 |
| `graph_retrieval_max_relations` | 100 | 请求内最多使用的 distinct relation 数 |
| `graph_retrieval_max_evidence` | 50 | 最终最多返回的 evidence 数 |
| `graph_retrieval_max_entity_scan` | 10000 | 单次 seed resolution 的 local entity 扫描上限 |

关系查询同时检查 relation、source、target 的知识库和 local document scope；canonical
bridge 同时检查两端 local entity、canonical entity 及 membership 的知识库。所有
evidence 必须通过 `SUPPORTED_BY` 从实际 local entity 或 relation 获得。缺少支撑的
图事实不会被当作检索证据。

## Evidence-first result 与排序

`GraphEvidence` 保留 `evidence_id`、`knowledge_base_id`、`document_id`、`chunk_id`、
`page_start/page_end`、`source_block_ids` 和 `section_path`。每条
`GraphEvidenceResult` 只对应一个去重后的 evidence，并携带 seed、一个或多个
`GraphPath` 以及 retrieval reason，因此仍可回到：

```text
evidence → chunk → source blocks → page → document → original PDF
```

同一 evidence 通过多个 seed、关系或 canonical bridge 到达时按 `evidence_id` 合并，
保留可解释路径；不同 evidence 即使指向同一 chunk 也保留各自的 source block lineage。
排序分数是确定性的简单信号组合：
`0.72 * seed_resolution_score + 0.2 / (1 + hop_distance) + path_bonus`，其中
direct relation、canonical bridge、mixed path 使用不同的固定 bonus；相同输入和图状态
产生相同顺序。该分数不是训练得到的相关性概率，供后续阶段做 fusion 时使用。

## 一致性与边界

A3.4 是只读检索流程，不写 PostgreSQL、文件系统、Qdrant 或 Neo4j，因此不会声称
跨系统原子性。它依赖 A3.3 已经维护好的当前 membership；若图正在被另一个写入流程
重建，读取结果可能反映某个事务提交前后的不同时间点。adapter 遇到 scope 不一致、
非法 ID、非法 evidence 或 malformed graph state 会失败并报告受控错误，不会降级为
未经支撑的结果。只有当前仍由同一知识库、同一 local document 的
`SUPPORTED_BY` evidence 支撑的 local entity 才能成为 seed；canonical entity 只能把
查询解析到这些有支撑的 local member。遍历中的 relation 也必须有当前同 KB/document
的 `SUPPORTED_BY` evidence，不能由端点 entity 的 evidence 代替。

每次 expansion 先按稳定 entity ID 选择有限数量的邻居，再为每个已选邻居保留一条
边，剩余边按邻居轮询并受请求级 relation 上限限制。因此高阶邻居不能独占预算，且
查询不会执行无界 Cypher expansion；这不是对整个图邻域的事后截断。

## 开发者入口与小样本

先显式确认 Neo4j 连接并初始化 schema：

```bash
uv run knowledgescope neo4j check
uv run knowledgescope neo4j schema
```

按知识库执行单次图检索：

```bash
uv run knowledgescope graph-search "跨文档实体" \
  --knowledge-base-id 00000000-0000-4000-8000-000000000032
```

对已有 A3.2 accepted extraction 和已经建立的本地 sample graph 生成 bounded review
pack：

```bash
uv run knowledgescope graph-retrieval-sample
```

review 输出写入被忽略的 `data/evaluation/a3-4/`，只包含 query、seed、路径、短来源
excerpt 和 evidence lineage，不包含 PDF、绝对路径、完整 canonical/chunk corpus 或
Neo4j raw object。该样本只记录解析/解析到证据的观察，不报告 precision、recall 或
accuracy；A3.4 也不执行全语料图构建。
