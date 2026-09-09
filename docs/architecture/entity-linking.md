# A3.3 实体链接与归一化基础

A3.3 在 A3.1 的 document-scoped local entity 之上增加显式、可撤销的
knowledge-base-scoped canonical entity。它只处理链接决策和 membership，不改写或删除
原有 `GraphEntity`、`GraphRelation` 或 `GraphProvenance`，也不执行跨知识库链接。

## 模型与身份

`knowledge_scope.linking.models` 提供 Pydantic v2、`extra="forbid"` 的模型：

- `CanonicalEntity`：`canonical_entity_id`、`knowledge_base_id`、规范名、类型和别名；
- `EntityLinkCandidate`：两个按 ID 排序的 local entity 与应用计算的候选信号；
- `EntityLinkDecision`：`LINK`、`NO_LINK` 或 `UNCERTAIN`，以及 method、confidence、reason、
  `created_at`、run identity、signals 和可选 provider/model/prompt version；
- `EntityCanonicalLink`：可删除、可重建的 local→canonical membership，并记录一个主决策
  及全部 supporting LINK decision ID。

A3.1 local identity 仍由 `knowledge_base_id`、`document_id`、规范化名称和类型决定。
`CanonicalEntity.canonical_entity_id` 则是创建时生成一次的 opaque
`canonical_entity_v1_<uuid>` 生命周期 ID：它不依赖 anchor local entity、canonical name、
aliases 或 membership。knowledge base 是显式且不可变的作用域；canonical name 和 aliases
是可更新的元数据，entity type 在持久化对象生命周期内保持不变。删除或重新处理 local
entity 不会重新生成仍在使用的 canonical ID；重建 plan 时必须显式传入已有 ID。

这是“local extracted identity → explicit linking decision → KB canonical identity”，不是
实体自动归一化。不同文档或知识库中同名同类型的 local entity 不会因为名称相同而合并；
A3.3 也不允许调用方通过复用一个 local ID 附加另一个文档的证据。当前阶段只允许在
已有 local entity 之间建立明确、可审计的链接，不实现更复杂的 entity
normalization/linking algorithm。

所有 linking ID 使用 `linking_identity_json`：固定的 `v2` identity schema、排序后的 JSON
key、固定 separators、UTF-8 和 SHA-256。字段使用结构化 JSON envelope，不使用 NUL 或其他
拼接分隔符，因此包含 NUL 或组合字符的合法值不会因分隔符歧义而碰撞。`link_pair_id` 是
同一 KB 内无序 local pair 的稳定 ID；`decision_id` 在 pair 之外加入 `run_id`，同一 pair
的不同运行可以保留历史决策。`canonical_link_id` 对 KB、local 和 opaque canonical ID
稳定生成。

## 候选与决策

候选生成使用名称/别名 exact block，另用规范化名称前两个字符的受控 prefix block 捕获
明显的 lexical 近邻；prefix block 每个实体只比较确定的相邻窗口，不做全量两两扫描。高频
block 在扩展前按 `max_block_size` 跳过，global `max_candidates` 在 materialize 前生效，
并在运行统计中记录跳过的 block/pair 与预算耗尽状态。

决策分层如下：

1. 类型不兼容或没有共享名称/别名信号时确定 `NO_LINK`；
2. 类型兼容且存在名称、别名或 lexical 信号时只保留为 `UNCERTAIN`，不自动 `LINK`；
3. 默认交给人工审核；显式使用 `--adjudicate` 时才通过 A2.6 `LLMGateway` 仲裁。

因此同名、共享别名或相似名称不会被确定性规则破坏性合并。LLM 只看到两个实体、最多
12 个 bounded aliases 和 bounded 来源片段，只能返回三态 decision、confidence 和 reason。
LLM 不接收也不提供任何内部 entity/canonical/decision ID；应用只使用自身的 pair 和
membership 生成最终 ID。unknown/extra JSON field、重复 JSON key、格式错误或 provider
失败会保留为 `UNCERTAIN`，不会产生 membership。Gateway 负责 provider/network retry；
linking 层不重复重试。调用和重试可能产生额外 provider cost，不承诺 exactly-once。

## Neo4j 表示与一致性

`Neo4jGraphStore` 使用以下最小 linking 结构：

```text
(:KnowledgeEntity)-[:HAS_LINK_DECISION]->(:KnowledgeLinkDecision)
(:KnowledgeLinkPair)-[:CURRENT_DECISION]->(:KnowledgeLinkDecision)
(:KnowledgeEntity)-[:CANONICAL_MEMBER_OF {
  link_id, primary_decision_id, decision_ids
}]->(:CanonicalEntity)
(:KnowledgeLinkDecision)-[:DECIDES_CANONICAL]->(:CanonicalEntity)
```

`CanonicalEntity`、`KnowledgeLinkDecision` 和 `KnowledgeLinkPair` 有唯一 ID 约束。pair 保存
当前 effective decision 指针；每次新的 run 使用新的 `decision_id`，历史 decision 节点的
provider/model/prompt/reason/created_at/fingerprint 不做 last-write-wins 覆盖。写入新 decision 前，
旧 decision 的 current support 会从 membership 的 `decision_ids` 中移除；如果 membership
还有其他独立 LINK support，则保留，否则移除 membership 和不再被引用的 support edge。
因此新的 `NO_LINK` 或 `UNCERTAIN` 不会留下只由旧 LINK 支撑的 stale membership。

一个 `LINK` decision 必须至少支撑一个 membership，membership 的所有 supporting decision
都必须是同一 KB 内、包含该 local entity 的 `LINK`。canonical entity 必须有至少两个 local
membership，且 local/canonical/type/decision scope 由模型和 Neo4j query 双重检查；不存在
evidence-backed membership 的 LINK 不会持久化为孤立决策。local entity、canonical entity
和 relation 的 A3.1 source evidence 保持原状，A3.3 不把 provenance 复制成一个可覆盖的
shared fact 字段。

别名在 Neo4j 事务内按排序后的集合并集更新，重复 upsert 不会替换已有别名；Neo4j 对同一
canonical 节点的写锁使并发 union 不采用应用侧 last-write-wins。该行为是事务级并发安全，
不是跨进程 exactly-once 保证。

一个 linking plan 的 canonical、decision、membership 写入使用一个 Neo4j managed
transaction；重复提交相同历史 decision fingerprint 是幂等的，复用同一 decision ID 但改动
payload 会失败。事务失败会回滚该 plan。它不是 PostgreSQL、文件系统、LLM 与 Neo4j 的
分布式事务，也不代表 exactly-once。

为兼容 A3.1 已存在的 `KnowledgeEntity`，`ensure_schema()`（以及
`uv run knowledgescope neo4j schema`）会在 schema 初始化后扫描缺少
`entity_type_normalized` 的节点，并使用应用当前的 `normalize_linking_label()` 回填。回填前
会先校验所有待处理的 `entity_type`；非字符串、空白或无法得到规范标签的值会明确失败，不会猜测。
回填只对仍然缺少该属性的节点执行，已有值不会被覆盖，重复运行是安全幂等的。完成后 A3.3
查询只使用规范化字段，不保留永久的双字段兼容分支。A3.1 当前没有封闭的实体类型枚举，
因此“有效”表示原始类型是非空字符串且能通过该规范化逻辑；未来若收紧 taxonomy，应另行提供
显式迁移规则。

## 删除与重新处理

`delete_document(document_id, knowledge_base_id=...)` 会在同一个 Neo4j transaction 中先清理
该文档 local entity 的 A3.3 current/membership/decision/pair，再清理 A3.1 evidence 和孤立
local graph facts。普通 API 文档删除路径会调用这一 scoped 方法；`delete_document_links()`
仍可单独执行 A3.3 linking cleanup，并有意保留 A3.1 local graph。

实体是 document-scoped，因此“两个文档共享同一个 local fact”不是自然状态。只有明确的
LINK decision 才能让多个 local entity 共同属于一个 canonical entity。删除一个文档会移除
其 local endpoint 相关的 decision 和 support；若另一个 local membership 仍有独立 support，
canonical entity 保留，否则没有 membership 的 canonical entity 被清理。重新计算 linking
decision 时，旧历史在不变更 ID 的前提下保留；新的 current 指针与 support 集合由补偿式
reconcile 更新。删除、重新计算和写入之间失败需要重试或 reconciliation。

## 开发者入口

```bash
uv run knowledgescope entity-linking-sample
uv run knowledgescope entity-linking-sample --adjudicate
uv run knowledgescope entity-linking-sample --persist
```

命令默认复用 A3.2 `debug-9`、`holdout-18` 和 `fresh-18` 的 accepted extraction，生成
被忽略的 bounded `data/evaluation/a3-3/review.jsonl` 和 `summary.json`。默认不调用网络；
`--adjudicate` 需要本地 `.env` 中的 `KNOWLEDGE_SCOPE_LLM_API_KEY`，`--persist` 才连接
Neo4j。review 记录只含短来源 excerpt、lineage、候选信号和决策审计字段，不含 PDF、绝对
路径、完整 corpus 或 provider raw response。该样本用于人工检查，不提供 precision、
recall 或 accuracy。
