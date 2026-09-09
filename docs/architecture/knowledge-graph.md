# A3.1 知识图谱 schema 与 Neo4j 基础设施

Phase A3.1 只建立知识图谱的规范化数据模型和 Neo4j 存储边界。它不执行
LLM 实体/关系抽取，不做实体链接或归一化，也不改变 A2 的 dense retrieval、
reranker、RAG 或 benchmark 结果。

## 规范化模型

`knowledge_scope.graph.models` 中的模型不依赖 Neo4j driver：

- `GraphEntity` 表示 `entity_id`、实体本地作用域
  (`knowledge_base_id` + `document_id`)、`canonical_name`、`entity_type`、可选
  aliases 和至少一条 `GraphProvenance`；
- `GraphRelation` 表示有方向的 `source_entity_id`、`target_entity_id`、关系本地
  作用域、`relation_type` 和至少一条来源 provenance；
- `GraphProvenance` 保存可回溯到原始 PDF 的稳定链路：必需的
  `document_id`、`knowledge_base_id`、`chunk_id`、`page_start/page_end`、
  `source_block_ids`、`section_path`，以及可选的证据级 `extraction_provenance`。

模型使用 Pydantic v2 且禁止额外字段。页码从 1 开始，`page_end` 不能早于
`page_start`，来源 block ID 必须非空且不能在同一条 provenance 中重复。实体或
关系至少有一条 provenance，且所有 provenance 必须属于同一知识库和该事实的
local `document_id`，因此不会在该边界创建无来源事实、跨知识库支撑或隐式跨文档
链接。

## ID 与幂等

当前 `GRAPH_ID_VERSION` 为 `v2`。所有 ID 都由同一个结构化序列化规则生成，
逻辑格式为：

```json
{"identity_schema_version":"v2","kind":"entity","payload":{}}
```

序列化使用固定 JSON separators、递归 key ordering、UTF-8 和 SHA-256；不使用
NUL 或其他字段拼接分隔符，因此字段中出现 NUL、换行或特殊 Unicode 时仍然无歧义。

`entity_id_for(name, type, knowledge_base_id=..., document_id=...)` 对名称和类型
做 NFKC、去首尾空白、折叠连续空白和大小写折叠，然后将以下字段纳入结构化
payload：identity schema version、`knowledge_base_id`、`document_id`、规范化名称
和类型，生成 `entity_v2_<64 hex>`。这是**抽取阶段的 provisional/local identity**：
同一知识库同一文档重复抽取同名同类型实体会幂等，不同文档或不同知识库不会因为
名称相同而自动合并。aliases 的变化不改变 ID。

这不是全局 canonical identity、实体归一化、同义词识别或实体链接。A3.1 不会
自动把两个 local entity 合并；未来 A3.3 应通过显式的 linking/merge 流程建立
canonical identity。为了避免在 A3.3 前通过复用 ID 形成隐式链接，`GraphEntity`
和 `GraphRelation` 的所有 provenance 必须同时属于该事实的知识库和 local
`document_id`；复用文档 A 的 ID 挂接文档 B 的 evidence 会被模型拒绝。
Neo4j 写入边界还会重新校验模型快照，防止调用方在构造模型后原地修改列表字段绕过
这些约束或写入无 provenance 的事实。

`relation_id_for(source, target, type, knowledge_base_id=..., document_id=...)` 将
有方向的 source、target、规范化 relation type 以及关系本地作用域纳入同一结构化
payload，生成 `relation_v2_<64 hex>`；交换 source/target 或作用域会得到不同 ID。
`evidence_id_for(provenance)` 对完整来源和证据级 extraction provenance 生成
`evidence_v2_<64 hex>`，同一支撑上下文可被多个实体或关系复用。

因此，同一文档重复 upsert 使用相同的实体、关系和 evidence ID。Neo4j 端使用
`MERGE`、唯一约束和单次事务，不会因重复处理而 silently duplicate；这不等于
分布式 exactly-once。

## Neo4j 图结构

适配器使用以下最小图结构：

```text
(:KnowledgeEntity)-[:SOURCE_OF]->(:KnowledgeRelation)-[:TARGET_OF]->(:KnowledgeEntity)
(:KnowledgeEntity)-[:SUPPORTED_BY]->(:KnowledgeEvidence)
(:KnowledgeRelation)-[:SUPPORTED_BY]->(:KnowledgeEvidence)
```

关系的 provenance evidence 也会连接到 source/target entity，使关系证据仍然
能支撑其端点的来源链路。`KnowledgeEvidence` 节点保存 `document_id`、
`knowledge_base_id`、页码、`chunk_id`、`source_block_ids`、`section_path` 和
证据级 extraction provenance；不会保存 `storage_key`、绝对文件路径或 PDF 内容。
沿着 `document_id` 可回到 PostgreSQL `Document`，再由应用内部的文件存储定位原始
PDF。

`uv run knowledgescope neo4j schema` 创建或确认：

- `KnowledgeEntity.entity_id` 唯一约束；
- `KnowledgeRelation.relation_id` 唯一约束；
- `KnowledgeEvidence.evidence_id` 唯一约束；
- entity type、relation type 和 evidence document ID 的查询索引。

schema version 当前为 `1.0`，保存在 readiness 和图节点属性中。readiness
执行安全的 `RETURN 1` 连接检查；它不会隐式创建 schema。

## 生命周期与一致性

- **upsert**：必须先 upsert 带 provenance 的实体，再 upsert relation；relation
  的 source/target 不存在或不属于关系的同一 local scope 时操作失败，不创建孤立
  关系。实体别名在 Neo4j 事务内与已有列表做排序后的集合并集；不会被后一次写入
  覆盖。extraction provenance 保存在 `KnowledgeEvidence` 上，随 evidence ID 和
  支撑边保存，不是 graph fact 上的单个可变字段。每次写入在 Neo4j managed
  transaction 中执行，`MERGE` + 唯一约束保证重复执行的幂等基础。
- **reprocess**：调用方应在同一文档的新图事实完成准备后，按需要先调用
  `delete_document(document_id)` 清理旧 evidence，再 upsert 新事实。仅重复
  upsert 不会删除已经不存在的旧事实。
- **delete**：适配器在一个 Neo4j transaction 中删除该 `document_id` 的
  evidence，再删除没有任何 evidence 支持的关系和实体。多个事实可以复用同一
  local document 内的 evidence；删除文档会一起移除这些支持及不再有支持的事实。
  跨文档共享事实不在 A3.1 的模型内，需等待 A3.3 的显式 linking/merge 设计。
- **Neo4j 失败**：managed transaction 失败时由 Neo4j 回滚，适配器抛出不含
  URI 密码或其他凭据的 `GraphStoreError`。应用不会把失败伪装成成功。

PostgreSQL、原始文件系统和 Neo4j 不构成分布式事务。A3.1 没有声称跨系统
atomic；如果未来文档状态变更和图写入分属不同系统，必须由上层编排补偿、重试
或 reconciliation。当前没有自动图抽取，因此文档上传/删除 API 尚未自动写入
或清理图事实。

## 开发者入口与边界

Neo4j 使用 `compose.yaml` 中的 `neo4j:5.26-community`，Bolt/HTTP 端口默认
只绑定到 `127.0.0.1`，数据使用 Docker named volume。应用配置通过
`Settings` 的 `KNOWLEDGE_SCOPE_NEO4J_*` 环境变量读取；密码使用 `SecretStr`，
不写入日志、readiness 响应或图属性。

```bash
uv run knowledgescope neo4j check
uv run knowledgescope neo4j schema
```

本阶段没有图查询 API、GraphRAG、混合检索、LLM 抽取、Neo4j 浏览器封装或前端
图可视化。普通 pytest 不需要运行 Neo4j；设置
`KNOWLEDGE_SCOPE_RUN_NEO4J_INTEGRATION=1` 并提供本地连接配置后，才运行可选的
真实 Neo4j 集成测试。
