# A3.3 实体链接小样本记录

本文档描述 A3.3 的可重复 review sample 口径，不把未完成人工审核的决策当作准确率。
运行时 `review.jsonl`、`summary.json` 和任何 provider 调用结果均位于被 `.gitignore`
忽略的 `data/evaluation/a3-3/`，不提交 PDF、完整 canonical/chunk corpus 或 LLM raw output。

## 样本与流程

命令从现有 A3.2 的 `debug-9`、`holdout-18` 和 `fresh-18` accepted extraction 记录加载
application-generated `GraphEntity`。同一 local entity 若在多个 chunk 出现，只合并其已
校验的 aliases/provenance；来源 excerpt 仍限制为 A3.2 已保存的 bounded review excerpt。
随后按名称/别名 exact block 和受控两字符 prefix block 生成候选。高频 block 在扩展前跳过，
global candidate budget 在 materialize 前生效；该流程不重新运行 MinerU，也不自动处理全量
255 个文档。

默认命令：

```bash
uv run knowledgescope entity-linking-sample
```

默认 deterministic run 不访问 LLM，因此 compatible candidates 会保持 `UNCERTAIN`，只有
类型不兼容或没有共享信号的候选才会确定为 `NO_LINK`。需要 LLM 仲裁时必须显式追加
`--adjudicate`；只有显式追加 `--persist` 才写入本地 Neo4j。

## 记录内容与统计口径

每条 review 记录分开保存 local entity A/B 的名称、类型、aliases、document/chunk/page/
source-block/section lineage 和 bounded source excerpt，以及 `link_pair_id`、本次运行的
`decision_id`/`run_id`、应用计算的 candidate signals、`LINK`/`NO_LINK`/`UNCERTAIN`、method、
confidence、reason、provider/model/prompt version。LLM 不可提供内部 ID；canonical ID 和
membership 由应用根据 LINK decision 生成。

summary 另记录 candidate count、跳过的高频 block/pair、candidate budget 状态、确定性
决策、人工审核 fallback、provider failure fallback、LLM 调用/失败、三态计数、token、累计
provider latency、配置成本和是否持久化。`manual_review_fallbacks`、`provider_failure_fallbacks`
和 `deterministic_decisions` 分开计数，不把“无 gateway 的人工待审”伪装成确定性规则。

运行样本的结果随本地 A3.2 runtime records 和当前配置变化，不在此文件中固化未经本次
运行核验的数字；可用 `summary.json` 查看实际计数。该样本只用于人工检查，不提供
precision、recall 或 accuracy。

## 一致性边界

一个 linking plan 的 canonical、historical decision、current pair pointer 和 membership
在一个 Neo4j managed transaction 内写入。新的 `NO_LINK`/`UNCERTAIN` 会撤销只由旧 decision
支持的 membership，但保留仍有独立 LINK support 的 membership；历史 decision 的 provider、
model、prompt、reason 和 fingerprint 不被后一次运行覆盖。Neo4j 事务内 aliases 使用排序
集合并集。

文档删除的普通 API 路径会调用 scoped `delete_document(document_id,
knowledge_base_id=...)`，在同一个 Neo4j transaction 中清理 A3.3 linking state 和 A3.1 graph
evidence；PostgreSQL、文件系统、Qdrant、LLM 与 Neo4j 之间仍是补偿式、非分布式 atomic
边界。删除或重新计算失败需要重试或 reconciliation，不承诺 exactly-once。

没有人工 ground truth 时不报告 precision、recall、accuracy，也不把该小样本外推到全库。
A3.3 仍不做 cross-KB linking、full-corpus linking、Graph Retrieval 或 GraphRAG。
