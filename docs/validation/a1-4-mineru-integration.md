# Phase A1.4 MinerU 集成验证

## 验证范围

本记录验证的是一次真实 MinerU 本地运行、KnowledgeScope adapter 和 canonical artifact 发布流程，不是全量语料 benchmark。共选取 9 个来自本机只读教材语料目录的分片，未修改、移动、重命名或加入 Git，也没有处理完整约 1.4 GB 语料。验证日期：2026-09-06。

## 运行时与单 PDF POC

- 主机：Ubuntu 24.04.4 LTS，Python `3.12.13`，uv `0.11.29`。
- GPU：NVIDIA GeForce RTX 3060 12GB，driver `595.84`，系统报告 CUDA `13.2`。
- Docker GPU 探针未通过：daemon 报告 `no known GPU vendor found`，因此没有把 MinerU 放进 Docker；实际使用仓库外独立虚拟环境。
- MinerU：`3.4.5`，固定 `pipeline` backend，安装包为独立运行时中的 `mineru[pipeline]`，模型和缓存均在仓库外。
- pipeline 日志报告 GPU memory `12 GB`、batch ratio `4`。外部 runtime 内 `torch.cuda.is_available=True`、`device_count=1`、设备名为 `NVIDIA GeForce RTX 3060`。一次真实 KnowledgeScope CLI 解析期间用 `nvidia-smi pmon` 观察到 MinerU 子进程以 compute 类型运行，采样到 SM 利用率最高 `60%`、显存 framebuffer 约 `2482 MiB`；这是采样到的进程显存，不宣称未采样时段的峰值。此前 9 份样本顺序执行的主机级采样最高约 `6664 MiB / 12288 MiB`，包含桌面及其他进程。

首次 POC 使用以下单个样本：

```text
最新【人教54制】9年级化学课本•全一册_39-57.pdf
```

文件大小 `2,196,997` bytes，MinerU CLI 运行约 `33.4` 秒并成功完成 19 页。输出包含 `*_content_list.json`、`*_content_list_v2.json`、`*_middle.json`、`*_model.json`、Markdown、layout/model/span/original PDF 和 59 个图片资产。adapter 使用稳定的 `*_content_list.json`，没有依赖 `content_list_v2`。

该 POC 的 content list 有 265 个输入项；适配后产生 220 个 canonical blocks，包括 33 个标题、128 个正文、12 个表格、8 个公式和 39 个图片，44 个页眉/页脚/页码/页脚注被 intentional skip，1 个空文本项形成 warning。实际文本和标题可读，公式项的 `text_format` 为 `latex`，表格项提供 HTML `table_body`，图片和表格/公式预览资产均实际存在。已抽查 layout 页面、文本/标题、一个公式资产、一个表格资产和一个图片资产；这里只记录结论，不复制教材正文。

此前因 bbox `[0, 699, 1000, 1001]` 失败的语文样本重跑后成功：19 页、307 个输入项、222 个 canonical blocks（15 个标题、198 个正文、1 个表格、0 个公式、8 个图片），80 个 intentional skip，5 个 unsupported，`bbox_clamped=1`、warning 6 个，耗时约 23.49 秒。item 226 的最后一个坐标被夹到 1000，canonical bbox 在 `[0,1]` 内；canonical JSON 通过 Pydantic 校验，8 个图片引用均指向 raw artifact 内的真实文件，失败 staging 清理规则也通过本次服务流程验证。

## 代表样本结果

成功项的 `canonical_valid` 均为 `true`，`elapsed` 是 MinerU runner 记录的秒数；`unsupported/warnings` 是输入处理统计，不是准确率。

| 科目 | basename | size (bytes) | 结果 | pages | input items | canonical blocks | title/text/table/formula/image | skipped | unsupported/warnings | elapsed (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| 语文 | `普通高中教科书·语文必修_下册_39-57.pdf` | 2,249,986 | success，item 226 `bbox_clamped=1` | 19 | 307 | 222 | 15/198/1/0/8 | 80 | 5/6 | 23.49 |
| 数学 | `普通高中教科书·数学（A版）必修_第二册_161-170.pdf` | 628,076 | success | 10 | 237 | 198 | 15/141/0/14/28 | 35 | 4/4 | 24.95 |
| 英语 | `普通高中教科书·英语必修_第二册_121-130.pdf` | 367,502 | success | 10 | 179 | 159 | 19/138/0/0/2 | 18 | 2/2 | 19.75 |
| 物理 | `普通高中教科书·物理选择性必修_第三册_11-20.pdf` | 9,125,086 | success | 10 | 143 | 121 | 28/70/0/0/23 | 21 | 1/1 | 20.87 |
| 化学 | `最新【人教54制】9年级化学课本•全一册_39-57.pdf` | 2,196,997 | success | 19 | 265 | 220 | 33/128/12/8/39 | 44 | 1/1 | 35.37 |
| 生物 | `普通高中教科书·生物学必修1_分子与细胞_111-120.pdf` | 2,395,599 | success | 10 | 185 | 150 | 37/95/1/1/16 | 32 | 3/3 | 29.20 |
| 历史 | `普通高中教科书·历史选择性必修1_国家制度与社会治理_39-57.pdf` | 2,719,243 | success | 19 | 323 | 253 | 71/145/0/0/37 | 67 | 3/3 | 22.56 |
| 地理 | `普通高中教科书·地理选择性必修3_资源、环境与国家安全_58-76.pdf` | 2,765,226 | success | 19 | 244 | 171 | 36/109/1/0/25 | 71 | 2/2 | 23.48 |
| 思想政治 | `普通高中教科书·思想政治必修3_政治与法治(1)_20-38.pdf` | 2,081,262 | success | 19 | 300 | 224 | 57/143/3/0/21 | 71 | 5/5 | 22.46 |

`title/text/table/formula/image` 的五个数字分别是对应 canonical block 数。9 个成功样本合计 135 页、2,183 个 MinerU 输入项、1,718 个 canonical blocks、18 个表格、23 个公式和 199 个图片；intentional skip 439 项，unsupported 26 项，warning 27 项，其中 `bbox_clamped=1`。

## 真实边界事件与策略

语文样本的 raw content list 中，item 226 是 `image`，`page_idx=13`，bbox 为 `[0, 699, 1000, 1001]`。最后一个坐标只超出 MinerU 约定的 `[0,1000]` 一个 source coordinate unit，adapter 将其夹到 1000，并产生 `bbox_clamped:item=226` warning；没有进行更大范围的几何修复。重跑后该文档的 canonical artifact 成功发布。

合成测试仍拒绝超出一单位容差的坐标、反向坐标和非法资产引用。其他样本中出现的空文本和 `code` 类型均通过 warning/statistic 显示；页眉、页脚、页码和页脚注按明确策略跳过。当前 canonical v1 没有 code block，因此不会把 code 伪装成普通正文。

## 结论与限制

本阶段已经证明外部 MinerU CLI、真实结构化输出、adapter、canonical 校验、实际 GPU 使用和原子 artifact 发布可以在本机工作；9/9 代表样本均成功。当前流程仍是开发者 CLI 单文档触发，不会在上传请求中自动解析，也没有前端解析按钮、异步任务队列、chunking、embedding、向量数据库、RAG、GraphRAG 或准确率/吞吐 benchmark。外部运行时需要按 [requirements/mineru.txt](../../requirements/mineru.txt) 安装，A1.5 再进行更大规模 corpus benchmark。

A1.4 hardening 另外使用临时开发文档完成了一次单文档回归：真实 CLI 成功解析 10 页并生成 198 个 canonical blocks，`manifest.json` 成功持久化 `parse_stats`（4 个 warning）；随后通过 Document DELETE 验证原始 PDF 和对应的 `data/parsing/<document_id>/` 均被清理。该回归未加入新的语料文件，也未改变上述 9 个样本结果。
