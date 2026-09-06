# MinerU 本地集成

## 范围

Phase A1.4 只提供开发者触发的单文档解析流程。KnowledgeScope 不把 MinerU 加入主 Python 环境，也不导入 MinerU 的内部 Python 模块；解析通过外部命令行进程完成。

```text
KnowledgeScope CLI
  -> PostgreSQL 中的 Document 与受控原始 PDF
  -> 外部 MinerU CLI（独立虚拟环境）
  -> *_content_list.json adapter
  -> CanonicalDocument 校验
  -> data/parsing/<document_id>/ 原子发布
```

当前验证的运行时为 MinerU `3.4.5`、`pipeline` backend，使用主机上的 NVIDIA GPU。独立运行目录位于仓库之外，例如 `~/.local/share/knowledgescope/mineru-3.4.5`；模型和缓存位于仓库之外，例如 `~/.cache/knowledgescope/mineru`。这些目录不会被 Git 跟踪。

## 安装独立运行时

下面的 `MINERU_RUNTIME_DIR` 和 `MINERU_CACHE_DIR` 必须指向仓库之外的目录。MinerU 依赖应只安装到这个虚拟环境：

```bash
MINERU_RUNTIME_DIR=/path/outside/repository/knowledgescope/mineru-3.4.5
MINERU_CACHE_DIR=/path/outside/repository/knowledgescope/mineru-cache
uv venv "$MINERU_RUNTIME_DIR"
uv pip install --python "$MINERU_RUNTIME_DIR/bin/python" -r requirements/mineru.txt
```

准备 pipeline 模型后，将模型配置与缓存保存在同一个外部运行边界内。ModelScope 适用于当前开发机网络环境；实际解析使用本地已下载模型：

```bash
MINERU_TOOLS_CONFIG_JSON="$MINERU_RUNTIME_DIR/mineru.json" \
MODELSCOPE_CACHE="$MINERU_CACHE_DIR/modelscope" \
HF_HOME="$MINERU_CACHE_DIR/huggingface" \
MINERU_MODEL_SOURCE=local \
"$MINERU_RUNTIME_DIR/bin/mineru-models-download" -s modelscope -m pipeline
```

MinerU 官方文档：[快速开始](https://github.com/opendatalab/MinerU/blob/master/docs/en/quick_start/index.md)、[输出文件](https://github.com/opendatalab/MinerU/blob/master/docs/en/reference/output_files.md)。

将外部环境的 `mineru` 放入当前 shell 的 `PATH`，或在未提交的 `.env` 中设置：

```dotenv
KNOWLEDGE_SCOPE_MINERU_COMMAND=mineru
KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS=1800
```

当命令能在 `PATH` 中解析且同一独立环境根目录下存在 `mineru.json` 时，runner 会自动将它作为 `MINERU_TOOLS_CONFIG_JSON`；也可以在启动 CLI 前显式导出该变量。runner 固定使用 `MINERU_MODEL_SOURCE=local`，避免解析过程中隐式下载模型。

## 解析命令

先确保 PostgreSQL 已启动、Document 已通过现有上传流程保存，然后执行：

```bash
uv run knowledgescope parse-document <document-id>
```

命令会从 PostgreSQL 读取 Document，定位并校验原始 PDF 的 SHA-256，调用 MinerU，读取唯一的 `*_content_list.json`，生成并校验 `CanonicalDocument`，最后打印页数、block 数、表格、公式、图片、跳过项、unsupported 项和 warning 数量。长耗时解析只在 CLI 进程中执行，不在 FastAPI 请求中执行。

当前没有前端解析按钮，也没有新的解析 REST API。

## Artifact 布局

成功解析只会在所有输出完整且 canonical 校验通过后发布：

```text
data/
└── parsing/
    └── <document_id>/
        ├── canonical.json
        ├── manifest.json
        └── mineru/
            └── <mineru-output>/auto/
                ├── *_content_list.json
                ├── *_middle.json
                ├── *.md
                └── images/
```

MinerU 原始输出、Markdown、middle/model 文件、日志和图片资产保留在 `mineru/` 下，供调试和复现使用。`canonical.json` 和 `manifest.json` 先写入随机 staging 目录，再原子提升到文档目录；失败解析会清理 staging，不会留下看似成功的 canonical 文件。删除 Document 时，API 会先将原始 PDF 和存在的 `data/parsing/<document_id>/` 一起移入应用控制的临时 staging，再提交数据库删除；数据库失败会尽力恢复两者，提交成功后永久清理。解析目录是可选的，不存在时不影响删除；解析与删除并发协调不在本阶段范围内。原始上传仍位于 `data/documents/`，不会被覆盖。

manifest 只保存 `document_id`、源文件 SHA-256、parser、MinerU 版本、backend、canonical schema 版本、UTC 解析时间、仓库内相对 artifact 引用，以及 `parse_stats`。`parse_stats` 包含 elapsed、各类 block/skip/unsupported 数、`bbox_clamped`、`table_asset_only`、`table_missing_content`、warning 数和最多 100 条截断后的 adapter warning；不保存密钥、绝对路径或准确率。

## Adapter 规则

adapter 是唯一理解 MinerU 字段名的模块，主输入是稳定的 `*_content_list.json`；实验性的 `content_list_v2` 不属于必需契约。页面把 MinerU 的零基 `page_idx` 转成从 1 开始的 `page_number`，并按每页输入顺序重新分配 `0..n-1` 的 `reading_order` 和应用控制的 block ID。

| MinerU item | Canonical block | 处理方式 |
| --- | --- | --- |
| `text` 且 `text_level > 0` | `TitleBlock` | 保留标题文本 |
| `text` | `TextBlock` | 保留正文文本 |
| `list` | `TextBlock` | 仅在条目可读时按顺序合并 |
| `equation` | `FormulaBlock` | 原样保留 LaTeX 文本 |
| `table` | `TableBlock` | `table_body` 为 Markdown/HTML 时保存对应结构化表示，并在 `img_path` 有效时同时保留 `asset_ref`；缺少 `table_body` 但有有效图片时保存 asset-only 表格 |
| `image` | `ImageBlock` | 只保存 artifact 根目录内的相对 opaque `asset_ref` |
| `chart` 且有真实图片 | `ImageBlock` | 复用真实图片资产并保留图注 |

页眉、页脚、页码和页脚注是有统计的 intentional skip。`list` 在内容可读时合并为 `TextBlock`；`chart` 在有真实图片资产时映射为 `ImageBlock`；`aside_text`、`code`、未知类型和没有可保留内容的类型不会伪装成其他 block，会计入 `unsupported_items` 和 warning。表格缺少 `table_body` 时，adapter 会优先尝试有效的 `img_path`：成功则生成 asset-only `TableBlock` 并记录 `table_asset_only`；没有可用资产则将该条目计为 `unsupported_items`，记录 `table_missing_content` warning，并继续适配同一文档的其他内容。显式不安全的路径和结构性错误仍然失败。图片和表格的资产路径会同时检查 POSIX/Windows 分隔符、绝对路径、URI、`.`/`..` 段、artifact 边界和文件存在性。MinerU 的 bbox 按当前 pipeline 实际使用的 `[0,1000]` 页面坐标归一化到 canonical `[0,1]`；超出边界不超过一个 source coordinate unit 时才会夹到边界，并产生 `bbox_clamped` 统计和 `bbox_clamped:item=<index>` warning；超出该容差或夹取后几何仍无效时直接失败，不静默修复。

## 故障排查

- `MinerU executable was not found`：检查独立 venv 的 `bin` 是否在 `PATH`，或设置 `KNOWLEDGE_SCOPE_MINERU_COMMAND`。
- 模型找不到：检查 `MINERU_TOOLS_CONFIG_JSON`、`mineru.json` 中的 pipeline 模型目录和 `MINERU_MODEL_SOURCE=local`。
- `No module named 'six'`：`mineru==3.4.5` 的实际 OCR 路径会导入 `six`，而当前安装元数据未自动带入它；按 `requirements/mineru.txt` 在独立 venv 补装 `six`，不要把 MinerU 或其依赖加入 KnowledgeScope 的 `pyproject.toml`。
- 超时：只在未提交的 `.env` 增大 `KNOWLEDGE_SCOPE_MINERU_TIMEOUT_SECONDS`，并确认 staging 已被清理。
- 源文件 SHA-256 不一致：停止解析并检查 `data/documents/` 中的受控原始文件，不要覆盖它。

当前未实现 chunking、embedding、向量数据库、RAG、GraphRAG 或产品级评估。A1.5 的全量语料 benchmark 是独立的开发者评估 harness，结果与限制见 [A1.5 基准报告](../benchmarks/a1-5-corpus-parsing.md)。
