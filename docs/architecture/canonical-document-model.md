# CanonicalDocument 规范

## 目的

Phase A1.3 定义文档的解析器无关规范化数据模型。A1.4 使用外部 MinerU CLI 将真实 PDF 转换为该模型，并把结果保存为本地解析 artifact。该模型为后续 chunking、检索和评估提供稳定的输入形状，但这些下游能力仍不在当前阶段。

`knowledge_scope.documents.models.Document` 仍然是 A1.2 的上传元数据和存储记录。`CanonicalDocument` 不替代它，也不向 `documents` 表增加字段；上传文件名、SHA-256、知识库信息和 `storage_key` 不在规范化文档中重复保存。

## 层级与字段

数据层级固定为：

```text
CanonicalDocument
├── document_id
└── pages[]
    ├── page_number
    └── blocks[]
        ├── block_id
        ├── type
        ├── reading_order
        └── type-specific content
```

- `CanonicalDocument.schema_version` 当前固定为字符串 `"1.0"`。它表示本规范的版本，不表示解析器或依赖包版本。
- `document_id` 使用现有上传记录的 UUID。嵌套的 `Page` 和 block 不重复保存文档 ID。
- 至少存在一个 `Page`。`page_number` 从 1 开始，在文档中唯一、递增且连续。
- `block_id` 是应用控制的非空 opaque ID，在一个文档内全局唯一，不使用 MinerU 或其他解析器的 ID。
- `reading_order` 是页面内从 0 开始的连续整数，必须严格等于 `0, 1, ..., n-1`；空页面没有 reading-order 值。不同页面可以从 0 重新开始。

v1 只接受以下五种 block，`type` 是 Pydantic discriminated union 的判别字段：

| `type` | 内容 |
| --- | --- |
| `title` | 非空白 `text`，表示标题或小节标题 |
| `text` | 非空白 `text`，表示普通正文 |
| `table` | `markdown` 或 `html` 二选一且非空白，可选 `caption` |
| `formula` | 非空白 LaTeX `latex`；模型不执行公式 |
| `image` | 非空 `asset_ref`，可选 `caption` |

未知的 `type` 和未定义的额外字段都会被拒绝。表格保留为非空 Markdown 或原始 HTML，不在 adapter 中做脆弱的格式转换。图片只保留存储边界使用的 opaque `asset_ref`，不代表绝对文件路径，也不得包含 `.` 或 `..` 路径段（支持检查 POSIX 和 Windows 分隔符）；A1.4 会把它限制在对应 MinerU artifact 目录内的相对引用。

## 坐标约定

可选 `bbox` 使用 `x0`、`y0`、`x1`、`y1` 表示归一化页面坐标，所有值都在 `[0, 1]` 内，并满足 `x0 < x1`、`y0 < y1`。原点位于页面左上角，x 轴向右、y 轴向下。这是与页面图像和 UI 更容易互操作的规范表示；未来适配器负责把解析器的坐标系转换到这里。

## Lineage

后续 chunk 可以按以下链路回溯原文：

```text
CanonicalDocument.document_id
  -> Page.page_number
    -> Block.block_id
```

因此，未来的 Chunk 只需要保存这条来源引用及必要的局部文本范围，不需要复制整份上传元数据。Chunk 模型不在 A1.3 实现。

## 版本化、解析器独立性与后续适配

模型通过 Pydantic v2 提供 JSON 兼容的 `model_dump_json()` 和 `model_validate_json()` 往返。`schema_version` 独立于任何 parser/package 版本；规范变化时应显式增加兼容版本并更新校验规则。

A1.4 通过 subprocess 调用外部 MinerU `pipeline` 后端，生产代码不导入 MinerU 的 Python 内部模块。adapter 只读取稳定的 `*_content_list.json`，转换页面与 block 内容、归一化坐标、生成应用控制的 ID，并在输出前完成本规范校验；`middle/raw` 和其他 MinerU 文件作为独立调试 artifact 保存。规范化结果当前保存为 `data/parsing/<document_id>/canonical.json`，不增加数据库字段或迁移。解析器特有字段不应泄漏到 CanonicalDocument。

下面是一个合成数据示例，不代表生产解析结果：

```json
{
  "schema_version": "1.0",
  "document_id": "11111111-1111-1111-1111-111111111111",
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "block_id": "p1-b1",
          "reading_order": 0,
          "type": "title",
          "text": "设备维护规范"
        },
        {
          "block_id": "p1-b2",
          "reading_order": 1,
          "type": "text",
          "text": "请按照以下步骤进行检查。"
        },
        {
          "block_id": "p1-b3",
          "reading_order": 2,
          "type": "table",
          "markdown": "| 项目 | 状态 |\\n| --- | --- |\\n| 温度 | 正常 |"
        },
        {
          "block_id": "p1-b4",
          "reading_order": 3,
          "type": "formula",
          "latex": "E = mc^2"
        },
        {
          "block_id": "p1-b5",
          "reading_order": 4,
          "type": "image",
          "asset_ref": "asset-image-1"
        }
      ]
    }
  ]
}
```
