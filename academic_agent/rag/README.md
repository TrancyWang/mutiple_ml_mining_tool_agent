# Academic Agentic RAG

这是 `doc-assistant/agentic-rag` 的目标项目内置版，不单独启动 FastAPI 服务。

## 工作方式

```text
用户上传文件 / 当前工作区文档
  -> DocumentParser
  -> Transformer（关键词、摘要、标题路径）
  -> Chunker（表格、公式、图片保持原子块）
  -> LocalHybridIndex（全文 + embedding）
  -> document_rag 工具
  -> Academic Agent 原有 Qwen-Agent 生成带引用回答
```

## 与现有项目的关系

- 使用目标项目现有的会话上传文件上下文和工作区权限边界。
- 优先复用目标项目已配置的 BGE embedding；不可用时回退到本地确定性哈希向量。
- 最终回答继续由目标项目现有 Qwen-Agent 生成，不重复创建第二个 LLM Agent。
- 不执行用户生成的 Python 或 SQL；数据分析走目标项目已有分析工具和本模块的文档统计能力。
- 索引默认写入 `runtime/rag_datasets`，也可通过 `RAG_DATA_DIR` 指定。

## Agent 工具

工具名：`document_rag`

参数：

- `query`：文档问题。
- `file_paths`：可选，限制到当前工作区或用户上传文件。
- `top_k`：可选，证据片段数量。

直接使用前，请在 Qt 界面上传文件，或把文档放进当前项目工作区。工具会自动按文件修改时间重新索引，未变化的文件会跳过。

## 目录结构

```text
academic_agent/rag/
├── application/       # RAG 用例编排与服务入口
├── ingestion/         # Parser、Transformer、Chunker、PDF 资产
├── retrieval/         # 混合检索索引
├── storage/           # 文档、索引和元数据文件存储
├── analysis/          # CSV/JSON 安全分析
├── providers/         # embedding 与 OpenAI-compatible 适配
├── common/            # 配置与公共数据契约
└── README.md
```
