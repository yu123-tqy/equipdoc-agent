# RAG 项目文档来源与优先级

本轮知识库扩展纳入两份项目文档的公开技术内容，并保留标题、章节编号、标题层级、段落关系、表格和图片引用。原始 Word 文件不进入仓库；仓库只保存面向检索的 Markdown、图片资产、精确切片和来源锚点。

## 来源优先级

1. **实验方案为主来源**：`source_authority=experiment_plan_primary`，`source_priority=100`。涉及研究对象、试验目标、系统组成、故障模拟、数据采集、特征工程和诊断方法的疑点，采用实验方案表述。
2. **合同为补充来源**：`source_authority=contract_supporting_spec`，`source_priority=70`。合同只补充设备型号、尺寸、数量、加工要求、供电、安全与验收等工程规格。
3. 如果两份来源存在冲突，检索排序只提供轻量优先级加权；最终答案仍须展示来源和适用边界，不应静默拼接互相矛盾的参数。

## 文档与切片

- 实验方案拆分为索引及六个章节文档：`pod_thrust_bearing_plan_index`、`pod_thrust_bearing_plan_ch1` 至 `pod_thrust_bearing_plan_ch6`。
- 合同公开技术内容整理为：`test_rig_technical_spec`。
- Word 精确切片共 206 条，逐条保存页码和 Word 块位置，正文不超过 500 个字符。
- 通用知识及仓库原有文档采用标题感知切片：目标 420 字符、上限 500 字符、重叠 80 字符；切片正文自动附加文档标题和标题路径。
- `data/knowledge_source_anchors.jsonl` 保存页码、块位置与 Markdown 章节的映射，用于审查追溯，不直接参与向量化。

## Chroma 元数据

每个 chunk 至少包含 `doc_id`、`chunk_id`、`title`、`section`、`source_path`、`source_type`、`source_authority` 和 `source_priority`。项目文档的精确切片还包含 `page_start`、`page_end`、`block_start`、`block_end` 等定位字段，可直接写入 Chroma collection metadata。

## 内容边界

- 联系方式、签章、账户、身份证号、单位内部地址等非技术信息未进入公开知识库。
- 图片只承担原文示意与人工复核作用；当前 Embedding 输入为切片中的文字、表格转写、公式转写和图片说明，不对图片像素直接生成向量。
- 原始实验方案中的方法描述代表项目设计方案，不等同于已经完成工业验证或取得既定准确率。
