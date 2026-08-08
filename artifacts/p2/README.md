# P2 最终评测证据

本目录只保留 Full 模式和本轮 RAG 扩展的最终机器可读证据。模型权重、向量数据库、密钥、私有数据和服务器绝对路径不进入仓库。

## 文件索引

| 文件 | 内容 |
|---|---|
| [`service_check.json`](service_check.json) | 2026-07-29 RTX 4090、Qwen 服务与 `READY` 探针记录 |
| [`full_llm_eval.json`](full_llm_eval.json) | 20 题真实 Qwen 证据选择评测及完整回答 |
| [`full_llm_human_review.csv`](full_llm_human_review.csv) | 20 题人工复核模板，人工字段保持空白 |
| [`rag_project_retrieval.json`](rag_project_retrieval.json) | 实验方案与合同资料 20 题检索结果 |
| [`rag_retrieval_expanded.json`](rag_retrieval_expanded.json) | 58 文档、426 切片上的 100 题扩展检索结果 |

## 最终结果

- 20 题真实 Qwen 自动严格通过 14/20，平均必需关键词召回 91.25%，引用原文逐字匹配率 100%，串行 p95 延迟 0.433 秒。
- 项目资料 20 题 BM25 检索 Hit@5 为 100%，MRR@10 为 83.25%。
- 扩展知识 100 题 BM25 检索 Hit@5 为 91%，MRR@10 为 77.24%。

上述指标分别衡量固定自动合同或文档检索命中，不是人工回答正确率、CNN 准确率或工业诊断准确率。当前项目的统一说明见仓库根目录 [`README.md`](../../README.md)。
