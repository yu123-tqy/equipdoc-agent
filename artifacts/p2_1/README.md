# P2.1 最终 Agentic 评测证据

本目录保留 2026-08-01 在 AutoDL RTX 4090 上运行真实 Qwen + CNN 的最终 P2.1 固定评测证据。评测输入为 56 cases / 64 turns，冻结输入 SHA-256 为 `7f7613ad09819300dbc6edbb98e9d2383d774a3f0cbfee4c939573392dbb23b8`。

## 文件索引

| 文件 | 内容 |
|---|---|
| [`formal_service_check.json`](formal_service_check.json) | 正式运行前的 Qwen、GPU 和 `READY` 探针记录 |
| [`agentic_eval.json`](agentic_eval.json) | 最终 64-turn 自动合同结果和完整回答 |
| [`agentic_eval_human_review.xlsx`](agentic_eval_human_review.xlsx) | 预填人工复核工作簿，人工评分保持空白 |

## 归档结果

- 固定自动合同通过 64/64；
- 平均 / p50 / p95 端到端延迟为 7.472 / 8.541 / 16.758 秒；
- 52 个规划 turn 中，25 个模型计划在首轮或重试后被接受，27 个使用确定性 fallback；
- 38 个证据回答使用抽取式 fallback；
- 知识检索 38 次、只读信号检查 6 次、轴承诊断 12 次；
- 人工质量复核为 0/64，CNN 工业诊断准确率未在该评测中测量。

64/64 表示冻结输入下，模型、规则、重试和 fallback 组成的联合系统通过自动结构化合同。它不等于人工回答正确率、模型规划准确率或工业诊断准确率。当前项目的统一说明见仓库根目录 [`README.md`](../../README.md)。
