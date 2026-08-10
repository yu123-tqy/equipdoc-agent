# P2.2 最终演示与正式回归证据

本目录保留 2026-08-01 在 RTX 4090 + `qwen-equipdoc` Full Agentic 环境产生的最终演示和正式回归证据，以及空白人工复核工作簿。

## 文件索引

| 文件 | 内容 |
|---|---|
| [`source_bundle_sha256.txt`](source_bundle_sha256.txt) | AutoDL 评测使用的源码包哈希 |
| [`formal_service_check.json`](formal_service_check.json) | 最终正式回归前的 Qwen、GPU 和 `READY` 探针 |
| [`demo_eval_run_3.json`](demo_eval_run_3.json) | 最终 13-turn 面试演示结果，13/13 |
| [`demo_human_review.xlsx`](demo_human_review.xlsx) | 13-turn 人工复核模板，人工字段保持空白 |
| [`code_sha256_fix3.txt`](code_sha256_fix3.txt) | 最终归档评测代码哈希 |
| [`formal_regression_64_turn_fix3.json`](formal_regression_64_turn_fix3.json) | 最终 56-case / 64-turn 正式回归，64/64 |
| [`MANIFEST.sha256`](MANIFEST.sha256) | 原始 AutoDL 证据包导入清单 |

`MANIFEST.sha256` 用于记录原始证据包曾包含的文件和哈希。最终展示分支已经主动移除修复前、预检和重复运行结果，因此该清单是历史导入审计记录，不是当前精简目录的完整性清单。

## 归档结果

- 最终 13-turn 演示通过 13/13；
- 最终 64-turn 自动合同通过 64/64；
- 平均 / p50 / p95 端到端延迟为 5.100 / 5.889 / 9.926 秒；
- 38 个知识回答使用结构化证据路径；
- 52 个规划 turn 中包含模型首轮、重试和 26 个确定性 fallback；
- 人工复核为 0/13，不能报告人工正确率、人工 groundedness 或引用有用率。

这些结果对应当时冻结的代码、输入和环境，不代表当前开放问题正确率或工业诊断准确率。当前项目的统一说明见仓库根目录 [`README.md`](../../README.md)。
