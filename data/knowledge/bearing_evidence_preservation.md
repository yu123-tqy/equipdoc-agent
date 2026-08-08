---
doc_id: bearing_evidence_preservation
title: 轴承故障证据保存、标识和审计要求
equipment: bearing
fault_type: root_cause_analysis
signal_type: records_inspection
source_type: authoritative_summary
source_refs: SKF_FAILURE_ANALYSIS|TIMKEN_DAMAGE_GUIDE|NSK_MAINTENANCE_GUIDE
review_status: draft_general_knowledge
keywords: 证据保存,拆检记录,照片,润滑样品,方向标记,链路追踪,原始数据,审计
---
# 拆卸前记录

记录设备、轴承位置、型号与批次、安装日期、运行小时、旋转方向、载荷方向、内外圈相对位置和异常时间线。用不会污染分析的方式在部件和包装上建立对应编号。若左右、工作侧和非工作侧混淆，载荷路径分析会失去意义。

## 数据保存

保留未经处理的原始波形、转速、温度、载荷和报警日志，同时保存分析参数、软件版本和处理结果。截图不能替代原始数据。若数据来自导出文件，应记录导出时间、通道、单位和哈希，防止后续版本混用。

## 照片与样品

拆卸、清洗前后均应拍照，包含全景、局部、尺度和方向。润滑剂应从有代表性的位置取样并避免交叉污染，记录取样点、时间和容器。密封件、过滤器残留和碎片可能包含根因证据，不应随意丢弃。

## 避免破坏痕迹

不要让拆卸力通过滚动体，不要在分析前打磨滚道、去除变色、混合不同位置的碎片或用不当清洗剂改变表面。必须切割时，应先完成无损检查并记录切割位置。

## 审计表达

报告应能从每个结论追溯到照片、尺寸、样品、监测数据或有效资料。无法追溯的现场口述应标为未验证信息。模型输出、预测标签和自动摘要必须与原始证据分层存放。
