---
doc_id: cwru_bearing_dataset_boundaries
title: CWRU轴承数据集的用途和泛化边界
equipment: bearing
fault_type: dataset
signal_type: vibration_dataset
source_type: authoritative_summary
source_refs: CWRU_BEARING_DATA
review_status: draft_general_knowledge
keywords: CWRU,轴承数据中心,试验台,人工故障,驱动端,风扇端,采样率,负载,领域偏移,泛化
---
# 数据集用途

Case Western Reserve University Bearing Data Center 公开了带有试验条件和轴承故障状态记录的电机轴承振动数据，常用于教学、算法原型和不同信号处理方法的对比。使用时应以官方页面和原始文件说明确认采样位置、采样率、负载、转速和故障设置。

## 标签与工况限制

公开试验数据来自受控台架和特定电机、轴承、传感器及人工设置故障。人工缺陷的几何和严重度不必然等同于现场自然退化。不同采样位置、负载和采样率的数据不能在没有记录的情况下混合。

## 常见泄漏风险

从同一长波形切出的相邻窗口高度相似。随机窗口划分可能让同源片段同时出现在训练和测试中，得到不可信的高准确率。应按原始文件或试验条件分组，并明确哪些工况完全留作外部测试。

## 工业泛化边界

CWRU 上的结果不能直接外推到推进器、大型低速推力轴承、不同结构传递路径、变速重载或真实复合故障。迁移应用需要目标设备数据、传感器与工况对齐、领域偏移评估、拒识机制和现场人工验证。

## 正确表述

可以说“模型在指定 CWRU 划分上完成了工具链验证”，不应说“轴承诊断工业准确率达到某个百分比”。若分组键和独立测试未保存，也不应报告可信泛化指标。
