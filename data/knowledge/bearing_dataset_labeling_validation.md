---
doc_id: bearing_dataset_labeling_validation
title: 轴承诊断数据集、标签和模型验证要求
equipment: bearing
fault_type: machine_learning_validation
signal_type: vibration_dataset
source_type: authoritative_summary
source_refs: CWRU_BEARING_DATA|ISO_20816_1
review_status: draft_general_knowledge
keywords: 数据集,标签,训练集,验证集,测试集,GroupSplit,数据泄漏,工况,混淆矩阵,宏平均F1
---
# 标签的含义

轴承数据标签可能来自人工制造缺陷、拆检确认、维修记录、专家判断或算法规则，其可信度不同。应记录故障位置、失效模式、严重度、轴承型号、工况、采样系统和标签来源。仅有“内圈、外圈、滚动体、正常”四类标签不能支持润滑、电蚀或根因判断。

## 防止数据泄漏

同一原始文件的相邻窗口高度相关，若随机分到训练集和测试集，会高估泛化性能。应按原始采集文件、轴承个体、设备、工况或试验批次进行 Group Split。归一化、特征选择和阈值调优也必须只使用训练数据拟合。

## 评估指标

除总体准确率外，应报告每类召回率、精确率、宏平均 F1、混淆矩阵、置信区间和各工况结果。类别不平衡时总体准确率可能掩盖少数类失败。还应分别测试未知工况、噪声、传感器变化和资料库外故障。

## 置信度校准

Softmax 最大值不天然等于真实正确概率。应在独立数据上评估可靠性和校准，并设置拒识或人工复核路径。模型输出60%置信度表示模型分布中的相对分数，不能说设备有60%的故障概率。

## 可复现记录

应保存数据清单、哈希、分组键、预处理、随机种子、代码提交、模型权重来源和运行环境。任何结果都要能对应到同一版本的代码、输入和配置。
