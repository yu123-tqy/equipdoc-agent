---
doc_id: bearing_vibration_features_frequencies
title: 轴承振动时域、频域、包络和故障特征频率
equipment: bearing
fault_type: condition_monitoring
signal_type: vibration
source_type: authoritative_summary
source_refs: SKF_VIBRATION_GUIDE|SKF_FAILURE_ANALYSIS
review_status: draft_general_knowledge
keywords: RMS,峭度,峰值因子,FFT,包络谱,BPFO,BPFI,BSF,FTF,谐波,边频带
---
# 时域指标

RMS 反映所选频带内的总体能量，峰值和峰峰值反映极端幅值，峭度与峰值因子对冲击较敏感。指标会受负载、速度、带宽、采样长度和噪声影响。早期局部缺陷可能在整体 RMS 变化不大时先表现为高频冲击，而严重磨损后峭度反而可能下降。

## 频谱和阶次

FFT 频谱用于观察转频、谐波、故障频率、边带和共振。恒速设备可按 Hz 分析，变速设备更适合转换到相对于转频的阶次。频谱峰值必须结合频率分辨率、窗函数、泄漏、平均方式和传递路径解释。

## 理论故障频率

在理想纯滚动和已知几何下，可由滚动体数量 `n`、滚动体直径 `d`、节圆直径 `D`、接触角 `θ` 和轴转频 `fr` 估算：`FTF = 0.5·fr·(1-d/D·cosθ)`，`BPFO = 0.5·n·fr·(1-d/D·cosθ)`，`BPFI = 0.5·n·fr·(1+d/D·cosθ)`，`BSF = D/(2d)·fr·(1-(d/D·cosθ)^2)`。实际滚动体滑动会造成偏差。

## 包络分析

局部缺陷的短时冲击可激励轴承座或结构的高频共振。对合适的高频带带通、整流或希尔伯特包络并分析包络谱，可突出冲击重复频率。频带选择不当、传感器带宽不足或机械冲击干扰会产生误判，应比较多个频带和原始波形。

## 边带与调制

内圈缺陷进入和离开载荷区、载荷变化、转速波动和保持架运动都可能形成调制与边带。边带间距和幅值趋势可提供线索，但不能单独确定故障严重度。应联合时域、频谱、包络、阶次、工况与趋势分析。
