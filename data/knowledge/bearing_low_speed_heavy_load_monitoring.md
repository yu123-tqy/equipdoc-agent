---
doc_id: bearing_low_speed_heavy_load_monitoring
title: 低速重载和大型轴承的监测特点
equipment: large_slow_speed_bearing
fault_type: condition_monitoring
signal_type: multi_sensor
source_type: authoritative_summary
source_refs: SKF_ROLLING_BEARINGS|SKF_VIBRATION_GUIDE|TIMKEN_DAMAGE_GUIDE
review_status: draft_general_knowledge
keywords: 低速轴承,重载轴承,大型轴承,推力载荷,冲击,声发射,超声,油液,磨粒,长时采样
---
# 低速诊断难点

低速轴承的故障冲击重复间隔长，短记录可能不足以包含多个故障周期；整体振动能量也可能较低。若仍使用为高速设备设置的短时采样和高通参数，容易漏检或得到不稳定特征。采样时长应覆盖足够转数并记录真实转速。

## 重载与润滑

低速重载接触需要关注润滑膜、基础油黏度、供油方式、启停边界润滑和静载塑性变形。大型推力轴承还应结合轴向位移、载荷分配、温度场和润滑流量。低速度不意味着低风险，重载、冲击和偏载可能主导寿命。

## 适用监测手段

除低频振动和长时波形外，可结合高频包络、超声或声发射、温度、油液颗粒、磨粒和扭矩或载荷信息。高频方法对传感器耦合和结构衰减敏感，油液方法又受系统混合和过滤影响，应以多源趋势交叉验证。

## 变速和正反转

低速设备常经历启停、爬行、正反转和负载切换。应按运行阶段分组建立基线，避免把静止振动、结构冲击或换向瞬态与滚动接触故障混为一谈。故障频率分析需要转速同步或按转数重采样。

## 处理边界

若记录时长不足一个或少数几个轴转周期，不能据此排除局部缺陷。大型轴承是否拆检成本高，应通过风险、趋势和多源证据制定检查窗口，而不能由单次模型分类自动决定。
