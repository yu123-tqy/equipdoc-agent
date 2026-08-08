---
doc_id: bearing_electrical_erosion
title: 轴承电蚀、电流损伤和防护思路
equipment: electric_motor_bearing
fault_type: electrical_erosion
signal_type: vibration_electrical_inspection
source_type: authoritative_summary
source_refs: SKF_FAILURE_ANALYSIS|SKF_MOTOR_GENERATOR|NSK_BEARING_DOCTOR
review_status: draft_general_knowledge
keywords: 电蚀,轴电流,轴电压,电火花,电弧,麻点,搓板纹,绝缘轴承,接地环,变频器
---
# 电蚀机理

当电流经内圈、滚动体和外圈形成通路时，薄润滑膜可能被击穿并发生局部放电。接触点瞬时加热、熔化和重新凝固，形成微坑；持续运行后，滚道可能出现带状暗痕或规则沟纹，润滑剂也可能受到热和放电影响。

## 可能电流来源

电机和发电机中的磁路不对称可能产生低频循环电流；变频驱动的高频共模电压、接地与屏蔽不合理、焊接电流路径和静电也可能使电流通过轴承。仅发现沟纹还不足以确定具体电流源，需要测量轴电压、轴承电流或相关高频信号并审查电气系统。

## 运行与拆检线索

电蚀可伴随噪声和振动逐渐增加，频谱可能出现与沟纹间距和转速相关的宽带或离散成分。拆检可观察微坑、熔融边缘、灰暗带或波纹状沟槽。类似沟纹也可能由机械振动和磨损形成，因此应结合显微外观、电气测量和润滑分析。

## 防护措施

可能措施包括优化接地和屏蔽、使用轴接地装置、绝缘轴承或绝缘套、在适当位置采用混合陶瓷轴承，并确保焊接回流不经过轴承。方案应由电气与机械专业共同确定；错误地只绝缘一端可能改变电流路径，使其他轴承或联轴器受损。

## 处理边界

RAG 可以提示电蚀候选原因和检查项，但不能根据振动波形自动决定绝缘结构或变频器参数。涉及带电测量、接地改造和驱动器调整时，应遵循设备制造商与电气安全规程。
