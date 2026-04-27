<!--
 * @Author: Yunwei Li 1084087910@qq.com
 * @Date: 2026-04-25 15:35:12
 * @LastEditors: Yunwei Li 1084087910@qq.com
 * @LastEditTime: 2026-04-25 15:35:56
 * @FilePath: /My_trajdata/Task_docs/KinematicEnvelopes.md
 * @Description:
 * 
 * Copyright (c) 2026 by Tsinghua University, All Rights Reserved. 
-->


### **任务指令：提取语义约束下的运动学包络 (Kinematic Envelopes)**

#### **1. 核心目标**
从轨迹数据中提取不同语义条件下（城市、车型、意图）的 $v-a$ 联合分布，并计算 **95% 置信区间边界（Envelope）**，用以识别超出常规驾驶范畴的“高危/极端行为（Corner Cases）”。

#### **2. 数据预处理（Data Engineering）**
* **噪声平滑：** 原始坐标转换出的 $a$（加速度）噪声很大。**必须**使用 `Savitzky-Golay` 滤波器或 `Kalman Filter` 对速度和加速度进行平滑处理。
* **语义切片（Semantic Slicing）：** 编写一个 `filter_tracks` 函数，能够根据以下标签组合提取子集：
    * `City`: [XiAn, ChangChun, etc.]
    * `AgentType`: [Car, E-bike, Pedestrian]
    * `Maneuver`: [Straight, Left-turn, Right-turn]
* **采样对齐：** 将同一 ID 在路口内的所有帧作为样本点，形成 $(v, a)$ 坐标对阵列。

#### **3. 边界计算算法（Statistical Method）**
建议使用以下两种方法之一来计算“包络线”：
* **方法 A (推荐)：2D 核密度估计 (2D-KDE)**。使用 `scipy.stats.gaussian_kde` 计算概率密度，提取累计概率为 95% 的等高线作为包络边界。
* **方法 B (快速)：Alpha Shape (凹包)**。如果数据分布较散，使用 Alpha Shape 算法（比 Convex Hull 更贴合数据边缘）提取最外层边界。

#### **4. 统计产出要求**
* **Outlier Detection：** 自动标记并导出所有落在 95% 包络线之外的轨迹 ID 和时间戳，这些将作为后续“高价值场景”的候选。
* **Cross-Domain Comparison：** 必须输出一张对比图，将不同城市的包络线重叠在一起，计算其**重叠面积比 (Overlap Ratio)** 以量化地域差异。
