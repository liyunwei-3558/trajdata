# SinD 交叉口几何构型与暴露度

更新时间：2026-05-05  
脚本：`risk_mining/geometric_topology_exposure.py`  
结果目录：`risk_mining/output_geometric_topology_exposure`

## 指标含义

本指标实现 “Geometric Topology & Exposure” 的讨论图组，核心目的是把无引导冲突区面积 `A_c` 与车辆在路口危险区域中的暴露代价关联起来。

当前使用用户给定的几何表：

| location | Ac m2 | angle deg |
|---|---:|---:|
| cc | 691.912 | approx 90 |
| tj | 1028.819 | approx 90 |
| cqIR | 1387.245 | 77.5 |
| cqNR | 1099.427 | 91.4 |
| cqR | 2422.354 | 86.8 |
| xasl | 2593.371 | 86.2 |

重要边界：

- 只有 6 个路口样本，相关系数只用于描述趋势，不做显著性或因果强宣称。
- `A_c` 与暴露指标不一定单调；当前结果中 `xasl` 对驻留时间趋势贡献明显。
- 路权重叠指标依赖 pedestrian/crosswalk 信号绑定，`cc/tj/xasl` 当前 VRU 合法绿灯过街不可观测，不能解释为这些路口没有 VRU 交互。

## 可讨论的相关关系

当前报告生成以下图组：

- `A_c` vs 平均/P90 路口核心驻留时间：说明无引导区域规模与车辆在核心冲突区的时间暴露。
- `A_c` vs 长驻留车辆比例：统计 `seconds_in_roi >= 5s` 和 `>=10s` 的比例，用于描述长时间暴露尾部。
- `A_c` vs 单车平均占用秒数：`seconds_in_roi_total / vehicle_tracks`，用于控制车流量差异后的驻留代价。
- `A_c` vs 多体博弈复杂度：`N>=3` 交互比例和平均交互阶数，用于讨论大面积路口的多主体博弈空间。
- `A_c` vs 博弈持续时长：P90 interaction duration 和 `>=5s` 长博弈比例。
- `A_c` vs 合法绿灯路权重叠：合法机动车 passage 与合法 VRU crossing 的空间重叠点，解释为什么合法通行也可能产生强交互。
- 角度偏斜度 `|90 - angle|` vs 暴露/冲突指标：作为辅助讨论，不作为主结论。

## 运行脚本

```bash
conda run -n trajdata python risk_mining/geometric_topology_exposure.py \
  --output-dir risk_mining/output_geometric_topology_exposure
```

输入来自已有指标输出：

- `risk_mining/output_intersection_spatiotemporal_density/density_summary.csv`
- `risk_mining/output_intersection_spatiotemporal_density/track_roi_residence.csv`
- `risk_mining/output_interaction_topology_complexity/interaction_degree_summary_by_location.csv`
- `risk_mining/output_interaction_topology_complexity/interaction_duration_summary.csv`
- `risk_mining/output_conflict_patterns_spatial_clustering/conflict_type_summary_by_location.csv`
- `risk_mining/output_right_of_way_overlap/right_of_way_overlap_summary_by_location.csv`

## 可视化入口

- 总报告：`risk_mining/output_geometric_topology_exposure/index.html`
- 主图：`risk_mining/output_geometric_topology_exposure/area_vs_residence_time.png`
- 长驻留比例图：`risk_mining/output_geometric_topology_exposure/area_vs_long_residence_ratio.png`
- 归一化占用图：`risk_mining/output_geometric_topology_exposure/area_vs_occupancy.png`
- 多体复杂度图：`risk_mining/output_geometric_topology_exposure/area_vs_interaction_complexity.png`
- 博弈持续时间图：`risk_mining/output_geometric_topology_exposure/area_vs_interaction_duration.png`
- 合法路权重叠图：`risk_mining/output_geometric_topology_exposure/area_vs_right_of_way_overlap.png`
- 角度偏斜辅助图：`risk_mining/output_geometric_topology_exposure/angle_skew_vs_exposure.png`
- 相关矩阵：`risk_mining/output_geometric_topology_exposure/geometric_correlation_heatmap.png`

## 结果保存位置

核心 CSV/JSON：

- `geometric_exposure_metrics.csv`：每个路口一行的几何与暴露指标合并表。
- `geometric_exposure_correlations.csv`：`A_c` 和角度偏斜度与各暴露指标的 Pearson/Spearman 相关系数。
- `methodology_notes.json`：样本量、几何数据来源和解释边界。

## 当前结果概览

| location | Ac m2 | mean residence | P90 residence | N>=3 ratio | P90 game duration | legal overlap cells |
|---|---:|---:|---:|---:|---:|---:|
| cc | 691.912 | 3.942s | 7.200s | 0.440 | 7.580s | 0 |
| tj | 1028.819 | 3.994s | 6.000s | 0.342 | 10.100s | 0 |
| cqIR | 1387.245 | 3.612s | 7.000s | 0.540 | 21.270s | 2 |
| cqNR | 1099.427 | 3.254s | 4.800s | 0.352 | 15.700s | 0 |
| cqR | 2422.354 | 3.275s | 4.900s | 0.423 | 8.900s | 33 |
| xasl | 2593.371 | 6.573s | 10.500s | 0.387 | 9.800s | 0 |

可优先讨论的趋势：

- `A_c` 与平均驻留时间的 Pearson 约为 `0.51`，但 Spearman 较弱，说明趋势主要由大面积且长驻留的 `xasl` 拉动。
- `cqR` 的 `A_c` 很大，同时合法绿灯路权重叠空间点最多，可作为“合法通行也产生路径重叠”的典型案例。
- `cqIR` 的面积中等但博弈 P90 持续时间高，说明几何面积不是唯一因素，信号相位、流量结构和渠化方式也会影响暴露代价。
