# SinD 已提取指标汇总

更新时间：2026-04-28  
项目路径：`/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata`

本文档汇总当前已经实现并跑通的 SinD 六路口指标，包含每个指标的含义、运行脚本、可视化入口和结果保存位置。所有命令默认在项目根目录执行，并使用 `trajdata` conda 环境。

## 0. 六个路口与城市合并规则

- 六个 SinD 路口：`cc`、`tj`、`cqIR`、`cqNR`、`cqR`、`xasl`
- 展示名：`ChangChun`、`TianJin`、`ChongQing-IR`、`ChongQing-NR`、`ChongQing-R`、`XiAn-Shanglin`
- 四城市合并：`cc -> ChangChun`，`tj -> TianJin`，`cqIR/cqNR/cqR -> ChongQing`，`xasl -> XiAn`

## 0.1 综合初步结论分析

### 指标含义

综合当前已提取的几何暴露、运动学 domain gap、多体交互、冲突模式、路权重叠、违规/非标行为和两轮车异质性指标，生成论文讨论草稿级结论。该报告强调跨指标证据链，而不是替代各单项指标报告。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/metric_conclusions_report.py \
  --output-dir risk_mining/output_metric_conclusions
```

### 可视化入口

- HTML：`risk_mining/output_metric_conclusions/index.html`
- 核心柱状图：`risk_mining/output_metric_conclusions/conclusion_key_bars.png`
- 六路口归一化指标矩阵：`risk_mining/output_metric_conclusions/conclusion_metric_matrix.png`

### 结果保存位置

- 目录：`risk_mining/output_metric_conclusions`
- 核心文件：`preliminary_findings.csv`、`conclusion_location_dashboard.csv`

### 当前结论摘要

当前报告提炼了 8 条初步结论：几何规模与暴露代价、多体博弈复杂度、冲突模式路口差异、合法路权路径重叠、两轮车动力学异质性、城市间 Wasserstein domain gap、规则背景噪声、驻留热力与冲突拓扑互补证据。

## 1. Kinematic Envelopes：语义约束下的运动学包络

### 指标含义

提取不同城市、交通体类型、意图下的速度-纵向加速度联合分布 `(v, a)`，用 2D-KDE 计算 95% 运动学包络。包络外样本可作为极端运动学行为或 corner cases 候选。

当前已跑结果使用：`AgentType=All`，`Maneuver=All`。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/kinematic_envelopes.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --agent-type All \
  --maneuver All \
  --output-dir risk_mining/output_kinematic_envelopes_six_all
```

### 可视化入口

- HTML：`risk_mining/output_kinematic_envelopes_six_all/index.html`
- 六路口包络叠加图：`risk_mining/output_kinematic_envelopes_six_all/cross_city_envelopes.png`
- 单路口包络图：`risk_mining/output_kinematic_envelopes_six_all/envelope_<city>.png`

### 结果保存位置

- 目录：`risk_mining/output_kinematic_envelopes_six_all`
- 核心文件：`outliers.csv`、`outlier_tracks.csv`、`top_outlier_tracks.csv`、`overlap_ratios.csv`、`envelopes.geojson`、`summary.json`

### 当前结果概览

| location | tracks | v-a points | outlier samples |
|---|---:|---:|---:|
| cc | 7,940 | 2,320,498 | 134,096 |
| tj | 12,963 | 3,789,405 | 198,304 |
| cqIR | 5,668 | 2,018,445 | 104,773 |
| cqNR | 3,848 | 3,443,299 | 181,921 |
| cqR | 12,804 | 3,358,513 | 157,018 |
| xasl | 6,630 | 2,800,783 | 146,609 |

## 2. Kinematic Wasserstein Domain Gap：城市间运动学分布差异

### 指标含义

用 Wasserstein Distance / Earth Mover's Distance 严格量化城市间 `(v, a)` 联合分布差异。距离越大，说明两个城市的运动学 domain gap 越大，跨城市迁移学习难度越高。

当前主指标为 robust-normalized 2D Wasserstein；同时输出 `W(v)` 和 `W(a)` 一维分解。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/kinematic_wasserstein_domain_gap.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --agent-type All \
  --maneuver All \
  --output-dir risk_mining/output_kinematic_wasserstein_domain_gap
```

### 可视化入口

- HTML：`risk_mining/output_kinematic_wasserstein_domain_gap/index.html`
- 4x4 城市主热力图：`risk_mining/output_kinematic_wasserstein_domain_gap/wasserstein_heatmap_city_2d.png`
- 6x6 路口热力图：`risk_mining/output_kinematic_wasserstein_domain_gap/wasserstein_heatmap_location_2d.png`
- 速度差异热力图：`risk_mining/output_kinematic_wasserstein_domain_gap/wasserstein_heatmap_city_velocity.png`
- 加速度差异热力图：`risk_mining/output_kinematic_wasserstein_domain_gap/wasserstein_heatmap_city_acceleration.png`

### 结果保存位置

- 目录：`risk_mining/output_kinematic_wasserstein_domain_gap`
- 核心文件：`wasserstein_2d_city_matrix.csv`、`wasserstein_2d_location_matrix.csv`、`wasserstein_city_pairs_long.csv`、`wasserstein_location_pairs_long.csv`、`wasserstein_1d_city_pairs.csv`、`domain_sample_summary.csv`、`summary.json`

### 当前 4x4 城市主矩阵

| city | ChangChun | TianJin | ChongQing | XiAn |
|---|---:|---:|---:|---:|
| ChangChun | 0.0000 | 0.9121 | 0.8070 | 0.5288 |
| TianJin | 0.9121 | 0.0000 | 0.2317 | 0.4477 |
| ChongQing | 0.8070 | 0.2317 | 0.0000 | 0.3548 |
| XiAn | 0.5288 | 0.4477 | 0.3548 | 0.0000 |

最大 domain gap 为 `ChangChun <-> TianJin`，最小 domain gap 为 `TianJin <-> ChongQing`。

## 3. Lateral Deviation Variance：转弯横向偏移轨迹方差

### 指标含义

针对左转/右转车辆，将实际轨迹投影到 Lanelet2 参考中心线，计算横向偏移均值、方差和 `±2σ` 轨迹包络。方差越大，说明切弯、大迂回、非结构化轨迹越严重。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/lateral_deviation_variance.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --agent-type Vehicle \
  --maneuvers left-turn right-turn \
  --output-dir risk_mining/output_lateral_deviation_variance
```

### 可视化入口

- HTML：`risk_mining/output_lateral_deviation_variance/index.html`
- 轨迹簇图：`risk_mining/output_lateral_deviation_variance/bundle_*.png`

### 结果保存位置

- 目录：`risk_mining/output_lateral_deviation_variance`
- 核心文件：`track_lateral_deviation.csv`、`cluster_lateral_deviation.csv`、`unmatched_tracks.csv`、`bundle_manifest.csv`、`summary.json`

### 当前结果提示

`cluster_lateral_deviation.csv` 中保存了每个 `location + maneuver + reference_lane_id` 的 `mean_track_var_lateral_offset_m2`、`pooled_var_lateral_offset_m2`、`p95_abs_lateral_offset_m` 等指标。部分 lane 的匹配轨迹数很少，论文使用时建议优先筛选 `matched_tracks` 足够大的簇。

## 4. Intersection Spatiotemporal Density：路口内部时空密度

### 指标含义

将路口内部区域划分为 `1m x 1m` 网格，累计车辆中心点在每个网格中的驻留时间。热力越高，说明该区域更可能是锁死区、博弈滞留区或高频交互区。

当前实现使用 Lanelet2 内部 connector 推断更紧的路口核心 ROI，并默认过滤长期静止/沿岸停车轨迹。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/intersection_spatiotemporal_density.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_intersection_spatiotemporal_density
```

### 可视化入口

- HTML：`risk_mining/output_intersection_spatiotemporal_density/index.html`
- 驻留热力图：`risk_mining/output_intersection_spatiotemporal_density/occupancy_heatmap_<city>.png`

### 结果保存位置

- 目录：`risk_mining/output_intersection_spatiotemporal_density`
- 核心文件：`density_summary.csv`、`hotspot_cells.csv`、`track_roi_residence.csv`、`filtered_static_tracks.csv`、`occupancy_grid_<city>.npz`、`core_roi_<city>.geojson`、`summary.json`

### 当前结果概览

| location | vehicle tracks | occupancy seconds | max cell seconds | occupied cells |
|---|---:|---:|---:|---:|
| cc | 7,206 | 28,407.0 | 149.0 | 709 |
| tj | 5,271 | 21,054.5 | 134.0 | 688 |
| cqIR | 3,551 | 12,827.6 | 100.3 | 725 |
| cqNR | 2,007 | 6,530.3 | 102.0 | 609 |
| cqR | 9,587 | 31,399.2 | 129.2 | 1,093 |
| xasl | 5,176 | 34,022.9 | 343.8 | 1,442 |

## 5. Critical Gap Acceptance：可接受间隙阈值

### 指标含义

在无保护左转场景中，统计左转 Ego 决定进入路口时，与直行冲突车辆之间的 accepted time gap 和 distance gap。时间间隙越小，说明抢行行为越激进。

当前实现：Ego 为 `left-turn` vehicle，冲突车为同 scene 内 `straight` vehicle，决策时刻为 Ego 首次进入路口核心 ROI。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/critical_gap_acceptance.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_critical_gap_acceptance
```

### 可视化入口

- HTML：`risk_mining/output_critical_gap_acceptance/index.html`
- 城市对比图：`risk_mining/output_critical_gap_acceptance/critical_gap_city_comparison.png`
- CDF 图：`risk_mining/output_critical_gap_acceptance/critical_gap_cdf_<city>.png`
- 直方图：`risk_mining/output_critical_gap_acceptance/critical_gap_hist_<city>.png`

### 结果保存位置

- 目录：`risk_mining/output_critical_gap_acceptance`
- 核心文件：`critical_gap_summary_by_city.csv`、`critical_gap_accepted.csv`、`critical_gap_pairs.csv`、`low_gap_cases.csv`、`summary.json`

### 当前结果概览

| location | accepted samples | time gap median | time gap P10-P75 | distance gap median |
|---|---:|---:|---:|---:|
| cc | 452 | 3.60s | 2.41-5.20s | 22.59m |
| tj | 299 | 5.10s | 2.88-6.35s | 27.24m |
| cqIR | 7 | 6.40s | 5.60-6.50s | 26.86m |
| cqNR | 55 | 5.00s | 3.00-6.30s | 29.18m |
| cqR | 2 | 7.55s | 7.51-7.58s | 51.61m |
| xasl | 276 | 4.60s | 2.60-5.90s | 37.02m |

`cqIR` 和 `cqR` 样本数较少，建议在论文中标注低置信度或作为补充观察。

## 6. Interaction Topology & Game Complexity：交互拓扑与博弈复杂度

### 指标含义

衡量路口交互复杂度，而不是危险性。逐帧构建交互图：若两个交通体距离足够近且未来轨迹存在潜在交点，则连边；连通分量大小 `N` 即交互阶数。`N >= 3` 的比例可用于证明中国路口存在大量多体博弈。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/interaction_topology_complexity.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_interaction_topology_complexity
```

### 可视化入口

- HTML：`risk_mining/output_interaction_topology_complexity/index.html`
- 六路口交互阶数堆叠柱状图：`risk_mining/output_interaction_topology_complexity/stacked_degree_by_location.png`
- 四城市交互阶数堆叠柱状图：`risk_mining/output_interaction_topology_complexity/stacked_degree_by_city.png`
- 持续时长分布：`risk_mining/output_interaction_topology_complexity/duration_distribution_by_location.png`
- 城市持续时长箱线图：`risk_mining/output_interaction_topology_complexity/duration_boxplot_by_city.png`

### 结果保存位置

- 目录：`risk_mining/output_interaction_topology_complexity`
- 核心文件：`interaction_degree_summary_by_location.csv`、`interaction_degree_summary_by_city.csv`、`interaction_frame_components.csv`、`interaction_duration_episodes.csv`、`interaction_duration_summary.csv`、`top_complex_episodes.csv`、`summary.json`

### 当前交互阶数结果

| location | component frames | 2-party | 3-party | 4+-party | N>=3 | mean N | max N |
|---|---:|---:|---:|---:|---:|---:|---:|
| cc | 90,047 | 0.560 | 0.242 | 0.198 | 0.440 | 2.765 | 11 |
| tj | 227,071 | 0.658 | 0.212 | 0.129 | 0.342 | 2.567 | 12 |
| cqIR | 134,552 | 0.460 | 0.220 | 0.320 | 0.540 | 3.460 | 28 |
| cqNR | 48,388 | 0.648 | 0.213 | 0.139 | 0.352 | 2.617 | 15 |
| cqR | 128,003 | 0.577 | 0.206 | 0.217 | 0.423 | 2.910 | 21 |
| xasl | 87,820 | 0.613 | 0.203 | 0.184 | 0.387 | 2.823 | 21 |

### 当前博弈持续时长结果

| location | episodes | median duration | P90 duration | >=5s ratio |
|---|---:|---:|---:|---:|
| cc | 2,693 | 2.0s | 7.58s | 0.194 |
| tj | 4,801 | 2.7s | 10.10s | 0.274 |
| cqIR | 1,454 | 3.6s | 21.27s | 0.393 |
| cqNR | 822 | 3.2s | 15.70s | 0.314 |
| cqR | 2,617 | 3.8s | 8.90s | 0.326 |
| xasl | 1,918 | 2.5s | 9.80s | 0.283 |

## 7. Conflict Patterns & Spatial Clustering：冲突模式与空间聚集

### 指标含义

统计车辆/两轮车 pair 在路口中以什么意图组合发生冲突，并提取显著减速避让点的空间热点。该指标回答“大家在哪里起冲突，以及以什么姿势起冲突”。

冲突 pair 条件：同帧距离 `<=15m` 且未来 `5s` 轨迹最近距离 `<=3m`。热点事件条件：`a_lon < -1.5m/s^2`，且前方存在潜在冲突参与者。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/conflict_patterns_spatial_clustering.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_conflict_patterns_spatial_clustering
```

### 可视化入口

- HTML：`risk_mining/output_conflict_patterns_spatial_clustering/index.html`
- 弦图：`risk_mining/output_conflict_patterns_spatial_clustering/conflict_chord_<city>.png`
- 城市合并弦图：`risk_mining/output_conflict_patterns_spatial_clustering/conflict_chord_city_groups.png`
- 减速避让热点图：`risk_mining/output_conflict_patterns_spatial_clustering/conflict_hotspot_<city>.png`

### 结果保存位置

- 目录：`risk_mining/output_conflict_patterns_spatial_clustering`
- 核心文件：`conflict_pair_events.csv`、`conflict_pair_episodes.csv`、`conflict_type_summary_by_location.csv`、`conflict_type_summary_by_city.csv`、`deceleration_hotspot_events.csv`、`deceleration_hotspot_cells.csv`、`conflict_hotspot_grid_<city>.npz`、`summary.json`

### 当前冲突类型比例

| location | pair episodes | Crossing | Merging | Weaving | Turning | Other |
|---|---:|---:|---:|---:|---:|---:|
| cc | 7,943 | 0.444 | 0.049 | 0.328 | 0.154 | 0.025 |
| tj | 9,256 | 0.464 | 0.159 | 0.196 | 0.165 | 0.016 |
| cqIR | 3,247 | 0.034 | 0.153 | 0.381 | 0.360 | 0.073 |
| cqNR | 792 | 0.370 | 0.063 | 0.367 | 0.187 | 0.013 |
| cqR | 6,060 | 0.020 | 0.014 | 0.660 | 0.243 | 0.064 |
| xasl | 4,169 | 0.299 | 0.068 | 0.339 | 0.169 | 0.124 |

### 当前显著减速避让热点事件数

| location | hotspot events |
|---|---:|
| cc | 4,237 |
| tj | 2,527 |
| cqIR | 2,512 |
| cqNR | 246 |
| cqR | 931 |
| xasl | 1,311 |


## 8. Violations & Non-compliant Baseline：违规与非标行为基线

### 指标含义

描述 SinD 路口中的规则背景噪声，包含机动车结构化违规、路权候选和 VRU 侵入/非标行为。当前已使用本地交通灯 pkl 与人工灯控绑定，红灯和黄灯分开统计；不可观测区域不会被误报为 0 违规。

机动车侧输出：红灯进入、黄灯进入、路口核心外进出路段逆行候选、入口实线区变道 proxy、车道方向规则不一致、路权不让行候选。路权候选要求不同来向且 ROI 内轨迹有共享冲突点。VRU 侧仅统计行人/自行车在斑马线外的机动车道占用、路口核心区非斑马线驻留，以及绑定人行灯为红/黄时的冲突区停留；正常过 crosswalk 和摩托车不计入 VRU 指标。

说明：`intersection_lane_switch_proxy` 已移除，不再把路口核心内部的最近 lanelet 切换作为违规指标。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/structured_violations_noncompliance.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_structured_violations_noncompliance
```

### 可视化入口

- HTML：`risk_mining/output_structured_violations_noncompliance/index.html`
- 汇总柱状图：`risk_mining/output_structured_violations_noncompliance/summary_violation_rates.png`
- 机动车事件热点：`risk_mining/output_structured_violations_noncompliance/vehicle_violation_hotspot_<location>.png`
- VRU 侵入热点：`risk_mining/output_structured_violations_noncompliance/vru_encroachment_<location>.png`
- 每类违规实例：`risk_mining/output_structured_violations_noncompliance/examples/<event_type>/*.png`

### 结果保存位置

- 目录：`risk_mining/output_structured_violations_noncompliance`
- 核心文件：`summary_by_location.csv`、`summary_by_city.csv`、`vehicle_violation_events.csv`、`vehicle_violation_track_metrics.csv`、`vru_noncompliance_events.csv`、`vru_noncompliance_track_metrics.csv`、`signal_observability_by_location.csv`、`right_of_way_yield_events.csv`、`wrong_way_events_by_agent_type.csv`、`violation_example_manifest.csv`、`lane_direction_diagnostics.csv`、`filtered_static_tracks.csv`、`violation_rois.geojson`、`methodology_notes.json`
- 详细说明：`Task_docs/ViolationsNonCompliance.md`

### 当前结果说明

该指标已升级为信号灯 pkl + 人工绑定版本。由于地图和绑定刚完成返工，正式论文数字建议重新完整运行后以 `summary_by_location.csv` 为准；报告中的 `examples/<event_type>/*.png` 可用于逐类人工核查事件合理性。

## 9. E-bike vs Bicycle Separation：两轮车动力学异质性分离

### 指标含义

将 SinD 中的行人、自行车、电动二轮车/电摩 proxy、三轮车分开统计，比较它们在路口核心区域的通行平均速度、巡航速度和最大起步加速度。该指标用于论证 powered two-wheeler 在中国路口中的高速穿插特性。

分类口径：原始 `motorcycle` 作为 `e_bike / powered two-wheeler proxy`；原始 `bicycle` 中若 p85 速度形成显著高速簇，则进一步拆为 likely e-bike/scooter。论文表述时应说明这是基于原始标签和速度聚类的 proxy，不是车辆注册类型的精确标注。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/two_wheeler_heterogeneity.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_two_wheeler_heterogeneity
```

### 可视化入口

- HTML：`risk_mining/output_two_wheeler_heterogeneity/index.html`
- 小提琴图：`risk_mining/output_two_wheeler_heterogeneity/passage_speed_violin.png`

### 结果保存位置

- 目录：`risk_mining/output_two_wheeler_heterogeneity`
- 核心文件：`two_wheeler_track_metrics.csv`、`two_wheeler_summary_by_class.csv`、`two_wheeler_summary_by_location.csv`、`bicycle_split_report.json`、`filtered_static_tracks.csv`、`roi_debug.csv`
- 详细说明：`Task_docs/TwoWheelerHeterogeneity.md`

### 当前结果概览

| class | tracks | mean passage speed | median passage speed | mean startup accel |
|---|---:|---:|---:|---:|
| pedestrian | 5,680 | 1.176 m/s | 1.167 m/s | 0.357 m/s^2 |
| bicycle | 1,954 | 2.622 m/s | 2.588 m/s | 0.516 m/s^2 |
| e_bike proxy | 8,308 | 5.435 m/s | 5.166 m/s | 0.476 m/s^2 |
| tricycle | 920 | 4.852 m/s | 4.623 m/s | 0.631 m/s^2 |

## 10. Spatio-Temporal Right-of-Way Overlap：时空路权重叠度

### 指标含义

统计同一绿灯相位期间，合法机动车通行轨迹簇与合法行人/自行车过街轨迹簇的空间交点数量。该指标解释“为什么合法行驶也会发生激烈交互”：如果绿灯释放的不同流线本身存在空间重叠，预测和规划算法就必须处理更复杂的多主体博弈。

当前主口径只统计双方信号均可观测且为 `GREEN` 的 passage；缺少 crosswalk/车道灯控绑定的轨迹进入 diagnostics，不混入分母。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/right_of_way_overlap.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_right_of_way_overlap
```

### 可视化入口

- HTML：`risk_mining/output_right_of_way_overlap/index.html`
- 汇总图：`risk_mining/output_right_of_way_overlap/right_of_way_overlap_summary.png`
- 单路口热点图：`risk_mining/output_right_of_way_overlap/overlap_hotspot_<location>.png`

### 结果保存位置

- 目录：`risk_mining/output_right_of_way_overlap`
- 核心文件：`legal_green_passages.csv`、`right_of_way_overlap_events.csv`、`right_of_way_overlap_summary_by_location.csv`、`right_of_way_overlap_cluster_summary.csv`、`passage_observability_diagnostics.csv`、`signal_table_reports.csv`、`methodology_notes.json`
- 详细说明：`Task_docs/RightOfWayOverlap.md`

### 当前结果概览

| location | legal vehicle | legal VRU crossing | overlap events | unique 1m cells |
|---|---:|---:|---:|---:|
| cc | 5,258 | 0 | 0 | 0 |
| tj | 3,919 | 0 | 0 | 0 |
| cqIR | 858 | 288 | 3 | 2 |
| cqNR | 1,325 | 387 | 0 | 0 |
| cqR | 1,380 | 251 | 215 | 33 |
| xasl | 3,372 | 0 | 0 | 0 |

`cc/tj/xasl` 的 VRU 合法绿灯过街当前为 0，主要表示 pedestrian/crosswalk 灯控绑定不可观测，不应解释为这些路口没有合法 VRU 过街交互。

## 11. Geometric Topology & Exposure：交叉口几何构型与暴露度

### 指标含义

将用户给定的无引导冲突区面积 `A_c` 和路口夹角，与车辆在路口核心区的驻留时间、占用强度、多体博弈复杂度、博弈持续时长和合法绿灯路权重叠进行相关讨论。该图组服务于“冲突区越大，车辆在危险区域的停留代价和交互预测负担越高”的章节叙事。

说明：当前只有 6 个路口样本，相关系数只用于描述趋势，不做强显著性宣称。

### 运行脚本

```bash
conda run -n trajdata python risk_mining/geometric_topology_exposure.py \
  --output-dir risk_mining/output_geometric_topology_exposure
```

### 可视化入口

- HTML：`risk_mining/output_geometric_topology_exposure/index.html`
- 面积-驻留时间图：`risk_mining/output_geometric_topology_exposure/area_vs_residence_time.png`
- 面积-长驻留比例图：`risk_mining/output_geometric_topology_exposure/area_vs_long_residence_ratio.png`
- 面积-归一化占用图：`risk_mining/output_geometric_topology_exposure/area_vs_occupancy.png`
- 面积-多体复杂度图：`risk_mining/output_geometric_topology_exposure/area_vs_interaction_complexity.png`
- 面积-博弈持续时间图：`risk_mining/output_geometric_topology_exposure/area_vs_interaction_duration.png`
- 面积-合法路权重叠图：`risk_mining/output_geometric_topology_exposure/area_vs_right_of_way_overlap.png`
- 相关矩阵：`risk_mining/output_geometric_topology_exposure/geometric_correlation_heatmap.png`

### 结果保存位置

- 目录：`risk_mining/output_geometric_topology_exposure`
- 核心文件：`geometric_exposure_metrics.csv`、`geometric_exposure_correlations.csv`、`methodology_notes.json`
- 详细说明：`Task_docs/GeometricTopologyExposure.md`

### 当前结果概览

| location | Ac m2 | mean residence | P90 residence | N>=3 ratio | P90 game duration | legal overlap cells |
|---|---:|---:|---:|---:|---:|---:|
| cc | 691.912 | 3.942s | 7.200s | 0.440 | 7.580s | 0 |
| tj | 1028.819 | 3.994s | 6.000s | 0.342 | 10.100s | 0 |
| cqIR | 1387.245 | 3.612s | 7.000s | 0.540 | 21.270s | 2 |
| cqNR | 1099.427 | 3.254s | 4.800s | 0.352 | 15.700s | 0 |
| cqR | 2422.354 | 3.275s | 4.900s | 0.423 | 8.900s | 33 |
| xasl | 2593.371 | 6.573s | 10.500s | 0.387 | 9.800s | 0 |

## 12. 快速打开所有 HTML 汇报

```bash
xdg-open risk_mining/output_metric_conclusions/index.html
xdg-open risk_mining/output_kinematic_envelopes_six_all/index.html
xdg-open risk_mining/output_kinematic_wasserstein_domain_gap/index.html
xdg-open risk_mining/output_lateral_deviation_variance/index.html
xdg-open risk_mining/output_intersection_spatiotemporal_density/index.html
xdg-open risk_mining/output_critical_gap_acceptance/index.html
xdg-open risk_mining/output_interaction_topology_complexity/index.html
xdg-open risk_mining/output_conflict_patterns_spatial_clustering/index.html
xdg-open risk_mining/output_structured_violations_noncompliance/index.html
xdg-open risk_mining/output_two_wheeler_heterogeneity/index.html
xdg-open risk_mining/output_right_of_way_overlap/index.html
xdg-open risk_mining/output_geometric_topology_exposure/index.html
```

如果在无 GUI 环境运行，可直接在文件浏览器或 VS Code 中打开上述 `index.html`。
