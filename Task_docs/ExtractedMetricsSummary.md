# SinD 已提取指标汇总

更新时间：2026-04-28  
项目路径：`/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata`

本文档汇总当前已经实现并跑通的 SinD 六路口指标，包含每个指标的含义、运行脚本、可视化入口和结果保存位置。所有命令默认在项目根目录执行，并使用 `trajdata` conda 环境。

## 0. 六个路口与城市合并规则

- 六个 SinD 路口：`cc`、`tj`、`cqIR`、`cqNR`、`cqR`、`xasl`
- 展示名：`ChangChun`、`TianJin`、`ChongQing-IR`、`ChongQing-NR`、`ChongQing-R`、`XiAn-Shanglin`
- 四城市合并：`cc -> ChangChun`，`tj -> TianJin`，`cqIR/cqNR/cqR -> ChongQing`，`xasl -> XiAn`

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

## 8. 快速打开所有 HTML 汇报

```bash
xdg-open risk_mining/output_kinematic_envelopes_six_all/index.html
xdg-open risk_mining/output_kinematic_wasserstein_domain_gap/index.html
xdg-open risk_mining/output_lateral_deviation_variance/index.html
xdg-open risk_mining/output_intersection_spatiotemporal_density/index.html
xdg-open risk_mining/output_critical_gap_acceptance/index.html
xdg-open risk_mining/output_interaction_topology_complexity/index.html
xdg-open risk_mining/output_conflict_patterns_spatial_clustering/index.html
```

如果在无 GUI 环境运行，可直接在文件浏览器或 VS Code 中打开上述 `index.html`。
