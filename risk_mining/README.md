# Risk Mining Framework

这份说明面向当前 `risk_mining` 的实际实现，重点是帮助你快速理解：

- 现在这套框架怎么跑
- 规则是在哪一层生效的
- 如果你要新增/修改规则，应该改哪里
- 输出结果和可视化文件怎么看

## 1. 当前框架概览

当前流程是：

`trajdata Scene -> Slicer -> Episode -> RuleRegistry.apply_all() -> SSTG -> SanityChecker -> DualLibrary / manual_review`

对应入口文件是 [run_pipeline.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/run_pipeline.py)。

目前这套实现是一个“可扩展的骨架版本”：

- `Slicer` 先从真实场景中切出候选 episode
- 规则系统负责往图里加节点和边
- `SanityChecker` 检查图里是否存在可解释的 `causal` 边
- 通过检查的 episode 存入 JSON 事件库
- 不通过的 episode 存入 `manual_review/`

## 2. 目录结构

```text
risk_mining/
├── config/
│   └── rules_config.yaml
├── src/
│   ├── core/
│   │   ├── scene_graph.py
│   │   └── slicer.py
│   ├── rules/
│   │   ├── base_rule.py
│   │   ├── spatial_rules.py
│   │   └── dynamic_rules.py
│   ├── library/
│   │   └── dual_repo.py
│   └── utils/
│       ├── checker.py
│       └── logger.py
├── run_pipeline.py
├── extract_episodes.py
└── visualize_episode.py
```

## 3. 核心数据对象

### 3.1 SSTG

文件：[scene_graph.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/core/scene_graph.py)

`SSTG` 是一个 `networkx.DiGraph` 包装器，当前统一使用三类边：

- `spatial`
- `temporal`
- `causal`

节点 `Node` 当前的关键字段：

- `agent_id`
- `timestamp`
- `type`
- `velocity`
- `acceleration`
- `position`
- `heading`
- `extent`
- `metadata`

其中 `timestamp` 目前不是全时间序列，而是三个语义时间点：

- `T_start`
- `T_peak`
- `T_end`

边 `Edge` 当前的关键字段：

- `source_id`
- `target_id`
- `source_timestamp`
- `target_timestamp`
- `edge_type`
- `weight`
- `relation`
- `metadata`

### 3.2 Episode

文件：[slicer.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/core/slicer.py)

`Episode` 是规则系统的输入对象。它包含：

- 场景元信息：`scene_id`, `scene_name`, `env_name`
- 时间窗：`t_start`, `t_peak`, `t_end`
- 自车：`ego_agent_id`
- 参与体：`involved_agents`
- 三个语义时刻的状态快照：`state_snapshots`
- 当前风险分数：`risk_score`
- 规则执行轨迹：`rule_trace`
- 最终图：`sstg`

## 4. 当前提取逻辑

### 4.1 Slicer 在做什么

`Slicer.extract_episodes()` 当前是一个 dummy 版本：

1. 在 scene 内选择 ego
2. 枚举 ego 与其它 agent 的 TTC
3. 取最小 TTC 对应时刻作为 `T_peak`
4. 用固定窗口得到：
   - `T_start = T_peak - 2s`
   - `T_end = T_peak + 2s`
5. 从这三个时刻抓状态，构造 `Episode`

### 4.2 ego 的选择策略

当前逻辑是：

1. 优先找名字为 `ego` 的 agent
2. 如果没有，就取第一个 `VEHICLE`
3. 再不行就取第一个 agent

这个逻辑在某些真实数据集上可能并不理想。如果后续发现“自车定义不合理”，优先修改这里。

## 5. 规则系统

### 5.1 当前设计

文件：[base_rule.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/base_rule.py)

规则系统是 Strategy Pattern。

每条规则都继承 `BaseRule`，并实现：

```python
def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
    ...
```

规则只能做两件事：

- 读取 `episode`
- 修改并返回 `current_graph`

主流程不会关心某条规则内部怎么计算，只按注册顺序执行。

### 5.2 当前已有规则

#### SpatialROIRule

文件：[spatial_rules.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/spatial_rules.py)

当前行为：

- 在 `T_start / T_peak / T_end` 取 ego 周围 `roi_radius` 内的 agent
- 把这些 agent 加入图
- 加 `spatial` 边：`relation="within_roi"`
- 给同一 agent 的不同语义时刻之间加 `temporal` 边：`relation="state_transition"`

可调参数：

- `roi_radius`

#### TTCCriticalRule

文件：[dynamic_rules.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/dynamic_rules.py)

当前行为：

- 只在 `T_peak` 检查 ego 与其它 agent
- 若 `ttc < ttc_threshold`
- 加 `causal` 边：`relation="has_collision_risk"`

可调参数：

- `ttc_threshold`

### 5.3 怎样新增规则

最简单的做法：

1. 在 `src/rules/` 下新建文件，例如 `custom_rules.py`
2. 继承 `BaseRule`
3. 实现 `apply()`
4. 在 [rules/__init__.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/__init__.py) 中导出
5. 在 [run_pipeline.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/run_pipeline.py) 的 `setup_pipeline()` 里注册

最小示例：

```python
from .base_rule import BaseRule
from ..core.scene_graph import Edge, EdgeType, SSTG
from ..core.slicer import Episode


class MyRule(BaseRule):
    def __init__(self, threshold: float = 1.0, enabled: bool = True):
        super().__init__(name="my_rule", enabled=enabled)
        self.threshold = threshold

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        # 1. 从 episode 读取节点
        # 2. 判断条件
        # 3. 往图里加节点/边
        return current_graph
```

### 5.4 修改规则时优先关注什么

如果你要调整“提取结果是否合理”，通常优先看这三层：

1. `Slicer`
原因：很多“抽到的片段不对”其实是 `T_peak` 不对，或者 ego 不对

2. `SpatialROIRule`
原因：很多“图里 agent 太多 / 太少”是 ROI 过滤不合适

3. `TTCCriticalRule`
原因：很多“因果边不合理 / 没解释”是 TTC 判定太松或太紧

## 6. 配置文件

文件：[rules_config.yaml](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/config/rules_config.yaml)

当前可配项主要有：

- 数据源
  - `desired_data`
  - `data_dirs`
  - `desired_dt`
- 规则
  - `rules.spatial.in_roi.enabled`
  - `rules.spatial.in_roi.roi_radius`
  - `rules.dynamic.ttc.enabled`
  - `rules.dynamic.ttc.ttc_threshold`
- slicer
  - `pre_buffer_sec`
  - `post_buffer_sec`
- 输出
  - `output_dir`

## 7. 输出结果

### 7.1 正常通过的结果

会写到：

- `libraries/risk_events/`
- `libraries/risk_elements/`

其中：

- `risk_events` 是完整 JSON 事件
- `risk_elements` 是轻量摘要

### 7.2 未通过检查的结果

会写到：

- `manual_review/`

### 7.3 shortlist 场景筛选

如果像这次一样做人工审查样本挑选，会额外生成：

- `top25_shortlist.json`

这个文件最适合先人工浏览，因为里面按风险分数排好序了。

## 8. Kinematic Envelopes

新增脚本 [kinematic_envelopes.py](./kinematic_envelopes.py) 用于从 SinD 轨迹中提取语义约束下的 `v-a` 运动学包络。

它会：

- 按 `City / AgentType / Maneuver` 过滤轨迹
- 对每条轨迹使用 Savitzky-Golay 平滑后计算速度和纵向加速度
- 用 2D-KDE 提取 95% 包络边界
- 导出包络外样本、轨迹级异常汇总、城市间重叠比和对比图

示例：

```bash
conda run -n trajdata python risk_mining/kinematic_envelopes.py \
  --cities xa cc \
  --agent-type Car \
  --maneuver Straight \
  --output-dir risk_mining/output_kinematic_envelopes
```

输出文件包括：

- `envelope_<city>.png`
- `cross_city_envelopes.png`
- `outliers.csv`
- `outlier_tracks.csv`
- `overlap_ratios.csv`
- `summary.json`

## 9. Kinematic Wasserstein Domain Gap

新增脚本 [kinematic_wasserstein_domain_gap.py](./kinematic_wasserstein_domain_gap.py) 用于用 Wasserstein / Earth Mover's Distance 严格量化城市间 `v-a` 运动学分布差异。

它会：

- 复用 Kinematic Envelopes 的平滑逻辑，提取语义切片下的 `(v, a)` 样本
- 默认使用 `All / All`，即所有交通体、所有意图，论证总体城市 domain gap
- 计算 robust-normalized 二维联合 Wasserstein 距离，作为主指标
- 同时计算一维 `W(v)` 和 `W(a)`，解释差异来自速度还是加速度
- 输出 `4x4` 城市合并矩阵和 `6x6` SinD 路口矩阵

示例：

```bash
conda run -n trajdata python risk_mining/kinematic_wasserstein_domain_gap.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --agent-type All \
  --maneuver All \
  --output-dir risk_mining/output_kinematic_wasserstein_domain_gap
```

输出文件包括：

- `index.html`
- `wasserstein_2d_city_matrix.csv`
- `wasserstein_2d_location_matrix.csv`
- `wasserstein_1d_city_pairs.csv`
- `wasserstein_1d_location_pairs.csv`
- `wasserstein_city_pairs_long.csv`
- `wasserstein_location_pairs_long.csv`
- `domain_sample_summary.csv`
- `wasserstein_heatmap_city_2d.png`
- `wasserstein_heatmap_location_2d.png`
- `wasserstein_heatmap_city_velocity.png`
- `wasserstein_heatmap_city_acceleration.png`
- `summary.json`

## 10. Lateral Deviation Variance

新增脚本 [lateral_deviation_variance.py](./lateral_deviation_variance.py) 用于统计车辆转弯轨迹相对 Lanelet2 参考中心线的横向偏移方差。

它会：

- 读取 SinD 六个路口的 Lanelet2 OSM 中心线作为 reference line
- 筛选 vehicle 类左/右转轨迹
- 将轨迹点投影到最匹配的中心线，计算 `mean lateral offset` 和 `sigma_lat^2`
- 按 `location + maneuver + reference_lane_id` 聚合轨迹簇指标
- 绘制 trajectory bundles：低透明度轨迹簇、reference centerline、均值轨迹和 `±2σ` 阴影带

示例：

```bash
conda run -n trajdata python risk_mining/lateral_deviation_variance.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --agent-type Vehicle \
  --maneuvers left-turn right-turn \
  --output-dir risk_mining/output_lateral_deviation_variance
```

输出文件包括：

- `index.html`
- `track_lateral_deviation.csv`
- `cluster_lateral_deviation.csv`
- `unmatched_tracks.csv`
- `bundle_manifest.csv`
- `bundle_*.png`
- `summary.json`

## 11. Intersection Spatiotemporal Density

新增脚本 [intersection_spatiotemporal_density.py](./intersection_spatiotemporal_density.py) 用于生成路口内部车辆驻留热力图。

它会：

- 将车辆中心点按 `1m x 1m` 网格累计驻留时间
- 使用 Lanelet2 内部 connector 中心线推断更紧的路口内部 ROI，排除停止线前后的排队区域
- 默认滤除长时间几乎不移动的静态/沿岸停车轨迹，可用 `--keep-static-tracks` 关闭
- 只统计 vehicle 类交通体
- 输出每个路口的 Occupancy Heatmap、热点网格和原始 occupancy grid

示例：

```bash
conda run -n trajdata python risk_mining/intersection_spatiotemporal_density.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_intersection_spatiotemporal_density
```

输出文件包括：

- `index.html`
- `occupancy_heatmap_<city>.png`
- `occupancy_grid_<city>.npz`
- `core_roi_<city>.geojson`
- `density_summary.csv`
- `hotspot_cells.csv`
- `track_roi_residence.csv`
- `filtered_static_tracks.csv`
- `summary.json`

## 12. Critical Gap Acceptance

新增脚本 [critical_gap_acceptance.py](./critical_gap_acceptance.py) 用于统计 SinD 六个城市路口中无保护左转（UPLT）的可接受间隙阈值。

当前实现采用几何近似：

- Ego：同场景内 maneuver 为 `left-turn` 的 vehicle
- 冲突车：同场景内 maneuver 为 `straight` 的 vehicle
- 决策时刻：Ego 首次进入 Lanelet2 推断的路口核心 ROI
- 冲突点：Ego 与直行车辆在决策时刻之后短时间窗口内轨迹距离最近且小于阈值的位置
- `time_gap_s`：直行冲突车到达冲突点时间减去 Ego 到达冲突点时间，只保留 `0 < gap <= max_time_gap_s`
- `distance_gap_m`：决策时刻后，直行冲突车沿自身轨迹到冲突点的剩余距离

为避免全量轨迹两两匹配过慢，脚本会先按进入时刻距离、航向差、车辆速度和短时轨迹包围盒筛选候选冲突车。默认参数偏向捕获中国路口中较激进的左转抢行样本：

- `--opposing-heading-min-deg 60`
- `--conflict-distance-m 3.0`
- `--max-time-gap-s 8.0`
- `--pair-initial-distance-m 90.0`
- `--max-conflict-candidates-per-ego 24`

示例：

```bash
conda run -n trajdata python risk_mining/critical_gap_acceptance.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_critical_gap_acceptance
```

输出文件包括：

- `index.html`
- `critical_gap_summary_by_city.csv`
- `critical_gap_accepted.csv`
- `critical_gap_pairs.csv`
- `critical_gap_skipped.csv`
- `low_gap_cases.csv`
- `critical_gap_hist_<city>.png`
- `critical_gap_cdf_<city>.png`
- `critical_gap_city_comparison.png`
- `summary.json`

## 13. Interaction Topology & Game Complexity

新增脚本 [interaction_topology_complexity.py](./interaction_topology_complexity.py) 用于统计路口交互拓扑复杂度，而不是危险性。

它会：

- 在 Lanelet2 推断的路口核心 ROI + buffer 内构建逐帧交互图
- 若两个交通体同帧距离小于 `15m`，且未来短时轨迹存在潜在交点，则建立一条 loose interaction edge
- 统计交互图连通分量大小 `N`，输出 `2-party / 3-party / 4+-party` 阶数分布
- 将跨帧连通分量按 agent 集合重叠合并成 interaction episode，统计博弈持续时长
- 同时输出六路口版和四城市合并版；参与者包含机动车、非机动车和行人
- 默认滤除长期静止/沿岸停车目标，避免虚假高阶交互

示例：

```bash
conda run -n trajdata python risk_mining/interaction_topology_complexity.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_interaction_topology_complexity
```

输出文件包括：

- `index.html`
- `interaction_degree_summary_by_location.csv`
- `interaction_degree_summary_by_city.csv`
- `interaction_frame_components.csv`
- `interaction_duration_episodes.csv`
- `interaction_duration_summary.csv`
- `top_complex_episodes.csv`
- `stacked_degree_by_location.png`
- `stacked_degree_by_city.png`
- `duration_distribution_by_location.png`
- `duration_boxplot_by_city.png`
- `interaction_rois.geojson`
- `filtered_static_tracks.csv`
- `summary.json`

## 14. Conflict Patterns & Spatial Clustering

新增脚本 [conflict_patterns_spatial_clustering.py](./conflict_patterns_spatial_clustering.py) 用于统计路口冲突类型比例和显著减速避让热点。

它会：

- 在 Lanelet2 推断的路口核心 ROI + buffer 内提取 loose conflict pair
- pair 条件为同帧距离 `<=15m` 且未来 `5s` 轨迹最近距离 `<=3m`
- 基于全局意图将车辆/两轮车 pair 归类为 `Crossing / Merging / Weaving / Turning / Other`
- 合并同一 scene-agent pair 的连续冲突帧，默认以 pair episode 统计冲突类型比例
- 提取 `a_lon < -1.5m/s^2` 且前方存在潜在冲突参与者的显著减速避让点
- 生成 maneuver chord diagram 和叠加高精地图的空间热点图

示例：

```bash
conda run -n trajdata python risk_mining/conflict_patterns_spatial_clustering.py \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_conflict_patterns_spatial_clustering
```

输出文件包括：

- `index.html`
- `conflict_pair_events.csv`
- `conflict_pair_episodes.csv`
- `conflict_type_summary_by_location.csv`
- `conflict_type_summary_by_city.csv`
- `deceleration_hotspot_events.csv`
- `deceleration_hotspot_cells.csv`
- `conflict_chord_<city>.png`
- `conflict_chord_city_groups.png`
- `conflict_hotspot_<city>.png`
- `conflict_hotspot_grid_<city>.npz`
- `conflict_rois.geojson`
- `filtered_static_tracks.csv`
- `summary.json`

## 15. Sanity Check 的当前规则

文件：[checker.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/utils/checker.py)

当前检查非常简单：

- 图中至少要有一条 `causal` 边

如果没有，就认为这个 episode 缺少“可解释的交互原因”。

这意味着当前框架其实偏向保守：

- 只要 TTC 因果边能加出来，就容易通过
- 如果未来你加了别的高价值规则，比如：
  - 急刹车
  - 让行
  - 冲突车道
  - 异常横穿
  
  那这里也应该同步升级，不然 checker 会过度依赖 TTC

## 16. 可视化

文件：[visualize_episode.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/visualize_episode.py)

当前可视化支持：

- Bokeh 交互 HTML
- 事件切换
- `T_start / T_peak / T_end` 切换
- 地图层显示
- agent box + heading marker
- `spatial / temporal / causal` 边层切换

如果你在人工检查时觉得“提取逻辑不对”，推荐工作流是：

1. 先看 `top25_shortlist.json`
2. 再打开交互 HTML 看具体事件
3. 记录问题是来自：
   - slicer
   - ego 选择
   - ROI 过滤
   - TTC 规则
   - sanity check

## 17. 当前实现的已知限制

这套框架现在是可用的 v1，但不是最终版。当前限制包括：

- 一个 scene 目前通常只产 1 个最关键 episode
- `T_peak` 只依赖 TTC，不包含更复杂的交互触发条件
- `causal` 边目前只有 TTC 一种来源
- 规则注册还是在 `run_pipeline.py` 里显式写的，还没完全配置化
- 不同城市/路口的“ego 定义”可能还不够稳定

## 18. 推荐的下一步改造顺序

如果你接下来要继续完善规则，建议按下面顺序改：

1. 先扩展 `TTCCriticalRule`
例如更严格的接近方向判断、横向冲突判断、最小时距过滤

2. 再新增新的 `causal` 规则
例如：
   - `DecelerationCausalRule`
   - `ConflictLaneRule`
   - `YieldingRule`
   - `PedestrianCrossingRule`

3. 最后再升级 `SanityChecker`
让它不只认 TTC，而是认“任意可信 causal edge”

如果你愿意，我下一步可以继续帮你做两件事里的任意一个：

- 把这份 README 再扩成“规则开发手册”
- 直接帮你加一条新规则模板，例如 `DecelerationRule` 或 `ConflictLaneRule`
