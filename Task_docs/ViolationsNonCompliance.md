# SinD 违规与非标行为基线

更新时间：2026-05-02  
脚本：`risk_mining/structured_violations_noncompliance.py`  
结果目录：`risk_mining/output_structured_violations_noncompliance`

## 指标含义

本指标实现 C. Violations & Non-compliant Baseline，用来描述 SinD 路口环境中的“规则背景噪声”，但不会在证据不足时过度宣称严格违法。

### 统计指标 5：机动车结构化违规率

当前可计算结果：

- `red_light_entry_violation`：绑定车道信号为红灯时，车辆进入推断出的路口核心 ROI。
- `yellow_light_entry_event`：绑定车道信号为黄灯时，车辆进入路口核心 ROI。黄灯单独统计，不混入红灯。
- `wrong_way_violation_candidate`：车辆/自行车/摩托车在路口核心 ROI 外的进入/驶出路段，持续沿最近 Lanelet2 中心线反方向运动。路口中央不做最近车道逆行判断，避免交错 connector 误匹配。
- `solid_line_lane_change_proxy`：停止线前/路口入口前区域按实线区处理，若发生最近车道 ID 切换则记为候选。
- `lane_direction_rule_violation`：先取车辆进入路口前稳定匹配的入口车道，推断该车道后续允许的动作集合；再用车辆驶入/驶出路口的整体几何方向判断实际动作，并与 allowed movements 对比。
- `right_of_way_yield_violation_candidate`：按 `行人 > 直行 > 左转 > 右转 > 掉头` 的路权顺序，提取低路权参与者抢先到达共享冲突点的候选事件。两者必须来自不同来向，且 ROI 内轨迹存在近距离交汇点；同一区域同向驶入的 pair 不计入。
- `official_red_light_running_label` / `official_yellow_light_running_label`：SinD pkl 中已有的官方信号违规标签，保留用于对照。

重要边界：

- 信号灯默认读取 `datasets/SinD_dataset/<location>/traffic_lights_<location>.pkl`，找不到时 fallback 到原始 CSV。
- 严格红/黄灯入口只在车道-信号绑定可观测时计入分母，汇总中使用 `*_rate_observable`，避免把缺失绑定误报为 0 违规。
- Lanelet2 地图方向已按返工后的 osm 使用。脚本仍输出 `lane_direction_diagnostics.csv` 供检查异常 lane。
- 当前地图没有显式实线/双黄线语义，因此入口实线区变道和双黄线相关行为仍标记为 proxy/candidate。
- `intersection_lane_switch_proxy` 已从正式输出指标中移除，因为路口核心内部 connector 交错较多，最近 lanelet 匹配不稳定，误检风险高。
- 路权项是行为挖掘候选，不做法律责任判定。

### 统计指标 6：VRU 侵入与规则无视底线

当前可计算两类结果：

- `vru_motor_lane_or_conflict_zone_encroachment`：仅统计行人和自行车在 crosswalk polygon 外占用机动车道附近，或位于路口核心 ROI 且不在斑马线内。正常走斑马线不计入，摩托车不参与该 VRU 指标。
- `outside_crosswalk_core_seconds_verified`：当 Lanelet2 地图有 crosswalk polygon 时，统计 VRU 位于路口核心 ROI 且不在斑马线内的时间。当前天津有 crosswalk polygon，其它路口多数地图缺少 crosswalk 语义，所以相应结果只作为未验证 proxy。
- `vru_red_conflict_zone_noncompliance`：绑定人行灯/斑马线信号为红灯时，行人/自行车位于 curbstone/core ROI 内且不在 crosswalk polygon 内。摩托车不参与该指标。
- `vru_yellow_conflict_zone_event`：同上，但信号为黄灯，单独统计。

重要边界：

- 行人/自行车红黄灯冲突区统计只在存在 pedestrian/crosswalk 绑定时作为可观测结果；无绑定区域只保留 proxy。
- 长时间停在路沿的静态目标会先被 `filter_static_tracks` 过滤，避免把停靠车辆/路侧目标误当作动态违规。

## 运行脚本

完整六路口运行：

```bash
conda run -n trajdata python risk_mining/structured_violations_noncompliance.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_structured_violations_noncompliance
```

快速 smoke test：

```bash
conda run -n trajdata python risk_mining/structured_violations_noncompliance.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cqIR cqR \
  --max-tracks-per-city 300 \
  --output-dir /tmp/sind_violation_smoke
```

常用阈值参数：

- `--lane-match-threshold-m 2.0`：车辆匹配机动车道中心线的最大距离。
- `--wrong-way-heading-threshold-deg 120`：车辆速度方向与车道方向夹角超过该阈值时，判为反向 proxy。
- `--min-wrong-way-duration-s 1.0`：反向运动至少持续该时间才形成事件。
- `--vru-motor-lane-threshold-m 1.5`：VRU 距机动车道中心线小于该距离，视为机动车道占用 proxy。
- `--disable-static-filter`：关闭长期静止目标过滤；不建议用于正式结果。
- `--examples-per-event-type 5`：每类违规输出的实例图数量。
- `--example-window-s 3.0`：实例图中事件前后轨迹窗口。
- `--right-of-way-conflict-distance-m 10.0`：路权候选冲突距离阈值。
- `--right-of-way-path-conflict-distance-m 4.0`：两条 ROI 内轨迹最近点小于该距离才认为共享冲突区域。
- `--right-of-way-min-approach-angle-deg 45.0`：两者进入方向夹角小于该阈值时视为同向/同来向，不做路权判断。

## 可视化入口

- 总报告：`risk_mining/output_structured_violations_noncompliance/index.html`
- 汇总柱状图：`risk_mining/output_structured_violations_noncompliance/summary_violation_rates.png`
- 机动车事件热点：`risk_mining/output_structured_violations_noncompliance/vehicle_violation_hotspot_<location>.png`
- VRU 侵入热点：`risk_mining/output_structured_violations_noncompliance/vru_encroachment_<location>.png`
- 违规实例图：`risk_mining/output_structured_violations_noncompliance/examples/<event_type>/*.png`
- 实例清单：`risk_mining/output_structured_violations_noncompliance/violation_example_manifest.csv`

## 结果保存位置

核心 CSV/JSON：

- `summary_by_location.csv`：六路口汇总指标。
- `summary_by_city.csv`：城市级汇总指标。
- `vehicle_violation_events.csv`：机动车事件级记录。
- `vehicle_violation_track_metrics.csv`：机动车轨迹级指标。
- `vru_noncompliance_events.csv`：VRU 事件级记录。
- `vru_noncompliance_track_metrics.csv`：VRU 轨迹级指标。
- `signal_observability_by_location.csv`：红/黄灯入口和 VRU 信号可观测性。
- `right_of_way_yield_events.csv`：路权不让行候选事件。
- `wrong_way_events_by_agent_type.csv`：按交通体类型聚合的逆行候选率。
- `violation_example_manifest.csv`：每类违规实例图索引。
- `lane_direction_diagnostics.csv`：Lanelet2 方向诊断，含 `possible_map_direction_error`。
- `filtered_static_tracks.csv`：被过滤的长静止/沿岸停车轨迹。
- `violation_rois.geojson`：路口核心 ROI 与已解析 crosswalk polygon。
- `methodology_notes.json`：方法边界和当前灯控绑定状态。

## 当前结果说明

脚本已支持完整六路口运行，但红绿灯绑定和 Lanelet2 地图刚完成返工，正式论文数字建议以重新完整运行后的 `summary_by_location.csv` 为准。小规模 smoke test 已验证 pkl 信号表、红/黄灯分离、VRU 红灯冲突区、路权候选和实例图片输出链路可用。
