# SinD 违规与非标行为基线

更新时间：2026-04-28  
脚本：`risk_mining/structured_violations_noncompliance.py`  
结果目录：`risk_mining/output_structured_violations_noncompliance`

## 指标含义

本指标实现 C. Violations & Non-compliant Baseline，用来描述 SinD 路口环境中的“规则背景噪声”，但不会在证据不足时过度宣称严格违法。

### 统计指标 5：机动车结构化违规率

当前可计算三类结果：

- `official_red_light_running_label`：SinD pkl 中已有的官方红灯越线标签；目前主要在天津存在。
- `wrong_way_or_opposing_lane_proxy`：车辆持续沿最近 Lanelet2 中心线反方向运动，作为逆行/驶入对向车道/压隔离线的 proxy。
- `intersection_lane_switch_proxy`：车辆在推断出的路口核心 ROI 内发生最近车道 ID 切换，作为路口内违规变道 proxy。

重要边界：

- 当前 `sind_traffic_light_mapping.json` 仍未完成人工灯控车道绑定，所以严格“红灯时越过停止线”的计算被标记为 `red_light_strict_available=False`，不会把缺失绑定误报为 0 违规。
- Lanelet2 地图中部分车道方向与合法行驶方向相反，因此脚本输出 `lane_direction_diagnostics.csv`，并在汇总中同时给出 raw wrong-way proxy 和 map-checked wrong-way proxy。论文中建议优先使用 `wrong_way_proxy_map_checked_rate`，等地图方向返工后再重新跑。
- 目前地图没有实线/双黄线语义，因此“双黄线”和“实线变道”只能作为 proxy，不作为严格法律标签。

### 统计指标 6：VRU 侵入与规则无视底线

当前可计算两类结果：

- `motor_lane_seconds` / `motor_lane_length_ratio`：行人、自行车、摩托车轨迹位于机动车道中心线附近的时间和路径比例。
- `outside_crosswalk_core_seconds_verified`：当 Lanelet2 地图有 crosswalk polygon 时，统计 VRU 位于路口核心 ROI 且不在斑马线内的时间。当前天津有 crosswalk polygon，其它路口多数地图缺少 crosswalk 语义，所以相应结果只作为未验证 proxy。

重要边界：

- 行人/非机动车红灯期间侵入机动车冲突区需要“行人灯/非机动车灯-斑马线/车道”的绑定，目前不做严格红灯违规统计。
- 长时间停在路沿的静态目标会先被 `filter_static_tracks` 过滤，避免把停靠车辆/路侧目标误当作动态违规。

## 运行脚本

完整六路口运行：

```bash
conda run -n trajdata python risk_mining/structured_violations_noncompliance.py \
  --data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_structured_violations_noncompliance
```

快速 smoke test：

```bash
conda run -n trajdata python risk_mining/structured_violations_noncompliance.py \
  --data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --cities tj cqNR \
  --max-tracks-per-city 300 \
  --output-dir /tmp/sind_violation_smoke
```

常用阈值参数：

- `--lane-match-threshold-m 2.0`：车辆匹配机动车道中心线的最大距离。
- `--wrong-way-heading-threshold-deg 120`：车辆速度方向与车道方向夹角超过该阈值时，判为反向 proxy。
- `--min-wrong-way-duration-s 1.0`：反向运动至少持续该时间才形成事件。
- `--vru-motor-lane-threshold-m 1.5`：VRU 距机动车道中心线小于该距离，视为机动车道占用 proxy。
- `--disable-static-filter`：关闭长期静止目标过滤；不建议用于正式结果。

## 可视化入口

- 总报告：`risk_mining/output_structured_violations_noncompliance/index.html`
- 汇总柱状图：`risk_mining/output_structured_violations_noncompliance/summary_violation_rates.png`
- 机动车事件热点：`risk_mining/output_structured_violations_noncompliance/vehicle_violation_hotspot_<location>.png`
- VRU 侵入热点：`risk_mining/output_structured_violations_noncompliance/vru_encroachment_<location>.png`

## 结果保存位置

核心 CSV/JSON：

- `summary_by_location.csv`：六路口汇总指标。
- `summary_by_city.csv`：城市级汇总指标。
- `vehicle_violation_events.csv`：机动车事件级记录。
- `vehicle_violation_track_metrics.csv`：机动车轨迹级指标。
- `vru_noncompliance_events.csv`：VRU 事件级记录。
- `vru_noncompliance_track_metrics.csv`：VRU 轨迹级指标。
- `lane_direction_diagnostics.csv`：Lanelet2 方向诊断，含 `possible_map_direction_error`。
- `filtered_static_tracks.csv`：被过滤的长静止/沿岸停车轨迹。
- `violation_rois.geojson`：路口核心 ROI 与已解析 crosswalk polygon。
- `methodology_notes.json`：方法边界和当前灯控绑定状态。

## 当前完整运行结果概览

| location | vehicle tracks | wrong-way checked | wrong-way raw | lane-switch proxy | official red label | VRU tracks | VRU encroachment | crosswalks | map-dir warning lanes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cc | 7,206 | 4.82% | 82.20% | 21.20% | 0.00% | 719 | 91.10% | 0 | 7 |
| tj | 5,271 | 13.05% | 23.15% | 1.71% | 2.68% | 7,508 | 94.98% | 4 | 2 |
| cqIR | 3,548 | 4.90% | 29.48% | 6.06% | 0.00% | 2,038 | 92.49% | 0 | 10 |
| cqNR | 2,004 | 11.93% | 71.21% | 6.74% | 0.00% | 1,571 | 84.34% | 0 | 9 |
| cqR | 9,584 | 6.83% | 45.33% | 3.57% | 0.00% | 3,144 | 97.07% | 0 | 10 |
| xasl | 5,170 | 6.23% | 82.55% | 24.51% | 0.00% | 1,279 | 95.07% | 0 | 16 |

解释：`wrong-way raw` 明显受部分 Lanelet2 车道方向错误影响；`wrong-way checked` 已排除被诊断为方向异常的车道，更适合作为临时论文数字。地图返工后建议重新运行并替换该表。
