# SinD 时空路权重叠度

更新时间：2026-05-04  
脚本：`risk_mining/right_of_way_overlap.py`  
结果目录：`risk_mining/output_right_of_way_overlap`

## 指标含义

本指标实现 “Spatio-Temporal Right-of-Way Overlap”，用于解释为什么在没有明显违规的情况下，路口内仍会出现强交互和冲突潜势。

核心问题是：在同一绿灯相位期间，合法释放的机动车通行轨迹簇，与合法释放的行人/自行车过街轨迹簇，是否在空间上发生交叉或近距离重叠。交点越多，说明规划和预测算法需要处理的 interaction prediction 难度越高。

重要边界：

- 这里统计的是“合法绿灯释放下的几何冲突潜势”，不是违规，也不是事故风险。
- 主口径要求双方信号都可观测且为 `GREEN`。缺少车道/斑马线灯控绑定的轨迹进入 diagnostics，不混入分母。
- `overlap_events` 是合法机动车轨迹与合法 VRU 过街轨迹的 pair-level 空间重叠数。
- `unique_spatial_overlap_cells_1m` 是将重叠点落到 `1m x 1m` 网格后的唯一空间交点数量，更接近“空间热点数量”。
- `simultaneous_overlap_events` 额外要求双方到达重叠点的时间差小于阈值，表示真实时刻上更接近的交互。当前结果中该项为 0，说明本指标主要揭示同一绿灯相位释放造成的路径层面重叠，而非同秒到达。

## 计算方法

1. 读取 SinD 六路口轨迹、Lanelet2 地图、crosswalk polygon、人工交通灯绑定和本地交通灯 pkl。
2. 对机动车轨迹，取进入路口核心 ROI 的时刻，查询最近绑定 lane 的信号状态；若为 `GREEN`，记录为合法机动车通行 passage。
3. 对行人/自行车轨迹，检测其通过的 crosswalk polygon，查询该 crosswalk 绑定信号；若进入 crosswalk 时为 `GREEN`，记录为合法 VRU crossing passage。
4. 在同一 scene 内，对合法机动车 passage 和合法 VRU passage 成对比较：
   - 两者 green interval 必须有重叠，默认至少 `1s`。
   - 两条 passage polyline 的最近空间距离必须小于 `2m`。
   - 记录空间交点/近交点、所属 `1m` 网格、绿灯重叠时长和双方到达该点的时间差。

## 运行脚本

完整六路口运行：

```bash
conda run -n trajdata python risk_mining/right_of_way_overlap.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_right_of_way_overlap
```

常用参数：

- `--spatial-threshold-m 2.0`：机动车轨迹与 VRU 过街轨迹最近距离小于该值才认为存在空间重叠。
- `--min-phase-overlap-s 1.0`：双方绿灯相位重叠的最小时长。
- `--simultaneous-time-gap-s 5.0`：双方到达重叠点时间差小于该值时，额外标记为 simultaneous overlap。
- `--min-crosswalk-path-m 2.0`：VRU 在 crosswalk 内至少通过该长度才算有效过街。
- `--include-motorcycle-as-vehicle`：将 `motorcycle` 也纳入机动车侧；默认不启用。

## 可视化入口

- 总报告：`risk_mining/output_right_of_way_overlap/index.html`
- 汇总柱状图：`risk_mining/output_right_of_way_overlap/right_of_way_overlap_summary.png`
- 单路口热点图：`risk_mining/output_right_of_way_overlap/overlap_hotspot_<location>.png`

## 结果保存位置

核心 CSV/JSON：

- `legal_green_passages.csv`：合法绿灯机动车 passage 和合法绿灯 VRU crossing。
- `right_of_way_overlap_events.csv`：pair-level 时空路权重叠事件。
- `right_of_way_overlap_summary_by_location.csv`：路口级汇总。
- `right_of_way_overlap_cluster_summary.csv`：按机动车控制 lane 与 VRU crosswalk 控制关系聚合的重叠簇。
- `passage_observability_diagnostics.csv`：不可观测、非绿灯、无 ROI passage、无 crosswalk passage 等诊断。
- `signal_table_reports.csv`：交通灯 pkl/CSV 读取报告。
- `filtered_static_tracks.csv`：长期静止目标过滤记录。
- `roi_debug.csv`：路口核心 ROI 推断信息。
- `methodology_notes.json`：方法边界和参数记录。

## 当前结果概览

| location | legal vehicle | legal VRU crossing | overlap events | unique 1m cells | median abs arrival gap |
|---|---:|---:|---:|---:|---:|
| cc | 5,258 | 0 | 0 | 0 | 0.000s |
| tj | 3,919 | 0 | 0 | 0 | 0.000s |
| cqIR | 858 | 288 | 3 | 2 | 52.753s |
| cqNR | 1,325 | 387 | 0 | 0 | 0.000s |
| cqR | 1,380 | 251 | 215 | 33 | 96.196s |
| xasl | 3,372 | 0 | 0 | 0 | 0.000s |

当前可解释结论：

- `cqR` 存在最明显的合法绿灯相位路径重叠，215 个 pair-level overlap，落在 33 个唯一 `1m` 空间网格中。
- `cqIR` 有少量合法路径重叠，3 个 pair-level overlap。
- `cqNR` 虽然有合法机动车和合法 VRU crossing，但在当前阈值下没有空间重叠。
- `cc/tj/xasl` 的 VRU 合法绿灯过街为 0，主要是当前人工绑定中缺少 pedestrian/crosswalk 信号可观测关系；这不是“无行人过街冲突”的结论，而是“该指标在这些路口的 VRU 侧暂不可观测”。
