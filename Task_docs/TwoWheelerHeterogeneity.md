# SinD 两轮车动力学异质性分离

更新时间：2026-05-04  
脚本：`risk_mining/two_wheeler_heterogeneity.py`  
结果目录：`risk_mining/output_two_wheeler_heterogeneity`

## 指标含义

本指标用于分析 SinD 路口中行人、自行车、电动二轮车/电摩、三轮车的动力学差异，重点支撑 “E-bike / powered two-wheeler 在路口中具有更高速度和更强穿插能力” 这一 domain gap 论点。

当前分类口径：

- `pedestrian`：原始标注为行人。
- `bicycle`：原始标注为自行车，且未被速度聚类判为高速两轮车。
- `e_bike`：原始标注为 `motorcycle` 的 powered two-wheeler proxy；同时将部分 `bicycle` 中 p85 速度显著更高的 fast cluster 归入 likely e-bike/scooter。
- `tricycle`：原始标注为三轮车。

重要边界：SinD 原始类别并不直接提供车辆注册意义上的“电动自行车”标签，因此 `e_bike` 应在论文中表述为 `powered two-wheeler / likely e-bike proxy`。速度聚类只作为补充拆分机制，不应过度解释为精确人工标注。

## 计算方法

每条目标轨迹先进入路口核心 ROI 统计；若 ROI 内点数不足，则回退到整条轨迹，避免只在路口边界短暂出现的轨迹被丢弃。默认复用路口驻留热力图中的长期静止目标过滤逻辑，减少路沿长期停车或静止目标对速度分布的污染。

输出的轨迹级指标包括：

- `mean_passage_speed_mps`：路口通行平均速度，violin plot 的主纵轴。
- `cruise_speed_mps`：过滤低速停顿后的巡航速度。
- `p85_speed_mps` / `p95_speed_mps`：速度高分位，用于描述高速穿插能力。
- `max_startup_accel_mps2`：从首次达到移动阈值后的起步窗口中取正加速度高分位，表征最大起步加速能力。
- `max_positive_accel_mps2`：ROI 内正加速度高分位。

## 运行脚本

完整六路口运行：

```bash
conda run -n trajdata python risk_mining/two_wheeler_heterogeneity.py \
  --data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --cities cc tj cqIR cqNR cqR xasl \
  --output-dir risk_mining/output_two_wheeler_heterogeneity
```

常用参数：

- `--moving-threshold-mps 0.5`：巡航速度和起步阶段的移动阈值。
- `--startup-window-s 3.0`：起步加速度统计窗口。
- `--min-bicycle-cluster-tracks 30`：某路口自行车样本低于该数量时不做速度聚类。
- `--min-ebike-speed-mps 3.5`：fast cluster 中心速度至少达到该值才拆为 likely e-bike。
- `--min-cluster-gap-mps 1.0`：fast/slow 两个 cluster 中心至少相差该值才拆分。
- `--disable-static-filter`：关闭长期静止目标过滤；不建议用于正式结果。

## 可视化入口

- 总报告：`risk_mining/output_two_wheeler_heterogeneity/index.html`
- 小提琴图：`risk_mining/output_two_wheeler_heterogeneity/passage_speed_violin.png`

## 结果保存位置

核心 CSV/JSON：

- `two_wheeler_track_metrics.csv`：轨迹级速度、加速度、分类来源。
- `two_wheeler_summary_by_class.csv`：四类交通体的全局统计。
- `two_wheeler_summary_by_location.csv`：六路口分组统计。
- `bicycle_split_report.json`：每个路口的 bicycle 速度聚类中心、是否拆分和快慢簇数量。
- `filtered_static_tracks.csv`：被过滤的长期静止轨迹。
- `roi_debug.csv`：每个路口核心 ROI 推断调试信息。

## 当前结果概览

六路口共得到 16,862 条有效行人/两轮/三轮轨迹。整体平均路口通行速度如下：

| class | tracks | mean passage speed | median passage speed | mean startup accel | max startup accel |
|---|---:|---:|---:|---:|---:|
| pedestrian | 5,680 | 1.176 m/s | 1.167 m/s | 0.357 m/s^2 | 9.581 m/s^2 |
| bicycle | 1,954 | 2.622 m/s | 2.588 m/s | 0.516 m/s^2 | 16.932 m/s^2 |
| e_bike proxy | 8,308 | 5.435 m/s | 5.166 m/s | 0.476 m/s^2 | 15.370 m/s^2 |
| tricycle | 920 | 4.852 m/s | 4.623 m/s | 0.631 m/s^2 | 11.591 m/s^2 |

分类来源：

- 原始 `motorcycle -> e_bike proxy`：7,087 条。
- 原始 `bicycle` 经速度聚类拆为 fast cluster：1,221 条。
- 保留为 `bicycle`：1,954 条。

该结果显示 powered two-wheeler proxy 的路口通行速度约为行人的 4.6 倍、普通自行车的 2.1 倍，适合作为中国路口两轮车高速穿插异质性的定量证据。
