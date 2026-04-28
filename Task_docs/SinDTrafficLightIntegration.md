# SinD Traffic Light Integration

本文档说明本项目对 SinD 交通信号灯数据的可选补充接入。该功能不会改变已有 pkl 轨迹、地图缓存和指标分析脚本；只有在显式设置外部 CSV 根目录时才会写入 trajdata 的 traffic-light cache。

## 数据来源与范围

- 信号灯 CSV 根目录：`/home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv`
- 启用方式：

```bash
export SIND_TRAFFIC_LIGHT_DIR="/home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv"
```

- 默认支持：`cc`、`xa`、`xasl`、`tj`、`cqNR`、`cqIR`、`cqR`
- 原始文件夹映射：
  - `cc -> Changchun_Pudong`
  - `xa/xasl -> Xi'an_shanglin`
  - `tj -> tianjin`
  - `cqNR -> Chongqing_NR`
  - `cqIR -> Chongqing_IR`
  - `cqR -> Chongqing_R`

## 接口含义

新增模块：`src/trajdata/dataset_specific/sind/sind_traffic_lights.py`

- 读取每个 scene 文件夹中的 `TrafficLight*.csv` 或 `Traffic_Lights.csv`
- 将 CSV 中的相位变化点扩展为逐 `scene_ts` 的状态表
- 输出 trajdata 标准格式：

```text
MultiIndex: lane_id, scene_ts
column: status
```

- 状态映射：
  - `0 -> TrafficLightStatus.RED`
  - `1 -> TrafficLightStatus.GREEN`
  - `3 -> TrafficLightStatus.YELLOW`
  - 其他值 -> `TrafficLightStatus.UNKNOWN`

当前 trajdata 的信号灯状态绑定在 `lane_id` 上，但 SinD CSV 只提供灯号，例如 `Vehicle Traffic light 1`。因此本轮实现采用“安全可用”的两级策略：

- 若 `sind_traffic_light_mapping.json` 中配置了真实 `light_id -> lane_id`，则写入真实 lane id。
- 若没有配置，则写入 synthetic id，例如 `sind_tl:cqNR:vehicle:1`，确保缓存可读取且不做错误车道绑定。

## 缓存入口

文件：`src/trajdata/dataset_specific/sind/sind_dataset.py`

在 `SindDataset.get_agent_info()` 完成 agent cache 后，会尝试执行可选信号灯缓存：

- 未设置 `SIND_TRAFFIC_LIGHT_DIR`：完全跳过，行为与旧版本一致。
- 找不到某个 scene 的 CSV：跳过该 scene 的信号灯，不影响轨迹缓存。
- 解析成功：调用 `cache_class.save_traffic_light_data()`，生成 trajdata 标准 traffic-light feather 文件。

典型缓存位置：

```text
~/.unified_data_cache/sind/<scene_name>/tls_data_dt0.10.feather
```

## 诊断脚本

脚本：`scripts/sind_audit_traffic_lights.py`

只检查原始 CSV 覆盖情况：

```bash
python scripts/sind_audit_traffic_lights.py \
  --traffic-light-dir "$SIND_TRAFFIC_LIGHT_DIR"
```

结合 pkl 数据检查 scene 对齐、状态分布和逐帧表生成：

```bash
python scripts/sind_audit_traffic_lights.py \
  --traffic-light-dir "$SIND_TRAFFIC_LIGHT_DIR" \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --output-json risk_mining/output_sind_traffic_lights/audit.json
```

输出 JSON 中包含：

- 每个 location/scene 是否找到 CSV
- 使用 `timestamp(ms)` 还是 `RawFrameID` 相对偏移对齐
- 灯号数量、生成行数、状态计数
- 未知编码统计

## 可视化验证入口

脚本：`scripts/sind_visualize_traffic_lights.py`

生成每个 scene 的相位时间轴、交通流响应曲线和汇总 HTML：

```bash
python scripts/sind_visualize_traffic_lights.py \
  --traffic-light-dir "$SIND_TRAFFIC_LIGHT_DIR" \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --locations cc tj xasl cqNR cqIR cqR \
  --output-dir risk_mining/output_sind_traffic_lights
```

快速抽样检查：

```bash
python scripts/sind_visualize_traffic_lights.py \
  --traffic-light-dir "$SIND_TRAFFIC_LIGHT_DIR" \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --locations tj xasl cqNR \
  --scenes-per-location 1 \
  --output-dir /tmp/sind_tls_visual_check
```

主要输出：

- `index.html`：总入口，列出每个 scene 的状态、灯号数量、UNKNOWN 比例和对齐方式。
- `scenes/<scene>.html`：单 scene 检查页。
- `figures/*_phase_matrix.png`：每个灯号的红/绿/黄/UNKNOWN 时间轴。
- `figures/*_response.png`：红绿灯状态占比与车辆进入路口、等待、速度曲线。
- `summary.csv` / `summary.json`：机器可读汇总。
- `mapping_coverage.json`：每个 location 的灯号-车道绑定覆盖率。

判读建议：

- 绿灯开始后，`Core entries / bin` 通常应上升，`Approach stopped vehicles` 应下降。
- 红灯开始后，入口等待车辆通常应上升，进入路口车辆数应下降。
- 如果相位切换与车辆响应整体错开很多秒，优先检查 `timestamp(ms)` / `RawFrameID` 对齐。
- 当前没有真实 `light_id -> lane_id` 映射时，响应分数是全局 sanity check，不代表某个灯号已经绑定到具体车道。

## 灯号-车道绑定标注工具

标注工具已经独立到项目同级目录：

```text
../SinD_TrafficLight_Annotator
```

该工具为每个路口提供一个静态 HTML 标注页：页面中显示 Lanelet2 车道和 lane id，官方信号灯编号图片保留在工具目录的 `city_images/` 中，需要单独打开原图查看。

入口：

```text
../SinD_TrafficLight_Annotator/index.html
```

核心文件：

```text
../SinD_TrafficLight_Annotator/generate_annotator.py
../SinD_TrafficLight_Annotator/merge_annotation.py
../SinD_TrafficLight_Annotator/mapping_template.json
../SinD_TrafficLight_Annotator/city_images/
../SinD_TrafficLight_Annotator/annotator_pages/
```

标注前必须先把 `city_images/` 中官方 PNG 的坐标系/路口方向与 HTML 中的 Lanelet2 车道图对齐，再开始选择灯号和点击车道。HTML 中不嵌入 PNG，因为缩放后不利于看清灯号。

标注流程：

1. 打开 `city_images/` 中对应城市的官方 PNG，在图片查看器中放大。
2. 打开对应 HTML，例如 `annotator_pages/cqNR.html`。
3. 先把官方 PNG 坐标系和 Lanelet2 图坐标系对齐。
4. 右侧选择一个灯号，例如 `vehicle:1` 或 `general:1`。
5. 在 Lanelet2 图上点击该灯控制的 lane / movement connector。
6. 使用 `Copy JSON` 或 `Download JSON` 导出标注。
7. 使用 `merge_annotation.py` 合并导出的标注 JSON。

合并示例：

```bash
cd ../SinD_TrafficLight_Annotator
python merge_annotation.py \
  /path/to/sind_tl_lane_mapping_cqNR.json \
  --traffic-light-dir /home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --lanelet-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/Lanelet_maps_SinD \
  --mapping-path mapping_template.json \
  --output merged_mapping.json
```

最终可将 `merged_mapping.json` 的内容同步回主项目：

```text
src/trajdata/dataset_specific/sind/sind_traffic_light_mapping.json
```

安全检查：

- 会验证 location 是否存在。
- 会验证灯号 key 是否出现在该路口 TrafficLight CSV 中。
- 会验证 lane id 是否存在于 Lanelet2 OSM 中。
- 如果一个 lane 被多个灯号绑定，会给出 warning。

注意：绑定关系使用 Lanelet2 lane id。如果需要在 trajdata map 中显示真实车道红绿灯，缓存地图时建议使用 `map_params={"use_lanelet2_maps": True}`。

## 后续补强

- 根据标注工具导出的结果，在 `sind_traffic_light_mapping.json` 中逐步补充真实 `light_to_lanes`。
- 补充后，trajdata 的 map API 可以通过真实 `(lane_id, scene_ts)` 查询对应红绿灯状态。
- 若需要论文级验证，可另做一个交通流反推脚本，对比车辆停止/启动行为与 CSV 相位的一致性。
