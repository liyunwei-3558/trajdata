# SinD Traffic Light Integration

本文档说明本项目对 SinD 交通信号灯数据的补充接入。当前支持两种来源：整理后的本地 traffic-light pkl，以及原始 `TrafficLight*.csv`。默认优先读取本地 pkl；若 pkl 缺失，再回退到外部 CSV。

## 数据来源与范围

- 原始信号灯 CSV 根目录：`/home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv`
- 整理后的本地 pkl 保存位置：`datasets/SinD_dataset/<location>/traffic_lights_<location>.pkl`
- 默认读取优先级：本地 pkl 优先，外部 CSV 回退。
- 若只使用本地 pkl，不需要设置环境变量；若需要从 CSV 生成或回退，可设置：

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

- 读取每个 location 的 `traffic_lights_<location>.pkl`，或回退读取每个 scene 文件夹中的 `TrafficLight*.csv` / `Traffic_Lights.csv`
- 将 CSV 中的相位变化点扩展为逐 `scene_ts` 的状态表，并可离线保存进本地 pkl
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

## 本地 pkl 生成

脚本：`scripts/sind_build_traffic_light_pkls.py`

```bash
conda run -n trajdata python scripts/sind_build_traffic_light_pkls.py \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --traffic-light-dir /home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv \
  --locations cc tj xasl cqNR cqIR cqR \
  --output-summary risk_mining/output_sind_traffic_lights/pkl_build_summary.json
```

每个 pkl 保存：

- `raw_changes`：原始 CSV 切换表，并附加 `_scene_ts` 对齐列。
- `traffic_light_status`：trajdata 标准 MultiIndex DataFrame，index 为 `(lane_id, scene_ts)`，列为 `status`。
- `report`：构建诊断信息，包括状态计数、对齐方式、原始 CSV 路径。

当前已生成：

| location | scenes | ok | skipped |
|---|---:|---:|---:|
| cc | 8 | 8 | 0 |
| tj | 23 | 23 | 0 |
| xasl | 15 | 15 | 0 |
| cqNR | 10 | 9 | 1 |
| cqIR | 9 | 7 | 2 |
| cqR | 10 | 8 | 2 |

## 缓存入口

文件：`src/trajdata/dataset_specific/sind/sind_dataset.py`

在 `SindDataset.get_agent_info()` 完成 agent cache 后，会尝试执行信号灯缓存：

- 找到 `datasets/SinD_dataset/<location>/traffic_lights_<location>.pkl`：优先从 pkl 读取。
- pkl 缺失且设置了 `SIND_TRAFFIC_LIGHT_DIR` 或传入 CSV root：回退到原始 CSV。
- 找不到某个 scene 的信号灯：跳过该 scene 的信号灯，不影响轨迹缓存。
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

结合本地 pkl 数据检查 scene 对齐、状态分布和逐帧表生成；不传 `--traffic-light-dir` 时即验证 pkl-only 路径：

```bash
python scripts/sind_audit_traffic_lights.py \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --locations cc tj xasl cqNR cqIR cqR \
  --output-json risk_mining/output_sind_traffic_lights/audit_pkl.json
```

如需验证 CSV 回退，也可同时传入 `--traffic-light-dir`。

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
  --sind-data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --locations cc tj xasl cqNR cqIR cqR \
  --output-dir risk_mining/output_sind_traffic_lights
```

快速抽样检查：

```bash
python scripts/sind_visualize_traffic_lights.py \
  --sind-data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
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
- 当前 `sind_traffic_light_mapping.json` 已写入人工标注的灯号-车道/人行横道绑定。若某个 CSV 灯号仍未标注，会保留 synthetic id 作为提示。

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
  --sind-data-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset \
  --lanelet-dir /home/lyw/1TBSSD/Datasets/ClaudeWork/SinD_TrafficLight_Annotator/lanelet_maps \
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

## 当前状态与后续补强

- 人工标注结果已经整合进 `src/trajdata/dataset_specific/sind/sind_traffic_light_mapping.json`。
- `cqNR/cqIR/cqR` 已包含 pedestrian crosswalk 绑定；其它路口当前导出结果未包含行人灯 crosswalk 绑定。
- `cqIR` 存在两个 lane 被多个灯号绑定的 warning；代码采用 deterministic last-wins 去重，避免 `(lane_id, scene_ts)` 重复破坏缓存和可视化。
- 后续若继续修订标注，需重新合并 mapping，并重新运行 `scripts/sind_build_traffic_light_pkls.py` 生成本地 pkl。
- 若需要论文级验证，可另做交通流反推脚本，对比车辆停止/启动行为与 CSV 相位的一致性。
