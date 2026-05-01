# SinD 场景过滤小工具

本文档说明同学整理的三个 SinD 过滤工具如何在本项目中使用。工具已经整合到 `trajdata` 包内，其他脚本可以直接导入：

```python
from trajdata.dataset_specific.sind.scene_filters import (
    build_area_boundary,
    build_lane_area,
    is_following_vehicle,
    is_position_invalid,
    is_vehicle_static,
    load_curbstone_points,
)
```

## 文件位置

| 文件 | 用途 |
|------|------|
| `src/trajdata/dataset_specific/sind/scene_filters.py` | 过滤器核心函数 |
| `src/trajdata/dataset_specific/sind/data/curbstone.json` | 七个 SinD 路口的边界和关键点 |
| `examples/sind_scene_filter_example.py` | 使用真实 SinD pkl 数据的示例 |

`curbstone.json` 中的 `key_points` 是每个城市的关键点，工具用这些点构建入口车道和出口车道区域，从而判断车辆是否在入口道路、出口道路或可行驶区域内。

## 三个过滤器

### 1. 静止车辆过滤

`is_vehicle_static(trajectory, threshold=0.01)`

判断轨迹起点和终点在 `x/y` 方向上的位移是否都小于阈值。返回 `True` 表示车辆静止，通常应过滤。

```python
is_static = is_vehicle_static(trajectory)
```

`trajectory` 通常是 `tp_info[scene_id][tp_id]["State"]`，需要包含 `x`、`y` 两列。

### 2. 跟车状态过滤

`is_following_vehicle(main_vehicle, other_vehicles, lane_area, distance_threshold=10, velocity_threshold=2)`

判断主车是否在入口车道内跟随前车。当前逻辑要求：

- 主车在入口车道；
- 候选前车位于主车航向前方约 30 度范围内；
- 前车也在入口车道；
- 两车距离小于 `10m`；
- 相对速度小于 `2m/s`。

返回 `True` 表示该车处于跟车状态，通常应过滤。

### 3. 位置合法性过滤

`is_position_invalid(main_vehicle, outer_path, lane_area, inner_paths=None)`

判断车辆位置是否不适合保留。当前逻辑中，满足任一条件就返回 `True`：

- 不在外层可行驶区域边界内；
- 在内部不可行驶区域内（传入 `inner_paths` 时生效）；
- 在出口车道内。

## 基本使用流程

```python
from trajdata.dataset_specific.sind.scene_filters import (
    build_area_boundary,
    build_lane_area,
    is_following_vehicle,
    is_position_invalid,
    is_vehicle_static,
    load_curbstone_points,
)

city = "cc"
city_points = load_curbstone_points()
outer_path, inner_paths = build_area_boundary(city_points, city)
lane_area = build_lane_area(city_points, city)

for vehicle in motor_vehicles:
    trajectory = scene_tp_info.get(vehicle["tp_id"], {}).get("State")
    if trajectory is None:
        continue

    if is_vehicle_static(trajectory):
        continue

    if is_following_vehicle(vehicle, motor_vehicles, lane_area):
        continue

    if is_position_invalid(vehicle, outer_path, lane_area, inner_paths):
        continue

    # vehicle 通过三个过滤器，可进入后续处理
```

## 运行示例

默认读取 `datasets/SinD_dataset`：

```bash
python examples/sind_scene_filter_example.py --city cc
```

指定数据集路径：

```bash
python examples/sind_scene_filter_example.py \
  --data-dir /home/lyw/1TBSSD/Datasets/SinD_dataset_Simple \
  --city tj
```

## 支持的城市

当前 `curbstone.json` 包含：

```text
cc, cqIR, cqNR, cqR, tj, xa, xasl
```

也可以在代码中查询：

```python
from trajdata.dataset_specific.sind.scene_filters import available_curbstone_locations

print(available_curbstone_locations())
```

## 数据格式约定

车辆对象沿用 SinD pkl 中 `frame_data` 的格式：

```python
vehicle = {
    "tp_id": 1,
    "vehicle_info": {
        "x": 10.0,
        "y": 20.0,
        "vx": 5.0,
        "vy": 0.0,
        "heading_rad": 0.0,
        "width": 2.0,
        "length": 4.5,
        "agent_type": "car",
    },
}
```

轨迹对象通常是 pandas DataFrame，至少需要 `x`、`y` 两列。跟车过滤还需要车辆字典中存在 `vx`、`vy`、`heading_rad`。
