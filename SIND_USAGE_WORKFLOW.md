# SinD 数据集使用流程

## 完整工作流程

```
准备数据 → 缓存数据 → 加载数据集 → 使用数据
```

---

## 第一步：准备数据

### 1.1 确认数据目录结构

确保 SinD 数据集按以下结构组织：

```bash
/path/to/SinD_dataset/
├── cc/
│   ├── tp_info_cc.pkl
│   ├── frame_data_cc.pkl
│   └── cc_map.json  (或在 output_json/cc_map.json)
├── xa/
│   ├── tp_info_xa.pkl
│   ├── frame_data_xa.pkl
│   └── xa_map.json
├── cqNR/
├── tj/
├── cqIR/
├── xasl/
└── cqR/
```

### 1.2 配置数据路径

```python
from pathlib import Path

SIND_DATA_DIR = Path("/path/to/your/SinD_dataset")
```

---

## 第二步：缓存数据

**重要说明：** 首次使用时会自动缓存，无需手动操作。但了解缓存过程有助于排查问题。

### 2.1 缓存位置

```bash
~/.unified_data_cache/sind/
```

### 2.2 缓存内容

| 缓存文件 | 说明 | 大小（参考）|
|---------|------|------------|
| `scenes_list.dill` | 场景列表（13个xa场景）| ~1KB |
| `maps/` | 矢量地图缓存 | ~10MB |
| `xa_{scene_name}/` | 每个场景的缓存 | ~10-50MB/场景 |

### 2.3 首次运行（自动缓存）

```python
from trajdata import UnifiedDataset, AgentType
from collections import defaultdict
from pathlib import Path

dataset = UnifiedDataset(
    desired_data=["sind-xa"],  # 只缓存 xa 位置
    data_dirs={"sind": str(SIND_DATA_DIR)},
    desired_dt=0.1,
    centric="agent",
    only_predict=[AgentType.VEHICLE],
    incl_raster_map=True,
    raster_map_params={
        "px_per_m": 2,
        "map_size_px": 224,
        "offset_frac_xy": (-0.5, 0.0),
    },
    verbose=True,  # 显示进度
)

print(f"缓存完成！数据集大小: {len(dataset):,}")
```

**预期输出：**
```
Getting Scenes from sind with scene tag [sind-xa]: 100%|██████████| 1/1 [00:04<00:00,  4.53s/it]
Calculating Agent Data (Serially): 100%██████████] 13/13 [01:14<00:00,  5.75s/it]
13 scenes in the scene index.
Creating Agent Data Index (Serially): 100%|██████████] 13/13 [00:00<00:00, 255.27it/s]
Dataset loaded successfully!
Number of samples: 1,931,736
```

---

## 第三步：使用数据

### 3.1 基础用法

```python
from torch.utils.data import DataLoader
from trajdata import UnifiedDataset, AgentType
from collections import defaultdict

# 1. 创建数据集
dataset = UnifiedDataset(
    desired_data=["sind-xa"],              # 指定位置
    data_dirs={"sind": str(SIND_DATA_DIR)},
    desired_dt=0.1,                         # 10Hz
    centric="agent",                        # 以智能体为中心
    history_sec=(2.0, 2.0),                 # 2秒历史
    future_sec=(4.0, 4.0),                  # 4秒未来
    only_predict=[AgentType.VEHICLE],       # 只预测车辆
    incl_robot_future=False,
    incl_raster_map=True,                   # 包含栅格地图
    raster_map_params={
        "px_per_m": 2,                      # 每米像素数
        "map_size_px": 224,                 # 地图大小
        "offset_frac_xy": (-0.5, 0.0),      # 智能体在地图中的位置
    },
)

# 2. 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=4,                           # 批次大小
    shuffle=True,                           # 打乱数据
    collate_fn=dataset.get_collate_fn(),   # 整合函数
    num_workers=0,                          # 工作进程数
)

# 3. 遍历数据
for batch in dataloader:
    # batch 是 AgentBatch 对象
    print(f"批次大小: {len(batch.agent_name)}")

    for i in range(len(batch.agent_name)):
        agent_name = batch.agent_name[i]
        curr_state = batch.curr_agent_state[i]

        x, y = curr_state.position[0].cpu().numpy()
        vx, vy = curr_state.velocity[0].cpu().numpy()

        print(f"  智能体 {agent_name}: 位置=({x:.2f}, {y:.2f}), 速度=({vx:.2f}, {vy:.2f})")
```

### 3.2 访问场景信息

```python
# 获取场景
scene = dataset.get_scene(0)  # 第一个场景

print(f"场景名称: {scene.name}")
print(f"位置: {scene.location}")
print(f"时长: {scene.length_timesteps} 时间步")
print(f"dt: {scene.dt} 秒")
print(f"智能体数量: {len(scene.agents)}")

# 查看智能体信息
for agent in scene.agents[:5]:  # 前5个智能体
    print(f"  {agent.name}: {agent.type}, "
          f"帧 [{agent.first_timestep}, {agent.last_timestep}]")

# 查看某个时间步的智能体
timestep = 100
agents_at_ts = scene.agent_presence[timestep]
print(f"时间步 {timestep} 有 {len(agents_at_ts)} 个智能体")
```

### 3.3 查询智能体状态

```python
from trajdata.caching.df_cache import DataFrameCache
from pathlib import Path

# 创建缓存对象
cache_path = Path.home() / ".unified_data_cache"
scene_cache = DataFrameCache(cache_path, scene)

# 查询特定时间步的智能体状态
timestep = 50
agents_at_ts = scene.agent_presence[timestep]
agent_ids = [agent.name for agent in agents_at_ts]

# 获取状态
states = scene_cache.get_states(agent_ids, timestep)

# 处理状态数据
for agent_id, state in zip(agent_ids, states):
    # 位置
    x, y = state.position[0], state.position[1]

    # 速度
    vx, vy = state.velocity[0], state.velocity[1]
    speed = (vx**2 + vy**2)**0.5

    # 加速度
    ax, ay = state.acceleration[0], state.acceleration[1]

    # 航向角
    heading = state.heading[0]

    print(f"{agent_id}: pos=({x:.2f}, {y:.2f}), speed={speed:.2f}m/s, "
          f"heading={heading:.2f}rad")
```

### 3.4 可视化

```python
from trajdata.visualization.vis import plot_agent_batch
import matplotlib.pyplot as plt

# 获取一个批次
batch = next(iter(dataloader))

# 可视化第一个智能体
plot_agent_batch(
    batch,
    batch_idx=0,
    show=True,
)
```

---

## 第四步：高级用法

### 4.1 加载多个位置

```python
# 加载多个位置（会使用更多内存）
dataset = UnifiedDataset(
    desired_data=["sind-xa", "sind-cc"],  # xa 和 cc 位置
    data_dirs={"sind": str(SIND_DATA_DIR)},
    ...
)
```

### 4.2 只加载地图

```python
# 只需要地图信息，不需要智能体数据
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    incl_raster_map=False,               # 不需要栅格地图
    incl_vector_map=True,                # 使用矢量地图
)
```

### 4.3 过滤智能体类型

```python
# 只预测车辆和行人
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
    no_types=[AgentType.BICYCLE],        # 排除自行车
    ...
)
```

### 4.4 设置邻居交互距离

```python
from collections import defaultdict

dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    agent_interaction_distances=defaultdict(lambda: 50.0),  # 50米半径
    ...
)
```

---

## 第五步：管理缓存

### 5.1 查看缓存大小

```bash
# 查看 sind 缓存目录大小
du -sh ~/.unified_data_cache/sind/

# 查看详细文件
ls -lh ~/.unified_data_cache/sind/
```

### 5.2 清除缓存

```bash
# 清除特定位置的缓存
rm -rf ~/.unified_data_cache/sind/xa_*

# 清除所有 sind 缓存
rm -rf ~/.unified_data_cache/sind/

# 清除特定场景缓存
rm -rf ~/.unified_data_cache/sind/xa_4.12\ morning\ 2\ xa/
```

### 5.3 强制重建缓存

```python
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    rebuild_cache=True,  # 强制重建缓存
    ...
)
```

---

## 常见使用场景

### 场景1：训练模型

```python
from torch.utils.data import DataLoader

# 创建数据集
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    desired_dt=0.1,
    centric="agent",
    history_sec=(2.0, 2.0),
    future_sec=(4.0, 4.0),
    only_predict=[AgentType.VEHICLE],
)

# 训练数据加载器
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.get_collate_fn(),
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        # batch.curr_agent_state: 当前状态
        # batch.agent_history: 历史轨迹
        # batch.agent_future: 未来轨迹
        # batch.raster_map: 栅格地图（如果启用）
        loss = model(batch)
        ...
```

### 场景2：数据分析和可视化

```python
# 加载数据集
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    desired_dt=0.1,
    centric="agent",
    history_sec=(2.0, 2.0),
    future_sec=(4.0, 4.0),
)

# 获取场景
scene = dataset.get_scene(0)

# 分析智能体数量分布
agent_counts = [len(scene.agent_presence[ts])
                for ts in range(scene.length_timesteps)]

print(f"平均智能体数: {sum(agent_counts)/len(agent_counts):.2f}")
print(f"最大智能体数: {max(agent_counts)}")
print(f"最小智能体数: {min(agent_counts)}")

# 可视化某个轨迹
from trajdata.caching.df_cache import DataFrameCache
cache = DataFrameCache(Path.home() / ".unified_data_cache", scene)

agent_id = "4"
agent = next(a for a in scene.agents if a.name == agent_id)

# 获取该智能体的所有状态
states = []
for ts in range(agent.first_timestep, agent.last_timestep + 1):
    state = cache.get_state(agent_id, ts)
    states.append(state)

# 绘制轨迹
import matplotlib.pyplot as plt

x_coords = [s.position[0] for s in states]
y_coords = [s.position[1] for s in states]

plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, 'b-', label='轨迹')
plt.scatter(x_coords[0], y_coords[0], c='g', s=100, label='起点')
plt.scatter(x_coords[-1], y_coords[-1], c='r', s=100, label='终点')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'智能体 {agent_id} 轨迹')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
```

### 场景3：查询特定场景和智能体

```python
# 加载特定位置的数据
dataset = UnifiedDataset(
    desired_data=["sind-xa"],
    data_dirs={"sind": str(SIND_DATA_DIR)},
    ...
)

# 列出所有场景
for i in range(dataset.num_scenes()):
    scene = dataset.get_scene(i)
    print(f"场景 {i}: {scene.name}, "
          f"位置: {scene.location}, "
          f"时长: {scene.length_timesteps}, "
          f"智能体数: {len(scene.agents)}")

# 查找特定场景
target_scene_name = "xa_4.12 morning 2 xa"
for i in range(dataset.num_scenes()):
    scene = dataset.get_scene(i)
    if scene.name == target_scene_name:
        print(f"找到场景！索引: {i}")
        break
```

---

## 性能优化建议

### 内存优化

```python
# 1. 只加载需要的位置
dataset = UnifiedDataset(
    desired_data=["sind-xa"],  # 只加载一个位置
    ...
)

# 2. 关闭不需要的功能
dataset = UnifiedDataset(
    ...
    incl_raster_map=False,      # 不使用栅格地图（节省内存）
    incl_vector_map=False,      # 不使用矢量地图
)

# 3. 减小历史和未来窗口
dataset = UnifiedDataset(
    ...
    history_sec=(1.0, 1.0),     # 减少历史长度
    future_sec=(2.0, 2.0),      # 减少未来长度
)
```

### 速度优化

```python
# 1. 使用多个工作进程
dataloader = DataLoader(
    dataset,
    num_workers=4,  # 使用4个进程
    ...
)

# 2. 增加批次大小
dataloader = DataLoader(
    dataset,
    batch_size=64,  # 更大的批次
    ...
)

# 3. 预缓存到内存（如果内存足够）
from trajdata.caching import SceneCache
cache = SceneCache(cache_path, scene)
cache.preload_into_memory()
```

---

## 故障排除

### 问题1：缓存损坏

**症状：** `IndexError: list index out of range`

**解决：**
```bash
# 删除缓存并重新运行
rm -rf ~/.unified_data_cache/sind/
```

### 问题2：内存不足

**症状：** 程序崩溃或系统变慢

**解决：**
```python
# 只加载一个位置
desired_data=["sind-xa"]

# 或使用 num_workers=0
dataloader = DataLoader(..., num_workers=0)
```

### 问题3：找不到地图文件

**症状：** `FileNotFoundError: map.json`

**解决：**
```python
# 检查地图文件位置
# 可能需要在 output_json/ 目录下
# 或创建符号链接
ln -s /path/to/output_json/xa_map.json /path/to/SinD_dataset/xa/xa_map.json
```

---

## 快速参考

### 常用参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `desired_dt` | 时间步长（秒）| 0.1 (10Hz, SinD原生频率) |
| `history_sec` | 历史时长 | (2.0, 2.0) |
| `future_sec` | 未来时长 | (4.0, 4.0) |
| `agent_interaction_distances` | 邻居半径 | 50.0 (米) |
| `batch_size` | 批次大小 | 4-32 |
| `num_workers` | 工作进程 | 0-4 |

### AgentType 枚举

```python
from trajdata import AgentType

AgentType.VEHICLE      # 车辆
AgentType.PEDESTRIAN   # 行人
AgentType.BICYCLE      # 自行车
AgentType.MOTORCYCLE   # 摩托车
```

### 场景标签格式

```python
"sind"           # 所有位置（不推荐，内存占用大）
"sind-xa"         # 只有 xa 位置
"sind-cc"         # 只有 cc 位置
"sind-tj"         # 只有 tj 位置
...
```
