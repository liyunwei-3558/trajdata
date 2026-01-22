正在收集工作区信息正在筛选到最相关的信息下面是一份基于 examples 目录的简明使用文档，覆盖常见工作流（加载数据 / 自定义 extras / 可视化 / 仿真 / 预处理）。文中给出关键代码片段及指向仓库中相关文件与符号的快捷链接，方便打开查看源码与示例。

概览
- examples 目录（可打开查看所有示例）：  
  batch_example.py  
  cache_and_filter_example.py  
  custom_batch_data.py  
  lane_query_example.py  
  map_api_example.py  
  preprocess_data.py  
  preprocess_maps.py  
  scene_batch_example.py  
  scenetimebatcher_example.py  
  sim_example.py  
  simple_map_api_example.py  
  simple_sim_example.py  
  speed_example.py  
  state_example.py  
  visualization_example.py  
  visualization_example.html

快速入门（最小示例）
- 核心类/函数：
  - 数据集：`trajdata.dataset.UnifiedDataset`  
  - 获取 batch 的 collate：`trajdata.dataset.UnifiedDataset.get_collate_fn`  
  - Collate 实现：`trajdata.data_structures.collation.agent_collate_fn`、`trajdata.data_structures.collation.scene_collate_fn`  
  - Batch 类型：`trajdata.data_structures.batch.AgentBatch`  
  - 仿真场景：`trajdata.simulation.sim_scene.SimulationScene`

最小代码（加载并迭代）：
````python
import os
from torch.utils.data import DataLoader
from trajdata import UnifiedDataset, AgentBatch

dataset = UnifiedDataset(
    desired_data=["nusc_mini"],
    centric="agent",
    desired_dt=0.1,
    incl_raster_map=True,
    data_dirs={"nusc_mini": "~/datasets/nuScenes"},
)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=dataset.get_collate_fn(), num_workers=0)

for batch in dataloader:  # batch 为 AgentBatch（参见上面链接）
    # 处理 batch
    break
````

自定义 extras（在 batch 中扩展自定义数据）
- 示例：examples/custom_batch_data.py  
- 用法要点：
  - 在构造 `UnifiedDataset` 时传入 extras 字典，函数会在构建每个 batch element 时被调用并加入 `batch.extras`。
  - extras 函数签名：接收 `AgentBatchElement` 或 `SceneBatchElement`，返回 numpy 数组或自定义可拼接对象（可参考 `trajdata.data_structures.batch_element.AgentBatchElement`）。
- 参考：custom_batch_data.py

可视化
- 同步绘图：`trajdata.visualization.vis.plot_agent_batch`  
- 交互式：`trajdata.visualization.interactive_vis.plot_agent_batch_interactive` 与动画工具 `trajdata.visualization.interactive_animation.InteractiveAnimation`  
- 示例：打开 visualization_example.py 查看典型工作流（加载 dataset -> dataloader -> plot）。

仿真（在场景上运行 agent）
- 主要类：`trajdata.simulation.sim_scene.SimulationScene`  
- 仿真缓存实现示例：`trajdata.simulation.sim_df_cache.SimulationDataFrameCache`  
- 示例：simple_sim_example.py 与更完整的 sim_example.py（包含统计与可视化）。
- 最小仿真循环示例：
````python
from trajdata.simulation import SimulationScene
from trajdata import UnifiedDataset

dataset = UnifiedDataset(desired_data=["nusc_mini"], data_dirs={"nusc_mini": "~/datasets/nuScenes"})
scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(env_name="sim", scene_name="sim_scene", scene=scene, dataset=dataset)
obs = sim_scene.reset()
for t in range(1, sim_scene.scene.length_timesteps):
    new_xyzh = {}  # dict[str, StateArray] 填入下一时刻状态
    obs = sim_scene.step(new_xyzh)
````

地图与预处理
- 地图 API：`trajdata.maps.map_api.MapAPI`  
- 预处理示例：preprocess_maps.py、preprocess_data.py  
- 缓存 loader：`trajdata.caching.df_cache.DataFrameCache.load_rtrees`（示例在代码中有 warning 与使用 dill 加载的逻辑）

数据增强与批内变换
- 批增强基类：`trajdata.augmentation.augmentation.BatchAugmentation`  
- 示例实现：`trajdata.augmentation.noise_histories.NoiseHistories`  
- 在构造 `UnifiedDataset` 时传入 augmentations 列表，collate 时会应用到 batch（参见 `trajdata.dataset.UnifiedDataset.get_collate_fn` 中对 batch_augments 的处理）。

常见技巧与注意事项
- padding 格式：get_collate_fn(pad_format="outside" | "right")，影响历史/未来序列对齐（详见 `trajdata.dataset.UnifiedDataset.get_collate_fn`）。  
- 多线程加载：通过 DataLoader 的 num_workers 控制（examples 中常用 num_workers=0/4），若遇到 pickling 问题可暂时用 0 调试。  
- 自定义 extras custom_batch_data.py 中展示先后顺序利用 dict 保序特性）。  
- 若需要地图空间查询或加速，检查是否已构建 map cache（见 preprocess_maps 示例与 df_cache 的 load_rtrees）。

常用命令（在项目根目录运行）
- 运行示例脚本：
````sh
python examples/visualization_example.py
python examples/simple_sim_example.py
python examples/custom_batch_data.py
````

进阶资源（源码定位）
- 数据集与 collate：dataset.py （类：`trajdata.dataset.UnifiedDataset`）  
- 批/元素数据结构：batch.py（`trajdata.data_structures.batch.AgentBatch`）  
- BatchElement：batch_element.py（`trajdata.data_structures.batch_element.AgentBatchElement`）  
- Collation：collation.py（`trajdata.data_structures.collation.agent_collate_fn` / `trajdata.data_structures.collation.scene_collate_fn`）  
- 可视化：vis.py、interactive_vis.py  
- 仿真：sim_scene.py、sim_df_cache.py、sim_metrics.py  
- Map API：map_api.py  
- Augmentation 示例：noise_histories.py

如果你希望我把这份文档整理成 README 或示例脚本（例如新增一个 minimal_readme.md 或 minimal_example.py），我可以直接生成文件内容供你粘贴或保存。