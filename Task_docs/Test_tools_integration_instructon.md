<!--
 * @Author: Yunwei Li 1084087910@qq.com
 * @Date: 2026-05-08 05:47:04
 * @LastEditors: Yunwei Li 1084087910@qq.com
 * @LastEditTime: 2026-05-08 06:12:57
 * @FilePath: /My_trajdata/Task_docs/Test_tools_integration_instructon.md
 * @Description: 
 * 
 * Copyright (c) 2026 by Tsinghua University, All Rights Reserved. 
-->
## 任务说明

当前的项目是在Trajdata工具的基础上将SinD数据集整合进来（设为A节点），然后在risk_mining 中进行了一些分析。

在进行到A节点的同时，我将当时的项目同步给了Anran同学，他来在A节点的项目基础上实现了测试工具链：即选取场景中的特定车辆作为主车（ego），让被测算法（如IDM，ASAPRL等）接管主车，在交通场景中进行测试；
测试主要包括：
1. 开环测试（也就是回放测试）——其他车辆按轨迹数据集记录播放运动，另一类是
2. 闭环测试——部分其他车辆会根据自车行为来决定后续行为，即交互式仿真，这个就需要其他车被算法控制或干预，Anran所使用的是Trace模型（一种基于Diffusion Model的轨迹预测模型）

现在需要将Anran所制作的项目文件进行分析，将核心必要的模块增加到本项目当中。

Anran所制作的项目文件位于：/media/lyw/KESU/sind_testing_toolchain
其中trajdata_-main.zip 是项目文件夹，ckpt.zip是ASAPRL算法所需的权重文件。你可以在/media/lyw/KESU/sind_testing_toolchain目录下解压来阅读理解

## 任务大局观
最终是想形成一个测试工具链，可以对被测对象、测试场景集合（来自xx数据集的xx场景），希望记录的指标（有可能想直接记录所有相关车辆的轨迹过程数据以便测试结束后的离线指标分析/或者想选择几个指标直接进行记录，现有指标可以看一下Anran项目中是否已经有一些评价指标了），以及一些其他测试设置（开环/闭环，等）

所以可以构思一个合适的项目框架来进行整合。

下一步我这边还会开展将现有场景打上一些语义标签，例如哪些场景是高风险的，哪些场景是视野遮挡的，我目前初步构思是会在数据中记录这些场景对应的开始和结束的时间戳等基础信息，来查询。你可以考虑到这点。

## 任务验收

* 需要有一个场景测试可视化工具，可以是bokeh交互式html，我可以查看测试过程中，每一时间步中，各个交通参与者的运动状态信息，对于主车，需要显示其被测算法名称和给出的控制命令数据（油门 转角  或者 加速度）。首先完成这个，在SinD数据集中抽样测试。我来检查

* 检查没问题了之后再尝试进行批量测试和记录。

## 当前集成状态

测试工具链已经整理到 `Simulation_test_toolchain/`，支持通过 YAML 配置运行 SinD 单场景测试并输出：

- `interactive.html`：Bokeh 交互式回放，显示地图、交通参与者、ego 高亮和控制命令。
- `trajectory_log.json`：结构化轨迹和元数据。
- `trajectory_log.csv`：扁平轨迹表，便于离线指标分析。

当前已接入的 ego policy：

- `risk_idm`
- `asaprl`
- `ground_truth`

当前限制与注意事项：

- 运行环境使用 `conda run -n trajdata ...`。
- `diffuser` 入口已预留，但依赖 TRACE/tbsim 和对应 checkpoint。
- SinD 全局 scene list 已支持多路口，若只出现单一路口，需要运行 `Simulation_test_toolchain/rebuild_sind_scene_list.py` 重建缓存。

## 语义场景接入

语义标签工具链已经接入测试工具链。配置中设置：

```yaml
scenario:
  semantic_label_id: SIND_CC_NARROW_10_CC_10_CC_PART_18_EGO24552
  semantic_label_path: datasets/SinD_dataset/Semantic_labels/scenarios.json
  semantic_min_num_steps: 150
```

运行时会自动从标签解析 `location`、`scene_index`、`init_timestep` 和 `ego_agent_name`，并保证测试片段至少覆盖标签时间窗和 `semantic_min_num_steps`。
