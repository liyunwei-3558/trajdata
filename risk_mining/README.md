# Risk Mining Framework

这份说明面向当前 `risk_mining` 的实际实现，重点是帮助你快速理解：

- 现在这套框架怎么跑
- 规则是在哪一层生效的
- 如果你要新增/修改规则，应该改哪里
- 输出结果和可视化文件怎么看

## 1. 当前框架概览

当前流程是：

`trajdata Scene -> Slicer -> Episode -> RuleRegistry.apply_all() -> SSTG -> SanityChecker -> DualLibrary / manual_review`

对应入口文件是 [run_pipeline.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/run_pipeline.py)。

目前这套实现是一个“可扩展的骨架版本”：

- `Slicer` 先从真实场景中切出候选 episode
- 规则系统负责往图里加节点和边
- `SanityChecker` 检查图里是否存在可解释的 `causal` 边
- 通过检查的 episode 存入 JSON 事件库
- 不通过的 episode 存入 `manual_review/`

## 2. 目录结构

```text
risk_mining/
├── config/
│   └── rules_config.yaml
├── src/
│   ├── core/
│   │   ├── scene_graph.py
│   │   └── slicer.py
│   ├── rules/
│   │   ├── base_rule.py
│   │   ├── spatial_rules.py
│   │   └── dynamic_rules.py
│   ├── library/
│   │   └── dual_repo.py
│   └── utils/
│       ├── checker.py
│       └── logger.py
├── run_pipeline.py
├── extract_episodes.py
└── visualize_episode.py
```

## 3. 核心数据对象

### 3.1 SSTG

文件：[scene_graph.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/core/scene_graph.py)

`SSTG` 是一个 `networkx.DiGraph` 包装器，当前统一使用三类边：

- `spatial`
- `temporal`
- `causal`

节点 `Node` 当前的关键字段：

- `agent_id`
- `timestamp`
- `type`
- `velocity`
- `acceleration`
- `position`
- `heading`
- `extent`
- `metadata`

其中 `timestamp` 目前不是全时间序列，而是三个语义时间点：

- `T_start`
- `T_peak`
- `T_end`

边 `Edge` 当前的关键字段：

- `source_id`
- `target_id`
- `source_timestamp`
- `target_timestamp`
- `edge_type`
- `weight`
- `relation`
- `metadata`

### 3.2 Episode

文件：[slicer.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/core/slicer.py)

`Episode` 是规则系统的输入对象。它包含：

- 场景元信息：`scene_id`, `scene_name`, `env_name`
- 时间窗：`t_start`, `t_peak`, `t_end`
- 自车：`ego_agent_id`
- 参与体：`involved_agents`
- 三个语义时刻的状态快照：`state_snapshots`
- 当前风险分数：`risk_score`
- 规则执行轨迹：`rule_trace`
- 最终图：`sstg`

## 4. 当前提取逻辑

### 4.1 Slicer 在做什么

`Slicer.extract_episodes()` 当前是一个 dummy 版本：

1. 在 scene 内选择 ego
2. 枚举 ego 与其它 agent 的 TTC
3. 取最小 TTC 对应时刻作为 `T_peak`
4. 用固定窗口得到：
   - `T_start = T_peak - 2s`
   - `T_end = T_peak + 2s`
5. 从这三个时刻抓状态，构造 `Episode`

### 4.2 ego 的选择策略

当前逻辑是：

1. 优先找名字为 `ego` 的 agent
2. 如果没有，就取第一个 `VEHICLE`
3. 再不行就取第一个 agent

这个逻辑在某些真实数据集上可能并不理想。如果后续发现“自车定义不合理”，优先修改这里。

## 5. 规则系统

### 5.1 当前设计

文件：[base_rule.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/base_rule.py)

规则系统是 Strategy Pattern。

每条规则都继承 `BaseRule`，并实现：

```python
def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
    ...
```

规则只能做两件事：

- 读取 `episode`
- 修改并返回 `current_graph`

主流程不会关心某条规则内部怎么计算，只按注册顺序执行。

### 5.2 当前已有规则

#### SpatialROIRule

文件：[spatial_rules.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/spatial_rules.py)

当前行为：

- 在 `T_start / T_peak / T_end` 取 ego 周围 `roi_radius` 内的 agent
- 把这些 agent 加入图
- 加 `spatial` 边：`relation="within_roi"`
- 给同一 agent 的不同语义时刻之间加 `temporal` 边：`relation="state_transition"`

可调参数：

- `roi_radius`

#### TTCCriticalRule

文件：[dynamic_rules.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/dynamic_rules.py)

当前行为：

- 只在 `T_peak` 检查 ego 与其它 agent
- 若 `ttc < ttc_threshold`
- 加 `causal` 边：`relation="has_collision_risk"`

可调参数：

- `ttc_threshold`

### 5.3 怎样新增规则

最简单的做法：

1. 在 `src/rules/` 下新建文件，例如 `custom_rules.py`
2. 继承 `BaseRule`
3. 实现 `apply()`
4. 在 [rules/__init__.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/rules/__init__.py) 中导出
5. 在 [run_pipeline.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/run_pipeline.py) 的 `setup_pipeline()` 里注册

最小示例：

```python
from .base_rule import BaseRule
from ..core.scene_graph import Edge, EdgeType, SSTG
from ..core.slicer import Episode


class MyRule(BaseRule):
    def __init__(self, threshold: float = 1.0, enabled: bool = True):
        super().__init__(name="my_rule", enabled=enabled)
        self.threshold = threshold

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        # 1. 从 episode 读取节点
        # 2. 判断条件
        # 3. 往图里加节点/边
        return current_graph
```

### 5.4 修改规则时优先关注什么

如果你要调整“提取结果是否合理”，通常优先看这三层：

1. `Slicer`
原因：很多“抽到的片段不对”其实是 `T_peak` 不对，或者 ego 不对

2. `SpatialROIRule`
原因：很多“图里 agent 太多 / 太少”是 ROI 过滤不合适

3. `TTCCriticalRule`
原因：很多“因果边不合理 / 没解释”是 TTC 判定太松或太紧

## 6. 配置文件

文件：[rules_config.yaml](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/config/rules_config.yaml)

当前可配项主要有：

- 数据源
  - `desired_data`
  - `data_dirs`
  - `desired_dt`
- 规则
  - `rules.spatial.in_roi.enabled`
  - `rules.spatial.in_roi.roi_radius`
  - `rules.dynamic.ttc.enabled`
  - `rules.dynamic.ttc.ttc_threshold`
- slicer
  - `pre_buffer_sec`
  - `post_buffer_sec`
- 输出
  - `output_dir`

## 7. 输出结果

### 7.1 正常通过的结果

会写到：

- `libraries/risk_events/`
- `libraries/risk_elements/`

其中：

- `risk_events` 是完整 JSON 事件
- `risk_elements` 是轻量摘要

### 7.2 未通过检查的结果

会写到：

- `manual_review/`

### 7.3 shortlist 场景筛选

如果像这次一样做人工审查样本挑选，会额外生成：

- `top25_shortlist.json`

这个文件最适合先人工浏览，因为里面按风险分数排好序了。

## 8. Sanity Check 的当前规则

文件：[checker.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/src/utils/checker.py)

当前检查非常简单：

- 图中至少要有一条 `causal` 边

如果没有，就认为这个 episode 缺少“可解释的交互原因”。

这意味着当前框架其实偏向保守：

- 只要 TTC 因果边能加出来，就容易通过
- 如果未来你加了别的高价值规则，比如：
  - 急刹车
  - 让行
  - 冲突车道
  - 异常横穿
  
  那这里也应该同步升级，不然 checker 会过度依赖 TTC

## 9. 可视化

文件：[visualize_episode.py](/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/risk_mining/visualize_episode.py)

当前可视化支持：

- Bokeh 交互 HTML
- 事件切换
- `T_start / T_peak / T_end` 切换
- 地图层显示
- agent box + heading marker
- `spatial / temporal / causal` 边层切换

如果你在人工检查时觉得“提取逻辑不对”，推荐工作流是：

1. 先看 `top25_shortlist.json`
2. 再打开交互 HTML 看具体事件
3. 记录问题是来自：
   - slicer
   - ego 选择
   - ROI 过滤
   - TTC 规则
   - sanity check

## 10. 当前实现的已知限制

这套框架现在是可用的 v1，但不是最终版。当前限制包括：

- 一个 scene 目前通常只产 1 个最关键 episode
- `T_peak` 只依赖 TTC，不包含更复杂的交互触发条件
- `causal` 边目前只有 TTC 一种来源
- 规则注册还是在 `run_pipeline.py` 里显式写的，还没完全配置化
- 不同城市/路口的“ego 定义”可能还不够稳定

## 11. 推荐的下一步改造顺序

如果你接下来要继续完善规则，建议按下面顺序改：

1. 先扩展 `TTCCriticalRule`
例如更严格的接近方向判断、横向冲突判断、最小时距过滤

2. 再新增新的 `causal` 规则
例如：
   - `DecelerationCausalRule`
   - `ConflictLaneRule`
   - `YieldingRule`
   - `PedestrianCrossingRule`

3. 最后再升级 `SanityChecker`
让它不只认 TTC，而是认“任意可信 causal edge”

如果你愿意，我下一步可以继续帮你做两件事里的任意一个：

- 把这份 README 再扩成“规则开发手册”
- 直接帮你加一条新规则模板，例如 `DecelerationRule` 或 `ConflictLaneRule`
