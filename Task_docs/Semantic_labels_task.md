<!--
 * @Author: Yunwei Li 1084087910@qq.com
 * @Date: 2026-05-09 18:24:57
 * @LastEditors: Yunwei Li 1084087910@qq.com
 * @LastEditTime: 2026-05-09 18:36:30
 * @FilePath: /My_trajdata/Task_docs/Semantic_labels_task.md
 * @Description: 
 * 
 * Copyright (c) 2026 by Tsinghua University, All Rights Reserved. 
-->
## 任务说明

现在需要向项目中实现  场景语义标签标注 
我们已经提取了一批高风险场景，有“高风险-MprTTC较低（风险较高）” “视野遮挡场景” “可行域狭窄场景”
以后可能会有更多丰富的语义标注

目前希望这些场景共用的标注包括：time window， ego_id

然后，每个场景可能有自己特殊的标签，例如“视野遮挡”会标注遮挡车id，被遮挡车id；“高风险场景”可能会标注MprTTC的值和关键冲突他车POV 等

需要设计代码实现方案，可能是用这样的json来标注。

"scenario_id": "XA_INT1_0001",
      "time_window": {
        "start_frame": 12500,
        "end_frame": 12650, 
        "duration_sec": 6.0
      },
      "agents": {
        "ego_id": 45,       // 被测对象（比如高风险的主动方）"challenger_id": 89 // 交互对象
      },
      "semantics": {...



然后需要有相应的读取这些场景的工具，例如我以后会有一个需求就是让RiskIDM等被测算法在这些语义场景下测试。

## 需要实现

1. 场景语义标注存储方案和工具链 

2. 对现有的几类场景实现标注json存储，其中三类场景都在/media/lyw/KESU/sind-extract/SinD_Valued_Scenario_extract 下，但是可能组织的方式不一样，需要理解并形成规范的json标注，保存在项目合适位置中（可以是datasets/SinD_dataset 中新建一个Semantic_labels/）

3. 相应的读取这些场景的工具，例如我以后会有一个需求就是让RiskIDM等被测算法在这些语义场景下测试。

## 当前实现

- 统一标签目录：`datasets/SinD_dataset/Semantic_labels/`
- 标签格式：`scenario_id` + `time_window` + `agents` + `semantic_tags` + `semantics` + `source`
- 导入脚本：`semantic_labels/import_sind_semantic_labels.py`
- 校验脚本：`semantic_labels/validate_sind_semantic_labels.py`
- 测试工具链接入：`Simulation_test_toolchain/core/config.py` 支持 `scenario.semantic_label_id`

## 当前数据规模

当前 full 模式导出的标签库已完成，并通过 trajdata scene 校验：

- total: `53901`
- `high_risk_mprttc`: `31005`
- `visual_shielding`: `1286`
- `narrow_feasible_area`: `21610`

说明：

- `high_risk_mprttc` 是按 event pkl 全量展开。
- `visual_shielding` 是按遮挡关系 key 全量展开。
- `narrow_feasible_area` 是按场景目录和 ego 记录展开，因此数量最大。
- 后续新增语义类型时，只需要新增导入器中的一个分支并保持同一 JSON schema。
