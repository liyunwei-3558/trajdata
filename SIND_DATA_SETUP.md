# SinD 数据集配置指南

> **面向新同学的快速上手指南** - 本文档说明如何获取、放置 SinD 数据集，以及现有的测试脚本功能。

---

## 一、SinD 数据集简介

**SinD (Signalized Intersections)** 是一个中国信号灯路口轨迹数据集，包含 7 个城市的路口数据：
- **车辆、行人、自行车** 等多种交通参与者的轨迹
- **10Hz 采样频率**（每 0.1 秒一帧）
- 每个位置约 300-600MB 数据

| 位置代码 | 城市名称 | 说明 |
|---------|---------|------|
| `cc` | 长春 | Changchun |
| `xa` | 西安 | Xi'an |
| `cqNR` | 重庆北区 | Chongqing Non-Red |
| `tj` | 天津 | Tianjin |
| `cqIR` | 重庆 | Chongqing Infra Red |
| `xasl` | 西安上林 | Xi'an Second Loop |
| `cqR` | 重庆 | Chongqing Red |

---

## 二、数据集目录结构

### 获取数据集后，请按以下结构组织文件：

```
/path/to/SinD_dataset/
├── cc/                                    # 长春
│   ├── tp_info_cc.pkl                     # 轨迹点信息 (~300-600MB)
│   ├── frame_data_cc.pkl                  # 帧级元数据
│   └── cc_map.json                        # 地图文件
├── xa/                                    # 西安
│   ├── tp_info_xa.pkl
│   ├── frame_data_xa.pkl
│   └── xa_map.json
├── cqNR/                                  # 重庆北区
│   ├── tp_info_cqNR.pkl
│   ├── frame_data_cqNR.pkl
│   └── cqNR_map.json
├── tj/                                    # 天津
│   ├── tp_info_tj.pkl
│   ├── frame_data_tj.pkl
│   └── tj_map.json
├── cqIR/                                  # 重庆
│   ├── tp_info_cqIR.pkl
│   ├── frame_data_cqIR.pkl
│   └── cqIR_map.json
├── xasl/                                  # 西安上林
│   ├── tp_info_xasl.pkl
│   ├── frame_data_xasl.pkl
│   └── xasl_map.json
└── cqR/                                   # 重庆
    ├── tp_info_cqR.pkl
    ├── frame_data_cqR.pkl
    └── cqR_map.json
```

**重要说明：**
- 每个位置包含 **3 个必要文件**：`tp_info_*.pkl`、`frame_data_*.pkl`、`*_map.json`
- 地图文件也可能在 `output_json/` 目录下，可创建符号链接

---

## 三、配置数据路径

### 方法 1：直接修改脚本中的路径

在测试脚本中找到并修改以下变量：

```python
# 将此路径修改为你的 SinD 数据集实际路径
sind_data_dir = Path("/path/to/your/SinD_dataset")
```

### 方法 2：使用环境变量（推荐）

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
export SIND_DATA_DIR="/path/to/your/SinD_dataset"
```

然后在代码中使用：

```python
import os
sind_data_dir = Path(os.environ.get("SIND_DATA_DIR", "/default/path"))
```

---

## 四、测试脚本说明

所有测试脚本位于 `SinD_integration_test_scripts/` 目录下。

### 测试脚本一览表

| 脚本 | 功能 | 适合场景 |
|------|------|---------|
| `test1_map_visualization.py` | 加载和可视化地图 | 验证地图是否正确加载 |
| `test2_batch_visualization.py` | 批量加载轨迹数据并可视化 | 查看轨迹数据样例 |
| `test3_read_agent_states.py` | 查询特定时间步的智能体状态 | 学习如何访问数据 |
| `test4_bokeh_interactive.py` | Bokeh 交互式可视化 | 交互式探索数据 |
| `test5_check_agent_types.py` | 检查智能体类型分布 | 了解数据中的智能体类型 |
| `test6_lanelet2_map.py` | 测试 Lanelet2 地图解析 | 使用高精度地图时 |

### 各脚本详细说明

#### 1. test1_map_visualization.py - 地图可视化

```python
# 功能：
# - 加载 SinD 地图（支持 7 个位置）
# - 栅格化矢量地图
# - 可视化道路区域、行人区域、车道分隔线

# 使用方法：
python test1_map_visualization.py

# 输出：
# - sind_map_{location}_visualization.png
```

**适用场景：**
- 验证地图文件是否正确放置
- 查看地图的覆盖范围和元素

#### 2. test2_batch_visualization.py - 批量轨迹可视化

```python
# 功能：
# - 使用 UnifiedDataset 加载数据
# - 创建 DataLoader 批量获取数据
# - 可视化智能体的历史轨迹、当前状态、未来轨迹

# 使用方法：
python test2_batch_visualization.py

# 输出：
# - sind_batch_1_visualization.png
# - sind_batch_2_visualization.png
# - sind_batch_3_visualization.png
```

**适用场景：**
- 学习如何使用 `UnifiedDataset` 和 `DataLoader`
- 查看实际轨迹数据的样式
- 验证智能体历史/未来窗口是否正确

#### 3. test3_read_agent_states.py - 智能体状态查询

```python
# 功能：
# - 加载场景
# - 查询特定时间步的所有智能体
# - 按类型过滤智能体
# - 获取位置、速度、加速度、航向角等状态

# 使用方法：
python test3_read_agent_states.py

# 输出：
# - 控制台打印智能体信息
```

**适用场景：**
- 学习如何访问和查询数据
- 了解 `agent_presence` 的使用方式
- 学习使用 `DataFrameCache` 获取状态

#### 4. test4_bokeh_interactive.py - 交互式可视化

```python
# 功能：
# - 使用 Bokeh 创建交互式可视化
# - 支持 JSON 和 Lanelet2 两种地图格式
# - 可播放历史轨迹动画

# 使用方法：
python test4_bokeh_interactive.py

# 输出：
# - 浏览器中打开交互式页面
```

**适用场景：**
- 交互式探索数据
- 演示和展示时使用
- 需要动态播放轨迹时

#### 5. test5_check_agent_types.py - 智能体类型检查

```python
# 功能：
# - 检查每个位置的智能体类型分布
# - 显示智能体的尺寸信息
# - 检查原始数据的 Type 和 Class 字段

# 使用方法：
python test5_check_agent_types.py

# 输出：
# - 控制台打印统计信息
```

**适用场景：**
- 了解数据集中有哪些类型的智能体
- 检查智能体尺寸是否正确
- 验证类型映射是否正确

#### 6. test6_lanelet2_map.py - Lanelet2 地图测试

```python
# 功能：
# - 测试 Lanelet2 OSM 地图加载
# - 显示车道连接关系（前后车道、左右相邻）
# - 比较 JSON 地图与 Lanelet2 地图

# 使用方法：
python test6_lanelet2_map.py

# 输出：
# - test6_lanelet2_{location}_map.png
```

**适用场景：**
- 使用高精度 Lanelet2 地图时
- 需要车道拓扑信息时
- 比较 JSON 和 Lanelet2 地图差异

---

## 五、快速开始

### 步骤 1：配置数据路径

编辑测试脚本，修改 `sind_data_dir` 变量：

```python
# 将此路径改为你的 SinD 数据集路径
sind_data_dir = Path("/your/path/to/SinD_dataset")
```

### 步骤 2：运行第一个测试

```bash
# 激活环境
conda activate trajdata

# 运行地图可视化测试
python SinD_integration_test_scripts/test1_map_visualization.py
```

### 步骤 3：检查输出

成功运行后，会在脚本目录下生成可视化图片：
```
sind_map_cc_visualization.png
```

### 步骤 4：运行更多测试

```bash
# 批量轨迹可视化
python SinD_integration_test_scripts/test2_batch_visualization.py

# 查询智能体状态
python SinD_integration_test_scripts/test3_read_agent_states.py
```

---

## 六、常见问题

### Q1: 找不到地图文件

**错误信息：** `FileNotFoundError: *.json`

**解决方法：**
- 检查地图文件是否在正确的位置
- 地图文件可能在 `output_json/` 目录下，创建符号链接：
```bash
ln -s /path/to/output_json/xa_map.json /path/to/SinD_dataset/xa/xa_map.json
```

### Q2: 内存不足

**错误信息：** `MemoryError` 或程序崩溃

**解决方法：**
- 只加载一个位置：`desired_data=["sind-xa"]`
- 设置 `num_workers=0`
- 减少 `batch_size`

### Q3: 缓存损坏

**错误信息：** 数据加载错误

**解决方法：**
```bash
# 删除缓存
rm -rf ~/.unified_data_cache/sind/

# 重新运行测试脚本
```

### Q4: 没有找到某个位置的智能体

**可能原因：**
- 该位置的数据文件缺失或不完整
- `desired_data` 标签格式错误（应为 `sind-xa` 格式）

---

## 七、参考文档

| 文档 | 内容 |
|------|------|
| `CLAUDE.md` | 项目总体说明 |
| `SIND_INTEGRATION.md` | SinD 集成技术文档 |
| `SIND_USAGE_WORKFLOW.md` | SinD 使用流程（中文） |

---

## 八、联系与支持

如有问题，请联系项目维护者或在项目 Issues 中提问。
