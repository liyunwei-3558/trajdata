# SinD Dataset Integration Documentation

## Overview

This document describes the integration of the **SinD (Signalized Intersections)** dataset into the trajdata framework. SinD is a Chinese intersection dataset containing vehicle, pedestrian, and bicycle trajectories at 7 signalized intersections.

**Original Paper:** SinD: A Large-Scale Multi-Agent Dataset for Heterogeneous Traffic at Signalized Intersections

**Project Location:** `/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/`

---

## 1. Original SinD Data Structure

### 1.1 File Organization

```
/path/to/SinD_dataset/
    ├── cc/                              # Location 1 (Changchun)
    │   ├── tp_info_cc.pkl               # Trajectory-point info (~300-600MB)
    │   ├── frame_data_cc.pkl            # Frame-level metadata
    │   └── cc_map.json                  # Map data (or in output_json/)
    ├── xa/                              # Location 2 (Xi'an)
    │   ├── tp_info_xa.pkl
    │   ├── frame_data_xa.pkl
    │   └── xa_map.json
    ├── cqNR/                            # Location 3
    ├── tj/                              # Location 4 (Tianjin)
    ├── cqIR/                            # Location 5
    ├── xasl/                            # Location 6
    └── cqR/                             # Location 7
```

### 1.2 SinD Locations (Used as Splits)

Unlike typical datasets with train/val/test splits, SinD uses **locations as splits**:

| Location | Name                | Description |
|----------|---------------------|-------------|
| `cc`     | Changchun           | Changchun city |
| `xa`     | Xi'an               | Xi'an city |
| `cqNR`   | Chongqing Non-Red  | Chongqing (non-red light) |
| `tj`     | Tianjin             | Tianjin city |
| `cqIR`   | Chongqing Infra Red| Chongqing (infrastructure red) |
| `xasl`   | Xi'an Second Loop  | Xi'an second loop |
| `cqR`    | Chongqing Red      | Chongqing (red light) |

### 1.3 Raw Data Format

#### tp_info_{location}.pkl

Dictionary mapping: `scene_id -> track_id -> track_data`

```python
{
    "xa_4.12 morning 2 xa": {
        "4": {
            "Type": "mv",                    # Broad category (motor vehicle)
            "Class": "car",                  # Specific type (PRIORITIZED)
            "Length": 4.5,                   # Meters
            "Width": 2.0,                    # Meters
            "InitialFrame": 0.0,             # NOTE: Not reliable
            "FinalFrame": 220.0,             # NOTE: Can be wrong
            "State": pd.DataFrame            # Time-series data
        },
        ...
    }
}
```

#### State DataFrame

| Column | Description | Notes |
|--------|-------------|-------|
| `track_id` | Unique agent identifier | String |
| `frame_id` | Frame index | **NOT 0-indexed, can have gaps** |
| `timestamp_ms` | Timestamp in milliseconds | |
| `agent_type` | Agent type string | From "Type" field |
| `x`, `y` | Position (meters) | |
| `vx`, `vy` | Velocity (m/s) | |
| `ax`, `ay` | Acceleration (m/s²) | May be NaN |
| `yaw_rad` | Heading (radians) | |
| `heading_rad` | Heading (radians) | Preferred over yaw |
| `length`, `width` | Agent size (m) | |

#### {location}_map.json

```json
{
    "drivable_area": [[[x1,y1], [x2,y2], ...]],    // Curbstone lines
    "pedestrian_area": [[[x1,y1], [x2,y2], ...]], // Walkway boundaries
    "road_divider": [[[x1,y1], [x2,y2], ...]],     // Road dividers
    "lane_divider": [[[x1,y1], [x2,y2], ...]]      // Lane dividers
}
```

**Important:** Despite "area" names, these are **polylines (lines)**, not filled polygons.

---

## 2. Data Processing Workflow

### 2.1 Complete Pipeline

```
Raw SinD Data
     ↓
[1] Lazy Loading (SindObject)
     ↓
[2] Scene Discovery (_get_matching_scenes_from_obj)
     ↓
[3] Scene Metadata Creation (SindSceneRecord)
     ↓
[4] Map Caching (cache_maps)
     ↓
[5] Agent Info Caching (get_agent_info)
     ↓
[6] DataFrame Cache (save_agent_data)
     ↓
Cached Data (Ready for DataLoader)
```

### 2.2 Detailed Steps

#### Step 1: Lazy Loading

**File:** `src/trajdata/dataset_specific/sind/sind_utils.py:68-99`

**Purpose:** Avoid loading 3-4GB of pickle files into memory simultaneously

```python
class SindObject:
    # Only stores file paths initially
    self.city_info: Dict[str, SindCityInfo] = {
        "xa": SindCityInfo(
            location="xa",
            tp_info_path=Path(".../tp_info_xa.pkl"),
            frame_data_path=Path(".../frame_data_xa.pkl"),
            map_path=Path(".../xa_map.json")
        )
    }

    # Data loaded on-demand
    def _load_city_data(self, location: str) -> SindCityData:
        # Loads pickle only when first accessed
        # Caches result in self._city_data_cache
```

**Memory Management:**
```python
# After caching maps for a location, free memory
dataset_obj.unload_city("xa")
```

#### Step 2: Scene Discovery

**File:** `src/trajdata/dataset_specific/sind/sind_dataset.py:82-152`

```python
def _get_matching_scenes_from_obj(self, scene_tag, env_cache):
    # 1. Filter by location from scene_tag (e.g., "sind-xa")
    tag_locations = [s for s in SIND_LOCATIONS if s in scene_tag]

    for location in dataset_obj.locations:
        if tag_locations and location not in tag_locations:
            continue  # Skip non-matching locations

        # 2. Get scene names from pickle
        scene_names = dataset_obj._get_scene_names_from_pickle(location)

        for scene_id in scene_names:
            scene_name = f"{location}_{scene_id}"
            scene_length = dataset_obj.get_scene_length(scene_name)

            if scene_length <= 1:
                continue  # Skip empty scenes

            # 3. Create record
            record = SindSceneRecord(
                name=scene_name,
                location=location,
                length=scene_length,
                split=location,
                data_idx=idx
            )
```

**Scene Name Format:** `"{location}_{scene_id}"` (e.g., `"xa_4.12 morning 2 xa"`)

#### Step 3: Agent Type Mapping

**File:** `src/trajdata/dataset_specific/sind/sind_utils.py:317-334`

**Two-tier mapping system:**

| SinD Type (Broad) | SinD Class (Specific) | trajdata AgentType |
|-------------------|----------------------|-------------------|
| `mv` (机动车)     | `car`, `truck`, `bus` | `VEHICLE` |
| `nmv` (非机动车)  | `bicycle`, `tricycle` | `BICYCLE` |
| -                 | `motorcycle`          | `MOTORCYCLE` |
| -                 | `pedestrian`          | `PEDESTRIAN` |

**Priority:** `Class` field is used over `Type` for accurate mapping.

#### Step 4: Timestep Range Correction (CRITICAL FIX)

**File:** `src/trajdata/dataset_specific/sind/sind_dataset.py:285-292`

**Problem:** SinD's `InitialFrame`/`FinalFrame` values are unreliable:
- `FinalFrame` can be weird float like `22022.02202202202`
- Values don't match actual `frame_id` range in State DataFrame

**Solution:** Compute from actual data:

```python
# Determine actual frame range from State DataFrame
state_df = tp_data.get("State")
first_frame = int(state_df["frame_id"].min())
last_frame = int(state_df["frame_id"].max())

# Update agent metadata
agent_metadata.first_timestep = first_frame
agent_metadata.last_timestep = last_frame
```

#### Step 5: Map Caching

**File:** `src/trajdata/dataset_specific/sind/sind_utils.py:381-415`

**Transformations:**

| SinD Element | trajdata Element | Transformation |
|--------------|------------------|----------------|
| `drivable_area` | `RoadArea` | Polylines closed to form polygons |
| `pedestrian_area` | `PedWalkway` | Polylines closed to form polygons |
| `road_divider` | `RoadLane` | Treated as lane centerlines |
| `lane_divider` | `RoadLane` | Treated as lane centerlines |

**Map ID Format:** `"sind:{location}"` (e.g., `"sind:xa"`)

#### Step 6: Agent Presence Construction

**File:** `src/trajdata/dataset_specific/sind/sind_dataset.py:265-295`

```python
# Initialize agent_presence list
agent_presence: List[List[AgentMetadata]] = [
    [] for _ in range(scene.length_timesteps)
]

# For each frame in agent's trajectory
for _, row in state_df.iterrows():
    frame_id = int(row["frame_id"])

    # Skip if out of range
    if frame_id >= scene.length_timesteps:
        continue

    # Add agent to presence list
    agent_presence[frame_id].append(agent_metadata)
```

**Result:** `agent_presence[timestep]` returns list of agents present at that timestep.

---

## 3. Key Data Structures

### 3.1 Scene

**File:** `src/trajdata/data_structures/scene_metadata.py`

```python
class Scene:
    env_metadata: EnvMetadata    # Dataset-level metadata
    name: str                    # "xa_4.12 morning 2 xa"
    location: str                # "xa" (also used as map name)
    data_split: str              # "xa" (same as location for SinD)
    length_timesteps: int        # Number of timesteps
    raw_data_idx: int            # Index in original dataset
    data_access_info: Any        # Unused for cached data

    # Populated when loaded from cache:
    agents: List[AgentMetadata]
    agent_presence: List[List[AgentMetadata]]  # agents[timestep]

    @property
    def dt(self) -> float:
        return self.env_metadata.dt  # 0.1 for SinD
```

### 3.2 AgentMetadata

**File:** `src/trajdata/data_structures/agent.py`

```python
class AgentMetadata:
    name: str                    # track_id (e.g., "4", "292")
    type: AgentType              # VEHICLE, PEDESTRIAN, BICYCLE, MOTORCYCLE
    first_timestep: int          # First frame_id where agent appears
    last_timestep: int           # Last frame_id where agent appears
    extent: FixedExtent          # (length, width, height)
```

**IMPORTANT:** Access agent type via `.type` (NOT `.agent_type`):

```python
# CORRECT
if agent.type == AgentType.VEHICLE:
    ...

# WRONG - agent.agent_type doesn't exist
```

### 3.3 DataFrame Cache

**File:** `src/trajdata/caching/df_cache.py`

**Storage Format:** Feather (fast columnar storage)

**Index:** MultiIndex `(agent_id, scene_ts)`

**Columns:** `x, y, z, vx, vy, ax, ay, heading, length, width, height`

**File Path:** `{cache_path}/sind/{scene_name}/agent_data_dt0.10.feather`

---

## 4. Special Handling for SinD Integration

### 4.1 Memory Optimization

**Problem:** All 7 location pickle files = 3-4GB RAM

**Solution:** Lazy loading + selective caching

```python
# Option 1: Load only specific location
dataset = UnifiedDataset(
    desired_data=["sind-xa"],  # Only xa location
    ...
)

# Option 2: Load all (not recommended - uses 3-4GB RAM)
dataset = UnifiedDataset(
    desired_data=["sind"],  # All locations
    ...
)
```

### 4.2 Heading Computation Priority

**File:** `src/trajdata/dataset_specific/sind/sind_dataset.py:305-312`

```python
# Priority order for heading:
# 1. heading_rad (preferred)
if "heading_rad" in row and pd.notna(row["heading_rad"]):
    heading = float(row["heading_rad"])
# 2. yaw_rad
elif "yaw_rad" in row and pd.notna(row["yaw_rad"]):
    heading = float(row["yaw_rad"])
# 3. Computed from velocity
else:
    heading = np.arctan2(vy, vx)
```

### 4.3 Acceleration Computation

**File:** `src/trajdata/dataset_specific/sind/sind_dataset.py:348-364`

```python
# If not provided or all NaN, compute from velocity
if "ax" not in df.columns or df["ax"].isna().all():
    df[["ax", "ay"]] = (
        arr_utils.agent_aware_diff(
            df[["vx", "vy"]].to_numpy(),
            df.index.get_level_values(0),  # agent_ids
        ) / self.metadata.dt
    )
```

### 4.4 Scene Length Calculation Fix

**File:** `src/trajdata/dataset_specific/sind/sind_utils.py:284-301`

**CRITICAL FIX:** Removed incorrect use of `Frame_nums` (count of frames)

```python
# BEFORE (BUGGY):
for tp_id, tp_data in tp_info.items():
    if "Frame_nums" in tp_data:
        max_frame = max(max_frame, tp_data["Frame_nums"])  # WRONG!
    elif "State" in tp_data:
        state_df = tp_data["State"]
        max_frame = max(max_frame, state_df["frame_id"].max())

# AFTER (FIXED):
for tp_id, tp_data in tp_info.items():
    if "State" in tp_data:
        state_df = tp_data["State"]
        max_frame = max(max_frame, state_df["frame_id"].max())
```

---

## 5. Cache Structure

### 5.1 Directory Layout

```
~/.unified_data_cache/sind/
    ├── scenes_list.dill                    # Scene records (13 scenes for xa)
    ├── maps/                               # Cached vector maps
    │   ├── xa_rtrees.dill
    │   ├── xa_kdtrees.dill
    │   └── xa_2.00px_m.dill
    └── {scene_name}/                       # Per-scene cache
        ├── scene_metadata_dt0.10.dill      # Scene object with agent_presence
        └── agent_data_dt0.10.feather       # State data DataFrame
```

### 5.2 Scene Metadata Cache

**File:** `{scene_name}/scene_metadata_dt{dt:.2f}.dill`

**Contains:** Scene object with populated `agents` and `agent_presence`

**Loaded via:**
```python
from trajdata.caching.env_cache import EnvCache
scene = EnvCache.load(scene_path)
```

### 5.3 DataFrame Cache

**File:** `{scene_name}/agent_data_dt{dt:.2f}.feather`

**Contains:** Agent state data (positions, velocities, etc.)

**Queried via:**
```python
from trajdata.caching.df_cache import DataFrameCache
cache = DataFrameCache(cache_path, scene)
state = cache.get_state(agent_id, timestep)
```

---

## 6. Usage Examples

### 6.1 Basic Dataset Loading

```python
from trajdata import UnifiedDataset, AgentType
from pathlib import Path

dataset = UnifiedDataset(
    desired_data=["sind-xa"],              # Only xa location
    data_dirs={"sind": "/path/to/SinD_dataset"},
    desired_dt=0.1,                         # 10Hz (native SinD frequency)
    centric="agent",                        # Agent-centric data
    only_predict=[AgentType.VEHICLE],       # Only vehicles
    history_sec=(2.0, 2.0),                 # 2 seconds history
    future_sec=(4.0, 4.0),                  # 4 seconds future
)
```

### 6.2 Accessing Scenes

```python
# Get scene by index
scene = dataset.get_scene(0)

print(f"Scene: {scene.name}")
print(f"Location: {scene.location}")
print(f"Length: {scene.length_timesteps} timesteps")
print(f"dt: {scene.dt} seconds")
print(f"Number of agents: {len(scene.agents)}")
```

### 6.3 Querying Agent States

```python
from trajdata.caching.df_cache import DataFrameCache
from pathlib import Path

cache_path = Path.home() / ".unified_data_cache"
scene_cache = DataFrameCache(cache_path, scene)

# Get agents at timestep 20
agents_at_ts = scene.agent_presence[20]

# Get states for these agents
agent_ids = [agent.name for agent in agents_at_ts]
states = scene_cache.get_states(agent_ids, 20)

# Access state properties
for agent_id, state in zip(agent_ids, states):
    x, y = state.position[0], state.position[1]
    vx, vy = state.velocity[0], state.velocity[1]
    heading = state.heading[0]
```

### 6.4 Batch Loading with DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=0,
)

for batch in dataloader:
    # batch is an AgentBatch object
    print(f"Batch size: {len(batch.agent_name)}")
    # Process batch...
```

---

## 7. Important Gotchas

### 7.1 Scene Length Calculation

**Issue:** Different agents have different `frame_id` ranges (not 0-indexed)

**Example from xa_4.12 morning 4 xa:**
- Agent 11: frame_id 0-127
- Agent 6858: frame_id 9265-9957
- Scene length: 10192 (max frame_id + 1)

**Solution:** Scene length is `max(frame_id) + 1` across ALL agents

### 7.2 Frame ID Gaps

**Issue:** `frame_id` values can have gaps (non-contiguous)

**Example:** Agent may appear at frames 0, 2, 5, 10 (skipping 1, 3, 4, 6-9)

**Solution:** `agent_presence[frame_id]` only lists agents actually present at that frame

### 7.3 Agent Type Field

**Issue:** AgentMetadata uses `.type`, not `.agent_type`

```python
# CORRECT
agent.type  # AgentType enum

# WRONG
agent.agent_type  # AttributeError!
```

### 7.4 Memory Management

**Always unload city data after caching:**

```python
# After caching maps
dataset.dataset_obj.unload_city("xa")
```

### 7.5 Location vs Split

For SinD, `location == split`. Each location is treated as its own split.

```python
scene.location  # "xa"
scene.data_split  # "xa"
```

---

## 8. File Reference

| Component | File Path |
|-----------|-----------|
| Dataset implementation | `src/trajdata/dataset_specific/sind/sind_dataset.py` |
| Utilities (SindObject, map conversion) | `src/trajdata/dataset_specific/sind/sind_utils.py` |
| Scene records (NamedTuple) | `src/trajdata/dataset_specific/scene_records.py` |
| Agent structures | `src/trajdata/data_structures/agent.py` |
| Scene structures | `src/trajdata/data_structures/scene_metadata.py` |
| DataFrame cache | `src/trajdata/caching/df_cache.py` |
| Scene cache | `src/trajdata/caching/scene_cache.py` |
| Environment cache | `src/trajdata/caching/env_cache.py` |
| Batch structures | `src/trajdata/data_structures/batch.py` |
| Test scripts | `SinD_integration_test_scripts/` |

---

## 9. Testing

### 9.1 Test Scripts

Located in `SinD_integration_test_scripts/`:

| Script | Purpose |
|--------|---------|
| `test1_map_visualization.py` | Map loading and visualization |
| `test2_batch_visualization.py` | Batch data loading and trajectory viz |
| `test3_read_agent_states.py` | Query agent states at specific timesteps |

### 9.2 Running Tests

```bash
# Activate environment
conda activate trajdata

# Run tests
python SinD_integration_test_scripts/test1_map_visualization.py
python SinD_integration_test_scripts/test2_batch_visualization.py
python SinD_integration_test_scripts/test3_read_agent_states.py
```

---

## 10. Summary of Key Fixes

During integration, the following bugs were identified and fixed:

1. **`_get_matching_scenes_from_cache`** - Changed to create Scene objects from records instead of loading non-existent dill files

2. **`get_agent_metadata`** - Fixed unreliable timestep initialization by computing from actual DataFrame

3. **`get_scene_length`** - Fixed incorrect use of `Frame_nums` (count) instead of actual `frame_id` values

These fixes ensure correct scene construction and data access for all agents and timesteps.
