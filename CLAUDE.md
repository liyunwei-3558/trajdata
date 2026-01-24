# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**trajdata** is a unified interface to multiple autonomous driving trajectory datasets (nuScenes, Waymo, Argoverse 2, Lyft, nuPlan, INTERACTION, ETH/UCY pedestrians, Stanford Drone Dataset, View-of-Delft). Published at NeurIPS 2023.

The library provides a two-stage architecture: optional preprocessing (caching) + data loading via PyTorch DataLoader.

## Installation

```bash
# Use conda environment (recommended)
conda activate trajdata

# Developer installation
pip install -e .

# With dataset-specific extras
pip install "trajdata[nusc,waymo,interaction,av2,lyft,vod]"
```

## Development Commands

```bash
# Use conda environment
conda activate trajdata
# OR: conda run -n trajdata <command>

# Run all tests
pytest tests/

# Run tests in parallel
pytest -n auto tests/

# Run specific test
python -m unittest tests/test_state.py

# Run SinD integration tests (from repository root)
python SinD_integration_test_scripts/test1_map_visualization.py
python SinD_integration_test_scripts/test2_batch_visualization.py
python SinD_integration_test_scripts/test3_read_agent_states.py

# Code formatting (uses black and isort)
black src/trajdata
isort src/trajdata

# Build package
python -m build
```

## Architecture

### Core Entry Point

**`src/trajdata/dataset.py`** (1173 lines) - Contains `UnifiedDataset`, the main class that:
- Accepts `desired_data` list of dataset identifiers (e.g., `["nusc_mini-mini_train"]`)
- Handles multi-dataset loading with automatic `dt` interpolation
- Returns PyTorch-compatible batches via `get_collate_fn()`

### Dataset-Specific Implementations

**`src/trajdata/dataset_specific/`** - Contains dataset implementations:

1. **`raw_dataset.py`** - Abstract `RawDataset` base class with methods to implement:
   - `compute_metadata(env_name, data_dir) -> EnvMetadata`
   - `load_dataset_obj(verbose=False)`
   - `_get_matching_scenes_from_obj(...) -> List[SceneMetadata]`
   - `_get_matching_scenes_from_cache(...) -> List[Scene]`
   - `get_scene(scene_info) -> Scene`
   - `get_agent_info(scene, cache_path, cache_class) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]`
   - `cache_maps(cache_path, map_cache_class, map_params)`

2. **Scene Records** (`scene_records.py`) - Each dataset defines a `NamedTuple` for cached scene info

3. **Individual dataset folders** (e.g., `nusc/`, `waymo/`, `sind/`) - Contain:
   - `__init__.py` - Exports dataset class
   - `{dataset}_dataset.py` - Main `RawDataset` subclass
   - `{dataset}_utils.py` - Helper functions (data loading, map conversion)

### Data Structures

**`src/trajdata/data_structures/`**:
- `agent.py` - `AgentType` enum, `AgentMetadata`, `FixedExtent`
- `scene.py` - `Scene`, `SceneMetadata`, `SceneTimeAgent`
- `batch.py` - `AgentBatch`, `SceneBatch`
- `state.py` - `StateArray`, `StateTensor` types

### Caching System

**`src/trajdata/caching/`** - Multi-stage caching system:
- Cache location: `~/.unified_data_cache/` by default
- **IMPORTANT**: `_get_matching_scenes_from_cache` should return Scene objects created from cached records, NOT loaded from `.dill` files. The Scene `.dill` files are only created later during agent info caching.
- `env_cache.py` - Environment-level scene list caching (`scenes_list.dill`)
- `scene_cache.py` - Scene-level agent data and map caching
- `df_cache.py` - DataFrame cache implementation (Feather format for fast columnar access)

**Cache Flow**:
1. `_get_matching_scenes_from_obj` creates `SceneMetadata` and saves scene records to `scenes_list.dill`
2. `_get_matching_scenes_from_cache` creates Scene objects from records (for loading from cache)
3. `cache_maps` caches vector maps
4. `get_agent_info` processes raw data, saves to `.feather` files, and saves Scene with `agent_presence` to `.dill` files

**Common gotcha**: When implementing `_get_matching_scenes_from_cache`, follow the pattern from nusc_dataset.py - create Scene objects directly from the cached records using the scene metadata fields, do not try to load non-existent Scene `.dill` files.

### Map System

**`src/trajdata/maps/`** - Vector map API:
- `vec_map.py` - `VectorMap` class with spatial indexing (KD-trees, R-trees)
- `vec_map_elements.py` - `RoadLane`, `RoadArea`, `PedCrosswalk`, `PedWalkway`, `Polyline`

### Registration

**`src/trajdata/utils/env_utils.py`** - `get_raw_dataset()` function registers all datasets:
```python
if "dataset_name" in dataset_name:
    from trajdata.dataset_specific.dataset import DatasetClass
    return DatasetClass(dataset_name, data_dir, parallelizable=True/False, has_maps=True/False)
```

## Adding New Datasets

To add a new dataset (from README lines 125-134):

1. Create folder under `src/trajdata/dataset_specific/` with:
   - `__init__.py` exporting the dataset class
   - `{name}_dataset.py` with `RawDataset` subclass
   - Optional `{name}_utils.py` for helpers

2. Add `NamedTuple` to `scene_records.py` for scene metadata

3. Add section to `DATASETS.md` with directory structure

4. Register in `env_utils.py` with `parallelizable` and `has_maps` flags

**Key implementation detail**: `parallelizable=False` if dataset object loads large data into memory (like nuScenes), preventing use in multiprocessing dataloaders.

## Agent Types

`AgentType` enum: `VEHICLE`, `PEDESTRIAN`, `BICYCLE`, `MOTORCYCLE`, `OTHER`

Map dataset-specific agent types to these in dataset utils (see `av2_utils.py` pattern).

## State Format

Default: `"x,y,xd,yd,xdd,ydd,h"` (position, velocity, acceleration, heading)

## Scene Tags

Datasets use hyphen-separated tags: `"{dataset_name}-{split}-{location}"` (e.g., `"nusc_mini-mini_train-boston"`)

## SinD Dataset Usage

SinD (Signalized Intersections) is a Chinese intersection dataset with 7 locations. Each location functions as a split.

### Documentation Files
- `SIND_INTEGRATION.md` - Complete integration documentation with architecture and gotchas
- `SIND_USAGE_WORKFLOW.md` - Step-by-step usage guide

### Data Structure
```
/path/to/SinD_dataset/
    ├── cc/
    │   ├── tp_info_cc.pkl
    │   ├── frame_data_cc.pkl
    │   ├── cc_map.json
    │   └── cc_map.osm (Lanelet2 format, optional)
    ├── xa/
    │   ├── tp_info_xa.pkl
    │   ├── frame_data_xa.pkl
    │   ├── xa_map.json
    │   └── xa_map.osm (Lanelet2 format, optional)
    ├── Lanelet_maps_SinD/     # Lanelet2 OSM files for all locations
    │   ├── lanelet2_tj.osm
    │   ├── lanelet2_cqNR.osm
    │   ├── lanelet2_cc.osm
    │   └── ... (one per location)
    └── ... (cqNR, tj, cqIR, xasl, cqR)
```

### Lanelet2 Map Support

SinD supports two map formats:
- **JSON format** (default): Simple drivable/pedestrian areas and dividers
- **Lanelet2 OSM format** (optional): Rich lane information with connectivity relations

To enable Lanelet2 maps:

```python
dataset = UnifiedDataset(
    desired_data=["sind-tj"],
    data_dirs={"sind": "/path/to/SinD_dataset"},
    map_params={"use_lanelet2_maps": True},  # Enable Lanelet2
)
```

Lanelet2 maps provide:
- **RoadLane** elements with centerlines and edge boundaries
- **PedCrosswalk** elements (convex hull of zebra_marking pairs)
- **Lane connectivity** (next/prev lanes, adjacent lanes)

The OSM files should be placed in:
```
/path/to/SinD_dataset/Lanelet_maps_SinD/lanelet2_{location}.osm
```

### Supported Agent Types
- VEHICLE: car, truck, bus, mv (motor vehicle/机动车)
- PEDESTRIAN: pedestrian
- BICYCLE: bicycle, tricycle, nmv (non-motor vehicle/非机动车)
- MOTORCYCLE: motorcycle

### Important Implementation Details

**Lazy Loading**: SinD uses lazy loading to avoid loading all 7 pickle files (300-600MB each) into memory at once. The `SindObject` class only stores file paths initially (`SindCityInfo`), and loads city data on-demand (`_load_city_data()`). After caching maps for a location, call `unload_city(location)` to free memory.

**Agent Type Field**: SinD data has both `Type` (broad category like "mv" for motor vehicle) and `Class` (specific type like "car"). The implementation prioritizes `Class` for type mapping to correctly identify vehicles.

**Location Filtering**: To load only specific locations (to save memory), use the format `desired_data=["sind-xa"]` instead of `desired_data=["sind"]`. This filters scenes and caches only the specified location.

**CRITICAL: Timestep Range Issues**: SinD's `InitialFrame`/`FinalFrame` values are unreliable (can be weird floats like 22022.02202202202). The implementation computes `first_timestep` and `last_timestep` from actual `frame_id` values in State DataFrame. Scene length is calculated as `max(frame_id) + 1` across ALL agents, not using `Frame_nums` (which is a count).

**NaN Extent Handling**: SinD data may contain NaN values for `Length`/`Width`. The implementation validates these values and falls back to default extents when they are invalid.

### Test Scripts
See `SinD_integration_test_scripts/` for examples:
- `test1_map_visualization.py` - Map loading and visualization
- `test2_batch_visualization.py` - Batch data loading and trajectory visualization
- `test3_read_agent_states.py` - Query agent states at specific timesteps
- `test4_bokeh_interactive.py` - Interactive Bokeh visualization with Lanelet2 maps
- `test6_lanelet2_map.py` - Lanelet2 map parsing tests

### Accessing Agent State Data

To query agent states from a cached scene, use `DataFrameCache`:

```python
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures import Scene

# Scene is obtained from dataset.get_scene() and has agents pre-loaded
scene_cache = DataFrameCache(cache_path, scene)

# Get states for agents at a timestep
agent_ids = [agent.name for agent in agents_at_ts]
states = scene_cache.get_states(agent_ids, query_timestep)

# Access state properties (returns StateArray)
x, y = state.position[0], state.position[1]
vx, vy = state.velocity[0], state.velocity[1]
heading = state.heading[0]
```

**Note**: `AgentMetadata` uses `.type` (not `.agent_type`) to access the `AgentType`.
