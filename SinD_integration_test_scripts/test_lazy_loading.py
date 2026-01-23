"""
Test: Verify Lazy Loading Works for SinD Dataset

This script tests that the lazy loading implementation works correctly
and only loads data on-demand, saving memory.

Usage:
    python test_lazy_loading.py
"""

import psutil
import os
from pathlib import Path

# Track memory before importing trajdata
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before importing trajdata: {mem_before:.2f} MB")

from trajdata import UnifiedDataset, AgentType
from collections import defaultdict

sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

mem_after_import = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after importing trajdata: {mem_after_import:.2f} MB")
print(f"Memory increase from imports: {mem_after_import - mem_before:.2f} MB")

# Create dataset with lazy loading - should NOT load all pickle files
print("\n=== Creating UnifiedDataset (lazy loading) ===")
dataset = UnifiedDataset(
    desired_data=["sind"],
    data_dirs={"sind": str(sind_data_dir)},
    desired_dt=0.1,
    centric="agent",
    history_sec=(0.5, 0.5),
    future_sec=(0.5, 0.5),
    agent_interaction_distances=defaultdict(lambda: 50.0),
    incl_raster_map=True,
    raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
    num_workers=0,
    verbose=True,
)

mem_after_init = process.memory_info().rss / 1024 / 1024  # MB
print(f"\nMemory after dataset init: {mem_after_init:.2f} MB")
print(f"Memory increase from dataset init: {mem_after_init - mem_after_import:.2f} MB")

print(f"\nDataset has {len(dataset)} samples")

# Get one scene to trigger actual data loading
print("\n=== Loading first scene (triggers data loading) ===")
scene = dataset.get_scene(0)
print(f"Scene: {scene.name}, Location: {scene.location}")

mem_after_scene = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after loading scene: {mem_after_scene:.2f} MB")
print(f"Memory increase from scene loading: {mem_after_scene - mem_after_init:.2f} MB")

# Unload cached data to free memory
print("\n=== Unloading cached city data ===")
dataset.envs[0].dataset_obj.unload_all()

mem_after_unload = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after unloading: {mem_after_unload:.2f} MB")
print(f"Memory freed: {mem_after_scene - mem_after_unload:.2f} MB")

print("\n=== Summary ===")
print(f"Total memory used: {mem_after_unload:.2f} MB")
print(f"If lazy loading works correctly, memory should be < 1000 MB")
print(f"(old implementation would use 3000+ MB)")
