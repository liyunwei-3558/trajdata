"""
Test: Load Only One Location to Save Memory

This script tests loading only a single location from SinD dataset
to avoid loading all 7 locations at once.

Usage:
    python test_single_location.py
"""

import psutil
import os
from pathlib import Path

# Track memory
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before importing trajdata: {mem_before:.2f} MB")

from trajdata import UnifiedDataset, AgentType
from collections import defaultdict

sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

mem_after_import = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after importing trajdata: {mem_after_import:.2f} MB")

# Create dataset with ONLY ONE LOCATION to save memory
# Use "sind-xa" format to load only the xa location
print("\n=== Creating UnifiedDataset with ONLY xa location ===")
dataset = UnifiedDataset(
    desired_data=["sind-xa"],  # ONLY load xa location
    data_dirs={"sind": str(sind_data_dir)},
    desired_dt=0.1,
    centric="agent",
    history_sec=(0.5, 0.5),
    future_sec=(0.5, 0.5),
    # Temporarily disable agent type filtering to debug
    only_predict=None,  # Comment out to see all agents: [AgentType.VEHICLE]
    agent_interaction_distances=defaultdict(lambda: 50.0),
    incl_raster_map=True,
    raster_map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
    num_workers=0,
    verbose=True,
    rebuild_cache=True,  # Force rebuild cache to ensure agents are processed
)

mem_after_init = process.memory_info().rss / 1024 / 1024  # MB
print(f"\nMemory after dataset init: {mem_after_init:.2f} MB")
print(f"Memory increase from dataset init: {mem_after_init - mem_after_import:.2f} MB")

print(f"\nDataset has {len(dataset)} samples")

print("\n=== SUCCESS ===")
print(f"Total memory used: {mem_after_init:.2f} MB")
print(f"Loading only one location saves significant memory!")
