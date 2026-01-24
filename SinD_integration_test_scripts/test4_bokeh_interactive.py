'''
Author: Yunwei Li 1084087910@qq.com
Date: 2026-01-23 22:08:13
LastEditors: Yunwei Li 1084087910@qq.com
LastEditTime: 2026-01-24 20:07:48
FilePath: /My_trajdata/SinD_integration_test_scripts/test4_bokeh_interactive.py
Description: 

Copyright (c) 2026 by Tsinghua University, All Rights Reserved. 
'''
"""
SinD Interactive Bokeh Visualization

This script demonstrates interactive visualization of SinD dataset using trajdata's
built-in interactive visualization tools. Supports both JSON and Lanelet2 map formats.

Configuration:
    - location: "tj" (Tianjin) or "cqNR" (Chongqing-North) for Lanelet2 maps
    - map_type: "json" or "lanelet2"

Usage:
    python test4_bokeh_interactive.py
"""

from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset, AgentBatch
from tqdm import tqdm
from trajdata.visualization.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
from trajdata.visualization.interactive_vis import plot_agent_batch_interactive
from trajdata.visualization.vis import plot_agent_batch


def main():
    # Configuration
    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")
    location = "cqNR"  # Tianjin (also supports cqNR for Lanelet2)

    # Choose map type: "json" or "lanelet2"
    # Note: Lanelet2 maps are only available for "tj" and "cqNR" locations
    map_type = "lanelet2"  # Options: "json", "lanelet2"

    print("=" * 60)
    print("SinD Interactive Bokeh Visualization")
    print("=" * 60)
    print(f"Location: {location}")
    print(f"Map Type: {map_type}")
    print("=" * 60)

    # Create dataset
    use_lanelet2 = map_type.lower() == "lanelet2"

    # if use_lanelet2 and location not in ["tj", "cqNR"]:
    #     print(f"WARNING: Lanelet2 maps are only available for 'tj' and 'cqNR' locations.")
    #     print(f"         Falling back to JSON format for location '{location}'.")
    #     use_lanelet2 = False

    dataset = UnifiedDataset(
        desired_data=[f"sind-{location}"],
        data_dirs={"sind": str(sind_data_dir)},
        desired_dt=0.1,
        centric="agent",
        history_sec=(2.0, 2.0),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
            "use_lanelet2_maps": use_lanelet2,  # Enable Lanelet2 if requested
        },
        rebuild_cache=True,  # Rebuild cache to include this location
        rebuild_maps=True,   # Rebuild maps with Lanelet2 if enabled
        num_workers=0,       # Use single process to avoid OOM
        verbose=True,
    )

    print(f"\nDataset loaded! Total samples: {len(dataset):,}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Reduce batch size to avoid OOM
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,  # Use single process
    )

    # Get a batch
    batch = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("Choose visualization mode:")
    print("1. Interactive plot (slider + zoom/pan)")
    print("2. Animation (auto-play through history)")
    print("=" * 60)

    # Option 1: Interactive plot with slider
    # print("\n1. Launching interactive plot...")
    # plot_agent_batch_interactive(
    #     batch,
    #     batch_idx=0,
    #     cache_path=dataset.cache_path
    # )

    # Option 2: Animation
    # print("\n2. Creating animation...")
    # animation = InteractiveAnimation(
    #     animate_agent_batch_interactive,
    #     batch=batch,
    #     batch_idx=0,
    #     cache_path=dataset.cache_path,
    # )
    # animation.show()
    
    test_count = 0
    
    batch: AgentBatch
    for batch in tqdm(dataloader):
        plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path) # BOKEh 交互式绘图
        # plot_agent_batch(batch, batch_idx=0) # 简单绘图

        animation = InteractiveAnimation( # 动画
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=0,
            cache_path=dataset.cache_path,
        )
        animation.show() 
        
        test_count += 1
        if test_count >= 4:
            break
        
    # break


if __name__ == "__main__":
    main()
