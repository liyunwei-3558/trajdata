"""
Test 2: Batch Scene Visualization for SinD Dataset

This script demonstrates how to:
1. Load SinD dataset using UnifiedDataset
2. Create a DataLoader to batch the data
3. Visualize agent batches with history, current state, and future trajectories
4. Include map visualization in the background

Usage:
    python test2_batch_visualization.py
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_agent_batch


# SinD locations (cities)
SIND_LOCATIONS = ("cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR")


def main():
    # Configuration
    # UPDATE THIS PATH to point to your SinD dataset
    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

    # Select which location(s) to use (can be multiple)
    # Options: "cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR"
    # TIP: To save memory, load only one location at a time
    selected_locations = ["xa"]  # Xi'an

    # Dataset configuration
    history_sec = (3.0, 3.0)  # 3 seconds of history
    future_sec = (4.0, 4.0)   # 4 seconds of future

    print(f"=== SinD Batch Visualization Test ===")
    print(f"Locations: {selected_locations}")
    print(f"History: {history_sec[0]}s, Future: {future_sec[0]}s")
    print(f"NOTE: Using lazy loading to save memory")

    # Create UnifiedDataset
    # Use specific location tags to load only selected cities
    desired_data_tags = [f"sind-{loc}" for loc in selected_locations]
    dataset = UnifiedDataset(
        desired_data=desired_data_tags,  # Only load selected locations
        data_dirs={
            "sind": str(sind_data_dir),
        },
        desired_dt=0.1,  # 10Hz
        centric="agent",  # Agent-centric data
        history_sec=history_sec,
        future_sec=future_sec,
        only_predict=[AgentType.VEHICLE],  # Only predict vehicles
        agent_interaction_distances=defaultdict(lambda: 50.0),  # 50m radius for neighbors
        incl_robot_future=False,  # No robot/ego vehicle in SinD
        incl_raster_map=True,  # Include rasterized map
        raster_map_params={
            "px_per_m": 2,  # 2 pixels per meter
            "map_size_px": 224,  # 224x224 pixel map
            "offset_frac_xy": (-0.5, 0.0),  # Center map on agent
        },
        num_workers=0,  # Single-threaded for debugging
        verbose=True,
    )

    print(f"\nDataset loaded successfully!")
    print(f"Number of samples: {len(dataset):,}")

    if len(dataset) == 0:
        print("\nNo samples found. Please check:")
        print("1. The data directory path is correct")
        print("2. The SinD dataset files are properly structured")
        print("3. The selected location has data")
        return

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Process 4 agents at a time
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,  # Single-threaded for debugging
    )

    print(f"\nStarting batch visualization...")
    print(f"Close the plot window to see the next batch.")

    # Visualize first few batches
    num_batches_to_plot = 3

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if batch_idx >= num_batches_to_plot:
            break

        batch: AgentBatch
        print(f"\n=== Batch {batch_idx + 1} ===")
        print(f"Batch size: {len(batch.agent_name)} agents")

        # Print batch information
        for i in range(len(batch.agent_name)):
            agent_name = batch.agent_name[i]
            agent_type = AgentType(batch.agent_type[i].item()).name
            num_neigh = batch.num_neigh[i].item()

            # Get current state
            curr_state = batch.curr_agent_state[i]
            x, y = curr_state.position.cpu().numpy()
            vx, vy = curr_state.velocity.cpu().numpy()
            speed = (vx**2 + vy**2)**0.5

            print(f"  Agent {i}: {agent_name} ({agent_type})")
            print(f"    Position: ({x:.2f}, {y:.2f})")
            print(f"    Speed: {speed:.2f} m/s")
            print(f"    Neighbors: {int(num_neigh)}")

        # Visualize the first agent in the batch
        print(f"\nVisualizing agent 0 of this batch...")
        fig, axes = plt.subplots(1, len(batch.agent_name), figsize=(6*len(batch.agent_name), 6))
        if len(batch.agent_name) == 1:
            axes = [axes]

        for i in range(len(batch.agent_name)):
            ax = axes[i]
            plot_agent_batch(
                batch,
                batch_idx=i,
                ax=ax,
                show=False,
                close=False,
            )
            agent_name = batch.agent_name[i]
            agent_type = AgentType(batch.agent_type[i].item()).name
            ax.set_title(f"{agent_name} ({agent_type})")

        plt.suptitle(f"SinD Batch {batch_idx + 1}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save the figure
        output_path = Path(__file__).parent / f"sind_batch_{batch_idx + 1}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

        plt.show()

    print(f"\n=== Visualization Complete ===")
    print(f"Visualized {min(num_batches_to_plot, len(dataloader))} batches")


def test_specific_scene():
    """Alternative: Visualize a specific scene by index."""
    # Configuration
    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

    dataset = UnifiedDataset(
        desired_data=["sind-xa"],  # Only load xa city
        data_dirs={"sind": str(sind_data_dir)},
        desired_dt=0.1,
        centric="agent",
        history_sec=(2.0, 2.0),
        future_sec=(3.0, 3.0),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 50.0),
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=0,
        verbose=True,
    )

    # Get a specific scene
    scene_idx = 0  # Change this to visualize different scenes
    scene = dataset.get_scene(scene_idx)
    print(f"Scene: {scene.name}")
    print(f"Location: {scene.location}")
    print(f"Length: {scene.length_timesteps} timesteps")

    # Create dataloader for this scene only
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    # Get first batch
    batch = next(iter(dataloader))
    plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()
    # To test a specific scene, uncomment the following line:
    # test_specific_scene()
