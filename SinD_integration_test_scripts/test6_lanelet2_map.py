"""Test Lanelet2 map parsing for SinD dataset.

This script tests the Lanelet2 map integration for SinD dataset locations
that have Lanelet2 OSM files (tj and cqNR).
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from trajdata import UnifiedDataset, AgentType
from trajdata.dataset_specific.sind.sind_lanelet2_utils import (
    lanelet2_map_to_vector_map,
    get_lanelet2_map_path,
)
from trajdata.maps.vec_map_elements import MapElementType


def test_lanelet2_map_loading():
    """Test loading Lanelet2 maps directly (without using UnifiedDataset)."""
    print("\n" + "=" * 60)
    print("Test 1: Direct Lanelet2 Map Loading")
    print("=" * 60)

    # Test locations with Lanelet2 maps
    lanelet2_locations = ["tj", "cqNR"]

    for location in lanelet2_locations:
        print(f"\n--- Testing {location} ---")

        # Get the Lanelet2 map path
        map_path = get_lanelet2_map_path(location, "")
        if map_path is None:
            print(f"  No Lanelet2 map found for {location}")
            continue

        print(f"  Lanelet2 map path: {map_path}")

        # Load the vector map
        vector_map = lanelet2_map_to_vector_map(f"sind:{location}", str(map_path))

        # Print map extent
        print(f"  Map extent: {vector_map.extent}")

        # Print number of lanes
        road_lanes = vector_map.elements.get(MapElementType.ROAD_LANE, {})
        print(f"  Number of lanes: {len(road_lanes)}")

        # Print lane connectivity for first few lanes
        count = 0
        for lane_id, lane in road_lanes.items():
            if count >= 3:  # Only print first 3 lanes
                break
            print(f"\n  Lane {lane_id}:")
            print(f"    Next lanes: {lane.next_lanes}")
            print(f"    Prev lanes: {lane.prev_lanes}")
            print(f"    Adj left: {lane.adj_lanes_left}")
            print(f"    Adj right: {lane.adj_lanes_right}")
            count += 1


def test_unified_dataset_with_lanelet2():
    """Test using UnifiedDataset with Lanelet2 map flag."""
    print("\n" + "=" * 60)
    print("Test 2: UnifiedDataset with Lanelet2 Maps")
    print("=" * 60)

    # NOTE: Update this path to your actual SinD dataset directory
    data_dir = str(Path.home() / "1TBSSD" / "Datasets" / "SinD_dataset")

    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"WARNING: SinD dataset directory not found: {data_dir}")
        print("Skipping UnifiedDataset test.")
        return

    # Test with Lanelet2 maps enabled
    print("\n--- Testing with use_lanelet2_maps=True ---")
    dataset_lanelet2 = UnifiedDataset(
        desired_data=["sind-tj"],
        data_dirs={"sind": data_dir},
        desired_dt=0.1,
        centric="agent",
        only_predict=[AgentType.VEHICLE],
        incl_robot_future=False,
        incl_history=True,
        history_sec=(2.0, 0.0),
        future_sec=(6.0, 0.0),
        agent_interaction_distances=defaultdict(lambda: 50.0),
        map_params={"use_lanelet2_maps": True},
        num_workers=0,
        verbose=True,
    )

    print(f"Dataset loaded with {len(dataset_lanelet2)} scenes")


def test_visualize_lanelet2_map():
    """Test visualization of Lanelet2 map."""
    print("\n" + "=" * 60)
    print("Test 3: Lanelet2 Map Visualization")
    print("=" * 60)

    # Test tj location
    location = "tj"
    map_path = get_lanelet2_map_path(location, "")

    if map_path is None:
        print(f"No Lanelet2 map found for {location}")
        return

    print(f"Loading Lanelet2 map: {map_path}")
    vector_map = lanelet2_map_to_vector_map(f"sind:{location}", str(map_path))

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 12))

    road_lanes = vector_map.elements.get(MapElementType.ROAD_LANE, {})

    # Plot each lane
    for lane_id, lane in road_lanes.items():
        # Plot centerline
        center_pts = lane.center.ps
        ax.plot(center_pts[:, 0], center_pts[:, 1], "b-", linewidth=1, alpha=0.5)

        # Plot left edge
        if lane.left_edge is not None:
            left_pts = lane.left_edge.ps
            ax.plot(left_pts[:, 0], left_pts[:, 1], "g-", linewidth=0.5, alpha=0.3)

        # Plot right edge
        if lane.right_edge is not None:
            right_pts = lane.right_edge.ps
            ax.plot(right_pts[:, 0], right_pts[:, 1], "r-", linewidth=0.5, alpha=0.3)

    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"{location} Lanelet2 Map ({len(road_lanes)} lanes)")
    ax.grid(True, alpha=0.3)

    # Save to file
    save_path = Path(__file__).parent / f"test6_lanelet2_{location}_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")
    plt.close()


def test_json_vs_lanelet2():
    """Compare JSON map vs Lanelet2 map for tj location."""
    print("\n" + "=" * 60)
    print("Test 4: JSON vs Lanelet2 Map Comparison")
    print("=" * 60)

    from trajdata.dataset_specific.sind.sind_utils import sind_map_to_vector_map

    location = "tj"

    # Load Lanelet2 map
    lanelet2_path = get_lanelet2_map_path(location, "")
    if lanelet2_path is None:
        print(f"No Lanelet2 map found for {location}")
        return

    print(f"Loading Lanelet2 map from: {lanelet2_path}")
    lanelet2_map = lanelet2_map_to_vector_map(f"sind:{location}", str(lanelet2_path))

    # Load JSON map (from dataset if available)
    data_dir = str(Path.home() / "1TBSSD" / "Datasets" / "SinD_dataset")
    json_map_path = Path(data_dir) / location / f"{location}_map.json"
    if json_map_path.exists():
        import json

        print(f"Loading JSON map from: {json_map_path}")
        with open(json_map_path, "r") as f:
            json_map_data = json.load(f)
        json_map = sind_map_to_vector_map(f"sind:{location}", json_map_data)

        # Compare extents
        print(f"\nComparison for {location}:")
        print(f"  Lanelet2 extent: {lanelet2_map.extent}")
        print(f"  JSON extent: {json_map.extent}")

        # Compare number of elements
        lanelet2_lanes = len(lanelet2_map.elements.get(MapElementType.ROAD_LANE, {}))
        json_lanes = len(json_map.elements.get(MapElementType.ROAD_LANE, {}))
        print(f"  Lanelet2 lanes: {lanelet2_lanes}")
        print(f"  JSON lanes: {json_lanes}")
    else:
        print(f"JSON map not found at: {json_map_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SinD Lanelet2 Map Integration Tests")
    print("=" * 60)

    # Run all tests
    test_lanelet2_map_loading()
    test_unified_dataset_with_lanelet2()
    test_visualize_lanelet2_map()
    test_json_vs_lanelet2()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
