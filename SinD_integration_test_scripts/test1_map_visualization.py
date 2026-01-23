"""
Test 1: Map Reading and Visualization for SinD Dataset

This script demonstrates how to:
1. Load a SinD map using MapAPI
2. Rasterize the vector map to an image
3. Visualize road areas and pedestrian areas
4. Query map elements (closest area, areas within radius)

Usage:
    python test1_map_visualization.py
"""

import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.utils import map_utils
from trajdata.utils import raster_utils


# SinD locations (cities)
SIND_LOCATIONS = ("cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR")

SIND_LOCATION_NAMES: Dict[str, str] = {
    "cc": "Changchun",
    "xa": "Xi'an",
    "cqNR": "Chongqing NR",
    "tj": "Tianjin",
    "cqIR": "Chongqing IR",
    "xasl": "Xi'an Shanglin",
    "cqR": "Chongqing R",
}


def main():
    # Configuration
    cache_path = Path("~/.unified_data_cache").expanduser()

    # Path to SinD dataset
    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

    # Select a location to visualize (you can change this)
    location = "cqR"  # Test cqR

    print(f"=== SinD Map Visualization Test ===")
    print(f"Location: {location} ({SIND_LOCATION_NAMES.get(location, 'Unknown')})")

    # Check if cache exists, if not, we need to preprocess first
    print(f"\nChecking cache at: {cache_path}")

    # Initialize MapAPI
    map_api = MapAPI(cache_path)

    # Map ID format: "sind:location"
    map_id = f"sind:{location}"
    print(f"Loading map: {map_id}")

    try:
        start = time.perf_counter()
        vec_map: VectorMap = map_api.get_map(
            map_id,
            incl_road_areas=True,
            incl_ped_walkways=True,
        )
        end = time.perf_counter()
        print(f"Map loading took {(end - start)*1000:.2f} ms")
    except Exception as e:
        print(f"Error loading map: {e}")
        print("\nNote: If the map is not cached, you need to preprocess first.")
        print("You can preprocess by running test2_batch_visualization.py first,")
        print("or by using the UnifiedDataset with preprocess_data=True")
        return

    print(f"\nMap Information:")
    print(f"  Map ID: {vec_map.map_id}")
    print(f"  Extent: {vec_map.extent}")
    print(f"  Number of RoadAreas: {len(vec_map.elements[MapElementType.ROAD_AREA])}")
    print(f"  Number of PedWalkways: {len(vec_map.elements[MapElementType.PED_WALKWAY])}")
    print(f"  Number of RoadLanes: {len(vec_map.elements[MapElementType.ROAD_LANE])}")

    # Rasterize the map using raster_utils which supports RoadAreas and PedWalkways
    print(f"\nRasterizing Map...")
    start = time.perf_counter()
    rasterized_map = raster_utils.rasterize_map(
        vec_map,
        resolution=2,  # pixels per meter
    )
    end = time.perf_counter()
    print(f"Map rasterization took {(end - start)*1000:.2f} ms")
    print(f"Rasterized image shape: {rasterized_map.shape}")

    # Convert rasterized map data to image format (CHW -> HWC for imshow)
    # rasterized_map.data is (C, H, W), need to transpose to (H, W, C) for imshow
    map_img = rasterized_map.data.transpose(1, 2, 0)
    raster_from_world = rasterized_map.metadata.map_from_world

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Rasterized map
    ax1 = axes[0]
    ax1.imshow(map_img, alpha=0.7, origin="lower")
    ax1.set_title(f"SinD Map: {location} ({SIND_LOCATION_NAMES.get(location, 'Unknown')}) - Rasterized")
    ax1.axis("equal")

    # Right plot: Vector map elements
    ax2 = axes[1]
    ax2.set_title(f"SinD Map: {location} - Vector Elements")

    # Draw all road areas
    print("\nDrawing road areas...")
    road_areas = vec_map.elements[MapElementType.ROAD_AREA]
    for area_id, road_area in road_areas.items():
        exterior_pts = road_area.exterior_polygon.xy
        # Transform to raster coordinates
        raster_pts = map_utils.transform_points(exterior_pts, raster_from_world)
        ax2.fill(raster_pts[:, 0], raster_pts[:, 1],
                 alpha=0.3, color='gray', edgecolor='black', linewidth=0.5)

    # Draw all pedestrian walkways
    print("Drawing pedestrian walkways...")
    ped_walkways = vec_map.elements[MapElementType.PED_WALKWAY]
    for walk_id, ped_walkway in ped_walkways.items():
        polygon_pts = ped_walkway.polygon.xy
        raster_pts = map_utils.transform_points(polygon_pts, raster_from_world)
        ax2.fill(raster_pts[:, 0], raster_pts[:, 1],
                 alpha=0.5, color='green', edgecolor='darkgreen', linewidth=0.5)

    # Draw all road lanes (from road_divider and lane_divider)
    print("Drawing road lanes...")
    road_lanes = vec_map.elements[MapElementType.ROAD_LANE]
    for lane_id, road_lane in road_lanes.items():
        center_pts = road_lane.center.xy
        raster_pts = map_utils.transform_points(center_pts, raster_from_world)
        ax2.plot(raster_pts[:, 0], raster_pts[:, 1],
                 alpha=0.6, color='blue', linewidth=1.0, linestyle='-')

    # Query example: Pick a random point and find closest area
    if len(road_areas) > 0:
        # Use center of map as query point
        center_x = (vec_map.extent[0] + vec_map.extent[3]) / 2
        center_y = (vec_map.extent[1] + vec_map.extent[4]) / 2
        query_point = np.array([center_x, center_y])

        print(f"\nQuerying map at point: ({query_point[0]:.2f}, {query_point[1]:.2f})")

        # Get closest road area
        start = time.perf_counter()
        closest_area = vec_map.get_closest_area(query_point, elem_type=MapElementType.ROAD_AREA)
        end = time.perf_counter()
        print(f"Closest road area query took {(end-start)*1000:.2f} ms")

        if closest_area is not None:
            raster_pts = map_utils.transform_points(
                closest_area.exterior_polygon.xy, raster_from_world
            )
            ax2.fill(raster_pts[:, 0], raster_pts[:, 1],
                     alpha=0.7, color='red', edgecolor='darkred', linewidth=2,
                     label='Closest Area')

        # Get road areas within 100m
        start = time.perf_counter()
        areas_within = vec_map.get_areas_within(
            query_point, elem_type=MapElementType.ROAD_AREA, dist=100.0
        )
        end = time.perf_counter()
        print(f"Areas within 100m query took {(end-start)*1000:.2f} ms")
        print(f"Found {len(areas_within)} areas within 100m")

        # Plot query point
        query_pt_raster = map_utils.transform_points(
            query_point[None, :], raster_from_world
        )
        ax2.scatter(query_pt_raster[:, 0], query_pt_raster[:, 1],
                    color='red', s=100, marker='*', zorder=10, label='Query Point')

    ax2.axis("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / f"sind_map_{location}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    # plt.show()  # Commented out to avoid blocking in batch mode
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    main()
