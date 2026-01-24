"""
Test 5: Check Agent Types in SinD Dataset

This script checks:
1. If there are pedestrians in xa location scenes
2. Agent type distribution and their extents
"""

from collections import defaultdict, Counter
from pathlib import Path

from trajdata import AgentType, UnifiedDataset


def check_agent_types_in_dataset():
    """Check agent types in SinD dataset."""
    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

    # Test different locations
    for location in ["xa", "cc", "tj"]:
        print(f"\n{'='*60}")
        print(f"Checking location: {location}")
        print(f"{'='*60}")

        dataset = UnifiedDataset(
            desired_data=[f"sind-{location}"],
            data_dirs={"sind": str(sind_data_dir)},
            desired_dt=0.1,
            centric="agent",
            history_sec=(2.0, 2.0),
            future_sec=(3.0, 3.0),
            agent_interaction_distances=defaultdict(lambda: 50.0),
            incl_robot_future=False,
            incl_raster_map=False,
            num_workers=0,
            verbose=False,
        )

        # Check all scenes
        all_agent_types = Counter()
        agents_by_type = defaultdict(list)

        for scene_idx in range(dataset.num_scenes()):
            scene = dataset.get_scene(scene_idx)

            for agent in scene.agents:
                all_agent_types[agent.type] += 1
                agents_by_type[agent.type].append({
                    "name": agent.name,
                    "scene": scene.name,
                    "type": agent.type,
                    "extent": agent.extent,
                    "first_ts": agent.first_timestep,
                    "last_ts": agent.last_timestep,
                })

        # Print results
        print(f"\nAgent type distribution for {location}:")
        print(f"  Total agents: {sum(all_agent_types.values())}")
        for agent_type, count in all_agent_types.most_common():
            print(f"  - {agent_type.name}: {count}")

        # Show sample extents for each type
        print(f"\nSample extents for each type:")
        for agent_type in agents_by_type.keys():
            sample = agents_by_type[agent_type][0]
            extent = sample["extent"]
            if hasattr(extent, "length"):
                print(f"  - {agent_type.name}: length={extent.length:.2f}m, width={extent.width:.2f}m, height={extent.height:.2f}m")
            else:
                print(f"  - {agent_type.name}: {extent}")

        # Check for pedestrians
        if AgentType.PEDESTRIAN in all_agent_types:
            print(f"\n  ✓ Found {all_agent_types[AgentType.PEDESTRIAN]} pedestrians in {location}")
        else:
            print(f"\n  ✗ No pedestrians found in {location}")

        # Show example agents of each type
        print(f"\nExample agents:")
        for agent_type in agents_by_type.keys():
            sample = agents_by_type[agent_type][0]
            print(f"  - {agent_type.name}: agent '{sample['name']}' in scene '{sample['scene']}'")


def check_raw_sind_data():
    """Check raw SinD data to see original agent types."""
    import pickle

    sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")

    print(f"\n{'='*60}")
    print("Checking Raw SinD Data (original types)")
    print(f"{'='*60}")

    for location in ["xa", "cc", "tj"]:
        city_path = sind_data_dir / location
        tp_info_path = city_path / f"tp_info_{location}.pkl"

        if not tp_info_path.exists():
            continue

        print(f"\n{location}:")
        with open(tp_info_path, "rb") as f:
            tp_info = pickle.load(f)

        # Get first scene
        first_scene = list(tp_info.keys())[0]
        scene_data = tp_info[first_scene]

        # Count original types
        type_counts = Counter()
        class_counts = Counter()

        for agent_id, agent_data in scene_data.items():
            if "Type" in agent_data:
                type_counts[agent_data["Type"]] += 1
            if "Class" in agent_data:
                class_counts[agent_data["Class"]] += 1

        print(f"  Scene: {first_scene}")
        print(f"  Original 'Type' field: {dict(type_counts)}")
        print(f"  Original 'Class' field: {dict(class_counts)}")


if __name__ == "__main__":
    print("="*60)
    print("SinD Agent Type Check")
    print("="*60)

    # Check raw data first
    check_raw_sind_data()

    # Check trajdata processed data
    check_agent_types_in_dataset()
