"""
Test 3: Read Traffic Participants States at Specific Scene and Timestep

This script demonstrates how to:
1. Load a SinD scene
2. Access agent states at a specific timestep
3. Query all traffic participants in a scene
4. Filter agents by type, position, etc.
5. Visualize the scene at a specific timestep

Usage:
    python test3_read_agent_states.py
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset
from trajdata.data_structures import Scene
from trajdata.caching.df_cache import DataFrameCache


# SinD locations (cities)
SIND_LOCATIONS = ("cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR")

# Agent type colors for visualization
AGENT_TYPE_COLORS = {
    AgentType.VEHICLE: 'blue',
    AgentType.PEDESTRIAN: 'green',
    AgentType.BICYCLE: 'orange',
    AgentType.MOTORCYCLE: 'purple',
}

# Agent type markers for visualization
AGENT_TYPE_MARKERS = {
    AgentType.VEHICLE: 's',  # square
    AgentType.PEDESTRIAN: 'o',  # circle
    AgentType.BICYCLE: '^',  # triangle
    AgentType.MOTORCYCLE: 'd',  # diamond
}


def print_agent_info(agent_metadata, states_dict: Dict[int, np.ndarray]):
    """Print detailed information about an agent."""
    print(f"\n  Agent: {agent_metadata.name}")
    print(f"    Type: {agent_metadata.type.name}")
    print(f"    Extent: L={agent_metadata.extent.length:.2f}m, "
          f"W={agent_metadata.extent.width:.2f}m, H={agent_metadata.extent.height:.2f}m")
    print(f"    Timestep Range: {agent_metadata.first_timestep} -> {agent_metadata.last_timestep}")

    # Print state at each timestep
    if agent_metadata.name in states_dict:
        print(f"    States:")
        for ts, state in states_dict[agent_metadata.name].items():
            # state is a StateArray with properties like position, velocity, heading
            x, y = state.position[0], state.position[1]
            vx, vy = state.velocity[0], state.velocity[1]
            heading = state.heading[0]
            speed = np.sqrt(vx**2 + vy**2)
            print(f"      ts={ts}: pos=({x:.2f}, {y:.2f}), vel=({vx:.2f}, {vy:.2f}), "
                  f"speed={speed:.2f}m/s, heading={heading:.2f}rad")


def visualize_scene_timestep(
    scene: Scene,
    cache_path: Path,
    timestep: int,
    vector_map=None,
    title: str = "Scene at Timestep"
):
    """Visualize all agents in a scene at a specific timestep."""
    print(f"\n=== Visualizing Scene at Timestep {timestep} ===")
    print("(Visualization simplified - state reading is the main functionality)")
    print("Scene visualization is skipped for now.")


def main():
    # Configuration
    cache_path = Path("~/.unified_data_cache").expanduser()

    # UPDATE THIS PATH to point to your SinD dataset
    sind_data_dir = Path("/path/to/SinD_dataset")

    # Scene and timestep to query
    scene_idx = 0  # Scene index (change to explore different scenes)
    query_timestep = 20  # Timestep to query (change to explore different timesteps)

    print(f"=== SinD Agent State Query Test ===")
    print(f"Scene Index: {scene_idx}")
    print(f"Query Timestep: {query_timestep}")

    # Create UnifiedDataset (lightweight, just for scene access)
    dataset = UnifiedDataset(
        desired_data=["sind"],
        data_dirs={"sind": str(sind_data_dir)},
        desired_dt=0.1,
        centric="agent",
        history_sec=(0.5, 0.5),
        future_sec=(0.5, 0.5),
        agent_interaction_distances=defaultdict(lambda: 100.0),
        num_workers=0,
        verbose=True,
    )

    if len(dataset) == 0:
        print("\nNo scenes found. Please check the data directory path.")
        return

    # Get the scene
    scene: Scene = dataset.get_scene(scene_idx)

    print(f"\n=== Scene Information ===")
    print(f"Name: {scene.name}")
    print(f"Location: {scene.location}")
    print(f"Data Split: {scene.data_split}")
    print(f"Length: {scene.length_timesteps} timesteps")
    print(f"dt: {scene.dt} seconds")

    # Get agent metadata from scene (already loaded from cache)
    agent_list = scene.agents
    agent_presence = scene.agent_presence

    print(f"\n=== Agent Information ===")
    print(f"Total agents in scene: {len(agent_list)}")

    # Group agents by type
    agents_by_type: Dict[AgentType, List] = defaultdict(list)
    for agent in agent_list:
        agents_by_type[agent.type].append(agent)

    for agent_type, agents in agents_by_type.items():
        print(f"{agent_type.name}: {len(agents)} agents")

    # Get agents present at query timestep
    if query_timestep >= len(agent_presence):
        print(f"\nTimestep {query_timestep} is out of range (max: {len(agent_presence) - 1})")
        query_timestep = len(agent_presence) - 1
        print(f"Using timestep {query_timestep} instead")

    agents_at_ts = agent_presence[query_timestep]

    print(f"\n=== Agents Present at Timestep {query_timestep} ===")
    print(f"Total: {len(agents_at_ts)} agents")

    # Load detailed state data from cache using DataFrameCache
    scene_cache = DataFrameCache(cache_path, scene)

    # Collect states for each agent
    states_dict: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)

    if len(agents_at_ts) > 0:
        # Get states for all agents at this timestep
        agent_ids = [agent.name for agent in agents_at_ts]
        states = scene_cache.get_states(agent_ids, query_timestep)

        for agent, state in zip(agents_at_ts, states):
            states_dict[agent.name][query_timestep] = state

    # Print detailed info for each agent at this timestep
    for agent in agents_at_ts:
        print_agent_info(agent, states_dict)

    # Load and visualize map
    print(f"\n=== Loading Map ===")
    try:
        from trajdata import MapAPI
        map_api = MapAPI(cache_path)
        map_id = f"sind:{scene.location}"
        vector_map = map_api.get_map(map_id)
        print(f"Loaded map: {map_id}")
    except Exception as e:
        print(f"Could not load map: {e}")
        vector_map = None

    # Visualize the scene at this timestep
    visualize_scene_timestep(
        scene,
        cache_path,
        query_timestep,
        vector_map,
        title=f"SinD Scene - Agents at Timestep {query_timestep}"
    )

    # Optional: Query specific agent types
    print(f"\n=== Filtering by Agent Type ===")

    # Example: Get all vehicles
    vehicles = [a for a in agents_at_ts if a.type == AgentType.VEHICLE]
    print(f"Vehicles at timestep {query_timestep}: {len(vehicles)}")
    for v in vehicles[:5]:  # Print first 5 only
        print(f"  - {v.name}")
    if len(vehicles) > 5:
        print(f"  ... and {len(vehicles) - 5} more")

    # Example: Get all pedestrians
    pedestrians = [a for a in agents_at_ts if a.type == AgentType.PEDESTRIAN]
    print(f"Pedestrians at timestep {query_timestep}: {len(pedestrians)}")
    for p in pedestrians:
        print(f"  - {p.name}")


def query_multiple_timesteps():
    """Example: Query the same agents across multiple timesteps."""
    cache_path = Path("~/.unified_data_cache").expanduser()
    sind_data_dir = Path("/path/to/SinD_dataset")

    dataset = UnifiedDataset(
        desired_data=["sind"],
        data_dirs={"sind": str(sind_data_dir)},
        desired_dt=0.1,
        centric="agent",
        history_sec=(0.5, 0.5),
        future_sec=(0.5, 0.5),
        num_workers=0,
        verbose=False,
    )

    scene = dataset.get_scene(0)
    scene_cache = SceneCache(cache_path, scene.env_metadata.name)
    agent_df = scene_cache.load_agent_data(scene)

    if agent_df is None:
        return

    # Track an agent across timesteps
    agent_list, _ = dataset.get_agent_info(scene, cache_path, SceneCache)

    if agent_list:
        agent = agent_list[0]  # Track first agent
        print(f"\n=== Tracking Agent {agent.name} Across Timesteps ===")
        print(f"Agent Type: {agent.type.name}")
        print(f"Timestep Range: {agent.first_timestep} -> {agent.last_timestep}")

        # Get states for this agent across all timesteps
        agent_states = agent_df.loc[agent.name, :]

        print(f"\nTimesteps for {agent.name}:")
        for ts, row in agent_states.iterrows():
            x, y = row['x'], row['y']
            vx, vy = row['vx'], row['vy']
            speed = np.sqrt(vx**2 + vy**2)
            print(f"  ts={ts}: pos=({x:.2f}, {y:.2f}), speed={speed:.2f}m/s, "
                  f"heading={row['heading']:.2f}")


if __name__ == "__main__":
    main()

    # Uncomment to see agent trajectories across multiple timesteps
    # query_multiple_timesteps()
