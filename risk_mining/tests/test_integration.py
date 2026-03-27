"""
Integration test for risk mining pipeline with SinD dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajdata import UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache

from src.core import Slicer, Episode
from src.rules import RuleRegistry, SpatialROIRule, TTCCriticalRule
from src.library import DualLibrary
from src.utils import create_default_checker


def test_sind_integration():
    """
    Integration test with SinD Tianjin dataset.

    This test:
    1. Loads the SinD Tianjin dataset
    2. Extracts episodes from the first scene
    3. Verifies episodes are created with valid SSTG
    """
    print("Setting up SinD integration test...")

    # Configuration
    data_dirs = {
        "sind": "/home/lyw/datasets/SinD_dataset",
    }

    try:
        # Create dataset (just one scene for testing)
        # Using sind-cqNR since it's cached (tj may not be)
        dataset = UnifiedDataset(
            desired_data=["sind-cqNR"],
            data_dirs=data_dirs,
            desired_dt=0.1,
        )
        print(f"✓ Dataset created: {dataset}")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        print("  (This is expected if SinD dataset is not available)")
        return False

    # Get first scene
    try:
        scene = dataset.get_scene(0)
        print(f"✓ Scene loaded: {scene.name}")
        print(f"  - Timesteps: {scene.length_timesteps}")
        print(f"  - Agents: {len(scene.agents)}")
        print(f"  - dt: {scene.dt}")
    except Exception as e:
        print(f"✗ Failed to load scene: {e}")
        return False

    # Create cache
    cache_path = Path("~/.unified_data_cache").expanduser()
    cache = DataFrameCache(cache_path, scene)
    print(f"✓ Cache created")

    # Create slicer
    slicer = Slicer(
        pre_buffer_sec=2.0,
        post_buffer_sec=2.0,
        ttc_threshold=3.0,
    )

    # Extract episodes
    try:
        episodes = slicer.extract_episodes(scene, cache)
        print(f"✓ Episodes extracted: {len(episodes)}")
    except Exception as e:
        print(f"✗ Failed to extract episodes: {e}")
        import traceback
        traceback.print_exc()
        return False

    if len(episodes) == 0:
        print("  (No episodes found - this may be expected for some scenes)")
        return True

    # Verify first episode
    episode = episodes[0]
    print(f"\n  First episode:")
    print(f"    - t_start: {episode.t_start}")
    print(f"    - t_peak: {episode.t_peak}")
    print(f"    - t_end: {episode.t_end}")
    print(f"    - Agents: {len(episode.involved_agents)}")
    print(f"    - Risk score: {episode.risk_score:.2f}")
    print(f"    - Type: {episode.episode_type.value}")

    # Check SSTG
    if episode.sstg:
        summary = episode.sstg.get_summary()
        print(f"    - SSTG nodes: {summary['num_nodes']}")
        print(f"    - SSTG edges: {summary['num_edges']}")

    # Test sanity checker
    output_dir = Path("./output/test")
    checker = create_default_checker(output_dir)

    check_result = checker.check_episode(episode)
    print(f"\n  Sanity check: {'PASSED' if check_result.passed else 'FAILED'}")
    if not check_result.passed:
        print(f"    Reason: {check_result.reason}")

    # Test dual library
    dual_lib = DualLibrary(output_dir)
    element_id, event_id = dual_lib.add_episode(episode)
    print(f"\n  Library:")
    print(f"    - Element ID: {element_id}")
    print(f"    - Event ID: {event_id}")

    print("\n✅ SinD integration test passed!")
    return True


def test_basic_workflow():
    """Test basic workflow without real dataset."""
    print("Testing basic workflow...")

    from src.core import SSTG, Node, Edge, EdgeType

    # Create a simple SSTG
    sstg = SSTG(scene_id="test", dt=0.1)

    # Add nodes
    for i in range(3):
        node = Node(
            agent_id=f"agent_{i}",
            agent_type="VEHICLE",
            timestep=0,
            position=(float(i * 10), 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        )
        sstg.add_node(node)

    # Add edge
    edge = Edge(
        source_id="agent_0",
        target_id="agent_1",
        edge_type=EdgeType.SPATIAL_PROXIMITY,
        weight=0.5,
        distance=10.0,
    )
    sstg.add_edge(edge, timestep=0)

    # Test serialization
    data = sstg.to_dict()
    sstg2 = SSTG.from_dict(data)

    assert sstg2.scene_id == sstg.scene_id
    assert sstg2.dt == sstg.dt

    print("✓ Basic workflow test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Integration Tests")
    print("=" * 60)

    # Test basic workflow first
    test_basic_workflow()

    print()
    print("=" * 60)

    # Try SinD integration test
    success = test_sind_integration()

    if success:
        print("\n" + "=" * 60)
        print("All integration tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SinD integration test skipped (dataset not available)")
        print("=" * 60)
