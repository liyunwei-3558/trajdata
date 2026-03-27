"""
Unit tests for core data structures.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core import SSTG, Node, Edge, EdgeType


def test_node_creation():
    """Test Node creation and properties."""
    node = Node(
        agent_id="agent_1",
        agent_type="VEHICLE",
        timestep=0,
        position=(10.0, 20.0),
        velocity=(5.0, 3.0),
        acceleration=(0.5, 0.2),
        heading=1.57,
        extent=(4.5, 2.0),
    )

    assert node.agent_id == "agent_1"
    assert node.agent_type == "VEHICLE"
    assert node.timestep == 0

    # Test properties
    speed = node.speed
    expected_speed = np.sqrt(5.0**2 + 3.0**2)
    assert abs(speed - expected_speed) < 0.01

    accel_mag = node.acceleration_magnitude
    expected_accel = np.sqrt(0.5**2 + 0.2**2)
    assert abs(accel_mag - expected_accel) < 0.01

    print("✓ test_node_creation passed")


def test_node_serialization():
    """Test Node to_dict and from_dict."""
    node = Node(
        agent_id="agent_1",
        agent_type="VEHICLE",
        timestep=0,
        position=(10.0, 20.0),
        velocity=(5.0, 3.0),
        acceleration=(0.5, 0.2),
        heading=1.57,
        extent=(4.5, 2.0),
    )

    # Serialize
    data = node.to_dict()

    # Deserialize
    node2 = Node.from_dict(data)

    assert node2.agent_id == node.agent_id
    assert node2.agent_type == node.agent_type
    assert node2.timestep == node.timestep
    assert node2.position == node.position
    assert node2.velocity == node.velocity
    assert node2.acceleration == node.acceleration
    assert abs(node2.heading - node.heading) < 0.001
    assert node2.extent == node.extent

    print("✓ test_node_serialization passed")


def test_edge_creation():
    """Test Edge creation."""
    edge = Edge(
        source_id="agent_1",
        target_id="agent_2",
        edge_type=EdgeType.SPATIAL_PROXIMITY,
        weight=0.8,
        ttc=2.5,
        distance=15.0,
    )

    assert edge.source_id == "agent_1"
    assert edge.target_id == "agent_2"
    assert edge.edge_type == EdgeType.SPATIAL_PROXIMITY
    assert edge.weight == 0.8
    assert edge.ttc == 2.5
    assert edge.distance == 15.0

    print("✓ test_edge_creation passed")


def test_edge_serialization():
    """Test Edge to_dict and from_dict."""
    edge = Edge(
        source_id="agent_1",
        target_id="agent_2",
        edge_type=EdgeType.INTERACTION,
        weight=0.9,
        ttc=1.5,
        distance=8.0,
    )

    # Serialize
    data = edge.to_dict()

    # Deserialize
    edge2 = Edge.from_dict(data)

    assert edge2.source_id == edge.source_id
    assert edge2.target_id == edge.target_id
    assert edge2.edge_type == edge.edge_type
    assert edge2.weight == edge.weight
    assert edge2.ttc == edge.ttc
    assert edge2.distance == edge.distance

    print("✓ test_edge_serialization passed")


def test_sstg_creation():
    """Test SSTG creation and basic operations."""
    sstg = SSTG(scene_id="test_scene", dt=0.1)

    # Add nodes
    node1 = Node(
        agent_id="agent_1",
        agent_type="VEHICLE",
        timestep=0,
        position=(0.0, 0.0),
        velocity=(5.0, 0.0),
        acceleration=(0.0, 0.0),
        heading=0.0,
        extent=(4.5, 2.0),
    )

    node2 = Node(
        agent_id="agent_2",
        agent_type="VEHICLE",
        timestep=0,
        position=(10.0, 0.0),
        velocity=(4.0, 0.0),
        acceleration=(0.0, 0.0),
        heading=0.0,
        extent=(4.5, 2.0),
    )

    sstg.add_node(node1)
    sstg.add_node(node2)

    # Add edge
    edge = Edge(
        source_id="agent_1",
        target_id="agent_2",
        edge_type=EdgeType.SPATIAL_PROXIMITY,
        weight=0.5,
        distance=10.0,
    )
    sstg.add_edge(edge, timestep=0)

    # Check properties
    assert sstg.scene_id == "test_scene"
    assert sstg.dt == 0.1
    assert len(sstg.timesteps) == 1
    assert sstg.timesteps[0] == 0

    # Get nodes at timestep
    nodes = sstg.get_nodes_at_timestep(0)
    assert len(nodes) == 2

    # Get edges at timestep
    edges = sstg.get_edges_at_timestep(0)
    assert len(edges) == 1

    print("✓ test_sstg_creation passed")


def test_sstg_serialization():
    """Test SSTG to_dict and from_dict."""
    sstg = SSTG(scene_id="test_scene", dt=0.1)

    # Add nodes and edges
    node1 = Node(
        agent_id="agent_1",
        agent_type="VEHICLE",
        timestep=0,
        position=(0.0, 0.0),
        velocity=(5.0, 0.0),
        acceleration=(0.0, 0.0),
        heading=0.0,
        extent=(4.5, 2.0),
    )

    sstg.add_node(node1)

    edge = Edge(
        source_id="agent_1",
        target_id="agent_1",
        edge_type=EdgeType.INTERACTION,
        weight=1.0,
        ttc=3.0,
    )
    sstg.add_edge(edge, timestep=0)

    # Serialize
    data = sstg.to_dict()

    # Deserialize
    sstg2 = SSTG.from_dict(data)

    assert sstg2.scene_id == sstg.scene_id
    assert sstg2.dt == sstg.dt
    assert len(sstg2.timesteps) == len(sstg.timesteps)

    print("✓ test_sstg_serialization passed")


def test_sstg_summary():
    """Test SSTG get_summary."""
    sstg = SSTG(scene_id="test_scene", dt=0.1)

    # Add multiple nodes
    for i in range(3):
        node = Node(
            agent_id=f"agent_{i}",
            agent_type="VEHICLE" if i < 2 else "PEDESTRIAN",
            timestep=0,
            position=(float(i * 10), 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        )
        sstg.add_node(node)

    summary = sstg.get_summary()

    assert summary["scene_id"] == "test_scene"
    assert summary["num_timesteps"] == 1
    assert summary["num_nodes"] == 3
    assert summary["node_counts_by_type"]["VEHICLE"] == 2
    assert summary["node_counts_by_type"]["PEDESTRIAN"] == 1

    print("✓ test_sstg_summary passed")


if __name__ == "__main__":
    print("Running core tests...")
    test_node_creation()
    test_node_serialization()
    test_edge_creation()
    test_edge_serialization()
    test_sstg_creation()
    test_sstg_serialization()
    test_sstg_summary()
    print("\n✅ All core tests passed!")
