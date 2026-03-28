"""
Unit tests for core SSTG data structures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Edge, EdgeType, Node, SSTG


def test_node_serialization_round_trip():
    node = Node(
        agent_id="ego",
        timestamp="T_peak",
        type="VEHICLE",
        position=(1.0, 2.0),
        velocity=(3.0, 4.0),
        acceleration=(0.5, 0.0),
        heading=1.2,
        extent=(4.5, 2.0, 1.5),
        metadata={"raw_timestep": 10},
    )

    rebuilt = Node.from_dict(node.to_dict())
    assert rebuilt == node
    assert rebuilt.speed == 5.0


def test_edge_serialization_round_trip():
    edge = Edge(
        source_id="ego",
        target_id="agent_1",
        source_timestamp="T_peak",
        target_timestamp="T_peak",
        edge_type=EdgeType.CAUSAL,
        weight=0.9,
        relation="has_collision_risk",
        metadata={"ttc": 1.5},
    )

    rebuilt = Edge.from_dict(edge.to_dict())
    assert rebuilt == edge


def test_sstg_add_and_query_nodes_and_edges():
    sstg = SSTG(scene_id="scene_1", dt=0.1)
    ego_start = Node("ego", "T_start", "VEHICLE", (1.0, 0.0), (0.0, 0.0), position=(0.0, 0.0))
    ego_peak = Node("ego", "T_peak", "VEHICLE", (1.0, 0.0), (0.0, 0.0), position=(2.0, 0.0))
    other_peak = Node("agent_1", "T_peak", "VEHICLE", (-1.0, 0.0), (0.0, 0.0), position=(8.0, 0.0))

    for node in (ego_start, ego_peak, other_peak):
        sstg.add_node(node)

    sstg.add_edge(
        Edge(
            source_id="ego",
            target_id="ego",
            source_timestamp="T_start",
            target_timestamp="T_peak",
            edge_type=EdgeType.TEMPORAL,
            relation="state_transition",
        )
    )
    sstg.add_edge(
        Edge(
            source_id="ego",
            target_id="agent_1",
            source_timestamp="T_peak",
            target_timestamp="T_peak",
            edge_type=EdgeType.SPATIAL,
            relation="within_roi",
        )
    )

    assert sstg.has_node("ego", "T_start")
    assert len(sstg.get_nodes_at_timestamp("T_peak")) == 2
    assert len(sstg.get_edges_at_timestamp("T_peak")) == 2


def test_sstg_to_dict_from_dict():
    sstg = SSTG(scene_id="scene_2", dt=0.2, metadata={"env_name": "test"})
    sstg.add_node(Node("ego", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(0.0, 0.0)))
    payload = sstg.to_dict()
    rebuilt = SSTG.from_dict(payload)

    assert rebuilt.scene_id == "scene_2"
    assert rebuilt.dt == 0.2
    assert rebuilt.metadata["env_name"] == "test"
    assert len(rebuilt.get_nodes_at_timestamp("T_peak")) == 1


def test_sstg_summary_uses_canonical_edge_types():
    sstg = SSTG(scene_id="scene_3", dt=0.1)
    sstg.add_node(Node("ego", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(0.0, 0.0)))
    sstg.add_node(Node("agent_1", "T_peak", "PEDESTRIAN", (0.0, 0.0), (0.0, 0.0), position=(3.0, 0.0)))
    sstg.add_edge(
        Edge(
            source_id="ego",
            target_id="agent_1",
            source_timestamp="T_peak",
            target_timestamp="T_peak",
            edge_type=EdgeType.CAUSAL,
            relation="has_collision_risk",
        )
    )

    summary = sstg.get_summary()
    assert summary["node_counts_by_type"]["VEHICLE"] == 1
    assert summary["node_counts_by_type"]["PEDESTRIAN"] == 1
    assert summary["edge_counts_by_type"][EdgeType.CAUSAL.value] == 1
