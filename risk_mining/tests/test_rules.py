"""
Unit tests for the strategy-based rule engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import EdgeType, Episode, EpisodeType, Node, SSTG
from src.rules import BaseRule, RuleRegistry, SpatialROIRule, TTCCriticalRule


def build_episode() -> Episode:
    return Episode(
        scene_id="env:scene",
        scene_name="scene",
        env_name="env",
        dt=0.1,
        ego_agent_id="ego",
        t_start=8,
        t_peak=10,
        t_end=12,
        involved_agents=["ego", "agent_1", "agent_far"],
        episode_type=EpisodeType.TTC_MIN_WINDOW,
        risk_score=0.6,
        state_snapshots={
            "T_start": {
                "ego": Node("ego", "T_start", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(0.0, 0.0)),
                "agent_1": Node("agent_1", "T_start", "VEHICLE", (-5.0, 0.0), (0.0, 0.0), position=(20.0, 0.0)),
            },
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(10.0, 0.0)),
                "agent_1": Node("agent_1", "T_peak", "VEHICLE", (-10.0, 0.0), (0.0, 0.0), position=(20.0, 0.0)),
                "agent_far": Node("agent_far", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(200.0, 0.0)),
            },
            "T_end": {
                "ego": Node("ego", "T_end", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(20.0, 0.0)),
                "agent_1": Node("agent_1", "T_end", "VEHICLE", (-10.0, 0.0), (0.0, 0.0), position=(10.0, 0.0)),
            },
        },
    )


def test_rule_registry_register_and_unregister():
    registry = RuleRegistry()

    class DummyRule(BaseRule):
        def __init__(self):
            super().__init__("dummy")

        def apply(self, episode, current_graph):
            current_graph.metadata["dummy"] = True
            return current_graph

    rule = DummyRule()
    registry.register(rule)
    assert registry.get_rule("dummy") is rule
    assert registry.get_enabled_rules() == [rule]
    registry.unregister("dummy")
    assert registry.get_rule("dummy") is None


def test_spatial_roi_rule_adds_spatial_and_temporal_edges():
    episode = build_episode()
    graph = SpatialROIRule(roi_radius=50.0).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    peak_nodes = graph.get_nodes_at_timestamp("T_peak")
    assert {node.agent_id for node in peak_nodes} == {"ego", "agent_1"}

    edge_types = {edge.edge_type for edge in graph.get_edges_at_timestamp("T_peak")}
    assert EdgeType.SPATIAL in edge_types
    assert EdgeType.TEMPORAL in edge_types


def test_ttc_rule_adds_causal_edge_at_peak():
    episode = build_episode()
    registry = RuleRegistry()
    registry.register(SpatialROIRule(roi_radius=50.0))
    registry.register(TTCCriticalRule(ttc_threshold=2.5))

    graph = registry.apply_all(episode)
    episode.sstg = graph

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert len(causal_edges) == 1
    assert causal_edges[0].relation == "has_collision_risk"
    assert episode.rule_trace == ["spatial_roi", "ttc_critical"]
