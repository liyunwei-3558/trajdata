"""
Unit tests for the strategy-based rule engine.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import EdgeType, Episode, EpisodeType, Node, SSTG
from src.rules import BaseRule, RuleRegistry, SpatialROIRule, TTCCriticalRule
from src.utils import create_default_checker


class FakeInterpolatedPolyline:
    def __init__(self, xyzh: np.ndarray):
        self.xyzh = xyzh


class FakePolyline:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape [N, 2]")
        deltas = np.diff(self.points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        self._cum_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def project_onto(self, xyzh: np.ndarray) -> np.ndarray:
        queries = np.asarray(xyzh, dtype=float)
        projections = []
        for query in queries:
            qx, qy = query[:2]
            best_point = None
            best_heading = 0.0
            best_distance = float("inf")
            for idx in range(len(self.points) - 1):
                start = self.points[idx]
                end = self.points[idx + 1]
                segment = end - start
                seg_norm_sq = float(np.dot(segment, segment))
                if seg_norm_sq <= 1e-9:
                    continue
                t = float(np.clip(np.dot(np.asarray([qx, qy]) - start, segment) / seg_norm_sq, 0.0, 1.0))
                projected = start + t * segment
                distance = float(np.linalg.norm(np.asarray([qx, qy]) - projected))
                if distance < best_distance:
                    best_distance = distance
                    best_point = projected
                    best_heading = float(np.arctan2(segment[1], segment[0]))
            if best_point is None:
                best_point = self.points[0]
            projections.append([best_point[0], best_point[1], 0.0, best_heading])
        return np.asarray(projections, dtype=float)

    def interpolate(self, max_dist: float = 1.0) -> FakeInterpolatedPolyline:
        samples = []
        for idx in range(len(self.points) - 1):
            start = self.points[idx]
            end = self.points[idx + 1]
            segment = end - start
            segment_length = float(np.linalg.norm(segment))
            if segment_length <= 1e-9:
                continue
            heading = float(np.arctan2(segment[1], segment[0]))
            num_steps = max(1, int(np.ceil(segment_length / max(max_dist, 1e-6))))
            for step in range(num_steps):
                t = step / num_steps
                point = start + t * segment
                samples.append([point[0], point[1], 0.0, heading])
        last_heading = samples[-1][3] if samples else 0.0
        last_point = self.points[-1]
        samples.append([last_point[0], last_point[1], 0.0, last_heading])
        return FakeInterpolatedPolyline(np.asarray(samples, dtype=float))


class FakeRoadLane:
    def __init__(self, lane_id, center, *, adj_left=None, adj_right=None, next_lanes=None, prev_lanes=None):
        self.id = lane_id
        self.center = center
        self.adj_lanes_left = set(adj_left or [])
        self.adj_lanes_right = set(adj_right or [])
        self.next_lanes = set(next_lanes or [])
        self.prev_lanes = set(prev_lanes or [])


class FakeVectorMap:
    def __init__(self, lanes):
        self._lanes = {lane.id: lane for lane in lanes}

    def get_current_lane(self, xyzh, max_dist=2.0, max_heading_error=np.pi / 8):
        x, y, _, heading = xyzh
        candidates = []
        for lane in self._lanes.values():
            proj = lane.center.project_onto(np.asarray([[x, y, 0.0, heading]], dtype=float))[0]
            dist = float(np.linalg.norm(np.asarray([x, y]) - proj[:2]))
            heading_err = abs(np.arctan2(np.sin(float(proj[3]) - heading), np.cos(float(proj[3]) - heading)))
            if dist <= max_dist and heading_err <= max_heading_error:
                candidates.append(lane)
        return candidates

    def get_lanes_within(self, xyz, dist):
        x, y = xyz[:2]
        lanes = []
        for lane in self._lanes.values():
            proj = lane.center.project_onto(np.asarray([[x, y, 0.0, 0.0]], dtype=float))[0]
            lane_dist = float(np.linalg.norm(np.asarray([x, y]) - proj[:2]))
            if lane_dist <= dist:
                lanes.append(lane)
        return lanes

    def get_closest_lane(self, xyz):
        x, y = xyz[:2]
        scored = []
        for lane in self._lanes.values():
            proj = lane.center.project_onto(np.asarray([[x, y, 0.0, 0.0]], dtype=float))[0]
            lane_dist = float(np.linalg.norm(np.asarray([x, y]) - proj[:2]))
            scored.append((lane_dist, lane))
        return min(scored, key=lambda item: item[0])[1]

    def get_road_lane(self, lane_id):
        return self._lanes[lane_id]


class FakeMapAPI:
    def __init__(self, vec_map):
        self.vec_map = vec_map

    def get_map(self, map_id, **kwargs):
        return self.vec_map


def make_lane(lane_id, points, *, adj_left=None, adj_right=None, next_lanes=None, prev_lanes=None):
    return FakeRoadLane(
        lane_id=lane_id,
        center=FakePolyline(np.asarray(points, dtype=float)),
        adj_left=set(adj_left or []),
        adj_right=set(adj_right or []),
        next_lanes=set(next_lanes or []),
        prev_lanes=set(prev_lanes or []),
    )


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
        involved_agents=["ego", "agent_1", "agent_chain", "agent_far"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        risk_score=0.6,
        state_snapshots={
            "T_start": {
                "ego": Node("ego", "T_start", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(0.0, 0.0)),
                "agent_1": Node("agent_1", "T_start", "VEHICLE", (-5.0, 0.0), (0.0, 0.0), position=(18.0, 0.0)),
            },
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(10.0, 0.0), heading=0.0),
                "agent_1": Node("agent_1", "T_peak", "VEHICLE", (-10.0, 0.0), (0.0, 0.0), position=(20.0, 0.0)),
                "agent_chain": Node("agent_chain", "T_peak", "VEHICLE", (-2.0, 0.0), (0.0, 0.0), position=(20.0, 10.0)),
                "agent_far": Node("agent_far", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(200.0, 0.0)),
            },
            "T_end": {
                "ego": Node("ego", "T_end", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(20.0, 0.0), heading=0.0),
                "agent_1": Node("agent_1", "T_end", "VEHICLE", (-10.0, 0.0), (0.0, 0.0), position=(60.0, 0.0)),
            },
        },
        metadata={
            "trigger_agent_id": "agent_1",
            "trigger_agent_ids": ["agent_1"],
            "primary_risk_agent_ids": ["agent_1"],
            "peak_metric_type": "ttc",
            "peak_metric_value": 0.5,
            "ttc_event_threshold": 2.5,
            "min_ttc": 0.5,
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


def test_spatial_roi_rule_uses_peak_templates_and_alignment():
    episode = build_episode()
    graph = SpatialROIRule(roi_radius=20.0).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    for timestamp in ("T_start", "T_peak", "T_end"):
        nodes = graph.get_nodes_at_timestamp(timestamp)
        assert {node.agent_id for node in nodes} == {"ego", "agent_1", "agent_chain"}

    peak_spatial_edges = [
        edge for edge in graph.get_edges_at_timestamp("T_peak") if edge.edge_type == EdgeType.SPATIAL
    ]
    relations = {(edge.source_id, edge.target_id, edge.relation) for edge in peak_spatial_edges}
    assert ("ego", "agent_1", "ego_risk_roi") in relations
    assert ("ego", "agent_chain", "ego_neighbor_roi") in relations
    assert ("agent_1", "agent_chain", "risk_context_roi") in relations
    assert all(edge.target_id != "agent_far" for edge in peak_spatial_edges)

    start_edges = {
        (edge.source_id, edge.target_id, edge.relation): edge
        for edge in graph.get_edges_at_timestamp("T_start")
        if edge.edge_type == EdgeType.SPATIAL
    }
    assert start_edges[("ego", "agent_1", "ego_risk_roi")].metadata["edge_status"] == "active"
    assert round(start_edges[("ego", "agent_1", "ego_risk_roi")].metadata["relative_angle_deg"], 3) == 0.0
    assert start_edges[("ego", "agent_chain", "ego_neighbor_roi")].metadata["edge_status"] == "out_of_scene"
    assert start_edges[("agent_1", "agent_chain", "risk_context_roi")].metadata["edge_status"] == "out_of_scene"
    assert start_edges[("agent_1", "agent_chain", "risk_context_roi")].metadata["distance"] == -1.0
    assert start_edges[("agent_1", "agent_chain", "risk_context_roi")].metadata["relative_angle_deg"] == -999.0

    end_edges = {
        (edge.source_id, edge.target_id, edge.relation): edge
        for edge in graph.get_edges_at_timestamp("T_end")
        if edge.edge_type == EdgeType.SPATIAL
    }
    assert end_edges[("ego", "agent_1", "ego_risk_roi")].metadata["edge_status"] == "outside_roi"
    assert end_edges[("ego", "agent_chain", "ego_neighbor_roi")].metadata["edge_status"] == "out_of_scene"
    assert end_edges[("agent_1", "agent_chain", "risk_context_roi")].metadata["edge_status"] == "out_of_scene"

    chain_end = graph.get_node("agent_chain", "T_end")
    assert chain_end is not None
    assert chain_end.position is None
    assert chain_end.metadata["is_placeholder"] is True
    assert chain_end.metadata["state_status"] == "out_of_scene"

    temporal_edges = [
        edge
        for edge in graph.graph.edges(data=True)
        if edge[2]["edge_type"] == EdgeType.TEMPORAL.value
    ]
    assert len(temporal_edges) == 6


def test_ttc_rule_adds_causal_edge_at_peak():
    episode = build_episode()
    registry = RuleRegistry()
    registry.register(TTCCriticalRule(ttc_threshold=2.5))
    registry.register(SpatialROIRule(roi_radius=20.0, context_roi_radius=12.0))

    graph = registry.apply_all(episode)
    episode.sstg = graph

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert any(
        edge.source_id == "ego" and edge.target_id == "agent_1" and edge.relation == "has_collision_risk"
        for edge in causal_edges
    )
    assert {edge.target_id for edge in causal_edges}.issubset({"agent_1", "agent_chain"})
    peak_spatial_relations = {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.SPATIAL
    }
    assert ("ego", "agent_1", "ego_risk_roi") in peak_spatial_relations
    assert ("ego", "agent_chain", "ego_risk_roi") in peak_spatial_relations
    assert episode.rule_trace == ["ttc_critical", "spatial_roi"]


def test_pet_rule_adds_post_encroachment_causal_edge():
    episode = build_episode()
    episode.metadata.update(
        {
            "peak_metric_type": "pet",
            "peak_metric_value": 1.2,
            "min_pet": 1.2,
            "pet_event_threshold": 2.0,
        }
    )

    graph = TTCCriticalRule(ttc_threshold=2.5).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert len(causal_edges) == 1
    assert causal_edges[0].source_id == "ego"
    assert causal_edges[0].target_id == "agent_1"
    assert causal_edges[0].relation == "has_post_encroachment_risk"
    assert causal_edges[0].metadata["peak_metric_type"] == "pet"
    assert causal_edges[0].weight > 0.0


def test_pet_rule_filters_far_distance_causal_edge():
    episode = build_episode()
    episode.state_snapshots["T_peak"]["agent_1"] = Node(
        "agent_1",
        "T_peak",
        "VEHICLE",
        (-10.0, 0.0),
        (0.0, 0.0),
        position=(50.0, 0.0),
        heading=np.pi,
    )
    episode.metadata.update(
        {
            "peak_metric_type": "pet",
            "peak_metric_value": 1.2,
            "min_pet": 1.2,
            "pet_event_threshold": 2.0,
        }
    )

    graph = TTCCriticalRule(ttc_threshold=2.5, causal_max_distance_m_for_pet_ttc=25.0).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert causal_edges == []
    assert episode.metadata["causal_metric_filter_results"][0]["reason"] == "distance_threshold_exceeded"
    assert episode.metadata["causal_metric_filter_results"][0]["target_id"] == "agent_1"


def test_tti_rule_adds_intersection_arrival_causal_edge():
    episode = build_episode()
    episode.metadata.update(
        {
            "peak_metric_type": "tti",
            "peak_metric_value": 1.1,
            "min_tti": 1.1,
            "tti_event_threshold": 2.0,
        }
    )

    graph = TTCCriticalRule(ttc_threshold=2.5).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert len(causal_edges) == 1
    assert causal_edges[0].source_id == "ego"
    assert causal_edges[0].target_id == "agent_1"
    assert causal_edges[0].relation == "has_intersection_arrival_risk"
    assert causal_edges[0].metadata["peak_metric_type"] == "tti"
    assert causal_edges[0].weight > 0.0


def test_ttc_rule_filters_far_distance_before_map_filter():
    episode = Episode(
        scene_id="env:scene",
        scene_name="scene",
        env_name="env",
        dt=0.1,
        ego_agent_id="ego",
        t_start=8,
        t_peak=10,
        t_end=12,
        involved_agents=["ego", "agent_far"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        state_snapshots={
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (15.0, 0.0), (0.0, 0.0), position=(0.0, 0.0), heading=0.0),
                "agent_far": Node(
                    "agent_far",
                    "T_peak",
                    "VEHICLE",
                    (-15.0, 0.0),
                    (0.0, 0.0),
                    position=(40.0, 0.0),
                    heading=np.pi,
                ),
            }
        },
        metadata={"peak_metric_type": "ttc", "trigger_agent_id": "agent_far", "trigger_agent_ids": ["agent_far"]},
    )

    graph = TTCCriticalRule(
        ttc_threshold=2.5,
        causal_max_distance_m_for_pet_ttc=25.0,
        causal_filter_enabled=False,
    ).apply(episode, SSTG(scene_id=episode.scene_id, dt=episode.dt))

    causal_edges = [edge for edge in graph.get_edges_at_timestamp("T_peak") if edge.edge_type == EdgeType.CAUSAL]
    assert causal_edges == []
    assert episode.metadata["ttc_candidate_filter_results"][0]["reason"] == "distance_threshold_exceeded"
    assert episode.metadata["ttc_candidate_filter_results"][0]["metadata"]["pair_distance_m"] == 40.0


def test_dynamics_rule_adds_ego_response_edge_for_trigger_agents():
    episode = build_episode()
    episode.metadata.update(
        {
            "peak_metric_type": "dynamics",
            "peak_metric_value": 4.0,
            "trigger_agent_ids": ["agent_1", "agent_chain"],
        }
    )

    graph = TTCCriticalRule(ttc_threshold=2.5).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    causal_edges = [
        edge
        for edge in graph.get_edges_at_timestamp("T_peak")
        if edge.edge_type == EdgeType.CAUSAL
    ]
    assert {(edge.source_id, edge.target_id, edge.relation) for edge in causal_edges} == {
        ("ego", "agent_1", "ego_kinematic_response"),
        ("ego", "agent_chain", "ego_kinematic_response"),
    }
    assert all(edge.metadata["peak_metric_type"] == "dynamics" for edge in causal_edges)


def test_spatial_roi_rule_supports_dynamic_semantic_timestamps():
    episode = build_episode()
    episode.semantic_timestep_map = {
        "T_start": 8,
        "T_peak": 10,
        "T_mid_1": 11,
        "T_end": 12,
    }
    episode.semantic_timestamp_order = ["T_start", "T_peak", "T_mid_1", "T_end"]
    episode.state_snapshots["T_mid_1"] = {
        "ego": Node("ego", "T_mid_1", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(15.0, 0.0), heading=0.0),
        "agent_1": Node("agent_1", "T_mid_1", "VEHICLE", (-8.0, 0.0), (0.0, 0.0), position=(25.0, 0.0), heading=np.pi),
    }

    graph = SpatialROIRule(roi_radius=20.0).apply(
        episode,
        SSTG(scene_id=episode.scene_id, dt=episode.dt),
    )

    assert graph.timestamps == ["T_start", "T_peak", "T_mid_1", "T_end"]
    temporal_edges = [
        edge_data
        for _, _, _, edge_data in graph.graph.edges(keys=True, data=True)
        if edge_data["edge_type"] == EdgeType.TEMPORAL.value
    ]
    assert len(temporal_edges) == 9


def test_ttc_rule_filters_separated_vehicle_lanes():
    lane_a = make_lane("lane_a", [[0.0, 0.0], [20.0, 0.0]])
    lane_b = make_lane("lane_b", [[20.0, 8.0], [0.0, 8.0]])
    rule = TTCCriticalRule(
        ttc_threshold=2.5,
        map_api=FakeMapAPI(FakeVectorMap([lane_a, lane_b])),
    )
    episode = Episode(
        scene_id="sind:test",
        scene_name="tj_test",
        env_name="sind",
        dt=0.1,
        ego_agent_id="ego",
        t_start=8,
        t_peak=10,
        t_end=12,
        involved_agents=["ego", "agent_sep"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        state_snapshots={
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(1.0, 0.0), heading=0.0),
                "agent_sep": Node("agent_sep", "T_peak", "VEHICLE", (-10.0, 0.0), (0.0, 0.0), position=(9.0, 8.0), heading=np.pi),
            }
        },
        metadata={"peak_metric_type": "ttc", "trigger_agent_id": "agent_sep", "trigger_agent_ids": ["agent_sep"]},
    )

    graph = rule.apply(episode, SSTG(scene_id=episode.scene_id, dt=episode.dt))
    causal_edges = [edge for edge in graph.get_edges_at_timestamp("T_peak") if edge.edge_type == EdgeType.CAUSAL]
    assert causal_edges == []
    assert episode.metadata["ttc_candidate_filter_results"][0]["reason"] == "lane_topology_separated"


def test_ttc_rule_keeps_bicycle_interaction_without_lane_filter():
    lane_a = make_lane("lane_a", [[0.0, 0.0], [20.0, 0.0]])
    lane_b = make_lane("lane_b", [[20.0, 8.0], [0.0, 8.0]])
    rule = TTCCriticalRule(
        ttc_threshold=2.5,
        map_api=FakeMapAPI(FakeVectorMap([lane_a, lane_b])),
    )
    episode = Episode(
        scene_id="sind:test",
        scene_name="tj_test",
        env_name="sind",
        dt=0.1,
        ego_agent_id="ego",
        t_start=8,
        t_peak=10,
        t_end=12,
        involved_agents=["ego", "bike_1"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        state_snapshots={
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (10.0, 0.0), (0.0, 0.0), position=(1.0, 0.0), heading=0.0),
                "bike_1": Node("bike_1", "T_peak", "BICYCLE", (-10.0, 0.0), (0.0, 0.0), position=(9.0, 8.0), heading=np.pi),
            }
        },
        metadata={"peak_metric_type": "ttc", "trigger_agent_id": "bike_1", "trigger_agent_ids": ["bike_1"]},
    )

    graph = rule.apply(episode, SSTG(scene_id=episode.scene_id, dt=episode.dt))
    causal_edges = [edge for edge in graph.get_edges_at_timestamp("T_peak") if edge.edge_type == EdgeType.CAUSAL]
    assert len(causal_edges) == 1
    assert causal_edges[0].target_id == "bike_1"


def test_ttc_rule_sends_track_birth_near_conflict_zone_to_manual_review(tmp_path):
    lane_a = make_lane("lane_a", [[0.0, 0.0], [20.0, 0.0]])
    lane_cross = make_lane("lane_cross", [[5.0, 10.0], [5.0, -10.0]])
    rule = TTCCriticalRule(
        ttc_threshold=2.5,
        map_api=FakeMapAPI(FakeVectorMap([lane_a, lane_cross])),
        intersection_proximity_radius_m=18.0,
    )
    episode = Episode(
        scene_id="sind:test",
        scene_name="tj_test",
        env_name="sind",
        dt=0.1,
        ego_agent_id="ego",
        t_start=0,
        t_peak=10,
        t_end=12,
        involved_agents=["ego", "agent_cross"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        state_snapshots={
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (4.0, 0.0), (0.0, 0.0), position=(2.0, 0.0), heading=0.0),
                "agent_cross": Node("agent_cross", "T_peak", "VEHICLE", (0.0, -4.0), (0.0, 0.0), position=(5.0, 3.0), heading=-np.pi / 2.0),
            }
        },
        metadata={
            "peak_metric_type": "ttc",
            "trigger_agent_id": "agent_cross",
            "trigger_agent_ids": ["agent_cross"],
            "conflict_zone": {"center": [5.0, 0.0], "radius": 6.0},
            "trigger_track_windows": {
                "agent_cross": {"birth_ts": 9, "birth_position": [5.0, 4.0], "death_ts": 20}
            },
        },
    )

    graph = rule.apply(episode, SSTG(scene_id=episode.scene_id, dt=episode.dt))
    episode.sstg = graph

    causal_edges = [edge for edge in graph.get_edges_at_timestamp("T_peak") if edge.edge_type == EdgeType.CAUSAL]
    assert causal_edges == []
    assert episode.metadata["manual_review_required"] is True
    checker = create_default_checker(tmp_path)
    validation = checker.validate_episode(episode)
    assert validation.passed is False
    assert "manual review" in validation.reason.lower()
