"""
Integration-style tests for the refactored risk mining pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Episode, EpisodeType, Node, SSTG, Slicer
from src.library import DualLibrary
from src.rules import RuleRegistry, SpatialROIRule, TTCCriticalRule
from src.utils import create_default_checker


class FakeState:
    def __init__(self, position, velocity, acceleration=(0.0, 0.0), heading=0.0):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.acceleration = np.asarray(acceleration, dtype=float)
        self.heading = np.asarray([heading], dtype=float)


class FakeExtent:
    def __init__(self, length=4.5, width=2.0, height=1.5):
        self.length = length
        self.width = width
        self.height = height

    def get_extents(self, start_ts, end_ts):
        return np.asarray([[self.length, self.width, self.height]], dtype=float)


class FakeAgent:
    def __init__(self, name, agent_type):
        self.name = name
        self.type = agent_type
        self.extent = FakeExtent()


class FakeAgentType:
    def __init__(self, name):
        self.name = name


class FakeScene:
    def __init__(self, agents, agent_presence, dt=1.0, name="fake_scene", env_name="fake_env"):
        self.env_name = env_name
        self.name = name
        self.dt = dt
        self.length_timesteps = len(agent_presence)
        self.agents = agents
        self.agent_presence = agent_presence


class FakeCache:
    def __init__(self, states):
        self.states = states

    def get_state(self, agent_id, scene_ts):
        return self.states[(agent_id, scene_ts)]


def _presence(agents, steps):
    return [list(agents) for _ in range(steps)]


def test_end_to_end_pipeline_with_fake_scene(tmp_path):
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    other = FakeAgent("agent_1", FakeAgentType("VEHICLE"))
    far = FakeAgent("agent_far", FakeAgentType("VEHICLE"))
    agents = [ego, other, far]
    scene = FakeScene(agents, _presence(agents, 5))

    states = {
        ("ego", 0): FakeState((0.0, 0.0), (10.0, 0.0)),
        ("agent_1", 0): FakeState((40.0, 0.0), (-5.0, 0.0)),
        ("agent_far", 0): FakeState((200.0, 0.0), (0.0, 0.0)),
        ("ego", 1): FakeState((10.0, 0.0), (10.0, 0.0)),
        ("agent_1", 1): FakeState((25.0, 0.0), (-5.0, 0.0)),
        ("agent_far", 1): FakeState((200.0, 0.0), (0.0, 0.0)),
        ("ego", 2): FakeState((20.0, 0.0), (10.0, 0.0)),
        ("agent_1", 2): FakeState((30.0, 0.0), (-10.0, 0.0)),
        ("agent_far", 2): FakeState((200.0, 0.0), (0.0, 0.0)),
        ("ego", 3): FakeState((30.0, 0.0), (10.0, 0.0)),
        ("agent_1", 3): FakeState((20.0, 0.0), (-10.0, 0.0)),
        ("agent_far", 3): FakeState((200.0, 0.0), (0.0, 0.0)),
        ("ego", 4): FakeState((40.0, 0.0), (10.0, 0.0)),
        ("agent_1", 4): FakeState((10.0, 0.0), (-10.0, 0.0)),
        ("agent_far", 4): FakeState((200.0, 0.0), (0.0, 0.0)),
    }
    cache = FakeCache(states)

    slicer = Slicer(pre_buffer_sec=2.0, post_buffer_sec=2.0, min_peak_gap_sec=4.0)
    episodes = slicer.extract_episodes(scene, cache)
    assert len(episodes) == 1

    episode = episodes[0]
    assert episode.episode_type == EpisodeType.MULTI_METRIC_WINDOW
    assert episode.metadata["peak_metric_type"] == "ttc"
    assert episode.metadata["trigger_agent_ids"] == ["agent_1"]
    assert episode.metadata["start_rule"] == "fallback_latency"
    assert episode.metadata["start_rule_boundary_crossing_suppressed"] is True
    assert episode.metadata["end_rule"] == "risk_clear"

    registry = RuleRegistry()
    registry.register(TTCCriticalRule(ttc_threshold=2.5))
    registry.register(SpatialROIRule(roi_radius=20.0, context_roi_radius=12.0))

    episode.sstg = registry.apply_all(episode)
    checker = create_default_checker(tmp_path)
    validation = checker.validate_episode(episode)
    assert validation.passed is True

    dual_lib = DualLibrary(tmp_path)
    element_id, event_id = dual_lib.add_episode(episode)
    event_path = tmp_path / "libraries" / "risk_events" / f"{event_id}.json"
    element_path = tmp_path / "libraries" / "risk_elements" / f"{element_id}.json"

    assert event_path.exists()
    assert element_path.exists()

    event_payload = json.loads(event_path.read_text())
    assert event_payload["event_id"] == event_id
    assert event_payload["sstg"]["scene_id"] == episode.scene_id
    assert event_payload["sstg"]["metadata"]["peak_included_agent_ids"] == ["agent_1", "ego"]
    assert event_payload["sstg"]["metadata"]["risk_anchor_agent_ids"] == ["agent_1"]

    spatial_edges = [
        edge["data"]
        for edge in event_payload["sstg"]["edges"]
        if edge["data"]["edge_type"] == "spatial"
    ]
    assert len(spatial_edges) == 3
    assert all(
        edge["metadata"]["edge_status"] in {"active", "outside_roi", "out_of_scene"}
        for edge in spatial_edges
    )
    assert all("relative_angle_deg" in edge["metadata"] for edge in spatial_edges)


def test_slicer_prefers_non_stationary_vehicle_as_ego():
    static_vehicle = FakeAgent("static_vehicle", FakeAgentType("VEHICLE"))
    moving_vehicle = FakeAgent("moving_vehicle", FakeAgentType("VEHICLE"))
    other = FakeAgent("agent_1", FakeAgentType("VEHICLE"))
    agents = [static_vehicle, moving_vehicle, other]
    scene = FakeScene(agents, _presence(agents, 5))
    states = {
        ("static_vehicle", 0): FakeState((0.0, 0.0), (0.0, 0.0)),
        ("moving_vehicle", 0): FakeState((0.0, 0.0), (8.0, 0.0)),
        ("agent_1", 0): FakeState((40.0, 0.0), (-4.0, 0.0)),
        ("static_vehicle", 1): FakeState((0.0, 0.0), (0.0, 0.0)),
        ("moving_vehicle", 1): FakeState((8.0, 0.0), (8.0, 0.0)),
        ("agent_1", 1): FakeState((28.0, 0.0), (-4.0, 0.0)),
        ("static_vehicle", 2): FakeState((0.0, 0.0), (0.0, 0.0)),
        ("moving_vehicle", 2): FakeState((16.0, 0.0), (8.0, 0.0)),
        ("agent_1", 2): FakeState((20.0, 0.0), (-8.0, 0.0)),
        ("static_vehicle", 3): FakeState((0.0, 0.0), (0.0, 0.0)),
        ("moving_vehicle", 3): FakeState((24.0, 0.0), (8.0, 0.0)),
        ("agent_1", 3): FakeState((12.0, 0.0), (-8.0, 0.0)),
        ("static_vehicle", 4): FakeState((0.0, 0.0), (0.0, 0.0)),
        ("moving_vehicle", 4): FakeState((32.0, 0.0), (8.0, 0.0)),
        ("agent_1", 4): FakeState((4.0, 0.0), (-8.0, 0.0)),
    }

    episodes = Slicer(pre_buffer_sec=2.0, post_buffer_sec=2.0, ego_motion_threshold=0.5).extract_episodes(
        scene,
        FakeCache(states),
    )
    assert len(episodes) == 1
    assert episodes[0].ego_agent_id == "moving_vehicle"


def test_slicer_extracts_multiple_ttc_peaks_from_one_scene():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    agent_a = FakeAgent("agent_a", FakeAgentType("VEHICLE"))
    agent_b = FakeAgent("agent_b", FakeAgentType("VEHICLE"))
    agents = [ego, agent_a, agent_b]
    scene = FakeScene(agents, _presence(agents, 10))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (10.0, 0.0)),
        ("agent_a", 0): FakeState((60.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 0): FakeState((200.0, 0.0), (-5.0, 0.0)),
        ("ego", 1): FakeState((10.0, 0.0), (10.0, 0.0)),
        ("agent_a", 1): FakeState((40.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 1): FakeState((180.0, 0.0), (-5.0, 0.0)),
        ("ego", 2): FakeState((20.0, 0.0), (10.0, 0.0)),
        ("agent_a", 2): FakeState((25.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 2): FakeState((160.0, 0.0), (-5.0, 0.0)),
        ("ego", 3): FakeState((30.0, 0.0), (10.0, 0.0)),
        ("agent_a", 3): FakeState((12.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 3): FakeState((140.0, 0.0), (-5.0, 0.0)),
        ("ego", 4): FakeState((40.0, 0.0), (10.0, 0.0)),
        ("agent_a", 4): FakeState((-5.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 4): FakeState((120.0, 0.0), (-5.0, 0.0)),
        ("ego", 5): FakeState((50.0, 0.0), (10.0, 0.0)),
        ("agent_a", 5): FakeState((-20.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 5): FakeState((95.0, 0.0), (-5.0, 0.0)),
        ("ego", 6): FakeState((60.0, 0.0), (10.0, 0.0)),
        ("agent_a", 6): FakeState((-35.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 6): FakeState((75.0, 0.0), (-5.0, 0.0)),
        ("ego", 7): FakeState((70.0, 0.0), (10.0, 0.0)),
        ("agent_a", 7): FakeState((-50.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 7): FakeState((58.0, 0.0), (-5.0, 0.0)),
        ("ego", 8): FakeState((80.0, 0.0), (10.0, 0.0)),
        ("agent_a", 8): FakeState((-65.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 8): FakeState((40.0, 0.0), (-5.0, 0.0)),
        ("ego", 9): FakeState((90.0, 0.0), (10.0, 0.0)),
        ("agent_a", 9): FakeState((-80.0, 0.0), (-5.0, 0.0)),
        ("agent_b", 9): FakeState((20.0, 0.0), (-5.0, 0.0)),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        ttc_event_threshold=2.5,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=5,
    ).extract_episodes(scene, FakeCache(states))

    assert len(episodes) == 2
    assert [episode.t_peak for episode in episodes] == [2, 6]
    assert [episode.metadata["peak_metric_type"] for episode in episodes] == ["ttc", "ttc"]
    assert [episode.metadata["trigger_agent_ids"] for episode in episodes] == [["agent_a"], ["agent_b"]]


def test_slicer_extracts_pet_peak_candidate():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    cross = FakeAgent("agent_cross", FakeAgentType("VEHICLE"))
    agents = [ego, cross]
    scene = FakeScene(agents, _presence(agents, 5))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 0): FakeState((6.0, -4.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 1): FakeState((2.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 1): FakeState((6.0, -2.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 2): FakeState((4.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 2): FakeState((6.0, 0.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 3): FakeState((6.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 3): FakeState((6.0, 2.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 4): FakeState((8.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 4): FakeState((6.0, 4.0), (0.0, 2.0), heading=np.pi / 2.0),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        pet_event_threshold=2.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
    ).extract_episodes(scene, FakeCache(states))

    assert len(episodes) == 1
    assert episodes[0].metadata["peak_metric_type"] == "pet"
    assert episodes[0].metadata["trigger_agent_ids"] == ["agent_cross"]
    assert episodes[0].metadata["min_pet"] < 2.0


def test_slicer_suppresses_boundary_crossing_when_peak_inside_conflict_zone():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    trigger = FakeAgent("agent_trigger", FakeAgentType("VEHICLE"))
    scene = FakeScene([ego, trigger], _presence([ego, trigger], 6))
    cache = FakeCache(
        {
            ("ego", 0): FakeState((-3.0, 0.0), (1.0, 0.0)),
            ("ego", 1): FakeState((-2.0, 0.0), (1.0, 0.0)),
            ("ego", 2): FakeState((-1.0, 0.0), (1.0, 0.0)),
            ("ego", 3): FakeState((0.0, 0.0), (1.0, 0.0)),
            ("ego", 4): FakeState((1.0, 0.0), (1.0, 0.0)),
            ("ego", 5): FakeState((2.0, 0.0), (1.0, 0.0)),
            ("agent_trigger", 0): FakeState((13.0, 0.0), (-2.0, 0.0)),
            ("agent_trigger", 1): FakeState((12.0, 0.0), (-2.0, 0.0)),
            ("agent_trigger", 2): FakeState((11.0, 0.0), (-2.0, 0.0)),
            ("agent_trigger", 3): FakeState((10.0, 0.0), (-2.0, 0.0)),
            ("agent_trigger", 4): FakeState((5.0, 0.0), (-2.0, 0.0)),
            ("agent_trigger", 5): FakeState((2.0, 0.0), (-2.0, 0.0)),
        }
    )
    slicer = Slicer()

    t_start, start_rule, metadata = slicer._find_t_start(
        scene=scene,
        cache=cache,
        ego_agent_id="ego",
        trigger_agent_ids=["agent_trigger"],
        t_peak=4,
        reaction_latency_ts=2,
        conflict_center=(3.0, 0.0),
    )

    assert t_start == 2
    assert start_rule == "fallback_latency"
    assert metadata["peak_inside_conflict_zone"] is True
    assert metadata["start_rule_boundary_crossing_suppressed"] is True
    assert metadata["start_rule_boundary_crossing_suppressed_reason"] == "peak_agent_already_inside_conflict_zone"


def test_slicer_filters_far_ttc_candidate_before_episode_creation():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    other = FakeAgent("agent_far_ttc", FakeAgentType("VEHICLE"))
    scene = FakeScene([ego, other], _presence([ego, other], 3))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (10.0, 0.0)),
        ("agent_far_ttc", 0): FakeState((80.0, 0.0), (-10.0, 0.0)),
        ("ego", 1): FakeState((10.0, 0.0), (10.0, 0.0)),
        ("agent_far_ttc", 1): FakeState((60.0, 0.0), (-10.0, 0.0)),
        ("ego", 2): FakeState((20.0, 0.0), (10.0, 0.0)),
        ("agent_far_ttc", 2): FakeState((50.0, 0.0), (-10.0, 0.0)),
    }

    episodes = Slicer(
        ttc_event_threshold=2.5,
        causal_max_distance_m_for_pet_ttc=25.0,
        enable_pet_peak=False,
        enable_tti_peak=False,
        enable_dynamics_peak=False,
    ).extract_episodes(scene, FakeCache(states))

    assert episodes == []


def test_slicer_builds_adaptive_intermediate_semantic_snapshots():
    slicer = Slicer()

    one_mid = slicer._build_semantic_timestep_map(t_start=10, t_peak=20, t_end=90)
    assert one_mid == {"T_start": 10, "T_peak": 20, "T_mid_1": 55, "T_end": 90}

    two_mid = slicer._build_semantic_timestep_map(t_start=10, t_peak=20, t_end=140)
    assert two_mid == {
        "T_start": 10,
        "T_peak": 20,
        "T_mid_1": 60,
        "T_mid_2": 100,
        "T_end": 140,
    }


def test_slicer_filters_following_behavior_from_pet_candidates():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    lead = FakeAgent("agent_lead", FakeAgentType("VEHICLE"))
    agents = [ego, lead]
    scene = FakeScene(agents, _presence(agents, 5))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (3.0, 0.0)),
        ("agent_lead", 0): FakeState((20.0, 0.5), (6.0, 0.0)),
        ("ego", 1): FakeState((3.0, 0.0), (3.0, 0.0)),
        ("agent_lead", 1): FakeState((26.0, 0.5), (6.0, 0.0)),
        ("ego", 2): FakeState((6.0, 0.0), (3.0, 0.0)),
        ("agent_lead", 2): FakeState((32.0, 0.5), (6.0, 0.0)),
        ("ego", 3): FakeState((9.0, 0.0), (3.0, 0.0)),
        ("agent_lead", 3): FakeState((38.0, 0.5), (6.0, 0.0)),
        ("ego", 4): FakeState((12.0, 0.0), (3.0, 0.0)),
        ("agent_lead", 4): FakeState((44.0, 0.5), (6.0, 0.0)),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        pet_event_threshold=2.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
        enable_pet_peak=True,
        enable_tti_peak=False,
        enable_ttc_peak=False,
        enable_dynamics_peak=False,
    ).extract_episodes(scene, FakeCache(states))

    assert episodes == []


def test_slicer_extracts_tti_peak_candidate_from_future_trajectories():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    cross = FakeAgent("agent_cross", FakeAgentType("VEHICLE"))
    agents = [ego, cross]
    scene = FakeScene(agents, _presence(agents, 6))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 0): FakeState((6.0, -4.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 1): FakeState((2.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 1): FakeState((6.0, -2.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 2): FakeState((4.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 2): FakeState((6.0, 0.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 3): FakeState((6.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 3): FakeState((6.0, 2.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 4): FakeState((8.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 4): FakeState((6.0, 4.0), (0.0, 2.0), heading=np.pi / 2.0),
        ("ego", 5): FakeState((10.0, 0.0), (2.0, 0.0)),
        ("agent_cross", 5): FakeState((6.0, 6.0), (0.0, 2.0), heading=np.pi / 2.0),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        tti_event_threshold=2.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
        enable_pet_peak=False,
        enable_tti_peak=True,
        enable_ttc_peak=False,
        enable_dynamics_peak=False,
    ).extract_episodes(scene, FakeCache(states))

    assert len(episodes) == 1
    assert episodes[0].metadata["peak_metric_type"] == "tti"
    assert episodes[0].metadata["trigger_agent_ids"] == ["agent_cross"]
    assert 0.0 < episodes[0].metadata["min_tti"] < 2.0
    assert episodes[0].metadata["tti_prefilter_enabled"] is True
    assert episodes[0].metadata["tti_ttc_prefilter_threshold"] > episodes[0].metadata["ttc_event_threshold"]
    assert episodes[0].metadata["peak_evidence"]["trajectory_intersection_angle_deg"] >= 25.0
    assert episodes[0].metadata["conflict_zone"]["center"] is not None


def test_slicer_filters_following_behavior_from_tti_candidates():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    leader = FakeAgent("agent_lead", FakeAgentType("VEHICLE"))
    agents = [ego, leader]
    scene = FakeScene(agents, _presence(agents, 6))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 0): FakeState((8.0, 0.0), (2.0, 0.0)),
        ("ego", 1): FakeState((2.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 1): FakeState((10.0, 0.0), (2.0, 0.0)),
        ("ego", 2): FakeState((4.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 2): FakeState((12.0, 0.0), (2.0, 0.0)),
        ("ego", 3): FakeState((6.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 3): FakeState((14.0, 0.0), (2.0, 0.0)),
        ("ego", 4): FakeState((8.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 4): FakeState((16.0, 0.0), (2.0, 0.0)),
        ("ego", 5): FakeState((10.0, 0.0), (2.0, 0.0)),
        ("agent_lead", 5): FakeState((18.0, 0.0), (2.0, 0.0)),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        tti_event_threshold=2.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
        enable_pet_peak=False,
        enable_tti_peak=True,
        enable_ttc_peak=False,
        enable_dynamics_peak=False,
    ).extract_episodes(scene, FakeCache(states))

    assert episodes == []


def test_slicer_extracts_dynamics_peak_candidate():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    context = FakeAgent("agent_context", FakeAgentType("VEHICLE"))
    agents = [ego, context]
    scene = FakeScene(agents, _presence(agents, 5))
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (5.0, 0.0), acceleration=(0.0, 0.0)),
        ("agent_context", 0): FakeState((40.0, 0.0), (0.0, 0.0)),
        ("ego", 1): FakeState((5.0, 0.0), (5.0, 0.0), acceleration=(0.0, 0.0)),
        ("agent_context", 1): FakeState((40.0, 0.0), (0.0, 0.0)),
        ("ego", 2): FakeState((10.0, 0.0), (5.0, 0.0), acceleration=(4.0, 0.0)),
        ("agent_context", 2): FakeState((40.0, 0.0), (0.0, 0.0)),
        ("ego", 3): FakeState((15.0, 0.0), (5.0, 0.0), acceleration=(0.0, 0.0)),
        ("agent_context", 3): FakeState((40.0, 0.0), (0.0, 0.0)),
        ("ego", 4): FakeState((20.0, 0.0), (5.0, 0.0), acceleration=(0.0, 0.0)),
        ("agent_context", 4): FakeState((40.0, 0.0), (0.0, 0.0)),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        ego_motion_threshold=0.5,
        dynamic_acc_threshold=3.0,
        dynamic_jerk_threshold=4.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
    ).extract_episodes(scene, FakeCache(states))

    assert len(episodes) == 1
    assert episodes[0].metadata["peak_metric_type"] == "dynamics"
    assert episodes[0].metadata["trigger_agent_ids"] == ["agent_context"]
    assert episodes[0].metadata["peak_evidence"]["ego_longitudinal_acc"] >= 4.0


def test_slicer_does_not_treat_track_birth_as_topological_flip():
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    late_target = FakeAgent("agent_late", FakeAgentType("VEHICLE"))
    scene = FakeScene([ego, late_target], [[ego], [ego], [ego], [ego], [ego, late_target], [ego, late_target], [ego, late_target]])
    states = {
        ("ego", 0): FakeState((0.0, 0.0), (10.0, 0.0)),
        ("ego", 1): FakeState((10.0, 0.0), (10.0, 0.0)),
        ("ego", 2): FakeState((20.0, 0.0), (10.0, 0.0)),
        ("ego", 3): FakeState((30.0, 0.0), (10.0, 0.0)),
        ("ego", 4): FakeState((40.0, 0.0), (10.0, 0.0)),
        ("ego", 5): FakeState((50.0, 0.0), (10.0, 0.0)),
        ("ego", 6): FakeState((60.0, 0.0), (10.0, 0.0)),
        ("agent_late", 4): FakeState((55.0, 0.0), (-5.0, 0.0)),
        ("agent_late", 5): FakeState((52.0, 0.0), (-8.0, 0.0)),
        ("agent_late", 6): FakeState((45.0, 0.0), (-10.0, 0.0)),
    }

    episodes = Slicer(
        pre_buffer_sec=1.0,
        post_buffer_sec=1.0,
        reaction_latency_sec=2.0,
        min_peak_gap_sec=3.0,
        max_episodes_per_scene=1,
    ).extract_episodes(scene, FakeCache(states))

    assert len(episodes) == 1
    assert episodes[0].metadata["start_rule"] == "fallback_latency"
    assert episodes[0].t_start == 3


def test_failed_graph_goes_to_manual_review(tmp_path):
    checker = create_default_checker(tmp_path)

    manual_episode = Episode(
        scene_id="fake_env:manual",
        scene_name="manual",
        env_name="fake_env",
        dt=1.0,
        ego_agent_id="ego",
        t_start=0,
        t_peak=1,
        t_end=2,
        involved_agents=["ego", "agent_1"],
        episode_type=EpisodeType.MULTI_METRIC_WINDOW,
        state_snapshots={
            "T_peak": {
                "ego": Node("ego", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(0.0, 0.0)),
                "agent_1": Node("agent_1", "T_peak", "VEHICLE", (0.0, 0.0), (0.0, 0.0), position=(20.0, 0.0)),
            }
        },
    )
    manual_episode.sstg = SSTG(scene_id=manual_episode.scene_id, dt=manual_episode.dt)
    manual_episode.sstg.add_node(manual_episode.state_snapshots["T_peak"]["ego"])
    manual_episode.sstg.add_node(manual_episode.state_snapshots["T_peak"]["agent_1"])

    validation = checker.validate_episode(manual_episode)
    assert validation.passed is False
    review_path = checker.save_for_review(manual_episode, validation)
    assert review_path.exists()
