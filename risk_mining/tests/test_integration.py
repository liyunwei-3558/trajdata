"""
Integration-style tests for the refactored risk mining pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Slicer
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
    def __init__(self, agents, agent_presence):
        self.env_name = "fake_env"
        self.name = "fake_scene"
        self.dt = 1.0
        self.length_timesteps = len(agent_presence)
        self.agents = agents
        self.agent_presence = agent_presence


class FakeCache:
    def __init__(self, states):
        self.states = states

    def get_state(self, agent_id, scene_ts):
        return self.states[(agent_id, scene_ts)]


def test_end_to_end_pipeline_with_fake_scene(tmp_path):
    ego = FakeAgent("ego", FakeAgentType("VEHICLE"))
    other = FakeAgent("agent_1", FakeAgentType("VEHICLE"))
    far = FakeAgent("agent_far", FakeAgentType("VEHICLE"))
    agents = [ego, other, far]
    agent_presence = [agents, agents, agents, agents, agents]
    scene = FakeScene(agents, agent_presence)

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

    slicer = Slicer(pre_buffer_sec=2.0, post_buffer_sec=2.0)
    episodes = slicer.extract_episodes(scene, cache)
    assert len(episodes) == 1

    registry = RuleRegistry()
    registry.register(SpatialROIRule(roi_radius=50.0))
    registry.register(TTCCriticalRule(ttc_threshold=2.5))

    episode = episodes[0]
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


def test_failed_graph_goes_to_manual_review(tmp_path):
    checker = create_default_checker(tmp_path)
    episode = Slicer().extract_episodes(
        FakeScene(
            [FakeAgent("ego", FakeAgentType("VEHICLE")), FakeAgent("agent_1", FakeAgentType("VEHICLE"))],
            [
                [FakeAgent("ego", FakeAgentType("VEHICLE")), FakeAgent("agent_1", FakeAgentType("VEHICLE"))],
                [FakeAgent("ego", FakeAgentType("VEHICLE")), FakeAgent("agent_1", FakeAgentType("VEHICLE"))],
                [FakeAgent("ego", FakeAgentType("VEHICLE")), FakeAgent("agent_1", FakeAgentType("VEHICLE"))],
            ],
        ),
        FakeCache(
            {
                ("ego", 0): FakeState((0.0, 0.0), (0.0, 0.0)),
                ("agent_1", 0): FakeState((50.0, 0.0), (0.0, 0.0)),
                ("ego", 1): FakeState((0.0, 0.0), (0.0, 0.0)),
                ("agent_1", 1): FakeState((50.0, 0.0), (0.0, 0.0)),
                ("ego", 2): FakeState((0.0, 0.0), (0.0, 0.0)),
                ("agent_1", 2): FakeState((50.0, 0.0), (0.0, 0.0)),
            }
        ),
    )

    assert episode == []

    # Build a graph with no causal edge to exercise manual review dump directly.
    from src.core import Episode, EpisodeType, Node, SSTG

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
        episode_type=EpisodeType.TTC_MIN_WINDOW,
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
