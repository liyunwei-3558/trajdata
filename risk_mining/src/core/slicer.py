"""
Dummy slicing logic for extracting a single TTC-driven episode per scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .scene_graph import Node, SSTG

if TYPE_CHECKING:
    from trajdata.caching.df_cache import DataFrameCache
    from trajdata.data_structures.scene_metadata import Scene


class EpisodeType(str, Enum):
    """Current episode extraction mode."""

    TTC_MIN_WINDOW = "ttc_min_window"


@dataclass
class Episode:
    """Scenario slice centered around a critical semantic moment."""

    scene_id: str
    scene_name: str
    env_name: str
    dt: float
    ego_agent_id: str
    t_start: int
    t_peak: int
    t_end: int
    involved_agents: List[str]
    state_snapshots: Dict[str, Dict[str, Node]]
    episode_type: EpisodeType = EpisodeType.TTC_MIN_WINDOW
    risk_score: float = 0.0
    sstg: Optional[SSTG] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    rule_trace: List[str] = field(default_factory=list)

    @property
    def semantic_timesteps(self) -> Dict[str, int]:
        return {"T_start": self.t_start, "T_peak": self.t_peak, "T_end": self.t_end}

    @property
    def ordered_timestamps(self) -> List[str]:
        return ["T_start", "T_peak", "T_end"]

    @property
    def duration_timesteps(self) -> int:
        return self.t_end - self.t_start + 1

    def get_nodes_at(self, timestamp: str) -> List[Node]:
        return sorted(self.state_snapshots.get(timestamp, {}).values(), key=lambda node: node.agent_id)

    def get_node(self, agent_id: str, timestamp: str) -> Optional[Node]:
        return self.state_snapshots.get(timestamp, {}).get(agent_id)

    def add_rule_trace(self, rule_name: str) -> None:
        if rule_name not in self.rule_trace:
            self.rule_trace.append(rule_name)


class Slicer:
    """Minimal scene slicer based on the minimum ego-to-agent TTC."""

    def __init__(self, pre_buffer_sec: float = 2.0, post_buffer_sec: float = 2.0):
        self.pre_buffer_sec = float(pre_buffer_sec)
        self.post_buffer_sec = float(post_buffer_sec)

    def extract_episodes(self, scene: "Scene", cache: "DataFrameCache") -> List[Episode]:
        ego_agent = self._select_ego_agent(scene)
        if ego_agent is None or not getattr(scene, "agent_presence", None):
            return []

        best_candidate: Optional[Tuple[int, str, float]] = None
        for scene_ts, agents in enumerate(scene.agent_presence):
            if not any(agent.name == ego_agent.name for agent in agents):
                continue

            ego_state = self._safe_get_state(cache, ego_agent.name, scene_ts)
            if ego_state is None:
                continue

            for agent in agents:
                if agent.name == ego_agent.name:
                    continue
                other_state = self._safe_get_state(cache, agent.name, scene_ts)
                if other_state is None:
                    continue

                ttc = self._compute_ttc(ego_state, other_state)
                if ttc is None:
                    continue

                if best_candidate is None or ttc < best_candidate[2]:
                    best_candidate = (scene_ts, agent.name, ttc)

        if best_candidate is None:
            return []

        t_peak, trigger_agent_id, min_ttc = best_candidate
        pre_buffer_ts = int(round(self.pre_buffer_sec / scene.dt))
        post_buffer_ts = int(round(self.post_buffer_sec / scene.dt))
        t_start = max(0, t_peak - pre_buffer_ts)
        t_end = min(scene.length_timesteps - 1, t_peak + post_buffer_ts)

        state_snapshots: Dict[str, Dict[str, Node]] = {}
        involved_agents = {ego_agent.name, trigger_agent_id}
        for label, timestep in {"T_start": t_start, "T_peak": t_peak, "T_end": t_end}.items():
            snapshot = self._collect_snapshot(scene, cache, timestep, label)
            if snapshot:
                involved_agents.update(snapshot.keys())
            state_snapshots[label] = snapshot

        episode = Episode(
            scene_id=f"{scene.env_name}:{scene.name}",
            scene_name=scene.name,
            env_name=scene.env_name,
            dt=scene.dt,
            ego_agent_id=ego_agent.name,
            t_start=t_start,
            t_peak=t_peak,
            t_end=t_end,
            involved_agents=sorted(involved_agents),
            state_snapshots=state_snapshots,
            risk_score=max(0.0, min(1.0, 1.0 / (1.0 + min_ttc))),
            metadata={
                "trigger_agent_id": trigger_agent_id,
                "min_ttc": min_ttc,
                "semantic_timesteps": {"T_start": t_start, "T_peak": t_peak, "T_end": t_end},
            },
        )
        return [episode]

    def _collect_snapshot(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        timestep: int,
        timestamp_label: str,
    ) -> Dict[str, Node]:
        snapshot: Dict[str, Node] = {}
        for agent in self._agents_at_timestep(scene, timestep):
            state = self._safe_get_state(cache, agent.name, timestep)
            if state is None:
                continue
            snapshot[agent.name] = self._state_to_node(agent, state, timestep, timestamp_label)
        return snapshot

    @staticmethod
    def _agents_at_timestep(scene: "Scene", timestep: int) -> Iterable[Any]:
        agent_presence = getattr(scene, "agent_presence", None) or []
        if timestep >= len(agent_presence):
            return []
        return agent_presence[timestep]

    @staticmethod
    def _safe_get_state(cache: "DataFrameCache", agent_id: str, scene_ts: int) -> Optional[Any]:
        try:
            return cache.get_state(agent_id, scene_ts)
        except Exception:
            return None

    def _select_ego_agent(self, scene: "Scene") -> Optional[Any]:
        agents = getattr(scene, "agents", None) or []
        for agent in agents:
            if agent.name == "ego":
                return agent

        for agent in agents:
            if self._agent_type_name(agent).upper() == "VEHICLE":
                return agent

        return agents[0] if agents else None

    @staticmethod
    def _agent_type_name(agent: Any) -> str:
        agent_type = getattr(agent, "type", None)
        if agent_type is None:
            return "UNKNOWN"
        return getattr(agent_type, "name", str(agent_type))

    def _state_to_node(self, agent: Any, state: Any, raw_timestep: int, timestamp_label: str) -> Node:
        return Node(
            agent_id=agent.name,
            timestamp=timestamp_label,
            type=self._agent_type_name(agent),
            velocity=self._to_pair(getattr(state, "velocity", (0.0, 0.0))),
            acceleration=self._to_pair(getattr(state, "acceleration", (0.0, 0.0))),
            position=self._to_pair(getattr(state, "position", (0.0, 0.0))),
            heading=self._to_scalar(getattr(state, "heading", None)),
            extent=self._extract_extent(agent, raw_timestep),
            metadata={"raw_timestep": raw_timestep},
        )

    @staticmethod
    def _to_pair(value: Any) -> Tuple[float, float]:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size < 2:
            return (0.0, 0.0)
        return (float(array[0]), float(array[1]))

    @staticmethod
    def _to_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return None
        return float(array[0])

    @staticmethod
    def _extract_extent(agent: Any, timestep: int) -> Optional[Tuple[float, ...]]:
        extent = getattr(agent, "extent", None)
        if extent is None:
            return None

        if hasattr(extent, "get_extents"):
            values = np.asarray(extent.get_extents(timestep, timestep), dtype=float).reshape(-1)
            return tuple(float(v) for v in values.tolist())

        if isinstance(extent, (list, tuple)):
            return tuple(float(v) for v in extent)

        dims = [getattr(extent, attr, None) for attr in ("length", "width", "height")]
        dims = [dim for dim in dims if dim is not None]
        if not dims:
            return None
        return tuple(float(v) for v in dims)

    @staticmethod
    def _compute_ttc(ego_state: Any, other_state: Any) -> Optional[float]:
        ego_position = np.asarray(getattr(ego_state, "position", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        other_position = np.asarray(getattr(other_state, "position", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        ego_velocity = np.asarray(getattr(ego_state, "velocity", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        other_velocity = np.asarray(getattr(other_state, "velocity", (0.0, 0.0)), dtype=float).reshape(-1)[:2]

        rel_position = other_position - ego_position
        rel_velocity = other_velocity - ego_velocity
        rel_speed = float(np.linalg.norm(rel_velocity))
        if rel_speed < 1e-6:
            return None

        closing_speed = -float(np.dot(rel_position, rel_velocity)) / max(float(np.linalg.norm(rel_position)), 1e-6)
        if closing_speed <= 0:
            return None

        distance = float(np.linalg.norm(rel_position))
        return distance / closing_speed if closing_speed > 0 else None
