"""
Multi-metric slicing logic for extracting one or more risk episodes per scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .scene_graph import Node, SSTG

if TYPE_CHECKING:
    from trajdata.caching.df_cache import DataFrameCache
    from trajdata.data_structures.scene_metadata import Scene


class EpisodeType(str, Enum):
    """Current episode extraction mode."""

    TTC_MIN_WINDOW = "ttc_min_window"
    MULTI_METRIC_WINDOW = "multi_metric_window"


class PeakMetricType(str, Enum):
    """Peak-selection evidence types."""

    PET = "pet"
    TTI = "tti"
    TTC = "ttc"
    DYNAMICS = "dynamics"


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
    episode_type: EpisodeType = EpisodeType.MULTI_METRIC_WINDOW
    risk_score: float = 0.0
    sstg: Optional[SSTG] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    rule_trace: List[str] = field(default_factory=list)
    semantic_timestep_map: Dict[str, int] = field(default_factory=dict)
    semantic_timestamp_order: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.semantic_timestep_map:
            self.semantic_timestep_map = {
                "T_start": self.t_start,
                "T_peak": self.t_peak,
                "T_end": self.t_end,
            }
        else:
            self.semantic_timestep_map = dict(self.semantic_timestep_map)
            self.semantic_timestep_map.setdefault("T_start", self.t_start)
            self.semantic_timestep_map.setdefault("T_peak", self.t_peak)
            self.semantic_timestep_map.setdefault("T_end", self.t_end)

        if not self.semantic_timestamp_order:
            self.semantic_timestamp_order = sorted(
                self.semantic_timestep_map.keys(),
                key=SSTG._timestamp_sort_key,
            )
        else:
            ordered = [label for label in self.semantic_timestamp_order if label in self.semantic_timestep_map]
            missing = [
                label
                for label in sorted(self.semantic_timestep_map.keys(), key=SSTG._timestamp_sort_key)
                if label not in ordered
            ]
            self.semantic_timestamp_order = ordered + missing

    @property
    def semantic_timesteps(self) -> Dict[str, int]:
        return {
            label: self.semantic_timestep_map[label]
            for label in self.semantic_timestamp_order
            if label in self.semantic_timestep_map
        }

    @property
    def ordered_timestamps(self) -> List[str]:
        return list(self.semantic_timestamp_order)

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


@dataclass(frozen=True)
class PeakCandidate:
    """Risk candidate centered at a semantic peak."""

    ego_agent_id: str
    trigger_agent_ids: Tuple[str, ...]
    t_peak: int
    metric_type: PeakMetricType
    metric_value: float
    risk_score: float
    evidence: Dict[str, Any]


class Slicer:
    """Scene slicer using PET/TTC/kinematic peaks with causal start/end heuristics."""

    METRIC_PRIORITY: Mapping[PeakMetricType, int] = {
        PeakMetricType.PET: 0,
        PeakMetricType.TTI: 1,
        PeakMetricType.TTC: 2,
        PeakMetricType.DYNAMICS: 3,
    }

    def __init__(
        self,
        pre_buffer_sec: float = 2.0,
        post_buffer_sec: float = 2.0,
        ego_motion_threshold: float = 0.5,
        ttc_event_threshold: float = 2.5,
        pet_event_threshold: float = 2.0,
        tti_event_threshold: float = 2.0,
        ttc_clear_threshold: float = 5.0,
        min_peak_gap_sec: Optional[float] = None,
        max_episodes_per_scene: Optional[int] = 50,
        reaction_latency_sec: float = 1.8,
        stable_duration_sec: float = 1.0,
        dynamic_acc_threshold: float = 3.0,
        dynamic_jerk_threshold: float = 4.0,
        lateral_speed_threshold: float = 0.5,
        launch_acc_threshold: float = 2.0,
        forward_roi_distance: float = 50.0,
        conflict_zone_radius: float = 6.0,
        visibility_fov_deg: float = 140.0,
        occlusion_lateral_threshold: float = 2.5,
        pet_prediction_horizon_sec: float = 5.0,
        tti_prediction_horizon_sec: float = 4.0,
        tti_conflict_radius: float = 3.0,
        tti_initial_roi_distance: float = 40.0,
        tti_min_intersection_angle_deg: float = 25.0,
        tti_prefilter_enabled: bool = True,
        tti_prefilter_ratio: float = 0.75,
        following_filter_enabled: bool = True,
        following_heading_threshold_deg: float = 20.0,
        following_lateral_threshold_m: float = 4.0,
        causal_max_distance_m_for_pet_ttc: float = 25.0,
        stop_speed_threshold: float = 0.1,
        zero_acc_threshold: float = 0.3,
        enable_pet_peak: bool = True,
        enable_tti_peak: bool = True,
        enable_ttc_peak: bool = True,
        enable_dynamics_peak: bool = True,
    ):
        self.pre_buffer_sec = float(pre_buffer_sec)
        self.post_buffer_sec = float(post_buffer_sec)
        self.ego_motion_threshold = float(ego_motion_threshold)
        self.ttc_event_threshold = float(ttc_event_threshold)
        self.pet_event_threshold = float(pet_event_threshold)
        self.tti_event_threshold = float(tti_event_threshold)
        self.ttc_clear_threshold = float(ttc_clear_threshold)
        self.min_peak_gap_sec = None if min_peak_gap_sec is None else float(min_peak_gap_sec)
        self.max_episodes_per_scene = (
            None if max_episodes_per_scene is None else int(max_episodes_per_scene)
        )
        self.reaction_latency_sec = float(reaction_latency_sec)
        self.stable_duration_sec = float(stable_duration_sec)
        self.dynamic_acc_threshold = float(dynamic_acc_threshold)
        self.dynamic_jerk_threshold = float(dynamic_jerk_threshold)
        self.lateral_speed_threshold = float(lateral_speed_threshold)
        self.launch_acc_threshold = float(launch_acc_threshold)
        self.forward_roi_distance = float(forward_roi_distance)
        self.conflict_zone_radius = float(conflict_zone_radius)
        self.visibility_fov_deg = float(visibility_fov_deg)
        self.occlusion_lateral_threshold = float(occlusion_lateral_threshold)
        self.pet_prediction_horizon_sec = float(pet_prediction_horizon_sec)
        self.tti_prediction_horizon_sec = float(tti_prediction_horizon_sec)
        self.tti_conflict_radius = float(tti_conflict_radius)
        self.tti_initial_roi_distance = float(tti_initial_roi_distance)
        self.tti_min_intersection_angle_deg = float(tti_min_intersection_angle_deg)
        self.tti_prefilter_enabled = bool(tti_prefilter_enabled)
        self.tti_prefilter_ratio = float(max(tti_prefilter_ratio, 1e-3))
        self.tti_ttc_prefilter_threshold = self.ttc_event_threshold / self.tti_prefilter_ratio
        self.tti_pet_prefilter_threshold = self.pet_event_threshold / self.tti_prefilter_ratio
        self.following_filter_enabled = bool(following_filter_enabled)
        self.following_heading_threshold_deg = float(following_heading_threshold_deg)
        self.following_lateral_threshold_m = float(following_lateral_threshold_m)
        self.causal_max_distance_m_for_pet_ttc = float(causal_max_distance_m_for_pet_ttc)
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.zero_acc_threshold = float(zero_acc_threshold)
        self.enable_pet_peak = bool(enable_pet_peak)
        self.enable_tti_peak = bool(enable_tti_peak)
        self.enable_ttc_peak = bool(enable_ttc_peak)
        self.enable_dynamics_peak = bool(enable_dynamics_peak)
        self._tti_cache: Dict[Tuple[str, str, int], Optional[Tuple[float, Tuple[float, float], float, float, float]]] = {}
        self._state_cache: Dict[Tuple[str, int], Optional[Any]] = {}
        self._future_trajectory_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, ...]] = {}

    def extract_episodes(self, scene: "Scene", cache: "DataFrameCache") -> List[Episode]:
        self._tti_cache: Dict[Tuple[str, str, int], Optional[Tuple[float, Tuple[float, float], float, float, float]]] = {}
        self._state_cache: Dict[Tuple[str, int], Optional[Any]] = {}
        self._future_trajectory_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, ...]] = {}
        if not getattr(scene, "agent_presence", None):
            return []

        ego_candidates = self._get_ego_candidates(scene)
        if not ego_candidates:
            return []

        pre_buffer_ts = int(round(self.pre_buffer_sec / scene.dt))
        post_buffer_ts = int(round(self.post_buffer_sec / scene.dt))
        reaction_latency_ts = max(1, int(round(self.reaction_latency_sec / scene.dt)))
        stable_duration_ts = max(1, int(round(self.stable_duration_sec / scene.dt)))
        min_peak_gap_ts = max(
            1,
            int(
                round(
                    (
                        self.min_peak_gap_sec
                        if self.min_peak_gap_sec is not None
                        else (self.pre_buffer_sec + self.post_buffer_sec)
                    )
                    / scene.dt
                )
            ),
        )

        agent_lookup = {agent.name: agent for agent in (getattr(scene, "agents", None) or [])}
        movement_cache: Dict[Tuple[str, int], bool] = {}
        peak_candidates: List[PeakCandidate] = []
        for scene_ts, agents in enumerate(scene.agent_presence):
            agents_by_name = {agent.name: agent for agent in agents}
            for ego_agent in ego_candidates:
                if ego_agent.name not in agents_by_name:
                    continue

                movement_key = (ego_agent.name, scene_ts)
                is_moving = movement_cache.get(movement_key)
                if is_moving is None:
                    is_moving = self._is_agent_moving_in_window(
                        cache=cache,
                        agent_id=ego_agent.name,
                        center_ts=scene_ts,
                        pre_buffer_ts=pre_buffer_ts,
                        post_buffer_ts=post_buffer_ts,
                        scene_length=scene.length_timesteps,
                    )
                    movement_cache[movement_key] = is_moving
                if not is_moving:
                    continue

                peak_candidates.extend(
                    self._collect_candidates_at_timestep(
                        scene=scene,
                        cache=cache,
                        ego_agent=ego_agent,
                        agents=agents,
                        scene_ts=scene_ts,
                    )
                )

        selected_candidates = self._select_peak_candidates(
            peak_candidates,
            min_peak_gap_ts=min_peak_gap_ts,
        )
        if not selected_candidates:
            return []

        episodes: List[Episode] = []
        for candidate_index, candidate in enumerate(selected_candidates):
            ego_agent = agent_lookup.get(candidate.ego_agent_id)
            if ego_agent is None:
                continue

            conflict_center = self._infer_conflict_center(
                cache=cache,
                ego_agent_id=candidate.ego_agent_id,
                trigger_agent_ids=candidate.trigger_agent_ids,
                t_peak=candidate.t_peak,
                evidence=candidate.evidence,
            )
            t_start, start_rule, start_rule_metadata = self._find_t_start(
                scene=scene,
                cache=cache,
                ego_agent_id=candidate.ego_agent_id,
                trigger_agent_ids=candidate.trigger_agent_ids,
                t_peak=candidate.t_peak,
                reaction_latency_ts=reaction_latency_ts,
                conflict_center=conflict_center,
            )
            t_end, end_rule = self._find_t_end(
                scene=scene,
                cache=cache,
                ego_agent_id=candidate.ego_agent_id,
                trigger_agent_ids=candidate.trigger_agent_ids,
                t_peak=candidate.t_peak,
                fallback_post_buffer_ts=post_buffer_ts,
                stable_duration_ts=stable_duration_ts,
            )

            semantic_timestep_map = self._build_semantic_timestep_map(
                t_start=t_start,
                t_peak=candidate.t_peak,
                t_end=t_end,
            )
            state_snapshots: Dict[str, Dict[str, Node]] = {}
            involved_agents = {candidate.ego_agent_id, *candidate.trigger_agent_ids}
            for label, timestep in semantic_timestep_map.items():
                snapshot = self._collect_snapshot(scene, cache, timestep, label)
                if snapshot:
                    involved_agents.update(snapshot.keys())
                state_snapshots[label] = snapshot

            risk_score = float(np.clip(candidate.risk_score, 0.0, 1.0))
            trigger_agent_ids = list(candidate.trigger_agent_ids)
            primary_trigger_id = trigger_agent_ids[0] if trigger_agent_ids else None
            trigger_track_windows = {
                agent_id: self._describe_track_window(
                    scene=scene,
                    cache=cache,
                    agent_id=agent_id,
                    reference_ts=candidate.t_peak,
                )
                for agent_id in trigger_agent_ids
            }
            metadata = {
                "trigger_agent_id": primary_trigger_id,
                "trigger_agent_ids": trigger_agent_ids,
                "primary_risk_agent_ids": trigger_agent_ids,
                "trigger_track_windows": trigger_track_windows,
                "min_ttc": candidate.evidence.get("ttc"),
                "min_pet": candidate.evidence.get("pet"),
                "min_tti": candidate.evidence.get("tti"),
                "peak_metric_type": candidate.metric_type.value,
                "peak_metric_value": candidate.metric_value,
                "peak_metric_score": risk_score,
                "peak_selection_reason": candidate.evidence.get("selection_reason", candidate.metric_type.value),
                "peak_evidence": self._json_safe(candidate.evidence),
                "semantic_timesteps": {
                    label: timestep
                    for label, timestep in semantic_timestep_map.items()
                },
                "semantic_timestamp_order": list(semantic_timestep_map.keys()),
                "semantic_snapshot_count": len(semantic_timestep_map),
                "start_rule": start_rule,
                "end_rule": end_rule,
                "peak_inside_conflict_zone": start_rule_metadata["peak_inside_conflict_zone"],
                "start_rule_boundary_crossing_suppressed": start_rule_metadata[
                    "start_rule_boundary_crossing_suppressed"
                ],
                "start_rule_boundary_crossing_suppressed_reason": start_rule_metadata[
                    "start_rule_boundary_crossing_suppressed_reason"
                ],
                "ego_motion_threshold": self.ego_motion_threshold,
                "ttc_event_threshold": self.ttc_event_threshold,
                "pet_event_threshold": self.pet_event_threshold,
                "tti_event_threshold": self.tti_event_threshold,
                "causal_max_distance_m_for_pet_ttc": self.causal_max_distance_m_for_pet_ttc,
                "tti_prefilter_enabled": self.tti_prefilter_enabled,
                "tti_prefilter_ratio": self.tti_prefilter_ratio,
                "tti_ttc_prefilter_threshold": self.tti_ttc_prefilter_threshold,
                "tti_pet_prefilter_threshold": self.tti_pet_prefilter_threshold,
                "following_filter_enabled": self.following_filter_enabled,
                "following_heading_threshold_deg": self.following_heading_threshold_deg,
                "following_lateral_threshold_m": self.following_lateral_threshold_m,
                "reaction_latency_sec": self.reaction_latency_sec,
                "stable_duration_sec": self.stable_duration_sec,
                "candidate_rank_in_scene": candidate_index,
                "conflict_zone": {
                    "center": list(conflict_center) if conflict_center is not None else None,
                    "radius": self.conflict_zone_radius,
                    "source": candidate.evidence.get("conflict_zone_source", "heuristic"),
                },
            }

            episodes.append(
                Episode(
                    scene_id=f"{scene.env_name}:{scene.name}",
                    scene_name=scene.name,
                    env_name=scene.env_name,
                    dt=scene.dt,
                    ego_agent_id=ego_agent.name,
                    t_start=t_start,
                    t_peak=candidate.t_peak,
                    t_end=t_end,
                    involved_agents=sorted(involved_agents),
                    state_snapshots=state_snapshots,
                    episode_type=EpisodeType.MULTI_METRIC_WINDOW,
                    risk_score=risk_score,
                    metadata=metadata,
                    semantic_timestep_map=semantic_timestep_map,
                    semantic_timestamp_order=list(semantic_timestep_map.keys()),
                )
            )

        return episodes

    def _describe_track_window(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        agent_id: str,
        reference_ts: int,
    ) -> Dict[str, Any]:
        birth_ts = reference_ts
        while birth_ts > 0 and self._safe_get_state(cache, agent_id, birth_ts - 1) is not None:
            birth_ts -= 1

        death_ts = reference_ts
        while death_ts + 1 < scene.length_timesteps and self._safe_get_state(cache, agent_id, death_ts + 1) is not None:
            death_ts += 1

        birth_state = self._safe_get_state(cache, agent_id, birth_ts)
        death_state = self._safe_get_state(cache, agent_id, death_ts)
        return {
            "birth_ts": birth_ts,
            "death_ts": death_ts,
            "birth_position": self._to_pair(getattr(birth_state, "position", (0.0, 0.0))) if birth_state is not None else None,
            "death_position": self._to_pair(getattr(death_state, "position", (0.0, 0.0))) if death_state is not None else None,
        }

    def _collect_candidates_at_timestep(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent: Any,
        agents: Sequence[Any],
        scene_ts: int,
    ) -> List[PeakCandidate]:
        candidates: List[PeakCandidate] = []
        ego_state = self._safe_get_state(cache, ego_agent.name, scene_ts)
        if ego_state is None:
            return candidates

        best_pet: Optional[Tuple[str, float, Tuple[float, float], float, float]] = None
        best_tti: Optional[Tuple[str, float, Tuple[float, float], float, float, float, float]] = None
        best_ttc: Optional[Tuple[str, float]] = None
        nearest_agent_id: Optional[str] = None
        nearest_distance = float("inf")

        for agent in agents:
            if agent.name == ego_agent.name:
                continue
            other_state = self._safe_get_state(cache, agent.name, scene_ts)
            if other_state is None:
                continue

            distance = self._state_distance(ego_state, other_state)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_agent_id = agent.name

            if self.following_filter_enabled and self._is_following_behavior(ego_state, other_state):
                continue

            needs_ttc = self.enable_ttc_peak or (self.enable_tti_peak and self.tti_prefilter_enabled)
            needs_pet = self.enable_pet_peak or (self.enable_tti_peak and self.tti_prefilter_enabled)
            within_causal_distance = distance <= self.causal_max_distance_m_for_pet_ttc

            ttc: Optional[float] = None
            if needs_ttc:
                ttc = self._compute_ttc(ego_state, other_state)
                if (
                    ttc is not None
                    and within_causal_distance
                    and ttc < self.ttc_event_threshold
                ):
                    if best_ttc is None or ttc < best_ttc[1]:
                        best_ttc = (agent.name, ttc)

            pet: Optional[Tuple[float, Tuple[float, float], float, float]] = None
            if needs_pet:
                pet = self._compute_pet(ego_state, other_state, scene.dt)
                if (
                    pet is not None
                    and within_causal_distance
                    and pet[0] < self.pet_event_threshold
                ):
                    if best_pet is None or pet[0] < best_pet[1]:
                        best_pet = (agent.name, pet[0], pet[1], pet[2], pet[3])

            if self.enable_tti_peak:
                if self.tti_prefilter_enabled:
                    relaxed_ttc_pass = ttc is not None and ttc < self.tti_ttc_prefilter_threshold
                    relaxed_pet_pass = pet is not None and pet[0] < self.tti_pet_prefilter_threshold
                    if not (relaxed_ttc_pass or relaxed_pet_pass):
                        continue
                tti = self._compute_tti(
                    cache=cache,
                    ego_agent_id=ego_agent.name,
                    other_agent_id=agent.name,
                    scene_ts=scene_ts,
                    dt=scene.dt,
                )
                if tti is not None and tti[0] < self.tti_event_threshold:
                    if best_tti is None or tti[0] < best_tti[1]:
                        best_tti = (agent.name, tti[0], tti[1], tti[2], tti[3], tti[4], distance)

        if self.enable_pet_peak and best_pet is not None:
            agent_id, pet_value, conflict_point, ego_arrival, trigger_arrival = best_pet
            candidates.append(
                PeakCandidate(
                    ego_agent_id=ego_agent.name,
                    trigger_agent_ids=(agent_id,),
                    t_peak=scene_ts,
                    metric_type=PeakMetricType.PET,
                    metric_value=pet_value,
                    risk_score=max(0.0, 1.0 - pet_value / max(self.pet_event_threshold, 1e-6)),
                    evidence={
                        "pet": pet_value,
                        "conflict_point": conflict_point,
                        "ego_arrival_sec": ego_arrival,
                        "trigger_arrival_sec": trigger_arrival,
                        "selection_reason": "pet_global_min_heuristic",
                        "conflict_zone_source": "pet_intersection_heuristic",
                    },
                )
            )

        if best_tti is not None:
            agent_id, tti_value, conflict_point, ego_arrival, trigger_arrival, heading_diff_deg, current_distance = best_tti
            tti_score = max(0.0, 1.0 - tti_value / max(self.tti_event_threshold, 1e-6))
            urgency_score = max(
                0.0,
                1.0 - max(ego_arrival, trigger_arrival) / max(self.tti_prediction_horizon_sec, 1e-6),
            )
            proximity_score = max(
                0.0,
                1.0 - current_distance / max(self.tti_initial_roi_distance, 1e-6),
            )
            candidates.append(
                PeakCandidate(
                    ego_agent_id=ego_agent.name,
                    trigger_agent_ids=(agent_id,),
                    t_peak=scene_ts,
                    metric_type=PeakMetricType.TTI,
                    metric_value=tti_value,
                    risk_score=float(np.clip(0.6 * tti_score + 0.25 * urgency_score + 0.15 * proximity_score, 0.0, 1.0)),
                    evidence={
                        "tti": tti_value,
                        "conflict_point": conflict_point,
                        "ego_arrival_sec": ego_arrival,
                        "trigger_arrival_sec": trigger_arrival,
                        "trajectory_intersection_angle_deg": heading_diff_deg,
                        "current_agent_distance_m": current_distance,
                        "tti_prefilter": {
                            "enabled": self.tti_prefilter_enabled,
                            "ttc_threshold": self.tti_ttc_prefilter_threshold,
                            "pet_threshold": self.tti_pet_prefilter_threshold,
                        },
                        "selection_reason": "tti_future_trajectory_conflict",
                        "conflict_zone_source": "tti_future_overlap_heuristic",
                    },
                )
            )

        if self.enable_ttc_peak and best_ttc is not None:
            agent_id, ttc_value = best_ttc
            candidates.append(
                PeakCandidate(
                    ego_agent_id=ego_agent.name,
                    trigger_agent_ids=(agent_id,),
                    t_peak=scene_ts,
                    metric_type=PeakMetricType.TTC,
                    metric_value=ttc_value,
                    risk_score=max(0.0, 1.0 - ttc_value / max(self.ttc_event_threshold, 1e-6)),
                    evidence={
                        "ttc": ttc_value,
                        "selection_reason": "ttc_global_min",
                        "conflict_zone_source": "midpoint_heuristic",
                    },
                )
            )

        dynamics_candidate = None
        if self.enable_dynamics_peak:
            dynamics_candidate = self._compute_dynamics_candidate(
                scene=scene,
                cache=cache,
                ego_agent=ego_agent,
                ego_state=ego_state,
                scene_ts=scene_ts,
                nearest_agent_id=nearest_agent_id,
                best_ttc_agent_id=best_ttc[0] if best_ttc is not None else None,
                best_pet_agent_id=(best_pet or best_tti)[0] if (best_pet is not None or best_tti is not None) else None,
            )
        if dynamics_candidate is not None:
            candidates.append(dynamics_candidate)

        return candidates

    def _select_peak_candidates(
        self,
        peak_candidates: List[PeakCandidate],
        min_peak_gap_ts: int,
    ) -> List[PeakCandidate]:
        if not peak_candidates:
            return []

        ranked_candidates = sorted(
            peak_candidates,
            key=lambda candidate: (
                self.METRIC_PRIORITY[candidate.metric_type],
                -candidate.risk_score,
                candidate.t_peak,
                candidate.metric_value,
            ),
        )
        selected: List[PeakCandidate] = []
        for candidate in ranked_candidates:
            if any(abs(candidate.t_peak - existing.t_peak) < min_peak_gap_ts for existing in selected):
                continue
            selected.append(candidate)
            if self.max_episodes_per_scene is not None and len(selected) >= self.max_episodes_per_scene:
                break

        return sorted(selected, key=lambda candidate: candidate.t_peak)

    def _compute_dynamics_candidate(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent: Any,
        ego_state: Any,
        scene_ts: int,
        nearest_agent_id: Optional[str],
        best_ttc_agent_id: Optional[str],
        best_pet_agent_id: Optional[str],
    ) -> Optional[PeakCandidate]:
        acceleration = np.asarray(getattr(ego_state, "acceleration", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        prev_state = self._safe_get_state(cache, ego_agent.name, scene_ts - 1) if scene_ts > 0 else None
        prev_acceleration = (
            np.asarray(getattr(prev_state, "acceleration", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
            if prev_state is not None
            else np.zeros(2, dtype=float)
        )
        jerk = float(np.linalg.norm(acceleration - prev_acceleration) / max(scene.dt, 1e-6))
        longitudinal_acc = abs(self._project_longitudinal_component(ego_state, acceleration))
        if longitudinal_acc < self.dynamic_acc_threshold and jerk < self.dynamic_jerk_threshold:
            return None

        trigger_agent_id = best_pet_agent_id or best_ttc_agent_id or nearest_agent_id
        if trigger_agent_id is None:
            return None

        metric_value = max(longitudinal_acc, jerk)
        risk_score = min(
            1.0,
            max(
                longitudinal_acc / max(self.dynamic_acc_threshold, 1e-6),
                jerk / max(self.dynamic_jerk_threshold, 1e-6),
            )
            / 2.0,
        )
        return PeakCandidate(
            ego_agent_id=ego_agent.name,
            trigger_agent_ids=(trigger_agent_id,),
            t_peak=scene_ts,
            metric_type=PeakMetricType.DYNAMICS,
            metric_value=metric_value,
            risk_score=risk_score,
            evidence={
                "ego_longitudinal_acc": longitudinal_acc,
                "ego_jerk": jerk,
                "selection_reason": "ego_kinematic_spike",
                "conflict_zone_source": "trigger_midpoint_heuristic",
            },
        )

    def _find_t_start(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        t_peak: int,
        reaction_latency_ts: int,
        conflict_center: Optional[Tuple[float, float]],
    ) -> Tuple[int, str, Dict[str, Any]]:
        search_floor = max(0, t_peak - max(reaction_latency_ts * 4, 1))
        suppression_metadata = self._get_boundary_crossing_suppression_metadata(
            cache=cache,
            ego_agent_id=ego_agent_id,
            trigger_agent_ids=trigger_agent_ids,
            t_peak=t_peak,
            conflict_center=conflict_center,
        )
        for timestep in range(t_peak, search_floor, -1):
            for trigger_agent_id in trigger_agent_ids:
                if self._detect_topological_flip(scene, cache, ego_agent_id, trigger_agent_id, timestep):
                    return timestep, "topological_flip_heuristic", suppression_metadata
                if (
                    not suppression_metadata["start_rule_boundary_crossing_suppressed"]
                    and self._detect_boundary_crossing(cache, trigger_agent_id, timestep, conflict_center)
                ):
                    return timestep, "boundary_crossing_heuristic", suppression_metadata
                if self._detect_intent_mutation(cache, trigger_agent_id, timestep):
                    return timestep, "intent_mutation", suppression_metadata

        return max(0, t_peak - reaction_latency_ts), "fallback_latency", suppression_metadata

    def _build_semantic_timestep_map(
        self,
        t_start: int,
        t_peak: int,
        t_end: int,
    ) -> Dict[str, int]:
        semantic_points: List[Tuple[str, int]] = [("T_start", t_start), ("T_peak", t_peak)]
        peak_to_end_gap = max(0, t_end - t_peak)
        mid_timesteps: List[int] = []
        if 51 <= peak_to_end_gap <= 100:
            mid_timesteps = [t_peak + peak_to_end_gap // 2]
        elif peak_to_end_gap > 100:
            mid_timesteps = [
                t_peak + peak_to_end_gap // 3,
                t_peak + (2 * peak_to_end_gap) // 3,
            ]

        deduped_mid_timesteps: List[int] = []
        for timestep in mid_timesteps:
            if t_peak < timestep < t_end and timestep not in deduped_mid_timesteps:
                deduped_mid_timesteps.append(timestep)

        for index, timestep in enumerate(deduped_mid_timesteps, start=1):
            semantic_points.append((f"T_mid_{index}", timestep))
        semantic_points.append(("T_end", t_end))
        return dict(semantic_points)

    def _get_boundary_crossing_suppression_metadata(
        self,
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        t_peak: int,
        conflict_center: Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:
        if conflict_center is None:
            return {
                "peak_inside_conflict_zone": False,
                "start_rule_boundary_crossing_suppressed": False,
                "start_rule_boundary_crossing_suppressed_reason": None,
            }

        peak_inside_conflict_zone = False
        ego_state = self._safe_get_state(cache, ego_agent_id, t_peak)
        if ego_state is not None and self._is_in_conflict_zone(ego_state, conflict_center):
            peak_inside_conflict_zone = True
        else:
            for trigger_agent_id in trigger_agent_ids:
                trigger_state = self._safe_get_state(cache, trigger_agent_id, t_peak)
                if trigger_state is not None and self._is_in_conflict_zone(trigger_state, conflict_center):
                    peak_inside_conflict_zone = True
                    break

        suppressed_reason = (
            "peak_agent_already_inside_conflict_zone"
            if peak_inside_conflict_zone
            else None
        )
        return {
            "peak_inside_conflict_zone": peak_inside_conflict_zone,
            "start_rule_boundary_crossing_suppressed": peak_inside_conflict_zone,
            "start_rule_boundary_crossing_suppressed_reason": suppressed_reason,
        }

    def _find_t_end(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        t_peak: int,
        fallback_post_buffer_ts: int,
        stable_duration_ts: int,
    ) -> Tuple[int, str]:
        fallback_end = min(scene.length_timesteps - 1, t_peak + fallback_post_buffer_ts)
        for timestep in range(t_peak, scene.length_timesteps):
            window_end = timestep + stable_duration_ts - 1
            if window_end >= scene.length_timesteps:
                break

            if all(
                self._is_risk_clear(scene, cache, ego_agent_id, trigger_agent_ids, check_ts)
                for check_ts in range(timestep, window_end + 1)
            ):
                return window_end, "risk_clear"

            if all(
                self._is_ego_dynamics_zeroed(cache, ego_agent_id, check_ts)
                for check_ts in range(timestep, window_end + 1)
            ):
                return window_end, "dynamics_zeroing"

            if all(
                self._is_out_of_roi(cache, ego_agent_id, trigger_agent_ids, check_ts)
                for check_ts in range(timestep, window_end + 1)
            ):
                return window_end, "out_of_roi"

        return fallback_end, "fallback_post_buffer"

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

    def _safe_get_state(self, cache: "DataFrameCache", agent_id: str, scene_ts: int) -> Optional[Any]:
        if scene_ts < 0:
            return None
        cache_key = (agent_id, scene_ts)
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        try:
            state = cache.get_state(agent_id, scene_ts)
        except Exception:
            state = None
        self._state_cache[cache_key] = state
        return state

    def _get_ego_candidates(self, scene: "Scene") -> List[Any]:
        agents = getattr(scene, "agents", None) or []
        named_ego = [
            agent
            for agent in agents
            if agent.name == "ego" and self._agent_type_name(agent).upper() == "VEHICLE"
        ]
        if named_ego:
            return named_ego

        return [
            agent
            for agent in agents
            if self._agent_type_name(agent).upper() == "VEHICLE"
        ]

    def _is_agent_moving_in_window(
        self,
        cache: "DataFrameCache",
        agent_id: str,
        center_ts: int,
        pre_buffer_ts: int,
        post_buffer_ts: int,
        scene_length: int,
    ) -> bool:
        t_start = max(0, center_ts - pre_buffer_ts)
        t_end = min(scene_length - 1, center_ts + post_buffer_ts)

        max_speed = 0.0
        for timestep in range(t_start, t_end + 1):
            state = self._safe_get_state(cache, agent_id, timestep)
            if state is None:
                continue
            velocity = np.asarray(getattr(state, "velocity", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
            max_speed = max(max_speed, float(np.linalg.norm(velocity)))
            if max_speed >= self.ego_motion_threshold:
                return True

        return False

    def _detect_topological_flip(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_id: str,
        timestep: int,
    ) -> bool:
        if timestep <= 0:
            return False
        current_target_state = self._safe_get_state(cache, trigger_agent_id, timestep)
        previous_target_state = self._safe_get_state(cache, trigger_agent_id, timestep - 1)
        if current_target_state is None or previous_target_state is None:
            return False
        if self._safe_get_state(cache, ego_agent_id, timestep) is None:
            return False
        if self._safe_get_state(cache, ego_agent_id, timestep - 1) is None:
            return False
        is_visible_now = self._is_visible_to_ego(scene, cache, ego_agent_id, trigger_agent_id, timestep)
        was_hidden_before = not self._is_visible_to_ego(scene, cache, ego_agent_id, trigger_agent_id, timestep - 1)
        return is_visible_now and was_hidden_before

    def _detect_boundary_crossing(
        self,
        cache: "DataFrameCache",
        trigger_agent_id: str,
        timestep: int,
        conflict_center: Optional[Tuple[float, float]],
    ) -> bool:
        if timestep <= 0 or conflict_center is None:
            return False
        prev_state = self._safe_get_state(cache, trigger_agent_id, timestep - 1)
        current_state = self._safe_get_state(cache, trigger_agent_id, timestep)
        if prev_state is None or current_state is None:
            return False
        was_inside = self._is_in_conflict_zone(prev_state, conflict_center)
        is_inside = self._is_in_conflict_zone(current_state, conflict_center)
        return (not was_inside) and is_inside

    def _detect_intent_mutation(
        self,
        cache: "DataFrameCache",
        trigger_agent_id: str,
        timestep: int,
    ) -> bool:
        if timestep <= 0:
            return False
        prev_state = self._safe_get_state(cache, trigger_agent_id, timestep - 1)
        current_state = self._safe_get_state(cache, trigger_agent_id, timestep)
        if prev_state is None or current_state is None:
            return False

        current_lateral_speed = abs(self._project_lateral_speed(current_state))
        prev_lateral_speed = abs(self._project_lateral_speed(prev_state))
        if current_lateral_speed > self.lateral_speed_threshold and prev_lateral_speed <= self.lateral_speed_threshold:
            return True

        current_lon_acc = abs(
            self._project_longitudinal_component(
                current_state,
                np.asarray(getattr(current_state, "acceleration", (0.0, 0.0)), dtype=float).reshape(-1)[:2],
            )
        )
        prev_lon_acc = abs(
            self._project_longitudinal_component(
                prev_state,
                np.asarray(getattr(prev_state, "acceleration", (0.0, 0.0)), dtype=float).reshape(-1)[:2],
            )
        )
        return current_lon_acc > self.launch_acc_threshold and prev_lon_acc <= self.launch_acc_threshold

    def _is_visible_to_ego(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent_id: str,
        target_agent_id: str,
        timestep: int,
    ) -> bool:
        ego_state = self._safe_get_state(cache, ego_agent_id, timestep)
        target_state = self._safe_get_state(cache, target_agent_id, timestep)
        if ego_state is None or target_state is None:
            return False

        ego_position = self._state_position(ego_state)
        target_position = self._state_position(target_state)
        if ego_position is None or target_position is None:
            return False

        delta = target_position - ego_position
        distance = float(np.linalg.norm(delta))
        if distance > self.forward_roi_distance * 1.5:
            return False

        ego_heading = self._state_heading(ego_state)
        rel_angle = math.degrees(
            math.atan2(
                math.sin(math.atan2(float(delta[1]), float(delta[0])) - ego_heading),
                math.cos(math.atan2(float(delta[1]), float(delta[0])) - ego_heading),
            )
        )
        if abs(rel_angle) > self.visibility_fov_deg / 2.0:
            return False

        for agent in self._agents_at_timestep(scene, timestep):
            if agent.name in {ego_agent_id, target_agent_id}:
                continue
            blocker_state = self._safe_get_state(cache, agent.name, timestep)
            if blocker_state is None:
                continue
            blocker_position = self._state_position(blocker_state)
            if blocker_position is None:
                continue
            blocker_delta = blocker_position - ego_position
            blocker_distance = float(np.linalg.norm(blocker_delta))
            if blocker_distance >= distance:
                continue
            lateral_distance = self._point_to_ray_distance(blocker_position, ego_position, target_position)
            if lateral_distance < self.occlusion_lateral_threshold:
                return False

        return True

    def _is_risk_clear(
        self,
        scene: "Scene",
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        timestep: int,
    ) -> bool:
        ego_state = self._safe_get_state(cache, ego_agent_id, timestep)
        if ego_state is None:
            return False

        for trigger_agent_id in trigger_agent_ids:
            trigger_state = self._safe_get_state(cache, trigger_agent_id, timestep)
            if trigger_state is None:
                continue

            ttc = self._compute_ttc(ego_state, trigger_state)
            pet = self._compute_pet(ego_state, trigger_state, 0.1)
            tti = self._compute_tti(
                cache=cache,
                ego_agent_id=ego_agent_id,
                other_agent_id=trigger_agent_id,
                scene_ts=timestep,
                dt=scene.dt,
            )
            rel_position = self._state_position(trigger_state) - self._state_position(ego_state)
            rel_velocity = self._state_velocity(trigger_state) - self._state_velocity(ego_state)
            distancing = float(np.dot(rel_position, rel_velocity)) > 0.0
            ttc_clear = ttc is None or ttc > self.ttc_clear_threshold
            pet_clear = pet is None or pet[0] > self.pet_event_threshold
            tti_clear = tti is None or tti[0] > self.tti_event_threshold
            if not ((ttc_clear and pet_clear and tti_clear and distancing) or float(np.linalg.norm(rel_position)) > self.forward_roi_distance):
                return False

        return True

    def _is_ego_dynamics_zeroed(
        self,
        cache: "DataFrameCache",
        ego_agent_id: str,
        timestep: int,
    ) -> bool:
        ego_state = self._safe_get_state(cache, ego_agent_id, timestep)
        if ego_state is None:
            return False
        speed = float(np.linalg.norm(self._state_velocity(ego_state)))
        accel = np.asarray(getattr(ego_state, "acceleration", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        lon_acc = abs(self._project_longitudinal_component(ego_state, accel))
        return speed < self.stop_speed_threshold and lon_acc < self.zero_acc_threshold

    def _is_out_of_roi(
        self,
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        timestep: int,
    ) -> bool:
        ego_state = self._safe_get_state(cache, ego_agent_id, timestep)
        if ego_state is None:
            return False
        ego_position = self._state_position(ego_state)
        ego_heading = self._state_heading(ego_state)

        for trigger_agent_id in trigger_agent_ids:
            trigger_state = self._safe_get_state(cache, trigger_agent_id, timestep)
            if trigger_state is None:
                continue
            trigger_position = self._state_position(trigger_state)
            delta = trigger_position - ego_position
            distance = float(np.linalg.norm(delta))
            if distance > self.forward_roi_distance:
                continue
            rel_angle = math.degrees(
                math.atan2(
                    math.sin(math.atan2(float(delta[1]), float(delta[0])) - ego_heading),
                    math.cos(math.atan2(float(delta[1]), float(delta[0])) - ego_heading),
                )
            )
            if abs(rel_angle) > 100.0:
                continue
            return False
        return True

    def _infer_conflict_center(
        self,
        cache: "DataFrameCache",
        ego_agent_id: str,
        trigger_agent_ids: Sequence[str],
        t_peak: int,
        evidence: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        if "conflict_point" in evidence and evidence["conflict_point"] is not None:
            point = evidence["conflict_point"]
            return (float(point[0]), float(point[1]))

        ego_state = self._safe_get_state(cache, ego_agent_id, t_peak)
        trigger_id = trigger_agent_ids[0] if trigger_agent_ids else None
        trigger_state = self._safe_get_state(cache, trigger_id, t_peak) if trigger_id is not None else None
        if ego_state is None or trigger_state is None:
            return None

        midpoint = (self._state_position(ego_state) + self._state_position(trigger_state)) / 2.0
        return (float(midpoint[0]), float(midpoint[1]))

    def _is_in_conflict_zone(
        self,
        state: Any,
        conflict_center: Tuple[float, float],
    ) -> bool:
        position = self._state_position(state)
        center = np.asarray(conflict_center, dtype=float)
        return float(np.linalg.norm(position - center)) <= self.conflict_zone_radius

    def _compute_pet(
        self,
        ego_state: Any,
        other_state: Any,
        dt: float,
    ) -> Optional[Tuple[float, Tuple[float, float], float, float]]:
        ego_position = self._state_position(ego_state)
        other_position = self._state_position(other_state)
        ego_velocity = self._state_velocity(ego_state)
        other_velocity = self._state_velocity(other_state)

        if float(np.linalg.norm(ego_velocity)) < self.ego_motion_threshold:
            return None
        if float(np.linalg.norm(other_velocity)) < self.ego_motion_threshold:
            return None

        system = np.column_stack((ego_velocity, -other_velocity))
        determinant = float(np.linalg.det(system))
        if abs(determinant) < 1e-3:
            return None

        try:
            ego_arrival, other_arrival = np.linalg.solve(system, other_position - ego_position)
        except np.linalg.LinAlgError:
            return None

        if ego_arrival < 0.0 or other_arrival < 0.0:
            return None
        if ego_arrival > self.pet_prediction_horizon_sec or other_arrival > self.pet_prediction_horizon_sec:
            return None

        conflict_point = ego_position + ego_velocity * ego_arrival
        pet = abs(float(ego_arrival - other_arrival))
        return (
            pet,
            (float(conflict_point[0]), float(conflict_point[1])),
            float(ego_arrival),
            float(other_arrival),
        )

    def _compute_tti(
        self,
        cache: "DataFrameCache",
        ego_agent_id: str,
        other_agent_id: str,
        scene_ts: int,
        dt: float,
    ) -> Optional[Tuple[float, Tuple[float, float], float, float, float]]:
        cache_key = (ego_agent_id, other_agent_id, scene_ts)
        if hasattr(self, "_tti_cache") and cache_key in self._tti_cache:
            return self._tti_cache[cache_key]

        ego_state = self._safe_get_state(cache, ego_agent_id, scene_ts)
        other_state = self._safe_get_state(cache, other_agent_id, scene_ts)
        if ego_state is None or other_state is None:
            return self._cache_tti(cache_key, None)

        if self._state_distance(ego_state, other_state) > self.tti_initial_roi_distance:
            return self._cache_tti(cache_key, None)

        horizon_ts = max(1, int(round(self.tti_prediction_horizon_sec / max(dt, 1e-6))))
        ego_future = np.asarray(self._collect_future_trajectory(cache, ego_agent_id, scene_ts, horizon_ts), dtype=float)
        other_future = np.asarray(self._collect_future_trajectory(cache, other_agent_id, scene_ts, horizon_ts), dtype=float)
        if len(ego_future) < 2 or len(other_future) < 2:
            return self._cache_tti(cache_key, None)

        ego_headings = np.asarray(self._trajectory_headings(ego_future), dtype=float)
        other_headings = np.asarray(self._trajectory_headings(other_future), dtype=float)

        deltas = ego_future[:, None, :] - other_future[None, :, :]
        distance_matrix = np.linalg.norm(deltas, axis=-1)
        heading_diff_matrix = np.abs(
            np.degrees(
                np.arctan2(
                    np.sin(ego_headings[:, None] - other_headings[None, :]),
                    np.cos(ego_headings[:, None] - other_headings[None, :]),
                )
            )
        )
        valid_mask = (
            (distance_matrix <= self.tti_conflict_radius)
            & (heading_diff_matrix >= self.tti_min_intersection_angle_deg)
            & (heading_diff_matrix <= 180.0 - self.tti_min_intersection_angle_deg)
        )
        valid_indices = np.argwhere(valid_mask)
        if valid_indices.size == 0:
            return self._cache_tti(cache_key, None)

        arrival_diffs = np.abs(valid_indices[:, 0] - valid_indices[:, 1]) * dt
        distances = distance_matrix[valid_mask]
        tie_breakers = np.minimum(valid_indices[:, 0], valid_indices[:, 1]) * dt
        order = np.lexsort((tie_breakers, distances, arrival_diffs))
        best_ego_index, best_other_index = valid_indices[order[0]]
        conflict_point = (ego_future[best_ego_index] + other_future[best_other_index]) / 2.0
        result = (
            float(max(arrival_diffs[order[0]], dt)),
            (float(conflict_point[0]), float(conflict_point[1])),
            float(best_ego_index * dt),
            float(best_other_index * dt),
            float(heading_diff_matrix[best_ego_index, best_other_index]),
        )
        return self._cache_tti(cache_key, result)

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

    def _collect_future_trajectory(
        self,
        cache: "DataFrameCache",
        agent_id: str,
        scene_ts: int,
        horizon_ts: int,
    ) -> List[np.ndarray]:
        cache_key = (agent_id, scene_ts, horizon_ts)
        if cache_key in self._future_trajectory_cache:
            return [point.copy() for point in self._future_trajectory_cache[cache_key]]

        positions: List[np.ndarray] = []
        for timestep in range(scene_ts, scene_ts + horizon_ts + 1):
            state = self._safe_get_state(cache, agent_id, timestep)
            if state is None:
                break
            positions.append(self._state_position(state))
        self._future_trajectory_cache[cache_key] = tuple(point.copy() for point in positions)
        return positions

    def _trajectory_headings(self, positions: Sequence[np.ndarray]) -> List[float]:
        if len(positions) == 0:
            return []
        if len(positions) == 1:
            return [0.0]

        headings: List[float] = []
        for index, position in enumerate(positions):
            if index == len(positions) - 1:
                delta = position - positions[index - 1]
            else:
                delta = positions[index + 1] - position
            if float(np.linalg.norm(delta)) < 1e-6:
                if headings:
                    headings.append(headings[-1])
                else:
                    headings.append(0.0)
                continue
            headings.append(float(math.atan2(float(delta[1]), float(delta[0]))))
        return headings

    def _cache_tti(
        self,
        cache_key: Tuple[str, str, int],
        result: Optional[Tuple[float, Tuple[float, float], float, float, float]],
    ) -> Optional[Tuple[float, Tuple[float, float], float, float, float]]:
        if hasattr(self, "_tti_cache"):
            self._tti_cache[cache_key] = result
        return result

    def _absolute_angle_diff_deg(self, angle_a: float, angle_b: float) -> float:
        return abs(math.degrees(math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))))

    def _is_following_behavior(self, ego_state: Any, other_state: Any) -> bool:
        ego_heading = self._motion_heading(ego_state)
        other_heading = self._motion_heading(other_state)
        heading_diff_deg = self._absolute_angle_diff_deg(ego_heading, other_heading)
        if heading_diff_deg > self.following_heading_threshold_deg:
            return False

        ego_position = self._state_position(ego_state)
        other_position = self._state_position(other_state)
        delta = other_position - ego_position
        direction = np.asarray([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
        lateral_direction = np.asarray([-direction[1], direction[0]], dtype=float)
        longitudinal = float(np.dot(delta, direction))
        lateral = abs(float(np.dot(delta, lateral_direction)))

        if lateral > self.following_lateral_threshold_m:
            return False
        if abs(longitudinal) <= lateral:
            return False
        return True

    def _motion_heading(self, state: Any) -> float:
        velocity = self._state_velocity(state)
        if float(np.linalg.norm(velocity)) >= self.ego_motion_threshold:
            return float(math.atan2(float(velocity[1]), float(velocity[0])))
        return self._state_heading(state)

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
    def _state_position(state: Any) -> np.ndarray:
        return np.asarray(getattr(state, "position", (0.0, 0.0)), dtype=float).reshape(-1)[:2]

    @staticmethod
    def _state_velocity(state: Any) -> np.ndarray:
        return np.asarray(getattr(state, "velocity", (0.0, 0.0)), dtype=float).reshape(-1)[:2]

    def _state_heading(self, state: Any) -> float:
        heading = self._to_scalar(getattr(state, "heading", None))
        if heading is not None:
            return float(heading)
        velocity = self._state_velocity(state)
        if float(np.linalg.norm(velocity)) < 1e-6:
            return 0.0
        return float(math.atan2(float(velocity[1]), float(velocity[0])))

    def _project_longitudinal_component(self, state: Any, vector: np.ndarray) -> float:
        heading = self._state_heading(state)
        direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
        return float(np.dot(vector, direction))

    def _project_lateral_speed(self, state: Any) -> float:
        velocity = self._state_velocity(state)
        heading = self._state_heading(state)
        lateral_direction = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
        return float(np.dot(velocity, lateral_direction))

    @staticmethod
    def _state_distance(state_a: Any, state_b: Any) -> float:
        return float(np.linalg.norm(Slicer._state_position(state_a) - Slicer._state_position(state_b)))

    @staticmethod
    def _point_to_ray_distance(point: np.ndarray, ray_origin: np.ndarray, ray_target: np.ndarray) -> float:
        ray_vector = ray_target - ray_origin
        ray_norm = float(np.linalg.norm(ray_vector))
        if ray_norm < 1e-6:
            return float(np.linalg.norm(point - ray_origin))
        projection = np.dot(point - ray_origin, ray_vector) / (ray_norm ** 2)
        projection = max(0.0, min(1.0, float(projection)))
        nearest = ray_origin + projection * ray_vector
        return float(np.linalg.norm(point - nearest))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Slicer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Slicer._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

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
