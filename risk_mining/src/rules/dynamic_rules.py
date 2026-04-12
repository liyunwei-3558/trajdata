"""
Dynamic graph construction rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .base_rule import BaseRule, RuleRegistry
from ..core.scene_graph import Edge, EdgeType, Node, SSTG
from ..core.slicer import Episode

if TYPE_CHECKING:
    from trajdata import MapAPI
    from trajdata.maps.vec_map import VectorMap
    from trajdata.maps.vec_map_elements import RoadLane


@dataclass(frozen=True)
class LaneAssignment:
    lane_id: str
    confidence: float
    distance_m: float
    heading_error_deg: float
    source: str


@dataclass(frozen=True)
class PairFilterDecision:
    allow_causal: bool
    reason: str
    metadata: Dict[str, Any]
    send_to_manual_review: bool = False


class TTCMapPlausibilityFilter:
    """Map-aware plausibility filter for TTC-based causal edges."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        map_cache_path: Optional[Path] = None,
        map_api: Optional[Any] = None,
        lane_constrained_types: Optional[Sequence[str]] = None,
        current_lane_max_dist: float = 2.5,
        lane_query_radius_m: float = 8.0,
        lane_heading_threshold_deg: float = 35.0,
        low_confidence_threshold: float = 0.4,
        intersection_proximity_radius_m: float = 18.0,
        lane_reachable_hops: int = 3,
        lane_conflict_distance_m: float = 3.5,
        lane_conflict_heading_diff_deg: float = 30.0,
        manual_review_on_low_confidence: bool = True,
        track_birth_manual_review_window_sec: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.map_cache_path = Path(map_cache_path).expanduser() if map_cache_path is not None else None
        self._map_api = map_api
        self.lane_constrained_types = {
            item.upper()
            for item in (
                lane_constrained_types
                if lane_constrained_types is not None
                else ("VEHICLE", "MOTORCYCLE")
            )
        }
        self.current_lane_max_dist = float(current_lane_max_dist)
        self.lane_query_radius_m = float(lane_query_radius_m)
        self.lane_heading_threshold_deg = float(lane_heading_threshold_deg)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.intersection_proximity_radius_m = float(intersection_proximity_radius_m)
        self.lane_reachable_hops = int(lane_reachable_hops)
        self.lane_conflict_distance_m = float(lane_conflict_distance_m)
        self.lane_conflict_heading_diff_deg = float(lane_conflict_heading_diff_deg)
        self.manual_review_on_low_confidence = bool(manual_review_on_low_confidence)
        self.track_birth_manual_review_window_sec = float(track_birth_manual_review_window_sec)

    def evaluate_pair(
        self,
        episode: Episode,
        ego_node: Node,
        target_node: Node,
    ) -> PairFilterDecision:
        if not self.enabled:
            return PairFilterDecision(True, "filter_disabled", {"filter_enabled": False})

        if ego_node.position is None or target_node.position is None:
            return PairFilterDecision(False, "missing_positions", {"filter_enabled": True})

        if not self._is_lane_constrained(ego_node.type, target_node.type):
            return PairFilterDecision(
                True,
                "non_lane_constrained_pair",
                {
                    "filter_enabled": True,
                    "ego_type": ego_node.type,
                    "target_type": target_node.type,
                },
            )

        vec_map = self._load_map(episode)
        if vec_map is None:
            return PairFilterDecision(
                True,
                "map_unavailable",
                {"filter_enabled": True, "map_available": False},
            )

        ego_assignment = self._assign_lane(vec_map, ego_node)
        target_assignment = self._assign_lane(vec_map, target_node)
        metadata: Dict[str, Any] = {
            "filter_enabled": True,
            "map_available": True,
            "ego_lane_assignment": self._lane_assignment_metadata(ego_assignment),
            "target_lane_assignment": self._lane_assignment_metadata(target_assignment),
        }

        if self._should_manual_review_track_birth(episode, target_node):
            metadata["manual_review_reason"] = "track_birth_near_conflict_zone"
            return PairFilterDecision(
                False,
                "track_birth_near_conflict_zone",
                metadata,
                send_to_manual_review=True,
            )

        if ego_assignment is None or target_assignment is None:
            metadata["manual_review_reason"] = "low_lane_confidence"
            return PairFilterDecision(
                allow_causal=not self.manual_review_on_low_confidence,
                reason="low_lane_confidence",
                metadata=metadata,
                send_to_manual_review=self.manual_review_on_low_confidence,
            )

        if min(ego_assignment.confidence, target_assignment.confidence) < self.low_confidence_threshold:
            metadata["manual_review_reason"] = "low_lane_confidence"
            return PairFilterDecision(
                allow_causal=not self.manual_review_on_low_confidence,
                reason="low_lane_confidence",
                metadata=metadata,
                send_to_manual_review=self.manual_review_on_low_confidence,
            )

        lane_relation = self._classify_lane_relation(vec_map, ego_assignment.lane_id, target_assignment.lane_id)
        metadata["ego_target_lane_relation"] = lane_relation
        if lane_relation in {"same_lane", "adjacent_lane", "reachable_lane"}:
            return PairFilterDecision(True, lane_relation, metadata)

        conflict_eval = self._evaluate_lane_conflict_zone(
            vec_map=vec_map,
            ego_node=ego_node,
            target_node=target_node,
            ego_lane_id=ego_assignment.lane_id,
            target_lane_id=target_assignment.lane_id,
            episode=episode,
        )
        metadata.update(conflict_eval["metadata"])

        if conflict_eval["allow"]:
            return PairFilterDecision(True, str(conflict_eval["reason"]), metadata)
        return PairFilterDecision(False, str(conflict_eval["reason"]), metadata)

    def _load_map(self, episode: Episode) -> Optional[Any]:
        map_id = self._infer_map_id(episode)
        if map_id is None:
            return None
        map_api = self._resolve_map_api()
        if map_api is None:
            return None
        try:
            return map_api.get_map(
                map_id,
                incl_road_lanes=True,
                incl_road_areas=True,
                incl_ped_crosswalks=True,
                incl_ped_walkways=True,
            )
        except Exception:
            return None

    def _resolve_map_api(self) -> Optional[Any]:
        if self._map_api is not None:
            return self._map_api
        if self.map_cache_path is None or not self.map_cache_path.exists():
            return None
        from trajdata import MapAPI

        self._map_api = MapAPI(self.map_cache_path, keep_in_memory=True)
        return self._map_api

    def _assign_lane(self, vec_map: Any, node: Node) -> Optional[LaneAssignment]:
        if node.position is None:
            return None

        xyzh = self._node_xyzh(node)
        current_lanes = vec_map.get_current_lane(
            xyzh,
            max_dist=self.current_lane_max_dist,
            max_heading_error=math.radians(self.lane_heading_threshold_deg),
        )
        if current_lanes:
            lane = current_lanes[0]
            return self._build_assignment(lane, node, source="current_lane")

        candidate_lanes = vec_map.get_lanes_within(np.asarray([node.position[0], node.position[1], 0.0]), self.lane_query_radius_m)
        if not candidate_lanes:
            try:
                candidate_lanes = [vec_map.get_closest_lane(np.asarray([node.position[0], node.position[1], 0.0]))]
            except Exception:
                return None

        scored = [self._build_assignment(lane, node, source="fallback_lane") for lane in candidate_lanes]
        scored = [assignment for assignment in scored if assignment is not None]
        if not scored:
            return None
        return sorted(scored, key=lambda item: (-item.confidence, item.distance_m, item.heading_error_deg))[0]

    def _build_assignment(self, lane: Any, node: Node, source: str) -> Optional[LaneAssignment]:
        if node.position is None:
            return None
        projection = lane.center.project_onto(self._node_xyzh(node)[None, :])[0]
        projected_xy = projection[:2]
        lane_heading = float(projection[3])
        heading_error_deg = abs(self._angle_diff_deg(float(node.heading or 0.0), lane_heading))
        distance_m = float(np.linalg.norm(np.asarray(node.position) - projected_xy))
        if source == "current_lane":
            confidence = max(0.75, 1.0 - distance_m / max(self.current_lane_max_dist, 1.0))
        else:
            confidence = max(
                0.05,
                0.7
                - distance_m / max(self.lane_query_radius_m, 1.0) * 0.5
                - heading_error_deg / max(self.lane_heading_threshold_deg * 2.0, 1.0) * 0.3,
            )
        return LaneAssignment(
            lane_id=lane.id,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            distance_m=distance_m,
            heading_error_deg=heading_error_deg,
            source=source,
        )

    def _classify_lane_relation(self, vec_map: Any, ego_lane_id: str, target_lane_id: str) -> str:
        if ego_lane_id == target_lane_id:
            return "same_lane"
        ego_lane = vec_map.get_road_lane(ego_lane_id)
        if target_lane_id in ego_lane.adj_lanes_left or target_lane_id in ego_lane.adj_lanes_right:
            return "adjacent_lane"
        if target_lane_id in self._reachable_lanes(vec_map, ego_lane_id, self.lane_reachable_hops):
            return "reachable_lane"
        return "separated_lane"

    def _evaluate_lane_conflict_zone(
        self,
        *,
        vec_map: Any,
        ego_node: Node,
        target_node: Node,
        ego_lane_id: str,
        target_lane_id: str,
        episode: Episode,
    ) -> Dict[str, Any]:
        conflict_point = self._find_lane_conflict_point(
            vec_map.get_road_lane(ego_lane_id),
            vec_map.get_road_lane(target_lane_id),
        )
        if conflict_point is None:
            return {
                "allow": False,
                "reason": "lane_topology_separated",
                "metadata": {"lane_conflict_point": None},
            }

        ego_distance = self._distance_to_point(ego_node.position, conflict_point)
        target_distance = self._distance_to_point(target_node.position, conflict_point)
        episode_conflict_center = episode.metadata.get("conflict_zone", {}).get("center")
        episode_conflict_distance = (
            self._distance_to_point(tuple(episode_conflict_center), conflict_point)
            if episode_conflict_center is not None
            else None
        )
        metadata = {
            "lane_conflict_point": list(conflict_point),
            "ego_conflict_distance_m": ego_distance,
            "target_conflict_distance_m": target_distance,
            "episode_conflict_center_distance_m": episode_conflict_distance,
        }
        if (
            ego_distance <= self.intersection_proximity_radius_m
            and target_distance <= self.intersection_proximity_radius_m
        ):
            return {"allow": True, "reason": "conflict_zone_proximal", "metadata": metadata}
        return {"allow": False, "reason": "outside_conflict_zone", "metadata": metadata}

    def _find_lane_conflict_point(
        self,
        ego_lane: Any,
        target_lane: Any,
    ) -> Optional[Tuple[float, float]]:
        ego_points = ego_lane.center.interpolate(max_dist=1.0).xyzh
        target_points = target_lane.center.interpolate(max_dist=1.0).xyzh
        deltas = ego_points[:, None, :2] - target_points[None, :, :2]
        distances = np.linalg.norm(deltas, axis=-1)
        min_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        min_distance = float(distances[min_index])
        if min_distance > self.lane_conflict_distance_m:
            return None

        ego_heading = float(ego_points[min_index[0], 3])
        target_heading = float(target_points[min_index[1], 3])
        heading_diff_deg = abs(self._angle_diff_deg(ego_heading, target_heading))
        if heading_diff_deg < self.lane_conflict_heading_diff_deg:
            return None

        midpoint = (
            float((ego_points[min_index[0], 0] + target_points[min_index[1], 0]) / 2.0),
            float((ego_points[min_index[0], 1] + target_points[min_index[1], 1]) / 2.0),
        )
        return midpoint

    def _should_manual_review_track_birth(self, episode: Episode, target_node: Node) -> bool:
        track_windows = episode.metadata.get("trigger_track_windows", {})
        track_window = track_windows.get(target_node.agent_id)
        if not track_window:
            return False

        birth_ts = track_window.get("birth_ts")
        birth_position = track_window.get("birth_position")
        if birth_ts is None or birth_position is None:
            return False

        review_window_ts = max(1, int(round(self.track_birth_manual_review_window_sec / max(episode.dt, 1e-6))))
        if int(episode.t_peak) - int(birth_ts) > review_window_ts:
            return False

        conflict_center = episode.metadata.get("conflict_zone", {}).get("center")
        if conflict_center is None:
            return False
        return self._distance_to_point(tuple(birth_position), tuple(conflict_center)) <= self.intersection_proximity_radius_m

    def _reachable_lanes(self, vec_map: Any, origin_lane_id: str, max_hops: int) -> Set[str]:
        visited: Set[str] = {origin_lane_id}
        frontier: deque[Tuple[str, int]] = deque([(origin_lane_id, 0)])
        while frontier:
            lane_id, hops = frontier.popleft()
            if hops >= max_hops:
                continue
            lane = vec_map.get_road_lane(lane_id)
            for neighbor in self._iter_lane_neighbors(lane):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, hops + 1))
        visited.discard(origin_lane_id)
        return visited

    @staticmethod
    def _iter_lane_neighbors(lane: Any) -> Iterable[str]:
        return (
            set(lane.adj_lanes_left)
            | set(lane.adj_lanes_right)
            | set(lane.next_lanes)
            | set(lane.prev_lanes)
        )

    @staticmethod
    def _node_xyzh(node: Node) -> np.ndarray:
        heading = float(node.heading or 0.0)
        return np.asarray([node.position[0], node.position[1], 0.0, heading], dtype=float)

    @staticmethod
    def _angle_diff_deg(angle_a: float, angle_b: float) -> float:
        return math.degrees(math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b)))

    @staticmethod
    def _distance_to_point(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float)))

    @staticmethod
    def _lane_assignment_metadata(assignment: Optional[LaneAssignment]) -> Optional[Dict[str, Any]]:
        if assignment is None:
            return None
        return {
            "lane_id": assignment.lane_id,
            "confidence": assignment.confidence,
            "distance_m": assignment.distance_m,
            "heading_error_deg": assignment.heading_error_deg,
            "source": assignment.source,
        }

    def _is_lane_constrained(self, ego_type: str, target_type: str) -> bool:
        return ego_type.upper() in self.lane_constrained_types and target_type.upper() in self.lane_constrained_types

    @staticmethod
    def _infer_map_id(episode: Episode) -> Optional[str]:
        if episode.env_name == "sind":
            location = episode.scene_name.split("_", 1)[0]
            return f"{episode.env_name}:{location}"
        return None


class TTCCriticalRule(BaseRule):
    """Add causal edges at T_peak using the selected peak metric semantics."""

    def __init__(
        self,
        ttc_threshold: float = 2.5,
        enabled: bool = True,
        *,
        map_cache_path: Optional[Path] = None,
        map_api: Optional[Any] = None,
        causal_max_distance_m_for_pet_ttc: float = 25.0,
        causal_filter_enabled: bool = True,
        lane_constrained_types: Optional[Sequence[str]] = None,
        current_lane_max_dist: float = 2.5,
        lane_query_radius_m: float = 8.0,
        lane_heading_threshold_deg: float = 35.0,
        low_confidence_threshold: float = 0.4,
        intersection_proximity_radius_m: float = 18.0,
        lane_reachable_hops: int = 3,
        lane_conflict_distance_m: float = 3.5,
        lane_conflict_heading_diff_deg: float = 30.0,
        manual_review_on_low_confidence: bool = True,
        track_birth_manual_review_window_sec: float = 2.0,
    ):
        super().__init__(name="ttc_critical", enabled=enabled)
        self.ttc_threshold = float(ttc_threshold)
        self.causal_max_distance_m_for_pet_ttc = float(causal_max_distance_m_for_pet_ttc)
        self.map_filter = TTCMapPlausibilityFilter(
            enabled=causal_filter_enabled,
            map_cache_path=map_cache_path,
            map_api=map_api,
            lane_constrained_types=lane_constrained_types,
            current_lane_max_dist=current_lane_max_dist,
            lane_query_radius_m=lane_query_radius_m,
            lane_heading_threshold_deg=lane_heading_threshold_deg,
            low_confidence_threshold=low_confidence_threshold,
            intersection_proximity_radius_m=intersection_proximity_radius_m,
            lane_reachable_hops=lane_reachable_hops,
            lane_conflict_distance_m=lane_conflict_distance_m,
            lane_conflict_heading_diff_deg=lane_conflict_heading_diff_deg,
            manual_review_on_low_confidence=manual_review_on_low_confidence,
            track_birth_manual_review_window_sec=track_birth_manual_review_window_sec,
        )

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        peak_timestamp = "T_peak"
        peak_metric_type = str(episode.metadata.get("peak_metric_type", "ttc"))
        trigger_agent_ids = list(episode.metadata.get("trigger_agent_ids", []))
        if not trigger_agent_ids and episode.metadata.get("trigger_agent_id") is not None:
            trigger_agent_ids = [str(episode.metadata["trigger_agent_id"])]

        ego_node = current_graph.get_node(episode.ego_agent_id, peak_timestamp)
        if ego_node is None:
            ego_node = episode.get_node(episode.ego_agent_id, peak_timestamp)
            if ego_node is None or ego_node.position is None:
                return current_graph
            if not current_graph.has_node(ego_node.agent_id, peak_timestamp):
                current_graph.add_node(ego_node)

        if ego_node.position is None:
            return current_graph

        if peak_metric_type == "ttc":
            return self._apply_ttc_rule(episode, current_graph, ego_node, peak_timestamp)

        relation = "ego_kinematic_response"
        metric_key = "peak_metric_value"
        if peak_metric_type == "pet":
            relation = "has_post_encroachment_risk"
            metric_key = "min_pet"
        elif peak_metric_type == "tti":
            relation = "has_intersection_arrival_risk"
            metric_key = "min_tti"
        metric_value = episode.metadata.get(metric_key)
        metric_threshold = (
            episode.metadata.get("pet_event_threshold")
            if peak_metric_type == "pet"
            else episode.metadata.get("tti_event_threshold")
            if peak_metric_type == "tti"
            else max(self.ttc_threshold, 1.0)
        )
        weight = 1.0
        if isinstance(metric_value, (float, int)) and isinstance(metric_threshold, (float, int)):
            threshold = max(float(metric_threshold), 1e-6)
            value = float(metric_value)
            if peak_metric_type in {"pet", "tti"}:
                weight = max(0.0, min(1.0, 1.0 - value / threshold))
            else:
                weight = max(0.2, min(1.0, value / threshold))

        for agent_id in trigger_agent_ids:
            node = current_graph.get_node(agent_id, peak_timestamp)
            if node is None:
                node = episode.get_node(agent_id, peak_timestamp)
            if node is None or node.position is None:
                continue
            if not current_graph.has_node(node.agent_id, peak_timestamp):
                current_graph.add_node(node)

            pair_distance = self._pair_distance(ego_node, node)
            if peak_metric_type == "pet" and pair_distance > self.causal_max_distance_m_for_pet_ttc:
                self._record_metric_distance_filter(
                    episode=episode,
                    current_graph=current_graph,
                    peak_metric_type=peak_metric_type,
                    target_id=node.agent_id,
                    pair_distance=pair_distance,
                    threshold=self.causal_max_distance_m_for_pet_ttc,
                )
                continue

            current_graph.add_edge(
                Edge(
                    source_id=episode.ego_agent_id,
                    target_id=node.agent_id,
                    source_timestamp=peak_timestamp,
                    target_timestamp=peak_timestamp,
                    edge_type=EdgeType.CAUSAL,
                    weight=weight,
                    relation=relation,
                    metadata={
                        "peak_metric_type": peak_metric_type,
                        "peak_metric_value": metric_value,
                        "threshold": metric_threshold,
                        "pair_distance_m": pair_distance,
                    },
                )
            )

        return current_graph

    def _apply_ttc_rule(
        self,
        episode: Episode,
        current_graph: SSTG,
        ego_node: Node,
        peak_timestamp: str,
    ) -> SSTG:
        candidate_nodes = [
            node
            for node in current_graph.get_nodes_at_timestamp(peak_timestamp)
            if node.agent_id != episode.ego_agent_id and node.position is not None
        ]
        if not candidate_nodes:
            candidate_nodes = [
                node
                for node in episode.get_nodes_at(peak_timestamp)
                if node.agent_id != episode.ego_agent_id and node.position is not None
            ]

        filtered_pairs: List[Dict[str, Any]] = []
        for node in candidate_nodes:
            if not current_graph.has_node(node.agent_id, peak_timestamp):
                current_graph.add_node(node)

            ttc = self._compute_ttc(ego_node, node)
            if ttc is None or ttc >= self.ttc_threshold:
                continue

            pair_distance = self._pair_distance(ego_node, node)
            if pair_distance > self.causal_max_distance_m_for_pet_ttc:
                filtered_pairs.append(
                    {
                        "target_id": node.agent_id,
                        "ttc": ttc,
                        "allow_causal": False,
                        "reason": "distance_threshold_exceeded",
                        "manual_review": False,
                        "metadata": {
                            "pair_distance_m": pair_distance,
                            "distance_threshold_m": self.causal_max_distance_m_for_pet_ttc,
                        },
                    }
                )
                continue

            decision = self.map_filter.evaluate_pair(episode, ego_node, node)
            filtered_pairs.append(
                {
                    "target_id": node.agent_id,
                    "ttc": ttc,
                    "allow_causal": decision.allow_causal,
                    "reason": decision.reason,
                    "manual_review": decision.send_to_manual_review,
                    "metadata": {
                        **decision.metadata,
                        "pair_distance_m": pair_distance,
                        "distance_threshold_m": self.causal_max_distance_m_for_pet_ttc,
                    },
                }
            )

            if decision.send_to_manual_review:
                self._mark_episode_for_manual_review(episode, current_graph, decision, node.agent_id, ttc)
                continue

            if not decision.allow_causal:
                continue

            current_graph.add_edge(
                Edge(
                    source_id=episode.ego_agent_id,
                    target_id=node.agent_id,
                    source_timestamp=peak_timestamp,
                    target_timestamp=peak_timestamp,
                    edge_type=EdgeType.CAUSAL,
                    weight=max(0.0, 1.0 - ttc / max(self.ttc_threshold, 1e-6)),
                    relation="has_collision_risk",
                    metadata={
                        "ttc": ttc,
                        "threshold": self.ttc_threshold,
                        "pair_distance_m": pair_distance,
                        "causal_filter_reason": decision.reason,
                        "causal_filter_metadata": {
                            **decision.metadata,
                            "pair_distance_m": pair_distance,
                            "distance_threshold_m": self.causal_max_distance_m_for_pet_ttc,
                        },
                    },
                )
            )

        if filtered_pairs:
            episode.metadata["ttc_candidate_filter_results"] = filtered_pairs
            current_graph.metadata["ttc_candidate_filter_results"] = filtered_pairs
        return current_graph

    @staticmethod
    def _mark_episode_for_manual_review(
        episode: Episode,
        current_graph: SSTG,
        decision: PairFilterDecision,
        target_id: str,
        ttc: float,
    ) -> None:
        details = {
            "target_id": target_id,
            "ttc": ttc,
            "reason": decision.reason,
            "filter_metadata": decision.metadata,
        }
        episode.metadata["manual_review_required"] = True
        episode.metadata["manual_review_status"] = "requested_by_ttc_map_filter"
        episode.metadata["manual_review_reason"] = f"TTC candidate '{target_id}' requires manual review: {decision.reason}"
        episode.metadata["manual_review_details"] = details
        current_graph.metadata["manual_review_status"] = "requested_by_ttc_map_filter"
        current_graph.metadata["manual_review_reason"] = episode.metadata["manual_review_reason"]
        current_graph.metadata["manual_review_details"] = details

    @staticmethod
    def _compute_ttc(ego_node: Node, other_node: Node) -> Optional[float]:
        ego_position = np.asarray(ego_node.position, dtype=float)
        other_position = np.asarray(other_node.position, dtype=float)
        ego_velocity = np.asarray(ego_node.velocity, dtype=float)
        other_velocity = np.asarray(other_node.velocity, dtype=float)

        rel_position = other_position - ego_position
        rel_velocity = other_velocity - ego_velocity
        rel_speed = float(np.linalg.norm(rel_velocity))
        if rel_speed < 1e-6:
            return None

        closing_rate = -float(np.dot(rel_position, rel_velocity)) / max(float(np.linalg.norm(rel_position)), 1e-6)
        if closing_rate <= 0:
            return None

        return float(np.linalg.norm(rel_position)) / closing_rate

    @staticmethod
    def _pair_distance(ego_node: Node, other_node: Node) -> float:
        return float(
            np.linalg.norm(
                np.asarray(ego_node.position, dtype=float) - np.asarray(other_node.position, dtype=float)
            )
        )

    @staticmethod
    def _record_metric_distance_filter(
        episode: Episode,
        current_graph: SSTG,
        peak_metric_type: str,
        target_id: str,
        pair_distance: float,
        threshold: float,
    ) -> None:
        record = {
            "target_id": target_id,
            "metric_type": peak_metric_type,
            "reason": "distance_threshold_exceeded",
            "pair_distance_m": pair_distance,
            "distance_threshold_m": threshold,
        }
        episode.metadata.setdefault("causal_metric_filter_results", []).append(record)
        current_graph.metadata.setdefault("causal_metric_filter_results", []).append(record)


def register_default_dynamic_rules(registry: RuleRegistry, ttc_threshold: float = 2.5, **kwargs: Any) -> None:
    registry.register(TTCCriticalRule(ttc_threshold=ttc_threshold, **kwargs))
