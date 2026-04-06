"""
Spatial graph construction rules.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .base_rule import BaseRule, RuleRegistry
from ..core.scene_graph import Edge, EdgeType, Node, SSTG
from ..core.slicer import Episode


MISSING_DISTANCE_VALUE = -1.0
MISSING_ANGLE_DEG_VALUE = -999.0


class SpatialROIRule(BaseRule):
    """Build a peak-aligned spatial subgraph and project it to all semantic timestamps."""

    def __init__(
        self,
        roi_radius: float = 20.0,
        context_roi_radius: Optional[float] = None,
        enabled: bool = True,
    ):
        super().__init__(name="spatial_roi", enabled=enabled)
        self.roi_radius = float(roi_radius)
        self.context_roi_radius = (
            float(context_roi_radius)
            if context_roi_radius is not None
            else min(self.roi_radius, 12.0)
        )

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        peak_timestamp = "T_peak"
        peak_nodes = {
            node.agent_id: node
            for node in episode.get_nodes_at(peak_timestamp)
            if node.position is not None
        }
        ego_peak_node = peak_nodes.get(episode.ego_agent_id)
        if ego_peak_node is None:
            return current_graph

        risk_anchor_agent_ids = self._get_risk_anchor_agent_ids(current_graph, episode)
        if not risk_anchor_agent_ids:
            primary_risk_agent_ids = set(episode.metadata.get("primary_risk_agent_ids", []))
            primary_risk_agent_ids.update(episode.metadata.get("trigger_agent_ids", []))
            primary_risk_agent_ids.add(episode.metadata.get("trigger_agent_id"))
            primary_risk_agent_ids.discard(None)
            risk_anchor_agent_ids = primary_risk_agent_ids

        included_agent_ids, edge_templates = self._build_peak_templates(
            ego_agent_id=episode.ego_agent_id,
            ego_peak_node=ego_peak_node,
            peak_nodes=peak_nodes,
            risk_anchor_agent_ids=risk_anchor_agent_ids,
        )
        if not included_agent_ids:
            included_agent_ids = {episode.ego_agent_id}

        node_lookup_by_timestamp: Dict[str, Dict[str, Node]] = {}
        for timestamp in episode.ordered_timestamps:
            node_lookup_by_timestamp[timestamp] = {}
            for agent_id in sorted(included_agent_ids):
                reference_node = peak_nodes.get(agent_id)
                if reference_node is None:
                    continue
                node = episode.get_node(agent_id, timestamp)
                if node is None:
                    node = self._make_placeholder_node(reference_node, timestamp)
                current_graph.add_node(node)
                node_lookup_by_timestamp[timestamp][agent_id] = node

        current_graph.metadata["peak_included_agent_ids"] = sorted(included_agent_ids)
        current_graph.metadata["risk_anchor_agent_ids"] = sorted(risk_anchor_agent_ids)
        current_graph.metadata["spatial_edge_templates"] = [
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "roi_radius": roi_radius,
            }
            for source_id, target_id, relation, roi_radius in edge_templates
        ]
        current_graph.metadata["spatial_missing_sentinels"] = {
            "distance": MISSING_DISTANCE_VALUE,
            "relative_angle_deg": MISSING_ANGLE_DEG_VALUE,
        }
        current_graph.metadata["spatial_rule_config"] = {
            "ego_roi_radius": self.roi_radius,
            "context_roi_radius": self.context_roi_radius,
        }

        for timestamp in episode.ordered_timestamps:
            ego_node = node_lookup_by_timestamp[timestamp].get(episode.ego_agent_id)
            if ego_node is None:
                continue
            for source_id, target_id, relation, roi_radius in edge_templates:
                source_node = node_lookup_by_timestamp[timestamp].get(source_id)
                target_node = node_lookup_by_timestamp[timestamp].get(target_id)
                if source_node is None or target_node is None:
                    continue
                current_graph.add_edge(
                    self._build_spatial_edge(
                        source_node=source_node,
                        target_node=target_node,
                        ego_node=ego_node,
                        timestamp=timestamp,
                        relation=relation,
                        roi_radius=roi_radius,
                    )
                )

        for agent_id in sorted(included_agent_ids):
            ordered = [
                timestamp
                for timestamp in episode.ordered_timestamps
                if agent_id in node_lookup_by_timestamp.get(timestamp, {})
            ]
            for source_timestamp, target_timestamp in zip(ordered, ordered[1:]):
                current_graph.add_edge(
                    Edge(
                        source_id=agent_id,
                        target_id=agent_id,
                        source_timestamp=source_timestamp,
                        target_timestamp=target_timestamp,
                        edge_type=EdgeType.TEMPORAL,
                        weight=1.0,
                        relation="state_transition",
                    )
                )

        return current_graph

    def _build_peak_templates(
        self,
        ego_agent_id: str,
        ego_peak_node: Node,
        peak_nodes: Dict[str, Node],
        risk_anchor_agent_ids: Set[str],
    ) -> Tuple[Set[str], List[Tuple[str, str, str, float]]]:
        included_agent_ids: Set[str] = {ego_agent_id}
        edge_templates: List[Tuple[str, str, str, float]] = []
        edge_template_keys: Set[Tuple[str, str, str]] = set()

        context_anchor_ids: List[str] = []
        for agent_id in sorted(risk_anchor_agent_ids):
            if agent_id == ego_agent_id:
                continue
            risk_node = peak_nodes.get(agent_id)
            if risk_node is None:
                continue
            included_agent_ids.add(agent_id)
            context_anchor_ids.append(agent_id)
            edge_key = (ego_agent_id, agent_id, "ego_risk_roi")
            if edge_key not in edge_template_keys:
                edge_template_keys.add(edge_key)
                edge_templates.append((ego_agent_id, agent_id, "ego_risk_roi", self.roi_radius))

        for node in peak_nodes.values():
            if node.agent_id == ego_agent_id:
                continue
            if self._distance(ego_peak_node, node) > self.roi_radius:
                continue
            included_agent_ids.add(node.agent_id)
            relation = "ego_risk_roi" if node.agent_id in risk_anchor_agent_ids else "ego_neighbor_roi"
            edge_key = (ego_agent_id, node.agent_id, relation)
            if edge_key not in edge_template_keys:
                edge_template_keys.add(edge_key)
                edge_templates.append((ego_agent_id, node.agent_id, relation, self.roi_radius))

        for anchor_id in context_anchor_ids:
            anchor_node = peak_nodes[anchor_id]
            for node in peak_nodes.values():
                if node.agent_id in {ego_agent_id, anchor_id}:
                    continue
                if node.agent_id in risk_anchor_agent_ids:
                    continue
                if self._distance(anchor_node, node) > self.context_roi_radius:
                    continue
                included_agent_ids.add(node.agent_id)
                edge_key = (anchor_id, node.agent_id, "risk_context_roi")
                if edge_key not in edge_template_keys:
                    edge_template_keys.add(edge_key)
                    edge_templates.append((anchor_id, node.agent_id, "risk_context_roi", self.context_roi_radius))

        return included_agent_ids, edge_templates

    def _build_spatial_edge(
        self,
        source_node: Node,
        target_node: Node,
        ego_node: Node,
        timestamp: str,
        relation: str,
        roi_radius: float,
    ) -> Edge:
        distance, relative_angle_deg, edge_status, is_active = self._measure_spatial_relation(
            source_node=source_node,
            target_node=target_node,
            ego_node=ego_node,
            roi_radius=roi_radius,
        )
        weight = max(0.0, 1.0 - distance / max(roi_radius, 1.0)) if is_active else 0.0
        metadata = {
            "distance": distance,
            "roi_radius": roi_radius,
            "relative_angle_deg": relative_angle_deg,
            "edge_status": edge_status,
            "is_active": is_active,
            "is_missing": edge_status == "out_of_scene",
        }
        if edge_status == "out_of_scene":
            metadata["missing_agent_ids"] = [
                node.agent_id
                for node in (source_node, target_node)
                if node.position is None
            ]

        return Edge(
            source_id=source_node.agent_id,
            target_id=target_node.agent_id,
            source_timestamp=timestamp,
            target_timestamp=timestamp,
            edge_type=EdgeType.SPATIAL,
            weight=weight,
            relation=relation,
            metadata=metadata,
        )

    def _measure_spatial_relation(
        self,
        source_node: Node,
        target_node: Node,
        ego_node: Node,
        roi_radius: float,
    ) -> Tuple[float, float, str, bool]:
        if source_node.position is None or target_node.position is None or ego_node.position is None:
            return MISSING_DISTANCE_VALUE, MISSING_ANGLE_DEG_VALUE, "out_of_scene", False

        distance = self._distance(source_node, target_node)
        relative_angle_deg = self._relative_angle_deg(ego_node, target_node)
        if distance <= roi_radius:
            return distance, relative_angle_deg, "active", True
        return distance, relative_angle_deg, "outside_roi", False

    @staticmethod
    def _get_risk_anchor_agent_ids(current_graph: SSTG, episode: Episode) -> Set[str]:
        risk_anchor_agent_ids: Set[str] = set()
        for edge in current_graph.get_edges_at_timestamp("T_peak"):
            if edge.edge_type != EdgeType.CAUSAL:
                continue
            risk_anchor_agent_ids.add(edge.source_id)
            risk_anchor_agent_ids.add(edge.target_id)

        risk_anchor_agent_ids.discard(None)
        if episode.ego_agent_id in risk_anchor_agent_ids:
            risk_anchor_agent_ids.remove(episode.ego_agent_id)
        return risk_anchor_agent_ids

    @staticmethod
    def _distance(node_a: Node, node_b: Node) -> float:
        return float(np.linalg.norm(np.asarray(node_a.position) - np.asarray(node_b.position)))

    @staticmethod
    def _relative_angle_deg(ego_node: Node, target_node: Node) -> float:
        ego_position = np.asarray(ego_node.position, dtype=float)
        target_position = np.asarray(target_node.position, dtype=float)
        delta = target_position - ego_position
        global_angle = math.atan2(float(delta[1]), float(delta[0]))
        ego_heading = float(ego_node.heading) if ego_node.heading is not None else 0.0
        relative_angle = math.atan2(
            math.sin(global_angle - ego_heading),
            math.cos(global_angle - ego_heading),
        )
        return math.degrees(relative_angle)

    @staticmethod
    def _make_placeholder_node(reference_node: Node, timestamp: str) -> Node:
        metadata = dict(reference_node.metadata)
        metadata.update(
            {
                "is_placeholder": True,
                "state_status": "out_of_scene",
                "reference_timestamp": reference_node.timestamp,
                "raw_timestep": None,
            }
        )
        return Node(
            agent_id=reference_node.agent_id,
            timestamp=timestamp,
            type=reference_node.type,
            velocity=(0.0, 0.0),
            acceleration=(0.0, 0.0),
            position=None,
            heading=reference_node.heading,
            extent=reference_node.extent,
            metadata=metadata,
        )


def register_default_spatial_rules(
    registry: RuleRegistry,
    roi_radius: float = 20.0,
    context_roi_radius: Optional[float] = None,
) -> None:
    registry.register(SpatialROIRule(roi_radius=roi_radius, context_roi_radius=context_roi_radius))
