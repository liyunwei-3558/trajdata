"""
Spatial graph construction rules.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np

from .base_rule import BaseRule, RuleRegistry
from ..core.scene_graph import Edge, EdgeType, Node, SSTG
from ..core.slicer import Episode


class SpatialROIRule(BaseRule):
    """Add nodes and spatial relations for agents near the ego vehicle."""

    def __init__(self, roi_radius: float = 50.0, enabled: bool = True):
        super().__init__(name="spatial_roi", enabled=enabled)
        self.roi_radius = float(roi_radius)

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        included_agents: Dict[str, Set[str]] = {}

        for timestamp in episode.ordered_timestamps:
            ego_node = episode.get_node(episode.ego_agent_id, timestamp)
            if ego_node is None or ego_node.position is None:
                continue

            current_graph.add_node(ego_node)
            included_agents.setdefault(ego_node.agent_id, set()).add(timestamp)

            for node in episode.get_nodes_at(timestamp):
                if node.position is None:
                    continue

                distance = self._distance(ego_node, node)
                if node.agent_id != episode.ego_agent_id and distance > self.roi_radius:
                    continue

                current_graph.add_node(node)
                included_agents.setdefault(node.agent_id, set()).add(timestamp)

                if node.agent_id == episode.ego_agent_id:
                    continue

                current_graph.add_edge(
                    Edge(
                        source_id=episode.ego_agent_id,
                        target_id=node.agent_id,
                        source_timestamp=timestamp,
                        target_timestamp=timestamp,
                        edge_type=EdgeType.SPATIAL,
                        weight=max(0.0, 1.0 - distance / max(self.roi_radius, 1.0)),
                        relation="within_roi",
                        metadata={"distance": distance, "roi_radius": self.roi_radius},
                    )
                )

        for agent_id, timestamps in included_agents.items():
            ordered = [label for label in episode.ordered_timestamps if label in timestamps]
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

    @staticmethod
    def _distance(node_a: Node, node_b: Node) -> float:
        return float(np.linalg.norm(np.asarray(node_a.position) - np.asarray(node_b.position)))


def register_default_spatial_rules(registry: RuleRegistry, roi_radius: float = 50.0) -> None:
    registry.register(SpatialROIRule(roi_radius=roi_radius))
