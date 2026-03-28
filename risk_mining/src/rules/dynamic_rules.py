"""
Dynamic graph construction rules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_rule import BaseRule, RuleRegistry
from ..core.scene_graph import Edge, EdgeType, Node, SSTG
from ..core.slicer import Episode


class TTCCriticalRule(BaseRule):
    """Add causal edges at T_peak when ego-to-agent TTC is critical."""

    def __init__(self, ttc_threshold: float = 2.5, enabled: bool = True):
        super().__init__(name="ttc_critical", enabled=enabled)
        self.ttc_threshold = float(ttc_threshold)

    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        peak_timestamp = "T_peak"
        ego_node = episode.get_node(episode.ego_agent_id, peak_timestamp)
        if ego_node is None or ego_node.position is None:
            return current_graph

        if not current_graph.has_node(ego_node.agent_id, peak_timestamp):
            current_graph.add_node(ego_node)

        for node in episode.get_nodes_at(peak_timestamp):
            if node.agent_id == episode.ego_agent_id or node.position is None:
                continue

            if not current_graph.has_node(node.agent_id, peak_timestamp):
                current_graph.add_node(node)

            ttc = self._compute_ttc(ego_node, node)
            if ttc is None or ttc >= self.ttc_threshold:
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
                    metadata={"ttc": ttc, "threshold": self.ttc_threshold},
                )
            )

        return current_graph

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


def register_default_dynamic_rules(registry: RuleRegistry, ttc_threshold: float = 2.5) -> None:
    registry.register(TTCCriticalRule(ttc_threshold=ttc_threshold))
