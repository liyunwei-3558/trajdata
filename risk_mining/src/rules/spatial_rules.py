"""
Spatial rules for risk assessment.

These rules evaluate spatial relationships between agents.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .base_rule import BaseRule, RuleResult, RulePriority, RuleRegistry
from ..core.scene_graph import Node


class SpatialROIRule(BaseRule):
    """
    Rule: Agents must be within a specified radius of a center point.

    This is typically used as a filter - if no agents are in the ROI,
    the rule fails (or passes with low score depending on use case).
    """

    def __init__(
        self,
        roi_radius: float = 50.0,
        min_agents: int = 1,
        priority: RulePriority = RulePriority.HIGH,
        weight: float = 1.0,
    ):
        """
        Initialize the spatial ROI rule.

        Args:
            roi_radius: Radius of the region of interest in meters
            min_agents: Minimum number of agents required in ROI
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="spatial_roi",
            priority=priority,
            weight=weight,
        )
        self.roi_radius = roi_radius
        self.min_agents = min_agents

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if agents are within the ROI.

        Args:
            nodes: List of nodes to evaluate
            center_point: (x, y) center of the ROI
            context: Additional context (unused)

        Returns:
            RuleResult with pass/fail and score
        """
        if center_point is None:
            # No center point specified - can't evaluate
            return RuleResult(
                passed=False,
                score=0.0,
                message="No center point specified for ROI rule",
                involved_agents=set(),
            )

        cx, cy = center_point
        agents_in_roi: Set[str] = set()
        distances: Dict[str, float] = {}

        for node in nodes:
            distance = np.sqrt(
                (node.position[0] - cx)**2 + (node.position[1] - cy)**2
            )
            if distance <= self.roi_radius:
                agents_in_roi.add(node.agent_id)
                distances[node.agent_id] = distance

        # Check if we have enough agents in ROI
        passed = len(agents_in_roi) >= self.min_agents

        # Score based on how many agents are in ROI vs minimum
        score = min(1.0, len(agents_in_roi) / max(1, self.min_agents))

        message = (
            f"Found {len(agents_in_roi)} agents in ROI (radius={self.roi_radius}m), "
            f"required {self.min_agents}. Agents: {list(agents_in_roi)}"
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "roi_radius": self.roi_radius,
                "center_point": center_point,
                "agents_in_roi": list(agents_in_roi),
                "distances": distances,
            },
            involved_agents=agents_in_roi,
        )


class ConflictLaneRule(BaseRule):
    """
    Rule: Agents should be on conflicting or intersecting lanes.

    This is a dummy implementation - real version would use map data.
    Currently detects close proximity as a proxy for lane conflicts.
    """

    def __init__(
        self,
        conflict_distance: float = 10.0,
        min_conflicts: int = 1,
        priority: RulePriority = RulePriority.MEDIUM,
        weight: float = 0.8,
    ):
        """
        Initialize the conflict lane rule.

        Args:
            conflict_distance: Distance threshold for considering a conflict (m)
            min_conflicts: Minimum number of conflicting pairs
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="conflict_lane",
            priority=priority,
            weight=weight,
        )
        self.conflict_distance = conflict_distance
        self.min_conflicts = min_conflicts

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if agents are in conflicting lane situations.

        Args:
            nodes: List of nodes to evaluate (should be at same timestep)
            center_point: Unused (for interface consistency)
            context: Additional context (unused)

        Returns:
            RuleResult with pass/fail and score
        """
        # Group nodes by agent
        agent_nodes: Dict[str, Node] = {n.agent_id: n for n in nodes}

        conflicts: List[Tuple[str, str, float]] = []

        # Check all pairs
        agent_ids = list(agent_nodes.keys())
        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                node1 = agent_nodes[aid1]
                node2 = agent_nodes[aid2]

                distance = np.sqrt(
                    (node1.position[0] - node2.position[0])**2 +
                    (node1.position[1] - node2.position[1])**2
                )

                if distance <= self.conflict_distance:
                    conflicts.append((aid1, aid2, distance))

        passed = len(conflicts) >= self.min_conflicts
        score = min(1.0, len(conflicts) / max(1, self.min_conflicts))

        involved_agents = set()
        for aid1, aid2, _ in conflicts:
            involved_agents.update({aid1, aid2})

        message = (
            f"Found {len(conflicts)} conflicting pairs (threshold={self.conflict_distance}m), "
            f"required {self.min_conflicts}."
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "conflict_distance": self.conflict_distance,
                "conflicts": conflicts,
            },
            involved_agents=involved_agents,
        )


class AgentTypePresenceRule(BaseRule):
    """
    Rule: Specific agent types must be present in the scenario.

    Useful for filtering scenarios with certain types of road users.
    """

    def __init__(
        self,
        required_types: Set[str],
        min_count: int = 1,
        priority: RulePriority = RulePriority.MEDIUM,
        weight: float = 0.5,
    ):
        """
        Initialize the agent type presence rule.

        Args:
            required_types: Set of agent types that must be present
            min_count: Minimum count for each required type
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="agent_type_presence",
            priority=priority,
            weight=weight,
        )
        self.required_types = required_types
        self.min_count = min_count

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if required agent types are present.

        Args:
            nodes: List of nodes to evaluate
            center_point: Unused
            context: Unused

        Returns:
            RuleResult with pass/fail and score
        """
        # Count agent types
        type_counts: Dict[str, int] = {}
        agent_ids_by_type: Dict[str, Set[str]] = {}

        for node in nodes:
            agent_type = node.agent_type
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
            if agent_type not in agent_ids_by_type:
                agent_ids_by_type[agent_type] = set()
            agent_ids_by_type[agent_type].add(node.agent_id)

        # Check if all required types are present
        missing_types = self.required_types - set(type_counts.keys())
        passed = len(missing_types) == 0

        # Score based on how many required types meet the minimum count
        types_with_count = sum(
            1 for t in self.required_types
            if type_counts.get(t, 0) >= self.min_count
        )
        score = types_with_count / len(self.required_types) if self.required_types else 1.0

        involved_agents: Set[str] = set()
        for t in self.required_types:
            involved_agents.update(agent_ids_by_type.get(t, set()))

        message = (
            f"Required types: {self.required_types}, found: {set(type_counts.keys())}. "
            f"Missing: {missing_types if missing_types else 'none'}."
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "required_types": list(self.required_types),
                "type_counts": type_counts,
                "missing_types": list(missing_types),
            },
            involved_agents=involved_agents,
        )


def register_default_spatial_rules(
    registry: RuleRegistry,
    roi_radius: float = 50.0,
    conflict_distance: float = 10.0,
) -> None:
    """
    Register default spatial rules to a registry.

    Args:
        registry: The rule registry to add rules to
        roi_radius: ROI radius for spatial filtering
        conflict_distance: Distance for conflict detection
    """
    registry.register(SpatialROIRule(roi_radius=roi_radius))
    registry.register(ConflictLaneRule(conflict_distance=conflict_distance))
