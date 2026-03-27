"""
Dynamic rules for risk assessment.

These rules evaluate time-dependent aspects of agent behavior (TTC, deceleration, etc.).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .base_rule import BaseRule, RuleResult, RulePriority, RuleRegistry
from ..core.scene_graph import Node


class TTCCriticalRule(BaseRule):
    """
    Rule: Time-to-collision should be above a threshold.

    This rule identifies critical TTC situations where agents are on
    a collision course.
    """

    def __init__(
        self,
        ttc_threshold: float = 3.0,
        min_critical_pairs: int = 1,
        priority: RulePriority = RulePriority.CRITICAL,
        weight: float = 1.0,
    ):
        """
        Initialize the TTC rule.

        Args:
            ttc_threshold: TTC threshold in seconds (below this is critical)
            min_critical_pairs: Minimum number of critical pairs required
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="ttc_critical",
            priority=priority,
            weight=weight,
        )
        self.ttc_threshold = ttc_threshold
        self.min_critical_pairs = min_critical_pairs

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if there are critical TTC situations.

        Args:
            nodes: List of nodes to evaluate (should be at same timestep)
            center_point: Unused
            context: Unused

        Returns:
            RuleResult with pass/fail and score
        """
        # Group nodes by agent
        agent_nodes: Dict[str, Node] = {n.agent_id: n for n in nodes}

        critical_pairs: List[Tuple[str, str, float]] = []

        # Check all pairs
        agent_ids = list(agent_nodes.keys())
        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                node1 = agent_nodes[aid1]
                node2 = agent_nodes[aid2]

                ttc = self._compute_pair_ttc(node1, node2)

                if ttc is not None and ttc < self.ttc_threshold:
                    critical_pairs.append((aid1, aid2, ttc))

        passed = len(critical_pairs) >= self.min_critical_pairs

        # Score: lower TTC = higher score (more critical)
        if critical_pairs:
            min_ttc = min(ttc for _, _, ttc in critical_pairs)
            score = 1.0 - (min_ttc / self.ttc_threshold)
        else:
            score = 0.0

        involved_agents: Set[str] = set()
        for aid1, aid2, _ in critical_pairs:
            involved_agents.update({aid1, aid2})

        message = (
            f"Found {len(critical_pairs)} critical TTC pairs (threshold={self.ttc_threshold}s), "
            f"required {self.min_critical_pairs}. Min TTC: {min(critical_pairs, key=lambda x: x[2])[2] if critical_pairs else 'N/A'}s"
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "ttc_threshold": self.ttc_threshold,
                "critical_pairs": critical_pairs,
            },
            involved_agents=involved_agents,
        )

    def _compute_pair_ttc(self, node1: Node, node2: Node) -> Optional[float]:
        """
        Compute time-to-collision between two agent nodes.

        Returns None if agents are not on a collision course.
        """
        # Position and velocity as arrays
        pos1 = np.array(node1.position)
        pos2 = np.array(node2.position)
        vel1 = np.array(node1.velocity)
        vel2 = np.array(node2.velocity)

        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1

        distance = np.linalg.norm(rel_pos)
        rel_speed = np.linalg.norm(rel_vel)

        # Need minimum relative speed to avoid division by zero
        if rel_speed < 0.1:
            return None

        # Check if approaching (dot product < 0 means approaching)
        if np.dot(rel_pos, rel_vel) >= 0:
            return None  # Moving apart or perpendicular

        # Compute TTC
        ttc = distance / rel_speed
        return ttc


class DecelerationRule(BaseRule):
    """
    Rule: Identify hard braking/deceleration events.

    This rule detects agents with significant deceleration,
    which may indicate risk scenarios.
    """

    def __init__(
        self,
        decel_threshold: float = 3.0,  # m/s^2
        min_braking_agents: int = 1,
        priority: RulePriority = RulePriority.HIGH,
        weight: float = 0.8,
    ):
        """
        Initialize the deceleration rule.

        Args:
            decel_threshold: Deceleration threshold (m/s^2)
            min_braking_agents: Minimum number of braking agents required
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="deceleration",
            priority=priority,
            weight=weight,
        )
        self.decel_threshold = decel_threshold
        self.min_braking_agents = min_braking_agents

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if there are significant deceleration events.

        Args:
            nodes: List of nodes to evaluate
            center_point: Unused
            context: Unused

        Returns:
            RuleResult with pass/fail and score
        """
        braking_agents: List[Tuple[str, float]] = []

        for node in nodes:
            decel = self._compute_deceleration(node)

            if decel is not None and decel > self.decel_threshold:
                braking_agents.append((node.agent_id, decel))

        passed = len(braking_agents) >= self.min_braking_agents

        # Score based on how much the max decel exceeds threshold
        if braking_agents:
            max_decel = max(d for _, d in braking_agents)
            score = min(1.0, (max_decel - self.decel_threshold) / 3.0 + 0.5)
        else:
            score = 0.0

        involved_agents = {aid for aid, _ in braking_agents}

        message = (
            f"Found {len(braking_agents)} braking agents (threshold={self.decel_threshold}m/s²), "
            f"required {self.min_braking_agents}. Max decel: {max(d for _, d in braking_agents) if braking_agents else 'N/A'}m/s²"
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "decel_threshold": self.decel_threshold,
                "braking_agents": braking_agents,
            },
            involved_agents=involved_agents,
        )

    def _compute_deceleration(self, node: Node) -> Optional[float]:
        """
        Compute deceleration magnitude for a node.

        Returns None if agent is not moving (speed < 0.1 m/s).
        """
        vx, vy = node.velocity
        ax, ay = node.acceleration

        speed = np.sqrt(vx**2 + vy**2)

        # Only consider moving agents
        if speed < 0.1:
            return None

        # Deceleration is the component of acceleration opposite to velocity
        decel = -(vx * ax + vy * ay) / speed

        return max(0.0, decel)  # Only positive deceleration (braking)


class VelocityDifferentialRule(BaseRule):
    """
    Rule: Large velocity differentials between agents indicate risk.

    This rule identifies scenarios with large speed differences,
    which can be risky (e.g., fast car vs slow pedestrian).
    """

    def __init__(
        self,
        speed_diff_threshold: float = 10.0,  # m/s
        min_high_diff_pairs: int = 1,
        priority: RulePriority = RulePriority.MEDIUM,
        weight: float = 0.6,
    ):
        """
        Initialize the velocity differential rule.

        Args:
            speed_diff_threshold: Speed difference threshold (m/s)
            min_high_diff_pairs: Minimum number of high-diff pairs required
            priority: Priority level
            weight: Weight for score combination
        """
        super().__init__(
            name="velocity_differential",
            priority=priority,
            weight=weight,
        )
        self.speed_diff_threshold = speed_diff_threshold
        self.min_high_diff_pairs = min_high_diff_pairs

    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate if there are large velocity differentials.

        Args:
            nodes: List of nodes to evaluate
            center_point: Unused
            context: Unused

        Returns:
            RuleResult with pass/fail and score
        """
        # Group nodes by agent
        agent_nodes: Dict[str, Node] = {n.agent_id: n for n in nodes}

        high_diff_pairs: List[Tuple[str, str, float]] = []

        # Check all pairs
        agent_ids = list(agent_nodes.keys())
        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                node1 = agent_nodes[aid1]
                node2 = agent_nodes[aid2]

                speed1 = np.sqrt(node1.velocity[0]**2 + node1.velocity[1]**2)
                speed2 = np.sqrt(node2.velocity[0]**2 + node2.velocity[1]**2)
                speed_diff = abs(speed1 - speed2)

                if speed_diff > self.speed_diff_threshold:
                    high_diff_pairs.append((aid1, aid2, speed_diff))

        passed = len(high_diff_pairs) >= self.min_high_diff_pairs

        # Score based on max speed diff
        if high_diff_pairs:
            max_diff = max(d for _, _, d in high_diff_pairs)
            score = min(1.0, (max_diff - self.speed_diff_threshold) / 10.0 + 0.5)
        else:
            score = 0.0

        involved_agents: Set[str] = set()
        for aid1, aid2, _ in high_diff_pairs:
            involved_agents.update({aid1, aid2})

        message = (
            f"Found {len(high_diff_pairs)} high speed-diff pairs (threshold={self.speed_diff_threshold}m/s), "
            f"required {self.min_high_diff_pairs}. Max diff: {max(d for _, _, d in high_diff_pairs) if high_diff_pairs else 'N/A'}m/s"
        )

        return RuleResult(
            passed=passed,
            score=score,
            message=message,
            metadata={
                "speed_diff_threshold": self.speed_diff_threshold,
                "high_diff_pairs": high_diff_pairs,
            },
            involved_agents=involved_agents,
        )


def register_default_dynamic_rules(
    registry: RuleRegistry,
    ttc_threshold: float = 3.0,
    decel_threshold: float = 3.0,
    speed_diff_threshold: float = 10.0,
) -> None:
    """
    Register default dynamic rules to a registry.

    Args:
        registry: The rule registry to add rules to
        ttc_threshold: TTC threshold for critical rule
        decel_threshold: Deceleration threshold
        speed_diff_threshold: Speed difference threshold
    """
    registry.register(TTCCriticalRule(ttc_threshold=ttc_threshold))
    registry.register(DecelerationRule(decel_threshold=decel_threshold))
    registry.register(VelocityDifferentialRule(speed_diff_threshold=speed_diff_threshold))
