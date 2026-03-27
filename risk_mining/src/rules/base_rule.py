"""
Base rule interface and registry for the risk mining pipeline.

Implements the Strategy Pattern for pluggable risk assessment rules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from ..core.scene_graph import Node


class RulePriority(Enum):
    """Priority levels for rule evaluation."""
    CRITICAL = 0    # Must pass
    HIGH = 1        # Important but not blocking
    MEDIUM = 2      # Optional
    LOW = 3         # Nice to have


@dataclass
class RuleResult:
    """
    Result of evaluating a rule.

    Attributes:
        passed: Whether the rule passed
        score: Confidence score (0-1)
        message: Human-readable description
        metadata: Additional data from the rule
        involved_agents: Set of agent IDs involved in this rule result
    """
    passed: bool
    score: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    involved_agents: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate rule result."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


class BaseRule(ABC):
    """
    Abstract base class for risk assessment rules.

    All rules must implement the evaluate method.
    """

    def __init__(
        self,
        name: str,
        priority: RulePriority = RulePriority.MEDIUM,
        enabled: bool = True,
        weight: float = 1.0,
    ):
        """
        Initialize a rule.

        Args:
            name: Unique identifier for this rule
            priority: Priority level for this rule
            enabled: Whether this rule is active
            weight: Weight for combining multiple rule scores
        """
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.weight = weight

    @abstractmethod
    def evaluate(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """
        Evaluate this rule on the given nodes.

        Args:
            nodes: List of nodes to evaluate
            center_point: Optional center point for spatial rules
            context: Additional context (scene data, etc.)

        Returns:
            RuleResult with pass/fail and score
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class RuleRegistry:
    """
    Registry for managing and applying multiple rules.

    Rules are evaluated in priority order and results are combined.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._rules: Dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """
        Register a rule.

        Args:
            rule: The rule to register

        Raises:
            ValueError: If a rule with the same name already exists
        """
        if rule.name in self._rules:
            raise ValueError(f"Rule '{rule.name}' already registered")
        self._rules[rule.name] = rule

    def unregister(self, name: str) -> None:
        """
        Unregister a rule.

        Args:
            name: Name of the rule to unregister
        """
        self._rules.pop(name, None)

    def get_rule(self, name: str) -> Optional[BaseRule]:
        """Get a rule by name."""
        return self._rules.get(name)

    def get_enabled_rules(self) -> List[BaseRule]:
        """Get all enabled rules sorted by priority."""
        return sorted(
            [r for r in self._rules.values() if r.enabled],
            key=lambda r: r.priority.value
        )

    def evaluate_all(
        self,
        nodes: List[Node],
        center_point: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        require_critical_pass: bool = True,
    ) -> Tuple[bool, float, Dict[str, RuleResult]]:
        """
        Evaluate all enabled rules on the given nodes.

        Args:
            nodes: List of nodes to evaluate
            center_point: Optional center point for spatial rules
            context: Additional context
            require_critical_pass: If True, all CRITICAL rules must pass

        Returns:
            (overall_passed, combined_score, results_by_rule_name)
        """
        results = {}
        enabled_rules = self.get_enabled_rules()

        if not enabled_rules:
            return True, 1.0, {}

        total_weight = 0.0
        weighted_score = 0.0
        all_critical_passed = True

        for rule in enabled_rules:
            result = rule.evaluate(nodes, center_point, context)
            results[rule.name] = result

            # Check critical rules
            if rule.priority == RulePriority.CRITICAL and not result.passed:
                all_critical_passed = False

            # Accumulate weighted score
            if result.passed:
                weighted_score += result.score * rule.weight
            total_weight += rule.weight

        combined_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Overall pass depends on critical rules
        overall_passed = all_critical_passed if require_critical_pass else True

        return overall_passed, combined_score, results

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        rule = self.get_rule(name)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        rule = self.get_rule(name)
        if rule:
            rule.enabled = False
            return True
        return False

    def __repr__(self) -> str:
        enabled = sum(1 for r in self._rules.values() if r.enabled)
        return f"RuleRegistry({len(self._rules)} rules, {enabled} enabled)"
