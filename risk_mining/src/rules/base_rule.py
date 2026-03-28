"""
Base interfaces for the rule engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..core.scene_graph import SSTG
from ..core.slicer import Episode


class BaseRule(ABC):
    """Strategy interface for pluggable graph construction rules."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def apply(self, episode: Episode, current_graph: SSTG) -> SSTG:
        """Read an episode and return the updated graph."""


class RuleRegistry:
    """Ordered rule container used by the main pipeline."""

    def __init__(self) -> None:
        self._rules: Dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        if rule.name in self._rules:
            raise ValueError(f"Rule '{rule.name}' is already registered.")
        self._rules[rule.name] = rule

    def unregister(self, name: str) -> None:
        self._rules.pop(name, None)

    def get_rule(self, name: str) -> Optional[BaseRule]:
        return self._rules.get(name)

    def get_enabled_rules(self) -> List[BaseRule]:
        return [rule for rule in self._rules.values() if rule.enabled]

    def apply_all(self, episode: Episode, current_graph: Optional[SSTG] = None) -> SSTG:
        graph = current_graph or SSTG(
            scene_id=episode.scene_id,
            dt=episode.dt,
            metadata={"scene_name": episode.scene_name, "env_name": episode.env_name},
        )

        for rule in self.get_enabled_rules():
            graph = rule.apply(episode, graph)
            episode.add_rule_trace(rule.name)

        return graph
