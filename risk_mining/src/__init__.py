"""
Risk Mining Pipeline for Autonomous Driving Trajectory Data.

This package extracts "Intersection Traffic Disturbance Scenarios" from
trajectory data and outputs Semantic Spatio-Temporal Graphs (SSTG) to
a dual library system.

Main Components:
- Core: SSTG data structures and episode slicer
- Rules: Strategy pattern for pluggable risk assessment rules
- Library: Dual library system for risk elements and events
- Utils: Sanity checker and logging utilities
"""

__version__ = "0.1.0"

from .core import SSTG, Node, Edge, EdgeType, Slicer, Episode, EpisodeType
from .rules import BaseRule, RuleResult, RuleRegistry, RulePriority
from .rules import (
    SpatialROIRule, ConflictLaneRule, AgentTypePresenceRule,
    TTCCriticalRule, DecelerationRule, VelocityDifferentialRule,
    register_default_spatial_rules, register_default_dynamic_rules,
)
from .library import RiskElementLibrary, RiskEventLibrary, DualLibrary
from .utils import SanityChecker, SanityCheckResult, create_default_checker, setup_logger, get_logger

__all__ = [
    # Core
    "SSTG", "Node", "Edge", "EdgeType",
    "Slicer", "Episode", "EpisodeType",
    # Rules
    "BaseRule", "RuleResult", "RuleRegistry", "RulePriority",
    "SpatialROIRule", "ConflictLaneRule", "AgentTypePresenceRule",
    "TTCCriticalRule", "DecelerationRule", "VelocityDifferentialRule",
    "register_default_spatial_rules", "register_default_dynamic_rules",
    # Library
    "RiskElementLibrary", "RiskEventLibrary", "DualLibrary",
    # Utils
    "SanityChecker", "SanityCheckResult", "create_default_checker",
    "setup_logger", "get_logger",
]
