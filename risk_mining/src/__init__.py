"""
Risk Mining Pipeline for Autonomous Driving Trajectory Data.
"""

__version__ = "0.2.0"

from .core import SSTG, Node, Edge, EdgeType, Slicer, Episode, EpisodeType
from .library import RiskElementLibrary, RiskEventLibrary, DualLibrary
from .rules import BaseRule, RuleRegistry, SpatialROIRule, TTCCriticalRule
from .rules import register_default_spatial_rules, register_default_dynamic_rules
from .utils import SanityChecker, SanityCheckResult, create_default_checker, validate_graph, setup_logger, get_logger

__all__ = [
    "SSTG",
    "Node",
    "Edge",
    "EdgeType",
    "Slicer",
    "Episode",
    "EpisodeType",
    "BaseRule",
    "RuleRegistry",
    "SpatialROIRule",
    "TTCCriticalRule",
    "register_default_spatial_rules",
    "register_default_dynamic_rules",
    "RiskElementLibrary",
    "RiskEventLibrary",
    "DualLibrary",
    "SanityChecker",
    "SanityCheckResult",
    "create_default_checker",
    "validate_graph",
    "setup_logger",
    "get_logger",
]
