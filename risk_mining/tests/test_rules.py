"""
Unit tests for rule engine.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core import Node
from src.rules import (
    BaseRule, RuleResult, RuleRegistry, RulePriority,
    SpatialROIRule, ConflictLaneRule, TTCCriticalRule,
    DecelerationRule, AgentTypePresenceRule,
)


def test_rule_result():
    """Test RuleResult creation."""
    result = RuleResult(
        passed=True,
        score=0.8,
        message="Test passed",
        metadata={"key": "value"},
        involved_agents={"agent_1", "agent_2"},
    )

    assert result.passed is True
    assert result.score == 0.8
    assert result.message == "Test passed"
    assert result.metadata["key"] == "value"
    assert result.involved_agents == {"agent_1", "agent_2"}

    print("✓ test_rule_result passed")


def test_rule_registry():
    """Test RuleRegistry basic operations."""
    registry = RuleRegistry()

    # Create a dummy rule
    class DummyRule(BaseRule):
        def evaluate(self, nodes, center_point=None, context=None):
            return RuleResult(
                passed=True,
                score=1.0,
                message="Dummy",
                involved_agents=set(),
            )

    rule = DummyRule(name="dummy_rule", priority=RulePriority.MEDIUM)

    # Register
    registry.register(rule)

    # Get rule
    retrieved = registry.get_rule("dummy_rule")
    assert retrieved is rule

    # Get enabled rules
    enabled = registry.get_enabled_rules()
    assert len(enabled) == 1
    assert enabled[0] is rule

    # Unregister
    registry.unregister("dummy_rule")
    assert registry.get_rule("dummy_rule") is None

    print("✓ test_rule_registry passed")


def test_spatial_roi_rule():
    """Test SpatialROIRule."""
    rule = SpatialROIRule(roi_radius=50.0, min_agents=2)

    # Create nodes
    nodes = [
        Node(
            agent_id=f"agent_{i}",
            agent_type="VEHICLE",
            timestep=0,
            position=(float(i * 10), 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        )
        for i in range(5)
    ]

    # Test with center point
    center_point = (20.0, 0.0)
    result = rule.evaluate(nodes, center_point)

    # Agents 1, 2, 3 should be in ROI (10, 20, 30 meters from center)
    assert result.passed is True
    assert len(result.involved_agents) >= 2

    # Test with no center point
    result_no_center = rule.evaluate(nodes, None)
    assert result_no_center.passed is False

    print("✓ test_spatial_roi_rule passed")


def test_conflict_lane_rule():
    """Test ConflictLaneRule."""
    rule = ConflictLaneRule(conflict_distance=15.0, min_conflicts=1)

    # Create nodes with some close pairs
    nodes = [
        Node(
            agent_id="agent_0",
            agent_type="VEHICLE",
            timestep=0,
            position=(0.0, 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        ),
        Node(
            agent_id="agent_1",
            agent_type="VEHICLE",
            timestep=0,
            position=(10.0, 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        ),
        Node(
            agent_id="agent_2",
            agent_type="VEHICLE",
            timestep=0,
            position=(50.0, 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        ),
    ]

    result = rule.evaluate(nodes)

    # agent_0 and agent_1 are within conflict distance
    assert result.passed is True
    assert "agent_0" in result.involved_agents
    assert "agent_1" in result.involved_agents

    print("✓ test_conflict_lane_rule passed")


def test_ttc_critical_rule():
    """Test TTCCriticalRule."""
    rule = TTCCriticalRule(ttc_threshold=3.0, min_critical_pairs=1)

    # Create nodes on collision course
    nodes = [
        Node(
            agent_id="agent_1",
            agent_type="VEHICLE",
            timestep=0,
            position=(0.0, 0.0),
            velocity=(10.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        ),
        Node(
            agent_id="agent_2",
            agent_type="VEHICLE",
            timestep=0,
            position=(20.0, 0.0),
            velocity=(-10.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=3.14,
            extent=(4.5, 2.0),
        ),
    ]

    result = rule.evaluate(nodes)

    # Approaching head-on, TTC = 20/20 = 1s < 3s threshold
    assert result.passed is True
    assert "agent_1" in result.involved_agents
    assert "agent_2" in result.involved_agents

    print("✓ test_ttc_critical_rule passed")


def test_deceleration_rule():
    """Test DecelerationRule."""
    rule = DecelerationRule(decel_threshold=3.0, min_braking_agents=1)

    # Create nodes with braking
    nodes = [
        Node(
            agent_id="agent_1",
            agent_type="VEHICLE",
            timestep=0,
            position=(0.0, 0.0),
            velocity=(10.0, 0.0),
            acceleration=(-5.0, 0.0),  # Braking!
            heading=0.0,
            extent=(4.5, 2.0),
        ),
        Node(
            agent_id="agent_2",
            agent_type="VEHICLE",
            timestep=0,
            position=(10.0, 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        ),
    ]

    result = rule.evaluate(nodes)

    # agent_1 is braking at 5 m/s^2 > 3 m/s^2 threshold
    assert result.passed is True
    assert "agent_1" in result.involved_agents

    print("✓ test_deceleration_rule passed")


def test_agent_type_presence_rule():
    """Test AgentTypePresenceRule."""
    rule = AgentTypePresenceRule(
        required_types={"VEHICLE", "PEDESTRIAN"},
        min_count=1,
    )

    # Create mixed nodes
    nodes = [
        Node(
            agent_id=f"agent_{i}",
            agent_type="VEHICLE" if i < 2 else "PEDESTRIAN",
            timestep=0,
            position=(float(i * 10), 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0) if i < 2 else (0.5, 0.5),
        )
        for i in range(3)
    ]

    result = rule.evaluate(nodes)

    # Both types present
    assert result.passed is True
    assert len(result.involved_agents) == 3

    print("✓ test_agent_type_presence_rule passed")


def test_registry_evaluate_all():
    """Test RuleRegistry evaluate_all."""
    registry = RuleRegistry()

    # Add rules
    registry.register(SpatialROIRule(roi_radius=50.0, min_agents=1))
    registry.register(ConflictLaneRule(conflict_distance=15.0, min_conflicts=1))

    # Create nodes
    nodes = [
        Node(
            agent_id=f"agent_{i}",
            agent_type="VEHICLE",
            timestep=0,
            position=(float(i * 10), 0.0),
            velocity=(5.0, 0.0),
            acceleration=(0.0, 0.0),
            heading=0.0,
            extent=(4.5, 2.0),
        )
        for i in range(3)
    ]

    # Evaluate all
    passed, score, results = registry.evaluate_all(nodes, center_point=(10.0, 0.0))

    assert passed is True
    assert 0 <= score <= 1
    assert len(results) == 2

    print("✓ test_registry_evaluate_all passed")


if __name__ == "__main__":
    print("Running rule tests...")
    test_rule_result()
    test_rule_registry()
    test_spatial_roi_rule()
    test_conflict_lane_rule()
    test_ttc_critical_rule()
    test_deceleration_rule()
    test_agent_type_presence_rule()
    test_registry_evaluate_all()
    print("\n✅ All rule tests passed!")
