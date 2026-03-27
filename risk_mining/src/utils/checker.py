"""
Sanity checker for validating extracted episodes.

Episodes that fail sanity checks are saved to the review buffer for manual inspection.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.slicer import Episode, EpisodeType
from ..core.scene_graph import EdgeType


class SanityCheckResult:
    """Result of sanity checking an episode."""

    def __init__(
        self,
        passed: bool,
        reason: str,
        details: Dict[str, Any],
    ):
        self.passed = passed
        self.reason = reason
        self.details = details


class SanityChecker:
    """
    Validates extracted episodes before adding to the risk event library.

    Episodes that fail checks are saved to the review buffer.
    """

    def __init__(
        self,
        min_timesteps: int = 5,
        max_timesteps: int = 100,
        min_agents: int = 2,
        max_agents: int = 10,
        min_risk_score: float = 0.1,
        require_edges: bool = True,
        require_interaction_edges: bool = False,
        review_buffer_dir: Optional[Path] = None,
    ):
        """
        Initialize the sanity checker.

        Args:
            min_timesteps: Minimum episode duration in timesteps
            max_timesteps: Maximum episode duration in timesteps
            min_agents: Minimum number of involved agents
            max_agents: Maximum number of involved agents
            min_risk_score: Minimum risk score threshold
            require_edges: Whether episodes must have edges (not just isolated nodes)
            require_interaction_edges: Whether episodes must have interaction edges
            review_buffer_dir: Directory to save failed episodes
        """
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.min_risk_score = min_risk_score
        self.require_edges = require_edges
        self.require_interaction_edges = require_interaction_edges
        self.review_buffer_dir = review_buffer_dir

    def check_episode(self, episode: Episode) -> SanityCheckResult:
        """
        Check if an episode passes all sanity checks.

        Args:
            episode: The episode to check

        Returns:
            SanityCheckResult with pass/fail status
        """
        checks = []

        # Check 1: Duration bounds
        duration = episode.duration_timesteps
        duration_ok = self.min_timesteps <= duration <= self.max_timesteps
        checks.append(("duration", duration_ok, {
            "duration": duration,
            "min": self.min_timesteps,
            "max": self.max_timesteps,
        }))

        # Check 2: Agent count bounds
        agent_count = len(episode.involved_agents)
        agent_count_ok = self.min_agents <= agent_count <= self.max_agents
        checks.append(("agent_count", agent_count_ok, {
            "count": agent_count,
            "min": self.min_agents,
            "max": self.max_agents,
        }))

        # Check 3: Risk score threshold
        risk_score_ok = episode.risk_score >= self.min_risk_score
        checks.append(("risk_score", risk_score_ok, {
            "score": episode.risk_score,
            "min": self.min_risk_score,
        }))

        # Check 4: Edges (if SSTG is available)
        edge_ok = True
        interaction_edge_ok = True
        edge_details = {}

        if episode.sstg is not None:
            total_edges = 0
            interaction_edges = 0

            for timestep in episode.sstg.timesteps:
                edges = episode.sstg.get_edges_at_timestep(timestep)
                total_edges += len(edges)

                for edge in edges:
                    if edge.edge_type in [
                        EdgeType.INTERACTION,
                        EdgeType.LEAD_FOLLOW,
                        EdgeType.CROSSING_PATH,
                    ]:
                        interaction_edges += 1

            edge_details = {
                "total_edges": total_edges,
                "interaction_edges": interaction_edges,
            }

            if self.require_edges:
                edge_ok = total_edges > 0

            if self.require_interaction_edges:
                interaction_edge_ok = interaction_edges > 0

        checks.append(("edges", edge_ok, edge_details))
        checks.append(("interaction_edges", interaction_edge_ok, edge_details))

        # Overall result
        all_passed = all(passed for _, passed, _ in checks)
        failed_checks = [name for name, passed, _ in checks if not passed]

        if all_passed:
            return SanityCheckResult(
                passed=True,
                reason="All sanity checks passed",
                details={name: details for name, _, details in checks},
            )
        else:
            return SanityCheckResult(
                passed=False,
                reason=f"Failed checks: {', '.join(failed_checks)}",
                details={name: details for name, _, details in checks},
            )

    def save_to_review_buffer(
        self,
        episode: Episode,
        check_result: SanityCheckResult,
    ) -> Path:
        """
        Save a failed episode to the review buffer.

        Args:
            episode: The episode that failed checks
            check_result: The sanity check result

        Returns:
            Path to the saved file
        """
        if self.review_buffer_dir is None:
            raise RuntimeError("review_buffer_dir not configured")

        self.review_buffer_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name = episode.metadata.get('scene_name', 'unknown')
        filename = f"failed_episode_{scene_name}_{episode.t_peak}_{timestamp}.json"
        filepath = self.review_buffer_dir / filename

        # Prepare data for serialization
        data = {
            "metadata": episode.metadata,
            "t_start": episode.t_start,
            "t_peak": episode.t_peak,
            "t_end": episode.t_end,
            "involved_agents": episode.involved_agents,
            "risk_score": episode.risk_score,
            "episode_type": episode.episode_type.value,
            "check_result": {
                "passed": check_result.passed,
                "reason": check_result.reason,
                "details": check_result.details,
            },
            "sstg_summary": episode.sstg.get_summary() if episode.sstg else None,
        }

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


def create_default_checker(
    output_dir: Path,
    strict: bool = False,
) -> SanityChecker:
    """
    Create a default sanity checker with standard settings.

    Args:
        output_dir: Output directory for review buffer
        strict: If True, use stricter thresholds

    Returns:
        Configured SanityChecker
    """
    review_buffer_dir = output_dir / "review_buffer"

    if strict:
        return SanityChecker(
            min_timesteps=10,
            max_timesteps=50,
            min_agents=2,
            max_agents=8,
            min_risk_score=0.3,
            require_edges=True,
            require_interaction_edges=True,
            review_buffer_dir=review_buffer_dir,
        )
    else:
        return SanityChecker(
            min_timesteps=5,
            max_timesteps=100,
            min_agents=2,
            max_agents=10,
            min_risk_score=0.1,
            require_edges=True,
            require_interaction_edges=False,
            review_buffer_dir=review_buffer_dir,
        )
