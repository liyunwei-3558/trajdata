"""
Sanity checks and manual review support for generated SSTGs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.scene_graph import EdgeType, SSTG
from ..core.slicer import Episode


@dataclass
class SanityCheckResult:
    passed: bool
    reason: str
    details: Dict[str, Any]
    review_path: Optional[Path] = None


def validate_graph(sstg: SSTG) -> SanityCheckResult:
    manual_review_status = sstg.metadata.get("manual_review_status")
    if manual_review_status is not None:
        return SanityCheckResult(
            passed=False,
            reason=str(sstg.metadata.get("manual_review_reason", "Graph flagged for manual review.")),
            details={
                "manual_review_status": manual_review_status,
                "manual_review_details": sstg.metadata.get("manual_review_details", {}),
            },
        )

    causal_edges = 0
    for _, _, data in sstg.graph.edges(data=True):
        if data.get("edge_type") == EdgeType.CAUSAL.value:
            causal_edges += 1

    if causal_edges > 0:
        return SanityCheckResult(
            passed=True,
            reason="Graph contains at least one causal edge.",
            details={"causal_edge_count": causal_edges},
        )

    return SanityCheckResult(
        passed=False,
        reason="Graph does not contain any causal or interaction-explaining edge.",
        details={"causal_edge_count": causal_edges},
    )


class SanityChecker:
    """Validate SSTGs and persist failures for manual review."""

    def __init__(self, manual_review_dir: Path):
        self.manual_review_dir = manual_review_dir

    def validate_episode(self, episode: Episode) -> SanityCheckResult:
        if episode.sstg is None:
            return SanityCheckResult(
                passed=False,
                reason="Episode has no SSTG attached.",
                details={"causal_edge_count": 0},
            )
        if episode.metadata.get("manual_review_required"):
            return SanityCheckResult(
                passed=False,
                reason=str(episode.metadata.get("manual_review_reason", "Episode flagged for manual review.")),
                details={
                    "manual_review_status": episode.metadata.get("manual_review_status", "requested"),
                    "manual_review_details": episode.metadata.get("manual_review_details", {}),
                },
            )
        return validate_graph(episode.sstg)

    def save_for_review(self, episode: Episode, result: SanityCheckResult) -> Path:
        self.manual_review_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_review_{episode.scene_name}_{episode.t_peak}_{timestamp}.json"
        path = self.manual_review_dir / filename

        payload = {
            "scene": {
                "scene_id": episode.scene_id,
                "scene_name": episode.scene_name,
                "env_name": episode.env_name,
            },
            "timeframe": {
                "t_start": episode.t_start,
                "t_peak": episode.t_peak,
                "t_end": episode.t_end,
                "semantic_timesteps": episode.semantic_timesteps,
            },
            "ego_agent_id": episode.ego_agent_id,
            "involved_agents": episode.involved_agents,
            "episode_type": episode.episode_type.value,
            "risk_score": episode.risk_score,
            "applied_rules": episode.rule_trace,
            "failure": {
                "reason": result.reason,
                "details": result.details,
            },
            "sstg": episode.sstg.to_dict() if episode.sstg is not None else None,
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        result.review_path = path
        return path


def create_default_checker(output_dir: Path) -> SanityChecker:
    return SanityChecker(manual_review_dir=output_dir / "manual_review")
