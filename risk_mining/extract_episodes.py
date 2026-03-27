#!/usr/bin/env python
"""
Extract risk episodes from multiple scenes.

Usage:
    python extract_episodes.py --scene-limit 5 --min-risk-score 0.3
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajdata import UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache

from src.core import Slicer, Episode
from src.library import DualLibrary
from src.utils import SanityChecker


def extract_multiple_scenes(
    scene_limit: int = 5,
    min_risk_score: float = 0.1,
    location: str = "sind-cqNR",
    max_agents: int = 30,  # Increased threshold
):
    """Extract episodes from multiple scenes."""

    # Configuration
    data_dirs = {
        "sind": "/home/lyw/datasets/SinD_dataset",
    }

    print(f"Loading dataset: {location}")
    dataset = UnifiedDataset(
        desired_data=[location],
        data_dirs=data_dirs,
        desired_dt=0.1,
    )

    # Create slicer
    slicer = Slicer(
        pre_buffer_sec=2.0,
        post_buffer_sec=2.0,
        ttc_threshold=3.0,
    )

    # Create checker with relaxed thresholds
    output_dir = Path("./output/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    checker = SanityChecker(
        min_timesteps=5,
        max_timesteps=100,
        min_agents=2,
        max_agents=max_agents,
        min_risk_score=0.1,
        require_interaction_edges=True,
        review_buffer_dir=output_dir / "review_buffer",
    )

    # Create dual library
    dual_lib = DualLibrary(output_dir)

    # Cache path
    cache_path = Path("~/.unified_data_cache").expanduser()

    # Statistics
    total_stats = {
        "scenes_processed": 0,
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    # Process scenes
    print(f"\nProcessing up to {scene_limit} scenes...")
    print("-" * 60)

    scene_iter = dataset.scenes()
    for i, scene in enumerate(scene_iter):
        if i >= scene_limit:
            break

        print(f"\nScene {i+1}/{scene_limit}: {scene.name}")
        print(f"  Timesteps: {scene.length_timesteps}, Agents: {len(scene.agents)}")

        # Create cache for this scene
        cache = DataFrameCache(cache_path, scene)

        # Extract episodes
        episodes = slicer.extract_episodes(scene, cache)

        # Filter by risk score
        high_risk_episodes = [e for e in episodes if e.risk_score >= min_risk_score]
        print(f"  Episodes extracted: {len(episodes)} (risk >= {min_risk_score}: {len(high_risk_episodes)})")

        for episode in high_risk_episodes:
            # Sanity check
            check_result = checker.check_episode(episode)

            if check_result.passed:
                dual_lib.add_episode(episode)
                total_stats["episodes_passed"] += 1
                print(f"    ✓ Episode passed: t_peak={episode.t_peak}, "
                      f"type={episode.episode_type.value}, score={episode.risk_score:.2f}, "
                      f"agents={len(episode.involved_agents)}")
            else:
                checker.save_to_review_buffer(episode, check_result)
                total_stats["episodes_failed"] += 1
                print(f"    ✗ Episode failed: t_peak={episode.t_peak}, "
                      f"reason={check_result.reason}")

        total_stats["scenes_processed"] += 1
        total_stats["episodes_extracted"] += len(high_risk_episodes)

    # Save summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Scenes processed: {total_stats['scenes_processed']}")
    print(f"Episodes extracted: {total_stats['episodes_extracted']}")
    print(f"Episodes passed: {total_stats['episodes_passed']}")
    print(f"Episodes failed: {total_stats['episodes_failed']}")

    # Save library summary
    summary_path = dual_lib.save_summary(additional_stats=total_stats)
    print(f"\nSummary saved to: {summary_path}")

    return dual_lib, total_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract risk episodes from multiple scenes")
    parser.add_argument("--scene-limit", type=int, default=5,
                       help="Number of scenes to process")
    parser.add_argument("--min-risk-score", type=float, default=0.2,
                       help="Minimum risk score threshold")
    parser.add_argument("--location", type=str, default="sind-cqNR",
                       help="Dataset location to use")
    parser.add_argument("--max-agents", type=int, default=30,
                       help="Maximum number of agents per episode")

    args = parser.parse_args()

    extract_multiple_scenes(
        scene_limit=args.scene_limit,
        min_risk_score=args.min_risk_score,
        location=args.location,
        max_agents=args.max_agents,
    )
