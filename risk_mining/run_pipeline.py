#!/usr/bin/env python
"""
Main entry point for the risk mining pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.core import Slicer
from src.library import DualLibrary
from src.rules import RuleRegistry, SpatialROIRule, TTCCriticalRule
from src.utils import SanityChecker, create_default_checker, setup_logger


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_pipeline(config: Dict[str, Any]) -> Tuple[Any, RuleRegistry, Slicer, SanityChecker, DualLibrary]:
    from trajdata import UnifiedDataset

    dataset = UnifiedDataset(
        desired_data=config["data"]["desired_data"],
        data_dirs=config["data"]["data_dirs"],
        desired_dt=config["data"].get("desired_dt", 0.1),
    )

    registry = RuleRegistry()
    if config["rules"]["spatial"]["in_roi"].get("enabled", False):
        registry.register(
            SpatialROIRule(roi_radius=config["rules"]["spatial"]["in_roi"].get("roi_radius", 50.0))
        )
    if config["rules"]["dynamic"]["ttc"].get("enabled", False):
        registry.register(
            TTCCriticalRule(ttc_threshold=config["rules"]["dynamic"]["ttc"].get("ttc_threshold", 2.5))
        )

    slicer = Slicer(
        pre_buffer_sec=config["slicer"].get("pre_buffer_sec", 2.0),
        post_buffer_sec=config["slicer"].get("post_buffer_sec", 2.0),
    )

    output_dir = Path(config["output"]["output_dir"]).expanduser()
    checker = create_default_checker(output_dir)
    dual_lib = DualLibrary(output_dir)
    return dataset, registry, slicer, checker, dual_lib


def process_scene(scene: Any, cache: Any, slicer: Slicer, registry: RuleRegistry, checker: SanityChecker, dual_lib: DualLibrary, logger=None) -> Dict[str, int]:
    stats = {"episodes_extracted": 0, "episodes_passed": 0, "episodes_failed": 0}

    episodes = slicer.extract_episodes(scene, cache)
    stats["episodes_extracted"] = len(episodes)
    if logger:
        logger.info(f"Scene {scene.name}: extracted {len(episodes)} episode(s)")

    for episode in episodes:
        episode.sstg = registry.apply_all(episode)
        validation = checker.validate_episode(episode)

        if validation.passed:
            dual_lib.add_episode(episode)
            stats["episodes_passed"] += 1
        else:
            checker.save_for_review(episode, validation)
            stats["episodes_failed"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the risk mining pipeline.")
    parser.add_argument("--config", type=str, default="config/rules_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir

    logger = setup_logger(level="DEBUG" if args.verbose else "INFO")
    dataset, registry, slicer, checker, dual_lib = setup_pipeline(config)

    from trajdata.caching.df_cache import DataFrameCache

    total_stats = {
        "scenes_processed": 0,
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    cache_path = Path("~/.unified_data_cache").expanduser()
    for scene_index, scene in enumerate(dataset.scenes()):
        if args.scene_limit is not None and scene_index >= args.scene_limit:
            break

        cache = DataFrameCache(cache_path, scene)
        scene_stats = process_scene(scene, cache, slicer, registry, checker, dual_lib, logger=logger)
        total_stats["scenes_processed"] += 1
        for key in ("episodes_extracted", "episodes_passed", "episodes_failed"):
            total_stats[key] += scene_stats[key]

    summary_path = dual_lib.save_summary(additional_stats=total_stats)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
