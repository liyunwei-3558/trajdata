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

    data_cfg = config["data"]
    dataset = UnifiedDataset(
        desired_data=data_cfg["desired_data"],
        data_dirs=data_cfg["data_dirs"],
        desired_dt=data_cfg.get("desired_dt", 0.1),
        centric=data_cfg.get("centric", "scene"),
        cache_location=data_cfg.get("cache_location", "~/.unified_data_cache"),
        rebuild_cache=data_cfg.get("rebuild_cache", False),
        rebuild_maps=data_cfg.get("rebuild_maps", False),
        num_workers=data_cfg.get("num_workers", 0),
        verbose=data_cfg.get("verbose", False),
    )

    registry = RuleRegistry()
    if config["rules"]["dynamic"]["ttc"].get("enabled", False):
        ttc_cfg = config["rules"]["dynamic"]["ttc"]
        causal_filter_cfg = ttc_cfg.get("causal_filter", {})
        registry.register(
            TTCCriticalRule(
                ttc_threshold=ttc_cfg.get("ttc_threshold", 2.5),
                map_cache_path=dataset.cache_path,
                causal_filter_enabled=causal_filter_cfg.get("enabled", True),
                lane_constrained_types=causal_filter_cfg.get("lane_constrained_types"),
                current_lane_max_dist=causal_filter_cfg.get("current_lane_max_dist", 2.5),
                lane_query_radius_m=causal_filter_cfg.get("lane_query_radius_m", 8.0),
                lane_heading_threshold_deg=causal_filter_cfg.get("lane_heading_threshold_deg", 35.0),
                low_confidence_threshold=causal_filter_cfg.get("low_confidence_threshold", 0.4),
                intersection_proximity_radius_m=causal_filter_cfg.get("intersection_proximity_radius_m", 18.0),
                lane_reachable_hops=causal_filter_cfg.get("lane_reachable_hops", 3),
                lane_conflict_distance_m=causal_filter_cfg.get("lane_conflict_distance_m", 3.5),
                lane_conflict_heading_diff_deg=causal_filter_cfg.get("lane_conflict_heading_diff_deg", 30.0),
                manual_review_on_low_confidence=causal_filter_cfg.get("manual_review_on_low_confidence", True),
                track_birth_manual_review_window_sec=causal_filter_cfg.get("track_birth_manual_review_window_sec", 2.0),
            )
        )
    if config["rules"]["spatial"]["in_roi"].get("enabled", False):
        registry.register(
            SpatialROIRule(
                roi_radius=config["rules"]["spatial"]["in_roi"].get("roi_radius", 50.0),
                context_roi_radius=config["rules"]["spatial"]["in_roi"].get("context_roi_radius"),
            )
        )

    slicer = Slicer(
        pre_buffer_sec=config["slicer"].get("pre_buffer_sec", 2.0),
        post_buffer_sec=config["slicer"].get("post_buffer_sec", 2.0),
        ego_motion_threshold=config["slicer"].get("ego_motion_threshold", 0.5),
        ttc_event_threshold=config["slicer"].get("ttc_event_threshold", 2.5),
        pet_event_threshold=config["slicer"].get("pet_event_threshold", 2.0),
        ttc_clear_threshold=config["slicer"].get("ttc_clear_threshold", 5.0),
        min_peak_gap_sec=config["slicer"].get("min_peak_gap_sec"),
        max_episodes_per_scene=config["slicer"].get("max_episodes_per_scene", 5),
        reaction_latency_sec=config["slicer"].get("reaction_latency_sec", 1.8),
        stable_duration_sec=config["slicer"].get("stable_duration_sec", 1.0),
        dynamic_acc_threshold=config["slicer"].get("dynamic_acc_threshold", 3.0),
        dynamic_jerk_threshold=config["slicer"].get("dynamic_jerk_threshold", 4.0),
        lateral_speed_threshold=config["slicer"].get("lateral_speed_threshold", 0.5),
        launch_acc_threshold=config["slicer"].get("launch_acc_threshold", 2.0),
        forward_roi_distance=config["slicer"].get("forward_roi_distance", 50.0),
        conflict_zone_radius=config["slicer"].get("conflict_zone_radius", 6.0),
        visibility_fov_deg=config["slicer"].get("visibility_fov_deg", 140.0),
        occlusion_lateral_threshold=config["slicer"].get("occlusion_lateral_threshold", 2.5),
        pet_prediction_horizon_sec=config["slicer"].get("pet_prediction_horizon_sec", 5.0),
        stop_speed_threshold=config["slicer"].get("stop_speed_threshold", 0.1),
        zero_acc_threshold=config["slicer"].get("zero_acc_threshold", 0.3),
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
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-maps", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.rebuild_cache:
        config.setdefault("data", {})["rebuild_cache"] = True
    if args.rebuild_maps:
        config.setdefault("data", {})["rebuild_maps"] = True
    if args.verbose:
        config.setdefault("data", {})["verbose"] = True

    logger = setup_logger(level="DEBUG" if args.verbose else "INFO")
    dataset, registry, slicer, checker, dual_lib = setup_pipeline(config)

    from trajdata.caching.df_cache import DataFrameCache

    total_stats = {
        "scenes_processed": 0,
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    for scene_index, scene in enumerate(dataset.scenes()):
        if args.scene_limit is not None and scene_index >= args.scene_limit:
            break

        cache = DataFrameCache(dataset.cache_path, scene)
        scene_stats = process_scene(scene, cache, slicer, registry, checker, dual_lib, logger=logger)
        total_stats["scenes_processed"] += 1
        for key in ("episodes_extracted", "episodes_passed", "episodes_failed"):
            total_stats[key] += scene_stats[key]

    summary_path = dual_lib.save_summary(additional_stats=total_stats)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
