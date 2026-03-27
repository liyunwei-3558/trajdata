#!/usr/bin/env python
"""
Main entry point for the risk mining pipeline.

Extracts "Intersection Traffic Disturbance Scenarios" from trajectory data
and outputs Semantic Spatio-Temporal Graphs (SSTG) to a dual library system.

Usage:
    python run_pipeline.py --config config/rules_config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajdata import UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache

from src.core import Slicer, Episode
from src.rules import RuleRegistry, register_default_spatial_rules, register_default_dynamic_rules
from src.library import DualLibrary
from src.utils import setup_logger, create_default_checker, SanityChecker


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_pipeline(config: Dict[str, Any]) -> Tuple:
    """
    Set up all pipeline components from config.

    Returns:
        (dataset, rule_registry, slicer, checker, dual_lib)
    """
    # Create dataset
    dataset = UnifiedDataset(
        desired_data=config["data"]["desired_data"],
        data_dirs=config["data"]["data_dirs"],
        desired_dt=config["data"].get("desired_dt", 0.1),
    )

    # Create rule registry
    rule_registry = RuleRegistry()

    # Register spatial rules
    if config["rules"]["spatial"].get("in_roi", {}).get("enabled", False):
        from src.rules import SpatialROIRule
        roi_config = config["rules"]["spatial"]["in_roi"]
        rule_registry.register(SpatialROIRule(
            roi_radius=roi_config.get("roi_radius", 50.0),
        ))

    if config["rules"]["spatial"].get("conflict_lane", {}).get("enabled", False):
        from src.rules import ConflictLaneRule
        conflict_config = config["rules"]["spatial"]["conflict_lane"]
        rule_registry.register(ConflictLaneRule(
            conflict_distance=conflict_config.get("conflict_distance", 10.0),
        ))

    # Register dynamic rules
    if config["rules"]["dynamic"].get("ttc", {}).get("enabled", False):
        from src.rules import TTCCriticalRule
        ttc_config = config["rules"]["dynamic"]["ttc"]
        rule_registry.register(TTCCriticalRule(
            ttc_threshold=ttc_config.get("ttc_threshold", 3.0),
        ))

    if config["rules"]["dynamic"].get("deceleration", {}).get("enabled", False):
        from src.rules import DecelerationRule
        decel_config = config["rules"]["dynamic"]["deceleration"]
        rule_registry.register(DecelerationRule(
            decel_threshold=decel_config.get("decel_threshold", 3.0),
        ))

    # Create slicer
    slicer_config = config.get("slicer", {})
    slicer = Slicer(
        pre_buffer_sec=slicer_config.get("pre_buffer_sec", 2.0),
        post_buffer_sec=slicer_config.get("post_buffer_sec", 2.0),
        ttc_threshold=config["rules"]["dynamic"].get("ttc", {}).get("ttc_threshold", 3.0),
    )

    # Create checker
    output_dir = Path(config["output"]["output_dir"]).expanduser()
    checker = create_default_checker(output_dir)

    # Create dual library
    dual_lib = DualLibrary(output_dir)

    return dataset, rule_registry, slicer, checker, dual_lib


def process_scene(
    scene,
    cache: DataFrameCache,
    slicer: Slicer,
    rule_registry: RuleRegistry,
    checker: SanityChecker,
    dual_lib: DualLibrary,
    center_point: Optional[Tuple[float, float]] = None,
    logger=None,
) -> Dict[str, int]:
    """
    Process a single scene and extract episodes.

    Returns:
        Statistics dict with counts of extracted/failed episodes
    """
    stats = {
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    # Extract episodes
    episodes = slicer.extract_episodes(scene, cache, center_point)
    stats["episodes_extracted"] = len(episodes)

    if logger:
        logger.info(f"Scene {scene.name}: Extracted {len(episodes)} episodes")

    # Process each episode
    for episode in episodes:
        # Sanity check
        check_result = checker.check_episode(episode)

        if check_result.passed:
            # Add to library
            dual_lib.add_episode(episode)
            stats["episodes_passed"] += 1

            if logger:
                logger.debug(
                    f"  Episode passed: t_peak={episode.t_peak}, "
                    f"type={episode.episode_type.value}, score={episode.risk_score:.2f}"
                )
        else:
            # Save to review buffer
            checker.save_to_review_buffer(episode, check_result)
            stats["episodes_failed"] += 1

            if logger:
                logger.debug(
                    f"  Episode failed: t_peak={episode.t_peak}, "
                    f"reason={check_result.reason}"
                )

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the risk mining pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/rules_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=None,
        help="Limit number of scenes to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Override output dir if specified
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(level=log_level)

    logger.info("Starting risk mining pipeline")
    logger.info(f"Config: {config_path}")

    # Setup pipeline components
    logger.info("Setting up pipeline components...")
    dataset, rule_registry, slicer, checker, dual_lib = setup_pipeline(config)

    logger.info(f"Dataset: {config['data']['desired_data']}")
    logger.info(f"Output directory: {config['output']['output_dir']}")
    logger.info(f"Enabled rules: {[r.name for r in rule_registry.get_enabled_rules()]}")

    # Process scenes
    logger.info("Processing scenes...")

    total_stats = {
        "scenes_processed": 0,
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    scene_iter = dataset.scenes()
    for i, scene in enumerate(scene_iter):
        if args.scene_limit is not None and i >= args.scene_limit:
            logger.info(f"Reached scene limit ({args.scene_limit}), stopping")
            break

        logger.info(f"Processing scene {i+1}: {scene.name}")

        # Get cache path
        cache_path = Path("~/.unified_data_cache").expanduser()

        # Create cache for this scene
        cache = DataFrameCache(cache_path, scene)

        # Process scene
        stats = process_scene(
            scene=scene,
            cache=cache,
            slicer=slicer,
            rule_registry=rule_registry,
            checker=checker,
            dual_lib=dual_lib,
            logger=logger,
        )

        # Accumulate stats
        total_stats["scenes_processed"] += 1
        total_stats["episodes_extracted"] += stats["episodes_extracted"]
        total_stats["episodes_passed"] += stats["episodes_passed"]
        total_stats["episodes_failed"] += stats["episodes_failed"]

    # Save summary
    logger.info("Pipeline complete, saving summary...")
    summary_path = dual_lib.save_summary(additional_stats=total_stats)

    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Total statistics:")
    logger.info(f"  Scenes processed: {total_stats['scenes_processed']}")
    logger.info(f"  Episodes extracted: {total_stats['episodes_extracted']}")
    logger.info(f"  Episodes passed: {total_stats['episodes_passed']}")
    logger.info(f"  Episodes failed: {total_stats['episodes_failed']}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Scenes processed: {total_stats['scenes_processed']}")
    print(f"Episodes extracted: {total_stats['episodes_extracted']}")
    print(f"Episodes passed: {total_stats['episodes_passed']}")
    print(f"Episodes failed: {total_stats['episodes_failed']}")
    print(f"\nOutput directory: {config['output']['output_dir']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
