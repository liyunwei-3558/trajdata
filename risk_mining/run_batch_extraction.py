#!/usr/bin/env python
"""
Batch runner for city-level risk mining extraction.
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_config, process_scene, setup_pipeline
from src.utils import setup_logger


LOCATION_GROUPS: Dict[str, List[str]] = {
    "tj": ["sind-tj"],
    "cq": ["sind-cqNR", "sind-cqIR", "sind-cqR"],
}


def _expand_locations(locations: List[str]) -> List[str]:
    expanded: List[str] = []
    for location in locations:
        key = location.strip()
        if key in LOCATION_GROUPS:
            expanded.extend(LOCATION_GROUPS[key])
            continue
        if key.startswith("sind-"):
            expanded.append(key)
            continue
        expanded.append(f"sind-{key}")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch extraction for selected SinD city groups.")
    parser.add_argument("--config", type=str, default="config/rules_config.yaml")
    parser.add_argument("--locations", nargs="+", default=["tj", "cq"])
    parser.add_argument("--output-root", type=str, default="./risk_mining/output/batch_runs")
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_config = load_config(Path(args.config).expanduser())
    output_root = Path(args.output_root).expanduser()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desired_tags = _expand_locations(args.locations)
    logger = setup_logger(level="DEBUG" if args.verbose else "INFO")

    for desired_tag in desired_tags:
        config = copy.deepcopy(base_config)
        config.setdefault("data", {})["desired_data"] = [desired_tag]
        tag_output_dir = output_root / f"{run_stamp}_{desired_tag.replace('sind-', '')}"
        config.setdefault("output", {})["output_dir"] = str(tag_output_dir)

        logger.info("Starting batch extraction for %s -> %s", desired_tag, tag_output_dir)
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
        logger.info("Finished %s with summary: %s", desired_tag, summary_path)


if __name__ == "__main__":
    main()
