#!/usr/bin/env python
"""
Helper script for running the pipeline over a subset of scenes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_pipeline import load_config, process_scene, setup_pipeline


def extract_multiple_scenes(scene_limit: int = 5, config_path: str = "config/rules_config.yaml"):
    config = load_config(Path(config_path))
    dataset, registry, slicer, checker, dual_lib = setup_pipeline(config)

    from trajdata.caching.df_cache import DataFrameCache

    cache_path = Path("~/.unified_data_cache").expanduser()
    total_stats = {
        "scenes_processed": 0,
        "episodes_extracted": 0,
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    for scene_index, scene in enumerate(dataset.scenes()):
        if scene_index >= scene_limit:
            break

        cache = DataFrameCache(cache_path, scene)
        stats = process_scene(scene, cache, slicer, registry, checker, dual_lib)
        total_stats["scenes_processed"] += 1
        for key in ("episodes_extracted", "episodes_passed", "episodes_failed"):
            total_stats[key] += stats[key]

    dual_lib.save_summary(additional_stats=total_stats)
    return dual_lib, total_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract episodes from a subset of scenes.")
    parser.add_argument("--scene-limit", type=int, default=5)
    parser.add_argument("--config", type=str, default="config/rules_config.yaml")
    args = parser.parse_args()
    extract_multiple_scenes(scene_limit=args.scene_limit, config_path=args.config)
