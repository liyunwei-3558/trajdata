from __future__ import annotations

import argparse
import gc
import csv
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from Simulation_test_toolchain.core.batch_metrics import (
    MetricContext,
    compute_batch_metrics,
    save_metrics,
)
from Simulation_test_toolchain.core.config import (
    CheckpointConfig,
    DatasetConfig,
    OutputConfig,
    PolicyConfig,
    ScenarioConfig,
    SimulationConfig,
    ToolchainConfig,
    VisualizationConfig,
)
from Simulation_test_toolchain.core.runner import (
    load_simulation_dataset,
    run_simulation,
)
from Simulation_test_toolchain.run_raw_open_loop_trajectory_batch import (
    _bool_rate,
    _csv_value,
    _is_metric_anomaly_row,
    _load_manifest,
    _manifest_key,
    _numeric_series,
    _safe_path_name,
)
from Simulation_test_toolchain.run_narrow_feasible_riskidm_batch import (
    SPEED_MODES,
    _boosted_speed,
    _candidate_sort_key as _narrow_candidate_sort_key,
    _has_valid_window,
    _motion_features,
    _motion_filter_level,
)


DEFAULT_LABEL_PATH = "datasets/SinD_dataset/Semantic_labels/scenarios.json"
DEFAULT_MPRTTC_EXTRA_LABEL_PATH = (
    "risk_mining/typical_risks_extract/high_risk_mprttc/output/"
    "high_risk_mprttc_non_tj_scenarios.json"
)
DEFAULT_QCNET_REPO = "/home/lyw/1TBSSD/lizhongze/DriverModel-FT/QCNet-main"
DEFAULT_QCNET_CKPT = (
    "/home/lyw/1TBSSD/lizhongze/DriverModel-FT/QCNet-main/QCNet_AV2.ckpt"
)
DEFAULT_INTERSECTIONS = ("cc", "tj", "cqIR", "cqNR", "cqR", "xa")
SCENARIO_TYPES = ("mprttc", "visual_shielding", "narrow_feasible_area")
SEMANTIC_AGENT_TYPES = ["VEHICLE", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"]
OUTPUT_NAMES = {
    "mprttc": "qcnet_semantic_mprttc_per_intersection_200",
    "visual_shielding": "qcnet_semantic_visual_shielding_per_intersection_200",
    "narrow_feasible_area": "qcnet_semantic_narrow_feasible_per_intersection_200",
}
POLICY_OUTPUT_DIR = "raw_open_loop_qcnet"
POLICY = "qcnet"
MANIFEST_BASE_FIELDS = (
    "policy",
    "scenario_type",
    "speed_mode",
    "pair_id",
    "intersection",
    "location",
    "scene_name",
    "scene_id",
    "scene_index",
    "run_id",
    "scenario_id",
    "agent_id",
    "challenger_id",
    "candidate_rank",
    "semantic_score",
    "label_start_frame",
    "label_end_frame",
    "init_timestep",
    "num_steps_requested",
    "original_initial_speed_mps",
    "initial_speed_override_mps",
    "speed_boost_factor",
    "gt_path_length_150m",
    "gt_displacement_150m",
    "motion_filter_level",
    "status",
    "run_dir",
    "trajectory_log",
    "metrics_json",
    "error",
    "elapsed_s",
    "metric_anomaly",
    "metric_anomaly_reason",
)


@dataclass(frozen=True)
class SemanticPlan:
    policy: str
    scenario_type: str
    intersection: str
    location: str
    scene_name: str
    scene_id: str
    scene_index: int
    run_id: str
    scenario_id: str
    agent_id: str
    challenger_id: Optional[str]
    candidate_rank: int
    semantic_score: float
    label_start_frame: int
    label_end_frame: int
    init_timestep: int
    num_steps: int
    semantic_label: Dict[str, Any]
    speed_mode: str = ""
    pair_id: str = ""
    original_initial_speed_mps: Optional[float] = None
    initial_speed_override_mps: Optional[float] = None
    speed_boost_factor: float = 1.0
    gt_path_length_150m: Optional[float] = None
    gt_displacement_150m: Optional[float] = None
    motion_filter_level: str = ""


@dataclass(frozen=True)
class NarrowCandidate:
    label: Dict[str, Any]
    intersection: str
    location: str
    scene_name: str
    scene_id: str
    scene_index: int
    agent_id: str
    init_timestep: int
    label_start_frame: int
    label_end_frame: int
    original_initial_speed_mps: float
    gt_path_length_150m: float
    gt_displacement_150m: float
    motion_filter_level: str


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else Path("Simulation_test_toolchain/batch_outputs")
        / OUTPUT_NAMES[args.scenario_type]
    )
    policy_dir = output_root / POLICY_OUTPUT_DIR
    if args.num_shards > 1:
        policy_dir = policy_dir / f"shard_{args.shard_index:02d}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    plans_by_intersection = build_plans(
        scenario_type=args.scenario_type,
        label_paths=[Path(path) for path in args.label_paths],
        data_dir=data_dir,
        intersections=tuple(args.intersections),
        target_per_intersection=args.target_per_intersection,
        target_total=args.target_total,
        num_steps=args.num_steps,
        history_steps=int(round(args.history_sec / args.dt)),
        future_steps=int(np.ceil(args.future_sec / args.dt)),
        candidate_buffer_factor=args.candidate_buffer_factor,
        strict_min_path_length=args.strict_min_path_length,
        strict_min_displacement=args.strict_min_displacement,
        strict_min_initial_speed=args.strict_min_initial_speed,
        relaxed_min_path_length=args.relaxed_min_path_length,
        relaxed_min_displacement=args.relaxed_min_displacement,
        relaxed_min_initial_speed=args.relaxed_min_initial_speed,
        max_original_initial_speed=args.max_original_initial_speed,
        require_boost_increase=not args.allow_non_increasing_boost,
        boost_min_speed=args.boost_min_speed,
        boost_max_speed=args.boost_max_speed,
        boost_scale=args.boost_scale,
        progress_interval=args.candidate_progress_interval,
        allow_unmet_requirements=args.allow_unmet_requirements,
    )
    total_plans = sum(len(plans) for plans in plans_by_intersection.values())
    print(
        f"[semantic-qcnet] scenario_type={args.scenario_type} planned={total_plans} "
        f"intersections={','.join(plans_by_intersection)} output={policy_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    metric_context = MetricContext(data_dir=data_dir, dt=args.dt)
    dataset_cache: Dict[Tuple[Any, ...], Any] = {}
    policy_cache: Dict[Tuple[Any, ...], Any] = {}
    rows_by_key = _load_manifest(policy_dir / "run_manifest.csv")
    if args.force:
        rows_by_key = {}

    for intersection, all_plans in plans_by_intersection.items():
        plans = _apply_shard(all_plans, args.shard_index, args.num_shards)
        print(
            f"[semantic-qcnet] start scenario_type={args.scenario_type} "
            f"intersection={intersection} planned_this_shard={len(plans)}",
            flush=True,
        )
        for plan in plans:
            key = _manifest_key(POLICY, plan.location, plan.run_id)
            if key in rows_by_key and not args.force:
                existing = rows_by_key[key]
                if existing.get("status") in {"completed", "failed"}:
                    continue
            run_dir = _run_dir(policy_dir, plan)
            row = _base_manifest_row(plan, run_dir)
            started = time.time()
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _make_config(plan, data_dir, run_dir, args)
                _write_config(run_dir / "config.yaml", cfg)
                print(
                    f"[semantic-qcnet] run type={plan.scenario_type} "
                    f"intersection={plan.intersection} rank={plan.candidate_rank} "
                    f"scenario={plan.scenario_id} location={plan.location} "
                    f"scene={plan.scene_name} ego={plan.agent_id} init={plan.init_timestep} "
                    f"score={plan.semantic_score:g}",
                    flush=True,
                )
                result = run_simulation(
                    cfg,
                    dataset_bundle=_dataset_bundle_for_config(cfg, dataset_cache),
                    policy_cache=policy_cache,
                )
                result.metadata.update(
                    {
                        "experiment_mode": f"dataset_open_loop_{plan.scenario_type}",
                        "open_loop_definition": (
                            "non-ego agents replay ground truth; ego is controlled by QCNet"
                        ),
                        "runner_mode": cfg.simulation.mode,
                        "ego_policy": POLICY,
                        "scenario_type": plan.scenario_type,
                        "speed_mode": plan.speed_mode,
                        "pair_id": plan.pair_id,
                        "run_id": plan.run_id,
                        "semantic_label_id": plan.scenario_id,
                        "semantic_label": plan.semantic_label,
                        "candidate_rank": plan.candidate_rank,
                        "semantic_score": plan.semantic_score,
                        "original_initial_speed_mps": plan.original_initial_speed_mps,
                        "initial_speed_override_mps": plan.initial_speed_override_mps,
                        "speed_boost_factor": plan.speed_boost_factor,
                        "gt_path_length_150m": plan.gt_path_length_150m,
                        "gt_displacement_150m": plan.gt_displacement_150m,
                        "motion_filter_level": plan.motion_filter_level,
                    }
                )
                result.save_json(run_dir / "trajectory_log.json")
                metrics = compute_batch_metrics(result, data_dir, metric_context)
                metrics.update(_compute_qcnet_command_metrics(result))
                save_metrics(run_dir / "metrics.json", result.metadata, metrics)
                row.update(metrics)
                row["status"] = "completed"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
                print(
                    f"[semantic-qcnet] failed scenario={plan.scenario_id}: {exc}",
                    flush=True,
                )
            finally:
                row["elapsed_s"] = round(time.time() - started, 3)
                rows_by_key[key] = row
                _write_outputs(policy_dir, rows_by_key.values())
                _cleanup_torch_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QCNet dataset-open-loop batches on semantic scenario labels."
    )
    parser.add_argument("--scenario-type", choices=SCENARIO_TYPES, required=True)
    parser.add_argument("--policy", choices=(POLICY,), default=POLICY)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--label-paths", nargs="+", default=None)
    parser.add_argument("--intersections", nargs="+", default=list(DEFAULT_INTERSECTIONS))
    parser.add_argument("--target-per-intersection", type=int, default=200)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--map-radius", type=float, default=150.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--qcnet-prediction-interval-steps", type=int, default=15)
    parser.add_argument("--qcnet-repo-path", type=str, default=DEFAULT_QCNET_REPO)
    parser.add_argument("--qcnet-ckpt-path", type=str, default=DEFAULT_QCNET_CKPT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--candidate-buffer-factor", type=float, default=2.0)
    parser.add_argument("--candidate-progress-interval", type=int, default=2000)
    parser.add_argument("--strict-min-path-length", type=float, default=8.0)
    parser.add_argument("--strict-min-displacement", type=float, default=5.0)
    parser.add_argument("--strict-min-initial-speed", type=float, default=0.5)
    parser.add_argument("--relaxed-min-path-length", type=float, default=4.0)
    parser.add_argument("--relaxed-min-displacement", type=float, default=2.0)
    parser.add_argument("--relaxed-min-initial-speed", type=float, default=0.2)
    parser.add_argument("--max-original-initial-speed", type=float, default=8.5)
    parser.add_argument("--allow-non-increasing-boost", action="store_true")
    parser.add_argument(
        "--allow-unmet-requirements",
        action="store_true",
        help=(
            "For visual_shielding, skip valid-window and history-window gating so "
            "scene requirements can be unmet and still count toward the target."
        ),
    )
    parser.add_argument("--boost-min-speed", type=float, default=9.0)
    parser.add_argument("--boost-max-speed", type=float, default=17.0)
    parser.add_argument("--boost-scale", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.label_paths is None:
        args.label_paths = [DEFAULT_LABEL_PATH]
        if args.scenario_type == "mprttc":
            args.label_paths.append(DEFAULT_MPRTTC_EXTRA_LABEL_PATH)
    return args


def build_plans(
    *,
    scenario_type: str,
    label_paths: Sequence[Path],
    data_dir: Path,
    intersections: Sequence[str],
    target_per_intersection: int,
    target_total: Optional[int],
    num_steps: int,
    history_steps: int,
    future_steps: int,
    candidate_buffer_factor: float,
    strict_min_path_length: float,
    strict_min_displacement: float,
    strict_min_initial_speed: float,
    relaxed_min_path_length: float,
    relaxed_min_displacement: float,
    relaxed_min_initial_speed: float,
    max_original_initial_speed: float,
    require_boost_increase: bool,
    boost_min_speed: float,
    boost_max_speed: float,
    boost_scale: float,
    progress_interval: int,
    allow_unmet_requirements: bool,
) -> Dict[str, List[SemanticPlan]]:
    if scenario_type == "visual_shielding" and target_total is not None:
        return build_visual_shielding_plans(
            label_paths=label_paths,
            data_dir=data_dir,
            intersections=intersections,
            target_total=target_total,
            num_steps=num_steps,
            history_steps=history_steps,
            future_steps=future_steps,
            allow_unmet_requirements=allow_unmet_requirements,
        )
    if scenario_type == "narrow_feasible_area":
        return build_narrow_plans(
            label_paths=label_paths,
            data_dir=data_dir,
            intersections=intersections,
            target_per_intersection=target_per_intersection,
            num_steps=num_steps,
            history_steps=history_steps,
            future_steps=future_steps,
            candidate_buffer_factor=candidate_buffer_factor,
            strict_min_path_length=strict_min_path_length,
            strict_min_displacement=strict_min_displacement,
            strict_min_initial_speed=strict_min_initial_speed,
            relaxed_min_path_length=relaxed_min_path_length,
            relaxed_min_displacement=relaxed_min_displacement,
            relaxed_min_initial_speed=relaxed_min_initial_speed,
            max_original_initial_speed=max_original_initial_speed,
            require_boost_increase=require_boost_increase,
            boost_min_speed=boost_min_speed,
            boost_max_speed=boost_max_speed,
            boost_scale=boost_scale,
            progress_interval=progress_interval,
        )
    return build_label_plans(
        scenario_type=scenario_type,
        label_paths=label_paths,
        data_dir=data_dir,
        intersections=intersections,
        target_per_intersection=target_per_intersection,
        num_steps=num_steps,
        history_steps=history_steps,
        future_steps=future_steps,
        allow_unmet_requirements=allow_unmet_requirements,
    )


def build_label_plans(
    *,
    scenario_type: str,
    label_paths: Sequence[Path],
    data_dir: Path,
    intersections: Sequence[str],
    target_per_intersection: int,
    num_steps: int,
    history_steps: int,
    future_steps: int,
    allow_unmet_requirements: bool = False,
) -> Dict[str, List[SemanticPlan]]:
    labels = _load_labels(label_paths, scenario_type)
    scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    scene_length_cache: Dict[Tuple[str, str], int] = {}
    scene_index_cache: Dict[str, Dict[str, int]] = {}
    best_by_window: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    allowed_intersections = set(intersections)
    target_buffer = target_per_intersection
    accepted_by_intersection: Dict[str, int] = {key: 0 for key in intersections}
    closed_intersections = set()
    for label in sorted(labels, key=lambda item: _label_sort_key(item, scenario_type)):
        location = str(label["location"])
        intersection = _canonical_intersection(location)
        if intersection not in allowed_intersections:
            continue
        if intersection in closed_intersections:
            if len(closed_intersections) == len(allowed_intersections):
                break
            continue
        if location not in scene_index_cache:
            scene_index_cache[location] = _scene_index_map(data_dir, location)
        scene_name = str(label["source_scene_name"])
        if scene_name not in scene_index_cache[location]:
            continue
        scene_id = _scene_id_from_name(location, scene_name)
        ego_id = str(label.get("agents", {}).get("ego_id", ""))
        if not ego_id:
            continue
        window = label.get("time_window", {})
        label_start = int(window.get("start_frame", 0))
        init_timestep = max(label_start, history_steps)
        scene_key = (location, scene_id)
        scene_tracks = scene_cache.get(scene_key)
        if scene_tracks is None:
            scene_tracks = _load_scene_tracks(data_dir, location, scene_id)
            scene_cache[scene_key] = scene_tracks
        scene_length = scene_length_cache.get(scene_key)
        if scene_length is None:
            scene_length = _scene_length(scene_tracks)
            scene_length_cache[scene_key] = scene_length
        if not allow_unmet_requirements:
            if not _has_valid_window(
                scene_tracks=scene_tracks,
                ego_id=ego_id,
                init_timestep=init_timestep,
                num_steps=num_steps,
                future_steps=future_steps,
                scene_length=scene_length,
            ):
                continue
            if not _has_history_window(scene_tracks, ego_id, init_timestep, history_steps):
                continue
        key = (location, scene_name, ego_id, init_timestep)
        existing = best_by_window.get(key)
        if existing is None or _label_sort_key(label, scenario_type) < _label_sort_key(
            existing, scenario_type
        ):
            if existing is None:
                accepted_by_intersection[intersection] += 1
            best_by_window[key] = label
        if accepted_by_intersection[intersection] >= target_buffer:
            closed_intersections.add(intersection)
            if len(closed_intersections) == len(allowed_intersections):
                break

    by_intersection: Dict[str, List[Dict[str, Any]]] = {
        intersection: [] for intersection in intersections
    }
    for label in best_by_window.values():
        intersection = _canonical_intersection(str(label["location"]))
        if intersection in by_intersection:
            by_intersection[intersection].append(label)

    plans_by_intersection: Dict[str, List[SemanticPlan]] = {}
    for intersection in intersections:
        candidates = by_intersection.get(intersection, [])
        candidates.sort(key=lambda item: _label_sort_key(item, scenario_type))
        selected = candidates[:target_per_intersection]
        plans = [
            _plan_from_label(
                label,
                rank,
                scenario_type=scenario_type,
                num_steps=num_steps,
                history_steps=history_steps,
                scene_index=scene_index_cache[str(label["location"])][
                    str(label["source_scene_name"])
                ],
            )
            for rank, label in enumerate(selected, start=1)
        ]
        plans_by_intersection[intersection] = plans
        print(
            f"[semantic-qcnet] type={scenario_type} intersection={intersection} "
            f"candidates={len(candidates)} selected={len(plans)}",
            flush=True,
        )
    return plans_by_intersection


def build_visual_shielding_plans(
    *,
    label_paths: Sequence[Path],
    data_dir: Path,
    intersections: Sequence[str],
    target_total: int,
    num_steps: int,
    history_steps: int,
    future_steps: int,
    allow_unmet_requirements: bool,
) -> Dict[str, List[SemanticPlan]]:
    labels = _load_labels(label_paths, "visual_shielding")
    scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    scene_length_cache: Dict[Tuple[str, str], int] = {}
    scene_index_cache: Dict[str, Dict[str, int]] = {}
    best_by_window: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    allowed_intersections = set(intersections)

    for label in sorted(labels, key=lambda item: _label_sort_key(item, "visual_shielding")):
        location = str(label["location"])
        intersection = _canonical_intersection(location)
        if intersection not in allowed_intersections:
            continue
        if location not in scene_index_cache:
            scene_index_cache[location] = _scene_index_map(data_dir, location)
        scene_name = str(label["source_scene_name"])
        if scene_name not in scene_index_cache[location]:
            continue
        scene_id = _scene_id_from_name(location, scene_name)
        ego_id = str(label.get("agents", {}).get("ego_id", ""))
        if not ego_id:
            continue
        window = label.get("time_window", {})
        label_start = int(window.get("start_frame", 0))
        label_end = int(window.get("end_frame", label_start))
        init_timestep = max(label_start, history_steps)
        scene_key = (location, scene_id)
        scene_tracks = scene_cache.get(scene_key)
        if scene_tracks is None:
            scene_tracks = _load_scene_tracks(data_dir, location, scene_id)
            scene_cache[scene_key] = scene_tracks
        scene_length = scene_length_cache.get(scene_key)
        if scene_length is None:
            scene_length = _scene_length(scene_tracks)
            scene_length_cache[scene_key] = scene_length
        if not allow_unmet_requirements:
            if not _has_valid_window(
                scene_tracks=scene_tracks,
                ego_id=ego_id,
                init_timestep=init_timestep,
                num_steps=num_steps,
                future_steps=future_steps,
                scene_length=scene_length,
            ):
                continue
            if not _has_history_window(scene_tracks, ego_id, init_timestep, history_steps):
                continue
        key = (location, scene_name, ego_id, init_timestep)
        existing = best_by_window.get(key)
        if existing is None or _label_sort_key(label, "visual_shielding") < _label_sort_key(
            existing, "visual_shielding"
        ):
            best_by_window[key] = label

    candidates = list(best_by_window.values())
    candidates.sort(key=lambda item: _label_sort_key(item, "visual_shielding"))
    selected = candidates[:target_total]
    by_intersection: Dict[str, List[Dict[str, Any]]] = {
        intersection: [] for intersection in intersections
    }
    for label in selected:
        intersection = _canonical_intersection(str(label["location"]))
        if intersection in by_intersection:
            by_intersection[intersection].append(label)

    plans_by_intersection: Dict[str, List[SemanticPlan]] = {}
    for intersection in intersections:
        candidates_for_intersection = by_intersection.get(intersection, [])
        plans = [
            _plan_from_label(
                label,
                rank,
                scenario_type="visual_shielding",
                num_steps=num_steps,
                history_steps=history_steps,
                scene_index=scene_index_cache[str(label["location"])][
                    str(label["source_scene_name"])
                ],
            )
            for rank, label in enumerate(candidates_for_intersection, start=1)
        ]
        plans_by_intersection[intersection] = plans
        print(
            f"[semantic-qcnet] type=visual_shielding intersection={intersection} "
            f"candidates={len(candidates_for_intersection)} selected={len(plans)}",
            flush=True,
        )
    return plans_by_intersection


def build_narrow_plans(
    *,
    label_paths: Sequence[Path],
    data_dir: Path,
    intersections: Sequence[str],
    target_per_intersection: int,
    num_steps: int,
    history_steps: int,
    future_steps: int,
    candidate_buffer_factor: float,
    strict_min_path_length: float,
    strict_min_displacement: float,
    strict_min_initial_speed: float,
    relaxed_min_path_length: float,
    relaxed_min_displacement: float,
    relaxed_min_initial_speed: float,
    max_original_initial_speed: float,
    require_boost_increase: bool,
    boost_min_speed: float,
    boost_max_speed: float,
    boost_scale: float,
    progress_interval: int,
) -> Dict[str, List[SemanticPlan]]:
    labels = _load_labels(label_paths, "narrow_feasible_area")
    scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    scene_length_cache: Dict[Tuple[str, str], int] = {}
    scene_index_cache: Dict[str, Dict[str, int]] = {}
    best_by_window: Dict[Tuple[str, str, str, int], NarrowCandidate] = {}
    allowed_intersections = set(intersections)
    target_pairs_per_intersection = max(1, int(target_per_intersection) // len(SPEED_MODES))
    target_buffer = max(
        1, int(np.ceil(target_pairs_per_intersection * candidate_buffer_factor))
    )
    accepted_by_intersection: Dict[str, int] = {key: 0 for key in intersections}
    closed_intersections = set()
    scanned = 0

    for label in labels:
        scanned += 1
        if progress_interval > 0 and scanned % progress_interval == 0:
            print(
                f"[semantic-qcnet] narrow scan scanned={scanned} "
                f"accepted={accepted_by_intersection}",
                flush=True,
            )
        location = str(label["location"])
        intersection = _canonical_intersection(location)
        if intersection not in allowed_intersections:
            continue
        if intersection in closed_intersections:
            if len(closed_intersections) == len(allowed_intersections):
                break
            continue
        if location not in scene_index_cache:
            scene_index_cache[location] = _scene_index_map(data_dir, location)
        scene_name = str(label["source_scene_name"])
        if scene_name not in scene_index_cache[location]:
            continue
        scene_id = _scene_id_from_name(location, scene_name)
        ego_id = str(label.get("agents", {}).get("ego_id", ""))
        if not ego_id:
            continue
        label_start = int(label.get("time_window", {}).get("start_frame", 0))
        label_end = int(label.get("time_window", {}).get("end_frame", label_start))
        init_timestep = max(label_start, history_steps)
        scene_key = (location, scene_id)
        scene_tracks = scene_cache.get(scene_key)
        if scene_tracks is None:
            scene_tracks = _load_scene_tracks(data_dir, location, scene_id)
            scene_cache[scene_key] = scene_tracks
        scene_length = scene_length_cache.get(scene_key)
        if scene_length is None:
            scene_length = _scene_length(scene_tracks)
            scene_length_cache[scene_key] = scene_length
        if not _has_valid_window(
            scene_tracks=scene_tracks,
            ego_id=ego_id,
            init_timestep=init_timestep,
            num_steps=num_steps,
            future_steps=future_steps,
            scene_length=scene_length,
        ):
            continue
        motion = _motion_features(scene_tracks, ego_id, init_timestep, num_steps)
        if motion is None:
            continue
        speed, path_length, displacement = motion
        if np.isfinite(max_original_initial_speed) and speed > max_original_initial_speed:
            continue
        if (
            require_boost_increase
            and _boosted_speed(speed, boost_min_speed, boost_max_speed, boost_scale)
            <= speed
        ):
            continue
        level = _motion_filter_level(
            initial_speed=speed,
            path_length=path_length,
            displacement=displacement,
            strict_min_path_length=strict_min_path_length,
            strict_min_displacement=strict_min_displacement,
            strict_min_initial_speed=strict_min_initial_speed,
            relaxed_min_path_length=relaxed_min_path_length,
            relaxed_min_displacement=relaxed_min_displacement,
            relaxed_min_initial_speed=relaxed_min_initial_speed,
        )
        if level is None:
            continue
        candidate = NarrowCandidate(
            label=label,
            intersection=intersection,
            location=location,
            scene_name=scene_name,
            scene_id=scene_id,
            scene_index=scene_index_cache[location][scene_name],
            agent_id=ego_id,
            init_timestep=init_timestep,
            label_start_frame=label_start,
            label_end_frame=label_end,
            original_initial_speed_mps=speed,
            gt_path_length_150m=path_length,
            gt_displacement_150m=displacement,
            motion_filter_level=level,
        )
        key = (location, scene_name, ego_id, init_timestep)
        existing = best_by_window.get(key)
        if existing is None or _narrow_candidate_sort_key(candidate) < _narrow_candidate_sort_key(existing):
            if existing is None:
                accepted_by_intersection[intersection] += 1
            best_by_window[key] = candidate
        if accepted_by_intersection[intersection] >= target_buffer:
            closed_intersections.add(intersection)
            if len(closed_intersections) == len(allowed_intersections):
                break

    by_intersection: Dict[str, List[NarrowCandidate]] = {
        intersection: [] for intersection in intersections
    }
    for candidate in best_by_window.values():
        if candidate.intersection in by_intersection:
            by_intersection[candidate.intersection].append(candidate)
    plans_by_intersection: Dict[str, List[SemanticPlan]] = {}
    for intersection in intersections:
        candidates = by_intersection.get(intersection, [])
        candidates.sort(key=_narrow_candidate_sort_key)
        selected = candidates[:target_pairs_per_intersection]
        plans: List[SemanticPlan] = []
        for rank, candidate in enumerate(selected, start=1):
            plans.extend(
                _plans_from_narrow_candidate(
                    candidate,
                    rank,
                    num_steps,
                    boost_min_speed=boost_min_speed,
                    boost_max_speed=boost_max_speed,
                    boost_scale=boost_scale,
                )
            )
        plans_by_intersection[intersection] = plans
        print(
            f"[semantic-qcnet] type=narrow_feasible_area intersection={intersection} "
            f"candidates={len(candidates)} selected_pairs={len(selected)} "
            f"selected_runs={len(plans)}",
            flush=True,
        )
    return plans_by_intersection


def _load_labels(label_paths: Sequence[Path], scenario_type: str) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    seen = set()
    for path in label_paths:
        if not path.exists():
            print(f"[semantic-qcnet] missing label path={path}", flush=True)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
        for label in items:
            if label.get("semantics", {}).get("type") != scenario_type:
                continue
            scenario_id = str(label.get("scenario_id"))
            if scenario_id in seen:
                continue
            seen.add(scenario_id)
            labels.append(label)
    print(
        f"[semantic-qcnet] loaded labels type={scenario_type} count={len(labels)}",
        flush=True,
    )
    return labels


def _plan_from_label(
    label: Dict[str, Any],
    rank: int,
    *,
    scenario_type: str,
    num_steps: int,
    history_steps: int,
    scene_index: int,
) -> SemanticPlan:
    location = str(label["location"])
    scene_name = str(label["source_scene_name"])
    scene_id = _scene_id_from_name(location, scene_name)
    agent_id = str(label["agents"]["ego_id"])
    label_start = int(label.get("time_window", {}).get("start_frame", 0))
    label_end = int(label.get("time_window", {}).get("end_frame", label_start))
    init_timestep = max(label_start, history_steps)
    scenario_id = str(label["scenario_id"])
    run_id = f"{scenario_id}__ego{agent_id}__t{init_timestep:04d}"
    return SemanticPlan(
        policy=POLICY,
        scenario_type=scenario_type,
        intersection=_canonical_intersection(location),
        location=location,
        scene_name=scene_name,
        scene_id=scene_id,
        scene_index=int(scene_index),
        run_id=run_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        challenger_id=_challenger_id(label),
        candidate_rank=rank,
        semantic_score=_semantic_score(label, scenario_type),
        label_start_frame=label_start,
        label_end_frame=label_end,
        init_timestep=init_timestep,
        num_steps=num_steps,
        semantic_label=label,
    )


def _plans_from_narrow_candidate(
    candidate: NarrowCandidate,
    rank: int,
    num_steps: int,
    *,
    boost_min_speed: float,
    boost_max_speed: float,
    boost_scale: float,
) -> List[SemanticPlan]:
    scenario_id = str(candidate.label["scenario_id"])
    boosted_speed = _boosted_speed(
        candidate.original_initial_speed_mps,
        boost_min_speed,
        boost_max_speed,
        boost_scale,
    )
    pair_id = f"narrow_pair_{candidate.intersection}_{rank:03d}__{scenario_id}"
    plans: List[SemanticPlan] = []
    for speed_mode in SPEED_MODES:
        override = (
            boosted_speed
            if speed_mode == "boosted_speed"
            else candidate.original_initial_speed_mps
        )
        boost_factor = (
            boosted_speed / candidate.original_initial_speed_mps
            if speed_mode == "boosted_speed"
            and candidate.original_initial_speed_mps > 1e-6
            else 1.0
        )
        run_id = (
            f"{scenario_id}__{speed_mode}__ego{candidate.agent_id}"
            f"__t{candidate.init_timestep:04d}"
        )
        plans.append(
            SemanticPlan(
                policy=POLICY,
                scenario_type="narrow_feasible_area",
                intersection=candidate.intersection,
                location=candidate.location,
                scene_name=candidate.scene_name,
                scene_id=candidate.scene_id,
                scene_index=candidate.scene_index,
                run_id=run_id,
                scenario_id=scenario_id,
                agent_id=candidate.agent_id,
                challenger_id=_challenger_id(candidate.label),
                candidate_rank=rank,
                semantic_score=_semantic_score(candidate.label, "narrow_feasible_area"),
                label_start_frame=candidate.label_start_frame,
                label_end_frame=candidate.label_end_frame,
                init_timestep=candidate.init_timestep,
                num_steps=num_steps,
                semantic_label=candidate.label,
                speed_mode=speed_mode,
                pair_id=pair_id,
                original_initial_speed_mps=candidate.original_initial_speed_mps,
                initial_speed_override_mps=override,
                speed_boost_factor=boost_factor,
                gt_path_length_150m=candidate.gt_path_length_150m,
                gt_displacement_150m=candidate.gt_displacement_150m,
                motion_filter_level=candidate.motion_filter_level,
            )
        )
    return plans


def _make_config(
    plan: SemanticPlan,
    data_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> ToolchainConfig:
    return ToolchainConfig(
        dataset=DatasetConfig(
            name="sind",
            location=plan.location,
            data_dir=str(data_dir),
            desired_dt=args.dt,
            use_lanelet2_maps=True,
            agent_types=list(SEMANTIC_AGENT_TYPES),
        ),
        scenario=ScenarioConfig(
            scene_index=plan.scene_index,
            scene_name=plan.scene_name,
            init_timestep=plan.init_timestep,
            num_steps=plan.num_steps,
            ego_selection_strategy="longest_trajectory",
            ego_agent_name=plan.agent_id,
            semantic_label_id=plan.scenario_id,
            semantic_label=plan.semantic_label,
            allow_ego_fallback=False,
        ),
        simulation=SimulationConfig(
            mode="ego_closed_loop",
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            neighbor_radius=args.neighbor_radius,
        ),
        policies=PolicyConfig(
            ego_policy=POLICY,
            non_ego_policy="ground_truth",
            ego={
                "device": args.device,
                "map_radius": args.map_radius,
                "prediction_interval_steps": args.qcnet_prediction_interval_steps,
                "execute_top1_cached_trajectory": True,
                **(
                    {"initial_velocity_override_mps": plan.initial_speed_override_mps}
                    if plan.initial_speed_override_mps is not None
                    else {}
                ),
            },
            non_ego={},
        ),
        checkpoints=CheckpointConfig(
            qcnet_ckpt_path=args.qcnet_ckpt_path,
            qcnet_repo_path=args.qcnet_repo_path,
        ),
        output=OutputConfig(
            project_name=run_dir.name,
            root_dir=str(run_dir.parent),
            save_html=False,
            save_json=True,
            save_csv=False,
        ),
        visualization=VisualizationConfig(enabled=False),
    )


def _dataset_bundle_for_config(
    cfg: ToolchainConfig,
    dataset_cache: Dict[Tuple[Any, ...], Any],
) -> Any:
    key = (
        cfg.dataset.name,
        cfg.dataset.location,
        cfg.dataset.data_dir,
        cfg.dataset.desired_dt,
        cfg.dataset.use_lanelet2_maps,
        cfg.simulation.history_sec,
        cfg.simulation.future_sec,
        cfg.simulation.neighbor_radius,
        cfg.simulation.mode,
        cfg.policies.ego_policy,
        cfg.policies.non_ego_policy,
        tuple(cfg.dataset.agent_types or ()),
        cfg.policies.ego.get("map_radius"),
    )
    bundle = dataset_cache.get(key)
    if bundle is None:
        bundle = load_simulation_dataset(cfg)
        dataset_cache[key] = bundle
    else:
        print(
            f"[semantic-qcnet] reuse dataset location={cfg.dataset.location} "
            f"scenes={len(bundle.scenes)}",
            flush=True,
        )
    return bundle


def _compute_qcnet_command_metrics(result: Any) -> Dict[str, Any]:
    commands = [frame.command or {} for frame in result.frames if frame.is_ego]
    prediction_count = sum(
        1
        for command in commands
        if command.get("policy") == POLICY
        and command.get("used_cached_trajectory") is not True
    )
    cached_count = sum(
        1 for command in commands if command.get("used_cached_trajectory") is True
    )
    intervals = [
        int(command["prediction_interval_steps"])
        for command in commands
        if command.get("prediction_interval_steps") not in (None, "")
    ]
    devices = [
        str(command["device"])
        for command in commands
        if command.get("device") not in (None, "")
    ]
    return {
        "qcnet_prediction_count": int(prediction_count),
        "qcnet_cached_trajectory_count": int(cached_count),
        "prediction_interval_steps": intervals[0] if intervals else np.nan,
        "device": devices[0] if devices else "",
    }


def _base_manifest_row(plan: SemanticPlan, run_dir: Path) -> Dict[str, Any]:
    return {
        "policy": plan.policy,
        "scenario_type": plan.scenario_type,
        "speed_mode": plan.speed_mode,
        "pair_id": plan.pair_id,
        "intersection": plan.intersection,
        "location": plan.location,
        "scene_name": plan.scene_name,
        "scene_id": plan.scene_id,
        "scene_index": plan.scene_index,
        "run_id": plan.run_id,
        "scenario_id": plan.scenario_id,
        "agent_id": plan.agent_id,
        "challenger_id": plan.challenger_id,
        "candidate_rank": plan.candidate_rank,
        "semantic_score": plan.semantic_score,
        "label_start_frame": plan.label_start_frame,
        "label_end_frame": plan.label_end_frame,
        "init_timestep": plan.init_timestep,
        "num_steps_requested": plan.num_steps,
        "original_initial_speed_mps": plan.original_initial_speed_mps,
        "initial_speed_override_mps": plan.initial_speed_override_mps,
        "speed_boost_factor": plan.speed_boost_factor,
        "gt_path_length_150m": plan.gt_path_length_150m,
        "gt_displacement_150m": plan.gt_displacement_150m,
        "motion_filter_level": plan.motion_filter_level,
        "status": "pending",
        "run_dir": str(run_dir),
        "trajectory_log": str(run_dir / "trajectory_log.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "error": "",
        "elapsed_s": "",
    }


def _write_config(path: Path, cfg: ToolchainConfig) -> None:
    payload = {
        "dataset": asdict(cfg.dataset),
        "scenario": asdict(cfg.scenario),
        "simulation": asdict(cfg.simulation),
        "policies": asdict(cfg.policies),
        "checkpoints": asdict(cfg.checkpoints),
        "output": asdict(cfg.output),
        "visualization": asdict(cfg.visualization),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_outputs(policy_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [_with_metric_anomaly_fields(dict(row)) for row in rows]
    _write_manifest(policy_dir / "run_manifest.csv", rows)
    _write_summaries(policy_dir, rows)


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _manifest_fields(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item.get("scenario_type", "")),
                str(item.get("intersection", "")),
                str(item.get("speed_mode", "")),
                _sort_int(item.get("candidate_rank")),
                str(item.get("run_id", "")),
            ),
        ):
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_summaries(policy_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return
    by_location = [_summary_row(location, group) for location, group in df.groupby("location")]
    pd.DataFrame.from_records(by_location).to_csv(
        policy_dir / "summary_by_location.csv", index=False
    )
    by_intersection = [
        _summary_row(intersection, group)
        for intersection, group in df.groupby("intersection")
    ]
    by_intersection_df = pd.DataFrame.from_records(by_intersection)
    if "location" in by_intersection_df.columns:
        by_intersection_df = by_intersection_df.rename(columns={"location": "intersection"})
    by_intersection_df.to_csv(policy_dir / "summary_by_intersection.csv", index=False)
    if "speed_mode" in df.columns and df["speed_mode"].notna().any():
        by_speed_mode = [
            _summary_row(speed_mode, group)
            for speed_mode, group in df.groupby("speed_mode")
        ]
        pd.DataFrame.from_records(by_speed_mode).to_csv(
            policy_dir / "summary_by_speed_mode.csv", index=False
        )
        by_intersection_speed = [
            _summary_row(f"{intersection}:{speed_mode}", group)
            for (intersection, speed_mode), group in df.groupby(["intersection", "speed_mode"])
        ]
        pd.DataFrame.from_records(by_intersection_speed).to_csv(
            policy_dir / "summary_by_intersection_and_speed_mode.csv", index=False
        )
    pd.DataFrame.from_records([_summary_row("ALL", df)]).to_csv(
        policy_dir / "summary_total.csv", index=False
    )


def _summary_row(label: str, df: pd.DataFrame) -> Dict[str, Any]:
    completed = df[df["status"] == "completed"].copy()
    metric_anomaly_mask = completed.apply(_is_metric_anomaly_row, axis=1)
    metric_valid = completed.copy()
    row: Dict[str, Any] = {
        "location": label,
        "planned_runs": int(len(df)),
        "completed_runs": int(len(completed)),
        "metric_valid_runs": int(len(metric_valid)),
        "metric_anomaly_runs": int(metric_anomaly_mask.sum()),
        "metric_anomaly_rate": (
            float(metric_anomaly_mask.mean()) if len(completed) else np.nan
        ),
        "failed_runs": int((df["status"] == "failed").sum()),
        "skipped_runs": int((df["status"] == "skipped").sum()),
    }
    for metric in (
        "ADE",
        "FDE",
        "MinTTC",
        "AveTTC",
        "MRD",
        "ARD",
        "qcnet_prediction_count",
        "qcnet_cached_trajectory_count",
    ):
        values = _numeric_series(metric_valid, metric)
        row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
    for metric in (
        "collision",
        "offroad",
        "violation",
        "wrong_way_violation",
        "red_light_violation",
        "lane_direction_rule_violation",
        "signal_observable",
    ):
        row[f"{metric}_rate"] = _bool_rate(metric_valid.get(metric))
    return row


def _manifest_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = list(MANIFEST_BASE_FIELDS)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _with_metric_anomaly_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("status") != "completed":
        row.setdefault("metric_anomaly", "")
        row.setdefault("metric_anomaly_reason", "")
        return row
    reasons: List[str] = []
    ade = _finite_float(row.get("ADE"))
    fde = _finite_float(row.get("FDE"))
    if ade is not None and ade > 10.0:
        reasons.append("ADE>10")
    if fde is not None and fde > 50.0:
        reasons.append("FDE>50")
    row["metric_anomaly"] = bool(reasons)
    row["metric_anomaly_reason"] = ";".join(reasons)
    return row


def _label_sort_key(label: Mapping[str, Any], scenario_type: str) -> Tuple[Any, ...]:
    window = label.get("time_window", {})
    scene_name = str(label.get("source_scene_name", ""))
    start_frame = int(window.get("start_frame", 0))
    ego_id = str(label.get("agents", {}).get("ego_id", ""))
    scenario_id = str(label.get("scenario_id", ""))
    if scenario_type == "mprttc":
        return (
            _semantic_score(label, scenario_type),
            start_frame,
            str(label.get("location", "")),
            scene_name,
            ego_id,
            scenario_id,
        )
    if scenario_type == "visual_shielding":
        return (
            -_semantic_score(label, scenario_type),
            scene_name,
            start_frame,
            ego_id,
            scenario_id,
        )
    return (
        _semantic_score(label, scenario_type),
        start_frame,
        str(label.get("location", "")),
        scene_name,
        ego_id,
        scenario_id,
    )


def _semantic_score(label: Mapping[str, Any], scenario_type: str) -> float:
    semantics = label.get("semantics", {})
    if scenario_type == "mprttc":
        value = semantics.get("min_mprttc")
        return float("inf") if value is None else float(value)
    if scenario_type == "visual_shielding":
        value = semantics.get("shielding_frame_length")
        if value is None:
            window = label.get("time_window", {})
            value = int(window.get("end_frame", 0)) - int(window.get("start_frame", 0))
        return float(value)
    value = semantics.get("min_max_area")
    return float("inf") if value is None else float(value)


def _challenger_id(label: Mapping[str, Any]) -> Optional[str]:
    agents = label.get("agents", {})
    for key in ("challenger_id", "shielding_id", "shielded_id"):
        value = agents.get(key)
        if value is not None:
            return str(value)
    return None


def _canonical_intersection(location: str) -> str:
    return "xa" if str(location) in {"xa", "xasl"} else str(location)


def _apply_shard(plans: Sequence[SemanticPlan], shard_index: int, num_shards: int) -> List[SemanticPlan]:
    if num_shards <= 1:
        return list(plans)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")
    return [plan for idx, plan in enumerate(plans) if idx % num_shards == shard_index]


def _run_dir(policy_dir: Path, plan: SemanticPlan) -> Path:
    return (
        policy_dir
        / "runs"
        / plan.scenario_type
        / plan.intersection
        / plan.location
        / _safe_path_name(plan.scene_name)
        / _safe_path_name(plan.run_id)
    )


def _load_scene_tracks(data_dir: Path, location: str, scene_id: str) -> Mapping[str, Any]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)[scene_id]


def _scene_index_map(data_dir: Path, location: str) -> Dict[str, int]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        tp_info = pickle.load(handle)
    mapping: Dict[str, int] = {}
    scene_index = 0
    for scene_id_raw, scene_tracks in tp_info.items():
        scene_id = str(scene_id_raw)
        scene_name = f"{location}_{scene_id}"
        if _scene_length(scene_tracks) <= 1:
            scene_index += 1
            continue
        mapping[scene_name] = scene_index
        scene_index += 1
    return mapping


def _scene_length(scene_tracks: Mapping[str, Any]) -> int:
    max_frame = 0
    for tp_data in scene_tracks.values():
        state = tp_data.get("State")
        if state is not None and not state.empty and "frame_id" in state.columns:
            max_frame = max(max_frame, int(state["frame_id"].max()))
    return max_frame + 1


def _has_history_window(
    scene_tracks: Mapping[str, Any],
    ego_id: str,
    init_timestep: int,
    history_steps: int,
) -> bool:
    tp_data = next((value for key, value in scene_tracks.items() if str(key) == ego_id), None)
    if tp_data is None:
        return False
    state = tp_data.get("State")
    if state is None or state.empty or "frame_id" not in state.columns:
        return False
    required_first = max(0, int(init_timestep) - int(history_steps))
    return int(state["frame_id"].min()) <= required_first


def _scene_id_from_name(location: str, scene_name: str) -> str:
    prefix = f"{location}_"
    return scene_name[len(prefix) :] if scene_name.startswith(prefix) else scene_name


def _sort_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _cleanup_torch_cuda() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
