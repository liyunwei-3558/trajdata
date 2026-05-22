from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
from Simulation_test_toolchain.core.fast_risk_idm import (
    load_scene_tracks as load_fast_scene_tracks,
    run_fast_risk_idm_simulation,
)
from Simulation_test_toolchain.core.records import SimulationResult
from Simulation_test_toolchain.core.runner import load_simulation_dataset, run_simulation
from Simulation_test_toolchain.run_raw_open_loop_trajectory_batch import (
    _bool_rate,
    _canonical_intersection,
    _csv_value,
    _is_metric_anomaly_row,
    _load_manifest,
    _manifest_key,
    _numeric_series,
    _safe_path_name,
)


DEFAULT_LABEL_PATH = "datasets/SinD_dataset/Semantic_labels/scenarios.json"
DEFAULT_INTERSECTIONS = ("cc", "tj", "cqIR", "cqNR", "cqR", "xa")
DEFAULT_ASAPRL_INTERSECTIONS = ("cc", "cqIR", "cqNR", "cqR", "xa")
POLICY = "risk_idm"
POLICY_OUTPUT_DIR = "raw_open_loop_risk_idm"
POLICY_OUTPUT_DIRS = {
    "risk_idm": "raw_open_loop_risk_idm",
    "asaprl": "raw_open_loop_asaprl",
}
SCENARIO_TYPE = "narrow_feasible_area"
SPEED_MODES = ("original_speed", "boosted_speed")
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
    "candidate_rank",
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
class NarrowPlan:
    policy: str
    scenario_type: str
    speed_mode: str
    pair_id: str
    intersection: str
    location: str
    scene_name: str
    scene_id: str
    scene_index: int
    run_id: str
    scenario_id: str
    agent_id: str
    candidate_rank: int
    label_start_frame: int
    label_end_frame: int
    init_timestep: int
    num_steps: int
    original_initial_speed_mps: float
    initial_speed_override_mps: Optional[float]
    speed_boost_factor: float
    gt_path_length_150m: float
    gt_displacement_150m: float
    motion_filter_level: str
    semantic_label: Dict[str, Any]


@dataclass(frozen=True)
class Candidate:
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
    output_root = args.output_root.expanduser()
    print(
        f"[narrow-batch] init target_pairs={args.target_pairs} num_steps={args.num_steps} "
        f"label_path={args.label_path} data_dir={data_dir}",
        flush=True,
    )
    policy_dir = output_root / POLICY_OUTPUT_DIRS[args.policy]
    if args.num_shards > 1:
        policy_dir = policy_dir / f"shard_{args.shard_index:02d}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    base_candidates = select_candidates(
        label_path=args.label_path,
        data_dir=data_dir,
        intersections=tuple(args.intersections),
        target_pairs=args.target_pairs,
        target_pairs_per_intersection=args.target_pairs_per_intersection,
        candidate_buffer_factor=args.candidate_buffer_factor,
        num_steps=args.num_steps,
        history_steps=int(round(args.history_sec / args.dt)),
        future_steps=int(np.ceil(args.future_sec / args.dt)),
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
    )
    plans = build_ab_plans(
        base_candidates,
        args.num_steps,
        policy=args.policy,
        boost_min_speed=args.boost_min_speed,
        boost_max_speed=args.boost_max_speed,
        boost_scale=args.boost_scale,
    )
    print(
        f"[narrow-batch] policy={args.policy} pairs={len(base_candidates)} runs={len(plans)} "
        f"output={policy_dir}",
        flush=True,
    )

    metric_context = MetricContext(data_dir=data_dir, dt=args.dt)
    dataset_cache: Dict[Tuple[Any, ...], Any] = {}
    policy_cache: Dict[Tuple[Any, ...], Any] = {}
    fast_scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    rows_by_key = _load_manifest(policy_dir / "run_manifest.csv")
    if args.force:
        rows_by_key = {}

    completed = 0 if args.force else _completed_count(rows_by_key.values())
    plans = _apply_shard(plans, args.shard_index, args.num_shards)
    print(
        f"[narrow-batch] start completed={completed} planned_this_shard={len(plans)}",
        flush=True,
    )
    for plan in plans:
        key = _manifest_key(plan.policy, plan.location, plan.run_id)
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
                f"[narrow-batch] run mode={plan.speed_mode} rank={plan.candidate_rank} "
                f"scenario={plan.scenario_id} location={plan.location} "
                f"scene={plan.scene_name} ego={plan.agent_id} init={plan.init_timestep} "
                f"v0={plan.original_initial_speed_mps:.3f} "
                f"override={_fmt_optional(plan.initial_speed_override_mps)}",
                flush=True,
            )
            if args.fast_risk_idm and plan.policy == "risk_idm":
                scene_tracks = load_fast_scene_tracks(
                    data_dir,
                    plan.location,
                    plan.scene_id,
                    cache=fast_scene_cache,
                )
                result = run_fast_risk_idm_simulation(cfg, scene_tracks=scene_tracks)
            else:
                result = run_simulation(
                    cfg,
                    dataset_bundle=_dataset_bundle_for_config(cfg, dataset_cache),
                    policy_cache=policy_cache,
                )
            result.metadata.update(
                {
                    "experiment_mode": "dataset_open_loop_narrow_feasible_area",
                    "open_loop_definition": (
                        "non-ego agents replay ground truth; ego is controlled by policy"
                    ),
                    "runner_mode": cfg.simulation.mode,
                    "ego_policy": plan.policy,
                    "scenario_type": SCENARIO_TYPE,
                    "speed_mode": plan.speed_mode,
                    "pair_id": plan.pair_id,
                    "run_id": plan.run_id,
                    "semantic_label_id": plan.scenario_id,
                    "semantic_label": plan.semantic_label,
                    "candidate_rank": plan.candidate_rank,
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
            metrics.update(_compute_command_metrics(result))
            save_metrics(run_dir / "metrics.json", result.metadata, metrics)
            row.update(metrics)
            row["status"] = "completed"
            completed += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(f"[narrow-batch] failed scenario={plan.scenario_id}: {exc}", flush=True)
        finally:
            row["elapsed_s"] = round(time.time() - started, 3)
            rows_by_key[key] = row
            _write_outputs(policy_dir, rows_by_key.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired policy tests on narrow feasible area labels."
    )
    parser.add_argument("--policy", choices=sorted(POLICY_OUTPUT_DIRS), default=POLICY)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--label-path", type=Path, default=Path(DEFAULT_LABEL_PATH))
    parser.add_argument("--intersections", nargs="+", default=list(DEFAULT_INTERSECTIONS))
    parser.add_argument("--target-pairs", type=int, default=100)
    parser.add_argument("--target-pairs-per-intersection", type=int, default=None)
    parser.add_argument("--candidate-buffer-factor", type=float, default=2.0)
    parser.add_argument("--candidate-progress-interval", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--desired-velocity", type=float, default=8.0)
    parser.add_argument("--max-acceleration", type=float, default=3.0)
    parser.add_argument("--min-acceleration", type=float, default=-5.0)
    parser.add_argument("--inference-interval-steps", type=int, default=5)
    parser.add_argument("--asaprl-target-speed", type=float, default=7.5)
    parser.add_argument("--asaprl-horizon", type=float, default=3.0)
    parser.add_argument(
        "--asaprl-ckpt-path",
        type=str,
        default="Simulation_test_toolchain/checkpoints/ckpt_best_iter497000_step417741440.pth.tar",
    )
    parser.add_argument("--asaprl-device", type=str, default="cuda")
    parser.add_argument("--asaprl-action-lat-scale", type=float, default=1.0)
    parser.add_argument("--asaprl-action-yaw-scale", type=float, default=2.0)
    parser.add_argument("--asaprl-action-speed-scale", type=float, default=3.0)
    parser.add_argument("--asaprl-observation-px-per-m", type=float, default=3.0)
    parser.add_argument("--asaprl-reference-heading-blend", type=float, default=0.8)
    parser.add_argument("--asaprl-reference-speed-blend", type=float, default=0.0)
    parser.add_argument("--asaprl-max-yaw-rate", type=float, default=0.3)
    parser.add_argument("--asaprl-max-speed", type=float, default=8.0)
    parser.add_argument("--asaprl-no-follow-reference-direction", action="store_true")
    parser.add_argument("--strict-min-path-length", type=float, default=8.0)
    parser.add_argument("--strict-min-displacement", type=float, default=5.0)
    parser.add_argument("--strict-min-initial-speed", type=float, default=0.5)
    parser.add_argument("--relaxed-min-path-length", type=float, default=4.0)
    parser.add_argument("--relaxed-min-displacement", type=float, default=2.0)
    parser.add_argument("--relaxed-min-initial-speed", type=float, default=0.2)
    parser.add_argument("--max-original-initial-speed", type=float, default=8.5)
    parser.add_argument("--allow-non-increasing-boost", action="store_true")
    parser.add_argument("--boost-min-speed", type=float, default=9.0)
    parser.add_argument("--boost-max-speed", type=float, default=17.0)
    parser.add_argument("--boost-scale", type=float, default=2.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--fast-risk-idm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a direct tp_info-based RiskIDM simulator for batch throughput. "
            "Disable to route through trajdata SimulationScene."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        suffix = (
            "narrow_feasible_asaprl_per_intersection_100pairs"
            if args.policy == "asaprl"
            else "narrow_feasible_riskidm_ab_100"
        )
        args.output_root = Path("Simulation_test_toolchain/batch_outputs") / suffix
    if (
        args.policy == "asaprl"
        and args.intersections == list(DEFAULT_INTERSECTIONS)
    ):
        args.intersections = list(DEFAULT_ASAPRL_INTERSECTIONS)
    return args


def select_candidates(
    *,
    label_path: Path,
    data_dir: Path,
    intersections: Sequence[str],
    target_pairs: int,
    target_pairs_per_intersection: Optional[int],
    candidate_buffer_factor: float,
    num_steps: int,
    history_steps: int,
    future_steps: int,
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
) -> List[Candidate]:
    labels = _load_narrow_labels(label_path)
    scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    scene_length_cache: Dict[Tuple[str, str], int] = {}
    scene_index_cache: Dict[str, Dict[str, int]] = {}
    best_by_window: Dict[Tuple[str, str, str, int], Candidate] = {}
    allowed_intersections = set(intersections)
    target_by_intersection = _target_candidates_by_intersection(
        target_pairs=target_pairs,
        intersections=intersections,
        buffer_factor=candidate_buffer_factor,
        target_pairs_per_intersection=target_pairs_per_intersection,
    )
    accepted_by_intersection: Dict[str, int] = {key: 0 for key in intersections}
    closed_intersections: Set[str] = set()
    scanned = 0

    for label in labels:
        scanned += 1
        if progress_interval > 0 and scanned % progress_interval == 0:
            print(
                f"[narrow-batch] candidate scan scanned={scanned} "
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
        agent_id = str(label.get("agents", {}).get("ego_id", ""))
        if not agent_id:
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
            ego_id=agent_id,
            init_timestep=init_timestep,
            num_steps=num_steps,
            future_steps=future_steps,
            scene_length=scene_length,
        ):
            continue
        motion = _motion_features(scene_tracks, agent_id, init_timestep, num_steps)
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
        candidate = Candidate(
            label=label,
            intersection=intersection,
            location=location,
            scene_name=scene_name,
            scene_id=scene_id,
            scene_index=scene_index_cache[location][scene_name],
            agent_id=agent_id,
            init_timestep=init_timestep,
            label_start_frame=label_start,
            label_end_frame=label_end,
            original_initial_speed_mps=speed,
            gt_path_length_150m=path_length,
            gt_displacement_150m=displacement,
            motion_filter_level=level,
        )
        key = (location, scene_name, agent_id, init_timestep)
        existing = best_by_window.get(key)
        if existing is None or _candidate_sort_key(candidate) < _candidate_sort_key(existing):
            if existing is None:
                accepted_by_intersection[intersection] += 1
            best_by_window[key] = candidate
        if accepted_by_intersection[intersection] >= target_by_intersection[intersection]:
            closed_intersections.add(intersection)
            print(
                f"[narrow-batch] candidate buffer filled intersection={intersection} "
                f"accepted={accepted_by_intersection[intersection]} scanned={scanned}",
                flush=True,
            )
            if len(closed_intersections) == len(allowed_intersections):
                break

    selected = _balanced_select(
        list(best_by_window.values()),
        intersections=intersections,
        target=target_pairs,
        target_per_intersection=target_pairs_per_intersection,
    )
    for rank, candidate in enumerate(selected, start=1):
        print(
            f"[narrow-batch] selected rank={rank} intersection={candidate.intersection} "
            f"scenario={candidate.label.get('scenario_id')} "
            f"path={candidate.gt_path_length_150m:.2f} "
            f"disp={candidate.gt_displacement_150m:.2f} "
            f"v0={candidate.original_initial_speed_mps:.2f} "
            f"filter={candidate.motion_filter_level}",
            flush=True,
        )
    return selected


def build_ab_plans(
    candidates: Sequence[Candidate],
    num_steps: int,
    *,
    policy: str = POLICY,
    boost_min_speed: float,
    boost_max_speed: float,
    boost_scale: float,
) -> List[NarrowPlan]:
    plans: List[NarrowPlan] = []
    for rank, candidate in enumerate(candidates, start=1):
        scenario_id = str(candidate.label["scenario_id"])
        pair_id = f"narrow_pair_{rank:03d}__{scenario_id}"
        boosted = _boosted_speed(
            candidate.original_initial_speed_mps,
            boost_min_speed,
            boost_max_speed,
            boost_scale,
        )
        for speed_mode in SPEED_MODES:
            override = (
                boosted
                if speed_mode == "boosted_speed"
                else candidate.original_initial_speed_mps
            )
            boost_factor = (
                boosted / candidate.original_initial_speed_mps
                if speed_mode == "boosted_speed"
                and candidate.original_initial_speed_mps > 1e-6
                else 1.0
            )
            run_id = (
                f"{scenario_id}__{speed_mode}__ego{candidate.agent_id}"
                f"__t{candidate.init_timestep:04d}"
            )
            plans.append(
                NarrowPlan(
                    policy=policy,
                    scenario_type=SCENARIO_TYPE,
                    speed_mode=speed_mode,
                    pair_id=pair_id,
                    intersection=candidate.intersection,
                    location=candidate.location,
                    scene_name=candidate.scene_name,
                    scene_id=candidate.scene_id,
                    scene_index=candidate.scene_index,
                    run_id=run_id,
                    scenario_id=scenario_id,
                    agent_id=candidate.agent_id,
                    candidate_rank=rank,
                    label_start_frame=candidate.label_start_frame,
                    label_end_frame=candidate.label_end_frame,
                    init_timestep=candidate.init_timestep,
                    num_steps=num_steps,
                    original_initial_speed_mps=candidate.original_initial_speed_mps,
                    initial_speed_override_mps=override,
                    speed_boost_factor=boost_factor,
                    gt_path_length_150m=candidate.gt_path_length_150m,
                    gt_displacement_150m=candidate.gt_displacement_150m,
                    motion_filter_level=candidate.motion_filter_level,
                    semantic_label=candidate.label,
                )
            )
    return plans


def _make_config(
    plan: NarrowPlan,
    data_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> ToolchainConfig:
    if plan.policy == "asaprl":
        ego_params: Dict[str, Any] = {
            "target_speed": args.asaprl_target_speed,
            "horizon": args.asaprl_horizon,
            "ckpt_path": args.asaprl_ckpt_path,
            "device": args.asaprl_device,
            "use_risk_idm": True,
            "neighbor_radius": args.neighbor_radius,
            "inference_interval_steps": args.inference_interval_steps,
            "require_raster_map": True,
            "action_lat_scale": args.asaprl_action_lat_scale,
            "action_yaw_scale": args.asaprl_action_yaw_scale,
            "action_speed_scale": args.asaprl_action_speed_scale,
            "observation_px_per_m": args.asaprl_observation_px_per_m,
            "reference_heading_blend": args.asaprl_reference_heading_blend,
            "reference_speed_blend": args.asaprl_reference_speed_blend,
            "max_yaw_rate": args.asaprl_max_yaw_rate,
            "max_speed": args.asaprl_max_speed,
            "follow_reference_direction": not args.asaprl_no_follow_reference_direction,
        }
    else:
        ego_params = {
            "desired_velocity": args.desired_velocity,
            "max_acceleration": args.max_acceleration,
            "min_acceleration": args.min_acceleration,
            "neighbor_radius": args.neighbor_radius,
            "inference_interval_steps": args.inference_interval_steps,
        }
    if plan.initial_speed_override_mps is not None:
        ego_params["initial_velocity_override_mps"] = plan.initial_speed_override_mps
    return ToolchainConfig(
        dataset=DatasetConfig(
            name="sind",
            location=plan.location,
            data_dir=str(data_dir),
            desired_dt=args.dt,
            use_lanelet2_maps=True,
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
        ),
        simulation=SimulationConfig(
            mode="ego_closed_loop",
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            neighbor_radius=args.neighbor_radius,
        ),
        policies=PolicyConfig(
            ego_policy=plan.policy,
            non_ego_policy="ground_truth",
            ego=ego_params,
            non_ego={},
        ),
        checkpoints=CheckpointConfig(
            asaprl_ckpt_path=(
                args.asaprl_ckpt_path if plan.policy == "asaprl" else None
            ),
            asaprl_device=args.asaprl_device if plan.policy == "asaprl" else None,
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
        bool(cfg.policies.ego.get("require_raster_map", False)),
        bool(cfg.policies.non_ego.get("require_raster_map", False)),
    )
    bundle = dataset_cache.get(key)
    if bundle is None:
        bundle = load_simulation_dataset(cfg)
        dataset_cache[key] = bundle
    else:
        print(
            f"[narrow-batch] reuse dataset location={cfg.dataset.location} "
            f"scenes={len(bundle.scenes)}",
            flush=True,
        )
    return bundle


def _compute_command_metrics(result: SimulationResult) -> Dict[str, Any]:
    ego_frames = [frame for frame in result.frames if frame.is_ego]
    accelerations: List[float] = []
    policy_speeds: List[float] = []
    ttc_values: List[float] = []
    cached_count = 0
    inferred_count = 0
    for frame in ego_frames:
        command = frame.command or {}
        accel = _finite_float(command.get("acceleration"))
        if accel is not None:
            accelerations.append(accel)
        speed = _finite_float(command.get("velocity"))
        if speed is not None:
            policy_speeds.append(speed)
        ttc = _finite_float(command.get("min_ttc"))
        if ttc is not None:
            ttc_values.append(ttc)
        if command.get("used_cached_control") is True:
            cached_count += 1
        elif command.get("used_cached_control") is False:
            inferred_count += 1

    first_policy_speed = policy_speeds[0] if policy_speeds else np.nan
    min_policy_speed = min(policy_speeds) if policy_speeds else np.nan
    min_acceleration = min(accelerations) if accelerations else np.nan
    max_deceleration = max(0.0, -min_acceleration) if accelerations else np.nan
    return {
        "min_acceleration": min_acceleration,
        "mean_acceleration": float(np.mean(accelerations)) if accelerations else np.nan,
        "max_deceleration": max_deceleration,
        "hard_brake_frame_count": int(sum(1 for value in accelerations if value <= -3.0)),
        "very_hard_brake_frame_count": int(
            sum(1 for value in accelerations if value <= -4.5)
        ),
        "min_policy_speed": min_policy_speed,
        "final_policy_speed": policy_speeds[-1] if policy_speeds else np.nan,
        "speed_drop_mps": (
            first_policy_speed - min_policy_speed if policy_speeds else np.nan
        ),
        "command_min_ttc": min(ttc_values) if ttc_values else np.nan,
        "policy_inference_count": int(inferred_count),
        "policy_cached_control_count": int(cached_count),
    }


def _base_manifest_row(plan: NarrowPlan, run_dir: Path) -> Dict[str, Any]:
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
        "candidate_rank": plan.candidate_rank,
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
                _sort_int(item.get("candidate_rank")),
                str(item.get("speed_mode", "")),
                str(item.get("run_id", "")),
            ),
        ):
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_summaries(policy_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return
    by_speed = [_summary_row(mode, group) for mode, group in df.groupby("speed_mode")]
    pd.DataFrame.from_records(by_speed).to_csv(
        policy_dir / "summary_by_speed_mode.csv", index=False
    )
    by_intersection_speed = [
        _summary_row(f"{intersection}:{mode}", group)
        for (intersection, mode), group in df.groupby(["intersection", "speed_mode"])
    ]
    pd.DataFrame.from_records(by_intersection_speed).to_csv(
        policy_dir / "summary_by_intersection_and_speed_mode.csv", index=False
    )
    pd.DataFrame.from_records([_summary_row("ALL", df)]).to_csv(
        policy_dir / "summary_total.csv", index=False
    )
    paired = _paired_comparison(df)
    if not paired.empty:
        paired.to_csv(policy_dir / "paired_ab_comparison.csv", index=False)


def _summary_row(label: str, df: pd.DataFrame) -> Dict[str, Any]:
    completed = df[df["status"] == "completed"].copy()
    metric_anomaly_mask = completed.apply(_is_metric_anomaly_row, axis=1)
    metric_valid = completed.copy()
    row: Dict[str, Any] = {
        "group": label,
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
        "min_acceleration",
        "max_deceleration",
        "hard_brake_frame_count",
        "min_policy_speed",
        "speed_drop_mps",
        "command_min_ttc",
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


def _paired_comparison(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    metrics = (
        "ADE",
        "FDE",
        "MinTTC",
        "collision",
        "offroad",
        "max_deceleration",
        "hard_brake_frame_count",
        "min_policy_speed",
        "speed_drop_mps",
        "command_min_ttc",
    )
    for pair_id, group in completed.groupby("pair_id"):
        by_mode = {str(row.speed_mode): row for row in group.itertuples(index=False)}
        if not all(mode in by_mode for mode in SPEED_MODES):
            continue
        original = by_mode["original_speed"]
        boosted = by_mode["boosted_speed"]
        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "candidate_rank": getattr(original, "candidate_rank", ""),
            "intersection": getattr(original, "intersection", ""),
            "location": getattr(original, "location", ""),
            "scene_name": getattr(original, "scene_name", ""),
            "scenario_id": getattr(original, "scenario_id", ""),
            "agent_id": getattr(original, "agent_id", ""),
            "original_initial_speed_mps": getattr(
                original, "original_initial_speed_mps", np.nan
            ),
            "boosted_initial_speed_mps": getattr(
                boosted, "initial_speed_override_mps", np.nan
            ),
        }
        for metric in metrics:
            orig_value = getattr(original, metric, np.nan)
            boost_value = getattr(boosted, metric, np.nan)
            row[f"original_{metric}"] = orig_value
            row[f"boosted_{metric}"] = boost_value
            row[f"delta_{metric}"] = _delta(boost_value, orig_value)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _manifest_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = list(MANIFEST_BASE_FIELDS)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _load_narrow_labels(label_path: Path) -> List[Dict[str, Any]]:
    print(f"[narrow-batch] loading labels path={label_path}", flush=True)
    raw = json.loads(label_path.read_text(encoding="utf-8"))
    items = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
    labels: List[Dict[str, Any]] = []
    seen = set()
    for label in items:
        if label.get("semantics", {}).get("type") != SCENARIO_TYPE:
            continue
        scenario_id = str(label.get("scenario_id"))
        if scenario_id in seen:
            continue
        seen.add(scenario_id)
        labels.append(label)
    print(f"[narrow-batch] loaded narrow labels={len(labels)}", flush=True)
    return labels


def _balanced_select(
    candidates: Sequence[Candidate],
    *,
    intersections: Sequence[str],
    target: int,
    target_per_intersection: Optional[int],
) -> List[Candidate]:
    by_intersection: Dict[str, List[Candidate]] = {key: [] for key in intersections}
    for candidate in candidates:
        if candidate.intersection in by_intersection:
            by_intersection[candidate.intersection].append(candidate)
    for values in by_intersection.values():
        values.sort(key=_candidate_sort_key)

    if target_per_intersection is not None:
        selected: List[Candidate] = []
        for intersection in intersections:
            selected.extend(by_intersection.get(intersection, [])[:target_per_intersection])
        selected.sort(key=lambda item: (item.intersection, _candidate_sort_key(item)))
        return selected

    base_quota = max(1, target // max(1, len(intersections)))
    selected: List[Candidate] = []
    selected_keys = set()
    for intersection in intersections:
        for candidate in by_intersection.get(intersection, [])[:base_quota]:
            selected.append(candidate)
            selected_keys.add(_candidate_unique_key(candidate))

    if len(selected) < target:
        remainder = sorted(candidates, key=_candidate_sort_key)
        for candidate in remainder:
            key = _candidate_unique_key(candidate)
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= target:
                break
    selected.sort(key=_candidate_sort_key)
    return selected[:target]


def _target_candidates_by_intersection(
    *,
    target_pairs: int,
    intersections: Sequence[str],
    buffer_factor: float,
    target_pairs_per_intersection: Optional[int],
) -> Dict[str, int]:
    count = max(1, len(intersections))
    target = (
        int(target_pairs_per_intersection)
        if target_pairs_per_intersection is not None
        else int(np.ceil(target_pairs / count))
    )
    per_intersection = int(np.ceil(target * max(1.0, buffer_factor)))
    return {intersection: per_intersection for intersection in intersections}


def _candidate_sort_key(candidate: Candidate) -> Tuple[Any, ...]:
    semantics = candidate.label.get("semantics", {})
    min_max_area = semantics.get("min_max_area")
    min_max_area_sort = float(min_max_area) if min_max_area is not None else float("inf")
    filter_rank = 0 if candidate.motion_filter_level == "strict" else 1
    speed_target = 4.5
    return (
        filter_rank,
        abs(candidate.original_initial_speed_mps - speed_target),
        min_max_area_sort,
        -candidate.gt_path_length_150m,
        -candidate.gt_displacement_150m,
        candidate.location,
        candidate.scene_name,
        candidate.agent_id,
        candidate.init_timestep,
        str(candidate.label.get("scenario_id", "")),
    )


def _candidate_unique_key(candidate: Candidate) -> Tuple[str, str, str, int]:
    return (
        candidate.location,
        candidate.scene_name,
        candidate.agent_id,
        candidate.init_timestep,
    )


def _motion_filter_level(
    *,
    initial_speed: float,
    path_length: float,
    displacement: float,
    strict_min_path_length: float,
    strict_min_displacement: float,
    strict_min_initial_speed: float,
    relaxed_min_path_length: float,
    relaxed_min_displacement: float,
    relaxed_min_initial_speed: float,
) -> Optional[str]:
    if (
        path_length >= strict_min_path_length
        and displacement >= strict_min_displacement
        and initial_speed >= strict_min_initial_speed
    ):
        return "strict"
    if (
        path_length >= relaxed_min_path_length
        and displacement >= relaxed_min_displacement
        and initial_speed >= relaxed_min_initial_speed
    ):
        return "relaxed"
    return None


def _motion_features(
    scene_tracks: Mapping[str, Any],
    ego_id: str,
    init_timestep: int,
    num_steps: int,
) -> Optional[Tuple[float, float, float]]:
    tp_data = next((value for key, value in scene_tracks.items() if str(key) == ego_id), None)
    if tp_data is None:
        return None
    state = tp_data.get("State")
    if state is None or state.empty:
        return None
    window = state[
        (state["frame_id"] >= init_timestep)
        & (state["frame_id"] <= init_timestep + num_steps)
    ].sort_values("frame_id")
    if len(window) < num_steps + 1:
        return None
    first = window.iloc[0]
    initial_speed = float(np.hypot(float(first["vx"]), float(first["vy"])))
    xy = window[["x", "y"]].to_numpy(dtype=float)
    deltas = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    path_length = float(np.nansum(seg_lengths))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    if not all(np.isfinite(value) for value in (initial_speed, path_length, displacement)):
        return None
    return initial_speed, path_length, displacement


def _has_valid_window(
    *,
    scene_tracks: Mapping[str, Any],
    ego_id: str,
    init_timestep: int,
    num_steps: int,
    future_steps: int,
    scene_length: int,
) -> bool:
    tp_data = next((value for key, value in scene_tracks.items() if str(key) == ego_id), None)
    if tp_data is None:
        return False
    state = tp_data.get("State")
    if state is None or state.empty or "frame_id" not in state.columns:
        return False
    required_last = init_timestep + num_steps
    required_scene_end = required_last + future_steps + 2
    if scene_length < required_scene_end:
        return False
    return int(state["frame_id"].min()) <= init_timestep and int(state["frame_id"].max()) >= required_last


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


def _scene_id_from_name(location: str, scene_name: str) -> str:
    prefix = f"{location}_"
    return scene_name[len(prefix) :] if scene_name.startswith(prefix) else scene_name


def _boosted_speed(
    original_speed: float,
    boost_min_speed: float = 9.0,
    boost_max_speed: float = 17.0,
    boost_scale: float = 2.0,
) -> float:
    return float(max(boost_min_speed, min(boost_scale * original_speed, boost_max_speed)))


def _apply_shard(
    plans: Sequence[NarrowPlan], shard_index: int, num_shards: int
) -> List[NarrowPlan]:
    if num_shards <= 1:
        return list(plans)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")
    return [plan for idx, plan in enumerate(plans) if idx % num_shards == shard_index]


def _run_dir(policy_dir: Path, plan: NarrowPlan) -> Path:
    return (
        policy_dir
        / "runs"
        / plan.speed_mode
        / plan.intersection
        / plan.location
        / _safe_path_name(plan.scene_name)
        / _safe_path_name(plan.run_id)
    )


def _completed_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("status") == "completed")


def _with_metric_anomaly_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("status") != "completed":
        row.setdefault("metric_anomaly", "")
        row.setdefault("metric_anomaly_reason", "")
        return row
    anomaly, reason = _metric_anomaly(row)
    row["metric_anomaly"] = anomaly
    row["metric_anomaly_reason"] = reason
    return row


def _metric_anomaly(row: Mapping[str, Any]) -> Tuple[bool, str]:
    reasons: List[str] = []
    ade = _finite_float(row.get("ADE"))
    fde = _finite_float(row.get("FDE"))
    if ade is not None and ade > 10.0:
        reasons.append("ADE>10")
    if fde is not None and fde > 50.0:
        reasons.append("FDE>50")
    return bool(reasons), ";".join(reasons)


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


def _delta(left: Any, right: Any) -> float:
    left_float = _finite_float(left)
    right_float = _finite_float(right)
    if left_float is None or right_float is None:
        return np.nan
    return left_float - right_float


def _fmt_optional(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
