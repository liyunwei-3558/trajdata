from __future__ import annotations

import argparse
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
from Simulation_test_toolchain.core.runner import run_simulation
from Simulation_test_toolchain.run_raw_open_loop_trajectory_batch import (
    _bool_rate,
    _canonical_intersection,
    _csv_value,
    _is_metric_anomaly_row,
    _load_manifest,
    _manifest_key,
    _metric_anomaly,
    _numeric_series,
    _safe_path_name,
    _to_bool,
    _with_metric_anomaly_fields,
)


DEFAULT_LABEL_PATHS = (
    "datasets/SinD_dataset/Semantic_labels/scenarios.json",
    "risk_mining/typical_risks_extract/high_risk_mprttc/output/high_risk_mprttc_non_tj_scenarios.json",
)
DEFAULT_INTERSECTIONS = ("cc", "tj", "cqIR", "cqNR", "cqR", "xa")
DEFAULT_POLICY = "risk_idm"
POLICY_OUTPUT_DIRS = {
    "risk_idm": "raw_open_loop_risk_idm",
    "asaprl": "raw_open_loop_asaprl",
}
MANIFEST_BASE_FIELDS = (
    "policy",
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
    "min_mprttc",
    "label_start_frame",
    "label_end_frame",
    "init_timestep",
    "num_steps_requested",
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
class MprTTCPlan:
    policy: str
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
    min_mprttc: float
    label_start_frame: int
    label_end_frame: int
    init_timestep: int
    num_steps: int
    semantic_label: Dict[str, Any]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_root = args.output_root.expanduser()
    policy = args.policy.lower()
    if policy not in POLICY_OUTPUT_DIRS:
        raise ValueError(
            f"Unknown policy {args.policy!r}; expected one of {sorted(POLICY_OUTPUT_DIRS)}"
        )
    policy_dir = output_root / POLICY_OUTPUT_DIRS[policy]
    if args.num_shards > 1:
        policy_dir = policy_dir / f"shard_{args.shard_index:02d}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    plans_by_intersection = build_mprttc_plans(
        label_paths=[Path(path) for path in args.label_paths],
        data_dir=data_dir,
        intersections=tuple(args.intersections),
        target_per_intersection=args.target_per_intersection,
        num_steps=args.num_steps,
        history_steps=int(round(args.history_sec / args.dt)),
        future_steps=int(np.ceil(args.future_sec / args.dt)),
        policy=policy,
    )
    total_plans = sum(len(plans) for plans in plans_by_intersection.values())
    print(
        f"[mprttc-batch] policy={policy} planned={total_plans} intersections={','.join(plans_by_intersection)} "
        f"output={policy_dir}",
        flush=True,
    )

    metric_context = MetricContext(data_dir=data_dir, dt=args.dt)
    rows_by_key = _load_manifest(policy_dir / "run_manifest.csv")
    if args.force:
        rows_by_key = {}

    for intersection, plans in plans_by_intersection.items():
        completed = 0 if args.force else _completed_count(rows_by_key.values(), intersection)
        print(
            f"[mprttc-batch] {policy} {intersection} start completed={completed} "
            f"target={args.target_per_intersection}",
            flush=True,
        )
        if completed >= args.target_per_intersection and not args.force:
            continue
        plans = _apply_shard(plans, args.shard_index, args.num_shards)
        for plan in plans:
            key = _manifest_key(policy, plan.location, plan.run_id)
            if key in rows_by_key and not args.force:
                existing = rows_by_key[key]
                if existing.get("status") == "completed":
                    completed += 1
                    if completed >= args.target_per_intersection:
                        break
                    continue
                if existing.get("status") == "failed":
                    continue

            run_dir = _run_dir(policy_dir, plan)
            row = _base_manifest_row(plan, run_dir)
            started = time.time()
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _make_config(plan, data_dir, run_dir, args)
                _write_config(run_dir / "config.yaml", cfg)
                print(
                    f"[mprttc-batch] run policy={policy} intersection={intersection} rank={plan.candidate_rank} "
                    f"scenario={plan.scenario_id} location={plan.location} "
                    f"scene={plan.scene_name} ego={plan.agent_id} init={plan.init_timestep} "
                    f"min_mprttc={plan.min_mprttc:g}",
                    flush=True,
                )
                result = run_simulation(cfg)
                result.metadata.update(
                    {
                        "experiment_mode": "dataset_open_loop_mprttc",
                        "open_loop_definition": (
                            "non-ego agents replay ground truth; ego is controlled by policy"
                        ),
                        "runner_mode": cfg.simulation.mode,
                        "ego_policy": policy,
                        "run_id": plan.run_id,
                        "semantic_label_id": plan.scenario_id,
                        "semantic_label": plan.semantic_label,
                        "candidate_rank": plan.candidate_rank,
                        "min_mprttc": plan.min_mprttc,
                    }
                )
                result.save_json(run_dir / "trajectory_log.json")
                metrics = compute_batch_metrics(result, data_dir, metric_context)
                save_metrics(run_dir / "metrics.json", result.metadata, metrics)
                row.update(metrics)
                row["status"] = "completed"
                completed += 1
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
                print(
                    f"[mprttc-batch] failed scenario={plan.scenario_id}: {exc}",
                    flush=True,
                )
            finally:
                row["elapsed_s"] = round(time.time() - started, 3)
                rows_by_key[key] = row
                _write_outputs(policy_dir, rows_by_key.values())
            if completed >= args.target_per_intersection:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset-open-loop tests on low-MprTTC semantic labels."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Simulation_test_toolchain/batch_outputs/mprttc_riskidm_1500"),
    )
    parser.add_argument("--label-paths", nargs="+", default=list(DEFAULT_LABEL_PATHS))
    parser.add_argument("--intersections", nargs="+", default=list(DEFAULT_INTERSECTIONS))
    parser.add_argument(
        "--policy",
        type=str,
        default=DEFAULT_POLICY,
        choices=sorted(POLICY_OUTPUT_DIRS),
    )
    parser.add_argument("--target-per-intersection", type=int, default=250)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--asaprl-target-speed", type=float, default=5.0)
    parser.add_argument("--asaprl-horizon", type=float, default=3.0)
    parser.add_argument("--asaprl-action-lat-scale", type=float, default=1.0)
    parser.add_argument("--asaprl-action-yaw-scale", type=float, default=2.0)
    parser.add_argument("--asaprl-action-speed-scale", type=float, default=3.0)
    parser.add_argument("--asaprl-observation-px-per-m", type=float, default=3.0)
    parser.add_argument("--asaprl-reference-heading-blend", type=float, default=0.8)
    parser.add_argument("--asaprl-reference-speed-blend", type=float, default=0.0)
    parser.add_argument("--asaprl-max-yaw-rate", type=float, default=0.3)
    parser.add_argument("--asaprl-max-speed", type=float, default=8.0)
    parser.add_argument("--asaprl-no-follow-reference-direction", action="store_true")
    parser.add_argument(
        "--asaprl-ckpt-path",
        type=str,
        default="Simulation_test_toolchain/checkpoints/ckpt_best_iter497000_step417741440.pth.tar",
    )
    parser.add_argument("--asaprl-device", type=str, default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_mprttc_plans(
    *,
    label_paths: Sequence[Path],
    data_dir: Path,
    intersections: Sequence[str],
    target_per_intersection: int,
    num_steps: int,
    history_steps: int,
    future_steps: int,
    policy: str,
) -> Dict[str, List[MprTTCPlan]]:
    labels = _load_mprttc_labels(label_paths)
    scene_cache: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    scene_length_cache: Dict[Tuple[str, str], int] = {}
    scene_index_cache: Dict[str, Dict[str, int]] = {}
    best_by_window: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for label in labels:
        location = str(label["location"])
        intersection = _canonical_intersection(location)
        if intersection not in intersections:
            continue
        if location not in scene_index_cache:
            scene_index_cache[location] = _scene_index_map(data_dir, location)
        scene_name = str(label["source_scene_name"])
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
        if not _has_valid_window(
            scene_tracks=scene_tracks,
            ego_id=ego_id,
            init_timestep=init_timestep,
            num_steps=num_steps,
            future_steps=future_steps,
            scene_length=scene_length,
        ):
            continue
        min_mprttc = _label_min_mprttc(label)
        key = (location, scene_name, ego_id, init_timestep)
        existing = best_by_window.get(key)
        if existing is None or min_mprttc < _label_min_mprttc(existing):
            best_by_window[key] = label

    candidates_by_intersection: Dict[str, List[Dict[str, Any]]] = {}
    for label in best_by_window.values():
        candidates_by_intersection.setdefault(
            _canonical_intersection(str(label["location"])), []
        ).append(label)

    plans_by_intersection: Dict[str, List[MprTTCPlan]] = {}
    for intersection in intersections:
        candidates = candidates_by_intersection.get(intersection, [])
        candidates.sort(
            key=lambda item: (
                _label_min_mprttc(item),
                int(item["time_window"]["start_frame"]),
                str(item["location"]),
                str(item["source_scene_name"]),
                str(item["agents"]["ego_id"]),
                str(item["scenario_id"]),
            )
        )
        selected = candidates[:target_per_intersection]
        plans: List[MprTTCPlan] = []
        for rank, label in enumerate(selected, start=1):
            plans.append(
                _plan_from_label(
                    label,
                    rank,
                    num_steps,
                    history_steps,
                    policy=policy,
                    scene_index=scene_index_cache[str(label["location"])][
                        str(label["source_scene_name"])
                    ],
                )
            )
        plans_by_intersection[intersection] = plans
        print(
            f"[mprttc-batch] intersection={intersection} candidates={len(candidates)} "
            f"selected={len(plans)} min_mprttc_top={_label_min_mprttc(plans[0].semantic_label) if plans else 'NA'}",
            flush=True,
        )
    return plans_by_intersection


def _load_mprttc_labels(label_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    seen = set()
    for path in label_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
        for label in items:
            if label.get("semantics", {}).get("type") != "mprttc":
                continue
            scenario_id = str(label.get("scenario_id"))
            if scenario_id in seen:
                continue
            seen.add(scenario_id)
            labels.append(label)
    return labels


def _plan_from_label(
    label: Dict[str, Any],
    rank: int,
    num_steps: int,
    history_steps: int,
    *,
    policy: str,
    scene_index: int,
) -> MprTTCPlan:
    location = str(label["location"])
    scene_name = str(label["source_scene_name"])
    scene_id = _scene_id_from_name(location, scene_name)
    ego_id = str(label["agents"]["ego_id"])
    label_start = int(label["time_window"]["start_frame"])
    label_end = int(label["time_window"]["end_frame"])
    init_timestep = max(label_start, history_steps)
    scenario_id = str(label["scenario_id"])
    run_id = f"{scenario_id}__ego{ego_id}__t{init_timestep:04d}"
    return MprTTCPlan(
        policy=policy,
        intersection=_canonical_intersection(location),
        location=location,
        scene_name=scene_name,
        scene_id=scene_id,
        scene_index=int(scene_index),
        run_id=run_id,
        scenario_id=scenario_id,
        agent_id=ego_id,
        challenger_id=(
            None
            if label.get("agents", {}).get("challenger_id") is None
            else str(label["agents"]["challenger_id"])
        ),
        candidate_rank=rank,
        min_mprttc=_label_min_mprttc(label),
        label_start_frame=label_start,
        label_end_frame=label_end,
        init_timestep=init_timestep,
        num_steps=num_steps,
        semantic_label=label,
    )


def _make_config(
    plan: MprTTCPlan,
    data_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> ToolchainConfig:
    if plan.policy == "asaprl":
        ego_params = {
            "target_speed": args.asaprl_target_speed,
            "horizon": args.asaprl_horizon,
            "ckpt_path": args.asaprl_ckpt_path,
            "device": args.asaprl_device,
            "use_risk_idm": True,
            "neighbor_radius": args.neighbor_radius,
            "inference_interval_steps": 5,
            "action_lat_scale": args.asaprl_action_lat_scale,
            "action_yaw_scale": args.asaprl_action_yaw_scale,
            "action_speed_scale": args.asaprl_action_speed_scale,
            "observation_px_per_m": args.asaprl_observation_px_per_m,
            "reference_heading_blend": args.asaprl_reference_heading_blend,
            "reference_speed_blend": args.asaprl_reference_speed_blend,
            "max_yaw_rate": args.asaprl_max_yaw_rate,
            "max_speed": args.asaprl_max_speed,
            "follow_reference_direction": not args.asaprl_no_follow_reference_direction,
            "require_raster_map": True,
        }
    else:
        ego_params = {
            "desired_velocity": 8.0,
            "max_acceleration": 3.0,
            "min_acceleration": -5.0,
            "inference_interval_steps": 5,
        }
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
            asaprl_ckpt_path=args.asaprl_ckpt_path,
            asaprl_device=args.asaprl_device,
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


def _base_manifest_row(plan: MprTTCPlan, run_dir: Path) -> Dict[str, Any]:
    return {
        "policy": plan.policy,
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
        "min_mprttc": plan.min_mprttc,
        "label_start_frame": plan.label_start_frame,
        "label_end_frame": plan.label_end_frame,
        "init_timestep": plan.init_timestep,
        "num_steps_requested": plan.num_steps,
        "status": "pending",
        "run_dir": str(run_dir),
        "trajectory_log": str(run_dir / "trajectory_log.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "error": "",
        "elapsed_s": "",
    }


def _apply_shard(
    plans: Sequence[MprTTCPlan], shard_index: int, num_shards: int
) -> List[MprTTCPlan]:
    if num_shards <= 1:
        return list(plans)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}]")
    return [plan for idx, plan in enumerate(plans) if idx % num_shards == shard_index]


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
                item.get("intersection", ""),
                _sort_int(item.get("candidate_rank")),
                item.get("run_id", ""),
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
        _summary_row(intersection, group) for intersection, group in df.groupby("intersection")
    ]
    by_intersection_df = pd.DataFrame.from_records(by_intersection)
    if "location" in by_intersection_df.columns:
        by_intersection_df = by_intersection_df.rename(columns={"location": "intersection"})
    by_intersection_df.to_csv(policy_dir / "summary_by_intersection.csv", index=False)
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
    for metric in ("ADE", "FDE", "MinTTC", "AveTTC", "MRD", "ARD"):
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


def _completed_count(rows: Iterable[Mapping[str, Any]], intersection: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("status") == "completed"
        and (row.get("intersection") or _canonical_intersection(str(row.get("location", ""))))
        == intersection
    )


def _sort_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _run_dir(policy_dir: Path, plan: MprTTCPlan) -> Path:
    return (
        policy_dir
        / "runs"
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


def _label_min_mprttc(label: Mapping[str, Any]) -> float:
    value = label.get("semantics", {}).get("min_mprttc")
    if value is None:
        return float("inf")
    return float(value)


if __name__ == "__main__":
    main()
