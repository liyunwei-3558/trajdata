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

DEFAULT_LOCATION_SCENE_COUNTS: Mapping[str, int] = {
    "cc": 7,
    "tj": 23,
    "cqIR": 8,
    "cqNR": 10,
    "cqR": 10,
    "xa": 13,
    "xasl": 15,
}
DEFAULT_LOCATIONS: Tuple[str, ...] = tuple(DEFAULT_LOCATION_SCENE_COUNTS.keys())
LOCATION_TO_INTERSECTION: Mapping[str, str] = {
    "xasl": "xa",
}
POLICY_OUTPUT_DIRS: Mapping[str, str] = {
    "risk_idm": "raw_open_loop_risk_idm",
    "asaprl": "raw_open_loop_asaprl",
}
ADE_ANOMALY_THRESHOLD = 10.0
FDE_ANOMALY_THRESHOLD = 50.0
VEHICLE_CLASSES = {"car", "truck", "bus", "tricycle", "mv"}
MANIFEST_BASE_FIELDS: Tuple[str, ...] = (
    "policy",
    "intersection",
    "location",
    "scene_name",
    "scene_id",
    "scene_index",
    "run_id",
    "agent_id",
    "candidate_rank",
    "ego_track_duration",
    "first_frame",
    "last_frame",
    "ego_agent",
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
class TrajectoryPlan:
    location: str
    scene_id: str
    scene_name: str
    scene_index: int
    run_id: str
    agent_id: str
    candidate_rank: int
    ego_track_duration: int
    first_frame: int
    last_frame: int
    init_timestep: int
    scene_length: int
    skip_reason: str = ""


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_root = args.output_root.expanduser()
    plans_by_intersection = build_trajectory_plans(
        data_dir=data_dir,
        locations=tuple(args.locations),
        num_steps=args.num_steps,
        history_steps=int(round(args.history_sec / args.dt)),
        future_steps=int(np.ceil(args.future_sec / args.dt)),
        target_per_location=args.target_per_location,
        min_ego_mean_speed=args.min_ego_mean_speed,
        min_ego_moving_rate=args.min_ego_moving_rate,
        moving_speed_threshold=args.moving_speed_threshold,
        min_ego_displacement=args.min_ego_displacement,
        max_ego_heading_change=args.max_ego_heading_change,
    )

    print(
        f"[batch] planned intersections={len(plans_by_intersection)} "
        f"target_per_location={args.target_per_location} "
        f"policies={','.join(args.policies)} output_root={output_root}",
        flush=True,
    )

    metric_context = MetricContext(data_dir=data_dir, dt=args.dt)
    for policy in args.policies:
        run_policy_batch(
            policy=policy,
            plans_by_location=plans_by_intersection,
            data_dir=data_dir,
            output_root=output_root,
            metric_context=metric_context,
            args=args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw SinD trajectory batches with a target count per intersection."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Simulation_test_toolchain/batch_outputs/raw_open_loop_250traj"),
    )
    parser.add_argument("--locations", nargs="+", default=list(DEFAULT_LOCATIONS))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICY_OUTPUT_DIRS),
        default=list(POLICY_OUTPUT_DIRS),
    )
    parser.add_argument("--target-per-location", type=int, default=250)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--min-ego-mean-speed", type=float, default=0.5)
    parser.add_argument("--min-ego-moving-rate", type=float, default=0.35)
    parser.add_argument("--moving-speed-threshold", type=float, default=0.5)
    parser.add_argument("--min-ego-displacement", type=float, default=12.0)
    parser.add_argument("--max-ego-heading-change", type=float, default=0.8)
    parser.add_argument("--asaprl-action-lat-scale", type=float, default=1.0)
    parser.add_argument("--asaprl-action-yaw-scale", type=float, default=2.0)
    parser.add_argument("--asaprl-action-speed-scale", type=float, default=3.0)
    parser.add_argument("--asaprl-observation-px-per-m", type=float, default=3.0)
    parser.add_argument("--asaprl-reference-heading-blend", type=float, default=0.8)
    parser.add_argument("--asaprl-reference-speed-blend", type=float, default=0.0)
    parser.add_argument("--asaprl-max-yaw-rate", type=float, default=0.3)
    parser.add_argument("--asaprl-max-speed", type=float, default=8.0)
    parser.add_argument(
        "--asaprl-no-follow-reference-direction",
        action="store_true",
        help="Disable ASAPRL reference-direction drift guard.",
    )
    parser.add_argument(
        "--asaprl-ckpt-path",
        type=str,
        default="Simulation_test_toolchain/checkpoints/ckpt_best_iter497000_step417741440.pth.tar",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_policy_batch(
    policy: str,
    plans_by_location: Mapping[str, Sequence[TrajectoryPlan]],
    data_dir: Path,
    output_root: Path,
    metric_context: MetricContext,
    args: argparse.Namespace,
) -> None:
    policy_dir = output_root / POLICY_OUTPUT_DIRS[policy]
    policy_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = policy_dir / "run_manifest.csv"
    rows_by_key = _load_manifest(manifest_path)

    for intersection, plans in plans_by_location.items():
        if args.force:
            rows_by_key = {
                key: row
                for key, row in rows_by_key.items()
                if (
                    row.get("intersection")
                    or _canonical_intersection(str(row.get("location", "")))
                )
                != intersection
            }
        completed = 0 if args.force else _completed_count(rows_by_key.values(), intersection)
        if completed >= args.target_per_location and not args.force:
            print(
                f"[batch] {policy} {intersection} already has {completed} completed runs",
                flush=True,
            )
            continue

        print(
            f"[batch] {policy} {intersection} starting at completed={completed} "
            f"target={args.target_per_location}",
            flush=True,
        )

        for plan in plans:
            key = _manifest_key(policy, plan.location, plan.run_id)
            row = _base_manifest_row(
                policy=policy,
                plan=plan,
                run_dir=_run_dir(policy_dir, plan),
                args=args,
            )
            started = time.time()

            if key in rows_by_key:
                existing = rows_by_key[key]
                if existing.get("status") == "completed" and not args.force:
                    completed += 1
                    if completed >= args.target_per_location:
                        break
                    continue
                if existing.get("status") == "failed" and not args.force:
                    continue

            try:
                run_dir = Path(row["run_dir"])
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _make_config(policy, plan, data_dir, run_dir, args)
                _write_config(run_dir / "config.yaml", cfg)
                print(
                    f"[batch] run policy={policy} intersection={intersection} location={plan.location} "
                    f"scene={plan.scene_name} agent={plan.agent_id} run={plan.run_id}",
                    flush=True,
                )
                result = run_simulation(cfg)
                result.metadata.update(
                    {
                        "experiment_mode": "dataset_open_loop",
                        "open_loop_definition": (
                            "non-ego agents replay ground truth; ego is controlled by policy"
                        ),
                        "runner_mode": cfg.simulation.mode,
                        "run_id": plan.run_id,
                        "candidate_rank": plan.candidate_rank,
                        "ego_track_duration": plan.ego_track_duration,
                        "first_frame": plan.first_frame,
                        "last_frame": plan.last_frame,
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
                    f"[batch] failed policy={policy} intersection={intersection} location={plan.location} "
                    f"run={plan.run_id}: {exc}",
                    flush=True,
                )
            finally:
                row["elapsed_s"] = round(time.time() - started, 3)
                rows_by_key[key] = row
                _write_outputs(policy_dir, rows_by_key.values())
                if completed >= args.target_per_location:
                    break

        if completed < args.target_per_location:
            print(
                f"[batch] warning: {policy} {intersection} completed={completed} "
                f"target={args.target_per_location}",
                flush=True,
            )


def build_trajectory_plans(
    data_dir: Path,
    locations: Sequence[str],
    num_steps: int,
    history_steps: int,
    future_steps: int,
    target_per_location: int,
    min_ego_mean_speed: float,
    min_ego_moving_rate: float,
    moving_speed_threshold: float,
    min_ego_displacement: float,
    max_ego_heading_change: float,
) -> Dict[str, List[TrajectoryPlan]]:
    plans_by_intersection: Dict[str, List[TrajectoryPlan]] = {}
    for location in locations:
        tp_info = _load_location_tp_info(data_dir, location)
        expected = DEFAULT_LOCATION_SCENE_COUNTS.get(location)
        if expected is not None and len(tp_info) != expected:
            print(
                f"[batch] warning: location={location} expected_scenes={expected} "
                f"actual_scenes={len(tp_info)}",
                flush=True,
            )

        candidates: List[TrajectoryPlan] = []
        scene_index = 0
        for scene_id_raw, scene_tracks in tp_info.items():
            scene_id = str(scene_id_raw)
            scene_name = f"{location}_{scene_id}"
            scene_length = _scene_length(scene_tracks)
            if scene_length <= 1:
                scene_index += 1
                continue
            for agent_id_raw, tp_data in scene_tracks.items():
                agent_id = str(agent_id_raw)
                class_name = str(tp_data.get("Class", tp_data.get("Type", ""))).lower()
                if class_name not in VEHICLE_CLASSES:
                    continue
                state = tp_data.get("State")
                if state is None or state.empty or "frame_id" not in state.columns:
                    continue
                first_frame = int(state["frame_id"].min())
                last_frame = int(state["frame_id"].max())
                init_timestep = max(first_frame + history_steps, history_steps)
                required_last = init_timestep + num_steps
                required_scene_end = required_last + future_steps + 2
                if last_frame < required_last or scene_length < required_scene_end:
                    continue
                if _is_waiting_segment(
                    state,
                    init_timestep,
                    required_last,
                    min_mean_speed=min_ego_mean_speed,
                    min_moving_rate=min_ego_moving_rate,
                    moving_speed_threshold=moving_speed_threshold,
                    min_displacement=min_ego_displacement,
                    max_heading_change=max_ego_heading_change,
                ):
                    continue
                duration = last_frame - first_frame + 1
                run_id = f"{scene_name}__ego{agent_id}__t{init_timestep:04d}"
                candidates.append(
                    TrajectoryPlan(
                        location=location,
                        scene_id=scene_id,
                        scene_name=scene_name,
                        scene_index=scene_index,
                        run_id=run_id,
                        agent_id=agent_id,
                        candidate_rank=0,
                        ego_track_duration=duration,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        init_timestep=init_timestep,
                        scene_length=scene_length,
                    )
                )
            scene_index += 1

        intersection = _canonical_intersection(location)
        plans_by_intersection.setdefault(intersection, []).extend(candidates)
        print(
            f"[batch] location={location} intersection={intersection} candidates={len(candidates)} "
            f"top_duration={candidates[0].ego_track_duration if candidates else 0}",
            flush=True,
        )

    for intersection, candidates in list(plans_by_intersection.items()):
        candidates.sort(
            key=lambda plan: (
                -plan.ego_track_duration,
                plan.location,
                plan.scene_name,
                plan.agent_id,
                plan.init_timestep,
            )
        )
        plans_by_intersection[intersection] = [
            _with_rank(plan, rank + 1) for rank, plan in enumerate(candidates)
        ]
        print(
            f"[batch] intersection={intersection} merged_candidates={len(candidates)} "
            f"locations={','.join(sorted({plan.location for plan in candidates}))}",
            flush=True,
        )
    return plans_by_intersection


def _with_rank(plan: TrajectoryPlan, rank: int) -> TrajectoryPlan:
    return TrajectoryPlan(
        location=plan.location,
        scene_id=plan.scene_id,
        scene_name=plan.scene_name,
        scene_index=plan.scene_index,
        run_id=plan.run_id,
        agent_id=plan.agent_id,
        candidate_rank=rank,
        ego_track_duration=plan.ego_track_duration,
        first_frame=plan.first_frame,
        last_frame=plan.last_frame,
        init_timestep=plan.init_timestep,
        scene_length=plan.scene_length,
        skip_reason=plan.skip_reason,
    )


def _is_waiting_segment(
    state: pd.DataFrame,
    init_timestep: int,
    required_last: int,
    min_mean_speed: float,
    min_moving_rate: float,
    moving_speed_threshold: float,
    min_displacement: float,
    max_heading_change: float,
) -> bool:
    window = state[
        (state["frame_id"] >= init_timestep)
        & (state["frame_id"] <= required_last)
    ]
    if window.empty:
        return True
    if {"vx", "vy"}.issubset(window.columns):
        speed = np.hypot(
            window["vx"].to_numpy(dtype=float),
            window["vy"].to_numpy(dtype=float),
        )
    else:
        xy = window[["x", "y"]].to_numpy(dtype=float)
        if len(xy) < 2:
            return True
        step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.1
        speed = np.concatenate([[step_speed[0]], step_speed])
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    mean_speed = float(np.mean(speed))
    moving_rate = float(np.mean(speed > moving_speed_threshold))
    xy = window[["x", "y"]].to_numpy(dtype=float)
    displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) >= 2 else 0.0
    heading_change = 0.0
    if "heading" in window.columns:
        heading = np.unwrap(window["heading"].to_numpy(dtype=float))
        heading_change = float(np.nanmax(heading) - np.nanmin(heading))
    return (
        mean_speed < min_mean_speed
        or moving_rate < min_moving_rate
        or displacement < min_displacement
        or heading_change > max_heading_change
    )


def _make_config(
    policy: str,
    plan: TrajectoryPlan,
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
        ),
        scenario=ScenarioConfig(
            scene_index=plan.scene_index,
            scene_name=plan.scene_name,
            init_timestep=plan.init_timestep,
            num_steps=args.num_steps,
            ego_selection_strategy="longest_trajectory",
            ego_agent_name=plan.agent_id,
        ),
        simulation=SimulationConfig(
            mode="ego_closed_loop",
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            neighbor_radius=args.neighbor_radius,
        ),
        policies=PolicyConfig(
            ego_policy=policy,
            non_ego_policy="ground_truth",
            ego=_policy_params(policy, args),
            non_ego={},
        ),
        checkpoints=CheckpointConfig(asaprl_ckpt_path=args.asaprl_ckpt_path),
        output=OutputConfig(
            project_name=run_dir.name,
            root_dir=str(run_dir.parent),
            save_html=False,
            save_json=True,
            save_csv=False,
        ),
        visualization=VisualizationConfig(enabled=False),
    )


def _policy_params(policy: str, args: argparse.Namespace) -> Dict[str, Any]:
    if policy == "risk_idm":
        return {
            "desired_velocity": 8.0,
            "max_acceleration": 3.0,
            "min_acceleration": -5.0,
            "inference_interval_steps": 5,
        }
    if policy == "asaprl":
        return {
            "target_speed": 7.5,
            "horizon": 3.0,
            "use_risk_idm": True,
            "inference_interval_steps": 5,
            "require_raster_map": True,
            "action_lat_scale": getattr(args, "asaprl_action_lat_scale", 1.0),
            "action_yaw_scale": getattr(args, "asaprl_action_yaw_scale", 2.0),
            "action_speed_scale": getattr(args, "asaprl_action_speed_scale", 3.0),
            "observation_px_per_m": getattr(args, "asaprl_observation_px_per_m", 3.0),
            "reference_heading_blend": getattr(
                args, "asaprl_reference_heading_blend", 0.8
            ),
            "reference_speed_blend": getattr(
                args, "asaprl_reference_speed_blend", 0.0
            ),
            "max_yaw_rate": getattr(args, "asaprl_max_yaw_rate", 0.3),
            "max_speed": getattr(args, "asaprl_max_speed", 10.0),
            "follow_reference_direction": not getattr(
                args, "asaprl_no_follow_reference_direction", False
            ),
        }
    raise ValueError(f"Unsupported policy: {policy}")


def _run_dir(policy_dir: Path, plan: TrajectoryPlan) -> Path:
    return policy_dir / "runs" / plan.location / _safe_path_name(plan.scene_name) / _safe_path_name(plan.run_id)


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
            key=lambda item: (item.get("location", ""), item.get("scene_name", ""), item.get("run_id", "")),
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
    df = df.copy()
    df["intersection"] = [
        row.get("intersection") or _canonical_intersection(str(row.get("location", "")))
        for row in df.to_dict("records")
    ]
    by_intersection = [
        _summary_row(intersection, group) for intersection, group in df.groupby("intersection")
    ]
    by_intersection_df = pd.DataFrame.from_records(by_intersection)
    if "location" in by_intersection_df.columns:
        by_intersection_df = by_intersection_df.rename(columns={"location": "intersection"})
    by_intersection_df.to_csv(
        policy_dir / "summary_by_intersection.csv", index=False
    )
    pd.DataFrame.from_records([_summary_row("ALL", df)]).to_csv(
        policy_dir / "summary_total.csv", index=False
    )


def _summary_row(label: str, df: pd.DataFrame) -> Dict[str, Any]:
    completed = df[df["status"] == "completed"].copy()
    metric_anomaly_mask = completed.apply(_is_metric_anomaly_row, axis=1)
    metric_valid = completed[~metric_anomaly_mask].copy()
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


def _with_metric_anomaly_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    is_anomaly, reason = _metric_anomaly(row)
    row["metric_anomaly"] = bool(is_anomaly)
    row["metric_anomaly_reason"] = reason
    return row


def _is_metric_anomaly_row(row: pd.Series) -> bool:
    is_anomaly, _ = _metric_anomaly(row)
    return is_anomaly


def _metric_anomaly(row: Mapping[str, Any]) -> Tuple[bool, str]:
    if row.get("status") != "completed":
        return False, ""
    reasons: List[str] = []
    ade = _to_float(row.get("ADE"))
    fde = _to_float(row.get("FDE"))
    if ade is not None and ade > ADE_ANOMALY_THRESHOLD:
        reasons.append(f"ADE>{ADE_ANOMALY_THRESHOLD:g}")
    if fde is not None and fde > FDE_ANOMALY_THRESHOLD:
        reasons.append(f"FDE>{FDE_ANOMALY_THRESHOLD:g}")
    return bool(reasons), ";".join(reasons)


def _bool_rate(series: Optional[pd.Series]) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return float(series.map(_to_bool).mean())


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["intersection"] = row.get("intersection") or _canonical_intersection(
                str(row.get("location", ""))
            )
            key = _manifest_key(row["policy"], row["location"], row["run_id"])
            rows[key] = dict(row)
    return rows


def _base_manifest_row(
    policy: str,
    plan: TrajectoryPlan,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "intersection": _canonical_intersection(plan.location),
        "location": plan.location,
        "scene_name": plan.scene_name,
        "scene_id": plan.scene_id,
        "scene_index": plan.scene_index,
        "run_id": plan.run_id,
        "agent_id": plan.agent_id,
        "candidate_rank": plan.candidate_rank,
        "ego_track_duration": plan.ego_track_duration,
        "first_frame": plan.first_frame,
        "last_frame": plan.last_frame,
        "ego_agent": plan.agent_id,
        "init_timestep": plan.init_timestep,
        "num_steps_requested": args.num_steps,
        "status": "pending",
        "run_dir": str(run_dir),
        "trajectory_log": str(run_dir / "trajectory_log.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "error": "",
        "elapsed_s": "",
    }


def _manifest_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = list(MANIFEST_BASE_FIELDS)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    return seen


def _manifest_key(policy: str, location: str, run_id: str) -> str:
    return f"{policy}|{location}|{run_id}"


def _load_location_tp_info(data_dir: Path, location: str) -> Mapping[str, Any]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def _scene_length(scene_tracks: Mapping[Any, Mapping[str, Any]]) -> int:
    max_frame = 0
    for tp_data in scene_tracks.values():
        state = tp_data.get("State")
        if state is not None and not state.empty and "frame_id" in state.columns:
            max_frame = max(max_frame, int(state["frame_id"].max()))
    return max_frame + 1


def _completed_count(rows: Iterable[Mapping[str, Any]], intersection: str) -> int:
    return sum(
        1
        for row in rows
        if (row.get("intersection") or _canonical_intersection(str(row.get("location", ""))))
        == intersection
        and row.get("status") == "completed"
    )


def _canonical_intersection(location: str) -> str:
    return LOCATION_TO_INTERSECTION.get(location, location)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_path_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


if __name__ == "__main__":
    main()
