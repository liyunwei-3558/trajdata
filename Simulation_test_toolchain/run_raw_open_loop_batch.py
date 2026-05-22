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
    load_result_json,
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
POLICY_OUTPUT_DIRS: Mapping[str, str] = {
    "risk_idm": "raw_open_loop_risk_idm",
    "asaprl": "raw_open_loop_asaprl",
}
VEHICLE_CLASSES = {"car", "truck", "bus", "tricycle", "mv"}
MANIFEST_BASE_FIELDS: Tuple[str, ...] = (
    "policy",
    "location",
    "scene_name",
    "scene_id",
    "scene_index",
    "ego_agent",
    "init_timestep",
    "num_steps_requested",
    "status",
    "run_dir",
    "trajectory_log",
    "metrics_json",
    "error",
    "elapsed_s",
)


@dataclass(frozen=True)
class ScenePlan:
    location: str
    scene_id: str
    scene_name: str
    scene_index: int
    scene_length: int
    ego_agent: Optional[str]
    init_timestep: Optional[int]
    skip_reason: str = ""


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_root = args.output_root.expanduser()
    locations = tuple(args.locations)
    policies = tuple(args.policies)

    plans = build_scene_plans(
        data_dir=data_dir,
        locations=locations,
        num_steps=args.num_steps,
        history_steps=int(round(args.history_sec / args.dt)),
        future_steps=int(np.ceil(args.future_sec / args.dt)),
    )
    if args.smoke:
        plans = _smoke_subset(plans, ("cc", "tj", "cqIR"))
    elif args.max_scenes_per_location is not None:
        plans = _limit_per_location(plans, args.max_scenes_per_location)

    print(
        f"[batch] planned scenes={len(plans)} policies={','.join(policies)} "
        f"output_root={output_root}",
        flush=True,
    )

    metric_context = MetricContext(data_dir=data_dir, dt=args.dt)
    for policy in policies:
        run_policy_batch(
            policy=policy,
            plans=plans,
            data_dir=data_dir,
            output_root=output_root,
            metric_context=metric_context,
            args=args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw SinD dataset-open-loop batches for RiskIDM/ASAPRL."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Simulation_test_toolchain/batch_outputs"),
    )
    parser.add_argument("--locations", nargs="+", default=list(DEFAULT_LOCATIONS))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICY_OUTPUT_DIRS),
        default=list(POLICY_OUTPUT_DIRS),
    )
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument(
        "--asaprl-ckpt-path",
        type=str,
        default="Simulation_test_toolchain/checkpoints/ckpt_best_iter497000_step417741440.pth.tar",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-scenes-per-location", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_policy_batch(
    policy: str,
    plans: Sequence[ScenePlan],
    data_dir: Path,
    output_root: Path,
    metric_context: MetricContext,
    args: argparse.Namespace,
) -> None:
    policy_dir = output_root / POLICY_OUTPUT_DIRS[policy]
    policy_dir.mkdir(parents=True, exist_ok=True)
    rows_by_key = _load_manifest(policy_dir / "run_manifest.csv")

    for plan in plans:
        key = _manifest_key(policy, plan.location, plan.scene_name)
        run_dir = policy_dir / "runs" / plan.location / _safe_path_name(plan.scene_name)
        trajectory_path = run_dir / "trajectory_log.json"
        metrics_path = run_dir / "metrics.json"
        row = _base_manifest_row(policy, plan, run_dir, trajectory_path, metrics_path, args)
        started = time.time()

        if plan.skip_reason:
            row["status"] = "skipped"
            row["error"] = plan.skip_reason
            rows_by_key[key] = row
            _write_outputs(policy_dir, rows_by_key)
            continue

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            if trajectory_path.exists() and metrics_path.exists() and not args.force:
                metrics = _load_metrics_flat(metrics_path)
                row.update(metrics)
                row["status"] = "completed"
                row["elapsed_s"] = 0.0
                rows_by_key[key] = row
                _write_outputs(policy_dir, rows_by_key)
                print(f"[batch] reuse {policy} {plan.scene_name}", flush=True)
                continue

            cfg = _make_config(policy, plan, data_dir, run_dir, args)
            _write_config(run_dir / "config.yaml", cfg)
            print(
                f"[batch] run policy={policy} location={plan.location} "
                f"scene={plan.scene_name} ego={plan.ego_agent} init={plan.init_timestep}",
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
                }
            )
            result.save_json(trajectory_path)
            metrics = compute_batch_metrics(result, data_dir, metric_context)
            save_metrics(metrics_path, result.metadata, metrics)
            row.update(metrics)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(
                f"[batch] failed policy={policy} scene={plan.scene_name}: {exc}",
                flush=True,
            )
        finally:
            row["elapsed_s"] = round(time.time() - started, 3)
            rows_by_key[key] = row
            _write_outputs(policy_dir, rows_by_key)


def build_scene_plans(
    data_dir: Path,
    locations: Sequence[str],
    num_steps: int,
    history_steps: int,
    future_steps: int,
) -> List[ScenePlan]:
    plans: List[ScenePlan] = []
    for location in locations:
        tp_info = _load_location_tp_info(data_dir, location)
        expected = DEFAULT_LOCATION_SCENE_COUNTS.get(location)
        if expected is not None and len(tp_info) != expected:
            print(
                f"[batch] warning: location={location} expected_scenes={expected} "
                f"actual_scenes={len(tp_info)}",
                flush=True,
            )
        scene_index = 0
        for scene_id_raw, scene_tracks in tp_info.items():
            scene_id = str(scene_id_raw)
            scene_length = _scene_length(scene_tracks)
            if scene_length <= 1:
                continue
            ego_agent, init_timestep, skip_reason = _select_ego_vehicle(
                scene_tracks,
                scene_length=scene_length,
                num_steps=num_steps,
                history_steps=history_steps,
                future_steps=future_steps,
            )
            plans.append(
                ScenePlan(
                    location=location,
                    scene_id=scene_id,
                    scene_name=f"{location}_{scene_id}",
                    scene_index=scene_index,
                    scene_length=scene_length,
                    ego_agent=ego_agent,
                    init_timestep=init_timestep,
                    skip_reason=skip_reason,
                )
            )
            scene_index += 1
    return plans


def _select_ego_vehicle(
    scene_tracks: Mapping[Any, Mapping[str, Any]],
    scene_length: int,
    num_steps: int,
    history_steps: int,
    future_steps: int,
) -> Tuple[Optional[str], Optional[int], str]:
    best: Optional[Tuple[int, str, int]] = None
    for raw_agent_id, tp_data in scene_tracks.items():
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
        duration = last_frame - first_frame + 1
        candidate = (duration, str(raw_agent_id), init_timestep)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None, None, f"no vehicle has a {num_steps}-step measurable window"
    _, agent_id, init_timestep = best
    return agent_id, init_timestep, ""


def _make_config(
    policy: str,
    plan: ScenePlan,
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
            ego_agent_name=plan.ego_agent,
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
            ego=_policy_params(policy),
            non_ego={},
        ),
        checkpoints=CheckpointConfig(asaprl_ckpt_path=args.asaprl_ckpt_path),
        output=OutputConfig(
            project_name=plan.scene_name,
            root_dir=str(run_dir),
            save_html=False,
            save_json=True,
            save_csv=False,
        ),
        visualization=VisualizationConfig(enabled=False),
    )


def _policy_params(policy: str) -> Dict[str, Any]:
    if policy == "risk_idm":
        return {
            "desired_velocity": 8.0,
            "max_acceleration": 3.0,
            "min_acceleration": -5.0,
        }
    if policy == "asaprl":
        return {
            "target_speed": 5.0,
            "horizon": 3.0,
            "use_risk_idm": True,
        }
    raise ValueError(f"Unsupported policy: {policy}")


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


def _write_outputs(policy_dir: Path, rows_by_key: Mapping[str, Mapping[str, Any]]) -> None:
    rows = list(rows_by_key.values())
    _write_manifest(policy_dir / "run_manifest.csv", rows)
    _write_summaries(policy_dir, rows)


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _manifest_fields(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (item.get("location", ""), item.get("scene_name", "")),
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
    pd.DataFrame.from_records([_summary_row("ALL", df)]).to_csv(
        policy_dir / "summary_total.csv", index=False
    )


def _summary_row(label: str, df: pd.DataFrame) -> Dict[str, Any]:
    completed = df[df["status"] == "completed"].copy()
    row: Dict[str, Any] = {
        "location": label,
        "planned_runs": int(len(df)),
        "completed_runs": int(len(completed)),
        "failed_runs": int((df["status"] == "failed").sum()),
        "skipped_runs": int((df["status"] == "skipped").sum()),
    }
    for metric in ("ADE", "FDE", "MinTTC", "AveTTC", "MRD", "ARD"):
        values = _numeric_series(completed, metric)
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
        row[f"{metric}_rate"] = _bool_rate(completed.get(metric))
    return row


def _bool_rate(series: Optional[pd.Series]) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return float(series.map(_to_bool).mean())


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _manifest_key(row["policy"], row["location"], row["scene_name"])
            rows[key] = dict(row)
    return rows


def _load_metrics_flat(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metrics", {}))


def _base_manifest_row(
    policy: str,
    plan: ScenePlan,
    run_dir: Path,
    trajectory_path: Path,
    metrics_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "location": plan.location,
        "scene_name": plan.scene_name,
        "scene_id": plan.scene_id,
        "scene_index": plan.scene_index,
        "ego_agent": plan.ego_agent or "",
        "init_timestep": plan.init_timestep if plan.init_timestep is not None else "",
        "num_steps_requested": args.num_steps,
        "status": "pending",
        "run_dir": str(run_dir),
        "trajectory_log": str(trajectory_path),
        "metrics_json": str(metrics_path),
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


def _manifest_key(policy: str, location: str, scene_name: str) -> str:
    return f"{policy}|{location}|{scene_name}"


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


def _smoke_subset(plans: Sequence[ScenePlan], locations: Sequence[str]) -> List[ScenePlan]:
    result: List[ScenePlan] = []
    for location in locations:
        plan = next(
            (
                item
                for item in plans
                if item.location == location and not item.skip_reason
            ),
            None,
        )
        if plan is not None:
            result.append(plan)
    return result


def _limit_per_location(plans: Sequence[ScenePlan], limit: int) -> List[ScenePlan]:
    counts: Dict[str, int] = {}
    result: List[ScenePlan] = []
    for plan in plans:
        count = counts.get(plan.location, 0)
        if count >= limit:
            continue
        result.append(plan)
        counts[plan.location] = count + 1
    return result


def _safe_path_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
