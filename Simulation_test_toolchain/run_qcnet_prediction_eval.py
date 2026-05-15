from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Legend,
    LegendItem,
    Select,
)
from bokeh.models.tools import WheelZoomTool
from bokeh.plotting import figure

from Simulation_test_toolchain.core.config import (
    DatasetConfig,
    PolicyConfig,
    ScenarioConfig,
    SimulationConfig,
    ToolchainConfig,
)
from Simulation_test_toolchain.core.runner import (
    _ensure_sim_cache_defaults,
    _find_agent_idx,
    _make_windowed_scene,
)
from Simulation_test_toolchain.core.state_utils import get_agent_world_pose, to_numpy
from Simulation_test_toolchain.policies.qcnet_adapter import (
    build_qcnet_sample_spec,
    spec_to_heterodata,
)


DEFAULT_QCNET_REPO = "/home/lyw/1TBSSD/lizhongze/DriverModel-FT/QCNet-main"
DEFAULT_QCNET_CKPT = (
    "/home/lyw/1TBSSD/lizhongze/DriverModel-FT/QCNet-main/QCNet_AV2.ckpt"
)
DEFAULT_TRACE_REPO = "Simulation_test_toolchain/external/trace"
VEHICLE_CLASSES = {"car", "truck", "bus", "tricycle", "mv"}


@dataclass(frozen=True)
class PredictionPlan:
    location: str
    scene_id: str
    scene_name: str
    scene_index: int
    run_id: str
    agent_id: str
    candidate_rank: int
    track_duration: int
    first_frame: int
    last_frame: int
    init_timestep: int
    scene_length: int
    displacement_m: float = 0.0
    mean_speed_mps: float = 0.0


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[prediction] loading QCNet ckpt={args.qcnet_ckpt_path} device={args.device}",
        flush=True,
    )
    torch, model = load_qcnet_model(
        repo_path=args.qcnet_repo_path,
        ckpt_path=args.qcnet_ckpt_path,
        device=args.device,
        strict=args.strict,
    )

    hist_steps = int(getattr(model, "num_historical_steps"))
    model_fut_steps = int(getattr(model, "num_future_steps"))
    output_dim = int(getattr(model, "output_dim", 2))
    if output_dim != 2:
        raise ValueError(
            "This prediction evaluator currently supports 2D QCNet outputs only."
        )

    eval_args = effective_window_args(args, hist_steps, model_fut_steps)
    run_args = argparse.Namespace(**vars(eval_args))
    run_args.no_html = True
    plans = build_prediction_plans(
        data_dir=eval_args.data_dir.expanduser(),
        location=eval_args.location,
        history_steps=hist_steps,
        future_steps=model_fut_steps,
        dt=eval_args.dt,
    )
    if eval_args.max_candidates is not None:
        plans = plans[: eval_args.max_candidates]
    print(
        f"[prediction] candidates={len(plans)} target_success={args.num_samples} "
        f"history_steps={hist_steps} model_future_steps={model_fut_steps}",
        flush=True,
    )

    dataset, scenes = build_dataset(eval_args)
    scene_by_name = {scene.name: scene for scene in scenes}
    rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    success_count = 0

    for plan in plans:
        if success_count >= args.num_samples:
            break
        row = base_row(plan)
        sample_started = time.time()
        try:
            scene = scene_by_name.get(plan.scene_name)
            if scene is None:
                if 0 <= plan.scene_index < len(scenes):
                    scene = scenes[plan.scene_index]
                else:
                    raise IndexError(
                        f"scene not found for {plan.scene_name!r} "
                        f"(scene_index={plan.scene_index}, scenes={len(scenes)})"
                    )
            sample = run_prediction_sample(
                plan=plan,
                scene=scene,
                dataset=dataset,
                model=model,
                torch=torch,
                args=eval_args,
                hist_steps=hist_steps,
                fut_steps=model_fut_steps,
            )
            row.update(sample["metrics"])
            row["status"] = "completed"
            row["error"] = ""
            row["valid_future_steps"] = sample["valid_future_steps"]
            row["top1_mode_index"] = sample["top1_mode_index"]
            row["best_mode_index"] = sample["best_mode_index"]
            row["top1_mode_prob"] = sample["top1_mode_prob"]
            row["best_mode_prob"] = sample["best_mode_prob"]
            predictions.append(sample["prediction"])
            success_count += 1
            print(
                f"[prediction] completed {success_count}/{args.num_samples} "
                f"{plan.run_id} minFDE={row['minFDE']:.3f} top1_FDE={row['top1_FDE']:.3f}",
                flush=True,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(f"[prediction] failed {plan.run_id}: {exc}", flush=True)
        finally:
            row["elapsed_s"] = round(time.time() - sample_started, 3)
            rows.append(row)
            write_outputs(
                output_dir=output_dir,
                args=run_args,
                rows=rows,
                predictions=predictions,
                elapsed_s=round(time.time() - started, 3),
            )

    if success_count < args.num_samples:
        print(
            f"[prediction] warning: completed={success_count} target={args.num_samples}",
            flush=True,
        )
    summary = summarize_rows(rows)
    write_outputs(
        output_dir=output_dir,
        args=eval_args,
        rows=rows,
        predictions=predictions,
        elapsed_s=round(time.time() - started, 3),
    )
    print("[prediction] summary", json.dumps(summary, indent=2), flush=True)
    print(f"[prediction] output: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-shot QCNet trajectory prediction evaluation on SinD."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--location", type=str, default="cqNR")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--history-sec", type=float, default=2.0)
    parser.add_argument("--future-sec", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--neighbor-radius", type=float, default=20.0)
    parser.add_argument("--map-radius", type=float, default=20.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--max-guesses", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--qcnet-repo-path", type=str, default=DEFAULT_QCNET_REPO)
    parser.add_argument("--qcnet-ckpt-path", type=str, default=DEFAULT_QCNET_CKPT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Simulation_test_toolchain/test_projects/qcnet_prediction_eval_cqNR"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    return parser.parse_args()


def add_trace_repo_path(repo_path: str = DEFAULT_TRACE_REPO) -> None:
    repo_dir = Path(repo_path).expanduser().resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def load_qcnet_model(repo_path: str, ckpt_path: str, device: str, strict: bool = False):
    repo_dir = Path(repo_path).expanduser().resolve()
    ckpt = Path(ckpt_path).expanduser()
    if not repo_dir.exists():
        raise FileNotFoundError(f"QCNet repo path not found: {repo_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"QCNet checkpoint not found: {ckpt}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import torch
    from predictors import QCNet

    try:
        model = QCNet.load_from_checkpoint(
            checkpoint_path=str(ckpt),
            strict=strict,
            map_location="cpu",
        )
    except TypeError:
        model = QCNet.load_from_checkpoint(checkpoint_path=str(ckpt), strict=strict)
    model.to(device)
    model.eval()
    return torch, model


def effective_window_args(
    args: argparse.Namespace, history_steps: int, future_steps: int
) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.history_sec = max(0.0, (int(history_steps) - 1) * float(args.dt))
    eval_args.future_sec = int(future_steps) * float(args.dt)
    return eval_args


def build_dataset(args: argparse.Namespace):
    from trajdata import AgentType, UnifiedDataset

    dataset = UnifiedDataset(
        desired_data=[f"sind-{args.location}"],
        data_dirs={"sind": str(args.data_dir.expanduser())},
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: float(args.neighbor_radius)),
        desired_dt=args.dt,
        centric="agent",
        history_sec=(args.history_sec, args.history_sec),
        future_sec=(args.future_sec, args.future_sec),
        incl_raster_map=False,
        incl_vector_map=True,
        vector_map_params={
            "collate": True,
            "associate_traffic_lights": False,
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        },
        verbose=True,
        num_workers=0,
    )
    scenes = list(dataset.scenes())
    print(
        f"[prediction] dataset ready: scenes={len(scenes)} samples={len(dataset)}",
        flush=True,
    )
    return dataset, scenes


def build_diffuser_dataset(args: argparse.Namespace):
    from trajdata import AgentType, UnifiedDataset

    dataset = UnifiedDataset(
        desired_data=[f"sind-{args.location}"],
        data_dirs={"sind": str(args.data_dir.expanduser())},
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: float(args.neighbor_radius)),
        desired_dt=args.dt,
        centric="agent",
        history_sec=(args.history_sec, args.history_sec),
        future_sec=(args.future_sec, args.future_sec),
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 12,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
            "use_lanelet2_maps": True,
        },
        incl_vector_map=False,
        vector_map_params={
            "collate": False,
            "associate_traffic_lights": False,
            "incl_road_lanes": True,
            "incl_road_areas": False,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        },
        verbose=True,
        num_workers=0,
    )
    scenes = list(dataset.scenes())
    print(
        f"[prediction] diffuser dataset ready: scenes={len(scenes)} samples={len(dataset)}",
        flush=True,
    )
    return dataset, scenes


def build_prediction_plans(
    data_dir: Path,
    location: str,
    history_steps: int,
    future_steps: int,
    min_displacement_m: float = 5.0,
    min_mean_speed_mps: float = 0.2,
    dt: float = 0.1,
) -> List[PredictionPlan]:
    tp_info = load_location_tp_info(data_dir, location)
    candidates: List[PredictionPlan] = []
    for scene_index, (scene_id_raw, scene_tracks) in enumerate(tp_info.items()):
        scene_id = str(scene_id_raw)
        scene_name = f"{location}_{scene_id}"
        scene_length = scene_length_from_tracks(scene_tracks)
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
            if last_frame < init_timestep + future_steps:
                continue
            if scene_length < init_timestep + future_steps + 2:
                continue
            motion = compute_track_motion(
                state,
                start_frame=init_timestep - history_steps + 1,
                end_frame=init_timestep + future_steps,
                dt=dt,
            )
            if (
                motion["displacement_m"] < min_displacement_m
                and motion["mean_speed_mps"] < min_mean_speed_mps
            ):
                continue
            duration = last_frame - first_frame + 1
            run_id = f"{scene_name}__ego{agent_id}__t{init_timestep:04d}"
            candidates.append(
                PredictionPlan(
                    location=location,
                    scene_id=scene_id,
                    scene_name=scene_name,
                    scene_index=scene_index,
                    run_id=run_id,
                    agent_id=agent_id,
                    candidate_rank=0,
                    track_duration=duration,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    init_timestep=init_timestep,
                    scene_length=scene_length,
                    displacement_m=motion["displacement_m"],
                    mean_speed_mps=motion["mean_speed_mps"],
                )
            )
    candidates.sort(
        key=lambda plan: (
            -plan.track_duration,
            plan.scene_name,
            plan.agent_id,
            plan.init_timestep,
        )
    )
    return [
        PredictionPlan(**{**asdict(plan), "candidate_rank": rank + 1})
        for rank, plan in enumerate(candidates)
    ]


def run_prediction_sample(
    *,
    plan: PredictionPlan,
    scene,
    dataset,
    model,
    torch,
    args: argparse.Namespace,
    hist_steps: int,
    fut_steps: int,
) -> Dict[str, Any]:
    from trajdata.simulation import SimulationScene

    cfg = ToolchainConfig(
        dataset=DatasetConfig(
            name="sind",
            location=plan.location,
            data_dir=str(args.data_dir),
            desired_dt=args.dt,
            use_lanelet2_maps=True,
        ),
        scenario=ScenarioConfig(
            scene_index=plan.scene_index,
            scene_name=plan.scene_name,
            init_timestep=plan.init_timestep,
            num_steps=1,
            ego_agent_name=plan.agent_id,
        ),
        simulation=SimulationConfig(
            mode="ego_closed_loop",
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            neighbor_radius=args.neighbor_radius,
        ),
        policies=PolicyConfig(ego_policy="qcnet", non_ego_policy="ground_truth"),
    )
    windowed_scene = _make_windowed_scene(scene, plan.init_timestep, cfg)
    sim_scene = SimulationScene(
        env_name=f"sind_{plan.location}_prediction_eval",
        scene_name=f"{plan.scene_name}_prediction_eval",
        scene=windowed_scene,
        dataset=dataset,
        init_timestep=plan.init_timestep,
        freeze_agents=True,
    )
    _ensure_sim_cache_defaults(sim_scene)
    obs = sim_scene.reset()
    ego_idx = _find_agent_idx(obs.agent_name, plan.agent_id)
    if ego_idx is None:
        raise ValueError(
            f"agent_id={plan.agent_id!r} is not present at timestep {plan.init_timestep}"
        )

    spec = build_qcnet_sample_spec(
        obs,
        ego_idx,
        map_radius=args.map_radius,
        hist_steps=hist_steps,
        fut_steps=fut_steps,
        dt=args.dt,
    )
    data = spec_to_heterodata(spec).to(args.device)
    with torch.no_grad():
        pred = model(data)

    pose = spec["ego_pose"]
    pred_modes_world, probs = prediction_modes_to_world(pred, pose)
    gt_world, valid_mask = future_to_world(obs, ego_idx, fut_steps)
    hist_world = history_to_world(obs, ego_idx, hist_steps)
    metrics, details = compute_prediction_metrics(
        pred_modes_world=pred_modes_world,
        probs=probs,
        gt_world=gt_world,
        valid_mask=valid_mask,
        max_guesses=args.max_guesses,
        miss_threshold=args.miss_threshold,
    )
    valid_steps = int(valid_mask.sum())
    prediction = {
        "run_id": plan.run_id,
        "location": plan.location,
        "scene_name": plan.scene_name,
        "scene_index": plan.scene_index,
        "agent_id": plan.agent_id,
        "init_timestep": plan.init_timestep,
        "valid_future_steps": valid_steps,
        "dt": args.dt,
        "mode_probabilities": probs.tolist(),
        "top1_mode_index": details["top1_mode_index"],
        "best_mode_index": details["best_mode_index"],
        "history_xy": hist_world.tolist(),
        "ground_truth_xy": gt_world[:valid_steps].tolist(),
        "top1_prediction_xy": pred_modes_world[
            details["top1_mode_index"], :valid_steps
        ].tolist(),
        "best_prediction_xy": pred_modes_world[
            details["best_mode_index"], :valid_steps
        ].tolist(),
    }
    return {
        "metrics": metrics,
        "prediction": prediction,
        "valid_future_steps": valid_steps,
        **details,
    }


def run_diffuser_prediction_sample(
    *,
    plan: PredictionPlan,
    scene,
    dataset,
    policy,
    args: argparse.Namespace,
    fut_steps: int,
    num_samples: int = 1,
) -> Dict[str, Any]:
    from trajdata.simulation import SimulationScene

    cfg = ToolchainConfig(
        dataset=DatasetConfig(
            name="sind",
            location=plan.location,
            data_dir=str(args.data_dir),
            desired_dt=args.dt,
            use_lanelet2_maps=True,
        ),
        scenario=ScenarioConfig(
            scene_index=plan.scene_index,
            scene_name=plan.scene_name,
            init_timestep=plan.init_timestep,
            num_steps=1,
            ego_agent_name=plan.agent_id,
        ),
        simulation=SimulationConfig(
            mode="ego_closed_loop",
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            neighbor_radius=args.neighbor_radius,
        ),
        policies=PolicyConfig(ego_policy="diffuser", non_ego_policy="ground_truth"),
    )
    windowed_scene = _make_windowed_scene(scene, plan.init_timestep, cfg)
    sim_scene = SimulationScene(
        env_name=f"sind_{plan.location}_diffuser_prediction_eval",
        scene_name=f"{plan.scene_name}_diffuser_prediction_eval",
        scene=windowed_scene,
        dataset=dataset,
        init_timestep=plan.init_timestep,
        freeze_agents=True,
    )
    _ensure_sim_cache_defaults(sim_scene)
    obs = sim_scene.reset()
    ego_idx = _find_agent_idx(obs.agent_name, plan.agent_id)
    if ego_idx is None:
        raise ValueError(
            f"agent_id={plan.agent_id!r} is not present at timestep {plan.init_timestep}"
        )

    batch = policy._build_batch(obs, ego_idx)
    with policy.torch.no_grad():
        pred = policy.model(
            batch,
            num_samp=num_samples,
            return_diffusion=False,
            return_guidance_losses=False,
            apply_guidance=False,
        )
    pose = get_agent_world_pose(obs, ego_idx)
    local_modes = pred["predictions"]["positions"][0]
    if hasattr(local_modes, "detach"):
        local_modes = local_modes.detach().cpu().numpy()
    local_modes = np.asarray(local_modes, dtype=np.float32)
    pred_modes_world = pose["position"][:2] + local_modes[:, :, :2] @ pose["rotation"].T
    pred_modes_world = pred_modes_world[:, :fut_steps]
    probs = np.ones((pred_modes_world.shape[0],), dtype=np.float32)
    probs = probs / max(float(probs.sum()), 1.0)
    gt_world, valid_mask = future_to_world(obs, ego_idx, fut_steps)
    hist_world = history_to_world(
        obs, ego_idx, min(31, int(round(args.history_sec / args.dt)) + 1)
    )
    metrics, details = compute_prediction_metrics(
        pred_modes_world=pred_modes_world,
        probs=probs,
        gt_world=gt_world,
        valid_mask=valid_mask,
        max_guesses=min(num_samples, args.max_guesses),
        miss_threshold=args.miss_threshold,
    )
    valid_steps = int(valid_mask.sum())
    prediction = {
        "run_id": plan.run_id,
        "location": plan.location,
        "scene_name": plan.scene_name,
        "scene_index": plan.scene_index,
        "agent_id": plan.agent_id,
        "init_timestep": plan.init_timestep,
        "valid_future_steps": valid_steps,
        "dt": args.dt,
        "mode_probabilities": probs.tolist(),
        "top1_mode_index": details["top1_mode_index"],
        "best_mode_index": details["best_mode_index"],
        "history_xy": hist_world.tolist(),
        "ground_truth_xy": gt_world[:valid_steps].tolist(),
        "top1_prediction_xy": pred_modes_world[
            details["top1_mode_index"], :valid_steps
        ].tolist(),
        "best_prediction_xy": pred_modes_world[
            details["best_mode_index"], :valid_steps
        ].tolist(),
    }
    return {
        "metrics": metrics,
        "prediction": prediction,
        "valid_future_steps": valid_steps,
        **details,
    }


def compute_track_motion(
    state,
    *,
    start_frame: int,
    end_frame: int,
    dt: float,
) -> Dict[str, float]:
    window = state[
        (state["frame_id"] >= start_frame) & (state["frame_id"] <= end_frame)
    ].sort_values("frame_id")
    if window.empty or not {"x", "y"}.issubset(window.columns):
        return {"displacement_m": 0.0, "mean_speed_mps": 0.0}
    xy = window[["x", "y"]].to_numpy(dtype=float)
    displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    if len(xy) < 2:
        mean_speed = 0.0
    elif "speed" in window.columns:
        speed = np.asarray(window["speed"].to_numpy(dtype=float), dtype=float)
        mean_speed = (
            float(np.nanmean(np.abs(speed))) if np.isfinite(speed).any() else 0.0
        )
    else:
        step_dist = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        mean_speed = float(np.nanmean(step_dist) / max(dt, 1e-6))
    return {"displacement_m": displacement, "mean_speed_mps": mean_speed}


def prediction_modes_to_world(
    pred: Mapping[str, Any], pose
) -> Tuple[np.ndarray, np.ndarray]:
    loc = pred["loc_refine_pos"][0, :, :, :2]
    pi = pred["pi"][0]
    if hasattr(loc, "detach"):
        loc = loc.detach().cpu().numpy()
    if hasattr(pi, "detach"):
        pi = pi.detach().cpu().numpy()
    loc = np.asarray(loc, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)
    probs = softmax(pi)
    world = pose.position[:2] + loc @ pose.rotation.T
    return world.astype(np.float32, copy=False), probs.astype(np.float32, copy=False)


def future_to_world(obs, ego_idx: int, fut_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    pose = get_agent_world_pose(obs, ego_idx)
    fut_len = int(to_numpy(obs.agent_fut_len[ego_idx]).reshape(-1)[0])
    fut_len = max(0, min(fut_len, fut_steps))
    gt = np.zeros((fut_steps, 2), dtype=np.float32)
    valid = np.zeros((fut_steps,), dtype=bool)
    if fut_len == 0:
        return gt, valid
    fut_pos = to_numpy(obs.agent_fut[ego_idx, :fut_len].position)
    fut_pos = np.asarray(fut_pos[:, :2], dtype=np.float32)
    gt[:fut_len] = fut_pos @ pose["rotation"].T + pose["position"][:2]
    valid[:fut_len] = np.isfinite(gt[:fut_len]).all(axis=-1)
    return gt, valid


def history_to_world(obs, ego_idx: int, hist_steps: int) -> np.ndarray:
    pose = get_agent_world_pose(obs, ego_idx)
    hist_len = int(to_numpy(obs.agent_hist_len[ego_idx]).reshape(-1)[0])
    hist_len = max(0, min(hist_len, hist_steps))
    if hist_len == 0:
        return np.zeros((0, 2), dtype=np.float32)
    hist_pos = to_numpy(obs.agent_hist[ego_idx, :hist_len].position)
    hist_pos = np.asarray(hist_pos[:, :2], dtype=np.float32)
    hist_world = hist_pos @ pose["rotation"].T + pose["position"][:2]
    return hist_world[np.isfinite(hist_world).all(axis=-1)].astype(
        np.float32, copy=False
    )


def compute_prediction_metrics(
    *,
    pred_modes_world: np.ndarray,
    probs: np.ndarray,
    gt_world: np.ndarray,
    valid_mask: np.ndarray,
    max_guesses: int = 6,
    miss_threshold: float = 2.0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    pred_modes_world = np.asarray(pred_modes_world, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    gt_world = np.asarray(gt_world, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_mask = valid_mask & np.isfinite(gt_world).all(axis=-1)
    if valid_mask.sum() == 0:
        raise ValueError("No valid future points available for metric computation.")

    pred_eval = pred_modes_world[:, valid_mask]
    gt_eval = gt_world[valid_mask]
    if not np.isfinite(pred_eval).all():
        raise ValueError("QCNet returned non-finite prediction coordinates.")

    distances = np.linalg.norm(pred_eval - gt_eval[None, :, :], axis=-1)
    ade_by_mode = distances.mean(axis=-1)
    fde_by_mode = distances[:, -1]

    top1_mode = int(np.nanargmax(probs)) if np.isfinite(probs).any() else 0
    sorted_modes = np.argsort(-np.nan_to_num(probs, nan=-np.inf))
    topk_modes = sorted_modes[: max(1, min(max_guesses, pred_modes_world.shape[0]))]
    best_mode = int(topk_modes[np.argmin(fde_by_mode[topk_modes])])

    top1_fde = float(fde_by_mode[top1_mode])
    best_fde = float(fde_by_mode[best_mode])
    metrics = {
        "top1_ADE": float(ade_by_mode[top1_mode]),
        "top1_FDE": top1_fde,
        "top1_MR": float(top1_fde > miss_threshold),
        "minADE": float(ade_by_mode[best_mode]),
        "minFDE": best_fde,
        "MR": float(best_fde > miss_threshold),
    }
    details = {
        "top1_mode_index": top1_mode,
        "best_mode_index": best_mode,
        "top1_mode_prob": float(probs[top1_mode]) if probs.size else float("nan"),
        "best_mode_prob": float(probs[best_mode]) if probs.size else float("nan"),
    }
    return metrics, details


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = np.nan_to_num(values, nan=-np.inf)
    max_value = np.max(finite)
    if not np.isfinite(max_value):
        return np.ones_like(values, dtype=np.float32) / max(1, values.size)
    exp_values = np.exp(finite - max_value)
    denom = float(exp_values.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(values, dtype=np.float32) / max(1, values.size)
    return exp_values / denom


def base_row(plan: PredictionPlan) -> Dict[str, Any]:
    row = asdict(plan)
    row.update(
        {
            "status": "pending",
            "error": "",
            "elapsed_s": "",
            "valid_future_steps": "",
            "top1_mode_index": "",
            "best_mode_index": "",
            "top1_mode_prob": "",
            "best_mode_prob": "",
            "top1_ADE": "",
            "top1_FDE": "",
            "top1_MR": "",
            "minADE": "",
            "minFDE": "",
            "MR": "",
        }
    )
    return row


def write_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    elapsed_s: float,
) -> None:
    write_csv(output_dir / "prediction_metrics.csv", rows)
    summary = summarize_rows(rows)
    write_csv(output_dir / "summary.csv", [summary])
    payload = {
        "metadata": {
            "experiment_mode": "qcnet_prediction_eval",
            "location": args.location,
            "num_samples_requested": args.num_samples,
            "history_sec": args.history_sec,
            "future_sec": args.future_sec,
            "dt": args.dt,
            "neighbor_radius": args.neighbor_radius,
            "map_radius": args.map_radius,
            "miss_threshold": args.miss_threshold,
            "max_guesses": args.max_guesses,
            "device": args.device,
            "qcnet_repo_path": args.qcnet_repo_path,
            "qcnet_ckpt_path": args.qcnet_ckpt_path,
            "elapsed_s": elapsed_s,
        },
        "summary": summary,
        "rows": list(rows),
        "predictions": list(predictions),
    }
    (output_dir / "prediction_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.no_html and predictions:
        save_prediction_html(
            predictions=list(predictions),
            rows=[dict(row) for row in rows],
            summary=summary,
            path=output_dir / "prediction_visualization.html",
        )


def save_prediction_html(
    *,
    predictions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    path: Path,
) -> None:
    row_by_run = {str(row.get("run_id")): row for row in rows}
    run_ids = [str(pred["run_id"]) for pred in predictions]
    data = {
        "run_id": [],
        "scene_name": [],
        "agent_id": [],
        "init_timestep": [],
        "history_x": [],
        "history_y": [],
        "gt_x": [],
        "gt_y": [],
        "top1_x": [],
        "top1_y": [],
        "best_x": [],
        "best_y": [],
        "summary_html": [],
    }
    for pred in predictions:
        run_id = str(pred["run_id"])
        row_data = row_by_run.get(run_id, {})
        hist = _xy_array(pred.get("history_xy", []))
        gt = _xy_array(pred.get("ground_truth_xy", []))
        top1 = _xy_array(pred.get("top1_prediction_xy", []))
        best = _xy_array(pred.get("best_prediction_xy", []))
        data["run_id"].append(run_id)
        data["scene_name"].append(str(pred.get("scene_name", "")))
        data["agent_id"].append(str(pred.get("agent_id", "")))
        data["init_timestep"].append(int(pred.get("init_timestep", 0)))
        data["history_x"].append(hist[:, 0].tolist())
        data["history_y"].append(hist[:, 1].tolist())
        data["gt_x"].append(gt[:, 0].tolist())
        data["gt_y"].append(gt[:, 1].tolist())
        data["top1_x"].append(top1[:, 0].tolist())
        data["top1_y"].append(top1[:, 1].tolist())
        data["best_x"].append(best[:, 0].tolist())
        data["best_y"].append(best[:, 1].tolist())
        data["summary_html"].append(_sample_html(pred, row_data))

    xs = [
        value
        for key in ("history_x", "gt_x", "top1_x", "best_x")
        for line in data[key]
        for value in line
    ]
    ys = [
        value
        for key in ("history_y", "gt_y", "top1_y", "best_y")
        for line in data[key]
        for value in line
    ]
    margin = 8.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    source = ColumnDataSource(data)
    selected = _slice_prediction_data(data, 0)
    selected_source = ColumnDataSource(selected)
    endpoints_source = ColumnDataSource(
        _endpoint_data(
            data["gt_x"][0],
            data["gt_y"][0],
            data["top1_x"][0],
            data["top1_y"][0],
            data["best_x"][0],
            data["best_y"][0],
        )
    )

    fig = figure(
        width=1100,
        height=760,
        match_aspect=True,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        title="QCNet 6s Prediction Evaluation",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.background_fill_color = "#fbfbfb"
    fig.grid.grid_line_alpha = 0.25
    fig.toolbar.active_scroll = next(
        (tool for tool in fig.tools if isinstance(tool, WheelZoomTool)), None
    )
    lines = fig.multi_line(
        xs="xs",
        ys="ys",
        color="color",
        line_width="line_width",
        line_dash="line_dash",
        source=selected_source,
    )
    circles = fig.circle(
        x="x",
        y="y",
        size=8,
        color="color",
        source=endpoints_source,
    )
    fig.add_tools(
        HoverTool(
            renderers=[circles],
            tooltips=[("point", "@label"), ("xy", "(@x{0.00}, @y{0.00})")],
        )
    )
    legend = Legend(
        items=[
            LegendItem(label="history / GT / predictions", renderers=[lines]),
            LegendItem(label="final points", renderers=[circles]),
        ],
        location="top_left",
        click_policy="hide",
    )
    fig.add_layout(legend)

    select = Select(title="Sample", value=run_ids[0], options=run_ids, width=760)
    info_div = Div(text=data["summary_html"][0], width=380)
    summary_div = Div(text=_overall_html(summary), width=700)
    select.js_on_change(
        "value",
        CustomJS(
            args=dict(
                source=source,
                selected_source=selected_source,
                endpoints_source=endpoints_source,
                info_div=info_div,
            ),
            code="""
const runId = cb_obj.value;
const data = source.data;
const idx = data['run_id'].indexOf(runId);
if (idx < 0) { return; }
selected_source.data = {
  xs: [
    data['history_x'][idx],
    data['gt_x'][idx],
    data['top1_x'][idx],
    data['best_x'][idx],
  ],
  ys: [
    data['history_y'][idx],
    data['gt_y'][idx],
    data['top1_y'][idx],
    data['best_y'][idx],
  ],
  label: ['history', 'ground truth', 'top-1', 'min-of-6'],
  color: ['#4c566a', '#2ca02c', '#d62728', '#1f77b4'],
  line_width: [3, 3, 2, 2],
  line_dash: ['solid', 'solid', 'dashed', 'dotdash'],
};
function endpoint(xs, ys) {
  if (!xs || xs.length === 0) { return [NaN, NaN]; }
  return [xs[xs.length - 1], ys[ys.length - 1]];
}
const gt = endpoint(data['gt_x'][idx], data['gt_y'][idx]);
const top1 = endpoint(data['top1_x'][idx], data['top1_y'][idx]);
const best = endpoint(data['best_x'][idx], data['best_y'][idx]);
endpoints_source.data = {
  x: [gt[0], top1[0], best[0]],
  y: [gt[1], top1[1], best[1]],
  label: ['GT final', 'top-1 final', 'min-of-6 final'],
  color: ['#2ca02c', '#d62728', '#1f77b4'],
};
info_div.text = data['summary_html'][idx];
""",
        ),
    )

    output_file(path, title="QCNet 6s Prediction Evaluation")
    save(column(row(select, summary_div), row(fig, info_div)))


def _xy_array(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return arr.reshape(-1, 2)


def _slice_prediction_data(
    data: Mapping[str, List[Any]], idx: int
) -> Dict[str, List[Any]]:
    return {
        "xs": [
            data["history_x"][idx],
            data["gt_x"][idx],
            data["top1_x"][idx],
            data["best_x"][idx],
        ],
        "ys": [
            data["history_y"][idx],
            data["gt_y"][idx],
            data["top1_y"][idx],
            data["best_y"][idx],
        ],
        "label": ["history", "ground truth", "top-1", "min-of-6"],
        "color": ["#4c566a", "#2ca02c", "#d62728", "#1f77b4"],
        "line_width": [3, 3, 2, 2],
        "line_dash": ["solid", "solid", "dashed", "dotdash"],
    }


def _endpoint_data(
    gt_x: Sequence[float],
    gt_y: Sequence[float],
    top1_x: Sequence[float],
    top1_y: Sequence[float],
    best_x: Sequence[float],
    best_y: Sequence[float],
) -> Dict[str, List[Any]]:
    def endpoint(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
        if not xs or not ys:
            return float("nan"), float("nan")
        return float(xs[-1]), float(ys[-1])

    gt = endpoint(gt_x, gt_y)
    top1 = endpoint(top1_x, top1_y)
    best = endpoint(best_x, best_y)
    return {
        "x": [gt[0], top1[0], best[0]],
        "y": [gt[1], top1[1], best[1]],
        "label": ["GT final", "top-1 final", "min-of-6 final"],
        "color": ["#2ca02c", "#d62728", "#1f77b4"],
    }


def _sample_html(pred: Mapping[str, Any], row_data: Mapping[str, Any]) -> str:
    fields = [
        ("run", pred.get("run_id", "")),
        ("scene", pred.get("scene_name", "")),
        ("agent", pred.get("agent_id", "")),
        ("timestep", pred.get("init_timestep", "")),
        ("valid future steps", pred.get("valid_future_steps", "")),
        ("top1 mode", pred.get("top1_mode_index", "")),
        ("best mode", pred.get("best_mode_index", "")),
        ("top1 ADE", _fmt(row_data.get("top1_ADE"))),
        ("top1 FDE", _fmt(row_data.get("top1_FDE"))),
        ("minADE", _fmt(row_data.get("minADE"))),
        ("minFDE", _fmt(row_data.get("minFDE"))),
    ]
    lines = ["<b>Sample</b>"]
    lines.extend(f"{name}: {value}" for name, value in fields)
    return "<br>".join(lines)


def _overall_html(summary: Mapping[str, Any]) -> str:
    return (
        "<b>Summary</b><br>"
        f"completed: {summary.get('completed_samples', 0)} / {summary.get('attempted_samples', 0)}<br>"
        f"top1 ADE mean: {_fmt(summary.get('top1_ADE_mean'))}<br>"
        f"top1 FDE mean: {_fmt(summary.get('top1_FDE_mean'))}<br>"
        f"minADE mean: {_fmt(summary.get('minADE_mean'))}<br>"
        f"minFDE mean: {_fmt(summary.get('minFDE_mean'))}<br>"
        f"MR rate: {_fmt(summary.get('MR_rate'))}"
    )


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    summary: Dict[str, Any] = {
        "attempted_samples": len(rows),
        "completed_samples": len(completed),
        "failed_samples": sum(1 for row in rows if row.get("status") == "failed"),
    }
    for metric in ("top1_ADE", "top1_FDE", "minADE", "minFDE"):
        values = numeric_values(completed, metric)
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else np.nan
        summary[f"{metric}_median"] = float(np.median(values)) if values else np.nan
    for metric in ("top1_MR", "MR"):
        values = numeric_values(completed, metric)
        summary[f"{metric}_rate"] = float(np.mean(values)) if values else np.nan
    return summary


def numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_location_tp_info(data_dir: Path, location: str) -> Mapping[str, Any]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def scene_length_from_tracks(scene_tracks: Mapping[Any, Mapping[str, Any]]) -> int:
    max_frame = 0
    for tp_data in scene_tracks.values():
        state = tp_data.get("State")
        if state is not None and not state.empty and "frame_id" in state.columns:
            max_frame = max(max_frame, int(state["frame_id"].max()))
    return max_frame + 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
