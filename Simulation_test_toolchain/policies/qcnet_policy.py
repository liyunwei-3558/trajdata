from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose

from .base import BasePolicy, PolicyAction, PolicyState
from .qcnet_adapter import (
    build_qcnet_action,
    build_qcnet_sample_spec,
    spec_to_heterodata,
)


class QcnetPolicy(BasePolicy):
    policy_name = "qcnet"

    def __init__(
        self,
        dt: float = 0.1,
        ckpt_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        device: Optional[str] = None,
        map_radius: float = 150.0,
        step_index: int = 0,
        prediction_interval_steps: int = 10,
        execute_top1_cached_trajectory: bool = False,
        initial_velocity_override_mps: Optional[float] = None,
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        if ckpt_path is None:
            raise ValueError(
                "QCNet requires checkpoints.qcnet_ckpt_path or policies.<role>.ckpt_path."
            )
        ckpt = Path(ckpt_path).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"QCNet checkpoint not found: {ckpt}")

        repo = repo_path or os.environ.get("QCNET_REPO_PATH")
        if repo is None:
            raise ValueError(
                "QCNet requires policies.<role>.repo_path, checkpoints.qcnet_repo_path, "
                "or QCNET_REPO_PATH."
            )
        repo_dir = Path(repo).expanduser()
        if not repo_dir.exists():
            raise FileNotFoundError(f"QCNet repo_path not found: {repo_dir}")

        try:
            import torch
        except Exception as exc:
            raise ImportError(
                "QCNet policy requires torch in the active conda environment."
            ) from exc

        self.torch = torch
        self.ckpt_path = str(ckpt)
        self.repo_path = str(repo_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.map_radius = float(map_radius)
        self.step_index = int(step_index)
        self.prediction_interval_steps = max(1, int(prediction_interval_steps))
        self.execute_top1_cached_trajectory = bool(execute_top1_cached_trajectory)
        self.initial_velocity_override_mps = (
            None
            if initial_velocity_override_mps is None
            else float(initial_velocity_override_mps)
        )
        self._cached_world_xy = np.zeros((0, 2), dtype=float)
        self._cached_step = 0
        self._cached_speed_scale = 1.0
        self.strict = bool(strict)
        self.model = self._load_model().to(self.device)
        self.model.eval()

        if (
            int(getattr(self.model, "input_dim", 2)) != 2
            or int(getattr(self.model, "output_dim", 2)) != 2
        ):
            raise ValueError(
                "QCNet policy wrapper currently supports 2D input_dim/output_dim checkpoints."
            )

    def _load_model(self):
        repo_path = str(Path(self.repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from predictors import QCNet
        except Exception as exc:
            raise ImportError(
                "Could not import QCNet from repo_path. Activate a QCNet-compatible env "
                "or install QCNet dependencies such as torch_geometric and torch_cluster."
            ) from exc

        try:
            return QCNet.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                strict=self.strict,
                map_location="cpu",
            )
        except TypeError:
            return QCNet.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                strict=self.strict,
            )

    def reset(self, obs, ego_idx: int = 0) -> PolicyState:
        self.state = PolicyState(
            agent_name=str(obs.agent_name[ego_idx]),
            dt=self.dt,
            initialized=True,
        )
        self._cached_world_xy = np.zeros((0, 2), dtype=float)
        self._cached_step = 0
        self._cached_speed_scale = 1.0
        return self.state

    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        if self.state is None:
            self.reset(obs, ego_idx)

        if self.execute_top1_cached_trajectory and self._has_cached_step():
            return self._cached_action(obs, ego_idx)

        spec = build_qcnet_sample_spec(
            obs,
            ego_idx,
            map_radius=self.map_radius,
            hist_steps=int(getattr(self.model, "num_historical_steps")),
            fut_steps=int(getattr(self.model, "num_future_steps")),
            dt=self.dt,
        )
        data = spec_to_heterodata(spec).to(self.device)

        with self.torch.no_grad():
            pred = self.model(data)

        if self.execute_top1_cached_trajectory:
            self._cached_world_xy = _top1_world_trajectory(pred, spec)
            self._cached_speed_scale = 1.0
            if self.initial_velocity_override_mps is not None:
                pose = get_agent_world_pose(obs, ego_idx)
                self._cached_world_xy, self._cached_speed_scale = (
                    _apply_initial_speed_override(
                        self._cached_world_xy,
                        np.asarray(pose["position"], dtype=float),
                        float(self.initial_velocity_override_mps),
                        self.dt,
                    )
                )
            self._cached_step = 0
            if self._has_cached_step():
                return self._cached_action(obs, ego_idx, used_new_prediction=True)

        xyh, command = build_qcnet_action(
            pred, spec, ego_idx=0, step_index=self.step_index
        )
        command = {
            **command,
            "device": str(self.device),
            "prediction_interval_steps": self.prediction_interval_steps,
            "used_cached_trajectory": False,
        }
        if self.initial_velocity_override_mps is not None:
            command["initial_velocity_override_mps"] = self.initial_velocity_override_mps
            pose = get_agent_world_pose(obs, ego_idx)
            scaled_xy, scale = _apply_initial_speed_override(
                np.asarray([xyh[:2]], dtype=float),
                np.asarray(pose["position"], dtype=float),
                float(self.initial_velocity_override_mps),
                self.dt,
            )
            xyh = np.array(
                [scaled_xy[0, 0], scaled_xy[0, 1], xyh[2]],
                dtype=float,
            )
            command["speed_scale"] = float(scale)
        if not np.isfinite(xyh).all():
            pose = get_agent_world_pose(obs, ego_idx)
            xyh = np.array(
                [pose["position"][0], pose["position"][1], pose["heading"]], dtype=float
            )
            command = {
                **command,
                "fallback": "hold_current_pose",
                "fallback_reason": "QCNet returned non-finite xyh",
            }

        self.last_command = command.copy()
        return PolicyAction(xyh=xyh, command=self.last_command.copy())

    def _has_cached_step(self) -> bool:
        max_steps = min(self.prediction_interval_steps, len(self._cached_world_xy))
        return self._cached_step < max_steps

    def _cached_action(
        self,
        obs,
        ego_idx: int,
        used_new_prediction: bool = False,
    ) -> PolicyAction:
        pose = get_agent_world_pose(obs, ego_idx)
        curr_xy = np.asarray(pose["position"], dtype=float)
        next_xy = np.asarray(self._cached_world_xy[self._cached_step], dtype=float)
        heading = _heading_from_points(
            curr_xy,
            next_xy,
            float(pose["heading"]),
            self._cached_world_xy,
            self._cached_step,
        )
        xyh = np.array([next_xy[0], next_xy[1], heading], dtype=float)
        command = {
            "policy": self.policy_name,
            "device": str(self.device),
            "used_cached_trajectory": not used_new_prediction,
            "prediction_interval_steps": self.prediction_interval_steps,
            "cached_step": int(self._cached_step),
            "cached_horizon_steps": int(len(self._cached_world_xy)),
            "mode": "top1",
        }
        if self.initial_velocity_override_mps is not None:
            command["initial_velocity_override_mps"] = self.initial_velocity_override_mps
            command["cached_speed_scale"] = float(self._cached_speed_scale)
        self._cached_step += 1
        if not np.isfinite(xyh).all():
            xyh = np.array(
                [pose["position"][0], pose["position"][1], pose["heading"]],
                dtype=float,
            )
            command = {
                **command,
                "fallback": "hold_current_pose",
                "fallback_reason": "cached QCNet trajectory returned non-finite xyh",
            }
        self.last_command = command.copy()
        return PolicyAction(xyh=xyh, command=self.last_command.copy())


def _top1_world_trajectory(pred, spec) -> np.ndarray:
    pred_np = _tensor_to_numpy(pred)
    pi = np.asarray(pred_np["pi"])
    mode_idx = int(np.argmax(pi[0] if pi.ndim == 2 else pi))
    loc = np.asarray(pred_np["loc_refine_pos"])
    if loc.ndim == 4:
        local_xy = loc[0, mode_idx]
    elif loc.ndim == 3:
        local_xy = loc[mode_idx]
    else:
        return np.zeros((0, 2), dtype=float)
    pose = spec["ego_pose"]
    world_xy = np.asarray(local_xy, dtype=float) @ pose.rotation.T + pose.position
    if not np.isfinite(world_xy).all():
        return np.zeros((0, 2), dtype=float)
    return world_xy


def _apply_initial_speed_override(
    world_xy: np.ndarray,
    curr_xy: np.ndarray,
    target_speed_mps: float,
    dt: float,
) -> Tuple[np.ndarray, float]:
    if world_xy.size == 0 or not np.isfinite(target_speed_mps) or target_speed_mps <= 0:
        return world_xy, 1.0
    target_step = float(target_speed_mps) * float(dt)
    first_delta = np.asarray(world_xy[0], dtype=float) - curr_xy
    first_dist = float(np.linalg.norm(first_delta))
    if first_dist <= 1e-6 or target_step <= 0:
        return world_xy, 1.0
    scale = target_step / first_dist
    if not np.isfinite(scale) or scale <= 0:
        return world_xy, 1.0
    scaled = curr_xy + (np.asarray(world_xy, dtype=float) - curr_xy) * scale
    if not np.isfinite(scaled).all():
        return world_xy, 1.0
    return scaled, float(scale)


def _tensor_to_numpy(value):
    if isinstance(value, dict):
        return {key: _tensor_to_numpy(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _heading_from_points(
    curr_xy: np.ndarray,
    next_xy: np.ndarray,
    fallback_heading: float,
    trajectory: np.ndarray,
    step: int,
) -> float:
    delta = next_xy - curr_xy
    if np.linalg.norm(delta) <= 1e-6 and step + 1 < len(trajectory):
        delta = np.asarray(trajectory[step + 1], dtype=float) - next_xy
    if np.linalg.norm(delta) <= 1e-6 and step > 0:
        delta = next_xy - np.asarray(trajectory[step - 1], dtype=float)
    if np.linalg.norm(delta) <= 1e-6:
        return float(fallback_heading)
    return float(np.arctan2(delta[1], delta[0]))
