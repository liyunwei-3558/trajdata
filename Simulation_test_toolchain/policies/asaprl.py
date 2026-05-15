from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from Simulation_test_toolchain.core.state_utils import (
    get_agent_world_pose,
    get_ground_truth_next_xyh,
    to_numpy,
)
from .base import BasePolicy, PolicyAction, PolicyState
from .motion_skill_model import motion_skill_model
from .risk_idm import risk_idm_formula
from .ttc_utils import compute_min_ttc


@dataclass
class ASAPRLState(PolicyState):
    velocity: float = 0.0
    acceleration: float = 0.0
    action_step: int = 0
    cached_acceleration: float = 0.0
    cached_yaw_rate: float = 0.0
    has_cached_control: bool = False


class ASAPRLPolicy(BasePolicy):
    policy_name = "asaprl"

    def __init__(
        self,
        dt: float = 0.1,
        target_speed: float = 7.5,
        horizon: float = 3.0,
        ckpt_path: Optional[str] = None,
        use_risk_idm: bool = True,
        neighbor_radius: float = 50.0,
        inference_interval_steps: int = 5,
        action_lat_scale: float = 1.0,
        action_yaw_scale: float = 2.0,
        action_speed_scale: float = 3.0,
        observation_px_per_m: float = 3.0,
        reference_heading_blend: float = 0.8,
        reference_speed_blend: float = 0.0,
        max_yaw_rate: float = 0.3,
        max_speed: float = 10.0,
        follow_reference_direction: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        if ckpt_path is None:
            raise ValueError("ASAPRL requires checkpoints.asaprl_ckpt_path.")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"ASAPRL checkpoint not found: {ckpt_path}")

        try:
            import torch
        except Exception as exc:
            raise ImportError(
                "ASAPRL requires a working torch installation in the active environment."
            ) from exc

        self.torch = torch
        self.target_speed = target_speed
        self.horizon = horizon
        self.ckpt_path = ckpt_path
        self.use_risk_idm = use_risk_idm
        self.neighbor_radius = neighbor_radius
        self.inference_interval_steps = max(1, int(inference_interval_steps))
        self.action_lat_scale = float(action_lat_scale)
        self.action_yaw_scale = float(action_yaw_scale)
        self.action_speed_scale = float(action_speed_scale)
        self.observation_px_per_m = float(observation_px_per_m)
        self.reference_heading_blend = float(np.clip(reference_heading_blend, 0.0, 1.0))
        self.reference_speed_blend = float(np.clip(reference_speed_blend, 0.0, 1.0))
        self.max_yaw_rate = max(0.0, float(max_yaw_rate))
        self.max_speed = max(0.0, float(max_speed))
        self.follow_reference_direction = bool(follow_reference_direction)
        self.model = self._load_actor()

    def _load_actor(self):
        torch = self.torch

        class ReparameterizationHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Sequential(torch.nn.Linear(64, 64))
                self.mu = torch.nn.Linear(64, 3)
                self.log_sigma_layer = torch.nn.Linear(64, 3)

            def forward(self, x):
                x = torch.relu(self.main(x))
                return {"mu": self.mu(x), "sigma": torch.exp(self.log_sigma_layer(x))}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 128, 3, 2),
                    torch.nn.Conv2d(128, 128, 3, 2),
                    torch.nn.Conv2d(128, 64, 3, 2),
                    torch.nn.Flatten(),
                )
                self.mid = torch.nn.Linear(64 * 24 * 24, 64)

            def forward(self, x):
                return self.mid(self.main(x))

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Sequential(
                    Encoder(),
                    torch.nn.ReLU(),
                    ReparameterizationHead(),
                )

            def forward(self, x):
                out = self.actor(x)
                return torch.tanh(out["mu"])

        model = Actor()
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        actor_state = {}
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key] = value
        missing, unexpected = model.load_state_dict(actor_state, strict=False)
        if len(actor_state) == 0:
            raise RuntimeError(f"No actor weights found in ASAPRL checkpoint {self.ckpt_path}")
        critical_missing = [
            key for key in missing if key.startswith("actor.0") or key.startswith("actor.2.mu")
        ]
        if critical_missing:
            raise RuntimeError(
                "ASAPRL checkpoint did not match the expected actor layout. "
                f"Missing keys include: {critical_missing[:5]}"
            )
        model.eval()
        return model

    def reset(self, obs, ego_idx: int = 0) -> ASAPRLState:
        info = get_agent_world_pose(obs, ego_idx)
        self.state = ASAPRLState(
            agent_name=obs.agent_name[ego_idx],
            dt=self.dt,
            initialized=True,
            velocity=float(np.linalg.norm(info["velocity"])),
        )
        return self.state

    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        if self.state is None:
            self.reset(obs, ego_idx)

        info = get_agent_world_pose(obs, ego_idx)
        self.state.velocity = float(np.nan_to_num(self.state.velocity, nan=0.0))
        self.state.acceleration = float(np.nan_to_num(self.state.acceleration, nan=0.0))
        should_infer = (
            not self.state.has_cached_control
            or self.state.action_step % self.inference_interval_steps == 0
        )
        if not should_infer:
            return self._repeat_cached_control(info)

        image = self._build_observation_image(obs, ego_idx)
        with self.torch.no_grad():
            tensor = self.torch.tensor(image[None], dtype=self.torch.float32)
            latent = self.model(tensor).cpu().numpy()[0]
        latent = np.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=-1.0)

        raw_latent = latent.copy()
        lat1 = float(latent[0] * self.action_lat_scale)
        yaw1 = float(latent[1] * self.action_yaw_scale)
        v1 = float((latent[2] + 1.0) * 0.5 * self.action_speed_scale)
        traj, lat1, yaw1, v1 = motion_skill_model(
            lat1, yaw1, self.state.velocity, self.state.acceleration, v1, self.horizon
        )
        next_lon, next_lat, next_v, next_yaw_deg = traj[min(1, len(traj) - 1)]
        heading = float(info["heading"])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        next_pos = info["position"] + np.array(
            [next_lon * cos_h - next_lat * sin_h, next_lon * sin_h + next_lat * cos_h]
        )
        next_heading = heading + np.deg2rad(next_yaw_deg)
        raw_yaw_rate = (next_heading - heading) / self.dt
        reference_xyh = get_ground_truth_next_xyh(obs, ego_idx)
        reference_yaw_rate = _angle_diff(float(reference_xyh[2]), heading) / self.dt
        yaw_rate = _blend_and_clip_yaw_rate(
            raw_yaw_rate,
            reference_yaw_rate,
            blend=self.reference_heading_blend,
            limit=self.max_yaw_rate,
        )
        next_heading = heading + yaw_rate * self.dt
        reference_speed = float(
            np.linalg.norm(np.asarray(reference_xyh[:2], dtype=float) - info["position"])
            / max(self.dt, 1e-6)
        )

        acceleration = (next_v - self.state.velocity) / self.dt
        if self.use_risk_idm:
            ttc, point, _ = compute_min_ttc(obs, ego_idx, self.neighbor_radius)
            se = (
                float(np.linalg.norm(point - info["position"]))
                if point is not None
                else float("inf")
            )
            acceleration = risk_idm_formula(
                self.state.velocity,
                se,
                ttc,
                v_0=self.target_speed,
                a_max=5.0,
                a_min=-5.0,
            )
            if self.reference_speed_blend > 0.0 and np.isfinite(reference_speed):
                reference_acceleration = (reference_speed - self.state.velocity) / max(
                    self.dt, 1e-6
                )
                acceleration = (
                    (1.0 - self.reference_speed_blend) * acceleration
                    + self.reference_speed_blend * reference_acceleration
                )
                acceleration = float(np.clip(acceleration, -5.0, 5.0))
            safe_v = max(0.0, self.state.velocity + acceleration * self.dt)
            if self.max_speed > 0.0:
                safe_v = min(safe_v, self.max_speed)
            if self.follow_reference_direction:
                direction = np.asarray(reference_xyh[:2], dtype=float) - info["position"]
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1e-6:
                    step_distance = 0.5 * (self.state.velocity + safe_v) * self.dt
                    next_pos = info["position"] + direction / direction_norm * step_distance
            elif next_v > 0.1:
                next_pos = info["position"] + (next_pos - info["position"]) * (
                    safe_v / next_v
                )
            next_v = safe_v

        self.state.velocity = float(next_v)
        self.state.acceleration = float(acceleration)
        self.state.cached_acceleration = float(acceleration)
        self.state.cached_yaw_rate = float(yaw_rate)
        self.state.has_cached_control = True
        self.state.action_step += 1
        next_pos = np.nan_to_num(next_pos, nan=0.0, posinf=0.0, neginf=0.0)
        next_heading = float(np.nan_to_num(next_heading, nan=info["heading"]))
        self.last_command = {
            "policy": self.policy_name,
            "inference_interval_steps": self.inference_interval_steps,
            "used_cached_control": False,
            "use_risk_idm": bool(self.use_risk_idm),
            "heading_control": "asaprl_motion_skill",
            "reference_heading_blend": self.reference_heading_blend,
            "reference_speed_blend": self.reference_speed_blend,
            "max_yaw_rate": self.max_yaw_rate,
            "max_speed": self.max_speed,
            "follow_reference_direction": self.follow_reference_direction,
            "raw_latent_lat": float(raw_latent[0]),
            "raw_latent_yaw": float(raw_latent[1]),
            "raw_latent_v": float(raw_latent[2]),
            "action_lat_scale": self.action_lat_scale,
            "action_yaw_scale": self.action_yaw_scale,
            "action_speed_scale": self.action_speed_scale,
            "latent_lat": lat1,
            "latent_yaw": yaw1,
            "latent_v": v1,
            "next_yaw_deg": float(next_yaw_deg),
            "raw_yaw_rate": float(raw_yaw_rate),
            "reference_yaw_rate": float(reference_yaw_rate),
            "reference_speed": float(reference_speed),
            "yaw_rate": float(yaw_rate),
            "acceleration": float(acceleration),
            "velocity": float(next_v),
        }
        return PolicyAction(
            xyh=np.array([next_pos[0], next_pos[1], next_heading]),
            command=self.last_command.copy(),
        )

    def _repeat_cached_control(self, info) -> PolicyAction:
        heading = float(info["heading"])
        velocity = float(np.nan_to_num(self.state.velocity, nan=0.0))
        acceleration = float(self.state.cached_acceleration)
        yaw_rate = float(self.state.cached_yaw_rate)
        next_v = max(0.0, velocity + acceleration * self.dt)
        next_heading = heading + yaw_rate * self.dt
        avg_v = 0.5 * (velocity + next_v)
        mid_heading = heading + 0.5 * yaw_rate * self.dt
        next_pos = info["position"] + avg_v * self.dt * np.array(
            [np.cos(mid_heading), np.sin(mid_heading)]
        )

        self.state.velocity = float(next_v)
        self.state.acceleration = float(acceleration)
        self.state.action_step += 1
        next_pos = np.nan_to_num(next_pos, nan=0.0, posinf=0.0, neginf=0.0)
        next_heading = float(np.nan_to_num(next_heading, nan=info["heading"]))
        self.last_command = {
            "policy": self.policy_name,
            "inference_interval_steps": self.inference_interval_steps,
            "used_cached_control": True,
            "use_risk_idm": bool(self.use_risk_idm),
            "heading_control": "cached_yaw_rate",
            "acceleration": float(acceleration),
            "yaw_rate": float(yaw_rate),
            "velocity": float(next_v),
        }
        return PolicyAction(
            xyh=np.array([next_pos[0], next_pos[1], next_heading]),
            command=self.last_command.copy(),
        )

    def _build_observation_image(self, obs, ego_idx: int) -> np.ndarray:
        image = np.zeros((5, 200, 200), dtype=np.float32)
        image[0] = self._build_map_channel(obs, ego_idx)
        image[1] = self._build_ego_history_channel(obs, ego_idx)
        image[2] = self._build_neighbor_channel(obs, ego_idx, offset=0)
        image[3] = self._build_neighbor_channel(obs, ego_idx, offset=1)
        image[4] = self._build_neighbor_channel(obs, ego_idx, offset=2)
        return image

    def _build_map_channel(self, obs, ego_idx: int) -> np.ndarray:
        if getattr(obs, "maps", None) is None or len(obs.maps) <= ego_idx:
            return np.zeros((200, 200), dtype=np.float32)
        map_np = obs.maps[ego_idx]
        if hasattr(map_np, "cpu"):
            map_np = map_np.cpu()
        if hasattr(map_np, "numpy"):
            map_np = map_np.numpy()
        map_np = np.asarray(map_np, dtype=np.float32)
        if map_np.ndim == 3:
            if map_np.shape[0] >= 2:
                map_np = np.max(map_np[:2], axis=0)
            else:
                map_np = map_np[0]
        map_np = np.clip(map_np, 0.0, 1.0) * 255.0
        rotated = np.rot90(map_np, k=-1)
        scale_factor = max(1, int(round(12.0 / max(self.observation_px_per_m, 1e-6))))
        downsampled_size = max(1, rotated.shape[0] // scale_factor)
        downsampled = cv2.resize(
            rotated,
            (downsampled_size, downsampled_size),
            interpolation=cv2.INTER_AREA,
        )
        result = np.zeros((200, 200), dtype=np.float32)
        copy_h = min(200, downsampled.shape[0])
        copy_w = min(200, downsampled.shape[1])
        top = (200 - copy_h) // 2
        left = (200 - copy_w) // 2
        src_top = max(0, (downsampled.shape[0] - 200) // 2)
        src_left = max(0, (downsampled.shape[1] - 200) // 2)
        result[top : top + copy_h, left : left + copy_w] = downsampled[
            src_top : src_top + copy_h,
            src_left : src_left + copy_w,
        ]
        return result

    def _build_ego_history_channel(self, obs, ego_idx: int) -> np.ndarray:
        canvas = np.zeros((200, 200), dtype=np.float32)
        if getattr(obs, "agent_hist", None) is None:
            return canvas
        hist_len = int(obs.agent_hist_len[ego_idx].item())
        if hist_len <= 0:
            return canvas
        hist = obs.agent_hist[ego_idx, -hist_len:]
        points = []
        for idx in range(max(0, hist_len - 5), hist_len):
            state = hist[idx]
            points.append(self._state_to_image_xy(state))
        self._draw_polyline(canvas, points, value=255.0)
        for pt in points:
            self._draw_disk(canvas, pt, radius=2, value=255.0)
        return canvas

    def _build_neighbor_channel(self, obs, ego_idx: int, offset: int) -> np.ndarray:
        canvas = np.zeros((200, 200), dtype=np.float32)
        if getattr(obs, "neigh_hist", None) is None:
            return canvas
        neigh_hist = obs.neigh_hist[ego_idx]
        neigh_len = obs.neigh_hist_len[ego_idx]
        neigh_ext = obs.neigh_hist_extents[ego_idx]
        for n in range(neigh_hist.shape[0]):
            hist_len = int(neigh_len[n].item())
            if hist_len <= offset:
                continue
            step_idx = -1 - offset
            state = neigh_hist[n, step_idx]
            extent = neigh_ext[n, step_idx]
            self._draw_vehicle(canvas, state, extent, value=255.0)
        return canvas

    def _state_to_image_xy(self, state) -> np.ndarray:
        pos = to_numpy(state.position) if hasattr(state, "position") else np.asarray(state[:2], dtype=float)
        pos = np.asarray(pos, dtype=float).reshape(-1)[:2]
        scale = self.observation_px_per_m
        x = 100.0 + pos[1] * scale
        y = 100.0 - pos[0] * scale
        return np.array([x, y], dtype=float)

    def _draw_polyline(self, canvas: np.ndarray, points: list[np.ndarray], value: float) -> None:
        if len(points) < 2:
            if len(points) == 1:
                self._draw_disk(canvas, points[0], radius=1, value=value)
            return
        pts = np.asarray(points, dtype=np.float32)
        for idx in range(len(pts) - 1):
            p0 = tuple(np.round(pts[idx]).astype(int))
            p1 = tuple(np.round(pts[idx + 1]).astype(int))
            cv2.line(canvas, p0, p1, color=float(value), thickness=1, lineType=cv2.LINE_AA)

    def _draw_disk(self, canvas: np.ndarray, point: np.ndarray, radius: int, value: float) -> None:
        center = tuple(np.round(point).astype(int))
        cv2.circle(canvas, center, radius, color=float(value), thickness=-1, lineType=cv2.LINE_AA)

    def _draw_vehicle(self, canvas: np.ndarray, state, extent, value: float) -> None:
        pos = to_numpy(state.position) if hasattr(state, "position") else np.asarray(state[:2], dtype=float)
        heading = 0.0
        if hasattr(state, "heading"):
            heading_arr = np.asarray(to_numpy(state.heading), dtype=float).reshape(-1)
            if heading_arr.size:
                heading = float(heading_arr[0])
        pos = np.asarray(pos, dtype=float).reshape(-1)[:2]
        extent_np = np.asarray(to_numpy(extent), dtype=float).reshape(-1)
        length = float(extent_np[0]) if extent_np.size > 0 else 4.2
        width = float(extent_np[1]) if extent_np.size > 1 else 1.8
        half_l = 0.5 * length
        half_w = 0.5 * width
        corners = np.array(
            [
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w],
                [-half_l, -half_w],
            ],
            dtype=np.float32,
        )
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        rotated = corners @ rot.T
        rotated_px = np.stack(
            [
                100.0 + (pos[1] + rotated[:, 1]) * self.observation_px_per_m,
                100.0 - (pos[0] + rotated[:, 0]) * self.observation_px_per_m,
            ],
            axis=1,
        )
        pts = np.round(rotated_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], color=float(value), lineType=cv2.LINE_AA)


def _angle_diff(target: float, source: float) -> float:
    return float((target - source + np.pi) % (2.0 * np.pi) - np.pi)


def _blend_and_clip_yaw_rate(
    raw_yaw_rate: float,
    reference_yaw_rate: float,
    blend: float,
    limit: float,
) -> float:
    yaw_rate = (1.0 - blend) * raw_yaw_rate + blend * reference_yaw_rate
    if limit > 0.0:
        yaw_rate = float(np.clip(yaw_rate, -limit, limit))
    return float(yaw_rate)
