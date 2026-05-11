from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose
from .base import BasePolicy, PolicyAction, PolicyState
from .motion_skill_model import motion_skill_model
from .risk_idm import risk_idm_formula
from .ttc_utils import compute_min_ttc


@dataclass
class ASAPRLState(PolicyState):
    velocity: float = 0.0
    acceleration: float = 0.0


class ASAPRLPolicy(BasePolicy):
    policy_name = "asaprl"

    def __init__(
        self,
        dt: float = 0.1,
        target_speed: float = 5.0,
        horizon: float = 3.0,
        ckpt_path: Optional[str] = None,
        use_risk_idm: bool = True,
        neighbor_radius: float = 50.0,
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
                x = self.main(x)
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
        image = self._build_observation_image(obs, ego_idx)
        with self.torch.no_grad():
            tensor = self.torch.tensor(image[None], dtype=self.torch.float32)
            latent = self.model(tensor).cpu().numpy()[0]
        latent = np.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=-1.0)

        lat1 = float(latent[0] * 5.0)
        yaw1 = float(latent[1] * 15.0)
        v1 = float((latent[2] + 1.0) * 0.5 * self.target_speed)
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
            safe_v = max(0.0, self.state.velocity + acceleration * self.dt)
            if next_v > 0.1:
                next_pos = info["position"] + (next_pos - info["position"]) * (
                    safe_v / next_v
                )
            next_v = safe_v

        self.state.velocity = float(next_v)
        self.state.acceleration = float(acceleration)
        next_pos = np.nan_to_num(next_pos, nan=0.0, posinf=0.0, neginf=0.0)
        next_heading = float(np.nan_to_num(next_heading, nan=info["heading"]))
        self.last_command = {
            "policy": self.policy_name,
            "latent_lat": lat1,
            "latent_yaw_deg": yaw1,
            "latent_target_speed": v1,
            "acceleration": float(acceleration),
            "velocity": float(next_v),
        }
        return PolicyAction(
            xyh=np.array([next_pos[0], next_pos[1], next_heading]),
            command=self.last_command.copy(),
        )

    def _build_observation_image(self, obs, ego_idx: int) -> np.ndarray:
        # Minimal 5-channel observation: map channels when available plus agents.
        image = np.zeros((5, 200, 200), dtype=np.float32)
        if getattr(obs, "maps", None) is not None and len(obs.maps) > ego_idx:
            map_np = obs.maps[ego_idx]
            if hasattr(map_np, "cpu"):
                map_np = map_np.cpu()
            if hasattr(map_np, "numpy"):
                map_np = map_np.numpy()
            for channel in range(min(2, map_np.shape[0])):
                image[channel, : min(200, map_np.shape[1]), : min(200, map_np.shape[2])] = (
                    map_np[channel, :200, :200]
                )
        return image
