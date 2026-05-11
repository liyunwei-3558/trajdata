from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose
from .base import BasePolicy, PolicyAction, PolicyState
from .ttc_utils import compute_min_ttc


@dataclass
class RiskIDMState(PolicyState):
    velocity: float = 0.0
    acceleration: float = 0.0


def risk_idm_formula(
    ve: float,
    se: float,
    r: float,
    v_0: float = 15.0,
    a_max: float = 5.0,
    delta: float = 4.0,
    delta_t0: float = 2.0,
    t_headway: float = 0.5,
    a_comf: float = 2.0,
    a_min: float = -5.0,
) -> float:
    if not np.isfinite(ve):
        ve = 0.0
    if not np.isfinite(r):
        r = 10.0
    v_0 = max(v_0, 1e-6)
    r = max(r, 1e-6)
    r_star = delta_t0 + t_headway
    accel = a_max * (1.0 - np.power(ve / v_0, delta) - np.power(r_star / r, 2))
    if not np.isfinite(se):
        accel = a_max * (1.0 - np.power(ve / v_0, delta))
    return float(np.clip(accel, a_min, a_max))


class RiskIDMPolicy(BasePolicy):
    policy_name = "risk_idm"

    def __init__(
        self,
        dt: float = 0.1,
        desired_velocity: float = 15.0,
        max_acceleration: float = 5.0,
        min_acceleration: float = -5.0,
        neighbor_radius: float = 50.0,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        self.desired_velocity = desired_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.neighbor_radius = neighbor_radius

    def reset(self, obs, ego_idx: int = 0) -> RiskIDMState:
        info = get_agent_world_pose(obs, ego_idx)
        self.state = RiskIDMState(
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
        ttc, collision_point, target_idx = compute_min_ttc(
            obs, ego_idx, neighbor_radius=self.neighbor_radius
        )
        se = (
            float(np.linalg.norm(collision_point - info["position"]))
            if collision_point is not None
            else float("inf")
        )
        acceleration = risk_idm_formula(
            ve=self.state.velocity,
            se=se,
            r=ttc,
            v_0=self.desired_velocity,
            a_max=self.max_acceleration,
            a_min=self.min_acceleration,
        )
        new_velocity = max(0.0, self.state.velocity + acceleration * self.dt)
        step_dist = new_velocity * self.dt
        heading = float(info["heading"])
        next_pos = info["position"] + step_dist * np.array(
            [np.cos(heading), np.sin(heading)]
        )

        self.state.velocity = new_velocity
        self.state.acceleration = acceleration
        self.last_command = {
            "policy": self.policy_name,
            "acceleration": acceleration,
            "velocity": new_velocity,
            "min_ttc": ttc,
            "target_agent": obs.agent_name[target_idx] if target_idx is not None else None,
        }
        return PolicyAction(
            xyh=np.array([next_pos[0], next_pos[1], heading]),
            command=self.last_command.copy(),
        )
