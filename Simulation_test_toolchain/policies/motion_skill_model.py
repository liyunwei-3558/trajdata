from __future__ import annotations

import math

import numpy as np


def _clip_motion(lat1, yaw1, current_v, current_a, v1, horizon):
    lat1 = float(np.clip(lat1, -5.0, 5.0))
    yaw1 = float(np.clip(yaw1, -25.0, 25.0))
    v1 = float(np.clip(v1, max(0.0, current_v - 2.0 * horizon), current_v + 2.0 * horizon))
    return lat1, yaw1, current_v, current_a, v1


def motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon=3):
    lat1, yaw1, current_v, current_a, v1 = _clip_motion(
        lat1, yaw1, current_v, current_a, v1, horizon
    )
    steps = max(2, int(round(horizon / 0.1)) + 1)
    ts = np.linspace(0.0, horizon, steps)
    speeds = np.linspace(current_v, v1, steps)
    lon = np.cumsum(speeds) * 0.1
    lon -= lon[0]
    progress = ts / max(horizon, 1e-6)
    lat = lat1 * (3 * progress**2 - 2 * progress**3)
    yaw = yaw1 * progress
    traj = np.stack([lon, lat, speeds, yaw], axis=1)
    return traj, lat1, yaw1, v1

