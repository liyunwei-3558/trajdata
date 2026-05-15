from __future__ import annotations

import math

import numpy as np


class PathParam:
    def __init__(self, lat0: float, yaw0: float, lon1: float, lat1: float, yaw1: float, lon_final: float = 80):
        yaw0 = math.tan(yaw0 / 180.0 * math.pi)
        yaw1 = math.tan(yaw1 / 180.0 * math.pi)
        self.horizon = lon1
        self.lon_final = lon_final
        self.a0 = lat0
        self.a1 = yaw0
        self.a3 = (2 * yaw0 + self.a1 * self.horizon + yaw1 * self.horizon - 2 * lat1) / (
            self.horizon**3
        )
        self.a2 = (yaw1 - self.a1 - 3 * self.a3 * (self.horizon**2)) / (2 * self.horizon)
        self._build_path_profile()

    def _build_path_profile(self) -> None:
        self.lon = np.arange(self.horizon * 10) / 10.0
        self.lat = self.a0 + self.a1 * self.lon + self.a2 * (self.lon**2) + self.a3 * (self.lon**3)
        self.lon = np.expand_dims(self.lon, 1)
        self.lat = np.expand_dims(self.lat, 1)
        self.yaw = np.arctan((self.lat[1:] - self.lat[:-1]) / (self.lon[1:] - self.lon[:-1])) / math.pi * 180
        self.yaw = np.vstack((self.yaw, self.yaw[-1]))
        self.path = np.hstack((self.lon, self.lat, self.yaw))
        self.path_length = [_path_length(self.path[: i + 1, :2]) for i in range(len(self.path - 1))]

    def get_pos_from_length(self, s: float) -> np.ndarray | None:
        if self.path_length[-1] < s:
            return None
        matched_ind = min(np.where(np.array(self.path_length) - s >= 0)[0])
        if matched_ind == 0:
            return self.path[:1, :]
        pos = np.zeros((1, 3))
        denom = _path_length(self.path[matched_ind - 1 : matched_ind + 1, :])
        percent = 0.0 if denom <= 1e-6 else (s - _path_length(self.path[:matched_ind, :])) / denom
        pos[0, 0] = self.path[matched_ind - 1, 0] + percent * (
            self.path[matched_ind, 0] - self.path[matched_ind - 1, 0]
        )
        pos[0, 1] = self.path[matched_ind - 1, 1] + percent * (
            self.path[matched_ind, 1] - self.path[matched_ind - 1, 1]
        )
        pos[0, 2] = self.path[matched_ind - 1, 2] + percent * (
            self.path[matched_ind, 2] - self.path[matched_ind - 1, 2]
        )
        return pos


class SpeedParam:
    def __init__(
        self,
        v0: float = 0.0,
        acc0: float = 0.0,
        v1: float = 0.0,
        acc1: float = 0.0,
        stop_time: float = 1.0,
        speed_pattern: str = "forward2",
        horizon: float = 3.0,
    ):
        self.horizon = horizon
        self.stop_time = stop_time
        self.speed_pattern = speed_pattern

        if self.speed_pattern == "forward1":
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (self.a1 * horizon + 2 * self.a0 - 2 * v1) / (horizon**3)
            self.a2 = (-self.a1 - 3 * self.a3 * (horizon**2)) / (2 * horizon)
        elif self.speed_pattern == "forward2":
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (self.a1 * horizon + 2 * self.a0 - 2 * v1) / (horizon**3)
            self.a2 = (acc1 - self.a1 - 3 * self.a3 * (horizon**2)) / (2 * horizon)
        else:
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (2 * self.a0 + self.a1 * stop_time) / (stop_time**3)
            self.a2 = (-self.a1 - 3 * self.a3 * stop_time**2) / (2 * stop_time)

    def get_distance(self, t: float) -> float:
        if t > self.horizon + 0.01:
            raise ValueError("The specified time exceeds the speed profile horizon.")
        if "forward" in self.speed_pattern:
            return 0.25 * self.a3 * t**4 + (1.0 / 3.0) * self.a2 * t**3 + 0.5 * self.a1 * t**2 + self.a0 * t
        if t <= self.stop_time:
            return 0.25 * self.a3 * t**4 + (1.0 / 3.0) * self.a2 * t**3 + 0.5 * self.a1 * t**2 + self.a0 * t
        return (
            0.25 * self.a3 * self.stop_time**4
            + (1.0 / 3.0) * self.a2 * self.stop_time**3
            + 0.5 * self.a1 * self.stop_time**2
            + self.a0 * self.stop_time
        )


def motion_skill_model(lat1: float, yaw1: float, current_v: float, current_a: float, v1: float, horizon: float = 3):
    """Original ASAPRL parameterized motion skill.

    The actor output is already a tanh-bounded latent action. Do not apply an
    additional scale before calling this function.
    """

    lat1, yaw1, current_v, current_a, v1 = _dynamic_constraint(lat1, yaw1, current_v, current_a, v1, horizon)
    path = PathParam(lat0=0, yaw0=0, lon1=30, lat1=lat1, yaw1=yaw1)
    speed_profile = SpeedParam(v0=current_v, acc0=current_a, v1=v1, horizon=horizon)

    steps = int(round(horizon)) + 1
    dist_lst = [speed_profile.get_distance(0.1 * step) for step in range(steps)]
    dist_lst = _dist_constraint(dist_lst)

    traj = np.zeros((steps, 4), dtype=float)
    for dist_num, s in enumerate(dist_lst):
        pos = path.get_pos_from_length(s)
        if pos is None:
            pos = path.path[-1:, :]
        traj[dist_num, :3] = pos

    traj[:-1, 3] = np.sqrt(np.sum(np.square(traj[1:, :2] - traj[:-1, :2]), axis=1)) * 10
    traj[-1, 3] = traj[-2, 3]
    traj[:, [2, 3]] = traj[:, [3, 2]]
    return traj, lat1, yaw1, v1


def _dynamic_constraint(lat1: float, yaw1: float, current_v: float, current_a: float, v1: float, horizon: float, lon1: float = 30):
    min_turning_radius = 8
    if lon1 <= min_turning_radius:
        lat1 = np.clip(lat1, -lon1, lon1)
    max_acc = 2
    v1 = np.clip(v1, current_v - horizon * max_acc, current_v + horizon * max_acc)
    return float(lat1), float(yaw1), float(current_v), float(current_a), float(v1)


def _dist_constraint(dist_lst):
    max_diff = 1
    negative_value = 0
    negative_info = {}
    exceed_info = {}
    dist_array = np.array(dist_lst, dtype=float)
    dist_diff = dist_array[1:] - dist_array[:-1]

    for i in np.where(dist_diff < negative_value)[0]:
        negative_info[i + 1] = dist_diff[i]
    for i in np.where(dist_diff > max_diff)[0]:
        exceed_info[i + 1] = dist_diff[i] - max_diff

    for step, value in negative_info.items():
        dist_array[step:] -= value
    for step, value in exceed_info.items():
        dist_array[step:] -= value
    return dist_array.tolist()


def _path_length(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum(np.square(pos[1:, :2] - pos[:-1, :2]), axis=1)), axis=0))
