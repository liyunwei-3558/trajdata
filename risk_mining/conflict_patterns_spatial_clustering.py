#!/usr/bin/env python
"""Compute SinD conflict patterns and spatial clustering metrics.

Metric 3 measures which maneuver pairs create intersection conflicts. Metric 4
maps strong deceleration-yield points inside the intersection core. The script
uses loose pair conflicts (distance + short-horizon path overlap) and does not
estimate collision risk.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, PathPatch, Wedge
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, mapping

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinematic_envelopes import (  # noqa: E402
    LOCATION_DISPLAY,
    TrackRecord,
    filter_tracks,
    load_sind_tracks,
    normalize_location,
)
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS, LaneReference, load_lane_references  # noqa: E402
from intersection_spatiotemporal_density import filter_static_tracks, infer_intersection_core_roi  # noqa: E402

VEHICLE_PAIR_CLASSES = {"car", "truck", "bus", "tricycle", "motorcycle", "bicycle"}
ALL_DYNAMIC_CLASSES = VEHICLE_PAIR_CLASSES | {"pedestrian"}
CITY_GROUPS = {
    "cc": "ChangChun",
    "tj": "TianJin",
    "cqIR": "ChongQing",
    "cqNR": "ChongQing",
    "cqR": "ChongQing",
    "xasl": "XiAn",
}
MANEUVER_NODES = ["straight", "left-turn", "right-turn", "unknown"]
CONFLICT_TYPES = ["Crossing", "Merging", "Weaving/Parallel Competition", "Turning Interaction", "Mixed/Other"]


@dataclass
class TrackCache:
    idx: int
    track: TrackRecord
    frames: np.ndarray
    xy: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    a_lon: np.ndarray
    inside_roi_buffer: np.ndarray


def _city_group(location: str) -> str:
    return CITY_GROUPS.get(location, LOCATION_DISPLAY.get(location, location))


def _polygon_to_path(polygon: Polygon) -> MplPath:
    return MplPath(np.asarray(polygon.exterior.coords, dtype=float))


def _track_arrays(track: TrackRecord) -> Tuple[np.ndarray, ...]:
    state = track.state.sort_values("frame_id")
    if not {"frame_id", "x", "y"}.issubset(state.columns):
        empty_xy = np.empty((0, 2), dtype=float)
        empty = np.empty((0,), dtype=float)
        return np.empty((0,), dtype=int), empty_xy, empty, empty, empty, empty, empty
    xy = state[["x", "y"]].to_numpy(dtype=float)
    frames = state["frame_id"].to_numpy(dtype=int)
    mask = np.isfinite(xy).all(axis=1) & np.isfinite(frames)
    xy = xy[mask]
    frames = frames[mask]

    def col(name: str) -> np.ndarray:
        if name in state.columns:
            values = state[name].to_numpy(dtype=float)[mask]
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(len(frames), dtype=float)

    vx = col("vx")
    vy = col("vy")
    if not np.any(vx) and len(xy) >= 2:
        vx = np.gradient(xy[:, 0], 0.1)
        vy = np.gradient(xy[:, 1], 0.1)
    ax = col("ax")
    ay = col("ay")
    if not np.any(ax) and len(xy) >= 3:
        ax = np.gradient(vx, 0.1)
        ay = np.gradient(vy, 0.1)
    if "a_lon" in state.columns:
        a_lon = np.nan_to_num(state["a_lon"].to_numpy(dtype=float)[mask], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        speed = np.hypot(vx, vy)
        a_lon = np.divide(vx * ax + vy * ay, np.maximum(speed, 1e-3), out=np.zeros_like(speed), where=speed > 1e-3)
    return frames, xy, vx, vy, ax, ay, a_lon


def make_scene_caches(tracks: Sequence[TrackRecord], roi_polygon: Polygon) -> List[TrackCache]:
    roi_path = _polygon_to_path(roi_polygon)
    caches: List[TrackCache] = []
    for track in tracks:
        frames, xy, vx, vy, ax, ay, a_lon = _track_arrays(track)
        if len(xy) < 3 or len(xy) != len(frames):
            continue
        inside = roi_path.contains_points(xy)
        if not inside.any():
            continue
        caches.append(
            TrackCache(
                idx=len(caches),
                track=track,
                frames=frames,
                xy=xy,
                vx=vx,
                vy=vy,
                ax=ax,
                ay=ay,
                a_lon=a_lon,
                inside_roi_buffer=inside,
            )
        )
    return caches


def _bbox_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a_min, a_max = a.min(axis=0), a.max(axis=0)
    b_min, b_max = b.min(axis=0), b.max(axis=0)
    gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gap))


def _future_window(cache: TrackCache, idx: int, horizon_steps: int, stride: int) -> np.ndarray:
    end = min(len(cache.xy), idx + horizon_steps + 1)
    return cache.xy[idx:end:max(1, stride)]


def _future_conflict(
    a: TrackCache,
    a_idx: int,
    b: TrackCache,
    b_idx: int,
    horizon_steps: int,
    stride: int,
    path_conflict_distance_m: float,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    a_future = _future_window(a, a_idx, horizon_steps, stride)
    b_future = _future_window(b, b_idx, horizon_steps, stride)
    if len(a_future) < 2 or len(b_future) < 2:
        return None
    if _bbox_distance(a_future, b_future) > path_conflict_distance_m:
        return None
    diff = a_future[:, None, :] - b_future[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    flat_idx = int(np.argmin(d2))
    ia, ib = np.unravel_index(flat_idx, d2.shape)
    min_dist = float(math.sqrt(float(d2[ia, ib])))
    if min_dist > path_conflict_distance_m:
        return None
    point = (a_future[ia] + b_future[ib]) / 2.0
    return min_dist, (float(point[0]), float(point[1]))


def _canonical_maneuver(maneuver: str) -> str:
    return maneuver if maneuver in MANEUVER_NODES else "unknown"


def _maneuver_pair(a: str, b: str) -> str:
    ma, mb = sorted([_canonical_maneuver(a), _canonical_maneuver(b)])
    return f"{ma}__{mb}"


def _direction_heading(direction: str) -> Optional[float]:
    text = str(direction).strip().lower()
    if "_" not in text:
        return None
    src, dst = text.split("_", 1)
    src = src[:1]
    dst = dst[:1]
    vectors = {
        "e": np.array([1.0, 0.0]),
        "w": np.array([-1.0, 0.0]),
        "n": np.array([0.0, 1.0]),
        "s": np.array([0.0, -1.0]),
    }
    if src not in vectors or dst not in vectors:
        return None
    initial_vec = -vectors[src]
    if np.linalg.norm(initial_vec) < 1e-6:
        initial_vec = vectors[dst]
    return float(math.atan2(initial_vec[1], initial_vec[0]))


def _heading_diff_deg(a: float, b: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def _conflict_type(a_maneuver: str, b_maneuver: str, a_direction: str = "", b_direction: str = "") -> str:
    pair = {_canonical_maneuver(a_maneuver), _canonical_maneuver(b_maneuver)}
    if pair == {"left-turn", "straight"}:
        return "Crossing"
    if pair == {"right-turn", "straight"}:
        return "Merging"
    if pair == {"straight"}:
        a_heading = _direction_heading(a_direction)
        b_heading = _direction_heading(b_direction)
        if a_heading is None or b_heading is None:
            return "Weaving/Parallel Competition"
        diff = _heading_diff_deg(a_heading, b_heading)
        if diff < 35.0:
            return "Weaving/Parallel Competition"
        if diff < 145.0:
            return "Crossing"
        return "Mixed/Other"
    if pair <= {"left-turn", "right-turn"}:
        return "Turning Interaction"
    return "Mixed/Other"


def _is_vehicle_pair(a: TrackCache, b: TrackCache) -> bool:
    return a.track.class_name in VEHICLE_PAIR_CLASSES and b.track.class_name in VEHICLE_PAIR_CLASSES


def _agent_pair_id(a: TrackCache, b: TrackCache) -> Tuple[str, str]:
    aid = str(a.track.agent_id)
    bid = str(b.track.agent_id)
    return (aid, bid) if aid <= bid else (bid, aid)


def _velocity_direction(cache: TrackCache, idx: int) -> np.ndarray:
    vel = np.array([cache.vx[idx], cache.vy[idx]], dtype=float)
    if np.linalg.norm(vel) >= 0.3:
        return vel
    end = min(len(cache.xy) - 1, idx + 5)
    if end > idx:
        vel = cache.xy[end] - cache.xy[idx]
    return vel


def _is_ahead(cache: TrackCache, idx: int, other_xy: np.ndarray, front_cone_deg: float) -> Tuple[bool, float]:
    direction = _velocity_direction(cache, idx)
    rel = other_xy - cache.xy[idx]
    dnorm = float(np.linalg.norm(direction))
    rnorm = float(np.linalg.norm(rel))
    if dnorm < 1e-6 or rnorm < 1e-6:
        return False, 180.0
    cos_value = float(np.clip(np.dot(direction, rel) / (dnorm * rnorm), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_value))
    return bool(angle <= front_cone_deg / 2.0), float(angle)


def _limit_scenes_per_location(tracks: Sequence[TrackRecord], max_scenes_per_location: Optional[int]) -> List[TrackRecord]:
    if max_scenes_per_location is None:
        return list(tracks)
    scenes_by_location: Dict[str, List[str]] = {}
    for track in tracks:
        scenes = scenes_by_location.setdefault(track.location, [])
        if track.scene_id not in scenes and len(scenes) < max_scenes_per_location:
            scenes.append(track.scene_id)
    allowed = {(loc, scene) for loc, scenes in scenes_by_location.items() for scene in scenes}
    return [track for track in tracks if (track.location, track.scene_id) in allowed]


def build_rois(
    data_dir: Path,
    cities: Sequence[str],
    roi_buffer_m: float,
    core_buffer_m: float,
    max_core_radius_m: float,
    min_core_radius_m: float,
    turn_heading_threshold_deg: float,
    lane_trim_ratio: float,
    central_quantile: float,
) -> Tuple[Dict[str, Polygon], Dict[str, Polygon], Dict[str, List[LaneReference]], Dict[str, Dict[str, Any]]]:
    refs_by_city = load_lane_references(data_dir, cities)
    core_rois: Dict[str, Polygon] = {}
    buffered_rois: Dict[str, Polygon] = {}
    debug: Dict[str, Dict[str, Any]] = {}
    for city in cities:
        polygon, roi_debug = infer_intersection_core_roi(
            refs_by_city.get(city, []),
            core_buffer_m=core_buffer_m,
            max_core_radius_m=max_core_radius_m,
            min_core_radius_m=min_core_radius_m,
            turn_heading_threshold_deg=turn_heading_threshold_deg,
            lane_trim_ratio=lane_trim_ratio,
            central_quantile=central_quantile,
        )
        core_rois[city] = polygon
        buffered_rois[city] = polygon.buffer(roi_buffer_m)
        debug[city] = roi_debug
    return core_rois, buffered_rois, refs_by_city, debug


def compute_scene_conflicts(
    location: str,
    scene_id: str,
    tracks: Sequence[TrackRecord],
    roi_polygon: Polygon,
    dt: float,
    distance_threshold_m: float,
    path_conflict_distance_m: float,
    future_horizon_s: float,
    future_stride: int,
    decel_threshold_mps2: float,
    front_cone_deg: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    caches = make_scene_caches(tracks, roi_polygon)
    if len(caches) < 2:
        return [], []

    frame_entries: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
    for cache in caches:
        for pos_idx, frame in enumerate(cache.frames):
            if cache.inside_roi_buffer[pos_idx]:
                frame_entries.setdefault(int(frame), []).append((cache.idx, pos_idx, cache.xy[pos_idx]))

    horizon_steps = max(1, int(round(future_horizon_s / dt)))
    pair_rows: List[Dict[str, Any]] = []
    hotspot_rows: List[Dict[str, Any]] = []

    for frame in sorted(frame_entries):
        entries = frame_entries[frame]
        if len(entries) < 2:
            continue
        points = np.asarray([entry[2] for entry in entries], dtype=float)
        pairs = cKDTree(points).query_pairs(distance_threshold_m)
        best_decel: Dict[int, Dict[str, Any]] = {}
        for local_a, local_b in pairs:
            a_cache_idx, a_pos_idx, a_xy = entries[local_a]
            b_cache_idx, b_pos_idx, b_xy = entries[local_b]
            a_cache = caches[a_cache_idx]
            b_cache = caches[b_cache_idx]
            conflict = _future_conflict(
                a_cache,
                a_pos_idx,
                b_cache,
                b_pos_idx,
                horizon_steps=horizon_steps,
                stride=future_stride,
                path_conflict_distance_m=path_conflict_distance_m,
            )
            if conflict is None:
                continue
            min_future_distance, conflict_point = conflict
            current_distance = float(np.linalg.norm(a_xy - b_xy))

            if _is_vehicle_pair(a_cache, b_cache):
                a_id, b_id = _agent_pair_id(a_cache, b_cache)
                a_first = str(a_cache.track.agent_id) == a_id
                first = a_cache if a_first else b_cache
                second = b_cache if a_first else a_cache
                pair_rows.append(
                    {
                        "location": location,
                        "city": LOCATION_DISPLAY.get(location, location),
                        "city_group": _city_group(location),
                        "scene_id": scene_id,
                        "frame_id": int(frame),
                        "agent_a_id": a_id,
                        "agent_b_id": b_id,
                        "agent_a_class": first.track.class_name,
                        "agent_b_class": second.track.class_name,
                        "agent_a_maneuver": _canonical_maneuver(first.track.maneuver),
                        "agent_b_maneuver": _canonical_maneuver(second.track.maneuver),
                        "agent_a_direction": first.track.direction,
                        "agent_b_direction": second.track.direction,
                        "maneuver_pair": _maneuver_pair(first.track.maneuver, second.track.maneuver),
                        "conflict_type": _conflict_type(first.track.maneuver, second.track.maneuver, first.track.direction, second.track.direction),
                        "current_distance_m": current_distance,
                        "min_future_distance_m": min_future_distance,
                        "conflict_point_x": conflict_point[0],
                        "conflict_point_y": conflict_point[1],
                    }
                )

            for ego_cache, ego_pos_idx, other_cache, other_pos_idx, other_xy in (
                (a_cache, a_pos_idx, b_cache, b_pos_idx, b_xy),
                (b_cache, b_pos_idx, a_cache, a_pos_idx, a_xy),
            ):
                a_lon = float(ego_cache.a_lon[ego_pos_idx])
                if a_lon >= decel_threshold_mps2:
                    continue
                ahead, front_angle = _is_ahead(ego_cache, ego_pos_idx, other_xy, front_cone_deg)
                if not ahead:
                    continue
                existing = best_decel.get(ego_cache.idx)
                if existing is not None and current_distance >= existing["participant_distance_m"]:
                    continue
                best_decel[ego_cache.idx] = {
                    "location": location,
                    "city": LOCATION_DISPLAY.get(location, location),
                    "city_group": _city_group(location),
                    "scene_id": scene_id,
                    "frame_id": int(frame),
                    "agent_id": ego_cache.track.agent_id,
                    "class_name": ego_cache.track.class_name,
                    "maneuver": _canonical_maneuver(ego_cache.track.maneuver),
                    "x": float(ego_cache.xy[ego_pos_idx, 0]),
                    "y": float(ego_cache.xy[ego_pos_idx, 1]),
                    "vx": float(ego_cache.vx[ego_pos_idx]),
                    "vy": float(ego_cache.vy[ego_pos_idx]),
                    "longitudinal_accel_mps2": a_lon,
                    "participant_agent_id": other_cache.track.agent_id,
                    "participant_class": other_cache.track.class_name,
                    "participant_maneuver": _canonical_maneuver(other_cache.track.maneuver),
                    "participant_distance_m": current_distance,
                    "front_angle_deg": front_angle,
                    "min_future_distance_m": min_future_distance,
                    "conflict_point_x": conflict_point[0],
                    "conflict_point_y": conflict_point[1],
                }
        hotspot_rows.extend(best_decel.values())
    return pair_rows, hotspot_rows


def compute_conflicts(
    tracks: Sequence[TrackRecord],
    buffered_rois: Mapping[str, Polygon],
    dt: float,
    distance_threshold_m: float,
    path_conflict_distance_m: float,
    future_horizon_s: float,
    future_stride: int,
    decel_threshold_mps2: float,
    front_cone_deg: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped: Dict[Tuple[str, str], List[TrackRecord]] = {}
    for track in tracks:
        grouped.setdefault((track.location, track.scene_id), []).append(track)
    pair_rows: List[Dict[str, Any]] = []
    hotspot_rows: List[Dict[str, Any]] = []
    for idx, ((location, scene_id), scene_tracks) in enumerate(sorted(grouped.items()), start=1):
        roi = buffered_rois.get(location)
        if roi is None:
            continue
        scene_pairs, scene_hotspots = compute_scene_conflicts(
            location,
            scene_id,
            scene_tracks,
            roi,
            dt=dt,
            distance_threshold_m=distance_threshold_m,
            path_conflict_distance_m=path_conflict_distance_m,
            future_horizon_s=future_horizon_s,
            future_stride=future_stride,
            decel_threshold_mps2=decel_threshold_mps2,
            front_cone_deg=front_cone_deg,
        )
        pair_rows.extend(scene_pairs)
        hotspot_rows.extend(scene_hotspots)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(grouped)} scenes; pair events={len(pair_rows):,}; decel events={len(hotspot_rows):,}", flush=True)
    return pd.DataFrame(pair_rows), pd.DataFrame(hotspot_rows)


def merge_pair_episodes(pair_df: pd.DataFrame, dt: float, max_gap_s: float) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    max_gap_frames = max(0, int(round(max_gap_s / dt)))
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "location",
        "city",
        "city_group",
        "scene_id",
        "agent_a_id",
        "agent_b_id",
        "agent_a_class",
        "agent_b_class",
        "agent_a_maneuver",
        "agent_b_maneuver",
        "agent_a_direction",
        "agent_b_direction",
        "maneuver_pair",
        "conflict_type",
    ]
    episode_id = 1
    for keys, group in pair_df.sort_values("frame_id").groupby(group_cols, dropna=False):
        key_data = dict(zip(group_cols, keys))
        frames = group["frame_id"].to_numpy(dtype=int)
        start_idx = 0
        for idx in range(1, len(frames) + 1):
            split = idx == len(frames) or frames[idx] - frames[idx - 1] > max_gap_frames + 1
            if not split:
                continue
            seg = group.iloc[start_idx:idx]
            start_frame = int(seg["frame_id"].min())
            end_frame = int(seg["frame_id"].max())
            row = {
                **key_data,
                "pair_episode_id": episode_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_s": float((end_frame - start_frame) * dt + dt),
                "event_frames": int(len(seg)),
                "median_current_distance_m": float(seg["current_distance_m"].median()),
                "min_future_distance_m": float(seg["min_future_distance_m"].min()),
                "mean_conflict_point_x": float(seg["conflict_point_x"].mean()),
                "mean_conflict_point_y": float(seg["conflict_point_y"].mean()),
            }
            rows.append(row)
            episode_id += 1
            start_idx = idx
    return pd.DataFrame(rows)


def summarize_conflict_types(df: pd.DataFrame, group_col: str, groups: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        sub = df[df[group_col] == group] if not df.empty else pd.DataFrame()
        total = int(len(sub))
        row: Dict[str, Any] = {group_col: group, "total_pair_episodes": total}
        for ctype in CONFLICT_TYPES:
            count = int((sub["conflict_type"] == ctype).sum()) if total else 0
            key = ctype.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
            row[f"{key}_count"] = count
            row[f"{key}_ratio"] = float(count / total) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def extract_hotspot_cells(hotspot_df: pd.DataFrame, rois: Mapping[str, Polygon], grid_size_m: float, top_k: int) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows: List[Dict[str, Any]] = []
    grids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for location, roi in rois.items():
        minx, miny, maxx, maxy = roi.bounds
        x_edges = np.arange(math.floor(minx), math.ceil(maxx) + grid_size_m, grid_size_m, dtype=float)
        y_edges = np.arange(math.floor(miny), math.ceil(maxy) + grid_size_m, grid_size_m, dtype=float)
        grid = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
        sub = hotspot_df[hotspot_df["location"] == location] if not hotspot_df.empty else pd.DataFrame()
        if not sub.empty:
            xy = sub[["x", "y"]].to_numpy(dtype=float)
            ix = np.floor((xy[:, 0] - x_edges[0]) / grid_size_m).astype(int)
            iy = np.floor((xy[:, 1] - y_edges[0]) / grid_size_m).astype(int)
            valid = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
            np.add.at(grid, (iy[valid], ix[valid]), 1.0)
        grids[location] = (grid, x_edges, y_edges)
        positive = np.flatnonzero(grid.ravel() > 0)
        if len(positive) == 0:
            continue
        order = positive[np.argsort(grid.ravel()[positive])[::-1]][:top_k]
        for rank, flat_idx in enumerate(order, start=1):
            iy, ix = np.unravel_index(int(flat_idx), grid.shape)
            rows.append(
                {
                    "location": location,
                    "city": LOCATION_DISPLAY.get(location, location),
                    "city_group": _city_group(location),
                    "rank": rank,
                    "x_center_m": float((x_edges[ix] + x_edges[ix + 1]) / 2.0),
                    "y_center_m": float((y_edges[iy] + y_edges[iy + 1]) / 2.0),
                    "deceleration_events": int(grid[iy, ix]),
                    "cell_ix": int(ix),
                    "cell_iy": int(iy),
                }
            )
    return pd.DataFrame(rows), grids


def plot_hotspot_map(location: str, grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, roi: Polygon, core_roi: Polygon, lane_refs: Sequence[LaneReference], output_path: Path) -> None:
    masked = np.ma.masked_where(grid <= 0, grid)
    vmax = float(np.quantile(grid[grid > 0], 0.995)) if np.any(grid > 0) else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    im = ax.imshow(masked, extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), origin="lower", cmap="YlOrRd", interpolation="nearest", vmin=0.0, vmax=vmax, alpha=0.9)
    for lane in lane_refs:
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#263e4b", linewidth=0.55, alpha=0.36)
    roi_xy = np.asarray(roi.exterior.coords)
    core_xy = np.asarray(core_roi.exterior.coords)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#6b6b6b", linewidth=1.1, linestyle="--", label="ROI buffer")
    ax.plot(core_xy[:, 0], core_xy[:, 1], color="#111111", linewidth=1.9, label="Core ROI")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{LOCATION_DISPLAY.get(location, location)} Deceleration-Yield Conflict Hotspots")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Strong deceleration-yield events per 1m cell")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _chord_matrix(df: pd.DataFrame, group_col: str, group_value: str) -> np.ndarray:
    matrix = np.zeros((len(MANEUVER_NODES), len(MANEUVER_NODES)), dtype=float)
    sub = df[df[group_col] == group_value] if not df.empty else pd.DataFrame()
    node_idx = {name: idx for idx, name in enumerate(MANEUVER_NODES)}
    for row in sub.to_dict("records"):
        a = _canonical_maneuver(str(row.get("agent_a_maneuver", "unknown")))
        b = _canonical_maneuver(str(row.get("agent_b_maneuver", "unknown")))
        ia, ib = node_idx[a], node_idx[b]
        matrix[ia, ib] += 1
        if ia != ib:
            matrix[ib, ia] += 1
    return matrix


def plot_chord(df: pd.DataFrame, group_col: str, group_value: str, output_path: Path, title: str) -> None:
    matrix = _chord_matrix(df, group_col, group_value)
    totals = matrix.sum(axis=1)
    total = float(totals.sum())
    fig, ax = plt.subplots(figsize=(7.4, 7.4), dpi=170)
    ax.set_aspect("equal")
    ax.axis("off")
    colors = {"straight": "#375c6c", "left-turn": "#b64032", "right-turn": "#d28b3c", "unknown": "#7b7d7d"}
    if total <= 0:
        ax.text(0, 0, "No pair episodes", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    start = 90.0
    gap = 4.0
    arc_angles: Dict[str, Tuple[float, float, float]] = {}
    usable = 360.0 - gap * len(MANEUVER_NODES)
    for name, node_total in zip(MANEUVER_NODES, totals):
        span = max(8.0, usable * float(node_total) / total) if node_total > 0 else 8.0
        theta1, theta2 = start, start - span
        arc_angles[name] = (theta1, theta2, (theta1 + theta2) / 2.0)
        wedge = Wedge((0, 0), 1.0, theta2, theta1, width=0.13, facecolor=colors[name], alpha=0.92, edgecolor="white", linewidth=1.4)
        ax.add_patch(wedge)
        mid = math.radians((theta1 + theta2) / 2.0)
        ax.text(1.18 * math.cos(mid), 1.18 * math.sin(mid), f"{name}\n{int(node_total/2 if name != 'unknown' else node_total)}", ha="center", va="center", fontsize=10)
        start = theta2 - gap

    max_value = float(matrix.max()) if matrix.size else 1.0
    for ia, a in enumerate(MANEUVER_NODES):
        for ib, b in enumerate(MANEUVER_NODES):
            if ib < ia:
                continue
            value = matrix[ia, ib]
            if value <= 0:
                continue
            mid_a = math.radians(arc_angles[a][2])
            mid_b = math.radians(arc_angles[b][2])
            p0 = np.array([0.87 * math.cos(mid_a), 0.87 * math.sin(mid_a)])
            p1 = np.array([0.87 * math.cos(mid_b), 0.87 * math.sin(mid_b)])
            if ia == ib:
                center = p0 * 0.56
                circ = Circle(center, 0.16 + 0.08 * value / max_value, fill=False, edgecolor=colors[a], alpha=0.35, linewidth=1.0 + 5.0 * value / max_value)
                ax.add_patch(circ)
                continue
            verts = [tuple(p0), (0.0, 0.0), tuple(p1)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            path = MplPath(verts, codes)
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colors[a], alpha=0.22 + 0.45 * value / max_value, linewidth=0.7 + 7.0 * value / max_value))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(title, fontsize=14, pad=18)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_geojson(output_path: Path, core_rois: Mapping[str, Polygon], buffered_rois: Mapping[str, Polygon], roi_debug: Mapping[str, Mapping[str, Any]]) -> None:
    features = []
    for location, polygon in core_rois.items():
        features.append({"type": "Feature", "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "roi_type": "core", **dict(roi_debug.get(location, {}))}, "geometry": mapping(polygon)})
        features.append({"type": "Feature", "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "roi_type": "buffered"}, "geometry": mapping(buffered_rois[location])})
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    parts = ["<table><thead><tr>"]
    for _, label in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cell = f"{value:.3f}"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html_report(output_path: Path, loc_summary: pd.DataFrame, city_summary: pd.DataFrame, hotspot_cells: pd.DataFrame, cities: Sequence[str]) -> None:
    cards = []
    for city in cities:
        cards.append(
            f"""
            <article class='card'>
              <h3>{html.escape(LOCATION_DISPLAY.get(city, city))} <span>{html.escape(city)}</span></h3>
              <a href='conflict_chord_{html.escape(city)}.png'><img src='conflict_chord_{html.escape(city)}.png' alt='chord'></a>
              <a href='conflict_hotspot_{html.escape(city)}.png'><img src='conflict_hotspot_{html.escape(city)}.png' alt='hotspot'></a>
            </article>
            """
        )
    summary_table = _html_table(
        loc_summary.to_dict("records"),
        [
            ("location", "Location"),
            ("total_pair_episodes", "Pair episodes"),
            ("crossing_ratio", "Crossing"),
            ("merging_ratio", "Merging"),
            ("weaving_parallel_competition_ratio", "Weaving"),
            ("turning_interaction_ratio", "Turning"),
            ("mixed_other_ratio", "Other"),
        ],
    )
    city_table = _html_table(
        city_summary.to_dict("records"),
        [
            ("city_group", "City"),
            ("total_pair_episodes", "Pair episodes"),
            ("crossing_ratio", "Crossing"),
            ("merging_ratio", "Merging"),
            ("weaving_parallel_competition_ratio", "Weaving"),
            ("turning_interaction_ratio", "Turning"),
            ("mixed_other_ratio", "Other"),
        ],
    )
    hotspot_table = _html_table(
        hotspot_cells.head(50).to_dict("records"),
        [("city", "City"), ("rank", "Rank"), ("x_center_m", "x"), ("y_center_m", "y"), ("deceleration_events", "Events")],
    )
    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Conflict Patterns & Spatial Clustering</title>
  <style>
    :root {{ --ink:#1d2326; --paper:#f4eadb; --card:#fffaf0; --line:#d8c5a6; --muted:#657074; --red:#b64032; --blue:#375c6c; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f4eadb,#e8eeee); }}
    header {{ padding:44px 5vw 74px; color:#fff; background:radial-gradient(circle at 82% 12%,#d99743,transparent 25%), linear-gradient(135deg,#263f47,#6b3329); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); }} header p {{ max-width:1040px; line-height:1.65; color:#f7ead8; font-size:17px; }}
    main {{ padding:0 5vw 58px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:18px; margin-top:-42px; }}
    .card,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 30px rgba(29,35,38,.08); }}
    .card h3 {{ margin:0 0 10px; }} .card h3 span {{ color:var(--muted); font-size:13px; }} section {{ margin-top:34px; }} h2 {{ font-size:30px; }}
    img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,240,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #eadcc8; text-align:left; font-size:14px; }} th {{ background:#ead6bd; }} tr:hover td {{ background:#fff0d8; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fffaf0; border:1px solid var(--line); border-radius:999px; color:var(--blue); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>Conflict Patterns & Spatial Clustering</h1>
  <p>展示不同全局意图的车辆 pair 如何产生交集，以及强减速避让点在高精地图上的空间聚集。弦图基于 pair episode，热点图基于 a_lon &lt; -1.5m/s² 且前方存在潜在冲突参与者的事件。</p>
</header>
<main>
  <div class='grid'>{''.join(cards)}</div>
  <section><h2>City Group Chord</h2><div class='panel'><a href='conflict_chord_city_groups.png'><img src='conflict_chord_city_groups.png'></a></div></section>
  <section><h2>Conflict Type by Location</h2>{summary_table}</section>
  <section><h2>Conflict Type by City Group</h2>{city_table}</section>
  <section><h2>Top Deceleration Hotspot Cells</h2>{hotspot_table}</section>
  <section class='links'><h2>Artifacts</h2>
    <a href='conflict_pair_events.csv'>conflict_pair_events.csv</a>
    <a href='conflict_pair_episodes.csv'>conflict_pair_episodes.csv</a>
    <a href='conflict_type_summary_by_location.csv'>summary by location</a>
    <a href='conflict_type_summary_by_city.csv'>summary by city</a>
    <a href='deceleration_hotspot_events.csv'>deceleration events</a>
    <a href='deceleration_hotspot_cells.csv'>hotspot cells</a>
    <a href='summary.json'>summary.json</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SinD conflict type proportions and spatial deceleration hotspots.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_conflict_patterns_spatial_clustering"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--distance-threshold-m", type=float, default=15.0)
    parser.add_argument("--path-conflict-distance-m", type=float, default=3.0)
    parser.add_argument("--future-horizon-s", type=float, default=5.0)
    parser.add_argument("--future-stride", type=int, default=2)
    parser.add_argument("--pair-episode-max-gap-s", type=float, default=0.3)
    parser.add_argument("--decel-threshold-mps2", type=float, default=-1.5)
    parser.add_argument("--front-cone-deg", type=float, default=100.0)
    parser.add_argument("--grid-size-m", type=float, default=1.0)
    parser.add_argument("--hotspot-top-k", type=int, default=30)
    parser.add_argument("--roi-buffer-m", type=float, default=5.0)
    parser.add_argument("--core-buffer-m", type=float, default=3.0)
    parser.add_argument("--min-core-radius-m", type=float, default=12.0)
    parser.add_argument("--max-core-radius-m", type=float, default=32.0)
    parser.add_argument("--turn-heading-threshold-deg", type=float, default=25.0)
    parser.add_argument("--lane-trim-ratio", type=float, default=0.20)
    parser.add_argument("--central-quantile", type=float, default=0.62)
    parser.add_argument("--keep-static-tracks", action="store_true")
    parser.add_argument("--static-min-duration-s", type=float, default=8.0)
    parser.add_argument("--static-max-displacement-m", type=float, default=2.0)
    parser.add_argument("--static-max-path-length-m", type=float, default=5.0)
    parser.add_argument("--static-max-speed-p95-mps", type=float, default=0.15)
    parser.add_argument("--static-min-slow-duration-s", type=float, default=60.0)
    parser.add_argument("--static-max-mean-path-speed-mps", type=float, default=0.25)
    parser.add_argument("--max-scenes-per-location", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cities = [normalize_location(city) for city in args.cities]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracks for {args.cities}...", flush=True)
    tracks = load_sind_tracks(args.data_dir, args.cities)
    tracks = filter_tracks(tracks, city=args.cities, agent_type="all", maneuver=None)
    tracks = [track for track in tracks if track.class_name in ALL_DYNAMIC_CLASSES]
    tracks = _limit_scenes_per_location(tracks, args.max_scenes_per_location)
    tracks, static_tracks_df = filter_static_tracks(
        tracks,
        dt=args.dt,
        enabled=not args.keep_static_tracks,
        static_min_duration_s=args.static_min_duration_s,
        static_max_displacement_m=args.static_max_displacement_m,
        static_max_path_length_m=args.static_max_path_length_m,
        static_max_speed_p95_mps=args.static_max_speed_p95_mps,
        static_min_slow_duration_s=args.static_min_slow_duration_s,
        static_max_mean_path_speed_mps=args.static_max_mean_path_speed_mps,
    )
    if not static_tracks_df.empty:
        print(f"Filtered {len(static_tracks_df)} long-duration static tracks.", flush=True)

    print("Building ROIs and loading lane references...", flush=True)
    core_rois, buffered_rois, refs_by_city, roi_debug = build_rois(
        args.data_dir,
        args.cities,
        roi_buffer_m=args.roi_buffer_m,
        core_buffer_m=args.core_buffer_m,
        max_core_radius_m=args.max_core_radius_m,
        min_core_radius_m=args.min_core_radius_m,
        turn_heading_threshold_deg=args.turn_heading_threshold_deg,
        lane_trim_ratio=args.lane_trim_ratio,
        central_quantile=args.central_quantile,
    )
    write_geojson(args.output_dir / "conflict_rois.geojson", core_rois, buffered_rois, roi_debug)

    print(f"Computing conflict pairs and deceleration hotspots over {len(tracks):,} dynamic tracks...", flush=True)
    pair_df, hotspot_df = compute_conflicts(
        tracks,
        buffered_rois=buffered_rois,
        dt=args.dt,
        distance_threshold_m=args.distance_threshold_m,
        path_conflict_distance_m=args.path_conflict_distance_m,
        future_horizon_s=args.future_horizon_s,
        future_stride=args.future_stride,
        decel_threshold_mps2=args.decel_threshold_mps2,
        front_cone_deg=args.front_cone_deg,
    )
    pair_episode_df = merge_pair_episodes(pair_df, dt=args.dt, max_gap_s=args.pair_episode_max_gap_s)

    location_groups = list(args.cities)
    city_groups = ["ChangChun", "TianJin", "ChongQing", "XiAn"]
    loc_summary = summarize_conflict_types(pair_episode_df, "location", location_groups)
    city_summary = summarize_conflict_types(pair_episode_df, "city_group", city_groups)
    hotspot_cells, grids = extract_hotspot_cells(hotspot_df, buffered_rois, args.grid_size_m, args.hotspot_top_k)

    pair_df.to_csv(args.output_dir / "conflict_pair_events.csv", index=False)
    pair_episode_df.to_csv(args.output_dir / "conflict_pair_episodes.csv", index=False)
    loc_summary.to_csv(args.output_dir / "conflict_type_summary_by_location.csv", index=False)
    city_summary.to_csv(args.output_dir / "conflict_type_summary_by_city.csv", index=False)
    hotspot_df.to_csv(args.output_dir / "deceleration_hotspot_events.csv", index=False)
    hotspot_cells.to_csv(args.output_dir / "deceleration_hotspot_cells.csv", index=False)
    static_tracks_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)

    for city in args.cities:
        plot_chord(pair_episode_df, "location", city, args.output_dir / f"conflict_chord_{city}.png", f"{LOCATION_DISPLAY.get(city, city)} Conflict Maneuver Chord")
        grid, x_edges, y_edges = grids[city]
        np.savez_compressed(args.output_dir / f"conflict_hotspot_grid_{city}.npz", grid=grid, x_edges=x_edges, y_edges=y_edges, roi_polygon_xy=np.asarray(buffered_rois[city].exterior.coords, dtype=float), grid_size_m=float(args.grid_size_m))
        plot_hotspot_map(city, grid, x_edges, y_edges, buffered_rois[city], core_rois[city], refs_by_city.get(city, []), args.output_dir / f"conflict_hotspot_{city}.png")
    # City-group chord uses all episodes colored by maneuver; group label is not a node.
    plot_chord(pair_episode_df.assign(all_city_groups="All City Groups"), "all_city_groups", "All City Groups", args.output_dir / "conflict_chord_city_groups.png", "All City Groups Conflict Maneuver Chord")

    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "dt": args.dt,
        "distance_threshold_m": args.distance_threshold_m,
        "path_conflict_distance_m": args.path_conflict_distance_m,
        "future_horizon_s": args.future_horizon_s,
        "decel_threshold_mps2": args.decel_threshold_mps2,
        "front_cone_deg": args.front_cone_deg,
        "grid_size_m": args.grid_size_m,
        "static_filter_enabled": not args.keep_static_tracks,
        "filtered_static_tracks_total": int(len(static_tracks_df)),
        "dynamic_tracks": int(len(tracks)),
        "conflict_pair_events": int(len(pair_df)),
        "conflict_pair_episodes": int(len(pair_episode_df)),
        "deceleration_hotspot_events": int(len(hotspot_df)),
        "summary_by_location": loc_summary.to_dict("records"),
        "summary_by_city": city_summary.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html_report(args.output_dir / "index.html", loc_summary, city_summary, hotspot_cells, args.cities)

    print(f"Wrote outputs to: {args.output_dir}")
    print(loc_summary.to_string(index=False))
    if not hotspot_df.empty:
        print(hotspot_df.groupby("location").size().rename("deceleration_hotspot_events").to_string())


if __name__ == "__main__":
    main()
