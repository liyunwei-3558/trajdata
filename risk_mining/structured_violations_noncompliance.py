#!/usr/bin/env python
"""Estimate SinD structured-violation and VRU non-compliance baselines.

This metric is intentionally conservative.  Items that cannot be judged from the
current map/signal bindings are exported as proxy or unavailable diagnostics
instead of being reported as strict legal violations.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import pickle
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon, mapping

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
from lateral_deviation_variance import (  # noqa: E402
    DEFAULT_SIX_LOCATIONS,
    LaneReference,
    load_lane_references,
    project_points_to_lane,
)
from intersection_spatiotemporal_density import (  # noqa: E402
    filter_static_tracks,
    infer_intersection_core_roi,
)
from trajdata.dataset_specific.sind.sind_lanelet2_utils import LL2XYProjector, orgins  # noqa: E402
from trajdata.dataset_specific.sind.sind_traffic_lights import (  # noqa: E402
    DEFAULT_MAPPING_PATH,
    build_traffic_light_dataframe,
    configured_traffic_light_root,
)
from trajdata.maps import TrafficLightStatus  # noqa: E402

MOTOR_VEHICLE_CLASSES: Tuple[str, ...] = ("car", "truck", "bus", "tricycle")
VRU_CLASSES: Tuple[str, ...] = ("pedestrian", "bicycle", "motorcycle")
DEFAULT_TL_ROOT = Path("/home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv")


@dataclass
class LaneIndex:
    tree: cKDTree
    points: np.ndarray
    lane_indices: np.ndarray
    lane_ids: np.ndarray
    directions: np.ndarray


@dataclass
class CrosswalkSet:
    polygons: List[Polygon]
    paths: List[MplPath]

    @property
    def available(self) -> bool:
        return bool(self.polygons)


def _lanelet2_path(data_dir: Path, location: str) -> Path:
    path = data_dir / "Lanelet_maps_SinD" / f"lanelet2_{location}.osm"
    if path.exists():
        return path
    fallback = REPO_ROOT / "Lanelet_maps_SinD" / f"lanelet2_{location}.osm"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Lanelet2 map not found for {location}: {path}")


def _safe_literal_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "<missing>"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def load_track_metadata(data_dir: Path, locations: Sequence[str]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Load optional official behavior labels that are not retained in TrackRecord."""
    rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for location in locations:
        tp_path = data_dir / location / f"tp_info_{location}.pkl"
        if not tp_path.exists():
            continue
        with tp_path.open("rb") as handle:
            city_tp_info = pickle.load(handle)
        for scene_id, scene_tracks in city_tp_info.items():
            for raw_agent_id, tp_data in scene_tracks.items():
                agent_id = str(tp_data.get("ID", raw_agent_id))
                signal_labels = _safe_literal_list(tp_data.get("Signal_Violation_Behavior"))
                rows[(location, str(scene_id), agent_id)] = {
                    "signal_violation_behavior": ";".join(signal_labels),
                    "official_red_light_running": any("red" in item.lower() for item in signal_labels),
                    "official_yellow_light_running": any("yellow" in item.lower() for item in signal_labels),
                    "retrograde_type": str(tp_data.get("retrograde_type", "")),
                    "cross_type": ";".join(_safe_literal_list(tp_data.get("CrossType"))),
                }
    return rows


def _dedupe_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6
    return points[keep]


def _sample_lane_points(lane: LaneReference, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    xy = _dedupe_points(lane.xy.astype(float))
    if len(xy) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    cumulative = lane.cumulative_s
    if len(cumulative) != len(lane.xy):
        seg_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.empty((0, 2)), np.empty((0, 2))
    num = max(2, int(math.ceil(total / spacing)) + 1)
    s_grid = np.linspace(0.0, total, num)
    px = np.interp(s_grid, cumulative, lane.xy[:, 0])
    py = np.interp(s_grid, cumulative, lane.xy[:, 1])
    eps = max(spacing, 0.25)
    s0 = np.clip(s_grid - eps, 0.0, total)
    s1 = np.clip(s_grid + eps, 0.0, total)
    x0 = np.interp(s0, cumulative, lane.xy[:, 0])
    y0 = np.interp(s0, cumulative, lane.xy[:, 1])
    x1 = np.interp(s1, cumulative, lane.xy[:, 0])
    y1 = np.interp(s1, cumulative, lane.xy[:, 1])
    direction = np.column_stack([x1 - x0, y1 - y0])
    norm = np.linalg.norm(direction, axis=1)
    valid = norm > 1e-9
    direction[valid] = direction[valid] / norm[valid, None]
    direction[~valid] = np.array([1.0, 0.0])
    return np.column_stack([px, py]), direction


def build_lane_indices(references: Mapping[str, Sequence[LaneReference]], spacing: float) -> Dict[str, LaneIndex]:
    indices: Dict[str, LaneIndex] = {}
    for location, lanes in references.items():
        points: List[np.ndarray] = []
        directions: List[np.ndarray] = []
        lane_indices: List[int] = []
        lane_ids: List[str] = []
        for lane_idx, lane in enumerate(lanes):
            sample_xy, sample_dir = _sample_lane_points(lane, spacing)
            if len(sample_xy) == 0:
                continue
            points.append(sample_xy)
            directions.append(sample_dir)
            lane_indices.extend([lane_idx] * len(sample_xy))
            lane_ids.extend([lane.lane_id] * len(sample_xy))
        if points:
            all_points = np.vstack(points)
            indices[location] = LaneIndex(
                tree=cKDTree(all_points),
                points=all_points,
                lane_indices=np.asarray(lane_indices, dtype=int),
                lane_ids=np.asarray(lane_ids, dtype=object),
                directions=np.vstack(directions),
            )
    return indices


def query_lane_index(index: LaneIndex, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, sample_idx = index.tree.query(xy, k=1)
    sample_idx = np.asarray(sample_idx, dtype=int)
    return (
        np.asarray(distances, dtype=float),
        index.lane_indices[sample_idx],
        index.lane_ids[sample_idx],
        index.directions[sample_idx],
    )


def _track_state(track: TrackRecord) -> pd.DataFrame:
    state = track.state.sort_values("frame_id").copy()
    required = ["x", "y"]
    if any(col not in state.columns for col in required):
        return pd.DataFrame()
    mask = np.isfinite(state[required].to_numpy(dtype=float)).all(axis=1)
    return state.loc[mask].reset_index(drop=True)


def _track_xy_time_vel(track: TrackRecord, dt: float) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = _track_state(track)
    if state.empty:
        return state, np.empty((0, 2)), np.array([]), np.empty((0, 2)), np.array([])
    xy = state[["x", "y"]].to_numpy(dtype=float)
    if "timestamp_ms" in state.columns:
        time_s = state["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    elif "frame_id" in state.columns:
        time_s = state["frame_id"].to_numpy(dtype=float) * dt
    else:
        time_s = np.arange(len(state), dtype=float) * dt
    if {"vx", "vy"}.issubset(state.columns):
        vel = state[["vx", "vy"]].to_numpy(dtype=float)
    elif len(xy) >= 2:
        vel = np.gradient(xy, dt, axis=0)
    else:
        vel = np.zeros_like(xy)
    speed = np.linalg.norm(vel, axis=1)
    return state, xy, time_s, vel, speed


def _points_inside_polygon(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    return MplPath(np.asarray(polygon.exterior.coords, dtype=float)).contains_points(points)


def _points_inside_crosswalk(points: np.ndarray, crosswalks: CrosswalkSet) -> np.ndarray:
    if len(points) == 0 or not crosswalks.available:
        return np.zeros((len(points),), dtype=bool)
    mask = np.zeros(len(points), dtype=bool)
    for path in crosswalks.paths:
        mask |= path.contains_points(points)
    return mask


def _run_lengths(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _dominant_value(values: Sequence[Any]) -> str:
    if len(values) == 0:
        return ""
    series = pd.Series(list(values))
    if series.empty:
        return ""
    return str(series.value_counts().index[0])


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _subpath_length(xy: np.ndarray, mask: np.ndarray) -> float:
    if len(xy) < 2 or len(mask) != len(xy):
        return 0.0
    seg_mask = mask[:-1] & mask[1:]
    if not np.any(seg_mask):
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0)[seg_mask], axis=1).sum())


def _parse_osm_crosswalks(data_dir: Path, location: str) -> CrosswalkSet:
    path = _lanelet2_path(data_dir, location)
    tree = ET.parse(path)
    root = tree.getroot()
    projector = LL2XYProjector(orgins[location][0], orgins[location][1])

    nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, List[int]] = {}
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        x, y = projector.latlon2xy(lat, lon)
        nodes[node_id] = (float(x), float(y))
    for way in root.findall("way"):
        ways[int(way.get("id"))] = [int(nd.get("ref")) for nd in way.findall("nd")]

    polygons: List[Polygon] = []
    for relation in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "crosswalk":
            continue
        boundary_points: List[Tuple[float, float]] = []
        for member in relation.findall("member"):
            ref = int(member.get("ref"))
            if member.get("role") not in {"left", "right"} or ref not in ways:
                continue
            boundary_points.extend(nodes[nid] for nid in ways[ref] if nid in nodes)
        if len(boundary_points) < 3:
            continue
        hull = MultiPoint(boundary_points).convex_hull
        if hull.geom_type == "Polygon" and hull.area > 0.1:
            polygons.append(hull)

    paths = [MplPath(np.asarray(poly.exterior.coords, dtype=float)) for poly in polygons]
    return CrosswalkSet(polygons=polygons, paths=paths)


def load_crosswalks(data_dir: Path, locations: Sequence[str]) -> Dict[str, CrosswalkSet]:
    result: Dict[str, CrosswalkSet] = {}
    for location in locations:
        result[location] = _parse_osm_crosswalks(data_dir, location)
    return result


def _mapping_has_lane_bindings(mapping_path: Path, locations: Sequence[str]) -> Tuple[bool, Dict[str, int]]:
    if not mapping_path.exists():
        return False, {loc: 0 for loc in locations}
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    counts: Dict[str, int] = {}
    for location in locations:
        loc_map = mapping.get("locations", {}).get(location, {})
        count = sum(len(v) for v in loc_map.get("light_to_lanes", {}).values())
        for scene_map in loc_map.get("scenes", {}).values():
            count += sum(len(v) for v in scene_map.get("light_to_lanes", {}).values())
        counts[location] = int(count)
    return any(counts.values()), counts


def _scene_length(track_group: Sequence[TrackRecord]) -> int:
    max_frame = 0
    for track in track_group:
        if "frame_id" in track.state.columns and not track.state.empty:
            max_frame = max(max_frame, int(pd.to_numeric(track.state["frame_id"], errors="coerce").max()))
    return max_frame + 1


def build_scene_signal_tables(
    tracks_by_scene: Mapping[Tuple[str, str], List[TrackRecord]],
    dt: float,
    traffic_light_root: Optional[Path],
    mapping_path: Path,
    enabled: bool,
) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], pd.DataFrame]:
    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    reports: List[Dict[str, Any]] = []
    if not enabled:
        return tables, pd.DataFrame()
    for (location, scene_id), scene_tracks in tracks_by_scene.items():
        scene_length = _scene_length(scene_tracks)
        scene_name = f"{location}_{scene_id}"
        table, report = build_traffic_light_dataframe(
            scene_name=scene_name,
            location=location,
            scene_id=scene_id,
            scene_length=scene_length,
            scene_dt=dt,
            root=traffic_light_root,
            mapping_path=mapping_path,
        )
        reports.append(report.to_dict())
        if table is not None:
            tables[(location, scene_id)] = table
    return tables, pd.DataFrame(reports)


def _status_at_scene_ts(tls_df: pd.DataFrame, lane_id: str, scene_ts: int) -> Optional[int]:
    try:
        lane_df = tls_df.xs(lane_id, level="lane_id")
    except KeyError:
        return None
    if scene_ts in lane_df.index:
        value = lane_df.loc[scene_ts, "status"]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return int(value)
    prior = lane_df[lane_df.index <= scene_ts]
    if prior.empty:
        return None
    return int(prior.iloc[-1]["status"])


def analyze_vehicle_tracks(
    tracks: Sequence[TrackRecord],
    references: Mapping[str, Sequence[LaneReference]],
    lane_indices: Mapping[str, LaneIndex],
    roi_polygons: Mapping[str, Polygon],
    metadata: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    dt: float,
    lane_match_threshold_m: float,
    wrong_way_heading_threshold_deg: float,
    min_wrong_way_duration_s: float,
    min_lane_change_run_s: float,
    min_lane_change_speed_mps: float,
    red_entry_roi_enabled: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    lane_diag_counts: Dict[Tuple[str, str], Dict[str, float]] = {}
    cos_threshold = math.cos(math.radians(wrong_way_heading_threshold_deg))
    min_wrong_pts = max(1, int(math.ceil(min_wrong_way_duration_s / dt)))
    min_run_pts = max(1, int(math.ceil(min_lane_change_run_s / dt)))

    for idx, track in enumerate(tracks, start=1):
        state, xy, time_s, vel, speed = _track_xy_time_vel(track, dt)
        if len(xy) < 2 or track.location not in lane_indices:
            continue
        lane_index = lane_indices[track.location]
        distances, lane_idx, lane_ids, lane_dirs = query_lane_index(lane_index, xy)
        valid_lane = (distances <= lane_match_threshold_m) & np.isfinite(speed)
        moving = speed >= min_lane_change_speed_mps
        vel_unit = np.divide(vel, np.maximum(speed[:, None], 1e-6), out=np.zeros_like(vel), where=speed[:, None] > 1e-6)
        cos_lane = np.sum(vel_unit * lane_dirs, axis=1)
        wrong_mask = valid_lane & (speed >= min_lane_change_speed_mps) & (cos_lane < cos_threshold)
        dominant_lane = _dominant_value(lane_ids[valid_lane])

        for lid in np.unique(lane_ids[valid_lane]):
            loc_key = (track.location, str(lid))
            sample_mask = valid_lane & (lane_ids == lid) & moving
            if not np.any(sample_mask):
                continue
            row = lane_diag_counts.setdefault(loc_key, {"samples": 0.0, "wrong_samples": 0.0})
            row["samples"] += float(np.count_nonzero(sample_mask))
            row["wrong_samples"] += float(np.count_nonzero(sample_mask & wrong_mask))

        meta = metadata.get((track.location, track.scene_id, track.agent_id), {})
        official_red = bool(meta.get("official_red_light_running", False))
        official_yellow = bool(meta.get("official_yellow_light_running", False))
        retrograde_type = str(meta.get("retrograde_type", ""))
        official_retrograde = retrograde_type in {"front_retrograde", "rear_retrograde", "full_retrograde"}

        wrong_runs = [(start, end) for start, end in _run_lengths(wrong_mask) if end - start >= min_wrong_pts]
        wrong_seconds = float(np.count_nonzero(wrong_mask) * dt)
        if wrong_runs:
            longest = max(wrong_runs, key=lambda item: item[1] - item[0])
            mid = (longest[0] + longest[1] - 1) // 2
            event_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    "event_type": "wrong_way_or_opposing_lane_proxy",
                    "severity": "proxy",
                    "x": float(xy[mid, 0]),
                    "y": float(xy[mid, 1]),
                    "time_s": float(time_s[mid]) if len(time_s) else float(mid * dt),
                    "duration_s": wrong_seconds,
                    "dominant_lane_id": dominant_lane,
                    "nearest_lane_id": str(lane_ids[mid]),
                    "lane_distance_m": float(distances[mid]),
                    "speed_mps": float(speed[mid]),
                    "cos_to_lane_direction": float(cos_lane[mid]),
                    "official_red_light_running": official_red,
                    "official_yellow_light_running": official_yellow,
                    "official_retrograde_type": retrograde_type,
                    "note": "Proxy; sensitive to Lanelet2 centerline direction correctness.",
                }
            )

        roi_mask = _points_inside_polygon(xy, roi_polygons[track.location])
        in_roi_lane = roi_mask & valid_lane & moving
        runs = []
        for start, end in _run_lengths(in_roi_lane):
            if end - start < min_run_pts:
                continue
            run_lane = _dominant_value(lane_ids[start:end])
            if run_lane:
                runs.append((start, end, run_lane))
        compressed: List[Tuple[int, int, str]] = []
        for run in runs:
            if compressed and compressed[-1][2] == run[2]:
                compressed[-1] = (compressed[-1][0], run[1], run[2])
            else:
                compressed.append(run)
        lane_switches = 0
        if len(compressed) >= 2:
            lane_switches = sum(1 for a, b in zip(compressed[:-1], compressed[1:]) if a[2] != b[2])
        if lane_switches > 0:
            switch_idx = compressed[1][0]
            event_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    "event_type": "intersection_lane_switch_proxy",
                    "severity": "proxy",
                    "x": float(xy[switch_idx, 0]),
                    "y": float(xy[switch_idx, 1]),
                    "time_s": float(time_s[switch_idx]) if len(time_s) else float(switch_idx * dt),
                    "duration_s": float(np.count_nonzero(in_roi_lane) * dt),
                    "dominant_lane_id": dominant_lane,
                    "nearest_lane_id": str(lane_ids[switch_idx]),
                    "lane_distance_m": float(distances[switch_idx]),
                    "speed_mps": float(speed[switch_idx]),
                    "cos_to_lane_direction": float(cos_lane[switch_idx]),
                    "official_red_light_running": official_red,
                    "official_yellow_light_running": official_yellow,
                    "official_retrograde_type": retrograde_type,
                    "note": "Proxy for lane switching inside inferred intersection core; map lacks solid-line semantics.",
                }
            )

        red_roi_candidate = False
        red_status_name = "unavailable"
        if red_entry_roi_enabled and signal_tables:
            roi_indices = np.flatnonzero(roi_mask & valid_lane)
            if len(roi_indices):
                entry_idx = int(roi_indices[0])
                tls_df = signal_tables.get((track.location, track.scene_id))
                if tls_df is not None:
                    lane_id = str(lane_ids[entry_idx])
                    scene_ts = int(state.iloc[entry_idx]["frame_id"]) if "frame_id" in state.columns else entry_idx
                    status = _status_at_scene_ts(tls_df, lane_id, scene_ts)
                    if status is not None:
                        red_status_name = TrafficLightStatus(status).name
                        if status == int(TrafficLightStatus.RED):
                            red_roi_candidate = True
                            event_rows.append(
                                {
                                    "location": track.location,
                                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                                    "scene_id": track.scene_id,
                                    "agent_id": track.agent_id,
                                    "class_name": track.class_name,
                                    "event_type": "red_at_roi_entry_candidate",
                                    "severity": "candidate",
                                    "x": float(xy[entry_idx, 0]),
                                    "y": float(xy[entry_idx, 1]),
                                    "time_s": float(time_s[entry_idx]) if len(time_s) else float(entry_idx * dt),
                                    "duration_s": 0.0,
                                    "dominant_lane_id": dominant_lane,
                                    "nearest_lane_id": lane_id,
                                    "lane_distance_m": float(distances[entry_idx]),
                                    "speed_mps": float(speed[entry_idx]),
                                    "cos_to_lane_direction": float(cos_lane[entry_idx]),
                                    "official_red_light_running": official_red,
                                    "official_yellow_light_running": official_yellow,
                                    "official_retrograde_type": retrograde_type,
                                    "note": "Candidate only; needs verified light-to-lane and stop-line binding before legal use.",
                                }
                            )

        if official_red:
            idx_mid = int(np.flatnonzero(roi_mask)[0]) if np.any(roi_mask) else len(xy) // 2
            event_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    "event_type": "official_red_light_running_label",
                    "severity": "official_label",
                    "x": float(xy[idx_mid, 0]),
                    "y": float(xy[idx_mid, 1]),
                    "time_s": float(time_s[idx_mid]) if len(time_s) else float(idx_mid * dt),
                    "duration_s": 0.0,
                    "dominant_lane_id": dominant_lane,
                    "nearest_lane_id": str(lane_ids[idx_mid]),
                    "lane_distance_m": float(distances[idx_mid]),
                    "speed_mps": float(speed[idx_mid]),
                    "cos_to_lane_direction": float(cos_lane[idx_mid]),
                    "official_red_light_running": official_red,
                    "official_yellow_light_running": official_yellow,
                    "official_retrograde_type": retrograde_type,
                    "note": "Official SinD trajectory label when present; available mainly in Tianjin pkl metadata.",
                }
            )

        track_rows.append(
            {
                "location": track.location,
                "city": LOCATION_DISPLAY.get(track.location, track.location),
                "scene_id": track.scene_id,
                "agent_id": track.agent_id,
                "class_name": track.class_name,
                "maneuver": track.maneuver,
                "points": int(len(xy)),
                "duration_s": float(len(xy) * dt),
                "path_length_m": _path_length(xy),
                "dominant_lane_id": dominant_lane,
                "lane_matched_point_ratio": float(np.mean(valid_lane)) if len(valid_lane) else 0.0,
                "seconds_in_intersection_core": float(np.count_nonzero(roi_mask) * dt),
                "wrong_way_proxy_seconds": wrong_seconds,
                "wrong_way_proxy_event": bool(wrong_runs),
                "intersection_lane_switch_count": int(lane_switches),
                "intersection_lane_switch_proxy_event": bool(lane_switches > 0),
                "official_red_light_running": official_red,
                "official_yellow_light_running": official_yellow,
                "official_retrograde_type": retrograde_type,
                "official_retrograde_event": official_retrograde,
                "red_at_roi_entry_candidate": bool(red_roi_candidate),
                "red_at_roi_entry_status": red_status_name,
            }
        )
        if idx % 5000 == 0:
            print(f"Analyzed {idx}/{len(tracks)} vehicle tracks...", flush=True)

    lane_diag_rows = []
    for (location, lane_id), counts in lane_diag_counts.items():
        samples = counts["samples"]
        wrong = counts["wrong_samples"]
        lane_diag_rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "lane_id": lane_id,
                "moving_matched_samples": int(samples),
                "wrong_way_proxy_samples": int(wrong),
                "wrong_way_proxy_ratio": float(wrong / samples) if samples else 0.0,
                "possible_map_direction_error": bool(samples >= 50 and wrong / samples >= 0.60),
            }
        )

    event_df = pd.DataFrame(event_rows)
    track_df = pd.DataFrame(track_rows)
    lane_diag_df = pd.DataFrame(lane_diag_rows)
    flagged: set[Tuple[str, str]] = set()
    if not lane_diag_df.empty:
        flagged = set(
            zip(
                lane_diag_df.loc[lane_diag_df["possible_map_direction_error"], "location"],
                lane_diag_df.loc[lane_diag_df["possible_map_direction_error"], "lane_id"],
            )
        )
    if not event_df.empty:
        event_df["possible_map_direction_error"] = [
            (row.location, str(row.nearest_lane_id)) in flagged or (row.location, str(row.dominant_lane_id)) in flagged
            for row in event_df.itertuples()
        ]
    if not track_df.empty:
        track_df["possible_map_direction_error"] = [
            (row.location, str(row.dominant_lane_id)) in flagged for row in track_df.itertuples()
        ]
        if not event_df.empty and "possible_map_direction_error" in event_df.columns:
            flagged_event_keys = set(
                zip(
                    event_df.loc[
                        (event_df["event_type"] == "wrong_way_or_opposing_lane_proxy")
                        & event_df["possible_map_direction_error"],
                        "location",
                    ],
                    event_df.loc[
                        (event_df["event_type"] == "wrong_way_or_opposing_lane_proxy")
                        & event_df["possible_map_direction_error"],
                        "scene_id",
                    ],
                    event_df.loc[
                        (event_df["event_type"] == "wrong_way_or_opposing_lane_proxy")
                        & event_df["possible_map_direction_error"],
                        "agent_id",
                    ],
                )
            )
        else:
            flagged_event_keys = set()
        track_df["wrong_way_proxy_event_map_checked"] = [
            bool(row.wrong_way_proxy_event)
            and (row.location, row.scene_id, row.agent_id) not in flagged_event_keys
            and not bool(row.possible_map_direction_error)
            for row in track_df.itertuples()
        ]
    return event_df, track_df, lane_diag_df


def analyze_vru_tracks(
    tracks: Sequence[TrackRecord],
    lane_indices: Mapping[str, LaneIndex],
    roi_polygons: Mapping[str, Polygon],
    crosswalks: Mapping[str, CrosswalkSet],
    dt: float,
    motor_lane_threshold_m: float,
    vru_event_min_length_ratio: float,
    vru_event_min_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks, start=1):
        _, xy, time_s, _, speed = _track_xy_time_vel(track, dt)
        if len(xy) < 2 or track.location not in lane_indices:
            continue
        distances, _, lane_ids, _ = query_lane_index(lane_indices[track.location], xy)
        near_motor_lane = distances <= motor_lane_threshold_m
        roi_mask = _points_inside_polygon(xy, roi_polygons[track.location])
        cw = crosswalks[track.location]
        in_crosswalk = _points_inside_crosswalk(xy, cw)
        outside_crosswalk_core = roi_mask & (~in_crosswalk if cw.available else np.ones(len(xy), dtype=bool))

        total_length = _path_length(xy)
        motor_lane_length = _subpath_length(xy, near_motor_lane)
        motor_lane_seconds = float(np.count_nonzero(near_motor_lane) * dt)
        motor_lane_length_ratio = float(motor_lane_length / total_length) if total_length > 1e-6 else 0.0
        outside_core_seconds = float(np.count_nonzero(outside_crosswalk_core) * dt)
        conflict_zone_seconds_verified = outside_core_seconds if cw.available else float("nan")
        event_reason: List[str] = []
        if motor_lane_length_ratio >= vru_event_min_length_ratio and motor_lane_length >= 1.0:
            event_reason.append("motor_lane_path_ratio")
        if motor_lane_seconds >= vru_event_min_seconds:
            event_reason.append("motor_lane_seconds")
        if cw.available and outside_core_seconds >= vru_event_min_seconds:
            event_reason.append("outside_crosswalk_conflict_zone_seconds")
        elif (not cw.available) and outside_core_seconds >= vru_event_min_seconds:
            event_reason.append("core_conflict_zone_seconds_crosswalk_unavailable")

        if event_reason:
            candidate_mask = near_motor_lane | outside_crosswalk_core
            event_indices = np.flatnonzero(candidate_mask)
            event_idx = int(event_indices[len(event_indices) // 2]) if len(event_indices) else len(xy) // 2
            event_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    "event_type": "vru_motor_lane_or_conflict_zone_encroachment",
                    "severity": "proxy" if not cw.available else "map_rule",
                    "x": float(xy[event_idx, 0]),
                    "y": float(xy[event_idx, 1]),
                    "time_s": float(time_s[event_idx]) if len(time_s) else float(event_idx * dt),
                    "speed_mps": float(speed[event_idx]) if len(speed) else 0.0,
                    "nearest_lane_id": str(lane_ids[event_idx]),
                    "lane_distance_m": float(distances[event_idx]),
                    "motor_lane_seconds": motor_lane_seconds,
                    "motor_lane_path_length_m": motor_lane_length,
                    "motor_lane_length_ratio": motor_lane_length_ratio,
                    "seconds_in_intersection_core": float(np.count_nonzero(roi_mask) * dt),
                    "outside_crosswalk_core_seconds": outside_core_seconds,
                    "outside_crosswalk_core_seconds_verified": conflict_zone_seconds_verified,
                    "crosswalk_available": cw.available,
                    "event_reason": ";".join(event_reason),
                    "note": "VRU lane occupancy uses distance-to-motor-centerline; red-phase non-compliance needs pedestrian-light binding.",
                }
            )

        track_rows.append(
            {
                "location": track.location,
                "city": LOCATION_DISPLAY.get(track.location, track.location),
                "scene_id": track.scene_id,
                "agent_id": track.agent_id,
                "class_name": track.class_name,
                "points": int(len(xy)),
                "duration_s": float(len(xy) * dt),
                "path_length_m": total_length,
                "motor_lane_seconds": motor_lane_seconds,
                "motor_lane_path_length_m": motor_lane_length,
                "motor_lane_length_ratio": motor_lane_length_ratio,
                "seconds_in_intersection_core": float(np.count_nonzero(roi_mask) * dt),
                "outside_crosswalk_core_seconds": outside_core_seconds,
                "outside_crosswalk_core_seconds_verified": conflict_zone_seconds_verified,
                "crosswalk_available": cw.available,
                "encroachment_event": bool(event_reason),
            }
        )
        if idx % 5000 == 0:
            print(f"Analyzed {idx}/{len(tracks)} VRU tracks...", flush=True)
    return pd.DataFrame(event_rows), pd.DataFrame(track_rows)


def summarize_results(
    locations: Sequence[str],
    vehicle_track_df: pd.DataFrame,
    vehicle_event_df: pd.DataFrame,
    vru_track_df: pd.DataFrame,
    vru_event_df: pd.DataFrame,
    lane_diag_df: pd.DataFrame,
    mapping_counts: Mapping[str, int],
    crosswalks: Mapping[str, CrosswalkSet],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for location in locations:
        veh = vehicle_track_df[vehicle_track_df["location"] == location] if not vehicle_track_df.empty else pd.DataFrame()
        vru = vru_track_df[vru_track_df["location"] == location] if not vru_track_df.empty else pd.DataFrame()
        ve = vehicle_event_df[vehicle_event_df["location"] == location] if not vehicle_event_df.empty else pd.DataFrame()
        vu = vru_event_df[vru_event_df["location"] == location] if not vru_event_df.empty else pd.DataFrame()
        lane_diag = lane_diag_df[lane_diag_df["location"] == location] if not lane_diag_df.empty else pd.DataFrame()
        vehicle_count = len(veh)
        vru_count = len(vru)
        rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "vehicle_tracks": int(vehicle_count),
                "vru_tracks": int(vru_count),
                "wrong_way_proxy_tracks": int(veh["wrong_way_proxy_event"].sum()) if not veh.empty else 0,
                "wrong_way_proxy_rate": float(veh["wrong_way_proxy_event"].mean()) if vehicle_count else 0.0,
                "wrong_way_proxy_map_checked_tracks": int(veh["wrong_way_proxy_event_map_checked"].sum()) if (not veh.empty and "wrong_way_proxy_event_map_checked" in veh) else 0,
                "wrong_way_proxy_map_checked_rate": float(veh["wrong_way_proxy_event_map_checked"].mean()) if (vehicle_count and "wrong_way_proxy_event_map_checked" in veh) else 0.0,
                "intersection_lane_switch_proxy_tracks": int(veh["intersection_lane_switch_proxy_event"].sum()) if not veh.empty else 0,
                "intersection_lane_switch_proxy_rate": float(veh["intersection_lane_switch_proxy_event"].mean()) if vehicle_count else 0.0,
                "official_red_light_running_tracks": int(veh["official_red_light_running"].sum()) if not veh.empty else 0,
                "official_red_light_running_rate": float(veh["official_red_light_running"].mean()) if vehicle_count else 0.0,
                "official_yellow_light_running_tracks": int(veh["official_yellow_light_running"].sum()) if not veh.empty else 0,
                "official_retrograde_tracks": int(veh["official_retrograde_event"].sum()) if not veh.empty else 0,
                "red_at_roi_entry_candidates": int(veh["red_at_roi_entry_candidate"].sum()) if not veh.empty else 0,
                "red_light_binding_lane_entries": int(mapping_counts.get(location, 0)),
                "red_light_strict_available": bool(mapping_counts.get(location, 0) > 0),
                "vru_encroachment_tracks": int(vru["encroachment_event"].sum()) if not vru.empty else 0,
                "vru_encroachment_rate": float(vru["encroachment_event"].mean()) if vru_count else 0.0,
                "mean_vru_motor_lane_length_ratio": float(vru["motor_lane_length_ratio"].mean()) if not vru.empty else 0.0,
                "vru_motor_lane_total_seconds": float(vru["motor_lane_seconds"].sum()) if not vru.empty else 0.0,
                "vru_conflict_zone_total_seconds_verified": float(vru["outside_crosswalk_core_seconds_verified"].sum(skipna=True)) if not vru.empty else 0.0,
                "crosswalk_polygons": int(len(crosswalks[location].polygons)),
                "map_direction_warning_lanes": int(lane_diag["possible_map_direction_error"].sum()) if not lane_diag.empty else 0,
                "vehicle_event_rows": int(len(ve)),
                "vru_event_rows": int(len(vu)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_city(summary_by_location: pd.DataFrame) -> pd.DataFrame:
    if summary_by_location.empty:
        return pd.DataFrame()
    rows = []
    for city, group in summary_by_location.groupby("city"):
        vehicle_tracks = int(group["vehicle_tracks"].sum())
        vru_tracks = int(group["vru_tracks"].sum())
        rows.append(
            {
                "city": city,
                "locations": ";".join(group["location"].astype(str).tolist()),
                "vehicle_tracks": vehicle_tracks,
                "vru_tracks": vru_tracks,
                "wrong_way_proxy_tracks": int(group["wrong_way_proxy_tracks"].sum()),
                "wrong_way_proxy_rate": float(group["wrong_way_proxy_tracks"].sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "wrong_way_proxy_map_checked_tracks": int(group.get("wrong_way_proxy_map_checked_tracks", pd.Series(dtype=float)).sum()),
                "wrong_way_proxy_map_checked_rate": float(group.get("wrong_way_proxy_map_checked_tracks", pd.Series(dtype=float)).sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "intersection_lane_switch_proxy_tracks": int(group["intersection_lane_switch_proxy_tracks"].sum()),
                "intersection_lane_switch_proxy_rate": float(group["intersection_lane_switch_proxy_tracks"].sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "official_red_light_running_tracks": int(group["official_red_light_running_tracks"].sum()),
                "official_red_light_running_rate": float(group["official_red_light_running_tracks"].sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "vru_encroachment_tracks": int(group["vru_encroachment_tracks"].sum()),
                "vru_encroachment_rate": float(group["vru_encroachment_tracks"].sum() / vru_tracks) if vru_tracks else 0.0,
                "vru_motor_lane_total_seconds": float(group["vru_motor_lane_total_seconds"].sum()),
                "vru_conflict_zone_total_seconds_verified": float(group["vru_conflict_zone_total_seconds_verified"].sum()),
                "red_light_strict_available_locations": int(group["red_light_strict_available"].sum()),
                "map_direction_warning_lanes": int(group["map_direction_warning_lanes"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("city")


def _scatter_events(ax: plt.Axes, event_df: pd.DataFrame, type_colors: Mapping[str, str], label_prefix: str = "") -> None:
    if event_df.empty:
        return
    for event_type, group in event_df.groupby("event_type"):
        ax.scatter(
            group["x"],
            group["y"],
            s=22,
            alpha=0.62,
            c=type_colors.get(event_type, "#222222"),
            edgecolors="none",
            label=f"{label_prefix}{event_type} ({len(group)})",
        )


def plot_vehicle_hotspot(
    location: str,
    lane_refs: Sequence[LaneReference],
    roi_polygon: Polygon,
    event_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    for lane in lane_refs:
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#52616b", linewidth=0.6, alpha=0.35)
    roi_xy = np.asarray(roi_polygon.exterior.coords)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#111111", linewidth=2.0, label="Intersection core ROI")
    colors = {
        "wrong_way_or_opposing_lane_proxy": "#c0392b",
        "intersection_lane_switch_proxy": "#d68910",
        "official_red_light_running_label": "#8e1b14",
        "red_at_roi_entry_candidate": "#6c3483",
    }
    _scatter_events(ax, event_df, colors)
    ax.set_title(f"{LOCATION_DISPLAY.get(location, location)} vehicle structured-violation hotspots")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.18)
    if not event_df.empty:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_vru_encroachment(
    location: str,
    lane_refs: Sequence[LaneReference],
    roi_polygon: Polygon,
    crosswalks: CrosswalkSet,
    event_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    for lane in lane_refs:
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#455a64", linewidth=0.6, alpha=0.35)
    for poly in crosswalks.polygons:
        px = np.asarray(poly.exterior.coords)
        ax.fill(px[:, 0], px[:, 1], color="#7fb3d5", alpha=0.28, label="crosswalk" if "crosswalk" not in ax.get_legend_handles_labels()[1] else None)
    roi_xy = np.asarray(roi_polygon.exterior.coords)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#111111", linewidth=2.0, label="Intersection core ROI")
    colors = {"vru_motor_lane_or_conflict_zone_encroachment": "#d35400"}
    _scatter_events(ax, event_df, colors)
    ax.set_title(f"{LOCATION_DISPLAY.get(location, location)} VRU encroachment / non-compliance proxy")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    locs = summary_df["location"].tolist()
    x = np.arange(len(locs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    wrong_rate_col = "wrong_way_proxy_map_checked_rate" if "wrong_way_proxy_map_checked_rate" in summary_df else "wrong_way_proxy_rate"
    ax.bar(x - width, summary_df[wrong_rate_col], width, label="Wrong-way proxy (map-checked)", color="#c0392b")
    ax.bar(x, summary_df["intersection_lane_switch_proxy_rate"], width, label="Lane-switch proxy", color="#d68910")
    ax.bar(x + width, summary_df["vru_encroachment_rate"], width, label="VRU encroachment", color="#2874a6")
    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel("Track-level rate")
    ax.set_ylim(0, min(1.0, max(0.05, float(summary_df[[wrong_rate_col, "intersection_lane_switch_proxy_rate", "vru_encroachment_rate"]].max().max()) * 1.25)))
    ax.set_title("SinD violation / non-compliance baseline proxies")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    parts = ["<table><thead><tr>"]
    for _, label in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (float, np.floating)):
                cell = f"{float(value):.4f}"
            elif isinstance(value, (bool, np.bool_)):
                cell = "yes" if bool(value) else "no"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    city_df: pd.DataFrame,
    vehicle_event_df: pd.DataFrame,
    vru_event_df: pd.DataFrame,
    methodology_notes: Mapping[str, Any],
) -> None:
    summary_rows = summary_df.to_dict("records") if not summary_df.empty else []
    city_rows = city_df.to_dict("records") if not city_df.empty else []
    total_vehicle = int(summary_df["vehicle_tracks"].sum()) if not summary_df.empty else 0
    total_vru = int(summary_df["vru_tracks"].sum()) if not summary_df.empty else 0
    total_events = int(len(vehicle_event_df) + len(vru_event_df))
    cards = []
    for row in summary_rows:
        loc = row["location"]
        cards.append(
            f"""
            <article class='city-card'>
              <h3>{html.escape(str(row['city']))} <span>{html.escape(str(loc))}</span></h3>
              <b>{int(row['vehicle_tracks']):,}</b><small>vehicle tracks</small>
              <b>{float(row.get('wrong_way_proxy_map_checked_rate', row['wrong_way_proxy_rate'])):.2%}</b><small>wrong-way proxy (map-checked)</small>
              <b>{float(row['intersection_lane_switch_proxy_rate']):.2%}</b><small>intersection lane-switch proxy</small>
              <b>{float(row['vru_encroachment_rate']):.2%}</b><small>VRU encroachment proxy</small>
              <div class='thumbs'>
                <a href='vehicle_violation_hotspot_{html.escape(str(loc))}.png'><img src='vehicle_violation_hotspot_{html.escape(str(loc))}.png' alt='vehicle hotspot'></a>
                <a href='vru_encroachment_{html.escape(str(loc))}.png'><img src='vru_encroachment_{html.escape(str(loc))}.png' alt='vru encroachment'></a>
              </div>
            </article>
            """
        )
    summary_table = _html_table(
        summary_rows,
        [
            ("city", "City"),
            ("vehicle_tracks", "Vehicle tracks"),
            ("wrong_way_proxy_map_checked_rate", "Wrong-way checked"),
            ("wrong_way_proxy_rate", "Wrong-way raw"),
            ("intersection_lane_switch_proxy_rate", "Lane-switch proxy"),
            ("official_red_light_running_rate", "Official red label"),
            ("vru_tracks", "VRU tracks"),
            ("vru_encroachment_rate", "VRU encroachment"),
            ("crosswalk_polygons", "Crosswalks"),
            ("red_light_strict_available", "Strict TL available"),
            ("map_direction_warning_lanes", "Map-dir warnings"),
        ],
    )
    city_table = _html_table(
        city_rows,
        [
            ("city", "City"),
            ("vehicle_tracks", "Vehicle tracks"),
            ("wrong_way_proxy_map_checked_rate", "Wrong-way checked"),
            ("wrong_way_proxy_rate", "Wrong-way raw"),
            ("intersection_lane_switch_proxy_rate", "Lane-switch proxy"),
            ("official_red_light_running_rate", "Official red label"),
            ("vru_tracks", "VRU tracks"),
            ("vru_encroachment_rate", "VRU encroachment"),
        ],
    )
    top_vehicle = vehicle_event_df.head(40).to_dict("records") if not vehicle_event_df.empty else []
    top_vru = vru_event_df.head(40).to_dict("records") if not vru_event_df.empty else []
    vehicle_table = _html_table(top_vehicle, [("city", "City"), ("event_type", "Type"), ("scene_id", "Scene"), ("agent_id", "Agent"), ("note", "Note")])
    vru_table = _html_table(top_vru, [("city", "City"), ("event_reason", "Reason"), ("scene_id", "Scene"), ("agent_id", "Agent"), ("motor_lane_length_ratio", "Motor-lane ratio")])
    notes = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>" for k, v in methodology_notes.items())
    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Violations & Non-compliance Baseline</title>
  <style>
    :root {{ --paper:#f4efe6; --ink:#1d2528; --card:#fffaf1; --line:#d7c7aa; --red:#a93226; --amber:#b46f00; --blue:#1f618d; --muted:#687477; }}
    body {{ margin:0; color:var(--ink); font-family: Georgia, 'Times New Roman', serif; background:linear-gradient(135deg,#f4efe6,#e6eee9); }}
    header {{ padding:42px 5vw 76px; color:#fff; background:radial-gradient(circle at 78% 20%,rgba(244,208,63,.55),transparent 23%),linear-gradient(135deg,#263238,#8a3d2f); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); letter-spacing:-.8px; }}
    header p {{ max-width:1050px; line-height:1.65; color:#f7eadf; font-size:17px; }}
    main {{ padding:0 5vw 56px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:-42px; }}
    .kpi,.city-card,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:20px; box-shadow:0 14px 34px rgba(29,37,40,.08); padding:18px; }}
    .kpi b,.city-card b {{ display:block; color:var(--red); font-size:26px; margin-top:8px; }} .kpi small,.city-card small {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:18px; margin-top:24px; }}
    .city-card h3 {{ margin:0 0 8px; }} .city-card span {{ color:var(--muted); font-size:14px; }}
    .thumbs {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }} img {{ width:100%; border-radius:12px; border:1px solid var(--line); background:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px 9px; vertical-align:top; }} th {{ background:#efe2cc; }}
    .panel {{ margin-top:24px; overflow:auto; }} .panel h2 {{ margin-top:0; }}
    .warning {{ border-left:6px solid var(--amber); }} ul {{ line-height:1.7; }}
  </style>
</head>
<body>
  <header>
    <h1>SinD 违规与非标行为基线</h1>
    <p>该报告输出机动车结构化违规与 VRU 侵入的可计算基线。由于当前红绿灯-车道绑定、停止线和实线/双黄线语义尚未完全人工确认，严格违法项会被标记为 unavailable，方向/变道/VRU 占道以 proxy 或 map_rule 形式展示。</p>
  </header>
  <main>
    <section class='kpis'>
      <div class='kpi'><small>Vehicle tracks</small><b>{total_vehicle:,}</b></div>
      <div class='kpi'><small>VRU tracks</small><b>{total_vru:,}</b></div>
      <div class='kpi'><small>Event rows</small><b>{total_events:,}</b></div>
      <div class='kpi'><small>Strict red-light status</small><b>{'available' if any(summary_df.get('red_light_strict_available', pd.Series(dtype=bool))) else 'unavailable'}</b></div>
    </section>
    <section class='panel warning'><h2>方法边界</h2><ul>{notes}</ul></section>
    <section class='panel'><h2>总览柱状图</h2><a href='summary_violation_rates.png'><img src='summary_violation_rates.png' alt='summary bars'></a></section>
    <section class='grid'>{''.join(cards)}</section>
    <section class='panel'><h2>路口级汇总</h2>{summary_table}</section>
    <section class='panel'><h2>城市级汇总</h2>{city_table}</section>
    <section class='panel'><h2>机动车事件样例</h2>{vehicle_table}</section>
    <section class='panel'><h2>VRU 事件样例</h2>{vru_table}</section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def write_roi_geojson(output_path: Path, roi_polygons: Mapping[str, Polygon], crosswalks: Mapping[str, CrosswalkSet]) -> None:
    features: List[Dict[str, Any]] = []
    for location, polygon in roi_polygons.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "kind": "intersection_core_roi"},
                "geometry": mapping(polygon),
            }
        )
        for idx, cw in enumerate(crosswalks[location].polygons, start=1):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "kind": "crosswalk", "index": idx},
                    "geometry": mapping(cw),
                }
            )
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _limit_tracks_per_city(tracks: Sequence[TrackRecord], max_tracks_per_city: Optional[int]) -> List[TrackRecord]:
    if max_tracks_per_city is None:
        return list(tracks)
    counts: Dict[str, int] = {}
    kept: List[TrackRecord] = []
    for track in tracks:
        count = counts.get(track.location, 0)
        if count >= max_tracks_per_city:
            continue
        kept.append(track)
        counts[track.location] = count + 1
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/home/lyw/1TBSSD/Datasets/SinD_dataset_Simple"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS), help="SinD locations/cities to analyze.")
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_structured_violations_noncompliance"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--traffic-light-dir", type=Path, default=DEFAULT_TL_ROOT if DEFAULT_TL_ROOT.exists() else None)
    parser.add_argument("--traffic-light-mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--max-tracks-per-city", type=int, default=None, help="Optional smoke-test cap after static filtering.")
    parser.add_argument("--lane-sample-spacing-m", type=float, default=1.0)
    parser.add_argument("--lane-match-threshold-m", type=float, default=2.0)
    parser.add_argument("--wrong-way-heading-threshold-deg", type=float, default=120.0)
    parser.add_argument("--min-wrong-way-duration-s", type=float, default=1.0)
    parser.add_argument("--min-lane-change-run-s", type=float, default=0.5)
    parser.add_argument("--min-lane-change-speed-mps", type=float, default=1.0)
    parser.add_argument("--vru-motor-lane-threshold-m", type=float, default=1.5)
    parser.add_argument("--vru-event-min-length-ratio", type=float, default=0.25)
    parser.add_argument("--vru-event-min-seconds", type=float, default=2.0)
    parser.add_argument("--roi-core-buffer-m", type=float, default=3.0)
    parser.add_argument("--roi-central-quantile", type=float, default=0.62)
    parser.add_argument("--roi-max-core-radius-m", type=float, default=32.0)
    parser.add_argument("--roi-min-core-radius-m", type=float, default=12.0)
    parser.add_argument("--roi-turn-heading-threshold-deg", type=float, default=25.0)
    parser.add_argument("--roi-lane-trim-ratio", type=float, default=0.20)
    parser.add_argument("--disable-static-filter", action="store_true")
    parser.add_argument("--static-min-duration-s", type=float, default=8.0)
    parser.add_argument("--static-max-displacement-m", type=float, default=1.0)
    parser.add_argument("--static-max-path-length-m", type=float, default=2.5)
    parser.add_argument("--static-max-speed-p95-mps", type=float, default=0.25)
    parser.add_argument("--static-min-slow-duration-s", type=float, default=20.0)
    parser.add_argument("--static-max-mean-path-speed-mps", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locations = [normalize_location(item) for item in args.cities]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SinD tracks for {locations}...", flush=True)
    all_tracks = load_sind_tracks(args.data_dir, locations)
    all_tracks, static_removed_df = filter_static_tracks(
        all_tracks,
        dt=args.dt,
        enabled=not args.disable_static_filter,
        static_min_duration_s=args.static_min_duration_s,
        static_max_displacement_m=args.static_max_displacement_m,
        static_max_path_length_m=args.static_max_path_length_m,
        static_max_speed_p95_mps=args.static_max_speed_p95_mps,
        static_min_slow_duration_s=args.static_min_slow_duration_s,
        static_max_mean_path_speed_mps=args.static_max_mean_path_speed_mps,
    )
    all_tracks = _limit_tracks_per_city(all_tracks, args.max_tracks_per_city)
    vehicle_tracks = filter_tracks(all_tracks, city=locations, agent_type="vehicle")
    vru_tracks = [track for track in all_tracks if track.class_name in VRU_CLASSES]
    print(f"Tracks after filtering: vehicles={len(vehicle_tracks)}, VRU={len(vru_tracks)}, static_removed={len(static_removed_df)}", flush=True)

    print("Loading Lanelet2 references and ROIs...", flush=True)
    references = load_lane_references(args.data_dir, locations)
    lane_indices = build_lane_indices(references, spacing=args.lane_sample_spacing_m)
    roi_polygons: Dict[str, Polygon] = {}
    roi_debug_rows: List[Dict[str, Any]] = []
    for location in locations:
        roi, debug = infer_intersection_core_roi(
            references[location],
            core_buffer_m=args.roi_core_buffer_m,
            central_quantile=args.roi_central_quantile,
            max_core_radius_m=args.roi_max_core_radius_m,
            min_core_radius_m=args.roi_min_core_radius_m,
            turn_heading_threshold_deg=args.roi_turn_heading_threshold_deg,
            lane_trim_ratio=args.roi_lane_trim_ratio,
        )
        roi_polygons[location] = roi
        roi_debug_rows.append({"location": location, "city": LOCATION_DISPLAY.get(location, location), **debug})
    crosswalks = load_crosswalks(args.data_dir, locations)
    metadata = load_track_metadata(args.data_dir, locations)

    has_bindings, mapping_counts = _mapping_has_lane_bindings(args.traffic_light_mapping, locations)
    tl_root = configured_traffic_light_root(args.traffic_light_dir)
    red_entry_roi_enabled = bool(has_bindings and tl_root is not None)
    tracks_by_scene: Dict[Tuple[str, str], List[TrackRecord]] = {}
    if red_entry_roi_enabled:
        for track in vehicle_tracks:
            tracks_by_scene.setdefault((track.location, track.scene_id), []).append(track)
    signal_tables, signal_report_df = build_scene_signal_tables(
        tracks_by_scene,
        dt=args.dt,
        traffic_light_root=tl_root,
        mapping_path=args.traffic_light_mapping,
        enabled=red_entry_roi_enabled,
    )

    print("Analyzing vehicle structured-violation proxies...", flush=True)
    vehicle_event_df, vehicle_track_df, lane_diag_df = analyze_vehicle_tracks(
        vehicle_tracks,
        references=references,
        lane_indices=lane_indices,
        roi_polygons=roi_polygons,
        metadata=metadata,
        signal_tables=signal_tables,
        dt=args.dt,
        lane_match_threshold_m=args.lane_match_threshold_m,
        wrong_way_heading_threshold_deg=args.wrong_way_heading_threshold_deg,
        min_wrong_way_duration_s=args.min_wrong_way_duration_s,
        min_lane_change_run_s=args.min_lane_change_run_s,
        min_lane_change_speed_mps=args.min_lane_change_speed_mps,
        red_entry_roi_enabled=red_entry_roi_enabled,
    )
    print("Analyzing VRU encroachment/non-compliance proxies...", flush=True)
    vru_event_df, vru_track_df = analyze_vru_tracks(
        vru_tracks,
        lane_indices=lane_indices,
        roi_polygons=roi_polygons,
        crosswalks=crosswalks,
        dt=args.dt,
        motor_lane_threshold_m=args.vru_motor_lane_threshold_m,
        vru_event_min_length_ratio=args.vru_event_min_length_ratio,
        vru_event_min_seconds=args.vru_event_min_seconds,
    )

    summary_df = summarize_results(locations, vehicle_track_df, vehicle_event_df, vru_track_df, vru_event_df, lane_diag_df, mapping_counts, crosswalks)
    city_df = aggregate_by_city(summary_df)

    print("Writing CSV/JSON outputs...", flush=True)
    vehicle_event_df.to_csv(args.output_dir / "vehicle_violation_events.csv", index=False)
    vehicle_track_df.to_csv(args.output_dir / "vehicle_violation_track_metrics.csv", index=False)
    lane_diag_df.to_csv(args.output_dir / "lane_direction_diagnostics.csv", index=False)
    vru_event_df.to_csv(args.output_dir / "vru_noncompliance_events.csv", index=False)
    vru_track_df.to_csv(args.output_dir / "vru_noncompliance_track_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_by_location.csv", index=False)
    city_df.to_csv(args.output_dir / "summary_by_city.csv", index=False)
    static_removed_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)
    pd.DataFrame(roi_debug_rows).to_csv(args.output_dir / "roi_debug.csv", index=False)
    if not signal_report_df.empty:
        signal_report_df.to_csv(args.output_dir / "traffic_light_build_reports.csv", index=False)
    write_roi_geojson(args.output_dir / "violation_rois.geojson", roi_polygons, crosswalks)

    methodology_notes = {
        "red_light_strict_metric": "Unavailable until light-to-lane and stop-line bindings are manually verified. Official pkl red/yellow labels are reported when present, mainly for Tianjin.",
        "wrong_way_double_yellow": "Reported as a proxy based on sustained motion opposite to nearest Lanelet2 centerline. Summary figures include both raw and map-checked rates; map-checked excludes lanes flagged as likely Lanelet2 direction errors.",
        "intersection_lane_change": "Reported as a proxy based on nearest-lane-id switching inside the inferred intersection core; current maps do not encode solid-line legality.",
        "vru_motor_lane": "Measured by VRU trajectory length/time within the threshold of motor-lane centerlines.",
        "vru_red_phase": "Not reported as strict red-phase non-compliance because pedestrian-light and crosswalk signal bindings are not yet established.",
        "static_filter": "Long-stationary tracks are filtered before analysis to reduce parked/roadside vehicle artifacts.",
        "traffic_light_binding_entries": mapping_counts,
        "traffic_light_root": str(tl_root) if tl_root is not None else "not configured",
    }
    (args.output_dir / "methodology_notes.json").write_text(json.dumps(methodology_notes, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Plotting reports...", flush=True)
    for location in locations:
        loc_vehicle_events = vehicle_event_df[vehicle_event_df["location"] == location] if not vehicle_event_df.empty else pd.DataFrame()
        loc_vru_events = vru_event_df[vru_event_df["location"] == location] if not vru_event_df.empty else pd.DataFrame()
        plot_vehicle_hotspot(location, references[location], roi_polygons[location], loc_vehicle_events, args.output_dir / f"vehicle_violation_hotspot_{location}.png")
        plot_vru_encroachment(location, references[location], roi_polygons[location], crosswalks[location], loc_vru_events, args.output_dir / f"vru_encroachment_{location}.png")
    plot_summary_bars(summary_df, args.output_dir / "summary_violation_rates.png")
    write_html_report(args.output_dir / "index.html", summary_df, city_df, vehicle_event_df, vru_event_df, methodology_notes)
    print(f"Done. Open {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
