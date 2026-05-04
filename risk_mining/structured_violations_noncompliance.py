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
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon, mapping

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
WRONG_WAY_CHECK_CLASSES: Tuple[str, ...] = ("car", "truck", "bus", "tricycle", "bicycle", "motorcycle")
VRU_CLASSES: Tuple[str, ...] = ("pedestrian", "bicycle")
DEFAULT_TL_ROOT = Path("/home/lyw/1TBSSD/Datasets/SinD-dataset-wangpan/可用-csv")
VRU_EVENT_COLUMNS: Tuple[str, ...] = (
    "location",
    "city",
    "scene_id",
    "agent_id",
    "class_name",
    "event_type",
    "severity",
    "x",
    "y",
    "time_s",
    "speed_mps",
    "nearest_lane_id",
    "lane_distance_m",
    "motor_lane_seconds",
    "motor_lane_path_length_m",
    "motor_lane_length_ratio",
    "seconds_in_intersection_core",
    "outside_crosswalk_core_seconds",
    "outside_crosswalk_core_seconds_verified",
    "crosswalk_available",
    "vru_red_conflict_zone_seconds",
    "vru_yellow_conflict_zone_seconds",
    "vru_signal_observable_seconds",
    "event_reason",
    "note",
)
VRU_TRACK_COLUMNS: Tuple[str, ...] = (
    "location",
    "city",
    "scene_id",
    "agent_id",
    "class_name",
    "points",
    "duration_s",
    "path_length_m",
    "motor_lane_seconds",
    "motor_lane_path_length_m",
    "motor_lane_length_ratio",
    "seconds_in_intersection_core",
    "outside_crosswalk_core_seconds",
    "outside_crosswalk_core_seconds_verified",
    "crosswalk_available",
    "vru_signal_observable_seconds",
    "vru_red_conflict_zone_seconds",
    "vru_yellow_conflict_zone_seconds",
    "vru_red_conflict_zone_event",
    "vru_yellow_conflict_zone_event",
    "encroachment_event",
)
ROW_EVENT_COLUMNS: Tuple[str, ...] = (
    "location",
    "city",
    "scene_id",
    "agent_id",
    "priority_agent_id",
    "class_name",
    "priority_class_name",
    "event_type",
    "severity",
    "x",
    "y",
    "time_s",
    "priority_time_s",
    "time_gap_s",
    "distance_gap_m",
    "approach_angle_deg",
    "violator_maneuver",
    "priority_maneuver",
    "violator_priority",
    "priority_agent_priority",
    "priority_a_lon_mps2",
    "note",
)
ROW_PRIORITY: Mapping[str, int] = {
    "pedestrian": 5,
    "straight": 4,
    "left-turn": 3,
    "right-turn": 2,
    "u-turn": 1,
    "unknown": 0,
}


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
    names: List[str]

    @property
    def available(self) -> bool:
        return bool(self.polygons)


@dataclass(frozen=True)
class LaneRuleIndex:
    allowed_by_lane: Mapping[str, Tuple[str, ...]]
    movement_by_lane: Mapping[str, str]


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
    names: List[str] = []
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
            names.append(str(tags.get("name") or f"crosswalk_{len(names) + 1}"))

    paths = [MplPath(np.asarray(poly.exterior.coords, dtype=float)) for poly in polygons]
    return CrosswalkSet(polygons=polygons, paths=paths, names=names)


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


def _mapping_binding_counts(mapping_path: Path, locations: Sequence[str]) -> Tuple[bool, bool, Dict[str, int], Dict[str, int]]:
    if not mapping_path.exists():
        return False, False, {loc: 0 for loc in locations}, {loc: 0 for loc in locations}
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    vehicle_counts: Dict[str, int] = {}
    pedestrian_counts: Dict[str, int] = {}
    for location in locations:
        loc_map = mapping_payload.get("locations", {}).get(location, {})
        veh_count = 0
        ped_count = 0
        maps_to_check = [loc_map.get("light_to_lanes", {})]
        maps_to_check.extend(scene_map.get("light_to_lanes", {}) for scene_map in loc_map.get("scenes", {}).values())
        for light_to_lanes in maps_to_check:
            for light_key, lane_ids in light_to_lanes.items():
                if not isinstance(lane_ids, list):
                    continue
                if str(light_key).startswith("pedestrian:"):
                    ped_count += len(lane_ids)
                else:
                    veh_count += len(lane_ids)
        vehicle_counts[location] = int(veh_count)
        pedestrian_counts[location] = int(ped_count)
    return any(vehicle_counts.values()), any(pedestrian_counts.values()), vehicle_counts, pedestrian_counts


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
    pkl_root: Path,
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
            pkl_root=pkl_root,
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


def _status_name(status: Optional[int]) -> str:
    if status is None:
        return "unavailable"
    try:
        return TrafficLightStatus(int(status)).name
    except Exception:
        return f"UNKNOWN_CODE_{status}"


def _status_for_candidates(tls_df: Optional[pd.DataFrame], lane_ids: Sequence[str], scene_ts: int) -> Tuple[Optional[int], str]:
    if tls_df is None:
        return None, ""
    for lane_id in lane_ids:
        status = _status_at_scene_ts(tls_df, str(lane_id), scene_ts)
        if status is not None:
            return status, str(lane_id)
    return None, ""


def _infer_lane_movement(lane_id: str) -> str:
    text = str(lane_id)
    if "_to_" not in text:
        return "unknown"
    src, dst = text.split("_to_", 1)
    src_arm = _lane_arm_token(src)
    dst_arm = _lane_arm_token(dst)
    if src_arm is None or dst_arm is None or src_arm == dst_arm:
        return "unknown"
    initial_vec = -DIRECTION_VECTOR_LOCAL[src_arm]
    final_vec = DIRECTION_VECTOR_LOCAL[dst_arm]
    dot = float(np.dot(initial_vec, final_vec))
    cross = float(initial_vec[0] * final_vec[1] - initial_vec[1] * final_vec[0])
    if dot < -0.65:
        return "u-turn"
    if abs(cross) < 0.35:
        return "straight"
    return "left-turn" if cross > 0 else "right-turn"


DIRECTION_VECTOR_LOCAL: Mapping[str, np.ndarray] = {
    "e": np.array([1.0, 0.0]),
    "w": np.array([-1.0, 0.0]),
    "n": np.array([0.0, 1.0]),
    "s": np.array([0.0, -1.0]),
    "r1": np.array([1.0, 0.0]),
    "r2": np.array([0.0, 1.0]),
    "r3": np.array([-1.0, 0.0]),
    "r4": np.array([0.0, -1.0]),
}


def _lane_arm_token(text: str) -> Optional[str]:
    match = re.search(r"\b([NSEW])(?:_|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"\b(R[1-4])(?:In|Out|_|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def _normalize_maneuver_name(value: Any) -> str:
    key = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "straight": "straight",
        "go-straight": "straight",
        "left": "left-turn",
        "left-turn": "left-turn",
        "leftturn": "left-turn",
        "right": "right-turn",
        "right-turn": "right-turn",
        "rightturn": "right-turn",
        "u-turn": "u-turn",
        "uturn": "u-turn",
    }
    return aliases.get(key, "unknown")


def build_lane_rule_indices(references: Mapping[str, Sequence[LaneReference]]) -> Dict[str, LaneRuleIndex]:
    result: Dict[str, LaneRuleIndex] = {}
    for location, lanes in references.items():
        movement_by_lane: Dict[str, str] = {}
        allowed_sets: Dict[str, set[str]] = {}
        for lane in lanes:
            lane_id = str(lane.lane_id)
            movement = _infer_lane_movement(lane_id)
            movement_by_lane[lane_id] = movement
            if movement != "unknown" and "_to_" in lane_id:
                entry_lane = lane_id.split("_to_", 1)[0]
                allowed_sets.setdefault(entry_lane, set()).add(movement)
                allowed_sets.setdefault(lane_id, set()).add(movement)
        allowed_by_lane = {lane_id: tuple(sorted(values)) for lane_id, values in allowed_sets.items()}
        result[location] = LaneRuleIndex(allowed_by_lane=allowed_by_lane, movement_by_lane=movement_by_lane)
    return result


def _allowed_movements_for_lane(lane_id: str, rule_index: Optional[LaneRuleIndex]) -> Tuple[str, ...]:
    if rule_index is None:
        return ()
    lane_id = str(lane_id)
    if lane_id in rule_index.allowed_by_lane:
        return tuple(rule_index.allowed_by_lane[lane_id])
    base = lane_id.split(":gap", 1)[0]
    if base in rule_index.allowed_by_lane:
        return tuple(rule_index.allowed_by_lane[base])
    if "_to_" in base:
        movement = rule_index.movement_by_lane.get(base) or _infer_lane_movement(base)
        return (movement,) if movement != "unknown" else ()
    return ()


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _last_true_index(mask: np.ndarray) -> Optional[int]:
    indices = np.flatnonzero(mask)
    return int(indices[-1]) if len(indices) else None


def _dominant_lane_in_window(lane_ids: np.ndarray, valid_mask: np.ndarray, start: int, end: int) -> str:
    start = max(0, int(start))
    end = min(len(lane_ids), int(end))
    if end <= start:
        return ""
    mask = valid_mask[start:end]
    if not np.any(mask):
        return ""
    return _dominant_value(lane_ids[start:end][mask])


def _stable_entry_lane(
    lane_ids: np.ndarray,
    valid_lane: np.ndarray,
    moving: np.ndarray,
    entry_idx: Optional[int],
    min_run_pts: int,
    lookback_pts: int,
) -> Tuple[str, Optional[int]]:
    if entry_idx is None or entry_idx <= 0:
        return "", None
    start_idx = max(0, entry_idx - lookback_pts)
    approach_mask = np.zeros(len(lane_ids), dtype=bool)
    approach_mask[start_idx:entry_idx] = valid_lane[start_idx:entry_idx] & moving[start_idx:entry_idx]
    runs: List[Tuple[int, int, str]] = []
    for run_start, run_end in _run_lengths(approach_mask):
        if run_end - run_start < min_run_pts:
            continue
        run_lane = _dominant_value(lane_ids[run_start:run_end])
        if run_lane:
            runs.append((run_start, run_end, run_lane))
    if runs:
        run = runs[-1]
        return run[2], run[1] - 1
    fallback = _dominant_lane_in_window(lane_ids, valid_lane, start_idx, entry_idx)
    return fallback, entry_idx - 1 if fallback else None


def _movement_from_vectors(entry_vec: np.ndarray, exit_vec: np.ndarray, straight_threshold_deg: float = 30.0, uturn_threshold_deg: float = 135.0) -> str:
    if np.linalg.norm(entry_vec) < 1e-6 or np.linalg.norm(exit_vec) < 1e-6:
        return "unknown"
    entry_ang = math.atan2(float(entry_vec[1]), float(entry_vec[0]))
    exit_ang = math.atan2(float(exit_vec[1]), float(exit_vec[0]))
    delta = math.atan2(math.sin(exit_ang - entry_ang), math.cos(exit_ang - entry_ang))
    deg = math.degrees(delta)
    if abs(deg) <= straight_threshold_deg:
        return "straight"
    if abs(deg) >= uturn_threshold_deg:
        return "u-turn"
    return "left-turn" if deg > 0 else "right-turn"


def _actual_maneuver_from_roi_geometry(xy: np.ndarray, roi_mask: np.ndarray, entry_idx: Optional[int], exit_idx: Optional[int], outside_pts: int) -> str:
    if entry_idx is None or exit_idx is None or len(xy) < 5:
        return "unknown"
    in0 = max(0, entry_idx - outside_pts)
    in1 = min(len(xy) - 1, entry_idx + max(2, outside_pts // 3))
    out0 = max(0, exit_idx - max(2, outside_pts // 3))
    out1 = min(len(xy) - 1, exit_idx + outside_pts)
    if in1 <= in0 or out1 <= out0:
        return "unknown"
    entry_vec = xy[in1] - xy[in0]
    exit_vec = xy[out1] - xy[out0]
    return _movement_from_vectors(entry_vec, exit_vec)


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros_like(vec, dtype=float)
    return np.asarray(vec, dtype=float) / norm


def _angle_between_vectors_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    ua = _unit_vector(vec_a)
    ub = _unit_vector(vec_b)
    if np.linalg.norm(ua) < 1e-6 or np.linalg.norm(ub) < 1e-6:
        return 0.0
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _nearest_crosswalk_names(points: np.ndarray, crosswalks: CrosswalkSet) -> List[str]:
    if len(points) == 0 or not crosswalks.available:
        return []
    selected: List[str] = []
    for point in points:
        shp = Point(float(point[0]), float(point[1]))
        distances = [poly.distance(shp) for poly in crosswalks.polygons]
        if not distances:
            continue
        idx = int(np.argmin(distances))
        if distances[idx] <= 2.0:
            selected.append(crosswalks.names[idx])
    return list(dict.fromkeys(selected))


def analyze_vehicle_tracks(
    tracks: Sequence[TrackRecord],
    references: Mapping[str, Sequence[LaneReference]],
    lane_indices: Mapping[str, LaneIndex],
    lane_rule_indices: Mapping[str, LaneRuleIndex],
    roi_polygons: Mapping[str, Polygon],
    metadata: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    dt: float,
    lane_match_threshold_m: float,
    wrong_way_heading_threshold_deg: float,
    min_wrong_way_duration_s: float,
    min_lane_change_run_s: float,
    min_lane_change_speed_mps: float,
    signal_entry_enabled: bool,
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
        roi_mask = _points_inside_polygon(xy, roi_polygons[track.location])
        outside_roi_mask = ~roi_mask
        wrong_way_class_observable = track.class_name in WRONG_WAY_CHECK_CLASSES
        wrong_mask = valid_lane & outside_roi_mask & (speed >= min_lane_change_speed_mps) & (cos_lane < cos_threshold) & wrong_way_class_observable
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
                    "event_type": "wrong_way_violation_candidate",
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
                    "note": "Candidate based on sustained opposite-direction motion on approach/departure lanes outside the intersection core ROI.",
                }
            )

        entry_idx = _first_true_index(roi_mask & valid_lane)
        exit_idx = _last_true_index(roi_mask & valid_lane)
        entry_lane_id, entry_lane_sample_idx = _stable_entry_lane(
            lane_ids,
            valid_lane,
            moving,
            entry_idx,
            min_run_pts=min_run_pts,
            lookback_pts=max(min_run_pts * 2, int(round(8.0 / dt))),
        )
        pre_entry_lane_switches = 0
        if entry_idx is not None and entry_idx > 2:
            lookback = max(min_run_pts * 2, int(round(5.0 / dt)))
            start_idx = max(0, entry_idx - lookback)
            approach_mask = np.zeros(len(xy), dtype=bool)
            approach_mask[start_idx:entry_idx] = valid_lane[start_idx:entry_idx] & moving[start_idx:entry_idx]
            approach_runs: List[Tuple[int, int, str]] = []
            for run_start, run_end in _run_lengths(approach_mask):
                if run_end - run_start < min_run_pts:
                    continue
                run_lane = _dominant_value(lane_ids[run_start:run_end])
                if run_lane:
                    approach_runs.append((run_start, run_end, run_lane))
            compressed_approach: List[Tuple[int, int, str]] = []
            for run in approach_runs:
                if compressed_approach and compressed_approach[-1][2] == run[2]:
                    compressed_approach[-1] = (compressed_approach[-1][0], run[1], run[2])
                else:
                    compressed_approach.append(run)
            pre_entry_lane_switches = sum(1 for a, b in zip(compressed_approach[:-1], compressed_approach[1:]) if a[2] != b[2])
            if pre_entry_lane_switches > 0:
                switch_idx = compressed_approach[1][0]
                event_rows.append(
                    {
                        "location": track.location,
                        "city": LOCATION_DISPLAY.get(track.location, track.location),
                        "scene_id": track.scene_id,
                        "agent_id": track.agent_id,
                        "class_name": track.class_name,
                        "event_type": "solid_line_lane_change_proxy",
                        "severity": "proxy",
                        "x": float(xy[switch_idx, 0]),
                        "y": float(xy[switch_idx, 1]),
                        "time_s": float(time_s[switch_idx]) if len(time_s) else float(switch_idx * dt),
                        "duration_s": float(np.count_nonzero(approach_mask) * dt),
                        "dominant_lane_id": dominant_lane,
                        "nearest_lane_id": str(lane_ids[switch_idx]),
                        "lane_distance_m": float(distances[switch_idx]),
                        "speed_mps": float(speed[switch_idx]),
                        "cos_to_lane_direction": float(cos_lane[switch_idx]),
                        "official_red_light_running": official_red,
                        "official_yellow_light_running": official_yellow,
                        "official_retrograde_type": retrograde_type,
                        "note": "Proxy: the approach area before the inferred intersection core is treated as a solid-line zone.",
                    }
                )

        lane_direction_rule_violation = False
        allowed_movements: Tuple[str, ...] = ()
        geometry_maneuver = _actual_maneuver_from_roi_geometry(xy, roi_mask, entry_idx, exit_idx, outside_pts=max(5, int(round(2.5 / dt))))
        label_maneuver = _normalize_maneuver_name(track.maneuver)
        actual_maneuver = geometry_maneuver if geometry_maneuver != "unknown" else label_maneuver
        if entry_idx is not None and entry_lane_id:
            rule_index = lane_rule_indices.get(track.location)
            allowed_movements = _allowed_movements_for_lane(entry_lane_id, rule_index)
            if allowed_movements and actual_maneuver != "unknown" and actual_maneuver not in allowed_movements:
                lane_direction_rule_violation = True
                event_idx = int(entry_lane_sample_idx) if entry_lane_sample_idx is not None else int(entry_idx)
                event_rows.append(
                    {
                        "location": track.location,
                        "city": LOCATION_DISPLAY.get(track.location, track.location),
                        "scene_id": track.scene_id,
                        "agent_id": track.agent_id,
                        "class_name": track.class_name,
                        "event_type": "lane_direction_rule_violation",
                        "severity": "map_rule",
                        "x": float(xy[event_idx, 0]),
                        "y": float(xy[event_idx, 1]),
                        "time_s": float(time_s[event_idx]) if len(time_s) else float(event_idx * dt),
                        "duration_s": 0.0,
                        "dominant_lane_id": dominant_lane,
                        "nearest_lane_id": entry_lane_id,
                        "lane_distance_m": float(distances[event_idx]),
                        "speed_mps": float(speed[event_idx]),
                        "cos_to_lane_direction": float(cos_lane[event_idx]),
                        "actual_maneuver": actual_maneuver,
                        "geometry_maneuver": geometry_maneuver,
                        "label_maneuver": label_maneuver,
                        "allowed_movements": ";".join(allowed_movements),
                        "official_red_light_running": official_red,
                        "official_yellow_light_running": official_yellow,
                        "official_retrograde_type": retrograde_type,
                        "note": "Map-rule candidate: actual whole-intersection maneuver is inconsistent with movements allowed by the stable approach lane before ROI entry.",
                    }
                )

        red_roi_candidate = False
        yellow_roi_event = False
        red_status_name = "unavailable"
        signal_observable_entry = False
        if signal_entry_enabled and signal_tables and entry_idx is not None:
            tls_df = signal_tables.get((track.location, track.scene_id))
            lane_id = str(lane_ids[entry_idx])
            scene_ts = int(state.iloc[entry_idx]["frame_id"]) if "frame_id" in state.columns else entry_idx
            status = _status_at_scene_ts(tls_df, lane_id, scene_ts) if tls_df is not None else None
            if status is not None:
                signal_observable_entry = True
                red_status_name = _status_name(status)
                if status in {int(TrafficLightStatus.RED), int(TrafficLightStatus.YELLOW)}:
                    red_roi_candidate = status == int(TrafficLightStatus.RED)
                    yellow_roi_event = status == int(TrafficLightStatus.YELLOW)
                    event_rows.append(
                        {
                            "location": track.location,
                            "city": LOCATION_DISPLAY.get(track.location, track.location),
                            "scene_id": track.scene_id,
                            "agent_id": track.agent_id,
                            "class_name": track.class_name,
                            "event_type": "red_light_entry_violation" if red_roi_candidate else "yellow_light_entry_event",
                            "severity": "map_rule" if red_roi_candidate else "yellow_event",
                            "x": float(xy[entry_idx, 0]),
                            "y": float(xy[entry_idx, 1]),
                            "time_s": float(time_s[entry_idx]) if len(time_s) else float(entry_idx * dt),
                            "duration_s": 0.0,
                            "dominant_lane_id": dominant_lane,
                            "nearest_lane_id": lane_id,
                            "lane_distance_m": float(distances[entry_idx]),
                            "speed_mps": float(speed[entry_idx]),
                            "cos_to_lane_direction": float(cos_lane[entry_idx]),
                            "traffic_light_status": red_status_name,
                            "signal_observable_entry": signal_observable_entry,
                            "official_red_light_running": official_red,
                            "official_yellow_light_running": official_yellow,
                            "official_retrograde_type": retrograde_type,
                            "note": "Entry into inferred intersection core under a bound signal state; red/yellow are reported separately.",
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
                "intersection_lane_switch_count": 0,
                "intersection_lane_switch_proxy_event": False,
                "solid_line_lane_change_proxy_event": bool(pre_entry_lane_switches > 0),
                "solid_line_lane_change_count": int(pre_entry_lane_switches),
                "lane_direction_rule_violation": bool(lane_direction_rule_violation),
                "allowed_movements": ";".join(allowed_movements),
                "actual_maneuver": actual_maneuver,
                "geometry_maneuver": geometry_maneuver,
                "label_maneuver": label_maneuver,
                "entry_lane_id": entry_lane_id,
                "official_red_light_running": official_red,
                "official_yellow_light_running": official_yellow,
                "official_retrograde_type": retrograde_type,
                "official_retrograde_event": official_retrograde,
                "signal_observable_entry": bool(signal_observable_entry),
                "red_light_entry_violation": bool(red_roi_candidate),
                "yellow_light_entry_event": bool(yellow_roi_event),
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
                        (event_df["event_type"] == "wrong_way_violation_candidate")
                        & event_df["possible_map_direction_error"],
                        "location",
                    ],
                    event_df.loc[
                        (event_df["event_type"] == "wrong_way_violation_candidate")
                        & event_df["possible_map_direction_error"],
                        "scene_id",
                    ],
                    event_df.loc[
                        (event_df["event_type"] == "wrong_way_violation_candidate")
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
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    dt: float,
    motor_lane_threshold_m: float,
    vru_event_min_length_ratio: float,
    vru_event_min_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks, start=1):
        state, xy, time_s, _, speed = _track_xy_time_vel(track, dt)
        if len(xy) < 2 or track.location not in lane_indices:
            continue
        distances, _, lane_ids, _ = query_lane_index(lane_indices[track.location], xy)
        near_motor_lane = distances <= motor_lane_threshold_m
        roi_mask = _points_inside_polygon(xy, roi_polygons[track.location])
        cw = crosswalks[track.location]
        in_crosswalk = _points_inside_crosswalk(xy, cw)
        outside_crosswalk_core = roi_mask & (~in_crosswalk if cw.available else np.ones(len(xy), dtype=bool))
        non_crosswalk_motor_lane = near_motor_lane & (~in_crosswalk if cw.available else np.ones(len(xy), dtype=bool))

        total_length = _path_length(xy)
        motor_lane_length = _subpath_length(xy, non_crosswalk_motor_lane)
        motor_lane_seconds = float(np.count_nonzero(non_crosswalk_motor_lane) * dt)
        motor_lane_length_ratio = float(motor_lane_length / total_length) if total_length > 1e-6 else 0.0
        outside_core_seconds = float(np.count_nonzero(outside_crosswalk_core) * dt)
        conflict_zone_seconds_verified = outside_core_seconds if cw.available else float("nan")
        red_conflict_mask = np.zeros(len(xy), dtype=bool)
        yellow_conflict_mask = np.zeros(len(xy), dtype=bool)
        signal_observable_seconds = 0.0
        tls_df = signal_tables.get((track.location, track.scene_id))
        if tls_df is not None and cw.available and np.any(outside_crosswalk_core):
            candidate_crosswalks = _nearest_crosswalk_names(xy[outside_crosswalk_core], cw)
            if candidate_crosswalks:
                for point_idx in np.flatnonzero(outside_crosswalk_core):
                    scene_ts = int(state.iloc[point_idx]["frame_id"]) if "frame_id" in state.columns else int(point_idx)
                    status, _ = _status_for_candidates(tls_df, candidate_crosswalks, scene_ts)
                    if status is None:
                        continue
                    signal_observable_seconds += dt
                    if status == int(TrafficLightStatus.RED):
                        red_conflict_mask[point_idx] = True
                    elif status == int(TrafficLightStatus.YELLOW):
                        yellow_conflict_mask[point_idx] = True
        red_conflict_seconds = float(np.count_nonzero(red_conflict_mask) * dt)
        yellow_conflict_seconds = float(np.count_nonzero(yellow_conflict_mask) * dt)
        event_reason: List[str] = []
        if motor_lane_length_ratio >= vru_event_min_length_ratio and motor_lane_length >= 1.0:
            event_reason.append("non_crosswalk_motor_lane_path_ratio")
        if motor_lane_seconds >= vru_event_min_seconds:
            event_reason.append("non_crosswalk_motor_lane_seconds")
        if cw.available and outside_core_seconds >= vru_event_min_seconds:
            event_reason.append("outside_crosswalk_conflict_zone_seconds")
        elif (not cw.available) and outside_core_seconds >= vru_event_min_seconds:
            event_reason.append("core_conflict_zone_seconds_crosswalk_unavailable")

        if event_reason:
            candidate_mask = non_crosswalk_motor_lane | outside_crosswalk_core
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
                    "vru_red_conflict_zone_seconds": red_conflict_seconds,
                    "vru_yellow_conflict_zone_seconds": yellow_conflict_seconds,
                    "vru_signal_observable_seconds": signal_observable_seconds,
                    "note": "Pedestrian/bicycle non-compliance outside crosswalk polygons; normal crosswalk traversal and motorcycles are excluded.",
                }
            )

        for signal_event_type, signal_mask, signal_seconds, severity in (
            ("vru_red_conflict_zone_noncompliance", red_conflict_mask, red_conflict_seconds, "map_rule"),
            ("vru_yellow_conflict_zone_event", yellow_conflict_mask, yellow_conflict_seconds, "yellow_event"),
        ):
            if signal_seconds < vru_event_min_seconds:
                continue
            signal_indices = np.flatnonzero(signal_mask)
            event_idx = int(signal_indices[len(signal_indices) // 2])
            event_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    "event_type": signal_event_type,
                    "severity": severity,
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
                    "vru_red_conflict_zone_seconds": red_conflict_seconds,
                    "vru_yellow_conflict_zone_seconds": yellow_conflict_seconds,
                    "vru_signal_observable_seconds": signal_observable_seconds,
                    "event_reason": signal_event_type,
                    "note": "Pedestrian/bicycle is inside the intersection conflict zone and outside crosswalk polygons while the bound pedestrian/crosswalk signal is red or yellow.",
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
                "vru_signal_observable_seconds": signal_observable_seconds,
                "vru_red_conflict_zone_seconds": red_conflict_seconds,
                "vru_yellow_conflict_zone_seconds": yellow_conflict_seconds,
                "vru_red_conflict_zone_event": bool(red_conflict_seconds >= vru_event_min_seconds),
                "vru_yellow_conflict_zone_event": bool(yellow_conflict_seconds >= vru_event_min_seconds),
                "encroachment_event": bool(event_reason),
            }
        )
        if idx % 5000 == 0:
            print(f"Analyzed {idx}/{len(tracks)} VRU tracks...", flush=True)
    return pd.DataFrame(event_rows, columns=VRU_EVENT_COLUMNS), pd.DataFrame(track_rows, columns=VRU_TRACK_COLUMNS)


def _track_entry_record(track: TrackRecord, roi_polygon: Polygon, dt: float) -> Optional[Dict[str, Any]]:
    state, xy, time_s, vel, speed = _track_xy_time_vel(track, dt)
    if len(xy) < 3:
        return None
    roi_mask = _points_inside_polygon(xy, roi_polygon)
    entry_idx = _first_true_index(roi_mask)
    exit_idx = _last_true_index(roi_mask)
    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        return None
    maneuver = "pedestrian" if track.class_name == "pedestrian" else _normalize_maneuver_name(track.maneuver)
    if maneuver == "unknown" and track.class_name in {"bicycle", "motorcycle"}:
        maneuver = _normalize_maneuver_name(track.maneuver)
    approach_pts = max(5, int(round(2.5 / dt)))
    entry_vec_start = max(0, entry_idx - approach_pts)
    entry_vec_end = min(len(xy) - 1, entry_idx + max(2, approach_pts // 3))
    exit_vec_start = max(0, exit_idx - max(2, approach_pts // 3))
    exit_vec_end = min(len(xy) - 1, exit_idx + approach_pts)
    entry_vec = xy[entry_vec_end] - xy[entry_vec_start]
    exit_vec = xy[exit_vec_end] - xy[exit_vec_start]
    roi_indices = np.flatnonzero(roi_mask)
    roi_indices = roi_indices[(roi_indices >= entry_idx) & (roi_indices <= exit_idx)]
    if len(roi_indices) < 2:
        return None
    if len(roi_indices) > 80:
        roi_indices = roi_indices[np.linspace(0, len(roi_indices) - 1, 80).round().astype(int)]
    return {
        "track": track,
        "state": state,
        "xy": xy,
        "time_s": time_s,
        "vel": vel,
        "speed": speed,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": float(time_s[entry_idx]) if len(time_s) else float(entry_idx * dt),
        "entry_xy": xy[entry_idx],
        "entry_vec": entry_vec,
        "exit_vec": exit_vec,
        "roi_indices": roi_indices,
        "roi_xy": xy[roi_indices],
        "roi_time_s": time_s[roi_indices] if len(time_s) else roi_indices.astype(float) * dt,
        "maneuver": maneuver,
        "priority": ROW_PRIORITY.get(maneuver, 0),
    }


def _path_conflict_between_records(low: Mapping[str, Any], high: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    low_xy = np.asarray(low["roi_xy"], dtype=float)
    high_xy = np.asarray(high["roi_xy"], dtype=float)
    if len(low_xy) == 0 or len(high_xy) == 0:
        return None
    diff = low_xy[:, None, :] - high_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    low_idx, high_idx = np.unravel_index(int(np.argmin(dist)), dist.shape)
    distance = float(dist[low_idx, high_idx])
    low_time = float(np.asarray(low["roi_time_s"], dtype=float)[low_idx])
    high_time = float(np.asarray(high["roi_time_s"], dtype=float)[high_idx])
    point = (low_xy[low_idx] + high_xy[high_idx]) / 2.0
    return {
        "distance_m": distance,
        "low_time_s": low_time,
        "high_time_s": high_time,
        "time_gap_s": high_time - low_time,
        "x": float(point[0]),
        "y": float(point[1]),
    }


def analyze_right_of_way_candidates(
    tracks: Sequence[TrackRecord],
    roi_polygons: Mapping[str, Polygon],
    dt: float,
    conflict_distance_m: float,
    path_conflict_distance_m: float,
    min_approach_angle_deg: float,
    max_entry_gap_s: float,
    priority_decel_threshold_mps2: float,
    max_pairs_per_scene: int,
) -> pd.DataFrame:
    event_rows: List[Dict[str, Any]] = []
    tracks_by_scene: Dict[Tuple[str, str], List[TrackRecord]] = {}
    for track in tracks:
        tracks_by_scene.setdefault((track.location, track.scene_id), []).append(track)

    for (location, scene_id), scene_tracks in tracks_by_scene.items():
        roi_polygon = roi_polygons.get(location)
        if roi_polygon is None:
            continue
        records = [record for track in scene_tracks if (record := _track_entry_record(track, roi_polygon, dt)) is not None]
        records = [record for record in records if record["priority"] > 0]
        if len(records) < 2:
            continue
        records.sort(key=lambda item: item["entry_time"])
        pair_count = 0
        for i, low in enumerate(records):
            for high in records:
                if low is high or low["priority"] >= high["priority"]:
                    continue
                approach_angle = _angle_between_vectors_deg(np.asarray(low["entry_vec"]), np.asarray(high["entry_vec"]))
                if approach_angle < min_approach_angle_deg:
                    continue
                conflict = _path_conflict_between_records(low, high)
                if conflict is None or conflict["distance_m"] > path_conflict_distance_m:
                    continue
                time_gap = float(conflict["time_gap_s"])
                if time_gap < -0.2 or time_gap > max_entry_gap_s:
                    continue
                high_speed = np.asarray(high["speed"], dtype=float)
                if len(high_speed) >= 3:
                    high_acc = np.gradient(high_speed, dt)
                    high_conflict_idx = int(np.asarray(high["roi_indices"], dtype=int)[int(np.argmin(np.linalg.norm(np.asarray(high["roi_xy"], dtype=float) - np.array([conflict["x"], conflict["y"]]), axis=1)))])
                    hi_idx = high_conflict_idx
                    lo = max(0, hi_idx - int(round(1.0 / dt)))
                    hi = min(len(high_acc), hi_idx + int(round(2.0 / dt)) + 1)
                    min_acc = float(np.nanmin(high_acc[lo:hi])) if hi > lo else 0.0
                else:
                    min_acc = 0.0
                if min_acc > priority_decel_threshold_mps2 and time_gap > 0.5:
                    continue
                low_track: TrackRecord = low["track"]
                high_track: TrackRecord = high["track"]
                event_rows.append(
                    {
                        "location": location,
                        "city": LOCATION_DISPLAY.get(location, location),
                        "scene_id": scene_id,
                        "agent_id": low_track.agent_id,
                        "priority_agent_id": high_track.agent_id,
                        "class_name": low_track.class_name,
                        "priority_class_name": high_track.class_name,
                        "event_type": "right_of_way_yield_violation_candidate",
                        "severity": "candidate",
                        "x": float(conflict["x"]),
                        "y": float(conflict["y"]),
                        "time_s": float(conflict["low_time_s"]),
                        "priority_time_s": float(conflict["high_time_s"]),
                        "time_gap_s": time_gap,
                        "distance_gap_m": float(conflict["distance_m"]),
                        "approach_angle_deg": approach_angle,
                        "violator_maneuver": low["maneuver"],
                        "priority_maneuver": high["maneuver"],
                        "violator_priority": int(low["priority"]),
                        "priority_agent_priority": int(high["priority"]),
                        "priority_a_lon_mps2": min_acc,
                        "note": "Candidate: lower-priority actor reaches a shared trajectory conflict point before or close to a higher-priority actor from a different approach direction.",
                    }
                )
                pair_count += 1
                if pair_count >= max_pairs_per_scene:
                    break
            if pair_count >= max_pairs_per_scene:
                break
    return pd.DataFrame(event_rows, columns=ROW_EVENT_COLUMNS)


def summarize_results(
    locations: Sequence[str],
    vehicle_track_df: pd.DataFrame,
    vehicle_event_df: pd.DataFrame,
    vru_track_df: pd.DataFrame,
    vru_event_df: pd.DataFrame,
    lane_diag_df: pd.DataFrame,
    mapping_counts: Mapping[str, int],
    pedestrian_mapping_counts: Mapping[str, int],
    crosswalks: Mapping[str, CrosswalkSet],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for location in locations:
        veh = vehicle_track_df[vehicle_track_df["location"] == location] if not vehicle_track_df.empty else pd.DataFrame()
        vru = vru_track_df[vru_track_df["location"] == location] if not vru_track_df.empty else pd.DataFrame()
        ve = vehicle_event_df[vehicle_event_df["location"] == location] if not vehicle_event_df.empty else pd.DataFrame()
        vu = vru_event_df[vru_event_df["location"] == location] if not vru_event_df.empty else pd.DataFrame()
        lane_diag = lane_diag_df[lane_diag_df["location"] == location] if not lane_diag_df.empty else pd.DataFrame()
        motor_veh = veh[veh["class_name"].isin(MOTOR_VEHICLE_CLASSES)] if not veh.empty else pd.DataFrame()
        vehicle_count = len(motor_veh) if not motor_veh.empty else len(veh)
        signal_observable = int(veh.get("signal_observable_entry", pd.Series(dtype=bool)).sum()) if not veh.empty else 0
        vru_count = len(vru)
        rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "vehicle_tracks": int(vehicle_count),
                "structured_tracks": int(len(veh)),
                "vru_tracks": int(vru_count),
                "wrong_way_proxy_tracks": int(veh["wrong_way_proxy_event"].sum()) if not veh.empty else 0,
                "wrong_way_proxy_rate": float(veh["wrong_way_proxy_event"].mean()) if len(veh) else 0.0,
                "wrong_way_proxy_map_checked_tracks": int(veh["wrong_way_proxy_event_map_checked"].sum()) if (not veh.empty and "wrong_way_proxy_event_map_checked" in veh) else 0,
                "wrong_way_proxy_map_checked_rate": float(veh["wrong_way_proxy_event_map_checked"].mean()) if (len(veh) and "wrong_way_proxy_event_map_checked" in veh) else 0.0,
                "intersection_lane_switch_proxy_tracks": 0,
                "intersection_lane_switch_proxy_rate": 0.0,
                "solid_line_lane_change_proxy_tracks": int(veh.get("solid_line_lane_change_proxy_event", pd.Series(dtype=bool)).sum()) if not veh.empty else 0,
                "solid_line_lane_change_proxy_rate": float(veh.get("solid_line_lane_change_proxy_event", pd.Series(dtype=float)).mean()) if len(veh) else 0.0,
                "lane_direction_rule_violation_tracks": int(veh.get("lane_direction_rule_violation", pd.Series(dtype=bool)).sum()) if not veh.empty else 0,
                "lane_direction_rule_violation_rate": float(veh.get("lane_direction_rule_violation", pd.Series(dtype=float)).mean()) if len(veh) else 0.0,
                "official_red_light_running_tracks": int(motor_veh["official_red_light_running"].sum()) if not motor_veh.empty else 0,
                "official_red_light_running_rate": float(motor_veh["official_red_light_running"].mean()) if vehicle_count else 0.0,
                "official_yellow_light_running_tracks": int(motor_veh["official_yellow_light_running"].sum()) if not motor_veh.empty else 0,
                "official_retrograde_tracks": int(veh["official_retrograde_event"].sum()) if not veh.empty else 0,
                "signal_observable_entry_tracks": signal_observable,
                "red_light_entry_violations": int(veh.get("red_light_entry_violation", pd.Series(dtype=bool)).sum()) if not veh.empty else 0,
                "red_light_entry_violation_rate_observable": float(veh.get("red_light_entry_violation", pd.Series(dtype=float)).sum() / signal_observable) if signal_observable else 0.0,
                "yellow_light_entry_events": int(veh.get("yellow_light_entry_event", pd.Series(dtype=bool)).sum()) if not veh.empty else 0,
                "yellow_light_entry_event_rate_observable": float(veh.get("yellow_light_entry_event", pd.Series(dtype=float)).sum() / signal_observable) if signal_observable else 0.0,
                "red_at_roi_entry_candidates": int(veh.get("red_at_roi_entry_candidate", pd.Series(dtype=bool)).sum()) if not veh.empty else 0,
                "red_light_binding_lane_entries": int(mapping_counts.get(location, 0)),
                "red_light_strict_available": bool(mapping_counts.get(location, 0) > 0),
                "pedestrian_light_binding_entries": int(pedestrian_mapping_counts.get(location, 0)),
                "pedestrian_light_strict_available": bool(pedestrian_mapping_counts.get(location, 0) > 0),
                "vru_encroachment_tracks": int(vru["encroachment_event"].sum()) if not vru.empty else 0,
                "vru_encroachment_rate": float(vru["encroachment_event"].mean()) if vru_count else 0.0,
                "mean_vru_motor_lane_length_ratio": float(vru["motor_lane_length_ratio"].mean()) if not vru.empty else 0.0,
                "vru_motor_lane_total_seconds": float(vru["motor_lane_seconds"].sum()) if not vru.empty else 0.0,
                "vru_conflict_zone_total_seconds_verified": float(vru["outside_crosswalk_core_seconds_verified"].sum(skipna=True)) if not vru.empty else 0.0,
                "vru_signal_observable_seconds": float(vru.get("vru_signal_observable_seconds", pd.Series(dtype=float)).sum()) if not vru.empty else 0.0,
                "vru_red_conflict_zone_seconds": float(vru.get("vru_red_conflict_zone_seconds", pd.Series(dtype=float)).sum()) if not vru.empty else 0.0,
                "vru_yellow_conflict_zone_seconds": float(vru.get("vru_yellow_conflict_zone_seconds", pd.Series(dtype=float)).sum()) if not vru.empty else 0.0,
                "vru_red_conflict_zone_events": int(vru.get("vru_red_conflict_zone_event", pd.Series(dtype=bool)).sum()) if not vru.empty else 0,
                "vru_yellow_conflict_zone_events": int(vru.get("vru_yellow_conflict_zone_event", pd.Series(dtype=bool)).sum()) if not vru.empty else 0,
                "crosswalk_polygons": int(len(crosswalks[location].polygons)),
                "map_direction_warning_lanes": int(lane_diag["possible_map_direction_error"].sum()) if not lane_diag.empty else 0,
                "right_of_way_candidates": int((ve["event_type"] == "right_of_way_yield_violation_candidate").sum()) if not ve.empty else 0,
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
                "intersection_lane_switch_proxy_tracks": 0,
                "intersection_lane_switch_proxy_rate": 0.0,
                "solid_line_lane_change_proxy_tracks": int(group.get("solid_line_lane_change_proxy_tracks", pd.Series(dtype=float)).sum()),
                "solid_line_lane_change_proxy_rate": float(group.get("solid_line_lane_change_proxy_tracks", pd.Series(dtype=float)).sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "lane_direction_rule_violation_tracks": int(group.get("lane_direction_rule_violation_tracks", pd.Series(dtype=float)).sum()),
                "lane_direction_rule_violation_rate": float(group.get("lane_direction_rule_violation_tracks", pd.Series(dtype=float)).sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "official_red_light_running_tracks": int(group["official_red_light_running_tracks"].sum()),
                "official_red_light_running_rate": float(group["official_red_light_running_tracks"].sum() / vehicle_tracks) if vehicle_tracks else 0.0,
                "signal_observable_entry_tracks": int(group.get("signal_observable_entry_tracks", pd.Series(dtype=float)).sum()),
                "red_light_entry_violations": int(group.get("red_light_entry_violations", pd.Series(dtype=float)).sum()),
                "yellow_light_entry_events": int(group.get("yellow_light_entry_events", pd.Series(dtype=float)).sum()),
                "right_of_way_candidates": int(group.get("right_of_way_candidates", pd.Series(dtype=float)).sum()),
                "vru_encroachment_tracks": int(group["vru_encroachment_tracks"].sum()),
                "vru_encroachment_rate": float(group["vru_encroachment_tracks"].sum() / vru_tracks) if vru_tracks else 0.0,
                "vru_motor_lane_total_seconds": float(group["vru_motor_lane_total_seconds"].sum()),
                "vru_conflict_zone_total_seconds_verified": float(group["vru_conflict_zone_total_seconds_verified"].sum()),
                "vru_signal_observable_seconds": float(group.get("vru_signal_observable_seconds", pd.Series(dtype=float)).sum()),
                "vru_red_conflict_zone_seconds": float(group.get("vru_red_conflict_zone_seconds", pd.Series(dtype=float)).sum()),
                "vru_yellow_conflict_zone_seconds": float(group.get("vru_yellow_conflict_zone_seconds", pd.Series(dtype=float)).sum()),
                "red_light_strict_available_locations": int(group["red_light_strict_available"].sum()),
                "pedestrian_light_strict_available_locations": int(group.get("pedestrian_light_strict_available", pd.Series(dtype=bool)).sum()),
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
        "wrong_way_violation_candidate": "#c0392b",
        "wrong_way_or_opposing_lane_proxy": "#c0392b",
        "solid_line_lane_change_proxy": "#b9770e",
        "lane_direction_rule_violation": "#1f618d",
        "right_of_way_yield_violation_candidate": "#7d3c98",
        "official_red_light_running_label": "#8e1b14",
        "red_at_roi_entry_candidate": "#6c3483",
        "red_light_entry_violation": "#922b21",
        "yellow_light_entry_event": "#f1c40f",
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
    colors = {
        "vru_motor_lane_or_conflict_zone_encroachment": "#d35400",
        "vru_red_conflict_zone_noncompliance": "#922b21",
        "vru_yellow_conflict_zone_event": "#f1c40f",
    }
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
    lane_col = "solid_line_lane_change_proxy_rate" if "solid_line_lane_change_proxy_rate" in summary_df else "intersection_lane_switch_proxy_rate"
    ax.bar(x, summary_df[lane_col], width, label="Solid-line lane-change proxy", color="#d68910")
    ax.bar(x + width, summary_df["vru_encroachment_rate"], width, label="VRU encroachment", color="#2874a6")
    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel("Track-level rate")
    ax.set_ylim(0, min(1.0, max(0.05, float(summary_df[[wrong_rate_col, lane_col, "vru_encroachment_rate"]].max().max()) * 1.25)))
    ax.set_title("SinD violation / non-compliance baseline proxies")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _track_lookup(tracks: Sequence[TrackRecord]) -> Dict[Tuple[str, str, str], TrackRecord]:
    return {(track.location, track.scene_id, track.agent_id): track for track in tracks}


def _agent_box_dimensions(class_name: str) -> Tuple[float, float, bool]:
    cls = str(class_name).lower()
    if cls == "pedestrian":
        return 0.8, 0.8, True
    if cls == "bicycle":
        return 1.8, 0.7, False
    if cls == "motorcycle":
        return 2.2, 0.85, False
    if cls == "bus":
        return 10.5, 2.6, False
    if cls == "truck":
        return 7.5, 2.5, False
    if cls == "tricycle":
        return 2.6, 1.2, False
    return 4.6, 1.9, False


def _agent_type_group(class_name: Any) -> str:
    cls = str(class_name).lower()
    if cls == "pedestrian":
        return "pedestrian"
    if cls in {"bicycle", "motorcycle"}:
        return "two_wheeler"
    return "vehicle"


def _draw_agent_marker(ax: plt.Axes, x: float, y: float, heading: float, class_name: str, color: str, label: str) -> None:
    length, width, as_circle = _agent_box_dimensions(class_name)
    if as_circle:
        patch = MplCircle((x, y), radius=0.45, facecolor=color, edgecolor="#111111", linewidth=0.9, alpha=0.92, zorder=6, label=label)
        ax.add_patch(patch)
    else:
        local = np.array(
            [
                [length / 2.0, width / 2.0],
                [length / 2.0, -width / 2.0],
                [-length / 2.0, -width / 2.0],
                [-length / 2.0, width / 2.0],
            ],
            dtype=float,
        )
        rot = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        corners = local @ rot.T + np.array([x, y])
        patch = MplPolygon(corners, closed=True, facecolor=color, edgecolor="#111111", linewidth=0.9, alpha=0.72, zorder=6, label=label)
        ax.add_patch(patch)
    ax.arrow(x, y, math.cos(heading) * 2.2, math.sin(heading) * 2.2, color="#111111", width=0.05, head_width=0.45, alpha=0.75, zorder=7)


def _heading_at(xy: np.ndarray, idx: int) -> float:
    if len(xy) < 2:
        return 0.0
    lo = max(0, idx - 2)
    hi = min(len(xy) - 1, idx + 2)
    vec = xy[hi] - xy[lo]
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    return float(math.atan2(vec[1], vec[0]))


def _event_scene_ts(event: Mapping[str, Any], track: TrackRecord, state: pd.DataFrame, time_s: np.ndarray, idx: int) -> int:
    if "frame_id" in state.columns and len(state) > idx:
        return int(state.iloc[idx]["frame_id"])
    if len(time_s):
        return int(round(float(event.get("time_s", time_s[idx])) / 0.1))
    return int(idx)


def _light_fill_for_status(status_name: str) -> Tuple[str, str]:
    key = str(status_name or "").upper()
    if key == "RED":
        return "#e74c3c", "red light controlled lane"
    if key == "YELLOW":
        return "#f4d03f", "yellow light controlled lane"
    if key == "GREEN":
        return "#58d68d", "green light controlled lane"
    return "#bdc3c7", "unknown light controlled lane"


def _highlight_signal_lane(
    ax: plt.Axes,
    event: Mapping[str, Any],
    lane_refs: Sequence[LaneReference],
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    track: TrackRecord,
    state: pd.DataFrame,
    time_s: np.ndarray,
    idx: int,
) -> None:
    lane_id = str(event.get("nearest_lane_id", "") or "")
    if not lane_id:
        return
    scene_ts = _event_scene_ts(event, track, state, time_s, idx)
    tls_df = signal_tables.get((track.location, track.scene_id))
    status = _status_at_scene_ts(tls_df, lane_id, scene_ts) if tls_df is not None else None
    status_name = _status_name(status)
    if status_name == "unavailable" and pd.notna(event.get("traffic_light_status", np.nan)):
        status_name = str(event.get("traffic_light_status"))
    color, label = _light_fill_for_status(status_name)
    matched = False
    for lane in lane_refs:
        if str(lane.lane_id) != lane_id:
            continue
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color=color, linewidth=10.0, alpha=0.24, solid_capstyle="round", label=f"{label}: {lane_id}", zorder=2)
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color=color, linewidth=2.0, alpha=0.72, solid_capstyle="round", zorder=3)
        matched = True
    if not matched:
        return


def plot_violation_example(
    event: Mapping[str, Any],
    track_lookup: Mapping[Tuple[str, str, str], TrackRecord],
    lane_refs: Sequence[LaneReference],
    roi_polygon: Polygon,
    crosswalks: CrosswalkSet,
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    output_path: Path,
    dt: float,
    window_s: float,
) -> bool:
    key = (str(event.get("location")), str(event.get("scene_id")), str(event.get("agent_id")))
    track = track_lookup.get(key)
    if track is None:
        return False
    state, xy, time_s, _, _ = _track_xy_time_vel(track, dt)
    if len(xy) == 0:
        return False
    event_time = float(event.get("time_s", time_s[len(time_s) // 2] if len(time_s) else 0.0))
    idx = int(np.argmin(np.abs(time_s - event_time))) if len(time_s) else len(xy) // 2
    mask = (time_s >= event_time - window_s) & (time_s <= event_time + window_s) if len(time_s) else np.ones(len(xy), dtype=bool)

    fig, ax = plt.subplots(figsize=(8.5, 8), dpi=180)
    for lane in lane_refs:
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#6b7280", linewidth=0.7, alpha=0.34)
        if len(lane.xy) > 3:
            mid = len(lane.xy) // 2
            vec = lane.xy[min(mid + 1, len(lane.xy) - 1)] - lane.xy[max(mid - 1, 0)]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
                ax.arrow(lane.xy[mid, 0], lane.xy[mid, 1], vec[0] * 2.0, vec[1] * 2.0, color="#6b7280", alpha=0.35, head_width=0.55, linewidth=0.4)
    _highlight_signal_lane(ax, event, lane_refs, signal_tables, track, state, time_s, idx)
    for poly in crosswalks.polygons:
        px = np.asarray(poly.exterior.coords)
        ax.fill(px[:, 0], px[:, 1], color="#7fb3d5", alpha=0.24, zorder=1)
    roi_xy = np.asarray(roi_polygon.exterior.coords)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#111111", linewidth=1.6, label="intersection ROI")
    ax.plot(xy[mask, 0], xy[mask, 1], color="#d35400", linewidth=2.2, alpha=0.9, label="violator/target trajectory")
    _draw_agent_marker(ax, float(xy[idx, 0]), float(xy[idx, 1]), _heading_at(xy, idx), track.class_name, "#d35400", f"{track.agent_id} ({track.class_name})")

    priority_agent_id = event.get("priority_agent_id")
    if isinstance(priority_agent_id, str) and priority_agent_id:
        other = track_lookup.get((key[0], key[1], priority_agent_id))
        if other is not None:
            _, oxy, otime, _, _ = _track_xy_time_vel(other, dt)
            if len(oxy):
                omask = (otime >= event_time - window_s) & (otime <= event_time + window_s) if len(otime) else np.ones(len(oxy), dtype=bool)
                oidx = int(np.argmin(np.abs(otime - float(event.get("priority_time_s", event_time))))) if len(otime) else len(oxy) // 2
                ax.plot(oxy[omask, 0], oxy[omask, 1], color="#1f618d", linewidth=2.2, alpha=0.9, label="higher-priority trajectory")
                _draw_agent_marker(ax, float(oxy[oidx, 0]), float(oxy[oidx, 1]), _heading_at(oxy, oidx), other.class_name, "#1f618d", f"{priority_agent_id} ({other.class_name})")

    title = f"{event.get('event_type')} | {event.get('location')} {event.get('scene_id')} | agent {event.get('agent_id')}"
    subtitle_items = []
    for col in ("traffic_light_status", "nearest_lane_id", "actual_maneuver", "allowed_movements", "time_gap_s"):
        if col in event and pd.notna(event[col]):
            subtitle_items.append(f"{col}={event[col]}")
    ax.set_title(title, fontsize=10)
    if subtitle_items:
        ax.text(0.01, 0.01, "\n".join(subtitle_items[:5]), transform=ax.transAxes, fontsize=7, va="bottom", ha="left", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd"})
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.16)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=7)
    margin = 16.0
    focus = xy[mask] if np.any(mask) else xy
    if len(focus):
        ax.set_xlim(float(np.nanmin(focus[:, 0]) - margin), float(np.nanmax(focus[:, 0]) + margin))
        ax.set_ylim(float(np.nanmin(focus[:, 1]) - margin), float(np.nanmax(focus[:, 1]) + margin))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def write_violation_examples(
    output_dir: Path,
    vehicle_events: pd.DataFrame,
    vru_events: pd.DataFrame,
    tracks: Sequence[TrackRecord],
    references: Mapping[str, Sequence[LaneReference]],
    roi_polygons: Mapping[str, Polygon],
    crosswalks: Mapping[str, CrosswalkSet],
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    dt: float,
    examples_per_event_type_location: int,
    window_s: float,
) -> pd.DataFrame:
    all_events = pd.concat([vehicle_events, vru_events], ignore_index=True, sort=False)
    if all_events.empty or examples_per_event_type_location <= 0:
        return pd.DataFrame()
    examples_root = output_dir / "examples"
    if examples_root.exists():
        shutil.rmtree(examples_root)
    lookup = _track_lookup(tracks)
    rows: List[Dict[str, Any]] = []
    sort_cols = [col for col in ["location", "event_type", "severity", "scene_id", "time_s", "agent_id"] if col in all_events.columns]
    all_events = all_events.sort_values(sort_cols, kind="mergesort") if sort_cols else all_events
    for (location_key, event_type), group in all_events.groupby(["location", "event_type"], sort=True):
        for rank, (_, row) in enumerate(group.head(examples_per_event_type_location).iterrows(), start=1):
            location = str(row.get("location"))
            safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("scene_id")))
            safe_agent = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("agent_id")))
            rel_path = Path("examples") / str(event_type) / str(location_key) / f"{location}_{safe_scene}_{safe_agent}_{rank:02d}.png"
            ok = plot_violation_example(
                row.to_dict(),
                lookup,
                references.get(location, []),
                roi_polygons[location],
                crosswalks[location],
                signal_tables,
                output_dir / rel_path,
                dt=dt,
                window_s=window_s,
            )
            if not ok:
                continue
            record = row.to_dict()
            record.update({"example_rank": rank, "example_path": rel_path.as_posix()})
            rows.append(record)
    return pd.DataFrame(rows)


EVENT_COLORS: Mapping[str, str] = {
    "wrong_way_violation_candidate": "#c0392b",
    "solid_line_lane_change_proxy": "#b9770e",
    "lane_direction_rule_violation": "#1f618d",
    "right_of_way_yield_violation_candidate": "#7d3c98",
    "official_red_light_running_label": "#8e1b14",
    "red_light_entry_violation": "#922b21",
    "yellow_light_entry_event": "#f1c40f",
    "vru_motor_lane_or_conflict_zone_encroachment": "#d35400",
    "vru_red_conflict_zone_noncompliance": "#922b21",
    "vru_yellow_conflict_zone_event": "#f1c40f",
}


def _event_plot_payload(event_df: pd.DataFrame, max_events: int) -> List[Dict[str, Any]]:
    if event_df.empty:
        return []
    df = event_df.copy()
    if len(df) > max_events:
        df = df.sort_values(["event_type", "time_s", "scene_id", "agent_id"], kind="mergesort").groupby("event_type", group_keys=False).head(max(1, max_events // max(1, df["event_type"].nunique())))
        if len(df) > max_events:
            df = df.head(max_events)
    payload: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        class_name = str(row_dict.get("class_name", "unknown"))
        payload.append(
            {
                "x": float(row_dict.get("x", 0.0)),
                "y": float(row_dict.get("y", 0.0)),
                "event_type": str(row_dict.get("event_type", "")),
                "agent_type": _agent_type_group(class_name),
                "class_name": class_name,
                "scene_id": str(row_dict.get("scene_id", "")),
                "agent_id": str(row_dict.get("agent_id", "")),
                "time_s": float(row_dict.get("time_s", 0.0)) if pd.notna(row_dict.get("time_s", np.nan)) else 0.0,
                "color": EVENT_COLORS.get(str(row_dict.get("event_type", "")), "#222222"),
            }
        )
    return payload


def _polyline_payload(lane_refs: Sequence[LaneReference]) -> List[List[List[float]]]:
    return [
        [[float(x), float(y)] for x, y in lane.xy]
        for lane in lane_refs
        if len(lane.xy) >= 2
    ]


def _polygon_payload(polygons: Sequence[Polygon]) -> List[List[List[float]]]:
    return [
        [[float(x), float(y)] for x, y in np.asarray(poly.exterior.coords)]
        for poly in polygons
        if not poly.is_empty
    ]


def write_interactive_location_report(
    output_path: Path,
    location: str,
    lane_refs: Sequence[LaneReference],
    roi_polygon: Polygon,
    crosswalks: CrosswalkSet,
    event_df: pd.DataFrame,
    max_events: int,
) -> None:
    events = _event_plot_payload(event_df, max_events=max_events)
    lanes = _polyline_payload(lane_refs)
    crosswalk_payload = _polygon_payload(crosswalks.polygons)
    roi_payload = [[float(x), float(y)] for x, y in np.asarray(roi_polygon.exterior.coords)]
    xs = [pt[0] for line in lanes for pt in line] + [item["x"] for item in events] + [pt[0] for pt in roi_payload]
    ys = [pt[1] for line in lanes for pt in line] + [item["y"] for item in events] + [pt[1] for pt in roi_payload]
    bbox = {
        "minX": min(xs) if xs else -50.0,
        "maxX": max(xs) if xs else 50.0,
        "minY": min(ys) if ys else -50.0,
        "maxY": max(ys) if ys else 50.0,
    }
    event_types = sorted({item["event_type"] for item in events})
    agent_types = [
        ("vehicle", "车辆"),
        ("two_wheeler", "二轮车"),
        ("pedestrian", "行人"),
    ]
    data_json = json.dumps(
        {
            "location": location,
            "city": LOCATION_DISPLAY.get(location, location),
            "lanes": lanes,
            "crosswalks": crosswalk_payload,
            "roi": roi_payload,
            "events": events,
            "bbox": bbox,
            "eventTypes": event_types,
        },
        ensure_ascii=False,
    )
    event_controls = "".join(
        f"<label><input type='checkbox' class='event-filter' value='{html.escape(event_type)}' checked> {html.escape(event_type)}</label>"
        for event_type in event_types
    )
    agent_controls = "".join(
        f"<label><input type='checkbox' class='agent-filter' value='{key}' checked> {label}</label>"
        for key, label in agent_types
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(LOCATION_DISPLAY.get(location, location))} Violation Filter</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; color:#1d2528; background:#f5f1e9; }}
    header {{ padding:16px 20px; background:#263238; color:white; }}
    header h1 {{ margin:0; font-size:22px; }}
    main {{ display:grid; grid-template-columns:300px minmax(0,1fr); gap:0; min-height:calc(100vh - 62px); }}
    aside {{ padding:14px; border-right:1px solid #d7c7aa; background:#fffaf1; overflow:auto; }}
    canvas {{ width:100%; height:calc(100vh - 62px); display:block; background:#ffffff; }}
    fieldset {{ border:1px solid #d7c7aa; margin:0 0 12px; padding:10px; }}
    legend {{ font-weight:700; }}
    label {{ display:block; margin:7px 0; font-size:13px; line-height:1.3; }}
    .row {{ display:flex; gap:8px; margin:10px 0; }}
    button {{ border:1px solid #9b8b70; background:#f0e4cf; padding:6px 10px; cursor:pointer; }}
    #stats {{ font-size:13px; color:#536064; margin-top:10px; line-height:1.5; }}
    #tip {{ position:fixed; pointer-events:none; display:none; background:rgba(29,37,40,.92); color:white; padding:8px 10px; border-radius:6px; font-size:12px; max-width:320px; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(LOCATION_DISPLAY.get(location, location))} ({html.escape(location)}) 违规事件筛选</h1></header>
  <main>
    <aside>
      <fieldset><legend>违规类型</legend><div class="row"><button id="allEvents">全选</button><button id="noneEvents">清空</button></div>{event_controls}</fieldset>
      <fieldset><legend>违规者类型</legend>{agent_controls}</fieldset>
      <div id="stats"></div>
    </aside>
    <canvas id="plot"></canvas>
  </main>
  <div id="tip"></div>
  <script>
    const DATA = {data_json};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    const tip = document.getElementById('tip');
    const stats = document.getElementById('stats');
    let scale = 1, ox = 0, oy = 0, visibleEvents = [];
    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}
    function checkedValues(cls) {{
      return new Set(Array.from(document.querySelectorAll('.' + cls + ':checked')).map(el => el.value));
    }}
    function project(x, y) {{
      return [ox + x * scale, canvas.getBoundingClientRect().height - (oy + y * scale)];
    }}
    function setupTransform() {{
      const rect = canvas.getBoundingClientRect();
      const pad = 28;
      const w = Math.max(1, DATA.bbox.maxX - DATA.bbox.minX);
      const h = Math.max(1, DATA.bbox.maxY - DATA.bbox.minY);
      scale = Math.min((rect.width - pad * 2) / w, (rect.height - pad * 2) / h);
      ox = pad - DATA.bbox.minX * scale;
      oy = pad - DATA.bbox.minY * scale;
    }}
    function drawPolyline(line, color, width, alpha) {{
      if (!line.length) return;
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      line.forEach((pt, idx) => {{ const [x, y] = project(pt[0], pt[1]); if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }});
      ctx.stroke(); ctx.restore();
    }}
    function drawPolygon(poly, fill, stroke, alpha) {{
      if (!poly.length) return;
      ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath();
      poly.forEach((pt, idx) => {{ const [x, y] = project(pt[0], pt[1]); if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }});
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
    }}
    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      setupTransform();
      DATA.lanes.forEach(line => drawPolyline(line, '#687477', 1, 0.38));
      DATA.crosswalks.forEach(poly => drawPolygon(poly, '#7fb3d5', '#4f87a5', 0.28));
      drawPolygon(DATA.roi, 'rgba(0,0,0,0)', '#111111', 1);
      const eventSet = checkedValues('event-filter');
      const agentSet = checkedValues('agent-filter');
      visibleEvents = DATA.events.filter(e => eventSet.has(e.event_type) && agentSet.has(e.agent_type));
      visibleEvents.forEach(e => {{
        const [x, y] = project(e.x, e.y);
        ctx.beginPath(); ctx.arc(x, y, e.agent_type === 'pedestrian' ? 3.2 : 4.2, 0, Math.PI * 2);
        ctx.fillStyle = e.color; ctx.globalAlpha = 0.72; ctx.fill(); ctx.globalAlpha = 1;
      }});
      stats.textContent = `显示 ${{visibleEvents.length}} / ${{DATA.events.length}} 个事件`;
    }}
    function setEventChecks(value) {{
      document.querySelectorAll('.event-filter').forEach(el => el.checked = value);
      draw();
    }}
    document.querySelectorAll('input').forEach(el => el.addEventListener('change', draw));
    document.getElementById('allEvents').addEventListener('click', () => setEventChecks(true));
    document.getElementById('noneEvents').addEventListener('click', () => setEventChecks(false));
    canvas.addEventListener('mousemove', ev => {{
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      let best = null, bestD = 999;
      visibleEvents.forEach(e => {{ const [x, y] = project(e.x, e.y); const d = Math.hypot(x - mx, y - my); if (d < bestD) {{ bestD = d; best = e; }} }});
      if (best && bestD < 9) {{
        tip.style.display = 'block'; tip.style.left = (ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
        tip.innerHTML = `${{best.event_type}}<br>type=${{best.class_name}} scene=${{best.scene_id}}<br>agent=${{best.agent_id}} time=${{best.time_s.toFixed(2)}}s`;
      }} else {{ tip.style.display = 'none'; }}
    }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def write_interactive_reports(
    output_dir: Path,
    locations: Sequence[str],
    references: Mapping[str, Sequence[LaneReference]],
    roi_polygons: Mapping[str, Polygon],
    crosswalks: Mapping[str, CrosswalkSet],
    vehicle_event_df: pd.DataFrame,
    vru_event_df: pd.DataFrame,
    max_events_per_location: int,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    all_events = pd.concat([vehicle_event_df, vru_event_df], ignore_index=True, sort=False)
    for location in locations:
        loc_events = all_events[all_events["location"] == location] if not all_events.empty else pd.DataFrame()
        rel_path = f"interactive_violation_filter_{location}.html"
        write_interactive_location_report(
            output_dir / rel_path,
            location,
            references[location],
            roi_polygons[location],
            crosswalks[location],
            loc_events,
            max_events=max_events_per_location,
        )
        paths[location] = rel_path
    return paths


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
    example_manifest_df: pd.DataFrame,
    interactive_paths: Mapping[str, str],
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
              <b>{float(row.get('solid_line_lane_change_proxy_rate', 0.0)):.2%}</b><small>solid-line lane-change proxy</small>
              <b>{float(row['vru_encroachment_rate']):.2%}</b><small>VRU encroachment proxy</small>
              <div class='thumbs'>
                <a href='vehicle_violation_hotspot_{html.escape(str(loc))}.png'><img src='vehicle_violation_hotspot_{html.escape(str(loc))}.png' alt='vehicle hotspot'></a>
                <a href='vru_encroachment_{html.escape(str(loc))}.png'><img src='vru_encroachment_{html.escape(str(loc))}.png' alt='vru encroachment'></a>
              </div>
              <p><a class='filter-link' href='{html.escape(str(interactive_paths.get(str(loc), "")))}'>Interactive filter</a></p>
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
            ("solid_line_lane_change_proxy_rate", "Solid-line LC proxy"),
            ("lane_direction_rule_violation_rate", "Lane rule"),
            ("red_light_entry_violation_rate_observable", "Red entry / obs"),
            ("yellow_light_entry_event_rate_observable", "Yellow entry / obs"),
            ("right_of_way_candidates", "ROW candidates"),
            ("vru_tracks", "VRU tracks"),
            ("vru_encroachment_rate", "VRU encroachment"),
            ("vru_red_conflict_zone_seconds", "VRU red-zone s"),
            ("crosswalk_polygons", "Crosswalks"),
            ("red_light_strict_available", "Strict TL available"),
            ("pedestrian_light_strict_available", "Ped TL available"),
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
            ("solid_line_lane_change_proxy_rate", "Solid-line LC proxy"),
            ("lane_direction_rule_violation_rate", "Lane rule"),
            ("red_light_entry_violations", "Red entries"),
            ("yellow_light_entry_events", "Yellow entries"),
            ("right_of_way_candidates", "ROW candidates"),
            ("vru_tracks", "VRU tracks"),
            ("vru_encroachment_rate", "VRU encroachment"),
        ],
    )
    top_vehicle = vehicle_event_df.head(40).to_dict("records") if not vehicle_event_df.empty else []
    top_vru = vru_event_df.head(40).to_dict("records") if not vru_event_df.empty else []
    vehicle_table = _html_table(top_vehicle, [("city", "City"), ("event_type", "Type"), ("scene_id", "Scene"), ("agent_id", "Agent"), ("traffic_light_status", "TL"), ("note", "Note")])
    vru_table = _html_table(top_vru, [("city", "City"), ("event_type", "Type"), ("event_reason", "Reason"), ("scene_id", "Scene"), ("agent_id", "Agent"), ("motor_lane_length_ratio", "Motor-lane ratio")])
    example_cards = []
    if not example_manifest_df.empty:
        for row in example_manifest_df.to_dict("records"):
            rel = html.escape(str(row.get("example_path", "")))
            label = html.escape(f"{row.get('event_type')} | {row.get('location')} | {row.get('scene_id')} | {row.get('agent_id')}")
            example_cards.append(f"<figure><a href='{rel}'><img src='{rel}' alt='{label}'></a><figcaption>{label}</figcaption></figure>")
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
    .filter-link {{ display:inline-block; margin-top:10px; color:var(--blue); font-weight:700; }}
    .examples {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }} figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:12px; line-height:1.4; margin-top:6px; }}
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
    <section class='panel'><h2>违规实例可视化</h2><div class='examples'>{''.join(example_cards) if example_cards else '<p>No example images were generated.</p>'}</div></section>
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
    parser.add_argument("--right-of-way-conflict-distance-m", type=float, default=10.0)
    parser.add_argument("--right-of-way-path-conflict-distance-m", type=float, default=4.0)
    parser.add_argument("--right-of-way-min-approach-angle-deg", type=float, default=45.0)
    parser.add_argument("--right-of-way-max-entry-gap-s", type=float, default=3.0)
    parser.add_argument("--right-of-way-priority-decel-threshold-mps2", type=float, default=-1.5)
    parser.add_argument("--right-of-way-max-pairs-per-scene", type=int, default=80)
    parser.add_argument("--examples-per-event-type", type=int, default=3, help="Examples per location and event type.")
    parser.add_argument("--example-window-s", type=float, default=3.0)
    parser.add_argument("--interactive-max-events-per-location", type=int, default=12000)
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
    structured_tracks = [track for track in all_tracks if track.class_name in set(MOTOR_VEHICLE_CLASSES + ("bicycle", "motorcycle"))]
    vru_tracks = [track for track in all_tracks if track.class_name in VRU_CLASSES]
    print(f"Tracks after filtering: vehicles={len(vehicle_tracks)}, structured={len(structured_tracks)}, VRU={len(vru_tracks)}, static_removed={len(static_removed_df)}", flush=True)

    print("Loading Lanelet2 references and ROIs...", flush=True)
    references = load_lane_references(args.data_dir, locations)
    lane_indices = build_lane_indices(references, spacing=args.lane_sample_spacing_m)
    lane_rule_indices = build_lane_rule_indices(references)
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

    has_vehicle_bindings, has_pedestrian_bindings, mapping_counts, pedestrian_mapping_counts = _mapping_binding_counts(args.traffic_light_mapping, locations)
    tl_root = configured_traffic_light_root(args.traffic_light_dir)
    signal_entry_enabled = bool(has_vehicle_bindings or has_pedestrian_bindings)
    tracks_by_scene: Dict[Tuple[str, str], List[TrackRecord]] = {}
    if signal_entry_enabled:
        for track in structured_tracks + vru_tracks:
            tracks_by_scene.setdefault((track.location, track.scene_id), []).append(track)
    signal_tables, signal_report_df = build_scene_signal_tables(
        tracks_by_scene,
        dt=args.dt,
        traffic_light_root=tl_root,
        mapping_path=args.traffic_light_mapping,
        pkl_root=args.data_dir,
        enabled=signal_entry_enabled,
    )

    print("Analyzing vehicle structured-violation proxies...", flush=True)
    vehicle_event_df, vehicle_track_df, lane_diag_df = analyze_vehicle_tracks(
        structured_tracks,
        references=references,
        lane_indices=lane_indices,
        lane_rule_indices=lane_rule_indices,
        roi_polygons=roi_polygons,
        metadata=metadata,
        signal_tables=signal_tables,
        dt=args.dt,
        lane_match_threshold_m=args.lane_match_threshold_m,
        wrong_way_heading_threshold_deg=args.wrong_way_heading_threshold_deg,
        min_wrong_way_duration_s=args.min_wrong_way_duration_s,
        min_lane_change_run_s=args.min_lane_change_run_s,
        min_lane_change_speed_mps=args.min_lane_change_speed_mps,
        signal_entry_enabled=signal_entry_enabled,
    )
    print("Analyzing right-of-way yield candidates...", flush=True)
    row_event_df = analyze_right_of_way_candidates(
        [track for track in all_tracks if track.class_name in set(MOTOR_VEHICLE_CLASSES + ("bicycle", "motorcycle", "pedestrian"))],
        roi_polygons=roi_polygons,
        dt=args.dt,
        conflict_distance_m=args.right_of_way_conflict_distance_m,
        path_conflict_distance_m=args.right_of_way_path_conflict_distance_m,
        min_approach_angle_deg=args.right_of_way_min_approach_angle_deg,
        max_entry_gap_s=args.right_of_way_max_entry_gap_s,
        priority_decel_threshold_mps2=args.right_of_way_priority_decel_threshold_mps2,
        max_pairs_per_scene=args.right_of_way_max_pairs_per_scene,
    )
    if not row_event_df.empty:
        vehicle_event_df = pd.concat([vehicle_event_df, row_event_df], ignore_index=True, sort=False)
    print("Analyzing VRU encroachment/non-compliance proxies...", flush=True)
    vru_event_df, vru_track_df = analyze_vru_tracks(
        vru_tracks,
        lane_indices=lane_indices,
        roi_polygons=roi_polygons,
        crosswalks=crosswalks,
        signal_tables=signal_tables,
        dt=args.dt,
        motor_lane_threshold_m=args.vru_motor_lane_threshold_m,
        vru_event_min_length_ratio=args.vru_event_min_length_ratio,
        vru_event_min_seconds=args.vru_event_min_seconds,
    )

    summary_df = summarize_results(locations, vehicle_track_df, vehicle_event_df, vru_track_df, vru_event_df, lane_diag_df, mapping_counts, pedestrian_mapping_counts, crosswalks)
    city_df = aggregate_by_city(summary_df)
    example_manifest_df = write_violation_examples(
        args.output_dir,
        vehicle_event_df,
        vru_event_df,
        all_tracks,
        references,
        roi_polygons,
        crosswalks,
        signal_tables,
        dt=args.dt,
        examples_per_event_type_location=args.examples_per_event_type,
        window_s=args.example_window_s,
    )
    interactive_paths = write_interactive_reports(
        args.output_dir,
        locations,
        references,
        roi_polygons,
        crosswalks,
        vehicle_event_df,
        vru_event_df,
        max_events_per_location=args.interactive_max_events_per_location,
    )

    print("Writing CSV/JSON outputs...", flush=True)
    vehicle_event_df.to_csv(args.output_dir / "vehicle_violation_events.csv", index=False)
    vehicle_track_df.to_csv(args.output_dir / "vehicle_violation_track_metrics.csv", index=False)
    lane_diag_df.to_csv(args.output_dir / "lane_direction_diagnostics.csv", index=False)
    vru_event_df.to_csv(args.output_dir / "vru_noncompliance_events.csv", index=False)
    vru_track_df.to_csv(args.output_dir / "vru_noncompliance_track_metrics.csv", index=False)
    row_event_df.to_csv(args.output_dir / "right_of_way_yield_events.csv", index=False)
    if not vehicle_track_df.empty:
        wrong_way_by_type = (
            vehicle_track_df.groupby(["location", "city", "class_name"], as_index=False)
            .agg(
                tracks=("agent_id", "count"),
                wrong_way_candidates=("wrong_way_proxy_event", "sum"),
                wrong_way_map_checked_candidates=("wrong_way_proxy_event_map_checked", "sum"),
            )
        )
        wrong_way_by_type["wrong_way_candidate_rate"] = wrong_way_by_type["wrong_way_candidates"] / wrong_way_by_type["tracks"].clip(lower=1)
        wrong_way_by_type["wrong_way_map_checked_rate"] = wrong_way_by_type["wrong_way_map_checked_candidates"] / wrong_way_by_type["tracks"].clip(lower=1)
        wrong_way_by_type.to_csv(args.output_dir / "wrong_way_events_by_agent_type.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "wrong_way_events_by_agent_type.csv", index=False)
    signal_obs_cols = [
        "location",
        "city",
        "signal_observable_entry_tracks",
        "red_light_entry_violations",
        "yellow_light_entry_events",
        "red_light_binding_lane_entries",
        "pedestrian_light_binding_entries",
        "vru_signal_observable_seconds",
        "vru_red_conflict_zone_seconds",
        "vru_yellow_conflict_zone_seconds",
    ]
    summary_df[[col for col in signal_obs_cols if col in summary_df.columns]].to_csv(args.output_dir / "signal_observability_by_location.csv", index=False)
    example_manifest_df.to_csv(args.output_dir / "violation_example_manifest.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_by_location.csv", index=False)
    city_df.to_csv(args.output_dir / "summary_by_city.csv", index=False)
    static_removed_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)
    pd.DataFrame(roi_debug_rows).to_csv(args.output_dir / "roi_debug.csv", index=False)
    if not signal_report_df.empty:
        signal_report_df.to_csv(args.output_dir / "traffic_light_build_reports.csv", index=False)
    write_roi_geojson(args.output_dir / "violation_rois.geojson", roi_polygons, crosswalks)

    methodology_notes = {
        "red_yellow_entry_metric": "Uses bound SinD traffic-light tables loaded from local pkl first, with CSV fallback. Red and yellow entries are reported separately over signal-observable lane entries.",
        "wrong_way_double_yellow": "Reported as a candidate based on sustained motion opposite to nearest Lanelet2 centerline on approach/departure segments outside the intersection core ROI.",
        "solid_line_lane_change": "The approach segment immediately before the inferred intersection core is treated as a solid-line zone because explicit line-marking semantics are unavailable.",
        "lane_direction_rule": "Allowed movements are inferred from the stable approach lane before ROI entry; actual movement is inferred from whole-intersection entry/exit geometry.",
        "right_of_way": "Candidate only. Priority order is pedestrian > straight > left-turn > right-turn > u-turn; pairs must come from different approach directions and share a close trajectory conflict point inside the ROI.",
        "vru_motor_lane": "Measured only for pedestrians and bicycles outside crosswalk polygons; normal crosswalk traversal and motorcycles are excluded.",
        "vru_red_yellow_phase": "When pedestrian/crosswalk bindings exist, reports pedestrian/bicycle time inside the core ROI and outside crosswalk polygons during red/yellow signal phases.",
        "static_filter": "Long-stationary tracks are filtered before analysis to reduce parked/roadside vehicle artifacts.",
        "traffic_light_vehicle_binding_entries": mapping_counts,
        "traffic_light_pedestrian_binding_entries": pedestrian_mapping_counts,
        "traffic_light_root": str(tl_root) if tl_root is not None else "not configured",
        "traffic_light_pkl_root": str(args.data_dir),
    }
    (args.output_dir / "methodology_notes.json").write_text(json.dumps(methodology_notes, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Plotting reports...", flush=True)
    for location in locations:
        loc_vehicle_events = vehicle_event_df[vehicle_event_df["location"] == location] if not vehicle_event_df.empty else pd.DataFrame()
        loc_vru_events = vru_event_df[vru_event_df["location"] == location] if not vru_event_df.empty else pd.DataFrame()
        plot_vehicle_hotspot(location, references[location], roi_polygons[location], loc_vehicle_events, args.output_dir / f"vehicle_violation_hotspot_{location}.png")
        plot_vru_encroachment(location, references[location], roi_polygons[location], crosswalks[location], loc_vru_events, args.output_dir / f"vru_encroachment_{location}.png")
    plot_summary_bars(summary_df, args.output_dir / "summary_violation_rates.png")
    write_html_report(args.output_dir / "index.html", summary_df, city_df, vehicle_event_df, vru_event_df, example_manifest_df, interactive_paths, methodology_notes)
    print(f"Done. Open {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
