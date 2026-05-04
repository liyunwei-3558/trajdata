#!/usr/bin/env python
"""Measure legal spatio-temporal right-of-way overlaps in SinD intersections.

The metric asks a different question from violation mining: when both sides are
legally released by a green signal, how many motor-vehicle paths and
pedestrian/bicycle crossing paths still geometrically intersect?
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
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intersection_spatiotemporal_density import filter_static_tracks, infer_intersection_core_roi  # noqa: E402
from kinematic_envelopes import LOCATION_DISPLAY, TrackRecord, load_sind_tracks, normalize_location  # noqa: E402
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS, LaneReference, load_lane_references  # noqa: E402
from structured_violations_noncompliance import (  # noqa: E402
    DEFAULT_TL_ROOT,
    MOTOR_VEHICLE_CLASSES,
    VRU_CLASSES,
    _first_true_index,
    _last_true_index,
    _points_inside_polygon,
    _status_at_scene_ts,
    _status_name,
    _track_xy_time_vel,
    build_lane_indices,
    build_scene_signal_tables,
    load_crosswalks,
    query_lane_index,
)
from trajdata.dataset_specific.sind.sind_traffic_lights import DEFAULT_MAPPING_PATH, configured_traffic_light_root  # noqa: E402
from trajdata.maps import TrafficLightStatus  # noqa: E402


GREEN_STATUS = int(TrafficLightStatus.GREEN)
DEFAULT_OUTPUT_DIR = Path("risk_mining/output_right_of_way_overlap")


@dataclass(frozen=True)
class LegalPassage:
    location: str
    city: str
    scene_id: str
    agent_id: str
    class_name: str
    passage_type: str
    control_id: str
    signal_status: str
    green_start_ts: int
    green_end_ts: int
    entry_ts: int
    exit_ts: int
    entry_time_s: float
    exit_time_s: float
    line: LineString
    sample_xy: np.ndarray
    sample_time_s: np.ndarray
    path_length_m: float


def _scene_ts_at(state: pd.DataFrame, idx: int) -> int:
    if "frame_id" in state.columns:
        return int(pd.to_numeric(state.iloc[idx]["frame_id"], errors="coerce"))
    return int(idx)


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _dedupe_xy_time(xy: np.ndarray, time_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(xy) <= 1:
        return xy, time_s
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-5
    return xy[keep], time_s[keep]


def _downsample_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    if len(indices) <= max_points:
        return indices
    take = np.linspace(0, len(indices) - 1, max_points).round().astype(int)
    return indices[take]


def _candidate_lane_ids(lane_ids: np.ndarray, idx: int, window: int) -> List[str]:
    lo = max(0, idx - window)
    hi = min(len(lane_ids), idx + window + 1)
    values = [str(lane_ids[idx])]
    if hi > lo:
        counts = pd.Series([str(v) for v in lane_ids[lo:hi] if str(v)]).value_counts()
        values.extend(str(v) for v in counts.index)
    return list(dict.fromkeys(values))


def _green_interval(
    tls_df: Optional[pd.DataFrame],
    control_ids: Sequence[str],
    scene_ts: int,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    if tls_df is None:
        return None, None, None, None
    for control_id in control_ids:
        try:
            lane_df = tls_df.xs(str(control_id), level="lane_id").sort_index()
        except KeyError:
            continue
        if lane_df.empty:
            continue
        status = _status_at_scene_ts(tls_df, str(control_id), scene_ts)
        if status is None:
            continue
        if int(status) != GREEN_STATUS:
            return str(control_id), int(status), None, None
        index = lane_df.index.to_numpy(dtype=int)
        status_values = lane_df["status"].to_numpy(dtype=int)
        pos = int(np.searchsorted(index, scene_ts, side="right") - 1)
        if pos < 0:
            continue
        while pos + 1 < len(index) and index[pos + 1] <= scene_ts:
            pos += 1
        start_pos = pos
        while start_pos > 0 and status_values[start_pos - 1] == GREEN_STATUS:
            start_pos -= 1
        end_pos = pos
        while end_pos + 1 < len(status_values) and status_values[end_pos + 1] == GREEN_STATUS:
            end_pos += 1
        return str(control_id), int(status), int(index[start_pos]), int(index[end_pos])
    return None, None, None, None


def _crosswalk_membership(xy: np.ndarray, crosswalk_set: Any) -> Tuple[np.ndarray, List[str]]:
    if len(xy) == 0 or not crosswalk_set.available:
        return np.full(len(xy), "", dtype=object), []
    names = np.full(len(xy), "", dtype=object)
    for name, path in zip(crosswalk_set.names, crosswalk_set.paths):
        mask = path.contains_points(xy)
        names[(names == "") & mask] = str(name)
    present = [str(name) for name in pd.Series(names[names != ""]).value_counts().index]
    return names, present


def _make_passage_line(xy: np.ndarray, time_s: np.ndarray, max_points: int) -> Tuple[Optional[LineString], np.ndarray, np.ndarray]:
    xy, time_s = _dedupe_xy_time(np.asarray(xy, dtype=float), np.asarray(time_s, dtype=float))
    if len(xy) < 2:
        return None, xy, time_s
    indices = _downsample_indices(np.arange(len(xy), dtype=int), max_points)
    sample_xy = xy[indices]
    sample_time_s = time_s[indices]
    sample_xy, sample_time_s = _dedupe_xy_time(sample_xy, sample_time_s)
    if len(sample_xy) < 2:
        return None, sample_xy, sample_time_s
    try:
        line = LineString(sample_xy)
    except Exception:
        return None, sample_xy, sample_time_s
    if line.is_empty or line.length <= 0.5:
        return None, sample_xy, sample_time_s
    return line, sample_xy, sample_time_s


def _extract_vehicle_passage(
    track: TrackRecord,
    roi_polygon: Polygon,
    lane_index: Any,
    tls_df: Optional[pd.DataFrame],
    dt: float,
    lane_match_threshold_m: float,
    lane_window: int,
    max_sample_points: int,
) -> Tuple[Optional[LegalPassage], Dict[str, Any]]:
    state, xy, time_s, _, _ = _track_xy_time_vel(track, dt)
    base_diag = {
        "location": track.location,
        "city": LOCATION_DISPLAY.get(track.location, track.location),
        "scene_id": track.scene_id,
        "agent_id": track.agent_id,
        "class_name": track.class_name,
        "passage_type": "vehicle",
        "status": "unprocessed",
        "control_id": "",
        "traffic_light_status": "unavailable",
    }
    if len(xy) < 3:
        base_diag["status"] = "too_short"
        return None, base_diag
    roi_mask = _points_inside_polygon(xy, roi_polygon)
    entry_idx = _first_true_index(roi_mask)
    exit_idx = _last_true_index(roi_mask)
    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        base_diag["status"] = "no_roi_passage"
        return None, base_diag
    distances, _, lane_ids, _ = query_lane_index(lane_index, xy)
    if float(distances[entry_idx]) > lane_match_threshold_m:
        base_diag["status"] = "entry_lane_too_far"
        base_diag["lane_distance_m"] = float(distances[entry_idx])
        return None, base_diag
    scene_ts = _scene_ts_at(state, entry_idx)
    control_id, status, green_start, green_end = _green_interval(
        tls_df,
        _candidate_lane_ids(lane_ids, entry_idx, lane_window),
        scene_ts,
    )
    base_diag["control_id"] = control_id or ""
    base_diag["traffic_light_status"] = _status_name(status)
    base_diag["entry_ts"] = scene_ts
    if control_id is None or status is None:
        base_diag["status"] = "signal_unobservable"
        return None, base_diag
    if status != GREEN_STATUS or green_start is None or green_end is None:
        base_diag["status"] = "not_green_at_entry"
        return None, base_diag
    roi_indices = np.flatnonzero(roi_mask)
    roi_indices = roi_indices[(roi_indices >= entry_idx) & (roi_indices <= exit_idx)]
    line, sample_xy, sample_time_s = _make_passage_line(xy[roi_indices], time_s[roi_indices], max_sample_points)
    if line is None:
        base_diag["status"] = "invalid_roi_line"
        return None, base_diag
    base_diag["status"] = "legal_green"
    return (
        LegalPassage(
            location=track.location,
            city=LOCATION_DISPLAY.get(track.location, track.location),
            scene_id=track.scene_id,
            agent_id=track.agent_id,
            class_name=track.class_name,
            passage_type="vehicle",
            control_id=control_id,
            signal_status="GREEN",
            green_start_ts=green_start,
            green_end_ts=green_end,
            entry_ts=scene_ts,
            exit_ts=_scene_ts_at(state, exit_idx),
            entry_time_s=float(time_s[entry_idx]) if len(time_s) else float(entry_idx * dt),
            exit_time_s=float(time_s[exit_idx]) if len(time_s) else float(exit_idx * dt),
            line=line,
            sample_xy=sample_xy,
            sample_time_s=sample_time_s,
            path_length_m=float(line.length),
        ),
        base_diag,
    )


def _extract_vru_passage(
    track: TrackRecord,
    roi_polygon: Polygon,
    crosswalk_set: Any,
    tls_df: Optional[pd.DataFrame],
    dt: float,
    min_crosswalk_path_m: float,
    max_sample_points: int,
) -> Tuple[Optional[LegalPassage], Dict[str, Any]]:
    state, xy, time_s, _, _ = _track_xy_time_vel(track, dt)
    base_diag = {
        "location": track.location,
        "city": LOCATION_DISPLAY.get(track.location, track.location),
        "scene_id": track.scene_id,
        "agent_id": track.agent_id,
        "class_name": track.class_name,
        "passage_type": "vru_crossing",
        "status": "unprocessed",
        "control_id": "",
        "traffic_light_status": "unavailable",
    }
    if len(xy) < 3:
        base_diag["status"] = "too_short"
        return None, base_diag
    if not crosswalk_set.available:
        base_diag["status"] = "crosswalk_unavailable"
        return None, base_diag
    roi_mask = _points_inside_polygon(xy, roi_polygon)
    names, present = _crosswalk_membership(xy, crosswalk_set)
    if not present:
        base_diag["status"] = "no_crosswalk_passage"
        return None, base_diag
    best_name = str(pd.Series(names[names != ""]).value_counts().index[0])
    crosswalk_mask = names == best_name
    passage_mask = crosswalk_mask & roi_mask
    if np.count_nonzero(passage_mask) < 2:
        passage_mask = crosswalk_mask
    entry_idx = _first_true_index(passage_mask)
    exit_idx = _last_true_index(passage_mask)
    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        base_diag["status"] = "invalid_crosswalk_passage"
        base_diag["control_id"] = best_name
        return None, base_diag
    path_len = _path_length(xy[passage_mask])
    if path_len < min_crosswalk_path_m:
        base_diag["status"] = "crosswalk_path_too_short"
        base_diag["control_id"] = best_name
        base_diag["crosswalk_path_m"] = path_len
        return None, base_diag
    scene_ts = _scene_ts_at(state, entry_idx)
    control_id, status, green_start, green_end = _green_interval(tls_df, [best_name], scene_ts)
    base_diag["control_id"] = best_name
    base_diag["traffic_light_status"] = _status_name(status)
    base_diag["entry_ts"] = scene_ts
    if control_id is None or status is None:
        base_diag["status"] = "signal_unobservable"
        return None, base_diag
    if status != GREEN_STATUS or green_start is None or green_end is None:
        base_diag["status"] = "not_green_at_entry"
        return None, base_diag
    indices = np.flatnonzero(passage_mask)
    indices = indices[(indices >= entry_idx) & (indices <= exit_idx)]
    line, sample_xy, sample_time_s = _make_passage_line(xy[indices], time_s[indices], max_sample_points)
    if line is None:
        base_diag["status"] = "invalid_crosswalk_line"
        return None, base_diag
    base_diag["status"] = "legal_green"
    return (
        LegalPassage(
            location=track.location,
            city=LOCATION_DISPLAY.get(track.location, track.location),
            scene_id=track.scene_id,
            agent_id=track.agent_id,
            class_name=track.class_name,
            passage_type="vru_crossing",
            control_id=control_id,
            signal_status="GREEN",
            green_start_ts=green_start,
            green_end_ts=green_end,
            entry_ts=scene_ts,
            exit_ts=_scene_ts_at(state, exit_idx),
            entry_time_s=float(time_s[entry_idx]) if len(time_s) else float(entry_idx * dt),
            exit_time_s=float(time_s[exit_idx]) if len(time_s) else float(exit_idx * dt),
            line=line,
            sample_xy=sample_xy,
            sample_time_s=sample_time_s,
            path_length_m=float(line.length),
        ),
        base_diag,
    )


def _passage_row(passage: LegalPassage) -> Dict[str, Any]:
    return {
        "location": passage.location,
        "city": passage.city,
        "scene_id": passage.scene_id,
        "agent_id": passage.agent_id,
        "class_name": passage.class_name,
        "passage_type": passage.passage_type,
        "control_id": passage.control_id,
        "signal_status": passage.signal_status,
        "green_start_ts": passage.green_start_ts,
        "green_end_ts": passage.green_end_ts,
        "entry_ts": passage.entry_ts,
        "exit_ts": passage.exit_ts,
        "entry_time_s": passage.entry_time_s,
        "exit_time_s": passage.exit_time_s,
        "path_length_m": passage.path_length_m,
    }


def extract_legal_passages(
    tracks: Sequence[TrackRecord],
    lane_indices: Mapping[str, Any],
    roi_polygons: Mapping[str, Polygon],
    crosswalks: Mapping[str, Any],
    signal_tables: Mapping[Tuple[str, str], pd.DataFrame],
    dt: float,
    lane_match_threshold_m: float,
    lane_window: int,
    min_crosswalk_path_m: float,
    max_sample_points: int,
    include_motorcycle_as_vehicle: bool,
) -> Tuple[Dict[Tuple[str, str], List[LegalPassage]], Dict[Tuple[str, str], List[LegalPassage]], pd.DataFrame, pd.DataFrame]:
    vehicle_by_scene: Dict[Tuple[str, str], List[LegalPassage]] = {}
    vru_by_scene: Dict[Tuple[str, str], List[LegalPassage]] = {}
    passage_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    vehicle_classes = set(MOTOR_VEHICLE_CLASSES)
    if include_motorcycle_as_vehicle:
        vehicle_classes.add("motorcycle")

    for idx, track in enumerate(tracks, start=1):
        if track.location not in roi_polygons:
            continue
        tls_df = signal_tables.get((track.location, track.scene_id))
        passage: Optional[LegalPassage] = None
        diag: Optional[Dict[str, Any]] = None
        if track.class_name in vehicle_classes and track.location in lane_indices:
            passage, diag = _extract_vehicle_passage(
                track,
                roi_polygons[track.location],
                lane_indices[track.location],
                tls_df,
                dt,
                lane_match_threshold_m,
                lane_window,
                max_sample_points,
            )
            if passage is not None:
                vehicle_by_scene.setdefault((track.location, track.scene_id), []).append(passage)
        elif track.class_name in VRU_CLASSES:
            passage, diag = _extract_vru_passage(
                track,
                roi_polygons[track.location],
                crosswalks[track.location],
                tls_df,
                dt,
                min_crosswalk_path_m,
                max_sample_points,
            )
            if passage is not None:
                vru_by_scene.setdefault((track.location, track.scene_id), []).append(passage)
        if passage is not None:
            passage_rows.append(_passage_row(passage))
        if diag is not None:
            diagnostic_rows.append(diag)
        if idx % 8000 == 0:
            print(f"Processed {idx}/{len(tracks)} tracks for legal green passages...", flush=True)

    return vehicle_by_scene, vru_by_scene, pd.DataFrame(passage_rows), pd.DataFrame(diagnostic_rows)


def _green_phase_overlap(a: LegalPassage, b: LegalPassage) -> Tuple[int, int, int]:
    start = max(a.green_start_ts, b.green_start_ts)
    end = min(a.green_end_ts, b.green_end_ts)
    return start, end, max(0, end - start + 1)


def _closest_samples(a: LegalPassage, b: LegalPassage) -> Tuple[float, float, float, float, float]:
    tree = cKDTree(b.sample_xy)
    distances, indices = tree.query(a.sample_xy, k=1)
    a_idx = int(np.argmin(distances))
    b_idx = int(indices[a_idx])
    point = (a.sample_xy[a_idx] + b.sample_xy[b_idx]) / 2.0
    return (
        float(distances[a_idx]),
        float(point[0]),
        float(point[1]),
        float(a.sample_time_s[a_idx]),
        float(b.sample_time_s[b_idx]),
    )


def _line_intersection_point(a: LineString, b: LineString) -> Tuple[float, float, str]:
    try:
        inter = a.intersection(b)
    except Exception:
        inter = None
    if inter is None or inter.is_empty:
        p1, p2 = nearest_points(a, b)
        return float((p1.x + p2.x) / 2.0), float((p1.y + p2.y) / 2.0), "nearest"
    geom_type = inter.geom_type
    if geom_type == "Point":
        return float(inter.x), float(inter.y), "exact_point"
    if geom_type in {"MultiPoint", "GeometryCollection"}:
        points = [geom for geom in getattr(inter, "geoms", []) if geom.geom_type == "Point"]
        if points:
            return float(points[0].x), float(points[0].y), "exact_multipoint"
    centroid = inter.centroid
    return float(centroid.x), float(centroid.y), f"exact_{geom_type.lower()}"


def compute_overlap_events(
    vehicle_by_scene: Mapping[Tuple[str, str], List[LegalPassage]],
    vru_by_scene: Mapping[Tuple[str, str], List[LegalPassage]],
    dt: float,
    spatial_threshold_m: float,
    min_phase_overlap_s: float,
    simultaneous_time_gap_s: float,
    max_pairs_per_scene: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    min_overlap_frames = max(1, int(math.ceil(min_phase_overlap_s / dt)))
    for scene_key, vehicles in vehicle_by_scene.items():
        vrus = vru_by_scene.get(scene_key, [])
        if not vehicles or not vrus:
            continue
        pair_count = 0
        for vehicle in vehicles:
            for vru in vrus:
                phase_start, phase_end, phase_frames = _green_phase_overlap(vehicle, vru)
                if phase_frames < min_overlap_frames:
                    continue
                distance_m, cx, cy, veh_time, vru_time = _closest_samples(vehicle, vru)
                if distance_m > spatial_threshold_m:
                    continue
                ix, iy, intersection_kind = _line_intersection_point(vehicle.line, vru.line)
                if intersection_kind.startswith("exact"):
                    cx, cy = ix, iy
                rows.append(
                    {
                        "location": vehicle.location,
                        "city": vehicle.city,
                        "scene_id": vehicle.scene_id,
                        "vehicle_agent_id": vehicle.agent_id,
                        "vehicle_class": vehicle.class_name,
                        "vehicle_control_id": vehicle.control_id,
                        "vehicle_green_start_ts": vehicle.green_start_ts,
                        "vehicle_green_end_ts": vehicle.green_end_ts,
                        "vru_agent_id": vru.agent_id,
                        "vru_class": vru.class_name,
                        "vru_control_id": vru.control_id,
                        "vru_green_start_ts": vru.green_start_ts,
                        "vru_green_end_ts": vru.green_end_ts,
                        "phase_overlap_start_ts": phase_start,
                        "phase_overlap_end_ts": phase_end,
                        "phase_overlap_s": float(phase_frames * dt),
                        "spatial_gap_m": distance_m,
                        "x": float(cx),
                        "y": float(cy),
                        "intersection_kind": intersection_kind,
                        "vehicle_time_at_overlap_s": veh_time,
                        "vru_time_at_overlap_s": vru_time,
                        "arrival_time_gap_s": float(vru_time - veh_time),
                        "abs_arrival_time_gap_s": float(abs(vru_time - veh_time)),
                        "simultaneous_overlap": bool(abs(vru_time - veh_time) <= simultaneous_time_gap_s),
                        "overlap_cell_x_1m": int(math.floor(cx)),
                        "overlap_cell_y_1m": int(math.floor(cy)),
                    }
                )
                pair_count += 1
                if pair_count >= max_pairs_per_scene:
                    break
            if pair_count >= max_pairs_per_scene:
                break
    return pd.DataFrame(rows)


def summarize(
    locations: Sequence[str],
    passage_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    for location in locations:
        passages = passage_df[passage_df["location"] == location] if not passage_df.empty else pd.DataFrame()
        diag = diagnostic_df[diagnostic_df["location"] == location] if not diagnostic_df.empty else pd.DataFrame()
        overlaps = overlap_df[overlap_df["location"] == location] if not overlap_df.empty else pd.DataFrame()
        legal_vehicle = passages[passages["passage_type"] == "vehicle"] if not passages.empty else pd.DataFrame()
        legal_vru = passages[passages["passage_type"] == "vru_crossing"] if not passages.empty else pd.DataFrame()
        unique_cells = (
            overlaps[["overlap_cell_x_1m", "overlap_cell_y_1m"]].drop_duplicates().shape[0]
            if not overlaps.empty
            else 0
        )
        rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "legal_vehicle_green_passages": int(len(legal_vehicle)),
                "legal_vru_green_crossings": int(len(legal_vru)),
                "overlap_events": int(len(overlaps)),
                "simultaneous_overlap_events": int(overlaps["simultaneous_overlap"].sum()) if not overlaps.empty else 0,
                "unique_spatial_overlap_cells_1m": int(unique_cells),
                "mean_spatial_gap_m": float(overlaps["spatial_gap_m"].mean()) if not overlaps.empty else 0.0,
                "median_abs_arrival_time_gap_s": float(overlaps["abs_arrival_time_gap_s"].median()) if not overlaps.empty else 0.0,
                "vehicle_signal_unobservable": int(((diag["passage_type"] == "vehicle") & (diag["status"] == "signal_unobservable")).sum()) if not diag.empty else 0,
                "vru_signal_unobservable": int(((diag["passage_type"] == "vru_crossing") & (diag["status"] == "signal_unobservable")).sum()) if not diag.empty else 0,
                "vru_no_crosswalk_passage": int(((diag["passage_type"] == "vru_crossing") & (diag["status"] == "no_crosswalk_passage")).sum()) if not diag.empty else 0,
            }
        )
        if not overlaps.empty:
            group_cols = ["location", "city", "vehicle_control_id", "vru_control_id"]
            for keys, group in overlaps.groupby(group_cols):
                cluster_rows.append(
                    {
                        "location": keys[0],
                        "city": keys[1],
                        "vehicle_control_id": keys[2],
                        "vru_control_id": keys[3],
                        "overlap_events": int(len(group)),
                        "simultaneous_overlap_events": int(group["simultaneous_overlap"].sum()),
                        "unique_spatial_overlap_cells_1m": int(group[["overlap_cell_x_1m", "overlap_cell_y_1m"]].drop_duplicates().shape[0]),
                        "mean_spatial_gap_m": float(group["spatial_gap_m"].mean()),
                        "median_abs_arrival_time_gap_s": float(group["abs_arrival_time_gap_s"].median()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cluster_rows)


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["overlap_events"], width, label="Green-phase spatial overlaps", color="#2f6f8f")
    ax.bar(x + width / 2, summary_df["unique_spatial_overlap_cells_1m"], width, label="Unique 1m cells", color="#c46a2b")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["location"])
    ax.set_ylabel("Count")
    ax.set_title("Legal Green-Phase Right-of-Way Overlap")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_location_hotspot(
    location: str,
    overlaps: pd.DataFrame,
    lane_refs: Sequence[LaneReference],
    crosswalk_set: Any,
    roi_polygon: Polygon,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.4), dpi=180)
    for lane in lane_refs:
        if len(lane.xy):
            ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#425563", linewidth=0.55, alpha=0.35)
    for poly, name in zip(crosswalk_set.polygons, crosswalk_set.names):
        coords = np.asarray(poly.exterior.coords, dtype=float)
        ax.add_patch(MplPolygon(coords, closed=True, facecolor="#8ed1c6", edgecolor="#087f73", alpha=0.22, linewidth=0.8))
        center = coords.mean(axis=0)
        ax.text(center[0], center[1], str(name).replace("_crosswalk", ""), fontsize=7, color="#075e57", ha="center", va="center")
    roi_xy = np.asarray(roi_polygon.exterior.coords, dtype=float)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#111111", linewidth=1.6, alpha=0.8)
    if not overlaps.empty:
        grouped = overlaps.groupby(["overlap_cell_x_1m", "overlap_cell_y_1m"]).agg({"x": "mean", "y": "mean", "simultaneous_overlap": "sum", "vehicle_agent_id": "count"}).reset_index()
        sizes = 22 + 7 * np.sqrt(grouped["vehicle_agent_id"].to_numpy(dtype=float))
        colors = grouped["vehicle_agent_id"].to_numpy(dtype=float)
        sc = ax.scatter(grouped["x"], grouped["y"], s=sizes, c=colors, cmap="magma", alpha=0.78, edgecolors="white", linewidths=0.35)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Overlap events per 1m cell")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{LOCATION_DISPLAY.get(location, location)} legal right-of-way overlap hotspots")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _html_table(df: pd.DataFrame, columns: Sequence[Tuple[str, str]], max_rows: int = 200) -> str:
    rows = df.head(max_rows).to_dict("records") if not df.empty else []
    parts = ["<table><thead><tr>"]
    for _, label in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (float, np.floating)):
                cell = f"{float(value):.3f}"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html(output_path: Path, summary_df: pd.DataFrame, cluster_df: pd.DataFrame, locations: Sequence[str]) -> None:
    summary_table = _html_table(
        summary_df,
        [
            ("location", "Location"),
            ("legal_vehicle_green_passages", "Legal vehicle"),
            ("legal_vru_green_crossings", "Legal VRU crossing"),
            ("overlap_events", "Overlap events"),
            ("simultaneous_overlap_events", "Simultaneous"),
            ("unique_spatial_overlap_cells_1m", "Unique 1m cells"),
            ("median_abs_arrival_time_gap_s", "Median |dt|"),
            ("vru_signal_unobservable", "VRU signal unobs."),
        ],
    )
    cluster_table = _html_table(
        cluster_df.sort_values("overlap_events", ascending=False) if not cluster_df.empty else cluster_df,
        [
            ("location", "Location"),
            ("vehicle_control_id", "Vehicle control"),
            ("vru_control_id", "VRU control"),
            ("overlap_events", "Overlap events"),
            ("unique_spatial_overlap_cells_1m", "Unique cells"),
            ("median_abs_arrival_time_gap_s", "Median |dt|"),
        ],
        max_rows=120,
    )
    location_imgs = "\n".join(
        f"<section class='panel'><h2>{html.escape(location)}</h2><a href='overlap_hotspot_{html.escape(location)}.png'><img src='overlap_hotspot_{html.escape(location)}.png' alt='{html.escape(location)} hotspot'></a></section>"
        for location in locations
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SinD Spatio-Temporal Right-of-Way Overlap</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; color:#172126; background:#f6f3ee; }}
    header {{ padding:32px 5vw; background:#1f3b4d; color:white; }}
    main {{ padding:24px 5vw 48px; }}
    .panel {{ background:white; border:1px solid #d7d0c6; padding:18px; margin:18px 0; overflow:auto; }}
    img {{ max-width:100%; background:white; border:1px solid #d7d0c6; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e3ddd4; padding:8px 9px; text-align:left; white-space:nowrap; }}
    th {{ background:#ece2d5; }}
    .note {{ color:#5b6870; line-height:1.55; }}
  </style>
</head>
<body>
  <header>
    <h1>SinD 时空路权重叠度</h1>
    <p>统计同一绿灯相位内，合法机动车通行轨迹与合法行人/自行车过街轨迹的空间交点。</p>
  </header>
  <main>
    <section class="panel">
      <p class="note">主指标要求双方入口/过街入口均为绑定信号 GREEN，且两个 green interval 有时间重叠；地图或灯控绑定不可观测的轨迹只进入 diagnostics，不计入合法重叠分母。</p>
      <a href="right_of_way_overlap_summary.png"><img src="right_of_way_overlap_summary.png" alt="summary"></a>
    </section>
    <section class="panel"><h2>Location Summary</h2>{summary_table}</section>
    <section class="panel"><h2>Top Control-Pair Clusters</h2>{cluster_table}</section>
    {location_imgs}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--traffic-light-dir", type=Path, default=DEFAULT_TL_ROOT)
    parser.add_argument("--traffic-light-mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--disable-traffic-lights", action="store_true")
    parser.add_argument("--disable-static-filter", action="store_true")
    parser.add_argument("--lane-match-threshold-m", type=float, default=3.0)
    parser.add_argument("--lane-window", type=int, default=8)
    parser.add_argument("--min-crosswalk-path-m", type=float, default=2.0)
    parser.add_argument("--spatial-threshold-m", type=float, default=2.0)
    parser.add_argument("--min-phase-overlap-s", type=float, default=1.0)
    parser.add_argument("--simultaneous-time-gap-s", type=float, default=5.0)
    parser.add_argument("--max-sample-points", type=int, default=80)
    parser.add_argument("--max-pairs-per-scene", type=int, default=4000)
    parser.add_argument("--include-motorcycle-as-vehicle", action="store_true")
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
    tracks = load_sind_tracks(args.data_dir, locations)
    tracks, static_removed_df = filter_static_tracks(
        tracks,
        dt=args.dt,
        enabled=not args.disable_static_filter,
        static_min_duration_s=args.static_min_duration_s,
        static_max_displacement_m=args.static_max_displacement_m,
        static_max_path_length_m=args.static_max_path_length_m,
        static_max_speed_p95_mps=args.static_max_speed_p95_mps,
        static_min_slow_duration_s=args.static_min_slow_duration_s,
        static_max_mean_path_speed_mps=args.static_max_mean_path_speed_mps,
    )
    target_classes = set(MOTOR_VEHICLE_CLASSES) | set(VRU_CLASSES)
    if args.include_motorcycle_as_vehicle:
        target_classes.add("motorcycle")
    tracks = [track for track in tracks if track.class_name in target_classes]
    print(f"Target tracks after static filtering: {len(tracks)}; static removed={len(static_removed_df)}", flush=True)

    print("Loading maps, lane indices, crosswalks and core ROIs...", flush=True)
    references = load_lane_references(args.data_dir, locations)
    lane_indices = build_lane_indices(references, spacing=1.0)
    crosswalks = load_crosswalks(args.data_dir, locations)
    roi_polygons: Dict[str, Polygon] = {}
    roi_debug_rows: List[Dict[str, Any]] = []
    for location in locations:
        roi, debug = infer_intersection_core_roi(references[location])
        roi_polygons[location] = roi
        roi_debug_rows.append({"location": location, "city": LOCATION_DISPLAY.get(location, location), **debug})

    print("Building scene signal tables...", flush=True)
    tracks_by_scene: Dict[Tuple[str, str], List[TrackRecord]] = {}
    for track in tracks:
        tracks_by_scene.setdefault((track.location, track.scene_id), []).append(track)
    signal_tables, signal_report_df = build_scene_signal_tables(
        tracks_by_scene,
        dt=args.dt,
        traffic_light_root=configured_traffic_light_root(args.traffic_light_dir),
        mapping_path=args.traffic_light_mapping,
        pkl_root=args.data_dir,
        enabled=not args.disable_traffic_lights,
    )

    print("Extracting legal green vehicle and VRU crossing passages...", flush=True)
    vehicle_by_scene, vru_by_scene, passage_df, diagnostic_df = extract_legal_passages(
        tracks,
        lane_indices=lane_indices,
        roi_polygons=roi_polygons,
        crosswalks=crosswalks,
        signal_tables=signal_tables,
        dt=args.dt,
        lane_match_threshold_m=args.lane_match_threshold_m,
        lane_window=args.lane_window,
        min_crosswalk_path_m=args.min_crosswalk_path_m,
        max_sample_points=args.max_sample_points,
        include_motorcycle_as_vehicle=args.include_motorcycle_as_vehicle,
    )

    print("Computing legal green-phase spatial overlaps...", flush=True)
    overlap_df = compute_overlap_events(
        vehicle_by_scene,
        vru_by_scene,
        dt=args.dt,
        spatial_threshold_m=args.spatial_threshold_m,
        min_phase_overlap_s=args.min_phase_overlap_s,
        simultaneous_time_gap_s=args.simultaneous_time_gap_s,
        max_pairs_per_scene=args.max_pairs_per_scene,
    )
    summary_df, cluster_df = summarize(locations, passage_df, diagnostic_df, overlap_df)

    print("Writing outputs and plots...", flush=True)
    passage_df.to_csv(args.output_dir / "legal_green_passages.csv", index=False)
    diagnostic_df.to_csv(args.output_dir / "passage_observability_diagnostics.csv", index=False)
    overlap_df.to_csv(args.output_dir / "right_of_way_overlap_events.csv", index=False)
    summary_df.to_csv(args.output_dir / "right_of_way_overlap_summary_by_location.csv", index=False)
    cluster_df.to_csv(args.output_dir / "right_of_way_overlap_cluster_summary.csv", index=False)
    signal_report_df.to_csv(args.output_dir / "signal_table_reports.csv", index=False)
    static_removed_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)
    pd.DataFrame(roi_debug_rows).to_csv(args.output_dir / "roi_debug.csv", index=False)
    plot_summary(summary_df, args.output_dir / "right_of_way_overlap_summary.png")
    for location in locations:
        plot_location_hotspot(
            location,
            overlap_df[overlap_df["location"] == location] if not overlap_df.empty else pd.DataFrame(),
            references[location],
            crosswalks[location],
            roi_polygons[location],
            args.output_dir / f"overlap_hotspot_{location}.png",
        )
    methodology = {
        "metric": "Spatio-Temporal Right-of-Way Overlap",
        "legal_vehicle_definition": "motor-vehicle track enters inferred core ROI while its bound map lane signal is GREEN",
        "legal_vru_definition": "pedestrian/bicycle traverses a Lanelet2 crosswalk while the bound crosswalk/pedestrian signal is GREEN",
        "overlap_definition": f"same scene, overlapping GREEN intervals >= {args.min_phase_overlap_s}s, and closest path gap <= {args.spatial_threshold_m}m",
        "simultaneous_definition": f"absolute arrival time gap at closest overlap point <= {args.simultaneous_time_gap_s}s",
        "excluded_from_denominator": "tracks with unavailable lane/crosswalk signal binding or missing crosswalk geometry",
        "include_motorcycle_as_vehicle": bool(args.include_motorcycle_as_vehicle),
    }
    (args.output_dir / "methodology_notes.json").write_text(json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(args.output_dir / "index.html", summary_df, cluster_df, locations)
    print(f"Done. Open {args.output_dir / 'index.html'}", flush=True)
    if not summary_df.empty:
        print(summary_df[["location", "legal_vehicle_green_passages", "legal_vru_green_crossings", "overlap_events", "unique_spatial_overlap_cells_1m"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
