#!/usr/bin/env python
"""Compute SinD turning-trajectory lateral deviation variance.

The metric measures signed lateral offsets between vehicle turning trajectories and
Lanelet2 reference centerlines. It exports per-track statistics, per-reference-line
cluster summaries, trajectory-bundle plots, and an HTML report.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
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
from scipy.spatial import cKDTree

# Allow running this file directly from the repository root.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinematic_envelopes import (  # noqa: E402
    LOCATION_DISPLAY,
    SIND_LOCATIONS,
    TrackRecord,
    filter_tracks,
    load_sind_tracks,
    normalize_location,
)
from trajdata.dataset_specific.sind.sind_lanelet2_utils import (  # noqa: E402
    LL2XYProjector,
    orgins,
)

DEFAULT_SIX_LOCATIONS: Tuple[str, ...] = ("cc", "tj", "cqIR", "cqNR", "cqR", "xasl")
VEHICLE_CLASSES = ("car", "truck", "bus", "tricycle")
TURN_MANEUVERS = ("left-turn", "right-turn")


@dataclass(frozen=True)
class LaneReference:
    location: str
    lane_id: str
    xy: np.ndarray
    cumulative_s: np.ndarray
    sample_xy: np.ndarray

    @property
    def length(self) -> float:
        return float(self.cumulative_s[-1]) if len(self.cumulative_s) else 0.0


@dataclass(frozen=True)
class TrackLateralResult:
    location: str
    city: str
    scene_id: str
    agent_id: str
    class_name: str
    maneuver: str
    reference_lane_id: str
    num_points: int
    mean_lateral_offset_m: float
    var_lateral_offset_m2: float
    std_lateral_offset_m: float
    max_abs_lateral_offset_m: float
    mean_abs_lateral_offset_m: float
    p95_abs_lateral_offset_m: float
    match_mean_distance_m: float
    s_min_m: float
    s_max_m: float


@dataclass(frozen=True)
class UnmatchedTrackResult:
    location: str
    city: str
    scene_id: str
    agent_id: str
    class_name: str
    maneuver: str
    best_lane_id: str
    best_mean_distance_m: float
    reason: str


@dataclass
class OffsetSeries:
    track_key: Tuple[str, str, str]
    location: str
    maneuver: str
    lane_id: str
    xy: np.ndarray
    s: np.ndarray
    offsets: np.ndarray


def normalize_maneuver_arg(value: str) -> str:
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "left": "left-turn",
        "left-turn": "left-turn",
        "leftturn": "left-turn",
        "right": "right-turn",
        "right-turn": "right-turn",
        "rightturn": "right-turn",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported turn maneuver: {value}")
    return aliases[key]


def _lanelet2_path(data_dir: Path, location: str) -> Path:
    path = data_dir / "Lanelet_maps_SinD" / f"lanelet2_{location}.osm"
    if path.exists():
        return path
    fallback = REPO_ROOT / "Lanelet_maps_SinD" / f"lanelet2_{location}.osm"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Lanelet2 map not found for {location}: {path}")


def _polyline_cumulative_s(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.array([], dtype=float)
    if len(xy) == 1:
        return np.array([0.0], dtype=float)
    seg_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def _resample_polyline(xy: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    if len(xy) < 2:
        return xy.copy()
    cumulative_s = _polyline_cumulative_s(xy)
    total = cumulative_s[-1]
    if total <= 0:
        return xy[[0]].copy()
    num = max(2, int(math.ceil(total / spacing)) + 1)
    s_grid = np.linspace(0.0, total, num)
    return np.column_stack(
        [np.interp(s_grid, cumulative_s, xy[:, 0]), np.interp(s_grid, cumulative_s, xy[:, 1])]
    )


def _dedupe_polyline(xy: np.ndarray) -> np.ndarray:
    if len(xy) <= 1:
        return xy
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6
    return xy[keep]


def load_lane_references(data_dir: Path, locations: Sequence[str]) -> Dict[str, List[LaneReference]]:
    references: Dict[str, List[LaneReference]] = {}
    for location in locations:
        path = _lanelet2_path(data_dir, location)
        tree = ET.parse(path)
        root = tree.getroot()
        projector = LL2XYProjector(orgins[location][0], orgins[location][1])

        nodes: Dict[int, Tuple[float, float]] = {}
        ways: Dict[int, List[int]] = {}
        lanelets: Dict[int, Dict[str, Optional[int]]] = {}

        for node in root.findall("node"):
            node_id = int(node.get("id"))
            lat = float(node.get("lat"))
            lon = float(node.get("lon"))
            x, y = projector.latlon2xy(lat, lon)
            nodes[node_id] = (float(x), float(y))

        for way in root.findall("way"):
            way_id = int(way.get("id"))
            ways[way_id] = [int(nd.get("ref")) for nd in way.findall("nd")]

        for relation in root.findall("relation"):
            tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
            if tags.get("type") != "lanelet" or tags.get("subtype") == "crosswalk":
                continue
            relation_id = int(relation.get("id"))
            left_way = None
            right_way = None
            for member in relation.findall("member"):
                role = member.get("role")
                ref = int(member.get("ref"))
                if role == "left":
                    left_way = ref
                elif role == "right":
                    right_way = ref
            lanelets[relation_id] = {
                "left": left_way,
                "right": right_way,
                "name": tags.get("name", f"lanelet_{relation_id}"),
            }

        loc_refs: List[LaneReference] = []
        for lanelet_id, lanelet in lanelets.items():
            left_pts = None
            right_pts = None
            if lanelet["left"] in ways:
                left_pts = np.array([nodes[nid] for nid in ways[lanelet["left"]] if nid in nodes], dtype=float)
            if lanelet["right"] in ways:
                right_pts = np.array([nodes[nid] for nid in ways[lanelet["right"]] if nid in nodes], dtype=float)
            if left_pts is not None and right_pts is not None and len(left_pts) and len(right_pts):
                min_len = min(len(left_pts), len(right_pts))
                center = (left_pts[:min_len] + right_pts[:min_len]) / 2.0
            elif left_pts is not None and len(left_pts):
                center = left_pts.copy()
            elif right_pts is not None and len(right_pts):
                center = right_pts.copy()
            else:
                continue
            center = _dedupe_polyline(center)
            if len(center) < 2:
                continue
            cumulative_s = _polyline_cumulative_s(center)
            if cumulative_s[-1] < 2.0:
                continue
            loc_refs.append(
                LaneReference(
                    location=location,
                    lane_id=str(lanelet["name"]),
                    xy=center,
                    cumulative_s=cumulative_s,
                    sample_xy=_resample_polyline(center, spacing=2.0),
                )
            )
        references[location] = loc_refs
    return references


def build_lane_kdtrees(references: Mapping[str, Sequence[LaneReference]]) -> Dict[str, Tuple[cKDTree, List[int]]]:
    kdtrees: Dict[str, Tuple[cKDTree, List[int]]] = {}
    for location, lanes in references.items():
        sample_points = []
        lane_indices = []
        for lane_idx, lane in enumerate(lanes):
            sample_points.extend(lane.sample_xy[:, :2])
            lane_indices.extend([lane_idx] * len(lane.sample_xy))
        if sample_points:
            kdtrees[location] = (cKDTree(np.asarray(sample_points, dtype=float)), lane_indices)
    return kdtrees


def project_points_to_lane(points_xy: np.ndarray, lane: LaneReference) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points to a lane centerline.

    Returns signed offsets, unsigned distances, arc-length coordinates, projected
    points, and left normals for the nearest segment.
    """
    if len(points_xy) == 0 or len(lane.xy) < 2:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.empty((0, 2)), np.empty((0, 2))

    p0 = lane.xy[:-1]
    p1 = lane.xy[1:]
    seg = p1 - p0
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 1e-9
    p0 = p0[valid]
    seg = seg[valid]
    seg_len = seg_len[valid]
    s0 = lane.cumulative_s[:-1][valid]
    if len(seg) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.empty((0, 2)), np.empty((0, 2))

    # Shape: points x segments x xy.
    delta = points_xy[:, None, :] - p0[None, :, :]
    seg_len_sq = np.maximum(seg_len**2, 1e-9)
    t = np.clip(np.sum(delta * seg[None, :, :], axis=2) / seg_len_sq[None, :], 0.0, 1.0)
    projected_all = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
    diff = points_xy[:, None, :] - projected_all
    dist_sq = np.sum(diff**2, axis=2)
    best_idx = np.argmin(dist_sq, axis=1)
    row_idx = np.arange(len(points_xy))
    projected = projected_all[row_idx, best_idx]
    distances = np.sqrt(dist_sq[row_idx, best_idx])
    unit = seg / seg_len[:, None]
    normals = np.column_stack([-unit[:, 1], unit[:, 0]])
    best_normals = normals[best_idx]
    signed_offsets = np.sum((points_xy - projected) * best_normals, axis=1)
    s = s0[best_idx] + t[row_idx, best_idx] * seg_len[best_idx]
    return signed_offsets, distances, s, projected, best_normals


def _track_points(track: TrackRecord) -> np.ndarray:
    state = track.state.sort_values("frame_id")
    if not {"x", "y"}.issubset(state.columns):
        return np.empty((0, 2), dtype=float)
    xy = state[["x", "y"]].to_numpy(dtype=float)
    return xy[np.isfinite(xy).all(axis=1)]


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[indices]


def candidate_lane_indices(
    location: str,
    points_xy: np.ndarray,
    references: Mapping[str, Sequence[LaneReference]],
    kdtrees: Mapping[str, Tuple[cKDTree, List[int]]],
    k_nearest: int,
) -> List[int]:
    if location not in kdtrees or not len(points_xy):
        return list(range(len(references.get(location, []))))
    tree, sample_lane_indices = kdtrees[location]
    k = min(k_nearest, len(sample_lane_indices))
    _, raw_indices = tree.query(points_xy, k=k)
    raw_indices = np.atleast_2d(raw_indices)
    lane_ids = {sample_lane_indices[int(idx)] for idx in raw_indices.ravel()}
    return sorted(lane_ids)


def match_track_to_lane(
    track: TrackRecord,
    references: Mapping[str, Sequence[LaneReference]],
    kdtrees: Mapping[str, Tuple[cKDTree, List[int]]],
    max_match_points: int,
    k_nearest_lanes: int,
) -> Tuple[Optional[LaneReference], float, np.ndarray]:
    points = _track_points(track)
    if len(points) < 2:
        return None, float("inf"), points
    match_points = _sample_points(points, max_match_points)
    lanes = references.get(track.location, [])
    candidates = candidate_lane_indices(track.location, match_points, references, kdtrees, k_nearest_lanes)
    best_lane = None
    best_dist = float("inf")
    for lane_idx in candidates:
        lane = lanes[lane_idx]
        _, distances, _, _, _ = project_points_to_lane(match_points, lane)
        if len(distances) == 0:
            continue
        mean_dist = float(np.mean(distances))
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_lane = lane
    return best_lane, best_dist, points


def analyze_tracks(
    tracks: Sequence[TrackRecord],
    references: Mapping[str, Sequence[LaneReference]],
    max_match_distance_m: float,
    max_match_points: int,
    k_nearest_lanes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], List[OffsetSeries]]]:
    kdtrees = build_lane_kdtrees(references)
    track_rows: List[TrackLateralResult] = []
    unmatched_rows: List[UnmatchedTrackResult] = []
    offset_groups: Dict[Tuple[str, str, str], List[OffsetSeries]] = {}

    for idx, track in enumerate(tracks, start=1):
        lane, best_dist, points = match_track_to_lane(
            track, references, kdtrees, max_match_points=max_match_points, k_nearest_lanes=k_nearest_lanes
        )
        city = LOCATION_DISPLAY.get(track.location, track.location)
        if lane is None:
            unmatched_rows.append(
                UnmatchedTrackResult(
                    track.location, city, track.scene_id, track.agent_id, track.class_name, track.maneuver,
                    "", float("inf"), "no_reference_lane_or_invalid_track"
                )
            )
            continue
        if best_dist > max_match_distance_m:
            unmatched_rows.append(
                UnmatchedTrackResult(
                    track.location, city, track.scene_id, track.agent_id, track.class_name, track.maneuver,
                    lane.lane_id, best_dist, "match_distance_exceeds_threshold"
                )
            )
            continue

        offsets, distances, s, _, _ = project_points_to_lane(points, lane)
        if len(offsets) < 2:
            unmatched_rows.append(
                UnmatchedTrackResult(
                    track.location, city, track.scene_id, track.agent_id, track.class_name, track.maneuver,
                    lane.lane_id, best_dist, "not_enough_projected_points"
                )
            )
            continue

        abs_offsets = np.abs(offsets)
        track_rows.append(
            TrackLateralResult(
                location=track.location,
                city=city,
                scene_id=track.scene_id,
                agent_id=track.agent_id,
                class_name=track.class_name,
                maneuver=track.maneuver,
                reference_lane_id=lane.lane_id,
                num_points=len(offsets),
                mean_lateral_offset_m=float(np.mean(offsets)),
                var_lateral_offset_m2=float(np.var(offsets)),
                std_lateral_offset_m=float(np.std(offsets)),
                max_abs_lateral_offset_m=float(np.max(abs_offsets)),
                mean_abs_lateral_offset_m=float(np.mean(abs_offsets)),
                p95_abs_lateral_offset_m=float(np.quantile(abs_offsets, 0.95)),
                match_mean_distance_m=best_dist,
                s_min_m=float(np.min(s)),
                s_max_m=float(np.max(s)),
            )
        )
        group_key = (track.location, track.maneuver, lane.lane_id)
        offset_groups.setdefault(group_key, []).append(
            OffsetSeries(
                track_key=(track.scene_id, track.agent_id, track.class_name),
                location=track.location,
                maneuver=track.maneuver,
                lane_id=lane.lane_id,
                xy=points,
                s=s,
                offsets=offsets,
            )
        )

        if idx % 1000 == 0:
            print(f"Analyzed {idx}/{len(tracks)} tracks...", flush=True)

    track_df = pd.DataFrame([row.__dict__ for row in track_rows])
    unmatched_df = pd.DataFrame([row.__dict__ for row in unmatched_rows])
    return track_df, unmatched_df, offset_groups


def aggregate_clusters(
    track_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    requested_locations: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "location", "city", "maneuver", "reference_lane_id", "matched_tracks", "matched_points",
        "mean_lateral_offset_m", "mean_abs_lateral_offset_m", "mean_track_var_lateral_offset_m2",
        "pooled_var_lateral_offset_m2", "mean_track_std_lateral_offset_m", "p95_abs_lateral_offset_m",
        "max_abs_lateral_offset_m", "mean_match_distance_m", "unmatched_tracks_same_city_maneuver",
        "match_rate_same_city_maneuver",
    ]
    if track_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = []
    city_maneuver_unmatched = (
        unmatched_df.groupby(["location", "maneuver"]).size().to_dict() if not unmatched_df.empty else {}
    )
    city_maneuver_matched = track_df.groupby(["location", "maneuver"]).size().to_dict()
    for (location, maneuver, lane_id), group in track_df.groupby(["location", "maneuver", "reference_lane_id"]):
        matched_same = city_maneuver_matched.get((location, maneuver), 0)
        unmatched_same = city_maneuver_unmatched.get((location, maneuver), 0)
        denom = matched_same + unmatched_same
        weights = group["num_points"].to_numpy(dtype=float)
        if weights.sum() > 0:
            pooled_mean_abs = float(np.average(group["mean_abs_lateral_offset_m"], weights=weights))
            pooled_mean = float(np.average(group["mean_lateral_offset_m"], weights=weights))
        else:
            pooled_mean_abs = float(group["mean_abs_lateral_offset_m"].mean())
            pooled_mean = float(group["mean_lateral_offset_m"].mean())
        grouped.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "maneuver": maneuver,
                "reference_lane_id": lane_id,
                "matched_tracks": int(len(group)),
                "matched_points": int(group["num_points"].sum()),
                "mean_lateral_offset_m": pooled_mean,
                "mean_abs_lateral_offset_m": pooled_mean_abs,
                "mean_track_var_lateral_offset_m2": float(group["var_lateral_offset_m2"].mean()),
                "pooled_var_lateral_offset_m2": float(np.average(group["var_lateral_offset_m2"], weights=weights)) if weights.sum() > 0 else float(group["var_lateral_offset_m2"].mean()),
                "mean_track_std_lateral_offset_m": float(group["std_lateral_offset_m"].mean()),
                "p95_abs_lateral_offset_m": float(group["p95_abs_lateral_offset_m"].mean()),
                "max_abs_lateral_offset_m": float(group["max_abs_lateral_offset_m"].max()),
                "mean_match_distance_m": float(group["match_mean_distance_m"].mean()),
                "unmatched_tracks_same_city_maneuver": int(unmatched_same),
                "match_rate_same_city_maneuver": float(matched_same / denom) if denom else 0.0,
            }
        )
    cluster_df = pd.DataFrame(grouped, columns=columns)
    return cluster_df.sort_values(["location", "maneuver", "matched_tracks"], ascending=[True, True, False])


def _reference_at_s(lane: LaneReference, s_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.interp(s_values, lane.cumulative_s, lane.xy[:, 0])
    y = np.interp(s_values, lane.cumulative_s, lane.xy[:, 1])
    eps = max(lane.length / 1000.0, 0.05)
    s_prev = np.clip(s_values - eps, 0.0, lane.length)
    s_next = np.clip(s_values + eps, 0.0, lane.length)
    x_prev = np.interp(s_prev, lane.cumulative_s, lane.xy[:, 0])
    y_prev = np.interp(s_prev, lane.cumulative_s, lane.xy[:, 1])
    x_next = np.interp(s_next, lane.cumulative_s, lane.xy[:, 0])
    y_next = np.interp(s_next, lane.cumulative_s, lane.xy[:, 1])
    tangent = np.column_stack([x_next - x_prev, y_next - y_prev])
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-9] = 1.0
    tangent = tangent / norm[:, None]
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return np.column_stack([x, y]), normals


def _interpolate_offsets(series: OffsetSeries, s_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(series.s)
    s_sorted = series.s[order]
    offsets_sorted = series.offsets[order]
    uniq_s, uniq_idx = np.unique(s_sorted, return_index=True)
    uniq_offsets = offsets_sorted[uniq_idx]
    if len(uniq_s) < 2:
        return np.full(len(s_grid), np.nan)
    values = np.interp(s_grid, uniq_s, uniq_offsets)
    values[(s_grid < uniq_s[0]) | (s_grid > uniq_s[-1])] = np.nan
    return values


def plot_bundle(
    lane: LaneReference,
    series_list: Sequence[OffsetSeries],
    output_path: Path,
    title: str,
    min_tracks_for_band: int = 3,
    max_raw_tracks: int = 300,
    num_grid: int = 120,
) -> bool:
    if len(series_list) < min_tracks_for_band:
        return False
    all_s = np.concatenate([series.s for series in series_list if len(series.s)])
    if len(all_s) < 2:
        return False
    s_lo = float(np.quantile(all_s, 0.02))
    s_hi = float(np.quantile(all_s, 0.98))
    if s_hi <= s_lo:
        s_lo, s_hi = 0.0, lane.length
    s_grid = np.linspace(s_lo, s_hi, num_grid)
    offset_matrix = np.vstack([_interpolate_offsets(series, s_grid) for series in series_list])
    valid_counts = np.sum(np.isfinite(offset_matrix), axis=0)
    valid_mask = valid_counts >= min_tracks_for_band
    if valid_mask.sum() < 5:
        return False
    mean_offsets = np.nanmean(offset_matrix, axis=0)
    std_offsets = np.nanstd(offset_matrix, axis=0)
    ref_xy, normals = _reference_at_s(lane, s_grid)
    mean_xy = ref_xy + normals * mean_offsets[:, None]
    upper_xy = ref_xy + normals * (mean_offsets + 2.0 * std_offsets)[:, None]
    lower_xy = ref_xy + normals * (mean_offsets - 2.0 * std_offsets)[:, None]

    fig, ax = plt.subplots(figsize=(8.5, 8), dpi=150)
    raw_series = list(series_list)
    if len(raw_series) > max_raw_tracks:
        step = max(1, len(raw_series) // max_raw_tracks)
        raw_series = raw_series[::step][:max_raw_tracks]
    for series in raw_series:
        ax.plot(series.xy[:, 0], series.xy[:, 1], color="#345c72", alpha=0.10, linewidth=0.8)
    ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#242424", linewidth=2.0, linestyle="--", label="Reference centerline")
    ax.plot(mean_xy[valid_mask, 0], mean_xy[valid_mask, 1], color="#c43b2f", linewidth=2.4, label="Mean trajectory")
    band_mask = valid_mask & np.isfinite(upper_xy).all(axis=1) & np.isfinite(lower_xy).all(axis=1)
    if band_mask.sum() >= 5:
        band_poly = np.vstack([upper_xy[band_mask], lower_xy[band_mask][::-1]])
        ax.fill(band_poly[:, 0], band_poly[:, 1], color="#d98f45", alpha=0.25, label="Mean +/- 2sigma")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def generate_bundle_plots(
    cluster_df: pd.DataFrame,
    offset_groups: Mapping[Tuple[str, str, str], List[OffsetSeries]],
    references: Mapping[str, Sequence[LaneReference]],
    output_dir: Path,
    top_k_per_city_maneuver: int,
    min_tracks: int,
) -> pd.DataFrame:
    rows = []
    if cluster_df.empty:
        return pd.DataFrame()
    lane_lookup = {(lane.location, lane.lane_id): lane for lanes in references.values() for lane in lanes}
    selected = []
    for (location, maneuver), group in cluster_df.groupby(["location", "maneuver"]):
        group = group[group["matched_tracks"] >= min_tracks]
        selected.extend(group.sort_values("matched_tracks", ascending=False).head(top_k_per_city_maneuver).to_dict("records"))
    for record in selected:
        key = (record["location"], record["maneuver"], record["reference_lane_id"])
        lane = lane_lookup.get((record["location"], record["reference_lane_id"]))
        if lane is None:
            continue
        safe_lane_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in record["reference_lane_id"])
        filename = f"bundle_{record['location']}_{record['maneuver'].replace('-', '_')}_lane_{safe_lane_id}.png"
        title = (
            f"{LOCATION_DISPLAY.get(record['location'], record['location'])} {record['maneuver']} "
            f"lane {record['reference_lane_id']} | n={record['matched_tracks']} | "
            f"var={record['mean_track_var_lateral_offset_m2']:.2f} m^2"
        )
        created = plot_bundle(
            lane,
            offset_groups.get(key, []),
            output_dir / filename,
            title=title,
            min_tracks_for_band=min_tracks,
        )
        if created:
            rows.append({**record, "bundle_png": filename})
    return pd.DataFrame(rows)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    track_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    bundle_df: pd.DataFrame,
) -> Dict[str, Any]:
    city_summary = []
    for location in args.cities:
        matched = track_df[track_df["location"] == location] if not track_df.empty else pd.DataFrame()
        unmatched = unmatched_df[unmatched_df["location"] == location] if not unmatched_df.empty else pd.DataFrame()
        total = len(matched) + len(unmatched)
        city_summary.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "matched_tracks": int(len(matched)),
                "unmatched_tracks": int(len(unmatched)),
                "match_rate": float(len(matched) / total) if total else 0.0,
                "mean_track_var_lateral_offset_m2": float(matched["var_lateral_offset_m2"].mean()) if not matched.empty else None,
                "mean_abs_lateral_offset_m": float(matched["mean_abs_lateral_offset_m"].mean()) if not matched.empty else None,
                "max_abs_lateral_offset_m": float(matched["max_abs_lateral_offset_m"].max()) if not matched.empty else None,
            }
        )
    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "agent_type": args.agent_type,
        "maneuvers": args.maneuvers,
        "max_match_distance_m": args.max_match_distance_m,
        "matched_tracks_total": int(len(track_df)),
        "unmatched_tracks_total": int(len(unmatched_df)),
        "clusters_total": int(len(cluster_df)),
        "bundle_plots_total": int(len(bundle_df)),
        "min_tracks_for_bundle": args.min_tracks_for_bundle,
        "city_summary": city_summary,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]], limit: Optional[int] = None) -> str:
    if limit is not None:
        rows = rows[:limit]
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


def write_html_report(
    output_path: Path,
    summary: Mapping[str, Any],
    cluster_df: pd.DataFrame,
    bundle_df: pd.DataFrame,
) -> None:
    city_cards = []
    for row in summary["city_summary"]:
        city_cards.append(
            f"""
            <article class='metric-card'>
              <h3>{html.escape(row['city'])} <span>{html.escape(row['location'])}</span></h3>
              <b>{row['matched_tracks']:,}</b><small>matched tracks</small>
              <b>{row['unmatched_tracks']:,}</b><small>unmatched tracks</small>
              <b>{row['match_rate']:.1%}</b><small>match rate</small>
              <b>{(row['mean_track_var_lateral_offset_m2'] or 0):.3f}</b><small>mean sigma_lat^2</small>
            </article>
            """
        )

    top_clusters = []
    if not cluster_df.empty:
        top_clusters = cluster_df.sort_values("mean_track_var_lateral_offset_m2", ascending=False).head(30).to_dict("records")
    cluster_table = _html_table(
        top_clusters,
        [
            ("city", "City"),
            ("maneuver", "Turn"),
            ("reference_lane_id", "Lane"),
            ("matched_tracks", "Tracks"),
            ("mean_track_var_lateral_offset_m2", "Mean sigma_lat^2"),
            ("mean_abs_lateral_offset_m", "Mean |offset|"),
            ("p95_abs_lateral_offset_m", "P95 |offset|"),
            ("mean_match_distance_m", "Match dist"),
            ("match_rate_same_city_maneuver", "City-turn match rate"),
        ],
    )

    bundle_cards = []
    if not bundle_df.empty:
        for row in bundle_df.sort_values(["location", "maneuver", "matched_tracks"], ascending=[True, True, False]).to_dict("records"):
            bundle_cards.append(
                f"""
                <article class='bundle-card'>
                  <h3>{html.escape(row['city'])} {html.escape(row['maneuver'])}</h3>
                  <p>Lane <code>{html.escape(str(row['reference_lane_id']))}</code>, tracks={int(row['matched_tracks'])}, sigma_lat^2={float(row['mean_track_var_lateral_offset_m2']):.3f} m^2</p>
                  <a href='{html.escape(row['bundle_png'])}'><img src='{html.escape(row['bundle_png'])}' alt='trajectory bundle'></a>
                </article>
                """
            )

    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Lateral Deviation Variance</title>
  <style>
    :root {{ --ink:#1f2722; --paper:#f3efe5; --card:#fffaf0; --muted:#65736b; --line:#d7c8b2; --accent:#9d3f2f; --green:#2f5d50; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; background:linear-gradient(120deg,#f3efe5,#e5eee8); color:var(--ink); }}
    header {{ padding:44px 5vw 70px; color:#fff; background:radial-gradient(circle at 85% 20%,#d79d58,transparent 24%), linear-gradient(135deg,#17342e,#526f62); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); letter-spacing:-1px; }}
    header p {{ max-width:980px; line-height:1.65; color:#edf4ef; font-size:17px; }}
    main {{ padding:0 5vw 56px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:-42px; }}
    .metric-card,.bundle-card {{ background:var(--card); border:1px solid var(--line); border-radius:20px; box-shadow:0 14px 30px rgba(31,39,34,.09); padding:18px; }}
    .metric-card h3,.bundle-card h3 {{ margin:0 0 10px; }} .metric-card h3 span {{ color:var(--muted); font-size:13px; }}
    .metric-card b {{ display:block; color:var(--accent); font-size:25px; margin-top:8px; }} .metric-card small {{ color:var(--muted); }}
    section {{ margin-top:34px; }} h2 {{ font-size:30px; margin-bottom:12px; }}
    .bundle-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:18px; }}
    img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,240,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #e8dccb; text-align:left; font-size:14px; }} th {{ background:#eadcc7; }} tr:hover td {{ background:#fff4dd; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fffaf0; border:1px solid var(--line); border-radius:999px; color:var(--green); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>SinD 六路口横向偏移轨迹方差</h1>
  <p>车辆左/右转轨迹投影到 Lanelet2 中心参考线，计算 lateral offset 的均值、方差 sigma_lat^2 和 ±2sigma trajectory bundle。匹配阈值：{summary['max_match_distance_m']} m。</p>
</header>
<main>
  <div class='cards'>{''.join(city_cards)}</div>
  <section>
    <h2>High-Variance Turn Clusters</h2>
    {cluster_table}
  </section>
  <section>
    <h2>Trajectory Bundles</h2>
    <div class='bundle-grid'>{''.join(bundle_cards)}</div>
  </section>
  <section class='links'>
    <h2>Artifacts</h2>
    <a href='summary.json'>summary.json</a>
    <a href='track_lateral_deviation.csv'>track_lateral_deviation.csv</a>
    <a href='cluster_lateral_deviation.csv'>cluster_lateral_deviation.csv</a>
    <a href='unmatched_tracks.csv'>unmatched_tracks.csv</a>
    <a href='bundle_manifest.csv'>bundle_manifest.csv</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SinD lateral deviation variance for turning vehicles.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_lateral_deviation_variance"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--agent-type", default="Vehicle", help="Default: Vehicle (car/truck/bus/tricycle).")
    parser.add_argument("--maneuvers", nargs="+", default=list(TURN_MANEUVERS))
    parser.add_argument("--max-match-distance-m", type=float, default=6.0)
    parser.add_argument("--max-match-points", type=int, default=80)
    parser.add_argument("--k-nearest-lanes", type=int, default=8)
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--max-tracks-per-city", type=int, default=None)
    parser.add_argument("--top-k-per-city-maneuver", type=int, default=1)
    parser.add_argument("--min-tracks-for-bundle", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cities = [normalize_location(city) for city in args.cities]
    args.maneuvers = [normalize_maneuver_arg(maneuver) for maneuver in args.maneuvers]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracks for {args.cities}...", flush=True)
    tracks = load_sind_tracks(args.data_dir, args.cities)
    tracks = filter_tracks(tracks, city=args.cities, agent_type=args.agent_type, maneuver=None)
    maneuvers = set(args.maneuvers)
    tracks = [track for track in tracks if track.maneuver in maneuvers]
    tracks = [track for track in tracks if len(_track_points(track)) >= args.min_track_points]
    if args.max_tracks_per_city is not None:
        limited = []
        counts: Dict[str, int] = {}
        for track in tracks:
            count = counts.get(track.location, 0)
            if count >= args.max_tracks_per_city:
                continue
            limited.append(track)
            counts[track.location] = count + 1
        tracks = limited
    if not tracks:
        raise RuntimeError("No tracks matched the requested city/type/maneuver filters.")

    print("Loading Lanelet2 reference lines...", flush=True)
    references = load_lane_references(args.data_dir, args.cities)
    for city in args.cities:
        print(f"  {city}: {len(references.get(city, []))} lane references", flush=True)

    print(f"Analyzing {len(tracks)} tracks...", flush=True)
    track_df, unmatched_df, offset_groups = analyze_tracks(
        tracks,
        references,
        max_match_distance_m=args.max_match_distance_m,
        max_match_points=args.max_match_points,
        k_nearest_lanes=args.k_nearest_lanes,
    )
    cluster_df = aggregate_clusters(track_df, unmatched_df, args.cities)

    track_df.to_csv(args.output_dir / "track_lateral_deviation.csv", index=False)
    unmatched_df.to_csv(args.output_dir / "unmatched_tracks.csv", index=False)
    cluster_df.to_csv(args.output_dir / "cluster_lateral_deviation.csv", index=False)

    print("Generating trajectory bundle plots...", flush=True)
    bundle_df = generate_bundle_plots(
        cluster_df,
        offset_groups,
        references,
        args.output_dir,
        top_k_per_city_maneuver=args.top_k_per_city_maneuver,
        min_tracks=args.min_tracks_for_bundle,
    )
    bundle_df.to_csv(args.output_dir / "bundle_manifest.csv", index=False)

    summary = write_summary(args.output_dir / "summary.json", args, track_df, unmatched_df, cluster_df, bundle_df)
    write_html_report(args.output_dir / "index.html", summary, cluster_df, bundle_df)

    print(f"Matched tracks: {len(track_df)}; unmatched tracks: {len(unmatched_df)}")
    print(f"Clusters: {len(cluster_df)}; bundle plots: {len(bundle_df)}")
    print(f"Wrote outputs to: {args.output_dir}")
    if not cluster_df.empty:
        cols = ["location", "maneuver", "reference_lane_id", "matched_tracks", "mean_track_var_lateral_offset_m2"]
        print(cluster_df.sort_values("mean_track_var_lateral_offset_m2", ascending=False)[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
