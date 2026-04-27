#!/usr/bin/env python
"""Compute SinD intersection-internal spatiotemporal vehicle density.

The script bins vehicle center positions into a 1m x 1m grid and accumulates the
time spent in each cell, clipped to an inferred intersection-core ROI so upstream
and downstream straight-road queues are excluded from the heatmap.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
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
)

VEHICLE_AGENT_TYPE = "Vehicle"
DIRECTION_OPPOSITES = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _heading_change_deg(xy: np.ndarray) -> float:
    if len(xy) < 4:
        return 0.0
    start_idx = min(3, len(xy) - 1)
    end_idx = max(0, len(xy) - 4)
    v0 = xy[start_idx] - xy[0]
    v1 = xy[-1] - xy[end_idx]
    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
        return 0.0
    a0 = math.atan2(v0[1], v0[0])
    a1 = math.atan2(v1[1], v1[0])
    delta = math.degrees(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)))
    return abs(delta)


def _directions_from_lane_id(lane_id: str) -> Optional[Tuple[str, str]]:
    upper = lane_id.upper().replace("-", "_")
    if "_TO_" not in upper:
        return None
    src_raw, dst_raw = upper.split("_TO_", 1)
    src = src_raw[0] if src_raw and src_raw[0] in DIRECTION_OPPOSITES else None
    dst = dst_raw[0] if dst_raw and dst_raw[0] in DIRECTION_OPPOSITES else None
    if src is None or dst is None:
        return None
    return src, dst


def _is_internal_connector_lane(lane: LaneReference) -> bool:
    upper = lane.lane_id.upper().replace("-", "_")
    return "_TO_" in upper or " TO " in upper


def _is_turning_or_conflict_lane(lane: LaneReference, turn_heading_threshold_deg: float) -> bool:
    if _is_internal_connector_lane(lane):
        return True
    return _heading_change_deg(lane.xy) >= turn_heading_threshold_deg


def _trim_lane_points(lane: LaneReference, trim_ratio: float) -> np.ndarray:
    if len(lane.xy) < 3 or lane.length <= 0:
        return lane.xy
    lo = lane.length * trim_ratio
    hi = lane.length * (1.0 - trim_ratio)
    mask = (lane.cumulative_s >= lo) & (lane.cumulative_s <= hi)
    trimmed = lane.xy[mask]
    return trimmed if len(trimmed) >= 2 else lane.xy


def _polygon_to_mpl_path(polygon: Polygon) -> MplPath:
    return MplPath(np.asarray(polygon.exterior.coords, dtype=float))


def _points_inside_polygon(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    path = _polygon_to_mpl_path(polygon)
    return path.contains_points(points)


def infer_intersection_core_roi(
    lane_refs: Sequence[LaneReference],
    core_buffer_m: float = 3.0,
    central_quantile: float = 0.62,
    max_core_radius_m: float = 32.0,
    min_core_radius_m: float = 12.0,
    turn_heading_threshold_deg: float = 25.0,
    lane_trim_ratio: float = 0.20,
) -> Tuple[Polygon, Dict[str, Any]]:
    """Infer a compact intersection-core ROI from Lanelet2 centerlines.

    Internal connector lanes seed the ROI, including turn and through movements
    whose ids contain ``to``. Approach/exit lane stubs do not contribute, so
    upstream/downstream stop-line queues are clipped out more aggressively.
    """
    all_points = np.vstack([lane.xy for lane in lane_refs if len(lane.xy)])
    rough_center = np.median(all_points, axis=0)

    seed_lanes = [lane for lane in lane_refs if _is_turning_or_conflict_lane(lane, turn_heading_threshold_deg)]
    if seed_lanes:
        seed_points = np.vstack([_trim_lane_points(lane, lane_trim_ratio) for lane in seed_lanes if len(lane.xy)])
        center = np.median(seed_points, axis=0)
    else:
        distances = np.linalg.norm(all_points - rough_center[None, :], axis=1)
        seed_points = all_points[distances <= np.quantile(distances, 0.55)]
        center = np.median(seed_points, axis=0)

    seed_dist = np.linalg.norm(seed_points - center[None, :], axis=1)
    auto_radius = float(np.quantile(seed_dist, central_quantile) + core_buffer_m)
    core_radius = min(max(auto_radius, min_core_radius_m), max_core_radius_m)

    core_points = seed_points[seed_dist <= core_radius]
    if len(core_points) < 6:
        core_points = seed_points
    polygon = MultiPoint(core_points).convex_hull.buffer(core_buffer_m)
    if polygon.geom_type != "Polygon":
        polygon = polygon.envelope.buffer(core_buffer_m)

    debug = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "core_radius_m": float(core_radius),
        "seed_lane_count": int(len(seed_lanes)),
        "lane_count": int(len(lane_refs)),
        "central_quantile": float(central_quantile),
        "roi_area_m2": float(polygon.area),
        "lane_trim_ratio": float(lane_trim_ratio),
    }
    return polygon, debug


def _track_xy(track: TrackRecord) -> np.ndarray:
    state = track.state.sort_values("frame_id")
    if not {"x", "y"}.issubset(state.columns):
        return np.empty((0, 2), dtype=float)
    xy = state[["x", "y"]].to_numpy(dtype=float)
    return xy[np.isfinite(xy).all(axis=1)]




def _track_motion_stats(track: TrackRecord, dt: float) -> Dict[str, float]:
    state = track.state.sort_values("frame_id")
    xy = _track_xy(track)
    if len(xy) < 2:
        return {
            "duration_s": float(len(xy) * dt),
            "net_displacement_m": 0.0,
            "path_length_m": 0.0,
            "p95_speed_mps": 0.0,
            "mean_path_speed_mps": 0.0,
        }

    step_dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_length = float(np.sum(step_dist))
    net_displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    duration = float(len(xy) * dt)

    if {"vx", "vy"}.issubset(state.columns):
        speed = np.hypot(state["vx"].to_numpy(dtype=float), state["vy"].to_numpy(dtype=float))
        speed = speed[np.isfinite(speed)]
    else:
        speed = step_dist / max(dt, 1e-6)
    p95_speed = float(np.quantile(speed, 0.95)) if len(speed) else 0.0
    return {
        "duration_s": duration,
        "net_displacement_m": net_displacement,
        "path_length_m": path_length,
        "p95_speed_mps": p95_speed,
        "mean_path_speed_mps": float(path_length / duration) if duration > 0 else 0.0,
    }


def _is_static_track(
    track: TrackRecord,
    dt: float,
    static_min_duration_s: float,
    static_max_displacement_m: float,
    static_max_path_length_m: float,
    static_max_speed_p95_mps: float,
    static_min_slow_duration_s: float,
    static_max_mean_path_speed_mps: float,
) -> Tuple[bool, Dict[str, float]]:
    stats = _track_motion_stats(track, dt)
    if stats["duration_s"] < static_min_duration_s:
        return False, stats
    nearly_fixed = (
        stats["net_displacement_m"] <= static_max_displacement_m
        and stats["path_length_m"] <= static_max_path_length_m
    )
    almost_never_moves = stats["p95_speed_mps"] <= static_max_speed_p95_mps
    slow_creeping_queue = (
        stats["duration_s"] >= static_min_slow_duration_s
        and stats["mean_path_speed_mps"] <= static_max_mean_path_speed_mps
    )
    return bool(nearly_fixed or almost_never_moves or slow_creeping_queue), stats


def filter_static_tracks(
    tracks: Sequence[TrackRecord],
    dt: float,
    enabled: bool,
    static_min_duration_s: float,
    static_max_displacement_m: float,
    static_max_path_length_m: float,
    static_max_speed_p95_mps: float,
    static_min_slow_duration_s: float,
    static_max_mean_path_speed_mps: float,
) -> Tuple[List[TrackRecord], pd.DataFrame]:
    if not enabled:
        return list(tracks), pd.DataFrame()
    kept: List[TrackRecord] = []
    removed_rows = []
    for track in tracks:
        is_static, stats = _is_static_track(
            track,
            dt=dt,
            static_min_duration_s=static_min_duration_s,
            static_max_displacement_m=static_max_displacement_m,
            static_max_path_length_m=static_max_path_length_m,
            static_max_speed_p95_mps=static_max_speed_p95_mps,
            static_min_slow_duration_s=static_min_slow_duration_s,
            static_max_mean_path_speed_mps=static_max_mean_path_speed_mps,
        )
        if is_static:
            removed_rows.append(
                {
                    "location": track.location,
                    "city": LOCATION_DISPLAY.get(track.location, track.location),
                    "scene_id": track.scene_id,
                    "agent_id": track.agent_id,
                    "class_name": track.class_name,
                    **stats,
                }
            )
        else:
            kept.append(track)
    return kept, pd.DataFrame(removed_rows)

def _limit_tracks_per_city(tracks: Sequence[TrackRecord], max_tracks_per_city: Optional[int]) -> List[TrackRecord]:
    if max_tracks_per_city is None:
        return list(tracks)
    limited: List[TrackRecord] = []
    counts: Dict[str, int] = {}
    for track in tracks:
        count = counts.get(track.location, 0)
        if count >= max_tracks_per_city:
            continue
        limited.append(track)
        counts[track.location] = count + 1
    return limited


def compute_city_occupancy(
    location: str,
    tracks: Sequence[TrackRecord],
    lane_refs: Sequence[LaneReference],
    grid_size_m: float,
    dt: float,
    core_buffer_m: float,
    max_core_radius_m: float,
    min_core_radius_m: float,
    turn_heading_threshold_deg: float,
    lane_trim_ratio: float,
    central_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Polygon, Dict[str, Any], pd.DataFrame]:
    roi_polygon, roi_debug = infer_intersection_core_roi(
        lane_refs,
        core_buffer_m=core_buffer_m,
        max_core_radius_m=max_core_radius_m,
        min_core_radius_m=min_core_radius_m,
        turn_heading_threshold_deg=turn_heading_threshold_deg,
        lane_trim_ratio=lane_trim_ratio,
        central_quantile=central_quantile,
    )
    minx, miny, maxx, maxy = roi_polygon.bounds
    x_edges = np.arange(math.floor(minx), math.ceil(maxx) + grid_size_m, grid_size_m, dtype=float)
    y_edges = np.arange(math.floor(miny), math.ceil(maxy) + grid_size_m, grid_size_m, dtype=float)
    grid = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    total_points = 0
    in_roi_points = 0
    track_rows = []
    path = _polygon_to_mpl_path(roi_polygon)
    for track in tracks:
        xy = _track_xy(track)
        if len(xy) == 0:
            continue
        total_points += len(xy)
        mask = path.contains_points(xy)
        xy_in = xy[mask]
        in_roi_points += len(xy_in)
        track_rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "scene_id": track.scene_id,
                "agent_id": track.agent_id,
                "class_name": track.class_name,
                "points_total": int(len(xy)),
                "points_in_roi": int(len(xy_in)),
                "seconds_in_roi": float(len(xy_in) * dt),
            }
        )
        if len(xy_in) == 0:
            continue
        ix = np.floor((xy_in[:, 0] - x_edges[0]) / grid_size_m).astype(int)
        iy = np.floor((xy_in[:, 1] - y_edges[0]) / grid_size_m).astype(int)
        valid = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
        np.add.at(grid, (iy[valid], ix[valid]), dt)

    occupied_cells = int(np.count_nonzero(grid))
    summary = {
        "location": location,
        "city": LOCATION_DISPLAY.get(location, location),
        "vehicle_tracks": int(len(tracks)),
        "state_points_total": int(total_points),
        "state_points_in_roi": int(in_roi_points),
        "seconds_in_roi_total": float(grid.sum()),
        "max_cell_seconds": float(grid.max()) if grid.size else 0.0,
        "mean_occupied_cell_seconds": float(grid[grid > 0].mean()) if occupied_cells else 0.0,
        "occupied_cells": occupied_cells,
        "grid_width": int(grid.shape[1]),
        "grid_height": int(grid.shape[0]),
        "grid_size_m": float(grid_size_m),
        **roi_debug,
    }
    return grid, x_edges, y_edges, roi_polygon, summary, pd.DataFrame(track_rows)


def extract_hotspots(
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    location: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if grid.size == 0:
        return []
    flat = grid.ravel()
    positive = np.flatnonzero(flat > 0)
    if len(positive) == 0:
        return []
    order = positive[np.argsort(flat[positive])[::-1]][:top_k]
    rows = []
    for rank, flat_idx in enumerate(order, start=1):
        iy, ix = np.unravel_index(int(flat_idx), grid.shape)
        rows.append(
            {
                "location": location,
                "city": LOCATION_DISPLAY.get(location, location),
                "rank": rank,
                "x_center_m": float((x_edges[ix] + x_edges[ix + 1]) / 2.0),
                "y_center_m": float((y_edges[iy] + y_edges[iy + 1]) / 2.0),
                "occupancy_seconds": float(grid[iy, ix]),
                "cell_ix": int(ix),
                "cell_iy": int(iy),
            }
        )
    return rows


def plot_city_heatmap(
    location: str,
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    roi_polygon: Polygon,
    lane_refs: Sequence[LaneReference],
    output_path: Path,
    vmax_quantile: float,
) -> None:
    masked = np.ma.masked_where(grid <= 0, grid)
    vmax = float(np.quantile(grid[grid > 0], vmax_quantile)) if np.any(grid > 0) else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    im = ax.imshow(
        masked,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap="Reds",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        alpha=0.92,
    )
    for lane in lane_refs:
        ax.plot(lane.xy[:, 0], lane.xy[:, 1], color="#344c5a", linewidth=0.55, alpha=0.35)
    roi_xy = np.asarray(roi_polygon.exterior.coords)
    ax.plot(roi_xy[:, 0], roi_xy[:, 1], color="#111111", linewidth=2.0, label="Intersection core ROI")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{LOCATION_DISPLAY.get(location, location)} Intersection Occupancy Heatmap")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vehicle occupancy time (s per 1m cell)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_roi_geojson(output_path: Path, location: str, polygon: Polygon, properties: Mapping[str, Any]) -> None:
    feature = {
        "type": "Feature",
        "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), **dict(properties)},
        "geometry": mapping(polygon),
    }
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2), encoding="utf-8")


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


def write_html_report(output_path: Path, summary_rows: Sequence[Mapping[str, Any]], hotspot_rows: Sequence[Mapping[str, Any]]) -> None:
    total_seconds = sum(float(row["seconds_in_roi_total"]) for row in summary_rows)
    total_tracks = sum(int(row["vehicle_tracks"]) for row in summary_rows)
    city_cards = []
    for row in summary_rows:
        loc = row["location"]
        city_cards.append(
            f"""
            <article class='city-card'>
              <h3>{html.escape(str(row['city']))} <span>{html.escape(str(loc))}</span></h3>
              <b>{int(row['vehicle_tracks']):,}</b><small>vehicle tracks</small>
              <b>{float(row['seconds_in_roi_total']):,.1f}s</b><small>occupancy seconds</small>
              <b>{float(row['max_cell_seconds']):.1f}s</b><small>max cell seconds</small>
              <b>{int(row['occupied_cells']):,}</b><small>occupied cells</small>
              <a href='occupancy_heatmap_{html.escape(str(loc))}.png'><img src='occupancy_heatmap_{html.escape(str(loc))}.png' alt='heatmap'></a>
            </article>
            """
        )
    summary_table = _html_table(
        summary_rows,
        [
            ("city", "City"),
            ("vehicle_tracks", "Tracks"),
            ("seconds_in_roi_total", "Occupancy s"),
            ("max_cell_seconds", "Max cell s"),
            ("mean_occupied_cell_seconds", "Mean occupied cell s"),
            ("occupied_cells", "Occupied cells"),
            ("roi_area_m2", "ROI area m2"),
            ("core_radius_m", "Core radius m"),
        ],
    )
    hotspot_table = _html_table(
        list(hotspot_rows)[:60],
        [
            ("city", "City"),
            ("rank", "Rank"),
            ("x_center_m", "x"),
            ("y_center_m", "y"),
            ("occupancy_seconds", "Occupancy s"),
        ],
    )
    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Intersection Spatiotemporal Density</title>
  <style>
    :root {{ --ink:#201b17; --paper:#f6efe4; --card:#fff9ee; --line:#dcc9ac; --muted:#71675d; --red:#b83a26; --blue:#254d5f; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f6efe4,#e9eee9); }}
    header {{ padding:44px 5vw 72px; color:#fff; background:radial-gradient(circle at 85% 15%,#e0a34f,transparent 24%), linear-gradient(135deg,#3b1f18,#6b392d); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); letter-spacing:-1px; }}
    header p {{ max-width:980px; line-height:1.65; color:#f7e9dc; font-size:17px; }}
    main {{ padding:0 5vw 58px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:-42px; }}
    .kpi,.city-card {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 30px rgba(32,27,23,.08); }}
    .kpi b,.city-card b {{ display:block; color:var(--red); font-size:25px; margin-top:8px; }} .kpi small,.city-card small {{ color:var(--muted); }}
    section {{ margin-top:34px; }} h2 {{ font-size:30px; margin-bottom:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:18px; }}
    .city-card h3 {{ margin:0 0 10px; }} .city-card h3 span {{ color:var(--muted); font-size:13px; }}
    img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,249,238,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #eadcc8; text-align:left; font-size:14px; }} th {{ background:#ead6bd; }} tr:hover td {{ background:#fff0d8; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fff9ee; border:1px solid var(--line); border-radius:999px; color:var(--blue); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>SinD 路口内部时空密度分布</h1>
  <p>车辆中心点按 1m x 1m 网格累计驻留时间，并裁剪到 Lanelet2 推断出的路口内部 ROI。红色越强，表示车辆在该路口内部区域滞留、博弈或反复占用的时间越长。</p>
</header>
<main>
  <div class='kpis'>
    <div class='kpi'><b>{len(summary_rows)}</b><small>intersections</small></div>
    <div class='kpi'><b>{total_tracks:,}</b><small>vehicle tracks</small></div>
    <div class='kpi'><b>{total_seconds:,.1f}s</b><small>total occupancy inside ROI</small></div>
  </div>
  <section>
    <h2>Occupancy Heatmaps</h2>
    <div class='grid'>{''.join(city_cards)}</div>
  </section>
  <section>
    <h2>Intersection Density Summary</h2>
    {summary_table}
  </section>
  <section>
    <h2>Top Hotspot Cells</h2>
    {hotspot_table}
  </section>
  <section class='links'>
    <h2>Artifacts</h2>
    <a href='summary.json'>summary.json</a>
    <a href='density_summary.csv'>density_summary.csv</a>
    <a href='hotspot_cells.csv'>hotspot_cells.csv</a>
    <a href='track_roi_residence.csv'>track_roi_residence.csv</a>
    <a href='filtered_static_tracks.csv'>filtered_static_tracks.csv</a>
    <a href='occupancy_grid_cc.npz'>occupancy_grid_*.npz</a>
    <a href='core_roi_cc.geojson'>core_roi_*.geojson</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SinD intersection-internal spatiotemporal density heatmaps.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_intersection_spatiotemporal_density"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--agent-type", default=VEHICLE_AGENT_TYPE)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--grid-size-m", type=float, default=1.0)
    parser.add_argument("--core-buffer-m", type=float, default=3.0)
    parser.add_argument("--min-core-radius-m", type=float, default=12.0)
    parser.add_argument("--max-core-radius-m", type=float, default=32.0)
    parser.add_argument("--turn-heading-threshold-deg", type=float, default=25.0)
    parser.add_argument("--lane-trim-ratio", type=float, default=0.20)
    parser.add_argument("--central-quantile", type=float, default=0.62)
    parser.add_argument("--keep-static-tracks", action="store_true", help="Disable filtering of long-duration static/parked vehicles.")
    parser.add_argument("--static-min-duration-s", type=float, default=8.0)
    parser.add_argument("--static-max-displacement-m", type=float, default=2.0)
    parser.add_argument("--static-max-path-length-m", type=float, default=5.0)
    parser.add_argument("--static-max-speed-p95-mps", type=float, default=0.15)
    parser.add_argument("--static-min-slow-duration-s", type=float, default=60.0)
    parser.add_argument("--static-max-mean-path-speed-mps", type=float, default=0.25)
    parser.add_argument("--hotspot-top-k", type=int, default=30)
    parser.add_argument("--max-tracks-per-city", type=int, default=None)
    parser.add_argument("--vmax-quantile", type=float, default=0.995)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cities = [normalize_location(city) for city in args.cities]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vehicle tracks for {args.cities}...", flush=True)
    tracks = load_sind_tracks(args.data_dir, args.cities)
    tracks = filter_tracks(tracks, city=args.cities, agent_type=args.agent_type, maneuver=None)
    tracks = _limit_tracks_per_city(tracks, args.max_tracks_per_city)
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
    tracks_by_city: Dict[str, List[TrackRecord]] = {city: [] for city in args.cities}
    for track in tracks:
        tracks_by_city.setdefault(track.location, []).append(track)

    if not static_tracks_df.empty:
        print(f"Filtered {len(static_tracks_df)} long-duration static tracks before occupancy accumulation.", flush=True)
    print("Loading Lanelet2 centerlines...", flush=True)
    refs_by_city = load_lane_references(args.data_dir, args.cities)

    summary_rows = []
    hotspot_rows = []
    track_residence_frames = []
    for city in args.cities:
        print(f"Computing occupancy for {city}: {len(tracks_by_city.get(city, []))} tracks", flush=True)
        grid, x_edges, y_edges, roi_polygon, summary, track_residence = compute_city_occupancy(
            city,
            tracks_by_city.get(city, []),
            refs_by_city.get(city, []),
            grid_size_m=args.grid_size_m,
            dt=args.dt,
            core_buffer_m=args.core_buffer_m,
            max_core_radius_m=args.max_core_radius_m,
            min_core_radius_m=args.min_core_radius_m,
            turn_heading_threshold_deg=args.turn_heading_threshold_deg,
            lane_trim_ratio=args.lane_trim_ratio,
            central_quantile=args.central_quantile,
        )
        summary_rows.append(summary)
        city_hotspots = extract_hotspots(grid, x_edges, y_edges, city, args.hotspot_top_k)
        hotspot_rows.extend(city_hotspots)
        if not track_residence.empty:
            track_residence_frames.append(track_residence)

        np.savez_compressed(
            args.output_dir / f"occupancy_grid_{city}.npz",
            grid=grid,
            x_edges=x_edges,
            y_edges=y_edges,
            roi_polygon_xy=np.asarray(roi_polygon.exterior.coords, dtype=float),
            grid_size_m=float(args.grid_size_m),
            dt=float(args.dt),
        )
        write_roi_geojson(args.output_dir / f"core_roi_{city}.geojson", city, roi_polygon, summary)
        plot_city_heatmap(
            city,
            grid,
            x_edges,
            y_edges,
            roi_polygon,
            refs_by_city.get(city, []),
            args.output_dir / f"occupancy_heatmap_{city}.png",
            vmax_quantile=args.vmax_quantile,
        )

    density_df = pd.DataFrame(summary_rows)
    hotspots_df = pd.DataFrame(hotspot_rows)
    residence_df = pd.concat(track_residence_frames, ignore_index=True) if track_residence_frames else pd.DataFrame()
    density_df.to_csv(args.output_dir / "density_summary.csv", index=False)
    hotspots_df.to_csv(args.output_dir / "hotspot_cells.csv", index=False)
    residence_df.to_csv(args.output_dir / "track_roi_residence.csv", index=False)
    static_tracks_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)

    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "agent_type": args.agent_type,
        "grid_size_m": args.grid_size_m,
        "dt": args.dt,
        "core_buffer_m": args.core_buffer_m,
        "min_core_radius_m": args.min_core_radius_m,
        "max_core_radius_m": args.max_core_radius_m,
        "lane_trim_ratio": args.lane_trim_ratio,
        "central_quantile": args.central_quantile,
        "static_filter_enabled": not args.keep_static_tracks,
        "filtered_static_tracks_total": int(len(static_tracks_df)),
        "static_filter": {
            "static_min_duration_s": args.static_min_duration_s,
            "static_max_displacement_m": args.static_max_displacement_m,
            "static_max_path_length_m": args.static_max_path_length_m,
            "static_max_speed_p95_mps": args.static_max_speed_p95_mps,
            "static_min_slow_duration_s": args.static_min_slow_duration_s,
            "static_max_mean_path_speed_mps": args.static_max_mean_path_speed_mps,
        },
        "summary_by_city": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html_report(args.output_dir / "index.html", summary_rows, hotspot_rows)

    print(f"Wrote outputs to: {args.output_dir}")
    print(density_df[["location", "vehicle_tracks", "seconds_in_roi_total", "max_cell_seconds", "occupied_cells", "roi_area_m2"]].to_string(index=False))


if __name__ == "__main__":
    main()
