#!/usr/bin/env python
"""Separate SinD human bicycles from e-bike-like two-wheelers.

The SinD labels include ``bicycle`` and ``motorcycle``.  In this project we use
``motorcycle`` as an annotated powered two-wheeler proxy, and optionally split
the ``bicycle`` label by speed clustering to catch likely e-bikes/scooters that
were annotated as bicycles.
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
from scipy.signal import savgol_filter
from shapely.geometry import Polygon

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intersection_spatiotemporal_density import filter_static_tracks, infer_intersection_core_roi  # noqa: E402
from kinematic_envelopes import LOCATION_DISPLAY, TrackRecord, load_sind_tracks, normalize_location  # noqa: E402
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS, load_lane_references  # noqa: E402


TARGET_CLASSES: Tuple[str, ...] = ("pedestrian", "bicycle", "motorcycle", "tricycle")
DISPLAY_ORDER: Tuple[str, ...] = ("pedestrian", "bicycle", "e_bike", "tricycle")
DISPLAY_LABELS: Mapping[str, str] = {
    "pedestrian": "Pedestrian",
    "bicycle": "Bicycle",
    "e_bike": "E-bike / scooter",
    "tricycle": "Tricycle",
}
DISPLAY_COLORS: Mapping[str, str] = {
    "pedestrian": "#7f8c8d",
    "bicycle": "#2e86c1",
    "e_bike": "#c0392b",
    "tricycle": "#b9770e",
}


@dataclass(frozen=True)
class TrackKinematicMetrics:
    location: str
    city: str
    scene_id: str
    agent_id: str
    original_class: str
    base_class: str
    separated_class: str
    split_source: str
    points: int
    roi_points: int
    duration_s: float
    roi_duration_s: float
    path_length_m: float
    roi_path_length_m: float
    mean_speed_mps: float
    mean_passage_speed_mps: float
    cruise_speed_mps: float
    p85_speed_mps: float
    p95_speed_mps: float
    max_startup_accel_mps2: float
    max_positive_accel_mps2: float


def _track_state(track: TrackRecord) -> pd.DataFrame:
    state = track.state.sort_values("frame_id").copy()
    if not {"x", "y"}.issubset(state.columns):
        return pd.DataFrame()
    xy = state[["x", "y"]].to_numpy(dtype=float)
    return state[np.isfinite(xy).all(axis=1)].reset_index(drop=True)


def _time_vector(state: pd.DataFrame, dt: float) -> np.ndarray:
    if "timestamp_ms" in state.columns:
        return pd.to_numeric(state["timestamp_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    if "frame_id" in state.columns:
        return pd.to_numeric(state["frame_id"], errors="coerce").to_numpy(dtype=float) * dt
    return np.arange(len(state), dtype=float) * dt


def _smooth(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) < 7:
        return values.astype(float)
    window = min(15, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 7:
        return values.astype(float)
    try:
        return savgol_filter(values.astype(float), window_length=window, polyorder=2, mode="interp")
    except Exception:
        return values.astype(float)


def _track_xy_speed_accel(track: TrackRecord, dt: float) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = _track_state(track)
    if state.empty:
        return state, np.empty((0, 2)), np.array([]), np.array([]), np.array([])
    xy = state[["x", "y"]].to_numpy(dtype=float)
    time_s = _time_vector(state, dt)
    if {"vx", "vy"}.issubset(state.columns):
        vx = pd.to_numeric(state["vx"], errors="coerce").to_numpy(dtype=float)
        vy = pd.to_numeric(state["vy"], errors="coerce").to_numpy(dtype=float)
        speed = np.hypot(vx, vy)
    elif len(xy) >= 2:
        vx = np.gradient(xy[:, 0], dt)
        vy = np.gradient(xy[:, 1], dt)
        speed = np.hypot(vx, vy)
    else:
        speed = np.zeros(len(xy), dtype=float)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    speed_s = np.clip(_smooth(speed, dt), 0.0, None)
    accel = np.gradient(speed_s, dt) if len(speed_s) >= 2 else np.zeros(len(speed_s), dtype=float)
    return state, xy, time_s, speed_s, accel


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _roi_mask(xy: np.ndarray, roi: Polygon) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    from matplotlib.path import Path as MplPath

    return MplPath(np.asarray(roi.exterior.coords, dtype=float)).contains_points(xy)


def _max_startup_accel(speed: np.ndarray, accel: np.ndarray, dt: float, moving_threshold_mps: float, window_s: float) -> float:
    moving = np.flatnonzero(speed >= moving_threshold_mps)
    if len(moving) == 0 or len(accel) == 0:
        return 0.0
    start = int(moving[0])
    end = min(len(accel), start + max(1, int(round(window_s / dt))))
    if end <= start:
        return 0.0
    return float(np.nanpercentile(np.maximum(accel[start:end], 0.0), 95))


def _initial_class(track: TrackRecord) -> str:
    if track.class_name == "motorcycle":
        return "e_bike"
    if track.class_name in {"pedestrian", "bicycle", "tricycle"}:
        return track.class_name
    return "unknown"


def _kmeans_1d(values: np.ndarray, iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int), np.array([])
    centers = np.array([np.nanpercentile(values, 30), np.nanpercentile(values, 75)], dtype=float)
    if abs(float(centers[1] - centers[0])) < 1e-6:
        return np.zeros(len(values), dtype=int), centers
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in (0, 1):
            if np.any(labels == idx):
                centers[idx] = float(np.mean(values[labels == idx]))
    order = np.argsort(centers)
    remapped = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(order):
        remapped[labels == old_idx] = new_idx
    return remapped, centers[order]


def split_bicycle_tracks(
    track_df: pd.DataFrame,
    min_cluster_tracks: int,
    min_ebike_speed_mps: float,
    min_center_gap_mps: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = track_df.copy()
    if "base_class" not in df.columns:
        df["base_class"] = df["separated_class"]
    df["separated_class"] = df["base_class"]
    df["split_source"] = np.where(df["original_class"] == "motorcycle", "annotated_motorcycle", "label")
    report: Dict[str, Any] = {
        "method": "annotated motorcycle -> e_bike; optional 1D k-means on bicycle p85 speed",
        "locations": {},
    }
    for location, group in df[df["original_class"] == "bicycle"].groupby("location"):
        idx = group.index.to_numpy()
        speeds = group["p85_speed_mps"].to_numpy(dtype=float)
        loc_report: Dict[str, Any] = {"bicycle_tracks": int(len(group))}
        if len(group) < min_cluster_tracks:
            loc_report["status"] = "skipped_not_enough_tracks"
            report["locations"][location] = loc_report
            continue
        labels, centers = _kmeans_1d(speeds)
        loc_report["centers_mps"] = centers.tolist()
        if len(centers) < 2 or centers[1] < min_ebike_speed_mps or (centers[1] - centers[0]) < min_center_gap_mps:
            loc_report["status"] = "kept_as_bicycle_cluster_gap_or_speed_too_small"
            report["locations"][location] = loc_report
            continue
        fast = labels == 1
        df.loc[idx[fast], "separated_class"] = "e_bike"
        df.loc[idx[fast], "split_source"] = "bicycle_speed_cluster_fast"
        df.loc[idx[~fast], "split_source"] = "bicycle_speed_cluster_slow"
        loc_report["status"] = "split"
        loc_report["fast_cluster_tracks"] = int(np.count_nonzero(fast))
        loc_report["slow_cluster_tracks"] = int(np.count_nonzero(~fast))
        report["locations"][location] = loc_report
    return df, report


def compute_track_metrics(
    tracks: Sequence[TrackRecord],
    roi_polygons: Mapping[str, Polygon],
    dt: float,
    moving_threshold_mps: float,
    startup_window_s: float,
    min_path_length_m: float,
) -> pd.DataFrame:
    rows: List[TrackKinematicMetrics] = []
    for track in tracks:
        if track.class_name not in TARGET_CLASSES:
            continue
        _, xy, time_s, speed, accel = _track_xy_speed_accel(track, dt)
        if len(xy) < 3:
            continue
        roi_mask = _roi_mask(xy, roi_polygons[track.location])
        use_mask = roi_mask if np.count_nonzero(roi_mask) >= 3 else np.ones(len(xy), dtype=bool)
        moving_mask = use_mask & (speed >= moving_threshold_mps)
        path_len = _path_length(xy)
        roi_path_len = _path_length(xy[use_mask])
        if path_len < min_path_length_m or not np.any(moving_mask):
            continue
        passage_speed = speed[use_mask]
        moving_speed = speed[moving_mask]
        rows.append(
            TrackKinematicMetrics(
                location=track.location,
                city=LOCATION_DISPLAY.get(track.location, track.location),
                scene_id=track.scene_id,
                agent_id=track.agent_id,
                original_class=track.class_name,
                base_class=_initial_class(track),
                separated_class=_initial_class(track),
                split_source="label",
                points=int(len(xy)),
                roi_points=int(np.count_nonzero(roi_mask)),
                duration_s=float(len(xy) * dt),
                roi_duration_s=float(np.count_nonzero(use_mask) * dt),
                path_length_m=path_len,
                roi_path_length_m=roi_path_len,
                mean_speed_mps=float(np.mean(speed)),
                mean_passage_speed_mps=float(np.mean(passage_speed)),
                cruise_speed_mps=float(np.mean(moving_speed)),
                p85_speed_mps=float(np.percentile(moving_speed, 85)),
                p95_speed_mps=float(np.percentile(moving_speed, 95)),
                max_startup_accel_mps2=_max_startup_accel(speed, accel, dt, moving_threshold_mps, startup_window_s),
                max_positive_accel_mps2=float(np.nanpercentile(np.maximum(accel[use_mask], 0.0), 95)),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def summarize(track_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if track_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg = {
        "agent_id": "count",
        "mean_passage_speed_mps": ["mean", "median"],
        "cruise_speed_mps": ["mean", "median"],
        "p85_speed_mps": "mean",
        "max_startup_accel_mps2": ["mean", "median", "max"],
        "max_positive_accel_mps2": ["mean", "median", "max"],
    }
    by_class = track_df.groupby("separated_class").agg(agg)
    by_class.columns = [
        "tracks",
        "mean_passage_speed_mps",
        "median_passage_speed_mps",
        "mean_cruise_speed_mps",
        "median_cruise_speed_mps",
        "mean_p85_speed_mps",
        "mean_startup_accel_mps2",
        "median_startup_accel_mps2",
        "max_startup_accel_mps2",
        "mean_positive_accel_mps2",
        "median_positive_accel_mps2",
        "max_positive_accel_mps2",
    ]
    by_class = by_class.reset_index()
    by_loc = track_df.groupby(["location", "city", "separated_class"]).agg(agg)
    by_loc.columns = by_class.columns[1:]
    by_loc = by_loc.reset_index()
    return by_class, by_loc


def plot_violin(track_df: pd.DataFrame, output_path: Path, speed_col: str = "mean_passage_speed_mps") -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    data = []
    labels = []
    colors = []
    for key in DISPLAY_ORDER:
        values = track_df.loc[track_df["separated_class"] == key, speed_col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            values = np.array([np.nan])
        data.append(values)
        labels.append(DISPLAY_LABELS[key])
        colors.append(DISPLAY_COLORS[key])
    positions = np.arange(1, len(data) + 1)
    valid_data = [values[np.isfinite(values)] for values in data]
    parts = ax.violinplot(valid_data, positions=positions, showmeans=True, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.55)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.0)
    rng = np.random.default_rng(20260504)
    for pos, values, color in zip(positions, valid_data, colors):
        if len(values) == 0:
            continue
        sample = values if len(values) <= 900 else rng.choice(values, size=900, replace=False)
        jitter = rng.normal(0.0, 0.045, size=len(sample))
        ax.scatter(np.full(len(sample), pos) + jitter, sample, s=6, alpha=0.22, color=color, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Intersection passage mean speed (m/s)")
    ax.set_title("SinD VRU / Two-wheeler Kinematic Heterogeneity")
    ax.grid(True, axis="y", linestyle="--", alpha=0.24)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _html_table(df: pd.DataFrame, columns: Sequence[Tuple[str, str]]) -> str:
    rows = df.to_dict("records") if not df.empty else []
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


def write_html(output_path: Path, summary_df: pd.DataFrame, location_df: pd.DataFrame, split_report: Mapping[str, Any]) -> None:
    summary_table = _html_table(
        summary_df,
        [
            ("separated_class", "Class"),
            ("tracks", "Tracks"),
            ("mean_passage_speed_mps", "Mean passage speed"),
            ("median_passage_speed_mps", "Median passage speed"),
            ("mean_cruise_speed_mps", "Mean cruise speed"),
            ("mean_startup_accel_mps2", "Mean startup accel"),
            ("max_startup_accel_mps2", "Max startup accel"),
        ],
    )
    location_table = _html_table(
        location_df,
        [
            ("location", "Location"),
            ("city", "City"),
            ("separated_class", "Class"),
            ("tracks", "Tracks"),
            ("mean_passage_speed_mps", "Mean passage speed"),
            ("mean_startup_accel_mps2", "Mean startup accel"),
        ],
    )
    report_text = html.escape(json.dumps(split_report, ensure_ascii=False, indent=2))
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SinD E-bike vs Bicycle Separation</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; color:#1d2528; background:#f5f1e9; }}
    header {{ padding:34px 5vw; background:#263238; color:white; }}
    main {{ padding:24px 5vw 48px; }}
    .panel {{ background:#fffaf1; border:1px solid #d7c7aa; padding:18px; margin:18px 0; overflow:auto; }}
    img {{ max-width:100%; border:1px solid #d7c7aa; background:white; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #d7c7aa; padding:8px 9px; text-align:left; }}
    th {{ background:#efe2cc; }}
    pre {{ white-space:pre-wrap; font-size:12px; background:#f7efe1; padding:12px; }}
  </style>
</head>
<body>
  <header>
    <h1>SinD 两轮车动力学异质性分离</h1>
    <p>将行人、自行车、电动二轮车/电摩、三轮车的路口通行速度和起步加速度分开统计。</p>
  </header>
  <main>
    <section class="panel"><h2>Violin Plot</h2><a href="passage_speed_violin.png"><img src="passage_speed_violin.png" alt="violin plot"></a></section>
    <section class="panel"><h2>Class Summary</h2>{summary_table}</section>
    <section class="panel"><h2>Location Summary</h2>{location_table}</section>
    <section class="panel"><h2>Bicycle Split Report</h2><pre>{report_text}</pre></section>
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
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_two_wheeler_heterogeneity"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--moving-threshold-mps", type=float, default=0.5)
    parser.add_argument("--startup-window-s", type=float, default=3.0)
    parser.add_argument("--min-path-length-m", type=float, default=2.0)
    parser.add_argument("--min-bicycle-cluster-tracks", type=int, default=30)
    parser.add_argument("--min-ebike-speed-mps", type=float, default=3.5)
    parser.add_argument("--min-cluster-gap-mps", type=float, default=1.0)
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
    tracks = [track for track in all_tracks if track.class_name in TARGET_CLASSES]
    print(f"Tracks after static filtering: target={len(tracks)}, static_removed={len(static_removed_df)}", flush=True)

    print("Inferring intersection core ROIs...", flush=True)
    references = load_lane_references(args.data_dir, locations)
    roi_polygons: Dict[str, Polygon] = {}
    roi_debug_rows: List[Dict[str, Any]] = []
    for location in locations:
        roi, debug = infer_intersection_core_roi(references[location])
        roi_polygons[location] = roi
        roi_debug_rows.append({"location": location, "city": LOCATION_DISPLAY.get(location, location), **debug})

    print("Computing per-track kinematics...", flush=True)
    track_df = compute_track_metrics(
        tracks,
        roi_polygons=roi_polygons,
        dt=args.dt,
        moving_threshold_mps=args.moving_threshold_mps,
        startup_window_s=args.startup_window_s,
        min_path_length_m=args.min_path_length_m,
    )
    track_df, split_report = split_bicycle_tracks(
        track_df,
        min_cluster_tracks=args.min_bicycle_cluster_tracks,
        min_ebike_speed_mps=args.min_ebike_speed_mps,
        min_center_gap_mps=args.min_cluster_gap_mps,
    )
    summary_df, location_df = summarize(track_df)

    print("Writing outputs...", flush=True)
    track_df.to_csv(args.output_dir / "two_wheeler_track_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "two_wheeler_summary_by_class.csv", index=False)
    location_df.to_csv(args.output_dir / "two_wheeler_summary_by_location.csv", index=False)
    static_removed_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)
    pd.DataFrame(roi_debug_rows).to_csv(args.output_dir / "roi_debug.csv", index=False)
    (args.output_dir / "bicycle_split_report.json").write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_violin(track_df, args.output_dir / "passage_speed_violin.png")
    write_html(args.output_dir / "index.html", summary_df, location_df, split_report)
    print(f"Done. Open {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
