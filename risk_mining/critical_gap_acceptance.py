#!/usr/bin/env python
"""Compute UPLT critical gap acceptance metrics for SinD.

The script treats vehicle left-turn tracks as ego candidates and same-scene
straight vehicle tracks as opposing conflict vehicles. For each ego, it measures
the accepted time/distance gap at the ego's first entry into the inferred
intersection-core ROI.
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
from matplotlib.path import Path as MplPath

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
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS, load_lane_references  # noqa: E402
from intersection_spatiotemporal_density import infer_intersection_core_roi  # noqa: E402

VEHICLE_AGENT_TYPE = "Vehicle"


@dataclass
class TrackCache:
    track: TrackRecord
    xy: np.ndarray
    frames: np.ndarray
    heading: Optional[float]
    median_speed_mps: float
    cumulative_s: np.ndarray


def _track_df(track: TrackRecord) -> pd.DataFrame:
    return track.state.sort_values("frame_id").reset_index(drop=True)


def _track_xy(track: TrackRecord) -> np.ndarray:
    df = _track_df(track)
    if not {"x", "y"}.issubset(df.columns):
        return np.empty((0, 2), dtype=float)
    xy = df[["x", "y"]].to_numpy(dtype=float)
    return xy[np.isfinite(xy).all(axis=1)]


def _track_xy_frames(track: TrackRecord) -> Tuple[np.ndarray, np.ndarray]:
    df = _track_df(track)
    if not {"x", "y", "frame_id"}.issubset(df.columns):
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
    xy = df[["x", "y"]].to_numpy(dtype=float)
    frames = df["frame_id"].to_numpy(dtype=int)
    mask = np.isfinite(xy).all(axis=1)
    return xy[mask], frames[mask]


def _frame_ids(track: TrackRecord) -> np.ndarray:
    df = _track_df(track)
    if "frame_id" not in df.columns:
        return np.arange(len(df), dtype=int)
    return df["frame_id"].to_numpy(dtype=int)


def _speeds(track: TrackRecord, dt: float) -> np.ndarray:
    df = _track_df(track)
    if {"vx", "vy"}.issubset(df.columns):
        speed = np.hypot(df["vx"].to_numpy(dtype=float), df["vy"].to_numpy(dtype=float))
        return speed[np.isfinite(speed)]
    xy = _track_xy(track)
    if len(xy) < 2:
        return np.array([], dtype=float)
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / max(dt, 1e-6)


def _speeds_from_xy(track: TrackRecord, xy: np.ndarray, dt: float) -> np.ndarray:
    df = _track_df(track)
    if {"vx", "vy"}.issubset(df.columns):
        speed = np.hypot(df["vx"].to_numpy(dtype=float), df["vy"].to_numpy(dtype=float))
        speed = speed[np.isfinite(speed)]
        if len(speed):
            return speed
    if len(xy) < 2:
        return np.array([], dtype=float)
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / max(dt, 1e-6)


def _trajectory_heading_xy(xy: np.ndarray) -> Optional[float]:
    if len(xy) < 5:
        return None
    span = min(10, max(2, len(xy) // 5))
    vec = xy[-1] - xy[0]
    if np.linalg.norm(vec) < 1.0:
        vec = xy[-1] - xy[max(0, len(xy) - span - 1)]
    if np.linalg.norm(vec) < 1e-6:
        return None
    return float(math.atan2(vec[1], vec[0]))


def _trajectory_heading(track: TrackRecord) -> Optional[float]:
    return _trajectory_heading_xy(_track_xy(track))


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def _path_cumulative(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.array([], dtype=float)
    if len(xy) == 1:
        return np.array([0.0], dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def _first_roi_entry_index(track: TrackRecord, roi_path: MplPath) -> Optional[int]:
    xy = _track_xy(track)
    if len(xy) == 0:
        return None
    inside = roi_path.contains_points(xy)
    if not inside.any():
        return None
    first = int(np.argmax(inside))
    # If the first visible sample is already inside the ROI, this is not a clean
    # accepted-gap observation because the decision boundary was missed.
    if first == 0:
        return None
    return first


def _first_roi_entry_index_xy(xy: np.ndarray, roi_path: MplPath) -> Optional[int]:
    if len(xy) == 0:
        return None
    inside = roi_path.contains_points(xy)
    if not inside.any():
        return None
    first = int(np.argmax(inside))
    if first == 0:
        return None
    return first


def _bbox_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a_min, a_max = a.min(axis=0), a.max(axis=0)
    b_min, b_max = b.min(axis=0), b.max(axis=0)
    gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gap))


def _make_cache(track: TrackRecord, dt: float) -> TrackCache:
    xy, frames = _track_xy_frames(track)
    speeds = _speeds_from_xy(track, xy, dt)
    median_speed = float(np.quantile(speeds, 0.5)) if len(speeds) else 0.0
    return TrackCache(
        track=track,
        xy=xy,
        frames=frames,
        heading=_trajectory_heading_xy(xy),
        median_speed_mps=median_speed,
        cumulative_s=_path_cumulative(xy),
    )


def _nearest_future_conflict(
    ego_xy: np.ndarray,
    other_xy: np.ndarray,
    ego_start_idx: int,
    other_start_idx: int,
    conflict_distance_m: float,
) -> Optional[Tuple[int, int, Tuple[float, float], float]]:
    ego_future = ego_xy[ego_start_idx:]
    other_future = other_xy[other_start_idx:]
    if len(ego_future) < 2 or len(other_future) < 2:
        return None

    # The caller already clips the horizon, and this hard cap prevents accidental
    # O(N^2) blow-ups on unusually long tracks.
    ego_future = ego_future[:220]
    other_future = other_future[:220]

    diff = ego_future[:, None, :] - other_future[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    flat_idx = int(np.argmin(d2))
    ego_rel, other_rel = np.unravel_index(flat_idx, d2.shape)
    min_dist = float(math.sqrt(float(d2[ego_rel, other_rel])))
    if min_dist > conflict_distance_m:
        return None
    conflict = (ego_future[ego_rel] + other_future[other_rel]) / 2.0
    return ego_start_idx + int(ego_rel), other_start_idx + int(other_rel), (float(conflict[0]), float(conflict[1])), min_dist


def _same_scene_tracks(tracks: Sequence[TrackRecord]) -> Dict[Tuple[str, str], List[TrackRecord]]:
    grouped: Dict[Tuple[str, str], List[TrackRecord]] = {}
    for track in tracks:
        grouped.setdefault((track.location, track.scene_id), []).append(track)
    return grouped


def _limit_scenes_per_city(tracks: Sequence[TrackRecord], max_scenes_per_city: Optional[int]) -> List[TrackRecord]:
    if max_scenes_per_city is None:
        return list(tracks)
    city_scenes: Dict[str, List[str]] = {}
    for track in tracks:
        scenes = city_scenes.setdefault(track.location, [])
        if track.scene_id not in scenes and len(scenes) < max_scenes_per_city:
            scenes.append(track.scene_id)
    allowed = {(city, scene) for city, scenes in city_scenes.items() for scene in scenes}
    return [track for track in tracks if (track.location, track.scene_id) in allowed]


def compute_critical_gaps(
    tracks: Sequence[TrackRecord],
    roi_paths: Mapping[str, MplPath],
    dt: float,
    conflict_distance_m: float,
    opposing_heading_min_deg: float,
    max_time_gap_s: float,
    min_other_speed_mps: float,
    pair_initial_distance_m: float,
    max_conflict_candidates_per_ego: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = _same_scene_tracks(tracks)
    pair_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    max_future_steps = int(math.ceil((max_time_gap_s + 3.0) / max(dt, 1e-6)))

    for (location, scene_id), scene_tracks in grouped.items():
        roi_path = roi_paths.get(location)
        if roi_path is None:
            continue
        left_tracks = [_make_cache(track, dt) for track in scene_tracks if track.maneuver == "left-turn"]
        straight_tracks = [_make_cache(track, dt) for track in scene_tracks if track.maneuver == "straight"]
        left_tracks = [
            info
            for info in left_tracks
            if len(info.xy) >= 5 and len(info.xy) == len(info.frames) and info.heading is not None
        ]
        straight_tracks = [
            info
            for info in straight_tracks
            if len(info.xy) >= 5
            and len(info.xy) == len(info.frames)
            and info.heading is not None
            and info.median_speed_mps >= min_other_speed_mps
        ]
        if not left_tracks or not straight_tracks:
            continue

        for ego_info in left_tracks:
            ego = ego_info.track
            ego_xy = ego_info.xy
            ego_frames = ego_info.frames
            entry_idx = _first_roi_entry_index_xy(ego_xy, roi_path)
            if entry_idx is None:
                skipped_rows.append(
                    {"location": location, "scene_id": scene_id, "ego_agent_id": ego.agent_id, "reason": "no_clean_roi_entry"}
                )
                continue
            entry_frame = int(ego_frames[entry_idx])
            ego_heading = ego_info.heading
            if ego_heading is None:
                continue
            ego_future_end = min(len(ego_xy), entry_idx + max_future_steps)
            ego_future = ego_xy[entry_idx:ego_future_end]
            if len(ego_future) < 2:
                continue

            ranked_candidates: List[Tuple[float, TrackCache, int, float]] = []
            for other_info in straight_tracks:
                other = other_info.track
                if other.agent_id == ego.agent_id:
                    continue
                heading_diff = _angle_diff_deg(ego_heading, other_info.heading)
                if heading_diff < opposing_heading_min_deg:
                    continue
                other_candidates = np.flatnonzero(other_info.frames >= entry_frame)
                if len(other_candidates) == 0:
                    continue
                other_start_idx = int(other_candidates[0])
                initial_distance = float(np.linalg.norm(other_info.xy[other_start_idx] - ego_xy[entry_idx]))
                if initial_distance > pair_initial_distance_m:
                    continue
                ranked_candidates.append((initial_distance, other_info, other_start_idx, heading_diff))

            if max_conflict_candidates_per_ego > 0 and len(ranked_candidates) > max_conflict_candidates_per_ego:
                ranked_candidates = sorted(ranked_candidates, key=lambda item: item[0])[:max_conflict_candidates_per_ego]

            for initial_distance, other_info, other_start_idx, heading_diff in ranked_candidates:
                other = other_info.track
                other_xy = other_info.xy
                other_frames = other_info.frames
                other_future_end = min(len(other_xy), other_start_idx + max_future_steps)
                other_future = other_xy[other_start_idx:other_future_end]
                if _bbox_distance(ego_future, other_future) > conflict_distance_m:
                    continue

                conflict = _nearest_future_conflict(
                    ego_xy[:ego_future_end],
                    other_xy[:other_future_end],
                    ego_start_idx=entry_idx,
                    other_start_idx=other_start_idx,
                    conflict_distance_m=conflict_distance_m,
                )
                if conflict is None:
                    continue
                ego_conflict_idx, other_conflict_idx, conflict_point, min_dist = conflict
                ego_arrival_s = (int(ego_frames[ego_conflict_idx]) - entry_frame) * dt
                other_arrival_s = (int(other_frames[other_conflict_idx]) - entry_frame) * dt
                time_gap_s = float(other_arrival_s - ego_arrival_s)
                if time_gap_s <= 0.0 or time_gap_s > max_time_gap_s:
                    continue
                distance_gap_m = float(
                    max(0.0, other_info.cumulative_s[other_conflict_idx] - other_info.cumulative_s[other_start_idx])
                )
                pair_rows.append(
                    {
                        "location": location,
                        "city": LOCATION_DISPLAY.get(location, location),
                        "scene_id": scene_id,
                        "ego_agent_id": ego.agent_id,
                        "ego_class": ego.class_name,
                        "conflict_agent_id": other.agent_id,
                        "conflict_class": other.class_name,
                        "ego_entry_frame": entry_frame,
                        "ego_conflict_frame": int(ego_frames[ego_conflict_idx]),
                        "conflict_agent_frame": int(other_frames[other_conflict_idx]),
                        "time_gap_s": time_gap_s,
                        "distance_gap_m": distance_gap_m,
                        "ego_arrival_to_conflict_s": float(ego_arrival_s),
                        "other_arrival_to_conflict_s": float(other_arrival_s),
                        "conflict_point_x": conflict_point[0],
                        "conflict_point_y": conflict_point[1],
                        "min_trajectory_distance_m": min_dist,
                        "heading_diff_deg": heading_diff,
                        "initial_distance_m": initial_distance,
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        accepted_df = pd.DataFrame()
    else:
        accepted_df = (
            pair_df.sort_values(["location", "scene_id", "ego_agent_id", "time_gap_s", "distance_gap_m"])
            .groupby(["location", "scene_id", "ego_agent_id"], as_index=False)
            .first()
            .sort_values(["location", "time_gap_s"])
        )
    skipped_df = pd.DataFrame(skipped_rows)
    return pair_df, accepted_df, skipped_df


def summarize_by_city(accepted_df: pd.DataFrame, pair_df: pd.DataFrame, cities: Sequence[str]) -> pd.DataFrame:
    rows = []
    for city in cities:
        city_acc = accepted_df[accepted_df["location"] == city] if not accepted_df.empty else pd.DataFrame()
        city_pairs = pair_df[pair_df["location"] == city] if not pair_df.empty else pd.DataFrame()
        row: Dict[str, Any] = {
            "location": city,
            "city": LOCATION_DISPLAY.get(city, city),
            "accepted_samples": int(len(city_acc)),
            "candidate_pairs": int(len(city_pairs)),
        }
        for col in ("time_gap_s", "distance_gap_m"):
            if city_acc.empty:
                row.update({f"{col}_p10": np.nan, f"{col}_p25": np.nan, f"{col}_median": np.nan, f"{col}_p75": np.nan, f"{col}_mean": np.nan})
            else:
                values = city_acc[col].to_numpy(dtype=float)
                row.update(
                    {
                        f"{col}_p10": float(np.quantile(values, 0.10)),
                        f"{col}_p25": float(np.quantile(values, 0.25)),
                        f"{col}_median": float(np.quantile(values, 0.50)),
                        f"{col}_p75": float(np.quantile(values, 0.75)),
                        f"{col}_mean": float(np.mean(values)),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_city_distributions(accepted_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path, cities: Sequence[str]) -> None:
    for city in cities:
        city_df = accepted_df[accepted_df["location"] == city] if not accepted_df.empty else pd.DataFrame()
        display = LOCATION_DISPLAY.get(city, city)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        if not city_df.empty:
            axes[0].hist(city_df["time_gap_s"], bins=30, color="#b9472a", alpha=0.82)
            axes[1].hist(city_df["distance_gap_m"], bins=30, color="#2f5d73", alpha=0.82)
        axes[0].set_title(f"{display} accepted time gap")
        axes[0].set_xlabel("Time gap (s)")
        axes[0].set_ylabel("Count")
        axes[1].set_title(f"{display} accepted distance gap")
        axes[1].set_xlabel("Distance gap (m)")
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"critical_gap_hist_{city}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
        if not city_df.empty:
            values = np.sort(city_df["time_gap_s"].to_numpy(dtype=float))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, color="#b9472a", linewidth=2.2)
        ax.set_title(f"{display} accepted time gap CDF")
        ax.set_xlabel("Time gap (s)")
        ax.set_ylabel("CDF")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"critical_gap_cdf_{city}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    plot_df = summary_df.copy()
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["time_gap_s_median"], color="#b9472a", alpha=0.82, label="Median")
    ax.errorbar(
        x,
        plot_df["time_gap_s_median"],
        yerr=[
            plot_df["time_gap_s_median"] - plot_df["time_gap_s_p25"],
            plot_df["time_gap_s_p75"] - plot_df["time_gap_s_median"],
        ],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        label="P25-P75",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["location"], rotation=0)
    ax.set_ylabel("Accepted time gap (s)")
    ax.set_title("Critical Gap Acceptance by Intersection")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "critical_gap_city_comparison.png")
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
            if isinstance(value, float):
                cell = "" if math.isnan(value) else f"{value:.3f}"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html_report(output_path: Path, summary_df: pd.DataFrame, low_gap_df: pd.DataFrame, cities: Sequence[str]) -> None:
    cards = []
    for row in summary_df.to_dict("records"):
        city = row["location"]
        cards.append(
            f"""
            <article class='card'>
              <h3>{html.escape(row['city'])} <span>{html.escape(city)}</span></h3>
              <b>{int(row['accepted_samples'])}</b><small>accepted samples</small>
              <b>{row['time_gap_s_median']:.2f}s</b><small>median time gap</small>
              <b>{row['distance_gap_m_median']:.1f}m</b><small>median distance gap</small>
              <a href='critical_gap_cdf_{html.escape(city)}.png'><img src='critical_gap_cdf_{html.escape(city)}.png' alt='cdf'></a>
            </article>
            """
        )
    summary_table = _html_table(
        summary_df.to_dict("records"),
        [
            ("city", "City"),
            ("accepted_samples", "Accepted"),
            ("candidate_pairs", "Pairs"),
            ("time_gap_s_p10", "TG P10"),
            ("time_gap_s_median", "TG Median"),
            ("time_gap_s_p75", "TG P75"),
            ("distance_gap_m_median", "DG Median"),
        ],
    )
    low_table = _html_table(
        low_gap_df.head(40).to_dict("records"),
        [
            ("city", "City"),
            ("scene_id", "Scene"),
            ("ego_agent_id", "Ego"),
            ("conflict_agent_id", "Conflict"),
            ("time_gap_s", "Time gap"),
            ("distance_gap_m", "Distance gap"),
            ("ego_entry_frame", "Entry frame"),
        ],
    )
    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Critical Gap Acceptance</title>
  <style>
    :root {{ --ink:#1e2426; --paper:#f5efe6; --card:#fffaf0; --line:#dac8ae; --muted:#667175; --red:#b9472a; --blue:#2f5d73; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f5efe6,#e8eeee); }}
    header {{ padding:44px 5vw 72px; color:#fff; background:radial-gradient(circle at 80% 15%,#dc9b55,transparent 24%), linear-gradient(135deg,#263b42,#6d392c); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); }} header p {{ max-width:980px; line-height:1.65; color:#f4eadf; font-size:17px; }}
    main {{ padding:0 5vw 58px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; margin-top:-42px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 30px rgba(30,36,38,.08); }}
    .card h3 {{ margin:0 0 10px; }} .card h3 span {{ color:var(--muted); font-size:13px; }} .card b {{ display:block; color:var(--red); font-size:25px; margin-top:8px; }} .card small {{ color:var(--muted); }}
    img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; margin-top:12px; }} section {{ margin-top:34px; }} h2 {{ font-size:30px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,240,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #eadcc8; text-align:left; font-size:14px; }} th {{ background:#ead6bd; }} tr:hover td {{ background:#fff0d8; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fffaf0; border:1px solid var(--line); border-radius:999px; color:var(--blue); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>SinD Critical Gap Acceptance</h1>
  <p>统计 UPLT 左转车辆首次进入路口核心 ROI 时，与对向直行冲突车辆的 accepted time gap 和 distance gap。较小的 critical gap 表示更激进的抢行行为。</p>
</header>
<main>
  <div class='grid'>{''.join(cards)}</div>
  <section><h2>Cross-City Comparison</h2><a href='critical_gap_city_comparison.png'><img src='critical_gap_city_comparison.png' alt='comparison'></a></section>
  <section><h2>Summary</h2>{summary_table}</section>
  <section><h2>Lowest Gap Cases</h2>{low_table}</section>
  <section class='links'><h2>Artifacts</h2>
    <a href='critical_gap_accepted.csv'>critical_gap_accepted.csv</a>
    <a href='critical_gap_pairs.csv'>critical_gap_pairs.csv</a>
    <a href='critical_gap_summary_by_city.csv'>critical_gap_summary_by_city.csv</a>
    <a href='low_gap_cases.csv'>low_gap_cases.csv</a>
    <a href='summary.json'>summary.json</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def build_roi_paths(data_dir: Path, cities: Sequence[str]) -> Dict[str, MplPath]:
    refs_by_city = load_lane_references(data_dir, cities)
    roi_paths = {}
    for city, refs in refs_by_city.items():
        polygon, _ = infer_intersection_core_roi(refs)
        roi_paths[city] = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
    return roi_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute UPLT critical gap acceptance for SinD.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_critical_gap_acceptance"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--conflict-distance-m", type=float, default=3.0)
    parser.add_argument("--opposing-heading-min-deg", type=float, default=60.0)
    parser.add_argument("--max-time-gap-s", type=float, default=8.0)
    parser.add_argument("--min-other-speed-mps", type=float, default=0.5)
    parser.add_argument("--pair-initial-distance-m", type=float, default=90.0)
    parser.add_argument("--max-conflict-candidates-per-ego", type=int, default=24)
    parser.add_argument("--max-scenes-per-city", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cities = [normalize_location(city) for city in args.cities]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracks for {args.cities}...", flush=True)
    tracks = load_sind_tracks(args.data_dir, args.cities)
    tracks = filter_tracks(tracks, city=args.cities, agent_type=VEHICLE_AGENT_TYPE, maneuver=None)
    tracks = [track for track in tracks if track.maneuver in {"left-turn", "straight"}]
    tracks = _limit_scenes_per_city(tracks, args.max_scenes_per_city)

    print("Building intersection ROI paths...", flush=True)
    roi_paths = build_roi_paths(args.data_dir, args.cities)
    print(f"Computing critical gaps over {len(tracks)} candidate tracks...", flush=True)
    pair_df, accepted_df, skipped_df = compute_critical_gaps(
        tracks,
        roi_paths=roi_paths,
        dt=args.dt,
        conflict_distance_m=args.conflict_distance_m,
        opposing_heading_min_deg=args.opposing_heading_min_deg,
        max_time_gap_s=args.max_time_gap_s,
        min_other_speed_mps=args.min_other_speed_mps,
        pair_initial_distance_m=args.pair_initial_distance_m,
        max_conflict_candidates_per_ego=args.max_conflict_candidates_per_ego,
    )
    summary_df = summarize_by_city(accepted_df, pair_df, args.cities)
    low_gap_df = accepted_df.sort_values("time_gap_s").head(100) if not accepted_df.empty else pd.DataFrame()

    pair_df.to_csv(args.output_dir / "critical_gap_pairs.csv", index=False)
    accepted_df.to_csv(args.output_dir / "critical_gap_accepted.csv", index=False)
    skipped_df.to_csv(args.output_dir / "critical_gap_skipped.csv", index=False)
    summary_df.to_csv(args.output_dir / "critical_gap_summary_by_city.csv", index=False)
    low_gap_df.to_csv(args.output_dir / "low_gap_cases.csv", index=False)
    plot_city_distributions(accepted_df, summary_df, args.output_dir, args.cities)
    write_html_report(args.output_dir / "index.html", summary_df, low_gap_df, args.cities)

    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "dt": args.dt,
        "conflict_distance_m": args.conflict_distance_m,
        "opposing_heading_min_deg": args.opposing_heading_min_deg,
        "max_time_gap_s": args.max_time_gap_s,
        "min_other_speed_mps": args.min_other_speed_mps,
        "pair_initial_distance_m": args.pair_initial_distance_m,
        "max_conflict_candidates_per_ego": args.max_conflict_candidates_per_ego,
        "candidate_tracks": int(len(tracks)),
        "candidate_pairs": int(len(pair_df)),
        "accepted_samples": int(len(accepted_df)),
        "summary_by_city": summary_df.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote outputs to: {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
