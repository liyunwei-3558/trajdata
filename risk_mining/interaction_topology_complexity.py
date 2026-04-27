#!/usr/bin/env python
"""Compute SinD interaction topology and game-complexity metrics.

This script measures complexity, not danger. It builds loose per-frame
interaction graphs inside the intersection core: agents are connected when they
are close and their short-horizon future paths can conflict. Connected component
sizes define interaction degree; temporally merged components define interaction
duration episodes.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS, load_lane_references  # noqa: E402
from intersection_spatiotemporal_density import (  # noqa: E402
    filter_static_tracks,
    infer_intersection_core_roi,
)

ALLOWED_CLASSES = {"car", "truck", "bus", "tricycle", "motorcycle", "bicycle", "pedestrian"}
CITY_GROUPS = {
    "cc": "ChangChun",
    "tj": "TianJin",
    "cqIR": "ChongQing",
    "cqNR": "ChongQing",
    "cqR": "ChongQing",
    "xasl": "XiAn",
}
DEGREE_BUCKETS = ("2-party", "3-party", "4+-party")


@dataclass
class TrackCache:
    idx: int
    track: TrackRecord
    frames: np.ndarray
    xy: np.ndarray
    inside_roi_buffer: np.ndarray


@dataclass
class ActiveEpisode:
    episode_id: int
    location: str
    scene_id: str
    start_frame: int
    end_frame: int
    last_frame: int
    members: Set[int]
    all_members: Set[int]
    max_degree: int
    degree_sum: int
    observed_frames: int


def _degree_bucket(degree: int) -> str:
    if degree <= 2:
        return "2-party"
    if degree == 3:
        return "3-party"
    return "4+-party"


def _city_group(location: str) -> str:
    return CITY_GROUPS.get(location, LOCATION_DISPLAY.get(location, location))


def _track_xy_frames(track: TrackRecord) -> Tuple[np.ndarray, np.ndarray]:
    state = track.state.sort_values("frame_id")
    if not {"x", "y", "frame_id"}.issubset(state.columns):
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
    xy = state[["x", "y"]].to_numpy(dtype=float)
    frames = state["frame_id"].to_numpy(dtype=int)
    mask = np.isfinite(xy).all(axis=1) & np.isfinite(frames)
    return xy[mask], frames[mask]


def _polygon_to_path(polygon: Polygon) -> MplPath:
    return MplPath(np.asarray(polygon.exterior.coords, dtype=float))


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


def _paths_conflict(
    a: TrackCache,
    a_idx: int,
    b: TrackCache,
    b_idx: int,
    horizon_steps: int,
    stride: int,
    path_conflict_distance_m: float,
) -> bool:
    a_future = _future_window(a, a_idx, horizon_steps, stride)
    b_future = _future_window(b, b_idx, horizon_steps, stride)
    if len(a_future) < 2 or len(b_future) < 2:
        return False
    if _bbox_distance(a_future, b_future) > path_conflict_distance_m:
        return False
    diff = a_future[:, None, :] - b_future[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return bool(float(np.min(d2)) <= path_conflict_distance_m**2)


def _connected_components(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[Set[int]]:
    adjacency: Dict[int, List[int]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    seen: Set[int] = set()
    components: List[Set[int]] = []
    for node in nodes:
        if node in seen or not adjacency.get(node):
            continue
        stack = [node]
        seen.add(node)
        comp: Set[int] = set()
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adjacency.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) >= 2:
            components.append(comp)
    return components


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _finalize_episode(ep: ActiveEpisode, caches: Sequence[TrackCache], dt: float) -> Dict[str, Any]:
    member_infos = [caches[idx] for idx in sorted(ep.all_members)]
    class_counts: Dict[str, int] = {}
    agent_ids = []
    for info in member_infos:
        class_counts[info.track.class_name] = class_counts.get(info.track.class_name, 0) + 1
        agent_ids.append(str(info.track.agent_id))
    duration_s = (ep.end_frame - ep.start_frame) * dt + dt
    return {
        "location": ep.location,
        "city": LOCATION_DISPLAY.get(ep.location, ep.location),
        "city_group": _city_group(ep.location),
        "scene_id": ep.scene_id,
        "episode_id": ep.episode_id,
        "start_frame": int(ep.start_frame),
        "end_frame": int(ep.end_frame),
        "duration_s": float(duration_s),
        "observed_frames": int(ep.observed_frames),
        "max_degree": int(ep.max_degree),
        "mean_degree": float(ep.degree_sum / max(1, ep.observed_frames)),
        "degree_bucket": _degree_bucket(ep.max_degree),
        "unique_agents": int(len(ep.all_members)),
        "agent_ids": ";".join(agent_ids),
        "class_counts_json": json.dumps(class_counts, ensure_ascii=False, sort_keys=True),
    }


def _merge_frame_components(
    active: List[ActiveEpisode],
    components: Sequence[Set[int]],
    frame: int,
    location: str,
    scene_id: str,
    next_episode_id: int,
    max_gap_frames: int,
    jaccard_threshold: float,
    caches: Sequence[TrackCache],
    dt: float,
    min_duration_s: float,
) -> Tuple[List[ActiveEpisode], List[Dict[str, Any]], int]:
    finalized: List[Dict[str, Any]] = []
    used_components: Set[int] = set()
    kept_active: List[ActiveEpisode] = []

    # Greedy matching from older episodes to the most overlapping current component.
    for ep in active:
        best_idx = None
        best_score = 0.0
        for comp_idx, comp in enumerate(components):
            if comp_idx in used_components:
                continue
            score = _jaccard(ep.members, comp)
            if score > best_score:
                best_score = score
                best_idx = comp_idx
        if best_idx is not None and best_score >= jaccard_threshold:
            comp = components[best_idx]
            used_components.add(best_idx)
            ep.end_frame = frame
            ep.last_frame = frame
            ep.members = set(comp)
            ep.all_members.update(comp)
            ep.max_degree = max(ep.max_degree, len(comp))
            ep.degree_sum += len(comp)
            ep.observed_frames += 1
            kept_active.append(ep)
        elif frame - ep.last_frame <= max_gap_frames:
            kept_active.append(ep)
        else:
            row = _finalize_episode(ep, caches, dt)
            if row["duration_s"] >= min_duration_s:
                finalized.append(row)

    for comp_idx, comp in enumerate(components):
        if comp_idx in used_components:
            continue
        kept_active.append(
            ActiveEpisode(
                episode_id=next_episode_id,
                location=location,
                scene_id=scene_id,
                start_frame=frame,
                end_frame=frame,
                last_frame=frame,
                members=set(comp),
                all_members=set(comp),
                max_degree=len(comp),
                degree_sum=len(comp),
                observed_frames=1,
            )
        )
        next_episode_id += 1

    return kept_active, finalized, next_episode_id


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
) -> Tuple[Dict[str, Polygon], Dict[str, Polygon], Dict[str, Dict[str, Any]]]:
    refs_by_city = load_lane_references(data_dir, cities)
    core_rois: Dict[str, Polygon] = {}
    buffered_rois: Dict[str, Polygon] = {}
    debug_rows: Dict[str, Dict[str, Any]] = {}
    for city in cities:
        polygon, debug = infer_intersection_core_roi(
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
        debug_rows[city] = debug
    return core_rois, buffered_rois, debug_rows


def make_scene_caches(tracks: Sequence[TrackRecord], roi_polygon: Polygon) -> List[TrackCache]:
    roi_path = _polygon_to_path(roi_polygon)
    caches: List[TrackCache] = []
    for track in tracks:
        xy, frames = _track_xy_frames(track)
        if len(xy) < 3 or len(xy) != len(frames):
            continue
        inside = roi_path.contains_points(xy)
        if not inside.any():
            continue
        caches.append(TrackCache(idx=len(caches), track=track, frames=frames, xy=xy, inside_roi_buffer=inside))
    return caches


def compute_scene_interactions(
    location: str,
    scene_id: str,
    tracks: Sequence[TrackRecord],
    roi_polygon: Polygon,
    dt: float,
    distance_threshold_m: float,
    path_conflict_distance_m: float,
    future_horizon_s: float,
    future_stride: int,
    min_duration_s: float,
    max_gap_s: float,
    jaccard_threshold: float,
    next_episode_id: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    caches = make_scene_caches(tracks, roi_polygon)
    if len(caches) < 2:
        return [], [], next_episode_id

    frame_entries: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
    for cache in caches:
        for pos_idx, frame in enumerate(cache.frames):
            if not cache.inside_roi_buffer[pos_idx]:
                continue
            frame_entries.setdefault(int(frame), []).append((cache.idx, pos_idx, cache.xy[pos_idx]))

    horizon_steps = max(1, int(round(future_horizon_s / dt)))
    max_gap_frames = max(0, int(round(max_gap_s / dt)))
    frame_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    active: List[ActiveEpisode] = []

    for frame in sorted(frame_entries):
        entries = frame_entries[frame]
        components: List[Set[int]] = []
        if len(entries) >= 2:
            points = np.asarray([entry[2] for entry in entries], dtype=float)
            pairs = cKDTree(points).query_pairs(distance_threshold_m)
            edges: List[Tuple[int, int]] = []
            for local_a, local_b in pairs:
                a_cache_idx, a_pos_idx, _ = entries[local_a]
                b_cache_idx, b_pos_idx, _ = entries[local_b]
                if _paths_conflict(
                    caches[a_cache_idx],
                    a_pos_idx,
                    caches[b_cache_idx],
                    b_pos_idx,
                    horizon_steps=horizon_steps,
                    stride=future_stride,
                    path_conflict_distance_m=path_conflict_distance_m,
                ):
                    edges.append((a_cache_idx, b_cache_idx))
            if edges:
                components = _connected_components([entry[0] for entry in entries], edges)

        for comp_idx, comp in enumerate(components):
            member_infos = [caches[idx] for idx in sorted(comp)]
            classes = [info.track.class_name for info in member_infos]
            class_counts = {cls: classes.count(cls) for cls in sorted(set(classes))}
            frame_rows.append(
                {
                    "location": location,
                    "city": LOCATION_DISPLAY.get(location, location),
                    "city_group": _city_group(location),
                    "scene_id": scene_id,
                    "frame_id": int(frame),
                    "component_id_in_frame": int(comp_idx),
                    "degree": int(len(comp)),
                    "degree_bucket": _degree_bucket(len(comp)),
                    "agent_ids": ";".join(str(info.track.agent_id) for info in member_infos),
                    "class_counts_json": json.dumps(class_counts, ensure_ascii=False, sort_keys=True),
                }
            )

        active, finalized, next_episode_id = _merge_frame_components(
            active,
            components,
            frame=frame,
            location=location,
            scene_id=scene_id,
            next_episode_id=next_episode_id,
            max_gap_frames=max_gap_frames,
            jaccard_threshold=jaccard_threshold,
            caches=caches,
            dt=dt,
            min_duration_s=min_duration_s,
        )
        episode_rows.extend(finalized)

    if frame_entries:
        final_frame = max(frame_entries)
        for ep in active:
            row = _finalize_episode(ep, caches, dt)
            if row["duration_s"] >= min_duration_s and final_frame - ep.last_frame <= max(max_gap_frames, final_frame - ep.last_frame):
                episode_rows.append(row)
    return frame_rows, episode_rows, next_episode_id


def compute_interaction_complexity(
    tracks: Sequence[TrackRecord],
    buffered_rois: Mapping[str, Polygon],
    dt: float,
    distance_threshold_m: float,
    path_conflict_distance_m: float,
    future_horizon_s: float,
    future_stride: int,
    min_duration_s: float,
    max_gap_s: float,
    jaccard_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped: Dict[Tuple[str, str], List[TrackRecord]] = {}
    for track in tracks:
        grouped.setdefault((track.location, track.scene_id), []).append(track)

    all_frame_rows: List[Dict[str, Any]] = []
    all_episode_rows: List[Dict[str, Any]] = []
    next_episode_id = 1
    for idx, ((location, scene_id), scene_tracks) in enumerate(sorted(grouped.items()), start=1):
        roi_polygon = buffered_rois.get(location)
        if roi_polygon is None:
            continue
        frame_rows, episode_rows, next_episode_id = compute_scene_interactions(
            location,
            scene_id,
            scene_tracks,
            roi_polygon,
            dt=dt,
            distance_threshold_m=distance_threshold_m,
            path_conflict_distance_m=path_conflict_distance_m,
            future_horizon_s=future_horizon_s,
            future_stride=future_stride,
            min_duration_s=min_duration_s,
            max_gap_s=max_gap_s,
            jaccard_threshold=jaccard_threshold,
            next_episode_id=next_episode_id,
        )
        all_frame_rows.extend(frame_rows)
        all_episode_rows.extend(episode_rows)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(grouped)} scenes; frame components={len(all_frame_rows):,}; episodes={len(all_episode_rows):,}", flush=True)

    return pd.DataFrame(all_frame_rows), pd.DataFrame(all_episode_rows)


def summarize_degree(frame_df: pd.DataFrame, group_col: str, ordered_groups: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in ordered_groups:
        sub = frame_df[frame_df[group_col] == group] if not frame_df.empty else pd.DataFrame()
        total = int(len(sub))
        row: Dict[str, Any] = {group_col: group, "component_frames": total}
        for bucket in DEGREE_BUCKETS:
            count = int((sub["degree_bucket"] == bucket).sum()) if total else 0
            row[f"{bucket}_count"] = count
            row[f"{bucket}_ratio"] = float(count / total) if total else 0.0
        row["n_ge_3_ratio"] = float(((sub["degree"] >= 3).sum() / total)) if total else 0.0
        row["mean_degree"] = float(sub["degree"].mean()) if total else 0.0
        row["max_degree"] = int(sub["degree"].max()) if total else 0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_duration(episode_df: pd.DataFrame, group_col: str, ordered_groups: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in ordered_groups:
        sub = episode_df[episode_df[group_col] == group] if not episode_df.empty else pd.DataFrame()
        row: Dict[str, Any] = {group_col: group, "episodes": int(len(sub))}
        if sub.empty:
            row.update({"duration_p10_s": 0.0, "duration_median_s": 0.0, "duration_p90_s": 0.0, "duration_mean_s": 0.0, "long_duration_ge_5s_ratio": 0.0})
        else:
            values = sub["duration_s"].to_numpy(dtype=float)
            row.update(
                {
                    "duration_p10_s": float(np.quantile(values, 0.10)),
                    "duration_median_s": float(np.quantile(values, 0.50)),
                    "duration_p90_s": float(np.quantile(values, 0.90)),
                    "duration_mean_s": float(np.mean(values)),
                    "long_duration_ge_5s_ratio": float(np.mean(values >= 5.0)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stacked_degree(summary_df: pd.DataFrame, group_col: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    x = np.arange(len(summary_df))
    bottom = np.zeros(len(summary_df), dtype=float)
    colors = {"2-party": "#375c6c", "3-party": "#d28b3c", "4+-party": "#b64032"}
    labels = {"2-party": "2-party", "3-party": "3-party", "4+-party": "4+ party"}
    for bucket in DEGREE_BUCKETS:
        values = summary_df[f"{bucket}_ratio"].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors[bucket], label=labels[bucket], width=0.72)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df[group_col].astype(str), rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of interaction component-frames")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_duration_distribution(episode_df: pd.DataFrame, group_col: str, groups: Sequence[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    bins = np.linspace(0, max(12.0, float(episode_df["duration_s"].quantile(0.98)) if not episode_df.empty else 12.0), 32)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(groups))))
    for color, group in zip(colors, groups):
        sub = episode_df[episode_df[group_col] == group] if not episode_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        ax.hist(sub["duration_s"], bins=bins, histtype="step", linewidth=2.0, color=color, label=str(group), density=True)
    ax.set_xlabel("Interaction duration (s)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_duration_boxplot(episode_df: pd.DataFrame, group_col: str, groups: Sequence[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.1), dpi=160)
    data = [episode_df.loc[episode_df[group_col] == group, "duration_s"].to_numpy(dtype=float) for group in groups]
    data = [values if len(values) else np.array([np.nan]) for values in data]
    ax.boxplot(data, tick_labels=list(groups), showfliers=False, patch_artist=True, boxprops={"facecolor": "#e8d5b7", "color": "#5a4a3f"}, medianprops={"color": "#b64032", "linewidth": 2})
    ax.set_ylabel("Interaction duration (s)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_geojson(output_path: Path, core_rois: Mapping[str, Polygon], buffered_rois: Mapping[str, Polygon], roi_debug: Mapping[str, Mapping[str, Any]]) -> None:
    features = []
    for location, polygon in core_rois.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "roi_type": "core", **dict(roi_debug.get(location, {}))},
                "geometry": mapping(polygon),
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": {"location": location, "city": LOCATION_DISPLAY.get(location, location), "roi_type": "buffered"},
                "geometry": mapping(buffered_rois[location]),
            }
        )
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


def write_html_report(
    output_path: Path,
    degree_loc: pd.DataFrame,
    degree_city: pd.DataFrame,
    duration_loc: pd.DataFrame,
    duration_city: pd.DataFrame,
    top_episodes: pd.DataFrame,
) -> None:
    total_components = int(degree_loc["component_frames"].sum()) if not degree_loc.empty else 0
    total_episodes = int(duration_loc["episodes"].sum()) if not duration_loc.empty else 0
    n_ge_3 = 0.0
    if total_components:
        n_ge_3 = float(
            sum(row["component_frames"] * row["n_ge_3_ratio"] for row in degree_loc.to_dict("records")) / total_components
        )
    degree_table = _html_table(
        degree_loc.to_dict("records"),
        [
            ("location", "Location"),
            ("component_frames", "Component-frames"),
            ("2-party_ratio", "2-party"),
            ("3-party_ratio", "3-party"),
            ("4+-party_ratio", "4+ party"),
            ("n_ge_3_ratio", "N>=3"),
            ("mean_degree", "Mean N"),
            ("max_degree", "Max N"),
        ],
    )
    city_degree_table = _html_table(
        degree_city.to_dict("records"),
        [
            ("city_group", "City"),
            ("component_frames", "Component-frames"),
            ("2-party_ratio", "2-party"),
            ("3-party_ratio", "3-party"),
            ("4+-party_ratio", "4+ party"),
            ("n_ge_3_ratio", "N>=3"),
        ],
    )
    duration_table = _html_table(
        duration_loc.to_dict("records"),
        [
            ("location", "Location"),
            ("episodes", "Episodes"),
            ("duration_median_s", "Median s"),
            ("duration_p90_s", "P90 s"),
            ("duration_mean_s", "Mean s"),
            ("long_duration_ge_5s_ratio", ">=5s"),
        ],
    )
    city_duration_table = _html_table(
        duration_city.to_dict("records"),
        [
            ("city_group", "City"),
            ("episodes", "Episodes"),
            ("duration_median_s", "Median s"),
            ("duration_p90_s", "P90 s"),
            ("long_duration_ge_5s_ratio", ">=5s"),
        ],
    )
    top_table = _html_table(
        top_episodes.head(40).to_dict("records"),
        [
            ("city", "City"),
            ("scene_id", "Scene"),
            ("duration_s", "Duration s"),
            ("max_degree", "Max N"),
            ("unique_agents", "Agents"),
            ("start_frame", "Start"),
            ("end_frame", "End"),
        ],
    )
    html_text = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Interaction Topology Complexity</title>
  <style>
    :root {{ --ink:#1f2421; --paper:#f3ead9; --card:#fffaf0; --line:#d9c6a6; --muted:#67716c; --red:#b64032; --blue:#375c6c; --gold:#d28b3c; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f3ead9,#e6eeee); }}
    header {{ padding:44px 5vw 74px; color:#fff; background:radial-gradient(circle at 82% 12%,#d99743,transparent 25%), linear-gradient(135deg,#233c43,#6a342a); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); }} header p {{ max-width:1040px; line-height:1.65; color:#f7ead8; font-size:17px; }}
    main {{ padding:0 5vw 58px; }} .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:-42px; }}
    .kpi,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 30px rgba(31,36,33,.08); }}
    .kpi b {{ display:block; color:var(--red); font-size:28px; }} .kpi small {{ color:var(--muted); }} section {{ margin-top:34px; }} h2 {{ font-size:30px; margin-bottom:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }} img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,240,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #eadcc8; text-align:left; font-size:14px; }} th {{ background:#ead6bd; }} tr:hover td {{ background:#fff0d8; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fffaf0; border:1px solid var(--line); border-radius:999px; color:var(--blue); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>SinD Interaction Topology & Game Complexity</h1>
  <p>This report measures how many agents are coupled in loose intersection games, not how dangerous they are. Edges require same-frame proximity and short-horizon potential path conflict; connected component size gives interaction degree.</p>
</header>
<main>
  <div class='kpis'>
    <div class='kpi'><b>{total_components:,}</b><small>interaction component-frames</small></div>
    <div class='kpi'><b>{total_episodes:,}</b><small>merged interaction episodes</small></div>
    <div class='kpi'><b>{n_ge_3:.1%}</b><small>N >= 3 component share</small></div>
  </div>
  <section><h2>Interaction Degree</h2><div class='grid'><div class='panel'><a href='stacked_degree_by_location.png'><img src='stacked_degree_by_location.png'></a></div><div class='panel'><a href='stacked_degree_by_city.png'><img src='stacked_degree_by_city.png'></a></div></div></section>
  <section><h2>Interaction Duration</h2><div class='grid'><div class='panel'><a href='duration_distribution_by_location.png'><img src='duration_distribution_by_location.png'></a></div><div class='panel'><a href='duration_boxplot_by_city.png'><img src='duration_boxplot_by_city.png'></a></div></div></section>
  <section><h2>Degree by Location</h2>{degree_table}</section>
  <section><h2>Degree by City Group</h2>{city_degree_table}</section>
  <section><h2>Duration by Location</h2>{duration_table}</section>
  <section><h2>Duration by City Group</h2>{city_duration_table}</section>
  <section><h2>Longest Complex Episodes</h2>{top_table}</section>
  <section class='links'><h2>Artifacts</h2>
    <a href='interaction_degree_summary_by_location.csv'>degree summary by location</a>
    <a href='interaction_degree_summary_by_city.csv'>degree summary by city</a>
    <a href='interaction_frame_components.csv'>frame components</a>
    <a href='interaction_duration_episodes.csv'>duration episodes</a>
    <a href='interaction_duration_summary.csv'>duration summary</a>
    <a href='top_complex_episodes.csv'>top complex episodes</a>
    <a href='summary.json'>summary.json</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SinD interaction topology and game-complexity metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_interaction_topology_complexity"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--distance-threshold-m", type=float, default=15.0)
    parser.add_argument("--path-conflict-distance-m", type=float, default=3.0)
    parser.add_argument("--future-horizon-s", type=float, default=5.0)
    parser.add_argument("--future-stride", type=int, default=2)
    parser.add_argument("--min-duration-s", type=float, default=0.5)
    parser.add_argument("--max-gap-s", type=float, default=0.3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--roi-buffer-m", type=float, default=5.0)
    parser.add_argument("--core-buffer-m", type=float, default=3.0)
    parser.add_argument("--min-core-radius-m", type=float, default=12.0)
    parser.add_argument("--max-core-radius-m", type=float, default=32.0)
    parser.add_argument("--turn-heading-threshold-deg", type=float, default=25.0)
    parser.add_argument("--lane-trim-ratio", type=float, default=0.20)
    parser.add_argument("--central-quantile", type=float, default=0.62)
    parser.add_argument("--keep-static-tracks", action="store_true", help="Disable long-duration static/parked track filtering.")
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
    tracks = [track for track in tracks if track.class_name in ALLOWED_CLASSES]
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

    print("Building intersection ROIs...", flush=True)
    core_rois, buffered_rois, roi_debug = build_rois(
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
    write_geojson(args.output_dir / "interaction_rois.geojson", core_rois, buffered_rois, roi_debug)

    print(f"Computing interaction topology over {len(tracks):,} dynamic tracks...", flush=True)
    frame_df, episode_df = compute_interaction_complexity(
        tracks,
        buffered_rois=buffered_rois,
        dt=args.dt,
        distance_threshold_m=args.distance_threshold_m,
        path_conflict_distance_m=args.path_conflict_distance_m,
        future_horizon_s=args.future_horizon_s,
        future_stride=args.future_stride,
        min_duration_s=args.min_duration_s,
        max_gap_s=args.max_gap_s,
        jaccard_threshold=args.jaccard_threshold,
    )

    location_groups = list(args.cities)
    city_groups = ["ChangChun", "TianJin", "ChongQing", "XiAn"]
    degree_loc = summarize_degree(frame_df, "location", location_groups)
    degree_city = summarize_degree(frame_df, "city_group", city_groups)
    duration_loc = summarize_duration(episode_df, "location", location_groups)
    duration_city = summarize_duration(episode_df, "city_group", city_groups)
    top_episodes = (
        episode_df.sort_values(["max_degree", "duration_s"], ascending=[False, False]).head(150)
        if not episode_df.empty
        else pd.DataFrame()
    )
    duration_summary = pd.concat(
        [duration_loc.assign(summary_level="location"), duration_city.assign(summary_level="city_group")],
        ignore_index=True,
        sort=False,
    )

    frame_df.to_csv(args.output_dir / "interaction_frame_components.csv", index=False)
    episode_df.to_csv(args.output_dir / "interaction_duration_episodes.csv", index=False)
    degree_loc.to_csv(args.output_dir / "interaction_degree_summary_by_location.csv", index=False)
    degree_city.to_csv(args.output_dir / "interaction_degree_summary_by_city.csv", index=False)
    duration_summary.to_csv(args.output_dir / "interaction_duration_summary.csv", index=False)
    top_episodes.to_csv(args.output_dir / "top_complex_episodes.csv", index=False)
    static_tracks_df.to_csv(args.output_dir / "filtered_static_tracks.csv", index=False)

    plot_stacked_degree(degree_loc, "location", args.output_dir / "stacked_degree_by_location.png", "Interaction Degree by SinD Location")
    plot_stacked_degree(degree_city, "city_group", args.output_dir / "stacked_degree_by_city.png", "Interaction Degree by City Group")
    plot_duration_distribution(episode_df, "location", location_groups, args.output_dir / "duration_distribution_by_location.png", "Interaction Duration Distribution by Location")
    plot_duration_boxplot(episode_df, "city_group", city_groups, args.output_dir / "duration_boxplot_by_city.png", "Interaction Duration by City Group")
    write_html_report(args.output_dir / "index.html", degree_loc, degree_city, duration_loc, duration_city, top_episodes)

    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "dt": args.dt,
        "distance_threshold_m": args.distance_threshold_m,
        "path_conflict_distance_m": args.path_conflict_distance_m,
        "future_horizon_s": args.future_horizon_s,
        "future_stride": args.future_stride,
        "min_duration_s": args.min_duration_s,
        "max_gap_s": args.max_gap_s,
        "jaccard_threshold": args.jaccard_threshold,
        "roi_buffer_m": args.roi_buffer_m,
        "static_filter_enabled": not args.keep_static_tracks,
        "filtered_static_tracks_total": int(len(static_tracks_df)),
        "dynamic_tracks": int(len(tracks)),
        "interaction_component_frames": int(len(frame_df)),
        "interaction_episodes": int(len(episode_df)),
        "degree_by_location": degree_loc.to_dict("records"),
        "degree_by_city": degree_city.to_dict("records"),
        "duration_by_location": duration_loc.to_dict("records"),
        "duration_by_city": duration_city.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote outputs to: {args.output_dir}")
    if not degree_loc.empty:
        print(degree_loc[["location", "component_frames", "2-party_ratio", "3-party_ratio", "4+-party_ratio", "n_ge_3_ratio", "mean_degree", "max_degree"]].to_string(index=False))
    if not duration_loc.empty:
        print(duration_loc[["location", "episodes", "duration_median_s", "duration_p90_s", "long_duration_ge_5s_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
