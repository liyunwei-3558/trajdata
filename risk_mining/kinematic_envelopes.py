#!/usr/bin/env python
"""Extract semantic kinematic envelopes from SinD trajectories.

This script reads SinD trajectory metadata, applies semantic slicing by city,
agent class and maneuver, smooths kinematics with Savitzky-Golay filtering, and
computes a 95% v-a envelope using 2D KDE. Outputs include envelope plots,
outlier samples, and cross-city overlap ratios.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde

try:
    from shapely.geometry import MultiPolygon, Polygon, mapping
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - shapely is optional at import time.
    MultiPolygon = None
    Polygon = None
    unary_union = None
    mapping = None


SIND_LOCATIONS: Tuple[str, ...] = ("cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR")
LOCATION_ALIASES: Mapping[str, str] = {
    "cc": "cc",
    "changchun": "cc",
    "changchun_pudong": "cc",
    "changchun pudong": "cc",
    "xa": "xa",
    "xian": "xa",
    "xi'an": "xa",
    "xi an": "xa",
    "xasl": "xasl",
    "xian_shanglin": "xasl",
    "xi'an_shanglin": "xasl",
    "xi an shanglin": "xasl",
    "tj": "tj",
    "tianjin": "tj",
    "cqir": "cqIR",
    "cq_ir": "cqIR",
    "chongqing_ir": "cqIR",
    "cqnr": "cqNR",
    "cq_nr": "cqNR",
    "chongqing_nr": "cqNR",
    "cqr": "cqR",
    "cq_r": "cqR",
    "chongqing_r": "cqR",
}
LOCATION_DISPLAY: Mapping[str, str] = {
    "cc": "ChangChun",
    "xa": "XiAn",
    "xasl": "XiAn-Shanglin",
    "tj": "TianJin",
    "cqIR": "ChongQing-IR",
    "cqNR": "ChongQing-NR",
    "cqR": "ChongQing-R",
}
AGENT_TYPE_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "car": ("car",),
    "vehicle": ("car", "truck", "bus", "tricycle"),
    "truck": ("truck",),
    "bus": ("bus",),
    "e-bike": ("bicycle", "motorcycle"),
    "ebike": ("bicycle", "motorcycle"),
    "e_bike": ("bicycle", "motorcycle"),
    "bicycle": ("bicycle",),
    "motorcycle": ("motorcycle",),
    "tricycle": ("tricycle",),
    "pedestrian": ("pedestrian",),
    "all": (),
}
STRAIGHT_PAIRS = {("w", "e"), ("e", "w"), ("n", "s"), ("s", "n")}
DIRECTION_VECTOR: Mapping[str, np.ndarray] = {
    "e": np.array([1.0, 0.0]),
    "w": np.array([-1.0, 0.0]),
    "n": np.array([0.0, 1.0]),
    "s": np.array([0.0, -1.0]),
}


@dataclass(frozen=True)
class TrackRecord:
    location: str
    scene_id: str
    agent_id: str
    class_name: str
    type_name: str
    maneuver: str
    direction: str
    state: pd.DataFrame


@dataclass
class EnvelopeResult:
    label: str
    points: pd.DataFrame
    kde: Any
    grid_x: np.ndarray
    grid_y: np.ndarray
    density_grid: np.ndarray
    density_threshold: float
    contour_segments: List[np.ndarray]
    polygon: Any = None


def normalize_location(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]
    if value in SIND_LOCATIONS:
        return value
    raise ValueError(f"Unknown SinD city/location: {value}")


def normalize_maneuver(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "all": None,
        "straight": "straight",
        "go-straight": "straight",
        "left": "left-turn",
        "left-turn": "left-turn",
        "leftturn": "left-turn",
        "right": "right-turn",
        "right-turn": "right-turn",
        "rightturn": "right-turn",
        "unknown": "unknown",
    }
    if key not in aliases:
        raise ValueError(f"Unknown maneuver: {value}")
    return aliases[key]


def normalize_agent_type(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    key = value.strip().lower().replace(" ", "-")
    if key not in AGENT_TYPE_GROUPS:
        raise ValueError(
            f"Unknown agent type: {value}. Supported: {', '.join(sorted(AGENT_TYPE_GROUPS))}"
        )
    classes = AGENT_TYPE_GROUPS[key]
    return None if not classes else classes


def _clean_direction_token(token: Any) -> Optional[str]:
    text = str(token).strip().lower()
    if not text or text == "nan" or text.startswith("nan"):
        return None
    first = text[0]
    return first if first in DIRECTION_VECTOR else None


def maneuver_from_direction_pair(direction_pair: Any) -> Optional[str]:
    """Infer straight/left/right from SinD x_y or cardinal-direction fields."""
    text = str(direction_pair).strip().lower()
    if not text or text == "nan" or "_" not in text:
        return None
    src_raw, dst_raw = text.split("_", 1)
    src = _clean_direction_token(src_raw)
    dst = _clean_direction_token(dst_raw)
    if src is None or dst is None or src == dst:
        return None
    if (src, dst) in STRAIGHT_PAIRS:
        return "straight"

    # src denotes the entry arm, so the initial motion points toward the junction.
    initial_vec = -DIRECTION_VECTOR[src]
    final_vec = DIRECTION_VECTOR[dst]
    cross = float(initial_vec[0] * final_vec[1] - initial_vec[1] * final_vec[0])
    if abs(cross) < 1e-6:
        return "straight"
    return "left-turn" if cross > 0 else "right-turn"


def maneuver_from_track_geometry(state_df: pd.DataFrame, straight_threshold_deg: float = 35.0) -> str:
    """Fallback maneuver inference for locations without x_y labels."""
    if state_df.empty or not {"x", "y"}.issubset(state_df.columns):
        return "unknown"
    xy = state_df[["x", "y"]].to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.shape[0] < 5:
        return "unknown"

    seg = max(2, min(8, xy.shape[0] // 5))
    start_vec = xy[seg] - xy[0]
    end_vec = xy[-1] - xy[-seg - 1]
    if np.linalg.norm(start_vec) < 0.5 or np.linalg.norm(end_vec) < 0.5:
        return "unknown"
    start_ang = math.atan2(start_vec[1], start_vec[0])
    end_ang = math.atan2(end_vec[1], end_vec[0])
    delta = math.atan2(math.sin(end_ang - start_ang), math.cos(end_ang - start_ang))
    if abs(math.degrees(delta)) < straight_threshold_deg:
        return "straight"
    return "left-turn" if delta > 0 else "right-turn"


def infer_maneuver(tp_data: Mapping[str, Any]) -> str:
    for key in ("x_y", "cardinal direction", "cardinal_direction"):
        maneuver = maneuver_from_direction_pair(tp_data.get(key, ""))
        if maneuver is not None:
            return maneuver
    state_df = tp_data.get("State")
    if isinstance(state_df, pd.DataFrame):
        return maneuver_from_track_geometry(state_df)
    return "unknown"


def load_sind_tracks(data_dir: Path, locations: Sequence[str]) -> List[TrackRecord]:
    tracks: List[TrackRecord] = []
    for location in locations:
        tp_path = data_dir / location / f"tp_info_{location}.pkl"
        if not tp_path.exists():
            raise FileNotFoundError(f"Missing SinD tp_info file: {tp_path}")
        with tp_path.open("rb") as handle:
            city_tp_info = pickle.load(handle)

        for scene_id, scene_tracks in city_tp_info.items():
            for raw_agent_id, tp_data in scene_tracks.items():
                state_df = tp_data.get("State")
                if not isinstance(state_df, pd.DataFrame) or state_df.empty:
                    continue
                class_name = str(tp_data.get("Class", tp_data.get("Type", "unknown"))).lower()
                type_name = str(tp_data.get("Type", "unknown")).lower()
                tracks.append(
                    TrackRecord(
                        location=location,
                        scene_id=str(scene_id),
                        agent_id=str(tp_data.get("ID", raw_agent_id)),
                        class_name=class_name,
                        type_name=type_name,
                        maneuver=infer_maneuver(tp_data),
                        direction=str(tp_data.get("x_y", tp_data.get("cardinal direction", ""))),
                        state=state_df,
                    )
                )
    return tracks


def filter_tracks(
    tracks: Iterable[TrackRecord],
    city: Optional[Sequence[str]] = None,
    agent_type: Optional[str] = None,
    maneuver: Optional[str] = None,
) -> List[TrackRecord]:
    """Return tracks matching City, AgentType and Maneuver semantic slices.

    Args:
        tracks: SinD tracks loaded by :func:`load_sind_tracks`.
        city: City/location names or aliases, e.g. ``["XiAn", "ChangChun"]``.
        agent_type: Class/group name, e.g. ``Car``, ``E-bike``, ``Pedestrian``.
        maneuver: ``Straight``, ``Left-turn`` or ``Right-turn``. ``None`` keeps all.
    """
    city_set = {normalize_location(item) for item in city} if city else None
    allowed_classes = normalize_agent_type(agent_type) if agent_type else None
    wanted_maneuver = normalize_maneuver(maneuver)

    filtered: List[TrackRecord] = []
    for track in tracks:
        if city_set is not None and track.location not in city_set:
            continue
        if allowed_classes is not None and track.class_name not in allowed_classes:
            continue
        if wanted_maneuver is not None and track.maneuver != wanted_maneuver:
            continue
        filtered.append(track)
    return filtered


def _valid_savgol_window(num_points: int, requested_window: int, polyorder: int) -> Optional[int]:
    if num_points <= polyorder + 2:
        return None
    window = min(requested_window, num_points if num_points % 2 == 1 else num_points - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return None
    if window % 2 == 0:
        window -= 1
    return max(window, min_window)


def smooth_track_kinematics(
    track: TrackRecord,
    dt: float = 0.1,
    window_sec: float = 0.7,
    polyorder: int = 3,
    min_track_points: int = 8,
) -> pd.DataFrame:
    """Compute smoothed speed and signed longitudinal acceleration samples."""
    df = track.state.sort_values("frame_id").copy()
    required = ["frame_id", "x", "y"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()
    df = df[np.isfinite(df[required].to_numpy(dtype=float)).all(axis=1)]
    if len(df) < min_track_points:
        return pd.DataFrame()

    frame_ids = df["frame_id"].to_numpy(dtype=int)
    timestamp_ms = (
        df["timestamp_ms"].to_numpy(dtype=float)
        if "timestamp_ms" in df.columns
        else frame_ids.astype(float) * dt * 1000.0
    )
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    requested_window = max(polyorder + 3, int(round(window_sec / dt)))
    if requested_window % 2 == 0:
        requested_window += 1
    window = _valid_savgol_window(len(df), requested_window, polyorder)

    if window is None:
        x_s = x
        y_s = y
        vx_s = np.gradient(x_s, dt)
        vy_s = np.gradient(y_s, dt)
        ax_s = np.gradient(vx_s, dt)
        ay_s = np.gradient(vy_s, dt)
    else:
        x_s = savgol_filter(x, window, polyorder, mode="interp")
        y_s = savgol_filter(y, window, polyorder, mode="interp")
        vx_s = savgol_filter(x, window, polyorder, deriv=1, delta=dt, mode="interp")
        vy_s = savgol_filter(y, window, polyorder, deriv=1, delta=dt, mode="interp")
        ax_s = savgol_filter(x, window, polyorder, deriv=2, delta=dt, mode="interp")
        ay_s = savgol_filter(y, window, polyorder, deriv=2, delta=dt, mode="interp")

    speed = np.hypot(vx_s, vy_s)
    accel = np.divide(
        vx_s * ax_s + vy_s * ay_s,
        np.maximum(speed, 1e-3),
        out=np.gradient(speed, dt),
        where=speed > 1e-3,
    )
    speed_window = _valid_savgol_window(len(df), requested_window, min(polyorder, 2))
    if speed_window is not None:
        speed = savgol_filter(speed, speed_window, min(polyorder, 2), mode="interp")
        accel = savgol_filter(accel, speed_window, min(polyorder, 2), mode="interp")

    result = pd.DataFrame(
        {
            "location": track.location,
            "city": LOCATION_DISPLAY.get(track.location, track.location),
            "scene_id": track.scene_id,
            "agent_id": track.agent_id,
            "class_name": track.class_name,
            "type_name": track.type_name,
            "maneuver": track.maneuver,
            "direction": track.direction,
            "frame_id": frame_ids,
            "timestamp_ms": timestamp_ms,
            "time_s": timestamp_ms / 1000.0,
            "x": x_s,
            "y": y_s,
            "v": speed,
            "a": accel,
        }
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["v", "a"])


def tracks_to_va_points(
    tracks: Sequence[TrackRecord],
    dt: float,
    smooth_window_sec: float,
    polyorder: int,
    min_track_points: int,
    max_tracks_per_city: Optional[int] = None,
) -> pd.DataFrame:
    if max_tracks_per_city is not None:
        limited: List[TrackRecord] = []
        counts: Dict[str, int] = {}
        for track in tracks:
            count = counts.get(track.location, 0)
            if count >= max_tracks_per_city:
                continue
            limited.append(track)
            counts[track.location] = count + 1
        tracks = limited

    frames = [
        smooth_track_kinematics(track, dt, smooth_window_sec, polyorder, min_track_points)
        for track in tracks
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    points = pd.concat(frames, ignore_index=True)
    return points[np.isfinite(points[["v", "a"]].to_numpy()).all(axis=1)].reset_index(drop=True)


def robust_axis_limits(values: np.ndarray, quantile: float = 0.995, pad_ratio: float = 0.08) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(finite, 1.0 - quantile))
    hi = float(np.quantile(finite, quantile))
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def sample_for_kde(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def contours_to_polygon(contour_segments: Sequence[np.ndarray]) -> Any:
    if Polygon is None or unary_union is None:
        return None
    polygons = []
    for segment in contour_segments:
        if segment.shape[0] < 3:
            continue
        polygon = Polygon(segment)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    if not polygons:
        return None
    return unary_union(polygons)


def compute_kde_envelope(
    points_df: pd.DataFrame,
    label: str,
    grid_size: int = 180,
    mass: float = 0.95,
    max_kde_points: int = 20000,
    seed: int = 7,
    axis_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> EnvelopeResult:
    points = points_df[["v", "a"]].to_numpy(dtype=float)
    if points.shape[0] < 20:
        raise ValueError(f"Not enough points for KDE envelope: {label} ({points.shape[0]})")
    rng = np.random.default_rng(seed)
    kde_points = sample_for_kde(points, max_kde_points, rng)

    if np.linalg.matrix_rank(np.cov(kde_points.T)) < 2:
        jitter = rng.normal(scale=[0.03, 0.03], size=kde_points.shape)
        kde_points = kde_points + jitter

    kde = gaussian_kde(kde_points.T)
    if axis_limits is None:
        v_lim = robust_axis_limits(points[:, 0])
        a_lim = robust_axis_limits(points[:, 1])
    else:
        v_lim, a_lim = axis_limits
    grid_x, grid_y = np.meshgrid(
        np.linspace(v_lim[0], v_lim[1], grid_size),
        np.linspace(a_lim[0], a_lim[1], grid_size),
    )
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density_grid = kde(grid_coords).reshape(grid_x.shape)
    density_values = density_grid.ravel()
    order = np.argsort(density_values)[::-1]
    cumulative = np.cumsum(density_values[order])
    cumulative /= cumulative[-1]
    idx = min(int(np.searchsorted(cumulative, mass)), len(order) - 1)
    density_threshold = float(density_values[order[idx]])

    fig, ax = plt.subplots()
    contour = ax.contour(grid_x, grid_y, density_grid, levels=[density_threshold])
    contour_segments = [seg.copy() for seg in contour.allsegs[0] if len(seg) >= 3]
    plt.close(fig)

    return EnvelopeResult(
        label=label,
        points=points_df,
        kde=kde,
        grid_x=grid_x,
        grid_y=grid_y,
        density_grid=density_grid,
        density_threshold=density_threshold,
        contour_segments=contour_segments,
        polygon=contours_to_polygon(contour_segments),
    )


def evaluate_outliers(result: EnvelopeResult, chunk_size: int = 50000) -> pd.DataFrame:
    """Flag samples outside the 95% contour using the KDE grid density.

    Direct KDE evaluation is O(num_kde_samples * num_points) and becomes slow for
    full SinD city-scale exports. Interpolating the already-computed grid keeps
    outlier detection consistent with the plotted contour and runs in linear time.
    """
    points = result.points[["v", "a"]].to_numpy(dtype=float)
    v_axis = result.grid_x[0, :]
    a_axis = result.grid_y[:, 0]
    interpolator = RegularGridInterpolator(
        (a_axis, v_axis),
        result.density_grid,
        bounds_error=False,
        fill_value=0.0,
    )
    densities = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        densities[start:end] = interpolator(np.column_stack([points[start:end, 1], points[start:end, 0]]))
    outliers = result.points.loc[densities < result.density_threshold].copy()
    outliers["density"] = densities[densities < result.density_threshold]
    outliers["density_threshold"] = result.density_threshold
    return outliers


def plot_envelope(result: EnvelopeResult, output_path: Path, title: str, scatter_limit: int = 20000) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    points = result.points
    if len(points) > scatter_limit:
        points = points.sample(scatter_limit, random_state=11)
    ax.scatter(points["v"], points["a"], s=2, alpha=0.12, color="#3b5360", label="samples")
    for idx, segment in enumerate(result.contour_segments):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#c33a2b",
            linewidth=2.0,
            label="95% KDE envelope" if idx == 0 else None,
        )
    ax.set_xlabel("Speed v (m/s)")
    ax.set_ylabel("Longitudinal acceleration a (m/s^2)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def overlap_ratio(poly_a: Any, poly_b: Any) -> float:
    if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
        return float("nan")
    union_area = poly_a.union(poly_b).area
    if union_area <= 0:
        return float("nan")
    return float(poly_a.intersection(poly_b).area / union_area)


def plot_cross_city_envelopes(
    results: Mapping[str, EnvelopeResult], output_path: Path, title: str
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    for (city, result), color in zip(results.items(), colors):
        for idx, segment in enumerate(result.contour_segments):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                linewidth=2.2,
                color=color,
                label=LOCATION_DISPLAY.get(city, city) if idx == 0 else None,
            )
    ax.set_xlabel("Speed v (m/s)")
    ax.set_ylabel("Longitudinal acceleration a (m/s^2)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    rows = []
    cities = list(results.keys())
    for i, city_a in enumerate(cities):
        for city_b in cities[i + 1 :]:
            rows.append(
                {
                    "city_a": city_a,
                    "city_b": city_b,
                    "city_a_display": LOCATION_DISPLAY.get(city_a, city_a),
                    "city_b_display": LOCATION_DISPLAY.get(city_b, city_b),
                    "overlap_ratio_intersection_over_union": overlap_ratio(
                        results[city_a].polygon, results[city_b].polygon
                    ),
                    "area_a": float(results[city_a].polygon.area)
                    if results[city_a].polygon is not None
                    else float("nan"),
                    "area_b": float(results[city_b].polygon.area)
                    if results[city_b].polygon is not None
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def export_envelopes_geojson(results: Mapping[str, EnvelopeResult], output_path: Path) -> None:
    if mapping is None:
        return
    features = []
    for city, result in results.items():
        if result.polygon is None or result.polygon.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "city": city,
                    "city_display": LOCATION_DISPLAY.get(city, city),
                    "label": result.label,
                    "density_threshold": result.density_threshold,
                    "area": float(result.polygon.area),
                },
                "geometry": mapping(result.polygon),
            }
        )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=2)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    city_track_counts: Mapping[str, int],
    city_point_counts: Mapping[str, int],
    results: Mapping[str, EnvelopeResult],
    outlier_counts: Mapping[str, int],
) -> None:
    summary = {
        "data_dir": str(args.data_dir),
        "cities": list(city_track_counts.keys()),
        "agent_type": args.agent_type,
        "maneuver": args.maneuver,
        "dt": args.dt,
        "smooth_window_sec": args.smooth_window_sec,
        "kde_mass": args.mass,
        "track_counts": dict(city_track_counts),
        "point_counts": dict(city_point_counts),
        "outlier_counts": dict(outlier_counts),
        "envelopes": {
            city: {
                "density_threshold": result.density_threshold,
                "area": float(result.polygon.area) if result.polygon is not None else None,
                "num_contours": len(result.contour_segments),
            }
            for city, result in results.items()
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 95% semantic kinematic envelopes from SinD trajectories."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_kinematic_envelopes"))
    parser.add_argument("--cities", nargs="+", default=list(SIND_LOCATIONS), help="SinD city/location aliases.")
    parser.add_argument("--agent-type", default="Car", help="Car, E-bike, Pedestrian, Vehicle, etc.")
    parser.add_argument("--maneuver", default="Straight", help="Straight, Left-turn, Right-turn, All.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--smooth-window-sec", type=float, default=0.7)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--mass", type=float, default=0.95)
    parser.add_argument("--max-kde-points", type=int, default=20000)
    parser.add_argument("--max-tracks-per-city", type=int, default=None)
    parser.add_argument("--write-points", action="store_true", help="Also export all smoothed v-a samples.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locations = [normalize_location(city) for city in args.cities]
    tracks = load_sind_tracks(args.data_dir, locations)
    sliced_tracks = filter_tracks(tracks, city=locations, agent_type=args.agent_type, maneuver=args.maneuver)
    if not sliced_tracks:
        raise RuntimeError("No tracks matched the requested semantic slice.")

    city_track_counts = {city: 0 for city in locations}
    for track in sliced_tracks:
        city_track_counts[track.location] = city_track_counts.get(track.location, 0) + 1

    points = tracks_to_va_points(
        sliced_tracks,
        dt=args.dt,
        smooth_window_sec=args.smooth_window_sec,
        polyorder=args.polyorder,
        min_track_points=args.min_track_points,
        max_tracks_per_city=args.max_tracks_per_city,
    )
    if points.empty:
        raise RuntimeError("Semantic slice matched tracks, but no usable kinematic samples were produced.")

    if args.write_points:
        points.to_csv(args.output_dir / "smoothed_va_points.csv", index=False)

    axis_limits = (
        robust_axis_limits(points["v"].to_numpy(), quantile=0.997),
        robust_axis_limits(points["a"].to_numpy(), quantile=0.997),
    )

    results: Dict[str, EnvelopeResult] = {}
    outlier_frames = []
    outlier_counts: Dict[str, int] = {}
    city_point_counts: Dict[str, int] = {}
    for city, city_points in points.groupby("location", sort=False):
        if len(city_points) < 50:
            print(f"Skipping {city}: only {len(city_points)} points after filtering", file=sys.stderr)
            continue
        label = f"{LOCATION_DISPLAY.get(city, city)} | {args.agent_type} | {args.maneuver}"
        result = compute_kde_envelope(
            city_points.reset_index(drop=True),
            label=label,
            grid_size=args.grid_size,
            mass=args.mass,
            max_kde_points=args.max_kde_points,
            axis_limits=axis_limits,
        )
        results[city] = result
        city_point_counts[city] = len(city_points)
        plot_envelope(
            result,
            args.output_dir / f"envelope_{city}.png",
            title=f"95% KDE Kinematic Envelope - {label}",
        )
        outliers = evaluate_outliers(result)
        outlier_counts[city] = len(outliers)
        outlier_frames.append(outliers)

    if not results:
        raise RuntimeError("No city had enough points to compute a KDE envelope.")

    outlier_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    outlier_df.to_csv(args.output_dir / "outliers.csv", index=False)
    if not outlier_df.empty:
        track_summary = (
            outlier_df.groupby(["location", "city", "scene_id", "agent_id", "class_name", "maneuver"], as_index=False)
            .agg(
                outlier_sample_count=("frame_id", "size"),
                first_outlier_frame=("frame_id", "min"),
                last_outlier_frame=("frame_id", "max"),
                min_density=("density", "min"),
                max_abs_accel=("a", lambda values: float(np.max(np.abs(values)))),
                max_speed=("v", "max"),
            )
            .sort_values(["outlier_sample_count", "max_abs_accel"], ascending=False)
        )
    else:
        track_summary = pd.DataFrame(
            columns=[
                "location",
                "city",
                "scene_id",
                "agent_id",
                "class_name",
                "maneuver",
                "outlier_sample_count",
                "first_outlier_frame",
                "last_outlier_frame",
                "min_density",
                "max_abs_accel",
                "max_speed",
            ]
        )
    track_summary.to_csv(args.output_dir / "outlier_tracks.csv", index=False)

    overlap_df = plot_cross_city_envelopes(
        results,
        args.output_dir / "cross_city_envelopes.png",
        title=f"Cross-City 95% Envelopes | {args.agent_type} | {args.maneuver}",
    )
    overlap_df.to_csv(args.output_dir / "overlap_ratios.csv", index=False)
    export_envelopes_geojson(results, args.output_dir / "envelopes.geojson")
    write_summary(
        args.output_dir / "summary.json",
        args,
        city_track_counts=city_track_counts,
        city_point_counts=city_point_counts,
        results=results,
        outlier_counts=outlier_counts,
    )

    print(f"Processed {len(sliced_tracks)} tracks and {len(points)} v-a samples.")
    print(f"Wrote outputs to: {args.output_dir}")
    if not overlap_df.empty:
        print(overlap_df[["city_a", "city_b", "overlap_ratio_intersection_over_union"]].to_string(index=False))


if __name__ == "__main__":
    main()
