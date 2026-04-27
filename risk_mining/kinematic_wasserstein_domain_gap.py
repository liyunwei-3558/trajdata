#!/usr/bin/env python
"""Quantify SinD cross-domain kinematic gaps with Wasserstein distance.

The script samples smoothed (v, a) points from semantic slices and computes both
2D joint Wasserstein distances and 1D decomposed speed/acceleration distances for
SinD locations and merged city groups.
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
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

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
    smooth_track_kinematics,
)
from lateral_deviation_variance import DEFAULT_SIX_LOCATIONS  # noqa: E402

CITY_GROUPS = {
    "cc": "ChangChun",
    "tj": "TianJin",
    "cqIR": "ChongQing",
    "cqNR": "ChongQing",
    "cqR": "ChongQing",
    "xasl": "XiAn",
}
DEFAULT_CITY_GROUP_ORDER = ("ChangChun", "TianJin", "ChongQing", "XiAn")


@dataclass
class DomainSample:
    name: str
    display_name: str
    level: str
    total_points: int
    track_count: int
    reservoir: np.ndarray
    sample_count: int = 0
    seen_points: int = 0

    def add_points(self, points: np.ndarray, rng: np.random.Generator) -> None:
        if points.size == 0:
            return
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return
        self.total_points += int(len(points))
        capacity = len(self.reservoir)
        if capacity <= 0:
            self.seen_points += int(len(points))
            return
        for point in points:
            self.seen_points += 1
            if self.sample_count < capacity:
                self.reservoir[self.sample_count] = point
                self.sample_count += 1
            else:
                j = int(rng.integers(0, self.seen_points))
                if j < capacity:
                    self.reservoir[j] = point

    @property
    def samples(self) -> np.ndarray:
        return self.reservoir[: self.sample_count].copy()


def city_group(location: str) -> str:
    return CITY_GROUPS.get(location, LOCATION_DISPLAY.get(location, location))


def limit_tracks_per_city(tracks: Sequence[TrackRecord], max_tracks_per_city: Optional[int]) -> List[TrackRecord]:
    if max_tracks_per_city is None:
        return list(tracks)
    counts: Dict[str, int] = {}
    limited: List[TrackRecord] = []
    for track in tracks:
        count = counts.get(track.location, 0)
        if count >= max_tracks_per_city:
            continue
        limited.append(track)
        counts[track.location] = count + 1
    return limited


def init_domains(locations: Sequence[str], reservoir_size: int) -> Tuple[Dict[str, DomainSample], Dict[str, DomainSample]]:
    loc_domains = {
        loc: DomainSample(
            name=loc,
            display_name=LOCATION_DISPLAY.get(loc, loc),
            level="location",
            total_points=0,
            track_count=0,
            reservoir=np.empty((reservoir_size, 2), dtype=float),
        )
        for loc in locations
    }
    group_names = [group for group in DEFAULT_CITY_GROUP_ORDER if any(city_group(loc) == group for loc in locations)]
    group_domains = {
        group: DomainSample(
            name=group,
            display_name=group,
            level="city_group",
            total_points=0,
            track_count=0,
            reservoir=np.empty((reservoir_size, 2), dtype=float),
        )
        for group in group_names
    }
    return loc_domains, group_domains


def collect_domain_samples(
    tracks: Sequence[TrackRecord],
    locations: Sequence[str],
    dt: float,
    smooth_window_sec: float,
    polyorder: int,
    min_track_points: int,
    reservoir_size: int,
    seed: int,
) -> Tuple[Dict[str, DomainSample], Dict[str, DomainSample]]:
    loc_domains, group_domains = init_domains(locations, reservoir_size)
    rng_by_domain = {f"loc:{loc}": np.random.default_rng(seed + idx * 17) for idx, loc in enumerate(loc_domains)}
    rng_by_domain.update({f"group:{name}": np.random.default_rng(seed + 1000 + idx * 29) for idx, name in enumerate(group_domains)})

    for idx, track in enumerate(tracks, start=1):
        points_df = smooth_track_kinematics(track, dt=dt, window_sec=smooth_window_sec, polyorder=polyorder, min_track_points=min_track_points)
        if points_df.empty:
            continue
        points = points_df[["v", "a"]].to_numpy(dtype=float)
        loc_domain = loc_domains.get(track.location)
        if loc_domain is not None:
            loc_domain.track_count += 1
            loc_domain.add_points(points, rng_by_domain[f"loc:{track.location}"])
        group = city_group(track.location)
        group_domain = group_domains.get(group)
        if group_domain is not None:
            group_domain.track_count += 1
            group_domain.add_points(points, rng_by_domain[f"group:{group}"])
        if idx % 5000 == 0:
            print(f"Smoothed {idx:,}/{len(tracks):,} tracks...", flush=True)
    return loc_domains, group_domains


def robust_normalization(domains: Mapping[str, DomainSample]) -> Dict[str, float]:
    samples = [domain.samples for domain in domains.values() if domain.sample_count]
    if not samples:
        return {"v_center": 0.0, "a_center": 0.0, "v_scale": 1.0, "a_scale": 1.0}
    all_points = np.vstack(samples)
    v = all_points[:, 0]
    a = all_points[:, 1]
    v_q25, v_q75 = np.quantile(v, [0.25, 0.75])
    a_q25, a_q75 = np.quantile(a, [0.25, 0.75])
    return {
        "v_center": float(np.median(v)),
        "a_center": float(np.median(a)),
        "v_scale": float(max(v_q75 - v_q25, np.std(v), 1e-6)),
        "a_scale": float(max(a_q75 - a_q25, np.std(a), 1e-6)),
    }


def normalize_points(points: np.ndarray, stats: Mapping[str, float]) -> np.ndarray:
    result = points.copy()
    result[:, 0] = (result[:, 0] - stats["v_center"]) / stats["v_scale"]
    result[:, 1] = (result[:, 1] - stats["a_center"]) / stats["a_scale"]
    return result


def choose_sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def compute_pairwise_distances(
    domains: Mapping[str, DomainSample],
    order: Sequence[str],
    norm_stats: Mapping[str, float],
    max_emd_points: int,
    max_1d_points: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(order)
    matrix_2d = pd.DataFrame(np.zeros((n, n), dtype=float), index=order, columns=order)
    matrix_v = pd.DataFrame(np.zeros((n, n), dtype=float), index=order, columns=order)
    matrix_a = pd.DataFrame(np.zeros((n, n), dtype=float), index=order, columns=order)
    rows: List[Dict[str, Any]] = []

    sampled_raw: Dict[str, np.ndarray] = {}
    sampled_norm: Dict[str, np.ndarray] = {}
    sampled_1d: Dict[str, np.ndarray] = {}
    for i, name in enumerate(order):
        raw = domains[name].samples
        sampled_raw[name] = choose_sample(raw, max_emd_points, seed + i * 31)
        sampled_norm[name] = normalize_points(sampled_raw[name], norm_stats)
        sampled_1d[name] = choose_sample(raw, max_1d_points, seed + i * 43 + 100)

    for i, a_name in enumerate(order):
        for j, b_name in enumerate(order):
            if j < i:
                continue
            if i == j:
                w2 = w2_raw = wv = wa = wv_raw = wa_raw = 0.0
            else:
                a_norm = sampled_norm[a_name]
                b_norm = sampled_norm[b_name]
                a_raw_2d = sampled_raw[a_name]
                b_raw_2d = sampled_raw[b_name]
                a_1d = sampled_1d[a_name]
                b_1d = sampled_1d[b_name]
                w2 = float(wasserstein_distance_nd(a_norm, b_norm))
                w2_raw = float(wasserstein_distance_nd(a_raw_2d, b_raw_2d))
                wv_raw = float(wasserstein_distance(a_1d[:, 0], b_1d[:, 0]))
                wa_raw = float(wasserstein_distance(a_1d[:, 1], b_1d[:, 1]))
                wv = float(wv_raw / norm_stats["v_scale"])
                wa = float(wa_raw / norm_stats["a_scale"])
            matrix_2d.loc[a_name, b_name] = matrix_2d.loc[b_name, a_name] = w2
            matrix_v.loc[a_name, b_name] = matrix_v.loc[b_name, a_name] = wv
            matrix_a.loc[a_name, b_name] = matrix_a.loc[b_name, a_name] = wa
            rows.append(
                {
                    "domain_a": a_name,
                    "domain_b": b_name,
                    "wasserstein_2d_normalized": w2,
                    "wasserstein_2d_raw": w2_raw,
                    "wasserstein_v_normalized": wv,
                    "wasserstein_a_normalized": wa,
                    "wasserstein_v_raw_mps": wv_raw,
                    "wasserstein_a_raw_mps2": wa_raw,
                    "sample_a_2d": int(len(sampled_raw[a_name])),
                    "sample_b_2d": int(len(sampled_raw[b_name])),
                    "sample_a_1d": int(len(sampled_1d[a_name])),
                    "sample_b_1d": int(len(sampled_1d[b_name])),
                }
            )
    return matrix_2d, matrix_v, matrix_a, pd.DataFrame(rows)


def domain_summary(domains: Mapping[str, DomainSample], order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name in order:
        domain = domains[name]
        samples = domain.samples
        rows.append(
            {
                "level": domain.level,
                "domain": name,
                "display_name": domain.display_name,
                "track_count": int(domain.track_count),
                "total_points": int(domain.total_points),
                "sample_points": int(domain.sample_count),
                "speed_mean_mps": float(samples[:, 0].mean()) if len(samples) else np.nan,
                "speed_median_mps": float(np.median(samples[:, 0])) if len(samples) else np.nan,
                "accel_mean_mps2": float(samples[:, 1].mean()) if len(samples) else np.nan,
                "accel_median_mps2": float(np.median(samples[:, 1])) if len(samples) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path, title: str, cmap: str = "YlOrRd") -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 6.6), dpi=170)
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-9))
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] > vmax * 0.55 else "#1d2326"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Wasserstein distance")
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
            if isinstance(value, float):
                cell = "" if math.isnan(value) else f"{value:.4f}"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html_report(output_path: Path, city_pairs: pd.DataFrame, loc_pairs: pd.DataFrame, sample_summary: pd.DataFrame) -> None:
    top_city = city_pairs[city_pairs["domain_a"] != city_pairs["domain_b"]].sort_values("wasserstein_2d_normalized", ascending=False).head(8)
    top_loc = loc_pairs[loc_pairs["domain_a"] != loc_pairs["domain_b"]].sort_values("wasserstein_2d_normalized", ascending=False).head(10)
    city_table = _html_table(
        top_city.to_dict("records"),
        [("domain_a", "Domain A"), ("domain_b", "Domain B"), ("wasserstein_2d_normalized", "W2D norm"), ("wasserstein_v_raw_mps", "Wv m/s"), ("wasserstein_a_raw_mps2", "Wa m/s2")],
    )
    loc_table = _html_table(
        top_loc.to_dict("records"),
        [("domain_a", "Location A"), ("domain_b", "Location B"), ("wasserstein_2d_normalized", "W2D norm"), ("wasserstein_v_raw_mps", "Wv m/s"), ("wasserstein_a_raw_mps2", "Wa m/s2")],
    )
    sample_table = _html_table(
        sample_summary.to_dict("records"),
        [("level", "Level"), ("domain", "Domain"), ("track_count", "Tracks"), ("total_points", "v-a points"), ("sample_points", "Sampled")],
    )
    html_text = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>SinD Kinematic Wasserstein Domain Gap</title>
  <style>
    :root {{ --ink:#1d2326; --paper:#f4eadb; --card:#fffaf0; --line:#d8c5a6; --muted:#657074; --red:#b64032; --blue:#375c6c; }}
    body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f4eadb,#e8eeee); }}
    header {{ padding:44px 5vw 74px; color:#fff; background:radial-gradient(circle at 82% 12%,#d99743,transparent 25%), linear-gradient(135deg,#263f47,#6b3329); }}
    header h1 {{ margin:0; font-size:clamp(32px,5vw,58px); }} header p {{ max-width:1040px; line-height:1.65; color:#f7ead8; font-size:17px; }}
    main {{ padding:0 5vw 58px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:18px; margin-top:-42px; }}
    .panel {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 30px rgba(29,35,38,.08); }}
    section {{ margin-top:34px; }} h2 {{ font-size:30px; }} img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:#fff; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; background:rgba(255,250,240,.96); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #eadcc8; text-align:left; font-size:14px; }} th {{ background:#ead6bd; }} tr:hover td {{ background:#fff0d8; }}
    .links a {{ display:inline-block; margin:6px 8px 6px 0; padding:9px 12px; background:#fffaf0; border:1px solid var(--line); border-radius:999px; color:var(--blue); text-decoration:none; }}
  </style>
</head>
<body>
<header>
  <h1>Kinematic Wasserstein Domain Gap</h1>
  <p>使用 Wasserstein / Earth Mover's Distance 严格量化城市间 v-a 联合分布差异。主图为 robust-normalized 2D Wasserstein，补充图展示速度和加速度的一维分解距离。</p>
</header>
<main>
  <div class='grid'>
    <div class='panel'><h2>4x4 City 2D EMD</h2><a href='wasserstein_heatmap_city_2d.png'><img src='wasserstein_heatmap_city_2d.png'></a></div>
    <div class='panel'><h2>6x6 Location 2D EMD</h2><a href='wasserstein_heatmap_location_2d.png'><img src='wasserstein_heatmap_location_2d.png'></a></div>
    <div class='panel'><h2>City Speed Gap</h2><a href='wasserstein_heatmap_city_velocity.png'><img src='wasserstein_heatmap_city_velocity.png'></a></div>
    <div class='panel'><h2>City Acceleration Gap</h2><a href='wasserstein_heatmap_city_acceleration.png'><img src='wasserstein_heatmap_city_acceleration.png'></a></div>
  </div>
  <section><h2>Largest City Domain Gaps</h2>{city_table}</section>
  <section><h2>Largest Location Domain Gaps</h2>{loc_table}</section>
  <section><h2>Sample Summary</h2>{sample_table}</section>
  <section class='links'><h2>Artifacts</h2>
    <a href='wasserstein_2d_city_matrix.csv'>wasserstein_2d_city_matrix.csv</a>
    <a href='wasserstein_2d_location_matrix.csv'>wasserstein_2d_location_matrix.csv</a>
    <a href='wasserstein_1d_city_pairs.csv'>wasserstein_1d_city_pairs.csv</a>
    <a href='wasserstein_1d_location_pairs.csv'>wasserstein_1d_location_pairs.csv</a>
    <a href='wasserstein_city_pairs_long.csv'>wasserstein_city_pairs_long.csv</a>
    <a href='wasserstein_location_pairs_long.csv'>wasserstein_location_pairs_long.csv</a>
    <a href='domain_sample_summary.csv'>domain_sample_summary.csv</a>
    <a href='summary.json'>summary.json</a>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def split_1d_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "domain_a",
        "domain_b",
        "wasserstein_v_normalized",
        "wasserstein_a_normalized",
        "wasserstein_v_raw_mps",
        "wasserstein_a_raw_mps2",
        "sample_a_1d",
        "sample_b_1d",
    ]
    return pair_df[cols].copy()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Wasserstein domain gaps between SinD kinematic v-a distributions.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_kinematic_wasserstein_domain_gap"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_SIX_LOCATIONS))
    parser.add_argument("--agent-type", default="All")
    parser.add_argument("--maneuver", default="All")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--smooth-window-sec", type=float, default=0.7)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--max-tracks-per-city", type=int, default=None)
    parser.add_argument("--max-emd-points-per-domain", type=int, default=300)
    parser.add_argument("--max-1d-points-per-domain", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cities = [normalize_location(city) for city in args.cities]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reservoir_size = max(args.max_emd_points_per_domain, args.max_1d_points_per_domain)
    print(f"Loading tracks for {args.cities} | {args.agent_type} | {args.maneuver}...", flush=True)
    tracks = load_sind_tracks(args.data_dir, args.cities)
    tracks = filter_tracks(tracks, city=args.cities, agent_type=args.agent_type, maneuver=args.maneuver)
    tracks = limit_tracks_per_city(tracks, args.max_tracks_per_city)
    if not tracks:
        raise RuntimeError("No tracks matched the requested semantic slice.")

    print(f"Collecting reservoir samples from {len(tracks):,} tracks...", flush=True)
    loc_domains, group_domains = collect_domain_samples(
        tracks,
        args.cities,
        dt=args.dt,
        smooth_window_sec=args.smooth_window_sec,
        polyorder=args.polyorder,
        min_track_points=args.min_track_points,
        reservoir_size=reservoir_size,
        seed=args.seed,
    )
    loc_order = [loc for loc in args.cities if loc_domains[loc].sample_count > 0]
    city_order = [group for group in DEFAULT_CITY_GROUP_ORDER if group in group_domains and group_domains[group].sample_count > 0]
    if len(loc_order) < 2 or len(city_order) < 2:
        raise RuntimeError("Need at least two populated domains for Wasserstein distance.")

    norm_stats = robust_normalization({name: loc_domains[name] for name in loc_order})
    print("Computing location Wasserstein matrices...", flush=True)
    loc_matrix_2d, loc_matrix_v, loc_matrix_a, loc_pairs = compute_pairwise_distances(
        loc_domains,
        loc_order,
        norm_stats,
        max_emd_points=args.max_emd_points_per_domain,
        max_1d_points=args.max_1d_points_per_domain,
        seed=args.seed,
    )
    print("Computing city-group Wasserstein matrices...", flush=True)
    city_matrix_2d, city_matrix_v, city_matrix_a, city_pairs = compute_pairwise_distances(
        group_domains,
        city_order,
        norm_stats,
        max_emd_points=args.max_emd_points_per_domain,
        max_1d_points=args.max_1d_points_per_domain,
        seed=args.seed + 5000,
    )

    loc_matrix_2d.to_csv(args.output_dir / "wasserstein_2d_location_matrix.csv")
    city_matrix_2d.to_csv(args.output_dir / "wasserstein_2d_city_matrix.csv")
    loc_matrix_v.to_csv(args.output_dir / "wasserstein_velocity_location_matrix.csv")
    loc_matrix_a.to_csv(args.output_dir / "wasserstein_acceleration_location_matrix.csv")
    city_matrix_v.to_csv(args.output_dir / "wasserstein_velocity_city_matrix.csv")
    city_matrix_a.to_csv(args.output_dir / "wasserstein_acceleration_city_matrix.csv")
    loc_pairs.to_csv(args.output_dir / "wasserstein_location_pairs_long.csv", index=False)
    city_pairs.to_csv(args.output_dir / "wasserstein_city_pairs_long.csv", index=False)
    split_1d_pairs(loc_pairs).to_csv(args.output_dir / "wasserstein_1d_location_pairs.csv", index=False)
    split_1d_pairs(city_pairs).to_csv(args.output_dir / "wasserstein_1d_city_pairs.csv", index=False)
    sample_summary = pd.concat([domain_summary(loc_domains, loc_order), domain_summary(group_domains, city_order)], ignore_index=True)
    sample_summary.to_csv(args.output_dir / "domain_sample_summary.csv", index=False)

    plot_heatmap(city_matrix_2d, args.output_dir / "wasserstein_heatmap_city_2d.png", "4x4 City Kinematic Domain Gap | 2D Wasserstein")
    plot_heatmap(loc_matrix_2d, args.output_dir / "wasserstein_heatmap_location_2d.png", "6x6 Location Kinematic Domain Gap | 2D Wasserstein")
    plot_heatmap(city_matrix_v, args.output_dir / "wasserstein_heatmap_city_velocity.png", "4x4 City Speed Distribution Gap | W(v)", cmap="YlGnBu")
    plot_heatmap(city_matrix_a, args.output_dir / "wasserstein_heatmap_city_acceleration.png", "4x4 City Acceleration Distribution Gap | W(a)", cmap="OrRd")

    summary = {
        "data_dir": str(args.data_dir),
        "cities": args.cities,
        "city_groups": city_order,
        "agent_type": args.agent_type,
        "maneuver": args.maneuver,
        "dt": args.dt,
        "smooth_window_sec": args.smooth_window_sec,
        "max_emd_points_per_domain": args.max_emd_points_per_domain,
        "max_1d_points_per_domain": args.max_1d_points_per_domain,
        "normalization": norm_stats,
        "location_domains": domain_summary(loc_domains, loc_order).to_dict("records"),
        "city_group_domains": domain_summary(group_domains, city_order).to_dict("records"),
        "city_pairs": city_pairs.to_dict("records"),
        "location_pairs": loc_pairs.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html_report(args.output_dir / "index.html", city_pairs, loc_pairs, sample_summary)

    print(f"Wrote outputs to: {args.output_dir}")
    print(city_matrix_2d.round(4).to_string())


if __name__ == "__main__":
    main()
