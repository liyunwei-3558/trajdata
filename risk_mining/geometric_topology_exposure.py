#!/usr/bin/env python
"""Relate SinD intersection geometry to exposure-cost metrics."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCATION_ORDER: Tuple[str, ...] = ("cc", "tj", "cqIR", "cqNR", "cqR", "xasl")
LOCATION_DISPLAY: Mapping[str, str] = {
    "cc": "ChangChun",
    "tj": "TianJin",
    "cqIR": "ChongQing-IR",
    "cqNR": "ChongQing-NR",
    "cqR": "ChongQing-R",
    "xasl": "XiAn-Shanglin",
}
GEOMETRY_ROWS: Tuple[Mapping[str, Any], ...] = (
    {"location": "cc", "conflict_area_m2": 691.912, "intersection_angle_deg": 90.0, "angle_note": "approx_90"},
    {"location": "cqIR", "conflict_area_m2": 1387.245, "intersection_angle_deg": 77.5, "angle_note": "measured"},
    {"location": "cqNR", "conflict_area_m2": 1099.427, "intersection_angle_deg": 91.4, "angle_note": "measured"},
    {"location": "cqR", "conflict_area_m2": 2422.354, "intersection_angle_deg": 86.8, "angle_note": "measured"},
    {"location": "tj", "conflict_area_m2": 1028.819, "intersection_angle_deg": 90.0, "angle_note": "approx_90"},
    {"location": "xasl", "conflict_area_m2": 2593.371, "intersection_angle_deg": 86.2, "angle_note": "measured"},
)


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


def _base_geometry_df() -> pd.DataFrame:
    df = pd.DataFrame(GEOMETRY_ROWS)
    df["city"] = df["location"].map(LOCATION_DISPLAY)
    df["angle_skew_deg"] = (90.0 - df["intersection_angle_deg"]).abs()
    df["location_order"] = df["location"].map({loc: idx for idx, loc in enumerate(LOCATION_ORDER)})
    return df.sort_values("location_order").drop(columns=["location_order"]).reset_index(drop=True)


def _residence_metrics(track_residence: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for location, group in track_residence.groupby("location"):
        seconds = pd.to_numeric(group["seconds_in_roi"], errors="coerce").dropna()
        rows.append(
            {
                "location": location,
                "residence_tracks": int(len(seconds)),
                "mean_roi_residence_s": float(seconds.mean()) if len(seconds) else 0.0,
                "median_roi_residence_s": float(seconds.median()) if len(seconds) else 0.0,
                "p90_roi_residence_s": float(seconds.quantile(0.90)) if len(seconds) else 0.0,
                "p95_roi_residence_s": float(seconds.quantile(0.95)) if len(seconds) else 0.0,
                "long_residence_ge_5s_ratio": float((seconds >= 5.0).mean()) if len(seconds) else 0.0,
                "long_residence_ge_10s_ratio": float((seconds >= 10.0).mean()) if len(seconds) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _prepare_metrics(args: argparse.Namespace) -> pd.DataFrame:
    root = args.risk_mining_dir
    geometry = _base_geometry_df()

    density = _require_csv(root / "output_intersection_spatiotemporal_density" / "density_summary.csv")
    residence = _require_csv(root / "output_intersection_spatiotemporal_density" / "track_roi_residence.csv")
    degree = _require_csv(root / "output_interaction_topology_complexity" / "interaction_degree_summary_by_location.csv")
    duration = _require_csv(root / "output_interaction_topology_complexity" / "interaction_duration_summary.csv")
    conflict = _require_csv(root / "output_conflict_patterns_spatial_clustering" / "conflict_type_summary_by_location.csv")
    row_overlap = _require_csv(root / "output_right_of_way_overlap" / "right_of_way_overlap_summary_by_location.csv")

    residence_summary = _residence_metrics(residence)
    density = density.copy()
    density["occupancy_seconds_per_vehicle"] = density["seconds_in_roi_total"] / density["vehicle_tracks"].replace(0, np.nan)
    density["occupied_cells_per_vehicle"] = density["occupied_cells"] / density["vehicle_tracks"].replace(0, np.nan)

    duration = duration[duration.get("summary_level", "location") == "location"].copy()

    keep_density = [
        "location",
        "vehicle_tracks",
        "seconds_in_roi_total",
        "max_cell_seconds",
        "mean_occupied_cell_seconds",
        "occupied_cells",
        "roi_area_m2",
        "occupancy_seconds_per_vehicle",
        "occupied_cells_per_vehicle",
    ]
    keep_degree = [
        "location",
        "component_frames",
        "2-party_ratio",
        "3-party_ratio",
        "4+-party_ratio",
        "n_ge_3_ratio",
        "mean_degree",
        "max_degree",
    ]
    keep_duration = [
        "location",
        "episodes",
        "duration_median_s",
        "duration_p90_s",
        "duration_mean_s",
        "long_duration_ge_5s_ratio",
    ]
    keep_conflict = [
        "location",
        "total_pair_episodes",
        "crossing_ratio",
        "merging_ratio",
        "weaving_parallel_competition_ratio",
        "turning_interaction_ratio",
    ]
    keep_overlap = [
        "location",
        "legal_vehicle_green_passages",
        "legal_vru_green_crossings",
        "overlap_events",
        "simultaneous_overlap_events",
        "unique_spatial_overlap_cells_1m",
        "median_abs_arrival_time_gap_s",
        "vru_signal_unobservable",
    ]

    df = geometry
    for part, columns in (
        (density, keep_density),
        (residence_summary, list(residence_summary.columns)),
        (degree, keep_degree),
        (duration, keep_duration),
        (conflict, keep_conflict),
        (row_overlap, keep_overlap),
    ):
        df = df.merge(part[columns], on="location", how="left")

    df["overlap_events_per_legal_vehicle"] = df["overlap_events"] / df["legal_vehicle_green_passages"].replace(0, np.nan)
    df["unique_overlap_cells_per_1000m2"] = df["unique_spatial_overlap_cells_1m"] / df["conflict_area_m2"] * 1000.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _corr_pair(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xy = pd.concat([x, y], axis=1).dropna()
    n = int(len(xy))
    if n < 3 or xy.iloc[:, 0].nunique() < 2 or xy.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan"), n
    pearson = float(xy.iloc[:, 0].corr(xy.iloc[:, 1], method="pearson"))
    spearman = float(xy.iloc[:, 0].corr(xy.iloc[:, 1], method="spearman"))
    return pearson, spearman, n


def compute_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    x_cols = [
        ("conflict_area_m2", "Conflict area"),
        ("angle_skew_deg", "Angle skew"),
    ]
    y_cols = [
        ("mean_roi_residence_s", "Mean ROI residence"),
        ("p90_roi_residence_s", "P90 ROI residence"),
        ("long_residence_ge_5s_ratio", "Long residence >=5s"),
        ("occupancy_seconds_per_vehicle", "Occupancy seconds / vehicle"),
        ("max_cell_seconds", "Max cell seconds"),
        ("n_ge_3_ratio", "N>=3 interaction ratio"),
        ("mean_degree", "Mean interaction degree"),
        ("duration_p90_s", "P90 interaction duration"),
        ("long_duration_ge_5s_ratio", "Long game >=5s ratio"),
        ("total_pair_episodes", "Conflict pair episodes"),
        ("crossing_ratio", "Crossing conflict ratio"),
        ("unique_spatial_overlap_cells_1m", "Legal overlap cells"),
        ("overlap_events", "Legal overlap events"),
    ]
    rows: List[Dict[str, Any]] = []
    for x_col, x_label in x_cols:
        for y_col, y_label in y_cols:
            pearson, spearman, n = _corr_pair(metrics[x_col], metrics[y_col])
            rows.append(
                {
                    "x_metric": x_col,
                    "x_label": x_label,
                    "y_metric": y_col,
                    "y_label": y_label,
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3 or len(np.unique(x[mask])) < 2:
        return None
    coef = np.polyfit(x[mask], y[mask], deg=1)
    xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), 100)
    ys = coef[0] * xs + coef[1]
    return xs, ys


def _annotated_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    marker: str = "o",
    label: Optional[str] = None,
    hollow_mask: Optional[pd.Series] = None,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    hollow = np.zeros(len(df), dtype=bool) if hollow_mask is None else hollow_mask.fillna(False).to_numpy(dtype=bool)
    solid = ~hollow
    if np.any(solid):
        ax.scatter(x[solid], y[solid], s=70, color=color, marker=marker, edgecolors="#202020", linewidths=0.55, label=label)
    if np.any(hollow):
        ax.scatter(x[hollow], y[hollow], s=82, facecolors="none", marker=marker, edgecolors=color, linewidths=1.8, label=(label + " (limited)") if label else None)
    for row in df.itertuples():
        value_x = getattr(row, x_col)
        value_y = getattr(row, y_col)
        if pd.notna(value_x) and pd.notna(value_y):
            ax.annotate(str(row.location), (float(value_x), float(value_y)), xytext=(5, 4), textcoords="offset points", fontsize=8)
    fit = _fit_line(x, y)
    if fit is not None:
        ax.plot(fit[0], fit[1], color=color, linewidth=1.4, alpha=0.65, linestyle="--")


def _corr_text(corr_df: pd.DataFrame, x_metric: str, y_metric: str) -> str:
    row = corr_df[(corr_df["x_metric"] == x_metric) & (corr_df["y_metric"] == y_metric)]
    if row.empty:
        return ""
    item = row.iloc[0]
    pearson = item["pearson_r"]
    spearman = item["spearman_r"]
    if pd.isna(pearson) or pd.isna(spearman):
        return f"N={int(item['n'])}"
    return f"Pearson r={pearson:.2f}, Spearman r={spearman:.2f}, N={int(item['n'])}"


def plot_area_vs_residence(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    _annotated_scatter(axes[0], metrics, "conflict_area_m2", "mean_roi_residence_s", "#2f6f8f", label="Mean")
    axes[0].set_title("Mean Core Residence")
    axes[0].set_ylabel("Seconds")
    axes[0].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "mean_roi_residence_s"), transform=axes[0].transAxes, va="top", fontsize=8)
    _annotated_scatter(axes[1], metrics, "conflict_area_m2", "p90_roi_residence_s", "#b65d2e", marker="s", label="P90")
    axes[1].set_title("P90 Core Residence")
    axes[1].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "p90_roi_residence_s"), transform=axes[1].transAxes, va="top", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Un-channelized conflict area Ac (m2)")
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Conflict Area vs Vehicle Exposure Time", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_area_vs_long_residence(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    _annotated_scatter(ax, metrics, "conflict_area_m2", "long_residence_ge_5s_ratio", "#8a4f9e", label=">=5s")
    _annotated_scatter(ax, metrics, "conflict_area_m2", "long_residence_ge_10s_ratio", "#d18f00", marker="^", label=">=10s")
    ax.set_xlabel("Un-channelized conflict area Ac (m2)")
    ax.set_ylabel("Long-residence track ratio")
    ax.set_title("Conflict Area vs Long Exposure Ratio")
    ax.text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "long_residence_ge_5s_ratio"), transform=ax.transAxes, va="top", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_area_vs_occupancy(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1), dpi=180)
    sizes = 35 + 0.010 * metrics["vehicle_tracks"].fillna(0).to_numpy(dtype=float)
    sc = ax.scatter(
        metrics["conflict_area_m2"],
        metrics["occupancy_seconds_per_vehicle"],
        s=sizes,
        c=metrics["max_cell_seconds"],
        cmap="viridis",
        edgecolors="#202020",
        linewidths=0.55,
        alpha=0.9,
    )
    for row in metrics.itertuples():
        ax.annotate(str(row.location), (row.conflict_area_m2, row.occupancy_seconds_per_vehicle), xytext=(5, 4), textcoords="offset points", fontsize=8)
    fit = _fit_line(metrics["conflict_area_m2"].to_numpy(dtype=float), metrics["occupancy_seconds_per_vehicle"].to_numpy(dtype=float))
    if fit is not None:
        ax.plot(fit[0], fit[1], color="#2f6f8f", linestyle="--", alpha=0.65)
    ax.set_xlabel("Un-channelized conflict area Ac (m2)")
    ax.set_ylabel("Occupancy seconds / vehicle")
    ax.set_title("Conflict Area vs Normalized Occupancy")
    ax.text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "occupancy_seconds_per_vehicle"), transform=ax.transAxes, va="top", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Max 1m cell occupancy seconds")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_area_vs_complexity(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    _annotated_scatter(axes[0], metrics, "conflict_area_m2", "n_ge_3_ratio", "#2364aa", label="N>=3 ratio")
    axes[0].set_title("Multi-Agent Interaction Ratio")
    axes[0].set_ylabel("Ratio")
    axes[0].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "n_ge_3_ratio"), transform=axes[0].transAxes, va="top", fontsize=8)
    _annotated_scatter(axes[1], metrics, "conflict_area_m2", "mean_degree", "#a23e48", marker="s", label="Mean N")
    axes[1].set_title("Mean Interaction Degree")
    axes[1].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "mean_degree"), transform=axes[1].transAxes, va="top", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Un-channelized conflict area Ac (m2)")
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Conflict Area vs Multi-Agent Game Complexity", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_area_vs_duration(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    _annotated_scatter(axes[0], metrics, "conflict_area_m2", "duration_p90_s", "#326273", label="P90 duration")
    axes[0].set_title("P90 Game Duration")
    axes[0].set_ylabel("Seconds")
    axes[0].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "duration_p90_s"), transform=axes[0].transAxes, va="top", fontsize=8)
    _annotated_scatter(axes[1], metrics, "conflict_area_m2", "long_duration_ge_5s_ratio", "#d95d39", marker="s", label=">=5s ratio")
    axes[1].set_title("Long Game Ratio")
    axes[1].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "long_duration_ge_5s_ratio"), transform=axes[1].transAxes, va="top", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Un-channelized conflict area Ac (m2)")
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Conflict Area vs Interaction Duration", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_area_vs_right_of_way(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    limited = metrics["legal_vru_green_crossings"].fillna(0) <= 0
    _annotated_scatter(
        axes[0],
        metrics,
        "conflict_area_m2",
        "unique_spatial_overlap_cells_1m",
        "#2a9d8f",
        label="Unique cells",
        hollow_mask=limited,
    )
    axes[0].set_title("Legal Green-Phase Overlap Cells")
    axes[0].set_ylabel("Unique 1m cells")
    axes[0].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "unique_spatial_overlap_cells_1m"), transform=axes[0].transAxes, va="top", fontsize=8)
    _annotated_scatter(
        axes[1],
        metrics,
        "conflict_area_m2",
        "overlap_events",
        "#e76f51",
        marker="s",
        label="Pair events",
        hollow_mask=limited,
    )
    axes[1].set_title("Legal Green-Phase Pair Overlaps")
    axes[1].text(0.02, 0.96, _corr_text(corr_df, "conflict_area_m2", "overlap_events"), transform=axes[1].transAxes, va="top", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Un-channelized conflict area Ac (m2)")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Conflict Area vs Legal Path Overlap", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_angle_skew(metrics: pd.DataFrame, corr_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), dpi=180)
    specs = [
        ("duration_p90_s", "P90 interaction duration (s)", "#326273"),
        ("crossing_ratio", "Crossing conflict ratio", "#8a4f9e"),
        ("n_ge_3_ratio", "N>=3 interaction ratio", "#d18f00"),
    ]
    for ax, (y_col, title, color) in zip(axes, specs):
        _annotated_scatter(ax, metrics, "angle_skew_deg", y_col, color, label=title)
        ax.set_xlabel("|90 - angle| (deg)")
        ax.set_title(title)
        ax.text(0.02, 0.96, _corr_text(corr_df, "angle_skew_deg", y_col), transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Auxiliary Geometry: Angle Skew vs Exposure/Conflict Metrics", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: Path) -> None:
    area_corr = corr_df[corr_df["x_metric"] == "conflict_area_m2"].copy()
    area_corr = area_corr.dropna(subset=["spearman_r"])
    if area_corr.empty:
        return
    values = area_corr["spearman_r"].to_numpy(dtype=float)[None, :]
    labels = area_corr["y_label"].tolist()
    fig_w = max(9.0, 0.46 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 3.0), dpi=180)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels(["Ac Spearman r"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for idx, value in enumerate(values[0]):
        ax.text(idx, 0, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#111111")
    ax.set_title("Conflict Area vs Exposure Metrics Correlation (N=6)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
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
                cell = "" if pd.isna(value) else f"{float(value):.3f}"
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def write_html(output_path: Path, metrics: pd.DataFrame, correlations: pd.DataFrame) -> None:
    metric_table = _html_table(
        metrics,
        [
            ("location", "Location"),
            ("conflict_area_m2", "Ac m2"),
            ("intersection_angle_deg", "Angle deg"),
            ("mean_roi_residence_s", "Mean residence"),
            ("p90_roi_residence_s", "P90 residence"),
            ("long_residence_ge_5s_ratio", ">=5s ratio"),
            ("occupancy_seconds_per_vehicle", "Occ s/veh"),
            ("n_ge_3_ratio", "N>=3"),
            ("duration_p90_s", "P90 game s"),
            ("unique_spatial_overlap_cells_1m", "Legal overlap cells"),
        ],
    )
    top_corr = correlations[correlations["x_metric"] == "conflict_area_m2"].copy()
    top_corr["abs_spearman"] = top_corr["spearman_r"].abs()
    top_corr = top_corr.sort_values("abs_spearman", ascending=False)
    corr_table = _html_table(
        top_corr,
        [
            ("y_label", "Exposure metric"),
            ("pearson_r", "Pearson r"),
            ("spearman_r", "Spearman r"),
            ("n", "N"),
        ],
        max_rows=30,
    )
    figures = [
        ("area_vs_residence_time.png", "Ac vs residence time"),
        ("area_vs_long_residence_ratio.png", "Ac vs long-residence ratio"),
        ("area_vs_occupancy.png", "Ac vs normalized occupancy"),
        ("area_vs_interaction_complexity.png", "Ac vs multi-agent complexity"),
        ("area_vs_interaction_duration.png", "Ac vs interaction duration"),
        ("area_vs_right_of_way_overlap.png", "Ac vs legal path overlap"),
        ("angle_skew_vs_exposure.png", "Angle skew auxiliary plots"),
        ("geometric_correlation_heatmap.png", "Correlation heatmap"),
    ]
    figure_html = "\n".join(
        f"<section class='panel'><h2>{html.escape(title)}</h2><a href='{html.escape(filename)}'><img src='{html.escape(filename)}' alt='{html.escape(title)}'></a></section>"
        for filename, title in figures
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SinD Geometric Topology & Exposure</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; color:#19252b; background:#f5f1eb; }}
    header {{ padding:32px 5vw; background:#233843; color:white; }}
    main {{ padding:24px 5vw 48px; }}
    .panel {{ background:white; border:1px solid #d8d0c5; padding:18px; margin:18px 0; overflow:auto; }}
    img {{ max-width:100%; background:white; border:1px solid #d8d0c5; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e3ddd3; padding:8px 9px; text-align:left; white-space:nowrap; }}
    th {{ background:#ece3d7; }}
    .note {{ line-height:1.55; color:#5b6870; }}
  </style>
</head>
<body>
  <header>
    <h1>SinD 交叉口几何构型与暴露度</h1>
    <p>将无引导冲突区面积 Ac 与路口核心驻留、占用、多体博弈和合法路权重叠指标关联。</p>
  </header>
  <main>
    <section class="panel">
      <p class="note">本报告只有 6 个路口样本，相关系数用于讨论趋势，不用于强显著性宣称。cc/tj/xasl 的 VRU 绿灯过街绑定当前不可观测，路权重叠图中以空心点提示。</p>
    </section>
    <section class="panel"><h2>Merged Metrics</h2>{metric_table}</section>
    <section class="panel"><h2>Ac Correlations</h2>{corr_table}</section>
    {figure_html}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-mining-dir", type=Path, default=Path("risk_mining"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_geometric_topology_exposure"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _prepare_metrics(args)
    correlations = compute_correlations(metrics)

    metrics.to_csv(args.output_dir / "geometric_exposure_metrics.csv", index=False)
    correlations.to_csv(args.output_dir / "geometric_exposure_correlations.csv", index=False)
    plot_area_vs_residence(metrics, correlations, args.output_dir / "area_vs_residence_time.png")
    plot_area_vs_long_residence(metrics, correlations, args.output_dir / "area_vs_long_residence_ratio.png")
    plot_area_vs_occupancy(metrics, correlations, args.output_dir / "area_vs_occupancy.png")
    plot_area_vs_complexity(metrics, correlations, args.output_dir / "area_vs_interaction_complexity.png")
    plot_area_vs_duration(metrics, correlations, args.output_dir / "area_vs_interaction_duration.png")
    plot_area_vs_right_of_way(metrics, correlations, args.output_dir / "area_vs_right_of_way_overlap.png")
    plot_angle_skew(metrics, correlations, args.output_dir / "angle_skew_vs_exposure.png")
    plot_correlation_heatmap(correlations, args.output_dir / "geometric_correlation_heatmap.png")
    notes = {
        "sample_size": 6,
        "geometry_source": "User-provided Ac and intersection angle table.",
        "interpretation": "Correlations are descriptive trend evidence only; N=6 is not enough for strong statistical significance claims.",
        "right_of_way_observability": "cc/tj/xasl currently have no observable legal VRU green crossing passages in the traffic-light binding output.",
    }
    (args.output_dir / "methodology_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(args.output_dir / "index.html", metrics, correlations)
    print(f"Done. Open {args.output_dir / 'index.html'}")
    print(metrics[["location", "conflict_area_m2", "mean_roi_residence_s", "p90_roi_residence_s", "n_ge_3_ratio", "duration_p90_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
